"""SQLite persistence for synchronous, idempotent AML Add/Search."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .errors import Conflict


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def payload_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class ReadOnlySQLiteStoreMixin:
    """Open an existing store without migrations, WAL changes, or filesystem writes."""

    def __init__(self, path: Path, processing_timeout_seconds: int = 1800):
        self.path = Path(path).resolve(strict=True)
        self.processing_timeout_seconds = processing_timeout_seconds

    @contextmanager
    def _connection(self):
        uri = "file:" + quote(self.path.as_posix(), safe="/:") + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        try:
            yield connection
        finally:
            connection.close()


class SQLiteStore:
    def __init__(self, path: Path, processing_timeout_seconds: int = 1800):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.processing_timeout_seconds = processing_timeout_seconds
        self._initialise()

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
        finally:
            connection.close()

    def _initialise(self) -> None:
        with self._connection() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS add_requests (
                    request_id TEXT PRIMARY KEY,
                    payload_hash TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('processing', 'complete')),
                    response_json TEXT,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memories (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL UNIQUE,
                    request_id TEXT NOT NULL UNIQUE,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    source_timestamp_ms INTEGER,
                    content_json TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(request_id) REFERENCES add_requests(request_id)
                );
                CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
                CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(user_id, session_id, seq);
                CREATE TABLE IF NOT EXISTS views (
                    view_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    FOREIGN KEY(memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_views_memory ON views(memory_id);
                """
            )
            db.commit()

    def claim_add(self, request_id: str, payload_hash: str) -> dict | None:
        """Claim a logical write; return its cached response when already complete."""
        now = time.time()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload_hash, state, response_json, updated_at FROM add_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
            if row is None:
                db.execute(
                    "INSERT INTO add_requests VALUES (?, ?, 'processing', NULL, ?)",
                    (request_id, payload_hash, now),
                )
                db.commit()
                return None
            if row["payload_hash"] != payload_hash:
                db.rollback()
                raise Conflict("request_id was already used with a different payload")
            if row["state"] == "complete":
                response = json.loads(row["response_json"])
                db.commit()
                return response
            if now - float(row["updated_at"]) <= self.processing_timeout_seconds:
                db.rollback()
                raise Conflict("this Add request is still processing")
            db.execute(
                "UPDATE add_requests SET updated_at=? WHERE request_id=?",
                (now, request_id),
            )
            db.commit()
            return None

    def abandon_add(self, request_id: str, payload_hash: str) -> None:
        with self._connection() as db:
            db.execute(
                "DELETE FROM add_requests WHERE request_id=? AND payload_hash=? AND state='processing'",
                (request_id, payload_hash),
            )
            db.commit()

    def complete_add(self, memory: dict, views: list[dict], response: dict) -> None:
        """Atomically persist the memory, all indexes, and successful idempotency result."""
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """INSERT INTO memories
                   (memory_id, request_id, user_id, session_id, source_timestamp_ms,
                    content_json, search_text, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    memory["memory_id"], memory["request_id"], memory["user_id"],
                    memory["session_id"], memory.get("source_timestamp_ms"),
                    canonical_json(memory["content"]), memory["search_text"], memory["created_at"],
                ),
            )
            for row in views:
                vector = np.asarray(row["embedding"], dtype=np.float32)
                db.execute(
                    "INSERT INTO views VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        row["view_id"], memory["memory_id"], row["kind"], row["text"],
                        sqlite3.Binary(vector.tobytes()), int(vector.shape[0]),
                    ),
                )
            changed = db.execute(
                """UPDATE add_requests SET state='complete', response_json=?, updated_at=?
                   WHERE request_id=? AND state='processing'""",
                (canonical_json(response), time.time(), memory["request_id"]),
            ).rowcount
            if changed != 1:
                db.rollback()
                raise Conflict("Add request lost its processing claim")
            db.commit()

    def views_for_user(self, user_id: str) -> list[dict]:
        with self._connection() as db:
            rows = db.execute(
                """SELECT v.view_id, v.memory_id, v.kind, v.text, v.embedding, v.dimension
                   FROM views v JOIN memories m ON m.memory_id=v.memory_id
                   WHERE m.user_id=? ORDER BY m.seq, v.view_id""",
                (user_id,),
            ).fetchall()
        return [
            {
                "view_id": row["view_id"], "memory_id": row["memory_id"],
                "kind": row["kind"], "text": row["text"],
                "embedding": np.frombuffer(row["embedding"], dtype=np.float32, count=row["dimension"]).copy(),
            }
            for row in rows
        ]

    def memories_by_ids(self, user_id: str, memory_ids: list[str]) -> dict[str, dict]:
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        with self._connection() as db:
            rows = db.execute(
                f"""SELECT seq, memory_id, session_id, source_timestamp_ms, content_json,
                           search_text, created_at
                    FROM memories WHERE user_id=? AND memory_id IN ({placeholders})""",
                (user_id, *memory_ids),
            ).fetchall()
        return {row["memory_id"]: self._memory_row(row) for row in rows}

    def memories_for_user(self, user_id: str, limit: int | None = None) -> list[dict]:
        """Return a stable, user-isolated read-only memory scan for bounded tools.

        This method is deliberately generic and does not encode Evidence ranking semantics.
        """
        sql = """SELECT seq, memory_id, session_id, source_timestamp_ms, content_json,
                        search_text, created_at FROM memories
                 WHERE user_id=? ORDER BY seq, memory_id"""
        params: tuple[object, ...] = (user_id,)
        if limit is not None:
            if limit < 1:
                return []
            sql += " LIMIT ?"
            params = (user_id, int(limit))
        with self._connection() as db:
            rows = db.execute(sql, params).fetchall()
        return [self._memory_row(row) for row in rows]

    def session_neighbors(self, user_id: str, memory_id: str, window: int) -> list[dict]:
        if window <= 0:
            return []
        with self._connection() as db:
            base = db.execute(
                "SELECT seq, session_id FROM memories WHERE user_id=? AND memory_id=?",
                (user_id, memory_id),
            ).fetchone()
            if base is None:
                return []
            before = db.execute(
                """SELECT seq, memory_id, session_id, source_timestamp_ms, content_json, search_text, created_at
                   FROM memories WHERE user_id=? AND session_id=? AND seq<?
                   ORDER BY seq DESC LIMIT ?""",
                (user_id, base["session_id"], base["seq"], window),
            ).fetchall()
            after = db.execute(
                """SELECT seq, memory_id, session_id, source_timestamp_ms, content_json, search_text, created_at
                   FROM memories WHERE user_id=? AND session_id=? AND seq>?
                   ORDER BY seq ASC LIMIT ?""",
                (user_id, base["session_id"], base["seq"], window),
            ).fetchall()
        return [self._memory_row(row) for row in reversed(before)] + [self._memory_row(row) for row in after]

    def adjacent_memories(
        self, user_id: str, memory_id: str, *, direction: str, window: int
    ) -> list[dict]:
        """Return strictly directional neighbors without changing legacy neighbor semantics."""
        if window <= 0 or direction not in {"before", "after", "both"}:
            return []
        with self._connection() as db:
            base = db.execute(
                "SELECT seq, session_id FROM memories WHERE user_id=? AND memory_id=?",
                (user_id, memory_id),
            ).fetchone()
            if base is None:
                return []
            before = []
            after = []
            if direction in {"before", "both"}:
                before = db.execute(
                    """SELECT seq, memory_id, session_id, source_timestamp_ms, content_json,
                              search_text, created_at FROM memories
                       WHERE user_id=? AND session_id=? AND seq<?
                       ORDER BY seq DESC LIMIT ?""",
                    (user_id, base["session_id"], base["seq"], window),
                ).fetchall()
            if direction in {"after", "both"}:
                after = db.execute(
                    """SELECT seq, memory_id, session_id, source_timestamp_ms, content_json,
                              search_text, created_at FROM memories
                       WHERE user_id=? AND session_id=? AND seq>?
                       ORDER BY seq ASC LIMIT ?""",
                    (user_id, base["session_id"], base["seq"], window),
                ).fetchall()
        return [self._memory_row(row) for row in reversed(before)] + [self._memory_row(row) for row in after]

    def same_session_memories(self, user_id: str, memory_id: str, limit: int) -> list[dict]:
        """Return a bounded same-session pool ordered by sequence distance then sequence."""
        if limit <= 0:
            return []
        with self._connection() as db:
            base = db.execute(
                "SELECT seq, session_id FROM memories WHERE user_id=? AND memory_id=?",
                (user_id, memory_id),
            ).fetchone()
            if base is None:
                return []
            rows = db.execute(
                """SELECT seq, memory_id, session_id, source_timestamp_ms, content_json,
                          search_text, created_at FROM memories
                   WHERE user_id=? AND session_id=? AND memory_id<>?
                   ORDER BY ABS(seq-?), seq, memory_id LIMIT ?""",
                (user_id, base["session_id"], memory_id, base["seq"], limit),
            ).fetchall()
        return [self._memory_row(row) for row in rows]

    def temporal_memories(
        self,
        user_id: str,
        memory_id: str,
        *,
        direction: str,
        time_window_ms: int,
        limit: int,
        same_session_only: bool,
    ) -> list[dict]:
        """Return real-timestamp neighbors; missing timestamps never fall back to insertion order."""
        if limit <= 0 or time_window_ms < 0 or direction not in {"before", "after", "around"}:
            return []
        with self._connection() as db:
            base = db.execute(
                """SELECT source_timestamp_ms, session_id FROM memories
                   WHERE user_id=? AND memory_id=?""",
                (user_id, memory_id),
            ).fetchone()
            if base is None or base["source_timestamp_ms"] is None:
                return []
            timestamp = int(base["source_timestamp_ms"])
            clauses = ["user_id=?", "memory_id<>?", "source_timestamp_ms IS NOT NULL"]
            params: list[object] = [user_id, memory_id]
            if same_session_only:
                clauses.append("session_id=?")
                params.append(base["session_id"])
            if direction == "before":
                clauses.extend(["source_timestamp_ms<?", "source_timestamp_ms>=?"])
                params.extend([timestamp, timestamp - time_window_ms])
            elif direction == "after":
                clauses.extend(["source_timestamp_ms>?", "source_timestamp_ms<=?"])
                params.extend([timestamp, timestamp + time_window_ms])
            else:
                clauses.append("source_timestamp_ms BETWEEN ? AND ?")
                params.extend([timestamp - time_window_ms, timestamp + time_window_ms])
            params.extend([timestamp, limit])
            rows = db.execute(
                f"""SELECT seq, memory_id, session_id, source_timestamp_ms, content_json,
                           search_text, created_at FROM memories
                    WHERE {' AND '.join(clauses)}
                    ORDER BY ABS(source_timestamp_ms-?), source_timestamp_ms, memory_id LIMIT ?""",
                tuple(params),
            ).fetchall()
        return [self._memory_row(row) for row in rows]

    @staticmethod
    def _memory_row(row: sqlite3.Row) -> dict:
        return {
            "seq": row["seq"], "memory_id": row["memory_id"], "session_id": row["session_id"],
            "source_timestamp_ms": row["source_timestamp_ms"],
            "content": json.loads(row["content_json"]), "search_text": row["search_text"],
            "created_at": row["created_at"],
        }


class ReadOnlySQLiteStore(ReadOnlySQLiteStoreMixin, SQLiteStore):
    """The Search interface backed by strictly read-only connections."""

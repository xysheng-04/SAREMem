"""Read transaction with metadata/index eager, original candidate content lazy."""
import copy, hashlib, json, sqlite3
from pathlib import Path
from urllib.parse import quote
from ..postseed.source import SQLiteReadView as OriginalReadView
from ..postseed.seed_port import digest
from ..postseed.contracts import Failure

class SQLiteReadView(OriginalReadView):
    def fingerprint(self):
        values=[]
        for suffix in ('','-wal','-journal'):
            p=Path(str(self.path)+suffix)
            s=p.stat() if p.exists() else None
            # Empty journals contain no committed state; SQLite may touch their
            # ctime merely while acquiring a read lock on a WAL-mode database.
            if suffix and s is not None and s.st_size==0:s=None
            values.append((suffix,s.st_size,s.st_mtime_ns,s.st_ctime_ns) if s else (suffix,None))
        return values

    def __enter__(self):
        self.before=self.fingerprint();self.cache={};self.content_reads=0;self.cache_hits=0
        # A frozen standalone file with no pending journal is safe for SQLite's
        # immutable reader, avoiding write attempts to shared-memory sidecars.
        frozen=not (self.path.stat().st_mode & 0o222) and all(v[1] is None for v in self.before[1:])
        suffix='?mode=ro&immutable=1' if frozen else '?mode=ro'
        self.db=sqlite3.connect('file:'+quote(str(self.path),safe='/:')+suffix,uri=True)
        self.db.row_factory=sqlite3.Row;self.db.execute('PRAGMA query_only=ON');self.db.execute('BEGIN')
        self.progress_failure=None;self.read_bytes=0
        def progress():
            try:self.budget.store(vm_steps=100);return 0
            except Failure as exc:self.progress_failure=exc;return 1
        self.db.set_progress_handler(progress,100)
        try:
            self.memories={r['memory_id']:dict(r) for r in self._fetch(
                'SELECT seq,memory_id,session_id,source_timestamp_ms,created_at,length(CAST(content_json AS BLOB)) AS content_bytes FROM memories WHERE user_id=? ORDER BY seq,memory_id',(self.user_id,))}
            self._views=self._fetch('SELECT v.view_id,v.memory_id,v.kind,v.text,v.embedding,v.dimension FROM views v JOIN memories m ON m.memory_id=v.memory_id WHERE m.user_id=? ORDER BY m.seq,v.view_id',(self.user_id,))
            self.read_view_digest=digest({'version':'simple.source.v1','scope':self.scope_digest,
                'file_generation':self.before,'memories':list(self.memories.values()),
                'views':[{**r,'embedding':hashlib.sha256(r['embedding']).hexdigest()} for r in self._views]})
            if self.fingerprint()!=self.before:raise Failure('source_drift','source')
            return self
        except BaseException:self.db.close();raise

    def contents(self,ids):
        if len(set(ids))!=len(ids) or any(mid not in self.memories for mid in ids):raise Failure('source_ids','source')
        result={}
        for mid in ids:
            if mid not in self.cache:
                if self.read_bytes+self.memories[mid]['content_bytes']>self.budget.limits.source_bytes:
                    raise Failure('budget_limited','source_bytes')
                rows=self._fetch('SELECT content_json FROM memories WHERE user_id=? AND memory_id=?',(self.user_id,mid))
                if len(rows)!=1:raise Failure('source_ids','source')
                self.cache[mid]=json.loads(rows[0]['content_json']);self.content_reads+=1
            else:self.cache_hits+=1
            result[mid]=copy.deepcopy(self.cache[mid])
        return result

    def __exit__(self,kind,value,trace):
        changed=self.fingerprint()!=self.before
        super().__exit__(kind,value,trace)
        if kind is None and changed:raise Failure('source_drift','source')

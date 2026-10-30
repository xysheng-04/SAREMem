"""The final HTTP projection, checked against scoped original rows, in every path."""
from contextlib import contextmanager
import json
import math
import sqlite3
from urllib.parse import quote
from pydantic import ValidationError

from ..media import MAX_SEARCH_IMAGE_BYTES, response_image_bytes
from ..schemas import SearchItem, SearchResponse
from ..errors import BadRequest
from ..postseed.seed_port import content_digest, digest
from .policy import SearchFault


@contextmanager
def reader(path):
    # mode=ro never silently creates an empty source when persistence is missing.
    db = sqlite3.connect('file:'+quote(str(path), safe='/:')+'?mode=ro', uri=True, timeout=.2)
    db.row_factory = sqlite3.Row
    try:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        yield db
    finally:
        db.close()


def original(db, user_id, mid, byte_limit):
    row = db.execute('SELECT user_id, length(CAST(content_json AS BLOB)) AS n '
                     'FROM memories WHERE memory_id=?', (mid,)).fetchone()
    if row is None:
        return None
    if row['user_id'] != user_id:
        raise SearchFault('scope_denied', terminal=True)
    if row['n'] > byte_limit:
        return None
    row = db.execute('SELECT content_json,created_at FROM memories WHERE user_id=? AND memory_id=?',
                     (user_id, mid)).fetchone()
    try:
        return json.loads(row['content_json']), row['created_at']
    except (ValueError, TypeError):
        raise SearchFault('source_integrity', terminal=True) from None


def seed_checkpoint(request, snapshot, view):
    from ..postseed.seed_port import load_snapshot
    snapshot = load_snapshot(snapshot.model_dump_json())
    b = snapshot.binding
    if (b.request_digest, b.scope_digest, b.read_view_digest) != (
            digest(request.model_dump(mode='json')), digest(request.user_id), view.read_view_digest):
        raise SearchFault('scope_denied', terminal=True)
    # The producer materialized all candidate hashes. Check again before exporting a checkpoint.
    for hit in snapshot.hits:
        if content_digest(view.contents([hit.memory_id])[hit.memory_id]) != hit.content_digest:
            raise SearchFault('source_integrity', terminal=True)
    return {'request': digest(request.model_dump(mode='json')),
            'hits': [{'id': h.memory_id, 'hash': h.content_digest, 'score': h.score} for h in snapshot.hits]}


def finalize(settings, request, policy, value, path, stats):
    """Keep original order; never fix content or use an unscoped record as evidence."""
    if hasattr(value, 'model_dump'):
        value = value.model_dump(mode='json')
    if not isinstance(value, dict) or set(value) != {'data'} or not isinstance(value['data'], list):
        raise SearchFault('response_validation')
    rows = value['data']
    if len(rows) > 100:
        raise SearchFault('response_validation')
    stats['candidate_count'] = len(rows)
    items, seen = [], set()
    used, previous = 0, math.inf
    with reader(settings.storage_path) as db:
        # Also verifies empty responses against a readable, correct source schema.
        db.execute('SELECT memory_id FROM memories WHERE user_id=? LIMIT 1', (request.user_id,)).fetchone()
        for row in rows:
            if not isinstance(row, dict) or set(row)-{'id','content','score','created_at'}:
                raise SearchFault('response_validation')
            mid = row.get('id')
            if not isinstance(mid, str) or not mid or mid in seen:
                stats['dropped'] += 1
                if path == 'simple':
                    raise SearchFault('candidate_id')
                continue
            seen.add(mid)
            raw = original(db, request.user_id, mid, policy.source_bytes)
            if raw is None:
                stats['dropped'] += 1
                if path == 'simple':
                    raise SearchFault('candidate_id')
                continue
            content, created_at = raw
            if row.get('content') != content or row.get('created_at') not in (None, created_at):
                stats['dropped'] += 1
                if path == 'simple':
                    raise SearchFault('response_validation')
                continue
            score = row.get('score')
            if type(score) not in (int, float) or not math.isfinite(score) or score > previous:
                raise SearchFault('ranking')
            previous = score
            try:
                item = SearchItem.model_validate(row)
                if not item.content or [p.model_dump(mode='json') for p in item.content] != content:
                    raise ValueError('content normalization')
                size = response_image_bytes(item.content)
            except (ValidationError, BadRequest, ValueError):
                stats['dropped'] += 1
                stats['media_filtered'] = True
                continue
            if used + size > MAX_SEARCH_IMAGE_BYTES:
                stats['dropped'] += 1
                stats['media_filtered'] = True
                continue
            if len(items) >= min(request.top_k, 20, settings.return_limit):
                continue
            used += size
            items.append(item)
    # Force JSON serialization here, inside the recoverable stage boundary.
    response = SearchResponse(data=items)
    serialized = response.model_dump_json()
    SearchResponse.model_validate_json(serialized)
    if len(serialized.encode()) > policy.max_wire_bytes - 4096:
        raise SearchFault('capacity')
    return json.loads(serialized)


def seed_response(settings, request, policy, checkpoint, stats):
    from ..media import plain_text
    from ..retrieval import lexical_rank
    with reader(settings.storage_path) as db:
        if checkpoint is None:
            raise SearchFault('seed_unavailable')
        if checkpoint.get('request') != digest(request.model_dump(mode='json')):
            raise SearchFault('scope_denied',terminal=True)
        hits=checkpoint.get('hits')
        if not isinstance(hits,list) or len(hits)>80:
            raise SearchFault('seed_integrity',terminal=True)
        ids=[h.get('id') for h in hits if isinstance(h,dict)]
        if len(ids)!=len(hits) or any(not isinstance(mid,str) or not mid for mid in ids) or len(ids)!=len(set(ids)):
            raise SearchFault('seed_integrity',terminal=True)
        stats['seed_existing']=True
        stats['candidate_count'] = len(hits)
        candidates = []
        read_bytes = 0
        for hit in hits:
            raw = original(db, request.user_id, hit['id'], policy.source_bytes)
            if raw is None:
                raise SearchFault('source_integrity',terminal=True)
            content, created = raw
            read_bytes += len(json.dumps(content,ensure_ascii=False).encode())
            if read_bytes > policy.source_bytes:
                raise SearchFault('capacity')
            if content_digest(content) != hit.get('hash'):
                raise SearchFault('source_integrity', terminal=True)
            score=hit.get('score')
            if type(score) not in (float,int) or not math.isfinite(score) or (candidates and score>candidates[-1]['score']):
                raise SearchFault('seed_integrity',terminal=True)
            candidates.append({'id': hit['id'], 'content': content,
                               'score': score, 'created_at': created})
        return {'data': candidates}

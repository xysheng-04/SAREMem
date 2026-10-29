from ..postseed.seed_port import digest
from ..postseed.contracts import Failure

def trusted_metadata(row):
    timestamp=row.get('source_timestamp_ms');seq=row.get('seq');session=row.get('session_id')
    if timestamp is not None and (type(timestamp)is not int or not 0<=timestamp<=253402300799999):
        raise Failure('source_timestamp','source')
    if seq is not None and (type(seq)is not int or seq<0):raise Failure('source_sequence','source')
    return {'source_timestamp_ms':timestamp,'source_timestamp_origin':'memory.source_timestamp_ms',
        'source_timestamp_semantics':'minimum_source_message_timestamp_ms_UTC_not_event_time',
        'created_at':row.get('created_at'),'created_at_semantics':'stored_created_at_not_assumed_event_time',
        'session_digest':digest(session) if session is not None else None,'sequence':seq,
        'has_order':session is not None and seq is not None}


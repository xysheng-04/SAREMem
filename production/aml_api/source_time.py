"""Public timestamps come exclusively from supplied source-message timestamps."""
from datetime import datetime, timedelta, timezone

def source_created_at(timestamp):
    # Do not coerce strings/bools/floats or infer time from insertion order/content.
    if type(timestamp) is not int or not 0 <= timestamp <= 253402300799999:
        return None
    value = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=timestamp)
    return value.isoformat().replace("+00:00", "Z")

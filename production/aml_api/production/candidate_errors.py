"""Only allowlisted per-candidate failures may be skipped; never fabricate a label."""
import sys
from .policy import SearchFault
from ..postseed.contracts import Failure
from ..errors import BadRequest

LOCAL_PROVIDER={'provider','provider_retry_exhausted','retry_limit'}
LOCAL_PROTOCOL={'assessment_schema','duplicate_fragment','unknown_fragment','missing_fragment','provider_json','provider_length','provider_failure'}
LOCAL_IMAGE={'image_decode','image_decode_limit','image_format','image_bytes','image_output_bytes'}
def candidate_error(exc):
    if isinstance(exc,BadRequest):
        return 'candidate_media'
    if isinstance(exc,SearchFault):
        return exc.code if not exc.terminal and exc.code in LOCAL_PROVIDER else None
    if isinstance(exc,Failure):
        return exc.code if exc.code in LOCAL_PROTOCOL else None
    module=sys.modules.get('aml_api.search_images')
    if module is not None and isinstance(exc,module.ImageInputError):
        return exc.code if exc.code in LOCAL_IMAGE else None
    return None

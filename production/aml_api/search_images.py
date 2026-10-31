"""Search-only, request-local deterministic visual derivatives. No disk or payload logs."""
from contextlib import contextmanager
from contextvars import ContextVar
from collections import OrderedDict
import base64
import hashlib
import io
import json
import math
import os
import time

ACTIVE = ContextVar('search_images', default=None)
VERSION = 'search-png-v1'

class ImageInputError(BaseException):
    def __init__(self, code):
        self.code = code
        super().__init__(code)

class Images:
    def __init__(self, deadline, mode='auto', max_edge=2048, max_pixels=2097152):
        if mode not in {'auto', 'off'} or not 64 <= max_edge <= 2048 or not 4096 <= max_pixels <= 4194304:
            raise ImageInputError('image_policy')
        self.deadline, self.mode = deadline, mode
        self.policy = dict(version=VERSION, mode=mode, max_edge=max_edge, max_pixels=max_pixels,
                           decode_pixels=60000000, decode_edge=32768, output='PNG', cache_bytes=33554432)
        self.binding = hashlib.sha256(json.dumps(self.policy, sort_keys=True).encode()).hexdigest()
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.records = []
        self.sends = []
        self.usage = []

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError()

    def image(self, uri):
        from .production.diagnostics import stage
        stage("image_processing")
        self.check()
        if self.mode == 'off':
            return uri
        key = hashlib.sha256(uri.encode()).hexdigest()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        start = time.monotonic()
        try:
            from PIL import Image, ImageOps
            header, encoded = uri.split(',', 1)
            if header.lower() not in {'data:image/jpeg;base64', 'data:image/png;base64', 'data:image/webp;base64'}:
                raise ImageInputError('image_format')
            if len(encoded) > 13981016:
                raise ImageInputError('image_bytes')
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) > 10*1024*1024:
                raise ImageInputError('image_bytes')
            with Image.open(io.BytesIO(raw)) as im:
                w,h = im.size
                fmt = im.format
                if fmt not in {'JPEG','PNG','WEBP'} or getattr(im, 'n_frames', 1) != 1:
                    raise ImageInputError('image_format')
                if w*h > self.policy['decode_pixels'] or max(w,h) > self.policy['decode_edge']:
                    raise ImageInputError('image_decode_limit')
                orientation = im.getexif().get(274, 1)
                scale = min(1., self.policy['max_edge']/max(w,h), math.sqrt(self.policy['max_pixels']/(w*h)))
                target = (max(1, math.floor(w*scale)), max(1, math.floor(h*scale)))
                changed = scale < 1 or orientation != 1 or im.mode not in {'RGB','RGBA','L','LA','P'}
                # JPEG draft reduces allocation before full decoding; thumbnail never crops.
                if scale < 1:
                    im.draft(im.mode, target)
                    im.thumbnail(target, Image.Resampling.LANCZOS, reducing_gap=3.)
                else:
                    im.load()  # Reject corrupt small images too.
                self.check()
                if changed:
                    out = ImageOps.exif_transpose(im)
                    alpha = out.mode in {'RGBA','LA'} or 'transparency' in out.info
                    out = out.convert('RGBA' if alpha else 'RGB')
                    out.info.clear()
                    buf = io.BytesIO()
                    out.save(buf, format='PNG', compress_level=3)
                    derived = buf.getvalue()
                    if len(derived) > 10*1024*1024:
                        raise ImageInputError('image_output_bytes')
                    result = 'data:image/png;base64,' + base64.b64encode(derived).decode('ascii')
                    dw,dh = out.size
                    outfmt = 'PNG'
                else:
                    result, derived, dw,dh,outfmt = uri,raw,w,h,fmt
            self.check()
        except (ImageInputError, TimeoutError):
            raise
        except MemoryError:
            raise ImageInputError('image_memory_limit') from None
        except Exception:
            raise ImageInputError('image_decode') from None
        record = dict(policy_binding=self.binding, version=VERSION, input_sha256=key,
                      output_sha256=hashlib.sha256(result.encode()).hexdigest(),
                      original_width=w, original_height=h, original_pixels=w*h, original_bytes=len(raw), original_format=fmt,
                      width=dw, height=dh, pixels=dw*dh, bytes=len(derived), format=outfmt,
                      resized=scale < 1, changed=changed, elapsed_ms=round((time.monotonic()-start)*1000,3))
        self.records.append(record)
        # Bounded derivative cache, keyed by original digest within this request only.
        while self.cache and self.cache_bytes + len(result) > self.policy['cache_bytes']:
            _, previous = self.cache.popitem(last=False)
            self.cache_bytes -= len(previous)
        if len(result) <= self.policy['cache_bytes']:
            self.cache[key] = result
            self.cache_bytes += len(result)
        return result

    def content(self, blocks):
        return [{**b, 'image_url': {**b['image_url'], 'url': self.image(b['image_url']['url'])}}
                if b.get('type') == 'image_url' else b for b in blocks]

@contextmanager
def scope(deadline):
    try:
        images = Images(deadline, os.environ.get('AML_SEARCH_IMAGE_MODE','auto'),
                        int(os.environ.get('AML_SEARCH_IMAGE_MAX_EDGE','2048')),
                        int(os.environ.get('AML_SEARCH_IMAGE_MAX_PIXELS','2097152')))
    except (ValueError, TypeError):
        raise ImageInputError('image_policy') from None
    token = ACTIVE.set(images)
    try:
        yield images
    finally:
        ACTIVE.reset(token)
        images.cache.clear()

def prepare(content):
    images = ACTIVE.get()
    return images.content(content) if images is not None else content


def usage(response):
    images = ACTIVE.get()
    if images is None:
        return
    u = getattr(response, 'usage', None)
    values = {k: getattr(u, k, None) for k in ('prompt_tokens','completion_tokens','total_tokens')}
    images.usage.append({k:v for k,v in values.items() if type(v) is int and v >= 0} or 'unavailable')

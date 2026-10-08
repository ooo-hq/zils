"""Bounded decoding of untrusted still images in a disposable subprocess."""

import hashlib
import io
import json
import struct
import subprocess
import sys
import warnings
from dataclasses import dataclass

MAX_ENCODED_BYTES = 10 * 1024 * 1024
MAX_CANONICAL_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_EDGE = 8192
MAX_TRAIN = 1024
MAX_CALIBRATION = 256
MAX_TEST = 512
MAX_JOB_BYTES = 1024 * 1024 * 1024
DECODE_SECONDS = 15
PREPROCESSOR = "zils-image-rgb-png/v1"


@dataclass(frozen=True)
class CanonicalImage:
    data: bytes
    source_sha256: str
    sha256: str
    pixel_sha256: str
    width: int
    height: int


class _BoundedOutput(io.BytesIO):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def write(self, data):
        if self.tell() + len(data) > self.limit:
            raise ValueError("canonical image exceeds the byte limit")
        return super().write(data)


def _canonicalize(source, output_limit=MAX_CANONICAL_BYTES):
    from PIL import Image, ImageOps

    if not isinstance(source, bytes) or not 0 < len(source) <= MAX_ENCODED_BYTES:
        raise ValueError("image exceeds the encoded byte limit")
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            with Image.open(io.BytesIO(source)) as original:
                width, height = original.size
                if (
                    original.format not in ("JPEG", "PNG")
                    or max(width, height) > MAX_EDGE
                    or width * height > MAX_PIXELS
                    or min(width, height) < 1
                    or getattr(original, "n_frames", 1) != 1
                ):
                    raise ValueError("unsupported image format or dimensions")
                original.verify()
            with Image.open(io.BytesIO(source)) as original:
                oriented = ImageOps.exif_transpose(original)
                rgb = oriented.convert("RGB")
                # A fresh image discards PNG/JPEG info, EXIF, profiles and comments.
                clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
                pixels = hashlib.sha256(
                    struct.pack("!II", *clean.size) + clean.tobytes()
                ).hexdigest()
                with _BoundedOutput(output_limit) as out:
                    clean.save(out, format="PNG")
                    data = out.getvalue()
                return CanonicalImage(
                    data,
                    hashlib.sha256(source).hexdigest(),
                    hashlib.sha256(data).hexdigest(),
                    pixels,
                    *clean.size,
                )
        except (OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise ValueError("invalid or oversized JPEG/PNG image") from None


def canonicalize(source):
    if not isinstance(source, bytes) or not 0 < len(source) <= MAX_ENCODED_BYTES:
        raise ValueError("image exceeds the encoded byte limit")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "zils.image_assets", "--worker"],
            input=source,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=DECODE_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise ValueError("image decoder exceeded its deadline") from None
    if result.returncode or len(result.stdout) > MAX_CANONICAL_BYTES + 1024:
        raise ValueError("invalid or oversized JPEG/PNG image")
    try:
        header, data = result.stdout.split(b"\n", 1)
        meta = json.loads(header)
        return CanonicalImage(data=data, **meta)
    except (ValueError, TypeError):
        raise ValueError("image decoder returned an invalid result") from None


def main():
    try:
        result = _canonicalize(sys.stdin.buffer.read(MAX_ENCODED_BYTES + 1))
        meta = {k: v for k, v in vars(result).items() if k != "data"}
        sys.stdout.buffer.write(json.dumps(meta).encode() + b"\n" + result.data)
    except (ValueError, MemoryError):
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

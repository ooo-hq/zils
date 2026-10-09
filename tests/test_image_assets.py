"""Untrusted image bytes are bounded, normalized and stripped before serving."""

import importlib
import importlib.util
import io
import struct
import unittest
import zlib

from PIL import Image, PngImagePlugin


def png(width=8, height=8):
    output = io.BytesIO()
    info = PngImagePlugin.PngInfo()
    info.add_text("private", "location")
    Image.new("RGB", (width, height), (10, 20, 30)).save(output, format="PNG", pnginfo=info)
    return output.getvalue()


def header(width, height):
    def chunk(kind, data):
        return (
            struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", b"")
        + chunk(b"IEND", b"")
    )


class CanonicalImageTest(unittest.TestCase):
    def test_slow_blob_download_cannot_outlive_finalization_lease(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock, patch

        from zils import cloud

        response = Mock(status_code=200, headers={})
        response.iter_content.return_value = [b"late"]
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(cloud.requests, "get", return_value=response),
            patch.object(cloud.time, "monotonic", side_effect=[0, 46]),
        ):
            with self.assertRaises(cloud.APIError) as failure:
                cloud._download_stream(
                    "https://storage.example/image", Path(temp) / "photo", 1024, max_seconds=45
                )
            self.assertEqual(failure.exception.status, 503)

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("zils.image_assets"), "canonicalizer missing")
        return importlib.import_module("zils.image_assets")

    def test_metadata_removed_without_changing_pixels(self):
        canonicalize = self.module().canonicalize
        first = canonicalize(png())
        second = canonicalize(first.data)
        self.assertEqual(first.pixel_sha256, second.pixel_sha256)
        self.assertEqual(first.sha256, second.sha256)
        self.assertNotEqual(first.source_sha256, first.sha256)
        self.assertEqual((first.width, first.height), (8, 8))
        with Image.open(io.BytesIO(first.data)) as image:
            self.assertEqual(image.info, {})
            self.assertEqual(image.getpixel((0, 0)), (10, 20, 30))
            self.assertEqual(image.mode, "RGB")

    def test_exif_orientation_applies_before_metadata_is_removed(self):
        source = io.BytesIO()
        image = Image.new("RGB", (3, 7), "red")
        exif = image.getexif()
        exif[274] = 6
        image.save(source, format="JPEG", exif=exif)
        result = self.module().canonicalize(source.getvalue())
        self.assertEqual((result.width, result.height), (7, 3))
        self.assertEqual(Image.open(io.BytesIO(result.data)).info, {})

    def test_malformed_animated_and_oversized_sources_are_rejected(self):
        module = self.module()
        jpeg = io.BytesIO()
        Image.new("RGB", (32, 32)).save(jpeg, format="JPEG")
        animation = io.BytesIO()
        Image.new("RGB", (4, 4), "red").save(
            animation, format="PNG", save_all=True, append_images=[Image.new("RGB", (4, 4), "blue")]
        )
        for source in (
            b"<svg/>",
            png()[:-20],
            jpeg.getvalue()[:-10],
            animation.getvalue(),
            b"x" * (10 * 1024 * 1024 + 1),
            header(4001, 4000),
            header(8193, 1),
        ):
            with self.subTest(size=len(source)), self.assertRaises(ValueError):
                module.canonicalize(source)

    def test_encoder_stops_before_exceeding_output_budget(self):
        module = self.module()
        # Small injected sink budget tests the real encoder's streaming boundary.
        with self.assertRaisesRegex(ValueError, "canonical"):
            module._canonicalize(png(), output_limit=10)

    def test_decoder_deadline_terminates_its_worker(self):
        module = self.module()
        import subprocess
        from unittest.mock import patch

        with patch.object(
            module.subprocess, "run", side_effect=subprocess.TimeoutExpired("decoder", 15)
        ):
            with self.assertRaisesRegex(ValueError, "deadline"):
                module.canonicalize(png())

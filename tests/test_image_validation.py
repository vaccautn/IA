from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vacca_vision import (  # noqa: E402
    ImageMetadata,
    ImageValidationConfig,
    ImageValidationError,
    Reason,
    ValidatedImage,
    validate_image,
)


PILLOW_AVAILABLE = importlib.util.find_spec("PIL") is not None


class ImageValidationStructureTests(unittest.TestCase):
    def test_rejects_public_validated_image_construction(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            "ValidatedImage instances must be created by validate_image",
        ):
            ValidatedImage(
                object(),
                path=Path("forged.jpg"),
                metadata=ImageMetadata(width=10, height=10),
                size_bytes=100,
                snapshot=b"forged",
            )

    def test_rejects_missing_path_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid_paths = (root / "missing.jpg", root)

            for path in invalid_paths:
                with self.subTest(path=path):
                    with self.assertRaises(ImageValidationError) as context:
                        validate_image(path)
                    self.assertEqual(context.exception.reason, Reason.INVALID_FILE)

    def test_rejects_unsupported_extension_case_insensitively(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.GIF"
            path.write_bytes(b"not-an-image")

            with self.assertRaisesRegex(
                ImageValidationError,
                "Image extension must be JPG, JPEG, or PNG",
            ):
                validate_image(path)

    def test_rejects_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.JPG"
            path.touch()

            with self.assertRaisesRegex(ImageValidationError, "Image file is empty"):
                validate_image(path)

    def test_rejects_file_above_configured_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.png"
            path.write_bytes(b"12345")

            with self.assertRaisesRegex(
                ImageValidationError,
                "Image file exceeds the maximum size of 4 bytes",
            ):
                validate_image(path, ImageValidationConfig(max_size_bytes=4))

    def test_rejects_invalid_validation_configuration(self) -> None:
        invalid_configs = (
            {"max_size_bytes": True},
            {"max_size_bytes": 0},
            {"min_width": False},
            {"min_width": 0},
            {"min_height": -1},
            {"max_width": True},
            {"max_width": 0},
            {"max_height": -1},
            {"max_pixels": 0},
        )

        for values in invalid_configs:
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    ImageValidationConfig(**values)


@unittest.skipUnless(PILLOW_AVAILABLE, "Pillow is not installed")
class PillowImageValidationTests(unittest.TestCase):
    @staticmethod
    def create_image(path: Path, size: tuple[int, int], image_format: str) -> None:
        from PIL import Image

        image = Image.new("RGB", size, color=(10, 20, 30))
        image.save(path, format=image_format)

    def test_validates_jpeg_and_png_without_modifying_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (("photo.JPEG", "JPEG"), ("photo.PnG", "PNG"))

            for filename, image_format in cases:
                with self.subTest(filename=filename):
                    path = root / filename
                    self.create_image(path, (12, 8), image_format)
                    before = hashlib.sha256(path.read_bytes()).digest()

                    validated = validate_image(path)

                    after = hashlib.sha256(path.read_bytes()).digest()
                    self.assertEqual(validated.path, path.resolve())
                    self.assertEqual(validated.metadata.width, 12)
                    self.assertEqual(validated.metadata.height, 8)
                    self.assertEqual(validated.size_bytes, path.stat().st_size)
                    self.assertEqual(before, after)

    def test_validated_snapshot_is_unchanged_when_original_is_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "photo.png"
            self.create_image(path, (12, 8), "PNG")
            validated = validate_image(path)
            original_snapshot = validated.snapshot_sha256

            with self.assertRaisesRegex(AttributeError, "ValidatedImage is immutable"):
                validated._snapshot = b"replacement"

            self.create_image(path, (12, 8), "PNG")
            path.write_bytes(path.read_bytes() + b"replacement-marker")

            self.assertEqual(validated.snapshot_sha256, original_snapshot)
            self.assertNotEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                original_snapshot,
            )

    def test_rejects_corrupt_supported_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.png"
            path.write_bytes(b"not-a-real-png")

            with self.assertRaisesRegex(
                ImageValidationError,
                "Image file cannot be decoded safely",
            ):
                validate_image(path)

    def test_rejects_dimensions_below_configured_minimum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.jpg"
            self.create_image(path, (9, 7), "JPEG")

            with self.assertRaisesRegex(
                ImageValidationError,
                "Image dimensions must be at least 10x8 pixels",
            ):
                validate_image(
                    path,
                    ImageValidationConfig(min_width=10, min_height=8),
                )

    def test_rejects_extension_and_decoded_format_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "renamed.jpg"
            self.create_image(path, (12, 8), "PNG")

            with self.assertRaisesRegex(
                ImageValidationError,
                "Decoded PNG image does not match .jpg extension",
            ):
                validate_image(path)

    def test_rejects_dimensions_and_pixel_count_above_configured_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.png"
            self.create_image(path, (20, 10), "PNG")
            cases = (
                (
                    ImageValidationConfig(max_width=19),
                    "Image dimensions exceed configured maximum 19x12000 pixels",
                ),
                (
                    ImageValidationConfig(max_height=9),
                    "Image dimensions exceed configured maximum 12000x9 pixels",
                ),
                (
                    ImageValidationConfig(max_pixels=199),
                    "Image pixel count exceeds configured maximum of 199 pixels",
                ),
            )

            for config, message in cases:
                with self.subTest(config=config):
                    with self.assertRaisesRegex(ImageValidationError, message):
                        validate_image(path, config)


if __name__ == "__main__":
    unittest.main()

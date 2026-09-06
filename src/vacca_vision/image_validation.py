from __future__ import annotations

import hashlib
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from collections.abc import Iterator

from .contracts import ImageMetadata, Reason


SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
SUPPORTED_FORMATS = frozenset({"JPEG", "PNG"})
FORMAT_BY_EXTENSION = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}


class ImageValidationError(ValueError):
    """A stable domain error for an invalid input image."""

    reason = Reason.INVALID_FILE


class ImageValidationDependencyError(RuntimeError):
    """Raised when the lightweight image validation dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class ImageValidationConfig:
    max_size_bytes: int = 10 * 1024 * 1024
    min_width: int = 1
    min_height: int = 1
    max_width: int = 12_000
    max_height: int = 12_000
    max_pixels: int = 50_000_000

    def __post_init__(self) -> None:
        for field_name in (
            "max_size_bytes",
            "min_width",
            "min_height",
            "max_width",
            "max_height",
            "max_pixels",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.min_width > self.max_width or self.min_height > self.max_height:
            raise ValueError("Minimum dimensions cannot exceed maximum dimensions")
        if self.min_width * self.min_height > self.max_pixels:
            raise ValueError("Minimum dimensions cannot exceed max_pixels")


class ValidatedImage:
    """An immutable image snapshot created only by validate_image."""

    __slots__ = ("_metadata", "_path", "_size_bytes", "_snapshot")

    def __init__(
        self,
        token: object,
        *,
        path: Path,
        metadata: ImageMetadata,
        size_bytes: int,
        snapshot: bytes,
    ) -> None:
        if token is not _VALIDATED_IMAGE_TOKEN:
            raise TypeError("ValidatedImage instances must be created by validate_image")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
            raise TypeError("size_bytes must be an integer")
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_metadata", metadata)
        object.__setattr__(self, "_size_bytes", size_bytes)
        object.__setattr__(self, "_snapshot", bytes(snapshot))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ValidatedImage is immutable")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def metadata(self) -> ImageMetadata:
        return self._metadata

    @property
    def size_bytes(self) -> int:
        return self._size_bytes

    @property
    def snapshot_sha256(self) -> str:
        return hashlib.sha256(self._snapshot).hexdigest()

    @contextmanager
    def _inference_image(self) -> Iterator[object]:
        from PIL import Image

        with Image.open(BytesIO(self._snapshot)) as decoded:
            decoded.load()
            detached = decoded.copy()
        try:
            yield detached
        finally:
            detached.close()


_VALIDATED_IMAGE_TOKEN = object()


def validate_image(
    source: str | Path,
    config: ImageValidationConfig | None = None,
) -> ValidatedImage:
    """Validate an image without writing to or transforming the source file."""

    settings = config or ImageValidationConfig()
    path = Path(source).expanduser()
    if not path.exists():
        raise ImageValidationError("Image path does not exist")
    if not path.is_file():
        raise ImageValidationError("Image path must reference a regular file")
    extension = path.suffix.casefold()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ImageValidationError("Image extension must be JPG, JPEG, or PNG")

    try:
        with path.open("rb") as image_file:
            if not stat.S_ISREG(os.fstat(image_file.fileno()).st_mode):
                raise ImageValidationError(
                    "Image path must reference a regular file"
                )
            snapshot = image_file.read(settings.max_size_bytes + 1)
    except ImageValidationError:
        raise
    except OSError:
        raise ImageValidationError("Image file cannot be read") from None
    size_bytes = len(snapshot)
    if size_bytes == 0:
        raise ImageValidationError("Image file is empty")
    if size_bytes > settings.max_size_bytes:
        raise ImageValidationError(
            f"Image file exceeds the maximum size of {settings.max_size_bytes} bytes"
        )

    metadata = _validate_image_bytes(
        snapshot,
        settings,
        expected_format=FORMAT_BY_EXTENSION[extension],
        expected_extension=extension,
    )

    return ValidatedImage(
        _VALIDATED_IMAGE_TOKEN,
        path=path.resolve(),
        metadata=metadata,
        size_bytes=size_bytes,
        snapshot=snapshot,
    )


def validate_image_bytes(
    source: bytes,
    config: ImageValidationConfig | None = None,
) -> ImageMetadata:
    """Validate an already bounded image upload and return its dimensions."""

    settings = config or ImageValidationConfig()
    if not isinstance(source, bytes):
        raise TypeError("source must be bytes")
    if len(source) == 0:
        raise ImageValidationError("Image file is empty")
    if len(source) > settings.max_size_bytes:
        raise ImageValidationError(
            f"Image file exceeds the maximum size of {settings.max_size_bytes} bytes"
        )
    return _validate_image_bytes(source, settings)


def _validate_image_bytes(
    snapshot: bytes,
    settings: ImageValidationConfig,
    *,
    expected_format: str | None = None,
    expected_extension: str | None = None,
) -> ImageMetadata:
    try:
        from PIL import Image
    except ImportError:
        raise ImageValidationDependencyError(
            "Pillow is required for image decoding; install the project dependencies"
        ) from None

    try:
        with Image.open(BytesIO(snapshot)) as image:
            width, height = image.size
            image_format = image.format
            if image_format not in SUPPORTED_FORMATS:
                raise ImageValidationError("Decoded image format must be JPEG or PNG")
            if expected_format is not None and image_format != expected_format:
                raise ImageValidationError(
                    f"Decoded {image_format} image does not match "
                    f"{expected_extension} extension"
                )
            _validate_dimensions(width, height, settings)
            image.verify()
    except ImageValidationError:
        raise
    except Exception:
        raise ImageValidationError("Image file cannot be decoded safely") from None
    return ImageMetadata(width=width, height=height)


def _validate_dimensions(
    width: int,
    height: int,
    settings: ImageValidationConfig,
) -> None:
    if width < settings.min_width or height < settings.min_height:
        raise ImageValidationError(
            "Image dimensions must be at least "
            f"{settings.min_width}x{settings.min_height} pixels"
        )
    if width > settings.max_width or height > settings.max_height:
        raise ImageValidationError(
            "Image dimensions exceed configured maximum "
            f"{settings.max_width}x{settings.max_height} pixels"
        )
    if width * height > settings.max_pixels:
        raise ImageValidationError(
            "Image pixel count exceeds configured maximum of "
            f"{settings.max_pixels} pixels"
        )

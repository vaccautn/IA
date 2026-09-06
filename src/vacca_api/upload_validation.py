"""Bounded and decoded image-upload validation shared by API endpoints."""

from __future__ import annotations

from fastapi import UploadFile

from vacca_vision import ImageValidationConfig, ImageValidationError, validate_image_bytes


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_UPLOAD_MIME_TYPES = frozenset({"image/jpeg", "image/jpg", "image/png"})


class UploadValidationError(ValueError):
    """A safe client-facing upload validation error."""

    def __init__(self, detail: str, event: str) -> None:
        super().__init__(detail)
        self.event = event


class UploadTooLargeError(UploadValidationError):
    """Raised when an upload exceeds the server-side byte limit."""


async def read_validated_upload(file: UploadFile) -> bytes:
    """Read at most one byte past the limit, then validate MIME and image data."""

    if file.content_type not in SUPPORTED_UPLOAD_MIME_TYPES:
        raise UploadValidationError(
            "File must be an image (JPEG or PNG)",
            "invalid_mime",
        )

    try:
        image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    except Exception:
        raise UploadValidationError(
            "Failed to read uploaded file",
            "upload_read_failed",
        ) from None

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise UploadTooLargeError(
            f"Image file exceeds the maximum size of {MAX_UPLOAD_BYTES} bytes",
            "upload_too_large",
        )
    if len(image_bytes) == 0:
        raise UploadValidationError("Empty file", "empty_upload")

    try:
        validate_image_bytes(
            image_bytes,
            ImageValidationConfig(max_size_bytes=MAX_UPLOAD_BYTES),
        )
    except ImageValidationError as exc:
        raise UploadValidationError(str(exc), "invalid_image") from None
    return image_bytes

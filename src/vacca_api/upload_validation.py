"""Shared validation for image uploads accepted by API endpoints."""

from __future__ import annotations

from fastapi import HTTPException, UploadFile


async def read_uploaded_image(file: UploadFile) -> bytes:
    """Read one image upload, preserving the API's stable 400 responses."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="File must be an image (JPEG, PNG, etc.)"
        )

    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read uploaded file") from None

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    return image_bytes

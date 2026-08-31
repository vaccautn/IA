from __future__ import annotations

import asyncio
import io
import math

import pytest
from fastapi import HTTPException, UploadFile
from pydantic import ValidationError
from starlette.datastructures import Headers

from vacca_api import main
from vacca_api.bcs import BCSScoreValidationError, round_bcs_score_for_backend
from vacca_api.schemas import BCSResponse


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (3.25, 3),
        (3.49, 3),
        (3.5, 3),
        (3.51, 4),
        (3.75, 4),
        (4.5, 4),
        (1, 1),
        (5, 5),
    ],
)
def test_backend_bcs_score_uses_decimal_half_down_rounding(
    score: float | int, expected: int
) -> None:
    assert round_bcs_score_for_backend(score) == expected


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_backend_bcs_score_rejects_nonfinite_values(score: float) -> None:
    with pytest.raises(BCSScoreValidationError, match="finite"):
        round_bcs_score_for_backend(score)


@pytest.mark.parametrize("score", [0, 5.01, 6])
def test_backend_bcs_score_rejects_out_of_range_values(score: float | int) -> None:
    with pytest.raises(BCSScoreValidationError, match="inclusive range 1..5"):
        round_bcs_score_for_backend(score)


@pytest.mark.parametrize("score", [None, "3.5", object(), True])
def test_backend_bcs_score_rejects_non_numeric_values(score: object) -> None:
    with pytest.raises(BCSScoreValidationError, match="numeric"):
        round_bcs_score_for_backend(score)


@pytest.mark.parametrize("score", [1, 2, 3, 4, 5])
def test_bcs_response_accepts_integer_scores(score: int) -> None:
    assert BCSResponse(bcs_score=score).bcs_score == score


def test_bcs_response_accepts_none_score() -> None:
    assert BCSResponse(bcs_score=None).bcs_score is None


@pytest.mark.parametrize("score", [True, False, "3", 3.0, 3.5, 0, 6])
def test_bcs_response_rejects_non_strict_or_out_of_range_scores(
    score: object,
) -> None:
    with pytest.raises(ValidationError):
        BCSResponse(bcs_score=score)


def test_bcs_placeholder_still_returns_null_score(monkeypatch) -> None:
    class FakeDetector:
        model_path = "fake-model.pt"
        gpu_available = False

        def detect(self, image_bytes: bytes):
            assert image_bytes == b"image"
            return [], 1, 1, 0.0

    monkeypatch.setattr(main, "get_detector", lambda **_: FakeDetector())

    response = asyncio.run(
        main.bcs(
            UploadFile(
                file=io.BytesIO(b"image"),
                filename="cow.jpg",
                headers=Headers({"content-type": "image/jpeg"}),
            )
        )
    )

    assert response.bcs_score is None


def test_openapi_declares_nullable_integer_bcs_score() -> None:
    property_schema = main.app.openapi()["components"]["schemas"]["BCSResponse"][
        "properties"
    ]["bcs_score"]

    if "anyOf" in property_schema:
        assert {item["type"] for item in property_schema["anyOf"]} == {
            "integer",
            "null",
        }
    else:
        assert property_schema["type"] == "integer"

    integer_schema = next(
        item for item in property_schema.get("anyOf", [property_schema])
        if item.get("type") == "integer"
    )
    assert integer_schema["minimum"] == 1
    assert integer_schema["maximum"] == 5


class _Upload:
    def __init__(self, content_type: str | None, payload: bytes = b"image", failure: Exception | None = None) -> None:
        self.content_type = content_type
        self.payload = payload
        self.failure = failure
        self.read_count = 0

    async def read(self) -> bytes:
        self.read_count += 1
        if self.failure is not None:
            raise self.failure
        return self.payload


@pytest.mark.parametrize("endpoint", [main.detect, main.bcs])
def test_upload_validation_rejects_non_image_without_reading(endpoint) -> None:
    upload = _Upload("text/plain", b"private payload", RuntimeError("private read"))
    with pytest.raises(HTTPException) as failure:
        asyncio.run(endpoint(upload))
    assert failure.value.status_code == 400
    assert failure.value.detail == "File must be an image (JPEG, PNG, etc.)"
    assert upload.read_count == 0
    assert "private" not in str(failure.value)


@pytest.mark.parametrize("endpoint", [main.detect, main.bcs])
def test_upload_validation_rejects_empty_bytes(endpoint) -> None:
    upload = _Upload("image/jpeg", b"")
    with pytest.raises(HTTPException) as failure:
        asyncio.run(endpoint(upload))
    assert (failure.value.status_code, failure.value.detail) == (400, "Empty file")
    assert upload.read_count == 1


@pytest.mark.parametrize("endpoint", [main.detect, main.bcs])
def test_upload_validation_maps_read_failures_to_sanitized_400(endpoint) -> None:
    upload = _Upload("image/jpeg", failure=RuntimeError("private upload failure"))
    with pytest.raises(HTTPException) as failure:
        asyncio.run(endpoint(upload))
    assert (failure.value.status_code, failure.value.detail) == (
        400,
        "Failed to read uploaded file",
    )
    assert upload.read_count == 1
    assert "private" not in str(failure.value)


def test_detect_uses_uploaded_bytes_once_and_preserves_success(monkeypatch) -> None:
    class FakeDetector:
        def detect(self, image_bytes: bytes):
            assert image_bytes == b"image"
            return [], 10, 20, 1.234

    upload = _Upload("image/jpeg")
    monkeypatch.setattr(main, "get_detector", lambda **_: FakeDetector())
    response = asyncio.run(main.detect(upload))

    assert response.detection_count == 0
    assert response.image_width == 10
    assert response.image_height == 20
    assert response.inference_time_ms == 1.23
    assert upload.read_count == 1


def test_bcs_valid_upload_keeps_placeholder_compatibility(monkeypatch) -> None:
    class FakeDetector:
        def detect(self, image_bytes: bytes):
            assert image_bytes == b"image"
            return [], 1, 1, 0.0

    upload = _Upload("image/png")
    monkeypatch.setattr(main, "get_detector", lambda **_: FakeDetector())
    response = asyncio.run(main.bcs(upload))

    assert response.status == "not_implemented"
    assert response.bcs_score is None
    assert response.cow_detected is False
    assert upload.read_count == 1

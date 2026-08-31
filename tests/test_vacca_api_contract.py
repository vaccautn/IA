from __future__ import annotations

import asyncio
import math
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from vacca_api import main
from vacca_api.bcs import BCSScoreValidationError, round_bcs_score_for_backend
from vacca_api.schemas import BCSReadinessResponse, BCSResponse
from vacca_bcs.serving import BCSInferenceExecutionError, BCSInferenceInputError


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


class _FakeService:
    def __init__(self, score: object = 3.5, failure: Exception | None = None) -> None:
        self.score = score
        self.failure = failure
        self.received: list[bytes] = []

    def infer(self, image_bytes: bytes):
        if self.failure is not None:
            raise self.failure
        self.received.append(image_bytes)
        return SimpleNamespace(continuous_score=self.score)


class _FakeRuntime:
    def __init__(self, service=None, status: str = "ready", failure: Exception | None = None) -> None:
        self.service = service
        self.status = status
        self.failure = failure
        self.get_calls = 0

    def get_service(self):
        self.get_calls += 1
        if self.failure is not None:
            raise self.failure
        return self.service


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
    service = _FakeService()
    runtime = _FakeRuntime(service)
    upload = _Upload("image/png")
    monkeypatch.setattr(main, "get_bcs_runtime", lambda: runtime)
    monkeypatch.setattr(main, "get_detector", lambda **_: pytest.fail("YOLO must not run"))
    response = asyncio.run(main.bcs(upload))

    assert response.status == "ok"
    assert response.message == "BCS score computed successfully."
    assert response.bcs_score == 3
    assert response.cow_detected is None
    assert service.received == [b"image"]
    assert upload.read_count == 1


@pytest.mark.parametrize("score, expected", [(1.0, 1), (3.5, 3), (5.0, 5)])
def test_bcs_rounds_only_at_http_boundary(monkeypatch, score: float, expected: int) -> None:
    service = _FakeService(score)
    monkeypatch.setattr(main, "get_bcs_runtime", lambda: _FakeRuntime(service))
    response = asyncio.run(main.bcs(_Upload("image/jpeg")))
    assert response.bcs_score == expected


@pytest.mark.parametrize(
    "failure, status, detail",
    [
        (RuntimeError("secret checkpoint"), 503, "BCS capability is unavailable"),
        (BCSInferenceInputError("secret image"), 400, "BCS image input is invalid"),
        (BCSInferenceExecutionError("secret model"), 500, "BCS inference failed"),
    ],
)
def test_bcs_failure_mapping_is_typed_and_sanitized(
    monkeypatch, failure: Exception, status: int, detail: str
) -> None:
    runtime = _FakeRuntime(failure=failure)
    if status != 503:
        runtime = _FakeRuntime(_FakeService(failure=failure))
    monkeypatch.setattr(main, "get_bcs_runtime", lambda: runtime)
    with pytest.raises(HTTPException) as captured:
        asyncio.run(main.bcs(_Upload("image/jpeg")))
    assert (captured.value.status_code, captured.value.detail) == (status, detail)
    assert "secret" not in str(captured.value)


def test_bcs_rounding_failure_is_sanitized_500(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_bcs_runtime", lambda: _FakeRuntime(_FakeService()))
    monkeypatch.setattr(
        main,
        "round_bcs_score_for_backend",
        lambda score: (_ for _ in ()).throw(RuntimeError("secret score")),
    )
    with pytest.raises(HTTPException) as captured:
        asyncio.run(main.bcs(_Upload("image/jpeg")))
    assert (captured.value.status_code, captured.value.detail) == (
        500,
        "BCS score could not be produced",
    )
    assert "secret" not in str(captured.value)


@pytest.mark.parametrize(
    ("status", "message"),
    [
        ("unconfigured", "BCS capability is not configured."),
        ("not_loaded", "BCS capability is configured but not loaded."),
        ("ready", "BCS capability is ready."),
        ("unavailable", "BCS capability is unavailable."),
    ],
)
def test_bcs_readiness_reports_exact_body_without_loading(
    monkeypatch, status: str, message: str
) -> None:
    runtime = _FakeRuntime(status=status)
    monkeypatch.setattr(main, "get_bcs_runtime", lambda: runtime)
    if status == "ready":
        response = main.bcs_readiness()
        body = response.model_dump()
    else:
        response = main.bcs_readiness()
        assert response.status_code == 503
        body = json.loads(response.body)
    assert body == {"status": status, "message": message}
    assert BCSReadinessResponse.model_validate(body).model_dump() == body
    assert runtime.get_calls == 0


def test_bcs_readiness_openapi_and_health_remain_bcs_independent(monkeypatch) -> None:
    runtime = _FakeRuntime(status="ready")
    monkeypatch.setattr(main, "get_bcs_runtime", lambda: pytest.fail("readiness must inspect only"))
    schema = main.app.openapi()
    readiness = schema["paths"]["/ready/bcs"]["get"]["responses"]
    assert "200" in readiness and "503" in readiness
    readiness_schema = readiness["200"]["content"]["application/json"]["schema"]
    assert readiness_schema["$ref"].endswith("/BCSReadinessResponse")
    assert schema["components"]["schemas"]["BCSReadinessResponse"]["properties"]["status"][
        "enum"
    ] == ["unconfigured", "not_loaded", "ready", "unavailable"]
    assert schema["paths"]["/bcs"]["post"]["responses"]["200"]["content"]

    class FakeDetector:
        model_path = "safe-model"
        gpu_available = False

    monkeypatch.setattr(main, "get_detector", lambda **_: FakeDetector())
    response = main.health()
    assert response.model_loaded is True
    assert runtime.get_calls == 0

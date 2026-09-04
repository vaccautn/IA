from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from vacca_api import main
from vacca_api.schemas import BCSReadinessResponse, BCSResponse
from vacca_bcs.serving import BCSInferenceExecutionError, BCSInferenceInputError


@pytest.mark.parametrize("category", [1, 2, 3, 4, 5])
def test_bcs_response_accepts_categories(category: int) -> None:
    assert BCSResponse(bcs_category=category).bcs_category == category


@pytest.mark.parametrize("category", [True, False, "3", 3.0, 3.5, 0, 6, None])
def test_bcs_response_rejects_non_strict_or_out_of_range_categories(
    category: object,
) -> None:
    with pytest.raises(ValidationError):
        BCSResponse(bcs_category=category)


def test_openapi_declares_required_integer_bcs_category() -> None:
    schema = main.app.openapi()["components"]["schemas"]["BCSResponse"]
    property_schema = schema["properties"]["bcs_category"]

    assert property_schema["type"] == "integer"
    integer_schema = property_schema
    assert integer_schema["minimum"] == 1
    assert integer_schema["maximum"] == 5
    assert "bcs_category" in schema["required"]
    with pytest.raises(ValidationError):
        BCSResponse()


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
    def __init__(self, category: object = 4, failure: Exception | None = None) -> None:
        self.category = category
        self.failure = failure
        self.received: list[bytes] = []

    def infer(self, image_bytes: bytes):
        if self.failure is not None:
            raise self.failure
        self.received.append(image_bytes)
        return SimpleNamespace(bcs_category=self.category)


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


def test_bcs_valid_upload_returns_category(monkeypatch) -> None:
    service = _FakeService()
    runtime = _FakeRuntime(service)
    upload = _Upload("image/png")
    monkeypatch.setattr(main, "get_bcs_runtime", lambda: runtime)
    monkeypatch.setattr(main, "get_detector", lambda **_: pytest.fail("YOLO must not run"))
    response = asyncio.run(main.bcs(upload))

    assert response.status == "ok"
    assert response.message == "BCS category 1..5 computed successfully."
    assert response.bcs_category == 4
    assert response.cow_detected is None
    assert service.received == [b"image"]
    assert upload.read_count == 1


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


def test_prototype_ui_uses_active_detection_model_label() -> None:
    html = (main.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "Modelo: combined-v2-finetune (Navid + BCS)" in html


def test_prototype_ui_delivery_and_bcs_contract() -> None:
    response = main.prototype_ui()
    assert response.status_code == 200
    html = response.body.decode("utf-8")
    for fragment in (
        'role="tablist"', 'id="detectTab"', 'id="bcsTab"', 'aria-selected=',
        'id="bcsReadiness"', 'GET /ready/bcs', 'POST /bcs', 'Calculate BCS',
        'bcs_category', 'Not reported', 'aria-live="polite"', 'combined-v2-finetune',
        'ready', 'not_loaded', 'unconfigured', 'unavailable',
    ):
        assert fragment in html
    assert "innerHTML" not in html
    assert "FileReader" not in html

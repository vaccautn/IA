from __future__ import annotations

import asyncio
import json
import sys
import threading
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from PIL import Image
from pydantic import ValidationError

from vacca_api import main
from vacca_api import detection
from vacca_api.bcs_runtime import BCSRuntime, BCSRuntimeStatus
from vacca_api.schemas import BCSReadinessResponse, BCSResponse
from vacca_bcs.serving import BCSInferenceExecutionError, BCSInferenceInputError
from starlette.concurrency import run_in_threadpool


def _request() -> object:
    return SimpleNamespace(app=main.app)


async def _call_endpoint(endpoint, upload: object):
    if endpoint is main.detect:
        return await endpoint(_request(), upload)
    return await endpoint(upload)


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

    async def read(self, size: int = -1) -> bytes:
        self.read_count += 1
        if self.failure is not None:
            raise self.failure
        return self.payload if size < 0 else self.payload[:size]


def _valid_image_bytes(image_format: str = "JPEG") -> bytes:
    stream = BytesIO()
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(stream, format=image_format)
    return stream.getvalue()


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
        asyncio.run(_call_endpoint(endpoint, upload))
    assert failure.value.status_code == 400
    assert failure.value.detail == "File must be an image (JPEG or PNG)"
    assert upload.read_count == 0
    assert "private" not in str(failure.value)


@pytest.mark.parametrize("endpoint", [main.detect, main.bcs])
def test_upload_validation_rejects_empty_bytes(endpoint) -> None:
    upload = _Upload("image/jpeg", b"")
    with pytest.raises(HTTPException) as failure:
        asyncio.run(_call_endpoint(endpoint, upload))
    assert (failure.value.status_code, failure.value.detail) == (400, "Empty file")
    assert upload.read_count == 1


@pytest.mark.parametrize("endpoint", [main.detect, main.bcs])
@pytest.mark.parametrize("image_format", ["JPEG", "PNG"])
def test_shared_upload_validation_maps_malformed_images_to_exact_detail(
    endpoint, image_format: str
) -> None:
    upload = _Upload(
        "image/jpeg" if image_format == "JPEG" else "image/png",
        b"malformed image bytes",
    )

    with pytest.raises(HTTPException) as failure:
        asyncio.run(_call_endpoint(endpoint, upload))

    assert (failure.value.status_code, failure.value.detail) == (
        400,
        "Image file cannot be decoded safely",
    )


@pytest.mark.parametrize("endpoint", [main.detect, main.bcs])
def test_shared_upload_validation_reports_decoded_unsupported_format(endpoint) -> None:
    upload = _Upload("image/jpeg", _valid_image_bytes("GIF"))

    with pytest.raises(HTTPException) as failure:
        asyncio.run(_call_endpoint(endpoint, upload))

    assert (failure.value.status_code, failure.value.detail) == (
        400,
        "Decoded image format must be JPEG or PNG",
    )


@pytest.mark.parametrize("endpoint", [main.detect, main.bcs])
def test_upload_validation_maps_read_failures_to_sanitized_400(endpoint) -> None:
    upload = _Upload("image/jpeg", failure=RuntimeError("private upload failure"))
    with pytest.raises(HTTPException) as failure:
        asyncio.run(_call_endpoint(endpoint, upload))
    assert (failure.value.status_code, failure.value.detail) == (
        400,
        "Failed to read uploaded file",
    )
    assert upload.read_count == 1
    assert "private" not in str(failure.value)


def test_detect_uses_uploaded_bytes_once_and_preserves_success(monkeypatch) -> None:
    class FakeDetector:
        def detect(self, image_bytes: bytes):
            assert image_bytes == payload
            return [], 10, 20, 1.234

    payload = _valid_image_bytes()
    upload = _Upload("image/jpeg", payload)
    monkeypatch.setattr(main.app.state, "detector", FakeDetector(), raising=False)
    monkeypatch.setattr(main, "get_detector", lambda **_: FakeDetector())
    response = asyncio.run(main.detect(_request(), upload))

    assert response.detection_count == 0
    assert response.image_width == 10
    assert response.image_height == 20
    assert response.inference_time_ms == 1.23
    assert upload.read_count == 1


def test_bcs_valid_upload_returns_category(monkeypatch) -> None:
    service = _FakeService()
    runtime = _FakeRuntime(service)
    upload = _Upload("image/png", _valid_image_bytes("PNG"))
    monkeypatch.setattr(main, "get_bcs_runtime", lambda: runtime)
    monkeypatch.setattr(main, "get_detector", lambda **_: pytest.fail("YOLO must not run"))
    response = asyncio.run(main.bcs(upload))

    assert response.status == "ok"
    assert response.message == "BCS category 1..5 computed successfully."
    assert response.bcs_category == 4
    assert response.cow_detected is None
    assert service.received == [upload.payload]
    assert upload.read_count == 1


def test_bcs_inference_failure_logs_safe_event_and_status(monkeypatch, caplog) -> None:
    failure = BCSInferenceExecutionError("secret model detail")
    runtime = _FakeRuntime(_FakeService(failure=failure))
    monkeypatch.setattr(main, "get_bcs_runtime", lambda: runtime)

    with caplog.at_level("ERROR", logger="vacca_api.main"):
        with pytest.raises(HTTPException) as captured:
            asyncio.run(main.bcs(_Upload("image/jpeg", _valid_image_bytes())))

    assert captured.value.status_code == 500
    assert any("BCS inference failed: BCSInferenceExecutionError" in message for message in caplog.messages)
    assert all("secret" not in message for message in caplog.messages)


def test_unconfigured_bcs_upload_returns_503_without_loading(monkeypatch) -> None:
    runtime = BCSRuntime({})
    monkeypatch.setattr(main, "get_bcs_runtime", lambda: runtime)

    with pytest.raises(HTTPException) as captured:
        asyncio.run(main.bcs(_Upload("image/jpeg", _valid_image_bytes())))

    assert (captured.value.status_code, captured.value.detail) == (
        503,
        "BCS capability is unavailable",
    )
    assert runtime.status is BCSRuntimeStatus.UNCONFIGURED


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
        asyncio.run(main.bcs(_Upload("image/jpeg", _valid_image_bytes())))
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
    detect_responses = schema["paths"]["/detect"]["post"]["responses"]
    bcs_responses = schema["paths"]["/bcs"]["post"]["responses"]
    assert detect_responses["200"]["content"]
    assert bcs_responses["200"]["content"]
    assert {"400", "413", "500", "503", "422"}.issubset(detect_responses)
    assert {"400", "413", "500", "503", "422"}.issubset(bcs_responses)
    assert detect_responses["503"]["description"] == "Inference capacity is busy"
    assert "Inference capacity is busy" in bcs_responses["503"]["description"]
    for responses in (detect_responses, bcs_responses):
        for status in ("400", "413", "500", "503"):
            assert responses[status]["content"]["application/json"]["schema"]["$ref"].endswith(
                "/ErrorResponse"
            )

    class FakeDetector:
        model_path = "safe-model"
        gpu_available = False

    detector = FakeDetector()
    monkeypatch.setattr(main.app.state, "detector", detector, raising=False)
    response = main.health(_request())
    assert response.model_loaded is True
    assert runtime.get_calls == 0


def test_detector_default_confidence_reaches_yolo_inference(monkeypatch) -> None:
    class RecordingModel:
        def __init__(self, model_path: str) -> None:
            self.model_path = model_path
            self.inference_confidence = None

        def __call__(self, image, *, conf: float, verbose: bool):
            self.inference_confidence = conf
            return []

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=RecordingModel))
    monkeypatch.setattr(detection, "_detector", None)
    detector = detection.get_detector(model_path="configured-model.pt")

    detector.detect(_valid_image_bytes())

    assert detector._model.inference_confidence == detection.DEFAULT_CONFIDENCE


@pytest.mark.parametrize("endpoint", [main.detect, main.bcs])
def test_blocked_inference_does_not_block_health_or_bcs_readiness(endpoint, monkeypatch) -> None:
    started = threading.Event()
    release = threading.Event()

    def wait_for_release() -> None:
        started.set()
        assert release.wait(timeout=2)

    async def exercise() -> None:
        monkeypatch.setattr(
            main.app.state,
            "inference_capacity_gate",
            main.InferenceCapacityGate(),
            raising=False,
        )
        monkeypatch.setattr(
            main.app.state,
            "detector",
            SimpleNamespace(gpu_available=False),
            raising=False,
        )
        if endpoint is main.detect:
            class BlockingDetector:
                gpu_available = False

                def detect(self, image_bytes: bytes):
                    wait_for_release()
                    return [], 8, 6, 1.0

            monkeypatch.setattr(main.app.state, "detector", BlockingDetector(), raising=False)
            task = asyncio.create_task(
                main.detect(_request(), _Upload("image/jpeg", _valid_image_bytes()))
            )
        else:
            class BlockingService:
                def infer(self, image_bytes: bytes):
                    wait_for_release()
                    return SimpleNamespace(bcs_category=3)

            service = BlockingService()
            runtime = SimpleNamespace(status="ready", get_service=lambda: service)
            monkeypatch.setattr(main, "get_bcs_runtime", lambda: runtime)
            task = asyncio.create_task(main.bcs(_Upload("image/jpeg", _valid_image_bytes())))

        await asyncio.wait_for(run_in_threadpool(started.wait, 2), timeout=2)
        health = await asyncio.wait_for(
            run_in_threadpool(main.health, _request()), timeout=2
        )
        assert health.model_loaded is True
        readiness = await asyncio.wait_for(
            run_in_threadpool(main.bcs_readiness), timeout=2
        )
        if endpoint is main.bcs:
            assert readiness.status == "ready"
        else:
            assert readiness.status_code == 503
        release.set()
        await task

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("blocked_endpoint", "busy_endpoint"),
    [(main.detect, main.bcs), (main.bcs, main.detect)],
)
def test_shared_inference_capacity_returns_busy_without_second_runtime_call(
    blocked_endpoint, busy_endpoint, monkeypatch
) -> None:
    started = threading.Event()
    release = threading.Event()
    detector_calls = 0
    runtime_calls = 0

    def wait_for_release() -> None:
        started.set()
        assert release.wait(timeout=2)

    class BlockingDetector:
        gpu_available = False

        def detect(self, image_bytes: bytes):
            wait_for_release()
            return [], 8, 6, 1.0

    class BlockingService:
        def infer(self, image_bytes: bytes):
            wait_for_release()
            return SimpleNamespace(bcs_category=3)

    def never_called_detector(image_bytes: bytes):
        nonlocal detector_calls
        detector_calls += 1
        raise AssertionError("busy request invoked detector")

    def never_called_runtime():
        nonlocal runtime_calls
        runtime_calls += 1
        raise AssertionError("busy request invoked BCS runtime")

    async def exercise() -> None:
        monkeypatch.setattr(
            main.app.state,
            "inference_capacity_gate",
            main.InferenceCapacityGate(),
            raising=False,
        )
        if blocked_endpoint is main.detect:
            monkeypatch.setattr(main.app.state, "detector", BlockingDetector(), raising=False)
            monkeypatch.setattr(main, "get_bcs_runtime", never_called_runtime)
        else:
            monkeypatch.setattr(
                main.app.state,
                "detector",
                SimpleNamespace(detect=never_called_detector, gpu_available=False),
                raising=False,
            )
            service = BlockingService()
            monkeypatch.setattr(
                main,
                "get_bcs_runtime",
                lambda: SimpleNamespace(status="ready", get_service=lambda: service),
            )

        first = asyncio.create_task(
            _call_endpoint(blocked_endpoint, _Upload("image/jpeg", _valid_image_bytes()))
        )
        try:
            await asyncio.wait_for(run_in_threadpool(started.wait, 2), timeout=2)
            with pytest.raises(HTTPException) as failure:
                await asyncio.wait_for(
                    _call_endpoint(busy_endpoint, _Upload("image/jpeg", _valid_image_bytes())),
                    timeout=1,
                )
            assert (failure.value.status_code, failure.value.detail) == (
                503,
                main.INFERENCE_CAPACITY_BUSY_DETAIL,
            )
            assert detector_calls == 0
            assert runtime_calls == 0
        finally:
            release.set()
            await first

    asyncio.run(exercise())


@pytest.mark.parametrize("endpoint", [main.detect, main.bcs])
def test_inference_capacity_releases_after_success(endpoint, monkeypatch) -> None:
    calls = 0

    class Detector:
        gpu_available = False

        def detect(self, image_bytes: bytes):
            nonlocal calls
            calls += 1
            return [], 8, 6, 1.0

    service = _FakeService()

    async def exercise() -> None:
        monkeypatch.setattr(
            main.app.state,
            "inference_capacity_gate",
            main.InferenceCapacityGate(),
            raising=False,
        )
        monkeypatch.setattr(main.app.state, "detector", Detector(), raising=False)
        monkeypatch.setattr(
            main,
            "get_bcs_runtime",
            lambda: SimpleNamespace(status="ready", get_service=lambda: service),
        )
        await _call_endpoint(endpoint, _Upload("image/jpeg", _valid_image_bytes()))
        await _call_endpoint(endpoint, _Upload("image/jpeg", _valid_image_bytes()))

    asyncio.run(exercise())
    if endpoint is main.detect:
        assert calls == 2
    else:
        assert len(service.received) == 2


@pytest.mark.parametrize("endpoint", [main.detect, main.bcs])
def test_inference_capacity_releases_after_failure(endpoint, monkeypatch) -> None:
    calls = 0

    class FailingThenWorkingDetector:
        gpu_available = False

        def detect(self, image_bytes: bytes):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("secret detector failure")
            return [], 8, 6, 1.0

    class FailingThenWorkingService:
        def infer(self, image_bytes: bytes):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise BCSInferenceExecutionError("secret BCS failure")
            return SimpleNamespace(bcs_category=3)

    async def exercise() -> None:
        monkeypatch.setattr(
            main.app.state,
            "inference_capacity_gate",
            main.InferenceCapacityGate(),
            raising=False,
        )
        if endpoint is main.detect:
            monkeypatch.setattr(
                main.app.state,
                "detector",
                FailingThenWorkingDetector(),
                raising=False,
            )
        else:
            service = FailingThenWorkingService()
            monkeypatch.setattr(
                main,
                "get_bcs_runtime",
                lambda: SimpleNamespace(status="ready", get_service=lambda: service),
            )

        with pytest.raises(HTTPException) as failure:
            await _call_endpoint(endpoint, _Upload("image/jpeg", _valid_image_bytes()))
        assert failure.value.status_code == 500
        await _call_endpoint(endpoint, _Upload("image/jpeg", _valid_image_bytes()))

    asyncio.run(exercise())
    assert calls == 2


def test_prototype_ui_uses_active_detection_model_label() -> None:
    html = (main.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "Modelo: vacca-yolo26n-v1.pt" in html


def test_prototype_ui_delivery_and_bcs_contract() -> None:
    response = main.prototype_ui()
    assert response.status_code == 200
    html = response.body.decode("utf-8")
    for fragment in (
        'role="tablist"', 'id="detectTab"', 'id="bcsTab"', 'aria-selected=',
        'id="bcsReadiness"', 'GET /ready/bcs', 'POST /bcs', 'Calculate BCS',
        'bcs_category', 'Not reported', 'aria-live="polite"', 'vacca-yolo26n-v1.pt',
        'ready', 'not_loaded', 'unconfigured', 'unavailable',
    ):
        assert fragment in html
    assert "innerHTML" not in html
    assert "FileReader" not in html

from __future__ import annotations

import asyncio
from io import BytesIO
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image
from starlette.datastructures import Headers
from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vacca_api import main as api_main  # noqa: E402
from vacca_api import upload_validation  # noqa: E402


class APIRouteContractTests(unittest.TestCase):
    def test_existing_api_routes_remain_available(self) -> None:
        routes = {
            (route.path, frozenset(route.methods))
            for route in api_main.app.routes
        }

        for expected in (
            ("/health", frozenset({"GET"})),
            ("/detect", frozenset({"POST"})),
            ("/bcs", frozenset({"POST"})),
            ("/ready/bcs", frozenset({"GET"})),
            ("/ui", frozenset({"GET"})),
        ):
            with self.subTest(route=expected[0]):
                self.assertIn(expected, routes)

    def test_cors_does_not_add_origin_or_preflight_headers(self) -> None:
        async def request(method: str, path: str, headers: dict[str, str]) -> tuple[int, dict[str, str]]:
            messages: list[dict[str, object]] = []

            async def receive() -> dict[str, object]:
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message: dict[str, object]) -> None:
                messages.append(message)

            await api_main.app(
                {
                    "type": "http",
                    "asgi": {"version": "3.0", "spec_version": "2.3"},
                    "http_version": "1.1",
                    "method": method,
                    "scheme": "http",
                    "path": path,
                    "raw_path": path.encode("ascii"),
                    "query_string": b"",
                    "headers": [(name.encode(), value.encode()) for name, value in headers.items()],
                    "client": ("test", 0),
                    "server": ("testserver", 80),
                    "root_path": "",
                },
                receive,
                send,
            )
            start = next(message for message in messages if message["type"] == "http.response.start")
            return int(start["status"]), {
                name.decode().lower(): value.decode()
                for name, value in start["headers"]  # type: ignore[index]
            }

        detector = SimpleNamespace(gpu_available=False)
        with patch.object(api_main.app.state, "detector", detector, create=True):
            status, response_headers = asyncio.run(
                request("GET", "/health", {"origin": "https://untrusted.example"})
            )
        self.assertEqual(status, 200)
        self.assertNotIn("access-control-allow-origin", response_headers)

        status, response_headers = asyncio.run(
            request(
                "OPTIONS",
                "/detect",
                {
                    "origin": "https://untrusted.example",
                    "access-control-request-method": "POST",
                },
            )
        )
        self.assertEqual(status, 405)
        self.assertNotIn("access-control-allow-origin", response_headers)
        self.assertNotIn("access-control-allow-methods", response_headers)

    def test_health_uses_a_stable_model_identifier(self) -> None:
        detector = SimpleNamespace(model_path="C:\\private\\model.pt", gpu_available=False)

        with patch.object(api_main.app.state, "detector", detector, create=True):
            response = api_main.health(Request({"type": "http", "app": api_main.app}))

        self.assertEqual(response.model_path, "vacca-yolo26n-v1.pt")
        self.assertNotIn("\\", response.model_path)


class DetectorOwnershipTests(unittest.TestCase):
    def test_handlers_use_the_lifespan_detector_instead_of_the_factory(self) -> None:
        detector_calls: list[bytes] = []
        detector = SimpleNamespace(
            detect=lambda payload: (detector_calls.append(payload) or ([], 8, 6, 1.0)),
            gpu_available=False,
        )
        service = SimpleNamespace(infer=lambda _: SimpleNamespace(bcs_category=3))
        runtime = SimpleNamespace(get_service=lambda: service)
        image = BytesIO()
        Image.new("RGB", (8, 6), color=(10, 20, 30)).save(image, format="JPEG")

        async def exercise() -> None:
            with patch.object(api_main, "get_detector", return_value=detector) as factory:
                async with api_main.app.router.lifespan_context(api_main.app):
                    factory.assert_called_once_with()
                    request = Request({"type": "http", "app": api_main.app})
                    with patch.object(
                        api_main,
                        "get_detector",
                        side_effect=AssertionError("request handler used the factory"),
                    ):
                        health = api_main.health(request)
                        detect = await api_main.detect(
                            request,
                            UploadFile(
                                file=BytesIO(image.getvalue()),
                                filename="upload.jpg",
                                headers=Headers({"content-type": "image/jpeg"}),
                            ),
                        )
                    self.assertIs(api_main.app.state.detector, detector)
                    self.assertFalse(health.gpu_available)
                    self.assertEqual(detect.detection_count, 0)
                    with patch.object(api_main, "get_bcs_runtime", return_value=runtime):
                        bcs = await api_main.bcs(
                            UploadFile(
                                file=BytesIO(image.getvalue()),
                                filename="upload.jpg",
                                headers=Headers({"content-type": "image/jpeg"}),
                            ),
                        )

                    self.assertEqual(bcs.bcs_category, 3)

        asyncio.run(exercise())


class UploadSafetyTests(unittest.TestCase):
    @staticmethod
    def image_bytes() -> bytes:
        stream = BytesIO()
        Image.new("RGB", (8, 6), color=(10, 20, 30)).save(stream, format="JPEG")
        return stream.getvalue()

    @staticmethod
    def upload(payload: bytes, content_type: str = "image/jpeg") -> UploadFile:
        return UploadFile(
            file=BytesIO(payload),
            filename="upload.jpg",
            headers=Headers({"content-type": content_type}),
        )

    def setUp(self) -> None:
        self.detector = SimpleNamespace(
            detect=Mock(return_value=([], 8, 6, 1.0)),
            gpu_available=False,
        )
        self.request = Request({"type": "http", "app": api_main.app})
        self.had_detector = hasattr(api_main.app.state, "detector")
        self.previous_detector = getattr(api_main.app.state, "detector", None)
        api_main.app.state.detector = self.detector
        self.bcs_runtime = SimpleNamespace(
            get_service=Mock(
                return_value=SimpleNamespace(
                    infer=Mock(return_value=SimpleNamespace(bcs_category=3))
                )
            )
        )

    def tearDown(self) -> None:
        if self.had_detector:
            api_main.app.state.detector = self.previous_detector
        else:
            del api_main.app.state.detector

    def call(self, endpoint: object, upload: UploadFile) -> object:
        if endpoint is api_main.detect:
            return asyncio.run(endpoint(self.request, upload))  # type: ignore[operator]
        return asyncio.run(endpoint(upload))  # type: ignore[operator]

    def test_exact_byte_boundary_is_accepted_by_both_endpoints(self) -> None:
        payload = self.image_bytes()
        boundary_payload = payload + b"x" * 5
        with patch.object(upload_validation, "MAX_UPLOAD_BYTES", len(boundary_payload)):
            with patch.object(api_main, "get_bcs_runtime", return_value=self.bcs_runtime):
                detect_response = self.call(api_main.detect, self.upload(boundary_payload))
                bcs_response = self.call(api_main.bcs, self.upload(boundary_payload))

        self.assertEqual(detect_response.detection_count, 0)
        self.assertEqual(bcs_response.bcs_category, 3)

    def test_image_jpg_is_accepted_by_both_endpoints(self) -> None:
        with patch.object(api_main, "get_bcs_runtime", return_value=self.bcs_runtime):
            detect_response = self.call(
                api_main.detect,
                self.upload(self.image_bytes(), "image/jpg"),
            )
            bcs_response = self.call(
                api_main.bcs,
                self.upload(self.image_bytes(), "image/jpg"),
            )

        self.assertEqual(detect_response.detection_count, 0)
        self.assertEqual(bcs_response.bcs_category, 3)

    def test_image_png_is_accepted_by_both_endpoints(self) -> None:
        payload = BytesIO()
        Image.new("RGB", (8, 6), color=(10, 20, 30)).save(payload, format="PNG")
        with patch.object(api_main, "get_bcs_runtime", return_value=self.bcs_runtime):
            detect_response = self.call(
                api_main.detect,
                self.upload(payload.getvalue(), "image/png"),
            )
            bcs_response = self.call(
                api_main.bcs,
                self.upload(payload.getvalue(), "image/png"),
            )

        self.assertEqual(detect_response.detection_count, 0)
        self.assertEqual(bcs_response.bcs_category, 3)

    def test_over_limit_is_rejected_with_413_before_inference_for_both_endpoints(self) -> None:
        payload = self.image_bytes()
        with patch.object(upload_validation, "MAX_UPLOAD_BYTES", len(payload) - 1):
            with patch.object(api_main, "get_bcs_runtime", return_value=self.bcs_runtime):
                for endpoint in (api_main.detect, api_main.bcs):
                    with self.subTest(endpoint=endpoint.__name__):
                        with self.assertRaises(HTTPException) as context:
                            self.call(endpoint, self.upload(payload))
                        self.assertEqual(context.exception.status_code, 413)
                        self.assertEqual(
                            context.exception.detail,
                            f"Image file exceeds the maximum size of {len(payload) - 1} bytes",
                        )
        self.detector.detect.assert_not_called()
        self.bcs_runtime.get_service.assert_not_called()

    def test_empty_upload_is_rejected_with_400_by_both_endpoints(self) -> None:
        with patch.object(api_main, "get_bcs_runtime", return_value=self.bcs_runtime):
            for endpoint in (api_main.detect, api_main.bcs):
                with self.subTest(endpoint=endpoint.__name__):
                    with self.assertRaises(HTTPException) as context:
                        self.call(endpoint, self.upload(b""))
                    self.assertEqual(context.exception.status_code, 400)
                    self.assertEqual(context.exception.detail, "Empty file")
        self.detector.detect.assert_not_called()
        self.bcs_runtime.get_service.assert_not_called()

    def test_upload_read_failure_is_rejected_with_400_by_both_endpoints(self) -> None:
        with patch.object(api_main, "get_bcs_runtime", return_value=self.bcs_runtime):
            for endpoint in (api_main.detect, api_main.bcs):
                upload = self.upload(self.image_bytes())
                upload.read = AsyncMock(side_effect=OSError("secret"))  # type: ignore[method-assign]
                with self.subTest(endpoint=endpoint.__name__):
                    with self.assertRaises(HTTPException) as context:
                        self.call(endpoint, upload)
                    self.assertEqual(context.exception.status_code, 400)
                    self.assertEqual(context.exception.detail, "Failed to read uploaded file")
        self.detector.detect.assert_not_called()
        self.bcs_runtime.get_service.assert_not_called()

    def test_invalid_mime_is_rejected_by_both_endpoints(self) -> None:
        with patch.object(api_main, "get_bcs_runtime", return_value=self.bcs_runtime):
            for endpoint in (api_main.detect, api_main.bcs):
                with self.subTest(endpoint=endpoint.__name__):
                    with self.assertRaises(HTTPException) as context:
                        self.call(
                            endpoint,
                            self.upload(self.image_bytes(), "application/octet-stream"),
                        )
                    self.assertEqual(context.exception.status_code, 400)
                    self.assertEqual(context.exception.detail, "File must be an image (JPEG or PNG)")
        self.detector.detect.assert_not_called()
        self.bcs_runtime.get_service.assert_not_called()

    def test_malformed_image_is_rejected_by_both_endpoints_without_traceback_logging(self) -> None:
        with patch.object(api_main, "get_bcs_runtime", return_value=self.bcs_runtime):
            with self.assertLogs(api_main.logger, level="INFO") as logs:
                for endpoint in (api_main.detect, api_main.bcs):
                    with self.subTest(endpoint=endpoint.__name__):
                        with self.assertRaises(HTTPException) as context:
                            self.call(endpoint, self.upload(b"not an image"))
                        self.assertEqual(context.exception.status_code, 400)

        self.assertTrue(all("invalid_image" in message for message in logs.output))
        self.assertTrue(all("Traceback" not in message for message in logs.output))
        self.detector.detect.assert_not_called()
        self.bcs_runtime.get_service.assert_not_called()

    def test_decoded_dimension_limit_is_rejected_by_both_endpoints(self) -> None:
        stream = BytesIO()
        Image.new("RGB", (12_001, 1), color=(10, 20, 30)).save(stream, format="JPEG")
        with patch.object(api_main, "get_bcs_runtime", return_value=self.bcs_runtime):
            for endpoint in (api_main.detect, api_main.bcs):
                with self.subTest(endpoint=endpoint.__name__):
                    with self.assertRaises(HTTPException) as context:
                        self.call(endpoint, self.upload(stream.getvalue()))
                    self.assertEqual(context.exception.status_code, 400)
        self.detector.detect.assert_not_called()
        self.bcs_runtime.get_service.assert_not_called()

    def test_inference_failure_logs_only_safe_error_type(self) -> None:
        detector = SimpleNamespace(detect=lambda _: (_ for _ in ()).throw(RuntimeError("secret")))
        with patch.object(api_main.app.state, "detector", detector):
            with self.assertLogs(api_main.logger, level="ERROR") as logs:
                with self.assertRaises(HTTPException) as context:
                    self.call(api_main.detect, self.upload(self.image_bytes()))

        self.assertEqual(context.exception.status_code, 500)
        self.assertIn("RuntimeError", logs.output[0])
        self.assertNotIn("secret", logs.output[0])

from __future__ import annotations

import asyncio
from io import BytesIO
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
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
            ("/ui", frozenset({"GET"})),
        ):
            with self.subTest(route=expected[0]):
                self.assertIn(expected, routes)

    def test_cors_is_not_enabled_for_the_server_to_server_api(self) -> None:
        self.assertFalse(
            any(
                "CORSMiddleware" in middleware.cls.__name__
                for middleware in api_main.app.user_middleware
            )
        )

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
                        bcs = await api_main.bcs(
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
                    self.assertEqual(bcs.status, "not_implemented")

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
            detect=lambda _: ([], 8, 6, 1.0),
            gpu_available=False,
        )
        self.request = Request({"type": "http", "app": api_main.app})
        self.had_detector = hasattr(api_main.app.state, "detector")
        self.previous_detector = getattr(api_main.app.state, "detector", None)
        api_main.app.state.detector = self.detector

    def tearDown(self) -> None:
        if self.had_detector:
            api_main.app.state.detector = self.previous_detector
        else:
            del api_main.app.state.detector

    def call(self, endpoint: object, upload: UploadFile) -> object:
        return asyncio.run(endpoint(self.request, upload))  # type: ignore[operator]

    def test_exact_byte_boundary_is_accepted_by_both_endpoints(self) -> None:
        payload = self.image_bytes()
        boundary_payload = payload + b"x" * 5
        with patch.object(upload_validation, "MAX_UPLOAD_BYTES", len(boundary_payload)):
            with patch.object(api_main, "get_detector", return_value=self.detector):
                detect_response = self.call(api_main.detect, self.upload(boundary_payload))
                bcs_response = self.call(api_main.bcs, self.upload(boundary_payload))

        self.assertEqual(detect_response.detection_count, 0)
        self.assertEqual(bcs_response.status, "not_implemented")

    def test_image_jpg_is_accepted_by_both_endpoints(self) -> None:
        with patch.object(api_main, "get_detector", return_value=self.detector):
            detect_response = self.call(
                api_main.detect,
                self.upload(self.image_bytes(), "image/jpg"),
            )
            bcs_response = self.call(
                api_main.bcs,
                self.upload(self.image_bytes(), "image/jpg"),
            )

        self.assertEqual(detect_response.detection_count, 0)
        self.assertEqual(bcs_response.status, "not_implemented")

    def test_over_limit_is_rejected_with_413_before_inference_for_both_endpoints(self) -> None:
        payload = self.image_bytes()
        with patch.object(upload_validation, "MAX_UPLOAD_BYTES", len(payload) - 1):
            with patch.object(api_main, "get_detector", return_value=self.detector) as get_detector:
                for endpoint in (api_main.detect, api_main.bcs):
                    with self.subTest(endpoint=endpoint.__name__):
                        with self.assertRaises(HTTPException) as context:
                            self.call(endpoint, self.upload(payload))
                        self.assertEqual(context.exception.status_code, 413)
                        self.assertEqual(
                            context.exception.detail,
                            f"Image file exceeds the maximum size of {len(payload) - 1} bytes",
                        )
                get_detector.assert_not_called()

    def test_empty_upload_is_rejected_with_400_by_both_endpoints(self) -> None:
        with patch.object(api_main, "get_detector", return_value=self.detector) as get_detector:
            for endpoint in (api_main.detect, api_main.bcs):
                with self.subTest(endpoint=endpoint.__name__):
                    with self.assertRaises(HTTPException) as context:
                        self.call(endpoint, self.upload(b""))
                    self.assertEqual(context.exception.status_code, 400)
                    self.assertEqual(context.exception.detail, "Empty file")
            get_detector.assert_not_called()

    def test_upload_read_failure_is_rejected_with_400_by_both_endpoints(self) -> None:
        with patch.object(api_main, "get_detector", return_value=self.detector) as get_detector:
            for endpoint in (api_main.detect, api_main.bcs):
                upload = self.upload(self.image_bytes())
                upload.read = AsyncMock(side_effect=OSError("secret"))  # type: ignore[method-assign]
                with self.subTest(endpoint=endpoint.__name__):
                    with self.assertRaises(HTTPException) as context:
                        self.call(endpoint, upload)
                    self.assertEqual(context.exception.status_code, 400)
                    self.assertEqual(context.exception.detail, "Failed to read uploaded file")
            get_detector.assert_not_called()

    def test_invalid_mime_is_rejected_by_both_endpoints(self) -> None:
        with patch.object(api_main, "get_detector", return_value=self.detector) as get_detector:
            for endpoint in (api_main.detect, api_main.bcs):
                with self.subTest(endpoint=endpoint.__name__):
                    with self.assertRaises(HTTPException) as context:
                        self.call(
                            endpoint,
                            self.upload(self.image_bytes(), "application/octet-stream"),
                        )
                    self.assertEqual(context.exception.status_code, 400)
                    self.assertEqual(context.exception.detail, "File must be an image (JPEG or PNG)")
            get_detector.assert_not_called()

    def test_malformed_image_is_rejected_by_both_endpoints_without_traceback_logging(self) -> None:
        with patch.object(api_main, "get_detector", return_value=self.detector):
            with self.assertLogs(api_main.logger, level="INFO") as logs:
                for endpoint in (api_main.detect, api_main.bcs):
                    with self.subTest(endpoint=endpoint.__name__):
                        with self.assertRaises(HTTPException) as context:
                            self.call(endpoint, self.upload(b"not an image"))
                        self.assertEqual(context.exception.status_code, 400)

        self.assertTrue(all("invalid_image" in message for message in logs.output))
        self.assertTrue(all("Traceback" not in message for message in logs.output))

    def test_decoded_dimension_limit_is_rejected_by_both_endpoints(self) -> None:
        stream = BytesIO()
        Image.new("RGB", (12_001, 1), color=(10, 20, 30)).save(stream, format="JPEG")
        with patch.object(api_main, "get_detector", return_value=self.detector):
            for endpoint in (api_main.detect, api_main.bcs):
                with self.subTest(endpoint=endpoint.__name__):
                    with self.assertRaises(HTTPException) as context:
                        self.call(endpoint, self.upload(stream.getvalue()))
                    self.assertEqual(context.exception.status_code, 400)

    def test_inference_failure_logs_only_safe_error_type(self) -> None:
        detector = SimpleNamespace(detect=lambda _: (_ for _ in ()).throw(RuntimeError("secret")))
        with patch.object(api_main.app.state, "detector", detector):
            with self.assertLogs(api_main.logger, level="ERROR") as logs:
                with self.assertRaises(HTTPException) as context:
                    self.call(api_main.detect, self.upload(self.image_bytes()))

        self.assertEqual(context.exception.status_code, 500)
        self.assertIn("RuntimeError", logs.output[0])
        self.assertNotIn("secret", logs.output[0])

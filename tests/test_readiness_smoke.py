from __future__ import annotations

import asyncio
import re
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_api, smoke_test_api  # noqa: E402
from vacca_api import detection  # noqa: E402
from vacca_api import main as api_main  # noqa: E402


class LauncherReadinessTests(unittest.TestCase):
    def test_default_invocation_reaches_uvicorn_with_local_defaults(self) -> None:
        calls: list[dict[str, object]] = []
        fake_uvicorn = SimpleNamespace(
            run=lambda *args, **kwargs: calls.append({"args": args, **kwargs})
        )

        with self.assertLogs(run_api.logger, level="INFO") as logs:
            with patch.dict(sys.modules, {"uvicorn": fake_uvicorn}), patch.object(
                sys, "argv", ["run_api.py"]
            ):
                run_api.main()

        emitted_messages = [record.getMessage() for record in logs.records]
        self.assertEqual(
            emitted_messages,
            [
                "Starting VACCA Vision API at http://127.0.0.1:8001",
                "API docs: http://127.0.0.1:8001/docs",
                "Test UI: http://127.0.0.1:8001/ui",
                "Health: http://127.0.0.1:8001/health",
            ],
        )
        for message in logs.output:
            message.encode("cp1252")

        self.assertEqual(
            calls,
            [
                {
                    "args": ("vacca_api.main:app",),
                    "host": "127.0.0.1",
                    "port": 8001,
                    "reload": False,
                    "log_level": "info",
                }
            ],
        )

    def test_cli_overrides_are_forwarded_to_uvicorn(self) -> None:
        calls: list[dict[str, object]] = []
        fake_uvicorn = SimpleNamespace(
            run=lambda *args, **kwargs: calls.append({"args": args, **kwargs})
        )

        with patch.dict(sys.modules, {"uvicorn": fake_uvicorn}), patch.object(
            sys,
            "argv",
            ["run_api.py", "--host", "0.0.0.0", "--port", "3000", "--reload"],
        ):
            run_api.main()

        self.assertEqual(calls[0]["host"], "0.0.0.0")
        self.assertEqual(calls[0]["port"], 3000)
        self.assertTrue(calls[0]["reload"])


class ModelPathReadinessTests(unittest.TestCase):
    def test_default_model_is_the_tracked_deployment_artifact(self) -> None:
        expected = ROOT / "models" / "deploy" / "vacca-yolo26n-v1.pt"

        self.assertEqual(detection.DEFAULT_MODEL, expected)
        self.assertTrue(expected.is_file())
        self.assertFalse(hasattr(api_main, "MODEL_PATH"))

    def test_startup_loads_detector_using_its_authoritative_default(self) -> None:
        detector = SimpleNamespace(model_path="models/deploy/vacca-yolo26n-v1.pt")

        with patch.object(api_main, "get_detector", return_value=detector) as get_detector:
            api_main._startup()

        get_detector.assert_called_once_with()
        self.assertIs(api_main.app.state.detector, detector)

    def test_lifespan_preloads_model_for_client_lifecycle_and_keeps_health_route(self) -> None:
        detector = SimpleNamespace(gpu_available=False)

        with patch.object(api_main, "get_detector", return_value=detector):
            async def request_health() -> tuple[int, object]:
                async with api_main.app.router.lifespan_context(api_main.app):
                    self.assertIs(api_main.app.state.detector, detector)
                    return await smoke_test_api._asgi_request("GET", "/health")

            status_code, body = asyncio.run(request_health())

        self.assertEqual(status_code, 200)
        self.assertEqual(body["status"], "ok")  # type: ignore[index]
        self.assertTrue(body["model_loaded"])  # type: ignore[index]

    def test_startup_failure_logs_safe_error_type(self) -> None:
        with patch.object(api_main, "get_detector", side_effect=RuntimeError("secret")):
            with self.assertLogs(api_main.logger, level="ERROR") as logs:
                with self.assertRaises(RuntimeError):
                    async def start_lifespan() -> None:
                        async with api_main.app.router.lifespan_context(api_main.app):
                            self.fail("startup should not complete")

                    asyncio.run(start_lifespan())

        self.assertIn("Model startup failed: RuntimeError", logs.output[0])
        self.assertNotIn("secret", logs.output[0])


class InProcessSmokeReadinessTests(unittest.TestCase):
    def test_parser_accepts_in_process_and_live_options(self) -> None:
        args = smoke_test_api.build_parser().parse_args(
            ["--image", "fixtures/cow.jpg", "--base-url", "http://127.0.0.1:8001", "--check-detect"]
        )

        self.assertEqual(args.image, Path("fixtures/cow.jpg"))
        self.assertEqual(args.base_url, "http://127.0.0.1:8001")
        self.assertTrue(args.check_detect)

    def test_no_image_validates_model_load_without_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "vacca-yolo26n-v1.pt"
            model.write_bytes(b"local model fixture")
            detector = SimpleNamespace(
                model_path=str(model), gpu_available=False, detect=lambda _: self.fail()
            )

            with patch.object(smoke_test_api, "DEFAULT_MODEL", model), patch.object(
                api_main, "get_detector", return_value=detector
            ) as get_detector:
                result = smoke_test_api.run()

        self.assertEqual(result, 0)
        self.assertEqual(get_detector.call_count, 1)
        get_detector.assert_any_call()

    def test_explicit_missing_image_fails_after_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "vacca-yolo26n-v1.pt"
            model.write_bytes(b"local model fixture")
            detector = SimpleNamespace(model_path=str(model), gpu_available=False)

            with patch.object(smoke_test_api, "DEFAULT_MODEL", model), patch.object(
                api_main, "get_detector", return_value=detector
            ):
                result = smoke_test_api.run(Path(directory) / "missing.jpg")

        self.assertEqual(result, 1)

    def test_explicit_image_runs_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "vacca-yolo26n-v1.pt"
            image = root / "cow.jpg"
            model.write_bytes(b"local model fixture")
            Image.new("RGB", (8, 6), color=(10, 20, 30)).save(image, format="JPEG")
            detected_bytes: list[bytes] = []
            detector = SimpleNamespace(
                model_path=str(model),
                gpu_available=False,
                detect=lambda payload: (
                    detected_bytes.append(payload) or ([], 640, 480, 1.25)
                ),
            )

            with patch.object(smoke_test_api, "DEFAULT_MODEL", model), patch.object(
                api_main, "get_detector", return_value=detector
            ):
                result = smoke_test_api.run(image)

        self.assertEqual(result, 0)
        self.assertEqual(len(detected_bytes), 1)

    def test_missing_deployment_model_fails_without_loading(self) -> None:
        missing_model = ROOT / "models" / "deploy" / "missing-readiness-model.pt"

        with patch.object(smoke_test_api, "DEFAULT_MODEL", missing_model), patch.object(
            api_main, "get_detector"
        ) as get_detector:
            result = smoke_test_api.run()

        self.assertEqual(result, 1)
        get_detector.assert_not_called()

    def test_startup_failure_returns_one_and_logs_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "vacca-yolo26n-v1.pt"
            model.write_bytes(b"local model fixture")
            with patch.object(smoke_test_api, "DEFAULT_MODEL", model), patch.object(
                api_main, "get_detector", side_effect=RuntimeError("secret")
            ):
                with self.assertLogs(smoke_test_api.logger, level="ERROR") as logs:
                    result = smoke_test_api.run()

        self.assertEqual(result, 1)
        self.assertIn("RuntimeError", logs.output[0])
        self.assertNotIn("secret", logs.output[0])

    def test_inference_failure_returns_one_and_logs_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "vacca-yolo26n-v1.pt"
            image = root / "cow.jpg"
            model.write_bytes(b"local model fixture")
            Image.new("RGB", (8, 6), color=(10, 20, 30)).save(image, format="JPEG")
            detector = SimpleNamespace(
                model_path=str(model),
                gpu_available=False,
                detect=lambda _: (_ for _ in ()).throw(RuntimeError("secret")),
            )

            with patch.object(smoke_test_api, "DEFAULT_MODEL", model), patch.object(
                api_main, "get_detector", return_value=detector
            ):
                with self.assertLogs(level="ERROR") as logs:
                    result = smoke_test_api.run(image)

        self.assertEqual(result, 1)
        self.assertTrue(any("RuntimeError" in message for message in logs.output))
        self.assertTrue(all("secret" not in message for message in logs.output))
        self.assertTrue(all("Traceback" not in message for message in logs.output))

    def test_non_200_health_response_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "vacca-yolo26n-v1.pt"
            model.write_bytes(b"local model fixture")
            with patch.object(smoke_test_api, "DEFAULT_MODEL", model), patch.object(
                smoke_test_api, "_request_health", return_value=(503, {"detail": "down"})
            ):
                result = smoke_test_api.run()

        self.assertEqual(result, 1)

    def test_malformed_health_response_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "vacca-yolo26n-v1.pt"
            model.write_bytes(b"local model fixture")
            with patch.object(smoke_test_api, "DEFAULT_MODEL", model), patch.object(
                smoke_test_api, "_request_health", return_value=(200, {"status": "ok"})
            ):
                result = smoke_test_api.run()

        self.assertEqual(result, 1)

    def test_unhealthy_status_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "vacca-yolo26n-v1.pt"
            model.write_bytes(b"local model fixture")
            for status in ("down", "degraded"):
                with self.subTest(status=status):
                    with patch.object(smoke_test_api, "DEFAULT_MODEL", model), patch.object(
                        smoke_test_api,
                        "_request_health",
                        return_value=(200, {
                            "status": status,
                            "model_loaded": True,
                            "model_path": model.name,
                            "gpu_available": False,
                        }),
                    ):
                        self.assertEqual(smoke_test_api.run(), 1)

    def test_model_loaded_false_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "vacca-yolo26n-v1.pt"
            model.write_bytes(b"local model fixture")
            with patch.object(smoke_test_api, "DEFAULT_MODEL", model), patch.object(
                smoke_test_api,
                "_request_health",
                return_value=(200, {
                    "status": "ok",
                    "model_loaded": False,
                    "model_path": model.name,
                    "gpu_available": False,
                }),
            ):
                self.assertEqual(smoke_test_api.run(), 1)


class LiveSmokeReadinessTests(unittest.TestCase):
    HEALTH = {
        "status": "ok",
        "model_loaded": True,
        "model_path": "vacca-yolo26n-v1.pt",
        "gpu_available": False,
    }

    DETECT = {
        "cow_detected": False,
        "detection_count": 0,
        "detections": [],
        "image_width": 8,
        "image_height": 6,
        "inference_time_ms": 1.25,
    }

    def test_live_health_and_controlled_detect_use_mocked_http(self) -> None:
        with patch.object(
            smoke_test_api,
            "_request_json",
            side_effect=[
                (200, self.HEALTH),
                (400, {"detail": smoke_test_api.LIVE_DETECT_INVALID_DETAIL}),
            ],
        ) as request:
            result = smoke_test_api.run(
                base_url="http://127.0.0.1:8001/",
                check_detect=True,
                timeout=1.5,
            )

        self.assertEqual(result, 0)
        self.assertEqual(request.call_args_list[0].args[0], "http://127.0.0.1:8001/health")
        self.assertEqual(request.call_args_list[1].args[0], "http://127.0.0.1:8001/detect")
        self.assertEqual(request.call_args_list[1].kwargs["timeout"], 1.5)

    def test_live_image_detect_validates_success_response_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "cow.jpg"
            Image.new("RGB", (8, 6), color=(10, 20, 30)).save(image, format="JPEG")
            with patch.object(
                smoke_test_api,
                "_request_json",
                side_effect=[(200, self.HEALTH), (200, self.DETECT)],
            ) as request:
                result = smoke_test_api.run(
                    image_path=image,
                    base_url="http://127.0.0.1:8001",
                )

        self.assertEqual(result, 0)
        self.assertEqual(request.call_args_list[1].args[0], "http://127.0.0.1:8001/detect")
        self.assertIn(b"Content-Type: image/jpeg", request.call_args_list[1].kwargs["body"])

    def test_live_malformed_detect_response_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "cow.jpg"
            Image.new("RGB", (8, 6), color=(10, 20, 30)).save(image, format="JPEG")
            with patch.object(
                smoke_test_api,
                "_request_json",
                side_effect=[(200, self.HEALTH), (200, {"cow_detected": False})],
            ):
                result = smoke_test_api.run(
                    image_path=image,
                    base_url="http://127.0.0.1:8001",
                )

        self.assertEqual(result, 1)

    def test_live_non_200_health_response_fails(self) -> None:
        with patch.object(smoke_test_api, "_request_json", return_value=(503, {"detail": "down"})):
            result = smoke_test_api.run(base_url="http://127.0.0.1:8001")

        self.assertEqual(result, 1)

    def test_live_malformed_health_response_fails(self) -> None:
        with patch.object(smoke_test_api, "_request_json", return_value=(200, {"status": "ok"})):
            result = smoke_test_api.run(base_url="http://127.0.0.1:8001")

        self.assertEqual(result, 1)

    def test_live_unhealthy_status_fails(self) -> None:
        for status in ("down", "degraded"):
            with self.subTest(status=status):
                with patch.object(
                    smoke_test_api,
                    "_request_json",
                    return_value=(200, {
                        "status": status,
                        "model_loaded": True,
                        "model_path": "vacca-yolo26n-v1.pt",
                        "gpu_available": False,
                    }),
                ):
                    self.assertEqual(
                        smoke_test_api.run(base_url="http://127.0.0.1:8001"),
                        1,
                    )

    def test_live_model_loaded_false_fails(self) -> None:
        with patch.object(
            smoke_test_api,
            "_request_json",
            return_value=(200, {
                "status": "ok",
                "model_loaded": False,
                "model_path": "vacca-yolo26n-v1.pt",
                "gpu_available": False,
            }),
        ):
            self.assertEqual(
                smoke_test_api.run(base_url="http://127.0.0.1:8001"),
                1,
            )

    def test_live_timeout_fails(self) -> None:
        with patch.object(
            smoke_test_api,
            "_request_json",
            side_effect=smoke_test_api.SmokeCheckError("live request failed"),
        ):
            result = smoke_test_api.run(base_url="http://127.0.0.1:8001", timeout=0.1)

        self.assertEqual(result, 1)


class ProjectDependencyMetadataTests(unittest.TestCase):
    def test_api_dependencies_are_bounded_in_project_metadata(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        api_dependencies = metadata["project"]["optional-dependencies"]["api"]

        self.assertEqual(
            api_dependencies,
            [
                "fastapi==0.141.1",
                "uvicorn==0.52.4",
                "python-multipart==0.0.32",
            ],
        )

    def test_api_lock_contains_exact_hashed_roots_and_transitives(self) -> None:
        lock = (ROOT / "requirements-api.txt").read_text(encoding="utf-8")
        entries = {
            name.lower().replace("_", "-"): (version, digest)
            for name, version, digest in re.findall(
                r"(?m)^([A-Za-z0-9_.-]+)==([^\s\\]+) \\\n\s+--hash=sha256:([0-9a-f]{64})$",
                lock,
            )
        }

        self.assertIn("--require-hashes", lock)
        self.assertIn("Windows x64 / CPython 3.13", lock)
        self.assertEqual(
            set(entries),
            {
                "annotated-doc",
                "annotated-types",
                "anyio",
                "click",
                "fastapi",
                "h11",
                "idna",
                "pydantic",
                "pydantic-core",
                "python-multipart",
                "starlette",
                "typing-extensions",
                "typing-inspection",
                "uvicorn",
            },
        )
        self.assertTrue(all(len(digest) == 64 for _, digest in entries.values()))

        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        for requirement in metadata["project"]["optional-dependencies"]["api"]:
            name, version = requirement.split("==")
            self.assertEqual(entries[name.lower()][0], version)

    def test_readme_uses_cpu_and_api_locks_without_dependency_resolution(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("- Python 3.13+", readme)
        self.assertIn(
            ".venv\\Scripts\\python -m pip install --require-hashes -r requirements-cpu.txt",
            readme,
        )
        self.assertIn(
            ".venv\\Scripts\\python -m pip install --require-hashes -r requirements-api.txt",
            readme,
        )
        self.assertIn(
            ".venv\\Scripts\\python -m pip install --no-deps --no-build-isolation -e \".[yolo,api]\"",
            readme,
        )
        self.assertIn('$env:IA_SERVICE_URL = "http://ia-api:8001/detect"', readme)
        self.assertIn('$env:IA_SERVICE_URL = "http://192.0.2.10:8001/detect"', readme)
        self.assertNotRegex(readme, r"(?m)^\s*IA_SERVICE_URL=")
        self.assertIn('& python -m venv "${deployRoot}\\.venv"', readme)


if __name__ == "__main__":
    unittest.main()

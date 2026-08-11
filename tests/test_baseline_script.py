from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_baseline import (  # noqa: E402
    ManifestError,
    _bounded_output_path,
    _runtime_versions,
    load_manifest,
    main,
    run,
    semantic_payload,
    verify_artifact,
)


class BaselineManifestTests(unittest.TestCase):
    def test_loads_versioned_cpu_manifest(self) -> None:
        manifest = load_manifest(ROOT / "configs" / "baseline_manifest.json")

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["runtime"]["device"], "cpu")
        self.assertEqual(manifest["runtime"]["ultralytics"], "8.4.115")
        self.assertEqual(manifest["model"]["expected_size_bytes"], 5_544_453)

    def test_rejects_non_cpu_or_incomplete_manifest(self) -> None:
        invalid_manifests = (
            {"schema_version": 1, "runtime": {"device": "cuda"}},
            {"schema_version": 2, "runtime": {"device": "cpu"}},
        )

        for payload in invalid_manifests:
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "manifest.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ManifestError):
                        load_manifest(path)

    def test_rejects_invalid_nested_manifest_values(self) -> None:
        valid = json.loads(
            (ROOT / "configs" / "baseline_manifest.json").read_text(encoding="utf-8")
        )
        mutations = (
            ("boolean schema version", lambda value: value.update(schema_version=True)),
            ("boolean seed", lambda value: value["runtime"].update(seed=True)),
            (
                "insecure model URL",
                lambda value: value["model"].update(
                    url="http://example.test/model.pt"
                ),
            ),
            ("uppercase digest", lambda value: value["fixture"].update(sha256="A" * 64)),
            (
                "invalid confidence",
                lambda value: value["thresholds"].update(model_confidence=1.1),
            ),
            (
                "invalid border margin",
                lambda value: value["thresholds"].update(border_margin_ratio=0.5),
            ),
            (
                "invalid dimensions",
                lambda value: value["thresholds"].update(minimum_width=13_000),
            ),
            ("path escape", lambda value: value["model"].update(path="../model.pt")),
            ("unknown nested key", lambda value: value["runtime"].update(extra="value")),
        )

        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                payload = copy.deepcopy(valid)
                mutate(payload)
                path = Path(directory) / "manifest.json"
                path.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaises(ManifestError):
                    load_manifest(path)

    def test_verifies_artifact_size_and_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            content = b"verified fixture"
            artifact.write_bytes(content)
            spec = {
                "path": "artifact.bin",
                "expected_size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }

            evidence = verify_artifact(root, spec, "fixture")

            self.assertEqual(evidence["size_bytes"], len(content))
            self.assertEqual(evidence["sha256"], spec["sha256"])

    def test_rejects_artifact_mismatch_and_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"actual")
            invalid_specs = (
                {
                    "path": "artifact.bin",
                    "expected_size_bytes": 999,
                    "sha256": hashlib.sha256(b"actual").hexdigest(),
                },
                {
                    "path": "../outside.bin",
                    "expected_size_bytes": 0,
                    "sha256": "0" * 64,
                },
            )

            for spec in invalid_specs:
                with self.subTest(spec=spec):
                    with self.assertRaises(ManifestError):
                        verify_artifact(root, spec, "artifact")

    def test_semantic_payload_ignores_only_timing(self) -> None:
        payload = {
            "status": "ACCEPTED",
            "detections": [{"class_name": "cow", "confidence": 0.9}],
            "timing": {"inference_ms": 10.0, "total_ms": 20.0},
            "provenance": {"device": "cpu"},
        }

        semantic = semantic_payload(payload)

        self.assertNotIn("timing", semantic)
        self.assertEqual(semantic["detections"], payload["detections"])
        self.assertEqual(payload["timing"]["total_ms"], 20.0)


class BaselineExecutionTests(unittest.TestCase):
    def _project(self, directory: str, fixture_digest: str = "f" * 64) -> tuple[Path, Path]:
        root = Path(directory)
        (root / "configs").mkdir()
        (root / "models" / "checkpoints").mkdir(parents=True)
        model_content = b"model snapshot"
        model_path = root / "models" / "checkpoints" / "model.pt"
        model_path.write_bytes(model_content)
        payload = json.loads(
            (ROOT / "configs" / "baseline_manifest.json").read_text(encoding="utf-8")
        )
        payload["model"].update(
            path="models/checkpoints/model.pt",
            expected_size_bytes=len(model_content),
            sha256=hashlib.sha256(model_content).hexdigest(),
        )
        payload["fixture"].update(
            path="data/fixture.jpg",
            expected_size_bytes=123,
            sha256=fixture_digest,
        )
        manifest_path = root / "configs" / "baseline_manifest.json"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return root, manifest_path

    @staticmethod
    def _args(manifest: Path, **overrides: object) -> argparse.Namespace:
        values: dict[str, object] = {
            "manifest": str(manifest),
            "output": None,
            "device": "cpu",
            "model_confidence": None,
            "min_confidence": None,
            "minimum_relative_area": None,
            "border_margin_ratio": None,
            "input_size": None,
            "disable_framing": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    @staticmethod
    def _runtime(_: dict[str, object]) -> dict[str, object]:
        return {
            "python": "3.13.5",
            "torch": "2.13.0+cpu",
            "torchvision": "0.28.0+cpu",
            "ultralytics": "8.4.115",
            "device": "cpu",
            "cuda_available": False,
        }

    def test_rejects_fixture_snapshot_mismatch_before_detector_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = self._project(directory)
            detector_calls: list[dict[str, object]] = []

            with self.assertRaisesRegex(ManifestError, "Fixture snapshot SHA-256"):
                run(
                    self._args(manifest),
                    root=root,
                    runtime_provider=self._runtime,
                    image_validator=lambda *_: SimpleNamespace(
                        size_bytes=123, snapshot_sha256="0" * 64
                    ),
                    detector_factory=lambda **kwargs: detector_calls.append(kwargs),
                )

            self.assertEqual(detector_calls, [])

    def test_run_wires_cpu_snapshot_and_cli_overrides_into_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture_digest = "f" * 64
            root, manifest = self._project(directory, fixture_digest)
            detector_calls: list[dict[str, object]] = []
            pipeline_calls: list[tuple[object, object]] = []
            image = SimpleNamespace(size_bytes=123, snapshot_sha256=fixture_digest)

            def detector_factory(**kwargs: object) -> object:
                detector = object()
                detector_calls.append(kwargs)
                return detector

            class FakeResult:
                @staticmethod
                def to_dict() -> dict[str, object]:
                    return {"status": "ACCEPTED", "timing": {"total_ms": 1.0}}

            class FakePipeline:
                def classify(self, received_image: object) -> FakeResult:
                    self_test.assertIs(received_image, image)
                    return FakeResult()

            self_test = self

            def pipeline_factory(detector: object, config: object) -> FakePipeline:
                pipeline_calls.append((detector, config))
                return FakePipeline()

            payload = run(
                self._args(
                    manifest,
                    output="outputs/result.json",
                    model_confidence=0.3,
                    min_confidence=0.4,
                    minimum_relative_area=0.2,
                    border_margin_ratio=0.1,
                    input_size=320,
                    disable_framing=True,
                ),
                root=root,
                runtime_provider=self._runtime,
                image_validator=lambda *_: image,
                detector_factory=detector_factory,
                pipeline_factory=pipeline_factory,
                diagnostic_stream=io.StringIO(),
            )

            self.assertEqual(detector_calls[0]["device"], "cpu")
            self.assertEqual(detector_calls[0]["prediction_confidence"], 0.3)
            self.assertEqual(detector_calls[0]["input_size"], 320)
            config = pipeline_calls[0][1]
            self.assertEqual(config.min_confidence, 0.4)
            self.assertEqual(config.min_relative_area, 0.2)
            self.assertEqual(config.border_margin_ratio, 0.1)
            self.assertFalse(config.framing_enabled)
            provenance = payload["provenance"]
            self.assertEqual(provenance["manifest"], "configs/baseline_manifest.json")
            self.assertFalse(Path(provenance["manifest"]).is_absolute())
            self.assertNotIn("..", Path(provenance["manifest"]).parts)
            self.assertEqual(
                provenance["runtime"],
                {
                    "python": "3.13.5",
                    "torch": "2.13.0+cpu",
                    "torchvision": "0.28.0+cpu",
                    "ultralytics": "8.4.115",
                    "device": "cpu",
                    "cuda_available": False,
                },
            )
            self.assertEqual(
                provenance["model_sha256"],
                "e529ba1059de57f2ed6c92dc7b8229a0de84470194e0a110fb3d4481dc2f98e7",
            )
            self.assertEqual(provenance["model_size_bytes"], 14)
            self.assertEqual(
                provenance["model_source"],
                {
                    "name": "yolo26n",
                    "version": "v8.4.0-release-asset",
                    "release_tag": "v8.4.0",
                    "url": (
                        "https://github.com/ultralytics/assets/releases/"
                        "download/v8.4.0/yolo26n.pt"
                    ),
                    "license": "AGPL-3.0",
                    "license_path": "LICENSE",
                    "trust_constraint": (
                        "Official release URL plus fixed digest; "
                        "no independent signature is available."
                    ),
                },
            )
            self.assertEqual(provenance["fixture_sha256"], "f" * 64)
            self.assertEqual(provenance["fixture_size_bytes"], 123)
            self.assertEqual(
                provenance["fixture_source"],
                {
                    "name": "cow_female_black_white",
                    "url": (
                        "https://upload.wikimedia.org/wikipedia/commons/"
                        "0/0c/Cow_female_black_white.jpg"
                    ),
                    "page_revision_url": (
                        "https://commons.wikimedia.org/w/index.php?"
                        "title=File:Cow_female_black_white.jpg&oldid=1202221419"
                    ),
                    "author": "Keith Weller, USDA Agricultural Research Service",
                    "license": "PD-USGov-USDA-ARS",
                },
            )
            self.assertEqual(
                provenance["thresholds"],
                {
                    "model_confidence": 0.3,
                    "pipeline_min_confidence": 0.4,
                    "minimum_relative_area": 0.2,
                    "border_margin_ratio": 0.1,
                    "framing_enabled": False,
                    "input_size": 320,
                    "maximum_image_size_bytes": 20_971_520,
                    "minimum_width": 1,
                    "minimum_height": 1,
                    "maximum_width": 12_000,
                    "maximum_height": 12_000,
                    "maximum_pixels": 50_000_000,
                },
            )
            written_payload = json.loads(
                (root / "outputs" / "result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(written_payload, payload)

    def test_rejects_non_cpu_cli_before_runtime_or_model_work(self) -> None:
        calls: list[str] = []
        with self.assertRaisesRegex(ManifestError, "immutable"):
            run(
                self._args(Path("unused.json"), device="cuda"),
                runtime_provider=lambda _: calls.append("runtime"),
            )
        self.assertEqual(calls, [])

    def test_rejects_runtime_version_mismatch(self) -> None:
        manifest = load_manifest(ROOT / "configs" / "baseline_manifest.json")

        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class FakeTorch:
            __version__ = "0.0.0+cpu"
            cuda = FakeCuda()

            @staticmethod
            def manual_seed(_: int) -> None:
                raise AssertionError("seed must not be set after a version mismatch")

        fake_torchvision = SimpleNamespace(__version__=manifest["runtime"]["torchvision"])
        with self.assertRaisesRegex(ManifestError, "Installed torch version"):
            _runtime_versions(
                manifest,
                python_version=manifest["runtime"]["python"],
                torch_module=FakeTorch(),
                torchvision_module=fake_torchvision,
                package_version=lambda _: manifest["runtime"]["ultralytics"],
            )

    def test_output_path_must_remain_under_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(
                _bounded_output_path("outputs/result.json", root),
                (root / "outputs" / "result.json").resolve(),
            )
            with self.assertRaises(ManifestError):
                _bounded_output_path("outside.json", root)


class BaselineMainTests(unittest.TestCase):
    @staticmethod
    def _manifest() -> dict[str, object]:
        return load_manifest(ROOT / "configs" / "baseline_manifest.json")

    def test_main_emits_stable_json_for_expected_errors_without_traceback(self) -> None:
        stderr = io.StringIO()

        def failing_runner(_: argparse.Namespace) -> dict[str, object]:
            raise ManifestError("safe failure")

        exit_code = main([], runner=failing_runner, stdout=io.StringIO(), stderr=stderr)

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            json.loads(stderr.getvalue()),
            {"status": "ERROR", "error": "safe failure"},
        )
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_main_converts_parser_errors_to_json(self) -> None:
        stderr = io.StringIO()

        exit_code = main(["--unknown"], stdout=io.StringIO(), stderr=stderr)

        self.assertEqual(exit_code, 1)
        self.assertEqual(json.loads(stderr.getvalue())["status"], "ERROR")

    def test_main_converts_missing_runtime_dependency_to_stable_json(self) -> None:
        for dependency_error in (
            ImportError("torch binary dependency is unavailable"),
            ModuleNotFoundError("No module named 'torch'"),
        ):
            with self.subTest(error_type=type(dependency_error).__name__):
                stdout = io.StringIO()
                stderr = io.StringIO()

                def missing_dependency(_: str) -> object:
                    raise dependency_error

                def runner(_: argparse.Namespace) -> dict[str, object]:
                    return _runtime_versions(
                        self._manifest(),
                        module_importer=missing_dependency,
                    )

                exit_code = main([], runner=runner, stdout=stdout, stderr=stderr)

                self.assertEqual(exit_code, 1)
                self.assertEqual(stdout.getvalue(), "")
                self.assertEqual(
                    json.loads(stderr.getvalue()),
                    {
                        "status": "ERROR",
                        "error": (
                            "Required runtime dependency 'torch' is unavailable"
                        ),
                    },
                )
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_main_converts_missing_runtime_metadata_to_stable_json(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        manifest = self._manifest()
        fake_torch = SimpleNamespace(
            __version__=manifest["runtime"]["torch"],
            cuda=SimpleNamespace(is_available=lambda: False),
            manual_seed=lambda _: None,
        )
        fake_torchvision = SimpleNamespace(
            __version__=manifest["runtime"]["torchvision"]
        )

        def missing_metadata(_: str) -> str:
            raise importlib.metadata.PackageNotFoundError("ultralytics")

        def runner(_: argparse.Namespace) -> dict[str, object]:
            return _runtime_versions(
                manifest,
                torch_module=fake_torch,
                torchvision_module=fake_torchvision,
                package_version=missing_metadata,
            )

        exit_code = main([], runner=runner, stdout=stdout, stderr=stderr)

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(
            json.loads(stderr.getvalue()),
            {
                "status": "ERROR",
                "error": "Required runtime metadata for 'ultralytics' is unavailable",
            },
        )
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_main_does_not_hide_unexpected_runtime_errors(self) -> None:
        def broken_runner(_: argparse.Namespace) -> dict[str, object]:
            raise RuntimeError("unexpected programming failure")

        with self.assertRaisesRegex(RuntimeError, "unexpected programming failure"):
            main(
                [],
                runner=broken_runner,
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )

    def test_main_prints_success_payload(self) -> None:
        stdout = io.StringIO()
        payload = {"status": "ACCEPTED"}

        exit_code = main([], runner=lambda _: payload, stdout=stdout, stderr=io.StringIO())

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), payload)


if __name__ == "__main__":
    unittest.main()

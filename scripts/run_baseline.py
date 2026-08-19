from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any, TextIO
from urllib.parse import urlsplit

from vacca_vision import (
    AptitudePipeline,
    ClassificationConfig,
    ImageValidationConfig,
    ImageValidationDependencyError,
    ImageValidationError,
    ModelIdentity,
    UltralyticsAdapterError,
    UltralyticsDependencyError,
    UltralyticsDetector,
    validate_image,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "baseline_manifest.json"


class ManifestError(ValueError):
    """Raised when a baseline manifest or its local artifacts are invalid."""


class RuntimeEnvironmentError(RuntimeError):
    """Raised when a required baseline runtime dependency is unavailable."""


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_RUNTIME_KEYS = frozenset(
    {"device", "python", "torch", "torchvision", "ultralytics", "seed"}
)
_MODEL_KEYS = frozenset(
    {
        "name",
        "version",
        "release_tag",
        "url",
        "path",
        "expected_size_bytes",
        "sha256",
        "license",
        "license_path",
        "trust_constraint",
    }
)
_FIXTURE_KEYS = frozenset(
    {
        "name",
        "url",
        "page_revision_url",
        "path",
        "expected_size_bytes",
        "sha256",
        "author",
        "license",
    }
)
_THRESHOLD_KEYS = frozenset(
    {
        "model_confidence",
        "pipeline_min_confidence",
        "minimum_relative_area",
        "border_margin_ratio",
        "framing_enabled",
        "input_size",
        "maximum_image_size_bytes",
        "minimum_width",
        "minimum_height",
        "maximum_width",
        "maximum_height",
        "maximum_pixels",
    }
)


def _require_exact_keys(
    payload: dict[str, Any], expected: frozenset[str], label: str
) -> None:
    if set(payload) != expected:
        raise ManifestError(f"{label} keys do not match schema")


def _require_string(payload: dict[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{label}.{key} must be a non-empty string")
    return value


def _require_positive_integer(payload: dict[str, Any], key: str, label: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestError(f"{label}.{key} must be a positive integer")
    return value


def _require_unit_number(payload: dict[str, Any], key: str, label: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{label}.{key} must be a number between 0 and 1")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ManifestError(f"{label}.{key} must be a number between 0 and 1")
    return result


def _validate_https_url(payload: dict[str, Any], key: str, label: str) -> None:
    value = _require_string(payload, key, label)
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ManifestError(f"{label}.{key} must be an HTTPS URL without credentials")


def _validate_relative_path(payload: dict[str, Any], key: str, label: str) -> None:
    value = _require_string(payload, key, label)
    path = Path(value)
    if path.is_absolute() or "\\" in value or any(part == ".." for part in path.parts):
        raise ManifestError(f"{label}.{key} must be a project-relative path")


def _validate_artifact_spec(
    payload: dict[str, Any], expected_keys: frozenset[str], label: str
) -> None:
    _require_exact_keys(payload, expected_keys, label)
    _validate_relative_path(payload, "path", label)
    _require_positive_integer(payload, "expected_size_bytes", label)
    digest = _require_string(payload, "sha256", label)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ManifestError(f"{label}.sha256 must be a lowercase SHA-256 digest")


def _validate_manifest(payload: dict[str, Any]) -> None:
    _require_exact_keys(
        payload,
        frozenset(
            {"schema_version", "created_at_utc", "runtime", "model", "fixture", "thresholds"}
        ),
        "manifest",
    )
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ManifestError("Unsupported baseline manifest schema")
    created_at = _require_string(payload, "created_at_utc", "manifest")
    try:
        dt.datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise ManifestError("manifest.created_at_utc must be a UTC ISO-8601 timestamp") from None

    runtime = payload.get("runtime")
    model = payload.get("model")
    fixture = payload.get("fixture")
    thresholds = payload.get("thresholds")
    if not all(isinstance(section, dict) for section in (runtime, model, fixture, thresholds)):
        raise ManifestError("Baseline manifest is missing required sections")
    assert isinstance(runtime, dict)
    assert isinstance(model, dict)
    assert isinstance(fixture, dict)
    assert isinstance(thresholds, dict)

    _require_exact_keys(runtime, _RUNTIME_KEYS, "runtime")
    if runtime.get("device") != "cpu":
        raise ManifestError("Baseline manifest must force CPU execution")
    for key in ("python", "torch", "torchvision", "ultralytics"):
        _require_string(runtime, key, "runtime")
    seed = runtime.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ManifestError("runtime.seed must be a non-negative integer")

    _validate_artifact_spec(model, _MODEL_KEYS, "model")
    for key in ("name", "version", "release_tag", "license", "trust_constraint"):
        _require_string(model, key, "model")
    _validate_https_url(model, "url", "model")
    _validate_relative_path(model, "license_path", "model")

    _validate_artifact_spec(fixture, _FIXTURE_KEYS, "fixture")
    for key in ("name", "author", "license"):
        _require_string(fixture, key, "fixture")
    for key in ("url", "page_revision_url"):
        _validate_https_url(fixture, key, "fixture")

    _require_exact_keys(thresholds, _THRESHOLD_KEYS, "thresholds")
    for key in (
        "model_confidence",
        "pipeline_min_confidence",
        "minimum_relative_area",
        "border_margin_ratio",
    ):
        _require_unit_number(thresholds, key, "thresholds")
    if thresholds["border_margin_ratio"] >= 0.5:
        raise ManifestError("thresholds.border_margin_ratio must be lower than 0.5")
    if not isinstance(thresholds.get("framing_enabled"), bool):
        raise ManifestError("thresholds.framing_enabled must be a boolean")
    for key in (
        "input_size",
        "maximum_image_size_bytes",
        "minimum_width",
        "minimum_height",
        "maximum_width",
        "maximum_height",
        "maximum_pixels",
    ):
        _require_positive_integer(thresholds, key, "thresholds")
    if (
        thresholds["minimum_width"] > thresholds["maximum_width"]
        or thresholds["minimum_height"] > thresholds["maximum_height"]
    ):
        raise ManifestError("Minimum dimensions cannot exceed maximum dimensions")
    if thresholds["minimum_width"] * thresholds["minimum_height"] > thresholds["maximum_pixels"]:
        raise ManifestError("Minimum dimensions cannot exceed maximum_pixels")


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ManifestError("Baseline manifest cannot be read") from None
    if not isinstance(payload, dict):
        raise ManifestError("Unsupported baseline manifest schema")
    _validate_manifest(payload)
    return payload


def verify_artifact(
    root: Path,
    spec: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    try:
        path = (root / str(spec["path"])).resolve()
        path.relative_to(root.resolve())
        expected_size = int(spec["expected_size_bytes"])
        expected_sha256 = str(spec["sha256"]).casefold()
    except (KeyError, TypeError, ValueError):
        raise ManifestError(f"{label} manifest entry is invalid") from None
    if not path.is_file():
        raise ManifestError(f"{label} file is missing")
    size_bytes = path.stat().st_size
    if size_bytes != expected_size:
        raise ManifestError(f"{label} size does not match manifest")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as artifact_file:
            for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        raise ManifestError(f"{label} file cannot be read") from None
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ManifestError(f"{label} SHA-256 does not match manifest")
    return {
        "path": path,
        "size_bytes": size_bytes,
        "sha256": actual_sha256,
    }


def semantic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    semantic = dict(payload)
    semantic.pop("timing", None)
    return semantic


def _runtime_versions(
    manifest: dict[str, Any],
    *,
    python_version: str | None = None,
    torch_module: Any = None,
    torchvision_module: Any = None,
    module_importer: Callable[[str], Any] = importlib.import_module,
    package_version: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, Any]:
    if torch_module is None:
        torch_module = _import_runtime_dependency("torch", module_importer)
    if torchvision_module is None:
        torchvision_module = _import_runtime_dependency("torchvision", module_importer)

    try:
        ultralytics_version = package_version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeEnvironmentError(
            "Required runtime metadata for 'ultralytics' is unavailable"
        ) from None

    actual = {
        "python": python_version or ".".join(map(str, sys.version_info[:3])),
        "torch": torch_module.__version__,
        "torchvision": torchvision_module.__version__,
        "ultralytics": ultralytics_version,
        "cuda_available": torch_module.cuda.is_available(),
        "device": "cpu",
    }
    for package in ("torch", "torchvision", "ultralytics"):
        if actual[package] != manifest["runtime"].get(package):
            raise ManifestError(f"Installed {package} version does not match manifest")
    if actual["python"] != manifest["runtime"].get("python"):
        raise ManifestError("Python version does not match manifest")
    torch_module.manual_seed(manifest["runtime"]["seed"])
    return actual


def _import_runtime_dependency(
    name: str,
    module_importer: Callable[[str], Any],
) -> Any:
    try:
        return module_importer(name)
    except ImportError:
        raise RuntimeEnvironmentError(
            f"Required runtime dependency '{name}' is unavailable"
        ) from None


def _bounded_output_path(value: str | None, root: Path = ROOT) -> Path | None:
    if value is None:
        return None
    output_root = (root / "outputs").resolve()
    output_path = (root / value).resolve()
    try:
        output_path.relative_to(output_root)
    except ValueError:
        raise ManifestError("Output path must stay under outputs/") from None
    return output_path


def _project_path(root: Path, value: str | Path, label: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise ManifestError(f"{label} path must stay within the project") from None
    return resolved


def _effective_unit_value(value: float | None, fallback: Any, label: str) -> float:
    result = float(fallback) if value is None else value
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ManifestError(f"{label} must be between 0 and 1")
    return result


def run(
    args: argparse.Namespace,
    *,
    root: Path = ROOT,
    runtime_provider: Callable[[dict[str, Any]], dict[str, Any]] = _runtime_versions,
    image_validator: Callable[..., Any] = validate_image,
    detector_factory: Callable[..., Any] = UltralyticsDetector,
    pipeline_factory: Callable[..., Any] = AptitudePipeline,
    diagnostic_stream: TextIO | None = None,
) -> dict[str, Any]:
    if args.device != "cpu":
        raise ManifestError("Baseline execution device is immutable and must be cpu")
    manifest_path = _project_path(root, args.manifest, "Manifest")
    manifest = load_manifest(manifest_path)
    runtime = runtime_provider(manifest)
    model_evidence = verify_artifact(root, manifest["model"], "Model")
    thresholds = manifest["thresholds"]

    fixture_path = _project_path(root, manifest["fixture"]["path"], "Fixture")
    image = image_validator(
        fixture_path,
        ImageValidationConfig(
            max_size_bytes=thresholds["maximum_image_size_bytes"],
            min_width=thresholds["minimum_width"],
            min_height=thresholds["minimum_height"],
            max_width=thresholds["maximum_width"],
            max_height=thresholds["maximum_height"],
            max_pixels=thresholds["maximum_pixels"],
        ),
    )
    if image.size_bytes != manifest["fixture"]["expected_size_bytes"]:
        raise ManifestError("Fixture snapshot size does not match manifest")
    if image.snapshot_sha256 != manifest["fixture"]["sha256"]:
        raise ManifestError("Fixture snapshot SHA-256 does not match manifest")

    model_confidence = _effective_unit_value(
        args.model_confidence, thresholds["model_confidence"], "model confidence"
    )
    pipeline_confidence = _effective_unit_value(
        args.min_confidence, thresholds["pipeline_min_confidence"], "minimum confidence"
    )
    minimum_relative_area = _effective_unit_value(
        args.minimum_relative_area,
        thresholds["minimum_relative_area"],
        "minimum relative area",
    )
    border_margin_ratio = _effective_unit_value(
        args.border_margin_ratio, thresholds["border_margin_ratio"], "border margin ratio"
    )
    if border_margin_ratio >= 0.5:
        raise ManifestError("border margin ratio must be lower than 0.5")
    framing_enabled = bool(thresholds["framing_enabled"]) and not args.disable_framing
    input_size = args.input_size if args.input_size is not None else thresholds["input_size"]
    if isinstance(input_size, bool) or input_size <= 0:
        raise ManifestError("input size must be a positive integer")

    ultralytics_config = (root / "outputs" / "ultralytics-config").resolve()
    ultralytics_config.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(ultralytics_config)
    detector = detector_factory(
        weights_path=model_evidence["path"],
        expected_weights_sha256=model_evidence["sha256"],
        device="cpu",
        model_identity=ModelIdentity(
            name=str(manifest["model"]["name"]),
            version=str(manifest["model"]["version"]),
        ),
        prediction_confidence=model_confidence,
        input_size=input_size,
    )
    pipeline = pipeline_factory(
        detector,
        ClassificationConfig(
            min_confidence=pipeline_confidence,
            min_relative_area=minimum_relative_area,
            border_margin_ratio=border_margin_ratio,
            framing_enabled=framing_enabled,
        ),
    )
    with redirect_stdout(diagnostic_stream or sys.stderr):
        result = pipeline.classify(image)
    payload = result.to_dict()
    payload["provenance"] = {
        "manifest": manifest_path.relative_to(root.resolve()).as_posix(),
        "runtime": runtime,
        "model_sha256": model_evidence["sha256"],
        "model_size_bytes": model_evidence["size_bytes"],
        "model_source": {
            "name": manifest["model"]["name"],
            "version": manifest["model"]["version"],
            "release_tag": manifest["model"]["release_tag"],
            "url": manifest["model"]["url"],
            "license": manifest["model"]["license"],
            "license_path": manifest["model"]["license_path"],
            "trust_constraint": manifest["model"]["trust_constraint"],
        },
        "fixture_sha256": image.snapshot_sha256,
        "fixture_size_bytes": image.size_bytes,
        "fixture_source": {
            "name": manifest["fixture"]["name"],
            "url": manifest["fixture"]["url"],
            "page_revision_url": manifest["fixture"]["page_revision_url"],
            "author": manifest["fixture"]["author"],
            "license": manifest["fixture"]["license"],
        },
        "thresholds": {
            "model_confidence": model_confidence,
            "pipeline_min_confidence": pipeline_confidence,
            "minimum_relative_area": minimum_relative_area,
            "border_margin_ratio": border_margin_ratio,
            "framing_enabled": framing_enabled,
            "input_size": input_size,
            "maximum_image_size_bytes": thresholds["maximum_image_size_bytes"],
            "minimum_width": thresholds["minimum_width"],
            "minimum_height": thresholds["minimum_height"],
            "maximum_width": thresholds["maximum_width"],
            "maximum_height": thresholds["maximum_height"],
            "maximum_pixels": thresholds["maximum_pixels"],
        },
    }
    output_path = _bounded_output_path(args.output, root)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ManifestError(f"Invalid command line: {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = _JsonArgumentParser(
        description="Run the verified CPU baseline through the VACCA adapter pipeline."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-confidence", type=float)
    parser.add_argument("--min-confidence", type=float)
    parser.add_argument("--minimum-relative-area", type=float)
    parser.add_argument("--border-margin-ratio", type=float)
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--disable-framing", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[[argparse.Namespace], dict[str, Any]] = run,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output_stream = stdout or sys.stdout
    error_stream = stderr or sys.stderr
    try:
        payload = runner(build_parser().parse_args(argv))
    except (
        ManifestError,
        RuntimeEnvironmentError,
        ImageValidationError,
        ImageValidationDependencyError,
        UltralyticsAdapterError,
        UltralyticsDependencyError,
        OSError,
    ) as error:
        print(json.dumps({"status": "ERROR", "error": str(error)}), file=error_stream)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True), file=output_stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

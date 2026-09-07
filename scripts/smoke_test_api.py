"""Run VACCA API lifespan/ASGI checks with optional live HTTP validation."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
import sys
from io import BytesIO
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fastapi import UploadFile
from starlette.datastructures import Headers
from starlette.requests import Request as StarletteRequest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vacca_api import main as api_main  # noqa: E402
from vacca_api.detection import DEFAULT_MODEL  # noqa: E402
from vacca_api.schemas import BCSReadinessResponse, BCSResponse, DetectResponse, HealthResponse  # noqa: E402
from vacca_bcs.serving_package import PRIVATE_PACKAGE_ID  # noqa: E402

logger = logging.getLogger(__name__)
LIVE_DETECT_INVALID_DETAIL = "File must be an image (JPEG or PNG)"


class SmokeCheckError(RuntimeError):
    """Raised when a smoke-test response or transport is not usable."""

def _validate_health(status_code: int, body: object) -> HealthResponse:
    if status_code != 200:
        raise SmokeCheckError(f"health returned HTTP {status_code}")
    health = HealthResponse.model_validate(body)
    if health.status != "ok":
        raise SmokeCheckError(f"health status is {health.status!r}")
    if health.model_loaded is not True:
        raise SmokeCheckError("health reports that the model is not loaded")
    return health


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the tracked deployment model and optionally run inference."
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional local JPG, JPEG, or PNG image for an inference check.",
    )
    parser.add_argument(
        "--base-url",
        help="Optional live API base URL, for example http://127.0.0.1:8001.",
    )
    parser.add_argument(
        "--check-detect",
        action="store_true",
        help="In live mode, exercise /detect with a controlled invalid upload when no image is given.",
    )
    parser.add_argument(
        "--check-bcs",
        action="store_true",
        help="Check lazy BCS readiness, a real fixture inference, category, and experimental status.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Live request timeout in seconds (default: 5).",
    )
    return parser


def run(
    image_path: Path | None = None,
    base_url: str | None = None,
    check_detect: bool = False,
    timeout: float = 5.0,
    check_bcs: bool = False,
) -> int:
    """Run either the safe in-process check or an optional live network check."""
    if base_url is not None:
        return _run_live(image_path, base_url, check_detect, check_bcs, timeout)

    return _run_in_process(image_path, check_bcs)


def _run_in_process(image_path: Path | None, check_bcs: bool = False) -> int:
    """Run the FastAPI lifecycle and request /health in-process."""
    if not DEFAULT_MODEL.is_file():
        logger.error("Tracked deployment model is missing: %s", DEFAULT_MODEL)
        return 1

    if check_bcs:
        image_path = ROOT / "fixtures" / "cow_female_black_white.jpg" if image_path is None else image_path
        return asyncio.run(_run_in_process_bcs(image_path))
    try:
        status_code, body = asyncio.run(_request_health())
        health = _validate_health(status_code, body)
    except Exception as exc:
        logger.error("Startup/health smoke check failed: %s", type(exc).__name__)
        return 1

    logger.info("Health: %s", health.model_dump_json())
    if image_path is None:
        logger.info("Startup and deployment model-load smoke check passed")
        return 0

    image_path = image_path.expanduser()
    if not image_path.is_file():
        logger.error("Smoke-test image is missing: %s", image_path)
        return 1

    try:
        image_bytes = image_path.read_bytes()
        detections = asyncio.run(_request_inference(image_path.name, image_bytes))
        response = DetectResponse.model_validate(detections)
    except Exception as exc:
        logger.error("Inference smoke check failed: %s", type(exc).__name__)
        return 1

    logger.info("Detection: %s", response.model_dump_json())
    logger.info("Startup, model-load, and inference smoke checks passed")
    return 0


async def _run_in_process_bcs(image_path: Path) -> int:
    if not image_path.expanduser().is_file():
        logger.error("BCS smoke image is missing: %s", image_path)
        return 1
    try:
        async with api_main.app.router.lifespan_context(api_main.app):
            status_code, body = await _asgi_request("GET", "/health")
            _validate_health(status_code, body)
            before = _readiness_payload(api_main.bcs_readiness())
            _validate_bcs_before(before)
            payload = image_path.read_bytes()
            upload = UploadFile(
                file=BytesIO(payload),
                filename=image_path.name,
                headers=Headers({"content-type": "image/jpeg"}),
            )
            response = await api_main.bcs(upload)
            _validate_bcs_response(response.model_dump())
            after = _readiness_payload(api_main.bcs_readiness())
            if after.status != "ready":
                raise SmokeCheckError("BCS readiness did not reach ready after inference")
    except Exception as exc:
        logger.error("In-process BCS smoke check failed: %s", type(exc).__name__)
        return 1
    logger.info("BCS readiness: not_loaded -> ready; category and experimental status passed")
    return 0


def _readiness_payload(response: object) -> BCSReadinessResponse:
    if isinstance(response, BCSReadinessResponse):
        return response
    status_code = getattr(response, "status_code", 200)
    if status_code != 200:
        raw = json.loads(response.body)
        return BCSReadinessResponse.model_validate(raw)
    return BCSReadinessResponse.model_validate(response)


def _validate_bcs_before(readiness: BCSReadinessResponse) -> None:
    if readiness.status == "not_installed":
        raise SmokeCheckError(
            "BCS package is not installed; obtain the private ZIP from the private VACCA Drive location or maintainer and run the installer"
        )
    if readiness.status != "not_loaded":
        raise SmokeCheckError(f"BCS readiness before inference was {readiness.status!r}")
    if readiness.model_status != "experimental_not_approved" or readiness.package_id != PRIVATE_PACKAGE_ID:
        raise SmokeCheckError("BCS smoke check requires the installed bundled experimental package")


def _validate_bcs_response(body: object) -> BCSResponse:
    response = BCSResponse.model_validate(body)
    if response.status != "ok" or response.bcs_category not in {1, 2, 3, 4, 5}:
        raise SmokeCheckError("BCS response category/status contract failed")
    if response.model_status != "experimental_not_approved":
        raise SmokeCheckError("BCS response did not identify the experimental package")
    if response.package_id != PRIVATE_PACKAGE_ID:
        raise SmokeCheckError("BCS response did not identify the bundled package")
    return response


async def _request_health() -> tuple[int, object]:
    async with api_main.app.router.lifespan_context(api_main.app):
        return await _asgi_request("GET", "/health")


async def _request_inference(filename: str, image_bytes: bytes) -> dict[str, object]:
    content_type = "image/png" if filename.casefold().endswith(".png") else "image/jpeg"
    upload = UploadFile(
        file=BytesIO(image_bytes),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )
    async with api_main.app.router.lifespan_context(api_main.app):
        response = await api_main.detect(
            StarletteRequest({"type": "http", "app": api_main.app}),
            upload,
        )
    return response.model_dump()


async def _asgi_request(
    method: str,
    path: str,
    *,
    body: bytes = b"",
    headers: dict[str, str] | None = None,
) -> tuple[int, object]:
    messages: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

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
            "headers": [
                (name.lower().encode("latin-1"), value.encode("latin-1"))
                for name, value in (headers or {}).items()
            ],
            "client": ("smoke-test", 0),
            "server": ("testserver", 80),
            "root_path": "",
        },
        receive,
        send,
    )
    status = next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    )
    body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return int(status), json.loads(body)


def _run_live(
    image_path: Path | None,
    base_url: str,
    check_detect: bool,
    check_bcs: bool,
    timeout: float,
) -> int:
    """Validate the actual listener, including its HTTP and multipart path."""

    if timeout <= 0:
        logger.error("Live smoke check failed: timeout must be positive")
        return 1
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        logger.error("Live smoke check failed: invalid base URL")
        return 1
    base_url = base_url.rstrip("/")

    try:
        status_code, body = _request_json(f"{base_url}/health", timeout=timeout)
        health = _validate_health(status_code, body)
    except Exception as exc:
        logger.error("Live health smoke check failed: %s", type(exc).__name__)
        return 1

    logger.info("Live health: %s", health.model_dump_json())
    explicit_image = image_path is not None
    if check_bcs:
        image_path = ROOT / "fixtures" / "cow_female_black_white.jpg" if image_path is None else image_path
        try:
            _, before_body = _request_json(f"{base_url}/ready/bcs", timeout=timeout)
            before = BCSReadinessResponse.model_validate(before_body)
            _validate_bcs_before(before)
            image_path = image_path.expanduser()
            if not image_path.is_file():
                raise SmokeCheckError("BCS smoke image is missing")
            bcs_body = _multipart_body(image_path.name, image_path.read_bytes(), "image/jpeg")
            status_code, response_body = _request_json(
                f"{base_url}/bcs",
                method="POST",
                body=bcs_body,
                headers={"Content-Type": f"multipart/form-data; boundary={_BOUNDARY.decode()}"},
                timeout=timeout,
            )
            if status_code != 200:
                raise SmokeCheckError(f"bcs returned HTTP {status_code}")
            _validate_bcs_response(response_body)
            after_status, after_body = _request_json(f"{base_url}/ready/bcs", timeout=timeout)
            after = BCSReadinessResponse.model_validate(after_body)
            if after_status != 200 or after.status != "ready":
                raise SmokeCheckError("live BCS readiness did not reach ready")
        except Exception as exc:
            logger.error("Live BCS smoke check failed: %s", type(exc).__name__)
            return 1
        logger.info("Live BCS readiness: not_loaded -> ready; category and experimental status passed")

    if check_bcs and not check_detect:
        return 0
    if image_path is None and not check_detect:
        logger.info("Live /health smoke check passed")
        return 0

    if image_path is not None and (not check_bcs or explicit_image):
        image_path = image_path.expanduser()
        if not image_path.is_file():
            logger.error("Smoke-test image is missing: %s", image_path)
            return 1
        try:
            image_bytes = image_path.read_bytes()
            content_type = (
                "image/png"
                if image_path.suffix.casefold() == ".png"
                else "image/jpeg"
            )
            body = _multipart_body(image_path.name, image_bytes, content_type)
            status_code, response_body = _request_json(
                f"{base_url}/detect",
                method="POST",
                body=body,
                headers={
                    "Content-Type": f"multipart/form-data; boundary={_BOUNDARY.decode()}"
                },
                timeout=timeout,
            )
            if status_code != 200:
                raise SmokeCheckError(f"detect returned HTTP {status_code}")
            response = DetectResponse.model_validate(response_body)
        except Exception as exc:
            logger.error("Live detection smoke check failed: %s", type(exc).__name__)
            return 1
        logger.info("Live detection: %s", response.model_dump_json())
    else:
        try:
            body = _multipart_body(
                "invalid.bin",
                b"not an image",
                "application/octet-stream",
            )
            status_code, response_body = _request_json(
                f"{base_url}/detect",
                method="POST",
                body=body,
                headers={
                    "Content-Type": f"multipart/form-data; boundary={_BOUNDARY.decode()}"
                },
                timeout=timeout,
            )
            if status_code != 400 or not isinstance(response_body, dict):
                raise SmokeCheckError("detect invalid-input contract failed")
            if response_body.get("detail") != LIVE_DETECT_INVALID_DETAIL:
                raise SmokeCheckError("detect invalid-input detail is malformed")
        except Exception as exc:
            logger.error("Live detection smoke check failed: %s", type(exc).__name__)
            return 1
        logger.info("Live /detect invalid-input contract passed")
    logger.info("Live health and backend-path smoke checks passed")
    return 0


_BOUNDARY = b"----vacca-vision-smoke-boundary"


def _multipart_body(filename: str, payload: bytes, content_type: str) -> bytes:
    """Build the single-file multipart payload without a requests dependency."""

    safe_filename = Path(filename).name.replace('"', "")
    return b"\r\n".join(
        (
            b"--" + _BOUNDARY,
            f'Content-Disposition: form-data; name="file"; filename="{safe_filename}"'.encode(),
            f"Content-Type: {content_type}".encode(),
            b"",
            payload,
            b"--" + _BOUNDARY + b"--",
            b"",
        )
    )


def _request_json(
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float,
) -> tuple[int, object]:
    request = Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            status_code = response.status
            raw_body = response.read()
    except HTTPError as exc:
        status_code = exc.code
        raw_body = exc.read()
    except (OSError, TimeoutError, URLError, socket.timeout):
        raise SmokeCheckError("live request failed") from None

    try:
        return status_code, json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SmokeCheckError("live response was not valid JSON") from None


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    if args.check_detect and args.base_url is None:
        build_parser().error("--check-detect requires --base-url")
    return run(
        args.image,
        args.base_url,
        args.check_detect,
        args.timeout,
        args.check_bcs,
    )


if __name__ == "__main__":
    sys.exit(main())

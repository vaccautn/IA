"""Run a local VACCA API startup/model-load smoke test without a server."""
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
from vacca_api.schemas import DetectResponse, HealthResponse  # noqa: E402

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
) -> int:
    """Run either the safe in-process check or an optional live network check."""
    if base_url is not None:
        return _run_live(image_path, base_url, check_detect, timeout)

    return _run_in_process(image_path)


def _run_in_process(image_path: Path | None) -> int:
    """Run the FastAPI lifecycle and request /health in-process."""
    if not DEFAULT_MODEL.is_file():
        logger.error("Tracked deployment model is missing: %s", DEFAULT_MODEL)
        return 1

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


async def _asgi_request(method: str, path: str) -> tuple[int, object]:
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
            "headers": [],
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
    if image_path is None and not check_detect:
        logger.info("Live /health smoke check passed")
        return 0

    if image_path is not None:
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
    return run(args.image, args.base_url, args.check_detect, args.timeout)


if __name__ == "__main__":
    sys.exit(main())

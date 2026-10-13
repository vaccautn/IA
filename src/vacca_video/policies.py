"""Video-only defaults; CLI previews and web previews intentionally differ."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_FPS = 5.0
DEFAULT_MAX_FRAMES = 300
CLI_PREVIEW_LIMIT = 120
WEB_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
WEB_DISK_RESERVE_BYTES = 50 * 1024 * 1024
WEB_MAX_ACTIVE_JOBS = 3
WORKER_TIMEOUT_SECONDS = 10 * 60
WORKER_POLL_SECONDS = 0.5
WORKER_TERMINATE_GRACE_SECONDS = 5
WORKER_SHUTDOWN_JOIN_SECONDS = 10


def require_single_server_process(workers: int) -> None:
    if workers > 1:
        raise SystemExit(
            f"VACCA video requires one principal server process; effective Uvicorn workers={workers}. "
            "Multiple workers are unsupported, including with --reload. "
            "Unset WEB_CONCURRENCY or set WEB_CONCURRENCY=1."
        )

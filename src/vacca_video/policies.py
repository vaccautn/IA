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
# Web tracking reads every frame (sampling caused ID switches in the test videos)
# of the window the sampled detection covers, capped by frames: ~150-400 ms per
# frame on a 4-core CPU keeps 1200 frames inside the timeout.
TRACKING_TIMEOUT_SECONDS = 10 * 60
TRACKING_MAX_SECONDS = DEFAULT_MAX_FRAMES / DEFAULT_SAMPLE_FPS
TRACKING_MAX_FRAMES = 1200
# Optional best-frame silhouettes; used only when these weights are already on disk.
SEGMENTATION_MODEL = ROOT / "models/segmentation/mobile_sam.pt"
TRACKING_TRACKER = "bytetrack.yaml"


def require_single_server_process(workers: int) -> None:
    if workers > 1:
        raise SystemExit(
            f"VACCA video requires one principal server process; effective Uvicorn workers={workers}. "
            "Multiple workers are unsupported, including with --reload. "
            "Unset WEB_CONCURRENCY or set WEB_CONCURRENCY=1."
        )

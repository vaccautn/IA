"""Optional per-track step of a web job, isolated in its own CLI subprocess.

It runs while the job is processing. Ordinary optional failures preserve valid
detections; the worker publishes completion only after the tracking outcome is
persisted in ``result/tracking-status.json``. Interruption fails the pipeline.
"""

import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .policies import (
    ROOT,
    SEGMENTATION_MODEL,
    TRACKING_MAX_FRAMES,
    TRACKING_MAX_SECONDS,
    TRACKING_TIMEOUT_SECONDS,
    TRACKING_TRACKER,
    WORKER_POLL_SECONDS,
    WORKER_TERMINATE_GRACE_SECONDS,
)

logger = logging.getLogger(__name__)
TRACKING_DIRECTORY = "tracking"
TRACKING_STATUS = "tracking-status.json"
TRACKING_ENV = "VACCA_VIDEO_TRACKING"


def write_status(result, status, reason=None):
    payload = {"status": status} if reason is None else {"status": status, "reason": reason}
    fd, temporary = tempfile.mkstemp(prefix=".tracking-", dir=result)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload) + "\n")
        os.replace(temporary, Path(result) / TRACKING_STATUS)
    finally:
        Path(temporary).unlink(missing_ok=True)


def tracking_command(directory):
    segmentation = ["--segment-model", str(SEGMENTATION_MODEL)] if SEGMENTATION_MODEL.is_file() else []
    return [
        sys.executable,
        "-u",
        str(ROOT / "scripts/track_video.py"),
        str(directory / "input.mp4"),
        "--output",
        str(directory / "result" / TRACKING_DIRECTORY),
        "--tracker",
        TRACKING_TRACKER,
        "--sample-fps",
        "0",
        "--clip-format",
        "webm",
        "--max-seconds",
        str(TRACKING_MAX_SECONDS),
        "--max-frames",
        str(TRACKING_MAX_FRAMES),
        *segmentation,
    ]


def _run(directory, stop):
    if os.environ.get(TRACKING_ENV, "1") == "0":
        return "disabled", "disabled_by_configuration"
    if stop.is_set():
        return "failed", "interrupted"
    with (directory / "tracking.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            tracking_command(directory),
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        ) as proc:
            polls = math.ceil(TRACKING_TIMEOUT_SECONDS / WORKER_POLL_SECONDS)
            for _ in range(polls):
                if proc.poll() is not None:
                    break
                if stop.wait(WORKER_POLL_SECONDS):
                    break
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=WORKER_TERMINATE_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                return "failed", "interrupted" if stop.is_set() else "timeout"
            if proc.returncode:
                return "failed", "process_failed"
    if not (directory / "result" / TRACKING_DIRECTORY / "summary.json").is_file():
        return "failed", "summary_missing"
    return "completed", None


def run_tracking(directory, stop):
    """Run tracking for one job directory; never raises into the worker."""
    directory = Path(directory)
    try:
        write_status(directory / "result", "running")
        status, reason = _run(directory, stop)
    except Exception as exc:
        status, reason = "failed", "interrupted" if stop.is_set() else "process_failed"
        logger.error("Video tracking failed: %s", type(exc).__name__)
    try:
        write_status(directory / "result", status, reason)
    except OSError as exc:
        logger.error("Video tracking status not saved: %s", type(exc).__name__)
    return status


def recover_interrupted(root):
    """At startup no tracking can be running: settle runs left by a stopped service."""
    recovered = 0
    for directory in Path(root).iterdir():
        status_path = directory / "result" / TRACKING_STATUS
        if not re.fullmatch(r"[A-Za-z0-9_-]+", directory.name) or status_path.is_symlink():
            continue
        try:
            recorded = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            continue
        if type(recorded) is dict and recorded.get("status") == "running":
            write_status(directory / "result", "failed", "interrupted")
            recovered += 1
    return recovered

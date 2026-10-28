"""Fixed job-scoped artifacts and atomic summary publication (no DB migration)."""

import json
import os
from pathlib import Path
import re
import tempfile

from .candidate import CANDIDATE_ARTIFACT

BCS_WRITE_ERROR = "No se pudo guardar el resultado BCS; las detecciones del video se conservaron."


def job_artifact(root, job_id, name):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise ValueError("Invalid job identifier")
    if name not in {CANDIDATE_ARTIFACT, "summary.json", "summary.js"}:
        raise ValueError("Invalid artifact name")
    root = Path(root)
    path = root / job_id / "result" / name
    for part in (root, root / job_id, path.parent, path):
        if part.is_symlink() or part.is_junction():
            raise ValueError("Symbolic artifact paths are not allowed")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact is outside job storage")
    return path


def frame_artifact(root, job_id, index):
    """Resolve one fixed sampled JPEG without following linked ancestors."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id) or type(index) is not int or not 0 <= index < 300:
        raise ValueError("Invalid frame path")
    root = Path(root)
    directory = root / job_id / "result" / "frames"
    path = directory / f"{index:06d}.jpg"
    for part in (root, root / job_id, root / job_id / "result", directory, path):
        if part.is_symlink() or part.is_junction():
            raise ValueError("Symbolic frame paths are not allowed")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Frame is outside job storage")
    return path


def read_summary(root, job_id):
    with job_artifact(root, job_id, "summary.json").open("rb") as stream:
        data = stream.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError("Summary is too large")
    return json.loads(data)


def publish_initial_summary(directory, summary):
    """Publish the required JSON/JS summary pair with rollback on write failure.

    The two replacements are not a filesystem transaction. If a replacement
    fails in-process, restore the prior pair (or remove newly-created files).
    """
    payload = json.dumps(summary, ensure_ascii=True, allow_nan=False)
    targets = (
        ("summary.json", payload + "\n"),
        ("summary.js", "window.VACCA_SUMMARY = " + payload + ";\n"),
    )
    directory = Path(directory)
    staged = []
    backups = {}
    replaced = []
    try:
        for name, content in targets:
            target = directory / name
            fd, temporary = tempfile.mkstemp(prefix=".summary-", dir=directory)
            staged.append((Path(temporary), target))
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(content)
        for name, _content in targets:
            target = directory / name
            if target.exists():
                fd, temporary = tempfile.mkstemp(prefix=".summary-backup-", dir=directory)
                backup = Path(temporary)
                backups[target] = backup
                with os.fdopen(fd, "wb") as stream:
                    stream.write(target.read_bytes())
        try:
            for temporary, target in staged:
                os.replace(temporary, target)
                replaced.append(target)
        except OSError:
            for target in reversed(replaced):
                backup = backups.pop(target, None)
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    os.replace(backup, target)
            raise
    finally:
        for temporary, _target in staged:
            temporary.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def publish_summary(root, job_id, summary):
    payload = json.dumps(summary, ensure_ascii=True, allow_nan=False)
    targets = [("summary.json", payload + "\n")]
    if job_artifact(root, job_id, "summary.js").exists():
        targets.insert(0, ("summary.js", "window.VACCA_SUMMARY = " + payload + ";\n"))
    staged = []
    backup = None
    js_replaced = False
    try:
        for name, content in targets:
            path = job_artifact(root, job_id, name)
            fd, temporary = tempfile.mkstemp(prefix=".bcs-", dir=path.parent)
            staged.append((Path(temporary), path))
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(content)
        if targets[0][0] == "summary.js":
            js_path = job_artifact(root, job_id, "summary.js")
            fd, temporary = tempfile.mkstemp(prefix=".bcs-", dir=js_path.parent)
            backup = Path(temporary)
            with os.fdopen(fd, "wb") as stream:
                stream.write(js_path.read_bytes())
        for temporary, path in staged:
            os.replace(temporary, path)
            js_replaced |= path.name == "summary.js"
    except OSError:
        if js_replaced and backup is not None:
            # Preserve the previous pair if publishing the authoritative JSON
            # fails. A rollback failure is still surfaced as unsaved BCS.
            os.replace(backup, job_artifact(root, job_id, "summary.js"))
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        if backup is not None:
            backup.unlink(missing_ok=True)


TRACK_MEDIA_TYPES = {
    "clip.webm": "video/webm",
    "clip.mp4": "video/mp4",
    "best-crop.png": "image/png",
    "best-original.jpg": "image/jpeg",
    "best-masked.png": "image/png",
    "best-mask.png": "image/png",
}
TRACKING_JSON_LIMIT = 4 * 1024 * 1024


def tracking_artifact(root, job_id, name, track_id=None):
    """Resolve a fixed tracking file of one job without following links."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise ValueError("Invalid job identifier")
    root = Path(root)
    result = root / job_id / "result"
    chain = [root, root / job_id, result]
    if track_id is None:
        if name == "tracking-status.json":
            path = result / name
        elif name == "summary.json":
            chain.append(result / "tracking")
            path = result / "tracking" / name
        else:
            raise ValueError("Invalid tracking artifact")
    else:
        if type(track_id) is not int or track_id < 0 or name not in TRACK_MEDIA_TYPES:
            raise ValueError("Invalid track artifact")
        tracks = result / "tracking" / "tracks"
        chain += [result / "tracking", tracks, tracks / str(track_id)]
        path = tracks / str(track_id) / name
    for part in chain + [path]:
        if part.is_symlink() or part.is_junction():
            raise ValueError("Symbolic tracking paths are not allowed")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Tracking artifact is outside job storage")
    return path


def read_tracking_json(root, job_id, name):
    with tracking_artifact(root, job_id, name).open("rb") as stream:
        data = stream.read(TRACKING_JSON_LIMIT + 1)
    if len(data) > TRACKING_JSON_LIMIT:
        raise ValueError("Tracking metadata is too large")
    return json.loads(data)

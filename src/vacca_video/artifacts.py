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


def read_summary(root, job_id):
    with job_artifact(root, job_id, "summary.json").open("rb") as stream:
        data = stream.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError("Summary is too large")
    return json.loads(data)


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

"""Turn tracking runs into a BCS labelling sheet and a category folder dataset.

The current BCS model was trained on whole images and failed its acceptance
controls; rear-view crops per animal are the missing training material. This
module never assigns a category: a person fills ``bcs_category`` in the sheet.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

EXPORT_SCHEMA = "vacca-bcs-track-crops-v1"
LABELS_FILE = "labels.csv"
GROUPS_FILE = "groups.json"
CATEGORIES = ("1", "2", "3", "4", "5")
COLUMNS = (
    "image", "image_kind", "group", "video", "video_sha256", "tracker", "track_id",
    "frame_index", "timestamp_ms", "track_status", "warnings", "mask_status",
    "bcs_category", "notes",
)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def export_crops(runs, output, include_review: bool = False) -> dict:
    """Copy one image per exported track and write an empty-label sheet.

    Each video may appear once: two trackers over the same video would export
    the same animals twice and leak them across train/validation splits.
    """
    output = Path(output)
    if output.exists():
        raise ValueError(f"La carpeta de salida ya existe: {output}")
    loaded, seen = [], {}
    for run in map(Path, runs):
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        video = Path(summary["input"])
        digest = _sha256(video) if video.is_file() else None
        key = digest or str(video)
        if key in seen:
            raise ValueError(f"El video de {run} ya se exportó desde {seen[key]}; usar un solo tracker por video")
        seen[key] = run
        loaded.append((run, summary, video, digest))
    images = output / "images"
    images.mkdir(parents=True)
    rows = []
    for run, summary, video, digest in loaded:
        video_id = digest[:12] if digest else run.name
        tracker = Path(summary.get("tracker", "tracker")).stem
        groups = {}
        for track in sorted(summary["tracks"], key=lambda item: item["track_id"]):
            track_id = track["track_id"]
            # A linked fragment is probably the same animal: keep it in one split group.
            origin = track.get("possible_continuation_of")
            groups[track_id] = groups.get(origin, f"{video_id}-t{track_id}")
            if track["status"] != "candidate" and not include_review:
                continue
            segmentation = track.get("segmentation") or {}
            masked = segmentation.get("status") == "ok" and segmentation.get("best_masked")
            source = run / (segmentation["best_masked"] if masked else track["best_crop"])
            name = f"{video_id}-{tracker}-t{track_id}.png"
            shutil.copyfile(source, images / name)
            rows.append({
                "image": name, "image_kind": "masked" if masked else "rect",
                "group": groups[track_id], "video": video.name, "video_sha256": digest or "",
                "tracker": tracker, "track_id": track_id, "frame_index": track["best_frame_index"],
                "timestamp_ms": round(track["best_timestamp_ms"], 3), "track_status": track["status"],
                "warnings": ";".join(track["warnings"]), "mask_status": segmentation.get("status", ""),
                "bcs_category": "", "notes": "",
            })
    with (output / LABELS_FILE).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    result = {"schema_version": EXPORT_SCHEMA, "runs": [str(run) for run, *_ in loaded],
              "include_review": include_review, "images": len(rows),
              "masked": sum(row["image_kind"] == "masked" for row in rows)}
    (output / "export.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def materialize_dataset(labels, output) -> dict:
    """Copy labelled crops into ``output/<1..5>/`` for ``BCSFolderDataset``.

    Rows without a category are skipped; any other value aborts before copying.
    ``groups.json`` lets split builders keep one animal out of two splits.
    """
    labels, output = Path(labels), Path(output)
    if output.exists():
        raise ValueError(f"La carpeta de salida ya existe: {output}")
    with labels.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    invalid = [f"fila {number}: {row.get('bcs_category')!r}" for number, row in enumerate(rows, start=2)
               if row.get("bcs_category", "").strip() not in ("", *CATEGORIES)]
    if invalid:
        raise ValueError("Categorías BCS inválidas (usar 1..5 o dejar vacío): " + ", ".join(invalid))
    images = labels.parent / "images"
    labelled = [row for row in rows if row["bcs_category"].strip()]
    for row in labelled:
        name = Path(row["image"]).name
        if name != row["image"] or not (images / name).is_file():
            raise ValueError(f"Imagen inválida o faltante en la planilla: {row['image']!r}")
    for category in CATEGORIES:
        (output / category).mkdir(parents=True)
    for row in labelled:
        shutil.copyfile(images / row["image"], output / row["bcs_category"].strip() / row["image"])
    groups = {row["image"]: row["group"] for row in labelled}
    (output / GROUPS_FILE).write_text(json.dumps(groups, indent=2, sort_keys=True), encoding="utf-8")
    counts = {category: sum(row["bcs_category"].strip() == category for row in labelled)
              for category in CATEGORIES}
    return {"labelled": len(labelled), "skipped": len(rows) - len(labelled), "counts": counts,
            "groups": len(set(groups.values()))}

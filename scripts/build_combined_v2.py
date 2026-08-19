"""Build combined-v2 dataset using ONLY unused BCS images + all Navid.

Scans data/combined/ to find which BCS images were used in v1,
then takes fresh images from the BCS pool.
"""
from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BCS_DIR = ROOT / "data" / "bcs" / "dataset"
NAVID_DIR = ROOT / "data" / "cow-detection-navids"
COMBINED_V1 = ROOT / "data" / "combined"
OUT_DIR = ROOT / "data" / "combined-v2"

MAX_PER_CLASS = 3000
VAL_RATIO = 0.2
SEED = 123


def xml_to_yolo(xml_path: Path, img_w: int = 1024, img_h: int = 576) -> list[str]:
    tree = ET.parse(xml_path)
    root_elem = tree.getroot()
    size = root_elem.find("size")
    if size is not None:
        img_w = int(size.find("width").text)  # type: ignore[union-attr]
        img_h = int(size.find("height").text)  # type: ignore[union-attr]

    lines: list[str] = []
    for obj in root_elem.findall("object"):
        name = obj.find("name").text  # type: ignore[union-attr]
        # Map all BCS scores -> class 0 (cow)
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)  # type: ignore[union-attr]
        ymin = float(bbox.find("ymin").text)  # type: ignore[union-attr]
        xmax = float(bbox.find("xmax").text)  # type: ignore[union-attr]
        ymax = float(bbox.find("ymax").text)  # type: ignore[union-attr]

        x_c = (xmin + xmax) / 2 / img_w
        y_c = (ymin + ymax) / 2 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        lines.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    return lines


def collect_used_names(combined_dir: Path) -> set[str]:
    """Return set of jpg filenames already used in combined v1."""
    used: set[str] = set()
    for split in ["train", "val"]:
        img_dir = combined_dir / split / "images"
        if img_dir.is_dir():
            for jpg in img_dir.glob("*.jpg"):
                used.add(jpg.name)
    return used


def merge_navid(dst_train_img: Path, dst_train_lbl: Path,
                dst_val_img: Path, dst_val_lbl: Path) -> tuple[int, int]:
    """Copy all Navid images into combined-v2. Returns (train_count, val_count)."""
    def _copy_split(src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path) -> int:
        count = 0
        for jpg in src_img.glob("*.jpg"):
            txt = src_lbl / f"{jpg.stem}.txt"
            dj = dst_img / jpg.name
            dt = dst_lbl / f"{jpg.stem}.txt"
            if not dj.exists():
                shutil.copy2(jpg, dj)
            if txt.is_file() and not dt.exists():
                shutil.copy2(txt, dt)
            count += 1
        return count

    n_train = _copy_split(NAVID_DIR / "train" / "images",
                          NAVID_DIR / "train" / "labels",
                          dst_train_img, dst_train_lbl)
    n_val = _copy_split(NAVID_DIR / "valid" / "images",
                        NAVID_DIR / "valid" / "labels",
                        dst_val_img, dst_val_lbl)
    return n_train, n_val


def main() -> None:
    random.seed(SEED)

    # Clean output
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for d in ["train/images", "train/labels", "val/images", "val/labels"]:
        (OUT_DIR / d).mkdir(parents=True)

    train_img = OUT_DIR / "train" / "images"
    train_lbl = OUT_DIR / "train" / "labels"
    val_img = OUT_DIR / "val" / "images"
    val_lbl = OUT_DIR / "val" / "labels"

    used = collect_used_names(COMBINED_V1)
    print(f"Images already used in v1: {len(used)}")

    total_bcs = 0
    for subset in sorted(BCS_DIR.iterdir()):
        if not subset.is_dir():
            continue

        # Collect unused pairs from this subset
        pairs: list[tuple[Path, Path]] = []
        for jpg in sorted(subset.glob("*.jpg")):
            if jpg.name in used:
                continue
            xml = subset / f"{jpg.stem}.xml"
            if xml.is_file():
                pairs.append((jpg, xml))

        available = len(pairs)
        n_take = min(MAX_PER_CLASS, available)
        random.shuffle(pairs)
        pairs = pairs[:n_take]
        n_val = max(1, int(n_take * VAL_RATIO))

        train_pairs = pairs[n_val:]
        val_pairs = pairs[:n_val]

        for jpg, xml in train_pairs:
            yolo_lines = xml_to_yolo(xml)
            if not yolo_lines:
                continue
            dj = train_img / jpg.name
            dt = train_lbl / f"{jpg.stem}.txt"
            if not dj.exists():
                shutil.copy2(jpg, dj)
            dt.write_text("\n".join(yolo_lines), encoding="utf-8")

        for jpg, xml in val_pairs:
            yolo_lines = xml_to_yolo(xml)
            if not yolo_lines:
                continue
            dj = val_img / jpg.name
            dt = val_lbl / f"{jpg.stem}.txt"
            if not dj.exists():
                shutil.copy2(jpg, dj)
            dt.write_text("\n".join(yolo_lines), encoding="utf-8")

        print(f"  {subset.name}: {available} unused -> took {n_take} "
              f"({len(train_pairs)} train, {len(val_pairs)} val)")
        total_bcs += n_take

    # Merge Navid
    n_train_navid, n_val_navid = merge_navid(train_img, train_lbl, val_img, val_lbl)
    print(f"\nNavid: {n_train_navid} train, {n_val_navid} val")

    total_train = len(list(train_img.glob("*.jpg")))
    total_val = len(list(val_img.glob("*.jpg")))

    # data.yaml
    (OUT_DIR / "data.yaml").write_text(
        f"# VACCA Fase 1 — Combined v2 (unused BCS + Navid)\n"
        f"path: {OUT_DIR.as_posix()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: 1\n"
        f"names: ['cow']\n"
        f"# BCS (unused): {total_bcs} imgs | Navid: {n_train_navid + n_val_navid} imgs\n"
    )

    print(f"\n[DONE] combined-v2: {total_train} train, {total_val} val "
          f"({total_train + total_val} total)")


if __name__ == "__main__":
    main()

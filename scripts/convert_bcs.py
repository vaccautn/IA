"""Convert BCS XML annotations to YOLO format and merge with Navid dataset.

BCS dataset structure:
    data/bcs/dataset/{3.25,3.5,3.75,4.0,4.25}/*.jpg + *.xml
    XML: Pascal VOC format, <name> = BCS score (3.25, 3.5, 3.75, 4.0, 4.25)

Output:
    data/combined/
        train/images/  train/labels/
        val/images/    val/labels/
        data.yaml

Strategy:
    - Map all BCS classes -> class 0 ("cow")
    - Take max_per_class images from each BCS subset to keep total manageable
    - 80/20 train/val split per class
    - Merge Navid's existing train/val into combined
"""
from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BCS_DIR = ROOT / "data" / "bcs" / "dataset"
NAVID_DIR = ROOT / "data" / "cow-detection-navids"
OUT_DIR = ROOT / "data" / "combined"

# BCS class names in XML -> all mapped to cow (class 0)
BCS_CLASS_MAP = {"3.25": 0, "3.5": 0, "3.75": 0, "4.0": 0, "4.25": 0}


def xml_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """Convert Pascal VOC XML to YOLO txt lines. Returns list of lines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is not None:
        img_w = int(size.find("width").text)  # type: ignore[union-attr]
        img_h = int(size.find("height").text)  # type: ignore[union-attr]

    lines: list[str] = []
    for obj in root.findall("object"):
        name = obj.find("name").text  # type: ignore[union-attr]
        if name not in BCS_CLASS_MAP:
            continue
        class_id = BCS_CLASS_MAP[name]
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)  # type: ignore[union-attr]
        ymin = float(bbox.find("ymin").text)  # type: ignore[union-attr]
        xmax = float(bbox.find("xmax").text)  # type: ignore[union-attr]
        ymax = float(bbox.find("ymax").text)  # type: ignore[union-attr]

        # YOLO: class x_center y_center width height (normalized)
        x_c = (xmin + xmax) / 2 / img_w
        y_c = (ymin + ymax) / 2 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    return lines


def collect_bcs_pairs(subset_dir: Path) -> list[tuple[Path, Path]]:
    """Return list of (jpg_path, xml_path) pairs in a BCS subset directory."""
    pairs: list[tuple[Path, Path]] = []
    for jpg in sorted(subset_dir.glob("*.jpg")):
        xml = subset_dir / f"{jpg.stem}.xml"
        if xml.is_file():
            pairs.append((jpg, xml))
    return pairs


def prepare_bcs_split(
    pairs: list[tuple[Path, Path]],
    train_img: Path,
    train_lbl: Path,
    val_img: Path,
    val_lbl: Path,
    max_per_class: int,
    val_ratio: float = 0.2,
) -> int:
    """Split BCS pairs into train/val, convert XML to YOLO, copy images.
    Returns number of images used.
    """
    random.shuffle(pairs)
    pairs = pairs[:max_per_class]

    n_val = max(1, int(len(pairs) * val_ratio))
    train_pairs = pairs[n_val:]
    val_pairs = pairs[:n_val]

    for jpg, xml in train_pairs:
        yolo_lines = xml_to_yolo(xml, 1024, 576)
        if not yolo_lines:
            continue
        dst_jpg = train_img / jpg.name
        dst_lbl = train_lbl / f"{jpg.stem}.txt"
        # Copy avoids file-in-use issues with shutil
        if not dst_jpg.exists():
            shutil.copy2(jpg, dst_jpg)
        dst_lbl.write_text("\n".join(yolo_lines), encoding="utf-8")

    for jpg, xml in val_pairs:
        yolo_lines = xml_to_yolo(xml, 1024, 576)
        if not yolo_lines:
            continue
        dst_jpg = val_img / jpg.name
        dst_lbl = val_lbl / f"{jpg.stem}.txt"
        if not dst_jpg.exists():
            shutil.copy2(jpg, dst_jpg)
        dst_lbl.write_text("\n".join(yolo_lines), encoding="utf-8")

    print(
        f"  BCS subset {xml.parent.name}: "
        f"{len(pairs)} imgs ({len(train_pairs)} train, {len(val_pairs)} val)"
    )
    return len(pairs)


def merge_navid_split(
    src_img: Path,
    src_lbl: Path,
    dst_img: Path,
    dst_lbl: Path,
) -> int:
    """Merge Navid images and labels into combined directory. Returns count."""
    count = 0
    for jpg in src_img.glob("*.jpg"):
        txt = src_lbl / f"{jpg.stem}.txt"
        dst_jpg = dst_img / jpg.name
        dst_txt = dst_lbl / f"{jpg.stem}.txt"
        if not dst_jpg.exists():
            shutil.copy2(jpg, dst_jpg)
        if txt.is_file() and not dst_txt.exists():
            shutil.copy2(txt, dst_txt)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BCS XML to YOLO + merge with Navid")
    parser.add_argument("--max-per-class", type=int, default=1000,
                        help="Max images per BCS class (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bcs-dir", default=str(BCS_DIR),
                        help="BCS dataset root")
    parser.add_argument("--navid-dir", default=str(NAVID_DIR),
                        help="Navid dataset root")
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                        help="Combined output directory")
    args = parser.parse_args()

    random.seed(args.seed)

    bcs_dir = Path(args.bcs_dir)
    navid_dir = Path(args.navid_dir)
    out_dir = Path(args.out_dir)

    # Clean and create output structure
    if out_dir.exists():
        shutil.rmtree(out_dir)
    train_img = out_dir / "train" / "images"
    train_lbl = out_dir / "train" / "labels"
    val_img = out_dir / "val" / "images"
    val_lbl = out_dir / "val" / "labels"
    for d in [train_img, train_lbl, val_img, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"BCS conversion: max {args.max_per_class} img/class, seed={args.seed}")

    # Process each BCS subset
    total_bcs = 0
    for subset in sorted(bcs_dir.iterdir()):
        if not subset.is_dir():
            continue
        pairs = collect_bcs_pairs(subset)
        print(f"  Found {len(pairs)} images in {subset.name}")
        used = prepare_bcs_split(pairs, train_img, train_lbl, val_img, val_lbl,
                                 max_per_class=args.max_per_class)
        total_bcs += used

    # Merge Navid
    print("\nMerging Navid dataset...")
    navid_train_img = navid_dir / "train" / "images"
    navid_train_lbl = navid_dir / "train" / "labels"
    navid_val_img = navid_dir / "valid" / "images"
    navid_val_lbl = navid_dir / "valid" / "labels"

    n_navid_train = merge_navid_split(navid_train_img, navid_train_lbl, train_img, train_lbl)
    n_navid_val = merge_navid_split(navid_val_img, navid_val_lbl, val_img, val_lbl)

    print(f"  Navid train: {n_navid_train} images")
    print(f"  Navid val: {n_navid_val} images")

    # Write data.yaml
    yaml_content = f"""# VACCA Fase 1 — Combined dataset: Navid HSM + BCS (subset)
# Generated by scripts/convert_bcs.py
path: {out_dir.as_posix()}
train: train/images
val: val/images

nc: 1
names: ['cow']

# Sources:
#   Navid HSM Cow Detection (CC BY 4.0) — {n_navid_train + n_navid_val} images
#   BCS-YOLO ScienceDB 10.57760/sciencedb.16704 (CC BY 4.0) — {total_bcs} images
"""

    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print(f"\n[LISTO] Combined dataset created at: {out_dir}")
    print(f"  Total images: {total_bcs + n_navid_train + n_navid_val}")
    print(f"  Train: {len(list(train_img.glob('*.jpg')))}")
    print(f"  Val: {len(list(val_img.glob('*.jpg')))}")
    print(f"  data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()

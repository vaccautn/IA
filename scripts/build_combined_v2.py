"""Build combined-v2 dataset using ONLY unused BCS images + all Navid.

Scans data/combined/ to find which BCS images were used in v1,
then takes fresh images from the BCS pool.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
import random
import re
import shutil
import stat
import tempfile
import uuid
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
JOURNAL_VERSION = 1
JOURNAL_PHASES = {
    "prepared", "backed_up", "published", "restoring", "restored", "committed"
}


def xml_to_yolo(xml_path: Path, img_w: int = 1024, img_h: int = 576) -> list[str]:
    tree = ET.parse(xml_path)
    root_elem = tree.getroot()
    size = root_elem.find("size")
    if size is not None:
        img_w = int(size.find("width").text)  # type: ignore[union-attr]
        img_h = int(size.find("height").text)  # type: ignore[union-attr]

    lines: list[str] = []
    for obj in root_elem.findall("object"):
        # Map all BCS source labels -> class 0 (cow)
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
                dst_val_img: Path, dst_val_lbl: Path,
                navid_dir: Path = NAVID_DIR) -> tuple[int, int]:
    """Copy all Navid images into combined-v2. Returns (train_count, val_count)."""
    def _copy_split(src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path) -> int:
        count = 0
        for jpg in src_img.glob("*.jpg"):
            txt = src_lbl / f"{jpg.stem}.txt"
            dj = dst_img / jpg.name
            dt = dst_lbl / f"{jpg.stem}.txt"
            if not dj.exists():
                _copy2_and_sync(jpg, dj)
            if txt.is_file() and not dt.exists():
                _copy2_and_sync(txt, dt)
            count += 1
        return count

    n_train = _copy_split(navid_dir / "train" / "images",
                          navid_dir / "train" / "labels",
                          dst_train_img, dst_train_lbl)
    n_val = _copy_split(navid_dir / "valid" / "images",
                        navid_dir / "valid" / "labels",
                        dst_val_img, dst_val_lbl)
    return n_train, n_val


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def _path_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _normalize_output_dir(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    if output_dir.name in {"", ".", ".."}:
        raise ValueError("output directory must name a sibling generation")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.parent.resolve() / output_dir.name
    if _is_reparse_point(output_dir):
        raise ValueError("published output cannot be a link or junction")
    if _path_exists(output_dir) and not output_dir.is_dir():
        raise ValueError("published output must be a directory")
    return output_dir


def _journal_path(output_dir: Path) -> Path:
    return output_dir.parent / f".{output_dir.name}.publish-journal.json"


def _generation_path(output_dir: Path, role: str, token: str) -> Path:
    if role not in {"stage", "backup"} or re.fullmatch(r"[0-9a-f]{32}", token) is None:
        raise ValueError("publication journal contains an invalid generation identity")
    return output_dir.parent / f".{output_dir.name}.{role}-{token}"


def _fsync_directory(directory: Path) -> None:
    """Sync directory entries where supported; Windows has no portable directory fsync."""
    if os.name == "nt":
        # Journal and generated files are flushed individually. Windows directory-entry
        # durability across power loss cannot be guaranteed through Python's portable API.
        return
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _single_writer_lock(output_dir: Path):
    """Use an OS-released advisory lock so crashed writers do not block recovery."""
    lock_path = output_dir.parent / f".{output_dir.name}.writer.lock"
    if _is_reparse_point(lock_path):
        raise RuntimeError("publication lock cannot be a link or junction")
    stream = lock_path.open("a+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise RuntimeError("another combined-v2 build is active") from None
            locked = True
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise RuntimeError("another combined-v2 build is active") from None
            locked = True
        yield
    finally:
        if locked:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _write_journal(output_dir: Path, journal: dict[str, object]) -> None:
    journal_path = _journal_path(output_dir)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_dir.name}.journal-", suffix=".tmp", dir=output_dir.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(journal, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, journal_path)
        _fsync_directory(output_dir.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_journal(output_dir: Path) -> dict[str, object]:
    path = _journal_path(output_dir)
    if _is_reparse_point(path):
        raise RuntimeError("publication journal cannot be a link or junction")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise RuntimeError("publication journal is unreadable; recovery stopped safely") from None
    expected_keys = {"version", "token", "had_output", "phase"}
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_keys
        or type(payload["version"]) is not int
        or payload["version"] != JOURNAL_VERSION
        or type(payload["had_output"]) is not bool
        or not isinstance(payload["phase"], str)
        or payload["phase"] not in JOURNAL_PHASES
        or not isinstance(payload["token"], str)
        or re.fullmatch(r"[0-9a-f]{32}", payload["token"]) is None
    ):
        raise RuntimeError("publication journal is malformed; recovery stopped safely")
    return payload


def _remove_generation(path: Path) -> None:
    if not _path_exists(path):
        return
    if _is_reparse_point(path):
        raise RuntimeError("refusing to remove a linked publication generation")
    if path.is_dir():
        if any(_is_reparse_point(child) for child in path.rglob("*")):
            raise RuntimeError("refusing to remove a generation containing links or junctions")

        def retry_readonly_delete(function, failed_path, error):
            if not isinstance(error, PermissionError):
                raise error
            target = Path(failed_path)
            if _is_reparse_point(target):
                raise error
            try:
                target.relative_to(path)
                mode = target.stat().st_mode
            except (OSError, ValueError):
                raise error
            writable_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
            if mode & writable_bits:
                raise error
            target.chmod(mode | stat.S_IWRITE)
            function(failed_path)

        shutil.rmtree(path, onexc=retry_readonly_delete)
    else:
        path.unlink()
    _fsync_directory(path.parent)


def _rename_sibling(source: Path, destination: Path) -> None:
    if source.parent != destination.parent:
        raise ValueError("publication paths must be same-volume siblings")
    if _is_reparse_point(source) or _path_exists(destination):
        raise RuntimeError("publication rename encountered an unsafe path")
    os.rename(source, destination)
    _fsync_directory(source.parent)


def _verify_generation(generation: Path, final_output: Path) -> None:
    if _is_reparse_point(generation) or not generation.is_dir():
        raise RuntimeError("combined-v2 generation root is unavailable")
    required_directories = (
        "train/images", "train/labels", "val/images", "val/labels"
    )
    for relative in required_directories:
        directory = generation
        for part in Path(relative).parts:
            directory /= part
            if _is_reparse_point(directory) or not directory.is_dir():
                raise RuntimeError("combined-v2 generation is missing a required split directory")
    data_yaml = generation / "data.yaml"
    if _is_reparse_point(data_yaml) or not data_yaml.is_file():
        raise RuntimeError("combined-v2 generation is missing data.yaml")
    expected_root = f"path: {final_output.as_posix()}"
    if expected_root not in data_yaml.read_text(encoding="utf-8").splitlines():
        raise RuntimeError("combined-v2 data.yaml does not reference the published directory")


def _rollback_uncommitted(output_dir: Path, journal: dict[str, object]) -> None:
    token = journal["token"]
    stage = _generation_path(output_dir, "stage", token)
    backup = _generation_path(output_dir, "backup", token)
    output_exists = _path_exists(output_dir)
    stage_exists = _path_exists(stage)
    backup_exists = _path_exists(backup)
    had_output = journal["had_output"]
    phase = journal["phase"]

    for path in (stage, backup):
        if _is_reparse_point(path):
            raise RuntimeError("publication recovery encountered a linked generation")

    if had_output:
        if backup_exists:
            if output_exists:
                if stage_exists:
                    raise RuntimeError("publication recovery found an ambiguous generation state")
                _rename_sibling(output_dir, stage)
            journal["phase"] = "restoring"
            _write_journal(output_dir, journal)
            _rename_sibling(backup, output_dir)
            journal["phase"] = "restored"
            _write_journal(output_dir, journal)
            _remove_generation(stage)
        elif output_exists and phase == "prepared":
            # The previous generation was never moved aside.
            journal["phase"] = "restored"
            _write_journal(output_dir, journal)
            _remove_generation(stage)
        elif output_exists and phase in {"restoring", "restored"}:
            # A prior recovery may have restored the backup before interruption.
            if phase != "restored":
                journal["phase"] = "restored"
                _write_journal(output_dir, journal)
            _remove_generation(stage)
        else:
            raise RuntimeError("last-good combined-v2 generation is unavailable; recovery stopped")
    else:
        if backup_exists:
            raise RuntimeError("unexpected backup in a first-generation publication")
        if phase == "restored" and output_exists:
            raise RuntimeError("restored first-generation state unexpectedly has an output")
        if phase not in {"restoring", "restored"}:
            if output_exists and stage_exists:
                raise RuntimeError("publication recovery found an ambiguous first-generation state")
            journal["phase"] = "restoring"
            _write_journal(output_dir, journal)
        if output_exists:
            if stage_exists:
                raise RuntimeError("publication recovery found an ambiguous first-generation state")
            _rename_sibling(output_dir, stage)
        journal["phase"] = "restored"
        _write_journal(output_dir, journal)
        _remove_generation(stage)


def _validate_recovery_state(output_dir: Path, journal: dict[str, object]) -> tuple[Path, Path]:
    stage = _generation_path(output_dir, "stage", journal["token"])
    backup = _generation_path(output_dir, "backup", journal["token"])
    paths = (output_dir, stage, backup)
    if any(_is_reparse_point(path) for path in paths):
        raise RuntimeError("publication recovery encountered a link or junction")
    state = tuple(_path_exists(path) for path in paths)
    phase = journal["phase"]
    had_output = journal["had_output"]
    allowed = {
        ("prepared", True): {(True, True, False), (False, True, True)},
        ("prepared", False): {(False, True, False)},
        ("backed_up", True): {(False, True, True), (True, False, True)},
        ("backed_up", False): {(False, True, False), (True, False, False)},
        ("published", True): {(True, False, True), (False, True, True)},
        ("published", False): {(True, False, False), (False, True, False)},
        ("restoring", True): {(False, True, True), (True, True, False)},
        ("restoring", False): {(True, False, False), (False, True, False)},
        ("restored", True): {(True, True, False), (True, False, False)},
        ("restored", False): {(False, True, False), (False, False, False)},
        ("committed", True): {(True, False, True), (True, False, False)},
        ("committed", False): {(True, False, False)},
    }
    if state not in allowed[(phase, had_output)]:
        raise RuntimeError("publication journal and generation paths are inconsistent")
    return stage, backup


def _recover_publication(output_dir: Path) -> None:
    journal_path = _journal_path(output_dir)
    if not _path_exists(journal_path):
        return
    journal = _read_journal(output_dir)
    stage, backup = _validate_recovery_state(output_dir, journal)
    if journal["phase"] == "committed":
        try:
            _verify_generation(output_dir, output_dir)
        except Exception:
            # Cleanup may have partially removed the old backup after commit.
            # Never roll a verified generation back to those incomplete bytes.
            raise RuntimeError(
                "committed combined-v2 output could not be verified; backup and journal retained"
            ) from None
        else:
            _remove_generation(stage)
            _remove_generation(backup)
            journal_path.unlink()
            _fsync_directory(output_dir.parent)
            return
    else:
        _rollback_uncommitted(output_dir, journal)
    journal_path.unlink()
    _fsync_directory(output_dir.parent)


def _validate_source_directories(bcs_dir: Path, navid_dir: Path) -> None:
    if not bcs_dir.is_dir():
        raise FileNotFoundError(f"BCS source directory is unavailable: {bcs_dir}")
    required = (
        navid_dir / "train" / "images",
        navid_dir / "train" / "labels",
        navid_dir / "valid" / "images",
        navid_dir / "valid" / "labels",
    )
    if any(not path.is_dir() for path in required):
        raise FileNotFoundError("Navid train/valid image and label directories are required")


def _copy2_and_sync(source: Path, destination: Path) -> None:
    """Flush copied bytes before applying source permissions and timestamps."""
    shutil.copyfile(source, destination)
    with destination.open("rb+") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    shutil.copystat(source, destination)


def _sync_generation(generation: Path) -> None:
    for path in generation.rglob("*"):
        if _is_reparse_point(path):
            raise RuntimeError("staged generation unexpectedly contains a link or junction")
        if path.is_file():
            writable_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
            if not path.stat().st_mode & writable_bits:
                # _copy2_and_sync flushed the content before copying read-only mode bits.
                continue
            with path.open("rb+") as stream:
                stream.flush()
                os.fsync(stream.fileno())
    for directory in sorted(
        (generation, *(path for path in generation.rglob("*") if path.is_dir())),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        _fsync_directory(directory)


def _populate_generation(
    stage: Path,
    final_output: Path,
    bcs_dir: Path,
    navid_dir: Path,
    combined_v1: Path,
) -> tuple[int, int, int, int, int]:
    for directory in ("train/images", "train/labels", "val/images", "val/labels"):
        (stage / directory).mkdir(parents=True)

    train_img, train_lbl = stage / "train/images", stage / "train/labels"
    val_img, val_lbl = stage / "val/images", stage / "val/labels"
    used = collect_used_names(combined_v1)
    print(f"Images already used in v1: {len(used)}")

    total_bcs = 0
    rng = random.Random(SEED)
    for subset in sorted(bcs_dir.iterdir()):
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
        rng.shuffle(pairs)
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
                _copy2_and_sync(jpg, dj)
            dt.write_text("\n".join(yolo_lines), encoding="utf-8")

        for jpg, xml in val_pairs:
            yolo_lines = xml_to_yolo(xml)
            if not yolo_lines:
                continue
            dj = val_img / jpg.name
            dt = val_lbl / f"{jpg.stem}.txt"
            if not dj.exists():
                _copy2_and_sync(jpg, dj)
            dt.write_text("\n".join(yolo_lines), encoding="utf-8")

        print(f"  {subset.name}: {available} unused -> took {n_take} "
              f"({len(train_pairs)} train, {len(val_pairs)} val)")
        total_bcs += n_take

    # Merge Navid
    n_train_navid, n_val_navid = merge_navid(
        train_img, train_lbl, val_img, val_lbl, navid_dir=navid_dir
    )

    total_train = len(list(train_img.glob("*.jpg")))
    total_val = len(list(val_img.glob("*.jpg")))

    # data.yaml
    (stage / "data.yaml").write_text(
        f"# VACCA Fase 1 — Combined v2 (unused BCS + Navid)\n"
        f"path: {final_output.as_posix()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: 1\n"
        f"names: ['cow']\n"
        f"# BCS (unused): {total_bcs} imgs | Navid: {n_train_navid + n_val_navid} imgs\n",
        encoding="utf-8",
    )

    return total_train, total_val, total_bcs, n_train_navid, n_val_navid


def build_dataset(
    *,
    bcs_dir: Path,
    navid_dir: Path,
    combined_v1: Path,
    output_dir: Path,
) -> tuple[int, int, int, int, int]:
    """Build a staged sibling generation and publish it with recoverable rollback."""
    output_dir = _normalize_output_dir(output_dir)
    with _single_writer_lock(output_dir):
        _recover_publication(output_dir)
        _validate_source_directories(bcs_dir, navid_dir)
        if _path_exists(output_dir) and (
            _is_reparse_point(output_dir) or not output_dir.is_dir()
        ):
            raise ValueError("published output must be a normal directory")

        token = uuid.uuid4().hex
        stage = _generation_path(output_dir, "stage", token)
        backup = _generation_path(output_dir, "backup", token)
        if _path_exists(stage) or _path_exists(backup):
            raise FileExistsError("generated publication identity already exists")
        stage.mkdir()
        journal_path = _journal_path(output_dir)
        journal = {
            "version": JOURNAL_VERSION,
            "token": token,
            "had_output": _path_exists(output_dir),
            "phase": "prepared",
        }
        try:
            counts = _populate_generation(
                stage, output_dir, bcs_dir, navid_dir, combined_v1
            )
            _sync_generation(stage)
            _verify_generation(stage, output_dir)
            _write_journal(output_dir, journal)

            if journal["had_output"]:
                _rename_sibling(output_dir, backup)
            journal["phase"] = "backed_up"
            _write_journal(output_dir, journal)
            _rename_sibling(stage, output_dir)
            journal["phase"] = "published"
            _write_journal(output_dir, journal)
            _verify_generation(output_dir, output_dir)
            journal["phase"] = "committed"
            _write_journal(output_dir, journal)
            _recover_publication(output_dir)
            return counts
        except BaseException as build_error:
            try:
                if _path_exists(journal_path):
                    recovery_state = _read_journal(output_dir)
                    if recovery_state["phase"] == "committed":
                        raise RuntimeError(
                            "committed combined-v2 cleanup or verification needs a later retry"
                        )
                    _recover_publication(output_dir)
                else:
                    _remove_generation(stage)
            except Exception as recovery_error:
                raise RuntimeError(
                    "combined-v2 build failed; safe publication recovery remains pending "
                    f"({type(build_error).__name__}/{type(recovery_error).__name__})"
                ) from None
            raise


def main() -> None:
    total_train, total_val, _, n_train_navid, n_val_navid = build_dataset(
        bcs_dir=BCS_DIR,
        navid_dir=NAVID_DIR,
        combined_v1=COMBINED_V1,
        output_dir=OUT_DIR,
    )
    print(f"\nNavid: {n_train_navid} train, {n_val_navid} val")
    print(f"\n[DONE] combined-v2: {total_train} train, {total_val} val "
          f"({total_train + total_val} total)")


if __name__ == "__main__":
    main()

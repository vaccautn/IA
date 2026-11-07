"""Export the trusted BCS candidate into a private deterministic ZIP bundle."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile, ZipInfo

import torch

from vacca_bcs.checkpoint_io import load_checkpoint_bytes
from vacca_bcs.serving import CHECKPOINT_ROOT, _validate_checkpoint
from vacca_bcs.model import BCSOrdinalModel
from vacca_bcs.serving_package import (
    CATALOG_PACKAGE_ROOT,
    ARCHIVE_COMPRESSION,
    PACKAGE_MEMBERS,
    PACKAGE_POLICY,
    PRIVATE_ARCHIVE_ROOT,
    ServingPackageError,
    build_manifest,
    read_publication_documents,
    validate_package_directory,
)

SOURCE_CHECKPOINT = "outputs/bcs-category-coral-v1/weights/best.pt"
SOURCE_SHA256 = PACKAGE_POLICY.source_checkpoint_sha256
ARCHIVE_MEMBERS = PACKAGE_MEMBERS


def _project_state(state: object) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("source model_state_dict is empty or invalid")
    if any(type(name) is not str or not isinstance(value, torch.Tensor) for name, value in state.items()):
        raise ValueError("source model_state_dict contains invalid entries")
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _compare_states(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> None:
    if set(left) != set(right):
        raise ValueError("source and projected state dictionaries have different names")
    for name in left:
        if left[name].dtype != right[name].dtype or tuple(left[name].shape) != tuple(right[name].shape):
            raise ValueError(f"source and projected tensor metadata differ for {name}")
        if not torch.equal(left[name], right[name]):
            raise ValueError(f"source and projected tensor values differ for {name}")


def _write_staged(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _deterministic_archive(package_root: Path) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w", compression=ARCHIVE_COMPRESSION, allowZip64=False) as archive:
        for member in ARCHIVE_MEMBERS:
            info = ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ARCHIVE_COMPRESSION
            info.create_system = 0
            info.external_attr = 0o600 << 16
            info.extra = b""
            info.comment = b""
            archive.writestr(info, (package_root / member).read_bytes())
    return output.getvalue()


def export(
    *,
    source_path: Path | None = None,
    output_root: Path | None = None,
    model_card_path: Path = CATALOG_PACKAGE_ROOT / PACKAGE_POLICY.model_card_path,
    third_party_notices_path: Path = CATALOG_PACKAGE_ROOT / PACKAGE_POLICY.third_party_notices_path,
) -> tuple[int, str]:
    output_dir = PRIVATE_ARCHIVE_ROOT if output_root is None else Path(output_root)
    final_dir = output_dir / PACKAGE_POLICY.archive_directory
    if final_dir.exists() or final_dir.is_symlink():
        raise FileExistsError(f"immutable private archive destination already exists: {final_dir}")

    documents = read_publication_documents(model_card_path, third_party_notices_path)
    source_file = SOURCE_CHECKPOINT if source_path is None else str(source_path)
    loaded = load_checkpoint_bytes(
        source_file,
        approved_roots=(CHECKPOINT_ROOT,),
        expected_sha256=SOURCE_SHA256,
        require_checkpoint_set=True,
    )
    if loaded.sha256 != SOURCE_SHA256:
        raise ValueError("trusted source checkpoint digest changed")
    source = _validate_checkpoint(loaded.payload)
    source_state = _project_state(source["model_state_dict"])

    source_model = BCSOrdinalModel(pretrained=False)
    source_model.load_state_dict(source_state, strict=True)
    source_model.eval()
    projected_buffer = BytesIO()
    torch.save(source_state, projected_buffer)
    artifact_bytes = projected_buffer.getvalue()
    projected_state = torch.load(BytesIO(artifact_bytes), map_location="cpu", weights_only=True)
    if not isinstance(projected_state, Mapping):
        raise ValueError("projected artifact did not reload as a mapping")
    _compare_states(source_state, projected_state)
    projection_model = BCSOrdinalModel(pretrained=False)
    projection_model.load_state_dict(projected_state, strict=True)
    projection_model.eval()
    torch.manual_seed(0)
    sample = torch.randn(2, 3, source["config"]["imgsz"], source["config"]["imgsz"])
    with torch.inference_mode():
        source_logits = source_model(sample)
        projection_logits = projection_model(sample)
    if not torch.equal(source_logits, projection_logits):
        raise ValueError("source and projected models produced different deterministic logits")

    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    documentation_digests = {
        name: hashlib.sha256(raw).hexdigest() for name, raw in documents.items()
    }
    manifest = build_manifest(
        source, len(artifact_bytes), artifact_sha256, documentation_digests, PACKAGE_POLICY
    )
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    build_staging = Path(tempfile.mkdtemp(prefix=f".{PACKAGE_POLICY.package_id}.build-", dir=output_dir))
    delivery_staging: Path | None = None
    published = False
    try:
        (build_staging / PACKAGE_POLICY.artifact_path).write_bytes(artifact_bytes)
        (build_staging / "manifest.json").write_bytes(manifest_bytes)
        (build_staging / PACKAGE_POLICY.model_card_path).write_bytes(documents["model_card"])
        (build_staging / PACKAGE_POLICY.third_party_notices_path).write_bytes(
            documents["third_party_notices"]
        )
        validate_package_directory(build_staging, PACKAGE_POLICY)
        archive_bytes = _deterministic_archive(build_staging)
        archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
        if PACKAGE_POLICY.archive_size_bytes and len(archive_bytes) != PACKAGE_POLICY.archive_size_bytes:
            raise ServingPackageError("generated archive size differs from trusted policy")
        if PACKAGE_POLICY.archive_sha256.strip("0") and archive_sha256 != PACKAGE_POLICY.archive_sha256:
            raise ServingPackageError("generated archive digest differs from trusted policy")
        delivery_staging = Path(tempfile.mkdtemp(prefix=f".{PACKAGE_POLICY.archive_directory}.delivery-", dir=output_dir))
        _write_staged(delivery_staging / PACKAGE_POLICY.archive_filename, archive_bytes)
        _write_staged(
            delivery_staging / f"{PACKAGE_POLICY.archive_filename}.sha256",
            f"{archive_sha256}  {PACKAGE_POLICY.archive_filename}\n".encode("ascii"),
        )
        if final_dir.exists() or final_dir.is_symlink():
            raise FileExistsError(f"immutable private archive destination already exists: {final_dir}")
        os.rename(delivery_staging, final_dir)
        published = True
        return len(archive_bytes), archive_sha256
    finally:
        if delivery_staging is not None and not published and delivery_staging.exists():
            shutil.rmtree(delivery_staging, ignore_errors=True)
        if build_staging.exists():
            shutil.rmtree(build_staging, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=PRIVATE_ARCHIVE_ROOT)
    parser.add_argument("--model-card", type=Path, default=CATALOG_PACKAGE_ROOT / PACKAGE_POLICY.model_card_path)
    parser.add_argument(
        "--third-party-notices",
        type=Path,
        default=CATALOG_PACKAGE_ROOT / PACKAGE_POLICY.third_party_notices_path,
    )
    args = parser.parse_args()
    size, digest = export(
        source_path=args.source,
        output_root=args.output_dir,
        model_card_path=args.model_card,
        third_party_notices_path=args.third_party_notices,
    )
    print(json.dumps({"archive": str(args.output_dir / PACKAGE_POLICY.archive_directory / PACKAGE_POLICY.archive_filename), "bytes": size, "sha256": digest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

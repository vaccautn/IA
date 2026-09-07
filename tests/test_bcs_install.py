from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import shutil
import stat
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import pytest

from scripts import export_bcs_serving_package as exporter
from scripts import install_bcs_serving_package as installer
from vacca_bcs import serving_package as package


def _write_archive(path: Path, members: list[tuple[str, bytes]], *, compression=ZIP_STORED) -> None:
    with ZipFile(path, "w", compression=compression) as archive:
        for name, payload in members:
            archive.writestr(name, payload)
    path.with_name(f"{path.name}.sha256").write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n", encoding="ascii"
    )


def _bind_archive_policy(monkeypatch, archive: Path) -> None:
    monkeypatch.setattr(
        installer,
        "PACKAGE_POLICY",
        replace(
            package.PACKAGE_POLICY,
            archive_size_bytes=archive.stat().st_size,
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        ),
    )


def _ensure_valid_install(private_model_archive: Path, tmp_path: Path) -> None:
    if package.PRIVATE_PACKAGE_ROOT.exists():
        try:
            package.validate_installed_package()
            return
        except package.ServingPackageError:
            shutil.move(str(package.PRIVATE_PACKAGE_ROOT), str(tmp_path / "preexisting-invalid"))
    installer.install(private_model_archive)


@pytest.fixture(autouse=True)
def isolated_private_installation(request, monkeypatch, tmp_path: Path):
    if request.node.get_closest_marker("private_model_installed") is None:
        yield
        return
    repo_root = tmp_path / "repo"
    private_root = repo_root / "models" / "private"
    package_root = private_root / package.PRIVATE_PACKAGE_ID
    private_root.mkdir(parents=True)
    monkeypatch.setattr(package, "REPO_ROOT", repo_root)
    monkeypatch.setattr(package, "PRIVATE_ROOT", private_root)
    monkeypatch.setattr(package, "PRIVATE_PACKAGE_ROOT", package_root)
    monkeypatch.setattr(package, "PRIVATE_MANIFEST_PATH", package_root / "manifest.json")
    monkeypatch.setattr(package, "PRIVATE_ARTIFACT_PATH", package_root / "model_state.pt")
    monkeypatch.setattr(installer, "REPO_ROOT", repo_root)
    monkeypatch.setattr(installer, "PRIVATE_ROOT", private_root)
    monkeypatch.setattr(installer, "PRIVATE_PACKAGE_ROOT", package_root)
    yield


@pytest.mark.private_model
def test_external_consumer_archive_and_sidecar_are_self_contained(
    private_model_archive: Path,
) -> None:
    members = installer._read_archive(private_model_archive)
    assert set(members) == installer._MEMBERS
    assert Path(f"{private_model_archive}.sha256").is_file()


@pytest.mark.private_model
@pytest.mark.private_model_installed
def test_archive_is_deterministic_and_has_exact_root_members(private_model_archive: Path, tmp_path: Path) -> None:
    _ensure_valid_install(private_model_archive, tmp_path)
    with ZipFile(private_model_archive) as archive:
        infos = archive.infolist()
        assert [info.filename for info in infos] == list(exporter.ARCHIVE_MEMBERS)
        assert all(info.compress_type == ZIP_STORED for info in infos)
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in infos)
        assert all(not info.filename.startswith(("/", "\\")) for info in infos)
    assert exporter._deterministic_archive(package.PRIVATE_PACKAGE_ROOT) == private_model_archive.read_bytes()


@pytest.mark.private_model
@pytest.mark.private_model_installed
def test_install_production_validates_and_keeps_destination_immutable(private_model_archive: Path, tmp_path: Path) -> None:
    _ensure_valid_install(private_model_archive, tmp_path)
    assert installer.install(private_model_archive) == package.PRIVATE_PACKAGE_ID
    assert package.validate_installed_package()["package_id"] == package.PRIVATE_PACKAGE_ID
    sentinel = package.PRIVATE_PACKAGE_ROOT / "manifest.json"
    original = sentinel.read_bytes()
    assert installer.install(private_model_archive) == package.PRIVATE_PACKAGE_ID
    assert sentinel.read_bytes() == original


@pytest.mark.private_model
@pytest.mark.private_model_installed
def test_clean_copy_needs_only_archive_and_sidecar(tmp_path: Path, private_model_archive: Path) -> None:
    _ensure_valid_install(private_model_archive, tmp_path)
    clean_download = tmp_path / "Downloads"
    clean_download.mkdir()
    copied_archive = clean_download / private_model_archive.name
    shutil.copyfile(private_model_archive, copied_archive)
    shutil.copyfile(Path(f"{private_model_archive}.sha256"), Path(f"{copied_archive}.sha256"))
    installer.uninstall()
    try:
        assert installer.install(copied_archive) == package.PRIVATE_PACKAGE_ID
    finally:
        installer.uninstall()
        installer.install(private_model_archive)
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("member_name", ["../escape", "/absolute", "nested/file"])
def test_zip_slip_and_nested_members_are_rejected(tmp_path: Path, monkeypatch, member_name: str) -> None:
    archive = tmp_path / "bad.zip"
    _write_archive(archive, [(member_name, b"bad")])
    _bind_archive_policy(monkeypatch, archive)
    with pytest.raises(package.ServingPackageError):
        installer._read_archive(archive)


def test_archive_rejects_duplicates_compression_and_excess_size(tmp_path: Path, monkeypatch) -> None:
    duplicate = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with ZipFile(duplicate, "w", compression=ZIP_STORED) as archive:
            archive.writestr("manifest.json", b"one")
            archive.writestr("manifest.json", b"two")
    duplicate.with_name(f"{duplicate.name}.sha256").write_text(
        f"{hashlib.sha256(duplicate.read_bytes()).hexdigest()}  {duplicate.name}\n", encoding="ascii"
    )
    _bind_archive_policy(monkeypatch, duplicate)
    with pytest.raises(package.ServingPackageError):
        installer._read_archive(duplicate)

    compressed = tmp_path / "compressed.zip"
    _write_archive(compressed, [(name, b"x") for name in installer._MEMBERS], compression=ZIP_DEFLATED)
    _bind_archive_policy(monkeypatch, compressed)
    with pytest.raises(package.ServingPackageError):
        installer._read_archive(compressed)


@pytest.mark.parametrize("metadata", ["archive_comment", "member_extra", "member_comment", "symlink"])
def test_archive_rejects_metadata_and_symlink_members(tmp_path: Path, monkeypatch, metadata: str) -> None:
    archive_path = tmp_path / f"{metadata}.zip"
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
        if metadata == "archive_comment":
            archive.comment = b"unexpected"
        for name in installer._MEMBERS:
            info = ZipInfo(name)
            if name == "manifest.json" and metadata == "member_extra":
                info.extra = b"unexpected"
            if name == "manifest.json" and metadata == "member_comment":
                info.comment = b"unexpected"
            if name == "manifest.json" and metadata == "symlink":
                info.create_system = 3
                info.external_attr = stat.S_IFLNK << 16
            archive.writestr(info, b"x")
    archive_path.with_name(f"{archive_path.name}.sha256").write_text(
        f"{hashlib.sha256(archive_path.read_bytes()).hexdigest()}  {archive_path.name}\n", encoding="ascii"
    )
    _bind_archive_policy(monkeypatch, archive_path)
    with pytest.raises(package.ServingPackageError):
        installer._read_archive(archive_path)


def test_private_destination_is_exact_and_cannot_escape(tmp_path: Path) -> None:
    with pytest.raises(package.ServingPackageError, match="expected package path"):
        package.validate_private_destination(tmp_path / package.PRIVATE_PACKAGE_ID)


@pytest.mark.private_model
@pytest.mark.private_model_installed
def test_install_cleans_staging_when_production_loader_rejects(tmp_path: Path, monkeypatch, private_model_archive: Path) -> None:
    _ensure_valid_install(private_model_archive, tmp_path)
    monkeypatch.setattr(installer, "load_package_model", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad model")))
    with pytest.raises(RuntimeError, match="bad model"):
        installer.install(private_model_archive)
    assert package.PRIVATE_PACKAGE_ROOT.is_dir()
    assert list(package.PRIVATE_ROOT.glob(f".{package.PRIVATE_PACKAGE_ID}.install-*")) == []


@pytest.mark.private_model
@pytest.mark.private_model_installed
def test_identical_package_is_verified_and_no_op(private_model_archive: Path, tmp_path: Path) -> None:
    _ensure_valid_install(private_model_archive, tmp_path)
    assert installer.install(private_model_archive) == package.PRIVATE_PACKAGE_ID
    assert package.validate_installed_package()["package_id"] == package.PRIVATE_PACKAGE_ID


@pytest.mark.private_model
@pytest.mark.private_model_installed
def test_repair_rolls_back_when_destination_rename_fails(monkeypatch, private_model_archive: Path, tmp_path: Path) -> None:
    _ensure_valid_install(private_model_archive, tmp_path)
    original_rename = installer.os.rename
    original_validate = installer.validate_package_directory
    failed = False

    def force_replacement(package_root, policy):
        if Path(package_root) == package.PRIVATE_PACKAGE_ROOT:
            raise package.ServingPackageError("synthetic replacement required")
        return original_validate(package_root, policy)

    def fail_replacement(source, destination):
        nonlocal failed
        if Path(source).name.startswith(f".{package.PRIVATE_PACKAGE_ID}.install-") and Path(destination) == package.PRIVATE_PACKAGE_ROOT:
            failed = True
            raise OSError("replacement boundary failure")
        return original_rename(source, destination)

    monkeypatch.setattr(installer.os, "rename", fail_replacement)
    monkeypatch.setattr(installer, "validate_package_directory", force_replacement)
    with pytest.raises(OSError, match="replacement boundary failure"):
        installer.install(private_model_archive, repair=True)
    assert failed
    assert package.validate_installed_package()["package_id"] == package.PRIVATE_PACKAGE_ID
    assert list(package.PRIVATE_ROOT.glob(f".{package.PRIVATE_PACKAGE_ID}.quarantine-*"))


@pytest.mark.private_model
@pytest.mark.private_model_installed
def test_repair_double_failure_preserves_backup_and_recovery_restores_it(monkeypatch, private_model_archive: Path, tmp_path: Path) -> None:
    _ensure_valid_install(private_model_archive, tmp_path)
    original_rename = installer.os.rename
    original_validate = installer.validate_package_directory

    def force_replacement(package_root, policy):
        if Path(package_root) == package.PRIVATE_PACKAGE_ROOT:
            raise package.ServingPackageError("synthetic replacement required")
        return original_validate(package_root, policy)

    def fail_publish_and_rollback(source, destination):
        source_name = Path(source).name
        if source_name.startswith(f".{package.PRIVATE_PACKAGE_ID}.install-") and Path(destination) == package.PRIVATE_PACKAGE_ROOT:
            raise OSError("publish boundary failure")
        if source_name.startswith(f".{package.PRIVATE_PACKAGE_ID}.backup-") and Path(destination) == package.PRIVATE_PACKAGE_ROOT:
            raise OSError("rollback boundary failure")
        return original_rename(source, destination)

    monkeypatch.setattr(installer.os, "rename", fail_publish_and_rollback)
    monkeypatch.setattr(installer, "validate_package_directory", force_replacement)
    before_backups = set(package.PRIVATE_ROOT.glob(f".{package.PRIVATE_PACKAGE_ID}.backup-*"))
    with pytest.raises(package.ServingPackageError, match="backup preserved at"):
        installer.install(private_model_archive, repair=True)
    backups = list(set(package.PRIVATE_ROOT.glob(f".{package.PRIVATE_PACKAGE_ID}.backup-*")) - before_backups)
    quarantines = list(package.PRIVATE_ROOT.glob(f".{package.PRIVATE_PACKAGE_ID}.quarantine-*"))
    assert backups and quarantines
    assert not package.PRIVATE_PACKAGE_ROOT.exists()

    monkeypatch.setattr(installer.os, "rename", original_rename)
    assert installer.recover_backup(backups[0]) == package.PRIVATE_PACKAGE_ID
    assert package.validate_installed_package()["package_id"] == package.PRIVATE_PACKAGE_ID


@pytest.mark.private_model
@pytest.mark.private_model_installed
def test_repair_state_matrix_handles_invalid_existing_destination_and_retains_old_bytes(
    private_model_archive: Path,
) -> None:
    installer.install(private_model_archive)
    sentinel = package.PRIVATE_PACKAGE_ROOT / "manifest.json"
    sentinel.write_text("invalid existing package", encoding="utf-8")
    unrelated = package.PRIVATE_ROOT / "unrelated-preserved.txt"
    unrelated.write_text("keep", encoding="ascii")

    installer.install(private_model_archive, repair=True)

    assert package.validate_installed_package()["package_id"] == package.PRIVATE_PACKAGE_ID
    backups = list(package.PRIVATE_ROOT.glob(f".{package.PRIVATE_PACKAGE_ID}.backup-*"))
    retained_invalid = [
        backup
        for backup in backups
        if (backup / "manifest.json").is_file()
        and (backup / "manifest.json").read_text(encoding="utf-8") == "invalid existing package"
    ]
    assert retained_invalid
    assert unrelated.read_text(encoding="ascii") == "keep"


@pytest.mark.private_model
@pytest.mark.private_model_installed
def test_repair_post_publish_validation_failure_quarantines_replacement_and_restores_old(
    monkeypatch, private_model_archive: Path
) -> None:
    installer.install(private_model_archive)
    (package.PRIVATE_PACKAGE_ROOT / "manifest.json").write_text("invalid existing package", encoding="utf-8")
    original_loader = installer.load_package_model
    calls = 0

    def fail_after_staging(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("post-publish validation failure")
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(installer, "load_package_model", fail_after_staging)
    with pytest.raises(package.ServingPackageError, match="replacement quarantined"):
        installer.install(private_model_archive, repair=True)
    assert (package.PRIVATE_PACKAGE_ROOT / "manifest.json").read_text(encoding="utf-8") == "invalid existing package"
    assert list(package.PRIVATE_ROOT.glob(f".{package.PRIVATE_PACKAGE_ID}.quarantine-*"))


@pytest.mark.private_model
@pytest.mark.private_model_installed
def test_repair_allows_existing_stale_backup_without_unrelated_deletion(private_model_archive: Path, tmp_path: Path) -> None:
    _ensure_valid_install(private_model_archive, tmp_path)
    stale = package.PRIVATE_ROOT / f".{package.PRIVATE_PACKAGE_ID}.backup-stale"
    stale.mkdir(exist_ok=True)
    (stale / "unrelated.txt").write_text("keep", encoding="ascii")
    (package.PRIVATE_PACKAGE_ROOT / "manifest.json").write_text("invalid existing package", encoding="utf-8")

    installer.install(private_model_archive, repair=True)

    assert stale.is_dir()
    assert (stale / "unrelated.txt").read_text(encoding="ascii") == "keep"
    assert package.validate_installed_package()["package_id"] == package.PRIVATE_PACKAGE_ID


@pytest.mark.private_model
@pytest.mark.private_model_installed
def test_recover_backup_absent_invalid_and_valid_destinations_are_safe(
    private_model_archive: Path, tmp_path: Path
) -> None:
    _ensure_valid_install(private_model_archive, tmp_path)
    valid_backup = tmp_path / f".{package.PRIVATE_PACKAGE_ID}.backup-valid"
    shutil.copytree(package.PRIVATE_PACKAGE_ROOT, valid_backup)
    # Recovery paths must remain direct children of the fixed private root.
    private_backup = package.PRIVATE_ROOT / valid_backup.name
    second_backup = package.PRIVATE_ROOT / f".{package.PRIVATE_PACKAGE_ID}.backup-valid-2"
    shutil.copytree(valid_backup, private_backup)
    shutil.copytree(valid_backup, second_backup)

    installer.uninstall()
    assert installer.recover_backup(private_backup) == package.PRIVATE_PACKAGE_ID
    with pytest.raises(package.ServingPackageError, match="already valid"):
        installer.recover_backup(second_backup)
    assert second_backup.is_dir()

    (package.PRIVATE_PACKAGE_ROOT / "manifest.json").write_text("invalid", encoding="utf-8")
    assert installer.recover_backup(second_backup) == package.PRIVATE_PACKAGE_ID
    assert package.validate_installed_package()["package_id"] == package.PRIVATE_PACKAGE_ID
    assert list(package.PRIVATE_ROOT.glob(f".{package.PRIVATE_PACKAGE_ID}.quarantine-*"))


@pytest.mark.private_model
@pytest.mark.private_model_installed
def test_uninstall_retains_validated_package_as_disabled_backup(private_model_archive: Path, tmp_path: Path) -> None:
    _ensure_valid_install(private_model_archive, tmp_path)
    disabled = installer.uninstall()
    assert disabled is not None
    assert disabled.name.startswith(f".{package.PRIVATE_PACKAGE_ID}.disabled-")
    assert package.PRIVATE_PACKAGE_ROOT.exists() is False
    assert package.validate_package_directory(disabled)["package_id"] == package.PRIVATE_PACKAGE_ID
    assert installer.recover_backup(disabled) == package.PRIVATE_PACKAGE_ID
    assert package.validate_installed_package()["package_id"] == package.PRIVATE_PACKAGE_ID


@pytest.mark.private_model
@pytest.mark.private_model_installed
@pytest.mark.parametrize(
    "crafted_name",
    [
        ".bcs-category-coral-2026-09-04.disabled",
        ".bcs-category-coral-2026-09-04.disabled-../escape",
        ".bcs-category-coral-2026-09-04.disabled-\\escape",
    ],
)
def test_recover_backup_rejects_crafted_disabled_names_and_escapes(
    private_model_archive: Path, crafted_name: str, tmp_path: Path
) -> None:
    _ensure_valid_install(private_model_archive, tmp_path)
    with pytest.raises(package.ServingPackageError):
        installer.recover_backup(package.PRIVATE_ROOT / crafted_name)


@pytest.mark.private_model
@pytest.mark.private_model_installed
def test_recover_backup_rejects_disabled_symlink(private_model_archive: Path, tmp_path: Path) -> None:
    _ensure_valid_install(private_model_archive, tmp_path)
    disabled = package.PRIVATE_ROOT / f".{package.PRIVATE_PACKAGE_ID}.disabled-symlink"
    try:
        disabled.symlink_to(package.PRIVATE_PACKAGE_ROOT, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this Windows checkout")
    with pytest.raises(package.ServingPackageError):
        installer.recover_backup(disabled)

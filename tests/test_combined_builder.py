from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from scripts import build_combined_v2 as builder


VALID_XML = """<annotation><size><width>10</width><height>10</height></size>
<object><bndbox><xmin>2</xmin><ymin>2</ymin><xmax>8</xmax><ymax>8</ymax></bndbox></object>
</annotation>"""


def make_sources(tmp_path, *, xml=VALID_XML):
    bcs = tmp_path / "bcs"
    subset = bcs / "class-fixture"
    subset.mkdir(parents=True)
    (subset / "bcs-1.jpg").write_bytes(b"synthetic bcs image")
    (subset / "bcs-1.xml").write_text(xml, encoding="utf-8")

    navid = tmp_path / "navid"
    for split in ("train", "valid"):
        images = navid / split / "images"
        labels = navid / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        (images / f"navid-{split}.jpg").write_bytes(f"synthetic {split} image".encode())
        (labels / f"navid-{split}.txt").write_text("0 .5 .5 .5 .5", encoding="utf-8")

    combined_v1 = tmp_path / "combined-v1"
    combined_v1.mkdir()
    return bcs, navid, combined_v1


def build(output, sources):
    bcs, navid, combined_v1 = sources
    return builder.build_dataset(
        bcs_dir=bcs,
        navid_dir=navid,
        combined_v1=combined_v1,
        output_dir=output,
    )


def assert_no_publication_artifacts(output):
    assert not list(output.parent.glob(f".{output.name}.stage-*"))
    assert not list(output.parent.glob(f".{output.name}.backup-*"))
    assert not (output.parent / f".{output.name}.publish-journal.json").exists()


def write_generation(path, final_output, marker):
    for relative in ("train/images", "train/labels", "val/images", "val/labels"):
        (path / relative).mkdir(parents=True)
    (path / "data.yaml").write_text(
        f"path: {final_output.as_posix()}\n", encoding="utf-8"
    )
    (path / "generation.txt").write_text(marker, encoding="utf-8")


@pytest.mark.parametrize("existing_generation", [True, False])
def test_missing_bcs_source_preserves_published_generation(
    tmp_path, monkeypatch, existing_generation
):
    output = tmp_path / "data" / "combined-v2"
    sentinel = output / "last-good.txt"
    if existing_generation:
        output.mkdir(parents=True)
        sentinel.write_text("previous generation", encoding="utf-8")
    monkeypatch.setattr(builder, "OUT_DIR", output)
    monkeypatch.setattr(builder, "BCS_DIR", tmp_path / "missing-bcs")
    monkeypatch.setattr(builder, "NAVID_DIR", tmp_path / "navid")
    monkeypatch.setattr(builder, "COMBINED_V1", tmp_path / "combined-v1")

    with pytest.raises(FileNotFoundError):
        builder.main()

    if existing_generation:
        assert sentinel.read_text(encoding="utf-8") == "previous generation"
    else:
        assert not output.exists()
    assert_no_publication_artifacts(output)


def test_active_writer_lock_rejects_a_second_builder(tmp_path):
    output = tmp_path / "data" / "combined-v2"
    output.parent.mkdir(parents=True)

    with builder._single_writer_lock(output):
        with pytest.raises(RuntimeError, match="another combined-v2 build is active"):
            with builder._single_writer_lock(output):
                pytest.fail("second writer entered the publication lock")


def test_successful_build_publishes_complete_generation_and_final_yaml_root(tmp_path):
    output = tmp_path / "data" / "combined-v2"
    sources = make_sources(tmp_path)
    readonly_image = sources[0] / "class-fixture" / "bcs-1.jpg"
    readonly_image.chmod(stat.S_IREAD)
    try:
        result = build(output, sources)
    finally:
        readonly_image.chmod(stat.S_IREAD | stat.S_IWRITE)

    assert result == (1, 2, 1, 1, 1)
    assert (output / "train/images/navid-train.jpg").read_bytes() == b"synthetic train image"
    assert (output / "train/labels/navid-train.txt").read_text(encoding="utf-8") == "0 .5 .5 .5 .5"
    assert (output / "val/images/bcs-1.jpg").read_bytes() == b"synthetic bcs image"
    yaml = (output / "data.yaml").read_text(encoding="utf-8")
    assert f"path: {output.as_posix()}" in yaml.splitlines()
    assert ".stage-" not in yaml


def test_malformed_selected_xml_preserves_published_generation(tmp_path):
    output = tmp_path / "data" / "combined-v2"
    output.mkdir(parents=True)
    sentinel = output / "last-good.txt"
    sentinel.write_text("previous generation", encoding="utf-8")

    with pytest.raises(builder.ET.ParseError):
        build(output, make_sources(tmp_path, xml="<annotation><object>"))

    assert sentinel.read_text(encoding="utf-8") == "previous generation"


def test_source_copy_failure_preserves_last_good_and_cleans_stage(tmp_path, monkeypatch):
    output = tmp_path / "data" / "combined-v2"
    output.mkdir(parents=True)
    sentinel = output / "last-good.txt"
    sentinel.write_text("previous generation", encoding="utf-8")

    def fail_copy(*args, **kwargs):
        raise OSError("synthetic source copy failure")

    monkeypatch.setattr(builder.shutil, "copyfile", fail_copy)
    with pytest.raises(OSError, match="synthetic source copy failure"):
        build(output, make_sources(tmp_path))

    assert sentinel.read_text(encoding="utf-8") == "previous generation"
    assert_no_publication_artifacts(output)


@pytest.mark.parametrize("failure_point", ["backup", "publish"])
def test_directory_rename_failure_restores_last_good(tmp_path, monkeypatch, failure_point):
    output = tmp_path / "data" / "combined-v2"
    output.mkdir(parents=True)
    sentinel = output / "last-good.txt"
    sentinel.write_text("previous generation", encoding="utf-8")
    original_rename = builder.os.rename

    def fail_selected_rename(source, destination):
        source, destination = Path(source), Path(destination)
        if failure_point == "backup" and source == output and ".backup-" in destination.name:
            raise PermissionError("synthetic open-handle rename failure")
        if failure_point == "publish" and ".stage-" in source.name and destination == output:
            raise PermissionError("synthetic publish rename failure")
        return original_rename(source, destination)

    monkeypatch.setattr(builder.os, "rename", fail_selected_rename)
    with pytest.raises(PermissionError, match="synthetic .* rename failure"):
        build(output, make_sources(tmp_path))

    assert sentinel.read_text(encoding="utf-8") == "previous generation"
    assert_no_publication_artifacts(output)


def test_post_publish_verification_failure_restores_last_good(tmp_path, monkeypatch):
    output = tmp_path / "data" / "combined-v2"
    output.mkdir(parents=True)
    sentinel = output / "last-good.txt"
    sentinel.write_text("previous generation", encoding="utf-8")
    verify = builder._verify_generation
    calls = 0

    def fail_after_publish(generation, final_output):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic post-publish verification failure")
        verify(generation, final_output)

    monkeypatch.setattr(builder, "_verify_generation", fail_after_publish)
    with pytest.raises(OSError, match="synthetic post-publish verification failure"):
        build(output, make_sources(tmp_path))

    assert sentinel.read_text(encoding="utf-8") == "previous generation"
    assert_no_publication_artifacts(output)


def test_initial_journal_write_failure_keeps_last_good(tmp_path, monkeypatch):
    output = tmp_path / "data" / "combined-v2"
    output.mkdir(parents=True)
    sentinel = output / "last-good.txt"
    sentinel.write_text("previous generation", encoding="utf-8")
    replace = builder.os.replace

    def fail_journal_replace(source, destination):
        if Path(destination) == builder._journal_path(output):
            raise OSError("synthetic journal replace failure")
        replace(source, destination)

    monkeypatch.setattr(builder.os, "replace", fail_journal_replace)

    with pytest.raises(OSError, match="synthetic journal replace failure"):
        build(output, make_sources(tmp_path))

    assert sentinel.read_text(encoding="utf-8") == "previous generation"
    assert_no_publication_artifacts(output)
    assert not list(output.parent.glob(f".{output.name}.journal-*.tmp"))


def test_journal_update_failure_after_backup_move_restores_last_good(tmp_path, monkeypatch):
    output = tmp_path / "data" / "combined-v2"
    output.mkdir(parents=True)
    sentinel = output / "last-good.txt"
    sentinel.write_text("previous generation", encoding="utf-8")
    write_journal = builder._write_journal
    writes = 0

    def fail_backed_up_record(output_path, journal):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("synthetic backed-up journal failure")
        write_journal(output_path, journal)

    monkeypatch.setattr(builder, "_write_journal", fail_backed_up_record)
    with pytest.raises(OSError, match="synthetic backed-up journal failure"):
        build(output, make_sources(tmp_path))

    assert sentinel.read_text(encoding="utf-8") == "previous generation"
    assert_no_publication_artifacts(output)


def test_first_generation_publish_failure_leaves_output_absent(tmp_path, monkeypatch):
    output = tmp_path / "data" / "combined-v2"
    original_rename = builder.os.rename

    def fail_publish(source, destination):
        if ".stage-" in Path(source).name and Path(destination) == output:
            raise PermissionError("synthetic first publish failure")
        original_rename(source, destination)

    monkeypatch.setattr(builder.os, "rename", fail_publish)
    with pytest.raises(PermissionError, match="synthetic first publish failure"):
        build(output, make_sources(tmp_path))

    assert not output.exists()
    assert_no_publication_artifacts(output)


@pytest.mark.parametrize(
    "phase,had_output,backup_exists,stage_exists,output_exists,expect_new,expect_prior",
    [
        ("prepared", True, True, True, False, False, True),
        ("backed_up", True, True, False, True, False, True),
        ("backed_up", False, False, False, True, False, False),
        ("committed", True, True, False, True, True, False),
    ],
)
def test_recovery_resolves_interrupted_publication_states(
    tmp_path, phase, had_output, backup_exists, stage_exists, output_exists, expect_new, expect_prior
):
    output = tmp_path / "data" / "combined-v2"
    output.parent.mkdir(parents=True)
    token = "a" * 32
    stage = builder._generation_path(output, "stage", token)
    backup = builder._generation_path(output, "backup", token)
    if backup_exists:
        backup.mkdir()
        (backup / "last-good.txt").write_text("previous generation", encoding="utf-8")
    if stage_exists:
        stage.mkdir()
        (stage / "partial.txt").write_text("incomplete stage", encoding="utf-8")
    if output_exists:
        write_generation(output, output, "verified new generation")
    builder._write_journal(
        output,
        {
            "version": builder.JOURNAL_VERSION,
            "token": token,
            "had_output": had_output,
            "phase": phase,
        },
    )

    builder._recover_publication(output)

    if expect_new:
        assert (output / "generation.txt").read_text(encoding="utf-8") == "verified new generation"
    elif expect_prior:
        assert (output / "last-good.txt").read_text(encoding="utf-8") == "previous generation"
    else:
        assert not output.exists()
    assert_no_publication_artifacts(output)


def test_recovery_resumes_after_backup_restore_before_stage_cleanup(
    tmp_path, monkeypatch
):
    output = tmp_path / "data" / "combined-v2"
    output.parent.mkdir(parents=True)
    token = "b" * 32
    stage = builder._generation_path(output, "stage", token)
    backup = builder._generation_path(output, "backup", token)
    stage.mkdir()
    (stage / "partial.txt").write_text("new generation", encoding="utf-8")
    backup.mkdir()
    (backup / "last-good.txt").write_text("previous generation", encoding="utf-8")
    builder._write_journal(
        output,
        {
            "version": builder.JOURNAL_VERSION,
            "token": token,
            "had_output": True,
            "phase": "backed_up",
        },
    )
    remove_generation = builder._remove_generation

    def fail_stage_cleanup(path):
        if path == stage:
            raise OSError("synthetic interruption after restoring backup")
        remove_generation(path)

    monkeypatch.setattr(builder, "_remove_generation", fail_stage_cleanup)
    with pytest.raises(OSError, match="synthetic interruption"):
        builder._recover_publication(output)
    assert (output / "last-good.txt").read_text(encoding="utf-8") == "previous generation"
    assert builder._journal_path(output).is_file()

    monkeypatch.setattr(builder, "_remove_generation", remove_generation)
    builder._recover_publication(output)

    assert (output / "last-good.txt").read_text(encoding="utf-8") == "previous generation"
    assert_no_publication_artifacts(output)


def test_next_build_recovers_last_good_before_validating_new_sources(tmp_path):
    output = tmp_path / "data" / "combined-v2"
    output.parent.mkdir(parents=True)
    token = "c" * 32
    stage = builder._generation_path(output, "stage", token)
    backup = builder._generation_path(output, "backup", token)
    stage.mkdir()
    (stage / "partial.txt").write_text("incomplete", encoding="utf-8")
    backup.mkdir()
    (backup / "last-good.txt").write_text("previous generation", encoding="utf-8")
    builder._write_journal(
        output,
        {
            "version": builder.JOURNAL_VERSION,
            "token": token,
            "had_output": True,
            "phase": "prepared",
        },
    )

    with pytest.raises(FileNotFoundError, match="BCS source directory"):
        build(output, (tmp_path / "missing-bcs", tmp_path / "missing-navid", tmp_path / "v1"))

    assert (output / "last-good.txt").read_text(encoding="utf-8") == "previous generation"
    assert_no_publication_artifacts(output)


def test_malformed_journal_cannot_escape_output_parent(tmp_path):
    output = tmp_path / "data" / "combined-v2"
    output.parent.mkdir(parents=True)
    outside = tmp_path / "keep.txt"
    outside.write_text("unrelated", encoding="utf-8")
    journal = builder._journal_path(output)
    journal.write_text(
        '{"version":1,"token":"../keep.txt","had_output":false,"phase":"prepared"}',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="malformed"):
        builder._recover_publication(output)

    assert outside.read_text(encoding="utf-8") == "unrelated"
    assert journal.is_file()


@pytest.mark.parametrize("verification_failure_call", [2, 3])
def test_committed_cleanup_verification_error_never_rolls_back_verified_output(
    tmp_path, monkeypatch, verification_failure_call
):
    output = tmp_path / "data" / "combined-v2"
    old_sources = make_sources(tmp_path / "old-sources")
    old_image = old_sources[0] / "class-fixture" / "bcs-1.jpg"
    old_image.write_bytes(b"old important image bytes")
    old_image.chmod(stat.S_IREAD)
    try:
        build(output, old_sources)
    finally:
        old_image.chmod(stat.S_IREAD | stat.S_IWRITE)

    new_sources = make_sources(tmp_path / "new-sources")
    new_image = new_sources[0] / "class-fixture" / "bcs-1.jpg"
    new_image.write_bytes(b"verified new image bytes")
    remove_generation = builder._remove_generation
    verify_generation = builder._verify_generation
    backup_partially_removed = False
    output_verifications = 0

    def fail_after_partial_backup_cleanup(path):
        nonlocal backup_partially_removed
        if ".backup-" in path.name and not backup_partially_removed:
            old_backup_image = path / "val/images/bcs-1.jpg"
            old_backup_image.chmod(stat.S_IREAD | stat.S_IWRITE)
            old_backup_image.unlink()
            backup_partially_removed = True
            raise PermissionError("synthetic interrupted backup cleanup")
        remove_generation(path)

    def transient_verify_error(generation, final_output):
        nonlocal output_verifications
        if generation == output and final_output == output:
            output_verifications += 1
            if output_verifications == verification_failure_call:
                raise PermissionError("synthetic transient verification error")
        verify_generation(generation, final_output)

    monkeypatch.setattr(builder, "_remove_generation", fail_after_partial_backup_cleanup)
    monkeypatch.setattr(builder, "_verify_generation", transient_verify_error)
    with pytest.raises(RuntimeError, match="safe publication recovery remains pending"):
        build(output, new_sources)

    assert (output / "val/images/bcs-1.jpg").read_bytes() == b"verified new image bytes"
    backup_dirs = list(output.parent.glob(f".{output.name}.backup-*"))
    assert len(backup_dirs) == 1
    assert (backup_dirs[0] / "val/images/bcs-1.jpg").exists() is (
        verification_failure_call == 2
    )
    journal = builder._read_journal(output)
    assert journal["phase"] == "committed"

    monkeypatch.setattr(builder, "_remove_generation", remove_generation)
    monkeypatch.setattr(builder, "_verify_generation", verify_generation)
    builder._recover_publication(output)

    assert (output / "val/images/bcs-1.jpg").read_bytes() == b"verified new image bytes"
    assert_no_publication_artifacts(output)


@pytest.mark.parametrize(
    "had_output,phase,stage_exists,backup_exists",
    [
        (False, "prepared", False, False),
        (True, "published", True, True),
        (False, "restored", False, True),
    ],
)
def test_unreachable_journal_states_preserve_all_generation_paths(
    tmp_path, had_output, phase, stage_exists, backup_exists
):
    output = tmp_path / "data" / "combined-v2"
    output.mkdir(parents=True)
    sentinel = output / "last-good.txt"
    sentinel.write_bytes(b"published generation bytes")
    token = "e" * 32
    stage = builder._generation_path(output, "stage", token)
    backup = builder._generation_path(output, "backup", token)
    if stage_exists:
        stage.mkdir()
        (stage / "stage.txt").write_bytes(b"stage bytes")
    if backup_exists:
        backup.mkdir()
        (backup / "backup.txt").write_bytes(b"backup bytes")
    journal = builder._journal_path(output)
    journal.write_text(
        json.dumps(
            {
                "version": builder.JOURNAL_VERSION,
                "token": token,
                "had_output": had_output,
                "phase": phase,
            }
        ),
        encoding="utf-8",
    )
    journal_bytes = journal.read_bytes()

    with pytest.raises(RuntimeError):
        builder._recover_publication(output)

    assert sentinel.read_bytes() == b"published generation bytes"
    assert journal.is_file()
    assert journal.read_bytes() == journal_bytes
    if stage_exists:
        assert (stage / "stage.txt").read_bytes() == b"stage bytes"
    else:
        assert not stage.exists()
    if backup_exists:
        assert (backup / "backup.txt").read_bytes() == b"backup bytes"
    else:
        assert not backup.exists()


def test_second_build_removes_readonly_files_from_owned_backup(tmp_path):
    output = tmp_path / "data" / "combined-v2"
    old_sources = make_sources(tmp_path / "old-sources")
    old_image = old_sources[0] / "class-fixture" / "bcs-1.jpg"
    old_image.chmod(stat.S_IREAD)
    try:
        build(output, old_sources)
    finally:
        old_image.chmod(stat.S_IREAD | stat.S_IWRITE)

    new_sources = make_sources(tmp_path / "new-sources")
    new_image = new_sources[0] / "class-fixture" / "bcs-1.jpg"
    new_image.write_bytes(b"second generation image")
    build(output, new_sources)

    assert (output / "val/images/bcs-1.jpg").read_bytes() == b"second generation image"
    assert_no_publication_artifacts(output)


def test_failed_stage_cleanup_handles_readonly_owned_files(tmp_path, monkeypatch):
    output = tmp_path / "data" / "combined-v2"
    output.mkdir(parents=True)
    sentinel = output / "last-good.txt"
    sentinel.write_text("previous generation", encoding="utf-8")
    sources = make_sources(tmp_path / "sources")
    readonly_image = sources[0] / "class-fixture" / "bcs-1.jpg"
    readonly_image.chmod(stat.S_IREAD)
    verify_generation = builder._verify_generation

    def fail_stage_verification(generation, final_output):
        if generation != final_output:
            raise OSError("synthetic staging verification failure")
        verify_generation(generation, final_output)

    monkeypatch.setattr(builder, "_verify_generation", fail_stage_verification)
    try:
        with pytest.raises(OSError, match="synthetic staging verification failure"):
            build(output, sources)
    finally:
        readonly_image.chmod(stat.S_IREAD | stat.S_IWRITE)

    assert sentinel.read_text(encoding="utf-8") == "previous generation"
    assert_no_publication_artifacts(output)


def test_owned_cleanup_does_not_suppress_non_readonly_permission_failure(
    tmp_path, monkeypatch
):
    output = tmp_path / "data" / "combined-v2"
    output.parent.mkdir(parents=True)
    stage = builder._generation_path(output, "stage", "f" * 32)
    stage.mkdir()
    payload = stage / "payload.bin"
    payload.write_bytes(b"owned stage bytes")
    permission_error = PermissionError("synthetic open-handle failure")

    def fail_delete(path, *, onexc):
        onexc(builder.os.unlink, str(payload), permission_error)

    monkeypatch.setattr(builder.shutil, "rmtree", fail_delete)
    with pytest.raises(PermissionError, match="synthetic open-handle failure"):
        builder._remove_generation(stage)

    assert payload.read_bytes() == b"owned stage bytes"


def test_owned_cleanup_refuses_nested_symlink_without_following_it(tmp_path):
    output = tmp_path / "data" / "combined-v2"
    output.parent.mkdir(parents=True)
    stage = builder._generation_path(output, "stage", "d" * 32)
    stage.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("outside bytes", encoding="utf-8")
    link = stage / "external"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(RuntimeError, match="links or junctions"):
        builder._remove_generation(stage)

    assert sentinel.read_text(encoding="utf-8") == "outside bytes"
    assert link.is_symlink()

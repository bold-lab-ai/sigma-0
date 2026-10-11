"""Tests for ``lobpipeline.storage.sqfs``.

Split in two. The path-and-digest logic is pure Python and always runs. The
build/mount/verify round trip needs ``mksquashfs`` and ``squashfuse``, which are
not on a CI runner, so those tests skip when the tools are absent and run on the
cluster.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from lobpipeline.storage import sqfs


# ---------------------------------------------------------------------------
# pure logic: no squashfs tools required
# ---------------------------------------------------------------------------

def test_image_and_sidecar_names_follow_the_directory():
    assert sqfs.image_for("/a/b/inference") == Path("/a/b/inference.sqfs")
    assert sqfs.overlay_for("/a/b/inference") == Path("/a/b/inference.overlay")
    assert sqfs.sidecar_path("/a/b/inference.sqfs") == Path("/a/b/index_inference.json")


def test_family_collapses_digit_runs():
    counts = sqfs.family_counts(
        "/unused",
        [
            "data_gen/GOOG_2026-01-02_message_real_id_1_gen_id_0.csv",
            "data_gen/GOOG_2026-01-02_message_real_id_2_gen_id_0.csv",
            "data_gen/GOOG_2026-01-02_audit_real_id_1_gen_id_0.npz",
            "sample_indices_rank0.json",
        ],
    )
    assert counts == {
        "data_gen/GOOG_#-#-#_audit_real_id_#_gen_id_#.npz": 1,
        "data_gen/GOOG_#-#-#_message_real_id_#_gen_id_#.csv": 2,
        "sample_indices_rank#.json": 1,
    }


def test_family_counts_are_measured_not_asserted():
    """The packer must not hard-code how many files a sequence produces.

    Observed counts vary with the frozen runtime worktree (10 / 8 / 8 / 7 have
    all occurred), so the sidecar records what was there rather than checking it
    against a fixed set.
    """
    source = sqfs.family_counts("/unused", ["d/a_1.csv", "d/a_2.csv", "d/b_1.npz"])
    assert set(source) == {"d/a_#.csv", "d/b_#.npz"}


def test_rolled_digest_is_order_independent_but_content_sensitive():
    a = {"x": "1" * 64, "y": "2" * 64}
    b = {"y": "2" * 64, "x": "1" * 64}
    assert sqfs.rolled_digest(a) == sqfs.rolled_digest(b)
    c = dict(a, y="3" * 64)
    assert sqfs.rolled_digest(a) != sqfs.rolled_digest(c)


def test_rolled_digest_matches_injection_verification_construction():
    """Same construction as ``injection_verification.arm_digest``.

    Kept identical on purpose so an image's digest can be compared directly with
    an arm digest recorded by the injection audit.
    """
    injection = pytest.importorskip("lobpipeline.injection_verification")
    digests = {"data_gen/a.csv": "a" * 64, "data_real/b.csv": "b" * 64}
    assert sqfs.rolled_digest(digests) == injection.arm_digest(digests)


def test_default_processors_is_bounded(monkeypatch):
    """A GH200 node reports 288 CPUs; mksquashfs's own default fails there."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "288")
    assert sqfs.default_processors() == sqfs.MAX_PROCESSORS
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert sqfs.default_processors() == 4
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    assert 1 <= sqfs.default_processors() <= sqfs.MAX_PROCESSORS


def test_list_rel_files_skips_symlinks(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "real.csv").write_text("a\n")
    (tmp_path / "link.csv").symlink_to(tmp_path / "sub" / "real.csv")
    assert sqfs.list_rel_files(tmp_path) == ["sub/real.csv"]


def test_resolve_missing_raises_by_default(tmp_path):
    with pytest.raises(sqfs.SqfsError):
        with sqfs.resolve(tmp_path / "nothing_here"):
            pass


def test_resolve_missing_can_pass_through(tmp_path):
    target = tmp_path / "nothing_here"
    with sqfs.resolve(target, require=False) as resolved:
        assert resolved == target


def test_resolve_prefers_an_existing_directory(tmp_path):
    """A directory wins over an image beside it: no mount, no cost.

    This is what keeps ``resolve`` free for every caller that has not been
    packed yet, so the same one-line change is correct before and after.
    """
    directory = tmp_path / "inference"
    directory.mkdir()
    sqfs.image_for(directory).write_bytes(b"not really an image")
    with sqfs.resolve(directory) as resolved:
        assert resolved == directory


def test_verify_reports_a_content_mismatch(tmp_path, monkeypatch):
    """A truncated file must be named, not absorbed.

    Stubs the mount so the comparison logic is exercised without squashfs. The
    real failure this stands in for is the recorded mksquashfs truncation, which
    LOB-Bench's ``on_bad_lines='skip'`` reader would otherwise swallow.
    """
    source = tmp_path / "src"
    (source / "data_gen").mkdir(parents=True)
    (source / "data_gen" / "a.csv").write_text("1,2,3\n4,5,6\n")
    truncated = tmp_path / "mnt"
    (truncated / "data_gen").mkdir(parents=True)
    (truncated / "data_gen" / "a.csv").write_text("1,2,3\n")

    import contextlib

    @contextlib.contextmanager
    def fake_mount(image, **kwargs):
        yield truncated

    monkeypatch.setattr(sqfs, "mount", fake_mount)
    image = tmp_path / "src.sqfs"
    image.write_bytes(b"stub")
    with pytest.raises(sqfs.VerificationError) as caught:
        sqfs.verify(source, image)
    assert "content mismatch   : 1" in str(caught.value)
    assert "data_gen/a.csv" in str(caught.value)


def test_verify_reports_a_missing_file(tmp_path, monkeypatch):
    source = tmp_path / "src"
    source.mkdir()
    (source / "a.csv").write_text("x\n")
    (source / "b.csv").write_text("y\n")
    partial = tmp_path / "mnt"
    partial.mkdir()
    (partial / "a.csv").write_text("x\n")

    import contextlib

    @contextlib.contextmanager
    def fake_mount(image, **kwargs):
        yield partial

    monkeypatch.setattr(sqfs, "mount", fake_mount)
    image = tmp_path / "src.sqfs"
    image.write_bytes(b"stub")
    with pytest.raises(sqfs.VerificationError) as caught:
        sqfs.verify(source, image)
    assert "missing from image : 1" in str(caught.value)


def test_verify_sidecar_refuses_a_sidecar_without_digests(tmp_path):
    image = tmp_path / "inference.sqfs"
    image.write_bytes(b"stub")
    sqfs.sidecar_path(image).write_text(json.dumps({"n_files": 3}))
    with pytest.raises(sqfs.SqfsError, match="no per-file digests"):
        sqfs.verify_sidecar(image)


# ---------------------------------------------------------------------------
# round trip: needs the squashfs tools
# ---------------------------------------------------------------------------

def _tools_present() -> bool:
    for name in ("mksquashfs", "squashfuse", "fusermount"):
        try:
            sqfs.tool(name)
        except sqfs.SqfsError:
            return False
    return True


needs_tools = pytest.mark.skipif(
    not _tools_present(), reason="mksquashfs/squashfuse not installed"
)


def _fixture_tree(root: Path, n: int = 4) -> Path:
    source = root / "inference"
    for sub in ("data_gen", "data_real", "data_cond"):
        (source / sub).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (source / sub / f"GOOG_2026-01-02_message_real_id_{i}.csv").write_text(
                "".join(f"{i},{j},1.5\n" for j in range(8)), encoding="utf-8"
            )
    (source / "sample_indices_rank0.json").write_text('{"idx": [0, 1]}', encoding="utf-8")
    return source


@needs_tools
def test_build_verify_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    source = _fixture_tree(tmp_path)
    result = sqfs.build(source)

    assert result.n_files == 13
    assert result.image.is_file()
    assert result.sidecar.is_file()
    sidecar = json.loads(result.sidecar.read_text())
    assert sidecar["n_files"] == 13
    assert len(sidecar["files"]) == 13
    assert sidecar["tree_digest"] == result.tree_digest

    assert sqfs.verify(source, result.image)["n_files"] == 13
    assert sqfs.verify_sidecar(result.image)["tree_digest"] == result.tree_digest


@needs_tools
def test_a_rebuilt_image_replaces_nothing_by_deleting(tmp_path, monkeypatch):
    """Rebuilding renames the old image aside; this repository forbids ``rm``."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    source = _fixture_tree(tmp_path)
    first = sqfs.build(source)
    sqfs.build(source)
    aside = [p.name for p in tmp_path.iterdir() if "_deprecated_" in p.name]
    assert any(name.startswith(first.image.name) for name in aside), aside


@needs_tools
def test_resolve_mounts_an_image_when_the_directory_is_gone(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    source = _fixture_tree(tmp_path)
    expected = sqfs.tree_digests(source)
    result = sqfs.build(source)
    source.rename(source.with_name("inference_moved_aside"))

    with sqfs.resolve(source) as resolved:
        assert resolved != source
        assert sqfs.tree_digests(resolved) == expected
    assert not resolved.exists(), "the mountpoint must be cleaned up on exit"


@needs_tools
def test_overlay_excludes_and_overrides_without_rebuilding(tmp_path, monkeypatch):
    """The 'small exclusion or override' asked for in issue #72.

    Checked through ``glob``, because that is what ``Simple_Loader`` uses: an
    excluded record has to disappear from the glob, not merely fail to open.
    """
    import glob as globmod

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    source = _fixture_tree(tmp_path)
    sqfs.build(source)
    source.rename(source.with_name("inference_moved_aside"))

    sqfs.overlay_override(source, "data_gen/GOOG_2026-01-02_message_real_id_1.csv", "NEW\n")
    sqfs.overlay_exclude(source, "data_gen/GOOG_2026-01-02_message_real_id_3.csv")

    with sqfs.resolve(source) as resolved:
        seen = sorted(
            os.path.basename(p)
            for p in globmod.glob(str(resolved / "data_gen" / "*message*.csv"))
        )
        assert len(seen) == 3, seen
        assert "GOOG_2026-01-02_message_real_id_3.csv" not in seen
        assert (
            resolved / "data_gen" / "GOOG_2026-01-02_message_real_id_1.csv"
        ).read_text() == "NEW\n"
        assert (
            resolved / "data_gen" / "GOOG_2026-01-02_message_real_id_0.csv"
        ).read_text().startswith("0,0,1.5")


@needs_tools
def test_cat_and_listing_need_no_mount(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    source = _fixture_tree(tmp_path)
    result = sqfs.build(source)
    assert sorted(sqfs.listing(result.image)) == sqfs.list_rel_files(source)
    assert sqfs.cat(result.image, "sample_indices_rank0.json") == b'{"idx": [0, 1]}'


@needs_tools
def test_content_corruption_is_caught_by_the_per_file_pass(tmp_path, monkeypatch):
    """Flip a byte inside the image and confirm the comparison notices.

    Either outcome counts: a digest mismatch, or ``EIO`` when squashfuse refuses
    to decompress the damaged block. What must not happen is a clean read of
    different bytes, because LOB-Bench's ``on_bad_lines='skip'`` reader would
    then absorb it into a shorter DataFrame and move the score silently.
    """
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    source = _fixture_tree(tmp_path, n=2)
    result = sqfs.build(source)
    data = bytearray(result.image.read_bytes())
    data[200] ^= 0xFF  # inside the compressed data region, not the padding
    damaged = result.image.with_name("damaged.sqfs")
    damaged.write_bytes(bytes(data))
    with pytest.raises((sqfs.SqfsError, OSError)):
        sqfs.verify(source, damaged)


@needs_tools
def test_tail_truncation_is_invisible_to_content_but_caught_by_the_image_hash(
    tmp_path, monkeypatch
):
    """An image can lose bytes and still read identically, because of padding.

    ``mksquashfs`` pads to a 4 KiB boundary. On this fixture the image is 4,096
    bytes with the last meaningful byte near 633, so chopping the tail removes
    only zeros and every file still reads correctly. That is why the sidecar
    also records the whole-image sha256: the two checks answer different
    questions and neither subsumes the other.
    """
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    source = _fixture_tree(tmp_path, n=2)
    result = sqfs.build(source)
    data = result.image.read_bytes()
    assert len(data) % 4096 == 0, "image is expected to be padded to a 4 KiB boundary"

    result.image.write_bytes(data[: len(data) - 64])
    assert sqfs.verify(source, result.image)["n_files"] == 7  # content unaffected
    with pytest.raises(sqfs.VerificationError, match="image sha256 differs"):
        sqfs.verify_sidecar(result.image)


# ---------------------------------------------------------------------------
# the property the packed read path depends on
# ---------------------------------------------------------------------------

def test_staging_links_survive_a_mountpoint_that_moved(tmp_path):
    """LOBBench staging must re-point when the previous mountpoint is gone.

    ``sqfs.resolve`` mounts under ``$TMPDIR`` at a fresh path each run, so the
    three ``bench_data/.../data_*`` symlinks that ``stage_inference`` creates
    point at a directory that no longer exists by the next run. If staging
    refused to replace those, a packed tree could be scored once and never
    again, and mounting on the original Lustre path would become mandatory.

    Measured here rather than argued from the source: a dangling symlink is
    replaced, and so is a live one pointing elsewhere. Only a real directory in
    the way is refused, which is what the error message says.
    """
    from lobpipeline.lob_bench.scoring import _replace_directory_link

    stage = tmp_path / "bench_data" / "model" / "GOOG" / "2026-01"
    stage.mkdir(parents=True)
    fresh = tmp_path / "new_mount" / "data_gen"
    fresh.mkdir(parents=True)

    # a mountpoint that has been unmounted and removed
    (stage / "data_gen").symlink_to(tmp_path / "gone" / "data_gen", target_is_directory=True)
    assert (stage / "data_gen").is_symlink() and not (stage / "data_gen").is_dir()
    _replace_directory_link(stage / "data_gen", fresh)
    assert os.readlink(stage / "data_gen") == str(fresh)

    # a live link pointing at some other directory
    other = tmp_path / "other" / "data_cond"
    other.mkdir(parents=True)
    (stage / "data_cond").symlink_to(other, target_is_directory=True)
    _replace_directory_link(stage / "data_cond", fresh)
    assert os.readlink(stage / "data_cond") == str(fresh)

    # a real directory is still refused
    (stage / "data_real").mkdir()
    with pytest.raises(FileExistsError, match="non-symlink"):
        _replace_directory_link(stage / "data_real", fresh)


def test_find_image_accepts_every_name_in_use(tmp_path):
    """Three suffixes already exist in this project and all must resolve.

    ``package_run.py`` writes ``.squashfs``; the trees packed by job 6423589 are
    ``.sqsh``; this module writes ``.sqfs``. Recognising only the one we write
    would mean `resolve` could not be pointed at an image somebody else built,
    which is exactly what happened the first time it was used for real.
    """
    for suffix in sqfs.IMAGE_SUFFIXES:
        directory = tmp_path / f"tree{suffix.replace('.', '_')}"
        image = directory.with_name(directory.name + suffix)
        image.write_bytes(b"stub")
        assert sqfs.find_image(directory) == image
        assert sqfs.find_image(image) == image
    assert sqfs.find_image(tmp_path / "absent") is None


def test_sidecar_name_strips_any_image_suffix():
    assert sqfs.sidecar_path("/a/inference.sqfs") == Path("/a/index_inference.json")
    assert sqfs.sidecar_path("/a/inference.squashfs") == Path("/a/index_inference.json")
    assert sqfs.sidecar_path("/a/inference.sqsh") == Path("/a/index_inference.json")

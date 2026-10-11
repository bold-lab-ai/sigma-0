"""Read and write inference trees as a single SquashFS image.

Why this module exists
----------------------
One LOB-Bench generation writes 25,092 files (3,136 sequences x 8 files, plus
four rank index files) into ``<evaluation>/inference/``.  On a shared Lustre
filesystem the scarce resource is not bytes but inodes: the project quota is
51,200,000 files, one generation costs 25,092 of them, and every one of those
files is also a metadata operation on the MDT at scoring time (``Simple_Loader``
issues roughly 12,545 ``glob`` calls against that tree).

A SquashFS image turns the whole tree into one file.  Mounted read-only with
``squashfuse`` it is indistinguishable from a directory to any consumer that
takes a path string, which is what every reader in this repository does, so the
scorer itself needs no change at all.

The one hazard, and why verification is mandatory
-------------------------------------------------
``src/base_model/data/lobster_dataloader.py`` records a real incident: in the
2025-08 SP500 shard about 0.1% of files were **silently truncated** by
``mksquashfs``.  The index header claimed N rows and the content held fewer.
That dataloader survives it because it cross-checks shapes against
``index.json``.

LOB-Bench has no such backstop and reads in the opposite direction:
``data_loading.py`` parses with ``pandas.read_csv(on_bad_lines="skip")`` and
``polars.read_csv(ignore_errors=True, truncate_ragged_lines=True)``.  A
truncated tail is absorbed into a slightly shorter DataFrame with no error at
all.  That is the only path by which this change could move a benchmark number
without anything going red, so :func:`verify` compares every file byte-for-byte
via sha256 and there is deliberately no flag to skip it.

Layout on disk
--------------
::

    <evaluation>/inference.sqfs          the image, one inode
    <evaluation>/index_inference.json    sidecar: per-file sha256 + counts
    <evaluation>/inference.overlay/      optional, see `resolve`

The optional overlay directory lets a few records be replaced or excluded
without rebuilding the image, using ``fuse-overlayfs`` with the mounted image as
the lower layer.  A whiteout entry excludes a file; a regular file with the same
relative path replaces it.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

__all__ = [
    "SqfsError",
    "VerificationError",
    "BuildResult",
    "build",
    "verify",
    "verify_sidecar",
    "mount",
    "resolve",
    "overlay_for",
    "overlay_exclude",
    "overlay_override",
    "cat",
    "listing",
    "sidecar_path",
    "image_for",
    "find_image",
    "tool",
]

DEFAULT_COMPRESSION = "zstd"
DEFAULT_BLOCK = "1M"
IMAGE_SUFFIX = ".sqfs"
# What we write is always .sqfs, but images already on disk in this project use
# three different names: package_run.py writes .squashfs, and the trees packed by
# job 6423589 are .sqsh. Recognising all three is what lets `resolve` be pointed
# at an image somebody else built.
IMAGE_SUFFIXES = (".sqfs", ".squashfs", ".sqsh")
OVERLAY_SUFFIX = ".overlay"

# ``squashfuse``, ``fusermount`` and ``fuse-overlayfs`` are system binaries on
# Isambard-AI.  ``mksquashfs`` and ``unsquashfs`` are not installed as rpms and
# live only inside a conda prefix, which differs per user, so nothing here may
# name one.  Resolution order below is: explicit override, PATH, whichever conda
# prefix the current environment points at, then the system directories.
TOOL_DIR_ENV = "SIGMA0_SQUASHFS_BIN"


def _candidate_dirs() -> list[str]:
    dirs: list[str] = []
    override = os.environ.get(TOOL_DIR_ENV, "")
    dirs.extend(part for part in override.split(os.pathsep) if part)
    for prefix_var in ("CONDA_PREFIX", "MAMBA_ROOT_PREFIX", "CONDA_ROOT"):
        prefix = os.environ.get(prefix_var)
        if prefix:
            dirs.append(os.path.join(prefix, "bin"))
    executable_dir = os.path.dirname(os.path.realpath(sys.executable))
    if executable_dir:
        dirs.append(executable_dir)
    dirs.extend(("/usr/bin", "/bin"))
    seen: set[str] = set()
    return [d for d in dirs if not (d in seen or seen.add(d))]


class SqfsError(RuntimeError):
    """A squashfs tool was missing, or an image failed to build or mount."""


class VerificationError(SqfsError):
    """An image did not reproduce its source tree byte-for-byte."""


def tool(name: str) -> str:
    """Absolute path to a squashfs helper binary.

    Raises ``SqfsError`` rather than letting ``subprocess`` fail later with a
    bare ``FileNotFoundError`` that does not say which tool was missing.
    """
    found = shutil.which(name)
    if found:
        return found
    searched = _candidate_dirs()
    for directory in searched:
        candidate = os.path.join(directory, name)
        if os.access(candidate, os.X_OK):
            return candidate
    raise SqfsError(
        f"{name} not found on PATH nor in {searched}. "
        "mksquashfs and unsquashfs are not installed system-wide on this "
        f"cluster; point {TOOL_DIR_ENV} at a conda prefix's bin directory that "
        "has them, or prepend that directory to PATH."
    )


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def image_for(directory: str | os.PathLike[str]) -> Path:
    """The image path that would stand in for ``directory``."""
    path = Path(directory)
    return path.with_name(path.name + IMAGE_SUFFIX)


def find_image(path: str | os.PathLike[str]) -> Path | None:
    """The image standing in for ``path``, whatever it is named.

    ``path`` may already be an image, or a directory name whose image sits
    beside it under any of :data:`IMAGE_SUFFIXES`. Returns ``None`` when there
    is none.
    """
    path = Path(path)
    if path.is_file() and path.suffix in IMAGE_SUFFIXES:
        return path
    for suffix in IMAGE_SUFFIXES:
        candidate = path.with_name(path.name + suffix)
        if candidate.is_file():
            return candidate
    return None


def overlay_for(directory: str | os.PathLike[str]) -> Path:
    """The optional overlay directory that would be layered over the image."""
    path = Path(directory)
    return path.with_name(path.name + OVERLAY_SUFFIX)


def sidecar_path(image: str | os.PathLike[str]) -> Path:
    """The sidecar JSON that records what went into ``image``.

    ``inference.sqfs`` -> ``index_inference.json``, matching the naming already
    used by ``lobpipeline.pipeline.package_run``.
    """
    image = Path(image)
    stem = image.name
    for suffix in IMAGE_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    else:
        stem = image.stem
    return image.with_name(f"index_{stem}.json")


# ---------------------------------------------------------------------------
# digests
# ---------------------------------------------------------------------------

def sha256_file(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def list_rel_files(root: str | os.PathLike[str]) -> list[str]:
    """Every regular file under ``root``, as sorted relative paths.

    One ``scandir`` per directory and no symlink following.  The trees this
    module packs are two levels deep (``data_gen/``, ``data_real/``,
    ``data_cond/`` plus a handful of top-level JSON files), so this is bounded
    and does not become the recursive walk that CLAUDE.md forbids on Lustre.
    """
    root = Path(root)
    out: list[str] = []
    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    out.append(os.path.relpath(entry.path, root))
    out.sort()
    return out


def tree_digests(root: str | os.PathLike[str]) -> dict[str, str]:
    """``{relative path: sha256}`` for every file under ``root``."""
    root = Path(root)
    return {rel: sha256_file(root / rel) for rel in list_rel_files(root)}


def rolled_digest(digests: dict[str, str]) -> str:
    """One hash standing for a whole ``{rel: sha256}`` map.

    Same construction as ``lobpipeline.injection_verification.arm_digest`` so
    the two can be compared directly.
    """
    accumulator = hashlib.sha256()
    for rel in sorted(digests):
        accumulator.update(f"{rel}:{digests[rel]}\n".encode())
    return accumulator.hexdigest()


MAX_PROCESSORS = 16


def default_processors() -> int:
    """How many compressor threads ``mksquashfs`` may start.

    Its own default is one thread per visible CPU.  A GH200 node reports 288, and
    on a login node -- where the per-user cgroup caps memory at 4 GiB and the
    process count is limited -- that default fails outright with
    ``FATAL ERROR: Failed to create thread``.  Prefer the Slurm allocation's
    share, fall back to the CPU affinity mask, and cap it: compression here is
    I/O-bound on a few hundred MB, so extra threads buy little and each one
    holds a block buffer.
    """
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        raw = os.environ.get(name)
        if raw and raw.isdigit() and int(raw) > 0:
            return max(1, min(int(raw), MAX_PROCESSORS))
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        available = os.cpu_count() or 1
    return max(1, min(available, MAX_PROCESSORS))


def _family(name: str) -> str:
    """Collapse a filename to its family by replacing digit runs with ``#``.

    How many files a sequence produces is not a constant: it varies with the
    frozen runtime worktree (10 / 8 / 8 / 7 have all been observed).  Recording
    the families measured at build time, rather than asserting a fixed set,
    keeps the packer honest about trees it has not seen.
    """
    return re.sub(r"\d+", "#", name)


def family_counts(root: str | os.PathLike[str], rels: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rel in rels:
        parent = os.path.dirname(rel)
        key = f"{parent}/{_family(os.path.basename(rel))}" if parent else _family(rel)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

@dataclass
class BuildResult:
    image: Path
    sidecar: Path
    n_files: int
    source_bytes: int
    image_bytes: int
    build_seconds: float
    verify_seconds: float = 0.0
    families: dict[str, int] = field(default_factory=dict)
    tree_digest: str = ""

    @property
    def compression_ratio(self) -> float:
        return (self.source_bytes / self.image_bytes) if self.image_bytes else float("nan")

    @property
    def inodes_before(self) -> int:
        return self.n_files

    @property
    def inodes_after(self) -> int:
        return 2  # the image and its sidecar


def _deprecate(path: Path) -> Path | None:
    """Move ``path`` aside with a UTC stamp.

    This repository forbids ``rm``: renaming is reversible and deletion is not,
    and the judgement "this one is safe to delete" is exactly where mistakes
    become unrecoverable.  Returns the new path, or ``None`` if there was
    nothing there.
    """
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    aside = path.with_name(f"{path.name}_deprecated_{stamp}")
    path.rename(aside)
    return aside


def build(
    source: str | os.PathLike[str],
    image: str | os.PathLike[str] | None = None,
    *,
    compression: str = DEFAULT_COMPRESSION,
    block: str = DEFAULT_BLOCK,
    processors: int | None = None,
    write_sidecar: bool = True,
    file_digests: bool = True,
    extra_metadata: dict | None = None,
) -> BuildResult:
    """Pack ``source`` into a SquashFS image and write its sidecar.

    ``image`` defaults to ``<source>.sqfs``.  An existing image is renamed aside
    with a UTC stamp rather than removed.  This only builds; call :func:`verify`
    before trusting the result.
    """
    source = Path(source)
    if not source.is_dir():
        raise SqfsError(f"source is not a directory: {source}")
    image = Path(image) if image is not None else image_for(source)
    image.parent.mkdir(parents=True, exist_ok=True)
    _deprecate(image)

    rels = list_rel_files(source)
    source_bytes = sum(os.path.getsize(source / rel) for rel in rels)

    command = [
        tool("mksquashfs"),
        str(source),
        str(image),
        "-comp",
        compression,
        "-b",
        block,
        "-noappend",
        "-no-progress",
        "-no-recovery",
        "-processors",
        str(processors or default_processors()),
    ]
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True)
    build_seconds = time.monotonic() - started
    if completed.returncode != 0:
        raise SqfsError(
            f"mksquashfs failed ({completed.returncode}) for {source}:\n"
            f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )

    digests = tree_digests(source) if file_digests else {}
    result = BuildResult(
        image=image,
        sidecar=sidecar_path(image),
        n_files=len(rels),
        source_bytes=source_bytes,
        image_bytes=image.stat().st_size,
        build_seconds=build_seconds,
        families=family_counts(source, rels),
        tree_digest=rolled_digest(digests) if digests else "",
    )

    if write_sidecar:
        payload = {
            "image": image.name,
            "source_name": source.name,
            "compression": compression,
            "block_size": block,
            "n_files": result.n_files,
            "source_bytes": source_bytes,
            "image_bytes": result.image_bytes,
            "image_sha256": sha256_file(image),
            "families": result.families,
            "tree_digest": result.tree_digest,
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mksquashfs": _tool_version("mksquashfs"),
        }
        if extra_metadata:
            payload.update(extra_metadata)
        if file_digests:
            payload["files"] = digests
        _deprecate(result.sidecar)
        result.sidecar.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def _tool_version(name: str) -> str:
    try:
        out = subprocess.run(
            [tool(name), "-version"], capture_output=True, text=True, timeout=30
        )
        return (out.stdout or out.stderr).splitlines()[0].strip()
    except Exception:  # pragma: no cover - version reporting must never fail a build
        return "unknown"


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def verify(
    source: str | os.PathLike[str],
    image: str | os.PathLike[str] | None = None,
    *,
    expected_digests: dict[str, str] | None = None,
) -> dict:
    """Compare every file in ``image`` against ``source`` by sha256.

    Raises :class:`VerificationError` on the first structural difference, or
    with the full list of differing files.  There is no bypass flag: see the
    module docstring for the truncation incident this guards.
    """
    image = Path(image) if image is not None else image_for(source)
    source_digests = expected_digests if expected_digests is not None else tree_digests(source)
    started = time.monotonic()
    with mount(image) as mountpoint:
        mounted_digests = tree_digests(mountpoint)
    seconds = time.monotonic() - started

    missing = sorted(set(source_digests) - set(mounted_digests))
    extra = sorted(set(mounted_digests) - set(source_digests))
    differing = sorted(
        rel
        for rel in set(source_digests) & set(mounted_digests)
        if source_digests[rel] != mounted_digests[rel]
    )
    if missing or extra or differing:
        raise VerificationError(
            f"squashfs image does not reproduce its source tree: {image}\n"
            f"  missing from image : {len(missing)} e.g. {missing[:5]}\n"
            f"  extra in image     : {len(extra)} e.g. {extra[:5]}\n"
            f"  content mismatch   : {len(differing)} e.g. {differing[:5]}\n"
            "This is the mksquashfs silent-truncation failure mode; the image "
            "must not be used for scoring."
        )
    return {
        "n_files": len(source_digests),
        "tree_digest": rolled_digest(source_digests),
        "verify_seconds": seconds,
    }


def verify_sidecar(image: str | os.PathLike[str], *, check_image_hash: bool = True) -> dict:
    """Re-check an image against the per-file digests recorded in its sidecar.

    Used when the source tree is gone (archived, or never on Lustre in the first
    place).  Requires the sidecar to have been written with ``file_digests``.

    The whole-image sha256 is checked first, and it is not redundant with the
    per-file pass.  ``mksquashfs`` pads an image out to a 4 KiB boundary, so a
    tail truncation can remove only padding: measured on a small fixture, the
    image was 4,096 bytes with the last meaningful byte at 633, and chopping 64
    bytes off the end changed nothing any reader could see.  Content bytes are a
    different matter and the per-file pass does catch those, either as a digest
    mismatch or as ``EIO`` from ``squashfuse``.  The two checks cover different
    damage.
    """
    image = Path(image)
    sidecar = sidecar_path(image)
    if not sidecar.is_file():
        raise SqfsError(f"no sidecar next to {image}: expected {sidecar}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))

    if check_image_hash and payload.get("image_sha256"):
        actual = sha256_file(image)
        if actual != payload["image_sha256"]:
            raise VerificationError(
                f"image sha256 differs from its sidecar: {image}\n"
                f"  sidecar: {payload['image_sha256']}\n"
                f"  on disk: {actual}\n"
                f"  recorded {payload.get('image_bytes')} bytes, found {image.stat().st_size}"
            )

    digests = payload.get("files")
    if not digests:
        raise SqfsError(
            f"{sidecar} carries no per-file digests; it was written with "
            "file_digests=False and cannot be re-verified without the source."
        )
    return verify(source=image.parent, image=image, expected_digests=digests)


# ---------------------------------------------------------------------------
# mount
# ---------------------------------------------------------------------------

def _is_stale_mount(path: Path) -> bool:
    """True when ``path`` is a FUSE mount whose server has died.

    A killed scoring job leaves the mountpoint behind; touching it then raises
    ``ENOTCONN`` instead of listing anything, and a later ``squashfuse`` onto
    the same path fails.  Detect and clear it rather than inheriting it.
    """
    try:
        os.listdir(path)
    except OSError as exc:
        return exc.errno in (errno.ENOTCONN, errno.ENOTDIR)
    return False


def _unmount(path: Path) -> None:
    for command in (
        [tool("fusermount"), "-u", str(path)],
        [tool("fusermount"), "-uz", str(path)],
    ):
        try:
            if subprocess.run(command, capture_output=True, text=True).returncode == 0:
                return
        except SqfsError:
            return


@contextlib.contextmanager
def mount(
    image: str | os.PathLike[str],
    *,
    mount_root: str | os.PathLike[str] | None = None,
    use_ll: bool = True,
) -> Iterator[Path]:
    """Mount ``image`` read-only and yield the mountpoint.

    Two details are load-bearing, both learned from real bugs in this
    repository's shell helpers:

    * The mountpoint is unique **per call**, from ``mkdtemp``, not per job.
      Under ``srun --overlap`` several steps share one ``SLURM_JOB_ID``, so a
      job-derived name lets two concurrent callers unmount each other.
    * Only a mount this call actually created is unmounted on the way out
      (the ``WE_MOUNTED`` discipline), so an inherited mount is left alone.
    """
    image = Path(image)
    if not image.is_file():
        raise SqfsError(f"squashfs image not found: {image}")
    root = Path(mount_root) if mount_root is not None else Path(
        os.environ.get("TMPDIR", tempfile.gettempdir())
    )
    root.mkdir(parents=True, exist_ok=True)
    mountpoint = Path(tempfile.mkdtemp(prefix=f"sqfs_{image.stem}_", dir=str(root)))
    if _is_stale_mount(mountpoint):
        _unmount(mountpoint)

    binary = "squashfuse_ll" if use_ll and shutil.which("squashfuse_ll") else "squashfuse"
    completed = subprocess.run(
        [tool(binary), str(image), str(mountpoint)], capture_output=True, text=True
    )
    if completed.returncode != 0:
        with contextlib.suppress(OSError):
            mountpoint.rmdir()
        raise SqfsError(
            f"{binary} failed ({completed.returncode}) mounting {image}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    we_mounted = True
    try:
        yield mountpoint
    finally:
        if we_mounted:
            _unmount(mountpoint)
            with contextlib.suppress(OSError):
                mountpoint.rmdir()


@contextlib.contextmanager
def _overlay(lower: Path, upper: Path, root: Path) -> Iterator[Path]:
    """Layer ``upper`` over the read-only ``lower`` with fuse-overlayfs.

    This is the "exclude or override a few records without rebuilding the
    image" path.  A regular file in ``upper`` replaces the one in the image; a
    character-device whiteout (created by ``fuse-overlayfs`` itself, or by
    ``mknod``) hides it.  A symlink farm was the alternative and is rejected:
    it has twice put node-local inode exhaustion and silently-incomplete farms
    into this repository's history.
    """
    merged = Path(tempfile.mkdtemp(prefix="sqfs_merged_", dir=str(root)))
    work = Path(tempfile.mkdtemp(prefix="sqfs_work_", dir=str(root)))
    completed = subprocess.run(
        [
            tool("fuse-overlayfs"),
            "-o",
            f"lowerdir={lower},upperdir={upper},workdir={work}",
            str(merged),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SqfsError(
            f"fuse-overlayfs failed ({completed.returncode}) for {upper} over {lower}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        yield merged
    finally:
        _unmount(merged)
        with contextlib.suppress(OSError):
            merged.rmdir()
        shutil.rmtree(work, ignore_errors=True)


@contextlib.contextmanager
def resolve(
    path: str | os.PathLike[str],
    *,
    mount_root: str | os.PathLike[str] | None = None,
    require: bool = True,
) -> Iterator[Path]:
    """Yield a real directory for ``path``, mounting an image if that is what exists.

    This is the one-line change every consumer needs::

        with sqfs.resolve(inference_dir) as real_dir:
            ...   # downstream code unchanged

    Resolution order:

    1. ``path`` is a directory -> yielded unchanged, no mount, no cost.
    2. ``<path>.sqfs`` exists -> mounted read-only and the mountpoint yielded.
       If ``<path>.overlay/`` also exists it is layered on top.
    3. ``path`` is itself an image file -> mounted.
    4. Neither -> ``SqfsError`` when ``require``, else the original path.
    """
    path = Path(path)
    root = Path(mount_root) if mount_root is not None else Path(
        os.environ.get("TMPDIR", tempfile.gettempdir())
    )

    if path.is_dir():
        yield path
        return

    image = find_image(path)
    if image is None:
        if require:
            raise SqfsError(
                f"neither a directory nor a squashfs image: {path} "
                f"(looked for {[path.name + x for x in IMAGE_SUFFIXES]})"
            )
        yield path
        return

    overlay = overlay_for(path)
    with mount(image, mount_root=root) as mountpoint:
        if overlay.is_dir():
            root.mkdir(parents=True, exist_ok=True)
            with _overlay(mountpoint, overlay, root) as merged:
                yield merged
        else:
            yield mountpoint


# ---------------------------------------------------------------------------
# overlay authoring
# ---------------------------------------------------------------------------

def overlay_override(
    directory: str | os.PathLike[str], relative: str, data: bytes | str
) -> Path:
    """Replace one record inside an image, without rebuilding it.

    Writes ``relative`` into ``<directory>.overlay/``.  The next
    :func:`resolve` of ``directory`` yields a merged view in which that path
    reads ``data`` and everything else still comes from the image.
    """
    target = overlay_for(directory) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        target.write_text(data, encoding="utf-8")
    else:
        target.write_bytes(data)
    return target


def overlay_exclude(directory: str | os.PathLike[str], relative: str) -> Path:
    """Hide one record inside an image, without rebuilding it.

    Creates the overlay whiteout that ``fuse-overlayfs`` recognises: a
    character device with major/minor 0/0.  After this, a ``glob`` over the
    resolved directory no longer returns ``relative`` -- which is what
    ``Simple_Loader`` uses, so the record simply stops existing for scoring.
    """
    target = overlay_for(directory) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    _deprecate(target)
    completed = subprocess.run(
        [tool("mknod"), str(target), "c", "0", "0"], capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise SqfsError(
            f"could not create overlay whiteout {target}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return target


# ---------------------------------------------------------------------------
# read without mounting
# ---------------------------------------------------------------------------

def cat(image: str | os.PathLike[str], relative: str) -> bytes:
    """One file's bytes out of an image, without FUSE.

    For login nodes and notebooks, where mounting is either unavailable or not
    worth the setup.  ``unsquashfs -cat`` is not fast, so this is for a handful
    of files, not for a scoring pass.
    """
    completed = subprocess.run(
        [tool("unsquashfs"), "-cat", str(image), relative],
        capture_output=True,
    )
    if completed.returncode != 0:
        raise SqfsError(
            f"unsquashfs -cat failed for {relative} in {image}: "
            f"{completed.stderr.decode(errors='replace').strip()}"
        )
    return completed.stdout


def listing(image: str | os.PathLike[str]) -> list[str]:
    """Relative paths of every regular file in an image, without mounting."""
    completed = subprocess.run(
        [tool("unsquashfs"), "-lln", str(image)], capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise SqfsError(
            f"unsquashfs -lln failed for {image}: {completed.stderr.strip()}"
        )
    out: list[str] = []
    for line in completed.stdout.splitlines():
        if not line.startswith("-"):  # only regular files
            continue
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        name = parts[5]
        if name.startswith("squashfs-root/"):
            name = name[len("squashfs-root/") :]
        elif name == "squashfs-root":
            continue
        out.append(name)
    out.sort()
    return out

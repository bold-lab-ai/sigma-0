"""Checkpoint inspection and lightweight restore helpers for base_model."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


class CheckpointLoadError(RuntimeError):
    """Raised when a checkpoint path or checkpoint payload cannot be read."""


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """A checkpoint path normalized to manager-root + concrete step form."""

    requested_path: str
    manager_path: Path
    step: int
    step_path: Path
    metadata_path: Path
    state_path: Path


@dataclass(frozen=True)
class CheckpointArrayInfo:
    """Array metadata from an Orbax state checkpoint."""

    keypath: tuple[str, ...]
    name: str
    top_level: str
    metadata_shape: tuple[int, ...]


@dataclass(frozen=True)
class LoadedArraySummary:
    """Small proof that a TensorStore-backed array was opened and read."""

    name: str
    metadata_shape: tuple[int, ...]
    tensor_shape: tuple[int, ...]
    dtype: str


def initialize_training(run_name: str) -> dict[str, str]:
    """Return initialization metadata for a named training run."""
    if not run_name:
        raise ValueError("run_name must be non-empty")
    return {"run_name": run_name, "phase": "initialized"}


def resolve_checkpoint_path(path: str | Path, step: int | None = None) -> ResolvedCheckpoint:
    """Resolve an Orbax checkpoint manager path or a concrete step directory.

    The legacy AlphaTrade launchers use the checkpoint manager root plus
    ``RESTORE_STEP``. Human-facing notes often paste the concrete step directory
    instead. This helper accepts either form.
    """
    requested = Path(path).expanduser()
    if not requested.exists():
        raise CheckpointLoadError(f"checkpoint path does not exist: {requested}")

    step_path: Path
    manager_path: Path
    resolved_step: int

    if step is not None:
        candidate = requested / str(step)
        if candidate.is_dir():
            manager_path = requested
            step_path = candidate
        elif requested.name == str(step) and requested.is_dir():
            manager_path = requested.parent
            step_path = requested
        else:
            raise CheckpointLoadError(
                f"could not find checkpoint step {step} under {requested}"
            )
        resolved_step = int(step)
    elif (requested / "state").is_dir() and (requested / "metadata").is_dir():
        if not requested.name.isdigit():
            raise CheckpointLoadError(
                f"step directory name must be numeric when step is omitted: {requested}"
            )
        manager_path = requested.parent
        step_path = requested
        resolved_step = int(requested.name)
    else:
        step_dirs = [
            child
            for child in requested.iterdir()
            if child.is_dir()
            and child.name.isdigit()
            and (child / "state").is_dir()
            and (child / "metadata").is_dir()
        ]
        if not step_dirs:
            raise CheckpointLoadError(
                f"no numeric Orbax step directories with state/metadata under {requested}"
            )
        step_path = max(step_dirs, key=lambda child: int(child.name))
        manager_path = requested
        resolved_step = int(step_path.name)

    metadata_path = _find_metadata_file(step_path)
    state_path = step_path / "state"
    if not state_path.is_dir():
        raise CheckpointLoadError(f"checkpoint state directory missing: {state_path}")

    return ResolvedCheckpoint(
        requested_path=str(path),
        manager_path=manager_path,
        step=resolved_step,
        step_path=step_path,
        metadata_path=metadata_path,
        state_path=state_path,
    )


def load_checkpoint_metadata(path: str | Path, step: int | None = None) -> dict[str, Any]:
    """Load the JSON metadata payload for a checkpoint step."""
    resolved = resolve_checkpoint_path(path, step=step)
    try:
        return json.loads(resolved.metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointLoadError(f"invalid checkpoint metadata JSON: {resolved.metadata_path}") from exc


def checkpoint_config(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return the training config portion from known Orbax metadata layouts."""
    config = metadata.get("config")
    if isinstance(config, dict):
        return config
    custom = metadata.get("custom_metadata")
    if isinstance(custom, dict):
        return custom
    custom = metadata.get("custom")
    if isinstance(custom, dict):
        return custom
    return metadata


def load_state_metadata(path: str | Path, step: int | None = None) -> dict[str, Any]:
    """Load the Orbax state ``_METADATA`` payload for a checkpoint step."""
    resolved = resolve_checkpoint_path(path, step=step)
    state_metadata_path = resolved.state_path / "_METADATA"
    if not state_metadata_path.is_file():
        raise CheckpointLoadError(f"state metadata file missing: {state_metadata_path}")
    try:
        return json.loads(state_metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointLoadError(f"invalid state metadata JSON: {state_metadata_path}") from exc


def iter_checkpoint_arrays(
    path: str | Path,
    step: int | None = None,
    *,
    verify_shapes: bool = False,
) -> Iterator[CheckpointArrayInfo]:
    """Yield array leaves described by the Orbax state metadata.

    ``write_shape`` in the state ``_METADATA`` file is the shape of the slice a
    single host wrote. For arrays partitioned across processes (the common
    case for multi-node training checkpoints) that is a per-shard shape, not
    the array's global shape. Pass ``verify_shapes=True`` to resolve the real
    global shape via TensorStore (falls back to the on-disk ``write_shape``
    when TensorStore is unavailable or the array cannot be opened, e.g. in
    synthetic test checkpoints).
    """
    resolved = resolve_checkpoint_path(path, step=step)
    state_metadata = load_state_metadata(resolved.step_path)
    tree_metadata = state_metadata.get("tree_metadata")
    if not isinstance(tree_metadata, dict):
        raise CheckpointLoadError("state metadata does not contain a tree_metadata map")
    use_zarr3 = bool(state_metadata.get("use_zarr3", False))

    for raw_keypath, payload in tree_metadata.items():
        keypath = _parse_keypath(raw_keypath)
        if not keypath:
            continue
        value_metadata = payload.get("value_metadata", {}) if isinstance(payload, dict) else {}
        if value_metadata.get("skip_deserialize"):
            # Not a real stored array (e.g. a Muon inner-state slot the
            # optimizer left as None for this param) - nothing to count.
            continue
        name = ".".join(keypath)
        shape = tuple(int(dim) for dim in value_metadata.get("write_shape", []))
        if verify_shapes:
            true_shape = _resolve_true_shape(resolved.state_path, name, use_zarr3)
            if true_shape is not None:
                shape = true_shape
        yield CheckpointArrayInfo(
            keypath=keypath,
            name=name,
            top_level=keypath[0],
            metadata_shape=shape,
        )


def summarize_checkpoint(
    path: str | Path,
    step: int | None = None,
    *,
    verify_shapes: bool = False,
) -> dict[str, Any]:
    """Return a summary of a checkpoint.

    ``verify_shapes=True`` resolves each array's true global shape via
    TensorStore instead of trusting the (possibly per-shard) on-disk
    ``write_shape``; see :func:`iter_checkpoint_arrays`.
    """
    resolved = resolve_checkpoint_path(path, step=step)
    metadata = load_checkpoint_metadata(resolved.step_path)
    config = checkpoint_config(metadata)
    from base_model.models import resolve_architecture

    architecture = resolve_architecture(
        config.get("architecture"),
        ssm_type=config.get("ssm_type"),
        model_type=config.get("model_type"),
    )
    arrays = list(iter_checkpoint_arrays(resolved.step_path, verify_shapes=verify_shapes))
    leaf_counts = Counter(array.top_level for array in arrays)
    metadata_elements = Counter()
    for array in arrays:
        metadata_elements[array.top_level] += _num_elements(array.metadata_shape)

    return {
        "requested_path": resolved.requested_path,
        "manager_path": str(resolved.manager_path),
        "step": resolved.step,
        "step_path": str(resolved.step_path),
        "metadata_path": str(resolved.metadata_path),
        "state_path": str(resolved.state_path),
        "model": architecture,
        "d_model": config.get("d_model"),
        "n_layers": config.get("n_layers"),
        "msg_seq_len": config.get("msg_seq_len"),
        "use_book_data": config.get("use_book_data"),
        "opt_config": config.get("opt_config"),
        "leaf_counts": dict(sorted(leaf_counts.items())),
        "metadata_elements": dict(sorted(metadata_elements.items())),
    }


def sample_checkpoint_arrays(
    path: str | Path,
    step: int | None = None,
    *,
    prefixes: Sequence[str] = ("params",),
    limit: int = 3,
) -> list[LoadedArraySummary]:
    """Open and read a small number of TensorStore arrays from the checkpoint."""
    if limit < 1:
        return []
    selected: list[CheckpointArrayInfo] = []
    allowed = set(prefixes)
    for array in iter_checkpoint_arrays(path, step=step):
        if array.top_level in allowed:
            selected.append(array)
            if len(selected) >= limit:
                break

    return [_read_array_summary(path, array, step=step) for array in selected]


def _find_metadata_file(step_path: Path) -> Path:
    candidates = [
        step_path / "metadata" / "metadata",
        step_path / "metadata" / "_ROOT_METADATA",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise CheckpointLoadError(f"checkpoint metadata file missing under {step_path / 'metadata'}")


def _parse_keypath(raw_keypath: str) -> tuple[str, ...]:
    try:
        keypath = ast.literal_eval(raw_keypath)
    except (SyntaxError, ValueError) as exc:
        raise CheckpointLoadError(f"invalid Orbax keypath in state metadata: {raw_keypath}") from exc
    if isinstance(keypath, tuple):
        return tuple(str(part) for part in keypath)
    return (str(keypath),)


def _num_elements(shape: Iterable[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _open_tensorstore_array(state_path: Path, array_name: str, use_zarr3: bool) -> Any:
    """Open a checkpoint array leaf as a TensorStore handle (no data read yet)."""
    import tensorstore as ts

    driver = "zarr3" if use_zarr3 else "zarr"
    if (state_path / "manifest.ocdbt").exists():
        kvstore = {"driver": "ocdbt", "base": f"file://{state_path}", "path": array_name}
    else:
        kvstore = {"driver": "file", "path": str(state_path / array_name)}
    return ts.open(ts.Spec({"driver": driver, "kvstore": kvstore}), open=True).result()


def _resolve_true_shape(state_path: Path, array_name: str, use_zarr3: bool) -> tuple[int, ...] | None:
    """Best-effort global shape of an on-disk array, or None if unresolvable.

    Returns None (rather than raising) when TensorStore is not installed or
    the array cannot be opened, so callers can fall back to the declared
    ``write_shape`` - e.g. for synthetic checkpoints in tests that have no
    real backing store.
    """
    try:
        tensor = _open_tensorstore_array(state_path, array_name, use_zarr3)
    except Exception:
        # Best-effort: missing TensorStore, no backing store (test fixtures),
        # or any other open failure all mean "fall back to write_shape".
        return None
    return tuple(int(dim) for dim in tensor.shape)


def _read_array_summary(
    path: str | Path,
    array: CheckpointArrayInfo,
    step: int | None = None,
) -> LoadedArraySummary:
    try:
        import tensorstore  # noqa: F401  (import check; opener below does the real import)
    except ImportError as exc:
        raise CheckpointLoadError(
            "TensorStore is required for reading checkpoint arrays; use the quant Miniforge Python"
        ) from exc

    resolved = resolve_checkpoint_path(path, step=step)
    state_metadata = load_state_metadata(resolved.step_path)
    use_zarr3 = bool(state_metadata.get("use_zarr3", False))
    tensor = _open_tensorstore_array(resolved.state_path, array.name, use_zarr3)
    payload = tensor.read().result()
    return LoadedArraySummary(
        name=array.name,
        metadata_shape=array.metadata_shape,
        tensor_shape=tuple(int(dim) for dim in payload.shape),
        dtype=str(payload.dtype),
    )

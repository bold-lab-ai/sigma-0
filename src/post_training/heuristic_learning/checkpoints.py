"""Pinned world-model checkpoints for the agentic market-making closed loop.

Two frozen checkpoints play two different roles, and they are deliberately
*different steps of different runs* rather than one checkpoint doing both jobs:

    background order flow  <- best distributional fit   (LOBbench mean KS)
    price-curve signal     <- best directional accuracy (direction accuracy)

Ranking by KS and ranking by direction accuracy do not agree, so picking one
checkpoint for both roles would give up one of them.

Why this module exists at all
-----------------------------
`run/benchmarking/*.batch` already pin checkpoints the right way: an explicit
step plus a SHA256 over ``<step>/metadata/metadata``, with no ``ls | sort |
tail`` anywhere (that pattern is banned on this filesystem). This module reuses
exactly that contract so there is one convention, not two.

It differs from the batch validator in exactly one respect, on purpose: the
batch scripts additionally assert that the pinned step is the *latest* step in
the directory. Both checkpoints here are deliberately NOT the latest:

    j4559297 ... latest step is 168200, we pin 150360
    j4553948 ... latest step is 143150, we pin 120000

Later is not better here. For j4559297, step 150360 has the best LOBbench fit
(mean KS 0.0835) and sits on the train-loss plateau, while step 168200 scores
about 3% worse. So `verify()` takes ``require_latest`` and defaults it to
False, which is the whole reason a separate validator is warranted.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


CHECKPOINT_ROOT = Path(
    "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments"
    "/exp_R1_Mamba3/checkpoints"
)

# Files Orbax writes for a complete step; same set the batch validator requires.
REQUIRED_STEP_FILES = (
    "_CHECKPOINT_METADATA",
    "metadata/metadata",
    "state/_METADATA",
    "state/manifest.ocdbt",
)

# Shared architecture of both pins, read back from metadata/metadata.
# Note the family is commonly labelled "350M" but actually carries 293,283,039
# parameters; the model-zoo tag for it is "290m". Same artifact, three names.
MAMBA3_293M_ARCH = {
    "ssm_type": "mamba3",
    "d_model": 2048,
    "n_layers": 6,
    "ssm_size_base": 2048,
    "blocks": 32,
    "n_message_layers": 2,
    "msg_seq_len": 500,
    "book_depth": 500,
    "use_book_data": True,
}


class CheckpointDrift(RuntimeError):
    """Raised when a pinned checkpoint is missing, incomplete, or has changed."""


@dataclass(frozen=True)
class CheckpointPin:
    """One frozen checkpoint, pinned by content hash rather than by recency."""

    name: str
    role: str
    run_dir: str
    step: int
    metadata_sha256: str
    rationale: str
    latest_step_in_dir: int
    arch: Dict[str, object] = field(default_factory=lambda: dict(MAMBA3_293M_ARCH))

    @property
    def root(self) -> Path:
        return CHECKPOINT_ROOT / self.run_dir

    @property
    def step_dir(self) -> Path:
        return self.root / str(self.step)

    @property
    def is_latest(self) -> bool:
        return self.step == self.latest_step_in_dir


PINS: Dict[str, CheckpointPin] = {
    "bg-flow-293m-s150360": CheckpointPin(
        name="bg-flow-293m-s150360",
        role="background_order_flow",
        run_dir="j4559297_u52a0g05_4559297",
        step=150360,
        metadata_sha256=(
            "7057a6196cd1f90169a782c69280dd75ee1a8766d48e6491c9ce52a91b4a55eb"
        ),
        rationale=(
            "Best LOBbench distributional fit of the chain (mean KS 0.0835, best "
            "on KS/WS/L1 together) and on the train-loss plateau. Realistic "
            "background flow is what this role needs, so distributional fit is "
            "the right criterion. Latest step 168200 is ~3% worse."
        ),
        latest_step_in_dir=168200,
    ),
    "signal-293m-s120000": CheckpointPin(
        name="signal-293m-s120000",
        role="price_curve_signal",
        run_dir="j4553948_tz2tmn5z_4553948",
        step=120000,
        metadata_sha256=(
            "e06ce2e52b713576974e28f1d6e35596c2fb58448ba6a183faedac0fe5639334"
        ),
        rationale=(
            "Highest direction accuracy in the chain (0.5523). This role only "
            "has to say which way the mid is going, so directional accuracy is "
            "the right criterion, and it peaks at a different step than KS does."
        ),
        latest_step_in_dir=143150,
    ),
}


def resolve(name: str) -> CheckpointPin:
    """Look up a pin by name.

    Deliberately a dict lookup and not a directory scan: enumerating step
    directories to find "the latest" is the metadata-storm pattern this
    filesystem's policy forbids.
    """
    try:
        return PINS[name]
    except KeyError:
        known = ", ".join(sorted(PINS))
        raise KeyError(f"unknown checkpoint pin {name!r}; known pins: {known}") from None


def metadata_sha256(pin: CheckpointPin) -> str:
    """Hash the pinned step's metadata blob, the same file the batch scripts hash."""
    return hashlib.sha256((pin.step_dir / "metadata" / "metadata").read_bytes()).hexdigest()


def verify(pin: CheckpointPin, *, require_latest: bool = False) -> CheckpointPin:
    """Check the pinned step is present, complete, and unchanged.

    Args:
        pin: the pin to check.
        require_latest: assert the pinned step is the newest in its directory.
            Defaults to False because both pins here are intentionally not the
            newest step. Enabling it costs one directory listing, so leave it
            off unless a caller genuinely needs freshness.

    Raises:
        CheckpointDrift: on a missing directory, an incomplete step, a metadata
            hash that no longer matches, or (when requested) a stale step.
    """
    if not pin.step_dir.is_dir():
        raise CheckpointDrift(f"{pin.name}: step directory absent: {pin.step_dir}")

    missing = [rel for rel in REQUIRED_STEP_FILES if not (pin.step_dir / rel).is_file()]
    if missing:
        raise CheckpointDrift(f"{pin.name}: incomplete checkpoint; missing={missing}")

    actual = metadata_sha256(pin)
    if actual != pin.metadata_sha256:
        raise CheckpointDrift(
            f"{pin.name}: checkpoint metadata drift; "
            f"expected {pin.metadata_sha256}, found {actual}"
        )

    if require_latest:
        steps = sorted(
            int(p.name) for p in pin.root.iterdir() if p.is_dir() and p.name.isdigit()
        )
        if not steps or steps[-1] != pin.step:
            raise CheckpointDrift(
                f"{pin.name}: latest step is {steps[-1] if steps else None}, "
                f"pinned step is {pin.step}"
            )

    return pin

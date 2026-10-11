"""Validation-only checkpoint selection for model-zoo comparisons."""

from __future__ import annotations

from typing import Any


def select_checkpoints(
    results: list[dict[str, Any]],
    targets: list[float],
    tolerance: float,
) -> dict[str, Any]:
    """Select best-validation and validation-CE-matched checkpoints."""
    if not results:
        return {"best_validation": None, "iso_ce": {}}

    best = min(results, key=lambda row: (row["validation_ce"], row["step"]))
    selected: dict[str, Any] = {}
    for target in targets:
        match = min(
            results,
            key=lambda row: (abs(row["validation_ce"] - target), row["step"]),
        )
        key = format(target, ".6g")
        delta = float(match["validation_ce"] - target)
        selected[key] = {
            "target": target,
            "status": "matched" if abs(delta) <= tolerance else "outside_tolerance",
            "target_reached": bool(best["validation_ce"] <= target),
            "step": int(match["step"]),
            "validation_ce": float(match["validation_ce"]),
            "delta": delta,
        }
    return {
        "best_validation": {
            "step": int(best["step"]),
            "validation_ce": float(best["validation_ce"]),
        },
        "iso_ce": selected,
    }


def convergence_step(results: list[dict[str, Any]]) -> int | None:
    """Return the first checkpoint meeting the thesis's 5k-step/1% rule."""
    for current in results:
        candidates = [
            row for row in results if row["step"] <= current["step"] - 5_000
        ]
        if not candidates:
            continue
        previous = max(candidates, key=lambda row: row["step"])
        improvement = (
            previous["validation_ce"] - current["validation_ce"]
        ) / previous["validation_ce"]
        if improvement < 0.01:
            return int(current["step"])
    return None

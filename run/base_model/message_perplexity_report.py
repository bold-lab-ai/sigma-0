#!/usr/bin/env python3
"""Derive message-level perplexity from committed ce-by-position records.

This is a pure derivation, not a measurement: everything it needs is the
`per_field` table that `evaluate_ce_by_position.py` already writes, so it runs on
a login node in a second and cannot disagree with the record it reads. Keeping it
separate from the evaluator is deliberate -- the evaluator's source hash is pinned
in `ci/evidence/profiling.json`, and a derivation has no business invalidating the
provenance of a cluster measurement.

    python run/base_model/message_perplexity_report.py \
        --records docs/experiments/trainloss_profiling_20260826/data \
        --output artifacts/message_perplexity/report.json

`--records` takes the record tree, not a glob over it. The record set grows by
subdirectory -- `pair/`, `xtok/` -- and a one-level glob is a frozen claim about
the directory's shape: it keeps working, silently over a smaller input set, on
the day one appears.

The output carries, per record, both variants (`main` and `full`), the additive
role decomposition, and the per-field nats, so any other subset can be read off
later without touching a GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from base_model.evaluation.message_perplexity import message_perplexity_from_per_field  # noqa: E402

# Field order per encoding, so `missing_fields` can be populated without importing
# the encodings (which would pull in jax). Read from the encoding sources by
# ci/check_message_perplexity.py, which fails if this drifts.
EXPECTED_FIELDS = (
    "event_type", "direction", "price", "size",
    "delta_t_s", "delta_t_ns", "time_s", "time_ns",
    "price_ref", "size_ref", "time_s_ref", "time_ns_ref",
)


# A checkpoint evaluated on data its vocabulary does not match still produces a
# number: the evaluator does not check, so a mismatch shows up only as a loss
# many times too large.
#
# Token accuracy was the first guard and it was calibrated on the zoo, where
# every checkpoint is trained to convergence: healthy runs sat at 0.81-0.92 and
# mismatched ones at 0.07-0.30, with nothing between. That threshold then
# rejected a legitimately trained model -- a 1100-step matched-pair arm at 0.589,
# which is weak rather than mismatched. A cut calibrated on one population
# misclassifies another, and "reads the data badly" is not "does not read it".
#
# What actually separates the two is whether any field beats its own uniform
# bound. A model that has learned nothing about a field converges to the
# marginal, which sits at or below the uniform entropy; going *above* it means
# assigning the truth less mass than a coin would, which takes a systematic
# misalignment rather than weakness. Measured, as the worst field ratio per
# record:
#
#     healthy zoo checkpoints        0.37 - 0.41
#     trained but weak (1100 steps)  1.00 - 1.01   <- time_ns exactly at uniform
#     not reading the data           1.79 - 58.4
#
# 1.5 sits in the gap with margin on both sides. Accuracy is still reported,
# because it is the number a reader recognises, but it no longer gates.
MAX_CE_OVER_CEILING = 1.5
REPORTED_ACCURACY_CONTEXT = 0.60  # informational only; see above


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def summarise(record: dict[str, Any], source: Path) -> dict[str, Any]:
    checkpoint = record.get("checkpoint", {})
    slice_info = record.get("slice", {})
    tokens_per_message = sum(int(e["width"]) for e in record["per_field"])
    token_mode = str(checkpoint.get("token_mode") or f"{tokens_per_message}tok")
    expected = EXPECTED_FIELDS if token_mode == "26tok" else None
    mp = message_perplexity_from_per_field(
        record["per_field"],
        token_mode=token_mode,
        n_messages=slice_info.get("messages"),
        expected_fields=expected,
    )
    out = mp.as_dict()
    out["source_record"] = str(source.relative_to(REPO)) if source.is_relative_to(REPO) \
        else str(source)
    label = slice_info.get("label")
    out["slice_label"] = label
    # With one model per record the source path was identity enough; with a whole
    # ladder in one report every entry has to say which model it is. Strip the
    # record's own slice label rather than a fixed pair of suffixes: the drift
    # sweep writes `<model>__<label>_<label>.json`, and hardcoding "heldout" left
    # the slice glued to the model name for every one of those.
    stem = source.name[:-len(".json")] if source.name.endswith(".json") else source.name
    for suffix in (f"__{label}_{label}", f"_{label}", f"__{label}"):
        if label and stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    out["model"] = stem
    out["checkpoint"] = {
        "step": checkpoint.get("step"),
        # The whole point of this metric is that it survives a change of
        # tokenizer, so every entry has to say which one produced it. Records
        # written before the evaluator recorded it are recovered from the widths,
        # which sum to the tokens per message by construction.
        "token_mode": checkpoint.get("token_mode") or f"{tokens_per_message}tok",
        "tokens_per_message": tokens_per_message,
        "architecture": checkpoint.get("architecture"),
        "d_model": checkpoint.get("d_model"),
        "n_layers": checkpoint.get("n_layers"),
        "seed": checkpoint.get("seed"),
        "ignore_times": checkpoint.get("ignore_times"),
    }
    # The record sums per-position means; this sums per-field means times widths.
    # Two paths over the same float32 evidence: agreement to ~1e-7 is the check
    # that the derivation reads the table the way the evaluator wrote it.
    accuracy = record.get("aggregate", {}).get("acc")
    out["token_accuracy"] = accuracy
    out["accuracy_is_low"] = (None if accuracy is None
                              else bool(accuracy < REPORTED_ACCURACY_CONTEXT))
    ratios = [(e["field"], e["ce"] / e["ceiling_nats"])
              for e in record["per_field"] if e.get("ceiling_nats")]
    worst_field, worst_ratio = max(ratios, key=lambda t: t[1]) if ratios else (None, None)
    out["worst_field_over_ceiling"] = {"field": worst_field, "ratio": worst_ratio}
    out["reads_the_data"] = (None if worst_ratio is None
                             else bool(worst_ratio <= MAX_CE_OVER_CEILING))

    recorded = record.get("aggregate", {}).get("ce_per_message_nats")
    if recorded is not None:
        out["cross_check"] = {
            "record_aggregate_nats": recorded,
            "derived_full_nats": mp.full_nats,
            "relative_difference": abs(mp.full_nats - recorded) / max(recorded, 1e-12),
        }
    return out


def _record_paths(inputs: list[Path]) -> list[Path]:
    """Resolve `--records` to the files to read: directories are walked recursively.

    Sorting the union keeps the report's entry order a function of the paths alone,
    so naming the tree and naming every file individually produce the same bytes.
    """
    found: list[Path] = []
    for item in inputs:
        if item.is_dir():
            found.extend(item.rglob("*.json"))
        else:
            found.append(item)
    if not found:
        raise SystemExit(
            "--records matched no files; pass the record tree "
            "(docs/experiments/trainloss_profiling_20260826/data) or the record files themselves")
    return sorted(set(found))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", type=Path, required=True,
                        help="record files, or directories to walk recursively")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-relative-difference", type=float, default=1e-5,
                        help="fail if a derived total drifts from the record's own")
    args = parser.parse_args()

    entries = []
    for path in _record_paths(args.records):
        record = json.loads(path.read_text(encoding="utf-8"))
        entry = summarise(record, path)
        rel = entry.get("cross_check", {}).get("relative_difference")
        if rel is not None and rel > args.max_relative_difference:
            raise SystemExit(
                f"{path}: derived total differs from the record's own aggregate by "
                f"{rel:.2e} (> {args.max_relative_difference:.0e}); the per-field "
                "table and the aggregate no longer describe the same run")
        if entry.get("reads_the_data") is False:
            worst = entry["worst_field_over_ceiling"]
            raise SystemExit(
                f"{path}: field {worst['field']} scores {worst['ratio']:.2f}x its own "
                f"uniform ceiling (limit {MAX_CE_OVER_CEILING}); the model assigns the "
                "truth less mass than a uniform guess would, which no amount of "
                "undertraining produces. This checkpoint is not reading this data, so "
                "its perplexity is meaningless rather than merely poor "
                f"(token accuracy {entry['token_accuracy']:.3f}, for context)")
        entries.append(entry)

    payload = {"schema_version": 1, "metric": "message_perplexity", "entries": entries}
    _write_json(args.output, payload)

    print(f"[message-perplexity] wrote {args.output}")
    header = (f"{'model':<26}{'slice':<14}{'tok':>5}{'main nats':>11}{'main bits':>11}"
              f"{'full nats':>11}{'full bits':>11}{'ref %':>8}")
    print(header)
    for entry in entries:
        main_v, full_v = entry["variants"]["main"], entry["variants"]["full"]
        comps = entry["components_nats_per_message"]
        ref_pct = 100.0 * comps["reference"] / full_v["nats_per_message"]
        print(f"{str(entry.get('model','?')):<26}{str(entry['slice_label']):<14}"
              f"{entry['checkpoint']['tokens_per_message']:>5}"
              f"{main_v['nats_per_message']:>11.4f}{main_v['bits_per_message']:>11.4f}"
              f"{full_v['nats_per_message']:>11.4f}{full_v['bits_per_message']:>11.4f}"
              f"{ref_pct:>8.1f}")
        if entry["missing_fields"]:
            print(f"{'':<14}note: full excludes {entry['missing_fields']} "
                  "(not scored on this run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

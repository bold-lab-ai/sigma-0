# Plan: locate the remaining nats in the flat training loss

Status: plan. Implementation, numbers, and the verdict land as commits and PR
comments; the reading rules below are registered before any number is produced.

## Problem

The training loss flattens fast and barely moves with capacity. Final training loss
across the scaling family (leaderboard board A):

| params | train loss (nats/token) | wandb |
|---:|---:|---|
| 8,099,567 | 0.5802 | `zkrtl2ef` |
| 14,445,623 | 0.5804 | `8kpf89b4` |
| 22,949,247 | 0.5735 | `edllydbz` |
| 33,610,439 | 0.5623 | `mquj2ymf` |
| 46,429,199 | 0.5290 | `dp8d1sdi` |
| 78,539,423 | 0.5634 | `pw8u0edj` |

10x the parameters buys ~3%. Two stories fit this table equally well, and the
aggregate number cannot tell them apart:

- **Floor story**: 0.55-0.58 is close to the conditional entropy of the corpus under
  this encoding; every size is parked at the same wall; the task, as encoded, is
  nearly solved.
- **Dilution story**: each message is 26 tokens and most positions are
  near-deterministic given the prefix (field structure, price high digits, copied
  reference fields). Their CE goes to ~0 in the first steps and then dilutes the
  average; the few informative positions may still be learning, invisibly.

The aggregate is one number over 26 positions with wildly different ceilings. This PR
decomposes it.

## What already exists (computed and then thrown away)

- The per-token CE vector of shape `(13000,)` = `(500 messages x 26 positions)` is
  already produced by the train step (`src/lob/train_helpers.py:1357`), accumulated
  when `log_ce_tables=True`, returned to the loop, bound to `ce_by_tok` at
  `src/lob/train.py:641`, and never logged anywhere.
- The 12-field layout of the 26 positions is a pair of constants:
  `FIELDS` and `TOK_LENS` in `src/lob/encoding_26tok.py:327` (widths 1,1,3,2,1,3,2,3,3,2,2,3).
- `syntax_validation_matrix()` (`src/lob/validation_helpers.py:56`) gives the set of
  syntactically legal tokens per position: `ln(#valid)` is a free per-position
  entropy ceiling.
- A deterministic fixed-slice CE evaluator over a chosen date range already exists as
  a template: `run/base_model/evaluate_model_zoo_ce.py`.

## Design

Four measurements, one verdict.

1. **In-training decomposition** (env `LOG_CE_BY_POSITION=1`, default off): reshape
   the existing CE vector to `(500, 26)`, average over messages, log 26 scalars
   `ce_pos/p00..p25` and 12 field aggregates `ce_field/<name>` (weighted by
   `TOK_LENS`) at the same cadence as `step_loss`. Also add `lr` to the step-cadence
   wandb dict (today it is logged only at epoch end, so an LR-schedule artefact
   cannot be ruled out from the curves; the hybrid branch made the same fix and this
   lands it on the main track).
2. **Offline decomposition** `run/base_model/evaluate_ce_by_position.py` (new,
   template above): teacher-forced per-position and per-field CE for a given
   checkpoint over (a) a training-period slice and (b) a held-out-period slice.
   Deterministic traversal, fixed sample count, JSON out.
3. **Two free reference lines** per position: the syntax ceiling `ln(#valid)`, and
   the corpus unigram entropy (one CPU pass over tokenized data). Model CE close to
   the unigram line means no conditional structure learned at that position; CE close
   to zero identifies the dilution positions.
4. **Capacity probe**: run the offline decomposition for the 8M and the 78M
   checkpoints. Wherever the 0.017-nat aggregate gap between them lives, that is
   where capacity still buys something.

## Reading rules (registered before the numbers exist)

- **Dilution verdict**: if the top 6 positions carry >= 80% of the aggregate CE and
  their held-out CE at the biggest size is still visibly above the 8M-vs-78M
  convergence level, the flat aggregate is dilution, not a floor: report headroom
  per field and stop quoting the 26-position average unaccompanied.
- **Floor verdict**: if per-position train and held-out CE agree everywhere and the
  8M-vs-78M per-position gaps are ~0, this family is at its effective wall on this
  corpus under this encoding, and further capacity spend is not justified by CE.
- The aggregate is never quoted without the decomposition table next to it (the
  hybrid PR learned this the hard way when one feature carried 55% of an aggregate).
- The 46M row (0.5290, below both its neighbours) must be explained from its wandb
  config before any cross-size claim is made; a family line with a non-monotonic
  point is not quotable until the point is understood.

## Work items

- [ ] `src/lob/train_helpers.py`: thread the interval-mean CE vector and `lr` into
      the wandb callback path, gated by `LOG_CE_BY_POSITION`
- [ ] `src/lob/train.py`: extend the mid-epoch wandb dict with the per-position and
      per-field keys
- [ ] `run/base_model/evaluate_ce_by_position.py`: offline decomposition, JSON out
- [ ] unigram + syntax reference computation (inside the same script, CPU path)
- [ ] `tests/unit/test_ce_decomposition.py`: field slicing vs the encoding constants
- [ ] `ci/check_ce_decomposition.py` + `.github/workflows/ce_decomposition.yml`
- [ ] `ci/evidence/profiling.json` recorded on the cluster once the numbers exist

## The check this PR ships

`ci/check_ce_decomposition.py`, pure stdlib + `ast`, never importing the modules it
audits, run with `shell: bash` so pipefail is active. Guards, each commented with the
hole it plugs:

1. the field table used for decomposition matches `FIELDS`/`TOK_LENS` in
   `src/lob/encoding_26tok.py` and the widths still sum to 26 (a silent encoding
   change would otherwise relabel every curve without breaking anything);
2. the offline script still derives its slicing from those constants instead of a
   private copy (a fork of the table is how relabeling sneaks in);
3. once recorded, `ci/evidence/profiling.json` source hashes still match the files
   that produced it (stale evidence must fail, not persuade).

`--self-test` corrupts copies of the guarded inputs and requires each guard to fail.

## Cluster evidence (lands as PR comments)

- Offline decomposition JSONs for the 78M and 8M checkpoints, train slice and
  held-out slice each, single GPU, attached to an existing allocation when one is
  idle.
- One smoke training run with `LOG_CE_BY_POSITION=1` showing the 12 field curves in
  wandb, plus the no-op guarantee (byte-identical logs with the env var absent).
- The verdict table filled in under the reading rules above.

## Non-goals

No new validation machinery (the validation PR owns that); no changes to the
encoding; no claims about SP500 until the 8-stock decomposition is read.

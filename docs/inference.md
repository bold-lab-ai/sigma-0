# Inference

## The runtime

```text
run/base_model/run_inference.py --config configs/inference/<yaml>     validates the request, prints the plan
run/base_model/runtime/inference.py                                    the CLI that generates
  -> base_model.inference.generate.sample_new                          conditioning + generation; every generated
                                                                       message goes through matching_engine.jaxob
  -> <save_dir>/data_cond/ data_gen/ data_real/                        one CSV per window and side, plus sample_indices_rank<k>.json
```

`data_cond` holds `N` conditioning messages and `N+1` book rows (the book
includes the initial snapshot). `data_gen` and `data_real` hold the generated
horizon and the real continuation over the same windows. This is the layout the
LOB-Bench adapter and the return bench read (`docs/benchmarking.md`).

## Running it

Login node, plan only:

```bash
PYTHONPATH=src python run/base_model/run_inference.py --config configs/inference/goog_2026.yaml --architecture mamba3
```

Compute node, one GPU:

```bash
CHECKPOINT_PATH=<manager root> DATA_DIR=<day directory> OUTPUT_ROOT=<writable dir> \
    sbatch run/base_model/smoke_inference.sbatch
```

or directly inside an allocation:

```bash
python -u -B run/base_model/runtime/inference.py --architecture=mamba3 --token_mode=26tok \
    --ckpt_path=<manager root> --checkpoint_step=<step> \
    --data_dir=<day directory> --save_dir=<out> --stock=GOOG \
    --n_sequences=8 --batch_size=1 --n_cond_msgs=16 --n_gen_msgs=4 --seed=42
```

Always pass `--checkpoint_step`. Without it the runtime asks Orbax for the
latest step, which lists the checkpoint directory. Pinned steps are in
`docs/checkpoints.md`.

## Inputs

- `--data_dir` must hold paired `*message*.npy` and `*book*.npy` files for one
  ticker (`docs/data.md`, section 1.1). The runtime globs that directory, so
  point it at a small per-job directory, not at a corpus root.
- The book must be a wide (L500) source. An L10 day fails with
  `wide-book width must be divisible by 4, found 38`. Wide books come from the
  per-ticker L500 SquashFS shards mounted with `infer_squashfs_setup_wide_book`
  (`docs/data.md`, section 1.3), which the model-zoo generation launchers do
  (`run/benchmarking/model_zoo_generation_array.batch`).
- `--architecture` is an assertion against the checkpoint's metadata, not a
  choice; the registry refuses a conflict.
- `MAMBA3_LEGACY_NORM=1` only for checkpoints written before 2026-04-22
  (`docs/checkpoints.md`).

## Validity

Generation is validated on semantics, not artifact counts: timestamp
monotonicity, nanosecond range, finite values, illegal-token counts.
`base_model.inference.validation` supplies the per-position syntax masks that
make illegal tokens impossible to sample for every architecture. The three
`-9999 ns` inter-arrivals that failed a LOB-Bench run in July 2026 came from a
START token left valid in a time position; that is why the masks exist
(`docs/lessons.md`).

## do(Q): injecting an order into a generated rollout

### Why the generator needs this at all

A world model trained on market data learns an observational distribution,
`p(market)`. Every question about acting in the market is interventional:
`p(market | do(our order))`. The two differ in a known direction, the
generator was never trained to punish us, so adverse selection is
under-counted, which is why the agentic market-making loop settles our own
fills in the matching engine rather than asking the generator for them.

That handles *our* orders when we are quoting. The remaining question is what
the generated market does when an order is placed into it: impact, decay,
whether the book refills. Answering it needs one order, of our choosing,
inserted at a chosen step of an otherwise untouched rollout, then the same
rollout without it. That is what this is.

### The API

`base_model.inference.generate.sample_new` takes five optional arguments:

| argument | meaning |
|---|---|
| `inject_step` | 0-indexed generation step to inject at; `None` (default) = off |
| `inject_event_type` | LOBSTER event type: `4` = marketable execution, `1` = new limit order |
| `inject_side` | `1` = bid side, `0` = ask side |
| `inject_qty` | Q, in shares |
| `inject_price_offset_ticks` | `p_abs = p_mid + offset * tick_size`; `0` = the tick-floored mid, **not** the touch (below) |

Underneath, `sample_new` builds two arrays over the generation horizon, `inject_flag_seq` `(n_gen,)` of 0/1 and `inject_msg_seq` `(n_gen, 4)` of
`(event_type, side, qty, offset_ticks)`, and passes them to `generate` /
`generate_batched`, which ride them along the `jax.lax.scan` as `xs`. At the
flagged step `_generate_msg` replaces the message the model produced with a
`construct_sim_msg` built from that row, and the engine processes it like any
other message.

The absolute price is computed **inside** the scan from the live `p_mid` of
that rollout, not fixed when the function is traced, the only version of a
relative price that is meaningful across a batch of rollouts that have already
diverged. The reference is the **tick-floored mid**, not the touch:
`p_mid = ((best_ask + best_bid) // 2 // tick) * tick`, refreshed from the book
after every processed message and held at its previous value while either side
is empty (`_get_new_mid_price`). With a 1-tick spread the floored mid happens
to coincide with the best bid; with a wider spread `offset=0` lands inside the
spread. `offset=-1` on the bid side is always passive (it cannot cross, since
an uncrossed book puts `mid - 1 tick` strictly below the best ask) and always
lands within the top two bid levels, i.e. inside the observable L2 window.

The counterfactual arm is the same call with `inject_step=None` and the same
seed: identical windows, identical sampling RNG, one order different.

### The guarantee when it is off

With `inject_step=None`, `inject_flag_seq` is `None`, and:

- `_generate_msg` never enters the `if inject_flag is not None:` branch, so
  `jax.lax.cond` is not traced and the generated `sim_msg` goes to the engine
  unchanged;
- `generate` selects `_xs = b_seq_real`, the scan it always had;
- the two extra positional slots carry `None`, which is an empty pytree: it
  contributes no traced values, so the compiled program is the one that was
  compiled before.

So the baseline path is not "close to" the old one, it is the old one, and
existing sealed numbers do not need re-measuring. The one place that is not
free: `generate_batched`'s `in_axes` has an entry per positional argument, so
those two slots are always *passed* on the token-per-field path (as `None`).
The 1tok generator has no injection point at all; requesting injection in
`token_mode='1tok'` raises rather than silently generating an uninjected
rollout.

### Provenance

The mechanism is Jess Carr's, from her FLAIR rotation work on evaluations for a
limit order book model (archive:
`/lus/lfs1aip2/projects/public/u6gb/lob_evals_jcarr/`, see its `README.md`,
`PATH_REMAP.md` and `report.pdf`). It reached this repository via LOBelia
(`alexandre-bismuth/LOBelia` @ `380b35f`, its `src/lob/inference_no_errcorr.py`, now `src/base_model/inference/generate.py` here),
which is also where `replay_real_msgs_single` / `compute_gt_divergence` came
from in #2.

Her analyses built on top of it, the (c,g) grids, the M(H)/I(H) and M(Q)
collators, the injection sweeps, deliberately stay in the archive. They are an
experiment, and belong in a fork; this is the primitive they used.

### Verification

The claim that gates, the injected order lands in the book at the flagged
step, and nothing else changes there, is measured, not trusted.
`ci/evidence/injection.json` freezes the measurement;
`python ci/check_evidence.py` replays its assertions on any machine, with no
jax, GPU or data.

The measurement is one injection-on rollout (arm **I**, flags
`--inject_step K --inject_event_type 1 --inject_side 1 --inject_qty Q
--inject_price_offset_ticks -1` on `run/base_model/runtime/inference.py`).
`tools/diagnostics/verify_injection.py` (stdlib only) decides the gating
judgement from the CSV artifacts alone and writes a fixed-template report
(`sigma0.injection_verification.v2`):

- **J4** between I's book rows K-1 and K the engine processed exactly one
  message, so the L2 delta must be `+Q` at `p_mid(K-1) + offset·tick` and
  nothing else (window shifts included). This is the only judgement in
  `overall_pass`.

Off-means-off is enforced statically: when `--inject_step` is absent the
`sample_new` call carries no injection kwargs at all, and
`tests/unit/test_injection_wiring.py` pins that shape (knobs default to off,
the 1tok path has no injection slots, vmap axes match the signature).

Three optional reference arms (P: merge-base; P2: P run twice; O: this branch,
injection off) add byte-identity diagnostics, J0 `P == P2`, J1 `P == O`,
J2/J3/J5 prefix and divergence agreement between O and I. Byte-identity across
GPU runs holds only on nodes that compile both programs identically (defect
D-X7; on 2026-08-27 the O-vs-I comparison itself was shown to be node-bound,
for the pre-existing commit pair too). Per PR #21 review those judgements are
recorded but never gate.

`--check report.json` re-audits a written report anywhere;
`ci/record_evidence.py --injection --injection-report report.json` freezes it
into the evidence file, hashing this file, the runtime CLI and the checker as
sources, change any of them and the audit demands a re-measurement.

On GitHub, the injection steps of `.github/workflows/checks.yml` run on every pull
request: they fail if the injection evidence is missing (the blanket evidence audit only
audits files that exist), re-run `--check` on the committed report, assert
the evidence and the report agree on digest, node, parameters and verdict
(the report itself is not hashed in the evidence sources, so this is the step
that stops it being swapped), and run the verifier's own unit tests on the
bare runner, none of which needs jax, a GPU or the data.

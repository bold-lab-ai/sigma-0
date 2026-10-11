# Validation sets: inventory and selection rule

This write-up condenses [PR #51](https://github.com/KangOxford/sigma-0/pull/51).
All measurements and original figures are pinned to source commit
[`f1f356bbf6b4`](https://github.com/KangOxford/sigma-0/commit/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8).
They are historical measurements, not a fresh audit of today's runs.
Images and reproducibility artifacts remain in that commit's history; they are
not additional files in this documentation change. Viewing them requires access
to this repository. A plain offline checkout will not contain the linked images.

## Which set answers the question?

| Question | Primary set | Qualification |
| --- | --- | --- |
| Fit a scaling law or compare model sizes | valset_v1, 30,720 windows | Use 307,200 when uncertainty requires it; verify the training-consumption fence. |
| Does the conclusion survive an unseen month? | Uniform Jan-2026 draw | Repeat the primary comparison on a genuinely held-out month. |
| Compare architectures | A common frozen valset | Match domain, encoding and aggregation; Transformer CE tables are prior measurements, not input samples. |
| Score generation fidelity | Model-zoo GOOG Jan-2026 rollout sample | Use LOB-Bench or return-bench on generated and real messages. CE alone does not answer this. |
| Compare with published feature-based LOB work | FI-2010 | Feature tables require a compatible model/protocol; they cannot directly feed our message-token models. |
| Reproduce historical tables | Legacy Jan-2026 per-ticker draw | Retain for provenance. Do not use it as the primary ruler for a new scaling fit. |

The recommendation is specific to the evaluated domains and training protocol.
Its construction evidence and behavioural-test limitations are in
[the leakage audit](leakage_audit.md).

## Nine entries, three kinds of location

The following inventory preserves the source census. “Frozen” and “held out”
refer to its recorded configuration, not a permanent guarantee for later runs.
`<hub>` is the historical scaling-law experiment root configured in
[build_evidence.py](https://github.com/KangOxford/sigma-0/blob/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/build_evidence.py). Cluster paths are source locations and have not
been checked for availability by this documentation change.

| Set | Role | Scale | Recorded location |
| --- | --- | --- | --- |
| valset_v1_30720 | in-distribution held-out | 30,720 windows (376,647,680 B) | &lt;hub&gt;/valset_ce_eval_20260730/valset/shard_valset_v1_30720.squashfs |
| valset_v1_307200 | in-distribution held-out (10x) | 307,200 windows | &lt;hub&gt;/valset_ce_eval_20260730/valset/ (3.51 GB shard, checksummed) |
| valset_v1_index | in-distribution held-out (full index) | 5,367,734 windows = 1.661% of 323,221,385 | &lt;hub&gt;/valset_ce_eval_20260730/inputs/ + valset/VALSET_V1_REPORT.md |
| jan2026_ticker_equal | forward-shifted test month (legacy sampling) | 487 tickers, equal windows per ticker, month-start heavy | /projects/public/s5e/quant_team/lob_preproc_sp500_squashfs/shard_2026-01.squashfs |
| jan2026_shuffle | forward-shifted test month (uniform sampling) | 30,720 windows drawn over the whole month | &lt;hub&gt;/valset_ce_eval_20260730/jan_shuffle/ |
| tf_valset_full | frozen valset, Transformer line (2022-25) | 30 curves, median 12 checkpoints | s5e_scalinglaw/valset_ce_tf_fit_ready_v2.csv |
| tf_valset_y2325 | frozen valset, Transformer line (2023-25 domain) | 30 curves, median 12 checkpoints | s5e_scalinglaw/valset_ce_tf_y2325_fit_ready_v2.csv |
| model_zoo_goog_jan2026 | generation benchmark sample | 3,136 of 226,002 sequences, 500 messages each | run/benchmarking/model_zoo_GOOG_jan2026_3136_seq500_seed42.txt.metadata.json |
| fi2010 | external public benchmark | train 362,401 x 150, test 31,938 x 150 | shanehans/FI2010 (MIT); local mirror in grokking/fi2010/ |

The two Transformer CSV paths refer to the archived source commit, rather than
files newly added to main: [full-domain table](https://github.com/KangOxford/sigma-0/blob/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/valset_ce_tf_fit_ready_v2.csv)
and [2023-25 table](https://github.com/KangOxford/sigma-0/blob/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/valset_ce_tf_y2325_fit_ready_v2.csv).
The model-zoo sample metadata is likewise a recorded source path.
The full-index census covers 488 tickers and 2022-01 through 2025-12;
the legacy January census covers 487 tickers. The smaller shard is nested in
the larger shard. Full source metadata, including original frozen/held-out
labels and checksum prefixes, is in [validation_sets_inventory.csv](https://github.com/KangOxford/sigma-0/blob/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/validation_sets_inventory.csv).

![Original inventory map: locations and intended eras of use.](https://github.com/KangOxford/sigma-0/raw/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/validation_sets_map.png)

Original inventory map: locations and intended eras of use.

## Why the sampling rule changes the conclusion

The July 30-31 evaluation used the same 132 checkpoints from 33 runs for the
last-quarter comparison. The endpoint fit uses 33 endpoints. Evaluation set
choice is the varying axis; the January alternatives share the month and the
forward-pass implementation.

| Fit window | Evaluation axis | alpha | At bound | beta | E | Points | Runs | Reported best size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| endpoint | jan2026_macro | 2.000000 | True | 1.408136 | 0.565524 | 33 | 33 | 23M |
| endpoint | valset_macro | 1.911948 | False | 1.726205 | 0.595692 | 33 | 33 | 120M |
| endpoint | jan_shuffle_macro | 1.886196 | False | 1.600871 | 0.523363 | 33 | 33 | 120M |
| last25 | jan2026_macro | 2.000000 | True | 0.801014 | 0.555200 | 132 | 33 | 23M |
| last25 | valset_macro | 1.922188 | False | 1.337441 | 0.595692 | 132 | 33 | 120M |
| last25 | jan_shuffle_macro | 1.892724 | False | 1.404460 | 0.523363 | 132 | 33 | 120M |

Source: [axis_fits.csv](https://github.com/KangOxford/sigma-0/blob/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/axis_fits.csv). The legacy fit reaches the imposed alpha
upper bound of 2 in both windows. This shows sensitivity to the ruler; it does
not establish that 2 is the true exponent. The two alternatives select the
120M labelled configuration while the legacy axis selects 23M. These are
minima in the recorded comparison, not a universal prescription for future
compute budgets. In particular, 46M and 120M are close on the alternative
rulers; point estimates alone do not establish a statistically resolved winner.

![The fitted exponent depends on the evaluation draw.](https://github.com/KangOxford/sigma-0/raw/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/axis_choice_changes_alpha.png)

The fitted exponent depends on the evaluation draw.

![The recorded size comparison changes with the ruler.](https://github.com/KangOxford/sigma-0/raw/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/three_rulers_by_size.png)

The recorded size comparison changes with the ruler.

The original narrative describes valset as about 0.073 nats harder than the
uniform January draw. The table supports an approximately constant gap over
most of the ladder, but the smallest labelled configuration has a larger gap
(about 0.093). The legacy-minus-uniform gap increases from about 0.025 around
6M to about 0.067 at the largest configuration; it is not monotonic over every
row. An exactly constant offset can be absorbed by a free loss-floor term;
a size-dependent distortion can change a scaling fit.

![Difficulty offset and size-dependent sampling bias, from the original evidence.](https://github.com/KangOxford/sigma-0/raw/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/offset_vs_slope_bias.png)

Difficulty offset and size-dependent sampling bias, from the original evidence.

The complete per-size values and actual parameter counts remain in
[ruler_by_size.csv](https://github.com/KangOxford/sigma-0/blob/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/ruler_by_size.csv). The smallest historical size labels are not
literal total parameter counts, so use that table's `n_params` for quantitative
size calculations.

## The 20% fence has a finite budget

The recorded domain has 323,221,385 windows. Training consumes a prefix of a
seeded permutation. The construction selects windows beyond the first 20%
for seeds 5, 42 and 137. At 128 windows per optimizer step, the conservative
whole-step limit is `floor(0.20 * 323221385 / 128) = 505033`.

| Seed | Recorded deepest consumer | Steps consumed | Domain consumed (%) | Step limit | Steps remaining |
| --- | --- | --- | --- | --- | --- |
| 5 | 700B-token long run | 420000 | 16.63 | 505033 | 85033 |
| 42 | 350M full-data run | 168200 | 6.66 | 505033 | 336833 |
| 137 | primary matrix (6M size) | 106909 | 4.23 | 505033 | 398124 |

Source: [valset_leakage_budget.csv](https://github.com/KangOxford/sigma-0/blob/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/valset_leakage_budget.csv). The maximum recorded
consumption here is **16.63%**, including the seed-5 long run. The talk's
4.2% illustration describes a narrower run cohort and must not replace this
census. The 85,033 remaining seed-5 steps are a historical budget, not current
headroom.

For a new run, recompute cumulative distinct exposure with the actual domain,
seed, permutation implementation, sampler, effective batch size and resume
history. The 2023-25 sub-domain needs its own calculation. An unverified seed
or a run crossing the fence loses this construction guarantee and requires
an overlap audit or a new validation set. Crossing 20% does not itself prove
that every validation sample was seen, nor retroactively contaminate older
checkpoints; it breaks the shared guarantee for the new comparison.

![Historical consumption budget; re-audit before new training.](https://github.com/KangOxford/sigma-0/raw/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/future_budget.png)

Historical consumption budget; re-audit before new training.

![Sample counts determine evaluation cost; the larger shard has ten times as many windows.](https://github.com/KangOxford/sigma-0/raw/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/set_scales.png)

Sample counts determine evaluation cost; the larger shard has ten times as many windows.

## Decision matrix and reproducibility

![Choose the set by the question, then verify the protocol.](https://github.com/KangOxford/sigma-0/raw/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/decision_matrix.png)

Choose the set by the question, then verify the protocol.

The original seven-figure notebook is [validation_sets.ipynb](https://github.com/KangOxford/sigma-0/blob/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/validation_sets.ipynb).
Its builder is [build_notebook.py](https://github.com/KangOxford/sigma-0/blob/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/build_notebook.py) and the input-freezing script is
[build_evidence.py](https://github.com/KangOxford/sigma-0/blob/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/build_evidence.py). The detailed selection matrix is
[decision_matrix.csv](https://github.com/KangOxford/sigma-0/blob/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/decision_matrix.csv). These files document historical experiments
and may require the original cluster environment.

This documentation consolidation introduces no launcher enforcement and no new
CI guard. The original PR's guard and scripts remain accessible at the pinned
commit. The pre-flight fence above is a required protocol check for future
experiments, not a claim that this change implements it.

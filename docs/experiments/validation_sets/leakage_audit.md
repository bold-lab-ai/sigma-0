# Can we trust the validation set? Evidence and limits

This write-up condenses [PR #51](https://github.com/KangOxford/sigma-0/pull/51).
All measurements and original figures are pinned to source commit
[`f1f356bbf6b4`](https://github.com/KangOxford/sigma-0/commit/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8).
They are historical measurements, not a fresh audit of today's runs.
Images and reproducibility artifacts remain in that commit's history; they are
not additional files in this documentation change. Viewing them requires access
to this repository. A plain offline checkout will not contain the linked images.

[Selection rule and inventory](README.md) explain when to use the set.
The conclusion here is narrower: the sampled windows satisfy the recorded
index-based exclusion rule for three seeds. The behavioural experiment does
not independently prove absence of leakage.

## 1. Construction check: two routes agree

Each training example is a 500-message window. The recorded training domain
contains 323,221,385 windows. For each seed, the verifier inverts the training
permutation and reads every validation window's fractional position:

```python
# Algorithm from the archived verify_fence.py; n and idx must match training.
g = torch.Generator()
g.manual_seed(seed)
perm = torch.randperm(n, generator=g).numpy()
pos = np.empty(n, dtype=np.int64)
pos[perm] = np.arange(n, dtype=np.int64)
fractional_position = pos[idx] / n
```

One route uses the builder's frozen `global_idx` provenance. The other obtains
the exact file list and window counts from the training dataset object and
reconstructs those indices. With the exact list, all 30,720 indices match and
both routes place zero samples before 0.20 for seeds 5, 42 and 137.
They independently reconstruct the indices but share the permutation rule and
underlying dataset assumptions; this is not a proof against every possible
form of data leakage or duplicate content.

| Seed | Minimum fractional position | Samples before fence | Fraction in last 2% |
| --- | --- | --- | --- |
| 5 | 0.2000298340408386 | 0 | 0.3436197916666667 |
| 42 | 0.2000119391852739 | 0 | 0.34469401041666664 |
| 137 | 0.2000574528817145 | 0 | 0.3375325520833333 |

Sources: [fence_verification.csv](https://github.com/KangOxford/sigma-0/blob/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/fence_verification.csv), [fence_verification_exact.csv](https://github.com/KangOxford/sigma-0/blob/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/fence_verification_exact.csv).
Every validation sample lands in the last 2% for at least one of the seeds,
which explains the tail spikes. The construction pool minimum (0.2000003)
and subset minima describe different populations; a subset need not include
the pool's closest-to-fence sample.

![Original fence illustration and histograms. The 4.2% training marker is cohort-specific; the broader recorded budget reaches 16.63%, as explained in the inventory.](https://github.com/KangOxford/sigma-0/raw/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/talk_fig1_fence.png)

Original fence illustration and histograms. The 4.2% training marker is cohort-specific; the broader recorded budget reaches 16.63%, as explained in the inventory.

## 2. Why the first reconstruction failed

A hand-built corpus index initially put about 19.6-19.9% of validation samples
before the 20% fence. That is close to the uniform chance rate. It is not
sufficient evidence of leakage when the index itself has not been matched to
training. Changing the indexed universe or window offsets can scramble the
permutation positions, so an approximate file-list match is insufficient.
This does not mean that every possible one-file error necessarily changes
every index; it means the exclusion claim requires the exact training index.

![Frozen provenance versus the incorrect corpus reconstruction. This panel predates the exact-file-list repair.](https://github.com/KangOxford/sigma-0/raw/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/talk_fig2_two_routes.png)

Frozen provenance versus the incorrect corpus reconstruction. This panel predates the exact-file-list repair.

Reading `message_files` and `_seqs_per_file` from the actual dataset object
resolved the mismatch. The hand-built census had 472,486 files versus 472,442
in training, and 302,594 extra candidate windows:

| Category | Files | Excess candidate windows | Detail |
| --- | --- | --- | --- |
| absent from the training list | 44 | 88805 | ABNB 8, AVGO 12, BAC 1, CRM 10, MU 13 |
| present but assigned zero windows (fewer than 500 rows) | 371 | 0 | truncated trading days; 2025-06-02 alone accounts for most |
| present but assigned zero windows (500+ rows) | 229 | 213789 | MU dominates with 41, one of them 9.7M rows |
| TOTAL | 44 | 302594 | file gap 44, window gap 302594 |

The TOTAL row reports the net file-count gap, not the sum of diagnostic
categories. The 371 short files contribute zero windows to the discrepancy.
The remaining two categories sum to `88805 + 213789 = 302594`.

![Exact training file-list reconciliation and complete index agreement.](https://github.com/KangOxford/sigma-0/raw/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/talk_fig3_reconcile.png)

Exact training file-list reconciliation and complete index agreement.

Sources and reproduction: [dump_train_filelist.py](https://github.com/KangOxford/sigma-0/blob/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/dump_train_filelist.py), [reconcile_filelist.py](https://github.com/KangOxford/sigma-0/blob/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/reconcile_filelist.py), [verify_fence.py](https://github.com/KangOxford/sigma-0/blob/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/verify_fence.py). These historical scripts
need the original data and environment. This consolidation did not rerun the
large permutation or access the cluster data.

## 3. Behaviour: the positive control is unresolved

The July 30 experiment compared three groups of 30,720 windows: SEEN from the
consumed prefix, MID from beyond the fence, and VAL from valset_v1. Exposure
is the intended comparison, but equal group sizes do not guarantee identical
content difficulty. The table gives mean CE in nats and 95% intervals for
contrasts, copied from [leakage_groups_ce.csv](https://github.com/KangOxford/sigma-0/blob/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/leakage_groups_ce.csv).

| Model | SEEN | MID | VAL | SEEN minus MID (95% interval) | VAL minus MID (95% interval) |
| --- | --- | --- | --- | --- | --- |
| 78M-s5 | 0.5597 | 0.5599 | 0.6045 | -0.0002 [-0.0073, +0.0070] | +0.0446 [+0.0380, +0.0514] |
| 350M-s5 | 0.5717 | 0.5732 | 0.6194 | -0.0015 [-0.0060, +0.0030] | +0.0461 [+0.0419, +0.0505] |

![VAL is harder than MID, but SEEN does not show a resolved advantage over MID.](https://github.com/KangOxford/sigma-0/raw/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/talk_fig4_behaviour.png)

VAL is harder than MID, but SEEN does not show a resolved advantage over MID.

The observed VAL-minus-MID sign is inconsistent with a simple large net
memorisation advantage. It cannot rule out an exposure effect masked by a
content-difficulty offset. More importantly, the SEEN-minus-MID intervals
include zero: the check cannot reliably distinguish even known-seen data in
this experiment. The interval widths describe uncertainty in these contrasts,
not a universal upper bound on leakage or a calibrated detection threshold
for arbitrary models.

## 4. Deliberate-exposure calibration

The follow-up split unseen data into PROBE and CONTROL, 15,360 windows each,
then repeatedly trained on PROBE and scored both groups. The archived
learning-rate sweep shows why control drift needs attention:

| Learning rate | Control drift | Probe-minus-control shift |
| --- | --- | --- |
| 1e-06 | -8e-05 | -0.00039 |
| 5e-06 | 4e-05 | -0.0017 |
| 2.8e-05 | 0.00205 | -0.00576 |
| 0.0005 | 0.16574 | -0.00565 |

Source: [exposure_lr_sweep.csv](https://github.com/KangOxford/sigma-0/blob/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/exposure_lr_sweep.csv). At 0.0005 the untouched group's
CE increases by 0.16574 nats. The lower-rate calibration reduces this drift.
An untouched group can legitimately change when shared model parameters are
updated; “not trained on” does not make its loss mathematically invariant.
Large drift nevertheless limits a clean interpretation as isolated exposure.

The calibration's difference-in-differences (DiD) is
`(probe_after - control_after) - (probe_before - control_before)`.
Negative values indicate a relative advantage on PROBE.

| Passes | DiD (nats) | 95% lower | 95% upper | Control CE change |
| --- | --- | --- | --- | --- |
| 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1 | -0.003981202095746994 | -0.0052245739428326485 | -0.003029557966316742 | 0.003969780790309112 |
| 2 | -0.007128646721442541 | -0.008574619412732621 | -0.006000120000292858 | 0.0047691867997248965 |
| 4 | -0.013587928625444572 | -0.01412271428387612 | -0.01312400492684295 | 0.006902175458769004 |
| 8 | -0.030117989828189213 | -0.03105518626359602 | -0.029323351794543366 | 0.01580490873505672 |

Source: [exposure_calibration.csv](https://github.com/KangOxford/sigma-0/blob/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/exposure_calibration.csv). A single pass produces about
-0.00398 nats of relative advantage. This is smaller than the rough
0.006-0.007-nat uncertainty scale discussed for the earlier group comparison.
The calibration itself uses before/after measurements and a different
contrast, so its own interval excluding zero does not validate the earlier
test as a membership detector.

![Original exposure calibration. The shaded band is an approximate comparison scale, not a universal power guarantee.](https://github.com/KangOxford/sigma-0/raw/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/talk_fig5_calibration.png)

Original exposure calibration. The shaded band is an approximate comparison scale, not a universal power guarantee.

Through four passes the control drift is at most about 0.00690 nats; at eight
it reaches 0.01580, exceeding that comparison scale. The eight-pass result
therefore needs particular caution. The first one-to-four passes are the
original report's usable range under that heuristic, not a formal proof of
an uncontaminated causal estimate.

The runner is [exposure_calibration.py](https://github.com/KangOxford/sigma-0/blob/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/exposure_calibration.py) and the full talk is
[can_we_trust_the_ruler.ipynb](https://github.com/KangOxford/sigma-0/blob/f1f356bbf6b48ac2f48f7610bb55b9c9a72716d8/s5e_scalinglaw/validation/can_we_trust_the_ruler.ipynb). Repeated exposure to a small subset,
learning-rate choices and access to a before measurement differ from auditing
a finished model. These results do not establish a universal minimum number
of exposures needed for detection or that a single exposure is undetectable
at every model scale.

## Practical conclusion

- Use the recorded index proof for the stated sample, domain and seeds,
  provided cumulative training consumption stays inside the audited prefix.
- Obtain file lists and window counts from the actual training dataset object;
  do not interpret a chance-rate result before matching its index exactly.
- Treat the behavioural check as an under-resolved cross-sectional contrast.
  Its lack of a SEEN advantage is a limitation, not additional proof of safety.
- Re-audit new seeds, domains, sampler changes and extended training. Preserve
  the exact data definition and cumulative exposure when resuming a run.

The original notebooks, tables, plotting code, fence scripts and CI guard
remain in the pinned PR history. This write-up carries their findings and
limitations into main without adding their generated outputs or experimental
runners to the active source tree.

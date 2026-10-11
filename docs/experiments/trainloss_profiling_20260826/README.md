# Which loss can carry an entropy claim

`entropy_floor_analysis.ipynb` answers the question that comes before this PR's own: the profiling
here is aimed at "the bottom of the training loss", and a training loss cannot carry an entropy
claim at all. Held-out CE equals `H(p) + KL(p||q)` and so bounds the entropy from above; training
CE has no such relation and goes to zero with enough capacity. On the checkpoint this PR already
measured, the two columns differ by 3.25 bits/msg.

Eight figures and two tables, committed with outputs so GitHub renders it inline.
`ci/check_ce_decomposition.py` refuses a version whose outputs have been stripped -- that failure
shows up in a diff as base64 churn and reaches the reader as a page of empty headings, so it is
the one thing review structurally cannot catch.

## Inputs

Nothing here is recomputed from the cluster. The notebook reads files already in the repository:

| input | where it comes from |
|---|---|
| `data/j5749206_*_train_period_v2.json` | GOOG 2025-12, inside the window, trained on |
| `data/j5749206_*_valset_2025.json` | valset_v1 2025-01..11, inside the window, never trained on |
| `data/j5749206_*_heldout_v2.json` | GOOG 2026-01, after the window, never trained on |
| `data/j5749206_*_valset_{2022,2023,2024,2025h1,2025h2,all}.json` | leak-free slices that differ only in period |
| `data/ruler_by_size.csv` | #51 -- 12 sizes scored on three different held-out rulers |
| `data/axis_fits.csv` | #51 -- what the choice of ruler does to the fitted exponent |
| `data/validation_sets_inventory.csv` | #51 -- what each of the nine named validation sets is |

The three CSVs are byte copies from `s5e_scalinglaw/validation/` on #51 at commit
`a678fcd8f0af3b5eae5670509112c287300c47e0`. They are copied rather than referenced because #51 is
based on #43 and this branch is based on `main`; an analysis that only runs if two unrelated PRs
merged first is not one anybody will re-run. 4 KB total.

Two numbers in the notebook are transcriptions, marked where they appear: the 7.57 bits/msg floor
claim (`NEURIPS_PLAN.md` line 100, on the cluster mirror) and the eval-budget constants
(`test_ce_eval_spec.md` line 42).

## Why the figures are 300 dpi and still small

Two constraints pull against each other. Figures render at `dpi=300`, because 100 dpi is visibly
soft on a high-density display and unusable in a paper. But GitHub stops rendering a notebook once
it gets large, and a notebook that does not render defeats the whole reason for committing outputs.
At 300 dpi with the original canvas this file was 981 KB, close enough to the edge that one more
figure would have tipped it silently.

Both are satisfied by shrinking the canvas rather than the resolution. Every figure size and every
point-valued quantity -- font sizes, marker sizes, line widths, scatter areas -- is scaled by the
same 0.78, which reproduces the identical layout at a smaller physical size; at 300 dpi that still
renders 2.1x the linear pixel density of the 110 dpi version it replaces. `show()` then writes a
128-colour palette PNG instead of the 32-bit one the inline backend embeds, which flat chart
colours take with no visible step. 981 KB becomes 770 KB at eight figures.

The quantisation lives in the notebook rather than in a post-processing pass, so re-executing
reproduces the committed bytes instead of growing the file back. It falls back to `plt.show()`
where Pillow is absent.

## Re-running it

```bash
cd docs/trainloss_profiling_20260826
jupyter nbconvert --execute --inplace entropy_floor_analysis.ipynb   # pandas, matplotlib, Pillow
python3 ../../ci/check_ce_decomposition.py                           # standard library only
```

## What it leaves open

One measurement, named in Figure 6 and conclusion 3: the two slices differ in **both** leakage and
time period, so the +0.0868 nats is memorisation and temporal shift added together with neither
term measured. Running this PR's decomposition on `valset_v1` -- frozen, leak-free, the same 488
tickers over the same 2022-2025 window -- fills the missing cell and separates them. No training,
no new data.

# Model-zoo experiment protocol

This document freezes the primary comparison before January-2026 LOBBench
results are inspected. Launcher assertions and `submission.json` are the
machine-readable counterparts of this protocol.

## Primary question and outcome

At approximately 8.1 million trainable parameters, do recurrent sequence
models produce more realistic closed-loop limit-order-book continuations than
attention models?

The primary outcome is corrected LOBBench WS-21 on GOOG January 2026. Teacher-
forced cross-entropy is used only to select comparable checkpoints; it is not
the paper-facing outcome and January-2026 data is never used for selection.

## Models and parameter matching

All models use the same dual message/book encoder, 26-token representation,
500-message training context, and six fused sequence layers.

| Architecture | Configuration | Trainable parameters |
|---|---|---:|
| Mamba3 | `d_model=256`, `d_state=128`, expand 2, head 64 | 8,099,567 |
| S5 | `d_model=512`, state 256, 4 blocks | 8,201,069 |
| GDN | `d_model=256`, 4 heads × 64, value expand 2 | 8,021,621 |
| KDA | `d_model=256`, 4 heads × 56, value expand 2 | 7,916,209 |
| Transformer | `d_model=224`, 4 heads, `d_ff=800` | 8,041,203 |
| NSA | `d_model=224`, 4 heads, `d_ff=800` | 8,056,828 |

Mamba2 is excluded because its relevant state-update family is treated as a
subset of the Mamba3 design being tested. Mamba3 remains the default; every
other architecture is selected explicitly.

## Data and randomness

- Training: GOOG, AAPL, NVDA, AMZN, META, TSLA, MSFT, and AMD, January 2022
  through November 2025.
- Checkpoint selection: the same eight tickers, December 2025 only.
- Final test: GOOG, January 2026 only.
- Training seeds: 5, 42, and 137.
- Generation seed: 2026, with identical sampled windows for every run.

Training cycles through the eight tickers uniformly. A global batch contains
one ticker only, all data-parallel ranks use disjoint slices of that same
ticker batch, and a sequence index is not reused within a run. This prevents
high-activity equities from dominating the finite training budget and matches
the thesis sampling protocol.

The fixed December checkpoint-selection sample follows the same weighting: by
default it contains one deterministic eight-sequence batch from each ticker,
and those 64 sequences are cached and reused for every checkpoint and model.

## Optimisation and budget

- AdamW/Adam parameter groups, as in the thesis architecture comparison.
- Peak learning rate `5e-4`, cosine decay, 1% linear warmup.
- Weight decay 0.05 on non-SSM parameters; SSM parameters use Adam without
  weight decay.
- Global gradient norm clipping at 1.0.
- Four nodes × four GPUs × eight sequences/GPU = global batch 128.
- Each update contains 1,664,000 token values (`128 × 500 × 26`).
- Maximum 30,000 optimiser updates (49.92 billion token presentations).
- Checkpoints every 1,000 updates.

The 30,000-update cap is a common maximum, not the comparison coordinate.
Models are compared at a predeclared December-2025 validation CE of 0.65. The
closest checkpoint must be within 0.01 CE or the run is reported as not
matched; no January-2026 metric may be consulted to replace it. Steps and
tokens required to reach the target are secondary efficiency outcomes.

For continuity with the thesis, validation also records the first checkpoint
at which CE improves by less than 1% over at least 5,000 updates. It does not
override the predeclared iso-CE checkpoint rule.

## LOBBench evaluation and uncertainty

Each selected checkpoint uses 3,136 fixed January-2026 windows, split as 784
windows on each of four inference ranks with batch size 8 (no padded or repeated
evaluation rows), plus 500 real conditioning messages and 500 generated
messages. The corrected tick-distance
WS-21 feature contract and the external LOBBench Git revision are recorded in
`submission.json`.

The main table reports each training seed and the across-seed mean and standard
deviation. With only three training seeds, individual values remain visible;
within-run LOBBench bootstrap intervals are reported separately and are not
presented as substitutes for between-training-seed uncertainty. Architecture
comparisons use paired seed IDs and common generation windows.

## Execution gates

Production submission requires, in order:

1. one real 26-token training step at the exact model shape and per-GPU batch
   for all six architectures;
2. a full-corpus shard/index and uniform-sampler smoke;
3. checkpoint restore plus held-out CE evaluation;
4. tiny real-data 26-token inference followed by authoritative LOBBench scoring
   for all six architectures;
5. a clean, commit-pinned worktree and a dry-run of the correlated training →
   validation → LOBBench dependency chain.

Failed gates are fixed and rerun only for affected tasks. Paper results are
never read from gate-sized inference runs.

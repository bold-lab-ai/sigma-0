# State model: the book as video

The message-level models in this repository generate one event at a time.  The
state model generates the *board position* instead: whole blocks of future book
snapshots, denoised jointly by a flow-matching DiT.  The framing follows
text-to-video practice — a prompt-conditioned model rolls a clip, and
post-training decides what "a good clip" means — with the limit order book as
the scene.

## The frame clock is wall time, 1 ms per frame by default

BookDiT samples one frame per ten messages (an event clock).  Here frames sit
on a fixed wall-clock grid: `frames.py` forward-fills event-level book states
onto one frame every `interval_ns` (default 1 000 000 ns = 1 ms).  That makes
the horizon a property of the model itself: a window of H frames always covers
H milliseconds, however bursty message arrival was.  Two consequences:

* BookDiT's `log1p(dt)` channel disappears — on a fixed grid it is a constant.
  An optional activity channel `log1p(events per frame)` replaces it, carrying
  exactly the information the fixed clock folds away.
* `events_per_frame` in `FrameSeries` reports how much intra-frame churn each
  frame absorbed; any honest evaluation should know its size.

Input is the repository's wide-book npy contract (43 columns: 3 metadata
`[delta_p_mid, time_s, time_ns]` + 4 per level), the same files the message
models train on.

## Representation: frames the model cannot make invalid

`representation.py` adopts BookDiT's structured chart: log-spread, mid change,
log ladder gaps, raw sizes.  Decoding maps spread and gaps through `exp` and
rebuilds the ladder cumulatively, so *any* real-valued tensor — including pure
noise — decodes to a positive-spread, strictly monotone book.  Validity is an
identity of the coordinate system, not a learned behaviour (BookDiT measures
0.000 crossed/locked/non-monotone rates in this chart, against 54.4% crossed
for the same model trained on raw coordinates).  The torch decode is
differentiable (`decode_frames_torch`), which is what lets raw-space rewards
train a chart-space model in ReFL and DMDR.

`min_gap` floors the log arguments and is calibrated to raw LOBSTER integer
prices; re-choose it for rescaled data or the ladder flattens silently.

## Backbone: one decision from each reference

| choice | taken from | why |
|---|---|---|
| WanDiT block (modulation table + shared 6-way time projection, qk-RMSNorm, bidirectional attention, zero-init head) | BookDiT | its parameter account is published (953,769 at width 128 / depth 4 / 41 channels) and asserted in this package's tests |
| mask-concat conditioning (`cat([x_t, cond, mask])`, conditioning columns zero-initialised) | sigma-flow | the one mechanism in either reference with zero train/inference gap: no RePaint pinning needed at sampling time |

`flow.py` fixes the convention (t = 0 data, t = 1 noise, `v* = eps - x0`) and
derives the three conversions every post-training method leans on
(`x0_from_v`, `eps_from_v`, `score_from_v`), plus the marginal-preserving SDE
whose Gaussian transitions give Flow-GRPO its likelihoods.

## Defects inherited from nowhere

The sigma-flow codebase (this package's direct ancestor) never trained
properly.  The autopsy found the causes; each is fixed structurally here:

| sigma-flow defect | fix in this package |
|---|---|
| RoPE cache built in bf16 collapsed 8192 positions onto 769 distinct values | fp32 cache, integer positions, sized from `max_seq_len`, non-persistent (`dit.py`) |
| FM loss covered context frames, whose velocity is analytically recoverable — half the objective was filler | loss masked to generated frames (`train.py`, `flow.fm_loss(frame_mask=...)`) |
| one context length per *batch* | context length drawn per *sample* (`train.py`) |
| normalisation optional and silently skipped | fitted by the trainer, stored inside every checkpoint (`train.py`, `ChartNormalizer`) |
| inference hard-coded a different input width than training | model config travels in the checkpoint |

## Files

    frames.py          event stream -> fixed-interval frames (forward fill)
    representation.py  chart encode/decode (+ differentiable torch decode)
    dataset.py         windowing, normalisation fitting
    dit.py             FrameDiT backbone
    flow.py            rectified-flow core, ODE/SDE samplers, step log-probs
    rewards.py         validity / sanity rewards over decoded books
    train.py           conditional flow-matching pretraining (with checkpoints
                       + breadcrumb resume)
    posttraining/      the four alignment methods — see its README

## References

* BookDiT (ICAIF '26 submission): chart representation, WanDiT backbone,
  event-clock framing this package deliberately departs from.
* sigma-flow (`KangOxford/sigma-flow`): mask-concat conditioning, and the
  defect list above.

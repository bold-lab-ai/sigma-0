# Ten milliseconds after a trade: an event-anchored return bench, and what it found

## TL;DR

**What was built.** An evaluation asking what the mid-price does **N milliseconds after a
real trade executes**, and whether a model's generated order flow predicts it. Two
departures from the existing return bench: the horizon is **wall-clock time** rather than
a message count, and the anchor is **the instant of execution** rather than a window
boundary.

**What was measured.** Four released checkpoints from the SP500 scaling ladder
(1M / 10M / 50M / 200M, same seed, same training set), GOOG data, 1024 trade anchors
across 11 trading days with buy/sell balanced 512/512, 256 sequences per model.

**The finding, in one line.** **All four models lose on both direction metrics to a rule
that uses no model at all** — "the trade hit the bid, so predict a fall". The baseline
scores **93.4–94.3%** direction accuracy against the models' **74.7–77.7%**, a gap of
about 17 points. Across a 200× parameter range every confidence interval overlaps every
other, so capacity buys nothing distinguishable.

**Why the baseline is that strong** (the core of this document): **it is not predicting,
it is bookkeeping.** A trade consumes liquidity from the side it hit, that side thins,
and the best quote moves accordingly. Three measurements: the **first mid move after the
trade agrees 90.9%** of the time; in windows with **no follow-on trade at all** the rule
still scores **92.1%**; and taking p₀ *before* the trade instead — which would fold the
trade's own impact into the return — moves it only from 93.85% to 95.45%, so **the main
effect is not the instant of execution**.

**The baseline is not looking ahead** (§2.5, four checks): no message after t₀ exists in
the conditioning window; **shuffling the pairing drops it from 93.85% to 50.23%** (leaked
information would survive shuffling); using the *previous* trade's side instead drops it
to 78.33% (a real microstructure effect decays as the signal ages; a leak would not).
Everything the baseline uses, the model saw during generation.

**The models carry no information the baseline lacks** (§3.7). On the 44 samples the
baseline gets wrong, the models hit **36.4%** — below a coin (p = 0.976). Samples where
only the baseline is right outnumber those where only the model is right by about 8×.
This is "baseline plus noise", not "catches what the baseline misses". (Caveat: 44
samples give low power, so this is *no evidence of* rather than *evidence against*.)

**What that means.** All three headline metrics (IC, Ranked IC, Direction Accuracy) are
**dominated by direction, and direction is largely mechanical**. The generative model's
only increment is in **magnitude** — among samples whose sign is already right, the rank
correlation between |predicted| and |realised| is **0.115**, which the baseline cannot
compute at all because it emits one magnitude for every sample. Weak, but real, and
structurally out of the baseline's reach.

**So the problem is the task, not the model.** A yardstick meant to separate models must
leave its predictable part *to* the models. Here 93% is taken by a mechanical rule and
only 7% remains — **discriminating four models on 7% of the signal has almost no power**,
which is exactly why the four intervals overlap completely.

**Cross-ticker replication** (§4, four tickers spanning 328× in message rate, same month,
same source): **all four lose to the baseline**, so the finding is not specific to GOOG.
The extreme case is NVR, where model direction accuracy falls to **46.58% — below a
coin** — while the baseline still scores 86.30%; NVR also has the highest share of
zero returns (50.7%), and the model has not learned that **most of the time nothing
should happen**. All four clock ratios sit in 0.96–1.13, so this particular deficit
cannot be blamed on emitting messages too fast or too slow.

**Where the room is.** The baseline decays with horizon, and **how fast depends on how
quickly liquidity is replenished** — NVDA loses 22.5 points between 1 ms and 500 ms
while the thin NVR loses 1.6. Busy names replenish fast and the trade's imprint is
washed out; thin names have no incoming flow and the imprint simply stays. **Sections 1–3
measured where the baseline is strongest**: GOOG at 10 ms leaves 6–11 points, while
**NVDA at 500 ms leaves about 30**. That is the next run.

**Five defects found along the way** (§6, each forced out by a check rather than
guessed): the index manifest was changing the do(Q) injection's byte-identity digest;
the same trading day was being drawn twice; `get_date` returned the wrong day under
explicit windows; attached inference preallocated 90% of a neighbour's GPU memory; and
`searchsorted` silently degraded an out-of-range horizon into "the end of the sequence".

---

## 1. What problem this solves

### 1.1 The existing return bench measures something else

`AlphaTrade/lob_pipeline/return_bench/run_return_bench.py` is the only place in the repo
that computes IC. Its horizon is a **message count** (`idx = h - 1`) and its anchor is
fixed at conditioning message 250. It does read the timestamps, but only to print a
`dt mean` column afterwards — they **take no part in any alignment**.

A trader cares where the price is 10 ms after a fill, not how many messages arrived
meanwhile. This is the event-study framing used in market microstructure.

### 1.2 The anchor is the harder half

`LOBSTER_Dataset` places windows on a fixed grid:

```
seq_start = seq_offsets[file] + seq_idx × n_messages
```

Windows do not overlap and the stride equals `n_messages`. **What kind of message a
conditioning window ends on is therefore decided by the grid, not chosen.** Measured over
2000 GOOG windows:

| last conditioning message | limit submission | partial cancel | full cancel | **trade** |
|---|---:|---:|---:|---:|
| share | 49.40% | 0.20% | 48.45% | **1.95%** |

Only 1.95% land on a trade. **Selecting dataset indices cannot move a boundary the grid
has fixed**, so the dataloader needs an explicit-window bypass. With targeted windows the
anchor purity is 100%, a **51× gain in sample efficiency**.

### 1.3 An objection that has to be answered head-on

sigma-0's own `src/lobpipeline/return_bench/event_anchored.py` opens with:

> "The horizon is counted in **events**, never in seconds. A generator whose clock runs
> slow will still be scored on the same number of book updates as the real stream, so a
> clock error cannot masquerade as a return-distribution improvement."

Counting in milliseconds is precisely what that avoids. The two do not conflict; they
measure different things. **Event counting holds the clock fixed** and asks only about
price dynamics. **Millisecond counting puts the clock itself under test** — a model that
emits messages too slowly will not move the price enough inside 10 ms, and that is a real
defect. The cost is that a degraded IC no longer separates "bad price prediction" from
"bad timestamps", **so clock diagnostics are printed beside every metric** and IC is
never reported alone.

---

## 2. What the baseline is, and why it scores so high

This is the most important section.

### 2.1 The baseline is one line

```python
def trade_sign_baseline(directions):
    return -directions
```

- `direction = +1`: the trade consumed an order resting on the **bid** → someone sold
  actively → predict the mid **falls**
- `direction = -1`: the trade consumed an order resting on the **ask** → someone bought
  actively → predict the mid **rises**

**It uses no model output.** That field is the **last message of the conditioning
window**, fully visible to the model.

Direction Accuracy here means: among samples where the mid actually moved
(`|real| > 1e-8`, about 86%), how often "which side the trade hit" alone already gave the
correct direction of the 10 ms move.

### 2.2 Three measurements showing it is not prediction

Figure (mechanism): stored in the results database, entry `sigma0-relocated-figures-20260905`, file `/lus/lfs1aip2/projects/public/u6gb/results/content/benchmarking/sigma0-relocated-figures-20260905/artifacts/fig_mechanism.png`.

**(left) The trade consumes one side.** The trade itself moves the mid in only **24.9%**
of anchors — but **when it does, it moves it the implied way 100% of the time**. The
third bar matters more: the **first mid move after the trade agrees with it 90.9%** of
the time.

**(centre) Follow-on flow is persistent.** Among follow-on trades inside the 10 ms
window, **69.8% are on the same side** as the anchor (a coin would give 50%). This is
order-flow autocorrelation, the classic result from the Bouchaud line of work.

**(right) But follow-on flow is not required.** Splitting the sample: windows with
follow-on trades score **95.0%**, and windows with **no follow-on trade at all still
score 92.1%** — only 2.9 points lower.

> **So the mechanism is liquidity removal, not order-flow autocorrelation.** The trade
> eats the resting orders on one side, that side thins, and from there even cancellations
> alone push the best quote that way. Same-side follow-on flow reinforces it, but
> removing it does not change the conclusion.

### 2.3 Ruling out a simpler explanation: is p₀ taken at the wrong point?

If p₀ were taken **before** the trade, the return would include the trade's own immediate
impact and 93% would be a definition rather than a finding.

Both variants measured:

| how p₀ is taken | baseline DirAcc |
|---|---:|
| **after the trade** (what the bench uses; correct) | **93.85%** |
| before the trade (counterfactual) | 95.45% |

**A gap of 1.6 points.** Even with the trade's own impact fully excluded, the direction
remains 93.85% predictable.

> In code this rests on an easy-to-miss detail: `data_cond`'s book has `n_cond+1` rows
> while its message file has `n_cond` (`inference_no_errcorr.py:1825`,
> `b_seq_pv[:, :n_cond+1, 3:]`), so `cond_mid[-1]` is the book **after** the anchoring
> trade. **One row off and you read the price from before it.**

### 2.4 What the baseline cannot do

The baseline emits ±1, i.e. **one magnitude for every sample**, so it cannot rank
magnitude at all:

```
distinct |values|:  baseline 1,  model 180
```

| predictor | Pearson IC | Ranked IC | DirAcc | **magnitude rank corr** |
|---|---:|---:|---:|---:|
| one-line rule | 0.605 | 0.774 | **93.37%** | **undefined** (nan) |
| model 50M | 0.523 | 0.683 | 77.35% | **0.115** |

**This is the one place the generative model is ahead.** Weak (0.115), but non-zero, and
structurally out of the baseline's reach.

### 2.5 Is the baseline looking ahead?

A rule scoring 93.85% should first be suspected of cheating. Four checks:

| # | check | result |
|---|---|---|
| 1 | any message after t₀ inside the conditioning window | **0 / 225**. The baseline reads one field: `direction` of the last conditioning message |
| 2 | does the scored return start after t₀ | yes. 92 samples have their first continuation message **at the same instant** as t₀ (several messages per microsecond), which is not an overrun |
| 3 | **shuffle the pairing** | 93.85% → **50.23% ± 3.56%** (2000 draws) |
| 4 | use the **previous** trade's side instead (equally causal, just staler) | **78.33%**, far below 93.85% |

**Check 3 is decisive.** If the baseline had seen the future, that information would
travel with the return and **would survive shuffling**. It drops to coin level, so the
skill lives entirely in the pairing between **this** trade and **this** window's future.

**Check 4 corroborates.** Decay with the age of the signal is the shape of a real
microstructure effect; a leak would not decay that way.

**Conclusion: no leakage.** Everything the baseline uses, the model saw during generation.

---

## 3. Results for the four models

Figure (capacity): stored in the results database, entry `sigma0-relocated-figures-20260905`, file `/lus/lfs1aip2/projects/public/u6gb/results/content/benchmarking/sigma0-relocated-figures-20260905/artifacts/fig_capacity.png`.

### 3.1 Main table (10 ms; errors from a day-block bootstrap)

| model | params | N | days | Ranked IC [95% CI] | Direction Accuracy [95% CI] | Pearson IC |
|---|---:|---:|---:|---|---|---:|
| `m3-sp500-1m-s42` | 1M | 220 | 11 | **0.6876** [0.6033, 0.7509] | **75.39%** [71.01, 79.41] | 0.146 |
| `m3-sp500-10m-s42` | 10M | 200 | 11 | **0.6604** [0.5423, 0.7619] | **74.71%** [65.95, 83.43] | 0.534 |
| `m3-sp500-50m-s42` | 50M | 211 | 11 | **0.6833** [0.6181, 0.7512] | **77.35%** [73.60, 82.43] | 0.523 |
| `m3-sp500-200m-s42` | 200M | 202 | 11 | **0.6089** [0.5047, 0.7161] | **77.71%** [73.46, 83.23] | 0.509 |

**All four 95% intervals overlap. A 200× parameter range buys no distinguishable
improvement.**

### 3.2 Against the baseline

| model | Ranked IC | baseline | Direction Accuracy | baseline | clears it? |
|---|---:|---:|---:|---:|:---:|
| 1M | 0.6876 | **0.7833** | 75.39% | **94.24%** | **no** |
| 10M | 0.6604 | **0.7806** | 74.71% | **93.68%** | **no** |
| 50M | 0.6833 | **0.7742** | 77.35% | **93.37%** | **no** |
| 200M | 0.6089 | **0.7968** | 77.71% | **94.29%** | **no** |

### 3.3 Pearson IC is not usable here

On the same samples Pearson IC reads 0.146 / 0.534 / 0.523 / 0.509 — a **3.7× spread**
across models whose rank correlation is flat. The 1M column is not even monotone in
horizon (0.24 → 0.11 → 0.15 → 0.15 → 0.45).

Log returns are heavy tailed and a handful of outliers dominate the linear correlation.
**Reporting Pearson alone would claim "10M is 3.7× stronger than 1M", which the ranks do
not support.**

> This only surfaced when the sample grew from 38 to 255. At 38 samples Pearson IC was
> 0.83 and looked convincing — a coincidence of small samples and extreme values.

### 3.4 Horizons and the clock

Figure (horizons): stored in the results database, entry `sigma0-relocated-figures-20260905`, file `/lus/lfs1aip2/projects/public/u6gb/results/content/benchmarking/sigma0-relocated-figures-20260905/artifacts/fig_horizons.png`.

| model | 1 ms | 5 ms | 10 ms | 25 ms | 50 ms |
|---|---:|---:|---:|---:|---:|
| 1M Ranked IC | 0.687 | 0.683 | 0.688 | 0.587 | 0.529 |
| 10M | 0.711 | 0.663 | 0.660 | 0.682 | 0.664 |
| 50M | 0.717 | 0.689 | 0.683 | 0.619 | 0.678 |
| 200M | 0.687 | 0.636 | 0.609 | 0.530 | 0.542 |
| **clock ratio** 1M | 0.646 | 0.667 | 0.689 | 0.758 | 0.750 |
| 10M | 1.027 | 1.151 | 1.190 | 1.260 | 1.198 |
| 50M | 0.923 | 0.968 | 1.037 | 1.102 | 1.047 |
| 200M | 0.962 | 1.149 | 1.172 | 1.202 | 1.150 |

Clock ratio = generated messages ÷ real messages inside the same millisecond window.
**1M is clearly too slow (0.65–0.75), 10M and 200M overshoot (1.15–1.26), 50M is closest
to 1.0.** Not a monotone improvement.

Coverage falls with horizon (79–86% at 10 ms, 56–73% at 50 ms) because a sample counts
only when **both** streams reach t₀+h — so **coverage is itself a reading of model
quality**: a generator that emits too slowly runs out of sequence sooner.

### 3.5 By trading day

Figure (by day): stored in the results database, entry `sigma0-relocated-figures-20260905`, file `/lus/lfs1aip2/projects/public/u6gb/results/content/benchmarking/sigma0-relocated-figures-20260905/artifacts/fig_by_day.png`.

The spread in this figure *is* the confidence interval above. **The trading day is the
resampling unit**, because anchors within a day share a regime and neighbouring anchors
can overlap the same order flow. Resampling anchors instead would narrow the errors by
roughly 3×, at which point the four models would **appear pairwise distinguishable** —
an artefact of treating correlated samples as independent.

### 3.6 By trade direction (real data; independent of the model)

Figure (direction): stored in the results database, entry `sigma0-relocated-figures-20260905`, file `/lus/lfs1aip2/projects/public/u6gb/results/content/benchmarking/sigma0-relocated-figures-20260905/artifacts/fig_direction.png`.

| model | side hit | n | realised mid move | share down |
|---|---|---:|---:|---:|
| 1M | bid (seller-initiated) | 111 | **−0.557 bp** | **93.0%** |
| 1M | ask (buyer-initiated) | 109 | **+0.494 bp** | 4.4% |
| 50M | bid | 104 | −0.558 bp | 92.5% |
| 50M | ask | 107 | +0.474 bp | 5.7% |

This column is near-identical across models because **it is a property of the real
data**. The signs match price-impact theory exactly, which is an independent check that
the numbers are wired correctly.

### 3.7 Do the models carry information the baseline lacks?

"Model 77% versus baseline 93%" has two very different readings, and those two numbers
look identical under both:

- **Reading A**: model = baseline + noise. It occasionally loses what the baseline gets,
  and also misses what the baseline misses → no independent information.
- **Reading B**: the model catches what the baseline cannot, drowned out by the 94% the
  baseline already owns → real added value.

They separate on the **samples where the two disagree** (10 ms, mid actually moved):

| model | both right | **only baseline** | **only model** | both wrong | model hit rate where baseline is wrong |
|---|---:|---:|---:|---:|---:|
| 1M | 142 | **38** | **2** | 9 | 2/11 = 18.2% |
| 10M | 126 | **37** | **4** | 7 | 4/11 = 36.4% |
| 50M | 136 | **33** | **4** | 8 | 4/12 = 33.3% |
| 200M | 130 | **35** | **6** | 4 | 6/10 = 60.0% |
| **total** | | | **16** | | **16/44 = 36.4%, p = 0.976** |

**It is reading A.** "Only baseline right" outnumbers "only model right" by about 8×, and
on the 44 samples the baseline gets wrong the models hit 36.4% — below a coin, with a
binomial test that is not close to significant.

> **An honest limit on sample size**: the baseline errs on only 44 samples in total. Power
> is low at that scale, so the accurate statement is "**no evidence that the model carries
> information beyond the baseline**", not "proof that it does not". Settling it needs more
> anchors — and by §3.5's logic, more **trading days** rather than more anchors inside the
> same days.

---

## 4. By ticker: where the baseline gives way

Sections 1–3 all sit on GOOG at 10 ms. This section asks two things: whether "the
baseline owns direction" is specific to GOOG, and **where the baseline lets go** — which
is where a model has room.

### 4.1 Ticker selection: 328× in liquidity

Four tickers from the SP500 SquashFS shards, **same month (2026-02), same
preprocessing**, so they are directly comparable:

| ticker | msgs/day | vs GOOG | trades in the month |
|---|---:|---:|---:|
| NVDA | 9,845,761 | 1.7× | 607,430 |
| GOOG | 5,650,327 | 1× | 169,805 |
| KO | 402,430 | 1/14 | 20,732 |
| **NVR** | 29,999 | **1/188** | **155** |

> NVR has 155 trades in the entire month; after discarding those whose window does not
> fit, **only 148 anchors exist** against the 256 requested. A thin name cannot even
> supply the sample, which is itself a result.

### 4.2 The baseline decays with horizon, at a rate set by liquidity

**The table below uses real data only, with no model output**, so it needs no GPU. Each
ticker uses that month's last 2 trading days; sample sizes are 352–40,000 (NVR is small
because the month holds 353 trades in total).

Figure (decay): stored in the results database, entry `sigma0-relocated-figures-20260905`, file `/lus/lfs1aip2/projects/public/u6gb/results/content/benchmarking/sigma0-relocated-figures-20260905/artifacts/fig_decay.png`.

**Baseline direction accuracy (%)**

| ticker | msgs/day | 1 ms | 5 ms | 10 ms | 25 ms | 50 ms | 100 ms | **500 ms** | **lost** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **NVDA** | 9.8M | 92.06 | 89.00 | 87.36 | 84.53 | 82.15 | 78.93 | **69.54** | **−22.5** |
| **GOOG** | 5.6M | 93.55 | 90.45 | 89.05 | 86.53 | 84.51 | 82.19 | **71.81** | **−21.7** |
| **KO** | 402K | 94.14 | 92.49 | 91.84 | 91.19 | 90.49 | 89.49 | **86.77** | **−7.4** |
| **NVR** | 30K | 84.95 | 85.13 | 84.04 | 84.78 | 84.32 | 84.04 | **83.33** | **−1.6** |

**Share of anchors where the mid actually moved (%)** — the precondition for any decay

| ticker | 1 ms | 5 ms | 10 ms | 25 ms | 50 ms | 100 ms | 500 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| NVDA | 76.2 | 77.9 | 78.5 | 78.7 | 79.8 | 80.6 | 84.9 |
| GOOG | 84.0 | 86.2 | 86.9 | 87.5 | 88.2 | 89.3 | 91.6 |
| KO | 77.3 | 78.4 | 78.6 | 78.9 | 79.3 | 79.4 | 79.5 |
| **NVR** | **52.8** | 55.4 | 53.4 | 52.3 | 52.6 | 53.4 | **56.2** |

**Two regularities:**

1. **The busier the name, the faster the baseline decays.** NVDA and GOOG lose 22 points
   between 1 ms and 500 ms, KO loses 7.4, NVR barely moves (1.6).
2. **NVR's mid does not move at all half the time** (52–56%, against 76–92% for the other
   three). Its baseline is the lowest in absolute terms (84–85%) and also the flattest.

**Mechanism**: busy names replenish liquidity quickly, so the imprint a trade leaves on
the book is washed out by fresh order flow and "which side the trade hit" goes stale
fast; thin names have almost no incoming flow, so the imprint simply stays.

### 4.3 This is what sets the model's opportunity window

Putting the two together:

| | baseline | room left for a model |
|---|---:|---|
| GOOG @ 10 ms (what §1–3 measured) | 89–94% | **about 6–11 points** |
| NVDA @ 500 ms | **69.5%** | **about 30 points** |

**So the earlier "the model loses to the baseline" was obtained where the baseline is
strongest.** It is not that the model has no room; **this horizon has almost no room to
give** — which also explains why the four confidence intervals overlap completely: the
separable part is those 6–11 points, and no sample size fixes that.

The next run should be **NVDA / GOOG × 500 ms–2 s**. The prerequisite is raising
`n_gen_msgs` from 250 to roughly 700 (250 messages span 0.64 s at the median; see the
coverage figures in §3.4).

### 4.4 The models measured on all four tickers

One model (50M) across four tickers, 256 sequences each (NVR could only supply 148),
at 10 ms:

| ticker | msgs/day | N | zero return | model Ranked IC | baseline | model Dir acc | baseline | clears it? | clock |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| **NVDA** | 9.8M | 201 | 25.9% | 0.6156 | **0.6883** | 71.14% | **89.93%** | **no** | 1.02 |
| **GOOG** | 5.6M | 211 | 14.2% | 0.6402 | **0.7365** | 71.82% | **92.27%** | **no** | 1.13 |
| **KO** | 402K | 242 | 22.7% | 0.6016 | **0.7429** | 67.91% | **93.58%** | **no** | 1.05 |
| **NVR** | 30K | 148 | **50.7%** | 0.4144 | **0.5199** | **46.58%** | **86.30%** | **no** | 0.96 |

**Three conclusions:**

1. **All four lose to the baseline.** The result from §1–3 is not a property of that one
   ticker; it holds across 328× in liquidity.
2. **On NVR the model's direction accuracy is 46.58% — below a coin**, while the baseline
   still scores 86.30%. On the thinnest name the model does not merely lack information,
   it **systematically gets the direction wrong**. NVR also has the highest share of zero
   returns (50.7%), so the model has plainly not learned that **most of the time nothing
   should happen**.
3. **The clock is not the excuse this time.** All four ratios sit in 0.96–1.13, unlike
   the 1M model's 0.65 in §3.4.

**Coverage runs opposite to liquidity**, which is counter-intuitive but mechanically
clear:

| ticker | 1 ms | 5 ms | 10 ms | 25 ms | 50 ms | 100 ms |
|---|---:|---:|---:|---:|---:|---:|
| NVDA | 100% | 90.8% | 84.5% | 72.3% | 63.9% | **49.2%** |
| GOOG | 100% | 91.3% | 83.1% | 70.9% | 58.3% | **51.6%** |
| KO | 100% | 97.3% | 94.5% | 91.8% | 90.6% | 88.3% |
| **NVR** | 100% | 100% | 100% | 100% | 100% | **100%** |

250 messages of a busy name span very little time, so 100 ms runs off the end of the
sequence; 250 messages of a thin name span a long time and 100 ms is reached easily.
**So "lengthen the horizon" needs a larger `n_gen_msgs` on busy names and none on thin
ones** — and it is precisely the busy names that have 30 points on offer.

### 4.5 Reproducing this

```bash
# Mount, pick anchors and generate — all in one srun. The farm path contains a process
# id, so a second step resolves a different file list and inference will (correctly)
# refuse the mismatch.
$T/run_ticker.sh <TICKER> m3-sp500-50m-s42 256 8

# The baseline alone needs no GPU; read the raw npy directly:
squashfuse $SHARD/shard_2026-02.squashfs $MNT
python3 baseline_only.py $MNT/<TICKER> <TICKER> 2
```

> ⚠️ **The raw npy book is 43 columns**: `[mid_diff, time_s, time_ns]` followed by 40
> columns of L10 book, so **ask_p1 is column 3 and bid_p1 is column 5**. The inference
> output csv is the same book with those first three dropped
> (`inference_no_errcorr.py:1825`), which is where columns 0 and 2 are correct.
> Using the latter's indices on the former reads `mid_diff` and `time_ns` as prices,
> marks nearly every row as a missing quote, and **cuts the sample from 962,120 to 16
> without raising anything**. It was caught because "962k trades but 16 samples" does not
> add up.

---

## 5. Why these numbers are checkable

Three layers, **each catching what the layer above cannot**.

### 5.1 Identity check (the strongest, and free)

Feed the real stream in as if it were generated. Across all 7 horizons:

```
IC = 1.000000000000    Ranked IC = 1.000000000000    DirAcc = 100.0000%    KS = 0.000
```

Largest deviation **1.1e-16**. Mispaired files, a timestamp off by one row, the wrong
book row, p₀ taken before the anchoring trade — any of them drops it below 1.0.
**It needs no GPU and no model, so there is no excuse for a result that has not passed
it.**

### 5.2 Shuffle null (what the no-skill level actually is)

Break the pairing inside each trading day, 800 draws:

| statistic | null level |
|---|---|
| Ranked IC | +0.036 ± 0.068 |
| Direction Accuracy | **41.5–45.4%** (**not** 50%, because ups and downs are not balanced) |

Without this, 47% reads as "worse than chance".

### 5.3 Mutation check (`ci/mutation_check.py`, the only layer that trusts nothing)

Five specific defects are planted in the metric, and **the suite is required to fail on
each**:

| planted defect | what it looks like on real data |
|---|---|
| drop the end-of-sequence check | a long horizon silently becomes "the return at the end of the sequence"; corrupts 42% of 500 ms samples over 2000 sequences |
| `searchsorted` right → left | skips a message sharing the target timestamp, and real data has several per microsecond |
| drop the `-1` | reads the row beyond the horizon — a look-ahead that makes results **better** |
| drop the trade-anchor requirement | event-anchored evaluation degrades into ordinary windowed evaluation, with every number still plausible |
| perturb the identity path by 1e-4 | a check that must return exactly 1.0 returns approximately 1.0 |

**All five are caught.** This step reads no evidence file; it only executes code.

> **Writing this check taught its own lesson.** The first version ran the tests from the
> real `tests/` directory with `PYTHONPATH` pointed at the mutant, and **all five mutants
> survived** — because `conftest.py` does `sys.path.insert(0, ROOT/"src")`, so no mutant
> was ever imported. It reported the exact opposite of the truth, in precisely the shape
> of the error it exists to detect. Each mutant now gets a complete tree, and a mutation
> whose pattern matches nothing is an error rather than a survivor.

### 5.4 The boundary of the evidence mechanism (stated plainly)

`ci/evidence/return_bench.json` holds both the numbers **and the conditions they must
satisfy, written by the same run**. Auditing it proves the numbers have not drifted from
the code that produced them — which has already caught real drift twice — but it
**cannot prove the numbers were right to begin with**: a wrong implementation would have
recorded a wrong number, and the assertion could have been written to accommodate it.

What can be enforced: source hashes make "re-record after editing" a rule;
`recorded.commit` must be an ancestor of HEAD; `recorded.host` must not be a CI runner.
**What cannot**: a hand-written evidence file. The recording happens off CI and no
auditable mechanism reaches it. **The mutation check is the one link that does not depend
on trusting the recorder.**

---

## 6. Five defects found during implementation

None was guessed; each was forced out by a specific check.

### 6.1 The index manifest was changing the do(Q) injection digest

`sample_indices_rank0.json` is hashed into the injection verification's arm digest
(`injection_verification.py`, `list_rel_files`). Three fields were added to it
unconditionally, so **every rollout's digest would move**, including runs that never
asked for event-anchored windows — breaking a byte-identity claim measured on one
specific node that cannot be re-measured elsewhere.

**The judgement that failed was "a new optional CLI flag cannot affect an injection
measurement"** — the manifest was the path not accounted for. The three fields are now
added only inside the branch that uses them, and the manifest is proven byte-identical
(sha256 `73b6afe4…`) when the feature is off. The evidence hash drift is what forced the
investigation.

### 6.2 The same trading day was being drawn twice

Three of eleven days exist under two filenames differing only in the LOBSTER time range.
They are **not two sessions**: preprocessing truncated both to 09:30–16:00, both have
8,386,647 rows, and they agree on `order_id`, `event_type`, `direction`, `size` and every
timestamp — **only the relative price columns differ**.

Drawing from both made one execution appear as two independent anchors, affecting
**463 of 1024** picked anchors. That inflates the sample count an IC's significance is
judged against, and nothing about "1024 anchors" reveals it.

### 6.3 `get_date` returned the wrong day under explicit windows

`get_date()` resolved the index through `_get_seq_location` (the grid path) rather than
`_window_bounds`, so **every output filename carried the wrong trading day** — 256
windows spanning 11 days all claimed `2026-02-05`.

It does not affect return values (pairing goes by `real_id`), so it stayed invisible —
**it surfaced only when the day-block bootstrap collapsed to a single block and returned
a zero-width interval**. That output looks like unusually high precision, which is the
most believable shape an error can take. Fixed, and no regeneration was needed: the
manifest's `anchor_windows` recovers `real_id → file_idx → day`.

### 6.4 Attached inference preallocated 90% of a neighbour's GPU

`XLA_PYTHON_CLIENT_MEM_FRACTION` was assigned unconditionally at import, so the
environment could not override it. When a training job already holds 24 GB, "90% of the
total" is not "90% of what is free". Now `setdefault`.

### 6.5 `searchsorted` truncates silently

`np.searchsorted(t, t0+h, 'right') - 1` returns the **final row** when the sequence ends
before t₀+h — no error, no NaN. Measured over 2000 real sequences: **42% of the 500 ms
samples were really "the return at the end of the sequence"**.

The first measurement, on 80 samples, reported `miss = 0` and looked perfectly healthy.
The check is now an explicit `t[-1] >= t0 + h`.

---

## 7. Conclusions and next steps

### 7.1 Conclusions

1. **The event-anchored return bench works and is wired into CI**, with identity, shuffle
   and mutation checks in place.
2. **All four SP500 models lose to a rule that uses no model**, by about 17 points on
   direction.
3. **1M → 200M shows no distinguishable difference**; all four intervals overlap.
4. **Direction at this scale is largely mechanical** (liquidity removal), so it should
   not be the primary axis of judgement.
5. **The model's only increment is magnitude** (rank correlation 0.115), which currently
   has no headline metric of its own.
6. **Baseline strength is not a constant**: it decays with horizon at a rate set by
   liquidity (−22.5 points on a busy name, −1.6 on a thin one). **Conclusions 1–5 were
   all obtained where the baseline is strongest.**
7. **All four tickers across 328× in liquidity lose to the baseline**, so conclusion 2 is
   not a GOOG artefact. On the thinnest name model direction accuracy falls to **46.58%
   (below a coin)**, and half the samples there have zero return — the most important
   thing to learn is "nothing happens", and the model has not learned it.

### 7.2 What this implies for the bench itself

All three headline metrics are dominated by direction, and direction is now shown to be
largely mechanical. **The next version's headline should be magnitude, with direction
demoted to a gate that must be cleared.** `magnitude_skill()` is implemented and wired
into the output, but it is not yet the first column of the main table.

### 7.3 Next steps, by value

| priority | what | why |
|---|---|---|
| **highest** | **NVDA / GOOG × 500 ms–2 s** | §4 measures the baseline falling to 69.5% there, leaving about 30 points. The current 10 ms leaves 6–11, where no sample size separates models |
| high | **promote magnitude to the main table** | direction is taken by the baseline; the model's actual increment is 0.115 there, which the baseline cannot compute |
| high | raise `n_gen_msgs` to about 700 | prerequisite for the above: 250 messages span 0.64 s at the median, so 500 ms is only 58% covered |
| medium | a stronger baseline | "trade sign + book imbalance". **If a five-field linear rule reaches 96%, this horizon should be replaced entirely** |
| medium | more trading days per ticker | this round's `test_split=0.1` truncated to one day per ticker, leaving no cross-day error estimate |
| low | more anchors | the error is dominated by cross-day heterogeneity; more anchors within the same days has diminishing returns |

---

## Appendix: how to reproduce

```bash
WT=/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees/return-bench-event-20260817
T=/lus/lfs1aip2/projects/public/u6gb/tasks/return_bench_scaling_20260817

# 1. Pick anchors (pure CPU; safe on a login node)
python3 $WT/run/base_model/eval/pick_trade_anchors.py \
    --data_dir $(readlink -f $WT/data/GOOG_data_npy)/GOOG \
    --out $T/anchors/goog_anchors.json \
    --n_cond_msgs 250 --n_gen_msgs 250 --n_samples 1024 --test_split 0.02 --seed 42

# 2. Generate (attach to a live allocation; cap memory so the neighbour is unaffected)
ALLOC=<allocation id> NSEQ=256 BSZ=8 $T/launch_ladder.sh

# 3. Score
PYTHONPATH=$WT/src python3 -m lobpipeline.return_bench.run_event_bench \
    --infer_dir $T/infer/<model> --horizons 1,5,10,25,50 --shuffle-null

# 4. Analysis and report (day-block bootstrap, figures, markdown)
python3 $T/report.py && python3 $T/figures.py && python3 $T/fig_mechanism.py \
    && python3 $T/make_report.py

# 5. Self-checks
PYTHONPATH=$WT/src python3 -m lobpipeline.return_bench.run_event_bench \
    --infer_dir <any result dir> --identity-check    # must be exactly 1.0
python3 $WT/ci/mutation_check.py                     # all five planted defects must be caught
```

**Code**: this repository. `src/lobpipeline/return_bench/{event_anchored,run_event_bench}.py`
is the metric itself, `run/base_model/eval/pick_trade_anchors.py` picks the anchors, and
the return-bench steps of `.github/workflows/checks.yml` are the audit.

**Data, outputs and analysis scripts**: on the cluster at
`/lus/lfs1aip2/projects/public/u6gb/tasks/return_bench_scaling_20260817/`, where
`anchors/` holds the anchors, `infer*/` the generated output (446 MB, not committed),
`report/` the figures and this document, and
`report/{results,tickers,baseline_decay}.json` every number in it.

The analysis scripts (`report.py`, `ticker_report.py`, `figures.py`, `fig_mechanism.py`,
`fig_decay.py`, `make_report.py`, `baseline_only.py`, `run_ticker.sh`) live there too.
They hard-code that working directory, being one-off experiment analysis rather than
portable tooling, so they stay in the experiment area rather than the repository. Every
number here can be recomputed from the three json files above without rerunning them.

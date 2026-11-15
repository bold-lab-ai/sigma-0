# Policy bench

Evaluate one trading policy against a fixed simulator and a fixed signal, and
report its result. Policy search is a separate client of this package.

- Simulator: nano-sigma `newtok_a_r1` (best LOB-Bench on the nano-sigma board).
- Signal: nano-sigma `returnft_lobert_pass1` (best return and PnL bench).
- Data: train, dev and test splits by date, all after the models' training data (below).

The state is the L3 order book plus the message log, not tokens. Each model
encodes its own input from that state.

## Data splits

Two layers. The frozen models were trained and tested on
`dataset_q4_2025_jan2026_v3`; the bench's own splits all lie after the models'
training data, because on those months the signal is in-sample and too good.

| layer | split | dates | use |
|---|---|---|---|
| models | train | 2025-10 to 2025-12 | trained the simulator and the signal |
| models | val | 939 windows inside the train months | validation while training |
| models | test | 2026-01 | the models' test scores |
| bench | train | 2026-01-02 to 01-15 (10 days) | fit a policy's parameters |
| bench | dev | 2026-01-16 to 01-30 (10 days) | compare and select policies; never fit on it |
| bench | test | 2026-02-02 to 02-13 (10 days) | score a frozen policy, once |

Bench train and dev are the models' test month; bench test was built for the
bench (see `/projects/public/u6oz/policy_bench/data/README.md`) and no model
has seen it. A date belongs to one split only, because episodes of one date
share the market. `splits.py` holds the dates; `make_episodes.py --split` draws
from one split only, and `evaluate.py` refuses an episode outside the config's
`split`. Nothing before 2026-01-01 may be used by a policy.

## Modules

| Module | What it does |
|---|---|
| `book.py` | L3 book. FIFO queue per price, an owner (background or agent) on every order. Updated by order id for real data and by queue slot for generated messages. `L10Writer` gives the 43-column L10 row the models read. |
| `daydata.py` | One stock-day of the v3 dataset and the replay that rebuilds its full book from the stored snapshots. `build_add_index` gives each order id's first add, stored once per day by `run/benchmarking/policy_bench/build_add_index.py`. |
| `codec.py` | The newtok `Message`: `encode` (identical to nano-sigma's `encode_window`) and a strict `decode`. |
| `simulator.py` | The newtok model with a KV cache. `BatchKVSimulator` holds several sequences of their own lengths: prefill one row, step every row by one token, cut back one row. `KVSimulator` is the one-row case. |
| `signal.py` | The return signal: processed rows from our book (`price_rel`, the reference columns), and the fine-tuned head's prediction at a decision. |
| `rules.py` | Legality of a background message in a book (what the sampler rejects), and how a legal message changes the book. |
| `types.py` | Plain data: `MarketOrder`, `LimitOrder`, `Cancel`, `Fill`, `Obs` (with `pending`, the orders on their way), `Episode`. |
| `history.py` | `RowLog`, the processed rows and L10 rows both models read, and `start_state`, an episode's book and log rebuilt from a real day. |
| `market.py` | `ReplayMarket`: real rows, virtual agent orders, market orders only, fill model `MidFill` or `BookFill`, latency (`latency` messages or `latency_ns`) and inventory cap. |
| `sampling.py` | How a background message is drawn: the message grammar with the absolute time forced to t_prev + dt, `finish` (decodes, canonical, before the close, `rules.check`, both sides left non-empty) and `legal_tokens`, the per-token masks used after 32 rejections. |
| `sim_market.py` | `SimMarket`: B slots decoded together with `BatchKVSimulator`, each row on its own (a row stops only at its decisions and its end). Agent orders are fed to the simulator as forced messages; background messages are drawn whole and rejected until legal; executions fill agent orders front of queue first. With `latency_ns`, a background draw later than an order's due time is discarded and drawn again after the order. |
| `rollout.py` | `rollout(policy, signal, market, episodes, every)` -> one `Trace` per episode. Keeps `market.batch` slots busy (a slot takes the next episode when its own ends), one policy copy per episode, the signal one row per forward. |
| `trace.py` | `Trace` (mids, end book as L10 and as full depth `depth_end`, fills, decisions, signal) and `save` / `load` as one npz. |
| `costs.py`, `terminal.py`, `score.py` | `MakerTaker` fees; `MarkToMid`, `MarkToMicro`, `FlattenTaker`; `score(traces, terminal, cost)`, a pure function of the traces; the ranking figure is `daily_pnl_usd`. |
| `episodes.py` | Episode sets. `return_bench_decisions` is nano-sigma's test decision set. |
| `splits.py` | The train, dev and test dates; `check_episodes` refuses an episode set that leaves its split. |
| `policies/` | `NoTrade`, `SignRule`, `SkewQuoter`. |
| `nanosigma/` | Vendored nano-sigma encoding code, byte for byte. See `nanosigma/PROVENANCE.md`. |

## v1: what is fixed

`configs/bench/policy_bench_v1.yaml` (dev, 2026-01-16 to 01-30) and `policy_bench_v1_test.yaml` (test,
2026-02-02 to 02-13, run only on frozen policies). Both: 200 episodes of 1,024
background messages; the simulator's context is the last 2,944 real messages
and the signal's the last 3,968 rows; a decision every 64 messages; orders
reach the market 100 ms after the decision (`latency_ns`); masked sampling
(`max_rejections: 0`); maker rebate $0.002 and taker fee $0.003 per share;
inventory cap +-1,000 shares; one rollout per episode. The evidence for each
choice is in `docs/experiments/policy_bench_v1/README.md`.

## Writing a policy

A policy is a class with `reset(episode)` and `step(obs) -> list of orders`
(`policies/no_trade.py` is the smallest). `rollout` keeps one copy per
episode. At each decision `obs` holds:

- `k`, `t_s`, `t_ns`: background messages so far and the time of the last row
- `l10`: the top 10 levels (ask price, ask shares, bid price, bid shares per level)
- `position` (shares) and `cash` (LOBSTER units x shares)
- `open_orders`: `(oid, side, price, remaining)` of the agent's resting orders
- `pending`: orders submitted that have not reached the market yet
- `fills`: the agent's fills since the previous decision
- `signal`: the signal's predicted returns in bps at horizons 16, 64, 128,
  512, 1,024 messages

Orders: `MarketOrder(side, size)`, `LimitOrder(side, price, size)` (a whole
number of cents, not crossing), `Cancel(oid)`. An order is refused if it would
cross, take the position past the cap, or leave a side of the book empty.
v1 policies read only `obs`; they bring no model of their own.

Run one on dev:

    PYTHON=<env>/bin/python sbatch --nodes=1 --output=<dir>/%x_%j.out \
        run/benchmarking/policy_bench/evaluate.sbatch --policy <module:Class> --kwargs '<json>' --out <file.json>

Rank by `scores.flatten_taker.daily_pnl_usd` and its `daily_pnl_t`: the dollar
PnL per episode after closing the position against the full end book, averaged
within each date, then over dates. An episode whose position exceeds the whole
side of the end book is unpriced and counted in `unpriced`.

## Rules a background message obeys

- New orders do not cross the opposite best.
- A cancel or delete names one background order by queue slot. Agent orders
  count in the numbering but can never be named. QSLOT 99 means slot 99 or
  later.
- Executions walk one side from the best price, take inner levels whole, and
  are filled front of queue first. That is the only way an agent order is
  filled by background flow.
- No message, background or agent, may leave a side of the book empty. dP is
  counted from the mid, and with one side empty the mid is computed from the
  placeholder price (about -5e9 or +5e9): no price could be written after it.
- A message ends no later than the file's last time (16:00 or 16:05).

## Known approximation: strict FIFO

Real executions are not always front of queue first. On the 40 January
stock-days of job 7832642, 96.2% of execution messages (180,489 of 187,678)
fill exactly the orders a front-first walk predicts. The other 3.8% skip an
older order at the level, typically a 100-share order skipped by an aggressor
smaller than 100 shares. The cause is not in the data. Orders with a minimum
execution quantity would explain it, but that is unverified.

The bench keeps strict FIFO. For an agent order this is conservative: it is
never filled ahead of an older order, which in real markets it sometimes is.

## Gates

| Gate | Check | Where |
|---|---|---|
| G1 | Real-day replay: queue slots equal `<stem>_queue.npy`, top 10 levels equal the L10 file, every snapshot equals the stored one | `run/benchmarking/policy_bench/check_book_codec.py` |
| G2 | `encode_stream` equals `encode_window`; decode inverts encode up to clipping; real messages pass `rules.check`; a slot-driven book stays equal to the id-driven one | same script |
| G3 | The vendored simulator reproduces `newtok_a_r1`'s test CE on its 939 test windows; our book gives byte-identical model inputs; the KV cache equals the full forward | `run/benchmarking/policy_bench/check_simulator.py` |
| G4 | The vendored signal's head equals nano-sigma's own return_features.py head on the 70,183 January test decisions (Pearson@128 0.29676; the board's 0.2979 is a ridge readout, not the head); processed rows rebuilt from our book equal the files | `run/benchmarking/policy_bench/check_signal.py` |
| G5 | `ReplayMarket` + `SignRule` (one share, latency 0, 128 rows, `MarkToMid`, no fees) against the PnL bench formula on the head (reference A 0.627753, B -1.863064; the board's 0.612 / -1.879 are the ridge readout). The bench's A and B must equal the formula on its own signs; signs that differ from nano-sigma's are counted | `run/benchmarking/policy_bench/check_replay.py` |

| G6 | `SimMarket` under a fixed seed is reproducible; one agent order adds exactly that message (job 7846639: 8 of 8 identical; the agent message is the only difference) | `run/benchmarking/policy_bench/check_sim.py` |
| G7 | Rejection rate of first draws, reasons, fallbacks to masks (job 7846639: 20.9% of first draws, 0.32% fallbacks; Kang's LOB-Bench board: 11.3% invalid, a measure not yet matched) | same script |
| G8 | The signal on generated flow (job 7846639: head Pearson 0.320 at 64, 0.222 at 128, vs 0.343 and 0.297 on real data); next, the sign rule in sim vs replay | same script |

Unit tests: `tests/unit/test_policy_bench_*.py`.

## Data for the two models

The simulator was trained on the `v2_20261013` shards, the signal on the
`bookfix20261012` shards. On January 2026 the two are byte-identical except
for 8 META stock-days, which are truncated in bookfix (job 7836700). The
bench reads `v2_20261013`. G4 scores the signal on bookfix so that its
reference number is reproduced as published.

The processed rows come from two preprocessing runs. Regular-session files
(`*_34200000_57600000_*`) clip `price_rel` at +-999 ticks, the others do not
(jobs 7837738, 7837813, 7837914). `signal.price_rel_clip` reads the convention from the
file name.

## Precision

The simulator runs in bf16 autocast, as nano-sigma trained and evaluated it.
In float32 the KV cache equals the full forward (max logit difference 3e-4,
job 7835151). In bf16 the two use different attention kernels, so logits
differ by up to about 2.5 and 1% of argmaxes flip, while the CE agrees to
1e-4. A rollout is reproducible only on the same code path and kernels.
Batched decoding is another code path: in bf16 its logits differ from one-row
decoding by up to 0.38, in float32 by 3e-5 (job 7845818).

The signal is affected the same way. G5 (job 7846117) predicts one decision
at a time, nano-sigma four: on the 70,159 decisions outside the 8 days whose
shards differ, 107 signs differ from nano-sigma's, all at predictions of at
most 0.025 bps, which moves PnL A from 0.62785 to 0.62926. Returns and the
accounting are exact there: A equals the PnL bench formula on the bench's own
signs to every digit, B to the last bit.

## Throughput

One generated token at a 67,036-token context, bf16, one GH200 (job 8449248,
results/speed): 121 ms without a cache (a full forward over the context),
7.8 ms with the one-row KV cache, 0.83 ms per token with 48 episodes decoded
together (40 ms per step), 145x the no-cache time. 200 episodes of 1,024
messages (about 16 tokens each) are about 3.3M tokens: about 110 h without a
cache, 46 min batched. A full dev run takes about 80 min (8439762-8439767).

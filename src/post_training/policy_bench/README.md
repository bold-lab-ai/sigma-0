# Policy bench

Evaluate one trading policy against a fixed simulator and a fixed signal, and
report its result. Policy search is a separate client of this package.

- Simulator: nano-sigma `newtok_a_r1` (best LOB-Bench on the nano-sigma board).
- Signal: nano-sigma `returnft_lobert_pass1` (best return and PnL bench).
- Data: `dataset_q4_2025_jan2026_v3`. Dev is January 2026, test is February 2026.

The state is the L3 order book plus the message log, not tokens. Each model
encodes its own input from that state.

## Modules

| Module | What it does |
|---|---|
| `book.py` | L3 book. FIFO queue per price, an owner (background or agent) on every order. Updated by order id for real data and by queue slot for generated messages. `L10Writer` gives the 43-column L10 row the models read. |
| `daydata.py` | One stock-day of the v3 dataset and the replay that rebuilds its full book from the stored snapshots. `build_add_index` gives each order id's first add, stored once per day by `run/post_training/policy_bench/build_add_index.py`. |
| `codec.py` | The newtok `Message`: `encode` (identical to nano-sigma's `encode_window`) and a strict `decode`. |
| `simulator.py` | The newtok model with a KV cache. `BatchKVSimulator` holds several sequences of their own lengths: prefill one row, step every row by one token, cut back one row. `KVSimulator` is the one-row case. |
| `signal.py` | The return signal: processed rows from our book (`price_rel`, the reference columns), and the fine-tuned head's prediction at a decision. |
| `rules.py` | Legality of a background message in a book (what the sampler rejects), and how a legal message changes the book. |
| `types.py` | Plain data: `MarketOrder`, `LimitOrder`, `Cancel`, `Fill`, `Obs`, `Episode`. |
| `history.py` | `RowLog`, the processed rows and L10 rows both models read, and `start_state`, an episode's book and log rebuilt from a real day. |
| `market.py` | `ReplayMarket`: real rows, virtual agent orders, market orders only, fill model `MidFill` or `BookFill`, latency and inventory cap. |
| `rollout.py` | `rollout(policy, signal, market, episodes, every)` -> one `Trace` per episode. |
| `trace.py` | `Trace` (mids, end book, fills, decisions, signal) and `save` / `load` as one npz. |
| `costs.py`, `terminal.py`, `score.py` | `MakerTaker` fees; `MarkToMid`, `MarkToMicro`, `FlattenTaker`; `score(traces, terminal, cost)`, a pure function of the traces. |
| `episodes.py` | Episode sets. `return_bench_decisions` is nano-sigma's test decision set. |
| `policies/` | `NoTrade`, `SignRule`. |
| `nanosigma/` | Vendored nano-sigma encoding code, byte for byte. See `nanosigma/PROVENANCE.md`. |

## Rules a background message obeys

- New orders do not cross the opposite best.
- A cancel or delete names one background order by queue slot. Agent orders
  count in the numbering but can never be named. QSLOT 99 means slot 99 or
  later.
- Executions walk one side from the best price, take inner levels whole, and
  are filled front of queue first. That is the only way an agent order is
  filled by background flow.

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
| G1 | Real-day replay: queue slots equal `<stem>_queue.npy`, top 10 levels equal the L10 file, every snapshot equals the stored one | `run/post_training/policy_bench/check_book_codec.py` |
| G2 | `encode_stream` equals `encode_window`; decode inverts encode up to clipping; real messages pass `rules.check`; a slot-driven book stays equal to the id-driven one | same script |
| G3 | The vendored simulator reproduces `newtok_a_r1`'s test CE on its 939 test windows; our book gives byte-identical model inputs; the KV cache equals the full forward | `run/post_training/policy_bench/check_simulator.py` |
| G4 | The vendored signal's head equals nano-sigma's own return_features.py head on the 70,183 January test decisions (Pearson@128 0.29676; the board's 0.2979 is a ridge readout, not the head); processed rows rebuilt from our book equal the files | `run/post_training/policy_bench/check_signal.py` |
| G5 | `ReplayMarket` + `SignRule` (one share, latency 0, 128 rows, `MarkToMid`, no fees) reproduces the PnL bench with the head: `MidFill` A 0.627753, `BookFill` B -1.863064 bps per bet (the board's 0.612 / -1.879 are the ridge readout) | `run/post_training/policy_bench/check_replay.py` |

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

## Throughput

One decoding step at a 67,036-token context, bf16, one GH200 (job 7845818):
29 ms at batch 1, 40 ms at batch 48 (1,198 tokens/s). The one-row KV cache
without a mask takes 7.2 ms (G3). Batch 48 is 8.6x that; 200 episodes of 1,024
messages (about 16 tokens each) take about 46 min.

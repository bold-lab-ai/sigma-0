# Policy bench v1

The policy bench scores one trading policy against a fixed market. The
policy is the only input; the simulator, the signal, the data and the costs
are fixed. v1 is the bench of policies that use our trained signal model.
Code: `src/post_training/policy_bench/` (its README has the modules and the
policy interface), launchers `run/post_training/policy_bench/`, configs
`configs/policy_bench/v1.yaml` and `v1_test.yaml`. Results and job logs are
under `/projects/public/u6oz/policy_bench/results/` (RESULTS below); the
path that led here is in `/projects/public/u6oz/policy_bench/LOGBOOK.md`.

## 1. What is fixed

| part | v1 |
|---|---|
| background flow | nano-sigma `newtok_a_r1` (best LOB-Bench on the nano-sigma board), sampled with masks (`max_rejections: 0`) |
| signal | nano-sigma `returnft_lobert_pass1`, predicted returns at 16, 64, 128, 512, 1,024 messages |
| matching | full L3 book from the real snapshot, strict FIFO, agent orders fed to the simulator as messages |
| episodes | 200 per split, uniform over stock-days; 1,024 background messages each (about 2 minutes) |
| context | simulator: the last 2,944 real messages (2,944 + 1,024 + 128 = 4,096, its training window); signal: the last 3,968 rows |
| decisions | every 64 background messages, 16 per episode |
| latency | an order reaches the market 100 ms after its decision |
| costs | maker rebate $0.002, taker fee $0.003 per share; inventory cap +-1,000 shares |
| end of episode | the open position is closed against the full end book (taker fee charged) |
| score | dollar PnL per episode, averaged within each date, then over dates, with its t over dates (`daily_pnl_usd`, `daily_pnl_t`) |
| rollouts | one per episode; more power comes from more episodes, not repeats |
| splits | dev: January 2026, 20 days. test: 2026-02-02 to 02-13, 10 days, run only on frozen policies |
| policy inputs | `Obs` only: L10, position, cash, open orders, orders on their way, fills, the signal |

## 2. Evidence for each choice

Every number below carries its job; each result directory has a README.

### The pipeline reproduces its sources

- G1, G2 (7832710, February 8363763): our book replays real days with 0 slot,
  0 L10 and 0 snapshot mismatches (January 12,950,231 rows, 3,141 snapshots;
  February 29,875,870 rows, 7,257 snapshots); the codec equals nano-sigma's.
- G3, G4: the vendored simulator reproduces `newtok_a_r1`'s test CE, and the
  signal's head equals nano-sigma's own on 70,183 decisions (Pearson@128
  0.29676).
- G5 (7846117): replay plus the sign rule equals nano-sigma's PnL bench
  formula on the bench's own signs to every digit.
- G6 (7846639): a seed reproduces a rollout; one agent order is the only
  difference it makes.

### Sampling: masks

- Drift (7850611, 7851665): with rejection the book drifts from the first
  block (half spread 2.41 to 0.80 bps against the real 2.7); with masks it
  does not (2.7 to 2.9).
- LOB-Bench, 864 histories (results/lobbench): unconditional Wasserstein mask
  0.062, keep 0.095, reject 0.162. Uniform legal fields score 0.782
  (8347559-8347565), so the score is the model's, not the masks'.
- Fill check v2 (8363757-8363759): masks are closest to the market in mid
  volatility (7.8 to 8.5 bps per 512 messages, real 7.9 to 8.5) and clock
  (52 to 66 s per 512 messages, real 52 to 67).

### Takers: impact

- Real sweeps fed to the simulator (8351238; 8416224, 8416225, 750 sweeps):
  the mid move and the spread's recovery agree with the market at every size
  and horizon, within about two standard errors. A sweep after a pause moves
  the price more in the market (about 5 against 3 bps); the simulator
  reproduces that in every gap group, fed 1 ns after the message before it or
  at its real time (results/sweeps_gap).

### Makers: fills

- Orders that join the best queue, real and simulated, same 500 windows
  (8389228): executed within 64 messages 0.089 vs 0.085, deleted first 0.80
  vs 0.81. Fill rates are right.
- Their markout 64 messages after the first execution is -0.41 bps real and
  +0.29 simulated (128: -0.52 vs +0.47). See section 4.

### Latency

- The signal takes 9.5 ms per decision at batch 1 on a GH200 (8407649:
  median 9.47 ms, p95 10.08 ms; 3.45 ms CPU inputs, 6.03 ms GPU).
- Latency sensitivity on dev, SignRule(2, 1), bps of notional after the
  flatten (results/latency_v2, 8427435-8427443; 100 ms from dev_v1c):

  | latency | 0 | 10 ms | 100 ms | 1 s |
  |---|---|---|---|---|
  | sim | -3.05 | -3.49 | -3.74 | -3.96 |
  | replay | -3.28 | -3.80 | -4.01 | -4.20 |

  The cost grows smoothly, and sim and replay differ by 0.2 to 0.3 bps at
  every latency, so the simulated market reacts during the latency as the
  real one does. 100 ms is above the signal's own 9.5 ms and within reach of
  a real deployment; 1 s would cost the taker only 0.2 bps more, but takes
  most of the maker's marked edge (SkewQuoter +12.9 $ at 0, +3.2 $ at
  100 ms, +1.0 $ at 1 s).
- Orders in flight. Decisions are counted in messages and latency in time;
  in a burst, 64 messages can pass in less than 100 ms. Before `Obs.pending`
  existed, a policy could not see an order it had sent and re-sent it
  (SignRule overshot its target in 146 of 200 simulated episodes, up to 900
  shares). With `Obs.pending` (commit 3d3b110) the overshoot is 0.

## 3. Baselines

Ranking figure: `scores.flatten_taker.daily_pnl_usd`, $ per episode averaged
within each day, then over days, with its t over days. Commit 3d3b110,
dev 8427420-8427424 (results/dev_v1c), test 8427425-8427429
(results/test_v1c); NoTrade 8411448 and 8416439. SignRule(index 2, size)
trades toward sign(signal at 128 messages) times size with market orders;
SkewQuoter quotes 100 shares at both best prices and drops the side the
signal argues against by more than 0.5 bps.

| policy | market | dev $ (t) | dev bps | test $ (t) | test bps |
|---|---|---|---|---|---|
| NoTrade | sim | 0 | | 0 | |
| SignRule(2, 1) | sim | -1.30 (-9.0) | -3.74 | -1.42 (-6.0) | -4.50 |
| SignRule(2, 1) | replay | -1.45 (-7.4) | -4.01 | -1.81 (-7.0) | -5.06 |
| SignRule(2, 100) | sim | -1,092 (-3.8) | -14.12 | -1,306 (-4.5) | -17.47 |
| SignRule(2, 100) | replay | -332 (-6.2) | -7.95 | -506 (-4.3) | -10.16 |
| SkewQuoter | sim | -44.6 (-5.0) | -3.44 | -46.0 (-2.1) | -4.06 |

- No baseline beats NoTrade. The signal's edge at 128 messages is smaller
  than the half spread plus the taker fee, so crossing the spread on it
  loses about 4 bps.
- SignRule-1 agrees between sim and replay; SignRule-100 loses 6 to 7 bps
  more in sim, which is the impact replay cannot have.
- SkewQuoter is positive marked to mid (+3.2 $ dev, +14.3 $ test) but ends
  with 160 to 180 shares on average, and the flatten costs more than the
  spread it earned. Its maker credit is too high anyway (section 4, limit 1).
- Why the score closes the position instead of marking it to mid: mark to
  mid lets a policy leave for free (no spread, fee or impact), so a policy
  ranked by it would learn to hold inventory. Both are in every result
  file; their gap is the cost of exiting. For SignRule it is about 10%.
- Unpriced: 0 in every run.

## 4. Known limits

1. Makers are credited too much: about 0.7 to 1.5 bps per fill at 64 to 128
   messages. Simulated executions take only part of a thin best level more
   often than real ones (2 to 9 shares: 53% vs 11%), so levels clear less and
   the mid moves less after a maker fill (results/fills/v3). Fill rates need
   no correction.
2. Maker and taker results are therefore not on the same footing. Compare
   policies of one kind with each other.
3. The signal is more predictive on generated flow than on real flow
   (Pearson 0.448 / 0.443 / 0.385 at 16 / 64 / 128 against 0.386 / 0.343 /
   0.297, job 7856963).
4. The model sees only the top 10 levels; the engine keeps every level.
5. Latency is a fixed 100 ms. A policy's own compute time is not charged.
   An order leaves `Obs.pending` the moment it reaches the market; a real
   system learns that only from an acknowledgement, later.
6. Strict FIFO: 96.2% of real executions fill front first (7832642).
7. Every realism check ran on January dev days.

## 5. Not done in v1

- A penalty for a position larger than the whole end book (counted as
  unpriced; none so far).
- A signal with a growing cache (only matters for deployment latency).
- Policies with their own models.

# Fidelity defect taxonomy and register

**This document fixes nothing. Its only purpose is to turn "the simulator is
not faithful", a sentence with no action in it, into a set of mutually
exclusive, exhaustively derived, individually falsifiable defect definitions.**

What prompted it: the gen arm's fidelity across eight windows was 0.2389,
0.3367, 0.4711, 0.6056, 0.6256, 0.6722, 0.6900, 0.7689. Without a
classification those eight numbers support exactly one conclusion, "the gen arm
is bad", which can neither guide a repair nor be falsified. With one, the same
eight numbers produce checkable propositions, one of which came with its own
boundary and one of which was later overturned by measurement (section 5).

---

## 1. The faithfulness identity

Write down what has to hold first. Every class below is derived from this
equation rather than enumerated from experience.

**Symbols**

| notation | meaning |
|---|---|
| `b̂ ∈ B̂` | the engine's full internal state: one row per live order, `(price, qty, oid, tid, time_s, time_ns)` |
| `b ∈ B_L` | the L-level L2 projection, `b = Π_L(b̂)`, i.e. one row of the orderbook CSV |
| `m ∈ M` | one message, `[type, side, qty, price, oid, tid, time_s, time_ns]` |
| `E_θ : B̂ × M → B̂` | the engine's transition, with `θ = (nOrders, book_depth, init_id, tick_size)` |
| `ι_θ : B_L → B̂` | rebuilding internal state from an L2 snapshot, i.e. `init_msgs_from_l2` |

**Record.** One generation run produces `R = (m_1 … m_T ; b_0 … b_T)`, where
`b_k` is the L2 snapshot the generator wrote after applying message k.

**Replayable.** A record `R` is replayable under engine `E_θ` exactly when

```
∀k ∈ [1, T]:   b_k  =  Π_L ( E_θ ( ι_θ(b_0), m_1, …, m_k ) )
                ↑                   ↑          ↑        ↑
          what the record       the rebuilt   the      the messages
             claims             initial state engine   in the record
```

The left side is what the record **claims**. The right side is what the record
**recomputes into**. Everything the phrase "the simulator is faithful" can mean
is that equals sign.

---

## 2. The classes, derived rather than listed

**The identity has five symbols. If the equals sign fails, one of those five is
what failed.** That is where exhaustiveness comes from, and it is
constructive: it does not rest on whether I happened to think of everything.

```
        b_k  =  Π_L ( E_θ ( ι_θ ( b_0 ) , m_1..m_k ) )
         │       │      │      │     │        │
         │       │      │      │     │        └── R  message recorded ≠ applied
         │       │      │      │     └─────────── R  initial book ≠ true state
         │       │      │      └───────────────── I  rebuild not isomorphic
         │       │      └──────────────────────── T  engine config differs
         │       └─────────────────────────────── O  projection cannot separate
         └─────────────────────────────────────── R  book recorded ≠ reached
```

| class | name | requirement violated | nature |
|---|---|---|---|
| **R** | Record | what is written equals what was applied | self-contradictory record, **fixable** |
| **I** | Init | `ι_θ(b_0)` isomorphic to the generating initial state, **order identity included** | **fixable** |
| **T** | Transition | both sides run the same `θ` | engine config drift, **fixable** |
| **O** | Observation | `Π_L` can separate the states being compared | **an information bound, only declarable** |

Two **meta-classes** sit outside the identity. They do not describe how the
equals sign breaks; they describe why you should believe you are testing it:

| class | name | requirement violated | why it must be separate |
|---|---|---|---|
| **X** | Execution | `R` really was produced by the code it claims | when artifact and code come apart, the equation is meaningless either way |
| **V** | Validation | the diagnostic can detect an R/I/T/O violation | when the diagnostic fails, a violation is invisible |

**Why exactly six.** R/I/T/O is a complete enumeration of the identity's
symbols: one more class would overlap an existing one, one fewer would leave a
symbol unowned. X and V are the meta layer, X covering "is the input what you
think it is" and V covering "would you see it if it were not". Together they
span the chain from committing code to drawing a conclusion.

---

## 3. The standardised record format

Every entry must fill every field below. **An entry with a missing field is not
a defined problem.**

| field | meaning | why it is mandatory |
|---|---|---|
| `id` | `D-<class><n>`, stable, never reused | citable across documents, PRs and tests |
| `statement` | the violated invariant, one sentence | forces "what does not hold", not "this looks odd" |
| `sites` | `file:line` | a defect has to land on code, otherwise it is a guess |
| `trigger` | the precise firing condition | decides whether it affects a given run |
| **`observable`** | what a detector should measure if it fired | **makes it falsifiable; unable to fill it means it is not yet defined** |
| `silent` | does it raise, or produce plausible wrong output | silent defects need an active probe, loud ones do not |
| `evidence` | a measurement, or an honest "static" | separates confirmed from suspected |
| `status` | `OPEN` / `FIXED` / `BOUND` / `REFUTED` | `BOUND` is a bound that cannot be fixed, `REFUTED` was suspected and ruled out |

`observable` is the load-bearing field. It turns a bug list into a **detector
specification**: every entry has to say which number would move if it happened.
An entry that leaves it blank has not defined the problem and is rejected at
the schema level by `tests/unit/test_defect_register.py`, which also rejects
duplicate ids, `BOUND` claimed outside class O, rotted `file:line` citations,
and drift between this document and the JSON.

---

## 4. The register

**The tables in this section are generated from `docs/defect_register.json` by
`tools/render_register.py`.** Do not hand-edit them. Change the JSON and
regenerate; a test rebuilds this section and fails if it differs, so the two
cannot drift.

<!-- REGISTER:BEGIN -->

### Class X, Execution: the artifact and the code came apart

9 of 10 closed.

| ID | statement | sites | trigger | observable | silent | evidence | status |
|---|---|---|---|---|---|---|---|
| **D-X1** | srun resolves its command through PATH without testing the execute bit, unlike the shell, so a non-executable file of the same name shadows the real binary; the repository itself contained two such call sites, `srun ... env VAR=... bash script` and `srun ... bash -c` | `$HOME/.local/bin/env`; `run/environment/run_episodes_multinode.sh:117` | the srun command line begins with `env` and ~/.local/bin precedes /usr/bin on PATH | sacct shows JobName=env, ExitCode=13:0, Elapsed~2s | no | measured: step 5836919.420, execve(): /projects/public/u6gb/.local/bin/env: Permission denied. FIXED: run/environment/preflight.sh detects a non-executable or directory first-match in 0.1s before the allocation is spent; run/environment/attach_run.sh never passes a bare command name as srun's first word. Pinned by tests/unit/test_launcher_provenance.py. A second, in-repo site was found only after the regression test was written, which is the point of writing it | **FIXED** |
| **D-X2** | WORKDIR defaults to a stale worktree, making new-data-from-old-code the default path | `run/environment/build_episodes.sbatch:34`; `run/environment/run_episodes_multinode.sh:28` | submitting without an explicit WORKDIR override | output directory timestamp is later than the commit time of the code that produced it | **yes** | already fired: tasks/agentic_mm_20260731T164839Z generated 07-31 16:48, after the fixes were committed 09:39-11:00, from a worktree cut before them. FIXED: WORKDIR is now required with :? in both launchers, and the manifest records workdir, git_sha and git_dirty_src_run so an artifact can name its own commit | **FIXED** |
| **D-X3** | the sbatch output path and log directory are hardcoded to the old worktree and do not follow a WORKDIR override | `run/environment/build_episodes.sbatch:30`; `run/environment/preflight_claude_p.sbatch:23`; `run/environment/preflight_claude_p.sbatch:27` | any submission that overrides WORKDIR | run log and produced data live in different worktrees | **yes** | static: 4 hardcoded occurrences. FIXED: every #SBATCH --output and every LOG_DIR now points at the neutral job-id-keyed root /lus/lfs1aip2/projects/public/u6gb/slurm_logs, so log, per-rank logs and manifest join on the job id | **FIXED** |
| **D-X4** | a tmux session started from inside an srun step is killed together with that step | `launch procedure` | using tmux to background work inside an allocation | process disappears within ~13s, log stops at the startup stage | **yes** | already fired in a previous round. FIXED: run/environment/attach_run.sh makes the payload script the srun step itself, removing the intermediate process that was being reaped | **FIXED** |
| **D-X5** | the monitor greps only for DONE and error strings; a killed process writes neither | `monitor procedure` | the monitored process is signal-killed or fails execve | log file mtime stalls while the progress count stays constant | **yes** | already fired: one hour reporting 1/9 without alarming. FIXED: run/environment/watch_log.sh alarms on mtime and size stalling, verified on all three outcomes (stall rc=1, error rc=2, done rc=0) | **FIXED** |
| **D-X6** | no preflight: an allocation is consumed before the run environment is verified usable | `missing` | every submission | the first step exits non-zero within 5 seconds | no | D-X1 would have been caught by a one-second probe. FIXED: build_episodes.sbatch sources preflight.sh and calls preflight_or_die before module load and before any Python, pinned by an ordering assertion | **FIXED** |
| **D-X7** | generation is reproducible only on the node that produced it, and the record did not name the node, so the artifact was not determined by the configuration it records | `run/base_model/runtime/inference.py`; `src/base_model/inference/generate.py` | regenerating any window, at least across different compute nodes | byte-compare the generated message CSVs of two runs with identical seed, checkpoint, window AND node; and whether the manifest names the node at all | **yes** | measured 2026-08-01 by comparing tasks/agentic_mm_20260801T141849Z against tasks/agentic_mm_20260801T144426Z. The conditioning streams are BYTE-IDENTICAL, so window selection is stable and the difference is purely on the generation side. All 8 generated streams differ, first at rows 7 to 118 and then in 87-99% of the remaining rows. The two runs used the same seed, checkpoint and code and differed only in the node they ran on, nid010147 against nid010148. The late first difference argues against an unseeded RNG, which would differ at row 1, and for floating-point drift flipping a sampling decision and then being amplified autoregressively. CAUSE ESTABLISHED by that experiment: two runs pinned to the SAME node (nid010148, GPU gate verified idle before each) produce output that is BYTE-IDENTICAL in 8 of 8 windows over 7200 messages. So generation is deterministic within a node, and the cross-node difference is floating-point behaviour flipping a sampling decision which autoregression then amplifies. The RNG is not implicated. FIXED by completing the record: the manifest now writes the node, because a configuration that determines the output has to be recorded like any other. The frozen-market design is unaffected, since a market is frozen by being written to disk and every policy replays that same record. n=1 pair on one node; strong (7200 messages byte-identical) but measured rather than proven in general. This does NOT affect any fidelity measurement, because fidelity compares a record against a replay of that same record; it does affect any baseline derived from regenerated data, which is how it was found: the lookup-miss sentinel count moved 355 -> 588 between two runs that should have been identical | **FIXED** |
| **D-X8** | grad_norms_or_logits was assigned on the scan and plain train paths but never on the gradient-accumulation path, so GRAD_ACCUM_STEPS>1 together with LOG_GRAD_NORMS=1 raised UnboundLocalError at the first optimizer step | `src/base_model/training/train_helpers.py:1083` | GRAD_ACCUM_STEPS>1 together with LOG_GRAD_NORMS=1 | UnboundLocalError naming grad_norms_or_logits at the first K-micro-batch boundary; the run dies before its first optimizer step completes | no | measured: the diagnostic run of 2026-08-13 00:58 died exactly this way -- the run that needed gradient norms most; fixed in 35a88ad by computing norms from accum_grads at the apply boundary, with an explicit None initialisation so an unassigned path costs one metric rather than the run | **FIXED** |
| **D-X9** | the checkpoint reshard broadcast (device_get -> broadcast_one_to_all -> device_put) builds a pure inter-node NCCL clique that the training path never constructs -- replica_groups [4,4]<=[4,4]T(1,0), one card per node, 100% Slingshot -- while its own comment claims the round trip exists to avoid exactly that | `src/base_model/training/train.py:321` | any multi-host checkpoint save on the broadcast path, at every save interval | the hang lands on the save-period boundary: main thread in futex_wait, exactly one CUDA spin thread per GPU, zero orbax file descriptors (stalled before opening files); compile-time, check_replica_groups.py shows the broadcast psum shares zero replica_groups with training | **yes** | measured: two hangs on 2026-08-13 at 10:36:26 and 12:36:27, both within +-30s of the 900s save boundary (about 0.4% chance if uniform); the compile-time replica-group analysis is reproducible on a login node. The fix (CKPT_RESHARD_MODE with a collectively agreed downgrade) ships in its own PR carrying a real multi-host save-and-resume round trip; OPEN until that lands | OPEN |
| **D-X10** | create_jit_train_step builds the gradient-accumulation pair only on the shard_map path, so grad_accum_steps>1 on a flat single-node mesh fell through to the plain per-micro-batch path and K was silently ignored: every micro-batch applied immediately, the effective batch was micro_bsz*n_devices instead of the declared one, and the LR schedule advanced per micro-batch | `src/base_model/training/train_helpers.py:1821` | GRAD_ACCUM_STEPS>1 on a topology where use_shard_map is false (single node, no 'tp' axis) | before the fix, nothing at launch: the banner printed the declared K and effective batch while checkpoint labels advanced at micro-batch rate (state.step spacing equal to micro-batches per save interval, 1800 per 900 s at 2.0 micro-steps/s, where the accumulation path would show 90); after the fix, ValueError from create_jit_train_step naming grad_accum_steps | **yes** | measured: the 2026-08-27 single-node baseline segment (banner 'K=20 (effective BSZ = 80)') saved checkpoints at state.step 346, 2153, 3952, 5749, 7547 -- 1800 apart per 900 s auto-save at 2.0 micro-steps/s, i.e. one state.step per micro-batch, effective batch 4; the 1.8 h segment was discarded (checkpoints renamed *_deprecated_bsz4) and the guard added with a unit detector | **FIXED** |

### Class I, Init: the rebuilt initial state is not isomorphic

2 of 3 closed.

| ID | statement | sites | trigger | observable | silent | evidence | status |
|---|---|---|---|---|---|---|---|
| **D-I1** | the generator initialises from a 500-level wide book while the replay initialises from a 10-level conditioning book. init_msgs_from_l2 maps L2 row j to the same physical (side, level) at any width, so ids within [-21,-2] name the same order on both sides; the defect bites only for ids beyond the replay's depth, oid < -21 | `src/base_model/inference/generate.py:1709-1715`; `src/post_training/heuristic_learning/fidelity.py:211-213`; `src/post_training/heuristic_learning/fidelity.py:162` | a cancel or delete referencing an init order below the replay's level count | count of recorded messages with oid < -(2 + 2*n_levels), and their step positions relative to the first divergence step | **yes** | measured, and it narrowed the claim. Out-of-depth ids occur in 1 of 8 windows (#258: 185 of 192 init-id references, 96%, deepest L2 index 84 = level 43); the other 7 windows carry only in-depth ids. #258 has by far the largest divergence magnitude (mean 28.55 ticks vs 0.30-2.80 elsewhere). But it does NOT initiate divergence: 4 of 8 windows have zero bad ids before their first divergence step, and #258's first out-of-depth id is at step 57 while divergence starts at step 200. D-I1 amplifies, it does not cause. Onset lies in class R or O. On regenerated data 2026-08-01 the out-of-depth count collapses from 185 to 19 across all 8 windows and #258 goes from 185 to 0, so most out-of-depth references were an artefact of the fuzzy lookup rather than of initialisation width. #258's gen fidelity rose 0.2389 -> 0.9044 accordingly. REFUTED as a distinct defect: measured 2026-08-01 on tasks/agentic_mm_20260801T132010Z, regenerated with faithful initialisation. with faithful initialisation the out-of-depth references resolve correctly because the orders they name now exist. It was a symptom of D-I3, not a defect of its own | REFUTED |
| **D-I2** | init order ids encode the L2 vector index (id = init_id - index), so order identity is stable only across snapshots of identical width and level ordering, and the record declares no width | `src/matching_engine/jaxob/JaxOrderBookArrays.py:1069` | any replay at a width other than the generating width | the record carries no wide_levels or book_depth field | **yes** | static: manifest.json records only checkpoints and seeds. FIXED twice over: the bookinit CSV's own column count declares the width (2 + 4*wide_levels), and the manifest declares wide_levels explicitly. The width is no longer a hidden parameter a reader has to know | **FIXED** |
| **D-I3** | iota_L reconstructs only the top L levels of the book, so everything below level L is absent from the replay's internal state; once the visible levels deplete, the reference book surfaces a level the replay never had and every later comparison is against a different book | `src/matching_engine/jaxob/JaxOrderBookArrays.py:1045-1073`; `src/post_training/heuristic_learning/fidelity.py:211-213` | any replay long enough for the top L levels to deplete, which is every 900-message window measured | the first-divergence price lies outside the initial visible range; and the first-divergence step rises monotonically as the comparison window is narrowed, because a narrower window stays inside what the replay does know | **yes** | measured 2026-08-01 by autopsy.py. 11 of 16 first-divergence prices lie outside the initial visible range, and on the real arm it is 8 of 8. Narrowing the comparison from 9 levels to 3 moves the real arm's median onset 106 -> 346 -> 588 -> 900, and comparing only the touch leaves 5 of 8 real windows with zero divergence across all 900 messages. Fixable rather than a bound: the L500 wide book exists at recon_2026-05 and the generator already initialises from it. CODE FIXED, verification pending: generation now writes data_cond/{stock}_{date}_bookinit_real_id_{i}.csv, one row of time_s, time_ns and 4*wide_levels book columns, and replay_stream gained a faithful mode mirroring get_sim exactly (reset on that book with ns zeroed, replay the conditioning messages, book_depth and nOrders from the input width). Backward compatible: runs without the file take the degraded path and reproduce their previous numbers bit for bit. The falsifiable prediction is that the real arm moves from 5/8 towards 8/8; a regeneration is running to test it. measured 2026-08-01 on tasks/agentic_mm_20260801T132010Z, regenerated with faithful initialisation. Both arms go to 8/8 exact on BOTH criteria, mid and full book, mean 1.0000, zero divergence over 900 messages in all 8 windows. Prediction was real arm 5/8 -> close to 8/8; the actual result also took the gen arm from 0/8 to 8/8. Validated by negative control: perturbing all quantities gives first divergence at step 0, deleting message 50 gives step 50, perturbing from message 300 gives step 306, and the same data through the degraded path still reports 0.3300. A criterion that reports 1.0000 everywhere has to be shown capable of reporting less | **FIXED** |

### Class R, Record: what was written is not what was applied

3 of 6 closed.

| ID | statement | sites | trigger | observable | silent | evidence | status |
|---|---|---|---|---|---|---|---|
| **D-R1** | the NaN branch feeds construct_dummy_sim_msg() to the generator's own sim but still returns the original msg_decoded for writing, so the record contains messages the generator never applied | `src/base_model/inference/generate.py:510-515` | the generated new-part contains NaN | messages present in the record that the generator did not apply; replay executes them | no | static; firing count unmeasured because the record carries no provenance flag, see D-R5. measured 2026-08-01 on tasks/agentic_mm_20260801T132010Z, regenerated with faithful initialisation. does not fire in these 8 windows: a message recorded but not applied would divergence the replay, and there is zero divergence across 7200 messages. MEASURED for the first time, 0 of 7200: the NaN branch does not fire on this data at all, which confirms by direct measurement what perfect fidelity had only implied. Still OPEN because the code still records the original message when it does fire, so the statement remains true. But it is no longer SILENT: provenance bit 2 records the branch, so an occurrence is now detectable from the artifact instead of invisible. That is a change in the defect's character, not a fix, and the field says so | OPEN |
| **D-R2** | NEGATIVE_RETURN_ID = -99, the total-lookup-miss sentinel, is written into the CSV as if it were a valid order id | `src/base_model/inference/generate.py:481-493`; `src/base_model/inference/generate.py:507`; `src/matching_engine/jaxob/jaxob_constants.py:10` | both the L1 and L2 lookups miss | count of recorded messages with oid == -99 | **yes** | measured: 15 in window #258, 6 in #1386, 0 in the other six on the pre-fix generator. On the regenerated data the count is 527 across all 8 windows (7.3% of messages), because exact-timestamp lookup turns former false resolutions into honest misses. The misses are correct; writing them out as if they were order ids is not. See D-R6 for what the value then does. measured 2026-08-01 on tasks/agentic_mm_20260801T132010Z, regenerated with faithful initialisation. NOT a fidelity defect: #258 carries 544 sentinels and still replays 8/8 exact on the full book, because the sentinel is applied identically on both sides. What remains is a semantic defect, the generator removing quantity from an unrelated bid level, which this register's subject (record -> replay fidelity) does not measure. Fidelity 1.0 means the replay reproduces what the generator did, including its mistakes. FIXED as far as its consequence goes: the sentinel is still written into the record, but it is no longer a valid order id at any depth, so it resolves to the price fallback rather than to an unrelated order. The record remains odd for a human reader, which is D-R5's territory | **FIXED** |
| **D-R3** | the written price is the model's p_abs while the written oid comes from the lookup, and the two are never cross-checked to name the same order | `src/base_model/inference/generate.py:506-507`; `src/base_model/inference/generate.py:465` | the resolved order sits at a price other than p_abs | count of recorded messages whose price differs from the price of their oid in the book | **yes** | measured 2026-08-01 and now the dominant gen-arm cause. All 5 gen divergences whose price lies INSIDE the initial visible range share one signature: the recorded message price is not the price at which the generator actually removed quantity. #133 8989700 vs 8989600, #258 9317800 vs 9318900, #1478 9499600 vs 9499800, #3216 9306700 vs 9305900, #3751 9286200 vs 9286100. That is 5 of 5, discrepancies of 1 to 11 ticks. It is depth-invariant: #3751 diverges at step 4 for every comparison width from 9 levels down to 1, so no amount of initialisation depth removes it. The two autopsy signatures APPLIED_NOT_RECORDED and DIFFERENT_PRICE are one defect, differing only in whether the replay's price fallback happened to remove something or nothing. REFUTED as a fidelity defect, and this overturns the section 5.5 attribution. get_sim_msg writes the SAME (price, oid) pair into the simulator message and into the CSV, and cancel_order resolves by oid alone, so generator and replay apply an identical message to an identical book and reach an identical result. The apparent 'recorded price is not where quantity was removed' was the engine doing what it does with an oid that lives at another price, and it is reproducible. It looked like a record defect only because the replay's book was different. measured 2026-08-01 on tasks/agentic_mm_20260801T132010Z, regenerated with faithful initialisation. gen arm 8/8 exact on the full book. The record remains internally odd for a human reader, which is a separate concern from replayability | REFUTED |
| **D-R4** | rel_price_ref and quantity_ref are decoded and never used, so the model's own reference-price prediction is discarded instead of validating or driving the lookup | `src/base_model/inference/generate.py:447-448` | every cancel or delete | count of cancel/delete messages where price != price_ref, recorded as provenance bit 3. Retained as a datum; it is NOT a defect count, see the evidence | **yes** | static: both appear only in their assignment lines and in one commented-out block. FIXED as stated: the variables are no longer discarded, they feed a recorded consistency check (provenance bit 3). Deliberately a check and not an input, because making the lookup follow the ref price would change the generated market and needs its own evidence. FIRST MEASUREMENT: it fires on 1968 of 7200 messages, which is 60.1% of the 3275 cancel/delete messages. INTERPRETATION IS OPEN and deliberately not claimed: this repository does not contain the preprocessing that builds price_ref, so it could not be confirmed that price and price_ref are relativised against the same mid. A high rate therefore means either the model emits inconsistent pairs or the two fields are not comparable in this formulation. The control that settles it is to compute the same bit on the real conditioning messages, which are consistent by construction: a high rate there proves incomparability, a near-zero rate proves the model is inconsistent. That control needs its own change and has not been run. What is established is that a discarded prediction is now a recorded number. REFUTED by the control this entry itself called for. The premise was that the discarded ref fields could validate the message they accompany. They cannot, at least not by equality: running the identical check on REAL market messages, which are consistent by construction because a LOBSTER cancel's own price IS the price of the order it cancels, fires on 55.83% of them (1204/1904 = 63.24% on the conditioning stream, 1784/3448 = 51.74% on the continuation stream) against 63.35% on generated. A check that flags 56% of known-consistent data is measuring something other than consistency: price and price_ref do not share a reference frame. The 60.1% figure previously recorded here means nothing about the model. The bit is kept, because it is a real well-defined datum and deleting it would delete the evidence that the control ran, but renamed to price_differs_from_ref so its name stops asserting a refuted conclusion. Establishing what relation price_ref actually bears to price needs the preprocessing that builds it, which is not in this repository | REFUTED |
| **D-R5** | the record carries no per-message provenance flag (L1 hit, L2 fallback, total miss, NaN-skipped), so D-R1 through D-R3 cannot be detected from the artifact alone | `record schema`; `src/base_model/inference/generate.py:510-515` | every message | the missing field itself | **yes** | static: the CSV has six columns time,type,oid,size,price,dir. Downgraded from the critical path 2026-08-01: the onset was determined WITHOUT per-message provenance, by comparing the two book paths as price->quantity maps at the divergence step. Provenance flags would still separate D-R1 from D-R3 by construction rather than by inference, but they are no longer blocking. FIXED: get_sim_msg returns a provenance bit field, threaded out through both scan paths and written to data_gen/{stock}_{date}_provenance_real_id_{i}_gen_id_0.csv. Bits 0-1 record the resolution, bit 2 the NaN branch, bit 3 a ref-price disagreement. A sidecar rather than a seventh column so existing readers keep working. Measured over 7200 generated messages: 3925 not cancels, 2854 exact-timestamp hits, 66 price fallbacks, 355 total misses, i.e. 10.8% of the 3275 cancel/delete messages resolve to nothing | **FIXED** |
| **D-R6** | the lookup-miss sentinel NEGATIVE_RETURN_ID = -99 is not disjoint from the init order id space, which spans [-(2L+1), -2]; once book_depth >= 49 the sentinel names a real init order, and the generator runs at L=500 | `src/matching_engine/jaxob/jaxob_constants.py:10`; `src/matching_engine/jaxob/JaxOrderBookArrays.py:1069`; `src/base_model/inference/generate.py:481-493` | a total lookup miss during generation at any book_depth >= 49 | whether init_id_span(book_depth) contains NEGATIVE_RETURN_ID; and, in the generator's own book, quantity removed from L2 row 97 (an odd row, hence a bid level) on a message whose id is the miss sentinel | **yes** | arithmetic, pinned by tests/unit/test_stream_diagnostics.py. CONFIRMED ACTIVE on regenerated data 2026-08-01: cancel_order (JaxOrderBookArrays.py:130) matches on order id ALONE, with no price constraint, so at L=500 a message carrying oid=-99 directly hits the init order at L2 row 97. Row 97 is odd, hence bid side, so only bid-side sentinels collide. Measured: 527 sentinels in 7200 messages (7.3%), of which 207 are bid-side direct hits. The exact-timestamp fix ee3807c raised gen fidelity from 0.5511 to 0.7856 while raising this exposure from 6 to 207, a 34x increase, because it converts silently-wrong resolutions into explicit misses and every miss writes the sentinel. measured 2026-08-01 on tasks/agentic_mm_20260801T132010Z, regenerated with faithful initialisation. NOT a fidelity defect: #258 carries 544 sentinels and still replays 8/8 exact on the full book, because the sentinel is applied identically on both sides. What remains is a semantic defect, the generator removing quantity from an unrelated bid level, which this register's subject (record -> replay fidelity) does not measure. Fidelity 1.0 means the replay reproduces what the generator did, including its mistakes. FIXED: NEGATIVE_RETURN_ID moves from -99 to -999,999,999, far outside any plausible init id range, so a failed lookup now falls through to the engine's price-based fallback, which is the intended behaviour, instead of cancelling a real order by id. JAXLOB_Configuration.__post_init__ refuses to construct when book_depth would make the init range reach the sentinel, so a future depth cannot recreate the class quietly. The diagnostic recognises the legacy -99 as well, because a constant that reaches persisted records is part of the file format and older records still carry it | **FIXED** |

### Class T, Transition: the two sides run different engine configs

5 of 5 closed.

| ID | statement | sites | trigger | observable | silent | evidence | status |
|---|---|---|---|---|---|---|---|
| **D-T1** | the replay constructs JAXLOB_Configuration with defaults instead of reading theta from the generation manifest | `src/post_training/heuristic_learning/fidelity.py:162` | any generator-side configuration differing from the defaults | field-by-field comparison of the two thetas | **yes** | measured: generator runs book_depth=500 nOrders=1050; defaults are book_depth=10 (jaxob_config.py:16) and nOrders=NORDER_CAP (jaxob_config.py:21). FIXED on the replay side: load_theta() reads what the run declares and window_report merges it under any caller override. book_depth and nOrders are excluded on purpose because faithful initialisation derives them from the width of the initialisation book, which is better than a manifest field: the data describes itself and cannot contradict itself. The generation half is D-T3 | **FIXED** |
| **D-T2** | which ids count as init orders is gated by book_depth, so the price-fallback path behaves differently at different depths given the same book | `src/matching_engine/jaxob/JaxOrderBookArrays.py:1088` | book_depth differs between the two sides | the same cancel takes different branches under two book_depth values | **yes** | static. FIXED as a consequence of D-T1 and D-T3 together: book_depth is now declared by the record and derived from the initialisation width by the replay, so the two sides gate init-order recognition on the same value instead of on unrelated defaults | **FIXED** |
| **D-T3** | the generation manifest records checkpoints, seeds and window counts but not the parameters that break replay: wide_levels, book_depth, nOrders, tick_size | `run/environment/build_episodes.sbatch:161-184` | every generation run | those keys are absent from manifest.json | **yes** | static. FIXED: the manifest now carries an engine block with wide_levels, book_depth, nOrders, tick_size and init_id. book_depth and nOrders are written as the same expressions sample_new derives them from, so the record cannot disagree with the run. wide_levels became one variable feeding the wide-book mount, the --wide_levels flag and the record, replacing three separate literal 500s. Verified end to end: the block round-trips through load_theta | **FIXED** |
| **D-T4** | the training forward honoured use_positional_encoding while prefill and __call_inference__ did not, so a NoPE-trained hybrid generated with a sinusoid injected into every query and key; training loss and held-out CE stay clean because they use the training path, so only generated sequences are wrong | `src/base_model/models/transformer.py:225-229`; `tests/unit/test_transformer_decode_parity.py:1` | a model trained with use_positional_encoding=False generates through the KV-cache decode path | run the training forward and prefill+step decode over identical inputs and compare outputs (test_transformer_decode_parity.py); the mismatch cost about 11 points of reference recall at 32,000 steps before the fix | **yes** | measured: fixed in facf656d together with the parity test; found during the 2026-08-12 500-to-2k work, where it read at first like an architectural result | **FIXED** |
| **D-T5** | the hybrid's attention positions received a KV cache sized by the 26,000-token default rather than by the requested conditioning-plus-generation window, so any request past 500+500 messages attended over a silently truncated history | `src/base_model/models/transformer.py:129`; `src/base_model/models/transformer.py:239-240` | conditioning plus generation exceeding 26,000 tokens, i.e. any window past 500+500 messages | cache arrays allocated with a length axis shorter than seq_len_cond + n_gen_msgs*msg_len; after the fix the length is derived from the request (attention_max_cache_len threads through the registry) | **yes** | measured: found and fixed in 32e09d34 during the 2026-08-12 500-to-2k extrapolation; it was a real defect on the generation path but not the cause of that run's recall collapse, which the demoted profiling write-up records | **FIXED** |

### Class O, Observation: information bounds, declared not fixed

2 of 2 closed.

| ID | statement | sites | trigger | observable | silent | evidence | status |
|---|---|---|---|---|---|---|---|
| **D-O1** | init_msgs_from_l2 collapses each price level into a single synthetic order, so within-level queue order is not recoverable from an L2 snapshot and a type=4 that partially consumes a level cannot be replayed exactly | `src/matching_engine/jaxob/JaxOrderBookArrays.py:1045-1073` | replay of any stream containing aggressive executions | first divergence lands on a type=4 and is independent of cancel semantics | **yes** | previously credited with the real arm's residual on the strength of an indirect argument: cancel_mode x book_depth were bit-identical, so divergence never reaches the cancel fallback, and the trigger is always a type 4. That reasoning is intact but the conclusion was wrong. Direct measurement 2026-08-01 attributes the real arm's residual to D-I3 instead: 8 of 8 first-divergence prices lie outside the initial visible range, and the onset moves to never when the comparison narrows to the touch. D-O1 remains a real bound with ZERO measured instances in this data. It is kept because it is true, not because it explains anything observed | **BOUND** |
| **D-O2** | the comparison uses mid only, so depth errors that do not move the mid are invisible | `src/post_training/heuristic_learning/fidelity.py:176` | every comparison | difference between the mid criterion and a full-depth criterion on the same pair of paths | **yes** | FIXED: compare_paths now reports exact_full_frac and first_full_divergence_step beside the mid criterion, and the report prints both. The gap is large and was invisible: on the 2026-08-01 episodes the real arm is mid 5/8 exact mean 0.9614 but full-book 0/8 exact mean 0.4058, and the gen arm is mid mean 0.7856 against full-book mean 0.0326. Even #258, exact on mid, is 0.9778 on the full book. Pinned by tests/unit/test_faithful_init.py, which asserts the full criterion is strictly stronger | **FIXED** |

### Class V, Validation: the diagnostic itself fails

5 of 5 closed.

| ID | statement | sites | trigger | observable | silent | evidence | status |
|---|---|---|---|---|---|---|---|
| **D-V1** | the refuted-hypothesis table records nOrders 100 vs 1050 bit-identical as ruling out capacity, but the experiment was mis-specified: varying nOrders without varying the init snapshot width and book_depth leaves the replay building 20 init orders either way | `src/post_training/heuristic_learning/fidelity.py:40` | reading the table and using it to rule out D-I1 | whether the experiment also varied the init_book width | **yes** | the negative result is real but carries zero information about D-I1. FIXED: the refuted-hypothesis table now carries the correction beside the verdict rather than deleting the entry, so the experiment can be judged along with its conclusion | **FIXED** |
| **D-V2** | the documentation attributes the residual to lost within-level queue order, which holds for the real arm but not for the gen arm, where D-I1 dominates | `src/post_training/heuristic_learning/fidelity.py:51` | reading the documentation | whether both arms' residuals are explained by a single mechanism | **yes** | the natural experiment in docs/fidelity_defect_taxonomy.md section 5. FIXED: the queue-order bound is now stated as explaining the real arm only; the gen arm's onset is recorded as unlocated | **FIXED** |
| **D-V3** | n_unresolvable_ids conflates two phenomena of opposite significance: benign cancels of orders created before the conditioning window, and fatal cancels naming an init order absent at this replay depth | `src/post_training/heuristic_learning/fidelity.py:143` | every report | split the count three ways by oid >= 0, oid == -99, and oid inside the init range | **yes** | measured: the 207 unresolvable ids in gen window #258 are exactly its 207 negative ids. FIXED: describe_stream splits the count into n_unresolved_positive / n_lookup_miss_sentinel / n_init_ref_in_depth / n_init_ref_out_of_depth, and records first_bad_id_step so onset can be separated from magnitude. Measured on the 8 windows: the real arm is 100% benign positive ids, out-of-depth references occur only in #258, and 4 of 8 gen windows show onset=NO | **FIXED** |
| **D-V4** | the integration test welds its baseline to a timestamped absolute path produced by a stale worktree | `tests/integration/test_simulator_fidelity.py:27` | after data is regenerated | the directory the test references is not the current output directory | no | static. FIXED: RUN_DIR now reads SIGMA_EPISODES_DIR with the newest measured set as default, so re-running generation does not orphan the baseline | **FIXED** |
| **D-V5** | EXPECTED_GEN_MEAN_MIN = 0.40 freezes a measurement now believed to be dominated by a fixable defect into an acceptance expectation | `tests/integration/test_simulator_fidelity.py:39` | every test run | whether the threshold's provenance is a post-fix measurement | no | static. FIXED: the floor is re-derived from the fixed generator (mean 0.7856, floor 0.70) and now carries its provenance in the comment, including that D-R6 is still open on this very data. A regression detector with a stated origin, not an acceptance bar with a forgotten one | **FIXED** |

<!-- REGISTER:END -->

---

## 5. Evidence, including what it overturned

A classification earns its keep by turning isolated numbers into checkable
propositions. This section records verification and **three self-refutations**,
which is what falsifiability actually looks like in practice.

### 5.1 First pass: a strong correlation

`D-I1` predicted that windows with more bad ids would replay worse. The bad-id
count spans 0 to 207 across these windows with no new experiment needed.

| window | `oid=-99`<br>(D-R2) | init id<br>(D-I1) | bad ids | **gen fidelity** | gen mean\|Δ\| | real fidelity |
|---|---:|---:|---:|---:|---:|---:|
| 2026-01-12 #1478 | 0 | 0 | **0** | **0.7689** | 0.30 | 1.0000 |
| 2026-01-27 #3751 | 0 | 5 | 5 | 0.6256 | 1.35 | 1.0000 |
| 2026-01-30 #4365 | 0 | 6 | 6 | 0.3367 | 2.80 | **0.7967** |
| 2026-01-02 #133 | 0 | 7 | 7 | 0.6900 | 0.67 | 0.9200 |
| 2026-01-23 #3216 | 0 | 7 | 7 | 0.6722 | 2.24 | 1.0000 |
| 2026-01-13 #1510 | 0 | 14 | 14 | 0.6056 | 1.59 | 0.9744 |
| 2026-01-12 #1386 | 6 | 9 | 15 | 0.4711 | 2.08 | 1.0000 |
| 2026-01-05 #258 | 15 | 192 | **207** | **0.2389** | **28.55** | 1.0000 |

Pearson r = −0.906 over the four windows measured at the time. `D-I1` looked
strongly supported.

### 5.2 Second pass: tightening the definition of "bad" shrank the claim to 1 in 8

`init_msgs_from_l2` reshapes `book_l2` to `(2L, 2)`, so row j is
`(book_l2[2j], book_l2[2j+1])`. With a LOBSTER layout of
`[ask_p1, ask_q1, bid_p1, bid_q1, ask_p2, …]`, **row j maps to the same
physical (side, level) at any width**: at 10 levels and at 500, id `-2-j` names
the same order.

`D-I1` can therefore only bite when `oid < -21`, that is, beyond the replay's
depth:

| window | init ids | inside 10 levels | **beyond 10 levels** | deepest index |
|---|---:|---:|---:|---:|
| #258 | 192 | 7 | **185 (96%)** | 84 (level 43) |
| the other 7 | 48 | **48** | **0** | ≤ 19 |

**Only 1 window in 8 actually hits it.**

### 5.3 Third pass: the causal half was refuted

Aligning the first-divergence step against the positions of the bad ids:

| window | first divergence | **bad ids before that step** |
|---|---:|---:|
| #4365 | 15 | **0** |
| #133 | 28 | **0** |
| #3751 | 56 | **0** |
| #1478 | 67 | **0** |
| #1510 | 104 | 4 |
| #258 | 200 | 6 (first **out-of-depth** id at step 57, 143 steps earlier) |
| #3216 | 262 | 1 |
| #1386 | 353 | 4 |

**Four of eight windows diverge with no bad id beforehand at all.** And 182 of
#258's 185 out-of-depth ids fall **after** its divergence point, which explains
its 28.55-tick magnitude against 0.30 to 2.80 elsewhere, and explains nothing
about why it began diverging at step 200.

**Conclusion corrected: `D-I1` amplifies divergence, it does not cause it.**
The r = −0.906 in 5.1 was driven almost entirely by one window and was
measuring **magnitude**, not **onset**.

### 5.4 Regenerating on the fixed code (2026-08-01)

After the execution class was closed, the same episodes were regenerated with
checkpoints, seeds and windows held identical so that **code was the only
variable**: `tasks/agentic_mm_20260801T014703Z`, manifest recording
`git_sha=4b9d5da`.

| | pre-fix generator | **fixed** (`ee3807c` exact-timestamp lookup) |
|---|---:|---:|
| gen arm mean | 0.5511 | **0.7856** |
| #258, the worst window | 0.2389 | **0.9044** |
| out-of-depth init refs | 185 | **19** |
| lookup-miss sentinels | 21 (0.3%) | **527 (7.3%)** |
| of which bid side (direct hits on the init order at id −99) | 6 | **207** |
| real arm | 5/8 · 0.9614 | 5/8 · 0.9614 (**unchanged**, as it must be) |

Three findings:

1. **The one-second-tolerant lookup was the main contaminant.** Out-of-depth
   references fell from 185 to 19, so most of them were an artefact of the
   fuzzy lookup rather than of initialisation width, narrowing `D-I1` again.
2. **There is a regression hidden inside the improvement.** Exact matching
   converts silently-wrong resolutions into honest misses, and every miss
   writes the sentinel, so `D-R6`'s direct-hit exposure went from 6 to 207, a
   34x increase.
3. **`D-R6` moved from arithmetic to measured.** `cancel_order`
   (`JaxOrderBookArrays.py:130`) matches **on order id alone, with no price
   constraint**, so at L=500 a message carrying `oid=-99` directly hits the
   init order at L2 row 97 and quantity is removed from it.

**None of those three was visible before the diagnostic was split.** All 527
sentinels would have been absorbed into `n_unresolvable_ids`. That is the
return on fixing class V before classes R, I and T.

### 5.5 The onset, located (2026-08-01, `autopsy.py`)

Section 5.3 left the onset at "class R or class O". A first-divergence autopsy
that compares books as **price to quantity maps** rather than positional arrays
takes it to named defects, **without changing any generation code**.

| attribution | real arm | gen arm |
|---|---|---|
| **D-I3** depth truncation (**fixable**) | **8 / 8** | 3 / 8 |
| **D-R3** price and id name different orders | 0 / 8 | **5 / 5 of the in-window ones** |
| **D-O1** within-level queue order (the previous attribution) | **0 instances** | **0 instances** |

**Decisive evidence 1 (D-I3).** Narrowing the comparison window moves the real
arm's first divergence later, monotonically: median `106 → 346 → 588 → 900` as
K goes 9 → 3, and comparing only the touch leaves 5 of 8 real windows with zero
divergence across all 900 messages. If the cause were anything other than "the
replay does not know what sits below level L", narrowing would not do that.

**Decisive evidence 2 (D-R3).** All 5 in-window gen divergences share one
signature: the recorded price is not where the generator removed quantity.

| window | message price | where the generator actually removed | delta |
|---|---:|---:|---:|
| #133 | 8989700 | 8989600 | 1 tick |
| #258 | 9317800 | 9318900 | 11 ticks |
| #1478 | 9499600 | 9499800 | 2 ticks |
| #3216 | 9306700 | 9305900 | 8 ticks |
| #3751 | 9286200 | 9286100 | 1 tick |

It is **depth-invariant**: #3751 diverges at step 4 at every comparison width
from 9 levels down to 1. The autopsy's `APPLIED_NOT_RECORDED` and
`DIFFERENT_PRICE` signatures are therefore **one defect**, differing only in
whether the replay's price fallback happened to remove something.

**Two corrections, both to conclusions of my own:**

1. `D-O1` (the queue-order bound) had been recorded as the explanation for the
   real arm's residual. The reasoning behind it was sound: `cancel_mode` had no
   effect, so divergence never reaches the cancel fallback, and the trigger is
   always a `type=4`. **The conclusion was still wrong**, and direct
   measurement points at `D-I3`. `D-O1` remains true as a bound with **zero
   measured instances** here. It is kept because it is true, not because it
   explains anything observed. **Calling a fixable defect an information bound
   stops people trying**, which is exactly the failure mode class V exists to
   catch.
2. `D-R5` (no per-message provenance) had been called the critical path,
   because without it you cannot separate `D-R1` from `D-R3`. **That holds only
   if you restrict yourself to the CSV.** The replay side has full information.
   `D-R5` is downgraded to useful but not blocking.

### 5.6 Limits of the autopsy, stated rather than hidden

Dropping the deepest level **cannot fully suppress the truncation edge**. Keep
the top L-1 levels of an L-level book and remove the top level: the kept set
slides from `{p0..p_{L-2}}` to `{p1..p_{L-1}}`, so one edge entry appears
however many levels are dropped.

A single step's signature is therefore evidence, not proof. The conclusions in
5.5 rest on the narrowing sweep and the out-of-range test, both of which are
robust to this effect. The limitation is in the module docstring and pinned by
a regression test.

---

### 5.7 D-I3 fixed, verified, and what it overturned (2026-08-01)

Faithful initialisation was implemented and the episodes regenerated with
checkpoints, seeds and windows unchanged, so code was again the only variable
(`tasks/agentic_mm_20260801T132010Z`).

| | degraded init | **faithful init** |
|---|---:|---:|
| real arm, mid | 5/8 exact, 0.9614 | **8/8 exact, 1.0000** |
| real arm, full book | 0/8 exact, 0.4058 | **8/8 exact, 1.0000** |
| gen arm, mid | 0/8 exact, 0.7856 | **8/8 exact, 1.0000** |
| gen arm, full book | 0/8 exact, 0.0326 | **8/8 exact, 1.0000** |

Zero divergence over 900 messages in all 8 windows, on the strict criterion.

**Validated by negative control**, because a criterion that reports 1.0000
everywhere has to be shown capable of reporting less:

| perturbation | full-book | first divergence |
|---|---:|---:|
| none | 1.0000 | none |
| every message quantity +1 | 0.0000 | step **0** |
| delete message 50 | 0.0556 | step **50** |
| quantities +1 from message 300 | 0.3400 | step **306** |
| same data, degraded initialisation | 0.3300 | step 6 |

The first-divergence step tracks the injection point. Two perturbations that
went undetected were checked rather than waved away: they touch prices 8912100
and 9659900 against a touch around 9286100, so they are outside the ten-level
window and invisible by construction, not by failure.

### 5.8 What the result overturns

**`D-R3` was not a record defect. It was a symptom of `D-I3`.** `get_sim_msg`
writes the same `(price, oid)` pair into the simulator message and into the
CSV, and `cancel_order` resolves by oid alone, so the generator and the replay
apply an identical message to an identical book and reach an identical result.
"The recorded price is not where quantity was removed" is the engine doing what
it does with an oid that lives at another price, and it is reproducible. It
looked like a record defect only because the replay's book was different. The
section 5.5 attribution of 5/5 in-window gen divergences to `D-R3` is
withdrawn.

`D-I1` goes the same way: with the orders it named now existing, its
out-of-depth references resolve.

**And a distinction the register had not been drawing.** Window #258 carries
**544 lookup-miss sentinels** and still replays 8/8 exact on the full book,
because the sentinel is applied identically on both sides. `D-R2` and `D-R6`
are therefore **not fidelity defects**. What remains of them is semantic: the
generator removes quantity from an unrelated bid level, and the replay
faithfully reproduces that.

> **Fidelity 1.0 does not mean the market is correct. It means the replay
> reproduces what the generator did, including its mistakes.**

This register's subject is record to replay fidelity, so it measures the first
and not the second. Defects that corrupt the market identically on both sides
need a different criterion, and saying so is the honest boundary of what has
been achieved here.

### 5.9 The last two measurements, and one conclusion deliberately not drawn

`D-R5` closed by making the record say how it resolved each id, and that made
`D-R1` and `D-R4` measurable for the first time. Both had stood as "unmeasured"
because the record carried nothing to measure them with.

Over 7200 generated messages, 3275 of them cancels or deletes:

| provenance | count | of all | of cancels |
|---|---:|---:|---:|
| not a cancel | 3925 | 54.5% | |
| exact-timestamp hit | 2854 | 39.6% | 87.1% |
| price fallback | 66 | 0.9% | 2.0% |
| **total miss** | **355** | 4.9% | **10.8%** |
| **NaN branch fired (D-R1)** | **0** | **0.00%** | |
| **ref price disagrees (D-R4)** | **1968** | 27.3% | **60.1%** |

**`D-R1` fires zero times.** That confirms by direct measurement what perfect
fidelity had only implied. It stays OPEN, because the code still records the
original message when it does fire, but it is **no longer silent**: provenance
bit 2 records the branch, so an occurrence is detectable from the artifact
rather than invisible. That is a change in the defect's character, and the
`silent` field now says so.

**`D-R4` fires on 60.1% of cancels, and what that means is left open on
purpose.** This repository does not contain the preprocessing that builds
`price_ref`, so it could not be confirmed that `price` and `price_ref` are
relativised against the same mid. A high rate therefore means **either** the
model emits inconsistent pairs **or** the two fields are not comparable in this
formulation, and nothing measured here separates those.

**The control that would settle it** is to compute the same bit on the real
conditioning messages, which are consistent by construction: a high rate there
proves the fields are not comparable, a near-zero rate proves the model is
inconsistent. It needs its own change and has not been run.

What is established is narrower and still worth having: a prediction that was
decoded and thrown away is now a recorded number. Asserting the wider claim
would repeat exactly the error this document has had to correct four times
already.

### 5.10 D-R4's control, and the fifth refutation

Section 5.9 measured `price != price_ref` on 60.1% of generated cancels and
**refused to say what it meant**, because this repository does not contain the
preprocessing that builds `price_ref` and so cannot confirm the two fields
share a reference frame. It named the control that would decide it. That
control has now been run.

**The design.** Real market messages are consistent by construction: a LOBSTER
cancel's own price *is* the price of the order it cancels. Running the
identical check on them therefore decides between the two readings, and the
decision rule was stated before the measurement:

| rate on real messages | conclusion |
|---|---|
| high | the fields are not comparable, the check is wrong |
| near zero | the model really does emit inconsistent pairs |

**The result.**

| stream | cancels | disagreements | rate |
|---|---:|---:|---:|
| real conditioning (control) | 1904 | 1204 | **63.24%** |
| real continuation (control) | 3448 | 1784 | **51.74%** |
| generated | 3416 | 2164 | 63.35% |
| **real, combined** | **5352** | **2988** | **55.83%** |

**A check that flags 56% of known-consistent data is measuring something other
than consistency.** `price` and `price_ref` do not share a reference frame, the
60.1% means nothing about the model, and `D-R4` is REFUTED: the premise that
the discarded ref fields could validate the message by equality is false.

Two things were done to make the control airtight rather than merely
suggestive. Both sides were changed to run **literally the same comparison**,
`price != price_ref` on the decoded fields, instead of the algebraically
equivalent absolute-price form: a control whose arithmetic differs from the
thing it controls for is not a control. And the field indices were re-verified
against `msg_to_lobster_format`, which reads the same decoded layout, because
a wrong index would have invalidated the control silently.

The bit is kept, and renamed. Kept because it is a real, well-defined datum and
because deleting it would delete the evidence that the control was run. Renamed
from `ref_price_disagrees` to `price_differs_from_ref` because **a field whose
name asserts a refuted conclusion will be read as that conclusion by the next
person**, and a test now pins the old name out of existence.

**What is different about this refutation.** The four in sections 5.3 and 5.5
followed the same shape: a conclusion was written down, and a later measurement
overturned it, after the wrong conclusion had been in the documentation for a
while. This one did not. The interpretation was withheld, the decision rule was
stated in advance, the control was run, and it killed the hypothesis the same
day. The discipline of not concluding what you cannot measure is what made the
difference, and this is the first place in this document where its benefit is
visible rather than asserted.

### 5.11 D-X7: generation is reproducible only on the node that produced it

Found by a tripwire rather than by looking. The lookup-miss sentinel count moved
355 to 588 between two runs, and no code between them touches generation.

**Byte-comparing the streams settled what changed.** The conditioning streams
are identical, so window selection is stable and the difference is purely on
the generation side. All 8 generated streams differ, first at rows 7 to 118 and
then in 87 to 99% of the remaining rows. The first difference not being at row
1 is itself evidence: an unseeded RNG would differ from the first message, so
this looked like numerical drift flipping a sampling decision and autoregression
amplifying it.

**The control, with the decision rule stated first.** Two runs pinned to the
same node: identical output implicates cross-device numerics, differing output
implicates the RNG.

| runs | result |
|---|---|
| nid010147 vs nid010148 | 8/8 windows differ |
| **nid010148 twice** | **8/8 windows byte-identical, 7200 messages** |

So generation is deterministic **within** a node. The RNG is not implicated.

**What follows, and what does not.** The frozen-market design is unaffected: a
market is frozen by being written to disk, and every candidate policy replays
that same record, so paired comparison is intact. What was broken is narrower
and was still real: **the node is part of the configuration, and the record did
not name it.** That is the same shape as `D-T3`, an incomplete record producing
an unreproducible artifact, and the fix is the same. The manifest now writes
the node.

Any baseline derived from regenerated data is therefore node-specific, and the
sentinel band in the integration test says so rather than pretending to be a
property of the code.

**A process failure worth recording.** The first attempt at this control failed
with GPU OOM on both runs, because the node had four other compute processes
holding 120 GB. `squeue` showing the allocation as mine says nothing about
whether its GPUs are free, and the physical-GPU gate that exists for exactly
this was skipped. The rerun re-checks the gate immediately before each launch
rather than reusing the earlier check, because a check is not a lease.

## 6. Self-check of the taxonomy

A classification has to answer "how do you know nothing is missing". The answer
here is **constructive**:

| check | answer |
|---|---|
| are the classes mutually exclusive | yes. Each binds **one** symbol of the identity; X and V are a meta layer and do not overlap them |
| are they exhaustive | yes. R/I/T/O enumerates the identity's symbols; drop one and a symbol has no owner |
| is every entry falsifiable | yes. `observable` is mandatory and enforced; an entry that cannot fill it is rejected |
| has anything unfixable been smuggled in | no. `D-O1` is explicitly `BOUND`. It is a bound, not a bug |
| what about hypotheses ruled out | kept and marked, and `D-V1` records a case where **a refutation was itself invalid** |
| is the meta layer enough | X covers "is the input what you think", V covers "would you see it". Together with the four entity classes they span committing code to drawing a conclusion |

**The first thing this taxonomy caught was `D-V1`**, a hypothesis recorded as
ruled out whose experiment was mis-specified and therefore carried no
information about the initialisation defect it was taken to eliminate. Working
as a flat bug list, that wrong refutation would have stayed in the document and
gone on discouraging anyone from suspecting initialisation depth. **The second
and third were `D-O1` and `D-R5`, both conclusions of my own** (section 5.5).

---

## 7. Repair order, derived from the classification and revised by measurement

The order is not by severity. It is by **dependency**, meta-classes before
entity classes and root causes before symptoms, and it has been revised twice
as measurements arrived.

| stage | what | status | why here |
|---|---|---|---|
| 0 | **class X** (D-X1 … D-X6) | **done** | while artifacts and code can come apart, no measurement downstream is worth anything. Prerequisite for everything |
| 1 | **class V** (D-V1 … D-V5) | **done** | the diagnostic has to be able to tell the classes apart, or a repair cannot be shown to have worked. It immediately caught the 34x sentinel regression hiding inside a 0.23 improvement |
| 2 | **D-I3** replay from the L500 wide book | next | cause of 8/8 of the real arm, and the wide book already exists at `recon_2026-05`, which the generator itself uses |
| 3 | **D-R3** make the written price and id name the same order | | 5/5 of the in-window gen divergences, and depth-invariant, so no amount of initialisation depth removes it |
| 4 | **D-R6** choose a miss sentinel outside the init id space | | 527 sentinels, 207 bid-side direct hits |
| 5 | **D-T3 / D-I2** make the record self-describe `θ` and its width | | stops the same class of mismatch recurring |
| 6 | **class O** | declare only | `D-O1` goes into the acceptance criteria as a known bound; `D-O2` gets a full-depth criterion if one is wanted |

**Stage 2's verification is already defined**: with 500-level initialisation the
real arm should go from 5/8 to close to 8/8. That prediction is falsifiable,
which is the point of stating it before doing the work.

---

## 8. Scope

This register covers the fidelity of the **record to replay** path. It does not
cover the correctness of the market-making policy itself, which
`tests/unit/test_mm_sim.py` covers through conservation and no-lookahead tests,
and it does not cover the world model's distributional quality, which LOBbench
covers.

Entries should have their `status` updated as repairs land. **Entries are never
deleted.** The existence of a `REFUTED` or `BOUND` entry is itself the record
that stops an investigation being repeated, and `D-V1` is the standing example
of a refutation that turned out to be worthless.

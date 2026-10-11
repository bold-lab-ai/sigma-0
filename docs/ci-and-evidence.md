# CI and evidence

The things worth testing need GH200s, Lustre, frozen checkpoints and market
data. A GitHub runner has none of them. So the split is:

```text
   CLUSTER  (GH200, Lustre, checkpoints, market data)
        |   ci/record_evidence.py            writes ci/evidence/<kind>.json
        v
   ci/evidence/*.json                        committed with the code they describe
        |
        v   ci/check_evidence.py             standard library only, seconds
   GITHUB ACTIONS                            audits: hashes still match, assertions still hold
```

Every record carries a SHA-256 of each source file its numbers depend on, the
metrics, and its assertions as data (`metric`, `op`, `value`, `why`). The audit
recomputes the hashes and evaluates the assertions. Change a hashed file and
CI fails naming it; the fix is to re-record on the cluster, never to edit a hash.

## The records

| record | what it claims | hashes | how to re-record | needs |
|---|---|---|---|---|
| `suite.json` | pytest inventory and outcome; a test count floor | every file under `tests/` | `JAX_PLATFORMS=cpu PYTHONPATH=src python ci/record_evidence.py --suite` | a compute node with all CPUs (`--exclusive`): one test pins its child to CPUs 0-3 |
| `register.json` | the defect register's state: counts, every live defect names an observable | `docs/defect_register.json`, `tools/render_register.py`, `tests/unit/test_defect_register.py` | `... --register` | nothing else; stdlib |
| `simulator.json` | market-making simulator fidelity, accounting invariants, determinism | `matching_engine/jaxob/{JaxOrderBookArrays,jorderbook}.py`, `post_training/heuristic_learning/{episode_builder,fidelity,mm_sim}.py` | `... --simulator --episodes <run dir>` | a generated episode run on Lustre; the recorder fingerprints the input files and refuses changed inputs without `--accept-new-inputs` |
| `return_bench.json` | identity check exact, shuffle-null size, per-horizon numbers | `lobpipeline/return_bench/{event_anchored,run_event_bench}.py`, `run/base_model/eval/pick_trade_anchors.py`, `base_model/data/data_files.py` | `... --return-bench --return-bench-dir <infer dir>` | an inference directory; the current one is `/lus/lfs1aip2/projects/public/u6gb/ci_evidence_inputs/return_bench_hp_base_member0` |
| `injection.json` | the injected order, and nothing else, lands at the flagged step | `base_model/inference/generate.py`, `run/base_model/runtime/inference.py`, `lobpipeline/injection_verification.py`, `tools/diagnostics/verify_injection.py` | a GPU rollout with `--inject_step`, then `tools/diagnostics/verify_injection.py`, then `... --injection --injection-report <report>` | a GPU, the pinned `m3-goog-78m-u6gb` checkpoint, a wide-book data day |
| `profiling.json` | the per-position CE decomposition records | two records under `docs/experiments/trainloss_profiling_20260826/data/`, `run/base_model/ce_by_position.sbatch`, `run/base_model/evaluate_ce_by_position.py`, `base_model/encoding/encoding_26tok.py` | the `ce_by_position.sbatch` job, then the recorder path documented in `ci/check_ce_decomposition.py` | a GH200 and the pinned checkpoint |

A file that moves byte-identically only changes its path key in the record;
the hash stays. That is how the September 2026 reorganisation kept five records
valid. Editing the content of a hashed file is what forces a re-record.

`recorded.host` must be a cluster host, and `recorded.commit` must be an
ancestor of the branch: a rebase that rewrites the recorded commit orphans the
record, so record after the last rebase.

## The other guards

All under `ci/`, standard library only, each with a `--self-test` that plants
a defect and must catch it. One workflow, `checks`, runs them as steps of one
job (each guard finishes in seconds; GitHub bills a job in whole minutes, so
nine workflows cost about ten minutes per push for thirty seconds of work).
Every step runs even when an earlier one fails; the final `Verdict` step fails
the job if any did.

| guard | step prefix in `checks` | refuses |
|---|---|---|
| `check_evidence.py` | evidence | a record whose source hashes or assertions no longer hold |
| `check_english_only.py` | english-only | Chinese outside the listed to-do files |
| `check_legacy_imports.py` | legacy-imports | any `s5.*`, `lob.*` or root-module import anywhere in `src/`, `run/`, `tests/`, `tools/` or `ci/` |
| `check_lobbench_merge.py` | lobbench-merge | a vendored scorer tree that no longer matches `merge.json`; the three model-zoo notebooks with stripped outputs |
| `check_ce_decomposition.py` | ce-decomposition | a private copy of the encoding table, a broken `LOG_CE_BY_POSITION` wiring, a stripped profiling notebook |
| `check_message_perplexity.py` | message-perplexity | a reporter whose field list drifts from the encodings, a stripped notebook, a committed report that a fresh derivation does not reproduce |
| `check_state_posttraining.py` | state-posttraining | the state model losing one of its four post-training methods or its parity test |
| `mutation_check.py` | return-bench | a return-bench test suite that no longer catches five planted defects |

The return-bench steps also refuse a record whose recorded commit is not an
ancestor of the branch or whose host looks like a CI runner, and a loosened
identity tolerance. The injection steps re-audit the newest
`docs/evidence/injection_report_*.json` and check it agrees with
`injection.json` on digest, node, parameters and verdict.

## When CI is red

1. Read the failing step's summary; every guard prints the file it objects to.
2. If it is `check_evidence.py` naming a source, you edited a hashed file. Re-record that kind on the cluster and commit the record with the code change.
3. If it is `suite.json` with `pytest.failed = 1`, the recorded run had a failure. Fix the test or the code, then re-record.
4. Never lower a test floor, delete a test, or hand-edit a record to get green. Say why in the PR if a floor must move.

Run everything locally first:

```bash
python3 ci/check_evidence.py
for g in english_only lobbench_merge ce_decomposition message_perplexity state_posttraining legacy_imports; do
  python3 ci/check_$g.py || break
done
```

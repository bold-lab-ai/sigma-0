# Provenance of the vendored nano-sigma code

The policy bench runs the two nano-sigma checkpoints that lead the
leaderboard: `newtok_a_r1` (simulator) and `returnft_lobert_pass1` (signal).
Their encodings must match training exactly, so the encoding and model code
is copied here byte for byte instead of being rewritten.

| File | Upstream | SHA-256 |
|---|---|---|
| `data.py` | `nanosigma/data.py` | `44de48d7165e7a1b7c00ba2d5f5ba2f824799395c579e7ac9009051a1a95ad05` |
| `newtok.py` | `nanosigma/newtok.py` | `5a903f14ea6f10d5f1917f2ab1536a6ea3ce6d361d0efad4635e519536842ab9` |
| `model.py` | `nanosigma/model.py` | `b62818bb170d920715f809e52d34140e55092b0124c35312fa2b8d50ba19e1bf` |
| `model_newtok.py` | `nanosigma/model_newtok.py` | `b6d1a6e7f24eaae57bf8e2a206e98c009d70fbad7db69d378641efbf56134e4c` |

- Upstream: Kang Li's nano-sigma working tree, copied on 2026-10-29. The
  checkpoints record commit `d9cbd4e` with uncommitted changes, so the hashes
  above, not the commit, identify the code.
- Upstream `model.py` was last changed on 2026-10-25, after both checkpoints
  were written. All three checkpoints still load into it with `strict=True`
  (job 7834746). That their outputs are unchanged is gate G4.
- The files are unmodified. `tests/unit/test_policy_bench_codec.py` checks the
  hashes, so an edit here fails the tests.
- Do not edit these files. Code that needs different behaviour lives in the
  parent package and is tested against these copies.

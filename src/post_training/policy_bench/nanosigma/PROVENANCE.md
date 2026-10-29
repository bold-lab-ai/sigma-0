# Provenance of the vendored nano-sigma code

The policy bench runs the two nano-sigma checkpoints that lead the
leaderboard: `newtok_a_r1` (simulator) and `returnft_lobert_pass1` (signal).
Their encodings must match training exactly, so the encoding code is copied
here byte for byte instead of being rewritten.

| File | Upstream | SHA-256 |
|---|---|---|
| `data.py` | `nanosigma/data.py` | `44de48d7165e7a1b7c00ba2d5f5ba2f824799395c579e7ac9009051a1a95ad05` |
| `newtok.py` | `nanosigma/newtok.py` | `5a903f14ea6f10d5f1917f2ab1536a6ea3ce6d361d0efad4635e519536842ab9` |

- Upstream: Kang Li's nano-sigma working tree, copied on 2026-10-29. The
  checkpoints record commit `d9cbd4e` with uncommitted changes, so the hashes
  above, not the commit, identify the code.
- The files are unmodified. `tests/unit/test_policy_bench_codec.py` checks the
  hashes, so an edit here fails the tests.
- Do not edit these files. Code that needs different behaviour lives in the
  parent package and is tested against these copies.

#!/usr/bin/env python3
"""CLI for the do(Q) injection rollout verifier.

Thin wrapper over ``lobpipeline.injection_verification`` so the checker runs
from a repo checkout with plain ``python3`` — no jax, no GPU, no data beyond
the rollout CSVs being compared (and none at all in ``--check`` mode, which
re-audits an already-written report on any machine).

    python3 tools/diagnostics/verify_injection.py \
        --dir-p ARM_P --dir-p2 ARM_P2 --dir-o ARM_O --dir-i ARM_I \
        --inject-step 8 --inject-qty 777 --inject-side 1 \
        --inject-event-type 1 --inject-offset-ticks -1 \
        --node $(hostname) --out injection_report.json

    python3 tools/diagnostics/verify_injection.py --check injection_report.json
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from lobpipeline.injection_verification import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

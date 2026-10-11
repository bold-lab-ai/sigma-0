#!/bin/bash
#SBATCH --job-name=sigma0-package-smoke
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Smoke test for the run-packaging stage: stage a tiny synthetic run in node-local
# $TMPDIR, build the self-contained squashfs, mount-verify it, then rsync the single
# squashfs (+ sidecar) to Lustre. No GPU; mksquashfs/squashfuse are CPU + local I/O.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
# Ensure mksquashfs/unsquashfs + a python>=3.10 are on PATH (squashfuse is system).
export PATH="/home/u6gb/kangli.u6gb/miniforge3/bin:$PATH"

echo "[env] python=$(command -v python) mksquashfs=$(command -v mksquashfs) squashfuse=$(command -v squashfuse)"

WORK="${TMPDIR:-/tmp}/sigma0_pkg_${SLURM_JOB_ID:-local}"
RUN="$WORK/run"
OUT="$WORK/out"
MNT="$WORK/mnt"
rm -rf "$WORK"
mkdir -p "$RUN/predictions" "$OUT" "$MNT"

# --- Stage a synthetic run: placeholder per-ticker predictions + canonical scores ---
python - "$RUN" <<'PY'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
pred = run / "predictions"
pred.mkdir(parents=True, exist_ok=True)
try:  # real .npy when numpy is importable; raw placeholder bytes otherwise
    import numpy as np
    for tk in ("GOOG", "AAPL"):
        np.save(pred / f"{tk}_2026-01.npy", np.zeros((4, 3), dtype="float32"))
except Exception:
    for tk in ("GOOG", "AAPL"):
        (pred / f"{tk}_2026-01.npy").write_bytes(b"\x00" * 48)
scores = {m: {"mean": 0.0, "best": 0.0, "worst": 0.0} for m in ("l1", "wasserstein", "ks")}
(run / "scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
print(f"[fixture] staged predictions + scores under {run}")
PY

# --- Build the self-contained squashfs into $TMPDIR ---
python run/benchmarking/run_package.py --execute \
  --run-name smoke \
  --predictions-root "$RUN/predictions" \
  --scores-path "$RUN/scores.json" \
  --output-dir "$OUT"

SQ="$OUT/smoke.squashfs"
test -f "$SQ" || { echo "[FAIL] squashfs not produced"; exit 1; }

# --- Mount-verify the image is self-contained ---
squashfuse "$SQ" "$MNT"
echo "[verify] top-level:"; ls -1 "$MNT"
echo "[verify] predictions/:"; ls -1 "$MNT/predictions"
if ! { test -f "$MNT/manifest.json" && test -f "$MNT/scores.json" && test -d "$MNT/predictions"; }; then
  fusermount -u "$MNT" || true
  echo "[FAIL] squashfs missing required members"; exit 1
fi
fusermount -u "$MNT"
echo "[verify] OK self-contained {manifest.json, scores.json, predictions/}"

# --- Persist the single squashfs + sidecar to Lustre (large-file rsync) ---
DEST="$PWD/artifacts/packages"
mkdir -p "$DEST"
rsync -a "$SQ" "$OUT/index_smoke.json" "$DEST/"
echo "[done] packaged -> $DEST/smoke.squashfs"
rm -rf "$WORK"

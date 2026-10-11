#!/bin/bash
# P-1 preflight: can `claude -p` reach the Anthropic API from a compute node?
#
# There is no precedent in this repo for running `claude -p` inside a SLURM
# allocation; only `codex exec` has done it (tasks/u6gb_16_nodes_daily_log).
# The agentic market-making OPRO loop needs one LLM call per round, so this
# probe decides where that loop runs:
#   PASS -> OPRO driver runs as a compute-node job (preferred)
#   FAIL -> OPRO driver falls back to a bounded, single-shot login-node script
#           (CPU-only, hard round cap, < 30 min, not a daemon)
#
# Probe scope: the prompt asks for one token and gives no reason to use a tool.
# Verified on the login node that `--allowed-tools ""` does NOT actually strip
# the tool list, so we disallow the write-capable tools explicitly and keep the
# default permission mode (which cannot auto-approve anything non-interactively).
# A hard `timeout` bounds the call either way.

#SBATCH --job-name=sigma0-preflight-claude-p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=00:05:00
# Neutral, job-id-keyed (D-X3): a #SBATCH directive cannot follow $WORKDIR.
#SBATCH --output=/lus/lfs1aip2/projects/public/u6gb/slurm_logs/%j.out

set -euo pipefail

OUT_DIR="${SIGMA_LOG_ROOT:-/lus/lfs1aip2/projects/public/u6gb/slurm_logs}"
mkdir -p "$OUT_DIR"

CLAUDE_BIN="${CLAUDE_BIN:-/home/u6gb/kangli.u6gb/miniforge3/bin/claude}"
PROBE_MODEL="${PROBE_MODEL:-claude-opus-5}"
RESULT_JSON="$OUT_DIR/${SLURM_JOB_ID}_result.json"

echo "[preflight] node=$(hostname)"
echo "[preflight] date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[preflight] HOME=$HOME"
echo "[preflight] claude_bin=$CLAUDE_BIN"

# --- gate 1: binary present -------------------------------------------------
if [ ! -x "$CLAUDE_BIN" ]; then
    echo "[preflight] FAIL: claude binary not executable at $CLAUDE_BIN" >&2
    exit 10
fi
"$CLAUDE_BIN" --version || { echo "[preflight] FAIL: --version failed" >&2; exit 11; }

# --- gate 2: credentials readable from the compute node ---------------------
CRED="$HOME/.claude/.credentials.json"
if [ -r "$CRED" ]; then
    echo "[preflight] credentials readable: yes ($(stat -c '%s bytes, mode %a' "$CRED"))"
else
    echo "[preflight] credentials readable: NO ($CRED)"
fi

# --- gate 3: raw outbound TCP/TLS to the API host ---------------------------
HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 \
    https://api.anthropic.com/v1/models 2>/dev/null || echo "curl_failed")
echo "[preflight] api.anthropic.com http_code=$HTTP_CODE  (401 = reachable, auth required)"

# --- gate 4: the real thing -- a one-shot headless call ---------------------
set +e
timeout 180 "$CLAUDE_BIN" -p 'Reply with exactly this token and nothing else: PREFLIGHT_OK' \
    --model "$PROBE_MODEL" \
    --disallowed-tools "Bash" "Write" "Edit" "NotebookEdit" "Task" \
    --output-format json \
    > "$RESULT_JSON" 2> "$OUT_DIR/${SLURM_JOB_ID}_stderr.txt"
CLAUDE_RC=$?
set -e

echo "[preflight] claude exit=$CLAUDE_RC"
echo "[preflight] --- raw result ($RESULT_JSON) ---"
head -c 2000 "$RESULT_JSON" || true
echo
echo "[preflight] --- stderr ---"
head -c 1000 "$OUT_DIR/${SLURM_JOB_ID}_stderr.txt" || true
echo

if [ "$CLAUDE_RC" -ne 0 ]; then
    echo "[preflight] VERDICT=FAIL (claude -p exit $CLAUDE_RC on compute node)"
    exit 12
fi

if grep -q "PREFLIGHT_OK" "$RESULT_JSON"; then
    echo "[preflight] VERDICT=PASS  -- OPRO loop can run on a compute node"
    exit 0
fi

echo "[preflight] VERDICT=FAIL (no PREFLIGHT_OK token in response)"
exit 13

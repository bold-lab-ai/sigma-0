#!/bin/bash
# Drive a pull request end to end: open it with a summary, validate the actual
# merge result on this cluster, comment the evidence, then merge when green.
#
# Why validation happens here and not in CI: the test suite needs JAX on
# aarch64 plus this cluster's conda environment. A GitHub-hosted runner would
# either fail to install it or would test a different stack than the one the
# code actually runs on. So the merge result is built and tested locally, and
# the evidence is posted to the PR.
#
# Validation builds the REAL merge result (base + head in a throwaway worktree)
# rather than testing the head branch alone, because a branch that is green on
# its own can still be broken by what it merges into.
#
# Usage:
#   pr_autopilot.sh open   <head-branch> <base-branch> <title> <body-file>
#   pr_autopilot.sh verify <pr-number>
#   pr_autopilot.sh merge  <pr-number> [--squash|--merge|--rebase]
#   pr_autopilot.sh drive  <head-branch> <base-branch> <title> <body-file>
#       open (idempotent) -> verify -> comment -> merge when green
#
# Exit codes: 0 ok, 2 usage, 3 validation failed, 4 not authenticated.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/lus/lfs1aip2/projects/public/u6gb/sigma-0}"
WORKTREE_BASE="${WORKTREE_BASE:-/lus/lfs1aip2/projects/public/u6gb/sigma-0-worktrees}"
PYTHON="${PYTHON:-/lus/lfs1aip2/projects/public/s5e/quant_team/quant/miniforge3/bin/python}"
GH="${GH:-gh}"
GIT="git -C $REPO_ROOT -c safe.directory=$REPO_ROOT"

log() { printf '[autopilot] %s\n' "$*"; }
die() { printf '[autopilot] ERROR: %s\n' "$*" >&2; exit "${2:-1}"; }

require_auth() {
    "$GH" auth status >/dev/null 2>&1 \
        || die "gh is not authenticated. Run: gh auth login" 4
}

# ---------------------------------------------------------------- open -----
# Idempotent: if a PR already exists for this head branch, print its number and
# return instead of erroring out, so `drive` can be re-run safely.
cmd_open() {
    local head="$1" base="$2" title="$3" body_file="$4"
    require_auth
    [ -f "$body_file" ] || die "body file not found: $body_file" 2

    local existing
    existing="$("$GH" pr list --repo "$(repo_slug)" --head "$head" --state open \
                    --json number --jq '.[0].number // empty' 2>/dev/null || true)"
    if [ -n "$existing" ]; then
        log "PR #$existing already open for $head; reusing it"
        echo "$existing"
        return 0
    fi

    "$GH" pr create --repo "$(repo_slug)" \
        --base "$base" --head "$head" \
        --title "$title" --body-file "$body_file" >/dev/null
    local num
    num="$("$GH" pr list --repo "$(repo_slug)" --head "$head" --state open \
               --json number --jq '.[0].number')"
    log "opened PR #$num  ($head -> $base)"
    echo "$num"
}

repo_slug() {
    $GIT remote get-url origin \
        | sed -e 's#^git@github.com:##' -e 's#^https://github.com/##' -e 's#\.git$##'
}

# -------------------------------------------------------------- verify -----
# Builds base+head in a scratch worktree and runs the suite there. Prints a
# markdown report to stdout; returns non-zero if anything failed.
cmd_verify() {
    local pr="$1"
    require_auth

    local head base
    head="$("$GH" pr view "$pr" --repo "$(repo_slug)" --json headRefName --jq .headRefName)"
    base="$("$GH" pr view "$pr" --repo "$(repo_slug)" --json baseRefName --jq .baseRefName)"

    local wt="$WORKTREE_BASE/_autopilot_pr${pr}"
    local report; report="$(mktemp)"
    local rc=0

    # A previous interrupted run may have left the worktree behind.
    if [ -d "$wt" ]; then
        $GIT worktree remove --force "$wt" >/dev/null 2>&1 || true
    fi

    log "building merge result: $base + $head"
    $GIT worktree add --detach "$wt" "origin/$base" >/dev/null 2>&1 \
        || $GIT worktree add --detach "$wt" "$base" >/dev/null

    local merge_status conflicts=""
    if git -C "$wt" -c safe.directory="$wt" \
           -c user.name=autopilot -c user.email=autopilot@localhost \
           merge --no-ff --no-edit "origin/$head" >/dev/null 2>&1 \
       || git -C "$wt" -c safe.directory="$wt" \
              -c user.name=autopilot -c user.email=autopilot@localhost \
              merge --no-ff --no-edit "$head" >/dev/null 2>&1; then
        merge_status="clean"
    else
        merge_status="CONFLICTED"
        conflicts="$(git -C "$wt" -c safe.directory="$wt" diff --name-only --diff-filter=U | head -20)"
        rc=3
    fi

    local compile_status="skipped" test_line="skipped" test_tail=""
    if [ "$merge_status" = "clean" ]; then
        if (cd "$wt" && PYTHONPATH=src "$PYTHON" -m compileall -q src >/dev/null 2>&1); then
            compile_status="pass"
        else
            compile_status="FAIL"; rc=3
        fi

        local out; out="$(mktemp)"
        if (cd "$wt" && JAX_PLATFORMS=cpu "$PYTHON" -m pytest tests -q >"$out" 2>&1); then
            :
        else
            rc=3
        fi
        test_line="$(grep -E '[0-9]+ (passed|failed|error)' "$out" | tail -1)"
        test_tail="$(tail -25 "$out")"
        rm -f "$out"
    fi

    {
        echo "## Automated validation"
        echo
        echo "Built the real merge result (\`$base\` + \`$head\`) in a scratch worktree and"
        echo "ran the suite on the cluster, where the aarch64 + JAX environment this code"
        echo "actually runs on lives."
        echo
        echo "| check | result |"
        echo "|---|---|"
        echo "| merge into \`$base\` | $merge_status |"
        echo "| \`compileall src\` | $compile_status |"
        echo "| \`pytest tests -q\` | ${test_line:-n/a} |"
        echo
        if [ -n "$conflicts" ]; then
            echo "Conflicting files:"; echo; echo '```'; echo "$conflicts"; echo '```'; echo
        fi
        if [ "$rc" -ne 0 ] && [ -n "$test_tail" ]; then
            echo "<details><summary>pytest tail</summary>"; echo
            echo '```'; echo "$test_tail"; echo '```'; echo; echo "</details>"; echo
        fi
        echo "_Ran on \`$(hostname)\` at $(date -u +%Y-%m-%dT%H:%M:%SZ) with \`$PYTHON\`._"
    } >"$report"

    $GIT worktree remove --force "$wt" >/dev/null 2>&1 || true
    cat "$report"
    rm -f "$report"
    return "$rc"
}

# --------------------------------------------------------------- merge -----
cmd_merge() {
    local pr="$1" strategy="${2:---merge}"
    require_auth
    log "merging PR #$pr with $strategy"
    "$GH" pr merge "$pr" --repo "$(repo_slug)" "$strategy" --delete-branch=false
    log "PR #$pr merged"
}

# --------------------------------------------------------------- drive -----
cmd_drive() {
    local head="$1" base="$2" title="$3" body_file="$4" strategy="${5:---merge}"
    local pr; pr="$(cmd_open "$head" "$base" "$title" "$body_file" | tail -1)"

    local report; report="$(mktemp)"
    local rc=0
    cmd_verify "$pr" >"$report" || rc=$?

    "$GH" pr comment "$pr" --repo "$(repo_slug)" --body-file "$report" >/dev/null
    log "posted validation comment to PR #$pr"
    cat "$report"; rm -f "$report"

    if [ "$rc" -ne 0 ]; then
        die "PR #$pr failed validation; NOT merging. Fix, push, and re-run drive." 3
    fi
    cmd_merge "$pr" "$strategy"
}

# ---------------------------------------------------------------- main -----
case "${1:-}" in
    open)   shift; [ $# -eq 4 ] || die "open <head> <base> <title> <body-file>" 2; cmd_open "$@" ;;
    verify) shift; [ $# -eq 1 ] || die "verify <pr-number>" 2; cmd_verify "$@" ;;
    merge)  shift; [ $# -ge 1 ] || die "merge <pr-number> [strategy]" 2; cmd_merge "$@" ;;
    drive)  shift; [ $# -ge 4 ] || die "drive <head> <base> <title> <body-file> [strategy]" 2; cmd_drive "$@" ;;
    *) die "usage: $0 {open|verify|merge|drive} ..." 2 ;;
esac

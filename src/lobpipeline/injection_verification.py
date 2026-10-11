"""Verify a do(Q) injection rollout from its on-disk CSV artifacts alone.

Consumes the save_dir trees written by ``run/base_model/runtime/inference.py``
and decides whether the injection is wired correctly. One judgement gates:

  J4  book row k     between the on-arm's book snapshots k-1 and k the engine
                     processed exactly one message: the injected order. For a
                     passive limit order the delta must be +qty at the derived
                     price and nothing else (modulo the L2 top-10 window
                     shifting). Preconditions (uncrossed book, passive price,
                     determinate mid) must hold in every window.

Five more are computed when reference arms are supplied, and are reported as
diagnostics only:

  J0  P == P2        the same binary run twice is byte-identical
  J1  P == O         pre-PR arm == PR arm with injection off, byte for byte
  J2  O ~ I prefix   the arms agree before the injected step
  J3  msg row k      the model's own message at step k is unchanged, and the
                     provenance sidecar agrees through k
  J5  divergence     the books of O and I first differ exactly at k

The diagnostics compare bytes across separate GPU runs, which holds only when
the node compiles both programs identically -- a node-bound property (defect
D-X7), shown on 2026-08-27 to vary across nodes even for the pre-existing
commit pair. Per PR #21 review, GPU determinism must not gate anything, so
``overall_pass`` reads J4 alone; the diagnostics stay in the report for nodes
where they are informative. Off-means-off is enforced statically instead, by
``tests/unit/test_injection_wiring.py``.

Stdlib only, deliberately: the report this module writes (and re-checks with
``--check``) is the fixed template other machines audit without a JAX
environment, a GPU, or the data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

SCHEMA = "sigma0.injection_verification.v2"

# get_L2_state writes empty ask levels as +maxint and empty bid levels as
# -maxint with zero volume (JaxOrderBookArrays.get_L2_state); maxint is the
# int32 max despite the enum's name.
EMPTY_ASK_P = 2_147_483_647
EMPTY_BID_P = -2_147_483_647
DEFAULT_TICK = 100

_GEN_OB = re.compile(
    r"^(?P<stock>.+)_(?P<date>\d{4}-\d{2}-\d{2})_orderbook_real_id_(?P<i>\d+)_gen_id_0\.csv$"
)

_DATA_SUBDIRS = ("data_cond", "data_real", "data_gen")


# --------------------------------------------------------------------------
# file plumbing (no recursion: one listdir per known subdirectory)
# --------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_rel_files(save_dir: str, flat: bool = False) -> list:
    """Relative paths of every artifact in a sample_new save_dir, or of a
    single flat directory when ``flat`` is set.

    Exactly one listdir per directory touched; nothing recursive (Lustre
    discipline).
    """
    if flat:
        return sorted(
            name for name in os.listdir(save_dir)
            if os.path.isfile(os.path.join(save_dir, name))
        )
    rels = []
    idx = "sample_indices_rank0.json"
    if os.path.isfile(os.path.join(save_dir, idx)):
        rels.append(idx)
    for sub in _DATA_SUBDIRS:
        d = os.path.join(save_dir, sub)
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                p = os.path.join(d, name)
                if os.path.isfile(p):
                    rels.append(os.path.join(sub, name))
    return sorted(rels)


def tree_digests(save_dir: str, flat: bool = False) -> dict:
    return {rel: sha256_file(os.path.join(save_dir, rel))
            for rel in list_rel_files(save_dir, flat=flat)}


def arm_digest(digests: dict) -> str:
    h = hashlib.sha256()
    for rel in sorted(digests):
        h.update(f"{rel}:{digests[rel]}\n".encode())
    return h.hexdigest()


def compare_trees(dir_a: str, dir_b: str, flat: bool = False) -> dict:
    da, db = tree_digests(dir_a, flat=flat), tree_digests(dir_b, flat=flat)
    only_a = sorted(set(da) - set(db))
    only_b = sorted(set(db) - set(da))
    common = sorted(set(da) & set(db))
    mismatched = [rel for rel in common if da[rel] != db[rel]]
    return {
        "pass": int(not only_a and not only_b and not mismatched),
        "n_files": len(common),
        "n_mismatch": len(mismatched),
        "files_only_a": only_a,
        "files_only_b": only_b,
        "mismatched_files": mismatched,
    }


def read_lines(path: str) -> list:
    with open(path, "r") as f:
        return f.read().splitlines()


def first_diff(lines_a: list, lines_b: list) -> int:
    """Index of the first differing row, or -1 if one is a prefix of the other
    and they otherwise agree (equal-length identical lists included)."""
    n = min(len(lines_a), len(lines_b))
    for i in range(n):
        if lines_a[i] != lines_b[i]:
            return i
    if len(lines_a) != len(lines_b):
        return n
    return -1


# --------------------------------------------------------------------------
# L2 book rows
# --------------------------------------------------------------------------

def parse_book_row(line: str) -> list:
    vals = [int(x) for x in line.split(",")]
    if len(vals) % 4 != 0 or not vals:
        raise ValueError(f"book row has {len(vals)} fields, expected a multiple of 4")
    return vals


def side_pairs(row: list, side: int) -> list:
    """(price, qty) pairs for one side. side=1 -> bids, side=0 -> asks.
    Row layout per level: [ask_p, ask_q, bid_p, bid_q]."""
    off = 2 if side == 1 else 0
    return [(row[4 * i + off], row[4 * i + off + 1]) for i in range(len(row) // 4)]


def valid_levels(pairs: list, side: int) -> list:
    # One predicate serves both sides: real prices are positive and below the
    # ask sentinel; the bid sentinel is negative so it fails 0 < p already.
    del side
    return [(p, q) for p, q in pairs if 0 < p < EMPTY_ASK_P]


def recompute_mid(ob_lines: list, upto_k: int, tick: int) -> tuple:
    """The tick-floored mid the generator carried into step ``upto_k``.

    Mirrors _get_new_mid_price: mid is refreshed from the post-step book each
    step and held when a side is empty, so the value entering step k is the
    one computed at the most recent row < k where both sides were valid.
    Returns (mid, source_row) or (None, None) when no gen row determines it
    (the pre-gen mid lives in the conditioning book, outside these CSVs).
    """
    for j in range(upto_k - 1, -1, -1):
        row = parse_book_row(ob_lines[j])
        ask_p, bid_p = row[0], row[2]
        if 0 < ask_p < EMPTY_ASK_P and 0 < bid_p < EMPTY_ASK_P:
            return (((ask_p + bid_p) // 2) // tick) * tick, j
    return None, None


def expected_pairs_after_add(prev_pairs: list, side: int, p_exp: int, qty: int) -> list:
    """The side's top-N (price, qty) pairs after adding a passive limit order,
    N = the L2 depth of the input row. Handles join / new level / improve-best
    uniformly, including the deepest level being pushed out of the window."""
    n = len(prev_pairs)
    book = {}
    for p, q in valid_levels(prev_pairs, side):
        book[p] = book.get(p, 0) + q
    book[p_exp] = book.get(p_exp, 0) + qty
    ordered = sorted(book.items(), key=lambda pq: -pq[0] if side == 1 else pq[0])
    ordered = ordered[:n]
    sentinel = (EMPTY_BID_P, 0) if side == 1 else (EMPTY_ASK_P, 0)
    while len(ordered) < n:
        ordered.append(sentinel)
    return ordered


def j4_window(ob_lines_i: list, k: int, side: int, qty: int,
              offset_ticks: int, tick: int, window_id: str) -> dict:
    out = {
        "window_id": window_id,
        "mid": None,
        "mid_source_row": None,
        "p_exp": None,
        "shape": None,
        "mid_determinate": 0,
        "precondition_uncrossed": 0,
        "precondition_passive": 0,
        "pass": 0,
    }
    if k < 1 or k >= len(ob_lines_i):
        out["shape"] = "k_out_of_range"
        return out
    mid, src = recompute_mid(ob_lines_i, k, tick)
    if mid is None:
        out["shape"] = "mid_indeterminate"
        return out
    out["mid_determinate"] = 1
    out["mid"] = mid
    out["mid_source_row"] = src
    p_exp = mid + offset_ticks * tick
    out["p_exp"] = p_exp

    prev = parse_book_row(ob_lines_i[k - 1])
    cur = parse_book_row(ob_lines_i[k])
    prev_ask_p1, prev_bid_p1 = prev[0], prev[2]
    if not (0 < prev_bid_p1 < prev_ask_p1 < EMPTY_ASK_P):
        out["shape"] = "book_crossed_or_empty"
        return out
    out["precondition_uncrossed"] = 1
    passive = p_exp < prev_ask_p1 if side == 1 else p_exp > prev_bid_p1
    if not passive:
        out["shape"] = "would_cross"
        return out
    out["precondition_passive"] = 1

    prev_same = side_pairs(prev, side)
    prev_valid = dict(valid_levels(prev_same, side))
    if p_exp in prev_valid:
        out["shape"] = "join"
    elif (side == 1 and p_exp > prev_bid_p1) or (side == 0 and p_exp < prev_ask_p1):
        out["shape"] = "improve_best"
    else:
        out["shape"] = "new_level"

    expected_same = expected_pairs_after_add(prev_same, side, p_exp, qty)
    actual_same = side_pairs(cur, side)
    other = 1 - side
    same_ok = expected_same == actual_same
    other_ok = side_pairs(cur, other) == side_pairs(prev, other)
    out["injected_side_matches"] = int(same_ok)
    out["other_side_unchanged"] = int(other_ok)
    out["pass"] = int(same_ok and other_ok)
    if not same_ok:
        out["expected_side"] = expected_same
        out["actual_side"] = actual_same
    return out


# --------------------------------------------------------------------------
# window discovery and the full verification
# --------------------------------------------------------------------------

def gen_windows(save_dir: str) -> dict:
    """window_id -> dict of the three data_gen files for that window."""
    d = os.path.join(save_dir, "data_gen")
    windows = {}
    for name in sorted(os.listdir(d)):
        m = _GEN_OB.match(name)
        if not m:
            continue
        wid = f"{m.group('date')}_{m.group('i')}"
        stem = name.replace("_orderbook_", "_{}_")
        windows[wid] = {
            "orderbook": os.path.join(d, name),
            "message": os.path.join(d, stem.format("message")),
            "provenance": os.path.join(d, stem.format("provenance")),
        }
    return windows


def _subtree_equal(dir_a: str, dir_b: str, sub: str) -> dict:
    out = compare_trees(os.path.join(dir_a, sub), os.path.join(dir_b, sub), flat=True)
    if out["n_files"] == 0:
        # An empty comparison must not read as agreement.
        out["pass"] = 0
    return out


def verify(dir_p: str, dir_o: str, dir_i: str, params: dict,
           dir_p2: str = None, extra_inputs: dict = None) -> dict:
    """Gate on J4 (mechanics, I arm alone); J0/J1/J2/J3/J5 are diagnostics
    computed only when their reference arms (``dir_p``/``dir_o``) are given."""
    k = params["inject_step"]
    side = params["inject_side"]
    qty = params["inject_qty"]
    offset = params["inject_price_offset_ticks"]
    tick = params.get("tick_size", DEFAULT_TICK)

    report = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params": params,
        "inputs": dict(extra_inputs or {}),
    }
    report["inputs"].update({
        "dir_p": dir_p, "dir_p2": dir_p2, "dir_o": dir_o, "dir_i": dir_i,
        "arm_digest_p": arm_digest(tree_digests(dir_p)) if dir_p else None,
        "arm_digest_p2": arm_digest(tree_digests(dir_p2)) if dir_p2 else None,
        "arm_digest_o": arm_digest(tree_digests(dir_o)) if dir_o else None,
        "arm_digest_i": arm_digest(tree_digests(dir_i)),
    })

    # J0 determinism diagnostic
    if dir_p and dir_p2:
        c = compare_trees(dir_p, dir_p2)
        report["j0_determinism"] = {
            "pass": c["pass"], "n_files": c["n_files"], "n_mismatch": c["n_mismatch"],
            "mismatched_files": c["mismatched_files"][:8],
        }
    else:
        report["j0_determinism"] = {"pass": 0, "n_files": 0, "n_mismatch": 0,
                                    "mismatched_files": [], "skipped": 1}

    # J1 off-means-off diagnostic
    if dir_p and dir_o:
        c = compare_trees(dir_p, dir_o)
        report["j1_off_means_off"] = {
            "pass": c["pass"], "n_files": c["n_files"], "n_mismatch": c["n_mismatch"],
            "mismatched_files": c["mismatched_files"][:8],
            "files_only_p": c["files_only_a"][:8], "files_only_o": c["files_only_b"][:8],
        }
    else:
        report["j1_off_means_off"] = {"pass": 0, "n_files": 0, "n_mismatch": 0,
                                      "mismatched_files": [], "files_only_p": [],
                                      "files_only_o": [], "skipped": 1}

    # J4, the gate: mechanics read off the I arm alone
    wins_i = gen_windows(dir_i)
    j4_windows = [
        j4_window(read_lines(wins_i[wid]["orderbook"]), k, side, qty, offset, tick, wid)
        for wid in sorted(wins_i)
    ]
    n_win = len(wins_i)
    n_j4 = sum(w["pass"] for w in j4_windows)
    precond_ok = all(
        w["mid_determinate"] and w["precondition_uncrossed"] and w["precondition_passive"]
        for w in j4_windows
    )
    report["j4_book_injection"] = {
        "pass": int(n_j4 == n_win and n_win > 0),
        "n_windows_pass": n_j4, "n_windows": n_win,
        "preconditions_pass": int(precond_ok),
        "per_window": j4_windows,
    }

    # J2/J3/J5 cross-arm diagnostics, only with an off arm to compare against
    if not dir_o:
        report["j2_prefix"] = {"pass": 0, "n_windows": 0, "skipped": 1}
        report["j3_msg_row_k"] = {"pass": 0, "n_windows": 0,
                                  "provenance_prefix_pass": 0, "skipped": 1}
        report["j5_divergence"] = {"pass": 0, "skipped": 1}
    else:
        wins_o = gen_windows(dir_o)
        if set(wins_o) != set(wins_i):
            report["j2_prefix"] = {"pass": 0, "n_windows": 0,
                                   "error": "window sets differ between O and I",
                                   "only_o": sorted(set(wins_o) - set(wins_i)),
                                   "only_i": sorted(set(wins_i) - set(wins_o))}
            report["j3_msg_row_k"] = {"pass": 0, "n_windows": 0,
                                      "provenance_prefix_pass": 0}
            report["j5_divergence"] = {"pass": 0}
        else:
            cond_eq = _subtree_equal(dir_o, dir_i, "data_cond")
            real_eq = _subtree_equal(dir_o, dir_i, "data_real")

            j2_ok, j3_ok, prov_ok = True, True, True
            fd_ob, fd_msg = [], []
            for wid in sorted(wins_o):
                ob_o, ob_i = read_lines(wins_o[wid]["orderbook"]), read_lines(wins_i[wid]["orderbook"])
                ms_o, ms_i = read_lines(wins_o[wid]["message"]), read_lines(wins_i[wid]["message"])
                pv_o, pv_i = read_lines(wins_o[wid]["provenance"]), read_lines(wins_i[wid]["provenance"])

                j2_ok &= (ob_o[:k] == ob_i[:k]) and (ms_o[:k] == ms_i[:k]) and (pv_o[:k] == pv_i[:k])
                j3_ok &= len(ms_o) > k and len(ms_i) > k and ms_o[k] == ms_i[k]
                prov_ok &= pv_o[:k + 1] == pv_i[:k + 1]

                fd_ob.append(first_diff(ob_o, ob_i))
                fd_msg.append(first_diff(ms_o, ms_i))

            report["j2_prefix"] = {
                "pass": int(j2_ok and cond_eq["pass"] == 1 and real_eq["pass"] == 1),
                "n_windows": n_win,
                "cond_tree_pass": cond_eq["pass"], "real_tree_pass": real_eq["pass"],
            }
            report["j3_msg_row_k"] = {
                "pass": int(j3_ok), "n_windows": n_win,
                "provenance_prefix_pass": int(prov_ok),
            }
            msg_prefix_ok = all(fd == -1 or fd > k for fd in fd_msg)
            report["j5_divergence"] = {
                "pass": int(n_win > 0 and all(fd == k for fd in fd_ob) and msg_prefix_ok),
                "orderbook_first_divergence_min": min(fd_ob) if fd_ob else None,
                "orderbook_first_divergence_max": max(fd_ob) if fd_ob else None,
                "msg_first_divergence_per_window": fd_msg,
                "msg_prefix_through_k_identical": int(msg_prefix_ok),
            }

    report["overall_pass"] = int(all((
        report["j4_book_injection"]["pass"] == 1,
        report["j4_book_injection"]["preconditions_pass"] == 1,
    )))
    return report


# --------------------------------------------------------------------------
# --check: re-audit an existing report on any machine, no data needed
# --------------------------------------------------------------------------

_REQUIRED_BLOCKS = (
    "j0_determinism", "j1_off_means_off", "j2_prefix",
    "j3_msg_row_k", "j4_book_injection", "j5_divergence",
)


def check_report(path: str) -> list:
    """Internal-consistency audit of a written report. Returns problem list."""
    problems = []
    with open(path) as f:
        rep = json.load(f)
    if rep.get("schema") != SCHEMA:
        problems.append(f"schema is {rep.get('schema')!r}, expected {SCHEMA!r}")
        return problems
    for key in _REQUIRED_BLOCKS + ("params", "inputs", "overall_pass"):
        if key not in rep:
            problems.append(f"missing block: {key}")
    if problems:
        return problems
    # Only J4 gates (PR #21 review: GPU byte-determinism must not gate).
    should_pass = all((
        rep["j4_book_injection"].get("pass") == 1,
        rep["j4_book_injection"].get("preconditions_pass") == 1,
    ))
    if int(should_pass) != rep["overall_pass"]:
        problems.append(
            f"overall_pass={rep['overall_pass']} inconsistent with blocks ({int(should_pass)})")
    j4 = rep["j4_book_injection"]
    if j4.get("n_windows_pass") != sum(w.get("pass", 0) for w in j4.get("per_window", [])):
        problems.append("j4 n_windows_pass disagrees with per_window")
    k = rep["params"].get("inject_step")
    j5 = rep["j5_divergence"]
    if j5.get("pass") == 1 and (j5.get("orderbook_first_divergence_min") != k
                                or j5.get("orderbook_first_divergence_max") != k):
        problems.append("j5 pass=1 but orderbook first divergence is not the injected step")
    v = rep["inputs"].get("arm_digest_i")
    if not (isinstance(v, str) and len(v) == 64):
        problems.append("inputs.arm_digest_i is not a sha256 hex digest")
    for arm in ("arm_digest_p", "arm_digest_p2", "arm_digest_o"):
        v = rep["inputs"].get(arm)
        if v is not None and not (isinstance(v, str) and len(v) == 64):
            problems.append(f"inputs.{arm} is neither null nor a sha256 hex digest")
    if rep["j1_off_means_off"].get("pass") == 1 and rep["inputs"].get("arm_digest_p") != \
            rep["inputs"].get("arm_digest_o"):
        problems.append("j1 pass=1 but arm digests of P and O differ")
    if not rep["inputs"].get("node"):
        problems.append("inputs.node missing (byte-identity diagnostics are node-bound, D-X7)")
    return problems


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", metavar="REPORT_JSON",
                    help="audit an existing report for internal consistency and exit")
    ap.add_argument("--dir-p", help="save_dir of the pre-PR arm (optional, diagnostics)")
    ap.add_argument("--dir-p2", help="save_dir of the pre-PR determinism rerun (optional, diagnostics)")
    ap.add_argument("--dir-o", help="save_dir of the PR arm, injection off (optional, diagnostics)")
    ap.add_argument("--dir-i", help="save_dir of the PR arm, injection on (the gated arm)")
    ap.add_argument("--inject-step", type=int)
    ap.add_argument("--inject-event-type", type=int, default=1)
    ap.add_argument("--inject-side", type=int, default=1, choices=[0, 1])
    ap.add_argument("--inject-qty", type=int)
    ap.add_argument("--inject-offset-ticks", type=int, default=-1)
    ap.add_argument("--tick-size", type=int, default=DEFAULT_TICK)
    ap.add_argument("--n-cond-msgs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--node", default=None)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--ckpt-pin", default=None)
    ap.add_argument("--ckpt-metadata-sha256", default=None)
    ap.add_argument("--checkpoint-step", type=int, default=None)
    ap.add_argument("--data-day", default=None)
    ap.add_argument("--commit-p", default=None)
    ap.add_argument("--commit-oi", default=None)
    ap.add_argument("--out", default=None, help="write the report JSON here")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.check:
        problems = check_report(args.check)
        if problems:
            for p in problems:
                print(f"[check] FAIL {p}")
            return 2
        with open(args.check) as f:
            rep = json.load(f)
        print(f"[check] ok schema={rep['schema']} overall_pass={rep['overall_pass']}")
        return 0 if rep["overall_pass"] == 1 else 2

    required = ("dir_i", "inject_step", "inject_qty")
    missing = [r for r in required if getattr(args, r) is None]
    if missing:
        print(f"missing required arguments: {missing}", file=sys.stderr)
        return 4
    if args.inject_event_type != 1:
        print("J4 verifies a passive new limit order; only --inject-event-type 1 "
              "is supported by this checker", file=sys.stderr)
        return 4

    params = {
        "inject_step": args.inject_step,
        "inject_event_type": args.inject_event_type,
        "inject_side": args.inject_side,
        "inject_qty": args.inject_qty,
        "inject_price_offset_ticks": args.inject_offset_ticks,
        "tick_size": args.tick_size,
        "n_cond_msgs": args.n_cond_msgs,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    extra = {
        "node": args.node, "gpu": args.gpu,
        "ckpt_pin": args.ckpt_pin, "ckpt_metadata_sha256": args.ckpt_metadata_sha256,
        "checkpoint_step": args.checkpoint_step, "data_day": args.data_day,
        "commit_p": args.commit_p, "commit_oi": args.commit_oi,
    }
    report = verify(args.dir_p, args.dir_o, args.dir_i, params,
                    dir_p2=args.dir_p2, extra_inputs=extra)

    n_gen = None
    try:
        wins = gen_windows(args.dir_i)
        if wins:
            first = wins[sorted(wins)[0]]
            n_gen = len(read_lines(first["orderbook"]))
    except OSError:
        pass
    report["params"]["n_gen_msgs"] = n_gen
    report["params"]["n_windows"] = report["j4_book_injection"]["n_windows"]

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, args.out)
        print(f"[verify_injection] wrote {args.out}")
    else:
        print(text, end="")

    for name in ("j0_determinism", "j1_off_means_off", "j2_prefix",
                 "j3_msg_row_k", "j4_book_injection", "j5_divergence"):
        block = report[name]
        if block.get("skipped"):
            state = "SKIPPED"
        else:
            state = "PASS" if block["pass"] else "FAIL"
        tag = "" if name == "j4_book_injection" else " (diagnostic)"
        print(f"[verify_injection] {name}: {state}{tag}")
    print(f"[verify_injection] overall_pass={report['overall_pass']} (gate: J4 only)")
    return 0 if report["overall_pass"] == 1 else 2


if __name__ == "__main__":
    sys.exit(main())

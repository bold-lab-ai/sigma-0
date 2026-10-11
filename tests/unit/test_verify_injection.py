"""Unit tests for lobpipeline.injection_verification.

Everything runs on synthetic CSV fixtures built in-test: no jax, no GPU, no
market data. The end-to-end case builds four arm directories the way
run/base_model/runtime/inference.py lays them out, verifies the green path,
then breaks each judgement individually and asserts exactly that judgement
turns red.
"""
import json

import pytest

from lobpipeline.injection_verification import (
    EMPTY_ASK_P,
    EMPTY_BID_P,
    check_report,
    compare_trees,
    expected_pairs_after_add,
    first_diff,
    recompute_mid,
    side_pairs,
    verify,
)

TICK = 100
K = 4
QTY = 777


# --------------------------------------------------------------------------
# row builders
# --------------------------------------------------------------------------

def mk_row(asks, bids):
    """40-column L2 row from 10 (p, q) pairs per side."""
    assert len(asks) == len(bids) == 10
    row = []
    for (ap, aq), (bp, bq) in zip(asks, bids):
        row += [ap, aq, bp, bq]
    return ",".join(str(v) for v in row)


def ladder(start, step, qty0, n=10, direction=-1):
    return [(start + direction * i * step, qty0 + i) for i in range(n)]


ASKS = ladder(100300, TICK, 50, direction=+1)
BIDS = ladder(100100, TICK, 40, direction=-1)
BASE_ROW = mk_row(ASKS, BIDS)
# mid of BASE_ROW: ((100300 + 100100)//2 // 100) * 100 = 100200; offset -1 tick
# puts the injected order exactly on the best bid => a "join".
P_EXP = 100100


def bump_ask_qty(row_line, delta):
    vals = [int(x) for x in row_line.split(",")]
    vals[1] += delta
    return ",".join(str(v) for v in vals)


def injected_row(prev_line, p_exp=P_EXP, qty=QTY):
    prev = [int(x) for x in prev_line.split(",")]
    bids = expected_pairs_after_add(side_pairs(prev, 1), 1, p_exp, qty)
    asks = side_pairs(prev, 0)
    return mk_row(asks, bids)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

def test_recompute_mid_basic():
    mid, src = recompute_mid([BASE_ROW, BASE_ROW], upto_k=2, tick=TICK)
    assert mid == 100200 and src == 1


def test_recompute_mid_falls_back_over_empty_side():
    empty_ask = mk_row([(EMPTY_ASK_P, 0)] * 10, BIDS)
    mid, src = recompute_mid([BASE_ROW, empty_ask], upto_k=2, tick=TICK)
    assert mid == 100200 and src == 0


def test_recompute_mid_indeterminate():
    empty = mk_row([(EMPTY_ASK_P, 0)] * 10, [(EMPTY_BID_P, 0)] * 10)
    assert recompute_mid([empty, empty], upto_k=2, tick=TICK) == (None, None)


def test_expected_pairs_join():
    out = expected_pairs_after_add(list(BIDS), 1, 100100, QTY)
    assert out[0] == (100100, 40 + QTY)
    assert out[1:] == BIDS[1:]


def test_expected_pairs_new_level_pushes_deepest_out():
    out = expected_pairs_after_add(list(BIDS), 1, 100050, QTY)
    assert out[0] == BIDS[0]
    assert out[1] == (100050, QTY)
    assert out[2:] == BIDS[1:9]  # old deepest level fell out of the window


def test_expected_pairs_improve_best():
    out = expected_pairs_after_add(list(BIDS), 1, 100200, QTY)
    assert out[0] == (100200, QTY)
    assert out[1:] == BIDS[:9]


def test_expected_pairs_pads_sentinels():
    sparse = BIDS[:3] + [(EMPTY_BID_P, 0)] * 7
    out = expected_pairs_after_add(sparse, 1, 100050, QTY)
    assert (100050, QTY) in out
    assert out[-6:] == [(EMPTY_BID_P, 0)] * 6


def test_first_diff():
    assert first_diff(["a", "b"], ["a", "b"]) == -1
    assert first_diff(["a", "b"], ["a", "c"]) == 1
    assert first_diff(["a"], ["a", "b"]) == 1  # proper prefix: tail differs


# --------------------------------------------------------------------------
# end-to-end on synthetic arm directories
# --------------------------------------------------------------------------

WINDOW_IDS = ("5", "9")
DATE = "2022-12-30"
N_GEN = 12


def gen_lines_off():
    ob = []
    prev = BASE_ROW
    for j in range(N_GEN):
        prev = bump_ask_qty(BASE_ROW, j)  # deterministic drift, all rows distinct
        ob.append(prev)
    msg = [f"34200.{j:09d},1,{1000 + j},10,100200,1" for j in range(N_GEN)]
    prov = [str(j % 4) for j in range(N_GEN)]
    return ob, msg, prov


def gen_lines_on():
    """The on arm: identical through row K-1, injected book at row K, the
    model's own message row K unchanged, divergence strictly after K."""
    ob_o, msg_o, prov_o = gen_lines_off()
    ob = list(ob_o)
    ob[K] = injected_row(ob_o[K - 1])
    for j in range(K + 1, N_GEN):
        ob[j] = bump_ask_qty(ob_o[j], 500)
    msg = list(msg_o)
    for j in range(K + 1, N_GEN):
        msg[j] = msg_o[j].replace(",10,", ",11,")
    prov = list(prov_o)
    for j in range(K + 1, N_GEN):
        prov[j] = str((int(prov_o[j]) + 1) % 4)
    return ob, msg, prov


def write_arm(root, ob, msg, prov):
    (root / "data_cond").mkdir(parents=True)
    (root / "data_real").mkdir()
    (root / "data_gen").mkdir()
    (root / "sample_indices_rank0.json").write_text(
        json.dumps({"selection_seed": 42, "all_indices": [5, 9]}) + "\n")
    for i in WINDOW_IDS:
        for sub, kind in (("data_cond", "message"), ("data_cond", "orderbook"),
                          ("data_real", "message"), ("data_real", "orderbook")):
            (root / sub / f"GOOG_{DATE}_{kind}_real_id_{i}.csv").write_text(
                f"shared,{sub},{kind},{i}\n")
        stem = f"GOOG_{DATE}_{{}}_real_id_{i}_gen_id_0.csv"
        (root / "data_gen" / stem.format("orderbook")).write_text("\n".join(ob) + "\n")
        (root / "data_gen" / stem.format("message")).write_text("\n".join(msg) + "\n")
        (root / "data_gen" / stem.format("provenance")).write_text("\n".join(prov) + "\n")


PARAMS = {
    "inject_step": K, "inject_event_type": 1, "inject_side": 1,
    "inject_qty": QTY, "inject_price_offset_ticks": -1, "tick_size": TICK,
}
INPUTS = {"node": "testnode", "gpu": "0"}


@pytest.fixture
def arms(tmp_path):
    off = gen_lines_off()
    on = gen_lines_on()
    dirs = {}
    for name, lines in (("p", off), ("p2", off), ("o", off), ("i", on)):
        root = tmp_path / f"arm_{name}"
        write_arm(root, *lines)
        dirs[name] = str(root)
    return dirs


def run_verify(dirs):
    return verify(dirs["p"], dirs["o"], dirs["i"], dict(PARAMS),
                  dir_p2=dirs["p2"], extra_inputs=dict(INPUTS))


def test_green_path(arms, tmp_path):
    rep = run_verify(arms)
    assert rep["j0_determinism"]["pass"] == 1
    assert rep["j1_off_means_off"]["pass"] == 1
    assert rep["j2_prefix"]["pass"] == 1
    assert rep["j3_msg_row_k"]["pass"] == 1
    assert rep["j3_msg_row_k"]["provenance_prefix_pass"] == 1
    assert rep["j4_book_injection"]["pass"] == 1
    assert all(w["shape"] == "join" for w in rep["j4_book_injection"]["per_window"])
    assert rep["j5_divergence"]["orderbook_first_divergence_min"] == K
    assert rep["j5_divergence"]["orderbook_first_divergence_max"] == K
    assert rep["j5_divergence"]["pass"] == 1
    assert rep["overall_pass"] == 1

    out = tmp_path / "report.json"
    out.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    assert check_report(str(out)) == []


def test_missing_p2_skips_j0_diagnostic(arms):
    rep = verify(arms["p"], arms["o"], arms["i"], dict(PARAMS),
                 dir_p2=None, extra_inputs=dict(INPUTS))
    assert rep["j0_determinism"]["skipped"] == 1
    assert rep["overall_pass"] == 1


def test_i_arm_alone_gates_on_mechanics(arms, tmp_path):
    """The minimal re-record: one injection-on rollout, no reference arms."""
    rep = verify(None, None, arms["i"], dict(PARAMS), extra_inputs=dict(INPUTS))
    for diag in ("j0_determinism", "j1_off_means_off", "j2_prefix",
                 "j3_msg_row_k", "j5_divergence"):
        assert rep[diag]["skipped"] == 1
    assert rep["inputs"]["arm_digest_p"] is None
    assert rep["j4_book_injection"]["pass"] == 1
    assert rep["overall_pass"] == 1

    out = tmp_path / "report_i_only.json"
    out.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    assert check_report(str(out)) == []


def test_off_arm_drift_is_diagnostic_not_gating(arms, tmp_path):
    victim = (tmp_path / "arm_o" / "data_gen"
              / f"GOOG_{DATE}_provenance_real_id_9_gen_id_0.csv")
    victim.write_text(victim.read_text() + "3\n")
    rep = run_verify(arms)
    assert rep["j1_off_means_off"]["pass"] == 0
    assert rep["overall_pass"] == 1


def test_discarded_msg_change_is_diagnostic_not_gating(arms, tmp_path):
    f = (tmp_path / "arm_i" / "data_gen"
         / f"GOOG_{DATE}_message_real_id_5_gen_id_0.csv")
    lines = f.read_text().splitlines()
    lines[K] = lines[K].replace(",1,", ",3,", 1)
    f.write_text("\n".join(lines) + "\n")
    rep = run_verify(arms)
    assert rep["j3_msg_row_k"]["pass"] == 0
    assert rep["overall_pass"] == 1


def test_wrong_quantity_fails_j4(arms, tmp_path):
    off_ob, _, _ = gen_lines_off()
    bad = injected_row(off_ob[K - 1], qty=QTY - 1)
    for i in WINDOW_IDS:
        f = (tmp_path / "arm_i" / "data_gen"
             / f"GOOG_{DATE}_orderbook_real_id_{i}_gen_id_0.csv")
        lines = f.read_text().splitlines()
        lines[K] = bad
        f.write_text("\n".join(lines) + "\n")
    rep = run_verify(arms)
    assert rep["j4_book_injection"]["pass"] == 0
    assert rep["overall_pass"] == 0


def test_early_divergence_is_diagnostic_not_gating(arms, tmp_path):
    for i in WINDOW_IDS:
        f = (tmp_path / "arm_i" / "data_gen"
             / f"GOOG_{DATE}_orderbook_real_id_{i}_gen_id_0.csv")
        lines = f.read_text().splitlines()
        lines[K - 2] = bump_ask_qty(lines[K - 2], 999)
        f.write_text("\n".join(lines) + "\n")
    rep = run_verify(arms)
    assert rep["j2_prefix"]["pass"] == 0
    assert rep["j5_divergence"]["pass"] == 0
    assert rep["overall_pass"] == 1


def test_check_report_catches_inconsistent_overall(arms, tmp_path):
    rep = run_verify(arms)
    rep["overall_pass"] = 0  # blocks all green but overall forced red
    out = tmp_path / "tampered.json"
    out.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    problems = check_report(str(out))
    assert any("inconsistent" in p for p in problems)


def test_compare_trees_detects_asymmetry(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "x.csv").write_text("1\n")
    (b / "x.csv").write_text("1\n")
    (b / "y.csv").write_text("2\n")
    out = compare_trees(str(a), str(b), flat=True)
    assert out["pass"] == 0 and out["files_only_b"] == ["y.csv"]

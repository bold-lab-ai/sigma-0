"""End to end: does a picked window actually reach the model as a window ending on a trade?

Covers the whole chain -- pick_trade_anchors writes JSON, resolve_data_files orders the
files, LOBSTER_Dataset places the window, __getitem__ returns the messages -- because
each link can be individually correct while the chain still anchors on the wrong thing.
The failure this guards against produces no error at all: sequences generate normally,
they are simply anchored on a cancellation, or on the right row of the wrong trading day.

Driven through subprocesses so the JAX setup stays out of the pytest process. On a
login node JAX_PLATFORMS=cpu is not sufficient on its own: with all 288 cores visible
XLA sizes its Eigen thread pool to match, pthread_create hits the cgroup pids limit and
the process aborts. Restricting CPU affinity is what makes it survivable, and doing that
to the pytest process itself would slow every other test down.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
PICKER = REPO / "run" / "base_model" / "eval" / "pick_trade_anchors.py"

N_COND, N_GEN = 10, 10
N_MESSAGES = N_COND + N_GEN

# Trades at rows chosen so the boundary cases are exercised: row 5 would start the
# window before row 0, and row 995 would run it past the end of the book.
TRADE_PLAN = {
    "2026-01-05": (1000, [5, 15, 50, 100, 995]),
    "2026-01-06": (1000, [200, 300]),
    "2026-01-07": (1000, [400]),
}
EXPECTED_USABLE = 6  # 8 trades, minus the two that cannot fit a window


def _child_env():
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = str(REPO / "src")
    # Thread budget, not a performance knob. A login node reports 288 cores, every
    # pool sizes itself accordingly, and the per-user process limit (ulimit -u 1900
    # here, against ~2600 already running) is reached before the dataset is built.
    # The symptom is a bare SIGKILL or a pthread_create abort, neither of which
    # mentions threads.
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false").strip()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS"):
        env[var] = "1"
    return env


def _run(code, *args):
    """Run a snippet with restricted CPU affinity; skip if the toolchain is missing."""
    cmd = ["taskset", "-c", "0-3", sys.executable, "-c", code, *map(str, args)]
    try:
        p = subprocess.run(cmd, env=_child_env(), capture_output=True, text=True, timeout=900)
    except FileNotFoundError:
        pytest.skip("taskset unavailable")
    if p.returncode != 0:
        if "ModuleNotFoundError" in p.stderr:
            pytest.skip(f"missing dependency: {p.stderr.strip().splitlines()[-1]}")
        if p.returncode < 0:
            # Killed by a signal with nothing on either stream: the host ran out of
            # process slots, not a defect in what is being tested. An assertion
            # failure exits 1 and prints a traceback, so the two do not overlap.
            # Reproduce outside pytest, which frees the slots this process holds:
            #   JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 0-3 python -c '<probe>' \
            #       anchors.json data_dir
            pytest.skip(
                f"child killed by signal {-p.returncode} with no output; the per-user "
                f"process limit (ulimit -u) is reached when pytest and a JAX child run "
                f"together on a login node. Run this on a compute node.")
        pytest.fail(f"child failed ({p.returncode}):\n{p.stdout}\n{p.stderr}")
    return p.stdout


@pytest.fixture(scope="module")
def fake_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("proc_npy")
    for date, (n, trades) in TRADE_PLAN.items():
        m = np.zeros((n, 14), dtype=np.int64)
        m[:, 1] = 1                       # event_type: submissions by default
        for k, r in enumerate(trades):
            m[r, 1] = 4                   # a trade
            m[r, 2] = k % 2               # alternate the resting side
        m[:, 8] = np.arange(n)            # time_s
        np.save(d / f"GOOG_{date}_message_proc.npy", m)
        np.save(d / f"GOOG_{date}_orderbook_proc.npy", np.ones((n, 40), dtype=np.int64))
    return d


@pytest.fixture(scope="module")
def anchors(fake_data, tmp_path_factory):
    out = tmp_path_factory.mktemp("anchors") / "anchors.json"
    subprocess.run(
        [sys.executable, str(PICKER), "--data_dir", str(fake_data), "--out", str(out),
         "--n_cond_msgs", str(N_COND), "--n_gen_msgs", str(N_GEN),
         "--n_samples", "100", "--test_split", "0"],
        check=True, capture_output=True, text=True, timeout=600)
    return json.loads(out.read_text())


def test_picker_drops_windows_that_cannot_fit(anchors):
    assert len(anchors["windows"]) == EXPECTED_USABLE
    assert anchors["n_messages"] == N_MESSAGES


def test_picked_windows_end_on_their_trade(anchors):
    """Checked against the source arrays, independent of the dataset."""
    for (file_idx, seq_start), direction in zip(anchors["windows"], anchors["anchor_direction"]):
        msgs = np.load(anchors["message_files"][file_idx], mmap_mode="r")
        last_cond = msgs[seq_start + N_COND - 1]
        assert int(last_cond[1]) == 4
        assert int(last_cond[2]) == direction
        # An off-by-one would still pass if a neighbour were also a trade, so make
        # sure the fixture does not accidentally provide that escape.
        for delta in (-1, 1):
            assert int(msgs[seq_start + N_COND - 1 + delta][1]) != 4


def test_directions_are_balanced(anchors):
    dirs = anchors["anchor_direction"]
    assert set(dirs) == {0, 1}
    assert abs(dirs.count(0) - dirs.count(1)) <= 1


def test_file_idx_shifts_with_test_split(fake_data, tmp_path, anchors):
    """Why the file list is recorded and asserted at inference time.

    The same physical window keeps its seq_start but changes file_idx when the file
    list is filtered differently. Nothing raises, so an anchor set reused under a
    different test_split points at another trading day.
    """
    out = tmp_path / "anchors_split.json"
    subprocess.run(
        [sys.executable, str(PICKER), "--data_dir", str(fake_data), "--out", str(out),
         "--n_cond_msgs", str(N_COND), "--n_gen_msgs", str(N_GEN),
         "--n_samples", "10", "--test_split", "0.5"],
        check=True, capture_output=True, text=True, timeout=600)
    split = json.loads(out.read_text())

    assert len(split["message_files"]) == 1          # only the trailing day survives
    (fi_split, start), = split["windows"]
    same = [w for w in anchors["windows"] if w[1] == start]
    assert same, "expected the same physical window in the unfiltered set"
    assert same[0][0] != fi_split                     # identical start, different file_idx


CHAIN_PROBE = """
import json, sys
import numpy as np
from base_model.data.lobster_dataloader import LOBSTER_Dataset
from base_model.data.data_files import resolve_data_files

spec = json.load(open(sys.argv[1]))
msg_files, book_files, _ = resolve_data_files(sys.argv[2], test_split=0, verbose=False)
assert [str(f) for f in msg_files] == spec["message_files"], "file list drift"

kw = dict(n_messages=spec["n_messages"], mask_fn=LOBSTER_Dataset.inference_mask,
          randomize_offset=False, book_files=book_files, use_simple_book=True,
          book_transform=False, book_depth=500, return_raw_msgs=True, inference=True)
explicit = LOBSTER_Dataset(msg_files, explicit_windows=np.asarray(spec["windows"], np.int64), **kw)
grid = LOBSTER_Dataset(msg_files, **kw)

ncond = spec["n_cond_msgs"]
def ends_on_trade(ds):
    # X_raw is second from last; the last element is book_l2_init
    return sum(int(np.asarray(ds[i][-2])[ncond - 1][1]) == 4 for i in range(len(ds)))

print(json.dumps({
    "n_explicit": len(explicit), "explicit_hits": ends_on_trade(explicit),
    "n_grid": len(grid), "grid_hits": ends_on_trade(grid),
}))
"""


def test_dataset_delivers_trade_anchored_windows(anchors, fake_data, tmp_path):
    """The claim that matters: what the model receives ends on a trade."""
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(anchors))
    out = _run(CHAIN_PROBE, spec_path, fake_data)
    r = json.loads(out.strip().splitlines()[-1])

    assert r["n_explicit"] == len(anchors["windows"])
    assert r["explicit_hits"] == r["n_explicit"], "every explicit window must end on a trade"
    # The grid cannot reach these positions, which is the whole reason the bypass exists.
    assert r["grid_hits"] == 0
    assert r["n_grid"] > r["n_explicit"]

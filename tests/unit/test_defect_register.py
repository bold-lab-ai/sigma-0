"""The defect register has to satisfy its own definition.

A problem statement is only standardised if the standard is enforced. These
tests are the enforcement: an entry that cannot name what you would measure if
it fired is not a defined problem, it is a suspicion, and it does not belong in
the register. The same goes for entries whose cited line numbers have rotted
away, and for a register that has drifted out of sync with the prose document
that explains it.

Stdlib only, no engine import, so this runs anywhere including CI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / "docs" / "defect_register.json"
DOCUMENT = ROOT / "docs" / "fidelity_defect_taxonomy.md"

REQUIRED = ("id", "class", "statement", "sites", "trigger",
            "observable", "silent", "evidence", "status")

# A site that points at code, as opposed to "missing" or "record schema".
CODE_SITE = re.compile(r"^(?P<path>[\w./-]+\.(?:py|sbatch|sh)):(?P<lo>\d+)(?:-(?P<hi>\d+))?$")


@pytest.fixture(scope="module")
def register():
    return json.loads(REGISTER.read_text())


@pytest.fixture(scope="module")
def defects(register):
    return register["defects"]


def test_every_entry_carries_every_required_field(defects):
    for d in defects:
        missing = [f for f in REQUIRED if f not in d]
        assert not missing, f"{d.get('id', '?')} missing {missing}"


def test_ids_are_unique_and_well_formed(register, defects):
    ids = [d["id"] for d in defects]
    assert len(ids) == len(set(ids)), "duplicate defect id"
    classes = set(register["classes"])
    for i in ids:
        m = re.fullmatch(r"D-([A-Z])(\d+)", i)
        assert m, f"malformed id {i}"
        assert m.group(1) in classes, f"{i} names an undeclared class"


def test_class_field_agrees_with_the_id(defects):
    for d in defects:
        assert d["id"] == f"D-{d['class']}{d['id'][3:]}", d["id"]


def test_status_is_one_of_the_declared_values(register, defects):
    allowed = set(register["statuses"])
    for d in defects:
        assert d["status"] in allowed, f"{d['id']} has status {d['status']}"


def test_every_live_defect_names_an_observable(defects):
    """The load-bearing rule. A defect with no observable cannot be confirmed,
    cannot be closed, and cannot be regression-tested. Refuted entries are
    exempt because they are kept only to stop the hypothesis being re-opened."""
    for d in defects:
        if d["status"] == "REFUTED":
            continue
        assert d["observable"].strip(), f"{d['id']} has no observable"
        assert len(d["observable"]) > 20, f"{d['id']} observable is too vague: {d['observable']!r}"


def test_every_entry_states_whether_it_is_silent(defects):
    """Silent defects need an active probe; loud ones do not. Getting this
    wrong is how a defect ends up with no detector at all."""
    for d in defects:
        assert isinstance(d["silent"], bool), f"{d['id']} silent must be a bool"


def test_evidence_is_present_and_labelled(defects):
    """Either a measurement or an honest 'static' / 'unmeasured'. What is not
    allowed is an empty field, which reads as evidence without being any."""
    for d in defects:
        assert d["evidence"].strip(), f"{d['id']} has no evidence field"


def test_cited_code_sites_still_exist(defects):
    """Line-number rot is the normal fate of a defect list. This turns it into
    a test failure rather than a silent loss of meaning."""
    for d in defects:
        for site in d["sites"]:
            m = CODE_SITE.match(site)
            if not m:
                continue
            path = ROOT / m.group("path")
            assert path.is_file(), f"{d['id']} cites missing file {site}"
            n = len(path.read_text().splitlines())
            hi = int(m.group("hi") or m.group("lo"))
            assert hi <= n, f"{d['id']} cites {site} but the file has {n} lines"


def test_the_document_and_the_register_list_the_same_defects(defects):
    """Prose and data drift apart the moment they are allowed to. They are not."""
    in_doc = set(re.findall(r"D-[A-Z]\d+", DOCUMENT.read_text()))
    in_reg = {d["id"] for d in defects}
    assert in_doc == in_reg, (
        f"only in document: {sorted(in_doc - in_reg)}; "
        f"only in register: {sorted(in_reg - in_doc)}"
    )


def test_the_rendered_register_section_is_current():
    """Stronger than matching id sets: the section is rebuilt from the JSON and
    compared byte for byte. The register existed twice, as JSON and as a
    hand-written table, which is two sources of truth for one set of facts and
    therefore a guarantee they disagree eventually. Now the prose is written
    and the tables are generated, and this is what stops someone editing the
    generated half."""
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "render_register.py"), "--check"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr or r.stdout


def test_every_declared_class_is_actually_used(register, defects):
    """An unused class means the taxonomy claims coverage it does not exercise."""
    used = {d["class"] for d in defects}
    declared = set(register["classes"])
    assert used == declared, f"declared but unused: {sorted(declared - used)}"


def test_bound_entries_are_confined_to_the_observation_class(defects):
    """BOUND means 'not fixable, it is an information bound'. Only the
    observation class can make that claim; anywhere else it is an excuse."""
    for d in defects:
        if d["status"] == "BOUND":
            assert d["class"] == "O", f"{d['id']} claims BOUND outside class O"


def test_the_register_evidence_is_current():
    """CI audits a recorded JSON rather than re-running anything, so the
    recording has to be current. This catches the register moving without the
    evidence being re-recorded, which would let a PR claim a defect count that
    is no longer true."""
    import json as _json

    ev = ROOT / "ci" / "evidence" / "register.json"
    if not ev.is_file():
        pytest.skip("register evidence has not been recorded yet")
    doc = _json.loads(ev.read_text())
    reg = _json.loads(REGISTER.read_text())
    assert doc["metrics"]["register"]["total"] == len(reg["defects"]), (
        "ci/evidence/register.json is stale; re-record with "
        "python ci/record_evidence.py --register"
    )
    live = sorted(d["id"] for d in reg["defects"] if d["status"] == "OPEN")
    assert doc["metrics"]["register"]["open_ids"] == live, (
        f"recorded open set {doc['metrics']['register']['open_ids']} != actual {live}"
    )

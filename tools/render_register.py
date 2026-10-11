#!/usr/bin/env python3
"""Render the register section of the taxonomy document from the JSON.

The register used to exist twice, once as JSON and once as a hand-written
markdown table, which is two sources of truth for the same data and therefore a
guarantee that they will disagree eventually. The prose stays hand-written
because it is an argument; the tables are data and are generated.

    tools/render_register.py            rewrite the section in place
    tools/render_register.py --check    exit 1 if the file is out of date

The --check mode is what the test uses, so drift is a test failure rather than
something a reader notices months later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "docs" / "defect_register.json"
DOCUMENT = ROOT / "docs" / "fidelity_defect_taxonomy.md"
BEGIN = "<!-- REGISTER:BEGIN -->"
END = "<!-- REGISTER:END -->"

CLASS_TITLE = {
    "X": "Class X, Execution: the artifact and the code came apart",
    "I": "Class I, Init: the rebuilt initial state is not isomorphic",
    "R": "Class R, Record: what was written is not what was applied",
    "T": "Class T, Transition: the two sides run different engine configs",
    "O": "Class O, Observation: information bounds, declared not fixed",
    "V": "Class V, Validation: the diagnostic itself fails",
}
ORDER = ["X", "I", "R", "T", "O", "V"]
BADGE = {"OPEN": "OPEN", "FIXED": "**FIXED**", "BOUND": "**BOUND**", "REFUTED": "REFUTED"}


def _cell(text: str) -> str:
    """Markdown table cells cannot hold a raw pipe or a newline."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def render(defects: List[Dict]) -> str:
    out: List[str] = []
    for cls in ORDER:
        rows = [d for d in defects if d["class"] == cls]
        if not rows:
            continue
        done = sum(1 for d in rows if d["status"] in ("FIXED", "BOUND"))
        out += [
            f"### {CLASS_TITLE[cls]}",
            "",
            f"{done} of {len(rows)} closed.",
            "",
            "| ID | statement | sites | trigger | observable | silent | evidence | status |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for d in rows:
            sites = "; ".join(f"`{s}`" for s in d["sites"])
            out.append(
                f"| **{d['id']}** | {_cell(d['statement'])} | {_cell(sites)} | "
                f"{_cell(d['trigger'])} | {_cell(d['observable'])} | "
                f"{'**yes**' if d['silent'] else 'no'} | {_cell(d['evidence'])} | "
                f"{BADGE[d['status']]} |"
            )
        out.append("")
    return "\n".join(out)


def splice(text: str, body: str) -> str:
    i, j = text.index(BEGIN), text.index(END)
    return text[: i + len(BEGIN)] + "\n\n" + body + "\n" + text[j:]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    defects = json.loads(REGISTER.read_text())["defects"]
    current = DOCUMENT.read_text()
    updated = splice(current, render(defects))

    if args.check:
        if current != updated:
            print("docs/fidelity_defect_taxonomy.md is out of date with "
                  "docs/defect_register.json; run tools/render_register.py",
                  file=sys.stderr)
            return 1
        print("register section is up to date")
        return 0

    DOCUMENT.write_text(updated)
    print(f"rendered {len(defects)} defects into {DOCUMENT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

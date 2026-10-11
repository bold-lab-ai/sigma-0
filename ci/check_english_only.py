"""Fail if tracked text carries CJK characters. Standard library only.

Why
---
Documents and comments here get read by people who do not read Chinese, and by tools
that do not either. A file that is half Chinese is not half as useful; it is a file
whose reader has to guess which half matters. The repository already settled on English
for docs/conventions.md, docs/checkpoints.md and the rest -- this makes that the rule
rather than the habit.

The failure this prevents is drift, not ignorance. Nobody adds a Chinese paragraph on
purpose; it arrives as one heading in an otherwise English file, or one word inside an
English comment (src/post_training/heuristic_learning/mm_sim.py had exactly that: one
ideograph wedged between "length" and "step"). One word is invisible in review and
permanent afterwards.

Binary files are skipped, and that matters. Bytes in a PNG land inside the CJK ranges
often enough that scanning them flags every figure in the repository. A check that is
always red gets `continue-on-error` and is then worth nothing, so the exclusion is part
of making the check real rather than a convenience.

Known offenders
---------------
KNOWN_CJK below is debt, not permission. Each entry records how many CJK characters the
file held when it was added, so the list visibly shrinks as files are translated and a
regression inside a listed file still shows up as a rising count. Adding a new entry
means writing down why the file cannot be translated now; it is not a way to land new
Chinese text.

Usage:
    python ci/check_english_only.py              # scan tracked files
    python ci/check_english_only.py --list       # show every offending line
    python ci/check_english_only.py --self-test  # prove the scan can fail
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# CJK ideographs, the Extension A block, CJK punctuation, and fullwidth forms. Latin
# text never lands in these, so a hit is unambiguous.
#
# Spelled as \u escapes, not literal characters. The first revision wrote the range
# endpoints as the ideographs themselves, which made this the one tracked file the scan
# could never pass: the detector contained what it detects. Escapes match the identical
# code points while keeping this source pure ASCII, and --self-test asserts that stays
# true.
CJK = re.compile(
    "["
    "\u4e00-\u9fff"  # CJK Unified Ideographs
    "\u3400-\u4dbf"  # Extension A
    "\u3000-\u303f"  # CJK punctuation: ideographic space and comma, corner brackets
    "\uff00-\uffef"  # Fullwidth forms: comma, colon, question mark and kin
    "]"
)

# Files that still contain Chinese, with the count at the time they were listed. This is
# a to-do list. Shrink it; do not extend it without a reason recorded here.
KNOWN_CJK = {
    # Long-form incident write-ups by an author who is not the person adding this check.
    # Translating technical prose one does not have the context for risks changing what
    # it claims, which is worse than leaving it in Chinese, so these wait for their
    # author rather than being machine-translated.
    "docs/experiments/lobbench_negative_time/root_cause.zh.md": 3240,
}


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files"],
                         capture_output=True, text=True, check=True).stdout
    return out.split()


def is_binary(path: Path) -> bool:
    """A NUL byte in the first 8 KiB is the usual heuristic, and it is enough here.

    Without this every PNG trips the scan: raw image bytes fall inside the CJK ranges
    routinely. Seven figures in this repository do.
    """
    try:
        chunk = path.open("rb").read(8192)
    except OSError:
        return True
    return b"\x00" in chunk


def scan(show_lines: bool) -> int:
    offenders: dict[str, list[tuple[int, str]]] = {}
    for rel in tracked_files():
        p = REPO / rel
        if not p.is_file() or is_binary(p):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits = [(i, line.rstrip()) for i, line in enumerate(text.splitlines(), 1)
                if CJK.search(line)]
        if hits:
            offenders[rel] = hits

    new = {f: h for f, h in offenders.items() if f not in KNOWN_CJK}
    grew = {f: len(CJK.findall((REPO / f).read_text(encoding="utf-8")))
            for f in offenders if f in KNOWN_CJK}
    grew = {f: n for f, n in grew.items() if n > KNOWN_CJK[f]}
    fixed = [f for f in KNOWN_CJK if f not in offenders]

    for f, n in sorted((f, len(h)) for f, h in offenders.items()):
        state = "known" if f in KNOWN_CJK else "NEW"
        print(f"  {state:5s} {n:>4} line(s)  {f}")
        if show_lines and (state == "NEW" or f in grew):
            for i, line in offenders[f][:12]:
                print(f"           {i}: {line[:100]}")

    print()
    if fixed:
        print(f"{len(fixed)} listed file(s) now clean -- remove them from KNOWN_CJK: {fixed}")
    if grew:
        for f, n in grew.items():
            print(f"REGRESSION: {f} grew from {KNOWN_CJK[f]} to {n} CJK characters",
                  file=sys.stderr)
    if new:
        print(f"\n{len(new)} file(s) contain Chinese and are not on the known list:",
              file=sys.stderr)
        for f in sorted(new):
            print(f"  {f}", file=sys.stderr)
        print("\nWrite it in English. If a file genuinely cannot be translated now, add "
              "it to KNOWN_CJK with a reason -- that list is debt to be paid down, not "
              "a way to land new Chinese text.", file=sys.stderr)
    if new or grew:
        return 1

    total = sum(KNOWN_CJK.values())
    print(f"no Chinese outside the known list "
          f"({len(KNOWN_CJK)} file(s), ~{total} characters, still to translate)")
    return 0


def self_test() -> int:
    """Prove the scan can fail, and that nothing it runs on every commit trips it.

    Two directions, both load-bearing. A scan that catches nothing is indistinguishable
    from a clean tree, so one character from each range is planted and must be hit. And
    a checker whose own machinery contains what it detects can never come back clean --
    this file spelled its regex with literal ideographs at first and flagged itself on
    every run -- so the checker and every workflow under .github/workflows are asserted
    clean here, where that regression would be caught at commit time. The workflows are
    globbed rather than named: the file that runs this scan has already been renamed once
    (english_only.yml folded into checks.yml) and a hard-coded name turned that rename
    into a FileNotFoundError on every push.
    """
    workflows = sorted((REPO / ".github" / "workflows").glob("*.yml"))
    if not workflows:
        print("self-test found no workflow under .github/workflows -- nothing runs this "
              "scan, so the probe below would assert nothing", file=sys.stderr)
        return 1
    probes = [
        ("plain ASCII stays clean", "def f():\n    return 1  # fine\n", False),
        ("an ideograph is caught", "# a different slice length \u6bcf step\n", True),
        ("Extension A is caught", "# \u3400\n", True),
        ("CJK punctuation is caught", "# \u3002\n", True),
        ("a fullwidth form is caught", "# \uff0c\n", True),
        ("this checker's own source is clean",
         Path(__file__).read_text(encoding="utf-8"), False),
    ] + [
        (f"the workflow that runs it is clean ({wf.name})",
         wf.read_text(encoding="utf-8"), False)
        for wf in workflows
    ]
    wrong = 0
    for name, text, want in probes:
        hit = bool(CJK.search(text))
        state = "caught" if hit else "clean"
        good = hit == want
        wrong += 0 if good else 1
        print(f"  {state:6s} {('ok' if good else 'WRONG'):5s}  {name}")
    if wrong:
        print(f"{wrong} self-test probe(s) failed", file=sys.stderr)
        return 1
    print("self-test passed: every planted range is caught, own machinery is clean")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="print each offending line")
    ap.add_argument("--self-test", action="store_true",
                    help="prove the scan catches each range and stays clean on itself")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    return scan(args.list)


if __name__ == "__main__":
    raise SystemExit(main())

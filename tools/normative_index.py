#!/usr/bin/env python3
"""Generate RFC-0001's normative-requirements index (Appendix D).

Run:  python tools/normative_index.py --check    # CI: is the appendix current?
      python tools/normative_index.py --write    # regenerate it in place

The appendix exists because a ~44,000-word specification carrying ~290 BCP 14
obligations cannot be reviewed for *consistency* by reading it: four of the
findings in the 2026-08-23 review pass were one failure repeated -- a rule
argued exhaustively in one section and not propagated to the places it binds
-- and every one of them is visible at a glance in a table that puts the
obligations about a single subject next to each other.

It is generated rather than written for the reason RFC-0001 gives itself for
dropping `provenance["order"]` (D15): a hand-maintained index is a cached
answer to an always-computable question, and the only thing it can do is go
stale. `tests/test_rfc0001_normative_index.py` fails when the body and the
appendix disagree, so the index cannot drift from the text it indexes.

**What counts as an obligation.** An all-capitals BCP 14 keyword, per the
document's own preamble, which states that the keywords bind "when, and only
when, they appear in all capitals". Occurrences inside fenced code blocks are
skipped -- those are examples, not clauses -- and so are the preamble's own
statement of the convention and the changelog, which quotes obligations
rather than imposing them. That distinction is `REVIEWING.md`'s: a count
check cannot tell a use from a mention.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

RFC = (
    Path(__file__).resolve().parents[1]
    / "rfcs"
    / "0001-persistence-diagram-interchange.md"
)

APPENDIX_HEADING = "## Appendix D — Normative requirements index"
_END_MARKER = "\n<!-- end normative index -->\n"

# All-capitals only, per the document's preamble. `MUST NOT` and `SHOULD NOT`
# are matched before their bare forms so the negation is never split off.
_KEYWORD = re.compile(r"\b(MUST NOT|SHOULD NOT|MUST|SHOULD|MAY)\b")

# Sections whose keywords are quotations rather than obligations. Section 12's
# own header says each row states the outcome "and points at the section that
# carries the normative requirement", so a D-row restating a MUST is by the
# document's own account a mention. Appendix C is the changelog and Appendix D
# is this index. `REVIEWING.md`: a count check cannot tell a use from a
# mention, so the distinction is drawn here rather than discovered later.
_QUOTING_SECTIONS = ("12", "Appendix C", "Appendix D")


@dataclass(frozen=True)
class Requirement:
    """One BCP 14 clause, as it appears in the body."""

    section: str
    keyword: str
    text: str

    @property
    def subject(self) -> str:
        """The first backticked identifier in the clause, or the empty string.

        Grouping by this is what makes the index catch the defect it exists
        for: two sections imposing different obligations on `essential_bars`
        sort together, so the disagreement is adjacent rather than 40 pages
        apart.
        """
        match = re.search(r"`([^`]+)`", self.text)
        return match.group(1) if match else ""


def _strip_code_blocks(lines: list[str]) -> list[str]:
    """Blank out fenced code, keeping line numbering intact.

    A keyword inside a fence is an example of a clause, not a clause. Math
    fences (`$$`) are left alone -- they carry no keywords and stripping them
    would need a second state machine for no gain.
    """
    out: list[str] = []
    in_fence = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append("")
            continue
        out.append("" if in_fence else line)
    return out


def _sentences(paragraph: str) -> list[str]:
    """Split on sentence boundaries that are not inside inline code or a number.

    Deliberately blunt. A clause that survives as two rows is a legible
    result; one silently merged with its neighbour is not.
    """
    protected = re.sub(r"`[^`]*`", lambda m: m.group(0).replace(".", "\x00"), paragraph)
    parts = re.split(r"(?<=[.:])\s+(?=[A-Z*`\[])", protected)
    return [p.replace("\x00", ".").strip() for p in parts if p.strip()]


def extract(source: str) -> list[Requirement]:
    """Every BCP 14 clause in the body, in document order."""
    lines = _strip_code_blocks(source.split("\n"))

    section = "(front matter)"
    requirements: list[Requirement] = []
    paragraph: list[str] = []

    def flush() -> None:
        if not paragraph:
            return
        text = " ".join(paragraph).strip()
        paragraph.clear()
        if section == "(front matter)" or section.startswith(_QUOTING_SECTIONS):
            return
        for sentence in _sentences(text):
            found = _KEYWORD.search(sentence)
            if found:
                requirements.append(
                    Requirement(section, found.group(1), _tidy(sentence))
                )

    for line in lines:
        heading = re.match(r"^#{2,3} (.+)$", line)
        if heading:
            flush()
            section = heading.group(1).strip()
            continue
        if not line.strip():
            flush()
            continue
        paragraph.append(line.strip())
    flush()
    return requirements


def _tidy(sentence: str) -> str:
    """One line, no emphasis markers, no table pipes."""
    text = re.sub(r"\*\*|\s+", lambda m: "" if m.group(0) == "**" else " ", sentence)
    return text.replace("|", "\\|").strip()


def _label(section: str) -> str:
    """`3.1` from `3.1 Invariants`, `A` from `Appendix A -- ...`."""
    match = re.match(r"^(\d+(?:\.\d+)?)", section)
    if match:
        return match.group(1)
    match = re.match(r"^Appendix ([A-Z])", section)
    return match.group(1) if match else section


def render(requirements: list[Requirement]) -> str:
    """The appendix, as markdown."""
    counts: dict[str, int] = {}
    rows = []
    for requirement in requirements:
        label = _label(requirement.section)
        counts[label] = counts.get(label, 0) + 1
        rows.append(
            f"| `N{label}-{counts[label]}` | §{label} | **{requirement.keyword}** "
            f"| {requirement.text} |"
        )

    totals = {}
    for requirement in requirements:
        totals[requirement.keyword] = totals.get(requirement.keyword, 0) + 1
    tally = ", ".join(f"{k} {v}" for k, v in sorted(totals.items()))

    return "\n".join(
        [
            APPENDIX_HEADING,
            "",
            "Generated by `tools/normative_index.py` from the body of this",
            "document, and regenerated by a test that fails when the two",
            "disagree. Do not edit it by hand: an index maintained separately",
            "from the text it indexes is a cached answer to an",
            "always-computable question, which is the ground D15 gives for",
            'dropping `provenance["order"]`.',
            "",
            "**What this is for.** The body carries the argument for each",
            "obligation and this carries the obligations, so a reader can",
            "check what conforming means without reading 44,000 words, and a",
            "reviewer can see every clause about one subject at once. Four of",
            "the defects found in the 2026-08-23 review pass were a single",
            "failure repeated -- a rule argued in one section and not",
            "propagated to the places it binds -- and each is visible here as",
            "two adjacent rows that disagree.",
            "",
            "**What it is not.** It is not normative. Where a row and the body",
            "differ, the body governs and the generator has a bug. Rows are",
            "clauses as they appear, so a sentence carrying two keywords",
            "appears once, under the first.",
            "",
            f"{len(requirements)} clauses: {tally}.",
            "",
            "| # | Section | Keyword | Clause |",
            "|---|---|---|---|",
            *rows,
            "",
            _END_MARKER.strip(),
        ]
    )


def splice(source: str, appendix: str) -> str:
    """Replace the existing appendix, or append it at the end."""
    start = source.find(APPENDIX_HEADING)
    if start == -1:
        return source.rstrip("\n") + "\n\n---\n\n" + appendix + "\n"
    end = source.find(_END_MARKER.strip(), start)
    assert end != -1, "appendix heading present without its end marker"
    return source[:start] + appendix + source[end + len(_END_MARKER.strip()) :]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if stale")
    parser.add_argument("--write", action="store_true", help="regenerate in place")
    args = parser.parse_args(argv)

    source = RFC.read_text(encoding="utf-8")
    body = (
        source[: source.find(APPENDIX_HEADING)]
        if APPENDIX_HEADING in source
        else source
    )
    updated = splice(source, render(extract(body)))

    if args.write:
        RFC.write_text(updated, encoding="utf-8")
        print(f"wrote {RFC}")
        return 0
    if updated != source:
        print("Appendix D is stale; run tools/normative_index.py --write")
        return 1
    print("Appendix D is current")
    return 0


if __name__ == "__main__":
    sys.exit(main())

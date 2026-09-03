#!/usr/bin/env python3
"""Generate RFC-0001's normative-requirements index (Appendix C).

Run:  python tools/normative_index.py --check    # CI: is the appendix current?
      python tools/normative_index.py --write    # regenerate it in place

The appendix exists because a specification this size cannot be checked for
*consistency* by reading it. Its failure mode is a rule argued exhaustively in
one section and not propagated to the places it binds, which is invisible in
prose and obvious in a table that puts every obligation about one subject next
to the others.

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

APPENDIX_HEADING = "## Appendix C — Normative requirements index"
CHANGELOG_HEADING = "## Appendix D — Changelog"
_END_MARKER = "\n<!-- end normative index -->\n"

# All-capitals only, per the document's preamble. `MUST NOT` and `SHOULD NOT`
# are matched before their bare forms so the negation is never split off.
_KEYWORD = re.compile(r"\b(MUST NOT|SHOULD NOT|MUST|SHOULD|MAY)\b")

# Sections whose keywords are quotations rather than obligations. Section 12's
# own header says each row states the outcome "and points at the section that
# carries the normative requirement", so a D-row restating a MUST is by the
# document's own account a mention. The appendices are evidence, rationale,
# this index and the changelog -- none of them a place a requirement lives,
# and Appendix B says outright that it is non-normative. `REVIEWING.md`: a
# count check cannot tell a use from a mention, so the line is drawn here
# rather than discovered later.
_QUOTING_SECTIONS = ("12",)
_APPENDIX_SECTION = re.compile(r"^(Appendix |[A-D]\.\d)")


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
        if not in_fence and re.match(r"^\s*[-*] ", line):
            # A list item is its own clause. Without this, a bulleted
            # requirement is glued to the paragraph above it and the row
            # reports two obligations under the first one's wording.
            out.append("")
            out.append(line)
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
        if _APPENDIX_SECTION.match(section):
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
    """`3.1` from `3.1 Invariants`, `A.7` from `A.7 array-api-compat -- ...`.

    Appendix subsections are headed by their own label rather than by the word
    "Appendix", so both spellings are matched and the label is only ever the
    number, never the title after it.
    """
    match = re.match(r"^Appendix ([A-Z])\b", section)
    if match:
        return match.group(1)
    match = re.match(r"^([A-Z]?\.?\d+(?:\.\d+)?|[A-Z]\.\d+)", section)
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
            "Generated by `tools/normative_index.py` from the body of this document, and",
            "regenerated by a test that fails when the two disagree. Do not edit it by hand:",
            "an index maintained separately from the text it indexes is a cached answer to an",
            "always-computable question, which is the ground D15 gives for",
            'dropping `provenance["order"]`.',
            "",
            "**What this is for.** The body carries the argument for each obligation; this",
            "carries the obligations. A reader can check what conforming means without",
            "reading the whole document, and every clause about one subject can be read",
            "together — which is where a rule stated in one section and contradicted in",
            "another shows up as two adjacent rows that disagree.",
            "",
            "**What it is not.** It is not normative. Where a row and the body differ, the",
            "body governs and the generator has a bug. Rows are clauses as they appear, so a",
            "sentence carrying two keywords appears once, under the first. §1 through §11 are",
            "indexed: §12 records decisions and points at the sections carrying them, and the",
            "appendices hold evidence and rationale, so a keyword in either is a quotation of",
            "an obligation rather than one.",
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
    """Replace the existing appendix, or insert it before the changelog.

    Position is not cosmetic. The changelog is removed when the comment window
    closes, by its own author's note, and this index is permanent; the
    permanent appendix therefore comes first, so that removing the other one
    leaves no gap in the lettering and renames nothing that outlives it.
    """
    start = source.find(APPENDIX_HEADING)
    if start == -1:
        anchor = source.find(CHANGELOG_HEADING)
        if anchor == -1:
            return source.rstrip("\n") + "\n\n---\n\n" + appendix + "\n"
        return source[:anchor] + appendix + "\n\n---\n\n" + source[anchor:]
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
        print("Appendix C is stale; run tools/normative_index.py --write")
        return 1
    print("Appendix C is current")
    return 0


if __name__ == "__main__":
    sys.exit(main())

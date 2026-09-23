"""Appendix D tracks the body it indexes. RFC-0001 §12.2 D15's reasoning.

An index maintained separately from the text it indexes is a cached answer to
an always-computable question, and the only thing it can do is go stale. So it
is generated, and this fails when the document and the generated appendix
disagree -- which is what makes it an index rather than a second copy with a
shorter half-life (`REVIEWING.md`).

This is also the trap a reader cannot see, made a standing regression test
rather than prose: the failure mode is a clause edited in the body while the
appendix keeps the old wording, and nobody notices because nobody diffs a
200-row table by eye.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "normative_index", _TOOLS / "normative_index.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def index() -> ModuleType:
    return _load()


def test_appendix_d_is_current(index: ModuleType) -> None:
    """The generated index and the committed one agree.

    Regenerate with `python tools/normative_index.py --write` after any change
    that adds, removes or reworks a BCP 14 clause. §10.2's bump rule fires on
    the same event, so a failure here is also a reminder that `spec_version`
    may owe an increment.
    """
    assert index.main(["--check"]) == 0


def test_code_blocks_are_not_mined_for_obligations(index: ModuleType) -> None:
    """A keyword inside a fence is an example of a clause, not a clause.

    `REVIEWING.md`: a count check cannot tell a use from a mention. This pins
    the distinction rather than trusting it.
    """
    source = "## 9. S\n\nA MUST clause.\n\n```python\n# arr MUST be sorted\n```\n"
    found = index.extract(source)

    assert [r.keyword for r in found] == ["MUST"]
    assert found[0].text == "A MUST clause."


def test_lowercase_keywords_are_not_obligations(index: ModuleType) -> None:
    """The document's preamble binds the keywords "only when" capitalised.

    Three sites are lowercase deliberately (`REVIEWING.md` names them), so an
    index that swept case-insensitively would report obligations the document
    says it is not making.
    """
    source = "## 9. S\n\nThe caller must sort, and the reader may not.\n"

    assert index.extract(source) == []


def test_the_decisions_section_is_read_as_quotation(index: ModuleType) -> None:
    """§12's rows point at the requirement; they are not the requirement.

    §12's own header says each row states the outcome "and points at the
    section that carries the normative requirement". Indexing them would list
    every settled obligation twice, once where it binds and once where it is
    described.
    """
    source = "## 12. Decisions\n\nD1: the adapter MUST refuse it.\n"

    assert index.extract(source) == []


def test_a_list_item_is_one_clause_across_its_line_breaks(index: ModuleType) -> None:
    """A bullet ends at the next blank line or the next marker, not at its
    first line break (#62).

    The shape that broke: the first line ends mid-clause and the keyword is on
    the second. Before the fix the first line was dropped for carrying no
    keyword and the second became a row starting mid-sentence.
    """
    source = (
        "## 9. S\n"
        "\n"
        "A paragraph above the list.\n"
        "- Every accessor on the surface, whatever its\n"
        "  namespace, MUST return a finite value.\n"
        "- The next item MAY differ.\n"
    )
    found = index.extract(source)

    assert [(r.keyword, r.text) for r in found] == [
        (
            "MUST",
            "- Every accessor on the surface, whatever its namespace, "
            "MUST return a finite value.",
        ),
        ("MAY", "- The next item MAY differ."),
    ]


def test_issue_62_clause_survives_whole(index: ModuleType) -> None:
    """The I10 bullet from #62, verbatim.

    The row is whole. Its keyword is still MAY: a sentence carrying two
    keywords appears once, under the first -- Appendix C's stated rule, which
    #62 does not change.
    """
    source = (
        "## 2. S\n"
        "\n"
        "- `birth`, `death` are **extended** real, with `birth <= death`. "
        "`death` MAY be\n"
        "  `+inf` and `birth` MAY be `-inf`, but a bar MUST NOT carry the same "
        "infinity\n"
        "  at both ends.\n"
    )
    found = index.extract(source)

    assert [(r.keyword, r.text) for r in found] == [
        (
            "MAY",
            "`death` MAY be `+inf` and `birth` MAY be `-inf`, but a bar MUST "
            "NOT carry the same infinity at both ends.",
        )
    ]


def test_a_list_item_ends_at_a_blank_line_or_a_nested_marker(
    index: ModuleType,
) -> None:
    """The item absorbs its own continuation lines and nothing else."""
    source = (
        "## 9. S\n"
        "\n"
        "- An outer item MUST end\n"
        "  here.\n"
        "  - A nested item SHOULD be its own clause.\n"
        "\n"
        "A paragraph after the list MAY stand alone.\n"
    )
    found = index.extract(source)

    assert [(r.keyword, r.text) for r in found] == [
        ("MUST", "- An outer item MUST end here."),
        ("SHOULD", "- A nested item SHOULD be its own clause."),
        ("MAY", "A paragraph after the list MAY stand alone."),
    ]


def _body(index: ModuleType) -> str:
    source: str = index.RFC.read_text(encoding="utf-8")
    return source.split(index.APPENDIX_HEADING, 1)[0]


# Clauses that legitimately open with a lowercase word. Add to this only for a
# name the document spells in lowercase, never to silence a truncated row.
_LOWERCASE_OPENINGS = ("giotto-tda",)


def test_every_clause_ends_where_a_sentence_ends(index: ModuleType) -> None:
    """A shape check on the generator's output rather than on its agreement
    with itself.

    `test_appendix_d_is_current` compares the appendix against the generator,
    so a generator defect appears identically on both sides. This compares a
    row against what any clause looks like. A trailing `*` is an italic
    paragraph closing, not a truncation.
    """
    for requirement in index.extract(_body(index)):
        assert requirement.text.rstrip("*_ ").endswith((".", ":", ";")), (
            requirement.text
        )


def test_no_clause_starts_mid_sentence(index: ModuleType) -> None:
    """The other half of a split clause starts lowercase."""
    for requirement in index.extract(_body(index)):
        opening = requirement.text.lstrip("-* _")
        assert not opening[:1].islower() or opening.startswith(_LOWERCASE_OPENINGS), (
            requirement.text
        )


_TWO_SECTIONS = (
    "## 3. A\n"
    "\n"
    "`alpha` SHOULD be tested.\n"
    "\n"
    "`beta` MUST be sorted.\n"
    "\n"
    "A clause naming nothing MUST still be indexed.\n"
    "\n"
    "## 11. B\n"
    "\n"
    "`alpha` MUST be tested.\n"
)


def test_clauses_about_one_subject_are_adjacent_across_sections(
    index: ModuleType,
) -> None:
    """#63: a rule stated in one section and contradicted in another shows up
    as two adjacent rows that disagree.

    `beta` has one clause and nothing to be read against, and the unnamed
    clause has no subject, so neither is repeated in the subject table.
    """
    rendered = index.render(index.extract(_TWO_SECTIONS))
    grouped = rendered.split(index.SUBJECT_HEADING, 1)[1]

    assert [line for line in grouped.splitlines() if line.startswith("| `")] == [
        "| `alpha` | `N3-1` | **SHOULD** | `alpha` SHOULD be tested. |",
        "| `alpha` | `N11-1` | **MUST** | `alpha` MUST be tested. |",
    ]


def test_the_document_order_table_is_unchanged_by_grouping(
    index: ModuleType,
) -> None:
    """Ids stay document-order and per section: the tests cite them."""
    rendered = index.render(index.extract(_TWO_SECTIONS))
    ordered = rendered.split(index.SUBJECT_HEADING, 1)[0]

    assert [
        line.split(" | ")[0] for line in ordered.splitlines() if line.startswith("| `N")
    ] == [
        "| `N3-1`",
        "| `N3-2`",
        "| `N3-3`",
        "| `N11-1`",
    ]

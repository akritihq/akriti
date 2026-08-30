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

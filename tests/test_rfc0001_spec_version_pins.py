"""RFC-0001's ``spec_version`` pins: every site found, every site registered.

Issue #59. A version bump must move every literal that quotes a current
``spec_version``. The checks before this module compared chosen sites with each
other, so a bump that moved neither side of a comparison went green: #53's
changelog said "the document becomes 1.1.1" while the Version row stayed at
1.1.0. This module compares a set against a set instead, the way
``tests/test_license_closure.py`` compares profiles against pyproject:

1. every registered site is found, exactly once;
2. every occurrence of a current value is a registered site, so a new pin site
   fails here the first time it appears;
3. the document pins agree with each other, and so do the implemented pins;
4. the two versions differ only when ``io.py`` says why.

There are two versions, not one. The *document* version is the Version row's.
The *implemented* version is what ``save`` writes -- §10.2's ``spec_version``,
"which revision of that specification the writer implemented". When a revision
widens a requirement ``core.py`` does not yet enforce, the second trails the
first, and ``_SPEC_VERSION_GAP`` in ``io.py`` must give the reason. A silent
divergence is the bug; a declared one is the design.

Assumptions:

- A new pin is written at a current value. The scan looks for the values the
  registered pins hold, not for every ``x.y.z`` in the tree, so a literal that
  is already stale when written is not found.
- Appendix D is the RFC's last section and lists its entries in order. Only its
  last "the document becomes x.y.z" is a current claim; every earlier entry is
  history, and Appendix D is not scanned.
- A pin lives in the RFC, or in a ``*.py`` or ``*.json`` file under ``src/`` or
  ``tests/``. In Python only string literals can be pins: comments, docstrings
  and the text of ``_SPEC_VERSION_GAP`` are prose, and during a gap the
  implemented version is a past one that prose legitimately quotes. This module
  holds no pin and is not scanned, because its examples would match.

Everything is read as source text, never by importing ``akriti``, so this runs
with no optional dependency installed.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

ROOT = Path(__file__).resolve().parents[1]
RFC = "rfcs/0001-persistence-diagram-interchange.md"
IO = "src/akriti/diagrams/io.py"
REVIEW_CLAUSES = "tests/test_rfc0001_review_clauses.py"
IO_TESTS = "tests/test_rfc0001_io.py"
CHANGELOG_HEADING = "## Appendix D — Changelog"
CHANGELOG_CLAIM = r"[Tt]he document becomes \**(\d+\.\d+\.\d+)"
GAP_NAME = "_SPEC_VERSION_GAP"
SCANNED_SUFFIXES = (".py", ".json")

Kind = Literal["document", "implemented"]


@dataclass(frozen=True)
class Pin:
    """A site quoting a current ``spec_version``.

    ``pattern`` is a multiline regex whose one group is the version. It must
    match exactly once in ``path``.
    """

    path: str
    pattern: str
    kind: Kind


REGISTERED_PINS = (
    Pin(RFC, r"^\| \*\*Version\*\* \| (\d+\.\d+\.\d+) — ", "document"),
    Pin(RFC, r'^  "spec_version": "(\d+\.\d+\.\d+)",$', "document"),
    Pin(RFC, r'`"(\d+\.\d+\.\d+)"` at time of writing', "document"),
    # Hand-written on purpose: the independent witness. Deriving it from the
    # header would make every comparison against it tautological.
    Pin(REVIEW_CLAUSES, r'^SPEC_VERSION = "(\d+\.\d+\.\d+)"$', "document"),
    Pin(IO, r'^_SPEC_VERSION = "(\d+\.\d+\.\d+)"$', "implemented"),
    Pin(IO_TESTS, r'^SPEC_VERSION = "(\d+\.\d+\.\d+)"$', "implemented"),
)


def version_key(version: str) -> tuple[int, int, int]:
    major, minor, patch = (int(part) for part in version.split("."))
    return major, minor, patch


def _assigns_gap(node: ast.AST) -> bool:
    if isinstance(node, ast.AnnAssign):
        targets: list[ast.expr] = [node.target]
    elif isinstance(node, ast.Assign):
        targets = node.targets
    else:
        return False
    return any(isinstance(t, ast.Name) and t.id == GAP_NAME for t in targets)


def declared_gap(io_source: str) -> str | None:
    """The value of ``_SPEC_VERSION_GAP`` in ``io.py``, read without importing."""
    for node in ast.parse(io_source).body:
        if _assigns_gap(node):
            assert isinstance(node, (ast.Assign, ast.AnnAssign))
            assert node.value is not None
            value = ast.literal_eval(node.value)
            assert value is None or isinstance(value, str), value
            return value
    raise AssertionError(f"{IO} no longer defines {GAP_NAME}")


def literal_lines(source: str) -> set[int]:
    """Lines of Python ``source`` spanned by a string literal that can be a pin.

    Docstrings and the value of ``_SPEC_VERSION_GAP`` are excluded; comments
    never reach the syntax tree.
    """
    tree = ast.parse(source)
    prose: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                prose.add(id(first.value))
        elif _assigns_gap(node):
            assert isinstance(node, (ast.Assign, ast.AnnAssign))
            prose.add(id(node.value))
    lines: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in prose
        ):
            lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return lines


def pin_problems(texts: dict[str, str]) -> list[str]:
    """Every way ``texts`` (repo-relative path -> contents) breaks the pin rules.

    An empty list means the pins are sound.
    """
    problems: list[str] = []
    values: dict[Kind, set[str]] = {"document": set(), "implemented": set()}
    registered: set[tuple[str, int]] = set()

    for pin in REGISTERED_PINS:
        text = texts.get(pin.path, "")
        matches = list(re.finditer(pin.pattern, text, re.M))
        if len(matches) != 1:
            problems.append(
                f"{pin.path}: registered pin {pin.pattern!r} matches "
                f"{len(matches)} times, not once -- update REGISTERED_PINS "
                "if the site moved"
            )
            continue
        values[pin.kind].add(matches[0].group(1))
        registered.add((pin.path, text.count("\n", 0, matches[0].start()) + 1))

    body, _, changelog = texts.get(RFC, "").partition(CHANGELOG_HEADING)
    claims = re.findall(CHANGELOG_CLAIM, changelog)
    if claims:
        values["document"].add(claims[-1])
    else:
        problems.append(f"{RFC}: Appendix D has no 'the document becomes x.y.z'")

    for kind, found in values.items():
        if len(found) > 1:
            problems.append(
                f"the {kind} pins disagree: {sorted(found)} -- a bump moved "
                "some of them and not all"
            )

    current = sorted(values["document"] | values["implemented"])
    occurrence = re.compile(
        r"(?<![\d.])(" + "|".join(map(re.escape, current)) + r")(?!\.?\d)"
    )
    for path, text in sorted({**texts, RFC: body}.items()):
        scanned = literal_lines(text) if path.endswith(".py") else None
        for number, line in enumerate(text.splitlines(), start=1):
            if (path, number) in registered:
                continue
            if scanned is not None and number not in scanned:
                continue
            match = occurrence.search(line) if current else None
            if match:
                problems.append(
                    f"{path}:{number}: {match.group(1)} is a current "
                    "spec_version but not a registered pin -- use the "
                    "module's constant, or add the site to REGISTERED_PINS"
                )

    if len(values["document"]) == 1 and len(values["implemented"]) == 1:
        (document,) = values["document"]
        (implemented,) = values["implemented"]
        gap = declared_gap(texts[IO])
        if document == implemented and gap is not None:
            problems.append(
                f"{IO}: {GAP_NAME} gives a reason, but both versions are "
                f"{document} -- set it back to None"
            )
        if document != implemented and not (gap and gap.strip()):
            problems.append(
                f"the document is {document} and save writes {implemented}, "
                f"but {IO} sets no {GAP_NAME} saying why"
            )
        if version_key(implemented) > version_key(document):
            problems.append(
                f"save writes {implemented}, a revision later than the "
                f"document's {document}"
            )
    return problems


def repo_texts() -> dict[str, str]:
    texts = {RFC: (ROOT / RFC).read_text(encoding="utf-8")}
    for top in ("src", "tests"):
        for path in sorted((ROOT / top).rglob("*")):
            if (
                path.suffix in SCANNED_SUFFIXES
                and path.is_file()
                and "__pycache__" not in path.parts
                and path != Path(__file__).resolve()
            ):
                relative = path.relative_to(ROOT).as_posix()
                texts[relative] = path.read_text(encoding="utf-8")
    return texts


def test_spec_version_pins_are_complete_and_agree() -> None:
    assert pin_problems(repo_texts()) == []


# --------------------------------------------------------------------------
# The check itself, on a synthetic tree, so a check that passes vacuously is
# caught too.
# --------------------------------------------------------------------------


def synthetic_tree(
    document: str = "3.4.5", implemented: str = "3.4.5", gap: str | None = None
) -> dict[str, str]:
    return {
        RFC: (
            f"| **Version** | {document} — `major.minor.patch` |\n"
            f'  "spec_version": "{document}",\n'
            f'| `spec_version` | `str` | `"{document}"` at time of writing |\n'
            f"{CHANGELOG_HEADING}\n"
            "- **(1)** — the document becomes 0.1.0.\n"
            f"- **(2)** — The document becomes {document}.\n"
        ),
        REVIEW_CLAUSES: f'SPEC_VERSION = "{document}"\n',
        IO: (f'_SPEC_VERSION = "{implemented}"\n{GAP_NAME}: str | None = {gap!r}\n'),
        IO_TESTS: f'SPEC_VERSION = "{implemented}"\n',
    }


def test_a_sound_tree_has_no_problems() -> None:
    assert pin_problems(synthetic_tree()) == []


def test_a_new_pin_site_is_found_the_first_time_it_appears() -> None:
    tree = synthetic_tree()
    tree["tests/test_new.py"] = 'meta = {"spec_version": "3.4.5"}\n'
    (problem,) = pin_problems(tree)
    assert problem.startswith("tests/test_new.py:1: 3.4.5 is a current")


def test_a_new_pin_site_in_the_rfc_body_is_found() -> None:
    tree = synthetic_tree()
    tree[RFC] = "Files written today say 3.4.5.\n" + tree[RFC]
    problems = pin_problems(tree)
    assert any(p.startswith(f"{RFC}:1: 3.4.5 is a current") for p in problems)


def test_comments_and_docstrings_are_not_pins() -> None:
    tree = synthetic_tree()
    tree["tests/test_history.py"] = (
        '"""This module was written against 3.4.5."""\n'
        "\n"
        "# The review pass landed 3.4.5.\n"
        "def f() -> None:\n"
        '    """Still 3.4.5."""\n'
    )
    assert pin_problems(tree) == []


def test_a_bump_that_misses_the_version_row_fails() -> None:
    """#53: the changelog claimed the new version and the header kept the old."""
    tree = synthetic_tree(document="3.4.6", implemented="3.4.6")
    tree[RFC] = tree[RFC].replace("| **Version** | 3.4.6", "| **Version** | 3.4.5")
    assert any("document pins disagree" in p for p in pin_problems(tree))


def test_a_registered_site_that_disappears_fails() -> None:
    tree = synthetic_tree()
    tree[RFC] = tree[RFC].replace('  "spec_version": "3.4.5",\n', "")
    (problem,) = pin_problems(tree)
    assert "matches 0 times" in problem


def test_declared_gap_reads_io_source() -> None:
    assert declared_gap(f"{GAP_NAME}: str | None = None\n") is None
    assert declared_gap(f'{GAP_NAME}: str | None = "#44"\n') == "#44"
    with pytest.raises(AssertionError, match="no longer defines"):
        declared_gap('_SPEC_VERSION = "3.4.5"\n')


@pytest.mark.parametrize(
    ("document", "implemented", "gap", "expected"),
    [
        ("3.5.0", "3.4.5", "3.5.0 widens I4; core.py enforces 3.4.5", None),
        ("3.5.0", "3.4.5", None, f"sets no {GAP_NAME}"),
        ("3.5.0", "3.4.5", "  ", f"sets no {GAP_NAME}"),
        ("3.4.5", "3.4.5", "left over from a gap", "set it back to None"),
        ("3.4.5", "3.5.0", "anything", "a revision later than"),
    ],
)
def test_the_gap_is_declared_rather_than_accidental(
    document: str, implemented: str, gap: str | None, expected: str | None
) -> None:
    problems = pin_problems(synthetic_tree(document, implemented, gap))
    if expected is None:
        assert problems == []
    else:
        (problem,) = problems
        assert expected in problem

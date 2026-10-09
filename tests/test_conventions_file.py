"""The conventions live in ``AGENTS.md`` alone, and nothing shadows it.

Claude Code reads ``AGENTS.md`` only when no ``CLAUDE.md`` sits in or above
the working directory. A committed ``CLAUDE.md`` -- ``/init`` writes one --
would silently replace every rule in ``AGENTS.md`` with whatever it held, for
every Claude session in the repository, and nothing else would notice.
`REVIEWING.md`: a trap a reader cannot see becomes a standing regression test,
not prose.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_agents_md_is_the_only_conventions_file() -> None:
    assert (ROOT / "AGENTS.md").is_file()
    shadows = [p for p in ("CLAUDE.md", ".claude/CLAUDE.md") if (ROOT / p).exists()]
    assert shadows == []

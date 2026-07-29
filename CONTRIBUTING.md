# Contributing to Akriti

Thank you for considering it. Akriti is built in the open from the first commit,
and contributions from outside the team are genuinely wanted — especially from
maintainers of the projects we build on.

## The most useful thing right now

**Comment on [RFC-0001](rfcs/0001-persistence-diagram-interchange.md).**

It specifies how persistence diagrams are exchanged between Python's TDA
backends, and it does not require you to install anything. If you maintain a TDA
library, or you have been bitten by a backend convention, your reading of it is
worth more to us than a patch. Open an issue with your comments.

If something here does not interoperate cleanly with your library, that is a bug
and we would like to hear about it.

## Setting up

```bash
git clone https://github.com/akritihq/akriti
cd akriti
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

`[dev]` installs the test and lint tooling plus the optional backends. The
default `pip install akriti` deliberately installs **no backend at all** — see
[DEPENDENCIES.md](DEPENDENCIES.md) for why.

```bash
pytest                     # tests
ruff check . && ruff format --check .
mypy                       # strict
python tools/check_license_closure.py
```

## Ground rules

These are short, and they are load-bearing.

1. **We delegate computation.** Persistence, bottleneck and Wasserstein
   distances go to GUDHI, Ripser and Hera. We do not reimplement them. If a
   delegation looks inadequate, open an issue — do not write our own.
2. **No new dependency without discussion.** The default install closure is
   permissive-only and CI enforces it. Read [DEPENDENCIES.md](DEPENDENCIES.md)
   before proposing one, and verify the package exists and is the one you mean.
3. **giotto-tda is AGPLv3 — do not read or reproduce its source.** The
   compatibility layer is clean-room, written from public API documentation
   only. This applies to prompting an AI model as much as to reading with your
   own eyes. AGPL-derived code in an Apache-2.0 codebase would be a serious
   problem for every downstream user.
4. **Numerical code cites its source.** Anything in `core/` or `castle/` must
   trace to a specific equation in the papers, cited in the docstring. Do not
   derive formulas in a pull request.
5. **Every docstring states its assumptions.** A domain scientist and their
   agent both read these.

## Pull requests

- Branch from `main`. Keep PRs small and single-purpose.
- Write a description that says *why*, not just what. A reviewer can read the
  diff.
- New behaviour needs tests. Numerical code needs property-based tests —
  stability bounds and invariances are an unusually good fit and catch what
  example-based tests miss.
- Public functions take a **leading batch dimension** and carry type hints.
- CI must be green: tests on Linux, macOS and Windows across Python 3.10–3.13,
  plus lint, types, and the licence-closure gate.
- All commits are Apache-2.0 by implication. There is no CLA.

### If a test needs a backend

Optional backends are absent from the default test environment on purpose. Mark
those tests:

```python
@pytest.mark.backend
def test_from_ripser_round_trip(): ...
```

Adapter tests should run against **real backend output**, not hand-written
arrays — the value of that layer is that it survives contact with what backends
actually emit. The exception is giotto-tda, which does not currently run on
scikit-learn ≥ 1.8; use the committed fixture arrays instead.

## Using AI assistance

We use it, by policy — capacity is our binding constraint. `CLAUDE.md` at the
repository root holds the conventions and is read automatically by agentic
tools. Two things we ask of contributors who use it:

- **Verify numerical output against the cited equation by hand.** Do not let a
  model derive a formula, and do not let one session write both a numerical
  function and the test that blesses it.
- **Verify every package name.** Models hallucinate them, and typosquatting
  against hallucinated names is a real attack.

A statistically plausible test with the wrong null is the failure mode this
project exists to protect users against. We hold ourselves to that first.

## Reporting bugs

Open an issue with the version, platform, a minimal reproduction, and what you
expected. If it is a **security** issue, do not open a public issue — see
[SECURITY.md](SECURITY.md).

Numerical or statistical incorrectness is the highest-priority class of bug
here. Report it publicly and we will treat it as urgent.

## Conduct

All participation is covered by our [Code of Conduct](CODE_OF_CONDUCT.md).

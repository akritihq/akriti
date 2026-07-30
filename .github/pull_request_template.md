## What and why

<!-- What changes, and why it should change. The reviewer can read the diff;
     tell them what the diff does not say. -->

## Checklist

<!-- Delete rows that genuinely do not apply. Do not delete a row because it
     failed -- say so instead. -->

- [ ] Tests added or updated, and they fail without this change
- [ ] Public functions carry type hints and take a leading batch dimension
- [ ] Docstrings state their assumptions
- [ ] `ruff check .`, `ruff format --check .`, `mypy` all clean

### If this touches numerical code (`core/`, `castle/`)

- [ ] Every formula traces to a specific equation, **cited in the docstring**
- [ ] The formula was checked by hand against the paper, not derived here
- [ ] The function and its test were **not** written in the same session
- [ ] Property-based tests cover the relevant invariance or stability bound

### If this touches `diagrams/`

- [ ] Behaviour matches [RFC-0001](../rfcs/0001-persistence-diagram-interchange.md);
      cite the section
- [ ] Essential bars stay `inf` — not dropped, not finitized implicitly
- [ ] Batches stay ragged
- [ ] Round-trip tests run against **real backend output**, not hand-written arrays

### If this touches dependencies

- [ ] The package exists and is the one intended — verified on PyPI, not recalled
- [ ] The whole transitive closure was checked, not just the direct dependency
- [ ] `python tools/check_license_closure.py` passes in a clean default-install venv
- [ ] [DEPENDENCIES.md](../DEPENDENCIES.md) updated

### If this touches `compat/giotto`

- [ ] Written **clean-room** from public API documentation only. No giotto-tda
      source was read, by a human or a model. giotto-tda is AGPLv3.

## Anything the reviewer should be suspicious of

<!-- Where are you least confident? What did you not test? What did you decide
     arbitrarily? This section is the most useful one in the template -- an
     honest answer here is worth more than a green checklist. -->

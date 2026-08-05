# Dependency and licensing policy

Akriti is Apache-2.0. This document records what enters the dependency closure,
under what licence, and why — verified rather than assumed.

It exists because the project's own working rules require every dependency to be
verified before it enters `pyproject.toml`, and because we are publishing an
audit that criticises this ecosystem's supply-chain hygiene. Our own closure has
to survive the same scrutiny.

`tools/check_license_closure.py` enforces this in CI. **Last verified by hand:
2026-07-29.**

---

## The rule

`pip install akriti` installs **permissively licensed code only** — MIT, BSD,
Apache-2.0, PSF, ISC, and equivalents. Anything copyleft, and anything whose
licence cannot be determined from metadata, goes behind an optional extra with
the consequence documented at the install boundary.

This is not licence purism. Akriti's whole strategic position rests on being
safely embeddable, including by users with restrictive redistribution
requirements. A GPL package appearing silently in a default install would
undermine that, and nobody would notice for a year.

---

## What we found

Three findings changed the packaging design. All are reproducible with
`tools/check_license_closure.py`.

### 1. `persim` pulls GPLv3 code into every install — via a package abandoned in 2019

`persim` (MIT) depends on **`hopcroftkarp`**, which is **GPLv3**. `ripser` (MIT)
depends on `persim`. So:

```
pip install ripser  ->  ripser (MIT) -> persim (MIT) -> hopcroftkarp (GPLv3)
```

`hopcroftkarp` 1.2.5 was uploaded **2019-10-11**. Three releases exist, all from
a single author, and there have been none in nearly seven years. It implements
one classical algorithm in pure Python.

**There is no arrangement in which `pip install akriti` ships a persistence
backend and keeps a permissive closure.** The original plan assumed a default
install containing GUDHI; that assumption does not survive contact with the
actual dependency graph.

**Resolution.** The default install is the interchange layer and **nothing
else** — no backend, and no numpy either. Every backend moves behind an extra.
This costs less than it sounds: `akriti.diagrams` needs no backend at all, and
"diagrams in" is the primary user path by design.

numpy left the closure one step later than the backends did, and for a
different reason. RFC-0001 §3.3 and §10.1 requirement 2 require
`diagrams/core.py` and `diagrams/adapters.py` to import nothing beyond the
standard library, working through `__array_namespace__` on whatever array the
caller already holds. Declaring numpy contradicts that no matter how the
modules are written: it installs numpy for a torch or JAX user who will never
touch it, and it turns "zero third-party dependencies" into a claim about
import statements rather than about what `pip install akriti` fetches. numpy
is needed only inside `io.py`'s `save`/`load` for the `.npz` payload, where
§3.3 requires a lazy, function-scoped import that raises a clear `ImportError`
naming numpy if it is absent.

**There is therefore no numpy version floor declared anywhere, deliberately.**
A floor is a constraint on an installed dependency and there is no longer one
to constrain. The baseline is a statement about the caller's environment
instead: array-API code paths need a caller-supplied `numpy>=2.0` — older
numpy has no main-namespace `__array_namespace__` — or another array-API-native
library, and an older numpy fails at the caller's own `__array_namespace__()`
call rather than inside akriti. `akriti[test]` is the only place numpy is
declared, and it pins that same `>=2.0`.

Measured cost of the split:

| Install | Distributions | Closure |
|---|---|---|
| `pip install akriti` | **1** | akriti alone; nothing third-party |
| `pip install akriti[rips]` | 22 | GPLv3 via `hopcroftkarp`; matplotlib (PSF) |

### 2. GUDHI ships no licence metadata at all

The `gudhi` wheel (3.11.0 and 3.13.0) declares **no `License-Expression`, no
licence classifier, and no `LICENSE` file inside the wheel**. The only signal is
a `Project-URL` pointing at <https://gudhi.inria.fr/licensing/>.

That page states GUDHI's own code is MIT, but that modules depending on **CGAL,
Miniball, or PyKeOps** carry GPL/LGPL restrictions and are marked
`Copyright: MIT (GPL v3)` in the documentation.

The PyPI wheel **bundles those modules** — `gudhi.AlphaComplex` imports and runs
from a plain `pip install gudhi`. The user therefore receives GPL-encumbered
functionality regardless of which modules they intend to call.

**Resolution.** Treat the whole `gudhi` wheel as GPLv3 for closure purposes.
It lives in `akriti[alpha]`, recorded in `MANUAL_LICENSES` in the checker with
the URL where the licence was confirmed. This is stricter than GUDHI's own
framing and deliberately so: we cannot ship a per-module licence claim we have
no mechanical way to verify.

*GUDHI is not the villain here. It is healthy, INRIA-funded, actively released,
and we build on it. Missing wheel metadata is a packaging gap, and a good
upstream contribution for us to offer.*

### 3. `giotto-tda` is AGPLv3 and does not currently run

Confirmed AGPLv3 in its own PyPI metadata. It must never enter a user-facing
extra, and the clean-room rule for `akriti.compat.giotto` is absolute — public
API documentation only, no source reading, including via a model.

Separately, `giotto-tda` 0.6.2 raises
`TypeError: check_array() got an unexpected keyword argument 'force_all_finite'`
on scikit-learn 1.8. The keyword was renamed in scikit-learn 1.6 and removed in
1.8. It is therefore test-only, pinned, and CI uses **committed fixture arrays**
rather than live calls. See RFC-0001 §9.2.

---

## Current closure

### Default — `pip install akriti`

**Empty.** No third-party distribution is installed, so there is no licence
surface to audit and the table below has no rows. That is the point of
`tools/check_license_closure.py` continuing to run against the default
profile: an empty closure is a property that has to keep being true, and the
cheapest way for it to stop being true is one convenience import.

| Package | Licence | Verified |
|---|---|---|
| *(none)* | — | — |

numpy appears in `akriti[test]` only. When `io.py` lands, `save`/`load` will
import it lazily at call time (RFC-0001 §3.3); that is a runtime requirement
on the caller's environment, not a declared dependency, and it does not
change this table.

### Extras

| Extra | Packages | Licence consequence |
|---|---|---|
| `rips` | `ripser` | MIT, **but GPLv3 transitively** via `persim` → `hopcroftkarp`; adds matplotlib |
| `alpha` | `gudhi` | **GPLv3** — CGAL/Miniball modules bundled in the wheel |
| `distances` | `persim` | MIT, **but GPLv3 transitively** via `hopcroftkarp` |
| `torch` | `torch` | BSD-3-Clause; multi-gigabyte, never a hard dependency |
| `bio` | `anndata` | BSD-3-Clause |
| `test` | `pytest`, `pytest-cov`, `hypothesis` | `hypothesis` is **MPL-2.0** — weak, file-level, test-only, never shipped |
| `lint` | `ruff`, `mypy` | MIT |

`hypothesis` is the one reviewed exception in the checker. MPL-2.0 obligations
attach per-file to MPL-licensed files; we neither modify nor redistribute them,
and test dependencies do not reach a user's runtime environment.

---

## Adding a dependency

1. **Verify the package exists and is the one you mean.** Models hallucinate
   package names, and typosquatting against hallucinated names is an established
   attack. Check the PyPI page, the source repository, and the maintainer.
2. **Check the whole closure, not the package.** `hopcroftkarp` is three levels
   down from anything we would write in `pyproject.toml`.
3. **Check maintenance.** Last release date, release count, number of
   maintainers. A single-maintainer package with no release since 2019 is a
   supply-chain risk independent of its licence.
4. **Run the checker** in a clean venv.
5. **Open a PR that says why.** A new default-closure dependency needs a
   maintainer's agreement, not just green CI.

Never silence the checker by widening its permissive list. Either move the
dependency behind an extra, or add an `ALLOWED_EXCEPTIONS` entry with a written
reason that a human signed off on.

---

## Open questions

- **Bottleneck and Wasserstein distance without GPL.** We delegate rather than
  reimplement (a hard rule), but every current delegate drags `hopcroftkarp` in.
  Options: GUDHI's own `bottleneck_distance` (already GPL-encumbered, so no
  improvement), Hera (no tagged release ever cut), or asking `persim` upstream
  whether the `hopcroftkarp` dependency is still needed — it may be replaceable
  with `scipy.optimize.linear_sum_assignment`. **Upstream contribution is the
  best outcome here and would benefit the whole ecosystem.** Owner: unassigned.
- **Should `akriti[rips]` be the documented default in the README quickstart?**
  It gives the two-line time-to-first-plot the plan commits to, at the cost of
  the clean closure. Undecided.

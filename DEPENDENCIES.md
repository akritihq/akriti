# Dependency and licensing policy

Akriti is Apache-2.0. This document records what enters the dependency closure,
under what licence, and why — verified rather than assumed.

It exists because the project's own working rules require every dependency to be
verified before it enters `pyproject.toml`, and because we are publishing an
audit that criticises this ecosystem's supply-chain hygiene. Our own closure has
to survive the same scrutiny.

`tools/check_license_closure.py` enforces this in CI. **Last verified by hand:
2026-08-10.**

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

These findings changed the packaging design. All are reproducible with
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
`diagrams/core.py` to import nothing beyond the standard library, and adapters
to work through the namespace of whatever array the caller already holds.
Declaring numpy in the default closure contradicts that no matter how the
modules are written: it installs numpy for a torch or JAX user who will never
touch it, and it turns "zero third-party dependencies" into a claim about
import statements rather than about what `pip install akriti` fetches. NumPy
is needed only at two lazy boundaries: `io.py`'s `save`/`load` for the `.npz`
payload, and an adapter input made only of Python rows, which contains no array
from which a namespace can be derived.

**The floor is nevertheless declared, and this reverses what an earlier
revision of this document said.** That revision concluded there was nowhere
to put a numpy version floor once numpy left the required closure. RFC-0001
D6 reverses it: leaving the *required* closure and being undeclared are
different things, and only the first is wanted. numpy is declared by two
extras at `numpy>=2.0`, the floor D6 established for main-namespace
`__array_namespace__`:

| Extra | For |
|---|---|
| `akriti[numpy]` | numpy as the array namespace, including the fallback for accepted Python-row adapter inputs that carry no array |
| `akriti[io]` | `.akd` `save`/`load` (RFC-0001 §10); resolves to `akriti[numpy]`, so the floor has one home |

They are separate because the wants are separate: a user who wants numpy as
their array backend should not have to install under the name of
serialization. `akriti[test]` also resolves to `akriti[numpy]` rather than
pinning its own copy.

**What the floor buys is an install-time failure instead of a runtime one.**
Undeclared, a user on numpy 1.24 gets an `AttributeError` from the first
array-API call, because a lazy import guarded on *presence* fires when numpy
is absent and stays silent when it is merely too old. Declared, `pip install
akriti[io]` fails to resolve, which is where a version problem belongs. §3.3
consequently requires the lazy import to check the version rather than
presence alone. Serialization failures name `akriti[io]`; row-sequence
adapter failures name `akriti[numpy]`. Neither says only "install numpy",
which is not an instruction a user who already has numpy 1.24 can act on.

None of this touches the default closure: `pip install akriti` still fetches
nothing third-party.

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

### 4. `array-api-compat` closes the torch namespace gap

`array-api-compat` exists on PyPI; **1.15.0** is the currently verified
release and requires Python >=3.10. PyPI records the **MIT** license, the
Consortium for Python Data API Standards as author, and the maintainers
`Aaron.Meurer`, `ev-br` and `rgommers`. It supplies Array API namespaces for
NumPy, PyTorch and other supported array libraries.

Akriti declares it only in `akriti[torch]` at `>=1.15.0`. Torch tensors do not
yet expose `__array_namespace__`, so RFC-0001 §3.3's `namespace_of` fallback
imports the package lazily and uses its torch namespace. The dependency is
unreachable from the default install and does not turn the interchange layer
into a torch dependency.

### 5. PyArrow is the Parquet escape hatch

`pyarrow` exists on PyPI; **25.0.0** is the current verified floor. PyPI records
Apache-2.0, Python >=3.10 (including 3.10--3.14), Apache Arrow as owner, and
Apache Arrow Developers as maintainer. It provides the Arrow table and Parquet
writer used by `to_parquet()`; it is not needed by any other exporter or by
diagram construction.

It therefore lives in `akriti[parquet]`, and the import remains lazy at the
`to_parquet()` call boundary. CI installs that extra in a **separate clean
venv** and enforces a permissive-only closure there; the separation prevents
cumulative copyleft from masking a future Parquet packaging regression.

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

numpy is absent from the default closure and reachable only through
`akriti[numpy]`, `akriti[io]` and `akriti[test]`, so it does not change this
table. The implemented optional boundaries are lazy and function-scoped:
row-sequence adapter inputs use `akriti[numpy]`, `namespace_of`'s torch
fallback uses `akriti[torch]`, and `to_parquet()` uses `akriti[parquet]`.
RFC-0001's planned `save`/`load` boundary will use `akriti[io]` when it lands.
Each implemented boundary checks the version where needed and names the
relevant extra on failure.

### Extras

| Extra | Packages | Licence consequence |
|---|---|---|
| `rips` | `ripser` | MIT, **but GPLv3 transitively** via `persim` → `hopcroftkarp`; adds matplotlib |
| `alpha` | `gudhi` | **GPLv3** — CGAL/Miniball modules bundled in the wheel |
| `distances` | `persim` | MIT, **but GPLv3 transitively** via `hopcroftkarp` |
| `numpy` | `numpy>=2.0` | BSD-3-Clause; the array namespace, not required by the default install |
| `io` | → `akriti[numpy]` | BSD-3-Clause; `.akd` `save`/`load` only (RFC-0001 §10) |
| `torch` | `torch`, `array-api-compat>=1.15.0` | Torch is BSD-3-Clause and multi-gigabyte; array-api-compat is MIT; both are optional and never hard dependencies |
| `parquet` | `pyarrow>=25.0.0` | Apache-2.0; Apache Arrow Developers; strict permissive-only closure audited separately |
| `bio` | `anndata` | BSD-3-Clause |
| `test` | `pytest`, `pytest-cov`, `hypothesis`, `packaging>=22`, `array_api_strict`, → `akriti[numpy]` | `hypothesis` is **MPL-2.0** — weak, file-level, test-only, never shipped |
| `lint` | `ruff`, `mypy` | MIT |

`hypothesis` is the one reviewed exception in the checker. MPL-2.0 obligations
attach per-file to MPL-licensed files; we neither modify nor redistribute them,
and test dependencies do not reach a user's runtime environment.

`packaging>=22` is declared directly for PEP 440 release/version validation. Its
verified licence is **Apache-2.0 OR BSD-2-Clause**. Pytest already pulls it
transitively, so the direct declaration makes the test requirement explicit
without changing the resolved test closure or the empty default closure.

The torch row is **report-only** in CI because the binary wheel is a large,
platform-specific closure and its transitive metadata is not a supported strict
permissive-only install contract. The optional torch job audits the already
installed row with `--allow-copyleft`; no second torch installation is made.

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

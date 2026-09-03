<p align="center">
  <img src="branding/akriti-logo.svg#gh-light-mode-only" alt="Akriti" width="360">
  <img src="branding/akriti-logo-dark.svg#gh-dark-mode-only" alt="Akriti" width="360">
</p>

<p align="center">
  <strong>Statistically grounded topological data analysis for Python.</strong><br>
  <a href="https://akriti.io">akriti.io</a> ·
  <a href="LICENSE">Apache-2.0</a> ·
  built on <a href="https://gudhi.inria.fr/">GUDHI</a> and <a href="https://ripser.scikit-tda.org/">Ripser</a>
</p>

---

Persistence diagrams tell you what shape your data has. Akriti tells you whether
the answer is **significant**.

It provides the statistical layer that Python's TDA stack lacks — hypothesis
tests, effect sizes, per-region significance, and sample-size calculation for
persistence diagrams — while delegating persistence computation to the
established engines rather than reimplementing them.

## Status

> **Early development.** The API is unstable and much of what is described below
> is not written yet. `akriti 0.1.0` is on PyPI as of 2026-08-27 — one working
> module, `akriti.diagrams`, and nothing else. Star the repository to follow
> progress, or watch [akriti.io](https://akriti.io).

We would rather be accurate than impressive, so:

| Module | What it is | State |
|---|---|---|
| `akriti.diagrams` | One persistence-diagram type, with adapters for GUDHI, Ripser, giotto-tda, persim and plain arrays — specified by [RFC-0001](rfcs/0001-persistence-diagram-interchange.md) | **building** |
| `akriti.castle` | Two-sample test, sample-size calculator, per-region significance map, robustness certificate, reporting card | planned — scaffold in review |
| `akriti.core` | Landmark embeddings (PLACE / PALACE), closed-form selectors, certificate radii | planned — `core/distances.py` is specified by RFC-0001 §9.1 and not yet written |
| `akriti.compute` | Diagrams from point clouds, images, time series and graphs — delegated, with defended defaults | planned |
| `akriti.vectorise` | Persistence images, landscapes, Betti curves, landmark embeddings, plus a maintained benchmark | planned |
| `akriti.compat` | Compatibility layer for giotto-tda pipelines | planned |

## Why

Three gaps, stated as precisely as we can:

1. **Statistical inference for diagrams lives in R, not Python.** The `TDA`,
   `TDAstats` and `tdaverse` packages have offered permutation tests and
   bootstrap confidence sets for years. Python users have had essentially
   nothing.
2. **No library, in any language, calculates sample size for a topological
   effect.** *"How many samples do I need to detect a bottleneck-distance
   difference of size δ?"* is a question applied statistics answers routinely,
   and topology has never answered at all.
3. **Python's general-purpose TDA layer has gone quiet.** giotto-tda has had no
   commits since 2024 while still being installed thousands of times a month.
   Its users deserve somewhere maintained to land.

## Design commitments

- **We delegate computation.** Persistence, bottleneck and Wasserstein distances
  go to GUDHI, Ripser and Hera. We do not reimplement them, and we will not.
- **Backend-agnostic input.** Bring diagrams from any library, or none.
- **Honest defaults.** Where our theory supports a principled choice of
  filtration, scale or descriptor, the library makes it and explains why. Where
  it provably does not — the landmark budget, placement, bandwidth, and the
  concatenation rules — the library says so and points you at cross-validation
  instead of pretending.
- **Permissive by default.** Apache-2.0, and the default install closure is
  permissive-only — verified in CI, not asserted. Every backend has a copyleft
  dependency somewhere in its closure, so every backend is an opt-in extra.

## Specifications

Before the code, the contract. **[RFC-0001 — Persistence Diagram
Interchange](rfcs/0001-persistence-diagram-interchange.md)** pins down what a
persistence diagram *is* across Python's backends: infinite bars, ordering,
precision, equality, metadata and serialization. It is open for comment, and it
is useful whether or not you ever install this library — the R ecosystem solved
interchange first, and Python has not.

Every convention in it was measured against GUDHI, Ripser, persim and giotto-tda
rather than recalled. `rfcs/evidence/probe_backends.py` reproduces every number.
Three findings you may want regardless of Akriti:

- **giotto-tda silently drops the essential H0 bar** — under every
  `infinity_values` setting. 40 points, 40 components, 39 reported.
- **giotto-tda's batch padding is indistinguishable from real bars.** The same
  point cloud yields 2 one-dimensional bars alone and 11 when batched with
  another; the padding is written with a genuine birth value.
- **`persim.bottleneck` returns a finite distance between diagrams that are
  infinitely far apart** — 0.5 where the answer is ∞. It does warn that it is
  dropping the infinite bars, but the warning describes the mechanism rather
  than the consequence, and it fires more often on the case it gets *right*
  than on the case it gets wrong.

## Install

```bash
pip install akriti              # interchange layer — zero dependencies
pip install akriti[rips]        # + Ripser    (MIT, GPLv3 transitively)
pip install akriti[alpha]       # + GUDHI     (GPLv3)
pip install akriti[distances]   # + persim    (MIT, GPLv3 transitively)
pip install akriti[numpy]       # + NumPy namespace / Python-row fallback
pip install akriti[parquet]     # + PyArrow   (Apache-2.0)
pip install akriti[torch]       # + torch and array-api-compat
pip install akriti[jax]         # + JAX       (Apache-2.0); see the note below
pip install akriti[bio]         # + anndata   (BSD-3)
```

> Currently a placeholder release; real functionality is coming.

**Nothing is a required dependency** — no persistence backend, and no NumPy
either. Native array inputs retain their Python array API namespace. Accepted
Python-row adapter inputs lazily use `akriti[numpy]`; torch tensors use the
compatibility resolver supplied by `akriti[torch]`; and Parquet imports
PyArrow only when requested through `akriti[parquet]`. JAX arrays need no such
boundary -- they expose the namespace natively, so `akriti[jax]` is a
convenience that installs JAX for you and nothing more. "Bring your own
diagrams" remains the primary path by design. The licence consequences above
are stated here rather than in a footnote because they are real: `persim`
depends on `hopcroftkarp`, which is GPLv3 and has had no release since 2019,
and the `gudhi` wheel bundles CGAL-dependent modules and ships no licence
metadata at all. See **[DEPENDENCIES.md](DEPENDENCIES.md)** for the verified
closure, and `tools/check_license_closure.py` for the CI gate that keeps it
honest.

**JAX needs a 64-bit configuration that you set, not us.** A diagram stores
`float64` births and deaths, and a default JAX install truncates both to
`float32` -- so building one raises a `ValueError` naming the dtype it got.
Either lever fixes it, and the first is narrower and preferred, changing only
what akriti asks for rather than every default dtype in your process:

```python
import jax

jax.config.update("jax_explicit_x64_dtypes", "allow")  # preferred
jax.config.update("jax_enable_x64", True)  # heavier alternative
```

akriti will not set either for you. Both are process-global, so a library
setting one would silently change the numerics of unrelated JAX code in your
program. See RFC-0001 §3.3 and D23.

## The research behind it

| | |
|---|---|
| **CASTLE** (Paper IV) | A practitioner's toolkit for topological two-sample testing, sample-size calculation and robustness certification · *in preparation* |
| **Paper III** | A statistical-inference pipeline for persistence-landmark kernels: CLT, Berry–Esseen and functional limits · *in preparation* |
| **PLACE** (Paper I) | A closed-form persistence-landmark pipeline for certified point-cloud and graph classification · [arXiv:2605.02836](https://arxiv.org/abs/2605.02836) · *TMLR, under review* |
| **PALACE** (Paper II) | Adaptive landmark embeddings for persistence diagrams · [arXiv:2605.04046](https://arxiv.org/abs/2605.04046) · *JMLR, under review* |

CASTLE is the practitioner-facing product; the others are the machinery that
makes its guarantees possible.

## Contributing

Contributions are welcome, including — especially — from maintainers of the
projects we build on. If you maintain a TDA library and something here does not
interoperate cleanly with yours, that is a bug and we would like to hear about
it.

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Security reports go to
[SECURITY.md](SECURITY.md).

## Team

- **Sushovan Majhi** — Data Science, GW · lead, library architecture
- **Pramita Bagchi** — biostatistics · practitioner statistics
- **Atish Mitra** — Mathematics, Montana Tech · theoretical foundations
- **Žiga Virk** — Mathematics, Ljubljana · theory advisor
- **Alexander Silberman** — GW · library development
- **Edward Bae** — GW · library development

Built on [GUDHI](https://gudhi.inria.fr/) (INRIA),
[Ripser](https://ripser.scikit-tda.org/), [Hera](https://github.com/anigmetov/hera)
and [persim](https://persim.scikit-tda.org/), and on the landmark embedding of
[Mitra & Virk (2024)](https://arxiv.org/abs/2402.04860). With thanks to the
wider TDA community, including [scikit-tda](https://scikit-tda.org/),
[giotto-tda](https://giotto-ai.github.io/gtda-docs/) and the R
[tdaverse](https://github.com/tdaverse).

## Licence

Apache-2.0 — see [LICENSE](LICENSE). The explicit patent grant is deliberate: it
is what makes the library usable inside institutions whose legal review would
otherwise block adoption.

## Contact

[hello@akriti.io](mailto:hello@akriti.io) · [@akritihq](https://x.com/akritihq) ·
[github.com/akritihq](https://github.com/akritihq)

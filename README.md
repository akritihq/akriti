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
> is not written yet. `akriti` on PyPI is currently a `0.0.0` placeholder holding
> the name. Star the repository to follow progress, or watch
> [akriti.io](https://akriti.io).

We would rather be accurate than impressive, so:

| Module | What it is | State |
|---|---|---|
| `akriti.diagrams` | One persistence-diagram type, with adapters for GUDHI, Ripser, giotto-tda, persim and plain arrays | **building** |
| `akriti.castle` | Two-sample test, sample-size calculator, per-region significance map, robustness certificate, reporting card | **building** |
| `akriti.core` | Landmark embeddings (PLACE / PALACE), closed-form selectors, certificate radii | **building** |
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
- **Permissive by default.** Apache-2.0, and the default install closure stays
  MIT/BSD. Backends with copyleft dependencies live behind opt-in extras.

## Install

```bash
pip install akriti          # placeholder release; real functionality is coming
```

## The research behind it

| | |
|---|---|
| **CASTLE** (Paper IV) | A practitioner's toolkit for topological two-sample testing, sample-size calculation and robustness certification · *in preparation* |
| **Paper III** | A statistical-inference pipeline for persistence-landmark kernels: CLT, Berry–Esseen and functional limits · *in preparation* |
| **PLACE** (Paper I) | A closed-form persistence-landmark pipeline for certified point-cloud and graph classification · *TMLR, under review* |
| **PALACE** (Paper II) | Adaptive landmark embeddings for persistence diagrams · *JMLR, under review* |

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

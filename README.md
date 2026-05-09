# Akriti

> Modern topological data analysis for Python.

**[akriti.io](https://akriti.io)** · **Apache-2.0** · **Built on [GUDHI](https://gudhi.inria.fr/)**

Akriti is an open-source Python library for topological data analysis — persistence, Mapper, topological deep learning, and topological signal processing. It computes diagrams, vectorises them, measures distances, classifies, compares two samples, and ships per-prediction certificates as a free upsell.

A distinguishing feature: every method choice in the AutoML pipeline — filtration, scale, landmark grid, descriptor — is fixed *analytically* from your training data, with provable guarantees. No held-out validation. No held-out anything.

## Status

**Library in active development.** First public release `v0.0.1` targeted for August 2026. Star this repository to follow progress.

The underlying research papers are at various stages of submission:

- **PLACE** (Paper I) — TMLR 2026, under review
- **PALACE** (Paper II) — JMLR 2026, under review
- **Paper III** (inferential theory) — in preparation, 2026
- **CASTLE** (Paper IV) — in preparation, 2026

## Install

Once `v0.0.1` ships:

```bash
pip install akriti
```

For now, follow the repo or sign up via [hello@akriti.io](mailto:hello@akriti.io) for early-access notification.

## Quick taste

```python
import akriti

# Detect cycles in a point cloud
result = akriti.cycles(X)
result.plot()

# Compare two biological samples topologically
test = akriti.compare(sample_A, sample_B)
print(test.p_value, test.certificate)

# Train a certified classifier
clf = akriti.classify(X_train, y_train)
preds = clf.predict(X_test)  # per-prediction certificates included
```

## Project layout

```
akriti/
├── branding/                    Brand assets (logo, wordmark, palette)
├── website/                     Quarto source for akriti.io
├── docs/                        Rendered website (GitHub Pages root)
├── LICENSE                      Apache-2.0
└── README.md
```

The Python package will land here as the library matures. Until then, this repository is the home for the website, brand assets, and project documentation.

## Building the website

```bash
cd website
quarto render
```

Output is written to `../docs/`, which GitHub Pages serves at [akriti.io](https://akriti.io).

## Founding team

Akriti is founded by:

- **Pramita Bagchi** — biostatistics, CASTLE
- **Sushovan Majhi** — Data Science · GW (lead PI), library architecture
- **Atish Mitra** — Mathematics · Montana Tech, theoretical foundations
- **Žiga Virk** — Mathematics · Ljubljana, theory advisor

And built by an open community. Apache-2.0 means contributions, integrations, vectorisations, and case studies are all welcome — just open an issue or a pull request.

## Built on

Akriti delegates persistence computation to [GUDHI](https://gudhi.inria.fr/) (INRIA) and builds on the landmark embedding of [Mitra & Virk (2024)](https://arxiv.org/abs/2402.04860). It is grateful to the broader TDA community — [giotto-tda](https://giotto-ai.github.io/gtda-docs/), [persim](https://persim.scikit-tda.org/), [ripser.py](https://ripser.scikit-tda.org/), and the many vectorisation and stability results that the library stands on.

## Licence

Apache-2.0. See [LICENSE](LICENSE).

## Contact

[hello@akriti.io](mailto:hello@akriti.io) · [@akritihq](https://x.com/akritihq) on X · [github.com/akritihq](https://github.com/akritihq)

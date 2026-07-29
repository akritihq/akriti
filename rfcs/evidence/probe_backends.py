#!/usr/bin/env python3
"""Reproduce every measured claim in RFC-0001.

Run:  python rfcs/evidence/probe_backends.py

Measured 2026-07-29 with gudhi 3.11.0, ripser 0.6.14, persim 0.3.8,
giotto-tda 0.6.2, numpy 2.4.4, scikit-learn 1.8.0, Python 3.12.11.

Clean-room note (onboarding §8): giotto-tda is AGPLv3. This script calls its
public API and inspects returned arrays. No giotto source is read, and none may
be read while implementing akriti.compat.giotto.
"""

from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

SEED = 0
N = 40
NOISE = 0.05


def sample_circle(n: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0, 2 * np.pi, n)
    pts = np.c_[np.cos(theta), np.sin(theta)] + rng.normal(0, noise, (n, 2))
    return np.ascontiguousarray(pts, dtype=np.float64)


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def patch_giotto() -> bool:
    """giotto-tda 0.6.2 calls check_array(force_all_finite=...), which
    scikit-learn renamed in 1.6 and removed in 1.8 (RFC-0001 §9.2).

    Translate the kwarg so the rest of the probe can run. This is a local
    workaround at the public-API boundary, not a fix and not a patch we ship.
    """
    import sklearn
    import sklearn.utils

    try:
        import gtda.utils.validation as gval
    except ImportError:
        return False

    original = sklearn.utils.check_array

    def shim(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return original(*args, **kwargs)

    sklearn.utils.check_array = shim
    gval.check_array = shim
    print(f"  [shim] scikit-learn {sklearn.__version__}: "
          "translated force_all_finite -> ensure_all_finite for giotto-tda")
    return True


def main() -> None:
    rng = np.random.default_rng(SEED)
    A = sample_circle(N, NOISE, rng)          # noisy circle: one clear H1 class
    B = rng.normal(0, 1, (N, 2))              # gaussian blob: many short H1 bars

    # ---------------------------------------------------------------- A.1
    rule("A.1  ESSENTIAL BARS — what each backend does with the infinite bar")

    import gudhi

    st = gudhi.RipsComplex(points=A, max_edge_length=4.0).create_simplex_tree(
        max_dimension=2)
    st.persistence()
    g0 = st.persistence_intervals_in_dimension(0)
    g1 = st.persistence_intervals_in_dimension(1)
    print(f"  gudhi   H0={len(g0):3d}  essential={int(np.isinf(g0[:, 1]).sum())}"
          f"  H1={len(g1)}")
    print(f"          persistence() entry form: {st.persistence()[:1]}")

    from ripser import ripser

    dgms = ripser(A, maxdim=1)["dgms"]
    r0, r1 = dgms[0], dgms[1]
    print(f"  ripser  H0={len(r0):3d}  essential={int(np.isinf(r0[:, 1]).sum())}"
          f"  H1={len(r1)}")

    have_giotto = patch_giotto()
    if have_giotto:
        from gtda.homology import VietorisRipsPersistence

        for iv in (None, np.inf, 99.0):
            vr = VietorisRipsPersistence(homology_dimensions=(0, 1),
                                         infinity_values=iv)
            g = vr.fit_transform(A[None])[0]
            h0 = g[g[:, 2] == 0]
            print(f"  giotto  H0={len(h0):3d}  essential="
                  f"{int((~np.isfinite(g)).sum())}"
                  f"  H1={int((g[:, 2] == 1).sum())}"
                  f"   infinity_values={iv!r} -> {vr.infinity_values_}")
        print("  => giotto drops the essential class under every setting.")

    # ---------------------------------------------------------------- A.2
    rule("A.2  GIOTTO BATCH PADDING — the diagram depends on the batch")

    if have_giotto:
        vr = VietorisRipsPersistence(homology_dimensions=(0, 1))
        solo = {name: vr.fit_transform(X[None])[0] for name, X in (("A", A), ("B", B))}
        batched = vr.fit_transform(np.stack([A, B]))

        for name, g in solo.items():
            print(f"  {name} alone   rows={len(g):3d}  "
                  f"H1={int((g[:, 2] == 1).sum()):3d}  "
                  f"trivial={int(np.isclose(g[:, 0], g[:, 1]).sum())}")
        for i, name in enumerate("AB"):
            g = batched[i]
            triv = np.isclose(g[:, 0], g[:, 1])
            print(f"  {name} batched rows={len(g):3d}  "
                  f"H1={int((g[:, 2] == 1).sum()):3d}  "
                  f"trivial={int(triv.sum())}")
            if triv.sum():
                print(f"     padding rows look like: {g[triv][0]}")
        print("  => A yields 2 H1 bars alone and 11 batched. Padding is written")
        print("     with a real birth value, so it is indistinguishable from a")
        print("     genuine zero-persistence bar.")

    # ---------------------------------------------------------------- A.3
    rule("A.3  PRECISION AND ORDERING — gudhi vs ripser on identical input")

    print(f"  ripser raw order:\n{r1}")
    print(f"  gudhi  raw order:\n{g1}")
    print(f"  same row order: {np.allclose(r1, g1, atol=1e-6)}")

    rs = r1[np.lexsort((r1[:, 1], r1[:, 0]))]
    gs = g1[np.lexsort((g1[:, 1], g1[:, 0]))]
    diff = np.abs(rs - gs).max()
    scale = np.abs(gs).max()
    print(f"  dtypes: ripser={r1.dtype} gudhi={g1.dtype}")
    print(f"  max |diff| after sorting : {diff:.3e}")
    print(f"  float32 eps at this scale: {np.finfo(np.float32).eps * scale:.3e}")
    print(f"  float64 eps at this scale: {np.finfo(np.float64).eps * scale:.3e}")
    print("  => ripser returns float64 arrays holding float32-precision values.")

    # ---------------------------------------------------------------- A.4
    rule("A.4  PERSIM — finite distance between infinitely distant diagrams")

    import persim

    inf_d = np.array([[0.0, np.inf], [0.1, 0.5]])
    fin_d = np.array([[0.0, 1.0], [0.1, 0.5]])
    empty = np.zeros((0, 2))

    cases = [
        ("inf vs itself", inf_d, inf_d, "0.0"),
        ("inf vs finite", inf_d, fin_d, "inf"),
        ("empty vs empty", empty, empty, "0.0"),
        ("empty vs finite", empty, fin_d, "0.5"),
    ]
    print(f"  {'case':<18}{'bottleneck':>12}{'wasserstein':>14}   correct bottleneck")
    for name, a, b, expected in cases:
        bn = persim.bottleneck(a, b)
        wn = persim.wasserstein(a, b)
        flag = "" if f"{bn}".startswith(expected[:3]) else "   <-- WRONG"
        print(f"  {name:<18}{bn:>12.4f}{wn:>14.4f}   {expected}{flag}")
    print("  => persim silently matches the essential bar away and returns a")
    print("     plausible finite number. core/distances.py must partition on")
    print("     `essential` before delegating (RFC-0001 §9.1).")


if __name__ == "__main__":
    main()

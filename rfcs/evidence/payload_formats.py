#!/usr/bin/env python3
"""Array-payload format comparison — evidence for RFC-0001 D12 (Appendix A.6).

Self-contained: synthetic bars, no dataset dependency, runs in CI's own
environment. Requires numpy (`akriti[io]`); `csv`, `sqlite3` and `tempfile` are
stdlib. Generating the CSV is the slow part (~1M `csv.writer` rows), so budget
a few seconds.

    python rfcs/evidence/payload_formats.py

Authored by the project lead and reported on PR #10; the figures it produced
are Appendix A.6's second table. Committed here verbatim in behaviour — only
formatting changed, to satisfy this repository's `ruff` configuration.

Reports unrounded load times, and A.6 records those rather than the ratios
between them. The ratio is not a stable number: it is a quotient by the
fastest thing in the table, so how the `.npz` baseline is sampled moves it
without anything about the formats changing. Best-of-3 rather than a single
run put that baseline at 0.0083 s here and the same measurement then reads
149x and 99x where a single run read 78x and 42x. The absolute times and the
order of magnitude between them are what survive re-measurement elsewhere.

The `exact` column asserts float64 round-trip and `inf` preservation per
format, so a payload that silently lost precision cannot quietly contribute a
size and a time to the comparison. All three pass — correctness was never the
discriminator here, and the table shows that rather than asserting it in prose.
"""

from __future__ import annotations

import csv
import os
import sqlite3
import tempfile
import time
from collections.abc import Callable

import numpy as np

N_BARS = 1_000_000  # ~1000 diagrams x ~1000 bars: see A.6's first table
REPEATS = 3
SEED = 0

Bars = tuple[np.ndarray, np.ndarray, np.ndarray]


def make_bars(n: int, seed: int = SEED) -> Bars:
    rng = np.random.default_rng(seed)
    dims = rng.integers(0, 3, n).astype(np.int32)
    births = rng.random(n)
    deaths = births + rng.random(n)
    deaths[rng.random(n) < 0.001] = np.inf  # essential bars, ~0.1%
    return dims, births, deaths


def best_of(fn: Callable[[], Bars], repeats: int = REPEATS) -> float:
    """Fastest of `repeats` runs — least contaminated by scheduling noise."""
    return min(_timed(fn) for _ in range(repeats))


def _timed(fn: Callable[[], Bars]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def bench_npz(
    path: str, dims: np.ndarray, births: np.ndarray, deaths: np.ndarray
) -> tuple[float, Bars]:
    np.savez(path, dims=dims, births=births, deaths=deaths)

    def load() -> Bars:
        z = np.load(path)
        return z["dims"], z["births"], z["deaths"]

    return best_of(load), load()


def bench_csv(
    path: str, dims: np.ndarray, births: np.ndarray, deaths: np.ndarray
) -> tuple[float, Bars]:
    # repr() gives the shortest string that round-trips a float64 exactly.
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dim", "birth", "death"])
        for dim, birth, death in zip(
            dims.tolist(), births.tolist(), deaths.tolist(), strict=True
        ):
            w.writerow((dim, repr(birth), repr(death)))

    def load() -> Bars:
        with open(path, newline="") as f:
            r = csv.reader(f)
            next(r)
            rows = [(int(a), float(b), float(c)) for a, b, c in r]
        return (
            np.fromiter((r[0] for r in rows), np.int32, len(rows)),
            np.fromiter((r[1] for r in rows), np.float64, len(rows)),
            np.fromiter((r[2] for r in rows), np.float64, len(rows)),
        )

    return best_of(load), load()


def bench_sqlite(
    path: str, dims: np.ndarray, births: np.ndarray, deaths: np.ndarray
) -> tuple[float, Bars]:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE bars(dim INT, birth REAL, death REAL)")
    con.executemany(
        "INSERT INTO bars VALUES (?,?,?)",
        zip(dims.tolist(), births.tolist(), deaths.tolist(), strict=True),
    )
    con.commit()
    con.close()

    def load() -> Bars:
        con = sqlite3.connect(path)
        rows = con.execute("SELECT dim, birth, death FROM bars").fetchall()
        con.close()
        return (
            np.fromiter((r[0] for r in rows), np.int32, len(rows)),
            np.fromiter((r[1] for r in rows), np.float64, len(rows)),
            np.fromiter((r[2] for r in rows), np.float64, len(rows)),
        )

    return best_of(load), load()


def main() -> None:
    dims, births, deaths = make_bars(N_BARS)
    tmp = tempfile.mkdtemp()
    results = []

    for name, fn in (
        ("bars.npz", bench_npz),
        ("bars.csv", bench_csv),
        ("bars.db", bench_sqlite),
    ):
        path = os.path.join(tmp, name)
        load_s, got = fn(path, dims, births, deaths)
        exact = (
            np.array_equal(got[0], dims)
            and np.array_equal(got[1], births)
            and np.array_equal(got[2], deaths)
        )  # array_equal: inf == inf
        results.append((name, os.path.getsize(path), load_s, exact))

    baseline = results[0][2]
    print(f"{N_BARS:,} bars, best of {REPEATS} loads, seed {SEED}\n")
    print(f"{'payload':10s} {'size':>12s} {'load':>12s} {'vs npz':>8s}  exact")
    for name, size, load_s, exact in results:
        print(
            f"{name:10s} {size / 1e6:9.1f} MB {load_s:10.4f} s "
            f"{load_s / baseline:7.1f}x  {'yes' if exact else 'NO'}"
        )
    print("\nSizes are uncompressed on disk. 'exact' covers float64 round-trip")
    print("and inf preservation; a NO invalidates the row's comparison.")
    print("The 'vs npz' column is baseline-sensitive and does not transfer to")
    print("another machine; A.6 records the absolute times instead.")


if __name__ == "__main__":
    main()

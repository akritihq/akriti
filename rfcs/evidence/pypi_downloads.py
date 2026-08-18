#!/usr/bin/env python3
"""PyPI download volume across Python TDA packages — evidence for RFC-0001 A.8.

    python rfcs/evidence/pypi_downloads.py

Reproduces A.8's table and nothing beyond it. Standard library only, no dataset
dependency; it needs network access to `pypistats.org`, which is the one thing
in Appendix A that does not re-run offline.

What A.8 uses this for is the rank within the general-purpose category, not the
absolute counts. pypistats reports a trailing 30-day window that includes mirror
and CI traffic, so the numbers move between runs and a reader should expect
different absolutes from the ones A.8 recorded on 2026-08-10.

`Kind` is RFC-0001's classification, not pypistats': "general-purpose" means a
pipeline over persistence (filtration, vectorisation, an estimator API), as
against a persistence engine or a consumer of finished diagrams. A.8's claim
depends on that boundary, so it is spelled out rather than implied.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

KIND = {
    "ripser": "persistence backend",
    "persim": "diagram consumer",
    "gudhi": "persistence backend",
    "giotto-tda": "general-purpose",
    "homcloud": "general-purpose",
    "scikit-tda": "general-purpose (meta-package)",
    "teaspoon": "general-purpose",
}

API = "https://pypistats.org/api/packages/{}/recent"

# pypistats rate-limits; without a pause between calls most of them return 429.
DELAY_S = 12.0
RETRIES = 4


def recent_downloads(package: str) -> int | None:
    """Trailing-30-day downloads, or None if the API would not answer."""
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(API.format(package), timeout=20) as response:
                return int(json.load(response)["data"]["last_month"])
        except (urllib.error.URLError, KeyError, ValueError):
            if attempt == RETRIES - 1:
                return None
            time.sleep(DELAY_S * (attempt + 1))
    return None


def main() -> None:
    rows = []
    for package in KIND:
        rows.append((package, recent_downloads(package)))
        time.sleep(DELAY_S)

    rows.sort(key=lambda row: -1 if row[1] is None else -row[1])

    print(f"| {'Package':<12} | Downloads / month | Kind |")
    print("|---|---|---|")
    for package, count in rows:
        shown = "unavailable" if count is None else f"{count:,}"
        print(f"| `{package}` | {shown} | {KIND[package]} |")


if __name__ == "__main__":
    main()

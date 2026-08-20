#!/usr/bin/env python3
"""Where §10.1 requirement 4's obligation actually falls — Appendix A.9.

Self-contained and offline: synthetic bars, `numpy` (`akriti[io]`) plus
`zipfile`, `hashlib` and `tempfile` from the standard library.

    python rfcs/evidence/npz_determinism.py

Requirement 4 says identical diagrams produce identical bytes. An `.akd` is a
zip holding a member that is itself a zip, so the requirement lands on two
archive layers, and neither is free. The five rows below separate them:

  npz-repeat     two `numpy.savez` writes to a file, separated in time
  npz-sink       `numpy.savez` to a seekable buffer vs. straight into the
                 `.akd` member handle -- same arrays, two writers
  akd-writestr   two `ZipFile.writestr(str, data)` writes, separated in time
  akd-pinned     the same, through an explicit `ZipInfo`
  akd-write      `ZipFile.write(path)` on the same bytes at two file modes

What each one is about.

*Timestamps.* `ZipInfo.__init__` defaults `date_time` to `(1980, 1, 1, 0, 0,
0)`, the zip epoch. `ZipFile.open(name, "w")` constructs one and keeps that
default, and `numpy.savez` writes every member through `ZipFile.open` -- which
is why npz-repeat holds without numpy promising anything. `writestr` with a
plain string name is the other path: it calls `ZipInfo._for_archive`, which
stamps the wall clock. `ZipFile.write` reads the source file's own mtime and
mode, so it additionally leaks the umask.

*Seekability.* `_savez` hands each member to `zipf.open(fname, "w",
force_zip64=True)`, which cannot know the length in advance. When the
destination can seek, `zipfile` returns afterwards and patches the real sizes
into the local header. When it cannot -- a pipe, or the `.akd` member handle a
`save()` might stream into -- the sizes go in a trailing data descriptor and
the general-purpose bit that says so is set. Same arrays, same `numpy.savez`,
two byte streams, both valid and both loading correctly.

The consequence for `save()` is that "`bars.npz` is deterministic" is a
property of a pinned writer rather than of the format, and the payload layer
needs pinning as much as the container does.

`SOURCE_DATE_EPOCH` is removed from the environment below, since `_for_archive`
honours it and would make the akd-writestr row pass for a reason that is not
`save()`'s doing.
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import time
import zipfile

import numpy as np

os.environ.pop("SOURCE_DATE_EPOCH", None)  # measure the default, not the CI env

N_BARS = 1000
SEED = 0
# A zip entry stores its timestamp in the DOS format, whose seconds field has
# 2-second resolution. A shorter gap lets two wall-clock writes land in the same
# bucket and report identical bytes for a reason that will not hold at 2.1 s --
# measured at 1.1 s, the akd-writestr row below comes out "yes" about half the
# time. Anything over 2 s makes the comparison say what it means to say.
GAP_S = 2.5
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
# The same instant as a Unix mtime. `ZipInfo.from_file` reads `st_mtime` through
# `time.localtime`, so this has to be the epoch in local time -- a UTC timestamp
# would land in 1979 west of Greenwich and zip rejects anything before 1980.
EPOCH_MTIME = time.mktime((*ZIP_EPOCH, 0, 1, -1))

Bars = tuple[np.ndarray, np.ndarray, np.ndarray]


def make_bars(n: int, seed: int = SEED) -> Bars:
    rng = np.random.default_rng(seed)
    dims = rng.integers(0, 3, n).astype(np.int32)
    births = rng.random(n)
    deaths = births + rng.random(n)
    deaths[rng.random(n) < 0.01] = np.inf  # essential bars, ~1%
    return dims, births, deaths


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def pinned_info(name: str) -> zipfile.ZipInfo:
    """The mechanism §10.1 requirement 4 names for the container."""
    info = zipfile.ZipInfo(name, date_time=ZIP_EPOCH)
    info.compress_type = zipfile.ZIP_STORED
    return info


def write_npz(path: str, bars: Bars) -> None:
    dims, births, deaths = bars
    np.savez(path, births=births, deaths=deaths, dims=dims)


def npz_buffered(bars: Bars) -> bytes:
    """`bars.npz` built in a seekable buffer, then handed over whole."""
    dims, births, deaths = bars
    buf = io.BytesIO()
    np.savez(buf, births=births, deaths=deaths, dims=dims)
    return buf.getvalue()


def npz_streamed(bars: Bars) -> bytes:
    """`bars.npz` written straight into the `.akd` member handle.

    `ZipFile.open(..., "w")` returns a writer that reports `seekable() ==
    False`, so the sizes cannot be patched back into each `.npy` member's local
    header and go in a data descriptor instead.
    """
    dims, births, deaths = bars
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z, z.open(pinned_info("bars.npz"), "w") as fid:
        np.savez(fid, births=births, deaths=deaths, dims=dims)
    with zipfile.ZipFile(io.BytesIO(outer.getvalue())) as z:
        return z.read("bars.npz")


def loads_back(payload: bytes, bars: Bars) -> bool:
    """Both npz forms must be valid npz -- the difference is bytes, not data."""
    dims, births, deaths = bars
    got = np.load(io.BytesIO(payload))
    return (
        np.array_equal(got["births"], births)
        and np.array_equal(got["deaths"], deaths)  # array_equal: inf == inf
        and np.array_equal(got["dims"], dims)
    )


def write_akd_writestr(path: str, payload: bytes, meta: bytes) -> None:
    """The obvious `save()`: member names as plain strings."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("meta.json", meta)
        z.writestr("bars.npz", payload)


def write_akd_pinned(path: str, payload: bytes, meta: bytes) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in (("meta.json", meta), ("bars.npz", payload)):
            z.writestr(pinned_info(name), data)


def write_akd_from_file(path: str, payload: bytes, meta: bytes, mode: int) -> None:
    """`ZipFile.write` on staged files -- the path that copies mtime and mode."""
    staged = tempfile.mkdtemp()
    with zipfile.ZipFile(path, "w") as z:
        for name, data in (("meta.json", meta), ("bars.npz", payload)):
            member = os.path.join(staged, name)
            with open(member, "wb") as f:
                f.write(data)
            os.chmod(member, mode)
            os.utime(member, (EPOCH_MTIME, EPOCH_MTIME))  # only the mode varies
            z.write(member, arcname=name)


def main() -> None:
    bars = make_bars(N_BARS)
    tmp = tempfile.mkdtemp()
    meta = b'{"format":"akriti.diagrams.akd","format_version":0}'
    payload = npz_buffered(bars)  # the .akd rows vary the container only
    rows = []

    a, b = os.path.join(tmp, "a.npz"), os.path.join(tmp, "b.npz")
    write_npz(a, bars)
    time.sleep(GAP_S)
    write_npz(b, bars)
    rows.append(
        (
            "npz-repeat",
            f"savez twice, {GAP_S} s apart",
            digest(read(a)) == digest(read(b)),
        )
    )

    # Not a repeat write: one set of arrays through two destinations. This row
    # says what "bars.npz is deterministic" does and does not cover.
    streamed = npz_streamed(bars)
    rows.append(
        ("npz-sink", "savez seekable vs. streamed", digest(payload) == digest(streamed))
    )

    a, b = os.path.join(tmp, "a1.akd"), os.path.join(tmp, "b1.akd")
    write_akd_writestr(a, payload, meta)
    time.sleep(GAP_S)
    write_akd_writestr(b, payload, meta)
    rows.append(("akd-writestr", f"writestr(str), {GAP_S} s apart", read(a) == read(b)))

    a, b = os.path.join(tmp, "a2.akd"), os.path.join(tmp, "b2.akd")
    write_akd_pinned(a, payload, meta)
    time.sleep(GAP_S)
    write_akd_pinned(b, payload, meta)
    rows.append(("akd-pinned", f"pinned ZipInfo, {GAP_S} s apart", read(a) == read(b)))

    a, b = os.path.join(tmp, "a3.akd"), os.path.join(tmp, "b3.akd")
    write_akd_from_file(a, payload, meta, 0o644)
    write_akd_from_file(b, payload, meta, 0o600)
    rows.append(("akd-write", "ZipFile.write, modes 644/600", read(a) == read(b)))

    print(f"{N_BARS:,} bars, seed {SEED}, numpy {np.__version__}\n")
    print(f"{'archive':14s} {'two writes':30s} identical bytes")
    for name, how, same in rows:
        print(f"{name:14s} {how:30s} {'yes' if same else 'NO'}")

    print(
        f"\nnpz-sink in detail: seekable {len(payload)} bytes, "
        f"streamed {len(streamed)} bytes;"
    )
    print(
        "  both load back to the same arrays:",
        loads_back(payload, bars) and loads_back(streamed, bars),
    )
    for label, data in (("seekable", payload), ("streamed", streamed)):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            info = z.infolist()[0]
            print(
                f"  {label:9s} {info.filename:12s} date_time {info.date_time} "
                f"flag {info.flag_bits:#04x} compress_type {info.compress_type}"
            )
    print("  flag 0x08 is the data descriptor: sizes follow the member, because")
    print("  an unseekable destination cannot have its local header patched.")

    print("\nBoth layers are save()'s obligation, and neither comes free:")
    print("  payload   -- build bars.npz in a seekable buffer, so npz-sink cannot")
    print("               split two conforming writers on the same diagram")
    print("  container -- write both members from an explicit ZipInfo, pinned to")
    print("               the zip epoch and ZIP_STORED, never via ZipFile.write")
    print("ZIP_STORED keeps the archive bytes a function of the payload alone --")
    print("under ZIP_DEFLATED they are a function of the zlib build as well.")


if __name__ == "__main__":
    main()

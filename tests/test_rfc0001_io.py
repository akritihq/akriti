"""RFC-0001 ``.akd`` interchange conformance tests (§§10.1, 10.2, 11.2).

This module deliberately imports :mod:`akriti.diagrams` as a module so every
round trip exercises the same public ``save``/``load`` surface users call.
Implementation-detail dependency tests resolve :mod:`akriti.diagrams.io`
lazily, preserving the package's import-without-NumPy contract.
"""

from __future__ import annotations

import importlib
import io
import json
import re
import struct
import subprocess
import sys
import warnings
import zipfile
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

import akriti.diagrams as diagrams

FORMAT = "akriti.diagrams.akd"
SPEC = "RFC-0001"
SPEC_VERSION = "1.1.0"

_ROOT = Path(__file__).resolve().parents[1]
_RFC_PATH = _ROOT / "rfcs/0001-persistence-diagram-interchange.md"


def make_diagram(
    dims: list[int], births: list[float], deaths: list[float], **meta: Any
) -> Any:
    """Construct a NumPy-backed diagram with the RFC's required dtypes."""
    return diagrams.PersistenceDiagram(
        dims=np.asarray(dims, dtype=np.int32),
        births=np.asarray(births, dtype=np.float64),
        deaths=np.asarray(deaths, dtype=np.float64),
        meta=diagrams.DiagramMeta(**meta),
    )


def rich_diagram() -> Any:
    """A deliberately unsorted diagram covering every bar edge case."""
    return make_diagram(
        [2, 0, 0, 3, 0, 2],
        [4.0, -0.0, 1.5, 7.0, 2.0, 0.0],
        [4.0, np.inf, 1.5, 7.0, 2.0, np.inf],
        filtration="rips",
        backend="ripser",
        backend_version="0.6.12",
        coeff_field=2,
        params={"metric": "euclidean", "nested": {"seed": 7, "exact": True}},
        provenance={
            "essential_bars": "faithful",
            "essential_bars_source": "faithful",
            "source_dtype": "float32",
        },
        description="unsorted fixture — essential, trivial, and repeated bars",
    )


def empty_diagram(**meta: Any) -> Any:
    return make_diagram([], [], [], **meta)


def call_save(obj: Any, path: str | Path) -> None:
    """Call the public API rather than reaching through the I/O module."""
    diagrams.save(obj, path)


def call_load(path: str | Path) -> Any:
    return diagrams.load(path)


def archive_bytes(
    meta: bytes | str,
    *,
    arrays: dict[str, Any] | None = None,
    members: list[tuple[str, bytes]] | None = None,
) -> bytes:
    """Build a small archive for negative and forward-compatibility tests."""
    if isinstance(meta, str):
        meta = meta.encode("utf-8")
    payload = npz_bytes(arrays or {})
    entries = [("meta.json", meta), ("bars.npz", payload)]
    if members:
        entries.extend(members)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return output.getvalue()


def npz_bytes(arrays: dict[str, Any]) -> bytes:
    payload = io.BytesIO()
    np.savez(payload, **arrays)
    return payload.getvalue()


def manual_npz_bytes(arrays: dict[str, Any]) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in ("births", "deaths", "dims", "offsets"):
            if name not in arrays:
                continue
            npy = io.BytesIO()
            np.save(npy, np.asarray(arrays[name]), allow_pickle=False)
            archive.writestr(f"{name}.npy", npy.getvalue())
    return payload.getvalue()


def archive_from_entries(entries: list[tuple[str, bytes]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return output.getvalue()


def compressed_archive_from_entries(
    entries: list[tuple[str, bytes]], compression: int
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return output.getvalue()


def write_bytes(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def envelope(kind: str = "diagram", **extra: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": FORMAT,
        "format_version": 0,
        "spec": SPEC,
        "spec_version": SPEC_VERSION,
        "kind": kind,
    }
    value.update(extra)
    return value


def valid_npz(diagram: Any | None = None, **extra: Any) -> dict[str, Any]:
    d = (diagram or rich_diagram()).canonical()
    value: dict[str, Any] = {
        "births": np.asarray(d.births),
        "deaths": np.asarray(d.deaths),
        "dims": np.asarray(d.dims),
    }
    value.update(extra)
    return value


def valid_meta_json(diagram: Any | None = None, *, kind: str = "diagram") -> str:
    d = diagram or rich_diagram()
    fields = {
        "filtration": d.meta.filtration,
        "backend": d.meta.backend,
        "backend_version": d.meta.backend_version,
        "coeff_field": d.meta.coeff_field,
        "params": dict(d.meta.params),
        "provenance": dict(d.meta.provenance),
        "description": d.meta.description,
    }
    key = "meta" if kind == "diagram" else "metas"
    payload = envelope(kind, **{key: fields if kind == "diagram" else [fields]})
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def valid_batch_json(batch: Any) -> str:
    def fields(meta: Any) -> dict[str, Any]:
        return {
            "filtration": meta.filtration,
            "backend": meta.backend,
            "backend_version": meta.backend_version,
            "coeff_field": meta.coeff_field,
            "params": dict(meta.params),
            "provenance": dict(meta.provenance),
            "description": meta.description,
        }

    payload = envelope("batch", metas=[fields(meta) for meta in batch.metas])
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def valid_batch_metadata(diagram: Any | None = None) -> dict[str, Any]:
    member = diagram or rich_diagram()
    batch = diagrams.DiagramBatch.from_diagrams([member])
    return json.loads(valid_batch_json(batch))


def saved_members(path: Path) -> tuple[list[str], bytes, bytes]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        return names, archive.read("meta.json"), archive.read("bars.npz")


def saved_infos(path: Path) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(path) as archive:
        return archive.infolist()


def saved_archive_comment(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        return archive.comment


def npz_member_names(payload: bytes) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return archive.namelist()


def npz_infos(payload: bytes) -> tuple[list[zipfile.ZipInfo], bytes]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return archive.infolist(), archive.comment


def npy_dtype_str(payload: bytes) -> str:
    stream = io.BytesIO(payload)
    version = np.lib.format.read_magic(stream)
    read_header = {
        (1, 0): np.lib.format.read_array_header_1_0,
        (2, 0): np.lib.format.read_array_header_2_0,
        (3, 0): np.lib.format.read_array_header_2_0,
    }[version]
    _, _, dtype = read_header(stream)
    return np.dtype(dtype).str


def npy_v3_bytes(array: Any) -> bytes:
    """Encode an NPY 3.0 member using NumPy's public header writer."""
    value = np.asarray(array)
    header = io.BytesIO()
    np.lib.format.write_array_header_2_0(
        header, np.lib.format.header_data_from_array_1_0(value)
    )
    encoded_header = bytearray(header.getvalue())
    encoded_header[6:8] = b"\x03\x00"
    return bytes(encoded_header) + value.tobytes(order="C")


def patch_zip_member_fields(
    payload: bytes,
    member: str,
    *,
    compression: int | None = None,
    flags: int | None = None,
    crc: int | None = None,
) -> bytes:
    """Patch local and central ZIP headers for one intentionally bad member."""
    result = bytearray(payload)
    member_bytes = member.encode("utf-8")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        info = archive.getinfo(member)
    local = info.header_offset
    name_length, _extra_length = struct.unpack_from("<HH", result, local + 26)
    assert bytes(result[local + 30 : local + 30 + name_length]) == member_bytes
    if compression is not None:
        struct.pack_into("<H", result, local + 8, compression)
    if flags is not None:
        struct.pack_into("<H", result, local + 6, flags)
    if crc is not None:
        struct.pack_into("<I", result, local + 14, crc)

    central = result.find(b"PK\x01\x02", local + 1)
    while central >= 0:
        central_name_length, central_extra_length, comment_length = struct.unpack_from(
            "<HHH", result, central + 28
        )
        name_start = central + 46
        name_end = name_start + central_name_length
        if bytes(result[name_start:name_end]) == member_bytes:
            if compression is not None:
                struct.pack_into("<H", result, central + 10, compression)
            if flags is not None:
                struct.pack_into("<H", result, central + 8, flags)
            if crc is not None:
                struct.pack_into("<I", result, central + 16, crc)
            break
        central = result.find(
            b"PK\x01\x02", name_end + central_extra_length + comment_length
        )
    else:
        raise AssertionError(f"central directory lacks {member!r}")
    return bytes(result)


def corrupt_zip_member_data(payload: bytes, member: str) -> bytes:
    """Flip compressed member data while leaving its ZIP metadata intact."""
    result = bytearray(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        info = archive.getinfo(member)
    name_length, extra_length = struct.unpack_from(
        "<HH", result, info.header_offset + 26
    )
    data_start = info.header_offset + 30 + name_length + extra_length
    assert info.compress_size > 0
    result[data_start + info.compress_size // 2] ^= 0xFF
    return bytes(result)


def strict_diagram() -> Any:
    xps = pytest.importorskip("array_api_strict")
    return diagrams.PersistenceDiagram(
        dims=xps.asarray([1, 0], dtype=xps.int32),
        births=xps.asarray([2.0, 0.0], dtype=xps.float64),
        deaths=xps.asarray([3.0, np.inf], dtype=xps.float64),
        meta=diagrams.DiagramMeta(
            filtration="rips",
            backend="strict-fixture",
            params={"namespace": "array_api_strict"},
            provenance={
                "essential_bars": "faithful",
                "essential_bars_source": "faithful",
            },
            description="strict namespace",
        ),
    )


def strict_off_host_diagram() -> Any:
    """`strict_diagram()`'s twin, on a device that is not the host.

    `array_api_strict` ships fake devices for exactly this purpose: an array on
    `Device("device1")` is a conforming array-API array whose conversion to
    NumPy raises the way a CUDA tensor's does, so §10.1's residency clause can
    be exercised here with no GPU present and no optional backend installed.
    The `torch` case runs the same refusal against real device memory.
    """
    xps = pytest.importorskip("array_api_strict")
    device = xps.Device("device1")
    return diagrams.PersistenceDiagram(
        dims=xps.asarray([1, 0], dtype=xps.int32, device=device),
        births=xps.asarray([2.0, 0.0], dtype=xps.float64, device=device),
        deaths=xps.asarray([3.0, np.inf], dtype=xps.float64, device=device),
        meta=diagrams.DiagramMeta(
            filtration="rips",
            backend="strict-fixture",
            params={"namespace": "array_api_strict"},
            description="off-host namespace",
        ),
    )


@composite
def json_value_strategy(draw: Any) -> Any:
    scalar = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(-100, 100),
        st.floats(-100, 100, allow_nan=False, allow_infinity=False),
        st.text(max_size=20),
    )
    return draw(
        st.recursive(
            scalar,
            lambda children: st.one_of(
                st.lists(children, max_size=3),
                st.dictionaries(st.text(max_size=8), children, max_size=3),
            ),
            max_leaves=8,
        )
    )


@composite
def json_mapping_strategy(draw: Any) -> dict[str, Any]:
    return draw(st.dictionaries(st.text(max_size=8), json_value_strategy(), max_size=4))


@composite
def diagram_strategy(draw: Any, metadata: dict[str, Any] | None = None) -> Any:
    count = draw(st.integers(min_value=0, max_value=7))
    births = draw(
        st.lists(
            st.floats(-10, 10, allow_nan=False, allow_infinity=False),
            min_size=count,
            max_size=count,
        )
    )
    lifetimes = draw(
        st.lists(
            st.floats(0, 10, allow_nan=False, allow_infinity=False),
            min_size=count,
            max_size=count,
        )
    )
    dimensions = draw(st.lists(st.integers(0, 4), min_size=count, max_size=count))
    essential = draw(st.lists(st.booleans(), min_size=count, max_size=count))
    deaths = [
        np.inf if is_essential else birth + life
        for birth, life, is_essential in zip(births, lifetimes, essential, strict=True)
    ]
    return make_diagram(
        dimensions,
        births,
        deaths,
        params=metadata or {},
        provenance={
            "essential_bars": "faithful",
            "essential_bars_source": "faithful",
        },
    )


def test_public_surface_exposes_save_and_load_without_collection_imports() -> None:
    assert callable(getattr(diagrams, "save", None))
    assert callable(getattr(diagrams, "load", None))
    assert "save" in diagrams.__all__
    assert "load" in diagrams.__all__


def test_diagram_roundtrip_preserves_bars_and_all_seven_metadata_fields(
    tmp_path: Path,
) -> None:
    """§10.1(1): exact bars, provenance, description, and multiplicity survive."""
    original = rich_diagram()
    path = tmp_path / "rich.akd"
    call_save(original, path)
    loaded = call_load(path)

    assert loaded == original
    assert loaded.same_provenance(original)
    assert loaded.meta == original.meta
    assert loaded.meta.description == original.meta.description
    assert loaded.meta.filtration == original.meta.filtration
    assert loaded.meta.backend == original.meta.backend
    assert loaded.meta.backend_version == original.meta.backend_version
    assert loaded.meta.coeff_field == original.meta.coeff_field
    assert loaded.meta.params == original.meta.params
    assert loaded.meta.provenance == original.meta.provenance
    assert loaded.n_bars == original.n_bars
    assert np.count_nonzero(loaded.essential) == 2
    assert np.count_nonzero((loaded.dims == 0) & (loaded.births == 2.0)) == 1


def test_frozen_ripser_output_roundtrips_through_io(
    ripser_dgms: Any, tmp_path: Path
) -> None:
    """§11.2: a verbatim captured backend result is a real input fixture."""
    original = diagrams.from_ripser(ripser_dgms("circle"), coeff_field=2)
    path = tmp_path / "frozen-ripser.akd"
    call_save(original, path)
    loaded = call_load(path)
    assert loaded == original
    assert loaded.same_provenance(original)


# §11.2's round-trip minimum, mapped onto the GUDHI captures that actually
# carry each property. The mapping is asserted in the test body rather than
# trusted, because a label that drifts from its fixture is a test that reports
# coverage it does not have -- and every case below still round-trips whether
# or not it holds the property its id claims.
_GUDHI_ROUNDTRIP_CASES = [
    # (cloud, full, the §11.2 case, a probe proving this capture is that case)
    ("circle", False, "essential-bars", "essential"),
    ("point", False, "empty-diagram", "empty"),
    ("duplicate", False, "one-degree-empty", "degree-gap"),
    ("duplicate", True, "zero-persistence-bar", "zero-persistence"),
    ("twin_pairs", False, "repeated-identical-bars", "multiplicity"),
]


@pytest.mark.parametrize(
    ("cloud", "full", "case", "probe"),
    _GUDHI_ROUNDTRIP_CASES,
    ids=[case for _, _, case, _ in _GUDHI_ROUNDTRIP_CASES],
)
def test_frozen_gudhi_output_roundtrips_through_io(
    gudhi_pairs: Any, cloud: str, full: bool, case: str, probe: str, tmp_path: Path
) -> None:
    """§11.2's round-trip minimum, over GUDHI, through `save`/`load`.

    The Ripser case above covers one cloud; §11.2 lists four properties and
    names GUDHI alongside Ripser for the essential-bar one. Serialization is
    where multiplicity and `inf` are easiest to lose -- a writer that
    deduplicated, or a reader that dropped a zero-length degree, would pass
    every adapter test and fail only here -- so the list is exercised against
    the format and not only against the adapter.

    Note `duplicate` appears twice, under `full=False` and `full=True`.
    Plain `persistence()` on that cloud returns two H0 bars and nothing in H1,
    which is the empty-in-one-degree case. The `full` capture is
    `persistence(min_persistence=-1.0, persistence_dim_max=True)`, and it is
    `min_persistence` that matters here: GUDHI's default drops bars of zero
    persistence, so `-1.0` is what makes `(0, 0.0, 0.0)` and `(1, 1.0, 1.0)`
    appear at all. They are the genuine zero-persistence bars §11.2 asks for --
    two different captures of one cloud, not one case listed twice.
    """
    original = diagrams.from_gudhi(gudhi_pairs(cloud, full=full), coeff_field=2)
    dims = np.asarray(original.dims)
    births = np.asarray(original.births)
    deaths = np.asarray(original.deaths)

    # The capture really is the case the id claims. Without this the suite
    # would keep reporting five §11.2 properties after a recapture quietly
    # stopped carrying one of them.
    if probe == "essential":
        assert int(np.count_nonzero(np.isinf(deaths))) >= 1
        assert len(set(dims.tolist())) >= 2, "not a multi-degree diagram"
    elif probe == "empty":
        assert original.n_bars == 0
    elif probe == "degree-gap":
        assert original.dim(0).n_bars > 0
        assert original.dim(1).n_bars == 0
    elif probe == "zero-persistence":
        assert int(np.count_nonzero((births == deaths) & np.isfinite(births))) >= 1
    elif probe == "multiplicity":
        rows = list(zip(dims.tolist(), births.tolist(), deaths.tolist(), strict=True))
        assert len(rows) > len(set(rows)), "no repeated identical bars"
    else:  # pragma: no cover - guards the table above against a typo
        raise AssertionError(f"unknown probe {probe!r}")

    path = tmp_path / f"frozen-gudhi-{cloud}-{case}.akd"
    call_save(original, path)
    loaded = call_load(path)

    assert loaded == original
    assert loaded.same_provenance(original)
    assert loaded.n_bars == original.n_bars, "bar count changed -- multiplicity lost"
    assert int(np.count_nonzero(np.asarray(loaded.essential))) == int(
        np.count_nonzero(np.isinf(deaths))
    ), "essential-bar count changed"


def test_array_api_strict_diagram_and_batch_save_load_at_numpy_boundary(
    tmp_path: Path,
) -> None:
    """§3.3/§10.1: save accepts another namespace; load is NumPy-backed."""
    original = strict_diagram()
    batch = diagrams.DiagramBatch.from_diagrams([original])
    diagram_path = tmp_path / "strict.akd"
    batch_path = tmp_path / "strict-batch.akd"
    call_save(original, diagram_path)
    call_save(batch, batch_path)
    loaded = call_load(diagram_path)
    loaded_batch = call_load(batch_path)
    canonical = original.canonical()
    canonical_batch = batch.canonical()
    np.testing.assert_array_equal(np.asarray(loaded.dims), np.asarray(canonical.dims))
    np.testing.assert_array_equal(
        np.asarray(loaded.births), np.asarray(canonical.births)
    )
    np.testing.assert_array_equal(
        np.asarray(loaded.deaths), np.asarray(canonical.deaths)
    )
    assert loaded.xp is np
    assert loaded.same_provenance(original)
    np.testing.assert_array_equal(
        np.asarray(loaded_batch.dims), np.asarray(canonical_batch.dims)
    )
    np.testing.assert_array_equal(
        np.asarray(loaded_batch.births), np.asarray(canonical_batch.births)
    )
    np.testing.assert_array_equal(
        np.asarray(loaded_batch.deaths), np.asarray(canonical_batch.deaths)
    )
    assert loaded_batch.xp is np
    assert loaded_batch.metas == batch.metas
    assert loaded_batch.same_provenance(batch)


# Marked, so the `optional / torch` row selects it. Without the marker this
# test ran nowhere: the default `test` job installs no torch and skips it,
# and the torch row selects on `-m torch` and never saw it. It is the only
# test in the suite that importorskip-ed a backend without the matching
# marker -- checked across every test module, not just this one.
@pytest.mark.torch
def test_torch_autograd_diagram_saves_at_numpy_boundary(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"torch\.asarray: unspecified requires_grad.*"
        )
        diagram = diagrams.PersistenceDiagram(
            dims=torch.tensor([1, 0], dtype=torch.int32),
            births=torch.tensor([2.0, 0.0], dtype=torch.float64, requires_grad=True),
            deaths=torch.tensor(
                [3.0, float("inf")], dtype=torch.float64, requires_grad=True
            ),
            meta=diagrams.DiagramMeta(
                filtration="rips",
                backend="torch-fixture",
                backend_version="2.0",
                coeff_field=2,
                params={"autograd": True},
                provenance={"source": "torch"},
                description="torch autograd boundary",
            ),
        )
    birth_requires_grad = diagram.births.requires_grad
    death_requires_grad = diagram.deaths.requires_grad
    path = tmp_path / "torch-autograd.akd"
    call_save(diagram, path)
    loaded = call_load(path)
    assert loaded.xp is np
    np.testing.assert_array_equal(loaded.dims, np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(loaded.births, np.asarray([0.0, 2.0]))
    np.testing.assert_array_equal(loaded.deaths, np.asarray([np.inf, 3.0]))
    assert loaded.same_provenance(diagram)
    assert diagram.births.requires_grad is birth_requires_grad
    assert diagram.deaths.requires_grad is death_requires_grad


def test_a_device_resident_diagram_is_refused_before_the_destination_is_opened(
    tmp_path: Path,
) -> None:
    """§10.1 requirement 1: `save` MUST raise `ValueError` naming the device
    and the remedy, and MUST make that check *before it opens the
    destination*, so a failed save leaves no partial file.

    The second half is what this test exists to pin. The ordering holds today
    for a reason `save` does not state: `_canonical_arrays` runs while the
    arguments to the `zipfile.ZipFile(...)` call are still being evaluated, so
    the refusal lands with the path untouched. A refactor that moved the
    conversion inside the `with` block -- or that opened the archive first to
    fail fast on an unwritable path -- would leave a truncated `.akd` behind
    and still pass every other test in this file. The assertion is on the
    whole directory rather than on the one path, so a partial write catches
    under any name, a temporary or lock file included.
    """
    diagram = strict_off_host_diagram()
    path = tmp_path / "off-host.akd"

    with pytest.raises(ValueError, match="host-resident") as caught:
        call_save(diagram, path)

    message = str(caught.value)
    assert "device1" in message, "the refusal must name the device"
    assert "host" in message
    assert ".cpu()" in message
    assert "jax.device_get" in message
    assert "from_dlpack" in message
    assert not path.exists()
    assert list(tmp_path.iterdir()) == [], "a refused save touched the destination"


@pytest.mark.torch
def test_a_cuda_diagram_is_refused_rather_than_moved_off_the_device(
    tmp_path: Path,
) -> None:
    """§10.1 requirement 1 against real device memory: a CUDA-resident diagram
    is refused, the caller's tensors stay where the caller put them, and the
    caller's own `.cpu()` is what makes the save go through.

    CUDA specifically, rather than whichever accelerator torch offers: §6.1
    stores births and deaths as `float64`, which MPS does not support, so a
    diagram cannot exist on that device to be refused in the first place.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("array_api_compat")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device is available to place a diagram on")
    device = torch.device("cuda", 0)
    diagram = diagrams.PersistenceDiagram(
        dims=torch.tensor([1, 0], dtype=torch.int32, device=device),
        births=torch.tensor([2.0, 0.0], dtype=torch.float64, device=device),
        deaths=torch.tensor([3.0, float("inf")], dtype=torch.float64, device=device),
        meta=diagrams.DiagramMeta(
            filtration="rips",
            backend="torch-fixture",
            description="cuda residency",
        ),
    )
    path = tmp_path / "cuda.akd"

    with pytest.raises(ValueError, match="host-resident") as caught:
        call_save(diagram, path)

    message = str(caught.value)
    assert "cuda" in message, "the refusal must name the device"
    assert ".cpu()" in message
    assert "jax.device_get" in message
    assert "from_dlpack" in message
    assert not path.exists()
    assert list(tmp_path.iterdir()) == [], "a refused save touched the destination"
    assert diagram.births.device.type == "cuda", "save moved the caller's tensors"

    moved = diagrams.PersistenceDiagram(
        dims=diagram.dims.cpu(),
        births=diagram.births.cpu(),
        deaths=diagram.deaths.cpu(),
        meta=diagram.meta,
    )
    call_save(moved, path)
    loaded = call_load(path)
    assert loaded.xp is np
    np.testing.assert_array_equal(loaded.births, np.asarray([0.0, 2.0]))
    np.testing.assert_array_equal(loaded.deaths, np.asarray([np.inf, 3.0]))


@pytest.mark.parametrize(
    "diagram",
    [
        empty_diagram(),
        make_diagram([0, 2], [1.0, 5.0], [1.0, np.inf]),
        make_diagram([1, 1], [0.0, 0.0], [2.0, 2.0]),
    ],
)
def test_empty_degrees_multiplicity_and_zero_persistence_roundtrip(
    diagram: Any, tmp_path: Path
) -> None:
    path = tmp_path / "edge.akd"
    call_save(diagram, path)
    loaded = call_load(path)
    assert loaded == diagram
    assert loaded.same_provenance(diagram)
    assert loaded.dim(99).n_bars == 0


def test_str_path_and_arbitrary_pathlike_are_accepted(tmp_path: Path) -> None:
    diagram = rich_diagram()
    path = tmp_path / "pathlike.akd"

    call_save(diagram, str(path))
    assert call_load(str(path)) == diagram

    class PathLike:
        def __fspath__(self) -> str:
            return str(path)

    call_save(diagram, PathLike())
    assert call_load(PathLike()) == diagram


def test_save_rejects_objects_outside_the_public_types(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        call_save(object(), tmp_path / "wrong.akd")


def test_repeated_saves_are_byte_identical(tmp_path: Path) -> None:
    diagram = rich_diagram()
    first = tmp_path / "first.akd"
    second = tmp_path / "second.akd"
    call_save(diagram, first)
    call_save(diagram, second)
    assert first.read_bytes() == second.read_bytes()


def test_semantically_equal_row_permutations_have_identical_bytes_and_do_not_mutate(
    tmp_path: Path,
) -> None:
    original = rich_diagram()
    permutation = np.array([4, 1, 5, 0, 3, 2], dtype=np.int64)
    permuted = diagrams.PersistenceDiagram(
        dims=original.dims[permutation],
        births=original.births[permutation],
        deaths=original.deaths[permutation],
        meta=original.meta,
    )
    before = (
        original.dims.copy(),
        original.births.copy(),
        original.deaths.copy(),
    )
    one = tmp_path / "one.akd"
    two = tmp_path / "two.akd"
    call_save(original, one)
    call_save(permuted, two)
    assert one.read_bytes() == two.read_bytes()
    assert np.array_equal(original.dims, before[0])
    assert np.array_equal(original.births, before[1])
    assert np.array_equal(original.deaths, before[2])


def test_signed_zero_is_semantically_deterministic_and_input_signs_are_unchanged(
    tmp_path: Path,
) -> None:
    plus = make_diagram([0, 0], [0.0, 1.0], [0.0, 2.0])
    minus = make_diagram([0, 0], [-0.0, 1.0], [-0.0, 2.0])
    before = (np.signbit(minus.births).copy(), np.signbit(minus.deaths).copy())
    left = tmp_path / "plus.akd"
    right = tmp_path / "minus.akd"
    call_save(plus, left)
    call_save(minus, right)
    assert plus == minus
    assert left.read_bytes() == right.read_bytes()
    assert np.array_equal(np.signbit(minus.births), before[0])
    assert np.array_equal(np.signbit(minus.deaths), before[1])


def test_archive_members_order_and_pinned_inspectable_metadata(tmp_path: Path) -> None:
    diagram = rich_diagram()
    path = tmp_path / "inspect.akd"
    call_save(diagram, path)
    names, metadata_bytes, bars_bytes = saved_members(path)

    assert names == ["meta.json", "bars.npz"]
    assert metadata_bytes.decode("utf-8") == valid_meta_json(diagram).encode(
        "utf-8"
    ).decode("utf-8")
    assert metadata_bytes == json.dumps(
        json.loads(metadata_bytes.decode("utf-8")),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert json.loads(metadata_bytes)["format"] == FORMAT
    assert json.loads(metadata_bytes)["format_version"] == 0
    assert json.loads(metadata_bytes)["spec"] == SPEC
    assert json.loads(metadata_bytes)["spec_version"] == SPEC_VERSION
    assert json.loads(metadata_bytes)["meta"]["description"] == diagram.meta.description
    assert saved_archive_comment(path) == b""
    for info in saved_infos(path):
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.create_system == 3
        assert info.external_attr == (0o600 << 16)
        assert info.flag_bits == 0
        assert info.extra == b""
        assert info.comment == b""
    assert npz_member_names(bars_bytes) == [
        "births.npy",
        "deaths.npy",
        "dims.npy",
    ]
    nested_infos, nested_comment = npz_infos(bars_bytes)
    assert nested_comment == b""
    for info in nested_infos:
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.create_system == 3
        assert info.external_attr == (0o600 << 16)
        assert info.flag_bits == 0
        assert info.extra == b""
        assert info.comment == b""
    with zipfile.ZipFile(io.BytesIO(bars_bytes)) as nested:
        assert {
            name: npy_dtype_str(nested.read(f"{name}.npy"))
            for name in ("births", "deaths", "dims")
        } == {"births": "<f8", "deaths": "<f8", "dims": "<i4"}
    with np.load(io.BytesIO(bars_bytes), allow_pickle=False) as payload:
        assert {"births", "deaths", "dims"} <= set(payload.files)
        canonical = diagram.canonical()
        np.testing.assert_array_equal(payload["dims"], canonical.dims)
        np.testing.assert_array_equal(payload["births"], canonical.births)
        np.testing.assert_array_equal(payload["deaths"], canonical.deaths)
        assert payload["births"].dtype.str == "<f8"
        assert payload["deaths"].dtype.str == "<f8"
        assert payload["dims"].dtype.str == "<i4"


def test_unknown_envelope_key_and_unknown_npz_array_are_ignored(
    tmp_path: Path,
) -> None:
    diagram = rich_diagram()
    metadata = envelope("diagram", meta=json.loads(valid_meta_json(diagram))["meta"])
    metadata["future_advisory"] = {"writer": "next"}
    path = write_bytes(
        tmp_path / "forward.akd",
        archive_bytes(
            json.dumps(metadata, separators=(",", ":")),
            arrays=valid_npz(diagram, future_array=np.arange(2, dtype=np.int16)),
        ),
    )
    loaded = call_load(path)
    assert loaded == diagram
    assert loaded.same_provenance(diagram)


def test_batch_roundtrip_preserves_ragged_order_segments_and_offsets(
    tmp_path: Path,
) -> None:
    first = make_diagram([2, 0], [3.0, 0.0], [4.0, np.inf], description="first")
    empty = empty_diagram(description="empty")
    one = make_diagram([4], [9.0], [9.0], description="one")
    same_a = make_diagram([0], [1.0], [2.0], description="same-a")
    same_b = make_diagram([0], [1.0], [2.0], description="same-b")
    last = make_diagram(
        [1, 0, 1], [2.0, 1.0, 0.0], [3.0, 1.0, np.inf], description="last"
    )
    batch = diagrams.DiagramBatch.from_diagrams(
        [same_b, empty, same_a, last, first, one]
    )
    path = tmp_path / "batch.akd"
    repeated = tmp_path / "batch-repeated.akd"
    call_save(batch, path)
    call_save(batch, repeated)
    loaded = call_load(path)

    assert isinstance(loaded, diagrams.DiagramBatch)
    assert loaded == batch
    assert loaded.same_provenance(batch)
    assert loaded.metas == batch.metas
    assert loaded.offsets.dtype == np.dtype(np.int64)
    assert loaded.offsets.tolist() == [0, 1, 1, 2, 5, 7, 8]
    assert [item.meta.description for item in loaded] == [
        "same-b",
        "empty",
        "same-a",
        "last",
        "first",
        "one",
    ]
    assert loaded[0] == same_b
    assert loaded[1] == empty
    assert loaded[2] == same_a
    assert loaded[3] == last
    assert loaded[4] == first
    assert loaded[5] == one
    assert path.read_bytes() == repeated.read_bytes()

    names, _, bars_bytes = saved_members(path)
    assert names == ["meta.json", "bars.npz"]
    assert saved_archive_comment(path) == b""
    for info in saved_infos(path):
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.create_system == 3
        assert info.external_attr == (0o600 << 16)
        assert info.flag_bits == 0
        assert info.extra == b""
        assert info.comment == b""
    assert npz_member_names(bars_bytes) == [
        "births.npy",
        "deaths.npy",
        "dims.npy",
        "offsets.npy",
    ]
    nested_infos, nested_comment = npz_infos(bars_bytes)
    assert nested_comment == b""
    for info in nested_infos:
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.create_system == 3
        assert info.external_attr == (0o600 << 16)
        assert info.flag_bits == 0
        assert info.extra == b""
        assert info.comment == b""
    with zipfile.ZipFile(io.BytesIO(bars_bytes)) as nested:
        assert {
            name: npy_dtype_str(nested.read(f"{name}.npy"))
            for name in ("births", "deaths", "dims", "offsets")
        } == {
            "births": "<f8",
            "deaths": "<f8",
            "dims": "<i4",
            "offsets": "<i8",
        }
    with np.load(io.BytesIO(bars_bytes), allow_pickle=False) as payload:
        assert payload["offsets"].dtype == np.dtype(np.int64)
        assert payload["offsets"].tolist() == [0, 1, 1, 2, 5, 7, 8]
        for index, expected in enumerate((same_b, empty, same_a, last, first, one)):
            lo, hi = payload["offsets"][index : index + 2]
            canonical = expected.canonical()
            np.testing.assert_array_equal(payload["dims"][lo:hi], canonical.dims)
            np.testing.assert_array_equal(payload["births"][lo:hi], canonical.births)
            np.testing.assert_array_equal(payload["deaths"][lo:hi], canonical.deaths)


def test_load_accepts_big_endian_required_diagram_arrays(tmp_path: Path) -> None:
    expected = rich_diagram().canonical()
    arrays = {
        "births": np.asarray(expected.births, dtype=">f8"),
        "deaths": np.asarray(expected.deaths, dtype=">f8"),
        "dims": np.asarray(expected.dims, dtype=">i4"),
    }
    path = write_bytes(
        tmp_path / "big-endian-diagram.akd",
        archive_from_entries(
            [
                ("meta.json", valid_meta_json(expected).encode("utf-8")),
                ("bars.npz", manual_npz_bytes(arrays)),
            ]
        ),
    )
    loaded = call_load(path)
    assert loaded == expected
    assert loaded.xp is np
    assert loaded.births.dtype.isnative
    assert loaded.deaths.dtype.isnative
    assert loaded.dims.dtype.isnative


def test_load_accepts_numpy_npy_format_v3_members(tmp_path: Path) -> None:
    expected = rich_diagram().canonical()
    nested_output = io.BytesIO()
    with zipfile.ZipFile(nested_output, "w", compression=zipfile.ZIP_STORED) as nested:
        for name, values in (
            ("births", expected.births),
            ("deaths", expected.deaths),
            ("dims", expected.dims),
        ):
            nested.writestr(f"{name}.npy", npy_v3_bytes(values))
    with zipfile.ZipFile(io.BytesIO(nested_output.getvalue())) as nested:
        for name in ("births", "deaths", "dims"):
            assert np.lib.format.read_magic(io.BytesIO(nested.read(f"{name}.npy"))) == (
                3,
                0,
            )
    path = write_bytes(
        tmp_path / "npy-v3.akd",
        archive_from_entries(
            [
                ("meta.json", valid_meta_json(expected).encode("utf-8")),
                ("bars.npz", nested_output.getvalue()),
            ]
        ),
    )
    loaded = call_load(path)
    assert loaded == expected
    assert loaded.xp is np


def test_load_accepts_numpy_npy_format_v3_batch_members(tmp_path: Path) -> None:
    expected = diagrams.DiagramBatch.from_diagrams(
        [rich_diagram(), make_diagram([0], [1.0], [np.inf], description="second")]
    ).canonical()
    nested_output = io.BytesIO()
    with zipfile.ZipFile(nested_output, "w", compression=zipfile.ZIP_STORED) as nested:
        for name, values in (
            ("births", expected.births),
            ("deaths", expected.deaths),
            ("dims", expected.dims),
            ("offsets", expected.offsets),
        ):
            nested.writestr(f"{name}.npy", npy_v3_bytes(values))
    with zipfile.ZipFile(io.BytesIO(nested_output.getvalue())) as nested:
        for name in ("births", "deaths", "dims", "offsets"):
            assert np.lib.format.read_magic(io.BytesIO(nested.read(f"{name}.npy"))) == (
                3,
                0,
            )
    path = write_bytes(
        tmp_path / "npy-v3-batch.akd",
        archive_from_entries(
            [
                ("meta.json", valid_batch_json(expected).encode("utf-8")),
                ("bars.npz", nested_output.getvalue()),
            ]
        ),
    )
    loaded = call_load(path)
    assert isinstance(loaded, diagrams.DiagramBatch)
    assert loaded == expected
    assert loaded.xp is np


def test_load_accepts_big_endian_required_batch_arrays(tmp_path: Path) -> None:
    expected = diagrams.DiagramBatch.from_diagrams(
        [rich_diagram(), make_diagram([0], [1.0], [np.inf], description="second")]
    ).canonical()
    arrays = {
        "births": np.asarray(expected.births, dtype=">f8"),
        "deaths": np.asarray(expected.deaths, dtype=">f8"),
        "dims": np.asarray(expected.dims, dtype=">i4"),
        "offsets": np.asarray(expected.offsets, dtype=">i8"),
    }
    path = write_bytes(
        tmp_path / "big-endian-batch.akd",
        archive_from_entries(
            [
                ("meta.json", valid_batch_json(expected).encode("utf-8")),
                ("bars.npz", manual_npz_bytes(arrays)),
            ]
        ),
    )
    loaded = call_load(path)
    assert isinstance(loaded, diagrams.DiagramBatch)
    assert loaded == expected
    assert loaded.xp is np
    assert loaded.births.dtype.isnative
    assert loaded.deaths.dtype.isnative
    assert loaded.dims.dtype.isnative
    assert loaded.offsets.dtype.isnative


def test_empty_batch_requires_explicit_numpy_namespace_and_roundtrips(
    tmp_path: Path,
) -> None:
    batch = diagrams.DiagramBatch.from_diagrams([], xp=np)
    path = tmp_path / "empty-batch.akd"
    call_save(batch, path)
    loaded = call_load(path)
    assert isinstance(loaded, diagrams.DiagramBatch)
    assert loaded == batch
    assert loaded.same_provenance(batch)
    assert loaded.offsets.dtype == np.dtype(np.int64)
    assert loaded.offsets.tolist() == [0]


def test_kind_dispatch_distinguishes_diagram_from_length_one_batch(
    tmp_path: Path,
) -> None:
    member = make_diagram([0], [0.0], [np.inf], description="member")
    single = diagrams.DiagramBatch.from_diagrams([member])
    diagram_path = tmp_path / "diagram.akd"
    batch_path = tmp_path / "length-one-batch.akd"
    call_save(member, diagram_path)
    call_save(single, batch_path)
    loaded_diagram = call_load(diagram_path)
    loaded_batch = call_load(batch_path)
    assert type(loaded_diagram) is diagrams.PersistenceDiagram
    assert type(loaded_batch) is diagrams.DiagramBatch
    assert loaded_batch == single
    assert len(loaded_batch) == 1


@pytest.mark.parametrize(
    ("metadata", "arrays", "members", "error_match"),
    [
        (b"not-json", valid_npz(), None, r"JSON"),
        (b"\xff", valid_npz(), None, r"UTF-8"),
        (
            json.dumps(envelope("diagram", meta={})),
            valid_npz(),
            [("wrong", b"x")],
            r"meta\.json|bars\.npz|member|archive",
        ),
    ],
)
def test_malformed_outer_members_and_json_are_rejected(
    tmp_path: Path,
    metadata: bytes,
    arrays: dict[str, Any],
    members: list[tuple[str, bytes]] | None,
    error_match: str,
) -> None:
    path = write_bytes(
        tmp_path / "bad.akd", archive_bytes(metadata, arrays=arrays, members=members)
    )
    with pytest.raises(ValueError, match=error_match):
        call_load(path)


def test_non_zip_input_is_rejected_as_a_value_error(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "not-zip.akd", b"this is not a zip archive")
    with pytest.raises(ValueError, match=r"ZIP|zip|archive"):
        call_load(path)


def test_unsupported_outer_compression_is_normalized_to_value_error(
    tmp_path: Path,
) -> None:
    archive = archive_bytes(valid_meta_json(), arrays=valid_npz())
    archive = patch_zip_member_fields(archive, "meta.json", compression=99)
    path = write_bytes(tmp_path / "unsupported-outer-compression.akd", archive)
    with pytest.raises(ValueError, match=r"compression|archive|ZIP|meta"):
        call_load(path)


def test_corrupt_inner_crc_is_normalized_to_value_error(tmp_path: Path) -> None:
    nested = patch_zip_member_fields(
        npz_bytes(valid_npz()), "births.npy", crc=0x12345678
    )
    path = write_bytes(
        tmp_path / "corrupt-inner-crc.akd",
        archive_from_entries(
            [
                ("meta.json", valid_meta_json().encode("utf-8")),
                ("bars.npz", nested),
            ]
        ),
    )
    with pytest.raises(ValueError, match=r"CRC|crc|bars\.npz|archive|payload"):
        call_load(path)


@pytest.mark.parametrize(
    ("compression", "compression_name"),
    [
        (zipfile.ZIP_DEFLATED, "deflated"),
        (zipfile.ZIP_LZMA, "lzma"),
    ],
)
def test_corrupt_compressed_inner_member_is_normalized_to_value_error(
    tmp_path: Path, compression: int, compression_name: str
) -> None:
    nested_output = io.BytesIO()
    try:
        with zipfile.ZipFile(nested_output, "w", compression=compression) as nested:
            for name, values in valid_npz().items():
                member = io.BytesIO()
                np.save(member, values, allow_pickle=False)
                nested.writestr(f"{name}.npy", member.getvalue())
    except (NotImplementedError, RuntimeError) as error:
        pytest.skip(f"{compression_name} compression backend unavailable: {error}")
    corrupted = corrupt_zip_member_data(nested_output.getvalue(), "births.npy")
    path = write_bytes(
        tmp_path / f"corrupt-{compression_name}.akd",
        archive_from_entries(
            [
                ("meta.json", valid_meta_json().encode("utf-8")),
                ("bars.npz", corrupted),
            ]
        ),
    )
    with pytest.raises(ValueError, match=r"decompress|CRC|crc|births|payload|archive"):
        call_load(path)


@pytest.mark.parametrize("member", ["meta.json", "bars.npz"])
def test_corrupt_bzip2_outer_member_is_normalized_to_value_error(
    tmp_path: Path, member: str
) -> None:
    valid = compressed_archive_from_entries(
        [
            ("meta.json", valid_meta_json().encode("utf-8")),
            ("bars.npz", npz_bytes(valid_npz())),
        ],
        zipfile.ZIP_BZIP2,
    )
    path = write_bytes(
        tmp_path / f"corrupt-bzip2-{member.replace('.', '-')}.akd",
        corrupt_zip_member_data(valid, member),
    )
    with pytest.raises(ValueError, match=r"decompress|CRC|crc|member|archive|payload"):
        call_load(path)


def test_corrupt_zstandard_outer_meta_is_normalized_to_value_error(
    tmp_path: Path,
) -> None:
    if not hasattr(zipfile, "ZIP_ZSTANDARD"):
        pytest.skip("ZIP_ZSTANDARD is unavailable")
    try:
        importlib.import_module("compression.zstd")
    except ImportError:
        pytest.skip("compression.zstd is unavailable")
    valid = compressed_archive_from_entries(
        [
            ("meta.json", valid_meta_json().encode("utf-8")),
            ("bars.npz", npz_bytes(valid_npz())),
        ],
        zipfile.ZIP_ZSTANDARD,
    )
    path = write_bytes(
        tmp_path / "corrupt-zstandard-meta.akd",
        corrupt_zip_member_data(valid, "meta.json"),
    )
    with pytest.raises(ValueError, match=r"decompress|CRC|crc|member|archive|payload"):
        call_load(path)


def test_large_corrupt_inner_crc_fails_during_lazy_array_read(
    tmp_path: Path,
) -> None:
    count = 100_000
    births = np.linspace(0.0, 1.0, count, dtype=np.float64)
    arrays = {
        "births": births,
        "deaths": births + 1.0,
        "dims": np.zeros(count, dtype=np.int32),
    }
    nested = manual_npz_bytes(arrays)
    with zipfile.ZipFile(io.BytesIO(nested)) as archive:
        assert archive.getinfo("births.npy").file_size > (1 << 19)
    nested = patch_zip_member_fields(nested, "births.npy", crc=0x12345678)
    path = write_bytes(
        tmp_path / "large-corrupt-inner-crc.akd",
        archive_from_entries(
            [
                ("meta.json", valid_meta_json().encode("utf-8")),
                ("bars.npz", nested),
            ]
        ),
    )
    with pytest.raises(ValueError, match=r"CRC|crc|births|bars\.npz|archive|payload"):
        call_load(path)


@pytest.mark.parametrize(
    ("entries", "error_match"),
    [
        ([], r"meta\.json|bars\.npz|member|archive"),
        ([("meta.json", b"{}")], r"bars\.npz|member|archive"),
        (
            [("bars.npz", b""), ("meta.json", b"{}")],
            r"order|meta\.json|bars\.npz|member|archive",
        ),
        ([("meta.json", b"{}"), ("wrong.json", b"{}")], r"meta\.json|member|archive"),
    ],
)
def test_missing_reordered_or_misnamed_outer_members_are_rejected(
    tmp_path: Path, entries: list[tuple[str, bytes]], error_match: str
) -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in entries:
            archive.writestr(name, data)
    path = write_bytes(tmp_path / "outer-members.akd", output.getvalue())
    with pytest.raises(ValueError, match=error_match):
        call_load(path)


def test_wrong_outer_order_rejects_an_otherwise_valid_archive(tmp_path: Path) -> None:
    metadata = valid_meta_json().encode("utf-8")
    arrays = npz_bytes(valid_npz())
    path = write_bytes(
        tmp_path / "wrong-order.akd",
        archive_from_entries([("bars.npz", arrays), ("meta.json", metadata)]),
    )
    with pytest.raises(ValueError, match=r"order|meta\.json|bars\.npz|member"):
        call_load(path)


def test_malformed_binary_payload_is_rejected(tmp_path: Path) -> None:
    metadata = json.dumps(
        envelope("diagram", meta=json.loads(valid_meta_json())["meta"])
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("meta.json", metadata)
        archive.writestr("bars.npz", b"not an npz")
    path = write_bytes(tmp_path / "bad-payload.akd", output.getvalue())
    with pytest.raises(ValueError, match=r"bars\.npz|npz|payload|archive"):
        call_load(path)


def test_duplicate_required_nested_npz_member_is_rejected(tmp_path: Path) -> None:
    diagram = rich_diagram().canonical()
    npy_buffers: dict[str, bytes] = {}
    for name, values in (
        ("births.npy", np.asarray(diagram.births)),
        ("deaths.npy", np.asarray(diagram.deaths)),
        ("dims.npy", np.asarray(diagram.dims)),
    ):
        payload = io.BytesIO()
        np.save(payload, values)
        npy_buffers[name] = payload.getvalue()
    nested_output = io.BytesIO()
    with zipfile.ZipFile(nested_output, "w", compression=zipfile.ZIP_STORED) as nested:
        nested.writestr("births.npy", npy_buffers["births.npy"])
        duplicate_births = io.BytesIO()
        np.save(duplicate_births, np.asarray(diagram.births))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            nested.writestr("births.npy", duplicate_births.getvalue())
        nested.writestr("deaths.npy", npy_buffers["deaths.npy"])
        nested.writestr("dims.npy", npy_buffers["dims.npy"])
    path = write_bytes(
        tmp_path / "duplicate-required-member.akd",
        archive_from_entries(
            [
                ("meta.json", valid_meta_json().encode("utf-8")),
                ("bars.npz", nested_output.getvalue()),
            ]
        ),
    )
    with pytest.raises(ValueError, match=r"duplicate"):
        call_load(path)


def test_oversized_npy_header_is_rejected_before_array_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays = valid_npz()
    births_header = io.BytesIO()
    np.lib.format.write_array_header_2_0(
        births_header,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(np.float64)),
            "fortran_order": False,
            "shape": (1 << 50,),
        },
    )
    nested_output = io.BytesIO()
    with zipfile.ZipFile(nested_output, "w", compression=zipfile.ZIP_STORED) as nested:
        nested.writestr("births.npy", births_header.getvalue())
        for name in ("deaths", "dims"):
            payload = io.BytesIO()
            np.save(payload, np.asarray(arrays[name]))
            nested.writestr(f"{name}.npy", payload.getvalue())
    with zipfile.ZipFile(io.BytesIO(nested_output.getvalue())) as nested:
        assert nested.getinfo("births.npy").file_size < 1024
    path = write_bytes(
        tmp_path / "oversized-header.akd",
        archive_from_entries(
            [
                ("meta.json", valid_meta_json().encode("utf-8")),
                ("bars.npz", nested_output.getvalue()),
            ]
        ),
    )

    called = False

    def forbid_array_load(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("NumPy array load occurred before NPY size validation")

    monkeypatch.setattr(np, "load", forbid_array_load)
    with pytest.raises(ValueError, match=r"size|header|length|payload|births"):
        call_load(path)
    assert not called


def test_deep_metadata_recursion_is_normalized_to_value_error(tmp_path: Path) -> None:
    depth = 1500
    deeply_nested = '{"x":' * depth + "0" + "}" * depth
    metadata = (
        '{"format":"akriti.diagrams.akd","format_version":0,'
        '"spec":"RFC-0001","spec_version":"1.1.0","kind":"diagram",'
        '"meta":{"filtration":null,"backend":null,"backend_version":null,'
        '"coeff_field":null,"params":{"deep":'
        + deeply_nested
        + '},"provenance":{},"description":null}}'
    )
    path = write_bytes(
        tmp_path / "deep-metadata.akd",
        archive_bytes(metadata, arrays=valid_npz()),
    )
    with pytest.raises(ValueError, match=r"JSON|metadata|nested|recursion|depth"):
        call_load(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("format", "wrong"),
        ("spec", "RFC-9999"),
        ("format_version", 99),
        ("spec_version", [0, 1, 0]),
        ("kind", "neither"),
    ],
)
def test_unsupported_envelope_identifiers_are_rejected(
    tmp_path: Path, field: str, value: Any
) -> None:
    diagram = rich_diagram()
    metadata = envelope("diagram", meta=json.loads(valid_meta_json(diagram))["meta"])
    metadata[field] = value
    path = write_bytes(
        tmp_path / f"bad-{field}.akd",
        archive_bytes(json.dumps(metadata), arrays=valid_npz(diagram)),
    )
    with pytest.raises(ValueError, match=field):
        call_load(path)


def test_spec_version_is_audit_metadata_not_a_dispatch_key(tmp_path: Path) -> None:
    diagram = rich_diagram()
    metadata = json.loads(valid_meta_json(diagram))
    metadata["spec_version"] = "9.9.9"
    path = write_bytes(
        tmp_path / "new-spec-revision.akd",
        archive_bytes(json.dumps(metadata), arrays=valid_npz(diagram)),
    )
    loaded = call_load(path)
    assert loaded == diagram
    assert loaded.same_provenance(diagram)


@pytest.mark.parametrize("root", ["[]", "1", '"diagram"'])
def test_json_root_must_be_an_object(tmp_path: Path, root: str) -> None:
    path = write_bytes(
        tmp_path / "json-root.akd", archive_bytes(root, arrays=valid_npz())
    )
    with pytest.raises(ValueError, match=r"JSON|object|envelope"):
        call_load(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("format", None),
        ("format_version", "0"),
        ("spec", None),
        ("spec_version", None),
        ("kind", None),
    ],
)
def test_required_envelope_fields_must_have_schema_types(
    tmp_path: Path, field: str, value: Any
) -> None:
    metadata = json.loads(valid_meta_json())
    metadata[field] = value
    path = write_bytes(
        tmp_path / "wrong-envelope-type.akd",
        archive_bytes(json.dumps(metadata), arrays=valid_npz()),
    )
    with pytest.raises(ValueError, match=field):
        call_load(path)


@pytest.mark.parametrize(
    "field", ["format", "format_version", "spec", "spec_version", "kind"]
)
def test_required_envelope_fields_must_be_present(tmp_path: Path, field: str) -> None:
    metadata = json.loads(valid_meta_json())
    metadata.pop(field)
    path = write_bytes(
        tmp_path / "missing-envelope-field.akd",
        archive_bytes(json.dumps(metadata), arrays=valid_npz()),
    )
    with pytest.raises(ValueError, match=field):
        call_load(path)


@pytest.mark.parametrize(
    ("metadata", "arrays"),
    [
        (
            envelope("diagram", meta=json.loads(valid_meta_json())["meta"], metas=[]),
            valid_npz(),
        ),
        (
            {**valid_batch_metadata(), "meta": json.loads(valid_meta_json())["meta"]},
            valid_npz(offsets=np.asarray([0, 6], dtype=np.int64)),
        ),
        (
            envelope("diagram", meta=json.loads(valid_meta_json())["meta"]),
            valid_npz(offsets=np.asarray([0, 6], dtype=np.int64)),
        ),
        (
            valid_batch_metadata(),
            valid_npz(),
        ),
    ],
)
def test_meta_metas_and_kind_payload_disagreements_are_rejected(
    tmp_path: Path, metadata: dict[str, Any], arrays: dict[str, Any]
) -> None:
    path = write_bytes(
        tmp_path / "disagreement.akd",
        archive_bytes(json.dumps(metadata), arrays=arrays),
    )
    with pytest.raises(ValueError, match=r"meta|metas|offsets|kind|payload"):
        call_load(path)


@pytest.mark.parametrize("meta_value", [None, [], "diagram"])
def test_diagram_meta_is_required_and_must_be_an_object(
    tmp_path: Path, meta_value: Any
) -> None:
    metadata = envelope("diagram", meta=meta_value)
    path = write_bytes(
        tmp_path / "bad-diagram-meta.akd",
        archive_bytes(json.dumps(metadata), arrays=valid_npz()),
    )
    with pytest.raises(ValueError, match=r"meta|object|diagram"):
        call_load(path)


def test_diagram_meta_must_be_present(tmp_path: Path) -> None:
    path = write_bytes(
        tmp_path / "missing-diagram-meta.akd",
        archive_bytes(json.dumps(envelope("diagram")), arrays=valid_npz()),
    )
    with pytest.raises(ValueError, match=r"meta|diagram"):
        call_load(path)


@pytest.mark.parametrize("metas_value", [None, {}, "batch"])
def test_batch_metas_are_required_and_must_be_an_array(
    tmp_path: Path, metas_value: Any
) -> None:
    metadata = envelope("batch", metas=metas_value)
    path = write_bytes(
        tmp_path / "bad-batch-metas.akd",
        archive_bytes(
            json.dumps(metadata),
            arrays=valid_npz(offsets=np.asarray([0, 6], dtype=np.int64)),
        ),
    )
    with pytest.raises(ValueError, match=r"metas|array|batch"):
        call_load(path)


def test_batch_metas_must_be_present(tmp_path: Path) -> None:
    path = write_bytes(
        tmp_path / "missing-batch-metas.akd",
        archive_bytes(
            json.dumps(envelope("batch")),
            arrays=valid_npz(offsets=np.asarray([0, 6], dtype=np.int64)),
        ),
    )
    with pytest.raises(ValueError, match=r"metas|batch"):
        call_load(path)


@pytest.mark.parametrize("item", [None, [], "metadata", 7])
def test_every_batch_metas_item_must_be_a_json_object(
    tmp_path: Path, item: Any
) -> None:
    metadata = envelope("batch", metas=[item])
    path = write_bytes(
        tmp_path / "non-object-meta-item.akd",
        archive_bytes(
            json.dumps(metadata),
            arrays=valid_npz(offsets=np.asarray([0, 6], dtype=np.int64)),
        ),
    )
    with pytest.raises(ValueError, match=r"metas|object|DiagramMeta"):
        call_load(path)


@pytest.mark.parametrize("kind", ["diagram", "batch"])
def test_unknown_nested_diagram_meta_field_is_rejected(
    tmp_path: Path, kind: str
) -> None:
    if kind == "diagram":
        metadata = json.loads(valid_meta_json())
        metadata["meta"]["not_a_diagram_meta_field"] = True
        arrays = valid_npz()
    else:
        metadata = valid_batch_metadata()
        metadata["metas"][0]["not_a_diagram_meta_field"] = True
        arrays = valid_npz(offsets=np.asarray([0, 6], dtype=np.int64))
    path = write_bytes(
        tmp_path / f"unknown-{kind}-meta-field.akd",
        archive_bytes(json.dumps(metadata), arrays=arrays),
    )
    with pytest.raises(ValueError, match=r"meta|field|unknown"):
        call_load(path)


@pytest.mark.parametrize("location", ["top-level", "meta"])
def test_duplicate_json_object_keys_are_rejected(tmp_path: Path, location: str) -> None:
    valid = json.loads(valid_meta_json())
    meta_text = json.dumps(
        valid["meta"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    if location == "meta":
        duplicate_meta = meta_text.replace(
            '"filtration":', '"filtration":"duplicate","filtration":', 1
        )
        metadata = (
            '{"format":"akriti.diagrams.akd","format_version":0,"kind":"diagram",'
            f'"meta":{duplicate_meta},"spec":"RFC-0001","spec_version":"1.1.0"}}'
        )
    else:
        metadata = (
            '{"format":"akriti.diagrams.akd","format_version":0,"kind":"diagram",'
            f'"meta":{meta_text},"meta":{meta_text},"spec":"RFC-0001",'
            '"spec_version":"1.1.0"}'
        )
    path = write_bytes(
        tmp_path / f"duplicate-{location}.akd",
        archive_bytes(metadata, arrays=valid_npz()),
    )
    with pytest.raises(ValueError, match=r"duplicate|JSON|key"):
        call_load(path)


@pytest.mark.parametrize(
    "arrays",
    [
        {"births": np.asarray([0.0]), "deaths": np.asarray([1.0])},
        {"births": np.asarray([0.0]), "dims": np.asarray([0], dtype=np.int32)},
        {"deaths": np.asarray([1.0]), "dims": np.asarray([0], dtype=np.int32)},
        {
            "births": np.asarray([0.0]),
            "deaths": np.asarray([1.0]),
            "dims": np.asarray([0], dtype=np.int64),
        },
    ],
)
def test_required_payload_arrays_and_dtypes_are_rejected(
    tmp_path: Path, arrays: dict[str, Any]
) -> None:
    metadata = envelope("diagram", meta=json.loads(valid_meta_json())["meta"])
    path = write_bytes(
        tmp_path / "missing-array.akd",
        archive_bytes(json.dumps(metadata), arrays=arrays),
    )
    with pytest.raises(ValueError, match=r"births|deaths|dims|required|dtype"):
        call_load(path)


@pytest.mark.parametrize(
    ("arrays", "error_match"),
    [
        (
            {
                **valid_npz(),
                "births": np.asarray(valid_npz()["births"], dtype=np.float32),
            },
            r"births|float64|dtype",
        ),
        (
            {
                **valid_npz(),
                "deaths": np.asarray(valid_npz()["deaths"], dtype=np.float32),
            },
            r"deaths|float64|dtype",
        ),
        (
            {**valid_npz(), "dims": np.asarray(valid_npz()["dims"], dtype=np.int64)},
            r"dims|int32|dtype",
        ),
        (
            {**valid_npz(), "births": valid_npz()["births"].reshape(6, 1)},
            r"births|rank",
        ),
        ({**valid_npz(), "deaths": valid_npz()["deaths"][:-1]}, r"deaths|length"),
        ({**valid_npz(), "dims": valid_npz()["dims"][:-1]}, r"dims|length"),
        (
            {
                **valid_npz(),
                "births": np.array(
                    [*valid_npz()["births"][:1], np.nan, *valid_npz()["births"][2:]]
                ),
            },
            r"births|NaN",
        ),
        (
            {**valid_npz(), "deaths": np.array([np.nan, *valid_npz()["deaths"][1:]])},
            r"deaths|NaN",
        ),
        (
            {**valid_npz(), "deaths": np.array([-np.inf, *valid_npz()["deaths"][1:]])},
            r"deaths|-inf|finite",
        ),
        (
            {
                **valid_npz(),
                "deaths": np.array([-999.0, *valid_npz()["deaths"][1:]]),
            },
            r"death|birth|I6|coordinate",
        ),
    ],
)
def test_payload_shapes_dtypes_and_coordinate_invariants_are_rejected(
    tmp_path: Path, arrays: dict[str, Any], error_match: str
) -> None:
    metadata = envelope("diagram", meta=json.loads(valid_meta_json())["meta"])
    path = write_bytes(
        tmp_path / "bad-payload-schema.akd",
        archive_bytes(json.dumps(metadata), arrays=arrays),
    )
    with pytest.raises(ValueError, match=error_match):
        call_load(path)


@pytest.mark.parametrize(
    ("metadata", "offsets", "error_match"),
    [
        (valid_batch_metadata(), np.asarray([0], dtype=np.int64), r"B1|offsets|length"),
        (valid_batch_metadata(), np.asarray([1, 6], dtype=np.int64), r"B2|offsets|0"),
        (
            valid_batch_metadata(),
            np.asarray([0, 5], dtype=np.int64),
            r"B3|offsets|total",
        ),
        (
            envelope("batch", metas=[*valid_batch_metadata()["metas"]] * 3),
            np.asarray([0, 2, 1, 6], dtype=np.int64),
            r"B4|offsets|non-decreasing",
        ),
        (
            valid_batch_metadata(),
            np.asarray([[0, 6]], dtype=np.int64),
            r"B6|offsets|rank",
        ),
        (
            valid_batch_metadata(),
            np.asarray([0, 6], dtype=np.int32),
            r"B7|offsets|int64",
        ),
    ],
)
def test_batch_offsets_and_invariants_are_rejected(
    tmp_path: Path,
    metadata: dict[str, Any],
    offsets: np.ndarray,
    error_match: str,
) -> None:
    path = write_bytes(
        tmp_path / "bad-offsets.akd",
        archive_bytes(json.dumps(metadata), arrays=valid_npz(offsets=offsets)),
    )
    with pytest.raises(ValueError, match=error_match):
        call_load(path)


def test_outer_archive_must_have_exactly_two_members(tmp_path: Path) -> None:
    diagram = rich_diagram()
    metadata = json.dumps(
        envelope("diagram", meta=json.loads(valid_meta_json(diagram))["meta"])
    )
    path = write_bytes(
        tmp_path / "extra-member.akd",
        archive_bytes(
            metadata, arrays=valid_npz(diagram), members=[("extra", b"ignored")]
        ),
    )
    with pytest.raises(ValueError, match=r"exactly|member|meta\.json|bars\.npz"):
        call_load(path)


def test_importing_diagrams_on_a_bare_path_does_not_import_numpy() -> None:
    code = (
        "import importlib, sys; "
        "importlib.import_module('akriti.diagrams'); "
        "print('numpy' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def _io_module() -> Any:
    """Resolve the optional implementation module only when a test needs it."""
    return importlib.import_module("akriti.diagrams.io")


def reload_io_with_numpy_import(
    monkeypatch: pytest.MonkeyPatch,
    numpy_import: Any,
    version: str | None = None,
) -> Any:
    """Patch stdlib import/metadata machinery, then reload the real I/O module."""
    io_module = _io_module()
    real_import_module = importlib.import_module

    def patched_import(name: str, package: str | None = None) -> Any:
        if name == "numpy":
            return numpy_import()
        if package is None:
            return real_import_module(name)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", patched_import)
    if version is not None:
        monkeypatch.setattr(metadata, "version", lambda name: version)
    io_module = importlib.reload(io_module)
    monkeypatch.setattr(diagrams, "save", io_module.save, raising=False)
    monkeypatch.setattr(diagrams, "load", io_module.load, raising=False)
    return io_module


def invoke_io(operation: str, path: Path) -> Any:
    if operation == "save":
        return call_save(rich_diagram(), path)
    path.write_bytes(archive_bytes(valid_meta_json(), arrays=valid_npz()))
    return call_load(path)


@pytest.mark.parametrize("operation", ["save", "load"])
@pytest.mark.parametrize("version", ["1.26.4", "2.0rc1"])
def test_io_rejects_missing_numpy_floor_and_same_floor_prerelease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
    version: str,
) -> None:
    reload_io_with_numpy_import(monkeypatch, lambda: np, version)
    with pytest.raises(ImportError, match=r"numpy.*2\.0.*akriti\[io\]"):
        invoke_io(operation, tmp_path / "numpy-floor.akd")


@pytest.mark.parametrize("operation", ["save", "load"])
@pytest.mark.parametrize("metadata_error", [metadata.PackageNotFoundError, ValueError])
def test_io_rejects_missing_or_malformed_numpy_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
    metadata_error: type[Exception],
) -> None:
    io_module = _io_module()
    real_import_module = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, package=None: (
            np
            if name == "numpy"
            else real_import_module(name)
            if package is None
            else real_import_module(name, package)
        ),
    )

    def broken_metadata(name: str) -> str:
        raise metadata_error("numpy distribution metadata unavailable")

    monkeypatch.setattr(metadata, "version", broken_metadata)
    io_module = importlib.reload(io_module)
    monkeypatch.setattr(diagrams, "save", io_module.save, raising=False)
    monkeypatch.setattr(diagrams, "load", io_module.load, raising=False)
    with pytest.raises(ImportError, match=r"numpy.*metadata.*akriti\[io\]"):
        invoke_io(operation, tmp_path / "numpy-meta.akd")


@pytest.mark.parametrize("operation", ["save", "load"])
@pytest.mark.parametrize("version", ["development", "2.1_1"])
def test_io_rejects_an_unparsable_numpy_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
    version: str,
) -> None:
    reload_io_with_numpy_import(monkeypatch, lambda: np, version)
    with pytest.raises(ImportError, match=r"parse|version|akriti\[io\]"):
        invoke_io(operation, tmp_path / "numpy-unparsable.akd")


@pytest.mark.parametrize("operation", ["save", "load"])
def test_io_rejects_missing_numpy_with_an_actionable_io_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, operation: str
) -> None:
    missing = ModuleNotFoundError("No module named 'numpy'", name="numpy")

    def missing_numpy() -> Any:
        raise missing

    reload_io_with_numpy_import(monkeypatch, missing_numpy)
    with pytest.raises(ImportError, match=r"akriti\[io\]") as caught:
        invoke_io(operation, tmp_path / "numpy-missing.akd")
    assert caught.value.__cause__ is missing


@pytest.mark.parametrize("operation", ["save", "load"])
def test_io_propagates_transitive_numpy_import_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, operation: str
) -> None:
    failure = ModuleNotFoundError(
        "No module named 'numpy.core._multiarray_umath'",
        name="numpy.core._multiarray_umath",
    )

    def broken_numpy() -> Any:
        raise failure

    reload_io_with_numpy_import(monkeypatch, broken_numpy)
    with pytest.raises(ModuleNotFoundError) as caught:
        invoke_io(operation, tmp_path / "numpy-transitive.akd")
    assert caught.value is failure


@pytest.mark.parametrize("operation", ["save", "load"])
@pytest.mark.parametrize(
    "version",
    [
        "2.0",
        "2.0.0+local",
        "2!2.1rc1",
        "2",
        "2.0.1rc1",
        "1!1.0",
        "2.0-1",
        "2.0_post1",
        "v2.0",
        " 2.0 ",
        "2.0-rev1",
        "2.0-r1",
        "2.1-rc1",
        "2.0.rev1",
        "2.0.r1",
        "2.1-dev1",
        "2.1_dev1",
        "2.1-preview1",
        "2.0post",
        "2.0-post",
        "2.0rev",
        "2.0-r",
        "2.1dev",
        "2.1-dev",
        "2.0-1+local",
        "2.0-rev+local",
        "2.1-rc1+local",
        "2.1-dev+local",
        "2.0.post-1",
        "2.0_rev_1",
        "2.1-rc-1",
        "2.1_dev_1",
        "2.1rc1.post1.dev1+local",
        "2.1rc1-1",
        "2.1-rc1-1",
        "2.1-1-dev1",
        "2.1rc1-1-dev1",
        "2.0postdev",
        "2.1revdev",
        "2.1rdev",
        "2.1rc1rdev+local",
        "v2.1_rc-1_rev_dev+LOCAL.1",
    ],
)
def test_io_accepts_supported_pep440_numpy_versions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
    version: str,
) -> None:
    if version in {
        "2.1rc1.post1.dev1+local",
        "v2.1_rc-1_rev_dev+LOCAL.1",
    }:
        try:
            from packaging.version import Version
        except ImportError:
            pass
        else:
            assert Version(version) >= Version("2.0")
    reload_io_with_numpy_import(monkeypatch, lambda: np, version)
    invoke_io(operation, tmp_path / "numpy-supported.akd")


@given(data=st.data())
@settings(max_examples=24, deadline=None)
def test_hypothesis_diagram_roundtrip_preserves_exactness_and_provenance(
    data: Any, tmp_path_factory: Any
) -> None:
    diagram = data.draw(diagram_strategy())
    permutation = data.draw(st.permutations(range(diagram.n_bars)))
    permuted = diagrams.PersistenceDiagram(
        dims=diagram.dims[list(permutation)],
        births=diagram.births[list(permutation)],
        deaths=diagram.deaths[list(permutation)],
        meta=diagram.meta,
    )
    tmp_path = tmp_path_factory.mktemp("diagram-property")
    first = tmp_path / "property.akd"
    second = tmp_path / "property-permuted.akd"
    call_save(diagram, first)
    call_save(permuted, second)
    loaded = call_load(first)
    assert diagram == permuted
    assert loaded == diagram
    assert loaded.same_provenance(diagram)
    assert first.read_bytes() == second.read_bytes()


@given(data=st.data())
@settings(max_examples=20, deadline=None)
def test_hypothesis_json_safe_metadata_and_ragged_empty_batches_roundtrip(
    data: Any, tmp_path_factory: Any
) -> None:
    metadata = data.draw(json_mapping_strategy())
    diagram = data.draw(diagram_strategy(metadata=metadata))
    other = data.draw(diagram_strategy())
    diagrams_to_save = data.draw(
        st.lists(st.sampled_from([diagram, other]), max_size=4)
    )
    batch = diagrams.DiagramBatch.from_diagrams(
        diagrams_to_save, xp=np if not diagrams_to_save else None
    )
    path = tmp_path_factory.mktemp("batch-property") / "batch-property.akd"
    call_save(batch, path)
    loaded = call_load(path)
    assert loaded == batch
    assert loaded.same_provenance(batch)


# ---------------------------------------------------------------------------
# The reader's own defences (§10.2)
# ---------------------------------------------------------------------------
#
# `load` reads a file it did not write, so every branch below is a refusal on
# input a conforming writer never produces. They were the least-covered part of
# this module, which is the wrong thing to leave untested in a parser: a guard
# that has never run is a guard nobody has seen work.
#
# Two kinds are separated deliberately. The refusals reachable from *bytes* are
# driven by bytes, because that is how a real malformed file arrives. The
# re-raise guards can only be reached by an exception the reader does not
# expect, so those are injected -- and what they assert is that the unexpected
# exception is *propagated* rather than relabelled as a malformed archive,
# which is the failure that would send someone debugging their file instead of
# the bug they actually hit.


def test_non_finite_json_constant_is_rejected(tmp_path: Path) -> None:
    """§10.2 pins the JSON dialect: `NaN` and `Infinity` are Python's
    extensions, not JSON, and §8 already requires every metadata value to be
    JSON-representable. A reader that accepted them would admit a `params`
    value no conforming writer can produce and no other reader can parse."""
    metadata = valid_meta_json().replace('"coeff_field":2', '"coeff_field":NaN', 1)
    assert "NaN" in metadata, "the fixture no longer carries coeff_field"
    path = write_bytes(
        tmp_path / "nan-constant.akd",
        archive_bytes(metadata, arrays=valid_npz()),
    )

    with pytest.raises(ValueError, match="non-finite"):
        call_load(path)


def test_meta_missing_a_required_field_is_rejected(tmp_path: Path) -> None:
    """The other half of the unknown-field rule (Appendix B entry 56). §10.2's
    table fixes the field list, so a `meta` short of one is as unreadable as
    one carrying a name the reader does not have -- and silently defaulting it
    would return a diagram whose metadata is less than the file's, which
    §10.1 requirement 1 makes a round-trip failure."""
    metadata = json.loads(valid_meta_json())
    del metadata["meta"]["provenance"]
    path = write_bytes(
        tmp_path / "missing-meta-field.akd",
        archive_bytes(json.dumps(metadata), arrays=valid_npz()),
    )

    with pytest.raises(ValueError, match="missing required field"):
        call_load(path)


@pytest.mark.parametrize("field", ["params", "provenance"])
@pytest.mark.parametrize("value", [[], "text", 3], ids=["list", "str", "int"])
def test_meta_mapping_field_that_is_not_an_object_is_rejected(
    tmp_path: Path, field: str, value: Any
) -> None:
    """§8 types both as `Mapping[str, Any]`. They are also the two open
    extension points entry 56 leans on when it argues nothing is lost by
    rejecting unknown *fields* -- so a file in which either is not an object
    has to be refused here, or the argument that a writer always has somewhere
    to put a new fact stops holding on read."""
    metadata = json.loads(valid_meta_json())
    metadata["meta"][field] = value
    path = write_bytes(
        tmp_path / f"{field}-not-object.akd",
        archive_bytes(json.dumps(metadata), arrays=valid_npz()),
    )

    with pytest.raises(ValueError, match=f"{field} must be an object"):
        call_load(path)


def test_unsupported_npy_version_inside_the_payload_is_rejected(
    tmp_path: Path,
) -> None:
    """§10.2 fixes the payload as `.npz`, whose members this reader parses
    itself before handing anything to NumPy. NPY 1.0, 2.0 and 3.0 are the
    versions that exist; a fourth is either a corrupt header or a format from
    the future, and the preflight names it rather than letting `np.load` fail
    somewhere less legible."""
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_STORED) as nested:
        for name in ("births", "deaths", "dims"):
            npy = io.BytesIO()
            np.save(npy, np.asarray(valid_npz()[name]), allow_pickle=False)
            raw = bytearray(npy.getvalue())
            raw[6:8] = b"\x04\x00"  # major 4, minor 0 -- no such NPY version
            nested.writestr(f"{name}.npy", bytes(raw))
    path = write_bytes(
        tmp_path / "npy-v4.akd",
        archive_from_entries(
            [
                ("meta.json", valid_meta_json().encode()),
                ("bars.npz", payload.getvalue()),
            ]
        ),
    )

    with pytest.raises(ValueError, match="NPY"):
        call_load(path)


def test_an_unexpected_error_from_numpy_is_propagated_not_relabelled(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The re-raise guard in `_payload_arrays`, and the reason it is written as
    a type check rather than a bare `except Exception`.

    The reader translates the errors a malformed payload actually raises --
    `EOFError`, `ValueError`, and zip read errors -- into one message about the
    file. Anything else is a bug in this library or in NumPy, and relabelling
    it "bars.npz payload could not be read" would send someone hunting a
    corrupt file they do not have. So the guard re-raises, and this is the test
    that it does."""
    io_module = _io_module()
    path = tmp_path / "propagate.akd"
    call_save(rich_diagram(), path)

    class UnexpectedError(Exception):
        pass

    real_load = io_module._numpy().load

    def exploding_load(*args: Any, **kwargs: Any) -> Any:
        raise UnexpectedError("not a file problem")

    monkeypatch.setattr(io_module._numpy(), "load", exploding_load)
    try:
        with pytest.raises(UnexpectedError):
            call_load(path)
    finally:
        monkeypatch.setattr(io_module._numpy(), "load", real_load)


def test_an_unexpected_error_from_the_archive_is_propagated(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The same guard one layer out, on the container rather than the payload.

    `load` turns a `BadZipFile` into "archive is malformed". A `MemoryError` or
    an `AttributeError` from the same call is not a statement about the
    archive, and reporting it as one loses the only information the caller
    had."""
    path = tmp_path / "propagate-outer.akd"
    call_save(rich_diagram(), path)

    class UnexpectedError(Exception):
        pass

    def exploding_read(self: Any, name: Any, *args: Any, **kwargs: Any) -> bytes:
        raise UnexpectedError("not an archive problem")

    monkeypatch.setattr(zipfile.ZipFile, "read", exploding_read)

    with pytest.raises(UnexpectedError):
        call_load(path)


def test_a_tensor_like_value_is_converted_through_detach_alone() -> None:
    """§3.3's namespace rule reaches the I/O boundary: `save` converts with
    `np.asarray`, and a torch tensor that requires grad refuses that call.

    The fallback tries `detach()` -- torch's own spelling for "give me
    something NumPy can see" -- rather than special-casing torch by name, so it
    works for any backend using the same protocol and imports nothing.
    It does **not** try `cpu()`, and the stub's unused `cpu()` is here to pin
    that: §10.1 requirement 1 makes a cross-device transfer the owner's to ask
    for, so an array that will not convert is refused rather than moved.
    Detaching survives that clause because it is not a transfer -- the clause
    is about residency, and the document says nothing about autograd. Exercised
    with a stub because torch is not in the default test environment by design;
    `test_rfc0001_torch_live.py` runs the real thing."""
    io_module = _io_module()
    moved: list[str] = []

    class NeedsDetach:
        """Refuses `np.asarray` until detached, the way a grad tensor does."""

        def __init__(self, values: Any, *, detached: bool = False) -> None:
            self._values = values
            self._detached = detached

        def __array__(self, *args: Any, **kwargs: Any) -> Any:
            if not self._detached:
                raise RuntimeError("call detach() first")
            return np.asarray(self._values, *args, **kwargs)

        def detach(self) -> NeedsDetach:
            return NeedsDetach(self._values, detached=True)

        def cpu(self) -> NeedsDetach:
            moved.append("cpu")
            return self

    converted = io_module._to_numpy(np, NeedsDetach([1.0, 2.0, 3.0]))

    assert isinstance(converted, np.ndarray)
    assert converted.tolist() == [1.0, 2.0, 3.0]
    assert moved == [], "save transferred an array the caller did not move"


def test_an_off_host_array_is_refused_before_a_conversion_is_attempted() -> None:
    """§10.1 requirement 1: `save` MUST *require* host residency rather than
    attempt the conversion and translate whatever comes back.

    The distinction is load-bearing rather than stylistic, and JAX is why. A
    CUDA tensor makes `np.asarray` raise, so a caught-and-translated failure
    would look equivalent there; a GPU-resident JAX array instead satisfies
    `np.asarray` by copying itself to the host first, which is precisely the
    "several seconds of unasked-for work" the clause rules out. Both stubs
    below convert successfully, so a `save` that reached for the conversion
    before reading the device would write a file and this test would fail.

    Two stubs because the two backends the clause names spell a device
    differently -- torch exposes `device.type`, JAX `device.platform` -- and
    neither is installed in the default test environment by design.
    """
    io_module = _io_module()
    attempts: list[str] = []

    class TorchLikeDevice:
        type = "cuda"

        def __str__(self) -> str:
            return "cuda:0"

    class JaxLikeDevice:
        platform = "gpu"

        def __str__(self) -> str:
            return "cuda(id=0)"

    class OnDevice:
        """Converts happily; the refusal must not depend on it failing."""

        def __init__(self, device: Any) -> None:
            self.device = device

        def __array__(self, *args: Any, **kwargs: Any) -> Any:
            attempts.append(str(self.device))
            return np.asarray([1.0, 2.0], *args, **kwargs)

    for device in (TorchLikeDevice(), JaxLikeDevice()):
        with pytest.raises(ValueError, match="host-resident") as caught:
            io_module._to_numpy(np, OnDevice(device))
        message = str(caught.value)
        assert str(device) in message, "the refusal must name the device"
        assert ".cpu()" in message
        assert "jax.device_get" in message
        assert "from_dlpack" in message

    assert attempts == [], "save attempted the transfer instead of refusing it"


def test_a_host_resident_array_is_not_refused_for_reporting_a_device() -> None:
    """The other side of the residency check. Every array-API array carries a
    `device`, NumPy's own included (`'cpu'` since 2.0), so a check that read
    the attribute and refused on sight would refuse every save in this file.

    The unrecognised case is the one worth stating: the standard gives no
    portable way to ask whether a device object *is* the host, so a device
    shaped like nothing this module knows is a "cannot tell" and MUST NOT be
    refused on its own -- `array_api_strict`'s `CPU_DEVICE` is exactly that
    shape, and §3.3 requires that namespace to reach `save`.
    """
    io_module = _io_module()

    class UnknownDevice:
        def __str__(self) -> str:
            return "some-accelerator-shaped-object"

    class OnDevice:
        def __init__(self, device: Any) -> None:
            self.device = device

        def __array__(self, *args: Any, **kwargs: Any) -> Any:
            return np.asarray([1.0, 2.0], *args, **kwargs)

    for device in ("cpu", UnknownDevice()):
        converted = io_module._to_numpy(np, OnDevice(device))
        assert converted.tolist() == [1.0, 2.0]


def test_a_value_that_cannot_convert_reports_the_original_failure() -> None:
    """The other end of the same fallback. When `detach()` exists but the
    result still will not convert, the error a caller sees must be the *first*
    one -- what `np.asarray` said about their actual value -- and not whatever
    the second attempt produced on an object they never passed."""
    io_module = _io_module()

    class NeverConverts:
        def __array__(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("original failure")

        def detach(self) -> NeverConverts:
            return self

    with pytest.raises(RuntimeError, match="original failure"):
        io_module._to_numpy(np, NeverConverts())


def test_spec_version_agrees_with_the_rfc_header() -> None:
    """``_SPEC_VERSION`` tracks RFC-0001's Version row, and nothing else does.

    §10.2 defines ``spec_version`` as which revision of the specification the
    writer implemented, and the header's Version row names *itself* as what
    §10.2 writes into every file. They are one fact recorded twice, so they can
    disagree -- and they have, twice. ``_SPEC_VERSION`` sat at ``0.1.0`` across
    three document revisions until changelog entry 65 noticed; the revision
    that opened the comment window moved the header to ``1.0.0`` and left the
    writer at ``0.3.0``.

    Neither drift was caught, and the reason both times is that the pins in
    this module move *with* ``io.py`` rather than against the document: they
    agree with each other while the file on disk claims conformance to a
    revision the specification does not describe. This asserts the comparison
    that was missing, against the document itself.
    """
    header = _RFC_PATH.read_text(encoding="utf-8")

    row = re.search(r"^\|\s*\*\*Version\*\*\s*\|\s*(\d+\.\d+\.\d+)", header, re.M)
    assert row is not None, "RFC-0001's header has no Version row to compare against"
    documented = row.group(1)

    assert documented == _io_module()._SPEC_VERSION
    assert documented == SPEC_VERSION


def test_the_rfc_example_metadata_block_carries_the_documented_version() -> None:
    """§10.2's illustrative ``meta.json`` is a literal, and drifts like one.

    A reader copies that block; if it names a revision the writer never emits,
    the document contradicts itself in the one place readers reach for first.
    """
    header = _RFC_PATH.read_text(encoding="utf-8")

    row = re.search(r"^\|\s*\*\*Version\*\*\s*\|\s*(\d+\.\d+\.\d+)", header, re.M)
    assert row is not None
    documented = row.group(1)

    examples = re.findall(r'"spec_version":\s*"(\d+\.\d+\.\d+)"', header)
    assert examples, "§10.2's example block no longer names spec_version"
    assert set(examples) == {documented}

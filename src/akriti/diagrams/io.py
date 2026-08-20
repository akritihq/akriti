"""Deterministic ``.akd`` serialization for persistence diagrams. RFC-0001 §10."""

from __future__ import annotations

import importlib
import io as _io
import json
import lzma
import re
import zipfile
import zlib
from importlib import metadata
from os import PathLike, fspath
from typing import Any

from akriti.diagrams.core import (
    DiagramBatch,
    DiagramMeta,
    PersistenceDiagram,
    _parse_optional_version,
)

_FORMAT = "akriti.diagrams.akd"
_SPEC = "RFC-0001"
_SPEC_VERSION = "0.3.0"
_FORMAT_VERSION = 0
_META_FIELDS = (
    "filtration",
    "backend",
    "backend_version",
    "coeff_field",
    "params",
    "provenance",
    "description",
)
_REQUIRED_ENVELOPE_FIELDS = ("format", "format_version", "spec", "spec_version", "kind")
try:
    _zstd = importlib.import_module("compression.zstd")
except ImportError:
    _ZSTD_ERROR = None
else:
    _ZSTD_ERROR = getattr(_zstd, "ZstdError", None)


def _is_zip_read_error(exc: BaseException) -> bool:
    """Recognize ZIP codec/read failures, including runtime Zstandard."""
    if isinstance(
        exc,
        (
            zipfile.BadZipFile,
            RuntimeError,
            NotImplementedError,
            OSError,
            lzma.LZMAError,
            zlib.error,
        ),
    ):
        return True
    return isinstance(_ZSTD_ERROR, type) and isinstance(exc, _ZSTD_ERROR)


def _numpy() -> Any:
    """Import NumPy at the serialization boundary and enforce numpy>=2.0."""
    try:
        np = importlib.import_module("numpy")
    except ModuleNotFoundError as exc:
        if exc.name == "numpy":
            raise ImportError(
                "akriti.diagrams save/load requires numpy>=2.0; install `akriti[io]`"
            ) from exc
        raise
    try:
        version = metadata.version("numpy")
    except (metadata.PackageNotFoundError, ValueError) as exc:
        raise ImportError(
            "could not determine numpy metadata for numpy>=2.0; "
            "install `akriti[io]` (numpy metadata is unavailable)"
        ) from exc
    normalized = _normalize_version(version)
    try:
        epoch, release, unstable = _parse_optional_version(normalized)
    except (TypeError, ValueError) as exc:
        raise ImportError(
            f"could not parse numpy version {version!r} (requires numpy>=2.0); "
            "install `akriti[io]`"
        ) from exc
    floor = (0, (2, 0, 0))
    if (epoch, release) < floor or ((epoch, release) == floor and unstable):
        raise ImportError(
            f"numpy {version} is unsupported; requires numpy>=2.0; install `akriti[io]`"
        )
    return np


def _normalize_version(version: str) -> str:
    """Normalize supported PEP 440 spellings before the shared parser."""
    normalized = version.strip()
    match = re.fullmatch(
        r"v?(?P<epoch>[0-9]+!)?"
        r"(?P<release>[0-9]+(?:\.[0-9]+)*)"
        r"(?:[-_.]?(?P<pre>preview|alpha|beta|rc|pre|a|b|c)"
        r"[-_.]?(?P<pre_num>[0-9]+)?)?"
        r"(?:-(?P<implicit_post>[0-9]+)"
        r"|[-_.]?(?P<post>post|rev|r)[-_.]?(?P<post_num>[0-9]+)?)?"
        r"(?:[-_.]?(?P<dev>dev)[-_.]?(?P<dev_num>[0-9]+)?)?"
        r"(?:\+(?P<local>[0-9A-Za-z]+(?:[._-][0-9A-Za-z]+)*))?",
        normalized,
        flags=re.IGNORECASE,
    )
    if match is None:
        return normalized
    groups = match.group
    epoch = groups("epoch") or ""
    release = groups("release")
    pre = groups("pre")
    pre_part = f"{pre}{groups('pre_num') or '0'}" if pre else ""
    if groups("implicit_post") is not None:
        post_part = f".post{groups('implicit_post')}"
    elif groups("post"):
        post_part = f".post{groups('post_num') or '0'}"
    else:
        post_part = ""
    dev_part = f".dev{groups('dev_num') or '0'}" if groups("dev") else ""
    local = f"+{groups('local')}" if groups("local") else ""
    return f"{epoch}{release}{pre_part}{post_part}{dev_part}{local}"


def _meta_dict(meta: DiagramMeta) -> dict[str, Any]:
    return {name: getattr(meta, name) for name in _META_FIELDS}


def _envelope(kind: str, value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "format": _FORMAT,
        "format_version": _FORMAT_VERSION,
        "spec": _SPEC,
        "spec_version": _SPEC_VERSION,
        "kind": kind,
    }
    result["meta" if kind == "diagram" else "metas"] = value
    return result


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    info.flag_bits = 0
    info.extra = b""
    info.comment = b""
    return info


def _read_zip_member(archive: zipfile.ZipFile, name: str, label: str) -> bytes:
    """Read one required member, normalizing codec and CRC failures."""
    try:
        return archive.read(name)
    except Exception as exc:
        if not _is_zip_read_error(exc):
            raise
        raise ValueError(f"archive member {label} could not be read: {exc}") from exc


def _array_bytes(np: Any, array: Any) -> bytes:
    output = _io.BytesIO()
    np.lib.format.write_array(output, array, allow_pickle=False)
    return output.getvalue()


def _npz_bytes(np: Any, arrays: list[tuple[str, Any]]) -> bytes:
    output = _io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in arrays:
            archive.writestr(_zip_info(f"{name}.npy"), _array_bytes(np, array))
        archive.comment = b""
    return output.getvalue()


def _canonical_arrays(
    np: Any, obj: PersistenceDiagram | DiagramBatch
) -> list[tuple[str, Any]]:
    if isinstance(obj, PersistenceDiagram):
        canonical = obj.canonical()
        arrays: list[tuple[str, Any]] = [
            (
                "births",
                np.array(_to_numpy(np, canonical.births), dtype="<f8", copy=True),
            ),
            (
                "deaths",
                np.array(_to_numpy(np, canonical.deaths), dtype="<f8", copy=True),
            ),
            ("dims", np.array(_to_numpy(np, canonical.dims), dtype="<i4", copy=True)),
        ]
    else:
        canonical_batch = obj.canonical()
        arrays = [
            (
                "births",
                np.array(_to_numpy(np, canonical_batch.births), dtype="<f8", copy=True),
            ),
            (
                "deaths",
                np.array(_to_numpy(np, canonical_batch.deaths), dtype="<f8", copy=True),
            ),
            (
                "dims",
                np.array(_to_numpy(np, canonical_batch.dims), dtype="<i4", copy=True),
            ),
        ]
        arrays.append(
            (
                "offsets",
                np.array(
                    _to_numpy(np, canonical_batch.offsets), dtype="<i8", copy=True
                ),
            )
        )
    for name, array in arrays:
        if name in ("births", "deaths"):
            array[array == 0] = 0.0
    return arrays


def _to_numpy(np: Any, value: Any) -> Any:
    """Convert a CPU-compatible array without importing an optional backend."""
    try:
        return np.asarray(value)
    except Exception as first_error:
        detach = getattr(value, "detach", None)
        if not callable(detach):
            raise first_error
        detached = detach()
        cpu = getattr(detached, "cpu", None)
        if callable(cpu):
            detached = cpu()
        try:
            return np.asarray(detached)
        except Exception:
            raise first_error from None


def save(obj: PersistenceDiagram | DiagramBatch, path: str | PathLike[str]) -> None:
    """Write a diagram or ragged batch to deterministic ``.akd`` bytes. §10.2.

    Assumes *obj* is a valid public diagram type and *path* is writable. The
    input arrays may use any supported array namespace; conversion to NumPy
    occurs only at this serialization boundary, and the input is not mutated.

    **The round trip preserves the diagram, not the buffer.** §10.1
    requirement 1 is stated over ``==`` and ``same_provenance``, and §6.3
    defines the first as "no tolerance" rather than "bit-identical", so three
    representation choices below are deliberate and none of them is visible to
    either comparison:

    - **Rows are written in canonical order** (§7, §10.2), not in the order
      the backend emitted them. Bar order within a diagram is not meaningful
      (§7) and ``==`` is order-insensitive. Diagram order within a batch *is*
      meaningful and is preserved exactly.
    - **Arrays are written little-endian** (``<f8``, ``<i4``, ``<i8``), so a
      big-endian host's buffers are byte-swapped on the way out and back.
      ``load`` accepts either byte order.
    - **``-0.0`` is normalised to ``+0.0``** in ``births`` and ``deaths``.
      This one is forced rather than chosen: ``-0.0 == 0.0`` in IEEE 754, so
      two diagrams differing only in the sign of a zero are the same diagram
      under ``==``, and §10.1 requirement 4 requires identical diagrams to
      produce identical bytes. Without the normalisation the file would depend
      on a sign no comparison here can see, and a canonical sort cannot
      separate the two either -- a stable sort leaves numerically equal keys in
      input order. ``content_hash`` normalises it for the same reason (§8.1).

    ``to_csv()`` (§10.3) does preserve the sign of zero, deliberately: it is a
    lossy human-readable export bound by no determinism requirement, where this
    is the normative format bound by one.
    """
    if not isinstance(obj, (PersistenceDiagram, DiagramBatch)):
        raise TypeError(
            f"save expects a PersistenceDiagram or DiagramBatch; got {type(obj)!r}"
        )
    np = _numpy()
    if isinstance(obj, PersistenceDiagram):
        kind = "diagram"
        value: Any = _meta_dict(obj.meta)
    else:
        kind = "batch"
        value = [_meta_dict(m) for m in obj.metas]
    metadata_bytes = json.dumps(
        _envelope(kind, value),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    bars_bytes = _npz_bytes(np, _canonical_arrays(np, obj))
    with zipfile.ZipFile(fspath(path), "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(_zip_info("meta.json"), metadata_bytes)
        archive.writestr(_zip_info("bars.npz"), bars_bytes)
        archive.comment = b""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _json_load(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("meta.json is not valid UTF-8") from exc
    try:
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (RecursionError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"meta.json is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("meta.json JSON envelope must be an object")
    return value


def _read_envelope(envelope: dict[str, Any]) -> tuple[str, Any]:
    for field in _REQUIRED_ENVELOPE_FIELDS:
        if field not in envelope:
            raise ValueError(f"meta.json envelope missing required field {field!r}")
    if type(envelope["format"]) is not str:
        raise ValueError("meta.json field 'format' must be a string")
    if envelope["format"] != _FORMAT:
        raise ValueError("unsupported meta.json field 'format'")
    if type(envelope["format_version"]) is not int:
        raise ValueError("meta.json field 'format_version' must be an integer")
    if envelope["format_version"] != _FORMAT_VERSION:
        raise ValueError("unsupported meta.json field 'format_version'")
    if type(envelope["spec"]) is not str:
        raise ValueError("meta.json field 'spec' must be a string")
    if envelope["spec"] != _SPEC:
        raise ValueError("unsupported meta.json field 'spec'")
    if type(envelope["spec_version"]) is not str:
        raise ValueError("meta.json field 'spec_version' must be a string")
    kind = envelope["kind"]
    if type(kind) is not str or kind not in ("diagram", "batch"):
        raise ValueError("unsupported meta.json field 'kind'")
    if kind == "diagram":
        if "meta" not in envelope:
            raise ValueError("diagram envelope requires 'meta'")
        if "metas" in envelope:
            raise ValueError("diagram envelope must not contain 'metas'")
        if not isinstance(envelope["meta"], dict):
            raise ValueError("diagram 'meta' must be an object")
        return kind, envelope["meta"]
    if "metas" not in envelope:
        raise ValueError("batch envelope requires 'metas'")
    if "meta" in envelope:
        raise ValueError("batch envelope must not contain 'meta'")
    if not isinstance(envelope["metas"], list):
        raise ValueError("batch 'metas' must be an array")
    return kind, envelope["metas"]


def _read_meta(value: Any, label: str) -> DiagramMeta:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    unknown = set(value).difference(_META_FIELDS)
    if unknown:
        raise ValueError(f"{label} has unknown field {sorted(unknown)[0]!r}")
    missing = [field for field in _META_FIELDS if field not in value]
    if missing:
        raise ValueError(f"{label} missing required field {missing[0]!r}")
    for field in ("params", "provenance"):
        if not isinstance(value[field], dict):
            raise ValueError(f"{label}.{field} must be an object")
    try:
        return DiagramMeta(**{field: value[field] for field in _META_FIELDS})
    except (RecursionError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc


def _preflight_npz(np: Any, raw: bytes, kind: str) -> None:
    """Validate required NPY members and byte sizes before NumPy loads data."""
    try:
        nested = zipfile.ZipFile(_io.BytesIO(raw), "r")
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        raise ValueError(f"bars.npz is malformed: {exc}") from exc
    with nested:
        infos = nested.infolist()
        seen: set[str] = set()
        for info in infos:
            logical = (
                info.filename[:-4] if info.filename.endswith(".npy") else info.filename
            )
            if logical in seen:
                raise ValueError(
                    f"bars.npz contains duplicate member {info.filename!r}"
                )
            seen.add(logical)
        required = ["births.npy", "deaths.npy", "dims.npy"]
        if kind == "batch":
            required.append("offsets.npy")
        by_name = {info.filename: info for info in infos}
        for name in required:
            member_info = by_name.get(name)
            if member_info is None:
                raise ValueError(f"bars.npz missing required array {name[:-4]!r}")
            try:
                with nested.open(member_info, "r") as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _fortran, dtype = np.lib.format.read_array_header_1_0(
                            stream
                        )
                    elif version in ((2, 0), (3, 0)):
                        shape, _fortran, dtype = np.lib.format.read_array_header_2_0(
                            stream
                        )
                    else:
                        raise ValueError(
                            f"unsupported NPY version {version!r} in {name}"
                        )
                    elements = 1
                    for dimension in shape:
                        if type(dimension) is not int or dimension < 0:
                            raise ValueError(f"invalid shape in {name}")
                        elements *= dimension
                    expected_size = stream.tell() + elements * dtype.itemsize
                    if expected_size != member_info.file_size:
                        raise ValueError(
                            f"{name} header/data size mismatch: "
                            f"expected {expected_size}, "
                            f"got {member_info.file_size}"
                        )
            except Exception as exc:
                if not isinstance(
                    exc, (EOFError, ValueError)
                ) and not _is_zip_read_error(exc):
                    raise
                raise ValueError(f"invalid NPY header for {name}: {exc}") from exc


def _payload_arrays(np: Any, raw: bytes, kind: str) -> dict[str, Any]:
    _preflight_npz(np, raw, kind)
    try:
        payload = np.load(_io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        if not isinstance(
            exc, (RecursionError, EOFError, ValueError)
        ) and not _is_zip_read_error(exc):
            raise
        raise ValueError(f"bars.npz payload could not be read: {exc}") from exc
    if not hasattr(payload, "files"):
        raise ValueError("bars.npz payload is not an npz archive")
    try:
        names = set(payload.files)
        for field in ("births", "deaths", "dims"):
            if field not in names:
                raise ValueError(f"bars.npz missing required array {field!r}")
        if kind == "diagram" and "offsets" in names:
            raise ValueError("diagram bars.npz must not contain 'offsets'")
        if kind == "batch" and "offsets" not in names:
            raise ValueError("batch bars.npz requires 'offsets'")
        arrays = {
            field: np.asarray(payload[field]) for field in ("births", "deaths", "dims")
        }
        if kind == "batch":
            arrays["offsets"] = np.asarray(payload["offsets"])
        return arrays
    except Exception as exc:
        if not isinstance(exc, (RecursionError, ValueError)) and not _is_zip_read_error(
            exc
        ):
            raise
        raise ValueError(f"bars.npz payload could not be read: {exc}") from exc
    finally:
        payload.close()


def _validate_array_schema(np: Any, arrays: dict[str, Any], kind: str) -> None:
    expected = {
        "births": np.dtype(np.float64),
        "deaths": np.dtype(np.float64),
        "dims": np.dtype(np.int32),
    }
    for field, dtype in expected.items():
        array = arrays[field]
        if array.ndim != 1:
            raise ValueError(f"{field} must be rank-1; got rank {array.ndim}")
        if array.dtype.kind != dtype.kind or array.dtype.itemsize != dtype.itemsize:
            raise ValueError(f"{field} must have dtype {dtype}; got {array.dtype}")
        arrays[field] = array.astype(dtype, copy=False)
    lengths = {field: arrays[field].shape[0] for field in expected}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"bar arrays must have equal length; got {lengths}")
    if kind == "batch":
        offsets = arrays["offsets"]
        if offsets.ndim != 1:
            raise ValueError(f"offsets must be rank-1; got rank {offsets.ndim}")
        expected_offsets = np.dtype(np.int64)
        if (
            offsets.dtype.kind != expected_offsets.kind
            or offsets.dtype.itemsize != expected_offsets.itemsize
        ):
            raise ValueError(f"offsets must have dtype int64; got {offsets.dtype}")
        arrays["offsets"] = offsets.astype(expected_offsets, copy=False)


def load(path: str | PathLike[str]) -> PersistenceDiagram | DiagramBatch:
    """Read an ``.akd`` archive and return NumPy-backed validated objects. §10.2.

    Assumes *path* names a readable archive produced under the RFC-0001
    layout; malformed archives raise ``ValueError`` after schema validation.
    """
    np = _numpy()
    try:
        archive = zipfile.ZipFile(fspath(path), "r")
    except (
        zipfile.BadZipFile,
        RuntimeError,
        NotImplementedError,
        lzma.LZMAError,
        zlib.error,
    ) as exc:
        raise ValueError(f"path is not a valid ZIP archive: {exc}") from exc
    try:
        with archive:
            names = archive.namelist()
            if names != ["meta.json", "bars.npz"]:
                raise ValueError(
                    "archive must contain exactly meta.json then bars.npz; got members "
                    f"{names!r}"
                )
            envelope = _json_load(_read_zip_member(archive, "meta.json", "meta.json"))
            kind, metadata_value = _read_envelope(envelope)
            arrays = _payload_arrays(
                np, _read_zip_member(archive, "bars.npz", "bars.npz"), kind
            )
    except Exception as exc:
        if not isinstance(exc, RecursionError) and not _is_zip_read_error(exc):
            raise
        raise ValueError(f"archive is malformed: {exc}") from exc
    _validate_array_schema(np, arrays, kind)
    if kind == "diagram":
        meta = _read_meta(metadata_value, "meta")
        try:
            return PersistenceDiagram(
                dims=arrays["dims"],
                births=arrays["births"],
                deaths=arrays["deaths"],
                meta=meta,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid diagram bars: {exc}") from exc
    metas = tuple(
        _read_meta(value, f"metas[{index}]")
        for index, value in enumerate(metadata_value)
    )
    offsets = arrays["offsets"]
    if offsets.shape[0] != len(metas) + 1:
        raise ValueError(
            "batch B1 requires len(offsets) == len(metas) + 1; "
            f"got {offsets.shape[0]} offsets for {len(metas)} metas"
        )
    try:
        return DiagramBatch(
            dims=arrays["dims"],
            births=arrays["births"],
            deaths=arrays["deaths"],
            offsets=offsets,
            metas=metas,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid batch bars or offsets: {exc}") from exc


__all__ = ["load", "save"]

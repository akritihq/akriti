"""RFC-0001 §10.3 exporter contract."""

from __future__ import annotations

import csv
import importlib
import io
import math
import warnings
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import akriti.diagrams.adapters as adapters
from akriti.diagrams import (
    DiagramBatch,
    DiagramMeta,
    PersistenceDiagram,
    from_array,
    to_arrays,
    to_csv,
    to_parquet,
)


def diagram() -> PersistenceDiagram:
    return PersistenceDiagram(
        dims=np.array([2, 0, 2, 0], dtype=np.int32),
        births=np.array([3.0, -0.0, 1.25, 2.0], dtype=np.float64),
        deaths=np.array([np.inf, 0.0, 4.5, 2.0], dtype=np.float64),
        meta=DiagramMeta(backend="test", provenance={"secret": "lost"}),
    )


def batch() -> DiagramBatch:
    return DiagramBatch(
        dims=np.array([1, 1, 0], dtype=np.int32),
        births=np.array([0.0, 2.0, -0.0], dtype=np.float64),
        deaths=np.array([1.0, np.inf, 0.0], dtype=np.float64),
        offsets=np.array([0, 2, 2, 3, 3], dtype=np.int64),
        metas=(
            DiagramMeta(backend="a"),
            DiagramMeta(backend="b"),
            DiagramMeta(),
            DiagramMeta(backend="trailing"),
        ),
    )


def assert_one_loss_warning(
    record: list[warnings.WarningMessage], extra: str = ""
) -> None:
    assert len(record) == 1
    assert record[0].category is UserWarning
    message = str(record[0].message)
    assert "DiagramMeta" in message
    if extra:
        assert extra in message


def test_exporters_are_public() -> None:
    import akriti.diagrams as pkg

    for name in ("to_arrays", "to_csv", "to_parquet"):
        assert name in pkg.__all__
        assert getattr(pkg, name) is globals()[name]


def test_to_arrays_groups_sorted_degrees_preserving_rows_namespace_and_inf() -> None:
    obj = diagram()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = to_arrays(obj)
    assert_one_loss_warning(record, "global inter-degree order")
    assert list(result) == [0, 2]
    assert all(type(key) is int for key in result)
    assert all(value.shape[1] == 2 for value in result.values())
    assert all(value.dtype == obj.xp.float64 for value in result.values())
    assert all(value.__array_namespace__() is obj.xp for value in result.values())
    np.testing.assert_array_equal(result[0], np.array([[-0.0, 0.0], [2.0, 2.0]]))
    np.testing.assert_array_equal(result[2], np.array([[3.0, np.inf], [1.25, 4.5]]))


def test_to_arrays_rejects_batch_and_empty_diagram_is_empty() -> None:
    with pytest.raises(TypeError, match="DiagramBatch"):
        to_arrays(batch())
    empty = PersistenceDiagram(
        dims=np.empty(0, dtype=np.int32),
        births=np.empty(0, dtype=np.float64),
        deaths=np.empty(0, dtype=np.float64),
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert to_arrays(empty) == {}
    assert_one_loss_warning(record, "global inter-degree order")


def test_to_arrays_preserves_exact_duplicate_rows() -> None:
    obj = PersistenceDiagram(
        dims=np.array([1, 1], dtype=np.int32),
        births=np.array([0.5, 0.5], dtype=np.float64),
        deaths=np.array([1.5, 1.5], dtype=np.float64),
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = to_arrays(obj)
    assert result[1].shape == (2, 2)
    np.testing.assert_array_equal(result[1], [[0.5, 1.5], [0.5, 1.5]])


def test_to_csv_single_round_trips_via_caller_side_parse_and_warns_once() -> None:
    """`to_csv` -> a caller-side parse -> `from_array`, per §10.3.

    There is no `from_csv`, and §10.3 declines to add one. The middle step is
    therefore the caller's own parse -- here `csv.reader` plus NumPy, standing
    in for the `numpy.genfromtxt` / `pandas.read_csv` / R `read.csv` the
    section names. Nothing below calls an akriti reader for CSV, because none
    exists.

    Turning `inf`-bearing text into a `float64` array is the step that needs
    NumPy, so this case needs `akriti[numpy]` to run (§3.3) -- supplied here by
    this module's top-level `import numpy as np`, which the `test` extra
    resolves through `akriti[numpy]`.

    Also pins the LF line terminator, the header row that carries the column
    order back in through `from_array(columns=...)`, and the single
    `DiagramMeta` loss warning.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        text = to_csv(diagram())
    assert_one_loss_warning(record)
    assert text == "dim,birth,death\n2,3.0,inf\n0,-0.0,0.0\n2,1.25,4.5\n0,2.0,2.0\n"
    rows = list(csv.reader(io.StringIO(text, newline="")))
    values = np.asarray([[float(x) for x in row] for row in rows[1:]], dtype=np.float64)
    restored = from_array(values, columns=rows[0])
    assert restored.meta.backend == "array"
    np.testing.assert_array_equal(restored.dims, diagram().dims)
    np.testing.assert_array_equal(restored.births, diagram().births)
    np.testing.assert_array_equal(restored.deaths, diagram().deaths)


def test_to_csv_batch_prepends_ids_and_reports_empty_members_are_lost() -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        text = to_csv(batch())
    assert_one_loss_warning(record, "empty-member")
    assert text == (
        "diagram_id,dim,birth,death\n0,1,0.0,1.0\n0,1,2.0,inf\n2,0,-0.0,0.0\n"
    )


def test_to_csv_empty_diagram_and_batch_are_header_only() -> None:
    empty_diagram = PersistenceDiagram(
        dims=np.empty(0, dtype=np.int32),
        births=np.empty(0, dtype=np.float64),
        deaths=np.empty(0, dtype=np.float64),
    )
    empty_batch = DiagramBatch(
        dims=np.empty(0, dtype=np.int32),
        births=np.empty(0, dtype=np.float64),
        deaths=np.empty(0, dtype=np.float64),
        offsets=np.array([0, 0], dtype=np.int64),
        metas=(DiagramMeta(),),
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert to_csv(empty_diagram) == "dim,birth,death\n"
        assert to_csv(empty_batch) == "diagram_id,dim,birth,death\n"


def test_to_csv_preserves_signed_zero() -> None:
    obj = diagram()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        text = to_csv(obj)
    values = list(csv.reader(io.StringIO(text)))[2]
    assert values[1] == "-0.0"
    assert math.copysign(1.0, float(values[1])) == -1.0


@pytest.mark.parametrize("obj", [object(), np.zeros((1, 3)), "diagram"])
def test_csv_and_arrays_reject_unsupported_types(obj: Any) -> None:
    for exporter in (to_arrays, to_csv):
        with pytest.raises(TypeError, match=r"PersistenceDiagram|DiagramBatch"):
            exporter(obj)


def test_parquet_is_lazy_and_does_not_import_pyarrow_for_other_exporters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module
    imported: list[str] = []

    def track(name: str, package: str | None = None) -> Any:
        imported.append(name)
        return real_import(name, package)

    monkeypatch.setattr(adapters, "import_module", track)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        to_arrays(diagram())
        to_csv(diagram())
    assert "pyarrow" not in imported


def test_parquet_rejects_unsupported_type_before_lazy_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    def track(name: str, package: str | None = None) -> Any:
        imported.append(name)
        return importlib.import_module(name, package)

    monkeypatch.setattr(adapters, "import_module", track)
    with pytest.raises(TypeError, match=r"PersistenceDiagram|DiagramBatch"):
        to_parquet(object())
    assert "pyarrow" not in imported


def test_parquet_missing_dependency_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str, package: str | None = None) -> Any:
        if name == "pyarrow":
            raise ModuleNotFoundError("No module named 'pyarrow'", name="pyarrow")
        return importlib.import_module(name, package)

    monkeypatch.setattr(adapters, "import_module", missing)
    with pytest.raises(ImportError, match=r"akriti\[parquet\]"):
        to_parquet(diagram())


@pytest.mark.backend
@pytest.mark.parquet
def test_to_parquet_live_schema_rows_and_inf() -> None:
    pytest.importorskip("pyarrow")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        table = to_parquet(diagram())
    assert_one_loss_warning(record)
    assert table.column_names == ["dim", "birth", "death"]
    assert table.schema.field("dim").type == importlib.import_module("pyarrow").int32()
    assert (
        table.schema.field("birth").type == importlib.import_module("pyarrow").float64()
    )
    assert (
        table.schema.field("death").type == importlib.import_module("pyarrow").float64()
    )
    assert table.schema.metadata is None
    rows = table.to_pylist()
    assert rows == [
        {"dim": 2, "birth": 3.0, "death": np.inf},
        {"dim": 0, "birth": -0.0, "death": 0.0},
        {"dim": 2, "birth": 1.25, "death": 4.5},
        {"dim": 0, "birth": 2.0, "death": 2.0},
    ]
    assert math.copysign(1.0, rows[1]["birth"]) == -1.0


@pytest.mark.backend
@pytest.mark.parquet
def test_to_parquet_live_batch_and_empty_schema() -> None:
    pytest.importorskip("pyarrow")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        table = to_parquet(batch())
    assert_one_loss_warning(record, "empty-member")
    assert table.column_names == ["diagram_id", "dim", "birth", "death"]
    assert table.to_pylist() == [
        {"diagram_id": 0, "dim": 1, "birth": 0.0, "death": 1.0},
        {"diagram_id": 0, "dim": 1, "birth": 2.0, "death": np.inf},
        {"diagram_id": 2, "dim": 0, "birth": -0.0, "death": 0.0},
    ]
    empty = DiagramBatch(
        dims=np.empty(0, dtype=np.int32),
        births=np.empty(0, dtype=np.float64),
        deaths=np.empty(0, dtype=np.float64),
        offsets=np.array([0, 0], dtype=np.int64),
        metas=(DiagramMeta(),),
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = to_parquet(empty)
    assert result.num_rows == 0
    assert result.column_names == ["diagram_id", "dim", "birth", "death"]
    assert str(result.schema.field("diagram_id").type) == "int64"


@pytest.mark.backend
@pytest.mark.parquet
def test_to_parquet_live_empty_single_has_explicit_schema() -> None:
    pytest.importorskip("pyarrow")
    empty = PersistenceDiagram(
        dims=np.empty(0, dtype=np.int32),
        births=np.empty(0, dtype=np.float64),
        deaths=np.empty(0, dtype=np.float64),
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        table = to_parquet(empty)
    assert_one_loss_warning(record)
    assert table.num_rows == 0
    assert table.column_names == ["dim", "birth", "death"]
    assert [str(field.type) for field in table.schema] == [
        "int32",
        "double",
        "double",
    ]


def test_to_arrays_works_with_array_api_strict_namespace() -> None:
    xps = pytest.importorskip("array_api_strict")
    obj = PersistenceDiagram(
        dims=xps.asarray([1, 1], dtype=xps.int32),
        births=xps.asarray([0.0, 0.0], dtype=xps.float64),
        deaths=xps.asarray([1.0, np.inf], dtype=xps.float64),
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = to_arrays(obj)
    assert result[1].shape == (2, 2)
    assert result[1].dtype == xps.float64
    assert result[1].__array_namespace__() is obj.xp


def test_parquet_version_errors_are_actionable_and_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = ModuleType("pyarrow")

    def load(name: str, package: str | None = None) -> Any:
        if name == "pyarrow":
            return fake
        return importlib.import_module(name, package)

    monkeypatch.setattr(adapters, "import_module", load)
    for version in (None, "24.0.0", "not-a-version"):

        def metadata_version(name: str, version: str | None = version) -> str:
            if version is None:
                raise adapters.metadata.PackageNotFoundError(name)
            return version

        monkeypatch.setattr(adapters.metadata, "version", metadata_version)
        with pytest.raises(ImportError, match=r"akriti\[parquet\]"):
            to_parquet(diagram())


@pytest.mark.parametrize(
    "version",
    [
        "25.0.0.post1.dev1",
        "26.0.0.dev1",
        "1!25.0.0",
        "25.0.0.0+local",
        "25.0.0+local",
    ],
)
def test_parquet_accepts_pep440_versions(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    fake = ModuleType("pyarrow")
    monkeypatch.setattr(adapters, "import_module", lambda name: fake)
    monkeypatch.setattr(adapters.metadata, "version", lambda name: version)
    assert adapters._load_pyarrow() is fake


@pytest.mark.parametrize(
    "version",
    ["25.0.0rc1", "25.0.0.dev1", "24.9.9", "development"],
)
def test_parquet_rejects_pep440_versions(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    fake = ModuleType("pyarrow")
    monkeypatch.setattr(adapters, "import_module", lambda name: fake)
    monkeypatch.setattr(adapters.metadata, "version", lambda name: version)
    with pytest.raises(ImportError, match=r"akriti\[parquet\]"):
        adapters._load_pyarrow()


def test_parquet_transitive_import_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(name: str, package: str | None = None) -> Any:
        if name == "pyarrow":
            raise ModuleNotFoundError(
                "No module named 'missing_dependency'", name="missing_dependency"
            )
        return importlib.import_module(name, package)

    monkeypatch.setattr(adapters, "import_module", broken)
    with pytest.raises(ModuleNotFoundError, match="missing_dependency"):
        to_parquet(diagram())

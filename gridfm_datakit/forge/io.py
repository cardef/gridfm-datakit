"""Scalable I/O primitives for GridFM Forge.

The raw GridFM DataKit output is a collection of partitioned Parquet datasets.
This module keeps every higher-level Forge component independent from the exact
on-disk layout and avoids loading a complete dataset into memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


TABLE_FILES: dict[str, str] = {
    "bus": "bus_data.parquet",
    "branch": "branch_data.parquet",
    "gen": "gen_data.parquet",
    "y_bus": "y_bus_data.parquet",
    "runtime": "runtime_data.parquet",
}


class DatasetLayoutError(FileNotFoundError):
    """Raised when a required table is absent from a raw dataset."""


def table_path(root: str | Path, table: str) -> Path:
    """Return the path of a named raw table.

    Args:
        root: Directory containing GridFM DataKit raw outputs.
        table: Logical table name (``bus``, ``branch``, ``gen``, ``y_bus`` or
            ``runtime``).
    """

    try:
        filename = TABLE_FILES[table]
    except KeyError as exc:
        known = ", ".join(sorted(TABLE_FILES))
        raise KeyError(f"Unknown table {table!r}; expected one of: {known}") from exc
    return Path(root) / filename


def open_table(root: str | Path, table: str, *, required: bool = True) -> ds.Dataset | None:
    """Open a partitioned Parquet table as a PyArrow dataset."""

    path = table_path(root, table)
    if not path.exists():
        if required:
            raise DatasetLayoutError(f"Missing required table: {path}")
        return None
    return ds.dataset(path, format="parquet", partitioning="hive")


def _scenario_filter(scenarios: Sequence[int] | None) -> ds.Expression | None:
    if scenarios is None:
        return None
    values = [int(value) for value in scenarios]
    if not values:
        return ds.field("scenario") == -1
    return ds.field("scenario").isin(values)


def read_arrow(
    root: str | Path,
    table: str,
    *,
    scenarios: Sequence[int] | None = None,
    columns: Sequence[str] | None = None,
    required: bool = True,
) -> pa.Table:
    """Read selected scenarios and columns from a raw table.

    Predicate pushdown is used whenever the Parquet engine can exploit it.
    """

    dataset = open_table(root, table, required=required)
    if dataset is None:
        return pa.table({})
    return dataset.to_table(columns=columns, filter=_scenario_filter(scenarios))


def read_frame(
    root: str | Path,
    table: str,
    *,
    scenarios: Sequence[int] | None = None,
    columns: Sequence[str] | None = None,
    required: bool = True,
) -> pd.DataFrame:
    """Read a raw table into pandas after applying Arrow-side filtering."""

    return read_arrow(
        root,
        table,
        scenarios=scenarios,
        columns=columns,
        required=required,
    ).to_pandas(split_blocks=True, self_destruct=True)


def iter_scenario_batches(
    scenarios: Iterable[int], batch_size: int
) -> Iterable[list[int]]:
    """Yield deterministic batches of scenario identifiers."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch: list[int] = []
    for scenario in scenarios:
        batch.append(int(scenario))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def scenario_ids(root: str | Path, table: str = "bus") -> list[int]:
    """Return sorted unique scenario identifiers without reading feature columns."""

    values = read_arrow(root, table, columns=["scenario"])["scenario"]
    return sorted({int(value.as_py()) for value in values})

"""End-to-end graph dataset compiler with provenance and leakage-safe splits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.dataset as ds

from .graph import compile_scenario
from .io import TABLE_FILES, iter_scenario_batches, read_frame, scenario_ids, table_path


@dataclass(frozen=True)
class ForgeConfig:
    input_dir: str
    output_dir: str
    seed: int = 0
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    batch_size: int = 128
    active_tolerance: float = 1e-3

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train_fraction + validation_fraction must be < 1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
    else:
        files = [path]
    for file in files:
        digest.update(str(file.relative_to(path.parent)).encode())
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _topology_signature(branch_frame) -> str:
    columns = ["from_bus", "to_bus"]
    if "br_status" in branch_frame:
        columns.append("br_status")
    values = branch_frame[columns].sort_values(columns[:2]).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()[:20]


def _assign_split(signature: str, seed: int, train: float, validation: float) -> str:
    value = int(hashlib.sha256(f"{seed}:{signature}".encode()).hexdigest()[:16], 16) / 16**16
    if value < train:
        return "train"
    if value < train + validation:
        return "validation"
    return "test"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_dataset(config: ForgeConfig) -> dict:
    """Compile raw DataKit output into portable per-scenario graph shards.

    Scenarios sharing the same topology signature are always assigned to the
    same split. This prevents the most common and least visible leakage mode in
    topology-generalisation experiments.
    """

    config.validate()
    input_dir = Path(config.input_dir).resolve()
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = scenario_ids(input_dir)
    split_counts = {"train": 0, "validation": 0, "test": 0}
    index: list[dict] = []

    for batch in iter_scenario_batches(scenarios, config.batch_size):
        bus = read_frame(input_dir, "bus", scenarios=batch)
        branch = read_frame(input_dir, "branch", scenarios=batch)
        gen = read_frame(input_dir, "gen", scenarios=batch)
        for scenario in batch:
            scenario_branch = branch.loc[branch["scenario"] == scenario]
            signature = _topology_signature(scenario_branch)
            split = _assign_split(
                signature,
                config.seed,
                config.train_fraction,
                config.validation_fraction,
            )
            sample = compile_scenario(
                scenario,
                bus,
                branch,
                gen,
                active_tolerance=config.active_tolerance,
            )
            relative = Path("shards") / split / f"scenario-{scenario:09d}.npz"
            path = output_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **sample.as_numpy_dict())
            split_counts[split] += 1
            index.append(
                {
                    "scenario": scenario,
                    "split": split,
                    "path": relative.as_posix(),
                    "topology_signature": signature,
                    **sample.metadata,
                }
            )

    source_tables = {}
    for name, filename in TABLE_FILES.items():
        path = table_path(input_dir, name)
        if path.exists():
            dataset = ds.dataset(path, format="parquet", partitioning="hive")
            source_tables[name] = {
                "path": filename,
                "sha256": _sha256(path),
                "schema": str(dataset.schema),
            }

    manifest = {
        "format": "gridfm-forge",
        "format_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "n_scenarios": len(index),
        "splits": split_counts,
        "source_tables": source_tables,
        "invariants": {
            "topology_grouped_splits": True,
            "framework_neutral": True,
            "original_bus_ids_preserved": True,
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "index.json", index)
    return manifest

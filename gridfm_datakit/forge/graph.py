"""Compile GridFM tabular outputs into graph-native tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


NODE_FEATURES = (
    "Pd", "Qd", "Pg", "Qg", "Vm", "Va", "vn_kv", "min_vm_pu", "max_vm_pu", "GS", "BS"
)
EDGE_FEATURES = (
    "r", "x", "b", "tap", "shift", "ang_min", "ang_max", "rate_a", "br_status"
)
EDGE_TARGETS = ("pf", "qf", "pt", "qt")
GEN_FEATURES = (
    "p_mw", "q_mvar", "min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar",
    "cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2", "in_service", "is_slack_gen"
)


@dataclass(frozen=True)
class GraphSample:
    """Framework-neutral graph sample for one solved grid scenario."""

    scenario: int
    node_ids: np.ndarray
    edge_ids: np.ndarray
    edge_index: np.ndarray
    node_features: np.ndarray
    edge_features: np.ndarray
    edge_targets: np.ndarray
    generator_bus: np.ndarray
    generator_features: np.ndarray
    labels: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def as_numpy_dict(self) -> dict[str, np.ndarray]:
        """Flatten the sample into arrays suitable for ``np.savez``."""

        output = {
            "scenario": np.asarray([self.scenario], dtype=np.int64),
            "node_ids": self.node_ids,
            "edge_ids": self.edge_ids,
            "edge_index": self.edge_index,
            "node_features": self.node_features,
            "edge_features": self.edge_features,
            "edge_targets": self.edge_targets,
            "generator_bus": self.generator_bus,
            "generator_features": self.generator_features,
        }
        output.update({f"label__{key}": value for key, value in self.labels.items()})
        return output


def _ordered(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.sort_values(key, kind="stable").reset_index(drop=True)


def _float_matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return np.empty((len(frame), 0), dtype=np.float32)
    return frame[available].to_numpy(dtype=np.float32, copy=True)


def _active_set_labels(branch: pd.DataFrame, gen: pd.DataFrame, tol: float) -> dict[str, np.ndarray]:
    rate = branch["rate_a"].to_numpy(dtype=np.float64)
    flow = np.maximum(
        np.hypot(branch["pf"].to_numpy(), branch["qf"].to_numpy()),
        np.hypot(branch["pt"].to_numpy(), branch["qt"].to_numpy()),
    )
    finite_limit = np.isfinite(rate) & (rate > 0)
    loading = np.zeros_like(flow)
    np.divide(flow, rate, out=loading, where=finite_limit)
    labels: dict[str, np.ndarray] = {
        "branch_loading": loading.astype(np.float32),
        "branch_active": (finite_limit & (loading >= 1.0 - tol)).astype(np.int8),
    }

    if not gen.empty:
        p = gen["p_mw"].to_numpy(dtype=np.float64)
        pmin = gen["min_p_mw"].to_numpy(dtype=np.float64)
        pmax = gen["max_p_mw"].to_numpy(dtype=np.float64)
        scale = np.maximum(1.0, np.maximum(np.abs(pmin), np.abs(pmax)))
        labels["gen_p_lower_active"] = (np.abs(p - pmin) <= tol * scale).astype(np.int8)
        labels["gen_p_upper_active"] = (np.abs(p - pmax) <= tol * scale).astype(np.int8)
    return labels


def compile_scenario(
    scenario: int,
    bus: pd.DataFrame,
    branch: pd.DataFrame,
    gen: pd.DataFrame,
    *,
    active_tolerance: float = 1e-3,
) -> GraphSample:
    """Compile one scenario into a stable graph representation.

    Bus identifiers are remapped to contiguous local indices while the original
    IDs are retained. This permits batching grids with arbitrary MATPOWER IDs.
    """

    bus = _ordered(bus.loc[bus["scenario"] == scenario], "bus")
    branch = _ordered(branch.loc[branch["scenario"] == scenario], "idx")
    gen = _ordered(gen.loc[gen["scenario"] == scenario], "idx")
    if bus.empty:
        raise ValueError(f"Scenario {scenario} has no buses")

    node_ids = bus["bus"].to_numpy(dtype=np.int64, copy=True)
    local = {int(node_id): index for index, node_id in enumerate(node_ids)}
    try:
        edge_index = np.vstack([
            branch["from_bus"].map(local).to_numpy(dtype=np.int64),
            branch["to_bus"].map(local).to_numpy(dtype=np.int64),
        ])
        generator_bus = gen["bus"].map(local).to_numpy(dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Scenario {scenario} references an unknown bus") from exc
    if np.isnan(edge_index.astype(float)).any():
        raise ValueError(f"Scenario {scenario} contains dangling branch endpoints")

    labels = _active_set_labels(branch, gen, active_tolerance)
    return GraphSample(
        scenario=int(scenario),
        node_ids=node_ids,
        edge_ids=branch["idx"].to_numpy(dtype=np.int64, copy=True),
        edge_index=edge_index,
        node_features=_float_matrix(bus, NODE_FEATURES),
        edge_features=_float_matrix(branch, EDGE_FEATURES),
        edge_targets=_float_matrix(branch, EDGE_TARGETS),
        generator_bus=generator_bus,
        generator_features=_float_matrix(gen, GEN_FEATURES),
        labels=labels,
        metadata={
            "n_nodes": len(bus),
            "n_edges": len(branch),
            "n_generators": len(gen),
            "node_features": [c for c in NODE_FEATURES if c in bus],
            "edge_features": [c for c in EDGE_FEATURES if c in branch],
            "generator_features": [c for c in GEN_FEATURES if c in gen],
        },
    )

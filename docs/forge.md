# GridFM Forge

GridFM Forge turns solved PF/OPF scenarios into research-grade, graph-native datasets.
It is intentionally framework-neutral: the canonical artifact is a set of compressed
NumPy graph shards plus a cryptographic manifest, not a PyTorch- or JAX-specific object.

## Why it exists

Raw Parquet tables are excellent for generation and analytics, but they leave several
high-impact research decisions implicit:

- graph indexing and feature order;
- topology-aware train/validation/test splits;
- active-constraint labels;
- source provenance and schema identity;
- reproducible conversion into model-ready samples.

Forge makes those decisions explicit and versioned.

## Build a graph dataset

```bash
gridfm-forge build path/to/network/raw path/to/network/forge \
  --seed 42 \
  --train-fraction 0.8 \
  --validation-fraction 0.1
```

The output contains:

```text
forge/
├── manifest.json
├── index.json
└── shards/
    ├── train/
    ├── validation/
    └── test/
```

Each scenario shard stores:

- contiguous `edge_index` and original bus/branch identifiers;
- node, edge and generator feature matrices;
- AC branch-flow targets;
- branch loading and active thermal-limit labels;
- active lower/upper generator-bound labels.

## Split semantics

A topology signature is derived from branch endpoints and in-service status. All
scenarios with the same topology signature are assigned to the same split using a
seeded content hash. This prevents topology leakage while remaining deterministic and
independent of scenario generation order.

## Provenance

`manifest.json` records the complete Forge configuration, Arrow schemas and SHA-256
fingerprints of all source tables. A trained checkpoint can therefore cite the exact
physical dataset from which it was produced.

## Research roadmap

The next layers should build on the same contract rather than introduce parallel
formats:

1. N-k and multi-topology scenario families with structural OOD splits.
2. Solver certificates: duals, KKT residuals, infeasibility and repair trajectories.
3. PTDF/LODF, spectral and cycle-space structural features.
4. Temporal trajectories and event graphs for dynamic security assessment.
5. Dataset cards, benchmark task registry and evaluation harness.
6. Optional zero-copy adapters for PyTorch Geometric, DGL and Jraph.
7. Distributed execution and object-store manifests for billion-edge corpora.

The long-term target is not merely a data generator. It is a reproducible operating
system for power-grid foundation-model research.

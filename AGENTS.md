# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

**gridfm-datakit** is a Python library for generating synthetic power grid datasets for machine learning solvers (power flow and optimal power flow). It handles realistic perturbations of loads, generator dispatches, network topologies, and branch parameters at scale (up to 30k buses for PF, 10k for OPF).

Key entry points: CLI via `gridfm_datakit` command, Jupyter interface in `scripts/interactive_interface.ipynb`, or direct Python API (`generate_power_flow_data`, `generate_power_flow_data_distributed` from `gridfm_datakit.generate`).

## Development Setup

### Quick Start

```bash
# Create virtualenv in parent directory (Code/FM) as per .venv location
python3 -m venv ../.venv
source ../.venv/bin/activate

# Install dev + test dependencies
pip install -e ".[dev,test]"

# Install Julia + PowerModels (one-time setup)
gridfm_datakit setup_pm
```

The `.venv` lives in `/Users/carmine/Code/FM/.venv` (Code/FM folder, not inside gridfm-datakit).

### Common Commands

- **Run tests**: `pytest tests/ -v` (add `-n <cores>` for parallel runs via pytest-xdist; set `SKIP_LARGE_GRIDS=1` in CI to skip slow large-grid tests)
- **Run one test**: `pytest tests/test_network.py::test_my_function -v -s`
- **Format + lint**: `pre-commit run --all` (ruff-check, ruff-format, flake8, trailing-whitespace, etc.)
- **Type check**: None configured; ruff/flake8 are the gates
- **Build docs**: `mkdocs build; mkdocs serve` (docs live in `docs/`, config in `mkdocs.yml`)
- **Generate data**: `gridfm_datakit generate scripts/config/default.yaml` (sample configs in `scripts/config/`)

## Code Architecture

### Core Modules

**gridfm_datakit/generate.py** — Main entry point
- `generate_power_flow_data()` and `generate_power_flow_data_distributed()` orchestrate the pipeline
- Handles config setup, load scenario generation, topology/generation/admittance perturbations, scenario processing, and output saving
- Uses multiprocessing (Process, Manager) to parallelize scenario chunks

**gridfm_datakit/network.py** — Grid representation
- `Network` class wraps a power grid (MATPOWER `.m` or PGLib)
- Methods: `get_pglib_file_path()`, `load_net_from_file()`, `load_net_from_pglib()`
- Stores buses, branches, generators, costs; enforces MATPOWER indexing conventions (see `idx_*.py` in utils)

**gridfm_datakit/process/** — Scenario processing
- `process_network.py`: `process_scenario_chunk()` and mode-specific functions (`process_scenario_pf_mode()`, `process_scenario_opf_mode()`)
- `solvers.py`: Wraps Julia/PowerModels solvers (AC-PF, DC-PF, OPF); calls out to `juliacall` for Newton-Raphson, Ipopt
- `solver_output.py`: Parses Julia solver results into DataFrames

**gridfm_datakit/perturbations/** — Scenario mutations
- `load_perturbation.py`: Generates load scenarios with global scaling + per-bus noise (preserves temporal/spatial correlations); see `load_scenarios_to_df()`, `plot_load_scenarios_combined()`
- `topology_perturbation.py`: N-k outages (random or exhaustive) for branches/generators
- `generator_perturbation.py`: Cost permutation or perturbation for OPF diversity
- `admittance_perturbation.py`: Random scaling of R/X per branch

**gridfm_datakit/utils/** — Helpers
- `column_names.py`: Output column definitions (`BUS_COLUMNS`, `GEN_COLUMNS`, `BRANCH_COLUMNS`, etc.)
- `idx_*.py`: MATPOWER indexing (bus/branch/gen/cost array indices)
- `param_handler.py`: `NestedNamespace` config wrapper; generator factories (`get_load_scenario_generator()`, etc.)
- `stats.py`: Data distribution plots
- `random_seed.py`: Seeding utilities

**gridfm_datakit/save.py** — Output writing
- `save_node_edge_data()` writes scenario results to parquet (bus_data, gen_data, branch_data, y_bus_data, runtime_data)

**gridfm_datakit/cli.py** — CLI
- Commands: `generate`, `validate`, `stats`, `plots`, `setup_pm`
- Mode reading from `args.log` (YAML dump of config)

**gridfm_datakit/validation.py** — Data integrity checks
- Constraint validation (voltage, branch flows, power balance) for generated scenarios

### Data Flow

1. Load network (MATPOWER or PGLib)
2. Generate load scenarios (global + local noise)
3. For each scenario:
   a. Apply topology, generation, admittance perturbations
   b. Solve PF/OPF (AC or DC, fast or Ipopt-based)
   c. Extract features (per-bus, per-gen, per-branch) and outcomes
   d. If violations → include in PF dataset; if feasible → OPF dataset
4. Write to parquet (bus_data, gen_data, branch_data, y_bus_data, runtime_data)

### Key Patterns

- **Configuration as dict/NestedNamespace**: YAML → `NestedNamespace(**config)` for nested access (e.g., `args.settings.mode`)
- **Multiprocessing with Manager**: Shared dict/list for collecting results across worker processes
- **Julia interop via juliacall**: `from juliacall import Main as jl; jl.seval(...)` for PowerModels solvers
- **Pandas DataFrames**: All output is parquet-backed DataFrames (rows = scenarios/buses/branches, columns = features)
- **MATPOWER conventions**: MATPOWER array indices hardcoded in `idx_*.py` (e.g., bus type, voltage, load, generation)

### Config Schema

Config is YAML with top-level keys: `network`, `load`, `topology_perturbation`, `generation_perturbation`, `admittance_perturbation`, `settings`. See `README.md` and sample configs in `scripts/config/` for parameter docs.

**settings** keys control the generation:
- `mode: "pf"` or `"opf"` — which dataset to produce
- `num_processes` — worker count for scenario processing
- `data_dir` — output path (relative to project root)
- `large_chunk_size` — batch size for scenario processing before IO
- `seed` — global random seed (null = auto-generated; reproducibility requires all config to match)
- `pf_fast`, `dcpf_fast` — use fast DC solvers from PowerModels or Ipopt-based ones

## Testing Notes

- Tests in `tests/` mirror the module structure (e.g., `test_network.py` tests `network.py`)
- Heavy Julia/solver tests (e.g., `test_solve.py`) may timeout or require real grids; CI skips large grids with `SKIP_LARGE_GRIDS=1`
- Coverage omits CLI, interactive, validation modules (see `coveragerc`)
- Reference data in `tests/reference_data/` for reproducibility checks

## Code Style & Linting

- **Ruff**: Format + lint (check), configured in pre-commit
- **Flake8**: Additional linting (ignore E501 long lines, W503 line breaks, E203 whitespace)
- **Pre-commit**: Runs on commit; CI enforces all checks on PR (`pre-commit run --verbose --all-files`)
- **Google-style docstrings** expected for functions/classes
- **Type hints** required (ruff will complain)

## CI/CD

- **ci-build.yaml**: Pre-commit (format + lint), security (bandit), unit tests (pytest with Julia caching), CodeQL
- **deploy_docs.yaml**: MkDocs build on release
- **release.yaml**: PyPI publish on tag

## Important Notes

- **Julia setup**: First use of Julia in a new environment will precompile PowerModels/Ipopt (slow; cached by CI). Local development: `python -c "from juliacall import Main as jl; jl.seval('using PowerModels, Ipopt, Memento')"` to warm the cache.
- **Large grids**: Some networks (e.g., case10000_goc) don't converge with `pf_fast=true`; switch to Ipopt-based solvers or set `pf_fast: false`.
- **Reproducibility**: Setting `seed` in config is necessary but not sufficient; all other config parameters must match to reproduce exact scenarios.
- **Output**: Generated data lives in `{data_dir}/{network.name}/raw/` with parquet files, logs (tqdm, error, args), and HTML scenario plots.

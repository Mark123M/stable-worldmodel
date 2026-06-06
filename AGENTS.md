# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`stable-worldmodel` (`swm`) is a platform for reproducible world-model research. It provides one interface for three stages: **collect data → train a model → evaluate with model-predictive control (MPC)**, across a large suite of Gymnasium environments. The library favors duck-typed `Protocol` extension points over inheritance so research code only implements the model/objective.

## Commands

Tooling is `uv`. The package builds with the `uv_build` backend (no `setup.py`).

```bash
# Dev setup (Python 3.10–3.12; lerobot extra needs 3.12+)
uv venv --python=3.10 && source .venv/bin/activate
uv sync --extra all --group dev

# Tests
uv run --group dev pytest                          # full suite
uv run --group dev pytest tests/data/test_lance.py # one file
uv run --group dev pytest tests/test_policy.py::test_name  # one test
uv run --group dev pytest --cov                    # with coverage (omits envs/)

# Lint / format (pre-commit drives ruff; CI runs pre-commit as a gate)
pre-commit run --all-files
uv run ruff format . && uv run ruff check --fix .

# Build & docs
uv build
uv run --group dev mkdocs serve   # docs live in docs/, config mkdocs.yaml

# CLI (entry point `swm` = stable_worldmodel.cli:app)
swm datasets | inspect <name> | envs | fovs <EnvId> | checkpoints | convert <name> --dest-format video
```

Ruff config (in `pyproject.toml`): **line-length 79, single quotes, 4-space indent, py310+ target**. Match this in new code.

## Optional-dependency gating

Dependencies are split into extras: `train`, `env`, `format`, `lerobot`, and `all` (= train+env+format). The base install is intentionally minimal. Two consequences:

- Library code imports heavy/optional deps **lazily inside functions**, not at module top (see `World.collect`, `data/formats/*`). Preserve this — a top-level `import h5py` would break base installs.
- Tests for optional features call `pytest.importorskip('<dep>')` (e.g. `lerobot`, `stable_pretraining`). Gate new optional-dep tests the same way rather than adding hard imports.

## Architecture

### The info dict is the universal data structure
Everything flows through a single `info` dict. `MegaWrapper` (`wrapper/default.py`) lifts *all* observations (pixels, state, goal, per-key factors-of-variation) into `info`. Tensor/array values are shaped `(num_envs, time, ...)`. Policies consume `info`; `World.collect` turns each non-underscore `info` key into a dataset column. Keys prefixed `_` (e.g. `_needs_flush`) are control signals, never persisted.

### `World` — the orchestrator (`world/world.py`)
Bundles a vectorized `EnvPool` (steps N envs in parallel, can mask terminated envs) + the `MegaWrapper` preprocessing chain + a rollout loop. The rollout core is `_run_iter`, a generator yielding `(env_idx, ep_count)` on each episode completion — this generator design is what lets `collect()` stream episodes to disk without threads. Public surface: `set_policy()`, `collect()`, `evaluate()`. `evaluate()` has two modes: **episodic** (`episodes=N`, auto-reset) and **dataset-driven** (start/goal states pulled from a dataset, one env per episode, `reset_mode='wait'`).

### Policies (`policy.py`) and the MPC loop
`BasePolicy` (aliased `Policy`) subclasses: `RandomPolicy`, `ExpertPolicy`, `FeedForwardPolicy` (single forward pass — GCBC-style), and `WorldModelPolicy` (planning). `_prepare_info` applies per-key `process` (normalizers) and `transform` (image transforms) before the model sees data.

`WorldModelPolicy` runs receding-horizon MPC: it keeps a per-env action buffer, replans only when a buffer empties, and calls a `Solver`. Planning behavior is set by `PlanConfig` (`horizon`, `receding_horizon`, `history_len`, `action_block` = frameskip, `warm_start`). Warm-start carries the unused tail of the previous plan into the next `init_action`.

### Solvers (`solver/`)
Sampling (CEM, iCEM, MPPI, PredictiveSampling, CategoricalCEM) and gradient/constrained (GradientSolver, PGD, Lagrangian) optimizers. All satisfy the `Solver` protocol (`solver/solver.py`): `configure(action_space, n_envs, config)` then `solve(info_dict, init_action) -> {'actions': ...}`. They optimize against a world model's cost (the model implements the `Costable` protocol: `get_cost`/`criterion`). `solver/callbacks/` holds optimization-loop hooks.

### Protocols are the extension contract
Add capabilities by satisfying a `Protocol`, not by subclassing:
- `Actionable.get_action` / `Costable.get_cost` (`protocols.py`) — what a model exposes to a policy/solver.
- `Transformable.transform`/`inverse_transform` — normalizers/scalers (`data/normalization.py`).
- `Solver` (`solver/solver.py`), `Format`/`Writer` (`data/format.py`).

Checkpoints are loaded by **scanning** a saved module tree for the protocol method: `AutoActionableModel`/`AutoCostModel` (`policy.py`) walk children looking for `get_action`/`get_cost`. A model is usable as long as it exposes the right method somewhere in its hierarchy.

### Data layer (`data/`)
A **format registry** (`data/format.py`) abstracts on-disk layout. Built-in formats register themselves on import when their extra is present: `lance` (default), `hdf5`, `folder`, `video`, `lerobot` (read-only). `swm.data.load_dataset(path, num_steps=...)` auto-detects format; `swm.data.convert(...)` migrates between them. Add a format by subclassing `Format`, implementing `detect`/`open_reader`/`open_writer`, and decorating `@register_format`.

`Dataset` (`data/dataset.py`) is the reader base: subclasses provide `column_names` + `_load_slice`; clip indexing, `__getitem__`, `load_chunk`, `load_episode` are derived. Composition wrappers: `MergeDataset` (join columns), `ConcatDataset` (stack episodes), `GoalDataset` (sample goals). `ReplayBuffer` (`data/buffer.py`) is an in-memory `Writer` you can pass to `World.collect(writer=...)` instead of a path.

### Environments and factors of variation (`envs/`, `spaces.py`)
Envs register under the `swm/` namespace via `register()` in `envs/__init__.py` (which tracks `WORLDS` / `DISCRETE_WORLDS`). Each env exposes a `variation_space` — an extended `swm.spaces.Dict` of independently controllable **factors of variation (FoV)** (lighting, textures, dynamics, morphology). This is the core mechanism for zero-shot generalization testing: resets sample/set FoVs via `reset_variation_space` and the `options={'variation': [...], 'variation_values': {...}}` reset API. `swm/spaces.py` extends Gym `Discrete`/`Box`/`Dict` with stateful value tracking on top of this. Adding an env only requires conforming to the Gymnasium interface.

### World models (`wm/`)
Baseline implementations: `lewm` (LeWM), `prejepa` (DINO-WM), `pldm`, `gcrl` (GCBC/GCIVL/GCIQL), `tdmpc2`. Each subpackage pairs `<name>.py` (the `nn.Module` with `encode`/`predict`/`rollout`) with `module.py` (training/wiring). `wm/probes.py` and `wm/loss.py` (e.g. `SIGReg`) support representation evaluation and JEPA-style objectives.

### Training (`scripts/train/`)
Reference training scripts (`lewm.py`, `prejepa.py`, `gcbc.py`, …) are **Hydra apps** (configs in `scripts/train/config/`) built on `stable-pretraining` (spt) + PyTorch Lightning + wandb. They are not imported by the library; run them as scripts or via `swm.utils.pretraining(script_path, dataset_name, output_model_name, args=...)`, which shells out and forwards Hydra overrides. Checkpoints land under `$STABLEWM_HOME/checkpoints/`.

## Storage

All artifacts live under `$STABLEWM_HOME` (default `~/.stable_worldmodel/`), with `datasets/` and `checkpoints/` subfolders resolved by `data.utils.get_cache_dir`. `load_dataset` will also resolve and download datasets from the HuggingFace Hub by repo id.

# SPET

Code and datasets for **SPET: Historical Position Transfer for Streaming Data-Driven Optimization**.

## Requirements

Python 3.11. Install the numerical dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Task-allocation data also use `requirements-cases.txt`. The database interface uses PostgreSQL, BenchBase, and a JDK; recorded versions are in `data/database.zip` under `setup/RUNTIME_MANIFEST.json`.

## Quick start

```bash
python src/main.py
```

The default run executes **5 windows** of F4/D4 with data seed 4101. Outputs are saved in a new directory under `results/runs/`.

For a complete **20-window** scenario:

```bash
python src/main.py --function 4 --drift 4 --data-seed 4101 --windows 20 --output results/runs/f4_d4
```

Use a new output directory for each run. Functions: 1–8; drifts: 1–5; repetition data seeds: 4101–4110; calibration seed: 3101. Defaults and the current-GP weight are in `src/config.py`.

## Directory structure

```text
src/       SPET implementation and execution entry points
data/      Input datasets and initialization parameters
results/   Recorded experiments and summary tables
```

`src/main.py` runs benchmark scenarios; `src/SPET.py` creates SPET policies; `src/sensitivity.py` provides the historical-source-count variant; `src/dataset.py` loads or extracts datasets.

- `src/core/`: surrogate modeling, historical-position transfer, search, and feedback updates.
- `src/benchmark/`: objective functions, drift transformations, and budgeted evaluation.
- `src/cases/`: database configuration and task-allocation interfaces.

## Data and results

| Location | Contents |
| --- | --- |
| `data/prior.json` | Surrogate initialization parameters. |
| `data/benchmark.zip` | 440 datasets: 40 scenarios × 11 seeds, with observations, objective states, and alarm schedules. |
| `data/task_allocation.zip` | Prepared inputs, priors, and outcomes for five teams. |
| `data/database.zip` | Experiment designs, runtime versions, and the BenchBase license. |
| `results/*.zip` | Benchmark, calibration, sensitivity, database, and task-allocation run records. |
| `results/summary/` | CSV tables of task metrics, case results, sensitivity, and historical-source use. |

Benchmark data are read directly from the archive. Extract application data with:

```bash
python src/dataset.py task_allocation
python src/dataset.py database
```

The extracted directories are `data/task_allocation/` and `data/database/`.

Benchmark records contain query positions and responses (`queries.json`), window errors (`windows.json`), recovery trajectories (`recovery.json`), and run summaries (`task.json`). **Benchmark `task.json` metrics are raw objective gaps; `results/summary/benchmark.csv` reports metrics normalized against passive sampling.** The recorded identifiers `GP_POSITIONS` and `R2` refer to SPET.

`CONTENTS.json` inside each archive and the root `CHECKSUMS.sha256` list file checksums. `资料说明.txt` gives a short Chinese directory guide.

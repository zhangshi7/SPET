"""Application boundary around the frozen five method implementations.

No optimizer, posterior, memory rule, or candidate source is reimplemented here.
The old R2 constructor has no prior argument.  Its loader is replaced only for
the duration of construction, under a process-local lock, and restored even if
construction fails.  Use process workers; do not instantiate the old policies
concurrently outside this factory in a thread.  Later fitting reads the copied
policy.bundle and never relies on the temporary loader.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

sys.dont_write_bytecode = True

import numpy as np

import _paths
CORE = Path(__file__).resolve().parents[1] / "core"

import baseline_loader
import position_policy

if Path(baseline_loader.__file__).resolve() != (CORE / "baseline_loader.py").resolve():
    raise ImportError("Unexpected baseline_loader implementation")
if Path(position_policy.__file__).resolve() != (CORE / "position_policy.py").resolve():
    raise ImportError("Unexpected frozen position policy implementation")

baseline_loader.configure_policy_paths()


ARMS = ("R2",)
_CONSTRUCTION_LOCK = threading.RLock()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def validate_prior(prior_bundle, dimension):
    """Validate shape only; provenance and split isolation belong to the runner."""
    if not isinstance(prior_bundle, dict):
        raise TypeError("An explicit JSON-compatible learned prior bundle is required")
    prior = prior_bundle["global_fallback_prior"]
    centers = np.asarray(prior["rbf_centers"], dtype=float)
    weights = np.asarray(prior["rbf_prior"]["weight_prior"], dtype=float)
    if centers.ndim != 2 or centers.shape[1] != dimension or not len(centers):
        raise ValueError("Application prior center dimension differs from the decision space")
    if weights.shape != (len(centers) + 1,):
        raise ValueError("Application RBF prior head does not match its centers")
    parameters = np.asarray([
        prior["gpr_prior"]["length_multiplier"], prior["gpr_prior"]["noise"],
        prior["rbf_prior"]["width"], prior["rbf_prior"]["ridge"],
    ], dtype=float)
    if not all(np.isfinite(item).all() for item in (centers, weights, parameters)):
        raise ValueError("Nonfinite application prior")
    if np.any(parameters <= 0):
        raise ValueError("Prior kernel and regularization parameters must be positive")
    return _digest(prior_bundle)


def train_application_prior(tasks, seed):
    """Learn one team's prior from its five public 50-row source windows.

    Each task is a mapping with X (50 by d), y (50,), and optional context_id.
    The first 25 rows are support and last 25 query, retaining their preregistered
    order.  All 250 labels are historical training data.  No method's active
    observations, diagnostic/lockbox data, or another team's tasks belong here.
    This function performs the complete original GPR/RBF candidate learning;
    its fixed legacy objective is not selected using application outcomes.
    """
    import conditional_meta
    learner_file = Path(conditional_meta.__file__).resolve()
    expected = (CORE / "conditional_meta.py").resolve()
    if learner_file != expected:
        raise ImportError("Unexpected original prior learner module")
    if len(tasks) != 5:
        raise ValueError("The registered application prior uses five source windows")
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    source_tasks, source_audit = [], []
    dimension = None
    for index, task in enumerate(tasks):
        x, y = np.asarray(task["X"], float), np.asarray(task["y"], float).reshape(-1)
        if (x.ndim != 2 or len(x) != 50 or len(y) != 50 or not x.shape[1]
                or not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError("Prior source windows require exactly 50 finite configuration/loss rows")
        if dimension is None:
            dimension = x.shape[1]
        if x.shape[1] != dimension:
            raise ValueError("Source window decision coordinates have inconsistent dimensions")
        source_tasks.append(conditional_meta.Task(
            int(seed), 0, 0, index,
            x[:25].copy(), y[:25].copy(), x[25:].copy(), y[25:].copy()))
        source_audit.append({
            "context_id": str(task.get("context_id", index)),
            "n": 50, "dimension": dimension,
            "support_row_indices": list(range(25)),
            "query_row_indices": list(range(25, 50)),
            "X_sha256": hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
            "y_sha256": hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
            "unique_configurations": len(np.unique(x, axis=0)),
            "distinct_loss_values": len(np.unique(y)),
        })
    library = conditional_meta.build_cluster_library(source_tasks, int(seed))
    prior = conditional_meta.choose_cluster_prior(library, "legacy")
    bundle = {
        "global_fallback_prior": prior,
        "application_prior_audit": {
            "schema_version": "fifar-shared-source-prior-v1",
            "seed": int(seed), "dimension": dimension,
            "training_rows": 250, "support_rows": 125, "query_rows": 125,
            "source_windows": source_audit,
            "required_scope": "one_team_public_raw_month4_windows_only",
            "data_scope_enforcement": "runner_must_validate_source_partition_and_team_ids",
            "active_outcomes_allowed": False,
            "diagnostic_or_lockbox_outcomes_allowed": False,
            "original_learner_file": str(learner_file),
            "original_learner_sha256": hashlib.sha256(learner_file.read_bytes()).hexdigest(),
            "original_functions": ["build_cluster_library", "choose_cluster_prior"],
            "objective_name": "legacy",
            "objective_weights_nrmse_rank_topk": list(conditional_meta.OBJECTIVES["legacy"]),
            "objective_chosen_by_application_validation": False,
            "rbf_center_count": len(prior["rbf_centers"]),
            "gpr_length_grid": list(conditional_meta.GPR_LENGTHS),
            "gpr_noise_grid": list(conditional_meta.GPR_NOISES),
            "rbf_width_grid": list(conditional_meta.RBF_WIDTHS),
            "rbf_ridge_grid": list(conditional_meta.RBF_RIDGES),
            "rbf_weight_prior_adaptation_steps": 3,
            "normalization": "original_support_only_mean_std",
            "gpr_candidate_curve": library["gpr_curve"],
            "rbf_candidate_curve": library["rbf_curve"],
            "learned_prior_sha256": _digest(prior),
            "training_wall_seconds": time.perf_counter() - started_wall,
            "training_cpu_seconds": time.process_time() - started_cpu,
        },
    }
    validate_prior(bundle, dimension)
    return bundle


def make_policy(arm, lower, upper, seed, prior_bundle, alpha):
    """Create an unchanged numerical method with an explicit application prior.

    DIRECT_POSITIONS is the matched-quota R2 position-source ablation, including
    its original historical GP computation.  alpha controls only the extra
    current-only GP role; original DOLEM conditional weights remain adaptive.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown application arm {arm!r}; expected one of {ARMS}")
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if (lower.ndim != 1 or not len(lower) or upper.shape != lower.shape or
            not np.isfinite(np.r_[lower, upper]).all() or np.any(upper <= lower)):
        raise ValueError("Finite ordered vector bounds are required")
    prior_digest = validate_prior(prior_bundle, len(lower))
    frozen_bundle = deepcopy(prior_bundle)
    with _CONSTRUCTION_LOCK:
        if arm in ("R2", "DIRECT_POSITIONS"):
            original = position_policy.original
            saved_loader = original.load_prior
            try:
                original.load_prior = lambda: deepcopy(frozen_bundle)
                policy = position_policy.PositionPolicy(
                    "GP_POSITIONS" if arm == "R2" else "DIRECT_POSITIONS",
                    lower, upper, seed, alpha=alpha)
            finally:
                original.load_prior = saved_loader
            if _digest(policy.bundle) != prior_digest:
                raise AssertionError("R2 did not retain the explicit application prior")
            if original.load_prior is not saved_loader:
                raise AssertionError("Frozen prior loader was not restored")
    policy.application_adapter_audit = {
        "arm": arm,
        "dimension": len(lower),
        "objective_direction": "minimize",
        "prior_bundle_sha256": prior_digest if arm in ("R2", "DIRECT_POSITIONS", "DOLEM") else None,
        "native_prior_mechanism_preserved": arm in ("DASE", "TRACE"),
        "frozen_optimizer_reused": True,
        "prior_loader_patch_scope": "constructor_only_restored" if arm in ("R2", "DIRECT_POSITIONS") else "none",
        "position_only_historical_transfer": arm in ("R2", "DIRECT_POSITIONS"),
        "prior_provenance_is_not_validated_by_constructor": True,
    }
    return policy

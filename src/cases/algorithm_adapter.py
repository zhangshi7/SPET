"""Audited application boundary around the frozen R2 and comparison policies.

The adapter uses normalized coordinates and positive, measured latency losses.
It does not implement a database, remap a knob, create passive observations,
modify the old optimizers, or expose held-out measurements to a policy.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import inspect
import json
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
import _paths
APPLICATION = Path(__file__).resolve().parent

import numpy as np
import application_methods as _frozen

if Path(_frozen.__file__).resolve() != (APPLICATION / "application_methods.py").resolve():
    raise ImportError("Unexpected application method adapter")

ARMS = ("R2",)
ACTIVE_ROUNDS = 8
BATCH_SIZE = 4
PASSIVE_ROWS = 50


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _arrays(X, y, dimension, *, rows):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)
    if X.shape != (rows, dimension) or y.shape != (rows,):
        raise ValueError(f"Expected {rows} rows and decision dimension {dimension}")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite configuration or observed latency")
    if np.any(X < 0) or np.any(X > 1) or np.any(y <= 0):
        raise ValueError("Normalized configurations must be in [0,1]; latency must be positive")
    return X.copy(), y.copy()


def seed_algorithm_streams(seed):
    """Call before constructing one trial in its own process (never threads)."""
    random.seed(int(seed))
    np.random.seed(int(seed) % 2**32)
    import torch
    torch.manual_seed(int(seed) % (2**63 - 1))
    torch.set_num_threads(1)


def train_database_prior(tasks, seed, *, source_partition):
    """Use the complete frozen prior learner on five independent 50-row windows.

    source_partition is an explicit manifest label, not proof of independence:
    the experiment runner must bind the five datasets and enforce that they do
    not contain formal evaluation or policy-selected active outcomes.
    """
    if not isinstance(source_partition, str) or not source_partition.strip():
        raise ValueError("An explicit independent prior-source partition is required")
    if len(tasks) != 5:
        raise ValueError("The shared prior uses exactly five source windows")
    dimension = np.asarray(tasks[0]["X"]).shape[1]
    for task in tasks:
        _arrays(task["X"], task["y"], dimension, rows=PASSIVE_ROWS)
    bundle = _frozen.train_application_prior(tasks, seed)
    audit = bundle["application_prior_audit"]
    audit.update(schema_version="database-shared-source-prior-v1",
                 required_scope=source_partition,
                 data_scope_enforcement="runner_validates_frozen_independent_partition",
                 frozen_numerical_learner_reused=True)
    _frozen.validate_prior(bundle, dimension)
    return bundle


class AuditedPolicy:
    """Fail-closed 50-passive / 8x4-active boundary; never cache DB feedback."""

    def __init__(self, policy, arm, dimension, seed):
        self.policy, self.arm = policy, arm
        self.dimension, self.seed = int(dimension), int(seed)
        self.observed_windows = 0
        self.last_searched_window = -1
        self.passive_rows = 0
        self.active_attempted = 0
        self.active_returned = 0
        self.failed = False
        self.search_ledgers = []

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def observe_passive(self, X, y):
        if self.failed:
            raise RuntimeError("This failed trial must be archived and reconstructed")
        X, y = _arrays(X, y, self.dimension, rows=PASSIVE_ROWS)
        result = self.policy.observe_passive(X, y)
        self.observed_windows += 1
        self.passive_rows += PASSIVE_ROWS
        return result

    def search(self, objective_callable, generations=ACTIVE_ROUNDS, batch_size=BATCH_SIZE):
        if self.failed:
            raise RuntimeError("This failed trial must be archived and reconstructed")
        if (generations, batch_size) != (ACTIVE_ROUNDS, BATCH_SIZE):
            raise ValueError("Every method requires exactly 8 rounds x 4 true feedbacks")
        if self.observed_windows < 2 or self.last_searched_window == self.observed_windows:
            raise RuntimeError("Search requires a new post-bootstrap passive window")
        ledger = {"arm": self.arm, "window": self.observed_windows,
                  "status": "RUNNING", "calls": [], "active_attempted": 0,
                  "active_returned": 0, "objective_direction": "minimize_latency",
                  "oracle_seconds": 0.0}
        self.search_ledgers.append(ledger)

        def measured_oracle(batch):
            X = np.asarray(batch, dtype=float)
            if X.shape != (BATCH_SIZE, self.dimension) or not np.isfinite(X).all():
                raise ValueError("Optimizer did not propose a finite four-point batch")
            if np.any(X < 0) or np.any(X > 1):
                raise ValueError("Optimizer proposed outside normalized bounds")
            if len(ledger["calls"]) >= ACTIVE_ROUNDS:
                raise RuntimeError("Optimizer attempted to exceed the true-feedback budget")
            record = {"round": len(ledger["calls"]), "X": X.tolist(), "status": "CALLED"}
            ledger["calls"].append(record)
            ledger["active_attempted"] += BATCH_SIZE
            self.active_attempted += BATCH_SIZE
            begin = time.perf_counter()
            try:
                values = objective_callable(X.copy())
                _, values = _arrays(X, values, self.dimension, rows=BATCH_SIZE)
            finally:
                elapsed = time.perf_counter() - begin
                record["wall_seconds"] = elapsed
                ledger["oracle_seconds"] += elapsed
            record.update(y=values.tolist(), status="RETURNED")
            ledger["active_returned"] += BATCH_SIZE
            self.active_returned += BATCH_SIZE
            return values

        begin = time.perf_counter()
        try:
            result = self.policy.search(measured_oracle, generations, batch_size)
            if len(ledger["calls"]) != ACTIVE_ROUNDS or ledger["active_returned"] != 32:
                raise AssertionError("Policy did not consume the registered 32 observations")
            queries = result["queries"]
            query_X = np.asarray([row["x"] for row in queries], dtype=float)
            query_y = np.asarray([row["y"] for row in queries], dtype=float)
            if (not np.array_equal(query_X, np.vstack([row["X"] for row in ledger["calls"]]))
                    or not np.array_equal(query_y, np.concatenate([row["y"] for row in ledger["calls"]]))):
                raise AssertionError("Policy query log differs from actually returned feedback")
            if len(result["generation_records"]) != ACTIVE_ROUNDS:
                raise AssertionError("Policy did not record eight feedback rounds")
            ledger["status"] = "COMPLETE"
            self.last_searched_window = self.observed_windows
        except BaseException:
            ledger["status"] = "FAILED"
            self.failed = True
            raise
        finally:
            ledger["search_seconds_including_oracle"] = time.perf_counter() - begin
        result["database_adapter"] = deepcopy(ledger)
        return result

    def export_database_audit(self):
        return deepcopy({"schema_version": "database-algorithm-boundary-v1", "arm": self.arm,
                         "dimension": self.dimension, "seed": self.seed,
                         "normalized_bounds": [0.0, 1.0], "passive_rows": self.passive_rows,
                         "active_attempted": self.active_attempted, "active_returned": self.active_returned,
                         "failed": self.failed, "search_ledgers": self.search_ledgers,
                         "physical_execution_cost_accounted_by_oracle": True,
                         "passive_data_cost_accounted_by_shared_runner": True,
                         "feedback_caching_performed_by_adapter": False})

    def mechanism_audit(self):
        if hasattr(self.policy, "mechanism_audit"):
            return self.policy.mechanism_audit()
        # Original FEMS has settings and timings, but no mechanism_audit API.
        # Expose its actual source identity without inventing native counters.
        source = Path(inspect.getfile(type(self.policy))).resolve()
        settings = getattr(self.policy, "settings", None)
        return {"method": self.arm, "implementation": type(self.policy).__name__,
                "source_file": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "settings": asdict(settings) if is_dataclass(settings) else None,
                "original_mechanism_audit_available": False,
                "application_adapter_audit": deepcopy(getattr(self.policy, "application_adapter_audit", {}))}


def make_policy(arm, dimension, seed, prior_bundle=None, alpha=0.50):
    """Build an unchanged frozen arm or the public current-only BoTorch arm.

    alpha=.50 is carried from independent synthetic calibration; no DB tuning
    evidence is implied. The caller must record this provenance in its manifest.
    """
    if arm not in ARMS or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("Unknown arm or invalid decision dimension")
    lower, upper = np.zeros(dimension), np.ones(dimension)
    policy = _frozen.make_policy(arm, lower, upper, seed, prior_bundle, alpha)
    return AuditedPolicy(policy, arm, dimension, seed)

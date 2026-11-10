"""Capacity-constrained FiFAR replay with scalar-only feedback.

Policies consume prepared estimated costs. Gold labels and complete expert
decisions are loaded only inside the scorer, joined on case_id. Evaluation
cohorts are disjoint from feedback cohorts and must be scored after optimization.
"""
from __future__ import annotations
from functools import lru_cache
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "data/task_allocation/application"
DATA = ROOT / "data/task_allocation/extracted/FiFAR"
STAGE_MONTH = {"prior": 4, "engineering": 5, "diagnostic": 6, "lockbox": 7}


def allocate_capacities(proportions, size):
    """Largest remainder; equal remainders favor the fixed agent column order."""
    p = np.asarray(proportions, dtype=np.float64)
    if p.ndim != 1 or np.any(p < 0) or not np.isfinite(p).all() or p.sum() <= 0:
        raise ValueError("Invalid capacity proportions")
    exact = p / p.sum() * int(size)
    counts = np.floor(exact).astype(np.int64)
    residual = int(size) - int(counts.sum())
    order = np.lexsort((np.arange(len(p)), -(exact - counts)))
    counts[order[:residual]] += 1
    assert counts.sum() == size and (counts >= 0).all()
    return counts


def solve_assignment(costs, capacities):
    costs = np.asarray(costs, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.int64)
    if costs.ndim != 2 or costs.shape[1] != len(capacities):
        raise ValueError("Cost/capacity shape mismatch")
    if int(capacities.sum()) != costs.shape[0] or np.any(capacities < 0):
        raise ValueError("Capacity does not partition the cohort")
    if not np.isfinite(costs).all():
        raise ValueError("Nonfinite estimated costs")
    slots = np.repeat(np.arange(len(capacities)), capacities)
    rows, cols = linear_sum_assignment(costs[:, slots])
    result = np.empty(len(rows), dtype=np.int64)
    result[rows] = slots[cols]
    assert np.array_equal(np.bincount(result, minlength=len(capacities)), capacities)
    return result


@lru_cache(maxsize=1)
def _manifest():
    return json.loads((APP / "application_manifest.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=20)
def _prepared(team, month):
    with np.load(APP / "predictions" / f"team_{team}_month_{month}.npz", allow_pickle=False) as src:
        out = {k: src[k].copy() for k in src.files}
    return out


def _truth_for(case_ids, experts):
    # Read only the scorer's current cohort; never return this to the optimizer.
    import pandas as pd
    ids = np.asarray(case_ids, dtype=np.int64)
    labels = pd.read_parquet(DATA / "alert_data" / "processed_data" / "alerts.parquet", columns=["fraud_bool"]).loc[ids, "fraud_bool"].to_numpy(dtype=np.int8)
    decisions = pd.read_parquet(DATA / "synthetic_experts" / "expert_predictions.parquet", columns=experts).loc[ids, experts].to_numpy(dtype=np.int8)
    return labels, decisions


class Environment:
    dimension = 10
    lower = np.full(10, -np.log(2.0), dtype=np.float64)
    upper = np.full(10, np.log(2.0), dtype=np.float64)

    def __init__(self, team, stage, phase):
        self.team, self.stage, self.phase = int(team), str(stage), int(phase)
        if self.stage not in STAGE_MONTH or not 1 <= self.team <= 5:
            raise ValueError("Unknown team or stage")
        self.month = STAGE_MONTH[self.stage]
        manifest = _manifest()
        phases = manifest["partitions"][self.stage]
        if not 0 <= self.phase < len(phases):
            raise ValueError("Unknown phase")
        self._phase = phases[self.phase]
        self.capacity_state = self._phase["capacity_state"]
        self.experts = list(manifest["teams"][str(self.team)]["experts"])
        data = _prepared(self.team, self.month)
        lookup = {int(v): i for i, v in enumerate(data["case_ids"])}
        self._ids, self._costs, self._ai, self._caps = {}, {}, {}, {}
        proportions = manifest["teams"][str(self.team)]["capacities"][self.capacity_state]
        for which in ["feedback", "evaluation"]:
            ids = np.asarray(self._phase[f"{which}_ids"], dtype=np.int64)
            row = np.asarray([lookup[int(v)] for v in ids], dtype=np.int64)
            self._ids[which] = ids
            self._costs[which] = data["predicted_costs"][row]
            self._ai[which] = data["ai_predictions"][row]
            self._caps[which] = allocate_capacities(proportions, len(ids))
        self._gold_cache = {}
        self._feedback_pairs = set()
        self.feedback_queries = 0
        self.feedback_case_outcome_reads = 0
        self.evaluation_queries = 0

    def ids(self, which="feedback"):
        return self._ids[which].copy()

    def capacities(self, which="feedback"):
        return self._caps[which].copy()

    def assign(self, x, which="feedback"):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (10,) or not np.isfinite(x).all() or np.any(x < self.lower - 1e-12) or np.any(x > self.upper + 1e-12):
            raise ValueError("Expected ten feasible log multipliers")
        scaled = self._costs[which] * np.r_[np.exp(x), 1.0][None, :]
        return solve_assignment(scaled, self._caps[which])

    def _score(self, X, which):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if which not in self._gold_cache:
            self._gold_cache[which] = _truth_for(self._ids[which], self.experts)
        y, expert_decisions = self._gold_cache[which]
        decision_matrix = np.column_stack((expert_decisions, self._ai[which]))
        results = []
        for x in X:
            assignment = self.assign(x, which)
            predicted = decision_matrix[np.arange(len(y)), assignment]
            cost = ((y == 1) & (predicted == 0)).astype(np.float64) + 0.057 * ((y == 0) & (predicted == 1))
            results.append(float(cost.mean()))
            if which == "feedback":
                self.feedback_queries += 1
                self.feedback_case_outcome_reads += len(y)
                self._feedback_pairs.update((int(case), int(agent)) for case, agent in zip(self._ids[which], assignment))
            else:
                self.evaluation_queries += 1
        return np.asarray(results, dtype=np.float64)

    def feedback(self, X):
        """Return scalar batch mean costs; no labels, assignments or expert outcomes."""
        return self._score(X, "feedback")

    def evaluate(self, X):
        """Evaluator-only, called after saved optimizer candidates are frozen."""
        return self._score(X, "evaluation")

    def feedback_ledger(self):
        return {"configuration_evaluations": self.feedback_queries,
                "case_outcome_reads_including_repeats": self.feedback_case_outcome_reads,
                "unique_cases": len(self._ids["feedback"]) if self.feedback_queries else 0,
                "unique_case_agent_pairs": len(self._feedback_pairs),
                "expert_case_agent_pairs": sum(agent < 10 for _, agent in self._feedback_pairs),
                "evaluation_queries": self.evaluation_queries}


def load_environment(team, stage, phase):
    if isinstance(team, str) and team.startswith("team_"):
        team = int(team.split("_")[-1])
    return Environment(team, stage, phase)

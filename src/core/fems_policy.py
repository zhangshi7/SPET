"""Frozen FEMS candidate and grouped F/L interventions.

The preserved GlobalBudgetMetaDF supplies the algorithm.  This wrapper changes
only the registered feedback/local switches, GP training cap and temperature.
Instrumentation is observational: it neither consumes RNG values nor queries the
objective.  Component clocks use exclusive nested accounting, so they can be
summed without counting objective or local fitting time twice.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import sys
import time

import numpy as np

from global_budget_meta_df import GlobalBudgetMetaDF


@dataclass(frozen=True)
class FEMSSettings:
    feedback: bool = True
    local: bool = True
    gpr_cap: int = 128
    temperature: float = 0.35

    def __post_init__(self):
        if type(self.feedback) is not bool or type(self.local) is not bool:
            raise ValueError("feedback/local must be boolean frozen interventions")
        if self.gpr_cap not in (64, 128, 256):
            raise ValueError("registered GPR caps are 64, 128 and 256")
        if self.temperature not in (0.175, 0.35, 0.7):
            raise ValueError("registered temperatures are 0.175, 0.35 and 0.7")


class _ExclusiveClock:
    """Time a call tree with disjoint wall/CPU component totals."""

    def __init__(self):
        self.totals = {}
        self.stack = []

    @contextmanager
    def measure(self, component):
        frame = {"wall": time.perf_counter(), "cpu": time.process_time(),
                 "children_wall": 0.0, "children_cpu": 0.0}
        self.stack.append(frame)
        try:
            yield
        finally:
            wall = time.perf_counter() - frame["wall"]
            cpu = time.process_time() - frame["cpu"]
            self.stack.pop()
            if self.stack:
                self.stack[-1]["children_wall"] += wall
                self.stack[-1]["children_cpu"] += cpu
            total = self.totals.setdefault(component, {"wall_seconds": 0.0,
                                                       "cpu_seconds": 0.0,
                                                       "calls": 0})
            total["wall_seconds"] += max(0.0, wall - frame["children_wall"])
            total["cpu_seconds"] += max(0.0, cpu - frame["children_cpu"])
            total["calls"] += 1

    def snapshot(self):
        return {name: dict(values) for name, values in self.totals.items()}


def _timing_delta(after, before):
    return {name: {key: value - before.get(name, {}).get(key, 0)
                   for key, value in values.items()}
            for name, values in after.items()}


class FEMS(GlobalBudgetMetaDF):
    """Feedback-guided Evolution with Meta-initialized Surrogates.

    F controls *active prediction-loss routing*, not all active-label use.
    L controls both local fitted roles and their local-optimization proposals.
    Every group keeps the revealed-label champion, radius update and active FE
    budget.  Global learned priors and temporal core roles remain unchanged.
    """

    def __init__(self, lower, upper, seed, prior_bundle, *, feedback=True,
                 local=True, gpr_cap=128, temperature=0.35):
        self.settings = FEMSSettings(feedback, local, gpr_cap, temperature)
        super().__init__(lower, upper, seed, prior_bundle,
                         variant="full" if feedback else "no_feedback")
        self.gpr_fit_cap = self.settings.gpr_cap
        self.config = replace(self.config,
                              weight_temperature=self.settings.temperature)
        self._clock = _ExclusiveClock()
        self._phase = "idle"
        self._pending_weights_state = None
        self.state_trace = []

    def timing_summary(self):
        """Cumulative disjoint components; I/O and offline training excluded."""
        components = self._clock.snapshot()
        return self._summarize_timing(components)

    @staticmethod
    def _summarize_timing(components):
        return {"components": components,
                "wall_seconds": sum(v["wall_seconds"] for v in components.values()),
                "cpu_seconds": sum(v["cpu_seconds"] for v in components.values()),
                "accounting": "exclusive_nested_components",
                "objective_included": True,
                "instrumentation_included": True,
                "offline_training_included": False,
                "data_loading_and_output_io_included": False}

    def observe_passive(self, x, y):
        previous = self._phase
        self._phase = "passive"
        try:
            with self._clock.measure("passive_state"):
                return super().observe_passive(x, y)
        finally:
            self._phase = previous

    def _build_pool(self):
        with self._clock.measure("core_fit"):
            super()._build_pool()
            # The preserved parent labels its default mode cap128.  The
            # registered sensitivity interventions need truthful audit labels.
            self.route_records[-1]["mode"] = f"global_meta_cap{self.gpr_fit_cap}"

    def _build_local(self, z, y):
        if not self.settings.local:
            self.local = {}
            return
        with self._clock.measure("local_fit"):
            super()._build_local(z, y)

    def _candidates(self, train_z, train_y, radius):
        with self._clock.measure("candidate_proposal"):
            return super()._candidates(train_z, train_y, radius)

    def _predict(self, z):
        component = ("passive_prediction" if self._phase == "passive"
                     else "candidate_prediction")
        with self._clock.measure(component):
            return super()._predict(z)

    def _weights(self, names):
        with self._clock.measure("weight_routing"):
            weights = super()._weights(names)
            with self._clock.measure("instrumentation"):
                # This snapshot is taken before the current batch is queried.
                self._pending_weights_state = {
                    "weights_before_query": dict(zip(names, weights.tolist())),
                    "passive_losses_before_query": {
                        name: float(self.losses.get(name, 0.75)) for name in names},
                    "active_losses_before_query": {
                        name: float(value) for name, value in self.active_losses.items()},
                    "feedback_enabled": self.settings.feedback,
                    "local_enabled": self.settings.local,
                    "measurement_point": "after_candidate_prediction_before_objective",
                }
            return weights

    def search(self, objective_callable, generations=8, batch_size=4):
        before = self._clock.snapshot()
        previous_phase = self._phase
        self._phase = "search"
        search_index = self.search_count
        weight_snapshots, batch_outcomes = [], []
        query_contributions = []
        # This is a same-search revealed-data incumbent, not an oracle metric.
        incumbent = (float(np.min(self.passive_y[-self.config.local_passive:]))
                     if len(self.passive_y) else float("inf"))

        def measured_objective(x):
            nonlocal incumbent
            with self._clock.measure("instrumentation"):
                if self._pending_weights_state is None:
                    raise RuntimeError("objective call preceded its weight snapshot")
                weight_snapshots.append({
                    key: dict(value) if isinstance(value, dict) else value
                    for key, value in self._pending_weights_state.items()})
            with self._clock.measure("objective"):
                values = np.asarray(objective_callable(x), float).reshape(-1)
            with self._clock.measure("instrumentation"):
                batch_before = incumbent
                for query_index, value in enumerate(values):
                    old = incumbent
                    incumbent = min(incumbent, float(value))
                    query_contributions.append({
                        "generation": len(batch_outcomes),
                        "query_in_generation": query_index,
                        "incumbent_before": old,
                        "observed_value": float(value),
                        "incumbent_after": incumbent,
                        "sequential_observed_improvement": max(0.0, old - incumbent),
                        "measurement_point": "after_objective_return",
                        "interpretation": "ordered_incumbent_bookkeeping_not_causal_attribution",
                    })
                batch_outcomes.append({"incumbent_before_batch": batch_before,
                                       "incumbent_after_batch": incumbent})
            return values

        try:
            with self._clock.measure("search_state"):
                result = super().search(measured_objective, generations, batch_size)
                with self._clock.measure("instrumentation"):
                    trace = []
                    for index, generation in enumerate(result["generation_records"]):
                        row = {"search_index": search_index, "passive_window": self.window,
                               "generation": index, **weight_snapshots[index],
                               **batch_outcomes[index],
                               "active_losses_after_return": dict(generation["active_losses_after"]),
                               "observed_active_losses": dict(generation["observed_active_loss"]),
                               "sponsors": list(generation["sponsors"]),
                               "trust_radius_before_query": generation["trust_radius"],
                               "role_count": generation["model_count"],
                               "gpr_training_cap": self.gpr_fit_cap,
                               "future_values_used": False}
                        trace.append(row)
                    for contribution, query in zip(query_contributions, result["queries"]):
                        contribution["sponsor"] = query["sponsor"]
                    self.state_trace.extend(trace)
                    result["decision_state_trace"] = trace
                    result["query_contributions"] = query_contributions
                    result["fems_settings"] = asdict(self.settings)
        finally:
            self._phase = previous_phase
        components = _timing_delta(self._clock.snapshot(), before)
        result["component_timing"] = self._summarize_timing(components)
        return result


# Explicit alias for experiment harnesses that prefer a Policy suffix.
FEMSPolicy = FEMS

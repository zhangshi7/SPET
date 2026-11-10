"""Strict three-pass query-family objective on the unmodified base DB oracle."""
from __future__ import annotations
from collections import Counter
import numpy as np
from database_oracle import DatabaseOracle, settings_from_x, canonical_digest


class SemanticDatabaseOracle(DatabaseOracle):
    def __init__(self, directory, *, design, scale=.1, timeout_ms=5000):
        self.configure_design(design)  # Validation before a JVM or database can start.
        super().__init__(directory, scale=scale, timeout_ms=timeout_ms)

    def configure_design(self, design):
        if design["physical_passes_per_logical_feedback"] != 3 or design["queries_per_physical_pass"] != 32:
            raise ValueError("Frozen measurement budget is three complete 32-query passes")
        original = {"scan_filter": [1,6,12,14,19,22], "join": [2,3,5,7,8,9,20],
                    "group_order": [4,10,11,13,15,16,17,18,21]}
        if design["families"] != original:
            raise ValueError("Only the three original semantic families are allowed")
        self._registered_batches = {}
        contexts = design["prior_contexts"] + [c for s in design["test_sequences"] for c in s["contexts"]]
        for context in contexts:
            family = original[context["family_id"]]
            for split in ("feedback", "evaluation"):
                queries, passes = context[split+"_queries"], context[split+"_passes"]
                if len(queries) != 32 or {q["query"] for q in queries} != set(family):
                    raise ValueError("Batch must cover exactly one known family")
                counts = Counter((q["query"], q["seed"]) for q in queries)
                if len(passes) != 3 or any(Counter((q["query"],q["seed"]) for q in p) != counts for p in passes):
                    raise ValueError("Every pass must preserve all registered request instances")
                key = canonical_digest(queries)
                if key in self._registered_batches:
                    raise ValueError("Duplicate registered batch")
                self._registered_batches[key] = (context["id"], split, passes)

    def registered_passes(self, query_specs, context, *, evaluation=None):
        item = self._registered_batches.get(canonical_digest(query_specs))
        if item is None or item[0] != context.get("context_id"):
            raise ValueError("Unregistered batch or mismatched context")
        if evaluation is not None and (item[1] == "evaluation") != evaluation:
            raise ValueError("Feedback and held-out evaluation batches cannot be exchanged")
        return item[2]

    def warmup_queries(self, settings, query_specs, *, context, evaluation=False):
        passes = self.registered_passes(query_specs, context, evaluation=evaluation)
        return [self.run_queries(settings, queries, role="warmup", context={**context,"physical_pass":i})
                for i, queries in enumerate(passes)]

    def evaluate(self, x, query_specs, weights, *, context, role="feedback", warmup_specs=None, repeats=1):
        if role not in ("feedback", "evaluation") or repeats != 1 or warmup_specs is not None:
            raise ValueError("One logical objective per call; explicit external warmups and evaluation repetitions")
        weights = np.asarray(weights, float)
        if weights.shape != (32,) or not np.array_equal(weights, np.ones(32)):
            raise ValueError("All 96 actual requests have uniform weight")
        settings = settings_from_x(x)
        passes = self.registered_passes(query_specs, context, evaluation=role=="evaluation")
        records, costs = [], []
        for index, queries in enumerate(passes):
            record = self.run_queries(settings, queries, role=role, context={**context,"physical_pass":index})
            if any(q["status"] not in ("OK", "TIMEOUT") for q in record["response"]["queries"]):
                raise RuntimeError("Unexpected query failure; no simplified fallback")
            costs.append(float(np.mean([q["elapsed_ms"] if q["status"]=="OK" else self.timeout_ms
                                        for q in record["response"]["queries"]])))
            records.append(record)
        value = float(np.mean(costs))
        return value, {"x":np.asarray(x).tolist(),"settings":settings,"query_specs":query_specs,
            "physical_pass_queries":passes,"weights":weights.tolist(),"logical_repeats":1,
            "physical_passes":3,"objective_ms":value,"measurement_costs_ms":costs,
            "logical_feedback_FE":1 if role=="feedback" else 0,
            "physical_queries":sum(len(r["response"]["queries"]) for r in records),
            "physical_wall_seconds":sum(r["wall_seconds"] for r in records),
            "request_ids":[r["request"]["id"] for r in records]}

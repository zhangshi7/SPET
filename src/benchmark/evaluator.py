"""Evaluate SPET on a stream with a fixed alarm schedule and query budget."""
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
import numpy as np
from .oracle import LabelConsistentObjective
def now():
    return datetime.now(timezone.utc).isoformat()

def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False, default=encode)

def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                   default=encode), encoding="utf-8")
    temporary.replace(path)

def run(job, directory, snapshot, schedule, policy_factory):
    algorithm, seed, function, drift = (
        job["algorithm"], job["seed"], job["function"], job["drift"])
    algorithm_seed = seed * 1000003 + function * 1009 + drift * 101
    random.seed(algorithm_seed)
    np.random.seed(algorithm_seed % 2**32)
    oracle = LabelConsistentObjective(snapshot)
    lo, hi = oracle.get_bound(function)
    lower, upper = np.full(snapshot["dimension"], lo), np.full(snapshot["dimension"], hi)
    policy = policy_factory(lower, upper, algorithm_seed)
    stream = snapshot["test_streams"][f"{function}:{drift}"]
    if job["window_limit"]:
        stream = stream[:job["window_limit"]]
    windows, queries, recovery, generations, mechanisms = [], [], [], [], []
    timing = {clock: {name: 0.0 for name in ("observe", "search", "objective")}
              for clock in ("wall", "process")}

    def timed(name, callback, *args, **kwargs):
        w, c = time.perf_counter(), time.process_time()
        try:
            return callback(*args, **kwargs)
        finally:
            timing["wall"][name] += time.perf_counter() - w
            timing["process"][name] += time.process_time() - c

    env_now, best, passive_best = None, float("inf"), float("inf")
    total_expected = 0
    with (directory / "events.jsonl").open("w", encoding="utf-8") as events:
        def event(kind, **values):
            events.write(canonical({"kind": kind, "time_utc": now(), **values}) + "\n")
            events.flush()

        event("job_start", **job, algorithm_seed=algorithm_seed)
        for window_index, batch in enumerate(stream):
            environment = int(batch["environment"])
            if environment != env_now:
                env_now, best, passive_best = environment, float("inf"), float("inf")
            x = np.asarray(batch["X"], float)
            y = np.asarray(batch["y"], float).reshape(-1)
            schedule_key = (function, drift, environment,
                            int(batch["sample_start"]), int(batch["sample_end"]))
            schedule_row = schedule[schedule_key]
            confirmed = str(schedule_row["confirmed_drift"]).lower() == "true"
            expected = int(schedule_row["real_candidate_queries"])
            assert expected == (32 if confirmed and window_index else 0)
            # Evaluator-only consistency check is never supplied to the policy.
            check_y = oracle.evaluate(function, drift, environment, x)
            np.testing.assert_allclose(check_y, y, rtol=2e-5, atol=2e-5)
            total_expected += expected
            event("passive_reveal", window=window_index, samples=len(y),
                  expected_active_fe=expected)
            timed("observe", policy.observe_passive, x.copy(), y.copy())
            best, passive_best = min(best, float(y.min())), min(passive_best, float(y.min()))
            optimum = oracle.optimum(function, drift, environment)
            before = len(queries)
            start_error = max(0.0, best - optimum)
            active_call = 0

            def evaluate(points):
                nonlocal best, active_call
                points = np.atleast_2d(np.asarray(points, float))
                if points.shape != (4, snapshot["dimension"]):
                    raise AssertionError(f"Expected 4-by-d query batch, got {points.shape}")
                assert np.all(np.isfinite(points))
                assert np.all(points >= lower - 1e-9) and np.all(points <= upper + 1e-9)
                spent = len(queries) - before
                assert spent + len(points) <= expected, "objective budget exceeded"
                event("before_objective", window=window_index, external_round=active_call,
                      spent_before=spent, requested=len(points), budget=expected,
                      x=points.tolist(), feedback_available=False)
                values = np.asarray(timed(
                    "objective", oracle.evaluate, function, drift, environment, points),
                    float).reshape(-1)
                assert values.shape == (4,) and np.all(np.isfinite(values))
                for query_in_batch, (point, value) in enumerate(zip(points, values)):
                    old = best
                    best = min(best, float(value))
                    queries.append({
                        "window": window_index, "call": len(queries) + 1,
                        "environment": environment, "external_round": active_call,
                        "query_in_generation": query_in_batch,
                        "fe_within_alarm": len(queries) - before + 1,
                        "batch_feedback_atomic": True,
                        "budget_before_batch": expected - spent,
                        "budget_after_batch": expected - spent - len(points),
                        "x": point.tolist(), "y": float(value),
                        "best_before": old, "best_after": best,
                        "reference_gap": max(0.0, best - optimum),
                        "feedback_time": "after_current_batch_objective_return",
                    })
                event("feedback_returned", window=window_index, external_round=active_call,
                      spent_after=len(queries) - before, y=values.tolist(),
                      feedback_available=True)
                active_call += 1
                return values.copy()

            if expected:
                result = timed("search", policy.search, evaluate, generations=8, batch_size=4)
                assert active_call == 8
                policy_queries = result.get("queries", [])
                assert len(policy_queries) == expected
                for ledger, model_query in zip(queries[before:], policy_queries):
                    np.testing.assert_allclose(ledger["x"], model_query["x"], rtol=0, atol=1e-10)
                    assert np.isclose(ledger["y"], model_query["y"], rtol=1e-10, atol=1e-10)
                    for key in ("sponsor", "predictions_before_query", "prediction_before_query"):
                        if key in model_query:
                            ledger[key] = model_query[key]
                    assert ("predictions_before_query" in ledger or
                            "prediction_before_query" in ledger), "prequery predictions absent"
                generations.extend({"window": window_index, **r}
                                   for r in result.get("generation_records", []))
                audit = result.get("mechanism_audit", {})
                if callable(audit):
                    audit = audit()
                mechanisms.append({"window": window_index, "audit": audit})
                curve = [start_error] + [r["reference_gap"] for r in queries[before:]]
                recovery.append({
                    "window": window_index, "environment": environment,
                    "passive_error": max(0.0, passive_best - optimum),
                    "errors": curve,
                    "checkpoints": {str(fe): curve[fe] for fe in (0, 4, 8, 16, 32)},
                })
            assert len(queries) - before == expected, "objective budget not exhausted"
            assert best - optimum >= -1e-7
            row = {
                "window": window_index, "environment": environment,
                "bootstrap": window_index == 0, "confirmed": confirmed,
                "active_fe": expected, "passive_n": len(y),
                "error": max(0.0, best - optimum),
                "passive_error": max(0.0, passive_best - optimum),
                "best_observed": best, "passive_best_observed": passive_best,
                "optimum": optimum,
                "reference_interpretation": "benchmark_theoretical_reference",
            }
            windows.append(row)
            event("window_complete", **row)
        event("job_complete", active_fe=len(queries), expected_active_fe=total_expected)
    assert len(queries) == total_expected
    terminal = [next(r for r in reversed(windows) if r["environment"] == environment)
                for environment in sorted({r["environment"] for r in windows})]
    scored = windows[1:]
    summary = {
        "status": "COMPLETE", "phase": job["run_id"], "algorithm": algorithm,
        "arm": algorithm, "seed": seed, "algorithm_seed": algorithm_seed,
        "function": function, "drift": drift,
        "manifest_digest": job["manifest_digest"],
        "Eoffline": float(np.mean([r["error"] for r in terminal])),
        "EDT": float(np.mean([r["error"] for r in scored])),
        "P95": float(np.percentile([r["error"] for r in scored], 95)),
        "passive_Eoffline": float(np.mean([r["passive_error"] for r in terminal])),
        "passive_EDT": float(np.mean([r["passive_error"] for r in scored])),
        "terminal_best_objective": float(np.mean([r["best_observed"] for r in terminal])),
        "active_fe": len(queries), "passive_n": sum(len(b["y"]) for b in stream),
        "alarms": sum(r["active_fe"] > 0 for r in windows),
        "timing": timing, "full_task": not bool(job["window_limit"]),
        "true_environment_supplied_to_policy": False, "native_detector_used": False,
        "duplicate_vectors_charged": True,
        "objective": "label_consistent_function_specific_transform_and_clipping",
        "evaluation": "best current-environment observation; evaluator-only reference/reset",
        "completed_time_utc": now(),
    }
    if hasattr(policy, "timing_summary"):
        summary["components"] = policy.timing_summary()
    if hasattr(policy, "mechanism_audit"):
        summary["mechanism_audit"] = policy.mechanism_audit()
    for filename, records in (
            ("windows.json", windows), ("queries.json", queries),
            ("recovery.json", recovery), ("generations.json", generations),
            ("mechanisms.json", mechanisms)):
        atomic_json(directory / filename, records)
    return summary

"""FiCSUM-indexed DETO transfer inside the unchanged DOLEM feedback loop.

Only passive observations enter fingerprints. Both passive and already revealed
active history can train source GPs. No hidden benchmark state enters this file.
"""
from __future__ import annotations

from dataclasses import asdict
import time
import numpy as np

from baseline_loader import dolem_class, load_prior
from deto_transfer import DETOConfig, DETOTransferExpert, SourceEntry
from ficsum_memory import FiCSUMMemory

FEMS = dolem_class()


class FrozenReference:
    """A DOLEM prior-initialized GP bound to one FiCSUM model version."""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return FEMS._pool_mean(self.model, np.asarray(X, float))


class FusionPolicy(FEMS):
    def __init__(self, lower, upper, seed):
        super().__init__(lower, upper, seed, load_prior())
        self.memory = FiCSUMMemory(
            dimension=self.dimension, seed=seed, window_size=75,
            buffer_ratio=0.25, similarity_gap=3, fingerprint_gap=50,
            repository_gap=100)
        self.transfer = DETOTransferExpert(lower, upper, seed=seed + 17011,
                                           config=DETOConfig())
        self._selected_sources = []
        self._transfer_name = None
        self._visible_count = 0
        self._active_history = {}
        self._reference_versions = 0
        self._episode_audits = []
        self._memory_events = []
        self._fusion_seconds = {"memory": 0.0, "transfer_fit": 0.0,
                                "transfer_proposal": 0.0}
        self._round_transfer = []
        self._last_candidates = None

    def _register_reference(self):
        if len(self.passive_y) < 50:
            return
        # The later 25 rows are a genuine holdout for this frozen GP. They are
        # never used to fit it, including its input/output standardizers.
        from meta_adaptation import build_with_cap
        X, y = self.passive_x[-50:], self.passive_y[-50:]
        pool, _ = build_with_cap(
            np.c_[X[:25], y[:25]], self.bundle["global_fallback_prior"],
            self.seed, 128)
        name = next(name for name in sorted(pool) if name.startswith("GPR"))
        model = FrozenReference(pool[name])
        self._reference_versions += 1
        state_id = self.memory.register_current(
            model, f"frozen-reference-{self._reference_versions}",
            calibration_X=X[25:].copy(), calibration_y=y[25:].copy(),
            payload={"reference_train_n": 25, "calibration_n": 25,
                     "registered_after_passive": self._visible_count,
                     "reference_training_end": self._visible_count - 25,
                     "reference_is_frozen": True})
        self._memory_events.append({
            "kind": "reference_registered", "memory_id": state_id,
            "visible_count": self._visible_count,
            "model_version": self._reference_versions,
            "calibration_is_holdout": True})

    def observe_passive(self, x, y):
        # Preserve DOLEM's prediction-before-append passive loss exactly.
        super().observe_passive(x, y)
        started = time.perf_counter()
        self._visible_count += len(y)
        events = self.memory.observe_passive(
            np.asarray(x, float).copy(), np.asarray(y, float).copy())
        self._memory_events.extend(events or [])
        for event in events or []:
            if event.get("kind") == "secondary_selection":
                previous, current = event.get("previous_id"), event.get("current_id")
                if previous != current and previous not in self.memory.entries and current is not None:
                    self._active_history.setdefault(current, []).extend(
                        self._active_history.pop(previous, []))
        if self.memory.current_id is None:
            self._register_reference()
        self._fusion_seconds["memory"] += time.perf_counter() - started

    def _collect_sources(self, matches):
        # Exclude the whole current DOLEM target window from historical tasks.
        target_start = self._visible_count - min(50, len(self.passive_y))
        selected, retrieval = [], []
        for match in matches:
            entry = self.memory.entries[match.memory_id]
            X = np.asarray(entry.stable_X, float).reshape(-1, self.dimension)
            y = np.asarray(entry.stable_y, float).reshape(-1)
            indices = np.asarray(entry.stream_indices, int)
            # FiCSUM stream indices are one-based, target rows start at +1.
            mask = indices <= target_start
            X, y = X[mask], y[mask]
            past_active = [
                item for item in self._active_history.get(match.memory_id, [])
                if item["visible_count"] <= target_start]
            if past_active:
                X = np.vstack([X] + [item["X"] for item in past_active])
                y = np.concatenate([y] + [item["y"] for item in past_active])
            if len(y) < 2:
                continue
            _, unique = np.unique(np.round(X, 11), axis=0, return_index=True)
            unique.sort()
            X, y = X[unique], y[unique]
            if len(y) < 2:
                continue
            selected.append(SourceEntry(
                source_id=int(match.memory_id), X=X.copy(), y=y.copy(),
                order=int(match.memory_id)))
            retrieval.append({
                "memory_id": int(match.memory_id),
                "similarity": float(match.similarity),
                "within_reuse_threshold": bool(match.accepted),
                "real_source_n": len(y), "target_window_excluded": True,
                "normal_mean": float(match.normal_mean),
                "normal_std": float(match.normal_std)})
            if len(selected) == 3:
                break
        return sorted(selected, key=lambda entry: entry.order), retrieval

    def _build_local(self, z, y):
        super()._build_local(z, y)
        if not self._selected_sources:
            return
        started = time.perf_counter()
        self.transfer.fit(self._selected_sources, self._denormalize(z), y)
        self._fusion_seconds["transfer_fit"] += time.perf_counter() - started
        self._round_transfer.append({
            "round": len(self._round_transfer), "real_target_n": len(y),
            "source_ids": list(self.transfer.selected_ids),
            "source_pseudo_n": sum(len(d["y"]) for d in self.transfer.pseudo_datasets),
            "hmogp": self.transfer.model.fit_audit,
            "uses_current_round_labels": False})

    def _predict(self, z):
        predictions = super()._predict(z)
        if self._transfer_name is not None and self.transfer.model is not None:
            mean, _ = self.transfer.predict(self._denormalize(z))
            if not np.isfinite(mean).all():
                raise FloatingPointError("Nonfinite transfer posterior")
            predictions[self._transfer_name] = mean
        return predictions

    def _candidates(self, train_z, train_y, radius):
        original = super()._candidates(train_z, train_y, radius)
        if not self._selected_sources:
            return original
        started = time.perf_counter()
        nominated = self.transfer.nominate(
            count=4, initial=self._denormalize(self.population)
            if len(self.population) else None)
        additions = np.vstack([self.transfer.warm_candidates, nominated])
        candidates = np.vstack([original, self._normalize(additions)])
        _, indices = np.unique(np.round(candidates, 11), axis=0, return_index=True)
        self._fusion_seconds["transfer_proposal"] += time.perf_counter() - started
        self._last_candidates = {
            "original_n": len(original), "warm_n": len(self.transfer.warm_candidates),
            "hybrid_n": len(nominated), "real_fe": 0}
        return candidates[np.sort(indices)]

    def search(self, objective_callable, generations=8, batch_size=4):
        if (generations, batch_size) != (8, 4):
            raise ValueError("Registered fusion protocol requires exactly 8 x 4 FE")
        started = time.perf_counter()
        matches = self.memory.on_alarm()
        # Normal acceptance controls FiCSUM state reuse; transfer can still use
        # related but distinct states, with DOLEM judging usefulness by truth.
        self._selected_sources, retrieval = self._collect_sources(matches)
        if self.memory.current_id is None:
            self._register_reference()
        self._fusion_seconds["memory"] += time.perf_counter() - started
        if self._selected_sources:
            ids = ",".join(str(entry.source_id) for entry in self._selected_sources)
            self._transfer_name = f"DETO-HMOGP[{ids}]"
        else:
            self._transfer_name = None
        self._round_transfer = []
        self.transfer.search_audit = []
        self._last_candidates = None
        result = super().search(objective_callable, generations, batch_size)
        state_id = self.memory.current_id
        if state_id is not None:
            self._active_history.setdefault(state_id, []).append({
                "visible_count": self._visible_count,
                "X": np.asarray([q["x"] for q in result["queries"]]),
                "y": np.asarray([q["y"] for q in result["queries"]])})
        audit = {
            "search_index": self.search_count - 1,
            "visible_passive_n": self._visible_count,
            "current_memory_id": state_id, "retrieved": retrieval,
            "transfer_role": self._transfer_name,
            "source_order": "observable_state_creation",
            "source_threshold_policy": "rank_related_states; normal_threshold_for_concept_reuse",
            "all_original_dolem_roles_retained": True,
            "all_roles_scored_on_same_prequery_predictions": True,
            "real_fe": result["real_queries"], "fingerprints_use_active_points": False,
            "round_transfer": self._round_transfer,
            "last_candidate_counts": self._last_candidates,
            "deto": self.transfer.audit() if self._selected_sources else None,
        }
        result["mechanism_audit"] = audit
        self._episode_audits.append({
            "search_index": audit["search_index"], "current_memory_id": state_id,
            "retrieved_ids": [r["memory_id"] for r in retrieval],
            "transfer_used": bool(self._selected_sources)})
        return result

    def mechanism_audit(self):
        return {
            "method": "FiCSUM-DETO-DOLEM",
            "deto_configuration": asdict(self.transfer.config),
            "fingerprint_features_per_source": 13,
            "behavior_sources": self.dimension + 4,
            "passive_samples_seen": self._visible_count,
            "memory_states": len(self.memory.entries),
            "episodes": self._episode_audits,
            "fusion_component_seconds": dict(self._fusion_seconds),
            "ficsum": self.memory.diagnostics(),
            "memory_events": self._memory_events,
            "regression_error_event": "initial_holdout_90_percentile_absolute_error",
            "reference_model": "frozen_prior_initialized_GP; 25_train_25_holdout",
            "real_objective_calls_inside_memory_or_transfer": 0}


def make_fusion(lower, upper, seed):
    return FusionPolicy(lower, upper, seed)

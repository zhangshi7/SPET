"""Two registered matching/management contrasts; original numerical search is inherited.

DIRECT_ORIGINAL replaces the fingerprint matching backend by a calibrated scalar
prediction-compatibility score. DIRECT_RECENT additionally uses alarm segments
and recency. Both retain the original passive release/drop clock. A scalar
VectorMoments instance is an API adapter, not a fingerprint representation.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from hashlib import sha256
import math
from pathlib import Path
import pickle
import random
import sys

import numpy as np

sys.dont_write_bytecode = True
CORE = Path(__file__).resolve().parent
import fusion_policy as original
import ficsum_memory as memory_original

for module, name in ((original, "fusion_policy.py"), (memory_original, "ficsum_memory.py")):
    if Path(module.__file__).resolve() != (CORE / name).resolve():
        raise ImportError("Only the frozen original implementation may be imported")

ALGORITHMS = ("DIRECT_ORIGINAL", "DIRECT_RECENT")
SCORE_DEFINITION = {
    "formula": "tau / (tau + mean(abs(y - frozen_reference.predict(X))))",
    "tau": "original 0.9 quantile of absolute residuals on the untouched 25-row passive holdout",
    "zero_over_zero": 1.0,
    "zero_tau_positive_mae": 0.0,
    "normal_record_capacity": 50,
    "normal_count_minimum": 2,
    "normal_std_floor": 0.01,
    "acceptance": "abs(score - normal_mean) <= 2 * normal_std; original version-clean guard",
    "fingerprint_features": False,
    "online_feature_normalization": False,
    "feature_weighting": False,
    "normal_record_rescaling_ratio": 1.0,
    "compatibility_storage_dimension": 1,
}


def array_digest(value):
    return sha256(np.asarray(value, dtype=float).tobytes()).hexdigest()


def data_digest(X, y):
    return sha256(np.asarray(X, dtype=float).tobytes() +
                  np.asarray(y, dtype=float).reshape(-1).tobytes()).hexdigest()


def direct_score(tau, mae):
    tau, mae = float(tau), float(mae)
    if not np.isfinite([tau, mae]).all() or min(tau, mae) < 0:
        raise ValueError("Direct-score inputs must be finite and nonnegative")
    if tau == 0:
        return 1.0 if mae == 0 else 0.0
    return tau / (tau + mae)


class DirectScoreExtractor:
    """Predict only: no meta-features, Shapley values, fitting, or objective call."""
    feature_names = ("direct_prediction_compatibility",)
    model_dependent = np.ones(1, dtype=bool)
    shap_permutations = 0
    shap_prediction_rows = 0
    degenerate_series = 0

    def __init__(self, dimension):
        self.dimension = int(dimension)
        self.calls = 0

    def unsupervised_base(self, X, y):
        # The original cache interface requests this slot; there is no summary.
        return None

    def extract(self, model, X, y, error_threshold, background, base=None):
        X, y = np.asarray(X, float), np.asarray(y, float).reshape(-1)
        if X.shape != (len(y), self.dimension) or not len(y):
            raise ValueError("Direct score needs a nonempty passive window")
        prediction = memory_original._predict(model, X)
        self.calls += 1
        return np.array([direct_score(error_threshold, np.abs(y - prediction).mean())])


class IdentityScoreNormalizer:
    """Inert adapter for the inherited state machine: it never rescales scores."""
    version = 0
    span = np.ones(1)

    def update(self, vector):
        value = np.asarray(vector, float)
        if value.shape != (1,) or not np.isfinite(value).all():
            raise ValueError("Expected one finite direct score")

    def transform(self, vector):
        return np.asarray(vector, float)


def _indices(window):
    return [int(row[0]) for row in window]


def _normal_scores(entry):
    return [float(pair[2]) for pair in entry.normal_similarity.pairs]


def _owner_rows(memory):
    return {str(i): entry.stream_indices.tolist() for i, entry in sorted(memory.entries.items())}


def _admitted_union(memory):
    return sorted({int(i) for entry in memory.entries.values() for i in entry.stream_indices})


def _state(memory):
    return {
        "seen": memory.seen, "segment_seen": memory.segment_seen,
        "current_id": memory.current_id, "next_id": memory.next_id,
        "delay_indices": _indices(memory.delay_buffer),
        "stable_indices": _indices(memory.stable_window),
        "active_indices": _indices(memory.active_window),
        "secondary_match_at": memory.secondary_match_at,
        "secondary_candidate_ids": sorted(memory.secondary_candidate_ids),
        "last_fingerprint": memory.last_fingerprint,
        "last_repository": memory.last_repository,
        "last_similarity": memory.last_similarity,
        "admitted_union": _admitted_union(memory),
        "entries": [{
            "memory_id": i, "reference_version": str(e.model_version),
            "created_at": e.created_at, "last_active_at": e.last_active_at,
            "version_clean_at": e.version_clean_at, "provisional": e.provisional,
            "ready": e.ready, "tau": e.error_threshold,
            "admitted_indices": e.stream_indices.tolist(),
            "normal_scores": _normal_scores(e),
            "score_moments": {"mean": e.fingerprint.mean.tolist(),
                              "std": e.fingerprint.std.tolist(),
                              "count": e.fingerprint.count.tolist()},
            "cross_current_ids": sorted(e.cross_fingerprints),
        } for i, e in sorted(memory.entries.items())],
    }


class DirectScoreMemory(memory_original.FiCSUMMemory):
    """Original state machine with a one-dimensional matching-backend adapter.

The initializer reproduces the frozen constructor's state fields so that it
does not construct FullMetaFeatures. The original register/observe methods are
inherited directly; instrumented methods delegate once to the original method.
"""
    management_name = "original_accept_reuse_and_secondary_merge"

    def __init__(self, dimension, seed=0):
        from river.drift import ADWIN
        self.dimension, self.seed = int(dimension), int(seed)
        self.window_size, self.buffer_ratio = 75, 0.25
        self.similarity_gap, self.fingerprint_gap, self.repository_gap = 3, 50, 100
        self.std_floor = 0.01
        self.extractor = DirectScoreExtractor(self.dimension)
        self.feature_names = self.extractor.feature_names
        self.normalizer = IdentityScoreNormalizer()
        self.entries = {}
        self.repository = self
        self.current_id = None
        self.next_id = self.seen = self.segment_seen = 0
        self.active_window = deque(maxlen=self.window_size)
        self.delay_buffer = deque()
        self.stable_window = deque(maxlen=self.window_size)
        self.detector = ADWIN(delta=0.002)
        self.last_fingerprint = self.last_repository = self.last_similarity = 0
        self.secondary_match_at = None
        self.secondary_candidate_ids = set()
        self.events = []
        self._cache = {}
        self._last_matches = []
        self.audit_operations = []
        self.retrieval_audits = []

    def _operation(self, kind, **values):
        record = {"operation_index": len(self.audit_operations), "kind": kind,
                  "seen": self.seen, **values}
        self.audit_operations.append(record)
        return record

    def weights(self):
        return np.ones(1)

    def _similarity(self, reference, observed, weights=None):
        value = np.asarray(observed, float)
        if value.shape != (1,) or not np.isfinite(value).all():
            raise ValueError("Malformed direct-score storage")
        return float(value[0])

    def _update_stable(self):
        current = self.current_id
        valid = current is not None and len(self.stable_window) >= 5
        before = None if not valid else self.entries[current]
        normal_before = [] if before is None else _normal_scores(before)
        ready_before = False if before is None else before.ready
        clean_at = None if before is None else before.version_clean_at
        window = _indices(self.stable_window)
        result = memory_original.FiCSUMMemory._update_stable(self)
        entry = None if not valid else self.entries[current]
        score = None if entry is None else float(self._fingerprint(entry, self.stable_window)[0])
        self._operation(
            "normal_update", current_id=current, skipped=not valid,
            reference_version=None if entry is None else str(entry.model_version),
            window_indices=window, ready_before=ready_before, version_clean_at=clean_at,
            normal_record_added=bool(valid and ready_before and self.seen >= clean_at),
            score=score, normal_scores_before=normal_before,
            normal_scores_after=[] if entry is None else _normal_scores(entry),
            last_fingerprint_after=self.last_fingerprint)
        return result

    def _update_repository(self):
        valid = self.current_id is not None and len(self.stable_window) >= 5
        compared = sorted(i for i in self.entries if i != self.current_id) if valid else []
        result = memory_original.FiCSUMMemory._update_repository(self)
        self._operation("repository_update", current_id=self.current_id, skipped=not valid,
                        compared_ids=compared, window_indices=_indices(self.stable_window),
                        last_repository_after=self.last_repository)
        return result

    def _audit_candidate(self, match):
        entry = self.entries[match.memory_id]
        X, y = self._arrays(self.active_window)
        # FrozenReference.predict is deterministic; fail if diagnostic prediction
        # changes either process-global RNG. No state is restored to hide a change.
        rng_before = pickle.dumps((random.getstate(), np.random.get_state()), protocol=5)
        prediction = memory_original._predict(entry.model, X)
        if pickle.dumps((random.getstate(), np.random.get_state()), protocol=5) != rng_before:
            raise AssertionError("Diagnostic reference prediction changed RNG state")
        mae = float(np.abs(y - prediction).mean())
        mu, sd, count, ratio = entry.normal_similarity.adjusted(self._similarity, self.std_floor)
        score = direct_score(entry.error_threshold, mae)
        if score != match.similarity or ratio != 1.0:
            raise AssertionError("Direct-score audit differs from inherited retrieval")
        return {
            "memory_id": match.memory_id, "reference_version": str(entry.model_version),
            "tau": entry.error_threshold, "predictions": prediction.tolist(),
            "mae": mae, "score": score, "normal_scores": _normal_scores(entry),
            "normal_mean": mu, "normal_std": sd, "normal_count": count,
            "normal_ratio": ratio, "version_clean_at": entry.version_clean_at,
            "accepted": bool(match.accepted),
        }

    def retrieve(self, top_k=3, accepted_only=True):
        returned = memory_original.FiCSUMMemory.retrieve(
            self, top_k=top_k, accepted_only=accepted_only)
        matches = self._last_matches if len(self.active_window) >= 5 else []
        X, y = self._arrays(self.active_window)
        candidates = [self._audit_candidate(m) for m in matches]
        for rank, item in enumerate(candidates):
            item["rank"] = rank
        record = {
            "retrieval_index": len(self.retrieval_audits), "kind": "direct_score",
            "seen": self.seen, "current_id": self.current_id,
            "window_indices": _indices(self.active_window),
            "window_X_sha256": array_digest(X), "window_y_sha256": array_digest(y),
            "top_k": top_k, "accepted_only": bool(accepted_only),
            "candidates": candidates, "returned_ids": [m.memory_id for m in returned],
        }
        self.retrieval_audits.append(record)
        self._operation("retrieval", retrieval_index=record["retrieval_index"])
        return returned

    def on_alarm(self):
        before = _state(self)
        retrieval_start = len(self.retrieval_audits)
        result = memory_original.FiCSUMMemory.on_alarm(self)
        self._operation("external_alarm", management=self.management_name,
                        before=before, after=_state(self),
                        dropped_indices=before["delay_indices"],
                        retrieval_index=retrieval_start,
                        returned_ids=[m.memory_id for m in result])
        return result

    def _secondary_match(self):
        before = _state(self)
        retrieval_start = len(self.retrieval_audits)
        result = memory_original.FiCSUMMemory._secondary_match(self)
        self._operation("secondary_match", before=before, after=_state(self),
                        retrieval_index=retrieval_start)
        return result

    def diagnostics(self):
        result = memory_original.FiCSUMMemory.diagnostics(self)
        result.update({
            "memory_type": "direct_prediction_score_with_" + self.management_name,
            "fingerprint_representation_present": False, "feature_count": 0,
            "feature_names": [], "compatibility_storage_dimension": 1,
            "meta_feature_calls": 0, "direct_score_calls": self.extractor.calls,
            "normalizer_present": False, "feature_weights_present": False,
            "legacy_fingerprint_count_fields": "scalar score moment counts, not fingerprints",
            "score_definition": dict(SCORE_DEFINITION),
        })
        return result


class DirectRecentMemory(DirectScoreMemory):
    """Same release clock; only admitted rows may move into a new alarm segment."""
    management_name = "alarm_segments_and_recent_three_eligible_sources"

    def __init__(self, dimension, seed=0):
        super().__init__(dimension, seed)
        self.segment_bounds = {}
        self._pending_rows = {}
        self._pending_target_start = None
        self._last_alarm_seen = None

    def register_current(self, model, model_version, calibration_X, calibration_y, payload=None):
        if self.current_id is not None:
            raise AssertionError("Recent segments never re-register or evolve an existing reference")
        state_id = memory_original.FiCSUMMemory.register_current(
            self, model, model_version, calibration_X, calibration_y, payload)
        entry = self.entries[state_id]
        for index, (x, y) in self._pending_rows.items():
            entry.stable_observations[index] = (x.copy(), float(y))
        start = 1 if self._pending_target_start is None else self._pending_target_start + 1
        self.segment_bounds[state_id] = {"first_index": start, "closed_at": None,
                                         "registered_at": self.seen}
        moved = sorted(self._pending_rows)
        self._pending_rows.clear()
        self._pending_target_start = None
        self._operation("recent_segment_registered", current_id=state_id,
                        moved_admitted_indices=moved, segment_bound=dict(self.segment_bounds[state_id]),
                        after=_state(self))
        return state_id

    def on_alarm(self):
        if self.current_id is None or self.seen < 50 or self._pending_rows:
            raise AssertionError("Recent alarm requires a registered current segment")
        if self._last_alarm_seen == self.seen:
            raise AssertionError("At most one alarm is permitted per passive window")
        self._last_alarm_seen = self.seen
        before = _state(self)
        previous = self.current_id
        target_start = self.seen - 50
        entry = self.entries[previous]
        moved = [int(i) for i in entry.stream_indices if i > target_start]
        self._pending_rows = {i: entry.stable_observations.pop(i) for i in moved}
        self._pending_target_start = target_start
        if any(i > target_start for other in self.entries.values() for i in other.stream_indices):
            raise AssertionError("Current target rows unexpectedly belong to another segment")
        self.segment_bounds[previous]["closed_at"] = target_start
        self.current_id = None
        self.secondary_match_at = None
        self.secondary_candidate_ids.clear()
        self.segment_seen = 0
        self.delay_buffer.clear()
        self.stable_window.clear()
        self.detector._reset()
        ids = sorted((i for i, e in self.entries.items() if e.ready), reverse=True)
        matches = [memory_original.Match(
            i, self.entries[i].model, self.entries[i].payload,
            0.0, 0.0, 0.0, False, 0, 0, self.entries[i].model_version) for i in ids]
        self._last_matches = matches
        retrieval_index = len(self.retrieval_audits)
        self.retrieval_audits.append({
            "retrieval_index": retrieval_index, "kind": "recency",
            "seen": self.seen, "current_id": None,
            "window_indices": _indices(self.active_window),
            "top_k": None, "accepted_only": False,
            "score_fields": "not_applicable; original collector uses only memory_id",
            "candidates": [{"rank": rank, "memory_id": i,
                            "reference_version": str(self.entries[i].model_version)}
                           for rank, i in enumerate(ids)],
            "returned_ids": ids,
        })
        self._operation("retrieval", retrieval_index=retrieval_index)
        self._event("external_alarm_segment_boundary", previous_id=previous,
                    current_id=None, target_start=target_start,
                    moved_admitted_indices=moved, candidate_ids=ids,
                    matching_scores_used=False, secondary_match_at=None)
        self._operation("external_alarm", management=self.management_name,
                        before=before, after=_state(self),
                        dropped_indices=before["delay_indices"], target_start=target_start,
                        pending_admitted_indices=moved, retrieval_index=retrieval_index,
                        returned_ids=ids)
        return matches

    def _secondary_match(self):
        raise AssertionError("Recent-segment management has no secondary selection or merge")

    def diagnostics(self):
        result = super().diagnostics()
        result.update({"segment_bounds": {str(i): dict(b) for i, b in self.segment_bounds.items()},
                       "matching_scores_used_for_management": False,
                       "matching_scores_used_for_retrieval": False,
                       "pending_admitted_indices": sorted(self._pending_rows)})
        return result


class IsolationPolicy(original.FusionPolicy):
    def __init__(self, algorithm, lower, upper, seed):
        if algorithm not in ALGORITHMS:
            raise ValueError("Unknown registered isolation algorithm")
        self.algorithm = algorithm
        original.FEMS.__init__(self, lower, upper, seed, original.load_prior())
        cls = DirectScoreMemory if algorithm == "DIRECT_ORIGINAL" else DirectRecentMemory
        self.memory = cls(self.dimension, seed)
        self.transfer = original.DETOTransferExpert(
            lower, upper, seed=seed + 17011, config=original.DETOConfig())
        self._selected_sources = []
        self._transfer_name = None
        self._visible_count = 0
        self._active_history = {}
        self._reference_versions = 0
        self._episode_audits = []
        self._memory_events = []
        self._fusion_seconds = {"memory": 0.0, "transfer_fit": 0.0, "transfer_proposal": 0.0}
        self._round_transfer = []
        self._last_candidates = None
        self._reference_archive = {}
        self._reference_metadata = []
        self._passive_batches = []
        self._snapshots = []
        self._search_records = []

    def _register_reference(self):
        before = _state(self.memory)
        previous_version = self._reference_versions
        result = original.FusionPolicy._register_reference(self)
        if self._reference_versions == previous_version:
            return result
        entry = self.memory.entries[self.memory.current_id]
        version = str(entry.model_version)
        if version in self._reference_archive:
            raise AssertionError("A frozen reference version cannot be replaced")
        self._reference_archive[version] = entry.model
        X, y = self.passive_x[-50:], self.passive_y[-50:]
        first = self._visible_count - 49
        record = {
            "reference_version": version, "memory_id": entry.memory_id,
            "registered_at": self._visible_count,
            "training_indices": list(range(first, first + 25)),
            "holdout_indices": list(range(first + 25, first + 50)),
            "training_X_sha256": array_digest(X[:25]),
            "training_y_sha256": array_digest(y[:25]),
            "holdout_X_sha256": array_digest(X[25:]),
            "holdout_y_sha256": array_digest(y[25:]),
            "tau": entry.error_threshold,
        }
        self._reference_metadata.append(record)
        self.memory._operation("reference_registered", reference=dict(record),
                               before=before, after=_state(self.memory))
        return result

    def observe_passive(self, x, y):
        X, values = np.asarray(x, float), np.asarray(y, float).reshape(-1)
        if X.shape != (50, self.dimension) or len(values) != 50:
            raise ValueError("The registered passive protocol uses 50-row batches")
        before = _state(self.memory)
        # A diagnostic-only replay of the unchanged clock establishes what the
        # inherited observer must release; it never supplies data to the policy.
        delay = deque(before["delay_indices"])
        stable = deque(before["stable_indices"], maxlen=75)
        released = []
        segment_seen = before["segment_seen"]
        for index in range(before["seen"] + 1, before["seen"] + 51):
            segment_seen += 1
            delay.append(index)
            if len(delay) >= max(math.floor(segment_seen * 0.25), 1):
                row_index = delay.popleft()
                released.append(row_index)
                stable.append(row_index)
        result = original.FusionPolicy.observe_passive(self, x, y)
        after = _state(self.memory)
        if (after["delay_indices"] != list(delay) or after["stable_indices"] != list(stable)
                or after["segment_seen"] != segment_seen):
            raise AssertionError("Inherited release clock differs from the frozen rule")
        expected_admitted = sorted(set(before["admitted_union"]) | set(released))
        if after["admitted_union"] != expected_admitted:
            raise AssertionError("A released passive row was lost or an un-released row entered history")
        self._passive_batches.append({
            "batch_index": len(self._passive_batches),
            "seen_before": before["seen"], "seen_after": after["seen"],
            "segment_seen_before": before["segment_seen"], "segment_seen_after": segment_seen,
            "delay_before": before["delay_indices"], "delay_after": after["delay_indices"],
            "stable_before": before["stable_indices"], "stable_after": after["stable_indices"],
            "released_indices": released, "admitted_union_after": after["admitted_union"],
            "owners_after": _owner_rows(self.memory),
        })
        self._snapshots.append({"snapshot_index": len(self._snapshots),
                                "kind": "after_passive_batch_and_registration", **after})
        return result

    def search(self, objective_callable, generations=8, batch_size=4):
        result = original.FusionPolicy.search(self, objective_callable, generations, batch_size)
        sources = [{"memory_id": source.source_id, "order": source.order,
                    "n": len(source.y), "X_sha256": array_digest(source.X),
                    "y_sha256": array_digest(source.y),
                    "data_sha256": data_digest(source.X, source.y)}
                   for source in self._selected_sources]
        Xq = np.asarray([q["x"] for q in result["queries"]], float)
        yq = np.asarray([q["y"] for q in result["queries"]], float)
        record = {
            "search_index": self.search_count - 1, "seen": self._visible_count,
            "current_id": self.memory.current_id,
            "query_n": len(yq), "query_data_sha256": data_digest(Xq, yq),
            "selected_sources": sources,
            "retrieved_ids": [r["memory_id"] for r in result["mechanism_audit"]["retrieved"]],
            "target_indices": list(range(self._visible_count - 49, self._visible_count + 1)),
            "target_data_sha256": data_digest(self.passive_x[-50:], self.passive_y[-50:]),
            "after": _state(self.memory),
            "active_history_blocks": {
                str(i): [{"visible_count": item["visible_count"], "n": len(item["y"]),
                          "data_sha256": data_digest(item["X"], item["y"])} for item in blocks]
                for i, blocks in sorted(self._active_history.items())},
        }
        self._search_records.append(record)
        result["mechanism_audit"]["isolation"] = {
            "algorithm": self.algorithm, "score_definition": dict(SCORE_DEFINITION),
            "management": self.memory.management_name, "fingerprints_present": False,
            "original_release_clock_preserved": True,
            "selected_sources": deepcopy(sources),
            "search_record_index": len(self._search_records) - 1,
            "legacy_retrieval_score_fields": ("actual_direct_scores" if self.algorithm == "DIRECT_ORIGINAL"
                                               else "not_applicable_recency_placeholders"),
        }
        return result

    def export_reference_models(self):
        # Keep even references whose provisional entries were later merged away.
        return dict(self._reference_archive)

    def export_memory_audit(self):
        return deepcopy({
            "schema_version": 1, "algorithm": self.algorithm,
            "score_definition": dict(SCORE_DEFINITION),
            "management": self.memory.management_name,
            "references": self._reference_metadata,
            "passive_batches": self._passive_batches,
            "snapshots": self._snapshots,
            "operations": self.memory.audit_operations,
            "retrievals": self.memory.retrieval_audits,
            "search_records": self._search_records,
            "original_memory_events": self.memory.events,
            "final_state": _state(self.memory),
            "final_diagnostics": self.memory.diagnostics(),
        })

    def mechanism_audit(self):
        result = original.FusionPolicy.mechanism_audit(self)
        result.update({"method": self.algorithm, "fingerprints_present": False,
                       "fingerprint_features_per_source": 0, "behavior_sources": 0,
                       "score_definition": dict(SCORE_DEFINITION),
                       "management": self.memory.management_name})
        return result


def make_isolation(algorithm, lower, upper, seed):
    return IsolationPolicy(algorithm, lower, upper, seed)

"""Full FiCSUM fingerprint repository, adapted explicitly to frozen regressors.

The implementation follows Sections III--IV / Table I of Halstead et al.,
Fingerprinting Concepts in Data Streams with Supervised and Unsupervised
Meta-Information.  See protocol/FICSUM_IMPLEMENTATION.md for the source mapping,
necessary regression changes, numerical conventions, and external-alarm policy.
There is deliberately no objective/evaluation callback in this module.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
import math
import warnings

import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression


UPSTREAM_COMMIT = "eebe4993ca4981df63a0119fb837705839f88ad3"
STAT_NAMES = (
    "mean", "stdev", "skew", "kurtosis", "acf_1", "acf_2",
    "pacf_1", "pacf_2", "MI", "turning_point_rate", "IMF_0", "IMF_1", "FI",
)


class Predictor(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...


def _predict(model: Predictor, X: np.ndarray) -> np.ndarray:
    result = model.predict(np.asarray(X, dtype=float))
    if isinstance(result, tuple):
        result = result[0]
    result = np.asarray(result, dtype=float).reshape(-1)
    if result.shape != (len(X),) or not np.isfinite(result).all():
        raise ValueError("Reference regressor must return one finite prediction per row.")
    return result


@dataclass
class VectorMoments:
    """Welford mean, sample standard deviation and per-coordinate count."""
    size: int
    mean: np.ndarray = field(init=False)
    m2: np.ndarray = field(init=False)
    count: np.ndarray = field(init=False)

    def __post_init__(self):
        self.mean = np.zeros(self.size)
        self.m2 = np.zeros(self.size)
        self.count = np.zeros(self.size, dtype=np.int64)

    def update(self, values: np.ndarray):
        values = np.asarray(values, dtype=float)
        if values.shape != self.mean.shape or not np.isfinite(values).all():
            raise ValueError("Fingerprint must have the declared shape and finite values.")
        self.count += 1
        delta = values - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (values - self.mean)

    @property
    def std(self):
        return np.sqrt(np.maximum(self.m2, 0) / np.maximum(self.count - 1, 1))

    def reset(self, mask: np.ndarray):
        self.mean[mask] = 0
        self.m2[mask] = 0
        self.count[mask] = 0

    @property
    def ready(self):
        return bool(np.all(self.count > 0))


@dataclass
class OnlineRange:
    size: int
    minimum: np.ndarray = field(init=False)
    maximum: np.ndarray = field(init=False)
    version: int = 0

    def __post_init__(self):
        self.minimum = np.full(self.size, np.inf)
        self.maximum = np.full(self.size, -np.inf)

    def update(self, vector: np.ndarray):
        low = np.minimum(self.minimum, vector)
        high = np.maximum(self.maximum, vector)
        if np.any(low != self.minimum) or np.any(high != self.maximum):
            self.version += 1
        self.minimum, self.maximum = low, high

    @property
    def span(self):
        span = self.maximum - self.minimum
        return np.where(np.isfinite(span) & (span > 0), span, 1.0)

    def transform(self, vector: np.ndarray):
        origin = np.where(np.isfinite(self.minimum), self.minimum, 0.0)
        return (np.asarray(vector) - origin) / self.span


def weighted_cosine(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    """Paper's cosine of W*a and W*b (therefore squared weights in the dot)."""
    a, b = np.asarray(a) * weights, np.asarray(b) * weights
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


class FullMetaFeatures:
    """All 13 Table-I quantities on all d+4 behaviour sources.

    Feature importance uses actual antithetic permutation Shapley marginals,
    with Monte Carlo integration over the model's fixed passive background.
    It is a model-agnostic adaptation of the author's approximate TreeSHAP;
    no input correlation or gradient is substituted for Shapley contributions.
    """
    def __init__(self, dimension: int, seed: int = 0, shap_permutations: int = 16):
        try:
            from PyEMD import EMD
            from antropy import perm_entropy
            from statsmodels.tsa.stattools import acf, pacf
        except ImportError as exc:
            raise ImportError(
                "Full FiCSUM requires EMD-signal, antropy and statsmodels. "
                "Missing meta-features may not be silently disabled."
            ) from exc
        self.EMD, self.perm_entropy, self.acf, self.pacf = EMD, perm_entropy, acf, pacf
        self.dimension, self.seed = int(dimension), int(seed)
        self.shap_permutations = int(shap_permutations)
        if self.shap_permutations < 1:
            raise ValueError("At least one forward/reverse Shapley permutation is required.")
        sources = [f"x{j}" for j in range(dimension)]
        sources += ["labels", "predictions", "errors", "error_distances"]
        self.feature_names = tuple(f"{source}:{name}" for source in sources for name in STAT_NAMES)
        self.model_dependent = np.array([
            source in ("predictions", "errors", "error_distances") or name == "FI"
            for source in sources for name in STAT_NAMES
        ])
        self.calls = 0
        self.shap_prediction_rows = 0
        self.degenerate_series = 0

    def series(self, values: np.ndarray, importance: float = 0.0) -> np.ndarray:
        x = np.asarray(values, dtype=float).reshape(-1)
        if not len(x):
            x = np.zeros(1)
        if not np.isfinite(x).all():
            raise ValueError("Non-finite passive observations cannot form a fingerprint.")
        out = np.zeros(len(STAT_NAMES))
        out[0] = x.mean()
        out[1] = x.std(ddof=1) if len(x) > 1 else 0.0
        out[-1] = float(importance)
        if len(x) < 3 or np.ptp(x) <= np.finfo(float).eps * max(1.0, np.abs(x).max()):
            self.degenerate_series += 1
            return out
        out[2], out[3] = stats.skew(x), stats.kurtosis(x)
        ac = self.acf(x, nlags=min(2, len(x) - 1), fft=True)
        out[4:4 + len(ac) - 1] = ac[1:]
        # PACF at lag k is not estimable unless k < n/2.
        available_lags = min(2, len(x) // 2 - 1)
        if available_lags > 0:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    pc = self.pacf(x, nlags=available_lags, method="ywadjusted")
                out[6:6 + available_lags] = pc[1:]
            except np.linalg.LinAlgError:
                # Singular constant/linearly dependent histories have undefined PACF.
                self.degenerate_series += 1
        if len(x) > 4:
            out[8] = mutual_info_regression(
                x[:-1].reshape(-1, 1), x[1:], random_state=42, copy=True
            )[0]
        diff = np.diff(x)
        out[9] = np.count_nonzero(diff[1:] * diff[:-1] < 0) / len(x)
        emd = self.EMD(spline_kind="slinear")
        emd.emd(x, max_imf=2)
        imfs, _residue = emd.get_imfs_and_residue()
        # Table I specifies IMF 1 and 2; the residual is not a third feature.
        for i in range(min(2, len(imfs))):
            out[10 + i] = self.perm_entropy(imfs[i], order=3, normalize=False)
        # Undefined statistical values for degenerate finite sequences are zero.
        # This is not a fallback for unavailable packages or execution errors.
        if not np.isfinite(out).all():
            self.degenerate_series += 1
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def shap_values(self, model: Predictor, X: np.ndarray, background: np.ndarray):
        X, background = np.asarray(X, dtype=float), np.asarray(background, dtype=float)
        if background.ndim != 2 or len(background) == 0 or background.shape[1] != self.dimension:
            raise ValueError("A non-empty passive calibration background is mandatory.")
        # Reuse identical Monte Carlo paths for every window/source: variation
        # in explanations then comes from data/models, not changing random draws.
        rng = np.random.default_rng(self.seed)
        result = np.zeros_like(X, dtype=float)
        paths, orders = [], []
        for _ in range(self.shap_permutations):
            order = rng.permutation(self.dimension)
            base = background[rng.integers(len(background))]
            for permutation in (order, order[::-1]):
                path = np.broadcast_to(base, (len(X), self.dimension)).copy()
                states = [path.copy()]
                for j in permutation:
                    path[:, j] = X[:, j]
                    states.append(path.copy())
                paths.append(np.stack(states, axis=1))
                orders.append(permutation)
        flat = np.concatenate([p.reshape(-1, self.dimension) for p in paths], axis=0)
        self.shap_prediction_rows += len(flat)
        pred = np.concatenate([_predict(model, flat[i:i + 8192]) for i in range(0, len(flat), 8192)])
        pred = pred.reshape(len(paths), len(X), self.dimension + 1)
        for values, order in zip(pred, orders):
            result[:, order] += np.diff(values, axis=1)
        return result / len(paths)

    def unsupervised_base(self, X: np.ndarray, y: np.ndarray):
        """Cache only model-independent feature/label statistics on one window."""
        return np.stack([self.series(X[:, j]) for j in range(self.dimension)] + [self.series(y)])

    def extract(
        self, model: Predictor, X: np.ndarray, y: np.ndarray,
        error_threshold: float, background: np.ndarray,
        base: np.ndarray | None = None,
    ) -> np.ndarray:
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)
        if X.shape != (len(y), self.dimension):
            raise ValueError("Passive X/y dimensions disagree.")
        p = _predict(model, X)
        errors = (np.abs(y - p) > error_threshold).astype(float)
        error_positions = np.flatnonzero(errors)
        gaps = np.diff(error_positions)
        if len(gaps) == 0:
            gaps = np.zeros(1)
        matrix = (self.unsupervised_base(X, y) if base is None else base.copy())
        # Author implementation aggregates sum(abs(SHAP)) over the window.
        importance = np.abs(self.shap_values(model, X, background)).sum(axis=0)
        matrix[:self.dimension, -1] = importance
        vector = np.concatenate([
            matrix.ravel(), self.series(p), self.series(errors), self.series(gaps),
        ])
        self.calls += 1
        return vector


@dataclass
class SimilarityRecord:
    size: int = 50
    pairs: deque = field(default_factory=deque)

    def append(self, reference: np.ndarray, observed: np.ndarray, similarity: float):
        self.pairs.append((reference.copy(), observed.copy(), float(similarity)))
        while len(self.pairs) > self.size:
            self.pairs.popleft()

    def adjusted(self, similarity_function: Callable, std_floor: float = 0.01):
        if not self.pairs:
            return 1.0, std_floor, 0, 1.0
        old = np.array([p[2] for p in self.pairs])
        # Section IV: re-evaluate saved pairs under the CURRENT normalizer and W.
        # Unlike stale scalar records, these remain comparable after weights move.
        sample = list(self.pairs)[::5]
        ratios = [
            similarity_function(a, b) / previous
            for a, b, previous in sample if abs(previous) > 1e-12
        ]
        ratio = float(np.mean(ratios)) if ratios else 1.0
        ratio = max(ratio, 0.0)
        mu = float(old.mean() * ratio)
        sigma = float(old.std(ddof=1) * ratio) if len(old) > 1 else 0.0
        return mu, max(sigma, std_floor), len(old), ratio


@dataclass
class MemoryEntry:
    memory_id: int
    model: Predictor
    model_version: Any
    fingerprint: VectorMoments
    error_threshold: float
    calibration_X: np.ndarray
    calibration_y: np.ndarray
    created_at: int
    payload: Any = None
    normal_similarity: SimilarityRecord = field(default_factory=SimilarityRecord)
    cross_fingerprints: dict[int, VectorMoments] = field(default_factory=dict)
    version_history: list = field(default_factory=list)
    last_active_at: int = 0
    version_clean_at: int = 0
    provisional: bool = False
    stable_observations: dict[int, tuple[np.ndarray, float]] = field(default_factory=dict)

    @property
    def ready(self):
        return self.fingerprint.ready

    def record_stable(self, row):
        index, x, y = row
        self.stable_observations[int(index)] = (np.asarray(x).copy(), float(y))

    @property
    def stream_indices(self):
        return np.array(sorted(self.stable_observations), dtype=np.int64)

    @property
    def stable_X(self):
        if not self.stable_observations:
            return np.empty((0, self.calibration_X.shape[1]))
        return np.stack([self.stable_observations[i][0] for i in self.stream_indices])

    @property
    def stable_y(self):
        return np.array([self.stable_observations[i][1] for i in self.stream_indices])


@dataclass
class Match:
    memory_id: int
    model: Predictor
    payload: Any
    similarity: float
    normal_mean: float
    normal_std: float
    accepted: bool
    normal_count: int
    normalization_version: int
    model_version: Any

    def diagnostic(self):
        return {
            "memory_id": self.memory_id, "similarity": self.similarity,
            "normal_mean": self.normal_mean, "normal_std": self.normal_std,
            "accepted": self.accepted, "normal_count": self.normal_count,
            "normalization_version": self.normalization_version,
            "model_version": self.model_version,
        }


class FiCSUMMemory:
    """Passive-stream concept memory with complete fingerprints and recurrence.

    Root controls all training and all objective evaluations. Call observe_passive
    exactly once per arriving passive observation. Register frozen models with
    untouched passive calibration data. on_alarm is the COMMON DOLEM alarm;
    internal ADWIN events are diagnostic and never consume evaluation budget.
    """
    def __init__(
        self, dimension: int, seed: int = 0, window_size: int = 75,
        buffer_ratio: float = 0.25, similarity_gap: int = 3,
        fingerprint_gap: int = 50, repository_gap: int = 100,
        shap_permutations: int = 16, std_floor: float = 0.01,
    ):
        from river.drift import ADWIN
        self.dimension, self.seed = int(dimension), int(seed)
        self.window_size, self.buffer_ratio = int(window_size), float(buffer_ratio)
        self.similarity_gap = int(similarity_gap)
        self.fingerprint_gap, self.repository_gap = int(fingerprint_gap), int(repository_gap)
        self.std_floor = float(std_floor)
        self.extractor = FullMetaFeatures(dimension, seed, shap_permutations)
        self.feature_names = self.extractor.feature_names
        self.normalizer = OnlineRange(len(self.feature_names))
        self.entries: dict[int, MemoryEntry] = {}
        self.repository = self  # repository.entries convenience for integration.
        self.current_id: int | None = None
        self.next_id, self.seen, self.segment_seen = 0, 0, 0
        self.active_window = deque(maxlen=self.window_size)
        self.delay_buffer = deque()
        self.stable_window = deque(maxlen=self.window_size)
        self.detector = ADWIN(delta=0.002)
        self.last_fingerprint = self.last_repository = self.last_similarity = 0
        self.secondary_match_at: int | None = None
        self.secondary_candidate_ids: set[int] = set()
        self.events: list[dict] = []
        self._cache: dict = {}
        self._last_matches: list[Match] = []

    def _event(self, kind: str, **values):
        event = {"kind": kind, "passive_seen": self.seen, **values}
        self.events.append(event)
        return event

    def get_entry(self, memory_id: int):
        return self.entries[memory_id]

    def passive_window(self):
        return self._arrays(self.active_window)

    def _arrays(self, window):
        if not window:
            return np.empty((0, self.dimension)), np.empty(0)
        return np.stack([r[1] for r in window]), np.array([r[2] for r in window])

    def _fingerprint(self, entry: MemoryEntry, window):
        if not window:
            raise ValueError("Cannot fingerprint an empty window.")
        key = (entry.memory_id, repr(entry.model_version), tuple(r[0] for r in window))
        if key not in self._cache:
            X, y = self._arrays(window)
            base_key = ("base", key[-1])
            if base_key not in self._cache:
                self._cache[base_key] = self.extractor.unsupervised_base(X, y)
            self._cache[key] = self.extractor.extract(
                entry.model, X, y, entry.error_threshold, entry.calibration_X,
                base=self._cache[base_key],
            )
        return self._cache[key]

    def register_current(
        self, model: Predictor, model_version: Any,
        calibration_X: np.ndarray, calibration_y: np.ndarray,
        payload: Any = None,
    ) -> int:
        """Register a new current concept, or change the current model version.

        calibration rows MUST be passive and absent from this version's training
        set. No training occurs here. A repeated version changes payload only.
        """
        X = np.asarray(calibration_X, dtype=float)
        y = np.asarray(calibration_y, dtype=float).reshape(-1)
        if X.shape != (len(y), self.dimension) or len(y) < 5:
            raise ValueError("At least 5 untouched passive calibration examples are required.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Calibration data must be finite.")
        if self.current_id is not None and self.entries[self.current_id].model_version == model_version:
            if payload is not None:
                self.entries[self.current_id].payload = payload
            return self.current_id
        threshold = float(np.quantile(np.abs(y - _predict(model, X)), 0.9))
        # A perfect predictor has a zero threshold; positive error is still error.
        if self.current_id is None:
            memory_id = self.next_id
            self.next_id += 1
            entry = MemoryEntry(
                memory_id, model, model_version, VectorMoments(len(self.feature_names)),
                threshold, X.copy(), y.copy(), self.seen, payload=payload,
                provisional=self.secondary_match_at is not None,
            )
            self.entries[memory_id] = entry
            self.current_id = memory_id
            for row in self.stable_window:
                entry.record_stable(row)
            entry.version_history.append({"version": model_version, "passive_seen": self.seen})
            self._event("new_concept", memory_id=memory_id, model_version=model_version,
                        calibration_size=len(y), error_threshold=threshold)
        else:
            entry = self.entries[self.current_id]
            entry.model, entry.model_version = model, model_version
            entry.error_threshold = threshold
            entry.calibration_X, entry.calibration_y = X.copy(), y.copy()
            if payload is not None:
                entry.payload = payload
            entry.fingerprint.reset(self.extractor.model_dependent)
            entry.cross_fingerprints.clear()
            entry.normal_similarity = SimilarityRecord()
            entry.version_clean_at = self.seen + self.window_size
            entry.version_history.append({"version": model_version, "passive_seen": self.seen})
            self._event("model_evolution", memory_id=entry.memory_id, model_version=model_version,
                        model_dependent_reset=True, clean_at=entry.version_clean_at)
        self._cache.clear()
        # Calibration is already observed passive data, so it can initialize the
        # representation without asking for any additional target values.
        initial = self.extractor.extract(model, X, y, threshold, X)
        self.normalizer.update(initial)
        entry.fingerprint.update(initial)
        self.last_fingerprint = self.seen
        return entry.memory_id

    def weights(self):
        """Paper III-B: inverse within-scale * max(inter, intra-classifier)."""
        ready = [e for e in self.entries.values() if e.ready]
        if not ready:
            return np.ones(len(self.feature_names))
        means = np.stack([self.normalizer.transform(e.fingerprint.mean) for e in ready])
        sigmas = np.stack([e.fingerprint.std / self.normalizer.span for e in ready])
        counts = np.stack([e.fingerprint.count for e in ready])
        # Source calc_weight: average WITHIN-concept sigma, weighted by its
        # observation count. A global mixed-concept sigma is not substituted.
        within_sigma = (sigmas * counts).sum(axis=0) / counts.sum(axis=0)
        scale = 1.0 / np.maximum(within_sigma, self.std_floor)
        if len(ready) < 2 and not any(e.cross_fingerprints for e in ready):
            return scale / max(float(scale.max()), 1e-12)
        inter = means.std(axis=0) / np.maximum(sigmas.max(axis=0), self.std_floor)
        intra_scores = []
        for entry in ready:
            group = [entry.fingerprint] + [
                moment for moment in entry.cross_fingerprints.values() if moment.ready
            ]
            if len(group) < 2:
                intra_scores.append(np.zeros_like(inter))
                continue
            group_means = np.stack([self.normalizer.transform(m.mean) for m in group])
            # Paper III-B v_sc denominator is sigma^{S,C}, the WITHIN
            # dispersion of source S evaluated on the current concept C.
            # The max-sigma construction belongs only to v_s above.
            current_group = (
                entry.fingerprint if entry.memory_id == self.current_id
                else entry.cross_fingerprints.get(self.current_id)
            )
            if current_group is None or not current_group.ready:
                intra_scores.append(np.zeros_like(inter))
                continue
            current_sigma = current_group.std / self.normalizer.span
            intra_scores.append(
                group_means.std(axis=0) /
                np.maximum(current_sigma, self.std_floor)
            )
        intra = np.mean(intra_scores, axis=0) if intra_scores else np.zeros_like(inter)
        weight = scale * np.maximum(inter, intra)
        # Constant common scaling cancels algebraically from weighted cosine.
        if weight.max() <= 0:
            weight = scale
        return weight / max(float(weight.max()), 1e-12)

    def _similarity(self, a, b, weights=None):
        if weights is None:
            weights = self.weights()
        return weighted_cosine(
            self.normalizer.transform(a), self.normalizer.transform(b), weights,
        )

    def _update_stable(self):
        if self.current_id is None or len(self.stable_window) < 5:
            return
        entry = self.entries[self.current_id]
        vector = self._fingerprint(entry, self.stable_window)
        self.normalizer.update(vector)
        if entry.ready and self.seen >= entry.version_clean_at:
            similarity = self._similarity(entry.fingerprint.mean, vector)
            entry.normal_similarity.append(entry.fingerprint.mean, vector, similarity)
        entry.fingerprint.update(vector)
        entry.last_active_at = self.seen
        self.last_fingerprint = self.seen

    def _update_repository(self):
        if self.current_id is None or len(self.stable_window) < 5:
            return
        for entry in self.entries.values():
            if entry.memory_id == self.current_id:
                continue
            vector = self._fingerprint(entry, self.stable_window)
            self.normalizer.update(vector)
            moment = entry.cross_fingerprints.setdefault(
                self.current_id, VectorMoments(len(self.feature_names))
            )
            moment.update(vector)
        self.last_repository = self.seen

    def observe_passive(
        self, X: np.ndarray, y: np.ndarray,
        reference_model: Predictor | None = None, model_version: Any = None,
    ) -> list[dict]:
        """Append only incoming passive data; never pass active query feedback."""
        if reference_model is not None:
            if self.current_id is None:
                raise ValueError("Register a calibrated frozen model before passing a reference.")
            if model_version != self.entries[self.current_id].model_version:
                raise ValueError("New model versions require register_current with fresh holdout.")
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape != (len(y), self.dimension) or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Invalid passive batch shape or non-finite values.")
        before = len(self.events)
        for x, target in zip(X, y):
            self.seen += 1
            self.segment_seen += 1
            row = (self.seen, x.copy(), float(target))
            self.active_window.append(row)
            self.delay_buffer.append(row)
            # Author's dynamic delay: floor(observations_since_transition*.25).
            buffer_length = math.floor(self.segment_seen * self.buffer_ratio)
            if len(self.delay_buffer) >= max(buffer_length, 1):
                stable_row = self.delay_buffer.popleft()
                self.stable_window.append(stable_row)
                if self.current_id is not None:
                    self.entries[self.current_id].record_stable(stable_row)
            self._cache.clear()
            if self.seen - self.last_fingerprint >= self.fingerprint_gap:
                self._update_stable()
            if self.seen - self.last_repository >= self.repository_gap:
                self._update_repository()
            if (
                self.current_id is not None and len(self.active_window) >= 5
                and self.seen - self.last_similarity >= self.similarity_gap
            ):
                entry = self.entries[self.current_id]
                if entry.ready and self.seen >= entry.version_clean_at:
                    vector = self._fingerprint(entry, self.active_window)
                    self.normalizer.update(vector)
                    similarity = self._similarity(entry.fingerprint.mean, vector)
                    normal_mu, normal_sd, _, _ = entry.normal_similarity.adjusted(self._similarity, self.std_floor)
                    self.detector.update(1.0 + abs(similarity - normal_mu) / normal_sd)
                    if self.detector.drift_detected:
                        self._event("internal_drift_signal", memory_id=self.current_id,
                                    similarity=similarity, budget_action="none")
                self.last_similarity = self.seen
            if self.secondary_match_at is not None and self.seen >= self.secondary_match_at:
                self._secondary_match()
        return self.events[before:]

    def retrieve(self, top_k: int | None = 3, accepted_only: bool = True) -> list[Match]:
        if len(self.active_window) < 5:
            return []
        observed = {}
        for entry in self.entries.values():
            if entry.ready:
                observed[entry.memory_id] = self._fingerprint(entry, self.active_window)
                self.normalizer.update(observed[entry.memory_id])
        # All comparisons use one COMMON normalization/weight snapshot.
        weights = self.weights()
        similarity_function = lambda a, b: self._similarity(a, b, weights)
        matches = []
        for memory_id, vector in observed.items():
            entry = self.entries[memory_id]
            sim = similarity_function(entry.fingerprint.mean, vector)
            mu, sd, count, ratio = entry.normal_similarity.adjusted(similarity_function, self.std_floor)
            accepted = bool(
                count >= 2 and abs(sim - mu) <= 2 * sd
                and self.seen >= entry.version_clean_at
            )
            match = Match(memory_id, entry.model, entry.payload, sim, mu, sd, accepted,
                          count, self.normalizer.version, entry.model_version)
            if accepted or not accepted_only:
                matches.append(match)
        matches.sort(key=lambda m: (-m.similarity, m.memory_id))
        self._last_matches = matches
        return matches if top_k is None else matches[:top_k]

    def on_alarm(self) -> list[Match]:
        """Use the common external alarm; select recurrence or request a new model."""
        matches = self.retrieve(top_k=None, accepted_only=False)
        previous = self.current_id
        accepted = [m for m in matches if m.accepted]
        self.secondary_candidate_ids = set(self.entries)
        self.current_id = accepted[0].memory_id if accepted else None
        self.secondary_match_at = self.seen + self.window_size
        self.segment_seen = 0
        self.delay_buffer.clear()
        self.stable_window.clear()
        self.detector._reset()
        self._event(
            "external_alarm_selection", previous_id=previous, current_id=self.current_id,
            accepted_ids=[m.memory_id for m in accepted],
            secondary_match_at=self.secondary_match_at,
            matches=[m.diagnostic() for m in matches],
        )
        return matches

    def _secondary_match(self):
        matches = self.retrieve(top_k=None, accepted_only=False)
        accepted = [
            m for m in matches if m.accepted and m.memory_id in self.secondary_candidate_ids
        ]
        previous = self.current_id
        if accepted:
            self.current_id = accepted[0].memory_id
            if previous is not None and previous != self.current_id:
                provisional = self.entries.get(previous)
                if provisional is not None and provisional.provisional:
                    self.entries[self.current_id].stable_observations.update(
                        provisional.stable_observations
                    )
                    del self.entries[previous]
                    for entry in self.entries.values():
                        entry.cross_fingerprints.pop(previous, None)
            # A just-created alternative is removed exactly as paper Sec. III-A.
        if self.current_id is not None:
            self.entries[self.current_id].provisional = False
        self._event("secondary_selection", previous_id=previous, current_id=self.current_id,
                    accepted_ids=[m.memory_id for m in accepted])
        self.secondary_match_at = None
        self.secondary_candidate_ids.clear()

    def diagnostics(self):
        return {
            "upstream_commit": UPSTREAM_COMMIT,
            "dimension": self.dimension,
            "feature_count": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "passive_seen": self.seen, "current_id": self.current_id,
            "entry_count": len(self.entries),
            "normalizer_version": self.normalizer.version,
            "meta_feature_calls": self.extractor.calls,
            "shap_prediction_rows": self.extractor.shap_prediction_rows,
            "shap_permutations": self.extractor.shap_permutations,
            "degenerate_series": self.extractor.degenerate_series,
            "parameters": {
                "window_size": self.window_size, "buffer_ratio": self.buffer_ratio,
                "similarity_gap": self.similarity_gap, "fingerprint_gap": self.fingerprint_gap,
                "repository_gap": self.repository_gap, "std_floor": self.std_floor,
            },
            "entries": [{
                "memory_id": e.memory_id, "model_version": e.model_version,
                "created_at": e.created_at, "last_active_at": e.last_active_at,
                "error_threshold": e.error_threshold, "calibration_size": len(e.calibration_y),
                "fingerprint_count_min": int(e.fingerprint.count.min()),
                "fingerprint_count_max": int(e.fingerprint.count.max()),
                "normal_similarity_count": len(e.normal_similarity.pairs),
                "cross_concepts": sorted(e.cross_fingerprints),
                "stable_observations": len(e.stable_observations),
                "stable_first_index": int(e.stream_indices.min()) if len(e.stream_indices) else None,
                "stable_last_index": int(e.stream_indices.max()) if len(e.stream_indices) else None,
                "version_history": e.version_history,
            } for e in self.entries.values()],
        }

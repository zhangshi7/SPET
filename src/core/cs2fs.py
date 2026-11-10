"""CS²-FS: causal selective shrinkage Fixed-Share ensemble for MI-10.

Selection is local to one evolutionary generation.  All experts first score
the generation's population, four strata are formed at the full-pool proxy
optimum, and exactly one expert is selected from each stratum using only
previously revealed decision loss plus complementarity.  The selected experts
are then shrinkage-stacked; they are not averaged uniformly.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata


EPS = 1e-8
Q = 4
TOP_K = 5
WINDOW_MEMORY = 5
RECENCY_DECAY = 0.80
FIXED_SHARE = 0.10
PRIOR_STRENGTH = 2.0
STABILITY_STRENGTH = 1.0
WEIGHT_FLOOR = 0.10
PGD_STEPS = 80
DECISION_WEIGHTS = (0.25, 0.35, 0.40)
DIVERSITY_STRENGTH = 0.15
CANONICAL_POOL = (
    ("GPR-global", "GPR", None),
    ("GPR-recent-50", "GPR", 50),
    ("GPR-recent-100", "GPR", 100),
    ("GPR-recent-200", "GPR", 200),
    ("RBF-global", "RBF", None),
    ("RBF-recent-50", "RBF", 50),
    ("RBF-recent-100", "RBF", 100),
)


def _family(name: str) -> str:
    if name.startswith("GPR-"):
        return "GPR"
    if name.startswith("RBF-"):
        return "RBF"
    raise ValueError(f"unknown CS2FS model family: {name}")


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)
    if len(a) < 2 or np.std(a) <= EPS or np.std(b) <= EPS:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else 0.0


def project_simplex_with_floor(values: Sequence[float], floor: float = WEIGHT_FLOOR) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("finite non-empty weight vector required")
    mass = 1.0 - len(values) * float(floor)
    if mass <= 0:
        raise ValueError("weight floor leaves no simplex mass")
    shifted = values - float(floor)
    ordered = np.sort(shifted)[::-1]
    cumulative = np.cumsum(ordered) - mass
    admissible = np.flatnonzero(
        ordered - cumulative / np.arange(1, len(values) + 1) > 0)
    theta = cumulative[admissible[-1]] / float(admissible[-1] + 1) if len(admissible) else 0.0
    result = np.maximum(shifted - theta, 0.0) + float(floor)
    result /= float(np.sum(result))
    if not np.all(np.isfinite(result)) or abs(float(np.sum(result)) - 1.0) > 1e-10:
        raise FloatingPointError("invalid projected weights")
    return result


def decision_metrics(y: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, float).reshape(-1)
    prediction = np.asarray(prediction, float).reshape(-1)
    scale = max(float(np.quantile(y, .90) - np.quantile(y, .10)),
                float(np.std(y)), EPS)
    nrmse = float(np.sqrt(np.mean((prediction - y) ** 2)) / scale)
    spearman = _safe_corr(rankdata(y), rankdata(prediction))
    selected = np.argsort(prediction)[:min(TOP_K, len(prediction))]
    regret = ((float(np.min(y[selected])) - float(np.min(y))) / scale)
    return {"nrmse": nrmse, "spearman": spearman,
            "rank_loss": 1.0 - spearman, "topk_regret": max(regret, 0.0)}


def decision_loss(metrics: Mapping[str, float]) -> float:
    return (DECISION_WEIGHTS[0] * float(metrics["nrmse"])
            + DECISION_WEIGHTS[1] * float(metrics["rank_loss"])
            + DECISION_WEIGHTS[2] * float(metrics["topk_regret"]))


class CausalSelectiveShrinkageFixedShareGate:
    """Causal stacking weights plus four-expert generation-local selection."""

    def __init__(self, prior_strength: float = PRIOR_STRENGTH,
                 stability_strength: float = STABILITY_STRENGTH,
                 diversity_strength: float = DIVERSITY_STRENGTH):
        self.prior_strength = float(prior_strength)
        self.stability_strength = float(stability_strength)
        self.diversity_strength = float(diversity_strength)
        self.names: List[str] = []
        self.weights = np.empty(0, dtype=float)
        self.history = deque(maxlen=WINDOW_MEMORY)
        self.last_scores: Dict[str, float] = {}
        self.last_metrics: Dict[str, Dict[str, float]] = {}
        self.last_correlations: Dict[str, Dict[str, float]] = {}
        self.update_count = 0
        self.active_generation = None
        self.active_selection: Tuple[str, ...] = ()
        self.active_weights: Dict[str, float] = {}
        self.active_score_calls = 0
        self.generation_records: List[Dict] = []
        self.selection_counts = Counter()

    def _reset(self, names: Sequence[str]) -> None:
        self.names = sorted(names)
        self.weights = np.full(len(self.names), 1.0 / len(self.names))
        self.history.clear()
        self.last_scores = {name: 1.0 for name in self.names}
        self.last_metrics = {}
        self.last_correlations = {}
        self.active_generation = None
        self.active_selection = ()
        self.active_weights = {}

    def update(self, validation: Mapping[str, Mapping]) -> None:
        """Consume predictions only after their current labels are revealed."""
        names = sorted(validation)
        if not names:
            return
        if names != self.names:
            self._reset(names)
        first = validation[names[0]]
        if "prediction_vector" not in first or "target_vector" not in first:
            raise ValueError("CS2FS requires pre-label prediction and revealed-target vectors")
        y = np.asarray(first["target_vector"], float).reshape(-1)
        predictions = np.column_stack([
            np.asarray(validation[name]["prediction_vector"], float).reshape(-1)
            for name in names])
        finite = np.isfinite(y) & np.all(np.isfinite(predictions), axis=1)
        y, predictions = y[finite], predictions[finite]
        if len(y) == 0:
            return
        location = float(np.median(y))
        scale = max(float(np.quantile(y, .90) - np.quantile(y, .10)),
                    float(np.std(y)), EPS)
        yz, pz = (y - location) / scale, (predictions - location) / scale
        window_metrics = {
            name: decision_metrics(y, predictions[:, index])
            for index, name in enumerate(names)}
        self.history.append({"predictions": pz, "target": yz,
                             "metrics": window_metrics})

        blocks_p, blocks_y, blocks_q = [], [], []
        newest = len(self.history) - 1
        for index, block in enumerate(self.history):
            recency = RECENCY_DECAY ** (newest - index)
            blocks_p.append(block["predictions"])
            blocks_y.append(block["target"])
            blocks_q.append(np.full(len(block["target"]), recency, dtype=float))
        matrix = np.vstack(blocks_p)
        targets = np.concatenate(blocks_y)
        sample_weights = np.concatenate(blocks_q)
        weight_sum = max(float(np.sum(sample_weights)), EPS)
        uniform = np.full(len(names), 1.0 / len(names))
        previous = self.weights.copy()
        weighted_matrix = matrix * np.sqrt(sample_weights[:, None] / weight_sum)
        lipschitz = (2.0 * float(np.linalg.norm(weighted_matrix, 2) ** 2)
                     + 2.0 * (self.prior_strength + self.stability_strength))
        step = 1.0 / max(lipschitz, EPS)
        weights = previous.copy()
        for _ in range(PGD_STEPS):
            residual = matrix @ weights - targets
            gradient = (2.0 * matrix.T @ (sample_weights * residual) / weight_sum
                        + 2.0 * self.prior_strength * (weights - uniform)
                        + 2.0 * self.stability_strength * (weights - previous))
            weights = project_simplex_with_floor(weights - step * gradient)
        # Fixed-Share applies to the entire pool, including experts omitted in
        # the previous generation, so their recovery channel remains open.
        weights = (1.0 - FIXED_SHARE) * weights + FIXED_SHARE * uniform
        self.weights = project_simplex_with_floor(weights)

        recency_denominator = sum(RECENCY_DECAY ** (newest - i)
                                  for i in range(len(self.history)))
        self.last_metrics = {}
        self.last_scores = {}
        for name in names:
            metrics = {
                metric: float(sum(
                    (RECENCY_DECAY ** (newest - i)) * self.history[i]["metrics"][name][metric]
                    for i in range(len(self.history))) / recency_denominator)
                for metric in ("nrmse", "spearman", "rank_loss", "topk_regret")}
            self.last_metrics[name] = metrics
            self.last_scores[name] = decision_loss(metrics)
        self.last_correlations = {left: {} for left in names}
        for i, left in enumerate(names):
            for j, right in enumerate(names):
                self.last_correlations[left][right] = _safe_corr(matrix[:, i], matrix[:, j])
        self.update_count += 1
        # Do not mutate active_selection/active_weights.  A revealed label may
        # affect only begin_generation() for a new generation token.

    def _complementarity(self, candidate: str, selected: Sequence[str],
                         current_predictions: Mapping[str, np.ndarray]) -> float:
        if not selected:
            return 0.0
        values = []
        for other in selected:
            if candidate in self.last_correlations and other in self.last_correlations[candidate]:
                correlation = self.last_correlations[candidate][other]
            else:
                correlation = _safe_corr(current_predictions[candidate], current_predictions[other])
            values.append(1.0 - abs(float(correlation)))
        return float(np.mean(values))

    def begin_generation(self, generation_token, models: Mapping,
                         candidates: np.ndarray) -> Dict[str, float]:
        """Freeze one Q=4 subset and its shrinkage weights for a generation."""
        if self.active_generation == generation_token:
            return dict(self.active_weights)
        names = sorted(models)
        if len(names) < Q:
            raise ValueError(f"CS2FS needs at least {Q} distinct models, got {len(names)}")
        if names != self.names:
            self._reset(names)
        candidates = np.asarray(candidates, float)
        predictions = {}
        uncertainties = {}
        for name in names:
            mean, uncertainty = models[name].predict(candidates)
            predictions[name] = np.asarray(mean, float).reshape(-1)
            uncertainties[name] = np.asarray(uncertainty, float).reshape(-1)
        rank_matrix = np.column_stack([rankdata(predictions[name]) for name in names])
        proxy_index = int(np.argmin(np.mean(rank_matrix, axis=1)))
        layered = sorted(names, key=lambda name: (predictions[name][proxy_index], name))
        strata = [list(group) for group in np.array_split(np.asarray(layered, object), Q)]
        selected: List[str] = []
        selection_details = []
        for stratum_index, stratum in enumerate(strata):
            scored = []
            for name in stratum:
                loss = float(self.last_scores.get(name, 1.0))
                complementarity = self._complementarity(name, selected, predictions)
                score = loss - self.diversity_strength * complementarity
                scored.append((score, loss, -complementarity, name, complementarity))
            choice = min(scored)[3]
            selected.append(choice)
            chosen = next(item for item in scored if item[3] == choice)
            selection_details.append({"stratum": stratum_index, "members": stratum,
                                      "selected": choice, "decision_loss": chosen[1],
                                      "complementarity": chosen[4], "selection_score": chosen[0]})

        families = {_family(name) for name in selected}
        for missing_family in ({"GPR", "RBF"} - families):
            alternatives = [name for name in names
                            if _family(name) == missing_family and name not in selected]
            if not alternatives:
                raise AssertionError(f"no {missing_family} expert available")
            replacement = min(alternatives, key=lambda name: self.last_scores.get(name, 1.0))
            replaceable = [name for name in selected
                           if sum(_family(item) == _family(name) for item in selected) > 1]
            victim = max(replaceable, key=lambda name: self.last_scores.get(name, 1.0))
            selected[selected.index(victim)] = replacement
            selection_details.append({"family_repair": missing_family,
                                      "removed": victim, "inserted": replacement})

        if len(selected) != Q or len(set(selected)) != Q:
            raise AssertionError(f"generation selection is not exactly Q={Q}: {selected}")
        if {_family(name) for name in selected} != {"GPR", "RBF"}:
            raise AssertionError(f"generation selection lacks a family: {selected}")
        lookup = dict(zip(self.names, self.weights))
        restricted = np.asarray([lookup[name] for name in selected], float)
        restricted /= max(float(np.sum(restricted)), EPS)
        restricted = project_simplex_with_floor(restricted)
        active_weights = {name: float(weight) for name, weight in zip(selected, restricted)}
        self.active_generation = generation_token
        self.active_selection = tuple(selected)
        self.active_weights = active_weights
        self.active_score_calls = 0
        self.selection_counts.update(selected)
        self.generation_records.append({
            "generation_token": str(generation_token),
            "proxy_candidate_index": proxy_index,
            "full_pool_models": names,
            "full_pool_size": len(names),
            "selected_models": selected,
            "selected_model_count": len(selected),
            "selected_weights": active_weights,
            "strata": selection_details,
            "decision_losses_from_revealed_history": dict(self.last_scores),
            "generation_fixed_selection_ok": True,
            "has_gpr": any(_family(name) == "GPR" for name in selected),
            "has_rbf": any(_family(name) == "RBF" for name in selected),
            "selection_uses_current_unrevealed_labels": False,
        })
        return dict(active_weights)

    def note_score_call(self, generation_token) -> None:
        if generation_token != self.active_generation:
            raise AssertionError("score call does not match active generation")
        self.active_score_calls += 1
        self.generation_records[-1]["score_calls"] = self.active_score_calls
        self.generation_records[-1]["generation_fixed_selection_ok"] = (
            tuple(self.generation_records[-1]["selected_models"]) == self.active_selection)

    def select(self, model_names: Sequence[str]) -> Dict[str, float]:
        names = list(model_names)
        if self.active_selection and all(name in names for name in self.active_selection):
            return dict(self.active_weights)
        if not names:
            return {}
        if sorted(names) != self.names:
            self._reset(names)
        return dict(zip(self.names, self.weights))

    def state(self) -> Dict:
        return {
            "name": "causal-selective-shrinkage-fixed-share-v1",
            "pool_models": self.names,
            "pool_weights": self.weights.tolist(),
            "active_generation": None if self.active_generation is None else str(self.active_generation),
            "active_selection": list(self.active_selection),
            "active_weights": dict(self.active_weights),
            "history_windows": len(self.history),
            "updates": self.update_count,
            "window_memory": WINDOW_MEMORY,
            "recency_decay": RECENCY_DECAY,
            "fixed_share": FIXED_SHARE,
            "prior_strength": self.prior_strength,
            "stability_strength": self.stability_strength,
            "weight_floor": WEIGHT_FLOOR,
            "selection_q": Q,
            "decision_weights_nrmse_rank_topk": list(DECISION_WEIGHTS),
            "selection_counts": dict(self.selection_counts),
            "last_decision_metrics": self.last_metrics,
            "last_correlations": self.last_correlations,
        }


class CS2FSEnsemble:
    def __init__(self, gate: CausalSelectiveShrinkageFixedShareGate):
        self.gate = gate

    def begin_generation(self, generation_token, models: Mapping,
                         candidates: np.ndarray) -> Dict[str, float]:
        return self.gate.begin_generation(generation_token, models, candidates)

    def predict(self, models: Mapping, x: np.ndarray):
        weights = self.gate.select(list(models))
        means, uncertainties = {}, {}
        for name in weights:
            means[name], uncertainties[name] = models[name].predict(x)
            means[name] = np.asarray(means[name], float).reshape(-1)
            uncertainties[name] = np.nan_to_num(
                np.asarray(uncertainties[name], float).reshape(-1), nan=0.0)
        mean = sum(weights[name] * means[name] for name in weights)
        variance = sum(weights[name] * (uncertainties[name] ** 2
                       + (means[name] - mean) ** 2) for name in weights)
        return np.asarray(mean), np.sqrt(np.maximum(variance, 1e-12)), weights


class GPRModel:
    def __init__(self, length_multiplier: float, noise: float):
        self.length_multiplier = float(length_multiplier)
        self.noise = float(noise)

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.x = np.asarray(x, float)
        y = np.asarray(y, float).reshape(-1)
        distances = np.sqrt(np.sum((self.x[:, None] - self.x[None, :]) ** 2, axis=-1))
        nonzero = distances[distances > EPS]
        self.length = max((float(np.median(nonzero)) if len(nonzero) else 1.0)
                          * self.length_multiplier, 1e-6)
        covariance = self._kernel(self.x, self.x) + (self.noise + 1e-6) * np.eye(len(self.x))
        try:
            self.inverse = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            self.inverse = np.linalg.pinv(covariance)
        self.alpha = self.inverse @ y
        return self

    def _kernel(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        distance2 = np.sum((left[:, None] - right[None, :]) ** 2, axis=-1)
        return np.exp(-0.5 * distance2 / max(self.length ** 2, EPS))

    def predict(self, x: np.ndarray):
        cross = self._kernel(np.asarray(x, float), self.x)
        mean = cross @ self.alpha
        variance = np.maximum(1.0 - np.sum((cross @ self.inverse) * cross, axis=1), EPS)
        return mean, np.sqrt(variance)


class RBFModel:
    def __init__(self, centers: np.ndarray, width: float, ridge: float,
                 weight_prior: np.ndarray):
        self.centers = np.asarray(centers, float)
        self.width = float(width)
        self.ridge = float(ridge)
        self.weight_prior = np.asarray(weight_prior, float)

    def _phi(self, x: np.ndarray) -> np.ndarray:
        distance2 = np.sum((np.asarray(x, float)[:, None] - self.centers[None, :]) ** 2, axis=-1)
        return np.c_[np.exp(-0.5 * distance2 / max(self.width ** 2, EPS)), np.ones(len(x))]

    def fit(self, x: np.ndarray, y: np.ndarray):
        phi = self._phi(x)
        prior = self.weight_prior
        if len(prior) != phi.shape[1]:
            raise ValueError("RBF output-weight prior dimension mismatch")
        matrix = phi.T @ phi + self.ridge * np.eye(phi.shape[1])
        rhs = phi.T @ np.asarray(y, float).reshape(-1) + self.ridge * prior
        try:
            self.weight = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            self.weight = np.linalg.pinv(matrix) @ rhs
        self.residual = max(float(np.std(phi @ self.weight - y)), 1e-3)
        return self

    def predict(self, x: np.ndarray):
        mean = self._phi(x) @ self.weight
        return mean, np.full(len(mean), self.residual)


class ScaleAdapter:
    def __init__(self, model, x_mean, x_scale, y_mean, y_scale, scope: Mapping):
        self.model = model
        self.x_mean = np.asarray(x_mean, float)
        self.x_scale = np.asarray(x_scale, float)
        self.y_mean = float(y_mean)
        self.y_scale = float(y_scale)
        self.scope = dict(scope)

    def predict(self, x: np.ndarray):
        normalized = (np.asarray(x, float) - self.x_mean) / self.x_scale
        mean, uncertainty = self.model.predict(normalized)
        return (self.y_mean + self.y_scale * np.asarray(mean),
                np.maximum(abs(self.y_scale) * np.asarray(uncertainty), 1e-8))

    def get_state(self) -> Dict:
        return {"family": type(self.model).__name__, **self.scope}


def build_heterogeneous_pool(visible_data: np.ndarray, prior: Mapping,
                             seed: int = 0) -> Tuple[Dict[str, ScaleAdapter], Dict]:
    """Build canonical scoped GPR/RBF experts, omitting duplicate scopes.

    If a recent scope equals the all-visible scope during cold start it is not
    instantiated as fake diversity.  Once more than 200 observations are
    visible, the exact seven-member canonical pool is present.
    """
    del seed  # deterministic priors; retained for a stable integration API
    data = np.asarray(visible_data, float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("visible_data must contain features and final-column target")
    x, y = data[:, :-1], data[:, -1]
    x_mean = np.mean(x, axis=0)
    x_scale = np.maximum(np.std(x, axis=0), 1e-6)
    y_mean = float(np.mean(y))
    y_scale = max(float(np.std(y)), 1e-6)
    xn, yn = (x - x_mean) / x_scale, (y - y_mean) / y_scale
    models: Dict[str, ScaleAdapter] = {}
    scope_rows, omitted, seen = [], [], set()
    for name, family, limit in CANONICAL_POOL:
        count = len(xn) if limit is None else min(int(limit), len(xn))
        start = len(xn) - count
        signature = (family, start, count)
        if signature in seen:
            omitted.append({"model": name, "reason": "duplicate_family_and_data_scope",
                            "start": start, "count": count})
            continue
        seen.add(signature)
        scoped_x, scoped_y = xn[start:], yn[start:]
        if family == "GPR":
            parameters = prior["gpr_prior"]
            model = GPRModel(parameters["length_multiplier"], parameters["noise"]).fit(
                scoped_x, scoped_y)
        else:
            parameters = prior["rbf_prior"]
            model = RBFModel(np.asarray(prior["rbf_centers"], float),
                             parameters["width"], parameters["ridge"],
                             np.asarray(parameters["weight_prior"], float)).fit(scoped_x, scoped_y)
        scope = {"model_name": name, "data_scope": "all_visible" if limit is None else f"recent_{limit}",
                 "fit_start_index": start, "fit_sample_count": count,
                 "causally_visible_only": True}
        models[name] = ScaleAdapter(model, x_mean, x_scale, y_mean, y_scale, scope)
        scope_rows.append(scope)
    audit = {
        "canonical_pool": [item[0] for item in CANONICAL_POOL],
        "active_pool": sorted(models),
        "active_pool_size": len(models),
        "fit_scopes": scope_rows,
        "omitted_duplicate_slots": omitted,
        "duplicate_models_instantiated": False,
        "visible_sample_count": len(data),
        "future_values_used": False,
    }
    return models, audit

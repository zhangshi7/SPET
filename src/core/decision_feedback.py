"""Causal decision-feedback extension of MetaCaSHE.

The policy accepts observations and an x-only objective callback.  Landscape
identifiers, true change points and known optimum locations have no API here.
All constants below are frozen independently of the target five-seed results.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

import conditional_meta
import cs2fs


@dataclass(frozen=True)
class DecisionFeedbackConfig:
    pool_cap: int = 256
    local_passive: int = 50
    candidate_count: int = 128
    population_size: int = 32
    persistent_decay: float = 0.5
    feedback_decay: float = 0.6
    weight_temperature: float = 0.35
    share: float = 0.05
    trust_initial: float = 0.35
    trust_min: float = 0.025
    trust_max: float = 0.7
    quadratic_ridge: float = 1e-4
    cubic_smoothing: float = 1e-7
    optimizer_iterations: int = 40
    duplicate_tolerance: float = 1e-8


class LocalSurrogate:
    """Regularized generic quadratic or cubic RBF on a revealed local block."""

    def __init__(self, kind, cfg):
        self.kind, self.cfg = kind, cfg

    def _features(self, z):
        z = np.atleast_2d(np.asarray(z, float))
        if self.kind == "quadratic":
            columns = [np.ones(len(z))] + [z[:, j] for j in range(z.shape[1])]
            columns += [z[:, i] * z[:, j]
                        for i in range(z.shape[1]) for j in range(i, z.shape[1])]
            return np.column_stack(columns)
        return np.c_[cdist(z, self.z) ** 3, np.ones(len(z)), z]

    def fit(self, x, y):
        self.z = np.asarray(x, float)
        y = np.asarray(y, float)
        self.location = float(np.median(y))
        self.scale = max(float(np.std(y)), float(np.ptp(y)) * 0.1, 1e-8)
        target = (y - self.location) / self.scale
        if self.kind == "quadratic":
            design = self._features(self.z)
            regularizer = self.cfg.quadratic_ridge * np.eye(design.shape[1])
            regularizer[0, 0] = 1e-10
            self.weight = np.linalg.solve(design.T @ design + regularizer,
                                          design.T @ target)
        else:
            n, d = self.z.shape
            polynomial = np.c_[np.ones(n), self.z]
            kernel = cdist(self.z, self.z) ** 3
            matrix = np.block([[kernel + self.cfg.cubic_smoothing * np.eye(n), polynomial],
                               [polynomial.T, np.zeros((d + 1, d + 1))]])
            rhs = np.r_[target, np.zeros(d + 1)]
            try:
                self.weight = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                self.weight = np.linalg.lstsq(matrix, rhs, rcond=1e-10)[0]
        return self

    def predict(self, z):
        return self.location + self.scale * (self._features(z) @ self.weight)

    def value_gradient(self, point):
        point = np.asarray(point, float)
        d = len(point)
        value = float(self.predict(point[None])[0])
        if self.kind == "quadratic":
            gradient = self.weight[1:d + 1].copy()
            offset = d + 1
            for i in range(d):
                for j in range(i, d):
                    coefficient = self.weight[offset]
                    gradient[i] += coefficient * point[j]
                    gradient[j] += coefficient * point[i]
                    offset += 1
        else:
            difference = point[None] - self.z
            distance = np.linalg.norm(difference, axis=1)
            gradient = (3 * np.sum((self.weight[:len(self.z)] * distance)[:, None]
                                   * difference, axis=0)
                        + self.weight[len(self.z) + 1:])
        return value, self.scale * gradient


class DecisionFeedbackPolicy:
    """Persistent slot evidence plus current-block active decision feedback.

    The original seven GPR/RBF roles remain candidates.  Their historical
    prequential state persists.  Recent local roles are refitted after queried
    labels return.  All roles are evaluated on precisely the same queried
    points before that refit, making active routing loss prequential as well.
    """

    def __init__(self, lower, upper, seed, prior_bundle: Mapping, variant="full"):
        if variant not in {"full", "no_feedback", "no_local", "trace_pool"}:
            raise ValueError(f"unknown variant: {variant}")
        self.lower = np.asarray(lower, float)
        self.upper = np.asarray(upper, float)
        if self.lower.shape != self.upper.shape or np.any(self.upper <= self.lower):
            raise ValueError("finite ordered box bounds required")
        self.width = self.upper - self.lower
        self.dimension = len(self.lower)
        self.seed, self.variant = int(seed), variant
        self.rng = np.random.default_rng(self.seed)
        self.bundle, self.config = dict(prior_bundle), DecisionFeedbackConfig()
        self.passive_x = np.empty((0, self.dimension))
        self.passive_y = np.empty(0)
        self.models = {}
        self.local = {}
        self.losses = {}
        self.active_losses = {}
        self.prequential_records = []
        self.route_records = []
        self.pool_records = []
        self.population = np.empty((0, self.dimension))
        self.last_selection = set()
        self.ever_selected = set()
        self.window = 0
        self.search_count = 0
        self.fit_seconds = 0.0
        self.local_fit_seconds = 0.0
        self.core_builder = None

    @property
    def build_seconds(self):
        return self.fit_seconds + self.local_fit_seconds

    @property
    def total_build(self):
        return self.build_seconds

    def _normalize(self, x):
        return (np.asarray(x, float) - self.lower) / self.width

    def _denormalize(self, z):
        return self.lower + np.asarray(z, float) * self.width

    @staticmethod
    def _loss(y, prediction):
        y, prediction = np.asarray(y, float), np.asarray(prediction, float)
        scale = max(float(np.std(y)), float(np.ptp(y)) * 0.25, 1e-8)
        residual = float(np.mean(np.minimum(np.abs(prediction - y) / scale, 5.0)))
        if len(y) >= 2 and np.std(y) > 1e-12 and np.std(prediction) > 1e-12:
            rho = float(np.corrcoef(rankdata(y), rankdata(prediction))[0, 1])
            rank_loss = (1.0 - float(np.nan_to_num(rho))) / 2.0
        else:
            rank_loss = 0.5
        regret = (float(y[int(np.argmin(prediction))]) - float(np.min(y))) / scale
        return float(0.25 * residual + 0.4 * rank_loss + 0.35 * min(regret, 5.0))

    @staticmethod
    def _pool_mean(model, x):
        # Candidate routing requires predictive means only; avoid an N^2 L
        # posterior-variance calculation that the decision rule does not use.
        if isinstance(model, cs2fs.ScaleAdapter) and isinstance(model.model, cs2fs.GPRModel):
            normalized = (np.asarray(x) - model.x_mean) / model.x_scale
            mean = model.model._kernel(normalized, model.model.x) @ model.model.alpha
            return model.y_mean + model.y_scale * mean
        prediction = model.predict(x)
        mean = prediction[0] if isinstance(prediction, tuple) else prediction
        return np.asarray(mean, float).reshape(-1)

    def _predict(self, z):
        z = np.atleast_2d(z)
        original = self._denormalize(z)
        result = {name: self._pool_mean(model, original) for name, model in self.models.items()}
        result.update({name: model.predict(z) for name, model in self.local.items()})
        return {name: np.nan_to_num(values, nan=1e100, posinf=1e100, neginf=-1e100)
                for name, values in result.items()}

    def observe_passive(self, x, y):
        """Score old state first, reveal and append the current passive block."""
        x, y = np.asarray(x, float), np.asarray(y, float).reshape(-1)
        if x.shape != (len(y), self.dimension) or len(y) == 0:
            raise ValueError("nonempty passive x/y block with matching shape required")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("finite passive observations required")
        if self.models or self.local:
            predictions = self._predict(self._normalize(x))
            measured = {name: self._loss(y, p) for name, p in predictions.items()}
            for name, loss in measured.items():
                self.losses[name] = (self.config.persistent_decay * self.losses.get(name, loss)
                                     + (1 - self.config.persistent_decay) * loss)
            self.prequential_records.append({"window": self.window, "loss": measured,
                                             "prediction_before_append": True,
                                             "n": len(y)})
        self.passive_x = np.vstack([self.passive_x, x])[-self.config.pool_cap:]
        self.passive_y = np.r_[self.passive_y, y][-self.config.pool_cap:]
        self.window += 1

    def _build_pool(self):
        start = time.perf_counter()
        if self.core_builder is not None:
            self.models = dict(self.core_builder(np.c_[self.passive_x, self.passive_y].copy()))
            if not self.models:
                raise ValueError("core_builder must return at least one fitted model")
            route = {"custom_core_builder": True, "meta_prior_used": False}
            audit = {"active_pool": sorted(self.models), "active_pool_size": len(self.models),
                     "canonical_pool": sorted(self.models)}
        elif self.variant == "trace_pool":
            raise ValueError("trace_pool requires an injected core_builder")
        elif "gpr_prior" in self.bundle:
            prior, route = self.bundle, {"meta_prior_fallback": False, "provided_prior": True}
        else:
            prior, route = conditional_meta.select_prior_from_visible(
                self.passive_x, self.passive_y, self.bundle,
                natural_count=min(len(self.passive_y), 50), active_count=0)
        if self.core_builder is None:
            self.models, audit = cs2fs.build_heterogeneous_pool(
                np.c_[self.passive_x, self.passive_y], prior, self.seed)
        self.route_records.append(route)
        audit.update({"pool_cap": self.config.pool_cap, "active_labels_in_pool": False,
                      "window": self.window})
        self.pool_records.append(audit)
        self.fit_seconds += time.perf_counter() - start

    def _build_local(self, z, y):
        self.local = {}
        if self.variant == "no_local":
            return
        start = time.perf_counter()
        _, unique = np.unique(np.round(z, decimals=11), axis=0, return_index=True)
        unique = np.sort(unique)
        z, y = np.asarray(z)[unique], np.asarray(y)[unique]
        for kind in ("quadratic", "cubic"):
            try:
                self.local[f"Local-{kind}"] = LocalSurrogate(kind, self.config).fit(z, y)
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                continue
        self.local_fit_seconds += time.perf_counter() - start

    def _weights(self, names):
        losses = []
        for name in names:
            passive = self.losses.get(name, 0.75)
            loss = self.active_losses.get(name, passive) if self.variant != "no_feedback" else passive
            losses.append(loss)
        values = np.asarray(losses)
        logits = -(values - np.min(values)) / self.config.weight_temperature
        weights = np.exp(np.clip(logits, -50, 0))
        weights /= np.sum(weights)
        return (1 - self.config.share) * weights + self.config.share / len(names)

    def _candidates(self, train_z, train_y, radius):
        champion = train_z[int(np.argmin(train_y))]
        n = self.config.candidate_count
        points = [self.rng.random((n // 2, self.dimension)),
                  np.clip(champion + self.rng.normal(0, radius / 3,
                                                    (n // 2, self.dimension)), 0, 1)]
        if len(self.population):
            pop = self.population
            a = pop[self.rng.integers(len(pop), size=len(pop))]
            b = pop[self.rng.integers(len(pop), size=len(pop))]
            points.append(np.clip(pop + 0.5 * (champion - pop) + 0.5 * (a - b), 0, 1))
        starts = train_z[np.argsort(train_y)[:min(3, len(train_y))]]
        bounds = list(zip(np.maximum(0, champion - radius), np.minimum(1, champion + radius)))
        for model in self.local.values():
            for start in starts:
                fitted = minimize(model.value_gradient,
                                  np.clip(start, np.array(bounds)[:, 0], np.array(bounds)[:, 1]),
                                  method="L-BFGS-B", jac=True, bounds=bounds,
                                  options={"maxiter": self.config.optimizer_iterations,
                                           "maxls": 12, "ftol": 1e-9})
                if np.all(np.isfinite(fitted.x)):
                    points.append(fitted.x[None])
        candidates = np.vstack(points)
        _, indices = np.unique(np.round(candidates, decimals=11), axis=0, return_index=True)
        return candidates[np.sort(indices)]

    def search(self, objective_callable, generations=8, batch_size=4):
        if len(self.passive_y) < 2:
            raise ValueError("at least two revealed passive samples required")
        if generations < 1 or batch_size < 1:
            raise ValueError("positive generations and batch size required")
        start = time.perf_counter()
        build_before, local_before = self.fit_seconds, self.local_fit_seconds
        self._build_pool()
        # Active observations are current-block state.  Only candidate positions
        # and persistent role evidence survive to a later passive window.
        train_z = self._normalize(self.passive_x[-self.config.local_passive:])
        train_y = self.passive_y[-self.config.local_passive:].copy()
        self.active_losses = {}
        radius = self.config.trust_initial
        queries, generation_records = [], []
        queried_positions = train_z.copy()
        for generation in range(int(generations)):
            self._build_local(train_z, train_y)
            candidates = self._candidates(train_z, train_y, radius)
            distances = np.min(cdist(candidates, queried_positions), axis=1)
            candidates = candidates[distances > self.config.duplicate_tolerance]
            while len(candidates) < batch_size:
                candidates = np.vstack([candidates, self.rng.random((batch_size, self.dimension))])
            predictions = self._predict(candidates)
            names = sorted(predictions)
            weights = self._weights(names)
            ranks = np.column_stack([rankdata(predictions[name]) / len(candidates) for name in names])
            consensus = ranks @ weights
            ordered_experts = np.argsort(-weights, kind="stable")
            selected, sponsors = [], []
            # Each query is a ranked recommendation, not a forced numerical
            # average of incompatible scales.  One consensus recommendation and
            # individual role recommendations share the same frozen batch.
            for query_index in range(batch_size):
                if query_index == 0:
                    scores, sponsor = consensus.copy(), "consensus"
                else:
                    expert = names[int(ordered_experts[(query_index - 1) % len(names)])]
                    scores, sponsor = ranks[:, names.index(expert)].copy(), expert
                if selected:
                    separation = np.min(cdist(candidates, candidates[selected]), axis=1)
                    scores += 0.1 * np.exp(-separation / max(radius * 0.1, 1e-6))
                    scores[selected] = np.inf
                index = int(np.argmin(scores))
                selected.append(index)
                sponsors.append(sponsor)
            selected = np.asarray(selected)
            xq = self._denormalize(candidates[selected])
            # Freeze the prediction record before objective evaluation.
            before_prediction = {name: predictions[name][selected].copy() for name in names}
            yq = np.asarray(objective_callable(xq.copy()), float).reshape(-1)
            if len(yq) != batch_size or not np.all(np.isfinite(yq)):
                raise ValueError("objective must return a finite value per queried point")
            measured = {name: self._loss(yq, p) for name, p in before_prediction.items()}
            if self.variant != "no_feedback":
                for name, loss in measured.items():
                    previous = self.active_losses.get(name, self.losses.get(name, loss))
                    self.active_losses[name] = (self.config.feedback_decay * previous
                                                + (1 - self.config.feedback_decay) * loss)
            for index in range(batch_size):
                queries.append({"generation": generation, "query_in_generation": index,
                                "x": xq[index].tolist(), "y": float(yq[index]),
                                "sponsor": sponsors[index],
                                "predictions_before_query": {name: float(p[index])
                                                              for name, p in before_prediction.items()}})
            selected_roles = set(sponsors) - {"consensus"}
            reactivated = sorted(selected_roles - self.last_selection & self.ever_selected)
            generation_records.append({"generation": generation, "weights": dict(zip(names, weights.tolist())),
                                       "prequery_losses": {name: float(self.losses.get(name, 0.75)) for name in names},
                                       "observed_active_loss": measured,
                                       "active_losses_after": dict(self.active_losses),
                                       "sponsors": sponsors, "reactivated_roles": reactivated,
                                       "trust_radius": float(radius), "candidate_count": len(candidates),
                                       "query_count": int(batch_size), "model_count": len(names),
                                       "future_values_used": False})
            self.last_selection, self.ever_selected = selected_roles, self.ever_selected | selected_roles
            improved = float(np.min(yq)) < float(np.min(train_y)) - 1e-10 * max(float(np.std(train_y)), 1)
            radius = min(self.config.trust_max, radius * 1.2) if improved else max(self.config.trust_min, radius * 0.8)
            train_z, train_y = np.vstack([train_z, candidates[selected]]), np.r_[train_y, yq]
            queried_positions = np.vstack([queried_positions, candidates[selected]])
            self.population = candidates[np.argsort(consensus)[:self.config.population_size]].copy()
        self.search_count += 1
        elapsed = time.perf_counter() - start
        return {"queries": queries, "generation_records": generation_records,
                "best_objective": min(query["y"] for query in queries),
                "real_queries": len(queries), "variant": self.variant,
                "pool_build_seconds": self.fit_seconds - build_before,
                "local_fit_seconds": self.local_fit_seconds - local_before,
                "search_seconds": elapsed, "config": asdict(self.config),
                "meta_route": self.route_records[-1], "pool_audit": self.pool_records[-1],
                "active_feedback_used": self.variant != "no_feedback",
                "true_environment_ids_used": False, "function_ids_used": False}

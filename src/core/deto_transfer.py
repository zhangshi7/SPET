"""Paper-derived DETO transfer expert; it never receives an objective callback.

References: Li, Chen & Yao, TEVC 28(5), 1396--1411 (2024), Sec. III.
The hierarchical covariance is Eq. (10), which also matches the author code.
Eq. (11)/(12) have inconsistent off-diagonal masks and are not used.
Independent implementation in NumPy/SciPy; the audited author snapshot is under
vendor/DETO. See protocol/DETO_IMPLEMENTATION.md for all integration decisions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Hashable, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import qmc


@dataclass(frozen=True)
class DETOConfig:
    """Numerical controls, not permission to omit any DETO mechanism."""

    minimize: bool = True
    gp_restarts: int = 10
    gp_max_iters: int = 1000
    target_noise: float = 1e-6
    population_size: int = 100
    generations: int = 20
    de_f: float = 0.5
    de_cr: float = 0.9
    beta: float = 3.0
    local_steps: int = 5
    local_learning_rate: float = 0.01
    local_displacement_threshold: float = 0.01
    initial_kappa: int = 5
    warm_count_per_dimension: int = 2
    warm_multistarts_per_dimension: int = 8
    warm_min_multistarts: int = 64
    warm_max_iters: int = 200
    warm_kkt_tolerance: float = 1e-3
    warm_distance_fraction: float = 0.01

    def __post_init__(self):
        if self.gp_restarts < 1 or self.gp_max_iters < 1:
            raise ValueError("MLE restarts and maximum iterations must be positive")
        if self.population_size < 4 or self.generations < 1 or self.local_steps < 1:
            raise ValueError("The hybrid must include DE and local search")
        if self.target_noise <= 0:
            raise ValueError("Numerical observation noise must be positive")


@dataclass
class SourceEntry:
    """A visible FiCSUM state, never a benchmark's hidden environment ID."""

    source_id: Hashable
    X: np.ndarray
    y: np.ndarray
    model: object | None = None
    order: int = 0


def source_entry(source_id, X, y, model=None, order=0):
    return SourceEntry(source_id, np.asarray(X, float), np.asarray(y, float).ravel(),
                       model=model, order=order)


def _arrays(X, y=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not len(X) or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty, finite two-dimensional array")
    if y is None:
        return X
    y = np.asarray(y, dtype=float).ravel()
    if len(y) != len(X) or not np.isfinite(y).all():
        raise ValueError("One finite label is required for every row of X")
    return X, y


def hierarchical_mask(task1, task2, layer):
    """Eq. (10), zero-indexed: a_l a_l^T, a_l(t)=1[t>=l]."""
    return ((np.asarray(task1, int)[:, None] >= layer)
            & (np.asarray(task2, int)[None, :] >= layer)).astype(float)


def hierarchical_kernel(X1, task1, X2, task2, variances, lengthscales):
    """Author-compatible RBF parameterization exp(-r^2/(2*l^2)).

    The paper's exp(-r^2/l_paper^2) is identical for l_paper=sqrt(2)*l.
    X values must already be normalized when normalization is desired.
    """
    X1, X2 = np.atleast_2d(X1), np.atleast_2d(X2)
    d2 = cdist(X1, X2, "sqeuclidean")
    K = np.zeros_like(d2)
    for layer, (variance, length) in enumerate(zip(variances, lengthscales)):
        K += (float(variance) * np.exp(-0.5 * d2 / float(length) ** 2)
              * hierarchical_mask(task1, task2, layer))
    return K


class HierarchicalGP:
    """Exact HMOGP posterior, marginal-likelihood training and analytic gradients.

    Source noise is learned per task, as in the author MixedNoise model. The
    target noise is fixed to 1e-6. No labels cross task boundaries.
    """

    def __init__(self, n_tasks, *, config=None, seed=0):
        self.n_tasks = int(n_tasks)
        if self.n_tasks < 1:
            raise ValueError("At least one task is required")
        self.config = config or DETOConfig()
        self.rng = np.random.default_rng(seed)
        self.theta = None
        self.fit_audit = {}

    def _unpack(self, theta):
        n = self.n_tasks
        variances = np.exp(theta[:n])
        lengthscales = np.exp(theta[n:2*n])
        noises = np.r_[np.exp(theta[2*n:]), self.config.target_noise]
        return variances, lengthscales, noises

    def _covariance_and_derivatives(self, theta):
        variances, lengths, noises = self._unpack(theta)
        n = len(self.X)
        K = np.zeros((n, n))
        dvar, dlength = [], []
        for layer in range(self.n_tasks):
            squared = self._d2 / lengths[layer] ** 2
            part = variances[layer] * np.exp(-0.5 * squared) * self._masks[layer]
            K += part
            dvar.append(part)
            dlength.append(part * squared)
        K.flat[::n+1] += noises[self.tasks]
        return K, dvar + dlength, noises

    @staticmethod
    def _factor(K):
        # Roundoff protection only: the model's task noise is already in K.
        for jitter in (0.0, 1e-12, 1e-10, 1e-8, 1e-6):
            try:
                trial = K.copy()
                trial.flat[::len(K)+1] += jitter
                return cho_factor(trial, lower=True, check_finite=False), jitter
            except np.linalg.LinAlgError:
                continue
        raise np.linalg.LinAlgError("HMOGP covariance could not be factored")

    def _objective(self, theta):
        K, derivatives, noises = self._covariance_and_derivatives(theta)
        try:
            factor, _ = self._factor(K)
        except np.linalg.LinAlgError:
            return 1e100, np.zeros_like(theta)
        alpha = cho_solve(factor, self.y_normalized, check_finite=False)
        nll = (0.5 * self.y_normalized @ alpha
               + np.log(np.diag(factor[0])).sum()
               + 0.5 * len(K) * np.log(2*np.pi))
        inverse = cho_solve(factor, np.eye(len(K)), check_finite=False)
        adjoint = inverse - np.outer(alpha, alpha)
        gradient = [0.5*np.einsum("ij,ij->", adjoint, d) for d in derivatives]
        diagonal = np.diag(adjoint)
        for source in range(self.n_tasks-1):
            gradient.append(0.5*noises[source]*diagonal[self.tasks == source].sum())
        return float(nll), np.asarray(gradient)

    def fit(self, X, y, tasks=None, *, optimize=True):
        X, y = _arrays(X, y)
        if tasks is None:
            tasks = np.full(len(X), self.n_tasks-1, int)
        tasks = np.asarray(tasks, int).ravel()
        if len(tasks) != len(X) or np.any(tasks < 0) or np.any(tasks >= self.n_tasks):
            raise ValueError("Invalid task indices")
        self.X_raw, self.y_raw, self.tasks = X.copy(), y.copy(), tasks.copy()
        self.x_mean = X.mean(axis=0)
        self.x_scale = X.std(axis=0)
        self.x_scale[self.x_scale < 1e-15] = 1.0
        self.y_mean = float(y.mean())
        self.y_scale = max(float(y.std()), 1e-15)
        # StandardScaler uses scale=1 for a constant target.
        if float(y.std()) < 1e-15:
            self.y_scale = 1.0
        self.X = (X-self.x_mean)/self.x_scale
        self.y_normalized = (y-self.y_mean)/self.y_scale
        self._d2 = cdist(self.X, self.X, "sqeuclidean")
        self._masks = [hierarchical_mask(tasks, tasks, i)
                       for i in range(self.n_tasks)]
        npar = 3*self.n_tasks-1
        if self.theta is None or len(self.theta) != npar:
            self.theta = np.r_[np.zeros(2*self.n_tasks),
                               np.zeros(self.n_tasks-1)]
        bounds = ([(-12., 12.)]*self.n_tasks + [(-6., 6.)]*self.n_tasks
                  + [(-16., 4.)]*(self.n_tasks-1))
        attempts = []
        if optimize:
            best_value, _ = self._objective(self.theta)
            best_theta = self.theta.copy()
            for restart in range(self.config.gp_restarts):
                initial = (self.theta.copy() if restart == 0 else
                           self.theta + self.rng.normal(0, 1, npar))
                initial = np.clip(initial, np.asarray(bounds)[:, 0],
                                  np.asarray(bounds)[:, 1])
                result = minimize(self._objective, initial, jac=True,
                                  method="L-BFGS-B", bounds=bounds,
                                  options={"maxiter": self.config.gp_max_iters,
                                           "ftol": 1e-9, "gtol": 1e-6,
                                           "maxls": 30})
                attempts.append({"restart": restart, "success": bool(result.success),
                                 "iterations": int(result.nit), "nll": float(result.fun)})
                if np.isfinite(result.fun) and result.fun < best_value:
                    best_value, best_theta = float(result.fun), result.x.copy()
            self.theta = best_theta
        K, _, _ = self._covariance_and_derivatives(self.theta)
        self.factor, jitter = self._factor(K)
        self.alpha = cho_solve(self.factor, self.y_normalized, check_finite=False)
        self.variances, self.lengthscales, self.noises = self._unpack(self.theta)
        self.fit_audit = {"samples": len(X), "tasks": self.n_tasks,
                          "task_counts": np.bincount(tasks, minlength=self.n_tasks).tolist(),
                          "optimized": bool(optimize), "attempts": attempts,
                          "jitter": jitter, "target_noise": self.config.target_noise,
                          "variances": self.variances.tolist(),
                          "lengthscales": self.lengthscales.tolist(),
                          "task_noises": self.noises.tolist()}
        return self

    def _cross(self, X, tasks, gradients=False):
        X = np.atleast_2d(np.asarray(X, float))
        z = (X-self.x_mean)/self.x_scale
        tasks = np.full(len(z), self.n_tasks-1, int) if tasks is None else np.asarray(tasks, int)
        distances = cdist(z, self.X, "sqeuclidean")
        K = np.zeros_like(distances)
        dK = np.zeros((*K.shape, z.shape[1])) if gradients else None
        diag = np.zeros(len(z))
        for layer, (variance, length) in enumerate(zip(self.variances, self.lengthscales)):
            part = (variance * np.exp(-0.5*distances/length**2)
                    * hierarchical_mask(tasks, self.tasks, layer))
            K += part
            diag += variance * (tasks >= layer)
            if gradients:
                dK += (part[:, :, None] * (self.X[None, :, :]-z[:, None, :])
                       / length**2 / self.x_scale)
        return K, diag, dK

    def predict(self, X, tasks=None):
        K, diag, _ = self._cross(X, tasks)
        mean = self.y_mean + self.y_scale*(K@self.alpha)
        v = solve_triangular(self.factor[0], K.T, lower=True, check_finite=False)
        variance = np.maximum(diag-np.einsum("ij,ij->j", v, v), 0.0)*self.y_scale**2
        return mean, variance

    def mean_and_gradient(self, X, tasks=None):
        K, _, dK = self._cross(X, tasks, gradients=True)
        mean = self.y_mean+self.y_scale*(K@self.alpha)
        dmean = self.y_scale*np.einsum("mnd,n->md", dK, self.alpha)
        return mean, dmean

    def predict_with_gradients(self, X, tasks=None):
        K, diag, dK = self._cross(X, tasks, gradients=True)
        mean = self.y_mean+self.y_scale*(K@self.alpha)
        solved = cho_solve(self.factor, K.T, check_finite=False)
        raw_variance = diag-np.einsum("mn,nm->m", K, solved)
        variance = np.maximum(raw_variance, 0.0)*self.y_scale**2
        dmean = self.y_scale*np.einsum("mnd,n->md", dK, self.alpha)
        dvariance = -2*self.y_scale**2*np.einsum("mnd,nm->md", dK, solved)
        dvariance[raw_variance <= 0] = 0.0
        return mean, variance, dmean, dvariance


class IndependentGP(HierarchicalGP):
    def __init__(self, *, config=None, seed=0):
        super().__init__(1, config=config, seed=seed)

    @property
    def hyperparameters(self):
        return np.array([self.lengthscales[0], self.variances[0]])


def select_representative_sources(entries, k=5, seed=0):
    """Original DETO Sec. III-B, retained for validation/ablation only.

    The combined method delegates retrieval to full FiCSUM instead. Returning
    this function's choices in the combined method would silently bypass it.
    """
    from sklearn.cluster import KMeans
    entries = list(entries)
    if len(entries) <= k:
        return sorted(entries, key=lambda e: e.order)
    if any(not isinstance(e.model, IndependentGP) for e in entries):
        raise ValueError("Representative clustering requires fitted independent GPs")
    h = np.asarray([e.model.hyperparameters for e in entries])
    h = (h-h.min(axis=0))/np.maximum(np.ptp(h, axis=0), 1e-15)
    cluster = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(h)
    chosen = []
    for label in range(k):
        members = np.flatnonzero(cluster.labels_ == label)
        if len(members):
            nearest = members[np.argmin(np.linalg.norm(
                h[members]-cluster.cluster_centers_[label], axis=1))]
            chosen.append(entries[int(nearest)])
    return sorted(chosen, key=lambda e: e.order)


def quality_distance_filter(X, values, limit, distance, *, minimize_objective=True):
    """Sec. III-C.2: predicted quality first, then Euclidean niche exclusion."""
    X, values = np.asarray(X, float), np.asarray(values, float)
    if int(limit) <= 0:
        return np.empty(0, int)
    order = np.argsort(values if minimize_objective else -values, kind="stable")
    selected = []
    for index in order:
        if not selected or np.all(np.linalg.norm(X[selected]-X[index], axis=1) >= distance):
            selected.append(int(index))
        if len(selected) >= limit:
            break
    return np.asarray(selected, int)


def source_local_optima(model, X_source, lower, upper, *, config, rng):
    """Bounded multistart stationary-point search of an independent source GP.

    All function/gradient evaluations are predictions. Boundaries are checked
    using the box projected KKT condition. Random starts are *not* returned as
    optima when their local solver fails.
    """
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    span, d = upper-lower, len(lower)
    cap = config.warm_count_per_dimension*d
    source_means, _ = model.mean_and_gradient(X_source)
    sign = 1.0 if config.minimize else -1.0
    best_indices = np.argsort(sign*source_means, kind="stable")[:cap]
    random_count = max(config.warm_min_multistarts,
                       config.warm_multistarts_per_dimension*d)
    latin = qmc.LatinHypercube(d=d, seed=int(rng.integers(2**31-1)))
    starts = np.vstack([np.clip((X_source[best_indices]-lower)/span, 0, 1),
                        latin.random(random_count)])
    optimum, diagnostics = [], []
    scale = model.y_scale

    def fun(z):
        mean, grad = model.mean_and_gradient(lower+z[None, :]*span)
        return float(sign*mean[0]/scale), sign*grad[0]*span/scale

    for initial in starts:
        result = minimize(fun, initial, jac=True, bounds=[(0., 1.)]*d,
                          method="L-BFGS-B",
                          options={"maxiter": config.warm_max_iters, "ftol": 1e-12,
                                   "gtol": config.warm_kkt_tolerance/10, "maxls": 40})
        z = np.clip(result.x, 0, 1)
        _, gradient = fun(z)
        projected = z-np.clip(z-gradient, 0, 1)
        residual = float(np.linalg.norm(projected, np.inf))
        acceptable = (np.isfinite(z).all() and np.isfinite(result.fun)
                      and residual <= config.warm_kkt_tolerance)
        diagnostics.append({"success": bool(result.success), "kkt_residual": residual,
                            "accepted": bool(acceptable), "iterations": int(result.nit)})
        if acceptable:
            optimum.append(lower+z*span)
    optima = np.asarray(optimum).reshape(-1, d)
    predicted = model.predict(optima)[0] if len(optima) else np.empty(0)
    selected = quality_distance_filter(
        optima, predicted, cap, config.warm_distance_fraction*float(span.max()),
        minimize_objective=config.minimize)
    selected_X = optima[selected]
    # Recompute in the final row layout so stored pseudo labels reproduce the
    # same prediction call even for highly correlated nearly-noiseless GPs.
    selected_y = model.predict(selected_X)[0] if len(selected_X) else np.empty(0)
    return selected_X, selected_y, {
        "starts": len(starts), "stationary_candidates": len(optima),
        "selected": len(selected), "requested_cap": cap,
        "shortfall": max(0, cap-len(selected)),
        "distance_threshold": config.warm_distance_fraction*float(span.max()),
        "solver": "bounded_L-BFGS-B_analytic_GP_mean_gradient",
        "solver_diagnostics": diagnostics,
    }


class DETOTransferExpert:
    """Fit selected histories, expose a posterior, and nominate free candidates.

    The only new labels accepted here are arrays already obtained by the
    caller. The caller owns the shared 8x4 real-evaluation budget.
    """

    def __init__(self, lower, upper, seed=0, config=None):
        self.lower, self.upper = np.asarray(lower, float), np.asarray(upper, float)
        if (self.lower.ndim != 1 or self.upper.shape != self.lower.shape
                or not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all()
                or np.any(self.upper <= self.lower)):
            raise ValueError("Finite ordered box bounds are required")
        self.span, self.dimension = self.upper-self.lower, len(self.lower)
        self.seed, self.config = int(seed), config or DETOConfig()
        self.rng = np.random.default_rng(seed)
        self._source_cache = {}
        self._warm_cache = {}
        self.model = None
        self.source_audit = []
        self.search_audit = []
        self.selected_ids = []
        self.warm_candidates = np.empty((0, self.dimension))
        self.pseudo_datasets = []
        self.target_X = None

    def _entry(self, value):
        if isinstance(value, SourceEntry):
            return value
        if isinstance(value, dict):
            return source_entry(value.get("source_id", value.get("id")),
                                value["X"], value["y"], value.get("model"),
                                value.get("order", 0))
        raise TypeError("Sources must be SourceEntry objects or explicit dictionaries")

    def fit(self, source_entries: Sequence[SourceEntry], target_X, target_y):
        target_X, target_y = _arrays(target_X, target_y)
        if target_X.shape[1] != self.dimension:
            raise ValueError("Target dimension differs from box dimension")
        entries = sorted([self._entry(e) for e in source_entries], key=lambda e: e.order)
        if len({e.source_id for e in entries}) != len(entries):
            raise ValueError("A selected source state may occur only once")
        X_parts, y_parts, task_parts = [], [], []
        self.source_audit, self.pseudo_datasets, self.selected_ids = [], [], []
        for entry in entries:
            X, y = _arrays(entry.X, entry.y)
            if X.shape[1] != self.dimension:
                raise ValueError("Source dimension differs from box dimension")
            digest = sha256(X.tobytes()+y.tobytes()).hexdigest()
            key = (entry.source_id, digest)
            if key not in self._source_cache:
                # A FiCSUM classifier/regressor is not the paper's independent GP.
                # Reuse only our independently MLE-trained GP with identical data.
                candidate = entry.model
                reusable = (isinstance(candidate, IndependentGP)
                            and hasattr(candidate, "X_raw")
                            and np.array_equal(candidate.X_raw, X)
                            and np.array_equal(candidate.y_raw, y))
                self._source_cache[key] = (candidate if reusable else
                    IndependentGP(config=self.config, seed=int(self.rng.integers(2**31-1)))
                    .fit(X, y))
            model = self._source_cache[key]
            if key not in self._warm_cache:
                self._warm_cache[key] = source_local_optima(
                    model, X, self.lower, self.upper, config=self.config, rng=self.rng)
            pseudo_X, pseudo_y, warm_audit = self._warm_cache[key]
            audit = {"source_id": str(entry.source_id), "order": int(entry.order),
                     "real_source_count": len(y), "source_data_sha256": digest,
                     "gp": model.fit_audit, "warm": warm_audit,
                     "pseudo_labels_are_target_truth": False}
            self.source_audit.append(audit)
            if not len(pseudo_X):
                audit["omitted_reason"] = "no_verified_source_GP_local_optimum"
                continue
            task = len(X_parts)
            X_parts.append(pseudo_X)
            y_parts.append(pseudo_y)
            task_parts.append(np.full(len(pseudo_X), task, int))
            self.selected_ids.append(entry.source_id)
            self.pseudo_datasets.append({"source_id": entry.source_id,
                                         "task_index": task,
                                         "X": pseudo_X.copy(), "y": pseudo_y.copy(),
                                         "label_kind": "source_GP_pseudo"})
        n_sources = len(X_parts)
        self.warm_candidates = (np.vstack(X_parts) if X_parts else
                                np.empty((0, self.dimension)))
        X_parts.append(target_X)
        y_parts.append(target_y)
        task_parts.append(np.full(len(target_X), n_sources, int))
        if self.model is None or self.model.n_tasks != n_sources+1:
            self.model = HierarchicalGP(n_sources+1, config=self.config,
                                        seed=int(self.rng.integers(2**31-1)))
        self.model.fit(np.vstack(X_parts), np.concatenate(y_parts), np.concatenate(task_parts))
        self.target_X, self.target_y = target_X.copy(), target_y.copy()
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Fit the transfer expert before prediction")
        return self.model.predict(X)

    def predict_with_gradients(self, X):
        if self.model is None:
            raise RuntimeError("Fit the transfer expert before prediction")
        return self.model.predict_with_gradients(X)

    def _acquisition(self, z, *, gradients=False):
        """A minimizable normalized LCB (or negative UCB for maximization)."""
        X = self.lower+np.atleast_2d(z)*self.span
        sign = 1.0 if self.config.minimize else -1.0
        if gradients:
            mean, variance, dmean, dvariance = self.predict_with_gradients(X)
        else:
            mean, variance = self.predict(X)
        std = np.sqrt(np.maximum(variance, 1e-24))
        scale = self.model.y_scale
        value = (sign*mean-self.config.beta*std)/scale
        if not gradients:
            return value
        derivative = (sign*dmean-self.config.beta*dvariance/(2*std[:, None]))
        return value, derivative*self.span/scale

    def nominate(self, count=4, initial=None):
        """Sec. III-D hybrid DE + adaptive local SGD, with no true evaluations.

        Returns up to count distinct members of its final population. Asking
        for multiple nominations is the explicit DOLEM batched integration;
        no pseudo observation or fantasy is added to the target task.
        """
        if self.model is None:
            raise RuntimeError("Fit before nomination")
        if int(count) < 1:
            raise ValueError("Request at least one candidate")
        cfg, n, d = self.config, self.config.population_size, self.dimension
        p = self.rng.random((n, d))
        seeds = self.warm_candidates
        if initial is not None:
            seeds = np.vstack([seeds, np.atleast_2d(initial)])
        if len(seeds):
            # A warm seed is an initial position, not a free real evaluation.
            scores = self._acquisition((seeds-self.lower)/self.span)
            chosen = np.argsort(scores, kind="stable")[:n]
            p[:len(chosen)] = np.clip((seeds[chosen]-self.lower)/self.span, 0, 1)
        kappa, traces = cfg.initial_kappa, []
        for generation in range(cfg.generations):
            children = np.empty_like(p)
            for i in range(n):
                pool = np.delete(np.arange(n), i)
                a, b, c = self.rng.choice(pool, 3, replace=False)
                mutant = np.clip(p[a]+cfg.de_f*(p[b]-p[c]), 0, 1)
                cross = self.rng.random(d) < cfg.de_cr
                cross[self.rng.integers(d)] = True
                children[i] = np.where(cross, mutant, p[i])
            joined = np.vstack([p, children])
            scores = self._acquisition(joined)
            archive = np.argsort(scores, kind="stable")[:min(kappa, 2*n)]
            old = joined[archive].copy()
            local = old.copy()
            # All local trajectories are independent, so vectorization is
            # algebraically the same as the paper's for-each-archive loop.
            for _ in range(cfg.local_steps):
                _, gradient = self._acquisition(local, gradients=True)
                local = np.clip(local-cfg.local_learning_rate*gradient, 0, 1)
            joined[archive] = local
            displacements = np.linalg.norm((local-old)*self.span, axis=1)
            before = kappa
            for displacement in displacements:
                kappa = (max(kappa-1, 1)
                         if displacement < cfg.local_displacement_threshold
                         else min(kappa+1, 2*n))
            scores = self._acquisition(joined)
            p = joined[np.argsort(scores, kind="stable")[:n]]
            traces.append({"generation": generation, "archive_size": len(archive),
                           "kappa_before": before, "kappa_after": kappa,
                           "local_displacements": displacements.tolist(),
                           "best_acquisition": float(np.min(scores))})
        points = self.lower+p*self.span
        values = self._acquisition(p)
        chosen = quality_distance_filter(points, values, int(count),
                                         1e-10*float(self.span.max()))
        self.search_audit.append({"requested": int(count), "returned": len(chosen),
                                  "generations": traces,
                                  "real_evaluations": 0})
        return points[chosen].copy()

    def audit(self):
        return {"component": "DETO_full_transfer_expert",
                "configuration": asdict(self.config),
                "selection": "FiCSUM_supplied_visible_states",
                "source_ids": [str(x) for x in self.selected_ids],
                "source_audit": self.source_audit,
                "hmogp": self.model.fit_audit if self.model is not None else None,
                "source_pseudo_count": sum(len(d["y"]) for d in self.pseudo_datasets),
                "target_real_count": len(self.target_y) if self.target_X is not None else 0,
                "searches": self.search_audit, "real_evaluations_inside_module": 0}

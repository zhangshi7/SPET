"""Causal conditional meta-prior development for MI-10.

The implementation deliberately separates the three data roles:

* seeds 11/12 fit descriptor scaling, clusters, and cluster priors;
* seed 13 selects K and the registered meta objective, then freezes OOD;
* seed 14 is not loaded until the configuration has been frozen.

Online routing accepts only causally visible observations.  Function and true
drift identifiers are retained on :class:`Task` solely for stratified reports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata
from sklearn.cluster import KMeans


EPS = 1e-8
TOP_K = 5
RBF_CENTERS = 8
GPR_LENGTHS = (0.5, 1.0, 2.0)
GPR_NOISES = (1e-4, 1e-3, 1e-2, 1e-1)
RBF_WIDTHS = (0.5, 1.0, 2.0)
RBF_RIDGES = (1e-4, 1e-2, 1e-1)
K_CANDIDATES = (2, 3, 4)
OBJECTIVES: Dict[str, Tuple[float, float, float]] = {
    "legacy": (1.00, 0.25, 0.25),
    "balanced": (0.50, 0.50, 0.50),
    "rank-heavy": (0.25, 0.75, 0.50),
    "topk-heavy": (0.25, 0.50, 1.00),
}
SELECTION_WEIGHTS = (0.25, 0.35, 0.40)
DESCRIPTOR_NAMES = (
    "y_skewness",
    "y_excess_kurtosis",
    "y_normalized_iqr",
    "y_lag1_autocorrelation",
    "xy_absolute_correlation_mean",
    "xy_absolute_correlation_max",
    "nearest_neighbor_normalized_y_difference",
    "arrival_adjacent_normalized_y_change",
)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def dump(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Task:
    seed: int
    function: int
    drift: int
    environment: int
    support_x: np.ndarray
    support_y: np.ndarray
    query_x: np.ndarray
    query_y: np.ndarray

    @property
    def key(self) -> str:
        return f"S{self.seed}:F{self.function}:D{self.drift}:E{self.environment}"


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)
    if len(a) < 2 or np.std(a) <= EPS or np.std(b) <= EPS:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else 0.0


def _robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, float).reshape(-1)
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = float((q75 - q25) / 1.349)
    if scale <= EPS:
        scale = float(np.std(values))
    return max(scale, 1e-6)


def causal_descriptor(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the frozen eight-dimensional descriptor from visible data only."""
    x = np.asarray(x, float)
    y = np.asarray(y, float).reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or len(y) < 2:
        raise ValueError("descriptor requires aligned 2-D X and at least two visible y values")
    mean = float(np.mean(y))
    std = max(float(np.std(y)), 1e-6)
    z = (y - mean) / std
    skew = float(np.mean(z ** 3))
    excess_kurtosis = float(np.mean(z ** 4) - 3.0)
    q25, q75 = np.quantile(y, [0.25, 0.75])
    normalized_iqr = float((q75 - q25) / std)
    lag1 = _safe_corr(y[:-1], y[1:])
    xy = np.asarray([abs(_safe_corr(x[:, j], y)) for j in range(x.shape[1])])

    xm = np.median(x, axis=0)
    xq25, xq75 = np.quantile(x, [0.25, 0.75], axis=0)
    xs = np.maximum((xq75 - xq25) / 1.349, 1e-6)
    xn = (x - xm) / xs
    d2 = np.sum((xn[:, None, :] - xn[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)
    nearest = np.argmin(d2, axis=1)
    y_scale = _robust_scale(y)
    nn_y_difference = float(np.mean(np.abs(y - y[nearest])) / y_scale)
    adjacent_change = float(np.mean(np.abs(np.diff(y))) / y_scale)
    result = np.asarray(
        [skew, excess_kurtosis, normalized_iqr, lag1, float(np.mean(xy)),
         float(np.max(xy)), nn_y_difference, adjacent_change],
        dtype=float,
    )
    if result.shape != (8,) or not np.all(np.isfinite(result)):
        raise FloatingPointError("non-finite causal task descriptor")
    return result


def fit_descriptor_scaler(descriptors: np.ndarray) -> Dict[str, List[float]]:
    descriptors = np.asarray(descriptors, float)
    median = np.median(descriptors, axis=0)
    q25, q75 = np.quantile(descriptors, [0.25, 0.75], axis=0)
    scale = q75 - q25
    fallback = np.std(descriptors, axis=0)
    scale = np.where(scale > EPS, scale, np.where(fallback > EPS, fallback, 1.0))
    return {"median": median.tolist(), "scale": scale.tolist(),
            "fit_split": "train_only", "fit_seeds": [11, 12]}


def transform_descriptors(descriptors: np.ndarray, scaler: Mapping) -> np.ndarray:
    return ((np.asarray(descriptors, float) - np.asarray(scaler["median"], float))
            / np.asarray(scaler["scale"], float))


def standardise(task: Task) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xm = np.mean(task.support_x, axis=0)
    xs = np.maximum(np.std(task.support_x, axis=0), 1e-6)
    ym = float(np.mean(task.support_y))
    ys = max(float(np.std(task.support_y)), 1e-6)
    return ((task.support_x - xm) / xs, (task.support_y - ym) / ys,
            (task.query_x - xm) / xs, (task.query_y - ym) / ys)


def collect_tasks(root: Path, trace_src: Path, seed: int,
                  functions: Sequence[int], drifts: Sequence[int]) -> List[Task]:
    """Collect window-0 support/window-1 query tasks for one data seed."""
    import causal_mlo_v1 as base

    adapter = base.SDDObenchAdapter(root, trace_src, int(seed), max_envs=5,
                                    n_samples=200, dim=10)
    tasks: List[Task] = []
    for function in functions:
        for drift in drifts:
            stream = adapter.stream(int(function), int(drift))
            by_environment: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
            while True:
                batch = stream.next_batch()
                if batch is None:
                    break
                # Offline meta-development may reveal a completed window.  It is
                # never used to route or predict an earlier window.
                labels = stream.reveal_labels(batch["batch_id"])
                by_environment.setdefault(int(batch["environment"]), []).append(
                    (np.asarray(batch["X"], float), np.asarray(labels, float)))
            for environment, windows in sorted(by_environment.items()):
                if len(windows) < 2:
                    continue
                support_x, support_y = windows[0]
                query_x, query_y = windows[1]
                tasks.append(Task(int(seed), int(function), int(drift), environment,
                                  support_x, support_y, query_x, query_y))
    return tasks


def kernel(a: np.ndarray, b: np.ndarray, length: float) -> np.ndarray:
    d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    return np.exp(-0.5 * d2 / max(float(length) ** 2, EPS))


def median_length(x: np.ndarray) -> float:
    d = np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
    nonzero = d[d > EPS]
    return float(np.median(nonzero)) if len(nonzero) else 1.0


def gpr_predict(sx: np.ndarray, sy: np.ndarray, qx: np.ndarray,
                length_multiplier: float, noise: float) -> np.ndarray:
    length = median_length(sx) * float(length_multiplier)
    covariance = kernel(sx, sx, length) + (float(noise) + 1e-6) * np.eye(len(sx))
    try:
        alpha = np.linalg.solve(covariance, sy)
    except np.linalg.LinAlgError:
        alpha = np.linalg.pinv(covariance) @ sy
    return kernel(qx, sx, length) @ alpha


def rbf_phi(x: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    d2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    return np.c_[np.exp(-0.5 * d2 / max(float(width) ** 2, EPS)), np.ones(len(x))]


def ridge_fit(phi: np.ndarray, y: np.ndarray, ridge: float,
              prior: np.ndarray | None = None) -> np.ndarray:
    prior = np.zeros(phi.shape[1]) if prior is None else np.asarray(prior, float)
    matrix = phi.T @ phi + float(ridge) * np.eye(phi.shape[1])
    rhs = phi.T @ y + float(ridge) * prior
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix) @ rhs


def learn_centers(tasks: Sequence[Task], seed: int) -> np.ndarray:
    pooled = np.vstack([standardise(task)[0][::5] for task in tasks])
    count = min(RBF_CENTERS, len(pooled))
    return KMeans(n_clusters=count, n_init=20, random_state=int(seed)).fit(pooled).cluster_centers_


def task_metrics(y: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, float).reshape(-1)
    prediction = np.asarray(prediction, float).reshape(-1)
    nrmse = float(np.sqrt(np.mean((prediction - y) ** 2)))
    spearman = _safe_corr(rankdata(y), rankdata(prediction))
    selected = np.argsort(prediction)[:min(TOP_K, len(prediction))]
    regret = ((float(np.min(y[selected])) - float(np.min(y)))
              / max(float(np.std(y)), EPS))
    return {"nrmse": nrmse, "spearman": spearman,
            "topk_regret": max(float(regret), 0.0)}


def weighted_loss(metrics: Mapping[str, float], weights: Sequence[float]) -> float:
    return (float(weights[0]) * float(metrics["nrmse"])
            + float(weights[1]) * (1.0 - float(metrics["spearman"]))
            + float(weights[2]) * float(metrics["topk_regret"]))


def aggregate(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty metric sequence")
    return {name: float(np.mean([float(row[name]) for row in rows]))
            for name in ("nrmse", "spearman", "topk_regret")}


def build_cluster_library(tasks: Sequence[Task], seed: int) -> Dict:
    """Fit every registered prior candidate using training tasks only."""
    if not tasks:
        raise ValueError("empty training cluster")
    gpr_curve = []
    for length in GPR_LENGTHS:
        for noise in GPR_NOISES:
            metrics = []
            for task in tasks:
                sx, sy, qx, qy = standardise(task)
                metrics.append(task_metrics(qy, gpr_predict(sx, sy, qx, length, noise)))
            gpr_curve.append({"length_multiplier": length, "noise": noise,
                              **aggregate(metrics)})

    centers = learn_centers(tasks, seed)
    base_width = median_length(centers)
    rbf_curve = []
    for multiplier in RBF_WIDTHS:
        width = max(base_width * multiplier, 0.1)
        for ridge in RBF_RIDGES:
            weight_prior = np.zeros(len(centers) + 1)
            for _ in range(3):
                adapted = []
                for task in tasks:
                    sx, sy, _, _ = standardise(task)
                    adapted.append(ridge_fit(rbf_phi(sx, centers, width), sy,
                                             ridge, weight_prior))
                weight_prior = np.mean(adapted, axis=0)
            metrics = []
            for task in tasks:
                sx, sy, qx, qy = standardise(task)
                weight = ridge_fit(rbf_phi(sx, centers, width), sy,
                                   ridge, weight_prior)
                metrics.append(task_metrics(qy, rbf_phi(qx, centers, width) @ weight))
            rbf_curve.append({"width_multiplier": multiplier, "width": width,
                              "ridge": ridge, "weight_prior": weight_prior.tolist(),
                              **aggregate(metrics)})
    return {"task_count": len(tasks), "rbf_centers": centers.tolist(),
            "gpr_curve": gpr_curve, "rbf_curve": rbf_curve}


def choose_cluster_prior(library: Mapping, objective_name: str) -> Dict:
    weights = OBJECTIVES[objective_name]
    gpr = min(library["gpr_curve"], key=lambda row: weighted_loss(row, weights))
    rbf = min(library["rbf_curve"], key=lambda row: weighted_loss(row, weights))
    return {
        "training_task_count": int(library["task_count"]),
        "gpr_prior": {key: gpr[key] for key in ("length_multiplier", "noise")},
        "rbf_centers": library["rbf_centers"],
        "rbf_prior": {key: rbf[key] for key in ("width_multiplier", "width", "ridge", "weight_prior")},
        "training_objective_values": {
            "gpr": weighted_loss(gpr, weights), "rbf": weighted_loss(rbf, weights)},
    }


def _legacy_prior(legacy: Mapping) -> Dict:
    return {
        "gpr_prior": {key: legacy["gpr_prior"][key]
                      for key in ("length_multiplier", "noise")},
        "rbf_centers": legacy["rbf_centers"],
        "rbf_prior": {key: legacy["rbf_prior"][key]
                      for key in ("width_multiplier", "width", "ridge", "weight_prior")},
    }


def route_descriptor(descriptor: np.ndarray, config: Mapping,
                     ood_threshold: float | None = None) -> Tuple[int, float, bool]:
    normalized = transform_descriptors(np.asarray(descriptor)[None, :],
                                       config["descriptor_scaler"])[0]
    centroids = np.asarray(config["centroids"], float)
    distances = np.sqrt(np.sum((centroids - normalized[None, :]) ** 2, axis=1))
    cluster = int(np.argmin(distances))
    distance = float(distances[cluster])
    fallback = ood_threshold is not None and distance > float(ood_threshold)
    return cluster, distance, bool(fallback)


def select_prior_from_visible(visible_x: np.ndarray, visible_y: np.ndarray,
                              bundle: Mapping, *, natural_count: int | None = None,
                              active_count: int = 0, recent_limit: int = 50) -> Tuple[Dict, Dict]:
    """Select a prior using the most recent visible observations only.

    There is intentionally no function or drift-type argument in this API.
    """
    x = np.asarray(visible_x, float)[-int(recent_limit):]
    y = np.asarray(visible_y, float).reshape(-1)[-int(recent_limit):]
    descriptor = causal_descriptor(x, y)
    cluster, distance, fallback = route_descriptor(
        descriptor, bundle, float(bundle["ood"]["distance_threshold"]))
    prior = (bundle["global_fallback_prior"] if fallback
             else bundle["clusters"][cluster])
    natural = len(y) - int(active_count) if natural_count is None else int(natural_count)
    telemetry = {
        "meta_prior_cluster": None if fallback else cluster,
        "meta_prior_distance": distance,
        "meta_prior_fallback": fallback,
        "descriptor_samples": len(y),
        "descriptor_natural_samples": natural,
        "descriptor_active_samples": int(active_count),
        "descriptor_source": "causally_visible_recent_observations",
        "future_values_used": False,
    }
    return dict(prior), telemetry


def predict_with_prior(task: Task, prior: Mapping) -> Dict[str, Dict[str, float]]:
    sx, sy, qx, qy = standardise(task)
    gpr = prior["gpr_prior"]
    gp_prediction = gpr_predict(sx, sy, qx, gpr["length_multiplier"], gpr["noise"])
    centers = np.asarray(prior["rbf_centers"], float)
    rbf = prior["rbf_prior"]
    support_phi = rbf_phi(sx, centers, float(rbf["width"]))
    weight = ridge_fit(support_phi, sy, float(rbf["ridge"]),
                       np.asarray(rbf["weight_prior"], float))
    rbf_prediction = rbf_phi(qx, centers, float(rbf["width"])) @ weight
    return {"GPR": task_metrics(qy, gp_prediction),
            "RBF": task_metrics(qy, rbf_prediction)}


def evaluate_config(tasks: Sequence[Task], config: Mapping,
                    global_prior: Mapping, *, ood_threshold: float | None) -> Dict:
    rows, distances, fallback_by_task = [], [], []
    for task in tasks:
        descriptor = causal_descriptor(task.support_x, task.support_y)
        cluster, distance, fallback = route_descriptor(descriptor, config, ood_threshold)
        prior = global_prior if fallback else config["clusters"][cluster]
        distances.append(distance)
        fallback_by_task.append(fallback)
        for family, metrics in predict_with_prior(task, prior).items():
            rows.append({"task": task.key, "seed": task.seed, "function": task.function,
                         "drift": task.drift, "environment": task.environment,
                         "family": family, "cluster": cluster,
                         "meta_prior_fallback": fallback, "descriptor_distance": distance,
                         **metrics})
    overall = aggregate(rows)
    by_family = {family: aggregate([row for row in rows if row["family"] == family])
                 for family in ("GPR", "RBF")}
    return {"metrics": overall, "selection_metric": weighted_loss(overall, SELECTION_WEIGHTS),
            "by_family": by_family, "rows": rows, "distances": distances,
            "ood_fallback_fraction": float(np.mean(fallback_by_task)) if tasks else 0.0}


def evaluate_global(tasks: Sequence[Task], global_prior: Mapping) -> Dict:
    rows = []
    for task in tasks:
        for family, metrics in predict_with_prior(task, global_prior).items():
            rows.append({"task": task.key, "seed": task.seed, "function": task.function,
                         "drift": task.drift, "environment": task.environment,
                         "family": family, **metrics})
    overall = aggregate(rows)
    return {"metrics": overall, "selection_metric": weighted_loss(overall, SELECTION_WEIGHTS),
            "by_family": {family: aggregate([r for r in rows if r["family"] == family])
                          for family in ("GPR", "RBF")}, "rows": rows}


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_delta(left: Mapping, right: Mapping) -> Dict[str, float]:
    return {name: float(left[name]) - float(right[name])
            for name in ("nrmse", "spearman", "topk_regret")}

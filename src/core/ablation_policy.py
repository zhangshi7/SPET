"""Historical source-model fitting and candidate extraction."""
from hashlib import sha256
import numpy as np
from deto_transfer import DETOTransferExpert, HierarchicalGP, IndependentGP, _arrays, source_local_optima
def array_digest(X, y=None):
    """Canonical float64/C-order hash, with X followed by y when supplied."""
    data = np.asarray(X, dtype=np.float64).tobytes(order="C")
    if y is not None:
        data += np.asarray(y, dtype=np.float64).tobytes(order="C")
    return sha256(data).hexdigest()

class WarmOnlyTransferExpert(DETOTransferExpert):
    """Original source GP/optimum extraction; posterior uses current truth only.

    The source extraction below is line-for-line the mathematical operation
    sequence in the frozen DETOTransferExpert.fit. It deliberately retains the
    independent-GP MLE restarts, all starts, KKT acceptance, quality/distance
    filtering and source caches. Its only removal is source pseudo observations
    from the final posterior. Extracted positions remain in warm_candidates,
    which the unchanged nominate() uses as DE initial seeds.
    """

    def fit(self, source_entries, target_X, target_y):
        target_X, target_y = _arrays(target_X, target_y)
        if target_X.shape[1] != self.dimension:
            raise ValueError("Target dimension differs from box dimension")
        entries = sorted([self._entry(e) for e in source_entries], key=lambda e: e.order)
        if len({e.source_id for e in entries}) != len(entries):
            raise ValueError("A selected source state may occur only once")
        warm_parts = []
        self.source_audit, self.pseudo_datasets, self.selected_ids = [], [], []
        self.extracted_source_pseudo = []
        for entry in entries:
            X, y = _arrays(entry.X, entry.y)
            if X.shape[1] != self.dimension:
                raise ValueError("Source dimension differs from box dimension")
            digest = sha256(X.tobytes() + y.tobytes()).hexdigest()
            key = (entry.source_id, digest)
            if key not in self._source_cache:
                candidate = entry.model
                reusable = (
                    isinstance(candidate, IndependentGP)
                    and hasattr(candidate, "X_raw")
                    and np.array_equal(candidate.X_raw, X)
                    and np.array_equal(candidate.y_raw, y)
                )
                self._source_cache[key] = (
                    candidate if reusable else
                    IndependentGP(config=self.config,
                                  seed=int(self.rng.integers(2**31 - 1))).fit(X, y)
                )
            model = self._source_cache[key]
            if key not in self._warm_cache:
                self._warm_cache[key] = source_local_optima(
                    model, X, self.lower, self.upper, config=self.config, rng=self.rng)
            pseudo_X, pseudo_y, warm_audit = self._warm_cache[key]
            audit = {
                "source_id": str(entry.source_id), "order": int(entry.order),
                "real_source_count": len(y), "source_data_sha256": digest,
                "gp": model.fit_audit, "warm": warm_audit,
                "pseudo_labels_are_target_truth": False,
                "used_in_target_posterior": False,
            }
            self.source_audit.append(audit)
            if not len(pseudo_X):
                audit["omitted_reason"] = "no_verified_source_GP_local_optimum"
                continue
            warm_parts.append(pseudo_X)
            self.selected_ids.append(entry.source_id)
            self.extracted_source_pseudo.append({
                "source_id": entry.source_id, "X": pseudo_X.copy(),
                "y": pseudo_y.copy(), "label_kind": "source_GP_pseudo_unused",
            })
        self.warm_candidates = (
            np.vstack(warm_parts) if warm_parts else np.empty((0, self.dimension)))
        if self.model is None or self.model.n_tasks != 1:
            self.model = HierarchicalGP(
                1, config=self.config, seed=int(self.rng.integers(2**31 - 1)))
        # Identical target-only call layout to DETOTransferExpert.fit([], X, y).
        self.model.fit(target_X, target_y, np.zeros(len(target_X), dtype=int))
        self.target_X, self.target_y = target_X.copy(), target_y.copy()
        return self

    def audit(self):
        result = super().audit()
        result.update({
            "component": "historical_warm_positions_current_only_GP",
            "historical_pseudo_observations_in_posterior": False,
            "extracted_warm_count": len(self.warm_candidates),
            "extracted_warm_sha256": array_digest(self.warm_candidates),
        })
        return result

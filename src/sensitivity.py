"""SPET sensitivity variant with a configurable historical-source count."""
import _paths
from position_policy import PositionPolicy,original
import numpy as np
SourceEntry=original.SourceEntry
class SPET(PositionPolicy):
    def __init__(self,arm,lower,upper,seed,*,alpha,history_k):
        super().__init__(arm,lower,upper,seed,alpha=alpha)
        assert history_k in (1,3,5)
        self.history_k=int(history_k)
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
            if len(selected) == self.history_k:
                break
        return sorted(selected, key=lambda entry: entry.order), retrieval

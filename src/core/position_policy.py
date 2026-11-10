"""Position-only R2 and an equal-quota historical-position source intervention.

The old source-GP extraction is retained in both arms. Historical pseudo labels
never enter the current posterior. Only the two proposal entrances read the
temporary warm positions; the fitted model, caches, and original field survive.
"""
from __future__ import annotations

import os
import sys
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import pickle
import random

import numpy as np

from isolation_policy import IsolationPolicy, original, array_digest, data_digest
from ablation_policy import WarmOnlyTransferExpert

ARMS = ("GP_POSITIONS", "DIRECT_POSITIONS")


def object_digest(value):
    return sha256(pickle.dumps(value, protocol=5)).hexdigest()


def json_digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def rng_digest(policy):
    return object_digest((random.getstate(), np.random.get_state(),
                          policy.rng.bit_generator.state, policy.transfer.rng.bit_generator.state))


def position_state(transfer):
    model = transfer.model
    target_X = getattr(transfer, "target_X", None)
    target_y = getattr(transfer, "target_y", None)
    return {
        "model_pickle_sha256": object_digest(model),
        "model_n_tasks": int(model.n_tasks) if model is not None else None,
        "model_fit_audit": deepcopy(getattr(model, "fit_audit", {})),
        "target_n": len(target_y) if target_y is not None else 0,
        "target_data_sha256": data_digest(target_X, target_y) if target_y is not None else None,
        "pseudo_datasets_sha256": object_digest(transfer.pseudo_datasets),
        "pseudo_dataset_count": len(transfer.pseudo_datasets),
        "extracted_source_pseudo_sha256": object_digest(getattr(transfer, "extracted_source_pseudo", [])),
    }


def source_choices(source_entries, extracted, lower, upper, arm, *, minimize=True):
    """Pure, RNG-free equal-count selection; observed points are never clipped."""
    if arm not in ARMS:
        raise ValueError("Unknown position arm")
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    d = len(lower)
    extracted_by_id = {int(row["source_id"]): row for row in extracted}
    if len(extracted_by_id) != len(extracted):
        raise ValueError("Duplicate extracted source")
    sources = sorted(source_entries, key=lambda row: row.order)
    source_ids = {int(row.source_id) for row in sources}
    if len(source_ids) != len(sources) or not set(extracted_by_id).issubset(source_ids):
        raise ValueError("Source identities differ")
    rows, pieces = [], []
    for source in sources:
        sid = int(source.source_id)
        X, y = np.asarray(source.X, float), np.asarray(source.y, float).reshape(-1)
        if X.ndim != 2 or X.shape != (len(y), d) or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Invalid real source data")
        extracted_row = extracted_by_id.get(sid)
        gp = np.asarray(extracted_row["X"], float).reshape(-1, d) if extracted_row is not None else np.empty((0, d))
        gp_y = np.asarray(extracted_row["y"], float).reshape(-1) if extracted_row is not None else np.empty(0)
        if len(gp_y) != len(gp) or not np.isfinite(gp).all() or not np.isfinite(gp_y).all():
            raise ValueError("Invalid extracted GP positions")
        if np.any(gp < lower) or np.any(gp > upper):
            raise ValueError("Original GP optimum outside the active bounds")
        inside = np.flatnonzero(np.all((X >= lower) & (X <= upper), axis=1))
        outside = np.flatnonzero(~np.all((X >= lower) & (X <= upper), axis=1))
        if len(inside):
            _, first = np.unique(np.round(X[inside], 11), axis=0, return_index=True)
            unique = inside[np.sort(first)]
        else:
            unique = np.empty(0, dtype=int)
        order = np.argsort(y[unique] if minimize else -y[unique], kind="stable")
        ranked = unique[order]
        q = min(len(gp), len(ranked))
        gp_selected, direct_selected = gp[:q].copy(), X[ranked[:q]].copy()
        selected = gp_selected if arm == "GP_POSITIONS" else direct_selected
        row = {"source_id": sid, "order": int(source.order),
               "source_X": X.tolist(), "source_y": y.tolist(), "source_data_sha256": data_digest(X, y),
               "gp_positions": gp.tolist(), "gp_pseudo_y_unused": gp_y.tolist(),
               "gp_positions_sha256": array_digest(gp),
               "in_bounds_indices": inside.tolist(), "out_of_bounds_indices": outside.tolist(),
               "unique_real_indices": unique.tolist(), "sorted_real_indices": ranked.tolist(),
               "duplicate_in_bounds_indices": sorted(set(inside.tolist()) - set(unique.tolist())),
               "n_gp": len(gp), "m_real": len(ranked), "q": q,
               "capacity_shortfall": len(ranked) < len(gp),
               "gp_selected": gp_selected.tolist(), "direct_selected": direct_selected.tolist(),
               "selected_positions": selected.tolist(), "selected_sha256": array_digest(selected),
               "selected_real_indices": ranked[:q].tolist() if arm == "DIRECT_POSITIONS" else [],
               "position_sets_differ": (set(map(tuple, np.round(gp_selected, 11))) !=
                                        set(map(tuple, np.round(direct_selected, 11)))),
               "position_arrays_differ": not np.array_equal(gp_selected, direct_selected),
               "bounds_handling": "filter_original_observed_rows_never_clip",
               "tie_break": "stable_original_source_row_order", "minimum_is_good": bool(minimize)}
        rows.append(row)
        pieces.append(selected)
    joined = np.vstack(pieces) if pieces else np.empty((0, d))
    return rows, joined


class PositionPolicy(IsolationPolicy):
    def __init__(self, arm, lower, upper, seed, *, alpha):
        if arm not in ARMS:
            raise ValueError("Unknown registered position-source arm")
        if not np.isfinite(float(alpha)) or not 0 <= float(alpha) <= 1:
            raise ValueError("alpha must be an explicit finite value in [0,1]")
        super().__init__("DIRECT_RECENT", lower, upper, seed)
        self.position_arm = arm
        self.fixed_alpha = float(alpha)
        # Exact pre-existing source-GP/warm extraction and current-only posterior.
        self.transfer = WarmOnlyTransferExpert(lower, upper, seed=seed + 17011,
                                               config=self.transfer.config)
        self._frozen_transfer_loss = None
        self._frozen_transfer_weight = None
        self._position_alarms = []
        self._position_alarm = None

    def _weights(self, names):
        live = {name: float(self.active_losses.get(name, self.losses.get(name, .75))) for name in names}
        routing = dict(live)
        enabled = self._transfer_name is not None and self._transfer_name in names
        if enabled:
            name = self._transfer_name
            if self._frozen_transfer_loss is None:
                self._frozen_transfer_loss = live[name]
            routing[name] = self._frozen_transfer_loss
            had_active, saved = name in self.active_losses, self.active_losses.get(name)
            self.active_losses[name] = self._frozen_transfer_loss
            try:
                raw_weights = original.FEMS._weights(self, names)
            finally:
                if had_active:
                    self.active_losses[name] = saved
                else:
                    self.active_losses.pop(name, None)
            index = names.index(name)
            other = np.arange(len(names)) != index
            if not np.any(other):
                raise AssertionError("Original DOLEM experts must remain present")
            weights = raw_weights.copy()
            if float(raw_weights[index]) != self.fixed_alpha:
                weights[other] *= (1 - self.fixed_alpha) / weights[other].sum()
                weights[index] = self.fixed_alpha
            self._frozen_transfer_weight = self.fixed_alpha
            self._pending_weights_state["weights_before_query"] = dict(zip(names, weights.tolist()))
            self._pending_weights_state["active_losses_before_query"] = {
                name: float(value) for name, value in self.active_losses.items()}
        else:
            raw_weights = original.FEMS._weights(self, names)
            weights = raw_weights.copy()
        record = {"round": len(self._position_alarm["routing"]) if self._position_alarm is not None else 0,
                  "names": list(names), "transfer_role": self._transfer_name,
                  "live_losses": live, "routing_losses": routing,
                  "raw_weights": dict(zip(names, raw_weights.tolist())),
                  "final_weights": dict(zip(names, weights.tolist())),
                  "fixed_transfer_credit": bool(enabled), "specified_transfer_alpha": self.fixed_alpha,
                  "frozen_transfer_loss": self._frozen_transfer_loss,
                  "frozen_transfer_weight": self._frozen_transfer_weight,
                  "original_role_count_in_share_floor": len(names),
                  "measurement_point": "before_current_batch_true_feedback",
                  "original_conditional_weights_preserved": True,
                  "transfer_diagnostic_loss_and_EMA_remain_live": True,
                  "live_transfer_loss_can_affect_original_conditional_weights": False}
        if self._position_alarm is not None:
            self._position_alarm["routing"].append(record)
        return weights

    def _candidates(self, train_z, train_y, radius):
        if not self._selected_sources:
            return original.FusionPolicy._candidates(self, train_z, train_y, radius)
        if self._position_alarm is None:
            raise AssertionError("Position proposals must occur inside an audited search")
        before = position_state(self.transfer)
        if before["model_n_tasks"] != 1 or before["pseudo_dataset_count"] != 0:
            raise AssertionError("Historical labels entered the target posterior")
        original_warm = self.transfer.warm_candidates
        rng_before = rng_digest(self)
        sources, selected = source_choices(self._selected_sources, self.transfer.extracted_source_pseudo,
                                            self.lower, self.upper, self.position_arm,
                                            minimize=self.transfer.config.minimize)
        rng_after = rng_digest(self)
        if rng_before != rng_after:
            raise AssertionError("Position-source selection consumed an optimizer RNG")
        gp_full = np.vstack([np.asarray(row["gp_positions"], float).reshape(-1, self.dimension) for row in sources])
        if not np.array_equal(gp_full, original_warm):
            raise AssertionError("Extracted GP positions and original warm field differ")
        source_hash = json_digest(sources)
        if self._position_alarm["sources"] is None:
            self._position_alarm["sources"] = sources
            self._position_alarm["sources_sha256"] = source_hash
        elif source_hash != self._position_alarm["sources_sha256"]:
            raise AssertionError("Historical source data or extracted positions changed within an alarm")
        row = {"round": len(self._position_alarm["rounds"]),
               "source_ids": [r["source_id"] for r in sources],
               "selected_n": len(selected), "selected_sha256": array_digest(selected),
               "gp_full_warm_n": len(original_warm), "gp_full_warm_sha256": array_digest(original_warm),
               "sources_sha256": source_hash, "before": before,
               "selection_rng_before": rng_before, "selection_rng_after": rng_after,
               "selection_did_not_consume_rng": True, "nominate_calls": []}
        self._position_alarm["rounds"].append(row)
        original_nominate = self.transfer.nominate
        had_nominate = "nominate" in self.transfer.__dict__
        saved_nominate = self.transfer.__dict__.get("nominate")

        def audited_nominate(count=4, initial=None):
            seeds = self.transfer.warm_candidates
            init = np.asarray(initial, float).reshape(-1, self.dimension) if initial is not None else np.empty((0, self.dimension))
            call = {"count": int(count), "warm_n": len(seeds), "warm_sha256": array_digest(seeds),
                    "initial_n": len(init), "initial_positions": init.tolist(), "initial_sha256": array_digest(init),
                    "total_seed_n": len(seeds) + len(init), "total_seed_sha256": array_digest(np.vstack([seeds, init]))}
            row["nominate_calls"].append(call)
            result = original_nominate(count=count, initial=initial)
            call.update(nominated_positions=np.asarray(result, float).tolist(), nominated_sha256=array_digest(result))
            return result

        try:
            self.transfer.warm_candidates = selected
            self.transfer.nominate = audited_nominate
            result = original.FusionPolicy._candidates(self, train_z, train_y, radius)
            row["candidate_counts"] = deepcopy(self._last_candidates)
            row["combined_candidate_n"] = len(result)
            row["combined_candidate_sha256"] = array_digest(result)
            row["direct_candidate_entrance_warm_n"] = len(self.transfer.warm_candidates)
            row["direct_candidate_entrance_warm_sha256"] = array_digest(self.transfer.warm_candidates)
            if len(row["nominate_calls"]) != 1 or row["nominate_calls"][0]["warm_sha256"] != row["selected_sha256"]:
                raise AssertionError("The historical DE seed entrance differs from the selected positions")
            return result
        finally:
            self.transfer.warm_candidates = original_warm
            if had_nominate:
                self.transfer.nominate = saved_nominate
            else:
                self.transfer.__dict__.pop("nominate", None)
            row["after"] = position_state(self.transfer)
            row["restored_original_warm_object"] = self.transfer.warm_candidates is original_warm
            row["restored_warm_sha256"] = array_digest(self.transfer.warm_candidates)
            row["restored_nominate_instance_state"] = ("nominate" in self.transfer.__dict__) == had_nominate
            row["posterior_and_pseudo_state_unchanged"] = row["before"] == row["after"]
            if not row["posterior_and_pseudo_state_unchanged"]:
                raise AssertionError("Position proposals modified target predictions or extracted source data")

    def search(self, objective_callable, generations=8, batch_size=4):
        if (generations, batch_size) != (8, 4):
            raise ValueError("Position-source protocol requires exactly 8 x 4 true feedback")
        if self._position_alarm is not None:
            raise AssertionError("Nested search is unsupported")
        self._frozen_transfer_loss = None
        self._frozen_transfer_weight = None
        alarm = {"search_index": self.search_count, "seen": self._visible_count,
                 "position_arm": self.position_arm, "alpha": self.fixed_alpha,
                 "sources": None, "sources_sha256": None, "rounds": [], "routing": [], "status": "RUNNING"}
        self._position_alarms.append(alarm)
        self._position_alarm = alarm
        try:
            result = super().search(objective_callable, generations, batch_size)
            alarm["sources"] = alarm["sources"] or []
            alarm["sources_sha256"] = alarm["sources_sha256"] or json_digest([])
            alarm.update(status="COMPLETE", real_fe=int(result["real_queries"]),
                         transfer_role=self._transfer_name, current_memory_id=self.memory.current_id)
            if alarm["real_fe"] != 32 or len(alarm["routing"]) != 8:
                raise AssertionError("Incomplete shared-budget search")
            result["mechanism_audit"]["position_source"] = {
                "position_arm": self.position_arm, "alpha": self.fixed_alpha,
                "history_model_posterior_transfer": False, "current_only_transfer_GP_retained": True,
                "source_record_index": len(self._position_alarms) - 1,
                "source_count": len(alarm["sources"]), "candidate_rounds": len(alarm["rounds"]),
                "matched_warm_n": sum(r["q"] for r in alarm["sources"]),
                "capacity_shortfall_sources": sum(r["capacity_shortfall"] for r in alarm["sources"]),
                "source_position_sets_differ": sum(r["position_sets_differ"] for r in alarm["sources"]),
                "same_nominate_and_direct_candidate_entrances": True}
            return result
        except BaseException:
            alarm["status"] = "FAILED"
            raise
        finally:
            self._position_alarm = None

    def export_position_audit(self):
        return deepcopy({"schema_version": 1, "algorithm": self.algorithm,
                         "position_arm": self.position_arm, "alpha": self.fixed_alpha,
                         "target_posterior": "current_truth_only", "historical_model_posterior_transfer": False,
                         "quantity_rule": "per_source_q=min(extracted_GP_count,unique_in_bound_observed_count)",
                         "direct_rule": "stable_true_y_rank_after_in_bounds_round11_first_row_dedup",
                         "source_GP_computation_retained_in_both_arms": True,
                         "fixed_share_scope": "transfer_only_original_DOLEM_conditional_weights_remain_adaptive",
                         "alarms": self._position_alarms})

    def mechanism_audit(self):
        result = super().mechanism_audit()
        result.update({"method": "R2_position_only_fixed_transfer_share",
                       "position_arm": self.position_arm, "fixed_alpha": self.fixed_alpha,
                       "history_model_posterior_transfer": False,
                       "historical_GP_extraction_retained": True,
                       "original_DOLEM_internal_adaptive_credit_retained": True,
                       "position_source_audit_alarms": len(self._position_alarms)})
        return result


def make_position(arm, lower, upper, seed, *, alpha):
    return PositionPolicy(arm, lower, upper, seed, alpha=alpha)

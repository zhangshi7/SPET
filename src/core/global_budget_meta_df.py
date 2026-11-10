"""Simplified experimental candidate: preserved DF + global meta prior + GP cap128.

This is the policy equivalent of the independently run global_cap128 control.
It preserves the original search, feedback, local fits and observation updates.
The original conditional DF code and its experiment outputs are not modified.
"""
from pathlib import Path
import sys
import time
import numpy as np

from decision_feedback import DecisionFeedbackPolicy
from meta_adaptation import build_with_cap


class GlobalBudgetMetaDF(DecisionFeedbackPolicy):
    """Use the inherited source-trained global prior, with 128 GP fitting rows."""
    gpr_fit_cap = 128

    def _build_pool(self):
        started = time.perf_counter()
        self.models, audit = build_with_cap(
            np.c_[self.passive_x, self.passive_y],
            self.bundle['global_fallback_prior'], self.seed, self.gpr_fit_cap)
        self.route_records.append({'prior_index':0,'gpr_training_cap':self.gpr_fit_cap,
                                   'mode':'global_cap128','target_offline_labels_used':False})
        self.pool_records.append(audit)
        self.fit_seconds += time.perf_counter()-started

"""Public entry point for the SPET numerical implementation."""
from pathlib import Path
from copy import deepcopy
from threading import RLock
import sys
import _paths
from position_policy import PositionPolicy,original
_LOCK=RLock()

def create_spet(lower,upper,seed,alpha=0.50,prior=None):
    """Create SPET; a supplied application prior is copied during construction."""
    with _LOCK:
        if prior is None:return PositionPolicy('GP_POSITIONS',lower,upper,seed,alpha=alpha)
        old=original.load_prior
        try:
            original.load_prior=lambda:deepcopy(prior)
            return PositionPolicy('GP_POSITIONS',lower,upper,seed,alpha=alpha)
        finally:original.load_prior=old

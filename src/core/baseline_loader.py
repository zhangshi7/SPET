"""Load the surrogate policy and its initialization parameters."""
import json
from pathlib import Path

CORE = Path(__file__).resolve().parent
ROOT = CORE.parents[1]
PRIOR = ROOT / 'data/prior.json'

def configure_policy_paths():
    return CORE

def load_prior():
    return json.loads(PRIOR.read_text(encoding='utf-8'))

def dolem_class():
    from fems_policy import FEMS
    return FEMS

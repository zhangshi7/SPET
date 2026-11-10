"""Resolve source locations relative to this repository."""
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parent
for directory in (SOURCE / 'core', SOURCE / 'cases'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

"""Benchmark objective replay from the supplied environment states."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

class LabelConsistentObjective:
    """Benchmark state replay consistent with the frozen passive labels.

    Frozen search bounds remain [-5,5]. The function-specific transformation
    and clipping bounds reproduce the function that generated the labels.
    The difference from the legacy oracle is isolated to the evaluator.
    """

    def __init__(self, snapshot):
        self.states = snapshot["backend_states"]
        self.dimension = int(snapshot["dimension"])
        directory = Path(__file__).resolve().parent
        self.objectives = _module(
            "_fdd_label_consistent_objectives", directory / "object_function.py")
        transform = _module(
            "_fdd_label_consistent_transform", directory / "transformation_function.py")
        config = _module(
            "_fdd_label_consistent_config", directory / "bench_config.py")
        self.transform = transform.transform_func
        self.config = config.SDDObenchConfig()

    def get_bound(self, function):
        """Search box of the frozen DOLEM protocol, not transform bounds."""
        return -5.0, 5.0

    def get_transform_bound(self, function):
        return self.objectives.get_bound(int(function))

    def evaluate(self, function, drift, environment, x):
        function = int(function)
        state = self.states[f"{function}:{int(drift)}:{int(environment)}"]
        x = np.atleast_2d(np.asarray(x, dtype=float))
        fn = self.objectives.get_objective_func(function)
        if function <= 3:
            h, w, c = [np.asarray(a, dtype=float) for a in state["peak_info"]]
            return np.asarray(fn(x, h, w, c), dtype=float).reshape(-1)
        lb, ub = self.get_transform_bound(function)
        transformed = self.transform(
            x, lb, ub, float(state["delta_t"]), severity=self.config.severity_x)
        return np.asarray(fn(np.clip(transformed, lb, ub)), dtype=float).reshape(-1)

    def optimum(self, function, drift, environment):
        """Original benchmark reference; not necessarily attainable in [-5,5]."""
        state = self.states[f"{int(function)}:{int(drift)}:{int(environment)}"]
        if int(function) <= 3:
            return float(np.min(np.asarray(state["peak_info"][0], dtype=float)))
        return 0.0

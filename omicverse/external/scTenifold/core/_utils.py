from functools import wraps, partial
import time
from typing import Callable, Optional

import numpy as np


__all__ = ["cal_fdr", "timer"]


def cal_fdr(p_vals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment for an array of p-values.

    Parameters
    ----------
    p_vals
        Raw p-values.

    Returns
    -------
    FDR-adjusted values, capped at 1.
    """
    p_vals = np.asarray(p_vals, dtype=float)
    order = np.argsort(p_vals)
    ranked = p_vals[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted[adjusted > 1] = 1
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def timer(func: Optional[Callable] = None) -> Callable:
    """Decorator that prints the wall-clock runtime of ``func``.

    A ``verbosity`` keyword is consumed from the call site (default 1);
    set to 0 to silence.
    """
    if func is None:
        return partial(timer)

    @wraps(func)
    def _counter(*args, **kwargs):
        if kwargs.get("verbosity") is not None:
            verbosity = kwargs.pop("verbosity")
        else:
            verbosity = 1
        if verbosity >= 1:
            start = time.perf_counter()
        sol = func(*args, **kwargs)
        if verbosity >= 1:
            end = time.perf_counter()
            print(func.__name__, " processing time: ", str(end - start))
        return sol
    return _counter

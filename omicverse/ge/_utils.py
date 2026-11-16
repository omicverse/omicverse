"""Small numpy-only helpers shared by ``ov.ge`` (no ``torch`` import)."""
from __future__ import annotations

import numpy as np


def gini(array) -> float:
    """Gini coefficient of a 1-D array (``simba.tools._utils._gini``)."""
    array = np.asarray(array, dtype=float).flatten()
    if array.min() < 0:
        array = array - array.min()
    array = array + 1e-7
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return float((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def plot_history(history: dict, keys=None, ax=None):
    """Quick per-epoch training-curve plot (``train_loss``/``eval_loss``/
    ``eval_mrr`` by default), matplotlib axis convention as ``ov.vi.plot_history``."""
    import matplotlib.pyplot as plt

    keys = keys or [k for k in ("train_loss", "eval_loss", "eval_mrr") if k in history]
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3.5))
    for k in keys:
        vals = history.get(k, [])
        if len(vals):
            ax.plot(np.arange(len(vals)), vals, label=k)
    ax.set_xlabel("epoch")
    ax.legend()
    return ax


__all__ = ["gini", "plot_history"]

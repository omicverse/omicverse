r"""Plotting for single-cell metabolism results (:class:`ov.single.Metabolism`).

:func:`metabolism_heatmap` summarises the ``adata.obsm['X_metabolism']``
matrix written by any :class:`ov.single.Metabolism` backend
(scMetabolism pathway activity, Compass reaction flux, scFEA module
flux) as a group-mean heatmap.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from anndata import AnnData


def metabolism_heatmap(
    adata: AnnData,
    *,
    groupby: str,
    features: Union[str, Sequence[str], None] = None,
    n_features: Optional[int] = 25,
    decorrelate: bool = True,
    standard_scale: Optional[str] = "var",
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] = (8.0, 6.0),
    ax=None,
):
    r"""Group-mean heatmap of a single-cell metabolism result.

    Reads ``adata.obsm['X_metabolism']`` and ``adata.uns['metabolism']``
    (written by :meth:`ov.single.Metabolism.run`), averages the metabolic
    features within each ``adata.obs[groupby]`` group, and draws a
    features × groups heatmap.

    Parameters
    ----------
    adata : AnnData
        Must carry ``obsm['X_metabolism']`` and ``uns['metabolism']``.
    groupby : str
        ``adata.obs`` column to average within (e.g. a cell-type column).
    features : str or list of str or None
        Restrict to these metabolic features. ``None`` uses all (then
        optionally trimmed to the ``n_features`` most variable).
    n_features : int or None
        Keep only the top-``n_features`` features by cross-group variance.
        ``None`` keeps all. Ignored when ``features`` is given.
    standard_scale : {'var', 'group', None}
        Z-score each feature (``'var'``), each group (``'group'``), or
        leave raw (``None``).
    cmap : str
        Matplotlib colormap.
    figsize : tuple
        Figure size, used when ``ax`` is None.
    ax : matplotlib.axes.Axes or None
        Draw into an existing axes instead of a new figure.

    Returns
    -------
    matplotlib.axes.Axes
        The heatmap axes.
    """
    import matplotlib.pyplot as plt

    if "metabolism" not in adata.uns or "X_metabolism" not in adata.obsm:
        raise ValueError(
            "adata has no metabolism result — run ov.single.Metabolism(...).run() first."
        )
    if groupby not in adata.obs.columns:
        raise ValueError(f"groupby {groupby!r} not in adata.obs")

    names = list(adata.uns["metabolism"]["features"])
    mat = pd.DataFrame(
        np.asarray(adata.obsm["X_metabolism"]),
        index=adata.obs_names,
        columns=names,
    )
    if features is not None:
        if isinstance(features, str):
            features = [features]
        mat = mat[list(features)]

    # group means → features × groups
    grouped = mat.groupby(adata.obs[groupby].to_numpy(), observed=True).mean().T

    if features is None and n_features is not None and grouped.shape[0] > n_features:
        ranked = grouped.var(axis=1).sort_values(ascending=False).index
        if decorrelate:
            # Rank by cross-group variance, then greedily drop near-duplicate
            # features (|corr| > 0.95 with an already-picked one) — keeps the
            # heatmap informative for pathway / module scores. Constraint-based
            # flux (Compass) is heavily flux-coupled, so few features survive;
            # pass decorrelate=False there to show the dense flux landscape.
            pool = list(ranked[: max(n_features * 8, 200)])
            corr = grouped.loc[pool].T.corr().abs()
            picked: list = []
            for f in pool:
                if len(picked) >= n_features:
                    break
                if all(corr.loc[f, p] < 0.95 for p in picked):
                    picked.append(f)
            grouped = grouped.loc[picked]
        else:
            grouped = grouped.loc[ranked[:n_features]]

    if standard_scale == "var":
        grouped = grouped.sub(grouped.mean(axis=1), axis=0).div(
            grouped.std(axis=1).replace(0, 1), axis=0
        )
    elif standard_scale == "group":
        grouped = grouped.sub(grouped.mean(axis=0), axis=1).div(
            grouped.std(axis=0).replace(0, 1), axis=1
        )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grouped.to_numpy(), aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(grouped.shape[1]))
    ax.set_xticklabels(grouped.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(grouped.shape[0]))
    ax.set_yticklabels(grouped.index, fontsize=7)
    ax.set_xlabel(groupby)
    method = adata.uns["metabolism"].get("method", "metabolism")
    ax.set_title(f"{method} — metabolic features by {groupby}")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel(
        "z-score" if standard_scale else "mean score", rotation=90, fontsize=8
    )
    return ax

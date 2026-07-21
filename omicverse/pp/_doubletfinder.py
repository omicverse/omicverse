"""Thin wrapper around ``doubletfinder-py`` for omicverse's QC pipeline.

Exposes a single ``doubletfinder(adata, ...)`` function used by ``ov.pp.qc``
with ``doublets_method="doubletfinder"``. Writes two columns to
``adata.obs``:

    * ``predicted_doublet`` — bool, for downstream compatibility with the
      scrublet branch of ``ov.pp.qc`` (the same filter logic reuses this name).
    * ``doublet_score``     — pANN score per real cell.

Unlike scrublet/sccomposite, the R-origin DoubletFinder requires an
*expected doublet rate* (``nExp``). We default to 7.5% (10x Chromium
rule-of-thumb), optionally adjusted for homotypic doublets when the
caller hands us a cluster-label column.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import anndata


def doubletfinder(
    adata: anndata.AnnData,
    *,
    pN: float = 0.25,
    pK: Optional[float] = None,
    expected_doublet_rate: float = 0.075,
    homotypic_annotations: Optional[str] = None,
    PCs: int = 10,
    n_top_genes: int = 2000,
    random_state: int = 1234,
    batch_key: Optional[str] = None,
) -> anndata.AnnData:
    """Run DoubletFinder and write calls to ``adata.obs``.

    Parameters
    ----------
    pN, pK
        DoubletFinder pN/pK. ``pK=None`` triggers the ``find_pK`` sweep;
        pass an explicit value to skip the sweep (much faster).
    expected_doublet_rate
        Used to compute ``nExp = round(rate * n_cells)``. 7.5% is the 10x
        Chromium default; override with the manufacturer's loading spec.
    homotypic_annotations
        Name of a cluster-label column in ``adata.obs``. When set, ``nExp``
        is downscaled by the estimated homotypic proportion (matches the
        R workflow's ``modelHomotypic`` step).
    PCs, n_top_genes, random_state
        Passed straight through to the ``DoubletFinder`` class.
    batch_key
        When set, DoubletFinder is run independently per batch and the
        calls are merged. Matches the scrublet branch's behavior.
    """
    try:
        from pydoubletfinder import DoubletFinder, model_homotypic
    except ImportError as exc:  # pragma: no cover - pip install missing
        raise ImportError(
            "pydoubletfinder is required for doublets_method='doubletfinder'. "
            "Install it with `pip install pydoubletfinder`."
        ) from exc

    # Reuse omicverse's chunked-GPU exact kNN for the pANN neighbour search when
    # a GPU is present. DoubletFinder's default pANN uses a full O(N^2) distance
    # matrix (OOMs at ~1e5 cells); the injected kNN is bit-identical and O(N·k).
    # Guarded with inspect so an older pydoubletfinder without the hook still works.
    import inspect as _inspect
    _extra = {}
    try:
        import torch as _torch
        if _torch.cuda.is_available() and "knn_fn" in _inspect.signature(DoubletFinder.run).parameters:
            from .pyg_knn_implementation import pyg_knn_search as _pyg_knn

            def _knn_fn(X, n_neighbors):
                idx, dist = _pyg_knn(X, k=int(n_neighbors), device="cuda",
                                     include_self=True)
                return dist, idx
            _extra["knn_fn"] = _knn_fn
    except Exception:
        _extra = {}

    def _run_one(sub: anndata.AnnData) -> tuple[np.ndarray, np.ndarray]:
        """Return (predicted_doublet, pANN) for a single (sub-)AnnData.

        ``sub`` is always a fresh, in-memory slice handed in by the caller —
        no extra ``.copy()`` is needed here before giving it to DoubletFinder.
        """
        df = DoubletFinder(sub, random_state=random_state)
        if pK is None:
            df.param_sweep(PCs=PCs, n_top_genes=n_top_genes, **_extra)
            df.summarize_sweep()
            df.find_pK()
            pK_use = df.optimal_pK
        else:
            pK_use = float(pK)
        nExp = int(round(expected_doublet_rate * sub.n_obs))
        if homotypic_annotations is not None and homotypic_annotations in sub.obs.columns:
            nExp = int(round((1.0 - model_homotypic(sub.obs[homotypic_annotations])) * nExp))
        nExp = max(1, min(nExp, sub.n_obs - 1))
        df.run(
            pN=pN, pK=pK_use, nExp=nExp,
            annotations=homotypic_annotations if homotypic_annotations in sub.obs.columns else None,
            PCs=PCs, n_top_genes=n_top_genes, **_extra,
        )
        # Robust lookup — if pydoubletfinder changes its column naming we want
        # a clear message, not a bare StopIteration.
        cols = df.adata.obs.columns
        df_matches = [c for c in cols if c.startswith("DF.classifications_")]
        pa_matches = [c for c in cols if c.startswith("pANN_")]
        if not df_matches or not pa_matches:
            raise RuntimeError(
                "pydoubletfinder did not produce the expected obs columns "
                "(DF.classifications_* and pANN_*). "
                f"Got: {list(cols)}. This usually means the installed "
                "pydoubletfinder version is incompatible — check "
                "`pip show pydoubletfinder`."
            )
        dfcol, pcol = df_matches[0], pa_matches[0]
        return (
            (df.adata.obs[dfcol].values == "Doublet"),
            df.adata.obs[pcol].astype(float).values,
        )

    predicted = np.zeros(adata.n_obs, dtype=bool)
    scores = np.zeros(adata.n_obs, dtype=np.float64)

    if batch_key is None or batch_key not in adata.obs.columns:
        predicted, scores = _run_one(adata.copy())
    else:
        for batch in adata.obs[batch_key].unique():
            mask = (adata.obs[batch_key] == batch).values
            sub = adata[mask].copy()
            p, s = _run_one(sub)
            predicted[mask] = p
            scores[mask] = s

    adata.obs["predicted_doublet"] = predicted
    adata.obs["doublet_score"] = scores
    adata.obs["doubletfinder_doublet"] = predicted
    adata.obs["doubletfinder_pANN"] = scores
    return adata

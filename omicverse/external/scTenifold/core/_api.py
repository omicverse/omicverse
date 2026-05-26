from typing import Iterable, Optional, Union

import pandas as pd

from omicverse.external.scTenifold.core._base import scTenifoldKnk, scTenifoldNet
from omicverse.external.scTenifold.core._networks import anndata_to_dataframe
from omicverse.external.scTenifold.core._types import Backend, ExpressionData, KOMethod, Kwargs, LayerName

__all__ = ["compare_networks", "virtual_knockout"]

def _network_kws(backend: Backend,
                 n_jobs: int,
                 random_state: int,
                 network_kws: Optional[Kwargs]) -> Kwargs:
    """Merge top-level parallel/random args into the ``nc_kws`` dict.

    Top-level ``backend``, ``n_jobs`` and ``random_state`` act as defaults;
    any value already present in ``network_kws`` wins.
    """
    kws = {} if network_kws is None else dict(network_kws)
    kws.setdefault("backend", backend)
    kws.setdefault("n_jobs", n_jobs)
    kws.setdefault("random_state", random_state)
    return kws


def compare_networks(x_data: ExpressionData,
                     y_data: ExpressionData,
                     x_label: str = "X",
                     y_label: str = "Y",
                     layer: LayerName = None,
                     backend: Backend = "serial",
                     n_jobs: int = 1,
                     random_state: int = 42,
                     qc_kws: Optional[Kwargs] = None,
                     network_kws: Optional[Kwargs] = None,
                     td_kws: Optional[Kwargs] = None,
                     ma_kws: Optional[Kwargs] = None,
                     dr_kws: Optional[Kwargs] = None) -> pd.DataFrame:
    """Run the full two-sample scTenifoldNet workflow.

    Builds PC networks for ``x_data`` and ``y_data``, performs tensor
    decomposition and manifold alignment, then returns the differential
    regulation table.

    Parameters
    ----------
    x_data, y_data
        Genes-by-cells expression matrices. Each may be a ``pandas.DataFrame``
        or an AnnData-like object (``X``/``var_names``/``obs_names``); the
        latter is converted via :func:`anndata_to_dataframe`.
    x_label, y_label
        Labels used internally and in the output to distinguish the two
        conditions.
    layer
        Optional AnnData layer name. If ``None``, ``adata.X`` is used.
    backend
        Parallel backend for PC network construction. One of
        ``"serial"``, ``"joblib-loky"``, ``"joblib-threading"``, ``"ray"``.
    n_jobs
        Worker count for the chosen backend. ``-1`` means all available
        cores. Ignored when ``backend="serial"``.
    random_state
        Seed propagated to the randomized SVD inside network construction.
    qc_kws, network_kws, td_kws, ma_kws, dr_kws
        Per-step keyword overrides forwarded to QC, network construction,
        tensor decomposition, manifold alignment and differential
        regulation, respectively. ``backend``/``n_jobs``/``random_state``
        already present in ``network_kws`` take precedence over the
        top-level arguments.

    Returns
    -------
    pandas.DataFrame
        Differential regulation table with one row per shared gene.
    """
    sc = scTenifoldNet(
        anndata_to_dataframe(x_data, layer=layer),
        anndata_to_dataframe(y_data, layer=layer),
        x_label=x_label,
        y_label=y_label,
        qc_kws=qc_kws,
        nc_kws=_network_kws(backend, n_jobs, random_state, network_kws),
        td_kws=td_kws,
        ma_kws=ma_kws,
        dr_kws=dr_kws,
    )
    return sc.build()


def virtual_knockout(data: ExpressionData,
                     ko_genes: Optional[Union[str, Iterable[str]]] = None,
                     layer: LayerName = None,
                     backend: Backend = "serial",
                     n_jobs: int = 1,
                     random_state: int = 42,
                     strict_lambda: float = 0,
                     ko_method: KOMethod = "default",
                     qc_kws: Optional[Kwargs] = None,
                     network_kws: Optional[Kwargs] = None,
                     td_kws: Optional[Kwargs] = None,
                     ma_kws: Optional[Kwargs] = None,
                     dr_kws: Optional[Kwargs] = None,
                     ko_kws: Optional[Kwargs] = None) -> pd.DataFrame:
    """Run the full scTenifoldKnk virtual-knockout workflow.

    Constructs a wild-type PC network from ``data``, simulates a knockout
    of ``ko_genes``, performs manifold alignment between WT and KO tensors
    and returns the differential regulation table.

    Parameters
    ----------
    data
        Genes-by-cells expression matrix (``pandas.DataFrame`` or
        AnnData-like).
    ko_genes
        Gene name or iterable of gene names to knock out. ``None`` and an
        empty iterable both mean "no genes", which is rarely useful.
    layer
        Optional AnnData layer name. If ``None``, ``adata.X`` is used.
    backend
        Parallel backend for PC network construction. One of
        ``"serial"``, ``"joblib-loky"``, ``"joblib-threading"``, ``"ray"``.
    n_jobs
        Worker count for the chosen backend. ``-1`` means all available
        cores.
    random_state
        Seed propagated to the randomized SVD inside network construction.
    strict_lambda
        Strength of the directional pruning applied by
        :func:`strict_direction` to the decomposed WT tensor.
    ko_method
        How the KO tensor is produced:

        - ``"default"`` — zero out the rows of the WT tensor for
          ``ko_genes``.
        - ``"propagation"`` — rebuild PC networks with the targeted genes
          masked using :func:`reconstruct_pcnets`, then re-decompose.
    qc_kws, network_kws, td_kws, ma_kws, dr_kws, ko_kws
        Per-step keyword overrides forwarded to QC, network construction,
        tensor decomposition, manifold alignment, differential regulation,
        and the KO step. ``backend``/``n_jobs``/``random_state`` already
        present in ``network_kws`` take precedence over the top-level
        arguments.

    Returns
    -------
    pandas.DataFrame
        Differential regulation table comparing the WT and KO tensors.
    """
    sc = scTenifoldKnk(
        anndata_to_dataframe(data, layer=layer),
        strict_lambda=strict_lambda,
        ko_method=ko_method,
        ko_genes=ko_genes,
        qc_kws=qc_kws,
        nc_kws=_network_kws(backend, n_jobs, random_state, network_kws),
        td_kws=td_kws,
        ma_kws=ma_kws,
        dr_kws=dr_kws,
        ko_kws=ko_kws,
    )
    return sc.build()

r"""Single-cell metabolism inference (scMetabolism / Compass / scFEA / MEBOCOST).

Two :func:`register_function`-decorated entry points, mirroring
:class:`omicverse.single.CNV`:

* :class:`Metabolism` — per-cell **metabolic state** with
  ``method='scmetabolism' | 'compass' | 'scfea'`` dispatch. All three
  backends share a unified output schema:

  * ``adata.obsm['X_metabolism']`` — cells × metabolic-feature matrix
    (pathway-activity scores / reaction activities / module fluxes)
  * ``adata.uns['metabolism']``    — ``dict`` with ``method``,
    ``features`` (column names of ``X_metabolism``) and backend metadata

* :class:`MetaboliteCCC` — metabolite-mediated **cell-cell communication**
  with MEBOCOST. Writes ``adata.uns['mebocost']`` and exposes
  :meth:`MetaboliteCCC.to_comm_adata` so the result feeds the existing
  ``ov.pl.ccc_heatmap`` / ``ccc_network_plot`` / ``ccc_stat_plot`` plots.

The pathway-scoring backend (scMetabolism) and the flux backend (scFEA)
are vendored under :mod:`omicverse.external`; Compass is a heavyweight,
solver-bound CLI tool, so :class:`Metabolism` integrates **precomputed
Compass output** rather than embedding the solver.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from .._registry import register_function

_METHODS = ("scmetabolism", "compass", "scfea")


def _check_dep(method: str) -> None:
    """Lazy-import gate. Raises a clean ImportError with an actionable hint."""
    if method == "scmetabolism":
        try:
            from ..external import py_scmetabolism  # noqa: F401
        except ImportError as e:  # pragma: no cover - vendored, should not happen
            raise ImportError(
                "the vendored py_scmetabolism backend is unavailable; "
                "reinstall omicverse so omicverse/external/py_scmetabolism is present."
            ) from e
    elif method == "scfea":
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "scFEA (method='scfea') needs PyTorch. Install via:\n"
                "  pip install torch"
            ) from e
        try:
            from ..external import scfea  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "the vendored scFEA backend is unavailable; reinstall omicverse "
                "so omicverse/external/scfea is present."
            ) from e
    elif method == "compass":
        # Compass is a solver-bound CLI tool (needs Gurobi/CPLEX); the
        # backend only requires a precomputed output directory at run time,
        # so there is nothing to import-check here.
        pass
    else:
        raise ValueError(
            f"unknown metabolism method {method!r} (expected one of {_METHODS})"
        )


@register_function(
    aliases=["代谢", "metabolism", "scMetabolism", "代谢活性", "代谢通量",
             "metabolic_flux", "metabolic_pathway", "compass", "scfea"],
    category="single",
    description=(
        "Single-cell metabolism inference. Wraps scMetabolism (pathway "
        "activity), Compass (constraint-based reaction flux) and scFEA "
        "(graph-neural-network module flux) behind a unified method= API; "
        "results land in adata.obsm['X_metabolism'] / adata.uns['metabolism']."
    ),
    requires={"var": ["gene symbols"]},
    produces={"obsm": ["X_metabolism"], "uns": ["metabolism"]},
    auto_fix="none",
    examples=[
        "# scMetabolism — metabolic pathway activity per cell",
        "met = ov.single.Metabolism(adata, method='scmetabolism')",
        "met.run(score_method='AUCell', metabolism_type='KEGG')",
        "",
        "# scFEA — graph-neural-network metabolic module flux",
        "met = ov.single.Metabolism(adata, method='scfea')",
        "met.run(n_epoch=100)",
        "",
        "# Compass — load a precomputed Compass run (needs Gurobi to compute)",
        "met = ov.single.Metabolism(adata, method='compass')",
        "met.run(compass_dir='compass_out/', group_key='cell_type')",
    ],
    related=["single.MetaboliteCCC", "pl.embedding", "single.CNV"],
)
class Metabolism:
    """Single-cell metabolic-state inference dispatcher.

    Parameters
    ----------
    adata : AnnData
        Single-cell expression data. For ``method='scmetabolism'`` and
        ``method='scfea'`` should hold normalised (log1p) expression in
        ``adata.X`` or ``layer``. For ``method='compass'`` the AnnData is
        only used to align the precomputed Compass output back onto cells.
    method : {'scmetabolism', 'compass', 'scfea'}
        Which backend to run.
    layer : str or None, default None
        Layer to score against instead of ``adata.X``.
    """

    def __init__(
        self,
        adata: AnnData,
        method: str = "scmetabolism",
        *,
        layer: Optional[str] = None,
    ) -> None:
        method = method.lower()
        if method not in _METHODS:
            raise ValueError(
                f"method must be one of {_METHODS}, got {method!r}"
            )
        _check_dep(method)
        self.adata = adata
        self.method = method
        self.layer = layer
        self.result: Any = None

    # ------------------------------------------------------------------ #
    # public                                                              #
    # ------------------------------------------------------------------ #

    def run(self, **kwargs: Any) -> "Metabolism":
        r"""Execute the chosen backend and write results into ``self.adata``.

        Keyword arguments are forwarded to the backend-specific runner
        (:meth:`_run_scmetabolism`, :meth:`_run_compass`, :meth:`_run_scfea`).

        Returns
        -------
        self
            Side-effects: writes ``adata.obsm['X_metabolism']`` (cells ×
            metabolic features) and ``adata.uns['metabolism']``.
        """
        if self.method == "scmetabolism":
            return self._run_scmetabolism(**kwargs)
        if self.method == "compass":
            return self._run_compass(**kwargs)
        return self._run_scfea(**kwargs)

    def get(self, features: Union[str, Sequence[str], None] = None) -> pd.DataFrame:
        r"""Return the metabolic-feature matrix as a tidy ``cells × features`` DataFrame.

        Parameters
        ----------
        features : str or list of str or None
            Restrict to these features (pathways / reactions / modules).
            ``None`` returns all.
        """
        if "metabolism" not in self.adata.uns:
            raise RuntimeError("run() has not been called yet.")
        names = list(self.adata.uns["metabolism"]["features"])
        df = pd.DataFrame(
            np.asarray(self.adata.obsm["X_metabolism"]),
            index=self.adata.obs_names,
            columns=names,
        )
        if features is not None:
            if isinstance(features, str):
                features = [features]
            df = df[list(features)]
        return df

    def to_obs(self, features: Union[str, Sequence[str]]) -> "Metabolism":
        r"""Copy named metabolic features into ``adata.obs`` for plotting.

        Convenience for ``ov.pl.embedding(adata, color=<feature>)``.
        """
        if isinstance(features, str):
            features = [features]
        sub = self.get(features)
        for col in sub.columns:
            self.adata.obs[col] = sub[col].to_numpy()
        return self

    # ------------------------------------------------------------------ #
    # backends                                                            #
    # ------------------------------------------------------------------ #

    def _run_scmetabolism(
        self,
        *,
        score_method: str = "AUCell",
        metabolism_type: str = "KEGG",
        imputation: bool = False,
        n_cores: int = 2,
        **_: Any,
    ) -> "Metabolism":
        r"""scMetabolism — metabolic pathway-activity scoring.

        Parameters
        ----------
        score_method : {'AUCell', 'VISION', 'ssGSEA', 'GSVA'}
            Pathway-activity scorer.
        metabolism_type : {'KEGG', 'REACTOME'}
            Curated metabolic gene-set collection.
        imputation : bool
            Run ALRA imputation before scoring (recommended for sparse 10x).
        n_cores : int
            Worker processes.
        """
        from ..external.py_scmetabolism import sc_metabolism_anndata

        valid = {"AUCell", "VISION", "ssGSEA", "GSVA"}
        if score_method not in valid:
            raise ValueError(f"score_method must be one of {valid}, got {score_method!r}")

        sc_metabolism_anndata(
            self.adata,
            method=score_method,
            metabolism_type=metabolism_type,
            imputation=imputation,
            ncores=n_cores,
            layer=self.layer,
            key_added="metabolism",
        )
        # sc_metabolism_anndata writes obsm['X_metabolism'] + uns['metabolism_pathways'].
        features = list(self.adata.uns.get("metabolism_pathways", []))
        self.adata.uns["metabolism"] = {
            "method": "scmetabolism",
            "features": features,
            "feature_type": "pathway",
            "score_method": score_method,
            "metabolism_type": metabolism_type,
            "level": "cell",
        }
        self.result = self.adata.obsm["X_metabolism"]
        return self

    def _run_scfea(
        self,
        *,
        species: str = "human",
        n_epoch: int = 100,
        sc_imputation: bool = False,
        output: str = "flux",
        verbose: bool = True,
        **_: Any,
    ) -> "Metabolism":
        r"""scFEA — graph-neural-network metabolic-module flux estimation.

        Parameters
        ----------
        species : {'human', 'mouse'}
            Selects the bundled M168 module / stoichiometry model.
        n_epoch : int
            GNN training epochs.
        sc_imputation : bool
            MAGIC-impute the expression first (recommended for sparse 10x).
        output : {'flux', 'balance'}
            Store module flux (cells × 168 modules) or metabolite balance
            (cells × 70 metabolites) in ``X_metabolism``. Both are kept in
            ``uns['metabolism']``.
        """
        from ..external.scfea import run_scfea

        if output not in {"flux", "balance"}:
            raise ValueError(f"output must be 'flux' or 'balance', got {output!r}")

        expr = self.adata.to_df(layer=self.layer)  # cells × genes
        res = run_scfea(
            expr,
            species=species,
            sc_imputation=sc_imputation,
            n_epoch=n_epoch,
            verbose=verbose,
        )
        flux = res["flux"].reindex(self.adata.obs_names)
        balance = res["balance"].reindex(self.adata.obs_names)
        chosen = flux if output == "flux" else balance

        self.adata.obsm["X_metabolism"] = chosen.to_numpy(dtype=np.float32)
        self.adata.obsm["X_scfea_flux"] = flux.to_numpy(dtype=np.float32)
        self.adata.obsm["X_scfea_balance"] = balance.to_numpy(dtype=np.float32)
        self.adata.uns["metabolism"] = {
            "method": "scfea",
            "features": list(chosen.columns),
            "feature_type": "module_flux" if output == "flux" else "metabolite_balance",
            "flux_features": list(flux.columns),
            "balance_features": list(balance.columns),
            "species": species,
            "n_epoch": n_epoch,
            "level": "cell",
        }
        self.result = res
        return self

    def _run_compass(
        self,
        *,
        compass_dir: Optional[str] = None,
        group_key: Optional[str] = None,
        output: str = "reactions",
        postprocess: bool = True,
        **_: Any,
    ) -> "Metabolism":
        r"""Compass — constraint-based metabolic reaction flux.

        Compass is a heavyweight, solver-bound CLI tool (it needs an IBM
        CPLEX or Gurobi licence and runs for hours). This backend therefore
        **loads a precomputed Compass output directory** and maps it back
        onto the AnnData. Run Compass itself separately::

            compass --data expr.tsv --model RECON2_mat --species homo_sapiens

        Parameters
        ----------
        compass_dir : str
            Directory holding Compass output (``reactions.tsv[.gz]``,
            optionally ``uptake.tsv[.gz]`` / ``secretions.tsv[.gz]``).
        group_key : str or None
            If Compass was run on cell groups / microclusters (its columns
            are ``adata.obs[group_key]`` categories rather than cell
            barcodes), the group-level result is broadcast to every cell.
        output : {'reactions', 'uptake', 'secretions'}
            Which Compass matrix to load into ``X_metabolism``.
        postprocess : bool
            Convert Compass penalties to reaction activity scores
            ``-log(penalty + 1)`` shifted to a non-negative minimum
            (the standard Compass post-processing).
        """
        if compass_dir is None:
            raise ValueError(
                "method='compass' needs compass_dir=<precomputed Compass output>. "
                "Compass is solver-bound (CPLEX/Gurobi); run it separately, then "
                "load its output here."
            )
        if output not in {"reactions", "uptake", "secretions"}:
            raise ValueError(
                f"output must be 'reactions'/'uptake'/'secretions', got {output!r}"
            )
        mat = _read_compass_matrix(compass_dir, output)  # features × columns
        if postprocess:
            mat = -np.log1p(mat)
            mat = mat - np.nanmin(mat.to_numpy())

        X, level = _align_compass_to_cells(mat, self.adata, group_key)
        self.adata.obsm["X_metabolism"] = X.astype(np.float32)
        self.adata.uns["metabolism"] = {
            "method": "compass",
            "features": list(mat.index),
            "feature_type": output[:-1] if output.endswith("s") else output,
            "level": level,
            "postprocessed": postprocess,
            "compass_dir": str(compass_dir),
        }
        self.result = mat
        return self


@register_function(
    aliases=["代谢物通讯", "MEBOCOST", "metabolite_communication",
             "代谢物细胞通讯", "mebocost", "metabolic_communication"],
    category="single",
    description=(
        "Metabolite-mediated cell-cell communication with MEBOCOST. Infers "
        "sender cell groups (high metabolite-enzyme expression) signalling to "
        "receiver groups (high sensor expression); results land in "
        "adata.uns['mebocost'] and convert to a communication AnnData for "
        "the ov.pl.ccc_* plots."
    ),
    requires={"obs": ["a cell-group / cell-type column"]},
    produces={"uns": ["mebocost"]},
    auto_fix="none",
    examples=[
        "mccc = ov.single.MetaboliteCCC(adata, group_key='cell_type')",
        "mccc.run(n_shuffle=1000)",
        "comm = mccc.to_comm_adata()           # for the ov.pl.ccc_* plots",
        "ov.pl.ccc_heatmap(comm, plot_type='heatmap')",
    ],
    related=["single.Metabolism", "pl.ccc_heatmap", "pl.ccc_network_plot",
             "single.to_comm_adata"],
)
class MetaboliteCCC:
    """Metabolite-mediated cell-cell communication (MEBOCOST).

    Parameters
    ----------
    adata : AnnData
        Single-cell expression with a cell-group annotation in ``obs``.
    group_key : str
        ``adata.obs`` column of cell groups / cell types — the sender and
        receiver units of communication.
    species : {'human', 'mouse'}
        Selects the MEBOCOST metabolite-sensor database.
    condition_key : str or None
        Optional ``obs`` column for condition-stratified inference.
    layer : str or None
        Expression layer; ``None`` uses ``adata.X``.
    """

    def __init__(
        self,
        adata: AnnData,
        *,
        group_key: str,
        species: str = "human",
        condition_key: Optional[str] = None,
        layer: Optional[str] = None,
    ) -> None:
        if group_key not in adata.obs.columns:
            raise ValueError(f"group_key {group_key!r} not in adata.obs")
        _check_mebocost_dep()
        self.adata = adata
        self.group_key = group_key
        self.species = species
        self.condition_key = condition_key
        self.layer = layer
        self.result: Optional[pd.DataFrame] = None

    def run(
        self,
        *,
        n_shuffle: int = 1000,
        seed: int = 12345,
        thread: int = 1,
        verbose: bool = True,
        **kwargs: Any,
    ) -> "MetaboliteCCC":
        r"""Run MEBOCOST metabolite-sensor communication inference.

        Returns
        -------
        self
            Side-effect: writes the communication table to
            ``adata.uns['mebocost']``.
        """
        from ..external.mebocost import run_mebocost

        res = run_mebocost(
            self.adata,
            group_key=self.group_key,
            species=self.species,
            condition_key=self.condition_key,
            n_shuffle=n_shuffle,
            seed=seed,
            thread=thread,
            verbose=verbose,
            **kwargs,
        )
        self.result = res
        self.adata.uns["mebocost"] = {
            "communication": res,
            "group_key": self.group_key,
            "species": self.species,
        }
        return self

    def to_comm_adata(self, *, pvalue_threshold: float = 1.0) -> AnnData:
        r"""Convert the MEBOCOST result to a communication AnnData.

        The returned object carries ``obs['sender']`` / ``obs['receiver']``
        and ``layers['means']`` / ``layers['pvalues']`` — the schema the
        ``ov.pl.ccc_heatmap`` / ``ccc_network_plot`` / ``ccc_stat_plot``
        plots consume.

        Parameters
        ----------
        pvalue_threshold : float
            Keep only communication events at or below this p-value.
        """
        if self.result is None:
            raise RuntimeError("run() has not been called yet.")
        return _mebocost_to_comm_adata(self.result, pvalue_threshold=pvalue_threshold)


# ------------------------------------------------------------------ #
# helpers                                                             #
# ------------------------------------------------------------------ #


def _check_mebocost_dep() -> None:
    try:
        from ..external import mebocost  # noqa: F401
    except ImportError as e:  # pragma: no cover - vendored
        raise ImportError(
            "the vendored MEBOCOST backend is unavailable; reinstall omicverse "
            "so omicverse/external/mebocost is present."
        ) from e


def _read_compass_matrix(compass_dir: str, output: str) -> pd.DataFrame:
    """Load a Compass output TSV (features × columns)."""
    for ext in (".tsv.gz", ".tsv"):
        path = os.path.join(compass_dir, f"{output}{ext}")
        if os.path.exists(path):
            return pd.read_csv(path, sep="\t", index_col=0)
    raise FileNotFoundError(
        f"no {output}.tsv[.gz] found in {compass_dir!r}"
    )


def _align_compass_to_cells(
    mat: pd.DataFrame,
    adata: AnnData,
    group_key: Optional[str],
) -> tuple[np.ndarray, str]:
    """Map a Compass (features × columns) matrix onto adata cells.

    Returns ``(cells × features array, level)`` where ``level`` is
    ``'cell'`` (columns matched cell barcodes) or ``'group'`` (columns
    matched ``obs[group_key]`` categories and were broadcast to cells).
    """
    cols = set(mat.columns)
    if cols.issuperset(set(adata.obs_names)):
        return mat.T.reindex(adata.obs_names).to_numpy(), "cell"
    if group_key is not None and group_key in adata.obs:
        groups = adata.obs[group_key].astype(str)
        if cols.issuperset(set(groups.unique())):
            return mat.T.reindex(groups.to_numpy()).to_numpy(), "group"
    raise ValueError(
        "Compass output columns match neither adata.obs_names nor "
        f"adata.obs[{group_key!r}]; pass the correct group_key."
    )


def _mebocost_to_comm_adata(res: pd.DataFrame, *, pvalue_threshold: float) -> AnnData:
    """Shape a MEBOCOST communication table into a comm-AnnData.

    The schema mirrors what ``ov.pl.ccc_*`` consumes for ligand-receptor
    inputs (LIANA): ``obs`` rows are sender→receiver cell-type pairs,
    ``var`` rows are metabolite→sensor interactions, and ``layers``
    carry per-pair × per-interaction matrices.

    var metadata:
      * ``gene_a`` — metabolite display name (resolves to "ligand" in
        downstream plots)
      * ``gene_b`` — sensor gene (resolves to "receptor")
      * ``interaction_name`` — ``metabolite → sensor``
      * ``classification`` — HMDB ``sub_class`` (fine-grained pathway-like
        grouping, ~50–100 categories) if the HMDB annotation table can be
        located; falls back to ``"Unclassified"``
      * ``classification_super`` — HMDB ``super_class`` (~20 broad groups)
    """
    df = res.copy()

    def _pick(*cands: str) -> str:
        for c in cands:
            if c in df.columns:
                return c
        raise KeyError(f"MEBOCOST result has none of {cands}; got {list(df.columns)}")

    sender = _pick("Sender", "sender", "Cell_Sender")
    receiver = _pick("Receiver", "receiver", "Cell_Receiver")
    metab = _pick("Metabolite_Name", "Metabolite", "metabolite")
    sensor = _pick("Sensor", "sensor")
    score = _pick("Commu_Score", "communication_score", "score")
    pval = _pick("permutation_test_fdr", "permutation_test_pval",
                 "permutation_test_pvalue", "ttest_fdr", "pvalue", "pval")
    hmdb_col = next((c for c in ("Metabolite", "HMDB_ID", "metabolite") if c in df.columns), None)

    df = df[df[pval] <= pvalue_threshold].copy()
    df["_metab"] = df[metab].astype(str)
    df["_sensor"] = df[sensor].astype(str)
    df["_sender"] = df[sender].astype(str)
    df["_receiver"] = df[receiver].astype(str)
    df["_pair"] = df["_sender"] + "|" + df["_receiver"]
    df["_interaction"] = df["_metab"] + " → " + df["_sensor"]

    # If multiple rows share (pair, interaction) keep the most-significant one
    # (smallest p, ties broken by larger score).
    df = (df.sort_values([pval, score], ascending=[True, False])
            .drop_duplicates(subset=["_pair", "_interaction"], keep="first"))

    # Index names left blank on purpose — ov.pl.ccc_* builds a long_df via
    # ``means.stack().reset_index()`` and expects the auto-generated
    # ``level_0`` / ``level_1`` columns to be renamed to ``pair_id`` /
    # ``feature_id`` (see ``_communication_long_table``). Naming the index
    # breaks that rename and surfaces as ``KeyError: 'pair_id'`` from the
    # downstream merge.
    pairs = pd.Index(sorted(df["_pair"].unique()))
    inters = pd.Index(sorted(df["_interaction"].unique()))

    means = (df.pivot(index="_pair", columns="_interaction", values=score)
               .reindex(index=pairs, columns=inters))
    pvalues = (df.pivot(index="_pair", columns="_interaction", values=pval)
                 .reindex(index=pairs, columns=inters))
    means_arr = means.fillna(0.0).to_numpy(dtype=np.float32)
    pvalues_arr = pvalues.fillna(1.0).to_numpy(dtype=np.float32)

    obs = pd.DataFrame(index=pairs)
    split = obs.index.to_series().str.split("|", n=1, expand=True)
    obs["sender"] = split[0].astype(str).to_numpy()
    obs["receiver"] = split[1].astype(str).to_numpy()

    # var carries the interaction-level metadata the ov.pl.ccc_* plots need.
    var = pd.DataFrame(index=inters)
    metab_for_var = (df.drop_duplicates("_interaction")
                       .set_index("_interaction")["_metab"]
                       .reindex(inters))
    sensor_for_var = (df.drop_duplicates("_interaction")
                        .set_index("_interaction")["_sensor"]
                        .reindex(inters))
    var["interaction_name"] = inters.to_numpy()
    var["interaction_name_2"] = inters.to_numpy()
    var["interacting_pair"] = inters.to_numpy()
    var["gene_a"] = metab_for_var.to_numpy()
    var["gene_b"] = sensor_for_var.to_numpy()
    var["ligand"] = metab_for_var.to_numpy()
    var["receptor"] = sensor_for_var.to_numpy()
    var["metabolite"] = metab_for_var.to_numpy()
    var["sensor"] = sensor_for_var.to_numpy()

    hmdb_id_for_var = None
    if hmdb_col is not None:
        hmdb_id_for_var = (df.drop_duplicates("_interaction")
                             .set_index("_interaction")[hmdb_col]
                             .reindex(inters).astype(str))
        var["hmdb_id"] = hmdb_id_for_var.to_numpy()

    sub_cls, super_cls = _hmdb_classification(metab_for_var, hmdb_id_for_var)
    var["classification"] = sub_cls
    var["classification_super"] = super_cls
    var["pathway_name"] = sub_cls
    var["signaling"] = sub_cls

    comm = AnnData(X=means_arr.copy(), obs=obs, var=var)
    comm.layers["means"] = means_arr
    comm.layers["pvalues"] = pvalues_arr
    comm.uns["mebocost_comm"] = True
    return comm


def _hmdb_classification(
    metab_for_var: pd.Series,
    hmdb_for_var: Optional[pd.Series],
) -> tuple[np.ndarray, np.ndarray]:
    """Join HMDB sub_class / super_class onto the metabolite list.

    Returns
    -------
    sub_cls, super_cls : two object arrays aligned to ``metab_for_var``.
    Missing annotations become ``"Unclassified"``.
    """
    n = len(metab_for_var)
    sub_default = np.array(["Unclassified"] * n, dtype=object)
    super_default = np.array(["Unclassified"] * n, dtype=object)
    try:
        path = os.path.join(
            os.path.dirname(__file__), "..", "external", "mebocost",
            "data", "mebocost_db", "common",
            "metabolite_annotation_HMDB_summary.tsv",
        )
        path = os.path.abspath(path)
        if not os.path.exists(path):
            return sub_default, super_default
        ann = pd.read_csv(path, sep="\t", low_memory=False)
    except Exception:
        return sub_default, super_default

    # The HMDB table has 'metabolite' (display name), 'sub_class', 'super_class'
    name_to_sub = (ann.dropna(subset=["metabolite"])
                      .drop_duplicates("metabolite")
                      .set_index("metabolite"))
    sub_by_name = name_to_sub["sub_class"].astype(str)
    super_by_name = name_to_sub["super_class"].astype(str)
    sub = metab_for_var.map(sub_by_name)
    sup = metab_for_var.map(super_by_name)

    if hmdb_for_var is not None and "HMDB_ID" in ann.columns:
        id_to_sub = (ann.dropna(subset=["HMDB_ID"])
                        .drop_duplicates("HMDB_ID")
                        .set_index("HMDB_ID"))
        sub_by_id = id_to_sub["sub_class"].astype(str)
        super_by_id = id_to_sub["super_class"].astype(str)
        sub = sub.fillna(hmdb_for_var.map(sub_by_id))
        sup = sup.fillna(hmdb_for_var.map(super_by_id))

    sub = sub.replace({"nan": np.nan}).fillna("Unclassified").astype(str)
    sup = sup.replace({"nan": np.nan}).fillna("Unclassified").astype(str)
    return sub.to_numpy(), sup.to_numpy()


@register_function(
    aliases=["差异代谢", "differential_metabolism", "代谢差异分析",
             "metabolic_differential", "differential_metabolic_activity"],
    category="single",
    description=(
        "Differential metabolic-feature analysis between two groups of "
        "cells. Tests every column of adata.obsm['X_metabolism'] (pathway "
        "activities / reaction or module fluxes) for a difference between "
        "two obs groups and returns a ranked statistics table."
    ),
    requires={"obsm": ["X_metabolism"], "uns": ["metabolism"]},
    auto_fix="none",
    examples=[
        "# which metabolic pathways are up in malignant vs the rest",
        "deg = ov.single.differential_metabolism(",
        "    adata, groupby='celltype', group1='Malignant')",
        "deg.query('padj < 0.05 and log2fc > 0').head(20)",
    ],
    related=["single.Metabolism", "pl.metabolism_heatmap"],
)
def differential_metabolism(
    adata: AnnData,
    *,
    groupby: str,
    group1: str,
    group2: Union[str, Sequence[str]] = "rest",
    method: str = "wilcoxon",
) -> pd.DataFrame:
    r"""Differential metabolic features between two groups of cells.

    Compares every metabolic feature in ``adata.obsm['X_metabolism']``
    (written by :meth:`Metabolism.run`) between ``group1`` and ``group2``
    of ``adata.obs[groupby]``.

    Parameters
    ----------
    adata : AnnData
        Must carry ``obsm['X_metabolism']`` and ``uns['metabolism']``.
    groupby : str
        ``adata.obs`` column defining the groups.
    group1 : str
        The focal group (positive ``log2fc`` = higher in ``group1``).
    group2 : str or list of str, default ``'rest'``
        The comparison group; ``'rest'`` uses every cell not in ``group1``.
    method : {'wilcoxon', 't-test'}
        Per-feature test — Mann-Whitney U (default) or Welch's t-test.

    Returns
    -------
    pandas.DataFrame
        One row per metabolic feature, sorted by adjusted p-value:
        ``feature``, ``mean1``, ``mean2``, ``log2fc``, ``statistic``,
        ``pval``, ``padj`` (Benjamini-Hochberg), ``direction`` (``'up'`` /
        ``'down'`` / ``'ns'`` at ``padj < 0.05``).
    """
    from scipy import stats as _stats
    from statsmodels.stats.multitest import multipletests

    if "metabolism" not in adata.uns or "X_metabolism" not in adata.obsm:
        raise ValueError(
            "no metabolism result — run ov.single.Metabolism(...).run() first."
        )
    if groupby not in adata.obs.columns:
        raise ValueError(f"groupby {groupby!r} not in adata.obs")
    if method not in {"wilcoxon", "t-test"}:
        raise ValueError(f"method must be 'wilcoxon' or 't-test', got {method!r}")

    labels = adata.obs[groupby].astype(str).to_numpy()
    m1 = labels == str(group1)
    if isinstance(group2, str) and group2 == "rest":
        m2 = ~m1
        g2name = "rest"
    else:
        g2set = {group2} if isinstance(group2, str) else set(map(str, group2))
        m2 = np.isin(labels, list(g2set))
        g2name = "|".join(sorted(g2set))
    if m1.sum() < 3 or m2.sum() < 3:
        raise ValueError(
            f"each group needs >=3 cells (group1={int(m1.sum())}, "
            f"group2={int(m2.sum())})."
        )

    feats = list(adata.uns["metabolism"]["features"])
    X = np.asarray(adata.obsm["X_metabolism"], dtype=float)
    A, B = X[m1], X[m2]
    rows = []
    for j, feat in enumerate(feats):
        a, b = A[:, j], B[:, j]
        if method == "wilcoxon":
            stat, pval = _stats.mannwhitneyu(a, b, alternative="two-sided")
        else:
            stat, pval = _stats.ttest_ind(a, b, equal_var=False)
        mean1, mean2 = float(np.mean(a)), float(np.mean(b))
        log2fc = float(np.log2((mean1 + 1e-9) / (mean2 + 1e-9)))
        rows.append((feat, mean1, mean2, log2fc, float(stat), float(pval)))

    df = pd.DataFrame(
        rows, columns=["feature", "mean1", "mean2", "log2fc", "statistic", "pval"]
    )
    df["padj"] = multipletests(df["pval"].to_numpy(), method="fdr_bh")[1]
    df["direction"] = np.where(
        df["padj"] >= 0.05, "ns",
        np.where(df["log2fc"] > 0, "up", "down"),
    )
    df.attrs["group1"] = str(group1)
    df.attrs["group2"] = g2name
    return df.sort_values("padj").reset_index(drop=True)

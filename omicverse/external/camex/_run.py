"""Programmatic entry point for the vendored CAMEX pipeline.

This reproduces the glue found in the upstream CAMEX analysis notebooks
(``analysis/1liver/Integrate_liver_across_4_species.ipynb`` and
``analysis/4cortex_annotation/integration_annotation_in_relatives_distant_species.ipynb``)::

    from CAMEX.base import Dataset
    from CAMEX.trainer import Trainer
    from params import PARAMS

    for k, v in PARAMS.items():
        v['time_start'] = time_start
        v['log_path'] = log_path

    dataset = Dataset(**PARAMS['preprocess'])
    adata_CAMEX = dataset.adata_whole
    dgl_data = dataset.dgl_data
    trainer = Trainer(adata_CAMEX, dgl_data, **PARAMS['train'])
    trainer.integration()          # writes adata_CAMEX.obsm['X_CAMEX_Integration']
    trainer.annotation()           # optional; writes adata_CAMEX.obsm['X_CAMEX_Annotation']
    adata_CAMEX.write_h5ad(log_path + 'adata_CAMEX.h5ad')

``run_camex`` wires this together from in-memory AnnData objects: it writes the
per-species ``.h5ad`` files and the pairwise many-to-many homology CSVs that
CAMEX expects on disk, builds the merged ``PARAMS`` dict, runs
``Dataset -> Trainer.integration()`` (and optionally ``.annotation()``), and
returns the combined AnnData.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _default_params():
    """The full CAMEX ``PARAMS`` skeleton (from ``params_template.py``).

    ``dataset_file`` / ``path`` are filled in by :func:`run_camex`.
    """
    return {
        "preprocess": {
            "path": "./dataset/",
            "dataset_file": None,  # filled by run_camex
            "graph_mode": "undirected",
            "feature_gene": "HIG",
            "sample_ratio": 1,
            "get_balance": "False",
        },
        "train": {
            "device": "cuda:0",
            "train_mode": "mini_batch",
            "epoch_integration": 10,
            "epoch_annotation": 10,
            "batch_size": 1024,
            "dim_hidden": 128,
            "gnn_layer_num": 2,
            "encoder": "GCN",
            "classifier": "GAT",
            "res": True,
            "share": True,
            "cluster": False,
            "epoch_cluster": 10,
            "cluster_num": 5,
            "domain": False,
            "reconstruct": True,
        },
        "postprocess": {},
    }


def _prepare_adata(adata, label_col: str):
    """Return a copy with raw counts in ``.X`` and ``cell_ontology_class`` set.

    CAMEX's ``Dataset`` requires ``obs['cell_ontology_class']`` for *every*
    dataset (reference and query) and expects raw counts in ``.X``.
    """
    import scanpy as sc  # noqa: F401  (kept for parity / lazy heavy import)

    ad = adata.copy()
    if label_col is not None and label_col in ad.obs.columns:
        ad.obs["cell_ontology_class"] = ad.obs[label_col].astype(str).values
    elif "cell_ontology_class" not in ad.obs.columns:
        # No labels available -> assign a single placeholder cell type so that
        # graph construction / DEG clustering still works (integration only).
        ad.obs["cell_ontology_class"] = "unknown"
    ad.obs["cell_ontology_class"] = ad.obs["cell_ontology_class"].astype(str)
    # ensure gene symbols are unique strings
    ad.var_names = [str(g) for g in ad.var_names]
    ad.var_names_make_unique()
    return ad


def run_camex(
    adatas: List,
    species: List[str],
    *,
    homology: Optional[Dict[Tuple[str, str], "pd.DataFrame"]] = None,
    ref_index: int = 0,
    label_col: str = "cell_type",
    device: str = "cuda",
    epoch_integration: int = 10,
    epoch_annotation: int = 10,
    batch_size: int = 1024,
    work_dir: str = "./camex_work",
    annotate: bool = False,
    **params,
):
    """Run the vendored CAMEX cross-species integration pipeline.

    Parameters
    ----------
    adatas
        List of per-species :class:`anndata.AnnData` objects holding **raw
        counts** in ``.X`` and gene symbols in ``.var_names``.
    species
        List of species names (parallel to ``adatas``). Used as the on-disk
        ``.h5ad`` file stems and the ``species`` / ``batch`` obs column.
    homology
        Optional dict ``{(src_species, dst_species): DataFrame}`` where each
        DataFrame's first two columns are ``[src_gene_symbol, dst_gene_symbol]``
        (a many-to-many homology table). If ``None``, every non-reference
        species is star-linked to the reference species and the caller must
        instead supply homology via this argument -- with ``None`` and no
        homology CAMEX cannot connect the species, so a value is required unless
        a single species is passed.
    ref_index
        Index (into ``adatas`` / ``species``) of the reference/annotated
        species. Its labels are marked available (``source label=True``).
    label_col
        ``obs`` column holding cell-type labels; copied to
        ``cell_ontology_class`` which CAMEX requires. Missing -> ``'unknown'``.
    device
        ``'cuda'`` / ``'cuda:0'`` / ``'cpu'``.
    epoch_integration, epoch_annotation, batch_size
        Training hyper-parameters.
    work_dir
        Directory used to stage the ``dataset/`` h5ads, homology CSVs and CAMEX
        output logs.
    annotate
        If True (and reference labels are present) also run
        ``Trainer.annotation()`` after integration, writing
        ``obsm['X_CAMEX_Annotation']``.
    **params
        Extra keys merged into the CAMEX ``PARAMS['train']`` dict
        (e.g. ``dim_hidden``, ``encoder``, ``classifier``, ``gnn_layer_num``).

    Returns
    -------
    anndata.AnnData
        The combined ``adata_whole`` with the integration embedding in
        ``.obsm['X_CAMEX_Integration']`` and ``species`` / ``batch`` in
        ``.obs``.
    """
    # Lazy heavy imports so ``import omicverse.external.camex`` stays light.
    from .base import Dataset
    from .trainer import Trainer

    if len(adatas) != len(species):
        raise ValueError("`adatas` and `species` must have the same length.")
    if len(adatas) < 2:
        raise ValueError("CAMEX needs at least two species to integrate.")
    if homology is None:
        raise ValueError(
            "`homology` is required: pass a dict {(src_species, dst_species): "
            "DataFrame[[src_gene, dst_gene]]} linking the species. With None "
            "CAMEX cannot build the cross-species gene graph."
        )

    species = [str(s) for s in species]
    ref_species = species[ref_index]

    # ---- work dirs ------------------------------------------------------
    work_dir = os.path.abspath(work_dir)
    dataset_dir = os.path.join(work_dir, "dataset") + os.sep
    time_start = time.strftime("%Y-%m-%d-%H-%M-%S")
    log_path = os.path.join(work_dir, "log", time_start) + os.sep
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # ---- write per-species h5ad ----------------------------------------
    fname = {sp: f"{sp}.h5ad" for sp in species}
    has_label = {}
    for sp, ad in zip(species, adatas):
        prepared = _prepare_adata(ad, label_col)
        # a species is "labelled" (reference-eligible) if real labels exist
        if label_col is not None and label_col in ad.obs.columns:
            has_label[sp] = True
        else:
            has_label[sp] = False
        prepared.write_h5ad(dataset_dir + fname[sp], compression="gzip")

    # ref is always marked as labelled reference (its labels drive annotation)
    has_label[ref_species] = has_label.get(ref_species, False) or (
        label_col is not None and label_col in adatas[ref_index].obs.columns
    )

    # ---- build homology CSVs + dataset_file ----------------------------
    # CAMEX's `_check_dataset_exist` asserts the homology CSV's two column
    # headers are exactly the source and destination *.h5ad filenames.
    rows = []
    for (src_sp, dst_sp), df in homology.items():
        src_sp, dst_sp = str(src_sp), str(dst_sp)
        if src_sp not in fname or dst_sp not in fname:
            raise ValueError(
                f"homology key ({src_sp!r}, {dst_sp!r}) references unknown "
                f"species; known: {species}"
            )
        pair = df.iloc[:, :2].copy()
        pair.columns = [fname[src_sp], fname[dst_sp]]
        csv_name = f"gene_matches_{src_sp}2{dst_sp}.csv"
        pair.to_csv(dataset_dir + csv_name, index=False)
        rows.append(
            [
                fname[src_sp],
                bool(has_label[src_sp]),
                csv_name,
                fname[dst_sp],
                bool(has_label[dst_sp]),
            ]
        )

    dataset_file = pd.DataFrame(
        data=rows,
        columns=["source", "source label", "relationship", "destination", "destination label"],
    )

    # ---- merged PARAMS --------------------------------------------------
    PARAMS = _default_params()
    PARAMS["preprocess"]["path"] = dataset_dir
    PARAMS["preprocess"]["dataset_file"] = dataset_file

    train = PARAMS["train"]
    train["device"] = device
    train["epoch_integration"] = int(epoch_integration)
    train["epoch_annotation"] = int(epoch_annotation)
    train["batch_size"] = int(batch_size)
    # merge user overrides (dim_hidden, encoder, classifier, gnn_layer_num, ...)
    for k, v in params.items():
        if k in PARAMS["preprocess"]:
            PARAMS["preprocess"][k] = v
        else:
            train[k] = v

    # CAMEX expects every sub-dict to carry time_start / log_path.
    for v in PARAMS.values():
        if isinstance(v, dict):
            v["time_start"] = time_start
            v["log_path"] = log_path

    # ---- run pipeline (faithful to the tutorial glue) ------------------
    dataset = Dataset(**PARAMS["preprocess"])
    adata_camex = dataset.adata_whole
    dgl_data = dataset.dgl_data

    trainer = Trainer(adata_camex, dgl_data, **PARAMS["train"])
    trainer.integration()

    if annotate and int(epoch_annotation) > 0 and bool(has_label[ref_species]):
        trainer.annotation()

    # ---- tidy obs -------------------------------------------------------
    # ``batch`` is set by CAMEX to the per-species data order; mirror it into
    # a ``species`` column for convenience.
    if "batch" in adata_camex.obs.columns:
        adata_camex.obs["species"] = adata_camex.obs["batch"].astype(str).values

    if "X_CAMEX_Integration" not in adata_camex.obsm:
        raise RuntimeError(
            "CAMEX integration finished but no 'X_CAMEX_Integration' embedding "
            "was written to adata.obsm -- check the training logs."
        )

    return adata_camex

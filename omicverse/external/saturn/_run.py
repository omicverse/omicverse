"""Importable wrapper around SATURN's ``train-saturn.py`` CLI script.

This module exposes :func:`run_saturn`, an in-memory orchestration of the
SATURN training pipeline (macrogene pretraining + metric learning). It reuses
the algorithm functions defined in the vendored ``_train_saturn`` module
(``pretrain_saturn``, ``train``, ``get_all_embeddings``,
``get_all_embeddings_metric``, ``create_output_anndata``, ``make_centroids``)
rather than reimplementing them, so behaviour stays faithful to upstream.

See the upstream reference: Rosen et al., "Toward universal cell embeddings:
integrating single-cell RNA-seq datasets across species with SATURN",
Nature Methods (2023). https://github.com/snap-stanford/SATURN
"""

from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.optim as optim

from . import distances, losses, miners, reducers
from .data.gene_embeddings import load_gene_embeddings_adata
from .data.multi_species_data import (
    ExperimentDatasetMulti,
    ExperimentDatasetMultiEqual,
    multi_species_collate_fn,
)
from .model.saturn_model import (
    SATURNMetricModel,
    SATURNPretrainModel,
    make_centroids,
)
from ._train_saturn import (
    create_output_anndata,
    get_all_embeddings,
    get_all_embeddings_metric,
    pretrain_saturn,
    train,
)


def run_saturn(
    adatas: List["sc.AnnData"],
    species: List[str],
    embedding_paths: Dict[str, str],
    *,
    in_label_col: str = "cell_type",
    ref_label_col: Optional[str] = None,
    embedding_model: str = "ESM2",
    n_macrogenes: int = 2000,
    hidden_dim: int = 256,
    model_dim: int = 256,
    hv_genes: int = 8000,
    hv_span: float = 0.3,
    epochs: Sequence[int] = (30, 50),
    pretrain: bool = True,
    vae: bool = False,
    l1_penalty: float = 0.0,
    pe_sim_penalty: float = 0.2,
    centroid_score_func: str = "default",
    pretrain_lr: float = 5e-4,
    metric_lr: float = 1e-3,
    pretrain_batch_size: int = 4096,
    batch_size: int = 4096,
    mnn: bool = True,
    use_ref_labels: bool = False,
    unfreeze_macrogenes: bool = False,
    equalize_triplets_species: bool = False,
    balance_pretrain: bool = False,
    non_species_batch_col: Optional[str] = None,
    seed: int = 0,
    device: str = "cuda",
    work_dir: str = "./saturn_work",
    centroids_init_path: Optional[str] = None,
    **kw,
) -> "sc.AnnData":
    """Run the SATURN cross-species integration pipeline programmatically.

    Parameters
    ----------
    adatas
        One :class:`anndata.AnnData` of **raw counts** per species. Each
        ``adata.var_names`` must be gene symbols matching the keys of that
        species' protein-embedding file. ``adata.obs`` must contain
        ``in_label_col``.
    species
        Species names, one per entry of ``adatas`` and in the same order. These
        are the keys used in ``embedding_paths``.
    embedding_paths
        Mapping ``species_name -> path`` to that species' precomputed protein
        (ESM) embedding file. Each file is a ``torch.load``-able dict mapping
        ``gene_symbol -> 1D FloatTensor`` (the ESM embedding vector). This is the
        ``gene_symbol_to_embedding_<model>.pt`` format produced by SATURN's
        ``protein_embeddings`` scripts.
    in_label_col
        Column in each ``adata.obs`` holding the (species-specific) cell-type
        label. Used to build the supervised macrogene / metric-learning labels.
    ref_label_col
        Column holding coarse reference labels shared across species (used when
        ``use_ref_labels=True``). Defaults to ``in_label_col`` if ``None``.
    embedding_model
        Which embedding model the ``embedding_paths`` correspond to (one of
        ``'ESM1b','MSA1b','protXL','ESM1b_protref','ESM2'``). Only used for
        bookkeeping since explicit ``embedding_paths`` are supplied.
    n_macrogenes
        Number of macrogenes (shared latent gene clusters).
    hidden_dim, model_dim
        Encoder hidden dimension and latent-embedding dimension.
    hv_genes, hv_span
        Highly-variable-gene count and loess span (scanpy ``seurat_v3``).
    epochs
        ``(pretrain_epochs, metric_epochs)`` tuple.
    pretrain
        Whether to run the pretraining (macrogene autoencoder) phase.
    vae
        Use a VAE latent layer instead of a plain autoencoder.
    non_species_batch_col
        Optional extra ``obs`` column used as a categorical batch covariate
        during pretraining (present in every ``adata``).
    device
        Torch device string, e.g. ``'cuda'`` or ``'cpu'``.
    work_dir
        Directory for intermediate artifacts (centroids/genes-to-macrogenes).
    centroids_init_path
        Optional path to load/save the macrogene centroid weights.
    **kw
        Ignored extra keyword arguments (accepted for forward compatibility).

    Returns
    -------
    anndata.AnnData
        Combined AnnData of all species. The SATURN latent embedding is stored
        both as ``adata.X`` and ``adata.obsm['X_saturn']``; macrogene values are
        in ``adata.obsm['macrogenes']``; ``adata.obs['species']`` records the
        species and ``adata.obs['labels']`` the (species-prefixed) cell type.
    """
    if len(adatas) != len(species):
        raise ValueError("`adatas` and `species` must have the same length")
    if ref_label_col is None:
        ref_label_col = in_label_col

    pretrain_epochs, metric_epochs = int(epochs[0]), int(epochs[1])

    # Seeds (mirrors the CLI script).
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    device_t = torch.device(device)
    work_dir_p = Path(work_dir)
    work_dir_p.mkdir(parents=True, exist_ok=True)

    species_to_adata = {sp: ad for sp, ad in zip(species, adatas)}
    species_to_embedding_paths = {sp: embedding_paths[sp] for sp in species}

    use_batch_labels = non_species_batch_col is not None

    # --- Build species-prefixed cell-type labels ------------------------------
    all_obs_names: List[str] = []
    for sp, adata in species_to_adata.items():
        species_str = pd.Series([sp] * adata.obs.shape[0])
        species_str.index = adata.obs[in_label_col].index
        adata.obs["species"] = species_str
        adata.obs["species_type_label"] = species_str.str.cat(
            adata.obs[in_label_col].astype(str), sep="_"
        )
        all_obs_names += list(adata.obs_names)

    unique_cell_types = set()
    for adata in species_to_adata.values():
        unique_cell_types |= set(adata.obs["species_type_label"])
    unique_cell_types = sorted(unique_cell_types)
    celltype_id_map = {ct: i for i, ct in enumerate(unique_cell_types)}
    for adata in species_to_adata.values():
        adata.obs["truth_labels"] = pd.Categorical(
            values=[celltype_id_map[ct] for ct in adata.obs["species_type_label"]]
        )

    # --- Optional batch labels ------------------------------------------------
    num_batch_labels = 0
    batchtype_id_map = None
    unique_batch_types = None
    if use_batch_labels:
        unique_batch_types = set()
        for adata in species_to_adata.values():
            unique_batch_types |= set(adata.obs[non_species_batch_col])
        unique_batch_types = sorted(unique_batch_types)
        batchtype_id_map = {bt: i for i, bt in enumerate(unique_batch_types)}
        for adata in species_to_adata.values():
            adata.obs["batch_labels"] = pd.Categorical(
                values=[batchtype_id_map[bt] for bt in adata.obs[non_species_batch_col]]
            )
        num_batch_labels = len(unique_batch_types)

    # --- Reference labels -----------------------------------------------------
    unique_ref_types = set()
    for adata in species_to_adata.values():
        unique_ref_types |= set(adata.obs[ref_label_col])
    unique_ref_types = sorted(unique_ref_types)
    reftype_id_map = {rt: i for i, rt in enumerate(unique_ref_types)}
    for adata in species_to_adata.values():
        adata.obs["ref_labels"] = pd.Categorical(
            values=[reftype_id_map[rt] for rt in adata.obs[ref_label_col]]
        )

    # --- Load gene embeddings (also filters genes to those with embeddings) ---
    species_to_gene_embeddings = {}
    for sp, adata in species_to_adata.items():
        adata, sp_gene_emb = load_gene_embeddings_adata(
            adata=adata,
            species=[sp],
            embedding_model=embedding_model,
            embedding_path=species_to_embedding_paths[sp],
        )
        species_to_gene_embeddings.update(sp_gene_emb)
        species_to_adata[sp] = adata

    sorted_species_names = sorted(species_to_gene_embeddings.keys())

    # --- Highly variable genes ------------------------------------------------
    species_to_gene_idx_hv = {}
    ct = 0
    for sp in sorted_species_names:
        adata = species_to_adata[sp]
        if use_batch_labels:
            sc.pp.highly_variable_genes(
                adata, flavor="seurat_v3", n_top_genes=hv_genes,
                batch_key=non_species_batch_col, span=hv_span,
            )
        else:
            sc.pp.highly_variable_genes(
                adata, flavor="seurat_v3", n_top_genes=hv_genes, span=hv_span,
            )
        hv_index = adata.var["highly_variable"]
        species_to_adata[sp] = adata[:, hv_index]
        species_to_gene_embeddings[sp] = species_to_gene_embeddings[sp][hv_index]
        n = species_to_gene_embeddings[sp].shape[0]
        species_to_gene_idx_hv[sp] = (ct, ct + n)
        ct += n

    # --- Macrogene centroids --------------------------------------------------
    X = torch.cat([species_to_gene_embeddings[sp] for sp in sorted_species_names])

    all_gene_names: List[str] = []
    for sp in sorted_species_names:
        adata = species_to_adata[sp]
        species_str = pd.Series([sp] * adata.var_names.shape[0])
        gene_names = pd.Series(adata.var_names)
        all_gene_names += list(species_str.str.cat(gene_names, sep="_"))

    species_genes_scores = None
    if centroids_init_path is not None:
        try:
            import pickle
            with open(centroids_init_path, "rb") as f:
                species_genes_scores = pickle.load(f)
        except Exception:
            species_genes_scores = None
    if species_genes_scores is None:
        species_genes_scores = make_centroids(
            X, all_gene_names, n_macrogenes, seed=seed,
            score_function=centroid_score_func,
        )
        if centroids_init_path is not None:
            import pickle
            with open(centroids_init_path, "wb") as f:
                pickle.dump(species_genes_scores, f, protocol=4)

    centroid_weights = []
    all_species_gene_names: List[str] = []
    for sp in sorted_species_names:
        adata = species_to_adata[sp]
        species_str = pd.Series([sp] * adata.var_names.shape[0])
        gene_names = pd.Series(adata.var_names)
        sp_gene_names = list(species_str.str.cat(gene_names, sep="_"))
        all_species_gene_names += sp_gene_names
        for sgn in sp_gene_names:
            centroid_weights.append(torch.tensor(species_genes_scores[sgn]))
    centroid_weights = torch.stack(centroid_weights)

    # --- Datasets -------------------------------------------------------------
    if use_batch_labels:
        batch_labs = {sp: ad.obs["batch_labels"] for sp, ad in species_to_adata.items()}
    else:
        batch_labs = {}
    ys = {sp: ad.obs["truth_labels"] for sp, ad in species_to_adata.items()}
    refs = {sp: ad.obs["ref_labels"] for sp, ad in species_to_adata.items()}

    dataset = ExperimentDatasetMultiEqual(
        all_data=species_to_adata, all_ys=ys, all_refs=refs, all_batch_labs=batch_labs
    )
    test_dataset = ExperimentDatasetMulti(
        all_data=species_to_adata, all_ys=ys, all_refs=refs, all_batch_labs=batch_labs
    )

    sorted_batch_labels_names = list(unique_batch_types) if use_batch_labels else None

    # --- Pretraining ----------------------------------------------------------
    pretrain_model = SATURNPretrainModel(
        gene_scores=centroid_weights, hidden_dim=hidden_dim, embed_dim=model_dim,
        dropout=0.1, species_to_gene_idx=species_to_gene_idx_hv, vae=vae,
        sorted_batch_labels_names=sorted_batch_labels_names,
        l1_penalty=l1_penalty, pe_sim_penalty=pe_sim_penalty,
    ).to(device_t)

    if pretrain:
        optim_pretrain = optim.Adam(pretrain_model.parameters(), lr=pretrain_lr)
        pretrain_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=multi_species_collate_fn,
            batch_size=pretrain_batch_size, shuffle=True,
        )
        pretrain_model = pretrain_saturn(
            pretrain_model, pretrain_loader, optim_pretrain, device,
            pretrain_epochs, sorted_species_names, balance=balance_pretrain,
            use_batch_labels=use_batch_labels, embeddings_tensor=X.to(device_t),
        )

    # Get macrogene values from the pretrained model.
    if use_batch_labels:
        (train_emb, train_lab, train_species, train_macrogenes,
         train_ref, train_batch) = get_all_embeddings(
            test_dataset, pretrain_model, device, use_batch_labels)
    else:
        train_batch = None
        (train_emb, train_lab, train_species, train_macrogenes,
         train_ref) = get_all_embeddings(
            test_dataset, pretrain_model, device, use_batch_labels)

    # --- Metric learning ------------------------------------------------------
    metric_dataset = dataset
    test_metric_dataset = test_dataset
    if not unfreeze_macrogenes:
        metric_dataset.num_genes = {sp: train_macrogenes.shape[1] for sp in sorted_species_names}
        c = 0
        for sp in sorted_species_names:
            n_cells = metric_dataset.xs[sp].shape[0]
            metric_dataset.xs[sp] = train_macrogenes[c:c + n_cells, :]
            c += n_cells
        test_metric_dataset.num_genes = {sp: train_macrogenes.shape[1] for sp in sorted_species_names}
        c = 0
        for sp in sorted_species_names:
            n_cells = test_metric_dataset.xs[sp].shape[0]
            test_metric_dataset.xs[sp] = train_macrogenes[c:c + n_cells, :]
            c += n_cells

        pretrain_model = pretrain_model.cpu()
        metric_model = SATURNMetricModel(
            input_dim=train_macrogenes.shape[1], hidden_dim=hidden_dim,
            embed_dim=model_dim, dropout=0.1,
            species_to_gene_idx=species_to_gene_idx_hv, vae=vae,
        )
        if pretrain_model.vae:
            metric_model.fc_mu = deepcopy(pretrain_model.fc_mu)
            metric_model.fc_var = deepcopy(pretrain_model.fc_var)
        metric_model.cl_layer_norm = deepcopy(pretrain_model.cl_layer_norm)
        metric_model.encoder = deepcopy(pretrain_model.encoder)
    else:
        metric_model = pretrain_model
        metric_model.metric_learning_mode = True

    metric_model.to(device_t)
    optimizer = optim.Adam(metric_model.parameters(), lr=metric_lr)

    train_loader = torch.utils.data.DataLoader(
        metric_dataset, collate_fn=multi_species_collate_fn,
        batch_size=batch_size, shuffle=True,
    )

    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(
        margin=0.2, distance=distance, type_of_triplets="semihard",
        miner_type="cross_species",
    )

    for epoch in range(1, metric_epochs + 1):
        train(
            metric_model, loss_func, mining_func, device, train_loader,
            optimizer, epoch, mnn, sorted_species_names,
            use_ref_labels=use_ref_labels, indices_counts={},
            equalize_triplets_species=equalize_triplets_species,
        )

    # --- Final embeddings -> AnnData -----------------------------------------
    if use_batch_labels:
        (train_emb, train_lab, train_species, train_macrogenes,
         train_ref, train_batch) = get_all_embeddings_metric(
            test_dataset, metric_model, device, use_batch_labels)
        adata = create_output_anndata(
            train_emb, train_lab, train_species, train_macrogenes.cpu().numpy(),
            train_ref, celltype_id_map, reftype_id_map, use_batch_labels,
            batchtype_id_map, train_batch, obs_names=all_obs_names,
        )
    else:
        (train_emb, train_lab, train_species, train_macrogenes,
         train_ref) = get_all_embeddings_metric(
            test_metric_dataset, metric_model, device, use_batch_labels)
        adata = create_output_anndata(
            train_emb, train_lab, train_species, train_macrogenes.cpu().numpy(),
            train_ref, celltype_id_map, reftype_id_map, obs_names=all_obs_names,
        )

    # Expose the SATURN embedding in obsm (X already holds it).
    adata.obsm["X_saturn"] = adata.X.copy()
    return adata

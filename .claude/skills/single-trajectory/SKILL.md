---
name: single-trajectory-analysis
title: Single-trajectory analysis
description: "Trajectory & RNA velocity: PAGA, Palantir, VIA, dynamo, scVelo, latentvelo, graphvelo backends via ov.single.Velo. Pseudotime, stream plots."
---

# Single-trajectory analysis skill

## Overview
This skill describes how to reproduce and extend the single-trajectory analysis workflow in `omicverse`, combining graph-based trajectory inference, RNA velocity coupling, and downstream fate scoring notebooks.

## Trajectory setup
- **PAGA (Partition-based graph abstraction)**
  - Build a neighborhood graph (`pp.neighbors`) on the preprocessed AnnData object.
  - Use `tl.paga` to compute cluster connectivity and `tl.draw_graph` or `tl.umap` with `init_pos='paga'` for embedding.
  - Interpret edge weights to prioritize branch resolution and seed paths.
- **Palantir**
  - Run `Palantir` on diffusion components, seeding with manually selected start cells (e.g., naïve T cells).
  - Extract pseudotime, branch probabilities, and differentiation potential for subsequent overlays.
- **VIA**
  - Execute `via.VIA` on the kNN graph to identify lineage progression with automatic root selection or user-defined roots.
  - Export terminal states and pseudotime for cross-validation against PAGA and Palantir results.

## Velocity coupling (VIA + scVelo)
- Use `scv.pp.filter_and_normalize`, `scv.pp.moments`, and `scv.tl.velocity` to generate velocity layers.
- Provide VIA with `adata.layers['velocity']` to refine lineage directionality (`via.VIA(..., velocity_weight=...)`).
- Compare VIA pseudotime with scVelo latent time (`scv.tl.latent_time`) to validate directionality and root selection.

## Advanced RNA Velocity Backends (ov.single.Velo)

OmicVerse provides a unified `Velo` class wrapping 4 velocity backends. Use this when you need more than basic scVelo:

### Backend selection guide

| Backend | Best for | GPU? | Prerequisites |
|---------|----------|------|---------------|
| **scvelo** | Standard velocity analysis | No | spliced/unspliced layers |
| **dynamo** | Kinetics modeling, vector fields | No | spliced/unspliced layers |
| **latentvelo** | VAE-based, batch correction, complex dynamics | Yes (torchdiffeq) | celltype_key, batch_key optional |
| **graphvelo** | Refinement layer on top of any backend | No | base velocity + connectivities |

### Unified Velo pipeline

```python
import omicverse as ov

velo = ov.single.Velo(adata)

# 1. Filter (scvelo backend) or preprocess (dynamo backend)
velo.filter_genes(min_shared_counts=20)     # For scvelo
# velo.preprocess(recipe='monocle', n_neighbors=30, n_pcs=30)  # For dynamo

# 2. Compute moments
velo.moments(backend='scvelo', n_pcs=30, n_neighbors=30)
# backend: 'scvelo' or 'dynamo'

# 3. Fit kinetic parameters
velo.dynamics(backend='scvelo')

# 4. Calculate velocity
velo.cal_velocity(method='scvelo')
# method: 'scvelo', 'dynamo', 'latentvelo', 'graphvelo'

# 5. Build velocity graph and project to embedding
velo.velocity_graph(basis='umap')
velo.velocity_embedding(basis='umap')
```

### latentvelo specifics (deep learning velocity)

latentvelo uses a VAE + neural ODE to learn latent dynamics. It handles batch effects and complex trajectories better than classical scVelo:

```python
velo.cal_velocity(
    method='latentvelo',
    celltype_key='cell_type',    # Optional: AnnotVAE uses cell type info
    batch_key='batch',           # Optional: batch correction
    velocity_key='velocity_S',
    n_top_genes=2000,
    latentvelo_VAE_kwargs={},    # Pass custom VAE hyperparameters
)
# Requires: pip install torchdiffeq
# Uses GPU if available, falls back to CPU
```

### graphvelo specifics (refinement layer)

GraphVelo refines velocity estimates from any base method by leveraging the cell graph structure. Run it after scvelo or dynamo:

```python
# First: compute base velocity with scvelo or dynamo
velo.cal_velocity(method='scvelo')

# Then: refine with graphvelo
velo.graphvelo(
    xkey='Ms',                          # Spliced moments key
    vkey='velocity_S',                  # Base velocity key to refine
    basis_keys=['X_umap', 'X_pca'],    # Project to multiple embeddings
    gene_subset=None,                   # Optional: restrict to gene subset
)
```

## Downstream fate scoring notebooks
- **CellFateGenie**: For pseudotime-associated gene discovery, use `search_skills('CellFateGenie fate genes')` to load the dedicated CellFateGenie skill.
- **`t_metacells.ipynb`**: Aggregate metacell trajectories for robustness checks and meta-state differential expression.

## Required preprocessing
1. Quality control: remove low-quality cells/genes, apply doublet filtering.
2. Normalization & log transformation (`sc.pp.normalize_total`, `sc.pp.log1p`).
3. Highly variable gene selection tailored to immune datasets (`sc.pp.highly_variable_genes`).
4. Batch correction if necessary (e.g., `scvi-tools`, `bbknn`).
5. Compute PCA, neighbor graph, and embedding (UMAP/FA) used by all trajectory methods.
6. For velocity: compute moments on the same neighbor graph before running VIA coupling.

## Parameter tuning
- Neighbor graph `n_neighbors` and `n_pcs` should be harmonized across PAGA, VIA, and Palantir to maintain consistency.
- In VIA, adjust `knn`, `too_big_factor`, and `root_user` for datasets with uneven sampling.
- Palantir requires careful start cell selection; use marker genes and velocity arrows to confirm.
- For PAGA, tweak `threshold` to control edge sparsity; ensure connected components reflect biological branches.
- Velocity estimation: compare `mode='stochastic'` vs `mode='dynamical'` in scVelo; recalibrate if terminal states disagree with VIA.

## Visualization and export
1. Overlay PAGA edges on UMAP (`scv.pl.paga`) and annotate branch labels.
2. Plot Palantir pseudotime and branch probabilities on embeddings.
3. Visualize VIA trajectories using `via.plot_fates` and `via.plot_scatter`.
4. Export pseudotime tables and fate probabilities to CSV for downstream notebooks.
5. Save high-resolution figures (PNG/SVG) and notebook artifacts for reproducibility.
6. Update notebooks with consistent color schemes and metadata columns before sharing.

## Defensive Validation Patterns

```python
# Before PAGA: verify neighbor graph exists
assert 'neighbors' in adata.uns, "Neighbor graph required. Run sc.pp.neighbors(adata) first."

# Before VIA velocity coupling: verify velocity layers exist
if 'velocity' not in adata.layers:
    print("WARNING: velocity layer missing. Run scv.tl.velocity(adata) first for VIA coupling.")
assert 'spliced' in adata.layers and 'unspliced' in adata.layers, \
    "Missing spliced/unspliced layers. Check loom/H5AD import preserved velocity layers."

# Before Palantir: verify PCA/diffusion components
assert 'X_pca' in adata.obsm, "PCA required. Run ov.pp.pca(adata) first."
```

## Troubleshooting tips
- **Missing velocity layers**: re-run `scv.pp.moments` and `scv.tl.velocity` ensuring `adata.layers['spliced']`/`['unspliced']` exist; verify loom/H5AD import preserved layers.
- **Disconnected PAGA graph**: inspect neighbor graph or adjust `n_neighbors`; confirm batch correction didn’t fragment the manifold.
- **Palantir convergence issues**: reduce diffusion components or reinitialize start cells; ensure no NaN values in data matrix.
- **VIA terminal states unstable**: increase iterations (`cluster_graph_pruning_iter`), or provide manual terminal state hints based on marker expression.
- **Notebook kernel memory errors**: downsample cells or precompute summaries (metacells) before rerunning.
- **latentvelo `ImportError: torchdiffeq`**: Install with `pip install torchdiffeq`. Required for neural ODE backend.
- **graphvelo returns NaN velocities**: Ensure base velocity (scvelo/dynamo) was computed first. graphvelo refines — it doesn't compute from scratch.
- **dynamo `preprocess` fails**: dynamo expects spliced/unspliced layers. Verify with `'spliced' in adata.layers`.

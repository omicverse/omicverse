---
name: single-cell-multi-omics-integration
title: Single-cell multi-omics integration
description: "Multi-omics integration: MOFA factor analysis, GLUE unpaired alignment, SIMBA batch correction, TOSICA label transfer, StaVIA trajectory. Covers scRNA+scATAC paired/unpaired workflows."
---

# Single-Cell Multi-Omics Integration

This skill covers OmicVerse's multi-omics integration tools for combining scRNA-seq, scATAC-seq, and other modalities. Each method addresses a different scenario—choose based on your data structure and analysis goal.

## Method Selection Guide

Pick the right tool before writing any code:

| Scenario | Method | Key Class |
|----------|--------|-----------|
| Paired RNA + ATAC from same cells | MOFA directly | `ov.single.pyMOFA` |
| Unpaired RNA + ATAC (different experiments) | GLUE pairing → MOFA | `ov.single.GLUE_pair` → `pyMOFA` |
| Multi-batch single-modality integration | SIMBA | `ov.single.pySIMBA` |
| Transfer labels from annotated reference | TOSICA | `ov.single.pyTOSICA` |
| Trajectory on preprocessed multi-omic data | StaVIA | `VIA.core.VIA` |

## Instructions

### 1. MOFA on paired multi-omics

Use MOFA when you have paired measurements (RNA + ATAC from the same cells). MOFA learns shared and modality-specific factors that explain variance across omics layers.

1. Load each modality as a **separate AnnData** object
2. Initialise `pyMOFA` with matching `omics` and `omics_name` lists
3. Run `mofa_preprocess()` to select HVGs, then `mofa_run(outfile=...)` to train
4. Inspect factors with `pyMOFAART(model_path=...)` for correlation, weights, and variance plots
5. Dependencies: `mofapy2`; CPU-only

### 2. GLUE pairing then MOFA

Use GLUE when RNA and ATAC come from different experiments (unpaired). GLUE aligns cells across modalities by learning a shared embedding, then MOFA identifies joint factors.

1. Start from GLUE-derived embeddings (`.h5ad` files with embeddings in `.obsm`)
2. Build `GLUE_pair` and call `correlation()` to match unpaired cells
3. Subset to HVGs and run MOFA as in the paired workflow
4. Dependencies: `mofapy2`, `scglue`, `scvi-tools`; GPU optional for MDE embedding

### 3. SIMBA batch integration

Use SIMBA for multi-batch single-modality data (e.g., multiple pancreas studies). SIMBA builds a graph from binned features and learns batch-corrected embeddings via PyTorch-BigGraph.

1. Load concatenated AnnData with a `batch` column in `.obs`
2. Initialise `pySIMBA(adata, workdir)` and run the preprocessing pipeline
3. Call `gen_graph()` then `train(num_workers=...)` to learn embeddings
4. Apply `batch_correction()` to get harmonised AnnData with `X_simba`
5. Dependencies: none (native `ov.ge.SIMBA`, pure PyTorch -- no `simba`/`simba_pbg` install); GPU optional

### 4. TOSICA reference transfer

Use TOSICA to transfer cell-type labels from a well-annotated reference to a query dataset. TOSICA uses a pathway-masked transformer that also provides attention-based interpretability.

1. Download gene-set GMT files with `ov.utils.download_tosica_gmt()`
2. Initialise `pyTOSICA` with reference AnnData, GMT path, label key, and project path
3. Train with `train(epochs=...)`, save, then predict on query data
4. Dependencies: TOSICA (PyTorch transformer); `depth=1` recommended (depth=2 doubles memory)

### 5. StaVIA trajectory cartography

Use StaVIA/VIA for trajectory inference on preprocessed data with velocity information. VIA computes pseudotime, cluster graphs, and stream plots.

1. Preprocess with OmicVerse (HVGs, scale, PCA, neighbors, UMAP)
2. Configure VIA with root selection, components, neighbors, and resolution
3. Run `v0.run_VIA()` and extract pseudotime from `single_cell_pt_markov`
4. Dependencies: `scvelo`, `pyVIA`; CPU-bound

## Critical API Reference

### MOFA: `omics` must be a list of separate AnnData objects

```python
# CORRECT — each modality is a separate AnnData
mofa = ov.single.pyMOFA(omics=[rna_adata, atac_adata], omics_name=['RNA', 'ATAC'])

# WRONG — do NOT pass a single concatenated AnnData
# mofa = ov.single.pyMOFA(omics=combined_adata, omics_name=['RNA', 'ATAC'])  # TypeError!
```

The `omics` list and `omics_name` list must have the same length. Each AnnData should contain cells from the same experiment (paired measurements).

### SIMBA: `preprocess()` must run before `gen_graph()`

```python
# CORRECT — preprocess first, then build graph
simba = ov.single.pySIMBA(adata, workdir)
simba.preprocess(batch_key='batch', min_n_cells=3, method='lib_size', n_top_genes=3000, n_bins=5)
simba.gen_graph()
simba.train(num_workers=6)

# WRONG — skipping preprocess causes gen_graph to fail
# simba.gen_graph()  # KeyError: missing binned features
```

### TOSICA: `gmt_path` must be an actual file path

```python
# CORRECT — download GMT files first, then pass the file path
ov.utils.download_tosica_gmt()
tosica = ov.single.pyTOSICA(adata=ref, gmt_path='genesets/GO_bp.gmt', ...)

# WRONG — passing a database name string instead of file path
# tosica = ov.single.pyTOSICA(adata=ref, gmt_path='GO_Biological_Process', ...)  # FileNotFoundError!
```

### MOFA HDF5: `outfile` directory must exist

```python
import os
os.makedirs('models', exist_ok=True)  # Create output directory first
mofa.mofa_run(outfile='models/rna_atac.hdf5')
```

## Defensive Validation Patterns

Always validate inputs before running integration methods:

```python
# Before MOFA: verify inputs are compatible
assert isinstance(omics, list), "omics must be a list of AnnData objects"
assert len(omics) == len(omics_name), f"omics ({len(omics)}) and omics_name ({len(omics_name)}) must match in length"
for i, a in enumerate(omics):
    assert a.n_obs > 0, f"AnnData '{omics_name[i]}' has 0 cells"
    assert a.n_vars > 0, f"AnnData '{omics_name[i]}' has 0 genes/features"

# Before SIMBA: verify batch column exists
assert 'batch' in adata.obs.columns, "adata.obs must contain a 'batch' column for SIMBA"
assert adata.obs['batch'].nunique() > 1, "Need >1 batch for batch integration"

# Before TOSICA: verify GMT file exists and reference has labels
import os
assert os.path.isfile(gmt_path), f"GMT file not found: {gmt_path}. Run ov.utils.download_tosica_gmt() first."
assert label_name in ref_adata.obs.columns, f"Label column '{label_name}' not found in reference AnnData"

# Before StaVIA: verify PCA and neighbors are computed
assert 'X_pca' in adata.obsm, "PCA required. Run ov.pp.pca(adata) first."
assert 'neighbors' in adata.uns, "Neighbor graph required. Run ov.pp.neighbors(adata) first."
```

## Troubleshooting

- **`PermissionError` or `OSError` writing MOFA HDF5**: The output directory for `mofa_run(outfile=...)` must exist and be writable. Create it with `os.makedirs()` before training.
- **GLUE `correlation()` returns empty DataFrame**: The RNA and ATAC embeddings have no overlapping features. Verify both AnnData objects have been through GLUE preprocessing and contain embeddings in `.obsm`.
- **SIMBA `gen_graph()` runs out of memory**: Reduce `n_top_genes` (try 2000) or increase `n_bins` to compress the feature space. SIMBA graph construction scales with gene count.
- **TOSICA `FileNotFoundError` after `download_tosica_gmt()`**: The download writes to `genesets/` in the current working directory. Verify the file exists at the expected path, or pass an absolute path.
- **StaVIA `root_user` mismatch**: The root must be a value that exists in the `true_label` array. Check `adata.obs['clusters'].unique()` to find valid root names.
- **`ImportError: No module named 'mofapy2'`**: Install with `pip install mofapy2`. SIMBA (`ov.single.pySIMBA`) needs no separate install -- it runs on omicverse's own `ov.ge` graph-embedding package (pure PyTorch).
- **MOFA factors all zero or NaN**: Input AnnData may have constant or all-zero features. Filter genes with `sc.pp.filter_genes(adata, min_cells=10)` before MOFA.

## Examples
- "I have paired scRNA and scATAC h5ad files—run MOFA to find shared factors and plot variance explained per factor."
- "Integrate three pancreas batches using SIMBA and visualise the corrected embedding coloured by batch and cell type."
- "Transfer cell type labels from my annotated reference to a new query dataset using TOSICA with GO biological process pathways."

## References
- MOFA tutorial: `t_mofa.ipynb`
- GLUE+MOFA tutorial: `t_mofa_glue.ipynb`
- SIMBA tutorial: `t_simba.ipynb`
- TOSICA tutorial: `t_tosica.ipynb`
- StaVIA tutorial: `t_stavia.ipynb`
- Quick copy/paste commands: [`reference.md`](reference.md)

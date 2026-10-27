---
name: spatial-transcriptomics-tutorials-with-omicverse
title: Spatial transcriptomics tutorials with omicverse
description: "Spatial transcriptomics: Visium/HD, Stereo-seq, Slide-seq preprocessing (crop, rotate, cellpose), deconvolution (Tangram, cell2location, Starfysh), clustering (GraphST, STAGATE), integration, trajectory, communication."
---

# Spatial Transcriptomics with OmicVerse

This skill covers spatial analysis workflows organized into three stages: Preprocessing, Deconvolution, and Downstream Analysis. Each stage includes the critical function calls, parameter guidance, and common pitfalls.

## Defensive Validation: Always Check Spatial Coordinates First

Before ANY spatial operation, verify that spatial coordinates exist and are numeric:

```python
# Required check before spatial analysis
assert 'spatial' in adata.obsm, \
    "Missing adata.obsm['spatial']. Load with ov.io.spatial.read_visium() or set manually."
# Cast to float64 to prevent coordinate precision issues during rotation/cropping
adata.obsm['spatial'] = adata.obsm['spatial'].astype('float64')
```

## Stage 1: Preprocessing

### Crop, Rotate, and Align Coordinates

Load Visium data and manipulate spatial coordinates for region selection and alignment:

```python
import scanpy as sc, omicverse as ov
ov.plot_set()

adata = sc.datasets.visium_sge(sample_id="V1_Breast_Cancer_Block_A_Section_1")
library_id = list(adata.uns['spatial'].keys())[0]

# Cast coordinates before manipulation
adata.obsm['spatial'] = adata.obsm['spatial'].astype('float64')

# Crop to region of interest
adata_crop = ov.space.crop_space_visium(adata, crop_loc=(0, 0), crop_area=(1000, 1000),
                                         library_id=library_id, scale=1)

# Rotate and auto-align
adata_rot = ov.space.rotate_space_visium(adata, angle=45, library_id=library_id)
ov.space.map_spatial_auto(adata_rot, method='phase')
# For manual refinement: ov.space.map_spatial_manual(adata_rot, ...)
```

### Visium HD Cell Segmentation

Segment Visium HD bins into cells using cellpose:

```python
adata = ov.space.read_visium_10x(path="binned_outputs/square_002um/",
                                  source_image_path="tissue_image.btf")
ov.pp.filter_genes(adata, min_cells=3)
ov.pp.filter_cells(adata, min_counts=1)

# H&E-based segmentation (in place, returns None: do not reassign adata);
# method="stardist" runs bin2cell's StarDist path instead of Cellpose
ov.space.visium_10x_hd_cellpose_he(adata, mpp=0.3, gpu=True, buffer=150)
# Expand labels to neighboring bins
ov.space.visium_10x_hd_cellpose_expand(adata, labels_key='labels_he',
                                        expanded_labels_key='labels_he_expanded', max_bin_distance=4)
# Gene-expression-driven seeds
ov.space.visium_10x_hd_cellpose_gex(adata, obs_key="n_counts_adjusted", mpp=0.3, sigma=5)
# Merge labels and aggregate to cell-level
ov.space.salvage_secondary_labels(adata, primary_label='labels_he_expanded',
                                   secondary_label='labels_gex', labels_key='labels_joint')
cdata = ov.space.bin2cell(adata, labels_key='labels_joint')
```

### Image and transcript segmentation (`ov.space.seg`)

For a raw microscopy image (DAPI / H&E / IF) or an imaging-platform
transcript table (Xenium, CosMx, MERSCOPE), use `ov.space.seg`. Check that it
exists first (`hasattr(ov.space, "seg")`); older omicverse releases lack it.

```python
ov.space.seg.methods()                      # backends, installed?, GPU?, input type
labels = ov.space.seg.segment("dapi.tif", method="cellpose", gpu=True)   # tiled, seam-free
# method: "watershed" (no deps) | "cellpose" (v3/v4) | "stardist" | "sam" | "sam2"
#         | "micro_sam" | "cellsam"; image_type="he" for H&E, diameter= in pixels
tx["cell"] = ov.space.seg.assign_transcripts(tx, labels, pixel_size=0.2125)
cdata = ov.space.seg.aggregate(tx, cell="cell", gene="feature_name", labels=labels,
                               pixel_size=0.2125)       # cells x genes AnnData + polygons
# Transcript-only (Baysor CLI must be on PATH), optionally with the image labels as a prior
tx2 = ov.space.seg.segment_transcripts(tx, method="baysor", scale=8, prior="cell")
# Reference-free CellSPA metrics and a ranked comparison of several segmentations
res = {m: ov.space.seg.evaluate(lab, transcripts=tx.assign(
          cell=ov.space.seg.assign_transcripts(tx, lab))) for m, lab in labels_by_method.items()}
ov.space.seg.compare(res)
```

- Coordinates: `pixel_size` (µm per pixel), `offset`, or a 3x3 `transform`
  map transcript units to image pixels; get them from the platform's metadata.
- Picking a method: in the Nat. Methods 2026 benchmark no method won
  everywhere. SAM / SAM 2 led the unadjusted overall score and transcript QC,
  Baysor ranked near the top once segmentation yield was accounted for, and
  Cellpose and watershed ranked lower. Run 2-3 candidates and pick with
  `ov.space.seg.compare` on your own data.
- `sam` / `sam2` fall back to CPU on Apple MPS; run them on a CUDA GPU.

## Stage 2: Deconvolution

### Critical API Reference: Method Selection

```python
# CORRECT — Tangram: method passed to deconvolution() call
decov_obj = ov.space.Deconvolution(adata_sc=sc_adata, adata_sp=sp_adata,
                                    celltype_key='Subset', result_dir='result/tangram')
decov_obj.preprocess_sc(max_cells=5000)
decov_obj.preprocess_sp()
decov_obj.deconvolution(method='tangram', num_epochs=1000)

# CORRECT — cell2location: method passed at INIT time
cell2_obj = ov.space.Deconvolution(adata_sc=sc_adata, adata_sp=sp_adata,
                                    celltype_key='Subset', result_dir='result/c2l',
                                    method='cell2location')
cell2_obj.deconvolution(max_epochs=30000)
cell2_obj.save_model('result/c2l/model')
```

Note: For cell2location, the `method` parameter is set at initialization, not at the `deconvolution()` call. For Tangram, it's passed to `deconvolution()`.

### Starfysh Archetypal Deconvolution

```python
from omicverse.external.starfysh import AA, utils, plot_utils

visium_args = utils.prepare_data(adata_path="data/counts.h5ad",
                                  signature_path="data/signatures.csv",
                                  min_cells=10, filter_hvg=True, n_top_genes=3000)
adata, adata_normed = visium_args.get_adata()
aa_model = AA.ArchetypalAnalysis(adata_orig=adata_normed)
aa_model.fit(k=12, n_init=10)
visium_args = utils.refine_anchors(visium_args, aa_model, add_marker=True)
model, history = utils.run_starfysh(visium_args, poe=False, n_repeat=5, lr=5e-3, max_epochs=500)
```

## Stage 3: Downstream Analysis

### Spatial Clustering

```python
# GraphST + mclust
ov.utils.cluster(adata, use_rep='graphst|original|X_pca', method='mclust', n_components=7)

# STAGATE
ov.utils.cluster(adata, use_rep='STAGATE', method='mclust', n_components=7)

# Merge small clusters
ov.space.merge_cluster(adata, groupby='mclust', resolution=0.5)
```

Algorithm choice: GraphST and STAGATE require precalculated latent spaces. For standard clustering without spatial-aware embeddings, use Leiden/Louvain directly.

### Multi-Slice Integration (STAligner)

```python
import anndata as ad
Batch_list = [ov.read(p) for p in slice_paths]
adata_concat = ad.concat(Batch_list, label='slice_name', keys=section_ids)
STAligner_obj = ov.space.pySTAligner(adata=adata_concat, batch_key='slice_name',
                                      hidden_dims=[256, 64], use_gpu=True)
STAligner_obj.train_STAligner_subgraph(nepochs=800, lr=1e-3)
STAligner_obj.train()
adata_aligned = STAligner_obj.predicted()
sc.pp.neighbors(adata_aligned, use_rep='STAligner')
```

### Spatial Trajectories

**SpaceFlow** — pseudo-spatial maps:
```python
sf_obj = ov.space.pySpaceFlow(adata)
sf_obj.train(spatial_regularization_strength=0.1, num_epochs=300, patience=50)
sf_obj.cal_pSM(n_neighbors=20, resolution=1.0)
```

**STT** — transition dynamics:
```python
STT_obj = ov.space.STT(adata, spatial_loc='xy_loc', region='Region', n_neighbors=20)
STT_obj.stage_estimate()
STT_obj.train(n_states=9, n_iter=15, weight_connectivities=0.5)
```

### Cell Communication (COMMOT + FlowSig)

```python
df_cellchat = ov.external.commot.pp.ligand_receptor_database(species='human', database='cellchat')
df_cellchat = ov.external.commot.pp.filter_lr_database(df_cellchat, adata, min_expr_frac=0.05)
ov.external.commot.tl.spatial_communication(adata, lr_database=df_cellchat,
                                              distance_threshold=500, result_prefix='cellchat')
# FlowSig network
adata.layers['normalized'] = adata.X.copy()
ov.external.flowsig.tl.construct_intercellular_flow_network(
    adata, commot_output_key='commot-cellchat',
    flowsig_output_key='flowsig-cellchat', edge_threshold=0.7)
```

### Structural Layers (GASTON) and Slice Alignment (SLAT)

**GASTON** — iso-depth estimation:
```python
gas_obj = ov.space.GASTON(adata)
A = gas_obj.prepare_inputs(n_pcs=50)
gas_obj.load_rescale(A)
gas_obj.train(hidden_dims=[64, 32], dropout=0.1, max_epochs=2000)
gaston_isodepth, gaston_labels = gas_obj.cal_iso_depth(n_layers=5)
```

**SLAT** — cross-slice alignment:
```python
from omicverse.external.scSLAT.model import Cal_Spatial_Net, load_anndatas, run_SLAT, spatial_match
Cal_Spatial_Net(adata1, k_cutoff=20, model='KNN')
Cal_Spatial_Net(adata2, k_cutoff=20, model='KNN')
edges, features = load_anndatas([adata1, adata2], feature='DPCA', check_order=False)
embeddings, *_ = run_SLAT(features, edges, LGCN_layer=5)
best, index, distance = spatial_match(embeddings, adatas=[adata1, adata2])
```

## Troubleshooting

- **`ValueError: spatial coordinates out of bounds` after rotation**: Cast `adata.obsm['spatial']` to `float64` BEFORE calling `rotate_space_visium`. Integer coordinates lose precision during trigonometric rotation.
- **Cellpose segmentation fails with memory error**: For large `.btf` images, use `backend='tifffile'` to memory-map the image. Reduce `buffer` parameter if GPU memory is insufficient.
- **Gene ID overlap failure in Tangram/cell2location**: Harmonise identifiers (ENSEMBL vs gene symbols) between `adata_sc` and `adata_sp` before calling `preprocess_sc`/`preprocess_sp`. Drop non-overlapping genes.
- **`mclust` clustering error**: Requires `rpy2` and the R `mclust` package. If R bindings are unavailable, switch to `method='louvain'` or `method='leiden'`.
- **STAligner/SpaceFlow embeddings collapse to a single point**: Verify `adata.obsm['spatial']` exists and coordinates are scaled appropriately. Tune learning rate (try `lr=5e-4`) and regularisation strength.
- **FlowSig returns empty network**: Build spatial neighbor graphs before Moran's I filtering. Increase bootstraps or lower `edge_threshold` (try 0.5) if the network is too sparse.
- **GASTON `RuntimeError` in training**: Provide a writable `out_dir` path. PyTorch nondeterminism may cause variation between runs—set `torch.manual_seed()` for reproducibility.
- **SLAT alignment has many low-quality matches**: Regenerate spatial graphs with a higher `k_cutoff` value. Inspect `low_quality_index` flags and filter cells with high distance scores.
- **STT pathway enrichment fails**: `gseapy` needs network access for gene set downloads. Cache gene sets locally with `ov.utils.geneset_prepare()` and pass the dictionary directly.

## Dependencies
- Core: `omicverse`, `scanpy`, `anndata`, `squidpy`, `numpy`, `matplotlib`
- Segmentation: `cellpose`, `opencv-python`/`tifffile`, optional GPU PyTorch; `pip install 'omicverse[seg]'` for the `ov.space.seg` backends (Cellpose, SAM, SAM 2)
- Deconvolution: `tangram-sc`, `cell2location`, `pytorch-lightning`; Starfysh needs `torch`, `scikit-learn`
- Downstream: `scikit-learn`, `commot`, `flowsig`, `gseapy`, torch-backed modules (STAligner, SpaceFlow, GASTON, SLAT)

## Examples
- "Crop and rotate my Visium slide, then run cellpose segmentation on the HD data and aggregate to cell-level AnnData."
- "Deconvolve my lymph node spatial data with Tangram and cell2location, compare proportions, and plot cell-type maps."
- "Integrate three DLPFC slices with STAligner, cluster with STAGATE, and infer communication with COMMOT+FlowSig."

## References
- Quick copy/paste commands: [`reference.md`](reference.md)

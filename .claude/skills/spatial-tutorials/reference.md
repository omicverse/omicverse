# Quick commands: Spatial tutorials

## Load Visium slide and adjust coordinates
```python
import scanpy as sc
import omicverse as ov
ov.plot_set(font_path='Arial')

adata = sc.datasets.visium_sge(sample_id="V1_Breast_Cancer_Block_A_Section_1")
library_id = list(adata.uns['spatial'].keys())[0]
adata.obsm['spatial'] = adata.obsm['spatial'].astype('float64')
adata_sp = ov.space.crop_space_visium(
    adata,
    crop_loc=(0, 0),
    crop_area=(1000, 1000),
    library_id=library_id,
    scale=1,
)
adata_rot = ov.space.rotate_space_visium(adata, angle=45, library_id=library_id)
ov.space.map_spatial_auto(adata_rot, method='phase')
```

## Visium HD cellpose segmentation
```python
adata = ov.space.read_visium_10x(
    path="binned_outputs/square_002um/",
    source_image_path="Visium_HD_Human_Colon_Cancer_tissue_image.btf",
)
ov.pp.filter_genes(adata, min_cells=3)
ov.pp.filter_cells(adata, min_counts=1)
ov.space.visium_10x_hd_cellpose_he(  # in place, returns None
    adata,
    mpp=0.3,
    he_save_path="stardist/he_colon1.tiff",
    prob_thresh=0,
    flow_threshold=0.4,
    gpu=True,
    buffer=150,
    backend='tifffile',
)
ov.space.visium_10x_hd_cellpose_expand(adata, labels_key='labels_he', expanded_labels_key='labels_he_expanded', max_bin_distance=4)
ov.space.visium_10x_hd_cellpose_gex(adata, obs_key="n_counts_adjusted", mpp=0.3, sigma=5)
ov.space.salvage_secondary_labels(adata, primary_label='labels_he_expanded', secondary_label='labels_gex', labels_key='labels_joint')
cdata = ov.space.bin2cell(adata, labels_key='labels_joint', spatial_keys=["spatial", "spatial_cropped_150_buffer"])
```

## Tangram / cell2location wrapper
```python
adata_sc = ov.read('data/sc.h5ad')
adata_sp = sc.datasets.visium_sge(sample_id="V1_Human_Lymph_Node")
adata_sp.var_names_make_unique()

decov_obj = ov.space.Deconvolution(
    adata_sc=adata_sc,
    adata_sp=adata_sp,
    celltype_key='Subset',
    result_dir='result/tangram',
)
decov_obj.preprocess_sc(max_cells=5000)
decov_obj.preprocess_sp()
decov_obj.deconvolution(method='tangram', num_epochs=1000)
ov.utils.save({'tangram_map': decov_obj.imputed_matrix}, 'result/tangram/map.npz')

# cell2location reuse
cell2_obj = ov.space.Deconvolution(
    adata_sc=adata_sc,
    adata_sp=adata_sp,
    celltype_key='Subset',
    result_dir='result/cell2location',
    method='cell2location',
)
cell2_obj.deconvolution(max_epochs=30000)
cell2_obj.save_model('result/cell2location/model')
```

## Starfysh anchors and training
```python
from omicverse.external.starfysh import AA, utils, plot_utils
visium_args = utils.prepare_data(
    adata_path="data/visium_counts.h5ad",
    signature_path="data/signatures.csv",
    min_cells=10,
    filter_hvg=True,
    n_top_genes=3000,
)
adata, adata_normed = visium_args.get_adata()
aa_model = AA.ArchetypalAnalysis(adata_orig=adata_normed)
aa_model.fit(k=12, n_init=10)
visium_args = utils.refine_anchors(visium_args, aa_model, add_marker=True)
model, history = utils.run_starfysh(
    visium_args,
    poe=False,
    n_repeat=5,
    lr=5e-3,
    max_epochs=500,
)
plot_utils.pl_spatial_inf_feature(visium_args.adata_plot, feature='ql_m', cmap='Blues')
```

## STAligner multi-slice integration
```python
import anndata as ad

Batch_list = [ov.read(path) for path in ["Stereo_seq.h5ad", "SlideseqV2.h5ad"]]
adata_concat = ad.concat(Batch_list, label='slice_name', keys=['Stereo', 'Slide'])
STAligner_obj = ov.space.pySTAligner(
    adata=adata_concat,
    batch_key='slice_name',
    hidden_dims=[256, 64],
    use_gpu=True,
)
STAligner_obj.train_STAligner_subgraph(nepochs=800, lr=1e-3, weight_decay=1e-5)
STAligner_obj.train()
adata_aligned = STAligner_obj.predicted()
sc.pp.neighbors(adata_aligned, use_rep='STAligner', random_state=666)
ov.utils.cluster(adata_aligned, use_rep='STAligner', method='GMM', n_components=7)
```

## SpaceFlow pseudo-spatial map
```python
adata = ov.io.spatial.read_visium(path='data', count_file='151676_filtered_feature_bc_matrix.h5')
sc.pp.calculate_qc_metrics(adata, inplace=True)
sf_obj = ov.space.pySpaceFlow(adata)
sf_obj.train(
    spatial_regularization_strength=0.1,
    num_epochs=300,
    patience=50,
    batch_size=1024,
)
sf_obj.cal_pSM(n_neighbors=20, resolution=1.0)
sc.pl.spatial(adata, color=['pSM_spaceflow'], cmap='RdBu_r')
```

## STT attractor dynamics
```python
adata = ov.read('mouse_brain.h5ad')
STT_obj = ov.space.STT(
    adata,
    spatial_loc='xy_loc',
    region='Region',
    n_neighbors=20,
    spa_weight=0.5,
)
STT_obj.stage_estimate()
STT_obj.train(n_states=9, n_iter=15, weight_connectivities=0.5)
pathway_dict = ov.utils.geneset_prepare('genesets/KEGG_2019_Mouse.txt', organism='Mouse')
STT_obj.compute_pathway(pathway_dict)
fig = STT_obj.plot_pathway(figsize=(10, 8), size=100)
```

## COMMOT + FlowSig
```python
df_cellchat = ov.external.commot.pp.ligand_receptor_database(species='human', database='cellchat')
df_cellchat = ov.external.commot.pp.filter_lr_database(
    df_cellchat,
    adata,
    min_expr_frac=0.05,
)
ov.external.commot.tl.spatial_communication(
    adata,
    lr_database=df_cellchat,
    distance_threshold=500,
    result_prefix='cellchat',
)
adata.layers['normalized'] = adata.X.copy()
ov.external.flowsig.tl.construct_intercellular_flow_network(
    adata,
    commot_output_key='commot-cellchat',
    flowsig_output_key='flowsig-cellchat',
    edge_threshold=0.7,
)
flow_network = ov.external.flowsig.tl.construct_intercellular_flow_network(
    adata,
    flowsig_output_key='flowsig-cellchat',
    adjacency_key='adjacency_validated_filtered',
)
ov.pl.plot_flowsig_network(flow_network=flow_network, node_size=800)
```

## GASTON iso-depth estimation
```python
adata = sc.datasets.visium_sge(sample_id="V1_Human_Lymph_Node")
adata.var_names_make_unique()
gas_obj = ov.space.GASTON(adata)
A = gas_obj.prepare_inputs(n_pcs=50)
gas_obj.load_rescale(A)
gas_obj.train(
    hidden_dims=[64, 32],
    dropout=0.1,
    weight_decay=1e-5,
    max_epochs=2000,
)
gaston_isodepth, gaston_labels = gas_obj.cal_iso_depth(n_layers=5)
adata.obs['gaston_labels'] = gaston_labels.astype(str)
sc.pl.spatial(adata, color='gaston_labels', title='GASTON IsoDepth')
```

## SLAT slice alignment
```python
from omicverse.external.scSLAT.model import (
    Cal_Spatial_Net,
    load_anndatas,
    run_SLAT,
    spatial_match,
)
from omicverse.external.scSLAT.viz import Sankey_multi

adata1 = ov.read('data/E115_Stereo.h5ad')
adata2 = ov.read('data/E125_Stereo.h5ad')
adata1.obs['week'] = 'E11.5'
adata2.obs['week'] = 'E12.5'
Cal_Spatial_Net(adata1, k_cutoff=20, model='KNN')
Cal_Spatial_Net(adata2, k_cutoff=20, model='KNN')
edges, features = load_anndatas([adata1, adata2], feature='DPCA', check_order=False)
embeddings, *_ = run_SLAT(features, edges, LGCN_layer=5)
best, index, distance = spatial_match(embeddings, adatas=[adata1, adata2], reorder=False)
adata2.obs['low_quality_index'] = distance
fig = Sankey_multi(
    adata_li=[adata1, adata2],
    obs_keys=['annotation', 'annotation'],
    matching=index,
    title='SLAT alignment',
)
fig.write_html('slat_sankey.html')
```

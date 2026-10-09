# Bulk2Single quick commands

```python
import scanpy as sc
import scvelo as scv
import anndata
import matplotlib.pyplot as plt
import omicverse as ov

ov.plot_set()

# load data (replace paths with user files)
bulk_df = ov.read('data/GSE74985_mergedCount.txt.gz', index_col=0)
bulk_df = ov.bulk.Matrix_ID_mapping(bulk_df, 'genesets/pair_GRCm39.tsv')

single_adata = scv.datasets.dentategyrus()
print(single_adata.obs['clusters'].value_counts())

model = ov.bulk2single.Bulk2Single(
    bulk_data=bulk_df,
    single_data=single_adata,
    celltype_key='clusters',
    bulk_group=['dg_d_1', 'dg_d_2', 'dg_d_3'],
    top_marker_num=200,
    ratio_num=1,
    gpu=0,
)

# Scaden is the default; use method='tape' to select TAPE explicitly.
fractions = model.predicted_fraction()
ax = fractions.plot(kind='bar', stacked=True, figsize=(8, 4))
ax.set_ylabel('Cell Fraction')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.show()

model.bulk_preprocess_lazy()
model.single_preprocess_lazy()
model.prepare_input()

vae_net = model.train(
    batch_size=512,
    learning_rate=1e-4,
    hidden_size=256,
    epoch_num=3500,
    vae_save_dir='data/bulk2single/save_model',
    vae_save_name='dg_vae',
    generate_save_dir='data/bulk2single/output',
    generate_save_name='dg',
)
model.plot_loss()

# resume later if needed
model.load('data/bulk2single/save_model/dg_vae.pth')

generated = model.generate()
filtered = model.filtered(generated, leiden_size=25)
filtered.write_h5ad('bulk2single_filtered.h5ad', compression='gzip')

ov.bulk2single.bulk2single_plot_cellprop(filtered, celltype_key='clusters')
plt.grid(False)

ov.bulk2single.bulk2single_plot_cellprop(single_adata, celltype_key='clusters')
plt.grid(False)

ov.bulk2single.bulk2single_plot_correlation(single_adata, filtered, celltype_key='clusters')
plt.grid(False)

filtered.obsm['X_mde'] = ov.utils.mde(filtered.obsm['X_pca'])
ov.pl.embedding(
    filtered,
    basis='X_mde',
    color=['clusters'],
    palette=ov.pl.sc_color(),
    frameon='small',
)
```

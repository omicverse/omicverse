# Visualize existing pycellchat results with OmicVerse

The adapter supports the single-dataset result schema of
[shaoleishen/pycellchat 0.1.0](https://github.com/shaoleishen/pycellchat/tree/697bbc8212331283a3db2dbc4a7481691da708df).
It does not run inference or require pycellchat's Rust backend to be installed.
Run the upstream analysis first, then convert the completed `CellChat` object:

```python
import omicverse as ov

# cellchat is an existing pycellchat.CellChat with compute_commun_prob results.
comm = ov.single.format_pycellchat_results(cellchat)

# Equivalent inputs: cellchat.cc, or its source AnnData.
comm = ov.single.to_comm_adata(data=cellchat.cc)
comm = ov.single.to_comm_adata(cellchat.adata)

ov.pl.ccc_heatmap(comm, plot_type="dot", display_by="interaction", show=False)
ov.pl.ccc_network_plot(comm, plot_type="circle", show=False)
ov.pl.ccc_stat_plot(comm, plot_type="bar", display_by="interaction", show=False)
comm.write_h5ad("pycellchat_communication.h5ad")
```

The input namespace is `cellchat.adata.uns['cellchat']`:

- `net['prob']` and `net['pval']`: matching K × K × L-R arrays, with sender, receiver, interaction axes
- `idents['names']`: the K group names in tensor-axis order
- `LR['LRsig']`: one row per L-R tensor slice, with ligand and receptor columns
- `LRsig['interaction_name']`: unique interaction identifiers; the table index is used when this column is absent
- `LR['interaction_name']`: when present and nonempty, must agree with LRsig order

Conversion preserves the input order, all ordered cell-type pairs (including
self-pairs), zero scores, nonsignificant entries, and original p-values. `X` and
`layers['means']` hold communication probabilities, not expression means.
Missing p-values, ambiguous identifiers, and mismatched dimensions are errors;
no p-values are invented. Plotters apply their own significance thresholds.

L-R metadata is retained in `var`; pathway names populate `classification`,
and ligand/receptor labels populate `gene_a`/`gene_b`. Complex labels are kept
intact. Inference options and the grouping column are recorded in `uns`. The
source object is not modified. Group labels containing `|` require a different
`separator`, for example `format_pycellchat_results(cellchat, separator="::")`.

Convert merged datasets separately. Pathway-only arrays and the upstream
`save_cellchat` summary CSV/JSON export do not contain the complete L-R
probability/p-value tensors needed by this adapter. Other packages with similar
names are outside this schema contract.

Schema references: upstream
[`object.py`](https://github.com/shaoleishen/pycellchat/blob/697bbc8212331283a3db2dbc4a7481691da708df/python/pycellchat/object.py)
and
[`modeling.py`](https://github.com/shaoleishen/pycellchat/blob/697bbc8212331283a3db2dbc4a7481691da708df/python/pycellchat/modeling.py).

---
name: single-cell-cellphonedb-communication-mapping
title: Single-cell CellPhoneDB communication mapping
description: "CellPhoneDB v5 ligand-receptor analysis, CellChatViz plots, and the newer ccc_heatmap / ccc_network_plot / ccc_stat_plot communication visualizations in OmicVerse."
---

# Single-cell CellPhoneDB communication mapping

## Overview
Apply this skill when a user wants to quantify ligand-receptor communication between annotated single-cell populations and visualize the result with OmicVerse's CellPhoneDB plotting stack. This skill now covers both the original low-level `ov.pl.CellChatViz` workflow and the newer public plotting APIs:

- `ov.pl.ccc_heatmap(...)`
- `ov.pl.ccc_network_plot(...)`
- `ov.pl.ccc_stat_plot(...)`

Use the `ccc_*` functions by default when the goal is to quickly generate publication-ready plots with a stable public API. Drop down to `CellChatViz` when the user needs method-level control or wants to combine several custom visualizations in one notebook.

## Instructions
1. **Prepare the environment**
   - Use an environment with `omicverse>=0.2`, `scanpy`, `anndata`, `pandas`, `matplotlib`, `seaborn`, and CellPhoneDB resources.
   - For the newer visualizations, also ensure these optional plotting dependencies are available when needed:
     - `marsilea` for heatmap and bubble matrix plotters.
     - `mpl-chord-diagram` for chord diagrams.
     - `networkx` for diffusion and network-style plots.
     - `adjustText` if the user wants improved automatic label repulsion.
   - Activate OmicVerse plotting defaults with `ov.plot_set()`.
2. **Load and validate the annotated AnnData**
   - Read the normalized expression matrix with `ov.read(...)`.
   - Keep the communication grouping column clean, categorical, and aligned with the intended identities.
   - Recommended checks:
     ```python
     celltype_key = "cell_labels"
     assert celltype_key in adata.obs.columns, f"{celltype_key} missing from adata.obs"
     adata.obs[celltype_key] = adata.obs[celltype_key].astype("category").cat.remove_unused_categories()
     assert not adata.obs[celltype_key].isna().any(), f"NaN values found in {celltype_key}"
     min_per_group = adata.obs[celltype_key].value_counts().min()
     if min_per_group < 10:
         print(f"WARNING: smallest group has {min_per_group} cells; sparse groups may destabilize permutations")
     ```
   - Confirm the matrix is log-normalized before running CellPhoneDB. Raw counts can distort permutation-based significance.
3. **Run CellPhoneDB through OmicVerse**
   - Use `ov.single.run_cellphonedb_v5(...)` and persist the outputs:
     ```python
     cpdb_results, adata_cpdb = ov.single.run_cellphonedb_v5(
         adata,
         cpdb_file_path="./cellphonedb.zip",
         celltype_key="cell_labels",
         min_cell_fraction=0.005,
         min_genes=200,
         min_cells=3,
         iterations=1000,
         threshold=0.1,
         pvalue=0.05,
         threads=10,
         output_dir="./cpdb_results",
         cleanup_temp=True,
     )
     ```
   - Save `cpdb_results` and `adata_cpdb` so downstream plotting can be repeated without rerunning permutations.
   - **CellChat alternative** (no R, no extra install beyond PyTorch): `ov.single.run_cellchat` reproduces R CellChat 2.2 (identical over-expressed genes/LR pairs, probabilities to ~1e-15) and stores `adata.uns['cellchat_res']`, which every `ov.pl.ccc_*` call reads directly:
     ```python
     ov.single.run_cellchat(adata, groupby="cell_labels", species="human")  # log-normalised X
     comm = ov.single.format_cellchat_results(adata)                         # optional explicit comm AnnData
     df = ov.single.cellchat_subset_communication(adata, signaling=["TGFb"])  # significant LR table
     centr = ov.single.cellchat_centrality(adata)                             # sender/receiver/mediator/influencer
     ```
     CellChat scores are probabilities (~1e-3), not expression; pathway summaries pick scale-aware thresholds automatically. Groups with `<= min_cells` (default 10) cells are removed, as in R `filterCommunication`.
4. **Prefer the new public plotting APIs for standard visualization requests**
   - Use `ov.pl.ccc_heatmap(...)` for matrix-like plots:
     - `plot_type="heatmap"` for aggregated pathway-level communication.
     - `plot_type="focused_heatmap"` to highlight stronger interactions after thresholding weak entries.
     - `plot_type="dot"` or `"bubble"` for interaction-level summaries.
     - `plot_type="pathway_bubble"` for pathway-focused Marsilea bubble summaries.
     - `plot_type="bubble_lr"` for ligand-receptor-pair-specific bubble matrices.
     - `plot_type="role_heatmap"`, `"role_network"`, or `"role_network_marsilea"` for signaling role summaries.
     - `plot_type="diff_heatmap"` when comparing two communication AnnData objects.
   - Use `ov.pl.ccc_network_plot(...)` for graph-like plots:
     - `plot_type="circle"` or `"circle_focused"` for global communication networks.
     - `plot_type="individual_outgoing"` / `"individual_incoming"` for sender- or receiver-centric circle panels.
     - `plot_type="individual"` for a single pathway and optionally a selected L-R pair.
     - `plot_type="chord"` for cell-type-level pathway chords.
     - `plot_type="gene_chord"` for gene-level chord diagrams across pathway-specific ligand and receptor nodes.
     - `plot_type="lr_chord"` for specified ligand-receptor pairs.
     - `plot_type="diffusion"` for pathway similarity and diffusion-style network structure.
     - `plot_type="diff_network"` when comparing two communication objects.
     - `plot_type="bipartite"`, `"arrow"`, `"sigmoid"`, or `"embedding_network"` for alternative layouts.
   - Use `ov.pl.ccc_stat_plot(...)` for statistics and summary panels:
     - `plot_type="pathway_summary"` to rank pathways by communication strength and significance.
     - `plot_type="lr_contribution"` to show the dominant ligand-receptor pairs within a pathway.
     - `plot_type="scatter"` or `"role_scatter"` to compare outgoing versus incoming signaling roles.
     - `plot_type="role_network"` or `"role_network_marsilea"` for matrix-style role summaries.
     - `plot_type="sankey"` for communication flow summaries.
5. **Use `CellChatViz` directly when the user needs method-level control**
   - Create a stable palette mapping from cell labels:
     ```python
     color_dict = dict(zip(
         adata.obs["cell_labels"].cat.categories,
         adata.uns["cell_labels_colors"]
     ))
     viz = ov.pl.CellChatViz(adata_cpdb, palette=color_dict)
     ```
   - Recommended direct workflow:
     - `viz.compute_aggregated_network(...)` then `viz.netVisual_circle(...)`.
     - `viz.compute_pathway_communication(...)` then `viz.get_significant_pathways_v2(...)`.
     - `viz.netVisual_heatmap_marsilea(...)` or `viz.netVisual_heatmap_marsilea_focused(...)`.
     - `viz.netVisual_bubble_marsilea(...)` for pathway bubbles.
     - `viz.netVisual_bubble_lr(...)` for selected ligand-receptor pairs.
     - `viz.netVisual_chord_cell(...)`, `viz.netVisual_chord_gene(...)`, and `viz.netVisual_chord_LR(...)`.
     - `viz.netVisual_individual(...)` for one pathway / one enriched pair.
     - `viz.netAnalysis_computeCentrality()` followed by role heatmap, scatter, and network plots.
     - `viz.netAnalysis_contribution(pathway)` for pathway-level pair contribution analysis.
     - `viz.netVisual_diffusion(...)` for pathway similarity structure.
6. **Highlight the new visualization capabilities clearly**
   - The newer additions worth surfacing in answers are:
     - `gene_chord`: gene-level chord diagrams, not just cell-type-level chords.
     - `bubble_lr`: Marsilea bubble summaries centered on explicit ligand-receptor pairs.
     - `focused_heatmap`: thresholded pathway heatmaps that suppress weak interactions.
     - `role_network_marsilea`: richer role summaries with dendrograms, color bars, and importance bars.
     - `diffusion`: pathway similarity network based on communication patterns.
     - `pathway_summary` and `lr_contribution`: higher-level summary/statistical views for prioritization.
   - When the user says "new visualization", prioritize demonstrating one of these rather than only the legacy circle plot.
7. **Parameter tips for the newer plots**
   - For pathway bubbles:
     - `group_pathways=True` groups by pathway rather than individual L-R pairs.
     - `transpose=True` is useful when too many cell-pair rows make labels unreadable.
     - `add_violin=True` can expose score distributions but makes figures denser.
   - For ligand-receptor bubbles:
     - `show_all_pairs=True` is useful when the user wants to compare a fixed panel of pairs even if some are weak or absent.
     - `pair_lr_use` or `interaction_use` should match the pair naming in `adata.var`.
   - For gene chords:
     - Require `adata.var["gene_a"]` and `adata.var["gene_b"]`.
     - Use `rotate_names=True` when genes or cell-type labels are long.
   - For focused heatmaps and focused circle plots:
     - Tune `min_interaction_threshold` to remove weak edges before plotting.
   - For role plots:
     - Run `viz.netAnalysis_computeCentrality()` first when using the low-level API.
     - Use `pattern="incoming"` and `pattern="outgoing"` separately if the user wants interpretable sender vs receiver programs.
8. **Troubleshooting**
   - **Metadata alignment**: the communication grouping column must be categorical and free of missing values.
   - **Database bundle**: `cpdb_file_path` must point to a valid CellPhoneDB v5 SQLite zip.
   - **Sparse groups**: very small sender or receiver groups often cause unstable or empty outputs.
   - **Missing columns in `adata.var`**:
     - `classification` is needed for pathway-filtered plotting.
     - `gene_a` and `gene_b` are needed for gene-level or ligand-receptor-specific plots.
   - **Optional dependency errors**:
     - install `marsilea` for `pathway_bubble`, `bubble_lr`, focused heatmaps, and role-network Marsilea views.
     - install `mpl-chord-diagram` for chord plots.
     - install `adjustText` if label overlap is severe.
   - **Palette mismatches**: rebuild the palette from sorted categories in `adata.obs[celltype_key].cat.categories` and the corresponding `adata.uns[f"{celltype_key}_colors"]`.

## Examples
- "Run CellPhoneDB and then use `ov.pl.ccc_heatmap(..., plot_type='focused_heatmap')` to show the strongest pathways."
- "Create a gene-level chord diagram for FGF signaling with `ov.pl.ccc_network_plot(..., plot_type='gene_chord')`."
- "Compare selected ligand-receptor pairs across sender-receiver combinations with `plot_type='bubble_lr'`."
- "Summarize the top pathways and then rank within-pathway ligand-receptor contributions with `ccc_stat_plot(..., plot_type='pathway_summary')` and `plot_type='lr_contribution'`."
- "Use `CellChatViz` directly to compute centrality and render Marsilea role-network plots."

## References
- Tutorial notebook: [`t_cellphonedb.ipynb`](../../omicverse_guide/docs/Tutorials-single/t_cellphonedb.ipynb)
- Example data: [`omicverse_guide/docs/Tutorials-single/data/cpdb/`](../../omicverse_guide/docs/Tutorials-single/data/cpdb/)
- Quick copy/paste commands: [`reference.md`](reference.md)

# ov.Agent Real-World Prompt Library

Use these scenario-driven prompts to ask `ov.Agent` to reproduce or adapt the workflows that mirror each tutorial notebook in `omicverse_guide/docs`. The prompt text intentionally avoids naming notebook files so it can be dropped directly into an ov.Agent conversation.

## Tutorials-velo

| Tutorial reference | Real-world ov.Agent prompt |
| --- | --- |
| `Tutorials-velo/t_graphvelo.ipynb` | "Our developmental biology lab captured single-cell RNA velocity data from midbrain organoids. Guide me through building a gene-level velocity graph, highlighting lineage-specific kinetic genes and exporting edge lists for downstream topology checks." |
| `Tutorials-velo/t_velo.ipynb` | "We have standard spliced/unspliced counts from a hematopoiesis study. Walk through preprocessing, latent time inference, and velocity stream visualization so we can compare erythroid vs myeloid differentiation speeds." |

## Tutorials-bulk

| Tutorial reference | Real-world ov.Agent prompt |
| --- | --- |
| `Tutorials-bulk/t_wgcna.ipynb` | "A cardiology consortium profiled 200 failing and healthy ventricles. Help me construct a weighted co-expression network, identify trait-associated modules, and summarize hub genes suitable for follow-up CRISPR screens." |
| `Tutorials-bulk/t_deseq2.ipynb` | "A CRO delivered bulk RNA-seq for treated vs control xenografts with three replicates each. Show every step to run DESeq2-style normalization, differential testing, shrinkage, and result visualization suitable for a slide deck." |
| `Tutorials-bulk/t_tcga.ipynb` | "I need to pull TCGA RNA-seq for lung adenocarcinoma, harmonize metadata, and run a survival-stratified gene expression analysis that surfaces prognostic signatures for an internal report." |
| `Tutorials-bulk/t_deg.ipynb` | "Our toxicology team compared liver samples exposed to two compounds. Walk me through a complete differential expression workflow, including QC plots, volcano charts, and gene set enrichment to flag hepatotoxic pathways." |
| `Tutorials-bulk/t_decov_bulk.ipynb` | "We want to deconvolve immune cell proportions from bulk tumor biopsies using a custom signature matrix. Detail the pipeline to import expression matrices, normalize, run deconvolution, and benchmark estimates against flow data." |
| `Tutorials-bulk/t_bulk_combat.ipynb` | "Multiple sites contributed bulk RNA-seq with strong batch effects. Guide me through visualizing the confounder structure and applying ComBat-style correction before downstream modeling." |
| `Tutorials-bulk/t_network.ipynb` | "Our translational team is mapping cytokine-gene regulatory networks from bulk expression plus interaction priors. Help me build an interpretable network, score edge confidence, and export GraphML for review." |

## Tutorials-bulk2single

| Tutorial reference | Real-world ov.Agent prompt |
| --- | --- |
| `Tutorials-bulk2single/t_bulktrajblend.ipynb` | "We need to blend pseudotime from single-cell data with matched bulk temporal samples to better resolve differentiation programs. Show how to align trajectories and report concordant vs discordant gene modules." |
| `Tutorials-bulk2single/t_bulk2single.ipynb` | "A consortium only has high-quality bulk RNA-seq but wants single-cell level insights. Demonstrate how to project bulk samples into a reference single-cell manifold and recover pseudo cell-type abundances." |
| `Tutorials-bulk2single/t_single2spatial.ipynb` | "Our pathology unit wants to transfer single-cell annotations onto spatial transcriptomics slides. Outline the workflow to anchor dissociated single-cell profiles onto spatial spots and score mapping accuracy." |

## Tutorials-plotting

| Tutorial reference | Real-world ov.Agent prompt |
| --- | --- |
| `Tutorials-plotting/t_visualize_colorsystem.ipynb` | "Marketing needs a harmonized color system for figures covering multiple cell states. Help me define a reusable palette with accessible contrasts and export reference swatches for designers." |
| `Tutorials-plotting/t_visualize_bulk.ipynb` | "Prepare publication-ready plots summarizing a bulk RNA-seq experiment: include sample QC metrics, PCA, heatmaps, and volcano charts with consistent styling." |
| `Tutorials-plotting/t_visualize_single.ipynb` | "Create a visualization suite for a new single-cell atlas, including UMAP/TSNE embeddings, marker expression faceting, and dot plots sized and colored for presentations." |

## Tutorials-space

| Tutorial reference | Real-world ov.Agent prompt |
| --- | --- |
| `Tutorials-space/t_starfysh.ipynb` | "Our neuro-oncology team generated multiplexed imaging and wants to map signaling niches. Show how to run the STARfysh workflow to recover ligand-receptor gradients across the tissue." |
| `Tutorials-space/t_decov.ipynb` | "We must deconvolve mixed spatial transcriptomics spots into cell-type proportions for FFPE slides. Provide the step-by-step pipeline, including reference selection and uncertainty reporting." |
| `Tutorials-space/t_cluster_space.ipynb` | "Assist in clustering Visium spots from inflamed colon tissue, highlighting spatial domains and overlaying histology boundaries for the pathologist." |
| `Tutorials-space/t_commot_flowsig.ipynb` | "Model directional cell-cell communication in a wound-healing spatial dataset, emphasizing flow-based signaling patterns across the lesion margin." |
| `Tutorials-space/t_spaceflow.ipynb` | "Our lab wants to infer spatial velocity fields describing how cell states propagate across a tissue section. Walk through the SpaceFlow analysis and interpret hotspot regions." |
| `Tutorials-space/t_stt.ipynb` | "Guide me through segmenting and quantifying Slide-seq style beads, including spatial smoothing and downstream clustering for metabolic profiling." |
| `Tutorials-space/t_cellpose.ipynb` | "The imaging core needs automated single-cell segmentation on large histology mosaics. Provide instructions to run Cellpose, tune models for nuclei vs cytoplasm, and export masks." |
| `Tutorials-space/t_crop_rotate.ipynb` | "We received rotated tissue tiles from the scanner. Explain how to batch crop, rotate, and register the images so that downstream spatial analyses align with pathology annotations." |
| `Tutorials-space/t_mapping.ipynb` | "Our spatial proteomics dataset requires mapping cell types back to anatomical regions. Show how to align coordinate systems, transfer annotations, and validate mappings with marker panels." |
| `Tutorials-space/t_staligner.ipynb` | "We have serial tissue sections and need to align their molecular signals. Provide a workflow to register slices, correct batch variation, and produce a unified 3D expression map." |
| `Tutorials-space/t_slat.ipynb` | "Guide an analysis that leverages spatial latent topics to uncover recurring microenvironments in tumor resections, complete with interpretable topic labels." |
| `Tutorials-space/t_gaston.ipynb` | "Our pharma partner wants graph-based analysis of cell neighborhoods in spatial datasets. Walk through constructing the spatial graph, running community detection, and exporting quantitative summaries." |
| `Tutorials-space/t_stagate.ipynb` | "Perform spatial domain detection using a graph attention model so we can highlight tumor-stroma interfaces with confidence intervals." |

## Tutorials-single

| Tutorial reference | Real-world ov.Agent prompt |
| --- | --- |
| `Tutorials-single/t_scdeg.ipynb` | "Single-cell data from treated organoids needs condition-specific differential expression per cell type. Lead a pipeline that controls for donor effects and outputs ranked gene tables." |
| `Tutorials-single/t_scenic.ipynb` | "Help reconstruct regulatory networks from single-cell data by running SCENIC-like steps, highlighting transcription factor regulons active in exhausted T cells." |
| `Tutorials-single/t_cellfate_genesets.ipynb` | "Our stem cell group curated custom fate gene sets. Show how to score these programs across cells and visualize trajectories towards desired fates." |
| `Tutorials-single/t_scdrug.ipynb` | "A pharmacology screen collected single-cell profiles post drug treatment. Demonstrate how to predict compound sensitivity per cell state and highlight resistant subpopulations." |
| `Tutorials-single/t_cellmatch.ipynb` | "We want to match rare cell states across datasets from different labs. Provide a workflow for cross-dataset label transfer with confidence metrics." |
| `Tutorials-single/t_anno_noref.ipynb` | "Annotate a new single-cell atlas when no good reference exists by leveraging marker discovery, clustering, and expert-curated dictionaries." |
| `Tutorials-single/t_mofa.ipynb` | "Integrate multi-omics single-cell measurements using a factor analysis approach, then interpret latent factors tied to metabolic rewiring." |
| `Tutorials-single/t_scmulan.ipynb` | "Our team needs multi-label annotations because cells can express overlapping programs. Show how to run a multi-label classifier and evaluate per-label metrics." |
| `Tutorials-single/t_via_velo.ipynb` | "Combine velocity-informed transition graphs with trajectory inference to map branching decisions in neurogenesis." |
| `Tutorials-single/t_cellvote_pbmc3k.ipynb` | "We have multiple volunteer PBMC datasets with conflicting annotations. Demonstrate how to apply a voting-based harmonization strategy and report agreement statistics." |
| `Tutorials-single/t_anno_trans.ipynb` | "Transfer annotations from a well-characterized atlas into a new patient cohort while correcting for platform shifts." |
| `Tutorials-single/t_cellphonedb.ipynb` | "Elucidate ligand-receptor interactions between immune and stromal cells using curated interaction databases, summarizing key signaling axes." |
| `Tutorials-single/t_metatime.ipynb` | "Order cells along pseudotime in a way that respects cyclic programs such as cell cycle, and report checkpoints for synchronization experiments." |
| `Tutorials-single/t_deg_single.ipynb` | "Perform differential expression directly on single-cell counts, including hurdle-model style testing and visualization per cluster." |
| `Tutorials-single/t_lazy.ipynb` | "Assist with fast exploratory clustering and marker detection to triage whether a newly sequenced dataset is worth deeper analysis." |
| `Tutorials-single/t_nocd.ipynb` | "Detect abrupt transcriptomic changes in ordered single-cell data (e.g., along pseudotime) to flag branch points for validation." |
| `Tutorials-single/t_cnmf.ipynb` | "Run consensus NMF on single-cell data to uncover transcriptional programs shared across donors, and export program signatures." |
| `Tutorials-single/t_anno_ref.ipynb` | "Carry out reference-based annotation using a curated atlas, including evaluation of label transfer accuracy and confusion matrices." |
| `Tutorials-single/t_via.ipynb` | "Infer trajectories using VIA, highlighting branch probabilities and delivering publication-ready trajectory plots." |
| `Tutorials-single/t_traj.ipynb` | "Build an end-to-end trajectory inference workflow, from preprocessing to pseudotime ordering and branch-specific gene trends." |
| `Tutorials-single/t_preprocess.ipynb` | "Guide the entire preprocessing of raw 10x data: QC, normalization, highly variable gene selection, and scaling ready for downstream tasks." |
| `Tutorials-single/t_single_batch.ipynb` | "Integrate multiple donors with strong batch effects, evaluate alignment quality, and keep biological differences intact." |
| `Tutorials-single/t_cellfate.ipynb` | "Predict future cell fates using RNA velocity plus latent time, highlighting driver genes per lineage." |
| `Tutorials-single/t_cluster.ipynb` | "Deliver a robust clustering workflow with parameter sweeps, silhouette diagnostics, and marker-based cluster naming." |
| `Tutorials-single/t_simba.ipynb` | "Use graph representation learning to align multi-omic single-cell data and extract interpretable embeddings for downstream classifiers." |
| `Tutorials-single/t_cellfate_gene.ipynb` | "Score curated fate genes to quantify lineage bias across progenitors and visualize fate bias on embeddings." |
| `Tutorials-single/t_alignment_1k.ipynb` | "Align a small 1k-cell dataset to a large atlas, emphasizing lightweight preprocessing suitable for resource-constrained laptops." |
| `Tutorials-single/t_gptanno.ipynb` | "Demonstrate how to couple LLM-based annotation hints with canonical markers to name clusters responsibly, logging provenance for reviewers." |
| `Tutorials-single/t_metacells.ipynb` | "Construct metacells to denoise sparse droplet data and compare downstream differential analysis vs raw cells." |
| `Tutorials-single/t_aucell.ipynb` | "Score gene sets per cell using AUCell and highlight top-scoring programs for each cluster." |
| `Tutorials-single/t_stavia.ipynb` | "Link single-cell transcriptomes to matched spatial data using STAVIA-style alignment, producing spot-level label probabilities." |
| `Tutorials-single/t_preprocess_gpu.ipynb` | "Accelerate preprocessing of very large single-cell datasets using GPU-backed steps, documenting performance gains." |
| `Tutorials-single/t_mofa_glue.ipynb` | "Integrate modalities with MOFA+GLUE hybrid modeling to uncover shared and modality-specific factors." |
| `Tutorials-single/t_tosica.ipynb` | "Apply a transformer-based ontology-guided annotator to newly sequenced immune cells, and export ontology-consistent labels." |
| `Tutorials-single/t_alignment_velocity.ipynb` | "Align datasets while respecting velocity information so that dynamic processes are preserved during integration." |
| `Tutorials-single/t_cellanno.ipynb` | "Automate annotation of single-cell clusters using a supervised classifier, followed by manual review summaries for stakeholders." |
| `Tutorials-single/t_preprocess_cpu.ipynb` | "Document a CPU-only preprocessing path for institutions without GPU access, emphasizing reproducible runtime tracking." |

## Tutorials-llm

| Tutorial reference | Real-world ov.Agent prompt |
| --- | --- |
| `Tutorials-llm/t_dr.ipynb` | "Demonstrate how to use diffusion or dimensionality-reduction LLM helpers to summarize single-cell manifolds and narrate findings." |
| `Tutorials-llm/t_scgpt.ipynb` | "Fine-tune a gene expression language model on lab-specific datasets and evaluate its ability to predict perturbation responses." |
| `Tutorials-llm/t_scfoundation.ipynb` | "Set up workflows that compare outputs from multiple single-cell foundation models, logging prompts, responses, and benchmarks." |
| `Tutorials-llm/t_geneformer.ipynb` | "Leverage a pretrained gene transformer to score pathway activity across new patient samples, including embeddings export." |
| `Tutorials-llm/t_cellplm.ipynb` | "Use protein language model embeddings to relate receptor sequences to transcriptomic states within single-cell datasets." |
| `Tutorials-llm/t_uce.ipynb` | "Run the universal cell embedding pipeline to place diverse datasets into a shared reference frame and report outlier detection." |
| `Tutorials-llm/t_ov_agent_pbmc3k.ipynb` | "Walk through orchestrating ov.Agent itself on the classic PBMC3k dataset, including how to route tasks between analysis skills and summarize outputs for lab notebooks." |

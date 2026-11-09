# How the SecAct reference values in secact_pbmc_fixture.npz / secact_st_fixture.npz were made.
#
# Inputs (stored in the .npz files):
#   * secact_pbmc_fixture: 200 PBMC3k cells (40 each of 5 cell types), raw counts of 367 genes,
#     and a 350-gene x 37-protein subset of the SecAct signature (SecAct.tsv.gz, sha256 eda4d822...).
#   * secact_st_fixture: 443 CosMx cells (smi_2219104, FOVs 279-280, a square window; coordinates
#     in um = px * 0.12028), raw counts of 455 genes, a 447-gene x 20-protein signature subset.
#   Genes that R's transferSymbol() would rename were removed, so map_alias=False reproduces R.
#
# Reference: R SecAct, GitHub data2intelligence/SecAct @ 8d9ce8355b3c7c05fff28f4dc3d393747a9f93cc,
# R 4.4.3, pure-R ridge backend (GSL MT19937 permutation table). The package R files are sourced;
# Seurat / SpaCET objects are replaced by minimal S4 classes with the slots SecAct reads, and
# metap::sumlog by its formula (pchisq(-2 * sum(log(p)), 2 * length(p), lower.tail = FALSE)).
#
#   for (f in c("ifun.R", "ridge.R", "activity.R", "downstream.R")) source(file.path(SecAct, "R", f))
#   setClass("Assay",  representation(counts = "dgCMatrix"))
#   setClass("Seurat", representation(assays = "list", meta.data = "data.frame", misc = "list"))
#   setClass("SpaCET", representation(input = "list", results = "list"))
#
#   # scRNA-seq (secact_pbmc_fixture: r_*)
#   obj <- new("Seurat", assays = list(RNA = new("Assay", counts = counts)), meta.data = meta, misc = list())
#   obj <- SecAct.CCC.scRNAseq(obj, cellType_meta = "ct", sigMatrix = "pbmc_small_sig.tsv")
#
#   # spatial (secact_st_fixture: r_*); padj_cutoff = 1.01 keeps every tested edge
#   obj <- new("SpaCET", input = list(counts = counts, spotCoordinates = xy_um, metaData = meta,
#                                     organism = "human"), results = list())
#   obj <- SecAct.activity.inference.ST(obj, sigMatrix = "cosmx_small_sig.tsv", nrand = 100)
#   obj <- SecAct.CCC.scST(obj, cellType_meta = "ct", coreNo = 1, padj_cutoff = 1.01)
#
# Numbers were written with sprintf("%.17g"). The py_* arrays of secact_pbmc_fixture come from
# SecActpy (GitHub main 13bb6d2) secact_ccc_scrnaseq(adata, "ct", sig_matrix=<same subset>).

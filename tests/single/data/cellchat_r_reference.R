# R CellChat reference for tests/single/test_cellchat.py (cellchat_r_reference.npz).
#
# Input: 400 PBMC3k cells (scanpy pbmc3k, normalize_total 1e4 + log1p; 120 CD4 T,
# 100 CD14+ Mono, 80 B, 60 NK, 32 FCGR3A+ Mono, 8 Megakaryocytes so that
# filterCommunication removes one group), genes restricted to CellChatDB.human,
# written as X_genes_x_cells.mtx (17 significant digits) + genes.txt + meta.csv.
# Run with CellChat 2.2.0.9001 (jinworks/CellChat), presto 1.1.0, igraph 2.3.4:
#   Rscript cellchat_r_reference.R <dir>
# then pack the R_*.csv/txt outputs into the .npz (see the PR description).
suppressMessages({library(CellChat); library(Matrix)})
args <- commandArgs(TRUE); d <- args[1]
X <- as(readMM(file.path(d, "X_genes_x_cells.mtx")), "CsparseMatrix")
meta <- read.csv(file.path(d, "meta.csv"), row.names = 1)
rownames(X) <- readLines(file.path(d, "genes.txt")); colnames(X) <- rownames(meta)
cc <- createCellChat(object = X, meta = meta, group.by = "cell_type")
cc@DB <- CellChatDB.human                      # full DB (all categories)
cc <- subsetData(cc)
future::plan("sequential")
cc <- identifyOverExpressedGenes(cc)
cc <- identifyOverExpressedInteractions(cc)
cc <- computeCommunProb(cc, type = "triMean", nboot = 100, seed.use = 1L)
cc <- filterCommunication(cc, min.cells = 10)
cc <- computeCommunProbPathway(cc)
cc <- aggregateNet(cc)
p <- cc@net$prob; pv <- cc@net$pval
cat("prob dims:", dim(p), "\n")
groups <- dimnames(p)[[1]]; lrs <- dimnames(p)[[3]]
writeLines(groups, file.path(d, "R_groups.txt")); writeLines(lrs, file.path(d, "R_lr.txt"))
write.csv(format(as.vector(p), digits=17), file.path(d, "R_prob_flat.csv"), row.names = FALSE)
write.csv(format(as.vector(pv), digits=17), file.path(d, "R_pval_flat.csv"), row.names = FALSE)
write.csv(cc@LR$LRsig, file.path(d, "R_LRsig.csv"))
df <- subsetCommunication(cc); write.csv(df, file.path(d, "R_df_net.csv"), row.names = FALSE)
write.csv(as.data.frame(as.table(cc@net$weight)), file.path(d, "R_weight.csv"), row.names = FALSE)
write.csv(as.data.frame(as.table(cc@net$count)), file.path(d, "R_count.csv"), row.names = FALSE)
cat("n LRsig:", nrow(cc@LR$LRsig), " n significant rows:", nrow(df), " n pathways:", length(cc@netP$pathways), "\n")
writeLines(cc@var.features$features, file.path(d, "R_features_sig.txt"))
writeLines(cc@netP$pathways, file.path(d, "R_pathways.txt"))
writeLines(rownames(cc@LR$LRsig), file.path(d, "R_LRsig_ids.txt"))
write.csv(as.vector(cc@netP$prob), file.path(d, "R_netP_flat.csv"), row.names = FALSE)
cc <- netAnalysis_computeCentrality(cc, slot.name = "netP")
cen <- do.call(rbind, lapply(names(cc@netP$centr), function(pw) {
  x <- cc@netP$centr[[pw]]
  data.frame(pathway = pw, group = names(x$outdeg), outdeg = x$outdeg, indeg = x$indeg, hub = x$hub,
             authority = x$authority, eigen = x$eigen, page_rank = x$page_rank, betweenness = x$betweenness,
             flowbet = as.numeric(x$flowbet), info = as.numeric(x$info)) }))
write.csv(cen, file.path(d, "R_centrality.csv"), row.names = FALSE)
cat("features.sig:", length(cc@var.features$features), "\n")

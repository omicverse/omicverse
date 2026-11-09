# Reference run of R CytoSignal (vignette defaults) with binary dumps for parity.
suppressMessages({library(cytosignal); library(Matrix)})
a <- commandArgs(TRUE)
datdir <- a[1]; out <- a[2]; perm.size <- as.numeric(a[3]); keep.file <- if (length(a) >= 4) a[4] else ""; rth <- if (length(a) >= 5) as.numeric(a[5]) else 100; sth <- if (length(a) >= 6) as.numeric(a[6]) else 100
dir.create(out, showWarnings = FALSE, recursive = TRUE)
wbin <- function(m, name) {
  m <- as.matrix(m); con <- file(file.path(out, paste0(name, ".bin")), "wb")
  writeBin(as.double(as.vector(m)), con, size = 8); close(con)
  writeLines(as.character(c(nrow(m), ncol(m))), file.path(out, paste0(name, ".dim")))
  if (!is.null(colnames(m))) writeLines(colnames(m), file.path(out, paste0(name, ".cols")))
  if (!is.null(rownames(m))) writeLines(rownames(m), file.path(out, paste0(name, ".rows")))
}
wsp <- function(m, name) {
  m <- as(m, "TsparseMatrix")
  df <- data.frame(i = m@i, j = m@j); con <- file(file.path(out, paste0(name, ".x.bin")), "wb")
  writeBin(as.double(m@x), con, size = 8); close(con)
  write.csv(df, gzfile(file.path(out, paste0(name, ".ij.csv.gz"))), row.names = FALSE)
  writeLines(as.character(c(nrow(m), ncol(m))), file.path(out, paste0(name, ".dim")))
}
T <- list(); t0 <- Sys.time()
dge <- readRDS(file.path(datdir, "SCP2170_annotated_dgCMatrix.rds"))
cluster <- read.csv(file.path(datdir, "SCP2170_cluster.csv"))
cluster <- factor(cluster$cell_type); names(cluster) <- colnames(dge)
spatial <- as.matrix(read.csv(file.path(datdir, "SCP2170_spatial.csv"), row.names = 1)); colnames(spatial) <- c("x", "y")
if (nchar(keep.file) > 0) {
  keep <- readLines(keep.file); dge <- dge[, keep]; spatial <- spatial[keep, ]; cluster <- cluster[keep, drop = TRUE]
}
T$load <- as.numeric(Sys.time() - t0, units = "secs"); t0 <- Sys.time()
cs <- createCytoSignal(raw.data = dge, cells.loc = spatial, clusters = cluster)
cs <- addIntrDB(cs, g_to_u, db.diff, db.cont, inter.index)
cs <- removeLowQuality(cs, counts.thresh = 300)
cs <- changeUniprot(cs)
cs <- inferEpsParams(cs, scale.factor = 0.73)
cs <- findNN(cs)
cs <- imputeLR(cs)
T$nn_impute <- as.numeric(Sys.time() - t0, units = "secs"); t0 <- Sys.time()
set.seed(42)
cs <- inferIntrScore(cs, perm.size = perm.size)
T$score_perm <- as.numeric(Sys.time() - t0, units = "secs"); t0 <- Sys.time()
cs <- inferSignif(cs, p.thresh = 0.05, reads.thresh = rth, sig.thresh = sth)
T$signif <- as.numeric(Sys.time() - t0, units = "secs")
writeLines(colnames(cs@counts), file.path(out, "cells.txt"))
writeLines(rownames(cs@counts), file.path(out, "uniprot_rows.txt"))
writeLines(as.character(cs@clusters), file.path(out, "clusters.txt"))
wbin(matrix(cs@parameters$lib.size, ncol = 1), "lib_size")
writeLines(format(c(cs@parameters$r.diffuse.scale, cs@parameters$sigma.scale), digits = 17), file.path(out, "eps_sigma.txt"))
wsp(cs@imputation$diffusion@nn.graph, "nn_gaueps")
wsp(cs@imputation$contact@nn.graph, "nn_dt")
wbin(cs@imputation$diffusion@imp.data, "imp_gaueps")
wbin(matrix(cs@imputation$diffusion@scale.fac, ncol = 1), "scalefac_gaueps")
wbin(cs@imputation$contact@imp.data, "imp_dt")
for (s in setdiff(names(cs@lrscore), "default")) {
  o <- cs@lrscore[[s]]; tag <- gsub("[^A-Za-z]", "_", s)
  wbin(o@score, paste0("score_", tag))
  if (grepl("smooth", s)) {
    nn <- cs@imputation[[o@recep.slot]]@nn.graph
    has <- diff(nn@p) > 0
    sc <- o@score[has, ]
    pv <- cytosignal:::getPvalues(sc, o@score.null)
    pa <- cytosignal:::graphSpatialFDRNew(nn[has, has], pv)
    dimnames(pa) <- dimnames(sc)
    wbin(pv, paste0("pval_", tag)); wbin(pa, paste0("padj_", tag))
    for (r in names(o@res.list)) {
      rl <- o@res.list[[r]]
      df <- data.frame(intr = rep(names(rl), lengths(rl)), cell = unlist(rl, use.names = FALSE))
      write.csv(df, gzfile(file.path(out, paste0("res_", tag, "_", r, ".csv.gz"))), row.names = FALSE)
    }
  } else {
    writeLines(as.character(unlist(lapply(o@perm.idx$perm, function(p) p[1:20]))), file.path(out, paste0("perm_head_", tag, ".txt")))
    writeLines(as.character(o@perm.idx$sample[1:50]), file.path(out, paste0("sample_head_", tag, ".txt")))
    wbin(o@score.null[1:min(2000, nrow(o@score.null)), , drop = FALSE], paste0("null_head_", tag))
  }
}
writeLines(paste(names(T), unlist(T)), file.path(out, "timing.txt"))
print(T)

# Run from the repository root with base R; no RUVSeq installation is needed.
# Execute the unmodified upstream matrix-method body, not a rewritten R oracle.
# This deliberately excludes the SeqExpressionSet method and package integration.
sha <- "f560e84eb493f0c2d72279612f9e4ca99493b434"
base <- paste0("https://raw.githubusercontent.com/drisso/RUVSeq/", sha, "/R/")
oracle <- new.env(parent = globalenv())
oracle$setMethod <- function(f, signature, definition, ...) {
    if (identical(unname(signature["x"]), "matrix")) oracle$ruvg <- definition
}
oracle$signature <- methods::signature
for (file in c("helper.R", "RUVg-methods.R")) {
    target <- tempfile(fileext = ".R")
    download.file(paste0(base, file), target, quiet = TRUE, mode = "wb")
    sys.source(target, envir = oracle)
    unlink(target)
}
stopifnot(is.function(oracle$ruvg))
destination <- "tests/bulk/fixtures/ruvg"
dir.create(destination, recursive = TRUE, showWarnings = FALSE)
set.seed(1701)
x <- matrix(rpois(18 * 12, lambda = 20), nrow = 18)
x[, 7:12] <- x[, 7:12] * 3L
x[1, 1:3] <- 0
controls <- 1:8
cases <- list(
    counts = list(x = x, k = 2),
    unrounded = list(x = x, k = 2, round = FALSE),
    uncentered = list(x = x, k = 2, center = FALSE, round = FALSE),
    dropped = list(x = x, k = 3, drop = 1, round = FALSE),
    logged = list(x = log(x + 1) - 1.5, k = 2, isLog = TRUE),
    offset = list(x = x, k = 2, epsilon = 0.5, round = FALSE),
    rank_one = list(x = rbind(matrix(rep(1:12, 8), nrow = 8, byrow = TRUE), x[9:18, ]),
                    k = 3, round = FALSE)
)
for (name in names(cases)) {
    args <- cases[[name]]
    result <- do.call(oracle$ruvg, c(args, list(cIdx = controls)))
    write.table(args$x, file.path(destination, paste0(name, "_input.csv")),
                sep = ",", row.names = FALSE, col.names = FALSE)
    write.table(result$W %*% t(result$W), file.path(destination, paste0(name, "_projection.csv")),
                sep = ",", row.names = FALSE, col.names = FALSE)
    write.table(result$normalizedCounts, file.path(destination, paste0(name, "_corrected.csv")),
                sep = ",", row.names = FALSE, col.names = FALSE)
}
writeLines(c(paste("Upstream RUVSeq commit:", sha), "Upstream DESCRIPTION version: 1.33.1",
             R.version.string, "Seed: 1701; control rows: 1:8 (R indexing)",
             "Executed upstream matrix method without installing RUVSeq/EDASeq.",
             "Gate fixed before comparison: projection atol=1e-10; correction atol=1e-8, rtol=1e-10.",
             "Rounded count outputs must match exactly."),
           file.path(destination, "PROVENANCE.txt"))

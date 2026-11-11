# Roadmap #989 standalone adapters

Algorithms live in independent packages. OmicVerse imports each backend only when called. These are review-stage source installs, not published PyPI releases. Base OmicVerse does not require these backends.

## Pinned installation

Use a separate analysis environment with these tested revisions:

    python -m pip install "git+https://github.com/omicverse/py-RUVSeq.git@9efdf826a5fb637253c08a743610b4401e1d7da6"
    python -m pip install "git+https://github.com/omicverse/py-SPARKX.git@fe490febbaf28e2c9a4eecd6cd0005303a979020"
    python -m pip install "git+https://github.com/omicverse/py-MuSiC.git@b23fcba0d35ef42c592420f243dd6bfb78050d1d"
    python -m pip install "git+https://github.com/omicverse/py-muscat.git@7b9c1a8394b1aa99203b5474f9f2f9ab1d1dba79"
    python -m pip install "git+https://github.com/hurry060215-tech/scranPY.git@17b6aeabd02fa6ceae8e4f5e3838a10276337b29"
    python -m pip install "git+https://github.com/hurry060215-tech/pyNebulosa.git@3f2678cb7b0825e669608e8734d894f6598efd86"

| Backend | Public API |
|---|---|
| omicverse/py-RUVSeq | ov.bulk.ruvg / ruvr / ruvs |
| omicverse/py-SPARKX | ov.space.sparkx |
| omicverse/py-MuSiC | ov.bulk.music_basis / music_prop |
| omicverse/py-muscat | ov.single.muscat_aggregate / muscat_pb_ds |
| hurry060215-tech/scranPY | ov.pp.scran_pooling |
| hurry060215-tech/pyNebulosa | ov.pl.nebulosa_density |

## Scope

- RUVSeq: genes x samples. DataFrame controls are labels; array controls are zero-based positions. RUVr requires aligned residuals; RUVs requires zero-based replicate indices padded with -1. Use W in the DE design and original counts for count-model testing.
- SPARK-X: AnnData cells x genes, count layer and spatial coordinates. Sparse counts are transposed without densification. Returns gene-labelled tables and filtering masks without changing AnnData.
- MuSiC: genes x cells reference and genes x samples bulk DataFrames, with aligned cell type/donor Series. Only the default independent cell-type variance path is covered; check convergence diagnostics.
- muscat: sum pseudobulks by cluster and biological sample, retaining absent combinations. Explicit sample-indexed designs and coefficient-indexed contrasts. edgeR and limma-voom return local and per-contrast global BH; untestable clusters are reported.
- scran pooling: patched scranPY QR only, dense counts, >=101 cells per cluster. No implicit densification, CVXPY constraint, cluster splitting or small-cluster fallback. Returns factors without modifying input. Fix: sfortma2/scranPY#5.
- Nebulosa: patched wkde grid with required explicit bandwidth, e.g. R ks::hpi output. Default Silverman bandwidth and scipy's ks replacement are not selected. Returns per-cell density. Fix: mdmanurung/pyNebulosa#10.

## Examples

    W = ov.bulk.ruvg(counts_df, control_genes, k=2)["W"]
    svg = ov.space.sparkx(adata, layer="counts", spatial_key="spatial")
    pb = ov.single.muscat_aggregate(adata, cluster_key="cell_type", sample_key="sample", group_key="condition")
    ds = ov.single.muscat_pb_ds(pb, design, contrasts, method="limma-voom")

## Validation

Standalone repositories include frozen R oracles, export coverage manifests and four executed notebooks each. Numerical gates were fixed before implementation. muscat additionally uses the public Kang subset; synthetic fixtures do not establish biological accuracy.

    python -m pytest tests/test_roadmap_standalone_ports.py tests/bulk/test_ruvg.py -q
    python scripts/validate_roadmap_ports.py --ports-root PATH

The first command checks adapter contracts and optional RUVg integration. Missing real backends cause skips, not successful backend validation. The second exercises all six real backends against the standalone/fork fixture layout (py-RUVSeq, py-SPARKX, py-MuSiC, py-muscat, references/scranPY and references/pyNebulosa).

# Roadmap #989 standalone adapters

Algorithms live in independent packages. OmicVerse imports each backend only when called. These four backends have published, installed-wheel-validated PyPI releases. Base OmicVerse does not require these backends.

## Published installation

The four optional backends require **Python 3.10 or newer**. On a supported
interpreter, install the tested PyPI 0.1.0 distributions directly:

    python -m pip install py-ruvseq-omicverse==0.1.0 py-sparkx-omicverse==0.1.0 py-music-omicverse==0.1.0 py-muscat-omicverse==0.1.0

On Python 3.10+, the optional `roadmap-ports` extra installs those same pins
from a source checkout that includes these adapters:

    python -m pip install ".[roadmap-ports]"

On Python 3.9, the extra selects none of the four backends. Calling an adapter
on Python 3.9 raises an error with instructions for switching to a supported
environment. The base OmicVerse Python requirement is unchanged, and the base
package does not install or import these backends eagerly.

| Distribution | Import module | Public API |
|---|---|---|
| py-ruvseq-omicverse | pyruvseq | ov.bulk.ruvg / ruvr / ruvs |
| py-sparkx-omicverse | pysparkx | ov.space.sparkx |
| py-music-omicverse | pymusic_deconv | ov.bulk.music_basis / music_prop |
| py-muscat-omicverse | pymuscat | ov.single.muscat_aggregate / muscat_pb_ds |

scran pooling and Nebulosa are deferred and are not included in this PR.

## Scope

- RUVSeq: genes x samples. DataFrame controls are labels; array controls are zero-based positions. RUVr requires aligned residuals; RUVs requires zero-based replicate indices padded with -1. Use W in the DE design and original counts for count-model testing.
- SPARK-X: AnnData cells x genes, count layer and spatial coordinates. Sparse counts are transposed without densification. Returns gene-labelled tables and filtering masks without changing AnnData.
- MuSiC: genes x cells reference and genes x samples bulk DataFrames, with aligned cell type/donor Series. Only the default independent cell-type variance path is covered; check convergence diagnostics.
- muscat: sum pseudobulks by cluster and biological sample, retaining absent combinations. Explicit sample-indexed designs and coefficient-indexed contrasts. edgeR and limma-voom return local and per-contrast global BH; untestable clusters are reported.

## Examples

    W = ov.bulk.ruvg(counts_df, control_genes, k=2)["W"]
    svg = ov.space.sparkx(adata, layer="counts", spatial_key="spatial")
    pb = ov.single.muscat_aggregate(adata, cluster_key="cell_type", sample_key="sample", group_key="condition")
    ds = ov.single.muscat_pb_ds(pb, design, contrasts, method="limma-voom")

## Validation

Standalone repositories include frozen R oracles, export coverage manifests and four executed notebooks each. Numerical gates were fixed before implementation. muscat additionally uses the public Kang subset; synthetic fixtures do not establish biological accuracy.

    python -m pytest tests/test_roadmap_standalone_ports.py tests/bulk/test_ruvg.py -q
    python scripts/validate_roadmap_ports.py --ports-root PATH

The first command checks adapter contracts and optional RUVg integration. Missing real backends cause skips, not successful backend validation. The second exercises all four real backends against the standalone/fork fixture layout (py-RUVSeq, py-SPARKX, py-MuSiC, py-muscat).

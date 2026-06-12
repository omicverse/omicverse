"""Tests for ov.pl.cnv_heatmap chromosome-segment tiling.

Regression coverage for the bug where a non-standard scaffold/alt-contig
interleaved between two standard chromosomes left an uncovered hole in the
rendered range, so a chromosome past the gap rendered only partially / on one
side. The invariant under test: the segments returned by ``_build_chr_segments``
ALWAYS tile ``[0, width)`` of the selected columns with no holes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

_cnv = pytest.importorskip("omicverse.pl._cnv")
_build_chr_segments = _cnv._build_chr_segments


def _width(selector, n_bins: int) -> int:
    """Number of columns a slice / fancy-index selector picks out of n_bins."""
    return int(np.zeros((2, n_bins))[:, selector].shape[1])


def _assert_tiles(segments, selector, n_bins):
    """Segments must tile [0, width) with no holes and start at 0."""
    width = _width(selector, n_bins)
    assert segments, "expected non-empty segments"
    assert segments[0][1] == 0
    assert segments[-1][2] == width
    assert sum(e - s for _, s, e in segments) == width
    # contiguity: each segment starts where the previous ended
    for (_, _, prev_end), (_, cur_start, _) in zip(segments, segments[1:]):
        assert cur_start == prev_end
    return width


def test_scaffold_interleaved_between_standard_chromosomes():
    # GL000220.1 sits between chr1 and chr2 -> must be dropped from BOTH the
    # ideogram and the rendered columns, leaving no hole.
    chr_pos = {"chr1": 0, "GL000220.1": 500, "chr2": 1000}
    segments, selector = _build_chr_segments(chr_pos, 1500, standard_only=True)

    assert [c for c, _, _ in segments] == ["chr1", "chr2"]
    width = _assert_tiles(segments, selector, 1500)
    assert width == 1000  # 500 (chr1) + 500 (chr2); scaffold's 500 dropped

    # selector must pick exactly chr1[0:500] + chr2[1000:1500] — guards against
    # a "right width, wrong columns" implementation.
    picked = np.arange(1500)[selector]
    np.testing.assert_array_equal(picked, np.r_[0:500, 1000:1500])

    # chrom_per_bin (as the marsilea branch builds it) must be fully covered.
    chrom_per_bin = np.empty(width, dtype=object)
    for c, s, e in segments:
        chrom_per_bin[s:e] = c.replace("chr", "")
    assert None not in set(chrom_per_bin.tolist())


def test_normal_contiguous_chromosomes_tile_full_width():
    chr_pos = {"1": 0, "2": 300, "3": 700}
    segments, selector = _build_chr_segments(chr_pos, 1000, standard_only=True)
    assert [c for c, _, _ in segments] == ["1", "2", "3"]
    width = _assert_tiles(segments, selector, 1000)
    assert width == 1000


def test_standard_only_false_keeps_everything():
    chr_pos = {"chr1": 0, "GL000220.1": 500, "chr2": 1000}
    segments, selector = _build_chr_segments(chr_pos, 1500, standard_only=False)
    assert selector == slice(0, 1500)
    assert [c for c, _, _ in segments] == ["chr1", "GL000220.1", "chr2"]
    assert _width(selector, 1500) == 1500


def test_trailing_scaffolds_dropped():
    # Common real case: scaffolds come last -> still tiles with no hole.
    chr_pos = {"chr1": 0, "chr2": 400, "KI270728.1": 900}
    segments, selector = _build_chr_segments(chr_pos, 1000, standard_only=True)
    assert [c for c, _, _ in segments] == ["chr1", "chr2"]
    width = _assert_tiles(segments, selector, 1000)
    assert width == 900


def test_all_standard_returns_view_slice():
    # Nothing dropped -> selector is a slice (a view), not a fancy index (copy).
    chr_pos = {"chr1": 0, "chr2": 400, "chr3": 700}
    _, selector = _build_chr_segments(chr_pos, 1000, standard_only=True)
    assert selector == slice(0, 1000)


def test_load_centromeres_bundled():
    cen38 = _cnv._load_centromeres("hg38")
    cen19 = _cnv._load_centromeres("hg19")
    assert cen38 and cen19
    # chr8 centromere is ~45 Mb in both builds (sanity, from UCSC cytoBand)
    assert 40_000_000 < cen38["chr8"] < 50_000_000
    assert _cnv._load_centromeres("nonexistent_build") == {}


def test_split_segments_by_arm_splits_and_tiles():
    cen = _cnv._load_centromeres("hg38")
    # one chr8 segment over rendered cols [0,200), bins uniformly spanning 0..90Mb
    bin_starts = np.linspace(0, 90_000_000, 200)
    out = _cnv._split_segments_by_arm([("chr8", 0, 200)], bin_starts, cen)
    k = int((bin_starts < cen["chr8"]).sum())
    assert out == [("chr8p", 0, k), ("chr8q", k, 200)]
    # tiling preserved
    assert out[0][1] == 0 and out[-1][2] == 200
    assert sum(e - s for _, s, e in out) == 200


def test_split_segments_by_arm_q_only_chromosome():
    cen = _cnv._load_centromeres("hg38")
    # all bins past the centromere -> single q-arm segment (acrocentric-like)
    bin_starts = np.linspace(cen["chr14"] + 1, cen["chr14"] + 10_000_000, 50)
    out = _cnv._split_segments_by_arm([("chr14", 0, 50)], bin_starts, cen)
    assert out == [("chr14q", 0, 50)]


def test_split_segments_by_arm_unknown_chrom_passthrough():
    out = _cnv._split_segments_by_arm([("scaffoldX", 0, 10)], np.arange(10), {"chr1": 5})
    assert out == [("scaffoldX", 0, 10)]


def test_cnv_heatmap_split_arms_renders():
    ad = pytest.importorskip("anndata")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    cnv_heatmap = _cnv.cnv_heatmap

    # two chromosomes, 200 bins each; bin_meta gives genomic positions so the
    # centromere split can land inside each chromosome.
    n_cells = 10
    per = 200
    starts1 = np.linspace(0, 145_000_000, per)   # chr1 spans the centromere
    starts8 = np.linspace(0, 145_000_000, per)   # chr8 spans the centromere
    bin_meta = pd.DataFrame(
        {
            "chromosome": ["chr1"] * per + ["chr8"] * per,
            "start": np.r_[starts1, starts8].astype(int),
            "end": np.r_[starts1, starts8].astype(int) + 1,
        }
    )
    X = np.zeros((n_cells, 2 * per), dtype="float32")
    adata = ad.AnnData(X=np.zeros((n_cells, 5), dtype="float32"))
    adata.obsm["X_cnv"] = X
    adata.uns["cnv"] = {
        "chr_pos": {"chr1": 0, "chr8": per},
        "bin_meta": bin_meta,
        "method": "infercnv",
    }
    fig, axes = cnv_heatmap(adata, backend="matplotlib", show=False,
                            split_arms=True, genome="hg38")
    assert set(axes) >= {"heatmap", "ideogram"}
    # without arm split there would be 2 segments; with it, 4 (1p,1q,8p,8q)
    fig2, _ = cnv_heatmap(adata, backend="matplotlib", show=False, split_arms=False)
    assert fig is not None and fig2 is not None


def test_matplotlib_backend_with_groupby_renders():
    # Regression for the `primary`->`groupby` NameError + scaffold tiling in the
    # real render path. Skips cleanly if heavy deps are unavailable.
    ad = pytest.importorskip("anndata")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    cnv_heatmap = _cnv.cnv_heatmap

    n_cells, n_bins = 12, 30
    X = np.zeros((n_cells, n_bins), dtype="float32")
    obs = pd.DataFrame(
        {"grp": pd.Categorical(["A"] * 6 + ["B"] * 6)},
        index=[f"c{i}" for i in range(n_cells)],
    )
    adata = ad.AnnData(X=np.zeros((n_cells, 5), dtype="float32"), obs=obs)
    adata.obsm["X_cnv"] = X
    # scaffold interleaved between chr1 and chr2 exercises the tiling fix too
    adata.uns["cnv"] = {
        "chr_pos": {"chr1": 0, "GL000220.1": 10, "chr2": 20},
        "method": "infercnv",
    }

    fig, axes = cnv_heatmap(adata, groupby="grp", backend="matplotlib", show=False)
    assert set(axes) >= {"heatmap", "ideogram"}

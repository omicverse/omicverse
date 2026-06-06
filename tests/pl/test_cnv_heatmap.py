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

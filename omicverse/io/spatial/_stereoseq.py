"""BGI Stereo-seq reader for OmicVerse spatial I/O.

Stereo-seq writes a GEM file: one row per (gene, x, y) with a count, at a spot
pitch of 500 nm. That resolution is below one cell, so nothing downstream is done
at native resolution — the first real decision in every Stereo-seq analysis is the
bin size, and it changes every result after it. ``bin_size=50`` (25 x 25 um) is the
common default and roughly cell-scale; smaller bins are sparser and noisier, larger
ones blur cell types together.

This reader does the aggregation explicitly and records what it did in
``adata.uns['stereoseq']``, so a downstream figure can always be traced back to the
bin it was drawn at.

Issue #760 asked for Stereo-seq support; before this, the BGI ecosystem was
reachable only by going through Stereopy first.
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from ..._registry import register_function

__all__ = ["read_stereoseq", "bin_stereoseq"]


def _open(path: Path):
    """GEM files ship gzipped as often as not."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def _read_header(path: Path) -> dict:
    """The ``#key=value`` preamble a GEM carries, if it has one."""
    meta: dict[str, str] = {}
    with _open(path) as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            if "=" in line:
                key, _, value = line[1:].strip().partition("=")
                meta[key.strip()] = value.strip()
    return meta


@register_function(
    aliases=["Stereo-seq读取", "read stereoseq", "读取GEM", "时空组学读取", "BGI空间转录组"],
    category="io",
    description="Read a BGI Stereo-seq GEM file and aggregate it into square bins",
    requires={},
    produces={'obsm': ['spatial'], 'uns': ['stereoseq']},
    auto_fix='none',
    examples=[
        "adata = ov.io.spatial.read_stereoseq('sample.gem.gz', bin_size=50)",
        "adata = ov.io.spatial.read_stereoseq('sample.gem.gz', bin_size=20, min_counts=10)",
    ],
)
def read_stereoseq(
    path: Union[str, Path],
    *,
    bin_size: int = 50,
    min_counts: int = 0,
    min_genes: int = 0,
    label_column: Optional[str] = None,
    chunksize: Optional[int] = None,
    dtype: str = "float32",
) -> AnnData:
    """Read a Stereo-seq GEM file into an AnnData at a chosen bin size.

    A GEM is a long table — ``geneID``, ``x``, ``y``, ``MIDCount`` — recorded on a
    500 nm grid. Bins of ``bin_size`` spots square are summed to make the
    observations, so ``bin_size=50`` gives 25 x 25 um bins.

    That choice is not cosmetic. It sets what an "observation" is for everything
    downstream: clustering, deconvolution, the niche statistics. Bins much smaller
    than a cell give sparse, noisy profiles; bins much larger mix cell types
    together and no deconvolution will fully undo that. The chosen size is recorded
    in ``adata.uns['stereoseq']['bin_size']``.

    If the file carries a cell-segmentation column, pass its name as
    ``label_column`` and the reader aggregates by cell instead of by bin, which is
    the better route when segmentation is available.

    Arguments:
        path: The ``.gem`` or ``.gem.gz`` file.
        bin_size: Side of the square bin, in native 500 nm spots. Default 50, i.e.
            25 x 25 um. Ignored when ``label_column`` is given.
        min_counts: Drop bins with fewer total counts than this. Default 0.
        min_genes: Drop bins expressing fewer genes than this. Default 0.
        label_column: Column naming a pre-computed cell label. When given, bins are
            replaced by those cells and ``bin_size`` is not used.
        chunksize: Read the table in chunks of this many rows, for files that do
            not fit in memory. Default reads in one pass.
        dtype: Data type of the count matrix. Default ``'float32'``.

    Returns:
        AnnData with counts in ``.X``, bin centres in ``.obsm['spatial']``, and the
        bin geometry in ``.uns['stereoseq']``.

    Examples:
        >>> import omicverse as ov
        >>> adata = ov.io.spatial.read_stereoseq('E14-16h.gem.gz', bin_size=50)
        >>> adata.uns['stereoseq']['bin_size']
        50
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No Stereo-seq GEM at {path}")
    if bin_size < 1:
        raise ValueError("bin_size must be >= 1.")

    header = _read_header(path)
    n_skip = len(header) if header else 0

    frames = pd.read_csv(path, sep="\t", comment="#", header=0,
                         chunksize=chunksize) if chunksize else [
        pd.read_csv(path, sep="\t", comment="#", header=0)
    ]

    pieces = []
    for chunk in frames:
        pieces.append(_bin_chunk(chunk, bin_size, label_column))
    table = pd.concat(pieces, ignore_index=True) if len(pieces) > 1 else pieces[0]
    table = table.groupby(["bin", "geneID"], observed=True, as_index=False)["count"].sum()

    adata = _to_anndata(table, dtype=dtype)

    # bin centres in native spot units
    if label_column is None:
        parts = adata.obs_names.str.split("_", expand=False)
        xy = np.array([[int(p[0]), int(p[1])] for p in parts], dtype=np.float64)
        adata.obsm["spatial"] = (xy + 0.5) * bin_size
    else:
        centres = table.merge(_bin_centres(pieces), on="bin", how="left")
        adata.obsm["spatial"] = centres.drop_duplicates("bin").set_index("bin").loc[
            adata.obs_names, ["x", "y"]
        ].to_numpy(dtype=np.float64)

    keep = np.ones(adata.n_obs, dtype=bool)
    if min_counts > 0:
        keep &= np.asarray(adata.X.sum(axis=1)).ravel() >= min_counts
    if min_genes > 0:
        keep &= np.asarray((adata.X > 0).sum(axis=1)).ravel() >= min_genes
    if not keep.all():
        adata = adata[keep].copy()

    adata.uns["stereoseq"] = {
        "bin_size": None if label_column else int(bin_size),
        "bin_size_um": None if label_column else bin_size * 0.5,
        "aggregated_by": label_column or "square bins",
        "source": str(path),
        "header": header,
        "spot_pitch_um": 0.5,
    }
    return adata


def _bin_chunk(chunk: pd.DataFrame, bin_size: int, label_column: Optional[str]) -> pd.DataFrame:
    """Reduce raw GEM rows to (bin, gene, count)."""
    cols = {c.lower(): c for c in chunk.columns}
    gene = cols.get("geneid") or cols.get("gene")
    x, y = cols.get("x"), cols.get("y")
    count = cols.get("midcount") or cols.get("umicount") or cols.get("count") or cols.get("mid_count")
    if gene is None or x is None or y is None:
        raise ValueError(
            "A Stereo-seq GEM needs geneID, x and y columns; found "
            f"{list(chunk.columns)}."
        )
    if count is None:
        chunk = chunk.assign(_count=1)
        count = "_count"

    if label_column is not None:
        if label_column not in chunk.columns:
            raise KeyError(f"No column {label_column!r} in the GEM; found {list(chunk.columns)}.")
        key = chunk[label_column].astype(str)
    else:
        key = ((chunk[x] // bin_size).astype(int).astype(str) + "_"
               + (chunk[y] // bin_size).astype(int).astype(str))

    out = pd.DataFrame({"bin": key, "geneID": chunk[gene].astype(str),
                        "count": chunk[count].astype(np.int64),
                        "x": chunk[x].astype(np.float64), "y": chunk[y].astype(np.float64)})
    return out.groupby(["bin", "geneID"], observed=True, as_index=False).agg(
        count=("count", "sum"), x=("x", "mean"), y=("y", "mean")
    )


def _bin_centres(pieces) -> pd.DataFrame:
    frame = pd.concat(pieces, ignore_index=True)
    return frame.groupby("bin", observed=True, as_index=False)[["x", "y"]].mean()


def _to_anndata(table: pd.DataFrame, dtype: str) -> AnnData:
    bins = pd.Index(pd.unique(table["bin"]), name=None)
    genes = pd.Index(pd.unique(table["geneID"]), name=None)
    row = pd.Categorical(table["bin"], categories=bins).codes
    col = pd.Categorical(table["geneID"], categories=genes).codes
    matrix = sparse.coo_matrix(
        (table["count"].to_numpy(), (row, col)), shape=(len(bins), len(genes))
    ).tocsr().astype(dtype)
    return AnnData(X=matrix, obs=pd.DataFrame(index=bins.astype(str)),
                   var=pd.DataFrame(index=genes.astype(str)))


@register_function(
    aliases=["重新分箱", "rebin", "bin stereoseq", "改变bin大小"],
    category="io",
    description="Re-aggregate an already-loaded Stereo-seq AnnData to a coarser bin size",
    requires={'obsm': ['spatial'], 'uns': ['stereoseq']},
    produces={'obsm': ['spatial'], 'uns': ['stereoseq']},
    auto_fix='none',
    examples=[
        "coarse = ov.io.spatial.bin_stereoseq(adata, bin_size=100)",
    ],
)
def bin_stereoseq(adata: AnnData, bin_size: int, *, dtype: str = "float32") -> AnnData:
    """Re-aggregate an already-binned Stereo-seq object to a coarser bin.

    Bin size is the one Stereo-seq choice worth revisiting, and re-reading a GEM to
    try another value is slow. Coarsening an object that is already in memory is
    cheap, and exact as long as the new bin is a whole multiple of the old one.

    Arguments:
        adata: An AnnData produced by :func:`read_stereoseq`.
        bin_size: The new bin side in native 500 nm spots. Must be a multiple of the
            current one.
        dtype: Data type of the new count matrix.

    Returns:
        A new AnnData at the coarser bin size.
    """
    meta = adata.uns.get("stereoseq")
    if not meta or meta.get("bin_size") is None:
        raise ValueError(
            "This object was not produced by read_stereoseq with square bins, so "
            "its current bin size is unknown. Re-read the GEM instead."
        )
    old = int(meta["bin_size"])
    if bin_size <= old:
        raise ValueError(f"bin_size must be coarser than the current {old}.")
    if bin_size % old:
        raise ValueError(
            f"bin_size {bin_size} is not a whole multiple of the current {old}; "
            "the bins would not nest and counts would be split across boundaries."
        )

    factor = bin_size // old
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    grid = np.floor(coords / bin_size).astype(int)
    keys = pd.Index([f"{a}_{b}" for a, b in grid])
    uniq = pd.unique(keys)
    codes = pd.Categorical(keys, categories=uniq).codes

    agg = sparse.csr_matrix(
        (np.ones(len(codes)), (codes, np.arange(len(codes)))),
        shape=(len(uniq), adata.n_obs),
    )
    matrix = (agg @ sparse.csr_matrix(adata.X)).astype(dtype)

    out = AnnData(X=matrix, obs=pd.DataFrame(index=pd.Index(uniq).astype(str)),
                  var=adata.var.copy())
    centres = np.array([[int(k.split("_")[0]), int(k.split("_")[1])] for k in uniq],
                       dtype=np.float64)
    out.obsm["spatial"] = (centres + 0.5) * bin_size
    out.uns["stereoseq"] = {**meta, "bin_size": int(bin_size),
                            "bin_size_um": bin_size * 0.5,
                            "rebinned_from": old, "rebin_factor": int(factor)}
    return out

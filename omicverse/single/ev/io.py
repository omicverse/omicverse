"""I/O for single-extracellular-vesicle (single-EV) proteomics data.

Single-EV proteomics measures the protein content of *individual* extracellular
vesicles (EVs). The natural data structure is an **EV x protein matrix** —
directly analogous to a single-cell *cell x gene* matrix, where a vesicle plays
the role of a cell and a protein/marker target plays the role of a gene. Three
measurement modalities produce three value types:

* **sequencing counts** — PBA (Proximity Barcoding Assay), DBS-Pro: sparse
  integer / UMI count matrices, typically 1e4-1e6 EVs x 8-250 protein targets.
* **flow / imaging intensity** — NanoFCM nano-flow cytometry, ExoView /
  SP-IRIS, FCS files: continuous fluorescence intensity per marker, 2-6 markers.
* **digital binary** — droplet-digital ELISA / Simoa: per-EV presence/absence
  calls.

Shared AnnData schema (every ``ov.single.ev`` function coordinates to this)
--------------------------------------------------------------------------
``adata.X``
    EV x protein matrix — ``obs`` = individual EVs, ``var`` = protein targets.
``adata.layers['counts']``
    The raw loaded values (kept untouched for re-processing).
``adata.obs``
    Per-EV metadata: ``sample``, ``platform``, and where available
    ``ev_size`` (nm) and ``capture_id`` (spot / droplet / well / position);
    plus computed ``total_signal`` (per-EV sum) and ``n_proteins`` (per-EV
    non-zero count).
``adata.var``
    Index = protein / marker name; ``misev_category`` column (filled later by
    the annotation step).
``adata.uns['ev']``
    Dict ``{'value_type': 'count'|'intensity'|'binary', 'platform': str}``.

This module is pure Python on numpy / scipy / pandas / anndata; the optional
``flowkit`` / ``fcsparser`` backends import lazily so plain ``import omicverse``
works without them.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..._registry import register_function


# ---------------------------------------------------------------------------
# Optional-dependency helper
# ---------------------------------------------------------------------------
def _require(modname: str, feature: str):
    """Lazy-import an optional backend with an actionable error message."""
    import importlib

    try:
        return importlib.import_module(modname)
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise ImportError(
            f"{feature} needs the optional '{modname}' backend. Install with: "
            f"pip install {modname}   (or pip install omicverse[ev])."
        ) from exc


_VALUE_TYPES = ("count", "intensity", "binary")


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------
def _read_table(path: str) -> pd.DataFrame:
    """Read a delimited / parquet table, auto-detecting the format."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    low = path.lower()
    if low.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if low.endswith((".tsv", ".txt")):
        return pd.read_csv(path, sep="\t")
    if low.endswith((".csv", ".csv.gz", ".gz")):
        return pd.read_csv(path)
    # fall back to sniffing the separator
    return pd.read_csv(path, sep=None, engine="python")


def _coerce_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce every column to float, replacing un-parseable cells with 0."""
    return frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(
        np.float32
    )


def _validate_value_type(value_type: str) -> str:
    vt = str(value_type).lower()
    if vt not in _VALUE_TYPES:
        raise ValueError(
            f"value_type must be one of {_VALUE_TYPES!r}, got {value_type!r}."
        )
    return vt


def _build_ev_anndata(
    matrix: pd.DataFrame,
    *,
    value_type: str,
    platform: str,
    obs: Optional[pd.DataFrame] = None,
    var: Optional[pd.DataFrame] = None,
):
    """Assemble a single-EV AnnData in the shared schema.

    Parameters
    ----------
    matrix
        EV (rows) x protein (columns) numeric :class:`pandas.DataFrame`.
    value_type
        ``'count'``, ``'intensity'`` or ``'binary'``.
    platform
        Free-text platform tag (e.g. ``'PBA'``, ``'ExoView'``).
    obs / var
        Optional pre-built metadata frames; missing columns are filled in.
    """
    from anndata import AnnData

    value_type = _validate_value_type(value_type)
    X = _coerce_numeric(matrix)
    var_names = [str(c) for c in matrix.columns]
    obs_names = [str(i) for i in matrix.index]

    if obs is None:
        obs = pd.DataFrame(index=obs_names)
    else:
        obs = obs.copy()
        obs.index = obs_names
    if var is None:
        var = pd.DataFrame(index=var_names)
    else:
        var = var.copy()
        var.index = var_names

    if "sample" not in obs.columns:
        obs["sample"] = "sample1"
    if "platform" not in obs.columns:
        obs["platform"] = platform
    if "misev_category" not in var.columns:
        var["misev_category"] = "unknown"

    adata = AnnData(X=X.values, obs=obs, var=var)
    adata.obs_names = obs_names
    adata.var_names = var_names
    adata.layers["counts"] = adata.X.copy()
    adata.uns["ev"] = {"value_type": value_type, "platform": str(platform)}
    refresh_ev_metrics(adata)
    return adata


def _matrix_from_long(
    df: pd.DataFrame,
    *,
    ev_col: str,
    protein_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Pivot a long (EV, protein, value) table into a wide EV x protein matrix."""
    for col in (ev_col, protein_col, value_col):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not found in the input table.")
    wide = df.pivot_table(
        index=ev_col, columns=protein_col, values=value_col,
        aggfunc="sum", fill_value=0,
    )
    wide.index = wide.index.astype(str)
    wide.columns = wide.columns.astype(str)
    return wide


# ---------------------------------------------------------------------------
# Shared metrics helper
# ---------------------------------------------------------------------------
@register_function(
    aliases=[
        "refresh_ev_metrics", "ev_metrics", "compute_ev_metrics",
        "刷新EV指标", "EV信号统计",
    ],
    category="ev",
    description=(
        "Compute / refresh the per-EV summary metrics on a single-EV AnnData: "
        "obs['total_signal'] (per-EV sum of X) and obs['n_proteins'] (per-EV "
        "count of detected, non-zero proteins). Run after loading or filtering."
    ),
    produces={"obs": ["total_signal", "n_proteins"]},
    examples=[
        "ov.single.ev.refresh_ev_metrics(adata)",
        "adata.obs[['total_signal', 'n_proteins']].describe()",
    ],
    related=["single.ev.read_ev_matrix", "single.ev.qc"],
)
def refresh_ev_metrics(adata):
    """Compute / refresh ``total_signal`` and ``n_proteins`` in ``obs``.

    Parameters
    ----------
    adata
        A single-EV :class:`~anndata.AnnData` (EV x protein).

    Returns
    -------
    :class:`~anndata.AnnData`
        The same object with ``obs['total_signal']`` (row sum of ``X``) and
        ``obs['n_proteins']`` (number of non-zero proteins per EV) updated
        in place.
    """
    import scipy.sparse as sp

    X = adata.X
    if sp.issparse(X):
        total = np.asarray(X.sum(axis=1)).ravel()
        n_det = np.asarray((X > 0).sum(axis=1)).ravel()
    else:
        X = np.asarray(X)
        total = X.sum(axis=1)
        n_det = (X > 0).sum(axis=1)
    adata.obs["total_signal"] = np.asarray(total, dtype=np.float64)
    adata.obs["n_proteins"] = np.asarray(n_det, dtype=np.int64)
    return adata


# ---------------------------------------------------------------------------
# Generic reader
# ---------------------------------------------------------------------------
@register_function(
    aliases=[
        "read_ev_matrix", "load_ev_matrix", "read_ev", "读取EV矩阵",
        "单囊泡矩阵读取",
    ],
    category="ev",
    description=(
        "Generic single-EV reader: an EV x protein table (CSV / TSV / parquet "
        "path, or a pandas DataFrame) where rows are individual extracellular "
        "vesicles and columns are protein/marker targets, into an AnnData in "
        "the ov.single.ev schema (X = EV x protein, layers['counts'] = raw, "
        "uns['ev'] = value-type metadata)."
    ),
    examples=[
        "adata = ov.single.ev.read_ev_matrix('ev_protein.csv', value_type='count')",
        "adata = ov.single.ev.read_ev_matrix(df, value_type='intensity')",
    ],
    related=["single.ev.read_pba", "single.ev.simulate_ev", "single.ev.qc"],
)
def read_ev_matrix(
    data: Union[str, pd.DataFrame],
    *,
    value_type: str = "count",
    platform: str = "generic",
    index_col: Union[int, str, None] = 0,
    transpose: bool = False,
    obs: Optional[pd.DataFrame] = None,
):
    """Read a generic EV x protein table into a single-EV AnnData.

    Parameters
    ----------
    data
        Path to a ``.csv`` / ``.tsv`` / ``.parquet`` file, or an in-memory
        :class:`pandas.DataFrame`. Rows are EVs, columns are proteins.
    value_type
        Measurement value type: ``'count'`` (sequencing UMIs), ``'intensity'``
        (flow / imaging fluorescence) or ``'binary'`` (digital presence calls).
    platform
        Free-text platform tag stored in ``uns['ev']`` and ``obs['platform']``.
    index_col
        Column to use as the EV index when reading from a file. Set to
        ``None`` if the file has no EV-id column.
    transpose
        If ``True`` the input is protein x EV and is transposed to EV x
        protein on load.
    obs
        Optional per-EV metadata frame; its rows are aligned to the matrix.

    Returns
    -------
    :class:`~anndata.AnnData`
        Single-EV AnnData in the shared ``ov.single.ev`` schema.
    """
    value_type = _validate_value_type(value_type)
    if isinstance(data, pd.DataFrame):
        matrix = data.copy()
    else:
        low = str(data).lower()
        if low.endswith((".parquet", ".pq")):
            matrix = pd.read_parquet(data)
            if index_col is not None and index_col in matrix.columns:
                matrix = matrix.set_index(index_col)
        else:
            if not os.path.exists(data):
                raise FileNotFoundError(data)
            sep = "\t" if low.endswith((".tsv", ".txt")) else ","
            matrix = pd.read_csv(data, sep=sep, index_col=index_col)
    if transpose:
        matrix = matrix.T
    return _build_ev_anndata(
        matrix, value_type=value_type, platform=platform, obs=obs
    )


# ---------------------------------------------------------------------------
# PBA / DBS-Pro sequencing counts
# ---------------------------------------------------------------------------
@register_function(
    aliases=[
        "read_pba", "read_dbs_pro", "load_pba", "读取PBA", "邻近条形码读取",
    ],
    category="ev",
    description=(
        "Read a PBA (Proximity Barcoding Assay) or DBS-Pro sequencing-based "
        "single-EV surface-protein count matrix (EV-tag x protein-tag) into a "
        "count-type single-EV AnnData. Accepts either a wide EV x protein "
        "matrix or a long (ev, protein, count) table."
    ),
    examples=[
        "adata = ov.single.ev.read_pba('pba_counts.csv')",
        "adata = ov.single.ev.read_pba(long_df, ev_col='EV', protein_col='target', value_col='umi')",
    ],
    related=["single.ev.read_ev_matrix", "single.ev.qc"],
)
def read_pba(
    data: Union[str, pd.DataFrame],
    *,
    platform: str = "PBA",
    index_col: Union[int, str, None] = 0,
    long_format: bool = False,
    ev_col: str = "ev_id",
    protein_col: str = "protein",
    value_col: str = "count",
    obs: Optional[pd.DataFrame] = None,
):
    """Read a PBA / DBS-Pro single-EV surface-protein count matrix.

    Parameters
    ----------
    data
        Path or :class:`pandas.DataFrame`. In wide form rows are EV-tags and
        columns are protein-tags. In long form (``long_format=True``) the
        table has one row per (EV, protein) observation.
    platform
        Platform tag (``'PBA'`` or ``'DBS-Pro'``).
    index_col
        EV-id column when reading a wide file.
    long_format
        If ``True`` the input is a long (ev, protein, count) table that is
        pivoted to a wide EV x protein matrix.
    ev_col, protein_col, value_col
        Column names in the long table (used only when ``long_format=True``).
    obs
        Optional per-EV metadata frame.

    Returns
    -------
    :class:`~anndata.AnnData`
        Count-type single-EV AnnData (``uns['ev']['value_type'] == 'count'``).
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = _read_table(data)
        if not long_format and index_col is not None:
            if isinstance(index_col, int):
                df = df.set_index(df.columns[index_col])
            elif index_col in df.columns:
                df = df.set_index(index_col)
    if long_format:
        matrix = _matrix_from_long(
            df, ev_col=ev_col, protein_col=protein_col, value_col=value_col
        )
    else:
        matrix = df
    return _build_ev_anndata(
        matrix, value_type="count", platform=platform, obs=obs
    )


# ---------------------------------------------------------------------------
# ExoView / SP-IRIS imaging intensity
# ---------------------------------------------------------------------------
@register_function(
    aliases=[
        "read_exoview", "read_spiris", "load_exoview", "读取ExoView",
        "单颗粒成像读取",
    ],
    category="ev",
    description=(
        "Read an ExoView / SP-IRIS exported per-particle table into an "
        "intensity-type single-EV AnnData. Each row is a captured particle; "
        "fluorescence-channel columns become protein markers, and the capture "
        "spot / position and particle size are stored in obs."
    ),
    examples=[
        "adata = ov.single.ev.read_exoview('exoview_particles.csv')",
        "adata = ov.single.ev.read_exoview(df, marker_cols=['CD9','CD63','CD81'])",
    ],
    related=["single.ev.read_nanofcm", "single.ev.read_ev_matrix"],
)
def read_exoview(
    data: Union[str, pd.DataFrame],
    *,
    marker_cols: Optional[Sequence[str]] = None,
    size_col: Optional[str] = None,
    capture_col: Optional[str] = None,
    sample_col: Optional[str] = None,
    platform: str = "ExoView",
):
    """Read an ExoView / SP-IRIS exported per-particle table.

    Parameters
    ----------
    data
        Path or :class:`pandas.DataFrame` of the per-particle export. Each
        row is one captured EV.
    marker_cols
        Columns holding fluorescence intensities (the protein markers, e.g.
        ``['CD9', 'CD63', 'CD81']``). If ``None`` every numeric column that is
        not a metadata column is treated as a marker.
    size_col
        Column with the particle diameter in nm -> stored as ``obs['ev_size']``.
    capture_col
        Column with the capture-spot / antibody-spot id -> ``obs['capture_id']``.
    sample_col
        Column with the sample id -> ``obs['sample']``.
    platform
        Platform tag (default ``'ExoView'``).

    Returns
    -------
    :class:`~anndata.AnnData`
        Intensity-type single-EV AnnData.
    """
    df = data.copy() if isinstance(data, pd.DataFrame) else _read_table(data)
    df = df.reset_index(drop=True)

    meta_cols = {c for c in (size_col, capture_col, sample_col) if c}
    if marker_cols is None:
        numeric = df.select_dtypes(include=[np.number]).columns
        marker_cols = [c for c in numeric if c not in meta_cols]
    marker_cols = list(marker_cols)
    if not marker_cols:
        raise ValueError(
            "No marker columns found — pass marker_cols explicitly."
        )

    matrix = df[marker_cols].copy()
    matrix.index = [f"particle{i:06d}" for i in range(len(df))]

    obs = pd.DataFrame(index=matrix.index)
    if size_col and size_col in df.columns:
        obs["ev_size"] = pd.to_numeric(df[size_col], errors="coerce").values
    if capture_col and capture_col in df.columns:
        obs["capture_id"] = df[capture_col].astype(str).values
    if sample_col and sample_col in df.columns:
        obs["sample"] = df[sample_col].astype(str).values

    return _build_ev_anndata(
        matrix, value_type="intensity", platform=platform, obs=obs
    )


# ---------------------------------------------------------------------------
# NanoFCM nano-flow cytometry
# ---------------------------------------------------------------------------
@register_function(
    aliases=[
        "read_nanofcm", "read_nano_flow", "load_nanofcm", "读取NanoFCM",
        "纳米流式读取",
    ],
    category="ev",
    description=(
        "Read a NanoFCM nano-flow-cytometry single-EV event table into an "
        "intensity-type single-EV AnnData. Each row is a detected EV event; "
        "fluorescence-channel columns become markers and side-scatter-derived "
        "particle size, where present, is stored in obs['ev_size']. CSV-export "
        "tables are read directly; FCS files are delegated to read_fcs."
    ),
    examples=[
        "adata = ov.single.ev.read_nanofcm('nanofcm_events.csv')",
        "adata = ov.single.ev.read_nanofcm('events.csv', marker_cols=['FITC','PE'])",
    ],
    related=["single.ev.read_fcs", "single.ev.read_exoview"],
)
def read_nanofcm(
    data: Union[str, pd.DataFrame],
    *,
    marker_cols: Optional[Sequence[str]] = None,
    size_col: Optional[str] = None,
    sample_col: Optional[str] = None,
    platform: str = "NanoFCM",
):
    """Read a NanoFCM nano-flow-cytometry single-EV event table.

    Parameters
    ----------
    data
        Path to a NanoFCM CSV / TSV export, a ``.fcs`` file (delegated to
        :func:`read_fcs`), or an in-memory :class:`pandas.DataFrame`.
    marker_cols
        Fluorescence-channel columns to keep as protein markers. If ``None``
        every numeric column that is not a metadata column is used.
    size_col
        Column with the side-scatter-derived particle size in nm ->
        ``obs['ev_size']``.
    sample_col
        Column with the sample id -> ``obs['sample']``.
    platform
        Platform tag (default ``'NanoFCM'``).

    Returns
    -------
    :class:`~anndata.AnnData`
        Intensity-type single-EV AnnData.
    """
    if isinstance(data, str) and data.lower().endswith(".fcs"):
        return read_fcs(data, marker_cols=marker_cols, platform=platform)

    df = data.copy() if isinstance(data, pd.DataFrame) else _read_table(data)
    df = df.reset_index(drop=True)

    meta_cols = {c for c in (size_col, sample_col) if c}
    if marker_cols is None:
        numeric = df.select_dtypes(include=[np.number]).columns
        marker_cols = [c for c in numeric if c not in meta_cols]
    marker_cols = list(marker_cols)
    if not marker_cols:
        raise ValueError(
            "No marker columns found — pass marker_cols explicitly."
        )

    matrix = df[marker_cols].copy()
    matrix.index = [f"event{i:06d}" for i in range(len(df))]

    obs = pd.DataFrame(index=matrix.index)
    if size_col and size_col in df.columns:
        obs["ev_size"] = pd.to_numeric(df[size_col], errors="coerce").values
    if sample_col and sample_col in df.columns:
        obs["sample"] = df[sample_col].astype(str).values

    return _build_ev_anndata(
        matrix, value_type="intensity", platform=platform, obs=obs
    )


@register_function(
    aliases=[
        "read_fcs", "load_fcs", "read_flow", "读取FCS", "流式文件读取",
    ],
    category="ev",
    description=(
        "Read a single-EV flow-cytometry FCS file into an intensity-type "
        "single-EV AnnData. Each FCS event is one EV and each fluorescence "
        "channel a protein marker. Uses the optional 'flowkit' or 'fcsparser' "
        "backend for FCS parsing; if neither is installed pass a CSV export "
        "instead (see read_nanofcm)."
    ),
    examples=[
        "adata = ov.single.ev.read_fcs('ev_events.fcs')",
        "adata = ov.single.ev.read_fcs('ev_events.fcs', marker_cols=['FITC-A'])",
    ],
    related=["single.ev.read_nanofcm", "single.ev.read_ev_matrix"],
)
def read_fcs(
    path: str,
    *,
    marker_cols: Optional[Sequence[str]] = None,
    platform: str = "FCS",
    sample: str = "sample1",
):
    """Read an FCS-format single-EV flow event file into an AnnData.

    Parameters
    ----------
    path
        Path to a ``.fcs`` file. If it ends in ``.csv`` / ``.tsv`` the call is
        forwarded to :func:`read_nanofcm` (the documented CSV-export path).
    marker_cols
        Channels to keep as protein markers. If ``None`` all channels are
        kept.
    platform
        Platform tag stored in ``uns['ev']``.
    sample
        Sample id stored in ``obs['sample']``.

    Returns
    -------
    :class:`~anndata.AnnData`
        Intensity-type single-EV AnnData.

    Notes
    -----
    FCS parsing requires the optional ``flowkit`` or ``fcsparser`` package
    (``pip install fcsparser``). When neither is available, export the events
    to CSV from the instrument software and use :func:`read_nanofcm`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    low = path.lower()
    if low.endswith((".csv", ".tsv", ".txt")):
        return read_nanofcm(path, marker_cols=marker_cols, platform=platform)

    events: Optional[pd.DataFrame] = None
    # try flowkit first, then fcsparser
    try:
        flowkit = _require("flowkit", "FCS parsing")
        fk_sample = flowkit.Sample(path)
        events = fk_sample.as_dataframe(source="raw")
        if hasattr(events.columns, "get_level_values"):
            events.columns = events.columns.get_level_values(-1)
    except ImportError:
        fcsparser = _require("fcsparser", "FCS parsing")
        _, events = fcsparser.parse(path, reformat_meta=True)

    events = events.reset_index(drop=True)
    if marker_cols is not None:
        events = events[list(marker_cols)]
    events.index = [f"event{i:06d}" for i in range(len(events))]
    obs = pd.DataFrame(index=events.index)
    obs["sample"] = sample
    return _build_ev_anndata(
        events, value_type="intensity", platform=platform, obs=obs
    )


# ---------------------------------------------------------------------------
# Digital / Simoa binary calls
# ---------------------------------------------------------------------------
@register_function(
    aliases=[
        "read_digital_ev", "read_simoa", "read_ddpcr_ev", "load_digital_ev",
        "读取数字EV", "单分子阵列读取",
    ],
    category="ev",
    description=(
        "Read a droplet-digital ELISA / Simoa per-EV binary call matrix "
        "(EV x protein, presence/absence) into a binary-type single-EV "
        "AnnData. Non-zero / truthy entries are coerced to 1, everything else "
        "to 0; the droplet/well id is stored in obs['capture_id']."
    ),
    examples=[
        "adata = ov.single.ev.read_digital_ev('simoa_calls.csv')",
        "adata = ov.single.ev.read_digital_ev(calls_df, threshold=0.5)",
    ],
    related=["single.ev.read_ev_matrix", "single.ev.subtract_isotype"],
)
def read_digital_ev(
    data: Union[str, pd.DataFrame],
    *,
    platform: str = "Simoa",
    index_col: Union[int, str, None] = 0,
    threshold: float = 0.0,
    capture_col: Optional[str] = None,
    sample_col: Optional[str] = None,
):
    """Read a droplet-digital / Simoa per-EV binary call matrix.

    Parameters
    ----------
    data
        Path or :class:`pandas.DataFrame`. Rows are EVs (droplets / wells),
        columns are proteins; values are presence calls or raw signals.
    platform
        Platform tag (``'Simoa'``, ``'ddELISA'`` ...).
    index_col
        EV-id column when reading from a file.
    threshold
        Values strictly greater than ``threshold`` are called present (1),
        all others absent (0).
    capture_col
        Column with the droplet / well id -> ``obs['capture_id']``.
    sample_col
        Column with the sample id -> ``obs['sample']``.

    Returns
    -------
    :class:`~anndata.AnnData`
        Binary-type single-EV AnnData (``uns['ev']['value_type'] == 'binary'``).
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = _read_table(data)
        if index_col is not None:
            if isinstance(index_col, int):
                df = df.set_index(df.columns[index_col])
            elif index_col in df.columns:
                df = df.set_index(index_col)

    obs_cols = {c for c in (capture_col, sample_col) if c}
    obs = pd.DataFrame(index=[str(i) for i in df.index])
    if capture_col and capture_col in df.columns:
        obs["capture_id"] = df[capture_col].astype(str).values
    if sample_col and sample_col in df.columns:
        obs["sample"] = df[sample_col].astype(str).values
    matrix = df.drop(columns=[c for c in obs_cols if c in df.columns])

    numeric = _coerce_numeric(matrix)
    binary = (numeric > threshold).astype(np.float32)
    binary.index = matrix.index
    binary.columns = matrix.columns
    return _build_ev_anndata(
        binary, value_type="binary", platform=platform, obs=obs
    )


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
@register_function(
    aliases=[
        "simulate_ev", "ev_simulate", "mock_ev", "模拟EV数据",
        "单囊泡模拟",
    ],
    category="ev",
    description=(
        "Simulate a single-EV proteomics AnnData with a configurable number "
        "of EVs, protein panel size, latent EV subpopulations and value type "
        "('count', 'intensity' or 'binary') — for tutorials and tests when "
        "real data is absent. Returns data in the ov.single.ev schema with a "
        "ground-truth obs['subpopulation'] label."
    ),
    examples=[
        "adata = ov.single.ev.simulate_ev(n_ev=2000, n_proteins=30)",
        "adata = ov.single.ev.simulate_ev(n_ev=5000, value_type='intensity', n_subpop=4)",
    ],
    related=["single.ev.read_ev_matrix", "single.ev.qc"],
)
def simulate_ev(
    n_ev: int = 1000,
    n_proteins: int = 30,
    *,
    value_type: str = "count",
    n_subpop: int = 3,
    n_samples: int = 2,
    sparsity: float = 0.6,
    contaminant_frac: float = 0.05,
    seed: int = 0,
):
    """Simulate a single-EV proteomics AnnData.

    Parameters
    ----------
    n_ev
        Number of extracellular vesicles (rows) to generate.
    n_proteins
        Size of the protein / marker panel (columns).
    value_type
        ``'count'`` (negative-binomial UMIs), ``'intensity'`` (log-normal
        fluorescence) or ``'binary'`` (Bernoulli presence calls).
    n_subpop
        Number of latent EV subpopulations, each with its own marker profile.
    n_samples
        Number of biological samples; assigned at random to ``obs['sample']``.
    sparsity
        Target fraction of zero entries (drop-out), applied for ``'count'``
        and ``'binary'`` value types.
    contaminant_frac
        Fraction of EVs spiked as co-isolated contaminant particles (high in
        a few contaminant-marker columns) — useful for testing
        :func:`omicverse.single.ev.contaminant_score`.
    seed
        Random seed.

    Returns
    -------
    :class:`~anndata.AnnData`
        Single-EV AnnData in the shared schema. ``obs`` carries the
        ground-truth ``subpopulation`` and ``is_contaminant`` labels, an
        ``ev_size`` (nm) and a ``capture_id``.
    """
    value_type = _validate_value_type(value_type)
    rng = np.random.default_rng(seed)
    n_ev = int(n_ev)
    n_proteins = int(n_proteins)
    n_subpop = max(1, int(n_subpop))

    # protein panel — include canonical EV / contaminant markers up front
    canonical = [
        "CD9", "CD63", "CD81", "TSG101", "ALIX", "FLOT1", "SDCBP",
        "APOA1", "APOB", "ALB", "CANX", "GOLGA2", "CYCS", "TOMM20",
    ]
    panel = canonical[:n_proteins]
    panel += [f"PROT{i:03d}" for i in range(len(panel), n_proteins)]
    panel = panel[:n_proteins]
    contaminant_markers = [
        m for m in ("APOA1", "APOB", "ALB", "CANX", "GOLGA2", "CYCS",
                    "TOMM20")
        if m in panel
    ]

    # per-subpopulation latent marker means
    subpop = rng.integers(0, n_subpop, size=n_ev)
    base = rng.gamma(shape=1.5, scale=1.0, size=(n_subpop, n_proteins))
    # each subpopulation strongly expresses a random marker block
    for k in range(n_subpop):
        hot = rng.choice(n_proteins, size=max(2, n_proteins // 4),
                         replace=False)
        base[k, hot] *= rng.uniform(4.0, 9.0, size=hot.size)
    mu = base[subpop]  # n_ev x n_proteins

    # contaminant EVs
    is_contam = rng.random(n_ev) < float(contaminant_frac)
    if contaminant_markers:
        cidx = [panel.index(m) for m in contaminant_markers]
        boost = np.zeros((n_ev, n_proteins))
        boost[np.ix_(is_contam, cidx)] = rng.uniform(
            8.0, 20.0, size=(int(is_contam.sum()), len(cidx))
        )
        mu = mu + boost

    if value_type == "count":
        # negative-binomial counts
        disp = 2.0
        p = disp / (disp + mu)
        X = rng.negative_binomial(disp, p).astype(np.float32)
        mask = rng.random(X.shape) < float(sparsity)
        X[mask] = 0.0
    elif value_type == "intensity":
        # log-normal fluorescence intensity
        X = rng.lognormal(mean=np.log1p(mu), sigma=0.4).astype(np.float32)
    else:  # binary
        prob = 1.0 - np.exp(-mu / (mu.mean() + 1e-9))
        prob = prob * (1.0 - float(sparsity))
        X = (rng.random(prob.shape) < prob).astype(np.float32)

    matrix = pd.DataFrame(
        X,
        index=[f"EV{i:06d}" for i in range(n_ev)],
        columns=panel,
    )

    sizes = np.clip(rng.normal(110, 35, size=n_ev), 30, 400)
    obs = pd.DataFrame(index=matrix.index)
    obs["sample"] = [f"sample{int(s) + 1}" for s in
                     rng.integers(0, max(1, int(n_samples)), size=n_ev)]
    obs["subpopulation"] = [f"subpop{int(s) + 1}" for s in subpop]
    obs["is_contaminant"] = is_contam
    obs["ev_size"] = sizes
    obs["capture_id"] = [f"spot{int(c):02d}" for c in
                         rng.integers(0, 16, size=n_ev)]

    return _build_ev_anndata(
        matrix, value_type=value_type, platform="simulated", obs=obs
    )

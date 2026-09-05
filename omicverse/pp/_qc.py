"""
Copy from pegasus and cellual

"""


import os
import logging
from functools import reduce
from random import seed, sample
from typing import Union, Optional, Sequence, Tuple, List, Dict
import pandas as pd 
import anndata
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse 
from matplotlib.backends.backend_pdf import PdfPages 
from matplotlib import rcParams
import seaborn as sns
from scipy.sparse import issparse

from .._settings import settings,print_gpu_usage_color,EMOJI,add_reference,Colors
from .._registry import register_function
from .._monitor import monitor
from ..report._provenance import tracked, note


# Local implementations to replace scanpy filtering functions
def _oom_axis_stats(X, axis):
    """Per-row (axis=1) or per-column (axis=0) sum and nnz of a backed
    matrix in ONE chunked pass — no densification, no materialisation.
    Used by the out-of-core filter paths so AnnDataOOM never loads X."""
    from scipy.sparse import issparse
    n_obs, n_vars = X.shape
    n = n_obs if axis == 1 else n_vars
    s = np.zeros(n, dtype=np.float64)
    nnz = np.zeros(n, dtype=np.int64)
    for start, end, chunk in X.chunked():
        if issparse(chunk):
            if axis == 1:
                s[start:end] = np.asarray(chunk.sum(axis=1)).ravel()
                nnz[start:end] = chunk.getnnz(axis=1)
            else:
                s += np.asarray(chunk.sum(axis=0)).ravel()
                nnz += chunk.getnnz(axis=0)
        else:
            chunk = np.asarray(chunk)
            if axis == 1:
                s[start:end] = chunk.sum(axis=1); nnz[start:end] = (chunk > 0).sum(axis=1)
            else:
                s += chunk.sum(axis=0); nnz += (chunk > 0).sum(axis=0)
    return s, nnz


def _filter_cells_impl(
    adata,
    min_counts=None,
    min_genes=None,
    max_counts=None,
    max_genes=None,
    inplace=True
):
    """Internal implementation of cell filtering."""
    X = adata.X
    if _is_oom(adata):
        # out-of-core: one chunked pass over X, never densify/materialise.
        n_counts, n_genes = _oom_axis_stats(X, axis=1)
    elif sparse.issparse(X):
        n_counts = np.asarray(X.sum(axis=1)).flatten()
        n_genes = np.asarray((X > 0).sum(axis=1)).flatten()
    else:
        n_counts = X.sum(axis=1)
        n_genes = (X > 0).sum(axis=1)

    cell_subset = np.ones(adata.n_obs, dtype=bool)

    if min_counts is not None:
        cell_subset &= n_counts >= min_counts
    if max_counts is not None:
        cell_subset &= n_counts <= max_counts
    if min_genes is not None:
        cell_subset &= n_genes >= min_genes
    if max_genes is not None:
        cell_subset &= n_genes <= max_genes

    if inplace:
        adata._inplace_subset_obs(cell_subset)
        return None
    else:
        return cell_subset, n_counts


def _filter_genes_impl(
    adata,
    min_counts=None,
    min_cells=None,
    max_counts=None,
    max_cells=None,
    inplace=True
):
    """Internal implementation of gene filtering."""
    X = adata.X
    if _is_oom(adata):
        n_counts, n_cells = _oom_axis_stats(X, axis=0)
    elif sparse.issparse(X):
        n_counts = np.asarray(X.sum(axis=0)).flatten()
        n_cells = np.asarray((X > 0).sum(axis=0)).flatten()
    else:
        n_counts = X.sum(axis=0)
        n_cells = (X > 0).sum(axis=0)

    gene_subset = np.ones(adata.n_vars, dtype=bool)

    if min_counts is not None:
        gene_subset &= n_counts >= min_counts
    if max_counts is not None:
        gene_subset &= n_counts <= max_counts
    if min_cells is not None:
        gene_subset &= n_cells >= min_cells
    if max_cells is not None:
        gene_subset &= n_cells <= max_cells

    if inplace:
        adata._inplace_subset_var(gene_subset)
        return None
    else:
        return gene_subset, n_counts


# Single source of truth for "is this adata an AnnDataOOM?" — delegates to
# the compat shim so it stays in lockstep with the anndataoom package (or its
# no-op fallback when anndataoom is not installed).
from .._oom_compat import is_oom as _is_oom


# Deprecated alias — AnnDataOOM is the only rust-backed AnnData we support in
# omicverse's preprocessing path. The standalone snapatac2 / anndata-rs paths
# were never publicly exposed. Kept as an alias so call sites that still read
# ``is_rust = _is_rust_backend(adata)`` keep working, but new code should use
# ``_is_oom`` directly.
def _is_rust_backend(adata) -> bool:
    return _is_oom(adata)


def _print_qc_metrics_table(adata):
    """Print QC metrics in a colored table format."""
    # Box drawing characters
    top_line = "┌" + "─" * 25 + "┬" + "─" * 20 + "┬" + "─" * 25 + "┐"
    mid_line = "├" + "─" * 25 + "┼" + "─" * 20 + "┼" + "─" * 25 + "┤"
    bot_line = "└" + "─" * 25 + "┴" + "─" * 20 + "┴" + "─" * 25 + "┘"

    print(f"\n{Colors.GREEN}   ✓ QC Metrics Summary:{Colors.ENDC}")
    print(f"   {Colors.CYAN}{top_line}{Colors.ENDC}")

    # Header
    header = f"   {Colors.CYAN}│{Colors.ENDC} {Colors.BOLD}{'Metric':<23}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC} {Colors.BOLD}{'Mean':<18}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC} {Colors.BOLD}{'Range (Min - Max)':<23}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC}"
    print(header)
    print(f"   {Colors.CYAN}{mid_line}{Colors.ENDC}")

    # Data rows
    metrics = [
        ("nUMIs", "nUMIs", adata.obs['nUMIs']),
        ("Detected Genes", "detected_genes", adata.obs['detected_genes']),
        ("Mitochondrial %", "mito_perc", adata.obs['mito_perc'] * 100),
        ("Ribosomal %", "ribo_perc", adata.obs['ribo_perc'] * 100),
        ("Hemoglobin %", "hb_perc", adata.obs['hb_perc'] * 100),
    ]

    for i, (display_name, col_name, values) in enumerate(metrics):
        mean_val = values.mean()
        min_val = values.min()
        max_val = values.max()

        # Format values based on metric type
        if "%" in display_name:
            mean_str = f"{mean_val:.1f}%"
            range_str = f"{min_val:.1f}% - {max_val:.1f}%"
        else:
            mean_str = f"{mean_val:.0f}"
            range_str = f"{min_val:.0f} - {max_val:.0f}"

        row = f"   {Colors.CYAN}│{Colors.ENDC} {Colors.BLUE}{display_name:<23}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC} {Colors.BOLD}{mean_str:<18}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC} {range_str:<23} {Colors.CYAN}│{Colors.ENDC}"
        print(row)

        if i < len(metrics) - 1:
            print(f"   {Colors.CYAN}{mid_line}{Colors.ENDC}")

    print(f"   {Colors.CYAN}{bot_line}{Colors.ENDC}")


def _print_gene_detection_table(mt_count, ribo_count, hb_count, mt_genes=None, ribo_genes=None, hb_genes=None):
    """Print gene family detection results in a colored table format."""
    top_line = "┌" + "─" * 30 + "┬" + "─" * 20 + "┬" + "─" * 20 + "┐"
    mid_line = "├" + "─" * 30 + "┼" + "─" * 20 + "┼" + "─" * 20 + "┤"
    bot_line = "└" + "─" * 30 + "┴" + "─" * 20 + "┴" + "─" * 20 + "┘"

    print(f"\n{Colors.GREEN}   ✓ Gene Family Detection:{Colors.ENDC}")
    print(f"   {Colors.CYAN}{top_line}{Colors.ENDC}")

    # Header
    header = f"   {Colors.CYAN}│{Colors.ENDC} {Colors.BOLD}{'Gene Family':<28}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC} {Colors.BOLD}{'Genes Found':<18}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC} {Colors.BOLD}{'Detection Method':<18}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC}"
    print(header)
    print(f"   {Colors.CYAN}{mid_line}{Colors.ENDC}")

    # Data rows
    families = [
        ("Mitochondrial", mt_count, "Custom list" if mt_genes is not None else "Auto (MT-)"),
        ("Ribosomal", ribo_count, "Custom list" if ribo_genes is not None else "Auto (RPS/RPL)"),
        ("Hemoglobin", hb_count, "Custom list" if hb_genes is not None else "Auto (regex)"),
    ]

    for i, (name, count, method) in enumerate(families):
        # Color code based on count
        if count == 0:
            count_color = Colors.FAIL
            count_str = f"{count:,} ⚠️"
        elif count < 5:
            count_color = Colors.WARNING
            count_str = f"{count:,}"
        else:
            count_color = Colors.GREEN
            count_str = f"{count:,}"

        row = f"   {Colors.CYAN}│{Colors.ENDC} {Colors.BLUE}{name:<28}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC} {count_color}{count_str:<18}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC} {method:<18} {Colors.CYAN}│{Colors.ENDC}"
        print(row)

        if i < len(families) - 1:
            print(f"   {Colors.CYAN}{mid_line}{Colors.ENDC}")

    print(f"   {Colors.CYAN}{bot_line}{Colors.ENDC}")


def _print_filtering_results_table(mode, tresh, n_total, mt_failed, umis_failed, genes_failed, nmads=None):
    """Print filtering results in a colored table format."""
    top_line = "┌" + "─" * 30 + "┬" + "─" * 25 + "┬" + "─" * 15 + "┐"
    mid_line = "├" + "─" * 30 + "┼" + "─" * 25 + "┼" + "─" * 15 + "┤"
    bot_line = "└" + "─" * 30 + "┴" + "─" * 25 + "┴" + "─" * 15 + "┘"

    print(f"\n{Colors.GREEN}   ✓ Filtering Results:{Colors.ENDC}")
    print(f"   {Colors.CYAN}{top_line}{Colors.ENDC}")

    # Header
    header = f"   {Colors.CYAN}│{Colors.ENDC} {Colors.BOLD}{'Filter Type':<28}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC} {Colors.BOLD}{'Threshold':<23}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC} {Colors.BOLD}{'Failed':<13}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC}"
    print(header)
    print(f"   {Colors.CYAN}{mid_line}{Colors.ENDC}")

    # Prepare filter data
    if mode == 'seurat':
        filters = [
            ("Mitochondrial %", f"≤ {tresh['mito_perc']*100:.0f}%", mt_failed),
            ("nUMIs", f"≥ {tresh['nUMIs']:,}", umis_failed),
            ("Detected Genes", f"≥ {tresh['detected_genes']:,}", genes_failed),
        ]
    else:  # mads
        filters = [
            ("Mitochondrial %", f"≤ {tresh['mito_perc']*100:.0f}%", mt_failed),
            ("nUMIs (MADs)", f"±{nmads} MADs", umis_failed),
            ("Detected Genes (MADs)", f"±{nmads} MADs", genes_failed),
        ]

    for i, (filter_name, threshold, failed) in enumerate(filters):
        fail_pct = (failed / n_total * 100) if n_total > 0 else 0

        # Color code based on failure rate
        if fail_pct > 50:
            fail_color = Colors.WARNING
        elif fail_pct > 20:
            fail_color = Colors.WARNING
        else:
            fail_color = Colors.GREEN

        fail_str = f"{failed:,} ({fail_pct:.1f}%)"

        row = f"   {Colors.CYAN}│{Colors.ENDC} {Colors.BLUE}{filter_name:<28}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC} {threshold:<23} {Colors.CYAN}│{Colors.ENDC} {fail_color}{fail_str:<13}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC}"
        print(row)

        if i < len(filters) - 1:
            print(f"   {Colors.CYAN}{mid_line}{Colors.ENDC}")

    print(f"   {Colors.CYAN}{bot_line}{Colors.ENDC}")

    # Total summary
    total_passed = n_total - (mt_failed + umis_failed + genes_failed)
    retention = (total_passed / n_total * 100) if n_total > 0 else 0
    print(f"   {Colors.GREEN}→ Total cells passing filters: {Colors.BOLD}{total_passed:,}{Colors.ENDC}{Colors.GREEN} ({retention:.1f}% retention){Colors.ENDC}")


def _print_final_summary_table(n_start, n_end, genes_start, genes_end):
    """Print final QC summary in a colored table format."""
    top_line = "┌" + "─" * 25 + "┬" + "─" * 20 + "┬" + "─" * 20 + "┐"
    mid_line = "├" + "─" * 25 + "┼" + "─" * 20 + "┼" + "─" * 20 + "┤"
    bot_line = "└" + "─" * 25 + "┴" + "─" * 20 + "┴" + "─" * 20 + "┘"

    print(f"\n{Colors.HEADER}{Colors.BOLD}📈 Final QC Summary:{Colors.ENDC}")
    print(f"   {Colors.CYAN}{top_line}{Colors.ENDC}")

    # Header
    header = f"   {Colors.CYAN}│{Colors.ENDC} {Colors.BOLD}{'Dimension':<23}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC} {Colors.BOLD}{'Before QC':<18}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC} {Colors.BOLD}{'After QC':<18}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC}"
    print(header)
    print(f"   {Colors.CYAN}{mid_line}{Colors.ENDC}")

    # Cells
    cells_removed = n_start - n_end
    cells_retention = (n_end / n_start * 100) if n_start > 0 else 0
    row1 = f"   {Colors.CYAN}│{Colors.ENDC} {Colors.BLUE}{'Cells':<23}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC} {n_start:>18,} {Colors.CYAN}│{Colors.ENDC} {Colors.BOLD}{Colors.GREEN}{n_end:>18,}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC}"
    print(row1)
    print(f"   {Colors.CYAN}{mid_line}{Colors.ENDC}")

    # Genes
    genes_removed = genes_start - genes_end
    genes_retention = (genes_end / genes_start * 100) if genes_start > 0 else 0
    row2 = f"   {Colors.CYAN}│{Colors.ENDC} {Colors.BLUE}{'Genes':<23}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC} {genes_start:>18,} {Colors.CYAN}│{Colors.ENDC} {Colors.BOLD}{Colors.GREEN}{genes_end:>18,}{Colors.ENDC} {Colors.CYAN}│{Colors.ENDC}"
    print(row2)

    print(f"   {Colors.CYAN}{bot_line}{Colors.ENDC}")

    # Retention summary
    if cells_retention >= 80:
        quality_color = Colors.GREEN
        quality_msg = "Excellent"
    elif cells_retention >= 60:
        quality_color = Colors.CYAN
        quality_msg = "Good"
    elif cells_retention >= 40:
        quality_color = Colors.WARNING
        quality_msg = "Moderate"
    else:
        quality_color = Colors.WARNING
        quality_msg = "Low"

    print(f"\n   {quality_color}💯 Quality Assessment: {Colors.BOLD}{quality_msg}{Colors.ENDC}{quality_color} retention{Colors.ENDC}")
    print(f"   {Colors.BLUE}   • Cells retained: {Colors.BOLD}{cells_retention:.1f}%{Colors.ENDC}{Colors.BLUE} ({cells_removed:,} removed){Colors.ENDC}")
    print(f"   {Colors.BLUE}   • Genes retained: {Colors.BOLD}{genes_retention:.1f}%{Colors.ENDC}{Colors.BLUE} ({genes_removed:,} removed){Colors.ENDC}")
    print(f"   {Colors.CYAN}{'─' * 70}{Colors.ENDC}")




@monitor
def mads(meta, cov, nmads=5, lt=None, batch_key=None):
    """
    Calculate Median Absolute Deviation (MAD) thresholds.
    
    If batch_key is provided, calculates MAD per batch and returns
    a dictionary mapping batch values to (lower, upper) thresholds.
    If batch_key is None, calculates global MAD as before.
    
    Parameters
    ----------
    meta : pd.DataFrame
        Metadata dataframe containing the column to analyze
    cov : str
        Column name to calculate MAD for
    nmads : float, default 5
        Number of MADs for threshold calculation
    lt : dict, optional
        Lower thresholds to use if calculated threshold <= 0
    batch_key : str, optional
        Column name for batch information. If provided, calculates per-batch MAD
        
    Returns
    -------
    If batch_key is None:
        tuple: (lower_threshold, upper_threshold)
    If batch_key is provided:
        dict: {batch_value: (lower_threshold, upper_threshold)}
    """
    if batch_key is None:
        # Original global MAD calculation
        x = meta[cov]
        mad = np.median(np.absolute(x - np.median(x)))
        t1 = np.median(x) - (nmads * mad)
        t1 = t1 if t1 > 0 else lt[cov]
        t2 = np.median(x) + (nmads * mad)
        return t1, t2
    else:
        # Batch-wise MAD calculation
        batch_thresholds = {}
        for batch in meta[batch_key].unique():
            batch_mask = meta[batch_key] == batch
            x_batch = meta.loc[batch_mask, cov]
            
            if len(x_batch) == 0:
                continue
                
            mad = np.median(np.absolute(x_batch - np.median(x_batch)))
            t1 = np.median(x_batch) - (nmads * mad)
            t1 = t1 if t1 > 0 else lt[cov]
            t2 = np.median(x_batch) + (nmads * mad)
            batch_thresholds[batch] = (t1, t2)
            
        return batch_thresholds

def mads_test(meta, cov, nmads=5, lt=None, batch_key=None):
    """
    Apply MAD-based filtering with optional batch awareness.
    
    Returns a boolean array with True values for entries within MAD thresholds.
    If batch_key is provided, applies per-batch MAD thresholds.
    
    Parameters
    ----------
    meta : pd.DataFrame
        Metadata dataframe containing the column to analyze
    cov : str
        Column name to test
    nmads : float, default 5
        Number of MADs for threshold calculation
    lt : dict, optional
        Lower thresholds to use if calculated threshold <= 0
    batch_key : str, optional
        Column name for batch information. If provided, applies per-batch MAD
        
    Returns
    -------
    pd.Series
        Boolean series indicating which cells pass the MAD filter
    """
    if batch_key is None:
        # Original global MAD test
        thresholds = mads(meta, cov, nmads=nmads, lt=lt)
        return (meta[cov] > thresholds[0]) & (meta[cov] < thresholds[1])
    else:
        # Batch-wise MAD test
        batch_thresholds = mads(meta, cov, nmads=nmads, lt=lt, batch_key=batch_key)
        result = pd.Series(False, index=meta.index, dtype=bool)
        
        for batch, (t1, t2) in batch_thresholds.items():
            batch_mask = meta[batch_key] == batch
            batch_test = (meta.loc[batch_mask, cov] > t1) & (meta.loc[batch_mask, cov] < t2)
            result.loc[batch_mask] = batch_test
            
        return result

# C. elegans mitochondrial gene prefixes (no single shared prefix)
_CE_MT_PREFIXES = ('ctc-', 'nduo-', 'ctb-')


# Minimum pyscdblfinder that ships the O(N·k) kNN doublet features. Earlier
# releases (<=0.1.0) built a full N×N distance matrix in the neighbour step,
# which needs ~N²·8 bytes (≈125 GB at 100k cells) and thrashes swap — the
# process shows near-zero CPU and never finishes on large atlases (issue #848).
_MIN_PYSCDBLFINDER = "0.2.0"


def _resolve_doublets_method(method: str) -> str:
    """Resolve a doublets_method string, falling back gracefully when the
    requested backend's package is missing or too old.

    `scdblfinder` (default) requires `pyscdblfinder`; if it's missing we
    print a single-line install hint and fall back to `scrublet` (which
    has lazy imports of its own and is broadly available). If an outdated
    pyscdblfinder (<0.2.0) is installed we also fall back to `scrublet`,
    because those versions stall for hours with near-zero CPU on large
    datasets (issue #848) — the pipeline should finish rather than hang.
    Other methods pass through unchanged — their wrappers raise their own
    ImportError with installation instructions.
    """
    if method == 'scdblfinder':
        try:
            import pyscdblfinder  # noqa: F401
        except ImportError:
            print(
                f"   {Colors.WARNING}⚠️  pyscdblfinder is not installed; "
                f"falling back to 'scrublet'.{Colors.ENDC}"
            )
            print(
                f"   {Colors.CYAN}💡 Install with: "
                f"`pip install \"pyscdblfinder>={_MIN_PYSCDBLFINDER}\"` "
                f"to use the new default.{Colors.ENDC}"
            )
            return 'scrublet'
        # Version read from distribution metadata (not pyscdblfinder.__version__,
        # which 0.2.0 ships stale as '0.1.0'); helper lives in utils so other
        # optional backends can reuse the same gate.
        from ..utils._versions import version_at_least
        ok, ver = version_at_least('pyscdblfinder', _MIN_PYSCDBLFINDER)
        if ver is not None and not ok:
            print(
                f"   {Colors.WARNING}⚠️  pyscdblfinder {ver} has a known "
                f"large-dataset stall (near-zero CPU for hours on ~100k+ "
                f"cells; issue #848); falling back to 'scrublet'.{Colors.ENDC}"
            )
            print(
                f"   {Colors.CYAN}💡 Upgrade with: "
                f"`pip install -U \"pyscdblfinder>={_MIN_PYSCDBLFINDER}\"` "
                f"to use scdblfinder on large data.{Colors.ENDC}"
            )
            return 'scrublet'
    return method


def _detect_mt_prefix(var_names) -> str:
    """Auto-detect mitochondrial gene prefix from variable names.

    Checks common prefixes across species and returns the one with the
    most matches.  Falls back to ``'MT-'`` if nothing is found.

    Supported species / prefixes:

    * Human, pig, cattle and most vertebrates: ``MT-``
    * Mouse, rat, zebrafish: ``mt-``
    * Some mixed annotations: ``Mt-``
    * Drosophila: ``mt:``
    * Arabidopsis thaliana: ``ATMG``
    * C. elegans: ``ctc-``, ``nduo-`` and ``ctb-`` (checked as a group)
    """
    if isinstance(var_names, list):
        var_names = pd.Index(var_names)

    candidates = [
        'MT-',    # Human / pig / cattle / most vertebrates
        'mt-',    # Mouse / rat / zebrafish
        'Mt-',    # Some annotations
        'mt:',    # Drosophila (e.g. mt:CoI, mt:ND1)
        'ATMG',   # Arabidopsis thaliana mitochondrial genes
    ]

    best_prefix = 'MT-'
    best_count = 0
    for prefix in candidates:
        count = int(var_names.str.startswith(prefix).sum())
        if count > best_count:
            best_count = count
            best_prefix = prefix

    # C. elegans: mitochondrial genes use heterogeneous names
    # (ctc-1, ctc-2, ctc-3, nduo-1 … nduo-6, ctb-1, etc.)
    # Note: 'ctc-' could appear in other contexts (e.g. clone names),
    # but this check only runs when no standard MT prefix was found.
    if best_count == 0:
        ce_count = int(var_names.str.startswith(_CE_MT_PREFIXES).sum())
        if ce_count > 0:
            best_prefix = 'ctc-'  # representative prefix
            best_count = ce_count

    if best_count == 0:
        # Case-insensitive fallback as last resort.
        # [:3] is safe because all MT- variants are exactly 3 characters.
        count = int(var_names.str.upper().str.startswith('MT-').sum())
        if count > 0:
            mt_mask = var_names.str.upper().str.startswith('MT-')
            first_mt = var_names[mt_mask][0]
            best_prefix = first_mt[:3]
            best_count = count

    return best_prefix


def _mt_mask(var_names, mt_startswith):
    """Return boolean mask for mitochondrial genes, handling multi-prefix species.

    When ``mt_startswith`` is any of the C. elegans prefixes (``'ctc-'``,
    ``'nduo-'``, ``'ctb-'``), all three prefixes are matched together
    because C. elegans mitochondrial genes do not share a single prefix.
    """
    if mt_startswith == 'auto':
        raise ValueError(
            "mt_startswith='auto' was not resolved before calling _mt_mask. "
            "Call _detect_mt_prefix() first."
        )
    if isinstance(var_names, list):
        var_names = pd.Index(var_names)
    if mt_startswith in _CE_MT_PREFIXES:
        return var_names.str.startswith(_CE_MT_PREFIXES)
    return var_names.str.startswith(mt_startswith)


@register_function(
    aliases=["qc_metrics", "calculate_qc_metrics", "质控指标", "质量指标"],
    category="preprocessing",
    description=(
        "Compute per-cell QC metrics WITHOUT filtering: flags mt/ribo/hb "
        "genes, runs scanpy calculate_qc_metrics, and adds omicverse aliases "
        "nUMIs / detected_genes / mito_perc. Inspect with ov.pl.qc, then "
        "filter with ov.pp.qc(tresh=...)."
    ),
    requires={},
    produces={"obs": ["nUMIs", "detected_genes", "mito_perc",
                       "pct_counts_mt", "pct_counts_ribo", "pct_counts_hb"],
              "var": ["mt", "ribo", "hb"]},
    examples=[
        "ov.pp.qc_metrics(adata)",
        "ov.pp.qc_metrics(adata, mt_startswith='mt-')  # mouse",
    ],
    related=["pp.qc", "pl.qc"],
)
def qc_metrics(
    adata,
    *,
    mt_startswith: str = "auto",
    mt_genes=None,
    ribo_startswith=("RPS", "RPL"),
    ribo_genes=None,
    hb_startswith="^HB[^(P)]",
    hb_genes=None,
):
    r"""Compute per-cell QC metrics without filtering.

    Flags mitochondrial / ribosomal / haemoglobin genes, runs scanpy's
    ``calculate_qc_metrics`` (``total_counts``, ``n_genes_by_counts``,
    ``pct_counts_{mt,ribo,hb}``), and adds the omicverse aliases ``nUMIs``,
    ``detected_genes`` and ``mito_perc`` so both naming conventions are
    available. Nothing is removed — this is the "look before you cut" step:
    inspect the distributions with :func:`omicverse.pl.qc` (optionally per
    sample), then filter with :func:`omicverse.pp.qc` using ``tresh``.

    Parameters
    ----------
    adata
        AnnData with raw counts in ``adata.X``.
    mt_startswith, mt_genes
        Mitochondrial gene prefix (``'auto'`` detects ``MT-``/``mt-``) or an
        explicit gene list (overrides the prefix).
    ribo_startswith, ribo_genes, hb_startswith, hb_genes
        Ribosomal / haemoglobin gene prefixes or explicit lists.

    Returns
    -------
    AnnData
        ``adata`` with QC metrics added to ``obs``/``var`` (in place).
    """
    import scanpy as sc

    var_names = adata.var_names
    if mt_startswith == "auto" and mt_genes is None:
        mt_startswith = _detect_mt_prefix(var_names)
    if mt_genes is not None:
        adata.var["mt"] = adata.var_names.isin(list(mt_genes))
    else:
        adata.var["mt"] = _mt_mask(var_names, mt_startswith)
    if ribo_genes is not None:
        adata.var["ribo"] = adata.var_names.isin(list(ribo_genes))
    else:
        adata.var["ribo"] = var_names.str.startswith(tuple(ribo_startswith))
    if hb_genes is not None:
        adata.var["hb"] = adata.var_names.isin(list(hb_genes))
    else:
        adata.var["hb"] = var_names.str.contains(hb_startswith)

    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=False,
        percent_top=None,
    )
    # omicverse aliases (same definitions ov.pp.qc filters on)
    adata.obs["nUMIs"] = adata.obs["total_counts"]
    adata.obs["detected_genes"] = adata.obs["n_genes_by_counts"]
    adata.obs["mito_perc"] = adata.obs["pct_counts_mt"] / 100.0
    return adata


@monitor
@register_function(
    aliases=["质控", "qc", "quality_control", "质量控制"],
    category="preprocessing",
    description="Perform comprehensive quality control on single-cell data. For seurat mode, use tresh dict with keys: 'mito_perc', 'nUMIs', 'detected_genes'",
    prerequisites={},
    requires={},
    produces={
        'obs': ['n_genes', 'n_counts', 'pct_counts_mt'],
        'var': ['mt', 'n_cells']
    },
    auto_fix='none',
    examples=[
        "ov.pp.qc(adata, tresh={'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250})",
        "ov.pp.qc(adata, mode='mads', nmads=5, doublets=True)"
    ],
    related=["preprocess", "filter_cells", "filter_genes", "scrublet"]
)
@tracked("qc", "ov.pp.qc")
def qc(
    adata,
    *,
    mode: str = "seurat",
    min_cells: int = 3,
    min_genes: int = 200,
    nmads: int = 5,
    max_cells_ratio: float = 1,
    max_genes_ratio: float = 1,
    batch_key=None,
    doublets: bool = True,
    doublets_method: str = "scdblfinder",
    filter_doublets: bool = True,
    path_viz=None,
    tresh=None,
    mt_startswith: str = "auto",
    mt_genes=None,
    ribo_startswith=("RPS", "RPL"),
    ribo_genes=None,
    hb_startswith="^HB[^(P)]",
    hb_genes=None,
    use_gpu: bool = True,
    batch_wise_mad=None,
    **kwargs,
):
    r'''
    Perform quality control on an AnnData object.

    Doublet detection is performed after QC threshold filtering and final cell/gene filtering.

    Args:
        adata : AnnData object
        mode : The filtering method to use. Valid options are 'seurat'
        and 'mads'. Default is 'seurat'.
        min_cells : The minimum number of cells expressing a gene to retain that gene. Default is 3.
        min_genes : The minimum number of genes for a cell to pass QC. Default is 200.
        max_cells_ratio : The maximum number of cells ratio for a sample to pass QC. Default is 1.
        max_genes_ratio : The maximum number of genes ratio for a cell to pass QC. Default is 1.
        nmads : The number of MADs to use for MADs filtering. Default is 5.
        batch_key : Column in adata.obs for per-batch MAD filtering and doublet detection. Default is None.
        doublets : Whether to perform doublet detection. Default is True.
        doublets_method : The doublet detection method to use. Options are 'scrublet', 'sccomposite', 'doubletfinder', or 'scdblfinder'. Default is 'scdblfinder' (Python port of R scDblFinder; xgboost on kNN+cxds features) for `cpu` and `cpu-gpu-mixed` modes, falling back to 'scrublet' if pyscdblfinder is not installed. The RAPIDS `gpu` mode keeps 'scrublet' as default for backwards compatibility.
        filter_doublets : Whether to filter out doublets (True) or just flag them (False). Default is True.
        path_viz : The path to save the QC plots. Default is None.
        tresh : A dictionary of QC thresholds. The keys should be 'mito_perc',
        'nUMIs', and 'detected_genes'.
            Only used if mode is 'seurat'. Default is None.
        mt_startswith : The prefix of mitochondrial genes. Default is 'auto',
            which automatically detects the prefix (e.g. 'MT-' for human,
            'mt-' for mouse). Set explicitly (e.g. 'MT-') to override.
        mt_genes : The list of mitochondrial genes. Default is None.
        if mt_genes is not None, mt_startswith will be ignored.

    Returns:
        adata : An AnnData object containing cells that passed QC filters.

    Examples:
        >>> import omicverse as ov
        >>> adata = ov.pp.qc(adata, tresh={'mito_perc': 0.2, 'nUMIs': 500, 'detected_genes': 250})
        >>> adata = ov.pp.qc(adata, mode='mads', nmads=5, doublets=True)
        >>> # Auto-detects 'mt-' for mouse data
        >>> adata = ov.pp.qc(adata)
        >>> # Explicit prefix
        >>> adata = ov.pp.qc(adata, mt_startswith='mt-')

    '''

    # Re-pack explicit kwargs so the per-mode implementations see them.
    kwargs.update(dict(
        mode=mode, min_cells=min_cells, min_genes=min_genes, nmads=nmads,
        max_cells_ratio=max_cells_ratio, max_genes_ratio=max_genes_ratio,
        batch_key=batch_key, doublets=doublets,
        doublets_method=doublets_method, filter_doublets=filter_doublets,
        path_viz=path_viz, tresh=tresh,
        mt_startswith=mt_startswith, mt_genes=mt_genes,
        ribo_startswith=ribo_startswith, ribo_genes=ribo_genes,
        hb_startswith=hb_startswith, hb_genes=hb_genes,
    ))
    # qc_cpu_gpu_mixed accepts these; qc_cpu/qc_gpu may not — filter
    # only where needed.
    mixed_only = dict(use_gpu=use_gpu, batch_wise_mad=batch_wise_mad)

    if _is_oom(adata):
        print(f"{Colors.HEADER}{Colors.BOLD}{EMOJI['cpu']} Using CPU mode for QC (out-of-memory)...{Colors.ENDC}")
        result = qc_cpu(adata, **kwargs)
        note(backend=f"omicverse({settings.mode}) · cpu(oom) · mode={mode}")
    elif settings.mode == 'gpu':
        print(f"{Colors.HEADER}{Colors.BOLD}{EMOJI['gpu']} Using RAPIDS GPU to calculate QC...{Colors.ENDC}")
        result = qc_gpu(adata, **kwargs)
        note(backend=f"omicverse({settings.mode}) · rapids · mode={mode}")
    elif settings.mode == 'cpu-gpu-mixed':
        print(f"{Colors.HEADER}{Colors.BOLD}{EMOJI['mixed']} Using CPU/GPU mixed mode for QC...{Colors.ENDC}")
        print_gpu_usage_color()
        result = qc_cpu_gpu_mixed(adata, **kwargs, **mixed_only)
        note(backend=f"omicverse({settings.mode}) · mixed · mode={mode}")
    else:
        print(f"{Colors.HEADER}{Colors.BOLD}{EMOJI['cpu']} Using CPU mode for QC...{Colors.ENDC}")
        result = qc_cpu(adata, **kwargs)
        note(backend=f"omicverse({settings.mode}) · cpu · mode={mode}")

    # Build viz from whatever actually ended up on the (possibly filtered)
    # AnnData. qc invokes a doublet backend internally; surface its plot
    # here rather than let it leak as a phantom sub-entry.
    target = result if (result is not None and hasattr(result, "uns")) else adata
    qc_metrics = [c for c in ("nUMIs", "detected_genes", "mito_perc",
                                "ribo_perc", "hb_perc", "doublet_score")
                   if c in target.obs.columns]
    viz = []
    if qc_metrics:
        viz.append({"function": "ov.pl.violin",
                    "kwargs": {"keys": qc_metrics}})
    if doublets and "doublet_score" in target.obs.columns:
        viz.append({"function": "ov.pl.doublet_score_histogram", "kwargs": {}})
    note(viz=viz)
    return result
    

def qc_cpu_gpu_mixed(adata:anndata.AnnData, mode='seurat',
       min_cells=3, min_genes=200, nmads=5,
       max_cells_ratio=1,max_genes_ratio=1,
       batch_key=None,doublets=True,doublets_method='scdblfinder',
       filter_doublets=True,
       path_viz=None, tresh=None,mt_startswith='auto',mt_genes=None,
       ribo_startswith=("RPS", "RPL"),ribo_genes=None,
       hb_startswith="^HB[^(P)]",hb_genes=None,
       use_gpu=True,batch_wise_mad=None):
    """
    Perform quality control on an AnnData object.

    This function calculates several QC metrics, including mitochondrial percentage, nUMIs,
    and detected genes, and produces several plots visualizing the QC metrics for each sample.
    Doublet detection is performed after QC threshold filtering and final cell/gene filtering.
    The function returns an AnnData object containing cells that passed QC filters.

    Arguments:
        adata : AnnData object
        mode : The filtering method to use. Valid options are 'seurat'
        and 'mads'. Default is 'seurat'.
        min_cells : The minimum number of cells expressing a gene to retain that gene. Default is 3.
        min_genes : The minimum number of genes for a cell to pass QC. Default is 200.
        max_cells_ratio : The maximum number of cells ratio for a sample to pass QC. Default is 1.
        max_genes_ratio : The maximum number of genes ratio for a cell to pass QC. Default is 1.
        nmads : The number of MADs to use for MADs filtering. Default is 5.
        batch_key : Column in adata.obs for per-batch MAD filtering and doublet detection. Default is None.
        doublets : Whether to perform doublet detection. Default is True.
        doublets_method : The doublet detection method to use. Options are 'scrublet', 'sccomposite', 'doubletfinder', or 'scdblfinder'. Default is 'scdblfinder' (Python port of R scDblFinder; xgboost on kNN+cxds features). Falls back to 'scrublet' if pyscdblfinder is not installed.
        filter_doublets : Whether to filter out doublets (True) or just flag them (False). Default is True.
        path_viz : The path to save the QC plots. Default is None.
        tresh : A dictionary of QC thresholds. The keys should be 'mito_perc',
        'nUMIs', and 'detected_genes'.
            Only used if mode is 'seurat'. Default is None.
        mt_startswith : The prefix of mitochondrial genes. Default is 'MT-'.
        mt_genes : The list of mitochondrial genes. Default is None.
        if mt_genes is not None, mt_startswith will be ignored.

    Returns:
        adata : An AnnData object containing cells that passed QC filters.

    """
    # Logging
    if tresh is None:
        tresh={'mito_perc': 0.15, 'nUMIs': 500, 'detected_genes': 250}
    
    removed_cells = []

    print(f"\n{Colors.HEADER}{Colors.BOLD}{EMOJI['start']} Quality Control Analysis (CPU-GPU Mixed):{Colors.ENDC}")
    print(f"   {Colors.CYAN}Dataset shape: {Colors.BOLD}{adata.shape[0]:,} cells × {adata.shape[1]:,} genes{Colors.ENDC}")
    print(f"   {Colors.BLUE}QC mode: {Colors.BOLD}{mode}{Colors.ENDC}")
    print(f"   {Colors.BLUE}Doublet detection: {Colors.BOLD}{doublets_method if doublets else 'disabled'}{Colors.ENDC}")
    # Auto-detect mitochondrial gene prefix
    if mt_startswith == 'auto' and mt_genes is None:
        mt_startswith = _detect_mt_prefix(adata.var_names)
        print(f"   {Colors.CYAN}Auto-detected mitochondrial prefix: {Colors.BOLD}'{mt_startswith}'{Colors.ENDC}")
    print(f"   {Colors.BLUE}Mitochondrial genes: {Colors.BOLD}{mt_startswith if mt_genes is None else 'custom list'}{Colors.ENDC}")

    # QC metrics
    print(f"\n{Colors.HEADER}{Colors.BOLD}📊 Step 1: Calculating QC Metrics{Colors.ENDC}")
    adata.var_names_make_unique()
    if mt_genes is not None:
        adata.var['mt']=False
        adata.var.loc[list(set(adata.var_names) & set(mt_genes)),'mt']=True
        mt_genes_found = sum(adata.var['mt'])
    # print(f"   {Colors.CYAN}Custom mitochondrial genes: {Colors.BOLD}{mt_genes_found}/{len(mt_genes)}{Colors.ENDC}{Colors.CYAN} found{Colors.ENDC}")
    else:
        if type(adata.var_names) is list:
            var_names = pd.Index(adata.var_names)
        else:
            var_names = adata.var_names
        adata.var["mt"] = _mt_mask(var_names, mt_startswith)
        mt_genes_found = sum(adata.var["mt"])
    # print(f"   {Colors.CYAN}Mitochondrial genes (prefix '{mt_startswith}'): {Colors.BOLD}{mt_genes_found}{Colors.ENDC}{Colors.CYAN} found{Colors.ENDC}")

    if ribo_genes is not None:
        adata.var["ribo"] = False
        adata.var.loc[list(set(adata.var_names) & set(ribo_genes)),'ribo']=True
        ribo_genes_found = sum(adata.var["ribo"])
    # print(f"   {Colors.CYAN}Ribosomal genes: {Colors.BOLD}{ribo_genes_found}/{len(ribo_genes)}{Colors.ENDC}{Colors.CYAN} found{Colors.ENDC}")
    else:
        if type(adata.var_names) is list:
            var_names = pd.Index(adata.var_names)
        else:
            var_names = adata.var_names
        adata.var["ribo"] = var_names.str.startswith(ribo_startswith)
        ribo_genes_found = sum(adata.var["ribo"])
    # print(f"   {Colors.CYAN}Ribosomal genes (prefix '{ribo_startswith}'): {Colors.BOLD}{ribo_genes_found}{Colors.ENDC}{Colors.CYAN} found{Colors.ENDC}")

    if hb_genes is not None:
        adata.var["hb"] = False
        adata.var.loc[list(set(adata.var_names) & set(hb_genes)),'hb']=True
        hb_genes_found = sum(adata.var["hb"])
    else:
        if type(adata.var_names) is list:
            var_names = pd.Index(adata.var_names)
        else:
            var_names = adata.var_names
        adata.var["hb"] = var_names.str.contains(hb_startswith)
        hb_genes_found = sum(adata.var["hb"])

    # Print gene detection table
    _print_gene_detection_table(mt_genes_found, ribo_genes_found, hb_genes_found, mt_genes, ribo_genes, hb_genes)

    # Check if it's a Rust backend
    is_rust = _is_rust_backend(adata)
    
    if issparse(adata.X):
        adata.obs['nUMIs'] = np.array(adata.X.sum(axis=1)).reshape(-1)
        adata.obs['mito_perc'] = np.array(adata[:, adata.var["mt"]].X.sum(axis=1)).reshape(-1) / \
        adata.obs['nUMIs'].values
        adata.obs['ribo_perc'] = np.array(adata[:, adata.var["ribo"]].X.sum(axis=1)).reshape(-1) / \
        adata.obs['nUMIs'].values
        adata.obs['hb_perc'] = np.array(adata[:, adata.var["hb"]].X.sum(axis=1)).reshape(-1) / \
        adata.obs['nUMIs'].values
        adata.obs['detected_genes'] = adata.X.getnnz(axis=1)
    elif is_rust:
        # For Rust backend (snapatac2) - use adata.X[:] and subset method
        adata.obs['nUMIs'] = np.array(adata.X[:].sum(axis=1)).reshape(-1)
        # Use subset method for Rust backend slicing
        mt_indices = np.where(adata.var["mt"])[0]
        ribo_indices = np.where(adata.var["ribo"])[0]
        hb_indices = np.where(adata.var["hb"])[0]
        if len(mt_indices) > 0:
            #adata.X[:,mt_indices].sum(axis=1) / adata.obs['nUMIs'].values
            adata.obs['mito_perc'] = np.array(adata.X[:,mt_indices].sum(axis=1)).reshape(-1) / adata.obs['nUMIs']
            adata.obs['ribo_perc'] = np.array(adata.X[:,ribo_indices].sum(axis=1)).reshape(-1) / adata.obs['nUMIs']
            adata.obs['hb_perc'] = np.array(adata.X[:,hb_indices].sum(axis=1)).reshape(-1) / adata.obs['nUMIs']
        else:
            adata.obs['mito_perc'] = np.zeros(adata.n_obs)
            adata.obs['ribo_perc'] = np.zeros(adata.n_obs)
            adata.obs['hb_perc'] = np.zeros(adata.n_obs)
        adata.obs['detected_genes'] = adata.X[:].getnnz(axis=1)
    else:
        # Regular pandas backend
        adata.obs['nUMIs'] = adata.X.sum(axis=1)
        adata.obs['mito_perc'] = adata[:, adata.var["mt"]].X.sum(axis=1) / \
        adata.obs['nUMIs'].values
        adata.obs['ribo_perc'] = adata[:, adata.var["ribo"]].X.sum(axis=1) / \
        adata.obs['nUMIs'].values
        adata.obs['hb_perc'] = adata[:, adata.var["hb"]].X.sum(axis=1) / \
        adata.obs['nUMIs'].values
        adata.obs['detected_genes'] = np.count_nonzero(adata.X, axis=1)
    adata.obs['cell_complexity'] = adata.obs['detected_genes'] / adata.obs['nUMIs']

    # Scanpy-convention aliases — let LLM-generated code that follows scanpy
    # tutorials (`adata.obs['pct_counts_mt']`, `adata.obs['n_counts']`,
    # `adata.obs['total_counts']`, `adata.obs['n_genes']`) work without
    # KeyError. ov uses 'mito_perc'/'nUMIs'/'detected_genes' internally;
    # we mirror the values under standard scanpy names so both call styles
    # succeed. Cheap (no data duplication, just dict assignment).
    adata.obs['n_counts']        = adata.obs['nUMIs']
    adata.obs['total_counts']    = adata.obs['nUMIs']
    adata.obs['n_genes']         = adata.obs['detected_genes']
    adata.obs['n_genes_by_counts'] = adata.obs['detected_genes']
    adata.obs['pct_counts_mt']   = adata.obs['mito_perc']
    if 'ribo_perc' in adata.obs.columns:
        adata.obs['pct_counts_ribo'] = adata.obs['ribo_perc']
    if 'hb_perc' in adata.obs.columns:
        adata.obs['pct_counts_hb']   = adata.obs['hb_perc']

    # Display QC statistics in table format
    _print_qc_metrics_table(adata)

    # Original QC plot
    n0 = adata.shape[0]

    # Post seurat or mads filtering QC plot

    # Filters
    print(f"\n{Colors.HEADER}{Colors.BOLD}🔧 Step 2: Quality Filtering ({mode.upper()}){Colors.ENDC}")
    print(f"   {Colors.CYAN}Thresholds: mito≤{tresh['mito_perc']}, nUMIs≥{tresh['nUMIs']}, genes≥{tresh['detected_genes']}{Colors.ENDC}")
    
    if mode == 'seurat':
        adata.obs['passing_mt'] = adata.obs['mito_perc'] < tresh['mito_perc']
        adata.obs['passing_nUMIs'] = adata.obs['nUMIs'] > tresh['nUMIs']
        adata.obs['passing_ngenes'] = adata.obs['detected_genes'] > tresh['detected_genes']
    elif mode == 'mads':
        adata.obs['passing_mt'] = adata.obs['mito_perc'] < tresh['mito_perc']
        adata.obs['passing_nUMIs'] = mads_test(adata.obs, 'nUMIs', nmads=nmads, lt=tresh, batch_key=batch_key)
        adata.obs['passing_ngenes'] = mads_test(adata.obs, 'detected_genes', nmads=nmads, lt=tresh, batch_key=batch_key)

    # Report
    n1 = adata.shape[0]
    mt_failed = n1-adata.obs["passing_mt"].sum()
    umis_failed = n1-adata.obs["passing_nUMIs"].sum()
    genes_failed = n1-adata.obs["passing_ngenes"].sum()
    
    if mode == 'seurat':
        print(f"   {Colors.BLUE}📊 Seurat Filter Results:{Colors.ENDC}")
        print(f"     {Colors.CYAN}• nUMIs filter (≥{tresh['nUMIs']}): {Colors.BOLD}{umis_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed ({umis_failed/n1*100:.1f}%){Colors.ENDC}")
        print(f"     {Colors.CYAN}• Genes filter (≥{tresh['detected_genes']}): {Colors.BOLD}{genes_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed ({genes_failed/n1*100:.1f}%){Colors.ENDC}")
        print(f"     {Colors.CYAN}• Mitochondrial filter (≤{tresh['mito_perc']}): {Colors.BOLD}{mt_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed ({mt_failed/n1*100:.1f}%){Colors.ENDC}")
    elif mode == 'mads':
        nUMIs_t = mads(adata.obs, 'nUMIs', nmads=nmads, lt=tresh)
        n_genes_t = mads(adata.obs, 'detected_genes', nmads=nmads, lt=tresh)
        print(f"   {Colors.BLUE}📊 MADs Filter Results (±{nmads} MADs):{Colors.ENDC}")
        print(f"     {Colors.CYAN}• nUMIs range ({nUMIs_t[0]:.0f}, {nUMIs_t[1]:.0f}): {Colors.BOLD}{umis_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed{Colors.ENDC}")
        print(f"     {Colors.CYAN}• Genes range ({n_genes_t[0]:.0f}, {n_genes_t[1]:.0f}): {Colors.BOLD}{genes_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed{Colors.ENDC}")
        print(f"     {Colors.CYAN}• Mitochondrial filter (≤{tresh['mito_perc']}): {Colors.BOLD}{mt_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed{Colors.ENDC}")

    # QC plot
    QC_test = (adata.obs['passing_mt']) & (adata.obs['passing_nUMIs']) & (adata.obs['passing_ngenes'])
    adata.obs['passing_qc_step1']=QC_test
    removed = list(np.array(adata.obs_names)[np.where(adata.obs["passing_qc_step1"]==False)[0]])
    keeped=list(np.array(adata.obs_names)[np.where(adata.obs["passing_qc_step1"]==True)[0]])

    removed_cells+=removed
    total_qc_failed = n1-len(removed)
    if is_rust:
        adata.subset(obs_indices=keeped)
    else:
        adata = adata[QC_test, :]
    n2 = adata.shape[0]

    print(f"   {Colors.GREEN}✓ Combined QC filters: {Colors.BOLD}{total_qc_failed:,}{Colors.ENDC}{Colors.GREEN} cells kept ({total_qc_failed/n1*100:.1f}%){Colors.ENDC}")

    # Last gene and cell filter
    print(f"\n{Colors.HEADER}{Colors.BOLD}🎯 Step 3: Final Filtering{Colors.ENDC}")
    print(f"   {Colors.CYAN}Parameters: min_genes={min_genes}, min_cells={min_cells}{Colors.ENDC}")
    print(f"   {Colors.CYAN}Ratios: max_genes_ratio={max_genes_ratio}, max_cells_ratio={max_cells_ratio}{Colors.ENDC}")
    
    cells_before_final = adata.shape[0]
    genes_before_final = adata.shape[1]
    
    if not is_rust:
        _filter_cells_impl(adata, min_genes=min_genes)
        _filter_genes_impl(adata, min_cells=min_cells)
        _filter_cells_impl(adata, max_genes=int(max_genes_ratio*adata.shape[1]))
        _filter_genes_impl(adata, max_cells=int(max_cells_ratio*adata.shape[0]))
    else:
        selected_cells = True
        if min_genes: selected_cells &= adata.obs["detected_genes"] >= min_genes
        if max_genes_ratio: selected_cells &= adata.obs["detected_genes"] <= max_genes_ratio*adata.shape[1]
        selected_cells = np.flatnonzero(selected_cells)
        adata.subset(obs_indices=selected_cells)

        selected_genes = True
        # Count non-zero cells per gene. For OOM, stream in chunks rather than
        # materialising X[:]; for raw rust, fall back to the one-shot read.
        if _is_oom(adata):
            from anndataoom import chunked_qc_metrics
            if "n_cells" not in adata.var.columns:
                chunked_qc_metrics(adata)
            adata.var["n_cell"] = adata.var["n_cells"].values
        else:
            adata.var["n_cell"] = np.array((adata.X[:] != 0).sum(axis=0)).reshape(-1)
        if min_cells: selected_genes &= adata.var["n_cell"] >= min_cells
        if max_cells_ratio: selected_genes &= adata.var["n_cell"] <= max_cells_ratio*adata.shape[0]
        selected_genes = np.flatnonzero(selected_genes)
        adata.subset(var_indices=selected_genes)

    
    cells_final_filtered = cells_before_final - adata.shape[0]
    genes_final_filtered = genes_before_final - adata.shape[1]
    
    print(f"   {Colors.GREEN}✓ Final filtering: {Colors.BOLD}{cells_final_filtered:,}{Colors.ENDC}{Colors.GREEN} cells, {Colors.BOLD}{genes_final_filtered:,}{Colors.ENDC}{Colors.GREEN} genes removed{Colors.ENDC}")
    
    n_after_final_filt = adata.shape[0]
    
    if doublets is True:
        print(f"\n{Colors.HEADER}{Colors.BOLD}🔍 Step 4: Doublet Detection{Colors.ENDC}")
        doublets_method = _resolve_doublets_method(doublets_method)
        if doublets_method not in ('scrublet', 'sccomposite', 'doubletfinder', 'scdblfinder'):
            raise ValueError(
                f"Unknown doublets_method={doublets_method!r}; "
                "expected 'scrublet', 'sccomposite', 'doubletfinder', or 'scdblfinder'."
            )
        if doublets_method=='scrublet':
            # Post doublets removal QC plot
            print(f"   {Colors.WARNING}⚠️  Note: 'scrublet' detection is legacy and may not work optimally{Colors.ENDC}")
            print(f"   {Colors.CYAN}💡 Consider using 'doublets_method=scdblfinder' (default) for better results{Colors.ENDC}")
            print(f"   {Colors.GREEN}{EMOJI['start']} Running scrublet doublet detection...{Colors.ENDC}")
            from ._scrublet import scrublet
            scrublet(adata, random_state=1234,batch_key=batch_key,use_gpu=use_gpu)

            if filter_doublets:
                if is_rust:
                    removed=list(np.array(adata.obs_names)[np.where(adata.obs['predicted_doublet']==True)[0]])
                    removed_cells.extend(removed)
                    adata.subset(obs_indices=np.array(adata.obs_names)[np.where(adata.obs['predicted_doublet']==False)[0]])
                else:
                    adata_remove = adata[adata.obs['predicted_doublet'], :]
                    removed_cells.extend(list(adata_remove.obs_names))
                    adata = adata[~adata.obs['predicted_doublet'], :]
                n1 = adata.shape[0]
                doublets_removed = n_after_final_filt-n1
                print(f"   {Colors.GREEN}✓ Scrublet completed: {Colors.BOLD}{doublets_removed:,}{Colors.ENDC}{Colors.GREEN} doublets removed ({doublets_removed/n_after_final_filt*100:.1f}%){Colors.ENDC}")
            else:
                n1 = adata.shape[0]
                doublets_flagged = adata.obs['predicted_doublet'].sum()
                print(f"   {Colors.GREEN}✓ Scrublet completed: {Colors.BOLD}{doublets_flagged:,}{Colors.ENDC}{Colors.GREEN} doublets flagged ({doublets_flagged/n_after_final_filt*100:.1f}%){Colors.ENDC}")
                print(f"   {Colors.CYAN}💡 Doublets retained in adata.obs['predicted_doublet'] for downstream analysis{Colors.ENDC}")

        elif doublets_method=='doubletfinder':
            from ._doubletfinder import doubletfinder as _run_df
            print(f"   {Colors.CYAN}💡 Running py-DoubletFinder (Python port of R DoubletFinder){Colors.ENDC}")
            print(f"   {Colors.GREEN}{EMOJI['start']} Running doubletfinder detection...{Colors.ENDC}")
            if is_rust:
                # Match the sccomposite branch: DoubletFinder needs an in-memory X
                adata = adata.to_memory()
            _run_df(adata, batch_key=batch_key, random_state=1234)
            if filter_doublets:
                if is_rust:
                    mask = ~adata.obs['predicted_doublet'].values
                    removed = list(adata.obs_names[~mask])
                    removed_cells.extend(removed)
                    adata = adata[~adata.obs['predicted_doublet'], :]
                else:
                    adata_remove = adata[adata.obs['predicted_doublet'], :]
                    removed_cells.extend(list(adata_remove.obs_names))
                    adata = adata[~adata.obs['predicted_doublet'], :]
                n1 = adata.shape[0]
                doublets_removed = n_after_final_filt-n1
                print(f"   {Colors.GREEN}✓ DoubletFinder completed: {Colors.BOLD}{doublets_removed:,}{Colors.ENDC}{Colors.GREEN} doublets removed ({doublets_removed/n_after_final_filt*100:.1f}%){Colors.ENDC}")
            else:
                n1 = adata.shape[0]
                doublets_flagged = adata.obs['predicted_doublet'].sum()
                print(f"   {Colors.GREEN}✓ DoubletFinder completed: {Colors.BOLD}{doublets_flagged:,}{Colors.ENDC}{Colors.GREEN} doublets flagged ({doublets_flagged/n_after_final_filt*100:.1f}%){Colors.ENDC}")
                print(f"   {Colors.CYAN}💡 Doublets retained in adata.obs['predicted_doublet']; pANN in adata.obs['doublet_score']{Colors.ENDC}")

        elif doublets_method=='sccomposite':
            if is_rust:
                adata=adata.to_memory()
            print(f"   {Colors.WARNING}⚠️  Note: 'sccomposite' typically removes more cells than 'scrublet'{Colors.ENDC}")
            print(f"   {Colors.GREEN}{EMOJI['start']} Running sccomposite doublet detection...{Colors.ENDC}")
            adata.obs['sccomposite_doublet']=0
            adata.obs['sccomposite_consistency']=0
            if batch_key is None:
                from ._sccomposite import composite_rna
                multiplet_classification, consistency = composite_rna(adata)
                adata.obs['sccomposite_doublet']=multiplet_classification
                adata.obs['sccomposite_consistency']=consistency
            else:
                print(f"   {Colors.CYAN}Processing {len(adata.obs[batch_key].unique())} batches separately...{Colors.ENDC}")
                for batch in adata.obs[batch_key].unique():
                    from ._sccomposite import composite_rna
                    adata_batch=adata[adata.obs[batch_key]==batch]
                    multiplet_classification, consistency = composite_rna(adata_batch)
                    adata.obs.loc[adata_batch.obs.index,'sccomposite_doublet']=\
                    multiplet_classification
                    adata.obs.loc[adata_batch.obs.index,'sccomposite_consistency']=consistency

            if filter_doublets:
                adata_remove = adata[adata.obs['sccomposite_doublet']!=0, :]
                removed_cells.extend(list(adata_remove.obs_names))
                adata = adata[adata.obs['sccomposite_doublet']==0, :]
                n1 = adata.shape[0]
                doublets_removed = n_after_final_filt-n1
                print(f"   {Colors.GREEN}✓ sccomposite completed: {Colors.BOLD}{doublets_removed:,}{Colors.ENDC}{Colors.GREEN} doublets removed ({doublets_removed/n_after_final_filt*100:.1f}%){Colors.ENDC}")
            else:
                n1 = adata.shape[0]
                doublets_flagged = (adata.obs['sccomposite_doublet']!=0).sum()
                print(f"   {Colors.GREEN}✓ sccomposite completed: {Colors.BOLD}{doublets_flagged:,}{Colors.ENDC}{Colors.GREEN} doublets flagged ({doublets_flagged/n_after_final_filt*100:.1f}%){Colors.ENDC}")
                print(f"   {Colors.CYAN}💡 Doublets retained in adata.obs['sccomposite_doublet'] for downstream analysis{Colors.ENDC}")

        elif doublets_method=='scdblfinder':
            from ._scdblfinder import scdblfinder as _run_scdbl
            print(f"   {Colors.CYAN}💡 Running pyscdblfinder (Python port of R scDblFinder){Colors.ENDC}")
            print(f"   {Colors.GREEN}{EMOJI['start']} Running scdblfinder detection...{Colors.ENDC}")
            if is_rust:
                adata=adata.to_memory()
            _run_scdbl(adata, batch_key=batch_key, random_state=1234)
            if filter_doublets:
                adata_remove = adata[adata.obs['predicted_doublet'], :]
                removed_cells.extend(list(adata_remove.obs_names))
                adata = adata[~adata.obs['predicted_doublet'], :]
                n1 = adata.shape[0]
                doublets_removed = n_after_final_filt-n1
                print(f"   {Colors.GREEN}✓ scDblFinder completed: {Colors.BOLD}{doublets_removed:,}{Colors.ENDC}{Colors.GREEN} doublets removed ({doublets_removed/n_after_final_filt*100:.1f}%){Colors.ENDC}")
            else:
                n1 = adata.shape[0]
                doublets_flagged = adata.obs['predicted_doublet'].sum()
                print(f"   {Colors.GREEN}✓ scDblFinder completed: {Colors.BOLD}{doublets_flagged:,}{Colors.ENDC}{Colors.GREEN} doublets flagged ({doublets_flagged/n_after_final_filt*100:.1f}%){Colors.ENDC}")
                print(f"   {Colors.CYAN}💡 Doublets retained in adata.obs['predicted_doublet']; score in adata.obs['doublet_score']{Colors.ENDC}")
    else:
        print(f"\n{Colors.BLUE}📊 Step 4: Doublet detection disabled{Colors.ENDC}")
        n1 = adata.shape[0]

    # Store status
    if 'status' not in adata.uns.keys():
        adata.uns['status'] = {}
    if 'status_args' not in adata.uns.keys():
        adata.uns['status_args'] = {}

    adata.uns['status']['qc']=True
    adata.uns['status_args']['qc'] = {
        'mode': mode,
        'min_cells': min_cells,
        'min_genes': min_genes,
        'nmads': nmads,
        'max_cells_ratio': max_cells_ratio,
        'max_genes_ratio': max_genes_ratio,
        'batch_key': batch_key,
        'doublets': doublets,
        'doublets_method': doublets_method,
        'path_viz': path_viz,
        'mito_perc': tresh['mito_perc'],
        'nUMIs': tresh['nUMIs'],
        'detected_genes': tresh['detected_genes'],
    }
    add_reference(adata,'qc','QC with scanpy')
    return adata


def qc_cpu(
    adata:anndata.AnnData, 
    mode='seurat',
    min_cells: Optional[int] = 3, 
    min_genes: Optional[int] = 200, 
    nmads: Optional[int] = 5,
    max_cells_ratio: Optional[float] = 1,
    max_genes_ratio: Optional[float] = 1,
    batch_key: Optional[str] = None,
    doublets: Optional[bool] = True,
    doublets_method: Optional[str] = 'scdblfinder',
    filter_doublets: Optional[bool] = True,
    path_viz: Optional[str] = None, 
    tresh: Optional[dict] = None,
    mt_startswith: Optional[str] = 'auto',
    mt_genes: Optional[list] = None,
    ribo_startswith: Optional[tuple] = ("RPS", "RPL"),
    ribo_genes: Optional[list] = None,
    hb_startswith: Optional[str] = "^HB[^(P)]",
    hb_genes: Optional[list] = None,
    **kwargs
):
    r"""
    Perform quality control on an AnnData object.

    This function calculates several QC metrics, including mitochondrial percentage, nUMIs,
    and detected genes, and produces several plots visualizing the QC metrics for each sample.
    Doublet detection is performed after QC threshold filtering and final cell/gene filtering.
    The function returns an AnnData object containing cells that passed QC filters.

    Arguments:
        adata : AnnData object
        mode : The filtering method to use. Valid options are 'seurat'
        and 'mads'. Default is 'seurat'.
        min_cells : The minimum number of cells expressing a gene to retain that gene. Default is 3.
        min_genes : The minimum number of genes for a cell to pass QC. Default is 200.
        max_cells_ratio : The maximum number of cells ratio for a sample to pass QC. Default is 1.
        max_genes_ratio : The maximum number of genes ratio for a cell to pass QC. Default is 1.
        nmads : The number of MADs to use for MADs filtering. Default is 5.
        batch_key : Column in adata.obs for per-batch MAD filtering and doublet detection. Default is None.
        doublets : Whether to perform doublet detection. Default is True.
        doublets_method : The doublet detection method to use. Options are 'scrublet', 'sccomposite', 'doubletfinder', or 'scdblfinder'. Default is 'scdblfinder' (Python port of R scDblFinder; xgboost on kNN+cxds features). Falls back to 'scrublet' if pyscdblfinder is not installed.
        filter_doublets : Whether to filter out doublets (True) or just flag them (False). Default is True.
        path_viz : The path to save the QC plots. Default is None.
        tresh : A dictionary of QC thresholds. The keys should be 'mito_perc',
        'nUMIs', and 'detected_genes'.
            Only used if mode is 'seurat'. Default is None.
        mt_startswith : The prefix of mitochondrial genes. Default is 'MT-'.
        mt_genes : The list of mitochondrial genes. Default is None.
        if mt_genes is not None, mt_startswith will be ignored.

    Returns:
        adata : An AnnData object containing cells that passed QC filters.

    """
    # Logging
    if tresh is None:
        tresh={'mito_perc': 0.15, 'nUMIs': 500, 'detected_genes': 250}
    # For each adata, produce a figure
    # with PdfPages(path_viz + 'original_QC_by_sample.pdf') as pdf:
    removed_cells = []

    # Auto-detect mitochondrial gene prefix
    if mt_startswith == 'auto' and mt_genes is None:
        mt_startswith = _detect_mt_prefix(adata.var_names)
        print(f"   {Colors.CYAN}Auto-detected mitochondrial prefix: {Colors.BOLD}'{mt_startswith}'{Colors.ENDC}")

    # QC metrics
    print(f"\n{Colors.HEADER}{Colors.BOLD}📊 Step 1: Calculating QC Metrics{Colors.ENDC}")
    adata.var_names_make_unique()
    if mt_genes is not None:
        adata.var['mt']=False
        adata.var.loc[list(set(adata.var_names) & set(mt_genes)),'mt']=True
        mt_genes_found = sum(adata.var['mt'])
    # print(f"   {Colors.CYAN}Custom mitochondrial genes: {Colors.BOLD}{mt_genes_found}/{len(mt_genes)}{Colors.ENDC}{Colors.CYAN} found{Colors.ENDC}")
    else:
        if type(adata.var_names) is list:
            var_names = pd.Index(adata.var_names)
        else:
            var_names = adata.var_names
        adata.var["mt"] = _mt_mask(var_names, mt_startswith)
        mt_genes_found = sum(adata.var["mt"])
    # print(f"   {Colors.CYAN}Mitochondrial genes (prefix '{mt_startswith}'): {Colors.BOLD}{mt_genes_found}{Colors.ENDC}{Colors.CYAN} found{Colors.ENDC}")

    if ribo_genes is not None:
        adata.var["ribo"] = False
        adata.var.loc[list(set(adata.var_names) & set(ribo_genes)),'ribo']=True
        ribo_genes_found = sum(adata.var["ribo"]) 
    # print(f"   {Colors.CYAN}Ribosomal genes: {Colors.BOLD}{ribo_genes_found}/{len(ribo_genes)}{Colors.ENDC}{Colors.CYAN} found{Colors.ENDC}")
    else:
        if type(adata.var_names) is list:
            var_names = pd.Index(adata.var_names)
        else:
            var_names = adata.var_names
        adata.var["ribo"] = var_names.str.startswith(ribo_startswith)
        ribo_genes_found = sum(adata.var["ribo"])
    # print(f"   {Colors.CYAN}Ribosomal genes (prefix '{ribo_startswith}'): {Colors.BOLD}{ribo_genes_found}/{len(ribo_genes)}{Colors.ENDC}{Colors.CYAN} found{Colors.ENDC}")

    if hb_genes is not None:
        adata.var["hb"] = False 
        adata.var.loc[list(set(adata.var_names) & set(hb_genes)),'hb']=True 
        hb_genes_found = sum(adata.var["hb"])
    else:
        if type(adata.var_names) is list:
            var_names = pd.Index(adata.var_names)
        else:
            var_names = adata.var_names
        adata.var["hb"] = var_names.str.contains(hb_startswith)
        hb_genes_found = sum(adata.var["hb"])

    # Print gene detection table
    _print_gene_detection_table(mt_genes_found, ribo_genes_found, hb_genes_found, mt_genes, ribo_genes, hb_genes)
    is_oom = _is_oom(adata)

    if is_oom:
        # Out-of-memory path — chunked, never loads full matrix
        from anndataoom import chunked_qc_metrics, chunked_gene_group_pct
        chunked_qc_metrics(adata)
        adata.obs['mito_perc'] = chunked_gene_group_pct(adata, adata.var["mt"].values)
        adata.obs['ribo_perc'] = chunked_gene_group_pct(adata, adata.var["ribo"].values)
        adata.obs['hb_perc'] = chunked_gene_group_pct(adata, adata.var["hb"].values)
    elif issparse(adata.X):
        adata.obs['nUMIs'] = np.array(adata.X.sum(axis=1)).reshape(-1)
        adata.obs['mito_perc'] = np.array(adata[:, adata.var["mt"]].X.sum(axis=1)).reshape(-1) / \
        adata.obs['nUMIs'].values
        adata.obs['ribo_perc'] = np.array(adata[:, adata.var["ribo"]].X.sum(axis=1)).reshape(-1) / \
        adata.obs['nUMIs'].values
        adata.obs['hb_perc'] = np.array(adata[:, adata.var["hb"]].X.sum(axis=1)).reshape(-1) / \
        adata.obs['nUMIs'].values
        adata.obs['detected_genes'] = adata.X.getnnz(axis=1)
    else:
        # Regular pandas backend
        adata.obs['nUMIs'] = adata.X.sum(axis=1)
        adata.obs['mito_perc'] = adata[:, adata.var["mt"]].X.sum(axis=1) / \
        adata.obs['nUMIs'].values
        adata.obs['ribo_perc'] = adata[:, adata.var["ribo"]].X.sum(axis=1) / \
        adata.obs['nUMIs'].values
        adata.obs['hb_perc'] = adata[:, adata.var["hb"]].X.sum(axis=1) / \
        adata.obs['nUMIs'].values
        adata.obs['detected_genes'] = np.count_nonzero(adata.X, axis=1)
    adata.obs['cell_complexity'] = adata.obs['detected_genes'] / adata.obs['nUMIs']

    # Scanpy-convention aliases — let LLM-generated code that follows scanpy
    # tutorials (`adata.obs['pct_counts_mt']`, `adata.obs['n_counts']`,
    # `adata.obs['total_counts']`, `adata.obs['n_genes']`) work without
    # KeyError. ov uses 'mito_perc'/'nUMIs'/'detected_genes' internally;
    # we mirror the values under standard scanpy names so both call styles
    # succeed. Cheap (no data duplication, just dict assignment).
    adata.obs['n_counts']        = adata.obs['nUMIs']
    adata.obs['total_counts']    = adata.obs['nUMIs']
    adata.obs['n_genes']         = adata.obs['detected_genes']
    adata.obs['n_genes_by_counts'] = adata.obs['detected_genes']
    adata.obs['pct_counts_mt']   = adata.obs['mito_perc']
    if 'ribo_perc' in adata.obs.columns:
        adata.obs['pct_counts_ribo'] = adata.obs['ribo_perc']
    if 'hb_perc' in adata.obs.columns:
        adata.obs['pct_counts_hb']   = adata.obs['hb_perc']

    # Display QC statistics in table format
    _print_qc_metrics_table(adata)

    # Original QC plot
    n0 = adata.shape[0]
    print(f"\n   {Colors.CYAN}📈 Original cell count: {Colors.BOLD}{n0:,}{Colors.ENDC}")

    # Post seurat or mads filtering QC plot

    # Filters
    print(f"\n{Colors.HEADER}{Colors.BOLD}🔧 Step 2: Quality Filtering ({mode.upper()}){Colors.ENDC}")
    print(f"   {Colors.CYAN}Thresholds: mito≤{tresh['mito_perc']}, nUMIs≥{tresh['nUMIs']}, genes≥{tresh['detected_genes']}{Colors.ENDC}")
    
    if mode == 'seurat':
        adata.obs['passing_mt'] = adata.obs['mito_perc'] < tresh['mito_perc']
        adata.obs['passing_nUMIs'] = adata.obs['nUMIs'] > tresh['nUMIs']
        adata.obs['passing_ngenes'] = adata.obs['detected_genes'] > tresh['detected_genes']
    elif mode == 'mads':
        adata.obs['passing_mt'] = adata.obs['mito_perc'] < tresh['mito_perc']
        adata.obs['passing_nUMIs'] = mads_test(adata.obs, 'nUMIs', nmads=nmads, lt=tresh, batch_key=batch_key)
        adata.obs['passing_ngenes'] = mads_test(adata.obs, 'detected_genes', nmads=nmads, lt=tresh, batch_key=batch_key)

    # Report
    n1 = adata.shape[0]
    mt_failed = n1-adata.obs["passing_mt"].sum()
    umis_failed = n1-adata.obs["passing_nUMIs"].sum()
    genes_failed = n1-adata.obs["passing_ngenes"].sum()
    
    if mode == 'seurat':
        print(f"   {Colors.BLUE}📊 Seurat Filter Results:{Colors.ENDC}")
        print(f"     {Colors.CYAN}• nUMIs filter (≥{tresh['nUMIs']}): {Colors.BOLD}{umis_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed ({umis_failed/n1*100:.1f}%){Colors.ENDC}")
        print(f"     {Colors.CYAN}• Genes filter (≥{tresh['detected_genes']}): {Colors.BOLD}{genes_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed ({genes_failed/n1*100:.1f}%){Colors.ENDC}")
        print(f"     {Colors.CYAN}• Mitochondrial filter (≤{tresh['mito_perc']}): {Colors.BOLD}{mt_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed ({mt_failed/n1*100:.1f}%){Colors.ENDC}")
    elif mode == 'mads':
        nUMIs_t = mads(adata.obs, 'nUMIs', nmads=nmads, lt=tresh)
        n_genes_t = mads(adata.obs, 'detected_genes', nmads=nmads, lt=tresh)
        print(f"   {Colors.BLUE}📊 MADs Filter Results (±{nmads} MADs):{Colors.ENDC}")
        print(f"     {Colors.CYAN}• nUMIs range ({nUMIs_t[0]:.0f}, {nUMIs_t[1]:.0f}): {Colors.BOLD}{umis_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed{Colors.ENDC}")
        print(f"     {Colors.CYAN}• Genes range ({n_genes_t[0]:.0f}, {n_genes_t[1]:.0f}): {Colors.BOLD}{genes_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed{Colors.ENDC}")
        print(f"     {Colors.CYAN}• Mitochondrial filter (≤{tresh['mito_perc']}): {Colors.BOLD}{mt_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed{Colors.ENDC}")
    print(f"   {Colors.GREEN}✓ Filters applied successfully{Colors.ENDC}")

    # QC plot
    QC_test = (adata.obs['passing_mt']) & (adata.obs['passing_nUMIs']) & (adata.obs['passing_ngenes'])
    if is_oom:
        passing = QC_test.values if hasattr(QC_test, 'values') else np.asarray(QC_test)
        removed = list(adata.obs_names[~passing])
        removed_cells.extend(removed)
        total_qc_failed = len(removed)
        adata._inplace_subset_obs(passing)
    else:
        removed = QC_test.loc[lambda x : x == False]
        removed_cells.extend(list(removed.index.values))
        total_qc_failed = n1-np.sum(QC_test)
        adata = adata[QC_test, :]
    
    
    n2 = adata.shape[0]
    
    print(f"   {Colors.GREEN}✓ Combined QC filters: {Colors.BOLD}{total_qc_failed:,}{Colors.ENDC}{Colors.GREEN} cells removed ({total_qc_failed/n1*100:.1f}%){Colors.ENDC}")

    # Last gene and cell filter
    print(f"\n{Colors.HEADER}{Colors.BOLD}🎯 Step 3: Final Filtering{Colors.ENDC}")
    print(f"   {Colors.CYAN}Parameters: min_genes={min_genes}, min_cells={min_cells}{Colors.ENDC}")
    print(f"   {Colors.CYAN}Ratios: max_genes_ratio={max_genes_ratio}, max_cells_ratio={max_cells_ratio}{Colors.ENDC}")
    
    cells_before_final = adata.shape[0]
    genes_before_final = adata.shape[1]

    if is_oom:
        # OOM path: use _inplace_subset and chunked column counts
        selected_cells = np.ones(adata.n_obs, dtype=bool)
        if min_genes:
            selected_cells &= adata.obs["detected_genes"].values >= min_genes
        if max_genes_ratio:
            selected_cells &= adata.obs["detected_genes"].values <= max_genes_ratio * adata.shape[1]
        adata._inplace_subset_obs(selected_cells)

        # Recompute n_cells after cell subsetting. Uses chunked streaming —
        # never materialises the full X.
        from anndataoom import chunked_qc_metrics
        chunked_qc_metrics(adata)
        n_cell = adata.var["n_cells"].values
        adata.var["n_cell"] = n_cell
        selected_genes = np.ones(adata.n_vars, dtype=bool)
        if min_cells:
            selected_genes &= n_cell >= min_cells
        if max_cells_ratio:
            selected_genes &= n_cell <= max_cells_ratio * adata.shape[0]
        adata._inplace_subset_var(selected_genes)
    else:
        _filter_cells_impl(adata, min_genes=min_genes)
        _filter_genes_impl(adata, min_cells=min_cells)
        _filter_cells_impl(adata, max_genes=int(max_genes_ratio*adata.shape[1]))
        _filter_genes_impl(adata, max_cells=int(max_cells_ratio*adata.shape[0]))

    cells_final_filtered = cells_before_final - adata.shape[0]
    genes_final_filtered = genes_before_final - adata.shape[1]

    print(f"   {Colors.GREEN}✓ Final filtering: {Colors.BOLD}{cells_final_filtered:,}{Colors.ENDC}{Colors.GREEN} cells, {Colors.BOLD}{genes_final_filtered:,}{Colors.ENDC}{Colors.GREEN} genes removed{Colors.ENDC}")

    n_after_final_filt = adata.shape[0]

    if doublets is True:
        print(f"\n{Colors.HEADER}{Colors.BOLD}🔍 Step 4: Doublet Detection{Colors.ENDC}")
        doublets_method = _resolve_doublets_method(doublets_method)
        if doublets_method not in ('scrublet', 'sccomposite', 'doubletfinder', 'scdblfinder'):
            raise ValueError(
                f"Unknown doublets_method={doublets_method!r}; "
                "expected 'scrublet', 'sccomposite', 'doubletfinder', or 'scdblfinder'."
            )
        if is_oom:
            # Scrublet/sccomposite require in-memory X — convert temporarily
            print(f"   {Colors.CYAN}Converting to in-memory for doublet detection...{Colors.ENDC}")
            adata_mem = adata.to_adata()
        if doublets_method=='scrublet':
            from ._scrublet import scrublet
            # Post doublets removal QC plot
            print(f"   {Colors.WARNING}⚠️  Note: 'scrublet' detection is too old and may not work properly{Colors.ENDC}")
            print(f"   {Colors.CYAN}💡 Consider using 'doublets_method=scdblfinder' (default) for better results{Colors.ENDC}")
            print(f"   {Colors.GREEN}{EMOJI['start']} Running scrublet doublet detection...{Colors.ENDC}")
            if is_oom:
                scrublet(adata_mem, random_state=1234, batch_key=batch_key)
                adata.obs['predicted_doublet'] = adata_mem.obs['predicted_doublet'].values
                if 'doublet_score' in adata_mem.obs.columns:
                    adata.obs['doublet_score'] = adata_mem.obs['doublet_score'].values
                del adata_mem
            else:
                scrublet(adata, random_state=1234,batch_key=batch_key)

            if filter_doublets:
                if is_oom:
                    doublet_mask = ~adata.obs['predicted_doublet'].values
                    removed = list(adata.obs_names[~doublet_mask])
                    removed_cells.extend(removed)
                    adata._inplace_subset_obs(doublet_mask)
                else:
                    adata_remove = adata[adata.obs['predicted_doublet'], :]
                    removed_cells.extend(list(adata_remove.obs_names))
                    adata = adata[~adata.obs['predicted_doublet'], :]
                n1 = adata.shape[0]
                doublets_removed = n_after_final_filt-n1
                print(f"   {Colors.GREEN}✓ Scrublet completed: {Colors.BOLD}{doublets_removed:,}{Colors.ENDC}{Colors.GREEN} doublets removed ({doublets_removed/n_after_final_filt*100:.1f}%){Colors.ENDC}")
            else:
                n1 = adata.shape[0]
                doublets_flagged = adata.obs['predicted_doublet'].sum()
                print(f"   {Colors.GREEN}✓ Scrublet completed: {Colors.BOLD}{doublets_flagged:,}{Colors.ENDC}{Colors.GREEN} doublets flagged ({doublets_flagged/n_after_final_filt*100:.1f}%){Colors.ENDC}")
                print(f"   {Colors.CYAN}💡 Doublets retained in adata.obs['predicted_doublet'] for downstream analysis{Colors.ENDC}")

        elif doublets_method=='doubletfinder':
            from ._doubletfinder import doubletfinder as _run_df
            print(f"   {Colors.CYAN}💡 Running py-DoubletFinder (Python port of R DoubletFinder){Colors.ENDC}")
            print(f"   {Colors.GREEN}{EMOJI['start']} Running doubletfinder detection...{Colors.ENDC}")
            if is_oom:
                _run_df(adata_mem, batch_key=batch_key, random_state=1234)
                # Copy all four columns the wrapper writes — not just the generic pair
                for col in ("predicted_doublet", "doublet_score",
                            "doubletfinder_doublet", "doubletfinder_pANN"):
                    if col in adata_mem.obs.columns:
                        adata.obs[col] = adata_mem.obs[col].values
                del adata_mem
            else:
                _run_df(adata, batch_key=batch_key, random_state=1234)

            if filter_doublets:
                if is_oom:
                    mask = ~adata.obs['predicted_doublet'].values
                    removed = list(adata.obs_names[~mask])
                    removed_cells.extend(removed)
                    adata._inplace_subset_obs(mask)
                else:
                    adata_remove = adata[adata.obs['predicted_doublet'], :]
                    removed_cells.extend(list(adata_remove.obs_names))
                    adata = adata[~adata.obs['predicted_doublet'], :]
                n1 = adata.shape[0]
                doublets_removed = n_after_final_filt-n1
                print(f"   {Colors.GREEN}✓ DoubletFinder completed: {Colors.BOLD}{doublets_removed:,}{Colors.ENDC}{Colors.GREEN} doublets removed ({doublets_removed/n_after_final_filt*100:.1f}%){Colors.ENDC}")
            else:
                n1 = adata.shape[0]
                doublets_flagged = adata.obs['predicted_doublet'].sum()
                print(f"   {Colors.GREEN}✓ DoubletFinder completed: {Colors.BOLD}{doublets_flagged:,}{Colors.ENDC}{Colors.GREEN} doublets flagged ({doublets_flagged/n_after_final_filt*100:.1f}%){Colors.ENDC}")
                print(f"   {Colors.CYAN}💡 Doublets retained in adata.obs['predicted_doublet']; pANN in adata.obs['doublet_score']{Colors.ENDC}")

        elif doublets_method=='sccomposite':
            # Pick the object sccomposite runs on: OOM uses the materialised
            # adata_mem and copies labels back; regular AnnData uses adata.
            if is_oom:
                sccomp_target = adata_mem
            else:
                sccomp_target = adata
            print(f"   {Colors.WARNING}⚠️  Note: the `sccomposite` will remove more cells than `scrublet`{Colors.ENDC}")
            print(f"   {Colors.GREEN}{EMOJI['start']} Running sccomposite doublet detection...{Colors.ENDC}")
            sccomp_target.obs['sccomposite_doublet']=0
            sccomp_target.obs['sccomposite_consistency']=0
            if batch_key is None:
                from ._sccomposite import composite_rna
                multiplet_classification, consistency = composite_rna(sccomp_target)
                sccomp_target.obs['sccomposite_doublet']=multiplet_classification
                sccomp_target.obs['sccomposite_consistency']=consistency
            else:
                for batch in sccomp_target.obs[batch_key].unique():
                    from ._sccomposite import composite_rna
                    adata_batch=sccomp_target[sccomp_target.obs[batch_key]==batch]
                    multiplet_classification, consistency = composite_rna(adata_batch)
                    sccomp_target.obs.loc[adata_batch.obs.index,'sccomposite_doublet']=\
                    multiplet_classification
                    sccomp_target.obs.loc[adata_batch.obs.index,'sccomposite_consistency']=consistency

            if is_oom:
                adata.obs['sccomposite_doublet']=sccomp_target.obs['sccomposite_doublet'].values
                adata.obs['sccomposite_consistency']=sccomp_target.obs['sccomposite_consistency'].values
                del adata_mem

            if filter_doublets:
                if is_oom:
                    mask = (adata.obs['sccomposite_doublet']==0).values
                    removed_cells.extend(list(adata.obs_names[~mask]))
                    adata._inplace_subset_obs(mask)
                else:
                    adata_remove = adata[adata.obs['sccomposite_doublet']!=0, :]
                    removed_cells.extend(list(adata_remove.obs_names))
                    adata = adata[adata.obs['sccomposite_doublet']==0, :]
                n1 = adata.shape[0]
                doublets_removed = n_after_final_filt-n1
                print(f"   {Colors.GREEN}✓ sccomposite completed: {Colors.BOLD}{doublets_removed:,}{Colors.ENDC}{Colors.GREEN} doublets removed ({doublets_removed/n_after_final_filt*100:.1f}%){Colors.ENDC}")
            else:
                n1 = adata.shape[0]
                doublets_flagged = (adata.obs['sccomposite_doublet']!=0).sum()
                print(f"   {Colors.GREEN}✓ sccomposite completed: {Colors.BOLD}{doublets_flagged:,}{Colors.ENDC}{Colors.GREEN} doublets flagged ({doublets_flagged/n_after_final_filt*100:.1f}%){Colors.ENDC}")
                print(f"   {Colors.CYAN}💡 Doublets retained in adata.obs['sccomposite_doublet'] for downstream analysis{Colors.ENDC}")

        elif doublets_method=='scdblfinder':
            from ._scdblfinder import scdblfinder as _run_scdbl
            print(f"   {Colors.CYAN}💡 Running pyscdblfinder (Python port of R scDblFinder){Colors.ENDC}")
            print(f"   {Colors.GREEN}{EMOJI['start']} Running scdblfinder detection...{Colors.ENDC}")
            if is_oom:
                _run_scdbl(adata_mem, batch_key=batch_key, random_state=1234)
                adata.obs['predicted_doublet'] = adata_mem.obs['predicted_doublet'].values
                adata.obs['doublet_score']     = adata_mem.obs['doublet_score'].values
                del adata_mem
            else:
                _run_scdbl(adata, batch_key=batch_key, random_state=1234)
            if filter_doublets:
                if is_oom:
                    mask = ~adata.obs['predicted_doublet'].values
                    removed_cells.extend(list(adata.obs_names[~mask]))
                    adata._inplace_subset_obs(mask)
                else:
                    adata_remove = adata[adata.obs['predicted_doublet'], :]
                    removed_cells.extend(list(adata_remove.obs_names))
                    adata = adata[~adata.obs['predicted_doublet'], :]
                n1 = adata.shape[0]
                doublets_removed = n_after_final_filt-n1
                print(f"   {Colors.GREEN}✓ scDblFinder completed: {Colors.BOLD}{doublets_removed:,}{Colors.ENDC}{Colors.GREEN} doublets removed ({doublets_removed/n_after_final_filt*100:.1f}%){Colors.ENDC}")
            else:
                n1 = adata.shape[0]
                doublets_flagged = adata.obs['predicted_doublet'].sum()
                print(f"   {Colors.GREEN}✓ scDblFinder completed: {Colors.BOLD}{doublets_flagged:,}{Colors.ENDC}{Colors.GREEN} doublets flagged ({doublets_flagged/n_after_final_filt*100:.1f}%){Colors.ENDC}")
                print(f"   {Colors.CYAN}💡 Doublets retained in adata.obs['predicted_doublet']; score in adata.obs['doublet_score']{Colors.ENDC}")
    else:
        print(f"\n{Colors.BLUE}📊 Step 4: Doublet detection disabled{Colors.ENDC}")
        n1 = adata.shape[0]

    if 'status' not in adata.uns.keys():
        adata.uns['status'] = {}
    if 'status_args' not in adata.uns.keys():
        adata.uns['status_args'] = {}

    adata.uns['status']['qc']=True
    adata.uns['status_args']['qc'] = {
        'mode': mode,
        'min_cells': min_cells,
        'min_genes': min_genes,
        'nmads': nmads,
        'max_cells_ratio': max_cells_ratio,
        'max_genes_ratio': max_genes_ratio,
        'batch_key': batch_key,
        'doublets': doublets,
        'doublets_method': doublets_method,
        'path_viz': path_viz,
        'mito_perc': tresh['mito_perc'],
        'nUMIs': tresh['nUMIs'],
        'detected_genes': tresh['detected_genes'],
    }
    add_reference(adata,'qc','QC with scanpy')
    return adata


def qc_gpu(adata, mode='seurat',
       min_cells=3, min_genes=200, nmads=5,
       max_cells_ratio=1,max_genes_ratio=1,
       batch_key=None,doublets=True,doublets_method='scrublet',
       filter_doublets=True,
       path_viz=None, tresh=None,mt_startswith='auto',mt_genes=None,
       ribo_startswith=("RPS", "RPL"),ribo_genes=None,
       hb_startswith="^HB[^(P)]",hb_genes=None):
    '''
    GPU-accelerated quality control using RAPIDS

    Doublet detection is performed after QC threshold filtering and final cell/gene filtering.
    '''
    import rapids_singlecell as rsc
    
    # Logging
    if tresh is None:
        tresh={'mito_perc': 0.15, 'nUMIs': 500, 'detected_genes': 250}
    
    removed_cells = []
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}{EMOJI['start']} Quality Control Analysis (GPU-Accelerated):{Colors.ENDC}")
    print(f"   {Colors.CYAN}Dataset shape: {Colors.BOLD}{adata.shape[0]:,} cells × {adata.shape[1]:,} genes{Colors.ENDC}")
    print(f"   {Colors.BLUE}QC mode: {Colors.BOLD}{mode}{Colors.ENDC}")
    print(f"   {Colors.BLUE}Doublet detection: {Colors.BOLD}{doublets_method if doublets else 'disabled'}{Colors.ENDC}")
    # Auto-detect mitochondrial gene prefix
    if mt_startswith == 'auto' and mt_genes is None:
        mt_startswith = _detect_mt_prefix(adata.var_names)
        print(f"   {Colors.CYAN}Auto-detected mitochondrial prefix: {Colors.BOLD}'{mt_startswith}'{Colors.ENDC}")
    print(f"   {Colors.BLUE}Mitochondrial genes: {Colors.BOLD}{mt_startswith if mt_genes is None else 'custom list'}{Colors.ENDC}")

    print(f"   {Colors.GREEN}{EMOJI['gpu']} Loading data to GPU...{Colors.ENDC}")
    rsc.get.anndata_to_GPU(adata)

    # QC metrics
    print(f"\n{Colors.HEADER}{Colors.BOLD}📊 Step 1: Calculating QC Metrics{Colors.ENDC}")
    adata.var_names_make_unique()
    if mt_genes is not None:
        adata.var['mt']=False
        adata.var.loc[list(set(adata.var_names) & set(mt_genes)),'mt']=True
        mt_genes_found = sum(adata.var['mt'])
    # print(f"   {Colors.CYAN}Custom mitochondrial genes: {Colors.BOLD}{mt_genes_found}/{len(mt_genes)}{Colors.ENDC}{Colors.CYAN} found{Colors.ENDC}")
    else:
        if mt_startswith in _CE_MT_PREFIXES:
            # C. elegans: multiple prefixes, flag manually
            adata.var["mt"] = _mt_mask(adata.var_names, mt_startswith)
        else:
            rsc.pp.flag_gene_family(adata, gene_family_name="mt", gene_family_prefix=mt_startswith)
        mt_genes_found = sum(adata.var["mt"])
    # print(f"   {Colors.CYAN}Mitochondrial genes (prefix '{mt_startswith}'): {Colors.BOLD}{mt_genes_found}{Colors.ENDC}{Colors.CYAN} found{Colors.ENDC}")

    if ribo_genes is not None:
        adata.var["ribo"] = False
        adata.var.loc[list(set(adata.var_names) & set(ribo_genes)),'ribo']=True
        ribo_genes_found = sum(adata.var["ribo"])
    # print(f"   {Colors.CYAN}Ribosomal genes: {Colors.BOLD}{ribo_genes_found}/{len(ribo_genes)}{Colors.ENDC}{Colors.CYAN} found{Colors.ENDC}")
    else:
        # For tuple of prefixes, we need to flag each prefix separately
        adata.var["ribo"] = False
        for prefix in ribo_startswith:
            rsc.pp.flag_gene_family(adata, gene_family_name="ribo_temp", gene_family_prefix=prefix)
            adata.var["ribo"] = adata.var["ribo"] | adata.var["ribo_temp"]
        adata.var.drop(columns=["ribo_temp"], inplace=True, errors='ignore')
        ribo_genes_found = sum(adata.var["ribo"])
    # print(f"   {Colors.CYAN}Ribosomal genes (prefix '{ribo_startswith}'): {Colors.BOLD}{ribo_genes_found}{Colors.ENDC}{Colors.CYAN} found{Colors.ENDC}")

    if hb_genes is not None:
        adata.var["hb"] = False
        adata.var.loc[list(set(adata.var_names) & set(hb_genes)),'hb']=True
        hb_genes_found = sum(adata.var["hb"])
    else:
        # Use regex pattern for hemoglobin genes
        if type(adata.var_names) is list:
            var_names = pd.Index(adata.var_names)
        else:
            var_names = adata.var_names
        adata.var["hb"] = var_names.str.contains(hb_startswith)
        hb_genes_found = sum(adata.var["hb"])

    # Print gene detection table
    _print_gene_detection_table(mt_genes_found, ribo_genes_found, hb_genes_found, mt_genes, ribo_genes, hb_genes)

    rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"])
    adata.obs['nUMIs'] = adata.obs['total_counts']
    adata.obs['mito_perc'] = adata.obs['pct_counts_mt']/100
    adata.obs['ribo_perc'] = adata.obs['pct_counts_ribo']/100
    adata.obs['hb_perc'] = adata.obs['pct_counts_hb']/100
    adata.obs['detected_genes'] = adata.obs['n_genes_by_counts']
    adata.obs['cell_complexity'] = adata.obs['detected_genes'] / adata.obs['nUMIs']

    # Scanpy-convention aliases — let LLM-generated code that follows scanpy
    # tutorials (`adata.obs['pct_counts_mt']`, `adata.obs['n_counts']`,
    # `adata.obs['total_counts']`, `adata.obs['n_genes']`) work without
    # KeyError. ov uses 'mito_perc'/'nUMIs'/'detected_genes' internally;
    # we mirror the values under standard scanpy names so both call styles
    # succeed. Cheap (no data duplication, just dict assignment).
    adata.obs['n_counts']        = adata.obs['nUMIs']
    adata.obs['total_counts']    = adata.obs['nUMIs']
    adata.obs['n_genes']         = adata.obs['detected_genes']
    adata.obs['n_genes_by_counts'] = adata.obs['detected_genes']
    adata.obs['pct_counts_mt']   = adata.obs['mito_perc']
    if 'ribo_perc' in adata.obs.columns:
        adata.obs['pct_counts_ribo'] = adata.obs['ribo_perc']
    if 'hb_perc' in adata.obs.columns:
        adata.obs['pct_counts_hb']   = adata.obs['hb_perc']

    # Display QC statistics in table format
    _print_qc_metrics_table(adata)

    # Original QC plot
    n0 = adata.shape[0]
    print(f"\n   {Colors.CYAN}📈 Original cell count: {Colors.BOLD}{n0:,}{Colors.ENDC}")

    # Filters
    print(f"\n{Colors.HEADER}{Colors.BOLD}🔧 Step 2: Quality Filtering ({mode.upper()}){Colors.ENDC}")
    print(f"   {Colors.CYAN}Thresholds: mito≤{tresh['mito_perc']}, nUMIs≥{tresh['nUMIs']}, genes≥{tresh['detected_genes']}{Colors.ENDC}")
    
    if mode == 'seurat':
        adata.obs['passing_mt'] = adata.obs['mito_perc'] < tresh['mito_perc']
        adata.obs['passing_nUMIs'] = adata.obs['nUMIs'] > tresh['nUMIs']
        adata.obs['passing_ngenes'] = adata.obs['detected_genes'] > tresh['detected_genes']
    elif mode == 'mads':
        adata.obs['passing_mt'] = adata.obs['mito_perc'] < tresh['mito_perc']
        adata.obs['passing_nUMIs'] = mads_test(adata.obs, 'nUMIs', nmads=nmads, lt=tresh, batch_key=batch_key)
        adata.obs['passing_ngenes'] = mads_test(adata.obs, 'detected_genes', nmads=nmads, lt=tresh, batch_key=batch_key)
    
    # Report
    n1 = adata.shape[0]
    mt_failed = n1-np.sum(adata.obs["passing_mt"])
    umis_failed = n1-np.sum(adata.obs["passing_nUMIs"])
    genes_failed = n1-np.sum(adata.obs["passing_ngenes"])
    
    if mode == 'seurat':
        print(f"   {Colors.BLUE}📊 Seurat Filter Results:{Colors.ENDC}")
        print(f"     {Colors.CYAN}• nUMIs filter (≥{tresh['nUMIs']}): {Colors.BOLD}{umis_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed ({umis_failed/n1*100:.1f}%){Colors.ENDC}")
        print(f"     {Colors.CYAN}• Genes filter (≥{tresh['detected_genes']}): {Colors.BOLD}{genes_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed ({genes_failed/n1*100:.1f}%){Colors.ENDC}")
        print(f"     {Colors.CYAN}• Mitochondrial filter (≤{tresh['mito_perc']}): {Colors.BOLD}{mt_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed ({mt_failed/n1*100:.1f}%){Colors.ENDC}")
    elif mode == 'mads':
        nUMIs_t = mads(adata.obs, 'nUMIs', nmads=nmads, lt=tresh)
        n_genes_t = mads(adata.obs, 'detected_genes', nmads=nmads, lt=tresh)
        print(f"   {Colors.BLUE}📊 MADs Filter Results (±{nmads} MADs):{Colors.ENDC}")
        print(f"     {Colors.CYAN}• nUMIs range ({nUMIs_t[0]:.0f}, {nUMIs_t[1]:.0f}): {Colors.BOLD}{umis_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed{Colors.ENDC}")
        print(f"     {Colors.CYAN}• Genes range ({n_genes_t[0]:.0f}, {n_genes_t[1]:.0f}): {Colors.BOLD}{genes_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed{Colors.ENDC}")
        print(f"     {Colors.CYAN}• Mitochondrial filter (≤{tresh['mito_perc']}): {Colors.BOLD}{mt_failed:,}{Colors.ENDC}{Colors.CYAN} cells failed{Colors.ENDC}")

    # QC plot
    QC_test = (adata.obs['passing_mt']) & (adata.obs['passing_nUMIs']) & (adata.obs['passing_ngenes'])
    removed = QC_test.loc[~QC_test.values]
    removed_cells.extend(list(removed.index.values))
    total_qc_failed = n1-np.sum(QC_test)
    adata = adata[QC_test, :]
    n2 = adata.shape[0]
    
    print(f"   {Colors.GREEN}✓ Combined QC filters: {Colors.BOLD}{total_qc_failed:,}{Colors.ENDC}{Colors.GREEN} cells removed ({total_qc_failed/n1*100:.1f}%){Colors.ENDC}")

    # Last gene and cell filter
    print(f"\n{Colors.HEADER}{Colors.BOLD}🎯 Step 3: Final Filtering{Colors.ENDC}")
    print(f"   {Colors.CYAN}Parameters: min_genes={min_genes}, min_cells={min_cells}{Colors.ENDC}")
    print(f"   {Colors.CYAN}Ratios: max_genes_ratio={max_genes_ratio}, max_cells_ratio={max_cells_ratio}{Colors.ENDC}")
    
    cells_before_final = adata.shape[0]
    genes_before_final = adata.shape[1]
    
    # NB: filter on *cells-per-gene* / *genes-per-cell*, not total UMI counts.
    # Using min_counts/max_counts here (the old bug, #862) dropped highly
    # expressed marker genes — e.g. with max_cells_ratio=1 any gene whose total
    # UMIs exceed n_cells (S100A8, LYZ, FCGR3B, ...) was removed. rapids-singlecell
    # exposes min_genes/max_genes/min_cells/max_cells separately, matching the CPU
    # path's _filter_cells_impl / _filter_genes_impl.
    rsc.pp.filter_cells(adata, min_genes=min_genes)
    rsc.pp.filter_cells(adata, max_genes=int(max_genes_ratio*adata.shape[1]))
    rsc.pp.filter_genes(adata, min_cells=min_cells)
    rsc.pp.filter_genes(adata, max_cells=int(max_cells_ratio*adata.shape[0]))
    
    cells_final_filtered = cells_before_final - adata.shape[0]
    genes_final_filtered = genes_before_final - adata.shape[1]
    
    print(f"   {Colors.GREEN}✓ Final filtering: {Colors.BOLD}{cells_final_filtered:,}{Colors.ENDC}{Colors.GREEN} cells, {Colors.BOLD}{genes_final_filtered:,}{Colors.ENDC}{Colors.GREEN} genes removed{Colors.ENDC}")
    
    n_after_final_filt = adata.shape[0]
    
    if doublets is True:
        print(f"\n{Colors.HEADER}{Colors.BOLD}🔍 Step 4: Doublet Detection{Colors.ENDC}")
        doublets_method = _resolve_doublets_method(doublets_method)
        if doublets_method not in ('scrublet', 'sccomposite', 'doubletfinder', 'scdblfinder'):
            raise ValueError(
                f"Unknown doublets_method={doublets_method!r}; "
                "expected 'scrublet', 'sccomposite', 'doubletfinder', or 'scdblfinder'."
            )
        # scrublet runs on the GPU (rapids-singlecell); the other backends are
        # CPU/NumPy tools (pyscdblfinder / pydoubletfinder / sccomposite) that
        # cannot consume the cupy matrix qc_gpu put on the GPU — feeding it in
        # flips the orientation and returns one value per *gene*, causing
        # "Length of values (n_genes) does not match length of index (n_cells)"
        # (issue #866). Move the matrix back to host memory for those.
        if doublets_method != 'scrublet':
            rsc.get.anndata_to_CPU(adata)
        if doublets_method=='scrublet':
            print(f"   {Colors.GREEN}{EMOJI['start']} Running GPU-accelerated scrublet...{Colors.ENDC}")
            rsc.pp.scrublet(adata, random_state=1234,batch_key=batch_key)

            if filter_doublets:
                adata_remove = adata[adata.obs['predicted_doublet'], :]
                removed_cells.extend(list(adata_remove.obs_names))
                adata = adata[~adata.obs['predicted_doublet'], :]
                n1 = adata.shape[0]
                doublets_removed = n_after_final_filt-n1
                print(f"   {Colors.GREEN}✓ Scrublet completed: {Colors.BOLD}{doublets_removed:,}{Colors.ENDC}{Colors.GREEN} doublets removed ({doublets_removed/n_after_final_filt*100:.1f}%){Colors.ENDC}")
            else:
                n1 = adata.shape[0]
                doublets_flagged = adata.obs['predicted_doublet'].sum()
                print(f"   {Colors.GREEN}✓ Scrublet completed: {Colors.BOLD}{doublets_flagged:,}{Colors.ENDC}{Colors.GREEN} doublets flagged ({doublets_flagged/n_after_final_filt*100:.1f}%){Colors.ENDC}")
                print(f"   {Colors.CYAN}💡 Doublets retained in adata.obs['predicted_doublet'] for downstream analysis{Colors.ENDC}")

        elif doublets_method=='doubletfinder':
            from ._doubletfinder import doubletfinder as _run_df
            print(f"   {Colors.CYAN}💡 Running py-DoubletFinder (Python port of R DoubletFinder){Colors.ENDC}")
            print(f"   {Colors.GREEN}{EMOJI['start']} Running doubletfinder detection...{Colors.ENDC}")
            _run_df(adata, batch_key=batch_key, random_state=1234)
            if filter_doublets:
                adata_remove = adata[adata.obs['predicted_doublet'], :]
                removed_cells.extend(list(adata_remove.obs_names))
                adata = adata[~adata.obs['predicted_doublet'], :]
                n1 = adata.shape[0]
                doublets_removed = n_after_final_filt-n1
                print(f"   {Colors.GREEN}✓ DoubletFinder completed: {Colors.BOLD}{doublets_removed:,}{Colors.ENDC}{Colors.GREEN} doublets removed ({doublets_removed/n_after_final_filt*100:.1f}%){Colors.ENDC}")
            else:
                n1 = adata.shape[0]
                doublets_flagged = adata.obs['predicted_doublet'].sum()
                print(f"   {Colors.GREEN}✓ DoubletFinder completed: {Colors.BOLD}{doublets_flagged:,}{Colors.ENDC}{Colors.GREEN} doublets flagged ({doublets_flagged/n_after_final_filt*100:.1f}%){Colors.ENDC}")
                print(f"   {Colors.CYAN}💡 Doublets retained in adata.obs['predicted_doublet']; pANN in adata.obs['doublet_score']{Colors.ENDC}")

        elif doublets_method=='sccomposite':
            print(f"   {Colors.GREEN}{EMOJI['start']} Running sccomposite doublet detection...{Colors.ENDC}")
            adata.obs['sccomposite_doublet']=0
            adata.obs['sccomposite_consistency']=0
            if batch_key is None:
                from ._sccomposite import composite_rna
                multiplet_classification, consistency = composite_rna(adata)
                adata.obs['sccomposite_doublet']=multiplet_classification
                adata.obs['sccomposite_consistency']=consistency
            else:
                print(f"   {Colors.CYAN}Processing {len(adata.obs[batch_key].unique())} batches separately...{Colors.ENDC}")
                for batch in adata.obs[batch_key].unique():
                    from ._sccomposite import composite_rna
                    adata_batch=adata[adata.obs[batch_key]==batch]
                    multiplet_classification, consistency = composite_rna(adata_batch)
                    adata.obs.loc[adata_batch.obs.index,'sccomposite_doublet']=\
                    multiplet_classification
                    adata.obs.loc[adata_batch.obs.index,'sccomposite_consistency']=consistency

            if filter_doublets:
                adata_remove = adata[adata.obs['sccomposite_doublet']!=0, :]
                removed_cells.extend(list(adata_remove.obs_names))
                adata = adata[adata.obs['sccomposite_doublet']==0, :]
                n1 = adata.shape[0]
                doublets_removed = n_after_final_filt-n1
                print(f"   {Colors.GREEN}✓ sccomposite completed: {Colors.BOLD}{doublets_removed:,}{Colors.ENDC}{Colors.GREEN} doublets removed ({doublets_removed/n_after_final_filt*100:.1f}%){Colors.ENDC}")
            else:
                n1 = adata.shape[0]
                doublets_flagged = (adata.obs['sccomposite_doublet']!=0).sum()
                print(f"   {Colors.GREEN}✓ sccomposite completed: {Colors.BOLD}{doublets_flagged:,}{Colors.ENDC}{Colors.GREEN} doublets flagged ({doublets_flagged/n_after_final_filt*100:.1f}%){Colors.ENDC}")
                print(f"   {Colors.CYAN}💡 Doublets retained in adata.obs['sccomposite_doublet'] for downstream analysis{Colors.ENDC}")

        elif doublets_method=='scdblfinder':
            from ._scdblfinder import scdblfinder as _run_scdbl
            print(f"   {Colors.CYAN}💡 Running pyscdblfinder (Python port of R scDblFinder){Colors.ENDC}")
            print(f"   {Colors.GREEN}{EMOJI['start']} Running scdblfinder detection...{Colors.ENDC}")
            _run_scdbl(adata, batch_key=batch_key, random_state=1234)
            if filter_doublets:
                adata_remove = adata[adata.obs['predicted_doublet'], :]
                removed_cells.extend(list(adata_remove.obs_names))
                adata = adata[~adata.obs['predicted_doublet'], :]
                n1 = adata.shape[0]
                doublets_removed = n_after_final_filt-n1
                print(f"   {Colors.GREEN}✓ scDblFinder completed: {Colors.BOLD}{doublets_removed:,}{Colors.ENDC}{Colors.GREEN} doublets removed ({doublets_removed/n_after_final_filt*100:.1f}%){Colors.ENDC}")
            else:
                n1 = adata.shape[0]
                doublets_flagged = adata.obs['predicted_doublet'].sum()
                print(f"   {Colors.GREEN}✓ scDblFinder completed: {Colors.BOLD}{doublets_flagged:,}{Colors.ENDC}{Colors.GREEN} doublets flagged ({doublets_flagged/n_after_final_filt*100:.1f}%){Colors.ENDC}")
                print(f"   {Colors.CYAN}💡 Doublets retained in adata.obs['predicted_doublet']; score in adata.obs['doublet_score']{Colors.ENDC}")

    # Store status
    if 'status' not in adata.uns.keys():
        adata.uns['status'] = {}
    adata.uns['status']['qc']=True

    # Final summary
    print(f"\n{Colors.GREEN}{EMOJI['done']} GPU Quality Control Analysis Completed!{Colors.ENDC}")
    print(f"\n{Colors.HEADER}{Colors.BOLD}📈 Final Summary:{Colors.ENDC}")
    print(f"   {Colors.CYAN}📊 Original: {Colors.BOLD}{n0:,}{Colors.ENDC}{Colors.CYAN} cells × {Colors.BOLD}{genes_before_final:,}{Colors.ENDC}{Colors.CYAN} genes{Colors.ENDC}")
    print(f"   {Colors.GREEN}✓ Final: {Colors.BOLD}{adata.shape[0]:,}{Colors.ENDC}{Colors.GREEN} cells × {Colors.BOLD}{adata.shape[1]:,}{Colors.ENDC}{Colors.GREEN} genes{Colors.ENDC}")
    print(f"   {Colors.BLUE}📉 Total removed: {Colors.BOLD}{n0-adata.shape[0]:,}{Colors.ENDC}{Colors.BLUE} cells ({(n0-adata.shape[0])/n0*100:.1f}%){Colors.ENDC}")
    print(f"   {Colors.BLUE}📉 Total removed: {Colors.BOLD}{genes_before_final-adata.shape[1]:,}{Colors.ENDC}{Colors.BLUE} genes ({(genes_before_final-adata.shape[1])/genes_before_final*100:.1f}%){Colors.ENDC}")
    
    # Quality assessment
    final_retention_rate = adata.shape[0] / n0
    if final_retention_rate >= 0.8:
        quality_color = Colors.GREEN
        quality_msg = "Excellent retention rate"
    elif final_retention_rate >= 0.6:
        quality_color = Colors.CYAN
        quality_msg = "Good retention rate"
    else:
        quality_color = Colors.WARNING
        quality_msg = "Low retention rate - consider relaxing thresholds"
    
    print(f"   {quality_color}💯 Quality: {Colors.BOLD}{final_retention_rate*100:.1f}%{Colors.ENDC}{quality_color} retention ({quality_msg}){Colors.ENDC}")
    print(f"\n{Colors.CYAN}{'─' * 60}{Colors.ENDC}")
    
    return adata

@monitor
@register_function(
    aliases=["细胞过滤", "filter_cells", "cell_filter", "低质量细胞过滤", "细胞质控过滤"],
    category="preprocessing",
    description="Filter out low-quality cells by gene/UMI count thresholds",
    prerequisites={
        "optional_functions": ["qc"]
    },
    requires={},
    produces={
        "obs": ["n_genes_by_counts", "total_counts"]
    },
    auto_fix="none",
    examples=[
        "ov.pp.filter_cells(adata, min_genes=200)",
        "ov.pp.filter_cells(adata, max_counts=50000)",
    ],
    related=["qc", "filter_genes", "preprocess"],
)
def filter_cells(adata: anndata.AnnData,
    min_counts: Optional[int] = None,
    min_genes: Optional[int] = None,
    max_counts: Optional[int] = None,
    max_genes: Optional[int] = None,
    inplace: bool = True,):
    """
    Filter cell outliers based on counts and numbers of genes expressed.
    
    For instance, only keep cells with at least `min_counts` counts or
    `min_genes` genes expressed. This is to filter measurement outliers,
    i.e. "unreliable" observations.
    
    Only provide one of the optional parameters `min_counts`, `min_genes`,
    `max_counts`, `max_genes` per call.
    
    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    min_counts
        Minimum number of counts required for a cell to pass filtering.
    min_genes
        Minimum number of genes expressed required for a cell to pass filtering.
    max_counts
        Maximum number of counts required for a cell to pass filtering.
    max_genes
        Maximum number of genes expressed required for a cell to pass filtering.
    inplace
        Perform computation inplace or return result.
    
    Returns
    -------
    Depending on `inplace`, returns the following arrays or directly subsets
    and annotates the data matrix:
    
    cells_subset
        Boolean index mask that does filtering. `True` means that the
        cell is kept. `False` means the cell is removed.
    number_per_cell
        Depending on what was tresholded (`counts` or `genes`),
        the array stores `n_counts` or `n_cells` per gene.
    
    """
    print(f"{Colors.GREEN}{EMOJI['start']} Filtering cells...{Colors.ENDC}")
    filter_params = []
    if min_counts is not None:
        filter_params.append(f"min_counts≥{min_counts}")
    if min_genes is not None:
        filter_params.append(f"min_genes≥{min_genes}")
    if max_counts is not None:
        filter_params.append(f"max_counts≤{max_counts}")
    if max_genes is not None:
        filter_params.append(f"max_genes≤{max_genes}")
    
    if filter_params:
        print(f"   {Colors.CYAN}Parameters: {', '.join(filter_params)}{Colors.ENDC}")
    
    cells_before = adata.shape[0]
    _filter_cells_impl(adata, min_genes=min_genes, min_counts=min_counts, max_counts=max_counts,
                       max_genes=max_genes, inplace=inplace)
    cells_filtered = cells_before - adata.shape[0]
    print(f"   {Colors.GREEN}✓ Filtered: {Colors.BOLD}{cells_filtered:,}{Colors.ENDC}{Colors.GREEN} cells removed{Colors.ENDC}")

@monitor
@register_function(
    aliases=["基因过滤", "filter_genes", "gene_filter", "低表达基因过滤", "基因质控过滤"],
    category="preprocessing",
    description="Filter genes by minimum/maximum detected cells or counts",
    prerequisites={
        "optional_functions": ["qc"]
    },
    requires={},
    produces={
        "var": ["n_cells_by_counts", "total_counts"]
    },
    auto_fix="none",
    examples=[
        "ov.pp.filter_genes(adata, min_cells=3)",
        "ov.pp.filter_genes(adata, min_counts=10)",
    ],
    related=["qc", "filter_cells", "highly_variable_genes"],
)
def filter_genes(adata: anndata.AnnData,
    min_counts: Optional[int] = None,
    min_cells: Optional[int] = None,
    max_counts: Optional[int] = None,
    max_cells: Optional[int] = None,
    inplace: bool = True,):
    """
    Filter genes based on number of cells or counts.
    
    Keep genes that have at least `min_counts` counts or are expressed in at
    least `min_cells` cells or have at most `max_counts` counts or are expressed
    in at most `max_cells` cells.
    
    Only provide one of the optional parameters `min_counts`, `min_cells`,
    `max_counts`, `max_cells` per call.
    
    Parameters
    ----------
    adata
        An annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    min_counts
        Minimum number of counts required for a gene to pass filtering.
    min_cells
        Minimum number of cells expressed required for a gene to pass filtering.
    max_counts
        Maximum number of counts required for a gene to pass filtering.
    max_cells
        Maximum number of cells expressed required for a gene to pass filtering.
    inplace
        Perform computation inplace or return result.
    
    Returns
    -------
    Depending on `inplace`, returns the following arrays or directly subsets
    and annotates the data matrix
    
    gene_subset
        Boolean index mask that does filtering. `True` means that the
        gene is kept. `False` means the gene is removed.
    number_per_gene
        Depending on what was tresholded (`counts` or `cells`), the array stores
        `n_counts` or `n_cells` per gene.
    
    """
    print(f"{Colors.GREEN}{EMOJI['start']} Filtering genes...{Colors.ENDC}")
    filter_params = []
    if min_counts is not None:
        filter_params.append(f"min_counts≥{min_counts}")
    if min_cells is not None:
        filter_params.append(f"min_cells≥{min_cells}")
    if max_counts is not None:
        filter_params.append(f"max_counts≤{max_counts}")
    if max_cells is not None:
        filter_params.append(f"max_cells≤{max_cells}")
    
    if filter_params:
        print(f"   {Colors.CYAN}Parameters: {', '.join(filter_params)}{Colors.ENDC}")
    
    genes_before = adata.shape[1]
    _filter_genes_impl(adata, min_counts=min_counts, min_cells=min_cells, max_counts=max_counts,
                       max_cells=max_cells, inplace=inplace)
    genes_filtered = genes_before - adata.shape[1]
    print(f"   {Colors.GREEN}✓ Filtered: {Colors.BOLD}{genes_filtered:,}{Colors.ENDC}{Colors.GREEN} genes removed{Colors.ENDC}")
    

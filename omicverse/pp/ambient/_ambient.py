"""Unified ambient / contamination-RNA removal for droplet scRNA-seq.

``ov.pp.ambient`` threads the major ambient-RNA decontamination methods
behind one registered, ``method=``-style dispatcher.  Ambient (cell-free,
"soup") RNA leaks into every droplet during library prep; left in place it
inflates marker genes in cell types that never expressed them and corrupts
downstream DE / annotation.

Methods
-------
Four **native, pure-Python R-parity backends** ship as separate releases
and need no heavyweight dependency:

* ``'soupx'``    — :mod:`pysoupx`   (Young & Behjati 2020). Needs the raw
  unfiltered matrix (empty droplets) to build the soup profile.
* ``'decontx'``  — :mod:`pydecontx` (Yang *et al.* 2020). Needs a filtered,
  *clustered* matrix; empty droplets optional.
* ``'fastcar'``  — :mod:`pyfastcar` (Berg *et al.* 2023). Needs the raw
  unfiltered matrix; deterministic per-gene subtraction.
* ``'sccdc'``    — :mod:`pysccdc`   (Wang *et al.* 2024). Needs a filtered,
  *clustered* matrix; corrects only Global-Contamination-causing Genes.

Two **optional deep-learning wrappers** (heavyweight, not installed by the
``omicverse[ambient]`` extra) are exposed via the ``_require`` pattern:

* ``'cellbender'`` — CellBender ``remove-background`` (Fleming *et al.*).
* ``'scar'``       — scAR ambient-RNA denoising (Sheng *et al.*).

Both raise a clean, actionable :class:`ImportError` when their package is
absent.
"""
from __future__ import annotations

import importlib
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..._registry import register_function
from ._envelope import run_envelope as _run_envelope
from ._diagnostics import count_integrity_check

_NATIVE_METHODS = ("soupx", "decontx", "fastcar", "sccdc", "envelope")
_DL_METHODS = ("cellbender", "scar")
_ALL_METHODS = _NATIVE_METHODS + _DL_METHODS

# SoupX rho used when neither cluster labels nor contamination_fraction exist
_SOUPX_FALLBACK_RHO = 0.1


# ---------------------------------------------------------------------------
# optional-dependency helper
# ---------------------------------------------------------------------------
def _require(modname: str, role: str, extra: str = "ambient"):
    """Lazy-import a backend with an actionable error message.

    Native backends are listed in ``omicverse[ambient]``; the deep-learning
    backends (CellBender, scAR) are heavyweight and must be installed
    directly.
    """
    try:
        return importlib.import_module(modname)
    except ImportError as exc:  # pragma: no cover - exercised in smoke test
        if modname in ("cellbender", "scar"):
            hint = (
                f"pip install {modname}   "
                f"({modname} is a heavyweight deep-learning tool and is "
                "NOT part of omicverse[ambient])")
        else:
            hint = (
                f"pip install omicverse[{extra}]   "
                f"(or pip install {modname})")
        raise ImportError(
            f"{role} needs the '{modname}' backend. Install with: {hint}."
        ) from exc


# ---------------------------------------------------------------------------
# small AnnData utilities
# ---------------------------------------------------------------------------
def _get_X(adata, layer: Optional[str]):
    return adata.layers[layer] if layer is not None else adata.X


def _set_X(adata, value, layer: Optional[str]):
    if layer is not None:
        adata.layers[layer] = value
    else:
        adata.X = value


def _as_dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def _removed_fraction(raw_x, corrected_x):
    """Per-cell share of counts removed by a correction (rows = cells).

    Works on sparse matrices without densifying them."""
    before = np.asarray(raw_x.sum(axis=1), dtype=np.float64).ravel()
    after = np.asarray(corrected_x.sum(axis=1), dtype=np.float64).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(before > 0, (before - after) / before, 0.0)


_CLUSTER_KEY_CANDIDATES = ("clusters", "leiden", "louvain", "cell_type",
                           "celltype")


def _find_cluster_key(adata, cluster_key: Optional[str]) -> Optional[str]:
    """Return ``cluster_key``, or the first candidate column in ``obs``.

    An explicit ``cluster_key`` is returned unchanged (even if it is not an
    ``obs`` column); ``None`` comes back only when no candidate is found.
    """
    if cluster_key is None:
        for cand in _CLUSTER_KEY_CANDIDATES:
            if cand in adata.obs:
                return cand
    return cluster_key


def _resolve_clusters(adata, cluster_key: Optional[str], method: str):
    """Return a cluster-label column name, validating it exists."""
    cluster_key = _find_cluster_key(adata, cluster_key)
    if cluster_key is None or cluster_key not in adata.obs:
        raise ValueError(
            f"method='{method}' needs a filtered, *clustered* AnnData. "
            "Cluster the cells first (e.g. ov.pp.leiden) and pass "
            "cluster_key=, or store labels in obs['clusters'].")
    return cluster_key


# ---------------------------------------------------------------------------
# native backend wrappers
# ---------------------------------------------------------------------------
def _run_soupx(adata, *, raw=None, layer=None, cluster_key=None,
               contamination_fraction=None, soup_range=(0, 100),
               round_to_int=False, verbose=False, **kwargs):
    """SoupX — needs the raw unfiltered matrix + empty droplets."""
    soupx = _require("pysoupx", "SoupX ambient removal")
    if raw is None:
        raise ValueError(
            "method='soupx' needs the raw, unfiltered droplet matrix "
            "(it estimates the soup profile from the empty droplets). "
            "Pass raw=<unfiltered AnnData>.")

    # Like DecontX / scCDC, pick up obs['clusters'/'leiden'/...] when no
    # cluster_key is given, so the documented call (raw= only) on a clustered
    # AnnData still gets SoupX's automatic rho estimate.
    requested_key = cluster_key
    cluster_key = _find_cluster_key(adata, cluster_key)
    has_clusters = cluster_key is not None and cluster_key in adata.obs
    if not has_clusters:
        cluster_key = None

    sc = soupx.SoupChannel.from_anndata(
        adata, raw=raw, cluster_key=cluster_key, soup_range=soup_range)

    if contamination_fraction is not None:
        soupx.set_contamination_fraction(sc, float(contamination_fraction))
    elif has_clusters:
        soupx.auto_est_cont(sc, verbose=verbose)
    else:
        # no clusters and no rho — fall back to a conservative default
        if requested_key is not None:
            why = (f"cluster_key={requested_key!r} is not a column of "
                   "adata.obs")
        else:
            why = ("no cluster labels found in adata.obs (looked for "
                   f"{', '.join(_CLUSTER_KEY_CANDIDATES)})")
        warnings.warn(
            f"method='soupx': {why}, so SoupX's automatic contamination "
            "estimate (auto_est_cont) was skipped and a fixed "
            f"contamination_fraction={_SOUPX_FALLBACK_RHO} is used for "
            "every cell. Pass cluster_key='<obs column>' (e.g. "
            "cluster_key='leiden') to estimate it automatically, or "
            "contamination_fraction=<value> to set it explicitly.",
            UserWarning,
            # user -> registry wrapper -> remove_ambient -> _run_soupx
            stacklevel=4,
        )
        soupx.set_contamination_fraction(sc, _SOUPX_FALLBACK_RHO)

    clusters = None if has_clusters else False
    corrected = soupx.adjust_counts(
        sc, clusters=clusters, round_to_int=round_to_int, verbose=int(verbose))

    # genes x cells -> cells x genes
    corrected_cg = sp.csr_matrix(corrected.T)
    rho = sc.meta_data["rho"].to_numpy(dtype=float)
    n_genes_corrected = int(adata.n_vars)
    meta = {
        "soup_range": soup_range,
        "contamination_fraction": float(np.mean(rho)),
        "auto_estimated": contamination_fraction is None and has_clusters,
        "cluster_key": cluster_key,
    }
    return corrected_cg, rho, n_genes_corrected, meta


def _run_decontx(adata, *, raw=None, layer=None, cluster_key=None,
                 batch_key=None, max_iter=500, seed=12345, n_jobs=1,
                 verbose=False, **kwargs):
    """DecontX — needs a filtered, clustered matrix."""
    decontx_mod = _require("pydecontx", "DecontX ambient removal")
    cluster_key = _resolve_clusters(adata, cluster_key, "decontx")
    z = adata.obs[cluster_key].astype(str).to_numpy()

    batch = None
    if batch_key is not None:
        if batch_key not in adata.obs:
            raise ValueError(
                f"method='decontx' batch_key='{batch_key}' not found in "
                "adata.obs.")
        batch = adata.obs[batch_key].astype(str).to_numpy()

    background = None
    if raw is not None:
        # genes x cells empty-droplet matrix anchors eta
        rw = raw[:, adata.var_names] if list(raw.var_names) != \
            list(adata.var_names) else raw
        background = sp.csc_matrix(rw.X).T

    res = decontx_mod.decontx(
        adata, z=z, batch=batch, background=background, max_iter=max_iter,
        seed=seed, layer=layer, verbose=verbose, n_jobs=n_jobs)
    # res.decontx_counts is genes x cells
    corrected_cg = sp.csr_matrix(res.decontx_counts.T)
    rho = np.asarray(res.contamination, dtype=float)
    n_genes_corrected = int(adata.n_vars)
    meta = {
        "cluster_key": cluster_key,
        "batch_key": batch_key,
        "max_iter": max_iter,
        "contamination_fraction": float(np.mean(rho)),
        "background_used": background is not None,
    }
    return corrected_cg, rho, n_genes_corrected, meta


def _run_fastcar(adata, *, raw=None, layer=None,
                 empty_droplet_cutoff=100,
                 contamination_chance_cutoff=0.005, **kwargs):
    """FastCAR — needs the raw unfiltered matrix + empty droplets."""
    fastcar = _require("pyfastcar", "FastCAR ambient removal")
    if raw is None:
        raise ValueError(
            "method='fastcar' needs the raw, unfiltered droplet matrix "
            "(it builds the ambient profile from the empty droplets). "
            "Pass raw=<unfiltered AnnData>.")
    rw = raw[:, adata.var_names] if list(raw.var_names) != \
        list(adata.var_names) else raw

    out = fastcar.correct_anndata(
        rw, cell_adata=adata,
        empty_droplet_cutoff=empty_droplet_cutoff,
        contamination_chance_cutoff=contamination_chance_cutoff,
        layer=layer, inplace=False)

    corrected_cg = sp.csr_matrix(_get_X(out, layer))
    rho = _removed_fraction(_get_X(adata, layer), corrected_cg)
    fc_info = out.uns.get("fastcar", {})
    n_genes_corrected = int(fc_info.get("n_genes_corrected", 0))
    meta = {
        "empty_droplet_cutoff": empty_droplet_cutoff,
        "contamination_chance_cutoff": contamination_chance_cutoff,
        "contamination_fraction": float(np.mean(rho)),
    }
    return corrected_cg, rho, n_genes_corrected, meta


def _run_sccdc(adata, *, raw=None, layer=None, cluster_key=None,
               restriction_factor=0.5, min_cell=50, auc_thres=0.9,
               random_state=1, **kwargs):
    """scCDC — needs a filtered, clustered matrix; corrects only GCGs."""
    sccdc = _require("pysccdc", "scCDC ambient removal")
    cluster_key = _resolve_clusters(adata, cluster_key, "sccdc")

    det_kwargs = {"restriction_factor": restriction_factor}
    # scCDC detection has its own (larger) min_cell default; honour min_cell
    gcg = sccdc.ContaminationDetection(
        adata, cluster_key=cluster_key, layer=layer,
        min_cell=max(min_cell, 100), random_state=random_state,
        **det_kwargs)
    gcgs = list(gcg.index)

    out = sccdc.ContaminationCorrection(
        adata, gcg, cluster_key=cluster_key, layer=layer,
        auc_thres=auc_thres, min_cell=min_cell, copy=True)
    ratio = sccdc.ContaminationQuantification(
        adata, gcg, cluster_key=cluster_key, layer=layer,
        auc_thres=auc_thres, min_cell=min_cell)

    corrected_cg = sp.csr_matrix(out.layers["Corrected"])
    rho = _removed_fraction(_get_X(adata, layer), corrected_cg)
    meta = {
        "cluster_key": cluster_key,
        "GCGs": gcgs,
        "restriction_factor": restriction_factor,
        "contamination_ratio": float(ratio),
        "contamination_fraction": float(np.mean(rho)),
    }
    return corrected_cg, rho, len(gcgs), meta


def _run_cellbender(adata, *, raw=None, layer=None, **kwargs):
    """CellBender remove-background — optional heavyweight DL wrapper."""
    _require("cellbender", "CellBender ambient removal")
    # If the package is installed the heavy path would run here. Kept thin:
    # CellBender's own CLI / API is the supported entry point.
    raise NotImplementedError(
        "CellBender is installed but the in-process wrapper is intentionally "
        "thin. Run CellBender's `remove-background` CLI on the raw 10x h5 "
        "and load the corrected output with ov.read.")


def _run_scar(adata, *, raw=None, layer=None, **kwargs):
    """scAR ambient denoising — optional heavyweight DL wrapper."""
    _require("scar", "scAR ambient removal")
    raise NotImplementedError(
        "scAR is installed but the in-process wrapper is intentionally thin. "
        "Use scar.model / scar.setup_anndata directly with the raw matrix.")


_DISPATCH = {
    "soupx": _run_soupx,
    "decontx": _run_decontx,
    "fastcar": _run_fastcar,
    "sccdc": _run_sccdc,
    "envelope": _run_envelope,
    "cellbender": _run_cellbender,
    "scar": _run_scar,
}


# ---------------------------------------------------------------------------
# public entry point — remove_ambient
# ---------------------------------------------------------------------------
@register_function(
    aliases=[
        "remove_ambient", "ambient_removal", "decontaminate",
        "去除环境RNA", "环境RNA去除", "去污染",
    ],
    category="preprocessing",
    description=(
        "Unified ambient / contamination-RNA removal for droplet scRNA-seq. "
        "method selects 'soupx' / 'fastcar' (need the raw unfiltered matrix "
        "with empty droplets), 'decontx' / 'sccdc' (need a filtered, "
        "clustered matrix), or the optional deep-learning wrappers "
        "'cellbender' / 'scar'. Writes the decontaminated counts back into "
        "the AnnData and records the per-cell contamination fraction in "
        "obs and method metadata in uns['ambient']."
    ),
    examples=[
        "ov.pp.ambient.remove_ambient(adata, method='soupx', raw=raw_adata)",
        "ov.pp.ambient.remove_ambient(adata, method='decontx', "
        "cluster_key='leiden')",
        "ov.pp.ambient.remove_ambient(adata, method='fastcar', raw=raw_adata)",
        "ov.pp.ambient.remove_ambient(adata, method='sccdc', "
        "cluster_key='cell_type')",
    ],
    related=[
        "pp.ambient.estimate_contamination", "pp.ambient.contamination_report",
        "pp.ambient.count_integrity_check", "pp.qc",
    ],
)
def remove_ambient(
    adata,
    *,
    method: str = "soupx",
    raw=None,
    layer: Optional[str] = None,
    cluster_key: Optional[str] = None,
    batch_key: Optional[str] = None,
    copy: bool = False,
    keep_raw_layer: bool = True,
    raw_layer_name: str = "ambient_raw",
    check_integrity: bool = True,
    verbose: bool = False,
    **kwargs,
):
    """Remove ambient ("soup") RNA contamination from droplet scRNA-seq.

    A single ``method=`` dispatcher over four native R-parity backends and
    two optional deep-learning wrappers. The decontaminated counts replace
    ``.X`` (or ``layer``); the raw counts are preserved in a layer; the
    per-cell contamination fraction goes to ``obs`` and method metadata to
    ``uns['ambient']``.

    Parameters
    ----------
    adata
        The **filtered** AnnData of real cells (cells x genes, raw counts).
    method
        ``'soupx'`` / ``'fastcar'`` — need ``raw`` (the unfiltered droplet
        matrix with empty droplets) to build the ambient profile.
        ``'decontx'`` / ``'sccdc'`` — need a *clustered* AnnData (pass
        ``cluster_key`` or have ``obs['clusters']``); ``raw`` is optional.
        ``'cellbender'`` / ``'scar'`` — optional heavyweight DL wrappers.
    raw
        The raw / unfiltered AnnData (droplets x genes) — required for
        SoupX and FastCAR, optional (background anchor) for DecontX.
    layer
        Count layer to read from and write the correction to. ``None`` uses
        ``.X``.
    cluster_key
        ``obs`` column with cluster / cell-type labels — required for
        DecontX and scCDC, optional for SoupX (enables auto rho). When
        ``None``, the first of ``obs['clusters']``, ``'leiden'``,
        ``'louvain'``, ``'cell_type'``, ``'celltype'`` that exists is used.
        SoupX without any cluster labels (and no
        ``contamination_fraction``) warns and falls back to rho = 0.1.
    batch_key
        ``obs`` column with per-cell batch / sample labels — optional.
        When set, DecontX decontaminates each batch separately.
    copy
        Return a new AnnData instead of modifying ``adata`` in place.
    keep_raw_layer
        Store the pre-correction counts in ``layers[raw_layer_name]``.
    raw_layer_name
        Name of that layer. Default ``'ambient_raw'``.
    check_integrity
        Run :func:`count_integrity_check` and store the result in
        ``uns['ambient']['count_integrity']``.
    verbose
        Print backend progress.
    **kwargs
        Method-specific options, forwarded to the backend. Notable ones:
        SoupX — ``contamination_fraction``, ``soup_range``,
        ``round_to_int``; DecontX — ``max_iter``, ``seed``, ``n_jobs``
        (threads; the result does not depend on it); FastCAR —
        ``empty_droplet_cutoff``, ``contamination_chance_cutoff``; scCDC —
        ``restriction_factor``, ``min_cell``, ``auc_thres``,
        ``random_state`` (default 1, which reproduces R scCDC: its curve
        fit always starts from ``set.seed(1)``).

    Returns
    -------
    :class:`anndata.AnnData`
        The decontaminated AnnData (the same object when ``copy=False``).
        ``obs['ambient_contamination']`` holds the per-cell contamination
        fraction; ``uns['ambient']`` records the method, genes corrected
        and count-integrity statistics.
    """
    method = method.lower()
    if method not in _ALL_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {_ALL_METHODS}.")

    out = adata.copy() if copy else adata

    before = _get_X(out, layer)
    raw_before = before.copy() if sp.issparse(before) else np.array(before, dtype=np.float64)
    if keep_raw_layer:
        out.layers[raw_layer_name] = sp.csr_matrix(_get_X(out, layer))

    runner = _DISPATCH[method]
    corrected_cg, rho, n_genes_corrected, meta = runner(
        out, raw=raw, layer=layer, cluster_key=cluster_key,
        batch_key=batch_key, verbose=verbose, **kwargs)

    _set_X(out, corrected_cg, layer)

    obs_key = "ambient_contamination"
    out.obs[obs_key] = np.asarray(rho, dtype=float)

    info = {
        "method": method,
        "contamination_obs_key": obs_key,
        "contamination_fraction": float(np.mean(rho)),
        "n_genes_corrected": int(n_genes_corrected),
        "raw_layer": raw_layer_name if keep_raw_layer else None,
        **meta,
    }
    if check_integrity:
        info["count_integrity"] = count_integrity_check(
            raw_before, corrected_cg)
    out.uns["ambient"] = info

    if verbose:
        print(f"[ov.pp.ambient] method={method}  "
              f"mean contamination={info['contamination_fraction']:.4f}  "
              f"genes corrected={n_genes_corrected}")
    return out


# ---------------------------------------------------------------------------
# public entry point — estimate_contamination
# ---------------------------------------------------------------------------
@register_function(
    aliases=[
        "estimate_contamination", "ambient_estimate",
        "估计污染", "污染估计", "环境RNA估计",
    ],
    category="preprocessing",
    description=(
        "Estimate the per-cell ambient-RNA contamination fraction WITHOUT "
        "modifying the count matrix (report-only). method selects 'decontx' "
        "(variational-EM, needs a clustered matrix), 'soupx' (auto rho from "
        "marker genes, needs raw + clusters), 'fastcar' or 'sccdc'. Returns "
        "a per-cell pandas.Series and writes it to "
        "obs['ambient_contamination_est']."
    ),
    examples=[
        "rho = ov.pp.ambient.estimate_contamination(adata, method='decontx', "
        "cluster_key='leiden')",
        "rho = ov.pp.ambient.estimate_contamination(adata, method='soupx', "
        "raw=raw_adata, cluster_key='leiden')",
    ],
    related=["pp.ambient.remove_ambient", "pp.ambient.contamination_report"],
)
def estimate_contamination(
    adata,
    *,
    method: str = "decontx",
    raw=None,
    layer: Optional[str] = None,
    cluster_key: Optional[str] = None,
    batch_key: Optional[str] = None,
    obs_key: str = "ambient_contamination_est",
    verbose: bool = False,
    **kwargs,
):
    """Estimate the per-cell contamination fraction without correcting.

    A report-only counterpart to :func:`remove_ambient`: it runs the
    chosen backend purely to read out the per-cell contamination fraction
    and leaves the counts untouched.

    Parameters
    ----------
    adata
        The filtered AnnData of real cells.
    method
        ``'decontx'`` / ``'soupx'`` / ``'fastcar'`` / ``'sccdc'`` — see
        :func:`remove_ambient` for each method's input requirements.
    raw
        Raw / unfiltered AnnData — required for SoupX and FastCAR.
    layer
        Count layer to read. ``None`` uses ``.X``.
    cluster_key
        ``obs`` column with cluster labels — required for DecontX / scCDC.
    batch_key
        ``obs`` column with per-cell batch / sample labels — optional.
        When set, DecontX decontaminates each batch separately.
    obs_key
        ``obs`` column the estimate is written to. Default
        ``'ambient_contamination_est'``.
    verbose
        Print backend progress.
    **kwargs
        Method-specific options forwarded to the backend.

    Returns
    -------
    :class:`pandas.Series`
        Per-cell contamination fraction, indexed by ``adata.obs_names``.
        ``.attrs`` carries the method and the dataset-level mean.
    """
    method = method.lower()
    if method not in _NATIVE_METHODS:
        raise ValueError(
            f"estimate_contamination supports {_NATIVE_METHODS}; "
            f"got '{method}'.")

    # run the backend on a copy so the user's counts stay untouched
    work = adata.copy()
    runner = _DISPATCH[method]
    _, rho, _, meta = runner(
        work, raw=raw, layer=layer, cluster_key=cluster_key,
        batch_key=batch_key, verbose=verbose, **kwargs)

    rho = np.asarray(rho, dtype=float)
    series = pd.Series(rho, index=adata.obs_names, name=obs_key)
    series.attrs["method"] = method
    series.attrs["mean_contamination"] = float(np.mean(rho))
    series.attrs.update(meta)

    adata.obs[obs_key] = rho
    if verbose:
        print(f"[ov.pp.ambient] estimate_contamination method={method}  "
              f"mean={series.attrs['mean_contamination']:.4f}")
    return series

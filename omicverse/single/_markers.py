"""Marker gene identification and retrieval utilities.

Native re-implementation of the scanpy rank_genes_groups pipeline
(t-test, Wilcoxon, logreg) — no scanpy dependency required.
Ported from:
  github.com/scverse/scanpy/blob/af57cffc6eb7fa77618b2ab026231f72cd029c12
  /src/scanpy/tools/_rank_genes_groups.py
"""
from __future__ import annotations

from typing import Generator, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from .._oom_compat import oom_guard as _oom_guard
from .._oom_compat import is_oom as _is_oom


def _oom_guard_unless_cpp_cosg(func):
    """Materialise an AnnDataOOM before find_markers, except for COSG with
    backend='cpp', whose group sums stream the on-disk matrix directly."""
    import functools
    guarded = _oom_guard(materialize=True, result_keys_uns=['*'])(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        adata = kwargs.get("adata", args[0] if args else None)
        method = kwargs.get("method", args[2] if len(args) > 2 else "cosg")
        if adata is not None and _is_oom(adata) and method == "cosg":
            from .._backend import resolve_backend
            if resolve_backend(kwargs.get("backend")) == "cpp":
                return func(*args, **kwargs)
        return guarded(*args, **kwargs)

    return wrapper
from scipy import sparse, stats

from .._registry import register_function
from .._settings import add_reference, Colors, EMOJI
from ..report._provenance import tracked, note


def _note_markers(method: str, groupby: str, key_added: Optional[str]) -> None:
    """Emit the standard find_markers provenance annotation.

    Both branches (cosg / native statistical tests) land in the same
    storage slot and visualize the same way, so hoist the note() call.
    """
    note(
        backend=f"omicverse · method={method}",
        viz=[{
            "function": "ov.pl.markers_dotplot",
            "kwargs": {
                "groupby": groupby,
                "key": key_added or "rank_genes_groups",
                "n_genes": 3,
                "standard_scale": "var",
            },
        }],
    )

# ── Constants ─────────────────────────────────────────────────────────────────
_CONST_MAX_SIZE: int = 10_000_000


# ── Low-level helpers (mirroring scanpy) ──────────────────────────────────────

def _rankdata_cols(data: np.ndarray) -> np.ndarray:
    """Column-wise rank, equivalent to scipy.stats.rankdata per column."""
    from scipy.stats import rankdata as _rd

    ranked = np.empty(data.shape, dtype=np.float64)
    for j in range(data.shape[1]):
        ranked[:, j] = _rd(data[:, j])
    return ranked


def _tiecorrect_cols(rankvals: np.ndarray) -> np.ndarray:
    """Tie-correction coefficient per column (matches scipy.stats.tiecorrect)."""
    from scipy.stats import tiecorrect as _tc

    result = np.ones(rankvals.shape[1], dtype=np.float64)
    for j in range(rankvals.shape[1]):
        result[j] = _tc(rankvals[:, j])
    return result


def _mean_var_axis0(X) -> tuple[np.ndarray, np.ndarray]:
    """Mean and unbiased variance (ddof=1) along axis 0, sparse-aware.

    Matches the formula used by ``scanpy.tools._rank_genes_groups._get_mean_var``
    (installed version):  ``var = E[X²] - E[X]²`` with Bessel's correction.
    For sparse data, element-wise squaring is done in the native dtype (float32)
    before accumulating in float64, which replicates scanpy's numba kernel.
    For dense data, numpy is used with explicit float64 promotion.
    """
    n = X.shape[0]
    if sparse.issparse(X):
        # X.mean(axis=0) accumulates in float64 for scipy sparse regardless of dtype
        mean = np.asarray(X.mean(axis=0), dtype=np.float64).ravel()
        # X.multiply(X) keeps native dtype (float32 * float32 = float32)
        # matching scanpy's numba kernel: squared_sums += value * value (float32 mul)
        mean_sq = np.asarray(X.multiply(X).mean(axis=0), dtype=np.float64).ravel()
        var = (mean_sq - mean ** 2) * n / (n - 1)
    else:
        X_arr = np.asarray(X)
        # elem-wise multiply in native dtype, then mean in float64 — mirrors scanpy
        mean = np.mean(X_arr, axis=0, dtype=np.float64)
        mean_sq = np.mean(X_arr * X_arr, axis=0, dtype=np.float64)
        var = (mean_sq - mean ** 2) * n / (n - 1)
    return mean, var


def _select_top_n(scores: np.ndarray, n_top: int) -> np.ndarray:
    n_from = scores.shape[0]
    reference_indices = np.arange(n_from, dtype=int)
    partition = np.argpartition(scores, -n_top)[-n_top:]
    partial_indices = np.argsort(scores[partition])[::-1]
    return reference_indices[partition][partial_indices]


# ── Rank-iterator (same chunking strategy as scanpy) ──────────────────────────

def _ranks_iter(
    X,
    mask_obs: np.ndarray | None = None,
    mask_obs_rest: np.ndarray | None = None,
) -> Generator[tuple[np.ndarray, int, int], None, None]:
    """Yield ``(ranks, left, right)`` chunks over genes."""
    n_genes = X.shape[1]
    masked = mask_obs is not None and mask_obs_rest is not None

    if masked:
        n_cells = int(mask_obs.sum()) + int(mask_obs_rest.sum())

        def _get_chunk(left: int, right: int) -> np.ndarray:
            a = X[mask_obs, left:right]
            b = X[mask_obs_rest, left:right]
            if sparse.issparse(a):
                return np.vstack([a.toarray(), b.toarray()])
            return np.vstack([a, b])
    else:
        n_cells = X.shape[0]

        def _get_chunk(left: int, right: int) -> np.ndarray:
            chunk = X[:, left:right]
            if sparse.issparse(chunk):
                return chunk.toarray()
            return np.asarray(chunk)

    max_chunk = max(_CONST_MAX_SIZE // n_cells, 1)
    for left in range(0, n_genes, max_chunk):
        right = min(left + max_chunk, n_genes)
        ranks = _rankdata_cols(_get_chunk(left, right))
        yield ranks, left, right


# ── Core statistics class ─────────────────────────────────────────────────────

class _RankGenesOV:
    """
    Native reimplementation of scanpy's ``_RankGenes``.

    Accepts the same interface so ``find_markers`` can delegate to it
    without importing scanpy.
    """

    def __init__(
        self,
        adata: AnnData,
        groups: Iterable[str] | str,
        groupby: str,
        *,
        reference: str = "rest",
        use_raw: bool | None = None,
        layer: str | None = None,
        comp_pts: bool = False,
    ) -> None:
        # expm1 with optional log-base
        base = adata.uns.get("log1p", {}).get("base")
        self.expm1_func = (
            (lambda x: np.expm1(x * np.log(base))) if base is not None
            else np.expm1
        )

        # Resolve group list
        obs_col = adata.obs[groupby]
        if not hasattr(obs_col, "cat"):
            obs_col = obs_col.astype("category")

        if isinstance(groups, str) and groups == "all":
            all_groups = obs_col.cat.categories.tolist()
        else:
            all_groups = [str(g) for g in groups]
            # Include reference in group list when using a specific reference
            if reference != "rest" and reference not in all_groups:
                all_groups.append(reference)

        # Guard against singleton groups
        counts = obs_col.value_counts()
        invalid = set(all_groups) & set(counts[counts < 2].index.tolist())
        if invalid:
            raise ValueError(
                f"Groups {invalid} contain < 2 cells; cannot compute statistics."
            )

        self.groups_order = np.array(all_groups)
        obs_str = obs_col.astype(str).values
        self.groups_masks_obs = np.stack(
            [obs_str == g for g in all_groups], axis=0
        )  # shape: (n_groups, n_cells)

        # Expression matrix
        if layer is not None:
            if use_raw:
                raise ValueError("Cannot use `layer` together with `use_raw=True`.")
            X = adata.layers[layer]
            self.var_names = adata.var_names
        elif use_raw and adata.raw is not None:
            X = adata.raw.X
            self.var_names = adata.raw.var_names
        else:
            X = adata.X
            self.var_names = adata.var_names

        if sparse.issparse(X):
            X.eliminate_zeros()
        self.X = X

        # Reference index
        self.ireference: int | None = None
        if reference != "rest":
            idx = np.where(self.groups_order == reference)[0]
            if idx.size == 0:
                raise ValueError(
                    f"reference='{reference}' not found in groups {all_groups}."
                )
            self.ireference = int(idx[0])

        self.means: np.ndarray | None = None
        self.vars: np.ndarray | None = None
        self.means_rest: np.ndarray | None = None
        self.vars_rest: np.ndarray | None = None

        self.comp_pts = comp_pts
        self.pts: np.ndarray | None = None
        self.pts_rest: np.ndarray | None = None
        self.stats: pd.DataFrame | None = None

        # For logreg
        self.grouping_mask = adata.obs[groupby].isin(all_groups).values
        self.grouping = (
            adata.obs.loc[self.grouping_mask, groupby].astype("category")
        )

    # ── Basic statistics ───────────────────────────────────────────────────────

    def _basic_stats(self) -> None:
        n_genes = self.X.shape[1]
        n_groups = len(self.groups_order)

        self.means = np.zeros((n_groups, n_genes))
        self.vars = np.zeros((n_groups, n_genes))
        if self.comp_pts:
            self.pts = np.zeros((n_groups, n_genes))

        if self.ireference is None:
            self.means_rest = np.zeros((n_groups, n_genes))
            self.vars_rest = np.zeros((n_groups, n_genes))
            if self.comp_pts:
                self.pts_rest = np.zeros((n_groups, n_genes))
        else:
            mask_ref = self.groups_masks_obs[self.ireference]
            self.means[self.ireference], self.vars[self.ireference] = (
                _mean_var_axis0(self.X[mask_ref])
            )

        get_nnz = (
            (lambda M: M.getnnz(axis=0)) if sparse.issparse(self.X)
            else (lambda M: np.count_nonzero(M, axis=0))
        )

        for gi, mask in enumerate(self.groups_masks_obs):
            X_mask = self.X[mask]
            if self.comp_pts:
                self.pts[gi] = get_nnz(X_mask) / X_mask.shape[0]
            if self.ireference is not None and gi == self.ireference:
                continue
            self.means[gi], self.vars[gi] = _mean_var_axis0(X_mask)
            if self.ireference is None:
                X_rest = self.X[~mask]
                self.means_rest[gi], self.vars_rest[gi] = _mean_var_axis0(X_rest)
                if self.comp_pts:
                    self.pts_rest[gi] = get_nnz(X_rest) / X_rest.shape[0]

    # ── Statistical tests ──────────────────────────────────────────────────────

    def t_test(
        self, method: str = "t-test"
    ) -> Generator[tuple[int, np.ndarray, np.ndarray], None, None]:
        self._basic_stats()
        for gi, (mask, mean_g, var_g) in enumerate(
            zip(self.groups_masks_obs, self.means, self.vars)
        ):
            if self.ireference is not None and gi == self.ireference:
                continue

            ns_group = int(mask.sum())
            if self.ireference is not None:
                mean_r = self.means[self.ireference]
                var_r = self.vars[self.ireference]
                ns_rest = int(self.groups_masks_obs[self.ireference].sum())
            else:
                mean_r = self.means_rest[gi]
                var_r = self.vars_rest[gi]
                ns_rest = self.X.shape[0] - ns_group

            ns_rest_eff = ns_group if method == "t-test_overestim_var" else ns_rest

            with np.errstate(invalid="ignore"):
                scores, pvals = stats.ttest_ind_from_stats(
                    mean1=mean_g, std1=np.sqrt(var_g), nobs1=ns_group,
                    mean2=mean_r, std2=np.sqrt(var_r), nobs2=ns_rest_eff,
                    equal_var=False,
                )
            scores[np.isnan(scores)] = 0.0
            pvals[np.isnan(pvals)] = 1.0
            yield gi, scores, pvals

    def wilcoxon(
        self, *, tie_correct: bool = False
    ) -> Generator[tuple[int, np.ndarray, np.ndarray], None, None]:
        self._basic_stats()
        n_genes = self.X.shape[1]

        if self.ireference is not None:
            # One group vs specific reference ─────────────────────────────────
            scores = np.zeros(n_genes)
            tc_coef = np.zeros(n_genes) if tie_correct else 1.0

            for gi, mask in enumerate(self.groups_masks_obs):
                if gi == self.ireference:
                    continue
                mask_ref = self.groups_masks_obs[self.ireference]
                n_active = int(mask.sum())
                m_active = int(mask_ref.sum())

                for ranks, left, right in _ranks_iter(self.X, mask, mask_ref):
                    scores[left:right] = ranks[:n_active, :].sum(axis=0)
                    if tie_correct:
                        tc_coef[left:right] = _tiecorrect_cols(ranks)

                std_dev = np.sqrt(
                    tc_coef * n_active * m_active * (n_active + m_active + 1) / 12.0
                )
                sc = (scores - n_active * (n_active + m_active + 1) / 2.0) / std_dev
                sc[np.isnan(sc)] = 0.0
                pvals = 2.0 * stats.norm.sf(np.abs(sc))
                yield gi, sc, pvals

        else:
            # One group vs rest ────────────────────────────────────────────────
            n_groups = len(self.groups_order)
            n_cells = self.X.shape[0]
            scores = np.zeros((n_groups, n_genes))
            tc_coef = (
                np.zeros((n_groups, n_genes)) if tie_correct else None
            )

            for ranks, left, right in _ranks_iter(self.X):
                for gi, mask in enumerate(self.groups_masks_obs):
                    scores[gi, left:right] = ranks[mask, :].sum(axis=0)
                    if tie_correct:
                        tc_coef[gi, left:right] = _tiecorrect_cols(ranks)

            for gi, mask in enumerate(self.groups_masks_obs):
                n_active = int(mask.sum())
                coef = tc_coef[gi] if tie_correct else 1.0
                std_dev = np.sqrt(
                    coef * n_active * (n_cells - n_active) * (n_cells + 1) / 12.0
                )
                sc = (scores[gi] - n_active * (n_cells + 1) / 2.0) / std_dev
                sc[np.isnan(sc)] = 0.0
                pvals = 2.0 * stats.norm.sf(np.abs(sc))
                yield gi, sc, pvals

    def logreg(
        self, **kwds
    ) -> Generator[tuple[int, np.ndarray, None], None, None]:
        from sklearn.linear_model import LogisticRegression

        if len(self.groups_order) == 1:
            raise ValueError("Cannot perform logistic regression on a single cluster.")

        x = self.X[self.grouping_mask]
        clf = LogisticRegression(**kwds)
        clf.fit(x, self.grouping.cat.codes)
        scores_all = clf.coef_
        existing_codes = np.unique(self.grouping.cat.codes)

        for igroup, cat in enumerate(self.groups_order):
            if len(self.groups_order) <= 2:
                scores = scores_all[0]
            else:
                cat_code = int(np.argmax(self.grouping.cat.categories == cat))
                scores_idx = int(np.argmax(existing_codes == cat_code))
                scores = scores_all[scores_idx]
            yield igroup, scores, None
            if len(self.groups_order) <= 2:
                break

    # ── Orchestration ──────────────────────────────────────────────────────────

    def compute_statistics(
        self,
        method: str,
        *,
        corr_method: str = "benjamini-hochberg",
        n_genes_user: int | None = None,
        rankby_abs: bool = False,
        tie_correct: bool = False,
        **kwds,
    ) -> None:
        if method in ("t-test", "t-test_overestim_var"):
            gen = self.t_test(method)
        elif method == "wilcoxon":
            gen = self.wilcoxon(tie_correct=tie_correct)
        elif method == "logreg":
            gen = self.logreg(**kwds)
        else:
            raise ValueError(f"Unknown method '{method}'.")

        self.stats = None
        n_genes = self.X.shape[1]

        for gi, scores, pvals in gen:
            group_name = str(self.groups_order[gi])

            if n_genes_user is not None:
                sort_scores = np.abs(scores) if rankby_abs else scores
                global_indices = _select_top_n(sort_scores, n_genes_user)
                first_col = "names"
            else:
                global_indices = slice(None)
                first_col = "scores"

            if self.stats is None:
                self.stats = pd.DataFrame(
                    columns=pd.MultiIndex.from_tuples([(group_name, first_col)])
                )

            if n_genes_user is not None:
                self.stats[group_name, "names"] = self.var_names[global_indices]

            self.stats[group_name, "scores"] = scores[global_indices]

            if pvals is not None:
                self.stats[group_name, "pvals"] = pvals[global_indices]

                pvals_corr = pvals.copy()
                pvals_corr[np.isnan(pvals_corr)] = 1.0
                if corr_method == "benjamini-hochberg":
                    from statsmodels.stats.multitest import multipletests

                    _, pvals_adj, _, _ = multipletests(
                        pvals_corr, alpha=0.05, method="fdr_bh"
                    )
                elif corr_method == "bonferroni":
                    pvals_adj = np.minimum(pvals_corr * n_genes, 1.0)
                else:
                    raise ValueError(
                        f"corr_method must be 'benjamini-hochberg' or 'bonferroni'."
                    )
                self.stats[group_name, "pvals_adj"] = pvals_adj[global_indices]

            # Log fold changes (from means)
            if self.means is not None:
                mean_g = self.means[gi]
                mean_r = (
                    self.means_rest[gi]
                    if self.ireference is None
                    else self.means[self.ireference]
                )
                fc = (self.expm1_func(mean_g) + 1e-9) / (
                    self.expm1_func(mean_r) + 1e-9
                )
                self.stats[group_name, "logfoldchanges"] = np.log2(fc[global_indices])

        if n_genes_user is None and self.stats is not None:
            self.stats.index = self.var_names


# ── Structured-array result storage (matching scanpy's format exactly) ─────────

_DTYPES = {
    "names": "O",
    "scores": "float32",
    "logfoldchanges": "float32",
    "pvals": "float64",
    "pvals_adj": "float64",
}


def _store_results(
    adata: AnnData,
    key_added: str,
    test_obj: _RankGenesOV,
    method: str,
    groupby: str,
    reference: str,
    use_raw: bool | None,
    layer: str | None,
    corr_method: str,
    pts: bool,
) -> None:
    """Write ``test_obj.stats`` into ``adata.uns[key_added]`` as structured arrays."""
    adata.uns[key_added] = {}
    adata.uns[key_added]["params"] = dict(
        groupby=groupby,
        reference=reference,
        method=method,
        use_raw=use_raw,
        layer=layer,
        corr_method=corr_method,
    )

    if test_obj.pts is not None:
        groups_names = [str(n) for n in test_obj.groups_order]
        adata.uns[key_added]["pts"] = pd.DataFrame(
            test_obj.pts.T, index=test_obj.var_names, columns=groups_names
        )
    if test_obj.pts_rest is not None:
        groups_names = [str(n) for n in test_obj.groups_order]
        adata.uns[key_added]["pts_rest"] = pd.DataFrame(
            test_obj.pts_rest.T, index=test_obj.var_names, columns=groups_names
        )

    if test_obj.stats is None:
        return

    test_obj.stats.columns = test_obj.stats.columns.swaplevel()
    for col in test_obj.stats.columns.levels[0]:
        dtype = _DTYPES.get(col, "O")
        adata.uns[key_added][col] = test_obj.stats[col].to_records(
            index=False, column_dtypes=dtype
        )


# ── pts helper for cosg ───────────────────────────────────────────────────────

def _cosg_add_pts(
    adata: AnnData,
    key_added: str,
    groupby: str,
    *,
    use_raw: bool = True,
    layer: str | None = None,
) -> None:
    """Compute pct_group / pct_rest for cosg results and store in adata.uns."""
    result = adata.uns[key_added]
    names_data = result.get("names")
    if names_data is None:
        return

    if isinstance(names_data, pd.DataFrame):
        group_names = names_data.columns.tolist()
    elif hasattr(names_data, "dtype") and names_data.dtype.names:
        group_names = list(names_data.dtype.names)
    else:
        return

    # Get expression matrix and var_names
    if layer is not None:
        X = adata.layers[layer]
        var_names = adata.var_names
    elif use_raw and adata.raw is not None:
        X = adata.raw.X
        var_names = adata.raw.var_names
    else:
        X = adata.X
        var_names = adata.var_names

    obs_col = adata.obs[groupby].astype(str).values
    n_cells = X.shape[0]
    get_nnz = (
        (lambda M: np.asarray(M.getnnz(axis=0)).ravel())
        if sparse.issparse(X)
        else (lambda M: np.count_nonzero(M, axis=0))
    )

    pts_dict: dict[str, np.ndarray] = {}
    pts_rest_dict: dict[str, np.ndarray] = {}
    for g in group_names:
        mask = obs_col == g
        n_g = int(mask.sum())
        n_r = n_cells - n_g
        if n_g == 0:
            continue
        pts_dict[g] = get_nnz(X[mask]) / n_g
        pts_rest_dict[g] = get_nnz(X[~mask]) / n_r if n_r > 0 else np.zeros(X.shape[1])

    if pts_dict:
        result["pts"] = pd.DataFrame(pts_dict, index=var_names)
    if pts_rest_dict:
        result["pts_rest"] = pd.DataFrame(pts_rest_dict, index=var_names)


# ── Public API ─────────────────────────────────────────────────────────────────

@register_function(
    aliases=["查找标记基因", "find_markers", "寻找marker", "marker查找", "cluster_markers_find"],
    category="single",
    description="Find marker genes per cluster using cosg, t-test, wilcoxon, or logreg (no scanpy dep for statistical methods)",
    examples=[
        "# COSG (recommended, fast)",
        "ov.single.find_markers(adata, groupby='leiden', method='cosg', n_genes=50)",
        "",
        "# Welch t-test",
        "ov.single.find_markers(adata, groupby='leiden', method='t-test',",
        "                       key_added='ttest_markers')",
        "",
        "# Wilcoxon rank-sum with tie correction",
        "ov.single.find_markers(adata, groupby='celltype', method='wilcoxon',",
        "                       tie_correct=True, key_added='wilcoxon_markers')",
        "",
        "# Subset groups with specific reference",
        "ov.single.find_markers(adata, groupby='leiden', method='t-test',",
        "                       groups=['0','1','2'], reference='0')",
        "",
        "# Retrieve & visualise",
        "df = ov.single.get_markers(adata, n_genes=10)",
        "ov.pl.markers_dotplot(adata, groupby='leiden', n_genes=5)",
    ],
    related=["single.get_markers", "pl.markers_dotplot", "single.cosg"],
)
@_oom_guard_unless_cpp_cosg
@tracked("find_markers", "ov.single.find_markers")
def find_markers(
    adata: AnnData,
    groupby: str,
    method: str = "cosg",
    n_genes: int = 50,
    key_added: Optional[str] = None,
    use_raw: Optional[bool] = None,
    layer: Optional[str] = None,
    groups: Union[str, Sequence[str]] = "all",
    reference: str = "rest",
    corr_method: str = "benjamini-hochberg",
    rankby_abs: bool = False,
    tie_correct: bool = False,
    pts: bool = True,
    backend: Optional[str] = None,
    **kwargs,
) -> None:
    r"""Find marker genes for each cluster / group in single-cell data.

    A unified wrapper supporting multiple algorithms.  For statistical methods
    (``t-test``, ``wilcoxon``, ``logreg``) the implementation is ported
    directly from the scanpy implementation — **no scanpy runtime dependency**.  Results are
    stored in ``adata.uns[key_added]`` using the same structured-array format
    as scanpy's ``rank_genes_groups``, so all downstream tools (including
    :func:`omicverse.single.get_markers` and
    :func:`omicverse.pl.markers_dotplot`) work out of the box.

    Parameters
    ----------
        adata: Annotated data matrix. **Data must be log-normalised** for
            statistical tests; raw counts are expected for ``method='cosg'``.
        groupby: Key in ``adata.obs`` to group cells by (e.g. ``'leiden'``).
        method: Algorithm. One of:

            * ``'cosg'`` — cosine-similarity-based, fast, recommended for
              large datasets.
            * ``'t-test'`` — Welch's t-test.
            * ``'t-test_overestim_var'`` — t-test with per-group variance
              overestimation (conservative).
            * ``'wilcoxon'`` — Wilcoxon rank-sum / Mann-Whitney U test.
            * ``'logreg'`` — logistic regression (requires scikit-learn).

            Default: ``'cosg'``.
        n_genes: Top marker genes per group to keep. Default: ``50``.
        key_added: Key in ``adata.uns`` to write results to.
            Default: ``'rank_genes_groups'``.
        use_raw: Use ``adata.raw`` for expression values.  ``None`` (default)
            means *use raw if it exists* (matching scanpy behaviour).
        layer: Layer to use instead of ``adata.X``. Default: ``None``.
        groups: Groups to compute markers for — ``'all'`` or a list of names.
            Default: ``'all'``.
        reference: Reference group.  ``'rest'`` (default) compares each group
            against the union of all other cells; a group name restricts the
            comparison to that group only.
        corr_method: Multiple-testing correction. ``'benjamini-hochberg'``
            (default) or ``'bonferroni'``. Ignored for ``'cosg'`` and
            ``'logreg'``.
        rankby_abs: Rank genes by absolute score instead of raw score.
            Default: ``False``.
        tie_correct: Apply tie correction for ``'wilcoxon'``. Default: ``False``.
        pts: Compute fraction of cells expressing each gene (stored as
            ``adata.uns[key_added]['pts']``). Default: ``False``.
        **kwargs: Forwarded to the underlying method (e.g. ``mu`` for cosg,
            or sklearn parameters for logreg).

    Returns
    -------
        ``None``.  Results are written to ``adata.uns[key_added]``.

    Examples:
        >>> import omicverse as ov
        >>> ov.single.find_markers(adata, groupby='leiden', method='cosg')
        >>> df = ov.single.get_markers(adata, n_genes=5)
        >>> ov.pl.markers_dotplot(adata, groupby='leiden', n_genes=5)
    """
    if key_added is None:
        key_added = "rank_genes_groups"

    # Resolve group count for display
    _obs_col = adata.obs[groupby]
    if not hasattr(_obs_col, "cat"):
        _obs_col = _obs_col.astype("category")
    if isinstance(groups, str) and groups == "all":
        _n_groups = len(_obs_col.cat.categories)
    else:
        _n_groups = len(groups)

    print(
        f"{Colors.CYAN}{EMOJI['start']} Finding marker genes{Colors.ENDC} | "
        f"{Colors.BOLD}method:{Colors.ENDC} {method} | "
        f"{Colors.BOLD}groupby:{Colors.ENDC} {groupby} | "
        f"{Colors.BOLD}n_groups:{Colors.ENDC} {_n_groups} | "
        f"{Colors.BOLD}n_genes:{Colors.ENDC} {n_genes}"
    )

    from ..pp._dask import is_dask
    if is_dask(adata.layers[layer] if layer is not None else adata.X):
        # A Dask X (ov.read(storage='disk') in GPU mode) is only streamed by the C++ COSG
        # group statistics; anything else would load the whole matrix.
        from .._backend import resolve_backend
        if method != "cosg" or resolve_backend(backend) != "cpp" or groups != "all" \
                or reference != "rest":
            raise NotImplementedError(
                "For a Dask X, find_markers supports method='cosg' with backend='cpp', "
                "groups='all' and reference='rest'.")

    # ── COSG delegates to the dedicated module ─────────────────────────────────
    if method == "cosg":
        from ._cosg import cosg

        _use_raw_cosg = use_raw if use_raw is not None else True
        cosg(
            adata,
            groupby=groupby,
            groups=groups,
            n_genes_user=n_genes,
            key_added=key_added,
            use_raw=_use_raw_cosg,
            layer=layer,
            reference=reference,
            backend=backend,
            **kwargs,
        )

        # ── Compute pct (pts / pts_rest) for cosg results ─────────────────
        if pts:
            from .._backend import resolve_backend
            if resolve_backend(backend) == 'cpp':
                from anndataoom_fast._omicverse import ov_cosg_add_pts
                ov_cosg_add_pts(adata, key_added, groupby, use_raw=_use_raw_cosg, layer=layer)
            else:
                _cosg_add_pts(adata, key_added, groupby, use_raw=_use_raw_cosg, layer=layer)

        print(
            f"{Colors.GREEN}{EMOJI['done']} Done{Colors.ENDC} | "
            f"{_n_groups} groups × {n_genes} genes | "
            f"stored in {Colors.BOLD}adata.uns['{key_added}']{Colors.ENDC}"
        )
        _note_markers(method, groupby, key_added)
        return

    # ── Native statistical tests ───────────────────────────────────────────────
    if method not in ("t-test", "t-test_overestim_var", "wilcoxon", "logreg"):
        raise ValueError(
            f"Unknown method '{method}'. Choose from: 'cosg', 't-test', "
            f"'t-test_overestim_var', 'wilcoxon', 'logreg'."
        )

    # Resolve use_raw: default mirrors scanpy (use raw if it exists)
    if use_raw is None:
        use_raw = adata.raw is not None

    # Resolve groups list (include reference when needed)
    if isinstance(groups, str) and groups == "all":
        groups_order: str | list = "all"
    else:
        groups_order = [str(g) for g in groups]
        if reference != "rest" and reference not in groups_order:
            groups_order.append(reference)

    test_obj = _RankGenesOV(
        adata,
        groups_order,
        groupby,
        reference=reference,
        use_raw=use_raw,
        layer=layer,
        comp_pts=pts,
    )

    # Clamp n_genes to available genes
    n_genes_user = min(n_genes, test_obj.X.shape[1])

    test_obj.compute_statistics(
        method,
        corr_method=corr_method,
        n_genes_user=n_genes_user,
        rankby_abs=rankby_abs,
        tie_correct=tie_correct,
        **kwargs,
    )

    _store_results(
        adata,
        key_added,
        test_obj,
        method=method,
        groupby=groupby,
        reference=reference,
        use_raw=use_raw,
        layer=layer,
        corr_method=corr_method,
        pts=pts,
    )

    _actual_n_groups = len(test_obj.groups_order)
    print(
        f"{Colors.GREEN}{EMOJI['done']} Done{Colors.ENDC} | "
        f"{_actual_n_groups} groups × {n_genes_user} genes | "
        f"corr: {corr_method} | "
        f"stored in {Colors.BOLD}adata.uns['{key_added}']{Colors.ENDC}"
    )

    add_reference(adata, "find_markers", f"marker gene identification with {method}")
    _note_markers(method, groupby, key_added)


# ── get_markers ────────────────────────────────────────────────────────────────

@register_function(
    aliases=["获取标记基因", "get_markers", "提取marker", "marker提取", "extract_markers"],
    category="single",
    description="Extract top marker genes from adata.uns as a clean DataFrame or dictionary, with optional cluster / score filtering",
    examples=[
        "# All groups, top 10 markers",
        "df = ov.single.get_markers(adata, n_genes=10)",
        "",
        "# Single cluster",
        "df = ov.single.get_markers(adata, n_genes=10, groups='0')",
        "",
        "# Multiple clusters",
        "df = ov.single.get_markers(adata, n_genes=5, groups=['0', '1', '2'])",
        "",
        "# Return as {cluster: [gene, ...]} dict",
        "d = ov.single.get_markers(adata, n_genes=5, return_type='dict')",
        "",
        "# Filter low fold-change genes from custom key",
        "df = ov.single.get_markers(adata, n_genes=10, key='wilcoxon_markers',",
        "                            min_logfoldchange=1.0)",
    ],
    related=["single.find_markers", "pl.markers_dotplot", "single.cosg"],
)
def get_markers(
    adata: AnnData,
    n_genes: int = 10,
    key: str = "rank_genes_groups",
    groups: Optional[Union[str, Sequence[str]]] = None,
    return_type: str = "dataframe",
    min_logfoldchange: Optional[float] = None,
    min_score: Optional[float] = None,
    min_pval_adj: Optional[float] = None,
) -> Union[pd.DataFrame, dict]:
    r"""Extract top marker genes from ``rank_genes_groups`` results.

    Works with results produced by :func:`find_markers`,
    :func:`omicverse.single.cosg`, or scanpy's ``rank_genes_groups``.

    Parameters
    ----------
        adata: Annotated data matrix.
        n_genes: Maximum number of top marker genes to return per group.
            Default: ``10``.
        key: Key in ``adata.uns`` holding the results.
            Default: ``'rank_genes_groups'``.
        groups: One or more group names to extract.  Accepts:

            * ``None`` — all groups (default).
            * ``'0'`` — single group as a string.
            * ``['0', '1', '3']`` — explicit list of groups.
        return_type: Output format:

            * ``'dataframe'`` — :class:`pandas.DataFrame` with columns
              ``[group, rank, names, scores, logfoldchanges, pvals, pvals_adj]``
              (columns present depend on which stats were computed).
            * ``'dict'`` — ``{group: [gene1, gene2, ...]}`` mapping.

            Default: ``'dataframe'``.
        min_logfoldchange: Keep only genes with
            ``|logfoldchange| ≥ min_logfoldchange``. Default: ``None``.
        min_score: Keep only genes with ``score ≥ min_score``. Default: ``None``.
        min_pval_adj: Keep only genes with ``pvals_adj ≤ min_pval_adj``.
            Default: ``None``.

    Returns
    -------
        :class:`pandas.DataFrame` or ``dict`` depending on ``return_type``.

    Examples:
        >>> import omicverse as ov
        >>> ov.single.find_markers(adata, groupby='leiden')
        >>> # All clusters
        >>> df = ov.single.get_markers(adata, n_genes=10)
        >>> # Single cluster
        >>> df0 = ov.single.get_markers(adata, n_genes=5, groups='0')
        >>> # As dict for annotation workflows
        >>> d = ov.single.get_markers(adata, n_genes=5, return_type='dict')
    """
    if key not in adata.uns:
        raise KeyError(
            f"Key '{key}' not found in adata.uns. "
            "Run ov.single.find_markers() first."
        )

    result = adata.uns[key]

    # ── Detect which stat columns are addressable by group name ────────────────
    def _col_is_keyed(col_key: str) -> bool:
        if col_key not in result:
            return False
        data = result[col_key]
        if data is None:
            return False
        if isinstance(data, pd.DataFrame):
            return True
        return hasattr(data, "dtype") and bool(data.dtype.names)

    stat_cols = ["scores", "logfoldchanges", "pvals", "pvals_adj"]
    available_cols = ["names"] + [c for c in stat_cols if _col_is_keyed(c)]

    # ── Resolve group names ────────────────────────────────────────────────────
    names_data = result["names"]
    if isinstance(names_data, pd.DataFrame):
        all_groups = names_data.columns.tolist()
    elif hasattr(names_data, "dtype") and names_data.dtype.names:
        all_groups = list(names_data.dtype.names)
    else:
        raise ValueError(
            f"Unexpected format for adata.uns['{key}']['names']. "
            "Expected a DataFrame or a structured numpy array."
        )

    # ── Filter groups ──────────────────────────────────────────────────────────
    if groups is not None:
        if isinstance(groups, str):
            groups = [groups]
        groups = [str(g) for g in groups]
        missing = set(groups) - set(all_groups)
        if missing:
            raise KeyError(
                f"Group(s) {missing} not found in '{key}'. "
                f"Available groups: {all_groups}"
            )
        all_groups = [g for g in all_groups if g in groups]

    def _get_col_values(col_key: str, group: str) -> np.ndarray:
        data = result[col_key]
        if isinstance(data, pd.DataFrame):
            return data[group].values
        return data[group]  # structured numpy array

    # ── pts lookup helpers (gene-name indexed DataFrames) ─────────────────────
    def _pts_df(col_key: str) -> Optional[pd.DataFrame]:
        """Return pts/pts_rest DataFrame if present and gene-indexed."""
        if col_key not in result:
            return None
        data = result[col_key]
        if isinstance(data, pd.DataFrame) and len(data) > 0:
            return data
        return None

    pts_df = _pts_df("pts")
    pts_rest_df = _pts_df("pts_rest")

    # ── Build records ──────────────────────────────────────────────────────────
    records = []
    for group in all_groups:
        col_data = {col: _get_col_values(col, group) for col in available_cols}
        n_available = len(col_data["names"])
        n_take = min(n_genes, n_available)
        for rank in range(n_take):
            rec = {"group": group, "rank": rank + 1}
            for col in available_cols:
                rec[col] = col_data[col][rank]
            # pct_group / pct_rest from pts DataFrames (gene-name indexed)
            gene = rec["names"]
            if pts_df is not None and group in pts_df.columns and gene in pts_df.index:
                rec["pct_group"] = float(pts_df.loc[gene, group])
            if pts_rest_df is not None and group in pts_rest_df.columns and gene in pts_rest_df.index:
                rec["pct_rest"] = float(pts_rest_df.loc[gene, group])
            records.append(rec)

    df = pd.DataFrame(records)

    # ── Apply filters ──────────────────────────────────────────────────────────
    if min_logfoldchange is not None and "logfoldchanges" in df.columns:
        df = df[df["logfoldchanges"].abs() >= min_logfoldchange]
    if min_score is not None and "scores" in df.columns:
        df = df[df["scores"] >= min_score]
    if min_pval_adj is not None and "pvals_adj" in df.columns:
        df = df[df["pvals_adj"] <= min_pval_adj]

    df = df.reset_index(drop=True)

    # ── Return ─────────────────────────────────────────────────────────────────
    if return_type == "dict":
        return {
            grp: df.loc[df["group"] == grp, "names"].tolist()
            for grp in df["group"].unique()
        }
    return df

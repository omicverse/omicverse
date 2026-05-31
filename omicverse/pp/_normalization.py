from __future__ import annotations

from operator import truediv
from typing import TYPE_CHECKING
from warnings import warn
from datetime import datetime

import numba
import numpy as np
#from fast_array_utils import stats

from .._settings import settings, EMOJI, Colors
from .._registry import register_function
from ._compat import CSBase, CSCBase, CSRBase, DaskArray, old_positionals

# Import local implementations from _scale.py
from ._scale import (
    axis_mul_or_truediv,
    view_to_actual,
    axis_sum,
    is_backed_type,
    _check_array_function_arguments,
    _get_obs_rep,
    _set_obs_rep,
    dematrix
)

# Local implementation of check_array from sklearn
def check_array(
    array,
    accept_sparse=False,
    dtype=None,
    copy=False,
    **kwargs
):
    """
    Simple check_array implementation for sparse matrices.

    This is a simplified version that handles the specific use cases in this file.
    For full functionality, use sklearn.utils.validation.check_array.
    """
    from scipy import sparse

    if accept_sparse:
        # Check if it's a sparse matrix
        if not sparse.issparse(array):
            msg = f"Expected sparse matrix, got {type(array)}"
            raise ValueError(msg)

        # Check sparse format
        if isinstance(accept_sparse, (tuple, list)):
            allowed_formats = accept_sparse
            if not any(isinstance(array, getattr(sparse, f"{fmt}_matrix")) or
                      isinstance(array, getattr(sparse, f"{fmt}_array", type(None)))
                      for fmt in allowed_formats):
                msg = f"Sparse matrix format {array.format} not in allowed formats {allowed_formats}"
                raise ValueError(msg)

    # Handle dtype
    if dtype is not None:
        if not isinstance(dtype, (tuple, list)):
            dtype = (dtype,)
        if array.dtype not in dtype:
            if copy:
                array = array.astype(dtype[0])
            else:
                msg = f"Array dtype {array.dtype} not in allowed dtypes {dtype}"
                raise ValueError(msg)

    # Handle copy
    if copy and not sparse.issparse(array):
        array = array.copy()

    return array

try:
    import dask
    import dask.array as da
except ImportError:
    da = None
    dask = None

if TYPE_CHECKING:
    from anndata import AnnData


def _compute_nnz_median(counts: np.ndarray | DaskArray) -> np.floating:
    """Given a 1D array of counts, compute the median of the non-zero counts."""
    if isinstance(counts, DaskArray):
        counts = counts.compute()
    counts_greater_than_zero = counts[counts > 0]
    median = np.median(counts_greater_than_zero)
    return median



def _normalize_csr(
    mat: CSRBase,
    *,
    rows,
    columns,
    exclude_highly_expressed: bool = False,
    max_fraction: float = 0.05,
    n_threads: int = 10,
):
    """For sparse CSR matrix, compute the normalization factors."""
    counts_per_cell = np.zeros(rows, dtype=mat.data.dtype)
    counts_per_cols = np.zeros(columns, dtype=np.int32)
    
    for i in numba.prange(rows):
        count = 0.0
        for j in range(mat.indptr[i], mat.indptr[i + 1]):
            count += mat.data[j]
        counts_per_cell[i] = count
    
    if exclude_highly_expressed:
        counts_per_cols_t = np.zeros((n_threads, columns), dtype=np.int32)

        for i in numba.prange(n_threads):
            for r in range(i, rows, n_threads):
                for j in range(mat.indptr[r], mat.indptr[r + 1]):
                    if mat.data[j] > max_fraction * counts_per_cell[r]:
                        minor_index = mat.indices[j]
                        counts_per_cols_t[i, minor_index] += 1
        for c in numba.prange(columns):
            counts_per_cols[c] = counts_per_cols_t[:, c].sum()

        for i in numba.prange(rows):
            count = 0.0
            for j in range(mat.indptr[i], mat.indptr[i + 1]):
                if counts_per_cols[mat.indices[j]] == 0:
                    count += mat.data[j]
            counts_per_cell[i] = count

    return counts_per_cell, counts_per_cols


def _normalize_total_helper(
    x: np.ndarray | CSBase | DaskArray,
    *,
    exclude_highly_expressed: bool,
    max_fraction: float,
    target_sum: float | None,
) -> tuple[np.ndarray | CSBase | DaskArray, np.ndarray, np.ndarray | None]:
    """Calculate the normalized data, counts per cell, and gene subset.

    Parameters
    ----------
    See `normalize_total` for details.

    Returns
    -------
    X
        The normalized data matrix.
    counts_per_cell
        The normalization factors used for each cell (counts / target_sum).
    gene_subset
        If `exclude_highly_expressed=True`, a boolean mask indicating which genes
        were not considered highly expressed. Otherwise, `None`.
    """
    gene_subset = None
    counts_per_cell = None
    if isinstance(x, CSRBase):
        n_threads = numba.get_num_threads()
        counts_per_cell, counts_per_cols = _normalize_csr(
            x,
            rows=x.shape[0],
            columns=x.shape[1],
            exclude_highly_expressed=exclude_highly_expressed,
            max_fraction=max_fraction,
            n_threads=n_threads,
        )
        if target_sum is None:
            target_sum = _compute_nnz_median(counts_per_cell)
        if exclude_highly_expressed:
            gene_subset = counts_per_cols == 0
    else:
        counts_per_cell = axis_sum(x, axis=1)
        if exclude_highly_expressed:
            # at least one cell as more than max_fraction of counts per cell
            hi_exp = dematrix(x > counts_per_cell[:, None] * max_fraction)
            gene_subset = axis_sum(hi_exp, axis=0) == 0

            counts_per_cell = axis_sum(x[:, gene_subset], axis=1)
        if target_sum is None:
            target_sum = _compute_nnz_median(counts_per_cell)

    counts_per_cell = counts_per_cell / target_sum
    out = x if isinstance(x, np.ndarray | CSBase) else None
    x = axis_mul_or_truediv(
        x, counts_per_cell, op=truediv, out=out, allow_divide_by_zero=False, axis=0
    )
    return x, counts_per_cell, gene_subset


@old_positionals(
    "target_sum",
    "exclude_highly_expressed",
    "max_fraction",
    "key_added",
    "layer",
    "inplace",
    "copy",
)
def normalize_total_old(  # noqa: PLR0912
    adata: AnnData,
    *,
    target_sum: float | None = None,
    exclude_highly_expressed: bool = False,
    max_fraction: float = 0.05,
    key_added: str | None = None,
    layer: str | None = None,
    inplace: bool = True,
    copy: bool = False,
) -> AnnData | dict[str, np.ndarray] | None:
    """Normalize counts per cell.

    Normalize each cell by total counts over all genes,
    so that every cell has the same total count after normalization.
    If choosing `target_sum=1e6`, this is CPM normalization.

    If `exclude_highly_expressed=True`, very highly expressed genes are excluded
    from the computation of the normalization factor (size factor) for each
    cell. This is meaningful as these can strongly influence the resulting
    normalized values for all other genes :cite:p:`Weinreb2017`.

    Similar functions are used, for example, by Seurat :cite:p:`Satija2015`, Cell Ranger
    :cite:p:`Zheng2017` or SPRING :cite:p:`Weinreb2017`.

    .. note::
        When used with a :class:`~dask.array.Array` in `adata.X`, this function will have to
        call functions that trigger `.compute()` on the :class:`~dask.array.Array` if `exclude_highly_expressed`
        is `True`, `layer_norm` is not `None`, or if `key_added` is not `None`.

    Parameters
    ----------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    target_sum
        If `None`, after normalization, each observation (cell) has a total
        count equal to the median of total counts for observations (cells)
        before normalization.
    exclude_highly_expressed
        Exclude (very) highly expressed genes for the computation of the
        normalization factor (size factor) for each cell. A gene is considered
        highly expressed, if it has more than `max_fraction` of the total counts
        in at least one cell. The not-excluded genes will sum up to
        `target_sum`.  Providing this argument when `adata.X` is a :class:`~dask.array.Array`
        will incur blocking `.compute()` calls on the array.
    max_fraction
        If `exclude_highly_expressed=True`, consider cells as highly expressed
        that have more counts than `max_fraction` of the original total counts
        in at least one cell.
    key_added
        Name of the field in `adata.obs` where the normalization factor is
        stored.
    layer
        Layer to normalize instead of `X`. If `None`, `X` is normalized.
    inplace
        Whether to update `adata` or return dictionary with normalized copies of
        `adata.X` and `adata.layers`.
    copy
        Whether to modify copied input object. Not compatible with inplace=False.

    Returns
    -------
    Returns dictionary with normalized copies of `adata.X` and `adata.layers`
    or updates `adata` with normalized version of the original
    `adata.X` and `adata.layers`, depending on `inplace`.

    Example
    -------
    >>> import sys
    >>> from anndata import AnnData
    >>> import scanpy as sc
    >>> sc.settings.verbosity = "info"
    >>> sc.settings.logfile = sys.stdout  # for doctests
    >>> np.set_printoptions(precision=2)
    >>> adata = AnnData(
    ...     np.array(
    ...         [
    ...             [3, 3, 3, 6, 6],
    ...             [1, 1, 1, 2, 2],
    ...             [1, 22, 1, 2, 2],
    ...         ],
    ...         dtype="float32",
    ...     )
    ... )
    >>> adata.X
    array([[ 3.,  3.,  3.,  6.,  6.],
           [ 1.,  1.,  1.,  2.,  2.],
           [ 1., 22.,  1.,  2.,  2.]], dtype=float32)
    >>> X_norm = sc.pp.normalize_total(adata, target_sum=1, inplace=False)["X"]
    normalizing counts per cell
        finished (0:00:00)
    >>> X_norm
    array([[0.14, 0.14, 0.14, 0.29, 0.29],
           [0.14, 0.14, 0.14, 0.29, 0.29],
           [0.04, 0.79, 0.04, 0.07, 0.07]], dtype=float32)
    >>> X_norm = sc.pp.normalize_total(
    ...     adata,
    ...     target_sum=1,
    ...     exclude_highly_expressed=True,
    ...     max_fraction=0.2,
    ...     inplace=False,
    ... )["X"]
    normalizing counts per cell
    The following highly-expressed genes are not considered during normalization factor computation:
    ['1', '3', '4']
        finished (0:00:00)
    >>> X_norm
    array([[ 0.5,  0.5,  0.5,  1. ,  1. ],
           [ 0.5,  0.5,  0.5,  1. ,  1. ],
           [ 0.5, 11. ,  0.5,  1. ,  1. ]], dtype=float32)

    """
    if copy:
        if not inplace:
            msg = "`copy=True` cannot be used with `inplace=False`."
            raise ValueError(msg)
        adata = adata.copy()

    if max_fraction < 0 or max_fraction > 1:
        msg = "Choose max_fraction between 0 and 1."
        raise ValueError(msg)

    view_to_actual(adata)

    x = _get_obs_rep(adata, layer=layer)
    if x is None:
        msg = f"Layer {layer!r} not found in adata."
        raise ValueError(msg)
    from ._qc import _is_rust_backend
    is_rust = _is_rust_backend(adata)
    if is_rust:
        x=x[:]
    if isinstance(x, CSCBase):
        x = x.tocsr()
    if not inplace:
        x = x.copy()
    if issubclass(x.dtype.type, int | np.integer):
        x = x.astype(np.float32)  # TODO: Check if float64 should be used

    print(f"\n{Colors.HEADER}{Colors.BOLD}{EMOJI['start']} Count Normalization:{Colors.ENDC}")
    print(f"   {Colors.CYAN}Target sum: {Colors.BOLD}{target_sum if target_sum is not None else 'median'}{Colors.ENDC}")
    print(f"   {Colors.CYAN}Exclude highly expressed: {Colors.BOLD}{exclude_highly_expressed}{Colors.ENDC}")
    if exclude_highly_expressed:
        print(f"   {Colors.CYAN}Max fraction threshold: {Colors.BOLD}{max_fraction}{Colors.ENDC}")
    start = datetime.now()

    x, counts_per_cell, gene_subset = _normalize_total_helper(
        x,
        exclude_highly_expressed=exclude_highly_expressed,
        max_fraction=max_fraction,
        target_sum=target_sum,
    )

    if exclude_highly_expressed:
        n_excluded = (~gene_subset).sum()
        print(f"   {EMOJI['warning']} {Colors.WARNING}Excluding {Colors.BOLD}{n_excluded:,}{Colors.ENDC}{Colors.WARNING} highly-expressed genes from normalization computation{Colors.ENDC}")
        if n_excluded <= 10:  # Only show gene names if not too many
            if not is_rust:
                print(f"   {Colors.WARNING}Excluded genes: {Colors.BOLD}{adata.var_names[~gene_subset].tolist()}{Colors.ENDC}")
            else:
                print(f"   {Colors.WARNING}Excluded genes: {Colors.BOLD}{np.array(adata.var_names)[gene_subset]}{Colors.ENDC}")
            #print(f"   {Colors.WARNING}Excluded genes: {Colors.BOLD}{adata.var_names[~gene_subset].tolist()}{Colors.ENDC}")

    cell_subset = counts_per_cell > 0
    if not isinstance(cell_subset, DaskArray) and not np.all(cell_subset):
        n_zero = (~cell_subset).sum()
        print(f"   {EMOJI['warning']} {Colors.WARNING}Warning: {Colors.BOLD}{n_zero:,}{Colors.ENDC}{Colors.WARNING} cells have zero counts{Colors.ENDC}")
        warn("Some cells have zero counts", UserWarning, stacklevel=2)

    dat = dict(
        X=x,
        norm_factor=counts_per_cell,
    )
    if inplace:
        if key_added is not None:
            adata.obs[key_added] = dat["norm_factor"]
        _set_obs_rep(adata, dat["X"], layer=layer)

    elapsed_time = datetime.now() - start
    print(f"\n{Colors.GREEN}{EMOJI['done']} Count Normalization Completed Successfully!{Colors.ENDC}")
    print(f"   {Colors.GREEN}✓ Processed: {Colors.BOLD}{adata.shape[0]:,}{Colors.ENDC}{Colors.GREEN} cells × {Colors.BOLD}{adata.shape[1]:,}{Colors.ENDC}{Colors.GREEN} genes{Colors.ENDC}")
    print(f"   {Colors.GREEN}✓ Runtime: {Colors.BOLD}{elapsed_time.total_seconds():.2f}s{Colors.ENDC}")
    if key_added is not None:
        print(f"   {Colors.CYAN}• Added '{Colors.BOLD}{key_added}{Colors.ENDC}{Colors.CYAN}': normalization factors (adata.obs){Colors.ENDC}")

    if copy:
        return adata
    elif not inplace:
        return dat
    return None
    

def _log1p(
    data: AnnData | np.ndarray | CSBase,
    *,
    base: Number | None = None,
    copy: bool = False,
    chunked: bool | None = None,
    chunk_size: int | None = None,
    layer: str | None = None,
    obsm: str | None = None,
) -> AnnData | np.ndarray | CSBase | None:
    r"""Logarithmize the data matrix.

    Computes :math:`X = \log(X + 1)`,
    where :math:`log` denotes the natural logarithm unless a different base is given.

    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    base
        Base of the logarithm. Natural logarithm is used by default.
    copy
        If an :class:`~anndata.AnnData` is passed, determines whether a copy
        is returned.
    chunked
        Process the data matrix in chunks, which will save memory.
        Applies only to :class:`~anndata.AnnData`.
    chunk_size
        `n_obs` of the chunks to process the data in.
    layer
        Entry of layers to transform.
    obsm
        Entry of obsm to transform.

    Returns
    -------
    Returns or updates `data`, depending on `copy`.

    """
    _check_array_function_arguments(
        chunked=chunked, chunk_size=chunk_size, layer=layer, obsm=obsm
    )
    return log1p_array(data, copy=copy, base=base)



def log1p_sparse(x: CSBase, *, base: Number | None = None, copy: bool = False):
    x = check_array(
        x, accept_sparse=("csr", "csc"), dtype=(np.float64, np.float32), copy=copy
    )
    x.data = _log1p(x.data, copy=False, base=base)
    return x



def log1p_array(x: np.ndarray, *, base: Number | None = None, copy: bool = False):
    # Can force arrays to be np.ndarrays, but would be useful to not
    # X = check_array(X, dtype=(np.float64, np.float32), ensure_2d=False, copy=copy)
    
    if copy:
        x = x.astype(float) if not np.issubdtype(x.dtype, np.floating) else x.copy()
    elif not (np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, complex)):
        x = x.astype(float)
    x=np.log1p(x)
    if base is not None:
        x=np.divide(x, np.log(base))
    return x

@register_function(
    aliases=["log1p", "对数归一化", "log_transform", "log变换", "对数转换"],
    category="preprocessing",
    description="Apply log(1+x) transform to expression matrix or selected layer/obsm",
    prerequisites={
        "optional_functions": ["normalize_total", "qc"]
    },
    requires={},
    produces={
        "uns": ["log1p"]
    },
    auto_fix="none",
    examples=[
        "ov.pp.log1p(adata)",
        "ov.pp.log1p(adata, layer='counts')",
        "adata = ov.pp.log1p(adata, copy=True)",
    ],
    related=["normalize_total", "highly_variable_genes", "preprocess"],
)
def log1p(
    adata: AnnData,
    *,
    base: Number | None = None,
    copy: bool = False,
    chunked: bool = False,
    chunk_size: int | None = None,
    layer: str | None = None,
    obsm: str | None = None,
) -> AnnData | None:
    """Log-transform expression values with ``log(1 + x)``.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression values.
    base : Number, optional
        Logarithm base. Uses natural logarithm when ``None``.
    copy : bool, default=False
        If ``True``, return a transformed copy instead of modifying in place.
    chunked : bool, default=False
        Whether to transform ``adata.X`` by chunks for memory efficiency.
    chunk_size : int, optional
        Number of cells per chunk when ``chunked=True``.
    layer : str, optional
        Layer name to transform instead of ``adata.X``.
    obsm : str, optional
        ``adata.obsm`` key to transform instead of ``adata.X``.

    Returns
    -------
    AnnData or None
        Returns a transformed AnnData when ``copy=True``; otherwise modifies
        ``adata`` in place and returns ``None``.
    """
    if "log1p" in adata.uns:
        print(f"{Colors.WARNING}adata.X seems to be already log-transformed.{Colors.ENDC}")

    adata = adata.copy() if copy else adata
    view_to_actual(adata)

    from ._qc import _is_rust_backend
    is_rust = _is_rust_backend(adata)
    
    if chunked:
        if (layer is not None) or (obsm is not None):
            msg = (
                "Currently cannot perform chunked operations on arrays not stored in X."
            )
            raise NotImplementedError(msg)
        if adata.isbacked and adata.file._filemode != "r+":
            msg = "log1p is not implemented for backed AnnData with backed mode not r+"
            raise NotImplementedError(msg)
        for chunk, start, end in adata.chunked_X(chunk_size):
            adata.X[start:end] = log1p(chunk, base=base, copy=False)
    else:
        x = _get_obs_rep(adata, layer=layer, obsm=obsm)
        if is_rust:
            x = x[:]
        if is_backed_type(x):
            msg = f"log1p is not implemented for matrices of type {type(x)}"
            if layer is not None:
                msg = f"{msg} from layers"
                raise NotImplementedError(msg)
            msg = f"{msg} without `chunked=True`"
            raise NotImplementedError(msg)
        x = _log1p(x, copy=False, base=base)
        _set_obs_rep(adata, x, layer=layer, obsm=obsm)
    # Set base to np.e if None (natural logarithm)
    if base is None:
        base = np.e
    adata.uns["log1p"] = {"base": base}
    if copy:
        return adata




# Removed duplicate imports - already imported at the top of the file
from scipy.sparse import issparse

try:
    import dask
    import dask.array as da
except ImportError:
    da = None
    dask = None

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Literal

    from anndata import AnnData


def _normalize_data(X, counts, after=None, *, copy: bool = False):
    X = X.copy() if copy else X
    if issubclass(X.dtype.type, int | np.integer):
        X = X.astype(np.float32)  # TODO: Check if float64 should be used
    if after is None:
        if isinstance(counts, DaskArray):

            def nonzero_median(x):
                return np.ma.median(np.ma.masked_array(x, x == 0)).item()

            after = da.from_delayed(
                dask.delayed(nonzero_median)(counts),
                shape=(),
                meta=counts._meta,
                dtype=counts.dtype,
            )
        else:
            counts_greater_than_zero = counts[counts > 0]
            after = np.median(counts_greater_than_zero, axis=0)
    counts = counts / after
    return axis_mul_or_truediv(
        X,
        counts,
        op=truediv,
        out=X if isinstance(X, np.ndarray) or issparse(X) else None,
        allow_divide_by_zero=False,
        axis=0,
    )


@old_positionals(
    "target_sum",
    "exclude_highly_expressed",
    "max_fraction",
    "key_added",
    "layer",
    "layers",
    "layer_norm",
    "inplace",
    "copy",
)
def normalize_total(
    adata: AnnData,
    *,
    target_sum: float | None = None,
    exclude_highly_expressed: bool = False,
    max_fraction: float = 0.05,
    key_added: str | None = None,
    layer: str | None = None,
    layers: Literal["all"] | Iterable[str] | None = None,
    layer_norm: str | None = None,
    inplace: bool = True,
    copy: bool = False,
) -> AnnData | dict[str, np.ndarray] | None:
    """\
    Normalize counts per cell.

    Normalize each cell by total counts over all genes,
    so that every cell has the same total count after normalization.
    If choosing `target_sum=1e6`, this is CPM normalization.

    If `exclude_highly_expressed=True`, very highly expressed genes are excluded
    from the computation of the normalization factor (size factor) for each
    cell. This is meaningful as these can strongly influence the resulting
    normalized values for all other genes :cite:p:`Weinreb2017`.

    Similar functions are used, for example, by Seurat :cite:p:`Satija2015`, Cell Ranger
    :cite:p:`Zheng2017` or SPRING :cite:p:`Weinreb2017`.

    .. note::
        When used with a :class:`~dask.array.Array` in `adata.X`, this function will have to
        call functions that trigger `.compute()` on the :class:`~dask.array.Array` if `exclude_highly_expressed`
        is `True`, `layer_norm` is not `None`, or if `key_added` is not `None`.

    Params
    ------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    target_sum
        If `None`, after normalization, each observation (cell) has a total
        count equal to the median of total counts for observations (cells)
        before normalization.
    exclude_highly_expressed
        Exclude (very) highly expressed genes for the computation of the
        normalization factor (size factor) for each cell. A gene is considered
        highly expressed, if it has more than `max_fraction` of the total counts
        in at least one cell. The not-excluded genes will sum up to
        `target_sum`.  Providing this argument when `adata.X` is a :class:`~dask.array.Array`
        will incur blocking `.compute()` calls on the array.
    max_fraction
        If `exclude_highly_expressed=True`, consider cells as highly expressed
        that have more counts than `max_fraction` of the original total counts
        in at least one cell.
    key_added
        Name of the field in `adata.obs` where the normalization factor is
        stored.
    layer
        Layer to normalize instead of `X`. If `None`, `X` is normalized.
    inplace
        Whether to update `adata` or return dictionary with normalized copies of
        `adata.X` and `adata.layers`.
    copy
        Whether to modify copied input object. Not compatible with inplace=False.

    Returns
    -------
    Returns dictionary with normalized copies of `adata.X` and `adata.layers`
    or updates `adata` with normalized version of the original
    `adata.X` and `adata.layers`, depending on `inplace`.

    Example
    --------
    >>> import sys
    >>> from anndata import AnnData
    >>> import scanpy as sc
    >>> sc.settings.verbosity = 'info'
    >>> sc.settings.logfile = sys.stdout  # for doctests
    >>> np.set_printoptions(precision=2)
    >>> adata = AnnData(np.array([
    ...     [3, 3, 3, 6, 6],
    ...     [1, 1, 1, 2, 2],
    ...     [1, 22, 1, 2, 2],
    ... ], dtype='float32'))
    >>> adata.X
    array([[ 3.,  3.,  3.,  6.,  6.],
           [ 1.,  1.,  1.,  2.,  2.],
           [ 1., 22.,  1.,  2.,  2.]], dtype=float32)
    >>> X_norm = sc.pp.normalize_total(adata, target_sum=1, inplace=False)['X']
    normalizing counts per cell
        finished (0:00:00)
    >>> X_norm
    array([[0.14, 0.14, 0.14, 0.29, 0.29],
           [0.14, 0.14, 0.14, 0.29, 0.29],
           [0.04, 0.79, 0.04, 0.07, 0.07]], dtype=float32)
    >>> X_norm = sc.pp.normalize_total(
    ...     adata, target_sum=1, exclude_highly_expressed=True,
    ...     max_fraction=0.2, inplace=False
    ... )['X']
    normalizing counts per cell. The following highly-expressed genes are not considered during normalization factor computation:
    ['1', '3', '4']
        finished (0:00:00)
    >>> X_norm
    array([[ 0.5,  0.5,  0.5,  1. ,  1. ],
           [ 0.5,  0.5,  0.5,  1. ,  1. ],
           [ 0.5, 11. ,  0.5,  1. ,  1. ]], dtype=float32)
    """
    if copy:
        if not inplace:
            msg = "`copy=True` cannot be used with `inplace=False`."
            raise ValueError(msg)
        adata = adata.copy()

    if max_fraction < 0 or max_fraction > 1:
        msg = "Choose max_fraction between 0 and 1."
        raise ValueError(msg)

    # Deprecated features
    if layers is not None:
        warn(
            FutureWarning(
                "The `layers` argument is deprecated. Instead, specify individual "
                "layers to normalize with `layer`."
            )
        )
    if layer_norm is not None:
        warn(
            FutureWarning(
                "The `layer_norm` argument is deprecated. Specify the target size "
                "factor directly with `target_sum`."
            )
        )

    if layers == "all":
        layers = adata.layers.keys()
    elif isinstance(layers, str):
        msg = f"`layers` needs to be a list of strings or 'all', not {layers!r}"
        raise ValueError(msg)

    view_to_actual(adata)

    x = _get_obs_rep(adata, layer=layer)
    from ._qc import _is_rust_backend
    is_rust = _is_rust_backend(adata)
    if is_rust:
        x=x[:]

    print(f"\n{Colors.HEADER}{Colors.BOLD}{EMOJI['start']} Count Normalization:{Colors.ENDC}")
    print(f"   {Colors.CYAN}Target sum: {Colors.BOLD}{target_sum if target_sum is not None else 'median'}{Colors.ENDC}")
    print(f"   {Colors.CYAN}Exclude highly expressed: {Colors.BOLD}{exclude_highly_expressed}{Colors.ENDC}")
    if exclude_highly_expressed:
        print(f"   {Colors.CYAN}Max fraction threshold: {Colors.BOLD}{max_fraction}{Colors.ENDC}")
    start = datetime.now()

    gene_subset = None
    counts_per_cell = axis_sum(x, axis=1)

    # ── Side product for downstream Pearson HVG ───────────────────────
    # `omicverse.pp.experimental._highly_variable_pearson_residuals` runs
    # its own first pass over X (`layer='counts'`) to gather per-gene
    # totals + per-cell totals + sum_total. On a 1M cells × 60606 genes
    # matrix that is the dominant single cost of the preprocess pipeline
    # (TS-1M: ~30% of 1874 s). We grab those quantities here as a side
    # product of the existing scan and stash them in `adata.uns` so the
    # HVG step can auto-skip its Pass 1. Same fusion that
    # `anndataoom.chunked_normalize_total` does for the OOM backend.
    # Only applies to the global (no-batch) case -- the per-batch HVG path
    # needs per-batch sums that this scan does not split out.
    _pre_x = x
    if hasattr(_pre_x, "tocsr") and not isinstance(_pre_x, np.ndarray):
        try:
            # Try a sparse-native sum-axis-0 / squared-sum-axis-0; both
            # are O(nnz) on a CSR / CSC. Fall back silently if the input
            # is a backed type we cannot stream cheaply.
            _sg = np.asarray(_pre_x.sum(axis=0)).ravel().astype(np.float64)
            _sq = np.asarray(
                _pre_x.multiply(_pre_x).sum(axis=0)
            ).ravel().astype(np.float64)
            _pearson_precompute = {
                "sums_genes": _sg,
                "sq_sums_genes": _sq,
                "sum_total": float(_sg.sum()),
                "n_obs": int(_pre_x.shape[0]),
                "n_vars": int(_pre_x.shape[1]),
            }
        except Exception:
            _pearson_precompute = None
    else:
        try:
            _arr = np.asarray(_pre_x, dtype=np.float64)
            _sg = _arr.sum(axis=0).ravel()
            _sq = (_arr ** 2).sum(axis=0).ravel()
            _pearson_precompute = {
                "sums_genes": _sg,
                "sq_sums_genes": _sq,
                "sum_total": float(_sg.sum()),
                "n_obs": int(_pre_x.shape[0]),
                "n_vars": int(_pre_x.shape[1]),
            }
        except Exception:
            _pearson_precompute = None

    if exclude_highly_expressed:
        counts_per_cell = np.ravel(counts_per_cell)

        # at least one cell as more than max_fraction of counts per cell

        gene_subset = axis_sum((x > counts_per_cell[:, None] * max_fraction), axis=0)
        gene_subset = np.asarray(np.ravel(gene_subset) == 0)

        n_excluded = (~gene_subset).sum()
        print(f"   {EMOJI['warning']} {Colors.WARNING}Excluding {Colors.BOLD}{n_excluded:,}{Colors.ENDC}{Colors.WARNING} highly-expressed genes from normalization computation{Colors.ENDC}")
        if n_excluded <= 10:  # Only show gene names if not too many
            if is_rust:
                # Convert to numpy array for compatibility with Rust backend
                excluded_genes = np.array(adata.var_names)[~gene_subset].tolist()
                print(f"   {Colors.WARNING}Excluded genes: {Colors.BOLD}{excluded_genes}{Colors.ENDC}")
            else:
                print(f"   {Colors.WARNING}Excluded genes: {Colors.BOLD}{adata.var_names[~gene_subset].tolist()}{Colors.ENDC}")
        if is_rust:
            # Use integer indexing for Rust backend
            gene_indices = np.where(gene_subset)[0]
            counts_per_cell = axis_sum(x[:, gene_indices], axis=1)
        else:
            counts_per_cell = axis_sum(x[:, gene_subset], axis=1)
    counts_per_cell = np.ravel(counts_per_cell)

    cell_subset = counts_per_cell > 0
    if not isinstance(cell_subset, DaskArray) and not np.all(cell_subset):
        n_zero = (~cell_subset).sum()
        print(f"   {EMOJI['warning']} {Colors.WARNING}Warning: {Colors.BOLD}{n_zero:,}{Colors.ENDC}{Colors.WARNING} cells have zero counts{Colors.ENDC}")
        warn(UserWarning("Some cells have zero counts"))

    # Compute target_sum if not provided (for norm_factor calculation)
    if target_sum is None:
        if isinstance(counts_per_cell, DaskArray):
            def nonzero_median(x):
                return np.ma.median(np.ma.masked_array(x, x == 0)).item()
            import dask
            import dask.array as da
            target_sum = da.from_delayed(
                dask.delayed(nonzero_median)(counts_per_cell),
                shape=(),
                meta=counts_per_cell._meta,
                dtype=counts_per_cell.dtype,
            )
            if isinstance(target_sum, DaskArray):
                target_sum = target_sum.compute()
        else:
            counts_greater_than_zero = counts_per_cell[counts_per_cell > 0]
            target_sum = np.median(counts_greater_than_zero, axis=0)

    if inplace:
        if key_added is not None:
            # Save normalization factor (counts / target_sum), consistent with scanpy
            adata.obs[key_added] = counts_per_cell / target_sum
        # Finalise the Pearson-HVG precompute: needs per-cell sums on the
        # FULL (unfiltered) gene panel — that is what `axis_sum(x, axis=1)`
        # produced before `exclude_highly_expressed` overwrote
        # `counts_per_cell` with the filtered version. Recompute from
        # `_pearson_precompute['sums_genes']` if available, else from the
        # current `counts_per_cell` snapshot (which is right when there is
        # no highly-expressed-gene exclusion).
        if _pearson_precompute is not None:
            # Per-cell sums on FULL panel = row marginals of the
            # un-normalised X. The `counts_per_cell` we have here may be
            # the filtered version; for the no-exclude case it is also
            # the full-panel one. For the exclude case the precompute
            # would have to recompute the full-panel row sums; we skip
            # the precompute stash entirely in that branch rather than
            # ship subtly wrong stats.
            if not exclude_highly_expressed:
                _pearson_precompute["sums_cells"] = np.asarray(
                    counts_per_cell, dtype=np.float64
                ).copy()
                adata.uns["_pearson_precompute"] = _pearson_precompute
            # else: caller will fall back to two-pass HVG.
        _set_obs_rep(
            adata, _normalize_data(x, counts_per_cell, target_sum), layer=layer
        )
    else:
        # not recarray because need to support sparse
        dat = dict(
            X=_normalize_data(x, counts_per_cell, target_sum, copy=True),
            norm_factor=counts_per_cell / target_sum,  # Save normalization factor, consistent with scanpy
        )

    # Deprecated features
    if layer_norm == "after":
        after = target_sum
    elif layer_norm == "X":
        if is_rust:
            # Use integer indexing for Rust backend
            cell_indices = np.where(cell_subset)[0]
            after = np.median(counts_per_cell[cell_indices])
        else:
            after = np.median(counts_per_cell[cell_subset])
    elif layer_norm is None:
        after = None
    else:
        msg = 'layer_norm should be "after", "X" or None'
        raise ValueError(msg)

    for layer_to_norm in layers if layers is not None else ():
        res = normalize_total(
            adata, layer=layer_to_norm, target_sum=after, inplace=inplace
        )
        if not inplace:
            dat[layer_to_norm] = res["X"]

    elapsed_time = datetime.now() - start
    print(f"\n{Colors.GREEN}{EMOJI['done']} Count Normalization Completed Successfully!{Colors.ENDC}")
    print(f"   {Colors.GREEN}✓ Processed: {Colors.BOLD}{adata.shape[0]:,}{Colors.ENDC}{Colors.GREEN} cells × {Colors.BOLD}{adata.shape[1]:,}{Colors.ENDC}{Colors.GREEN} genes{Colors.ENDC}")
    print(f"   {Colors.GREEN}✓ Runtime: {Colors.BOLD}{elapsed_time.total_seconds():.2f}s{Colors.ENDC}")
    if key_added is not None:
        print(f"   {Colors.CYAN}• Added '{Colors.BOLD}{key_added}{Colors.ENDC}{Colors.CYAN}': normalization factors (adata.obs){Colors.ENDC}")

    if copy:
        return adata
    elif not inplace:
        return dat

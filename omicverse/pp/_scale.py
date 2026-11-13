from __future__ import annotations

import warnings
from functools import singledispatch, wraps
from operator import mul, truediv
from typing import TYPE_CHECKING, Literal, Any, TypedDict
import logging

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

import numba
import numpy as np
from anndata import AnnData
import h5py
from anndata._core.sparse_dataset import BaseCompressedSparseDataset
#from fast_array_utils.stats import mean_var
from ._utils import axis_mean, sparse_mean_variance_axis,_get_mean_var

from ._compat import CSBase, CSCBase, CSRBase, DaskArray, njit, old_positionals
from scipy import sparse

from ._qc import _is_rust_backend

# Use standard warnings.warn instead of scanpy's warn
warn = warnings.warn

# install dask if available
try:
    import dask.array as da
except ImportError:
    da = None

if TYPE_CHECKING:
    from typing import TypeVar, Callable
    from collections.abc import Collection, Iterable, Mapping, KeysView
    from pathlib import Path
    from pandas._typing import Dtype as PdDtype
    from numpy.typing import ArrayLike, NDArray
    from anndata._core.views import ArrayView

    _A = TypeVar("_A", bound=CSBase | np.ndarray | DaskArray)

# Create a simple logger for this module
logg = logging.getLogger(__name__)


# ================================================================================
# Utility functions from scanpy (to remove scanpy dependency)
# ================================================================================

def renamed_arg(old_name, new_name, *, pos_0: bool = False):
    """Decorator to handle renamed function arguments."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            __tracebackhide__ = True
            if old_name in kwargs:
                f_name = func.__name__
                pos_str = (
                    (
                        f" at first position. Call it as `{f_name}(val, ...)` "
                        f"instead of `{f_name}({old_name}=val, ...)`"
                    )
                    if pos_0
                    else ""
                )
                msg = (
                    f"In function `{f_name}`, argument `{old_name}` "
                    f"was renamed to `{new_name}`{pos_str}."
                )
                warn(msg, FutureWarning)
                if pos_0:
                    args = (kwargs.pop(old_name), *args)
                else:
                    kwargs[new_name] = kwargs.pop(old_name)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _check_array_function_arguments(**kwargs):
    """Check for invalid arguments when an array is passed.

    Helper for functions that work on either AnnData objects or array-likes.
    """
    invalid_args = [k for k, v in kwargs.items() if v is not None]
    if len(invalid_args) > 0:
        msg = f"Arguments {invalid_args} are only valid if an AnnData object is passed."
        raise TypeError(msg)


# Alias for backwards compatibility
check_array_function_arguments = _check_array_function_arguments


def is_backed_type(x: object, /) -> bool:
    """Check if x is a backed type."""
    return isinstance(x, BaseCompressedSparseDataset | h5py.File | h5py.Dataset)


@singledispatch
def axis_sum(x: ArrayLike, /, *, axis: Literal[0, 1]) -> np.ndarray:
    """
    Sum array along an axis.

    Arguments:
        x: Array to sum.
        axis: Axis along which to sum.

    Returns:
        Array of sums.
    """
    return np.sum(x, axis=axis)


@axis_sum.register(sparse.csr_matrix)
@axis_sum.register(sparse.csc_matrix)
def _axis_sum_sparse(x: CSBase, /, *, axis: Literal[0, 1]) -> np.ndarray:
    """Sum sparse matrix along an axis."""
    return np.asarray(x.sum(axis=axis)).flatten()


# Register for sparse array types (scipy >= 1.8)
try:
    @axis_sum.register(sparse.csr_array)
    @axis_sum.register(sparse.csc_array)
    def _axis_sum_sparse_array(x: CSBase, /, *, axis: Literal[0, 1]) -> np.ndarray:
        """Sum sparse array along an axis."""
        return np.asarray(x.sum(axis=axis)).flatten()
except AttributeError:
    # sparse arrays not available in older scipy versions
    pass


if da is not None:
    @axis_sum.register(da.Array)
    def _axis_sum_dask(x: DaskArray, /, *, axis: Literal[0, 1]) -> DaskArray:
        """Sum dask array along an axis."""
        return x.sum(axis=axis)


def raise_not_implemented_error_if_backed_type(x: object, method_name: str, /) -> None:
    """Raise error if x is a backed type."""
    if is_backed_type(x):
        msg = f"{method_name} is not implemented for matrices of type {type(x)}"
        raise NotImplementedError(msg)


def view_to_actual(adata: AnnData) -> None:
    """Convert AnnData view to actual."""
    if adata.is_view:
        msg = "Received a view of an AnnData. Making a copy."
        warn(msg, UserWarning)
        adata._init_as_actual(adata.copy())


def _broadcast_axis(divisor: np.ndarray | DaskArray, axis: Literal[0, 1]) -> np.ndarray | DaskArray:
    """Broadcast divisor along axis."""
    divisor = np.ravel(divisor)
    if axis:
        return divisor[None, :]
    return divisor[:, None]


def _check_op(op) -> None:
    """Check if op is valid."""
    if op not in {truediv, mul}:
        msg = f"{op} not one of truediv or mul"
        raise ValueError(msg)


@singledispatch
def axis_mul_or_truediv(
    x: ArrayLike,
    /,
    scaling_array: np.ndarray,
    axis: Literal[0, 1],
    op: Callable[[Any, Any], Any],
    *,
    allow_divide_by_zero: bool = True,
    out: ArrayLike | None = None,
) -> np.ndarray:
    """Scale array by multiplying or dividing along an axis."""
    _check_op(op)
    scaling_array = _broadcast_axis(scaling_array, axis)
    if op is mul:
        return np.multiply(x, scaling_array, out=out)
    if not allow_divide_by_zero:
        scaling_array = scaling_array.copy() + (scaling_array == 0)
    return np.true_divide(x, scaling_array, out=out)


# Register for each sparse type individually (Python 3.10 doesn't support union types in register)
def _axis_mul_or_truediv_sparse(
    x: sparse.spmatrix | sparse.sparray,
    /,
    scaling_array: np.ndarray,
    axis: Literal[0, 1],
    op: Callable[[Any, Any], Any],
    *,
    allow_divide_by_zero: bool = True,
    out: sparse.spmatrix | sparse.sparray | None = None,
) -> sparse.spmatrix | sparse.sparray:
    """Scale sparse matrix by multiplying or dividing along an axis."""
    _check_op(op)
    if out is not None and x.data is not out.data:
        msg = "`out` argument provided but not equal to X.  This behavior is not supported for sparse matrix scaling."
        raise ValueError(msg)
    if not allow_divide_by_zero and op is truediv:
        scaling_array = scaling_array.copy() + (scaling_array == 0)

    row_scale = axis == 0
    column_scale = axis == 1
    if row_scale:

        def new_data_op(x):
            return op(x.data, np.repeat(scaling_array, np.diff(x.indptr)))

    elif column_scale:

        def new_data_op(x):
            return op(x.data, scaling_array.take(x.indices, mode="clip"))

    if x.format == "csr":
        indices = x.indices
        indptr = x.indptr
        if out is not None and row_scale and x.data.size >= 1 << 24:
            # large CSR, per-row factor: fill the result row block by row block
            # on all threads (same values and dtype as new_data_op)
            d, ip, f = x.data, indptr, np.asarray(scaling_array)
            try:  # one threaded C++ pass (anndataoom-fast), bit-identical
                from anndataoom_fast.memory import scale_rows as _scale_rows
            except ImportError:
                _scale_rows = None
            if _scale_rows is not None and d.dtype in (np.float32, np.float64) \
                    and f.dtype in (np.float32, np.float64):
                x.data = _scale_rows(x, f, divide=op is truediv)
                return x
            from ._qc import threaded_chunks
            res = np.empty(d.size, dtype=np.result_type(d.dtype, f.dtype))
            ufunc = np.true_divide if op is truediv else np.multiply

            def _rows(r0, r1):
                a, b = int(ip[r0]), int(ip[r1])
                ufunc(d[a:b], np.repeat(f[r0:r1], np.diff(ip[r0:r1 + 1])), out=res[a:b])

            threaded_chunks(_rows, x.shape[0], min_chunk=1 << 12)
            x.data = res
            return x
        if out is not None:
            x.data = new_data_op(x)
            return x
        return type(x)((new_data_op(x), indices.copy(), indptr.copy()), shape=x.shape)
    transposed = x.T
    return axis_mul_or_truediv(
        transposed,
        scaling_array,
        op=op,
        axis=1 - axis,
        out=transposed,
        allow_divide_by_zero=allow_divide_by_zero,
    ).T

# Register the sparse implementation for all sparse types
for sparse_type in [sparse.csr_matrix, sparse.csc_matrix, sparse.csr_array, sparse.csc_array]:
    axis_mul_or_truediv.register(sparse_type)(_axis_mul_or_truediv_sparse)


def _make_axis_chunks(
    x: DaskArray, axis: Literal[0, 1]
) -> tuple[tuple[int], tuple[int]]:
    """Make axis chunks for dask array."""
    if axis == 0:
        return (x.chunks[axis], (1,))
    return ((1,), x.chunks[axis])


@axis_mul_or_truediv.register(DaskArray)
def _(
    x: DaskArray,
    /,
    scaling_array: np.ndarray | DaskArray,
    axis: Literal[0, 1],
    op: Callable[[Any, Any], Any],
    *,
    allow_divide_by_zero: bool = True,
    out: None = None,
) -> DaskArray:
    """Scale dask array by multiplying or dividing along an axis."""
    _check_op(op)
    if out is not None:
        msg = "`out` is not `None`. Do not do in-place modifications on dask arrays."
        raise TypeError(msg)

    import dask.array as da

    scaling_array = _broadcast_axis(scaling_array, axis)
    row_scale = axis == 0
    column_scale = axis == 1

    if isinstance(scaling_array, DaskArray):
        if (row_scale and x.chunksize[0] != scaling_array.chunksize[0]) or (
            column_scale
            and (
                (
                    len(scaling_array.chunksize) == 1
                    and x.chunksize[1] != scaling_array.chunksize[0]
                )
                or (
                    len(scaling_array.chunksize) == 2
                    and x.chunksize[1] != scaling_array.chunksize[1]
                )
            )
        ):
            msg = "Rechunking scaling_array in user operation"
            warn(msg, UserWarning)
            scaling_array = scaling_array.rechunk(_make_axis_chunks(x, axis))
    else:
        scaling_array = da.from_array(
            scaling_array,
            chunks=_make_axis_chunks(x, axis),
        )
    return da.map_blocks(
        axis_mul_or_truediv,
        x,
        scaling_array,
        axis,
        op,
        meta=x._meta,
        out=out,
        allow_divide_by_zero=allow_divide_by_zero,
    )


class _ObsRep(TypedDict, total=False):
    """Type for observation representation choices."""
    use_raw: bool
    layer: str | None
    obsm: str | None
    obsp: str | None


def _get_obs_rep(
    adata: AnnData, **choices: Unpack[_ObsRep]
) -> (
    np.ndarray | CSBase | None
):
    """Choose array aligned with obs annotation."""
    if not isinstance(use_raw := choices.get("use_raw", False), bool):
        msg = f"use_raw expected to be bool, was {type(use_raw)}."
        raise TypeError(msg)
    assert choices.keys() <= {"layer", "use_raw", "obsm", "obsp"}

    match [(k, v) for k, v in choices.items() if v not in {None, False}]:
        case []:
            return adata.X
        case [("layer", layer)]:
            return adata.layers[layer]
        case [("use_raw", True)]:
            return adata.raw.X
        case [("obsm", obsm)]:
            return adata.obsm[obsm]
        case [("obsp", obsp)]:
            return adata.obsp[obsp]
        case _:
            valid = [f"`{k}`" for k in choices]
            valid[-1] = f"or {valid[-1]}"
            msg = f"Only one of {', '.join(valid)} can be specified."
            raise ValueError(msg)


def _set_obs_rep(
    adata: AnnData,
    val: Any,
    *,
    use_raw: bool = False,
    layer: str | None = None,
    obsm: str | None = None,
    obsp: str | None = None,
):
    """Set value for observation rep."""
    is_layer = layer is not None
    is_raw = use_raw is not False
    is_obsm = obsm is not None
    is_obsp = obsp is not None
    choices_made = sum((is_layer, is_raw, is_obsm, is_obsp))
    assert choices_made <= 1
    if choices_made == 0:
        adata.X = val
    elif is_layer:
        adata.layers[layer] = val
    elif use_raw:
        adata.raw.X = val
    elif is_obsm:
        adata.obsm[obsm] = val
    elif is_obsp:
        adata.obsp[obsp] = val
    else:
        msg = (
            "That was unexpected. Please report this bug at:\n\n"
            "\thttps://github.com/scverse/omicverse/issues"
        )
        raise AssertionError(msg)


def _check_mask(
    data: AnnData | np.ndarray | CSBase | DaskArray,
    mask: str | NDArray[np.bool_] | None,
    dim: Literal["obs", "var"],
    *,
    allow_probabilities: bool = False,
) -> NDArray[np.bool_] | None:
    """Validate mask argument.

    Params
    ------
    data
        Annotated data matrix or numpy array.
    mask
        Mask (or probabilities if `allow_probabilities=True`).
        Either an appropriatley sized array, or name of a column.
    dim
        The dimension being masked.
    allow_probabilities
        Whether to allow probabilities as `mask`
    """
    if mask is None:
        return mask
    desc = "mask/probabilities" if allow_probabilities else "mask"

    if isinstance(mask, str):
        if not isinstance(data, AnnData):
            msg = f"Cannot refer to {desc} with string without providing anndata object as argument"
            raise ValueError(msg)

        import pandas as pd
        annot: pd.DataFrame = getattr(data, dim)
        if mask not in annot.columns:
            msg = (
                f"Did not find `adata.{dim}[{mask!r}]`. "
                f"Either add the {desc} first to `adata.{dim}`"
                f"or consider using the {desc} argument with an array."
            )
            raise ValueError(msg)
        mask_array = annot[mask].to_numpy()
    else:
        if len(mask) != data.shape[0 if dim == "obs" else 1]:
            msg = f"The shape of the {desc} do not match the data."
            raise ValueError(msg)
        mask_array = mask

    import pandas as pd
    is_bool = pd.api.types.is_bool_dtype(mask_array.dtype)
    if not allow_probabilities and not is_bool:
        msg = "Mask array must be boolean."
        raise ValueError(msg)
    elif allow_probabilities and not (
        is_bool or pd.api.types.is_float_dtype(mask_array.dtype)
    ):
        msg = f"{desc} array must be boolean or floating point."
        raise ValueError(msg)

    return mask_array


# ================================================================================
# Main scaling functions
# ================================================================================

@singledispatch
def clip(x: ArrayLike | _A, *, max_value: float, zero_center: bool = True) -> _A:
    return clip_array(x, max_value=max_value, zero_center=zero_center)


def _clip_sparse(x: sparse.spmatrix | sparse.sparray, *, max_value: float, zero_center: bool = True) -> sparse.spmatrix | sparse.sparray:
    x.data = clip(x.data, max_value=max_value, zero_center=zero_center)
    return x

# Register for all sparse types
for sparse_type in [sparse.csr_matrix, sparse.csc_matrix, sparse.csr_array, sparse.csc_array]:
    clip.register(sparse_type)(_clip_sparse)


@clip.register(DaskArray)
def _(x: DaskArray, *, max_value: float, zero_center: bool = True) -> DaskArray:
    return x.map_blocks(
        clip, max_value=max_value, zero_center=zero_center, dtype=x.dtype, meta=x._meta
    )


@njit
def clip_array(
    x: NDArray[np.floating], /, *, max_value: float, zero_center: bool
) -> NDArray[np.floating]:
    a_min, a_max = -max_value, max_value
    if x.ndim > 1:
        for r, c in numba.pndindex(x.shape):
            if x[r, c] > a_max:
                x[r, c] = a_max
            elif x[r, c] < a_min and zero_center:
                x[r, c] = a_min
    else:
        for i in numba.prange(x.size):
            if x[i] > a_max:
                x[i] = a_max
            elif x[i] < a_min and zero_center:
                x[i] = a_min
    return x


@renamed_arg("X", "data", pos_0=True)
@old_positionals("zero_center", "max_value", "copy", "layer", "obsm")
@singledispatch
def _scale(
    data: AnnData | _A,
    *,
    zero_center: bool = True,
    max_value: float | None = None,
    copy: bool = False,
    layer: str | None = None,
    obsm: str | None = None,
    mask_obs: NDArray[np.bool_] | str | None = None,
) -> AnnData | _A | None:
    """Scale data to unit variance and zero mean.

    .. note::
        Variables (genes) that do not display any variation (are constant across
        all observations) are retained and (for zero_center==True) set to 0
        during this operation. In the future, they might be set to NaNs.

    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    zero_center
        If `False`, omit zero-centering variables, which allows to handle sparse
        input efficiently.
    max_value
        Clip (truncate) to this value after scaling. If `None`, do not clip.
    copy
        Whether this function should be performed inplace. If an AnnData object
        is passed, this also determines if a copy is returned.
    layer
        If provided, which element of layers to scale.
    obsm
        If provided, which element of obsm to scale.
    mask_obs
        Restrict both the derivation of scaling parameters and the scaling itself
        to a certain set of observations. The mask is specified as a boolean array
        or a string referring to an array in :attr:`~anndata.AnnData.obs`.
        This will transform data from csc to csr format if `issparse(data)`.

    Returns
    -------
    Returns `None` if `copy=False`, else returns an updated `AnnData` object. Sets the following fields:

    `adata.X` | `adata.layers[layer]` : :class:`numpy.ndarray` | :class:`scipy.sparse.csr_matrix` (dtype `float`)
        Scaled count data matrix.
    `adata.var['mean']` : :class:`pandas.Series` (dtype `float`)
        Means per gene before scaling.
    `adata.var['std']` : :class:`pandas.Series` (dtype `float`)
        Standard deviations per gene before scaling.
    `adata.var['var']` : :class:`pandas.Series` (dtype `float`)
        Variances per gene before scaling.

    """
    check_array_function_arguments(layer=layer, obsm=obsm)
    if layer is not None:
        msg = f"`layer` argument inappropriate for value of type {type(data)}"
        raise ValueError(msg)
    if obsm is not None:
        msg = f"`obsm` argument inappropriate for value of type {type(data)}"
        raise ValueError(msg)
    return scale_array(
        data, zero_center=zero_center, max_value=max_value, copy=copy, mask_obs=mask_obs
    )



def scale_array(
    x: _A,
    *,
    zero_center: bool = True,
    max_value: float | None = None,
    copy: bool = False,
    return_mean_std: bool = False,
    mask_obs: NDArray[np.bool_] | None = None,
) -> (
    _A
    | tuple[
        _A,
        NDArray[np.float64] | DaskArray,
        NDArray[np.float64],
    ]
):
    if copy:
        x = x.copy()

    if not zero_center and max_value is not None:
        logg.info(  # Be careful of what? This should be more specific
            "... be careful when using `max_value` without `zero_center`."
        )

    if np.issubdtype(x.dtype, np.integer):
        logg.info(
            "... as scaling leads to float results, integer "
            "input is cast to float, returning copy."
        )
        x = x.astype(np.float64)

    mask_obs = (
        # For CSR matrices, default to a set mask to take the `scale_array_masked` path.
        # This is faster than the maskless `axis_mul_or_truediv` path.
        np.ones(x.shape[0], dtype=np.bool_)
        if isinstance(x, CSRBase) and mask_obs is None and not zero_center
        else _check_mask(x, mask_obs, "obs")
    )
    if mask_obs is not None:
        return scale_array_masked(
            x,
            mask_obs,
            zero_center=zero_center,
            max_value=max_value,
            return_mean_std=return_mean_std,
        )

    mean, var = _get_mean_var(x, axis=0)
    std = np.sqrt(var)
    std[std == 0] = 1
    if zero_center:
        if isinstance(x, CSBase) or (
            isinstance(x, DaskArray) and isinstance(x._meta, CSBase)
        ):
            msg = "zero-centering a sparse array/matrix densifies it."
            warnings.warn(msg, UserWarning, stacklevel=2)
        x -= mean
        x = dematrix(x)

    x = axis_mul_or_truediv(
        x,
        std,
        op=truediv,
        out=x if isinstance(x, np.ndarray | CSBase) else None,
        axis=1,
    )

    # do the clipping
    if max_value is not None:
        x = clip(x, max_value=max_value, zero_center=zero_center)
    if return_mean_std:
        return x, mean, std
    else:
        return x

def dematrix(x: _A | np.matrix) -> _A:
    """Convert matrix to array if needed."""
    if isinstance(x, np.matrix):
        return x.A
    if isinstance(x, DaskArray) and isinstance(x._meta, np.matrix):
        return x.map_blocks(np.asarray, meta=np.array([], dtype=x.dtype))
    return x

def scale_array_masked(
    x: _A,
    mask_obs: NDArray[np.bool_],
    *,
    zero_center: bool = True,
    max_value: float | None = None,
    return_mean_std: bool = False,
) -> (
    _A
    | tuple[
        _A,
        NDArray[np.float64] | DaskArray,
        NDArray[np.float64],
    ]
):
    if isinstance(x, CSBase) and not zero_center:
        if isinstance(x, CSCBase):
            x = x.tocsr()
        mean, var = _get_mean_var(x[mask_obs, :], axis=0)
        std = np.sqrt(var)
        std[std == 0] = 1

        scale_and_clip_csr(
            x.indptr,
            x.indices,
            x.data,
            std=std,
            mask_obs=mask_obs,
            max_value=max_value,
        )
    else:
        x[mask_obs, :], mean, std = scale_array(
            x[mask_obs, :],
            zero_center=zero_center,
            max_value=max_value,
            return_mean_std=True,
        )

    if return_mean_std:
        return x, mean, std
    else:
        return x


@njit
def scale_and_clip_csr(
    indptr: NDArray[np.integer],
    indices: NDArray[np.integer],
    data: NDArray[np.floating],
    *,
    std: NDArray[np.floating],
    mask_obs: NDArray[np.bool_],
    max_value: float | None,
) -> None:
    for i in numba.prange(len(indptr) - 1):
        if mask_obs[i]:
            for j in range(indptr[i], indptr[i + 1]):
                if max_value is not None:
                    data[j] = min(max_value, data[j] / std[indices[j]])
                else:
                    data[j] /= std[indices[j]]



def scale_anndata(
    adata: AnnData,
    *,
    zero_center: bool = True,
    max_value: float | None = None,
    copy: bool = False,
    layer: str | None = None,
    obsm: str | None = None,
    mask_obs: NDArray[np.bool_] | str | None = None,
) -> AnnData | None:
    is_rust = _is_rust_backend(adata)
    if not is_rust:
        adata = adata.copy() if copy else adata
    else:
        pass
    str_mean_std = ("mean", "std")
    if mask_obs is not None:
        if isinstance(mask_obs, str):
            str_mean_std = (f"mean of {mask_obs}", f"std of {mask_obs}")
        else:
            str_mean_std = ("mean with mask", "std with mask")
        mask_obs = _check_mask(adata, mask_obs, "obs")
    view_to_actual(adata)
    x = _get_obs_rep(adata, layer=layer, obsm=obsm)
    if is_rust:
        x = x[:]
    raise_not_implemented_error_if_backed_type(x, "scale")
    x, adata.var[str_mean_std[0]], adata.var[str_mean_std[1]] = scale_array(
        x,
        zero_center=zero_center,
        max_value=max_value,
        copy=False,  # because a copy has already been made, if it were to be made
        return_mean_std=True,
        mask_obs=mask_obs,
    )
    _set_obs_rep(adata, x, layer=layer, obsm=obsm)
    return adata if copy else None
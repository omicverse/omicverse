r"""Let the ``AnnData``-shaped plots in ``ov.pl`` accept a plain table.

Most plotting functions in ``ov.pl`` only ever touch five things on an
``AnnData``: ``.obs``, ``.var_names``, the expression matrix (``.X`` /
``.layers`` / ``.raw``), ``.obsm``, and ``.uns`` for stored colours. Requiring
the whole container for that is a needless barrier — a `DataFrame` of cell
metadata carries everything a proportion plot or a boxplot needs.

Two tools here, for two situations.

:func:`as_plotdata` is for **new** code. It returns a :class:`PlotData` with a
single accessor, ``values(key)``, that resolves a key against metadata columns
first and features second, whatever the underlying container is.

:func:`accepts_frame` is for **existing** code. It is a decorator that wraps a
``DataFrame`` in a minimal ``AnnData``-shaped view so a function written
against ``adata.obs`` keeps working unchanged. Deliberately *not* a fake
``AnnData``: it exposes only what it really has, so a function that reaches for
``.X`` fails immediately with a clear message instead of on some later line.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["PlotData", "ObsView", "as_plotdata", "accepts_frame", "get_values",
           "get_matrix"]


def _is_anndata_like(obj: Any) -> bool:
    return all(hasattr(obj, attr) for attr in ("obs", "var_names", "obs_names"))


def _densify(values: Any) -> np.ndarray:
    """1-D dense array, whatever came out of the matrix."""
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values).ravel()


def _pick_matrix(data: Any, key: str, layer: Optional[str],
                 use_raw: Optional[bool]):
    """Choose the matrix to read ``key`` from, and the index within it.

    Returns ``(matrix, index, source)`` where ``source`` names what was used,
    for error messages and provenance.
    """
    raw = getattr(data, "raw", None)
    in_var = key in data.var_names
    in_raw = raw is not None and key in raw.var_names

    if layer is not None:
        if use_raw:
            raise ValueError(
                "Give either `layer=` or `use_raw=True`, not both — "
                "`.raw` has no layers."
            )
        layers = getattr(data, "layers", {})
        if layer not in layers:
            available = ", ".join(map(str, layers)) or "none"
            raise KeyError(f"No layer {layer!r}. Available: {available}.")
        if not in_var:
            raise KeyError(
                f"{key!r} is not in `.var_names`, so it cannot be read from "
                f"layer {layer!r}."
            )
        return layers[layer], data.var_names.get_loc(key), f"layers[{layer!r}]"

    if use_raw is True:
        if not in_raw:
            raise KeyError(
                f"`use_raw=True` but {key!r} is not in `.raw.var_names`"
                + ("." if raw is not None else " — this object has no `.raw`.")
            )
        return raw.X, raw.var_names.get_loc(key), ".raw.X"

    if in_var:
        return data.X, data.var_names.get_loc(key), ".X"

    if use_raw is None and in_raw:
        # the usual reason a name is missing from .var_names is HVG subsetting
        return raw.X, raw.var_names.get_loc(key), ".raw.X"

    return None, None, None


def get_values(data: Any, key: str, *, layer: Optional[str] = None,
               use_raw: Optional[bool] = None,
               dense: bool = True) -> np.ndarray:
    r"""Read one named vector — a metadata column or a feature.

    This is the single place ``ov.pl`` resolves "a name the user typed" into
    numbers. Before it existed, nine modules each had their own version of the
    same three lines, and they disagreed: some densified sparse matrices and
    crashed on dense ones, some did the opposite, some preferred ``.raw`` over
    ``.obs``, and only two honoured ``layer=``.

    Arguments
    ---------
    data
        ``AnnData`` (or anything AnnData-shaped), ``DataFrame``, ``dict``, or
        a :class:`PlotData`.
    key
        Name to resolve.
    layer
        Read the feature from ``.layers[layer]`` instead of ``.X``. Cannot be
        combined with ``use_raw=True``.
    use_raw
        ``True`` forces ``.raw``; ``False`` forbids it; ``None`` (default)
        reads ``.X`` and falls back to ``.raw`` **only** when the name is
        absent from ``.var_names`` — the situation left behind by
        highly-variable-gene subsetting.

        Note this differs from ``scanpy``'s plotting default, which prefers
        ``.raw`` whenever it exists. The rule here never silently swaps the
        matrix under a name that ``.X`` can already answer; pass
        ``use_raw=True`` for the scanpy behaviour.
    dense
        Return a 1-D dense array. ``False`` returns whatever the matrix
        column was.

    Returns
    -------
    ``numpy.ndarray`` of length ``n_obs``.

    Raises
    ------
    KeyError
        With the near-misses listed, when the name resolves to nothing.
    """
    if isinstance(data, PlotData):
        return data.values(key, layer=layer)

    frame = data if isinstance(data, pd.DataFrame) else getattr(data, "obs", None)
    if isinstance(data, dict):
        frame = pd.DataFrame({k: np.asarray(v).ravel() for k, v in data.items()})

    # metadata wins: a name that is both a column and a gene means the column
    if isinstance(frame, pd.DataFrame) and key in frame.columns:
        if layer is not None:
            raise ValueError(
                f"{key!r} is a metadata column; `layer=` only applies to "
                f"features."
            )
        return frame[key].to_numpy()

    if _is_anndata_like(data):
        matrix, index, _ = _pick_matrix(data, key, layer, use_raw)
        if matrix is not None:
            column = matrix[:, index]
            return _densify(column) if dense else column

    _raise_unknown_key(data, frame, key, use_raw)


def _raise_unknown_key(data, frame, key, use_raw):
    from difflib import get_close_matches

    pool: List[str] = []
    if isinstance(frame, pd.DataFrame):
        pool += [str(c) for c in frame.columns]
    if _is_anndata_like(data):
        pool += [str(v) for v in data.var_names[:5000]]
    hint = get_close_matches(str(key), pool, n=3, cutoff=0.6)
    suffix = f" Did you mean: {', '.join(hint)}?" if hint else ""
    raw = getattr(data, "raw", None)
    if use_raw is False and raw is not None and key in raw.var_names:
        suffix += " It is present in `.raw` — pass `use_raw=True`."
    raise KeyError(
        f"{key!r} is neither a metadata column nor a feature of this "
        f"object.{suffix}"
    )


def get_matrix(data: Any, keys: Sequence[str], *, layer: Optional[str] = None,
               use_raw: Optional[bool] = None) -> np.ndarray:
    r"""Read several names at once into an ``(n_obs, len(keys))`` array.

    Same resolution rules as :func:`get_values`, but it reads each matrix once
    rather than once per key — which matters for a dot plot over 50 genes.
    """
    keys = list(keys)
    if not keys:
        raise ValueError("`keys` is empty.")
    columns = [get_values(data, key, layer=layer, use_raw=use_raw)
               for key in keys]
    lengths = {len(c) for c in columns}
    if len(lengths) > 1:
        raise ValueError(f"Resolved vectors have different lengths: {lengths}.")
    return np.column_stack(columns).astype(float, copy=False)


class PlotData:
    """Uniform read access to metadata and feature values.

    Attributes
    ----------
    obs : pandas.DataFrame
        Per-observation metadata.
    var_names : pandas.Index
        Names that :meth:`values` can resolve as features.
    obsm : Mapping
        Embeddings, keyed as in ``AnnData.obsm``. Empty for a plain table
        unless one was supplied.
    source : Any
        The original object, for functions that still need it.
    """

    def __init__(self, obs: pd.DataFrame, *, var_names: Optional[pd.Index] = None,
                 feature_getter: Optional[Callable[..., np.ndarray]] = None,
                 obsm: Optional[Any] = None, source: Any = None):
        self.obs = obs
        self.var_names = pd.Index([] if var_names is None else var_names)
        self.obsm = obsm if obsm is not None else {}
        self.source = source
        self._feature_getter = feature_getter

    @property
    def n_obs(self) -> int:
        return int(len(self.obs))

    def has(self, key: str) -> bool:
        """Whether ``key`` resolves to either a metadata column or a feature."""
        return key in self.obs.columns or key in self.var_names

    def values(self, key: str, layer: Optional[str] = None, *,
               use_raw: Optional[bool] = None) -> np.ndarray:
        """Values of ``key`` — a metadata column, or a feature.

        Metadata wins when a name is both, which is almost always what the
        caller meant. See :func:`get_values` for the ``layer`` / ``use_raw``
        rules; this method is the same resolver bound to one object.
        """
        if key in self.obs.columns:
            if layer is not None:
                raise ValueError(
                    f"{key!r} is a metadata column; `layer=` only applies to "
                    f"features."
                )
            return self.obs[key].to_numpy()
        if self._feature_getter is not None and (
                key in self.var_names or _is_anndata_like(self.source)):
            return self._feature_getter(key, layer, use_raw)
        from difflib import get_close_matches

        pool = list(map(str, self.obs.columns)) + list(map(str, self.var_names[:2000]))
        hint = get_close_matches(str(key), pool, n=3, cutoff=0.6)
        suffix = f" Did you mean: {', '.join(hint)}?" if hint else ""
        raise KeyError(
            f"{key!r} is neither a metadata column nor a known feature.{suffix}"
        )

    def embedding(self, basis: str) -> np.ndarray:
        """Coordinates of ``basis``, accepting ``'umap'`` for ``'X_umap'``."""
        for candidate in (basis, f"X_{basis}"):
            if candidate in self.obsm:
                return np.asarray(self.obsm[candidate])
        available = ", ".join(map(str, list(self.obsm))) or "none"
        raise KeyError(f"No embedding {basis!r}. Available: {available}.")

    def __repr__(self) -> str:
        return (f"PlotData({self.n_obs} obs, {self.obs.shape[1]} metadata "
                f"columns, {len(self.var_names)} features)")


def as_plotdata(data: Any, *, obs: Optional[pd.DataFrame] = None,
                obsm: Optional[Any] = None,
                layer: Optional[str] = None) -> PlotData:
    r"""Wrap ``data`` in a :class:`PlotData`.

    Accepts an ``AnnData`` (or anything AnnData-shaped, including the
    out-of-core variants), a ``DataFrame``, a ``dict`` of columns, or a 2-D
    array with ``obs=`` supplied separately.

    No data is copied — the view holds references.

    Examples
    --------
    >>> pd_view = ov.pl.as_plotdata(adata)
    >>> pd_view.values('CD3D')            # a gene, from .X
    >>> pd_view.values('leiden')          # a column of .obs
    >>> table = ov.pl.as_plotdata(df)
    >>> table.values('score')             # a column of the frame
    """
    if isinstance(data, PlotData):
        return data

    if _is_anndata_like(data):
        def feature_getter(key: str, layer_name: Optional[str],
                           use_raw: Optional[bool] = None) -> np.ndarray:
            matrix, index, _ = _pick_matrix(data, key, layer_name, use_raw)
            if matrix is None:
                _raise_unknown_key(data, data.obs, key, use_raw)
            return _densify(matrix[:, index])

        return PlotData(data.obs, var_names=data.var_names,
                        feature_getter=feature_getter,
                        obsm=getattr(data, "obsm", {}), source=data)

    if isinstance(data, pd.DataFrame):
        numeric = data.columns[[pd.api.types.is_numeric_dtype(data[c])
                                for c in data.columns]]
        return PlotData(data, var_names=numeric, obsm=obsm or {}, source=data)

    if isinstance(data, dict):
        frame = pd.DataFrame({k: np.asarray(v).ravel() for k, v in data.items()})
        return as_plotdata(frame, obsm=obsm)

    array = np.asarray(data)
    if array.ndim == 2:
        if obs is None:
            raise TypeError(
                "A 2-D array needs `obs=` — a DataFrame of per-row metadata — "
                "before it can be plotted."
            )
        columns = pd.Index([f"feature_{i}" for i in range(array.shape[1])])

        def array_getter(key: str, layer_name: Optional[str],
                         use_raw: Optional[bool] = None) -> np.ndarray:
            if layer_name is not None:
                raise ValueError("A bare array has no layers.")
            return array[:, columns.get_loc(key)]

        return PlotData(obs, var_names=columns, feature_getter=array_getter,
                        obsm=obsm or {}, source=data)

    raise TypeError(
        f"Cannot plot a {type(data).__name__}. Pass an AnnData, a DataFrame, "
        f"a dict of columns, or a 2-D array together with `obs=`."
    )


class ObsView:
    """An ``AnnData``-shaped, metadata-only view over a ``DataFrame``.

    Exposes ``.obs``, ``.obs_names``, ``.var_names``, ``.uns``, ``.shape``,
    ``.n_obs`` and ``.n_vars`` — and nothing else. A function that only reads
    and annotates ``.obs`` works against this unchanged; one that reaches for
    ``.X`` gets an ``AttributeError`` naming the problem rather than a
    confusing failure further in.

    ``.obs`` is a **copy** of the input frame. Several ``ov.pl`` functions
    coerce their grouping column to ``category`` in place, and mutating the
    caller's DataFrame as a side effect of drawing a plot would be a
    surprise.

    String columns are converted to ``category`` on construction, because
    ``anndata`` does the same when an ``AnnData`` is built — so plots that
    reach for ``.cat.categories`` (most of the grouped ones do) find what they
    expect instead of failing on a plain object column.
    """

    def __init__(self, frame: pd.DataFrame, *, uns: Optional[dict] = None,
                 categorical: bool = True):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"ObsView needs a DataFrame, got {type(frame).__name__}.")
        self.obs = frame.copy()
        if categorical:
            for column in self.obs.columns:
                dtype = self.obs[column].dtype
                if dtype == object or pd.api.types.is_string_dtype(dtype):
                    self.obs[column] = self.obs[column].astype("category")
        self.uns: dict = dict(uns or {})
        self.var_names = pd.Index([])

    @property
    def obs_names(self) -> pd.Index:
        return self.obs.index

    @property
    def n_obs(self) -> int:
        return int(len(self.obs))

    @property
    def n_vars(self) -> int:
        return 0

    @property
    def shape(self):
        return (self.n_obs, 0)

    def __len__(self) -> int:
        return self.n_obs

    def __getattr__(self, name: str):
        if name in {"X", "layers", "raw", "var", "obsm", "varm", "obsp"}:
            raise AttributeError(
                f"This plot was given a DataFrame, which has no {name!r}. "
                f"Pass an AnnData if you need expression values — or use "
                f"ov.pl.as_plotdata() to see what a table can supply."
            )
        raise AttributeError(name)

    def __repr__(self) -> str:
        return f"ObsView({self.n_obs} rows x {self.obs.shape[1]} columns)"


def accepts_frame(func: Optional[Callable] = None, *, argument: int = 0):
    r"""Let a metadata-only ``AnnData`` plot also take a ``DataFrame``.

    Wraps the ``argument``-th positional parameter: a ``DataFrame`` or ``dict``
    becomes an :class:`ObsView`, anything ``AnnData``-shaped passes straight
    through untouched.

    Use it only on functions that read metadata. A function that needs
    expression values will raise from :class:`ObsView` with an actionable
    message, which is the intended outcome — better than silently plotting
    nothing.

    Examples
    --------
    >>> @accepts_frame
    ... def cellproportion(adata, celltype_clusters, groupby, ...):
    ...     ...
    >>> cellproportion(df, celltype_clusters='cell_type', groupby='sample')
    """
    def decorate(inner: Callable) -> Callable:
        @functools.wraps(inner)
        def wrapper(*args, **kwargs):
            args = list(args)
            if len(args) > argument:
                candidate = args[argument]
                if isinstance(candidate, (pd.DataFrame, dict)):
                    frame = (candidate if isinstance(candidate, pd.DataFrame)
                             else pd.DataFrame(candidate))
                    args[argument] = ObsView(frame)
            return inner(*args, **kwargs)

        wrapper.__doc__ = (inner.__doc__ or "") + (
            "\n\n    Accepts a ``pandas.DataFrame`` of per-observation metadata "
            "wherever an\n    ``AnnData`` is expected — the frame is used as "
            "``.obs``. See\n    :func:`omicverse.pl.accepts_frame`.\n"
        )
        return wrapper

    return decorate if func is None else decorate(func)

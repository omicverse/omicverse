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
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["PlotData", "ObsView", "as_plotdata", "accepts_frame"]


def _is_anndata_like(obj: Any) -> bool:
    return all(hasattr(obj, attr) for attr in ("obs", "var_names", "obs_names"))


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
                 feature_getter: Optional[Callable[[str, Optional[str]], np.ndarray]] = None,
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

    def values(self, key: str, layer: Optional[str] = None) -> np.ndarray:
        """Values of ``key`` — a metadata column, or a feature.

        Metadata wins when a name is both, which matches ``scanpy``'s
        behaviour and is almost always what the caller meant.
        """
        if key in self.obs.columns:
            if layer is not None:
                raise ValueError(
                    f"{key!r} is a metadata column; `layer=` only applies to "
                    f"features."
                )
            return self.obs[key].to_numpy()
        if key in self.var_names and self._feature_getter is not None:
            return self._feature_getter(key, layer)
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
        def feature_getter(key: str, layer_name: Optional[str]) -> np.ndarray:
            index = data.var_names.get_loc(key)
            if layer_name is not None:
                matrix = data.layers[layer_name]
            elif getattr(data, "raw", None) is not None and key in data.raw.var_names:
                matrix = data.raw.X
                index = data.raw.var_names.get_loc(key)
            else:
                matrix = data.X
            column = matrix[:, index]
            if hasattr(column, "toarray"):
                column = column.toarray()
            return np.asarray(column).ravel()

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

        def array_getter(key: str, layer_name: Optional[str]) -> np.ndarray:
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

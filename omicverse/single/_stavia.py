from __future__ import annotations

import contextlib
import importlib
from typing import Any, Iterable

import numpy as np
import pandas as pd
from anndata import AnnData

from .._optional import build_optional_dependency_error, missing_optional_dependency
from .._registry import register_function
from .._settings import add_reference


def _stavia_required_modules(*, rw2: bool = False) -> tuple[str, ...]:
    dependencies = ("leidenalg", "hnswlib", "pygam")
    if rw2:
        dependencies += ("pecanpy", "numba_progress")
    return dependencies


def _raise_stavia_dependency_error(
    dependencies: Iterable[str],
    *,
    rw2: bool = False,
    cause: BaseException | None = None,
) -> None:
    error = build_optional_dependency_error(
        "omicverse.single.StaVIA",
        dependencies,
        install_hint=(
            "RW2 mode requires optional packages. "
            "Install them with `pip install pecanpy numba-progress`."
            if rw2
            else (
                "StaVIA uses the vendored VIA backend. Install the VIA runtime "
                "dependencies with `pip install leidenalg hnswlib pygam` or "
                "`pip install \"omicverse[full]\"`."
            )
        ),
    )
    if cause is not None:
        raise error from cause
    raise error


def _require_modules(dependencies: Iterable[str], *, rw2: bool = False) -> None:
    missing = []
    cause: BaseException | None = None
    for dependency in dependencies:
        try:
            importlib.import_module(dependency)
        except Exception as exc:  # pragma: no cover - exact import failure varies by platform
            missing.append(dependency)
            cause = exc
    if missing:
        _raise_stavia_dependency_error(missing, rw2=rw2, cause=cause)


def _load_via_backend(*, rw2_mode: bool = False):
    base_dependencies = _stavia_required_modules()
    _require_modules(base_dependencies)
    if rw2_mode:
        rw2_dependencies = tuple(
            dependency
            for dependency in _stavia_required_modules(rw2=True)
            if dependency not in base_dependencies
        )
        _require_modules(rw2_dependencies, rw2=True)

    try:
        return importlib.import_module("..external.VIA", package=__package__)
    except ImportError as exc:
        dependencies = _stavia_required_modules(rw2=rw2_mode)
        if missing_optional_dependency(exc, dependencies):
            _raise_stavia_dependency_error(dependencies, rw2=rw2_mode, cause=exc)
        raise


def _as_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return list(value)


@contextlib.contextmanager
def _suppress_via_plots():
    """Prevent backend auto-display while preserving VIA console output."""
    import matplotlib.pyplot as plt

    existing_figures = set(plt.get_fignums())
    original_show = plt.show

    def _no_show(*args: Any, **kwargs: Any) -> None:
        return None

    plt.show = _no_show
    try:
        yield
    finally:
        for figure_number in set(plt.get_fignums()) - existing_figures:
            plt.close(figure_number)
        plt.show = original_show


@register_function(
    aliases=["StaVIA", "stavia", "spatial trajectory", "spatio-temporal trajectory", "VIA spatial"],
    category="single",
    description=(
        "OV-native StaVIA wrapper for spatial, temporal, and spatio-temporal "
        "trajectory inference on AnnData objects."
    ),
    prerequisites={
        "functions": ["pca"],
        "optional_functions": ["neighbors", "umap"],
    },
    requires={
        "obsm": ["X_pca"],
    },
    produces={
        "obs": ["stavia_pseudotime", "stavia_cluster"],
        "obsm": ["stavia_lineage_probabilities"],
        "uns": ["stavia"],
    },
    auto_fix="none",
    examples=[
        "stavia = ov.single.StaVIA(adata, use_rep='X_pca', basis='X_umap',",
        "                         cluster_key='clusters', spatial_key='spatial')",
        "stavia.fit()",
        "ov.pl.plot_stream(stavia, method='stavia')",
    ],
    related=["single.TrajInfer", "single.Monocle"],
)
class StaVIA:
    """AnnData-native wrapper around the StaVIA/VIA backend.

    The public interface follows OmicVerse conventions: users pass AnnData keys
    such as ``spatial_key`` and ``time_key``. The raw VIA backend only receives
    arrays after validation inside this wrapper.
    """

    def __init__(
        self,
        adata: AnnData,
        *,
        use_rep: str = "X_pca",
        n_comps: int | None = 30,
        basis: str | None = "X_umap",
        cluster_key: str | None = "clusters",
        spatial_key: str | None = "spatial",
        time_key: str | None = None,
        sample_key: str | None = None,
        key_added: str = "stavia",
        root: Any = None,
        knn: int = 30,
        spatial_knn: int = 15,
        do_spatial_knn: bool | None = None,
        do_spatial_layout: bool = False,
        random_seed: int = 42,
        memory: float = 5,
        rw2_mode: bool = False,
        via_kwargs: dict[str, Any] | None = None,
        **backend_kwargs: Any,
    ) -> None:
        self.adata = adata
        self.use_rep = use_rep
        self.n_comps = n_comps
        self.basis = basis
        self.cluster_key = cluster_key
        self.spatial_key = spatial_key
        self.time_key = time_key
        self.sample_key = sample_key
        self.key_added = key_added
        self.root = root
        self.knn = knn
        self.spatial_knn = spatial_knn
        self.do_spatial_knn = do_spatial_knn
        self.do_spatial_layout = do_spatial_layout
        self.random_seed = random_seed
        self.memory = memory
        self.rw2_mode = rw2_mode
        self.via_kwargs = dict(via_kwargs or {})
        self.via_kwargs.update(backend_kwargs)
        self.model = None

    @property
    def pseudotime_key(self) -> str:
        return f"{self.key_added}_pseudotime"

    @property
    def cluster_result_key(self) -> str:
        return f"{self.key_added}_cluster"

    @property
    def lineage_probability_key(self) -> str:
        return f"{self.key_added}_lineage_probabilities"

    def _effective_rw2_mode(self) -> bool:
        return bool(self.via_kwargs.get("RW2_mode", self.rw2_mode))

    def _obsm_array(self, key: str, *, name: str, min_cols: int | None = None) -> np.ndarray:
        if key not in self.adata.obsm:
            raise KeyError(f"`{name}={key!r}` was not found in `adata.obsm`.")
        array = np.asarray(self.adata.obsm[key])
        if array.ndim != 2:
            raise ValueError(f"`adata.obsm[{key!r}]` must be a two-dimensional array.")
        if array.shape[0] != self.adata.n_obs:
            raise ValueError(
                f"`adata.obsm[{key!r}]` has {array.shape[0]} rows, "
                f"but `adata` has {self.adata.n_obs} observations."
            )
        if min_cols is not None and array.shape[1] < min_cols:
            raise ValueError(
                f"`adata.obsm[{key!r}]` must have at least {min_cols} columns; "
                f"got {array.shape[1]}."
            )
        return array

    def _data_matrix(self) -> np.ndarray:
        data = self._obsm_array(self.use_rep, name="use_rep")
        if self.n_comps is None:
            return data
        if self.n_comps <= 0:
            raise ValueError("`n_comps` must be positive or None.")
        if self.n_comps > data.shape[1]:
            raise ValueError(
                f"`n_comps={self.n_comps}` exceeds `adata.obsm[{self.use_rep!r}]` "
                f"with {data.shape[1]} columns."
            )
        return data[:, : self.n_comps]

    def _embedding(self) -> np.ndarray | None:
        if self.basis is None:
            return None
        return self._obsm_array(self.basis, name="basis", min_cols=2)[:, :2]

    def _true_labels(self):
        if self.cluster_key is None:
            return None
        if self.cluster_key not in self.adata.obs:
            raise KeyError(f"`cluster_key={self.cluster_key!r}` was not found in `adata.obs`.")
        return self.adata.obs[self.cluster_key]

    def _spatial_coords(self) -> np.ndarray | None:
        if self.spatial_key is None:
            return None
        return self._obsm_array(self.spatial_key, name="spatial_key", min_cols=2)[:, :2]

    def _time_labels(self) -> list[float] | None:
        if self.time_key is None:
            return None
        if self.time_key not in self.adata.obs:
            raise KeyError(f"`time_key={self.time_key!r}` was not found in `adata.obs`.")
        series = self.adata.obs[self.time_key]
        if isinstance(series.dtype, pd.CategoricalDtype) and series.cat.ordered:
            return series.cat.codes.astype(int).tolist()
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().any():
            raise ValueError(
                f"`adata.obs[{self.time_key!r}]` must be numeric or an ordered categorical series."
            )
        return numeric.tolist()

    def _spatial_aux(self) -> list[str]:
        if self.sample_key is None:
            return []
        if self.sample_key not in self.adata.obs:
            raise KeyError(f"`sample_key={self.sample_key!r}` was not found in `adata.obs`.")
        return self.adata.obs[self.sample_key].astype(str).tolist()

    def _backend_params(self) -> dict[str, Any]:
        spatial_coords = self._spatial_coords()
        time_labels = self._time_labels()
        root_user = _as_list(self.root)
        do_spatial_knn = self.do_spatial_knn
        if do_spatial_knn is None:
            do_spatial_knn = spatial_coords is not None

        params: dict[str, Any] = {
            "data": self._data_matrix(),
            "true_label": self._true_labels(),
            "embedding": self._embedding(),
            "knn": self.knn,
            "random_seed": self.random_seed,
            "memory": self.memory,
            "time_series": time_labels is not None,
            "time_series_labels": time_labels,
            "spatial_coords": spatial_coords,
            "do_spatial_knn": bool(do_spatial_knn),
            "do_spatial_layout": self.do_spatial_layout,
            "spatial_knn": self.spatial_knn,
            "spatial_aux": self._spatial_aux(),
            "RW2_mode": self._effective_rw2_mode(),
            "do_compute_embedding": False,
        }
        if root_user is not None:
            params["root_user"] = root_user
        params.update(self.via_kwargs)
        return params

    def fit(self, **backend_overrides: Any) -> "StaVIA":
        """Run StaVIA and write standardized results back to ``adata``."""
        if backend_overrides:
            self.via_kwargs.update(backend_overrides)
        params = self._backend_params()
        VIA = _load_via_backend(rw2_mode=bool(params.get("RW2_mode")))
        with _suppress_via_plots():
            self.model = VIA.core.VIA(**params)
            self.model.run_VIA()
        self._write_results(params)
        add_reference(self.adata, "StaVIA", "spatial-temporal trajectory inference with StaVIA")
        return self

    def _write_results(self, params: dict[str, Any]) -> None:
        if self.model is None:
            raise RuntimeError("Run `fit()` before writing StaVIA results.")

        pseudotime = getattr(self.model, "single_cell_pt_markov", None)
        if pseudotime is None:
            raise RuntimeError("The StaVIA backend did not expose `single_cell_pt_markov`.")
        pseudotime = np.asarray(pseudotime, dtype=float)
        if pseudotime.shape[0] != self.adata.n_obs:
            raise RuntimeError(
                "The StaVIA backend returned pseudotime with length "
                f"{pseudotime.shape[0]}, expected {self.adata.n_obs}."
            )
        self.adata.obs[self.pseudotime_key] = pseudotime

        labels = getattr(self.model, "labels", None)
        if labels is not None:
            labels = np.asarray(labels)
            if labels.shape[0] == self.adata.n_obs:
                self.adata.obs[self.cluster_result_key] = pd.Categorical(labels.astype(str))

        lineage_probs = getattr(self.model, "single_cell_bp", None)
        if lineage_probs is not None:
            lineage_probs = np.asarray(lineage_probs, dtype=float)
            if lineage_probs.ndim == 2:
                if lineage_probs.shape[0] != self.adata.n_obs and lineage_probs.shape[1] == self.adata.n_obs:
                    lineage_probs = lineage_probs.T
                if lineage_probs.shape[0] == self.adata.n_obs:
                    terminal_clusters = list(getattr(self.model, "terminal_clusters", []))
                    columns = [
                        f"lineage_{terminal_clusters[i]}" if i < len(terminal_clusters) else f"lineage_{i}"
                        for i in range(lineage_probs.shape[1])
                    ]
                    self.adata.obsm[self.lineage_probability_key] = pd.DataFrame(
                        lineage_probs,
                        index=self.adata.obs_names,
                        columns=columns,
                    )

        self.adata.uns[self.key_added] = {
            "method": "StaVIA",
            "backend": "omicverse.external.VIA.core.VIA",
            "use_rep": self.use_rep,
            "n_comps": self.n_comps,
            "basis": self.basis,
            "cluster_key": self.cluster_key,
            "spatial_key": self.spatial_key,
            "time_key": self.time_key,
            "sample_key": self.sample_key,
            "pseudotime_key": self.pseudotime_key,
            "cluster_key_added": self.cluster_result_key,
            "lineage_probability_key": (
                self.lineage_probability_key
                if self.lineage_probability_key in self.adata.obsm
                else None
            ),
            "terminal_clusters": list(getattr(self.model, "terminal_clusters", [])),
            "params": {
                "knn": params.get("knn"),
                "spatial_knn": params.get("spatial_knn"),
                "do_spatial_knn": params.get("do_spatial_knn"),
                "do_spatial_layout": params.get("do_spatial_layout"),
                "time_series": params.get("time_series"),
                "random_seed": params.get("random_seed"),
                "memory": params.get("memory"),
                "RW2_mode": params.get("RW2_mode"),
            },
        }

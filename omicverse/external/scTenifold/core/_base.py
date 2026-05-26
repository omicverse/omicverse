import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Union
import inspect

import numpy as np
import pandas as pd
from scipy import sparse

from omicverse.external.scTenifold.core._networks import *
from omicverse.external.scTenifold.core._networks import anndata_to_dataframe
from omicverse.external.scTenifold.core._types import ExpressionData, KOMethod, Kwargs
from omicverse.external.scTenifold.core._QC import sc_QC
from omicverse.external.scTenifold.core._norm import cpm_norm
from omicverse.external.scTenifold.core._decomposition import tensor_decomp
from omicverse.external.scTenifold.core._ko import reconstruct_pcnets
from omicverse.external.scTenifold.plotting import plot_hist
from omicverse.external.scTenifold.data import read_folder

__all__ = ["scTenifoldNet", "scTenifoldKnk"]


def _fill_dataframe_diagonal(df: pd.DataFrame, value: float) -> pd.DataFrame:
    data = df.to_numpy(copy=True)
    np.fill_diagonal(data, value)
    return pd.DataFrame(data, index=df.index, columns=df.columns)


class scBase:
    """Shared scaffolding for the scTenifold workflows.

    Holds per-step keyword arguments (``qc_kws``, ``nc_kws``, ``td_kws``,
    ``ma_kws``, ``dr_kws``), the intermediate result dictionaries, and the
    save/load plumbing used by both :class:`scTenifoldNet` and
    :class:`scTenifoldKnk`.
    """

    cls_prop = ["shared_gene_names", "strict_lambda"]
    kw_sigs = {"qc_kws": inspect.signature(sc_QC),
               "nc_kws": inspect.signature(make_networks),
               "td_kws": inspect.signature(tensor_decomp),
               "ma_kws": inspect.signature(manifold_alignment),
               "dr_kws": inspect.signature(d_regulation)}

    def __init__(self,
                 qc_kws: Optional[Kwargs] = None,
                 nc_kws: Optional[Kwargs] = None,
                 td_kws: Optional[Kwargs] = None,
                 ma_kws: Optional[Kwargs] = None,
                 dr_kws: Optional[Kwargs] = None,
                 ) -> None:
        """Initialise empty step dicts and bind per-step keyword arguments."""
        self.data_dict = {}
        self.QC_dict = {}
        self.network_dict = {}
        self.tensor_dict = {}
        self.manifold: Optional[pd.DataFrame] = None
        self.d_regulation: Optional[pd.DataFrame] = None
        self.shared_gene_names = None
        self.qc_kws = {} if qc_kws is None else qc_kws
        self.nc_kws = {} if nc_kws is None else nc_kws
        self.td_kws = {} if td_kws is None else td_kws
        self.ma_kws = {} if ma_kws is None else ma_kws
        self.dr_kws = {} if dr_kws is None else dr_kws
        self.step_comps = {"qc": self.QC_dict,
                           "nc": self.network_dict,
                           "td": self.tensor_dict,
                           "ma": self.manifold,
                           "dr": self.d_regulation}

    @classmethod
    def _load_comp(cls,
                   file_dir: Path,
                   comp):
        if comp == "qc":
            dic = {}
            for d in file_dir.iterdir():
                if d.is_file():
                    dic[d.stem] = pd.read_csv(d, index_col=0)
            obj_name = "QC_dict"
        elif comp == "nc":
            dic = {}
            for d in file_dir.iterdir():
                if d.is_dir():
                    dic[d.stem] = []
                    nt = 0
                    while (d / Path(f"network_{nt}.npz")).exists():
                        dic[d.stem].append(sparse.load_npz(d / Path(f"network_{nt}.npz")))
                        nt += 1
            obj_name = "network_dict"
        elif comp == "td":
            dic = {}
            for d in file_dir.iterdir():
                if d.is_file():
                    dic[d.stem] = sparse.load_npz(d).toarray()
            obj_name = "tensor_dict"
        elif comp in ["ma", "dr"]:
            dic = None
            for d in file_dir.iterdir():
                if d.is_file():
                    dic = pd.read_csv(d, index_col=0)
                    break
            obj_name = "manifold" if comp == "ma" else "d_regulation"
        else:
            raise ValueError("The component is not a valid one")
        return dic, obj_name

    @classmethod
    def load(cls,
             file_dir: Union[str, Path],
             **kwargs: object) -> "scBase":
        """Reconstruct an instance previously written by :meth:`save`.

        Parameters
        ----------
        file_dir
            Directory containing the ``kws.json`` and per-step subfolders.
        **kwargs
            Extra constructor kwargs that override values loaded from disk.

        Returns
        -------
        Fully populated subclass instance.
        """
        parent_dir = Path(file_dir)
        kw_path = parent_dir / Path("kws.json")
        with open(kw_path, "r") as f:
            kws = json.load(f)
        kwargs.update(kws)
        kwarg_props = {k: kwargs.pop(k)
                       for k in cls.cls_prop if k in kwargs}
        ins = cls(**kwargs)
        for name, obj in ins.step_comps.items():
            if (parent_dir / Path(name)).exists():
                dic, name = cls._load_comp(parent_dir / Path(name), name)
                setattr(ins, name, dic)
        ins.step_comps = {"qc": ins.QC_dict,
                          "nc": ins.network_dict,
                          "td": ins.tensor_dict,
                          "ma": ins.manifold,
                          "dr": ins.d_regulation}
        for k, prop in kwarg_props.items():
            setattr(ins, k, prop)
        return ins

    @classmethod
    def list_kws(cls, step_name: str) -> Kwargs:
        """Return the default keyword arguments for the named pipeline step.

        Parameters
        ----------
        step_name
            One of ``"qc_kws"``, ``"nc_kws"``, ``"td_kws"``, ``"ma_kws"``,
            ``"dr_kws"``.

        Returns
        -------
        Mapping of keyword name to default value.
        """
        return {n: p.default for n, p in cls.kw_sigs[f"{step_name}"].parameters.items()
                if not (p.default is p.empty)}

    @staticmethod
    def _infer_groups(*args: Kwargs) -> List[str]:
        grps = set()
        for kw in args:
            grps |= set(kw.keys())
        return list(grps)

    def _QC(self, label, plot: bool = True, **kwargs):
        self.QC_dict[label] = self.data_dict[label].copy()
        self.QC_dict[label].loc[:, "gene"] = self.QC_dict[label].index
        self.QC_dict[label] = self.QC_dict[label].groupby(by="gene").sum()
        self.QC_dict[label] = sc_QC(self.QC_dict[label], **kwargs)
        if plot:
            plot_hist(self.QC_dict[label], label)

    def _make_networks(self, label, data, **kwargs):
        self.network_dict[label] = make_networks(data, **kwargs)

    def _tensor_decomp(self, label, gene_names, **kwargs):
        self.tensor_dict[label] = tensor_decomp(np.concatenate([np.expand_dims(network.toarray(), -1)
                                                                for network in self.network_dict[label]], axis=-1),
                                                gene_names, **kwargs)

    def _save_comp(self,
                   file_dir: Path,
                   comp: str,
                   verbose: bool):
        if comp == "qc":
            for label, obj in self.step_comps["qc"].items():
                label_fn = (file_dir / Path(label)).with_suffix(".csv")
                obj.to_csv(label_fn)
                if verbose:
                    print(f"{label_fn.name} has been saved successfully.")
        elif comp == "nc":
            for label, obj in self.step_comps["nc"].items():
                (file_dir / Path(f"{label}")).mkdir(parents=True, exist_ok=True)
                for i, npx in enumerate(obj):
                    file_name = file_dir / Path(f"{label}/network_{i}").with_suffix(".npz")
                    sparse.save_npz(file_name, npx)
                    if verbose:
                        print(f"{file_name.name} has been saved successfully.")
        elif comp == "td":
            for label, obj in self.step_comps["td"].items():
                sp = sparse.coo_matrix(obj)
                label_fn = (file_dir / Path(label)).with_suffix(".npz")
                sparse.save_npz(label_fn, sp)
                if verbose:
                    print(f"{label_fn.name} has been saved successfully.")
        elif comp in ["ma", "dr"]:
            if isinstance(self.step_comps[comp], pd.DataFrame):
                fn = (file_dir / Path("manifold_alignment" if comp == "ma" else "d_regulation")).with_suffix(".csv")
                self.step_comps[comp].to_csv(fn)
                if verbose:
                    print(f"{fn.name} has been saved successfully.")
        else:
            raise ValueError(f"This step is not valid, please choose from {list(self.step_comps.keys())}")

    def save(self,
             file_dir: Union[str, Path],
             comps: Union[str, List[str]] = "all",
             verbose: bool = True,
             **kwargs: object) -> None:
        """Persist intermediate results and the active kw config to disk.

        Parameters
        ----------
        file_dir
            Output directory; created if missing.
        comps
            ``"all"`` (default) saves every populated step; otherwise pass a
            list of step keys (``"qc"``, ``"nc"``, ``"td"``, ``"ma"``,
            ``"dr"``).
        verbose
            If True, print a line per file written.
        **kwargs
            Extra config entries merged into ``kws.json``.
        """
        dir_path = Path(file_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        if comps == "all":
            comps = [k for k, v in self.step_comps.items()
                     if v is not None and (not isinstance(v, dict) or len(v) != 0)]
        for c in comps:
            subdir = dir_path / Path(c)
            subdir.mkdir(parents=True, exist_ok=True)
            self._save_comp(subdir, c, verbose)
        configs = {"qc_kws": self.qc_kws, "nc_kws": self.nc_kws, "td_kws": self.td_kws, "ma_kws": self.ma_kws}
        if hasattr(self, "ko_kws"):
            configs.update({"ko_kws": getattr(self, "ko_kws")})
        if hasattr(self, "dr_kws"):
            configs.update({"dr_kws": getattr(self, "dr_kws")})
        if self.shared_gene_names is not None:
            configs.update({"shared_gene_names": self.shared_gene_names})
        configs.update(kwargs)
        with open(dir_path / Path('kws.json'), 'w') as f:
            json.dump(configs, f)


class scTenifoldNet(scBase):
    """Two-sample scTenifoldNet workflow.

    Pipeline order: ``qc`` → ``nc`` (PC network construction) → ``td``
    (tensor decomposition) → ``ma`` (manifold alignment) → ``dr``
    (differential regulation). Each step persists its output on the
    instance so it can be inspected, saved, or rerun individually via
    :meth:`run_step`. :meth:`build` runs the full pipeline and returns
    the differential regulation table.

    Parameters
    ----------
    x_data, y_data
        Genes-by-cells expression DataFrames (or AnnData-like objects;
        converted via :func:`anndata_to_dataframe`). The two conditions
        being compared.
    x_label, y_label
        Short labels used as keys in ``data_dict``/``QC_dict``/...
        and to identify each condition in saved output.
    qc_kws
        Overrides for :func:`sc_QC` during the QC step.
    nc_kws
        Overrides for :func:`make_networks` during PC network
        construction. Use this dict to pass ``backend``, ``n_jobs``,
        ``random_state``, etc.
    td_kws
        Overrides for :func:`tensor_decomp`.
    ma_kws
        Overrides for :func:`manifold_alignment`.
    dr_kws
        Overrides for :func:`d_regulation`.
    """

    def __init__(self,
                 x_data: ExpressionData,
                 y_data: ExpressionData,
                 x_label: str,
                 y_label: str,
                 qc_kws: Optional[Kwargs] = None,
                 nc_kws: Optional[Kwargs] = None,
                 td_kws: Optional[Kwargs] = None,
                 ma_kws: Optional[Kwargs] = None,
                 dr_kws: Optional[Kwargs] = None) -> None:
        """See class docstring for parameter descriptions."""
        super().__init__(qc_kws=qc_kws, nc_kws=nc_kws, td_kws=td_kws, ma_kws=ma_kws, dr_kws=dr_kws)
        self.x_label, self.y_label = x_label, y_label
        self.data_dict[x_label] = pd.DataFrame() if isinstance(x_data, str) and x_data == "" else anndata_to_dataframe(x_data)
        self.data_dict[y_label] = pd.DataFrame() if isinstance(y_data, str) and y_data == "" else anndata_to_dataframe(y_data)

    @classmethod
    def get_empty_config(cls) -> Dict[str, object]:
        """Return a blank scTenifoldNet config dict populated with step defaults."""
        config = {"x_data_path": None, "y_data_path": None,
                  "x_label": None, "y_label": None}
        for kw, sig in cls.kw_sigs.items():
            config[kw] = cls.list_kws(kw)
        return config

    @classmethod
    def load_config(cls, config: Dict[str, object]) -> "scTenifoldNet":
        """Build a scTenifoldNet from a config dict, reading data from disk.

        ``x_data_path`` and ``y_data_path`` may be either a 10x folder
        (loaded via :func:`read_folder`) or a CSV/TSV file.
        """
        x_data_path = Path(config.pop("x_data_path"))
        y_data_path = Path(config.pop("y_data_path"))
        if x_data_path.is_dir():
            x_data = read_folder(x_data_path)
        else:
            x_data = pd.read_csv(x_data_path, sep='\t' if x_data_path.suffix == ".tsv" else ",")
        if y_data_path.is_dir():
            y_data = read_folder(y_data_path)
        else:
            y_data = pd.read_csv(y_data_path, sep='\t' if y_data_path.suffix == ".tsv" else ",")
        return cls(x_data, y_data, **config)

    def save(self,
             file_dir: Union[str, Path],
             comps: Union[str, List[str]] = "all",
             verbose: bool = True,
             **kwargs: object) -> None:
        """Save state plus ``x_label``/``y_label`` so :meth:`load` can rebuild."""
        super().save(file_dir, comps, verbose,
                     x_data="", y_data="",
                     x_label=self.x_label, y_label=self.y_label)

    def _norm(self, label):
        self.QC_dict[label] = cpm_norm(self.QC_dict[label])

    def run_step(self,
                 step_name: Literal["qc", "nc", "td", "ma", "dr"],
                 **kwargs: object) -> None:
        """Run a single step of the scTenifoldNet pipeline.

        Steps must be invoked in order — each depends on the state
        produced by the previous one.

        Parameters
        ----------
        step_name
            Which step to run. One of:

            - ``"qc"`` — quality control + CPM normalisation on both
              conditions. Reads ``self.data_dict``; writes
              ``self.QC_dict``.
            - ``"nc"`` — PC network construction on the shared gene
              set. Reads ``self.QC_dict``; writes ``self.network_dict``
              and ``self.shared_gene_names``.
            - ``"td"`` — tensor decomposition + symmetrisation. Reads
              ``self.network_dict``; writes ``self.tensor_dict``.
            - ``"ma"`` — manifold alignment between the two decomposed
              tensors. Writes ``self.manifold``.
            - ``"dr"`` — differential regulation from the aligned
              manifold. Writes ``self.d_regulation``.
        **kwargs
            One-shot overrides for the step. When non-empty these
            replace the dict stored on the instance (``qc_kws``,
            ``nc_kws``, etc.) for this call only.

        Raises
        ------
        ValueError
            If ``step_name`` is not one of the five values above.
        """
        start_time = time.perf_counter()
        if step_name == "qc":
            for label in self.data_dict:
                self._QC(label,
                         **(self.qc_kws if kwargs == {} else kwargs))
                self._norm(label)
                print("finish QC:", label)
        elif step_name == "nc":
            y_gene_names = set(self.QC_dict[self.y_label].index)
            self.shared_gene_names = [gene for gene in self.QC_dict[self.x_label].index if gene in y_gene_names]
            for label, qc_data in self.QC_dict.items():
                self._make_networks(label, data=qc_data.loc[self.shared_gene_names, :],
                                    **(self.nc_kws if kwargs == {} else kwargs))
        elif step_name == "td":
            for label, qc_data in self.QC_dict.items():
                self._tensor_decomp(label, self.shared_gene_names, **(self.td_kws if kwargs == {} else kwargs))
            self.tensor_dict[self.x_label] = (self.tensor_dict[self.x_label] + self.tensor_dict[self.x_label].T) / 2
            self.tensor_dict[self.y_label] = (self.tensor_dict[self.y_label] + self.tensor_dict[self.y_label].T) / 2
        elif step_name == "ma":
            self.manifold = manifold_alignment(self.tensor_dict[self.x_label],
                                               self.tensor_dict[self.y_label],
                                               **(self.ma_kws if kwargs == {} else kwargs))
            self.step_comps["ma"] = self.manifold
        elif step_name == "dr":
            self.d_regulation = d_regulation(self.manifold, **(self.dr_kws if kwargs == {} else kwargs))
            self.step_comps["dr"] = self.d_regulation
        else:
            raise ValueError("This step name is not valid, please choose from qc, nc, td, ma, dr")

        print(f"process {step_name} finished in {time.perf_counter() - start_time} secs.")

    def build(self) -> pd.DataFrame:
        """
        Run the whole pipeline of scTenifoldNet

        Returns
        -------
        d_regulation_df: pd.DataFrame
            Differential regulation result dataframe
        """
        self.run_step("qc")
        self.run_step("nc")
        self.run_step("td")
        self.run_step("ma")
        self.run_step("dr")
        return self.d_regulation


class scTenifoldKnk(scBase):
    """Single-sample scTenifoldKnk virtual-knockout workflow.

    Pipeline order: ``qc`` → ``nc`` → ``td`` → ``ko`` → ``ma`` → ``dr``.
    A wild-type PC network is built from ``data``; the ``ko`` step
    produces a perturbed tensor and the remaining steps compare WT vs.
    KO. Run end-to-end with :meth:`build`, or step-wise with
    :meth:`run_step`.

    Parameters
    ----------
    data
        Genes-by-cells expression matrix (``pandas.DataFrame`` or
        AnnData-like).
    strict_lambda
        Pruning strength forwarded to :func:`strict_direction` when
        post-processing the decomposed WT tensor. ``0`` disables
        pruning.
    ko_method
        How the KO tensor is generated:

        - ``"default"`` — zero out the WT tensor rows for ``ko_genes``.
        - ``"propagation"`` — rebuild PC networks with the targeted
          gene columns masked using :func:`reconstruct_pcnets`, then
          re-decompose.
    ko_genes
        Gene name or iterable of names to knock out. ``None`` stores
        an empty list.
    qc_kws
        Overrides for :func:`sc_QC`. If ``min_exp_avg`` /
        ``min_exp_sum`` are missing, ``run_step("qc")`` injects KO
        defaults (0.05 and 25).
    nc_kws
        Overrides for :func:`make_networks` (``backend``, ``n_jobs``,
        etc.).
    td_kws
        Overrides for :func:`tensor_decomp`.
    ma_kws
        Overrides for :func:`manifold_alignment`. Defaults to
        ``{"d": 2}`` when ``None``.
    dr_kws
        Overrides for :func:`d_regulation`.
    ko_kws
        Extra kwargs forwarded to the KO step (e.g. ``degree`` for the
        propagation method).
    """

    def __init__(self,
                 data: ExpressionData,
                 strict_lambda: float = 0,
                 ko_method: KOMethod = "default",
                 ko_genes: Optional[Union[str, Iterable[str]]] = None,
                 qc_kws: Optional[Kwargs] = None,
                 nc_kws: Optional[Kwargs] = None,
                 td_kws: Optional[Kwargs] = None,
                 ma_kws: Optional[Kwargs] = None,
                 dr_kws: Optional[Kwargs] = None,
                 ko_kws: Optional[Kwargs] = None) -> None:
        """See class docstring for parameter descriptions."""
        ma_kws = {"d": 2} if ma_kws is None else ma_kws
        super().__init__(qc_kws=qc_kws, nc_kws=nc_kws, td_kws=td_kws, ma_kws=ma_kws, dr_kws=dr_kws)
        self.data_dict["WT"] = pd.DataFrame() if isinstance(data, str) and data == "" else anndata_to_dataframe(data)
        self.strict_lambda = strict_lambda
        self.ko_genes = ko_genes if ko_genes is not None else []
        self.ko_method = ko_method
        self.ko_kws = {} if ko_kws is None else ko_kws

    @classmethod
    def get_empty_config(cls) -> Dict[str, object]:
        """Return a blank scTenifoldKnk config dict populated with step defaults."""
        config = {"data_path": None, "strict_lambda": 0,
                  "ko_method": "default", "ko_genes": []}
        for kw, sig in cls.kw_sigs.items():
            config[kw] = cls.list_kws(kw)
        return config

    @classmethod
    def load_config(cls, config: Dict[str, object]) -> "scTenifoldKnk":
        """Build a scTenifoldKnk from a config dict, reading data from disk."""
        data_path = Path(config.pop("data_path"))
        if data_path.is_dir():
            data = read_folder(data_path)
        else:
            data = pd.read_csv(data_path, sep='\t' if data_path.suffix == ".tsv" else ",")
        return cls(data, **config)

    def save(self,
             file_dir: Union[str, Path],
             comps: Union[str, List[str]] = "all",
             verbose: bool = True,
             **kwargs: object) -> None:
        """Save state plus KO-specific fields so :meth:`load` can rebuild."""
        super().save(file_dir, comps, verbose,
                     data="",
                     ko_method=self.ko_method,
                     strict_lambda=self.strict_lambda, ko_genes=self.ko_genes)

    def _get_ko_tensor(self, ko_genes, **kwargs):
        if self.ko_method == "default":
            self.tensor_dict["KO"] = self.tensor_dict["WT"].copy()
            self.tensor_dict["KO"].loc[ko_genes, :] = 0
        elif self.ko_method == "propagation":
            print(self.QC_dict["WT"].index)
            self.network_dict["KO"] = reconstruct_pcnets(self.network_dict["WT"],
                                                         self.QC_dict["WT"],
                                                         ko_gene_id=[self.QC_dict["WT"].index.get_loc(i)
                                                                     for i in ko_genes],
                                                         degree=kwargs.get("degree", 1),
                                                         **self.nc_kws)
            self._tensor_decomp("KO", self.shared_gene_names, **self.td_kws)
            self.tensor_dict["KO"] = strict_direction(self.tensor_dict["KO"], self.strict_lambda).T.copy()
            self.tensor_dict["KO"] = _fill_dataframe_diagonal(self.tensor_dict["KO"], 0)
        else:
            raise ValueError("No such method")

    def run_step(self,
                 step_name: Literal["qc", "nc", "td", "ko", "ma", "dr"],
                 **kwargs: object) -> None:
        """Run a single step of the scTenifoldKnk pipeline.

        Steps must be invoked in order — each depends on the state
        produced by the previous one.

        Parameters
        ----------
        step_name
            Which step to run. One of:

            - ``"qc"`` — quality control on the WT sample (no
              normalisation; injects KO-friendly defaults for
              ``min_exp_avg``/``min_exp_sum`` if missing).
            - ``"nc"`` — PC network construction on the WT QC matrix.
              Writes ``self.network_dict["WT"]`` and
              ``self.shared_gene_names``.
            - ``"td"`` — tensor decomposition of the WT networks plus
              :func:`strict_direction` pruning controlled by
              ``self.strict_lambda``.
            - ``"ko"`` — produce the KO tensor according to
              ``self.ko_method``. ``kwargs`` may contain ``ko_genes``
              to override ``self.ko_genes`` for this call.
            - ``"ma"`` — manifold alignment of WT vs. KO tensors.
            - ``"dr"`` — differential regulation from the aligned
              manifold.
        **kwargs
            One-shot overrides for the step. When non-empty these
            replace the corresponding ``*_kws`` dict on the instance
            for this call only. For ``"ko"`` an explicit ``ko_genes``
            kwarg is consumed before the override logic.

        Raises
        ------
        ValueError
            If ``step_name`` is not one of the six values above, or if
            ``self.ko_method`` is unrecognised during the KO step.
        """
        start_time = time.perf_counter()
        if step_name == "qc":
            if "min_exp_avg" not in self.qc_kws:
                self.qc_kws["min_exp_avg"] = 0.05
            if "min_exp_sum" not in self.qc_kws:
                self.qc_kws["min_exp_sum"] = 25
            self._QC("WT", **(self.qc_kws if kwargs == {} else kwargs))
            # no norm
            print("finish QC: WT")
        elif step_name == "nc":
            self._make_networks("WT", self.QC_dict["WT"], **(self.nc_kws if kwargs == {} else kwargs))
            self.shared_gene_names = self.QC_dict["WT"].index.to_list()
        elif step_name == "td":
            self._tensor_decomp("WT", self.shared_gene_names, **(self.td_kws if kwargs == {} else kwargs))
            self.tensor_dict["WT"] = strict_direction(self.tensor_dict["WT"], self.strict_lambda).T.copy()
        elif step_name == "ko":
            self.tensor_dict["WT"] = _fill_dataframe_diagonal(self.tensor_dict["WT"], 0)
            ko_kwargs = dict(self.ko_kws)
            if kwargs.get("ko_genes") is not None:
                ko_genes = kwargs.pop("ko_genes")
            else:
                ko_genes = self.ko_genes
            ko_kwargs.update(kwargs)
            self._get_ko_tensor(ko_genes, **ko_kwargs)
        elif step_name == "ma":
            self.manifold = manifold_alignment(self.tensor_dict["WT"],
                                               self.tensor_dict["KO"],
                                               **(self.ma_kws if kwargs == {} else kwargs))
            self.step_comps["ma"] = self.manifold
        elif step_name == "dr":
            self.d_regulation = d_regulation(self.manifold, **(self.dr_kws if kwargs == {} else kwargs))
            self.step_comps["dr"] = self.d_regulation
        else:
            raise ValueError("No such step")
        print(f"process {step_name} finished in {time.perf_counter() - start_time} secs.")

    def build(self) -> pd.DataFrame:
        """
        Run the whole pipeline of scTenifoldKnk

        Returns
        -------
        d_regulation_df: pd.DataFrame
            Differential regulation result dataframe
        """
        self.run_step("qc")
        self.run_step("nc")
        self.run_step("td")
        self.run_step("ko")
        self.run_step("ma")
        self.run_step("dr")
        return self.d_regulation

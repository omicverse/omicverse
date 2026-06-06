r"""Single-cell copy-number variation inference (CopyKAT / inferCNV).

Wraps `pycopykat <https://github.com/omicverse/py-CopyKAT>`_ and
`pyinfercnv <https://github.com/omicverse/py-inferCNV>`_ behind a single
:class:`CNV` class with ``method='copykat' | 'infercnv'`` dispatch.

Both backends share a unified output schema:

* ``adata.obsm['X_cnv']``      — cells × genomic-bin CN matrix
* ``adata.uns['cnv']``         — ``dict`` with ``chr_pos`` (chromosome →
  start-bin index), ``method`` and any backend-specific metadata
* ``adata.obs['cnv_score']``   — per-cell mean |CN|
* ``adata.obs['cnv_prediction']`` — categorical (CopyKAT: ``'aneuploid' /
  'diploid'``; inferCNV: NaN by default since the kernel does not
  classify cells without a downstream threshold)

so the plotting helpers in :mod:`omicverse.pl` work with either backend.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from .._registry import register_function


def _check_dep(backend: str) -> None:
    """Lazy-import gate. Raises a clean ImportError pointing at the GitHub repo."""
    if backend == "copykat":
        try:
            import pycopykat  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pycopykat is required for method='copykat'. Install via:\n"
                "  pip install git+https://github.com/omicverse/py-CopyKAT.git"
            ) from e
    elif backend == "infercnv":
        try:
            import pyinfercnv  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pyinfercnv is required for method='infercnv'. Install via:\n"
                "  pip install git+https://github.com/omicverse/py-inferCNV.git"
            ) from e
    else:
        raise ValueError(f"unknown CNV backend {backend!r} (expected 'copykat' or 'infercnv')")


@register_function(
    aliases=["拷贝数变异", "CNV", "copy_number", "拷贝数", "CNA", "copykat", "infercnv"],
    category="single",
    description=(
        "Single-cell copy-number variation inference. Wraps the pure-Python "
        "CopyKAT and inferCNV implementations behind a unified API; results "
        "land in adata.obsm['X_cnv'] / adata.uns['cnv'] / adata.obs['cnv_*'] "
        "and feed the ov.pl.cnv_heatmap / cnv_summary / cnv_umap plots."
    ),
    requires={
        "var": [
            "gene symbols matching the reference genome",
            "chromosome / start / end coordinates (inferCNV only)",
        ],
    },
    produces={
        "obsm": ["X_cnv"],
        "obs": ["cnv_prediction", "cnv_score"],
        "uns": ["cnv"],
    },
    auto_fix="none",
    examples=[
        "# CopyKAT — unsupervised aneuploid / diploid classification",
        "cnv = ov.single.CNV(adata, method='copykat', genome='hg20')",
        "cnv.run()",
        "ov.pl.cnv_heatmap(adata)",
        "",
        "# inferCNV — reference-anchored CN matrix (platform sets gene cutoff)",
        "cnv = ov.single.CNV(adata, method='infercnv')",
        "cnv.run(reference_key='cell_type', reference_cat=['T cell CD4', 'Macrophage'], platform='10x')",
        "ov.pl.cnv_heatmap(adata, groupby='cell_type', split_arms=True)  # p/q arms",
        "",
        "# inferCNV phase2/3 — HMM state calls + CNV regions + subclusters",
        "cnv = ov.single.CNV(adata, method='infercnv')",
        "cnv.run(reference_key='cell_type', reference_cat=['T cell CD4'], platform='10x', HMM=True)",
        "ov.pl.cnv_heatmap(adata, groupby='cnv_subcluster', split_arms=True)",
    ],
    related=["pl.cnv_heatmap", "pl.cnv_summary", "pl.cnv_umap"],
)
class CNV:
    """Single-cell CNV inference dispatcher.

    Parameters
    ----------
    adata : AnnData
        Single-cell expression data. For ``method='copykat'`` should contain
        raw or near-raw counts. For ``method='infercnv'`` should contain
        chromosome coordinates in ``adata.var`` (``chromosome``, ``start``,
        ``end`` columns) and a cell-type annotation column referenced by
        ``reference_key`` in :meth:`run`.
    method : {'copykat', 'infercnv'}
        Which backend to run.
    genome : str, default ``'hg20'``
        Reference genome. ``'hg20'`` (CopyKAT human GRCh38) or ``'mm10'`` for
        CopyKAT; for inferCNV the chromosome/start/end metadata in
        ``adata.var`` is authoritative.
    layer : str or None, default None
        Layer to score against (instead of ``adata.X``).
    raw : bool, default False
        Use ``adata.raw.X`` instead of ``adata.X`` (CopyKAT only — inferCNV
        always uses ``layer`` or ``.X``).
    """

    def __init__(
        self,
        adata: AnnData,
        method: str = "copykat",
        *,
        genome: str = "hg20",
        layer: Optional[str] = None,
        raw: bool = False,
    ) -> None:
        if method not in {"copykat", "infercnv"}:
            raise ValueError(f"method must be 'copykat' or 'infercnv', got {method!r}")
        _check_dep(method)
        self.adata = adata
        self.method = method
        self.genome = genome
        self.layer = layer
        self.raw = raw
        # Populated by .run()
        self.result: Any = None

    # ------------------------------------------------------------------ #
    # public                                                              #
    # ------------------------------------------------------------------ #

    def run(
        self,
        *,
        reference_key: Optional[str] = None,
        reference_cat: Union[str, Sequence[str], None] = None,
        exclude_chromosomes: Optional[Sequence[str]] = ("chrX", "chrY", "chrM"),
        platform: Optional[str] = None,
        n_jobs: int = 1,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "CNV":
        r"""Execute the chosen backend and write results into ``self.adata``.

        Parameters
        ----------
        reference_key : str or None
            Column in ``adata.obs`` that identifies reference (normal) cells
            for inferCNV. Required for ``method='infercnv'`` when you want
            a reference-anchored CN matrix; CopyKAT ignores this (it
            classifies unsupervised).
        reference_cat : str or list of str
            Value(s) in ``adata.obs[reference_key]`` that mark reference cells.
        exclude_chromosomes : sequence of str
            Chromosomes to drop before CNV inference. Default skips sex +
            mitochondrial chromosomes, which carry biological signal that
            confounds the deviation from the genome-wide diploid baseline.
        platform : {'10x', 'smartseq2'} or None
            Sequencing platform for ``method='infercnv'``. Sets the gene-filter
            ``cutoff`` (minimum mean expression) the way R inferCNV recommends:
            ``'10x'`` → ``cutoff=0.1`` (sparse UMI), ``'smartseq2'`` →
            ``cutoff=1.0`` (full-length). **Required for inferCNV** unless you
            pass an explicit ``cutoff=`` — otherwise a ``ValueError`` is raised,
            because the wrong cutoff silently over-/under-filters genes (e.g.
            the SmartSeq2 default 1.0 drops ~3/4 of bins on 10x data, giving an
            over-smoothed heatmap). Ignored by CopyKAT.
        n_jobs : int
            Parallel workers (CopyKAT only).
        verbose : bool
            Stream progress logs from the underlying backend.
        **kwargs
            Forwarded to the backend (``CopykatConfig`` or ``InferCNVConfig``).
            For ``method='infercnv'`` these reach ``InferCNVConfig`` verbatim
            (R kwarg names preserved: ``HMM``, ``HMM_type``, ``mask_nonDE_genes``,
            ``denoise``, ``analysis_mode``). Passing ``HMM=True`` (optionally
            ``HMM_type='i6'|'i3'``) runs inferCNV phase 2/3 and additionally
            writes ``obs['cnv_subcluster']``, ``obsm['X_cnv_hmm_states']``
            (or ``'X_cnv_hmm_states_i3'``) and ``uns['cnv']['cnv_regions']``,
            so ``ov.pl.cnv_heatmap(adata, groupby='cnv_subcluster')`` works.
            Note: phase-3-only results (denoised matrix, Bayes posterior) are
            NOT written back to AnnData — they stay on the returned
            ``InferCNVResult`` (``cnv.result``). ``obs['cnv_prediction']``
            remains NaN for inferCNV (it has no per-cell tumour/normal call;
            threshold ``obs['cnv_score']`` yourself if you need one).

        Returns
        -------
        self
            ``CNV`` instance with ``.result`` populated. Side-effects:
            writes ``adata.obsm['X_cnv']``, ``adata.uns['cnv']``,
            ``adata.obs['cnv_score']`` and (CopyKAT only) ``adata.obs['cnv_prediction']``.
        """
        if self.method == "copykat":
            return self._run_copykat(
                exclude_chromosomes=exclude_chromosomes,
                n_jobs=n_jobs,
                verbose=verbose,
                **kwargs,
            )
        return self._run_infercnv(
            reference_key=reference_key,
            reference_cat=reference_cat,
            exclude_chromosomes=exclude_chromosomes,
            platform=platform,
            verbose=verbose,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # backends                                                            #
    # ------------------------------------------------------------------ #

    def _run_copykat(
        self,
        *,
        exclude_chromosomes: Optional[Sequence[str]],
        n_jobs: int,
        verbose: bool,
        **kwargs: Any,
    ) -> "CNV":
        import logging
        from pycopykat import CopykatConfig, copykat

        if verbose:
            logging.getLogger("pycopykat").setLevel(logging.INFO)

        # CopyKAT wants a (genes × cells) DataFrame with gene names in the index.
        mat = self._extract_counts_df()

        cfg = CopykatConfig(
            id_type=kwargs.pop("id_type", "Symbol"),
            genome=kwargs.pop("genome", self.genome),
            n_jobs=n_jobs,
            **kwargs,
        )
        result = copykat(mat, config=cfg)

        # cna_mat: bins (MultiIndex chrom/start/end) × cells. Transpose to cells × bins
        # and align ordering with adata.obs_names so the obsm assignment is safe.
        cna = result.cna_mat
        cell_to_col = {c: i for i, c in enumerate(cna.columns)}
        # Some upstream cells get filtered by stage-1 chrom coverage; we right-pad
        # those rows with NaN so the obsm matrix still has shape (n_obs, n_bins).
        n_bins = cna.shape[0]
        X_cnv = np.full((self.adata.n_obs, n_bins), np.nan, dtype=np.float32)
        for i, name in enumerate(self.adata.obs_names):
            col = cell_to_col.get(name)
            if col is not None:
                X_cnv[i] = cna.iloc[:, col].to_numpy(dtype=np.float32)

        chr_pos = _chr_pos_from_bin_index(cna.index)
        bin_meta = pd.DataFrame(cna.index.to_frame(index=False))

        # Prediction (aneuploid / diploid). Missing for cells filtered out.
        pred_col = "copykat.pred" if "copykat.pred" in result.prediction.columns else result.prediction.columns[-1]
        pred = result.prediction.set_index("cell")[pred_col].reindex(self.adata.obs_names)

        self.adata.obsm["X_cnv"] = X_cnv
        self.adata.obs["cnv_prediction"] = pd.Categorical(pred)
        self.adata.obs["cnv_score"] = np.nanmean(np.abs(X_cnv), axis=1).astype(np.float32)
        self.adata.uns["cnv"] = {
            "method": "copykat",
            "chr_pos": chr_pos,
            "bin_meta": bin_meta,
            "exclude_chromosomes": list(exclude_chromosomes or []),
        }
        self.result = result
        return self

    def _run_infercnv(
        self,
        *,
        reference_key: Optional[str],
        reference_cat: Union[str, Sequence[str], None],
        exclude_chromosomes: Optional[Sequence[str]],
        platform: Optional[str],
        verbose: bool,
        **kwargs: Any,
    ) -> "CNV":
        import logging
        from pyinfercnv import InferCNVConfig, infercnv

        if verbose:
            logging.getLogger("pyinfercnv").setLevel(logging.INFO)

        # Resolve the gene-filter cutoff from `platform` (R inferCNV convention:
        # 10x -> 0.1, SmartSeq2 -> 1.0). Require an explicit choice so the wrong
        # default cannot silently over-/under-filter genes.
        if "cutoff" not in kwargs:
            if platform is None:
                raise ValueError(
                    "method='infercnv' requires platform='10x' or "
                    "platform='smartseq2' (sets the gene-filter cutoff: "
                    "10x->0.1, smartseq2->1.0), or an explicit cutoff=<float>. "
                    "Pass one, e.g. cnv.run(..., platform='10x')."
                )
            _p = platform.lower().replace("-", "").replace("_", "").replace(" ", "")
            _platform_cutoff = {
                "10x": 0.1, "tenx": 0.1,
                "smartseq2": 1.0, "smartseq": 1.0, "ss2": 1.0,
            }
            if _p not in _platform_cutoff:
                raise ValueError(
                    f"unknown platform {platform!r}; use '10x' or 'smartseq2', "
                    "or pass an explicit cutoff=<float>."
                )
            kwargs["cutoff"] = _platform_cutoff[_p]
        elif platform is not None:
            raise ValueError(
                "pass either platform= or an explicit cutoff=, not both."
            )

        # inferCNV needs chromosome / start / end in adata.var.
        for col in ("chromosome", "start", "end"):
            if col not in self.adata.var.columns:
                raise ValueError(
                    f"adata.var must contain {col!r} for method='infercnv'. "
                    "Annotate gene coordinates first (see pyinfercnv.io.genome.load_gene_positions)."
                )

        cfg = InferCNVConfig(
            counts_layer=self.layer,
            **kwargs,
        )
        # write_to_anndata(key_added='cnv') is called internally when inplace=True
        result = infercnv(
            self.adata,
            config=cfg,
            reference_key=reference_key,
            reference_cat=reference_cat,
            exclude_chromosomes=exclude_chromosomes,
            key_added="cnv",
            inplace=True,
        )

        # Per-cell mean |CN| as a portable "tumour-ness" score. inferCNV does
        # not classify cells; downstream users can threshold this.
        X_cnv = self.adata.obsm["X_cnv"]
        score = np.asarray(np.abs(X_cnv).mean(axis=1)).reshape(-1).astype(np.float32)
        self.adata.obs["cnv_score"] = score
        # Tag method + carry chr_pos for the plot layer
        uns = dict(self.adata.uns.get("cnv", {}))
        uns["method"] = "infercnv"
        # Build a chromosome-bin lookup if not present (pyinfercnv writes chr_pos already)
        uns.setdefault("chr_pos", _chr_pos_from_obsm_var(self.adata))
        self.adata.uns["cnv"] = uns
        # No prediction column from inferCNV — leave it unset so plotting code
        # falls back to cnv_score / cell-type colouring.
        if "cnv_prediction" not in self.adata.obs:
            self.adata.obs["cnv_prediction"] = pd.Categorical(
                pd.Series([pd.NA] * self.adata.n_obs, index=self.adata.obs_names)
            )
        self.result = result
        return self

    # ------------------------------------------------------------------ #
    # helpers                                                             #
    # ------------------------------------------------------------------ #

    def _extract_counts_df(self) -> pd.DataFrame:
        """Materialise a (genes × cells) DataFrame for CopyKAT."""
        if self.raw and self.adata.raw is not None:
            X = self.adata.raw.X
            var_names = self.adata.raw.var_names
        elif self.layer is not None:
            X = self.adata.layers[self.layer]
            var_names = self.adata.var_names
        else:
            X = self.adata.X
            var_names = self.adata.var_names

        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X)
        # AnnData is cells × genes — CopyKAT wants genes × cells.
        return pd.DataFrame(X.T, index=var_names, columns=self.adata.obs_names)


# ------------------------------------------------------------------ #
# small utilities                                                     #
# ------------------------------------------------------------------ #


def _chr_pos_from_bin_index(idx: pd.MultiIndex) -> dict[str, int]:
    """Map chromosome → start-bin index from a CopyKAT bin MultiIndex."""
    df = idx.to_frame(index=False)
    chrom_col = df.columns[0]
    out: dict[str, int] = {}
    for i, c in enumerate(df[chrom_col].astype(str)):
        out.setdefault(str(c), i)
    return out


def _chr_pos_from_obsm_var(adata: AnnData) -> dict[str, int]:
    """Fallback chr_pos from adata.var (used if pyinfercnv didn't write one)."""
    if "chromosome" not in adata.var:
        return {}
    chrom = adata.var["chromosome"].astype(str)
    out: dict[str, int] = {}
    for i, c in enumerate(chrom):
        out.setdefault(c, i)
    return out

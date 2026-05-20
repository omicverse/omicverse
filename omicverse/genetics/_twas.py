"""Transcriptome-wide association study for ``ov.genetics`` — TWAS backend.

Wraps the standalone :mod:`pytwas` package — the (S-)PrediXcan /
S-MultiXcan family (Gamazon *et al.* 2015; Barbeira *et al.* 2018, 2019).
:func:`twas` is a single dispatcher: ``method='spredixcan'`` runs a
single-tissue summary-statistics TWAS, ``method='smultixcan'`` combines
tissues, and ``method='predixcan'`` runs the individual-level TWAS.
"""
from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from .._registry import register_function


def _canonicalize_gwas(
    df: pd.DataFrame,
    snp_column, effect_allele_column, non_effect_allele_column,
    zscore_column, beta_column, se_column, pvalue_column,
):
    """Rename a raw GWAS DataFrame to pytwas' canonical schema.

    ``pytwas.spredixcan`` only applies column mapping when it reads from
    ``gwas_file``; a DataFrame passed via ``gwas=`` must already use the
    canonical ``snp`` / ``effect_allele`` / ``non_effect_allele`` /
    ``zscore`` (or ``beta`` / ``se`` / ``pvalue``) names. This helper
    bridges that gap so users can pass a plain results DataFrame.
    """
    canon = {"snp", "effect_allele", "non_effect_allele"}
    if canon.issubset(set(df.columns)):
        return df  # already canonical (e.g. from pytwas.load_gwas).
    rename = {
        snp_column: "snp",
        effect_allele_column: "effect_allele",
        non_effect_allele_column: "non_effect_allele",
    }
    if zscore_column:
        rename[zscore_column] = "zscore"
    if beta_column:
        rename[beta_column] = "beta"
    if se_column:
        rename[se_column] = "se"
    if pvalue_column:
        rename[pvalue_column] = "pvalue"
    rename = {k: v for k, v in rename.items() if k in df.columns}
    return df.rename(columns=rename)


def _require_twas():
    """Import :mod:`pytwas` with a friendly error if it is missing."""
    try:
        import pytwas  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.genetics.twas requires the pytwas backend: "
            "`pip install pytwas` (or `pip install omicverse[genetics]`)."
        ) from exc
    return pytwas


def _spredixcan_in_memory(
    pytwas, model, covariance, gwas,
    snp_column, effect_allele_column, non_effect_allele_column,
):
    """Run S-PrediXcan from an in-memory model + covariance.

    ``pytwas.spredixcan`` reads its model and covariance from disk. When a
    notebook has already built a :class:`pytwas.PredictionModel` (e.g.
    from :func:`build_twas_model`) and a per-gene SNP covariance, this
    routes straight through ``pytwas.gene_association`` — no SQLite /
    text-file round trip — and returns the same per-gene TWAS table.
    """
    from scipy import stats as _stats

    if isinstance(covariance, pd.DataFrame):
        covariance = pytwas.CovarianceDB(covariance)

    weights = model.weights
    gwas_idx = gwas.set_index(snp_column)
    rows = []
    for gene, sub in weights.groupby("gene"):
        gi = {}
        for _, w in sub.iterrows():
            rsid = w["rsid"]
            if rsid not in gwas_idx.index:
                continue
            rec = gwas_idx.loc[rsid]
            if isinstance(rec, pd.DataFrame):           # duplicate id
                rec = rec.iloc[0]
            same = str(rec[effect_allele_column]) == str(w["effect_allele"])
            sign = 1.0 if same else -1.0
            z = float(rec["Z"]) if "Z" in rec else float(rec.get("zscore", np.nan))
            b = (float(rec["BETA"]) if "BETA" in rec
                 else float(rec.get("beta", np.nan)))
            gi[rsid] = (sign * z, sign * b)
        if not gi:
            continue
        ga = pytwas.gene_association(gene, model, gi, covariance)
        rows.append({
            "gene": gene, "zscore": ga.zscore,
            "effect_size": ga.effect_size,
            "n_snps_used": ga.n_snps_used,
        })
    res = pd.DataFrame(rows)
    if not res.empty:
        res["pvalue"] = 2.0 * _stats.norm.sf(np.abs(res["zscore"].astype(float)))
        res = res.sort_values("pvalue").reset_index(drop=True)
    return res


@register_function(
    aliases=[
        "twas", "transcriptome_wide_association", "predixcan", "spredixcan",
        "smultixcan", "metaxcan",
        "转录组关联分析", "TWAS分析", "全转录组关联",
    ],
    category="genetics",
    description=(
        "Transcriptome-wide association study (TWAS) with the PrediXcan / "
        "MetaXcan family — tests genetically-predicted gene expression for "
        "association with a trait, nominating candidate effector genes "
        "from a GWAS. ``method`` selects the algorithm: ``'spredixcan'`` "
        "(default; single-tissue S-PrediXcan from GWAS summary statistics "
        "+ a prediction model + a SNP-covariance file), ``'smultixcan'`` "
        "(S-MultiXcan, aggregates S-PrediXcan results across tissues into "
        "one chi-square test), or ``'predixcan'`` (individual-level "
        "PrediXcan from genotype dosages + phenotype). Wraps the pytwas "
        "backend."
    ),
    examples=[
        "ov.genetics.twas('model.db', covariance='cov.txt.gz', gwas=gwas_df, method='spredixcan')",
        "ov.genetics.twas(spx_results, models=models, covariance='snp_cov.txt.gz', method='smultixcan')",
        "ov.genetics.twas('model.db', dosages=dos, phenotype=y, method='predixcan')",
    ],
    related=["ov.genetics.load_twas_model", "ov.genetics.heritability",
             "ov.genetics.manhattan"],
)
def twas(
    gwas=None,
    model: Optional[Union[str, Dict]] = None,
    covariance: Optional[Union[str, object]] = None,
    *,
    method: str = "spredixcan",
    gwas_file: Optional[str] = None,
    models: Optional[Dict] = None,
    dosages: Optional[pd.DataFrame] = None,
    phenotype: Optional[np.ndarray] = None,
    mode: str = "linear",
    snp_column: str = "SNP",
    effect_allele_column: str = "A1",
    non_effect_allele_column: str = "A2",
    zscore_column: Optional[str] = None,
    beta_column: Optional[str] = None,
    se_column: Optional[str] = None,
    pvalue_column: Optional[str] = None,
    **kwargs,
):
    """Run a transcriptome-wide association study.

    Parameters
    ----------
    gwas
        For ``method='spredixcan'``: a GWAS summary-statistics DataFrame.
        For ``method='smultixcan'``: a dict ``{tissue: spredixcan_df}``.
    model
        Prediction model — a ``.db`` path (PrediXcan / S-PrediXcan) or a
        loaded :class:`pytwas.PredictionModel`.
    covariance
        SNP-covariance file path (or :class:`pytwas.CovarianceDB`).
    method
        ``'spredixcan'``, ``'smultixcan'`` or ``'predixcan'``.
    gwas_file
        Alternative to ``gwas`` for ``spredixcan`` — a GWAS file path.
    models
        For ``method='smultixcan'``: a dict ``{tissue: model}``.
    dosages
        For ``method='predixcan'``: a ``samples x SNPs`` dosage DataFrame.
    phenotype
        For ``method='predixcan'``: the per-sample phenotype vector.
    mode
        For ``method='predixcan'``: ``'linear'`` or ``'logistic'``.
    snp_column, effect_allele_column, non_effect_allele_column
        GWAS column names (S-PrediXcan).
    zscore_column, beta_column, se_column, pvalue_column
        GWAS effect-size column names — supply whichever your file has.
    **kwargs
        Forwarded to the backend function.

    Returns
    -------
    pandas.DataFrame
        Per-gene TWAS table — z-score / chi-square, effect size and
        p-value.
    """
    pytwas = _require_twas()
    key = str(method).lower().strip()

    if key == "spredixcan":
        if model is None or covariance is None:
            raise ValueError(
                "method='spredixcan' requires a model (.db) and covariance."
            )
        if gwas is None and gwas_file is None:
            raise ValueError(
                "method='spredixcan' requires gwas= (DataFrame) or gwas_file=."
            )
        # In-memory route: a PredictionModel + a covariance DataFrame /
        # CovarianceDB run directly, no .db / .txt files needed. This lets
        # a notebook build a model from an eQTL table in memory and run
        # S-PrediXcan without round-tripping through SQLite on disk.
        if not isinstance(model, str):
            return _spredixcan_in_memory(
                pytwas, model, covariance, gwas,
                snp_column, effect_allele_column, non_effect_allele_column,
            )
        gwas_df = gwas
        if isinstance(gwas_df, pd.DataFrame):
            gwas_df = _canonicalize_gwas(
                gwas_df, snp_column, effect_allele_column,
                non_effect_allele_column, zscore_column, beta_column,
                se_column, pvalue_column,
            )
        return pytwas.spredixcan(
            model, covariance, gwas_file=gwas_file, gwas=gwas_df,
            snp_column=snp_column,
            effect_allele_column=effect_allele_column,
            non_effect_allele_column=non_effect_allele_column,
            zscore_column=zscore_column, beta_column=beta_column,
            se_column=se_column, pvalue_column=pvalue_column, **kwargs,
        )

    if key == "smultixcan":
        if gwas is None or models is None or covariance is None:
            raise ValueError(
                "method='smultixcan' requires gwas= (a {tissue: spredixcan_df} "
                "dict), models= and covariance=."
            )
        return pytwas.smultixcan(gwas, models, covariance, **kwargs)

    if key == "predixcan":
        if model is None or dosages is None or phenotype is None:
            raise ValueError(
                "method='predixcan' requires model, dosages and phenotype."
            )
        return pytwas.predixcan(
            model, dosages, np.asarray(phenotype, dtype=float),
            mode=mode, **kwargs,
        )

    raise ValueError(
        f"method must be 'spredixcan', 'smultixcan' or 'predixcan', "
        f"got {method!r}"
    )


@register_function(
    aliases=["load_twas_model", "load_model", "prediction_model", "TWAS模型加载"],
    category="genetics",
    description=(
        "Load a PrediXcan / S-PrediXcan gene-expression prediction model "
        "from a ``.db`` SQLite file into a :class:`pytwas.PredictionModel`. "
        "Wraps :func:`pytwas.load_model`."
    ),
    examples=[
        "ov.genetics.load_twas_model('elastic_net_model.db')",
    ],
    related=["ov.genetics.twas"],
)
def load_twas_model(path: str, *, snp_key: Optional[str] = None):
    """Load a TWAS prediction model from a ``.db`` file.

    Parameters
    ----------
    path
        Path to the SQLite ``.db`` prediction-model file.
    snp_key
        Optional SNP-key column override.

    Returns
    -------
    pytwas.PredictionModel
        The loaded prediction model.
    """
    pytwas = _require_twas()
    return pytwas.load_model(path, snp_key=snp_key)


@register_function(
    aliases=[
        "make_twas_covariance", "twas_covariance", "snp_covariance",
        "构建TWAS协方差", "SNP协方差矩阵",
    ],
    category="genetics",
    description=(
        "Build the per-gene SNP-covariance table that S-PrediXcan needs, "
        "from a TWAS prediction model and a reference genotype AnnData. For "
        "each gene, the covariance of its model SNPs is computed from the "
        "reference dosages; SNPs absent from the reference fall back to the "
        "Hardy-Weinberg variance 2p(1-p) of an allele-frequency table. "
        "Returns the ``GENE RSID1 RSID2 VALUE`` long-format table that "
        "``ov.genetics.twas(method='spredixcan')`` accepts in memory. "
        "Pure numpy / pandas."
    ),
    examples=[
        "cov = ov.genetics.make_twas_covariance(model, geno_qc)",
        "cov = ov.genetics.make_twas_covariance(model, geno_qc, "
        "freq=eqtl.set_index('variant')['maf'])",
    ],
    related=["ov.genetics.twas", "ov.genetics.build_twas_model"],
)
def make_twas_covariance(model, reference=None, *, freq=None):
    """Build the per-gene SNP-covariance table for S-PrediXcan.

    Parameters
    ----------
    model
        A :class:`pytwas.PredictionModel` (e.g. from
        :func:`build_twas_model`) whose ``.weights`` lists the SNPs of
        every gene.
    reference
        Optional genotype AnnData (``samples x SNPs``, 0/1/2 dosages) used
        as the LD reference. SNPs found here get an empirical covariance.
    freq
        Optional ``Series`` mapping SNP id -> allele frequency. SNPs not in
        ``reference`` get the Hardy-Weinberg variance ``2p(1-p)`` from
        this; off-diagonal covariances of such SNPs are set to 0.

    Returns
    -------
    pandas.DataFrame
        A ``GENE RSID1 RSID2 VALUE`` long-format covariance table.
    """
    ref_X = None
    ref_idx = {}
    if reference is not None:
        ref_X = np.asarray(reference.X, dtype=float)
        if hasattr(reference.X, "toarray"):
            ref_X = reference.X.toarray().astype(float)
        ref_idx = {s: i for i, s in enumerate(reference.var_names)}
    if freq is None:
        freq = {}
    else:
        freq = pd.Series(freq)
        freq = freq[~freq.index.duplicated()].to_dict()

    rows = []
    for gene, sub in model.weights.groupby("gene"):
        snps = list(dict.fromkeys(sub["rsid"]))
        in_ref = [s for s in snps if s in ref_idx]
        cov = None
        if ref_X is not None and len(in_ref) > 0:
            cols = [ref_idx[s] for s in in_ref]
            cov = np.atleast_2d(np.cov(ref_X[:, cols], rowvar=False))
        ref_pos = {s: k for k, s in enumerate(in_ref)}
        for i, s1 in enumerate(snps):
            for s2 in snps[i:]:
                if s1 in ref_pos and s2 in ref_pos and cov is not None:
                    val = float(cov[ref_pos[s1], ref_pos[s2]])
                elif s1 == s2:
                    p = float(freq.get(s1, np.nan))
                    val = 2.0 * p * (1.0 - p) if not np.isnan(p) else np.nan
                else:
                    val = 0.0
                rows.append((gene, s1, s2, val))
    return pd.DataFrame(rows, columns=["GENE", "RSID1", "RSID2", "VALUE"])

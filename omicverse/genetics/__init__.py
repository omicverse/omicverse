"""
Statistical genetics for omicverse — a unified post-GWAS analysis suite.

``ov.genetics`` is the analogue of ``ov.protein`` for human statistical
genetics: it threads the major post-GWAS analyses — eQTL mapping,
fine-mapping, colocalization, Mendelian randomization, single-cell
disease-relevance scoring, LD score regression, and TWAS — behind one
registered, dispatch-based API. Unlike single-cell modules, genetics
data is heterogeneous: GWAS summary statistics are DataFrames, genotypes
are matrices, single-cell data is AnnData — so each function takes the
data type its task naturally uses rather than forcing everything into
AnnData.

The statistical machinery lives in **seven standalone packages** that
omicverse ships as separate releases, plus a backend-free GWAS core:

* :mod:`pymatrixeqtl`  — Matrix eQTL: cis / trans eQTL mapping (Shabalin 2012)
* :mod:`pysusie`       — SuSiE: Bayesian fine-mapping (Wang 2020)
* :mod:`pycoloc`       — coloc: Bayesian colocalization (Giambartolomei 2014)
* :mod:`pytwosamplemr` — TwoSampleMR / MendelianRandomization estimators
* :mod:`pyscdrs`       — scDRS: single-cell disease-relevance scoring (Zhang 2022)
* :mod:`pyldsc`        — LDSC: LD score regression / SEG (Bulik-Sullivan 2015)
* :mod:`pytwas`        — (S-)PrediXcan / S-MultiXcan TWAS (Barbeira 2018)

Each is a thin sidecar; ``ov.genetics`` wraps them behind ``method=``
dispatchers so users learn one API instead of seven. The GWAS core
(``gwas_qc``, ``gwas_association``, ``genomic_inflation``) is implemented
directly on numpy / scipy / statsmodels — no backend required.

Install the backends with::

    pip install omicverse[genetics]

Quick-start
-----------
>>> import omicverse as ov
>>> # 1. eQTL mapping
>>> eq = ov.genetics.eqtl_map(genotype, expression, model='linear')
>>> # 2. fine-map a locus from summary stats
>>> fit = ov.genetics.finemap(z=zscores, R=ld, n=10000, method='susie_rss')
>>> cs = ov.genetics.get_credible_sets(fit, R=ld)
>>> # 3. colocalize a GWAS with an eQTL
>>> co = ov.genetics.colocalize(gwas_dataset, eqtl_dataset, method='abf')
>>> # 4. Mendelian randomization
>>> mr = ov.genetics.mendelian_randomization(mr_input, method='ivw')
>>> # 5. single-cell disease relevance
>>> df = ov.genetics.disease_relevance_score(adata, gene_set=disease_genes)
>>> # 6. heritability / TWAS
>>> h2 = ov.genetics.heritability('trait.sumstats', ref_ld='ldsc', w_ld='w')
>>> tw = ov.genetics.twas('model.db', covariance='cov.txt.gz', gwas=gwas_df)

Pipeline stages
---------------
I/O                       ``read_sumstats``, ``read_plink``, ``read_vcf``
GWAS core (no backend)    ``gwas_qc``, ``gwas_association``,
                          ``genomic_inflation``
eQTL mapping              ``eqtl_map`` (linear / anova / linear_cross)
Fine-mapping              ``finemap`` (susie / susie_rss),
                          ``get_credible_sets``, ``get_pip``
Colocalization            ``colocalize`` (abf / susie / signals),
                          ``finemap_abf``, ``coloc_sensitivity``
Mendelian randomization   ``mendelian_randomization`` (ivw / egger /
                          median / mode / maxlik / divw / conmix /
                          lasso / cml / all), ``harmonize``,
                          ``mr_steiger``, ``mr_heterogeneity``,
                          ``mr_pleiotropy``
Single-cell relevance     ``disease_relevance_score``, ``score_downstream``
Heritability / LDSC       ``heritability``, ``genetic_correlation``,
                          ``partitioned_heritability``, ``ldsc_cell_type``,
                          ``munge_sumstats``
TWAS                      ``twas`` (spredixcan / smultixcan / predixcan),
                          ``load_twas_model``
Plotting                  ``manhattan``, ``qqplot``, ``regional_plot``,
                          ``coloc_plot``, ``mr_scatter``, ``mr_forest``,
                          ``finemap_plot``

All backend imports (``pymatrixeqtl``, ``pysusie`` …) are deferred to
call-time — ``import omicverse.genetics`` does no heavy work and succeeds
even when some backends are not installed.
"""
from __future__ import annotations

import importlib as _importlib


# Lazy public surface — single source of truth for what's exposed.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # I/O
    "read_sumstats":             (".io", "read_sumstats"),
    "read_plink":                (".io", "read_plink"),
    "read_vcf":                  (".io", "read_vcf"),
    # GWAS core (backend-free)
    "gwas_qc":                   ("._gwas", "gwas_qc"),
    "gwas_association":          ("._gwas", "gwas_association"),
    "genomic_inflation":         ("._gwas", "genomic_inflation"),
    # eQTL mapping
    "eqtl_map":                  ("._eqtl", "eqtl_map"),
    # Fine-mapping
    "finemap":                   ("._finemap", "finemap"),
    "get_credible_sets":         ("._finemap", "get_credible_sets"),
    "get_pip":                   ("._finemap", "get_pip"),
    # Colocalization
    "colocalize":                ("._coloc", "colocalize"),
    "finemap_abf":               ("._coloc", "finemap_abf"),
    "coloc_sensitivity":         ("._coloc", "coloc_sensitivity"),
    # Mendelian randomization
    "mendelian_randomization":   ("._mr", "mendelian_randomization"),
    "harmonize":                 ("._mr", "harmonize"),
    "mr_steiger":                ("._mr", "mr_steiger"),
    "mr_heterogeneity":          ("._mr", "mr_heterogeneity"),
    "mr_pleiotropy":             ("._mr", "mr_pleiotropy"),
    # Single-cell disease relevance
    "disease_relevance_score":   ("._scdrs", "disease_relevance_score"),
    "score_downstream":          ("._scdrs", "score_downstream"),
    # Heritability / LD score regression
    "heritability":              ("._ldsc", "heritability"),
    "genetic_correlation":       ("._ldsc", "genetic_correlation"),
    "partitioned_heritability":  ("._ldsc", "partitioned_heritability"),
    "ldsc_cell_type":            ("._ldsc", "ldsc_cell_type"),
    "munge_sumstats":            ("._ldsc", "munge_sumstats"),
    # TWAS
    "twas":                      ("._twas", "twas"),
    "load_twas_model":           ("._twas", "load_twas_model"),
    # Plotting
    "manhattan":                 (".plotting", "manhattan"),
    "qqplot":                    (".plotting", "qqplot"),
    "regional_plot":             (".plotting", "regional_plot"),
    "coloc_plot":                (".plotting", "coloc_plot"),
    "mr_scatter":                (".plotting", "mr_scatter"),
    "mr_forest":                 (".plotting", "mr_forest"),
    "finemap_plot":              (".plotting", "finemap_plot"),
}

_LAZY_SUBMODULES = {"io", "plotting"}

_REGISTRY_SUBMODULES = (
    ".io",
    "._gwas",
    "._eqtl",
    "._finemap",
    "._coloc",
    "._mr",
    "._scdrs",
    "._ldsc",
    "._twas",
    ".plotting",
)


def _hydrate_registry() -> None:
    """Force-import every @register_function-bearing submodule so the global
    registry sees ov.genetics at export time. Called from
    :mod:`omicverse._registry._hydrate_registry_for_export`."""
    for mod in _REGISTRY_SUBMODULES:
        try:
            _importlib.import_module(mod, __name__)
        except Exception:
            # Optional backends (pymatrixeqtl, pysusie, …) may be missing —
            # register whatever loads cleanly.
            continue


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = _importlib.import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    if name in _LAZY_SUBMODULES:
        module = _importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(list(globals().keys())
                      + list(_LAZY_ATTRS.keys())
                      + list(_LAZY_SUBMODULES)))


__version__ = "0.1.0"

__all__ = list(_LAZY_ATTRS.keys()) + list(_LAZY_SUBMODULES)

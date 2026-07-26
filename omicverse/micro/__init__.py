r"""
Microbiome analysis utilities.

Downstream 16S / amplicon analysis on top of the AnnData produced by
:mod:`omicverse.alignment.amplicon_16s_pipeline`
(samples × ASVs, ``var`` carries 7-rank SINTAX taxonomy + ASV sequence).

Classes
-------
Alpha : Alpha-diversity (Shannon / Simpson / Chao1 / Observed / Faith PD)
Beta  : Beta-diversity distance matrices (Bray-Curtis / Jaccard / Aitchison
          / UniFrac — UniFrac requires ``unifrac`` package + phylogenetic tree)
Ordinate : PCoA / NMDS / RDA / CCA ordinations (via scikit-bio / sklearn)
DA    : Differential-abundance testing (Wilcoxon, pyDESeq2, ANCOM-BC)

Functions
---------
rarefy                 : Subsample counts to a common depth
filter_by_prevalence   : Drop rare taxa below a prevalence threshold
collapse_taxa          : Collapse ASV counts to a given rank (e.g. genus)
clr / ilr              : Centred / isometric log-ratio transforms
attach_tree            : Attach a newick phylogenetic tree to adata.uns

Examples
--------
>>> import omicverse as ov
>>> adata = ov.alignment.amplicon_16s_pipeline(fastq_dir='raw/', ...)
>>> # diversity
>>> alpha = ov.micro.Alpha(adata).run(metrics=['shannon', 'observed_otus'])
>>> beta  = ov.micro.Beta(adata).run(metric='braycurtis', rarefy=True)
>>> # ordination
>>> ord_  = ov.micro.Ordinate(adata, dist_key='braycurtis').pcoa(n=3)
>>> # differential abundance at genus level
>>> da    = ov.micro.DA(adata).wilcoxon(group_key='group', rank='genus')
"""
from __future__ import annotations

import importlib

# Map each public name → (relative submodule, attribute). Imports are
# deferred until the name is first accessed, so environments missing
# optional extras (scikit-bio, ete3, pydeseq2, …) can still
# ``import omicverse.micro`` without failing.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # diversity
    "Alpha":           ("._diversity", "Alpha"),
    "Beta":            ("._diversity", "Beta"),
    "ALPHA_METRICS":   ("._diversity", "ALPHA_METRICS"),
    "BETA_METRICS":    ("._diversity", "BETA_METRICS"),
    # ordination
    "Ordinate":        ("._ord", "Ordinate"),
    # differential abundance
    "DA":              ("._da", "DA"),
    # preprocessing
    "rarefy":               ("._pp", "rarefy"),
    "filter_by_prevalence": ("._pp", "filter_by_prevalence"),
    "collapse_taxa":        ("._pp", "collapse_taxa"),
    "clr":                  ("._pp", "clr"),
    "ilr":                  ("._pp", "ilr"),
    # phylogeny
    "attach_tree":          ("._phylo", "attach_tree"),
    # meta-analysis
    "combine_studies":      ("._meta", "combine_studies"),
    "meta_da":              ("._meta", "meta_da"),
    # paired microbe ↔ metabolite integration
    "fetch_franzosa_ibd_2019": ("._pair", "fetch_franzosa_ibd_2019"),
    "simulate_paired":      ("._pair", "simulate_paired"),
    "paired_spearman":      ("._pair", "paired_spearman"),
    "paired_cca":           ("._pair", "paired_cca"),
    "MMvec":                ("._pair", "MMvec"),
    "plot_mmvec_training":           ("._pair", "plot_mmvec_training"),
    "plot_cca_scatter":              ("._pair", "plot_cca_scatter"),
    "plot_cooccurrence":             ("._pair", "plot_cooccurrence"),
    "plot_embedding_biplot":         ("._pair", "plot_embedding_biplot"),
    "plot_paired_method_comparison": ("._pair", "plot_paired_method_comparison"),
}

__all__ = sorted(_LAZY_ATTRS)

_REGISTRY_SUBMODULES = ("._diversity", "._ord", "._da", "._pp", "._phylo",
                        "._meta", "._pair")


def _hydrate_registry() -> None:
    """Force-import every ``@register_function``-bearing submodule so the
    global registry sees ``ov.micro`` at export time. Called from
    :func:`omicverse._registry._hydrate_registry_for_export`."""
    for mod in _REGISTRY_SUBMODULES:
        try:
            importlib.import_module(mod, package=__name__)
        except Exception:
            # Optional backends may be missing — register what loads cleanly.
            continue


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_path, package=__name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_ATTRS})

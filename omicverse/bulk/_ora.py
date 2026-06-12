"""Pure-NumPy/SciPy over-representation analysis (ORA) — a single-process,
offline replacement for ``gseapy.enrichr``'s *local* mode.

Why this exists
---------------
``gseapy.enrichr`` either (a) POSTs the gene list to the Enrichr web service
(needs network, non-deterministic, fails on outages) or (b) runs a local
hypergeometric test but still drags in the whole ``gseapy`` package. omicverse
now ships its own GSEA/ORA backends, so this module reimplements the local
hypergeometric over-representation test directly:

* gene-set libraries given by name (``'GO_Biological_Process_2021'``,
  ``'KEGG_2021_Human'``, …) are resolved **offline** through
  :func:`omicverse.utils.geneset_prepare` (download-and-cache), then tested
  locally — no Enrichr compute API;
* the statistic is the one-sided Fisher/hypergeometric test
  (``scipy.stats.hypergeom.sf``) with Benjamini-Hochberg FDR, identical to the
  upstream gseapy local mode;
* the result mimics gseapy (``.res2d`` / ``.results`` DataFrames with the same
  columns) so :func:`omicverse.bulk.geneset_enrichment` and friends are
  unchanged.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


class EnrichrResult:
    """gseapy-compatible ORA result container.

    Attributes
    ----------
    res2d, results : pandas.DataFrame
        One row per enriched term. Columns: ``Gene_set, Term, Overlap,
        P-value, Adjusted P-value, Odds Ratio, Combined Score, Genes``. Both
        attributes hold the full table across every queried library (gseapy
        exposed ``res2d`` as the last library only; here they are identical and
        always the concatenation, which is what omicverse's callers expect).
    """

    def __init__(self, res2d: pd.DataFrame):
        self.res2d = res2d
        self.results = res2d


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (monotone, clipped to 1)."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    factor = np.arange(1, n + 1) / float(n)
    q = ranked / factor
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def _calc_term_stats(query: set, gene_sets: Dict[str, List[str]], background):
    """Hypergeometric over-representation for every term in ``gene_sets``.

    Mirrors gseapy's ``calc_pvalues``: for a 2×2 table with ``x`` overlap,
    gene-set size ``m``, query size ``k`` and background ``bg``,
    ``p = hypergeom.sf(x - 1, bg, m, k)`` and the Haldane-Anscombe-corrected
    odds ratio. Returns lists ``(terms, pvals, oddr, overlap, gs_size, hits)``.
    """
    if isinstance(background, set):
        bg = len(background)
        query = query.intersection(background)
    elif isinstance(background, (int, np.integer)):
        bg = int(background)
    else:
        raise ValueError("background should be a set or int")

    k = len(query)
    terms, pvals, oddr, olsz, gssz, genes = [], [], [], [], [], []
    for s in sorted(gene_sets.keys()):
        category = set(gene_sets[s])
        if isinstance(background, set):
            category = category.intersection(background)
        hits = query.intersection(category)
        x = len(hits)
        if x < 1:
            continue
        m = len(category)
        pval = hypergeom.sf(x - 1, bg, m, k)
        # Haldane-Anscombe correction (gseapy issue #132)
        ratio = ((x + 0.5) * (bg - m + 0.5)) / ((m + 0.5) * (k - x + 0.5))
        terms.append(s)
        pvals.append(pval)
        oddr.append(ratio)
        olsz.append(x)
        gssz.append(m)
        genes.append(sorted(hits))
    return terms, pvals, oddr, olsz, gssz, genes


def _resolve_library(name: str, organism: str) -> Dict[str, List[str]]:
    """Resolve a library name / .gmt / .txt path to a ``{term: [genes]}`` dict,
    downloading and caching Enrichr libraries offline as needed."""
    from ..utils._data import geneset_prepare
    return geneset_prepare(name, organism=organism)


def _resolve_background(background, gene_sets: Dict[str, List[str]]):
    """Turn the ``background`` argument into a set or int.

    ``None`` / unrecognised strings (e.g. a BioMart dataset name we cannot
    query offline) fall back to the union of all gene-set genes — the same
    default upstream gseapy adopted to avoid brittle Ensembl BioMart queries.
    """
    if background is None:
        bg = set()
        for genes in gene_sets.values():
            bg.update(genes)
        return bg
    if isinstance(background, (int, np.integer)):
        return int(background)
    if isinstance(background, str):
        if background.isdigit():
            return int(background)
        # A file of background genes?
        import os
        if os.path.isfile(background):
            with open(background) as fh:
                return set(g.strip() for g in fh if g.strip())
        # Otherwise (e.g. 'hsapiens_gene_ensembl') we cannot resolve it
        # offline — fall back to the gene-set union.
        bg = set()
        for genes in gene_sets.values():
            bg.update(genes)
        return bg
    # array-like
    try:
        return set(background)
    except TypeError:
        raise ValueError("Unsupported background data type: %r" % type(background))


def enrichr(
    gene_list,
    gene_sets,
    organism: str = 'human',
    description: str = '',
    background=None,
    outdir: Optional[str] = None,
    cutoff: float = 0.05,
    **_ignored,
) -> Optional[EnrichrResult]:
    """Local hypergeometric over-representation analysis (Enrichr-compatible).

    Parameters
    ----------
    gene_list
        Query gene symbols (a list/tuple/Series, or a single string).
    gene_sets
        A prepared ``{term: [genes]}`` dict, a ``.gmt``/``.txt`` path, an
        Enrichr library name, or a list of any of these.
    organism
        Used only when a library *name* must be resolved/normalised.
    background
        Background universe: ``None`` (gene-set union), an int, a list/set of
        genes, or a path to a gene file. BioMart dataset names are not queried
        (offline) — they fall back to the gene-set union.
    cutoff
        FDR ``alpha`` used for the BH correction reporting (kept for
        signature compatibility; all terms with ≥1 hit are returned).

    Returns
    -------
    EnrichrResult or None
        ``None`` if no term had any overlap with the query.
    """
    if isinstance(gene_list, str):
        genes = [gene_list]
    elif isinstance(gene_list, pd.Series):
        genes = gene_list.astype(str).tolist()
    else:
        genes = [str(g) for g in gene_list]
    query = set(g.strip() for g in genes if str(g).strip())

    # Normalise gene_sets into an ordered mapping {library_label: {term: genes}}
    libraries: "OrderedDict[str, Dict[str, List[str]]]" = OrderedDict()
    if isinstance(gene_sets, dict):
        libraries['gs_ind'] = gene_sets
    elif isinstance(gene_sets, str):
        for name in gene_sets.split(','):
            name = name.strip()
            if name:
                libraries[name] = _resolve_library(name, organism)
    elif isinstance(gene_sets, (list, tuple)):
        for i, g in enumerate(gene_sets):
            if isinstance(g, dict):
                libraries['gs_ind_%d' % i] = g
            else:
                libraries[str(g)] = _resolve_library(str(g), organism)
    else:
        raise ValueError("Unsupported gene_sets type: %r" % type(gene_sets))

    frames = []
    for label, gmt in libraries.items():
        bg = _resolve_background(background, gmt)
        terms, pvals, oddr, olsz, gssz, hits = _calc_term_stats(query, gmt, bg)
        if len(terms) == 0:
            continue
        fdrs = _bh_fdr(np.asarray(pvals))
        pvals = np.asarray(pvals)
        oddr = np.asarray(oddr)
        # Enrichr-style combined score (local analogue): -ln(p) * ln(OR),
        # positive and large for strongly enriched terms.
        with np.errstate(divide='ignore'):
            combined = -np.log(np.clip(pvals, 1e-300, None)) * np.log(np.clip(oddr, 1e-300, None))
        odict = OrderedDict()
        odict['Gene_set'] = label
        odict['Term'] = terms
        odict['Overlap'] = ['%d/%d' % (h, g) for h, g in zip(olsz, gssz)]
        odict['P-value'] = pvals
        odict['Adjusted P-value'] = fdrs
        odict['Odds Ratio'] = oddr
        odict['Combined Score'] = combined
        odict['Genes'] = [';'.join(g) for g in hits]
        frames.append(pd.DataFrame(odict))

    if not frames:
        return None
    res2d = pd.concat(frames, ignore_index=True)
    res2d = res2d.sort_values('P-value').reset_index(drop=True)

    if outdir is not None:
        import os
        os.makedirs(outdir, exist_ok=True)
        for label, sub in res2d.groupby('Gene_set'):
            out = os.path.join(outdir, '%s.%s.enrichr.reports.txt' % (label, organism))
            sub.to_csv(out, index=False, sep='\t', encoding='utf-8')

    return EnrichrResult(res2d)

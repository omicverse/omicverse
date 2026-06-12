r"""Metabolite-Set Enrichment Analysis (MSEA).

Two enrichment flavours, both keyed off KEGG-compound pathway membership:

1. **ORA** (over-representation analysis) — classic Fisher's-exact on
   a pre-selected hit list vs the background universe. Fast, no
   ranking needed.
2. **GSEA-style** — full ranked-list enrichment via omicverse's own
   single-process NumPy pre-ranked GSEA (``ov.bulk._gsea_numpy``).
   Uses the Welch-t statistic (or any user-supplied metric) as the
   ranking; the output schema matches the classic GSEA report.

Both use the local pathway table at
``omicverse/metabol/data/kegg_pathways.csv`` by default, and translate
metabolite names → KEGG compound IDs via :mod:`omicverse.metabol._id_mapping`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ._id_mapping import map_ids, normalize_name
from ._utils import bh_fdr as _bh_fdr

from .._registry import register_function


@register_function(
    aliases=[
        'load_pathways',
        'kegg_pathways',
    ],
    category='metabolomics',
    description='Fetch the full KEGG pathway→compound map (~550 pathways) via KEGG REST. Cached under ~/.cache/omicverse/metabol/.',
    examples=[
        "ov.metabol.load_pathways(organism='hsa')",
    ],
    related=[
        'metabol.fetch_kegg_pathways',
        'metabol.msea_ora',
    ],
)
def load_pathways(
    path: Optional[Path] = None,
    *,
    organism: Optional[str] = None,
) -> dict[str, list[str]]:
    """Return ``{pathway_name: [kegg_id, ...]}`` — the pathway database
    used by :func:`msea_ora` / :func:`msea_gsea` / :func:`mummichog_basic`.

    The default is the **full KEGG pathway database** fetched from the
    public KEGG REST endpoint (cached under
    ``~/.cache/omicverse/metabol/``; ~550 pathways). First call needs
    network; subsequent calls are free.

    Parameters
    ----------
    path
        Override with your own pathway CSV (columns: ``pathway_name``,
        ``kegg_compounds`` — the latter a ``;``-joined list of KEGG IDs).
        Useful for domain-specific pathway collections (e.g. SMPDB,
        Reactome-metabolite, or a curated clinical panel). Skips the
        network entirely.
    organism
        KEGG organism code for species-specific pathways (``"hsa"``
        human, ``"mmu"`` mouse, …). Default ``None`` → species-agnostic
        ``map#####`` reference metabolic pathways, which is what
        enrichment papers usually use.
    """
    if path is not None:
        df = pd.read_csv(path)
        return {
            row["pathway_name"]: row["kegg_compounds"].split(";")
            for _, row in df.iterrows()
        }
    from ._fetchers import fetch_kegg_pathways
    return fetch_kegg_pathways(organism=organism)


@register_function(
    aliases=[
        'msea_ora',
        'ora_metabolites',
        '代谢物过表达',
    ],
    category='metabolomics',
    description="Over-representation analysis (Fisher's exact) of differential metabolites against KEGG pathways.",
    examples=[
        'ov.metabol.msea_ora(hits, background)',
    ],
    related=[
        'metabol.msea_gsea',
        'metabol.pathway_dot',
    ],
)
def msea_ora(
    hits: Iterable[str],
    background: Iterable[str],
    *,
    pathways: Optional[dict[str, list[str]]] = None,
    min_size: int = 3,
    mass_db: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Over-representation analysis via Fisher's exact test.

    Parameters
    ----------
    hits
        Metabolite names (e.g. from ``pyMetabo.significant_metabolites()``).
    background
        All tested metabolite names (the universe). Usually
        ``adata.var_names`` after filtering.
    pathways
        Optional override of ``{pathway_name: [kegg_id, ...]}``. Default
        is the local KEGG subset shipped with omicverse.
    min_size
        Skip pathways with fewer than this many overlapping background
        compounds.
    mass_db
        Optional pre-fetched ChEBI DataFrame from
        :func:`fetch_chebi_compounds`. When supplied, ``map_ids`` uses
        it as an in-memory lookup for the ~54 k ChEBI names and only
        falls back to PubChem for names not resolved there. On a cold
        session this turns the ``map_ids`` cost from
        ``O(n_features)`` HTTP round-trips into a single dict probe
        per feature — often a 30–100x speedup on the first call.

    Returns
    -------
    pd.DataFrame
        Columns: ``pathway``, ``overlap``, ``set_size``, ``universe_size``,
        ``odds_ratio``, ``pvalue``, ``padj`` (BH).
    """
    if pathways is None:
        pathways = load_pathways()
    # Map names → KEGG IDs (forward mass_db so we avoid PubChem per-name).
    # Only request the kegg target since that's the only ID MSEA uses —
    # requesting extra targets triggers a PubChem fallback for every
    # name that has a kegg hit in mass_db but no hmdb/chebi hit.
    #
    # Perf: map the **union** (background ∪ hits) in a single call so
    # cached PubChem lookups and the ChEBI index are reused. A naïve
    # "map hits, then map background" doubles the cost for hits (they
    # sit inside background) and forces us to pay the same network
    # round-trips twice.
    hits_list = list(hits)
    bg_list = list(background)
    all_names = list(dict.fromkeys(bg_list + hits_list))  # preserve order
    id_map = map_ids(all_names, targets=("kegg",), mass_db=mass_db)
    name_to_kegg = id_map["kegg"].to_dict()

    hit_kegg = set(
        name_to_kegg.get(n, "") for n in hits_list
    ) - {""}
    bg_kegg = set(
        name_to_kegg.get(n, "") for n in bg_list
    ) - {""}
    if not hit_kegg:
        raise ValueError(
            "None of the hit metabolite names resolve to KEGG compound IDs — "
            "check spelling or extend metabolite_lookup.csv."
        )

    rows = []
    for pw_name, pw_ids in pathways.items():
        pw_set = set(pw_ids) & bg_kegg
        if len(pw_set) < min_size:
            continue
        # 2x2 contingency: in_hit & in_pw | in_hit & not_pw
        #                  not_hit & in_pw | not_hit & not_pw
        overlap = hit_kegg & pw_set
        a = len(overlap)
        b = len(hit_kegg - pw_set)
        c = len(pw_set - hit_kegg)
        d = len(bg_kegg - hit_kegg - pw_set)
        if a == 0:
            continue
        try:
            odds_ratio, pvalue = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
        except ValueError:
            continue
        rows.append({
            "pathway": pw_name,
            "overlap": a,
            "set_size": len(pw_set),
            "universe_size": len(bg_kegg),
            "odds_ratio": odds_ratio,
            "pvalue": pvalue,
            "hit_kegg": ";".join(sorted(overlap)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["padj"] = _bh_fdr(out["pvalue"].to_numpy())
    return out.sort_values("pvalue").reset_index(drop=True)


@register_function(
    aliases=[
        'msea_gsea',
        'gsea_metabolites',
        'MSEA',
    ],
    category='metabolomics',
    description='GSEA-style ranked enrichment of metabolites against KEGG pathways (NumPy pre-ranked GSEA backend).',
    examples=[
        "ov.metabol.msea_gsea(deg, stat_col='stat', n_perm=1000)",
    ],
    related=[
        'metabol.msea_ora',
        'metabol.pathway_dot',
    ],
)
def msea_gsea(
    deg: pd.DataFrame,
    *,
    stat_col: str = "stat",
    pathways: Optional[dict[str, list[str]]] = None,
    n_perm: int = 1000,
    min_size: int = 3,
    max_size: int = 500,
    seed: int = 0,
    mass_db: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """GSEA-style ranked enrichment via omicverse's NumPy pre-ranked GSEA.

    Parameters
    ----------
    deg
        Output DataFrame from :func:`differential`. Rows indexed by
        metabolite name; column ``stat_col`` provides the ranking metric.
    stat_col
        Which column of ``deg`` to rank on. Default ``"stat"`` (signed
        t-statistic); ``"log2fc"`` is another common choice.
    pathways
        Dict mapping pathway name to list of KEGG compound IDs.
    n_perm
        Permutation count for the empirical null. 1000 is fine for
        tutorials; bump to ≥10000 for publication.
    mass_db
        Optional pre-fetched ChEBI DataFrame from
        :func:`fetch_chebi_compounds` — same role as in
        :func:`msea_ora`. Recommended for cold-cache runs to avoid
        per-name PubChem REST round-trips.

    Returns
    -------
    pd.DataFrame
        Columns: ``Term``, ``NES``, ``NOM p-val``, ``FDR q-val``,
        ``ES``, ``Lead_genes`` (metabolites driving the enrichment).
    """
    # omicverse's own single-process NumPy pre-ranked GSEA (no gseapy / no
    # loky process pool that could dead-lock). Enrichment scores are identical
    # to gseapy's (same Subramanian 2005 algorithm).
    from ..bulk._gsea_numpy import prerank as _prerank

    if pathways is None:
        pathways = load_pathways()
    # Build rank: metabolite-name → score
    rank_df = deg[[stat_col]].copy()
    rank_df["name"] = rank_df.index
    rank_df = rank_df.reset_index(drop=True)
    # Resolve names → KEGG (forward mass_db so we avoid PubChem per-name).
    # Request only the kegg target — see msea_ora docstring for the
    # rationale.
    id_map = map_ids(rank_df["name"].tolist(), targets=("kegg",),
                     mass_db=mass_db)
    rank_df["kegg"] = id_map["kegg"].values
    rank_df = rank_df[rank_df["kegg"] != ""].drop_duplicates("kegg")
    if rank_df.empty:
        raise ValueError(
            "None of the differential-result metabolites resolve to KEGG IDs."
        )
    rnk = rank_df.set_index("kegg")[stat_col].sort_values(ascending=False)

    result = _prerank(
        rnk=rnk, gene_sets=pathways, min_size=min_size, max_size=max_size,
        permutation_num=n_perm, seed=seed,
    )
    if not hasattr(result, "res2d") or result.res2d.empty:
        return pd.DataFrame()
    # Map the NumPy backend's column names onto the gseapy-style schema this
    # function has always returned (Term, ES, NES, NOM p-val, FDR q-val,
    # Lead_genes) so downstream callers/plots are unchanged.
    out = result.res2d.reset_index().rename(columns={
        "es": "ES", "nes": "NES", "pval": "NOM p-val",
        "fdr": "FDR q-val", "lead_genes": "Lead_genes",
    })
    keep = ["Term", "ES", "NES", "NOM p-val", "FDR q-val", "Lead_genes"]
    out = out[[c for c in keep if c in out.columns]].reset_index(drop=True)
    return out



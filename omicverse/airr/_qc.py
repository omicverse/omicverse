"""Receptor-chain quality control for single-cell AIRR data.

``chain_qc`` classifies every cell by the configuration of its TCR / BCR
chains — a clean reimplementation of scirpy's ``tl.chain_qc``.  It is the
recommended first step after :func:`omicverse.airr.io.read_10x_vdj`: cells
with implausible chain configurations (multichains, orphan chains) can then
be filtered or flagged before clonotype definition.
"""
from __future__ import annotations

import pandas as pd

from .._registry import register_function


def _has(val) -> bool:
    """True if a chain slot is populated."""
    return val is not None and val == val and str(val) not in ("", "None", "nan")


#: Every value ``chain_pairing`` can take, ordered from the cleanest
#: configuration to the least interpretable. Matches scirpy's vocabulary.
CHAIN_PAIRING_CATEGORIES = (
    "single pair",
    "extra VJ",
    "extra VDJ",
    "two full chains",
    "orphan VJ",
    "orphan VDJ",
    "multichain",
    "ambiguous",
    "no IR",
)

#: Configurations that carry exactly one productive VJ/VDJ pair plus, at most,
#: one extra chain of either arm — the biologically documented dual-receptor
#: cells. Useful as ``chain_qc``'s starting point for a permissive filter.
PAIRED_CATEGORIES = ("single pair", "extra VJ", "extra VDJ", "two full chains")


def _is_true(val) -> bool:
    """Interpret a ``multi_chain``-style flag written by the readers."""
    if val is None or val != val:
        return False
    return str(val).strip().lower() in {"true", "t", "yes", "1", "1.0"}


def _chain_pairing(n_vj: int, n_vdj: int, *, multi: bool, ambiguous: bool) -> str:
    """Map a chain count to a ``chain_pairing`` label.

    Follows scirpy's ``tl.chain_qc``: an extra chain on top of a complete
    pair is reported as such (``extra VJ`` / ``extra VDJ`` /
    ``two full chains``) rather than lumped into ``multichain``. Only cells
    whose reader dropped chains — more than two in an arm — are
    ``multichain``.
    """
    if multi:
        return "multichain"
    if ambiguous:
        return "ambiguous"
    if n_vj == 0 and n_vdj == 0:
        return "no IR"
    if n_vdj == 0:
        return "orphan VJ"
    if n_vj == 0:
        return "orphan VDJ"
    if n_vj == 2 and n_vdj == 2:
        return "two full chains"
    if n_vj == 2:
        return "extra VJ"
    if n_vdj == 2:
        return "extra VDJ"
    return "single pair"


def _receptor_subtype(loci: set[str]) -> str:
    """Map the set of loci present in a cell to a receptor subtype."""
    if not loci:
        return "no IR"
    if loci <= {"TRA", "TRB"}:
        return "TRA+TRB"
    if loci <= {"TRG", "TRD"}:
        return "TRG+TRD"
    if loci <= {"IGH", "IGK"}:
        return "IGH+IGK"
    if loci <= {"IGH", "IGL"}:
        return "IGH+IGL"
    if loci <= {"IGH", "IGK", "IGL"}:
        return "IGH+IGK/L"
    return "ambiguous"


def _classify_cell(row: pd.Series) -> tuple[str, str]:
    """Return ``(chain_pairing, receptor_subtype)`` for one cell."""
    n_vj = int(_has(row.get("VJ_1_junction_aa"))) + int(_has(row.get("VJ_2_junction_aa")))
    n_vdj = int(_has(row.get("VDJ_1_junction_aa"))) + int(_has(row.get("VDJ_2_junction_aa")))

    loci = set()
    for slot in ("VJ_1", "VJ_2", "VDJ_1", "VDJ_2"):
        loc = row.get(f"{slot}_locus")
        if _has(loc):
            loci.add(str(loc).upper())
    subtype = _receptor_subtype(loci)

    pairing = _chain_pairing(
        n_vj,
        n_vdj,
        multi=_is_true(row.get("multi_chain")),
        ambiguous=subtype == "ambiguous",
    )
    return pairing, subtype


@register_function(
    aliases=["chain_qc", "airr_chain_qc", "免疫链质控", "链配对质控"],
    category="airr",
    description=(
        "Classify single-cell AIRR cells by their TCR/BCR chain "
        "configuration: 'single pair', 'extra VJ', 'extra VDJ', "
        "'two full chains', 'orphan VJ', 'orphan VDJ', 'multichain', "
        "'ambiguous' or 'no IR'. Writes obs['chain_pairing'] and "
        "obs['receptor_subtype']."
    ),
    requires={"obs": ["VJ_1_junction_aa", "VDJ_1_junction_aa"]},
    produces={"obs": ["chain_pairing", "receptor_subtype"]},
    examples=[
        "ov.airr.chain_qc(adata)",
        "adata = adata[adata.obs['chain_pairing'] == 'single pair']",
        "# keep dual-TCR cells too (2 TRA + 1 TRB, 1 TRA + 2 TRB, 2 + 2)",
        "adata = ov.airr.filter_chains(adata, keep='paired')",
    ],
    related=["airr.read_10x_vdj", "airr.filter_chains", "airr.define_clonotypes"],
)
def chain_qc(adata, *, inplace: bool = True):
    """Classify cells by their immune-receptor chain configuration.

    The categories follow scirpy's ``tl.chain_qc``. In particular a cell that
    carries a complete VJ/VDJ pair *plus* one extra chain is reported as
    ``'extra VJ'`` / ``'extra VDJ'`` / ``'two full chains'``, not as
    ``'multichain'`` — dual-TCRα expression is a documented consequence of
    incomplete allelic exclusion at the *TRA* locus, so those cells are a
    biological population rather than a technical artefact and it is the
    user's call whether to keep them. ``'multichain'`` is reserved for cells
    where the reader had to drop chains (more than two in one arm), which is
    the configuration that really does suggest a doublet.

    Parameters
    ----------
    adata
        AnnData produced by :func:`omicverse.airr.read_10x_vdj` /
        :func:`~omicverse.airr.read_airr`.
    inplace
        If ``True`` (default) annotate ``adata.obs`` in place; otherwise
        return the classification :class:`pandas.DataFrame`.

    Returns
    -------
    AnnData or :class:`pandas.DataFrame`
        ``adata.obs`` gains:

        ``chain_pairing``
            One of :data:`CHAIN_PAIRING_CATEGORIES`:

            ``'single pair'``
                one VJ + one VDJ chain (e.g. 1 TRA + 1 TRB).
            ``'extra VJ'``
                a complete pair plus a second VJ chain (2 TRA + 1 TRB).
            ``'extra VDJ'``
                a complete pair plus a second VDJ chain (1 TRA + 2 TRB).
            ``'two full chains'``
                two VJ and two VDJ chains (2 TRA + 2 TRB).
            ``'orphan VJ'`` / ``'orphan VDJ'``
                only one arm of the receptor was recovered.
            ``'multichain'``
                more than two chains in one arm — chains were dropped at
                read time; usually a doublet.
            ``'ambiguous'``
                loci that do not form a known receptor (e.g. TRA + IGH).
            ``'no IR'``
                no receptor chain at all.
        ``receptor_subtype``
            e.g. ``'TRA+TRB'``, ``'IGH+IGK'``.
        ``receptor_type``
            ``'TCR'`` / ``'BCR'`` / ``'ambiguous'`` / ``'no IR'``
            (kept from the reader if already present).

    See Also
    --------
    filter_chains : subset an AnnData by the categories computed here.
    """
    res = adata.obs.apply(_classify_cell, axis=1, result_type="expand")
    res.columns = ["chain_pairing", "receptor_subtype"]
    if not inplace:
        return res
    adata.obs["chain_pairing"] = pd.Categorical(
        res["chain_pairing"].values, categories=CHAIN_PAIRING_CATEGORIES
    )
    adata.obs["receptor_subtype"] = res["receptor_subtype"].values
    return adata


@register_function(
    aliases=["filter_chains", "airr_filter_chains", "链配对筛选", "免疫链筛选"],
    category="airr",
    description=(
        "Subset an AIRR AnnData by obs['chain_pairing']. Accepts an explicit "
        "list of categories or the presets 'strict' (single pair only), "
        "'paired' (single pair + dual-receptor cells) and 'any IR'."
    ),
    requires={"obs": ["chain_pairing"]},
    examples=[
        "ov.airr.filter_chains(adata, keep='strict')",
        "ov.airr.filter_chains(adata, keep='paired')",
        "ov.airr.filter_chains(adata, keep=['single pair', 'extra VJ'])",
    ],
    related=["airr.chain_qc", "airr.define_clonotypes"],
)
def filter_chains(adata, *, keep="strict", copy: bool = True):
    """Subset cells by the ``chain_pairing`` label from :func:`chain_qc`.

    Parameters
    ----------
    adata
        AnnData already annotated by :func:`chain_qc`.
    keep
        Either an explicit iterable of categories, or one of the presets:

        ``'strict'``
            ``'single pair'`` only — the conservative default that most
            clonotype workflows assume.
        ``'paired'``
            adds ``'extra VJ'``, ``'extra VDJ'`` and ``'two full chains'``,
            i.e. keeps dual-receptor cells. Use this when dual-TCRα cells
            are part of the biology you are studying.
        ``'any IR'``
            everything except ``'no IR'`` — including orphan chains,
            multichains and ambiguous cells.
    copy
        Return a new object (default). ``False`` returns a view.

    Returns
    -------
    AnnData
        The subset. Cells are kept in their original order.
    """
    if "chain_pairing" not in adata.obs:
        raise KeyError(
            "adata.obs['chain_pairing'] not found — run ov.airr.chain_qc(adata) first."
        )

    presets = {
        "strict": ("single pair",),
        "paired": PAIRED_CATEGORIES,
        "any IR": tuple(c for c in CHAIN_PAIRING_CATEGORIES if c != "no IR"),
    }
    if isinstance(keep, str):
        try:
            categories = presets[keep]
        except KeyError:
            raise ValueError(
                f"keep must be one of {sorted(presets)} or an iterable of "
                f"categories from {list(CHAIN_PAIRING_CATEGORIES)}, got {keep!r}"
            ) from None
    else:
        categories = tuple(keep)
        unknown = set(categories) - set(CHAIN_PAIRING_CATEGORIES)
        if unknown:
            raise ValueError(
                f"unknown chain_pairing categories {sorted(unknown)}; expected "
                f"a subset of {list(CHAIN_PAIRING_CATEGORIES)}"
            )

    mask = adata.obs["chain_pairing"].astype(str).isin(categories).values
    sub = adata[mask]
    return sub.copy() if copy else sub

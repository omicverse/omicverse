r"""The A→B handoff: given a reaction, which enzyme should catalyse it?

``pathway_search`` and ``retro_biosynthesis`` find a route from substrate to
product. Layer B can then design, score and check an enzyme. Between the two
there was nothing: the route came out as a list of chemical transformations, and
somebody had to go and decide, by hand, which enzyme performs each step.

:func:`match_enzymes` closes that gap. Given a reaction — as an EC number, a
reaction SMARTS, or a set of metabolites — it returns ranked candidate enzymes
with sequences, so the next call can be
:func:`~omicverse.synbio.enzyme_kcat` or
:func:`~omicverse.synbio.predict_expression_level` rather than a literature
search.

Sources, in the order they are tried:

* a **local sequence database** you supply (``database=``) — a FASTA whose
  headers carry EC numbers, which is what a curated in-house enzyme collection
  looks like;
* **UniProt** via its REST API when ``allow_network=True`` — the same place
  Selenzyme draws from, queried for the EC number directly;
* the **model's own genes**, when a :class:`cobra.Model` is passed, so a
  reaction already present in your organism resolves to the gene you have.

Nothing is bundled. An enzyme database that ships inside an analysis package is
stale the day it is written, and the honest interface is one that tells you
which source an answer came from — :attr:`EnzymeCandidate.source` does.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover
    import cobra
    import pandas as pd


_EC_RE = re.compile(r"\b(\d+\.\d+\.\d+\.[\dn-]+)\b")


@dataclass
class EnzymeCandidate:
    """One enzyme proposed for one reaction."""

    identifier: str
    sequence: str = ""
    ec_number: str = ""
    organism: str = ""
    source: str = ""
    score: float = 0.0
    description: str = ""

    @property
    def has_sequence(self) -> bool:
        return bool(self.sequence)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"EnzymeCandidate({self.identifier!r}, EC={self.ec_number!r}, "
                f"{len(self.sequence)} aa, source={self.source!r}, "
                f"score={self.score:.2f})")


@dataclass
class EnzymeMatch:
    """Ranked enzyme candidates for one reaction."""

    reaction: str
    ec_number: str = ""
    candidates: List[EnzymeCandidate] = field(default_factory=list)
    sources_tried: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def best(self) -> Optional[EnzymeCandidate]:
        return self.candidates[0] if self.candidates else None

    @property
    def n_with_sequence(self) -> int:
        return sum(1 for c in self.candidates if c.has_sequence)

    def to_frame(self) -> "pd.DataFrame":
        import pandas as pd
        if not self.candidates:
            return pd.DataFrame(columns=["identifier", "ec_number", "organism",
                                         "length", "source", "score"])
        return pd.DataFrame([{
            "identifier": c.identifier, "ec_number": c.ec_number,
            "organism": c.organism, "length": len(c.sequence),
            "source": c.source, "score": c.score,
        } for c in self.candidates]).set_index("identifier")

    def __repr__(self) -> str:  # pragma: no cover
        return (f"EnzymeMatch({self.reaction!r}, EC={self.ec_number!r}, "
                f"{len(self.candidates)} candidates, "
                f"{self.n_with_sequence} with sequences)")


def _parse_fasta_with_ec(path: str) -> List[Tuple[str, str, str, str]]:
    """``[(id, ec, organism, sequence)]`` from a FASTA with EC in the header."""
    out: List[Tuple[str, str, str, str]] = []
    ident = ec = org = ""
    chunks: List[str] = []

    def flush():
        if ident:
            out.append((ident, ec, org, "".join(chunks)))

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
                ident = header.split()[0] if header else ""
                m = _EC_RE.search(header)
                ec = m.group(1) if m else ""
                om = re.search(r"OS=([^=]+?)(?:\s+\w\w=|$)", header)
                org = om.group(1).strip() if om else ""
                chunks = []
            else:
                chunks.append(line.strip())
    flush()
    return out


def _ec_similarity(a: str, b: str) -> float:
    """How many of the four EC levels agree, as a fraction.

    Partial matches matter: EC 1.1.1.1 and 1.1.1.2 act on the same chemistry
    with different specificity, and for a designed pathway the near-miss is
    often the better starting point than nothing.
    """
    if not a or not b:
        return 0.0
    pa, pb = a.split("."), b.split(".")
    n = 0
    for x, y in zip(pa, pb):
        if x == y:
            n += 1
        else:
            break
    return n / 4.0


@register_function(
    aliases=["match_enzymes", "反应找酶", "Selenzyme", "selenzyme",
             "酶匹配", "reaction_to_enzyme", "通路选酶"],
    category="synthetic_biology",
    description="给一个反应找催化它的候选酶(A→B 的交接口)。pathway_search / retro_biosynthesis 只给出化学转化,每一步该用哪个酶原本要人工查文献。支持三个来源:本地 FASTA 库(表头带 EC 号)、UniProt REST(allow_network=True)、以及传入 cobra 模型时用模型自身的基因。本模块不内置酶库 —— 打包进分析软件的酶库写下即过时;结果的 source 字段标明每条候选来自哪里。Match candidate enzymes to a reaction.",
    examples=[
        "m = ov.synbio.match_enzymes('1.1.1.1', database='/db/enzymes.faa')",
        "m = ov.synbio.match_enzymes('2.7.1.11', model=gem)",
        "m.best.sequence  # -> ov.synbio.enzyme_kcat(...)",
    ],
    related=["synbio.pathway_search", "synbio.retro_biosynthesis",
             "synbio.enzyme_kcat", "synbio.enzyme_function",
             "synbio.predict_expression_level"],
    requires={},
    produces={},
)
def match_enzymes(
    reaction: str,
    *,
    database: Optional[str] = None,
    model: "Optional[cobra.Model]" = None,
    allow_network: bool = False,
    organism: Optional[str] = None,
    max_candidates: int = 10,
    min_ec_levels: int = 3,
    timeout: float = 20.0,
) -> EnzymeMatch:
    """Find enzymes that can catalyse ``reaction``.

    Parameters
    ----------
    reaction
        An EC number (``'1.1.1.1'``), or any string containing one — a reaction
        description, a KEGG entry, a model reaction annotation.
    database
        FASTA of candidate enzymes whose headers carry EC numbers. Defaults to
        ``OMICOS_ENZYME_DB``.
    model
        A metabolic model. When the reaction id exists in it, its genes are
        offered as candidates — an organism that already runs the step is the
        obvious first answer.
    allow_network
        Query UniProt for the EC number. Off by default: a library function
        should not reach the network unless asked.
    organism
        Restrict UniProt results, e.g. ``'Escherichia coli'``.
    max_candidates
        Cap on returned candidates.
    min_ec_levels
        Minimum EC levels that must agree for a local-database hit to count.
        3 means "same chemistry, possibly different substrate specificity".

    Returns
    -------
    EnzymeMatch
    """
    ec_match = _EC_RE.search(str(reaction))
    ec = ec_match.group(1) if ec_match else ""

    tried: List[str] = []
    notes: List[str] = []
    candidates: List[EnzymeCandidate] = []

    # --- the model's own genes -------------------------------------------
    if model is not None:
        tried.append("model")
        try:
            rxn = model.reactions.get_by_id(str(reaction))
        except Exception:
            rxn = None
        if rxn is not None:
            for gene in rxn.genes:
                candidates.append(EnzymeCandidate(
                    identifier=gene.id, ec_number=ec,
                    organism=getattr(model, "id", ""), source="model",
                    score=1.0,
                    description=f"gene of {rxn.id} in the supplied model"))
            if not rxn.genes:
                notes.append(
                    f"{rxn.id} 在模型里没有基因关联(自发反应或注释缺失)。")
        elif ec:
            notes.append(f"{reaction!r} 不是模型里的反应 id,已按 EC {ec} 继续检索。")

    # --- local database ---------------------------------------------------
    db = database or os.environ.get("OMICOS_ENZYME_DB", "")
    if db:
        db = os.path.expanduser(db)
        if not os.path.exists(db):
            raise FileNotFoundError(f"酶库文件不存在:{db}")
        tried.append(os.path.basename(db))
        if not ec:
            notes.append(
                "没有从 reaction 里解析出 EC 号,本地库按 EC 匹配无法进行。")
        else:
            for ident, hit_ec, org, seq in _parse_fasta_with_ec(db):
                sim = _ec_similarity(ec, hit_ec)
                if sim * 4 < min_ec_levels:
                    continue
                if organism and organism.lower() not in org.lower():
                    continue
                candidates.append(EnzymeCandidate(
                    identifier=ident, sequence=seq, ec_number=hit_ec,
                    organism=org, source=os.path.basename(db), score=sim,
                    description=f"EC match at {int(sim * 4)}/4 levels"))

    # --- UniProt ----------------------------------------------------------
    if allow_network and ec:
        tried.append("uniprot")
        try:
            candidates.extend(_uniprot_by_ec(ec, organism, max_candidates,
                                             timeout))
        except Exception as exc:
            notes.append(f"UniProt 查询失败({type(exc).__name__}: {exc})。")

    if not candidates and not tried:
        raise ValueError(
            "没有任何可用的酶来源。请给 database=(表头含 EC 号的 FASTA)、"
            "model=(cobra 模型,用它自己的基因),或 allow_network=True 查 UniProt。"
            "本模块不内置酶库 —— 内置的库写下即过时,且无法反映你有权使用的序列集合。")

    candidates.sort(key=lambda c: (-c.score, -len(c.sequence), c.identifier))
    return EnzymeMatch(reaction=str(reaction), ec_number=ec,
                       candidates=candidates[:max_candidates],
                       sources_tried=tried, notes=notes)


def _uniprot_by_ec(ec: str, organism: Optional[str], limit: int,
                   timeout: float) -> List[EnzymeCandidate]:
    """Query UniProtKB's REST API for reviewed entries with this EC number."""
    import json
    import urllib.parse
    import urllib.request

    query = f"(ec:{ec}) AND (reviewed:true)"
    if organism:
        query += f' AND (organism_name:"{organism}")'
    url = ("https://rest.uniprot.org/uniprotkb/search?"
           + urllib.parse.urlencode({
               "query": query, "format": "json", "size": str(min(limit, 25)),
               "fields": "accession,protein_name,organism_name,ec,sequence",
           }))
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        payload = json.load(fh)

    out: List[EnzymeCandidate] = []
    for entry in payload.get("results", []):
        acc = entry.get("primaryAccession", "")
        seq = (entry.get("sequence") or {}).get("value", "")
        org = ((entry.get("organism") or {}).get("scientificName", ""))
        names = (entry.get("proteinDescription") or {}).get("recommendedName", {})
        desc = (names.get("fullName") or {}).get("value", "")
        out.append(EnzymeCandidate(
            identifier=acc, sequence=seq, ec_number=ec, organism=org,
            source="uniprot", score=1.0, description=desc))
    return out


@register_function(
    aliases=["match_pathway_enzymes", "通路选酶", "pathway_enzymes",
             "为通路配酶"],
    category="synthetic_biology",
    description="为一条通路的每一步找酶,并汇总哪些步骤有候选、哪些是断点 —— 断点才是真正的工程风险所在。Match enzymes to every step of a pathway and report which steps have no candidate.",
    examples=[
        "res = ov.synbio.match_pathway_enzymes(['1.1.1.1','2.7.1.11'], database=db)",
        "res['gaps']",
    ],
    related=["synbio.match_enzymes", "synbio.pathway_search",
             "synbio.retro_biosynthesis"],
    requires={},
    produces={},
)
def match_pathway_enzymes(steps: Sequence[str], **kwargs) -> Dict[str, object]:
    """Run :func:`match_enzymes` over a pathway and summarise the coverage."""
    matches = {}
    gaps = []
    for step in steps:
        m = match_enzymes(step, **kwargs)
        matches[str(step)] = m
        if not m.candidates:
            gaps.append(str(step))
    return {
        "matches": matches,
        "gaps": gaps,
        "coverage": (len(steps) - len(gaps)) / len(steps) if steps else 0.0,
    }


@register_function(
    aliases=["plot_enzyme_match", "选酶图", "酶候选图", "plot_selenzyme"],
    category="synthetic_biology",
    description="画通路选酶的覆盖情况:每一步的候选数与最佳 EC 匹配层级,没有候选的步骤标红 —— 那些是工程断点。Plot enzyme-match coverage across a pathway, highlighting the gaps.",
    examples=["ov.synbio.plot_enzyme_match(res)"],
    related=["synbio.match_pathway_enzymes", "synbio.match_enzymes"],
    requires={},
    produces={},
)
def plot_enzyme_match(result, ax=None):
    """Bar per pathway step: how many candidates, and whether it is a gap."""
    from ._plot import _mpl
    plt = _mpl()

    matches = result["matches"] if isinstance(result, dict) else result
    steps = list(matches)
    counts = [len(matches[s].candidates) for s in steps]
    colours = ["#E41A1C" if c == 0 else "#4DAF4A" for c in counts]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4.0, 0.8 * len(steps) + 2), 3.4))
    else:
        fig = ax.figure
    ax.bar(range(len(steps)), counts, color=colours)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("candidate enzymes")
    n_gaps = sum(1 for c in counts if c == 0)
    ax.set_title(f"{len(steps) - n_gaps}/{len(steps)} steps have a candidate"
                 + (f" — {n_gaps} gaps (red)" if n_gaps else ""))
    fig.tight_layout()
    return fig, ax


__all__ = ["match_enzymes", "EnzymeMatch", "EnzymeCandidate",
           "match_pathway_enzymes", "plot_enzyme_match"]

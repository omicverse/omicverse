r"""Part registries, and picking a backbone to put the design in.

``read_sbol`` / ``write_sbol`` could exchange designs but could not *find* one.
Every construct starts from parts somebody already characterised, and the
registries where they live — the iGEM Registry, SynBioHub, Addgene — were
reachable only through a browser.

* :func:`query_parts` — search a registry by keyword, part type or identifier.
* :func:`fetch_part` — retrieve one part with its sequence.
* :func:`select_backbone` — choose a vector from the requirements that actually
  decide it: host, copy number, selection marker, insert size, and whether the
  design needs to be compatible with an assembly standard.
* :func:`plot_backbone_choice` — the shortlist and why each was ranked.

Network access is **opt-in** (``allow_network=True``). A library function that
silently reaches the internet is a problem in a batch job and a worse one behind
an institutional proxy, so the default is a small built-in catalogue of the
standard backbones — pUC, pET, pACYC, pBBR1, pSB1C3 and friends — whose defining
properties (origin, copy number, marker, host range) are stable published facts
rather than a snapshot of a database.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from .._registry import register_function

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


REGISTRIES = ("igem", "synbiohub", "addgene")


@dataclass
class Part:
    """A registry part."""

    identifier: str
    name: str = ""
    part_type: str = ""
    description: str = ""
    sequence: str = ""
    registry: str = ""
    url: str = ""

    @property
    def length(self) -> int:
        return len(self.sequence)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Part({self.identifier!r}, {self.part_type or '?'}, "
                f"{self.length} bp, {self.registry})")


@dataclass
class Backbone:
    """A cloning/expression vector backbone."""

    name: str
    origin: str
    copy_number: str            # 'high' | 'medium' | 'low' | 'single'
    marker: str
    host: str
    promoter: str = ""
    max_insert_kb: float = 10.0
    standard: str = ""          # assembly standard compatibility
    notes: str = ""
    size_kb: float = 0.0

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Backbone({self.name}, {self.origin}/{self.copy_number}-copy, "
                f"{self.marker}, {self.host})")


# Defining properties of standard backbones. These are stable published facts —
# an origin's copy number does not change — which is why they can be bundled
# where a parts catalogue could not.
BACKBONE_CATALOGUE: List[Backbone] = [
    Backbone("pUC19", "pMB1(rop-)", "high", "ampicillin", "e_coli",
             promoter="lac", max_insert_kb=5.0, size_kb=2.7,
             notes="cloning workhorse; high copy for prep yield, not expression"),
    Backbone("pET28a", "pBR322", "medium", "kanamycin", "e_coli",
             promoter="T7", max_insert_kb=6.0, size_kb=5.4,
             notes="T7 expression; needs a DE3 lysogen host"),
    Backbone("pET22b", "pBR322", "medium", "ampicillin", "e_coli",
             promoter="T7", max_insert_kb=6.0, size_kb=5.5,
             notes="pelB leader for periplasmic secretion"),
    Backbone("pACYC184", "p15A", "low", "chloramphenicol", "e_coli",
             max_insert_kb=8.0, size_kb=4.2,
             notes="p15A is compatible with pMB1/pBR322 — use for co-transformation"),
    Backbone("pCDF-1b", "CloDF13", "low", "streptomycin", "e_coli",
             promoter="T7", max_insert_kb=6.0, size_kb=3.8,
             notes="fourth compatible origin for multi-plasmid systems"),
    Backbone("pBBR1MCS-2", "pBBR1", "medium", "kanamycin", "broad",
             max_insert_kb=8.0, size_kb=5.1,
             notes="broad host range; works outside E. coli"),
    Backbone("pSB1C3", "pMB1", "high", "chloramphenicol", "e_coli",
             max_insert_kb=5.0, standard="BioBrick RFC10", size_kb=2.1,
             notes="iGEM standard assembly backbone"),
    Backbone("pYES2", "2micron", "high", "URA3", "s_cerevisiae",
             promoter="GAL1", max_insert_kb=6.0, size_kb=5.9,
             notes="galactose-inducible yeast expression"),
    Backbone("pPICZ-A", "pUC(E. coli)", "single", "zeocin", "p_pastoris",
             promoter="AOX1", max_insert_kb=5.0, size_kb=3.3,
             notes="integrates into the Pichia genome; methanol induction"),
    Backbone("pHT01", "pUB110", "medium", "chloramphenicol", "b_subtilis",
             promoter="Pgrac", max_insert_kb=6.0, size_kb=8.0,
             notes="B. subtilis expression, IPTG inducible"),
    Backbone("pTwist-Kan-Ori", "ColE1", "high", "kanamycin", "e_coli",
             max_insert_kb=10.0, standard="Golden Gate (BsaI-free)",
             size_kb=2.6, notes="domesticated for Type IIS assembly"),
]

_COPY_RANK = {"single": 0, "low": 1, "medium": 2, "high": 3}


@register_function(
    aliases=["query_parts", "部件库查询", "iGEM", "SynBioHub", "Addgene",
             "parts_registry", "查部件", "registry_search"],
    category="synthetic_biology",
    description="远程查询标准生物部件库(iGEM Registry / SynBioHub / Addgene)。read_sbol/write_sbol 只能交换设计,找不到设计。需要 allow_network=True 显式开网 —— 库函数不应该默默联网。Search a parts registry by keyword or identifier.",
    examples=[
        "hits = ov.synbio.query_parts('T7 promoter', registry='synbiohub', allow_network=True)",
        "part = ov.synbio.fetch_part('BBa_J23100', allow_network=True)",
    ],
    related=["synbio.fetch_part", "synbio.select_backbone", "synbio.read_sbol"],
    requires={},
    produces={},
)
def query_parts(
    query: str,
    registry: str = "synbiohub",
    *,
    allow_network: bool = False,
    part_type: Optional[str] = None,
    limit: int = 20,
    timeout: float = 20.0,
) -> List[Part]:
    """Search ``registry`` for parts matching ``query``.

    Parameters
    ----------
    query
        Free text, or a part identifier such as ``'BBa_J23100'``.
    registry
        ``'synbiohub'`` | ``'igem'`` | ``'addgene'``.
    allow_network
        Must be True. Off by default so that importing and calling this in a
        batch job cannot block on a network round-trip.
    part_type
        Filter by SBOL role, e.g. ``'promoter'``, ``'cds'``, ``'terminator'``.
    limit
        Maximum results.
    """
    if registry not in REGISTRIES:
        raise ValueError(f"registry must be one of {list(REGISTRIES)}, got {registry!r}")
    if not allow_network:
        raise ValueError(
            "query_parts 需要联网访问部件库,请显式传 allow_network=True。"
            "默认关闭是因为库函数不该在批处理作业里悄悄发起网络请求"
            "(在机构代理后面尤其容易卡住)。载体骨架选择 select_backbone "
            "有内置目录,不需要联网。")

    if registry == "synbiohub":
        return _synbiohub_search(query, part_type, limit, timeout)
    if registry == "igem":
        return _igem_search(query, limit, timeout)
    return _addgene_search(query, limit, timeout)


def _synbiohub_search(query, part_type, limit, timeout) -> List[Part]:
    base = "https://synbiohub.org/search/"
    term = urllib.parse.quote(str(query))
    role = f"role={urllib.parse.quote(part_type)}&" if part_type else ""
    url = f"{base}{role}{term}/?offset=0&limit={int(limit)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        payload = json.load(fh)
    out = []
    for item in payload[:limit]:
        out.append(Part(
            identifier=str(item.get("displayId", "")),
            name=str(item.get("name", "")),
            description=str(item.get("description", "")),
            part_type=str(item.get("type", "")),
            registry="synbiohub", url=str(item.get("uri", "")),
        ))
    return out


def _igem_search(query, limit, timeout) -> List[Part]:
    """iGEM parts are mirrored in SynBioHub's igem collection."""
    parts = _synbiohub_search(query, None, limit, timeout)
    for p in parts:
        p.registry = "igem"
    return [p for p in parts if p.identifier.startswith("BBa_")] or parts


def _addgene_search(query, limit, timeout) -> List[Part]:
    raise NotImplementedError(
        "Addgene 没有公开的检索 API(只有网页界面与逐条的 plasmid 页面)。"
        "请用 registry='synbiohub',或从 Addgene 导出 GenBank 后用 "
        "ov.synbio.read_genbank 读入。")


@register_function(
    aliases=["fetch_part", "取部件", "get_part", "下载部件"],
    category="synthetic_biology",
    description="按 id 取回一个部件及其序列(SynBioHub / iGEM)。需要 allow_network=True。Fetch one part with its sequence by identifier.",
    examples=["p = ov.synbio.fetch_part('BBa_J23100', allow_network=True)"],
    related=["synbio.query_parts", "synbio.annotate_construct"],
    requires={},
    produces={},
)
def fetch_part(identifier: str, *, registry: str = "synbiohub",
               allow_network: bool = False, timeout: float = 20.0) -> Part:
    """Retrieve a single part, sequence included."""
    if not allow_network:
        raise ValueError(
            "fetch_part 需要联网,请显式传 allow_network=True。")
    if registry not in ("synbiohub", "igem"):
        raise ValueError("fetch_part 目前支持 registry='synbiohub' 或 'igem'。")

    ident = str(identifier)
    url = (f"https://synbiohub.org/public/igem/{ident}/1/sbol"
           if ident.startswith("BBa_") else
           f"https://synbiohub.org/public/igem/{ident}/sbol")
    req = urllib.request.Request(url, headers={"Accept": "application/xml"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        text = fh.read().decode("utf-8", errors="replace")

    import re
    seq = ""
    m = re.search(r"<sbol:elements[^>]*>([^<]+)</sbol:elements>", text)
    if m:
        seq = "".join(m.group(1).split()).upper()
    name = ""
    m = re.search(r"<dcterms:title[^>]*>([^<]+)</dcterms:title>", text)
    if m:
        name = m.group(1).strip()
    return Part(identifier=ident, name=name, sequence=seq, registry=registry,
                url=url)


# ---------------------------------------------------------------------------
# backbone selection
# ---------------------------------------------------------------------------

@dataclass
class BackboneChoice:
    backbone: Backbone
    score: float
    reasons: List[str] = field(default_factory=list)
    disqualified: bool = False

    def __repr__(self) -> str:  # pragma: no cover
        tag = "REJECTED" if self.disqualified else f"{self.score:.2f}"
        return f"BackboneChoice({self.backbone.name}, {tag})"


@register_function(
    aliases=["select_backbone", "载体选择", "骨架选择", "vector_selection",
             "选载体", "plasmid_backbone"],
    category="synthetic_biology",
    description="按真正决定选择的条件挑载体骨架:宿主、拷贝数、抗性标记、插入片段大小、是否要与共存质粒的复制起点相容、是否需要组装标准兼容。内置标准骨架目录(pUC/pET/pACYC/pBBR1/pSB1C3 等)—— 复制起点的拷贝数是稳定的已发表事实,可以内置,而部件库快照不行。无需联网。Rank vector backbones against the requirements that actually decide the choice.",
    examples=[
        "picks = ov.synbio.select_backbone(host='e_coli', insert_kb=2.5, expression=True)",
        "picks[0].backbone.name, picks[0].reasons",
        "picks = ov.synbio.select_backbone(host='e_coli', incompatible_with='pBR322')",
    ],
    related=["synbio.query_parts", "synbio.golden_gate", "synbio.write_sbol",
             "synbio.plot_backbone_choice"],
    requires={},
    produces={},
)
def select_backbone(
    *,
    host: str = "e_coli",
    insert_kb: float = 1.0,
    expression: bool = False,
    secretion: bool = False,
    copy_number: Optional[str] = None,
    marker: Optional[str] = None,
    avoid_markers: Optional[Sequence[str]] = None,
    incompatible_with: Optional[str] = None,
    standard: Optional[str] = None,
    catalogue: Optional[Sequence[Backbone]] = None,
) -> List[BackboneChoice]:
    """Rank backbones for a design.

    Parameters
    ----------
    host
        ``'e_coli'``, ``'b_subtilis'``, ``'s_cerevisiae'``, ``'p_pastoris'``, or
        ``'broad'``. A backbone whose host does not match is disqualified, not
        merely down-ranked — it will not replicate.
    insert_kb
        Size of what is going in. Backbones whose capacity is exceeded are
        disqualified.
    expression
        Require an expression promoter rather than a cloning vector.
    secretion
        Require a secretion leader — pairs with
        :func:`~omicverse.synbio.predict_signal_peptide`.
    copy_number
        ``'high'`` for yield, ``'low'`` for a toxic product or a stable circuit.
    marker, avoid_markers
        Require or exclude selection markers.
    incompatible_with
        An origin already in the cell. Two plasmids sharing a replication origin
        cannot be stably co-maintained, so anything with this origin is
        disqualified — the most common reason a two-plasmid system quietly loses
        one of them.
    standard
        Assembly-standard compatibility, e.g. ``'BioBrick'`` or ``'Golden Gate'``.

    Returns
    -------
    list of BackboneChoice
        Best first; disqualified entries are kept, with the reason, so a
        rejected option is visible rather than absent.
    """
    cat = list(catalogue if catalogue is not None else BACKBONE_CATALOGUE)
    avoid = {m.lower() for m in (avoid_markers or [])}
    out: List[BackboneChoice] = []

    for bb in cat:
        reasons: List[str] = []
        disq = False
        score = 0.0

        if bb.host != host and bb.host != "broad":
            reasons.append(f"宿主不符({bb.host} vs {host}),无法复制")
            disq = True
        elif bb.host == "broad":
            score += 0.5
            reasons.append("广宿主范围")
        else:
            score += 1.0

        if insert_kb > bb.max_insert_kb:
            reasons.append(f"插入片段 {insert_kb} kb 超过容量 {bb.max_insert_kb} kb")
            disq = True
        else:
            score += 0.5 * (1.0 - insert_kb / max(bb.max_insert_kb, 1e-9))

        if expression:
            if bb.promoter:
                score += 1.0
                reasons.append(f"带表达启动子 {bb.promoter}")
            else:
                reasons.append("没有表达启动子(克隆载体)")
                score -= 1.0

        if secretion:
            if "secretion" in bb.notes or "periplasmic" in bb.notes:
                score += 1.0
                reasons.append("带分泌信号(周质/胞外)")
            else:
                score -= 0.5

        if copy_number:
            if bb.copy_number == copy_number:
                score += 0.8
                reasons.append(f"{copy_number} 拷贝数符合要求")
            else:
                score -= 0.3 * abs(_COPY_RANK.get(bb.copy_number, 2)
                                   - _COPY_RANK.get(copy_number, 2))

        if marker and bb.marker.lower() != marker.lower():
            reasons.append(f"标记是 {bb.marker},要求 {marker}")
            disq = True
        if bb.marker.lower() in avoid:
            reasons.append(f"标记 {bb.marker} 在排除列表里")
            disq = True

        if incompatible_with and incompatible_with.lower() in bb.origin.lower():
            reasons.append(
                f"复制起点 {bb.origin} 与已有的 {incompatible_with} 不相容,"
                f"两者无法稳定共存")
            disq = True
        elif incompatible_with:
            score += 0.6
            reasons.append(f"起点 {bb.origin} 与 {incompatible_with} 相容")

        if standard:
            if standard.lower() in bb.standard.lower():
                score += 1.0
                reasons.append(f"兼容 {bb.standard}")
            else:
                score -= 0.5

        out.append(BackboneChoice(backbone=bb, score=score, reasons=reasons,
                                  disqualified=disq))

    out.sort(key=lambda c: (c.disqualified, -c.score, c.backbone.name))
    return out


@register_function(
    aliases=["plot_backbone_choice", "载体选择图", "骨架排名图",
             "plot_vector_choice"],
    category="synthetic_biology",
    description="画载体骨架排名:合格候选按分数排序,被否决的标红并给出否决原因(宿主不符/容量不足/起点不相容)。Plot backbone ranking with the disqualification reasons.",
    examples=["ov.synbio.plot_backbone_choice(picks)"],
    related=["synbio.select_backbone"],
    requires={},
    produces={},
)
def plot_backbone_choice(choices: Sequence[BackboneChoice], ax=None):
    """Horizontal bars, rejected options in red with their reason."""
    from ._plot import _mpl
    plt = _mpl()

    items = list(choices)
    names = [c.backbone.name for c in items]
    scores = [c.score for c in items]
    colours = ["#E41A1C" if c.disqualified else "#4DAF4A" for c in items]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(items) + 1.6))
    else:
        fig = ax.figure
    ax.barh(range(len(items)), scores, color=colours)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, c="grey", lw=0.8)
    ax.set_xlabel("suitability score")
    n_ok = sum(1 for c in items if not c.disqualified)
    ax.set_title(f"{n_ok}/{len(items)} usable (red = disqualified)")
    for i, c in enumerate(items):
        if c.disqualified and c.reasons:
            ax.text(min(scores) if scores else 0, i, "  " + c.reasons[0],
                    va="center", fontsize=6, color="#E41A1C")
    fig.tight_layout()
    return fig, ax


__all__ = ["query_parts", "fetch_part", "Part", "select_backbone",
           "Backbone", "BackboneChoice", "BACKBONE_CATALOGUE",
           "plot_backbone_choice", "REGISTRIES"]

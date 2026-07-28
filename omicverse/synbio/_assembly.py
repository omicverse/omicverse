r"""DNA assembly & construct design (Biopython).

Plan how parts become a plasmid:

* :func:`restriction_map` — restriction sites for a panel of enzymes.
* :func:`golden_gate` — Type IIS (BsaI/BsmBI) assembly by 4-nt fusion sites:
  digest each fragment, read its overhangs, and order fragments into a circle.
* :func:`gibson_assembly` — join fragments sharing terminal homology into one
  (circular) sequence.
* :func:`annotate_construct` — quick feature annotation (ORFs + a small motif
  library: promoters, RBS, terminators, common oris / resistance markers).
* :func:`read_genbank` / :func:`write_genbank` — annotated construct I/O.

All CPU, on Biopython (already an omicverse dependency).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from .._registry import register_function

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def _bio(fn: str):
    try:
        import Bio  # noqa: F401
        return Bio
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"ov.synbio.{fn} 需要 Biopython。请 pip install biopython "
            "(或 pip install 'omicverse[synbio]')。"
        ) from exc


# small motif library for quick annotation (name, pattern, type).
_FEATURE_MOTIFS = [
    ("T7_promoter", "TAATACGACTCACTATAG", "promoter"),
    ("lac_promoter", "TTTACACTTTATGCTTCCGGCTCG", "promoter"),
    ("Shine_Dalgarno", "AGGAGG", "RBS"),
    ("T7_terminator", "CTAGCATAACCCCTTGGGGCCTCTAAACGGGTCTTGAGGGGTTTTTTG", "terminator"),
    ("rrnB_T1_terminator", "AGAAAGGAAGAGT", "terminator"),
    ("BsaI_site", "GGTCTC", "TypeIIS"),
    ("BsmBI_site", "CGTCTC", "TypeIIS"),
]


@register_function(
    aliases=["restriction_map", "酶切图谱", "限制性酶切", "restriction_sites",
             "酶切位点", "restriction"],
    category="synthetic_biology",
    description="限制性酶切图谱:列出给定酶(默认常用酶)在序列上的切点位置。Restriction-enzyme site map for a DNA sequence.",
    examples=[
        "ov.synbio.restriction_map(dna, enzymes=['EcoRI','BsaI','XhoI'])",
    ],
    related=["synbio.golden_gate", "synbio.gibson_assembly"],
    requires={},
    produces={},
)
def restriction_map(sequence: str,
                    enzymes: Optional[Sequence[str]] = None,
                    linear: bool = True) -> Dict[str, List[int]]:
    """Return ``{enzyme: [cut positions]}`` for *sequence*."""
    _bio("restriction_map")
    from Bio.Seq import Seq
    from Bio import Restriction

    seq = Seq(sequence.upper().replace("U", "T"))
    if enzymes is None:
        batch = Restriction.CommOnly           # commercially available enzymes
    else:
        from Bio.Restriction import RestrictionBatch
        batch = RestrictionBatch(list(enzymes))
    result = batch.search(seq, linear=linear)
    return {str(enz): sites for enz, sites in result.items() if sites}


@dataclass
class AssemblyResult:
    sequence: str
    circular: bool
    n_fragments: int
    junctions: List[str] = field(default_factory=list)
    method: str = ""

    def __repr__(self) -> str:  # pragma: no cover
        return (f"AssemblyResult({self.method}, {self.n_fragments} fragments -> "
                f"{len(self.sequence)} bp, circular={self.circular})")


@register_function(
    aliases=["golden_gate", "金门组装", "GoldenGate", "typeIIS_assembly",
             "golden_gate_assembly", "BsaI组装"],
    category="synthetic_biology",
    description="Golden Gate 组装:用 Type IIS 酶(BsaI/BsmBI)切出 4-nt 融合位点,按互补悬挂端把片段拼成环状构建体。Golden Gate (Type IIS) assembly by 4-nt overhangs.",
    examples=[
        "res = ov.synbio.golden_gate([frag1, frag2, frag3], enzyme='BsaI')",
    ],
    related=["synbio.gibson_assembly", "synbio.restriction_map"],
    requires={},
    produces={},
)
@register_function(
    aliases=["domesticate", "加酶切位点", "驯化片段", "flank_fragments",
             "add_typeIIS_sites", "golden_gate_ready"],
    category="synthetic_biology",
    description="给片段两端加上 Type IIS 识别位点和指定的 4 nt 悬挂端,使其可用于 Golden Gate 组装 —— 这是设计到组装之间必做的一步('驯化'),不做 golden_gate 会直接拒绝片段。悬挂端按相邻片段首尾相接的顺序串起来:第 i 个片段的右端与第 i+1 个的左端用同一个悬挂端,最后一个的右端接回第一个的左端(环状)。同时检查片段内部是否已含该酶的位点 —— 内部位点会被酶切开,必须先在密码子层面去掉。Add Type IIS sites and overhangs so fragments can be Golden Gate assembled.",
    examples=[
        "ready = ov.synbio.domesticate([promoter, cds, term], overhangs=['AATG','AGGT','GCTT'])",
        "asm = ov.synbio.golden_gate(ready)",
    ],
    related=["synbio.golden_gate", "synbio.design_overhang_set",
             "synbio.overhang_fidelity", "synbio.restriction_map"],
    requires={},
    produces={},
)
def domesticate(fragments: Sequence[str], overhangs: Sequence[str],
                enzyme: str = "BsaI", spacer: str = "A") -> List[str]:
    """Flank each fragment with a Type IIS site and its junction overhangs.

    ``overhangs[i]`` becomes the junction between fragment *i* and fragment
    *i+1*, and the last wraps round to close the circle — so ``len(overhangs)``
    must equal ``len(fragments)``.

    Raises if a fragment already contains the enzyme's recognition site on either
    strand: an internal site is cut by the enzyme and destroys the assembly, and
    it cannot be fixed here — it has to be removed upstream, which for a coding
    sequence means ``codon_optimize(..., avoid_enzymes=[enzyme])``.
    """
    site = {"BsaI": "GGTCTC", "BsmBI": "CGTCTC", "BbsI": "GAAGAC"}.get(
        enzyme, "GGTCTC")
    rsite = _revcomp(site)
    frags = [str(f).upper().replace("U", "T") for f in fragments]
    overs = [str(o).upper().replace("U", "T") for o in overhangs]
    if len(overs) != len(frags):
        raise ValueError(
            f"{len(frags)} 个片段需要 {len(frags)} 个悬挂端(每个连接一个,最后一个"
            f"闭合成环),给了 {len(overs)} 个。")
    for over in overs:
        if len(over) != 4:
            raise ValueError(f"悬挂端必须是 4 nt,得到 {over!r}")
    if len(set(overs)) != len(overs):
        raise ValueError(f"悬挂端有重复,连接会错配:{overs}")

    ready: List[str] = []
    for i, frag in enumerate(frags):
        for found, where in ((frag.find(site), "正向"), (frag.find(rsite), "反向")):
            if found >= 0:
                raise ValueError(
                    f"第 {i + 1} 个片段内部第 {found} 位含 {enzyme} 位点"
                    f"({where}) —— 酶会把片段切开。这一步无法修正:编码序列请用 "
                    f"codon_optimize(..., avoid_enzymes=['{enzyme}']) 重新生成,"
                    f"非编码片段请换一个酶或改序列。")
        left = overs[i - 1]                      # closes with the previous one
        right = overs[i]
        ready.append(site + spacer + left + frag + right + spacer + rsite)
    return ready


@register_function(
    aliases=["gibson_arms", "加同源臂", "Gibson同源臂", "overlap_arms",
             "gibson_ready", "设计重叠区"],
    category="synthetic_biology",
    description="给片段两端加上与相邻片段的同源重叠区,使其可用于 Gibson/NEBuilder 组装 —— 与 Golden Gate 的驯化(domesticate)对应的必做一步,不做 gibson_assembly 会直接拒绝片段。每个片段的 3' 端接上下一个片段的前 overlap 个碱基;circular=True 时最后一个接回第一个。这些重叠区就是订购引物时要加的 5' 尾巴。Add homology arms so fragments can be Gibson assembled.",
    examples=[
        "ready = ov.synbio.gibson_arms([promoter, cds, term], overlap=25)",
        "asm = ov.synbio.gibson_assembly(ready)",
    ],
    related=["synbio.gibson_assembly", "synbio.domesticate",
             "synbio.design_primers"],
    requires={},
    produces={},
)
def gibson_arms(fragments: Sequence[str], overlap: int = 25,
                circular: bool = True) -> List[str]:
    """Append each fragment's downstream neighbour's first ``overlap`` bases.

    These arms are what you actually order as 5' tails on the amplification
    primers. ``gibson_assembly`` refuses fragments without them, which is correct
    — Gibson joins by annealing chewed-back ends, and blunt fragments have nothing
    to anneal to.
    """
    frags = [str(f).upper().replace("U", "T") for f in fragments]
    if len(frags) < 2:
        raise ValueError("Gibson 组装至少需要 2 个片段。")
    if overlap < 15:
        raise ValueError(
            f"同源重叠区 {overlap} bp 太短;Gibson 通常用 20–40 bp,"
            f"gibson_assembly 的下限是 20。")
    for i, frag in enumerate(frags):
        if len(frag) < overlap:
            raise ValueError(
                f"第 {i + 1} 个片段只有 {len(frag)} bp,短于 {overlap} bp 的重叠区 —— "
                f"这么短的片段应该直接合成成寡核苷酸,不走 Gibson。")
    ready: List[str] = []
    last = len(frags) if circular else len(frags) - 1
    for i, frag in enumerate(frags):
        if i < last:
            neighbour = frags[(i + 1) % len(frags)]
            ready.append(frag + neighbour[:overlap])
        else:
            ready.append(frag)
    return ready


def golden_gate(fragments: Sequence[str], enzyme: str = "BsaI") -> AssemblyResult:
    """Simulate a Golden Gate assembly.

    Each fragment must carry the Type IIS recognition site (BsaI ``GGTCTC`` /
    BsmBI ``CGTCTC``) near both ends; the enzyme cuts downstream leaving 4-nt
    overhangs. Fragments are ordered so complementary overhangs ligate into a
    circle."""
    _bio("golden_gate")
    site = {"BsaI": "GGTCTC", "BsmBI": "CGTCTC"}.get(enzyme, "GGTCTC")
    rsite = _revcomp(site)   # the enzyme's site on the bottom strand (top: GAGACC)
    off = 1  # spacer between recognition site and the 4-nt overhang

    # extract each fragment's insert + its 5' (left) and 3' (right) overhangs,
    # all read on the top strand 5'->3' so complementary ends compare directly.
    parts = []
    for frag in fragments:
        s = frag.upper().replace("U", "T")
        fwd = s.find(site)              # 5' site GGTCTC(N)NNNN
        rev = s.rfind(rsite)            # 3' site (top strand) NNNN(N)GAGACC
        if fwd < 0 or rev < 0:
            raise ValueError(
                f"片段缺少 {enzyme} 位点({site} 于 5' 端、{rsite} 于 3' 端);"
                "Golden Gate 需要每个片段两端各含一个 Type IIS 位点。")
        left5 = fwd + len(site) + off
        left_oh = s[left5:left5 + 4]
        right_end = rev - off           # bottom-strand cut leaves a 4-nt overhang
        right_oh = s[right_end - 4:right_end]
        insert = s[left5 + 4:right_end - 4]
        parts.append({"left": left_oh, "insert": insert, "right": right_oh})

    # greedy chain: match each right overhang to the next left overhang
    order = [0]
    used = {0}
    while len(order) < len(parts):
        cur = parts[order[-1]]
        nxt = next((i for i in range(len(parts))
                    if i not in used and parts[i]["left"] == cur["right"]), None)
        if nxt is None:
            break
        order.append(nxt)
        used.add(nxt)

    seq = ""
    junctions = []
    for idx in order:
        p = parts[idx]
        seq += p["left"] + p["insert"]
        junctions.append(p["left"])
    circular = len(order) == len(parts) and parts[order[-1]]["right"] == parts[order[0]]["left"]
    return AssemblyResult(sequence=seq, circular=circular, n_fragments=len(parts),
                          junctions=junctions, method=f"GoldenGate/{enzyme}")


@register_function(
    aliases=["gibson_assembly", "gibson组装", "Gibson", "等温组装",
             "overlap_assembly", "同源组装"],
    category="synthetic_biology",
    description="Gibson 等温组装:将末端有同源重叠的片段按重叠区拼接成一条(环状)序列。Gibson assembly — join fragments by terminal homology overlaps.",
    examples=[
        "res = ov.synbio.gibson_assembly([frag1, frag2], min_overlap=20)",
    ],
    related=["synbio.golden_gate", "synbio.restriction_map"],
    requires={},
    produces={},
)
def gibson_assembly(fragments: Sequence[str], min_overlap: int = 20,
                    circular: bool = True) -> AssemblyResult:
    """Join fragments sharing ``>= min_overlap`` bp terminal homology.

    Fragments are stitched in the given order; each junction's shared overlap is
    detected and merged once. If ``circular``, the last fragment's 3' end is
    expected to overlap the first fragment's 5' end."""
    _bio("gibson_assembly")
    frags = [f.upper().replace("U", "T") for f in fragments]

    def _overlap(a: str, b: str) -> int:
        for k in range(min(len(a), len(b)), min_overlap - 1, -1):
            if a[-k:] == b[:k]:
                return k
        return 0

    seq = frags[0]
    junctions = []
    for nxt in frags[1:]:
        k = _overlap(seq, nxt)
        if k == 0:
            raise ValueError(
                f"相邻片段无 >= {min_overlap} bp 的末端同源重叠;Gibson 组装需要设计重叠区。")
        junctions.append(seq[-k:])
        seq = seq + nxt[k:]
    if circular:
        k = _overlap(seq, frags[0])
        if k >= min_overlap:
            seq = seq[:-k]
            junctions.append("(circular)")
    return AssemblyResult(sequence=seq, circular=circular, n_fragments=len(frags),
                          junctions=junctions, method="Gibson")


@register_function(
    aliases=["annotate_construct", "构建体注释", "质粒注释", "plasmid_annotation",
             "feature_annotation", "序列注释", "annotate"],
    category="synthetic_biology",
    description="构建体/质粒注释:标注 ORF 与常见元件(启动子/RBS/终止子/Type IIS 位点)——快速理解一段构建序列。Annotate ORFs + common parts in a construct sequence.",
    examples=["feats = ov.synbio.annotate_construct(plasmid_dna)"],
    related=["synbio.read_genbank", "synbio.write_genbank"],
    requires={},
    produces={},
)
def annotate_construct(sequence: str, min_orf_aa: int = 40) -> List[Dict]:
    """Annotate ORFs and common regulatory motifs in *sequence*.

    Returns a list of ``{name, type, start, end, strand}`` features."""
    seq = sequence.upper().replace("U", "T")
    feats: List[Dict] = []

    # motif library (both strands)
    for strand, s, flip in (("+", seq, False), ("-", _revcomp(seq), True)):
        for name, patt, ftype in _FEATURE_MOTIFS:
            start = 0
            while True:
                i = s.find(patt, start)
                if i < 0:
                    break
                a, b = (len(seq) - i - len(patt), len(seq) - i) if flip else (i, i + len(patt))
                feats.append({"name": name, "type": ftype, "start": a,
                              "end": b, "strand": strand})
                start = i + 1

    # ORFs (ATG..stop) on both strands
    stops = {"TAA", "TAG", "TGA"}
    for strand, s in (("+", seq), ("-", _revcomp(seq))):
        for frame in range(3):
            i = frame
            while i < len(s) - 2:
                if s[i:i + 3] == "ATG":
                    j = i
                    while j < len(s) - 2:
                        if s[j:j + 3] in stops:
                            aa = (j - i) // 3
                            if aa >= min_orf_aa:
                                a, b = (i, j + 3) if strand == "+" else \
                                       (len(seq) - (j + 3), len(seq) - i)
                                feats.append({"name": f"ORF_{aa}aa", "type": "CDS",
                                              "start": a, "end": b, "strand": strand})
                            i = j
                            break
                        j += 3
                i += 3
    feats.sort(key=lambda f: f["start"])
    return feats


@register_function(
    aliases=["write_genbank", "导出genbank", "save_genbank", "genbank导出"],
    category="synthetic_biology",
    description="导出 GenBank:把序列 + 注释特征写成带注释的 .gb 文件(Biopython)。Write an annotated construct to a GenBank file.",
    examples=["ov.synbio.write_genbank(seq, feats, 'construct.gb')"],
    related=["synbio.annotate_construct", "synbio.read_genbank"],
    requires={},
    produces={},
)
def write_genbank(sequence: str, features: Optional[List[Dict]], path: str,
                  name: str = "construct", circular: bool = True) -> str:
    """Write *sequence* + *features* to a GenBank file at *path*."""
    _bio("write_genbank")
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.SeqFeature import SeqFeature, FeatureLocation
    from Bio import SeqIO

    rec = SeqRecord(Seq(sequence.upper()), id=name, name=name[:16],
                    description=f"ov.synbio construct {name}")
    rec.annotations["molecule_type"] = "DNA"
    rec.annotations["topology"] = "circular" if circular else "linear"
    for f in (features or []):
        loc = FeatureLocation(int(f["start"]), int(f["end"]),
                              strand=1 if f.get("strand", "+") == "+" else -1)
        rec.features.append(SeqFeature(loc, type=f.get("type", "misc_feature"),
                                       qualifiers={"label": [f.get("name", "")]}))
    SeqIO.write(rec, path, "genbank")
    return path


@register_function(
    aliases=["read_genbank", "读取genbank", "load_genbank", "genbank读取"],
    category="synthetic_biology",
    description="读取 GenBank:解析 .gb 文件为序列 + 特征列表(Biopython)。Read a GenBank file → sequence + features.",
    examples=["seq, feats = ov.synbio.read_genbank('construct.gb')"],
    related=["synbio.write_genbank", "synbio.annotate_construct"],
    requires={},
    produces={},
)
def read_genbank(path: str) -> Tuple[str, List[Dict]]:
    """Read a GenBank file → ``(sequence, features)``."""
    _bio("read_genbank")
    from Bio import SeqIO

    rec = next(SeqIO.parse(path, "genbank"))
    feats = []
    for f in rec.features:
        label = f.qualifiers.get("label", f.qualifiers.get("gene", [""]))[0]
        feats.append({"name": label, "type": f.type,
                      "start": int(f.location.start), "end": int(f.location.end),
                      "strand": "+" if f.location.strand != -1 else "-"})
    return str(rec.seq), feats


__all__ = [
    "gibson_arms",
    "domesticate","restriction_map", "golden_gate", "gibson_assembly",
           "annotate_construct", "write_genbank", "read_genbank",
           "AssemblyResult"]

r"""SBOL — the Synthetic Biology Open Language interchange standard.

`SBOL <https://sbolstandard.org/>`_ is the community standard for exchanging
genetic-design data (parts, devices, constructs) between tools and registries
(iGEM, SynBioHub, Benchling…). These helpers bridge an ``ov.synbio`` construct
(a sequence + a list of ``{name, type, start, end, strand}`` features, as
produced by :func:`annotate_construct`) to and from an SBOL2 document:

* :func:`write_sbol` — export a construct to an SBOL2 XML file.
* :func:`read_sbol` — read an SBOL2 file back into ``(sequence, features)``.

Uses `pySBOL2 <https://github.com/SynBioDex/pySBOL2>`_.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .._registry import register_function

# ov feature type -> Sequence Ontology term (SBOL role)
_SO = {
    "promoter": "SO:0000167", "RBS": "SO:0000139", "CDS": "SO:0000316",
    "ORF": "SO:0000236", "terminator": "SO:0000141", "TypeIIS": "SO:0001687",
    "primer": "SO:0000112", "misc_feature": "SO:0000110",
}
_SO_INV = {v: k for k, v in _SO.items()}


def _sbol(fn: str):
    try:
        import sbol2
        return sbol2
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"ov.synbio.{fn} 需要 pySBOL2。请 pip install sbol2 "
            "(或 pip install 'omicverse[synbio]')。"
        ) from exc


@register_function(
    aliases=["write_sbol", "导出SBOL", "sbol_export", "保存SBOL", "to_sbol",
             "SBOL导出", "写SBOL"],
    category="synthetic_biology",
    description="导出 SBOL:把构建体(序列 + 特征)写成 SBOL2 标准 XML,便于与 iGEM/SynBioHub/Benchling 等工具和注册库互操作。Export a construct (sequence + features) to an SBOL2 file.",
    examples=[
        "feats = ov.synbio.annotate_construct(dna)",
        "ov.synbio.write_sbol(dna, feats, name='myDevice', path='device.xml')",
    ],
    related=["synbio.annotate_construct", "synbio.read_sbol", "synbio.write_genbank"],
    requires={},
    produces={},
)
def write_sbol(sequence: str, features: Optional[List[Dict]] = None,
               name: str = "construct", path: Optional[str] = None,
               uri_prefix: str = "http://omicverse.org/synbio") -> str:
    """Write *sequence* + *features* to SBOL2. Returns the file path (or, if
    ``path`` is None, the SBOL document as a string)."""
    sbol2 = _sbol("write_sbol")
    seq = sequence.upper().replace("U", "T")
    if features is None:
        from ._assembly import annotate_construct
        features = annotate_construct(seq)

    sbol2.setHomespace(uri_prefix)
    doc = sbol2.Document()
    comp = sbol2.ComponentDefinition(name, sbol2.BIOPAX_DNA)
    comp.roles = ["https://identifiers.org/SO:0000804"]   # engineered_region
    doc.addComponentDefinition(comp)

    sobj = sbol2.Sequence(f"{name}_seq", seq, sbol2.SBOL_ENCODING_IUPAC)
    doc.addSequence(sobj)
    comp.sequences = [sobj.identity]

    for i, f in enumerate(features):
        role = _SO.get(f.get("type", "misc_feature"), _SO["misc_feature"])
        sub = sbol2.ComponentDefinition(f"{name}_f{i}", sbol2.BIOPAX_DNA)
        sub.roles = [f"https://identifiers.org/so/{role}"]
        sub.name = str(f.get("name", f"feature_{i}"))
        doc.addComponentDefinition(sub)
        sa = comp.sequenceAnnotations.create(f"{name}_ann{i}")
        sa.name = str(f.get("name", f"feature_{i}"))
        loc = sa.locations.createRange("range")
        loc.start = int(f["start"]) + 1                 # SBOL is 1-based inclusive
        loc.end = int(f["end"])
        loc.orientation = (sbol2.SBOL_ORIENTATION_INLINE
                           if f.get("strand", "+") == "+"
                           else sbol2.SBOL_ORIENTATION_REVERSE_COMPLEMENT)

    if path is None:
        return doc.writeString()
    doc.write(path)
    return path


@register_function(
    aliases=["read_sbol", "读取SBOL", "sbol_import", "导入SBOL", "from_sbol",
             "SBOL读取", "解析SBOL"],
    category="synthetic_biology",
    description="读取 SBOL:解析 SBOL2 XML,取回序列与特征列表({name,type,start,end,strand}),与 ov.synbio 其余工具互通。Read an SBOL2 file → (sequence, features).",
    examples=["seq, feats = ov.synbio.read_sbol('device.xml')"],
    related=["synbio.write_sbol", "synbio.annotate_construct", "synbio.read_genbank"],
    requires={},
    produces={},
)
def read_sbol(path: str) -> Tuple[str, List[Dict]]:
    """Read an SBOL2 document → ``(sequence, features)``."""
    sbol2 = _sbol("read_sbol")
    doc = sbol2.Document()
    doc.read(path)

    seq = ""
    feats: List[Dict] = []
    for comp in doc.componentDefinitions:
        anns = list(comp.sequenceAnnotations)
        if not anns and not comp.sequences:
            continue
        # the top-level construct is the one carrying sequence annotations
        if comp.sequences:
            try:
                sobj = doc.getSequence(comp.sequences[0])
                if sobj and sobj.elements and len(sobj.elements) > len(seq):
                    seq = str(sobj.elements).upper()
            except Exception:
                pass
        for sa in anns:
            for loc in sa.locations:
                start = getattr(loc, "start", None)
                end = getattr(loc, "end", None)
                if start is None or end is None:
                    continue
                orient = getattr(loc, "orientation", "")
                strand = "-" if "reverseComplement" in str(orient) else "+"
                feats.append({
                    "name": str(getattr(sa, "name", "") or sa.displayId),
                    "type": "misc_feature", "start": int(start) - 1,
                    "end": int(end), "strand": strand})
    feats.sort(key=lambda f: f["start"])
    return seq, feats


__all__ = ["write_sbol", "read_sbol"]

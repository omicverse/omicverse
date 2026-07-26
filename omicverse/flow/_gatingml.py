r"""Gating-ML 2.0 interchange.

Gating-ML is the ISAC Recommendation for exchanging a gating strategy between
tools. It is the only vendor-neutral way to hand a hierarchy to someone who does
not use omicverse, which makes it the difference between a reproducible analysis
and a screenshot.

WHY THIS IS WRITTEN HERE RATHER THAN DELEGATED
----------------------------------------------
The spec types ``gating:id`` as ``xs:ID``, which is an XML ``NCName``: it may
not start with a digit and may not contain ``+``, ``-`` at the start, spaces, or
most punctuation. Gate names in real life are ``CD3+``, ``CD4+CD8-``,
``Live cells``, ``CD45RA+CCR7+`` — ``+`` is the single most common character in
a cytometry gate name.

FlowKit does not sanitise these and does not raise. ``export_gml`` succeeds and
writes a document that its own ``parse_gating_xml`` then rejects. So any
unmediated wrapper produces corrupt interchange BY DEFAULT, for the most common
naming convention in the field.

This module therefore owns a display-name <-> ``xs:ID`` mapping: every gate gets
a generated, valid id, and its human name is carried in ``gating:custom_info``
where it belongs. Round-tripping restores the display names exactly.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ._gates import (
    BooleanGate, EllipsoidGate, Gate, PolygonGate, QuadrantGate, RectangleGate,
)
from ._strategy import ROOT, GatingStrategy
from ._transforms import Transform, make_transform

__all__ = ["to_gatingml", "from_gatingml", "write_gatingml", "read_gatingml",
           "is_valid_xml_id", "sanitize_id"]

NS = {
    "gating": "http://www.isac-net.org/std/Gating-ML/v2.0/gating",
    "transforms": "http://www.isac-net.org/std/Gating-ML/v2.0/transformations",
    "data-type": "http://www.isac-net.org/std/Gating-ML/v2.0/datatypes",
}

# xs:ID == xs:NCName: a letter or underscore, then letters, digits, '.', '-', '_'.
_NCNAME = re.compile(r"^[A-Za-z_][A-Za-z0-9._-]*$")


def is_valid_xml_id(name: str) -> bool:
    """Whether ``name`` may be used verbatim as an ``xs:ID``.

    Exposed because the answer is surprising: almost no real gate name is.
    """
    return bool(_NCNAME.match(str(name)))


def sanitize_id(name: str, taken: Optional[set] = None) -> str:
    """Derive a valid ``xs:ID`` from a display name, uniquely.

    ``+``/``-`` become ``pos``/``neg`` rather than being stripped, because
    ``CD4+CD8-`` and ``CD4-CD8+`` must not collide — they are opposite
    populations, and silently merging them would be the worst possible failure
    of an interchange format.
    """
    s = str(name)
    s = re.sub(r"\+", "_pos", s)
    s = re.sub(r"-(?=$|[^0-9])", "_neg", s)
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    if not s or not re.match(r"^[A-Za-z_]", s):
        s = "gate_" + s
    if taken is not None:
        base, i = s, 2
        while s in taken:
            s = f"{base}_{i}"
            i += 1
        taken.add(s)
    return s


def _q(prefix: str, tag: str) -> str:
    return f"{{{NS[prefix]}}}{tag}"


def _transform_id(dim: str, tr: Transform) -> str:
    return sanitize_id(f"xf_{dim}_{tr.name}")


def _add_transform(root: ET.Element, xid: str, tr: Transform) -> None:
    el = ET.SubElement(root, _q("transforms", "transformation"),
                       {_q("transforms", "id"): xid})
    kind = tr.name
    if kind == "logicle":
        ET.SubElement(el, _q("transforms", "logicle"), {
            _q("transforms", "T"): repr(float(tr.t)),
            _q("transforms", "W"): repr(float(tr.w)),
            _q("transforms", "M"): repr(float(tr.m)),
            _q("transforms", "A"): repr(float(tr.a)),
        })
    elif kind == "hyperlog":
        ET.SubElement(el, _q("transforms", "hyperlog"), {
            _q("transforms", "T"): repr(float(tr.t)),
            _q("transforms", "W"): repr(float(tr.w)),
            _q("transforms", "M"): repr(float(tr.m)),
            _q("transforms", "A"): repr(float(tr.a)),
        })
    elif kind == "asinh":
        ET.SubElement(el, _q("transforms", "fasinh"), {
            _q("transforms", "T"): repr(float(tr.t)),
            _q("transforms", "M"): repr(float(tr.m)),
            _q("transforms", "A"): repr(float(tr.a)),
        })
    elif kind == "log":
        ET.SubElement(el, _q("transforms", "flog"), {
            _q("transforms", "T"): repr(float(tr.t)),
            _q("transforms", "M"): repr(float(tr.m)),
        })
    else:  # linear
        ET.SubElement(el, _q("transforms", "flin"), {
            _q("transforms", "T"): repr(float(tr.t)),
            _q("transforms", "A"): repr(float(tr.a)),
        })


def _dim(parent: ET.Element, name: str, xf: Optional[str],
         mn: Optional[float] = None, mx: Optional[float] = None) -> ET.Element:
    attrs = {}
    if mn is not None:
        attrs[_q("gating", "min")] = repr(float(mn))
    if mx is not None:
        attrs[_q("gating", "max")] = repr(float(mx))
    if xf:
        attrs[_q("gating", "transformation-ref")] = xf
    d = ET.SubElement(parent, _q("gating", "dimension"), attrs)
    ET.SubElement(d, _q("data-type", "fcs-dimension"),
                  {_q("data-type", "name"): name})
    return d


def to_gatingml(strategy: GatingStrategy) -> ET.ElementTree:
    """Serialise a strategy to a Gating-ML 2.0 document."""
    root = ET.Element(_q("gating", "Gating-ML"))
    for p, uri in NS.items():
        root.set(f"xmlns:{p}", uri)

    ids: Dict[str, str] = {}
    taken: set = set()
    for name in strategy.gates:
        ids[name] = sanitize_id(name, taken)

    # Transforms are referenced by id, so they must exist before the gates.
    seen_xf: Dict[str, Transform] = {}
    for gate in strategy.gates.values():
        for dim, tr in gate.transforms.items():
            xid = _transform_id(dim, tr)
            if xid not in seen_xf:
                seen_xf[xid] = tr
                _add_transform(root, xid, tr)

    for name, gate in strategy.gates.items():
        parent = strategy.parent_of(name)
        attrs = {_q("gating", "id"): ids[name]}
        if parent != ROOT and parent in ids:
            attrs[_q("gating", "parent_id")] = ids[parent]

        if isinstance(gate, RectangleGate):
            el = ET.SubElement(root, _q("gating", "RectangleGate"), attrs)
            for d, (lo, hi) in zip(gate.dims, gate.bounds):
                tr = gate.transforms.get(d)
                _dim(el, d, _transform_id(d, tr) if tr else None, lo, hi)
        elif isinstance(gate, PolygonGate):
            el = ET.SubElement(root, _q("gating", "PolygonGate"), attrs)
            for d in gate.dims:
                tr = gate.transforms.get(d)
                _dim(el, d, _transform_id(d, tr) if tr else None)
            for vx, vy in gate.vertices:
                v = ET.SubElement(el, _q("gating", "vertex"))
                for c in (vx, vy):
                    ET.SubElement(v, _q("gating", "coordinate"),
                                  {_q("data-type", "value"): repr(float(c))})
        elif isinstance(gate, EllipsoidGate):
            el = ET.SubElement(root, _q("gating", "EllipsoidGate"), attrs)
            for d in gate.dims:
                tr = gate.transforms.get(d)
                _dim(el, d, _transform_id(d, tr) if tr else None)
            mean = ET.SubElement(el, _q("gating", "mean"))
            for c in gate.mean:
                ET.SubElement(mean, _q("gating", "coordinate"),
                              {_q("data-type", "value"): repr(float(c))})
            cov = ET.SubElement(el, _q("gating", "covarianceMatrix"))
            for row in gate.covariance:
                r = ET.SubElement(cov, _q("gating", "row"))
                for c in row:
                    ET.SubElement(r, _q("gating", "entry"),
                                  {_q("data-type", "value"): repr(float(c))})
            ET.SubElement(el, _q("gating", "distanceSquare"),
                          {_q("data-type", "value"): repr(float(gate.distance_square))})
        elif isinstance(gate, BooleanGate):
            el = ET.SubElement(root, _q("gating", "BooleanGate"), attrs)
            op = ET.SubElement(el, _q("gating", gate.operator))
            for operand in gate.operands:
                ET.SubElement(op, _q("gating", "gateReference"),
                              {_q("gating", "ref"): ids.get(operand, sanitize_id(operand))})
        elif isinstance(gate, QuadrantGate):
            el = ET.SubElement(root, _q("gating", "QuadrantGate"), attrs)
            for d, div in zip(gate.dims, gate.dividers):
                tr = gate.transforms.get(d)
                dv = ET.SubElement(el, _q("gating", "divider"), {
                    _q("gating", "id"): sanitize_id(f"div_{name}_{d}"),
                    **({_q("gating", "transformation-ref"): _transform_id(d, tr)} if tr else {}),
                })
                ET.SubElement(dv, _q("data-type", "fcs-dimension"),
                              {_q("data-type", "name"): d})
                ET.SubElement(dv, _q("gating", "value")).text = repr(float(div))
        else:                                                 # pragma: no cover
            raise TypeError(f"cannot serialise gate type {type(gate).__name__}")

        # The DISPLAY name lives here, because it usually cannot be an xs:ID.
        info = ET.SubElement(el, _q("gating", "custom_info"))
        ET.SubElement(info, "omicverse-name").text = name
        if isinstance(gate, QuadrantGate):
            ET.SubElement(info, "omicverse-quadrants").text = "\t".join(gate.quadrant_names)

    return ET.ElementTree(root)


def write_gatingml(strategy: GatingStrategy, path: str) -> str:
    """Write a strategy to ``path`` as Gating-ML 2.0 and return the path.

    The document is re-parsed before it is returned. That is not paranoia: the
    failure this module exists to prevent is exactly "writes fine, will not read
    back", and it costs microseconds to be sure.
    """
    tree = to_gatingml(strategy)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    ET.parse(path)
    return path


def _f(el: ET.Element, prefix: str, attr: str) -> Optional[float]:
    v = el.get(_q(prefix, attr))
    return float(v) if v is not None else None


def _name_of(el: ET.Element, fallback: str) -> str:
    info = el.find(_q("gating", "custom_info"))
    if info is not None:
        node = info.find("omicverse-name")
        if node is not None and node.text:
            return node.text
    return fallback


def from_gatingml(tree: ET.ElementTree) -> GatingStrategy:
    """Parse a Gating-ML 2.0 document back into a strategy."""
    root = tree.getroot() if isinstance(tree, ET.ElementTree) else tree

    xforms: Dict[str, Transform] = {}
    for el in root.findall(_q("transforms", "transformation")):
        xid = el.get(_q("transforms", "id"))
        for tag, kind in (("logicle", "logicle"), ("hyperlog", "hyperlog"),
                          ("fasinh", "asinh"), ("flog", "log"), ("flin", "linear")):
            node = el.find(_q("transforms", tag))
            if node is None:
                continue
            p = {k: float(node.get(_q("transforms", k)))
                 for k in ("T", "W", "M", "A") if node.get(_q("transforms", k)) is not None}
            kw = {"t": p.get("T", 262144.0)}
            if "M" in p:
                kw["m"] = p["M"]
            if "W" in p:
                kw["w"] = p["W"]
            if "A" in p:
                kw["a"] = p["A"]
            xforms[xid] = make_transform(kind, **kw)
            break

    gs = GatingStrategy()
    id_to_name: Dict[str, str] = {}
    pending: List[Tuple[Gate, Optional[str]]] = []

    for el in root:
        tag = el.tag.split("}")[-1]
        if tag not in ("RectangleGate", "PolygonGate", "EllipsoidGate",
                       "BooleanGate", "QuadrantGate"):
            continue
        xid = el.get(_q("gating", "id"))
        name = _name_of(el, xid)
        id_to_name[xid] = name
        parent_id = el.get(_q("gating", "parent_id"))

        dims_el = el.findall(_q("gating", "dimension"))
        dims, transforms, bounds = [], {}, []
        for d in dims_el:
            fd = d.find(_q("data-type", "fcs-dimension"))
            dname = fd.get(_q("data-type", "name")) if fd is not None else ""
            dims.append(dname)
            ref = d.get(_q("gating", "transformation-ref"))
            if ref and ref in xforms:
                transforms[dname] = xforms[ref]
            bounds.append((_f(d, "gating", "min"), _f(d, "gating", "max")))

        if tag == "RectangleGate":
            gate: Gate = RectangleGate(name=name, dims=tuple(dims),
                                       transforms=transforms, bounds=tuple(bounds))
        elif tag == "PolygonGate":
            verts = [[float(c.get(_q("data-type", "value")))
                      for c in v.findall(_q("gating", "coordinate"))]
                     for v in el.findall(_q("gating", "vertex"))]
            gate = PolygonGate(name=name, dims=tuple(dims), transforms=transforms,
                               vertices=np.asarray(verts, dtype=float))
        elif tag == "EllipsoidGate":
            mean = [float(c.get(_q("data-type", "value")))
                    for c in el.find(_q("gating", "mean")).findall(_q("gating", "coordinate"))]
            cov = [[float(e.get(_q("data-type", "value")))
                    for e in r.findall(_q("gating", "entry"))]
                   for r in el.find(_q("gating", "covarianceMatrix")).findall(_q("gating", "row"))]
            d2 = float(el.find(_q("gating", "distanceSquare")).get(_q("data-type", "value")))
            gate = EllipsoidGate(name=name, dims=tuple(dims), transforms=transforms,
                                 mean=np.asarray(mean), covariance=np.asarray(cov),
                                 distance_square=d2)
        elif tag == "BooleanGate":
            for op in ("and", "or", "not"):
                node = el.find(_q("gating", op))
                if node is not None:
                    refs = [r.get(_q("gating", "ref"))
                            for r in node.findall(_q("gating", "gateReference"))]
                    gate = BooleanGate(name=name, dims=(), operator=op,
                                       operands=tuple(refs))
                    break
            else:                                             # pragma: no cover
                raise ValueError(f"boolean gate {xid!r} has no and/or/not child")
        else:
            divs, qdims = [], []
            for dv in el.findall(_q("gating", "divider")):
                fd = dv.find(_q("data-type", "fcs-dimension"))
                qdims.append(fd.get(_q("data-type", "name")) if fd is not None else "")
                ref = dv.get(_q("gating", "transformation-ref"))
                if ref and ref in xforms:
                    transforms[qdims[-1]] = xforms[ref]
                divs.append(float(dv.find(_q("gating", "value")).text))
            info = el.find(_q("gating", "custom_info"))
            qn = info.find("omicverse-quadrants") if info is not None else None
            gate = QuadrantGate(
                name=name, dims=tuple(qdims), transforms=transforms,
                dividers=tuple(divs),
                quadrant_names=tuple(qn.text.split("\t")) if qn is not None and qn.text else (),
            )
        pending.append((gate, parent_id))

    for gate, parent_id in pending:
        if isinstance(gate, BooleanGate):
            gate.operands = tuple(id_to_name.get(o, o) for o in gate.operands)
        gs.add_gate(gate, parent=id_to_name.get(parent_id, ROOT) if parent_id else ROOT)
    return gs


def read_gatingml(path: str) -> GatingStrategy:
    """Read a Gating-ML 2.0 file into a strategy."""
    return from_gatingml(ET.parse(path))

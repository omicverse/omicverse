"""Gating-ML 2.0 interchange.

The defect this module exists to avoid: `gating:id` is typed `xs:ID` (an XML
NCName), and almost no real cytometry gate name is one. FlowKit does not
sanitise and does not raise — `export_gml` succeeds and its own parser then
rejects the output.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

import omicverse.flow as fl


@pytest.mark.parametrize("name", ["CD3+", "CD4+CD8-", "Live cells", "CD45RA+CCR7+", "4-1BB+"])
def test_real_gate_names_are_not_valid_xml_ids(name):
    """Establishes that this is the NORM, not an edge case: '+' is the most
    common character in a cytometry gate name."""
    assert not fl.is_valid_xml_id(name)
    assert fl.is_valid_xml_id(fl.sanitize_id(name))


def test_opposite_populations_do_not_collide():
    """Stripping '+'/'-' would map CD4+CD8- and CD4-CD8+ onto the same id.
    Silently merging two opposite populations is the worst thing an interchange
    format can do."""
    assert fl.sanitize_id("CD4+CD8-") != fl.sanitize_id("CD4-CD8+")


def test_ids_stay_unique_within_a_document():
    taken = set()
    a = fl.sanitize_id("CD3 cells", taken)
    b = fl.sanitize_id("CD3+cells", taken)
    assert a != b


def strategy():
    lg = fl.make_transform("logicle", t=262144, w=0.5, m=4.5, a=0.0)
    return (fl.GatingStrategy("panel")
            .add_gate(fl.RectangleGate(name="Cells", dims=("FSC-A", "SSC-A"),
                                       bounds=((2e4, 1.2e5), (1e4, 9e4))))
            .add_gate(fl.PolygonGate(name="Singlets", dims=("FSC-A", "FSC-H"),
                      vertices=np.array([[2e4, 1.6e4], [1.2e5, 1.1e5],
                                         [1.2e5, 1.3e5], [2e4, 2.4e4]])), parent="Cells")
            .add_gate(fl.RectangleGate(name="CD3+", dims=("CD3",), bounds=((0.35, None),),
                      transforms={"CD3": lg}), parent="Singlets")
            .add_gate(fl.EllipsoidGate(name="Blast", dims=("FSC-A", "SSC-A"),
                      mean=np.array([5e4, 3e4]), covariance=np.diag([1e8, 5e7]),
                      distance_square=4.0), parent="Singlets")
            .add_gate(fl.BooleanGate(name="CD3+ not blast", dims=(), operator="not",
                      operands=("Blast",)), parent="Singlets"))


def test_the_document_reparses(tmp_path):
    """The failure mode is 'writes fine, will not read back'."""
    p = str(tmp_path / "s.xml")
    fl.write_gatingml(strategy(), p)
    ET.parse(p)


def test_every_written_id_is_a_valid_xml_id(tmp_path):
    p = str(tmp_path / "s.xml")
    fl.write_gatingml(strategy(), p)
    root = ET.parse(p).getroot()
    ids = [v for el in root.iter() for k, v in el.attrib.items() if k.endswith("}id")]
    assert ids, "no ids written"
    assert all(fl.is_valid_xml_id(i) for i in ids), [i for i in ids if not fl.is_valid_xml_id(i)]


def test_display_names_survive_the_round_trip(tmp_path):
    p = str(tmp_path / "s.xml")
    gs = strategy()
    fl.write_gatingml(gs, p)
    back = fl.read_gatingml(p)
    assert sorted(back.gates) == sorted(gs.gates)
    assert "CD3+" in back.gates, "the '+' must come back, not the sanitised id"


def test_the_hierarchy_survives(tmp_path):
    p = str(tmp_path / "s.xml")
    gs = strategy()
    fl.write_gatingml(gs, p)
    back = fl.read_gatingml(p)
    for name in gs.gates:
        assert back.parent_of(name) == gs.parent_of(name), name


def test_transforms_survive(tmp_path):
    p = str(tmp_path / "s.xml")
    gs = strategy()
    fl.write_gatingml(gs, p)
    back = fl.read_gatingml(p)
    assert back.gates["CD3+"].transforms["CD3"].to_dict() == \
        gs.gates["CD3+"].transforms["CD3"].to_dict()


def test_geometry_survives(tmp_path):
    p = str(tmp_path / "s.xml")
    gs = strategy()
    fl.write_gatingml(gs, p)
    back = fl.read_gatingml(p)
    assert np.allclose(back.gates["Singlets"].vertices, gs.gates["Singlets"].vertices)
    assert np.allclose(back.gates["Blast"].covariance, gs.gates["Blast"].covariance)
    assert back.gates["Blast"].distance_square == gs.gates["Blast"].distance_square
    assert back.gates["Cells"].bounds == gs.gates["Cells"].bounds


def test_boolean_operands_are_remapped_to_display_names(tmp_path):
    """Operands are written as xs:IDs and must come back as the names the rest
    of the strategy uses, or the gate references a population that does not
    exist."""
    p = str(tmp_path / "s.xml")
    fl.write_gatingml(strategy(), p)
    back = fl.read_gatingml(p)
    assert back.gates["CD3+ not blast"].operands == ("Blast",)


def test_the_round_tripped_strategy_gates_identically(tmp_path):
    """The end-to-end property: same file in, same events out."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 3000
    X = np.column_stack([
        np.abs(rng.normal(6e4, 1.5e4, n)), np.abs(rng.normal(6e4, 1.5e4, n)),
        np.abs(rng.normal(4e4, 1.2e4, n)), rng.normal(1e4, 4e3, n),
    ]).astype(np.float32)
    a = ad.AnnData(X, var=pd.DataFrame(index=["FSC-A", "FSC-H", "SSC-A", "CD3"]))

    gs = strategy()
    p = str(tmp_path / "s.xml")
    fl.write_gatingml(gs, p)
    back = fl.read_gatingml(p)

    r1 = gs.apply(a.copy(), write_obs=False)
    r2 = back.apply(a.copy(), write_obs=False)
    for name in r1.masks:
        assert np.array_equal(r1.masks[name], r2.masks[name]), name


def test_quadrants_survive(tmp_path):
    lg = fl.make_transform("logicle", t=262144, w=0.5, m=4.5, a=0.0)
    gs = fl.GatingStrategy().add_gate(
        fl.QuadrantGate(name="Q", dims=("CD4", "CD8"), dividers=(0.4, 0.4),
                        transforms={"CD4": lg, "CD8": lg}))
    p = str(tmp_path / "q.xml")
    fl.write_gatingml(gs, p)
    back = fl.read_gatingml(p)
    assert back.gates["Q"].dividers == gs.gates["Q"].dividers
    assert back.gates["Q"].quadrant_names == gs.gates["Q"].quadrant_names

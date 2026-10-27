"""Synthetic tests for cluster_marker_profile / score_cluster_signatures /
cluster_neighbor_composition (labels and signatures are arbitrary names)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_P = Path(__file__).resolve().parents[2] / "omicverse" / "space" / "_cluster_signature.py"


def _load():
    try:
        import omicverse.space as s
        return s
    except Exception:  # light-weight import when the full package is unavailable
        import sys, types
        reg = types.ModuleType("omicverse._registry")
        reg.register_function = lambda **k: (lambda f: f)
        sys.modules.setdefault("omicverse", types.ModuleType("omicverse"))
        sys.modules["omicverse._registry"] = reg
        sys.modules.setdefault("omicverse.space", types.ModuleType("omicverse.space"))
        spec = importlib.util.spec_from_file_location("omicverse.space._cs", _P)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m


cs = _load()


def _panel(seed=0):
    """Clusters: a (m1 high), b1/b2 (m2 high; split by m3 vs m4), q (nothing),
    plus a 'contaminated' cluster c with m2 moderate and m1 high."""
    rng = np.random.default_rng(seed)
    spec = {
        "a": dict(m1=0.9), "c": dict(m1=0.8, m2=0.35),
        "b1": dict(m2=0.9, m3=0.7), "b2": dict(m2=0.85, m4=0.7),
        "q1": {}, "q2": {},
    }
    rows = []
    for cl, mk in spec.items():
        for _ in range(200):
            r = {m: max(0.0, rng.normal(mk.get(m, 0.03), 0.05)) for m in ("m1", "m2", "m3", "m4")}
            r["cluster"] = cl
            rows.append(r)
    return pd.DataFrame(rows)


SIGS = {"A": {"m1": 1.0, "m2": -0.5}, "B": {"m2": 1.0, "m1": -0.5},
        "B_x": {"m3": 1.0, "m4": -0.5}, "B_y": {"m4": 1.0, "m3": -0.5}}


def test_profile_columns_and_standout():
    p = cs.cluster_marker_profile(_panel(), "cluster")
    assert {"mean:m1", "rel:m1", "z:m1", "pos:m1", "zpos:m1"} <= set(p.columns)
    assert p.attrs["standout"]["a"][0] == "m1"
    assert (p.filter(like="rel:") <= 1 + 1e-9).all().all()


def test_signatures_presence_parents_and_residual():
    p = cs.cluster_marker_profile(_panel(), "cluster")
    r = cs.score_cluster_signatures(p, SIGS, parents={"B_x": "B", "B_y": "B"}, polar=["B"],
                                    residual_label="R")
    assert r.at["a", "label"] == "A" and r.at["c", "label"] == "A"
    assert r.at["b1", "label"] == "B_x" and r.at["b2", "label"] == "B_y"
    assert r.at["q1", "label"] == "R" and r.at["q2", "label"] == "R"


def test_contrast_split_keeps_single_parent():
    df = _panel()
    df = df[df.cluster != "b2"]
    p = cs.cluster_marker_profile(df, "cluster")
    r = cs.score_cluster_signatures(p, SIGS, parents={"B_x": "B", "B_y": "B"})
    assert r.at["b1", "label"] == "B"          # one cluster: generic parent stays


def test_unmatched_and_alias_resolution():
    df = _panel().rename(columns={"m1": "M1a"})
    p = cs.cluster_marker_profile(df, "cluster")
    r = cs.score_cluster_signatures(p, {"A": {"m1": 1.0, "zz": 1.0}})
    assert r.attrs["marker_map"]["m1"] == "M1a"
    assert r.attrs["unmatched"] == ["zz"]


def test_neighbor_composition_assigns_residual():
    rng = np.random.default_rng(1)
    df = _panel()
    xy = np.zeros((len(df), 2))
    # q1 cells sit among a-cells, q2 among their own kind
    for cl, centre in {"a": (0, 0), "q1": (0, 0), "q2": (50, 50), "c": (100, 0),
                       "b1": (0, 100), "b2": (100, 100)}.items():
        m = (df.cluster == cl).to_numpy()
        xy[m] = rng.normal(centre, 3, size=(m.sum(), 2))
    p = cs.cluster_marker_profile(df, "cluster")
    first = cs.score_cluster_signatures(p, SIGS, parents={"B_x": "B", "B_y": "B"}, residual_label="R")
    nc = cs.cluster_neighbor_composition(xy, df.cluster, labels=first["top_label"].fillna("R").to_dict())
    r = cs.score_cluster_signatures(p, SIGS, parents={"B_x": "B", "B_y": "B"}, residual_label="R",
                                    neighbor_composition=nc, residual_alternatives=["A"])
    assert r.at["q1", "label"] == "A"
    assert r.at["q2", "label"] == "R"
    assert np.allclose(nc.sum(axis=1), 1.0)

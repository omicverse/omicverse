"""Synthetic tests for write_cluster_labels / audit_cluster_labels."""
import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

import pandas as pd

_P = Path(__file__).resolve().parents[2] / "omicverse" / "space" / "_cluster_labels.py"


def _load():
    try:
        from omicverse.space import _cluster_labels as m
        return m
    except Exception:  # light-weight import when the full package is unavailable
        pass
    reg = types.ModuleType("omicverse._registry")
    reg.register_function = lambda **k: (lambda f: f)
    sys.modules.setdefault("omicverse", types.ModuleType("omicverse"))
    sys.modules["omicverse._registry"] = reg
    sys.modules.setdefault("omicverse.space", types.ModuleType("omicverse.space"))
    spec = importlib.util.spec_from_file_location("omicverse.space._cl", _P)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cl = _load()


def _res(labels, reviews):
    """reviews: 'fixed' or 'open...' -> evidence columns under the default policy"""
    lz = [1.0 if r == "fixed" else -0.5 for r in reviews]
    return pd.DataFrame({"label": labels, "review": reviews, "lineage_z": lz,
                         "alt_lineage_z": [float("nan")] * len(labels)},
                        index=[str(i) for i in range(len(labels))])


def test_write_from_scorer_and_overrides():
    d = tempfile.mkdtemp()
    res = _res(["A", "B", "C"], ["fixed", "open: winner evidence z=-0.2 below 0", "fixed"])
    path = os.path.join(d, "ann.csv")
    tab, diff = cl.write_cluster_labels(
        res, path, columns=("cluster", "cell_class"),
        overrides={"1": ("D", "m4 z=2.1, m2 z=-0.3"), "2": ("A", "hand guess")})
    out = pd.read_csv(path)
    assert list(out.columns) == ["cluster", "cell_class"]
    assert out["cell_class"].tolist() == ["A", "D", "C"]       # fixed cluster 2 refused
    assert diff["cluster"].tolist() == ["1"]
    aud = pd.read_csv(os.path.join(d, "ann.audit.csv"))
    assert set(aud["source"]) >= {"scorer", "override", "override refused (fixed)"}
    assert cl.audit_cluster_labels(path, res, columns=("cluster", "cell_class")).empty


def test_audit_reports_hand_edits_missing_and_extra():
    d = tempfile.mkdtemp()
    res = _res(["A", "B", "C"], ["fixed", "fixed", "fixed"])
    path = os.path.join(d, "ann.csv")
    pd.DataFrame({"cluster": [0, 1, 7], "cell_class": ["A", "X", "C"]}).to_csv(path, index=False)
    bad = cl.audit_cluster_labels(path, res, columns=("cluster", "cell_class"))
    probs = set(zip(bad["cluster"].astype(str), bad["problem"]))
    assert ("1", "differs from scorer without a recorded override") in probs
    assert ("2", "missing") in probs and ("7", "not in scorer result") in probs


def test_grouped_layout_and_label_map():
    d = tempfile.mkdtemp()
    r1, r2 = _res(["a", "b"], ["fixed", "fixed"]), _res(["b"], ["open: x"])
    path = os.path.join(d, "multi.csv")
    tab, _ = cl.write_cluster_labels({"d1": r1, "d2": r2}, path,
                                     columns=("dataset", "cluster", "Annotation"),
                                     label_map={"a": "A cell"},
                                     overrides={("d2", "0"): ("a", "m1 z=1.9")})
    out = pd.read_csv(path)
    assert out.values.tolist() == [["d1", 0, "A cell"], ["d1", 1, "b"], ["d2", 0, "A cell"]]
    assert cl.audit_cluster_labels(path, {"d1": r1, "d2": r2},
                                   columns=("dataset", "cluster", "Annotation"),
                                   label_map={"a": "A cell"}).empty


def test_override_needs_evidence_and_allowed_labels():
    d = tempfile.mkdtemp()
    res = _res(["A"], ["open: x"])
    for bad in ({"0": ("B", "")}, {"0": "B"}):
        try:
            cl.write_cluster_labels(res, os.path.join(d, "a.csv"), overrides=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("override without evidence accepted")
    try:
        cl.write_cluster_labels(res, os.path.join(d, "b.csv"), allowed=["B"])
    except ValueError:
        pass
    else:
        raise AssertionError("label outside the allowed set written")


def test_forced_lineage_results_are_refused_and_reported():
    d = tempfile.mkdtemp()
    res = _res(["A", "B"], ["fixed", "fixed"])
    res["lineage_forced"] = [False, True]
    try:
        cl.write_cluster_labels(res, os.path.join(d, "x.csv"))
    except ValueError as e:
        assert "review_below=None" in str(e)
    else:
        raise AssertionError("forced lineage result written")
    path = os.path.join(d, "y.csv")
    pd.DataFrame({"cluster": [0, 1], "label": ["A", "B"]}).to_csv(path, index=False)
    bad = cl.audit_cluster_labels(path, res)
    assert any("forced" in p for p in bad["problem"])


def test_writer_recomputes_default_flags_ignoring_review_text():
    d = tempfile.mkdtemp()
    # the scorer was called with a loose policy: every cluster reads "open"
    res = pd.DataFrame({"label": ["A", "B", "C"],
                        "review": ["open: below 10.0"] * 3,
                        "lineage_z": [2.0, -0.3, 1.0], "alt_lineage_z": [0.5, float("nan"), 1.5]},
                       index=["0", "1", "2"])
    tab, diff = cl.write_cluster_labels(
        res, os.path.join(d, "a.csv"),
        overrides={"0": ("X", "ev"), "1": ("Y", "ev"), "2": ("Z", "ev")})
    assert tab["label"].tolist() == ["A", "Y", "Z"]   # 0 fixed by default; 1 weak; 2 contested
    # results without evidence columns: every override refused
    bare = pd.DataFrame({"label": ["A"], "review": ["open"]}, index=["0"])
    tab2, _ = cl.write_cluster_labels(bare, os.path.join(d, "b.csv"), overrides={"0": ("X", "ev")})
    assert tab2["label"].tolist() == ["A"]

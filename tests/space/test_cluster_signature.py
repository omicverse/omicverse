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
    r = cs.score_cluster_signatures(p, SIGS, parents={"B_x": "B", "B_y": "B"},
                                    children_support_parent=False)
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
                                    neighbor_composition=nc, residual_alternatives=["A"],
                                    residual_ratio=1.0, residual_to_testable=True)
    assert r.at["q1", "label"] == "A"
    assert r.at["q2", "label"] == "R"
    assert np.allclose(nc.sum(axis=1), 1.0)
    # cluster-level composition: shares exclude the cluster's own cells
    ncc = cs.cluster_neighbor_composition(xy, df.cluster)
    r2 = cs.score_cluster_signatures(p, SIGS, parents={"B_x": "B", "B_y": "B"}, residual_label="R",
                                     neighbor_composition=ncc, residual_alternatives=["A"],
                                     residual_to_testable=True)
    assert r2.at["q1", "label"] == "A"
    assert r2.at["q2", "label"] == "R"
    # default: A has positive markers the quiet cluster lacks -> stays residual
    r3 = cs.score_cluster_signatures(p, SIGS, parents={"B_x": "B", "B_y": "B"}, residual_label="R",
                                     neighbor_composition=ncc, residual_alternatives=["A"])
    assert r3.at["q1", "label"] == "R"


# the original z-score scorer (before rel / shrink / contrast defaults)
ZKW = dict(stat="z", shrink=0, contrast=0, presence_from="zero", presence=0.3,
           weighted_presence=False, neg="same", weak_fraction=0, defining_check=None)


def test_min_weight_shrinks_weakly_supported_labels():
    p = cs.cluster_marker_profile(_panel(), "cluster")
    # "W" rests on m1 with a weight far below one defining marker
    sigs = {"A": {"m1": 1.0, "m2": -0.5}, "W": {"m1": 0.05}}
    plain = cs.score_cluster_signatures(p, sigs, min_weight=0, norm="l1", **ZKW)
    shrunk = cs.score_cluster_signatures(p, sigs, min_weight=1.0, norm="l1", **ZKW)
    z = p.at["a", "z:m1"]
    assert np.isclose(plain.at["a", "score:W"], z)                # weighted mean: full strength
    assert np.isclose(shrunk.at["a", "score:W"], 0.05 * z)        # shrunk to its evidence
    assert shrunk.at["a", "label"] == "A"


def test_defining_markers_majority_and_candidates():
    p = cs.cluster_marker_profile(_panel(), "cluster")
    sigs = {"A": {"m1": 1.0}, "N": {"m1": 1.0, "absent1": 1.0, "absent2": 1.0}}
    r = cs.score_cluster_signatures(p, sigs, defining={"A": ["m1"], "N": ["m1", "absent1", "absent2"]})
    assert "N" not in r.at["a", "eligible"].split(",")      # panel measures 1 of 3 defining markers
    assert r.at["a", "candidates"].startswith("A ") and "m1 z=" in r.at["a", "candidates"]


def test_children_support_parent_and_lineage_override():
    p = cs.cluster_marker_profile(_panel(), "cluster")
    sigs = {"A": {"m1": 1.0, "m2": -0.5}, "B": {"m2": 0.2, "m1": -0.1},
            "B_x": {"m3": 1.0, "m2": 0.5}, "B_y": {"m4": 1.0, "m2": 0.5}}
    par = {"B_x": "B", "B_y": "B"}
    r = cs.score_cluster_signatures(p, sigs, parents=par, children_support_parent=True, **ZKW)
    assert r.at["b1", "score"] >= r.at["b1", "score:B_x"] - 1e-9
    # a reviewed lineage: child labels are lifted to their parent and re-split
    r2 = cs.score_cluster_signatures(p, sigs, parents=par, polar=["B"],
                                     lineage={"b1": "B_y", "q1": "zzz"})
    assert r2.at["b1", "top_label"] == "B" and r2.at["b1", "label"] == "B_x"
    assert r2.at["q1", "label"] == "zzz"


def test_opposed_poles_and_promotable_children():
    p = cs.cluster_marker_profile(_panel(), "cluster")
    sigs = dict(SIGS)
    sigs["B_same"] = {"m2": 1.0, "m1": -2.0}        # nothing of its own beyond the parent
    par = {"B_x": "B", "B_y": "B", "B_same": "B"}
    r = cs.score_cluster_signatures(p, sigs, parents=par, polar={"B": ["B_x", "B_y"]})
    assert {r.at["b1", "label"], r.at["b2", "label"]} == {"B_x", "B_y"}
    assert "B_same" not in set(r["label"])


def test_defining_check_prefers_standout_defining_marker():
    p = cs.cluster_marker_profile(_panel(), "cluster")
    # "X" wins on a weak, broad signature; "B" has its defining marker standing out
    sigs = {"X": {"m1": 0.3, "m3": 0.3, "m4": 0.3}, "B": {"m2": 1.0}}
    d = {"X": ["m1"], "B": ["m2"]}
    plain = cs.score_cluster_signatures(p, sigs, defining=d, norm="l2")
    chk = cs.score_cluster_signatures(p, sigs, defining=d, norm="l2", defining_check=(0.5, 1.0))
    assert chk.at["b1", "label"] == "B"
    assert "defining-marker check" in chk.at["b1", "reason"] or plain.at["b1", "label"] == "B"


def test_promote_min_requires_own_evidence():
    p = cs.cluster_marker_profile(_panel(), "cluster")
    sigs = dict(SIGS)
    sigs["B_tiny"] = {"m2": 1.0, "m3": 0.05, "m4": 0.05}     # parent's marker + trace weights
    par = {"B_x": "B", "B_y": "B", "B_tiny": "B"}
    r = cs.score_cluster_signatures(p, sigs, parents=par, promote_min=0.1)
    assert "B_tiny" not in set(r["label"])


def test_pole_axis_split():
    p = cs.cluster_marker_profile(_panel(), "cluster")
    sigs = dict(SIGS)
    sigs["B_z"] = {"m2": 1.0, "m3": 0.6, "m4": 0.6}         # a broad child that is not a pole
    par = {"B_x": "B", "B_y": "B", "B_z": "B"}
    r = cs.score_cluster_signatures(p, sigs, parents=par, polar={"B": ["B_x", "B_y"]})
    assert r.at["b1", "label"] == "B_x" and r.at["b2", "label"] == "B_y"
    assert "by poles" in r.at["b1", "reason"]


def test_review_flags_and_enforced_lineage():
    p = cs.cluster_marker_profile(_panel(), "cluster")
    d = {"A": ["m1"], "B": ["m2"], "B_x": ["m3"], "B_y": ["m4"]}
    par = {"B_x": "B", "B_y": "B"}
    r = cs.score_cluster_signatures(p, SIGS, parents=par, defining=d)
    assert r.at["a", "review"] == "fixed" and r.at["a", "lineage_z"] > 0
    # a change of a fixed cluster's lineage is ignored and explained
    r2 = cs.score_cluster_signatures(p, SIGS, parents=par, defining=d, lineage={"a": "B"})
    assert r2.at["a", "label"] == "A" and "ignored" in r2.at["a", "reason"]
    # without enforcement it is applied
    # the policy cannot be loosened: None and larger values are ignored
    for rb in (None, 10.0):
        r3 = cs.score_cluster_signatures(p, SIGS, parents=par, defining=d, lineage={"a": "B"},
                                         review_below=rb)
        assert r3.at["a", "top_label"] == "A" and r3.at["a", "review"] == "fixed"
        assert "ignored" in r3.attrs["review_note"]
    # a stricter (smaller) value is honoured: fewer open clusters
    r4 = cs.score_cluster_signatures(p, SIGS, parents=par, defining=d, review_below=-5.0)
    r0 = cs.score_cluster_signatures(p, SIGS, parents=par, defining=d)
    n_open = lambda r: sum(str(v).startswith("open") for v in r["review"])
    assert n_open(r4) <= n_open(r0) and r4.attrs["review_below"] == -5.0


def test_review_states_open_resolved_fixed():
    st = getattr(cs, "_review_state", None)
    if st is None:
        from omicverse.space._cluster_signature import _review_state as st
    assert st(1.0, 0.5, 0.0, "A", "B", 1.0, 0.2, (0.5, 1.5)) == "fixed"
    # another candidate leads on one defining marker only: resolved, winner kept
    s = st(0.8, 1.4, 0.0, "A", "B", 0.9, 0.5, (0.5, 1.5))
    assert s.startswith("resolved: keep A") and "0.90" in s and "z=1.4" in s
    # weak winner, alternative above the mean: open
    assert st(-0.4, 0.6, 0.0, "A", "B", 0.3, 0.2, None).startswith("open")
    # weak winner, no alternative reaches the mean: resolved
    assert st(-0.4, -0.1, 0.0, "A", "B", 0.3, 0.2, None, "R").startswith("resolved: keep A")
    assert st(-0.4, -0.1, 0.0, "A", "B", 0.3, 0.2, None).startswith("open")
    assert st(float("nan"), float("nan"), 0.0, "A", None, 0.3, float("nan"), None, "R").startswith("resolved")


def test_lineage_ignored_on_fixed_or_resolved_clusters():
    p = cs.cluster_marker_profile(_panel(), "cluster")
    d = {"A": ["m1"], "B": ["m2"], "B_x": ["m3"], "B_y": ["m4"]}
    par = {"B_x": "B", "B_y": "B"}
    r = cs.score_cluster_signatures(p, SIGS, parents=par, defining=d, residual_label="R")
    # no signature present: the label of exclusion is a default, left open
    assert str(r.at["q1", "review"]).startswith("open") and r.at["q1", "label"] == "R"
    resolved = [c for c in r.index if str(r.at[c, "review"]).startswith("resolved")]
    fixed = [c for c in r.index if r.at[c, "review"] == "fixed"]
    c0 = (resolved or fixed)[0]
    other = "B" if r.at[c0, "top_label"] == "A" else "A"
    r2 = cs.score_cluster_signatures(p, SIGS, parents=par, defining=d, residual_label="R",
                                     lineage={c0: other, "q1": "A"})
    assert r2.at[c0, "top_label"] == r.at[c0, "top_label"] and "ignored" in r2.at[c0, "reason"]
    assert r2.at["q1", "top_label"] == "A"


def test_residual_label_signature_does_not_compete():
    p = cs.cluster_marker_profile(_panel(), "cluster")
    sigs = dict(SIGS, R={"m1": 0.05})            # weak signature of the exclusion label
    r = cs.score_cluster_signatures(p, sigs, residual_label="R")
    assert "R" not in set(r["top_label"].dropna())
    assert r.at["q1", "label"] == "R" and r.at["a", "label"] == "A"
    assert "score:R" in r.columns


def test_weighted_presence_ignores_weak_markers():
    # B carries a weak secondary marker m1; cluster 'a' (m1 high) must not
    # become eligible for B through it when presence is weighted
    p = cs.cluster_marker_profile(_panel(), "cluster")
    sigs = {"A": {"m1": 1.0}, "B": {"m2": 1.0, "m1": 0.1}}
    r = cs.score_cluster_signatures(p, sigs)
    assert "B" not in r.at["a", "eligible"].split(",")
    r0 = cs.score_cluster_signatures(p, sigs, weighted_presence=False)
    assert "B" in r0.at["a", "eligible"].split(",")


def test_quiet_clusters_score_near_zero_with_rel():
    p = cs.cluster_marker_profile(_panel(), "cluster")
    r = cs.score_cluster_signatures(p, SIGS, stat="rel", shrink=0, contrast=0)
    assert abs(r.at["q1", "score:A"]) < 0.1 and r.at["a", "score:A"] > 0.8


def _make(spec, markers, n=150, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for cl, mk in spec.items():
        for _ in range(n):
            r = {m: max(0.0, rng.normal(mk.get(m, 0.02), 0.04)) for m in markers}
            r["cluster"] = cl
            rows.append(r)
    return pd.DataFrame(rows)


def test_single_marker_label_does_not_take_coherent_programme():
    # 'my' expresses a broad programme (a, b, c) plus some f and c4 (shared
    # background); a label resting on f / c4 must not take it
    mk = ["a", "b", "c", "f", "c4", "t"]
    df = _make({"my": dict(a=0.8, b=0.7, c=0.6, f=0.25, c4=0.4),
                "tr": dict(f=0.6, c4=0.7, t=0.7), "t4": dict(c4=0.8, t=0.8),
                "q": {}}, mk)
    sigs = {"M": {"a": 1.0, "b": 0.9, "c": 0.8, "c4": 0.03},
            "R": {"f": 1.16, "c4": 0.6, "t": 0.23},
            "T4": {"c4": 0.73, "t": 0.22, "f": -0.05}}
    r = cs.score_cluster_signatures(cs.cluster_marker_profile(df, "cluster"), sigs, residual_label="X")
    assert r.at["my", "label"] == "M"
    assert r.at["tr", "label"] == "R" and r.at["t4", "label"] == "T4"
    assert r.at["q", "label"] == "X"


def test_contrast_split_promotes_every_cluster_above_the_gap():
    mk = ["p", "k"]
    df = _make({"c1": dict(p=0.8, k=0.8), "c2": dict(p=0.8, k=0.75), "c3": dict(p=0.8),
                "c4": dict(p=0.75)}, mk)
    sigs = {"P": {"p": 1.0}, "K": {"k": 1.0, "p": 0.5}}
    r = cs.score_cluster_signatures(cs.cluster_marker_profile(df, "cluster"), sigs, parents={"K": "P"})
    assert set(r.index[r["label"] == "K"]) == {"c1", "c2"}
    assert set(r.index[r["label"] == "P"]) == {"c3", "c4"}


def test_weak_signature_does_not_compete():
    mk = ["a", "b", "c", "z"]
    df = _make({"x": dict(a=0.8, b=0.7, z=0.9), "y": dict(c=0.8)}, mk)
    sigs = {"A": {"a": 1.0, "b": 1.0}, "C": {"c": 1.0, "b": 0.5}, "Z": {"z": 0.05}}
    r = cs.score_cluster_signatures(cs.cluster_marker_profile(df, "cluster"), sigs)
    assert "Z" in r.attrs["weak"] and r.at["x", "label"] == "A"


def test_label_without_measured_defining_marker_needs_a_standout():
    p = cs.cluster_marker_profile(_panel(), "cluster")
    sigs = {"A": {"m1": 1.0}, "U": {"m2": 0.3, "m1": 0.1}}      # U's defining marker is off-panel
    d = {"A": ["m1"], "U": ["offpanel"]}
    r = cs.score_cluster_signatures(p, sigs, defining=d, stat="z", shrink=0, contrast=0,
                                    defining_check=(0.5, 1.0))
    assert "U" not in r.at["q1", "eligible"].split(",")        # quiet cluster: no stand-out
    assert "U" in r.at["b1", "eligible"].split(",")            # m2 stands out in b1


def test_one_configuration_both_regimes():
    """Defaults, two regimes.  (1) A label of exclusion: clusters carrying only
    a broadly expressed marker (at background) fall to it.  (2) No label of
    exclusion: a dim cluster takes the label whose defining marker it shows."""
    mk = ["lin1", "lin2", "broad", "x"]
    # "broad" sits at its typical (median-cluster) level in most clusters — its
    # background — and is high only in s1, which also carries lin1
    spec = {"t1": dict(broad=0.3), "t2": dict(broad=0.3), "t3": dict(broad=0.3),
            "t4": dict(broad=0.3), "s1": dict(lin1=0.8, broad=0.8),
            "m1": dict(lin2=0.8, broad=0.3), "m2": dict(lin2=0.7, broad=0.3)}
    df = _make(spec, mk)
    p = cs.cluster_marker_profile(df, "cluster")
    sigs = {"S": {"lin1": 1.0, "broad": 0.5}, "M": {"lin2": 1.0}, "T": {"x": 0.05}}
    d = {"S": ["lin1"], "M": ["lin2"], "T": []}
    r = cs.score_cluster_signatures(p, sigs, defining=d, residual_label="T")
    assert {r.at[c, "label"] for c in ("t1", "t2", "t3", "t4")} == {"T"}
    assert r.at["s1", "label"] == "S" and r.at["m1", "label"] == "M"
    # regime 2: no exclusion label; a dim lin1 cluster still takes S
    spec2 = {"a": dict(lin1=0.9), "dim": dict(lin1=0.3), "b": dict(lin2=0.9), "c": dict(lin2=0.8)}
    p2 = cs.cluster_marker_profile(_make(spec2, mk), "cluster")
    r2 = cs.score_cluster_signatures(p2, {"S": {"lin1": 1.0}, "M": {"lin2": 1.0}},
                                     defining={"S": ["lin1"], "M": ["lin2"]})
    assert r2.at["dim", "label"] == "S"


def test_lineage_marker_beats_moderate_spillover_programme():
    # 't8s': a T-like cluster whose defining marker is at 2/3 of the bright
    # T cluster, plus contact spill-over of a broad myeloid-like programme at
    # half its source level; six genuine myeloid clusters
    mk = ["d8", "d3", "a", "b", "c", "e", "f", "d4"]
    src = dict(a=0.9, b=0.8, c=0.7, e=0.6, f=0.5)
    spec = {"t8a": dict(d8=0.9, d3=0.6), "t8s": dict(d8=0.6, d3=0.4, **{k: 0.5 * v for k, v in src.items()}),
            "t4": dict(d4=0.8, d3=0.7), "q": {}}
    for i in range(6):
        spec[f"m{i}"] = dict(src, d4=0.25)
    sigs = {"M": {"a": 1.0, "b": 0.9, "c": 0.85, "e": 0.6, "f": 0.5, "d4": 0.05, "d3": -0.1},
            "T8": {"d8": 1.0, "d3": 0.2, "d4": -0.1},
            "T4": {"d4": 0.75, "d3": 0.22, "d8": -0.1}}
    p = cs.cluster_marker_profile(_make(spec, mk), "cluster")
    r = cs.score_cluster_signatures(p, sigs, residual_label="X")
    assert r.at["t8s", "label"] == "T8"
    assert r.at["t4", "label"] == "T4"
    assert r.at["m0", "label"] == "M" and r.at["q", "label"] == "X"


def test_single_excellent_marker_label_is_testable():
    mk = ["v", "a", "b", "c"]
    df = _make({"en": dict(v=0.8), "x": dict(a=0.8, b=0.7, c=0.6), "q": {}}, mk)
    sigs = {"EN": {"v": 1.0}, "X": {"a": 1.0, "b": 0.9, "c": 0.8}, "Y": {"b": 0.6, "c": 1.0},
            "W": {"a": 0.02}}
    r = cs.score_cluster_signatures(cs.cluster_marker_profile(df, "cluster"), sigs, residual_label="O")
    assert "EN" not in r.attrs["weak"] and "W" in r.attrs["weak"]
    assert r.at["en", "label"] == "EN" and r.at["q", "label"] == "O"


def test_lineage_markers_exclude_a_state_label_without_its_lineage():
    # 'kx' carries a state marker (k) and a moderate tumour-like marker (h)
    # but its lineage marker t is below the background every cluster carries
    mk = ["t", "k", "h"]
    spec = {"t1": dict(t=0.8), "q1": dict(t=0.2), "q2": dict(t=0.2), "q3": dict(t=0.2),
            "kx": dict(k=0.8, h=0.3, t=0.0), "h1": dict(h=0.8, t=0.2)}
    p = cs.cluster_marker_profile(_make(spec, mk), "cluster")
    sigs = {"TK": {"k": 1.0, "t": 0.15}, "H": {"h": 1.0}, "T": {"t": 1.0}}
    d = {"TK": ["k"], "H": ["h"], "T": ["t"]}
    plain = cs.score_cluster_signatures(p, sigs, defining=d)
    assert plain.at["kx", "label"] == "TK"
    r = cs.score_cluster_signatures(p, sigs, defining=d, lineage_markers={"TK": ["t"]})
    assert r.at["kx", "label"] == "H"
    assert r.at["t1", "label"] == "T"


def test_defining_check_weighs_marker_contrast():
    # marker s is carried by every cluster near its maximum (little contrast
    # over background); m is a clean marker.  Both stand out in x, s more so;
    # x's whole-signature winner X has its own defining marker a below 0.5
    mk = ["a", "b", "c", "s", "m"]
    base = dict(s=0.5)
    spec = {"x": dict(a=0.3, b=0.8, c=0.8, s=0.65, m=0.3), "y": dict(base, m=0.2, a=0.3),
            "u": dict(base, a=0.6), "v": dict(base, b=0.3), "w": dict(base, c=0.3), "z": dict(base)}
    p = cs.cluster_marker_profile(_make(spec, mk), "cluster")
    assert p.at["x", "z:s"] > p.at["x", "z:m"] > 1.5
    sigs = {"X": {"a": 1.0, "b": 1.0, "c": 1.0}, "P": {"s": 1.0}, "M": {"m": 1.0}}
    d = {"X": ["a"], "P": ["s"], "M": ["m"]}
    r = cs.score_cluster_signatures(p, sigs, defining=d)
    assert r.at["x", "label"] == "M" and "defining-marker check" in r.at["x", "reason"]


def test_increment_split_axis_follows_what_a_child_adds():
    # two members of lineage P; the bright one is high on g (child C1's own
    # marker) and somewhat on h; C2 lists g and h weakly on top of P's marker
    mk = ["p", "g", "h", "o"]
    spec = {"x": dict(p=0.9, g=0.8, h=0.6), "y": dict(p=0.3, g=0.05, h=0.05),
            "o1": dict(o=0.8), "o2": dict(o=0.6), "o3": {}}
    p = cs.cluster_marker_profile(_make(spec, mk), "cluster")
    sigs = {"P": {"p": 1.0}, "C1": {"p": 1.0, "g": 0.24}, "C2": {"p": 1.0, "g": 0.08, "h": 0.08},
            "O": {"o": 1.0}}
    par = {"C1": "P", "C2": "P"}
    d = {"P": ["p"], "C1": ["p"], "C2": ["p"], "O": ["o"]}
    old = cs.score_cluster_signatures(p, sigs, parents=par, defining=d, split_axis="signature")
    new = cs.score_cluster_signatures(p, sigs, parents=par, defining=d)
    assert new.at["x", "label"] == "C1" and new.at["y", "label"] == "P"
    assert old.at["x", "label"] in ("C1", "C2")


def test_single_marker_conflict_switches_to_specific_coherent_evidence():
    # x: the whole-signature winner MO leads through a child whose defining
    # marker (sh) is shared with another lineage (N); MA's child marker r is
    # specific and stands out more -> MA takes the cluster
    mk = ["p", "sh", "q", "r", "o"]
    spec = {"x": dict(p=0.75, sh=0.9, q=0.3, r=0.85), "n1": dict(sh=0.6), "ma1": dict(q=0.8, r=0.3),
            "mo1": dict(p=0.8), "o1": dict(o=0.8), "o2": dict(o=0.5), "e": {}}
    p = cs.cluster_marker_profile(_make(spec, mk), "cluster")
    sigs = {"MO": {"p": 1.0}, "MO_c": {"p": 1.0, "sh": 1.0}, "MA": {"q": 1.0, "r": 0.3},
            "MA_c": {"r": 1.0, "q": 0.3}, "N": {"sh": 1.0}, "O": {"o": 1.0}}
    par = {"MO_c": "MO", "MA_c": "MA"}
    d = {"MO": ["p"], "MO_c": ["sh"], "MA": ["q"], "MA_c": ["r"], "N": ["sh"], "O": ["o"]}
    off = cs.score_cluster_signatures(p, sigs, parents=par, defining=d, switch_rule="none")
    assert off.at["x", "top_label"] == "MO" and str(off.at["x", "review"]).startswith("resolved")
    on = cs.score_cluster_signatures(p, sigs, parents=par, defining=d)
    assert on.at["x", "top_label"] == "MA" and "single-marker conflict" in on.at["x", "reason"]


def test_single_marker_conflict_keeps_winner_when_other_is_incoherent():
    # S stands out on v, but its other defining marker w is below the mean in x
    mk = ["p", "sh", "v", "w", "o"]
    spec = {"x": dict(p=0.75, sh=0.9, v=0.85), "n1": dict(sh=0.6), "s1": dict(v=0.5, w=0.8),
            "mo1": dict(p=0.8), "o1": dict(o=0.8), "o2": dict(o=0.5), "e": dict(w=0.3)}
    p = cs.cluster_marker_profile(_make(spec, mk), "cluster")
    sigs = {"MO": {"p": 1.0}, "MO_c": {"p": 1.0, "sh": 1.0}, "S": {"v": 1.0, "w": 1.0},
            "N": {"sh": 1.0}, "O": {"o": 1.0}}
    d = {"MO": ["p"], "MO_c": ["sh"], "S": ["v", "w"], "N": ["sh"], "O": ["o"]}
    r = cs.score_cluster_signatures(p, sigs, parents={"MO_c": "MO"}, defining=d)
    assert r.at["x", "alt_label"] == "S" and r.at["x", "alt_lineage_z"] > r.at["x", "lineage_z"]
    assert r.at["x", "top_label"] == "MO" and str(r.at["x", "review"]).startswith("resolved: keep MO")


def test_label_with_most_defining_markers_unmeasured_is_not_a_candidate():
    # F is defined by f1 and f2; the panel measures only f2, which x carries
    # alongside its own lineage marker (as with an exclusion label in play)
    mk = ["f2", "g", "o"]
    spec = {"x": dict(f2=0.8, g=0.7), "g1": dict(g=0.8), "o1": dict(o=0.8), "q": {}}
    p = cs.cluster_marker_profile(_make(spec, mk), "cluster")
    sigs = {"F": {"f1": 1.0, "f2": 0.8, "g": 0.4}, "G": {"g": 1.0}, "O": {"o": 1.0}}
    d = {"F": ["f1", "f2"], "G": ["g"], "O": ["o"]}
    r = cs.score_cluster_signatures(p, sigs, defining=d, residual_label="R")
    assert r.at["x", "label"] == "G"
    loose = cs.score_cluster_signatures(p, sigs, defining=d, residual_label="R", identifiable="any")
    assert loose.at["x", "label"] == "F"

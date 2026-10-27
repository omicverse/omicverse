"""Tests for ov.genetics trait_determinants / effect_modifier_scan / transfer_trait_predict."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from omicverse.genetics._determinants import (
    effect_modifier_scan,
    trait_determinants,
    transfer_trait_predict,
)


def _simulate(seed=0, n=4000, p=40):
    """Presence/absence panel with three kinds of trait-associated features.

    f0-f2  main-effect determinants
    f3-f4  determinants acting only when the background locus f5 is present
    f5     background (permissive) locus with no effect of its own
    f6-f7  passengers: co-carried with f0 (same mobile element), no effect
    """
    rng = np.random.default_rng(seed)
    X = (rng.random((n, p)) < 0.3).astype(float)
    X[:, 5] = (rng.random(n) < 0.6).astype(float)
    carrier = rng.random(n) < 0.35
    X[:, 0] = carrier.astype(float)
    for j in (6, 7):
        X[:, j] = np.where(carrier, rng.random(n) < 0.9, rng.random(n) < 0.05).astype(float)
    eta = (-2.0 + 2.2 * X[:, 0] + 2.0 * X[:, 1] + 1.8 * X[:, 2]
           + 3.0 * X[:, 3] * X[:, 5] + 3.0 * X[:, 4] * X[:, 5])
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    cols = [f"f{j}" for j in range(p)]
    return pd.DataFrame(X, columns=cols, index=[f"s{i}" for i in range(n)]), y


def test_modifier_scan_finds_background_locus():
    X, y = _simulate()
    scan = effect_modifier_scan(X, y, shortlist=10)
    assert scan.loc[0, "modifier"] == "f5"
    assert scan.loc[0, "p_value"] < 1e-6
    assert {"lr_stat", "df", "frac_present"} <= set(scan.columns)


def test_stepwise_selection_sizes_the_shortlist():
    X, y = _simulate()
    res = trait_determinants(X, y, shortlist=10)
    det = set(res.loc[res.is_determinant, "feature"])
    assert det == {"f0", "f1", "f2", "f3", "f4"}          # passengers f6/f7 and modifier f5 out
    assert res.attrs["n_determinants"] == 5
    assert list(res.head(5)["is_determinant"]) == [True] * 5
    path = res.attrs["selection_path"]
    assert not path["accepted"].iloc[-1]                  # stopped at a rejected candidate
    res_b = trait_determinants(X, y, shortlist=10, select="bonferroni")
    assert {"f0", "f1", "f2", "f3", "f4"} <= set(res_b.loc[res_b.is_determinant, "feature"])
    res_n = trait_determinants(X, y, shortlist=10, select=None)
    assert res_n.attrs["n_determinants"] is None


def test_trait_determinants_top_set_excludes_passengers_and_modifier():
    X, y = _simulate()
    res = trait_determinants(X, pd.Series(y, index=X.index), shortlist=10)
    assert res.attrs["modifier"] == "f5"
    top5 = set(res.head(5)["feature"])
    assert top5 == {"f0", "f1", "f2", "f3", "f4"}
    row = res.set_index("feature")
    assert row.loc["f3", "acts_in"] == "modifier_present"
    assert bool(row.loc["f5", "is_modifier"])
    assert list(res["rank"][:3]) == [1, 2, 3]


def test_trait_determinants_without_modifier_and_string_labels():
    X, y = _simulate(seed=1)
    lab = np.where(y == 1, "R", "S")
    lab = pd.Series(lab, index=X.index, dtype=object)
    lab.iloc[:50] = np.nan                      # unlabelled samples are dropped
    res = trait_determinants(X, lab, modifier=None)
    assert res.attrs["modifier"] is None
    assert (res["acts_in"] == "all").all()
    with pytest.raises(ValueError):
        trait_determinants(X, y, modifier=None, direction="up")


def test_transfer_metadata_matching_and_consensus():
    # sources a1 (single agent) and b2 (combination): a determinant f0 acts in a1 only
    X, _ = _simulate(seed=4)
    rng = np.random.default_rng(5)
    base = -2.0 + 2.5 * X["f1"] + 2.5 * X["f2"]
    def draw(eta):
        return (rng.random(len(eta)) < 1 / (1 + np.exp(-eta))).astype(int)
    Y = pd.DataFrame({"a1": draw(base + 6.0 * X["f0"]), "b2": draw(base)}, index=X.index)
    meta = pd.DataFrame({"drug": ["a1", "b2", "t_single", "t_combo", "t_new"],
                         "cls": ["A"] * 5,
                         "smiles": ["CC", "CC.OO", "CC", "CC.NN", "CC.NN.SS"]})
    r1 = transfer_trait_predict(X, Y, "t_combo", groups=meta)       # trait names in a column
    assert r1["sources"] == ["b2"] and r1["decision"] == "pooled"
    r2 = transfer_trait_predict(X, Y, "t_new", groups=meta)
    assert set(r2["sources"]) == {"a1", "b2"} and r2["decision"] == "consensus"
    carriers = (X["f0"] == 1) & (X["f1"] == 0) & (X["f2"] == 0)
    assert r2["call"][carriers].mean() < 0.2                       # f0 alone not trusted
    assert r2["discordant_features"].index[0] == "f0"
    assert r2["per_source_proba"].shape[1] == 2
    r3 = transfer_trait_predict(X, Y, "t_new", groups=meta, decision="pooled")
    assert r3["call"][carriers].mean() > r2["call"][carriers].mean()


def test_transfer_trait_predict_group_pooling():
    X, y = _simulate(seed=2)
    rng = np.random.default_rng(3)
    flip = lambda v: np.where(rng.random(len(v)) < 0.05, 1 - v, v)
    Y = pd.DataFrame({"a1": flip(y), "a2": flip(y), "b1": rng.integers(0, 2, len(y))},
                     index=X.index)
    groups = {"a1": "A", "a2": "A", "a3": "A", "b1": "B"}
    res = transfer_trait_predict(X, Y, "a3", groups=groups)
    assert res["sources"] == ["a1", "a2"]
    assert res["proba"].between(0, 1).all()
    assert set(res["call"].unique()) <= {0, 1}
    val = res["validation"]
    assert list(val["held_out"]) == ["a1", "a2"]
    assert (val["auroc"] > 0.75).all()
    with pytest.raises(ValueError):
        transfer_trait_predict(X, Y, "zz", groups={"zz": "C", **groups})
    with pytest.raises(ValueError, match="has no group"):
        transfer_trait_predict(X, Y, "a4", groups=groups)
    res2 = transfer_trait_predict(X, Y, "a3", groups=pd.Series(groups), validate=False)
    assert res2["sources"] == ["a1", "a2"] and res2["validation"] is None

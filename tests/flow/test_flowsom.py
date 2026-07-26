"""FlowSOM — one implementation, two callers.

`ov.single.ev.flowsom` predates `ov.flow` and carried its own SOM. The maths is
not EV-specific, so it was promoted rather than duplicated. These tests exist
mainly to pin that the EV entry point is UNCHANGED by that move — it had no
tests of its own before, anywhere in the repo.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import omicverse.flow as fl


def panel(n=1500, seed=0):
    """Three well-separated phenotypes over four markers."""
    rng = np.random.default_rng(seed)
    centres = np.array([[8.0, 1.0, 1.0, 1.0],
                        [1.0, 8.0, 1.0, 1.0],
                        [1.0, 1.0, 8.0, 8.0]])
    which = rng.integers(0, 3, n)
    X = (centres[which] + rng.normal(0, 0.6, (n, 4))).astype(np.float32)
    a = ad.AnnData(X, var=pd.DataFrame(index=["CD3", "CD19", "CD14", "CD16"]))
    a.obs["truth"] = pd.Categorical([str(w) for w in which])
    return a


def test_flowsom_recovers_separated_populations():
    a = panel()
    fl.flowsom(a, n_clusters=3, grid=(6, 6), n_epochs=15, random_state=0)
    # Each true population should map to essentially one cluster.
    ct = pd.crosstab(a.obs["truth"], a.obs["flowsom"])
    purity = (ct.max(axis=1) / ct.sum(axis=1)).min()
    assert purity > 0.9, f"worst population purity {purity:.2f}\n{ct}"


def test_results_are_deterministic_for_a_seed():
    a, b = panel(), panel()
    fl.flowsom(a, n_clusters=3, grid=(5, 5), n_epochs=10, random_state=7)
    fl.flowsom(b, n_clusters=3, grid=(5, 5), n_epochs=10, random_state=7)
    assert list(a.obs["flowsom"]) == list(b.obs["flowsom"])


def test_markers_argument_restricts_the_panel():
    """Including scatter lets it dominate and you get shape clusters, not
    phenotypes — so restricting has to actually restrict."""
    a = panel()
    fl.flowsom(a, n_clusters=2, grid=(4, 4), markers=["CD3", "CD19"], random_state=0)
    assert a.uns["flow"]["flowsom"]["markers"] == ["CD3", "CD19"]
    assert len(a.obs["flowsom"]) == a.n_obs


def test_too_many_clusters_for_the_grid_is_refused():
    a = panel()
    with pytest.raises(ValueError, match="cannot exceed the SOM node count"):
        fl.flowsom(a, n_clusters=50, grid=(3, 3))


def test_som_metacluster_is_anndata_free():
    """The shared core takes a matrix. That is what lets two namespaces write
    their results into two different places without a second implementation."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (200, 3)), rng.normal(8, 1, (200, 3))])
    node_of_obs, node_meta, codes = fl.som_metacluster(X, n_clusters=2, grid=(4, 4),
                                                       n_epochs=10, random_state=0)
    assert node_of_obs.shape == (400,)
    assert codes.shape == (16, 3)
    labels = node_meta[node_of_obs]
    assert len(set(labels[:200])) == 1 and len(set(labels[200:])) == 1
    assert labels[0] != labels[-1]


# ── the EV entry point must be unchanged by the move ────────────────────────

def ev_panel(n=800, seed=0):
    rng = np.random.default_rng(seed)
    centres = np.array([[8.0, 1.0, 1.0], [1.0, 8.0, 1.0], [1.0, 1.0, 8.0]])
    which = rng.integers(0, 3, n)
    X = (centres[which] + rng.normal(0, 0.5, (n, 3))).astype(np.float32)
    a = ad.AnnData(X, var=pd.DataFrame(index=["CD9", "CD63", "CD81"]))
    a.obs["truth"] = pd.Categorical([str(w) for w in which])
    a.uns["ev"] = {"value_type": "intensity", "platform": "test"}
    return a


def test_ev_flowsom_still_writes_its_own_uns_and_obs():
    from omicverse.single.ev._cluster import flowsom as ev_flowsom

    a = ev_panel()
    ev_flowsom(a, n_clusters=3, grid=(5, 5), n_epochs=12, layer=None, random_state=0)
    assert "flowsom" in a.obs and "flowsom_som" in a.obs
    # EV keeps its own namespace — the move must not have relocated it.
    assert "flowsom" in a.uns["ev"]
    assert a.uns["ev"]["flowsom"]["grid"] == (5, 5)
    assert "flow" not in a.uns


def test_ev_flowsom_and_flow_flowsom_agree_on_the_same_input():
    """Same seed, same grid, same data -> same partition. If these ever diverge
    there are two implementations again, which is exactly what the promotion
    was for."""
    from omicverse.single.ev._cluster import flowsom as ev_flowsom

    a, b = ev_panel(), ev_panel()
    ev_flowsom(a, n_clusters=3, grid=(5, 5), n_epochs=12, layer=None, random_state=3)
    fl.flowsom(b, n_clusters=3, grid=(5, 5), n_epochs=12, layer=None, random_state=3)
    assert list(a.obs["flowsom"]) == list(b.obs["flowsom"])


def test_ev_flowsom_still_recovers_populations():
    from omicverse.single.ev._cluster import flowsom as ev_flowsom

    a = ev_panel()
    ev_flowsom(a, n_clusters=3, grid=(5, 5), n_epochs=15, layer=None, random_state=0)
    ct = pd.crosstab(a.obs["truth"], a.obs["flowsom"])
    assert (ct.max(axis=1) / ct.sum(axis=1)).min() > 0.9

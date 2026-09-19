from __future__ import annotations

import warnings
import sys
import types

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from omicverse import space
from omicverse.space import _svg as svg_module
from scipy import sparse
from statsmodels.stats.multitest import multipletests

from omicverse.space._svg import _select_significant_svg_names, svg, spatial_neighbors, spatial_autocorr


def test_qvalue_selection_never_promotes_nonsignificant_genes():
    qvals = pd.Series(
        [0.001, 0.2, np.nan, 0.01],
        index=["g1", "g2", "g3", "g4"],
    )

    selected = _select_significant_svg_names(
        qvals,
        n_svgs=100,
        qval_threshold=0.05,
    )

    assert selected.tolist() == ["g1", "g4"]


def test_qvalue_selection_validates_threshold():
    with pytest.raises(ValueError, match="qval_threshold"):
        _select_significant_svg_names(
            pd.Series([0.1], index=["g1"]),
            n_svgs=1,
            qval_threshold=1.1,
        )


def test_pearson_residual_mode_discloses_that_it_is_not_spatial(monkeypatch):
    adata = AnnData(
        X=np.array(
            [
                [1, 0, 2],
                [0, 2, 1],
                [3, 1, 0],
                [1, 1, 1],
            ],
            dtype=np.float32,
        )
    )
    adata.var_names = ["g1", "g2", "g3"]
    adata.obsm["spatial"] = np.array(
        [[0, 0], [1, 0], [0, 1], [1, 1]],
        dtype=np.float64,
    )

    def fake_preprocess(data, **kwargs):
        data.var["highly_variable"] = [True, False, True]
        return data

    monkeypatch.setattr("omicverse.pp.preprocess", fake_preprocess)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = svg(adata, mode="pearson_residuals", n_svgs=2)

    assert result.var["space_variable_features"].tolist() == [True, False, True]
    assert result.uns["space_svg_runs"]["pearson_residuals"]["spatial_evidence"] is False
    assert any("does not use spatial coordinates" in str(item.message) for item in caught)


def _spatial_adata_for_svg_backends():
    adata = AnnData(
        X=np.array(
            [
                [1, 2, 3],
                [2, 1, 2],
                [3, 2, 1],
                [2, 3, 2],
            ],
            dtype=np.float32,
        )
    )
    adata.var_names = ["g1", "g2", "g3"]
    adata.layers["counts"] = adata.X.copy()
    adata.obsm["spatial"] = np.array(
        [[0, 0], [1, 0], [0, 1], [1, 1]],
        dtype=np.float64,
    )
    return adata


def test_somde_branch_applies_qvalue_threshold(monkeypatch):
    class FakeSomNode:
        def __init__(self, coords, k):
            self.genes = None

        def reTrain(self, epochs):
            return None

        def mtx(self, frame):
            self.genes = list(frame.index)
            return None, None

        def norm(self):
            return None

        def run(self, n_jobs=1):
            return (
                pd.DataFrame(
                    {
                        "g": self.genes,
                        "LLR": [5.0, 1.0, 4.0],
                        "pval": [0.001, 0.4, 0.01],
                        "qval": [0.002, 0.5, 0.02],
                        "FSV": [0.8, 0.1, 0.7],
                    }
                ),
                2,
            )

    fake_somde = types.ModuleType("omicverse.external.somde")
    fake_somde.SomNode = FakeSomNode
    monkeypatch.setitem(sys.modules, "omicverse.external.somde", fake_somde)

    result = svg(
        _spatial_adata_for_svg_backends(),
        mode="somde",
        n_svgs=3,
        qval_threshold=0.05,
    )

    assert result.var_names[result.var["space_variable_features"]].tolist() == ["g1", "g3"]


def test_spatialde_branch_applies_qvalue_threshold(monkeypatch):
    fake_spatialde = types.ModuleType("omicverse.external.SpatialDE")

    def fake_run(coords, expression, **kwargs):
        return pd.DataFrame(
            {
                "g": list(expression.columns),
                "LLR": [5.0, 1.0, 4.0],
                "pval": [0.001, 0.4, 0.01],
                "qval": [0.002, 0.5, 0.02],
                "FSV": [0.8, 0.1, 0.7],
                "l": [1.0, 1.0, 1.0],
            }
        )

    fake_spatialde.run = fake_run
    fake_naivede = types.ModuleType("omicverse.external.NaiveDE")
    fake_naivede.stabilize = lambda matrix: matrix
    fake_naivede.regress_out = lambda sample_info, matrix, formula: matrix
    monkeypatch.setitem(sys.modules, "omicverse.external.SpatialDE", fake_spatialde)
    monkeypatch.setitem(sys.modules, "omicverse.external.NaiveDE", fake_naivede)

    result = svg(
        _spatial_adata_for_svg_backends(),
        mode="spatialde",
        n_svgs=3,
        qval_threshold=0.05,
        show_progress=False,
    )

    assert result.var_names[result.var["space_variable_features"]].tolist() == ["g1", "g3"]


def lattice(seed=1):
    rng = np.random.default_rng(seed)
    coords = np.indices((6, 6)).reshape(2, -1).T.astype(float)
    x = np.column_stack([coords[:, 0] + 1, rng.poisson(3, 36), np.ones(36)])
    a = AnnData(x)
    a.var_names = ['gradient', 'noise', 'constant']
    a.obsm['spatial'] = coords
    spatial_neighbors(a, n_neighs=4)
    return a


@pytest.mark.parametrize('mode', ['moran', 'geary'])
def test_single_library_matches_esda_reference(mode):
    esda = pytest.importorskip('esda')
    from libpysal.weights import WSP
    a = lattice()
    g = a.obsp['spatial_connectivities']
    w = WSP(g).to_W()
    ours = spatial_autocorr(a, mode=mode, genes=['gradient', 'noise'], copy=True)
    for gene in ['gradient', 'noise']:
        values = np.asarray(a[:, gene].X).ravel()
        ref = esda.Moran(values, w, permutations=0, two_tailed=False) if mode == 'moran' else esda.Geary(values, w, permutations=0)
        np.testing.assert_allclose(ours.loc[gene, 'I' if mode == 'moran' else 'C'], ref.I if mode == 'moran' else ref.C, rtol=1e-10)
        from scipy.stats import norm
        # ESDA chooses a one-sided tail based on the observed sign; OV tests
        # positive spatial clustering (Moran upper tail / Geary lower tail).
        expected_p = norm.sf(ref.z_norm) if mode == 'moran' else norm.cdf(ref.z_norm)
        np.testing.assert_allclose(ours.loc[gene, 'pval_norm'], expected_p, rtol=1e-8, atol=1e-12)


def test_multilibrary_statistics_equal_separate_runs_with_joint_bh(tmp_path):
    import anndata as ad
    a, b = lattice(1), lattice(2)
    b.X += 100
    combined = ad.concat([a, b], label='slice', keys=['a', 'b'], index_unique='-')
    spatial_neighbors(combined, n_neighs=4, library_key='slice')
    result = spatial_autocorr(combined, library_key='slice', copy=True)
    for name, original in [('a', a), ('b', b)]:
        reference = spatial_autocorr(original, copy=True).reindex(original.var_names)
        np.testing.assert_allclose(result[result.library == name].set_index('gene').reindex(original.var_names)['I'], reference['I'], equal_nan=True)
    valid = result.pval_norm.notna()
    np.testing.assert_allclose(result.loc[valid, 'pval_adj'], multipletests(result.loc[valid, 'pval_norm'], method='fdr_bh')[1])
    combined.uns['moranI'] = result
    combined.write_h5ad(tmp_path / 'stats.h5ad')


def test_significance_is_default_and_top_n_is_explicit():
    a = lattice()
    svg(a, mode='moran', n_svgs=3, n_perms=None)
    assert not a.var.loc['constant', 'space_variable_features']
    assert a.var.loc['gradient', 'space_variable_features']
    b = lattice()
    svg(b, mode='moran', n_svgs=2, n_perms=None, selection='top_n')
    assert b.var['space_variable_features'].sum() == 2


def test_permutation_probabilities_are_probabilities():
    a = lattice()
    result = spatial_autocorr(a, n_perms=19, seed=4, two_tailed=True, copy=True)
    finite = result[result.testable]
    assert finite.pval_sim.between(1 / 20, 1).all()
    assert finite.pval_z_sim.between(0, 1).all()


def test_finite_difference_matches_legacy_scipy_when_available():
    from omicverse.external._finite_difference import derivative
    for n in (1, 2):
        assert derivative(lambda x: x*x, 3, n=n) == (6 if n == 1 else 2)
        try:
            from scipy.misc import derivative as legacy
        except ImportError:
            continue
        np.testing.assert_allclose(derivative(np.sin, 0.3, n=n), legacy(np.sin, 0.3, n=n))


def test_multilibrary_svg_masks_and_no_batch_only_signal(tmp_path):
    import anndata as ad
    a, b = lattice(), lattice()
    b.X += 100
    combined = ad.concat([a, b], label='slice', keys=['a', 'b'], index_unique='-')
    svg(combined, mode='moran', library_key='slice', n_svgs=1, n_perms=None)
    masks = combined.varm['space_variable_features_by_library']
    assert not masks.loc['constant'].any()
    assert (masks.sum(axis=0) <= 1).all()
    np.testing.assert_array_equal(combined.var.space_variable_features, masks.any(axis=1))
    combined.write_h5ad(tmp_path / 'svg.h5ad')


@pytest.mark.parametrize('mode', ['spatialde', 'somde'])
def test_real_svg_backend_has_matching_selected_qvalues(mode, monkeypatch):
    if mode == 'somde':
        pytest.importorskip('somoclu')
    observed = {}
    if mode == 'spatialde':
        from omicverse.external import SpatialDE
        original = SpatialDE.run
        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            observed['raw'] = result.copy()
            return result
        monkeypatch.setattr(SpatialDE, 'run', capture)
    else:
        from omicverse.external.somde import SomNode
        original = SomNode.run
        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            observed['raw'] = result[0].copy()
            return result
        monkeypatch.setattr(SomNode, 'run', capture)
    a = lattice()
    # Nonconstant count features are needed by the backend variance stabilizer.
    rng = np.random.default_rng(3)
    a.X = rng.poisson(a.X + 2).astype(float)
    a.layers['counts'] = sparse.csr_matrix(a.X)
    options = {'k': 3} if mode == 'somde' else {'show_progress': False, 'kernel_space': {'SE': [1., 2.], 'const': 0}}
    svg(a, mode=mode, n_svgs=3, **options)
    selected = a.var.space_variable_features
    assert (a.var.loc[selected, mode + '_qval'] < 0.05).all()
    assert (a.var[mode + '_pval'].dropna().between(0, 1)).all()
    expected = observed['raw'].set_index('g').reindex(a.var_names)
    np.testing.assert_allclose(a.var[mode + '_pval'], expected.pval, equal_nan=True)
    np.testing.assert_allclose(a.var[mode + '_qval'], expected.qval, equal_nan=True)


def test_spatial_neighbors_never_connects_different_libraries():
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.01, 0.01],
            [1.01, 0.01],
            [0.01, 1.01],
        ]
    )
    adata = AnnData(
        np.ones((6, 1), dtype=np.float32),
        obs=pd.DataFrame(
            {"slice": ["A"] * 3 + ["B"] * 3},
            index=[f"c{i}" for i in range(6)],
        ),
    )
    adata.obsm["spatial"] = coords

    space.spatial_neighbors(adata, n_neighs=2, library_key="slice")

    graph = adata.obsp["spatial_connectivities"].tocoo()
    labels = adata.obs["slice"].to_numpy()
    assert not np.any(labels[graph.row] != labels[graph.col])
    assert adata.uns["spatial_neighbors"]["params"]["library_sizes"] == {
        "A": 3,
        "B": 3,
    }


def test_spatial_neighbors_counts_nonself_neighbors():
    adata = AnnData(np.ones((5, 1), dtype=np.float32))
    adata.obsm["spatial"] = np.column_stack(
        [np.arange(5, dtype=float), np.zeros(5, dtype=float)]
    )

    space.spatial_neighbors(adata, n_neighs=2)

    degree = adata.obsp["spatial_connectivities"].getnnz(axis=1)
    assert degree[0] >= 2
    assert degree[-1] >= 2


def _overlapping_library_adata():
    local_coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=float,
    )
    adata = AnnData(
        np.arange(16, dtype=np.float32).reshape(8, 2) + 1,
        obs=pd.DataFrame(
            {"slice": ["A"] * 4 + ["B"] * 4},
            index=[f"cell_{i}" for i in range(8)],
        ),
    )
    adata.var_names = ["g0", "g1"]
    adata.obsm["spatial"] = np.vstack([local_coords, local_coords])
    adata.uns["spatial"] = {"A": {}, "B": {}}
    return adata


def _assert_no_cross_library_edges(adata):
    graph = adata.obsp["spatial_connectivities"].tocoo()
    labels = adata.obs["slice"].to_numpy()
    assert not np.any(labels[graph.row] != labels[graph.col])


def test_morani_auto_graph_forwards_library_key_without_mutating_input(monkeypatch):
    adata = _overlapping_library_adata()
    x_before = np.asarray(adata.X).copy()
    obs_before = adata.obs.copy(deep=True)
    var_before = adata.var.copy(deep=True)
    uns_keys_before = set(adata.uns)
    obsp_keys_before = set(adata.obsp)
    captured = {}

    original_spatial_neighbors = svg_module.spatial_neighbors

    def capture_graph(*args, **kwargs):
        result = original_spatial_neighbors(*args, **kwargs)
        captured["library_key"] = kwargs.get("library_key")
        captured["copy"] = kwargs.get("copy")
        captured["connectivities"] = result[0]
        return result

    monkeypatch.setattr(svg_module, "spatial_neighbors", capture_graph)

    result = space.moranI(
        adata,
        genes=["g0"],
        auto_spatial_neighbors=True,
        n_neighs=2,
        library_key="slice",
        corr_method=None,
        copy=True,
    )

    assert result['gene'].tolist() == ['g0', 'g0']
    assert result['library'].tolist() == ['A', 'B']
    assert captured["library_key"] == "slice"
    assert captured["copy"] is True
    graph = captured["connectivities"].tocoo()
    labels = adata.obs["slice"].to_numpy()
    assert not np.any(labels[graph.row] != labels[graph.col])
    assert "moranI" not in adata.uns
    assert set(adata.uns) == uns_keys_before
    assert set(adata.obsp) == obsp_keys_before
    np.testing.assert_array_equal(np.asarray(adata.X), x_before)
    pd.testing.assert_frame_equal(adata.obs, obs_before)
    pd.testing.assert_frame_equal(adata.var, var_before)


def test_morani_stratification_does_not_call_a_pure_library_shift_spatial():
    adata = _overlapping_library_adata()
    adata.X = np.concatenate(
        [
            np.zeros((4, 2), dtype=np.float32),
            np.full((4, 2), 10.0, dtype=np.float32),
        ],
        axis=0,
    )

    result = space.moranI(
        adata,
        genes=["g0"],
        auto_spatial_neighbors=True,
        n_neighs=2,
        n_perms=19,
        seed=3,
        library_key="slice",
        corr_method=None,
        copy=True,
    )

    assert not result['testable'].any()
    assert result['I'].isna().all()
    assert result['pval_sim'].isna().all()


def test_svg_moran_forwards_library_key_to_automatic_graph():
    adata = _overlapping_library_adata()

    result = space.svg(
        adata,
        mode="moran",
        n_svgs=1,
        n_perms=None,
        library_key="slice",
    )

    assert result is adata
    assert int(adata.var["space_variable_features"].sum()) == 0
    assert list(adata.varm['space_variable_features_by_library'].columns) == ['A', 'B']
    assert set(adata.uns['spatial_features_by_library']['library']) == {'A', 'B'}


def test_spatial_autocorr_copy_does_not_mutate_uns():
    adata = lattice()
    keys_before = set(adata.uns)

    result = space.spatial_autocorr(
        adata,
        genes=list(adata.var_names[:2]),
        copy=True,
    )

    assert set(result.index) == set(adata.var_names[:2])
    assert "moranI" not in adata.uns
    assert set(adata.uns) == keys_before


@pytest.mark.parametrize('mode', ['moran', 'somde', 'spatialde'])
def test_svg_requires_library_identity_before_inference(mode):
    adata = _overlapping_library_adata()
    with pytest.raises(ValueError, match='library_key'):
        svg(adata, mode=mode, n_svgs=1)


@pytest.mark.parametrize('explicit', [False, True])
def test_autocorr_rejects_colliding_library_labels_on_existing_graph(explicit):
    adata = _overlapping_library_adata()
    spatial_neighbors(adata, library_key='slice')
    adata.obs['slice'] = np.array([1] * 4 + ['1'] * 4, dtype=object)
    with pytest.raises(ValueError, match='collide'):
        spatial_autocorr(adata, library_key='slice' if explicit else None, copy=True)

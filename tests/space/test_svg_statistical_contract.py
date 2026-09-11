import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse
from statsmodels.stats.multitest import multipletests

from omicverse.space._svg import spatial_neighbors, spatial_autocorr, svg


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

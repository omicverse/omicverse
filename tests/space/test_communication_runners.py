import importlib.util

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from omicverse.external.commot.tools import spatial_communication
from omicverse.space import create_communication_anndata, run_commot, run_flowsig


def fixture(n=12):
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(3, (n, 3)).astype(float))
    a.var_names = ['L', 'R', 'extra']
    a.obsm['spatial'] = np.column_stack([np.arange(n), np.zeros(n)]).astype(float)
    a.obsm['X_gem'] = rng.uniform(0.1, 2, (n, 2))
    a.obs['block'] = np.repeat(np.arange(3), n // 3)
    a.obs['type'] = ['A'] * (n // 2) + ['B'] * (n // 2)
    db = pd.DataFrame([['L', 'R', 'P']], columns=['ligand', 'receptor', 'pathway'])
    return a, db


def test_commot_matches_backend_and_preserves_input_layers(tmp_path):
    a, db = fixture()
    a.layers['expr'] = a.X.copy()
    a.X = a.X * 2
    raw = a.copy()
    raw.X = raw.layers['expr'].copy()
    spatial_communication(raw, database_name='test', df_ligrec=db, dis_thr=2, pathway_sum=True, heteromeric=True)
    before = a.copy()
    out = run_commot(a, database_name='test', df_ligrec=db[['pathway', 'receptor', 'ligand']],
                     dis_thr=2, distance_unit='arbitrary', layer='expr', inplace=False)
    np.testing.assert_allclose(out.obsp['commot-test-L-R'].toarray(), raw.obsp['commot-test-L-R'].toarray())
    np.testing.assert_array_equal(out.X, before.X)
    assert 'commot-test-info' not in a.uns
    summary = create_communication_anndata(out, 'type', 9, level='lr')
    assert summary.n_vars == 1
    out.write_h5ad(tmp_path / 'out.h5ad')
    assert ad.read_h5ad(tmp_path / 'out.h5ad').uns['commot-test-info']['omicverse_run']['layer'] == 'expr'
    with pytest.raises(ValueError, match='exists'):
        run_commot(out, database_name='test', df_ligrec=db, dis_thr=2, distance_unit='arbitrary')


def test_validation_failure_does_not_write_partial_results():
    a, db = fixture()
    a.obs['sample'] = ['one'] * 6 + ['two'] * 6
    with pytest.raises(ValueError, match='each library'):
        run_commot(a, database_name='test', df_ligrec=db, dis_thr=2,
                   distance_unit='arbitrary', library_key='sample')
    assert not a.uns
    with pytest.raises(MemoryError):
        run_commot(a, database_name='test', df_ligrec=db, dis_thr=2,
                   distance_unit='arbitrary', max_distance_matrix_mb=0.00001)


def test_failed_backend_preserves_input_and_existing_outputs(monkeypatch):
    a, db = fixture()
    a.uns['commot-test-info'] = {'sentinel': 1}
    before_x = a.X.copy()
    from omicverse.external.commot import tools

    def fail(work, **kwargs):
        work.uns['commot-test-info'] = {'partial': True}
        raise RuntimeError('backend failed')

    monkeypatch.setattr(tools, 'spatial_communication', fail)
    with pytest.raises(RuntimeError, match='backend failed'):
        run_commot(a, database_name='test', df_ligrec=db, dis_thr=2,
                   distance_unit='arbitrary', overwrite=True)
    assert a.uns['commot-test-info'] == {'sentinel': 1}
    np.testing.assert_array_equal(a.X, before_x)


def test_namespace_prefix_overlap_is_rejected():
    a, db = fixture()
    a.uns['commot-test-other-info'] = {'df_ligrec': db}
    with pytest.raises(ValueError, match='overlapping'):
        run_commot(a, database_name='test', df_ligrec=db, dis_thr=2,
                   distance_unit='arbitrary', overwrite=True)


@pytest.mark.skipif(importlib.util.find_spec('causaldag') is None, reason='requires causaldag')
def test_real_commot_flowsig_workflow_and_roundtrip(tmp_path):
    a, db = fixture(24)
    run_commot(a, database_name='test', df_ligrec=db, dis_thr=2, distance_unit='arbitrary')
    result = run_flowsig(a, commot_output_key='commot-test', block_key='block',
                         n_bootstraps=2, n_jobs=1, inplace=False)
    assert 'flowsig_network' not in a.uns
    net = result.uns['flowsig_network']['network']
    for key in ['adjacency', 'adjacency_validated', 'adjacency_validated_filtered']:
        assert np.isfinite(net[key]).all()
        assert net[key].shape == (4, 4)
    result.write_h5ad(tmp_path / 'flow.h5ad')
    restored = ad.read_h5ad(tmp_path / 'flow.h5ad')
    assert restored.uns['flowsig_network']['omicverse_run']['variable_selection'] == 'all'

"""Numerical contracts for COMMOT aggregation and FlowSig signal identities."""
import builtins

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from omicverse.external.commot.tools import _spatial_communication as ct
from omicverse.external.flowsig.preprocessing import _flow_expressions as fs


@pytest.mark.parametrize('as_sparse', [False, True])
@pytest.mark.parametrize('scale, expected', [('sum', 6.), (None, 1.), (1, 1.), (3., 3.), ('auto', 10.)])
@pytest.mark.parametrize('missing_cupy', [False, True])
def test_cpu_aggregation_honors_scale(as_sparse, scale, expected, missing_cupy, monkeypatch):
    matrix = np.zeros((5, 5))
    matrix[:2, 2:] = 1.
    labels = np.array(['A', 'A', 'B', 'B', 'B'])
    if as_sparse:
        matrix = sparse.csr_matrix(matrix)
    if missing_cupy:
        original_import = builtins.__import__

        def no_cupy(name, *args, **kwargs):
            if name == 'cupy':
                raise ImportError('test missing CuPy')
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', no_cupy)
    rng_state = np.random.get_state()
    try:
        np.random.seed(7)
        _, reference_p = ct.summarize_cluster_optimized(matrix, labels, ['A', 'B', 'empty'], 19)
        np.random.seed(7)
        values, pvalues = ct.summarize_cluster_gpu(
            matrix, labels, ['A', 'B', 'empty'], 19,
            use_gpu=missing_cupy, scale_factor=scale,
        )
    finally:
        np.random.set_state(rng_state)
    assert values.loc['A', 'B'] == pytest.approx(expected)
    assert np.all(values.loc['empty'] == 0)
    pd.testing.assert_frame_equal(pvalues, reference_p)


def _flow_fixture(as_sparse):
    ligands = ['WNT1', 'WNT10', 'signal', 'L_A']
    x = np.arange(8, dtype=float).reshape(2, 4) + 1
    adata = AnnData(sparse.csr_matrix(x) if as_sparse else x)
    adata.var_names = ligands
    adata.obsm['X_gem'] = np.array([[1., 2.], [3., 4.]])
    database = pd.DataFrame({
        'ligand': ['WNT1', 'WNT10', 'signal', 'L_A', 'WNT1'],
        'receptor': ['R', 'R', 'R', 'R_B', 'R'],
        'pathway': ['P'] * 5,
    })
    adata.uns['commot-test-info'] = {'df_ligrec': database}
    scores = {'WNT1-R': [1., 2.], 'WNT10-R': [10., 20.],
              'signal-R': [3., 4.], 'L_A-R_B': [5., 6.], 'WNT1-pathway': [100., 200.]}
    for direction, prefix in [('receiver', 'r-'), ('sender', 's-')]:
        adata.obsm[f'commot-test-sum-{direction}'] = pd.DataFrame(
            {prefix + key: value for key, value in scores.items()}, index=adata.obs_names)
    return adata


@pytest.mark.parametrize('as_sparse', [False, True])
def test_flowsig_exact_ligand_identity_and_complete_construction(as_sparse):
    adata = _flow_fixture(as_sparse)
    inflow, _ = fs.construct_inflow_signals_commot(adata, 'commot-test')
    for ligand, expected in [('WNT1', [1, 2]), ('WNT10', [10, 20]),
                             ('signal', [3, 4]), ('L_A', [5, 6])]:
        np.testing.assert_allclose(np.asarray(inflow[:, ['inflow-' + ligand]].X).ravel(), expected)
    outflow, _ = fs.construct_outflow_signals_commot(adata, 'commot-test')
    assert outflow.var.loc['WNT1', 'interactions'] == 'WNT1-R'
    assert inflow.var.loc['inflow-WNT1', 'interactions'] == 'WNT1-R'
    fs.construct_flows_from_commot(adata, 'commot-test', scale_gem_expr=False)
    variables = adata.uns['flowsig_network']['flow_var_info'].index
    np.testing.assert_allclose(adata.obsm['X_flow'][:, variables.get_loc('inflow-WNT1')], [1, 2])


def test_real_commot_output_is_consumed_by_flowsig():
    adata = AnnData(np.array([[1., 0.], [0., 2.], [0., 3.]]))
    adata.var_names = ['L', 'R']
    adata.obsm['spatial'] = np.array([[0., 0.], [1., 0.], [10., 0.]])
    database = pd.DataFrame([['L', 'R', 'P']], columns=['ligand', 'receptor', 'pathway'])
    ct.spatial_communication(
        adata, database_name='toy', df_ligrec=database, dis_thr=2., pathway_sum=True,
    )
    signal = adata.obsp['commot-toy-L-R']
    assert signal[0, 1] > 0
    assert signal[0, 2] == 0
    inflow, _ = fs.construct_inflow_signals_commot(adata, 'commot-toy')
    np.testing.assert_allclose(
        np.asarray(inflow.X).ravel(), np.asarray(signal.sum(axis=0)).ravel(),
    )


def test_real_cupy_sum_matches_cpu():
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('No CUDA device')
    matrix = np.zeros((5, 5))
    matrix[:2, 2:] = 1.
    labels = np.array(['A', 'A', 'B', 'B', 'B'])
    cpu, _ = ct.summarize_cluster_gpu(matrix, labels, ['A', 'B'], 9, use_gpu=False, scale_factor='sum')
    gpu, _ = ct.summarize_cluster_gpu(matrix, labels, ['A', 'B'], 9, use_gpu=True, scale_factor='sum')
    np.testing.assert_allclose(cpu, gpu)

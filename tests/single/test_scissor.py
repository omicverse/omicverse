"""Optional Scissor adapter tests against real OV imports (no network/data downloads)."""
import builtins
import copy
import json
import socket
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

ad = pytest.importorskip('anndata')


@pytest.fixture
def scissor_class(monkeypatch, tmp_path):
    # Exercise actual OmicVerse, including its registry. Keep its existing
    # telemetry/cache side effects out of this test's network/user environment.
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path/'mpl'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path/'cache'))
    import requests
    def no_network(*args, **kwargs):
        raise OSError('Network disabled for offline Scissor adapter test')
    monkeypatch.setattr(requests.Session, 'request', no_network)
    monkeypatch.setattr(socket, 'getaddrinfo', no_network)
    for name in ('connect', 'connect_ex', 'sendto', 'sendmsg'):
        if hasattr(socket.socket, name):
            monkeypatch.setattr(socket.socket, name, no_network)
    import omicverse as ov
    return ov.single.Scissor


def inputs():
    rng = np.random.default_rng(600)
    genes = pd.Index([f'g{i}' for i in range(30)])
    cells = pd.Index([f'c{i}' for i in range(6)])
    bulk = pd.DataFrame(rng.poisson(8, (30, 18)), index=genes, columns=[f's{i}' for i in range(18)])
    counts = rng.poisson(7, (6, 30)).astype(float)
    data = ad.AnnData(sparse.csr_matrix(np.log1p(counts)), obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=genes))
    data.layers['counts'] = sparse.csr_matrix(counts)
    data.obsp['network'] = sparse.diags([np.ones(5), np.ones(5)], [-1, 1], format='csr')
    data.obs['original'] = pd.Categorical(['a','a','b','b','b','a'])
    data.uns['original'] = {'source':'synthetic'}
    return data, bulk, pd.Series(np.arange(18)%2, index=bulk.columns)


def test_optional_backend_is_checked_only_when_run_is_called(scissor_class, monkeypatch):
    original_import = builtins.__import__
    def missing_backend(name, *args, **kwargs):
        if name == 'scissor_py':
            raise ModuleNotFoundError("No module named 'scissor_py'", name='scissor_py')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', missing_backend)
    model = scissor_class(None, bulk=None, phenotype=None, family='binomial', graph_key='network')
    with pytest.raises(ImportError, match='omicverse/py-Scissors'): model.run()
    assert model.result_ is None


def test_backend_internal_import_errors_are_not_hidden(scissor_class, monkeypatch):
    original_import = builtins.__import__
    def broken_backend(name, *args, **kwargs):
        if name == 'scissor_py':
            raise ModuleNotFoundError('broken dependency', name='broken_dependency')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', broken_backend)
    model = scissor_class(None, bulk=None, phenotype=None, family='binomial', graph_key='network')
    with pytest.raises(ModuleNotFoundError) as error: model.run()
    assert error.value.name == 'broken_dependency'


@pytest.mark.parametrize('family', ['gaussian','binomial','cox'])
@pytest.mark.parametrize('layer', [None,'counts'])
def test_family_layer_and_writeback_match_actual_backend(scissor_class, family, layer):
    backend = pytest.importorskip('scissor_py')
    data, bulk, y = inputs()
    if family == 'gaussian': y = pd.Series(np.sin(np.arange(18)), index=bulk.columns)
    elif family == 'cox': y = pd.DataFrame({'time':np.ceil(np.arange(1,19)/2), 'status':np.ones(18)}, index=bulk.columns)
    original = data.copy()
    expected = backend.scissor_from_anndata(original, bulk, y, layer='X' if layer is None else layer,
        graph_key='network', family=family, semantics='corrected', alpha=.4, lambdas=[1], foldid=np.arange(18)%3)
    model = scissor_class(data, bulk=bulk, phenotype=y, family=family, layer=layer,
                          graph_key='network', semantics='corrected')
    result = model.run(alpha=.4, lambdas=[1], foldid=np.arange(18)%3)
    assert model.result_ is result
    pd.testing.assert_series_equal(result.coefficients, expected.coefficients)
    assert 'scissor_coef' in data.obs
    assert data.obs['scissor'].cat.categories.tolist() == ['Scissor+','Scissor-','background']
    assert json.loads(data.uns['scissor']['sharedgenes']) == list(result.prepared.gene_ids)
    assert (data.X != original.X).nnz == 0
    assert (data.layers['counts'] != original.layers['counts']).nnz == 0
    assert (data.obsp['network'] != original.obsp['network']).nnz == 0
    assert data.uns['original'] == original.uns['original']


def test_string_mapping_id_alignment_h5ad_and_collision(scissor_class, tmp_path):
    backend = pytest.importorskip('scissor_py')
    data, bulk, y = inputs()
    y = y.map({0:'wildtype', 1:'mutant'}).iloc[::-1]
    folds = pd.Series(np.arange(18)%3, index=bulk.columns).iloc[::-1]
    model = scissor_class(data, bulk=bulk, phenotype=y, family='binomial', layer='counts',
        graph_key='network', positive_label='mutant', semantics='corrected', key_added='TP53')
    result = model.run(alpha=.4, lambdas=[1], foldid=folds)
    assert json.loads(data.uns['TP53']['encoding']) == {'positive':'mutant','negative':'wildtype'}
    path = tmp_path/'result.h5ad';backend.write_h5ad_atomic(data, path)
    recovered = ad.read_h5ad(path)
    pd.testing.assert_frame_equal(recovered.obs, data.obs)
    assert recovered.uns['TP53'] == data.uns['TP53']
    previous = copy.deepcopy(data.uns)
    with pytest.raises(ValueError, match='already exists'): model.run(alpha=.4, lambdas=[1], foldid=folds)
    assert model.result_ is result and data.uns == previous


def test_missing_graph_guides_explicit_preprocessing(scissor_class):
    pytest.importorskip('scissor_py')
    data, bulk, y = inputs();before = data.copy()
    model = scissor_class(data, bulk=bulk, phenotype=y, family='binomial', layer='counts', graph_key='missing')
    with pytest.raises(ValueError, match='does not automatically build'): model.run()
    pd.testing.assert_frame_equal(data.obs, before.obs)
    assert data.uns == before.uns and model.result_ is None

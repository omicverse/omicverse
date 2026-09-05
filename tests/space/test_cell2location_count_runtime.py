import numpy as np
import pytest
from anndata import AnnData


def test_real_cell2location_count_layers():
    pytest.importorskip('pyro')
    pytest.importorskip('scvi')
    from omicverse.space import Deconvolution
    rng = np.random.default_rng(9)
    ref = AnnData(rng.poisson(5, (60, 20)).astype(np.float32))
    ref.obs['cell_type'] = np.repeat(['A', 'B', 'C'], 20)
    ref.layers['counts'] = ref.X.copy()
    spatial = AnnData(rng.poisson(12, (12, 20)).astype(np.float32))
    spatial.layers['counts'] = spatial.X.copy()
    spatial.obs['sample'] = 'one'
    ref.X = np.log1p(ref.X)
    spatial.X = np.log1p(spatial.X)
    model = Deconvolution(adata_sp=spatial, adata_sc=ref)
    model.deconvolution(method='cell2location', celltype_key_sc='cell_type', batch_key_sp='sample',
        cell2location_scrna_kwargs={'max_epochs': 1, 'batch_size': 30, 'train_size': 1, 'accelerator': 'cpu', 'device': 'auto'},
        cell2location_spatial_kwargs={'max_epochs': 1, 'batch_size': None, 'train_size': 1, 'accelerator': 'cpu', 'device': 'auto'},
        sample_kwargs={'num_samples': 2, 'batch_size': 12})
    assert model.adata_cell2location.n_obs == 12
    np.testing.assert_array_equal(ref.layers['counts'], np.rint(ref.layers['counts']))
    for fitted in (model.mod_sc, model.mod_sp):
        registered = fitted.adata_manager.get_from_registry('X')
        expected = fitted.adata.layers['counts']
        registered = registered.toarray() if hasattr(registered, 'toarray') else np.asarray(registered)
        expected = expected.toarray() if hasattr(expected, 'toarray') else np.asarray(expected)
        np.testing.assert_array_equal(registered, expected)
        np.testing.assert_array_equal(registered, np.rint(registered))

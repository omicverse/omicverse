import sys
import types
import numpy as np
import pandas as pd
from anndata import AnnData


def test_cytotrace_receives_counts_and_gene_identifiers(monkeypatch):
    from omicverse.single._traj import TrajInfer
    a = AnnData(np.zeros((3,12)), var=pd.DataFrame(index=['ERCC-control']+[f'g{i}' for i in range(11)]))
    a.layers['counts'] = np.arange(36).reshape(3,12)
    seen = {}
    def backend(matrix, **kwargs):
        seen.update(matrix=matrix, **kwargs)
        return types.SimpleNamespace(cytotrace=np.array([0.,.5,1.]))
    monkeypatch.setitem(sys.modules, 'pycytotrace', types.SimpleNamespace(cytotrace_run=backend))
    TrajInfer(a).inference(method='cytotrace')
    np.testing.assert_array_equal(seen['matrix'], a.layers['counts'].T)
    assert seen['gene_names'] == list(a.var_names)
    assert seen['cell_names'] == list(a.obs_names)
    np.testing.assert_array_equal(a.obs['cytotrace_pseudotime'],[1.,.5,0.])

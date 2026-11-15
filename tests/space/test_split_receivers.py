import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from omicverse import space as splitst


def test_residuals_only_reach_non_rejected_second_type_neighbors():
    a = AnnData(np.array([[10.], [3.], [4.], [5.]]))
    a.layers['counts'] = a.X.copy()
    a.layers['split_balanced'] = np.array([[6.], [3.], [4.], [5.]])
    a.obsm['spatial'] = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    a.obs['first_type'] = ['A', 'B', 'B', 'A']
    a.obs['second_type'] = ['B', 'A', 'A', 'B']
    a.obs['spot_class'] = ['singlet', 'singlet', 'reject', 'singlet']
    splitst.split_reassign_residuals(a, k=3, mode='uniform')
    np.testing.assert_allclose(a.layers['split_reassigned'].toarray().ravel(), [6, 7, 4, 5])
    np.testing.assert_allclose(a.uns['split_reassignment_operator'].toarray()[0], [0, 1, 0, 0])
    splitst.split_reassign_residuals(a, k=3, mode='uniform', receiver_policy='all')
    np.testing.assert_allclose(a.uns['split_reassignment_operator'].toarray()[0], [0, 1/3, 1/3, 1/3])

import numpy as np
import pandas as pd
from anndata import AnnData
from omicverse import space as splitst


def test_shift_uses_residual_and_does_not_match_missing_classes():
    a = AnnData(np.array([[10., 6.], [5., 3.], [4., 2.]]))
    a.layers['counts'] = a.X.copy()
    a.layers['split_purified'] = np.array([[7., 2.], [5., 3.], [2., 1.]])
    a.obs['purification_status'] = ['purified', 'raw', 'purified']
    a.obs['neighborhood_weights_second_type'] = 1.
    a.obs['first_type'] = ['A', 'A', 'A']
    a.obs['second_type'] = ['B', None, 'B']
    a.obs['weight_first_type'] = .7
    a.obs['weight_second_type'] = .3
    a.obs['first_type_neighborhood_agreement'] = False
    a.obs['first_type_class_neighborhood_agreement'] = False
    a.obs['second_type_class'] = ['B', None, 'B']
    a.obs['first_type_class_neighborhood'] = ['B', None, 'B']
    a.obs['spot_class'] = ['singlet', 'singlet', 'reject']
    splitst.split_balance(a,swap_labels=True,spot_class_key='spot_class')
    np.testing.assert_allclose(a.layers['split_balanced'].toarray(),[[3,4],[5,3],[0,0]])
    assert a.obs.split_shift_swap.tolist() == [True, False, True]
    assert a.obs.split_balance_status.tolist() == ['purified', 'raw', 'removed']

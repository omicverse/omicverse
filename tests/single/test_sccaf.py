"""ov.single.SCCAF_assessment with the default logistic-regression classifier."""

import numpy as np
import pandas as pd
import pytest

import omicverse as ov
from omicverse.single._sccaf import get_topmarkers


def _blobs(n_clusters):
    # cluster k over-expresses genes g{8k}..g{8k+3}
    rng = np.random.default_rng(0)
    labels = pd.Series(np.repeat([f"c{k}" for k in range(n_clusters)], 60))
    X = rng.normal(size=(len(labels), 40))
    for k in range(n_clusters):
        X[(labels == f"c{k}").to_numpy(), 8 * k:8 * k + 4] += 3
    return X, labels


@pytest.mark.parametrize("n_clusters", [2, 5])
def test_sccaf_assessment_lr(n_clusters):
    X, labels = _blobs(n_clusters)
    np.random.seed(0)
    y_prob, y_pred, y_test, clf, cvsm, acc = ov.single.SCCAF_assessment(X, labels)
    assert cvsm > 0.9 and acc > 0.9
    assert y_prob.shape == (len(y_test), n_clusters)
    np.testing.assert_allclose(y_prob.sum(axis=1), 1)


def test_sccaf_topmarkers():
    X, labels = _blobs(5)
    np.random.seed(0)
    clf = ov.single.SCCAF_assessment(X, labels)[3]
    markers = get_topmarkers(clf, [f"g{i}" for i in range(40)], topn=4)
    for k in range(5):
        genes = markers.loc[markers["cell_type"] == f"c{k}", "gene"]
        assert set(genes) == {f"g{i}" for i in range(8 * k, 8 * k + 4)}

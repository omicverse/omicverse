import numpy as np
import pandas as pd

from omicverse.external.pyscenic.binarization import derive_threshold


def test_derive_threshold_hdt_does_not_require_numpy_msort(monkeypatch):
    monkeypatch.delattr(np, "msort", raising=False)
    auc_mtx = pd.DataFrame({"regulon": [0.1, 0.2, 0.2, 0.8, 0.9, 1.0]})

    threshold = derive_threshold(auc_mtx, "regulon", seed=1, method="hdt")

    assert np.isfinite(threshold)

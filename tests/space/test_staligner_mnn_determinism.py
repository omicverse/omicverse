from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from omicverse.external.STAligner import mnn_utils


def test_mnn_dictionary_sorts_anchors_and_positive_neighbors(monkeypatch):
    adata = AnnData(
        np.ones((4, 1), dtype=np.float32),
        obs=pd.DataFrame(
            {"batch": ["a", "a", "b", "b"]},
            index=["a2", "a1", "b2", "b1"],
        ),
    )
    adata.obsm["embed"] = np.arange(4, dtype=float)[:, None]
    monkeypatch.setattr(
        mnn_utils,
        "mnn",
        lambda *args, **kwargs: {
            ("b2", "a2"),
            ("b1", "a2"),
            ("b1", "a1"),
        },
    )

    result = mnn_utils.create_dictionary_mnn(
        adata,
        use_rep="embed",
        batch_name="batch",
        k=2,
        approx=False,
        verbose=0,
    )["a_b"]

    assert list(result) == ["a1", "a2", "b1", "b2"]
    assert result["b1"] == ["a1", "a2"]

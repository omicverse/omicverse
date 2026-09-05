from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from omicverse.space._deconvolution import _starfysh_visium_metadata


def _visium_adata():
    adata = AnnData(
        np.ones((2, 1), dtype=np.float32),
        obs=pd.DataFrame(
            {"array_row": [3, 4], "array_col": [7, 8]},
            index=["a", "b"],
        ),
    )
    adata.obsm["spatial"] = np.array([[101.0, 23.0], [211.0, 37.0]])
    adata.uns["spatial"] = {
        "slide": {
            "images": {"hires": np.zeros((20, 30, 3), dtype=np.uint8)},
            "scalefactors": {"tissue_hires_scalef": 0.1},
        }
    }
    return adata


def test_starfysh_metadata_preserves_canonical_xy_axis_meaning():
    sample_id, metadata = _starfysh_visium_metadata(_visium_adata())

    assert sample_id == "slide"
    np.testing.assert_array_equal(metadata["map_info"]["imagecol"], [101.0, 211.0])
    np.testing.assert_array_equal(metadata["map_info"]["imagerow"], [23.0, 37.0])


def test_starfysh_rejects_ambiguous_multi_library_objects():
    adata = _visium_adata()
    adata.uns["spatial"]["other"] = adata.uns["spatial"]["slide"].copy()

    with pytest.raises(ValueError, match="one Visium library.*Subset"):
        _starfysh_visium_metadata(adata)

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from omicverse.space._stt import STT


def test_stt_validates_spatial_and_region_inputs_before_backend_work():
    adata = AnnData(np.ones((3, 2), dtype=np.float32))
    with pytest.raises(KeyError, match="spatial_loc"):
        STT(adata)

    adata.obsm["spatial"] = np.ones((3, 2), dtype=float)
    with pytest.raises(KeyError, match="region"):
        STT(adata)


def test_stt_preserves_legacy_xy_default_when_both_coordinate_keys_exist():
    adata = AnnData(np.ones((3, 2), dtype=np.float32))
    adata.obs["Region"] = ["a", "a", "b"]
    adata.obsm["xy_loc"] = np.zeros((3, 2), dtype=float)
    adata.obsm["spatial"] = np.ones((3, 2), dtype=float)

    with pytest.warns(FutureWarning, match="legacy default"):
        stt = STT(adata)

    assert stt.spatial_loc == "xy_loc"


def test_stt_defaults_to_standard_spatial_key_for_new_objects():
    adata = AnnData(np.ones((3, 2), dtype=np.float32))
    adata.obs["Region"] = ["a", "a", "b"]
    adata.obsm["spatial"] = np.ones((3, 2), dtype=float)

    stt = STT(adata)

    assert stt.spatial_loc == "spatial"

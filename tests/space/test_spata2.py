import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from omicverse import space


def make_spatial_adata():
    coords = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [12.0, 12.0],
        ],
        dtype=float,
    )
    x = np.array(
        [
            [1.0, 0.0, 5.0],
            [2.0, 1.0, 4.0],
            [3.0, 1.0, 3.0],
            [4.0, 2.0, 2.0],
            [5.0, 3.0, 1.0],
            [8.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    obs = pd.DataFrame(
        {
            "region": ["core", "core", "edge", "edge", "core", "artifact"],
            "total_counts": x.sum(axis=1),
        },
        index=[f"spot_{idx}" for idx in range(coords.shape[0])],
    )
    adata = AnnData(X=x, obs=obs, var=pd.DataFrame(index=["GeneA", "GeneB", "GeneC"]))
    adata.obsm["spatial"] = coords
    return adata


def test_spata2_get_coords_reads_spatial_coordinates():
    adata = make_spatial_adata()

    coords = space.spata2_get_coords(adata, include_obs=["region"])

    assert list(coords.columns) == ["barcode", "x", "y", "region"]
    assert coords.loc["spot_2", "x"] == 1.0
    assert coords.loc["spot_2", "barcode"] == "spot_2"


def test_spata2_extract_variables_from_obs_and_expression():
    adata = make_spatial_adata()

    values = space.spata2_extract_variables(adata, ["region", "GeneA", "GeneC"])

    assert values.loc["spot_0", "region"] == "core"
    assert values.loc["spot_3", "GeneA"] == 4.0
    assert values.loc["spot_4", "GeneC"] == 1.0


def test_spata2_extract_variables_supports_sparse_layers():
    adata = make_spatial_adata()
    adata.layers["scaled"] = sparse.csr_matrix(adata.X * 2.0)

    values = space.spata2_extract_variables(adata, "GeneB", layer="scaled")

    assert values["GeneB"].tolist() == [0.0, 2.0, 2.0, 4.0, 6.0, 0.0]


def test_spata2_join_variables_combines_coords_and_values():
    adata = make_spatial_adata()

    table = space.spata2_join_variables(adata, ["region", "GeneB"])

    assert list(table.columns) == ["barcode", "x", "y", "region", "GeneB"]
    assert table.loc["spot_1", "GeneB"] == 1.0


def test_spata2_tissue_outline_writes_hull_to_uns():
    adata = make_spatial_adata()

    outline = space.spata2_tissue_outline(adata)

    assert {"x", "y"} == set(outline.columns)
    assert len(outline) >= 3
    assert "spata2_tissue_outline" in adata.uns


def test_spata2_identify_and_remove_outliers():
    adata = make_spatial_adata()

    outliers = space.spata2_identify_outliers(adata, radius=1.6, min_neighbors=2)
    filtered = space.spata2_remove_outliers(adata)

    assert outliers.loc["spot_5"]
    assert not outliers.loc["spot_0"]
    assert filtered.n_obs == 5
    assert "spot_5" not in set(filtered.obs_names)
    assert adata.n_obs == 6


def test_spata2_unit_conversion_roundtrip():
    units = space.spata2_pixels_to_unit(np.array([0.0, 10.0, 25.0]), pixels_per_unit=5.0)
    pixels = space.spata2_unit_to_pixels(units, pixels_per_unit=5.0)

    np.testing.assert_allclose(units, [0.0, 2.0, 5.0])
    np.testing.assert_allclose(pixels, [0.0, 10.0, 25.0])


def test_spata2_missing_variable_raises_keyerror():
    adata = make_spatial_adata()

    with pytest.raises(KeyError):
        space.spata2_extract_variables(adata, "MissingGene")

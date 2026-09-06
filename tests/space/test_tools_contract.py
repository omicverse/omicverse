from __future__ import annotations

import inspect

import numpy as np
import pytest
from anndata import AnnData

from omicverse._registry import get_registry
from omicverse import space
from omicverse.space import _tools


def _visium_adata(coords):
    adata = AnnData(np.ones((len(coords), 1), dtype=np.float32))
    adata.obsm["spatial"] = np.asarray(coords)
    image = np.zeros((7, 7), dtype=np.uint8)
    image[3, 5] = 1
    adata.uns["spatial"] = {
        "sample": {
            "images": {"hires": image},
            "scalefactors": {"tissue_hires_scalef": 1.0},
        }
    }
    return adata


def _multilibrary_adata():
    adata = AnnData(np.ones((4, 1), dtype=np.float32))
    adata.obs_names = ["a1", "a2", "b1", "b2"]
    adata.obs["library"] = ["A", "A", "B", "B"]
    adata.obsm["spatial"] = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    )
    image = np.zeros((7, 7), dtype=np.uint8)
    adata.uns["spatial"] = {
        key: {
            "images": {"hires": image.copy()},
            "scalefactors": {"tissue_hires_scalef": 1.0},
        }
        for key in ("A", "B")
    }
    return adata


def _multiresolution_adata():
    adata = AnnData(np.ones((2, 1), dtype=np.float32))
    adata.obs_names = ["selected", "outside"]
    adata.obsm["spatial"] = np.array([[4.0, 2.0], [8.0, 6.0]])
    hires = np.zeros((8, 10), dtype=np.uint8)
    lowres = np.zeros((4, 5), dtype=np.uint8)
    hires[2, 4] = 1
    lowres[1, 2] = 1
    adata.uns["spatial"] = {
        "sample": {
            "images": {
                "hires": hires,
                "lowres": lowres,
                "segmentation": np.zeros_like(hires),
            },
            "scalefactors": {
                "tissue_hires_scalef": 1.0,
                "tissue_lowres_scalef": 0.5,
            },
        }
    }
    return adata


def test_rotate_keeps_integer_input_precise_and_aligned_to_image():
    adata = _visium_adata(np.array([[5, 3], [3, 3]], dtype=np.int64))

    rotated = _tools.rotate_space_visium(
        adata,
        angle=90,
        center=(3, 3),
        library_id="sample",
        interpolation_order=0,
    )

    assert np.issubdtype(rotated.obsm["spatial"].dtype, np.floating)
    assert np.allclose(rotated.obsm["spatial"], [[3, 1], [3, 3]])
    assert np.argwhere(rotated.uns["spatial"]["sample"]["images"]["hires"] > 0).tolist() == [[1, 3]]


def test_rotate_only_transforms_the_selected_library():
    adata = _multilibrary_adata()

    rotated = _tools.rotate_space_visium(
        adata,
        angle=90,
        center=(0, 0),
        library_id="A",
        library_key="library",
        interpolation_order=0,
    )

    assert np.allclose(rotated.obsm["spatial"][:2], [[0, -1], [1, 0]])
    assert np.array_equal(rotated.obsm["spatial"][2:], adata.obsm["spatial"][2:])


def test_rotate_synchronizes_all_scaled_image_resolutions_and_drops_stale_images():
    adata = _multiresolution_adata()

    rotated = _tools.rotate_space_visium(
        adata,
        angle=90,
        center=(2, 2),
        library_id="sample",
        interpolation_order=0,
    )

    images = rotated.uns["spatial"]["sample"]["images"]
    assert set(images) == {"hires", "lowres"}
    assert np.argwhere(images["hires"] > 0).tolist() == [[0, 2]]
    assert np.argwhere(images["lowres"] > 0).tolist() == [[0, 1]]
    assert np.allclose(rotated.obsm["spatial"][0], [2, 0])
    records = rotated.uns["spatial"]["sample"]["metadata"][
        "omicverse_removed_stale_images"
    ]
    record = records[sorted(records)[-1]]
    assert record["operation"] == "rotate"
    assert record["image_keys"].tolist() == ["segmentation"]


def test_torch_phase_correlation_aligns_2d_and_channel_first_images():
    torch = pytest.importorskip("torch")

    image = torch.zeros((16, 17), dtype=torch.float32)
    image[5:8, 6:9] = 1.0
    shifted = torch.roll(image, shifts=(2, 3), dims=(0, 1))

    offset_2d, aligned_2d = _tools.find_image_offset_phase_correlation_torch(
        image,
        shifted,
    )
    assert offset_2d == (-3, -2)
    assert aligned_2d.shape == image.shape
    assert torch.equal(aligned_2d, shifted)

    rgb = torch.stack([image, image * 2, image * 3])
    rgb_shifted = torch.roll(rgb, shifts=(2, 3), dims=(1, 2))
    offset_rgb, aligned_rgb = _tools.find_image_offset_phase_correlation_torch(
        rgb,
        rgb_shifted,
    )
    assert offset_rgb == (-3, -2)
    assert aligned_rgb.shape == rgb.shape
    assert torch.equal(aligned_rgb, rgb_shifted)


def test_manual_mapping_uses_documented_pixel_direction_on_integer_coordinates():
    adata = _visium_adata(np.array([[1, 2], [3, 4]], dtype=np.int64))

    result = _tools.map_spatial_manual(
        adata,
        offset=(10, -5),
        offset_mode="absolute",
    )

    assert result is adata
    assert np.allclose(adata.obsm["spatial1"], [[11, -3], [13, -1]])
    assert np.issubdtype(adata.obsm["spatial1"].dtype, np.floating)


def test_manual_mapping_preserves_legacy_scaling_explicitly():
    adata = _visium_adata(np.array([[2, 4], [4, 8]], dtype=np.int64))

    with pytest.warns(FutureWarning, match="offset_mode='legacy'"):
        _tools.map_spatial_manual(adata, offset=(2, -4), offset_mode="legacy")

    assert np.allclose(adata.obsm["spatial1"], [[0.5, 7.0], [2.5, 11.0]])


def test_manual_mapping_only_translates_the_selected_library():
    adata = _multilibrary_adata()

    result = _tools.map_spatial_manual(
        adata,
        offset=(10, -5),
        offset_mode="absolute",
        library_id="A",
        library_key="library",
    )

    assert np.allclose(result.obsm["spatial1"][:2], [[11, -5], [10, -4]])
    assert np.array_equal(result.obsm["spatial1"][2:], adata.obsm["spatial"][2:])


def test_manual_mapping_requires_explicit_selection_for_multiple_libraries():
    adata = _multilibrary_adata()

    with pytest.raises(ValueError, match="library_id"):
        _tools.map_spatial_manual(
            adata,
            offset=(1, 1),
            offset_mode="absolute",
        )


def test_auto_mapping_needs_no_magic_obs_column_or_cwd_temp_files(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'cv2', None)
    adata = _visium_adata(np.array([[1.0, 1.0], [2.0, 2.0]]))
    original_subplots = _tools.plt.subplots

    def small_subplots(*args, **kwargs):
        kwargs["figsize"] = (2, 2)
        return original_subplots(*args, **kwargs)

    def fake_embedding(data, *, ax, **kwargs):
        ax.scatter(data.obsm["spatial"][:, 0], data.obsm["spatial"][:, 1])
        return ax

    monkeypatch.setattr(_tools.plt, "subplots", small_subplots)
    monkeypatch.setattr(_tools.sc.pl, "embedding", fake_embedding)
    monkeypatch.setattr(
        _tools,
        "find_image_offset_phase_correlation_array_input",
        lambda image1, image2: ((0, 0), image1),
    )

    result = _tools.map_spatial_auto(adata, method="phase")

    assert result is adata
    assert "test" not in adata.obs
    assert "spatial1" in adata.obsm


@pytest.mark.parametrize('channels', [None, 1, 3, 4])
def test_real_phase_correlation_without_opencv(monkeypatch, channels):
    import sys
    monkeypatch.setitem(sys.modules, 'cv2', None)
    rng = np.random.default_rng(17)
    reference = rng.uniform(0, 255, (32, 40)).astype(np.float32)
    if channels is not None:
        reference = np.repeat(reference[..., None], channels, axis=2)
    moved = np.roll(reference, shift=(3, -5), axis=(0, 1))
    offset, aligned = _tools.find_image_offset_phase_correlation_array_input(moved, reference)
    assert offset == (-5, 3)
    assert aligned.shape == (32, 40)
    if channels in (None, 1):
        expected = reference if channels is None else reference[..., 0]
    else:
        expected = reference[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    np.testing.assert_allclose(aligned[:29, 5:], expected[:29, 5:], atol=1e-4)


def test_auto_mapping_real_phase_pipeline_without_opencv(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'cv2', None)
    coords = np.array([[1., 1.], [2., 3.], [4., 2.]])
    adata = _visium_adata(coords)
    result = _tools.map_spatial_auto(adata, method='phase')
    assert result is adata
    np.testing.assert_array_equal(result.obsm['spatial'], coords)
    assert result.obsm['spatial1'].shape == coords.shape
    assert np.isfinite(result.obsm['spatial1']).all()


def test_auto_mapping_moves_centered_coordinates_and_only_selected_library(monkeypatch):
    adata = _multilibrary_adata()
    original_subplots = _tools.plt.subplots

    def small_subplots(*args, **kwargs):
        kwargs["figsize"] = (2, 2)
        return original_subplots(*args, **kwargs)

    def fake_embedding(data, *, ax, **kwargs):
        ax.scatter(data.obsm["spatial"][:, 0], data.obsm["spatial"][:, 1])
        return ax

    monkeypatch.setattr(_tools.plt, "subplots", small_subplots)
    monkeypatch.setattr(_tools.sc.pl, "embedding", fake_embedding)
    monkeypatch.setattr(
        _tools,
        "find_image_offset_phase_correlation_array_input",
        lambda image1, image2: ((2, -1), image1),
    )

    result = _tools.map_spatial_auto(
        adata,
        method="phase",
        library_id="A",
        library_key="library",
    )

    assert not np.array_equal(result.obsm["spatial1"][:2], adata.obsm["spatial"][:2])
    assert np.array_equal(result.obsm["spatial1"][2:], adata.obsm["spatial"][2:])


def test_auto_mapping_renders_spots_in_the_selected_image_pixel_frame(monkeypatch):
    adata = _visium_adata(np.array([[10.0, 20.0], [30.0, 40.0]]))
    adata.uns["spatial"]["sample"]["scalefactors"]["tissue_hires_scalef"] = 0.2
    captured = {}
    original_subplots = _tools.plt.subplots

    def small_subplots(*args, **kwargs):
        kwargs["figsize"] = (2, 2)
        return original_subplots(*args, **kwargs)

    def fake_embedding(data, *, ax, **kwargs):
        captured["coords"] = data.obsm["spatial"].copy()
        ax.scatter(data.obsm["spatial"][:, 0], data.obsm["spatial"][:, 1])
        return ax

    monkeypatch.setattr(_tools.plt, "subplots", small_subplots)
    monkeypatch.setattr(_tools.sc.pl, "embedding", fake_embedding)
    monkeypatch.setattr(
        _tools,
        "find_image_offset_phase_correlation_array_input",
        lambda image1, image2: ((0, 0), image1),
    )

    _tools.map_spatial_auto(adata, method="phase")

    assert np.allclose(captured["coords"], adata.obsm["spatial"] * 0.2)


def test_crop_uses_documented_xy_width_height_order():
    adata = AnnData(np.ones((2, 1), dtype=np.float32))
    adata.obs_names = ["doc_expected", "transposed"]
    adata.obsm["spatial"] = np.array([[6.0, 2.0], [2.0, 6.0]])
    adata.uns["spatial"] = {
        "sample": {
            "images": {"hires": np.zeros((8, 10), dtype=np.uint8)},
            "scalefactors": {"tissue_hires_scalef": 1.0},
        }
    }

    cropped = _tools.crop_space_visium(
        adata,
        crop_loc=(5, 1),
        crop_area=(3, 3),
        library_id="sample",
        scale=1,
        coordinate_order="xy",
    )

    assert cropped.obs_names.tolist() == ["doc_expected"]
    assert np.allclose(cropped.obsm["spatial"], [[1.0, 1.0]])
    assert cropped.uns["spatial"]["sample"]["images"]["hires"].shape == (3, 3)


def test_crop_only_returns_observations_from_the_selected_library():
    adata = _multilibrary_adata()

    cropped = _tools.crop_space_visium(
        adata,
        crop_loc=(0, 0),
        crop_area=(3, 3),
        library_id="A",
        library_key="library",
        scale=1,
        coordinate_order="xy",
    )

    assert cropped.obs_names.tolist() == ["a1", "a2"]
    assert list(cropped.uns["spatial"]) == ["A"]


def test_crop_synchronizes_all_scaled_image_resolutions_and_drops_stale_images():
    adata = _multiresolution_adata()

    cropped = _tools.crop_space_visium(
        adata,
        crop_loc=(2, 0),
        crop_area=(4, 4),
        library_id="sample",
        scale=1,
        coordinate_order="xy",
    )

    images = cropped.uns["spatial"]["sample"]["images"]
    assert set(images) == {"hires", "lowres"}
    assert images["hires"].shape == (4, 4)
    assert images["lowres"].shape == (2, 2)
    assert cropped.obs_names.tolist() == ["selected"]
    assert np.allclose(cropped.obsm["spatial"], [[2, 2]])
    records = cropped.uns["spatial"]["sample"]["metadata"][
        "omicverse_removed_stale_images"
    ]
    record = records[sorted(records)[-1]]
    assert record["operation"] == "crop"
    assert record["image_keys"].tolist() == ["segmentation"]


def test_crop_resamples_fractional_secondary_origin_instead_of_shifting_it():
    adata = _multiresolution_adata()
    base = np.add.outer(
        np.arange(4, dtype=np.float16) * 10,
        np.arange(5, dtype=np.float16),
    )
    lowres = np.stack([base, base + 100, base + 200], axis=-1)
    adata.uns["spatial"]["sample"]["images"]["lowres"] = lowres

    cropped = _tools.crop_space_visium(
        adata,
        crop_loc=(1, 1),
        crop_area=(4, 4),
        library_id="sample",
        scale=1,
        coordinate_order="xy",
    )

    # At scale 0.5 the full-resolution origin (1, 1) is pixel (0.5, 0.5).
    # Bilinear sampling of this linear test image must therefore start at 5.5;
    # floor-slicing would incorrectly start at pixel (0, 0), value 0.
    cropped_lowres = cropped.uns["spatial"]["sample"]["images"]["lowres"]
    expected = np.stack(
        [
            np.array([[5.5, 6.5], [15.5, 16.5]], dtype=np.float16) + shift
            for shift in (0, 100, 200)
        ],
        axis=-1,
    )
    assert cropped_lowres.shape == (2, 2, 3)
    assert np.allclose(cropped_lowres, expected)


def test_stale_image_records_survive_h5ad_roundtrip(tmp_path):
    adata = _multiresolution_adata()
    cropped = _tools.crop_space_visium(
        adata,
        crop_loc=(2, 0),
        crop_area=(4, 4),
        library_id="sample",
        scale=1,
        coordinate_order="xy",
    )

    path = tmp_path / "cropped.h5ad"
    cropped.write_h5ad(path)
    import anndata as ad

    restored = ad.read_h5ad(path)
    records = restored.uns["spatial"]["sample"]["metadata"][
        "omicverse_removed_stale_images"
    ]
    record = records[sorted(records)[-1]]
    assert record["operation"] == "crop"
    assert record["image_keys"].tolist() == ["segmentation"]


def test_crop_and_subset_window_have_distinct_registry_entries():
    crop = get_registry().get_function("crop_space_visium")
    subset = get_registry().get_function("subset_window")

    assert crop is not None and subset is not None
    assert "crop_loc" in inspect.signature(crop).parameters
    assert "xlim" in inspect.signature(subset).parameters


def test_space_all_includes_the_public_tool_and_tissue_zone_surfaces():
    assert set(_tools.__all__).issubset(space.__all__)
    assert {"TissueZones", "nmf_tissue_zones"}.issubset(space.__all__)


def test_bin2cell_can_skip_the_optional_geometry_stack(monkeypatch):
    import omicverse.external.bin2cell as backend

    expected = AnnData(np.ones((1, 1), dtype=np.float32))
    monkeypatch.setattr(backend, "bin_to_cell", lambda *args, **kwargs: expected)

    result = _tools.bin2cell(
        _visium_adata([[1, 1]]),
        add_geometry=False,
        show_progress=False,
    )

    assert result is expected


def test_segmentation_wrappers_return_the_mutated_adata(monkeypatch, tmp_path):
    import omicverse.external.bin2cell as backend

    for name in (
        "destripe",
        "scaled_he_image",
        "cellseg",
        "insert_labels",
        "expand_labels",
        "grid_image",
        "salvage_secondary_labels",
    ):
        monkeypatch.setattr(backend, name, lambda *args, **kwargs: None)

    adata = _visium_adata([[1.0, 1.0]])
    adata.obsm["spatial_cropped_150_buffer"] = adata.obsm["spatial"].copy()
    adata.obs["n_counts_adjusted"] = [1]
    he_path = tmp_path / "he.tiff"
    gex_path = tmp_path / "gex.tiff"

    assert _tools.visium_10x_hd_cellpose_he(adata, he_save_path=str(he_path)) is adata
    assert _tools.visium_10x_hd_cellpose_expand(adata) is adata
    assert _tools.visium_10x_hd_cellpose_gex(adata, gex_save_path=str(gex_path)) is adata
    assert _tools.salvage_secondary_labels(adata) is adata

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from omicverse.io.spatial import _visium


@pytest.mark.parametrize('format_name', ['legacy', 'csv', 'parquet'])
def test_real_10x_count_and_position_files_match_reference(format_name, tmp_path):
    import json
    import scanpy as sc
    from matplotlib.image import imsave
    root = tmp_path / 'outs'
    spatial = root / 'spatial'
    spatial.mkdir(parents=True)
    with h5py.File(root / 'filtered_feature_bc_matrix.h5', 'w') as f:
        f.attrs['library_ids'] = np.array([b'sample'])
        m = f.create_group('matrix')
        for name, values in {'data': [1, 2], 'indices': [0, 0], 'indptr': [0, 1, 2], 'shape': [1, 2]}.items():
            m.create_dataset(name, data=np.asarray(values, dtype=np.int64))
        m.create_dataset('barcodes', data=np.array([b'bc1', b'bc2']))
        features = m.create_group('features')
        for name, values in {'name': [b'G'], 'id': [b'idG'], 'feature_type': [b'Gene Expression'], 'genome': [b'mouse']}.items():
            features.create_dataset(name, data=np.array(values))
    table = pd.DataFrame([['bc1', 1, 0, 0, 100, 200], ['bc2', 1, 0, 1, 110, 220]],
        columns=['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres'])
    table.to_csv(spatial / 'tissue_positions.csv', index=False)
    for res in ('hires', 'lowres'):
        imsave(spatial / f'tissue_{res}_image.png', np.zeros((8, 8, 3)))
    (spatial / 'scalefactors_json.json').write_text(json.dumps({'tissue_hires_scalef': 0.02, 'tissue_lowres_scalef': 0.02}), encoding='utf-8')
    reference = sc.read_visium(root)
    if format_name == 'legacy':
        (spatial / 'tissue_positions.csv').unlink()
        table.to_csv(spatial / 'tissue_positions_list.csv', index=False, header=False)
    elif format_name == 'parquet':
        pytest.importorskip('pyarrow')
        table.to_parquet(spatial / 'tissue_positions.parquet', index=False)
    result = _visium.read_visium(root)
    np.testing.assert_array_equal(result.obsm['spatial'], reference.obsm['spatial'])
    np.testing.assert_array_equal(result.X.toarray(), reference.X.toarray())
    assert result.obs.loc['bc1', 'pxl_row_in_fullres'] == 100


def test_legacy_tissue_positions_keeps_first_barcode_and_xy_order(monkeypatch, tmp_path):
    root = tmp_path / "outs"
    spatial = root / "spatial"
    spatial.mkdir(parents=True)
    count_path = root / "filtered_feature_bc_matrix.h5"
    with h5py.File(count_path, "w") as handle:
        handle.attrs["library_ids"] = np.array([b"sample"])

    (spatial / "tissue_positions_list.csv").write_text(
        "bc1,1,0,0,100,200\n"
        "bc2,1,0,1,110,220\n",
        encoding="utf-8",
    )

    counts = AnnData(np.ones((2, 1), dtype=np.float32))
    counts.obs_names = ["bc1", "bc2"]
    monkeypatch.setattr(_visium, "read_10x_h5", lambda *args, **kwargs: counts.copy())

    adata = _visium.read_visium(root)

    assert adata.obs["in_tissue"].notna().all()
    assert adata.obs.loc["bc1", "pxl_row_in_fullres"] == 100
    assert adata.obs.loc["bc1", "pxl_col_in_fullres"] == 200
    assert np.array_equal(adata.obsm["spatial"], [[200, 100], [220, 110]])


def test_modern_tissue_positions_csv_uses_the_same_xy_order(monkeypatch, tmp_path):
    root = tmp_path / "outs"
    spatial = root / "spatial"
    spatial.mkdir(parents=True)
    count_path = root / "filtered_feature_bc_matrix.h5"
    with h5py.File(count_path, "w") as handle:
        handle.attrs["library_ids"] = np.array([b"sample"])

    (spatial / "tissue_positions.csv").write_text(
        "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n"
        "bc1,1,0,0,100,200\n"
        "bc2,1,0,1,110,220\n",
        encoding="utf-8",
    )

    counts = AnnData(np.ones((2, 1), dtype=np.float32))
    counts.obs_names = ["bc1", "bc2"]
    monkeypatch.setattr(_visium, "read_10x_h5", lambda *args, **kwargs: counts.copy())

    adata = _visium.read_visium(root)

    assert np.array_equal(adata.obsm["spatial"], [[200, 100], [220, 110]])

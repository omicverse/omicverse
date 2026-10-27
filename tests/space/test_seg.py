"""``ov.space.seg`` on synthetic images and molecule tables.

Ground truth is drawn, not borrowed: discs of known centre and radius, some
pairs touching, molecules scattered inside known discs. Deep-learning backends
are exercised through stand-ins with their real call signatures; the tiling,
stitching, assignment, counting and metrics code is all real.
"""
from __future__ import annotations

import sys
import textwrap
import types

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from skimage.measure import label

from omicverse.space import seg
from omicverse.space.seg._backends import BACKENDS, _masks_to_labels


def _discs(H, W, centres, radius):
    yy, xx = np.mgrid[:H, :W]
    truth = np.zeros((H, W), np.int32)
    for i, (cy, cx) in enumerate(centres, start=1):
        truth[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = i
    return truth


def _grid(H, W, spacing, offset):
    return [(y, x) for y in range(offset, H - offset + 1, spacing)
            for x in range(offset, W - offset + 1, spacing)]


def _one_label_per_object(truth, labels):
    """Every true object is covered by exactly one label, and no label spans two."""
    labels = labels.toarray() if sparse.issparse(labels) else labels
    bad = 0
    for i in range(1, truth.max() + 1):
        ids = np.unique(labels[truth == i])
        if len(ids) != 1 or ids[0] == 0:
            bad += 1
    owners = pd.Series(truth[labels > 0]).groupby(labels[labels > 0]).nunique()
    return bad, int((owners > 1).sum())


def _majority_matches(truth, labels):
    """True objects whose majority label is a distinct, non-background label."""
    labels = labels.toarray() if sparse.issparse(labels) else labels
    major = [int(np.bincount(labels[truth == i]).argmax()) for i in range(1, truth.max() + 1)]
    major = [m for m in major if m]
    return len(set(major)), len(major) - len(set(major))


# ---------------------------------------------------------------------------
# tiling engine
# ---------------------------------------------------------------------------
def test_tiled_stitching_keeps_every_object_whole():
    truth = _discs(1200, 1100, _grid(1200, 1100, 40, 20), 12)
    image = (truth > 0).astype(np.float32)
    calls = []

    def threshold_segmenter(tile):
        calls.append(tile.shape)
        return label(tile > 0.5)

    dense = seg.segment_tiled(image, threshold_segmenter, tile_size=256, context=32,
                              show_progress=False)
    assert len(calls) == 5 * 5  # ceil(1200/256) x ceil(1100/256)
    assert dense.max() == truth.max()
    assert set(np.unique(dense)) == set(range(truth.max() + 1))
    assert _one_label_per_object(truth, dense) == (0, 0)

    sp = seg.segment_tiled(image, threshold_segmenter, tile_size=256, context=32,
                           sparse=True, show_progress=False)
    assert sparse.issparse(sp)
    np.testing.assert_array_equal(sp.toarray(), dense)


def test_tiled_min_size_and_empty_tiles():
    truth = _discs(300, 300, [(50, 50), (200, 200)], 10)
    truth[150, 150] = 99  # a one-pixel speck
    out = seg.segment_tiled(truth, lambda t: label(t > 0), tile_size=128, context=24,
                            min_size=5, show_progress=False)
    assert out.max() == 2
    blank = seg.segment_tiled(np.zeros((100, 100)), lambda t: np.zeros(t.shape, int),
                              tile_size=64, context=8, show_progress=False)
    assert blank.max() == 0 and blank.dtype == np.int32


def test_bin2cell_cellseg_has_no_seam_artifacts(tmp_path, monkeypatch):
    """Regression: the old IoU paste summed label ids where tiles overlapped."""
    import tifffile

    truth = _discs(1200, 1200, _grid(1200, 1200, 40, 20), 12)
    tifffile.imwrite(tmp_path / "img.tiff", ((truth > 0) * 200).astype(np.uint8))
    seen = {}

    class FakeCellpose:
        def __init__(self, gpu=False, pretrained_model="cpsam", **kw):
            seen["init"] = dict(gpu=gpu, pretrained_model=pretrained_model)

        def eval(self, x, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
                 normalize=True, channel_axis=None):
            seen["eval"] = dict(flow_threshold=flow_threshold,
                                cellprob_threshold=cellprob_threshold, normalize=normalize)
            return label(x > 0.5), None, None

    fake = types.ModuleType("cellpose")
    fake.models = types.SimpleNamespace(CellposeModel=FakeCellpose)
    monkeypatch.setitem(sys.modules, "cellpose", fake)
    monkeypatch.setitem(sys.modules, "cellpose.models", fake.models)

    from omicverse.external.bin2cell import cellseg
    cellseg(str(tmp_path / "img.tiff"), str(tmp_path / "labels.npz"),
            stardist_model="2D_versatile_fluo", block_size=512, min_overlap=128,
            context=32, prob_thresh=0.01, flow_threshold=0.3, show_progress=False)
    labels = sparse.load_npz(tmp_path / "labels.npz")
    assert labels.shape == truth.shape
    assert labels.max() == truth.max()
    assert _one_label_per_object(truth, labels) == (0, 0)
    assert seen["eval"] == dict(flow_threshold=0.3, cellprob_threshold=0.01, normalize=False)


# ---------------------------------------------------------------------------
# watershed backend (no extra dependency, runs for real)
# ---------------------------------------------------------------------------
def _touching_scene(seed=0):
    rng = np.random.default_rng(seed)
    centres = [(y + rng.integers(-3, 4), x + rng.integers(-3, 4))
               for y, x in _grid(600, 600, 60, 40)]
    # every third object gets a neighbour it touches
    centres += [(y, x + 20) for y, x in centres[::3]]
    truth = _discs(600, 620, centres, 11)
    return truth, len(centres)


def test_watershed_fluorescence_splits_touching_nuclei():
    from scipy.ndimage import gaussian_filter

    truth, n = _touching_scene()
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[:truth.shape[0], :truth.shape[1]]
    img = np.zeros(truth.shape, np.float32)
    from skimage.measure import regionprops
    for r in regionprops(truth):  # dome-shaped nuclei: brightest at the centre
        cy, cx = r.centroid
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        img = np.maximum(img, np.where(truth == r.label, 1.0 - d2 / (2 * 11 ** 2), 0))
    img = gaussian_filter(img, 1.0) + rng.normal(0, 0.03, img.shape)
    img = (np.clip(img, 0, 1) * 4000).astype(np.uint16)

    labels = seg.segment(img, method="watershed", diameter=22, show_progress=False)
    assert labels.shape == truth.shape and labels.dtype == np.int32
    assert abs(labels.max() - n) <= 0.05 * n
    distinct, shared = _majority_matches(truth, labels)
    assert distinct >= 0.9 * n and shared <= 0.05 * n  # touching pairs are split, not merged


def test_watershed_on_he_uses_the_haematoxylin_channel():
    truth, n = _touching_scene(seed=3)
    rgb = np.empty(truth.shape + (3,), np.uint8)
    rgb[:] = (236, 196, 214)                       # eosin-pink background
    rgb[truth > 0] = (88, 58, 142)                 # haematoxylin-purple nuclei
    rgb = np.clip(rgb + np.random.default_rng(2).normal(0, 6, rgb.shape), 0, 255).astype(np.uint8)

    labels = seg.segment(rgb, method="watershed", image_type="auto", diameter=22,
                         show_progress=False)
    assert abs(labels.max() - n) <= 0.1 * n
    assert (labels[truth == 0] > 0).mean() < 0.02  # background stays background


def test_segment_tiles_large_images_and_expands(tmp_path):
    import tifffile

    truth = _discs(900, 900, _grid(900, 900, 50, 25), 10)
    img = ((truth > 0) * 3000).astype(np.uint16)
    tifffile.imwrite(tmp_path / "dapi.tif", img)
    small = seg.segment(str(tmp_path / "dapi.tif"), method="watershed", diameter=20,
                        tile_size=200, show_progress=False)
    assert small.max() == truth.max()
    assert _majority_matches(truth, small) == (truth.max(), 0)  # no seam splits or merges
    grown = seg.segment(img, method="watershed", diameter=20, tile_size=200, expand=5,
                        sparse=True, show_progress=False)
    assert sparse.issparse(grown)
    assert grown.nnz > (small > 0).sum()


def test_unknown_method_and_missing_dependency(monkeypatch):
    with pytest.raises(ValueError, match="Unknown method"):
        seg.segment(np.zeros((32, 32)), method="baysor")
    monkeypatch.setitem(sys.modules, "stardist", None)
    with pytest.raises(ImportError, match="pip install stardist"):
        seg.segment(np.zeros((64, 64)), method="stardist", show_progress=False)


def test_methods_table_lists_every_backend():
    table = seg.methods()
    assert set(BACKENDS) | {"baysor"} == set(table.index)
    assert table.loc["watershed", "installed"]
    assert {"input", "needs", "gpu", "installed", "reference"} <= set(table.columns)


# ---------------------------------------------------------------------------
# SAM-family mask painting
# ---------------------------------------------------------------------------
def test_masks_to_labels_drops_background_and_keeps_small_over_large():
    H = W = 64
    yy, xx = np.mgrid[:H, :W]
    tissue = np.ones((H, W), bool)
    blob = (yy - 30) ** 2 + (xx - 30) ** 2 <= 15 ** 2
    nucleus = (yy - 30) ** 2 + (xx - 30) ** 2 <= 5 ** 2
    other = (yy - 55) ** 2 + (xx - 55) ** 2 <= 4 ** 2
    masks = [{"segmentation": m, "area": int(m.sum())} for m in (tissue, blob, nucleus, other)]
    out = _masks_to_labels(masks, min_area=10, max_area=1000)
    assert out.max() == 3
    assert len(np.unique(out[nucleus])) == 1 and out[30, 30] != out[30, 42]
    assert out[0, 0] == 0  # the whole-tile mask was dropped
    assert _masks_to_labels([]) is None


def test_sam_backend_uses_generator_output(monkeypatch):
    class FakeAMG:
        def __init__(self, model, **kw):
            self.kw = kw

        def generate(self, rgb):
            assert rgb.dtype == np.uint8 and rgb.shape[-1] == 3
            m = np.zeros(rgb.shape[:2], bool)
            m[10:20, 10:20] = True
            return [{"segmentation": m, "area": 100}]

    fake = types.ModuleType("segment_anything")
    fake.SamAutomaticMaskGenerator = FakeAMG
    fake.sam_model_registry = {"vit_b": lambda checkpoint: types.SimpleNamespace(to=lambda d: None)}
    monkeypatch.setitem(sys.modules, "segment_anything", fake)
    labels = seg.segment(np.random.default_rng(0).random((64, 64)), method="sam",
                         model="vit_b", checkpoint="unused.pth", max_area=1000,
                         show_progress=False)
    assert labels.max() == 1 and (labels > 0).sum() == 100


# ---------------------------------------------------------------------------
# molecules
# ---------------------------------------------------------------------------
def _molecules(truth, pixel_size, offset, per_cell=40, background=300, seed=0):
    """Molecules inside each disc; cell type (odd/even label) decides the genes."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, truth.max() + 1):
        ys, xs = np.nonzero(truth == i)
        pick = rng.integers(0, len(ys), per_cell)
        genes = rng.choice(["A1", "A2", "A3"] if i % 2 else ["B1", "B2", "B3"], per_cell)
        rows.append(pd.DataFrame({
            "x": (xs[pick] + rng.random(per_cell)) * pixel_size + offset[0],
            "y": (ys[pick] + rng.random(per_cell)) * pixel_size + offset[1],
            "gene": genes, "true_cell": i}))
    ys, xs = np.nonzero(truth == 0)
    pick = rng.integers(0, len(ys), background)
    rows.append(pd.DataFrame({
        "x": (xs[pick] + rng.random(background)) * pixel_size + offset[0],
        "y": (ys[pick] + rng.random(background)) * pixel_size + offset[1],
        "gene": rng.choice(["A1", "B1", "N1"], background), "true_cell": 0}))
    return pd.concat(rows, ignore_index=True)


def test_assign_and_aggregate_round_trip():
    truth = _discs(200, 240, _grid(200, 240, 40, 20), 12)
    px, off = 0.5, (100.0, -20.0)
    tx = _molecules(truth, px, off)
    tx.loc[len(tx)] = [1e6, 1e6, "A1", 0]   # far outside the image

    cell = seg.assign_transcripts(tx, truth, pixel_size=px, offset=off)
    np.testing.assert_array_equal(cell, tx["true_cell"].to_numpy())
    np.testing.assert_array_equal(
        seg.assign_transcripts(tx, sparse.csr_matrix(truth), pixel_size=px, offset=off), cell)
    affine = np.array([[1 / px, 0, -off[0] / px], [0, 1 / px, -off[1] / px], [0, 0, 1]])
    np.testing.assert_array_equal(seg.assign_transcripts(tx, truth, transform=affine), cell)

    adata = seg.aggregate(tx.assign(cell=cell), labels=truth, pixel_size=px, offset=off)
    assert adata.n_obs == truth.max()
    assert adata.obs["n_transcripts"].eq(40).all()
    assert set(adata.var_names) == {"A1", "A2", "A3", "B1", "B2", "B3"}
    odd = adata.obs_names.astype(int) % 2 == 1
    assert adata[odd, ["B1", "B2", "B3"]].X.sum() == 0
    np.testing.assert_allclose(adata.obs["area"], np.bincount(truth.ravel())[1:] * px ** 2)

    from shapely import wkt
    from shapely.geometry import Point
    for (cx, cy), poly in zip(adata.obsm["spatial"], adata.obs["geometry"]):
        shape = wkt.loads(poly)
        assert shape.contains(Point(cx, cy))
        assert abs(shape.area - np.pi * (12 * px) ** 2) < 0.15 * np.pi * (12 * px) ** 2
    assert "seg" in adata.uns["spatial"]


def test_aggregate_with_counts_and_string_cells():
    tx = pd.DataFrame({"x": [0, 1, 2, 3, 4], "y": [0, 0, 0, 0, 0],
                       "gene": ["g1", "g1", "g2", "g2", "g1"],
                       "cell": ["c1", "c1", "c2", "", np.nan],
                       "MIDCount": [2, 3, 1, 5, 5]})
    adata = seg.aggregate(tx, count="MIDCount")
    assert list(adata.obs_names) == ["c1", "c2"]
    assert adata["c1", "g1"].X.toarray().item() == 5
    assert adata.obs.loc["c2", "n_transcripts"] == 1
    with pytest.raises(ValueError, match="not integer label ids"):
        seg.aggregate(tx, labels=np.zeros((5, 5), int))


def test_evaluate_and_compare_rank_a_better_segmentation_first():
    from scipy.ndimage import binary_erosion
    from skimage.morphology import disk

    truth = _discs(240, 240, _grid(240, 240, 40, 20), 12)
    tx = _molecules(truth, 1.0, (0.0, 0.0), per_cell=60, background=0)
    eroded = truth * binary_erosion(truth > 0, disk(5))      # under-segmented cells
    res = {}
    for name, lab in {"good": truth, "eroded": eroded}.items():
        res[name] = seg.evaluate(lab, tx.assign(cell=seg.assign_transcripts(tx, lab)))

    good = res["good"]
    assert good.attrs["pct_assigned"] == pytest.approx(100.0)
    assert good["n_transcripts"].eq(60).all() and good["n_genes"].le(3).all()
    assert good["circularity"].median() > 0.85 and good["solidity"].min() > 0.9
    assert good["elongation"].max() < 1.1 and good["sphericity"].median() > 0.85
    assert res["eroded"].attrs["pct_assigned"] < 80

    table = seg.compare(res, min_transcripts=20, min_genes=1)
    assert list(table.index) == ["good", "eroded"]
    assert table.loc["good", "rank"] == 1
    assert {"n_cells", "n_hq_cells", "pct_assigned", "area", "circularity", "overall"} <= set(table)

    morph_only = seg.evaluate(truth)
    assert "n_transcripts" not in morph_only and len(morph_only) == truth.max()


FAKE_BAYSOR = textwrap.dedent("""\
    #!{python}
    import sys, json, pandas as pd
    args = sys.argv[1:]
    json.dump(args, open("argv.json", "w"))
    out = args[args.index("-o") + 1]
    csv = [a for a in args if a.endswith(".csv") and a != out][0]
    tx = pd.read_csv(csv)
    tx["molecule_id"] = range(1, len(tx) + 1)
    tx["cell"] = ["CRx-" + str(int(v) // 10) if v % 10 else "" for v in tx["x"]]
    tx["is_noise"] = (tx["x"] % 10 == 0)
    tx["assignment_confidence"] = 0.9
    tx.to_csv(out, index=False)
""")


def test_segment_transcripts_with_baysor_cli(tmp_path):
    exe = tmp_path / "baysor"
    exe.write_text(FAKE_BAYSOR.format(python=sys.executable))
    exe.chmod(0o755)
    tx = pd.DataFrame({"x": np.arange(40, dtype=float), "y": 0.0,
                       "gene": ["g%d" % (i % 3) for i in range(40)],
                       "nucleus": [i // 10 if i % 2 else 0 for i in range(40)]})
    work = tmp_path / "work"
    out = seg.segment_transcripts(tx, gene="gene", prior="nucleus", prior_confidence=0.6,
                                  scale=5, executable=str(exe), workdir=work)
    argv = pd.read_json(work / "argv.json", typ="series").tolist()
    assert argv[0] == "run" and argv[-1] == ":nucleus"
    assert argv[argv.index("-s") + 1] == "5" and argv[argv.index("-g") + 1] == "gene"
    assert argv[argv.index("--prior-segmentation-confidence") + 1] == "0.6"
    assert out["cell"].dtype.kind == "i"
    assert (out.loc[out["x"] % 10 == 0, "cell"] == 0).all()
    assert out.loc[out["x"] % 10 != 0, "cell"].nunique() == 4
    assert out["is_noise"].sum() == 4 and "cell_confidence" in out
    adata = seg.aggregate(out)
    assert adata.n_obs == 4 and adata.obs["n_transcripts"].eq(9).all()

    with pytest.raises(ValueError, match="needs `scale`"):
        seg.segment_transcripts(tx, executable=str(exe))
    with pytest.raises(FileNotFoundError, match="Baysor executable"):
        seg.segment_transcripts(tx, scale=5, executable="definitely-not-baysor")


# ---------------------------------------------------------------------------
# Visium HD wrappers route method + method-aware thresholds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method, he_expected, gex_expected", [
    ("cellpose", {"prob_thresh": 0, "flow_threshold": 0.2}, {"prob_thresh": 0.01}),
    ("stardist", {"prob_thresh": 0.01}, {"prob_thresh": 0.05, "nms_thresh": 0.5}),
    ("sam2", {}, {}),
])
def test_visium_hd_wrappers_forward_method(monkeypatch, tmp_path, method, he_expected,
                                           gex_expected):
    import anndata as ad
    import omicverse.external.bin2cell as b2c
    import omicverse as ov

    calls = []
    monkeypatch.setattr(b2c, "cellseg", lambda **kw: calls.append(kw))
    for name in ("destripe", "scaled_he_image", "insert_labels", "grid_image"):
        monkeypatch.setattr(b2c, name, lambda *a, **k: None)
    adata = ad.AnnData(np.zeros((3, 2)))
    adata.obs["n_counts_adjusted"] = 1.0
    he = tmp_path / "he.tiff"
    gex = tmp_path / "gex.tiff"
    ov.space.visium_10x_hd_cellpose_he(adata, he_save_path=str(he), method=method)
    ov.space.visium_10x_hd_cellpose_gex(adata, gex_save_path=str(gex), method=method)
    he_call, gex_call = calls
    keys = {"prob_thresh", "flow_threshold", "nms_thresh"}
    assert he_call["backend"] == gex_call["backend"] == method
    assert {k: v for k, v in he_call.items() if k in keys} == he_expected
    assert {k: v for k, v in gex_call.items() if k in keys} == gex_expected
    assert he_call["stardist_model"] == "2D_versatile_he"
    assert gex_call["stardist_model"] == "2D_versatile_fluo"

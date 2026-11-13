"""Tests for ov.tl smear morphometry and the two-axis (shape x stain) classifier."""
import numpy as np
import pandas as pd
import pytest

from omicverse.tl import _smear_morph_core as core

VIOLET = np.array([0.90, 0.39, 0.19])   # OD direction, red-absorbing dye
PINK = np.array([0.06, 0.97, 0.21])


def _paint(img, mask, od_dir, amount):
    od = amount * od_dir
    rgb = 245.0 * 10 ** (-od)
    img[mask] = rgb


def _field(rng, shape, dye, n=6, size=200):
    img = np.full((size, size, 3), 245.0)
    img += rng.normal(0, 1.5, img.shape)
    yy, xx = np.mgrid[:size, :size]
    boxes = []
    for k in range(n):
        cx, cy = 30 + (k % 3) * 60, 40 + (k // 3) * 90
        if shape == "round":      # chain of 3 round cells, radius 4
            m = np.zeros((size, size), bool)
            for j in range(3):
                m |= (xx - (cx + 8 * j)) ** 2 + (yy - cy) ** 2 <= 16
        elif shape == "rod":      # one rod 24 x 7
            m = (np.abs(xx - cx - 8) <= 12) & (np.abs(yy - cy) <= 3.5)
        else:                     # big oval
            m = ((xx - cx - 8) / 14.0) ** 2 + ((yy - cy) / 10.0) ** 2 <= 1
        _paint(img, m, VIOLET if dye == "v" else PINK, 0.8)
        ys, xs = np.nonzero(m)
        boxes.append((xs.min() - 1, ys.min() - 1, xs.max() + 1, ys.max() + 1))
    return np.clip(img, 0, 255), boxes


def test_measure_objects_shape_and_colour():
    rng = np.random.default_rng(0)
    img_r, b_r = _field(rng, "round", "v")
    img_b, b_b = _field(rng, "rod", "p")
    vec = np.vstack([VIOLET, PINK])
    r = core.measure_objects(img_r, b_r, 0.1, stain_vectors=vec)
    b = core.measure_objects(img_b, b_b, 0.1, stain_vectors=vec)
    assert len(r) == len(b_r)
    for rec in r + b:
        assert rec["long_axis_um"] >= rec["short_axis_um"] >= 0.2
        assert np.isfinite(rec["stain_index"])
    # chains of round cells repeat every cell; rods do not -> lower per-cell aspect
    assert np.median([x["cell_aspect"] for x in r]) < np.median([x["cell_aspect"] for x in b])
    assert np.median([x["stain_primary_fraction"] for x in r]) > 0.8
    assert np.median([x["stain_primary_fraction"] for x in b]) < 0.2


def test_estimate_stain_vectors_orders_primary_first():
    rng = np.random.default_rng(1)
    a = rng.uniform(0.1, 1.0, (2000, 2))
    od = a[:, :1] * VIOLET + a[:, 1:] * PINK
    v = core.estimate_stain_vectors(od)
    assert v.shape == (2, 3)
    assert v[0] @ VIOLET / np.linalg.norm(VIOLET) > 0.95
    assert v[1] @ PINK / np.linalg.norm(PINK) > 0.95


def test_group_laplace_prior_counts_groups():
    ctx = ["round"] * 30 + ["rod"] * 30
    out = ["v"] * 30 + ["p"] * 20 + ["v"] * 10
    grp = [f"g{i // 10}" for i in range(60)]
    pr = core.group_laplace_prior(ctx, out, grp, alpha=1.0)
    assert pr["round"]["v"] == pytest.approx(4 / 5)    # 3 groups v, Laplace over 2 outcomes
    assert pr["rod"]["p"] == pytest.approx(3 / 5)
    assert pr["round"]["p"] > 0


def test_balanced_weights_equalise_groups_and_classes():
    y = np.array([0] * 8 + [1] * 2)
    g = np.array(["a"] * 6 + ["b"] * 2 + ["c"] * 2)
    w = core.balanced_weights(y, g)
    assert w[y == 0].sum() == pytest.approx(w[y == 1].sum())
    assert w[g == "a"].sum() == pytest.approx(w[g == "b"].sum())


def _cohort(rng, specs, per=4):
    rows, labels, groups = [], [], []
    for gi, (lab, shape, dye) in enumerate(specs):
        for _ in range(per):
            img, boxes = _field(rng, shape, dye)
            recs = core.measure_objects(img, boxes, 0.1, stain_vectors=np.vstack([VIOLET, PINK]))
            rows.append(core.field_features(recs))
            labels.append(lab)
            groups.append(f"{lab}-{gi}")
    return pd.DataFrame(rows), np.array(labels), np.array(groups)


def test_two_axis_classifier_uses_shape_prior_against_colour():
    rng = np.random.default_rng(2)
    specs = [("A_round", "round", "v")] * 3 + [("A_rod", "rod", "v")] * 2 + \
            [("B_rod", "rod", "p")] * 3 + [("big", "big", "v")] * 2
    X, y, g = _cohort(rng, specs)
    axes = {"A_round": ("round", "A"), "A_rod": ("rod", "A"), "B_rod": ("rod", "B"),
            "B_round": ("round", "B"), "big": ("big", None)}
    m = core.fit_two_axis_classifier(X, y, g, axes, positive_stain="A")
    assert m["prior"]["round"]["A"] > m["prior"]["rod"]["A"]
    oof = core.group_holdout_two_axis(X, y, g, label_axes=axes, positive_stain="A")
    per, macro = core.species_balanced_recall(y, oof, g)
    assert macro > 0.7
    # never emits a label absent from training
    Xn, _, _ = _cohort(rng, [("B_round", "round", "p")])
    pred = core.predict_two_axis(m, Xn)
    assert "B_round" not in set(pred)


def test_source_is_self_contained(tmp_path):
    import omicverse as ov
    p = tmp_path / "core.py"
    text = ov.tl.smear_morphometry_source(p)
    assert "import omicverse" not in text and "from omicverse" not in text
    assert p.read_text() == text


def test_measure_box_tables_roundtrip(tmp_path):
    from PIL import Image
    rng = np.random.default_rng(3)
    img, boxes = _field(rng, "rod", "p")
    (tmp_path / "f").mkdir()
    Image.fromarray(img.astype(np.uint8)).save(tmp_path / "f" / "a.png")
    fields = pd.DataFrame({"field_id": ["a"], "image_path": ["/elsewhere/f/a.png"], "um_per_px": [0.1]})
    objs = pd.DataFrame([{"organism_id": f"o{i}", "field_id": "a", "x_min": b[0], "y_min": b[1],
                          "x_max": b[2], "y_max": b[3]} for i, b in enumerate(boxes)])
    o, f, vec = core.measure_box_tables(fields, objs, root=str(tmp_path),
                                        stain_vectors=np.vstack([VIOLET, PINK]))
    assert list(o.organism_id) == list(objs.organism_id)
    assert f.shape[0] == 1 and "cell_aspect_q50" in f.columns


def test_fit_and_export_predictor_runs_without_omicverse(tmp_path):
    import json
    from PIL import Image
    import omicverse as ov
    rng = np.random.default_rng(4)
    specs = [("A_round", "round", "v")] * 3 + [("A_rod", "rod", "v")] * 3 + [("B_rod", "rod", "p")] * 3
    data = tmp_path / "data"
    (data / "fields").mkdir(parents=True)
    frows, orows, lrows = [], [], []
    for gi, (lab, shape, dye) in enumerate(specs):
        for k in range(2):
            fid = f"f{gi}_{k}"
            img, boxes = _field(rng, shape, dye)
            Image.fromarray(img.astype(np.uint8)).save(data / "fields" / f"{fid}.png")
            frows.append({"field_id": fid, "image_path": f"fields/{fid}.png", "um_per_px": 0.1})
            lrows.append({"field_id": fid, "culture_class": lab, "group_id": f"g{gi}"})
            for j, b in enumerate(boxes):
                orows.append({"organism_id": f"{fid}_o{j}", "field_id": fid, "x_min": b[0],
                              "y_min": b[1], "x_max": b[2], "y_max": b[3]})
    fields, objs, labels = pd.DataFrame(frows), pd.DataFrame(orows), pd.DataFrame(lrows)
    fields.to_csv(data / "fields.csv", index=False)
    objs.to_csv(data / "organisms.csv", index=False)
    axes = {"A_round": ("round", "A"), "A_rod": ("rod", "A"), "B_rod": ("rod", "B"),
            "B_round": ("round", "B")}
    model, rep = ov.tl.smear_fit_predictor(fields, objs, labels, label_col="culture_class",
                                           group_col="group_id", label_axes=axes,
                                           positive_stain="A", root=str(data))
    assert rep["n_groups"] == 9 and "logo_species_balanced_recall" in rep
    res = ov.tl.smear_export_predictor(model, tmp_path / "sub", check_input_dir=str(data),
                                       n_jobs=1, time_limit=600)
    assert res["check_returncode"] == 0
    assert res["fields_covered"] and res["objects_covered"] and res["all_finite"]
    assert set(res["labels_emitted"]) <= {"A_round", "A_rod", "B_rod"}
    assert "two_axis" not in (tmp_path / "sub" / "predict.py").read_text()
    cfg = json.loads((tmp_path / "sub" / "predictor_config.json").read_text())
    assert cfg["object_out_cols"]["stain"] == "stain_index"

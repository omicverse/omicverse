r"""Stained-smear / bright-field object morphometry and unseen-group field calls (``ov.tl``).

Registered entry points around :mod:`omicverse.tl._smear_morph_core` (a
numpy / scipy / scikit-image / scikit-learn-only module that a stand-alone
predictor can embed verbatim, see :func:`smear_morphometry_source`):

``smear_measure``               per-object size, per-cell shape and two-dye
                                colour for boxed objects of many images, plus
                                per-image (field) summaries.
``smear_stain_vectors``         label-free two-dye optical-density directions.
``smear_field_features``        robust per-field summaries of object tables.
``two_axis_classifier_fit``     field classifier with separate morphology
                                (stain-free) and stain evidence, a group-level
                                prior P(stain | shape), group-balanced weights.
``two_axis_classifier_predict`` apply it.
``group_holdout_two_axis``      leave-one-group-out predictions of it.
``species_balanced_recall``     per-class recall averaged over held-out groups.
``group_laplace_prior``         P(outcome | context) counted over groups.
``smear_morphometry_source``    the core's source text for a portable predictor.
``smear_fit_predictor``         tables -> measured, leave-one-group-out validated, fitted model.
``smear_export_predictor``      write predict.py + core + model (+ offline self-check).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .._registry import register_function
from . import _smear_morph_core as _core

__all__ = ["smear_measure", "smear_stain_vectors", "smear_field_features",
           "two_axis_classifier_fit", "two_axis_classifier_predict", "group_holdout_two_axis",
           "species_balanced_recall", "group_laplace_prior", "smear_morphometry_source",
           "smear_fit_predictor", "smear_export_predictor"]


@register_function(
    aliases=["涂片形态测量", "革兰染色形态", "明场目标形态", "smear morphometry", "stained object morphometry",
             "bright-field object measurement", "microbe morphology from boxes", "gram stain measurement",
             "per-cell shape descriptors", "two-dye stain unmixing"],
    category="utils",
    description="Measure every boxed object in transmitted-light images of stained smears / slides "
                "(bacteria, yeasts, parasites, cells): optical density against each image's white "
                "reference, Otsu object mask clipped to the box, ellipse long/short axes (um), area, "
                "solidity, fill; per-cell shape independent of how many cells a box holds (medial-axis "
                "width, autocorrelation repeat length and repeat/width aspect, constriction-count cell "
                "length and aspect, distance-transform round-piece fraction); two-dye colour by "
                "unmixing onto dye vectors estimated label-free from the cohort (primary-dye fraction, "
                "amounts, background-subtracted versions; keys start with 'stain_'). Returns the object "
                "table, per-field quantile summaries and the dye vectors.",
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "objs, fields_X, vec = ov.tl.smear_measure(fields_df, organisms_df, root='/data', n_jobs=4)",
        "objs[['organism_id','long_axis_um','short_axis_um','stain_primary_fraction']]",
    ],
)
def smear_measure(fields, objects, root=None, *, field_id="field_id", image_col="image_path",
                  um_col="um_per_px", object_id="organism_id",
                  box_cols=("x_min", "y_min", "x_max", "y_max"), stain_vectors="estimate",
                  n_jobs=1):
    """See :func:`omicverse.tl._smear_morph_core.measure_box_tables`."""
    return _core.measure_box_tables(fields, objects, root=root, field_id=field_id,
                                    image_col=image_col, um_col=um_col, object_id=object_id,
                                    box_cols=box_cols, stain_vectors=stain_vectors, n_jobs=n_jobs)


@register_function(
    aliases=["染料向量估计", "stain vectors", "two-dye stain vectors", "colour deconvolution vectors"],
    category="utils",
    description="Estimate two dye optical-density directions from pooled object pixels by non-negative "
                "factorisation (label-free); the first row is the blue/violet (red-absorbing) dye.",
    requires={}, produces={}, auto_fix="none",
    examples=["vec = ov.tl.smear_stain_vectors(od_pixels)"],
)
def smear_stain_vectors(od_pixels, n_dyes=2, seed=0):
    """Two dye OD directions (primary / red-absorbing first); see core ``estimate_stain_vectors``."""
    return _core.estimate_stain_vectors(od_pixels, n_dyes=n_dyes, seed=seed)


@register_function(
    aliases=["视野特征汇总", "field feature summary", "per-image object summaries"],
    category="utils",
    description="Robust per-field summaries (quantiles, mean, std, count) of a list of per-object "
                "measurement dicts.",
    requires={}, produces={}, auto_fix="none",
    examples=["row = ov.tl.smear_field_features(records)"],
)
def smear_field_features(records, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)):
    """Per-field quantile / mean / std summaries of per-object dicts; see core ``field_features``."""
    return _core.field_features(records, quantiles=quantiles)


@register_function(
    aliases=["形态-染色双轴分类", "two-axis classifier", "shape and stain classifier",
             "morphology-first classification", "unseen species classification", "gram call classifier",
             "group-held-out classifier", "prior-weighted stain decision"],
    category="utils",
    description="Fit a field/image classifier whose labels factor into a morphology value and a stain "
                "value (label_axes={label: (shape, stain or None)}). Morphology: group- and class-"
                "balanced random forest on stain-free columns. Stain: group- and class-balanced "
                "logistic regression on 'stain_' columns (a prior-free log likelihood ratio). The two "
                "are combined with P(stain | shape) counted over training GROUPS with Laplace smoothing, "
                "so colour that disagrees with the shape-typical stain needs stronger evidence, which "
                "generalises to new species/slides whose colour is unreliable. Labels unseen in training "
                "are not emitted. For new laboratories / species: stain_feature='core' (default; "
                "amount-invariant colour of each object's densest pixels, background-subtracted "
                "variants) and prior='tempered' (default; half the group prior's log-odds), 'flat', "
                "'em' (Saerens 2002 label-shift EM on the prediction batch) or 'group' (original).",
    requires={}, produces={}, auto_fix="none",
    examples=[
        "m = ov.tl.two_axis_classifier_fit(Xf, y, groups, label_axes={'A_round': ('round','pos'), "
        "'A_rod': ('rod','pos'), 'B_rod': ('rod','neg'), 'big': ('big', None)}, positive_stain='pos')",
        "pred = ov.tl.two_axis_classifier_predict(m, Xf_new)",
        "m = ov.tl.two_axis_classifier_fit(Xf, y, groups, label_axes=axes, positive_stain='pos', "
        "stain_feature='all', prior='group')  # original behaviour",
    ],
)
def two_axis_classifier_fit(X, labels, groups, label_axes, *, feature_names=None,
                            shape_columns=None, stain_columns=None, positive_stain=None,
                            shape_estimator="random_forest", stain_C=0.1, prior_alpha=1.0, seed=0,
                            stain_feature=_core.DEFAULT_STAIN_FEATURE, prior=_core.DEFAULT_PRIOR,
                            prior_weight=0.5, em_strength=2.0):
    """Fit the shape x stain field classifier; see core ``fit_two_axis_classifier``.

    ``stain_feature`` picks the colour measurements of the stain stage
    (``"all"`` = every ``stain_*`` summary, the pre-1.x behaviour; presets in
    ``ov.tl._smear_morph_core.STAIN_FEATURE_SETS``); ``prior`` picks how the
    group-level P(stain | shape) enters (``"group"`` original, ``"tempered"``,
    ``"flat"``, ``"em"`` batch label-shift EM)."""
    return _core.fit_two_axis_classifier(
        X, labels, groups, label_axes, feature_names=feature_names, shape_columns=shape_columns,
        stain_columns=stain_columns, positive_stain=positive_stain,
        shape_estimator=shape_estimator, stain_C=stain_C, prior_alpha=prior_alpha, seed=seed,
        stain_feature=stain_feature, prior=prior, prior_weight=prior_weight, em_strength=em_strength)


@register_function(
    aliases=["双轴分类预测", "two-axis predict"],
    category="utils",
    description="Predict labels with a model from two_axis_classifier_fit (DataFrame columns are "
                "aligned to the training features; missing ones imputed).",
    requires={}, produces={}, auto_fix="none",
    examples=["pred, info = ov.tl.two_axis_classifier_predict(m, Xf_new, return_scores=True)"],
)
def two_axis_classifier_predict(model, X, return_scores=False):
    """Predict with a two-axis model; see core ``predict_two_axis``."""
    return _core.predict_two_axis(model, X, return_scores=return_scores)


@register_function(
    aliases=["留一组交叉验证双轴", "leave-one-group-out two-axis", "leave-species-out validation"],
    category="utils",
    description="Leave-one-group-out predictions of two_axis_classifier_fit (every group predicted by "
                "a model that never saw it); score with species_balanced_recall.",
    requires={}, produces={}, auto_fix="none",
    examples=["oof = ov.tl.group_holdout_two_axis(Xf, y, groups, label_axes=axes)"],
)
def group_holdout_two_axis(X, labels, groups, **kw):
    """Leave-one-group-out labels of the two-axis classifier."""
    return _core.group_holdout_two_axis(X, labels, groups, **kw)


@register_function(
    aliases=["物种平衡召回", "species-balanced recall", "group-balanced recall"],
    category="utils",
    description="Per class: recall within each group of that class, averaged over groups; plus macro.",
    requires={}, produces={}, auto_fix="none",
    examples=["per_class, macro = ov.tl.species_balanced_recall(y, oof, groups)"],
)
def species_balanced_recall(y_true, y_pred, groups):
    """(per-class recall averaged over groups, macro) for held-out predictions."""
    return _core.species_balanced_recall(y_true, y_pred, groups)


@register_function(
    aliases=["组水平先验", "group Laplace prior", "group-level conditional prior"],
    category="utils",
    description="P(outcome | context) where every group votes once, Laplace-smoothed, so outcomes never "
                "seen with a context keep nonzero probability for new groups.",
    requires={}, produces={}, auto_fix="none",
    examples=["prior = ov.tl.group_laplace_prior(shape, stain, groups)"],
)
def group_laplace_prior(context, outcome, groups, outcomes=None, alpha=1.0):
    """Group-counted, Laplace-smoothed P(outcome | context)."""
    return _core.group_laplace_prior(context, outcome, groups, outcomes=outcomes, alpha=alpha)


@register_function(
    aliases=["涂片形态源码", "smear morphometry source", "portable predictor source"],
    category="utils",
    description="Return (or write to path) the numpy/scipy/scikit-image/scikit-learn-only source of the "
                "smear morphometry + two-axis classifier core, so a stand-alone predict.py can import "
                "it from its own directory with no omicverse at run time.",
    requires={}, produces={}, auto_fix="none",
    examples=["ov.tl.smear_morphometry_source('/root/submission/smear_core.py')"],
)
def smear_morphometry_source(path: Optional[Union[str, Path]] = None) -> str:
    """Source text of the portable core; written to ``path`` if given."""
    text = Path(_core.__file__).read_text(encoding="utf-8")
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


@register_function(
    aliases=["涂片分类器训练", "smear fit predictor", "fit field classifier from tables",
             "train gram call model", "leave-species-out fit"],
    category="utils",
    description="One call from the labelled tables to a fitted two-axis field classifier: measure every "
                "boxed object (smear_measure), join field labels and groups, report leave-one-group-out "
                "per-class species-balanced recall, then fit on all groups with the library defaults. "
                "Returns (model, report). Export it with smear_export_predictor.",
    requires={}, produces={}, auto_fix="none",
    examples=["model, rep = ov.tl.smear_fit_predictor(fields, objs, labels, label_col='culture_class', "
              "group_col='group_id', label_axes=axes, positive_stain='pos', root=data, n_jobs=4)"],
)
def smear_fit_predictor(fields, objects, labels, *, label_col, group_col, label_axes, positive_stain,
                        root=None, field_id="field_id", image_col="image_path", um_col="um_per_px",
                        object_id="organism_id", box_cols=("x_min", "y_min", "x_max", "y_max"),
                        n_jobs=1, validate=True, stain_feature=None, prior=None, prior_weight=0.5,
                        em_strength=2.0):
    """Measure, validate leave-one-group-out and fit the two-axis classifier with defaults.

    ``stain_feature`` / ``prior`` (None = library defaults ``DEFAULT_STAIN_FEATURE`` /
    ``DEFAULT_PRIOR``) are passed to ``fit_two_axis_classifier``; use
    ``stain_feature="all", prior="group"`` for the original behaviour."""
    import numpy as np
    kw = {"stain_feature": _core.DEFAULT_STAIN_FEATURE if stain_feature is None else stain_feature,
          "prior": _core.DEFAULT_PRIOR if prior is None else prior,
          "prior_weight": prior_weight, "em_strength": em_strength}
    _, X, vec = _core.measure_box_tables(fields, objects, root=root, field_id=field_id,
                                         image_col=image_col, um_col=um_col, object_id=object_id,
                                         box_cols=box_cols, n_jobs=n_jobs)
    Xl = X.merge(labels[[field_id, label_col, group_col]], on=field_id, how="inner")
    feats = [c for c in X.columns if c != field_id]
    y, g = Xl[label_col].to_numpy(object), Xl[group_col].to_numpy(object)
    report = {"n_fields": int(len(Xl)), "n_groups": int(len(np.unique(g))), "dye_vectors": vec.tolist()}
    if validate:
        oof = _core.group_holdout_two_axis(Xl[feats], y, g, label_axes=label_axes,
                                           positive_stain=positive_stain, **kw)
        per, macro = _core.species_balanced_recall(y, oof, g)
        report.update({"logo_species_balanced_recall": per, "logo_macro": macro,
                       "logo_predictions": dict(zip(Xl[field_id], oof))})
    model = _core.fit_two_axis_classifier(Xl[feats], y, g, label_axes, positive_stain=positive_stain, **kw)
    report.update({"stain_feature": kw["stain_feature"], "prior": kw["prior"]})
    model["table_columns"] = {"field_id": field_id, "image_col": image_col, "um_col": um_col,
                              "object_id": object_id, "box_cols": list(box_cols)}
    return model, report


_NO_OV = (
    "import sys, runpy\n"
    "class _Block:\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name == 'omicverse' or name.startswith('omicverse.'):\n"
    "            raise ImportError('omicverse is not available to the exported predictor')\n"
    "sys.meta_path.insert(0, _Block())\n"
    "sys.argv = ['predict.py'] + sys.argv[1:]\n"
    "runpy.run_path('predict.py', run_name='__main__')\n"
)


@register_function(
    aliases=["导出离线预测器", "smear export predictor", "write predict.py", "portable submission",
             "offline predictor bundle"],
    category="utils",
    description="Write a complete offline predictor for a fitted two-axis smear model into a directory: "
                "predict.py (--input-dir / --output-dir; no scoring code to edit), smear_core.py "
                "(numpy/scipy/skimage/sklearn only), model.pkl and predictor_config.json (table, column and "
                "output names only). With check_input_dir it copies the bundle, runs it with omicverse "
                "imports blocked, and reports coverage, finiteness, label set and runtime.",
    requires={}, produces={}, auto_fix="none",
    examples=["ov.tl.smear_export_predictor(model, '/root/submission', check_input_dir='/root/data')"],
)
def smear_export_predictor(model, submission_dir, *, fields_table="fields.csv",
                           objects_table="organisms.csv", field_output="field_calls.csv",
                           field_out_id="field_id", field_out_label="field_class",
                           object_output="organism_measurements.csv", object_out_id="organism_id",
                           object_out_cols=None, n_jobs=4, check_input_dir=None, python=None,
                           time_limit=None):
    """Write predict.py + smear_core.py + model.pkl + predictor_config.json; optionally self-check."""
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    import time
    import pandas as pd
    os.makedirs(submission_dir, exist_ok=True)
    tc = model.get("table_columns", {})
    cfg = {"fields_table": fields_table, "objects_table": objects_table,
           "field_id": tc.get("field_id", "field_id"), "image_col": tc.get("image_col", "image_path"),
           "um_col": tc.get("um_col", "um_per_px"), "object_id": tc.get("object_id", "organism_id"),
           "box_cols": tc.get("box_cols", ["x_min", "y_min", "x_max", "y_max"]),
           "field_output": field_output, "field_out_id": field_out_id, "field_out_label": field_out_label,
           "object_output": object_output, "object_out_id": object_out_id,
           "object_out_cols": object_out_cols or {"long": "long_axis_um", "short": "short_axis_um",
                                                  "stain": "stain_index"},
           "n_jobs": int(n_jobs)}
    smear_morphometry_source(os.path.join(submission_dir, "smear_core.py"))
    with open(os.path.join(submission_dir, "predict.py"), "w") as fh:
        fh.write(_core.PREDICT_PY)
    with open(os.path.join(submission_dir, "predictor_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=1)
    _core.save_predictor(model, os.path.join(submission_dir, "model.pkl"))
    result = {"files": sorted(os.listdir(submission_dir)), "config": cfg}
    if check_input_dir is None:
        return result
    tmp = tempfile.mkdtemp(prefix="smear-check-")
    bundle = os.path.join(tmp, "submission")
    shutil.copytree(submission_dir, bundle, ignore=shutil.ignore_patterns("__pycache__"))
    out = os.path.join(tmp, "out")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    proc = subprocess.run([python or sys.executable, "-c", _NO_OV, "--input-dir", str(check_input_dir),
                           "--output-dir", out], cwd=bundle, env=env, capture_output=True, text=True)
    result["check_seconds"] = round(time.time() - t0, 1)
    result["check_returncode"] = proc.returncode
    if proc.returncode != 0:
        result["check_stderr"] = proc.stderr[-3000:]
        raise RuntimeError("exported predictor failed:\n" + proc.stderr[-3000:])
    f_in = pd.read_csv(os.path.join(check_input_dir, fields_table))
    o_in = pd.read_csv(os.path.join(check_input_dir, objects_table))
    f_out = pd.read_csv(os.path.join(out, field_output))
    o_out = pd.read_csv(os.path.join(out, object_output))
    vals = o_out[list(cfg["object_out_cols"].values())].to_numpy(float)
    import numpy as np
    result.update({
        "fields_covered": bool(set(f_in[cfg["field_id"]]) == set(f_out[field_out_id])),
        "objects_covered": bool(set(o_in[cfg["object_id"]]) == set(o_out[object_out_id])),
        "all_finite": bool(np.isfinite(vals).all()),
        "labels_emitted": sorted(f_out[field_out_label].astype(str).unique().tolist()),
    })
    if time_limit is not None:
        result["within_time_limit"] = bool(result["check_seconds"] <= time_limit)
    shutil.rmtree(tmp, ignore_errors=True)
    return result

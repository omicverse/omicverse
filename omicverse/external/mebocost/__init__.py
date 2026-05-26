# -*- coding: utf-8 -*-
"""
Vendored MEBOCOST -- metabolite-mediated cell-cell communication inference.

Vendored from https://github.com/kaifuchenlab/MEBOCOST (Zheng et al., 2022,
"MEBOCOST: Metabolite-mediated Cell Communication Modeling by Single Cell
Transcriptome"). Vendored at commit 3e0be6f638b71c0f5bd08cef8b1448ae5230560e
(v1.2.2, kaifuchenlab fork).

Why vendored: MEBOCOST is pip-installable but ships *without* its metabolite
database (the ``data/mebocost_db`` folder), so a bare ``pip install mebocost``
cannot actually run. This sub-package bundles both the source and the required
human/mouse database files so that it works out-of-the-box, regardless of the
current working directory.

The original MEBOCOST resolves its database via an INI file (``mebocost.conf``)
that contains absolute file paths. Here, :func:`run_mebocost` generates that
conf file at runtime with paths pointing at the vendored ``data/`` directory
(see :func:`_write_runtime_conf`), so no absolute paths are baked in.

Public API
----------
run_mebocost
    Convenience wrapper running ``create_obj`` -> ``estimator`` -> ``infer_commu``.
"""

import os as _os
import tempfile as _tempfile

__all__ = ["run_mebocost", "create_obj", "load_obj", "save_obj"]

# directory of this vendored package
_PKG_DIR = _os.path.dirname(_os.path.abspath(__file__))
_DB_DIR = _os.path.join(_PKG_DIR, "data", "mebocost_db")


def _db_path(*parts):
    """Resolve a path inside the vendored mebocost_db directory."""
    p = _os.path.join(_DB_DIR, *parts)
    return p


def _write_runtime_conf():
    """
    Generate a MEBOCOST ``mebocost.conf`` at runtime whose paths point at the
    vendored ``data/mebocost_db`` directory. Returns the path of the conf file.

    All paths are resolved relative to this package directory, so the result is
    independent of the current working directory. The conf is written to a
    temp file (MEBOCOST only reads it; it is never modified).
    """
    common = _db_path("common")
    human = _db_path("human")
    mouse = _db_path("mouse")
    conf_text = (
        "[common]\n"
        "hmdb_info_path = {hmdb}\n"
        "scfea_info_path = {scfea}\n"
        "compass_rxt_ann_path = {crxn}\n"
        "compass_met_ann_path = {cmet}\n"
        "\n"
        "[human]\n"
        "met_enzyme_path = {h_enz}\n"
        "met_sensor_path = {h_sen}\n"
        "\n"
        "[mouse]\n"
        "met_enzyme_path = {m_enz}\n"
        "met_sensor_path = {m_sen}\n"
    ).format(
        hmdb=_os.path.join(common, "metabolite_annotation_HMDB_summary.tsv"),
        # scFEA / Compass annotation files are NOT vendored (only needed for the
        # scFEA-/compass-based met_est modes); paths are kept for completeness.
        scfea=_os.path.join(common, "Human_M168_information.symbols.csv"),
        crxn=_os.path.join(common, "rxn_md.csv"),
        cmet=_os.path.join(common, "met_md.csv"),
        h_enz=_os.path.join(
            human, "metabolite_associated_gene_reaction_HMDB_summary.tsv"
        ),
        h_sen=_os.path.join(human, "human_met_sensor_update_Oct21_2025.tsv"),
        m_enz=_os.path.join(
            mouse, "metabolite_associated_gene_reaction_HMDB_summary_mouse.tsv"
        ),
        m_sen=_os.path.join(mouse, "mouse_met_sensor_update_Oct21_2025.tsv"),
    )
    fd, conf_path = _tempfile.mkstemp(prefix="mebocost_", suffix=".conf")
    with _os.fdopen(fd, "w") as fh:
        fh.write(conf_text)
    return conf_path


def _get_create_obj():
    """Deferred import of MEBOCOST core (pulls in scanpy/statsmodels/etc.)."""
    from .mebocost import create_obj, load_obj, save_obj

    return create_obj, load_obj, save_obj


def __getattr__(name):
    # lazily expose the heavy MEBOCOST symbols at package level
    if name in ("create_obj", "load_obj", "save_obj"):
        create_obj, load_obj, save_obj = _get_create_obj()
        return {"create_obj": create_obj,
                "load_obj": load_obj,
                "save_obj": save_obj}[name]
    raise AttributeError(
        "module {!r} has no attribute {!r}".format(__name__, name)
    )


def run_mebocost(adata, *, group_key, species='human', condition_key=None,
                 n_shuffle=1000, seed=12345, thread=1,
                 sensor_type=('Receptor', 'Transporter', 'Nuclear Receptor'),
                 cutoff_exp='auto', cutoff_met='auto', cutoff_prop=0.15,
                 pval_method='permutation_test_fdr', pval_cutoff=0.05,
                 min_cell_number=50, verbose=True):
    """Infer metabolite-mediated cell-cell communication with MEBOCOST.

    Thin wrapper around MEBOCOST's ``create_obj`` -> ``.estimator()`` ->
    ``.infer_commu()`` pipeline. The MEBOCOST algorithm is used faithfully;
    only data/config I/O is adapted so the bundled database is used.

    Parameters
    ----------
    adata
        ``AnnData`` object. ``adata.X`` is expected to hold normalized
        (log) expression with genes in ``var_names``. MEBOCOST recommends
        using ``adata.raw.to_adata()`` so that all genes are available.
    group_key
        str, column name in ``adata.obs`` defining the cell groups
        (e.g. cell types) used as senders/receivers.
    species
        ``'human'`` or ``'mouse'``. Selects which vendored database is used.
    condition_key
        str or None, column name in ``adata.obs`` defining
        sample/condition labels for multi-condition mCCC. Default None.
    n_shuffle
        int, number of cell-label permutations for the null distribution
        used to compute p-values. Default 1000.
    seed
        int, random seed for the permutations (reproducibility). Default 12345.
    thread
        int, number of cores used for computation. Default 1.
    sensor_type
        list/tuple of sensor categories to model, a subset of
        ``('Receptor', 'Transporter', 'Nuclear Receptor')``.
    cutoff_exp, cutoff_met
        ``'auto'`` or float, expression/metabolite low-abundance cutoffs.
    cutoff_prop
        float in [0, 1], min fraction of abundant cells for a metabolite/gene.
    pval_method
        one of ``'ttest_pval'``, ``'ranksum_test_pval'``,
        ``'permutation_test_pval'``, ``'ttest_fdr'``, ``'ranksum_test_fdr'``,
        ``'permutation_test_fdr'``. Default ``'permutation_test_fdr'``.
    pval_cutoff
        float, significance cutoff applied to the returned events.
    min_cell_number
        int, cell groups with fewer cells are dropped (p-value set to 1).
    verbose
        bool, if False suppress MEBOCOST's stdout logging.

    Returns
    -------
    pandas.DataFrame
        The communication result table. Columns include the sender/receiver
        cell groups (``Sender``, ``Receiver``), ``Metabolite``,
        ``Metabolite_Name``, ``Sensor``, ``Annotation``, the communication
        score (``Commu_Score``), and p-value columns (e.g.
        ``permutation_test_fdr``). Filtered to ``pval_cutoff``.

    Notes
    -----
    The MEBOCOST object itself is attached to the returned DataFrame's
    ``.attrs['mebocost_obj']`` for downstream plotting / differential
    analysis.
    """
    import contextlib
    import io

    create_obj, _, _ = _get_create_obj()

    if group_key not in adata.obs.columns:
        raise KeyError(
            "group_key {!r} not in adata.obs columns: {}".format(
                group_key, list(adata.obs.columns)
            )
        )
    if condition_key is not None and condition_key not in adata.obs.columns:
        raise KeyError(
            "condition_key {!r} not in adata.obs columns: {}".format(
                condition_key, list(adata.obs.columns)
            )
        )

    conf_path = _write_runtime_conf()
    # MEBOCOST's sensor_type filter requires a list (it does an explicit
    # ``type(sensor_type) == type([])`` check), so coerce tuples to list.
    sensor_type = list(sensor_type)

    # MEBOCOST's permutation test uses multiprocessing.Pool with thread>1.
    # Under newer pandas, fork-pickling an AnnData whose ``obs`` carries a
    # CategoricalDtype column blows up in the worker with
    # ``NotImplementedError`` from ``Categorical.__setstate__``. Cast
    # categorical obs columns to str on a shallow copy of the AnnData so the
    # worker receives a clean object. The user's original adata is not
    # modified.
    if thread > 1:
        try:
            cat_cols = [c for c in adata.obs.columns
                        if str(adata.obs[c].dtype) == "category"]
        except Exception:
            cat_cols = []
        if cat_cols:
            adata = adata.copy()
            for c in cat_cols:
                adata.obs[c] = adata.obs[c].astype(str)

    def _run():
        obj = create_obj(
            adata=adata,
            group_col=group_key,
            condition_col=condition_key,
            species=species,
            met_est='mebocost',
            config_path=conf_path,
            cutoff_exp=cutoff_exp,
            cutoff_met=cutoff_met,
            cutoff_prop=cutoff_prop,
            sensor_type=sensor_type,
            thread=thread,
        )
        result = obj.infer_commu(
            n_shuffle=n_shuffle,
            seed=seed,
            Return=True,
            thread=thread,
            pval_method=pval_method,
            pval_cutoff=pval_cutoff,
            min_cell_number=min_cell_number,
        )
        return obj, result

    try:
        if verbose:
            obj, result = _run()
        else:
            # MEBOCOST is chatty across stdout, stderr (tqdm) and the
            # logging module — silence all three when verbose=False.
            import logging as _logging
            _root = _logging.getLogger()
            _prev_level = _root.level
            _root.setLevel(_logging.ERROR)
            try:
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    obj, result = _run()
            finally:
                _root.setLevel(_prev_level)
    finally:
        try:
            _os.remove(conf_path)
        except OSError:
            pass

    try:
        result.attrs['mebocost_obj'] = obj
    except Exception:
        pass
    return result

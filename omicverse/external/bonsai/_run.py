r"""Bonsai tree reconstruction (``ov.tl.bonsai``).

Bonsai reconstructs a maximum-likelihood *tree* over cells rather than a 2-D
embedding: every cell is a leaf, internal vertices are inferred ancestral states,
and branch lengths carry the inferred distance in expression space. Unlike
UMAP/t-SNE it consumes the measurement error on each feature, so cells whose
state is poorly determined are placed close to their parent instead of being
scattered by noise.

Licence — read this before using it commercially
------------------------------------------------
The upstream Bonsai core is vendored in ``bonsai/`` beside this file, under its
own **CC BY-NC 4.0** licence (Attribution-NonCommercial), carried verbatim in
``LICENSE-CC-BY-NC-4.0.md``. That licence **prohibits use by any commercial
entity, including for research**, and it is a more restrictive term than
omicverse's own GPL-3.0. Running this function emits a warning once per session
for that reason; commercial users should contact the authors (Biozentrum,
University of Basel) before running it.

``bonsai_path`` or ``$BONSAI_HOME`` point the runner at a different checkout if
you would rather track upstream yourself.

Reference
---------
de Groot, D. H., Morillo Leonardo, S. X., Pachkov, M., & van Nimwegen, E. (2026).
Bonsai-data-representation (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.20370956
"""
from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np

__all__ = ["run_bonsai"]


# ── locating the user's Bonsai checkout ──────────────────────────────────────
_NC_WARNING = (
    "Bonsai is licensed CC BY-NC 4.0 (Attribution-NonCommercial). Use by a "
    "commercial entity is prohibited by that licence, including for research. "
    "See omicverse/external/bonsai/LICENSE-CC-BY-NC-4.0.md."
)

# The upstream core is vendored beside this file. Upstream's own scripts do
#     parent = dirname(dirname(__file__)); sys.path.append(parent)
#     from bonsai.bonsai_helpers import ...
# The vendored core is a real package with relative imports, so nothing here
# touches sys.path; this is only the anchor for the config template and for
# resolving an alternative checkout.
_VENDORED_ROOT = os.path.dirname(os.path.abspath(__file__))
_NC_WARNED = False


def _resolve_bonsai(bonsai_path: Optional[str]) -> str:
    """Return the Bonsai checkout root.

    Defaults to the copy vendored in this directory. ``bonsai_path`` or
    ``$BONSAI_HOME`` override it, which is how a user runs a newer upstream, or
    one they have patched, without touching omicverse.
    """
    root = bonsai_path or os.environ.get("BONSAI_HOME") or _VENDORED_ROOT
    root = os.path.abspath(os.path.expanduser(root))
    main = os.path.join(root, "bonsai", "bonsai_main.py")
    if not os.path.isfile(main):
        raise FileNotFoundError(
            f"{main} not found — bonsai_path must be the repository root, i.e. "
            "the directory that contains 'bonsai/'."
        )
    return root


def _feature_matrix(adata, use_rep, layer):
    """Return ``(features, feature_names)`` as an (n_features, n_cells) array.

    Bonsai's ``features.txt`` is features-by-objects, i.e. the transpose of the
    AnnData convention, so the transpose happens once, here.
    """
    if use_rep is not None:
        if use_rep not in adata.obsm:
            raise KeyError(f"use_rep='{use_rep}' is not in adata.obsm "
                           f"({list(adata.obsm)})")
        X = np.asarray(adata.obsm[use_rep], dtype=float)
        names = [f"{use_rep}_{i}" for i in range(X.shape[1])]
    elif layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"layer='{layer}' is not in adata.layers "
                           f"({list(adata.layers)})")
        X = adata.layers[layer]
        X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=float)
        names = list(adata.var_names)
    else:
        X = adata.X
        X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=float)
        names = list(adata.var_names)

    if not np.isfinite(X).all():
        raise ValueError("Non-finite values in the feature matrix; Bonsai's "
                         "likelihood model needs finite means.")
    return X.T, names


def _write_inputs(adata, data_dir, features, feature_names, sds):
    """Write the four plain-text files Bonsai reads. No header, no index."""
    os.makedirs(data_dir, exist_ok=True)
    np.savetxt(os.path.join(data_dir, "features.txt"), features, delimiter="\t")
    with open(os.path.join(data_dir, "cellID.txt"), "w") as fh:
        fh.write("\n".join(map(str, adata.obs_names)) + "\n")
    with open(os.path.join(data_dir, "geneID.txt"), "w") as fh:
        fh.write("\n".join(map(str, feature_names)) + "\n")

    if sds is None:
        return "features.txt"
    np.savetxt(os.path.join(data_dir, "standard_deviations.txt"), sds, delimiter="\t")
    return "features.txt,standard_deviations.txt"


_CONFIG_TEMPLATE = os.path.join(
    _VENDORED_ROOT, "bonsai", "config_template_do_not_change", "config_template.yaml")

# Upstream's argparse namespace, reduced to the five attributes bonsai_main.main
# reads before it rebuilds everything else from the YAML.
class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _build_config(cfg_path, **overrides):
    """Write the run's YAML using Bonsai's own config machinery.

    ``Run_Configs`` needs an existing YAML to start from, so we seed it with the
    template that ships with the vendored core and override only the fields this
    run sets. Writing it out is not ceremony: ``main`` re-reads the file, and
    keeping it beside the results is how a run stays reproducible.
    """
    from .bonsai.bonsai_helpers import Run_Configs

    ns = _Args(**overrides)
    cfg = Run_Configs(_CONFIG_TEMPLATE, args=ns, args_to_copy=list(overrides))
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    for k, v in overrides.items():
        cfg.config_yaml[k] = v
    cfg.store_yaml(cfg_path)
    return cfg_path


def _parse_results(results_dir):
    """Read the ``final_bonsai_*`` directory into plain Python objects."""
    # Newest by mtime, not alphabetically last: a backbone run writes one
    # final_bonsai_* per round, named after the tree it started from, and those
    # names do not sort in the order they were produced.
    finals = sorted((d for d in os.listdir(results_dir)
                     if d.startswith("final_bonsai_")
                     and os.path.isdir(os.path.join(results_dir, d))),
                    key=lambda d: os.path.getmtime(os.path.join(results_dir, d)))
    if not finals:
        raise FileNotFoundError(
            f"No 'final_bonsai_*' directory in {results_dir} — the run did not "
            "reach its final step."
        )
    final = os.path.join(results_dir, finals[-1])

    nwk_files = [f for f in os.listdir(final) if f.endswith(".nwk")]
    newick = None
    if nwk_files:
        with open(os.path.join(final, nwk_files[0])) as fh:
            newick = fh.read().strip()

    # edgeInfo.txt: parent vertInd, child vertInd, edge length
    edges = np.loadtxt(os.path.join(final, "edgeInfo.txt"), ndmin=2)
    if edges.shape[1] < 3:
        raise ValueError(f"edgeInfo.txt in {final} has {edges.shape[1]} columns, "
                         "expected at least 3 (vertInd, vertInd, length).")

    # vertInfo.txt is 'vertInd  nodeInd  vertName' with a header. vertName is
    # what links a leaf back to a cell in adata.obs_names; nodeInd sits between
    # them and is upstream's own historical artefact, so the columns are located
    # by name rather than by position.
    vert_names, vert_inds = [], []
    with open(os.path.join(final, "vertInfo.txt")) as fh:
        header = fh.readline().strip().split("\t")
        try:
            i_ind, i_name = header.index("vertInd"), header.index("vertName")
        except ValueError:
            i_ind, i_name = 0, len(header) - 1
        for raw in fh:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) <= max(i_ind, i_name):
                continue
            vert_inds.append(int(float(parts[i_ind])))
            vert_names.append(parts[i_name])

    return {
        "results_dir": final,
        "newick": newick,
        "edges": np.column_stack([edges[:, 0].astype(int),
                                  edges[:, 1].astype(int)]),
        "edge_lengths": edges[:, 2].astype(float),
        "vert_ind": np.asarray(vert_inds, dtype=int),
        "vert_name": np.asarray(vert_names, dtype=object),
    }


def run_bonsai(
    adata,
    *,
    use_rep: Optional[str] = None,
    layer: Optional[str] = None,
    sd_key: Optional[str] = None,
    bonsai_path: Optional[str] = None,
    results_dir: Optional[str] = None,
    zscore_cutoff: float = 1.0,
    ub_ellipsoid_size: float = 1.0,
    use_knn: int = 10,
    nnn_n_randommoves: int = 1000,
    nnn_n_randomtrees: int = 10,
    n_initial_cells: Optional[int] = None,
    growth_factor_guide: float = 10.0,
    fast: bool = True,
    verbose: bool = False,
    key_added: str = "bonsai",
    copy: bool = False,
):
    r"""Reconstruct a Bonsai tree over the cells of ``adata``.

    Bonsai infers the maximum-likelihood tree whose leaves are the observed
    cells, treating each cell as a Gaussian measurement in feature space. Where
    UMAP and t-SNE compress the data into two dimensions and let noise spread
    cells apart, Bonsai keeps the uncertainty: a cell measured imprecisely is
    placed on a short branch near its parent rather than pushed to its own
    corner of the plot.

    The heavy lifting is done by the upstream Bonsai CLI, which this function
    drives — see the module docstring for why it is not bundled, and note that
    its **CC BY-NC 4.0** licence forbids commercial use.

    Parameters
    ----------
    adata
        Annotated data matrix. Cells become the leaves of the tree.
    use_rep
        Key in ``adata.obsm`` holding the feature estimates, e.g.
        ``'scaled|original|X_pca'``. Preferred over ``layer``/``X``: Bonsai's
        cost grows with the number of features, and a PCA representation is
        both smaller and closer to the Gaussian model it assumes.
    layer
        Key in ``adata.layers`` to use instead of ``use_rep``. Falls back to
        ``adata.X`` when neither is given.
    sd_key
        Key in ``adata.obsm`` holding per-cell, per-feature standard deviations,
        the same shape as the chosen features. This is the input that separates
        Bonsai from a plain hierarchical clustering — without it Bonsai assumes
        the estimates are near-exact and the tree will over-commit to noise.
    bonsai_path
        Path to the Bonsai repository root. Defaults to ``$BONSAI_HOME``.
    results_dir
        Where to write Bonsai's inputs and results. Defaults to a directory
        named ``bonsai_<key_added>`` in the working directory; it is kept, not
        cleaned up, because a finished run is expensive to reproduce.
    zscore_cutoff, ub_ellipsoid_size, use_knn
        Passed through to Bonsai's configuration file unchanged; see the
        upstream README's "Detailed description of the run configurations".
    nnn_n_randommoves, nnn_n_randomtrees
        How hard the nearest-neighbour-interchange search looks for a better
        tree. These defaults are upstream's own (``1000``/``10``), so a run here
        is the same computation ``bonsai_main.py`` would do -- and they dominate
        the cost. Upstream's shipped example uses ``10``/``2`` instead, which is
        roughly two orders of magnitude less search; that is the setting to reach
        for when exploring, at the price of a less thoroughly optimised tree.
    n_initial_cells
        Switch to **backbone-based** reconstruction, which is what upstream
        provides for datasets a direct run cannot reach. It builds a tree on this
        many cells, grafts the rest onto it, refines, and repeats until every
        cell is placed. Leave ``None`` for a direct run: the direct cost grows as
        about ``n**1.4`` (measured on PBMC: 18 s at 100 cells, 87 s at 400,
        229 s at 800 on one core), so the crossover is in the low thousands.
        Upstream's own default for this argument is ``10000``.
    growth_factor_guide
        How much the tree may grow in one grafting round before it is refined
        and used as the next backbone. Upstream's default is ``10``.
    fast
        Use the compiled kernels in :mod:`omicverse.external.bonsai._fast`
        (about 2.5x on a PCA representation). They replace two hot numerical
        functions and the bounded two-variable optimiser the tree search calls
        ~78k times per run. Measured over five independent subsets the trees come
        out equivalent in log-likelihood -- differences of 0.2 to 1.5 against a
        total near 1100, in neither direction consistently -- but they are **not
        bit-identical to upstream**, because the search compares quantities that
        carry only about three significant digits and any reimplementation
        diverges there. Set ``False`` to run upstream's own code path.
    verbose
        Stream Bonsai's own output instead of capturing it.
    key_added
        Key under ``adata.uns`` for the result.
    copy
        Return a modified copy instead of writing in place.

    Returns
    -------
    :class:`anndata.AnnData` or None
        Writes ``adata.uns[key_added]`` with

        ``newick``
            The tree in Newick format.
        ``edges`` / ``edge_lengths``
            ``(n_edges, 2)`` vertex indices and their branch lengths.
        ``vert_ind`` / ``vert_name``
            Vertex indices and names; leaf names match ``adata.obs_names``.
        ``leaf_of_obs``
            Vertex index per cell, aligned to ``adata.obs_names``, ``-1`` where a
            cell did not end up as a leaf.
        ``results_dir``
            The ``final_bonsai_*`` directory, for :func:`omicverse.pl.bonsai`
            and for re-reading without recomputing.

    Examples
    --------
    >>> import omicverse as ov
    >>> ov.pp.pca(adata, layer='scaled', n_pcs=50)
    >>> ov.tl.bonsai(adata, use_rep='scaled|original|X_pca')
    >>> ov.pl.bonsai(adata, color='leiden')
    """
    adata = adata.copy() if copy else adata

    global _NC_WARNED
    if not _NC_WARNED:
        warnings.warn(_NC_WARNING, stacklevel=2)
        _NC_WARNED = True

    from . import _fast
    _prev_fast = _fast.FAST_MAX_FEATURES
    if not fast:
        _fast.FAST_MAX_FEATURES = 0

    root = _resolve_bonsai(bonsai_path)
    features, feature_names = _feature_matrix(adata, use_rep, layer)

    sds = None
    if sd_key is not None:
        if sd_key not in adata.obsm:
            raise KeyError(f"sd_key='{sd_key}' is not in adata.obsm")
        sds = np.asarray(adata.obsm[sd_key], dtype=float).T
        if sds.shape != features.shape:
            raise ValueError(
                f"sd_key='{sd_key}' has shape {sds.T.shape}, but the features "
                f"have shape {features.T.shape}; they must match cell-by-feature."
            )

    out = os.path.abspath(results_dir or f"bonsai_{key_added}")
    data_dir = os.path.join(out, "data")
    res_dir = os.path.join(out, "results")
    os.makedirs(res_dir, exist_ok=True)
    filenames_data = _write_inputs(adata, data_dir, features, feature_names, sds)

    cfg = _build_config(
        os.path.join(out, "bonsai_config.yaml"),
        dataset=key_added,
        data_folder=data_dir,
        results_folder=res_dir,
        input_is_sanity_output=False,
        filenames_data=filenames_data,
        zscore_cutoff=float(zscore_cutoff),
        UB_ellipsoid_size=float(ub_ellipsoid_size),
        use_knn=int(use_knn),
        nnn_n_randommoves=int(nnn_n_randommoves),
        nnn_n_randomtrees=int(nnn_n_randomtrees),
        verbose=bool(verbose),
    )

    # In process, not a subprocess: the core is vendored inside this package, so
    # importing it is the direct route. Upstream ran its pipeline at module level
    # and reached for sys.path to find its own siblings; the vendored copy is a
    # real package with relative imports and its body wrapped in ``main``.
    if n_initial_cells is None:
        from .bonsai import bonsai_main
        step_mod, step_args = bonsai_main, dict(step="all")
    else:
        # Upstream drives the backbone rounds itself; it only needs the same
        # config plus how big the first tree is and how far it may grow.
        from .backbone import backbone_based_bonsai as step_mod
        step_args = dict(n_initial_cells=int(n_initial_cells),
                         growth_factor_guide=float(growth_factor_guide),
                         return_commands=False, select_target="cluster_centers",
                         search_tol=2, growth_before_cleanup=0.5,
                         iterative_cell_lists="")

    # Build the namespace from upstream's own parser defaults, then override.
    # Guessing which attributes a step reads is how you get AttributeError three
    # steps into a long run; this way every attribute it expects exists, with
    # upstream's value, and only what we set differs.
    ns = step_mod._build_parser().parse_args([])
    for k, v in dict(step_args, config_filepath=cfg, pickup_intermediate=False,
                     store_all_nwk_folder="", print_annotations="").items():
        setattr(ns, k, v)

    try:
        step_mod.main(ns)
    except SystemExit:
        # Upstream calls bare exit() on three paths -- two of them are ordinary
        # termination and one is an error, and all three carry no exit code, so
        # they cannot be told apart here. Swallow it (a notebook kernel must not
        # die on a successful run) and let _parse_results decide: a run that
        # failed leaves no final_bonsai_* directory behind.
        pass

    _fast.FAST_MAX_FEATURES = _prev_fast
    res = _parse_results(res_dir)

    # Leaves carry vertName == the cell ID we wrote into cellID.txt.
    name_to_vert = {n: i for i, n in zip(res["vert_ind"], res["vert_name"]) if n}
    res["leaf_of_obs"] = np.array(
        [name_to_vert.get(str(n), -1) for n in adata.obs_names], dtype=int)
    unmatched = int((res["leaf_of_obs"] < 0).sum())
    if unmatched:
        print(f"[bonsai] {unmatched}/{adata.n_obs} cells have no leaf in the "
              f"tree; ov.pl.bonsai will draw the tree without them.")

    adata.uns[key_added] = res
    return adata if copy else None

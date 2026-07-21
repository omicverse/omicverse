from __future__ import annotations

import inspect
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

from ..pp import scale,pca
import scanpy as sc
import numpy as np
import anndata
from .._settings import add_reference,settings
from .._registry import register_function
from .._monitor import monitor
from ..report._provenance import tracked, note


def _split_kwargs_by_signature(
    kwargs: Dict[str, Any],
    *destinations: Tuple[str, Callable[..., Any]],
) -> Tuple[Dict[str, Any], ...]:
    """Partition ``kwargs`` across multiple callables by inspecting each
    callable's signature.

    The historical ``ov.single.batch_correction(..., **kwargs)`` API forwarded
    every keyword to ``scvi.model.SCVI(**kwargs)``, leaving ``.train()``
    un-parameterisable. Modern scvi-tools splits user-tunable parameters
    between ``Model.__init__`` (architecture: ``n_hidden`` / ``n_latent`` / …)
    and ``Model.train`` (optimisation: ``max_epochs`` / ``batch_size`` /
    ``accelerator`` / …). This helper makes that split automatic.

    Parameters
    ----------
    kwargs
        The full keyword dict supplied by the user.
    destinations
        One ``(label, callable)`` pair per target, in priority order. The
        ``label`` is used in warnings; the ``callable`` is introspected via
        ``inspect.signature``.

    Returns
    -------
    tuple of dict
        One dict per destination, in the same order as ``destinations``.

    Routing rules
    -------------
    1. A kwarg whose name is a named parameter of exactly ONE destination
       routes there (precise match).
    2. A kwarg whose name is a named parameter of MULTIPLE destinations
       routes to the first match in ``destinations`` order, with a warning
       naming the others.
    3. A kwarg whose name is in NO destination's named params is sent to the
       first destination that has ``**kwargs`` (``VAR_KEYWORD``) — this
       preserves the legacy "send everything to init" behaviour for unknown
       names without silently mis-routing the ones we now recognise.
    4. A kwarg that no destination would accept (no named match, no
       ``VAR_KEYWORD`` anywhere) is dropped with a warning listing the
       accepted names.

    Notes
    -----
    Only ``POSITIONAL_OR_KEYWORD`` and ``KEYWORD_ONLY`` parameters count as
    "named". ``self`` is filtered out automatically. Signature introspection
    failures (C-extension callables, decorator obfuscation) collapse to "no
    named params, no VAR_KEYWORD" — those destinations effectively become
    inert.
    """
    sigs = []
    for label, fn in destinations:
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            params = {}
        named = {
            n for n, p in params.items()
            if n != "self" and p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        has_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        sigs.append((label, named, has_var_kw))

    buckets: list[Dict[str, Any]] = [dict() for _ in destinations]
    unrecognised: list[str] = []

    for key, val in kwargs.items():
        matches = [i for i, (_, named, _) in enumerate(sigs) if key in named]

        if len(matches) == 1:
            buckets[matches[0]][key] = val
        elif len(matches) > 1:
            winner = sigs[matches[0]][0]
            losers = ", ".join(sigs[i][0] for i in matches[1:])
            warnings.warn(
                f"kwarg {key!r} is a named parameter of both {winner} and "
                f"{losers}; routed to {winner}. Rename or pass positionally "
                f"to disambiguate.",
                UserWarning, stacklevel=3,
            )
            buckets[matches[0]][key] = val
        else:
            var_kw_dests = [i for i, (_, _, has) in enumerate(sigs) if has]
            if var_kw_dests:
                buckets[var_kw_dests[0]][key] = val
            else:
                unrecognised.append(key)

    if unrecognised:
        accepted = sorted({n for _, named, _ in sigs for n in named})
        preview = ", ".join(accepted[:20])
        if len(accepted) > 20:
            preview += ", …"
        warnings.warn(
            f"Dropped unrecognised kwarg(s) {sorted(unrecognised)!r} — none "
            f"of {[d[0] for d in destinations]} accept them. Accepted "
            f"names include: {preview}.",
            UserWarning, stacklevel=3,
        )

    return tuple(buckets)


# Per-method obsm key the integrated embedding lands in. Used by the
# tracked() viz spec so the report's diagnostic plot is colored by
# batch_key on the embedding the user just produced.
_BATCH_OBSM = {
    "harmony":   "X_pca_harmony",
    "combat":    "X_combat",
    "scanorama": "X_scanorama",
    "scVI":      "X_scVI",
    "scANVI":    "X_scANVI",
    "SCANVI":    "X_scANVI",
    "totalVI":   "X_totalVI",
    "TOTALVI":   "X_totalVI",
    "scPoli":    "X_scPoli",
    "SCPOLI":    "X_scPoli",
    "CellANOVA": "X_cellanova",
    "Concord":   "X_concord",
    "cca":       "X_cca",
    "seurat_cca": "X_cca",
    "CCA":       "X_cca",
    "rpca":       "X_rpca",
    "seurat_rpca": "X_rpca",
    "RPCA":       "X_rpca",
}

@monitor
@register_function(
    aliases=["批次校正", "batch_correction", "batch_correct", "数据整合", "去批次效应"],
    category="single",
    description="Comprehensive batch effect correction using multiple methods including Harmony, Combat, Scanorama, scVI, CellANOVA, Concord, and Seurat-style CCA",
    prerequisites={
        'optional_functions': ['preprocess', 'scale', 'pca']
    },
    requires={
        'obsm': [],  # Flexible - some methods use X_pca, others raw data
        'obs': []    # Requires batch_key column (user-specified)
    },
    produces={
        'obsm': []  # Dynamic: X_pca_harmony, X_combat, X_scanorama, X_scVI, or X_cellanova
    },
    auto_fix='none',
    examples=[
        "# Harmony batch correction (recommended for most cases)",
        "ov.single.batch_correction(adata, batch_key='batch', methods='harmony')",
        "# Combat batch correction",
        "ov.single.batch_correction(adata, batch_key='batch', methods='combat')",
        "# Scanorama integration",
        "ov.single.batch_correction(adata, batch_key='batch', methods='scanorama')",
        "# GPU-accelerated scVI (requires GPU)",
        "model = ov.single.batch_correction(adata, batch_key='batch',",
        "                                   methods='scVI', n_layers=2, n_latent=30)",
        "# CellANOVA with control samples",
        "control_dict = {'pool1': ['batch1', 'batch2']}",
        "ov.single.batch_correction(adata, batch_key='batch', methods='CellANOVA',",
        "                           control_dict=control_dict)",
        "# Seurat-style CCA (no PyTorch, no CUDA; pure scipy)",
        "ov.single.batch_correction(adata, batch_key='batch', methods='cca',",
        "                           n_pcs=30)",
    ],
    related=["pp.preprocess", "utils.mde", "utils.embedding"]
)
@tracked("batch_correction", "ov.single.batch_correction")
def batch_correction(adata:anndata.AnnData,batch_key:str,
                     use_rep='scaled|original|X_pca',
                     methods:str='harmony',n_pcs:int=50,**kwargs)->anndata.AnnData:
    """Run batch-effect correction for single-cell data integration.

    Parameters
    ----------
    adata : anndata.AnnData
        Input data matrix. Different methods require different preprocessing:
        for example, ``scVI`` typically needs raw counts in ``adata.layers['counts']``.
    batch_key : str
        Column in ``adata.obs`` defining batch/domain labels.
    use_rep : str, default='scaled|original|X_pca'
        Embedding key used by embedding-based methods (for example Harmony).
    methods : str, default='harmony'
        Integration backend. Supported values include ``'harmony'``,
        ``'combat'``, ``'scanorama'``, ``'scVI'``, ``'scANVI'``
        (semi-supervised VAE; requires ``labels_key=``),
        ``'totalVI'`` (joint RNA+protein; requires
        ``protein_expression_obsm_key=``), ``'scPoli'`` (scArches
        conditional VAE; supports ``cell_type_keys=`` for prototype
        learning), ``'CellANOVA'``, ``'Concord'``, ``'cca'`` /
        ``'seurat_cca'`` — the pure-Python port of ``Seurat::RunCCA``
        (via the `pyccasc` package, no R / rpy2 required) — and
        ``'rpca'`` / ``'seurat_rpca'`` — a faithful Python port of Seurat's
        ``IntegrateLayers(method = RPCAIntegration)`` (reciprocal-PCA anchors,
        no R required; writes ``adata.obsm['X_rpca']``). RPCA accepts
        ``features=``, ``reference=``, ``k_anchor=``, ``k_weight=``,
        ``sd_weight=`` and ``orig_rep=`` (the joint reduction to correct).
    n_pcs : int, default=50
        Number of principal components / canonical components used by the
        selected backend. For ``'cca'`` this is the number of canonical
        components (``num_cc`` in pyccasc terms).
    **kwargs
        Additional method-specific keyword arguments passed to the selected
        backend implementation.

    Returns
    -------
    anndata.AnnData or object
        Returns ``adata`` for most methods, scVI model object for ``'scVI'``,
        and Concord object for ``'Concord'``. The integrated embeddings are
        written to ``adata.obsm``.
    """

    # Declare the report's viz/backend up front from the static
    # methods → obsm-key mapping. @tracked discards the entry on
    # exception, so a method that fails mid-way never leaks a phantom
    # entry pointing at an obsm key that was never written.
    _basis = _BATCH_OBSM.get(methods)
    if _basis is not None:
        note(backend=f"omicverse · {methods}",
             viz=[{"function": "ov.pl.embedding",
                    "kwargs": {"basis": _basis,
                                "color": batch_key,
                                "frameon": "small"}}])

    from ..pp._qc import _is_oom
    _oom = _is_oom(adata)

    if _oom and methods not in ('harmony',):
        print(
            f"[AnnDataOOM] '{methods}' requires the full expression matrix in memory.\n"
            f"  Converting to in-memory AnnData automatically.\n"
            f"  Tip: 'harmony' works directly on PCA embeddings without materialisation —\n"
            f"       consider ov.single.batch_correction(adata, methods='harmony') instead."
        )
        adata_mem = adata.to_adata()
        batch_correction(adata_mem, batch_key, use_rep=use_rep,
                         methods=methods, n_pcs=n_pcs, **kwargs)
        # Copy results back onto the OOM adata so we preserve the OOM contract.
        for k in adata_mem.obsm:
            if k not in adata.obsm:
                adata.obsm[k] = adata_mem.obsm[k]
        for k in adata_mem.obs.columns:
            if k not in adata.obs.columns:
                adata.obs[k] = adata_mem.obs[k].values
        for k in adata_mem.uns:
            if k not in adata.uns:
                adata.uns[k] = adata_mem.uns[k]
        del adata_mem
        return adata

    print(f'...Begin using {methods} to correct batch effect')

    if methods=='harmony':
        from ..external.harmony import run_harmony

        adata3=adata.copy()
        if 'scaled|original|X_pca' not in adata3.obsm.keys() and use_rep=='scaled|original|X_pca':
            scale(adata3)
            pca(adata3,layer='scaled',n_pcs=n_pcs)

        # Map deprecated use_gpu to device for backward compatibility
        import warnings
        harmony_kwargs = dict(kwargs)
        if 'use_gpu' in harmony_kwargs:
            use_gpu = harmony_kwargs.pop('use_gpu')
            warnings.warn(
                "Harmony parameter 'use_gpu' is deprecated, use device='cuda'/'cpu' instead.",
                FutureWarning, stacklevel=2,
            )
            if 'device' not in harmony_kwargs:
                harmony_kwargs['device'] = None if use_gpu else 'cpu'
        # Warn on removed parameters that had functional meaning
        for _removed in ('reference_values', 'cluster_prior', 'cluster_fn'):
            if _removed in harmony_kwargs:
                warnings.warn(
                    f"Harmony parameter '{_removed}' is no longer supported "
                    "in harmonypy v0.2.0 and will be ignored.",
                    FutureWarning,
                    stacklevel=2,
                )
                harmony_kwargs.pop(_removed)
        harmony_kwargs.pop('plot_convergence', None)

        harmony_out = run_harmony(adata3.obsm[use_rep], adata3.obs, batch_key, **harmony_kwargs)
        harmony_result = harmony_out.result()
        adata.obsm['X_pca_harmony'] = harmony_result
        adata.obsm['X_harmony'] = harmony_result
        
        add_reference(adata,'Harmony','batch correction with Harmony')
        
    elif methods=='combat':
        adata2=adata.copy()
        sc.pp.combat(adata2, key=batch_key,**kwargs)
        scale(adata2)
        pca(adata2,layer='scaled',n_pcs=n_pcs)
        adata2.obsm['X_combat']=adata2.obsm[use_rep].copy()
        adata.obsm['X_combat']=adata2.obsm['X_combat'].copy()
        del adata2
        add_reference(adata,'Combat','batch correction with Combat')
        #return adata2
    elif methods=='scanorama':
        try:
            import intervaltree
            import fbpca
            #print('mofax have been install version:',mfx.__version__)
        except ImportError:
            raise ImportError(
                'Please install the intervaltree: `pip install intervaltree fbpca`.'
            )
        from ..external.scanorama import integrate_scanpy
        batches = adata.obs[batch_key].cat.categories.tolist()
        alldata = {}
        for batch in batches:
            alldata[batch] = adata[adata.obs[batch_key] == batch,]
        alldata2 = dict()
        for ds in alldata.keys():
            print(ds)
            alldata2[ds] = alldata[ds]
        
        #convert to list of AnnData objects
        adatas = list(alldata2.values())
        
        # run scanorama.integrate
        integrate_scanpy(adatas, dimred = n_pcs,**kwargs)
        scanorama_int = [ad.obsm['X_scanorama'] for ad in adatas]

        # make into one matrix.
        all_s = np.concatenate(scanorama_int)
        print(all_s.shape)
        
        # add to the AnnData object, create a new object first
        adata.obsm["X_scanorama"] = all_s
        add_reference(adata,'Scanorama','batch correction with Scanorama')
        return adata
    elif methods=='scVI':
        try:
            import scvi 
            #print('mofax have been install version:',mfx.__version__)
        except ImportError:
            raise ImportError(
                'Please install the scVI: `pip install scvi-tools`. or `conda install scvi-tools -c conda-forge`'
            )
        import scvi
        scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)
        # The user's **kwargs may target either SCVI.__init__ (architecture:
        # n_hidden / n_latent / dropout_rate / dispersion / gene_likelihood /
        # latent_distribution / …) or SCVI.train (optimisation: max_epochs /
        # batch_size / accelerator / devices / early_stopping / …). Route by
        # signature so train-side parameters are no longer silently swallowed
        # by SCVI.__init__'s **kwargs catch-all.
        init_kwargs, train_kwargs = _split_kwargs_by_signature(
            kwargs,
            ("scvi.model.SCVI.__init__", scvi.model.SCVI.__init__),
            ("scvi.model.SCVI.train", scvi.model.SCVI.train),
        )
        model = scvi.model.SCVI(adata, **init_kwargs)
        model.train(**train_kwargs)
        SCVI_LATENT_KEY = "X_scVI"
        adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()
        add_reference(adata,'scVI','batch correction with scVI')
        return model
    elif methods in ('scANVI', 'SCANVI'):
        try:
            import scvi
        except ImportError:
            raise ImportError(
                'Please install scvi-tools: `pip install scvi-tools` '
                'or `conda install scvi-tools -c conda-forge`'
            )
        # scANVI is semi-supervised — it needs a cell-type column with an
        # explicit "unknown" marker for cells without a confident label.
        labels_key = kwargs.pop('labels_key', None)
        unlabeled_category = kwargs.pop('unlabeled_category', 'Unknown')
        if labels_key is None:
            raise ValueError(
                "methods='scANVI' requires labels_key= naming a column in "
                "adata.obs with cell-type labels (use the value passed as "
                "unlabeled_category= for cells you want the model to predict "
                "labels for). See scvi-tools docs: "
                "https://docs.scvi-tools.org/en/stable/user_guide/models/scanvi.html"
            )
        scvi.model.SCANVI.setup_anndata(
            adata, layer="counts", batch_key=batch_key,
            labels_key=labels_key, unlabeled_category=unlabeled_category,
        )
        init_kwargs, train_kwargs = _split_kwargs_by_signature(
            kwargs,
            ("scvi.model.SCANVI.__init__", scvi.model.SCANVI.__init__),
            ("scvi.model.SCANVI.train", scvi.model.SCANVI.train),
        )
        model = scvi.model.SCANVI(adata, **init_kwargs)
        model.train(**train_kwargs)
        adata.obsm["X_scANVI"] = model.get_latent_representation()
        # scANVI also predicts labels for unlabeled cells — surface them.
        try:
            adata.obs["scANVI_predicted_labels"] = model.predict()
        except Exception:
            pass
        add_reference(adata, 'scANVI', 'batch correction with scANVI (semi-supervised)')
        return model
    elif methods in ('totalVI', 'TOTALVI'):
        try:
            import scvi
        except ImportError:
            raise ImportError(
                'Please install scvi-tools: `pip install scvi-tools` '
                'or `conda install scvi-tools -c conda-forge`'
            )
        # totalVI is a JOINT RNA + protein (CITE-seq / Total-seq) model. It
        # needs the protein matrix attached to adata.obsm[<key>] BEFORE call.
        protein_expression_obsm_key = kwargs.pop('protein_expression_obsm_key', None)
        if protein_expression_obsm_key is None:
            raise ValueError(
                "methods='totalVI' requires protein_expression_obsm_key= naming "
                "an obsm slot with the (cell × protein) ADT count matrix. "
                "If you only have RNA, use methods='scVI' (or 'scANVI' with "
                "labels). See scvi-tools docs: "
                "https://docs.scvi-tools.org/en/stable/user_guide/models/totalvi.html"
            )
        scvi.model.TOTALVI.setup_anndata(
            adata, batch_key=batch_key, layer="counts",
            protein_expression_obsm_key=protein_expression_obsm_key,
        )
        init_kwargs, train_kwargs = _split_kwargs_by_signature(
            kwargs,
            ("scvi.model.TOTALVI.__init__", scvi.model.TOTALVI.__init__),
            ("scvi.model.TOTALVI.train", scvi.model.TOTALVI.train),
        )
        model = scvi.model.TOTALVI(adata, **init_kwargs)
        model.train(**train_kwargs)
        adata.obsm["X_totalVI"] = model.get_latent_representation()
        add_reference(adata, 'totalVI', 'batch correction with totalVI (joint RNA+protein)')
        return model
    elif methods in ('scPoli', 'SCPOLI'):
        try:
            from scarches.models.scpoli import scPoli
        except ImportError:
            raise ImportError(
                'Please install scArches: `pip install scarches`'
            )
        # scPoli is a conditional VAE that learns per-condition prototypes
        # (optionally per cell type, if cell_type_keys is supplied). It takes
        # condition_keys instead of batch_key; we forward batch_key there.
        cell_type_keys = kwargs.pop('cell_type_keys', None)
        init_kwargs, train_kwargs = _split_kwargs_by_signature(
            kwargs,
            ("scarches.models.scpoli.scPoli.__init__", scPoli.__init__),
            ("scarches.models.scpoli.scPoli.train", scPoli.train),
        )
        scpoli_init = {
            "adata": adata,
            "condition_keys": batch_key,
            **init_kwargs,
        }
        if cell_type_keys is not None:
            scpoli_init["cell_type_keys"] = cell_type_keys
        model = scPoli(**scpoli_init)
        model.train(**train_kwargs)
        # scPoli's canonical embedding accessor is get_latent(adata, mean=True);
        # fall back to get_latent() with no kwargs for older releases.
        try:
            latent = model.get_latent(adata, mean=True)
        except TypeError:
            latent = model.get_latent(adata)
        adata.obsm["X_scPoli"] = latent
        add_reference(adata, 'scPoli', 'batch correction with scPoli (conditional VAE)')
        return model
    elif methods=='CellANOVA':
        from ..external.cellanova.model import calc_ME,calc_BE,calc_TE
        if ('highly_variable_features' in adata.var.columns) and ('highly_variable' not in adata.var.columns):
            adata.var['highly_variable']=adata.var['highly_variable_features']
        adata= calc_ME(adata, integrate_key=batch_key)
        adata = calc_BE(adata,  integrate_key=batch_key, **kwargs)
        adata = calc_TE(adata,  integrate_key=batch_key)
        from scipy.sparse import csr_matrix
        adata.layers['denoised']=csr_matrix(adata.layers['denoised'])
        ## create an independent anndata object for cellanova-integrated data
        pca(adata,layer='denoised',n_pcs=n_pcs)
        adata.obsm['X_cellanova']=adata.obsm['denoised|original|X_pca'].copy()
        add_reference(adata,'CellANOVA','batch correction with CellANOVA')
    elif methods=='Concord' or methods=='concord':
        try:
            import concord as ccd
        except ImportError:
            raise ImportError(
                'Please install the concord: `pip install concord-sc`.'
            )
        import torch
        # Set device to cpu or to gpu (if your torch has been set up correctly to use GPU), for mac you can use either torch.device('mps') or torch.device('cpu')
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if 'highly_variable' in adata.var.columns:
            feature_list = adata.var.loc[adata.var['highly_variable']==True].index.tolist()
        elif 'highly_variable_features' in adata.var.columns:
            feature_list = adata.var.loc[adata.var['highly_variable_features']==True].index.tolist()
        else:
            print('No highly variable features found, using all features')
            feature_list = adata.var_names.tolist()

        # Initialize Concord with an AnnData object, skip input_feature to use all features, set preload_dense=False if your data is very large
        # Provide 'domain_key' if integrating across batches, see below
        cur_ccd = ccd.Concord(
            adata=adata, input_feature=feature_list, 
            preload_dense=True,domain_key=batch_key, **kwargs
        ) 

        # Encode data, saving the latent embedding in adata.obsm['Concord']
        cur_ccd.fit_transform(output_key='X_concord')
        add_reference(adata,'Concord','batch correction with Concord')
        return cur_ccd
    elif methods in ('cca', 'seurat_cca', 'CCA'):
        # Helper raises ImportError with a pyccasc install hint on its own
        # if cca_py is missing; no need for a double import guard here.
        _run_cca_batch_correction(
            adata, batch_key=batch_key, n_pcs=n_pcs, **kwargs,
        )
        add_reference(adata, 'CCA', 'batch correction with Seurat CCA '
                                     '(pyccasc)')
        return adata
    elif methods in ('rpca', 'seurat_rpca', 'RPCA'):
        from ._rpca import rpca_integrate
        # RPCA anchors correct an existing joint PCA (Seurat orig.reduction).
        # Default to omicverse's scaled PCA, computing it if absent — same
        # convention as the harmony branch.
        rpca_kwargs = dict(kwargs)
        orig_rep = rpca_kwargs.pop('orig_rep', None)
        if orig_rep is None:
            if 'scaled|original|X_pca' in adata.obsm.keys():
                orig_rep = 'scaled|original|X_pca'
            elif use_rep in adata.obsm.keys():
                orig_rep = use_rep
            elif use_rep == 'scaled|original|X_pca':
                scale(adata)
                pca(adata, layer='scaled', n_pcs=n_pcs)
                orig_rep = 'scaled|original|X_pca'
        rpca_integrate(
            adata, batch_key=batch_key, n_pcs=n_pcs,
            orig_rep=orig_rep, **rpca_kwargs,
        )
        return adata
    else:
        print('Not supported')


_CCA_HELPER_KWARGS = {
    "reference", "layer", "features", "standardize_inputs",
    "seed", "key_added",
}


def _run_cca_batch_correction(
    adata: anndata.AnnData,
    *,
    batch_key: str,
    n_pcs: int = 50,
    reference: Optional[str] = None,
    layer: Optional[str] = None,
    features: Optional[list] = None,
    standardize_inputs: bool = True,
    seed: Optional[int] = None,
    key_added: str = "X_cca",
    **kwargs,
) -> anndata.AnnData:
    """Seurat-style CCA integration over an arbitrary batch count.

    Delegates the 2-batch case to ``cca_py.run_cca_anndata`` and
    generalises to N batches by running pairwise CCA against a
    reference batch (the largest batch by default, or the batch named
    by ``reference=``). Non-reference batches get their own CCA
    projection; the reference uses its projection from the *first*
    pair.

    All per-batch embeddings are concatenated in the original sample
    order and written to ``adata.obsm[key_added]`` so downstream
    neighbours/UMAP/clustering can consume it like any other embedding.

    Parameters
    ----------
    adata, batch_key
        As in :func:`batch_correction`.
    n_pcs
        Forwarded as ``num_cc`` to pyccasc.
    reference
        Name of the batch to use as the CCA reference when there are
        >2 batches. ``None`` → use the largest batch. Ignored for the
        2-batch case (either batch can play the reference role).
    layer, features, standardize_inputs, key_added
        Forwarded to ``cca_py.run_cca_anndata``.
    seed
        Reproducibility seed forwarded to pyccasc. ``None`` (default)
        matches the scikit-learn ``random_state=None`` convention —
        pass e.g. ``seed=42`` for deterministic runs.
    **kwargs
        Unrecognised kwargs are warned and dropped rather than raised,
        to match the loose-kwarg contract of the other
        :func:`batch_correction` backends.
    """
    import warnings
    try:
        from cca_py import run_cca_anndata
    except ImportError as exc:
        raise ImportError(
            "methods='cca' needs the pyccasc package "
            "(pure-Python Seurat RunCCA, no R required). "
            "Install with `pip install pyccasc`."
        ) from exc

    if kwargs:
        warnings.warn(
            "methods='cca' ignored unknown kwargs "
            f"{sorted(kwargs)} — supported kwargs are "
            f"{sorted(_CCA_HELPER_KWARGS)}.",
            UserWarning, stacklevel=3,
        )

    batches = adata.obs[batch_key].astype(str).to_numpy()
    # order-preserving deduplification: keep batches in first-seen order
    unique = list(dict.fromkeys(batches))
    if len(unique) < 2:
        raise ValueError(
            f"{batch_key!r} has {len(unique)} unique values; "
            f"CCA integration requires ≥2 batches."
        )

    # One split per batch label (keeps indices so we can re-assemble)
    splits = {b: adata[batches == b].copy() for b in unique}

    if len(unique) == 2:
        a_name, b_name = unique
        run_cca_anndata(
            splits[a_name], splits[b_name],
            features=features, layer=layer,
            num_cc=n_pcs,
            standardize_inputs=standardize_inputs,
            key_added=key_added,
            seed=seed,
        )
    else:
        # Choose reference batch: explicit arg > largest
        if reference is not None:
            if reference not in splits:
                raise KeyError(
                    f"reference={reference!r} not in {batch_key!r} "
                    f"(have {unique})"
                )
            ref_name = reference
        else:
            ref_name = max(unique, key=lambda b: splits[b].n_obs)
        non_ref = [b for b in unique if b != ref_name]
        # Invariant: we keep the reference's CCA projection from the
        # **first** pair. Subsequent pairs would overwrite it inside
        # splits[ref_name].obsm, so we snapshot after the first call and
        # restore at the end. ``ref_emb`` is guaranteed to be populated
        # because non_ref is non-empty (len(unique) >= 3 here).
        ref_emb = None
        for i, other in enumerate(non_ref):
            # run_cca_anndata writes both adata1 (ref) and adata2 (other)
            # — we reuse the reference's projection from the *first*
            # pair, discard it on subsequent pairs. Strict anchors à la
            # Seurat 3 are a v2 follow-up.
            run_cca_anndata(
                splits[ref_name], splits[other],
                features=features, layer=layer,
                num_cc=n_pcs,
                standardize_inputs=standardize_inputs,
                key_added=key_added,
                seed=seed,
            )
            if i == 0:
                ref_emb = splits[ref_name].obsm[key_added].copy()
        # Re-install the first-pair ref embedding (overwritten by later
        # pair calls).
        splits[ref_name].obsm[key_added] = ref_emb

    # Concatenate back to the original adata in sample order
    emb_dim = next(iter(splits.values())).obsm[key_added].shape[1]
    full = np.empty((adata.n_obs, emb_dim), dtype=np.float64)
    for b in unique:
        idx = np.where(batches == b)[0]
        full[idx] = splits[b].obsm[key_added]
    adata.obsm[key_added] = full
    adata.uns.setdefault("cca", {})[key_added] = {
        "num_cc": int(n_pcs),
        "n_batches": len(unique),
        "batches": unique,
        "reference": reference,
    }
    return adata

    

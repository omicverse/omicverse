"""Lazy adapters for independently validated roadmap #989 packages."""
import numpy as np
import pandas as pd
from scipy import sparse
from ._optional import import_optional_module
from ._registry import register_function


_BACKENDS = {
    'pyruvseq': 'py-ruvseq-omicverse==0.1.0',
    'pysparkx': 'py-sparkx-omicverse==0.1.0',
    'pymusic_deconv': 'py-music-omicverse==0.1.0',
    'pymuscat': 'py-muscat-omicverse==0.1.0',
}


def _backend(module):
    return import_optional_module(
        module, feature='Roadmap #989 analysis', dependencies=(module,),
        install_hint=f'Install the tested backend with `python -m pip install "{_BACKENDS[module]}"`.',
    )


@register_function(aliases=['RUVg'],category='bulk',description='Estimate unwanted factors from control genes')
def ruvg(expression,controls,k=1,*,drop=0,center=True,round_counts=True,epsilon=1.,tolerance=1e-8,is_log=False):
    """Run pyruvseq RUVg on genes x samples; return W and reconstructed expression.

    Use W in the DE design with original counts. Input is not modified.
    DataFrame controls are gene labels; array controls are zero-based positions.
    """
    return _backend('pyruvseq').ruvg(expression,controls,k,drop=drop,center=center,
        round_counts=round_counts,epsilon=epsilon,tolerance=tolerance,is_log=is_log)


@register_function(aliases=['RUVr'],category='bulk',description='Estimate unwanted factors from first-pass residuals')
def ruvr(expression,controls,k,residuals,**kwargs):
    """Run RUVr with caller-supplied genes x samples residuals; no model is fitted here."""
    return _backend('pyruvseq').ruvr(expression,controls,k,residuals,**kwargs)


@register_function(aliases=['RUVs'],category='bulk',description='Estimate unwanted factors from replicate samples')
def ruvs(expression,controls,k,replicate_indices,**kwargs):
    """Run RUVs; replicate indices are zero-based sample positions padded with -1."""
    return _backend('pyruvseq').ruvs(expression,controls,k,replicate_indices,**kwargs)


@register_function(aliases=['MuSiC reference'],category='bulk',description='Build a multi-donor MuSiC reference')
def music_basis(counts,clusters,samples,**kwargs):
    """Build MuSiC reference tables from genes x cells and aligned metadata Series."""
    return _backend('pymusic_deconv').music_basis(counts,clusters,samples,**kwargs)


@register_function(aliases=['MuSiC'],category='bulk',description='Deconvolve bulk samples with a multi-donor reference')
def music_prop(bulk,reference,clusters,samples,**kwargs):
    """Run the default MuSiC path; returns proportions and convergence diagnostics."""
    return _backend('pymusic_deconv').music_prop(bulk,reference,clusters,samples,**kwargs)


@register_function(aliases=['SPARK-X'],category='space',description='Test spatially variable genes with SPARK-X')
def sparkx(adata,*,layer='counts',spatial_key='spatial',covariates=None,option='mixture'):
    """Run SPARK-X and return gene-labelled tables without modifying AnnData.

    Counts are transposed without densifying sparse input. Covariate DataFrames
    must match obs_names in order. Zero-count rows/columns are filtered by the
    backend; returned keep_spots/keep_genes masks refer to the input axes.
    """
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError('Cell and gene names must be unique.')
    if spatial_key not in adata.obsm:raise KeyError(spatial_key)
    if isinstance(covariates,pd.DataFrame) and not covariates.index.equals(adata.obs_names):
        raise ValueError('Covariates must match obs_names in order.')
    counts=adata.X if layer is None else adata.layers[layer]
    result=_backend('pysparkx').sparkx(counts.T,adata.obsm[spatial_key],covariates,option=option)
    genes=adata.var_names[np.asarray(result['keep_genes'],dtype=bool)]
    for key in ('statistics','pvalues','results'):
        result[key].index=genes.copy()
    return result


@register_function(aliases=['muscat aggregateData'],category='single',description='Aggregate cells into sample-level pseudobulks')
def muscat_aggregate(adata,*,cluster_key='cluster_id',sample_key='sample_id',group_key='group_id',layer='counts'):
    """Sum cluster x sample counts, preserving missing combinations and replication."""
    return _backend('pymuscat').aggregate_data(adata,cluster_key=cluster_key,sample_key=sample_key,group_key=group_key,layer=layer)


@register_function(aliases=['muscat pbDS'],category='single',description='Run pseudobulk differential-state tests')
def muscat_pb_ds(pseudobulk,design,contrasts,*,method='edgeR',min_cells=10,filter='both'):
    """Run edgeR or limma-voom; return local/global FDR and skipped-cluster reasons."""
    return _backend('pymuscat').pb_ds(pseudobulk,design,contrasts,method=method,min_cells=min_cells,filter=filter)

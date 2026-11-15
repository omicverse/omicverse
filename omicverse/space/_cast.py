from scipy.sparse import csr_matrix,issparse
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from .._registry import register_function

@register_function(
    aliases=["CAST", "空间CAST", "CAST整合", "spatial_cast", "空间样本嵌入"],
    category="space",
    description="Learn shared spatial embeddings across multiple samples with CAST",
    prerequisites={
        "optional_functions": ["pp.preprocess", "space.svg"]
    },
    requires={
        "obs": ["sample_key"],
        "obsm": ["spatial"]
    },
    produces={
        "obsm": ["X_cast"]
    },
    auto_fix="none",
    examples=[
        "adata = ov.space.CAST(adata, sample_key='sample', basis='spatial', layer='norm_1e4')",
        "cast_emb = adata.obsm['X_cast']",
    ],
    related=["space.pySTAligner", "space.pySTAGATE", "space.svg"],
)
def CAST(adata, sample_key=None, basis='spatial', layer='norm_1e4',
         output_path='output/CAST_Mark', gpu_t=0, device='cuda:0', **kwargs):
    """
    CAST (Cell Annotation for Spatial Transcriptomics) embedding for multiple spatial samples.

    This function implements the CAST algorithm to learn unified embeddings across
    multiple spatial transcriptomics samples, enabling joint analysis and integration
    of spatial data from different sources.

    Parameters
    ----------
    adata : anndata.AnnData
        Multi-sample spatial AnnData object.
    sample_key : str, optional
        Column in ``adata.obs`` identifying sample/batch labels.
    basis : str, default="spatial"
        Key in ``adata.obsm`` storing spatial coordinates.
    layer : str, default="norm_1e4"
        Layer key containing normalized expression used by CAST.
    output_path : str, default="output/CAST_Mark"
        Directory for CAST intermediate files and outputs.
    gpu_t : int, default=0
        GPU index used by CAST backend.
    device : str, default="cuda:0"
        Torch device string passed to CAST backend.
    **kwargs
        Extra keyword arguments forwarded to ``CAST_MARK``.

    Returns
    -------
    anndata.AnnData
        Updated AnnData with embedding saved in ``adata.obsm['X_cast']``.

    Notes:
        - Requires normalized expression data in specified layer
        - GPU acceleration is enabled by default
        - Creates output directory if it doesn't exist
        - Progress is shown with tqdm progress bars

    Examples:
        >>> import omicverse as ov
        >>> # Load multiple samples
        >>> adata = ov.read('spatial_samples.h5ad')
        >>> # Run CAST integration
        >>> adata = ov.space.CAST(
        ...     adata,
        ...     sample_key='sample_id',
        ...     basis='spatial',
        ...     layer='normalized_counts'
        ... )
        >>> # Access CAST embeddings
        >>> cast_embeddings = adata.obsm['X_cast']
    """
    if issparse(adata.obsm[basis]):
        adata.obsm[basis] = adata.obsm[basis].toarray()
    adata.obs['x'] = adata.obsm[basis][:,0]
    adata.obs['y'] = adata.obsm[basis][:,1]

    # Get the coordinates and expression data for each sample
    samples = np.unique(adata.obs[sample_key]) # used samples in adata
    coords_raw = {sample_t: np.array(adata.obs[['x','y']])[adata.obs[sample_key] == sample_t] for sample_t in samples}
    exp_dict = {sample_t: adata[adata.obs[sample_key] == sample_t].layers[layer] for sample_t in samples}

    os.makedirs(output_path, exist_ok=True)
    
    from ..external.CAST import CAST_MARK
    embed_dict = CAST_MARK(coords_raw, exp_dict, output_path, gpu_t=gpu_t, device=device, **kwargs)

    adata.obsm['X_cast'] = np.zeros((adata.shape[0], 512))
    
    adata.obsm['X_cast'] = pd.DataFrame(adata.obsm['X_cast'], index=adata.obs.index)
    for key in tqdm(embed_dict.keys()):
        adata.obsm['X_cast'].loc[adata.obs[sample_key]==key] += embed_dict[key].cpu().numpy()
    adata.obsm['X_cast'] = adata.obsm['X_cast'].values
    print('CAST embedding is saved in adata.obsm[\'X_cast\']')
    return adata

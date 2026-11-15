def _get_starfysh_modules():
    from ..external.starfysh import AA, _starfysh as sf_model, plot_utils, post_analysis, utils

    return AA, utils, plot_utils, post_analysis, sf_model

class STARFYSH:
    """
    STARFYSH spatial transcriptomics analysis class.
    
    A comprehensive tool for analyzing spatial transcriptomics data, providing
    advanced capabilities for spatial pattern detection and cell type deconvolution.

    Arguments:
        adata: AnnData
            Annotated data matrix containing spatial transcriptomics data.
            Must contain spatial coordinates in adata.obsm['spatial'].

    Attributes:
        adata: AnnData
            The input AnnData object containing spatial data.

    Notes:
        - STARFYSH requires properly formatted spatial transcriptomics data
        - The input AnnData object should contain normalized gene expression data
        - Spatial coordinates should be stored in adata.obsm['spatial']

    Examples:
        >>> import omicverse as ov
        >>> # Load spatial transcriptomics data
        >>> adata = ov.read('spatial_data.h5ad')
        >>> # Initialize STARFYSH object
        >>> starfysh = ov.space.STARFYSH(adata)
    """

    def __init__(self, adata):
        """
        Initialize STARFYSH analysis object.
        
        Arguments:
            adata: AnnData
                Annotated data matrix containing spatial transcriptomics data.
                Must contain spatial coordinates in adata.obsm['spatial'].
        """
        self.adata = adata
        self.AA, self.utils, self.plot_utils, self.post_analysis, self.sf_model = _get_starfysh_modules()

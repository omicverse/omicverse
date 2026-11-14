import os 
import pandas as pd
import numpy as np
from typing import Optional
from ._SCSA import Process,Annotator
import sys
import argparse
import gzip
import time
import requests
import anndata
from ..pp._preprocess import scale
from ..utils import gen_mpl_labels

from .._settings import add_reference, Colors, EMOJI
from .._registry import register_function
from ..datasets import download_data_requests

DATA_DOWNLOAD_LINK_DICT = {
    'whole':{
        'figshare':'https://figshare.com/ndownloader/files/37262710',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/pySCSA_2023.db',
    },
    'pySCSA_2023_v2_plus':{
        'figshare':'https://figshare.com/ndownloader/files/41369037',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/pySCSA_2023_v2_plus.db',
    },

}


def get_anno_dataset_url(dataset_name: str, prefer_stanford: bool = True) -> str:
    """Get URL for an annotation dataset by name, preferring Stanford over Figshare.

    Parameters
    ----------
    dataset_name : str
        Annotation dataset key (for example ``'whole'`` or ``'pySCSA_2023_v2_plus'``).
    prefer_stanford : bool
        Whether to prioritize Stanford mirror over Figshare.

    Returns
    -------
    str
        Download URL for selected annotation dataset.

    Raises
    ------
    ValueError
        If ``dataset_name`` is not defined in ``DATA_DOWNLOAD_LINK_DICT``.
    """
    if dataset_name not in DATA_DOWNLOAD_LINK_DICT:
        raise ValueError(f"Dataset '{dataset_name}' not found in DATA_DOWNLOAD_LINK_DICT")

    dataset_urls = DATA_DOWNLOAD_LINK_DICT[dataset_name]

    if prefer_stanford and 'stanford' in dataset_urls:
        print(f"{Colors.CYAN}Using Stanford mirror for {dataset_name}{Colors.ENDC}")
        return dataset_urls['stanford']
    elif 'figshare' in dataset_urls:
        if prefer_stanford:
            print(f"{Colors.WARNING}{EMOJI['warning']} Stanford link not available for {dataset_name}, using Figshare{Colors.ENDC}")
        return dataset_urls['figshare']
    else:
        raise ValueError(f"No valid URL found for dataset '{dataset_name}'")

def global_imports(modulename,shortname = None, asfunction = False):
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:        
        globals()[shortname] = __import__(modulename)

metatime_install=False
def check_metatime():
    r"""Check if metatime is installed and import it.

    Returns
    -------
    None
        Raises ``ImportError`` if ``metatime`` is unavailable.

    Notes
    -----
    `metatime` (current release 1.3.0) still references the deprecated
    `np.float` alias that NumPy removed in 1.24. We restore the alias as
    a compat shim before the import so the package keeps working on
    modern NumPy — no upstream fork required.
    """
    global metatime_install
    import numpy as _np
    if not hasattr(_np, 'float'):
        _np.float = float       # noqa: deprecated alias for compat
    if not hasattr(_np, 'int'):
        _np.int = int           # noqa: same for completeness
    if not hasattr(_np, 'bool'):
        _np.bool = bool         # noqa: same for completeness
    try:
        import metatime
        metatime_install=True
        print('metatime have been install version:',metatime.__version__)
    except ImportError:
        raise ImportError(
            'Please install the metatime: `pip install metatime`.'
        )


# Deprecated: data_downloader has been replaced by download_data_requests
# Use get_anno_dataset_url() and download_data_requests() instead


def data_preprocess(adata,clustertype='leiden',
                    path='temp/rna.csv',layer='scaled',rank_rep=False):
    r"""Data preprocess for SCSA.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData containing cluster labels and expression matrix.
    clustertype : str
        Column in ``adata.obs`` used as cluster assignment.
    path : str
        Output CSV path for ranked marker table.
    layer : str
        Reserved layer argument for compatibility.
    rank_rep : bool
        Whether to recompute ``rank_genes_groups`` even if present.

    Returns
    -------
    pd.DataFrame
        Marker-ranking table exported for SCSA.
    """
    dirname, _ = os.path.split(path)
    try:
        if not os.path.isdir(dirname):
            print("......Creating directory {}".format(dirname))
            os.makedirs(dirname, exist_ok=True)
    except OSError as e:
        print("......Unable to create directory {}. Reason {}".format(dirname,e))

    from ._sc_compat import rank_genes_groups
    if rank_rep==False and 'rank_genes_groups' not in adata.uns.keys():
        rank_genes_groups(adata, clustertype, method='wilcoxon')
    elif rank_rep==True:
        rank_genes_groups(adata, clustertype, method='wilcoxon')
    else:
        pass
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    dat = pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'logfoldchanges','scores','pvals']})
    dat.to_csv(path)
    return dat

def __cell_annotate(data,
                foldchange=1.5,pvalue=0.05,
                output='temp/rna_anno.txt',
                outfmt='txt',Gensymbol=True,
                species='Human',weight=100,tissue='All',
                celltype='normal',norefdb=False,noprint=True,list_tissue=False):
    r"""Cell annotation by SCSA.
    
    Parameters
    ----------
    data : pd.DataFrame
        Marker/rank table exported from AnnData for SCSA CLI input.
    foldchange : float
        Fold-change threshold used by SCSA for marker filtering.
    pvalue : float
        P-value threshold used by SCSA.
    output : str
        Output file path for SCSA annotation report.
    outfmt : str
        Output format used by SCSA CLI.
    Gensymbol : bool
        Whether gene symbols are used in input markers.
    species : str
        Species label used by SCSA database lookup.
    weight : int
        Weight parameter forwarded to SCSA scoring.
    tissue : str
        Tissue filter used by SCSA.
    celltype : str
        Cell-type mode argument for SCSA.
    norefdb : bool
        Whether to disable reference database scoring.
    noprint : bool
        Whether to suppress CLI printing.
    list_tissue : bool
        Whether to list available tissues and exit.
    
    Returns
    -------
    pd.DataFrame
        SCSA annotation result table.
    """
    data.to_csv('temp/rna.csv')

    # Get URL from dataset dict and download using Stanford mirror
    url = get_anno_dataset_url('whole')
    model_path = download_data_requests(url=url, file_path='whole.db', dir='./temp')
    print(f'{Colors.CYAN}......Auto annotate cell{Colors.ENDC}')
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', default = "temp/rna.csv")
    parser.add_argument('-o', '--output',default=output)
    parser.add_argument('-d', '--db', default = model_path,)
    parser.add_argument('-s', '--source', default = "scanpy",)
    parser.add_argument('-c', '--cluster', default = "all",)
    parser.add_argument('-f',"--fc",default = "1.5",)
    parser.add_argument('-fc',"--foldchange",default =foldchange,)
    parser.add_argument('-p',"--pvalue",default = pvalue,)
    parser.add_argument('-w',"--weight",default = weight,)
    parser.add_argument('-g',"--species",default = species,)
    parser.add_argument('-k',"--tissue",default = tissue,)
    parser.add_argument('-m', '--outfmt', default = outfmt, )
    parser.add_argument('-T',"--celltype",default = celltype,)
    parser.add_argument('-t', '--target', default = "cellmarker",)
    parser.add_argument('-E',"--Gensymbol",action = "store_true",default=Gensymbol,)
    parser.add_argument('-N',"--norefdb",action = "store_true",default=norefdb,)
    parser.add_argument('-b',"--noprint",action = "store_true",default=noprint,)
    parser.add_argument('-l',"--list_tissue",action = "store_true",default = False,)
    parser.add_argument('-M', '--MarkerDB',)
    args = parser.parse_args()

    p = Process()
    if args.list_tissue:
        p.list_tissue(args)
    p.run_cmd(args)

    result=pd.read_csv('temp/rna_anno.txt',sep='\t')
    return result


def __cell_anno_print(anno):
    r"""Print the annotation result.

    Parameters
    ----------
    anno : pd.DataFrame
        SCSA annotation result table.

    Returns
    -------
    None
    """
    for i in set(anno['Cluster']):
        test=anno.loc[anno['Cluster']==i].iloc[:2]
        if len(test) >= 2 and pd.notna(test.iloc[0]['Z-score']) and pd.notna(test.iloc[1]['Z-score']):
            if test.iloc[0]['Z-score']>test.iloc[1]['Z-score']*2:
                print('Nice:Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,test.iloc[0]['Cell Type'],
                                                            np.around(test.iloc[0]['Z-score'],3)))
            else:
                print('Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,('|').join(test['Cell Type'].values.tolist()),
                                                            ('|').join(np.around(test['Z-score'].values,3).astype(str).tolist())))
        elif len(test) >= 1 and pd.notna(test.iloc[0]['Z-score']):
            print('Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,test.iloc[0]['Cell Type'],
                                                        np.around(test.iloc[0]['Z-score'],3)))
        else:
            print('Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,'Unknown','NaN'))

def scanpy_lazy(adata:anndata.AnnData,min_genes:int=200,min_cells:int=3,drop_doublet:bool=True,
                n_genes_by_counts:int=4300,pct_counts_mt:int=25,
                target_sum:float=1e4,min_mean:float=0.0125, max_mean:int=3, min_disp:float=0.5,max_value:int=10,
                n_comps:int=100, svd_solver:str="auto",
                n_neighbors:int=15, random_state:int = 112, n_pcs:int=50,
                )->anndata.AnnData:
    r"""Scanpy lazy analysis pipeline.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Input AnnData for quick QC and clustering.
    min_genes : int
        Minimum genes per cell during QC filtering.
    min_cells : int
        Minimum cells per gene during QC filtering.
    drop_doublet : bool
        Whether to run scrublet and remove predicted doublets.
    n_genes_by_counts : int
        Upper threshold for ``obs.n_genes_by_counts``.
    pct_counts_mt : int
        Upper threshold for mitochondrial fraction percentage.
    target_sum : float
        Library-size normalization target.
    min_mean : float
        Minimum mean expression for HVG selection.
    max_mean : int
        Maximum mean expression for HVG selection.
    min_disp : float
        Minimum dispersion for HVG selection.
    max_value : int
        Clipping value used in scaling.
    n_comps : int
        Number of principal components.
    svd_solver : str
        SVD solver used in PCA.
    n_neighbors : int
        Number of neighbors for graph construction.
    random_state : int
        Random seed used in neighbors computation.
    n_pcs : int
        Number of PCs used in neighbor graph.

    Returns
    -------
    anndata.AnnData
        Processed AnnData with QC, HVG, PCA, neighbors, and UMAP results.
    """
    from ..pp import (filter_cells, filter_genes, scrublet, calculate_qc_metrics,
                      normalize_total, log1p, highly_variable_genes, neighbors,
                      leiden, umap)
    from ._sc_compat import scale_X, pca_X, paga, paga_pos
    #filter cells and genes
    filter_cells(adata, min_genes=min_genes)
    filter_genes(adata, min_cells=min_cells)
    #filter the doublets cells
    if drop_doublet:
        scrublet(adata) #estimates doublets
        adata = adata[adata.obs['predicted_doublet'] == False] #do the actual filtering
    #calculate the proportion of mito-genes
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.n_genes_by_counts < n_genes_by_counts, :]
    adata = adata[adata.obs.pct_counts_mt < pct_counts_mt, :]
    #save the raw counts
    adata.layers["counts"] = adata.X.copy()
    #normalization, the max counts of total_counts is 20000 means the amount is 10e4
    normalize_total(adata, target_sum=target_sum)
    #log
    log1p(adata)
    #select high-variable genes
    highly_variable_genes(adata, flavor='seurat', min_mean=min_mean, max_mean=max_mean, min_disp=min_disp)
    #save and filter
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable].copy()
    #scale
    scale_X(adata, max_value=max_value)
    #pca analysis
    pca_X(adata, n_comps=n_comps, svd_solver=svd_solver)
    #cell neighbors graph construct
    neighbors(adata, n_neighbors=n_neighbors, random_state = random_state, n_pcs=n_pcs)
    #umap
    leiden(adata, random_state=0, backend='python')
    paga(adata)
    paga_pos(adata)
    # PAGA-initialised UMAP (scanpy's ``init_pos='paga'``, seeded with 0)
    from ..pp._compat import get_init_pos_from_paga
    umap(adata, init_pos=get_init_pos_from_paga(adata, random_state=0, neighbors_key='neighbors'),
         random_state=0)
    return adata

@register_function(
    aliases=["字典注释", "scanpy_cellanno_from_dict", "manual_annotation", "手动注释", "字典映射注释"],
    category="single",
    description="Manual cell type annotation from cluster-to-celltype dictionary mapping",
    prerequisites={
        'functions': ['leiden']
    },
    requires={
        'obs': []  # Dynamic: requires user-specified clustertype column
    },
    produces={
        'obs': []  # Dynamic: creates {anno_name}_celltype column
    },
    auto_fix='none',
    examples=[
        "# Basic manual annotation from dictionary",
        "cluster2annotation = {",
        "    '0': 'T cell', '1': 'B cell', '2': 'Monocyte',",
        "    '3': 'NK cell', '4': 'Dendritic cell'",
        "}",
        "ov.single.scanpy_cellanno_from_dict(adata, anno_dict=cluster2annotation,",
        "                                    clustertype='leiden')",
        "# Custom annotation key",
        "ov.single.scanpy_cellanno_from_dict(adata, anno_dict=cluster2annotation,",
        "                                    clustertype='leiden',",
        "                                    key_added='manual_celltype')",
        "# Compare with automatic annotation",
        "ov.pl.embedding(adata, color=['manual_celltype', 'scsa_celltype'])"
    ],
    related=["single.pySCSA", "single.get_celltype_marker", "utils.embedding"]
)
def scanpy_cellanno_from_dict(adata:anndata.AnnData,
                               anno_dict:dict,
                               anno_name:str='major',
                               clustertype:str='leiden',
                               )->None:
    r"""Add cell type annotation from dict to anndata object.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object to which annotation labels are added.
    anno_dict : dict
        Mapping from cluster label to cell-type name.
    anno_name : str
        Prefix used to create output ``obs`` column ``{anno_name}_celltype``.
    clustertype : str
        Cluster column in ``adata.obs`` used as mapping key.

    Returns
    -------
    None
    """

    adata.obs[anno_name+'_celltype'] = adata.obs[clustertype].map(anno_dict).astype('category')
    print('...cell type added to {}_celltype on obs of anndata'.format(anno_name))

@register_function(
    aliases=["细胞类型标记基因", "get_celltype_marker", "celltype_markers", "标记基因", "差异基因"],
    category="single",
    description="Extract cell type-specific marker genes from differential expression analysis",
    prerequisites={
        'functions': ['leiden']
    },
    requires={
        'obs': []  # Dynamic: requires user-specified clustertype column
    },
    produces={
        'uns': ['rank_genes_groups']
    },
    auto_fix='escalate',
    examples=[
        "# Get markers for all cell types",
        "marker_dict = ov.single.get_celltype_marker(adata,",
        "                                           clustertype='leiden')",
        "print(marker_dict.keys())  # Show available cell types",
        "# Get markers for specific cell type annotation",
        "marker_dict = ov.single.get_celltype_marker(adata,",
        "                                           clustertype='scsa_celltype')",
        "# View markers for specific cell type",
        "b_cell_markers = marker_dict['B cell']",
        "print(b_cell_markers[:10])  # Top 10 markers",
        "# Use with plotting",
        "ov.pl.dotplot(adata, marker_dict, groupby='leiden')"
    ],
    related=["single.pySCSA", "single.scanpy_cellanno_from_dict", "pl.dotplot"]
)
def get_celltype_marker(adata:anndata.AnnData,
                            clustertype:str='leiden',
                            log2fc_min:int=2,scores_type='scores',
                            pval_cutoff:float=0.05,rank:bool=True,
                            key='rank_genes_groups',method='wilcoxon',
                            foldchange=None,topgenenumber=10,unique=True,
                            global_unique=False,use_raw:Optional[bool]=None,
                            layer:Optional[str]=None,**kwargs)->dict:
    r"""Get marker genes for each cluster/cell type.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData containing cluster annotations and expression matrix.
    clustertype : str
        Column in ``adata.obs`` used to define groups.
    log2fc_min : int
        Minimum log2 fold-change threshold when extracting DE markers.
    scores_type : str
        Statistic field used for thresholding (for example ``'scores'``).
    pval_cutoff : float
        Maximum adjusted p-value cutoff for retained markers.
    rank : bool
        Whether to run :func:`omicverse.single.find_markers` before extraction.
    key : str
        Key in ``adata.uns`` storing ranked-gene results.
    method : str
        Differential-expression method for ``rank_genes_groups``.
    foldchange : float or None
        Optional manual score threshold; auto-derived when ``None``.
    topgenenumber : int
        Maximum number of markers retained per cluster.
    unique : bool
        Whether to deduplicate markers within each cluster.
    global_unique : bool
        Whether to enforce uniqueness across all clusters.
    use_raw : Optional[bool]
        Forwarded to ``rank_genes_groups`` to control raw usage.
    layer : Optional[str]
        Layer passed to ``rank_genes_groups``.
    **kwargs
        Additional keyword arguments for ``rank_genes_groups``.

    Returns
    -------
    dict
        Dictionary mapping cluster labels to marker gene lists.
    """
    from ._sc_compat import rank_genes_groups, rank_genes_groups_df
    print('...get cell type marker')
    celltypes = sorted(adata.obs[clustertype].unique())
    cell_marker_dict={}
    if key not in adata.uns.keys() or rank:
        rg_kwargs = dict(method=method, key_added=key)
        if use_raw is not None:
            rg_kwargs['use_raw'] = use_raw
        if layer is not None:
            rg_kwargs['layer'] = layer
        rg_kwargs.update(kwargs)
        rank_genes_groups(adata, clustertype, **rg_kwargs)
    for celltype in celltypes:
        degs = rank_genes_groups_df(adata, group=celltype, key=key, log2fc_min=log2fc_min,
                                        pval_cutoff=pval_cutoff)
        foldp=np.histogram(degs[scores_type])
        if foldchange is None:
            try:
                foldchange=(foldp[1][np.where(foldp[1]>0)[0][-5]]+foldp[1][np.where(foldp[1]>0)[0][-6]])/2
            except:
                foldchange=degs[scores_type].mean()
                
        cellmarker=degs.loc[degs[scores_type]>foldchange]['names'].values[:topgenenumber]
        cell_marker_dict[celltype]=cellmarker
    if unique==True:
        for key in cell_marker_dict.keys():
            cell_marker_dict[key]=list(set(cell_marker_dict[key]))
    
    # Global uniqueness across all cell types
    if global_unique:
        used_genes = set()
        for celltype in celltypes:
            if celltype in cell_marker_dict:
                # Filter out genes that have been used in previous cell types
                unique_genes = [gene for gene in cell_marker_dict[celltype] if gene not in used_genes]
                cell_marker_dict[celltype] = unique_genes
                used_genes.update(unique_genes)
    
    return cell_marker_dict






@register_function(
    aliases=["单细胞注释", "pySCSA", "cell_annotation", "细胞类型注释", "自动注释"],
    category="single",
    description="Automated cell type annotation using SCSA (Single Cell Signature Analysis) with multiple databases. IMPORTANT: Use 'clustertype' parameter (NOT 'cluster') in cell_auto_anno()!",
    prerequisites={
        'optional_functions': ['preprocess', 'leiden']
    },
    requires={
        'var': [],  # Flexible - works with raw or processed data
        'obs': []   # Clustering recommended but not required
    },
    produces={
        'obs': ['scsa_celltype']
    },
    auto_fix='none',
    examples=[
        "# CRITICAL: Use clustertype='leiden', NOT cluster='leiden'!",
        "# Step 1: Initialize pySCSA",
        "scsa = ov.single.pySCSA(adata, foldchange=1.5, pvalue=0.01,",
        "                        species='Human', tissue='All', target='cellmarker')",
        "",
        "# Step 2: Run annotation - NOTE: parameter is 'clustertype'!",
        "anno = scsa.cell_anno(clustertype='leiden', cluster='all')",
        "",
        "# Step 3: Add annotations to adata.obs",
        "scsa.cell_auto_anno(adata, clustertype='leiden', key='scsa_celltype')",
        "# Results are now in adata.obs['scsa_celltype']",
        "",
        "# WRONG - DO NOT USE:",
        "# scsa.cell_auto_anno(adata, cluster='leiden')  # ERROR! 'cluster' is NOT valid!",
        "",
        "# Using PanglaoDB database",
        "scsa = ov.single.pySCSA(adata, target='panglaodb', tissue='All')",
        "anno = scsa.cell_anno(clustertype='leiden', cluster='all')",
        "scsa.cell_auto_anno(adata, clustertype='leiden', key='scsa_panglaodb')"
    ],
    related=["single.scanpy_cellanno_from_dict", "single.get_celltype_marker", "utils.embedding"]
)
class pySCSA(object):
    """
    Automated cell-type annotation using SCSA marker-enrichment scoring.

    Parameters
    ----------
    adata : anndata.AnnData
        Query AnnData for cell-type annotation.
    foldchange : float, optional, default=1.5
        Fold-change cutoff for marker filtering.
    pvalue : float, optional, default=0.05
        P-value cutoff for marker filtering.
    output : str, optional, default='temp/rna_anno.txt'
        Output path for SCSA annotation report.
    model_path : str, optional, default=''
        Path to local SCSA database/model.
    outfmt : str, optional, default='txt'
        Output format for intermediate annotation report.
    Gensymbol : bool, optional, default=True
        Whether gene symbols are used as identifiers.
    species : str, optional, default='Human'
        Species used for marker database matching.
    weight : int, optional, default=100
        Marker-weight scaling factor used by SCSA scoring.
    tissue : str, optional, default='All'
        Tissue filter for marker database query.
    target : str, optional, default='cellmarker'
        Marker database target (for example ``'cellmarker'`` or ``'panglaodb'``).
    celltype : str, optional, default='normal'
        Annotation context/type mode used by SCSA.
    norefdb : bool, optional, default=False
        If ``True``, skip reference database matching.
    cellrange : str, optional, default=None
        Optional range/filter for cell selection.
    noprint : bool, optional, default=True
        If ``True``, suppress verbose console output.
    list_tissue : bool, optional, default=False
        If ``True``, list available tissues and exit.
    tissuename : str, optional, default=None
        Compatibility alias for ``tissue``.
    speciename : str, optional, default=None
        Compatibility alias for ``species``.
    
    Returns
    -------
    None
        Initializes SCSA annotation settings and database options.
    
    Examples
    --------
    >>> # CRITICAL: Use clustertype='leiden', NOT cluster='leiden'!
    """

    def __init__(self,adata:anndata.AnnData,
                foldchange:float=1.5,pvalue:float=0.05,
                output:str='temp/rna_anno.txt',
                model_path:str='',
                outfmt:str='txt',Gensymbol:bool=True,
                species:str='Human',weight:int=100,tissue:str='All',target:str='cellmarker',
                celltype:str='normal',norefdb:bool=False,cellrange:str=None,
                noprint:bool=True,list_tissue:bool=False,
                # Compatibility aliases used by older prompts/agents
                tissuename:str=None,speciename:str=None) -> None:

        r"""
        Initialize SCSA annotation workflow configuration.

        Parameters
        ----------
        adata : anndata.AnnData
            Query AnnData object.
        foldchange : float
            Fold-change threshold used for marker filtering.
        pvalue : float
            P-value threshold used for marker filtering.
        output : str
            Output path of annotation report.
        model_path : str
            Local SCSA database path. If empty, downloads default database.
        outfmt : str
            Output format for SCSA report.
        Gensymbol : bool
            Whether input gene identifiers are gene symbols.
        species : str
            Species used for marker-database lookup.
        weight : int
            SCSA weighting parameter.
        tissue : str
            Tissue filter used for database matching.
        target : str
            Marker database target (for example ``cellmarker``).
        celltype : str
            Cell-type mode used by SCSA.
        norefdb : bool
            Whether to disable reference database.
        cellrange : str or None
            Optional lineage restriction (for example T-cell subtypes only).
        noprint : bool
            Whether to suppress verbose output.
        list_tissue : bool
            Whether to list available tissues.
        tissuename : str or None
            Compatibility alias for ``tissue``.
        speciename : str or None
            Compatibility alias for ``species``.
        """

        #create temp directory
        try:
            if not os.path.isdir('temp'):
                print("...Creating directory {}".format('temp'))
                os.makedirs('temp', exist_ok=True)
        except OSError as e:
            print("...Unable to create directory {}. Reason {}".format('temp',e))

        self.adata=adata
        self.foldchange=foldchange
        self.pvalue=pvalue
        self.output=output
        self.outfmt=outfmt
        self.Gensymbol=Gensymbol
        # Backwards compatibility: accept alias field names some agents produce
        if speciename is not None:
            species = speciename
        if tissuename is not None and tissue == 'All':
            tissue = tissuename

        self.species=species
        self.weight=weight
        self.tissue=tissue
        self.celltype=celltype
        self.norefdb=norefdb
        self.noprint=noprint
        self.list_tissue=list_tissue
        self.target=target
        self.cellrange=cellrange
        if model_path =='':
            url = get_anno_dataset_url('pySCSA_2023_v2_plus')
            self.model_path = download_data_requests(url=url, file_path='pySCSA_2023_v2_plus.db', dir='./temp')
        else:
            self.model_path=model_path

    def get_model_tissue(self,species:str="Human")->None:
        r"""List all available tissues in the database.
        
        Parameters
        ----------
        species : str
            Species name used to query available tissues in marker database.

        Returns
        -------
        None
        """
        
        anno = Annotator(foldchange=self.foldchange,
                    weight=self.weight,
                    pvalue=self.pvalue,
                    tissue=self.tissue,
                    species=self.species,
                    target=self.target,
                    norefdb=self.norefdb,
                    MarkerDB=None,
                    db=self.model_path,
                    noprint=self.noprint,
                    input="temp/rna.csv",
                    output=self.output,
                    source="scanpy",
                    cluster='all',
                    fc=self.foldchange,
                    outfmt=self.outfmt,
                    celltype=self.celltype,
                    Gensymbol=self.Gensymbol,
                    list_tissue=self.list_tissue,
                    cellrange=self.cellrange)
        anno.load_pickle_module(self.model_path)
        anno.get_list_tissue(species)


    def cell_anno(self,clustertype:str='leiden',
                  cluster:str='all',rank_rep=False)->pd.DataFrame:
        r"""Annotate cell type for each cluster.
        
        Parameters
        ----------
        clustertype : str
            Cluster column in ``adata.obs`` used to compute markers.
        cluster : str
            Cluster subset for SCSA annotation; ``'all'`` annotates all clusters.
        rank_rep : bool
            Whether to rerun differential ranking before annotation.
        
        Returns
        -------
        pd.DataFrame
            SCSA annotation result table.
        """

        dat=data_preprocess(self.adata,clustertype=clustertype,path='temp/rna.csv',rank_rep=rank_rep)
        dat.to_csv('temp/rna.csv')

        print('...Auto annotate cell')
        
        output_path = self.output or 'temp/rna_anno.txt'

        p = Process()
        p.run_cmd_p(foldchange=self.foldchange,
                    weight=self.weight,
                    pvalue=self.pvalue,
                    tissue=self.tissue,
                    species=self.species,
                    target=self.target,
                    norefdb=self.norefdb,
                    MarkerDB=None,
                    db=self.model_path,
                    noprint=self.noprint,
                    input="temp/rna.csv",
                    output=output_path,
                    source="scanpy",
                    cluster=cluster,
                    fc=self.foldchange,
                    outfmt=self.outfmt,
                    celltype=self.celltype,
                    Gensymbol=self.Gensymbol,
                    list_tissue=self.list_tissue,
                    cellrange=self.cellrange)

        # Load results from the requested output path (fallback to default)
        result_path = output_path if output_path else 'temp/rna_anno.txt'
        result=pd.read_csv(result_path,sep='\t')
        self.result=result
        add_reference(self.adata,'pySCSA','cell annotation with SCSA')
        return result
    
    def cell_anno_print(self)->None:
        r"""Print the annotation result.

        Returns
        -------
        None
        """
        for i in set(self.result['Cluster']):
            test=self.result.loc[self.result['Cluster']==i].iloc[:2]
            if len(test) >= 2 and pd.notna(test.iloc[0]['Z-score']) and pd.notna(test.iloc[1]['Z-score']):
                if test.iloc[0]['Z-score']>test.iloc[1]['Z-score']*2:
                    print('Nice:Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,test.iloc[0]['Cell Type'],
                                                                np.around(test.iloc[0]['Z-score'],3)))
                else:
                    print('Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,('|').join(test['Cell Type'].values.tolist()),
                                                                ('|').join(np.around(test['Z-score'].values,3).astype(str).tolist())))
            elif len(test) >= 1 and pd.notna(test.iloc[0]['Z-score']):
                print('Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,test.iloc[0]['Cell Type'],
                                                            np.around(test.iloc[0]['Z-score'],3)))
            else:
                print('Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,'Unknown','NaN'))

    def cell_auto_anno(self,adata:anndata.AnnData,
                       clustertype:str='leiden',key='scsa_celltype')->None:
        r"""Add cell type annotation to anndata.obs['scsa_celltype'].
        
        Parameters
        ----------
        adata : anndata.AnnData
            AnnData object to receive annotation labels.
        clustertype : str
            Cluster column in ``adata.obs`` used as mapping key.
        key : str
            ``obs`` column name used to store predicted cell types.
        
        Returns
        -------
        None
        """
        # If annotation results are not present, run cell_anno first
        if not hasattr(self, "result") or self.result is None:
            self.cell_anno(clustertype=clustertype, cluster='all')

        test_li=[]
        for i in adata.obs[clustertype].value_counts().index:
            if int(i) in self.result['Cluster'].values:
                test_li.append(self.result.loc[self.result['Cluster']==int(i)].iloc[0]['Cell Type'])
            else:
                test_li.append('Unknown')
        scsa_anno=dict(zip([str(i) for i in adata.obs[clustertype].value_counts().index],
            test_li))
        adata.obs[key] = adata.obs[clustertype].map(scsa_anno).astype('category')
        print('...cell type added to {} on obs of anndata'.format(key))

    def get_celltype_marker(self,adata:anndata.AnnData,
                            clustertype:str='leiden',
                            log2fc_min:int=2,scores_type='scores',
                            pval_cutoff:float=0.05,rank:bool=True,
                            unique:bool=True,global_unique:bool=False)->dict:
        r"""Get marker genes for each clusters.
        
        Parameters
        ----------
        adata : anndata.AnnData
            AnnData containing clustering and expression data.
        clustertype : str
            Cluster column in ``adata.obs``.
        log2fc_min : int
            Minimum log2 fold-change for retained markers.
        pval_cutoff : float
            Maximum p-value threshold for retained markers.
        rank : bool
            Whether to rerun ``rank_genes_groups`` before extraction.
        scores_type : str
            Statistic used to rank markers (for example ``scores``).
        unique : bool
            Whether to deduplicate markers within each cluster.
        global_unique : bool
            Whether to enforce uniqueness across clusters.

        Returns
        -------
        dict
            Marker dictionary keyed by cluster label.
        """
        print('...get cell type marker')
        cell_marker_dict=get_celltype_marker(adata=adata,
                            clustertype=clustertype,
                            log2fc_min=log2fc_min,scores_type=scores_type,
                            pval_cutoff=pval_cutoff,rank=rank,
                            unique=unique,global_unique=global_unique)

        return cell_marker_dict
    


@register_function(
    aliases=['MetaTiME注释器', 'MetaTiME', 'tumor microenvironment meta-components'],
    category="single",
    description="MetaTiME wrapper for tumor microenvironment cell-state annotation using pretrained meta-components.",
    prerequisites={'optional_functions': ['pp.preprocess', 'pp.neighbors']},
    requires={'obs': ['cluster labels'], 'obsm': ['embedding (recommended)']},
    produces={'obs': ['MetaTiME annotations'], 'uns': ['MetaTiME scoring tables']},
    auto_fix='none',
    examples=['TiME_object = ov.single.MetaTiME(adata, mode="table")', 'TiME_object.predictTiME()'],
    related=['single.generate_scRNA_report', 'utils.plot_embedding_celltype']
)
class MetaTiME(object):
    """
    MetaTiME wrapper for tumor microenvironment cell-state annotation.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData to annotate with MetaTiME meta-components.
    mode : str, optional, default='table'
        Output mapping mode for component-to-cell-state interpretation.
    
    Returns
    -------
    None
        Initializes MetaTiME resources and annotation mode.
    
    Examples
    --------
    >>> TiME_object = ov.single.MetaTiME(adata, mode="table")
    """
    
    def __init__(self,adata:anndata.AnnData,mode:str='table'):
        """
        Initialize MetaTiME model resources and annotation table.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData object for MetaTiME annotation.
        mode : str
            Name-mapping mode in ``metatime.mecs.load_mecname``. Choose from
            ``'mecnamedict'``, ``'table'`` or ``'meciddict'``.

        """
        check_metatime()
        global metatime_install
        if metatime_install==True:
            global mecs
            global mecmapper
            global annotator
            global config
            from metatime import mecs
            from metatime import mecmapper
            from metatime import annotator
            from metatime import config

        self.adata=adata
        # Load the pre-trained MeCs
        print('...load pre-trained MeCs')
        self.mecmodel = mecs.MetatimeMecs.load_mec_precomputed()

        # Load functional annotation for MetaTiME-TME
        print('...load functional annotation for MetaTiME-TME')
        self.mectable = mecs.load_mecname(mecDIR = config.SCMECDIR, mode =mode )
        self.mecnamedict = mecs.getmecnamedict_ct(self.mectable) 
        
        
    
    def overcluster(self,resolution : float=8, 
                random_state: int= 0, 
                clustercol :str = 'overcluster'):
        """
        Perform high-resolution Leiden clustering for MetaTiME input.

        Parameters
        ----------
        resolution : float
            Resolution parameter for Leiden overclustering.
        random_state : int
            Random seed for Leiden clustering.
        clustercol : str
            Output cluster column name stored in ``adata.obs``.
        
        """

        print('...overclustering using leiden')
        from ..pp import leiden
        leiden(self.adata, resolution=resolution, key_added = clustercol, random_state=random_state,
               backend='python')
        self.clustercol=clustercol
        #self.adata = annotator.overcluster(self.adata,resolution,random_state,clustercol) # this generates a 'overcluster' columns in adata.obs
        
    def predictTiME(self,save_obs_name:str='MetaTiME'):
        """
        Predict tumor microenvironment cell states for each cell.

        Parameters
        ----------
        save_obs_name : str
            Prefix used for predicted annotation columns in ``adata.obs``.

        Returns
        -------
        anndata.AnnData
            AnnData with MetaTiME annotations stored in ``obs``.
        
        """
        print('...projecting MeC scores')
        self.pdata=mecmapper.projectMecAnn(self.adata, self.mecmodel.mec_score)
        projmat, mecscores = annotator.pdataToTable(self.pdata, self.mectable, gcol = self.clustercol)
        projmat, gpred, gpreddict = annotator.annotator(projmat,  self.mecnamedict, gcol = self.clustercol)
        self.adata = annotator.saveToAdata( self.adata, projmat)
        #self.pdata = annotator.saveToPdata( self.pdata, self.adata, projmat )
        self.adata.obs[save_obs_name] = self.adata.obs['{}_{}'.format(save_obs_name,self.clustercol)].str.split(': ').str.get(1)
        #self.pdata.obs[save_obs_name] = self.pdata.obs['{}_{}'.format(save_obs_name,self.clustercol)].str.split(': ').str.get(1)
        self.adata.obs['MetaTiME']=self.adata.obs['MetaTiME'].fillna('Unknown')
        self.adata.obs['Major_{}'.format(save_obs_name)]=[i.split('_')[0] for i in self.adata.obs[save_obs_name]]
        for i in self.adata.obs.columns:
            if 'MeC_' in i:
                del self.adata.obs[i]
        #self.pdata.obs['Major_{}'.format(save_obs_name)]=[i.split('_')[0] for i in self.pdata.obs[save_obs_name]]
        print('......The predicted celltype have been saved in obs.{}'.format(save_obs_name))
        print('......The predicted major celltype have been saved in obs.Major_{}'.format(save_obs_name))
        add_reference(self.adata,'MetaTiME','cell annotation with MetaTiME')
        return self.adata
    
    def plot(self,basis:str='X_umap',cluster_key:str='MetaTiME',fontsize:int=8, 
             min_cell:int=5, title=None,figsize:tuple=(6,6),
             dpi:int=80,frameon:bool=False,legend_loc=None,palette=None):
        """
        Plot MetaTiME annotations with optional label collision adjustment.

        Parameters
        ----------
        basis : str
            Embedding key in ``adata.obsm`` used for plotting.
        cluster_key : str
            ``obs`` column used for coloring/labeling.
        fontsize : int
            Text label font size.
        min_cell : int
            Minimum cells per group retained for visualization.
        title : str or None
            Figure title; defaults to ``cluster_key``.
        figsize : tuple
            Figure size in inches as ``(width, height)``.
        dpi : int
            Figure DPI.
        frameon : bool
            Whether to draw plot frame.
        legend_loc : str or None
            Matplotlib legend placement.
        palette : Any
            Color palette passed to ``ov.pl.embedding``.

        Returns
        -------
        tuple
            ``(fig, ax)`` Matplotlib figure and axis handles.
        
        """
        import matplotlib.pyplot as plt
        if not title:
            title = cluster_key

        if( min_cell >0 ):
            groupcounts = self.adata.obs.groupby( cluster_key ).count()
            groupcounts = groupcounts[groupcounts.columns[0]]
            group_with_good_counts = groupcounts[groupcounts>= min_cell ].index.tolist()
            self.adata = self.adata[ self.adata.obs[ cluster_key ].isin( group_with_good_counts ) ]

        if palette ==None:
            palette=plt.cycler("color",plt.cm.tab20(np.linspace(0,1,20)))

        with plt.rc_context({"figure.figsize": figsize, "figure.dpi": dpi, "figure.frameon": frameon}):
            from ..pl import embedding
            ax = embedding(self.adata, basis=basis,color= cluster_key , show=False, legend_loc=None, add_outline=False, 
                    #legend_loc='on data',legend_fontsize=6, legend_fontoutline=2,
                    title= title, 
                    palette=palette, 
                    #palette=plt.cycler("color",plt.cm.Set1(np.linspace(0,1,9))), 
                    frameon=frameon
                    )
            gen_mpl_labels(
                self.adata,
                cluster_key,
                exclude=("None",),  
                basis=basis,
                ax=ax,
                adjust_kwargs=dict(arrowprops=dict(arrowstyle='-', color='black')),
                text_kwargs=dict(fontsize= fontsize ,weight='bold'),
            )
            fig = ax.get_figure()
            fig.tight_layout()
            #plt.show()
            return( fig,ax )

        

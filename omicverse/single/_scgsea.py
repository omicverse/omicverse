import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata
import seaborn as sns
from .._settings import add_reference
from ._aucell import derive_auc_threshold,fast_rank,_rank_sparse_row,aucell
from .._registry import register_function


_OV_ES_AUCELL_NOTICE = (
    "[ov.single AUCell] The new GPU-accelerated API is `ov.es.aucell` — "
    "pass signatures as a dict, e.g. "
    "`ov.es.aucell(adata, signatures={'name': genes}, engine='gpu')`. "
    "The legacy function still works."
)


def geneset_aucell_tmp(adata, geneset_name, geneset, AUC_threshold=0.01, seed=42, chunk_size=10000):
    r"""Calculate the AUC-ell score for a given gene set.

    Parameters
    ----------
    adata:anndata.AnnData
        AnnData containing expression matrix.
    geneset_name:str
        Name of gene set; used as output column prefix.
    geneset:list
        Gene symbols composing the gene set.
    AUC_threshold:float
        AUCell rank threshold percentile.
    seed:int
        Random seed for ranking backend.
    chunk_size:int
        Number of cells processed per chunk.

    Returns
    -------
    None
        Writes AUCell score to ``adata.obs[f'{geneset_name}_aucell']``.
    """
    print(_OV_ES_AUCELL_NOTICE)
    from ..external.ctxcore.recovery import aucs

    matrix = adata.X.copy()
    percentiles = derive_auc_threshold(matrix, AUC_threshold)
    auc_threshold = percentiles[AUC_threshold]

    n_cells = matrix.shape[0]
    auc_results = np.zeros(n_cells, dtype=np.float64)

    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        chunk = matrix[start:end]
        np_rnk_sparse = fast_rank(chunk, seed=seed)
        rnk = pd.DataFrame(np_rnk_sparse[:, np.where(adata.var_names.isin(geneset))[0]])

        if rnk.empty or (float(len(np.where(adata.var_names.isin(geneset))[0])) / float(len(geneset))) < 0.80:
            print(
                f"Less than 80% of the genes in {geneset_name} are present in the "
                "expression matrix."
            )
        else:
            weights = np.array([1 for _ in geneset])
            auc_results[start:end] = aucs(rnk, np_rnk_sparse.shape[1], weights, auc_threshold)

    adata.obs[f'{geneset_name}_aucell'] = auc_results

def global_imports(modulename,shortname = None, asfunction = False):
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:        
        globals()[shortname] = __import__(modulename)

@register_function(
    aliases=['基因集 AUCell', 'geneset_aucell', 'gene set activity scoring'],
    category="single",
    description="Score per-cell activity of a custom gene set using AUCell ranking-based enrichment robust to library-size variation.",
    prerequisites={'optional_functions': ['pp.preprocess']},
    requires={'var': ['gene names'], 'layers': ['normalized expression (recommended)']},
    produces={'obs': ['aucell scores'], 'uns': ['aucell parameters']},
    auto_fix='none',
    examples=['ov.single.geneset_aucell(adata, geneset_name="IFN_response", geneset=ifn_genes)'],
    related=['single.pathway_aucell', 'single.pathway_aucell_enrichment']
)
def geneset_aucell(adata,geneset_name,geneset,AUC_threshold=0.01,seed=42):
    r"""Calculate the AUC-ell score for a given gene set.

    Parameters
    ----------
    adata:anndata.AnnData
        AnnData containing expression matrix.
    geneset_name:str
        Name of gene set; used as output column prefix.
    geneset:list
        Gene symbols composing the gene set.
    AUC_threshold:float
        AUCell rank threshold percentile.
    seed:int
        Random seed for ranking backend.

    Returns
    -------
    None
        Writes AUCell score to ``adata.obs[f'{geneset_name}_aucell']``.
    """
    print(_OV_ES_AUCELL_NOTICE)
    from ..external.ctxcore.recovery import aucs

    matrix = adata.X.copy()
    percentiles = derive_auc_threshold(matrix, AUC_threshold)
    auc_threshold = percentiles[AUC_threshold]

    np_rnk_sparse=fast_rank(matrix, seed= seed)

    rnk  = pd.DataFrame(np_rnk_sparse[:, np.where(adata.var_names.isin(geneset))[0]])

    if rnk.empty or (float(np_rnk_sparse.shape[1]) / float(len(geneset))) < 0.80:
        print(
            f"Less than 80% of the genes in {geneset_name} are present in the "
            "expression matrix."
        )
        adata.obs['{}_aucell'.format(geneset_name)]=np.zeros(shape=(rnk.shape[0]), dtype=np.float64)
    else:
        weights = np.array([1 for i in geneset])
        adata.obs['{}_aucell'.format(geneset_name)]=aucs(rnk,
        np_rnk_sparse.shape[1],weights,auc_threshold
        )
    
@register_function(
    aliases=['通路 AUCell', 'pathway_aucell', 'pathway activity scoring'],
    category="single",
    description="Compute single-cell pathway activity scores across multiple pathways using AUCell and write scores into cell metadata.",
    prerequisites={'optional_functions': ['pp.preprocess']},
    requires={'var': ['gene names']},
    produces={'obs': ['pathway auc scores'], 'uns': ['pathway auc settings']},
    auto_fix='none',
    examples=['ov.single.pathway_aucell(adata, pathway_names=list(pathways.keys()), pathways_dict=pathways)'],
    related=['single.geneset_aucell', 'single.pathway_aucell_enrichment']
)
def pathway_aucell(adata,pathway_names,pathways_dict,AUC_threshold=0.01,seed=42):
    r"""Calculate the area under the curve (AUC) for a set of pathways in an AnnData object.

    Parameters
    ----------
    adata:anndata.AnnData
        AnnData containing expression matrix.
    pathway_names:list
        Ordered pathway names to score.
    pathways_dict:dict
        Mapping from pathway name to list of genes.
    AUC_threshold:float
        AUCell rank threshold percentile.
    seed:int
        Random seed for ranking backend.

    Returns
    -------
    None
        Writes per-pathway AUCell scores to ``adata.obs``.
    """
    print(_OV_ES_AUCELL_NOTICE)
    from ..external.ctxcore.recovery import aucs

    matrix = adata.X.copy()
    percentiles = derive_auc_threshold(matrix, AUC_threshold)
    auc_threshold = percentiles[AUC_threshold]

    np_rnk_sparse=fast_rank(matrix, seed= seed)

    for pathway_name in pathway_names:
        pathway_genes=pathways_dict[pathway_name]

        rnk  = pd.DataFrame(np_rnk_sparse[:, np.where(adata.var_names.isin(pathway_genes))[0]]).copy()
        if rnk.empty or (float(np_rnk_sparse.shape[1]) / float(len(pathway_genes))) < 0.80:
            print(
                f"Less than 80% of the genes in {pathway_name} are present in the "
                "expression matrix."
            )
            adata.obs['{}_aucell'.format(pathway_name)]=np.zeros(shape=(rnk.shape[0]), dtype=np.float64)
        else:
            weights = np.array([1 for i in pathway_genes])
            adata.obs['{}_aucell'.format(pathway_name)]=aucs(rnk,
            np_rnk_sparse.shape[1],weights,auc_threshold
            )
    add_reference(adata,'AUCell','pathway activity score with AUCell')
            
def pathway_aucell_tmp(adata, pathway_names, pathways_dict, AUC_threshold=0.01, seed=42, chunk_size=10000):
    r"""Calculate the area under the curve (AUC) for a set of pathways in an AnnData object.

    Parameters
    ----------
    adata:anndata.AnnData
        AnnData containing expression matrix.
    pathway_names:list
        Ordered pathway names to score.
    pathways_dict:dict
        Mapping from pathway name to list of genes.
    AUC_threshold:float
        AUCell rank threshold percentile.
    seed:int
        Random seed for ranking backend.
    chunk_size:int
        Number of cells processed per chunk.

    Returns
    -------
    None
        Writes per-pathway AUCell scores to ``adata.obs``.
    """
    print(_OV_ES_AUCELL_NOTICE)
    from ..external.ctxcore.recovery import aucs

    matrix = adata.X.copy()
    percentiles = derive_auc_threshold(matrix, AUC_threshold)
    auc_threshold = percentiles[AUC_threshold]
    
    n_cells = matrix.shape[0]
    
    for pathway_name in pathway_names:
        pathway_genes = pathways_dict[pathway_name]
        auc_results = np.zeros(n_cells, dtype=np.float64)
        
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            chunk = matrix[start:end]
            np_rnk_sparse = fast_rank(chunk, seed=seed)
            
            rnk = pd.DataFrame(np_rnk_sparse[:, np.where(adata.var_names.isin(pathway_genes))[0]])
            if rnk.empty or (float(len(np.where(adata.var_names.isin(pathway_genes))[0])) / float(len(pathway_genes))) < 0.80:
                print(
                    f"Less than 80% of the genes in {pathway_name} are present in the "
                    "expression matrix."
                )
            else:
                weights = np.array([1 for _ in pathway_genes])
                auc_results[start:end] = aucs(rnk, np_rnk_sparse.shape[1], weights, auc_threshold)

        adata.obs[f'{pathway_name}_aucell'] = auc_results
            
@register_function(
    aliases=['通路活性富集', 'pathway_aucell_enrichment', 'pathway overactivity screen'],
    category="single",
    description="Identify pathways with robust AUCell activity signals by integrating score distribution and gene-overlap constraints.",
    prerequisites={'functions': ['pathway_aucell']},
    requires={'obs': ['pathway auc scores']},
    produces={'uns': ['pathway_aucell_enrichment']},
    auto_fix='none',
    examples=['ov.single.pathway_aucell_enrichment(adata, pathways_dict=pathways, AUC_threshold=0.01)'],
    related=['single.pathway_aucell', 'single.pathway_enrichment']
)
def pathway_aucell_enrichment(adata,pathways_dict,AUC_threshold=0.01,seed=42,num_workers=1,gene_overlap_threshold=0.80):
    r"""Enrich cell annotations with pathway activity scores using the AUC-ell method.

    Parameters
    ----------
    adata:anndata.AnnData
        AnnData containing expression matrix.
    pathways_dict:dict
        Mapping from pathway name to list of genes.
    AUC_threshold:float
        AUCell rank threshold percentile.
    seed:int
        Random seed for AUCell backend.
    num_workers:int
        Number of workers used in AUCell computation.
    gene_overlap_threshold:float
        Minimum fraction of pathway genes present in ``adata.var_names``.

    Returns
    -------
    anndata.AnnData
        AnnData whose ``X`` stores pathway AUCell activity matrix.
    """
    print(_OV_ES_AUCELL_NOTICE)
    from ..external.ctxcore.genesig import GeneSignature

    test_gmt=[]
    for i in pathways_dict.keys():
        test_gmt.append(GeneSignature(name=i,gene2weight=dict(zip(pathways_dict[i],[1 for i in pathways_dict[i]]))))

    # Use sparse matrix for both derive_auc_threshold and aucell
    matrix_sparse = adata.X.copy()
    percentiles = derive_auc_threshold(matrix_sparse, AUC_threshold)
    auc_threshold = percentiles[AUC_threshold]

    # Pass sparse matrix directly to aucell with index and columns
    aucs_mtx = aucell(matrix_sparse, signatures=test_gmt, auc_threshold=auc_threshold,
                     num_workers=num_workers, seed=seed,
                     index=adata.obs_names, columns=adata.var_names,
                     gene_overlap_threshold=gene_overlap_threshold)
    
    adata_aucs=anndata.AnnData(aucs_mtx)
    add_reference(adata,'AUCell','pathway activity score with AUCell')
    return adata_aucs

def pathway_aucell_enrichment_tmp(adata, pathways_dict, AUC_threshold=0.01, seed=42, 
                              num_workers=1, chunk_size=10000):
    r"""Enrich cell annotations with pathway activity scores using the AUC-ell method.

    Parameters
    ----------
    adata:anndata.AnnData
        AnnData containing expression matrix.
    pathways_dict:dict
        Mapping from pathway name to list of genes.
    AUC_threshold:float
        AUCell rank threshold percentile.
    seed:int
        Random seed for AUCell backend.
    num_workers:int
        Number of workers used in AUCell computation.
    chunk_size:int
        Number of cells processed per chunk.

    Returns
    -------
    anndata.AnnData
        AnnData whose ``X`` stores pathway AUCell activity matrix.
    """
    print(_OV_ES_AUCELL_NOTICE)
    from tqdm import tqdm
    from ..external.ctxcore.genesig import GeneSignature

    test_gmt = []
    for i in pathways_dict.keys():
        test_gmt.append(GeneSignature(name=i, gene2weight=dict(zip(pathways_dict[i], [1 for _ in pathways_dict[i]]))))

    # Use sparse matrix for derive_auc_threshold
    matrix_sparse = adata.X.copy()
    percentiles = derive_auc_threshold(matrix_sparse, AUC_threshold)
    auc_threshold = percentiles[AUC_threshold]

    # Process in chunks using sparse matrix slicing
    aucs_mtx_list = []
    n_cells = matrix_sparse.shape[0]
    
    for start in tqdm(range(0, n_cells, chunk_size)):
        end = min(start + chunk_size, n_cells)
        # Slice sparse matrix directly
        chunk_sparse = matrix_sparse[start:end]
        chunk_index = adata.obs_names[start:end]
        
        aucs_chunk = aucell(chunk_sparse, signatures=test_gmt, auc_threshold=auc_threshold, 
                           num_workers=num_workers, seed=seed,
                           index=chunk_index, columns=adata.var_names)
        aucs_mtx_list.append(aucs_chunk)

    # Concatenate the results
    aucs_mtx = pd.concat(aucs_mtx_list)

    adata_aucs = anndata.AnnData(aucs_mtx)
    return adata_aucs


@register_function(
    aliases=['差异通路富集', 'pathway_enrichment', 'deg pathway enrichment'],
    category="single",
    description="Run pathway enrichment on cluster/group differential genes to connect transcriptional changes with biological processes.",
    prerequisites={'optional_functions': ['pp.neighbors', 'pp.leiden']},
    requires={'obs': ['group labels'], 'var': ['gene names']},
    produces={'uns': ['pathway_enrichment_results']},
    auto_fix='escalate',
    examples=['ov.single.pathway_enrichment(adata, pathways_dict=pathways, organism="Human", group_by="leiden")'],
    related=['single.pathway_enrichment_plot', 'single.pathway_aucell_enrichment']
)
def pathway_enrichment(adata, pathways_dict,organism='Human',group_by='louvain', 
                       cutoff=0.05, logfc_threshold=2,pvalue_type='adjust',plot=True):
    r"""Perform pathway enrichment analysis on gene expression data.
    
    Parameters
    ----------
    adata:anndata.AnnData
        AnnData with ``rank_genes_groups`` results and cluster labels.
    pathways_dict:dict
        Mapping from pathway/set name to member genes.
    organism:str
        Organism name used by enrichment backend.
    group_by:str
        ``adata.obs`` key used as grouping variable.
    cutoff:float
        Significance cutoff for pathway filtering.
    logfc_threshold:float
        Minimum log2 fold-change threshold for DE genes.
    pvalue_type:str
        P-value type used for final filtering: ``'adjust'`` or ``'raw'``.
    plot:bool
        Whether to draw per-cluster barplots for enriched pathways.
    
    Returns
    -------
    pd.DataFrame
        Enrichment result table with cluster labels and pathway statistics.
    
    Examples:
        >>> # Perform pathway enrichment analysis on adata using pathways_dict
        >>> res = pathway_enrichment(adata, pathways_dict)
    
    Reference:
        The code for pathway_enrichment() function was adapted from scanpy workflows: 
        https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html#Gene-set-enrichment-analysis.
    """

    from ..bulk._ora import enrichr
    df_list = []
    cluster_list = []
    celltypes = sorted(adata.obs[group_by].unique())

    for celltype in celltypes:
        degs = sc.get.rank_genes_groups_df(adata, group=celltype, key='rank_genes_groups', log2fc_min=logfc_threshold, 
                                    pval_cutoff=cutoff)['names'].squeeze()
        if isinstance(degs, str):
            degs = [degs.strip()]
        else:
            degs = degs.str.strip().tolist()
        
        if not degs:
            continue
        if (organism == 'Mouse') or (organism == 'mouse') or (organism == 'mm'):
            background='mmusculus_gene_ensembl'
        elif (organism == 'Human') or (organism == 'human') or (organism == 'hs'):
            background='hsapiens_gene_ensembl'
        else:
            background=adata.var.index.tolist()
        enr = enrichr(gene_list=degs,
                description='',
                gene_sets=pathways_dict,
                organism=organism,
                background=background,
                cutoff=0.5,
                )
        if (enr is not None) and hasattr(enr, 'res2d') and (enr.res2d.shape[0] > 0):
            df_list.append(enr.res2d)
            cluster_list.append(celltype)

    columns = ['Cluster', 'Gene_set', 'Term', 'Overlap', 'Odds Ratio','P-value', 'Adjusted P-value', 'Genes']

    df = pd.DataFrame(columns = columns)
    for cluster_ind, df_ in zip(cluster_list, df_list):
        df_ = df_[df_['Adjusted P-value'] <= cutoff]
        df_ = df_.assign(Cluster = cluster_ind)
        if (df_.shape[0] > 0):
            df = pd.concat([df, df_[columns]], sort=False)
            df_tmp = df_.loc[:, ['Term', 'Adjusted P-value']][:min(10, df_.shape[0])]
            df_tmp['Term'] = [x.split('(',1)[0] for x in df_tmp['Term']]
            df_tmp['-log_adj_p'] = - np.log10(df_tmp['Adjusted P-value'])
            df_tmp = df_tmp.sort_values(by='-log_adj_p', ascending=True)
            if plot==True:
                ax = df_tmp.plot.barh(y='-log_adj_p', x='Term', legend=False, grid=False, figsize=(12,4))
                ax.set_title('Cluster {}'.format(cluster_ind))
                ax.set_ylabel('')
                ax.set_xlabel('-log(Adjusted P-value)')
            #pp.savefig(ax.figure, bbox_inches='tight')
            #plt.close()
        else:
            print('No pathway with an adjusted P-value less than the cutoff (={}) for cluster {}'.format(cutoff, cluster_ind))
    
    if pvalue_type=='adjust':
        enrich_res=df[df['Adjusted P-value']<cutoff]
        enrich_res['logp']=-np.log10(enrich_res['Adjusted P-value'])
    else:
        enrich_res=df[df['P-value']<cutoff]
        enrich_res['logp']=-np.log10(enrich_res['P-value'])
    enrich_res['logc']=np.log(enrich_res['Odds Ratio'])
    enrich_res['num']=[int(i.split('/')[0]) for i in enrich_res['Overlap']]
    enrich_res['fraction']=[int(i.split('/')[0])/int(i.split('/')[1]) for i in enrich_res['Overlap']]
    add_reference(adata,'Enrichr','pathway over-representation analysis (hypergeometric test)')
    
    return enrich_res

@register_function(
    aliases=['通路富集绘图', 'pathway_enrichment_plot', 'enrichment barplot'],
    category="single",
    description="Visualize top enriched pathways with effect size and significance to facilitate biological interpretation of DEG-derived signatures.",
    prerequisites={'functions': ['pathway_enrichment']},
    requires={'uns': ['pathway_enrichment_results']},
    produces={},
    auto_fix='none',
    examples=['ov.single.pathway_enrichment_plot(enrich_res, term_num=20, figsize=(6,8))'],
    related=['single.pathway_enrichment', 'bulk.geneset_plot']
)
def pathway_enrichment_plot(enrich_res,term_num=5,return_table=False,figsize=(3,10),plot_title='',**kwds):
    r"""Visualize the pathway enrichment analysis results as a heatmap.

    Parameters
    ----------
    enrich_res:pd.DataFrame
        DataFrame returned by ``pathway_enrichment``.
    term_num:int
        Number of top terms kept per cluster.
    return_table:bool
        Whether to return pivot/heatmap table instead of plotting.
    figsize:tuple
        Figure size for heatmap.
    plot_title:str
        Heatmap title.
    **kwds
        Additional keyword arguments forwarded to ``sns.heatmap``.

    Returns
    -------
    matplotlib.axes.Axes or pd.DataFrame
        Heatmap axes, or table when ``return_table=True``.

    Examples:
        >>> res = pathway_enrichment(adata, pathways_dict)
        >>> pathway_enrichment_plot(res, term_num=10, return_table=True, figsize=(6,12), cmap='Blues')
    """

    celltypes=enrich_res['Cluster'].unique()
    plot_heatmap=pd.DataFrame(index=celltypes)
    for celltype in celltypes:
        res_test=enrich_res.loc[enrich_res['Cluster']==celltype].iloc[:term_num]
        plot_heatmap_test=pd.DataFrame(res_test[['Term','logp']])
        plot_heatmap_test=plot_heatmap_test.set_index(plot_heatmap_test['Term'])
        del plot_heatmap_test['Term']
        #plot_heatmap[plot_heatmap_test.index]=0
        for i in plot_heatmap_test.index:
            if len(enrich_res.loc[(enrich_res['Cluster']==celltype)&(enrich_res['Term']==i),'logp'].values)!=0:
                plot_heatmap.loc[celltype,i]=enrich_res.loc[(enrich_res['Cluster']==celltype)&(enrich_res['Term']==i),'logp'].values[0]
            else:
                plot_heatmap.loc[celltype,i]=0
    plot_heatmap.fillna(0,inplace=True)
    if return_table==True:
        return plot_heatmap
    
    fig,ax=plt.subplots(figsize=figsize)
    axr=sns.heatmap(plot_heatmap.T,ax=ax,**kwds)
    axr.tick_params(right=True, top=False,left=False, labelright=True, labelleft=False,labeltop=False,rotation=0)
    ax.set_title(plot_title,fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize = 10)

    
    return ax



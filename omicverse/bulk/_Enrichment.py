import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import networkx as nx
import seaborn as sns
import pandas as pd
from sklearn import decomposition as skldec
from typing import Optional, Any, Dict, List, Tuple

from ..utils import plot_text_set
from .._registry import register_function
import matplotlib


def _resolve_genesets(pathways, organism: str = 'Human') -> dict:
    """Normalise a gene-set argument to a ``{pathway: [genes]}`` dict.

    Accepts a ready dict, a path to a ``.gmt`` / ``.txt`` file, or an
    Enrichr library name (e.g. ``'MSigDB_Hallmark_2020'`` /
    ``'KEGG_2021_Human'``). Paths and library names are resolved with
    :func:`omicverse.utils.geneset_prepare`, which auto-downloads and
    caches the library on first use; a dict is returned unchanged.
    """
    if isinstance(pathways, dict):
        return pathways
    if isinstance(pathways, str):
        from ..utils import geneset_prepare
        return geneset_prepare(pathways, organism=organism)
    raise TypeError(
        "pathways_dict must be a dict, or a str (a .gmt/.txt path or an "
        f"Enrichr library name); got {type(pathways).__name__}"
    )


@register_function(
    aliases=["基因集富集", "geneset_enrichment", "enrichr_analysis", "pathway_enrichment", "富集分析"],
    category="bulk",
    description="Over-representation (ORA) gene-set enrichment for a discrete gene list. pathways_dict accepts a prepared dict, a .gmt/.txt path, or an Enrichr library name (resolved + auto-downloaded internally).",
    prerequisites={
        'optional_functions': ['geneset_prepare', 'download_pathway_database']
    },
    examples=[
        "# pathways_dict accepts an Enrichr library name directly —",
        "# it is auto-downloaded and cached on first use.",
        "enr = ov.bulk.geneset_enrichment(",
        "    gene_list=deg_genes,",
        "    pathways_dict='MSigDB_Hallmark_2020',",
        "    pvalue_type='auto',",
        "    organism='Human',",
        ")",
        "",
        "# A prepared dict or a local .gmt/.txt path also work:",
        "# pathways_dict=ov.utils.geneset_prepare('genesets/h.all.symbols.gmt')",
        "# pathways_dict='genesets/GO_Biological_Process_2021.txt'"
    ],
    related=["utils.geneset_prepare", "utils.download_pathway_database", "bulk.geneset_plot", "bulk.pyGSEA"]
)
def geneset_enrichment(gene_list:list,pathways_dict:dict,
                       pvalue_threshold:float=0.05,pvalue_type:str='auto',
                       organism:str='Human',description:str='None',
                       background:list=None,
                       outdir:str='./enrichr',cutoff:float=0.5)->pd.DataFrame:
    r"""Perform pathway enrichment analysis using Enrichr-compatible gene-set libraries.

    Parameters
    ----------
    gene_list:list
        Input gene symbols (typically DEGs) for enrichment testing.
    pathways_dict:dict|str
        Gene sets — a prepared dict (``ov.utils.geneset_prepare``), a
        ``.gmt``/``.txt`` path, or an Enrichr library name (resolved and
        auto-downloaded internally).
    pvalue_threshold:float, optional
        Significance threshold used to filter enrichment terms.
    pvalue_type:str, optional
        P-value mode: ``auto``/``adjust``/raw ``P-value`` filtering.
    organism:str, optional
        Organism label passed to Enrichr backend (for example ``Human``/``Mouse``).
    description:str, optional
        Job description tag stored in output metadata.
    background:list|None, optional
        Optional background gene universe. If ``None``, species defaults are used.
    outdir:str, optional
        Directory for enrichment output files.
    cutoff:float, optional
        Enrichr internal cutoff threshold.

    Returns
    -------
    pandas.DataFrame
        Enrichment result table with statistics and derived plotting columns.
    """
    pathways_dict = _resolve_genesets(pathways_dict, organism)
    from ._ora import enrichr
    # omicverse's own single-process hypergeometric ORA (``_ora``). A
    # ``background=None`` resolves to the union of genes in ``pathways_dict``
    # (matches upstream gseapy's ``parse_background``), avoiding the brittle
    # Ensembl BioMart MySQL query the previous default triggered.
    enr = enrichr(gene_list=gene_list,
                 gene_sets=pathways_dict,
                 organism=organism, # don't forget to set organism to the one you desired! e.g. Yeast
                 description=description,
                 background=background,
                 outdir=outdir,
                 cutoff=cutoff # test dataset, use lower value from range(0,1)
                )
    if pvalue_type=='auto':
        if enr.res2d.shape[0]>100:
            enrich_res=enr.res2d[enr.res2d['Adjusted P-value']<pvalue_threshold]
            enrich_res['logp']=-np.log10(enrich_res['Adjusted P-value'])
        else:
            enrich_res=enr.res2d[enr.res2d['P-value']<pvalue_threshold]
            enrich_res['logp']=-np.log10(enrich_res['P-value'])
    elif pvalue_type=='adjust':
        enrich_res=enr.res2d[enr.res2d['Adjusted P-value']<pvalue_threshold]
        enrich_res['logp']=-np.log10(enrich_res['Adjusted P-value'])
    else:
        enrich_res=enr.res2d[enr.res2d['P-value']<pvalue_threshold]
        enrich_res['logp']=-np.log10(enrich_res['P-value'])
    enrich_res['logc']=np.log(enrich_res['Odds Ratio'])
    enrich_res['num']=[int(i.split('/')[0]) for i in enrich_res['Overlap']]
    enrich_res['fraction']=[int(i.split('/')[0])/int(i.split('/')[1]) for i in enrich_res['Overlap']]
    return enrich_res

def enrichment_multi_concat(enr_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    def process_df(df: pd.DataFrame, term_col_name: str) -> pd.DataFrame:
        new_data = []
        for _, row in df.iterrows():
            genes = row['Genes'].split(';')
            for gene in genes:
                new_data.append({
                    'Gene': gene,
                    term_col_name: row['Term'],

                })
        return pd.DataFrame(new_data)
    new_dict={}
    new_li=[]
    for key in enr_dict.keys():
        new_dict[key]=process_df(enr_dict[key], key)
        new_li.append(new_dict[key])
    # 合并两个DataFrame
    merged_df = pd.concat(new_li, ignore_index=True).fillna('')
    #return merged_df

    #print(dict(zip(enr_dict.keys(),
    #        [lambda x: '|'.join(x.dropna().unique()) for i in range(len(enr_dict.keys()))])))
    
    # 按基因分组，并将相同基因的Term合并
    result_df = merged_df.groupby('Gene').agg(dict(zip(enr_dict.keys(),
            [lambda x: '|'.join(x.dropna().unique()) for i in range(len(enr_dict.keys()))]))
                                             ).reset_index()
    return result_df



def geneset_enrichment_GSEA(gene_rnk:pd.DataFrame,pathways_dict:dict,
                            processes:int=1,
                     permutation_num:int=1000,
                     outdir:str='./enrichr_gsea', format:str='png', seed:int=112,
                     organism:str='Human', backend:str='numpy',
                     weight:float=1.0, min_size:int=15, max_size:int=500,
                     progress:bool=True)->dict:
    r"""Pre-ranked GSEA on a ranked gene list.

    Parameters
    ----------
        gene_rnk: Pre-ranked correlation table / pandas DataFrame (the GSEA ``.rnk`` input).
        pathways_dict: Gene sets — a dict, a ``.gmt``/``.txt`` path, or an Enrichr library name.
        processes: Deprecated/ignored (the NumPy backend is single-process).
        permutation_num: Permutations for the significance null. (1000)
        outdir: Output directory (kept for backward compatibility).
        format: Matplotlib figure format.
        seed: Random seed.
        backend: ``'numpy'`` (default and only option) — fast single-process
            pure-NumPy GSEA (no multiprocessing dead-locks). The legacy
            ``'gseapy'`` backend has been removed; any other value warns and
            falls back to NumPy.
        weight: Enrichment-score weighting exponent (1.0 = classic weighted GSEA).
        min_size, max_size: Keep gene sets whose matched size is in this range.

    Returns
    -------
        pre_res: A prerank result object (``.ranking`` / ``.res2d`` / ``.results``).

    Notes
    -----
    The default ``backend='numpy'`` replaces ``gseapy.prerank``'s
    ``processes=8`` joblib/loky pool, which can dead-lock inside long-lived /
    multi-threaded kernels (notebooks, agents, macOS ``spawn``). The NumPy path
    is single-process, deterministic, and ~5–20× faster; its enrichment scores
    match gseapy exactly (validated, ES Pearson r = 1.0).
    """
    pathways_dict = _resolve_genesets(pathways_dict, organism)
    if backend != 'numpy':
        import warnings
        warnings.warn(
            "The 'gseapy' GSEA backend has been removed (it relied on a "
            "joblib/loky process pool that could dead-lock); using the "
            "single-process NumPy backend instead.", stacklevel=2)
    from ._gsea_numpy import prerank as _prerank_numpy
    return _prerank_numpy(gene_rnk, pathways_dict,
                          permutation_num=permutation_num, weight=weight,
                          min_size=min_size, max_size=max_size, seed=seed,
                          progress=progress)


@register_function(
    aliases=['多组富集可视化', 'geneset_plot_multi', 'multi geneset plot'],
    category="bulk",
    description="Visualize multiple gene-set enrichment result tables side-by-side to compare pathway activity patterns across conditions.",
    prerequisites={'functions': ['geneset_enrichment']},
    requires={'uns': ['enrichment results']},
    produces={},
    auto_fix='none',
    examples=['ov.bulk.geneset_plot_multi(enr_dict=enrich_dict, colors_dict=color_dict, num=10)'],
    related=['bulk.geneset_enrichment', 'bulk.geneset_plot']
)
def geneset_plot_multi(enr_dict: Dict[str, pd.DataFrame], colors_dict: Dict[str, str], num: int = 5, fontsize: int = 10,
                        fig_title: str = '', fig_xlabel: str = 'Fractions of genes',
                        figsize: tuple = (2, 4), cmap: str = 'YlGnBu',
                        text_knock: int = 5, text_maxsize: int = 20, ax: Optional[matplotlib.axes._axes.Axes] = None
                        ) -> matplotlib.axes._axes.Axes:
    r"""Plot multiple enrichment result tables in a unified dot-clustermap panel.

    Parameters
    ----------
    enr_dict:dict[str,pandas.DataFrame]
        Mapping from group/condition name to enrichment result DataFrame.
    colors_dict:dict[str,str]
        Color mapping for each group in ``enr_dict``.
    num:int, optional
        Number of top terms taken from each group.
    fontsize:int, optional
        Base font size for labels and legends.
    fig_title:str, optional
        Figure title.
    fig_xlabel:str, optional
        X-axis label.
    figsize:tuple, optional
        Figure size.
    cmap:str, optional
        Colormap used for enrichment significance values.
    text_knock:int, optional
        Trim length applied to long term names.
    text_maxsize:int, optional
        Maximum wrapped text size for term labels.
    ax:matplotlib.axes.Axes|None, optional
        Existing axis; if ``None`` a new figure/axis is created.

    Returns
    -------
    marsilea.SizedHeatmap
        The rendered Marsilea dot-heatmap board (call ``.save(path)`` or access
        ``.figure`` to export). Rows are pathway terms; dot size = gene count,
        dot colour = -log10 adjusted-p; rows are split/coloured by group.
    """
    # Rendered with Marsilea (PyComplexHeatmap was dropped here): a sized
    # dot-heatmap where each row is a pathway term, dot **size** encodes the
    # gene count (``num``) and dot **colour** the significance
    # (``logp`` = -log10 adjusted-p). Rows are split & colour-coded by their
    # source group (``Type``); a side bar shows the gene ``fraction``.
    import marsilea as ma
    import marsilea.plotter as mp

    for key in enr_dict.keys():
        enr_dict[key]['Type'] = key
    enr_all = pd.concat([enr_dict[i].iloc[:num] for i in enr_dict.keys()], axis=0)
    enr_all['Term'] = [plot_text_set(i.split('(')[0], text_knock=text_knock,
                                     text_maxsize=text_maxsize)
                       for i in enr_all.Term.tolist()]
    enr_all.index = enr_all.Term
    # some GO terms exist in multiple categories (BP/CC/MF) — keep the first
    enr_all = enr_all.loc[~enr_all.index.duplicated(keep='first')]
    enr_all['Term1'] = list(enr_all.index)
    del enr_all['Term']

    # group order follows enr_dict insertion order
    type_order = [k for k in enr_dict.keys() if k in set(enr_all['Type'])]
    enr_all['Type'] = pd.Categorical(enr_all['Type'], categories=type_order,
                                     ordered=True)
    enr_all = enr_all.sort_values('Type')

    size_m = enr_all['num'].to_numpy(dtype=float).reshape(-1, 1)
    color_m = enr_all['logp'].to_numpy(dtype=float).reshape(-1, 1)
    types = enr_all['Type'].astype(str).to_numpy()
    terms = enr_all['Term1'].tolist()
    fractions = enr_all['fraction'].to_numpy(dtype=float)

    height = max(figsize[1], 0.32 * len(enr_all))
    h = ma.SizedHeatmap(
        size=size_m, color=color_m, cmap=cmap,
        vmin=-1 * np.log10(0.1), vmax=-1 * np.log10(1e-10),
        sizes=(20, 200), width=max(figsize[0] * 0.4, 0.6), height=height,
        color_legend_kws=dict(title=r'$-Log_{10}(P_{adjusted})$'),
        size_legend_kws=dict(title='Gene number'),
    )
    # category colour strip + split rows by group
    h.add_left(mp.Colors(types, palette=colors_dict, label='Category'),
               size=0.2, pad=0.05)
    h.group_rows(types, order=type_order, spacing=0.015)
    # gene fraction as a side bar, then the term names
    h.add_right(mp.Numbers(fractions, color='#c2c2c2', label=fig_xlabel,
                           show_value=False), size=0.8, pad=0.05)
    h.add_right(mp.Labels(terms, fontsize=fontsize), pad=0.05)
    if fig_title:
        h.add_title(top=fig_title, pad=0.1)
    h.add_legends(side='right', pad=0.1)
    h.render(figure=ax.figure if ax is not None else None)
    return h

@register_function(
    aliases=["富集分析可视化", "geneset_plot", "enrichment_plot", "通路富集图", "pathway_plot"],
    category="bulk",
    description="Visualize gene set enrichment analysis results with bubble plot",
    examples=[
        "# Basic usage",
        "ov.bulk.geneset_plot(enrich_res, num=10)",
        "# Custom appearance",
        "ov.bulk.geneset_plot(enrich_res, num=15, figsize=(3,5), cmap='RdBu')",
        "# Adjust node sizes and colors",
        "ov.bulk.geneset_plot(enrich_res, node_size=[10,20,30], fig_title='KEGG Pathways')"
    ],
    related=["bulk.geneset_enrichment", "bulk.geneset_plot_multi", "pl.volcano", "pl.dotplot"]
)
def geneset_plot(enrich_res: pd.DataFrame, num: int = 10, node_size: list = [5, 10, 15],
                        cax_loc: list = [2, 0.55, 0.5, 0.02], cax_fontsize: int = 12,
                        fig_title: str = '', fig_xlabel: str = 'Fractions of genes',
                        figsize: tuple = (2, 4), cmap: str = 'YlGnBu',
                        text_knock: int = 5, text_maxsize: int = 20,
                        bbox_to_anchor_used: tuple = (-0.45, -13), node_diameter: int = 10,
                        custom_ticks: list = [5, 10], ax: Optional[matplotlib.axes._axes.Axes] = None) -> matplotlib.axes._axes.Axes:
    r"""Plot enrichment results as a bubble plot.

    Parameters
    ----------
    enrich_res:pandas.DataFrame
        Enrichment result table from ``geneset_enrichment``/``pyGSEA.enrichment``.
    num:int, optional
        Number of top enriched terms to display.
    node_size:list[int], optional
        Bubble-size legend entries.
    cax_loc:list[float], optional
        Colorbar axis rectangle ``[left,bottom,width,height]``.
    cax_fontsize:int, optional
        Font size for colorbar and legend text.
    fig_title:str, optional
        Plot title.
    fig_xlabel:str, optional
        X-axis label.
    figsize:tuple, optional
        Figure size.
    cmap:str, optional
        Colormap for enrichment significance.
    text_knock:int, optional
        Truncation parameter for long term names.
    text_maxsize:int, optional
        Maximum text length/size control for term labels.
    bbox_to_anchor_used:tuple, optional
        Legend anchor coordinates.
    node_diameter:int, optional
        Bubble size scale factor.
    custom_ticks:list[int], optional
        Custom colorbar ticks.
    ax:matplotlib.axes.Axes|None, optional
        Existing axis to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing enrichment bubble plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    plot_data2=enrich_res.sort_values('P-value')[:num].sort_values('logc')
    st=ax.scatter(plot_data2['fraction'],range(len(plot_data2['logc'])),
            s=plot_data2['num']*node_diameter,linewidths=1,edgecolors='black',c=plot_data2['logp'],cmap=cmap)
    ax.yaxis.tick_right()
    plt.yticks(range(len(plot_data2['fraction'])),[plot_text_set(i.split('(')[0],text_knock=text_knock,text_maxsize=text_maxsize) for i in plot_data2['Term']],
            fontsize=10,)
    plt.xticks(fontsize=12,)
    plt.title(fig_title,fontsize=12)
    plt.xlabel(fig_xlabel,fontsize=12)

    fig = plt.gcf()
    cax = fig.add_axes(cax_loc)
    cb=fig.colorbar(st,shrink=0.25,cax=cax,orientation='horizontal')
    cb.set_label(r'$−Log_{10}(P_{adjusted})$',fontdict={'size':cax_fontsize})
    # new code to add custom ticks
    cb.set_ticks(custom_ticks)

    gl_li=[]
    for i in node_size:
        gl_li.append(ax.scatter([],[], s=i*node_diameter, marker='o', color='white',edgecolors='black'))

    plt.legend(gl_li,
        [str(i) for i in node_size],
        loc='lower left',
        ncol=3,bbox_to_anchor=bbox_to_anchor_used,
        fontsize=cax_fontsize)
    return ax

class pyGSE(object):

    def __init__(self,gene_list:list,pathways_dict:dict,pvalue_threshold:float=0.05,pvalue_type:str='auto',
                 background=None,organism:str='Human',description:str='None',outdir:str='./enrichr',cutoff:float=0.5) -> None:
        """Initialize the pyGSE class.

        Parameters
        ----------
        gene_list:list
            Input gene symbols for enrichment testing.
        pathways_dict:dict
            Prepared pathway dictionary from ``ov.utils.geneset_prepare``.
        pvalue_threshold:float, optional
            Significance threshold used in result filtering.
        pvalue_type:str, optional
            P-value filtering strategy (``auto``/``adjust``/raw p-value).
        background:list|None, optional
            Optional background gene set.
        organism:str, optional
            Organism label for Enrichr backend.
        description:str, optional
            Job description string.
        outdir:str, optional
            Output directory for enrichment artifacts.
        cutoff:float, optional
            Enrichr internal cutoff threshold.
        """

        self.gene_list=gene_list
        self.pathways_dict=_resolve_genesets(pathways_dict, organism)
        self.pvalue_threshold=pvalue_threshold
        self.pvalue_type=pvalue_type
        self.organism=organism
        self.description=description
        self.outdir=outdir
        self.cutoff=cutoff
        if background is None:
            if (organism == 'Mouse') or (organism == 'mouse') or (organism == 'mm'):
                background='mmusculus_gene_ensembl'
            elif (organism == 'Human') or (organism == 'human') or (organism == 'hs'):
                background='hsapiens_gene_ensembl'
            self.background=background
        else:
            self.background=background
    
    def enrichment(self):
        """gene set enrichment analysis.
        
        Returns
        -------
            A pandas.DataFrame object containing the enrichment results.
        """

        enrich_res=geneset_enrichment(self.gene_list,self.pathways_dict,self.pvalue_threshold,self.pvalue_type,
                                  self.organism,self.description,self.background,self.outdir,self.cutoff)
        self.enrich_res=enrich_res
        return enrich_res
    
    
    def plot_enrichment(self,num:int=10,node_size:list=[5,10,15],
                        cax_loc:list=[2,0.55,0.5,0.02],cax_fontsize:int=12,
                        fig_title:str='',fig_xlabel:str='Fractions of genes',
                        figsize:tuple=(2,4),cmap:str='YlGnBu',text_knock:int=2,text_maxsize:int=20,**kwargs)->matplotlib.axes._axes.Axes:

        """Plot the gene set enrichment result.
        
        Parameters
        ----------
            num: The number of enriched terms to plot. Default is 10.
            node_size: A list of integers defining the size of nodes in the plot. Default is [5,10,15].
            cax_loc: The location of the colorbar on the plot. Default is 2.
            cax_fontsize: The fontsize of the colorbar label. Default is 12.
            fig_title: The title of the plot. Default is an empty string.
            fig_xlabel: The label of the x-axis. Default is 'Fractions of genes'.
            figsize: The size of the plot. Default is (2,4).
            cmap: The colormap to use for the plot. Default is 'YlGnBu'.

        Returns
        -------
            A matplotlib.axes.Axes object.
        """
        return geneset_plot(self.enrich_res,num=num,node_size=node_size,
                            cax_loc=cax_loc,cax_fontsize=cax_fontsize,
                            fig_title=fig_title,fig_xlabel=fig_xlabel,
                            figsize=figsize,cmap=cmap,text_knock=text_knock,
                            text_maxsize=text_maxsize,**kwargs)
    
@register_function(
    aliases=["GSEA分析", "pyGSEA", "gene_set_enrichment", "基因集富集分析"],
    category="bulk",
    description="Pre-ranked Gene Set Enrichment Analysis (GSEA) for a ranked gene list. pathways_dict accepts a prepared dict, a .gmt/.txt path, or an Enrichr library name (resolved + auto-downloaded internally).",
    examples=[
        "# Initialize GSEA — pathways_dict can be an Enrichr library name",
        "gsea_obj = ov.bulk.pyGSEA(ranked_genes, 'MSigDB_Hallmark_2020')",
        "# Run enrichment analysis",
        "enrich_res = gsea_obj.enrichment()",
        "# Visualize enrichment results",
        "gsea_obj.plot_enrichment(num=10, figsize=(3,5))",
        "# Plot GSEA for specific term",
        "gsea_obj.plot_gsea(term_num=0, gene_set_title='KEGG Pathway')"
    ],
    related=["bulk.geneset_enrichment", "bulk.pyDEG.ranking2gsea", "utils.geneset_prepare"]
)
class pyGSEA(object):
    """
    Gene Set Enrichment Analysis (GSEA) wrapper for ranked gene lists.

    Parameters
    ----------
    gene_rnk:pd.DataFrame
        Ranked gene table used for enrichment scoring.
    pathways_dict:dict|str
        Gene sets — a prepared dict, a ``.gmt``/``.txt`` path, or an
        Enrichr library name (resolved and auto-downloaded internally).
    processes:int, optional, default=8
        Number of worker processes.
    permutation_num:int, optional, default=100
        Number of permutations for enrichment significance.
    outdir:str, optional, default='./enrichr_gsea'
        Output directory for reports and plots.
    cutoff:float, optional, default=0.5
        Significance/score threshold for result filtering.
    
    Returns
    -------
    None
        Initializes GSEA analysis settings.
    
    Examples
    --------
    >>> # Initialize GSEA object
    """

    def __init__(self,gene_rnk:pd.DataFrame,pathways_dict:dict,
                 processes:int=1,permutation_num:int=1000,
                 outdir:str='./enrichr_gsea',cutoff:float=0.5,
                 organism:str='Human',backend:str='numpy',
                 weight:float=1.0,min_size:int=15,max_size:int=500,
                 progress:bool=True) -> None:
        """Initialize pyGSEA with ranked genes and pathway libraries.

        Parameters
        ----------
        gene_rnk:pandas.DataFrame
            Ranked gene table equivalent to GSEA ``.rnk`` input.
        pathways_dict:dict
            Dictionary of pathway collections/gene sets.
        processes:int, optional
            Number of parallel worker processes.
        permutation_num:int, optional
            Number of permutations for null distribution estimation.
        outdir:str, optional
            Output directory for GSEA artifacts.
        cutoff:float, optional
            Internal GSEA cutoff threshold.
        """

        self.gene_rnk=gene_rnk
        self.pathways_dict=_resolve_genesets(pathways_dict, organism)
        self.processes=processes
        self.permutation_num=permutation_num
        self.outdir=outdir
        self.cutoff=cutoff
        self.backend=backend
        self.weight=weight
        self.min_size=min_size
        self.max_size=max_size
        self.progress=progress


    def enrichment(self,format:str='png', pval=0.05,seed:int=112)->pd.DataFrame:
        """Run GSEA and return filtered enrichment results.

        Parameters
        ----------
        format:str, optional
            Figure export format used by gseapy output.
        pval:float, optional
            FDR threshold used to filter enriched terms.
        seed:int, optional
            Random seed for permutation reproducibility.

        Returns
        -------
        pandas.DataFrame
            Filtered GSEA result table augmented with plotting columns.
        """

        
        pre_res=geneset_enrichment_GSEA(self.gene_rnk,self.pathways_dict,
                                           self.processes,self.permutation_num,
                                           self.outdir,format,seed,
                                           backend=self.backend,weight=self.weight,
                                           min_size=self.min_size,max_size=self.max_size,
                                           progress=getattr(self,'progress',True))
        self.pre_res=pre_res
        enrich_res=pre_res.res2d[pre_res.res2d['fdr']<pval]
        enrich_res['logp']=-np.log10(enrich_res['fdr']+0.0001)
        enrich_res['logc']=enrich_res['nes']
        enrich_res['num']=enrich_res['matched_size']
        enrich_res['fraction']=enrich_res['matched_size']/enrich_res['geneset_size']
        enrich_res['Term']=enrich_res.index.tolist()
        enrich_res['P-value']=enrich_res['fdr']
        self.enrich_res=enrich_res
        return enrich_res
    
    def plot_gsea(self,term_num:int=0,
                  gene_set_title:str='',
                  figsize:tuple=(3,4),
                  cmap:str='RdBu_r',
                  title_fontsize:int=12,
                  title_y:float=0.95)->matplotlib.figure.Figure:
        """Plot running-enrichment curve for one selected GSEA term.

        Parameters
        ----------
        term_num:int, optional
            Index of term in ``self.enrich_res`` to visualize.
        gene_set_title:str, optional
            Custom plot title. If empty, uses term name.
        figsize:tuple, optional
            Figure size.
        cmap:str, optional
            Colormap for rank metric background.
        title_fontsize:int, optional
            Plot title font size.
        title_y:float, optional
            Y-position of title.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the GSEA running score plot.
        """
        from ._enrich_plot import GSEAPlot
        terms = self.enrich_res.index
        g = GSEAPlot(
        rank_metric=self.pre_res.ranking, term=terms[term_num],figsize=figsize,cmap=cmap,
            **self.pre_res.results[terms[term_num]]
            )
        if gene_set_title=='':
            g.fig.suptitle(terms[term_num],fontsize=title_fontsize,y=title_y)
        else:
            g.fig.suptitle(gene_set_title,fontsize=title_fontsize,y=title_y)
        g.add_axes()
        return g.fig
    
    
    def plot_enrichment(self,num:int=10,node_size:list=[5,10,15],
                        cax_loc:list=[2,0.55,0.5,0.02],cax_fontsize:int=12,
                        fig_title:str='',fig_xlabel:str='Fractions of genes',
                        figsize:tuple=(2,4),cmap:str='YlGnBu',
                        text_knock:int=2,text_maxsize:int=20,**kwargs)->matplotlib.axes._axes.Axes:

        """Plot top GSEA terms as bubble enrichment chart.

        Parameters
        ----------
        num:int, optional
            Number of top enriched terms to visualize.
        node_size:list[int], optional
            Bubble-size legend entries.
        cax_loc:int|list[float], optional
            Colorbar position argument forwarded to plotting helper.
        cax_fontsize:int, optional
            Font size of colorbar/legend text.
        fig_title:str, optional
            Figure title.
        fig_xlabel:str, optional
            X-axis label.
        figsize:tuple, optional
            Figure size.
        cmap:str, optional
            Colormap for significance scale.
        text_knock:int, optional
            Truncation length for long term names.
        text_maxsize:int, optional
            Maximum rendered text size for term labels.

        Returns
        -------
        matplotlib.axes.Axes
            Axis containing the enrichment bubble plot.
        """
        return geneset_plot(self.enrich_res,num=num,node_size=node_size,
                            cax_loc=cax_loc,cax_fontsize=cax_fontsize,
                            fig_title=fig_title,fig_xlabel=fig_xlabel,
                            figsize=figsize,cmap=cmap,text_knock=text_knock,
                            text_maxsize=text_maxsize,**kwargs)

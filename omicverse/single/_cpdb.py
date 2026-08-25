r"""
The downanlysis of cellphonedb
"""

from threading import RLock

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.patches as mpatches
import scanpy as sc
import matplotlib
import anndata
from .._registry import register_function

kpy_install = False
_cpdb_scoring_lock = RLock()


def _call_cpdb_statistical_analysis(
    cpdb_statistical_analysis_method,
    analysis_params,
):
    """Call CellPhoneDB with unique gene names in its scoring matrix."""
    scoring_utils = getattr(cpdb_statistical_analysis_method, "scoring_utils", None)
    scorer_name = "heteromer_geometric_expression_per_cell_type"
    if scoring_utils is None or not hasattr(scoring_utils, scorer_name):
        return cpdb_statistical_analysis_method.call(**analysis_params)

    with _cpdb_scoring_lock:
        original = getattr(scoring_utils, scorer_name)

        def unique_gene_rows(*args, **kwargs):
            matrix = original(*args, **kwargs)
            if matrix.index.has_duplicates:
                # A gene can map to multiple CellPhoneDB protein records.
                matrix = matrix.groupby(level=0, sort=False).mean()
            return matrix

        setattr(scoring_utils, scorer_name, unique_gene_rows)
        try:
            return cpdb_statistical_analysis_method.call(**analysis_params)
        finally:
            setattr(scoring_utils, scorer_name, original)


def global_imports(modulename,shortname = None, asfunction = False):
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:        
        globals()[shortname] = __import__(modulename)

def check_kpy():
    r"""Check if ktplotspy package is installed for CellPhoneDB analysis.
    
    Raises:
        ImportError: If ktplotspy is not installed
    """
    global kpy_install
    try:
        import ktplotspy as kpy
        kpy_install=True
        print('ktplotspy have been install version:',kpy.__version__)
    except ImportError:
        raise ImportError(
            'Please install the ktplotspy: `pip install ktplotspy`.'
        )

def cpdb_network_cal(adata:anndata.AnnData,pvals:list,celltype_key:str)->dict:
    r"""
    Calculate a CPDB (Cell Phone Database) network using gene expression data and return a dictionary of results.

    Parameters
    ----------
    adata:anndata.AnnData
        AnnData object used by CellPhoneDB plotting utilities.
    pvals:list
        CellPhoneDB p-value result table/list passed to ``ktplotspy``.
    celltype_key:str
        Column in ``adata.obs`` containing cell-type annotations.

    Returns
    -------
    dict
        Result dictionary returned by ``kpy.plot_cpdb_heatmap(..., return_tables=True)``.

    """
    check_kpy()
    global kpy_install
    if kpy_install==True:
        global_imports("ktplotspy","kpy")

    cpdb_dict=kpy.plot_cpdb_heatmap(
        adata = adata,
        pvals = pvals,
        celltype_key = celltype_key,
        figsize = (1,1),
        title = "",
        symmetrical = False,
        return_tables=True,
    )
    return cpdb_dict


def cpdb_plot_network(adata:anndata.AnnData,interaction_edges:pd.DataFrame,
                      celltype_key:str,nodecolor_dict=None,
                      edgeswidth_scale:int=10,nodesize_scale:int=1,
                      pos_scale:int=1,pos_size:int=10,figsize:tuple=(5,5),title:str='',
                      legend_ncol:int=3,legend_bbox:tuple=(1,0.2),legend_fontsize:int=10,
                     return_graph:bool=False)->nx.Graph:
    r"""
    Plot a network of interactions between cell types using gene expression data.

    Parameters
    ----------
    adata:anndata.AnnData
        AnnData containing cell-type annotations and optional color metadata.
    interaction_edges:pd.DataFrame
        Interaction edge table with ``SOURCE``, ``TARGET``, and ``COUNT`` columns.
    celltype_key:str
        Column in ``adata.obs`` used to map cell types and colors.
    nodecolor_dict:dict or None
        Optional mapping from cell type to color.
    edgeswidth_scale:int
        Divisor applied to edge counts when computing edge widths.
    nodesize_scale:int
        Divisor applied to node interaction totals when computing node sizes.
    pos_scale:int
        Scale parameter for spring layout coordinates.
    pos_size:int
        ``k`` parameter base used in spring layout.
    figsize:tuple
        Figure size for network plot.
    title:str
        Plot title.
    legend_ncol:int
        Number of legend columns.
    legend_bbox:tuple
        Legend anchor location.
    legend_fontsize:int
        Legend font size.
    return_graph:bool
        Whether to return network object instead of axes.

    Returns
    -------
    matplotlib.axes.Axes or nx.Graph
        Axes if ``return_graph=False``; network graph otherwise.

    """
    check_kpy()
    global kpy_install
    if kpy_install==True:
        global_imports("ktplotspy","kpy")

    #set Digraph of cellphonedb
    G=nx.DiGraph()
    for i in interaction_edges.index:
        G.add_edge(interaction_edges.loc[i,'SOURCE'],
                   interaction_edges.loc[i,'TARGET'],
                   weight=interaction_edges.loc[i,'COUNT'],)
    
    #set celltypekey's color
    if nodecolor_dict!=None:
        type_color_all=nodecolor_dict
    else:
        if '{}_colors'.format(celltype_key) in adata.uns:
            type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,adata.uns['{}_colors'.format(celltype_key)]))
        else:
            if len(adata.obs[celltype_key].cat.categories)>28:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.default_102))
            else:
                type_color_all=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.zeileis_28))
    
    #set G_nodes_dict
    nodes=[]
    G_degree=dict(G.degree(G.nodes()))


    G_nodes_dict={}
    links = []
    for i in G.edges:
        if i[0] not in G_nodes_dict.keys():
            G_nodes_dict[i[0]]=0
        if i[1] not in G_nodes_dict.keys():
            G_nodes_dict[i[1]]=0
        links.append({"source": i[0], "target": i[1]})
        weight=G.get_edge_data(i[0],i[1])['weight']
        G_nodes_dict[i[0]]+=weight
        G_nodes_dict[i[1]]+=weight
        
    #plot
    fig, ax = plt.subplots(figsize=figsize) 
    pos = nx.spring_layout(G, scale=pos_scale, k=(pos_size)/np.sqrt(G.order()))
    p=dict(G.nodes)

    nodesize=np.array([G_nodes_dict[u] for u in G.nodes()])/nodesize_scale
    nodecolos=[type_color_all[u] for u in G.nodes()]
    nx.draw_networkx_nodes(G, pos, nodelist=p,node_size=nodesize,node_color=nodecolos)

    edgewidth = np.array([G.get_edge_data(u, v)['weight'] for u, v in G.edges()])/edgeswidth_scale
    nx.draw_networkx_edges(G, pos,width=edgewidth)


    #label_options = {"ec": "white", "fc": "white", "alpha": 0.6}
    #nx.draw_networkx_labels(G, pos, font_size=10,) #bbox=label_options)
    plt.grid(False)
    plt.axis("off")
    plt.xlim(-2,2)
    plt.ylim(-2,1.5)

    labels = adata.obs[celltype_key].cat.categories
    #用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend
    color = [type_color_all[u] for u in labels]
    patches = [mpatches.Patch(color=type_color_all[u], label=u) for u in labels ] 

    #plt.xlim(-0.05, 1.05)
    #plt.ylim(-0.05, 1.05)
    plt.axis("off")
    plt.title(title)
    plt.legend(handles=patches,
               bbox_to_anchor=legend_bbox, 
               ncol=legend_ncol,
               fontsize=legend_fontsize)
    if return_graph==True:
        return G
    else:
        return ax
    #return {'Graph':G,'ax':ax}

def cpdb_plot_interaction(adata:anndata.AnnData,cell_type1:str,cell_type2:str,
                          means:pd.DataFrame,pvals:pd.DataFrame,
                          celltype_key:str,genes=None,
                         keep_significant_only:bool=True,figsize:tuple = (4,8),title:str="",
                         max_size:int=1,highlight_size:float = 0.75,standard_scale:bool = True,
                         cmap_name:str='viridis',
                         ytickslabel_fontsize:int=8,xtickslabel_fontsize:int=8,title_fontsize:int=10)->matplotlib.axes._axes.Axes:
    r"""Plot CellPhoneDB cell-cell interactions between two cell types.

    Parameters
    ----------
    adata:anndata.AnnData
        AnnData used for CellPhoneDB interaction plotting.
    cell_type1:str
        Sender cell type.
    cell_type2:str
        Receiver cell type.
    means:pd.DataFrame
        CellPhoneDB means table.
    pvals:pd.DataFrame
        CellPhoneDB p-value table.
    celltype_key:str
        Column in ``adata.obs`` storing cell-type labels.
    genes:list or None
        Optional subset of genes/interactions to display.
    keep_significant_only:bool
        Whether to retain only significant interactions.
    figsize:tuple
        Figure size.
    title:str
        Plot title.
    max_size:int
        Maximum dot size in bubble plot.
    highlight_size:float
        Size used for highlighted interactions.
    standard_scale:bool
        Whether to standard-scale values before plotting.
    cmap_name:str
        Colormap name.
    ytickslabel_fontsize:int
        Font size of y-axis tick labels.
    xtickslabel_fontsize:int
        Font size of x-axis tick labels.
    title_fontsize:int
        Title font size.
    
    Returns
    -------
    matplotlib.axes.Axes
        Axes containing CellPhoneDB interaction bubble plot.

    """
    check_kpy()
    global kpy_install
    if kpy_install==True:
        global_imports("ktplotspy","kpy")

    fig=kpy.plot_cpdb(
        adata = adata,
        cell_type1 = cell_type1,
        cell_type2 = cell_type2, 
        means = means,
        pvals = pvals,
        celltype_key = celltype_key,
        genes = genes,
        keep_significant_only=keep_significant_only,
        figsize = figsize,
        title = "",
        max_size = max_size,
        highlight_size = highlight_size,
        standard_scale = standard_scale,
        cmap_name=cmap_name
    ).draw()
    
    #ytickslabels
    labels=fig.get_axes()[0].yaxis.get_ticklabels()
    plt.setp(labels, fontsize=ytickslabel_fontsize)

    #xtickslabels
    labels=fig.get_axes()[0].xaxis.get_ticklabels()
    plt.setp(labels, fontsize=xtickslabel_fontsize)

    fig.get_axes()[0].set_title(title,fontsize=title_fontsize)
    
    return fig.get_axes()[0]

def cpdb_submeans_exacted(means:pd.DataFrame,cell_names:str,cell_type:str='ligand')->pd.DataFrame:
    r"""Extract subset of CellPhoneDB means DataFrame for specific cell type.

    Parameters
    ----------
    means:pd.DataFrame
        CellPhoneDB means table.
    cell_names:str
        Cell type name used in pair columns.
    cell_type:str
        Extraction side, either ``'ligand'`` or ``'receptor'``.

    Returns
    -------
    pd.DataFrame
        Filtered means table containing requested side-specific columns.

    """
    if cell_type=='ligand':
        means_columns=means.columns[:11].tolist()+means.columns[means.columns.str.contains('{}\|'.format(cell_names))].tolist()
    elif cell_type=='receptor':
        means_columns=means.columns[:11].tolist()+means.columns[means.columns.str.contains('\|{}'.format(cell_names))].tolist()
    else:
        raise ValueError('cell_type must be ligand or receptor')
    return means.loc[:,means_columns]

def cpdb_interaction_filtered(adata:anndata.AnnData,cell_type1:str,cell_type2:str,
                              means:pd.DataFrame,pvals:pd.DataFrame,celltype_key:str,genes=None,
                         keep_significant_only:bool=True,figsize:tuple = (0,0),
                         max_size:int=1,highlight_size:float = 0.75,standard_scale:bool = True,cmap_name:str='viridis',)->list:
    r"""
    Return unique interaction groups between two cell types after filtering.

    Parameters
    ----------
    adata:anndata.AnnData
        AnnData object used for interaction filtering context.
    cell_type1:str
        Sender cell type.
    cell_type2:str
        Receiver cell type.
    means:pd.DataFrame
        CellPhoneDB means table.
    pvals:pd.DataFrame
        CellPhoneDB p-value table.
    celltype_key:str
        Cell-type annotation column in ``adata.obs``.
    genes:list or None
        Optional genes/interactions to restrict the query.
    keep_significant_only:bool
        Whether to keep only statistically significant interactions.
    figsize:tuple
        Placeholder figure size passed to ``ktplotspy``.
    max_size:int
        Bubble-size cap used in backend plotting helper.
    highlight_size:float
        Highlight size used in backend plotting helper.
    standard_scale:bool
        Whether to standard-scale interaction values.
    cmap_name:str
        Colormap name.

    Returns
    -------
    list
        Unique ``interaction_group`` values passing filters.

    """
    check_kpy()
    global kpy_install
    if kpy_install==True:
        global_imports("ktplotspy","kpy")

    res=kpy.plot_cpdb(
        adata = adata,
        cell_type1 = cell_type1,
        cell_type2 = cell_type2, 
        means = means,
        pvals = pvals,
        celltype_key = celltype_key,
        genes = genes,
        keep_significant_only=keep_significant_only,
        figsize = figsize,
        title = "",
        max_size = max_size,
        highlight_size = highlight_size,
        standard_scale = standard_scale,
        cmap_name=cmap_name,
        return_table=True
    )

    return list(set(res['interaction_group']))

def cpdb_exact_target(means,target_cells):
    import re
    
    t_dict=[]
    for t in target_cells:
        escaped_str = re.escape('|'+t)
        target_names=means.columns[means.columns.str.contains(escaped_str)].tolist()
        t_dict+=target_names
    #print(t_dict)
    target_sub=means[means.columns[:10].tolist()+t_dict]
    return target_sub

def cpdb_exact_source(means,source_cells):
    import re
    
    t_dict=[]
    for t in source_cells:
        escaped_str = re.escape(t+'|')
        source_names=means.columns[means.columns.str.contains(escaped_str)].tolist()
        t_dict+=source_names
    #print(t_dict)
    source_sub=means[means.columns[:10].tolist()+t_dict]
    return source_sub


from tqdm import tqdm
def cpdb2cellchat(df):
    new_columns = ['source', 'target', 'ligand', 'receptor', 'prob', 'pval', 'interaction_name', 'interaction_name_2', 'pathway_name', 'annotation', 'evidence']
    new_df = pd.DataFrame(columns=new_columns)
    
    # 遍历每一行和细胞对列
    for index, row in tqdm(df.iterrows()):
        for col in df.columns[14:]:  # 假设从第14列开始是细胞对
            if pd.notna(row[col]):
                source, target = col.split('|')  # 通过 '|' 分割 source 和 target
                if pd.notna(row['gene_a']) and pd.notna(row['gene_b']):
                    interaction_name=row['interacting_pair'].split('_')
                    if len(interaction_name)>2:
                        interaction_name_2=row['interacting_pair'].split('_')[0]+' - ('+'+'.join(row['interacting_pair'].split('_')[1:])+')'
                    else:
                        interaction_name_2=row['interacting_pair'].split('_')[0]+' - '+'+'.join(row['interacting_pair'].split('_')[1:])
                    new_row = {
                        'source': source,
                        'target': target,
                        'ligand': row['gene_a'],
                        'receptor': row['gene_b'],
                        'prob': row['rank'],  # 假设 rank 是概率
                        'pval': 0,
                        'interaction_name': row['interacting_pair'],
                        'interaction_name_2': interaction_name_2,  # 假设相同
                        'pathway_name': row['classification'],
                        'annotation': row['annotation_strategy'],
                        'evidence': 'curated'  # 假设证据为 curated
                    }
                    new_df = new_df.append(new_row, ignore_index=True)
    
    # 显示新的 DataFrame
    return new_df


def cellphonedb_v5(adata,
                           celltype_key='celltype',
                           min_cell_fraction=0.005,
                           min_genes=200,
                           min_cells=3,
                           cpdb_file_path=None,
                           auto_download=True,
                           iterations=1000,
                           threshold=0.1,
                           pvalue=0.05,
                           threads=10,
                           output_dir=None,
                           temp_dir=None,
                           cleanup_temp=True,
                           debug=False,
                           separator='|',
                           results_key: str = 'cpdb_results',
                           comm_key: str = 'cpdb_comm',
                           **kwargs):
    """
    Run CellPhoneDB statistical analysis with proper file handling
    
    Parameters
    ----------
    adata:AnnData
        Annotated data matrix
    celltype_key:str
        Column name in adata.obs containing cell type annotations
    min_cell_fraction:float
        Minimum fraction of total cells required for a cell type to be included
    min_genes:int
        Minimum number of genes required per cell
    min_cells:int
        Minimum number of cells required per gene
    cpdb_file_path:str or None
        Path to CellPhoneDB database zip file. If None, will try to find automatically
        (cache → ./cellphonedb.zip → ~/cellphonedb.zip), and auto-download if missing.
    auto_download:bool, default=True
        If True and the database is not found locally, automatically download
        from the official CellPhoneDB GitHub release into
        ``$OVAGENT_HOME/data_lake/cellphonedb.zip``. Disable to fail fast in
        offline environments.
    iterations:int
        Number of shufflings performed in the analysis
    threshold:float
        Min % of cells expressing a gene for this to be employed in the analysis
    pvalue:float
        P-value threshold to employ for significance
    threads:int
        Number of threads to use in the analysis
    output_dir:str or None
        Directory to save results. If None, creates temporary directory
    temp_dir:str or None
        Directory for temporary files. If None, uses system temp
    cleanup_temp:bool
        Whether to clean up temporary files after analysis
    debug:bool
        Saves all intermediate tables employed during the analysis
    separator:str
        String to employ to separate cells in the results dataframes
    **kwargs:dict
        Additional parameters forwarded to ``cpdb_statistical_analysis_method.call``.
        
    Returns
    -------
    Tuple[dict,anndata.AnnData]
        Raw CellPhoneDB result dict and formatted communication AnnData. The
        same objects are also written to ``adata.uns[results_key]`` and
        ``adata.uns[comm_key]`` so downstream ``ov.pl.ccc_*`` plots can work
        directly on the original ``adata``.
    """
    import os
    import tempfile
    import shutil
    from pathlib import Path
    import pandas as pd
    import scanpy as sc
    
    # Validate inputs
    if celltype_key not in adata.obs.columns:
        raise ValueError(f"celltype_key '{celltype_key}' not found in adata.obs")
    
    print("🔬 Starting CellPhoneDB analysis...")
    print(f"   - Original data: {adata.shape[0]} cells, {adata.shape[1]} genes")
    
    # Step 1: Filter cell types by minimum cell fraction
    ct_counts = adata.obs[celltype_key].value_counts()
    min_cells_required = int(adata.shape[0] * min_cell_fraction)
    valid_celltypes = ct_counts[ct_counts > min_cells_required].index
    
    print(f"   - Cell types passing {min_cell_fraction*100}% threshold: {len(valid_celltypes)}")
    print(f"   - Minimum cells required: {min_cells_required}")
    
    if len(valid_celltypes) == 0:
        raise ValueError("No cell types pass the minimum cell fraction threshold")
    
    # Step 2: Subset and preprocess data
    adata_filtered = adata[adata.obs[celltype_key].isin(valid_celltypes)].copy()
    
    # Use raw data if available, otherwise use X
    if adata_filtered.raw is not None:
        adata_pp = adata_filtered.raw.to_adata()
        adata_pp.obs = adata_filtered.obs.copy()
    else:
        adata_pp = adata_filtered.copy()
    
    print(f"   - After filtering: {adata_pp.shape[0]} cells, {adata_pp.shape[1]} genes")
    
    # Apply standard preprocessing
    sc.pp.filter_cells(adata_pp, min_genes=min_genes)
    sc.pp.filter_genes(adata_pp, min_cells=min_cells)
    
    print(f"   - After preprocessing: {adata_pp.shape[0]} cells, {adata_pp.shape[1]} genes")
    
    # Step 3: Setup directories
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix='cpdb_temp_')
        temp_created = True
    else:
        temp_created = False
        os.makedirs(temp_dir, exist_ok=True)
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='cpdb_results_')
        output_created = True
    else:
        output_created = False
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"   - Temporary directory: {temp_dir}")
    print(f"   - Output directory: {output_dir}")
    
    try:
        # Step 4: Create temporary files for CellPhoneDB
        # Create counts file (AnnData format)
        counts_file = os.path.join(temp_dir, 'counts_matrix.h5ad')
        adata_counts = sc.AnnData(
            X=adata_pp.X,
            obs=pd.DataFrame(index=adata_pp.obs.index),
            var=pd.DataFrame(index=adata_pp.var.index)
        )
        adata_counts.write_h5ad(counts_file, compression='gzip')
        
        # Create metadata file
        meta_file = os.path.join(temp_dir, 'metadata.tsv')
        df_meta = pd.DataFrame({
            'Cell': list(adata_pp.obs.index),
            'cell_type': list(adata_pp.obs[celltype_key])
        })
        df_meta.set_index('Cell', inplace=True)
        df_meta.to_csv(meta_file, sep='\t')
        
        print("   - Created temporary input files")
        
        # Step 5: Find CellPhoneDB database if not provided
        if cpdb_file_path is None:
            try:
                from ..utils._ovagent_paths import ovagent_home
                cache_path = ovagent_home() / "data_lake" / "cellphonedb.zip"
            except Exception:
                cache_path = Path(os.path.expanduser("~/.omicverse/data_lake/cellphonedb.zip"))

            if cache_path.exists():
                cpdb_file_path = str(cache_path)
            elif auto_download:
                print("   - CellPhoneDB DB not found locally; auto-downloading...")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cpdb_file_path = download_cellphonedb_database(str(cache_path))
            else:
                raise FileNotFoundError(
                    "CellPhoneDB database not found. Pass cpdb_file_path=, "
                    "set auto_download=True, or call "
                    "ov.single.download_cellphonedb_database() first."
                )
        
        print(f"   - Using CellPhoneDB database: {cpdb_file_path}")
        
        # Step 6: Run CellPhoneDB analysis
        try:
            from cellphonedb.src.core.methods import cpdb_statistical_analysis_method
        except ImportError:
            raise ImportError(
                "CellPhoneDB not installed. Please install with: "
                "pip install cellphonedb"
            )
        
        print("   - Running CellPhoneDB statistical analysis...")
        
        # Prepare parameters
        analysis_params = {
            'cpdb_file_path': cpdb_file_path,
            'meta_file_path': meta_file,
            'counts_file_path': counts_file,
            'counts_data': 'hgnc_symbol',
            'active_tfs_file_path': None,
            'microenvs_file_path': None,
            'score_interactions': True,
            'iterations': iterations,
            'threshold': threshold,
            'threads': threads,
            'debug_seed': 42,
            'result_precision': 3,
            'pvalue': pvalue,
            'subsampling': False,
            'subsampling_log': False,
            'subsampling_num_pc': 100,
            'subsampling_num_cells': 1000,
            'separator': separator,
            'debug': debug,
            'output_path': output_dir,
            'output_suffix': None
        }
        
        # Update with any additional kwargs
        analysis_params.update(kwargs)
        
        # Run analysis
        cpdb_results = _call_cpdb_statistical_analysis(
            cpdb_statistical_analysis_method,
            analysis_params,
        )
        
        print("   - CellPhoneDB analysis completed successfully!")
        
        # Step 7: Format results for visualization
        print("   - Formatting results for visualization...")
        
        adata_cpdb = format_cpdb_results(cpdb_results, separator=separator)
        adata.uns[results_key] = cpdb_results
        adata.uns[comm_key] = adata_cpdb
        
        print(f"   - Created visualization AnnData: {adata_cpdb.shape}")
        print(f"   - Cell interactions: {adata_cpdb.n_obs}")
        print(f"   - L-R pairs: {adata_cpdb.n_vars}")
        
        return cpdb_results, adata_cpdb
        
    finally:
        # Step 8: Cleanup temporary files if requested
        if cleanup_temp:
            if temp_created and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"   - Cleaned up temporary directory: {temp_dir}")
            
            if output_created and os.path.exists(output_dir):
                # Only clean output if we created it and user didn't specify it
                pass  # Keep results by default
        
        print("✅ CellPhoneDB analysis pipeline completed!")


def format_cpdb_results(cpdb_results, separator='|'):
    """
    Format CellPhoneDB results into AnnData object for CellChatViz
    
    Parameters
    ----------
    cpdb_results:dict
        Result dictionary returned by CellPhoneDB statistical analysis.
    separator:str
        Separator used in sender|receiver pair column names.

    Returns
    -------
    anndata.AnnData
        AnnData formatted for CellChatViz-style plotting.
    """
    import pandas as pd
    import scanpy as sc
    
    # Extract results
    means_df = cpdb_results['means']
    pvalues_df = cpdb_results['pvalues']
    
    # Identify cell type pair columns (usually start after column 12 or 13)
    info_cols = []
    pair_cols = []
    
    for col in means_df.columns:
        if separator in str(col):
            pair_cols.append(col)
        else:
            info_cols.append(col)
    
    print(f"   - Found {len(info_cols)} info columns and {len(pair_cols)} cell type pairs")
    
    if len(pair_cols) == 0:
        raise ValueError(f"No cell type pair columns found with separator '{separator}'")
    
    # Create AnnData object
    # X matrix: cell pairs (obs) x L-R interactions (vars)
    X_data = means_df[pair_cols].T  # Transpose so pairs are observations
    
    adata_cpdb = sc.AnnData(X=X_data)
    
    # Add layers
    adata_cpdb.layers['means'] = means_df[pair_cols].T
    adata_cpdb.layers['pvalues'] = pvalues_df[pair_cols].T
    
    # Add variable (L-R pair) information
    adata_cpdb.var = means_df[info_cols].copy()
    adata_cpdb.var['interaction_name'] = adata_cpdb.var['interacting_pair']
    
    # Add observation (cell pair) information
    adata_cpdb.obs['sender'] = [pair.split(separator)[0] for pair in adata_cpdb.obs.index]
    adata_cpdb.obs['receiver'] = [pair.split(separator)[1] for pair in adata_cpdb.obs.index]
    
    # Add interaction classification if available
    if 'classification' in adata_cpdb.var.columns:
        print(f"   - Found {adata_cpdb.var['classification'].nunique()} pathway classifications")
    adata_cpdb.uns["comm_source"] = "cellphonedb"
    adata_cpdb.uns["cpdb_separator"] = separator
    return adata_cpdb


def format_cpdb_results_for_viz(cpdb_results, separator='|'):
    """Backward-compatible alias of :func:`format_cpdb_results`."""
    return format_cpdb_results(cpdb_results, separator=separator)


def create_cellchatviz_from_cpdb(cpdb_results, separator='|', palette=None):
    """
    Create CellChatViz object directly from CellPhoneDB results
    
    Parameters
    ----------
    cpdb_results:dict
        Result dictionary returned by CellPhoneDB analysis.
    separator:str
        Separator used in sender|receiver pair column names.
    palette:dict or list or None
        Optional palette for cell-type colors.

    Returns
    -------
    CellChatViz
        Initialized visualization object.
    """
    # Format results
    adata_cpdb = format_cpdb_results(cpdb_results, separator=separator)
    
    # Create and return CellChatViz object
    from ..pl._cpdbviz import CellChatViz
    viz = CellChatViz(adata_cpdb, palette=palette)
    
    print(f"✅ Created CellChatViz with {viz.n_cell_types} cell types")
    
    return viz


@register_function(
    aliases=[
        'CellPhoneDB 数据库下载', 'CellPhoneDB database download',
        'download_cellphonedb_database', 'download_cellphonedb',
        'cpdb download', 'fetch cellphonedb',
    ],
    category="single",
    description=(
        "Download the CellPhoneDB v5 ligand-receptor database (cellphonedb.zip) "
        "from the official GitHub mirror. Required by ov.single.cellphonedb_v5 "
        "before running cell-cell communication analysis. Caches under "
        "$OVAGENT_HOME/data_lake/cellphonedb.zip and is idempotent."
    ),
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=[
        'ov.single.download_cellphonedb_database()',
        'ov.single.download_cellphonedb_database("./cellphonedb.zip")',
    ],
    related=['single.cellphonedb_v5', 'single.validate_cpdb_database']
)
def download_cellphonedb_database(download_path=None, force_download=False):
    """
    Download CellPhoneDB database with fallback URLs.

    Uses ``ov.datasets.download_data_requests`` (the standard omicverse
    downloader, with realistic User-Agent + tqdm progress + caching).

    Parameters
    ----------
    download_path:str or None
        Target path of downloaded ``cellphonedb.zip`` file. If ``None``,
        defaults to ``$OVAGENT_HOME/data_lake/cellphonedb.zip`` (so the
        same cache is shared across ov.Agent sessions).
    force_download:bool
        Whether to redownload when file already exists.

    Returns
    -------
    str
        Path to downloaded database archive.
    """
    import os
    from pathlib import Path
    from ..datasets._datasets import download_data_requests

    if download_path is None:
        try:
            from ..utils._ovagent_paths import ovagent_home
            download_path = ovagent_home() / "data_lake" / "cellphonedb.zip"
        except Exception:
            download_path = Path("./cellphonedb.zip")

    download_path = Path(download_path)

    if download_path.exists() and not force_download:
        print(f"✅ CellPhoneDB database already exists: {download_path}")
        return str(download_path)
    if download_path.exists() and force_download:
        download_path.unlink()

    download_path.parent.mkdir(parents=True, exist_ok=True)

    download_urls = [
        "https://github.com/ventolab/cellphonedb-data/raw/refs/heads/master/cellphonedb.zip",
        "https://starlit.oss-cn-beijing.aliyuncs.com/single/cellphonedb.zip",
    ]

    print("📥 Downloading CellPhoneDB database...")
    last_err = None
    for i, url in enumerate(download_urls, 1):
        try:
            print(f"   - Trying URL {i}/{len(download_urls)}: {url}")
            result = download_data_requests(
                url=url,
                file_path=download_path.name,
                dir=str(download_path.parent),
            )
            if Path(result).stat().st_size > 0:
                print(f"✅ Successfully downloaded CellPhoneDB database to: {result}")
                return str(result)
            Path(result).unlink(missing_ok=True)
        except Exception as exc:
            last_err = exc
            print(f"   ✗ {url} failed: {exc}")
            try:
                download_path.unlink()
            except FileNotFoundError:
                pass
            continue

    raise RuntimeError(
        "Failed to download CellPhoneDB database from all mirrors. "
        f"Last error: {last_err}. Check network or pass cpdb_file_path= manually."
    )


def validate_cpdb_database(cpdb_file_path):
    """
    Validate CellPhoneDB database file
    
    Parameters
    ----------
    cpdb_file_path:str
        Path to local CellPhoneDB database archive.

    Returns
    -------
    str
        Validated (or auto-downloaded) database path.
    """
    import os
    from pathlib import Path
    
    if cpdb_file_path is None:
        raise ValueError("cpdb_file_path is required. Use download_cellphonedb_database() to get the database.")
    
    cpdb_path = Path(cpdb_file_path)
    
    # If path doesn't exist, try to download
    if not cpdb_path.exists():
        print(f"❌ Database not found at: {cpdb_file_path}")
        print("🔄 Attempting to download database...")
        
        # If the provided path looks like a filename, use it for download
        if cpdb_path.name.endswith('.zip'):
            return download_cellphonedb_database(cpdb_file_path)
        else:
            # Download to default location and return that path
            downloaded_path = download_cellphonedb_database()
            print(f"💡 Database downloaded to: {downloaded_path}")
            print(f"   You can use this path in future calls: cpdb_file_path='{downloaded_path}'")
            return downloaded_path
    
    # Validate file size
    file_size = cpdb_path.stat().st_size
    if file_size == 0:
        print(f"❌ Database file is empty: {cpdb_file_path}")
        print("🔄 Re-downloading database...")
        return download_cellphonedb_database(cpdb_file_path, force_download=True)
    
    print(f"✅ Valid CellPhoneDB database found: {cpdb_file_path} ({file_size/1024/1024:.1f} MB)")
    return str(cpdb_path)


@register_function(
    aliases=['CellPhoneDB 分析', 'run_cellphonedb_v5', 'cell-cell communication v5'],
    category="single",
    description="Run CellPhoneDB v5 statistical ligand-receptor analysis to identify significant cell-cell communication pairs.",
    prerequisites={'optional_functions': ['pp.qc', 'pp.preprocess']},
    requires={'obs': ['celltype labels'], 'var': ['gene symbols']},
    produces={
        'layers': ['means', 'pvalues'],
        'obs': ['sender', 'receiver'],
        'var': ['interaction metadata'],
    },
    auto_fix='escalate',
    examples=['ov.single.run_cellphonedb_v5(adata, cpdb_file_path="./cellphonedb.zip", celltype_key="cell_labels", iterations=1000, pvalue=0.05)'],
    related=['pl.CellChatViz', 'single.pathway_enrichment']
)
def run_cellphonedb_v5(adata, 
                           cpdb_file_path,  # Now mandatory
                           celltype_key='celltype',
                           min_cell_fraction=0.005,
                           min_genes=200,
                           min_cells=3,
                           iterations=1000,
                           threshold=0.1,
                           pvalue=0.05,
                           threads=10,
                           output_dir=None,
                           temp_dir=None,
                           cleanup_temp=True,
                           debug=False,
                           separator='|',
                           results_key: str = 'cpdb_results',
                           comm_key: str = 'cpdb_comm',
                           **kwargs):
    """
    Run CellPhoneDB statistical analysis with automatic database download
    
    Parameters
    ----------
    adata:AnnData
        Annotated data matrix
    cpdb_file_path:str
        Path to CellPhoneDB database zip file (REQUIRED)
        If file doesn't exist, will attempt automatic download
    celltype_key:str
        Column name in adata.obs containing cell type annotations
    min_cell_fraction:float
        Minimum fraction of total cells required for a cell type to be included
    min_genes:int
        Minimum number of genes required per cell
    min_cells:int
        Minimum number of cells required per gene
    iterations:int
        Number of shufflings performed in the analysis
    threshold:float
        Min % of cells expressing a gene for this to be employed in the analysis
    pvalue:float
        P-value threshold to employ for significance
    threads:int
        Number of threads to use in the analysis
    output_dir:str or None
        Directory to save results. If None, creates temporary directory
    temp_dir:str or None
        Directory for temporary files. If None, uses system temp
    cleanup_temp:bool
        Whether to clean up temporary files after analysis
    debug:bool
        Saves all intermediate tables employed during the analysis
    separator:str
        String to employ to separate cells in the results dataframes
    **kwargs:dict
        Additional parameters forwarded to ``cpdb_statistical_analysis_method.call``.

    Returns
    -------
    Tuple[dict,anndata.AnnData]
        Raw CellPhoneDB result dict and visualization-ready communication
        AnnData. The same objects are also written to
        ``adata.uns[results_key]`` and ``adata.uns[comm_key]`` so downstream
        ``ov.pl.ccc_*`` plots can work directly on the original ``adata``.

    Examples
    --------
    # Basic usage - will download database automatically if needed
    cpdb_results, adata_cpdb = run_cellphonedb_analysis(
        adata, 
        cpdb_file_path='./cellphonedb.zip',
        celltype_key='celltype_minor'
    )
    
    # Advanced usage
    cpdb_results, adata_cpdb = run_cellphonedb_analysis(
        adata,
        cpdb_file_path='/path/to/cellphonedb.zip',
        celltype_key='celltype_minor',
        min_cell_fraction=0.01,
        iterations=2000,
        threads=20
    )
    """
    import os
    import tempfile
    import shutil
    from pathlib import Path
    import pandas as pd
    import scanpy as sc
    
    # Step 1: Validate and download database if necessary
    print("🔬 Starting CellPhoneDB analysis...")
    cpdb_file_path = validate_cpdb_database(cpdb_file_path)
    
    # Validate inputs
    if celltype_key not in adata.obs.columns:
        raise ValueError(f"celltype_key '{celltype_key}' not found in adata.obs")
    
    print(f"   - Original data: {adata.shape[0]} cells, {adata.shape[1]} genes")
    
    # Step 2: Filter cell types by minimum cell fraction
    ct_counts = adata.obs[celltype_key].value_counts()
    min_cells_required = int(adata.shape[0] * min_cell_fraction)
    valid_celltypes = ct_counts[ct_counts > min_cells_required].index
    
    print(f"   - Cell types passing {min_cell_fraction*100}% threshold: {len(valid_celltypes)}")
    print(f"   - Minimum cells required: {min_cells_required}")
    
    if len(valid_celltypes) == 0:
        raise ValueError("No cell types pass the minimum cell fraction threshold")
    
    # Step 3: Subset and preprocess data
    adata_filtered = adata[adata.obs[celltype_key].isin(valid_celltypes)].copy()
    
    # Use raw data if available, otherwise use X
    if adata_filtered.raw is not None:
        adata_pp = adata_filtered.raw.to_adata()
        adata_pp.obs = adata_filtered.obs.copy()
    else:
        adata_pp = adata_filtered.copy()
    
    print(f"   - After filtering: {adata_pp.shape[0]} cells, {adata_pp.shape[1]} genes")
    
    # Apply standard preprocessing
    sc.pp.filter_cells(adata_pp, min_genes=min_genes)
    sc.pp.filter_genes(adata_pp, min_cells=min_cells)
    
    print(f"   - After preprocessing: {adata_pp.shape[0]} cells, {adata_pp.shape[1]} genes")
    
    # Step 4: Setup directories
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix='cpdb_temp_')
        temp_created = True
    else:
        temp_created = False
        os.makedirs(temp_dir, exist_ok=True)
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='cpdb_results_')
        output_created = True
    else:
        output_created = False
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"   - Temporary directory: {temp_dir}")
    print(f"   - Output directory: {output_dir}")
    
    try:
        # Step 5: Create temporary files for CellPhoneDB
        # Create counts file (AnnData format)
        counts_file = os.path.join(temp_dir, 'counts_matrix.h5ad')
        adata_counts = sc.AnnData(
            X=adata_pp.X,
            obs=pd.DataFrame(index=adata_pp.obs.index),
            var=pd.DataFrame(index=adata_pp.var.index)
        )
        adata_counts.write_h5ad(counts_file, compression='gzip')
        
        # Create metadata file
        meta_file = os.path.join(temp_dir, 'metadata.tsv')
        df_meta = pd.DataFrame({
            'Cell': list(adata_pp.obs.index),
            'cell_type': list(adata_pp.obs[celltype_key])
        })
        df_meta.set_index('Cell', inplace=True)
        df_meta.to_csv(meta_file, sep='\t')
        
        print("   - Created temporary input files")
        
        # Step 6: Run CellPhoneDB analysis
        try:
            from cellphonedb.src.core.methods import cpdb_statistical_analysis_method
        except ImportError:
            raise ImportError(
                "CellPhoneDB not installed. Please install with: "
                "pip install cellphonedb"
            )
        
        print("   - Running CellPhoneDB statistical analysis...")
        
        # Prepare parameters
        analysis_params = {
            'cpdb_file_path': cpdb_file_path,
            'meta_file_path': meta_file,
            'counts_file_path': counts_file,
            'counts_data': 'hgnc_symbol',
            'active_tfs_file_path': None,
            'microenvs_file_path': None,
            'score_interactions': True,
            'iterations': iterations,
            'threshold': threshold,
            'threads': threads,
            'debug_seed': 42,
            'result_precision': 3,
            'pvalue': pvalue,
            'subsampling': False,
            'subsampling_log': False,
            'subsampling_num_pc': 100,
            'subsampling_num_cells': 1000,
            'separator': separator,
            'debug': debug,
            'output_path': output_dir,
            'output_suffix': None
        }
        
        # Update with any additional kwargs
        analysis_params.update(kwargs)
        
        # Run analysis
        cpdb_results = _call_cpdb_statistical_analysis(
            cpdb_statistical_analysis_method,
            analysis_params,
        )
        
        print("   - CellPhoneDB analysis completed successfully!")
        
        # Step 7: Format results for visualization
        print("   - Formatting results for visualization...")
        
        adata_cpdb = format_cpdb_results(cpdb_results, separator=separator)
        adata.uns[results_key] = cpdb_results
        adata.uns[comm_key] = adata_cpdb
        
        print(f"   - Created visualization AnnData: {adata_cpdb.shape}")
        print(f"   - Cell interactions: {adata_cpdb.n_obs}")
        print(f"   - L-R pairs: {adata_cpdb.n_vars}")
        
        return cpdb_results, adata_cpdb
        
    finally:
        # Step 8: Cleanup temporary files if requested
        if cleanup_temp:
            if temp_created and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"   - Cleaned up temporary directory: {temp_dir}")
            
            if output_created and os.path.exists(output_dir):
                # Only clean output if we created it and user didn't specify it
                pass  # Keep results by default
        
        print("✅ CellPhoneDB analysis pipeline completed!")

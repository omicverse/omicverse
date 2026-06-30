import matplotlib.pyplot as plt
import matplotlib.patches as mpatches




def plot_tensor_single(adata, adata_aggr = None, state = 'joint', 
                       attractor = None, basis = 'umap', color ='attractor', 
                       color_map = None, size = 20, alpha = 0.5, ax = None, 
                       show = None, filter_cells = False, member_thresh = 0.05, density =2,
                       n_jobs = -1,**kwargs):
    """
    Function to plot a single tensor graph with assgined components
    
    Parameters
    ----------
    adata: AnnData object
    adata_aggr: AnnData object
    state: str
        State of the tensor graph, 'spliced', 'unspliced' or 'joint'
    attractor: int
        Attractor index
    basis: str
        Dimensionality reduction basis for the plot
    color: str
        Color of the cells, 'attractor' or 'rho'
    color_map: str
        Color map for the plot
    size: int
        Size of the cells
    alpha: float    
        Transparency of the cells
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes for the plot
    show: bool
        Show the plot
    filter_cells: bool
        Filter cells based on the member threshold
    member_thresh: float
        Member threshold
    density: int
        Density of the streamlines
    
    Returns
    ------- 
    None, but plots the tensor graph

    """
    import scvelo as scv
    if attractor == None:
        velo =  adata.obsm['tensor_v_aver'].copy()
        title = 'All attractors'
    else:
        velo = adata.obsm['tensor_v'][:,:,:,attractor].copy()
        color = adata.obsm['rho'][:,attractor] 
        title = 'Attractor '+str(attractor)
        color_map = 'coolwarm'
        if filter_cells:
            cell_id_filtered = adata.obsm['rho'][:,attractor] < member_thresh
            velo[cell_id_filtered,:,:] = 0
    adata_aggr_copy = adata_aggr.copy()
    adata_copy = adata.copy()
    adata_aggr = adata_aggr_copy[:,adata_aggr.uns['gene_subset']].copy()
    gene_select = [x in adata.uns['gene_subset'] for x in adata.var_names]
    adata = adata_copy[:,gene_select].copy()
        
    if state == 'spliced':
        adata.layers['vs'] = velo[:,gene_select,1]
        scv.tl.velocity_graph(adata, vkey = 'vs', xkey = 'Ms',n_jobs = n_jobs,show_progress_bar=False)
        scv.pl.velocity_embedding_stream(adata, vkey = 'vs', basis=basis, color=color, title = title+','+'Spliced',color_map = color_map, size = size, alpha = alpha, ax = ax, show = show,**kwargs)
    if state == 'unspliced':
        adata.layers['vu'] = velo[:,gene_select,0]
        scv.tl.velocity_graph(adata, vkey = 'vu', xkey = 'Mu',n_jobs = n_jobs,show_progress_bar=False)
        scv.pl.velocity_embedding_stream(adata, vkey = 'vu',basis=basis, color=color, title = title+','+'Unspliced',color_map = color_map, size = size, alpha = alpha, ax = ax, show = show,**kwargs)
    if state == 'joint':
        print("check that the input includes aggregated object")
        #adata_aggr.layers['vj'] = np.concatenate((velo[:,gene_select,0],velo[:,gene_select,1]),axis = 1)
        scv.tl.velocity_graph(adata_aggr, vkey = 'vj', xkey = 'Ms',n_jobs = n_jobs,show_progress_bar=False)
        scv.pl.velocity_embedding_stream(adata_aggr, vkey = 'vj',basis=basis, color=color, 
        title = title+','+'Joint',color_map = color_map, size = size, 
        alpha = alpha, ax = ax, show = show, density =density,**kwargs)
        
    del adata_copy
    del adata_aggr_copy
    import gc
    gc.collect()

        
def plot_tensor(adata, adata_aggr, list_state =['joint','spliced','unspliced'], list_attractor ='all', basis = 'umap',figsize = (8,8),hspace = 0.2,wspace = 0.2, color_map = None,size = 20,alpha = 0.5, filter_cells = False, member_thresh = 0.05, density =2):
    """
    Function to plot a series of tensor graphs with assgined components
    
    Parameters
    ----------
    adata: AnnData object
    adata_aggr: AnnData object
    list_state: list
        List of states of the tensor graph, 'spliced', 'unspliced' or 'joint'
    list_attractor: list
        List of attractor index
    basis: str
        Dimensionality reduction basis for the plot
    figsize: tuple
        Size of the figure
    hspace: float
        Height space between subplots
    wspace: float
        Width space between subplots
    color_map: str
        Color map for the plot
    size: int  
        Size of the cells
    alpha: float
        Transparency of the cells
    filter_cells: bool
        Filter streamlines shown on cells based on the member threshold
    member_thresh: float
        Member threshold
    density: int
        Density of the streamlines
    
    Returns 
    -------
    None, but plots the tensor graphs
    """
    if list_attractor == 'all':
        list_attractor =[None]+list(range(len(adata.obs['attractor'].unique())))
    
    nrows = len(list_state)
    ncols = len(list_attractor)
    plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=hspace,wspace = wspace)
    fig_id = 1
    
    
    for state in list_state:
        for attractor in list_attractor:
            if state !='joint':
                basis_plot = basis+'_aggr'
            else:
                basis_plot = basis    
            ax = plt.subplot(nrows, ncols, fig_id)
            fig_id+=1
            plot_tensor_single(adata, adata_aggr, attractor = attractor, state = state, basis = basis_plot,ax = ax,show = False, member_thresh = member_thresh, filter_cells = filter_cells, size = size, alpha = alpha, density = density) 


def plot_tensor_pathway(adata,adata_aggr,pathway_name,basis,
                        ax=None,**kwargs):
    """
    Function to plot the tensor graph of the pathway
    
    Parameters
    ----------
    adata: AnnData object
    adata_aggr: AnnData object
    pathway_name: str
        Name of the pathway
    basis: str
        Dimensionality reduction basis for the plot
    
    Returns
    -------
    None, but plots the tensor graph of the pathway
    """
    pathway_set = adata.uns['pathway_select']
    subset = list(pathway_set[pathway_name])
    subset_orig = adata.uns['gene_subset'] 
    adata.uns['gene_subset'] = subset
    adata_aggr.uns['gene_subset'] = subset+[x+'_u' for x in subset]
    if ax==None:
        fig,ax=plt.subplots(1,1,figsize=(4,4))
    plot_tensor_single(adata, adata_aggr, basis = basis,
     state= 'joint',ax=ax,show=False,**kwargs)
    adata.uns['gene_subset'] = subset_orig
    adata_aggr.uns['gene_subset'] = subset_orig
    return ax


def plot_pathway(adata,figsize = (10,10),fontsize = 12,cmp='Set2',size = 20):
    """
    Function to plot the low dimensional emebedding of pathway similarity matrix
    
    Parameters
    ----------
    adata: AnnData object
    figsize: tuple
        Size of the figure
    fontsize: int
        Font size of the labels
    cmp: str    
        Color map for clusters of pathways based on similariy
    size: int
        Size of the cells
    
    Returns 
    -------
    None, but plots the low dimensional emebedding of pathway similarity matrix
    """

    pathway_select = adata.uns['pathway_select']
    # Plot the results
    umap_embedding = adata.uns['pathway_embedding']
    x = umap_embedding[:, 0]
    y = umap_embedding[:, 1]
    labels = pathway_select.keys()
    # Create the scatter plot
    fig, ax = plt.subplots(figsize = figsize)
    c_labels = adata.uns['pathway_labels']
    num_clusters = max(c_labels)+1
    cmap = plt.get_cmap(cmp, num_clusters)

    # Map the labels to colors using the colormap
    colors = cmap(c_labels/ num_clusters)

    # Plot the scatter plot with colors based on the labels
    sc = plt.scatter(x, y, c=colors, s=size)

    # Remove the square outline
    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)

    arrow_properties = dict(
        arrowstyle='->',  # Simple arrow with a head
        color='red',      # Arrow color
        linewidth=0.5,    # Arrow line width
        alpha=0.8,         # Arrow transparency     
        mutation_scale=5 
    )
    # Annotate points with labels
    texts = []
    for i, txt in enumerate(labels):
        texts.append(ax.annotate(txt, (x[i], y[i]), fontsize=fontsize))
        
    # Adjust the annotation positions to avoid overlaps
    from adjustText import adjust_text
    adjust_text(texts,arrowprops=arrow_properties)

    # Hide axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Hide axis labels (tick labels)
    plt.xlabel('Embedding 1')
    plt.ylabel('Embedding 2')

    # Create a list of patches for the legend
    patches = [mpatches.Patch(color=cmap((i - 1) / (num_clusters - 1)), label=f'Cluster {i}') for i in range(1, num_clusters + 1)]

    # Add the legend to the plot
    plt.legend(handles=patches, title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Display the plot
    plt.show()
    return fig
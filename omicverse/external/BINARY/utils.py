import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt

import scanpy as sc
import torch
from torch_geometric.data import Data
import anndata as ad
from .._pymclustr import fit_pymclustr


def Count2Binary(adata):
    """
    Convert raw count data to binary format (0 or 1) in an AnnData object.

    Parameters:
    adata : AnnData
        The input AnnData object containing the matrix of raw counts. The matrix
        should be accessible via `adata.X`. This matrix should be numerical.

    Returns:
    AnnData
        The modified AnnData object where the original raw count matrix is
        replaced with a binary matrix. In this binary matrix, any non-zero value
        in the original data is set to 1, and zero values remain zero.

    Example:
    >>> adata = AnnData(np.array([[0, 2], [3, 0], [1, 1]]))
    >>> Count2Binary(adata)
    >>> print(adata.X)
    array([[0., 1.],
           [1., 0.],
           [1., 1.]])
    """
    # Convert raw count data to binary by checking if each element is greater than 0.
    # If true, the value becomes 1 (converted to float64). If false, remains 0.
    adata.X = (adata.X > 0).astype(np.float64)
    return adata



def Multi_Refine_label(adata, radius=30, key='mclust', add_key='refine_label', obs_key='slice_id'):
    """
    Refine labels for each unique section in an AnnData object by calling the Refine_label function.

    Parameters:
    adata : AnnData
        The input AnnData object containing the data and labels to be refined.

    radius : int, optional (default=30)
        The number of neighboring cells used to refine the labels in the Refine_label function.

    key : str, optional (default='mclust')
        The key in adata.obs used to access the old labels that need to be refined.

    add_key : str, optional (default='refine_label')
        The key in adata.obs where the refined labels will be stored.

    obs_key : str, optional (default='slice_id')
        The key in adata.obs used to access unique sections.

    Returns:
    AnnData
        The modified AnnData object with refined labels.

    """
    unique_sections = adata.obs[obs_key].unique()
    for section in unique_sections:
        section_data = adata[adata.obs[obs_key] == section].copy()
        refined_labels = Refine_label(section_data, radius=radius, key=key)
        adata.obs.loc[adata.obs[obs_key] == section, add_key] = refined_labels
    return adata


def Refine_label(adata, radius=30, key='label'):
    """
    Refine cell labels based on their spatial neighbors' labels.

    Parameters:
    adata : AnnData
        The input AnnData object containing the data and labels to be refined.

    radius : int, optional (default=30)
        The number of neighboring cells used for refining the label of each cell.

    key : str, optional (default='label')
        The key in adata.obs used to access the old labels that need to be refined.

    Returns:
    list
        A list of refined labels as strings.

    """
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    import ot
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    return new_type


def find_optimal_resolution(adata, desired_clusters, algorithm="leiden", add_key=None, start=0, end=5, step=1, max_iter=50):
    """
    Find the optimal resolution using a binary search method.
    
    Parameters:
        adata: AnnData object
        desired_clusters: Desired number of clusters
        algorithm: Clustering algorithm, choose between 'leiden' or 'louvain'
        add_key: Key used to store clustering results in adata.obs
        start: Starting resolution
        end: Ending resolution
        step: Step size to adjust resolution
        max_iter: Maximum number of iterations
        
    Returns:
        best_resolution: The optimal resolution
    """
    
    if algorithm not in ["leiden", "louvain"]:
        raise ValueError("Invalid algorithm choice. Choose 'leiden' or 'louvain'")
    
    if add_key is None:
        add_key = algorithm
    
    iteration = 0
    
    while iteration < max_iter:
        midpoint = (start + end) / 2
        
        if algorithm == "leiden":
            sc.tl.leiden(adata, resolution=midpoint, key_added=add_key)
        else:
            sc.tl.louvain(adata, resolution=midpoint, key_added=add_key)
        
        num_clusters = len(np.unique(adata.obs[add_key]))
        
        if num_clusters == desired_clusters:
            print(f"The optimal resolution is: {midpoint}")
            print(f"Stored in adata.obs[{add_key}]:", add_key)
            return midpoint
        elif num_clusters < desired_clusters:
            start = midpoint
        else:
            end = midpoint
            
        step = step / 2
        iteration += 1
        
    print("The optimal resolution was not found within the given number of iterations.")
    print("Try to fine-tune the start and end values.")


def Transfer_pytorch_Data(adata):
    """
    Convert an AnnData object to a PyTorch-Geometric Data object, incorporating spatial relationships.

    Returns:
    
    A PyTorch-Geometric Data object with the edge_index representing the 
    spatial relationships between cells and x representing the expression data.

    """
    G_df = adata.uns['Spatial_Graph'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell_i'] = G_df['Cell_i'].map(cells_id_tran)
    G_df['Cell_j'] = G_df['Cell_j'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell_i'], G_df['Cell_j'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data


def Construct_Spatial_Graph(adata, use_method='KNN', cutoff = None):
    """
    Construct a spatial graph based on either K-Nearest Neighbors (KNN) or a fixed radius.

    Parameters:
    adata : AnnData
        The input AnnData object containing spatial coordinates in the obsm['spatial'] attribute.

    use_method : str (default='KNN')
        The method used to construct the spatial graph. 
        Acceptable values: 'KNN' or 'Radius'.

    cutoff : int
        For 'KNN' method, it determines the number of nearest neighbors.
        For 'Radius' method, it determines the distance threshold for considering neighbors.

    Returns:
    None
        Modifies the input adata object to include a 'Spatial_Graph' uns attribute containing the spatial graph.
    """
    
    assert(use_method in ['Radius', 'KNN'])
    print('------Constructing spatial graph...------')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if use_method == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))
    
    if use_method == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
            
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell_i', 'Cell_j', 'Distance']

    Spatial_Graph = KNN_df.copy()
    Spatial_Graph = Spatial_Graph.loc[Spatial_Graph['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Graph['Cell_i'] = Spatial_Graph['Cell_i'].map(id_cell_trans)
    Spatial_Graph['Cell_j'] = Spatial_Graph['Cell_j'].map(id_cell_trans)
    print('The graph contains %d edges, %d cells.' %(Spatial_Graph.shape[0], adata.n_obs))
    print('%.4f neighbors per cell on average.' %(Spatial_Graph.shape[0]/adata.n_obs))

    adata.uns['Spatial_Graph'] = Spatial_Graph
    
    if use_method == 'Radius':
        Stats_Spatial_Graph(adata)



def Mutil_Construct_Spatial_Graph(adata, use_method='KNN', cutoff=None, obs_key=None, use_list=None):
    """
    Construct spatial graphs for either all or a subset of values in adata.obs[obs_key].

    Parameters:
    adata : AnnData
        The input AnnData object containing spatial coordinates in the obsm['spatial'] attribute.

    use_method : str (default='KNN')
        The method used to construct the spatial graph.
        Acceptable values: 'KNN' or 'Radius'.

    cutoff : int
        For 'KNN' method, it determines the number of nearest neighbors.
        For 'Radius' method, it determines the distance threshold for considering neighbors.

    obs_key : str
        The key in adata.obs used to access unique values.

    use_list : list or None (default=None)
        A list of specific values in adata.obs[obs_key] to be processed.
        If None, all unique values in adata.obs[obs_key] will be processed.

    Returns:
    AnnData
        A modified AnnData object with a 'Spatial_Graph' uns attribute containing the combined spatial graphs.
    """
    
    if use_list is None:
        # Process all unique values in adata.obs[obs_key]
        all_values = adata.obs[obs_key].unique()
        all_nets = []  # Store individual Spatial_Graph DataFrames
        for value in all_values:
            #temp_adata = adata.loc[adata.obs[obs_key] == value, :].copy()
            temp_adata = adata[adata.obs[obs_key] == value].copy()
            Construct_Spatial_Graph(temp_adata, use_method=use_method, cutoff=cutoff)
            all_nets.append(temp_adata.uns['Spatial_Graph'])
        
        # Combine all Spatial_Graph DataFrames
        combined_net = pd.concat(all_nets)
        adata.uns['Spatial_Graph'] = combined_net

    else:
        # Process a subset of values defined by use_list
        selected_values = set(use_list)
        selected_nets = []  # Store individual Spatial_Graph DataFrames
        for value in selected_values:
            temp_adata = adata[adata.obs[obs_key] == value].copy()
            Construct_Spatial_Graph(temp_adata, use_method=use_method, cutoff=cutoff)
            selected_nets.append(temp_adata.uns['Spatial_Graph'])
        
        # Combine selected Spatial_Graph DataFrames
        combined_net = pd.concat(selected_nets)
        adata.uns['Spatial_Graph'] = combined_net
        
        # Only keep rows in adata where adata.obs[obs_key] is in use_list
        adata = adata[adata.obs[obs_key].isin(use_list)].copy()
    
    return adata

def concat_adatas(adatas, section_list):
    """
    Process and store a list of AnnData objects in a dictionary.

    Parameters:
        adatas (list): List of AnnData objects.
        section_list (list): List of section IDs.

    Returns:
        dict: Dictionary containing processed AnnData objects.
    """
    
    if len(adatas) != len(section_list):
        raise ValueError("Length of adatas and section_list should be the same.")

    adata_list = {}

    for idx, (adata, section_id) in enumerate(zip(adatas, section_list)):
        # Add slice_id to obs using section_id
        adata.obs['slice_id'] = section_id

        # Make the spot name unique
        adata.obs_names = [f"{x}_{section_id}" for x in adata.obs_names]

        # Store the processed adata in the dictionary
        adata_list[section_id] = adata.copy()

    return adata_list

def merge_adatas(adatas, slice_keys):
    """
    Merge multiple AnnData objects, capturing and merging 'var' and 'uns' information present in any given object.
    
    Parameters:
        adatas (list): List of AnnData objects
        slice_keys (list): List of keys to mark each AnnData object

    Returns:
        AnnData: Merged AnnData object
    """
    if len(adatas) != len(slice_keys):
        raise ValueError("Length of adatas and section_list should be the same.")
    
    # Add 'slice_id' to the obs of each AnnData object using the provided slice_keys
    for adata, slice_key in zip(adatas, slice_keys):
        adata.obs['slice_id'] = slice_key
    
    # Use the concatenate function to merge all AnnData objects
    merged_adata = sc.concat(adatas)
    """
    # Merge 'var' attributes
    for adata in adatas:
        if hasattr(adata, 'var') and not adata.var.empty:
            for key, value in adata.var.items():
                if key not in merged_adata.var:
                    merged_adata.var[key] = value
                    
    # Merge 'uns' attributes
    for adata in adatas:
        if hasattr(adata, 'uns') and adata.uns:
            for key, value in adata.uns.items():
                if key not in merged_adata.uns:
                    merged_adata.uns[key] = value
    """
    return merged_adata


def clean_adata(adata_raw, save_obs=[]):
    """
    Clean the input AnnData object by retaining only specific attributes.
    
    Parameters:
        adata_raw (AnnData): The original AnnData object to be cleaned.
        save_obs (list, optional): A list of keys for the obs attributes to retain.
        
    Returns:
        AnnData: Cleaned AnnData object.
    """
    
    # Extract required attributes
    X = adata_raw.X
    var_names = adata_raw.var_names
    obs_names = adata_raw.obs_names
    spatial = adata_raw.obsm['spatial']
    
    # Create a new AnnData object
    adata_cleaned = ad.AnnData(X)
    adata_cleaned.var_names = var_names
    adata_cleaned.obs_names = obs_names
    adata_cleaned.obsm['spatial'] = spatial
    
    # Retain the obs attributes specified by save_obs
    for key in save_obs:
        if key in adata_raw.obs:
            adata_cleaned.obs[key] = adata_raw.obs[key]
    
    # Preserve 'spatial' key in uns if it exists
    if 'spatial' in adata_raw.uns:
        adata_cleaned.uns['spatial'] = adata_raw.uns['spatial']
        
    return adata_cleaned


def Stats_Spatial_Graph(adata):
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Existing calculations
    Num_edge = adata.uns['Spatial_Graph']['Cell_i'].shape[0]
    Mean_edge = Num_edge/adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Graph']['Cell_i']))
    plot_df = plot_df/adata.shape[0]
    
    # Setting up the figure and axis
    fig, ax = plt.subplots(figsize=[5, 4]) # Increase figure size for better readability
    bars = ax.bar(plot_df.index, plot_df, color='skyblue') # Using a softer color
    
    # Adding data values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005, round(yval, 2), ha='center', va='bottom')
    
    # Enhancing axis labels and title
    plt.ylabel('Percentage', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Neighbors', fontsize=12, fontweight='bold')
    plt.title('Number of Neighbors (Mean=%.2f)' % Mean_edge, fontsize=14, fontweight='bold')
    
    # Adjusting aesthetics
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False) # Hide top border
    ax.spines['right'].set_visible(False) # Hide right border
    
    plt.tight_layout()
    plt.show()
"""
def Stats_Spatial_Graph(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Graph']['Cell_i'].shape[0]
    Mean_edge = Num_edge/adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Graph']['Cell_i']))
    plot_df = plot_df/adata.shape[0]
    fig, ax = plt.subplots(figsize=[3,2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)'%Mean_edge)
    ax.bar(plot_df.index, plot_df)
    
"""

def mclust_R(adata, num_cluster, add_key = 'mclust', modelNames='EEE', used_obsm='BINARY', random_seed=2020):
    """
    Cluster an AnnData object using the pure-Python pymclustR backend.
    
    The name is kept for backward compatibility; nothing here calls R any
    more. See :func:`omicverse.external._pymclustr.fit_pymclustr` for where
    the substitution is and is not numerically transparent.
    
    Parameters:
    ----------
    adata : AnnData
        The AnnData object containing the data to be clustered.
        
    num_cluster : int
        The number of clusters to be constructed.
        
    add_key : str, optional (default='mclust')
        The key added to `adata.obs` to store the cluster results.
        
    modelNames : str, optional (default='EEE')
        The parameter for the Mclust method in R specifying the model to be used for clustering. 
        For details on the possible models, refer to the mclust documentation.
        
    used_obsm : str, optional (default='BINARY')
        The key in `adata.obsm` that refers to the data matrix used for clustering.
        
    random_seed : int, optional (default=2020)
        The random seed for reproducibility.
        
    Returns:
    -------
    AnnData
        The input AnnData object with added clustering results in `adata.obs[add_key]`.
    """
    
    labels, _ = fit_pymclustr(
        adata.obsm[used_obsm],
        n_components=num_cluster,
        model_names=modelNames,
        random_state=random_seed,
    )
    adata.obs[add_key] = labels
    adata.obs[add_key] = adata.obs[add_key].astype('int')
    adata.obs[add_key] = adata.obs[add_key].astype('category')
    return adata

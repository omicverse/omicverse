import pandas as pd
import numpy as np
import os
import random
import numba
import torch
import scanpy as sc
from sklearn.neighbors import kneighbors_graph
from sklearn import metrics
import scipy.sparse as sp
import scipy.stats as stats
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from statsmodels.stats.multitest import fdrcorrection
import multiprocessing as mp
from tqdm import trange, tqdm
from .._pymclustr import fit_pymclustr



def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_adj(adata, mode = 'neighbour', k_neighbors = 7, min_distance = 150, self_loop = True):
    """
    Calculate adjacency matrix for ST data.
    
    Parameters
    ----------
    mode : str ['neighbour','distance'] (default: 'neighbour')
        The way to define neighbourhood. 
        If `mode='neighbour'`: Calculate adjacency matrix with specified number of nearest neighbors;
        If `mode='distance'`: Calculate adjacency matrix with neighbors within the specified distance.
    k_neighbors : int (default: 7)
        For `mode = 'neighbour'`, set the number of nearest neighbors if `mode='neighbour'`.
    min_distance : int (default: 150)
        For `mode = 'distance'`, set the distance of nearest neighbors if `mode='distance'`.
    self_loop : bool (default: True)
        Whether to add selfloop to the adjacency matrix.
        
    Returns
    -------
    adj : matrix of shape (n_samples, n_samples)
        Adjacency matrix where adj[i, j] is assigned the weight of edge that connects i to j.
    """
    spatial = adata.obsm["spatial"]   
    if mode == 'distance':
        assert min_distance is not None,"Please set `min_diatance` for `get_adj()`"
        adj = metrics.pairwise_distances(spatial, metric='euclidean')
        adj[adj > min_distance] = 0
        if self_loop:
            adj += np.eye(adj.shape[0])  
        adj = np.int64(adj>0)
        return adj
    
    elif mode == 'neighbour':
        assert k_neighbors is not None,"Please set `k_neighbors` for `get_adj()`"
        adj = kneighbors_graph(spatial, n_neighbors = k_neighbors, include_self = self_loop)
        return adj
        

def var_stabilize(data):
    varx = np.var(data, 1)
    meanx = np.mean(data, 1)
    fun = lambda phi, varx, meanx : meanx + phi * meanx ** 2 - varx
    target_phi = least_squares(fun, x0 = 1, args = (varx, meanx))
    return np.log(data + 1 / (2 * target_phi.x))


def minmax_normalize(data):
    maxdata = np.max(data)
    mindata = np.min(data)
    return (data - mindata)/(maxdata - mindata)


@numba.jit
def get_image_idx_1D(image_idx_2d):
    print("\nCalculating image index 1D:")
    max_value = np.max(image_idx_2d[:])
    image_idx_1d = np.ones(max_value).astype(np.int64)
    for i in range(1, max_value + 1):
        idx = np.where(image_idx_2d.T.flatten() == i)[0]
        if len(idx) > 0:
            image_idx_1d[i-1] = idx[0] + 1
    return image_idx_1d


def make_image(genecount, locates, platform = "visium", get_image_idx = False, 
               grid_size = 20, interpolation_method='linear'): # 1d ==> 2d
    """
    Convert one-dimensional gene count into two-dimensional interpolated gene image.
    
    Parameters
    ----------
    genecount : pandas.DataFrame
        The matrix of gene count expression. Rows correspond to genes and columns to cells. 
    locates : matrix of shape (n_samples, 2)
        The matrix of gene expression locates. Rows correspond to cells and columns 
        to X-coordinate  and Y-coordinateof the position. 
    platform : str ['visium','Slide-seq','Stereo-seq','osmFISH','SeqFISH'] (default: 'visium')
        Sequencing platforms for generating ST data.
    get_image_idx : bool (default: False)
        If `get_image_idx=True`, calculate `image_idx_1d`. 
        
    grid_size : int (default: 20)
        The size of grid for interpolating irregular spatial gene expression to regular grids.
    interpolation_method : str ['nearest','linear',cubic'] (default: linear)
        The method for interpolating irregular spatial gene expression to regular grids.
        Same as `scipy.interpolate.griddata`
         
    Returns
    -------
    image : ndarray
        2-D gene spatial expression images displayed in a regular pixels.
    image_idx_1d
        If `get_image_idx=True`, which could be input to function `PROST.gene_img_flatten()`.
    """ 
    if platform=="visium":
        xloc = np.round(locates[:, 0]).astype(int)
        maxx = np.max(xloc)
        minx = np.min(xloc)
        yloc = np.round(locates[:, 1]).astype(int)
        maxy = np.max(yloc)
        miny = np.min(yloc)
        
        image = np.zeros((maxy, maxx))    
        image_idx_2d = np.zeros((maxy, maxx)).astype(int)  
        for i in range(len(xloc)):
            temp_y = yloc[i]
            temp_x = xloc[i]
            temp_value = genecount[i]
            image[temp_y - 1, temp_x - 1] = temp_value
            image_idx_2d[temp_y - 1 , temp_x - 1] = i+1
            
        image = np.delete( image, range(miny - 1), 0)
        image = np.delete( image, range(minx - 1), 1)
        image_idx_2d = np.delete(image_idx_2d, range(miny - 1), 0) 
        image_idx_2d = np.delete(image_idx_2d, range(minx - 1), 1)
        image_idx_1d = np.ones(np.max(image_idx_2d[:])).astype(int)
        if get_image_idx:
            image_idx_1d = get_image_idx_1D(image_idx_2d)
                
        return image, image_idx_1d
    #--------------------------------------------------------------------------
    else:
        xloc = locates[:, 0]
        maxx, minx = np.max(xloc), np.min(xloc)

        yloc = locates[:, 1]
        maxy, miny = np.max(yloc), np.min(yloc)

        xloc_new = np.round(locates[:, 0]).astype(int)
        maxx_new, minx_new = np.max(xloc_new), np.min(xloc_new)
        
        yloc_new = np.round(locates[:, 1]).astype(int)
        maxy_new, miny_new = np.max(yloc_new), np.min(yloc_new)

        #Interpolation
        grid_x, grid_y = np.mgrid[minx_new: maxx_new+1: grid_size, miny_new: maxy_new+1: grid_size]       
        image = griddata(locates, genecount, (grid_x,grid_y), method = interpolation_method) #'nearest''linear''cubic'

        return image, image.shape
        

@numba.jit
def gene_img_flatten(I, image_idx_1d): # 2d ==> 1d
    """
    Convert two-dimensional interpolated gene image into one-dimensional gene count.
      
    Parameters
    ----------
    I
        The 2-D gene interpolated image.
    image_idx_1d
        The 2-D index for 1-D gene count. Calculated by function `PROST.make_image()` 
        with setting `get_image_idx = True`

    Returns
    -------
    One-dimensional gene count.
    """ 
    I_1d = I.T.flatten()
    output = np.zeros(image_idx_1d.shape)
    for ii in range(len(image_idx_1d)):
        idx = image_idx_1d[ii]
        output[ii] = I_1d[idx - 1]
    return output


def gau_filter_for_single_gene(gene_data, locates, platform = "visium", image_idx_1d = None):
    """
    Gaussian filter for two-dimensional gene spatial expression images displayed 
    in a regular pixels.
    
    Parameters
    ----------
    gene_data : pandas.DataFrame
        The matrix of gene count expression. Rows correspond to genes and columns to cells. 
    locates : matrix of shape (n_samples, 2)
        The matrix of gene expression locates. Rows correspond to cells and columns 
        to X-coordinate  and Y-coordinateof the position. 
    platform : str ['visium','Slide-seq','Stereo-seq','osmFISH','SeqFISH'] (default: 'visium')
        Sequencing platforms for generating ST data.
    image_idx_1d
        The 2-D index for 1-D gene count. Calculated by function `PROST.make_image()` 
        with setting `get_image_idx = True` 
        
    Returns
    -------
    One-dimensional gene count.
    """ 
    if platform=="visium":
        I,_ = make_image(gene_data, locates, platform)  
        I = gaussian_filter(I, sigma = 1, truncate = 2)
        output = gene_img_flatten(I, image_idx_1d)
    #--------------------------------------------------------------------------
    else:
        I,_ = make_image(gene_data, locates, platform) 
        I = gaussian_filter(I, sigma = 1, truncate = 2)
        output = I.flatten()
    return output


def pre_process(adata, percentage = 0.1, var_stabilization = True):
    """
    Pre-process gene count. 
    
    Parameters
    ----------
    adata : Anndata
        The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond 
        to cells and columns to genes.
    percentage : float (default: 0.1)
        For each gene, count the number of spots (cells) with expression greater 
        than 0, if number is less than a `percentage` of the total spots (cells) 
        number, remove this gene.
    var_stabilization : bool (default: True)
        Var-stabilize transformation.
        
    Returns
    -------
    gene_use
        Index of genes that `percentage` greater than threshold.
    rawcount
        Expression matrix of genes that `percentage` greater than threshold.
    """     
    if sp.issparse(adata.X):
        rawcount = adata.X.toarray().T
    else:
        rawcount = adata.X.T
        
    equal_rows = np.all(rawcount[:, 1:] == rawcount[:, :-1], axis=1)
    rawcount[equal_rows,:] = 0

    if percentage > 0:
        count_sum = np.sum(rawcount > 0, 1) 
        threshold = int(np.size(rawcount, 1) * percentage)
        gene_use = np.where(count_sum >= threshold)[0]
        print("\nFiltering genes ...")
        rawcount = rawcount[gene_use, :]
    else:
        gene_use = np.array(range(len(rawcount)))
          
    if var_stabilization:
        print("\nVariance-stabilizing transformation to each gene ...")
        rawcount = var_stabilize(rawcount) 

    return gene_use, rawcount
      

def refine_clusters(result, adj, p=0.5):
    """
    Reassigning Cluster Labels Using Spatial Domain Information.
    
    Parameters
    ----------
    result
        Clustering result to refine.
    adj
        Adjcency matrix.
    k_neighbors or min_distance
        Different way to calculate adj.
    p : float (default: 0.5)
        Rate of label changes in terms of neighbors
        
    Returns
    -------
    Check post_processed cluster label.
    """
    if sp.issparse(adj):
        adj = adj.toarray()

    pred_after = []  
    for i in tqdm(range(result.shape[0])):
        temp = list(adj[i])  
        temp_list = []
        for index, value in enumerate(temp):
            if value > 0:
                temp_list.append(index) 
        self_pred = result[i]
        neighbour_pred = []      
        for j in temp_list:
            neighbour_pred.append(result[j])
        if (neighbour_pred.count(self_pred) < (len(neighbour_pred))*p) and (neighbour_pred.count(max(set(neighbour_pred), key=neighbour_pred.count))>(len(neighbour_pred))*p):
            pred_after.append(np.argmax(np.bincount(np.array(neighbour_pred))))
        else:
            pred_after.append(self_pred)
    return np.array(pred_after)
      

def cluster_post_process(adata, platform, k_neighbors = None, min_distance = None, 
                         key_added = "pp_clustering", p = 0.5, run_times = 3):
    """
    Post_processing tool for cluster label that integrates neighborhood information.
    
    Parameters
    ----------
    adata : Anndata
        The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond 
        to cells and columns to genes.
    platform : str ['visium','Slide-seq','Stereo-seq','osmFISH','SeqFISH'] (default: 'visium')
        Sequencing platforms for generating ST data.
    k_neighbors : int (default: None)
        Same as `PROST.get_adj()`.
    min_distance : int (default: None)
        Same as `PROST.get_adj()`.
    key_added : str (default: 'pp_clustering')
        `adata.obs` key under which to add the cluster labels.
    p : float (default: 0.5)
        Rate of label changes in terms of neighbors.
    run_times : int (default: 3)
        Number of post-process runs. If the label does not change in two consecutive 
        processes, the run is also terminated.
        
    Returns
    -------
    adata.obs[key_added]
        Array of dim (number of samples) that stores the post-processed cluster 
        label for each cell.
    """
    
    print("\nPost-processing for clustering result ...")
    clutser_result = adata.obs["clustering"]
    # nonlocal PP_adj
    if platform == "visium":
        PP_adj = get_adj(adata, mode = "neighbour", k_neighbors = k_neighbors)
    else:
        PP_adj = get_adj(adata, mode = "distance", min_distance = min_distance)


    result_final = pd.DataFrame(np.zeros(clutser_result.shape[0]))
    i = 1             
    while True:        
        clutser_result = refine_clusters(clutser_result, PP_adj, p)
        print("Refining clusters, run times: {}/{}".format(i,run_times))
        result_final.loc[:, i] = clutser_result        
        if result_final.loc[:, i].equals(result_final.loc[:, i-1]) or i == run_times:
            adata.obs[key_added] = np.array(result_final.loc[:, i])
            adata.obs[key_added] = adata.obs[key_added].astype('category')
            return adata
        i += 1


def mclust(data, num_cluster, modelNames = 'EEE', random_seed = 818):
    """Cluster data with pymclustR while preserving R's 1-based labels."""
    labels, _ = fit_pymclustr(
        data,
        n_components=num_cluster,
        model_names=modelNames,
        random_state=random_seed,
    )
    return labels


def calc_I(y, w):
    """
    Calculate Moran's I.
    
    Parameters
    ----------
    y : numpy.ndarray
        Attribute vector.
    w : scipy.sparse.csr.csr_matrix
        Spatial weights.

    Returns
    -------
    The value of Moran's I.
    """ 
    y = np.array(y)

    z = y - y.mean()
    z = z.reshape(len(z),1)
    zl = np.multiply(w, z)
    num = np.multiply(zl, z.T).sum()
    z2ss = (z * z).sum()
    return y.shape[0] / w.sum() * num / z2ss


def batch_morans_I(Y, w):
    """
    Calculate Moran's I for a batch of y vectors.
    
    Parameters:
    - Y: a numpy matrix of your data, with each column being a y vector (n x m)
    - w: a spatial weight matrix (n x n)

    Returns:
    - A numpy array containing Moran's I values for each y (1 x m)
    """
    # Ensure Y and w are numpy arrays
    Y = np.array(Y)
    w = np.array(w)

    n, m = Y.shape
    
    # Calculate mean of Y along the rows
    mean_Y = np.mean(Y, axis=0, keepdims=True)
    
    # Create a deviation from mean matrix
    Z = Y - mean_Y
    
    # Calculate numerator for each y
    num = np.sum(w @ Z * Z, axis=0)
    
    # Calculate denominator for each y
    denom = np.sum(Z**2, axis=0)
    
    # Calculate Moran's I for each y
    I = n / np.sum(w) * num / denom
    
    return I


def calc_C(w, y):
    """
    Calculate Geary's C.
    
    Parameters
    ----------
    w : scipy.sparse.csr.csr_matrix
        Spatial weights.
    y : numpy.ndarray
        Attribute vector.
    
    Returns
    -------
    The value of Geary's C.
    """   
    n = y.shape[0]
    s0 = w.sum()
    z = y - y.mean()
    z = z.reshape(len(z),1)
    z2ss = (z * z).sum()
    den = z2ss * s0 * 2.0
    a, b = w.nonzero()
    num = (w.data * ((y[a] - y[b]) ** 2)).sum()
    return (n - 1) * num / den


def cal_eachGene(arglist):
    """
    Compute various statistical metrics for a given gene.

    Parameters:
    -----------
    arglist : list
        A list containing parameters required for the calculations. 
        gene_i : int
            Index of the gene being analyzed.
        exp : np.array
            Expression values of the gene.
        w : scipy.sparse matrix
            Spatial weights matrix.
        permutations : int
            Number of permutations for significance testing.
        n : int
            Total number of samples/observations.
        n2 : int
            Square of n (total number of samples).
        s1, s2, s02 : float
            Precomputed values for the spatial weights matrix.
        E : float
            Expected value of Moran's I.
        V_norm : float
            Normal variance for significance testing.

    Returns:
    --------
    list
        A list containing the computed metrics:
        - Index of the gene
        - Moran's I value
        - Geary's C value
        - p-value (normal)
        - p-value (random)
        - p-value (simulation) [if permutations are used]

    Notes:
    ------
    This function computes Moran's I, Geary's C, and significance values 
    for a given gene expression dataset. If permutations are used, 
    it also computes a simulation-based p-value for Moran's I.
    """

    # Unpack input arguments
    gene_i, exp, w, permutations, n, n2, s1, s2, s02, E, V_norm = arglist

    # Compute Moran's I and Geary's C
    _moranI = calc_I(exp, w.todense())
    _gearyC = calc_C(w, exp)
    
    # Computations for significance testing
    z = exp - exp.mean()
    z2 = z ** 2
    z4 = z ** 4
    D = (z4.sum() / n) / ((z2.sum() / n) ** 2)
    A = n * ((n2 - 3 * n + 3) * s1 - n * s2 + 3 * s02)
    B = D * ((n2 - n) * s1 - 2 * n * s2 + 6 * s02)
    C = ((n - 1) * (n - 2) * (n - 3) * s02)
    E_2 = (A - B) / C
    V_rand = E_2 - E * E

    # Z-scores for significance testing
    z_norm = (_moranI-E) / V_norm**(1 / 2.0)
    z_rand = (_moranI-E) / V_rand**(1 / 2.0)

    # Compute p-values
    _p_norm = stats.norm.sf(abs(z_norm))
    _p_rand = stats.norm.sf(abs(z_rand))

    # If permutations are used, compute simulation-based p-value
    if permutations:    
        data_perm = np.array([np.random.permutation(exp) for _ in range(permutations)])
        sim = batch_morans_I(data_perm.T, w.todense())
        sim = np.array(sim)
        larger = np.sum(sim >= _moranI)
        if (permutations - larger) < larger:
            larger = permutations - larger
        _p_sim = (larger+1) / (permutations+1)

        return [gene_i, _moranI, _gearyC, _p_norm, _p_rand, _p_sim]

    return [gene_i, _moranI, _gearyC, _p_norm, _p_rand]


def spatial_autocorrelation(adata, layer='counts',
                            k = 10, permutations = None, multiprocess = True):
    """
    Statistical test of spatial autocorrelation for each gene.
    
    Parameters
    ----------
    adata : Anndata
        The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    k : int (default: 10)
        Number of neighbors to define neighborhood.
    permutations : int (default: None)
        Number of random permutations for calculating pseudo p-values. 
        Default is 'none' to skip this step.
    multiprocess : bool (default: True)
        multiprocess
        
    Returns
    -------
    adata : Anndata
        adata.var["Moran_I"] : Moran's I
        adata.var["Geary_C"] : Moran's C
        adata.var["p_norm"] : p-value under normality assumption
        adata.var["p_rand"] : p-value under randomization assumption
        adata.var["fdr_norm"] : FDR under normality assumption
        adata.var["fdr_rand"] : FDR under randomization assumption
        
        if set `permutations`:
        adata.var["p_sim"] : p-value based on permutation test
        adata.var["fdr_sim"] : FDR based on permutation test
    """

    if layer=='X' or layer=='raw':
        if sp.issparse(adata.X):
            genes_exp = adata.X.toarray()
        else:
            genes_exp = adata.X     
    else:
        if sp.issparse(adata.layers[layer]):
            genes_exp = adata.layers[layer].toarray()
        else:
            genes_exp = adata.layers[layer]
    spatial = adata.obsm['spatial'] 
    w = kneighbors_graph(spatial, n_neighbors = k, include_self = False).toarray()

    s0 = w.sum()
    s02 = s0 * s0
    t = w + w.transpose()
    s1 = np.multiply(t, t).sum()/2.0
    s2 = (np.array(w.sum(1) + w.sum(0).transpose()) ** 2).sum()
    n = len(genes_exp)
    n2 = n * n
    E = -1.0 / (n - 1)

    v_num = n2 * s1 - n * s2 + 3 * s02
    v_den = (n - 1) * (n + 1) * s02
    V_norm = v_num / v_den - (1.0 / (n - 1)) ** 2

    w = sp.csr_matrix(w)
    N_gene = genes_exp.shape[1]

    def sel_data():  # data generater
        for gene_i in range(N_gene):
            yield [gene_i, genes_exp[:, gene_i], w, permutations, n, n2, s1, s2, s02, E, V_norm]

    if multiprocess:
        num_cores = int(mp.cpu_count() / 2)         # default core is half of total
        with mp.Pool(processes=num_cores) as pool:
            results = list(tqdm(pool.imap(cal_eachGene, sel_data()), total=N_gene))
    else:
        results = list(tqdm(map(cal_eachGene, sel_data()), total=N_gene))

    col = ['idx', 'moranI', 'gearyC', 'p_norm', 'p_rand']
    if len(results[0]) == 6:
        col.append('p_sim')

    results = pd.DataFrame(results, columns=col)

    _, fdr_norm = fdrcorrection(results.p_norm, alpha=0.05)
    _, fdr_rand = fdrcorrection(results.p_rand, alpha=0.05)

    adata.var["Moran_I"] = results.moranI.values
    adata.var["Geary_C"] = results.gearyC.values
    adata.var["p_norm"] = results.p_norm.values # p-value under normality assumption
    adata.var["p_rand"] = results.p_rand.values # p-value under randomization assumption
    adata.var["fdr_norm"] = fdr_norm
    adata.var["fdr_rand"] = fdr_rand

    if permutations:
        _, fdr_sim = fdrcorrection(results.p_sim, alpha=0.05)
        adata.var["p_sim"] = results.p_sim.values
        adata.var["fdr_sim"] = fdr_sim
    
    return adata


def preprocess_graph(adj, layer = 2, norm = 'sym', renorm = True, k = 2/3):
    """
    Preprocess adj matrix.
    """
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj  
    rowsum = np.array(adj_.sum(1)) 
    
    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
        
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized  
        
    reg = [k] * layer
    adjs = []
    for i in range(len(reg)):
        adjs.append(ident-(reg[i] * laplacian))
    return adjs


def feature_selection(adata, selected_gene_name = None, by = 'prost', n_top_genes = 3000):
    """
    A feature selection tool for ST data.
    
    Parameters
    ----------
    adata : Anndata
        The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond 
        to cells and columns to genes.
    selected_gene_name : list (default: None)
        Manually set `selected_gene_name` to select genes by name. 
        If `selected_gene_name` are set, other feature selection methods will not work.
    by : str ["prost", "scanpy"] (default: None)
        Method for feature selection. 
        If `by=="prost"`, feature will be selected by PI;
        If `by=="scanpy"`, feature will be selected by Seurat.
    n_top_genes : int (default: 3000)
        Number of features (spatially variable genes) to select.
        
    Returns
    -------
    adata that include only selected genes.
    """
    if selected_gene_name is None:
        if by == "prost":
            try:
                pi_score = adata.var["PI"]
            except:
                raise KeyError("Can not find key 'PI' in 'adata.var', please run 'PROST.cal_prost_index()' first !")
            pi_score = adata.var["PI"]
            sorted_score = pi_score.sort_values(ascending = False)
            gene_num = np.sum(sorted_score>0)
            selected_num = np.minimum(gene_num, n_top_genes)
            selected_gene_name = list(sorted_score[:selected_num].index)          
        elif by == "scanpy":
            sc.pp.highly_variable_genes(adata, n_top_genes = n_top_genes)
            adata = adata[:, adata.var.highly_variable]
            return adata
    else:    
        assert isinstance(selected_gene_name, list),"Please input the 'selected_gene_name' as type 'list' !"
    selected_gene_name = [i.upper() for i in selected_gene_name]
    raw_gene_name = [i.upper() for i in list(adata.var_names)]
    
    adata.var['space_variable_features'] = False
    for i in range(len(raw_gene_name)):
        name = raw_gene_name[i]
        if name in selected_gene_name:
            adata.var['space_variable_features'][i] = True

    #adata = adata[:, adata.var.selected]
    return adata
    

def cal_metrics_for_DLPFC(labels_pred, labels_true_path=None, print_result = True):

    # labels_true processing
    labels_true = pd.read_csv(labels_true_path)
    labels_true['ground_truth'] = labels_true['ground_truth'].str[-1]
    labels_true = labels_true.fillna(8)   
    for i in range(labels_true.shape[0]):
        temp = labels_true['ground_truth'].iloc[i]
        if temp == 'M':
            labels_true['ground_truth'].iloc[i] = 7       
    labels_true = pd.DataFrame(labels_true['ground_truth'], dtype=np.int64).values
    labels_true = labels_true[:,0]    
    #
    ARI = metrics.adjusted_rand_score(labels_true, labels_pred)
    AMI = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    NMI = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    v_measure_score = metrics.v_measure_score(labels_true, labels_pred)
    silhouette_score = metrics.silhouette_score(np.array(labels_true).reshape(-1, 1), np.array(labels_pred).reshape(-1, 1).ravel())
    if print_result:
        print('\nARI =', ARI, '\nAMI =', AMI, '\nNMI =', NMI, 
              '\nv_measure_score =', v_measure_score, '\nsilhouette_score =',silhouette_score,
              '\n==================================================================')
    return ARI, NMI, silhouette_score


def simulateH5Data(adata, rr=0.0, mu=0.0, sigma=1.0, alpha=1.0):
    """
    Simulate dropout and noise in real ST data.
    
    Parameters
    ----------
    adata : Anndata
        H5 object. The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    rr : float (default: 0.0) 
        Dropout rate.
    mu : float (default: 0.0) 
        Mean of the Gaussian noise.
    sigma : float (default: 1.0) 
        Standard deviation of the Gaussian noise.
    alpha : float (default: 1.0) 
        Scale of the Gaussian noise.

    Returns
    -------
    Gene expression with dropout and Gaussion noise.
    """
    if rr > 1 or rr < 0:
        print("Warning! Dropout rate is illegal!")
        return 0
    print("\nruning simulateH5Data...")

    import numpy as np
    from random import sample
    import scipy.sparse as sp
    import copy
    
    # get expression matrix
    issparse = 0
    if sp.issparse(adata.X):
        data_ori_dense = adata.X.toarray()
        issparse = 1
    else:
        data_ori_dense = adata.X
    
    # add Gaussian noise
    n_r, n_c = len(data_ori_dense), len(data_ori_dense[0])
    Gnoise = np.random.normal(mu, sigma, (n_r, n_c))
    data_ori_dense = data_ori_dense + alpha * Gnoise
    
    data_ori_dense = np.clip(data_ori_dense, 0, None)
    
    print(f"Adding Gaussian noise: {alpha} * gauss({mu}, {sigma})")

    # sample from non-zero
    flagXY = np.where(data_ori_dense != 0)      
    ncount = len(flagXY[0])

    # sample rr% -> 0.0
    flag = sample(range(ncount), k=int(rr * ncount))
    dropX, dropY = flagXY[0][flag], flagXY[1][flag]

    # update anndata
    data_new = data_ori_dense.copy()
    for dx, dy in zip(dropX, dropY):
        data_new[dx, dy] = 0.0
    reCount = (data_new != 0).sum()
    if issparse:
        data_new = sp.csr_matrix(data_new)
    print(f"Dropout rate = {rr}")

    # new adata, return
    newAdata = copy.deepcopy(adata)
    newAdata.X = data_new

    # Note:not Updata metadata!
    print(f"Done! Remain {100 * round(reCount/ncount, 2)}% ({reCount}/{ncount})") 
    return newAdata

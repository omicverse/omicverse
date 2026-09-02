import sys
import time
import scipy
import pandas as pd
import scanpy as sc
import numpy as np
import sklearn
from sklearn import preprocessing
import scipy.sparse as sp
from anndata import AnnData

from tqdm import tqdm
from typing import Optional
from ..._anndata_compat import as_dtype as _as_dtype


# Chromatin structure data related
def aggregate_peaks_10x(adata_atac, peak_annot_file, linkage_file, peak_dist=2000, min_corr=0.5, gene_body=False, return_dict=False, verbose=False):
    """Peak to gene aggregation.
    This function aggregates promoter and enhancer peaks to genes based on the 10X linkage file.
    Adapted from the MultiVelo.
    Parameters
    ----------
    adata_atac: :class:`~anndata.AnnData`
        ATAC anndata object which stores raw peak counts.
    peak_annot_file: `str`
        Peak annotation file from 10X CellRanger ARC.
    linkage_file: `str`
        Peak-gene linkage file from 10X CellRanger ARC. This file stores highly correlated peak-peak
        and peak-gene pair information.
    peak_dist: `int` (default: 2000)
        Maximum distance for peaks to be included for a gene.
    min_corr: `float` (default: 0.5)
        Minimum correlation for a peak to be considered as enhancer.
    gene_body: `bool` (default: `False`)
        Whether to add gene body peaks to the associated promoters.
    return_dict: `bool` (default: `False`)
        Whether to return promoter and enhancer dictionaries.
    verbose: `bool` (default: `False`)
        Whether to print number of genes with promoter peaks.
    Returns
    -------
    A new ATAC anndata object which stores gene aggreagted peak counts.
    Additionally, if `return_dict==True`:
        A dictionary which stores genes and promoter peaks.
        And a dictionary which stores genes and enhancer peaks.
    """
    promoter_dict = {}
    distal_dict = {}
    gene_body_dict = {}
    corr_dict = {}

    # read annotations
    with open(peak_annot_file) as f:
        header = next(f)
        tmp = header.split('\t')
        if len(tmp) == 4:
            cellranger_version = 1
        elif len(tmp) == 6:
            cellranger_version = 2
        else:
            raise ValueError('Peak annotation file should contain 4 columns (CellRanger ARC 1.0.0) or 5 columns (CellRanger ARC 2.0.0)')
        if verbose:
            print(f'CellRanger ARC identified as {cellranger_version}.0.0')
        if cellranger_version == 1:
            for line in f:
                tmp = line.rstrip().split('\t')
                tmp1 = tmp[0].split('_')
                peak = f'{tmp1[0]}:{tmp1[1]}-{tmp1[2]}'
                if tmp[1] != '':
                    genes = tmp[1].split(';')
                    dists = tmp[2].split(';')
                    types = tmp[3].split(';')
                    for i,gene in enumerate(genes):
                        dist = dists[i]
                        annot = types[i]
                        if annot == 'promoter':
                            if gene not in promoter_dict:
                                promoter_dict[gene] = [peak]
                            else:
                                promoter_dict[gene].append(peak)
                        elif annot == 'distal':
                            if dist == '0':
                                if gene not in gene_body_dict:
                                    gene_body_dict[gene] = [peak]
                                else:
                                    gene_body_dict[gene].append(peak)
                            else:
                                if gene not in distal_dict:
                                    distal_dict[gene] = [peak]
                                else:
                                    distal_dict[gene].append(peak)
        else:
            for line in f:
                tmp = line.rstrip().split('\t')
                peak = f'{tmp[0]}:{tmp[1]}-{tmp[2]}'
                gene = tmp[3]
                dist = tmp[4]
                annot = tmp[5]
                if annot == 'promoter':
                    if gene not in promoter_dict:
                        promoter_dict[gene] = [peak]
                    else:
                        promoter_dict[gene].append(peak)
                elif annot == 'distal':
                    if dist == '0':
                        if gene not in gene_body_dict:
                            gene_body_dict[gene] = [peak]
                        else:
                            gene_body_dict[gene].append(peak)
                    else:
                        if gene not in distal_dict:
                            distal_dict[gene] = [peak]
                        else:
                            distal_dict[gene].append(peak)

    # read linkages
    with open(linkage_file) as f:
        for line in f:
            tmp = line.rstrip().split('\t')
            if tmp[12] == "peak-peak":
                peak1 = f'{tmp[0]}:{tmp[1]}-{tmp[2]}'
                peak2 = f'{tmp[3]}:{tmp[4]}-{tmp[5]}'
                tmp2 = tmp[6].split('><')[0][1:].split(';')
                tmp3 = tmp[6].split('><')[1][:-1].split(';')
                corr = float(tmp[7])
                for t2 in tmp2:
                    gene1 = t2.split('_')
                    for t3 in tmp3:
                        gene2 = t3.split('_')
                        # one of the peaks is in promoter, peaks belong to the same gene or are close in distance
                        if ((gene1[1] == "promoter") != (gene2[1] == "promoter")) and ((gene1[0] == gene2[0]) or (float(tmp[11]) < peak_dist)):
                            if gene1[1] == "promoter":
                                gene = gene1[0]
                            else:
                                gene = gene2[0]
                            if gene in corr_dict:
                                # peak 1 is in promoter, peak 2 is not in gene body -> peak 2 is added to gene 1
                                if peak2 not in corr_dict[gene] and gene1[1] == "promoter" and (gene2[0] not in gene_body_dict or peak2 not in gene_body_dict[gene2[0]]):
                                    corr_dict[gene][0].append(peak2)
                                    corr_dict[gene][1].append(corr)
                                # peak 2 is in promoter, peak 1 is not in gene body -> peak 1 is added to gene 2
                                if peak1 not in corr_dict[gene] and gene2[1] == "promoter" and (gene1[0] not in gene_body_dict or peak1 not in gene_body_dict[gene1[0]]):
                                    corr_dict[gene][0].append(peak1)
                                    corr_dict[gene][1].append(corr)
                            else:
                                # peak 1 is in promoter, peak 2 is not in gene body -> peak 2 is added to gene 1
                                if gene1[1] == "promoter" and (gene2[0] not in gene_body_dict or peak2 not in gene_body_dict[gene2[0]]):
                                    corr_dict[gene] = [[peak2], [corr]]
                                # peak 2 is in promoter, peak 1 is not in gene body -> peak 1 is added to gene 2
                                if gene2[1] == "promoter" and (gene1[0] not in gene_body_dict or peak1 not in gene_body_dict[gene1[0]]):
                                    corr_dict[gene] = [[peak1], [corr]]
            elif tmp[12] == "peak-gene":
                peak1 = f'{tmp[0]}:{tmp[1]}-{tmp[2]}'
                tmp2 = tmp[6].split('><')[0][1:].split(';')
                gene2 = tmp[6].split('><')[1][:-1]
                corr = float(tmp[7])
                for t2 in tmp2:
                    gene1 = t2.split('_')
                    # peak 1 belongs to gene 2 or are close in distance -> peak 1 is added to gene 2
                    if ((gene1[0] == gene2) or (float(tmp[11]) < peak_dist)):
                        gene = gene1[0]
                        if gene in corr_dict:
                            if peak1 not in corr_dict[gene] and gene1[1] != "promoter" and (gene1[0] not in gene_body_dict or peak1 not in gene_body_dict[gene1[0]]):
                                corr_dict[gene][0].append(peak1)
                                corr_dict[gene][1].append(corr)
                        else:
                            if gene1[1] != "promoter" and (gene1[0] not in gene_body_dict or peak1 not in gene_body_dict[gene1[0]]):
                                corr_dict[gene] = [[peak1], [corr]]
            elif tmp[12] == "gene-peak":
                peak2 = f'{tmp[3]}:{tmp[4]}-{tmp[5]}'
                gene1 = tmp[6].split('><')[0][1:]
                tmp3 = tmp[6].split('><')[1][:-1].split(';')
                corr = float(tmp[7])
                for t3 in tmp3:
                    gene2 = t3.split('_')
                    # peak 2 belongs to gene 1 or are close in distance -> peak 2 is added to gene 1
                    if ((gene1 == gene2[0]) or (float(tmp[11]) < peak_dist)):
                        gene = gene1
                        if gene in corr_dict:
                            if peak2 not in corr_dict[gene] and gene2[1] != "promoter" and (gene2[0] not in gene_body_dict or peak2 not in gene_body_dict[gene2[0]]):
                                corr_dict[gene][0].append(peak2)
                                corr_dict[gene][1].append(corr)
                        else:
                            if gene2[1] != "promoter" and (gene2[0] not in gene_body_dict or peak2 not in gene_body_dict[gene2[0]]):
                                corr_dict[gene] = [[peak2], [corr]]

    gene_dict = promoter_dict
    enhancer_dict = {}
    promoter_genes = list(promoter_dict.keys())
    if verbose:
        print(f'Found {len(promoter_genes)} genes with promoter peaks')
    for gene in promoter_genes:
        if gene_body: # add gene-body peaks
            if gene in gene_body_dict:
                for peak in gene_body_dict[gene]:
                    if peak not in gene_dict[gene]:
                        gene_dict[gene].append(peak)
        enhancer_dict[gene] = []
        if gene in corr_dict: # add enhancer peaks
            for j, peak in enumerate(corr_dict[gene][0]):
                corr = corr_dict[gene][1][j]
                if corr > min_corr:
                    if peak not in gene_dict[gene]:
                        gene_dict[gene].append(peak)
                        enhancer_dict[gene].append(peak)

    # aggregate to genes
    adata_atac_X_copy = adata_atac.X.toarray()
    gene_mat = np.zeros((adata_atac.shape[0], len(promoter_genes)))
    var_names = adata_atac.var_names.to_numpy()
    for i, gene in tqdm(enumerate(promoter_genes), total=len(promoter_genes)):
        peaks = gene_dict[gene]
        for peak in peaks:
            if peak in var_names:
                peak_index = np.where(var_names == peak)[0][0]
                gene_mat[:,i] += adata_atac_X_copy[:,peak_index]
    gene_mat[gene_mat < 0] = 0
    gene_mat = AnnData(X=_as_dtype(sp.csr_matrix(gene_mat), np.float32))
    gene_mat.obs_names = pd.Index(list(adata_atac.obs_names))
    gene_mat.var_names = pd.Index(promoter_genes)
    gene_mat = gene_mat[:,gene_mat.X.sum(0) > 0]
    if return_dict:
        return gene_mat, promoter_dict, enhancer_dict
    else:
        return gene_mat
    
def tfidf_norm(adata_atac, scale_factor=1e4, copy=False):
    """TF-IDF normalization.

    This function normalizes counts in an AnnData object with TF-IDF.

    Parameters
    ----------
    adata_atac: :class:`~anndata.AnnData`
        ATAC anndata object.
    scale_factor: `float` (default: 1e4)
        Value to be multiplied after normalization.
    copy: `bool` (default: `False`)
        Whether to return a copy or modify `.X` directly.

    Returns
    -------
    If `copy==True`, a new ATAC anndata object which stores normalized counts
    in `.X`.
    """
    npeaks = adata_atac.X.sum(1)
    npeaks_inv = sp.csr_matrix(1.0/npeaks)
    tf = adata_atac.X.multiply(npeaks_inv)
    idf = sp.diags(np.ravel(adata_atac.X.shape[0] / adata_atac.X.sum(0))).log1p()
    if copy:
        adata_atac_copy = adata_atac.copy()
        adata_atac_copy.X = tf.dot(idf) * scale_factor
        return adata_atac_copy
    else:
        adata_atac.X = tf.dot(idf) * scale_factor

def lsi(
        adata: AnnData, n_components: int = 20, layer: Optional[str] = None,
        use_highly_variable: Optional[bool] = None, **kwargs
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    Parameters
    ----------
    adata
        Input dataset
    n_components
        Number of dimensions to use
    layer
        Layer key to extract count data froma
    use_highly_variable
        Whether to use highly variable features only, stored in
        ``adata.var['highly_variable']``. By default uses them if they
        have been determined beforehand.
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.utils.extmath.randomized_svd`
    """
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0  # Keep deterministic as the default behavior
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = adata_use.X if layer is None else adata_use.layers[layer]
    X_norm = preprocessing.normalize(X, norm="l1")
    X_norm = np.log1p(X_norm * 1e4)
    U, Sigma, VT = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)
    X_lsi = U - U.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi.astype(np.float32)

    adata.uns['lsi'] = {}
    lsi_dict = adata.uns['lsi']
    lsi_dict['U'] = U
    lsi_dict['Sigma'] = Sigma
    lsi_dict['VT'] = VT
    lsi_dict['params'] = {'n_components': n_components}


# WNN related
# pyWNN is a package developed by Dylan Kotliar (GitHub username: dylkot), published under the MIT license.
# The original release, including tutorials, can be found here: https://github.com/dylkot/pyWNN
def get_nearestneighbor(knn, neighbor=1):
    '''For each row of knn, returns the column with the lowest value
    I.e. the nearest neighbor'''
    indices = knn.indices
    indptr = knn.indptr
    data = knn.data
    nn_idx = []
    for i in range(knn.shape[0]):
        cols = indices[indptr[i]:indptr[i+1]]
        rowvals = data[indptr[i]:indptr[i+1]]
        idx = np.argsort(rowvals)
        nn_idx.append(cols[idx[neighbor-1]])
    return(np.array(nn_idx))


def compute_bw(knn_adj, embedding, n_neighbors=20):
    intersect = knn_adj.dot(knn_adj.T)
    indices = intersect.indices
    indptr = intersect.indptr
    data = intersect.data
    data = data / ((n_neighbors*2) - data)
    bandwidth = []
    for i in range(intersect.shape[0]):
        cols = indices[indptr[i]:indptr[i+1]]
        rowvals = data[indptr[i]:indptr[i+1]]
        idx = np.argsort(rowvals)
        valssort = rowvals[idx]
        numinset = len(cols)
        if numinset<n_neighbors:
            sys.exit('Fewer than 20 cells with Jacard sim > 0')
        else:
            curval = valssort[n_neighbors]
            for num in range(n_neighbors, numinset):
                if valssort[num]!=curval:
                    break
                else:
                    num+=1
            minjacinset = cols[idx][:num]
            if num <n_neighbors:
                print('shouldnt end up here')
                sys.exit(-1)
            else:
                euc_dist = ((embedding[minjacinset,:]-embedding[i,:])**2).sum(axis=1)**.5
                euc_dist_sorted = np.sort(euc_dist)[::-1]
                bandwidth.append( np.mean(euc_dist_sorted[:n_neighbors]) )
    return(np.array(bandwidth))


def compute_affinity(dist_to_predict, dist_to_nn, bw):
    affinity = dist_to_predict-dist_to_nn
    affinity[affinity<0]=0
    affinity = affinity * -1
    affinity = np.exp(affinity / (bw-dist_to_nn))
    return(affinity)


def dist_from_adj(adjacency, embed1, embed2, nndist1, nndist2):
    dist1 = sp.lil_matrix(adjacency.shape)
    dist2 = sp.lil_matrix(adjacency.shape)

    count = 0
    indices = adjacency.indices
    indptr = adjacency.indptr
    ncells = adjacency.shape[0]

    tic = time.perf_counter()
    for i in range(ncells):
        for j in range(indptr[i], indptr[i+1]):
            col = indices[j]
            a = (((embed1[i,:] - embed1[col,:])**2).sum()**.5) - nndist1[i]
            if a == 0: dist1[i,col] = np.nan
            else: dist1[i,col] = a
            b = (((embed2[i,:] - embed2[col,:])**2).sum()**.5) - nndist2[i]
            if b == 0: dist2[i,col] = np.nan
            else: dist2[i,col] = b

        if (i % 2000) == 0:
            toc = time.perf_counter()
            print('%d out of %d %.2f seconds elapsed' % (i, ncells, toc-tic))

    return(sp.csr_matrix(dist1), sp.csr_matrix(dist2))


def select_topK(dist,  n_neighbors=20):
    indices = dist.indices
    indptr = dist.indptr
    data = dist.data
    nrows = dist.shape[0]

    final_data = []
    final_col_ind = []

    tic = time.perf_counter()
    for i in range(nrows):
        cols = indices[indptr[i]:indptr[i+1]]
        rowvals = data[indptr[i]:indptr[i+1]]
        idx = np.argsort(rowvals)
        final_data.append(rowvals[idx[(-1*n_neighbors):]])
        final_col_ind.append(cols[idx[(-1*n_neighbors):]])

    final_data = np.concatenate(final_data)
    final_col_ind = np.concatenate(final_col_ind)
    final_row_ind = np.tile(np.arange(nrows), (n_neighbors, 1)).reshape(-1, order='F')

    result = sp.csr_matrix((final_data, (final_row_ind, final_col_ind)), shape=(nrows, dist.shape[1]))

    return(result)


class pyWNN():

    def __init__(self, adata, reps=['X_pca', 'X_apca'], n_neighbors=30, npcs=[20, 20], seed=0, distances=None):
        """
        Class for running weighted nearest neighbors analysis as described in Hao
        et al 2021.
        """

        self.seed = seed
        np.random.seed(seed)

        if len(reps)>2:
            sys.exit('WNN currently only implemented for 2 modalities')

        self.adata = adata.copy()
        self.reps = [r+'_norm' for r in reps]
        self.npcs = npcs
        for (i,r) in enumerate(reps):
            self.adata.obsm[self.reps[i]] = preprocessing.normalize(adata.obsm[r][:,0:npcs[i]])

        self.n_neighbors = n_neighbors
        if distances is None:
            sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, n_pcs=npcs[0], use_rep=self.reps[0], metric='euclidean', key_added='1')
            sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, n_pcs=npcs[1], use_rep=self.reps[1], metric='euclidean', key_added='2')
            sc.pp.neighbors(self.adata, n_neighbors=200, n_pcs=npcs[0], use_rep=self.reps[0], metric='euclidean', key_added='1_200')
            sc.pp.neighbors(self.adata, n_neighbors=200, n_pcs=npcs[1], use_rep=self.reps[1], metric='euclidean', key_added='2_200')
            self.distances = ['1_distances', '2_distances', '1_200_distances', '2_200_distances']
        else:
            self.distances = distances

        for d in self.distances:
            if type(self.adata.obsp[d]) is not sp.csr_matrix:
                self.adata.obsp[d] = sp.csr_matrix(self.adata.obsp[d])

        self.NNdist = []
        self.NNidx = []
        self.NNadjacency = []
        self.BWs = []

        for (i,r) in enumerate(self.reps):
            nn = get_nearestneighbor(self.adata.obsp[self.distances[i]])
            dist_to_nn = ((self.adata.obsm[r]-self.adata.obsm[r][nn, :])**2).sum(axis=1)**.5
            nn_adj = (self.adata.obsp[self.distances[i]]>0).astype(int)
            nn_adj_wdiag = nn_adj.copy()
            nn_adj_wdiag.setdiag(1)
            bw = compute_bw(nn_adj_wdiag, self.adata.obsm[r], n_neighbors=self.n_neighbors)
            self.NNidx.append(nn)
            self.NNdist.append(dist_to_nn)
            self.NNadjacency.append(nn_adj)
            self.BWs.append(bw)

        self.weights = []
        self.WNN = None

    def compute_weights(self):
        cmap = {0:1, 1:0}
        affinity_ratios = []
        self.within = []
        self.cross = []
        for (i,r) in enumerate(self.reps):
            within_predict = self.NNadjacency[i].dot(self.adata.obsm[r]) / (self.n_neighbors-1)
            cross_predict = self.NNadjacency[cmap[i]].dot(self.adata.obsm[r]) / (self.n_neighbors-1)

            within_predict_dist = ((self.adata.obsm[r] - within_predict)**2).sum(axis=1)**.5
            cross_predict_dist = ((self.adata.obsm[r] - cross_predict)**2).sum(axis=1)**.5
            within_affinity = compute_affinity(within_predict_dist, self.NNdist[i], self.BWs[i])
            cross_affinity = compute_affinity(cross_predict_dist, self.NNdist[i], self.BWs[i])
            affinity_ratios.append(within_affinity / (cross_affinity + 0.0001))
            self.within.append(within_predict_dist)
            self.cross.append(cross_predict_dist)

        self.weights.append( 1 / (1+ np.exp(affinity_ratios[1]-affinity_ratios[0])) )
        self.weights.append( 1 - self.weights[0] )


    def compute_wnn(self, adata):
        self.compute_weights()
        union_adj_mat = ((self.adata.obsp[self.distances[2]]+self.adata.obsp[self.distances[3]]) > 0).astype(int)

        full_dists = dist_from_adj(union_adj_mat, self.adata.obsm[self.reps[0]], self.adata.obsm[self.reps[1]],
                                   self.NNdist[0], self.NNdist[1])
        weighted_dist = sp.csr_matrix(union_adj_mat.shape)
        for (i,dist) in enumerate(full_dists):
            dist = sp.diags(-1 / (self.BWs[i] - self.NNdist[i]), format='csr').dot(dist)
            dist.data = np.exp(dist.data)
            ind = np.isnan(dist.data)
            dist.data[ind] = 1
            dist = sp.diags(self.weights[i]).dot(dist)
            weighted_dist += dist

        self.WNN = select_topK(weighted_dist,  n_neighbors=self.n_neighbors)
        WNNdist = self.WNN.copy()
        x = (1-WNNdist.data) / 2
        x[x<0]=0
        x[x>1]=1
        WNNdist.data = np.sqrt(x)
        self.WNNdist = WNNdist

        adata.obsp['WNN'] = self.WNN
        adata.obsp['WNN_distance'] = self.WNNdist
        adata.obsm[self.reps[0]] = self.adata.obsm[self.reps[0]]
        adata.obsm[self.reps[1]] = self.adata.obsm[self.reps[1]]
        adata.uns['WNN'] = {'connectivities_key': 'WNN',
                                     'distances_key': 'WNN_distance',
                                     'params': {'n_neighbors': self.n_neighbors,
                                      'method': 'WNN',
                                      'random_state': self.seed,
                                      'metric': 'euclidean',
                                      'use_rep': self.reps[0],
                                      'n_pcs': self.npcs[0]}}
        return(adata)
    

def gen_wnn(
    adata,
    g_basis='X_pca',
    c_basis='X_lsi', 
    k=30, 
    dims=None, 
    copy=False, 
    random_state=0):
    """Computes inputs for KNN smoothing.

    This function calculates the nn_idx and nn_dist matrices needed
    to run knn_smooth_chrom().
    Adapted from MultiVelo.

    Parameters
    ----------
    rna: :class:`~anndata.AnnData`
        RNA anndata object.
    adata_atac: :class:`~anndata.AnnData`
        ATAC anndata object.
    dims: `List[int]`
        Dimensions of data for RNA (index=0) and ATAC (index=1)
    k: `int` (default: `None`)
        Top N neighbors to extract for each cell in the connectivities matrix.

    Returns
    -------
    nn_idx: `np.darray` (default: `None`)
        KNN index matrix of size (cells, k).
    nn_dist: `np.darray` (default: `None`)
        KNN distance matrix of size (cells, k).
    """
    if copy:
        adata = adata.copy()
    if dims is None:
        dims = [adata.obsm[g_basis].shape[1], adata.obsm[c_basis].shape[1]]

    # run WNN
    WNNobj = pyWNN(adata,
                      reps=[g_basis, c_basis],
                      npcs=dims,
                      n_neighbors=k,
                      seed=random_state)
    adata = WNNobj.compute_wnn(adata)

    # get the matrix storing the distances between each cell and its neighbors
    cx = scipy.sparse.coo_matrix(adata.obsp["WNN_distance"])

    # the number of cells
    cells = adata.obsp['WNN_distance'].shape[0]

    # define the shape of our final results
    # and make the arrays that will hold the results
    new_shape = (cells, k)
    nn_dist = np.zeros(shape=new_shape)
    nn_idx = np.zeros(shape=new_shape)

    # new_col defines what column we store data in
    # our result arrays
    new_col = 0

    # loop through the distance matrices
    for i, j, v in zip(cx.row, cx.col, cx.data):

        # store the distances between neighbor cells
        nn_dist[i][new_col % k] = v

        # for each cell's row, store the row numbers of its neighbor cells
        # (1-indexing instead of 0- is a holdover from R multimodalneighbors())
        nn_idx[i][new_col % k] = int(j)

        new_col += 1

    adata.uns['WNN']['indices'] = nn_idx.astype(int)

    from umap.umap_ import fuzzy_simplicial_set
    X_tmp = sp.coo_matrix(([], ([], [])), shape=(nn_idx.shape[0], 1))
    conn, sigmas, rhos, dists = fuzzy_simplicial_set(X_tmp, nn_idx.shape[1],
                                                    None, None,
                                                    knn_indices=nn_idx,
                                                    knn_dists=nn_dist,
                                                    return_dists=True)
    conn = conn.tocsr().copy()
    conn.setdiag(1)
    adata.obsp['WNN_connectivities'] = conn

    return adata
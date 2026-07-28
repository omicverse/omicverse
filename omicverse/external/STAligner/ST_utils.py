
import pandas as pd
import numpy as np
import sklearn.neighbors
import networkx as nx
from .mnn_utils import create_dictionary_mnn
from .._pymclustr import fit_pymclustr

def match_cluster_labels(true_labels,est_labels):
    true_labels_arr = np.array(list(true_labels))
    est_labels_arr = np.array(list(est_labels))
    org_cat = list(np.sort(list(pd.unique(true_labels))))
    est_cat = list(np.sort(list(pd.unique(est_labels))))
    B = nx.Graph()
    B.add_nodes_from([i+1 for i in range(len(org_cat))], bipartite=0)
    B.add_nodes_from([-j-1 for j in range(len(est_cat))], bipartite=1)
    for i in range(len(org_cat)):
        for j in range(len(est_cat)):
            weight = np.sum((true_labels_arr==org_cat[i])* (est_labels_arr==est_cat[j]))
            B.add_edge(i+1,-j-1, weight=-weight)
    match = nx.algorithms.bipartite.matching.minimum_weight_full_matching(B)
#     match = minimum_weight_full_matching(B)
    if len(org_cat)>=len(est_cat):
        return np.array([match[-est_cat.index(c)-1]-1 for c in est_labels_arr])
    else:
        unmatched = [c for c in est_cat if not (-est_cat.index(c)-1) in match.keys()]
        l = []
        for c in est_labels_arr:
            if (-est_cat.index(c)-1) in match: 
                l.append(match[-est_cat.index(c)-1]-1)
            else:
                l.append(len(org_cat)+unmatched.index(c))
        return np.array(l)      


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None,
                    max_neigh=50, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = sklearn.neighbors.NearestNeighbors(
        n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    if model == 'KNN':
        indices = indices[:, 1:k_cutoff + 1]
        distances = distances[:, 1:k_cutoff + 1]
    if model == 'Radius':
        indices = indices[:, 1:]
        distances = distances[:, 1:]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    if model == 'Radius':
        Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    # self_loops = pd.DataFrame(zip(Spatial_Net['Cell1'].unique(), Spatial_Net['Cell1'].unique(),
    #                  [0] * len((Spatial_Net['Cell1'].unique())))) ###add self loops
    # self_loops.columns = ['Cell1', 'Cell2', 'Distance']
    # Spatial_Net = pd.concat([Spatial_Net, self_loops], axis=0)

    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net

    #########
    X = pd.DataFrame(adata.X.toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        
    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])  # self-loop
    adata.uns['adj'] = G
    

def Stats_Spatial_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge / adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df / adata.shape[0]
    fig, ax = plt.subplots(figsize=[3, 2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)' % Mean_edge)
    ax.bar(plot_df.index, plot_df)
    plt.show()


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STAGATE', random_seed=666):
    """Cluster an AnnData embedding with the pure-Python pymclustR backend.

    The name is kept for backward compatibility; nothing here calls R any
    more. See :func:`omicverse.external._pymclustr.fit_pymclustr` for where
    the substitution is and is not numerically transparent.
    """
    labels, _ = fit_pymclustr(
        adata.obsm[used_obsm],
        n_components=num_cluster,
        model_names=modelNames,
        random_state=random_seed,
    )
    adata.obs['mclust'] = labels
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

import scipy.sparse as sp
def prepare_graph_data(adj):
    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    # adj = adj + sp.eye(num_nodes)# self-loop  ##new !!
    #data =  adj.tocoo().data
    #adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()

    # adj = normalize(adj, norm="l1")

    return (adj, indices, adj.data, adj.shape)


def prune_spatial_Net(Graph_df, label):
    print('------Pruning the graph...')
    print('%d edges before pruning.' %Graph_df.shape[0])
    pro_labels_dict = dict(zip(list(label.index), label))
    Graph_df['Cell1_label'] = Graph_df['Cell1'].map(pro_labels_dict)
    Graph_df['Cell2_label'] = Graph_df['Cell2'].map(pro_labels_dict)
    Graph_df = Graph_df.loc[Graph_df['Cell1_label']==Graph_df['Cell2_label'],]
    print('%d edges after pruning.' %Graph_df.shape[0])
    return Graph_df


# https://github.com/ClayFlannigan/icp
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    # assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def ICP_align(adata_concat, adata_target, adata_ref, slice_target, slice_ref, landmark_domain, plot_align=False):
    ### find MNN pairs in the landmark domain with knn=1    
    adata_slice1 = adata_target[adata_target.obs['louvain'].isin(landmark_domain)]
    adata_slice2 = adata_ref[adata_ref.obs['louvain'].isin(landmark_domain)]
    
    
    batch_pair = adata_concat[adata_concat.obs['batch_name'].isin([slice_target, slice_ref]) & adata_concat.obs['louvain'].isin(landmark_domain)]
    mnn_dict = create_dictionary_mnn(batch_pair, use_rep='STAligner', batch_name='batch_name', k=1, iter_comb=None, verbose=0)
    adata_1 = batch_pair[batch_pair.obs['batch_name']==slice_target]
    adata_2 = batch_pair[batch_pair.obs['batch_name']==slice_ref]
    
    anchor_list = []
    positive_list = []
    for batch_pair_name in mnn_dict.keys(): 
        for anchor in mnn_dict[batch_pair_name].keys():
            positive_spot = mnn_dict[batch_pair_name][anchor][0]
            ### anchor should only in the ref slice, pos only in the target slice
            if anchor in adata_1.obs_names and positive_spot in adata_2.obs_names:                 
                anchor_list.append(anchor)
                positive_list.append(positive_spot)
                             
    batch_as_dict = dict(zip(list(adata_concat.obs_names), range(0, adata_concat.shape[0])))
    anchor_ind = list(map(lambda _: batch_as_dict[_], anchor_list))
    positive_ind = list(map(lambda _: batch_as_dict[_], positive_list))
    anchor_arr = adata_concat.obsm['STAligner'][anchor_ind, ]
    positive_arr = adata_concat.obsm['STAligner'][positive_ind, ]
    dist_list = [np.sqrt(np.sum(np.square(anchor_arr[ii, :] - positive_arr[ii, :]))) for ii in range(anchor_arr.shape[0])]
    
    
    key_points_src =  np.array(anchor_list)[dist_list < np.percentile(dist_list, 50)] ## remove remote outliers
    key_points_dst =  np.array(positive_list)[dist_list < np.percentile(dist_list, 50)]
    #print(len(anchor_list), len(key_points_src))
    
    coor_src = adata_slice1.obsm["spatial"] ## to_be_aligned
    coor_dst = adata_slice2.obsm["spatial"] ## reference_points

    ## index number
    MNN_ind_src = [list(adata_1.obs_names).index(key_points_src[ii]) for ii in range(len(key_points_src))]
    MNN_ind_dst = [list(adata_2.obs_names).index(key_points_dst[ii]) for ii in range(len(key_points_dst))]
    
    
    ####### ICP alignment
    init_pose = None
    max_iterations = 100
    tolerance = 0.001

    coor_used = coor_src ## Batch_list[1][Batch_list[1].obs['annotation']==2].obsm["spatial"]
    coor_all = adata_target.obsm["spatial"].copy()
    coor_used = np.concatenate([coor_used, np.expand_dims(np.ones(coor_used.shape[0]), axis=1)], axis=1).T    
    coor_all = np.concatenate([coor_all, np.expand_dims(np.ones(coor_all.shape[0]), axis=1)], axis=1).T    
    A = coor_src  ## to_be_aligned
    B = coor_dst  ## reference_points

    m = A.shape[1] # get number of dimensions

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)
    prev_error = 0

    for ii in range(max_iterations + 1):
        p1 = src[:m, MNN_ind_src].T
        p2 = dst[:m, MNN_ind_dst].T
        T, _, _ = best_fit_transform(src[:m, MNN_ind_src].T,
                                              dst[:m, MNN_ind_dst].T) ## compute the transformation matrix based on MNNs
        import math
        distances = np.mean([math.sqrt(((p1[kk, 0] - p2[kk, 0]) ** 2) + ((p1[kk, 1] - p2[kk, 1]) ** 2))
                             for kk in range(len(p1))])

        # update the current source
        src = np.dot(T, src)
        coor_used = np.dot(T, coor_used)
        coor_all = np.dot(T, coor_all)
        
        # check error
        mean_error = np.mean(distances)
        # print(mean_error)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
   
    aligned_points = coor_used.T    # MNNs in the landmark_domain
    aligned_points_all = coor_all.T # all points in the slice

    if plot_align:
        import matplotlib.pyplot as plt
        plt.rcParams["figure.figsize"] = (3, 3)
        fig, ax = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={'wspace': 0.5, 'hspace': 0.1})
        ax[0].scatter(adata_slice2.obsm["spatial"][:, 0], adata_slice2.obsm["spatial"][:, 1],
                    c="blue", cmap=plt.cm.binary_r, s=1)
        ax[0].set_title('Reference '+slice_ref, size=14)
        ax[1].scatter(aligned_points[:, 0], aligned_points[:, 1],
                    c="blue", cmap=plt.cm.binary_r, s=1)
        ax[1].set_title('Target '+slice_target, size=14)

        plt.axis("equal")
        # plt.axis("off")
        plt.show()
        
    #adata_target.obsm["spatial"] = aligned_points_all[:,:2] 
    return aligned_points_all[:,:2]
    
    
    
# https://scikit-learn.org/dev/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors
def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    # assert src.shape == dst.shape
    neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()



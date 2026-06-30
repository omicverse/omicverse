import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def optimal_kmeans(data, k_range):
    # Function to determine the optimal number of clusters for KMeans
    best_score = -1
    best_k = 2

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)

        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_k = k

    # Fitting KMeans with the optimal number of clusters
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    final_labels = kmeans.fit_predict(data)
    return final_labels

def compute_pathway(adata,adata_aggr,pathway,gene_num = 3,n_components=10,n_jobs=10):
    """
    Compute tensor similarities among pathways

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix
    adata_aggr: AnnData
        Aggregated data matrix
    db_name: str
        Name of the database
    gene_num: int
        Minimum number of genes in the pathway overlapped with STT multi-stable genes

    Returns
    -------
    None, but updates adata.uns with the following
    pathway_select: dict
        Selected pathways satisfying the gene_num condition
    pathway_embedding: np.ndarray
        UMAP embedding of the pathway similarities
    pathway_labels: np.ndarray
        Cluster labels of the pathway embedding
    
    """
    import scvelo as scv
    import umap

    adata_aggr_var_names = set(adata_aggr.var_names)
    pathway_select = {}
    temp = set()  # 用set加速查重
    key_gene_list = []

    # Step 1: 预筛选所有可用pathway，避免无用循环
    for key, gene_list in pathway.items():
        gene_select = [x for x in gene_list if x in adata_aggr_var_names]
        if len(gene_select) >= gene_num:
            gene_pathway = gene_select+[x+'_u' for x in gene_select]
            gene_select_tuple = tuple(sorted(gene_select))
            if gene_select_tuple not in temp:
                key_gene_list.append((key, gene_select, gene_pathway))
                temp.add(gene_select_tuple)
                pathway_select[key] = gene_select

    idx = 0
    vj_graph_array = np.zeros((len(key_gene_list), adata_aggr.n_obs**2))
    for (key, gene_select, gene_pathway) in key_gene_list:
        adata_aggr_select = adata_aggr[:,gene_pathway].copy()
        scv.tl.velocity_graph(adata_aggr_select, vkey = 'vj', xkey = 'Ms', n_jobs = n_jobs, show_progress_bar = False)
        adata_aggr.uns['vj_graph_'+key] = adata_aggr_select.uns['vj_graph']
        vj_graph_array[idx,] = adata_aggr_select.uns['vj_graph'].toarray().reshape(-1) 
        idx += 1
    
    adata.uns['pathway_select'] = pathway_select
    cor_matrix = np.corrcoef(vj_graph_array)
    # np.fill_diagonal(cor_matrix, 0)               ## 对角线赋值0

    max_dim = min(n_components, min(cor_matrix.shape))
    max_dim = 2 if max_dim < 2 else max_dim

    pca = PCA(n_components=max_dim)
    pca_embedding = pca.fit_transform(cor_matrix)
    max_dim = min(n_components, max(pca_embedding.shape))
    max_dim = 2 if max_dim < 2 else max_dim

    max_neighbors = min(15, pca_embedding.shape[0]-1)
    max_neighbors = 2 if max_neighbors < 2 else max_neighbors

    umap_embedding = None
    for i in reversed(range(2,max_dim+1)):
        try:
            umap_reducer = umap.UMAP(random_state=42, n_components=i, n_neighbors=max_neighbors, n_jobs=1)
            umap_embedding = umap_reducer.fit_transform(pca_embedding)
            print(f"umap n_components set to: {i}")
            break
        except Exception as e:
            print(f"Error processing n_components {i}: {e}")
            continue
    else:
        print(f"umap don't have a suitable n_components!")
        print(f"Skip hierarchical clustering ... !")

    if umap_embedding is not None:
        # Perform hierarchical clustering
        adata.uns['pathway_embedding'] = umap_embedding
        c_labels = optimal_kmeans(umap_embedding, range(3,min(n_components, umap_embedding.shape[0])))
        adata.uns['pathway_labels'] = c_labels

    
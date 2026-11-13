import pandas as pd
import os
import numpy as np
import anndata
from sklearn.neighbors import KNeighborsTransformer
from .._registry import register_function

#These function were created by Lisa Sikemma in scArches

@register_function(
    aliases=["KNN训练器", "weighted_knn_trainer", "knn_trainer", "细胞类型迁移训练", "跨模态训练器"],
    category="utils",
    description="Train weighted KNN classifier for cross-modal cell type annotation transfer",
    examples=[
        "# Train KNN classifier from RNA-seq data",
        "knn_model = ov.utils.weighted_knn_trainer(",
        "    train_adata=rna_adata, train_adata_emb='X_glue', n_neighbors=15)",
        "# Use with high-dimensional embeddings",
        "knn_model = ov.utils.weighted_knn_trainer(",
        "    train_adata=ref_adata, train_adata_emb='X_scvi', n_neighbors=30)",
        "# Use raw expression data",
        "knn_model = ov.utils.weighted_knn_trainer(",
        "    train_adata=ref_adata, train_adata_emb='X', n_neighbors=20)",
        "# Transfer to target modality",
        "labels, uncert = ov.utils.weighted_knn_transfer(",
        "    query_adata=atac_adata, knn_model=knn_model)"
    ],
    related=["utils.weighted_knn_transfer", "single.batch_correction", "utils.mde"]
)
def weighted_knn_trainer(train_adata:anndata.AnnData, 
                         train_adata_emb:str, 
                         n_neighbors:int=50)->KNeighborsTransformer:
    """Trains a weighted KNN classifier on ``train_adata``.

    Arguments
        train_adata: :Annotated dataset to be used to train KNN classifier with ``label_key`` as the target variable.
        train_adata_emb: Name of the obsm layer to be used for calculation of neighbors. If set to "X", anndata.X will be used
        n_neighbors: Number of nearest neighbors in KNN classifier.

    Returns
        k_neighbors_transformer: KNeighborsTransformer

    """
    print(
        f"Weighted KNN with n_neighbors = {n_neighbors} ... ",
        end="",
    )
    k_neighbors_transformer = KNeighborsTransformer(
        n_neighbors=n_neighbors,
        mode="distance",
        algorithm="brute",
        metric="euclidean",
        n_jobs=-1,
    )
    if train_adata_emb == "X":
        train_emb = train_adata.X
    elif train_adata_emb in train_adata.obsm.keys():
        train_emb = train_adata.obsm[train_adata_emb]
    else:
        raise ValueError(
            "train_adata_emb should be set to either 'X' or the name of the obsm layer to be used!"
        )
    k_neighbors_transformer.fit(train_emb)
    return k_neighbors_transformer    

@register_function(
    aliases=["KNN标签迁移", "weighted_knn_transfer", "knn_transfer", "细胞类型迁移", "跨模态标签迁移"],
    category="utils",
    description="Transfer cell type annotations across modalities using trained weighted KNN classifier",
    prerequisites={
        'functions': ['weighted_knn_trainer'],
        'optional_functions': ['batch_correction']
    },
    requires={
        'obsm': []
    },
    produces={
        'obs': ['transferred_labels']
    },
    auto_fix='escalate',
    examples=[
        "# Basic cross-modal annotation transfer",
        "labels, uncert = ov.utils.weighted_knn_transfer(",
        "    query_adata=atac_adata, query_adata_emb='X_glue',",
        "    label_keys='celltype', knn_model=knn_model,",
        "    ref_adata_obs=rna_adata.obs)",
        "# Transfer with uncertainty thresholding",
        "labels, uncert = ov.utils.weighted_knn_transfer(",
        "    query_adata=query_adata, query_adata_emb='X_scvi',",
        "    label_keys='major_celltype', knn_model=trained_knn,",
        "    ref_adata_obs=ref_adata.obs, threshold=0.7, pred_unknown=True)",
        "# Add results to query data",
        "query_adata.obs['transferred_celltype'] = labels.loc[query_adata.obs.index, 'celltype']",
        "query_adata.obs['transfer_uncertainty'] = uncert.loc[query_adata.obs.index, 'celltype']",
        "# Visualize transfer results",
        "ov.pl.embedding(query_adata, color=['transferred_celltype', 'transfer_uncertainty'])"
    ],
    related=["utils.weighted_knn_trainer", "single.pySCSA", "utils.embedding"]
)
def weighted_knn_transfer(
    query_adata:anndata.AnnData,
    query_adata_emb:str,
    ref_adata_obs:pd.DataFrame,
    label_keys:str,
    knn_model:KNeighborsTransformer,
    threshold:int=1,
    pred_unknown:bool=False,
    mode:str="package",
)->tuple:
    """Annotates ``query_adata`` cells with an input trained weighted KNN classifier.

    Arguments
        query_adata: Annotated dataset to be used to queryate KNN classifier. Embedding to be used
        query_adata_emb: Name of the obsm layer to be used for label transfer. If set to "X", query_adata.X will be used
        ref_adata_obs: obs of ref Anndata
        label_keys: Names of the columns to be used as target variables (e.g. cell_type) in ``query_adata``.
        knn_model: knn model trained on reference adata with weighted_knn_trainer function
        threshold: Threshold of uncertainty used to annotating cells as "Unknown". cells with uncertainties higher than this value will be annotated as "Unknown". Set to 1 to keep all predictions. This enables one to later on play with thresholds.
        pred_unknown: ``False`` by default. Whether to annotate any cell as "unknown" or not. If `False`, ``threshold`` will not be used and each cell will be annotated with the label which is the most common in its ``n_neighbors`` nearest cells.
        mode: Has to be one of "paper" or "package". If mode is set to "package", uncertainties will be 1 - P(pred_label), otherwise it will be 1 - P(true_label).

    Returns
        pred_labels: Dataframe with predicted labels for each cell in ``query_adata``.
        uncertainties: Dataframe with uncertainties for each cell in ``query_adata``.
    
    """
    if not type(knn_model) == KNeighborsTransformer:
        raise ValueError(
            "knn_model should be of type sklearn.neighbors._graph.KNeighborsTransformer!"
        )

    if query_adata_emb == "X":
        query_emb = query_adata.X
    elif query_adata_emb in query_adata.obsm.keys():
        query_emb = query_adata.obsm[query_adata_emb]
    else:
        raise ValueError(
            "query_adata_emb should be set to either 'X' or the name of the obsm layer to be used!"
        )
    top_k_distances, top_k_indices = knn_model.kneighbors(X=query_emb)

    stds = np.std(top_k_distances, axis=1)
    stds = (2.0 / stds) ** 2
    stds = stds.reshape(-1, 1)

    top_k_distances_tilda = np.exp(-np.true_divide(top_k_distances, stds))

    weights = top_k_distances_tilda / np.sum(
        top_k_distances_tilda, axis=1, keepdims=True
    )
    cols = ref_adata_obs.columns[ref_adata_obs.columns.str.startswith(label_keys)]
    uncertainties = pd.DataFrame(columns=cols, index=query_adata.obs_names)
    pred_labels = pd.DataFrame(columns=cols, index=query_adata.obs_names)
    for i in range(len(weights)):
        for j in cols:
            y_train_labels = ref_adata_obs[j].values
            unique_labels = np.unique(y_train_labels[top_k_indices[i]])
            best_label, best_prob = None, 0.0
            for candidate_label in unique_labels:
                candidate_prob = weights[
                    i, y_train_labels[top_k_indices[i]] == candidate_label
                ].sum()
                if best_prob < candidate_prob:
                    best_prob = candidate_prob
                    best_label = candidate_label

            if pred_unknown:
                if best_prob >= threshold:
                    pred_label = best_label
                else:
                    pred_label = "Unknown"
            else:
                pred_label = best_label

            if mode == "package":
                uncertainties.iloc[i][j] = (max(1 - best_prob, 0))

            else:
                raise Exception("Inquery Mode!")

            pred_labels.iloc[i][j] = (pred_label)

    print("finished!")

    return pred_labels, uncertainties
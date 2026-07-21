from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import scanpy as sc
import numpy as np
import pandas as pd


def logreg_epoch_score(adata, epoch):
    """
    Returns the median logreg test score averaged across species.
    Also returns dict of values.
    """
    
    unique_species = adata.obs["species"].unique()
    score = {}

    for species in unique_species:
        adata_species = adata[adata.obs["species"] == species]

        X = adata_species.X.toarray()
        Y = adata_species.obs["labels2"]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
        clf = LogisticRegression(random_state=0).fit(X_train, y_train)
        score[species] = clf.score(X_test, y_test)

    epoch_score = np.mean(list(score.values()))
    score["epoch"] = epoch
    score["score"] = epoch_score
    score["type"] = "logreg"
    return score


def median_min_distance_score(adata, epoch):
    """
    Returns the the median minimum distances between cells of one species and remaining cells averaged across species.
    """
    
    unique_species = adata.obs["species"].unique()
    score = {}

    for species in unique_species:
        adata_species = adata[adata.obs["species"] == species]
        adata_non_species = adata[adata.obs["species"] != species]

        A = adata_species.X.toarray()
        B = adata_non_species.X.toarray()
        try:
            a_choice = np.random.choice(np.arange(A.shape[0]), size=min(5000, int(A.shape[0])), replace=False)
            b_choice = np.random.choice(np.arange(B.shape[0]), size=min(5000, int(B.shape[0])), replace=False)
        except:
            print(A.shape[0], min(5000, int(A.shape[0])), np.arange(A.shape[0]))
            1/0

        A = A[a_choice, :]
        B = B[b_choice, :]

        # https://stackoverflow.com/questions/47778117/find-minimum-distances-between-groups-of-points-in-2d-fast-and-not-too-memory-c
        min_dists, min_dist_idx = cKDTree(B).query(A, 1)
        score[species] = np.median(min_dists)
    epoch_score = np.average(list(score.values()))

    score["epoch"] = epoch
    score["score"] = epoch_score
    score["type"] = "MMD"
    return score
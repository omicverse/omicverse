from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import rankdata

import scanpy as sc
import pandas as pd
from anndata import AnnData
import warnings
from builtins import int
warnings.filterwarnings('ignore')
from . import losses, miners, distances, reducers, testers
from .utils.accuracy_calculator import AccuracyCalculator

import numpy as np
from .utils import logging_presets as logging_presets
try:
    import record_keeper
except Exception:  # optional dependency, only needed for logging hooks
    record_keeper = None

from tqdm import trange

import argparse
from datetime import datetime
from pathlib import Path
import sys
from tqdm import tqdm
sys.path.append('../')


from scipy.special import logit

from sklearn.neighbors import NearestNeighbors
import networkx as nx

import seaborn as sns
import matplotlib.pyplot as plt
import argparse

import random
import tqdm


"""
Utility functions and classes for cross species
analysis

@yhr91
"""

from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import spearmanr
import plotly.express as px
import numpy as np
import pandas as pd
import warnings
import scanpy as sc
from sklearn.metrics import adjusted_mutual_info_score, rand_score
from collections import Counter
from scipy.stats import mode
import operator

# --------

class cross_species_acc():
    """
    Class for calculating cross species accuracy metrics
    """

    def __init__(self, adata, base_species='human', 
                 target_species='mouse', label_col='CL_class_coarse', 
                 metric='cosine', medoid=False, space='raw'):

        self.adata = adata
        self.base_species = base_species
        self.target_species = target_species
        self.label_col = label_col
        self.metric = metric
        self.medoid=medoid
        self.space=space
        
        # Calculate accuracy metrics
        self.calc_cross_species_label_matches()

    def find_all_species_centres(self):
        """
        Finds all species-specific centroids given an AnnData object
        """
        if self.space == 'umap':
            key = 'X_umap'
        elif self.space == 'samap':
            key = 'X_umap_samap'
        elif self.space == 'scanorama':
            key = 'X_scanorama'
        elif self.space == 'harmony':
            key = 'X_harmony'
        
        centres = {}
        centres['size'] = {}
        base_cluster_sizes = {}
        for species in self.adata.obs['species'].unique():
            centres[species] = {}
            species_set = self.adata[self.adata.obs['species']==species]
            
            for l in species_set.obs[self.label_col].unique():
                subset = species_set[species_set.obs[self.label_col] == l]
                
                # If space is not raw then use the right obsm column
                if self.space != 'raw':
                    subset_data = subset.obsm[key]
                else:
                    subset_data = subset.X.toarray()

                # Deal with exceptions
                if len(subset)<1:
                    continue
                elif len(subset)==1:
                    centres[species][l] = subset_data[0]
                
                # Use centroid or medoid
                centroid = np.mean(subset_data, 0)
                if self.medoid:
                    centres[species][l] =\
                        self.get_medoid(subset_data, centroid)
                else:
                    centres[species][l] = centroid
                    
                # This is for normalization of distances
                if species == self.base_species:
                    dist_mat = euclidean_distances(subset_data)
                    centres['size'][l] = np.max(dist_mat)
            
        return pd.DataFrame(centres).dropna()

    
    def calc_cross_species_label_matches(self):
        """
        Given Anndata object, returns:
        - matches: number of cluster centres in base species that 
        have the same cluster label in the target species as nn
        - dist: 'normalized' distance between cluster centre of base 
        species and the cluster centre with the same label in target 
        species
        
        TODO: This is not generalized to more than 2 species
        """
        warnings.filterwarnings("ignore")
        
        centres = self.find_all_species_centres()
        dist = 0
        norm_dist = 0
        matches = 0
        matches_names = []
        matches_names_all = []
        target_centres = np.vstack(
                   centres.loc[:,self.target_species].values)
        
        for idx, ctype in enumerate(centres.index):
            base = centres.loc[ctype, self.base_species]
            base_targets = np.vstack([base, target_centres])

            if self.metric=='cosine':
                distances = cosine_distances(base_targets)[0][1:]
                
            pred_match = np.argmin(distances)
            if  pred_match == idx:
                matches += 1
                matches_names.append(ctype)
            
            matches_names_all.append((ctype,
                         centres.index[pred_match]))
            dist += distances[idx]
            norm_dist += distances[idx]/centres.loc[ctype, 'size']

        self.cross_species_label_dist = dist
        self.cross_species_label_norm_dist = norm_dist
        self.cross_species_label_matches = matches
        self.cross_species_label_matches_names = matches_names
        self.cross_species_label_matches_names_all = matches_names_all
        
        warnings.filterwarnings("always")
     
    
    def get_medoid(self, data, centroid):
        dists = euclidean_distances(np.vstack([centroid,data]))[0]
        return data[np.argsort(dists)[1]-1]
        
        
# --------


class embedding_CL_comparison():
    """
    Class for comparing embedding with cell ontology
    """

    def __init__(self, adata, label_col='CL_class_coarse', CL_ID_col='CL_ID_coarse',
                 metric='cosine', features='raw'):

        warnings.filterwarnings("ignore")
        self.adata = adata
        self.label_col = label_col
        self.CL_ID_col = CL_ID_col
        self.metric = metric
        self.features = features
        self.labels = self.adata.obs[self.label_col].unique()
        self.centres = []
        self.centres_ranked = []
        self.CL_centres_ranked = []

        # Get centres, nns and ranks
        self.get_centre_ranks()
        self.get_CL_ranks()

        # Calculate metrics
        self.spearman_corr = {}
        self.hits_at_k = {}
        for id_ in self.labels:         
            self.spearman_corr[id_] = spearmanr(self.CL_centres_ranked[id_], 
                self.centres_ranked[id_])[0]
            self.hits_at_k[id_] = self.get_hits_topk(self.CL_centres_ranked[id_], 
                self.centres_ranked[id_])
            
        warnings.filterwarnings("always")

        
    # Implement cluster centroid

    def find_centre(self, cluster, medioid=False):
        """
        Find cluster centre: either centroid or medioid
        """
        if medioid:
            dist = euclidean_distances(cluster)
            medioid = np.argmin(dist.sum(0))
            return cluster[medioid].toarray()

        else:
            return np.mean(cluster,0)
    
    
    def get_outlier_idx(self, CL_centres, centres, k=10):
        """
        Get top or bottom ranked nn
        """
        outliers = []
        for i, pair in enumerate(list(zip(CL_centres, centres))):
            if pair[0] < k or pair[1] < k:
                outliers.append(i)
        return outliers

    
    def get_hits_topk(self, CL_centres, centres, k=10):
        """
        Get numbers of matches within top k
        """
        return len(set(CL_centres[:k]).intersection(set(centres[:k])))
        

    def get_centre_ranks(self):
        """
        Get nn ranks for cluster centres
        """
        for cell_type in self.labels:
            if self.features=='raw':
                self.centres.append(self.find_centre(
                    self.adata[self.adata.obs[self.label_col] == cell_type].X))
        self.centres = np.vstack(self.centres)
                
        if self.metric=='euclidean':
            centres_dist = euclidean_distances(self.centres)

        if self.metric=='cosine':
            centres_dist = cosine_distances(self.centres)

        self.centres_ranked = {k:v for k,v in zip(
            self.labels, np.argsort(centres_dist))}
        
    def get_CL_ranks(self):
        """
        Get nn ranks for cell ontology cluster centres
        """
        all_CL_distances = pd.read_csv('/dfs/project/cross-species/data/lung/shared/CL_similarity_RW.csv',
                                       index_col=0)

        CL_sim_matrix = all_CL_distances.loc[self.adata.obs[self.CL_ID_col].unique(),
                                             self.adata.obs[self.CL_ID_col].unique()]
        
        ID_dict = self.adata.obs.set_index(self.label_col).to_dict()['CL_ID_coarse']
        inv_ID_dict = {v: k for k, v in ID_dict.items()}

        self.CL_centres_ranked = {inv_ID_dict[k]:v for k,v in zip(self.adata.obs['CL_ID_coarse'].unique(),
                                                           np.argsort(-CL_sim_matrix.values))}

    
    def plot_rank_scatter(self):
        """
        Create rank scatter plot between embedding nn and CL nn
        """
        fig, axs = plt.subplots(5, 5, sharex=True, sharey=True, figsize=[20,15])
        it = 0
        spearman_corr = {}
        hits_at_k = {}
        outlier=False

        for i in range(5):
            for j in range(5):
                if it == len(self.labels):
                    break

                id_ = self.labels[it]
                if outlier:
                    outlier_idx = get_outlier_idx(self.CL_centres_ranked[id_], 
                        self.centres_ranked[id_])
                    axs[i, j].scatter(self.CL_centres_ranked[id_][outlier_idx], 
                    self.centres_ranked[id_][outlier_idx])
                else:
                    axs[i, j].scatter(self.CL_centres_ranked[id_], 
                    self.centres_ranked[id_])
                axs[i, j].set_title(id_)
                axs[i, j].plot([0,32],[0,32], 'k')   
                it += 1
                
    def plot_hits_at_k(self):
        plot_df = pd.DataFrame.from_dict(self.hits_at_k, orient='index')
        plot_df = plot_df.rename(columns={0:'Value'})
        plot_df = plot_df.sort_values('Value')

        plt.figure(figsize=[8,12])
        plt.barh(plot_df.index, plot_df['Value'])
        plt.ylabel('Cell Type')
        plt.xlabel('Hits @ k')
        plt.title('Hits @ k (Embedding space compared to Cell Ontology)')
        plt.xlim([0,10])
        
        
    def plot_spearman(self):
        plot_df = pd.DataFrame.from_dict(self.spearman_corr, orient='index')
        plot_df = plot_df.rename(columns={0:'Value'})
        plot_df = plot_df.sort_values('Value')

        plt.figure(figsize=[8,12])
        plt.barh(plot_df.index, plot_df['Value'])
        plt.ylabel('Cell Type')
        plt.xlabel('Spearman Correlation')
        plt.title('Spearman Correlation (Embedding space compared to Cell Ontology)')
        plt.xlim([-1,1])
          
        
# --------

## KNN analysis per cell
## TODO integrate these functions into cross_species_acc class

def get_knn_label(cell_names, adata, col):
    """
    Returns majority class labels of nearest neighbors. 
    Will return random label in case of tie
    """
    
    return adata[cell_names].obs[col].value_counts().index[0]


def cross_species_knn_all(adata, k=1, species='human', space='raw',
                          col = 'cell_type', metric='euclidean',
                          verbose = False, consider_same_species=False):
    """Runs cross species k nearest neighbor on all cells
    """

    # Create distance matrix
    if space == 'raw':
        X = adata.X
    elif space == 'umap':
        X = adata.obsm['X_umap']
    elif space == 'samap':
        X = adata.obsm['X_umap_samap']
    elif space == 'scanorama':
        X = adata.obsm['X_scanorama']
    elif space == 'harmony':
        X = adata.obsm['X_harmony']
    
    # Slow step
    if metric == 'euclidean':
        dist_mat = euclidean_distances(X)
    elif metric == 'cosine':
        dist_mat = cosine_distances(X)
        
    if consider_same_species:
        # Get indices for species and nonspecies cells
        species_idx = np.where(adata.obs['species']==species)[0]
        adata.obs['temp_label'] = adata.obs['species'].astype(str) +\
                            '_' + adata.obs[col].astype(str)

        nns = []
        for idx in species_idx:
            curr_temp_label = adata.obs['temp_label'][idx]
            row = dist_mat[idx,:]
            possible_nbrs = np.where(adata.obs['temp_label'] != curr_temp_label)[0]
            
            row = row[possible_nbrs]
            nns.append(possible_nbrs[np.argpartition(row, k)[:k]])
            
        nbrs = [(adata.obs[col][x], Mode(adata.obs[col][y].astype('str').values)) 
                        for x,y in zip(species_idx, nns)]
    
    else:
        # Get indices for species and nonspecies cells
        species_idx = np.where(adata.obs['species']==species)[0]
        nonspecies_idx = np.where(adata.obs['species'] != species)[0]

        # Slow step
        reduced_dist_mat = dist_mat[species_idx,:][:,nonspecies_idx]
        
        nns = [list(nonspecies_idx[y]) 
               for y in np.argpartition(reduced_dist_mat, k)[:,:k]]

        nbrs = [(adata.obs[col][x], Mode(adata.obs[col][y].astype('str').values)) 
                        for x,y in zip(species_idx, nns)]
    
    return nbrs


def cluster_knn(cluster_knn_df, label):
    """
    Given majority k nearest cross species neigbhor class for each cell, 
    identifies the k nearest neighbors for the given cluster
    """
    
    list_ = list(cluster_knn_df[cluster_knn_df['Source_Cell']==label].
                value_counts().items())
    x = pd.DataFrame([(s[0][1],s[1]) for s in list_])
    x['Source_Cluster'] = list_[0][0][0]
    x = x.rename(columns={0:'Cross_Species_KNN_Label', 1:'Score'})
    x['Score'] = x['Score']/x['Score'].sum()
    return x


def cluster_knn_all(all_nbrs):
    """
    Given majority k nearest cross species neigbhor class for each cell, 
    identifies the k nearest neighbors for all clusters
    """
    cluster_knn_df = pd.DataFrame(all_nbrs)
    cluster_knn_df = cluster_knn_df.rename(columns={0:'Source_Cell',1:'Cross_Species_KNN'})

    return [cluster_knn(cluster_knn_df, c) 
         for c in cluster_knn_df['Source_Cell'].unique()]   

def plot_cluster_knn_bar(df, source='human', other='mouse'):
    """
    Creates a stacked bar plot to identify majority k nearest neighbors for
    a given cluster
    """
    plt.figure(figsize = (6,17))
    bars = defaultdict(int)
    colors = defaultdict(int)
    tick = -1; 
    tick_pos = {}
    colors_list = ['','r','g','b','y','k','purple','g','b','y','k','g','b','y',
                                                       'k','g','b','y','k',
                                                       'k','g','b','y','k',
                                                       'k','g','b','y','k']
    df = df.sort_values('Score', ascending=False)

    for i in df.iterrows():
        color = 'k'
        x = i[1]['Source_Cluster']
        y = i[1]['Score']        

        if i[1]['Cross_Species_KNN_Label'] == i[1]['Source_Cluster']:
            color= 'w'
        left = bars[x]
        bars[x] = bars[x] + y
        colors[x] = colors[x] + 1
        if colors_list[colors[x]] == 'r':    
            tick = tick+1
            tick_pos[x] = tick
            if bars[x]>=0.50:
                plt.text(0.1,tick-0.2,i[1]['Cross_Species_KNN_Label'], color=color)
        plt.barh(tick_pos[x], y, left=left, color=colors_list[colors[x]], alpha=0.5)
        
    keys = tick_pos.keys()
    vals = [tick_pos[k] for k in keys]
    plt.yticks(vals, keys)
    plt.ylabel(source)
    plt.xlabel('Percentage of cells with cross-species KNN class')
    
    return 

## ---------------------------------
## Alignment scores
## -----------------------------------

def alignment_score(fname, col='cell_type', space='raw', k=1,
                    species='human', consider_same_species=False):
    adata = sc.read_h5ad(fname)
    if space=='umap':
        sc.pp.pca(adata, n_comps=50)
        sc.pp.neighbors(adata, n_neighbors=15)
        sc.tl.umap(adata)
    all_nbrs = cross_species_knn_all(adata, col=col, metric='cosine', space=space, k=k,
                                     species=species, consider_same_species=consider_same_species)
    all_cluster_nbrs  = [i for i in cluster_knn_all(all_nbrs)]
    all_cluster_nbrs = pd.concat(all_cluster_nbrs).reset_index(drop=True) 
    return all_cluster_nbrs

def compare_matches(alignments, true_alignments):
    aligns = alignments.merge(true_alignments, on=['Source_Cluster', 'Cross_Species_KNN_Label'])
    #aligns = aligns.merge(true_alignments, on=['Source_Cluster'], how='outer').fillna(0)
    
    return aligns

def score_matches(alignments, true_alignments, thresh=0.5, ret_matches=False):
    matches = compare_matches(alignments, true_alignments)
    if ret_matches:
        return matches[matches['Score']>thresh]
    else:
        return sum(matches['Score']>thresh)
    return

def create_comparison_plot_df(all_cluster_nbrs1, all_cluster_nbrs2, true_map):
    
    knn_scores_1 = compare_matches(all_cluster_nbrs1, true_map)
    knn_scores_2 = compare_matches(all_cluster_nbrs2, true_map)

    plot_df = knn_scores_1.merge(knn_scores_2, on='Source_Cluster') 
    return plot_df

def get_comparison_plot(plot_df, bars=2, labels_=['Method1', 'Method2']):
    plt.figure(figsize=[20,5])
    ax=plt.gca()

    labels = plot_df['Source_Cluster'].values
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    #fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, plot_df['Score_x'].values, width, label=labels_[0])
    if bars == 2:
        rects2 = ax.bar(x + width/2, plot_df['Score_y'].values, width, label=labels_[1])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('% of nearest neighbors of\n correct cross species label')
    ax.set_title('Cross species cell type alignment')
    plt.xticks(x, labels, rotation='vertical')
    ax.set_xticklabels(labels)
    ax.legend() 

    
def get_cell_alignment(align_df, true_df, count_df):
    align_df['Source_Cluster'] = align_df['Source_Cluster'].astype('str')
    count_df['Source_Cluster'] = count_df['Source_Cluster'].astype('str')
    true_df['Source_Cluster'] = true_df['Source_Cluster'].astype('str')
    align_df['Cross_Species_KNN_Label'] = align_df['Cross_Species_KNN_Label'].astype('str')                  
    true_df['Cross_Species_KNN_Label'] = true_df['Cross_Species_KNN_Label'].astype('str')
    
    df = align_df.merge(true_df, on=['Source_Cluster', 'Cross_Species_KNN_Label'])
    df = df.merge(count_df, on='Source_Cluster', how='outer').fillna(0)
    return sum((df['Score']*df['count'])/sum(df['count']))


def get_alignment_metrics(fname, out_label = 'labels2', orig_label='CL_class_coarse',
                          space='raw', species=['human','mouse'], k=1,
                 true_labels_path=None, ret_matches = False, consider_same_species=False):
    """
    Function for computing evaluation metrics for embedding
    
    Outputs:
    - species1_nn: Number of cross-species label matches (species 1)
    - species2_nn: Number of cross-species label matches (species 2)
    - union_nn: Number of cross-species label matches in either species 
    - mutual_nn: Number of cross-species label matches in both species 
    - cell_score1: Percentage of cells in species 1 with cross species nn of correct label
    - cell_score2: Percentage of cells in species 2 with cross species nn of correct label
    - cell_score_combine: Percentage of cells in both species with cross species nn of correct label
    - centroid_matches_species1: Number of species 1 centroids that are nn with correct species 2 centroid
    - centroid_matches_species2: Number of species 2 centroids that are nn with correct species 1 centroid
    - centroid_matches_union: Union of centroid lists
    - medoid_matches_species1: Number of species 1 medoids that are nn with correct species 2 medoid
    - medoid_matches_species2: Number of species 2 medoids that are nn with correct species 1 medoid
    - medoid_matches_union: Union of medoid lists
    """
    
    # TODO: This is a very ugly function that needs to be made into a class alongwith
    # the functions above it
        
    # Get cross-species only alignments
    alignments = []
    print('Finding nns for species 1')
    alignments.append(alignment_score(fname, out_label, space=space, species=species[0], k=k,
                                     consider_same_species=consider_same_species))
    print('Finding nns for species 2')
    alignments.append(alignment_score(fname, out_label, space=space, species=species[1], k=k,
                                     consider_same_species=consider_same_species))
    
    # Get true labels
    if true_labels_path is None:
        if orig_label == 'CL_class_coarse':
            true_labels_path = '/dfs/project/cross-species/data/lung/shared/true_CL_class_coarse.csv'
        elif orig_label == 'cell_type':
            true_labels_path = '/dfs/project/cross-species/data/lung/shared/true_cell_type.csv'
        else:
            print("ERROR: True labels unavailable for this column!, Please set manually")
            return
    
    true_labels = pd.read_csv(true_labels_path, index_col=0)
    cols = []
    results = {}
    cols.append([c for c in true_labels.columns if species[0] in c][0])
    cols.append([c for c in true_labels.columns if species[1] in c][0])

    true_dfs = []
    true_dfs.append(true_labels.rename(columns={
        cols[0]:'Source_Cluster', cols[1]:'Cross_Species_KNN_Label'}))
    true_dfs.append(true_labels.rename(columns={
        cols[1]:'Source_Cluster', cols[0]:'Cross_Species_KNN_Label'}))

    # Score matches
    matches = []
    matches.append(score_matches(alignments[0], 
                    true_dfs[0], thresh=0.5, ret_matches=True))
    matches.append(score_matches(alignments[1], 
                    true_dfs[1], thresh=0.5, ret_matches=True))
        
    for m in matches:
        m = m.rename(columns = {'Cross_Species_KNN_Label_x':'Cross_Species_KNN_Label'})
        m = m.loc[:,['Score', 'Source_Cluster', 'Cross_Species_KNN_Label']]
    results['species1_nn'] = len(matches[0])
    results['species2_nn'] = len(matches[1])

    # Combine matches
    all_matches = matches[0].merge(matches[1], 
                         left_on=['Source_Cluster', 'Cross_Species_KNN_Label'],
                         right_on=['Cross_Species_KNN_Label', 'Source_Cluster'], how='outer')
    results['union_nn'] = len(all_matches)
    results['mutual_nn'] = len(matches[0].merge(matches[1], 
                         left_on=['Source_Cluster', 'Cross_Species_KNN_Label'],
                         right_on=['Cross_Species_KNN_Label', 'Source_Cluster'], how='inner'))

    # Get per-cell alignment scores
    adata = sc.read_h5ad(fname)
    adata = adata[adata.obs['species'].isin(species)]
    ratio1 = sum(adata.obs['species']==species[0])/len(adata)
    ratio2 = sum(adata.obs['species']==species[1])/len(adata)

    adata1 = adata[adata.obs['species']==species[0]]
    adata2 = adata[adata.obs['species']==species[1]]
    count_dfs = []
    count_dfs.append(pd.DataFrame(adata1.obs[out_label].value_counts()).reset_index().rename(
                    columns={'index':'Source_Cluster', out_label:'count'}))
    count_dfs.append(pd.DataFrame(adata2.obs[out_label].value_counts()).reset_index().rename(
                    columns={'index':'Source_Cluster', out_label:'count'}))

    cell_scores = []
    cell_scores.append(get_cell_alignment(alignments[0], true_dfs[0], count_dfs[0]))
    cell_scores.append(get_cell_alignment(alignments[1], true_dfs[1], count_dfs[1]))
    results['cell_score1'] = cell_scores[0]
    results['cell_score2'] = cell_scores[1]
    results['cell_score_combine'] = ratio1*cell_scores[0] + ratio2*cell_scores[1]

    # Get centroid nn score:
    for centre,flag in [('centroid', False), ('medoid', True)]:
        c_nn1 = cross_species_acc(adata, base_species=species[0], target_species=species[1], 
                          label_col=out_label, medoid=flag, space=space)
        c_nn2 = cross_species_acc(adata, base_species=species[1], target_species=species[0], 
                          label_col=out_label, medoid=flag, space=space)
       
    
        results[centre+'_matches_species1'] = c_nn1.cross_species_label_matches
        results[centre+'_matches_species2'] = c_nn2.cross_species_label_matches
        results[centre+'_matches_union'] = len(set(c_nn1.cross_species_label_matches_names).union(
                                            set(c_nn2.cross_species_label_matches_names)))
    
    if ret_matches == True:
        return (results, matches, alignments)
    
    else:
        return results
    
    
def get_louvain_metrics(fname, label='cell_type'):
    # Compute adjusted rand index for measuring label alignment across species using ground truth information
    
    adata = sc.read_h5ad(fname)
    
    try:
        sc.pp.pca(adata, n_comps=50)
        sc.pp.neighbors(adata, n_neighbors=15)
    except:
        pass
    
    if '_' not in adata.obs[label].values[0]:
        adata.obs[label].str.cat(adata.obs["species"], sep="_")
    
    metrics = {}
    for resolution in [10, 5, 2, 1, 0.8, 0.5, 0.4, 0.2, 0.1, 0.01, 0.001, 0.0001]:
        print('Calculating for resolution: ', str(resolution))
        sc.tl.louvain(adata, resolution=resolution)

        true_clusters = pd.read_csv('/dfs/project/cross-species/data/lung/shared/true_cell_type_clusters.csv', index_col=0)
        true_clusters = true_clusters.merge(adata.obs, left_on='cell_type', right_on=label)
        
        metrics[resolution] = {
            #'ARI':adjusted_rand_score(true_clusters['cluster'], true_clusters['louvain'].astype('int')),
            'RI':rand_score(true_clusters['cluster'], true_clusters['louvain'].astype('int')),
            'AMI': adjusted_mutual_info_score(true_clusters['cluster'], true_clusters['louvain'].astype('int'))
        }

    return metrics


# Maria's cell type reannotation function
def reannotate(adata, source='human', target='mouse', label='cell_type'):
    for resolution in [2, 1, 0.8, 0.6, 0.4, 0.2, 0.1]:
        sc.tl.louvain(adata, resolution)
        louvain_clusters = set(adata.obs['louvain'])

        reannotated = {}
        for c in louvain_clusters:
            current_cluster = adata[adata.obs['louvain']==c]
            if len(set(current_cluster.obs['species']))==2:
                cluster_source = current_cluster[current_cluster.obs['species']==source]
                cluster_target = current_cluster[current_cluster.obs['species']==target]
                c = Counter(cluster_source.obs[label])
                major_cell_type = max(c.items(), key=operator.itemgetter(1))[0]      
                for c in cluster_target.obs_names:
                    if c not in reannotated:
                        reannotated[c] = major_cell_type
    adata_source = adata[adata.obs['species']==source]
    tmp = dict(zip(adata_source.obs_names, adata_source.obs[label]))
    reannotated = {**reannotated, **tmp}
    adata.obs['reannotated_'+source] = [reannotated[c] if c in reannotated else 'None' 
                                  for c in adata.obs_names]
    
def get_reannotation_metrics(fname, label='cell_type', source='human', target='mouse', true_labels_path='/dfs/project/cross-species/data/lung/shared/true_cell_type.csv'):
    # This is current specific to mouse reannotation
    
    adata = sc.read_h5ad(fname)
    sc.pp.neighbors(adata)
    
    if '_' in adata.obs[label].values[0]:
        label = 'labels2'
    
    reannotate(adata, source=source, target=target, label=label)
    m = adata[adata.obs['species']==target]
    
    true_labels = pd.read_csv(true_labels_path, index_col=0)
    results_df = true_labels.merge(m.obs, left_on=source+'_cell_type', right_on='reannotated_'+source, how="right")
    return np.mean(results_df[target+'_cell_type'] == results_df[label])
## ---------------------------------
## General helper functions
## -----------------------------------

def plotly_scatter(adata, embed = 'X_umap', label= 'cell_type', 
                   hover_cols = ['cell_type', 'species']):
    plot_df = pd.DataFrame(adata.obsm[embed])
    plot_df[label] = adata.obs[label].values
    for c in hover_cols:
        plot_df[c] = adata.obs[c].values
    
    fig = px.scatter(plot_df, x=0, y=1, 
                     hover_name=label,
                     color = label,
                     hover_data=hover_cols)

    fig.show()
    
def Mode(arr):
    # Wrapper for mode
    return mode(arr)[0][0]

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
#from sklearnex import patch_sklearn
#patch_sklearn()
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
#from tap import Tap



def classify_cell_types(embedding_data: sc.AnnData,
                        cell_type_mapping: pd.DataFrame,
                        cell_type_column: str = 'labels2',
                        species_column: str = 'species',
                        species_1: str = 'human',
                        species_2: str = 'mouse',
                        C: float = 0.005,
                        verbose: bool = False,
                        save_path: Optional[Path] = None) -> Dict[str, float]:
    """Trains a classifier to predict cell types across species using embeddings output from metric learning.

    :param embedding_data: AnnData containing embeddings and cell type labels output from metric learning.
    :param cell_type_mapping: DataFrame containing a mapping between cell types of the two species.
    :param cell_type_column: Name of the column containing the cell type label.
    :param species_column: Name of the column containing the species label.
    :param species_1: Name of the first species.
    :param species_2: Name of the second species.
    :param C: Logistic regression regularization parameter.
    :param verbose: Whether to print out the results.
    :param save_path: Path to .pdf file where confusion matrix for cross-species predictions will be saved.
    :return: A dictionary containing results of the logistic regression and dummy models trained on species 1
             and tested on both species 1 and species 2.
    """
    # Process cell type mapping
    species_1_to_2_cell_type_map = dict(zip(cell_type_mapping[f'{species_1}_cell_type'],
                                            cell_type_mapping[f'{species_2}_cell_type']))

    # Separate into two species
    species_1_embedding_data = embedding_data[embedding_data.obs[species_column] == species_1]
    species_1_embeddings = species_1_embedding_data.X
    species_1_labels = species_1_embedding_data.obs[cell_type_column]

    species_2_embedding_data = embedding_data[embedding_data.obs[species_column] == species_2]
    species_2_embeddings = species_2_embedding_data.X
    species_2_labels = species_2_embedding_data.obs[cell_type_column]

    # Split species 1 into train and test
    species_1_embeddings_train, species_1_embeddings_test, species_1_labels_train, species_1_labels_test = \
        train_test_split(species_1_embeddings, species_1_labels, test_size=0.2, random_state=0)

    # Train majority baseline classifier on species 1
    dummy_model = DummyClassifier(strategy='prior')
    dummy_model.fit(species_1_embeddings_train, species_1_labels_train)

    # Train logistic regression classifier on species 1
    logreg_model = LogisticRegression(C=C, random_state=0, n_jobs=-1, multi_class='multinomial')
    logreg_model.fit(species_1_embeddings_train, species_1_labels_train)

    # Test dummy classifier on species 1
    species_1_dummy_preds_train = dummy_model.predict(species_1_embeddings_train)
    species_1_dummy_train_accuracy = accuracy_score(species_1_labels_train, species_1_dummy_preds_train)

    species_1_dummy_preds_test = dummy_model.predict(species_1_embeddings_test)
    species_1_dummy_test_accuracy = accuracy_score(species_1_labels_test, species_1_dummy_preds_test)

    # Test logistic regression on species 1
    species_1_logreg_preds_train = logreg_model.predict(species_1_embeddings_train)
    species_1_logreg_train_accuracy = accuracy_score(species_1_labels_train, species_1_logreg_preds_train)

    species_1_logreg_preds_test = logreg_model.predict(species_1_embeddings_test)
    species_1_logreg_test_accuracy = accuracy_score(species_1_labels_test, species_1_logreg_preds_test)
    species_1_logreg_test_accuracy_balanced = balanced_accuracy_score(species_1_labels_test, species_1_logreg_preds_test)

    # Test dummy classifier on species 2
    species_2_dummy_preds = dummy_model.predict(species_2_embeddings)
    species_2_dummy_preds = [str(species_1_to_2_cell_type_map[cell_type_pred]) for cell_type_pred in species_2_dummy_preds]
    species_2_dummy_accuracy = accuracy_score(species_2_labels, species_2_dummy_preds)

    # Test logistic regression on species 2
    species_2_logreg_preds = logreg_model.predict(species_2_embeddings)
    species_2_logreg_preds = [str(species_1_to_2_cell_type_map.get(cell_type_pred, -1)) for cell_type_pred in species_2_logreg_preds]

    species_2_logreg_preds = np.array(species_2_logreg_preds)
    np.nan_to_num(species_2_logreg_preds)
    species_2_logreg_accuracy = accuracy_score(species_2_labels, species_2_logreg_preds)
    species_2_logreg_accuracy_balanced = balanced_accuracy_score(species_2_labels, species_2_logreg_preds)

    # Maximum accuracy on species 2 by transferring labels from species 1
    possible_transfer_labels = set(species_1_to_2_cell_type_map.values())
    species_2_labels_guess = pd.Series(np.array(species_2_labels))

    species_2_labels_guess[~species_2_labels_guess.isin(possible_transfer_labels)] = "not in"
    species_2_max_accuracy = accuracy_score(species_2_labels_guess, species_2_labels)
    species_2_max_accuracy_balanced = balanced_accuracy_score(species_2_labels_guess, species_2_labels)


    
    
    #species_2_logreg_probs = logreg_model.predict_proba(species_2_embeddings)
    #print(species_2_logreg_probs.shape[1], len(np.unique()))
    #species_2_logreg_accuracy = roc_auc_score(species_2_labels_guess, species_2_logreg_probs, multi_class="ovr")
    

    # Print results
    if verbose:
        print(f'Dummy accuracy (species 1 train ==> 1 train) = {species_1_dummy_train_accuracy:.3f}')
        print(f'Dummy accuracy (species 1 train ==> 1 test) = {species_1_dummy_test_accuracy:.3f}')
        print(f'Logistic regression accuracy (species 1 train ==> 1 train) = {species_1_logreg_train_accuracy:.3f}')
        print(f'Logistic regression accuracy (species 1 train ==> 1 test) = {species_1_logreg_test_accuracy:.3f}')
        print(f'Logistic regression accuracy (species 1 train ==> 1 test) (balanced) = {species_1_logreg_test_accuracy_balanced:.3f}')
        print(f'Dummy accuracy (species 1 train ==> 2) = {species_2_dummy_accuracy:.3f}')
        print(f'Logistic regression accuracy (species 1 train ==> 2) = {species_2_logreg_accuracy:.3f}')
        print(f'Maximum theoretical transfer accuracy (species 1 ==> 2) = {species_2_max_accuracy:.3f}')
        print(f'Logistic regression accuracy (species 1 train ==> 2) (balanced) = {species_2_logreg_accuracy_balanced:.3f}')
        print(f'Maximum theoretical transfer accuracy (species 1 ==> 2) (balanced) = {species_2_max_accuracy_balanced:.3f}')

    # Save confusion matrix
    if save_path:
        plt.rcParams['figure.figsize'] = (24, 32)
        ConfusionMatrixDisplay.from_predictions(species_2_labels, species_2_logreg_preds, xticks_rotation='vertical')
        plt.title('Logistic Regression Cell Type Transfer from Species 1 to 2')
        plt.savefig(save_path)

    # Create results dict
    results = {
        'species_1_dummy_train_accuracy': species_1_dummy_train_accuracy,
        'species_1_dummy_test_accuracy': species_1_dummy_test_accuracy,
        'species_1_logreg_train_accuracy': species_1_logreg_train_accuracy,
        'species_1_logreg_test_accuracy': species_1_logreg_test_accuracy,
        'species_2_dummy_accuracy': species_2_dummy_accuracy,
        'species_2_logreg_accuracy': species_2_logreg_accuracy,
        'species_2_max_accuracy': species_2_max_accuracy,
        'species_2_logreg_accuracy_balanced': species_2_logreg_accuracy_balanced,
    }

    return results

def metric_learning_init_checker(features, labels_col, nns = [1, 5, 10], species_1="human", species_2="mouse", metric="cosine"):
    s1 = features[features.obs["species"] == species_1]
    s2 = features[features.obs["species"] == species_2]

    s1x = s1.X.toarray()
    s2x = s2.X.toarray()


    NN = max(nns)
    nbrs_s1 = NearestNeighbors(n_neighbors=NN + 1, metric=metric).fit(s1x)
    nbrs_s2 = NearestNeighbors(n_neighbors=NN + 1, metric=metric).fit(s2x) 

    # Self neighbors
    _, indices11 = nbrs_s1.kneighbors(s1x)
    _, indices22 = nbrs_s2.kneighbors(s2x)

    _, indices12 = nbrs_s1.kneighbors(s2x)
    _, indices21 = nbrs_s2.kneighbors(s1x)    

    s1_labels = s1.obs[labels_col].values.astype(str)
    s2_labels = s2.obs[labels_col].values.astype(str)

    all_scores = {}
    for nn in nns:
        # self s1
        s1_self_nn = [s1_labels[row] for row in indices11[:, 1:(nn+1)]]
        s1_self_eq = [row == s1_labels[i] for i,row in enumerate(s1_self_nn)]
        s1_self_score = np.mean(s1_self_eq, axis=1)
        s1_self_score = np.mean(s1_self_score)

        # self s2
        s2_self_nn = [s2_labels[row] for row in indices22[:, 1:(nn+1)]]
        s2_self_eq = [row == s2_labels[i] for i,row in enumerate(s2_self_nn)]
        s2_self_score = np.mean(s2_self_eq, axis=1)
        s2_self_score = np.mean(s2_self_score)


        # cross        
        # predict s2's nn from s1
        s2_s1_nn = [s1_labels[row] for row in indices12[:, 0:(nn)]]
        s2_s1_eq = [row == s2_labels[i] for i,row in enumerate(s2_s1_nn)]
        s2_s1_score = np.mean(s2_s1_eq, axis=1)
        s2_s1_score = np.mean(s2_s1_score)

        # predict s1's nn from s2
        s1_s2_nn = [s2_labels[row] for row in indices21[:, 0:(nn)]]
        s1_s2_eq = [row == s1_labels[i] for i,row in enumerate(s1_s2_nn)]
        s1_s2_score = np.mean(s1_s2_eq, axis=1)
        s1_s2_score = np.mean(s1_s2_score)


        all_scores[nn] = {
            "1nn_1score":s1_self_score,
            "2nn_2score":s2_self_score,
            "1nn_2score":s2_s1_score,
            "2nn_1score":s1_s2_score,
        }
    return all_scores


def get_all_scores(adata_path, ct_map_path, cell_type, species_1, species_2, num_scores=4, verbose=True):
    
    adata = sc.read(adata_path)
    
    cell_type_mapping = pd.read_csv(ct_map_path)
    
    if verbose:
        print("-------------------------")
        print("------Log Reg Scores-----")
    lr_score = classify_cell_types(adata, cell_type_column=cell_type, cell_type_mapping=cell_type_mapping,
                        verbose=verbose, species_1=species_1, species_2=species_2)
    if num_scores == 1:
        return lr_score

    if verbose:
        print("-------------------------")
        print("------Reannot Score-----")
    reannot_score = get_reannotation_metrics(fname = adata_path, label=cell_type, source=species_1, target=species_2, true_labels_path=ct_map_path)
    
    
    if num_scores == 2:
        return lr_score, reannot_score
    if verbose:
        print(reannot_score)
        print("-------------------------")
        print("------Metric Scores------")
    init_scores = metric_learning_init_checker(adata, cell_type, nns=[1], species_1=species_1, species_2=species_2, metric="cosine")
    
    init_scores[1]
    print("\n".join([f"{k}\t{v}" for k,v in init_scores[1].items()]))

    if num_scores == 3:
        return lr_score
    if verbose:
        print("-------------------------")
        print("-----Alignment Scores----")
    print(get_alignment_metrics(adata_path, out_label = cell_type, orig_label=cell_type, species=[species_1, species_2], true_labels_path=ct_map_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Score one or more adatas.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--adata', help='adata path')
    parser.add_argument('--species1', help='Name of species 1')
    parser.add_argument('--species2', help='Name of species 2')
    parser.add_argument('--ct_map_path', help='Cell type mapping path')
    parser.add_argument('--label', help='Cell type column label')
    parser.add_argument('--seed', help='random seed')
    parser.add_argument('--scores', type=int, help='1 for logreg only')
    parser.add_argument('--multiple_files', type=bool, nargs='?', const=True,
                        help='If the adata path should be treated as a path to csv of format seed,path')
    
    
    parser.set_defaults(
        seed=0,
        scores=4,
        multiple_files=False
    )
    
    args = parser.parse_args()
    
    # seed = int(args.seed)
    if args.seed is None:
        seed= 0
    else:
        seed = int(args.seed)
    print(seed)
    np.random.seed(seed)
    random.seed(seed)

    
    species_1 = args.species1
    species_2 = args.species2
    cell_type = args.label
    
    adata_path = args.adata
    ct_map_path = args.ct_map_path
    
    if ct_map_path is None:
        if species_1 or species_2 == "human":
            if cell_type == "cell_type":
                ct_map_path = '/dfs/project/cross-species/data/lung/shared/true_cell_type.csv'
            elif cell_type == "CL_class_coarse":
                ct_map_path = '/dfs/project/cross-species/data/lung/shared/true_CL_class_coarse.csv'
        elif species_1 or species_2 == "zebrafish":
            ct_map_path = '/dfs/project/cross-species/yanay/fz_true_ct.csv'
            
    if args.multiple_files:
        # read the adata path
        paths_df = pd.read_csv(adata_path) # actually a path to a csv not an adata
        paths = paths_df["path"]
        lr_scores = []
        balanced_scores = []
        reannot_scores = []
        transfer_labels = []
        for path in tqdm.tqdm(paths):
            if args.scores == 1:
                lr_score, reannot_score = get_all_scores(path, ct_map_path, cell_type, species_1, species_2, num_scores=args.scores, verbose=False), None
            else:
                lr_score, reannot_score = get_all_scores(path, ct_map_path, cell_type, species_1, species_2, num_scores=args.scores, verbose=False)
            lr_scores.append(lr_score["species_2_logreg_accuracy"])
            balanced_scores.append(lr_score["species_2_logreg_accuracy_balanced"])
            reannot_scores.append(reannot_score)
            transfer_labels.append(f"{species_1} to {species_2}")
            
        for path in tqdm.tqdm(paths):
            if args.scores == 1:
                lr_score, reannot_score = get_all_scores(path, ct_map_path, cell_type, species_2, species_1, num_scores=args.scores, verbose=False), None
            else:
                lr_score, reannot_score = get_all_scores(path, ct_map_path, cell_type, species_2, species_1, num_scores=args.scores, verbose=False)
            lr_scores.append(lr_score["species_2_logreg_accuracy"])
            balanced_scores.append(lr_score["species_2_logreg_accuracy_balanced"])
            reannot_scores.append(reannot_score)
            transfer_labels.append(f"{species_2} to {species_1}")
        paths_df = pd.concat([paths_df, paths_df])
        paths_df["Logistic Regression"] = lr_scores
        paths_df["Balanced Regression"] = balanced_scores
        paths_df["Reannotation"] = reannot_scores
        paths_df["Label"] = transfer_labels
        
        
        new_path = adata_path.replace(".csv", "_scores.csv")
        print(new_path)
        print(paths_df)
        paths_df.to_csv(new_path, index=False)
    else:
        get_all_scores(adata_path, ct_map_path, cell_type, species_1, species_2, num_scores=args.scores)
    
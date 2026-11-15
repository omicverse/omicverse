import os


import warnings
warnings.filterwarnings('ignore')

import cellxgene_census
from tqdm import tqdm
import anndata

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.data as data
import torch
import numpy as np
from numpy import array
import os
import pickle as pkl
import glob

def data_to_torch_X(X):
    if isinstance(X, anndata.AnnData):
        X = X.X
    if not isinstance(X, np.ndarray):
            X = X.toarray()
    return torch.from_numpy(X).float()
    
import sys
sys.path.append('../')

from gene_embeddings import load_gene_embeddings_adata
import pandas as pd
import numpy as np
from anndata import AnnData
from omicverse.pp import filter_cells, filter_genes, highly_variable_genes
from omicverse.pp._highly_variable_genes import check_nonnegative_integers
from multiprocessing import Pool, Process, Manager

import multiprocessing.pool as mpp
# https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    if self._state != mpp.RUN:
        raise ValueError("Pool not running")

    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self._cache)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap


VERSION = "2023-04-25"
N_TOP_GENES = 12000


print(cellxgene_census.get_census_version_description(VERSION))

census = cellxgene_census.open_soma(census_version=VERSION)
census_datasets = census["census_info"]["datasets"].read().concat().to_pandas()

# for convenience, indexing on the soma_joinid which links this to other census data.
census_datasets = census_datasets.set_index("soma_joinid")

species_to_readable = {
    "Homo sapiens":"human",
    "Mus musculus":"mouse"    
}

def process_row(row, num_genes, num_cells, paths, all_species, covar_cols, dataset_title, h5_root="/dfs/project/uce/cxg_data/anndatas/", npz_root="/dfs/project/uce/cxg_data/npzs/"):
    dataset_id = row[1].dataset_id
    #dataset_title = row[1].dataset_title.lower().replace(' ', '_').replace(",", "").replace("/", "")
    
    save_path = h5_root + f"{dataset_title}.h5ad"
    no_primary_path = save_path.replace(".h5ad", "_no_primary.h5ad")
    proc_path = save_path.replace(".h5ad", "_proc.h5ad")
    npz_path = npz_root + f"{dataset_title}_counts.npz"
    # Download the anndata
    
    if os.path.exists(no_primary_path):
        print("No Primary, skipping")
        return
    
    if not os.path.exists(save_path) and not os.path.exists(no_primary_path):
        cellxgene_census.download_source_h5ad(
            dataset_id, to_path=save_path
        )
    if os.path.exists(proc_path) and os.path.exists(npz_path):
        print("Already Proc")
        try:
            ad = anndata.read_h5ad(proc_path)
        except:
            print()
            print()
            print("Error reading on:", dataset_title)
            print()
            print()
            return
        # Get organism
        if "organism" in ad.obs.columns:
            unique_organisms = list(ad.obs.organism.unique().categories)
            unique_organism_str = ", ".join(unique_organisms)
        else:
            unique_organism_str = "human"
        species = species_to_readable.get(unique_organism_str, "human")
        # don't need to do hv if already proc
        if "sample" in ad.obs.columns:
            covar_cols[dataset_title] = "sample"
        elif "batch" in ad.obs.columns:
            covar_cols[dataset_title] = "batch"
        else:
            covar_cols[dataset_title] = ""


        num_genes[dataset_title] = ad.X.shape[1]
        num_cells[dataset_title] = ad.X.shape[0]
        paths[dataset_title] = f"{dataset_title}.h5ad"
        all_species[dataset_title] = species
        
        return # Skip everything else
    # Read the raw AD
    ad = anndata.read_h5ad(save_path)
    
    # Change to counts
    if not check_nonnegative_integers(ad.X):
        # don't have counts yet, need raw
        if ad.raw is None:
            print("Skipped, no counts")
            return
        ad.X = ad.raw.X.toarray()
    if not check_nonnegative_integers(ad.X):
        print("Skipped, no counts")
        return
        
    # SUBSET TO primary data
    if len(np.unique(ad.obs["is_primary_data"])) >= 1:
        primary_data = ad.obs.is_primary_data.value_counts()
        ad = ad[ad.obs.is_primary_data]
    if ad.X.shape[0] == 0:
        print("no primary data")
        print(primary_data)
        os.rename(save_path, no_primary_path)
        return # No primary data
    print("has primary data")
    # Switch to gene symbols
    ad.var["feature_id_orig"] = list(ad.var.index)
    ad.var_names = list(ad.var.feature_name)

    # Get organism
    if "organism" in ad.obs.columns:
        unique_organisms = list(ad.obs.organism.unique().categories)
        unique_organism_str = ", ".join(unique_organisms)
    else:
        unique_organism_str = "human"
    species = species_to_readable.get(unique_organism_str, "human")
    # Filter to gene symbols with protein embeddings
    ad, _ = load_gene_embeddings_adata(
        adata=ad,
        species=[species],
        embedding_model="ESM2"
    )
    
    ad = ad.copy()
    # Simple filtering by counts
    filter_cells(ad, min_genes=200)
    filter_genes(ad, min_cells=10)
    
    #print(ad)
    
    if "sample" in ad.obs.columns:
        try:
            highly_variable_genes(ad, flavor="seurat_v3", n_top_genes=N_TOP_GENES, subset=True, batch_key="sample")
        except:
            try:
                highly_variable_genes(ad, flavor="seurat_v3", n_top_genes=N_TOP_GENES, subset=True, batch_key="sample", span=1)
            except:
                print(f"can't hv gene subset {dataset_title}")
        covar_cols[dataset_title] = "sample"
    elif "batch" in ad.obs.columns:
        try:
            highly_variable_genes(ad, flavor="seurat_v3", n_top_genes=N_TOP_GENES, subset=True, batch_key="batch")
        except:
            try:
                highly_variable_genes(ad, flavor="seurat_v3", n_top_genes=N_TOP_GENES, subset=True, batch_key="batch", span=1)
            except:
                print(f"can't hv gene subset {dataset_title}")
        covar_cols[dataset_title] = "batch"
    else:
        try:
            highly_variable_genes(ad, flavor="seurat_v3", n_top_genes=N_TOP_GENES, subset=True)
        except:
            try:
                highly_variable_genes(ad, flavor="seurat_v3", n_top_genes=N_TOP_GENES, subset=True, span=1)
            except:
                print(f"can't hv gene subset {dataset_title}")
        covar_cols[dataset_title] = ""
        
    num_genes[dataset_title] = ad.X.shape[1]
    num_cells[dataset_title] = ad.X.shape[0]
    paths[dataset_title] = f"{dataset_title}.h5ad"
    all_species[dataset_title] = species
    
    print("writing proc")
    ad.write(proc_path)
    
    arr = data_to_torch_X(ad.X).numpy()
    
    shape = arr.shape
    
    fp = np.memmap(npz_path, dtype='int64', mode='w+', shape=shape)
    fp[:] = arr[:]
    fp.flush()
    
    return
    
if __name__ == '__main__':
    '''
    manager = Manager()
    num_genes = manager.dict()
    num_cells = manager.dict()
    paths = manager.dict()
    all_species = manager.dict()
    covar_cols = manager.dict()
    '''
    num_genes = {}
    num_cells = {}
    paths = {}
    all_species = {}
    covar_cols = {}

    df = pd.DataFrame()
    # Shuffle the dataset 
    census_datasets = census_datasets#.iloc[270:]
    iterrows = list(census_datasets.iterrows())
    #p = Pool(8)
    #for row in tqdm(iterrows, total=len(census_datasets)):
    #    p.apply_async(process_row, args=(row, num_genes, num_cells, paths, all_species, covar_cols))            
    #p.close()
    #p.join()
    '''
    with Pool(1) as p:
        nrows = len(iterrows)
        inputs = zip(iterrows, [num_genes]*nrows, [num_cells]*nrows, [paths]*nrows, [all_species]*nrows, [covar_cols]*nrows)
        for _ in tqdm(p.istarmap(process_row, inputs),
                           total=nrows):
            pass
    
    '''
    
    if os.path.exists("dataset_rows_mouse_fixed.pkl"):
        dataset_rows = {}
        for path in glob.glob("dataset_rows_mouse_fixed*.pkl"):
            with open(path, "rb") as f:
                dataset_rows_path = pkl.load(f)
                dataset_rows.update(dataset_rows_path)
                
        print(f"{len(dataset_rows)} already counted")
    else:
        dataset_rows = {}
    
    
    pbar = tqdm(iterrows)
    all_errors = []
    total_number_of_cells = 0
    
    duplicate_titles = ['Dissection: Body of hippocampus (HiB) - Rostral DG-CA4', 'Retina',
       'Colon', 'Myeloid cells', 'Ileum', 'Airway']
    duplicate_titles_2 = ['retina', 'airway', 'myeloid_cells', 'colon', 'ileum', 'immune_cells']
    
    for row in pbar:
        dataset_title = row[1].dataset_title
        if dataset_title in duplicate_titles:
            dataset_title = row[1].collection_name + row[1].dataset_title

        dataset_title = dataset_title.lower().replace(' ', '_').replace(",", "").replace("/", "")

        if dataset_title in duplicate_titles_2:
            dataset_title = (row[1].collection_name + "_" + dataset_title).lower().replace(' ', '_').replace(",", "").replace("/", "")
        
        print(f"{total_number_of_cells} cells done")
        if dataset_title in dataset_rows:
            paths[dataset_title] = dataset_rows[dataset_title][0]
            all_species[dataset_title] = dataset_rows[dataset_title][1]
            covar_cols[dataset_title] = dataset_rows[dataset_title][2]
            num_cells[dataset_title] = dataset_rows[dataset_title][3]
            num_genes[dataset_title] = dataset_rows[dataset_title][4]
            #print("skipped read of proc")
            
            total_number_of_cells += dataset_rows[dataset_title][3]
            continue # Skip!
        else:
            pbar.set_description(f"{dataset_title} proc")
            try:
                process_row(row, num_genes, num_cells, paths, all_species, covar_cols, dataset_title=dataset_title)
            except:
                print(f"****{dataset_title} ERROR****")
                all_errors.append(dataset_title)
                
                
            pbar.set_description(f"{dataset_title} done")
            
            if dataset_title in paths:
                dataset_rows[dataset_title] = [paths[dataset_title], all_species[dataset_title], covar_cols[dataset_title], num_cells[dataset_title], num_genes[dataset_title], dataset_title]

                total_number_of_cells += dataset_rows[dataset_title][3]

                with open("dataset_rows_mouse_fixed.pkl", "wb") as f:
                    pkl.dump(dataset_rows, f)
                    print("wrote pkl")
            
    # path,species,covar_col,num_cells,names
    
    df["path"] = list(paths.values())
    df["species"] = list(all_species.values())
    df["covar_col"] = list(covar_cols.values())
    df["num_cells"] = list(num_cells.values())
    df["num_genes"] = list(num_genes.values())
    df["names"] = list(paths.keys())

    print(df.head(20))
    print()
    print("Errors:")
    print(all_errors)
    df.to_csv("cxg_datasets.csv", index=False)

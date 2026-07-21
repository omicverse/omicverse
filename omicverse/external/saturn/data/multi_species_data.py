'''
Created on Nov 7, 2022

@author: Yanay Rosen
'''

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch.utils.data as data
import torch
import numpy as np
import scanpy as sc


def data_to_torch_X(X):
    if isinstance(X, sc.AnnData):
        X = X.X
    if not isinstance(X, np.ndarray):
            X = X.toarray()
    return torch.from_numpy(X).float()

class ExperimentDatasetMulti(data.Dataset):
    def __init__(self,
                all_data: dict, all_ys: dict, all_refs: dict, all_batch_labs: dict) -> None:
        super(ExperimentDatasetMulti, self).__init__()
        self.xs = {}
        self.num_cells = {}
        self.num_genes = {}
        self.ref_labels = {}
        self.batch_labels = {}
        
        for species, data in all_data.items():
            X = data_to_torch_X(data)
            num_cells, num_genes = X.shape
            self.xs[species] = X
            self.num_cells[species] = num_cells
            self.num_genes[species] = num_genes
            
            
        self.ys = {}
        for species, y in all_ys.items():
            y = torch.LongTensor(y)
            self.ys[species] = y
        for species, ref in all_refs.items():
            r = torch.LongTensor(ref)
            self.ref_labels[species] = r
        if len(all_batch_labs) != 0: # if we have an additional batch column like for tissue
            for species, tissue in all_batch_labs.items():
                t = torch.LongTensor(tissue)
                self.batch_labels[species] = t
                
        self.species = sorted(list(all_data.keys()))

    def __getitem__(self, idx):
        if isinstance(idx, int):
            count = 0
            for species in self.species:
                if idx < self.num_cells[species]:
                    if len(self.batch_labels) != 0:
                        batch_ret = self.batch_labels[species][idx]
                    else:
                        batch_ret = None
                    
                    return self.xs[species][idx], self.ys[species][idx], self.ref_labels[species][idx], species, batch_ret
                else:
                    idx -= self.num_cells[species]
            raise IndexError
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return sum(self.num_cells.values())

    def get_dim(self) -> Dict[str, int]:
        return self.num_genes
    
    
class ExperimentDatasetMultiEqual(data.Dataset):
    def __init__(self,
                all_data: dict, all_ys: dict, all_refs: dict, all_batch_labs:dict) -> None:
        super(ExperimentDatasetMultiEqual, self).__init__()
        self.xs = {}
        self.num_cells = {}
        self.num_genes = {}
        self.ref_labels = {}
        self.batch_labels = {}
        
        self.max_cells = 0
        
        for species, data in all_data.items():
            X = data_to_torch_X(data)
            num_cells, num_genes = X.shape
            self.xs[species] = X
            self.num_cells[species] = num_cells
            self.num_genes[species] = num_genes
            
            self.max_cells = max(self.max_cells, num_cells)
            
            
        self.ys = {}
        for species, y in all_ys.items():
            y = torch.LongTensor(y)
            self.ys[species] = y
        for species, ref in all_refs.items():
            r = torch.LongTensor(ref)
            self.ref_labels[species] = r
        if len(all_batch_labs) != 0: # if we have an additional batch column like for tissue
            for species, tissue in all_batch_labs.items():
                t = torch.LongTensor(tissue)
                self.batch_labels[species] = t
        self.species = sorted(list(all_data.keys()))
        
        self.total_num_cells = self.max_cells * len(self.species) # number of cells * number of species

    def __getitem__(self, idx):
        if isinstance(idx, int):
            count = 0
            species = self.species[idx % len(self.species)] # get species
            count_idx = idx // len(self.species) # index within this species
            idx = count_idx
            if idx < self.num_cells[species]:
                if len(self.batch_labels) != 0:
                        batch_ret = self.batch_labels[species][idx]
                else:
                    batch_ret = None             
                return self.xs[species][idx], self.ys[species][idx], self.ref_labels[species][idx], species, batch_ret
            elif idx < self.max_cells:
                #idx = idx % self.num_cells[species]
                idx = np.random.choice(np.arange(self.num_cells[species]))
                if len(self.batch_labels) != 0:
                        batch_ret = self.batch_labels[species][idx]
                else:
                    batch_ret = None   
                return self.xs[species][idx], self.ys[species][idx], self.ref_labels[species][idx], species, batch_ret
            raise IndexError
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return self.total_num_cells

    def get_dim(self) -> Dict[str, int]:
        return self.num_genes
    

def multi_species_collate_fn(batch: List[Tuple[torch.FloatTensor, torch.LongTensor, str]]) -> dict:
    species_to_data = defaultdict(list)
    species_to_labels = defaultdict(list)
    species_to_refs = defaultdict(list)
    species_to_batch_labels = defaultdict(list)
    
    has_batch_labels = False
    
    for data, labels, refs, species, batch_labels in batch:
        species_to_data[species].append(data)
        species_to_labels[species].append(labels)
        species_to_refs[species].append(refs)
        
        if batch_labels is not None:
            species_to_batch_labels[species].append(batch_labels)
            has_batch_labels = True

    # assert 1 <= len(species_to_data) <= 2
    batch_dict = {}
    all_species = sorted(species_to_data)
    for species in all_species:
        if has_batch_labels:
            data, labels, refs, batch_labels = torch.stack(species_to_data[species]), torch.stack(species_to_labels[species]), torch.stack(species_to_refs[species]), torch.stack(species_to_batch_labels[species])
        else:
            data, labels, refs, batch_labels = torch.stack(species_to_data[species]), torch.stack(species_to_labels[species]), torch.stack(species_to_refs[species]), None
        
        batch_dict[species] = (data, labels, refs, batch_labels)

    return batch_dict #{species:[data, label, ref_label, batch_label]}



# Deprecated (don't use this)
class ExperimentDatasetMultiEqualCT(data.Dataset):
    def __init__(self,
                all_data: dict, ys_col: str, refs_col: str, cluster_col: str) -> None:
        super(ExperimentDatasetMultiEqualCT, self).__init__()
        self.xs = {}
        self.num_cells = {}
        self.num_genes = {}
        self.ref_labels = {}
        
        
        
        # a huge number to min with
        self.max_cells = 0
        self.min_cells = 10e15
        for species, data in all_data.items():
            self.max_cells = max(data.X.shape[0], self.max_cells)
            self.min_cells = min(data.X.shape[0], self.min_cells)
            
        #max_cells_type = int(self.min_cells * 0.05)
        #print(f"Max Cells: {self.max_cells} Min Cells: {self.min_cells}, Subset: {max_cells_type}")
        #max_cells_type = 1000 #int(0.05 * ) # hardcode to 1000
        
        
        species_to_new_adata = {}
        for species, data in all_data.items():
            max_cells_type = int(data.X.shape[0] * 0.025) # specific to this adata            
            print(f"Species Cells: {data.X.shape[0]}, Subset: {max_cells_type}")
            
            adatas = [data[data.obs[cluster_col].isin([clust])] for clust in data.obs[cluster_col].cat.categories]

            for dat in adatas:
                if dat.n_obs > max_cells_type:
                    sc.pp.subsample(dat, n_obs=max_cells_type, random_state=0)

            data = adatas[0].concatenate(*adatas[1:])
            species_to_new_adata[species] = data
            
        all_ys = {species:adata.obs[ys_col] for (species, adata) in species_to_new_adata.items()}
        all_refs = {species:adata.obs[refs_col] for (species, adata) in species_to_new_adata.items()}
        
        self.max_cells = 0
        for species, data in species_to_new_adata.items():
            X = data_to_torch_X(data)
            num_cells, num_genes = X.shape
            self.xs[species] = X
            self.num_cells[species] = num_cells
            self.num_genes[species] = num_genes
            
            self.max_cells = max(self.max_cells, num_cells)
            
            
        self.ys = {}
        for species, y in all_ys.items():
            y = torch.LongTensor(y)
            self.ys[species] = y
        for species, ref in all_refs.items():
            r = torch.LongTensor(ref)
            self.ref_labels[species] = r
        self.species = sorted(list(all_data.keys()))
        
        self.total_num_cells = self.max_cells * len(self.species) # number of cells * number of species

    def __getitem__(self, idx):
        if isinstance(idx, int):
            count = 0
            species = self.species[idx % len(self.species)] # get species
            count_idx = idx // len(self.species) # index within this species
            idx = count_idx
            if idx < self.num_cells[species]:
                return self.xs[species][idx], self.ys[species][idx], self.ref_labels[species][idx], species
            elif idx < self.max_cells:
                # TODO this is biased towards ordering at the begining of the adata hopefully doesn't make a huge
                # difference
                #idx = idx % self.num_cells[species]
                idx = np.random.choice(np.arange(self.num_cells[species]))
                return self.xs[species][idx], self.ys[species][idx], self.ref_labels[species][idx], species
            else:
                raise IndexError
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return self.total_num_cells

    def get_dim(self) -> Dict[str, int]:
        return self.num_genes
from PIL import ImageFilter
import random
from anndata._core.anndata import AnnData
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import scanpy as sc
import torch
from copy import deepcopy
import numpy as np
import pandas as pd
    

class scRNAMatrixInstance(Dataset):
    def __init__(self,
                 adata: AnnData = None,
                 obs_label_colname: str = "x",
                 transform: bool = False,
                 args_transformation: dict = {}
                 ):

        super().__init__()

        self.adata = adata

        # data
        # scipy.sparse.csr.csr_matrix or numpy.ndarray
        if isinstance(self.adata.X, np.ndarray):
            self.data = self.adata.X
        else:
            self.data = self.adata.X.toarray()

        # label (if exist, build the label encoder)
        if self.adata.obs.get(obs_label_colname) is not None:
            self.label = self.adata.obs[obs_label_colname]
            self.unique_label = list(set(self.label))
            self.label_encoder = {k: v for k, v in zip(self.unique_label, range(len(self.unique_label)))}
            self.label_decoder = {v: k for k, v in self.label_encoder.items()}
        else:
            self.label = None
            print("Can not find corresponding labels")

        # do the transformation
        self.transform = transform
        self.num_cells, self.num_genes = self.adata.shape
        self.args_transformation = args_transformation
        
        self.dataset_for_transform = deepcopy(self.data)

        
    def RandomTransform(self, sample):
        #tr_sample = deepcopy(sample)
        tr = transformation(self.dataset_for_transform, sample)
        
        # the crop operation

        # Mask
        tr.random_mask(self.args_transformation['mask_percentage'], self.args_transformation['apply_mask_prob'])

        # (Add) Gaussian noise
        tr.random_gaussian_noise(self.args_transformation['noise_percentage'], self.args_transformation['sigma'], self.args_transformation['apply_noise_prob'])

        # inner swap
        tr.random_swap(self.args_transformation['swap_percentage'], self.args_transformation['apply_swap_prob'])
        
        # cross over with one instance
        tr.instance_crossover(self.args_transformation['cross_percentage'], self.args_transformation['apply_cross_prob'])

        # cross over with many instances
        tr.tf_idf_based_replacement(self.args_transformation['change_percentage'], self.args_transformation['apply_mutation_prob'], True)
        tr.ToTensor()

        return tr.cell_profile


    def __getitem__(self, index):
        
        sample = self.data[index]

        if self.label is not None:
            label = self.label_encoder[self.label[index]]
        else:
            label = -1

        if self.transform:
            sample_1 = self.RandomTransform(sample)
            sample_2 = self.RandomTransform(sample)
            sample = [sample_1, sample_2]
        
        return sample, index, label

    def __len__(self):
        return self.adata.X.shape[0]


class transformation():
    
    def __init__(self, 
                 dataset,
                 sample, 
                 batch=None):
        self.dataset = dataset
        self.sample = sample # (1, 3); X, orig_label, species
        
        self.cell_profile = sample[0].numpy()
        self.curr_label = sample[1]
        self.curr_species = sample[2]
        
        self.gene_num = self.dataset.shape[1] # num of genes
        self.cell_num = len(self.dataset) # total num of cells
        self.batch = batch # (batch_size, 3), X, orig_labels (raw), species
        
        # todo: convert these to torch tensors
        self.batch_x = batch[0]
        self.orig_labels = batch[1]
        self.species = batch[2]
        print('orig_labels type: ', type(orig_labels))
        print('species type: ', type(species))
    
    def build_mask(self, masked_percentage: float):
        mask = np.concatenate([np.ones(int(self.gene_num * masked_percentage), dtype=bool), 
                               np.zeros(self.gene_num - int(self.gene_num * masked_percentage), dtype=bool)])
        np.random.shuffle(mask)
        return mask
    

    def RandomCrop(self,
                   crop_percentage=0.8):
        mask = self.build_mask(crop_percentage)
        self.cell_profile = self.cell_profile[mask]
        self.gene_num = len(self.cell_profile)
        self.dataset = self.dataset[:,mask]

    def random_mask(self, 
                    mask_percentage: float = 0.15, 
                    apply_mask_prob: float = 0.5):

        s = np.random.uniform(0,1)
        if s<apply_mask_prob:
            # create the mask for mutation
            mask = self.build_mask(mask_percentage)
            
            # do the mutation with prob
            self.cell_profile[mask] = 0

    def random_gaussian_noise(self, 
                              noise_percentage: float=0.2, 
                              sigma: float=0.5, 
                              apply_noise_prob: float=0.3):

        s = np.random.uniform(0,1)
        if s < apply_noise_prob:
            # create the mask for mutation
            mask = self.build_mask(noise_percentage)
            
            # create the noise
            noise = np.random.normal(0, 0.5, int(self.gene_num*noise_percentage))
            
            # do the mutation (maybe not add, simply change the value?)
            self.cell_profile[mask] += noise


    def random_swap(self,
                    swap_percentage: float=0.1,
                    apply_swap_prob: float=0.5):

        ##### for debug
        #     from copy import deepcopy
        #     before_swap = deepcopy(cell_profile)
        s = np.random.uniform(0,1)
        if s<apply_swap_prob:
            # create the number of pairs for swapping 
            swap_instances = int(self.gene_num*swap_percentage/2)
            swap_pair = np.random.randint(self.gene_num, size=(swap_instances,2))
            
            # do the inner crossover with p
        
            self.cell_profile[swap_pair[:,0]], self.cell_profile[swap_pair[:,1]] = \
                self.cell_profile[swap_pair[:,1]], self.cell_profile[swap_pair[:, 0]]



    def instance_crossover(self,
                           cross_percentage: float=0.25,
                           apply_cross_prob: float=0.4):
        
        # it's better to choose a similar profile to crossover
        
        s = np.random.uniform(0,1)
        if s < apply_cross_prob:
            # choose one instance for crossover
            cross_idx = np.random.randint(self.cell_num)
            print(cross_idx)
            cross_instance = self.dataset[cross_idx]
            
            # build the mask
            mask = self.build_mask(cross_percentage)
            
            # apply instance crossover with p
            tmp = cross_instance[mask].copy()
        
            cross_instance[mask], self.cell_profile[mask]  = self.cell_profile[mask], tmp


    def same_type_crossover(self,
                           cross_percentage: float=0.25,
                           apply_cross_prob: float=0.4):
        s = np.random.uniform(0,1)
        if s < apply_cross_prob:
            # choose one instance for crossover
            cross_idx = np.random.randint(self.cell_num)
            
            # todo: instead of choosing random cross_idx, select cell_idx from within batch, O(1)
            sp_inds = torch.where(self.species != self.curr_species)[0]
            lbl_inds = torch.where(self.orig_labels == self.curr_label)[0]
            combined = torch.cat((sp_inds, lbl_inds))
            cross_inds = combined.unique()
            
            n_cross_inds = len(cross_inds)
            if n_cross_inds > 0: 
                cross_idx = cross_inds[np.random.randint(n_cross_inds)] # select random cross_idx
            elif n_lbl_inds > 0: 
                cross_idx = lbl_inds[np.random.randint(len(lbl_inds))]
            else: 
                return
                  
            cross_instance = self.batch_x[cross_idx] # get other cell to swap with
            
            # build the mask
            mask = self.build_mask(cross_percentage)
            
            # apply instance crossover with p
            tmp = cross_instance[mask].copy()
        
            cross_instance[mask], self.cell_profile[mask]  = self.cell_profile[mask], tmp
            
    def tf_idf_based_replacement(self, 
                                 change_percentage: float=0.25,
                                 apply_mutation_prob: float=0.2,
                                 new=False):

        # 
        s = np.random.uniform(0,1)

        # the speed is too slow

        if s<apply_mutation_prob:
            if not new:
                mask = self.build_mask(change_percentage)
                chosen = self.dataset[:,mask]
                mutations = np.apply_along_axis(random_substitution, axis=0, arr=chosen)
                self.cell_profile[mask] = mutations[0]
            else:
                mask = self.build_mask(change_percentage)
                cell_random = np.random.randint(self.cell_num, size=int(self.gene_num * change_percentage))
                chosen = self.dataset[cell_random, mask]
                self.cell_profile[mask] = chosen


    def ToTensor(self):
        self.cell_profile = torch.from_numpy(self.cell_profile)


def random_substitution(x):
    random_cell = np.random.randint(x.shape)
    return x[random_cell]
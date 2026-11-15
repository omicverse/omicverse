import os


import warnings
warnings.filterwarnings("ignore")

import anndata
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os
import argparse
import logging
import time

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd


import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


from accelerate import Accelerator
import anndata
from data_utils import adata_path_to_prot_chrom_starts, get_spec_chrom_csv



from torch.utils.data import dataset
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import binom




def padding_tensor(sequences):
    """
    :param sequences: list of tensors
    :return:
    """
    num = len(sequences)
    max_len = max([s.size(0) for s in sequences])
    out_dims = (num, max_len, 1280)
    
    
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    out_dims2 = (num, max_len)
    
    mask = sequences[0].data.new(*out_dims2).fill_(float('-inf'))
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
        mask[i, :length] = 1
    return out_tensor.permute(1, 0, 2), mask


from pathlib import Path
# ESM1b
'''
EMBEDDING_DIR = Path('/dfs/project/cross-species/data/proteome/embeddings')
human_pe_dir =  EMBEDDING_DIR / 'Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM1b.pt'
mouse_pe_dir =  EMBEDDING_DIR / 'Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM1b.pt'
lemur_pe_dir =  Path("/dfs/project/cross-species/yanay/data/proteome/embeddings/") / 'Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM1b.pt'

'''

# Upgrade to ESM2
EMBEDDING_DIR = Path('/dfs/project/cross-species/data/proteome/embeddings')
EMBEDDING_DIR = Path('/dfs/project/cross-species/yanay/data/proteome/embeddings')

embeddings_paths = {
        'human': EMBEDDING_DIR / 'Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt',
        'mouse': EMBEDDING_DIR / 'Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt',
        'frog': EMBEDDING_DIR / 'Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt',
        'zebrafish': EMBEDDING_DIR / 'Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt',
        "mouse_lemur": EMBEDDING_DIR / "Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt",
        "pig": EMBEDDING_DIR / 'Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt',
        "macaca_fascicularis": EMBEDDING_DIR / 'Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt',
        "macaca_mulatta": EMBEDDING_DIR / 'Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt',
    }

species_to_pe = {
    species:torch.load(pe_dir) for species, pe_dir in embeddings_paths.items()   
}

species_to_pe = {species:{k.upper(): v for k,v in pe.items()} for species, pe in species_to_pe.items()}

#species_to_keys = {species:list(pe.keys()) for species, pe in species_to_pe.items()}
#species_to_keys = {species:dict(zip(keys, np.arange(len(keys)))) for species, keys in species_to_keys.items()}


#datasets_df = pd.read_csv("/dfs/project/cross-species/yanay/code/UCE/data_proc/full_train_datasets.csv")
datasets_df = pd.read_csv("tissue_datasets.csv")
datasets_df = pd.read_csv("perturb_datasets.csv")
datasets_df = pd.read_csv("../new_perturb_datasets.csv")


#pd.concat((#pd.read_csv("new_datasets.csv"),
                             #pd.read_csv("pbmcs_nohvg.csv"), 
                             #pd.read_csv("lung_nohvg.csv"),
                             #pd.read_csv("new_tabula_datasets.csv"),
                             #pd.read_csv("updated_datasets.csv"),
  #                           #pd.read_csv("sanger_heart_atlas_datasets.csv"),
 #                            pd.read_csv("tissue_datasets.csv")
 #                       ))




#datasets_df = pd.read_csv("cell_cycle_datasets.csv")
#datasets_df = pd.read_csv("spatial_datasets.csv")
#datasets_df = pd.read_csv("perturb_datasets.csv")
#datasets_df = pd.read_csv("ccle_datasets.csv")
#datasets_df = pd.read_csv("pancreas_datasets.csv")



sorted_dataset_names = sorted(datasets_df["names"])
with open("dataset_shapes.pkl", "rb") as f:
    shapes_dict = pickle.load(f)
    

shapes_dict.update({
 "madissoon_novel_lung":(190728, 8000),   
 'flores_cerebellum_human': (20232, 8000),
 'osuch_gut_human': (272310, 8000),
 'msk_ovarian_human': (929690, 8000),
 'htan_vmuc_dis_epi_human': (65084, 8000),
 'htan_vmuc_val_epi_human': (57564, 8000),
 'htan_vmuc_non_epi_human': (9099, 8000),
 'hao_pbmc_3p_human': (161764, 8000),
 'hao_pbmc_5p_human': (49147, 8000),
 'gao_tumors_human': (36111, 8000),
 'swabrick_breast_human': (92427, 8000),
 'wu_cryo_tumors_human': (105662, 8000),
 'cell_line_het_human': (53513, 8000),
 'bi_allen_metastasis_human': (27787, 8000),
 'zheng68k_human': (68579, 8000),
 'zheng68k_12k_human': (68579, 12000),
 'mouse_embryo_ct': (153597, 12000),
 "regev_gtex_heart": (36574, 8000),
 "tabula_sapiens_heart": (11505, 8000),
 "10k_pbmcs":(11990, 12000),
 "epo_ido":(35834,12000),
 'tabula_sapiens_kidney': (9641, 8000),
 'tabula_microcebus_kidney': (14592, 8000),
 'tabula_muris_kidney': (2781, 8000),
 'tabula_muris_senis_kidney': (19610, 8000),
  'immune_human': (33506, 8000)
                   })

for row in datasets_df.iterrows():
    ngenes = row[1].num_genes
    ncells = row[1].num_cells
    name = row[1].names
    if not np.isnan(ngenes):
        shapes_dict[name] = (int(ncells), int(ngenes))
                   
#with open("dataset_shapes.pkl", "wb") as f:
#    pickle.dump(shapes_dict, f)
token_dim = 5120
mmap_dict = {}

root_dir = "/lfs/local/0/yanay/uce_h5s/"
root_dir_census = "/lfs/local/0/yanay/cxg_h5s/"

dataset_to_paths = {r[1]["names"]:root_dir + r[1]["path"].replace(".h5ad", "_proc.h5ad") for r in datasets_df.iterrows()}
for row in datasets_df.iterrows():
    name = row[1].names
    census = row[1].census
    
    if census == "yes":
        dataset_to_paths[name] = dataset_to_paths[name].replace(root_dir, root_dir_census)


datasets_to_species = {r[1]["names"]:r[1]["species"] for r in datasets_df.iterrows()}

#species_to_pe = {"mouse":mouse_pe, "human":human_pe, "mouse_lemur":lemur_pe}

#dataset_to_protein_embeddings_all = {k:species_to_pe[v] for k, v in datasets_to_species.items()}

dataset_to_protein_embeddings = {}


#dataset_to_protein_embeddings_all["madissoon_novel_lung"] = species_to_pe["human"]
datasets_to_species["madissoon_novel_lung"] = "human"
#dataset_to_paths["madissoon_novel_lung"] = "/lfs/local/0/yanay/uce_h5s/madissoon_novel_lung_proc.h5ad"



# New Chrom Based Code
gene_to_chrom_pos = get_spec_chrom_csv()
species_to_chrom_categories = {}

for species in np.unique(gene_to_chrom_pos["species"]):
    species_to_chrom_categories[species] = pd.Categorical(gene_to_chrom_pos["chromosome"]).categories

    
dataset_to_chroms = {}
dataset_to_starts = {}

sorted_species_names = sorted(species_to_pe.keys())
print(sorted_species_names)

if os.path.exists(f"/dfs/project/uce/all_species_pe_tokens.torch"):
    all_pe = torch.load(f"/dfs/project/uce/all_species_pe_tokens.torch")
    with open("/dfs/project/uce/all_species_offsets.pkl", "rb") as f:
        species_to_offsets = pickle.load(f)
    print("Loaded PE", all_pe.shape)
else:
    torch.manual_seed(8)
    MASK_TENSOR = torch.zeros((1, token_dim)) # this is the padding token
    CHROM_TENSOR_LEFT = torch.normal(mean=0, std=1, size=(1, token_dim))
    CHROM_TENSOR_RIGHT = torch.normal(mean=0, std=1, size=(1, token_dim))
    CLS_TENSOR = torch.normal(mean=0, std=1, size=(1, token_dim))
    species_to_offsets = {}

    all_pe = [MASK_TENSOR, CHROM_TENSOR_LEFT, CHROM_TENSOR_RIGHT, CLS_TENSOR]
    offset = len(all_pe) # special tokens at the top!
    for species in sorted_species_names:
        pe_stacked = torch.stack(list(species_to_pe[species].values()))
        all_pe.append(pe_stacked)
        species_to_offsets[species] = offset
        offset += pe_stacked.shape[0]

    all_pe = torch.vstack(all_pe)
    print(all_pe.shape)
    torch.save(all_pe, f"/dfs/project/uce/all_species_pe_tokens.torch")
    with open("/dfs/project/uce/all_species_offsets.pkl", "wb+") as f:
        pickle.dump(species_to_offsets, f)
    print("Saved PE")

# Load in already saved!
if os.path.exists(f"/lfs/local/0/yanay/reduced_datasets_to_pe_chrom_{token_dim}_new.torch"):
    dataset_to_protein_embeddings = torch.load(f"/lfs/local/0/yanay/reduced_datasets_to_pe_chrom_{token_dim}_new.torch")

    with open("/lfs/local/0/yanay/dataset_to_chroms_new.pkl", "rb") as f:
        dataset_to_chroms = pickle.load(f)
    with open("/lfs/local/0/yanay/dataset_to_starts_new.pkl", "rb") as f:
        dataset_to_starts = pickle.load(f)
else:
    dataset_to_protein_embeddings = {}
    dataset_to_chroms = {}
    dataset_to_starts = {}


# Add the new ones
print("creating reduced size protein embeddings file")

redo = True

for dataset, path in tqdm(list(dataset_to_paths.items())):
    if dataset in dataset_to_protein_embeddings.keys() and not redo:
        continue # skip since already procced
    print(dataset)
    adata = anndata.read_h5ad(path)
    dataset_species = datasets_to_species[dataset]
    spec_pe_genes = list(species_to_pe[dataset_species].keys())
    offset = species_to_offsets[dataset_species]
    
    # Get proper idxs
    pe_row_idxs, dataset_chroms, dataset_pos = adata_path_to_prot_chrom_starts(adata, dataset_species, spec_pe_genes, gene_to_chrom_pos, offset)
    # Add to dicts
    dataset_to_chroms[dataset] = dataset_chroms
    dataset_to_starts[dataset] = dataset_pos
    dataset_to_protein_embeddings[dataset] = pe_row_idxs
    
    del adata
# save Dicts and idxs
torch.save(dataset_to_protein_embeddings, f"/lfs/local/0/yanay/reduced_datasets_to_pe_chrom_{token_dim}_new.torch")

with open("/lfs/local/0/yanay/dataset_to_chroms_new.pkl", "wb+") as f:
    pickle.dump(dataset_to_chroms, f)
with open("/lfs/local/0/yanay/dataset_to_starts_new.pkl", "wb+") as f:
    pickle.dump(dataset_to_starts, f)        
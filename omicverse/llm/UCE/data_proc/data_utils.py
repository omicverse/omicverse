import warnings
warnings.filterwarnings("ignore")

import torch

from torch import nn, Tensor
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import pickle
import os
import argparse
import logging
import time

from tqdm.auto import tqdm
import pandas as pd

import math
import anndata
from pathlib import Path


from torch.utils.data import dataset
from torch.utils.data import DataLoader, TensorDataset, dataset
from scipy.stats import binom
from typing import Dict, List, Optional, Tuple
from anndata import AnnData


# Import moved to function to allow path patching first
# from data_proc.gene_embeddings import load_gene_embeddings_adata

# No global variables needed - using parameter passing instead

def load_gene_embeddings_adata_inline(adata, species, embedding_model, protein_embeddings_dir=None):
    """
    Inline implementation of gene embeddings loading with configurable path.
    
    Args:
        adata: AnnData object containing gene expression data for cells
        species: List of species corresponding to this adata
        embedding_model: The gene embedding model whose embeddings will be loaded
        protein_embeddings_dir: Directory containing protein embedding files
    
    Returns:
        Tuple containing:
        - A subset of the data only containing genes with embeddings in all species
        - A dictionary mapping species name to the corresponding gene embedding matrix
    """
    from pathlib import Path
    import torch
    
    # Use provided directory or fall back to default
    if protein_embeddings_dir is not None:
        embedding_dir = Path(protein_embeddings_dir)
    else:
        embedding_dir = Path('model_files/protein_embeddings')
    
    # Define species to embedding file mapping
    species_to_embedding_files = {
        'human': 'Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt',
        'mouse': 'Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt',
        'frog': 'Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt',
        'zebrafish': 'Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt',
        'mouse_lemur': 'Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt',
        'pig': 'Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt',
        'macaca_fascicularis': 'Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt',
        'macaca_mulatta': 'Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt',
    }
    
    # Get species names
    species_names = species
    species_names_set = set(species_names)
    available_species = set(species_to_embedding_files.keys())
    
    # Ensure embeddings are available for all species
    if not (species_names_set <= available_species):
        raise ValueError(f'The following species do not have gene embeddings: {species_names_set - available_species}')
    
    # Load gene embeddings for desired species (and convert gene symbols to lower case)
    species_to_gene_symbol_to_embedding = {}
    for species_name in species_names:
        embedding_file = species_to_embedding_files[species_name]
        embedding_path = embedding_dir / embedding_file
        
        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
            
        embeddings = torch.load(embedding_path)
        species_to_gene_symbol_to_embedding[species_name] = {
            gene_symbol.lower(): gene_embedding
            for gene_symbol, gene_embedding in embeddings.items()
        }
    
    # Determine which genes to include based on gene expression and embedding availability
    genes_with_embeddings = set.intersection(*[
        set(gene_symbol_to_embedding)
        for gene_symbol_to_embedding in species_to_gene_symbol_to_embedding.values()
    ])
    genes_to_use = {gene for gene in adata.var_names if gene.lower() in genes_with_embeddings}
    
    # Subset data to only use genes with embeddings
    adata = adata[:, adata.var_names.isin(genes_to_use)]
    
    # Set up dictionary mapping species to gene embedding matrix (num_genes, embedding_dim)
    species_to_gene_embeddings = {
        species_name: torch.stack([
            species_to_gene_symbol_to_embedding[species_name][gene_symbol.lower()]
            for gene_symbol in adata.var_names
        ])
        for species_name in species_names
    }
    
    return adata, species_to_gene_embeddings

def data_to_torch_X(X):
    if isinstance(X, AnnData):
        X = X.X
    if not isinstance(X, np.ndarray):
            X = X.toarray()
    return torch.from_numpy(X).float()

class SincleCellDataset(data.Dataset):
    def __init__(self,
                expression: torch.tensor, # Subset to hv genes, count data! cells x genes
                protein_embeddings: torch.tensor, # same order as expression, also subset genes x pe
                labels: None, # optional, tensor of labels
                covar_vals: None, # tensor of covar values or none
                ) -> None:
        super(SincleCellDataset, self).__init__()
        
        # Set expression
        self.expression = expression
        
        row_sums = self.expression.sum(1) # UMI Counts
        log_norm_count_adj = torch.log1p(self.expression / (self.expression.sum(1)).unsqueeze(1) * torch.tensor(1000))       
        
        # Set log norm and count adjusted expression
        max_vals, max_idx = torch.max(log_norm_count_adj, dim=0)
        self.expression_mod =  log_norm_count_adj / max_vals
        
        # Calculate dropout likliehoods of each gene
        self.dropout_vec = (self.expression == 0).float().mean(0) # per gene dropout percentages
        
        # Set data info
        self.num_cells = self.expression.shape[0]
        self.num_genes = self.expression.shape[1]
        
        # Set optional label info, including categorical covariate index
        self.covar_vals = covar_vals
        self.labels = labels
        
        # Set protein embeddings
        self.protein_embeddings = protein_embeddings
        
        self.item_mode = "expression"
        if self.covar_vals is not None:
            self.item_mode = "expression+covar"
        
        
    def __getitem__(self, idx):
        if self.item_mode == "expression":
            if isinstance(idx, int):
                if idx < self.num_cells:
                    return self.expression[idx, :]
                else:
                    raise IndexError
            else:
                raise NotImplementedError
        elif self.item_mode == "expression+covar":
            if isinstance(idx, int):
                if idx < self.num_cells:
                    return self.expression[idx, :], self.covar_vals[idx]
                else:
                    raise IndexError
            else:
                raise NotImplementedError
            

    def __len__(self) -> int:
        return self.num_cells

    def get_dim(self) -> Dict[str, int]:
        return self.num_genes


def data_to_torch_X(X):
    if isinstance(X, AnnData):
        X = X.X
    if not isinstance(X, np.ndarray):
            X = X.toarray()
    return torch.from_numpy(X).float()


def anndata_to_sc_dataset(adata:AnnData, 
                                 species:str="human", 
                                 labels:list=[],
                                 covar_col:str=None,
                                 hv_genes=None,
                                 embedding_model="ESM2",
                                 protein_embeddings_dir=None,
                                ):
    
    # Use inline implementation instead of gene_embeddings.py to avoid path issues
    adata, protein_embeddings = load_gene_embeddings_adata_inline(
        adata=adata,
        species=[species],
        embedding_model=embedding_model,
        protein_embeddings_dir=protein_embeddings_dir
    )
    
    if hv_genes is not None:
        from omicverse.pp import highly_variable_genes
        highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=hv_genes)  # Expects Count Data
    
        hv_index = adata.var["highly_variable"]
        adata = adata[:, hv_index] # Subset to hv genes only
    
        protein_embeddings = protein_embeddings[species][hv_index]
    else:
        protein_embeddings = protein_embeddings[species]
    expression = data_to_torch_X(adata.X)
    
    covar_vals = None
    if len(labels) > 0:
        assert covar_col is None or covar_col in labels, "Covar needs to be in labels" # make sure you keep track of covar column!
        labels = adata.obs.loc[:, labels].values
        
        if covar_col is not None:
            # we have a categorical label to use as covariate
            covar_vals = torch.tensor(pd.Categorical(adata.obs[covar_col]).codes)
    return SincleCellDataset(
        expression=expression,
        protein_embeddings=protein_embeddings,
        labels=labels,
        covar_vals=covar_vals
    ), adata    
    
def adata_path_to_prot_chrom_starts(adata, dataset_species, spec_pe_genes, gene_to_chrom_pos, offset):
    """
        Given a :path: to an h5ad, 
    """    
    pe_row_idxs = torch.tensor([spec_pe_genes.index(k.upper()) + offset for k in adata.var_names]).long()
    print(len(np.unique(pe_row_idxs)))
    
    spec_chrom = gene_to_chrom_pos[gene_to_chrom_pos["species"] == dataset_species].set_index("gene_symbol")

    gene_chrom = spec_chrom.loc[[k.upper() for k in adata.var_names]]

    dataset_chroms = gene_chrom["spec_chrom"].cat.codes # now this is correctely indexed by species and chromosome
    print("Max Code:", max(dataset_chroms))
    dataset_pos = gene_chrom["start"].values
    return pe_row_idxs, dataset_chroms, dataset_pos



def process_raw_anndata(row, h5_folder_path, npz_folder_path, scp, skip,
                        additional_filter, root, protein_embeddings_dir=None):
        path = row.path
        if not os.path.isfile(root + "/" + path):
            print( "**********************************")
            print(f"***********{root + '/' + path} File Missing****")
            print( "**********************************")
            print(path, root)
            return None

        name = path.replace(".h5ad", "")
        proc_path = path.replace(".h5ad", "_proc.h5ad")
        if skip:
            if os.path.isfile(h5_folder_path + proc_path):
                print(f"{name} already processed. Skipping")
                return None, None, None

        print(f"Proccessing {name}")

        species = row.species
        covar_col = row.covar_col

        ad = anndata.read_h5ad(root + "/" + path)
        labels = []
        if "cell_type" in ad.obs.columns:
            labels.append("cell_type")


        if covar_col is np.nan or np.isnan(covar_col):
            covar_col = None
        else:
            labels.append(covar_col)


        dataset, adata = anndata_to_sc_dataset(ad, species=species, labels=labels, covar_col=covar_col, hv_genes=None, protein_embeddings_dir=protein_embeddings_dir)
        adata = adata.copy()
        
        num_cells = adata.X.shape[0]
        num_genes = adata.X.shape[1]

        adata_path = h5_folder_path + proc_path
        adata.write(adata_path)

        arr = data_to_torch_X(adata.X).numpy()

        print(arr.max()) # this is a nice check to make sure it's counts
        filename = npz_folder_path + f"{name}_counts.npz"
        shape = arr.shape
        print(name, shape)
        fp = np.memmap(filename, dtype='int64', mode='w+', shape=shape)
        fp[:] = arr[:]
        fp.flush()
        
        if scp != "":
            subprocess.call(["scp", filename, f"{scp}:{filename}"])
            subprocess.call(["scp", adata_path, f"{scp}:{adata_path}"])
            
        return adata, num_cells, num_genes
    
    
def get_species_to_pe(EMBEDDING_DIR):
    """
    Given an embedding directory, return all embeddings as a dictionary coded by species.
    Note: In the current form, this function is written such that the directory needs all of the following species embeddings.
    """
    EMBEDDING_DIR = Path(EMBEDDING_DIR)

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
    # Try to load extra species from various possible locations
    possible_csv_paths = [
        "./model_files/new_species_protein_embeddings.csv",
        Path.cwd() / "new_species_protein_embeddings.csv",
        EMBEDDING_DIR.parent / "new_species_protein_embeddings.csv"
    ]
    
    for csv_path in possible_csv_paths:
        try:
            if Path(csv_path).exists():
                extra_species = pd.read_csv(csv_path).set_index("species").to_dict()["path"]
                embeddings_paths.update(extra_species)  # adds new species
                break
        except Exception:
            continue  # Ignore errors and try next path
    
    
    
    species_to_pe = {
        species:torch.load(pe_dir) for species, pe_dir in embeddings_paths.items()   
    }

    species_to_pe = {species:{k.upper(): v for k,v in pe.items()} for species, pe in species_to_pe.items()}
    return species_to_pe


def get_spec_chrom_csv(path=None):
    """
    Get the species to chrom csv file
    """
    if path is None:
        cwd_csv = Path.cwd() / "all_to_chrom_pos.csv"
        if cwd_csv.exists():
            path = str(cwd_csv)
        else:
            raise FileNotFoundError(
                "all_to_chrom_pos.csv not found in current directory. Pass an "
                "explicit path= or place the file at ./all_to_chrom_pos.csv."
            )
    
    gene_to_chrom_pos = pd.read_csv(path)
    gene_to_chrom_pos["spec_chrom"] = pd.Categorical(gene_to_chrom_pos["species"] + "_" +  gene_to_chrom_pos["chromosome"]) # add the spec_chrom list
    return gene_to_chrom_pos
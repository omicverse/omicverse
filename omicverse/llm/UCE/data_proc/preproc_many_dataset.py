import os


from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.data as data
import numpy as np
from numpy import array
import subprocess
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


from .gene_embeddings import load_gene_embeddings_adata
import pandas as pd
import numpy as np
from anndata import AnnData
from .data_utils import process_raw_anndata

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
                                 hv_genes:int=12000,
                                 embedding_model="ESM1b",
                                ) -> (SincleCellDataset, AnnData):
    
    # Subset to just genes we have embeddings for
    adata, protein_embeddings = load_gene_embeddings_adata(
        adata=adata,
        species=[species],
        embedding_model=embedding_model
    )
    
    if DO_HVG:
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
    
def proc(args):
    datasets_df = pd.read_csv(args.datasets_df)
    datasets_df["covar_col"] = np.nan
    skip = args.skip
    additional_filter = args.filter
    DO_HVG = args.DO_HVG
    
    num_genes = {}
    num_cells = {}
    
    ir = list(datasets_df.iterrows())
    for i, row in tqdm(ir, total=len(datasets_df)):
        _, ncells, ngenes = process_raw_anndata(row, h5_folder_path, npz_folder_path, scp, skip, additional_filter, root=args.file_root_path)
        if (ncells is not None) and (ngenes is not None):
            num_genes[path] = adata.X.shape[1]
            num_cells[path] = ngenes
        
    if "num_cells" not in datasets_df.columns:
        datasets_df["num_cells"] = 0
    if "num_genes" not in datasets_df.columns:
        datasets_df["num_genes"] = 0
    for k in num_genes.keys():
        ng = num_genes[k]
        nc = num_cells[k]
        datasets_df.loc[datasets_df["path"] == k, "num_cells"] = nc
        datasets_df.loc[datasets_df["path"] == k, "num_genes"] = ng
    # Write with the cells and genes info back to the original path
    datasets_df.to_csv(args.datasets_df, index=False)
if __name__=="__main__":
    # Parse command-line arguments
    
    parser = argparse.ArgumentParser(description='Preproc datasets h5ad datasets.')

    # Define command-line arguments
    parser.add_argument('--scp', type=str, default="", help='Name of a SNAP server to SCP the results to. It should have the same folders as the script is already saving to.')
    parser.add_argument('--h5_folder_path', type=str, default="/lfs/local/0/yanay/uce_h5s/", help='Folder to save H5s to.')
    parser.add_argument('--npz_folder_path', type=str, default="/lfs/local/0/yanay/uce_proc/", help='Folder to save NPZs to.')
    
    
    parser.add_argument('--datasets_df', type=str, default="/dfs/project/uce/new_perturb_datasets.csv", help='Path to datasets csv. Will be overwritten to have the correct num cells and num genes for each dataset.')
    
    parser.add_argument('--filter', type=bool, default=True, help='Should you do an additional gene/cell filtering? This can be a good step since even if you have already done it, subsetting to protein embeddings can make some cells sparser.')
    parser.add_argument('--skip', type=bool, default=True, help='Should you skip datasets that appear to have already been created in the h5 folder?')
    
    parser.add_argument('--DO_HVG', type=bool, default=False, help='Should a HVG subset be done.')
    
    
    parse
    args = parser.parse_args()
    main(args)
    
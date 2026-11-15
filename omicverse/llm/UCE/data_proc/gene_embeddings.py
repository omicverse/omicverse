"""Helper functions for loading pretrained gene embeddings."""
from pathlib import Path
from typing import Dict, Tuple

import torch

from anndata import AnnData
import numpy as np
import pandas as pd


# Global variable that will be set by the UCE model wrapper
EMBEDDING_DIR = None

def _get_default_embedding_dir():
    """Get default embedding directory path."""
    return Path('model_files/protein_embeddings')

def _build_species_paths(embedding_dir):
    """Build species to gene embedding path mapping."""
    if embedding_dir is None:
        embedding_dir = _get_default_embedding_dir()
    
    return {
        'human': embedding_dir / 'Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt',
        'mouse': embedding_dir / 'Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt',
        'frog': embedding_dir / 'Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt',
        'zebrafish': embedding_dir / 'Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt',
        "mouse_lemur": embedding_dir / "Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt",
        "pig": embedding_dir / 'Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt',
        "macaca_fascicularis": embedding_dir / 'Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt',
        "macaca_mulatta": embedding_dir / 'Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt',
    }

# Initialize with default paths, can be updated by external code
# Use a lazy initialization approach to avoid hardcoded path issues
MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH = None

def _get_model_to_species_paths():
    """Get the model to species paths mapping, creating it if needed."""
    global MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH
    if MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH is None:
        # If EMBEDDING_DIR is still None, this means paths weren't set properly
        if EMBEDDING_DIR is None:
            raise RuntimeError("EMBEDDING_DIR not set. Call set_embedding_directory() first before using gene embeddings.")
        MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH = {
            'ESM2': _build_species_paths(EMBEDDING_DIR)
        }
    return MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH

def set_embedding_directory(new_embedding_dir):
    """Update the embedding directory and rebuild all paths."""
    global EMBEDDING_DIR, MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH
    EMBEDDING_DIR = Path(new_embedding_dir)
    MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH = {
        'ESM2': _build_species_paths(EMBEDDING_DIR)
    }

#extra_species = pd.read_csv("./model_files/new_species_protein_embeddings.csv").set_index("species").to_dict()["path"]
#MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH["ESM2"].update(extra_species) # adds new species


def load_gene_embeddings_adata(adata: AnnData, 
                               species: list, 
                               embedding_model: str
                               ) -> Tuple[AnnData, Dict[str, torch.FloatTensor]]:
    """Loads gene embeddings for all the species/genes in the provided data.

    :param data: An AnnData object containing gene expression data for cells.
    :param species: Species corresponding to this adata
    
    :param embedding_model: The gene embedding model whose embeddings will be loaded.
    :return: A tuple containing:
               - A subset of the data only containing the gene expression for genes with embeddings in all species.
               - A dictionary mapping species name to the corresponding gene embedding matrix (num_genes, embedding_dim).
    """
    # Get species names
    species_names = species
    species_names_set = set(species_names)

    # Get embedding paths for the model
    model_paths = _get_model_to_species_paths()
    species_to_gene_embedding_path = model_paths[embedding_model]
    available_species = set(species_to_gene_embedding_path)

    # Ensure embeddings are available for all species
    if not (species_names_set <= available_species):
        raise ValueError(f'The following species do not have gene embeddings: {species_names_set - available_species}')

    # Load gene embeddings for desired species (and convert gene symbols to lower case)
    species_to_gene_symbol_to_embedding = {
        species: {
            gene_symbol.lower(): gene_embedding
            for gene_symbol, gene_embedding in torch.load(species_to_gene_embedding_path[species]).items()
        }
        for species in species_names
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

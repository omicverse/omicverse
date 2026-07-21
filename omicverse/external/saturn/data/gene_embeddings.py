"""Helper functions for loading pretrained gene embeddings."""
from pathlib import Path
from typing import Dict, Tuple

import torch

from scanpy import AnnData
import numpy as np


EMBEDDING_DIR = Path('/dfs/project/cross-species/data/proteome/embeddings')
FZ_EMBEDDING_DIR = Path('/dfs/project/cross-species/yanay/data/proteome/embeddings')
MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH = {
    'ESM1b': {
        'human': EMBEDDING_DIR / 'Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM1b.pt',
        'mouse': EMBEDDING_DIR / 'Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM1b.pt',
        'frog': FZ_EMBEDDING_DIR / 'Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM1b.pt',
        #'frog': FZ_EMBEDDING_DIR / 'new_frog/xtropProtein.fasta.1.gene_symbol_to_embedding_ESM1b.pt',
        'zebrafish': FZ_EMBEDDING_DIR / 'Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM1b.pt',
        'bat':  FZ_EMBEDDING_DIR / 'Rhinolophus_ferrumequinum.mRhiFer1_v1.gene_symbol_to_embedding_ESM1b.pt',
        "mouse_lemur": FZ_EMBEDDING_DIR / "Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM1b.pt",
        "sea_squirt": FZ_EMBEDDING_DIR / 'Ciona_intestinalis.KH.gene_symbol_to_embedding_ESM1b.pt',
        "chicken": FZ_EMBEDDING_DIR / 'Gallus_gallus.GRCg6a.gene_symbol_to_embedding_ESM1b.pt',
        "fly": FZ_EMBEDDING_DIR / 'Drosophila_melanogaster.BDGP6.32.gene_symbol_to_embedding_ESM1b.pt',
        "pig": FZ_EMBEDDING_DIR / 'Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM1b.pt',
        "macaca_fascicularis": FZ_EMBEDDING_DIR / 'Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM1b.pt',
        "macaca_mulatta": FZ_EMBEDDING_DIR / 'Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM1b.pt',
        "rat": FZ_EMBEDDING_DIR / 'Rattus_norvegicus.mRatBN7.2.gene_symbol_to_embedding_ESM1b.pt',
        "tree_shrew": FZ_EMBEDDING_DIR / 'Tupaia_belangeri.TREESHREW.gene_symbol_to_embedding_ESM1b.pt',
        
    },
    'MSA1b': {
        'human': EMBEDDING_DIR / 'Homo_sapiens.GRCh38.gene_symbol_to_embedding_MSA1b.pt',
        'mouse': EMBEDDING_DIR / 'Mus_musculus.GRCm39.gene_symbol_to_embedding_MSA1b.pt'
    },
    "protXL": {
        'human': FZ_EMBEDDING_DIR / 'Homo_sapiens.GRCh38.gene_symbol_to_embedding_protxl.pt',
        'mouse': FZ_EMBEDDING_DIR / 'Mus_musculus.GRCm39.gene_symbol_to_embedding_protxl.pt',
        'frog': FZ_EMBEDDING_DIR / 'Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_protxl.pt',
        'zebrafish': FZ_EMBEDDING_DIR / 'Danio_rerio.GRCz11.gene_symbol_to_embedding_protxl.pt',
    },
    "ESM1b_protref": {
        'human': FZ_EMBEDDING_DIR / 'human.protref.gene_symbol_to_embedding_ESM1b.pt',
        'mouse': FZ_EMBEDDING_DIR / 'mouse.protref.gene_symbol_to_embedding_ESM1b.pt',
        'frog': FZ_EMBEDDING_DIR / 'frog.protref.gene_symbol_to_embedding_ESM1b.pt',
        'zebrafish': FZ_EMBEDDING_DIR / 'zebrafish.protref.gene_symbol_to_embedding_ESM1b.pt',
    },
    'ESM2': {
        'human': FZ_EMBEDDING_DIR / 'Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt',
        'mouse': FZ_EMBEDDING_DIR / 'Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt',
        'frog': FZ_EMBEDDING_DIR / 'Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt',
        'zebrafish': FZ_EMBEDDING_DIR / 'Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt',
        "mouse_lemur": FZ_EMBEDDING_DIR / "Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt",
        "pig": FZ_EMBEDDING_DIR / 'Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt',
        "macaca_fascicularis": FZ_EMBEDDING_DIR / 'Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt',
        "macaca_mulatta": FZ_EMBEDDING_DIR / 'Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt',
    },
}

def load_gene_embeddings_adata(adata: AnnData, species: list, embedding_model: str, embedding_path: str = None) -> Tuple[AnnData, Dict[str, torch.FloatTensor]]:
    """Loads gene embeddings for all the species/genes in the provided data.

    :param data: An AnnData object containing gene expression data for cells.
    :param species: Species corresponding to this adata
    :param embedding_path: path to gene embeddings to use for given adata, if blank, will use paths set in this file.
    
    :param embedding_model: The gene embedding model whose embeddings will be loaded.
    :return: A tuple containing:
               - A subset of the data only containing the gene expression for genes with embeddings in all species.
               - A dictionary mapping species name to the corresponding gene embedding matrix (num_genes, embedding_dim).
    """
    # Get species names
    species_names = species
    species_names_set = set(species_names)

    if embedding_path is None:
        # Get embedding paths for the model
        species_to_gene_embedding_path = MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH[embedding_model]
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
    else:
        # Load gene embeddings for desired embedding_path (and convert gene symbols to lower case)
        species_to_gene_symbol_to_embedding = {
            species: {
                gene_symbol.lower(): gene_embedding
                for gene_symbol, gene_embedding in torch.load(embedding_path).items()
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

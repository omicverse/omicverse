# ruff: noqa: F401
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")  # noqa # isort:skip

# Note: Geneformer dictionary files are not included in the package to reduce size.
# Users must provide their own paths to these files when initializing the tokenizer.
# Default paths point to None - users must specify custom paths.

# Placeholder paths - these will not work and require user to provide custom paths
GENE_MEDIAN_FILE = None
TOKEN_DICTIONARY_FILE = None  
ENSEMBL_DICTIONARY_FILE = None
ENSEMBL_MAPPING_FILE = None

GENE_MEDIAN_FILE_30M = None
TOKEN_DICTIONARY_FILE_30M = None
ENSEMBL_DICTIONARY_FILE_30M = None
ENSEMBL_MAPPING_FILE_30M = None

def get_default_file_paths(model_version="V1"):
    """
    Get suggested default file paths for Geneformer dictionary files.
    
    Args:
        model_version: "V1" for 104M model, "V2" for 30M model
        
    Returns:
        dict: Dictionary with suggested file paths
        
    Note:
        These are just suggested names. Users need to download the actual files
        from https://huggingface.co/ctheodoris/Geneformer and specify the paths.
    """
    if model_version == "V2":
        return {
            'gene_median_file': 'gene_median_dictionary_gc30M.pkl',
            'token_dictionary_file': 'token_dictionary_gc30M.pkl',
            'gene_mapping_file': 'ensembl_mapping_dict_gc30M.pkl'
        }
    else:
        return {
            'gene_median_file': 'gene_median_dictionary_gc104M.pkl', 
            'token_dictionary_file': 'token_dictionary_gc104M.pkl',
            'gene_mapping_file': 'ensembl_mapping_dict_gc104M.pkl'
        }

def show_download_instructions():
    """Show instructions for downloading Geneformer dictionary files."""
    
    instructions = """
📥 Geneformer Dictionary Files Download Instructions:

Since dictionary files are large, they are not included in the package.
Please download them separately:

1. From Hugging Face (recommended):
   git clone https://huggingface.co/ctheodoris/Geneformer
   # Files will be in the Geneformer/ directory

2. From official Geneformer repository:
   git clone https://github.com/ctheodoris/Geneformer.git
   
3. Required files for 104M model (V1):
   - gene_median_dictionary_gc104M.pkl
   - token_dictionary_gc104M.pkl  
   - ensembl_mapping_dict_gc104M.pkl
   
4. Required files for 30M model (V2):
   - gene_median_dictionary_gc30M.pkl
   - token_dictionary_gc30M.pkl
   - ensembl_mapping_dict_gc30M.pkl

5. Usage with custom paths:
   model = ov.llm.ModelFactory.create_model('geneformer')
   model.load_model('/path/to/model', 
       gene_median_file='/path/to/gene_median_dictionary_gc104M.pkl',
       token_dictionary_file='/path/to/token_dictionary_gc104M.pkl',
       gene_mapping_file='/path/to/ensembl_mapping_dict_gc104M.pkl')
"""
    print(instructions)

from . import (
    collator_for_classification,
    emb_extractor,
    in_silico_perturber,
    in_silico_perturber_stats,
    pretrainer,
    tokenizer,
)
from .collator_for_classification import (
    DataCollatorForCellClassification,
    DataCollatorForGeneClassification,
)
from .emb_extractor import EmbExtractor, get_embs
from .in_silico_perturber import InSilicoPerturber
from .in_silico_perturber_stats import InSilicoPerturberStats
from .pretrainer import GeneformerPretrainer
from .tokenizer import TranscriptomeTokenizer

from . import classifier  # noqa # isort:skip
from .classifier import Classifier  # noqa # isort:skip

from . import mtl_classifier  # noqa # isort:skip
from .mtl_classifier import MTLClassifier  # noqa # isort:skip

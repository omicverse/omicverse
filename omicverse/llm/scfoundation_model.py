"""
scFoundation model implementation with simplified interface.
"""

import copy
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from anndata import AnnData
from scipy.sparse import issparse

try:
    from .base import SCLLMBase, ModelConfig, TaskConfig
    from .utils.output_utils import SCLLMOutput, ModelProgressManager, operation_start, operation_complete
except ImportError:
    from base import SCLLMBase, ModelConfig, TaskConfig
    from utils.output_utils import SCLLMOutput, ModelProgressManager, operation_start, operation_complete

# Import scFoundation components with error handling
try:
    from .scfoundation.load import load_model_frommmf, gatherData, getEncoerDecoderData, main_gene_selection
    from .scfoundation.pretrainmodels.select_model import select_model
    _scfoundation_imports_available = True
except ImportError as e:
    import warnings
    warnings.warn(f"scFoundation components not available: {e}")
    _scfoundation_imports_available = False
    
    # Create placeholder functions
    def load_model_frommmf(*args, **kwargs):
        raise ImportError("scFoundation load_model_frommmf not available due to missing dependencies")
    
    def gatherData(*args, **kwargs):
        raise ImportError("scFoundation gatherData not available due to missing dependencies")
    
    def getEncoerDecoderData(*args, **kwargs):
        raise ImportError("scFoundation getEncoerDecoderData not available due to missing dependencies")
    
    def main_gene_selection(*args, **kwargs):
        raise ImportError("scFoundation main_gene_selection not available due to missing dependencies")
    
    def select_model(*args, **kwargs):
        raise ImportError("scFoundation select_model not available due to missing dependencies")


class ScFoundationModel(SCLLMBase):
    """
    Simplified scFoundation model interface.
    
    This class provides an easy-to-use interface for scFoundation model operations
    including loading, preprocessing, training, and inference.
    """
    
    def __init__(self, device: Optional[str] = None, seed: int = 0):
        """
        Initialize scFoundation model.
        
        Args:
            device: Device to run the model on
            seed: Random seed for reproducibility
        """
        super().__init__("scFoundation", device)
        self.seed = seed
        self._set_seed(seed)
        
        # Model components
        self.model = None
        self.config = None
        self.gene_list = None
        
        # Default parameters for scFoundation
        self.default_config = ModelConfig(
            seq_len=19266,  # scFoundation uses 19264 genes + 2 special tokens
            n_class=19266,  # Token vocabulary size
            pad_token_id=19264,  # Padding token ID
            mask_token_id=19265,  # Mask token ID
            bin_alpha=0.0,  # Binning parameter
            bin_num=51,  # Number of bins
            max_seq_len=19266,
            # Model architecture
            model="mae_autobin",
            encoder={
                "module_type": "performer",
                "hidden_dim": 512,
                "depth": 6,
                "heads": 8,
                "dim_head": 64,
                "ff_dropout": 0.1,
                "attn_dropout": 0.1
            },
            decoder={
                "module_type": "performer", 
                "hidden_dim": 512,
                "depth": 6,
                "heads": 8,
                "dim_head": 64,
                "ff_dropout": 0.1,
                "attn_dropout": 0.1
            },
            # Task-specific parameters
            output_type="cell",  # 'cell', 'gene', 'gene_batch', 'gene_expression'
            pool_type="all",  # 'all', 'max'
            input_type="singlecell",  # 'singlecell', 'bulk'
            pre_normalized="F",  # 'F', 'T', 'A'
            tgthighres="t4",  # Target high resolution
            version="ce",  # 'ce', 'rde', 'noversion'
        )
        
        # Load gene list
        self._load_gene_list()
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        import random
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _load_gene_list(self):
        """Load the gene list for scFoundation."""
        try:
            # Try to load gene list from the scFoundation directory
            gene_file = Path(__file__).parent / "scfoundation" / "OS_scRNA_gene_index.19264.tsv"
            if gene_file.exists():
                gene_list_df = pd.read_csv(gene_file, header=0, delimiter='\t')
                self.gene_list = list(gene_list_df['gene_name'])
                SCLLMOutput.status(f"Gene list loaded: {len(self.gene_list)} genes", 'loaded')
            else:
                SCLLMOutput.status(f"Gene list file not found at {gene_file}", 'warning')
                self.gene_list = None
        except Exception as e:
            SCLLMOutput.status(f"Could not load gene list: {e}", 'warning')
            self.gene_list = None
    
    def load_model(self, model_path: Union[str, Path], **kwargs) -> None:
        """
        Load a pre-trained scFoundation model.
        
        Args:
            model_path: Path to the model file (.ckpt)
            **kwargs: Additional parameters
                - key: Model key for multi-model checkpoints ('cell', 'gene', 'rde')
                - version: Model version ('ce', 'rde', 'noversion')
        """
        if not _scfoundation_imports_available:
            raise ImportError("Cannot load scFoundation model: required dependencies not available")
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist")
        
        # Get model parameters
        version = kwargs.get('version', self.default_config.version)
        key = kwargs.get('key', None)
        
        # Determine key based on version if not provided
        if key is None:
            if version == 'ce':
                key = 'cell'
            elif version == 'rde':
                key = 'rde'
            elif version == 'gene':
                key = 'gene'
            else:
                key = 'cell'  # default
        
        # Auto-detect key if the specified one doesn't exist
        def auto_detect_key(checkpoint_path, preferred_key):
            """Auto-detect the correct key for MMF checkpoints"""
            try:
                checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
                if isinstance(checkpoint, dict):
                    available_mmf_keys = [k for k in ['gene', 'cell', 'rde'] if k in checkpoint]
                    if available_mmf_keys:
                        if preferred_key in available_mmf_keys:
                            return preferred_key
                        else:
                            SCLLMOutput.status(f"Preferred key '{preferred_key}' not found, auto-selecting", 'warning')
                            return available_mmf_keys[0]  # Use first available
                return preferred_key
            except Exception:
                return preferred_key
        
        # Try to auto-detect the correct key
        detected_key = auto_detect_key(model_path, key)
        if detected_key != key:
            SCLLMOutput.status(f"Auto-detected key: {key} → {detected_key}", 'info')
            key = detected_key
        
        SCLLMOutput.status(f"Loading scFoundation model with key '{key}'", 'loading')
        
        # First, inspect the checkpoint to understand its format
        SCLLMOutput.status(f"Inspecting checkpoint format", 'preprocessing')
        try:
            checkpoint_preview = torch.load(str(model_path), map_location='cpu')
            if isinstance(checkpoint_preview, dict):
                # Checkpoint keys details removed for cleaner output
                
                # Check if it's MMF format
                if key in checkpoint_preview:
                    SCLLMOutput.status(f"Found '{key}' key - MMF format", 'loaded')
                    if isinstance(checkpoint_preview[key], dict):
                        sub_keys = list(checkpoint_preview[key].keys())
                        # Sub-keys details removed for cleaner output
                elif any(k in checkpoint_preview for k in ['gene', 'cell', 'rde']):
                    available_keys = [k for k in ['gene', 'cell', 'rde'] if k in checkpoint_preview]
                    SCLLMOutput.status(f"MMF format: '{key}' not found. Available: {available_keys}", 'warning')
                else:
                    SCLLMOutput.status(f"Not MMF format, trying standard loading", 'info')
            del checkpoint_preview  # Free memory
        except Exception as preview_e:
            SCLLMOutput.status(f"Could not preview checkpoint: {preview_e}", 'warning')
        
        try:
            # Try to load model using scFoundation's loader
            self.model, self.config = load_model_frommmf(str(model_path), key=key)
            
            # Update config with user parameters
            for k, v in kwargs.items():
                if k in self.config:
                    self.config[k] = v
            
            # Set model to evaluation mode
            self.model.eval()
            
            SCLLMOutput.status(f"scFoundation model loaded successfully", 'loaded')
            SCLLMOutput.status(f"Model type: {self.config.get('model_type', 'unknown')}", indent=1)
            
            self.is_loaded = True
            SCLLMOutput.status(f"Model ready on {self.device}", 'loaded')
            
        except KeyError as e:
            error_msg = str(e)
            if "not found in checkpoint" in error_msg:
                # This is our improved error message with available keys
                SCLLMOutput.status(error_msg, 'failed')
                
                # Extract available keys from error message if possible
                if "Available MMF keys:" in error_msg:
                    SCLLMOutput.status(f"Try different key: 'gene', 'cell', or 'rde'", 'info', indent=1)
                
                # Don't try fallback for key errors - user needs to specify correct key
                raise RuntimeError(f"Model loading failed: {error_msg}")
                
            elif str(e) == "'configs'":
                SCLLMOutput.status(f"Old format checkpoint, trying direct loading", 'warning')
                try:
                    # Try alternative loading method
                    model_data = torch.load(str(model_path), map_location=self.device)
                    
                    # Check if this is the right format
                    if key in model_data:
                        model_data = model_data[key]
                        
                        # Use default config if not available
                        self.config = self.default_config.to_dict()
                        
                        # Try to create model with default config
                        self.model = select_model(self.config)
                        
                        # Try to load state dict
                        if 'model_state_dict' in model_data:
                            self.model.load_state_dict(model_data['model_state_dict'])
                        else:
                            raise ValueError("No model_state_dict found in checkpoint")
                        
                        self.model.to(self.device)
                        self.model.eval()
                        self.is_loaded = True
                        
                        SCLLMOutput.status(f"Model loaded with default configuration", 'loaded')
                        SCLLMOutput.status(f"Using default parameters", 'warning')
                        
                    else:
                        raise ValueError(f"Key '{key}' not found in model checkpoint")
                        
                except Exception as fallback_e:
                    raise RuntimeError(f"Failed to load scFoundation model with both methods. "
                                     f"Original error: {e}, Fallback error: {fallback_e}")
            else:
                raise RuntimeError(f"Failed to load scFoundation model: {e}")
        except Exception as e:
            # Try a more direct loading approach
            SCLLMOutput.status(f"Standard loading failed, trying direct loading", 'warning')
            try:
                # Load checkpoint directly
                checkpoint = torch.load(str(model_path), map_location=self.device)
                
                # Debug: print checkpoint structure
                SCLLMOutput.status(f"Debugging checkpoint structure", 'preprocessing')
                # Checkpoint structure details removed for cleaner output
                
                # Try different loading strategies
                loaded_successfully = False
                
                # Strategy 1: Check if 'cell' key exists and contains model data
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    model_data = checkpoint[key]
                    SCLLMOutput.status(f"Found '{key}' data", 'info')
                    
                    # Look for state dict in different places
                    state_dict = None
                    if 'state_dict' in model_data:
                        state_dict = model_data['state_dict']
                        SCLLMOutput.status(f"Found state_dict", 'loaded', indent=1)
                    elif 'model_state_dict' in model_data:
                        state_dict = model_data['model_state_dict'] 
                        SCLLMOutput.status(f"Found model_state_dict", 'loaded', indent=1)
                    elif 'model' in model_data and isinstance(model_data['model'], dict):
                        state_dict = model_data['model']
                        SCLLMOutput.status(f"Found model dict", 'loaded', indent=1)
                    
                    if state_dict is not None:
                        # Use default configuration
                        self.config = self.default_config.to_dict()
                        SCLLMOutput.status(f"Using default configuration", 'info', indent=1)
                        
                        # Create model
                        self.model = select_model(self.config)
                        
                        try:
                            self.model.load_state_dict(state_dict)
                            self.model.to(self.device)
                            self.model.eval()
                            self.is_loaded = True
                            loaded_successfully = True
                            SCLLMOutput.status(f"Model loaded with Strategy 1", 'loaded')
                        except Exception as load_e:
                            SCLLMOutput.status(f"Strategy 1 failed: {load_e}", 'warning')
                
                # Strategy 2: Try to load entire checkpoint as state dict
                if not loaded_successfully:
                    SCLLMOutput.status(f"Trying Strategy 2: entire checkpoint", 'preprocessing')
                    try:
                        self.config = self.default_config.to_dict()
                        self.model = select_model(self.config)
                        self.model.load_state_dict(checkpoint)
                        self.model.to(self.device)
                        self.model.eval()
                        self.is_loaded = True
                        loaded_successfully = True
                        SCLLMOutput.status(f"Model loaded with Strategy 2", 'loaded')
                    except Exception as load_e:
                        SCLLMOutput.status(f"Strategy 2 failed: {load_e}", 'warning')
                
                # Strategy 3: Try to find any dict that looks like a state dict
                if not loaded_successfully:
                    SCLLMOutput.status(f"Trying Strategy 3: searching patterns", 'preprocessing')
                    for top_key, top_value in checkpoint.items():
                        if isinstance(top_value, dict):
                            # Look for common model parameter patterns
                            param_like_keys = [k for k in top_value.keys() if any(
                                pattern in k for pattern in ['weight', 'bias', 'embedding', 'linear', 'layer']
                            )]
                            
                            if len(param_like_keys) > 5:  # Likely a state dict
                                SCLLMOutput.status(f"Found potential state dict in '{top_key}'", 'info')
                                try:
                                    self.config = self.default_config.to_dict()
                                    self.model = select_model(self.config)
                                    self.model.load_state_dict(top_value)
                                    self.model.to(self.device)
                                    self.model.eval()
                                    self.is_loaded = True
                                    loaded_successfully = True
                                    SCLLMOutput.status(f"Model loaded from '{top_key}' with Strategy 3", 'loaded')
                                    break
                                except Exception as load_e:
                                    SCLLMOutput.status(f"Strategy 3 with '{top_key}' failed: {load_e}", 'warning')
                
                if not loaded_successfully:
                    # Final attempt: print more detailed error information
                    SCLLMOutput.status(f"All loading strategies failed", 'failed')
                    
                    
                    if isinstance(checkpoint, dict):
                        print_dict_structure(checkpoint)
                    
                    raise ValueError("Could not find a valid state dict in the checkpoint. "
                                   "Please check the model file format.")
                    
            except Exception as direct_e:
                raise RuntimeError(f"All loading methods failed. "
                                 f"Original error: {e}, Direct loading error: {direct_e}")
        
        if self.is_loaded:
            SCLLMOutput.status(f"Model ready for inference on {self.device}", 'loaded')
    
    def preprocess(self, adata: AnnData, **kwargs) -> AnnData:
        """
        Preprocess data for scFoundation.
        
        Args:
            adata: Input AnnData object
            **kwargs: Preprocessing parameters
                - pre_normalized: 'F', 'T', or 'A' (default: 'F')
                - input_type: 'singlecell' or 'bulk' (default: 'singlecell')
                
        Returns:
            Preprocessed AnnData object
        """
        adata_processed = adata.copy()
        
        # Get preprocessing parameters
        pre_normalized = kwargs.get('pre_normalized', 'F')
        input_type = kwargs.get('input_type', 'singlecell')
        
        SCLLMOutput.status(f"Preprocessing data for scFoundation", 'preprocessing')
        SCLLMOutput.status(f"Input: {adata_processed.shape}, Type: {input_type}", 'info')
        
        # Step 1: Gene selection and padding to match scFoundation's gene set
        if self.gene_list is not None:
            SCLLMOutput.status(f"Filtering genes to match scFoundation gene set", 'preprocessing')
            
            # Convert to DataFrame for gene selection
            if issparse(adata_processed.X):
                X_df = pd.DataFrame(adata_processed.X.toarray(), 
                                  index=adata_processed.obs_names,
                                  columns=adata_processed.var_names)
            else:
                X_df = pd.DataFrame(adata_processed.X,
                                  index=adata_processed.obs_names, 
                                  columns=adata_processed.var_names)
            
            # Use scFoundation's gene selection function
            X_df_processed, to_fill_columns = main_gene_selection(X_df, self.gene_list)
            
            # Update adata with processed data
            adata_processed = AnnData(
                X=X_df_processed.values,
                obs=adata_processed.obs.copy(),
                var=pd.DataFrame(index=X_df_processed.columns)
            )
            
            # Add mask information
            adata_processed.var['mask'] = [1 if gene in to_fill_columns else 0 
                                         for gene in adata_processed.var_names]
            
            genes_matched = (adata_processed.var['mask'] == 0).sum()
            genes_padded = (adata_processed.var['mask'] == 1).sum()
            
            SCLLMOutput.status(f"Gene matching: {genes_matched} matched, {genes_padded} padded", 'loaded')
            
            if genes_matched == 0:
                raise ValueError("No genes matched the scFoundation gene set! Check gene naming conventions.")
        
        # Step 2: Normalization based on input type and pre_normalized flag
        if input_type == 'singlecell':
            if pre_normalized == 'F':
                # Raw counts - need normalization
                SCLLMOutput.status(f"Applying normalization and log transformation", 'preprocessing')
                from omicverse.pp import normalize_total, log1p
                normalize_total(adata_processed, target_sum=1e4)
                log1p(adata_processed)
            elif pre_normalized == 'T':
                # Already normalized and log-transformed
                SCLLMOutput.status(f"Data already normalized and log-transformed", 'info')
            elif pre_normalized == 'A':
                # Special case - data normalized with total count appended
                SCLLMOutput.status(f"Data pre-normalized with total count appended", 'info')
            else:
                raise ValueError("pre_normalized must be 'F', 'T', or 'A'")
                
        elif input_type == 'bulk':
            if pre_normalized == 'F':
                # Raw bulk data - apply normalization
                SCLLMOutput.status(f"Applying bulk data normalization", 'preprocessing')
                from omicverse.pp import normalize_total, log1p
                normalize_total(adata_processed)
                log1p(adata_processed)
            elif pre_normalized == 'T':
                # Already log10 normalized
                SCLLMOutput.status(f"Bulk data already normalized", 'info')
            else:
                raise ValueError("For bulk data, pre_normalized must be 'F' or 'T'")
        else:
            raise ValueError("input_type must be 'singlecell' or 'bulk'")
        
        # Step 3: Add total count information for scFoundation
        if input_type == 'singlecell' and pre_normalized != 'A':
            # Calculate total counts for each cell
            if issparse(adata_processed.X):
                total_counts = np.array(adata_processed.X.sum(axis=1)).flatten()
            else:
                total_counts = adata_processed.X.sum(axis=1)
            
            adata_processed.obs['total_count'] = total_counts
            SCLLMOutput.status(f"Added total count info (mean: {total_counts.mean():.2f})", 'loaded')
        
        SCLLMOutput.status(f"Preprocessing completed: {adata_processed.shape}", 'loaded')
        
        return adata_processed
    
    def predict(self, adata: AnnData, task: str = "embedding", **kwargs) -> Dict[str, Any]:
        """
        Make predictions using the scFoundation model.
        
        Args:
            adata: Input AnnData object
            task: Task type ('embedding', 'annotation', 'integration', 'generation')
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing predictions and metadata
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess data
        adata_processed = self.preprocess(adata, **kwargs)
        
        if task == "embedding":
            return self._predict_embedding(adata_processed, **kwargs)
        elif task == "annotation":
            return self._predict_annotation(adata_processed, **kwargs)
        elif task == "integration":
            return self._predict_integration(adata_processed, **kwargs)
        elif task == "generation":  
            return self._predict_generation(adata_processed, **kwargs)
        else:
            raise ValueError(f"Unknown task: {task}. Available tasks: embedding, annotation, integration, generation")
    
    def _predict_embedding(self, adata: AnnData, **kwargs) -> Dict[str, Any]:
        """Get cell embeddings using scFoundation."""
        output_type = kwargs.get('output_type', 'cell')
        pool_type = kwargs.get('pool_type', 'all')
        tgthighres = kwargs.get('tgthighres', 't4')
        input_type = kwargs.get('input_type', 'singlecell')
        pre_normalized = kwargs.get('pre_normalized', 'T')  # Assume preprocessed
        
        SCLLMOutput.status(f"Generating {output_type} embeddings...", 'embedding', indent=1)
        
        # Get expression data
        if issparse(adata.X):
            gexpr_data = adata.X.toarray()
        else:
            gexpr_data = adata.X
        
        SCLLMOutput.status(f"Data shape: {gexpr_data.shape}", indent=1)
        
        all_embeddings = []
        batch_container = []
        
        # Process each cell
        SCLLMOutput.status(f"Processing cells...", 'preprocessing', indent=1)
        pbar = SCLLMOutput.progress_bar(
            total=adata.n_obs, 
            desc="Cell embeddings", 
            model_name="scFoundation"
        )
        
        for i in range(adata.n_obs):
            pbar.update(1)
            with torch.no_grad():
                if input_type == 'singlecell':
                    # Single cell processing
                    if pre_normalized == 'F':
                        # Normalize per cell
                        cell_data = gexpr_data[i, :]
                        cell_sum = cell_data.sum()
                        tmpdata = np.log1p(cell_data / cell_sum * 1e4).tolist()
                    elif pre_normalized == 'T':
                        tmpdata = gexpr_data[i, :].tolist()
                    elif pre_normalized == 'A':
                        tmpdata = gexpr_data[i, :-1].tolist()
                    else:
                        raise ValueError('pre_normalized must be T, F or A')
                    
                    # Calculate total count
                    if pre_normalized == 'A':
                        total_count = gexpr_data[i, -1]
                    else:
                        total_count = gexpr_data[i, :].sum()
                    
                    # Add resolution tokens based on tgthighres
                    if tgthighres[0] == 'f':
                        pretrain_gene_x = torch.tensor(
                            tmpdata + [np.log10(total_count * float(tgthighres[1:])), np.log10(total_count)]
                        ).unsqueeze(0).to(self.device)
                    elif tgthighres[0] == 'a':
                        pretrain_gene_x = torch.tensor(
                            tmpdata + [np.log10(total_count) + float(tgthighres[1:]), np.log10(total_count)]
                        ).unsqueeze(0).to(self.device)
                    elif tgthighres[0] == 't':
                        pretrain_gene_x = torch.tensor(
                            tmpdata + [float(tgthighres[1:]), np.log10(total_count)]
                        ).unsqueeze(0).to(self.device)
                    else:
                        raise ValueError('tgthighres must start with f, a or t')
                        
                elif input_type == 'bulk':
                    # Bulk processing
                    if pre_normalized == 'T':
                        total_count = gexpr_data[i, :].sum()
                    elif pre_normalized == 'F':
                        total_count = np.log10(gexpr_data[i, :].sum())
                    else:
                        raise ValueError('For bulk data, pre_normalized must be T or F')
                    
                    tmpdata = gexpr_data[i, :].tolist()
                    pretrain_gene_x = torch.tensor(tmpdata + [total_count, total_count]).unsqueeze(0).to(self.device)
                else:
                    raise ValueError('input_type must be singlecell or bulk')
                
                # Create gene IDs
                data_gene_ids = torch.arange(pretrain_gene_x.shape[1], device=self.device).repeat(pretrain_gene_x.shape[0], 1)
                
                # Gather data for model input
                value_labels = pretrain_gene_x > 0
                x, x_padding = gatherData(pretrain_gene_x, value_labels, self.config['pad_token_id'])
                
                if output_type == 'cell':
                    # Cell embedding
                    position_gene_ids, _ = gatherData(data_gene_ids, value_labels, self.config['pad_token_id'])
                    
                    # Token embedding
                    x = self.model.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
                    
                    # Position embedding
                    position_emb = self.model.pos_emb(position_gene_ids)
                    x += position_emb
                    
                    # Encoder forward
                    geneemb = self.model.encoder(x, x_padding)
                    
                    # Pool embeddings
                    geneemb1 = geneemb[:, -1, :]  # Last token
                    geneemb2 = geneemb[:, -2, :]  # Second to last token
                    geneemb3, _ = torch.max(geneemb[:, :-2, :], dim=1)  # Max pooling
                    geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)  # Mean pooling
                    
                    if pool_type == 'all':
                        geneembmerge = torch.concat([geneemb1, geneemb2, geneemb3, geneemb4], axis=1)
                    elif pool_type == 'max':
                        geneembmerge, _ = torch.max(geneemb, dim=1)
                    else:
                        raise ValueError('pool_type must be all or max')
                    
                    all_embeddings.append(geneembmerge.detach().cpu().numpy())
                
                elif output_type == 'gene_batch':
                        # Process all cells in batch
                        batch_container = torch.concat(batch_container, axis=0)
                        self.model.to_final = None
                        
                        encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, \
                        decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, \
                        decoder_position_gene_ids = getEncoerDecoderData(
                            batch_container, batch_container, self.config
                        )
                        
                        out = self.model.forward(
                            x=encoder_data, 
                            padding_label=encoder_data_padding,
                            encoder_position_gene_ids=encoder_position_gene_ids,
                            encoder_labels=encoder_labels,
                            decoder_data=decoder_data,
                            mask_gene_name=False,
                            mask_labels=None,
                            decoder_position_gene_ids=decoder_position_gene_ids,
                            decoder_data_padding_labels=decoder_data_padding,
                        )
                        
                        all_embeddings = out[:, :19264, :].contiguous().detach().cpu().numpy()
                        break
                
                elif output_type == 'gene':
                    # Gene embedding for single cell
                    self.model.to_final = None
                    
                    encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, \
                    decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, \
                    decoder_position_gene_ids = getEncoerDecoderData(
                        pretrain_gene_x.float(), pretrain_gene_x.float(), self.config
                    )
                    
                    out = self.model.forward(
                        x=encoder_data,
                        padding_label=encoder_data_padding,
                        encoder_position_gene_ids=encoder_position_gene_ids,
                        encoder_labels=encoder_labels,
                        decoder_data=decoder_data,
                        mask_gene_name=False,
                        mask_labels=None,
                        decoder_position_gene_ids=decoder_position_gene_ids,
                        decoder_data_padding_labels=decoder_data_padding,
                    )
                    
                    out = out[:, :19264, :].contiguous()
                    all_embeddings.append(out.detach().cpu().numpy())
        
        pbar.close()
        
        # Format results
        if output_type == 'gene_batch':
            embeddings = all_embeddings
        else:
            embeddings = np.squeeze(np.array(all_embeddings))
        
        SCLLMOutput.status(f"Generated embeddings: {embeddings.shape}", indent=1)
        
        return {
            "embeddings": embeddings,
            "output_type": output_type,
            "pool_type": pool_type if output_type == 'cell' else None,
        }
    
    def _predict_annotation(self, adata: AnnData, **kwargs) -> Dict[str, Any]:
        """Predict cell type annotations using fine-tuned model or embeddings."""
        
        # Check if fine-tuned model is available
        if hasattr(self, 'finetune_model') and hasattr(self, 'celltype_mapping'):
            SCLLMOutput.status(f"Using fine-tuned model for annotation...", 'predicting', indent=1)
            return self.predict_with_finetune(adata, **kwargs)
        else:
            SCLLMOutput.status(f"No fine-tuned model available", 'warning', indent=1)
            SCLLMOutput.status(f"Use fine_tune() first for cell type annotation", indent=2)
            
            result = self._predict_embedding(adata, **kwargs)
            
            # Add placeholder predictions
            n_cells = adata.n_obs
            predictions = np.zeros(n_cells, dtype=int)  # Placeholder
            
            result.update({
                "predictions": predictions,
                "predicted_celltypes": ["Unknown"] * n_cells,
                "note": "Use fine_tune() first for cell type annotation, or train a classifier on the embeddings."
            })
            
            return result
    
    def _predict_integration(self, adata: AnnData, **kwargs) -> Dict[str, Any]:
        """
        Perform batch integration using scFoundation embeddings with post-hoc correction.
        
        Args:
            adata: Input AnnData with batch information
            **kwargs: Integration parameters
                - batch_key: Column name for batch labels (default: 'batch')
                - correction_method: Integration method ('harmony', 'combat', 'scanorama', 'mnn')
                - output_type: Type of embeddings for integration (default: 'cell')
                
        Returns:
            Dictionary with integrated embeddings and metadata
        """
        batch_key = kwargs.get('batch_key', 'batch')
        correction_method = kwargs.get('correction_method', 'harmony')
        
        # Check for batch information
        if batch_key not in adata.obs:
            raise ValueError(f"Batch information '{batch_key}' not found in adata.obs. "
                           f"Integration requires batch labels.")
        
        SCLLMOutput.status(f"Starting scFoundation batch integration", 'integrating')
        SCLLMOutput.status(f"Batch key: '{batch_key}', Method: '{correction_method}'", 'info')
        
        # Get unique batches
        batch_labels = adata.obs[batch_key].astype('category')
        unique_batches = batch_labels.cat.categories
        batch_codes = batch_labels.cat.codes.values
        
        SCLLMOutput.status(f"Found {len(unique_batches)} batches", 'info')
        
        # Extract scFoundation embeddings
        SCLLMOutput.status(f"Extracting scFoundation embeddings", 'embedding')
        embedding_result = self._predict_embedding(adata, **kwargs)
        embeddings = embedding_result["embeddings"]
        
        SCLLMOutput.status(f"Embeddings: {embeddings.shape}", 'loaded')
        
        # Apply post-hoc batch correction
        if correction_method == 'harmony':
            integrated_embeddings = self._apply_harmony_correction(embeddings, batch_codes, **kwargs)
        elif correction_method == 'combat':
            integrated_embeddings = self._apply_combat_correction(embeddings, batch_codes, **kwargs)
        elif correction_method == 'scanorama':
            integrated_embeddings = self._apply_scanorama_correction(embeddings, batch_codes, unique_batches, **kwargs)
        elif correction_method == 'mnn':
            integrated_embeddings = self._apply_mnn_correction(embeddings, batch_codes, unique_batches, **kwargs)
        else:
            raise ValueError(f"Unknown correction method: {correction_method}. "
                           f"Available methods: harmony, combat, scanorama, mnn")
        
        SCLLMOutput.status(f"Integration completed using {correction_method}", 'loaded')
        
        return {
            "embeddings": integrated_embeddings,
            "original_embeddings": embeddings,
            "batch_key": batch_key,
            "correction_method": correction_method,
            "n_batches": len(unique_batches),
            "batch_labels": unique_batches.tolist()
        }
    
    def _apply_harmony_correction(self, embeddings: np.ndarray, batch_codes: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Harmony correction to embeddings."""
        try:
            import harmonypy as hm
        except ImportError:
            raise ImportError("harmonypy is required for Harmony correction. Install with: pip install harmonypy")
        
        SCLLMOutput.status(f"Applying Harmony correction", 'preprocessing')
        
        # Create metadata DataFrame for Harmony
        meta_data = pd.DataFrame({'batch': batch_codes})
        
        # Apply Harmony correction
        # Extract specific harmony parameters to avoid conflicts
        harmony_params = {
            'max_iter_harmony': kwargs.get('max_iter_harmony', 10),
            'random_state': kwargs.get('random_state', 42)
        }
        
        harmony_out = hm.run_harmony(
            embeddings.T,  # Harmony expects genes x cells
            meta_data,
            vars_use=['batch'],
            **harmony_params
        )
        
        # Return corrected embeddings (cells x features)
        return harmony_out.Z_corr.T
    
    def _apply_combat_correction(self, embeddings: np.ndarray, batch_codes: np.ndarray, **kwargs) -> np.ndarray:
        """Apply ComBat correction to embeddings."""
        try:
            import pandas as pd
            from combat.pycombat import pycombat
        except ImportError:
            raise ImportError("pycombat is required for ComBat correction. Install with: pip install combat")
        
        SCLLMOutput.status(f"Applying ComBat correction", 'preprocessing')
        
        # Convert to DataFrame
        df_embeddings = pd.DataFrame(embeddings.T)  # ComBat expects features x samples
        
        # Apply ComBat correction
        corrected = pycombat(df_embeddings, batch_codes)
        
        return corrected.T  # Return as cells x features
    
    def _apply_scanorama_correction(self, embeddings: np.ndarray, batch_codes: np.ndarray, 
                                  unique_batches, **kwargs) -> np.ndarray:
        """Apply Scanorama correction to embeddings."""
        try:
            import scanorama
        except ImportError:
            raise ImportError("scanorama is required for Scanorama correction. Install with: pip install scanorama")
        
        SCLLMOutput.status(f"Applying Scanorama correction", 'preprocessing')
        
        # Split embeddings by batch
        batch_embeddings = []
        for batch in range(len(unique_batches)):
            batch_mask = batch_codes == batch
            batch_embeddings.append(embeddings[batch_mask])
        
        # Scanorama integration
        integrated, genes = scanorama.integrate(
            batch_embeddings,
            [f"batch_{i}" for i in range(len(unique_batches))]
        )
        
        # Reconstruct full embedding matrix
        integrated_embeddings = np.zeros_like(embeddings)
        start_idx = 0
        
        for i, batch_emb in enumerate(integrated):
            batch_size = batch_emb.shape[0]
            batch_mask = batch_codes == i
            integrated_embeddings[batch_mask] = batch_emb
            start_idx += batch_size
        
        return integrated_embeddings
    
    def _apply_mnn_correction(self, embeddings: np.ndarray, batch_codes: np.ndarray, 
                            unique_batches, **kwargs) -> np.ndarray:
        """Apply MNN (Mutual Nearest Neighbors) correction to embeddings."""
        try:
            import mnnpy
        except ImportError as e:
            raise ImportError(
                "mnnpy is required for MNN correction "
                "(https://github.com/chriscainx/mnnpy)."
            ) from e
        from anndata import AnnData

        SCLLMOutput.status(f"Applying MNN correction", 'preprocessing')

        # mnnpy corrects one AnnData per batch and returns the corrected,
        # concatenated object (batches in the order given).
        mnn_kwargs = {k: v for k, v in kwargs.items() if k not in ['batch_key', 'correction_method', 'pre_normalized']}
        batch_codes = np.asarray(batch_codes)
        codes = [i for i in range(len(unique_batches)) if np.any(batch_codes == i)]
        datas = [AnnData(X=np.asarray(embeddings[batch_codes == i])) for i in codes]
        corrected = mnnpy.mnn_correct(
            *datas,
            batch_key='batch',
            batch_categories=[f"batch_{i}" for i in codes],
            do_concatenate=True,
            **mnn_kwargs
        )[0]
        corrected_X = corrected.X.toarray() if issparse(corrected.X) else np.asarray(corrected.X)

        integrated = np.empty_like(np.asarray(embeddings, dtype=corrected_X.dtype))
        start = 0
        for i in codes:
            mask = batch_codes == i
            n = int(mask.sum())
            integrated[mask] = corrected_X[start:start + n]
            start += n
        return integrated
    
    def _predict_generation(self, adata: AnnData, **kwargs) -> Dict[str, Any]:
        """Generate gene expression (placeholder)."""
        SCLLMOutput.status(f"Gene expression generation not implemented for scFoundation", 'warning')
        return {
            "generated_expression": None,
            "note": "Generation functionality not implemented"
        }
    
    def fine_tune(self, 
                  train_adata: AnnData,
                  valid_adata: Optional[AnnData] = None,
                  task: str = "annotation",
                  **kwargs) -> Dict[str, Any]:
        """
        Fine-tune the scFoundation model using Linear Probing.
        
        Args:
            train_adata: Training data with labels in .obs['celltype']
            valid_adata: Validation data (optional)
            task: Task type (currently supports 'annotation')
            **kwargs: Training parameters
                - epochs: Number of training epochs (default: 10)
                - batch_size: Batch size (default: 32)
                - lr: Learning rate (default: 1e-3)
                - frozen_more: Whether to freeze token and position embeddings (default: True)
                - n_classes: Number of classes (auto-detected if not provided)
                
        Returns:
            Training results and metrics
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if task != "annotation":
            raise ValueError("Currently only 'annotation' task is supported for fine-tuning")
        
        if 'celltype' not in train_adata.obs:
            raise ValueError("train_adata must have 'celltype' column in .obs")
        
        SCLLMOutput.status(f"Starting scFoundation fine-tuning for {task}", 'fine_tuning')
        
        # Get training parameters
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 32)
        lr = kwargs.get('lr', 1e-3)
        frozen_more = kwargs.get('frozen_more', True)
        
        # Prepare cell type mapping
        unique_celltypes = train_adata.obs['celltype'].astype('category').cat.categories
        celltype_to_id = {ct: i for i, ct in enumerate(unique_celltypes)}
        id_to_celltype = {i: ct for i, ct in enumerate(unique_celltypes)}
        n_classes = kwargs.get('n_classes', len(unique_celltypes))
        
        SCLLMOutput.status(f"Found {n_classes} cell types", 'info')
        
        # Add celltype_id to data
        train_adata.obs['celltype_id'] = train_adata.obs['celltype'].map(celltype_to_id)
        if valid_adata is not None:
            if 'celltype' not in valid_adata.obs:
                raise ValueError("valid_adata must have 'celltype' column in .obs")
            valid_adata.obs['celltype_id'] = valid_adata.obs['celltype'].map(celltype_to_id)
        
        # Preprocess data
        train_processed = self.preprocess(train_adata, **kwargs)
        valid_processed = self.preprocess(valid_adata, **kwargs) if valid_adata is not None else None
        
        # Create fine-tuning model
        finetune_model = LinearProbingClassifier(
            pretrained_model=self.model,
            config=self.config,
            n_classes=n_classes,
            frozen_more=frozen_more
        )
        finetune_model.to(self.device)
        
        SCLLMOutput.status(f"Fine-tuning model created: {n_classes} classes", 'loaded')
        
        # Prepare training data
        train_dataset = self._prepare_finetune_data(train_processed)
        valid_dataset = self._prepare_finetune_data(valid_processed) if valid_processed is not None else None
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False) if valid_dataset else None
        
        # Setup training
        optimizer = torch.optim.Adam(finetune_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        best_accuracy = 0.0
        best_model_state = None
        training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # Create progress bar for epochs  
        progress_bar = SCLLMOutput.progress_bar(
            total=epochs,
            desc="Fine-tuning epochs",
            model_name="scFoundation"
        )
        
        for epoch in range(epochs):
            progress_bar.update(1)
            # Training
            train_loss, train_acc = self._train_finetune_epoch(
                finetune_model, train_loader, optimizer, criterion
            )
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            
            # Validation
            if valid_loader is not None:
                val_loss, val_acc = self._validate_finetune_epoch(
                    finetune_model, valid_loader, criterion
                )
                training_history['val_loss'].append(val_loss)
                training_history['val_acc'].append(val_acc)
                
                # Display epoch results
                progress_bar.set_description(
                    f"Fine-tuning epochs - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
                
                # Save best model
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    best_model_state = copy.deepcopy(finetune_model.state_dict())
                    SCLLMOutput.status(f"New best validation accuracy: {best_accuracy:.4f}", 'best')
            else:
                # Display epoch results (training only)
                progress_bar.set_description(
                    f"Fine-tuning epochs - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                )
                
                if train_acc > best_accuracy:
                    best_accuracy = train_acc
                    best_model_state = copy.deepcopy(finetune_model.state_dict())
            
            scheduler.step()
        
        progress_bar.close()
        
        # Load best model
        if best_model_state is not None:
            finetune_model.load_state_dict(best_model_state)
            SCLLMOutput.status(f"Best model loaded: accuracy={best_accuracy:.4f}", 'loaded')
        
        # Store the fine-tuned model and mapping
        self.finetune_model = finetune_model
        self.celltype_mapping = {
            'celltype_to_id': celltype_to_id,
            'id_to_celltype': id_to_celltype,
            'n_classes': n_classes
        }
        
        SCLLMOutput.status(f"Fine-tuning completed", 'complete')
        
        return {
            'best_accuracy': best_accuracy,
            'training_history': training_history,
            'celltype_mapping': self.celltype_mapping,
            'n_classes': n_classes,
            'finetune_model': finetune_model
        }
    
    def get_embeddings(self, adata: AnnData, **kwargs) -> np.ndarray:
        """
        Get cell embeddings.
        
        Args:
            adata: Input AnnData object
            **kwargs: Additional parameters
            
        Returns:
            Cell embeddings
        """
        # Start embedding extraction with unified output
        SCLLMOutput.data_summary(adata, model_name="scFoundation")
        operation_start("get_embeddings", "scFoundation", {
            "cells": f"{adata.n_obs:,}",
            "genes": f"{adata.n_vars:,}"
        })
        
        result = self.predict(adata, task="embedding", **kwargs)
        
        operation_complete("get_embeddings", {
            "embedding_shape": f"{result['embeddings'].shape}",
            "embedding_dim": result['embeddings'].shape[1]
        })
        
        return result["embeddings"]
    
    def _save_model_specific(self, save_path: Path, **kwargs) -> None:
        """Save scFoundation model."""
        if self.model is not None:
            # Save model state dict
            model_dict = {
                'model_state_dict': self.model.state_dict(),
                'configs': self.config
            }
            torch.save(model_dict, save_path / "model.ckpt")
        
        if self.config is not None:
            with open(save_path / "config.json", 'w') as f:
                json.dump(self.config, f, indent=2)
        
        SCLLMOutput.status(f"scFoundation model saved", 'loaded')
    
    def _prepare_finetune_data(self, adata: AnnData):
        """Prepare data for fine-tuning."""
        return FineTuneDataset(adata)
    
    def _train_finetune_epoch(self, model, dataloader: DataLoader, 
                             optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        """Train fine-tuning model for one epoch."""
        model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_data in dataloader:
            x = batch_data['x'].to(self.device)
            targets = batch_data['targets'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(x, targets)
            loss = criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = logits.argmax(1)
            total_correct += (predictions == targets).sum().item()
            total_samples += targets.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def _validate_finetune_epoch(self, model, dataloader: DataLoader, 
                                criterion: nn.Module) -> Tuple[float, float]:
        """Validate fine-tuning model for one epoch."""
        model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                x = batch_data['x'].to(self.device)
                targets = batch_data['targets'].to(self.device)
                
                # Forward pass
                logits = model(x, targets)
                loss = criterion(logits, targets)
                
                # Statistics
                total_loss += loss.item()
                predictions = logits.argmax(1)
                total_correct += (predictions == targets).sum().item()
                total_samples += targets.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def predict_with_finetune(self, adata: AnnData, **kwargs) -> Dict[str, Any]:
        """
        Predict cell types using fine-tuned model.
        
        Args:
            adata: Input AnnData object
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if not hasattr(self, 'finetune_model'):
            raise ValueError("No fine-tuned model available. Call fine_tune() first.")
        
        if not hasattr(self, 'celltype_mapping'):
            raise ValueError("No cell type mapping available. Call fine_tune() first.")
        
        SCLLMOutput.status(f"Predicting cell types using scFoundation", 'predicting')
        
        # Preprocess data
        adata_processed = self.preprocess(adata, **kwargs)
        
        # Create dataset for prediction (no labels needed)
        dataset = PredictionDataset(adata_processed)
        dataloader = DataLoader(dataset, batch_size=kwargs.get('batch_size', 32), shuffle=False)
        
        # Make predictions
        self.finetune_model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_data in dataloader:
                x = batch_data['x'].to(self.device)
                
                logits = self.finetune_model(x)
                probabilities = torch.softmax(logits, dim=1)
                predictions = logits.argmax(1)
                
                all_predictions.append(predictions.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
        
        # Combine results
        predictions = np.concatenate(all_predictions)
        probabilities = np.concatenate(all_probabilities)
        
        # Convert IDs to cell type names
        id_to_celltype = self.celltype_mapping['id_to_celltype']
        predicted_celltypes = [id_to_celltype.get(pred, f"Unknown_{pred}") 
                             for pred in predictions]
        
        # Show prediction summary
        from collections import Counter
        type_counts = Counter(predicted_celltypes)
        SCLLMOutput.status(f"Predicted cell types for {len(predicted_celltypes)} cells", 'loaded')
        SCLLMOutput.status(f"Prediction summary:", 'info')
        for celltype, count in type_counts.most_common():
            percentage = count / len(predicted_celltypes) * 100
            SCLLMOutput.status(f"{celltype}: {count} cells ({percentage:.1f}%)", indent=1)
        
        return {
            'predictions': predictions,
            'predicted_celltypes': predicted_celltypes,
            'probabilities': probabilities,
            'celltype_mapping': self.celltype_mapping
        }
    
    def predict_celltypes(self, query_adata: AnnData, **kwargs) -> Dict[str, Any]:
        """
        Predict cell types for query data using fine-tuned scFoundation model.
        
        This function provides a convenient interface for cell type prediction,
        similar to scGPT's predict_celltypes function.
        
        Args:
            query_adata: Query data to predict cell types for
            **kwargs: Additional parameters including:
                - batch_size: Batch size for prediction (default: 32)
                - pre_normalized: Whether data is pre-normalized ('T', 'F', or 'A')
                - Other preprocessing parameters
            
        Returns:
            Dictionary containing:
                - predictions: Numeric predictions (class IDs)
                - predicted_celltypes: Cell type names
                - probabilities: Prediction probabilities
                - celltype_mapping: Mapping between IDs and cell type names
                - prediction_summary: Summary statistics
        """
        if not hasattr(self, 'celltype_mapping') or self.celltype_mapping is None:
            raise ValueError("Model has not been fine-tuned for cell type annotation. "
                           "Call fine_tune() first or load a fine-tuned model.")
        
        SCLLMOutput.status(f"Predicting cell types for query data", 'predicting')
        SCLLMOutput.status(f"Query data: {query_adata.n_obs} cells × {query_adata.n_vars} genes", indent=1)
        
        # Use the existing predict_with_finetune method
        results = self.predict_with_finetune(query_adata, **kwargs)
        
        if 'predictions' in results and 'probabilities' in results:
            # Convert IDs to cell type names using the mapping
            id_to_celltype = self.celltype_mapping['id_to_celltype']
            predicted_celltypes = [
                id_to_celltype.get(int(pred), f"Unknown_{int(pred)}") 
                for pred in results['predictions']
            ]
            
            # Add cell type names to results
            results['predicted_celltypes'] = predicted_celltypes
            
            SCLLMOutput.status(f"Predicted cell types for {len(predicted_celltypes)} cells", 'loaded')
            
            # Generate prediction summary
            from collections import Counter
            type_counts = Counter(predicted_celltypes)
            
            SCLLMOutput.status(f"Prediction summary:", 'info')
            prediction_summary = {}
            for celltype, count in type_counts.most_common():
                percentage = count / len(predicted_celltypes) * 100
                SCLLMOutput.status(f"{celltype}: {count} cells ({percentage:.1f}%)", indent=1)
                prediction_summary[celltype] = {
                    'count': count,
                    'percentage': percentage
                }
            
            results['prediction_summary'] = prediction_summary
            
            # Add confidence statistics
            if 'probabilities' in results:
                import numpy as np
                probs = results['probabilities']
                max_probs = np.max(probs, axis=1)
                
                confidence_stats = {
                    'mean_confidence': float(np.mean(max_probs)),
                    'median_confidence': float(np.median(max_probs)),
                    'min_confidence': float(np.min(max_probs)),
                    'max_confidence': float(np.max(max_probs)),
                    'high_confidence_cells': int(np.sum(max_probs > 0.8)),  # cells with >80% confidence
                    'low_confidence_cells': int(np.sum(max_probs < 0.5))   # cells with <50% confidence
                }
                
                results['confidence_stats'] = confidence_stats
                
                SCLLMOutput.status(f"Confidence statistics:", 'info')
                SCLLMOutput.status(f"Mean: {confidence_stats['mean_confidence']:.3f}, High (>80%): {confidence_stats['high_confidence_cells']}, Low (<50%): {confidence_stats['low_confidence_cells']}", indent=1)
        else:
            SCLLMOutput.status(f"No predictions found in results", 'warning')
        
        return results


class LinearProbingClassifier(nn.Module):
    """
    Linear Probing Classifier for fine-tuning scFoundation model.
    
    This implementation follows the approach described in the scFoundation paper,
    where only the last few layers are unfrozen for efficient fine-tuning.
    """
    
    def __init__(self, pretrained_model, config: Dict[str, Any], n_classes: int, frozen_more: bool = True):
        """
        Initialize Linear Probing Classifier.
        
        Args:
            pretrained_model: Pre-trained scFoundation model
            config: Model configuration
            n_classes: Number of output classes
            frozen_more: Whether to freeze token and position embeddings
        """
        super().__init__()
        
        self.config = config
        self.n_classes = n_classes
        self.frozen_more = frozen_more
        
        # Extract components from pre-trained model
        self.token_emb = pretrained_model.token_emb
        self.pos_emb = pretrained_model.pos_emb
        self.encoder = pretrained_model.encoder
        
        # Freeze most of the model
        self._freeze_pretrained_layers()
        
        # Get hidden dimension
        hidden_dim = config['encoder']['hidden_dim']
        
        # Classification head
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )
        
        # Batch normalization
        self.norm = nn.BatchNorm1d(hidden_dim, affine=False, eps=1e-6)
        
        SCLLMOutput.status(f"LinearProbingClassifier: {n_classes} classes", 'loaded')
        SCLLMOutput.status(f"Hidden dimension: {hidden_dim}, Frozen: {frozen_more}", indent=1)
    
    def _freeze_pretrained_layers(self):
        """Freeze pre-trained layers according to the fine-tuning strategy."""
        
        # Freeze token and position embeddings if requested
        if self.frozen_more:
            for _, p in self.token_emb.named_parameters():
                p.requires_grad = False
            for _, p in self.pos_emb.named_parameters():
                p.requires_grad = False
            SCLLMOutput.status(f"Token and position embeddings frozen", 'info')
        
        # Freeze all encoder layers
        for na, param in self.encoder.named_parameters():
            param.requires_grad = False
        
        # Unfreeze the last encoder layer for fine-tuning
        if hasattr(self.encoder, 'transformer_encoder') and len(self.encoder.transformer_encoder) > 2:
            for na, param in self.encoder.transformer_encoder[-2].named_parameters():
                param.requires_grad = True
                SCLLMOutput.status(f"Unfrozen encoder layer: {na}", 'info')
        else:
            SCLLMOutput.status(f"Could not find transformer_encoder layers to unfreeze", 'warning')
    
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            targets: Target labels (optional, for training)
            
        Returns:
            Logits tensor of shape (batch_size, n_classes)
        """
        # Prepare data following scFoundation's approach
        value_labels = x > 0
        x_gathered, x_padding = gatherData(x, value_labels, self.config['pad_token_id'])
        
        # Create gene position IDs
        batch_size = x.shape[0]
        data_gene_ids = torch.arange(self.config['seq_len'] - 2, device=x.device).repeat(batch_size, 1)
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels, self.config['pad_token_id'])
        
        # Token embedding
        x_emb = self.token_emb(torch.unsqueeze(x_gathered, 2).float(), output_weight=0)
        
        # Position embedding
        position_emb = self.pos_emb(position_gene_ids)
        x_emb += position_emb
        
        # Encoder forward pass
        logits = self.encoder(x_emb, x_padding)
        
        # Max pooling across sequence dimension
        logits, _ = torch.max(logits, dim=1)  # (batch_size, hidden_dim)
        
        # Normalization
        logits = self.norm(logits)
        
        # Classification head
        logits = self.fc1(logits)  # (batch_size, n_classes)
        
        return logits


    def integrate(self, adata: AnnData, batch_key: str = "batch", **kwargs) -> Dict[str, Any]:
        """
        Perform batch integration using scFoundation model.
        
        This method uses the scFoundation model embeddings with post-hoc batch correction
        methods to achieve integration across different batches.
        
        Args:
            adata: Input data with batch information
            batch_key: Column name for batch labels in adata.obs
            **kwargs: Additional parameters including:
                - correction_method: Post-hoc correction method ('combat', 'mnn', 'center_scale', 'none')
                - batch_size: Batch size for embedding extraction (default: 32)
                - Other integration parameters
            
        Returns:
            Dictionary with integrated embeddings and batch statistics
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        if batch_key not in adata.obs:
            raise ValueError(f"Batch information '{batch_key}' not found in adata.obs")
        
        SCLLMOutput.status(f"Performing batch integration with scFoundation", 'integrating')
        SCLLMOutput.status(f"Batch key: {batch_key}", indent=1)
        
        # Get batch information
        batch_labels = adata.obs[batch_key].astype('category').cat.codes.values
        unique_batches = np.unique(batch_labels)
        num_batches = len(unique_batches)
        batch_distribution = np.bincount(batch_labels)
        
        SCLLMOutput.status(f"Found {num_batches} batches", indent=1)
        
        # Extract embeddings from scFoundation model
        SCLLMOutput.status(f"Extracting cell embeddings", 'embedding', indent=1)
        embeddings = self.get_embeddings(adata, **kwargs)
        
        # Apply post-hoc batch correction
        correction_method = kwargs.get('correction_method', 'center_scale')
        
        if correction_method == 'combat':
            corrected_embeddings = self._apply_combat_correction_simple(embeddings, batch_labels, **kwargs)
        elif correction_method == 'mnn':
            corrected_embeddings = self._apply_mnn_correction_simple(embeddings, batch_labels, **kwargs)
        elif correction_method == 'center_scale':
            corrected_embeddings = self._apply_center_scale_correction(embeddings, batch_labels, **kwargs)
        elif correction_method == 'none':
            SCLLMOutput.status(f"Using embeddings without correction", 'info', indent=1)
            corrected_embeddings = embeddings
        else:
            SCLLMOutput.status(f"Unknown method '{correction_method}', using center_scale", 'warning', indent=1)
            corrected_embeddings = self._apply_center_scale_correction(embeddings, batch_labels, **kwargs)
        
        results = {
            'embeddings': corrected_embeddings,
            'original_embeddings': embeddings,
            'batch_labels': batch_labels,
            'correction_method': correction_method,
            'integration_stats': {
                'num_batches': num_batches,
                'batch_distribution': batch_distribution.tolist(),
                'total_cells': len(batch_labels),
                'correction_applied': correction_method,
                'method': 'scfoundation_post_hoc_correction'
            }
        }
        
        SCLLMOutput.status(f"Integration completed using {correction_method}", 'loaded')
        return results
    
    def _apply_combat_correction_simple(self, embeddings: np.ndarray, batch_labels: np.ndarray, **kwargs) -> np.ndarray:
        """Apply simple ComBat-style batch correction for integrate function."""
        SCLLMOutput.status(f"Applying ComBat-style correction", 'preprocessing', indent=1)
        
        try:
            from sklearn.preprocessing import StandardScaler
            
            # Simple ComBat-like correction
            corrected = embeddings.copy()
            
            # Standardize within each batch
            for batch_id in np.unique(batch_labels):
                batch_mask = batch_labels == batch_id
                batch_data = corrected[batch_mask]
                
                if batch_data.shape[0] > 1:  # Need at least 2 samples
                    # Center and scale within batch
                    batch_mean = batch_data.mean(axis=0)
                    batch_std = batch_data.std(axis=0) + 1e-8  # Avoid division by zero
                    
                    # Remove batch-specific mean and scale
                    corrected[batch_mask] = (batch_data - batch_mean) / batch_std
            
            # Global standardization
            global_scaler = StandardScaler()
            corrected = global_scaler.fit_transform(corrected)
            
            SCLLMOutput.status(f"ComBat applied to {embeddings.shape[0]} cells", 'loaded', indent=2)
            return corrected
            
        except Exception as e:
            SCLLMOutput.status(f"ComBat failed: {e}", 'warning', indent=2)
            return embeddings
    
    def _apply_mnn_correction_simple(self, embeddings: np.ndarray, batch_labels: np.ndarray, **kwargs) -> np.ndarray:
        """Apply simple MNN-style batch correction for integrate function."""
        SCLLMOutput.status(f"Applying MNN-style correction", 'preprocessing', indent=1)
        
        try:
            from sklearn.neighbors import NearestNeighbors
            
            corrected = embeddings.copy()
            unique_batches = np.unique(batch_labels)
            
            if len(unique_batches) < 2:
                SCLLMOutput.status(f"Only one batch, no correction needed", 'info', indent=2)
                return embeddings
            
            # Simple MNN-style correction between consecutive batches
            for i in range(len(unique_batches) - 1):
                batch1_id = unique_batches[i]
                batch2_id = unique_batches[i + 1]
                
                batch1_mask = batch_labels == batch1_id
                batch2_mask = batch_labels == batch2_id
                
                batch1_data = corrected[batch1_mask]
                batch2_data = corrected[batch2_mask]
                
                if batch1_data.shape[0] > 5 and batch2_data.shape[0] > 5:
                    # Find mutual nearest neighbors
                    k = min(5, min(batch1_data.shape[0], batch2_data.shape[0]) // 2)
                    
                    # Find nearest neighbors from batch2 to batch1
                    nn1 = NearestNeighbors(n_neighbors=k).fit(batch1_data)
                    distances1, indices1 = nn1.kneighbors(batch2_data)
                    
                    # Find nearest neighbors from batch1 to batch2
                    nn2 = NearestNeighbors(n_neighbors=k).fit(batch2_data)
                    distances2, indices2 = nn2.kneighbors(batch1_data)
                    
                    # Apply simple correction by moving batches closer
                    batch1_centroid = batch1_data.mean(axis=0)
                    batch2_centroid = batch2_data.mean(axis=0)
                    correction_vector = (batch1_centroid - batch2_centroid) * 0.5
                    
                    corrected[batch2_mask] += correction_vector
            
            SCLLMOutput.status(f"MNN applied to {len(unique_batches)} batches", 'loaded', indent=2)
            return corrected
            
        except Exception as e:
            SCLLMOutput.status(f"MNN failed: {e}", 'warning', indent=2)
            return self._apply_center_scale_correction(embeddings, batch_labels, **kwargs)
    
    def _apply_center_scale_correction(self, embeddings: np.ndarray, batch_labels: np.ndarray, **kwargs) -> np.ndarray:
        """Apply center and scale batch correction."""
        SCLLMOutput.status(f"Applying center and scale correction", 'preprocessing', indent=1)
        
        try:
            corrected = embeddings.copy()
            
            # Calculate global statistics
            global_mean = corrected.mean(axis=0)
            global_std = corrected.std(axis=0) + 1e-8
            
            # Correct each batch
            for batch_id in np.unique(batch_labels):
                batch_mask = batch_labels == batch_id
                batch_data = corrected[batch_mask]
                
                if batch_data.shape[0] > 1:
                    # Calculate batch statistics
                    batch_mean = batch_data.mean(axis=0)
                    batch_std = batch_data.std(axis=0) + 1e-8
                    
                    # Center and scale to global statistics
                    corrected[batch_mask] = (batch_data - batch_mean) / batch_std * global_std + global_mean
            
            SCLLMOutput.status(f"Center-scale applied to {embeddings.shape[0]} cells", 'loaded', indent=2)
            return corrected
            
        except Exception as e:
            SCLLMOutput.status(f"Center-scale failed: {e}", 'warning', indent=2)
            return embeddings


class FineTuneDataset(Dataset):
    """Dataset class for fine-tuning data."""
    
    def __init__(self, adata: AnnData):
        """
        Initialize dataset from AnnData.
        
        Args:
            adata: Preprocessed AnnData object with celltype_id in .obs
        """
        # Get expression data
        if issparse(adata.X):
            self.expressions = torch.from_numpy(adata.X.toarray()).float()
        else:
            self.expressions = torch.from_numpy(adata.X).float()
        
        # Get labels - handle both Categorical and numeric types
        celltype_values = adata.obs['celltype_id']
        if hasattr(celltype_values, 'cat'):
            # It's a Categorical - get the codes (numeric values)
            label_array = celltype_values.cat.codes.values
        else:
            # It's already numeric
            label_array = celltype_values.values
        
        self.labels = torch.from_numpy(label_array).long()
        
        SCLLMOutput.status(f"FineTuneDataset: {len(self.expressions)} samples", 'loaded')
        SCLLMOutput.status(f"Expression: {self.expressions.shape}, Labels: {self.labels.shape}", indent=1)
    
    def __len__(self):
        return len(self.expressions)
    
    def __getitem__(self, idx):
        return {
            'x': self.expressions[idx],
            'targets': self.labels[idx]
        }


class PredictionDataset(Dataset):
    """Dataset class for prediction (no labels required)."""
    
    def __init__(self, adata: AnnData):
        """
        Initialize dataset from AnnData for prediction.
        
        Args:
            adata: Preprocessed AnnData object (no celltype_id required)
        """
        # Get expression data
        if issparse(adata.X):
            self.expressions = torch.from_numpy(adata.X.toarray()).float()
        else:
            self.expressions = torch.from_numpy(adata.X).float()
        
        SCLLMOutput.status(f"PredictionDataset: {len(self.expressions)} samples", 'loaded')
        SCLLMOutput.status(f"Expression shape: {self.expressions.shape}", indent=1)
    
    def __len__(self):
        return len(self.expressions)
    
    def __getitem__(self, idx):
        return {
            'x': self.expressions[idx]
        }


class SimpleDataset(Dataset):
    """Simple dataset class for scFoundation data."""
    
    def __init__(self, data: Union[np.ndarray, torch.Tensor]):
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data)
        else:
            self.data = data
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]
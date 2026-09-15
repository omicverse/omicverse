"""
scGPT model implementation with simplified interface.
"""

import copy
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

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
# Import scGPT components with error handling
try:
    from .scgpt.model import TransformerModel, TransformerGenerator
    from .scgpt.tokenizer.gene_tokenizer import GeneVocab
    from .scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
    from .scgpt.preprocess import Preprocessor
    from .scgpt.loss import masked_mse_loss, criterion_neg_log_bernoulli
    from .scgpt.utils import set_seed, map_raw_id_to_vocab_id
    _scgpt_imports_available = True
except ImportError as e:
    import warnings
    warnings.warn(f"scGPT components not available: {e}")
    _scgpt_imports_available = False
    
    # Create placeholder classes/functions
    class TransformerModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("scGPT TransformerModel not available due to missing dependencies")
    
    class GeneVocab:
        @classmethod
        def from_file(cls, *args, **kwargs):
            raise ImportError("scGPT GeneVocab not available due to missing dependencies")
    
    def tokenize_and_pad_batch(*args, **kwargs):
        raise ImportError("scGPT tokenization not available due to missing dependencies")
    
    def random_mask_value(*args, **kwargs):
        raise ImportError("scGPT masking not available due to missing dependencies")
    
    class Preprocessor:
        def __init__(self, *args, **kwargs):
            raise ImportError("scGPT Preprocessor not available due to missing dependencies")
    
    def masked_mse_loss(*args, **kwargs):
        raise ImportError("scGPT losses not available due to missing dependencies")
    
    def criterion_neg_log_bernoulli(*args, **kwargs):
        raise ImportError("scGPT losses not available due to missing dependencies")
    
    def set_seed(*args, **kwargs):
        import random
        import numpy as np
        import torch
        if args:
            seed = args[0]
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)


class ScGPTModel(SCLLMBase):
    """
    Simplified scGPT model interface.
    
    This class provides an easy-to-use interface for scGPT model operations
    including loading, preprocessing, training, and inference.
    """
    
    def __init__(self, device: Optional[str] = None, seed: int = 0):
        """
        Initialize scGPT model.
        
        Args:
            device: Device to run the model on
            seed: Random seed for reproducibility
        """
        super().__init__("scGPT", device)
        self.seed = seed
        set_seed(seed)
        
        # Model components
        self.model = None
        self.vocab = None
        self.preprocessor = None
        self.config = None
        
        # Perturbation-specific components
        self.generator_model = None
        self.gene_ids_mapping = None
        
        # Default parameters matching Tutorial exactly
        self.default_config = ModelConfig(
            embsize=128,  # Tutorial uses 128, not 512
            nhead=4,      # Tutorial uses 4, not 8
            d_hid=128,    # Tutorial uses 128, not 512
            nlayers=4,    # Tutorial uses 4, not 12
            nlayers_cls=3,  # Tutorial uses 3
            dropout=0.2,  # Tutorial uses 0.2
            n_bins=51,    # Tutorial uses 51 bins
            max_seq_len=3001,  # Tutorial max sequence length
            pad_token="<pad>",
            special_tokens=["<pad>", "<cls>", "<eoc>"],
            input_style="binned",
            output_style="binned", 
            input_emb_style="continuous",  # Tutorial uses continuous
            cell_emb_style="cls",  # Tutorial uses cls token
            fast_transformer=True,
            pre_norm=False,
            # Additional Tutorial-specific parameters
            explicit_zero_prob=False,
            use_fast_transformer=True,
            fast_transformer_backend="flash",
            mvc_decoder_style="inner product",
            ecs_threshold=0.0,
            # Integration-specific parameters (Tutorial_Integration)
            do_dab=False,            # Domain Adversarial Batch correction
            do_mvc=False,            # Masked Value Prediction (GEPC)
            do_ecs=False,            # Elastic Cell Similarity
            use_batch_labels=False,  # Whether to use batch labels
            domain_spec_batchnorm=False,  # Domain-Specific Batch Normalization
            num_batch_labels=None,   # Number of batch types
            dab_weight=1.0,          # DAB loss weight
            ecs_weight=10.0,         # ECS loss weight
            gepc_weight=1.0,         # GEPC (MVC) loss weight
            mask_ratio=0.4,          # Masking ratio for integration (higher than annotation)
        )
    
    def load_model(self, model_path: Union[str, Path], **kwargs) -> None:
        """
        Load a pre-trained scGPT model.
        
        Args:
            model_path: Path to the model directory
            **kwargs: Additional parameters
        """
        if not _scgpt_imports_available:
            raise ImportError("Cannot load scGPT model: required dependencies not available")
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist")
        
        # Load vocabulary
        vocab_file = model_path / "vocab.json"
        if vocab_file.exists():
            self.vocab = GeneVocab.from_file(vocab_file)
            SCLLMOutput.status(f"Loaded vocabulary: {len(self.vocab):,} genes", 'loaded')
        else:
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
        
        # Add special tokens
        for token in self.default_config.special_tokens:
            if token not in self.vocab:
                # Note: GeneVocab may not have append_token method
                # This is a simplified approach
                SCLLMOutput.status(f"Special token {token} not in vocabulary", 'warning')
        
        self.vocab.set_default_index(self.vocab[self.default_config.pad_token])
        
        # Load model configuration
        config_file = model_path / "args.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                model_config = json.load(f)
            
            # Update default config with loaded config
            self.config = copy.deepcopy(self.default_config)
            self.config.update(**model_config)
            self.config.update(**kwargs)  # Override with user parameters
            
            SCLLMOutput.status(f"Loaded model config from {config_file.name}", 'loaded')
            
            # Show key config parameters
            config_params = {}
            for key in ['embsize', 'nheads', 'd_hid', 'nlayers', 'n_layers_cls']:
                if key in model_config:
                    config_params[key] = model_config[key]
            
            if config_params:
                SCLLMOutput.model_info("Key Parameters", config_params)
        else:
            self.config = self.default_config
            self.config.update(**kwargs)
            SCLLMOutput.status(f"Using default configuration", 'info')
        
        # Determine number of classes - try to infer from loaded model weights
        n_cls = kwargs.get('n_cls')
        
        if not n_cls:
            # Try to get from config
            n_cls = (getattr(self.config, 'n_cls', None) or 
                    getattr(self.config, 'num_types', None))
        
        if not n_cls:
            # Try to infer from model weights if they exist
            model_file = model_path / "best_model.pt"
            if not model_file.exists():
                model_file = model_path / "model.pt"
            
            if model_file.exists():
                try:
                    checkpoint = torch.load(model_file, map_location="cpu")
                    SCLLMOutput.status(f"Analyzing model checkpoint for n_cls inference...", 'preprocessing')
                    
                    # Look for classifier output layer - be more specific
                    classifier_candidates = []
                    for key, tensor in checkpoint.items():
                        if ('cls' in key.lower() and 'weight' in key):
                            if 'out' in key.lower() or key.endswith('weight'):
                                if len(tensor.shape) == 2:
                                    classifier_candidates.append((key, tensor.shape[0]))
                    
                    # Choose the most likely classifier layer
                    if classifier_candidates:
                        # Sort by potential n_cls and choose reasonable one
                        classifier_candidates.sort(key=lambda x: x[1])
                        for key, potential_n_cls in classifier_candidates:
                            if potential_n_cls > 1 and potential_n_cls < 1000:
                                n_cls = potential_n_cls
                                SCLLMOutput.status(f"Inferred n_cls={n_cls} from {key}", 'loaded')
                                break
                    else:
                        SCLLMOutput.status(f"No classifier layers found in checkpoint", 'warning')
                        
                except Exception as e:
                    SCLLMOutput.status(f"Could not infer n_cls from model weights: {e}", 'warning')
        
        # Use default if still not found
        if not n_cls:
            n_cls = 50
            SCLLMOutput.status(f"Using default n_cls={n_cls}", 'info')
        else:
            SCLLMOutput.status(f"Using n_cls={n_cls} classes", 'info')
        
        # Store n_cls for later use
        self.n_cls = n_cls
        
        # Initialize model
        self.model = TransformerModel(
            ntoken=len(self.vocab),
            d_model=self.config.embsize,
            nhead=self.config.nhead,
            d_hid=self.config.d_hid,
            nlayers=self.config.nlayers,
            nlayers_cls=self.config.nlayers_cls,
            n_cls=n_cls,  # Use proper number of classes
            vocab=self.vocab,
            dropout=self.config.dropout,
            pad_token=self.config.pad_token,
            pad_value=-2,
            do_mvc=False,  # Default to False for basic usage
            do_dab=False,  # Default to False for basic usage
            use_batch_labels=False,  # Default to False for basic usage
            num_batch_labels=None,
            domain_spec_batchnorm=False,
            input_emb_style=self.config.input_emb_style,
            n_input_bins=self.config.n_bins + 2,
            cell_emb_style=self.config.cell_emb_style,
            mvc_decoder_style="inner product",
            ecs_threshold=0.3,
            explicit_zero_prob=False,
            use_fast_transformer=self.config.fast_transformer,
            fast_transformer_backend="flash",
            pre_norm=self.config.pre_norm,
        )
        
        # Check if user wants to override classifier for multi-class prediction
        force_multiclass = kwargs.get('force_multiclass', False)
        expected_n_cls = kwargs.get('expected_n_cls', None)
        
        # Load model weights
        model_file = model_path / "best_model.pt"
        if not model_file.exists():
            model_file = model_path / "model.pt"
        
        if model_file.exists():
            try:
                checkpoint = torch.load(model_file, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                SCLLMOutput.status(f"Model weights loaded", 'loaded')
                
            except Exception as e:
                SCLLMOutput.status(f"Loading compatible weights only", 'warning')
                
                # Load compatible weights only
                model_dict = self.model.state_dict()
                checkpoint = torch.load(model_file, map_location=self.device)
                
                compatible_dict = {}
                incompatible_dict = {}
                
                for k, v in checkpoint.items():
                    if k in model_dict:
                        if v.shape == model_dict[k].shape:
                            compatible_dict[k] = v
                        else:
                            incompatible_dict[k] = f"shape mismatch: {v.shape} vs {model_dict[k].shape}"
                    else:
                        incompatible_dict[k] = "key not found in model"
                
                model_dict.update(compatible_dict)
                self.model.load_state_dict(model_dict)
                
                SCLLMOutput.status(f"Compatible weights loaded: {len(compatible_dict)}/{len(checkpoint)}", 'loaded')
                if incompatible_dict:
                    SCLLMOutput.status(f"Some weights incompatible ({len(incompatible_dict)})", 'warning')
        
        # Check if the loaded model has the right number of classes
        if hasattr(self.model, 'cls_decoder') and hasattr(self.model.cls_decoder, 'out_layer'):
            actual_n_cls = self.model.cls_decoder.out_layer.out_features
            SCLLMOutput.status(f"Model classes: {actual_n_cls}", 'info')
            
            if actual_n_cls == 1 and (force_multiclass or expected_n_cls):
                target_n_cls = expected_n_cls or n_cls
                SCLLMOutput.status(f"Model needs fine-tuning ({actual_n_cls} → {target_n_cls} classes)", 'warning')
                
                if force_multiclass:
                    SCLLMOutput.status(f"Reinitializing classifier: {target_n_cls} classes", 'preprocessing')
                    # Reinitialize the classifier layer
                    import torch.nn as nn
                    if hasattr(self.model.cls_decoder, '_decoder'):
                        # Get the input dimension from the previous layer
                        prev_dim = self.model.cls_decoder._decoder[-2].out_features
                    else:
                        prev_dim = self.model.d_model
                    
                    # Replace the output layer
                    self.model.cls_decoder.out_layer = nn.Linear(prev_dim, target_n_cls)
                    self.model.cls_decoder.out_layer.to(self.device)
                    SCLLMOutput.status(f"Classifier reinitialized: {target_n_cls} classes", 'loaded')
                    SCLLMOutput.status(f"Classifier needs training", 'warning')
        
        self.model.to(self.device)
        self.is_loaded = True
        SCLLMOutput.status(f"Model ready on {self.device}", 'loaded')
    
    def setup_generator_for_perturbation(self, **kwargs) -> None:
        """
        Setup TransformerGenerator model for perturbation prediction.
        This adds the missing layers needed for perturbation tasks.
        
        Args:
            **kwargs: Additional parameters for generator setup
        """
        if not self.is_loaded:
            raise ValueError("Base model not loaded. Call load_model() first.")
        
        if not _scgpt_imports_available:
            raise ImportError("Cannot setup generator: required dependencies not available")
        
        SCLLMOutput.status(f"Setting up perturbation generator", 'preprocessing')
        
        # Get generator-specific parameters
        pert_pad_id = kwargs.get('pert_pad_id', 0)
        pad_value = kwargs.get('pad_value', -2)
        
        # Create TransformerGenerator with same architecture as base model
        self.generator_model = TransformerGenerator(
            ntoken=len(self.vocab),
            d_model=self.config.embsize,
            nhead=self.config.nhead,
            d_hid=self.config.d_hid,
            nlayers=self.config.nlayers,
            nlayers_cls=self.config.nlayers_cls,
            n_cls=1,  # Generator typically has 1 output class
            vocab=self.vocab,
            dropout=self.config.dropout,
            pad_token=self.config.pad_token,
            pad_value=pad_value,
            pert_pad_id=pert_pad_id,
            use_fast_transformer=False,
        )
        
        # Transfer weights from base model to generator
        SCLLMOutput.status(f"Transferring weights to generator", 'preprocessing')
        base_state_dict = self.model.state_dict()
        generator_state_dict = self.generator_model.state_dict()
        
        # Map common layers
        transferred_layers = []
        missed_layers = []
        
        for key in generator_state_dict.keys():
            if key in base_state_dict:
                if base_state_dict[key].shape == generator_state_dict[key].shape:
                    generator_state_dict[key] = base_state_dict[key].clone()
                    transferred_layers.append(key)
                else:
                    missed_layers.append(f"{key}: shape mismatch {base_state_dict[key].shape} vs {generator_state_dict[key].shape}")
            else:
                missed_layers.append(f"{key}: not found in base model")
        
        self.generator_model.load_state_dict(generator_state_dict)
        self.generator_model.to(self.device)
        
        SCLLMOutput.status(f"Transferred {len(transferred_layers)} layers to generator", 'loaded')
        if missed_layers:
            SCLLMOutput.status(f"{len(missed_layers)} layers use random initialization", 'warning')
        
        SCLLMOutput.status(f"Generator setup complete", 'loaded')
        SCLLMOutput.status(f"Generator ready for inference", 'info', indent=1)
    
    def predict_perturbation(self, 
                           adata: AnnData, 
                           target_genes: List[str],
                           **kwargs) -> AnnData:
        """
        Predict gene expression after perturbation of target genes.
        This method correctly implements perturbation prediction by using perturbation flags
        that the model learns to interpret, rather than manually modifying gene expression.
        
        Args:
            adata: Input AnnData object (control cells)
            target_genes: List of genes to perturb
            **kwargs: Additional parameters including:
                - include_zero_gene: Gene inclusion strategy ("all", "batch-wise") 
                - batch_size: Batch size for prediction
                - perturbation_type: Type of perturbation ("knockout", "overexpression")
                
        Returns:
            AnnData object with predicted perturbed gene expression values
        """
        if self.generator_model is None:
            SCLLMOutput.status(f"Setting up generator model", 'preprocessing')
            self.setup_generator_for_perturbation(**kwargs)
        
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        SCLLMOutput.status(f"Predicting perturbation: {', '.join(target_genes)}", 'predicting')
        
        # Validate target genes
        for gene in target_genes:
            if gene not in self.vocab:
                raise ValueError(f"Target gene '{gene}' not found in vocabulary")
        
        # Preprocess input data
        adata_processed = self.preprocess(adata.copy(), **kwargs)
        
        # Get parameters
        include_zero_gene = kwargs.get('include_zero_gene', "all")
        batch_size = kwargs.get('batch_size', 32)
        perturbation_type = kwargs.get('perturbation_type', "knockout")
        
        # Get gene expression data
        input_layer_key = "X_binned"
        if input_layer_key not in adata_processed.layers:
            raise ValueError(f"Required layer '{input_layer_key}' not found")
        
        all_counts = (
            adata_processed.layers[input_layer_key].toarray()
            if issparse(adata_processed.layers[input_layer_key])
            else adata_processed.layers[input_layer_key]
        )
        
        # Get genes and create perturbation flags
        genes = adata_processed.var_names.tolist()
        n_genes = len(genes)
        n_cells = all_counts.shape[0]
        
        # Create gene IDs mapping
        gene_ids = np.array([self.vocab[gene] for gene in genes], dtype=int)
        self.gene_ids_mapping = gene_ids
        
        # Create perturbation flags - this is the key difference!
        # Instead of modifying gene expression, we use flags to tell the model which genes are perturbed
        pert_flags = np.zeros((n_cells, n_genes), dtype=int)
        
        for target_gene in target_genes:
            if target_gene in genes:
                gene_idx = genes.index(target_gene)
                if perturbation_type == "knockout":
                    pert_flags[:, gene_idx] = 1  # Flag: gene is knocked out
                elif perturbation_type == "overexpression":
                    pert_flags[:, gene_idx] = 2  # Flag: gene is overexpressed
                SCLLMOutput.status(f"Flagged {target_gene}: {perturbation_type}", 'info', indent=1)
        
        # Use ORIGINAL gene expression as input (not modified!)
        # The model will learn to predict perturbed expression based on:
        # 1. Original expression values
        # 2. Perturbation flags
        original_counts = all_counts.copy()  # Keep original expression unchanged
        
        SCLLMOutput.status(f"Starting perturbation prediction: {n_cells} cells", 'predicting')
        SCLLMOutput.status(f"Using perturbation flags for prediction", 'info')
        
        # Make predictions using the generator model's pred_perturb-like functionality
        self.generator_model.eval()
        all_predictions = []
        
        with torch.no_grad():
            # Process in batches
            pbar = SCLLMOutput.progress_bar(total=(n_cells + batch_size - 1) // batch_size, desc="Perturbation batches", model_name="scGPT")
            for batch_idx, i in enumerate(range(0, n_cells, batch_size)):
                pbar.update(1)
                end_idx = min(i + batch_size, n_cells)
                batch_cells = end_idx - i
                
                # Get batch data
                batch_counts = original_counts[i:end_idx]
                batch_pert_flags = pert_flags[i:end_idx]
                
                # Convert to tensors
                ori_gene_values = torch.from_numpy(batch_counts).float().to(self.device)
                pert_flags_tensor = torch.from_numpy(batch_pert_flags).long().to(self.device)
                
                if include_zero_gene == "all":
                    # Use all genes
                    input_gene_ids = torch.arange(n_genes, dtype=torch.long).repeat(batch_cells, 1)
                    
                    # Map raw gene indices to vocabulary IDs
                    mapped_input_gene_ids = map_raw_id_to_vocab_id(
                        torch.arange(n_genes), gene_ids
                    ).repeat(batch_cells, 1).to(self.device)
                    
                    input_values = ori_gene_values
                    input_pert_flags = pert_flags_tensor
                    
                    # Create padding mask
                    src_key_padding_mask = torch.zeros_like(
                        input_values, dtype=torch.bool, device=self.device
                    )
                    
                else:
                    # batch-wise: only use non-zero genes
                    nonzero_gene_mask = (ori_gene_values.sum(0) > 0)
                    input_gene_indices = torch.where(nonzero_gene_mask)[0]
                    
                    if len(input_gene_indices) > self.config.max_seq_len:
                        # Randomly sample if too many genes
                        selected_indices = torch.randperm(len(input_gene_indices))[:self.config.max_seq_len]
                        input_gene_indices = input_gene_indices[selected_indices]
                    
                    input_values = ori_gene_values[:, input_gene_indices]
                    input_pert_flags = pert_flags_tensor[:, input_gene_indices]
                    
                    # Map to vocabulary IDs
                    mapped_input_gene_ids = map_raw_id_to_vocab_id(
                        input_gene_indices, gene_ids
                    ).repeat(batch_cells, 1).to(self.device)
                    
                    src_key_padding_mask = torch.zeros_like(
                        input_values, dtype=torch.bool, device=self.device
                    )
                
                # Model forward pass - this is where the magic happens!
                # The model uses original expression + perturbation flags to predict perturbed expression
                output_dict = self.generator_model(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,  # This tells the model which genes are perturbed
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=False,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                )
                
                # Extract predictions
                if "mlm_output" in output_dict:
                    predictions = output_dict["mlm_output"]
                    
                    if include_zero_gene != "all":
                        # Reconstruct full gene expression for batch-wise case
                        full_predictions = torch.zeros((batch_cells, n_genes), device=self.device)
                        full_predictions[:, input_gene_indices] = predictions
                        predictions = full_predictions
                    
                    all_predictions.append(predictions.cpu().numpy())
                
                if batch_idx == 0:
                    SCLLMOutput.status(f"First batch: {batch_cells} cells", 'info', indent=1)
                    SCLLMOutput.status(f"Perturbed positions: {torch.sum(input_pert_flags > 0).item()}", 'info', indent=1)
        
        if not all_predictions:
            raise RuntimeError("No predictions were generated")
        
        # Combine predictions
        predicted_expressions = np.concatenate(all_predictions, axis=0)
        pbar.close()
        SCLLMOutput.status(f"Generated predictions: {predicted_expressions.shape[0]} cells", 'loaded')
        
        # Create new AnnData with predicted expressions using processed genes
        # This avoids dimension mismatch issues and is cleaner
        from anndata import AnnData
        
        # Use processed genes as the var (since predictions are based on processed data)
        processed_genes = adata_processed.var_names
        
        # Create new AnnData with predicted expressions
        result_adata = AnnData(
            X=predicted_expressions,  # Use predictions as main expression matrix
            obs=adata.obs.copy(),     # Keep original cell metadata
            var=adata_processed.var.copy(),  # Use processed gene metadata
        )
        
        # Copy obsm (cell embeddings, etc.) if they exist
        if hasattr(adata, 'obsm') and len(adata.obsm) > 0:
            for key, value in adata.obsm.items():
                result_adata.obsm[key] = value.copy()
        
        # Store predicted expressions in layers
        result_adata.layers["X_predicted_binned"] = predicted_expressions
        
        # Convert binned predictions back to expression values if possible
        if hasattr(self.preprocessor, 'binning_edges') or 'X_binned' in adata_processed.layers:
            # Simple de-binning: use bin centers
            bin_centers = np.arange(self.config.n_bins)
            predicted_expression_values = bin_centers[predicted_expressions.astype(int)]
            result_adata.layers["X_predicted"] = predicted_expression_values
            result_adata.X = predicted_expression_values  # Also set as main matrix
        else:
            # If no binning info, use predicted values directly
            result_adata.layers["X_predicted"] = predicted_expressions
        
        # Add perturbation metadata
        result_adata.obs["perturbation_genes"] = ",".join(target_genes)
        result_adata.obs["perturbation_type"] = perturbation_type
        result_adata.uns["perturbation_info"] = {
            "target_genes": target_genes,
            "perturbation_type": perturbation_type,
            "include_zero_gene": include_zero_gene,
            "n_cells_predicted": predicted_expressions.shape[0],
            "perturbation_flags_used": True  # Indicate we used proper perturbation flags
        }
        
        SCLLMOutput.status(f"Perturbation prediction complete", 'complete')
        SCLLMOutput.status(f"Target genes: {', '.join(target_genes)}", indent=1)
        SCLLMOutput.status(f"Type: {perturbation_type}", indent=1)
        SCLLMOutput.status(f"Predictions: {predicted_expressions.shape[0]} cells", indent=1)
        
        return result_adata
    
    def fine_tune_generator(self, 
                          train_adata: AnnData,
                          valid_adata: Optional[AnnData] = None,
                          target_genes: List[str] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Fine-tune the generator model for perturbation prediction.
        
        Args:
            train_adata: Training data with perturbation information
            valid_adata: Validation data (optional)
            target_genes: List of genes to use for perturbation training
            **kwargs: Training parameters
            
        Returns:
            Training results and metrics
        """
        if self.generator_model is None:
            SCLLMOutput.status(f"Setting up generator for fine-tuning", 'preprocessing')
            self.setup_generator_for_perturbation(**kwargs)
        
        SCLLMOutput.status(f"Starting generator fine-tuning", 'fine_tuning')
        
        # Get training parameters
        epochs = kwargs.get('epochs', 15)
        batch_size = kwargs.get('batch_size', 64) 
        lr = kwargs.get('lr', 1e-4)
        mask_ratio = kwargs.get('mask_ratio', 0.0)  # No additional masking needed
        
        SCLLMOutput.status(f"Training parameters:", 'info')
        SCLLMOutput.status(f"epochs: {epochs}, batch_size: {batch_size}, lr: {lr}", indent=1)
        SCLLMOutput.status(f"mask_ratio: {mask_ratio}", indent=1)
        
        # Setup training components
        optimizer = torch.optim.Adam(self.generator_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        criterion = masked_mse_loss
        
        # Prepare training data 
        train_results = self._prepare_perturbation_data(train_adata, target_genes, **kwargs)
        valid_results = self._prepare_perturbation_data(valid_adata, target_genes, **kwargs) if valid_adata is not None else None
        
        # Training loop
        best_loss = float('inf')
        best_model_state = None
        training_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Generator epoch handled by progress bar
            
            # Training
            train_loss = self._train_generator_epoch(
                train_results, optimizer, criterion, batch_size
            )
            training_history['train_loss'].append(train_loss)
            
            # Validation  
            if valid_results is not None:
                val_loss = self._validate_generator_epoch(valid_results, criterion, batch_size)
                training_history['val_loss'].append(val_loss)
                
                # Loss metrics displayed in progress bar
                
                # Save best model
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_state = copy.deepcopy(self.generator_model.state_dict())
                    SCLLMOutput.status(f"New best validation loss: {best_loss:.4f}", 'best')
            else:
                # Loss metrics displayed in progress bar
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_model_state = copy.deepcopy(self.generator_model.state_dict())
            
            scheduler.step()
        
        # Load best model
        if best_model_state is not None:
            self.generator_model.load_state_dict(best_model_state)
            SCLLMOutput.status(f"Best generator model loaded: {best_loss:.4f}", 'loaded')
        
        SCLLMOutput.status(f"Generator fine-tuning completed", 'complete')
        
        return {
            'best_loss': best_loss,
            'training_history': training_history,
            'target_genes': target_genes
        }
    
    def _prepare_perturbation_data(self, adata: AnnData, target_genes: List[str], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Prepare data for perturbation training following Tutorial_Perturbation exactly.
        
        The key insight is that we need to simulate how the GEARS dataset works:
        - Original expression as input
        - Perturbation flags indicating which genes are perturbed
        - Target expression showing the actual effect of perturbation
        
        For training, we can simulate perturbations by creating artificial perturbation effects.
        """
        if adata is None:
            return None
            
        # Preprocess data
        adata_processed = self.preprocess(adata.copy(), **kwargs)
        
        # Get expression data
        input_layer_key = "X_binned"
        if input_layer_key not in adata_processed.layers:
            raise ValueError(f"Required layer '{input_layer_key}' not found")
        
        all_counts = (
            adata_processed.layers[input_layer_key].toarray()
            if issparse(adata_processed.layers[input_layer_key])
            else adata_processed.layers[input_layer_key]
        )
        
        genes = adata_processed.var_names.tolist()
        n_genes = len(genes)
        n_cells = all_counts.shape[0]
        
        # Create gene IDs mapping
        gene_ids = np.array([self.vocab[gene] for gene in genes], dtype=int)
        
        # Create perturbation flags for training
        pert_flags = np.zeros((n_cells, n_genes), dtype=int)
        
        # Create synthetic perturbation targets for training
        # This simulates what real perturbation data would look like
        perturbed_counts = all_counts.copy()
        
        if target_genes:
            # Use specified target genes for perturbation training
            for target_gene in target_genes:
                if target_gene in genes:
                    gene_idx = genes.index(target_gene)
                    # Randomly assign perturbation to subset of cells
                    n_perturbed_cells = max(1, n_cells // 4)  # 25% of cells get perturbation
                    random_cells = np.random.choice(n_cells, size=n_perturbed_cells, replace=False)
                    pert_flags[random_cells, gene_idx] = 1  # Mark as perturbed
                    
                    # Create synthetic perturbation effect for training
                    # Option 1: Knockout effect - reduce expression
                    knockout_factor = kwargs.get('knockout_simulation_factor', 0.1)
                    perturbed_counts[random_cells, gene_idx] = (
                        perturbed_counts[random_cells, gene_idx] * knockout_factor
                    ).astype(perturbed_counts.dtype)
                    
        else:
            # Random perturbations for general training
            n_pert_per_cell = kwargs.get('n_perturbations_per_cell', 1)
            for cell_idx in range(n_cells):
                # Randomly select genes to perturb
                pert_genes = np.random.choice(n_genes, size=n_pert_per_cell, replace=False)
                pert_flags[cell_idx, pert_genes] = 1
                
                # Apply synthetic perturbation effects
                for gene_idx in pert_genes:
                    # Simulate knockdown/knockout
                    knockout_factor = np.random.uniform(0.0, 0.3)  # Reduce to 0-30% of original
                    perturbed_counts[cell_idx, gene_idx] = (
                        perturbed_counts[cell_idx, gene_idx] * knockout_factor
                    ).astype(perturbed_counts.dtype)
        
        # Tokenize input (original expression) following Tutorial_Perturbation exactly
        tokenized_data = tokenize_and_pad_batch(
            all_counts,  # Original expression as input
            gene_ids,
            max_len=self.config.max_seq_len,
            vocab=self.vocab,
            pad_token=self.config.pad_token,
            pad_value=-2,
            append_cls=True,
            include_zero_gene=False,
        )
        
        # Tokenize target (perturbed expression)
        target_tokenized = tokenize_and_pad_batch(
            perturbed_counts,  # Perturbed expression as target
            gene_ids,
            max_len=self.config.max_seq_len,
            vocab=self.vocab,
            pad_token=self.config.pad_token,
            pad_value=-2,
            append_cls=True,
            include_zero_gene=False,
        )
        
        # Create perturbation flags tensor that matches tokenized length
        # The tokenized data may be shorter due to max_seq_len, so we need to align
        tokenized_length = tokenized_data["genes"].shape[1]
        if tokenized_length < n_genes:
            # Truncate perturbation flags to match tokenized length
            pert_flags_aligned = pert_flags[:, :tokenized_length]
        else:
            # Pad perturbation flags if needed (shouldn't happen normally)
            pad_size = tokenized_length - n_genes
            pert_flags_aligned = np.pad(pert_flags, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)
        
        SCLLMOutput.status(f"Training data: {n_cells} cells, {n_genes} genes", 'info')
        SCLLMOutput.status(f"Perturbations: {np.sum(pert_flags > 0)}", indent=1)
        
        return {
            "gene_ids": tokenized_data["genes"],
            "values": tokenized_data["values"],  # Original expression
            "target_values": target_tokenized["values"],  # Target perturbed expression
            "pert_flags": torch.from_numpy(pert_flags_aligned).long(),  # Perturbation flags
        }
    
    def _train_generator_epoch(self, data: Dict[str, torch.Tensor], optimizer, criterion, batch_size: int) -> float:
        """Train generator for one epoch."""
        self.generator_model.train()
        
        dataset = SimpleDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_data in dataloader:
            input_gene_ids = batch_data["gene_ids"].to(self.device)
            input_values = batch_data["values"].to(self.device)
            target_values = batch_data["target_values"].to(self.device)
            
            # Get perturbation flags - need to match tokenized length
            if input_gene_ids.shape[1] <= data["pert_flags"].shape[1]:
                input_pert_flags = batch_data["pert_flags"][:, :input_gene_ids.shape[1]].to(self.device)
            else:
                # Pad if needed
                pad_size = input_gene_ids.shape[1] - data["pert_flags"].shape[1]
                input_pert_flags = torch.cat([
                    batch_data["pert_flags"], 
                    torch.zeros(batch_data["pert_flags"].shape[0], pad_size, dtype=torch.long)
                ], dim=1).to(self.device)
            
            src_key_padding_mask = input_gene_ids.eq(self.vocab[self.config.pad_token])
            
            optimizer.zero_grad()
            
            # Generator forward pass - following Tutorial_Perturbation exactly
            output_dict = self.generator_model(
                input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=False,
                CCE=False,
                MVC=False,
                ECS=False,
            )
            
            if "mlm_output" in output_dict:
                output_values = output_dict["mlm_output"]
                
                # Use all positions for perturbation training (no masking)
                masked_positions = torch.ones_like(input_values, dtype=torch.bool)
                loss = criterion(output_values, target_values, masked_positions)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator_model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_generator_epoch(self, data: Dict[str, torch.Tensor], criterion, batch_size: int) -> float:
        """Validate generator for one epoch."""
        self.generator_model.eval()
        
        dataset = SimpleDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                input_gene_ids = batch_data["gene_ids"].to(self.device)
                input_values = batch_data["values"].to(self.device)
                target_values = batch_data["target_values"].to(self.device)
                
                # Handle perturbation flags
                if input_gene_ids.shape[1] <= data["pert_flags"].shape[1]:
                    input_pert_flags = batch_data["pert_flags"][:, :input_gene_ids.shape[1]].to(self.device)
                else:
                    pad_size = input_gene_ids.shape[1] - data["pert_flags"].shape[1]
                    input_pert_flags = torch.cat([
                        batch_data["pert_flags"], 
                        torch.zeros(batch_data["pert_flags"].shape[0], pad_size, dtype=torch.long)
                    ], dim=1).to(self.device)
                
                src_key_padding_mask = input_gene_ids.eq(self.vocab[self.config.pad_token])
                
                output_dict = self.generator_model(
                    input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=False,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                )
                
                if "mlm_output" in output_dict:
                    output_values = output_dict["mlm_output"]
                    masked_positions = torch.ones_like(input_values, dtype=torch.bool)
                    loss = criterion(output_values, target_values, masked_positions)
                    
                    total_loss += loss.item()
                    num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def preprocess(self, adata: AnnData, **kwargs) -> AnnData:
        """
        Preprocess data for scGPT following Tutorial implementation exactly.
        
        Args:
            adata: Input AnnData object
            **kwargs: Preprocessing parameters
                skip_normalization (bool): Force skip normalization (default: auto-detect)
                force_normalization (bool): Force apply normalization even if detected as normalized
                data_is_raw (bool): Whether input data is raw counts (default: True)
                
        Returns:
            Preprocessed AnnData object
        """
        adata_processed = adata.copy()
        
        # Step 1: Filter genes by vocabulary BEFORE preprocessing (like Tutorial)
        if self.vocab is not None:
            SCLLMOutput.status(f"Filtering genes by vocabulary", 'preprocessing')
            adata_processed.var["id_in_vocab"] = [
                1 if gene in self.vocab else -1 
                for gene in adata_processed.var_names
            ]
            
            genes_in_vocab = (adata_processed.var["id_in_vocab"] >= 0).sum()
            total_genes = len(adata_processed.var)
            SCLLMOutput.status(f"Matched {genes_in_vocab}/{total_genes} genes", 'info')
            
            if genes_in_vocab == 0:
                raise ValueError("No genes matched the vocabulary! Check gene naming conventions.")
            
            # Keep only genes in vocabulary
            adata_processed = adata_processed[:, adata_processed.var["id_in_vocab"] >= 0].copy()
            SCLLMOutput.status(f"Retained {adata_processed.n_vars} genes", 'loaded')
        
        # Step 2: Initialize preprocessor with Tutorial-exact parameters
        if self.preprocessor is None:
            # Get data_is_raw flag - crucial for Tutorial compatibility
            data_is_raw = kwargs.get('data_is_raw', True)  # Tutorial assumes raw data by default
            
            self.preprocessor = Preprocessor(
                use_key="X",  # Use adata.X as input
                filter_gene_by_counts=kwargs.get('filter_gene_by_counts', False),
                filter_cell_by_counts=kwargs.get('filter_cell_by_counts', False),
                normalize_total=kwargs.get('normalize_total', 1e4),  # Tutorial uses 1e4
                result_normed_key="X_normed",
                log1p=kwargs.get('log1p', data_is_raw),  # Tutorial: log1p if data_is_raw
                result_log1p_key="X_log1p",
                subset_hvg=kwargs.get('subset_hvg', False),  # Tutorial: False
                hvg_flavor=kwargs.get('hvg_flavor', "seurat_v3" if data_is_raw else "cell_ranger"),
                binning=self.config.n_bins,  # Tutorial: n_bins = 51
                result_binned_key="X_binned",
            )
            SCLLMOutput.status(f"Preprocessor initialized", 'loaded')
            SCLLMOutput.status(f"n_bins: {self.config.n_bins}, normalize: {self.preprocessor.normalize_total}", indent=1)
        
        # Step 3: Smart preprocessing with user control
        skip_preprocessing = False
        
        # Check user preferences
        skip_normalization = kwargs.get('skip_normalization', None)
        force_normalization = kwargs.get('force_normalization', False)
        
        # Check if already fully preprocessed
        if hasattr(adata_processed, 'layers') and 'X_binned' in adata_processed.layers:
            SCLLMOutput.status(f"Data already preprocessed, skipping", 'info')
            skip_preprocessing = True
        else:
            # Inspect data to detect normalization status
            if issparse(adata_processed.X):
                X_data = adata_processed.X.toarray()
            else:
                X_data = adata_processed.X
            
            # Calculate statistics to detect normalization
            cell_totals = X_data.sum(axis=1)
            mean_total = cell_totals.mean()
            median_total = np.median(cell_totals)
            
            SCLLMOutput.status(f"Data inspection - Mean: {mean_total:.1f}, Range: [{X_data.min():.3f}, {X_data.max():.3f}]", 'info')
            
            # Auto-detect normalization status
            auto_detected_normalized = False
            if 9000 <= mean_total <= 11000:  # Close to 10k normalization
                auto_detected_normalized = True
                SCLLMOutput.status(f"Auto-detected: normalized to ~10k", 'info', indent=1)
            elif 900000 <= mean_total <= 1100000:  # Close to 1M normalization  
                auto_detected_normalized = True
                SCLLMOutput.status(f"Auto-detected: normalized to ~1M", 'info', indent=1)
            elif mean_total < 1000:  # Very low counts, likely already log-transformed
                auto_detected_normalized = True
                SCLLMOutput.status(f"Auto-detected: log-transformed", 'info', indent=1)
            else:
                SCLLMOutput.status(f"Auto-detected: raw counts", 'info', indent=1)
            
            # Determine final normalization decision
            should_skip_normalization = False
            
            if force_normalization:
                SCLLMOutput.status(f"User override: forcing normalization", 'warning', indent=1)
                should_skip_normalization = False
            elif skip_normalization is True:
                SCLLMOutput.status(f"User override: skipping normalization", 'warning', indent=1)
                should_skip_normalization = True
            elif skip_normalization is False:
                SCLLMOutput.status(f"User override: applying normalization", 'warning', indent=1)
                should_skip_normalization = False
            else:
                # Use auto-detection
                should_skip_normalization = auto_detected_normalized
                if should_skip_normalization:
                    SCLLMOutput.status(f"Decision: skipping normalization", 'loaded', indent=1)
                else:
                    SCLLMOutput.status(f"Decision: applying normalization", 'loaded', indent=1)
            
            # Adjust preprocessor settings if skipping normalization
            if should_skip_normalization:
                SCLLMOutput.status(f"Adjusting preprocessor settings", 'preprocessing', indent=1)
                # Store original settings
                original_normalize_total = self.preprocessor.normalize_total
                original_log1p = self.preprocessor.log1p
                
                # Modify settings
                self.preprocessor.normalize_total = None  # Skip normalization
                self.preprocessor.log1p = False
                # Skip log1p notification included in settings adjustment
                
                # Settings details removed for cleaner output
            else:
                SCLLMOutput.status(f"Will apply normalization", 'loaded', indent=1)
        
        if not skip_preprocessing:
            SCLLMOutput.status(f"Applying preprocessing pipeline", 'preprocessing')
            self.preprocessor(adata_processed, batch_key=kwargs.get('batch_key', None))
            
            # Restore original settings if they were modified
            if 'original_normalize_total' in locals():
                self.preprocessor.normalize_total = original_normalize_total
                self.preprocessor.log1p = original_log1p
                SCLLMOutput.status(f"Preprocessor settings restored", 'loaded', indent=1)
            
            SCLLMOutput.status(f"Preprocessing completed", 'loaded')
            
            # Debug: Check preprocessing results
            if 'X_binned' in adata_processed.layers:
                binned_data = adata_processed.layers['X_binned']
                SCLLMOutput.status(f"Binned data: {binned_data.shape}, {len(np.unique(binned_data))} unique values", 'loaded')
                
                # Verify binning is correct (should be integers from 0 to n_bins-1, plus special values)
                unique_vals = np.unique(binned_data)
                # Detailed binning info removed for cleaner output
                
                if binned_data.max() > self.config.n_bins:
                    SCLLMOutput.status(f"Binned values exceed n_bins ({self.config.n_bins})", 'warning', indent=1)
                if binned_data.min() < -2:
                    SCLLMOutput.status(f"Unexpected negative values in binned data", 'warning', indent=1)
        else:
            SCLLMOutput.status(f"Using existing preprocessed data", 'info')
        
        # AnnData/Scanpy filtering can drop unused categories and renumber domains.
        batch_key = kwargs.get('batch_key', 'batch')
        if batch_key in adata.obs and isinstance(adata.obs[batch_key].dtype, pd.CategoricalDtype):
            adata_processed.obs[batch_key] = adata_processed.obs[batch_key].cat.set_categories(
                adata.obs[batch_key].cat.categories,
                ordered=adata.obs[batch_key].cat.ordered,
            )
        return adata_processed
    
    def predict(self, adata: AnnData, task: str = "annotation", **kwargs) -> Dict[str, Any]:
        """
        Make predictions using the scGPT model.
        
        Args:
            adata: Input AnnData object
            task: Task type ('annotation', 'embedding', 'integration')
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing predictions and metadata
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if task == "integration":
            if adata.n_obs == 0:
                raise ValueError("Integration requires at least one cell")
            if self.model.cell_emb_style != "cls":
                raise ValueError("Integration requires cell_emb_style='cls'")

        # Preprocess data
        adata_processed = self.preprocess(adata, **kwargs)
        
        if task == "annotation":
            return self._predict_annotation(adata_processed, **kwargs)
        elif task == "embedding":
            return self._predict_embedding(adata_processed, **kwargs)
        elif task == "integration":
            return self._predict_integration(adata_processed, **kwargs)
        else:
            raise ValueError(f"Unknown task: {task}. Available tasks: annotation, embedding, integration")
    
    def _predict_annotation(self, adata: AnnData, **kwargs) -> Dict[str, Any]:
        """Predict cell type annotations following Tutorial exactly."""
        # Get gene expression data from binned layer (Tutorial approach)
        input_layer_key = "X_binned"
        
        if input_layer_key not in adata.layers:
            raise ValueError(f"Required layer '{input_layer_key}' not found in adata.layers")
        
        all_counts = (
            adata.layers[input_layer_key].toarray()
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )
        
        SCLLMOutput.status(f"Data shape: {all_counts.shape}", indent=1)
        SCLLMOutput.status(f"Data range: [{all_counts.min():.3f}, {all_counts.max():.3f}]", indent=1)
        
        # Get genes and their vocab IDs (following Tutorial exactly)
        genes = adata.var_names.tolist()
        
        # Tutorial approach: get vocab IDs for all genes
        gene_ids = []
        for gene in genes:
            if gene in self.vocab:
                gene_ids.append(self.vocab[gene])
            else:
                # This should not happen if preprocessing was done correctly
                raise ValueError(f"Gene {gene} not found in vocabulary")
        
        gene_ids = np.array(gene_ids, dtype=int)
        SCLLMOutput.status(f"Gene IDs: {gene_ids.shape[0]} genes mapped", indent=1)
        
        # Tutorial exact tokenization
        SCLLMOutput.status(f"Tokenizing data...", 'preprocessing', indent=1)
        tokenized_data = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=self.config.max_seq_len,
            vocab=self.vocab,
            pad_token=self.config.pad_token,
            pad_value=-2,  # Tutorial uses -2 for padding
            append_cls=True,  # Tutorial appends CLS token
            include_zero_gene=kwargs.get('include_zero_gene', False),
        )
        
        SCLLMOutput.status(f"Tokenized: {tokenized_data['values'].shape[0]} cells x {tokenized_data['values'].shape[1]} tokens", indent=1)
        
        # Tutorial exact masking (even with mask_ratio=0.0)
        mask_ratio = kwargs.get('mask_ratio', 0.0)
        if mask_ratio > 0:
            SCLLMOutput.status(f"Applying masking (ratio={mask_ratio})...", 'preprocessing', indent=1)
        
        input_values = random_mask_value(
            tokenized_data["values"],
            mask_ratio=mask_ratio,
            mask_value=-1,  # Tutorial uses -1 for masking
            pad_value=-2,   # Tutorial uses -2 for padding
        )
        
        # Masking debug info removed for cleaner output
        
        # Create dataset exactly like Tutorial
        dataset_dict = {
            "gene_ids": tokenized_data["genes"],  # Note: Tutorial uses "genes" key
            "values": input_values,
            "target_values": tokenized_data["values"],
        }
        
        # Create data loader - Tutorial approach
        batch_size = kwargs.get('batch_size', 32)
        dataset = SimpleDataset(dataset_dict)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        SCLLMOutput.status(f"Created dataloader: {len(dataloader)} batches (batch_size={batch_size})", indent=1)
        
        # Make predictions following Tutorial exactly
        self.model.eval()
        all_predictions = []
        all_embeddings = []  
        all_logits = []
        
        SCLLMOutput.status(f"Running model inference...", 'predicting', indent=1)
        
        with torch.no_grad():
            pbar = SCLLMOutput.progress_bar(
                total=len(dataloader), 
                desc="Prediction batches", 
                model_name="scGPT"
            )
            
            for batch_idx, batch_data in enumerate(dataloader):
                pbar.update(1)
                # Get batch data and move to device
                input_gene_ids = batch_data["gene_ids"].to(self.device)
                input_values = batch_data["values"].to(self.device)
                
                # Create padding mask (Tutorial approach)
                src_key_padding_mask = input_gene_ids.eq(self.vocab[self.config.pad_token])
                
                if batch_idx == 0:
                    SCLLMOutput.status(f"Batch shape: {input_gene_ids.shape}", indent=2)
                    SCLLMOutput.status(f"Padding tokens: {src_key_padding_mask.sum().item()}", indent=2)
                
                # Tutorial exact model call - critical parameters
                output_dict = self.model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,  # Tutorial: None for inference
                    CLS=True,           # Tutorial: True for classification
                    CCE=False,          # Tutorial: False during evaluation
                    MVC=False,          # Tutorial: False during evaluation
                    ECS=False,          # Tutorial: False during evaluation 
                    do_sample=False,    # Tutorial: False during inference
                )
                
                # Process model outputs - following Tutorial exactly
                if "cls_output" in output_dict:
                    logits = output_dict["cls_output"]  # Tutorial: cls_output
                    predictions = logits.argmax(1).cpu().numpy()
                    all_predictions.append(predictions)
                    all_logits.append(logits.cpu().numpy())
                    
                    # Detailed debug info for first batch
                    if batch_idx == 0:
                        # First batch debug info removed for cleaner output
                        
                        # Check each class logit
                        mean_logits_per_class = logits.mean(dim=0)
                        # Logits per class details removed for cleaner output
                        
                        # Prediction samples removed for cleaner output
                        
                        # Check if model is producing reasonable outputs
                        if logits.std().item() < 0.1:
                            SCLLMOutput.status(f"Low logits variance: {logits.std().item():.4f}", 'warning', indent=2)
                        
                        # Check for gradient/parameter issues
                        if torch.isnan(logits).any():
                            SCLLMOutput.status(f"NaN values in logits!", 'failed', indent=2)
                        if torch.isinf(logits).any():
                            SCLLMOutput.status(f"Inf values in logits!", 'failed', indent=2)
                
                # Get cell embeddings if available
                if "cell_emb" in output_dict:
                    embeddings = output_dict["cell_emb"].cpu().numpy()
                    all_embeddings.append(embeddings)
                    
                    if batch_idx == 0:
                        SCLLMOutput.status(f"Embeddings: {embeddings.shape[1]} dimensions", indent=2)
                elif batch_idx == 0:
                    SCLLMOutput.status(f"No cell embeddings found in output", 'warning', indent=2)
            
            pbar.close()
        
        results = {}
        if all_predictions:
            predictions = np.concatenate(all_predictions)
            results["predictions"] = predictions
            
            # Prediction analysis
            unique_preds, counts = np.unique(predictions, return_counts=True)
            SCLLMOutput.status(f"Predictions: {len(unique_preds)} classes for {len(predictions):,} cells", indent=1)
            
            if len(unique_preds) == 1:
                SCLLMOutput.status(f"All cells predicted as class {unique_preds[0]}", 'warning', indent=1)
                
                if all_logits:
                    all_logits_concat = np.concatenate(all_logits)
                    SCLLMOutput.status(f"Logits analysis: mean={all_logits_concat.mean():.3f}, std={all_logits_concat.std():.3f}", 'info', indent=2)
                    
                    # Check if one class dominates
                    mean_logits_per_class = all_logits_concat.mean(axis=0)
                    dominant_class = mean_logits_per_class.argmax()
                    SCLLMOutput.status(f"Dominant class: {dominant_class}", 'info', indent=2)
                    
        if all_embeddings:
            embeddings = np.concatenate(all_embeddings)
            results["embeddings"] = embeddings
            SCLLMOutput.status(f"Extracted embeddings: {embeddings.shape}", indent=1)
        
        return results
    
    def _predict_embedding(self, adata: AnnData, **kwargs) -> Dict[str, Any]:
        """Get cell embeddings."""
        # Similar to annotation but only return embeddings
        result = self._predict_annotation(adata, **kwargs)
        return {"embeddings": result.get("embeddings")}
    
    def _predict_integration(self, adata: AnnData, **kwargs) -> Dict[str, Any]:
        """Return L2-normalized CLS embeddings in input cell order.

        Inference is unmasked by default. Batch categories must retain the
        category order used during training, including when subsetting cells.
        """
        # Check for batch information
        batch_key = kwargs.get('batch_key', 'batch')
        if batch_key not in adata.obs:
            raise ValueError(f"Batch information '{batch_key}' not found in adata.obs. "
                           f"Integration requires batch labels.")
        
        operation_start("integrate", "scGPT", {
            "batch_key": batch_key,
            "cells": f"{adata.n_obs:,}",
            "genes": f"{adata.n_vars:,}"
        })
        
        # Get gene expression data from binned layer
        input_layer_key = "X_binned"
        if input_layer_key not in adata.layers:
            raise ValueError(f"Required layer '{input_layer_key}' not found in adata.layers")
        
        all_counts = (
            adata.layers[input_layer_key].toarray()
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )
        
        SCLLMOutput.status(f"Data shape: {all_counts.shape}", indent=1)
        SCLLMOutput.status(f"Data range: [{all_counts.min():.3f}, {all_counts.max():.3f}]", indent=1)
        
        # Process batch labels
        batch_labels = adata.obs[batch_key].astype('category').cat.codes.values
        if (batch_labels < 0).any():
            raise ValueError("Integration batch labels must not contain missing values")
        unique_batches = np.unique(batch_labels)
        num_batches = len(unique_batches)
        
        SCLLMOutput.status(f"Batches: {num_batches} batches detected", 'batches', indent=1)
        batch_counts = np.bincount(batch_labels)
        for i, count in enumerate(batch_counts):
            SCLLMOutput.status(f"Batch {i}: {count:,} cells", indent=2)
        
        # Get genes and their vocab IDs 
        genes = adata.var_names.tolist()
        gene_ids = []
        for gene in genes:
            if gene in self.vocab:
                gene_ids.append(self.vocab[gene])
            else:
                raise ValueError(f"Gene {gene} not found in vocabulary")
        
        gene_ids = np.array(gene_ids, dtype=int)
        
        # Tutorial exact tokenization for integration
        SCLLMOutput.status(f"Tokenizing data for integration...", 'preprocessing', indent=1)
        tokenized_data = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=self.config.max_seq_len,
            vocab=self.vocab,
            pad_token=self.config.pad_token,
            pad_value=-2,
            append_cls=True,
            include_zero_gene=kwargs.get('include_zero_gene', True),
        )
        
        # Evaluation uses unmasked values; masking remains an explicit opt-in.
        mask_ratio = kwargs.get('mask_ratio', 0.0)
        input_values = tokenized_data["values"].float()
        if mask_ratio > 0:
            input_values = random_mask_value(
                input_values, mask_ratio=mask_ratio, mask_value=-1, pad_value=-2,
            )

        batch_size = kwargs.get('batch_size', 32)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        input_gene_ids = tokenized_data["genes"]
        padding_mask = input_gene_ids.eq(self.vocab[self.config.pad_token])
        needs_batch_labels = (
            self.model.use_batch_labels or self.model.domain_spec_batchnorm
        )
        # DSBN chooses its domain from the first label of each minibatch.
        # Encode each domain separately, then restore the caller's cell order.
        groups = (
            [np.flatnonzero(batch_labels == label) for label in unique_batches]
            if getattr(self.model, "dsbn", None) is not None
            else [np.arange(adata.n_obs)]
        )
        self.model.eval()
        embeddings = np.empty((adata.n_obs, self.model.d_model), dtype=np.float32)
        with torch.no_grad():
            for indices in groups:
                embeddings[indices] = self.model.encode_batch(
                    input_gene_ids[indices],
                    input_values[indices],
                    src_key_padding_mask=padding_mask[indices],
                    batch_size=batch_size,
                    batch_labels=torch.from_numpy(batch_labels[indices]).long()
                    if needs_batch_labels else None,
                    time_step=0,
                    return_np=True,
                )
        # Match scGPT's integration evaluation representation.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms == 0).any():
            raise RuntimeError("Integration produced non-finite or zero-norm embeddings")
        embeddings /= norms

        results = {}
        if len(embeddings):
            results["embeddings"] = embeddings
            results["batch_labels"] = batch_labels
            results["integrated_embeddings"] = embeddings  # Same as embeddings for compatibility
            
            # Add batch statistics
            results["integration_stats"] = {
                "num_batches": num_batches,
                "batch_distribution": np.bincount(batch_labels).tolist(),
                "total_cells": len(batch_labels)
            }
            
            operation_complete("integrate", {
                "embedding_shape": f"{embeddings.shape}",
                "batches_integrated": num_batches,
                "total_cells": len(batch_labels)
            })
        else:
            raise RuntimeError("No embeddings were generated during integration")
        
        return results
    
    def integrate(self, adata: AnnData, batch_key: str = "batch", **kwargs) -> Dict[str, Any]:
        """
        Perform batch integration using scGPT model.
        
        This method automatically chooses the appropriate integration approach:
        - If model is fine-tuned for integration: uses trained integration capabilities
        - If model is pre-trained only: uses embeddings + post-hoc correction methods
        
        Args:
            adata: Input data with batch information
            batch_key: Column name for batch labels
            **kwargs: Additional parameters including:
                - correction_method: Post-hoc correction method ('combat', 'mnn', 'center_scale', 'none')
                - include_zero_gene: For fine-tuned integration, include
                  zero-expression genes (default True, matching scGPT evaluation).
                  The current fine-tuning helpers use
                  nonzero genes; pass False to use that gene inclusion policy.
                  Sequences exceeding config.max_seq_len randomly sample genes
                  while retaining CLS, so repeated calls may differ.
                - Other integration parameters
            
        Returns:
            Dictionary with integrated embeddings and batch statistics
        """
        if batch_key not in adata.obs:
            raise ValueError(f"Batch information '{batch_key}' not found in adata.obs")
        
        # Check if model is fine-tuned for integration
        has_integration_training = hasattr(self, '_integration_trained') and self._integration_trained
        
        if has_integration_training:
            SCLLMOutput.status(f"Using fine-tuned integration model", 'info')
            # Use the trained integration capabilities
            return self.predict(adata, task="integration", batch_key=batch_key, **kwargs)
        else:
            SCLLMOutput.status(f"Using pre-trained model with post-hoc correction", 'info')
            return self._apply_post_hoc_integration(adata, batch_key=batch_key, **kwargs)
    
    def _apply_post_hoc_integration(self, adata: AnnData, batch_key: str = "batch", **kwargs) -> Dict[str, Any]:
        """Apply post-hoc batch correction methods to pre-trained embeddings."""
        
        # Get pre-trained embeddings
        SCLLMOutput.status(f"Extracting pre-trained embeddings", 'embedding', indent=1)
        embeddings = self.get_embeddings(adata, **kwargs)
        
        batch_labels = adata.obs[batch_key].astype('category').cat.codes.values
        unique_batches = np.unique(batch_labels)
        num_batches = len(unique_batches)
        
        SCLLMOutput.status(f"Found {num_batches} batches", 'info', indent=1)
        
        # Apply post-hoc batch correction methods
        correction_method = kwargs.get('correction_method', 'combat')
        
        if correction_method == 'combat':
            corrected_embeddings = self._apply_combat_correction(embeddings, batch_labels, **kwargs)
        elif correction_method == 'mnn':
            corrected_embeddings = self._apply_mnn_correction(embeddings, batch_labels, **kwargs)
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
                'batch_distribution': np.bincount(batch_labels).tolist(),
                'total_cells': len(batch_labels),
                'correction_applied': correction_method,
                'method': 'post_hoc_correction'
            }
        }
        
        SCLLMOutput.status(f"Integration completed using {correction_method} correction", 'loaded')
        return results
    
    def _apply_combat_correction(self, embeddings: np.ndarray, batch_labels: np.ndarray, **kwargs) -> np.ndarray:
        """Apply ComBat-style batch correction to embeddings."""
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
    
    def _apply_mnn_correction(self, embeddings: np.ndarray, batch_labels: np.ndarray, **kwargs) -> np.ndarray:
        """Apply MNN-inspired batch correction."""
        SCLLMOutput.status(f"Applying MNN-inspired correction", 'preprocessing', indent=1)
        
        try:
            from sklearn.neighbors import NearestNeighbors
            
            corrected = embeddings.copy()
            unique_batches = np.unique(batch_labels)
            
            if len(unique_batches) < 2:
                SCLLMOutput.status(f"Only one batch, no correction needed", 'info', indent=2)
                return corrected
            
            # Find mutual nearest neighbors between batches
            k = kwargs.get('mnn_k', 10)
            
            for i, batch1 in enumerate(unique_batches[:-1]):
                for batch2 in unique_batches[i+1:]:
                    mask1 = batch_labels == batch1
                    mask2 = batch_labels == batch2
                    
                    data1 = embeddings[mask1]
                    data2 = embeddings[mask2]
                    
                    if data1.shape[0] > k and data2.shape[0] > k:
                        # Find nearest neighbors
                        nn1 = NearestNeighbors(n_neighbors=min(k, data2.shape[0]))
                        nn2 = NearestNeighbors(n_neighbors=min(k, data1.shape[0]))
                        
                        nn1.fit(data2)
                        nn2.fit(data1)
                        
                        # Find MNNs and apply correction
                        distances1, indices1 = nn1.kneighbors(data1)
                        distances2, indices2 = nn2.kneighbors(data2)
                        
                        # Simple correction: move towards batch centroid
                        centroid1 = data1.mean(axis=0)
                        centroid2 = data2.mean(axis=0)
                        correction_vector = (centroid1 + centroid2) / 2
                        
                        # Apply partial correction
                        alpha = kwargs.get('mnn_alpha', 0.3)
                        corrected[mask1] = (1 - alpha) * data1 + alpha * correction_vector
                        corrected[mask2] = (1 - alpha) * data2 + alpha * correction_vector
            
            SCLLMOutput.status(f"MNN correction applied", 'loaded', indent=2)
            return corrected
            
        except Exception as e:
            SCLLMOutput.status(f"MNN failed: {e}", 'warning', indent=2)
            return embeddings
    
    def _apply_center_scale_correction(self, embeddings: np.ndarray, batch_labels: np.ndarray, **kwargs) -> np.ndarray:
        """Apply simple centering and scaling correction."""
        SCLLMOutput.status(f"Applying center-scale correction", 'preprocessing', indent=1)
        
        corrected = embeddings.copy()
        unique_batches = np.unique(batch_labels)
        
        # Calculate global statistics
        global_mean = embeddings.mean(axis=0)
        global_std = embeddings.std(axis=0) + 1e-8
        
        # Correct each batch
        for batch_id in unique_batches:
            batch_mask = batch_labels == batch_id
            batch_data = corrected[batch_mask]
            
            if batch_data.shape[0] > 1:
                # Center to global mean
                batch_mean = batch_data.mean(axis=0)
                batch_centered = batch_data - batch_mean + global_mean
                
                # Scale to global variance
                batch_std = batch_data.std(axis=0) + 1e-8
                scale_factor = global_std / batch_std
                scale_factor = np.clip(scale_factor, 0.5, 2.0)  # Avoid extreme scaling
                
                corrected[batch_mask] = batch_centered * scale_factor
        
        SCLLMOutput.status(f"Center-scale applied to {len(unique_batches)} batches", 'loaded', indent=2)
        return corrected
    
    def fine_tune(self, 
                  train_adata: AnnData,
                  valid_adata: Optional[AnnData] = None,
                  task: str = "annotation",
                  **kwargs) -> Dict[str, Any]:
        """
        Fine-tune the scGPT model for cell type annotation.
        
        Args:
            train_adata: Training data with 'celltype' in .obs
            valid_adata: Validation data (optional)
            task: Task type ('annotation')
            **kwargs: Training parameters
            
        Returns:
            Training results and metrics
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Validate input data
        if 'celltype' not in train_adata.obs:
            raise ValueError("train_adata must have 'celltype' column in .obs")
        
        SCLLMOutput.section_header(f"Fine-tuning for {task} task", "scGPT")
        
        # Get training parameters
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 32)
        lr = kwargs.get('lr', 1e-4)
        mask_ratio = kwargs.get('mask_ratio', 0.0)  # No masking for annotation task
        
        # Show training configuration
        SCLLMOutput.model_info("Training Configuration", {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'mask_ratio': mask_ratio
        })
        
        # Prepare cell type mapping
        unique_celltypes = train_adata.obs['celltype'].astype('category').cat.categories
        celltype_to_id = {ct: i for i, ct in enumerate(unique_celltypes)}
        id_to_celltype = {i: ct for i, ct in enumerate(unique_celltypes)}
        n_celltypes = len(unique_celltypes)
        
        SCLLMOutput.status(f"Cell types: {n_celltypes} classes", 'info')
        
        # Add celltype_id to data
        train_adata.obs['celltype_id'] = train_adata.obs['celltype'].map(celltype_to_id)
        if valid_adata is not None:
            if 'celltype' not in valid_adata.obs:
                raise ValueError("valid_adata must have 'celltype' column in .obs")
            valid_adata.obs['celltype_id'] = valid_adata.obs['celltype'].map(celltype_to_id)
        
        # Preprocess data
        SCLLMOutput.status(f"Preprocessing data...", 'preprocessing')
        train_processed = self.preprocess(train_adata, **kwargs)
        valid_processed = self.preprocess(valid_adata, **kwargs) if valid_adata is not None else None
        
        # Update model's classifier if needed
        if hasattr(self.model, 'cls_decoder') and hasattr(self.model.cls_decoder, 'out_layer'):
            current_n_cls = self.model.cls_decoder.out_layer.out_features
            if current_n_cls != n_celltypes:
                SCLLMOutput.status(f"Updating classifier: {current_n_cls} → {n_celltypes} classes", 'preprocessing')
                # Get input dimension
                if hasattr(self.model.cls_decoder, '_decoder') and len(self.model.cls_decoder._decoder) > 0:
                    prev_layer = self.model.cls_decoder._decoder[-2]
                    if hasattr(prev_layer, 'out_features'):
                        prev_dim = prev_layer.out_features
                    else:
                        prev_dim = self.model.d_model
                else:
                    prev_dim = self.model.d_model
                
                # Replace classifier
                import torch.nn as nn
                self.model.cls_decoder.out_layer = nn.Linear(prev_dim, n_celltypes)
                self.model.cls_decoder.out_layer.to(self.device)
                SCLLMOutput.status(f"Classifier updated", 'loaded')
        
        # Prepare training data
        train_results = self._prepare_training_data(train_processed, mask_ratio)
        valid_results = self._prepare_training_data(valid_processed, mask_ratio) if valid_processed is not None else None
        
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        best_accuracy = 0.0
        best_model_state = None
        training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        SCLLMOutput.status(f"Starting training...", 'training')
        
        with ModelProgressManager("Fine-tuning", "scGPT", epochs) as progress:
            for epoch in range(epochs):
                # Training
                train_loss, train_acc = self._train_epoch(
                    train_results, optimizer, criterion, batch_size
                )
                training_history['train_loss'].append(train_loss)
                training_history['train_acc'].append(train_acc)
                
                # Validation
                if valid_results is not None:
                    val_loss, val_acc = self._validate_epoch(valid_results, criterion, batch_size)
                    training_history['val_loss'].append(val_loss)
                    training_history['val_acc'].append(val_acc)
                    
                    # Update progress with metrics
                    progress.update(1, 
                        train_loss=f"{train_loss:.4f}",
                        train_acc=f"{train_acc:.4f}",  
                        val_loss=f"{val_loss:.4f}",
                        val_acc=f"{val_acc:.4f}"
                    )
                    
                    # Save best model
                    if val_acc > best_accuracy:
                        best_accuracy = val_acc
                        best_model_state = copy.deepcopy(self.model.state_dict())
                        SCLLMOutput.status(f"New best validation accuracy: {best_accuracy:.4f}", 'best')
                else:
                    progress.update(1, 
                        train_loss=f"{train_loss:.4f}",
                        train_acc=f"{train_acc:.4f}"
                    )
                    if train_acc > best_accuracy:
                        best_accuracy = train_acc
                        best_model_state = copy.deepcopy(self.model.state_dict())
                
                scheduler.step()
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            SCLLMOutput.status(f"Loaded best model (accuracy: {best_accuracy:.4f})", 'loaded')
        
        # Store celltype mapping for inference
        self.celltype_mapping = {
            'celltype_to_id': celltype_to_id,
            'id_to_celltype': id_to_celltype,
            'n_celltypes': n_celltypes
        }
        
        return {
            'best_accuracy': best_accuracy,
            'training_history': training_history,
            'celltype_mapping': self.celltype_mapping,
            'n_celltypes': n_celltypes
        }
    
    def train_integration(self, 
                         train_adata: AnnData,
                         valid_adata: Optional[AnnData] = None,
                         batch_key: str = "batch",
                         **kwargs) -> Dict[str, Any]:
        """
        Train the scGPT model for batch integration following Tutorial_Integration.
        
        Args:
            train_adata: Training data with batch labels in .obs[batch_key]
            valid_adata: Validation data (optional)
            batch_key: Column name for batch labels (default: "batch")
            **kwargs: Training parameters
            
        Returns:
            Training results and metrics
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Validate input data
        if batch_key not in train_adata.obs:
            raise ValueError(f"train_adata must have '{batch_key}' column in .obs")
        
        SCLLMOutput.status(f"Starting scGPT integration training", 'training')
        
        # Get training parameters (Tutorial_Integration defaults)
        epochs = kwargs.get('epochs', 20)  # Integration typically needs more epochs
        batch_size = kwargs.get('batch_size', 32)
        lr = kwargs.get('lr', 1e-4)
        mask_ratio = kwargs.get('mask_ratio', 0.4)  # Higher masking for integration
        dab_weight = kwargs.get('dab_weight', 1.0)
        ecs_weight = kwargs.get('ecs_weight', 10.0)
        gepc_weight = kwargs.get('gepc_weight', 1.0)
        
        # Process batch labels
        train_batch_labels = train_adata.obs[batch_key].astype('category').cat.codes.values
        unique_batches = np.unique(train_batch_labels)
        num_batches = len(unique_batches)
        
        SCLLMOutput.status(f"Batch information:", 'info')
        SCLLMOutput.status(f"Number of batches: {num_batches}", indent=1)
        SCLLMOutput.status(f"Training cells: {np.bincount(train_batch_labels)}", indent=1)
        
        if valid_adata is not None:
            if batch_key not in valid_adata.obs:
                raise ValueError(f"valid_adata must have '{batch_key}' column in .obs")
            valid_batch_labels = valid_adata.obs[batch_key].astype('category').cat.codes.values
            SCLLMOutput.status(f"Validation cells: {np.bincount(valid_batch_labels)}", indent=1)
        
        # Update model configuration for integration
        self.config.use_batch_labels = True
        self.config.num_batch_labels = num_batches
        self.config.do_dab = kwargs.get('do_dab', True)
        self.config.do_mvc = kwargs.get('do_mvc', True)  # GEPC
        self.config.do_ecs = kwargs.get('do_ecs', True)
        self.config.domain_spec_batchnorm = kwargs.get('domain_spec_batchnorm', True)
        
        SCLLMOutput.status(f"Integration training configuration:", 'info')
        SCLLMOutput.status(f"epochs: {epochs}, batch_size: {batch_size}, lr: {lr}", indent=1)
        SCLLMOutput.status(f"mask_ratio: {mask_ratio}", indent=1)
        SCLLMOutput.status(f"DAB: {self.config.do_dab}, MVC: {self.config.do_mvc}, ECS: {self.config.do_ecs}", indent=1)
        SCLLMOutput.status(f"DSBN: {self.config.domain_spec_batchnorm}", indent=1)
        SCLLMOutput.status(f"Loss weights - DAB: {dab_weight}, ECS: {ecs_weight}, GEPC: {gepc_weight}", indent=1)
        
        # Check if model needs to be reconfigured for integration
        if hasattr(self.model, 'use_batch_labels') and not self.model.use_batch_labels:
            SCLLMOutput.status(f"Model reconfiguration needed for batch integration", 'warning')
            SCLLMOutput.status(f"Reinitializing model with batch support", 'preprocessing', indent=1)
            self._reconfigure_model_for_integration(num_batches)
        
        # Preprocess data
        train_processed = self.preprocess(train_adata, **kwargs)
        valid_processed = self.preprocess(valid_adata, **kwargs) if valid_adata is not None else None
        
        # Prepare integration training data
        train_results = self._prepare_integration_data(train_processed, train_batch_labels, mask_ratio)
        valid_results = self._prepare_integration_data(valid_processed, valid_batch_labels, mask_ratio) if valid_processed is not None else None
        
        # Setup training components
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        
        # Integration uses multiple loss functions
        mse_criterion = torch.nn.MSELoss()
        dab_criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        best_loss = float('inf')
        best_model_state = None
        training_history = {
            'train_total_loss': [], 'train_mse': [], 'train_dab': [], 'train_ecs': [],
            'val_total_loss': [], 'val_mse': [], 'val_dab': [], 'val_ecs': []
        }
        
        for epoch in range(epochs):
            # Integration epoch info handled by progress bar
            
            # Training
            train_losses = self._train_integration_epoch(
                train_results, optimizer, mse_criterion, dab_criterion, 
                batch_size, dab_weight, ecs_weight, gepc_weight
            )
            
            for key, value in train_losses.items():
                training_history[f'train_{key}'].append(value)
            
            # Validation
            if valid_results is not None:
                val_losses = self._validate_integration_epoch(
                    valid_results, mse_criterion, dab_criterion, 
                    batch_size, dab_weight, ecs_weight, gepc_weight
                )
                
                for key, value in val_losses.items():
                    training_history[f'val_{key}'].append(value)
                
                total_val_loss = val_losses['total_loss']
                # Training metrics displayed in progress bar
                #if train_losses['dab'] > 0:
                    # DAB metrics displayed in progress bar
                #if train_losses['ecs'] > 0:
                    # ECS metrics displayed in progress bar
                
                # Save best model
                if total_val_loss < best_loss:
                    best_loss = total_val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    SCLLMOutput.status(f"New best validation loss: {best_loss:.4f}", 'best')
            else:
                total_train_loss = train_losses['total_loss']
                # Training loss displayed in progress bar
                if total_train_loss < best_loss:
                    best_loss = total_train_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
            
            scheduler.step()
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            SCLLMOutput.status(f"Best model loaded: loss={best_loss:.4f}", 'loaded')
        
        # Store integration metadata
        self.integration_metadata = {
            'batch_key': batch_key,
            'num_batches': num_batches,
            'batch_labels': {i: f"batch_{i}" for i in range(num_batches)}
        }
        
        # Mark as integration trained
        self._integration_trained = True
        
        SCLLMOutput.status(f"Integration training completed", 'complete')
        
        return {
            'best_loss': best_loss,
            'training_history': training_history,
            'integration_metadata': self.integration_metadata,
            'num_batches': num_batches
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
        SCLLMOutput.data_summary(adata, model_name="scGPT")
        operation_start("get_embeddings", "scGPT", {
            "cells": f"{adata.n_obs:,}",
            "genes": f"{adata.n_vars:,}"
        })
        
        result = self.predict(adata, task="embedding", **kwargs)
        
        operation_complete("get_embeddings", {
            "embedding_shape": f"{result['embeddings'].shape}",
            "embedding_dim": result['embeddings'].shape[1]
        })
        
        return result["embeddings"]
    
    def _prepare_training_data(self, adata: AnnData, mask_ratio: float = 0.0) -> Dict[str, torch.Tensor]:
        """Prepare data for training."""
        # Get expression data
        input_layer_key = "X_binned"
        if input_layer_key not in adata.layers:
            raise ValueError(f"Required layer '{input_layer_key}' not found")
        
        all_counts = (
            adata.layers[input_layer_key].toarray()
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )
        
        genes = adata.var_names.tolist()
        gene_ids = np.array([self.vocab[gene] for gene in genes], dtype=int)
        
        # Tokenize
        tokenized_data = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=self.config.max_seq_len,
            vocab=self.vocab,
            pad_token=self.config.pad_token,
            pad_value=-2,
            append_cls=True,
            include_zero_gene=False,
        )
        
        # Apply masking
        input_values = random_mask_value(
            tokenized_data["values"],
            mask_ratio=mask_ratio,
            mask_value=-1,
            pad_value=-2,
        )
        
        # Get labels
        celltype_labels = torch.from_numpy(adata.obs['celltype_id'].values.astype(int)).long()
        
        return {
            "gene_ids": tokenized_data["genes"],
            "values": input_values,
            "target_values": tokenized_data["values"],
            "celltype_labels": celltype_labels,
        }
    
    def _reconfigure_model_for_integration(self, num_batches: int):
        """Reconfigure model to support batch integration training."""
        
        # Store current model state
        current_state = self.model.state_dict()
        
        # Create new model with batch label support
        from .scgpt.model import TransformerModel
        
        old_model = self.model
        self.model = TransformerModel(
            ntoken=len(self.vocab),
            d_model=self.config.embsize,
            nhead=self.config.nhead,
            d_hid=self.config.d_hid,
            nlayers=self.config.nlayers,
            nlayers_cls=self.config.nlayers_cls,
            n_cls=self.n_cls,
            vocab=self.vocab,
            dropout=self.config.dropout,
            pad_token=self.config.pad_token,
            pad_value=-2,
            do_mvc=self.config.do_mvc,
            do_dab=self.config.do_dab,
            use_batch_labels=True,  # Enable batch labels
            num_batch_labels=num_batches,  # Set number of batches
            domain_spec_batchnorm=self.config.domain_spec_batchnorm,
            input_emb_style=self.config.input_emb_style,
            n_input_bins=self.config.n_bins + 2,
            cell_emb_style=self.config.cell_emb_style,
            mvc_decoder_style="inner product",
            ecs_threshold=0.3,
            explicit_zero_prob=False,
            use_fast_transformer=self.config.fast_transformer,
            fast_transformer_backend="flash",
            pre_norm=self.config.pre_norm,
        )
        
        # Transfer compatible weights from old model
        new_state = self.model.state_dict()
        transferred_keys = 0
        
        for key, value in current_state.items():
            if key in new_state and new_state[key].shape == value.shape:
                new_state[key] = value
                transferred_keys += 1
        
        self.model.load_state_dict(new_state)
        self.model.to(self.device)
        
        SCLLMOutput.status(f"Model reconfigured with batch support", 'loaded', indent=1)
        SCLLMOutput.status(f"Transferred {transferred_keys}/{len(new_state)} parameters", indent=1)
        SCLLMOutput.status(f"Batch labels: {num_batches} batches", indent=1)
    
    def _prepare_integration_data(self, adata: AnnData, batch_labels: np.ndarray, mask_ratio: float = 0.4) -> Dict[str, torch.Tensor]:
        """Prepare data for integration training."""
        # Get expression data
        input_layer_key = "X_binned"
        if input_layer_key not in adata.layers:
            raise ValueError(f"Required layer '{input_layer_key}' not found")
        
        all_counts = (
            adata.layers[input_layer_key].toarray()
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )
        
        genes = adata.var_names.tolist()
        gene_ids = np.array([self.vocab[gene] for gene in genes], dtype=int)
        
        # Tokenize
        tokenized_data = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=self.config.max_seq_len,
            vocab=self.vocab,
            pad_token=self.config.pad_token,
            pad_value=-2,
            append_cls=True,
            include_zero_gene=False,
        )
        
        # Apply masking (higher ratio for integration)
        input_values = random_mask_value(
            tokenized_data["values"],
            mask_ratio=mask_ratio,
            mask_value=-1,
            pad_value=-2,
        )
        
        return {
            "gene_ids": tokenized_data["genes"],
            "values": input_values,
            "target_values": tokenized_data["values"],
            "batch_labels": torch.from_numpy(batch_labels).long(),
        }
    
    def _train_integration_epoch(self, data: Dict[str, torch.Tensor], optimizer, mse_criterion, dab_criterion, 
                               batch_size: int, dab_weight: float, ecs_weight: float, gepc_weight: float) -> Dict[str, float]:
        """Train integration for one epoch."""
        self.model.train()
        
        dataset = SimpleDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        total_loss = 0.0
        total_mse = 0.0
        total_dab = 0.0
        total_ecs = 0.0
        num_batches = 0
        
        for batch_data in dataloader:
            input_gene_ids = batch_data["gene_ids"].to(self.device)
            input_values = batch_data["values"].to(self.device)
            target_values = batch_data["target_values"].to(self.device)
            batch_labels = batch_data["batch_labels"].to(self.device)
            
            src_key_padding_mask = input_gene_ids.eq(self.vocab[self.config.pad_token])
            
            optimizer.zero_grad()
            
            # Integration model forward pass
            output_dict = self.model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if self.config.use_batch_labels else None,
                CLS=False,                    # No classification for integration
                CCE=False,                    # Can be enabled if needed
                MVC=self.config.do_mvc,      # Masked Value Prediction (GEPC)
                ECS=self.config.do_ecs,      # Elastic Cell Similarity
                do_sample=False,
            )
            
            # Compute integration losses
            loss = 0.0
            mse_loss = 0.0
            dab_loss = 0.0
            ecs_loss = 0.0
            
            # 1. MLM (Masked Language Modeling) loss - reconstruction
            if "mlm_output" in output_dict:
                # Create mask for loss computation
                masked_positions = (input_values == -1)  # -1 is mask_value
                if masked_positions.any():
                    mse_loss = mse_criterion(
                        output_dict["mlm_output"][masked_positions],
                        target_values[masked_positions]
                    )
                    loss += mse_loss
            
            # 2. GEPC (MVC) loss - Gene Expression Prediction for Cells
            if self.config.do_mvc and "mvc_output" in output_dict:
                masked_positions = (input_values == -1)
                if masked_positions.any():
                    gepc_loss = mse_criterion(
                        output_dict["mvc_output"][masked_positions],
                        target_values[masked_positions]
                    )
                    loss += gepc_weight * gepc_loss
            
            # 3. DAB (Domain Adversarial Batch) loss
            if self.config.do_dab and "dab_output" in output_dict:
                dab_loss = dab_criterion(output_dict["dab_output"], batch_labels)
                loss += dab_weight * dab_loss
            
            # 4. ECS (Elastic Cell Similarity) loss
            if self.config.do_ecs and "loss_ecs" in output_dict:
                ecs_loss = output_dict["loss_ecs"]
                loss += ecs_weight * ecs_loss
            
            if torch.isnan(loss):
                SCLLMOutput.status(f"NaN loss detected, skipping batch", 'warning')
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item() if isinstance(mse_loss, torch.Tensor) else mse_loss
            total_dab += dab_loss.item() if isinstance(dab_loss, torch.Tensor) else dab_loss
            total_ecs += ecs_loss.item() if isinstance(ecs_loss, torch.Tensor) else ecs_loss
            num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'mse': total_mse / num_batches,
            'dab': total_dab / num_batches,
            'ecs': total_ecs / num_batches,
        }
    
    def _validate_integration_epoch(self, data: Dict[str, torch.Tensor], mse_criterion, dab_criterion,
                                  batch_size: int, dab_weight: float, ecs_weight: float, gepc_weight: float) -> Dict[str, float]:
        """Validate integration for one epoch."""
        self.model.eval()
        
        dataset = SimpleDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.0
        total_mse = 0.0
        total_dab = 0.0
        total_ecs = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                input_gene_ids = batch_data["gene_ids"].to(self.device)
                input_values = batch_data["values"].to(self.device)
                target_values = batch_data["target_values"].to(self.device)
                batch_labels = batch_data["batch_labels"].to(self.device)
                
                src_key_padding_mask = input_gene_ids.eq(self.vocab[self.config.pad_token])
                
                output_dict = self.model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if self.config.use_batch_labels else None,
                    CLS=False,
                    CCE=False,
                    MVC=self.config.do_mvc,
                    ECS=self.config.do_ecs,
                    do_sample=False,
                )
                
                # Compute validation losses (same as training)
                loss = 0.0
                mse_loss = 0.0
                dab_loss = 0.0
                ecs_loss = 0.0
                
                if "mlm_output" in output_dict:
                    masked_positions = (input_values == -1)
                    if masked_positions.any():
                        mse_loss = mse_criterion(
                            output_dict["mlm_output"][masked_positions],
                            target_values[masked_positions]
                        )
                        loss += mse_loss
                
                if self.config.do_mvc and "mvc_output" in output_dict:
                    masked_positions = (input_values == -1)
                    if masked_positions.any():
                        gepc_loss = mse_criterion(
                            output_dict["mvc_output"][masked_positions],
                            target_values[masked_positions]
                        )
                        loss += gepc_weight * gepc_loss
                
                if self.config.do_dab and "dab_output" in output_dict:
                    dab_loss = dab_criterion(output_dict["dab_output"], batch_labels)
                    loss += dab_weight * dab_loss
                
                if self.config.do_ecs and "loss_ecs" in output_dict:
                    ecs_loss = output_dict["loss_ecs"]
                    loss += ecs_weight * ecs_loss
                
                total_loss += loss.item()
                total_mse += mse_loss.item() if isinstance(mse_loss, torch.Tensor) else mse_loss
                total_dab += dab_loss.item() if isinstance(dab_loss, torch.Tensor) else dab_loss
                total_ecs += ecs_loss.item() if isinstance(ecs_loss, torch.Tensor) else ecs_loss
                num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'mse': total_mse / num_batches,
            'dab': total_dab / num_batches,
            'ecs': total_ecs / num_batches,
        }
    
    def _train_epoch(self, data: Dict[str, torch.Tensor], optimizer, criterion, batch_size: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        dataset = SimpleDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_data in dataloader:
            input_gene_ids = batch_data["gene_ids"].to(self.device)
            input_values = batch_data["values"].to(self.device)
            celltype_labels = batch_data["celltype_labels"].to(self.device)
            
            src_key_padding_mask = input_gene_ids.eq(self.vocab[self.config.pad_token])
            
            optimizer.zero_grad()
            
            output_dict = self.model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
                CLS=True,
                CCE=False,
                MVC=False,
                ECS=False,
                do_sample=False,
            )
            
            if "cls_output" in output_dict:
                logits = output_dict["cls_output"]
                loss = criterion(logits, celltype_labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                predictions = logits.argmax(1)
                total_correct += (predictions == celltype_labels).sum().item()
                total_samples += celltype_labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, data: Dict[str, torch.Tensor], criterion, batch_size: int) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        dataset = SimpleDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                input_gene_ids = batch_data["gene_ids"].to(self.device)
                input_values = batch_data["values"].to(self.device)
                celltype_labels = batch_data["celltype_labels"].to(self.device)
                
                src_key_padding_mask = input_gene_ids.eq(self.vocab[self.config.pad_token])
                
                output_dict = self.model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    CLS=True,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=False,
                )
                
                if "cls_output" in output_dict:
                    logits = output_dict["cls_output"]
                    loss = criterion(logits, celltype_labels)
                    
                    total_loss += loss.item()
                    predictions = logits.argmax(1)
                    total_correct += (predictions == celltype_labels).sum().item()
                    total_samples += celltype_labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def predict_celltypes(self, query_adata: AnnData, **kwargs) -> Dict[str, Any]:
        """
        Predict cell types for query data using fine-tuned model.
        
        Args:
            query_adata: Query data to predict
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if not hasattr(self, 'celltype_mapping'):
            raise ValueError("Model has not been fine-tuned. Call fine_tune() first.")
        
        operation_start("predict_celltypes", "scGPT", {
            "cells": f"{query_adata.n_obs:,}",
            "genes": f"{query_adata.n_vars:,}"
        })
        
        # Preprocess query data
        SCLLMOutput.status(f"Preprocessing query data...", 'preprocessing', indent=1)
        query_processed = self.preprocess(query_adata, **kwargs)
        
        # Get predictions
        results = self._predict_annotation(query_processed, **kwargs)
        
        if 'predictions' in results:
            # Convert IDs to cell type names
            id_to_celltype = self.celltype_mapping['id_to_celltype']
            predicted_celltypes = [id_to_celltype.get(pred, f"Unknown_{pred}") 
                                 for pred in results['predictions']]
            
            results['predicted_celltypes'] = predicted_celltypes
            
            # Show prediction summary
            from collections import Counter
            type_counts = Counter(predicted_celltypes)
            
            operation_complete("predict_celltypes", {
                "total_cells": len(predicted_celltypes),
                "unique_types": len(type_counts),
                "most_common": type_counts.most_common(1)[0][0] if type_counts else "None"
            })
            
            SCLLMOutput.status(f"Cell type distribution:", indent=1)
            for celltype, count in type_counts.most_common():
                percentage = count / len(predicted_celltypes) * 100
                SCLLMOutput.status(f"{celltype}: {count:,} cells ({percentage:.1f}%)", indent=2)
        
        return results
    
    def _save_model_specific(self, save_path: Path, **kwargs) -> None:
        """Save scGPT model."""
        if self.model is not None:
            torch.save(self.model.state_dict(), save_path / "model.pt")
        
        if self.vocab is not None:
            self.vocab.save_json(save_path / "vocab.json")
        
        if self.config is not None:
            with open(save_path / "config.json", 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
        
        # Save celltype mapping if available
        if hasattr(self, 'celltype_mapping'):
            with open(save_path / "celltype_mapping.json", 'w') as f:
                json.dump(self.celltype_mapping, f, indent=2)
            SCLLMOutput.status(f"Celltype mapping saved", 'loaded')
    
    def load_celltype_mapping(self, model_path: Union[str, Path]) -> None:
        """Load celltype mapping from saved model."""
        model_path = Path(model_path)
        mapping_file = model_path / "celltype_mapping.json"
        
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                self.celltype_mapping = json.load(f)
            SCLLMOutput.status(f"Loaded celltype mapping: {self.celltype_mapping['n_celltypes']} types", 'loaded')
        else:
            SCLLMOutput.status(f"No celltype mapping found", 'warning')


class SimpleDataset(Dataset):
    """Simple dataset class for tokenized data."""
    
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data
        # Determine the key to use for length
        self.length_key = None
        for key in ["genes", "gene_ids"]:
            if key in data:
                self.length_key = key
                break
        if self.length_key is None:
            raise ValueError("No suitable key found for dataset length")
    
    def __len__(self):
        return self.data[self.length_key].shape[0]
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}
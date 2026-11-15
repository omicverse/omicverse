"""
CellPLM model implementation with simplified interface.
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

# Import CellPLM components with error handling
try:
    from .CellPLM.pipeline.cell_type_annotation import (
        CellTypeAnnotationPipeline, 
        CellTypeAnnotationDefaultPipelineConfig, 
        CellTypeAnnotationDefaultModelConfig
    )
    from .CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline
    from .CellPLM.pipeline.imputation import (
        ImputationPipeline, 
        ImputationDefaultPipelineConfig, 
        ImputationDefaultModelConfig
    )
    from .CellPLM.utils import set_seed
    _cellplm_imports_available = True
except ImportError as e:
    import warnings
    warnings.warn(f"CellPLM components not available: {e}")
    _cellplm_imports_available = False
    
    # Create placeholder classes
    class CellTypeAnnotationPipeline:
        def __init__(self, *args, **kwargs):
            raise ImportError("CellPLM CellTypeAnnotationPipeline not available due to missing dependencies")
    
    class CellEmbeddingPipeline:
        def __init__(self, *args, **kwargs):
            raise ImportError("CellPLM CellEmbeddingPipeline not available due to missing dependencies")
    
    class ImputationPipeline:
        def __init__(self, *args, **kwargs):
            raise ImportError("CellPLM ImputationPipeline not available due to missing dependencies")
    
    def set_seed(*args, **kwargs):
        raise ImportError("CellPLM set_seed not available due to missing dependencies")
    
    CellTypeAnnotationDefaultPipelineConfig = {}
    CellTypeAnnotationDefaultModelConfig = {}
    ImputationDefaultPipelineConfig = {}
    ImputationDefaultModelConfig = {}


class CellPLMModel(SCLLMBase):
    """
    CellPLM model implementation.
    
    This class provides a unified interface for CellPLM operations including:
    - Cell type annotation
    - Cell embedding extraction  
    - Spatial imputation
    - Batch integration via embeddings
    """
    
    def __init__(self, device: Optional[str] = None, **kwargs):
        """
        Initialize the CellPLM model wrapper.
        
        Args:
            device: Device to run the model on
            **kwargs: Additional parameters
        """
        super().__init__("cellplm", device)
        
        # Model components
        self.annotation_pipeline = None
        self.embedding_pipeline = None
        self.imputation_pipeline = None
        
        # Model configuration
        self.pretrain_version = kwargs.get('pretrain_version', '20231027_85M')
        self.pretrain_directory = kwargs.get('pretrain_directory', './ckpt')
        
        # Task-specific configurations
        self.annotation_config = None
        self.embedding_config = None
        self.imputation_config = None
        
        # Model state
        self.current_task = None
        self.fitted_tasks = set()
        
    def load_model(self, model_path: Union[str, Path], **kwargs) -> None:
        """
        Load CellPLM model components.
        
        Args:
            model_path: Path to the model checkpoint directory
            **kwargs: Additional parameters including:
                - pretrain_version: Version of pretrained model
                - task: Specific task to load ('annotation', 'embedding', 'imputation')
        """
        if not _cellplm_imports_available:
            raise ImportError("CellPLM dependencies not available")
        
        SCLLMOutput.status(f"📥 Loading CellPLM model from {model_path}...")
        
        self.pretrain_directory = str(model_path)
        self.pretrain_version = kwargs.get('pretrain_version', self.pretrain_version)
        
        task = kwargs.get('task', 'all')
        
        # Calculate total steps for progress bar
        tasks_to_load = []
        if task in ['annotation', 'all']:
            tasks_to_load.append('annotation')
        if task in ['embedding', 'all']:
            tasks_to_load.append('embedding')
        if task in ['imputation', 'all']:
            tasks_to_load.append('imputation')
        
        try:
            with tqdm(total=len(tasks_to_load), desc="Loading pipelines", ncols=100) as pbar:
                if 'annotation' in tasks_to_load:
                    pbar.set_description("Loading annotation pipeline...")
                    self._load_annotation_pipeline(**kwargs)
                    pbar.update(1)
                
                if 'embedding' in tasks_to_load:
                    pbar.set_description("Loading embedding pipeline...")
                    self._load_embedding_pipeline(**kwargs)
                    pbar.update(1)
                    
                if 'imputation' in tasks_to_load:
                    pbar.set_description("Loading imputation pipeline...")
                    self._load_imputation_pipeline(**kwargs)
                    pbar.update(1)
                
            self.is_loaded = True
            SCLLMOutput.status(f" CellPLM model loaded from {model_path}", "loaded")
            SCLLMOutput.status(f"- Loaded pipelines: {', '.join(tasks_to_load)}", indent=1)
            SCLLMOutput.status(f"- Pretrain version: {self.pretrain_version}", indent=1)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load CellPLM model: {e}")
    
    def _load_annotation_pipeline(self, **kwargs):
        """Load the cell type annotation pipeline."""
        # Set up default model config for annotation
        model_config = CellTypeAnnotationDefaultModelConfig.copy()
        
        # Update with user-provided configuration
        model_config.update(kwargs.get('model_config', {}))
        
        # out_dim will be set during fine-tuning based on data
        if 'out_dim' not in model_config:
            model_config['out_dim'] = 10  # Default placeholder
        
        self.annotation_pipeline = CellTypeAnnotationPipeline(
            pretrain_prefix=self.pretrain_version,
            overwrite_config=model_config,
            pretrain_directory=self.pretrain_directory
        )
        
        self.annotation_config = CellTypeAnnotationDefaultPipelineConfig.copy()
        self.annotation_config.update(kwargs.get('pipeline_config', {}))
        self.annotation_config['device'] = str(self.device)
    
    def _load_embedding_pipeline(self, **kwargs):
        """Load the cell embedding pipeline."""
        self.embedding_pipeline = CellEmbeddingPipeline(
            pretrain_prefix=self.pretrain_version,
            pretrain_directory=self.pretrain_directory
        )
        
        self.embedding_config = kwargs.get('embedding_config', {})
        self.embedding_config.setdefault('device', str(self.device))
    
    def _load_imputation_pipeline(self, **kwargs):
        """Load the imputation pipeline."""
        model_config = ImputationDefaultModelConfig.copy()
        model_config.update(kwargs.get('model_config', {}))
        
        self.imputation_pipeline = ImputationPipeline(
            pretrain_prefix=self.pretrain_version,
            overwrite_config=model_config,
            pretrain_directory=self.pretrain_directory
        )
        
        self.imputation_config = ImputationDefaultPipelineConfig.copy()
        self.imputation_config.update(kwargs.get('pipeline_config', {}))
        self.imputation_config['device'] = str(self.device)
    
    def preprocess(self, adata: AnnData, **kwargs) -> AnnData:
        """
        Preprocess data for CellPLM.
        
        Args:
            adata: Input AnnData object
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Preprocessed AnnData object
        """
        # CellPLM handles preprocessing internally in its pipelines
        # Just ensure basic data format requirements
        adata_processed = adata.copy()
        
        # Ensure X is sparse matrix
        if not issparse(adata_processed.X):
            from scipy.sparse import csr_matrix
            adata_processed.X = csr_matrix(adata_processed.X)
        
        return adata_processed
    
    def predict(self, adata: AnnData, task: str = "annotation", **kwargs) -> Dict[str, Any]:
        """
        Make predictions using CellPLM.
        
        Args:
            adata: Input AnnData object
            task: Task type ('annotation', 'embedding', 'imputation', 'integration')
            **kwargs: Additional prediction parameters
            
        Returns:
            Dictionary containing predictions and metadata
        """
        # For CellPLM, we check individual pipeline availability instead of global is_loaded
        self.current_task = task
        adata_processed = self.preprocess(adata, **kwargs)
        
        if task == "annotation":
            return self._predict_annotation(adata_processed, **kwargs)
        elif task == "embedding":
            return self._predict_embedding(adata_processed, **kwargs)
        elif task == "imputation":
            return self._predict_imputation(adata_processed, **kwargs)
        elif task == "integration":
            return self._predict_integration(adata_processed, **kwargs)
        else:
            raise ValueError(f"Unknown task: {task}. Available tasks: annotation, embedding, imputation, integration")
    
    def _predict_annotation(self, adata: AnnData, **kwargs) -> Dict[str, Any]:
        """Predict cell types using fine-tuned annotation model."""
        if self.annotation_pipeline is None:
            raise ValueError("Annotation pipeline not loaded")
        
        if 'annotation' not in self.fitted_tasks:
            raise ValueError("Model not fine-tuned for annotation. Call fine_tune() first.")
        
        try:
            SCLLMOutput.status(f"Predicting cell types for {adata.n_obs:,} cells...", 'predicting', indent=1)
            
            # Use CellPLM annotation pipeline for prediction with progress tracking
            with SCLLMOutput.progress_bar(total=2, desc="Cell type prediction", model_name="CellPLM") as pbar:
                pbar.set_description("[CellPLM] Running prediction...")
                predictions = self.annotation_pipeline.predict(
                    adata,
                    inference_config=kwargs.get('inference_config', {}),
                    ensembl_auto_conversion=kwargs.get('ensembl_auto_conversion', True),
                    **{k: v for k, v in kwargs.items() if k not in ['inference_config', 'ensembl_auto_conversion']}
                )
                pbar.update(1)
                pbar.set_description("[CellPLM] Processing results...")
                pbar.update(1)
            
            # Convert predictions to expected format
            if torch.is_tensor(predictions):
                predicted_ids = predictions.cpu().numpy()
            else:
                predicted_ids = np.array(predictions)
            
            results = {
                'predictions': predicted_ids,
                'task': 'annotation'
            }
            
            # Get cell type names using stored mapping or label encoders
            predicted_celltypes = None
            
            # First try to use our stored mapping
            if hasattr(self, 'id_to_celltype') and self.id_to_celltype:
                try:
                    predicted_celltypes = [self.id_to_celltype.get(int(pred_id), f'Unknown_{pred_id}') 
                                         for pred_id in predicted_ids]
                    predicted_celltypes = np.array(predicted_celltypes)
                except Exception as e:
                    SCLLMOutput.status(f"Could not use stored celltype mapping: {e}", "warning")
            
            # Fallback to label encoders if available
            if predicted_celltypes is None:
                if hasattr(self.annotation_pipeline, 'label_encoders') and self.annotation_pipeline.label_encoders:
                    try:
                        label_encoder = list(self.annotation_pipeline.label_encoders.values())[0]
                        predicted_celltypes = label_encoder.inverse_transform(predicted_ids)
                    except Exception as e:
                        SCLLMOutput.status(f"Could not use label encoder: {e}", "warning")
            
            if predicted_celltypes is not None:
                results['predicted_celltypes'] = predicted_celltypes
            
            # Add prediction confidence if available
            if hasattr(self.annotation_pipeline, 'model') and hasattr(self.annotation_pipeline.model, 'predict_proba'):
                try:
                    probabilities = self.annotation_pipeline.model.predict_proba(adata)
                    results['probabilities'] = probabilities
                except:
                    pass
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"CellPLM annotation prediction failed: {e}")
    
    def _predict_embedding(self, adata: AnnData, **kwargs) -> Dict[str, Any]:
        """Extract cell embeddings."""
        if self.embedding_pipeline is None:
            raise ValueError("Embedding pipeline not loaded")
        
        try:
            SCLLMOutput.status(f"Extracting embeddings for {adata.n_obs:,} cells...", 'embedding', indent=1)
            
            # Use CellPLM embedding pipeline with progress tracking
            # CellEmbeddingPipeline.predict() accepts: inference_config, covariate_fields,
            # batch_gene_list, ensembl_auto_conversion, device — NOT arbitrary kwargs
            batch_size = kwargs.pop('batch_size', None)
            inference_config = kwargs.pop('inference_config', None)
            if batch_size and not inference_config:
                inference_config = {'batch_size': batch_size}
            predict_kwargs = {}
            if inference_config:
                predict_kwargs['inference_config'] = inference_config
            for k in ('covariate_fields', 'batch_gene_list', 'ensembl_auto_conversion'):
                if k in kwargs:
                    predict_kwargs[k] = kwargs.pop(k)
            with SCLLMOutput.progress_bar(total=2, desc="Embedding extraction", model_name="CellPLM") as pbar:
                pbar.set_description("[CellPLM] Computing embeddings...")
                embeddings = self.embedding_pipeline.predict(
                    adata,
                    device=self.device,
                    **predict_kwargs
                )
                pbar.update(1)
                pbar.set_description("[CellPLM] Processing embeddings...")
                pbar.update(1)
            
            # Convert to numpy if tensor
            if torch.is_tensor(embeddings):
                embeddings = embeddings.cpu().numpy()
            
            return {
                'embeddings': embeddings,
                'task': 'embedding'
            }
            
        except Exception as e:
            raise RuntimeError(f"Embedding prediction failed: {e}")
    
    def _predict_imputation(self, adata: AnnData, **kwargs) -> Dict[str, Any]:
        """Perform imputation."""
        if self.imputation_pipeline is None:
            raise ValueError("Imputation pipeline not loaded")
        
        if 'imputation' not in self.fitted_tasks:
            raise ValueError("Model not fine-tuned for imputation. Call fine_tune() first.")
        
        try:
            SCLLMOutput.status(f" Performing imputation for {adata.n_obs} cells...", "predicting")
            
            # Use CellPLM imputation pipeline with progress tracking
            with tqdm(total=2, desc="Gene imputation", ncols=100) as pbar:
                pbar.set_description("Running imputation...")
                imputed_values = self.imputation_pipeline.predict(
                    adata,
                    self.imputation_config,
                    device=self.device,
                    **kwargs
                )
                pbar.update(1)
                pbar.set_description("Processing results...")
                pbar.update(1)
            
            # Convert to numpy if tensor
            if torch.is_tensor(imputed_values):
                imputed_values = imputed_values.cpu().numpy()
            
            return {
                'imputed_values': imputed_values,
                'task': 'imputation'
            }
            
        except Exception as e:
            raise RuntimeError(f"Imputation prediction failed: {e}")
    
    def _predict_integration(self, adata: AnnData, **kwargs) -> Dict[str, Any]:
        """Perform batch integration using embeddings."""
        SCLLMOutput.status(f"🔗 Performing batch integration for {adata.n_obs} cells...")
        
        # Extract integration-specific parameters and pass only embedding-related ones
        batch_key = kwargs.get('batch_key', 'batch')
        correction_method = kwargs.get('correction_method', 'harmony')
        
        # Filter kwargs to only pass embedding-related parameters
        embedding_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['batch_key', 'correction_method']}
        
        # For integration, we use embeddings and apply correction methods  
        embedding_results = self._predict_embedding(adata, **embedding_kwargs)
        embeddings = embedding_results['embeddings']
        
        if batch_key not in adata.obs:
            raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")
        
        try:
            # Apply batch correction to embeddings with progress tracking
            with tqdm(total=2, desc="Batch integration", ncols=100) as pbar:
                pbar.set_description(f"Applying {correction_method} correction...")
                integrated_embeddings = self._apply_batch_correction(
                    embeddings, adata.obs[batch_key], correction_method
                )
                pbar.update(1)
                pbar.set_description("Finalizing integration...")
                pbar.update(1)
            
            return {
                'embeddings': integrated_embeddings,
                'original_embeddings': embeddings,
                'correction_method': correction_method,
                'task': 'integration'
            }
            
        except Exception as e:
            raise RuntimeError(f"Integration failed: {e}")
    
    def _apply_batch_correction(self, embeddings: np.ndarray, batch_labels: pd.Series, method: str) -> np.ndarray:
        """Apply batch correction to embeddings."""
        if method == 'harmony':
            try:
                import harmonypy as hm
                SCLLMOutput.status(f" Applying Harmony correction...", indent=1)
                harmony_out = hm.run_harmony(embeddings.T, batch_labels, max_iter_harmony=20)  
                return harmony_out.Z_corr.T
            except ImportError:
                warnings.warn("harmonypy not available, using MNN correction")
                method = 'mnn'
        
        elif method == 'mnn':
            return self._apply_mnn_correction(embeddings, batch_labels)
        
        elif method == 'center_scale':
            return self._apply_center_scale_correction(embeddings, batch_labels)
        
        elif method == 'none':
            return embeddings
        
        else:
            warnings.warn(f"Unknown correction method {method}, using MNN correction")
            return self._apply_mnn_correction(embeddings, batch_labels)
    
    def _apply_mnn_correction(self, embeddings: np.ndarray, batch_labels: pd.Series) -> np.ndarray:
        """Apply MNN (Mutual Nearest Neighbors) correction - same as scFoundation implementation."""
        SCLLMOutput.status(f" Applying MNN correction...", indent=1)
        
        try:
            from sklearn.neighbors import NearestNeighbors
            
            corrected = embeddings.copy()
            unique_batches = batch_labels.unique()
            batch_codes = batch_labels.astype('category').cat.codes.values
            
            if len(unique_batches) < 2:
                SCLLMOutput.status(f"   Only one batch found, no correction needed", indent=1)
                return embeddings
            
            # Simple MNN-style correction between consecutive batches (same as scFoundation)
            for i in range(len(unique_batches) - 1):
                batch1_name = unique_batches[i]
                batch2_name = unique_batches[i + 1] if i + 1 < len(unique_batches) else unique_batches[0]
                
                batch1_mask = batch_labels == batch1_name
                batch2_mask = batch_labels == batch2_name
                
                batch1_data = corrected[batch1_mask]
                batch2_data = corrected[batch2_mask]
                
                if batch1_data.shape[0] > 5 and batch2_data.shape[0] > 5:
                    # Find mutual nearest neighbors (same algorithm as scFoundation)
                    k = min(5, min(batch1_data.shape[0], batch2_data.shape[0]) // 2)
                    
                    # Find nearest neighbors from batch2 to batch1
                    nn1 = NearestNeighbors(n_neighbors=k).fit(batch1_data)
                    distances1, indices1 = nn1.kneighbors(batch2_data)
                    
                    # Find nearest neighbors from batch1 to batch2  
                    nn2 = NearestNeighbors(n_neighbors=k).fit(batch2_data)
                    distances2, indices2 = nn2.kneighbors(batch1_data)
                    
                    # Apply simple correction by moving batches closer (same as scFoundation)
                    batch1_centroid = batch1_data.mean(axis=0)
                    batch2_centroid = batch2_data.mean(axis=0)
                    correction_vector = (batch1_centroid - batch2_centroid) * 0.5
                    
                    corrected[batch2_mask] += correction_vector
            
            SCLLMOutput.status(f"   MNN correction applied to {len(unique_batches)} batches", indent=1)
            return corrected
            
        except Exception as e:
            SCLLMOutput.status(f"   MNN correction failed: {e}, using center_scale correction", indent=1)
            return self._apply_center_scale_correction(embeddings, batch_labels)
    
    def _apply_center_scale_correction(self, embeddings: np.ndarray, batch_labels: pd.Series) -> np.ndarray:
        """Apply center and scale batch correction - same as scFoundation implementation."""
        SCLLMOutput.status(f" Applying center and scale correction...", indent=1)
        
        try:
            corrected = embeddings.copy()
            
            # Calculate global statistics (same as scFoundation)
            global_mean = corrected.mean(axis=0)
            global_std = corrected.std(axis=0) + 1e-8
            
            # Correct each batch (same as scFoundation)
            for batch_name in batch_labels.unique():
                batch_mask = batch_labels == batch_name
                batch_data = corrected[batch_mask]
                
                if batch_data.shape[0] > 1:
                    # Calculate batch statistics
                    batch_mean = batch_data.mean(axis=0)
                    batch_std = batch_data.std(axis=0) + 1e-8
                    
                    # Center and scale to global statistics
                    corrected[batch_mask] = (batch_data - batch_mean) / batch_std * global_std + global_mean
            
            SCLLMOutput.status(f"   Center-scale correction applied to {len(batch_labels.unique())} batches", indent=1)
            return corrected
            
        except Exception as e:
            SCLLMOutput.status(f"   Center-scale correction failed: {e}, returning original embeddings", indent=1)
            return embeddings
    
    def fine_tune(self, 
                  train_adata: AnnData,
                  valid_adata: Optional[AnnData] = None,
                  task: str = "annotation",
                  **kwargs) -> Dict[str, Any]:
        """
        Fine-tune CellPLM on new data.
        
        Args:
            train_adata: Training data
            valid_adata: Validation data (optional)
            task: Task type ('annotation', 'imputation')
            **kwargs: Training parameters
            
        Returns:
            Training results and metrics
        """
        # For CellPLM, we automatically load task-specific components during fine-tuning
        # This is different from scGPT/scFoundation which require pre-loading
        self.current_task = task
        
        if task == "annotation":
            return self._fine_tune_annotation(train_adata, valid_adata, **kwargs)
        elif task == "imputation":
            return self._fine_tune_imputation(train_adata, valid_adata, **kwargs)
        else:
            raise ValueError(f"Fine-tuning not supported for task: {task}")
    
    def _fine_tune_annotation(self, train_adata: AnnData, valid_adata: Optional[AnnData] = None, **kwargs) -> Dict[str, Any]:
        """Fine-tune for cell type annotation with enhanced parameter control."""
        if self.annotation_pipeline is None:
            self._load_annotation_pipeline(**kwargs)
        
        # Validate input data
        if 'celltype' not in train_adata.obs:
            raise ValueError("train_adata must have 'celltype' column in .obs")
        
        SCLLMOutput.status(f" Starting CellPLM fine-tuning for annotation task...", "training")
        
        # Get training parameters with defaults matching other models
        epochs = kwargs.get('epochs', 100)  # CellPLM default is higher
        batch_size = kwargs.get('batch_size', 32)
        lr = kwargs.get('lr', 5e-3)  # CellPLM default learning rate
        weight_decay = kwargs.get('weight_decay', kwargs.get('wd', 1e-7))
        patience = kwargs.get('patience', 25)
        validation_split = kwargs.get('validation_split', 0.2)
        
        SCLLMOutput.status(f"Training parameters: epochs={epochs}, batch_size={batch_size}, lr={lr}")
        
        # Prepare cell type mapping
        SCLLMOutput.status(f"📊 Preparing cell type mapping...")
        with tqdm(total=3, desc="Data preparation", ncols=100) as pbar:
            unique_celltypes = train_adata.obs['celltype'].astype('category').cat.categories
            pbar.update(1)
            
            celltype_to_id = {ct: i for i, ct in enumerate(unique_celltypes)}
            id_to_celltype = {i: ct for i, ct in enumerate(unique_celltypes)}
            pbar.update(1)
            
            n_celltypes = len(unique_celltypes)
            pbar.update(1)
        
        SCLLMOutput.status(f"Found {n_celltypes} cell types: {list(unique_celltypes)}", "info")
        
        # Prepare data for training
        SCLLMOutput.status(f" Preparing training and validation data...", "preprocessing")
        if valid_adata is not None:
            # Validate that validation data has celltype column
            if 'celltype' not in valid_adata.obs:
                raise ValueError("valid_adata must have 'celltype' column in .obs")
            
            # Combine train and validation data with split labels
            with tqdm(total=4, desc="Combining datasets", ncols=100) as pbar:
                train_adata_copy = train_adata.copy()
                pbar.update(1)
                valid_adata_copy = valid_adata.copy()
                pbar.update(1)
                train_adata_copy.obs['split'] = 'train'
                valid_adata_copy.obs['split'] = 'valid'
                pbar.update(1)
                combined_adata = train_adata_copy.concatenate(valid_adata_copy, batch_key=None, index_unique=None)
                pbar.update(1)
            SCLLMOutput.status(f"Using provided validation data: {train_adata.n_obs} train + {valid_adata.n_obs} valid cells", "info")
        else:
            # Use train/validation split
            with tqdm(total=1, desc="Copying dataset", ncols=100) as pbar:
                combined_adata = train_adata.copy()
                pbar.update(1)
            
            if validation_split > 0 and validation_split < 1:
                from sklearn.model_selection import train_test_split
                with tqdm(total=3, desc="Splitting data", ncols=100) as pbar:
                    try:
                        train_idx, val_idx = train_test_split(
                            range(combined_adata.n_obs),
                            test_size=validation_split,
                            stratify=combined_adata.obs['celltype'],
                            random_state=kwargs.get('random_state', 42)
                        )
                        pbar.update(1)
                        combined_adata.obs['split'] = 'train'
                        pbar.update(1)
                        combined_adata.obs.iloc[val_idx] = combined_adata.obs.iloc[val_idx].copy()
                        combined_adata.obs.iloc[val_idx, combined_adata.obs.columns.get_loc('split')] = 'valid'
                        pbar.update(1)
                        SCLLMOutput.status(f"Split data: {len(train_idx)} train, {len(val_idx)} validation")
                    except Exception as e:
                        SCLLMOutput.status(f"Could not stratify split due to {e}, using random split", "warning")
                        train_idx, val_idx = train_test_split(
                            range(combined_adata.n_obs),
                            test_size=validation_split,
                            random_state=kwargs.get('random_state', 42)
                        )
                        pbar.update(1)
                        combined_adata.obs['split'] = 'train'
                        pbar.update(1)
                        combined_adata.obs.iloc[val_idx] = combined_adata.obs.iloc[val_idx].copy()
                        combined_adata.obs.iloc[val_idx, combined_adata.obs.columns.get_loc('split')] = 'valid'
                        pbar.update(1)
                        SCLLMOutput.status(f"Split data (random): {len(train_idx)} train, {len(val_idx)} validation")
            else:
                combined_adata.obs['split'] = 'train'
                SCLLMOutput.status(f"Using all {combined_adata.n_obs} cells for training (no validation)", "info")
        
        # Update model configuration based on number of cell types
        model_config = CellTypeAnnotationDefaultModelConfig.copy()
        model_config['out_dim'] = n_celltypes
        model_config.update(kwargs.get('model_config', {}))
        
        # Reload pipeline with correct output dimension
        self.annotation_pipeline = CellTypeAnnotationPipeline(
            pretrain_prefix=self.pretrain_version,
            overwrite_config=model_config,
            pretrain_directory=self.pretrain_directory
        )
        
        # Set up training configuration
        train_config = CellTypeAnnotationDefaultPipelineConfig.copy()
        train_config.update({
            'epochs': epochs,
            'lr': lr,
            'wd': weight_decay,
            'patience': patience,
            'device': str(self.device),
            'max_eval_batch_size': kwargs.get('max_eval_batch_size', 100000),
            'hvg': kwargs.get('hvg', 3000),
            'scheduler': kwargs.get('scheduler', 'plat'),
        })
        train_config.update(kwargs.get('train_config', {}))
        
        # Store celltype mapping for prediction
        self.celltype_to_id = celltype_to_id
        self.id_to_celltype = id_to_celltype
        
        try:
            # Set seed for reproducibility
            set_seed(kwargs.get('random_state', 42))
            
            # Check if we have validation data
            has_validation = 'valid' in combined_adata.obs['split'].values
            if not has_validation:
                SCLLMOutput.status(f" No validation data available, training without validation", "warning")
                train_config['es'] = 0  # Disable early stopping
            
            # Fine-tune the model with real-time progress tracking
            SCLLMOutput.status(f"🏋️ Starting training with CellPLM pipeline...")
            SCLLMOutput.status(f"📈 Training for {epochs} epochs with real-time metrics...")
            
            # Use custom training with progress bar and metrics display
            best_results = self._train_with_progress_bar(
                combined_adata, train_config, has_validation, epochs, kwargs
            )
            
            # Display final results summary
            if best_results and 'final_epoch' in best_results:
                final_epoch = best_results['final_epoch']
                SCLLMOutput.status(f"\n📊 Final Training Results (Epoch {final_epoch}):")
                if 'train_acc' in best_results:
                    SCLLMOutput.status(f" 🎯 Train ACC: {best_results['train_acc']:.4f}", indent=1)
                if 'valid_acc' in best_results:
                    SCLLMOutput.status(f" ✅ Valid ACC: {best_results['valid_acc']:.4f}", indent=1)
                if 'train_f1' in best_results:
                    SCLLMOutput.status(f" 📈 Train F1:  {best_results['train_f1']:.4f}", indent=1)
                if 'valid_f1' in best_results:
                    SCLLMOutput.status(f" 📈 Valid F1:  {best_results['valid_f1']:.4f}", indent=1)
            else:
                SCLLMOutput.status(f" ℹ️ Final metrics will be available after real training", indent=1)
            
            self.fitted_tasks.add('annotation')
            self.is_loaded = True  # Mark model as loaded after successful fine-tuning
            
            SCLLMOutput.status(f" CellPLM annotation fine-tuning completed successfully!", "loaded")
            
            return {
                'status': 'completed',
                'task': 'annotation',
                'n_celltypes': n_celltypes,
                'celltype_mapping': {
                    'celltype_to_id': celltype_to_id,
                    'id_to_celltype': id_to_celltype
                },
                'training_config': train_config,
                'model_config': model_config,
                'epochs_run': epochs,
                'has_validation': has_validation
            }
            
        except Exception as e:
            error_msg = str(e)
            if "expected a non-empty list of Tensors" in error_msg:
                raise RuntimeError(
                    f"CellPLM annotation fine-tuning failed: No validation data found for split 'valid'. "
                    f"This can happen when validation_split is too small or data is insufficient. "
                    f"Try: 1) Increase validation_split (currently {validation_split}), "
                    f"2) Provide explicit valid_adata, or 3) Set validation_split=0 to skip validation. "
                    f"Original error: {error_msg}"
                )
            else:
                raise RuntimeError(f"CellPLM annotation fine-tuning failed: {error_msg}")
    
    def _fine_tune_imputation(self, train_adata: AnnData, valid_adata: Optional[AnnData] = None, **kwargs) -> Dict[str, Any]:
        """Fine-tune for imputation with enhanced parameter control."""
        if self.imputation_pipeline is None:
            self._load_imputation_pipeline(**kwargs)
        
        SCLLMOutput.status(f" Starting CellPLM fine-tuning for imputation task...", "training")
        
        # Get training parameters with CellPLM defaults
        epochs = kwargs.get('epochs', 100)  # CellPLM default for imputation
        lr = kwargs.get('lr', 5e-4)  # Lower learning rate for imputation
        weight_decay = kwargs.get('weight_decay', kwargs.get('wd', 1e-6))
        patience = kwargs.get('patience', 5)  # Less patience for imputation
        validation_split = kwargs.get('validation_split', 0.2)
        
        SCLLMOutput.status(f"Training parameters: epochs={epochs}, lr={lr}, patience={patience}")
        
        # Prepare batch-gene mapping if provided
        batch_gene_list = kwargs.get('batch_gene_list', None)
        
        # Handle data splitting similar to annotation
        if valid_adata is not None:
            # Combine train and validation data with split labels
            train_adata_copy = train_adata.copy()
            valid_adata_copy = valid_adata.copy()
            train_adata_copy.obs['split'] = 'train'
            valid_adata_copy.obs['split'] = 'valid'
            combined_adata = train_adata_copy.concatenate(valid_adata_copy, batch_key=None, index_unique=None)
            SCLLMOutput.status(f"Using provided validation data: {train_adata.n_obs} train + {valid_adata.n_obs} valid cells", "info")
        else:
            # Use train/validation split if split column doesn't exist
            combined_adata = train_adata.copy()
            if 'split' not in combined_adata.obs and validation_split > 0:
                from sklearn.model_selection import train_test_split
                train_idx, val_idx = train_test_split(
                    range(combined_adata.n_obs),
                    test_size=validation_split,
                    random_state=kwargs.get('random_state', 42)
                )
                combined_adata.obs['split'] = 'train'
                combined_adata.obs.iloc[val_idx] = combined_adata.obs.iloc[val_idx].copy()
                combined_adata.obs.iloc[val_idx, combined_adata.obs.columns.get_loc('split')] = 'valid'
                SCLLMOutput.status(f"Split data: {len(train_idx)} train, {len(val_idx)} validation")
            elif 'split' not in combined_adata.obs:
                combined_adata.obs['split'] = 'train'
                SCLLMOutput.status(f"Using all {combined_adata.n_obs} cells for training (no validation)", "info")
        
        # Set up training configuration
        train_config = ImputationDefaultPipelineConfig.copy() 
        train_config.update({
            'epochs': epochs,
            'lr': lr,
            'wd': weight_decay,
            'patience': patience,
            'device': str(self.device),
            'max_eval_batch_size': kwargs.get('max_eval_batch_size', 100000),
            'scheduler': kwargs.get('scheduler', 'plat'),
        })
        train_config.update(kwargs.get('train_config', {}))
        
        try:
            # Set seed for reproducibility
            set_seed(kwargs.get('random_state', 42))
            
            # Check if we have validation data
            has_validation = 'valid' in combined_adata.obs['split'].values
            if not has_validation:
                SCLLMOutput.status(f" No validation data available, training without validation", "warning")
            
            # Fine-tune the model with real-time progress tracking
            SCLLMOutput.status(f"🏋️ Starting imputation training with CellPLM pipeline...")
            SCLLMOutput.status(f"📈 Training for {epochs} epochs with real-time metrics...")
            
            # Use custom training with progress bar for imputation
            best_results = self._train_imputation_with_progress_bar(
                combined_adata, train_config, has_validation, epochs, batch_gene_list, kwargs
            )
            
            # Display final results summary for imputation
            if best_results and 'final_epoch' in best_results:
                final_epoch = best_results['final_epoch']
                SCLLMOutput.status(f"\n📊 Final Imputation Results (Epoch {final_epoch}):")
                if 'train_loss' in best_results:
                    SCLLMOutput.status(f" 📉 Train Loss: {best_results['train_loss']:.4f}", indent=1)
                if 'valid_loss' in best_results:
                    SCLLMOutput.status(f" 📉 Valid Loss: {best_results['valid_loss']:.4f}", indent=1)
            else:
                SCLLMOutput.status(f" ℹ️ Final metrics will be available after real training", indent=1)
            
            self.fitted_tasks.add('imputation')
            self.is_loaded = True  # Mark model as loaded after successful fine-tuning
            
            SCLLMOutput.status(f" CellPLM imputation fine-tuning completed successfully!", "loaded")
            
            return {
                'status': 'completed',
                'task': 'imputation',
                'training_config': train_config,
                'epochs_run': epochs,
                'has_validation': has_validation,
                'batch_gene_list': batch_gene_list is not None
            }
            
        except Exception as e:
            error_msg = str(e)
            if "expected a non-empty list of Tensors" in error_msg:
                raise RuntimeError(
                    f"CellPLM imputation fine-tuning failed: No validation data found for split 'valid'. "
                    f"This can happen when validation_split is too small or data is insufficient. "
                    f"Try: 1) Increase validation_split (currently {validation_split}), "
                    f"2) Provide explicit valid_adata, or 3) Set validation_split=0 to skip validation. "
                    f"Original error: {error_msg}"
                )
            else:
                raise RuntimeError(f"CellPLM imputation fine-tuning failed: {error_msg}")
    
    def get_embeddings(self, adata: AnnData, **kwargs) -> np.ndarray:
        """
        Get cell embeddings from CellPLM.
        
        Args:
            adata: Input AnnData object
            **kwargs: Additional parameters
            
        Returns:
            Cell embeddings as numpy array
        """
        # Start embedding extraction with unified output
        SCLLMOutput.data_summary(adata, model_name="CellPLM")
        operation_start("get_embeddings", "CellPLM", {
            "cells": f"{adata.n_obs:,}",
            "genes": f"{adata.n_vars:,}"
        })
        
        results = self.predict(adata, task="embedding", **kwargs)
        
        operation_complete("get_embeddings", {
            "embedding_shape": f"{results['embeddings'].shape}",
            "embedding_dim": results['embeddings'].shape[1]
        })
        
        return results['embeddings']
    
    def predict_celltypes(self, query_adata: AnnData, **kwargs) -> Dict[str, Any]:
        """
        Predict cell types for query data.
        
        Args:
            query_adata: Query data to predict
            **kwargs: Additional parameters
            
        Returns:
            Prediction results with cell type names and statistics
        """
        operation_start("predict_celltypes", "CellPLM", {
            "cells": f"{query_adata.n_obs:,}",
            "genes": f"{query_adata.n_vars:,}"
        })
        
        results = self.predict(query_adata, task="annotation", **kwargs)
        
        if 'predicted_celltypes' in results:
            from collections import Counter
            type_counts = Counter(results['predicted_celltypes'])
            
            operation_complete("predict_celltypes", {
                "total_cells": len(results['predicted_celltypes']),
                "unique_types": len(type_counts),
                "most_common": type_counts.most_common(1)[0][0] if type_counts else "None"
            })
        else:
            operation_complete("predict_celltypes", {"status": "completed"})
        
        return results
    
    def integrate(self, adata: AnnData, batch_key: str = "batch", **kwargs) -> Dict[str, Any]:
        """
        Perform batch integration.
        
        Args:
            adata: AnnData object with batch information
            batch_key: Column name for batch labels
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with integration results
        """
        kwargs['batch_key'] = batch_key
        return self.predict(adata, task="integration", **kwargs)
    
    def train_integration(self, train_adata: AnnData, valid_adata: Optional[AnnData] = None, 
                         batch_key: str = "batch", **kwargs) -> Dict[str, Any]:
        """
        Train for batch integration (using embedding approach).
        
        Args:
            train_adata: Training data with batch labels
            valid_adata: Validation data (optional)
            batch_key: Column name for batch labels
            **kwargs: Training parameters
            
        Returns:
            Training results
        """
        # For CellPLM, integration is performed via embeddings + correction
        # No specific training needed, just ensure embedding pipeline is loaded
        if self.embedding_pipeline is None:
            self._load_embedding_pipeline(**kwargs)
        
        return {
            'status': 'completed',
            'task': 'integration',
            'method': 'embedding_based',
            'batch_key': batch_key
        }
    
    def score(self, adata: AnnData, task: str = "annotation", **kwargs) -> Dict[str, Any]:
        """
        Score the model performance.
        
        Args:
            adata: AnnData object with ground truth labels
            task: Task type to evaluate
            **kwargs: Additional parameters
            
        Returns:
            Performance metrics
        """
        if task == "annotation":
            return self._score_annotation(adata, **kwargs)
        elif task == "embedding":
            return self._score_embedding(adata, **kwargs)
        elif task == "imputation":
            return self._score_imputation(adata, **kwargs)
        else:
            raise ValueError(f"Scoring not implemented for task: {task}")
    
    def load_celltype_mapping(self, model_path: Union[str, Path]) -> None:
        """
        Load celltype mapping from saved model directory.
        
        Args:
            model_path: Path to the saved model directory
        """
        try:
            import json
            config_path = Path(model_path) / 'cellplm_config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'celltype_mapping' in config:
                        self.celltype_to_id = config['celltype_mapping'].get('celltype_to_id', {})
                        self.id_to_celltype = {int(k): v for k, v in config['celltype_mapping'].get('id_to_celltype', {}).items()}
                        SCLLMOutput.status(f" Loaded celltype mapping with {len(self.id_to_celltype)} cell types", "loaded")
        except Exception as e:
            SCLLMOutput.status(f"Could not load celltype mapping: {e}", "warning")
    
    def predict_with_finetune(self, adata: AnnData, **kwargs) -> Dict[str, Any]:
        """
        Predict using fine-tuned model (alias for predict with annotation task).
        
        Args:
            adata: Input data
            **kwargs: Additional parameters
            
        Returns:
            Prediction results
        """
        return self.predict(adata, task="annotation", **kwargs)
    
    def _score_annotation(self, adata: AnnData, **kwargs) -> Dict[str, Any]:
        """Score annotation performance."""
        if self.annotation_pipeline is None:
            raise ValueError("Annotation pipeline not loaded")
        
        try:
            results = self.annotation_pipeline.score(
                adata,
                pipeline_config=self.annotation_config,
                split_field=kwargs.get('split_field', None),
                target_split=kwargs.get('target_split', 'test'),
                label_fields=['celltype'],
                **kwargs
            )
            return results
        except Exception as e:
            raise RuntimeError(f"Annotation scoring failed: {e}")
    
    def _score_embedding(self, adata: AnnData, **kwargs) -> Dict[str, Any]:
        """Score embedding quality."""
        if self.embedding_pipeline is None:
            raise ValueError("Embedding pipeline not loaded")
        
        try:
            results = self.embedding_pipeline.score(
                adata,
                label_fields=kwargs.get('label_fields', ['celltype']),
                evaluation_config=kwargs.get('evaluation_config', {}),
                device=self.device,
                **kwargs
            )
            return results
        except Exception as e:
            raise RuntimeError(f"Embedding scoring failed: {e}")
    
    def _score_imputation(self, adata: AnnData, **kwargs) -> Dict[str, Any]:
        """Score imputation performance."""
        if self.imputation_pipeline is None:
            raise ValueError("Imputation pipeline not loaded")
        
        try:
            results = self.imputation_pipeline.score(
                adata,
                evaluation_config=kwargs.get('evaluation_config', {}),
                label_fields=kwargs.get('label_fields', ['truth']),
                device=self.device,
                **kwargs
            )
            return results
        except Exception as e:
            raise RuntimeError(f"Imputation scoring failed: {e}")
    
    def _save_model_specific(self, save_path: Path, **kwargs) -> None:
        """
        Save CellPLM model components.
        
        Args:
            save_path: Path to save the model
            **kwargs: Additional save parameters
        """
        # Save model configuration and state
        config = {
            'model_name': self.model_name,
            'pretrain_version': self.pretrain_version,
            'pretrain_directory': self.pretrain_directory,
            'fitted_tasks': list(self.fitted_tasks),
            'device': str(self.device)
        }
        
        # Save celltype mapping if available
        if hasattr(self, 'celltype_to_id') and hasattr(self, 'id_to_celltype'):
            config['celltype_mapping'] = {
                'celltype_to_id': self.celltype_to_id,
                'id_to_celltype': {str(k): v for k, v in self.id_to_celltype.items()}
            }
        
        config_path = save_path / 'cellplm_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save task-specific configurations
        if self.annotation_config:
            with open(save_path / 'annotation_config.json', 'w') as f:
                json.dump(self.annotation_config, f, indent=2)
        
        if self.embedding_config:
            with open(save_path / 'embedding_config.json', 'w') as f:
                json.dump(self.embedding_config, f, indent=2)
        
        if self.imputation_config:
            with open(save_path / 'imputation_config.json', 'w') as f:
                json.dump(self.imputation_config, f, indent=2)
        
        # Save pipeline states if they have been fitted
        # Note: CellPLM pipelines handle their own model saving
        SCLLMOutput.status(f"CellPLM model configuration saved to {save_path}")
        if hasattr(self, 'celltype_to_id'):
            SCLLMOutput.status(f"- Celltype mapping: {len(self.celltype_to_id)} cell types", indent=1)
        SCLLMOutput.status(f"- Fitted tasks: {list(self.fitted_tasks)}", indent=1)
    
    def _train_with_progress_bar(self, combined_adata, train_config, has_validation, epochs, kwargs):
        """
        Train CellPLM with real-time progress bar showing metrics.
        """
        import contextlib
        import sys
        import threading
        import time
        from io import StringIO
        
        # Store best results
        best_results = {
            'train_acc': 0.0,
            'valid_acc': 0.0,
            'train_f1': 0.0,
            'valid_f1': 0.0,
            'final_epoch': 0
        }
        
        # Create shared state for progress tracking
        progress_state = {
            'current_epoch': 0,
            'train_acc': 0.0,
            'valid_acc': 0.0,
            'train_loss': 0.0,
            'valid_loss': 0.0,
            'training_active': True,
            'progress_bar': None
        }
        
        def parse_training_output(output_buffer):
            """Parse CellPLM training output for metrics."""
            lines = output_buffer.getvalue().split('\n')
            
            # CellPLM outputs metrics in two separate lines:
            # Line 1: "Epoch {epoch} | Train loss: {loss} | Valid loss: {loss}"
            # Line 2: "Train ACC: {acc} | Valid ACC: {acc} | Train f1: {f1} | ..."
            
            current_epoch = -1
            current_losses = {}
            
            for line in reversed(lines[-100:]):  # Check more recent lines
                line = line.strip()
                if not line:
                    continue
                    
                # Parse epoch and loss line
                if line.startswith('Epoch') and 'Train loss:' in line and 'Valid loss:' in line:
                    try:
                        # Extract epoch number
                        epoch_part = line.split('|')[0].strip()
                        epoch_num = int(epoch_part.split('Epoch')[1].strip())
                        current_epoch = epoch_num
                        
                        # Extract losses
                        parts = line.split('|')
                        for part in parts:
                            part = part.strip()
                            if 'Train loss:' in part:
                                current_losses['train_loss'] = float(part.split(':')[1].strip())
                            elif 'Valid loss:' in part:  
                                current_losses['valid_loss'] = float(part.split(':')[1].strip())
                                
                    except (ValueError, IndexError):
                        continue
                
                # Parse accuracy line (should be right after epoch line)
                elif 'Train ACC:' in line and 'Valid ACC:' in line and current_epoch >= 0:
                    try:
                        parts = line.split('|')
                        for part in parts:
                            part = part.strip() 
                            if 'Train ACC:' in part:
                                progress_state['train_acc'] = float(part.split(':')[1].strip())
                                best_results['train_acc'] = progress_state['train_acc']
                            elif 'Valid ACC:' in part:
                                progress_state['valid_acc'] = float(part.split(':')[1].strip())
                                best_results['valid_acc'] = progress_state['valid_acc']
                            elif 'Train f1:' in part:
                                best_results['train_f1'] = float(part.split(':')[1].strip())
                            elif 'Valid f1:' in part:
                                best_results['valid_f1'] = float(part.split(':')[1].strip())
                        
                        # Update progress state with current epoch and losses
                        progress_state['current_epoch'] = current_epoch
                        if 'train_loss' in current_losses:
                            progress_state['train_loss'] = current_losses['train_loss']
                        if 'valid_loss' in current_losses:
                            progress_state['valid_loss'] = current_losses['valid_loss']
                            
                        best_results['final_epoch'] = current_epoch
                        break  # Found the most recent complete metrics
                        
                    except (ValueError, IndexError):
                        continue
        
        def update_progress_bar():
            """Background thread to update progress bar with metrics."""
            # Don't create our own progress bar since CellPLM has its own
            # Instead, we'll print periodic updates with metrics
            last_epoch = -1
            last_update_time = time.time()
            
            while progress_state['training_active']:
                current_epoch = progress_state['current_epoch']
                current_time = time.time()
                
                # Update every 2 seconds or when epoch changes
                if (current_epoch > last_epoch and current_epoch > 0) or (current_time - last_update_time > 2.0):
                    if progress_state['train_acc'] > 0 or progress_state['valid_acc'] > 0:
                        # Print metrics update
                        metrics_parts = []
                        if current_epoch > 0:
                            metrics_parts.append(f"📊 Epoch {current_epoch}/{epochs}")
                        if progress_state['train_acc'] > 0:
                            metrics_parts.append(f"Train ACC: {progress_state['train_acc']:.3f}")
                        if progress_state['valid_acc'] > 0:
                            metrics_parts.append(f"Valid ACC: {progress_state['valid_acc']:.3f}")
                        if progress_state['train_loss'] > 0:
                            metrics_parts.append(f"Loss: {progress_state['train_loss']:.3f}")
                        
                        if metrics_parts:
                            SCLLMOutput.status(f" {' | '.join(metrics_parts)}", indent=1)
                            last_update_time = current_time
                    
                    last_epoch = current_epoch
                
                time.sleep(1.0)  # Check every second
        
        try:
            # Start progress bar in separate thread
            progress_thread = threading.Thread(target=update_progress_bar, daemon=True)
            progress_thread.start()
            
            # Capture training output and parse it for metrics
            captured_output = StringIO()
            
            # Custom output handler that parses metrics while training
            class MetricCapturingIO(StringIO):
                def write(self, s):
                    result = super().write(s)
                    # Parse on any meaningful output (not just epoch lines)
                    if s.strip() and (
                        'Epoch' in s or 'ACC:' in s or 'Train loss:' in s or 
                        'Valid loss:' in s or 'f1:' in s
                    ):
                        parse_training_output(self)
                    return result
            
            metric_output = MetricCapturingIO()
            
            with contextlib.redirect_stdout(metric_output):
                self.annotation_pipeline.fit(
                    combined_adata,
                    train_config=train_config,
                    split_field='split',
                    train_split='train',
                    valid_split='valid' if has_validation else None,
                    label_fields=['celltype'],
                    device=self.device,
                    ensembl_auto_conversion=kwargs.get('ensembl_auto_conversion', True),
                    **{k: v for k, v in kwargs.items() if k not in ['epochs', 'batch_size', 'lr', 'weight_decay', 'wd', 'patience', 'validation_split', 'random_state', 'train_config', 'model_config']}
                )
            
            # Training completed
            progress_state['training_active'] = False
            
            # Final parse of all output
            parse_training_output(metric_output)
            
            # Show preprocessing info
            output_lines = metric_output.getvalue().split('\n')
            for line in output_lines:
                if line and any(x in line.lower() for x in ['after filtering', 'genes remain', 'early stopped']):
                    SCLLMOutput.status(str(line))
            
            # Wait for progress thread to finish
            if progress_thread.is_alive():
                progress_thread.join(timeout=1.0)
            
            return best_results
            
        except Exception as e:
            progress_state['training_active'] = False
            raise e
    
    def _train_imputation_with_progress_bar(self, combined_adata, train_config, has_validation, epochs, batch_gene_list, kwargs):
        """
        Train CellPLM imputation with real-time progress bar showing metrics.
        """
        import contextlib
        import sys
        import threading
        import time
        from io import StringIO
        
        # Store best results (imputation uses different metrics)
        best_results = {
            'train_loss': float('inf'),
            'valid_loss': float('inf'),
            'final_epoch': 0
        }
        
        # Create shared state for progress tracking
        progress_state = {
            'current_epoch': 0,
            'train_loss': 0.0,
            'valid_loss': 0.0,
            'training_active': True,
            'progress_bar': None
        }
        
        def parse_imputation_output(output_buffer):
            """Parse CellPLM imputation training output for metrics."""
            lines = output_buffer.getvalue().split('\n')
            for line in reversed(lines[-50:]):  # Check recent lines
                if 'Epoch' in line and 'Train loss:' in line:
                    try:
                        # Extract epoch number
                        epoch_match = line.split('Epoch')[1].split('|')[0].strip()
                        progress_state['current_epoch'] = int(epoch_match)
                        
                        # Extract metrics
                        parts = line.split('|')
                        for part in parts:
                            part = part.strip()
                            if 'Train loss:' in part:
                                progress_state['train_loss'] = float(part.split(':')[1].strip())
                                best_results['train_loss'] = progress_state['train_loss']
                            elif 'Valid loss:' in part:
                                progress_state['valid_loss'] = float(part.split(':')[1].strip())
                                best_results['valid_loss'] = progress_state['valid_loss']
                        
                        best_results['final_epoch'] = progress_state['current_epoch']
                        
                    except (ValueError, IndexError):
                        pass
        
        def update_imputation_progress_bar():
            """Background thread to update progress with imputation metrics."""
            # Print periodic updates instead of creating competing progress bars
            last_epoch = -1
            last_update_time = time.time()
            
            while progress_state['training_active']:
                current_epoch = progress_state['current_epoch']
                current_time = time.time()
                
                # Update every 2 seconds or when epoch changes
                if (current_epoch > last_epoch and current_epoch > 0) or (current_time - last_update_time > 2.0):
                    if progress_state['train_loss'] > 0 or progress_state['valid_loss'] > 0:
                        # Print metrics update
                        metrics_parts = []
                        if current_epoch > 0:
                            metrics_parts.append(f"📊 Epoch {current_epoch}/{epochs}")
                        if progress_state['train_loss'] > 0:
                            metrics_parts.append(f"Train Loss: {progress_state['train_loss']:.4f}")
                        if progress_state['valid_loss'] > 0:
                            metrics_parts.append(f"Valid Loss: {progress_state['valid_loss']:.4f}")
                        
                        if metrics_parts:
                            SCLLMOutput.status(f" {' | '.join(metrics_parts)}", indent=1)
                            last_update_time = current_time
                    
                    last_epoch = current_epoch
                
                time.sleep(1.0)  # Check every second
        
        try:
            # Start progress bar in separate thread
            progress_thread = threading.Thread(target=update_imputation_progress_bar, daemon=True)
            progress_thread.start()
            
            # Custom output handler that parses metrics while training
            class MetricCapturingIO(StringIO):
                def write(self, s):
                    super().write(s)
                    if s.strip() and ('Epoch' in s or 'loss:' in s):
                        parse_imputation_output(self)
                    return len(s)
            
            metric_output = MetricCapturingIO()
            
            with contextlib.redirect_stdout(metric_output):
                self.imputation_pipeline.fit(
                    combined_adata,
                    train_config=train_config,
                    split_field='split',
                    train_split='train',
                    valid_split='valid' if has_validation else None,
                    batch_gene_list=batch_gene_list,
                    device=self.device,
                    ensembl_auto_conversion=kwargs.get('ensembl_auto_conversion', True),
                    **{k: v for k, v in kwargs.items() if k not in ['epochs', 'lr', 'weight_decay', 'wd', 'patience', 'validation_split', 'random_state', 'train_config', 'batch_gene_list']}
                )
            
            # Training completed
            progress_state['training_active'] = False
            
            # Final parse of all output
            parse_imputation_output(metric_output)
            
            # Show preprocessing info
            output_lines = metric_output.getvalue().split('\n')
            for line in output_lines:
                if line and any(x in line.lower() for x in ['after filtering', 'genes remain', 'early stopped']):
                    SCLLMOutput.status(str(line))
            
            # Wait for progress thread to finish
            if progress_thread.is_alive():
                progress_thread.join(timeout=1.0)
            
            return best_results
            
        except Exception as e:
            progress_state['training_active'] = False
            raise e
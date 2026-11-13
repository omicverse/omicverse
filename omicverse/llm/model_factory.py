"""
Model factory for creating and managing different scLLM models.
"""

from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import numpy as np

from .._registry import register_function
from .base import SCLLMBase



class ModelFactory:
    """
    Factory class for creating different single-cell language models.
    
    This provides a unified interface to instantiate and manage different models
    like scGPT, scBERT, etc.
    """
    
    # Registry of available models
    from .scgpt_model import ScGPTModel
    from .scfoundation_model import ScFoundationModel
    from .geneformer_model import GeneformerModel
    from .cellplm_model import CellPLMModel
    from .uce_model import UCEModel
    from .scmulan_model import ScMulanModel
    from .external_model_stubs import (
        ScBERTModel,
        GeneCompassModel,
        NicheformerModel,
        TGPTModel,
        CellFMModel,
        ScCelloModel,
        ScPrintModel,
        AiDocellModel,
        PulsarModel,
        AtacformerModel,
        ScPlantLLMModel,
        LangCellModel,
        Cell2SentenceModel,
        GenePTModel,
        ChatCellModel,
        TabulaModel,
    )

    _models = {
        "scgpt": ScGPTModel,
        "scfoundation": ScFoundationModel,
        "geneformer": GeneformerModel,
        "cellplm": CellPLMModel,
        "uce": UCEModel,
        "scmulan": ScMulanModel,
        "scbert": ScBERTModel,
        "genecompass": GeneCompassModel,
        "nicheformer": NicheformerModel,
        "tgpt": TGPTModel,
        "cellfm": CellFMModel,
        "sccello": ScCelloModel,
        "scprint": ScPrintModel,
        "aidocell": AiDocellModel,
        "pulsar": PulsarModel,
        "atacformer": AtacformerModel,
        "scplantllm": ScPlantLLMModel,
        "langcell": LangCellModel,
        "cell2sentence": Cell2SentenceModel,
        "genept": GenePTModel,
        "chatcell": ChatCellModel,
        "tabula": TabulaModel,
    }
    
    
    @classmethod
    def create_model(cls, 
                     model_type: str, 
                     model_path: Optional[Union[str, Path]] = None,
                     device: Optional[str] = None,
                     **kwargs):
        """
        Create a model instance.
        
        Args:
            model_type: Type of model ('scgpt', 'scbert', etc.)
            model_path: Path to pre-trained model (optional)
            device: Device to run the model on
            **kwargs: Additional model-specific parameters
            
        Returns:
            Model instance
        """
        if model_type.lower() == "geneformer":
            try:
                from tdigest import TDigest
                
            except ImportError:
                import warnings
                warnings.warn("tdigest is recommended for full Geneformer functionality. Install with `pip install tdigest`.")
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError:
                import warnings
                warnings.warn("peft is recommended for full Geneformer functionality. Install with `pip install peft`.")
            
        elif model_type.lower() == "scfoundation":
            try:
                from datasets import Dataset
            except ImportError:
                raise ImportError("datasets is required for scFoundation model. Please install it using `pip install datasets`.")
        
        elif model_type.lower() == "cellplm":
            try:
                import torch
            except ImportError:
                import warnings
                warnings.warn("torch is recommended for full CellPLM functionality.")
        
        elif model_type.lower() == "uce":
            try:
                import torch
            except ImportError:
                import warnings
                warnings.warn("torch is recommended for full UCE functionality.")
            try:
                from accelerate import Accelerator
            except ImportError:
                import warnings
                warnings.warn("accelerate is recommended for UCE functionality. Install with `pip install accelerate`.")
        elif model_type.lower() == "scmulan":
            try:
                import torch
                from scipy.sparse import csc_matrix  # noqa: F401
            except ImportError:
                import warnings
                warnings.warn("torch and scipy are recommended for scMulan functionality.")
            
        # Extract UCE-specific asset paths from kwargs
        uce_assets = {}
        if model_type.lower() == "uce":
            uce_asset_keys = ['token_file', 'protein_embeddings_dir', 'spec_chrom_csv_path', 'offset_pkl_path']
            uce_assets = {key: kwargs.pop(key, None) for key in uce_asset_keys if key in kwargs}
            
        if model_type.lower() not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available models: {available_models}")
        
        model_class = cls._models[model_type.lower()]
        
        # Handle UCE-specific asset paths
        if model_type.lower() == "uce":
            model = model_class(device=device, **uce_assets, **kwargs)
        else:
            model = model_class(device=device, **kwargs)
        
        # Load pre-trained model if path provided
        if model_path is not None:
            if model_type.lower() == "uce":
                # For UCE, pass the asset paths back to load_model
                model.load_model(model_path, **uce_assets, **kwargs)
            else:
                model.load_model(model_path, **kwargs)
        
        return model
    
    @classmethod
    def available_models(cls) -> list:
        """Get list of available model types."""
        return list(cls._models.keys())
    
    @classmethod
    def register_model(cls, model_type: str, model_class: type) -> None:
        """
        Register a new model type.
        
        Args:
            model_type: Name of the model type
            model_class: Model class that inherits from SCLLMBase
        """
        if not issubclass(model_class, SCLLMBase):
            raise ValueError("Model class must inherit from SCLLMBase")
        
        cls._models[model_type.lower()] = model_class


@register_function(
    aliases=[
        # Direct manager / class names
        "SCLLMManager", "scllm manager",
        # Foundation-model / single-cell language model identifiers — narrow
        # so that generic queries like "cell type annotation" or "batch
        # integration" go to the canonical ov.pp / ov.single.Annotation /
        # ov.pp.batch_correction wrappers, and only foundation-model-explicit
        # queries land here.
        "single-cell foundation model", "single cell foundation model",
        "single-cell language model", "scLLM", "scllm",
        "foundation model annotate", "foundation model annotation",
        "foundation model embedding", "foundation model integration",
        "foundation model perturbation", "foundation model fine-tune",
        "zero-shot foundation model",
        "scgpt", "geneformer", "scfoundation", "uce", "cellplm", "tabula",
        "scbert", "scmulan", "tgpt", "cellfm", "sccello",
        "scgpt annotate", "geneformer annotate", "cellplm annotate",
        "scgpt embedding", "geneformer embedding", "cellplm embedding",
        # Chinese
        "单细胞基础模型", "单细胞大模型", "scGPT 注释", "Geneformer 注释",
    ],
    category="llm",
    description=(
        "High-level manager for single-cell foundation models (RECOMMENDED entry "
        "point for scGPT, Geneformer, scFoundation, UCE, CellPLM, scBERT, scMulan, "
        "tGPT, CellFM, scCello, Tabula, etc.). Supports cell-type annotation, "
        "embedding extraction, batch integration, fine-tuning, and perturbation "
        "prediction through a single object. Each model_type may not implement "
        "every task — check the model docstring for capabilities."
    ),
    requires={"obs": ["gene-expression matrix"], "var": ["gene identifiers"]},
    produces={"obsm": ["X_<model>"], "obs": ["predicted_celltype (after annotate_cells)"]},
    auto_fix="none",
    examples=[
        "manager = ov.llm.SCLLMManager(model_type='scgpt', model_path='/path/to/scgpt')",
        "embeddings = manager.get_embeddings(adata)",
        "result = manager.annotate_cells(adata)",
        "manager.integrate(adata, batch_key='batch')",
        "manager.fine_tune(train_adata, valid_adata, task='annotation')",
    ],
    related=[
        "llm.ModelFactory", "single.SCENIC",
        "single.Annotation",
    ],
)
class SCLLMManager:
    """
    High-level manager for single-cell language models.

    This provides the simplest interface for common operations.
    """

    def __init__(self,
                 model_type: str = "scgpt",
                 model_path: Optional[Union[str, Path]] = None,
                 device: Optional[str] = None,
                 **kwargs):
        """
        Initialize the model manager.
        
        Args:
            model_type: Type of model to use
            model_path: Path to pre-trained model
            device: Device to run the model on
            **kwargs: Additional parameters
        """
        self.model = ModelFactory.create_model(
            model_type=model_type,
            model_path=model_path,
            device=device,
            **kwargs
        )
    
    def annotate_cells(self, adata, **kwargs):
        """
        Annotate cell types.
        
        Args:
            adata: AnnData object with gene expression data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with predictions and metadata
        """
        return self.model.predict(adata, task="annotation", **kwargs)
    
    def get_embeddings(self, adata, **kwargs):
        """
        Get cell embeddings.
        
        Args:
            adata: AnnData object
            **kwargs: Additional parameters
            
        Returns:
            Cell embeddings as numpy array
        """
        return self.model.get_embeddings(adata, **kwargs)
    
    def fine_tune(self, train_adata, valid_adata=None, task="annotation", **kwargs):
        """
        Fine-tune the model.
        
        Args:
            train_adata: Training data
            valid_adata: Validation data
            task: Task type
            **kwargs: Training parameters
            
        Returns:
            Training results
        """
        return self.model.fine_tune(
            train_adata=train_adata,
            valid_adata=valid_adata,
            task=task,
            **kwargs
        )
    
    def train_integration(self, train_adata, valid_adata=None, batch_key="batch", **kwargs):
        """
        Train the model for batch integration.
        
        Args:
            train_adata: Training data with batch labels
            valid_adata: Validation data (optional)
            batch_key: Column name for batch labels
            **kwargs: Training parameters
            
        Returns:
            Training results
        """
        return self.model.train_integration(
            train_adata=train_adata,
            valid_adata=valid_adata,
            batch_key=batch_key,
            **kwargs
        )
    
    def integrate(self, adata, batch_key="batch", **kwargs):
        """
        Perform batch integration.
        
        Args:
            adata: AnnData object with batch information
            batch_key: Column name for batch labels
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with integration results
        """
        return self.model.integrate(adata, batch_key=batch_key, **kwargs)
    
    def predict_celltypes(self, query_adata, **kwargs):
        """
        Predict cell types for query data.
        
        Args:
            query_adata: Query data to predict
            **kwargs: Additional parameters
            
        Returns:
            Prediction results with cell type names and statistics
        """
        if hasattr(self.model, 'predict_celltypes'):
            return self.model.predict_celltypes(query_adata, **kwargs)
        else:
            # Fallback to general prediction
            return self.model.predict(query_adata, task="annotation", **kwargs)
    
    def perturb_genes(self, adata, target_genes, perturb_type="overexpress", **kwargs):
        """
        Perform in silico gene perturbation experiments.
        
        This method is particularly useful for Geneformer models.
        
        Args:
            adata: Input data
            target_genes: List of genes to perturb
            perturb_type: Type of perturbation ('overexpress', 'inhibit', 'delete')
            **kwargs: Additional parameters
            
        Returns:
            Perturbation results
        """
        if hasattr(self.model, 'perturb_genes'):
            return self.model.perturb_genes(adata, target_genes, perturb_type, **kwargs)
        else:
            raise NotImplementedError(f"Gene perturbation not supported for {self.model.__class__.__name__}")
    
    def save_model(self, save_path, **kwargs):
        """Save the current model."""
        self.model.save_model(save_path, **kwargs)
    
    def __repr__(self):
        return f"SCLLMManager({self.model})"


# Convenience functions for quick access
def load_scgpt(model_path: Union[str, Path], device: Optional[str] = None, **kwargs):
    """
    Quick function to load a scGPT model.
    
    Args:
        model_path: Path to the scGPT model directory
        device: Device to run the model on
        **kwargs: Additional parameters
        
    Returns:
        Loaded scGPT model
    """
    return ModelFactory.create_model("scgpt", model_path, device, **kwargs)


def annotate_with_scgpt(adata, model_path: Union[str, Path], device: Optional[str] = None, **kwargs):
    """
    Quick function to annotate cells with scGPT.
    
    Args:
        adata: AnnData object
        model_path: Path to the scGPT model
        device: Device to run the model on
        **kwargs: Additional parameters
        
    Returns:
        Annotation results
    """
    model = load_scgpt(model_path, device, **kwargs)
    return model.predict(adata, task="annotation", **kwargs)


def fine_tune_scgpt(train_adata, 
                   model_path: Union[str, Path],
                   valid_adata=None,
                   save_path: Optional[Union[str, Path]] = None,
                   device: Optional[str] = None,
                   **kwargs):
    """
    Complete workflow for fine-tuning scGPT on reference data.
    
    Args:
        train_adata: Training data with 'celltype' in .obs
        model_path: Path to pretrained scGPT model
        valid_adata: Validation data (optional)
        save_path: Path to save fine-tuned model (optional)
        device: Device for training
        **kwargs: Training parameters (epochs, lr, batch_size, etc.)
        
    Returns:
        Dictionary with fine-tuning results and trained model
    """
    print("🚀 Starting scGPT fine-tuning workflow...")
    
    # Load pretrained model
    model = load_scgpt(model_path, device, **kwargs)
    
    # Fine-tune the model
    results = model.fine_tune(
        train_adata=train_adata,
        valid_adata=valid_adata,
        task="annotation",
        **kwargs
    )
    
    # Save fine-tuned model if requested
    if save_path is not None:
        model.save_model(save_path)
        print(f"💾 Fine-tuned model saved to: {save_path}")
    
    return {
        "model": model,
        "results": results
    }


def predict_celltypes_workflow(query_adata,
                             finetuned_model_path: Union[str, Path],
                             device: Optional[str] = None,
                             save_predictions: bool = True,
                             **kwargs):
    """
    Complete workflow for predicting cell types on query data.
    
    Args:
        query_adata: Query data to predict
        finetuned_model_path: Path to fine-tuned scGPT model
        device: Device for prediction
        save_predictions: Whether to save predictions to query_adata.obs
        **kwargs: Prediction parameters
        
    Returns:
        Dictionary with predictions and metadata
    """
    print("🔍 Starting cell type prediction workflow...")
    
    # Load fine-tuned model
    manager = SCLLMManager(
        model_type="scgpt",
        model_path=finetuned_model_path,
        device=device
    )
    
    # Load celltype mapping
    manager.model.load_celltype_mapping(finetuned_model_path)
    
    # Predict cell types
    results = manager.model.predict_celltypes(query_adata, **kwargs)
    
    # Add predictions to adata if requested
    if save_predictions and 'predicted_celltypes' in results:
        query_adata.obs['predicted_celltype'] = results['predicted_celltypes']
        query_adata.obs['predicted_celltype_id'] = results['predictions']
        print("✓ Predictions added to query_adata.obs")
    
    return results


def end_to_end_scgpt_annotation(reference_adata,
                               query_adata,
                               pretrained_model_path: Union[str, Path],
                               save_finetuned_path: Optional[Union[str, Path]] = None,
                               device: Optional[str] = None,
                               validation_split: float = 0.2,
                               **kwargs):
    """
    Complete end-to-end workflow: fine-tune on reference data and predict on query data.
    
    Args:
        reference_adata: Reference data with known cell types
        query_adata: Query data to predict
        pretrained_model_path: Path to pretrained scGPT model
        save_finetuned_path: Path to save fine-tuned model
        device: Device for computation
        validation_split: Fraction of reference data for validation
        **kwargs: Training and prediction parameters
        
    Returns:
        Dictionary with fine-tuning results and predictions
    """
    from sklearn.model_selection import train_test_split
    
    print("🎯 Starting end-to-end scGPT annotation workflow...")
    
    # Split reference data into train/validation
    if validation_split > 0:
        train_idx, val_idx = train_test_split(
            range(reference_adata.n_obs),
            test_size=validation_split,
            stratify=reference_adata.obs['celltype'],
            random_state=kwargs.get('random_state', 42)
        )
        train_adata = reference_adata[train_idx].copy()  
        valid_adata = reference_adata[val_idx].copy()
        print(f"📊 Split data: {len(train_idx)} train, {len(val_idx)} validation")
    else:
        train_adata = reference_adata.copy()
        valid_adata = None
        print(f"📊 Using all {reference_adata.n_obs} cells for training")
    
    # Fine-tune model
    fine_tune_result = fine_tune_scgpt(
        train_adata=train_adata,
        model_path=pretrained_model_path, 
        valid_adata=valid_adata,
        save_path=save_finetuned_path,
        device=device,
        **kwargs
    )
    
    # Predict on query data using fine-tuned model
    if save_finetuned_path is not None:
        prediction_results = predict_celltypes_workflow(
            query_adata=query_adata,
            finetuned_model_path=save_finetuned_path,
            device=device,
            **kwargs
        )
    else:
        # Use the fine-tuned model directly
        prediction_results = fine_tune_result["model"].predict_celltypes(query_adata, **kwargs)
        query_adata.obs['predicted_celltype'] = prediction_results['predicted_celltypes']
        query_adata.obs['predicted_celltype_id'] = prediction_results['predictions']
    
    print("🎉 End-to-end annotation completed!")
    
    return {
        "fine_tune_results": fine_tune_result["results"],
        "prediction_results": prediction_results,
        "model": fine_tune_result["model"]
    }


def train_integration_scgpt(train_adata,
                           model_path: Union[str, Path],
                           batch_key: str = "batch",
                           valid_adata=None,
                           save_path: Optional[Union[str, Path]] = None,
                           device: Optional[str] = None,
                           **kwargs):
    """
    Complete workflow for training scGPT on integration task.
    
    Args:
        train_adata: Training data with batch labels in .obs[batch_key]
        model_path: Path to pretrained scGPT model
        batch_key: Column name for batch labels (default: "batch")
        valid_adata: Validation data (optional)
        save_path: Path to save trained model (optional)
        device: Device for training
        **kwargs: Training parameters (epochs, lr, batch_size, etc.)
        
    Returns:
        Dictionary with training results and trained model
    """
    print("🚀 Starting scGPT integration training workflow...")
    
    # Load pretrained model with integration support
    model = load_scgpt(model_path, device, **kwargs)
    
    # Train for integration
    results = model.train_integration(
        train_adata=train_adata,
        valid_adata=valid_adata,
        batch_key=batch_key,
        **kwargs
    )
    
    # Save trained model if requested
    if save_path is not None:
        model.save_model(save_path)
        print(f"💾 Integration model saved to: {save_path}")
    
    return {
        "model": model,
        "results": results
    }


def integrate_batches_workflow(query_adata,
                             integration_model_path: Union[str, Path],
                             batch_key: str = "batch",
                             device: Optional[str] = None,
                             save_embeddings: bool = True,
                             **kwargs):
    """
    Complete workflow for batch integration on query data.
    
    Args:
        query_adata: Query data with batch information
        integration_model_path: Path to trained integration model
        batch_key: Column name for batch labels
        device: Device for integration
        save_embeddings: Whether to save embeddings to query_adata.obsm
        **kwargs: Integration parameters
        
    Returns:
        Dictionary with integration results
    """
    print("🔍 Starting batch integration workflow...")
    
    # Load integration model
    manager = SCLLMManager(
        model_type="scgpt",
        model_path=integration_model_path,
        device=device
    )
    
    # Load integration metadata
    if hasattr(manager.model, 'load_integration_metadata'):
        manager.model.load_integration_metadata(integration_model_path)
    
    # Perform integration
    results = manager.model.predict(query_adata, task="integration", batch_key=batch_key, **kwargs)
    
    # Add integrated embeddings to adata if requested
    if save_embeddings and 'embeddings' in results:
        query_adata.obsm['X_scgpt_integrated'] = results['embeddings']
        print("✓ Integrated embeddings added to query_adata.obsm['X_scgpt_integrated']")
    
    return results


def end_to_end_scgpt_integration(train_adata,
                                query_adata,
                                pretrained_model_path: Union[str, Path],
                                batch_key: str = "batch",
                                save_integration_path: Optional[Union[str, Path]] = None,
                                device: Optional[str] = None,
                                validation_split: float = 0.2,
                                **kwargs):
    """
    Complete end-to-end workflow: train integration model and integrate query data.
    
    Args:
        train_adata: Training data with batch labels for integration
        query_adata: Query data to integrate
        pretrained_model_path: Path to pretrained scGPT model
        batch_key: Column name for batch labels
        save_integration_path: Path to save integration model
        device: Device for computation
        validation_split: Fraction of training data for validation
        **kwargs: Training and integration parameters
        
    Returns:
        Dictionary with training results and integration results
    """
    from sklearn.model_selection import train_test_split
    
    print("🎯 Starting end-to-end scGPT integration workflow...")
    
    # Validate batch information
    if batch_key not in train_adata.obs:
        raise ValueError(f"Training data must have '{batch_key}' column in .obs")
    if batch_key not in query_adata.obs:
        raise ValueError(f"Query data must have '{batch_key}' column in .obs")
    
    print(f"Training data: {train_adata.n_obs} cells from {train_adata.obs[batch_key].nunique()} batches")
    print(f"Query data: {query_adata.n_obs} cells from {query_adata.obs[batch_key].nunique()} batches")
    
    # Split training data into train/validation if requested
    if validation_split > 0 and validation_split < 1:
        # Stratify by batch to ensure all batches are represented
        train_idx, val_idx = train_test_split(
            range(train_adata.n_obs),
            test_size=validation_split,
            stratify=train_adata.obs[batch_key],
            random_state=kwargs.get('random_state', 42)
        )
        train_split = train_adata[train_idx].copy()  
        valid_split = train_adata[val_idx].copy()
        print(f"📊 Split training data: {len(train_idx)} train, {len(val_idx)} validation")
    else:
        train_split = train_adata.copy()
        valid_split = None
        print(f"📊 Using all {train_adata.n_obs} cells for training")
    
    # Train integration model
    train_result = train_integration_scgpt(
        train_adata=train_split,
        model_path=pretrained_model_path,
        batch_key=batch_key,
        valid_adata=valid_split,
        save_path=save_integration_path,
        device=device,
        **kwargs
    )
    
    # Integrate query data using trained model
    if save_integration_path is not None:
        integration_results = integrate_batches_workflow(
            query_adata=query_adata,
            integration_model_path=save_integration_path,
            batch_key=batch_key,
            device=device,
            **kwargs
        )
    else:
        # Use the trained model directly
        integration_results = train_result["model"].predict(
            query_adata, task="integration", batch_key=batch_key, **kwargs
        )
        query_adata.obsm['X_scgpt_integrated'] = integration_results['embeddings']
    
    print("🎉 End-to-end integration completed!")
    
    return {
        "train_results": train_result["results"],
        "integration_results": integration_results,
        "model": train_result["model"]
    }


def integrate_with_scgpt(query_adata,
                        model_path: Union[str, Path],
                        batch_key: str = "batch",
                        correction_method: str = "combat",
                        device: Optional[str] = None,
                        save_embeddings: bool = True,
                        **kwargs):
    """
    Perform batch integration using scGPT model.
    
    This function automatically detects whether to use:
    - Fine-tuned integration model (if available)
    - Pre-trained model with post-hoc correction methods
    
    Args:
        query_adata: Query data with batch information
        model_path: Path to scGPT model (pre-trained or fine-tuned)
        batch_key: Column name for batch labels
        correction_method: Post-hoc correction method ('combat', 'mnn', 'center_scale', 'none')
        device: Device for computation
        save_embeddings: Whether to save integrated embeddings to query_adata.obsm
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with integration results
    """
    print("🔄 Starting scGPT batch integration...")
    
    # Load model
    manager = SCLLMManager(
        model_type="scgpt",
        model_path=model_path,
        device=device
    )
    
    # Perform integration using unified interface
    results = manager.model.integrate(
        query_adata, 
        batch_key=batch_key,
        correction_method=correction_method,
        **kwargs
    )
    
    # Add integrated embeddings to adata if requested
    if save_embeddings and 'embeddings' in results:
        query_adata.obsm['X_scgpt_integrated'] = results['embeddings']
        print("✓ Integrated embeddings added to query_adata.obsm['X_scgpt_integrated']")
        
        # Also save original embeddings if available (for post-hoc methods)
        if 'original_embeddings' in results:
            query_adata.obsm['X_scgpt_original'] = results['original_embeddings']
            print("✓ Original embeddings added to query_adata.obsm['X_scgpt_original']")
    
    return results


# Convenience functions for scFoundation
def load_scfoundation(model_path: Union[str, Path], 
                      device: Optional[str] = None, **kwargs):
    """
    Quick function to load a scFoundation model.
    
    Args:
        model_path: Path to the scFoundation model file (.ckpt)
        device: Device to run the model on
        **kwargs: Additional parameters (key, version, etc.)
        
    Returns:
        Loaded scFoundation model
    """
    return ModelFactory.create_model("scfoundation", model_path, device, **kwargs)


def get_embeddings_with_scfoundation(adata, model_path: Union[str, Path], 
                                    device: Optional[str] = None, 
                                    output_type: str = "cell",
                                    **kwargs):
    """
    Quick function to get embeddings with scFoundation.
    
    Args:
        adata: AnnData object
        model_path: Path to the scFoundation model
        device: Device to run the model on
        output_type: Type of embeddings ('cell', 'gene', 'gene_batch')
        **kwargs: Additional parameters
        
    Returns:
        Embeddings
    """
    model = load_scfoundation(model_path, device, **kwargs)
    return model.get_embeddings(adata, output_type=output_type, **kwargs)


def end_to_end_scfoundation_embedding(adata,
                                     model_path: Union[str, Path],
                                     save_embeddings_path: Optional[Union[str, Path]] = None,
                                     device: Optional[str] = None,
                                     output_type: str = "cell",
                                     pool_type: str = "all",
                                     **kwargs):
    """
    Complete workflow for extracting embeddings with scFoundation.
    
    Args:
        adata: Input AnnData object
        model_path: Path to scFoundation model
        save_embeddings_path: Path to save embeddings (optional)
        device: Device for computation
        output_type: Type of embeddings ('cell', 'gene', 'gene_batch') 
        pool_type: Pooling type for cell embeddings ('all', 'max')
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with embeddings and metadata
    """
    print("🚀 Starting scFoundation embedding extraction workflow...")
    
    # Load model
    manager = SCLLMManager(
        model_type="scfoundation",
        model_path=model_path,
        device=device
    )
    
    # Extract embeddings
    results = manager.get_embeddings(
        adata, 
        output_type=output_type,
        pool_type=pool_type,
        **kwargs
    )
    
    # Add embeddings to adata
    if output_type == "cell":
        adata.obsm['X_scfoundation'] = results
        print("✓ Cell embeddings added to adata.obsm['X_scfoundation']")
    elif output_type in ["gene", "gene_batch"]:
        adata.varm[f'X_scfoundation_{output_type}'] = results
        print(f"✓ Gene embeddings added to adata.varm['X_scfoundation_{output_type}']")
    
    # Save embeddings if requested
    if save_embeddings_path is not None:
        np.save(save_embeddings_path, results)
        print(f"💾 Embeddings saved to: {save_embeddings_path}")
    
    print("🎉 scFoundation embedding extraction completed!")
    
    return {
        "embeddings": results,
        "output_type": output_type,
        "pool_type": pool_type if output_type == "cell" else None,
        "model_path": str(model_path)
    }


def fine_tune_scfoundation(train_adata,
                          model_path: Union[str, Path],
                          valid_adata=None,
                          save_path: Optional[Union[str, Path]] = None,
                          device: Optional[str] = None,
                          **kwargs):
    """
    Complete workflow for fine-tuning scFoundation on reference data.
    
    Args:
        train_adata: Training data with 'celltype' in .obs
        model_path: Path to pretrained scFoundation model (.ckpt)
        valid_adata: Validation data (optional)
        save_path: Path to save fine-tuned model (optional)
        device: Device for training
        **kwargs: Training parameters (epochs, lr, batch_size, frozen_more, etc.)
        
    Returns:
        Dictionary with fine-tuning results and trained model
    """
    print("🚀 Starting scFoundation fine-tuning workflow...")
    
    # Load pretrained model
    model = load_scfoundation(model_path, device, **kwargs)
    
    # Fine-tune the model
    results = model.fine_tune(
        train_adata=train_adata,
        valid_adata=valid_adata,
        task="annotation",
        **kwargs
    )
    
    # Save fine-tuned model if requested
    if save_path is not None:
        model.save_model(save_path)
        print(f"💾 Fine-tuned model saved to: {save_path}")
    
    return {
        "model": model,
        "results": results
    }


def predict_celltypes_with_scfoundation(query_adata,
                                       finetuned_model_path: Union[str, Path],
                                       device: Optional[str] = None,
                                       save_predictions: bool = True,
                                       **kwargs):
    """
    Complete workflow for predicting cell types with fine-tuned scFoundation.
    
    Args:
        query_adata: Query data to predict
        finetuned_model_path: Path to fine-tuned scFoundation model
        device: Device for prediction
        save_predictions: Whether to save predictions to query_adata.obs
        **kwargs: Prediction parameters
        
    Returns:
        Dictionary with predictions and metadata
    """
    print("🔍 Starting scFoundation cell type prediction workflow...")
    
    # Load fine-tuned model
    manager = SCLLMManager(
        model_type="scfoundation",
        model_path=finetuned_model_path,
        device=device
    )
    
    # Predict cell types
    results = manager.model.predict_with_finetune(query_adata, **kwargs)
    
    # Add predictions to adata if requested
    if save_predictions and 'predicted_celltypes' in results:
        query_adata.obs['predicted_celltype'] = results['predicted_celltypes']
        query_adata.obs['predicted_celltype_id'] = results['predictions']
        query_adata.obsm['prediction_probabilities'] = results['probabilities']
        print("✓ Predictions added to query_adata.obs")
    
    return results


def end_to_end_scfoundation_annotation(reference_adata,
                                      query_adata,
                                      pretrained_model_path: Union[str, Path],
                                      save_finetuned_path: Optional[Union[str, Path]] = None,
                                      device: Optional[str] = None,
                                      validation_split: float = 0.2,
                                      **kwargs):
    """
    Complete end-to-end workflow: fine-tune scFoundation and predict on query data.
    
    Args:
        reference_adata: Reference data with known cell types
        query_adata: Query data to predict
        pretrained_model_path: Path to pretrained scFoundation model
        save_finetuned_path: Path to save fine-tuned model
        device: Device for computation
        validation_split: Fraction of reference data for validation
        **kwargs: Training and prediction parameters
        
    Returns:
        Dictionary with fine-tuning results and predictions
    """
    from sklearn.model_selection import train_test_split
    
    print("🎯 Starting end-to-end scFoundation annotation workflow...")
    
    # Validate data
    if 'celltype' not in reference_adata.obs:
        raise ValueError("Reference data must have 'celltype' column in .obs")
    
    # Split reference data into train/validation
    if validation_split > 0:
        train_idx, val_idx = train_test_split(
            range(reference_adata.n_obs),
            test_size=validation_split,
            stratify=reference_adata.obs['celltype'],
            random_state=kwargs.get('random_state', 42)
        )
        train_adata = reference_adata[train_idx].copy()  
        valid_adata = reference_adata[val_idx].copy()
        print(f"📊 Split data: {len(train_idx)} train, {len(val_idx)} validation")
    else:
        train_adata = reference_adata.copy()
        valid_adata = None
        print(f"📊 Using all {reference_adata.n_obs} cells for training")
    
    # Fine-tune model
    fine_tune_result = fine_tune_scfoundation(
        train_adata=train_adata,
        model_path=pretrained_model_path, 
        valid_adata=valid_adata,
        save_path=save_finetuned_path,
        device=device,
        **kwargs
    )
    
    # Predict on query data using fine-tuned model
    if save_finetuned_path is not None:
        prediction_results = predict_celltypes_with_scfoundation(
            query_adata=query_adata,
            finetuned_model_path=save_finetuned_path,
            device=device,
            **kwargs
        )
    else:
        # Use the fine-tuned model directly
        prediction_results = fine_tune_result["model"].predict_with_finetune(query_adata, **kwargs)
        query_adata.obs['predicted_celltype'] = prediction_results['predicted_celltypes']
        query_adata.obs['predicted_celltype_id'] = prediction_results['predictions']
        query_adata.obsm['prediction_probabilities'] = prediction_results['probabilities']
    
    print("🎉 End-to-end scFoundation annotation completed!")
    
    return {
        "fine_tune_results": fine_tune_result["results"],
        "prediction_results": prediction_results,
        "model": fine_tune_result["model"]
    }


# Integration functions for scFoundation
def integrate_with_scfoundation(query_adata,
                               model_path: Union[str, Path],
                               batch_key: str = "batch",
                               correction_method: str = "harmony",
                               device: Optional[str] = None,
                               save_embeddings: bool = True,
                               **kwargs):
    """
    Complete workflow for batch integration using scFoundation.
    
    Args:
        query_adata: Query data with batch information
        model_path: Path to scFoundation model
        batch_key: Column name for batch labels
        correction_method: Integration method ('harmony', 'combat', 'scanorama', 'mnn')
        device: Device for computation
        save_embeddings: Whether to save embeddings to query_adata.obsm
        **kwargs: Integration parameters
        
    Returns:
        Dictionary with integration results
    """
    print("🔍 Starting scFoundation batch integration workflow...")
    
    # Load scFoundation model
    manager = SCLLMManager(
        model_type="scfoundation",
        model_path=model_path,
        device=device
    )
    
    # Perform integration
    results = manager.model.predict(
        query_adata, 
        task="integration", 
        batch_key=batch_key,
        correction_method=correction_method,
        **kwargs
    )
    
    # Add integrated embeddings to adata if requested
    if save_embeddings and 'embeddings' in results:
        query_adata.obsm['X_scfoundation_integrated'] = results['embeddings']
        print("✓ Integrated embeddings added to query_adata.obsm['X_scfoundation_integrated']")
        
        # Also save original embeddings for comparison
        if 'original_embeddings' in results:
            query_adata.obsm['X_scfoundation_original'] = results['original_embeddings']
            print("✓ Original embeddings added to query_adata.obsm['X_scfoundation_original']")
    
    return results


def integrate_batches_with_scfoundation(query_adata,
                                       model_path: Union[str, Path],
                                       batch_key: str = "batch",
                                       correction_method: str = "harmony",
                                       device: Optional[str] = None,
                                       save_embeddings: bool = True,
                                       **kwargs):
    """
    Alias for integrate_with_scfoundation for consistency with scGPT naming.
    """
    return integrate_with_scfoundation(
        query_adata=query_adata,
        model_path=model_path,
        batch_key=batch_key,
        correction_method=correction_method,
        device=device,
        save_embeddings=save_embeddings,
        **kwargs
    )


def end_to_end_scfoundation_integration(query_adata,
                                       model_path: Union[str, Path],
                                       batch_key: str = "batch",
                                       correction_methods: List[str] = ["harmony"],
                                       device: Optional[str] = None,
                                       save_all_methods: bool = False,
                                       **kwargs):
    """
    Complete end-to-end integration workflow with multiple correction methods.
    
    Args:
        query_adata: Query data with batch information
        model_path: Path to scFoundation model
        batch_key: Column name for batch labels
        correction_methods: List of correction methods to try
        device: Device for computation
        save_all_methods: Whether to save results from all methods
        **kwargs: Integration parameters
        
    Returns:
        Dictionary with results from all methods
    """
    print("🎯 Starting end-to-end scFoundation integration workflow...")
    
    # Load model once
    manager = SCLLMManager(
        model_type="scfoundation",
        model_path=model_path,
        device=device
    )
    
    all_results = {}
    
    for method in correction_methods:
        print(f"\n📊 Testing integration method: {method}")
        
        try:
            # Perform integration with this method
            results = manager.model.predict(
                query_adata,
                task="integration",
                batch_key=batch_key,
                correction_method=method,
                **kwargs
            )
            
            all_results[method] = results
            
            # Save embeddings if requested
            if save_all_methods or method == correction_methods[0]:
                if method == correction_methods[0]:
                    # Save the first method as the default
                    query_adata.obsm['X_scfoundation_integrated'] = results['embeddings']
                    print(f"✓ {method} embeddings saved as default integration")
                
                if save_all_methods:
                    query_adata.obsm[f'X_scfoundation_{method}'] = results['embeddings']
                    print(f"✓ {method} embeddings saved to query_adata.obsm")
            
        except Exception as e:
            print(f"❌ Integration with {method} failed: {e}")
            all_results[method] = {"error": str(e)}
    
    # Save original embeddings for comparison
    if len(all_results) > 0:
        first_successful = next((r for r in all_results.values() if 'embeddings' in r), None)
        if first_successful and 'original_embeddings' in first_successful:
            query_adata.obsm['X_scfoundation_original'] = first_successful['original_embeddings']
            print("✓ Original embeddings saved for comparison")
    
    print("🎉 End-to-end scFoundation integration completed!")
    
    # Summary
    successful_methods = [m for m, r in all_results.items() if 'embeddings' in r]
    failed_methods = [m for m, r in all_results.items() if 'error' in r]
    
    print(f"\n📈 Integration Summary:")
    print(f"  Successful methods: {successful_methods}")
    if failed_methods:
        print(f"  Failed methods: {failed_methods}")
    
    return {
        "results": all_results,
        "successful_methods": successful_methods,
        "failed_methods": failed_methods,
        "batch_key": batch_key,
        "model_path": str(model_path)
    }


# Geneformer convenience functions
def load_geneformer(model_path: Union[str, Path],
                   gene_median_file: Union[str, Path],
                   token_dictionary_file: Union[str, Path], 
                   gene_mapping_file: Union[str, Path],
                   device: Optional[str] = None, 
                   model_version: str = "V1",
                   **kwargs):
    """
    Quick function to load a Geneformer model with external dictionary files.
    
    Args:
        model_path: Path to the Geneformer model directory
        gene_median_file: Path to gene median dictionary file
        token_dictionary_file: Path to token dictionary file
        gene_mapping_file: Path to gene mapping file
        device: Device to run the model on
        model_version: Model version ('V1' or 'V2')
        **kwargs: Additional parameters
        
    Returns:
        Loaded Geneformer model
        
    Example:
        model = load_geneformer(
            '/path/to/geneformer/model',
            gene_median_file='/path/to/gene_median_dictionary_gc104M.pkl',
            token_dictionary_file='/path/to/token_dictionary_gc104M.pkl',
            gene_mapping_file='/path/to/ensembl_mapping_dict_gc104M.pkl'
        )
        
    Note:
        Dictionary files are not included in the package. Download them from:
        https://huggingface.co/ctheodoris/Geneformer
    """
    model = ModelFactory.create_model(
        "geneformer", 
        device=device, 
        model_version=model_version,
        **kwargs
    )
    model.load_model(
        model_path,
        gene_median_file=gene_median_file,
        token_dictionary_file=token_dictionary_file,
        gene_mapping_file=gene_mapping_file,
        **kwargs
    )
    return model


def annotate_with_geneformer(adata, 
                           model_path: Union[str, Path], 
                           device: Optional[str] = None,
                           model_version: str = "V1",
                           **kwargs):
    """
    Quick function to annotate cells with Geneformer.
    
    Args:
        adata: AnnData object
        model_path: Path to the Geneformer model
        device: Device to run the model on
        model_version: Model version ('V1' or 'V2')
        **kwargs: Additional parameters
        
    Returns:
        Annotation results
    """
    model = load_geneformer(model_path, device, model_version, **kwargs)
    return model.predict(adata, task="annotation", **kwargs)


def extract_geneformer_embeddings(adata,
                                 model_path: Union[str, Path],
                                 device: Optional[str] = None,
                                 model_version: str = "V1",
                                 emb_layer: int = 0,
                                 **kwargs):
    """
    Extract cell embeddings using Geneformer.
    
    Args:
        adata: AnnData object
        model_path: Path to the Geneformer model
        device: Device to run the model on
        model_version: Model version ('V1' or 'V2')
        emb_layer: Which layer to extract embeddings from
        **kwargs: Additional parameters
        
    Returns:
        Cell embeddings
    """
    model = load_geneformer(model_path, device, model_version, **kwargs)
    return model.get_embeddings(adata, emb_layer=emb_layer, **kwargs)


def perturb_genes_with_geneformer(adata,
                                 target_genes: List[str],
                                 model_path: Union[str, Path],
                                 perturb_type: str = "overexpress",
                                 device: Optional[str] = None,
                                 model_version: str = "V1",
                                 **kwargs):
    """
    Perform in silico gene perturbation using Geneformer.
    
    Args:
        adata: AnnData object
        target_genes: List of genes to perturb
        model_path: Path to the Geneformer model
        perturb_type: Type of perturbation ('overexpress', 'inhibit', 'delete')
        device: Device to run the model on
        model_version: Model version ('V1' or 'V2')
        **kwargs: Additional parameters
        
    Returns:
        Perturbation results
    """
    model = load_geneformer(model_path, device, model_version, **kwargs)
    return model.perturb_genes(adata, target_genes, perturb_type, **kwargs)


def fine_tune_geneformer(train_adata,
                        model_path: Union[str, Path],
                        valid_adata=None,
                        save_path: Optional[Union[str, Path]] = None,
                        device: Optional[str] = None,
                        model_version: str = "V1",
                        task: str = "annotation",
                        **kwargs):
    """
    Fine-tune a Geneformer model for cell classification.
    
    Args:
        train_adata: Training data with cell type labels
        model_path: Path to pre-trained Geneformer model
        valid_adata: Validation data (optional)
        save_path: Where to save the fine-tuned model
        device: Device for computation
        model_version: Model version ('V1' or 'V2')
        task: Task type ('annotation', 'gene_classification')
        **kwargs: Training parameters
        
    Returns:
        Training results and metrics
    """
    print("🔧 Starting Geneformer fine-tuning...")
    
    # Load model
    model = load_geneformer(model_path, device, model_version, **kwargs)
    
    # Fine-tune
    results = model.fine_tune(
        train_adata=train_adata,
        valid_adata=valid_adata,
        task=task,
        **kwargs
    )
    
    # Save if requested
    if save_path is not None:
        model.save_model(save_path)
        print(f"✓ Fine-tuned model saved to {save_path}")
    
    return results


# Convenience functions for CellPLM
def load_cellplm(model_path: Union[str, Path], 
                 pretrain_version: str = "20231027_85M",
                 device: Optional[str] = None, 
                 **kwargs):
    """
    Quick function to load a CellPLM model.
    
    Args:
        model_path: Path to the CellPLM checkpoint directory
        pretrain_version: Version of pretrained model (e.g., '20231027_85M')
        device: Device to run the model on
        **kwargs: Additional parameters
        
    Returns:
        Loaded CellPLM model
    """
    return ModelFactory.create_model(
        "cellplm", 
        model_path, 
        device, 
        pretrain_version=pretrain_version,
        **kwargs
    )


def get_embeddings_with_cellplm(adata, 
                               model_path: Union[str, Path],
                               pretrain_version: str = "20231027_85M",
                               device: Optional[str] = None,
                               **kwargs):
    """
    Quick function to get embeddings with CellPLM.
    
    Args:
        adata: AnnData object
        model_path: Path to the CellPLM checkpoint directory
        pretrain_version: Version of pretrained model
        device: Device to run the model on
        **kwargs: Additional parameters
        
    Returns:
        Cell embeddings
    """
    model = load_cellplm(model_path, pretrain_version, device, **kwargs)
    return model.get_embeddings(adata, **kwargs)


def end_to_end_cellplm_embedding(adata,
                                model_path: Union[str, Path],
                                pretrain_version: str = "20231027_85M",
                                save_embeddings_path: Optional[Union[str, Path]] = None,
                                device: Optional[str] = None,
                                **kwargs):
    """
    Complete workflow for extracting embeddings with CellPLM.
    
    Args:
        adata: Input AnnData object
        model_path: Path to CellPLM checkpoint directory
        pretrain_version: Version of pretrained model
        save_embeddings_path: Path to save embeddings (optional)
        device: Device for computation
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with embeddings and metadata
    """
    print("🚀 Starting CellPLM embedding extraction workflow...")
    
    # Load model
    manager = SCLLMManager(
        model_type="cellplm",
        model_path=model_path,
        device=device,
        pretrain_version=pretrain_version
    )
    
    # Extract embeddings
    embeddings = manager.get_embeddings(adata, **kwargs)
    
    # Add embeddings to adata
    adata.obsm['X_cellplm'] = embeddings
    print("✓ Cell embeddings added to adata.obsm['X_cellplm']")
    
    # Save embeddings if requested
    if save_embeddings_path is not None:
        np.save(save_embeddings_path, embeddings)
        print(f"💾 Embeddings saved to: {save_embeddings_path}")
    
    print("🎉 CellPLM embedding extraction completed!")
    
    return {
        'embeddings': embeddings,
        'model_path': str(model_path),
        'pretrain_version': pretrain_version
    }


def fine_tune_cellplm(train_adata,
                     model_path: Union[str, Path],
                     pretrain_version: str = "20231027_85M",
                     valid_adata=None,
                     save_path: Optional[Union[str, Path]] = None,
                     device: Optional[str] = None,
                     task: str = "annotation",
                     **kwargs):
    """
    Complete workflow for fine-tuning CellPLM on reference data.
    
    Args:
        train_adata: Training data with 'celltype' in .obs (for annotation)
        model_path: Path to CellPLM checkpoint directory
        pretrain_version: Version of pretrained model
        valid_adata: Validation data (optional)
        save_path: Path to save fine-tuned model (optional)
        device: Device for training
        task: Task type ('annotation', 'imputation')
        **kwargs: Training parameters
        
    Returns:
        Dictionary with fine-tuning results and trained model
    """
    print(f"🚀 Starting CellPLM {task} fine-tuning workflow...")
    
    # Load model
    model = load_cellplm(model_path, pretrain_version, device, task=task, **kwargs)
    
    # Fine-tune the model
    results = model.fine_tune(
        train_adata=train_adata,
        valid_adata=valid_adata,
        task=task,
        **kwargs
    )
    
    # Save fine-tuned model if requested
    if save_path is not None:
        model.save_model(save_path)
        print(f"💾 Fine-tuned model saved to: {save_path}")
    
    return {
        "model": model,
        "results": results
    }


def predict_celltypes_with_cellplm(query_adata,
                                  finetuned_model_path: Union[str, Path],
                                  pretrain_version: str = "20231027_85M",
                                  device: Optional[str] = None,
                                  save_predictions: bool = True,
                                  **kwargs):
    """
    Complete workflow for predicting cell types with fine-tuned CellPLM.
    
    Args:
        query_adata: Query data to predict
        finetuned_model_path: Path to fine-tuned CellPLM model
        pretrain_version: Version of pretrained model
        device: Device for prediction
        save_predictions: Whether to save predictions to query_adata.obs
        **kwargs: Prediction parameters
        
    Returns:
        Dictionary with predictions and metadata
    """
    print("🔍 Starting CellPLM cell type prediction workflow...")
    
    # Load fine-tuned model
    manager = SCLLMManager(
        model_type="cellplm",
        model_path=finetuned_model_path,
        device=device,
        pretrain_version=pretrain_version,
        task="annotation"
    )
    
    # Predict cell types
    results = manager.model.predict_celltypes(query_adata, **kwargs)
    
    # Add predictions to adata if requested
    if save_predictions:
        if 'predicted_celltypes' in results:
            query_adata.obs['predicted_celltype'] = results['predicted_celltypes']
        if 'predictions' in results:
            query_adata.obs['predicted_celltype_id'] = results['predictions']
        print("✓ Predictions added to query_adata.obs")
    
    return results


def end_to_end_cellplm_annotation(reference_adata,
                                 query_adata,
                                 model_path: Union[str, Path],
                                 pretrain_version: str = "20231027_85M",
                                 save_finetuned_path: Optional[Union[str, Path]] = None,
                                 device: Optional[str] = None,
                                 validation_split: float = 0.2,
                                 **kwargs):
    """
    Complete end-to-end workflow: fine-tune CellPLM and predict on query data.
    
    Args:
        reference_adata: Reference data with known cell types
        query_adata: Query data to predict
        model_path: Path to CellPLM checkpoint directory
        pretrain_version: Version of pretrained model
        save_finetuned_path: Path to save fine-tuned model
        device: Device for computation
        validation_split: Fraction of reference data for validation
        **kwargs: Training and prediction parameters
        
    Returns:
        Dictionary with fine-tuning results and predictions
    """
    print("🎯 Starting end-to-end CellPLM annotation workflow...")
    
    # Validate data
    if 'celltype' not in reference_adata.obs:
        raise ValueError("Reference data must have 'celltype' column in .obs")
    
    # Split reference data into train/validation
    if validation_split > 0:
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            range(reference_adata.n_obs),
            test_size=validation_split,
            stratify=reference_adata.obs['celltype'],
            random_state=kwargs.get('random_state', 42)
        )
        train_adata = reference_adata[train_idx].copy()  
        valid_adata = reference_adata[val_idx].copy()
        print(f"📊 Split data: {len(train_idx)} train, {len(val_idx)} validation")
    else:
        train_adata = reference_adata.copy()
        valid_adata = None
        print(f"📊 Using all {reference_adata.n_obs} cells for training")
    
    # Fine-tune model
    fine_tune_result = fine_tune_cellplm(
        train_adata=train_adata,
        model_path=model_path,
        pretrain_version=pretrain_version,
        valid_adata=valid_adata,
        save_path=save_finetuned_path,
        device=device,
        task="annotation",
        **kwargs
    )
    
    # Predict on query data using fine-tuned model
    if save_finetuned_path is not None:
        prediction_results = predict_celltypes_with_cellplm(
            query_adata=query_adata,
            finetuned_model_path=save_finetuned_path,
            pretrain_version=pretrain_version,
            device=device,
            **kwargs
        )
    else:
        # Use the fine-tuned model directly
        prediction_results = fine_tune_result["model"].predict_celltypes(query_adata, **kwargs)
        if 'predicted_celltypes' in prediction_results:
            query_adata.obs['predicted_celltype'] = prediction_results['predicted_celltypes']
        if 'predictions' in prediction_results:
            query_adata.obs['predicted_celltype_id'] = prediction_results['predictions']
    
    print("🎉 End-to-end CellPLM annotation completed!")
    
    return {
        "fine_tune_results": fine_tune_result["results"],
        "prediction_results": prediction_results,
        "model": fine_tune_result["model"]
    }


# Integration functions for CellPLM
def integrate_with_cellplm(query_adata,
                          model_path: Union[str, Path],
                          pretrain_version: str = "20231027_85M",
                          batch_key: str = "batch",
                          correction_method: str = "harmony",
                          device: Optional[str] = None,
                          save_embeddings: bool = True,
                          **kwargs):
    """
    Complete workflow for batch integration using CellPLM.
    
    Args:
        query_adata: Query data with batch information
        model_path: Path to CellPLM checkpoint directory
        pretrain_version: Version of pretrained model
        batch_key: Column name for batch labels
        correction_method: Integration method ('harmony', 'center_scale', 'none')
        device: Device for computation
        save_embeddings: Whether to save embeddings to query_adata.obsm
        **kwargs: Integration parameters
        
    Returns:
        Dictionary with integration results
    """
    print("🔍 Starting CellPLM batch integration workflow...")
    
    # Load CellPLM model
    manager = SCLLMManager(
        model_type="cellplm",
        model_path=model_path,
        device=device,
        pretrain_version=pretrain_version,
        task="embedding"
    )
    
    # Perform integration
    results = manager.model.integrate(
        query_adata, 
        batch_key=batch_key,
        correction_method=correction_method,
        **kwargs
    )
    
    # Add integrated embeddings to adata if requested
    if save_embeddings and 'embeddings' in results:
        query_adata.obsm['X_cellplm_integrated'] = results['embeddings']
        print("✓ Integrated embeddings added to query_adata.obsm['X_cellplm_integrated']")
        
        # Also save original embeddings for comparison
        if 'original_embeddings' in results:
            query_adata.obsm['X_cellplm_original'] = results['original_embeddings']
            print("✓ Original embeddings added to query_adata.obsm['X_cellplm_original']")
    
    return results


def end_to_end_cellplm_integration(query_adata,
                                  model_path: Union[str, Path],
                                  pretrain_version: str = "20231027_85M",
                                  batch_key: str = "batch",
                                  correction_methods: List[str] = ["harmony"],
                                  device: Optional[str] = None,
                                  save_all_methods: bool = False,
                                  **kwargs):
    """
    Complete end-to-end integration workflow with multiple correction methods.
    
    Args:
        query_adata: Query data with batch information
        model_path: Path to CellPLM checkpoint directory
        pretrain_version: Version of pretrained model
        batch_key: Column name for batch labels
        correction_methods: List of correction methods to try
        device: Device for computation
        save_all_methods: Whether to save results from all methods
        **kwargs: Integration parameters
        
    Returns:
        Dictionary with results from all methods
    """
    print("🎯 Starting end-to-end CellPLM integration workflow...")
    
    # Load model once
    manager = SCLLMManager(
        model_type="cellplm",
        model_path=model_path,
        device=device,
        pretrain_version=pretrain_version,
        task="embedding"
    )
    
    all_results = {}
    
    for method in correction_methods:
        print(f"\n📊 Testing integration method: {method}")
        
        try:
            # Perform integration with this method
            results = manager.model.integrate(
                query_adata,
                batch_key=batch_key,
                correction_method=method,
                **kwargs
            )
            
            all_results[method] = results
            
            # Save embeddings if requested
            if save_all_methods or method == correction_methods[0]:
                if method == correction_methods[0]:
                    # Save the first method as the default
                    query_adata.obsm['X_cellplm_integrated'] = results['embeddings']
                    print(f"✓ {method} embeddings saved as default integration")
                
                if save_all_methods:
                    query_adata.obsm[f'X_cellplm_{method}'] = results['embeddings']
                    print(f"✓ {method} embeddings saved to query_adata.obsm")
            
        except Exception as e:
            print(f"❌ Integration with {method} failed: {e}")
            all_results[method] = {"error": str(e)}
    
    # Save original embeddings for comparison
    if len(all_results) > 0:
        first_successful = next((r for r in all_results.values() if 'embeddings' in r), None)
        if first_successful and 'original_embeddings' in first_successful:
            query_adata.obsm['X_cellplm_original'] = first_successful['original_embeddings']
            print("✓ Original embeddings saved for comparison")
    
    print("🎉 End-to-end CellPLM integration completed!")
    
    # Summary
    successful_methods = [m for m, r in all_results.items() if 'embeddings' in r]
    failed_methods = [m for m, r in all_results.items() if 'error' in r]
    
    print(f"\n📈 Integration Summary:")
    print(f"  Successful methods: {successful_methods}")
    if failed_methods:
        print(f"  Failed methods: {failed_methods}")
    
    return {
        "results": all_results,
        "successful_methods": successful_methods,
        "failed_methods": failed_methods,
        "batch_key": batch_key,
        "model_path": str(model_path)
    }

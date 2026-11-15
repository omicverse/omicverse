"""
UCE model implementation for omicverse.

This module provides a wrapper for the UCE (Universal Cell Embeddings) model,
following the exact logic of the original eval_single_anndata.py script.
"""

import os
import tempfile
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from anndata import AnnData
from scipy.sparse import issparse

try:
    from .base import SCLLMBase, ModelConfig, TaskConfig
    from .utils.output_utils import SCLLMOutput, ModelProgressManager
except ImportError:
    from base import SCLLMBase, ModelConfig, TaskConfig
    from utils.output_utils import SCLLMOutput, ModelProgressManager


class UCEModel(SCLLMBase):
    """
    UCE (Universal Cell Embeddings) model wrapper.
    
    This implementation follows the exact logic of eval_single_anndata.py
    to ensure identical results.
    """
    
    def __init__(self, device: Optional[str] = None, **kwargs):
        """
        Initialize UCE model.
        
        Args:
            device: Device to run the model on ('cpu' or 'cuda')
            **kwargs: Additional UCE-specific parameters
        """
        super().__init__(model_name="UCE", device=device)
        
        # UCE configuration parameters (matching eval_single_anndata.py defaults)
        self.config = {
            'nlayers': kwargs.get('nlayers', 4),
            'output_dim': kwargs.get('output_dim', 1280),
            'd_hid': kwargs.get('d_hid', 5120),
            'token_dim': kwargs.get('token_dim', 5120),
            'nhead': 20,  # Fixed in UCE
            'dropout': 0.05,  # Fixed in UCE
            'pad_length': kwargs.get('pad_length', 1536),
            'sample_size': kwargs.get('sample_size', 1024),
            'batch_size': kwargs.get('batch_size', 25),
            'species': kwargs.get('species', 'human'),
            'multi_gpu': kwargs.get('multi_gpu', False),
            # Token indices (UCE defaults)
            'pad_token_idx': 0,
            'chrom_token_left_idx': 1,
            'chrom_token_right_idx': 2,
            'cls_token_idx': 3,
            'CHROM_TOKEN_OFFSET': 143574
        }
        
        # File paths (all external, no hardcoding)
        self.model_path = None
        self.token_file = None
        self.protein_embeddings_dir = None
        self.spec_chrom_csv_path = None
        self.offset_pkl_path = None
        
        SCLLMOutput.status("UCE model initialized", 'loaded')
    
    def load_model(self, model_path: Union[str, Path], 
                   token_file: Union[str, Path],
                   protein_embeddings_dir: Union[str, Path],
                   spec_chrom_csv_path: Union[str, Path],
                   offset_pkl_path: Union[str, Path],
                   **kwargs) -> None:
        """
        Load UCE model and required assets.
        
        Args:
            model_path: Path to UCE model weights (.torch file)
            token_file: Path to token embeddings file
            protein_embeddings_dir: Path to protein embeddings directory
            spec_chrom_csv_path: Path to species chromosome CSV file
            offset_pkl_path: Path to species offsets pickle file
            **kwargs: Additional parameters
        """
        SCLLMOutput.status("Loading UCE model and assets", 'loading')
        
        # Store all required file paths
        self.model_path = Path(model_path)
        self.token_file = Path(token_file)
        self.protein_embeddings_dir = Path(protein_embeddings_dir)
        self.spec_chrom_csv_path = Path(spec_chrom_csv_path)
        self.offset_pkl_path = Path(offset_pkl_path)
        
        # Validate and report all required files
        required_files = [
            (self.model_path, "Model weights"),
            (self.token_file, "Token embeddings"),
            (self.spec_chrom_csv_path, "Species chromosome mapping"),
            (self.offset_pkl_path, "Species offsets")
        ]
        
        SCLLMOutput.status("=== UCE Asset Files Validation ===", 'loading')
        for file_path, description in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"{description} not found: {file_path}")
            else:
                SCLLMOutput.status(f"✓ {description}: {file_path}", 'loaded')
        
        if not self.protein_embeddings_dir.exists():
            raise FileNotFoundError(f"Protein embeddings directory not found: {self.protein_embeddings_dir}")
        else:
            SCLLMOutput.status(f"✓ Protein embeddings directory: {self.protein_embeddings_dir}", 'loaded')
            
        # Check and report key protein embedding files
        key_protein_files = [
            "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
            "Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt"
        ]
        
        for filename in key_protein_files:
            file_path = self.protein_embeddings_dir / filename
            if file_path.exists():
                SCLLMOutput.status(f"✓ Found protein embedding: {filename}", 'loaded')
            else:
                SCLLMOutput.status(f"⚠ Missing protein embedding: {filename}", 'warning')
        
        # Report configuration
        SCLLMOutput.status("=== UCE Configuration ===", 'loading')
        config_items = [
            ('Species', self.config['species']),
            ('Batch size', self.config['batch_size']),
            ('Model layers', self.config['nlayers']),
            ('Output dimension', self.config['output_dim']),
            ('Token dimension', self.config['token_dim']),
            ('Hidden dimension', self.config['d_hid'])
        ]
        
        for name, value in config_items:
            SCLLMOutput.status(f"• {name}: {value}", 'loaded')
        
        # Update config with any provided parameters
        for key in ['nlayers', 'output_dim', 'd_hid', 'token_dim', 'batch_size', 'species']:
            if key in kwargs:
                old_value = self.config[key]
                self.config[key] = kwargs[key]
                if old_value != kwargs[key]:
                    SCLLMOutput.status(f"• Updated {key}: {old_value} → {kwargs[key]}", 'loaded')
        
        # Set global protein embeddings directory for UCE modules
        self._set_global_protein_embeddings_dir()
        
        # Initialize UCE TransformerModel directly during load_model
        self._initialize_uce_transformer_model()
        
        self.is_loaded = True
        SCLLMOutput.status("UCE model loaded successfully", 'loaded')
    
    def _initialize_uce_transformer_model(self):
        """Initialize UCE TransformerModel during load_model for fine-tuning access."""
        import torch
        import torch.nn as nn
        
        # Import UCE components
        try:
            from .UCE.model import TransformerModel
        except ImportError:
            import sys
            uce_path = Path(__file__).parent / "UCE"
            if str(uce_path) not in sys.path:
                sys.path.insert(0, str(uce_path))
            from model import TransformerModel
        
        SCLLMOutput.status("Initializing UCE TransformerModel for fine-tuning", 'loading')
        
        # Model parameters (same as in _run_eval_custom)
        token_dim = self.config['token_dim']
        emsize = 1280  # embedding dimension (fixed in UCE)
        d_hid = self.config['d_hid']
        nlayers = self.config['nlayers']
        nhead = 20  # fixed in UCE
        dropout = 0.05  # fixed in UCE
        
        # Create UCE TransformerModel
        model = TransformerModel(token_dim=token_dim, d_model=emsize, nhead=nhead,
                               d_hid=d_hid, nlayers=nlayers, dropout=dropout,
                               output_dim=self.config['output_dim'])
        
        # Initialize empty protein embeddings
        empty_pe = torch.zeros(145469, 5120)
        empty_pe.requires_grad = False
        model.pe_embedding = nn.Embedding.from_pretrained(empty_pe)
        
        # Load model weights
        model.load_state_dict(torch.load(str(self.model_path), map_location="cpu"), strict=True)
        
        # Load real token embeddings
        all_pe = self._load_uce_token_embeddings()
        if all_pe.shape[0] != 145469:
            all_pe.requires_grad = False
            model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
        
        # Store the model components for fine-tuning
        self.model = model.eval()
        self.model_pe_embedding = model.pe_embedding
        
        SCLLMOutput.status("UCE TransformerModel initialized successfully", 'loaded')
    
    def _load_uce_token_embeddings(self):
        """Load UCE token embeddings (same logic as _get_ESM2_embeddings)."""
        import torch
        
        # Load ESM2 embeddings and special tokens
        all_pe = torch.load(str(self.token_file))
        if all_pe.shape[0] == 143574:
            # For now, use zeros for chromosome tokens since we don't have access to the model yet
            # This will be properly handled when the model is used
            chrom_tokens = torch.zeros(1895, 5120)  # 1895 is the number of chromosome choices
            all_pe = torch.vstack((all_pe, chrom_tokens))
            all_pe.requires_grad = False
        
        return all_pe

    def get_embeddings(self, adata: AnnData, **kwargs) -> np.ndarray:
        """
        Extract cell embeddings using UCE model directly from memory.
        
        This method processes AnnData directly in memory without file I/O:
        1. Process adata in memory
        2. Generate protein embeddings and chromosome mappings
        3. Run UCE model inference
        4. Return embeddings (affected by fine-tuning if model was fine-tuned)
        
        Args:
            adata: Input AnnData object
            **kwargs: Additional parameters
            
        Returns:
            Cell embeddings as numpy array
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Process adata directly in memory
        try:
            # 检查模型是否已微调
            if hasattr(self, 'is_fine_tuned') and self.is_fine_tuned:
                SCLLMOutput.status("🎯 Extracting embeddings using fine-tuned UCE model", 'embedding')
                # 使用微调后的模型提取embeddings
                embeddings = self._extract_embeddings_from_finetuned_model(adata, **kwargs)
            else:
                SCLLMOutput.status("Extracting cell embeddings using UCE", 'embedding')
                # 使用原始预训练模型
                embeddings = self._run_uce_workflow_direct(adata, **kwargs)
                SCLLMOutput.status(f"Extracted embeddings: {embeddings.shape}", 'embedding')
            
            return embeddings
            
        except Exception as e:
            SCLLMOutput.status(f"UCE embedding extraction failed: {e}", 'error')
            raise RuntimeError(f"UCE embedding extraction failed: {e}")
    
    
    def _run_uce_workflow(self, adata_path: str, output_dir: str, **kwargs) -> np.ndarray:
        """
        Run the exact UCE workflow from eval_single_anndata.py.
        
        Args:
            adata_path: Path to temporary adata file
            output_dir: Output directory for UCE processing
            **kwargs: Additional parameters
            
        Returns:
            Cell embeddings as numpy array
        """
        from accelerate import Accelerator
        
        # Import UCE components
        try:
            from .UCE.evaluate import AnndataProcessor, run_eval, get_ESM2_embeddings
            from .UCE.eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator
            from .UCE.model import TransformerModel
        except ImportError:
            # Fallback for direct execution
            import sys
            uce_path = Path(__file__).parent / "UCE"
            if str(uce_path) not in sys.path:
                sys.path.insert(0, str(uce_path))
            from evaluate import AnndataProcessor, run_eval, get_ESM2_embeddings
            from eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator
            from model import TransformerModel
        
        # Create args object exactly like eval_single_anndata.py
        args = self._create_uce_args(adata_path, output_dir, **kwargs)
        
        # Create accelerator exactly like eval_single_anndata.py
        accelerator = Accelerator(project_dir=output_dir)
        
        # Step 1: Create processor and preprocess data
        processor = AnndataProcessor(args, accelerator)
        processor.preprocess_anndata()
        processor.generate_idxs()
        
        # Step 2: Load shapes_dict (needed for run_eval)
        with open(processor.shapes_dict_path, "rb") as f:
            shapes_dict = pickle.load(f)
        
        # Step 3: Run evaluation with custom implementation that returns embeddings
        embeddings = self._run_eval_custom(processor.adata, processor.name, 
                                         processor.pe_idx_path, processor.chroms_path,
                                         processor.starts_path, shapes_dict, 
                                         accelerator, args)
        
        return embeddings
    
    def _run_uce_workflow_direct(self, adata: AnnData, **kwargs) -> np.ndarray:
        """
        Run UCE workflow directly on memory adata object without file I/O.
        
        Args:
            adata: Input AnnData object
            **kwargs: Additional parameters
            
        Returns:
            Cell embeddings as numpy array
        """
        from accelerate import Accelerator
        
        # Import UCE components
        try:
            from .UCE.eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator
            from .UCE.model import TransformerModel
        except ImportError:
            # Fallback for direct execution
            import sys
            uce_path = Path(__file__).parent / "UCE"
            if str(uce_path) not in sys.path:
                sys.path.insert(0, str(uce_path))
            from eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator
            from model import TransformerModel
        
        # Create accelerator
        accelerator = Accelerator()
        
        # Step 1: Process adata directly in memory
        SCLLMOutput.status("Processing data in memory", 'embedding')
        processed_adata, pe_row_idxs, dataset_chroms, dataset_pos = self._process_adata_direct(adata, **kwargs)
        
        # Step 2: Create shapes dict
        shapes_dict = {"memory_adata": (processed_adata.n_obs, processed_adata.n_vars)}
        
        # Step 3: Run evaluation with direct data
        embeddings = self._run_eval_direct(
            processed_adata, pe_row_idxs, dataset_chroms, dataset_pos, shapes_dict, accelerator
        )
        
        return embeddings
    
    def _process_adata_direct(self, adata: AnnData, **kwargs) -> Tuple[AnnData, torch.Tensor, np.ndarray, np.ndarray]:
        """
        Process AnnData directly in memory to extract gene embeddings and chromosome information.
        
        Args:
            adata: Input AnnData object
            
        Returns:
            Tuple of (processed_adata, pe_row_idxs, dataset_chroms, dataset_pos)
        """
        # Make a copy and ensure dense matrix
        adata_processed = adata.copy()
        if issparse(adata_processed.X):
            adata_processed.X = adata_processed.X.toarray()
        
        # Apply basic filtering if requested
        filter_cells = kwargs.get('filter_cells', False)
        filter_genes = kwargs.get('filter_genes', False)
        
        if filter_cells:
            from omicverse.pp import filter_cells as _filter_cells
            min_genes = kwargs.get('min_genes', 25)
            _filter_cells(adata_processed, min_genes=min_genes)
        
        if filter_genes:
            from omicverse.pp import filter_genes as _filter_genes
            min_cells = kwargs.get('min_cells', 10)
            _filter_genes(adata_processed, min_cells=min_cells)
        
        # Load gene embeddings using our inline function
        try:
            from .UCE.data_proc.data_utils import load_gene_embeddings_adata_inline
        except ImportError:
            import sys
            uce_path = Path(__file__).parent / "UCE"
            if str(uce_path) not in sys.path:
                sys.path.insert(0, str(uce_path))
            from data_proc.data_utils import load_gene_embeddings_adata_inline
        
        # Process gene embeddings
        adata_processed, protein_embeddings = load_gene_embeddings_adata_inline(
            adata_processed, 
            species=[self.config['species']],
            embedding_model="ESM2",
            protein_embeddings_dir=str(self.protein_embeddings_dir)
        )
        
        # Generate protein embedding indices and chromosome information
        adata_processed, pe_row_idxs, dataset_chroms, dataset_pos = self._generate_indices_direct(adata_processed)
        
        return adata_processed, pe_row_idxs, dataset_chroms, dataset_pos
    
    def _generate_indices_direct(self, adata_processed: AnnData) -> Tuple[AnnData, torch.Tensor, np.ndarray, np.ndarray]:
        """
        Generate protein embedding indices and chromosome information directly from adata.
        
        Args:
            adata_processed: Processed AnnData object
            
        Returns:
            Tuple of (pe_row_idxs, dataset_chroms, dataset_pos)
        """
        # Import UCE data processing functions
        try:
            from .UCE.data_proc.data_utils import get_species_to_pe, get_spec_chrom_csv
        except ImportError:
            import sys
            uce_path = Path(__file__).parent / "UCE"
            if str(uce_path) not in sys.path:
                sys.path.insert(0, str(uce_path))
            from data_proc.data_utils import get_species_to_pe, get_spec_chrom_csv
        
        # Load species protein embeddings
        species_to_pe = get_species_to_pe(str(self.protein_embeddings_dir))
        
        # Load species offsets
        with open(self.offset_pkl_path, "rb") as f:
            species_to_offsets = pickle.load(f)
        
        # Load chromosome position information
        gene_to_chrom_pos = get_spec_chrom_csv(str(self.spec_chrom_csv_path))
        
        # Generate indices
        dataset_species = self.config['species']
        spec_pe_genes = list(species_to_pe[dataset_species].keys())
        offset = species_to_offsets[dataset_species]
        
        # Create protein embedding row indices with error handling
        valid_genes = []
        pe_indices = []
        missing_genes = []
        
        for gene in adata_processed.var_names:
            gene_upper = gene.upper()
            if gene_upper in spec_pe_genes:
                valid_genes.append(gene)
                pe_indices.append(spec_pe_genes.index(gene_upper) + offset)
            else:
                missing_genes.append(gene)
        
        if missing_genes:
            SCLLMOutput.status(f"Warning: {len(missing_genes)} genes not found in UCE protein embeddings", 'warning')
            if len(missing_genes) <= 10:
                SCLLMOutput.status(f"Missing genes: {missing_genes}", 'warning')
            else:
                SCLLMOutput.status(f"First 10 missing genes: {missing_genes[:10]}...", 'warning')
            
            # Filter adata to only include valid genes
            valid_gene_mask = [gene in valid_genes for gene in adata_processed.var_names]
            adata_processed = adata_processed[:, valid_gene_mask].copy()
            SCLLMOutput.status(f"Filtered data to {len(valid_genes)} valid genes", 'warning')
        
        pe_row_idxs = torch.tensor(pe_indices).long()
        
        # Create chromosome mappings with error handling
        spec_chrom = gene_to_chrom_pos[gene_to_chrom_pos["species"] == dataset_species].set_index("gene_symbol")
        
        # Filter genes that exist in chromosome mapping
        chrom_genes = []
        chrom_codes = []
        chrom_positions = []
        final_pe_indices = []
        
        for i, gene in enumerate(adata_processed.var_names):
            gene_upper = gene.upper()
            if gene_upper in spec_chrom.index:
                chrom_genes.append(gene)
                chrom_codes.append(spec_chrom.loc[gene_upper, "spec_chrom"])
                chrom_positions.append(spec_chrom.loc[gene_upper, "start"])
                final_pe_indices.append(pe_indices[i])
        
        if len(chrom_genes) < len(adata_processed.var_names):
            SCLLMOutput.status(f"Further filtered to {len(chrom_genes)} genes with chromosome info", 'warning')
            # Final filtering of adata
            final_gene_mask = [gene in chrom_genes for gene in adata_processed.var_names]
            adata_processed = adata_processed[:, final_gene_mask].copy()
        
        pe_row_idxs = torch.tensor(final_pe_indices).long()
        
        # Convert chromosome codes to categorical codes
        import pandas as pd
        chrom_cat = pd.Categorical(chrom_codes)
        dataset_chroms = chrom_cat.codes
        dataset_pos = np.array(chrom_positions)
        
        return adata_processed, pe_row_idxs, dataset_chroms, dataset_pos
    
    def _run_eval_direct(self, adata: AnnData, pe_row_idxs: torch.Tensor, 
                        dataset_chroms: np.ndarray, dataset_pos: np.ndarray,
                        shapes_dict: dict, accelerator) -> np.ndarray:
        """
        Run UCE model evaluation directly with in-memory data.
        
        Args:
            adata: Processed AnnData object
            pe_row_idxs: Protein embedding indices
            dataset_chroms: Chromosome codes
            dataset_pos: Genomic positions
            shapes_dict: Shape information dictionary
            accelerator: Accelerator object
            
        Returns:
            Cell embeddings as numpy array
        """
        # Import UCE components
        try:
            from .UCE.eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator
            from .UCE.model import TransformerModel
        except ImportError:
            import sys
            uce_path = Path(__file__).parent / "UCE"
            if str(uce_path) not in sys.path:
                sys.path.insert(0, str(uce_path))
            from eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator
            from model import TransformerModel
        
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm
        
        # Set random seeds for reproducibility (crucial for identical results)
        torch.manual_seed(23)
        np.random.seed(23)
        
        # Set deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        #### Set up the model exactly like evaluate.py ####
        token_dim = self.config['token_dim']
        emsize = 1280  # embedding dimension (fixed in UCE)
        d_hid = self.config['d_hid']
        nlayers = self.config['nlayers']
        nhead = 20  # fixed in UCE
        dropout = 0.05  # fixed in UCE
        
        # Create model exactly like UCE
        model = TransformerModel(token_dim=token_dim, d_model=emsize, nhead=nhead,
                               d_hid=d_hid, nlayers=nlayers, dropout=dropout,
                               output_dim=self.config['output_dim'])
        
        # Initialize empty protein embeddings
        empty_pe = torch.zeros(145469, 5120)
        empty_pe.requires_grad = False
        model.pe_embedding = nn.Embedding.from_pretrained(empty_pe)
        
        # Load model weights
        model.load_state_dict(torch.load(str(self.model_path), map_location="cpu"), strict=True)
        
        # Load real token embeddings using UCE's function
        all_pe = self._get_ESM2_embeddings_direct()
        if all_pe.shape[0] != 145469:
            all_pe.requires_grad = False
            model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
        
        SCLLMOutput.status("UCE model loaded and ready for inference", 'embedding')
        model = model.eval()
        
        # STORE THE LOADED MODEL AS CLASS ATTRIBUTE FOR FINE-TUNING
        self.model = model  # This is the actual UCE TransformerModel
        self.model_pe_embedding = model.pe_embedding  # Store the protein embedding layer
        self.accelerator = accelerator
        
        # Let accelerator handle the device placement
        model = accelerator.prepare(model)
        
        # After accelerator.prepare, ensure embedding weights are on the same device as model
        if hasattr(model, 'pe_embedding') or (hasattr(model, 'module') and hasattr(model.module, 'pe_embedding')):
            pe_embedding = model.pe_embedding if hasattr(model, 'pe_embedding') else model.module.pe_embedding
            model_device = next(model.parameters()).device
            if pe_embedding.weight.device != model_device:
                pe_embedding.weight.data = pe_embedding.weight.data.to(model_device)
        
        #### Create dataset directly from memory data ####
        # Instead of using file-based dataset, create data directly
        dataset_embeds = []
        
        # Convert adata to the format expected by UCE
        X_data = adata.X.astype(np.int64)  # UCE expects integer counts
        
        # Process data in batches
        batch_size = self.config['batch_size']
        n_cells = X_data.shape[0]
        
        # Create a proper UCE-style dataset
        SCLLMOutput.status("Creating UCE dataset in memory", 'embedding')
        dataset = self._create_uce_dataset_direct(
            adata, pe_row_idxs, dataset_chroms, dataset_pos, shapes_dict
        )
        
        # Create data loader with UCE's collator
        try:
            from .UCE.eval_data import MultiDatasetSentenceCollator
        except ImportError:
            import sys
            uce_path = Path(__file__).parent / "UCE"
            if str(uce_path) not in sys.path:
                sys.path.insert(0, str(uce_path))
            from eval_data import MultiDatasetSentenceCollator
        
        # Create args for the collator and multi-gpu check
        args = self._create_uce_args_direct()
        
        collator = MultiDatasetSentenceCollator(args)
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=collator, num_workers=0)
        dataloader = accelerator.prepare(dataloader)
        
        # Run inference using UCE's exact format
        pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process, desc="UCE inference")
        
        with torch.no_grad():
            for batch in pbar:
                batch_sentences, mask, idxs = batch[0], batch[1], batch[2]
                
                # Move tensors to correct device
                device = next(model.parameters()).device
                batch_sentences = batch_sentences.to(device)
                mask = mask.to(device)
                
                # Transpose to UCE's expected format: (seq_len, batch_size)
                batch_sentences = batch_sentences.permute(1, 0)
                
                # Apply protein embeddings
                if args.multi_gpu:
                    batch_sentences = model.module.pe_embedding(batch_sentences.long())
                else:
                    batch_sentences = model.pe_embedding(batch_sentences.long())
                    
                batch_sentences = nn.functional.normalize(batch_sentences, dim=2)
                
                # Forward pass with proper mask
                _, embedding = model.forward(batch_sentences, mask=mask)
                
                # Gather embeddings
                accelerator.wait_for_everyone()
                embeddings = accelerator.gather_for_metrics(embedding)
                if accelerator.is_main_process:
                    dataset_embeds.append(embeddings.detach().cpu().numpy())
        
        # Combine embeddings
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            final_embeddings = np.vstack(dataset_embeds)
            return final_embeddings
        else:
            return np.array([])  # Non-main process returns empty
    
    def _get_ESM2_embeddings_direct(self):
        """
        Load ESM2 embeddings directly (exact copy of UCE's get_ESM2_embeddings function).
        """
        # Load ESM2 embeddings and special tokens
        all_pe = torch.load(str(self.token_file))
        if all_pe.shape[0] == 143574:
            # CHROM_TENSORS should be loaded from the actual UCE model
            # This is a critical component that needs to match the trained model
            if hasattr(self.model, 'token_emb') and hasattr(self.model.token_emb, 'weight'):
                # Extract chromosome token embeddings from the model
                token_emb_size = self.model.token_emb.weight.shape[0]
                chrom_start_idx = self.config.get('CHROM_TOKEN_OFFSET', 5120)
                chrom_end_idx = min(chrom_start_idx + 1895, token_emb_size)
                actual_chrom_size = chrom_end_idx - chrom_start_idx
                if actual_chrom_size > 0:
                    CHROM_TENSORS = self.model.token_emb.weight[chrom_start_idx:chrom_end_idx].detach().clone()
                    SCLLMOutput.status(f"Using actual chromosome tokens from model: {CHROM_TENSORS.shape}", 'processing')
                    # 1895 is the total number of chromosome choices, hardcoded in UCE
                    all_pe = torch.vstack((all_pe, CHROM_TENSORS))
                    all_pe.requires_grad = False
                else:
                    raise ValueError(f"Invalid chromosome token range: {chrom_start_idx}-{chrom_end_idx}")
            else:
                raise ValueError("Cannot access chromosome token embeddings from UCE model - model not properly loaded")
        
        return all_pe
    
    def _create_uce_args_direct(self) -> Any:
        """Create args object for direct processing (no file paths needed)."""
        class Args:
            pass
        
        args = Args()
        
        # UCE configuration (no file paths)
        args.pad_length = self.config['pad_length']
        args.sample_size = self.config['sample_size']
        args.cls_token_idx = self.config['cls_token_idx']
        args.chrom_token_left_idx = self.config['chrom_token_left_idx'] 
        args.chrom_token_right_idx = self.config['chrom_token_right_idx']
        args.pad_token_idx = self.config['pad_token_idx']
        args.CHROM_TOKEN_OFFSET = self.config['CHROM_TOKEN_OFFSET']
        args.multi_gpu = self.config['multi_gpu']
        
        return args
    
    def _create_uce_dataset_direct(self, adata: AnnData, pe_row_idxs: torch.Tensor,
                                  dataset_chroms: np.ndarray, dataset_pos: np.ndarray,
                                  shapes_dict: dict):
        """Create a UCE-style dataset from memory data."""
        
        class MemoryUCEDataset:
            def __init__(self, adata, pe_row_idxs, dataset_chroms, dataset_pos, shapes_dict, args):
                self.adata = adata
                self.pe_row_idxs = pe_row_idxs
                self.dataset_chroms = dataset_chroms  
                self.dataset_pos = dataset_pos
                self.shapes_dict = shapes_dict
                self.args = args
                self.n_obs = adata.n_obs
                self.X_data = adata.X.astype(np.int64)  # UCE expects integer counts
                
                # Setup exactly like MultiDatasetSentences
                self.dataset_name = "memory_adata"
                self.num_cells = {self.dataset_name: self.n_obs}
                self.num_genes = {self.dataset_name: adata.n_vars}
                self.total_num_cells = self.n_obs
                
                # Create the mappings that UCE expects
                self.dataset_to_protein_embeddings = {self.dataset_name: pe_row_idxs}
                self.dataset_to_chroms = {self.dataset_name: dataset_chroms}
                self.dataset_to_starts = {self.dataset_name: dataset_pos}
            
            def __len__(self):
                return self.n_obs
                
            def __getitem__(self, idx):
                if isinstance(idx, int):
                    # Get the count data for this cell
                    counts = torch.tensor(self.X_data[idx]).unsqueeze(0)
                    weights = torch.log1p(counts)
                    weights = (weights / torch.sum(weights))
                    
                    # Use UCE's sample_cell_sentences function
                    try:
                        from .UCE.eval_data import sample_cell_sentences
                    except ImportError:
                        import sys
                        uce_path = Path(__file__).parent / "UCE"  # Fixed path
                        if str(uce_path) not in sys.path:
                            sys.path.insert(0, str(uce_path))
                        from eval_data import sample_cell_sentences
                    
                    batch_sentences, mask, seq_len, cell_sentences = \
                        sample_cell_sentences(counts, weights, self.dataset_name, self.args,
                                            dataset_to_protein_embeddings=self.dataset_to_protein_embeddings,
                                            dataset_to_chroms=self.dataset_to_chroms,
                                            dataset_to_starts=self.dataset_to_starts)
                    
                    return batch_sentences, mask, idx, seq_len, cell_sentences
                else:
                    raise NotImplementedError
        
        args = self._create_uce_args_direct()
        return MemoryUCEDataset(adata, pe_row_idxs, dataset_chroms, dataset_pos, shapes_dict, args)
    
    def _create_uce_args(self, adata_path: str, output_dir: str, **kwargs):
        """Create args object matching eval_single_anndata.py argument parser."""
        class Args:
            pass
        
        args = Args()
        
        # Required arguments
        args.adata_path = adata_path
        args.dir = output_dir
        args.model_loc = str(self.model_path)
        args.species = self.config['species']
        
        # UCE asset paths
        args.token_file = str(self.token_file)
        args.protein_embeddings_dir = str(self.protein_embeddings_dir)
        args.spec_chrom_csv_path = str(self.spec_chrom_csv_path)
        args.offset_pkl_path = str(self.offset_pkl_path)
        
        # Model parameters
        args.nlayers = self.config['nlayers']
        args.output_dim = self.config['output_dim']
        args.d_hid = self.config['d_hid']
        args.token_dim = self.config['token_dim']
        args.batch_size = self.config['batch_size']
        args.pad_length = self.config['pad_length']
        args.sample_size = self.config['sample_size']
        args.multi_gpu = self.config['multi_gpu']
        
        # Token indices
        args.pad_token_idx = self.config['pad_token_idx']
        args.chrom_token_left_idx = self.config['chrom_token_left_idx']
        args.chrom_token_right_idx = self.config['chrom_token_right_idx']
        args.cls_token_idx = self.config['cls_token_idx']
        args.CHROM_TOKEN_OFFSET = self.config['CHROM_TOKEN_OFFSET']
        
        # Other parameters
        args.filter = kwargs.get('filter', False)
        args.skip = kwargs.get('skip', False)
        args.CXG = kwargs.get('CXG', False)
        
        return args
    
    def _run_eval_custom(self, adata, name, pe_idx_path, chroms_path, starts_path, 
                        shapes_dict, accelerator, args) -> np.ndarray:
        """
        Custom implementation of run_eval that returns embeddings instead of writing files.
        
        This follows the exact logic of UCE's run_eval function.
        """
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm
        
        # Import UCE components
        try:
            from .UCE.eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator
            from .UCE.model import TransformerModel
        except ImportError:
            from eval_data import MultiDatasetSentences, MultiDatasetSentenceCollator
            from model import TransformerModel
        
        # Set random seeds for reproducibility (crucial for identical results)
        torch.manual_seed(23)
        np.random.seed(23)
        
        # Set deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        #### Set up the model exactly like evaluate.py ####
        token_dim = args.token_dim
        emsize = 1280  # embedding dimension (fixed in UCE)
        d_hid = args.d_hid
        nlayers = args.nlayers
        nhead = 20  # fixed in UCE
        dropout = 0.05  # fixed in UCE
        
        # Create model exactly like UCE
        model = TransformerModel(token_dim=token_dim, d_model=emsize, nhead=nhead,
                               d_hid=d_hid, nlayers=nlayers, dropout=dropout,
                               output_dim=args.output_dim)
        
        # Initialize empty protein embeddings
        empty_pe = torch.zeros(145469, 5120)
        empty_pe.requires_grad = False
        model.pe_embedding = nn.Embedding.from_pretrained(empty_pe)
        
        # Load model weights
        model.load_state_dict(torch.load(args.model_loc, map_location="cpu"), strict=True)
        
        # Load real token embeddings using UCE's function
        all_pe = self._get_ESM2_embeddings(args)
        if all_pe.shape[0] != 145469:
            all_pe.requires_grad = False
            model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
        
        SCLLMOutput.status(f"Model loaded from: {args.model_loc}", 'loaded')
        model = model.eval()
        model = accelerator.prepare(model)
        
        #### Run the model exactly like evaluate.py ####
        # Create dataset
        dataset = MultiDatasetSentences(sorted_dataset_names=[name],
                                      shapes_dict=shapes_dict,
                                      args=args, npzs_dir=args.dir,
                                      dataset_to_protein_embeddings_path=pe_idx_path,
                                      datasets_to_chroms_path=chroms_path,
                                      datasets_to_starts_path=starts_path)
        
        # Create data loader
        multi_dataset_sentence_collator = MultiDatasetSentenceCollator(args)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=multi_dataset_sentence_collator, num_workers=0)
        dataloader = accelerator.prepare(dataloader)
        
        # STORE THE LOADED MODEL AS CLASS ATTRIBUTE FOR FINE-TUNING
        self.model = model  # This is the actual UCE TransformerModel
        self.model_pe_embedding = model.pe_embedding  # Store the protein embedding layer
        self.accelerator = accelerator
        self.args = args

        # Run inference
        pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        dataset_embeds = []
        
        with torch.no_grad():
            for batch in pbar:
                batch_sentences, mask, idxs = batch[0], batch[1], batch[2]
                batch_sentences = batch_sentences.permute(1, 0)
                
                # Apply protein embeddings
                if args.multi_gpu:
                    batch_sentences = model.module.pe_embedding(batch_sentences.long())
                else:
                    batch_sentences = model.pe_embedding(batch_sentences.long())
                    
                batch_sentences = nn.functional.normalize(batch_sentences, dim=2)
                _, embedding = model.forward(batch_sentences, mask=mask)
                
                # Gather embeddings (exactly like UCE)
                accelerator.wait_for_everyone()
                embeddings = accelerator.gather_for_metrics((embedding))
                if accelerator.is_main_process:
                    dataset_embeds.append(embeddings.detach().cpu().numpy())
        
        # Combine embeddings
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            final_embeddings = np.vstack(dataset_embeds)
            return final_embeddings
        else:
            return np.array([])  # Non-main process returns empty
    
    def _get_ESM2_embeddings(self, args):
        """
        Exact copy of UCE's get_ESM2_embeddings function.
        """
        # Load ESM2 embeddings and special tokens
        all_pe = torch.load(args.token_file)
        if all_pe.shape[0] == 143574:
            # CHROM_TENSORS should be loaded from the actual UCE model
            if hasattr(self.model, 'token_emb') and hasattr(self.model.token_emb, 'weight'):
                # Extract chromosome token embeddings from the model
                token_emb_size = self.model.token_emb.weight.shape[0]
                chrom_start_idx = args.CHROM_TOKEN_OFFSET
                chrom_end_idx = min(chrom_start_idx + 1895, token_emb_size)
                actual_chrom_size = chrom_end_idx - chrom_start_idx
                if actual_chrom_size > 0:
                    CHROM_TENSORS = self.model.token_emb.weight[chrom_start_idx:chrom_end_idx].detach().clone()
                    SCLLMOutput.status(f"Using actual chromosome tokens: {CHROM_TENSORS.shape}", 'processing')
                    # 1895 is the total number of chromosome choices, hardcoded in UCE
                    all_pe = torch.vstack((all_pe, CHROM_TENSORS))
                    all_pe.requires_grad = False
                else:
                    raise ValueError(f"Invalid chromosome token range: {chrom_start_idx}-{chrom_end_idx}")
            else:
                raise ValueError("Cannot access chromosome token embeddings from UCE model - model not properly loaded")
        
        return all_pe
    
    def preprocess(self, adata: AnnData, **kwargs) -> AnnData:
        """
        Preprocess AnnData for UCE model.
        
        UCE handles preprocessing internally during the workflow,
        so this method performs minimal preprocessing.
        
        Args:
            adata: Input AnnData object
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Preprocessed AnnData object
        """
        SCLLMOutput.status("Preprocessing data for UCE", 'preprocessing')
        
        adata_processed = adata.copy()
        
        # Convert sparse matrix to dense if needed
        if issparse(adata_processed.X):
            adata_processed.X = adata_processed.X.toarray()
        
        # Basic filtering (optional, UCE can handle filtering internally)
        filter_cells = kwargs.get('filter_cells', False)
        filter_genes = kwargs.get('filter_genes', False)
        
        if filter_cells:
            min_genes = kwargs.get('min_genes', 200)
            from omicverse.pp import filter_cells as _filter_cells
            _filter_cells(adata_processed, min_genes=min_genes)
            SCLLMOutput.status(f"Filtered cells with < {min_genes} genes", 'preprocessing')
        
        if filter_genes:
            min_cells = kwargs.get('min_cells', 3)
            from omicverse.pp import filter_genes as _filter_genes
            _filter_genes(adata_processed, min_cells=min_cells)
            SCLLMOutput.status(f"Filtered genes expressed in < {min_cells} cells", 'preprocessing')
        
        SCLLMOutput.status(f"Preprocessed data: {adata_processed.n_obs} cells × {adata_processed.n_vars} genes", 'preprocessing')
        
        return adata_processed
    
    def _set_global_protein_embeddings_dir(self):
        """
        Create new_species_protein_embeddings.csv file if needed.
        The protein_embeddings_dir is now passed directly via parameters.
        """
        try:
            SCLLMOutput.status("=== UCE Configuration ===", 'loading')
            SCLLMOutput.status(f"✓ Protein embeddings directory: {self.protein_embeddings_dir}", 'loaded')
            
            # Create new_species_protein_embeddings.csv file if needed
            self._create_new_species_csv()
            
            SCLLMOutput.status("✓ UCE configuration completed successfully", 'loaded')
            
        except Exception as e:
            SCLLMOutput.status(f"✗ Could not configure UCE: {e}", 'warning')
            raise
    
    def _create_new_species_csv(self):
        """
        Create new_species_protein_embeddings.csv file in the output directory.
        This prevents errors from hardcoded CSV path in data_utils.py.
        """
        # The CSV file is used in data_utils.py line 258
        # We create an empty one to prevent file not found errors
        import pandas as pd
        
        # Create empty CSV with required columns
        empty_df = pd.DataFrame(columns=['species', 'path'])
        
        # We need to create this file in multiple potential locations
        # since different UCE functions might look for it in different places
        potential_dirs = [
            Path.cwd(),  # Current working directory
            self.protein_embeddings_dir.parent,  # Parent of protein embeddings dir
        ]
        
        for directory in potential_dirs:
            csv_path = directory / "new_species_protein_embeddings.csv"
            try:
                if not csv_path.exists():
                    csv_path.parent.mkdir(parents=True, exist_ok=True)
                    empty_df.to_csv(csv_path, index=False)
                    SCLLMOutput.status(f"✓ Created new_species CSV: {csv_path}", 'loaded')
                else:
                    SCLLMOutput.status(f"✓ Found existing new_species CSV: {csv_path}", 'loaded')
            except Exception as e:
                SCLLMOutput.status(f"⚠ Could not create CSV in {directory}: {e}", 'warning')
                continue  # Ignore errors for individual paths
    
    def predict(self, adata: AnnData, task: str = "embedding", **kwargs) -> Dict[str, Any]:
        """Make predictions using UCE model."""
        if task == "embedding":
            embeddings = self.get_embeddings(adata, **kwargs)
            result = {
                'embeddings': embeddings,
                'task': task,
                'model_name': self.model_name
            }
            
            # 如果模型已微调，添加微调信息
            if hasattr(self, 'is_fine_tuned') and self.is_fine_tuned:
                result['fine_tuned'] = True
                result['note'] = 'Embeddings generated using fine-tuned UCE model'
                
            return result
            
        elif task == "annotation" and hasattr(self, 'is_fine_tuned') and self.is_fine_tuned:
            # 如果是注释任务且模型已微调，使用分类头
            predicted_celltypes = self.predict_celltype(adata, **kwargs)
            return {
                'predictions': predicted_celltypes,
                'task': task,
                'model_name': self.model_name,
                'fine_tuned': True
            }
        else:
            available_tasks = ["embedding"]
            if hasattr(self, 'is_fine_tuned') and self.is_fine_tuned:
                available_tasks.append("annotation")
            raise ValueError(f"Task '{task}' not supported. Available tasks: {available_tasks}")
    
    def fine_tune(self, train_adata: AnnData, valid_adata: Optional[AnnData] = None,
                  task: str = "annotation", **kwargs) -> Dict[str, Any]:
        """
        Fine-tune the UCE model for downstream tasks.
        
        Following scGPT's approach, this method creates a complete fine-tunable UCE model
        with classification head and trains the entire model end-to-end.
        
        Args:
            train_adata: Training data with labels in .obs['celltype'] 
            valid_adata: Validation data (optional)
            task: Task type (currently supports 'annotation')
            **kwargs: Training parameters
                - epochs: Number of training epochs (default: 10)
                - batch_size: Batch size (default: 32) 
                - lr: Learning rate (default: 1e-4)
                - freeze_backbone: Whether to freeze UCE encoder (default: False)
                
        Returns:
            Training results and metrics
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if task != "annotation":
            raise ValueError("Currently only 'annotation' task is supported for fine-tuning")
            
        if 'celltype' not in train_adata.obs:
            raise ValueError("train_adata must have 'celltype' column in .obs")
        
        # 根据freeze_backbone参数显示正确的训练模式
        freeze_backbone = kwargs.get('freeze_backbone', False)
        if freeze_backbone:
            SCLLMOutput.status("🎯 Starting UCE linear probing fine-tuning for cell type annotation", 'fine_tuning')
        else:
            SCLLMOutput.status("🎯 Starting UCE end-to-end fine-tuning for cell type annotation", 'fine_tuning')
        
        # Get training parameters
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 32)
        lr = kwargs.get('lr', 1e-4)  # Lower lr for full model training
        freeze_backbone = kwargs.get('freeze_backbone', False)
        
        # Prepare cell type mapping
        unique_celltypes = train_adata.obs['celltype'].astype('category').cat.categories
        celltype_to_id = {ct: i for i, ct in enumerate(unique_celltypes)}
        id_to_celltype = {i: ct for i, ct in enumerate(unique_celltypes)}
        n_classes = len(unique_celltypes)
        
        SCLLMOutput.status(f"Found {n_classes} cell types", 'info')
        
        # Add celltype_id to data
        train_adata.obs['celltype_id'] = train_adata.obs['celltype'].map(celltype_to_id)
        if valid_adata is not None:
            if 'celltype' not in valid_adata.obs:
                raise ValueError("valid_adata must have 'celltype' column in .obs")
            valid_adata.obs['celltype_id'] = valid_adata.obs['celltype'].map(celltype_to_id)
        
        # Preprocess data for UCE format
        train_processed = self.preprocess(train_adata, **kwargs)
        valid_processed = self.preprocess(valid_adata, **kwargs) if valid_adata is not None else None

        # Create fine-tuning model that combines UCE backbone with classifier
        if freeze_backbone:
            # 线性探测模式：预计算embeddings，只训练分类头
            finetune_model, train_embeddings, valid_embeddings = self._create_linear_probing_model(
                n_classes, train_processed, valid_processed
            )
        else:
            # 端到端模式：完整的可微分训练
            finetune_model = self._create_uce_finetune_model(n_classes, freeze_backbone)
            train_embeddings = None
            valid_embeddings = None
        
        # Prepare training datasets
        if freeze_backbone:
            # 线性探测：使用预计算的embeddings
            train_dataset = self._prepare_linear_probing_dataset(train_embeddings, train_processed.obs['celltype_id'])
            valid_dataset = self._prepare_linear_probing_dataset(valid_embeddings, valid_processed.obs['celltype_id']) if valid_processed else None
        else:
            # 端到端：使用原始数据
            train_dataset = self._prepare_finetune_dataset(train_processed)
            valid_dataset = self._prepare_finetune_dataset(valid_processed) if valid_processed else None
        
        # Setup training  
        from torch.utils.data import DataLoader
        
        # 为端到端训练定义collate函数
        def uce_collate_fn(batch):
            """自定义collate函数，处理UCE的(batch_sentences, mask)格式"""
            batch_data_list = []
            labels_list = []
            
            for (batch_sentences, mask), label in batch:
                batch_data_list.append((batch_sentences, mask))
                labels_list.append(label)
            
            # 将多个(batch_sentences, mask)组合成batch
            if len(batch_data_list) > 0:
                # Stack batch_sentences and masks
                all_batch_sentences = torch.stack([item[0] for item in batch_data_list])
                all_masks = torch.stack([item[1] for item in batch_data_list])
                batched_data = (all_batch_sentences, all_masks)
                batched_labels = torch.stack(labels_list)
                return batched_data, batched_labels
            else:
                return batch_data_list, labels_list
        
        # 根据训练模式选择collate函数
        if freeze_backbone:
            # 线性探测：使用默认collate（处理embeddings）
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False) if valid_dataset else None
        else:
            # 端到端：使用自定义collate（处理UCE数据格式）
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=uce_collate_fn)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=uce_collate_fn) if valid_dataset else None
        
        optimizer = torch.optim.Adam(finetune_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        if freeze_backbone:
            # 线性探测：使用简化的训练循环
            training_results = self._train_linear_probing_model(
                finetune_model, train_loader, valid_loader,
                optimizer, scheduler, criterion, epochs
            )
        else:
            # 端到端：使用完整的UCE训练循环
            training_results = self._train_uce_finetune_model(
                finetune_model, train_loader, valid_loader, 
                optimizer, scheduler, criterion, epochs
            )
        
        # Store the trained components
        self.finetune_model = finetune_model
        self.celltype_to_id = celltype_to_id
        self.id_to_celltype = id_to_celltype
        self.is_fine_tuned = True
        
        SCLLMOutput.status("✅ UCE fine-tuning completed successfully", 'fine_tuning')
        
        return {
            'task': task,
            'model_name': self.model_name,
            'n_classes': n_classes,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'freeze_backbone': freeze_backbone,
            'method': 'end_to_end_training',
            **training_results
        }
    
    # Fine-tuning helper methods - now using real UCE embeddings instead of random data
    
    def _prepare_labels(self, adata: AnnData) -> Tuple[np.ndarray, Any, int]:
        """Prepare and encode labels for training."""
        from sklearn.preprocessing import LabelEncoder
        
        raw_labels = adata.obs['celltype'].values
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(raw_labels)
        n_classes = len(label_encoder.classes_)
        
        return encoded_labels, label_encoder, n_classes
    
    def _encode_labels(self, adata: AnnData, label_encoder) -> np.ndarray:
        """Encode labels using existing label encoder."""
        raw_labels = adata.obs['celltype'].values
        try:
            encoded_labels = label_encoder.transform(raw_labels)
        except ValueError as e:
            # Handle unknown labels in validation set
            SCLLMOutput.status(f"Warning: Unknown labels in validation set: {e}", 'warning')
            # Map unknown labels to -1 and filter them out during validation
            encoded_labels = []
            for label in raw_labels:
                try:
                    encoded_labels.append(label_encoder.transform([label])[0])
                except ValueError:
                    encoded_labels.append(-1)  # Unknown label marker
            encoded_labels = np.array(encoded_labels)
            
        return encoded_labels
    
    def _create_classification_head(self, n_classes: int) -> nn.Module:
        """Create a linear classification head."""
        import torch.nn as nn
        
        # UCE output dimension is typically 1280
        input_dim = self.config['output_dim']
        
        classification_head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(256, n_classes)
        )
        
        # Move to appropriate device
        if self.device and self.device != 'cpu':
            classification_head = classification_head.to(self.device)
            
        return classification_head
    
    
    
    
    def _train_classification_head(self, model, train_embeddings: np.ndarray, train_labels: np.ndarray,
                                 valid_embeddings: Optional[np.ndarray], valid_labels: Optional[np.ndarray],
                                 epochs: int, batch_size: int, lr: float) -> Dict[str, Any]:
        """Train the classification head."""
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import accuracy_score, f1_score
        
        # Debug info
        SCLLMOutput.status(f"Training data: {train_embeddings.shape[0]} samples, {train_embeddings.shape[1]} features", 'fine_tuning')
        SCLLMOutput.status(f"Label distribution: {dict(zip(*np.unique(train_labels, return_counts=True)))}", 'fine_tuning')
        SCLLMOutput.status(f"Embedding stats - mean: {train_embeddings.mean():.4f}, std: {train_embeddings.std():.4f}", 'fine_tuning')
        
        # Convert to tensors
        train_X = torch.FloatTensor(train_embeddings)
        train_y = torch.LongTensor(train_labels)
        
        if self.device and self.device != 'cpu':
            train_X = train_X.to(self.device)
            train_y = train_y.to(self.device)
        
        # Create data loader
        train_dataset = TensorDataset(train_X, train_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        train_losses = []
        train_accuracies = []
        valid_accuracies = []
        
        from tqdm.auto import tqdm
        
        SCLLMOutput.status(f"Starting training with {len(train_loader)} batches per epoch", 'fine_tuning')
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            all_preds = []
            all_labels = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch_idx, (batch_X, batch_y) in enumerate(pbar):
                optimizer.zero_grad()
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    SCLLMOutput.status(f"NaN loss detected at epoch {epoch+1}, batch {batch_idx}", 'warning')
                    break
                    
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Track predictions for accuracy
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Calculate training metrics
            avg_loss = epoch_loss / len(train_loader)
            train_acc = accuracy_score(all_labels, all_preds)
            
            train_losses.append(avg_loss)
            train_accuracies.append(train_acc)
            
            # Validation
            valid_acc = None
            if valid_embeddings is not None and valid_labels is not None:
                valid_acc = self._validate_classification_head(
                    model, valid_embeddings, valid_labels, batch_size
                )
                valid_accuracies.append(valid_acc)
                
            # Progress update
            metrics_str = f"Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}"
            if valid_acc is not None:
                metrics_str += f", Valid Acc: {valid_acc:.4f}"
            SCLLMOutput.status(f"Epoch {epoch+1}/{epochs} - {metrics_str}", 'fine_tuning')
            
            # Early stopping if training accuracy is very low
            if epoch >= 2 and train_acc < 0.1:
                SCLLMOutput.status("Warning: Very low training accuracy. Check your data and labels.", 'warning')
        
        final_train_acc = train_accuracies[-1] if train_accuracies else 0.0
        final_valid_acc = valid_accuracies[-1] if valid_accuracies else None
        
        SCLLMOutput.status(f"Final training accuracy: {final_train_acc:.4f}", 'fine_tuning')
        if final_valid_acc is not None:
            SCLLMOutput.status(f"Final validation accuracy: {final_valid_acc:.4f}", 'fine_tuning')
        
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'valid_accuracies': valid_accuracies,
            'final_train_acc': final_train_acc,
            'final_valid_acc': final_valid_acc
        }
    
    def _validate_classification_head(self, model, valid_embeddings: np.ndarray, 
                                    valid_labels: np.ndarray, batch_size: int) -> float:
        """Validate the classification head."""
        from sklearn.metrics import accuracy_score
        from torch.utils.data import DataLoader, TensorDataset
        
        # Filter out unknown labels (-1)
        valid_mask = valid_labels != -1
        if not valid_mask.any():
            return 0.0
            
        valid_embeddings = valid_embeddings[valid_mask]
        valid_labels = valid_labels[valid_mask]
        
        # Convert to tensors
        valid_X = torch.FloatTensor(valid_embeddings)
        valid_y = torch.LongTensor(valid_labels)
        
        if self.device and self.device != 'cpu':
            valid_X = valid_X.to(self.device)
            valid_y = valid_y.to(self.device)
        
        # Create data loader
        valid_dataset = TensorDataset(valid_X, valid_y)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in valid_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        return accuracy_score(all_labels, all_preds)
    
    def predict_celltype(self, adata: AnnData, **kwargs) -> np.ndarray:
        """
        Predict cell types using fine-tuned UCE model.
        
        Args:
            adata: Input AnnData object
            **kwargs: Additional parameters
            
        Returns:
            Predicted cell type labels
        """
        if not hasattr(self, 'finetune_model') or not hasattr(self, 'celltype_to_id'):
            raise ValueError("Model not fine-tuned. Call fine_tune() first.")
        
        SCLLMOutput.status("Predicting cell types using fine-tuned UCE model", 'predicting')
        
        # 检测微调模型类型
        is_linear_probing = 'UCELinearProbingModel' in str(type(self.finetune_model))
        
        if is_linear_probing:
            # 线性探测模型：直接使用预计算的embeddings，避免重复处理原始数据
            SCLLMOutput.status("Using linear probing model for prediction", 'predicting')
            SCLLMOutput.status("Computing embeddings for linear probing prediction...", 'embedding')
            
            # 对于线性探测，UCE backbone权重冻结未改变，直接使用预训练模型
            # 跳过额外的预处理，直接使用原始数据
            SCLLMOutput.status("Using original UCE model for embeddings (backbone frozen)", 'embedding')
            embeddings = self._run_uce_workflow_direct(adata, **kwargs)
            SCLLMOutput.status(f"Generated embeddings: {embeddings.shape}", 'embedding')
            
            # 使用embeddings直接预测
            import torch
            X = torch.FloatTensor(embeddings)
            if self.device and self.device != 'cpu':
                X = X.to(self.device)
            
            self.finetune_model.eval()
            all_predictions = []
            
            # 分批处理以节省内存
            batch_size = 1000
            with torch.no_grad():
                for i in range(0, X.shape[0], batch_size):
                    batch_X = X[i:i+batch_size]
                    # 直接通过分类头预测（跳过UCE backbone）
                    logits = self.finetune_model.classifier(batch_X)
                    _, predicted = torch.max(logits.data, 1)
                    all_predictions.extend(predicted.cpu().numpy())
            
            # 转换预测结果为cell type名称
            predicted_celltypes = [self.id_to_celltype[pred_id] for pred_id in all_predictions]
            SCLLMOutput.status(f"Predicted cell types for {len(predicted_celltypes)} cells", 'predicting')
            return np.array(predicted_celltypes)
            
        else:
            # 端到端模型：需要完整的数据处理流程
            SCLLMOutput.status("Using end-to-end model for prediction", 'predicting')
            # 端到端模式需要预处理
            adata_processed = self.preprocess(adata, **kwargs)
            predict_dataset = self._prepare_finetune_dataset(adata_processed)
            
            from torch.utils.data import DataLoader
            
            # 定义自定义collate函数处理UCE数据格式
            def uce_collate_fn(batch):
                """自定义collate函数，处理UCE的(batch_sentences, mask)格式"""
                batch_data_list = []
                labels_list = []
                
                for (batch_sentences, mask), label in batch:
                    batch_data_list.append((batch_sentences, mask))
                    labels_list.append(label)
                
                # 将多个(batch_sentences, mask)组合成batch
                if len(batch_data_list) > 0:
                    # Stack batch_sentences and masks
                    all_batch_sentences = torch.stack([item[0] for item in batch_data_list])
                    all_masks = torch.stack([item[1] for item in batch_data_list])
                    batched_data = (all_batch_sentences, all_masks)
                    batched_labels = torch.stack(labels_list)
                    return batched_data, batched_labels
                else:
                    return batch_data_list, labels_list
            
            predict_loader = DataLoader(predict_dataset, batch_size=32, shuffle=False, collate_fn=uce_collate_fn)
            
            # Predict using fine-tuned model
            self.finetune_model.eval()
            all_predictions = []
            
            with torch.no_grad():
                for batch_data, _ in predict_loader:
                    # Move to device with proper handling of different data types
                    if self.device and self.device != 'cpu':
                        if isinstance(batch_data, tuple) and len(batch_data) == 2:
                            # Format: (batch_sentences, mask)
                            batch_sentences, mask = batch_data
                            if hasattr(batch_sentences, 'to'):
                                batch_sentences = batch_sentences.to(self.device)
                            if hasattr(mask, 'to'):
                                mask = mask.to(self.device)
                            batch_data = (batch_sentences, mask)
                        elif isinstance(batch_data, list):
                            # Handle list of data - convert to proper format
                            SCLLMOutput.status("Converting list batch_data to tensor format", 'preprocessing')
                            # This shouldn't happen normally, but as a safety check
                            if len(batch_data) == 2 and hasattr(batch_data[0], 'to'):
                                batch_data = (batch_data[0].to(self.device), batch_data[1].to(self.device))
                            else:
                                raise ValueError(f"Unexpected batch_data format: {type(batch_data)}, length: {len(batch_data) if hasattr(batch_data, '__len__') else 'unknown'}")
                        elif hasattr(batch_data, 'to'):
                            # Single tensor
                            batch_data = batch_data.to(self.device)
                        else:
                            SCLLMOutput.status(f"Warning: Cannot move batch_data to device, type: {type(batch_data)}", 'warning')
                    
                    # Forward pass through fine-tuned UCE model
                    logits = self.finetune_model(batch_data)
                    _, predicted = torch.max(logits.data, 1)
                    all_predictions.extend(predicted.cpu().numpy())
            
            # Convert predictions to cell type names
            predicted_celltypes = [self.id_to_celltype[pred_id] for pred_id in all_predictions]
            SCLLMOutput.status(f"Predicted cell types for {len(predicted_celltypes)} cells", 'predicting')
            return np.array(predicted_celltypes)
    
    def integrate(self, adata: AnnData, batch_key: str = "batch", 
                  correction_method: str = "mnn", **kwargs) -> Dict[str, Any]:
        """
        Perform batch integration using UCE embeddings.
        
        Args:
            adata: AnnData object with batch information
            batch_key: Column name for batch labels in adata.obs
            correction_method: Batch correction method ('harmony', 'mnn', 'center_scale', 'none')
            **kwargs: Additional parameters for embedding generation
            
        Returns:
            Dictionary with integration results including corrected embeddings
        """
        # 检查模型是否已微调
        if hasattr(self, 'is_fine_tuned') and self.is_fine_tuned:
            SCLLMOutput.status(f"🔗🎯 Performing batch integration using fine-tuned UCE for {adata.n_obs} cells", 'integration')
        else:
            SCLLMOutput.status(f"🔗 Performing batch integration for {adata.n_obs} cells", 'integration')
        
        if batch_key not in adata.obs:
            raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")
        
        # Extract embeddings for integration
        SCLLMOutput.status("Extracting embeddings for integration", 'integration')
        
        # 对于线性探测模式，使用原始UCE模型（backbone权重未改变）
        if hasattr(self, 'is_fine_tuned') and self.is_fine_tuned:
            is_linear_probing = hasattr(self, 'finetune_model') and 'UCELinearProbingModel' in str(type(self.finetune_model))
            
            if is_linear_probing:
                SCLLMOutput.status("Using original UCE model for integration (backbone frozen)", 'integration')
                # 跳过预处理，直接使用原始数据避免重复处理
                embeddings = self._run_uce_workflow_direct(adata, **kwargs)
            else:
                SCLLMOutput.status("Using fine-tuned UCE model for integration", 'integration')
                embeddings = self.get_embeddings(adata, **kwargs)
        else:
            embeddings = self.get_embeddings(adata, **kwargs)
        
        # Apply batch correction
        try:
            from tqdm.auto import tqdm
            with tqdm(total=2, desc="Batch integration", ncols=100) as pbar:
                pbar.set_description(f"Applying {correction_method} correction...")
                integrated_embeddings = self._apply_batch_correction(
                    embeddings, adata.obs[batch_key], correction_method
                )
                pbar.update(1)
                pbar.set_description("Finalizing integration...")
                pbar.update(1)
            
            SCLLMOutput.status(f"Integration completed with {correction_method} correction", 'integration')
            
            return {
                'embeddings': integrated_embeddings,
                'original_embeddings': embeddings,
                'batch_key': batch_key,
                'correction_method': correction_method,
                'model_name': self.model_name,
                'task': 'integration'
            }
            
        except Exception as e:
            SCLLMOutput.status(f"Integration failed: {e}", 'error')
            raise RuntimeError(f"Batch integration failed: {e}")
    
    def _apply_batch_correction(self, embeddings: np.ndarray, batch_labels: pd.Series, method: str) -> np.ndarray:
        """Apply batch correction to embeddings."""
        import warnings
        
        if method == 'harmony':
            try:
                import harmonypy as hm
                SCLLMOutput.status("Applying Harmony correction", 'integration')
                harmony_out = hm.run_harmony(embeddings.T, batch_labels, max_iter_harmony=20)  
                return harmony_out.Z_corr.T
            except ImportError:
                warnings.warn("harmonypy not available, using MNN correction")
                method = 'mnn'
        
        if method == 'mnn':
            return self._apply_mnn_correction(embeddings, batch_labels)
        
        elif method == 'center_scale':
            return self._apply_center_scale_correction(embeddings, batch_labels)
        
        elif method == 'none':
            return embeddings
        
        else:
            warnings.warn(f"Unknown correction method {method}, using MNN correction")
            return self._apply_mnn_correction(embeddings, batch_labels)
    
    def _apply_mnn_correction(self, embeddings: np.ndarray, batch_labels: pd.Series) -> np.ndarray:
        """Apply MNN (Mutual Nearest Neighbors) correction - same as CellPLM implementation."""
        SCLLMOutput.status("Applying MNN correction", 'integration')
        
        try:
            from sklearn.neighbors import NearestNeighbors
            
            corrected = embeddings.copy()
            unique_batches = batch_labels.unique()
            
            if len(unique_batches) < 2:
                SCLLMOutput.status("Only one batch found, no correction needed", 'integration')
                return embeddings
            
            # Simple MNN-style correction between consecutive batches (same as CellPLM)
            for i in range(len(unique_batches) - 1):
                batch1_name = unique_batches[i]
                batch2_name = unique_batches[i + 1] if i + 1 < len(unique_batches) else unique_batches[0]
                
                batch1_mask = batch_labels == batch1_name
                batch2_mask = batch_labels == batch2_name
                
                batch1_data = corrected[batch1_mask]
                batch2_data = corrected[batch2_mask]
                
                if batch1_data.shape[0] > 5 and batch2_data.shape[0] > 5:
                    # Find mutual nearest neighbors (same algorithm as CellPLM)
                    k = min(5, min(batch1_data.shape[0], batch2_data.shape[0]) // 2)
                    
                    # Find nearest neighbors from batch2 to batch1
                    nn1 = NearestNeighbors(n_neighbors=k).fit(batch1_data)
                    distances1, indices1 = nn1.kneighbors(batch2_data)
                    
                    # Find nearest neighbors from batch1 to batch2  
                    nn2 = NearestNeighbors(n_neighbors=k).fit(batch2_data)
                    distances2, indices2 = nn2.kneighbors(batch1_data)
                    
                    # Apply simple correction by moving batches closer (same as CellPLM)
                    batch1_centroid = batch1_data.mean(axis=0)
                    batch2_centroid = batch2_data.mean(axis=0)
                    correction_vector = (batch1_centroid - batch2_centroid) * 0.5
                    
                    corrected[batch2_mask] += correction_vector
            
            SCLLMOutput.status(f"MNN correction applied to {len(unique_batches)} batches", 'integration')
            return corrected
            
        except Exception as e:
            SCLLMOutput.status(f"MNN correction failed: {e}, using center_scale correction", 'warning')
            return self._apply_center_scale_correction(embeddings, batch_labels)
    
    def _apply_center_scale_correction(self, embeddings: np.ndarray, batch_labels: pd.Series) -> np.ndarray:
        """Apply center and scale batch correction - same as CellPLM implementation."""
        SCLLMOutput.status("Applying center and scale correction", 'integration')
        
        try:
            corrected = embeddings.copy()
            
            # Calculate global statistics (same as CellPLM)
            global_mean = corrected.mean(axis=0)
            global_std = corrected.std(axis=0) + 1e-8
            
            # Correct each batch (same as CellPLM)
            for batch_name in batch_labels.unique():
                batch_mask = batch_labels == batch_name
                batch_data = corrected[batch_mask]
                
                if batch_data.shape[0] > 1:
                    # Calculate batch statistics
                    batch_mean = batch_data.mean(axis=0)
                    batch_std = batch_data.std(axis=0) + 1e-8
                    
                    # Center and scale to global statistics
                    corrected[batch_mask] = (batch_data - batch_mean) / batch_std * global_std + global_mean
            
            SCLLMOutput.status(f"Center-scale correction applied to {len(batch_labels.unique())} batches", 'integration')
            return corrected
            
        except Exception as e:
            SCLLMOutput.status(f"Center-scale correction failed: {e}, returning original embeddings", 'warning')
            return embeddings
    
    def _extract_embeddings_from_finetuned_model(self, adata: AnnData, **kwargs) -> np.ndarray:
        """
        使用微调后的UCE模型提取cell embeddings。
        
        Args:
            adata: Input AnnData object
            **kwargs: Additional parameters
            
        Returns:
            Cell embeddings from fine-tuned UCE backbone
        """
        if not hasattr(self, 'finetune_model') or self.finetune_model is None:
            SCLLMOutput.status("Warning: Fine-tuned model not found, using original model", 'warning')
            return self._run_uce_workflow_direct(adata, **kwargs)
        
        # 检查微调模型类型
        is_linear_probing = 'UCELinearProbingModel' in str(type(self.finetune_model))
        
        if is_linear_probing:
            SCLLMOutput.status("Extracting embeddings from linear probing model (frozen UCE backbone)", 'embedding')
        else:
            SCLLMOutput.status("Extracting embeddings from end-to-end fine-tuned UCE model", 'embedding')
        
        # 预处理数据
        adata_processed = self.preprocess(adata, **kwargs)
        
        # 准备dataset（复用微调时的数据准备逻辑）
        dataset = self._prepare_finetune_dataset(adata_processed)
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # 使用微调后的UCE backbone提取embeddings
        self.finetune_model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for batch_data, _ in dataloader:
                # Move to device
                if self.device and self.device != 'cpu':
                    batch_sentences, mask = batch_data
                    batch_sentences = batch_sentences.to(self.device)
                    mask = mask.to(self.device)
                    batch_data = (batch_sentences, mask)
                
                # 通过UCE backbone获取cell embeddings（不通过分类头）
                batch_sentences, mask = batch_data
                
                # 检查模型是否有pe_embedding和uce_backbone属性
                if not hasattr(self.finetune_model, 'pe_embedding') or not hasattr(self.finetune_model, 'uce_backbone'):
                    raise AttributeError(f"Fine-tuned model {type(self.finetune_model)} missing pe_embedding or uce_backbone")
                
                # Forward pass through UCE backbone only
                batch_sentences = batch_sentences.permute(1, 0)
                batch_sentences = self.finetune_model.pe_embedding(batch_sentences.long())
                batch_sentences = torch.nn.functional.normalize(batch_sentences, dim=2)
                
                # Forward through UCE transformer to get cell embeddings
                _, cell_embeddings = self.finetune_model.uce_backbone.forward(batch_sentences, mask)
                
                all_embeddings.append(cell_embeddings.cpu().numpy())
        
        # 合并所有embeddings
        final_embeddings = np.vstack(all_embeddings)
        
        SCLLMOutput.status(f"Extracted embeddings from fine-tuned UCE: {final_embeddings.shape}", 'embedding')
        return final_embeddings

    def _create_linear_probing_model(self, n_classes: int, train_adata, valid_adata=None):
        """创建线性探测模型，预计算embeddings."""
        import torch.nn as nn
        
        SCLLMOutput.status("Pre-computing UCE embeddings for linear probing...", 'preprocessing')
        
        # 预计算训练数据的embeddings
        train_embeddings = self.get_embeddings(train_adata)
        valid_embeddings = self.get_embeddings(valid_adata) if valid_adata is not None else None
        
        SCLLMOutput.status(f"Pre-computed embeddings: train={train_embeddings.shape}", 'preprocessing')
        if valid_embeddings is not None:
            SCLLMOutput.status(f"Pre-computed embeddings: valid={valid_embeddings.shape}", 'preprocessing')
        
        # 创建完整的线性探测模型，包含UCE backbone + 分类头
        class UCELinearProbingModel(nn.Module):
            def __init__(self, uce_model, input_dim, n_classes):
                super().__init__()
                self.uce_model = uce_model
                # Store the UCE backbone for consistent interface
                self.uce_backbone = uce_model.model
                self.pe_embedding = uce_model.model_pe_embedding
                
                # 简单分类头（只在线性探测中训练）
                self.classifier = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, n_classes)
                )
                
                # 冻结UCE backbone
                for param in self.uce_backbone.parameters():
                    param.requires_grad = False
                for param in self.pe_embedding.parameters():
                    param.requires_grad = False
            
            def forward(self, batch_data):
                """处理两种输入：batch_data (原始数据) 或 embeddings (预计算)"""
                if isinstance(batch_data, tuple) and len(batch_data) == 2:
                    # 原始数据格式用于预测：(batch_sentences, mask)
                    batch_sentences, mask = batch_data
                    with torch.no_grad():
                        batch_sentences = batch_sentences.permute(1, 0)
                        batch_sentences = self.pe_embedding(batch_sentences.long())
                        batch_sentences = nn.functional.normalize(batch_sentences, dim=2)
                        _, cell_embedding = self.uce_backbone.forward(batch_sentences, mask)
                    return self.classifier(cell_embedding)
                else:
                    # 预计算embeddings格式用于训练
                    return self.classifier(batch_data)
        
        # 创建模型
        input_dim = train_embeddings.shape[1]  # UCE embedding dimension
        model = UCELinearProbingModel(self, input_dim, n_classes)
        model.to(self.device)
        
        return model, train_embeddings, valid_embeddings

    def _prepare_linear_probing_dataset(self, embeddings, labels):
        """准备线性探测数据集，使用预计算的embeddings."""
        from torch.utils.data import Dataset
        import torch
        
        class LinearProbingDataset(Dataset):
            def __init__(self, embeddings, labels):
                self.embeddings = torch.FloatTensor(embeddings)
                self.labels = torch.LongTensor(labels.values)
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                return self.embeddings[idx], self.labels[idx]
        
        return LinearProbingDataset(embeddings, labels)

    def _train_linear_probing_model(self, model, train_loader, valid_loader, optimizer, scheduler, criterion, epochs):
        """训练线性探测模型（仅分类头，使用预计算embeddings）."""
        import torch
        from sklearn.metrics import accuracy_score
        
        training_results = {
            'epoch_losses': [],
            'epoch_accuracies': [],
            'best_accuracy': 0.0,
            'best_model_state': None
        }
        
        SCLLMOutput.status(f"Starting linear probing training for {epochs} epochs", 'training')
        
        # Create progress bar for epochs
        from tqdm.auto import tqdm
        epoch_pbar = tqdm(range(epochs), desc="Linear Probing", 
                         bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        
        for epoch in epoch_pbar:
            # Training phase
            model.train()
            epoch_loss = 0.0
            all_preds = []
            all_labels = []
            
            for batch_embeddings, batch_labels in train_loader:
                if self.device and self.device != 'cpu':
                    batch_embeddings = batch_embeddings.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                
                # 简单的前向传播：直接分类预计算的embeddings
                outputs = model(batch_embeddings)
                loss = criterion(outputs, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
            
            # Calculate training metrics
            avg_loss = epoch_loss / len(train_loader)
            train_acc = accuracy_score(all_labels, all_preds)
            
            # Validation phase
            valid_acc = None
            if valid_loader is not None:
                model.eval()
                valid_preds = []
                valid_labels = []
                
                with torch.no_grad():
                    for batch_embeddings, batch_labels in valid_loader:
                        if self.device and self.device != 'cpu':
                            batch_embeddings = batch_embeddings.to(self.device)
                            batch_labels = batch_labels.to(self.device)
                        
                        outputs = model(batch_embeddings)
                        _, predicted = torch.max(outputs.data, 1)
                        valid_preds.extend(predicted.cpu().numpy())
                        valid_labels.extend(batch_labels.cpu().numpy())
                
                valid_acc = accuracy_score(valid_labels, valid_preds)
                
                # Save best model
                if valid_acc > training_results['best_accuracy']:
                    training_results['best_accuracy'] = valid_acc
                    training_results['best_model_state'] = model.state_dict().copy()
            
            # Update progress bar with metrics
            if valid_acc is not None:
                epoch_pbar.set_postfix({
                    'Loss': f"{avg_loss:.4f}",
                    'TrainAcc': f"{train_acc:.3f}",
                    'ValidAcc': f"{valid_acc:.3f}"
                })
            else:
                epoch_pbar.set_postfix({
                    'Loss': f"{avg_loss:.4f}",
                    'TrainAcc': f"{train_acc:.3f}"
                })
            
            # Store results
            training_results['epoch_losses'].append(avg_loss)
            training_results['epoch_accuracies'].append(valid_acc if valid_acc is not None else train_acc)
            
            # Update learning rate scheduler
            if scheduler is not None:
                if valid_acc is not None:
                    scheduler.step(valid_acc)
                else:
                    scheduler.step(avg_loss)
        
        epoch_pbar.close()
        
        # Load best model if validation was used
        if training_results['best_model_state'] is not None:
            model.load_state_dict(training_results['best_model_state'])
            SCLLMOutput.status(f"Best validation accuracy: {training_results['best_accuracy']:.4f}", 'training')
        
        return training_results

    def _save_model_specific(self, save_path: Path, **kwargs) -> None:
        """UCE model saving (not implemented)."""
        SCLLMOutput.status("Model saving not implemented for UCE", 'warning')
    
    def __repr__(self):
        return f"UCEModel(device={self.device}, loaded={self.is_loaded})"
    
    def _create_uce_finetune_model(self, n_classes: int, freeze_backbone: bool = False):
        """Create a complete UCE fine-tuning model with differentiable forward pass."""
        import torch.nn as nn
        
        class UCEFineTuneModel(nn.Module):
            def __init__(self, uce_model, n_classes, freeze_backbone=False):
                super().__init__()
                self.uce_model = uce_model
                self.freeze_backbone = freeze_backbone
                
                # Get the actual UCE TransformerModel
                if not hasattr(uce_model, 'model') or uce_model.model is None:
                    raise ValueError("UCE TransformerModel not initialized. This should not happen after load_model().")
                
                # Store the UCE backbone for end-to-end training
                self.uce_backbone = uce_model.model
                self.pe_embedding = uce_model.model_pe_embedding
                
                # Classification head - similar to scGPT
                hidden_dim = uce_model.config['output_dim']  # UCE output dimension (typically 1280)
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, n_classes)
                )
                
                # Configure backbone training
                if freeze_backbone:
                    for param in self.uce_backbone.parameters():
                        param.requires_grad = False
                    SCLLMOutput.status("UCE backbone frozen - linear probing mode", 'info')
                else:
                    for param in self.uce_backbone.parameters():
                        param.requires_grad = True
                    SCLLMOutput.status("UCE backbone unfrozen - end-to-end fine-tuning mode", 'info')
            
            def forward(self, batch_data):
                """Forward pass through UCE + classifier with optional backbone freezing."""
                # batch_data contains (batch_sentences, mask)
                batch_sentences, mask = batch_data
                
                if self.freeze_backbone:
                    # Linear probing mode: use frozen backbone embeddings
                    with torch.no_grad():
                        # Forward pass through UCE backbone without gradients
                        batch_sentences = batch_sentences.permute(1, 0)
                        batch_sentences = self.pe_embedding(batch_sentences.long())
                        batch_sentences = nn.functional.normalize(batch_sentences, dim=2)
                        _, cell_embedding = self.uce_backbone.forward(batch_sentences, mask)
                    
                    # Only the classifier will have gradients
                    logits = self.classifier(cell_embedding)
                    return logits
                else:
                    # End-to-end mode: full gradient flow
                    batch_sentences = batch_sentences.permute(1, 0)
                    batch_sentences = self.pe_embedding(batch_sentences.long())
                    batch_sentences = nn.functional.normalize(batch_sentences, dim=2)
                    _, cell_embedding = self.uce_backbone.forward(batch_sentences, mask)
                    logits = self.classifier(cell_embedding)
                    return logits
        
        # Initialize fine-tuning model
        finetune_model = UCEFineTuneModel(self, n_classes, freeze_backbone)
        finetune_model.to(self.device)
        return finetune_model
    
    def _prepare_finetune_dataset(self, adata_processed):
        """Prepare dataset for UCE end-to-end fine-tuning."""
        from torch.utils.data import Dataset
        
        class UCEFineTuneDataset(Dataset):
            def __init__(self, adata, uce_model):
                self.adata = adata
                self.uce_model = uce_model
                self.n_obs = adata.n_obs
                
                # Extract cell type labels if available
                if 'celltype_id' in adata.obs:
                    self.labels = adata.obs['celltype_id'].values.astype(int)
                    self.has_labels = True
                else:
                    self.labels = np.zeros(self.n_obs, dtype=int)  # Dummy labels
                    self.has_labels = False
                
                # Prepare UCE-style data processing for differentiable forward pass
                SCLLMOutput.status(f"Preparing UCE raw data for {adata.n_obs} cells × {adata.n_vars} genes...", 'preprocessing')
                self._prepare_uce_raw_data(adata)
                    
            def _prepare_uce_raw_data(self, adata):
                """Prepare raw data in UCE format for differentiable forward pass."""
                # Process adata to UCE format (similar to _process_adata_direct)
                adata_filtered, pe_row_idxs, dataset_chroms, dataset_pos = self.uce_model._generate_indices_direct(adata)
                
                # Update the stored adata to use the filtered version
                self.adata = adata_filtered
                
                # Store processed data
                self.pe_row_idxs = pe_row_idxs
                self.dataset_chroms = dataset_chroms
                self.dataset_pos = dataset_pos
                self.X_data = adata_filtered.X.astype(np.int64)  # UCE expects integer counts
                
                # Update dataset size after filtering
                self.n_obs = adata_filtered.n_obs
                self.adata = adata_filtered
                
                SCLLMOutput.status(f"Raw UCE data prepared: {self.X_data.shape}", 'preprocessing')
            
            def __len__(self):
                return self.n_obs
                
            def __getitem__(self, idx):
                # Get raw count data for this cell
                counts = torch.tensor(self.X_data[idx]).unsqueeze(0)
                weights = torch.log1p(counts)
                weights = (weights / torch.sum(weights))
                
                # Generate UCE sentence format (same as in _create_uce_dataset_direct)
                try:
                    from .UCE.eval_data import sample_cell_sentences
                except ImportError:
                    import sys
                    uce_path = Path(__file__).parent / "UCE"
                    if str(uce_path) not in sys.path:
                        sys.path.insert(0, str(uce_path))
                    from eval_data import sample_cell_sentences
                
                # Create UCE args for sampling
                args = self.uce_model._create_uce_args_direct()
                dataset_name = "memory_adata"
                
                dataset_to_protein_embeddings = {dataset_name: self.pe_row_idxs}
                dataset_to_chroms = {dataset_name: self.dataset_chroms}
                dataset_to_starts = {dataset_name: self.dataset_pos}
                
                batch_sentences, mask, seq_len, cell_sentences = sample_cell_sentences(
                    counts, weights, dataset_name, args,
                    dataset_to_protein_embeddings=dataset_to_protein_embeddings,
                    dataset_to_chroms=dataset_to_chroms,
                    dataset_to_starts=dataset_to_starts
                )
                
                # Remove the batch dimension since we're processing single cells
                # batch_sentences shape: (1, seq_len) -> (seq_len,)
                # mask shape: (1, seq_len) -> (seq_len,)  
                batch_sentences = batch_sentences.squeeze(0)
                mask = mask.squeeze(0)
                
                # Return data in format expected by forward pass
                label = torch.tensor(self.labels[idx], dtype=torch.long)
                return (batch_sentences, mask), label
        
        return UCEFineTuneDataset(adata_processed, self)
    
    def _train_uce_finetune_model(self, model, train_loader, valid_loader, 
                                 optimizer, scheduler, criterion, epochs):
        """Train the UCE end-to-end fine-tuning model with optimized progress display."""
        import contextlib
        import io
        from sklearn.metrics import accuracy_score
        
        training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            all_preds = []
            all_labels = []
            
            # Capture verbose output to reduce clutter
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                total_batches = len(train_loader)
                batch_count = 0
                
                for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
                    batch_count += 1
                    optimizer.zero_grad()
                    
                    # Move to device
                    if self.device and self.device != 'cpu':
                        batch_sentences, mask = batch_data
                        batch_sentences = batch_sentences.to(self.device)
                        mask = mask.to(self.device)
                        batch_data = (batch_sentences, mask)
                        batch_labels = batch_labels.to(self.device)
                    
                    # Forward pass through UCE + classifier
                    try:
                        logits = model(batch_data)
                        loss = criterion(logits, batch_labels)
                        
                        # Backward pass
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                        
                        # Track predictions
                        _, predicted = torch.max(logits.data, 1)
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(batch_labels.cpu().numpy())
                        
                        # Show minimal progress every 10% of batches
                        if batch_count % max(1, total_batches // 10) == 0 or batch_count == total_batches:
                            progress = int((batch_count / total_batches) * 100)
                            SCLLMOutput.status(f"🎯 Epoch {epoch+1}/{epochs} [{progress:3d}%] Loss: {loss.item():.4f}", 'fine_tuning')
                    
                    except Exception as e:
                        SCLLMOutput.status(f"Training error: {e}", 'error')
                        raise e
            
            # Calculate training metrics
            avg_train_loss = train_loss / len(train_loader)
            train_acc = accuracy_score(all_labels, all_preds)
            training_history['train_loss'].append(avg_train_loss)
            training_history['train_acc'].append(train_acc)
            
            # Validation phase
            val_loss = 0.0
            val_acc = 0.0
            if valid_loader is not None:
                model.eval()
                val_preds = []
                val_labels = []
                
                with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    for batch_data, batch_labels in valid_loader:
                        # Move to device
                        if self.device and self.device != 'cpu':
                            batch_sentences, mask = batch_data
                            batch_sentences = batch_sentences.to(self.device)
                            mask = mask.to(self.device)
                            batch_data = (batch_sentences, mask)
                            batch_labels = batch_labels.to(self.device)
                            
                        logits = model(batch_data)
                        loss = criterion(logits, batch_labels)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(logits.data, 1)
                        val_preds.extend(predicted.cpu().numpy())
                        val_labels.extend(batch_labels.cpu().numpy())
                
                val_loss = val_loss / len(valid_loader)
                val_acc = accuracy_score(val_labels, val_preds)
                training_history['val_loss'].append(val_loss)
                training_history['val_acc'].append(val_acc)
                
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
            
            # Update scheduler
            scheduler.step()
            
            # Print epoch summary
            metrics_str = f"Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}"
            if valid_loader is not None:
                metrics_str += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            SCLLMOutput.status(f"🎯 Epoch {epoch+1}/{epochs} completed - {metrics_str}", 'fine_tuning')
        
        return training_history
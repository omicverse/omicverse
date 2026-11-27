"""
scLLM: Single-cell Language Models
==================================

A unified interface for single-cell language models including scGPT, scFoundation and others.

Main classes:
- ModelFactory: Factory for creating model objects (``create_model``)
- ScGPTModel: scGPT model implementation
- ScFoundationModel: scFoundation model implementation
- GeneformerModel: Geneformer model implementation
- CellPLMModel: CellPLM model implementation
- UCEModel: UCE (Universal Cell Embeddings) model implementation

Quick start:
```python
import omicverse as ov

print(ov.llm.ModelFactory.available_models())

# scGPT: annotation / embeddings / fine-tuning / integration
model = ov.llm.ModelFactory.create_model(
    "scgpt", "/path/to/scgpt/model", device=None
)
results = model.predict(adata, task="annotation")
embeddings = model.get_embeddings(adata)
results = model.fine_tune(train_adata, task="annotation")
integration_results = model.integrate(adata, batch_key="batch")

# scFoundation
model = ov.llm.ModelFactory.create_model(
    "scfoundation", "/path/to/scfoundation/model.ckpt"
)

# CellPLM
model = ov.llm.ModelFactory.create_model(
    "cellplm", "/path/to/cellplm/checkpoint",
    pretrain_version="20231027_85M",
)

# UCE
model = ov.llm.ModelFactory.create_model(
    "uce", "/path/to/uce/model.torch", species="human"
)
```

Per-model helper functions (``load_scgpt``, ``annotate_with_scgpt``,
``get_embeddings_with_scfoundation``, ``perturb_genes_with_geneformer``, ...)
live in ``omicverse.llm.model_factory``.
"""

from .._optional import bind_optional_symbols
from . import dr
from .dr import ResearchManager

bind_optional_symbols(
    globals(),
    ".base",
    ["SCLLMBase", "ModelConfig", "TaskConfig", "ModelExecutionBlockedError"],
    package=__name__,
    feature="omicverse.llm",
    dependencies=("torch",),
)

bind_optional_symbols(
    globals(),
    ".model_factory",
    ["ModelFactory"],
    package=__name__,
    feature="omicverse.llm",
    dependencies=("torch",),
)

__all__ = [
    "SCLLMBase",
    "ModelConfig",
    "TaskConfig",
    "ModelExecutionBlockedError",
    "ModelFactory",
    "dr",
    "ResearchManager",
]

"""``ov.single.pySIMBA`` -- thin convenience wrapper over :class:`ov.ge.SIMBA`.

Keeps the original ``pySIMBA`` public API (constructor + ``preprocess`` /
``gen_graph`` / ``train`` / ``load`` / ``batch_correction``), but it is now
backed end to end by omicverse's own ``ov.ge`` graph-embedding package --
no ``import simba`` / ``import simba_pbg`` anywhere. See
``ovfast/ge/SPEC_ge_simba.md`` for the full design.
"""
from __future__ import annotations

from .._settings import add_reference
from .._registry import register_function


@register_function(
    aliases=['SIMBA整合器', 'pySIMBA', 'single-cell batch integration simba'],
    category="single",
    description="SIMBA wrapper for batch correction and integrated manifold learning across single-cell datasets.",
    prerequisites={'optional_functions': ['pp.preprocess']},
    requires={'obs': ['batch labels']},
    produces={'obsm': ['X_simba'], 'uns': ['simba graph/model']},
    auto_fix='none',
    examples=['simba_object = ov.single.pySIMBA(adata, workdir)', 'simba_object.preprocess(batch_key="batch")'],
    related=['single.batch_correction', 'pp.neighbors', 'pp.umap', 'ge.SIMBA']
)
class pySIMBA(object):
    """
    SIMBA wrapper for single-cell batch integration and graph-embedding construction.

    Backed by :class:`omicverse.ge.SIMBA` (a native, pure-PyTorch
    re-implementation of SIMBA/PyTorch-BigGraph) -- no external ``simba`` /
    ``simba_pbg`` install required.

    Parameters
    ----------
    adata:AnnData
        Input AnnData with batch labels in ``obs``.
    workdir:str, optional
        Directory used to persist the trained graph/model (``save``/``load``).

    Returns
    -------
    None
        Initializes SIMBA runtime and stores input data/workspace.

    Examples
    --------
    >>> simba_object = ov.single.pySIMBA(adata, workdir)
    """

    def __init__(self, adata, workdir="simba_result") -> None:
        r"""Initialize SIMBA object for batch correction and multimodal analysis.

        Parameters
        ----------
        adata:anndata.AnnData
            Input AnnData containing single-cell profiles and batch labels.
        workdir:str
            Working directory used to persist the trained model.
        """
        self.adata = adata
        self.workdir = workdir
        self.simba_model = None
        self._batch_key = "batch"
        self._ref_batch = None

    def preprocess(self, batch_key='batch', min_n_cells=3,
                  method='lib_size', n_top_genes=3000, n_bins=5, span=0.3):
        r"""Preprocess the AnnData object for SIMBA analysis.

        Parameters
        ----------
        batch_key:str
            Column in ``adata.obs`` containing batch labels.
        min_n_cells:int
            Minimum number of cells required to keep a gene.
        method:str
            Normalization method used by SIMBA preprocessing (``'lib_size'``
            or ``'tf_idf'``).
        n_top_genes:int
            Number of variable genes selected per batch.
        n_bins:int
            Number of bins used in discretization.
        span:float
            Loess smoothing span for highly-variable-gene selection (SIMBA's
            default is ``0.3``; raise it if ``select_variable_genes`` raises
            a loess near-singularity error on a very small/sparse batch).
        """
        from .. import ge

        self._batch_key = batch_key
        adata = self.adata
        adata.obs[batch_key] = adata.obs[batch_key].astype('category')
        adata_dict = {}
        for batch in adata.obs[batch_key].cat.categories:
            sub = adata[adata.obs[batch_key] == batch].copy()
            n_cells_expr = (sub.X > 0).sum(axis=0)
            import numpy as np
            keep = (np.asarray(n_cells_expr).flatten() >= min_n_cells)
            sub = sub[:, keep].copy()
            ge.normalize(sub, method=method)
            ge.log_transform(sub)
            ge.select_variable_genes(sub, n_top_genes=min(n_top_genes, sub.n_vars - 1), span=span)
            ge.discretize(sub, n_bins=n_bins)
            adata_dict[batch] = sub
        adata.uns['simba_batch_dict'] = adata_dict

    def gen_graph(self, n_components=15, k=15, copy=False, dirname='graph0'):
        r"""Generate the graph structure for batch correction.

        Parameters
        ----------
        n_components:int
            Number of components used for edge inference.
        k:int
            Number of neighbors in inter-batch edge inference.
        copy:bool
            Unused (kept for API compatibility with the original wrapper).
        dirname:str
            Unused (kept for API compatibility with the original wrapper).
        """
        from .. import ge

        batch_dict = self.adata.uns['simba_batch_dict']
        sizes = {b: a.shape[0] for b, a in batch_dict.items()}
        result_max = max(sizes, key=sizes.get)
        self._ref_batch = result_max

        edge_dict = {}
        for batch in batch_dict:
            if batch == result_max:
                continue
            edge_dict[batch] = ge.infer_edges(batch_dict[result_max], batch_dict[batch],
                                              n_components=n_components, k=k)
        self.adata.uns['simba_batch_edge_dict'] = edge_dict

        model = ge.SIMBA()
        for batch, sub in batch_dict.items():
            model.add_cg(sub, cell_type=str(batch), gene_type='G')
        for batch, edge_adata in edge_dict.items():
            model.add_cc(edge_adata, lhs_type=str(result_max), rhs_type=str(batch))
        model.build()
        self.simba_model = model

    def train(self, num_workers=12, auto_wd=True, save_wd=True, output='model'):
        r"""Train the SIMBA model for batch correction.

        Parameters
        ----------
        num_workers:int
            Unused: ``ov.ge`` trains with one vectorised process (CPU/GPU
            tensor ops) rather than PBG's Hogwild multi-process workers.
        auto_wd:bool
            Whether to enable automatic weight-decay tuning (SIMBA/PBG's
            sample-size-scaled rule, see ``ge.HeteroGraph.auto_wd``).
        save_wd:bool
            Unused (kept for API compatibility).
        output:str
            Unused (kept for API compatibility).
        """
        if self.simba_model is None:
            raise RuntimeError("call gen_graph() before train()")
        self.simba_model.train(num_epochs=10, batch_size=1000, lr=0.1, auto_wd=auto_wd,
                               num_batch_negs=50, num_uniform_negs=50)
        add_reference(self.adata, 'SIMBA', 'batch correction with SIMBA')

    def load(self, model_path=None):
        r"""Load a pre-trained SIMBA model for batch correction.

        Parameters
        ----------
        model_path:str or None
            Path to a model directory saved by ``ge.SIMBA.save``. If
            ``None``, the already-trained in-process model (from
            ``train()`` in this session) is used as-is.
        """
        if model_path is None:
            if self.simba_model is None:
                raise RuntimeError("no trained model in this session; pass model_path=")
            return
        from .. import ge
        self.simba_model = ge.SIMBA.load(model_path)

    def batch_correction(self, use_precomputed=False):
        r"""Perform batch correction using the trained SIMBA model.

        Parameters
        ----------
        use_precomputed:bool
            Reuse an already-computed softmax projection
            (``adata_query.layers['softmax']``) instead of recomputing it.

        Returns
        -------
        anndata.AnnData
            Batch-corrected AnnData with integrated embedding in ``obsm['X_simba']``.
        """
        if self.simba_model is None:
            raise RuntimeError("call gen_graph() and train() before batch_correction()")
        batches = list(self.adata.obs[self._batch_key].cat.categories)
        ref_batch = self._ref_batch or batches[0]
        query_batches = [str(b) for b in batches if b != ref_batch]

        adata_all = self.simba_model.embed_query(str(ref_batch), query_batches)

        import numpy as np
        order = adata_all.obs_names.get_indexer(self.adata.obs_names.astype(str))
        if (order < 0).any():
            missing = int((order < 0).sum())
            raise ValueError(f"{missing} cells are missing from the SIMBA embedding")
        self.adata.obsm['X_simba'] = np.asarray(adata_all.X)[order]
        return self.adata

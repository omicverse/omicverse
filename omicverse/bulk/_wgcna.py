"""Weighted Gene Co-expression Network Analysis registry shim.

The full implementation lives in :mod:`omicverse.external.PyWGCNA` (vendored
from the upstream PyWGCNA package). The registry scanner intentionally
ignores everything under ``external/`` to keep third-party code out of the
public API surface, so a class registered there is invisible to
``ov.utils.registry_lookup``.

This module re-publishes :class:`pyWGCNA` and :func:`readWGCNA` under the
canonical ``ov.bulk`` namespace and attaches the ``@register_function``
metadata that the scanner indexes — making the function discoverable to
agents searching for ``"wgcna"`` / ``"co-expression"`` / similar queries.

The ``external`` import is **deferred to call time** so that simply
importing :mod:`omicverse.bulk` never pulls in PyWGCNA's optional
dependencies (CI on minimal environments stays green; the scanner only
needs the AST of this file, not its imports).
"""

from __future__ import annotations

from .._registry import register_function


@register_function(
    aliases=[
        "pyWGCNA",
        "WGCNA",
        "wgcna",
        "wgcna_analysis",
        "co-expression network",
        "gene co-expression",
        "weighted gene co-expression network analysis",
        "module detection",
        "WGCNA分析",
        "加权基因共表达网络",
    ],
    category="bulk",
    description=(
        "Weighted Gene Co-expression Network Analysis. Detects co-expression "
        "modules from a samples × genes expression matrix, computes module "
        "eigengenes and module-trait correlations. Wraps the vendored "
        "PyWGCNA implementation."
    ),
    examples=[
        "import omicverse as ov, pandas as pd",
        "data = pd.read_csv('expressionList.csv', index_col=0)  # rows=samples, cols=genes",
        "wgcna = ov.bulk.pyWGCNA(name='MyAnalysis', species='mus musculus', geneExp=data, save=True)",
        "wgcna.preprocess()       # filter low-expressed genes / outlier samples",
        "wgcna.findModules()      # soft-threshold + dynamic tree cut",
        "# for large gene counts, run the memory-bounded blockwise pipeline:",
        "wgcna.findModules(max_block_size=5000)  # R blockwiseModules analogue",
        "wgcna.analyseWGCNA()     # module-trait correlations against sample metadata",
    ],
    related=["ov.bulk.readWGCNA", "ov.bulk.pyDEG", "ov.bulk.geneset_enrichment"],
)
class pyWGCNA:
    """Weighted Gene Co-expression Network Analysis.

    Identifies highly co-expressed gene modules and relates them to clinical
    traits / sample metadata. Standard WGCNA workflow:

    1. **Preprocess** — remove low-expressed genes (TPM cutoff) and outlier
       samples (Euclidean distance to mean).
    2. **Soft-thresholding** — pick a power that yields scale-free topology
       in the gene-gene correlation network.
    3. **Adjacency + TOM** — adjacency = ``|cor|^power``; topological
       overlap matrix (TOM) measures shared neighbourhood.
    4. **Dynamic tree cut** — hierarchical clustering on ``1 - TOM``; tree
       cut yields gene modules (named by colour).
    5. **Module eigengenes** — first principal component of each module's
       expression matrix.
    6. **Module-trait correlation** — Pearson correlation of each module
       eigengene against numeric sample traits, with FDR-corrected p-values.

    Parameters
    ----------
    name : str
        Analysis label, used for output file names.
    species : str
        Organism (e.g. ``"mus musculus"``, ``"homo sapiens"``).
    geneExp : pandas.DataFrame
        Expression matrix shaped (samples × genes): sample identifiers are
        the row index, gene identifiers are the column names — the same
        orientation AnnData uses. (The upstream PyWGCNA docstring claims
        the opposite, but its constructor in ``geneExp.py`` treats rows as
        samples and columns as genes, so pass samples × genes.)
    TPMcutoff : float, default 1
        Per-gene TPM threshold; genes whose maximum across samples falls
        below this are dropped during ``preprocess``.
    powers : list[int], optional
        Candidate soft-threshold powers. Defaults to a 1–30 sweep.
    networkType : {"signed", "unsigned", "signed hybrid"}
        How adjacency is computed from correlation.
    minModuleSize : int, default 50
        Smallest module size kept by the dynamic tree cut.
    save : bool, default False
        Whether to persist results to disk.

    Notes
    -----
    Expression CSVs are usually already shaped samples × genes (rows =
    samples, columns = genes); pass them directly — do **not** transpose.

    Methods (call in this order — each step populates the attributes
    listed under it). Use the high-level :meth:`runWGCNA` to chain
    everything end-to-end, or the explicit methods below for finer
    control:

    - ``preprocess()`` — drop low-TPM genes, drop outlier samples
      (updates ``self.datExpr``).
    - ``calculate_soft_threshold()`` — scale-free fit power scan; sets
      ``self.power`` (int, **not** ``self.softPower``) and ``self.sft``
      (DataFrame with R²/slope/k per power).
    - ``calculating_adjacency_matrix()`` — sets ``self.adjacency``.
    - ``calculating_TOM_similarity_matrix()`` — sets ``self.TOM``.
    - ``calculate_geneTree()`` — sets ``self.geneTree`` (linkage matrix).
    - ``calculate_dynamicMods(kwargs_function={...})`` — sets
      ``self.dynamicMods`` and ``self.datExpr.var['dynamicColors']``.
    - ``calculate_gene_module(kwargs_function={...})`` — merges close
      modules, sets ``self.datExpr.var['moduleColors']``,
      ``self.datExpr.var['moduleLabels']``, ``self.MEs``, ``self.datME``.
    - ``findModules()`` — convenience that runs the soft-threshold +
      adjacency + TOM + tree + module merge as one call (preferred).
      Pass ``findModules(max_block_size=5000)`` for large gene counts:
      this switches to the memory-bounded blockwise pipeline (the R
      WGCNA ``blockwiseModules`` analogue) — genes are pre-clustered
      into size-capped blocks and the dense gene×gene adjacency / TOM
      is built one block at a time, so peak memory is ``max_block_size²``
      instead of ``N²``. Without it, ``N >> 10000`` genes can OOM.
    - ``runWGCNA()`` — runs ``preprocess()`` then ``findModules()``.
    - ``analyseWGCNA(geneList=None)`` — module–trait correlation; sets
      ``self.moduleTraitCor`` and ``self.moduleTraitPvalue``.
      Requires sample metadata (set via ``updateSampleInfo(...)`` or
      passed via ``sampleInfo`` at construction).

    Attributes (state machine — populated in this order). The class is
    a thin shim that delegates to the upstream PyWGCNA implementation;
    these are the **actual attribute names** on the returned instance,
    which agents commonly mis-spell:

    - ``self.geneExpr`` — AnnData (genes × samples) holding the
      original input expression.
    - ``self.datExpr`` — AnnData (genes × samples), filtered after
      ``preprocess()``. Per-gene module annotations live on
      ``self.datExpr.var``.
    - ``self.power`` (int) — chosen soft-threshold power. **The
      attribute is ``power``, NOT ``softPower``.** Set after
      ``calculate_soft_threshold()`` or ``findModules()``; before that
      it is ``0``.
    - ``self.sft`` (pandas.DataFrame) — scale-free fit table per
      candidate power (columns: ``Power``, ``SFT.R.sq``, ``slope``,
      ``mean(k)``, …). Set together with ``self.power``.
    - ``self.adjacency`` (pandas.DataFrame) — gene-gene weighted
      adjacency. ``None`` until ``calculating_adjacency_matrix()`` /
      ``findModules()`` runs.
    - ``self.TOM`` (numpy.ndarray) — topological overlap matrix.
      ``None`` until ``calculating_TOM_similarity_matrix()`` /
      ``findModules()`` runs.
    - ``self.geneTree`` — scipy linkage matrix from ``1 - TOM``.
    - ``self.dynamicMods`` — initial dynamic-tree-cut module integer
      labels per gene.
    - ``self.datExpr.var['dynamicColors']`` — initial module colour per
      gene (string, e.g. ``'turquoise'``).
    - ``self.datExpr.var['moduleColors']`` — final module colour per
      gene (after merging close modules). Use this for downstream.
    - ``self.datExpr.var['moduleLabels']`` — integer label per gene
      aligned to ``moduleColors``.
    - ``self.MEs`` (pandas.DataFrame) — module eigengenes, samples ×
      modules. **Do not compute this manually** — the class already
      provides it; manual mean-by-mask is not equivalent (eigengene =
      first PC of the module's expression, not the mean).
    - ``self.datME`` — pre-merge eigengene matrix; usually
      ``self.MEs`` is what you want.
    - ``self.moduleTraitCor`` (pandas.DataFrame) — module × trait
      Pearson correlations. ``None`` until ``analyseWGCNA()`` runs.
    - ``self.moduleTraitPvalue`` (pandas.DataFrame) — parallel p-value
      table. ``None`` until ``analyseWGCNA()`` runs.

    Examples
    --------
    >>> import pandas as pd, omicverse as ov
    >>> data = pd.read_csv('expressionList.csv', index_col=0)  # samples × genes
    >>> wgcna = ov.bulk.pyWGCNA(
    ...     name='5xFAD',
    ...     species='mus musculus',
    ...     geneExp=data,             # rows = samples, columns = genes
    ...     TPMcutoff=1,
    ...     networkType='signed hybrid',
    ... )
    >>> wgcna.preprocess()
    >>> wgcna.findModules()              # or findModules(max_block_size=5000)
    """

    def __new__(
        cls,
        name="WGCNA",
        TPMcutoff=1,
        powers=None,
        RsquaredCut=0.85,
        MeanCut=100,
        networkType="signed hybrid",
        TOMType="signed",
        minModuleSize=50,
        naColor="grey",
        cut=float("inf"),
        MEDissThres=0.2,
        species=None,
        level="gene",
        anndata=None,
        geneExp=None,
        geneExpPath=None,
        sep=",",
        geneInfo=None,
        sampleInfo=None,
        save=False,
        outputPath=None,
        figureType="pdf",
    ):
        # Lazy import — keeps `external/` off `bulk` import path, so CI
        # without PyWGCNA's optional deps still imports `omicverse.bulk`
        # cleanly. The registry scanner only needs this file's AST, not
        # its imports, so the discoverability is unaffected.
        from ..external.PyWGCNA.wgcna import pyWGCNA as _impl
        return _impl(
            name=name,
            TPMcutoff=TPMcutoff,
            powers=powers,
            RsquaredCut=RsquaredCut,
            MeanCut=MeanCut,
            networkType=networkType,
            TOMType=TOMType,
            minModuleSize=minModuleSize,
            naColor=naColor,
            cut=cut,
            MEDissThres=MEDissThres,
            species=species,
            level=level,
            anndata=anndata,
            geneExp=geneExp,
            geneExpPath=geneExpPath,
            sep=sep,
            geneInfo=geneInfo,
            sampleInfo=sampleInfo,
            save=save,
            outputPath=outputPath,
            figureType=figureType,
        )


def readWGCNA(file):
    """Load a previously saved WGCNA object from disk.

    Lazy wrapper around :func:`omicverse.external.PyWGCNA.utils.readWGCNA`.

    Parameters
    ----------
    file : str
        Path to the pickled object produced by ``pyWGCNA.saveWGCNA(...)``.

    Returns
    -------
    pyWGCNA
        Restored analysis object with all attributes (``datExpr``,
        ``MEs``, ``moduleTraitCor`` …) populated.
    """
    from ..external.PyWGCNA.utils import readWGCNA as _impl
    return _impl(file)


__all__ = ["pyWGCNA", "readWGCNA"]

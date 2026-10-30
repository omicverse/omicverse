"""
In-silico gene perturbation with gene-response and trajectory outputs.

This module exposes :func:`perturb` — a unified entry point that knocks
out / over-expresses a gene (or list of genes) in an AnnData and returns
backend-specific gene responses, trajectory information and, where
supported, regulatory networks. Dynamo stores responses in added layers
without replacing baseline ``X`` and does not reconstruct a GRN.

Backends (selected via ``backend=``):

* ``"sctenifoldknk"`` — purpose-built for **scRNA-only** in-silico KO /
  OE. Reconstructs a PCNet from the scRNA counts, perturbs the gene's
  edges (set to 0 for KO; boosted for OE), and diff-compares the two
  networks. **Default for plain expression AnnData.**
* ``"cell_oracle"`` — GRN-based simulation. Uses a base GRN (from ATAC
  + motif if available, otherwise the package-bundled mm10/hg38 base
  GRN) and propagates the perturbation through it. Returns the
  simulated post-perturbation GRN + a cell-state shift vector field.
* ``"dynamo"`` — vector-field / Jacobian simulation (``dynamo.pd``).
  By default, perturbs the target's expression and propagates it through the
  **analytical Jacobian** of the learned PCA vector field
  (``δY = J · δX``, Qiu 2022) to predict per-cell, per-gene response
  plus a perturbation transition matrix. Requires a dynamo-processed
  AnnData (PCA + learned ``VecFld_pca``); with ``auto_pipeline=True``
  the standard dynamo recipe is run on demand. When the AnnData
  already carries a dynamo vector field, ``backend='auto'`` picks it.
* ``"auto"`` (default) — picks ``dynamo`` when a learned dynamo vector
  field is present (``adata.uns['VecFld_pca']``), ``cell_oracle`` if a
  base GRN is present, otherwise ``sctenifoldknk``.

The dependencies (``sctenifoldpy``, ``celloracle``, ``dynamo-release``)
are loaded **lazily** so ``omicverse.single`` imports cleanly even when
only one backend is installed.

Example
-------
>>> import omicverse as ov
>>> ov.style(font_path='Arial')
>>>
>>> # Knock out Sox2 and reconstruct the downstream GRN
>>> result = ov.single.perturb(adata, target='Sox2', mode='ko',
...                            backend='sctenifoldknk', grn_output=True)
>>> result.adata_perturbed       # predicted AnnData after KO
>>> result.grn                   # networkx.DiGraph of the perturbed GRN
>>> result.delta_grn             # DataFrame of edge weight changes
>>> result.delta_expr             # DataFrame of gene-expression changes
>>> result.summary(top_n=10)     # printable summary

Over-expression mirrors the KO call with ``mode='oe'`` and an optional
``fold_change`` multiplier::

    result = ov.single.perturb(adata, target='Gata1', mode='oe',
                                fold_change=3.0)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .._registry import register_function
from .._optional import build_optional_dependency_error


__all__ = ["perturb", "PerturbResult"]


_VALID_MODES = ("ko", "kd", "oe")
_VALID_BACKENDS = ("auto", "sctenifoldknk", "cell_oracle", "dynamo")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PerturbResult:
    """Bundled output of :func:`perturb`.

    Attributes
    ----------
    target : str or list of str
        Gene(s) perturbed.
    mode : str
        One of ``'ko'`` (knockout), ``'kd'`` (knockdown — partial
        knockout via fold-change reduction), or ``'oe'`` (over-expression).
    backend : str
        Name of the backend that produced this result.
    adata_perturbed : AnnData or None
        Predicted post-perturbation AnnData (when the backend supports
        it; otherwise ``None`` and the user inspects ``delta_expr``
        instead). For dynamo, this is the input object or its copy with
        response layers and projection outputs added. Its ``X`` remains
        the baseline expression, possibly preprocessed by the automatic
        pipeline; it is not replaced with predicted expression. Read
        ``delta_X`` or ``layers[meta['perturbation_key']]`` for responses.
    grn : networkx.DiGraph or None
        Post-perturbation gene regulatory network (TF → target edges
        with weight). ``None`` when ``grn_output=False`` was requested,
        and always ``None`` for dynamo, which does not reconstruct a GRN.
    grn_base : networkx.DiGraph or None
        Pre-perturbation (baseline) GRN. Always ``None`` for dynamo.
    delta_grn : pandas.DataFrame
        Per-edge weight change with columns
        ``[source, target, weight_base, weight_pert, delta]``. Empty for
        dynamo, regardless of ``grn_output``.
    delta_expr : pandas.DataFrame
        Per-gene expression change with columns
        ``[gene, mean_base, mean_pert, delta, log2_fc]``. For dynamo,
        ``delta`` is the mean model response and ``mean_pert`` is the
        derived sum ``mean_base + delta``, not a fitted count estimate.
        Signed responses are preserved even when this sum is negative.
        ``log2_fc`` uses a pseudocount of ``1e-6`` and is ``NaN`` if either
        mean is negative or non-finite; it is a response-derived summary,
        not an experimentally measured expression fold change.
    trajectory_shift : Any
        Cell × cell transition probability matrix. Populated automatically
        when an embedding is available (``adata.obsm[embedding_name]``).
        Compute / refresh post-hoc with :meth:`compute_transition_prob`.
    delta_X : numpy.ndarray or None
        Per-cell, per-gene predicted change (``cells × genes``). All three
        backends populate this — for CellOracle it's
        ``adata.layers['simulated_count'] - adata.layers['imputed_count']``;
        for scTenifoldKnk it's ``X @ (KO_pcnet − WT_pcnet)``; for dynamo it
        is the native perturbation response layer. It is the
        common input that drives every downstream method.
    cell_names : Iterable[str] or None
        Row index for ``delta_X``.
    gene_names : Iterable[str] or None
        Column index for ``delta_X``.
    embedding : numpy.ndarray or None
        2-D embedding used for trajectory analysis (typically a UMAP).
    meta : dict
        Backend-specific extras (timings, hyper-parameters, …).
    """

    target: str | list[str]
    mode: str
    backend: str
    adata_perturbed: Any = None
    grn: Any = None
    grn_base: Any = None
    delta_grn: pd.DataFrame = field(default_factory=pd.DataFrame)
    delta_expr: pd.DataFrame = field(default_factory=pd.DataFrame)
    trajectory_shift: Any = None
    delta_X: Any = None
    cell_names: Any = None
    gene_names: Any = None
    embedding: Any = None
    meta: dict = field(default_factory=dict)

    def summary(self, top_n: int = 10) -> pd.DataFrame:
        """Print + return the top-``n`` most-affected downstream genes.

        Useful as a one-line diagnostic right after :func:`perturb`.
        """
        if self.delta_expr is None or self.delta_expr.empty:
            print(f"[ov.single.perturb] no delta_expr available — "
                  f"backend={self.backend!r} did not emit one.")
            return pd.DataFrame()
        df = self.delta_expr.copy()
        df = df.reindex(df["delta"].abs().sort_values(ascending=False).index)
        top = df.head(top_n)
        print(f"[ov.single.perturb] target={self.target!r} mode={self.mode!r} "
              f"backend={self.backend!r}  — top {top_n} downstream genes "
              f"by |Δexpr|:")
        print(top.to_string(index=False))
        return top

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist this result to a ``.pkl`` on disk.

        All attributes (``delta_X``, ``trajectory_shift``, ``grn``,
        ``delta_expr``, ``adata_perturbed`` …) are picklable, so we
        delegate to :func:`omicverse.utils.save` for cross-version
        compatibility. Reload with :meth:`PerturbResult.load`.
        """
        from .. import utils as ov_utils
        ov_utils.save(self, path)

    @classmethod
    def load(cls, path: str) -> "PerturbResult":
        """Restore a result previously saved via :meth:`save`."""
        from .. import utils as ov_utils
        obj = ov_utils.load(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Loaded object from {path!r} is a {type(obj).__name__}, "
                f"not a PerturbResult."
            )
        return obj

    # ------------------------------------------------------------------
    # Tier A: shared downstream methods
    # ------------------------------------------------------------------

    def compute_transition_prob(
        self,
        adata=None,
        *,
        embedding_name: str = "X_umap",
        n_neighbors: int = 30,
        sigma_corr: float = 0.05,
        n_jobs: int = 4,
    ):
        """Compute / refresh the cell × cell transition probability.

        Both backends use the **same** correlation-kernel formulation —
        see :func:`compute_transition_prob`. After this call,
        :attr:`trajectory_shift` is the cell × cell matrix and every
        downstream method that needs trajectories will work regardless
        of which backend produced ``delta_X``.

        Parameters
        ----------
        adata
            AnnData providing the expression matrix (``X``) and embedding
            (``obsm[embedding_name]``). If ``None``, falls back to the
            embedding stored on this ``PerturbResult`` and to
            ``adata_perturbed.X``.
        embedding_name
            Key under ``adata.obsm`` to use for the kNN graph.
        """
        delta_X, X, embedding, gene_names = _resolve_inputs(
            self, adata=adata, embedding_name=embedding_name
        )
        tp = compute_transition_prob(
            delta_X=delta_X,
            X=X,
            embedding=embedding,
            n_neighbors=n_neighbors,
            sigma_corr=sigma_corr,
            n_jobs=n_jobs,
        )
        self.trajectory_shift = tp
        self.embedding = embedding
        return tp

    def add_significance(
        self,
        adata=None,
        *,
        n_perms: int = 100,
        random_state: int = 0,
        layer: str | None = None,
    ):
        """Attach Z-score + p-value columns to :attr:`delta_expr`.

        For ``cell_oracle`` the null distribution is the per-gene
        absolute |Δ| produced by ``n_perms`` permutations of the
        expression matrix (a non-parametric null that does not require
        re-fitting the GRN). For ``sctenifoldknk`` the columns are
        already present and this is a no-op.

        The added columns are ``z_score``, ``p_value``,
        ``adj_p_value`` (BH-corrected). They overwrite any existing
        columns with the same names.
        """
        if self.backend == "sctenifoldknk":
            return self  # already has Z/p
        if self.delta_expr is None or self.delta_expr.empty:
            return self
        delta_X, X, _, gene_names = _resolve_inputs(self, adata=adata)
        z, p, adj_p = _permutation_significance(
            delta_X=delta_X, X=X, n_perms=n_perms, random_state=random_state
        )
        df = self.delta_expr.copy()
        if "gene" in df.columns and gene_names is not None:
            order = pd.Index(df["gene"]).map(
                {g: i for i, g in enumerate(gene_names)}
            )
            order = order.to_numpy()
            mask = ~pd.isna(order)
            df.loc[mask, "z_score"] = z[order[mask].astype(int)]
            df.loc[mask, "p_value"] = p[order[mask].astype(int)]
            df.loc[mask, "adj_p_value"] = adj_p[order[mask].astype(int)]
        else:
            df["z_score"] = z
            df["p_value"] = p
            df["adj_p_value"] = adj_p
        self.delta_expr = df
        return self

    # ------------------------------------------------------------------
    # Tier B: trajectory analyses + visual primitives
    # ------------------------------------------------------------------

    def delta_embedding(self, adata=None, *, embedding_name: str = "X_umap"):
        """Per-cell 2-D arrow vector on the embedding.

        Replicates CellOracle's ``calculate_embedding_shift`` (and the
        velocyto formulation it inherits): the per-cell displacement is
        a probability-weighted sum of **unit** direction vectors to
        every neighbour, not of raw coordinate differences. That keeps
        the arrow magnitudes O(1) regardless of the embedding's scale
        (UMAP vs PCA vs force-directed graph layout) and makes the PS
        values comparable to CellOracle's published range.

            Δemb[i] = Σ_j T[i,j] · ( (emb[j] − emb[i]) / ||emb[j] − emb[i]|| )
        """
        if self.trajectory_shift is None:
            self.compute_transition_prob(adata=adata, embedding_name=embedding_name)
        _, _, embedding, _ = _resolve_inputs(self, adata=adata,
                                             embedding_name=embedding_name)
        tp = np.asarray(self.trajectory_shift)
        diffs = embedding[None, :, :] - embedding[:, None, :]  # (n, n, 2)
        norms = np.linalg.norm(diffs, axis=-1, keepdims=True)
        # Avoid 0/0 on the diagonal; the row probability is 0 there anyway
        unit_vecs = np.where(norms > 1e-12, diffs / np.maximum(norms, 1e-12), 0.0)
        return np.einsum("ij,ijk->ik", tp, unit_vecs)

    def perturbation_score(
        self,
        adata=None,
        *,
        pseudotime: "str | np.ndarray",
        embedding_name: str = "X_umap",
        n_neighbors: int = 30,
        grid_size: int = 30,
        min_mass: float = 1.0,
        level: str = "cell",
    ):
        """CellOracle-style perturbation score (PS).

        PS is the **raw dot product** of the perturbation flow vector and
        the developmental-gradient vector on a 2-D embedding — exactly
        the construction of CellOracle's
        :class:`Oracle_development_module.calculate_inner_product`:

            simulation_flow_on_grid · pseudotime_gradient_on_grid

        Both vector fields are first **aggregated onto a grid** of
        ``grid_size × grid_size`` points using a Gaussian neighbourhood
        kernel (CellOracle's ``calculate_p_mass`` / ``flow_grid``
        construction); the dot product is computed at each grid point.
        Sparse grid points are filtered with ``min_mass``.

        Sign convention: **PS > 0** = perturbation **promotes**
        differentiation along the local pseudotime gradient; **PS < 0**
        = perturbation **blocks** it.

        Parameters
        ----------
        level
            ``'cell'`` (default) — return a per-cell Series indexed by
            cell name; each cell is assigned the PS of its nearest
            grid point. Suitable for the PS-vs-pseudotime scatter and
            per-cluster boxplot.
            ``'grid'`` — return a dict with the grid arrays
            (``grid_pts``, ``flow_grid``, ``ref_flow_grid``,
            ``ps_grid``, ``mass``, ``keep``) — what
            :func:`ov.pl.perturb_inner_product_on_grid` uses.
        pseudotime : str or array-like
            Column name in ``adata.obs`` or a per-cell array of
            pseudotime values.
        """
        delta_emb = self.delta_embedding(adata=adata, embedding_name=embedding_name)
        _, _, embedding, _ = _resolve_inputs(self, adata=adata,
                                             embedding_name=embedding_name)
        if isinstance(pseudotime, str):
            if adata is None or pseudotime not in adata.obs:
                raise ValueError(f"pseudotime column {pseudotime!r} not found in adata.obs")
            pt = np.asarray(adata.obs[pseudotime].values, dtype=np.float64)
        else:
            pt = np.asarray(pseudotime, dtype=np.float64)

        grid = _compute_ps_grid(
            embedding=embedding,
            delta_emb=delta_emb,
            pseudotime=pt,
            grid_size=grid_size,
            min_mass=min_mass,
            n_neighbors=n_neighbors,
        )
        if level == "grid":
            return grid

        # Per-cell PS = nearest grid-point PS (so each cell carries the
        # local CellOracle-style raw dot product, which is what gets
        # plotted in the PS-vs-pseudotime scatter and the per-cluster
        # boxplot in `visualize_development_module_layout_0`).
        from sklearn.neighbors import NearestNeighbors
        valid = grid["keep"]
        if valid.any():
            nn = NearestNeighbors(n_neighbors=1).fit(grid["grid_pts"][valid])
            _, ix = nn.kneighbors(embedding)
            ps_cell = np.asarray(grid["ps_grid"][valid][ix.ravel()], dtype=np.float64)
        else:
            ps_cell = np.zeros(embedding.shape[0], dtype=np.float64)
        idx = self.cell_names if self.cell_names is not None else range(len(ps_cell))
        return pd.Series(ps_cell, index=list(idx), name="perturbation_score")

    def cluster_transitions(
        self,
        adata=None,
        *,
        cluster_col: str = "leiden",
    ) -> pd.DataFrame:
        """Aggregate ``trajectory_shift`` into a source × target cluster matrix.

        Each row is row-stochastic (rows sum to 1): row ``c`` shows
        where cells of cluster ``c`` are predicted to flow after the
        perturbation, by destination cluster.
        """
        if self.trajectory_shift is None:
            self.compute_transition_prob(adata=adata)
        if adata is None:
            raise ValueError("Need an `adata=` to read the cluster annotation.")
        if cluster_col not in adata.obs:
            raise ValueError(f"cluster_col {cluster_col!r} not in adata.obs")
        labels = pd.Categorical(adata.obs[cluster_col])
        cats = list(labels.categories)
        codes = np.asarray(labels.codes)
        tp = np.asarray(self.trajectory_shift)
        n_clusters = len(cats)
        out = np.zeros((n_clusters, n_clusters), dtype=np.float64)
        for c_src in range(n_clusters):
            mask = codes == c_src
            if not mask.any():
                continue
            avg = tp[mask].mean(axis=0)
            for c_dst in range(n_clusters):
                out[c_src, c_dst] = avg[codes == c_dst].sum()
        return pd.DataFrame(out, index=cats, columns=cats)

    # ------------------------------------------------------------------
    # Tier C: enrichment, Markov, ground-truth validation
    # ------------------------------------------------------------------

    def _ranked_genes(self, *, top_n: int, by: str | None = None) -> list[str]:
        if self.delta_expr is None or self.delta_expr.empty:
            return []
        df = self.delta_expr.copy()
        if by is None:
            for cand in ("Z", "z_score", "delta"):
                if cand in df.columns:
                    by = cand
                    break
        if by is None:
            return []
        col = df[by]
        if "p-value" in df.columns or "p_value" in df.columns:
            df = df.iloc[col.abs().sort_values(ascending=False).index]
        else:
            df = df.iloc[col.abs().sort_values(ascending=False).index]
        return df.head(top_n)["gene"].astype(str).tolist()

    def pathway_enrichment(
        self,
        *,
        top_n: int = 200,
        gene_sets: "str | list[str]" = "GO_Biological_Process_2023",
        organism: str = "mouse",
        rank_by: str | None = None,
        cutoff: float = 0.05,
    ) -> pd.DataFrame:
        """Enrichr over the top-``n`` perturbed genes.

        Uses omicverse's local hypergeometric over-representation test
        (:func:`omicverse.bulk._ora.enrichr`) against any Enrichr library
        (``GO_Biological_Process_2023``, ``KEGG_2021_Human``,
        ``Reactome_2022``, …) — the library is downloaded and cached, then
        tested offline (no Enrichr web API). Genes are ranked by ``|Z|`` if
        present (sctenifoldknk) or by ``|delta|`` (cell_oracle) — pass
        ``rank_by=`` to override.

        Returns
        -------
        pandas.DataFrame
            Enrichr terms with ``Term``, ``P-value``, ``Adjusted P-value``,
            ``Combined Score``, ``Genes`` columns. Empty if no enrichment.
        """
        from ..bulk._ora import enrichr
        genes = self._ranked_genes(top_n=top_n, by=rank_by)
        if not genes:
            return pd.DataFrame()
        if isinstance(gene_sets, str):
            gene_sets = [gene_sets]
        enr = enrichr(
            gene_list=genes,
            gene_sets=gene_sets,
            organism=organism,
            outdir=None,
            cutoff=cutoff,
        )
        if enr is None or enr.results is None:
            return pd.DataFrame()
        return enr.results.copy()

    def phenotype_enrichment(
        self,
        *,
        top_n: int = 200,
        db: str = "MGI_Mammalian_Phenotype_Level_4_2024",
        organism: str = "mouse",
        rank_by: str | None = None,
    ) -> pd.DataFrame:
        """Phenotype-database enrichment for the top perturbed genes.

        Useful aliases:
            * ``'MGI_Mammalian_Phenotype_Level_4_2024'`` — mouse phenotypes
            * ``'Human_Phenotype_Ontology'`` — HPO
            * ``'DisGeNET'`` — disease associations

        Thin wrapper around :meth:`pathway_enrichment` with a different
        Enrichr library — the scTenifoldKnk paper highlights this as
        the recommended downstream analysis (Osorio 2022).
        """
        return self.pathway_enrichment(
            top_n=top_n, gene_sets=db, organism=organism, rank_by=rank_by
        )

    def run_markov(
        self,
        *,
        start_cells,
        n_steps: int = 20,
        n_walks_per_cell: int = 50,
        random_state: int = 0,
        adata=None,
    ) -> pd.DataFrame:
        """Sample Markov walks from ``trajectory_shift``.

        Starting from each of ``start_cells`` (integer indices or
        cell-name strings), follow the transition matrix for ``n_steps``
        steps, ``n_walks_per_cell`` times.

        Returns
        -------
        pandas.DataFrame indexed by start-cell with columns
        ``end_cell_idx`` (one column per walk) — useful for aggregating
        endpoint distributions by cluster.
        """
        if self.trajectory_shift is None:
            self.compute_transition_prob(adata=adata)
        tp = np.asarray(self.trajectory_shift)
        n_cells = tp.shape[0]
        rng = np.random.default_rng(random_state)

        if self.cell_names is None:
            name_to_idx = {i: i for i in range(n_cells)}
        else:
            name_to_idx = {n: i for i, n in enumerate(self.cell_names)}
        start_ix = []
        for c in start_cells:
            if isinstance(c, str):
                if c not in name_to_idx:
                    raise ValueError(f"start cell {c!r} not in result.cell_names")
                start_ix.append(name_to_idx[c])
            else:
                start_ix.append(int(c))

        # cumulative distribution per row for fast sampling
        cdf = np.cumsum(tp, axis=1)
        cdf = cdf / cdf[:, -1:].clip(min=1e-12)

        ends = np.empty((len(start_ix), n_walks_per_cell), dtype=np.int64)
        for i, s in enumerate(start_ix):
            for w in range(n_walks_per_cell):
                cur = s
                for _ in range(n_steps):
                    r = rng.random()
                    cur = int(np.searchsorted(cdf[cur], r, side="left"))
                ends[i, w] = cur
        idx = [self.cell_names[s] if self.cell_names else s for s in start_ix]
        cols = [f"walk_{w}" for w in range(n_walks_per_cell)]
        return pd.DataFrame(ends, index=idx, columns=cols)

    def permutation_test(
        self,
        adata=None,
        *,
        n_perms: int = 100,
        random_state: int = 0,
    ) -> dict:
        """Overall robustness Z against a sign-flip null on ΔX.

        Returns ``{'Z_obs', 'Z_mean_null', 'Z_std_null', 'p_value'}``
        — a single scalar p-value asking 'is the perturbation's overall
        consistency higher than chance?' Complementary to
        :meth:`add_significance` (per-gene).
        """
        delta_X, _, _, _ = _resolve_inputs(self, adata=adata)
        rng = np.random.default_rng(random_state)
        n_cells = delta_X.shape[0]
        sd = delta_X.std(axis=0) + 1e-12
        z_obs = float(np.mean(np.abs(delta_X.mean(axis=0) / (sd / np.sqrt(n_cells)))))
        nulls = np.empty(n_perms)
        for k in range(n_perms):
            signs = rng.choice([-1.0, 1.0], size=n_cells)
            dX_null = delta_X * signs[:, None]
            sd_null = dX_null.std(axis=0) + 1e-12
            nulls[k] = float(np.mean(np.abs(dX_null.mean(axis=0) / (sd_null / np.sqrt(n_cells)))))
        p = (np.sum(nulls >= z_obs) + 1) / (n_perms + 1)
        return dict(
            Z_obs=z_obs, Z_mean_null=float(nulls.mean()),
            Z_std_null=float(nulls.std()), p_value=float(p),
        )

    def validate_against_perturbseq(
        self,
        perturbed_adata,
        control_adata,
        *,
        gene_layer: str | None = None,
        top_k: int = 50,
    ) -> dict:
        """Compare predicted ``delta_expr`` against observed Perturb-seq Δ.

        Computes:
            * Pearson + Spearman correlation between predicted and
              observed gene-level Δ on the shared genes.
            * Top-``k`` precision: of the top-``k`` predicted genes by
              |Δ|, how many are in the top-``k`` observed.

        Parameters
        ----------
        perturbed_adata, control_adata
            AnnData of cells from the same perturbed-vs-control screen
            (e.g. Perturb-seq, ECCITE-seq). The observed Δ is
            ``mean(perturbed) − mean(control)`` per gene on the shared
            gene set.

        Returns
        -------
        dict with keys ``pearson_r``, ``pearson_p``, ``spearman_r``,
        ``spearman_p``, ``top_k_precision``, ``n_shared_genes``,
        ``shared_top_genes``.
        """
        from scipy.stats import pearsonr, spearmanr
        if self.delta_expr is None or self.delta_expr.empty:
            raise ValueError("PerturbResult has no delta_expr to compare.")

        pX = _expression_matrix(perturbed_adata, layer=gene_layer).mean(axis=0)
        cX = _expression_matrix(control_adata, layer=gene_layer).mean(axis=0)
        obs = pd.Series(np.asarray(pX - cX).ravel(),
                        index=list(perturbed_adata.var_names))

        pred = self.delta_expr.set_index("gene")["delta"].copy()
        shared = sorted(set(pred.index) & set(obs.index))
        if not shared:
            raise ValueError("No genes shared between predicted and observed.")
        a = pred.loc[shared].astype(float).to_numpy()
        b = obs.loc[shared].astype(float).to_numpy()

        pr, pp = pearsonr(a, b)
        sr, sp = spearmanr(a, b)
        # Top-k precision (by |Δ|)
        pred_top = pred.loc[shared].abs().sort_values(ascending=False).head(top_k).index
        obs_top = obs.loc[shared].abs().sort_values(ascending=False).head(top_k).index
        inter = sorted(set(pred_top) & set(obs_top))
        return dict(
            n_shared_genes=len(shared),
            pearson_r=float(pr), pearson_p=float(pp),
            spearman_r=float(sr), spearman_p=float(sp),
            top_k_precision=len(inter) / top_k,
            shared_top_genes=inter,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@register_function(
    aliases=[
        "perturb", "in_silico_ko", "in-silico knockout",
        "虚拟敲除", "虚拟扰动", "基因敲除模拟", "基因过表达模拟",
        "knockout simulation", "overexpression simulation",
        "cellOracle wrapper", "scTenifoldKnk wrapper", "GRN perturbation",
    ],
    category="single",
    description=(
        "Unified in-silico gene perturbation (knockout / knockdown / "
        "over-expression) with backend-specific response outputs. Dispatches "
        "to scTenifoldKnk (scRNA-only), CellOracle (RNA + optional ATAC "
        "base GRN) or dynamo (vector-field Jacobian, δY = J·δX) backends. "
        "Returns a PerturbResult with model responses and transition outputs; "
        "GRN outputs are available for network-based backends. Dynamo keeps "
        "baseline expression in X and stores responses in added layers."
    ),
    examples=[
        "import omicverse as ov",
        "ov.style(font_path='Arial')",
        "# Knock out Sox2 using scRNA-only PCNet (scTenifoldKnk)",
        "result = ov.single.perturb(adata, target='Sox2', mode='ko',",
        "                           backend='sctenifoldknk')",
        "result.summary(top_n=20)",
        "result.grn.number_of_edges()",
        "",
        "# Over-express Gata1 via CellOracle",
        "result = ov.single.perturb(adata, target='Gata1', mode='oe',",
        "                           fold_change=3.0, backend='cell_oracle')",
        "result.adata_perturbed",
        "",
        "result = ov.single.perturb(adata, target='Gata1', mode='ko',",
        "                           backend='dynamo')",
        "ov.pl.perturb_quiver(adata, result, embedding_name='X_umap')",
        "",
        "# Visualise top affected TFs as a funky heatmap",
        "ov.pl.funky_heatmap(result.summary(20).reset_index(drop=True))",
    ],
    related=[
        "single.Velo",
        "pl.funky_heatmap",
        "single.SCENIC",
    ],
    auto_fix="escalate",
)
@register_function(
    aliases=[
        "lineage_pseudotime", "谱系特异性伪时", "lineage_specific_pseudotime",
        "branch_pseudotime", "diffusion_pseudotime_per_lineage",
    ],
    category="single",
    description=(
        "Lineage-specific diffusion pseudotime via CellOracle's "
        "`Pseudotime_calculator`. Runs DPT separately per lineage (one "
        "root cell each) and merges into a single `adata.obs['Pseudotime']` "
        "that's monotonic on every branch — required input for the "
        "perturbation-score downstream when the dataset has multiple "
        "terminal cell types."
    ),
    requires={"obsm": ["{obsm_key}"], "obs": ["{cluster_column_name}"]},
    produces={"obs": ["Pseudotime"]},
    auto_fix="none",
    examples=[
        "ov.single.lineage_pseudotime(",
        "    adata,",
        "    lineage_dictionary={",
        "        'Lineage_ME': ['Ery_0','Ery_1',...,'MEP_0','Mk_0'],",
        "        'Lineage_GM': ['GMP_0','GMP_1',...,'Mo_2'],",
        "    },",
        "    root_cells={'Lineage_ME': '1539', 'Lineage_GM': '2244'},",
        "    obsm_key='X_draw_graph_fa',",
        "    cluster_column_name='louvain_annot',",
        ")",
    ],
    related=[
        "single.perturb", "pl.perturb_celloracle_layout",
        "single.PerturbResult.perturbation_score",
    ],
)
def lineage_pseudotime(
    adata,
    lineage_dictionary: dict,
    root_cells: dict,
    *,
    obsm_key: str = "X_umap",
    cluster_column_name: str = "leiden",
    obs_key: str = "Pseudotime",
):
    """Compute lineage-specific DPT pseudotime via CellOracle.

    Wraps :class:`celloracle.applications.Pseudotime_calculator` — runs
    DPT separately on each lineage (root cell per lineage) so the
    resulting :obs ``'Pseudotime'`` is monotonic along every branch.
    Required for the Perturbation-Score downstream when the dataset
    has multiple terminal cell types.

    Parameters
    ----------
    lineage_dictionary
        ``{lineage_name: [cluster_ids_in_lineage]}``.
    root_cells
        ``{lineage_name: cell_index_or_name}`` — the start cell for
        each lineage.
    """
    try:
        from celloracle.applications import Pseudotime_calculator
    except ImportError as exc:  # pragma: no cover
        raise build_optional_dependency_error(
            feature="ov.single.lineage_pseudotime",
            dependencies=("celloracle",),
            install_hint="pip install celloracle",
        ) from exc
    pt = Pseudotime_calculator(
        adata=adata, obsm_key=obsm_key,
        cluster_column_name=cluster_column_name,
    )
    pt.set_lineage(lineage_dictionary=lineage_dictionary)
    pt.set_root_cells(root_cells=root_cells)
    pt.get_pseudotime_per_each_lineage()
    adata.obs[obs_key] = pt.adata.obs["Pseudotime"].values
    return adata


def perturb(
    adata,
    target: str | Sequence[str],
    *,
    mode: str = "ko",
    backend: str = "auto",
    fold_change: float = 2.0,
    grn_base=None,
    grn_output: bool = True,
    return_delta: bool = True,
    layer: str | None = None,
    n_propagation: int = 3,
    backend_kwargs: dict | None = None,
    copy: bool = False,
):
    """In-silico gene perturbation with gene-response and trajectory outputs.

    Parameters
    ----------
    adata : AnnData
        Cells × genes AnnData. For ``backend='sctenifoldknk'`` raw scRNA
        counts are sufficient; for ``backend='cell_oracle'`` a base GRN
        is required (passed via ``grn_base``, looked up from
        ``adata.uns['base_grn']``, or auto-loaded from the CellOracle
        prepackaged mm10/hg38 GRN if neither is set).
    target : str or sequence of str
        Gene name(s) to perturb. Multiple targets are perturbed in the
        same simulation.
    mode : {'ko', 'kd', 'oe'}, default ``'ko'``
        ``'ko'`` — knockout (expression clamped to 0).
        ``'kd'`` — knockdown (expression multiplied by ``1/fold_change``).
        ``'oe'`` — over-expression (expression multiplied by ``fold_change``).
        For dynamo, KD/OE set each target to a common level across cells:
        its mean expression in ``layer`` (or ``X``) divided/multiplied by
        ``fold_change``. They do not scale each cell independently.
        ``backend_kwargs['expression']`` overrides these levels using
        native raw-expression or z-score units, according to ``perturb_mode``.
    backend : {'auto', 'sctenifoldknk', 'cell_oracle', 'dynamo'}, default ``'auto'``
        ``'auto'`` picks ``dynamo`` when a learned dynamo vector field is
        present (``adata.uns['VecFld_pca']``), ``cell_oracle`` when
        ``grn_base`` or ``adata.uns['base_grn']`` is present, else
        ``sctenifoldknk``. ``'dynamo'`` propagates the perturbation through
        the analytical Jacobian of a learned PCA vector field by default (Qiu 2022)
        and needs a dynamo-processed AnnData; pass ``auto_pipeline=True``
        via ``backend_kwargs`` to prepare missing steps on demand. Velocity
        estimation requires spliced/unspliced or metabolic-labeling layers.
        When both a vector field and base GRN are present, ``'auto'`` uses
        dynamo and ignores ``grn_base``; select ``'cell_oracle'`` explicitly
        to use the GRN.
    fold_change : float, default ``2.0``
        Multiplier for OE / KD modes. Ignored for KO.
    grn_base : networkx.DiGraph or DataFrame or None
        Optional baseline GRN for CellOracle (TF → target edges with
        weights). Ignored by scTenifoldKnk (it learns its own PCNet).
    grn_output : bool, default ``True``
        Include the post-perturbation GRN in :class:`PerturbResult.grn`.
        Dynamo does not reconstruct a GRN: ``grn`` and ``grn_base`` are
        always ``None`` and ``delta_grn`` is empty for that backend.
    return_delta : bool, default ``True``
        Include the Δ-edge and Δ-expression tables.
    layer : str or None
        AnnData layer to use as input. ``None`` uses ``adata.X``.
    n_propagation : int, default ``3``
        GRN-propagation steps for CellOracle. Ignored by other backends.
    backend_kwargs : dict or None
        Extra keyword args forwarded to the backend (see individual
        backend docs).
    copy : bool, default ``False``
        If ``True``, do not modify the input ``adata`` in place.

    Returns
    -------
    :class:`PerturbResult`
        Dataclass with ``adata_perturbed``, ``grn``, ``grn_base``,
        ``delta_grn``, ``delta_expr``, ``trajectory_shift``, ``meta``.
        For dynamo, ``adata_perturbed.X`` is baseline expression; responses
        are in ``delta_X`` and the layer named by
        ``meta['perturbation_key']``. ``mean_pert`` in ``delta_expr`` is a
        response-derived summary, not predicted counts. Its ``log2_fc`` is
        undefined (``NaN``) when either mean is negative or non-finite.

    Notes
    -----
    For scTenifoldKnk, KD/OE are one-step linear approximations, not native
    backend simulations: target rows of the full WT network are scaled once
    and ``delta_X = X @ (W_perturbed - W_WT)``. GRN outputs are thresholded
    summaries; their edge-based ``delta_expr`` is not predicted expression.

    See ``Tutorials-single/t_perturb_in_silico.ipynb`` for end-to-end
    KO / OE workflows on a public dataset.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"`mode` must be one of {_VALID_MODES}, got {mode!r}"
        )
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"`backend` must be one of {_VALID_BACKENDS}, got {backend!r}"
        )

    if isinstance(target, str):
        targets = [target]
    else:
        targets = list(target)
    if not targets:
        raise ValueError("`target` must name at least one gene")

    missing = [g for g in targets if g not in adata.var_names]
    if missing:
        raise KeyError(
            f"target gene(s) not in adata.var_names: {missing[:5]}"
            f"{' …' if len(missing) > 5 else ''}"
        )

    if copy:
        adata = adata.copy()

    # ---------------- backend dispatch -----------------
    if backend == "auto":




        has_vector_field = "VecFld_pca" in adata.uns
        has_base_grn = (grn_base is not None) or (
            "base_grn" in adata.uns or "celloracle_base_grn" in adata.uns
        )
        if has_vector_field:
            backend = "dynamo"
        elif has_base_grn:
            backend = "cell_oracle"
        else:
            backend = "sctenifoldknk"

    backend_kwargs = dict(backend_kwargs or {})

    if backend == "sctenifoldknk":
        return _run_sctenifoldknk(
            adata,
            targets=targets,
            mode=mode,
            fold_change=fold_change,
            layer=layer,
            grn_output=grn_output,
            return_delta=return_delta,
            backend_kwargs=backend_kwargs,
        )
    if backend == "cell_oracle":
        return _run_cell_oracle(
            adata,
            targets=targets,
            mode=mode,
            fold_change=fold_change,
            grn_base=grn_base,
            layer=layer,
            n_propagation=n_propagation,
            grn_output=grn_output,
            return_delta=return_delta,
            backend_kwargs=backend_kwargs,
        )
    if backend == "dynamo":
        return _run_dynamo(
            adata,
            targets=targets,
            mode=mode,
            fold_change=fold_change,
            layer=layer,
            grn_output=grn_output,
            return_delta=return_delta,
            backend_kwargs=backend_kwargs,
        )
    raise ValueError(f"unreachable: backend={backend!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Backend: scTenifoldKnk
# ---------------------------------------------------------------------------


def _run_sctenifoldknk(
    adata,
    *,
    targets: Sequence[str],
    mode: str,
    fold_change: float,
    layer: str | None,
    grn_output: bool,
    return_delta: bool,
    backend_kwargs: dict,
) -> PerturbResult:
    """scTenifoldKnk backend — scRNA-only KO / OE via PCNet perturbation.

    Strategy
    --------
    1. Build (or take) a PCNet from the scRNA counts via scTenifoldKnk
       / scTenifoldNet under the hood.
    2. For each target gene g:
         - ``mode='ko'``: use the backend's knockout network.
         - ``mode='kd'``: scale by ``1/fold_change``.
         - ``mode='oe'``: scale by ``fold_change``.
    3. Compare control vs perturbed network. The Δ-edge table is the
       direct output. Downstream Δ-expression is approximated by
       propagating the perturbation one step through the PCNet
       (matrix-vector product with the network).
    """
    sctenifold = _try_import_sctenifoldknk()

    # Real scTenifold API (installed via `pip install sctenifoldpy`):
    #
    #   from scTenifold import scTenifoldKnk
    #   knk = scTenifoldKnk(data=counts_df_gene_x_cell, ko_genes=[...])
    #   knk.build()
    #   knk.tensor_dict['WT']  ← gene×gene tensor (numpy.ndarray)
    #   knk.tensor_dict['KO']  ← gene×gene tensor after virtual KO
    #   knk.d_regulation       ← DataFrame: Gene, Distance, Z, FC, p-value, …
    #   knk.shared_gene_names  ← row/col index for the tensors
    #
    # The counts DataFrame is gene-rows × cell-columns (transposed
    # relative to AnnData).
    counts = _expression_matrix(adata, layer=layer).T  # genes × cells
    counts_df = pd.DataFrame(
        counts,
        index=list(adata.var_names),
        columns=list(adata.obs_names),
    )

    try:
        knk = sctenifold.scTenifoldKnk(
            data=counts_df,
            ko_genes=list(targets) if mode == "ko" else None,
            **backend_kwargs,
        )
        knk.build()
        wt_tensor = knk.tensor_dict.get("WT") if hasattr(knk, "tensor_dict") else None
        ko_tensor = knk.tensor_dict.get("KO") if hasattr(knk, "tensor_dict") else None
        gene_names = getattr(knk, "shared_gene_names", None) or list(counts_df.index)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "scTenifoldKnk backend failed; see traceback for the cause "
            "(common ones: too few cells, NaNs in counts, target gene "
            "missing from the network after filtering)."
        ) from exc

    grn_base = _tensor_to_graph(wt_tensor, gene_names=gene_names)
    # For KO the KO tensor produced by scTenifoldKnk is the perturbed graph.
    # For KD / OE we scale the outgoing WT edges of each target.
    if mode == "ko" and ko_tensor is not None:
        grn_pert = _tensor_to_graph(ko_tensor, gene_names=gene_names)
    else:
        grn_pert = _apply_perturbation_to_graph(
            grn_base, targets=targets, mode=mode, fold_change=fold_change
        )

    delta_grn = _diff_grn(grn_base, grn_pert) if return_delta else pd.DataFrame()

    # Prefer the d_regulation table that scTenifoldKnk emits — it carries
    # statistical significance (Z, FC, p-value, adjusted p-value) for each
    # gene's downstream change. Fall back to our generic GRN-propagation
    # estimate when scTenifold didn't compute it (KD / OE modes).
    delta_expr = pd.DataFrame()
    if return_delta:
        d_reg = getattr(knk, "d_regulation", None)
        if mode == "ko" and isinstance(d_reg, pd.DataFrame) and not d_reg.empty:
            delta_expr = d_reg.rename(
                columns={"Gene": "gene", "FC": "log2_fc", "Distance": "delta"}
            ).copy()
            # Add the columns generic clients expect
            if "mean_base" not in delta_expr.columns:
                delta_expr["mean_base"] = np.nan
            if "mean_pert" not in delta_expr.columns:
                delta_expr["mean_pert"] = np.nan
        else:
            delta_expr = _delta_from_grn(
                grn_base, grn_pert,
                targets=targets, mode=mode, fold_change=fold_change,
            )

    # Per-cell ΔX via one-step propagation through the perturbed PCNet:
    #   ΔX[cell, gene_j] = sum_i X[cell, gene_i] * (perturbed[i,j] - WT[i,j])
    # The PCNets are gene × gene weight matrices indexed by
    # ``shared_gene_names`` (the genes that survived scTenifold's QC).
    delta_X = None
    cell_names = None
    embedding = None
    transition_prob = None
    if wt_tensor is not None and ko_tensor is not None:
        try:
            shared = list(gene_names)
            # X restricted to shared genes, in the same order as the PCNet
            shared_in_adata = [g for g in shared if g in adata.var_names]
            if len(shared_in_adata) == len(shared):
                X_sub = _expression_matrix(adata[:, shared], layer=layer)
                if mode == "ko":
                    pert_arr = np.asarray(ko_tensor, dtype=np.float64)
                    wt_arr = np.asarray(wt_tensor, dtype=np.float64)
                else:
                    wt_arr = np.asarray(wt_tensor, dtype=np.float64)
                    pert_arr = wt_arr.copy()
                    target_rows = [i for i, gene in enumerate(shared) if gene in targets]
                    factor = 1.0 / fold_change if mode == "kd" else fold_change
                    pert_arr[target_rows, :] *= factor
                delta_X = X_sub @ (pert_arr - wt_arr)
                cell_names = list(adata.obs_names)
                # Compute transition_prob if an embedding is available
                for emb_key in ("X_umap", "X_draw_graph_fa", "X_pca"):
                    if emb_key in adata.obsm:
                        embedding = np.asarray(adata.obsm[emb_key])
                        break
                if embedding is not None and delta_X.shape[0] > 5:
                    try:
                        transition_prob = compute_transition_prob(
                            delta_X=delta_X,
                            X=X_sub,
                            embedding=embedding,
                            n_neighbors=min(30, adata.n_obs - 1),
                        )
                    except Exception as exc:  # pragma: no cover
                        warnings.warn(
                            f"compute_transition_prob failed ({exc!r}); "
                            "trajectory_shift not populated.",
                            RuntimeWarning,
                        )
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"delta_X computation failed ({exc!r}); per-cell outputs "
                "not populated.",
                RuntimeWarning,
            )

    return PerturbResult(
        target=targets[0] if len(targets) == 1 else list(targets),
        mode=mode,
        backend="sctenifoldknk",
        adata_perturbed=None,  # backend doesn't synthesise a new AnnData
        grn=grn_pert if grn_output else None,
        grn_base=grn_base if grn_output else None,
        delta_grn=delta_grn,
        delta_expr=delta_expr,
        trajectory_shift=transition_prob,
        delta_X=delta_X,
        cell_names=cell_names,
        gene_names=list(gene_names),
        embedding=embedding,
        meta={"library": "sctenifoldpy", "n_cells": adata.n_obs,
              "n_shared_genes": len(gene_names)},
    )


def _try_import_sctenifoldknk():
    """Return the ``scTenifold`` top-level module.

    Resolution order:
      1. ``sys.modules['scTenifold']`` if already imported (respects
         test fakes injected via ``sys.modules``).
      2. System-installed ``scTenifold`` if importable.
      3. Vendored copy at ``omicverse.external.scTenifold`` — bundled
         so end-users don't need ``pip install sctenifoldpy``.
    """
    import sys
    if "scTenifold" in sys.modules:
        return sys.modules["scTenifold"]
    try:
        import scTenifold  # type: ignore
        return scTenifold
    except ImportError:
        pass
    try:
        from ..external import scTenifold as vendored_sct
        return vendored_sct
    except Exception as exc:  # pragma: no cover
        raise build_optional_dependency_error(
            feature="ov.single.perturb (backend='sctenifoldknk')",
            dependencies=("scTenifold",),
            install_hint="pip install sctenifoldpy  # or use the vendored omicverse.external.scTenifold",
        ) from exc


# ---------------------------------------------------------------------------
# Backend: CellOracle
# ---------------------------------------------------------------------------


def _run_cell_oracle(
    adata,
    *,
    targets: Sequence[str],
    mode: str,
    fold_change: float,
    grn_base,
    layer: str | None,
    n_propagation: int,
    grn_output: bool,
    return_delta: bool,
    backend_kwargs: dict,
) -> PerturbResult:
    """CellOracle backend — GRN-based simulation of TF KO / OE.

    Builds (or takes) a CellOracle ``Oracle`` object, runs
    ``simulate_shift`` with the per-target value dict, and extracts:

    * post-perturbation GRN edges from ``oracle.coef_matrix`` after the
      simulation, vs the same matrix on the baseline (``grn_base``).
    * per-cell delta-expression from ``oracle.adata`` ``.layers``
      (``imputed_count`` vs ``simulated_count``).
    * trajectory shift = the ``transition_prob`` matrix that
      CellOracle emits — passed through to the user.
    """
    co = _try_import_celloracle()

    # Build / reuse oracle. The user can stash one on `adata.uns` to skip
    # the (expensive) build step on subsequent calls.
    oracle = adata.uns.get("celloracle_oracle")
    if oracle is None:
        oracle = co.Oracle()
        oracle.import_anndata_as_normalized_count(
            adata=adata,
            cluster_column_name=backend_kwargs.pop("cluster_column_name", None),
            embedding_name=backend_kwargs.pop("embedding_name", "X_umap"),
        )
        # CellOracle requires a kNN-imputed expression layer
        # (``adata.layers['imputed_count']``) before simulate_shift can run.
        # Run the standard PCA + kNN imputation pipeline now unless the
        # caller already did it themselves.
        if (
            "imputed_count" not in getattr(oracle.adata, "layers", {})
            and hasattr(oracle, "perform_PCA")
            and hasattr(oracle, "knn_imputation")
        ):
            oracle.perform_PCA()
            n_cells = oracle.adata.shape[0]
            k = backend_kwargs.pop("knn_k", max(4, min(8, n_cells - 1)))
            oracle.knn_imputation(
                n_pca_dims=backend_kwargs.pop("n_pca_dims", 20),
                k=k,
                balanced=False,
                b_sight=backend_kwargs.pop("b_sight", min(max(n_cells // 4, k * 2), 200)),
                b_maxl=backend_kwargs.pop("b_maxl", min(max(n_cells // 10, k), 50)),
                n_jobs=backend_kwargs.pop("knn_n_jobs", 4),
            )
        if grn_base is None:
            grn_base = adata.uns.get("base_grn")
        if grn_base is None:
            grn_base = adata.uns.get("celloracle_base_grn")
        if grn_base is None:
            raise ValueError(
                "CellOracle backend needs a base GRN. Pass `grn_base=` or "
                "stash one at adata.uns['base_grn']; for human/mouse you "
                "can use `co.data.load_human_promoter_base_GRN()` or "
                "`co.data.load_mouse_promoter_base_GRN()`."
            )
        oracle.import_TF_data(TF_info_matrix=grn_base)
        oracle.fit_GRN_for_simulation(**backend_kwargs)

    # Build the per-target value dict.
    value_dict: dict[str, float] = {}
    for g in targets:
        x = adata[:, g].X
        if hasattr(x, "toarray"):
            x = x.toarray()
        base = float(np.asarray(x).mean())
        if mode == "ko":
            value_dict[g] = 0.0
        elif mode == "kd":
            value_dict[g] = base / fold_change
        elif mode == "oe":
            value_dict[g] = base * fold_change

    oracle.simulate_shift(
        perturb_condition=value_dict,
        n_propagation=n_propagation,
    )

    # CellOracle computes the cell→cell transition probability matrix
    # in a separate step (estimate_transition_prob). Run it so
    # ``result.trajectory_shift`` is populated — guard for tiny demos.
    if hasattr(oracle, "estimate_transition_prob"):
        n_cells = oracle.adata.shape[0]
        n_neighbors = backend_kwargs.pop(
            "transition_n_neighbors", min(200, max(20, n_cells // 5))
        )
        sigma_corr = backend_kwargs.pop("transition_sigma_corr", 0.05)
        try:
            oracle.estimate_transition_prob(
                n_neighbors=n_neighbors,
                knn_random=True,
                sampled_fraction=1.0,
                threads=backend_kwargs.pop("transition_threads", 4),
            )
            if hasattr(oracle, "calculate_embedding_shift"):
                oracle.calculate_embedding_shift(sigma_corr=sigma_corr)
        except Exception as exc:  # pragma: no cover - transition_prob is optional
            warnings.warn(
                f"CellOracle estimate_transition_prob failed ({exc!r}); "
                "trajectory_shift will be None.",
                RuntimeWarning,
            )

    # CellOracle stores its inferred GRN as either:
    #   * ``oracle.coef_matrix``               (when GRN_unit='whole')
    #   * ``oracle.coef_matrix_per_cluster``   (when GRN_unit='cluster', default)
    # Propagation through the GRN changes expression — not edges — so
    # ``grn`` and ``grn_base`` reflect the inferred GRN, and ``delta_grn``
    # is empty for this backend (the user-facing change is in
    # ``delta_expr`` / ``adata_perturbed``).
    coef_whole = getattr(oracle, "coef_matrix", None)
    coef_per_cluster = getattr(oracle, "coef_matrix_per_cluster", None)
    if coef_whole is not None:
        inferred_coef = coef_whole
    elif coef_per_cluster:
        inferred_coef = _average_coef_matrices(coef_per_cluster)
    else:
        inferred_coef = None
    grn_post = _ensure_networkx(inferred_coef, var_names=adata.var_names)
    grn_pre = _celloracle_base_to_graph(grn_base)

    delta_grn = pd.DataFrame()

    # Pull delta-expression from the oracle's stored layers
    delta_expr = pd.DataFrame()
    if return_delta and hasattr(oracle, "adata"):
        try:
            base = oracle.adata.layers["imputed_count"].mean(axis=0)
            pert = oracle.adata.layers["simulated_count"].mean(axis=0)
            delta_expr = pd.DataFrame({
                "gene": oracle.adata.var_names,
                "mean_base": np.asarray(base).ravel(),
                "mean_pert": np.asarray(pert).ravel(),
            })
            delta_expr["delta"] = delta_expr["mean_pert"] - delta_expr["mean_base"]
            delta_expr["log2_fc"] = np.log2(
                (delta_expr["mean_pert"] + 1e-6) / (delta_expr["mean_base"] + 1e-6)
            )
        except (KeyError, AttributeError):  # pragma: no cover
            delta_expr = pd.DataFrame()

    transition_prob = getattr(oracle, "transition_prob", None)

    # Per-cell ΔX = simulated_count − imputed_count (CellOracle's native output).
    delta_X = None
    cell_names = None
    embedding = None
    if hasattr(oracle, "adata"):
        try:
            imp = oracle.adata.layers["imputed_count"]
            sim = oracle.adata.layers["simulated_count"]
            imp = imp.toarray() if hasattr(imp, "toarray") else np.asarray(imp)
            sim = sim.toarray() if hasattr(sim, "toarray") else np.asarray(sim)
            delta_X = np.asarray(sim - imp, dtype=np.float64)
            cell_names = list(oracle.adata.obs_names)
            emb_key = getattr(oracle, "embedding_name", "X_umap")
            if emb_key in oracle.adata.obsm:
                embedding = np.asarray(oracle.adata.obsm[emb_key])
        except (KeyError, AttributeError):  # pragma: no cover
            pass

    return PerturbResult(
        target=targets[0] if len(targets) == 1 else list(targets),
        mode=mode,
        backend="cell_oracle",
        adata_perturbed=getattr(oracle, "adata", None),
        grn=grn_post if grn_output else None,
        grn_base=grn_pre if grn_output else None,
        delta_grn=delta_grn,
        delta_expr=delta_expr,
        trajectory_shift=transition_prob,
        delta_X=delta_X,
        cell_names=cell_names,
        gene_names=list(oracle.adata.var_names) if hasattr(oracle, "adata") else None,
        embedding=embedding,
        meta={"library": "celloracle", "n_propagation": n_propagation,
              "n_cells": adata.n_obs, "oracle": oracle},
    )


def _try_import_celloracle():
    try:
        import celloracle  # type: ignore
        return celloracle
    except ImportError as exc:  # pragma: no cover
        raise build_optional_dependency_error(
            feature="ov.single.perturb (backend='cell_oracle')",
            dependencies=("celloracle",),
            install_hint="pip install celloracle",
        ) from exc






def _try_import_dynamo():
    try:
        import dynamo
        return dynamo
    except ImportError as exc:
        raise build_optional_dependency_error(
            feature="ov.single.perturb (backend='dynamo')",
            dependencies=("dynamo-release",),
            install_hint="pip install 'dynamo-release<=1.5.3'",
        ) from exc


def _dynamo_pca_ready(adata, options=None) -> bool:
    """Whether the selected PCA coordinates and reconstruction metadata exist."""
    options = options or {}
    pca_key = options.get("pca_key")
    pcs_key = options.get("PCs_key")
    means_key = options.get("pca_mean_key")
    return (
        (options.get("X_pca") is not None
         or (pca_key if pca_key is not None else "X_pca") in adata.obsm)
        and (isinstance(pcs_key, np.ndarray) or (pcs_key or "PCs") in adata.uns)
        and (isinstance(means_key, np.ndarray) or (means_key or "pca_mean") in adata.uns)
        and "use_for_pca" in adata.var.columns
    )


def _dynamo_neighbors_ready(adata) -> bool:
    """Whether dynamo can retrieve the default neighborhood graph."""
    return (
        "indices" in adata.uns.get("neighbors", {})
        or "distances" in adata.obsp
    )


def _dynamo_ready(adata, emb_basis="umap", options=None, method="j_delta_x") -> bool:
    """Whether the selected model and its projection prerequisites exist."""
    options = options or {}
    field_key = f"VecFld_{options.get('basis', 'pca')}"
    cached_jacobian = options.get("jac_key", "jacobian_pca") in adata.uns
    return (
        _dynamo_pca_ready(adata, options)
        and (field_key in adata.uns or (method.lower().startswith("j_") and cached_jacobian))
        and "dynamics" in adata.uns
        and _dynamo_neighbors_ready(adata)
        and f"X_{emb_basis}" in adata.obsm
    )


def _run_dynamo_pipeline(adata, dyn, *, emb_basis="umap", targets=()) -> None:
    """Prepare missing PCA, dynamics, neighbors, embedding and vector field.

    Velocity estimation requires spliced/unspliced or metabolic-labeling
    layers; an expression matrix alone is insufficient. Existing PCA and
    vector fields are reused when only the graph or embedding is missing.
    The Monocle preprocessor preserves input gene identifiers and includes
    targets among PCA features without fetching annotations. Kinetic data
    with separate splicing and labeling measurements use dynamo's two-step
    estimator to avoid the 1.5.3 direct-mixture species mismatch.
    """
    needs_pca = not _dynamo_pca_ready(adata)
    needs_dynamics = needs_pca or "dynamics" not in adata.uns
    if needs_dynamics:
        velocity_layers = (
            {"spliced", "unspliced"}, {"new", "total"}, {"uu", "ul", "su", "sl"},
            {"X_spliced", "X_unspliced"}, {"X_new", "X_total"},
            {"X_uu", "X_ul", "X_su", "X_sl"},
        )
        if not any(keys.issubset(adata.layers) for keys in velocity_layers):
            raise RuntimeError(
                "auto_pipeline=True requires spliced/unspliced or metabolic-labeling "
                "layers for dynamo velocity estimation; adata.X counts alone are "
                "insufficient. Supply velocity-compatible data or use backend='sctenifoldknk'."
            )
    if needs_pca:
        preprocessor = dyn.pp.Preprocessor(
            convert_gene_name_function=lambda data: data,
            gene_append_list=list(targets),
        )
        preprocessor.preprocess_adata(adata, recipe="monocle")
    if needs_dynamics:
        preprocessing = adata.uns.get("pp", {})
        dynamics_kwargs = {}
        if (preprocessing.get("experiment_type") == "kin"
                and preprocessing.get("has_splicing")
                and not preprocessing.get("splicing_labeling")):
            dynamics_kwargs["est_method"] = "twostep"
        dyn.tl.dynamics(adata, **dynamics_kwargs)
    if needs_pca or not _dynamo_neighbors_ready(adata):
        dyn.tl.neighbors(adata, X_data=adata.obsm["X_pca"], basis="pca")
    if f"X_{emb_basis}" not in adata.obsm:
        dyn.tl.reduceDimension(
            adata, X_data=adata.obsm["X_pca"], basis="pca",
            n_pca_components=adata.obsm["X_pca"].shape[1], reduction_method=emb_basis,
        )
    if needs_dynamics or "VecFld_pca" not in adata.uns:
        dyn.tl.cell_velocities(adata, basis="pca", enforce=True)
        dyn.vf.VectorField(adata, basis="pca")
        adata.uns.pop("jacobian_pca", None)


def _run_dynamo(
    adata,
    *,
    targets: Sequence[str],
    mode: str,
    fold_change: float,
    layer: str | None,
    grn_output: bool,
    return_delta: bool,
    backend_kwargs: dict,
) -> PerturbResult:
    """dynamo backend — vector-field in-silico perturbation (Qiu 2022).

    Wraps :func:`dynamo.pd.perturbation`: the target's expression is set
    (KO → 0, KD → mean/``fold_change``, OE → mean×``fold_change``) and the
    default ``j_delta_x`` method propagates the perturbation vector ``δX``
    through the **analytical Jacobian** of the learned PCA vector field,

        ``δY = J · δX``,

    giving per-cell, per-gene response (``adata.layers[perturbation_key]``)
    and a perturbation transition matrix
    (``adata.obsp['perturbation_transition_matrix']``) — mapped onto
    ``PerturbResult.delta_X`` / ``.trajectory_shift`` so every downstream
    method and ``ov.pl.perturb_*`` helper works unchanged.

    Backend kwargs (parsed here before dispatch):
        ``emb_basis`` (default ``'umap'``), ``perturb_mode``
        (``'raw'`` | ``'z_score'``), ``perturbation_method``
        (``'j_delta_x'`` | ``'j_x_prime'`` | ``'j_jv'`` |
        ``'f_x_prime'`` | ``'f_x_prime_minus_f_x_0'``),
        ``auto_pipeline`` (run the dynamo recipe when prerequisites are
        missing — default ``False``), ``vf_ko`` (additionally knock the
        genes out of the vector field itself via ``dynamo.pd.KO``),
        ``group`` (``adata.obs`` column for dynamo's per-group mean signed
        and mean absolute per-cell response rankings, limited to dynamics
        genes). ``vf_ko`` adds native KO velocities; shared response and
        transition outputs follow the selected perturbation method, which
        uses the Jacobian by default. The method also
        accepts dynamo's spelling
        ``pertubation_method``. Projection and perturbation options above
        are passed explicitly to ``dynamo.pd.perturbation``.
        ``add_delta_Y_key``, ``add_transition_key``, ``add_velocity_key`` and
        ``add_embedding_key`` override the response, transition, velocity
        and embedding output keys; resolved keys are forwarded and used
        for result extraction and metadata. With ``vf_ko=True`` in KO mode,
        native KO coordinate/velocity keys and ``pearson_transition_matrix``
        are reserved; conflicting keys are rejected before data mutation.
        ``expression`` optionally overrides generated levels with a finite
        scalar or one value per unique target in caller order, in the units
        of ``perturb_mode``. Selected PCA keys/arrays are respected;
        automatic preparation only fits the default PCA model.
        Remaining kwargs are forwarded unchanged. The caller's dictionary
        is not modified. The 1.5.3 precomputed ``delta_Y`` path and unsafe
        target-zeroing indices are rejected before perturbation. Use
        ``X_pca`` for coordinate arrays, not ``pca_key``.
        ``projection_method='transform'`` does not produce a transition
        matrix, so ``trajectory_shift`` is ``None`` for that method.
        In 1.5.3 it may project baseline velocities instead of the response;
        a warning recommends ``projection_method='fp'``.

    In ``z_score`` mode, generated expression levels are converted using
    the PCA-reconstructed gene distribution used by dynamo itself. Targets
    must all be PCA genes and are aligned to PCA order before dispatch.
    Automatic preprocessing preserves input gene identifiers. Kinetic data
    with separate splicing and labeling layers use ``est_method='twostep'``;
    the actual estimator is recorded in ``meta['dynamics_est_method']``.
    """
    dyn = _try_import_dynamo()

    if backend_kwargs.get("delta_Y") is not None:
        raise NotImplementedError(
            "delta_Y is not supported by dynamo-release 1.5.3 perturbation: "
            "its precomputed-response path leaves PCA loadings undefined."
        )
    if backend_kwargs.get("pca_key") is not None and not isinstance(backend_kwargs["pca_key"], str):
        raise TypeError("pca_key must be a string; pass coordinate arrays through X_pca instead.")
    if backend_kwargs.get("projection_method") == "transform":
        warnings.warn(
            "dynamo-release 1.5.3 transform projection reloads baseline velocities and "
            "may ignore the perturbation response. Use projection_method='fp' for "
            "perturbation-driven transitions and arrows.", RuntimeWarning, stacklevel=2,
        )
    emb_basis = backend_kwargs.pop("emb_basis", "umap")
    perturb_mode = backend_kwargs.pop("perturb_mode", "raw")

    method = backend_kwargs.pop(
        "pertubation_method",
        backend_kwargs.pop("perturbation_method", "j_delta_x"),
    )
    auto_pipeline = backend_kwargs.pop("auto_pipeline", False)
    run_vf_ko = backend_kwargs.pop("vf_ko", False)
    rank_group = backend_kwargs.pop("group", None)
    requested_expression = backend_kwargs.pop("expression", None)
    output_defaults = {
        "add_delta_Y_key": f"{method}_perturbation",
        "add_transition_key": "perturbation_transition_matrix",
        "add_velocity_key": f"velocity_{emb_basis}_perturbation",
        "add_embedding_key": f"X_{emb_basis}_perturbation",
    }
    for key, default in output_defaults.items():
        if backend_kwargs.get(key) is None:
            backend_kwargs[key] = default
    if run_vf_ko and mode == "ko":
        reserved_obsm = {
            "X_pca_KO", "velocity_pca_KO",
            f"X_{emb_basis}_KO", f"velocity_{emb_basis}_KO",
        }
        collisions = {
            key: backend_kwargs[key]
            for key in ("add_delta_Y_key", "add_velocity_key", "add_embedding_key")
            if backend_kwargs[key] in reserved_obsm
        }
        if backend_kwargs["add_transition_key"] == "pearson_transition_matrix":
            collisions["add_transition_key"] = "pearson_transition_matrix"
        if collisions:
            raise ValueError(
                f"vf_ko=True output keys collide with dynamo KO outputs: {collisions}. "
                "Choose distinct add_* output keys for the perturbation response."
            )
    pkey = backend_kwargs["add_delta_Y_key"]
    transition_key = backend_kwargs["add_transition_key"]
    velocity_key = backend_kwargs["add_velocity_key"]

    if backend_kwargs.get("X_pca") is not None:
        if backend_kwargs.get("PCs_key") is None:
            backend_kwargs["PCs_key"] = "PCs"
        if backend_kwargs.get("pca_mean_key") is None:
            backend_kwargs["pca_mean_key"] = "pca_mean"
    needs_ko_model = run_vf_ko and mode == "ko" and (
        not _dynamo_pca_ready(adata) or "VecFld_pca" not in adata.uns
    )
    if needs_ko_model and not auto_pipeline:
        raise RuntimeError(
            "vf_ko=True requires the default PCA model (X_pca, PCs, pca_mean) "
            "and VecFld_pca. A cached Jacobian or custom PCA alone is insufficient. "
            "Prepare the native KO model or set vf_ko=False."
        )
    if not _dynamo_ready(adata, emb_basis, backend_kwargs, method) or needs_ko_model:
        if not auto_pipeline:
            raise RuntimeError(
                "backend='dynamo' needs the selected PCA coordinates/loadings/means "
                "(pca_key or X_pca, PCs_key, pca_mean_key; defaults: X_pca, PCs, "
                "pca_mean), var['use_for_pca'], a vector field or cached Jacobian, "
                "uns['dynamics'], a neighbors graph and "
                f"obsm['X_{emb_basis}']. For unprocessed velocity-compatible "
                "data use dyn.pp.Preprocessor with gene-name conversion disabled "
                "and recipe='monocle' → dyn.tl.dynamics (est_method='twostep' "
                "for kinetic data with separate splicing/labeling) → "
                "dyn.tl.neighbors(basis='pca') → "
                f"dyn.tl.reduceDimension(reduction_method='{emb_basis}') → "
                "dyn.tl.cell_velocities(basis='pca') → "
                "dyn.vf.VectorField(basis='pca'). On an existing vector field, "
                "prepare only the missing graph/embedding. Alternatively pass "
                "backend_kwargs={'auto_pipeline': True} to prepare missing "
                "steps in place; estimating velocities requires spliced/unspliced "
                "or metabolic-labeling layers, not adata.X counts alone."
            )
        custom_pca = any(
            isinstance(backend_kwargs.get(key), np.ndarray)
            or backend_kwargs.get(key) not in (None, default)
            for key, default in (("pca_key", "X_pca"), ("PCs_key", "PCs"),
                                 ("pca_mean_key", "pca_mean"), ("basis", "pca"))
        ) or backend_kwargs.get("X_pca") is not None
        if custom_pca:
            raise RuntimeError(
                "Automatic preparation fits the default PCA model. Prepare the selected "
                "custom PCA, dynamics, neighbors, embedding and vector field explicitly."
            )
        _run_dynamo_pipeline(adata, dyn, emb_basis=emb_basis, targets=targets)

    pca_genes = adata.var_names[adata.var.use_for_pca]
    missing_pca = [g for g in targets if g not in pca_genes]
    if missing_pca:
        raise ValueError(
            f"dynamo targets must all be PCA genes (var['use_for_pca']): {missing_pca}. "
            "Recompute PCA and the vector field including these targets."
        )
    ordered_targets = list(pca_genes.intersection(targets))
    if backend_kwargs.get("zero_perturb_genes_vel") and any(
        pca_genes.get_loc(g) != adata.var_names.get_loc(g) for g in ordered_targets
    ):
        raise ValueError(
            "zero_perturb_genes_vel would zero incorrect full-gene columns in "
            "dynamo-release 1.5.3 for these PCA indices. Leave it False and "
            "select target columns explicitly when inspecting responses."
        )
    X = _expression_matrix(adata, layer=layer)
    var_names = list(adata.var_names)
    requested_levels = None
    if requested_expression is not None:
        if np.isscalar(requested_expression):
            requested_levels = {g: float(requested_expression) for g in ordered_targets}
        else:
            levels = np.asarray(requested_expression, dtype=float)
            if levels.ndim != 1 or len(levels) != len(targets) or len(set(targets)) != len(targets):
                raise ValueError("expression must be a scalar or one value per unique target, in target order.")
            requested_levels = dict(zip(targets, levels))
        if not all(np.isfinite(value) for value in requested_levels.values()):
            raise ValueError("expression values must be finite.")
    reconstructed = None
    if perturb_mode == "z_score" and requested_levels is None:
        pca_key = backend_kwargs.get("pca_key") or "X_pca"
        pcs_key = backend_kwargs.get("PCs_key")
        means_key = backend_kwargs.get("pca_mean_key")
        pcs = pcs_key if isinstance(pcs_key, np.ndarray) else adata.uns[pcs_key or "PCs"]
        means = means_key if isinstance(means_key, np.ndarray) else adata.uns[means_key or "pca_mean"]
        coordinates = backend_kwargs.get("X_pca")
        if coordinates is None:
            coordinates = adata.obsm[pca_key]
        positions = pca_genes.get_indexer(ordered_targets)
        selected_means = np.asarray(means)
        if selected_means.ndim:
            selected_means = selected_means[positions]
        reconstructed = np.asarray(coordinates @ pcs[positions].T) + selected_means

    expression: list[float] = []
    for i, g in enumerate(ordered_targets):
        if requested_levels is not None:
            expression.append(float(requested_levels[g]))
            continue
        base = float(np.asarray(X[:, var_names.index(g)]).mean())
        if mode == "ko":
            value = 0.0
        elif mode == "kd":
            value = base / fold_change
        else:
            value = base * fold_change
        if perturb_mode == "z_score":
            x = reconstructed[:, i]
            m, sd = float(x.mean()), float(x.std())
            if not np.isfinite(sd) or sd == 0:
                raise ValueError(
                    f"PCA-reconstructed target {g!r} has zero or non-finite variance; "
                    "use perturb_mode='raw' instead."
                )
            value = (value - m) / sd
        expression.append(float(value))

    dyn.pd.perturbation(
        adata,
        genes=ordered_targets,
        expression=expression,
        perturb_mode=perturb_mode,
        emb_basis=emb_basis,
        pertubation_method=method,
        **backend_kwargs,
    )


    dX = adata.layers[pkey]
    delta_X = dX.toarray() if hasattr(dX, "toarray") else np.asarray(dX)
    delta_X = np.asarray(delta_X, dtype=np.float64)
    cell_names = list(adata.obs_names)
    gene_names = list(adata.var_names)


    delta_expr = pd.DataFrame()
    if return_delta:
        mean_base = np.asarray(X.mean(axis=0)).ravel()
        mean_delta = delta_X.mean(axis=0)
        delta_expr = pd.DataFrame({
            "gene": gene_names,
            "mean_base": mean_base,
            "mean_pert": mean_base + mean_delta,
        })
        delta_expr["delta"] = delta_expr["mean_pert"] - delta_expr["mean_base"]
        defined = (
            np.isfinite(mean_base) & np.isfinite(delta_expr["mean_pert"])
            & (mean_base >= 0) & (delta_expr["mean_pert"] >= 0)
        )
        delta_expr["log2_fc"] = np.nan
        delta_expr.loc[defined, "log2_fc"] = np.log2(
            (delta_expr.loc[defined, "mean_pert"] + 1e-6)
            / (delta_expr.loc[defined, "mean_base"] + 1e-6)
        )


    transition_prob = (
        None if backend_kwargs.get("projection_method", "fp") == "transform"
        else adata.obsp.get(transition_key, None)
    )
    if transition_prob is not None:
        transition_prob = (
            transition_prob.toarray()
            if hasattr(transition_prob, "toarray")
            else np.asarray(transition_prob)
        )

    embedding = (
        np.asarray(adata.obsm[f"X_{emb_basis}"])
        if f"X_{emb_basis}" in adata.obsm
        else None
    )


    rank_key = None
    if rank_group is not None:
        if rank_group not in adata.obs:
            raise ValueError(f"group {rank_group!r} not in adata.obs")
        dyn.pd.rank_perturbation_genes(adata, pkey=pkey, groups=rank_group)
        rank_key = f"rank_{pkey}"


    vf_ko_keys: dict = {}
    if run_vf_ko:
        if mode != "ko":
            warnings.warn(
                "vf_ko=True only makes sense with mode='ko'; skipping the "
                "vector-field knockout.", RuntimeWarning,
            )
        else:
            dyn.pd.KO(adata, KO_genes=list(targets), emb_basis=emb_basis)
            vf_ko_keys = {
                "pca": "velocity_pca_KO",
                "emb_basis": f"velocity_{emb_basis}_KO",
            }

    return PerturbResult(
        target=targets[0] if len(targets) == 1 else list(targets),
        mode=mode,
        backend="dynamo",
        adata_perturbed=adata,
        grn=None,
        grn_base=None,
        delta_grn=pd.DataFrame(),
        delta_expr=delta_expr,
        trajectory_shift=transition_prob,
        delta_X=delta_X,
        cell_names=cell_names,
        gene_names=gene_names,
        embedding=embedding,
        meta={
            "library": "dynamo",
            "perturbation_key": pkey,
            "perturbation_method": method,
            "perturb_mode": perturb_mode,
            "emb_basis": emb_basis,
            "expression": dict(zip(ordered_targets, expression)),
            "expression_source": "native" if requested_levels is not None else "mode",
            "transition_key": transition_key,
            "velocity_embedding_key": velocity_key,
            "embedding_key": backend_kwargs["add_embedding_key"],
            "rank_key": rank_key,
            "vf_ko_keys": vf_ko_keys,
            "n_cells": adata.n_obs,
            "dynamics_est_method": adata.uns.get("dynamics", {}).get("est_method"),
        },
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_inputs(result, *, adata=None, embedding_name: str = "X_umap"):
    """Pull ``delta_X``, ``X``, ``embedding`` from a result + (optional) adata.

    The downstream methods on :class:`PerturbResult` all reduce to these
    four arrays, so we centralise the lookup.
    """
    if result.delta_X is None:
        raise ValueError(
            "PerturbResult has no per-cell delta_X. "
            "Re-run perturb() with this version of omicverse, or call "
            "result.compute_transition_prob(adata=...) once delta_X has "
            "been populated."
        )
    delta_X = np.asarray(result.delta_X)

    # Expression matrix
    if adata is not None:
        X = _expression_matrix(adata, layer=None)
        if X.shape != delta_X.shape:
            # subset adata to the gene columns of delta_X
            gene_names = list(result.gene_names) if result.gene_names is not None else None
            if gene_names is not None and all(g in adata.var_names for g in gene_names):
                X = _expression_matrix(adata[:, gene_names], layer=None)
    else:
        ap = result.adata_perturbed
        if ap is None:
            raise ValueError(
                "Need an `adata=` (or result.adata_perturbed) to read X."
            )
        X = _expression_matrix(ap, layer=None)

    # Embedding
    embedding = None
    if adata is not None and embedding_name in adata.obsm:
        embedding = np.asarray(adata.obsm[embedding_name])
    elif result.embedding is not None:
        embedding = np.asarray(result.embedding)
    elif result.adata_perturbed is not None and embedding_name in result.adata_perturbed.obsm:
        embedding = np.asarray(result.adata_perturbed.obsm[embedding_name])

    gene_names = list(result.gene_names) if result.gene_names is not None else None
    return delta_X, X, embedding, gene_names


def compute_transition_prob(
    delta_X,
    X,
    embedding,
    *,
    n_neighbors: int = 30,
    sigma_corr: float = 0.05,
    n_jobs: int = 4,
):
    """Cell × cell transition probability from per-cell ΔX.

    Implements the CellOracle / velocyto-style correlation kernel
    (Kamimoto 2023 Methods, La Manno 2018): for each cell ``i`` and each
    of its ``n_neighbors`` neighbours ``j`` in the embedding,

        corr[i,j] = cos( ΔX[i] , X[j] − X[i] )
        T[i,j]    = exp(corr[i,j] / σ)     and is row-normalised.

    The output is a dense ``(n_cells, n_cells)`` matrix that's sparse in
    structure (only k entries per row are non-zero).
    """
    delta_X = np.ascontiguousarray(np.asarray(delta_X), dtype=np.float64)
    X = np.ascontiguousarray(np.asarray(X), dtype=np.float64)
    embedding = np.ascontiguousarray(np.asarray(embedding), dtype=np.float64)
    n_cells = X.shape[0]
    if delta_X.shape != X.shape:
        raise ValueError(
            f"delta_X{delta_X.shape} must match X{X.shape} (cells × genes)"
        )
    if embedding.shape[0] != n_cells:
        raise ValueError(
            f"embedding{embedding.shape} must have {n_cells} rows"
        )

    from sklearn.neighbors import NearestNeighbors  # local import
    k = min(n_neighbors + 1, n_cells)
    nn = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs)
    nn.fit(embedding)
    _, neigh_ixs = nn.kneighbors(embedding)

    # Normalise ΔX rows once
    norm_dx = np.linalg.norm(delta_X, axis=1) + 1e-12
    dx_unit = delta_X / norm_dx[:, None]

    tp = np.zeros((n_cells, n_cells), dtype=np.float64)
    for i in range(n_cells):
        nbrs = neigh_ixs[i, 1:]  # drop self
        diffs = X[nbrs] - X[i]   # (k, n_genes)
        norm_diff = np.linalg.norm(diffs, axis=1) + 1e-12
        diff_unit = diffs / norm_diff[:, None]
        corr = diff_unit @ dx_unit[i]
        weights = np.exp(corr / sigma_corr)
        tp[i, nbrs] = weights
        s = tp[i].sum()
        if s > 0:
            tp[i] /= s
    return tp


def _local_pseudotime_gradient(embedding, pseudotime, *, n_neighbors: int = 30):
    """Per-cell 2-D pseudotime gradient via local linear regression.

    For each cell, fit ``pt ~ emb_x + emb_y`` on its ``n_neighbors``
    nearest neighbours; the regression slopes (β_x, β_y) form the local
    pseudotime gradient vector. This is the same construction
    CellOracle's ``Gradient_calculator`` uses.
    """
    from sklearn.neighbors import NearestNeighbors
    embedding = np.asarray(embedding, dtype=np.float64)
    pt = np.asarray(pseudotime, dtype=np.float64)
    n_cells = embedding.shape[0]
    k = min(n_neighbors + 1, n_cells)
    nn = NearestNeighbors(n_neighbors=k).fit(embedding)
    _, neigh = nn.kneighbors(embedding)

    grad = np.zeros_like(embedding)
    for i in range(n_cells):
        ix = neigh[i]
        e = embedding[ix]
        p = pt[ix]
        if np.allclose(p, p[0]):
            continue
        # center and regress (least squares)
        ec = e - e.mean(axis=0)
        pc = p - p.mean()
        # β = (E^T E)^-1 E^T p  ; 2-D so 2x2 inverse
        EtE = ec.T @ ec
        try:
            beta = np.linalg.solve(EtE + 1e-9 * np.eye(2), ec.T @ pc)
        except np.linalg.LinAlgError:  # pragma: no cover
            continue
        grad[i] = beta
    return grad


def _compute_ps_grid(
    *,
    embedding,
    delta_emb,
    pseudotime,
    grid_size: int = 30,
    min_mass: float = 1.0,
    n_neighbors: int = 30,
):
    """CellOracle-style PS computation on a 2-D grid.

    Replicates the construction in
    ``celloracle.applications.Oracle_development_module``:

      * ``flow_grid``      — simulation flow vectors aggregated onto a
        regular grid via a Gaussian neighbourhood kernel.
      * ``ref_flow_grid``  — pseudotime gradient vectors aggregated the
        same way (CellOracle's ``Gradient_calculator`` ref_flow).
      * ``ps_grid[g]``     = ``flow_grid[g] · ref_flow_grid[g]``
        (a raw dot product, **not** cosine similarity).

    Returns a dict with the grid arrays so callers can reuse them for
    both the heatmap (``perturb_inner_product_on_grid``) and the
    per-cell lookup (``perturbation_score(level='cell')``).
    """
    from scipy.spatial import cKDTree
    from sklearn.neighbors import KNeighborsRegressor
    embedding = np.asarray(embedding, dtype=np.float64)
    delta_emb = np.asarray(delta_emb, dtype=np.float64)
    pt = np.asarray(pseudotime, dtype=np.float64)

    # Grid construction
    xs = np.linspace(embedding[:, 0].min(), embedding[:, 0].max(), grid_size)
    ys = np.linspace(embedding[:, 1].min(), embedding[:, 1].max(), grid_size)
    GX, GY = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])

    # ---------- ref_flow on grid (CellOracle Gradient_calculator) -----------
    # CellOracle's pipeline:
    #   1. transfer_data_into_grid: KNN-regress pseudotime onto each grid point
    #      so we get a 2-D scalar field pseudotime_on_grid[grid].
    #   2. calculate_gradient: np.gradient on the (grid_size × grid_size)
    #      reshape gives finite-difference (dx, dy) at every grid point.
    #   3. Normalise / scale (we just normalise — scaling is up to the plot).
    knn = KNeighborsRegressor(n_neighbors=min(n_neighbors, embedding.shape[0]))
    knn.fit(embedding, pt)
    pt_on_grid = knn.predict(grid_pts)
    pt_on_grid_2d = pt_on_grid.reshape(grid_size, grid_size)
    dy, dx = np.gradient(pt_on_grid_2d)
    ref_flow_grid = np.stack([dx.ravel(), dy.ravel()], axis=1)
    # Normalise component-wise by the mean L2 norm so |arrows| are O(1)
    rfg_norms = np.linalg.norm(ref_flow_grid, axis=1)
    if rfg_norms.mean() > 0:
        ref_flow_grid = ref_flow_grid / rfg_norms.mean()

    # ---------- flow_grid (KO-simulation arrows) ----------------------------
    tree = cKDTree(embedding)
    radius = float(np.linalg.norm([xs[1] - xs[0], ys[1] - ys[0]]))
    flow_grid = np.zeros_like(grid_pts)
    mass = np.zeros(grid_pts.shape[0])
    for g_i, p in enumerate(grid_pts):
        ix = tree.query_ball_point(p, r=radius * 1.5)
        if not ix:
            continue
        d = np.linalg.norm(embedding[ix] - p, axis=1)
        w = np.exp(-(d ** 2) / (2 * (radius / 2) ** 2))
        mass[g_i] = w.sum()
        if mass[g_i] > 0:
            flow_grid[g_i] = (delta_emb[ix] * w[:, None]).sum(axis=0) / mass[g_i]
    keep = mass >= min_mass

    # CellOracle's PS = raw dot product per grid point
    ps_grid = (flow_grid * ref_flow_grid).sum(axis=1)

    return dict(
        grid_pts=grid_pts,
        flow_grid=flow_grid,
        ref_flow_grid=ref_flow_grid,
        ps_grid=ps_grid,
        pseudotime_on_grid=pt_on_grid,
        mass=mass,
        keep=keep,
        grid_size=grid_size,
    )


def _permutation_significance(
    delta_X,
    X,
    *,
    n_perms: int = 100,
    random_state: int = 0,
):
    """Per-gene significance for the per-cell ΔX matrix.

    For each gene, the Z-statistic measures whether ``ΔX[:, g]`` is
    consistently non-zero across cells:

        Z[g] = mean(ΔX[:, g]) / SE(ΔX[:, g])
             = mean(ΔX[:, g]) / (std(ΔX[:, g]) / sqrt(n_cells))

    This is the equivalent of a one-sample t-statistic against zero —
    "do most cells shift in the same direction for gene g?" — and is
    the per-cell-consistency scoring used by CellOracle's downstream
    `evaluate_simulated_gene_distribution_range` workflow.

    A permutation null (``n_perms``) is then built by sign-flipping the
    rows of ΔX, which preserves the per-cell magnitude but destroys the
    cross-cell direction agreement. The two-sided empirical p is the
    fraction of permutations with |Z_null| ≥ |Z_obs|, BH-adjusted.
    """
    from scipy.stats import norm as _norm
    rng = np.random.default_rng(random_state)
    delta_X = np.asarray(delta_X, dtype=np.float64)
    n_cells, n_genes = delta_X.shape

    mu_obs = delta_X.mean(axis=0)
    sd_obs = delta_X.std(axis=0) + 1e-12
    z_obs = mu_obs / (sd_obs / np.sqrt(n_cells))

    # Parametric two-sided p-value from the normal Z (avoids the n_perms cap
    # at min p ≈ 1/(n_perms+1)). When n_perms > 0 we also blend in the
    # empirical right-tail fraction for the largest |Z| values.
    p_param = 2.0 * (1.0 - _norm.cdf(np.abs(z_obs)))

    if n_perms > 0:
        extreme = np.zeros(n_genes, dtype=np.int64)
        abs_z_obs = np.abs(z_obs)
        for _ in range(n_perms):
            signs = rng.choice([-1.0, 1.0], size=n_cells)
            dX_null = delta_X * signs[:, None]
            mu_null = dX_null.mean(axis=0)
            z_null = mu_null / (sd_obs / np.sqrt(n_cells))
            extreme += np.abs(z_null) >= abs_z_obs
        p_emp = (extreme + 1) / (n_perms + 1)
        # For very-significant genes use parametric p; for the borderline /
        # null genes use the empirical p (which is conservative).
        p_val = np.minimum(p_param, p_emp)
    else:
        p_val = p_param
    # BH-FDR
    order = np.argsort(p_val)
    ranked = p_val[order]
    n = len(ranked)
    adj = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adj_p = np.empty_like(adj)
    adj_p[order] = np.clip(adj, 0.0, 1.0)
    return z_obs, p_val, adj_p


def _expression_matrix(adata, layer: str | None):
    """Return a numpy expression matrix from ``adata[, layer]``."""
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X
    arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return arr


def _tensor_to_graph(tensor, *, gene_names: Iterable[str]):
    """Wrap a 2D gene×gene tensor (numpy.ndarray) into a networkx.DiGraph.

    scTenifoldKnk returns its WT / KO PCNet as ``numpy.ndarray`` of shape
    ``(n_genes, n_genes)`` with ``shared_gene_names`` for the row / col
    index. We build a thresholded DiGraph (drop zero / near-zero edges)
    so the result is usable directly with networkx.draw().
    """
    if tensor is None:
        return None
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise build_optional_dependency_error(
            feature="ov.single.perturb (GRN output)",
            dependencies=("networkx",),
            install_hint="pip install networkx",
        ) from exc
    arr = np.asarray(tensor)
    genes = list(gene_names)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] != len(genes):
        # fall back to the generic adjacency loader
        return _ensure_networkx(arr, var_names=genes)
    # Threshold at max(|arr|)/200 to drop the dense low-weight noise
    # scTenifoldKnk emits; the visible structure is in the strongest 1-5%
    # of edges anyway.
    abs_arr = np.abs(arr)
    if abs_arr.max() == 0:
        thresh = 0.0
    else:
        thresh = abs_arr.max() / 200.0
    rows, cols = np.where(abs_arr > thresh)
    G = nx.DiGraph()
    for g in genes:
        G.add_node(g)
    for r, c in zip(rows.tolist(), cols.tolist()):
        if r == c:
            continue
        G.add_edge(genes[r], genes[c], weight=float(arr[r, c]))
    return G


def _ensure_networkx(graph_like, *, var_names: Iterable[str] | None = None):
    """Coerce a network (DataFrame / ndarray / DiGraph / None) into a
    :class:`networkx.DiGraph`.

    Imports networkx lazily so the module loads when networkx is missing
    (only the perturb call would then fail).
    """
    if graph_like is None:
        return None
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise build_optional_dependency_error(
            feature="ov.single.perturb (GRN output)",
            dependencies=("networkx",),
            install_hint="pip install networkx",
        ) from exc

    if hasattr(graph_like, "edges") and hasattr(graph_like, "nodes"):
        return graph_like
    if isinstance(graph_like, pd.DataFrame):
        # square TF×target weight matrix
        return nx.from_pandas_adjacency(graph_like, create_using=nx.DiGraph)
    arr = np.asarray(graph_like)
    if arr.ndim == 2 and var_names is not None and arr.shape[0] == arr.shape[1]:
        df = pd.DataFrame(arr, index=list(var_names), columns=list(var_names))
        return nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
    raise TypeError(f"cannot coerce {type(graph_like)!r} to a networkx graph")


def _average_coef_matrices(coef_per_cluster: dict) -> "pd.DataFrame | None":
    """Average CellOracle's per-cluster GRN coefficient matrices.

    ``oracle.coef_matrix_per_cluster`` is a ``{cluster: DataFrame}`` of TF×target
    coefficients (same row/col index per cluster). We mean across clusters to
    give a single population-level GRN suitable for a networkx graph.
    """
    if not coef_per_cluster:
        return None
    mats = list(coef_per_cluster.values())
    if not mats:
        return None
    # All DataFrames share the same index/columns; mean across them
    arr = np.stack([m.values for m in mats], axis=0).mean(axis=0)
    return pd.DataFrame(arr, index=mats[0].index, columns=mats[0].columns)


def _celloracle_base_to_graph(grn_base):
    """Convert a CellOracle base-GRN DataFrame into a ``networkx.DiGraph``.

    CellOracle base GRNs are wide-format: rows are target genes (with
    ``gene_short_name`` + ``peak_id`` columns) and one extra column per TF
    holding a 0/1 indicator that the TF regulates the row's target. We
    add an edge TF → target for every non-zero cell.
    """
    if grn_base is None:
        return None
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise build_optional_dependency_error(
            feature="ov.single.perturb (GRN output)",
            dependencies=("networkx",),
            install_hint="pip install networkx",
        ) from exc

    if hasattr(grn_base, "edges") and hasattr(grn_base, "nodes"):
        return grn_base
    if not isinstance(grn_base, pd.DataFrame):
        return _ensure_networkx(grn_base)

    df = grn_base
    if "gene_short_name" not in df.columns:
        # already an adjacency-like DataFrame
        return _ensure_networkx(df)

    meta_cols = {"gene_short_name", "peak_id"}
    tf_cols = [c for c in df.columns if c not in meta_cols]
    G = nx.DiGraph()
    for _, row in df.iterrows():
        target = row["gene_short_name"]
        for tf in tf_cols:
            w = row[tf]
            if w and float(w) != 0.0:
                G.add_edge(tf, target, weight=float(w))
    return G


def _apply_perturbation_to_graph(graph, *, targets, mode, fold_change):
    """Return a copy of ``graph`` with the perturbed edges adjusted."""
    if graph is None:
        return None
    import networkx as nx
    pert = graph.copy()
    for g in dict.fromkeys(targets):
        if g not in pert:
            continue
        edges = list(pert.out_edges(g, data=True))
        if mode == "ko":
            edges += list(pert.in_edges(g, data=True))
        for u, v, data in edges:
            w = float(data.get("weight", 1.0))
            if mode == "ko":
                w_new = 0.0
            elif mode == "kd":
                w_new = w / fold_change
            elif mode == "oe":
                w_new = w * fold_change
            else:  # pragma: no cover
                w_new = w
            pert[u][v]["weight"] = w_new
    return pert


def _diff_grn(grn_base, grn_pert) -> pd.DataFrame:
    """Return a long-format edge-weight diff table."""
    if grn_base is None or grn_pert is None:
        return pd.DataFrame()
    edges_b = {(u, v): float(d.get("weight", 1.0)) for u, v, d in grn_base.edges(data=True)}
    edges_p = {(u, v): float(d.get("weight", 1.0)) for u, v, d in grn_pert.edges(data=True)}
    keys = sorted(set(edges_b) | set(edges_p))
    if not keys:
        return pd.DataFrame()
    rows = []
    for u, v in keys:
        wb = edges_b.get((u, v), 0.0)
        wp = edges_p.get((u, v), 0.0)
        if wb == 0.0 and wp == 0.0:
            continue
        rows.append((u, v, wb, wp, wp - wb))
    return pd.DataFrame(rows, columns=["source", "target", "weight_base", "weight_pert", "delta"])


def _delta_from_grn(grn_base, grn_pert, *, targets, mode, fold_change) -> pd.DataFrame:
    """One-step propagation: estimate Δ-expression at each downstream
    node by summing the changed in-edge weights, normalised by the
    node's in-degree.

    Intentionally simple — for proper transcriptome-level prediction
    use the CellOracle backend (which propagates through the GRN
    `n_propagation` times).
    """
    if grn_base is None or grn_pert is None:
        return pd.DataFrame()
    import networkx as nx
    all_nodes = sorted(set(grn_base.nodes) | set(grn_pert.nodes))
    rows = []
    for g in all_nodes:
        in_b = sum(float(d.get("weight", 1.0)) for _, _, d in grn_base.in_edges(g, data=True))
        in_p = sum(float(d.get("weight", 1.0)) for _, _, d in grn_pert.in_edges(g, data=True))
        delta = in_p - in_b
        log2_fc = np.log2((in_p + 1e-6) / (in_b + 1e-6))
        rows.append((g, in_b, in_p, delta, log2_fc))
    df = pd.DataFrame(rows, columns=["gene", "mean_base", "mean_pert", "delta", "log2_fc"])
    return df

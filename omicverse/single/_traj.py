import scanpy as sc
import pandas as pd
import anndata
import numpy as np
import time
import numpy as np
from scipy.sparse import issparse, csr_matrix
import igraph
import torch
from tqdm import tqdm
import sys
from ..utils._neighboors import update_rep, eff_n_jobs,neighbors,W_from_rep
from .._settings import add_reference
from .._registry import register_function

from ._cosg import cosg


def _get_palantir_backend():
    from ..external.palantir.core import run_palantir
    from ..external.palantir.plot import plot_branch_selection, plot_gene_trends, plot_palantir_results
    from ..external.palantir.presults import compute_gene_trends, select_branch_cells
    from ..external.palantir.utils import determine_multiscale_space, run_diffusion_maps, run_magic_imputation

    return {
        "plot_palantir_results": plot_palantir_results,
        "plot_branch_selection": plot_branch_selection,
        "plot_gene_trends": plot_gene_trends,
        "run_diffusion_maps": run_diffusion_maps,
        "determine_multiscale_space": determine_multiscale_space,
        "run_magic_imputation": run_magic_imputation,
        "run_palantir": run_palantir,
        "select_branch_cells": select_branch_cells,
        "compute_gene_trends": compute_gene_trends,
    }


@register_function(
    aliases=["轨迹推断", "TrajInfer", "trajectory_inference", "轨迹分析", "发育轨迹"],
    category="single",
    description="Comprehensive trajectory inference for single-cell data — Palantir, diffusion maps, Slingshot, StaVIA, scTour, plus the dynbenchmark zoo (SCORPIUS, TSCAN, destiny, URD, Monocle3, CytoTRACE)",
    prerequisites={
        'functions': ['pca', 'neighbors'],
        'optional_functions': ['leiden', 'umap']
    },
    requires={
        'obsm': ['X_pca'],
        'uns': ['neighbors']
    },
    produces={
        'obs': ['palantir_pseudotime'],
        'obsm': ['X_palantir', 'branch_probs'],
        'uns': ['palantir_imp', 'gene_trends']
    },
    auto_fix='auto',
    examples=[
        "# Initialize TrajInfer",
        "traj = ov.single.TrajInfer(adata, basis='X_umap', groupby='clusters',",
        "                           use_rep='X_pca', n_comps=50)",
        "# Set origin and terminal cells",
        "traj.set_origin_cells('stem_cells')",
        "traj.set_terminal_cells(['differentiated_A', 'differentiated_B'])",
        "# Diffusion map trajectory inference",
        "traj.inference(method='diffusion_map')",
        "# Slingshot trajectory inference",
        "traj.inference(method='slingshot', num_epochs=1)",
        "# Palantir trajectory inference",
        "traj.inference(method='palantir', num_waypoints=500)",
        "# StaVIA spatial trajectory inference",
        "traj.inference(method='stavia', spatial_key='spatial', spatial_knn=15)",
        "# Visualize Palantir results",
        "traj.palantir_plot_pseudotime(embedding_basis='X_umap', cmap='RdBu_r')",
        "# Calculate branch probabilities",
        "traj.palantir_cal_branch(eps=0)",
        "# Compute gene expression trends",
        "gene_trends = traj.palantir_cal_gene_trends(layers='MAGIC_imputed_data')",
        "# Plot gene trends along trajectories",
        "traj.palantir_plot_gene_trends(['gene1', 'gene2', 'gene3'])"
    ],
    related=["utils.cal_paga", "utils.plot_paga", "pp.neighbors", "external.palantir"]
)
class TrajInfer(object):
    r"""Trajectory inference class for single-cell data analysis.
    
    This class provides methods for inferring developmental trajectories using
    various algorithms including Palantir, diffusion maps, and Slingshot.
    """
    
    def __init__(self,adata:anndata.AnnData,
                 basis:str='X_umap',use_rep:str='X_pca',n_comps:int=50,
                 n_neighbors:int=15,
                groupby:str='clusters',):
        r"""Initialize trajectory inference object.
        
        Parameters
        ----------
        adata : anndata.AnnData
            AnnData object containing single-cell expression and embeddings.
        basis : str
            Embedding key in ``adata.obsm`` used for visualization (for example
            ``'X_umap'``).
        use_rep : str
            Representation key in ``adata.obsm`` used for trajectory inference.
        n_comps : int
            Number of components from ``use_rep`` used for graph/diffusion
            computation.
        n_neighbors : int
            Number of neighbors used in graph construction.
        groupby : str
            Cluster/annotation key in ``adata.obs``.
        """
        self.adata=adata
        self.use_rep=use_rep
        self.n_comps=n_comps
        self.basis=basis
        self.groupby=groupby
        self.n_neighbors=n_neighbors
        
        self.origin=None
        self.terminal=None
        
    def set_terminal_cells(self,terminal:list):
        r"""Set terminal cell types for trajectory inference.
        
        Parameters
        ----------
        terminal : list
            List of terminal cell-state labels in ``adata.obs[groupby]``.

        Returns
        -------
        None
        """
        self.terminal=terminal
        
    def _dense_counts(self):
        """Return (X, var_names) — dense cells × genes counts matrix + gene labels.

        Many zoo backends (SCORPIUS, TSCAN, destiny, URD, CytoTRACE) need RAW
        counts because they log-transform internally. The resolution order is:

        1. ``adata.layers['counts']`` if present (omicverse preprocess saves
           raw counts here before normalisation).
        2. ``adata.raw.X`` with ``adata.raw.var_names`` if a raw is set
           (note: omicverse stores the *normalised* matrix as raw, so this
           is a fallback only when no counts layer exists).
        3. ``adata.X`` with ``adata.var_names`` (post-HVG, post-normalisation).
        """
        if 'counts' in self.adata.layers:
            X = self.adata.layers['counts']
            var_names = list(self.adata.var_names)
        elif self.adata.raw is not None:
            X = self.adata.raw.X
            var_names = list(self.adata.raw.var_names)
        else:
            X = self.adata.X
            var_names = list(self.adata.var_names)
        if issparse(X):
            X = X.toarray()
        return np.asarray(X), var_names

    def _origin_row_index(self) -> int:
        """Resolve ``self.origin`` to a row index in ``adata``.

        Used as the root cell for diffusion-map-based methods. Falls back to
        the first cell of the origin cluster, or to the global extremum of the
        first PCA component if origin is unset.
        """
        if self.origin is not None and self.groupby in self.adata.obs.columns:
            mask = self.adata.obs[self.groupby].astype(str) == str(self.origin)
            if mask.any():
                return int(np.where(mask.values)[0][0])
        if self.use_rep in self.adata.obsm:
            return int(np.argmin(self.adata.obsm[self.use_rep][:, 0]))
        return 0

    def _monocle3_root_fallback(self, cds) -> list:
        """Pick a data-driven root for Monocle3 when origin is unspecified."""
        for k in ('UMAP', 'X_umap'):
            if k in cds.obsm:
                return [cds.obs_names[int(np.argmin(cds.obsm[k][:, 0]))]]
        return [cds.obs_names[0]]

    def set_origin_cells(self,origin:str):
        r"""Set origin cell type for trajectory inference.
        
        Parameters
        ----------
        origin : str
            Origin/start cell-state label in ``adata.obs[groupby]``.

        Returns
        -------
        None
        """
        self.origin=origin
        
    def inference(self,method:str='palantir',**kwargs):
        r"""Perform trajectory inference using specified method.
        
        Parameters
        ----------
        method : str
            Trajectory inference backend. Supported values include
            ``'palantir'``, ``'diffusion_map'``, ``'slingshot'``,
            ``'sctour'`` and ``'stavia'``.
        **kwargs
            Additional backend-specific keyword arguments.
            
        Returns
        -------
        object or None
            Palantir result object for ``method='palantir'``; otherwise updates
            ``self.adata`` in place and returns ``None``.
        """
        
        if method=='palantir':
            palantir = _get_palantir_backend()

            dm_res = palantir["run_diffusion_maps"](self.adata,
                                                       pca_key=self.use_rep, 
                                                       n_components=self.n_comps)
            ms_data = palantir["determine_multiscale_space"](self.adata)
            imputed_X = palantir["run_magic_imputation"](self.adata)

            sc.tl.rank_genes_groups(self.adata, groupby=self.groupby, 
                        method='t-test',use_rep=self.use_rep,)
            cosg(self.adata, key_added=f'{self.groupby}_cosg', groupby=self.groupby)
            
            ## terminal cells calculation
            if self.terminal is None:
                terminal_states = None  # let Palantir compute terminal cells
            else:
                terminal_index=[]
                for t in self.terminal:
                    gene=sc.get.rank_genes_groups_df(self.adata, group=t, key=f'{self.groupby}_cosg')['names'][0]
                    terminal_index.append(self.adata[self.adata.obs[self.groupby]==t].to_df()[gene].sort_values().index[-1])
                
                terminal_states = pd.Series(
                    self.terminal,
                    index=terminal_index,
                )
            
            ## origin cells calculation
            origin_cell=self.origin
            gene=sc.get.rank_genes_groups_df(self.adata, group=origin_cell, key=f'{self.groupby}_cosg')['names'][0]
            origin_cell_index=self.adata[self.adata.obs[self.groupby]==origin_cell].to_df()[gene].sort_values().index[-1]
            
            start_cell = origin_cell_index
            pr_res = palantir["run_palantir"](
                self.adata, early_cell=start_cell, terminal_states=terminal_states,
                **kwargs
            )
            
            self.adata.obs['palantir_pseudotime']=pr_res.pseudotime
            add_reference(self.adata,'Palantir','trajectory inference with Palantir')
            return pr_res
        elif method=='diffusion_map':
            sc.pp.neighbors(self.adata, n_neighbors=self.n_neighbors, n_pcs=self.n_comps,
               use_rep=self.use_rep)
            sc.tl.diffmap(self.adata)
            sc.pp.neighbors(self.adata, n_neighbors=self.n_neighbors, use_rep='X_diffmap')
            sc.tl.draw_graph(self.adata)
            self.adata.uns['iroot'] = np.flatnonzero(self.adata.obs[self.groupby]  == self.origin)[0]
            sc.tl.dpt(self.adata)
            sc.pp.neighbors(self.adata, n_neighbors=self.n_neighbors, n_pcs=self.n_comps,
               use_rep=self.use_rep)
            add_reference(self.adata,'diffusion_map','trajectory inference with diffusion map')
        elif method=='slingshot':
            #sc.pp.neighbors(self.adata, n_neighbors=self.n_neighbors, n_pcs=self.n_comps,
            #   use_rep=self.use_rep)
            #sc.tl.umap(self.adata)
            from ._pyslingshot import Slingshot
            slingshot = Slingshot(self.adata, 
                      celltype_key=self.groupby, 
                      obsm_key=self.basis, 
                      start_node=self.origin,
                      end_nodes=self.terminal,
                      debug_level='verbose')
            slingshot.fit(**kwargs)
            pseudotime = slingshot.unified_pseudotime
            self.adata.obs['slingshot_pseudotime']=pseudotime
            self.slingshot=slingshot
            add_reference(self.adata,'slingshot','trajectory inference with slingshot')
        elif method=='sctour':
            import sctour as sct

            sctour_adata = self.adata.copy()
            if issparse(sctour_adata.X):
                sctour_adata.X = sctour_adata.X.toarray()
            else:
                sctour_adata.X = np.asarray(sctour_adata.X)

            tnode = sct.train.Trainer(
                sctour_adata, loss_mode='nb', 
                **kwargs
            )
            tnode.train()
            self.adata.obs['sctour_pseudotime'] = tnode.get_time()
            mix_zs, zs, pred_zs = tnode.get_latentsp(alpha_z=0.5, alpha_predz=0.5)
            self.adata.obsm['X_TNODE'] = mix_zs
            self.adata.obsm['X_VF'] = tnode.get_vector_field(
                self.adata.obs['sctour_pseudotime'].values, 
                self.adata.obsm['X_TNODE']
            )
            add_reference(self.adata,'sctour','trajectory inference with sctour')
            self.tnode=tnode
        elif method=='stavia':
            from ._stavia import StaVIA

            stavia_kwargs = dict(kwargs)
            stavia_kwargs.setdefault("use_rep", self.use_rep)
            stavia_kwargs.setdefault("n_comps", self.n_comps)
            stavia_kwargs.setdefault("basis", self.basis)
            stavia_kwargs.setdefault("cluster_key", self.groupby)

            model = StaVIA(self.adata, **stavia_kwargs)
            model.fit()
            self.stavia = model
            return model
        elif method == 'scorpius':
            # Linear-trajectory method via SCORPIUS (Cannoodt 2016).
            # Backend: pyscorpius — `pip install omicverse[trajectory]`
            import pyscorpius
            counts, _ = self._dense_counts()
            space = pyscorpius.reduce_dimensionality(counts, ndim=kwargs.pop('ndim', 3))
            traj = pyscorpius.infer_trajectory(space, **kwargs)
            time_vec = traj['time'] if isinstance(traj, dict) else getattr(traj, 'time', None)
            self.adata.obs['scorpius_pseudotime'] = np.asarray(time_vec, dtype=float)
            self.adata.obsm['X_scorpius'] = np.asarray(space)
            add_reference(self.adata, 'SCORPIUS', 'trajectory inference with SCORPIUS')
        elif method == 'tscan':
            # mclust-based pseudotime (Ji & Ji 2016).
            # Backend: pytscan — `pip install omicverse[trajectory]`
            import pytscan
            counts, var_names = self._dense_counts()
            expr = pd.DataFrame(counts.T,
                                index=var_names,
                                columns=list(self.adata.obs_names))
            mclust_kwargs = {k: kwargs.pop(k) for k in
                             ('clusternum', 'reduce', 'modelNames', 'clustermethod')
                             if k in kwargs}
            res = pytscan.exprmclust(expr, **mclust_kwargs)
            order = pytscan.TSCANorder(res, orderonly=True)
            n = max(len(order.ordered_names) - 1, 1)
            cell_to_t = {c: i / n for i, c in enumerate(order.ordered_names)}
            pt = np.array([cell_to_t.get(c, np.nan) for c in self.adata.obs_names])
            if np.any(np.isnan(pt)):
                pt = np.where(np.isnan(pt), np.nanmedian(pt), pt)
            self.adata.obs['tscan_pseudotime'] = pt
            self.adata.obs['tscan_cluster'] = pd.Categorical([f'C{c}' for c in res.clusterid])
            add_reference(self.adata, 'TSCAN', 'trajectory inference with TSCAN')
        elif method == 'destiny':
            # DiffusionMap + DPT pseudotime (Haghverdi 2016 / Angerer 2016).
            # Backend: pydestiny — `pip install omicverse[trajectory]`
            import pydestiny
            counts, _ = self._dense_counts()
            expr = np.log2(counts + 1)
            dm = pydestiny.DiffusionMap.fit(
                expr,
                sigma=kwargs.pop('sigma', 'local'),
                n_eigs=kwargs.pop('n_eigs', 5),
                k=kwargs.pop('k', min(20, max(2, expr.shape[0] // 5))),
            )
            root = self._origin_row_index()
            dpt = pydestiny.DPT(dm, root=int(root))
            self.adata.obs['destiny_pseudotime'] = np.asarray(dpt, dtype=float)
            try:
                self.adata.obsm['X_destiny'] = np.asarray(dm.eigenvectors)
            except Exception:
                pass
            self.destiny = dm
            add_reference(self.adata, 'destiny', 'trajectory inference with destiny DPT')
        elif method == 'urd':
            # Flood-pseudotime with diffusion-map transitions (Farrell 2018).
            # Backend: pyurd + pydestiny — `pip install omicverse[trajectory]`
            import pydestiny, pyurd
            counts, _ = self._dense_counts()
            expr = np.log2(counts + 1)
            dm = pydestiny.DiffusionMap.fit(
                expr,
                sigma=kwargs.pop('sigma', 'local'),
                n_eigs=kwargs.pop('n_eigs', 5),
                k=kwargs.pop('k', min(20, max(2, expr.shape[0] // 5))),
            )
            urd = pyurd.URD(
                cell_names=list(self.adata.obs_names),
                transitions=dm.transitions,
                dm_eigenvectors=dm.eigenvectors,
                dm_eigenvalues=dm.eigenvalues,
            )
            n_root = kwargs.pop('n_root', max(5, len(self.adata) // 8))
            # Match R URD: roots = the n_root cells closest to one extremum of DC1.
            # If `self.origin` is set, choose the DC1 extremum whose mean lies inside
            # the origin cluster — otherwise default to the smaller-DC1 end.
            dc1 = np.asarray(dm.eigenvectors)[:, 0]
            order_low  = np.argsort(dc1)[:n_root]
            order_high = np.argsort(-dc1)[:n_root]
            if (self.origin is not None
                    and self.groupby in self.adata.obs.columns):
                origin_mask = (self.adata.obs[self.groupby].astype(str) == str(self.origin)).values
                low_hit  = origin_mask[order_low].mean()
                high_hit = origin_mask[order_high].mean()
                root_idx = list(order_low if low_hit >= high_hit else order_high)
            else:
                root_idx = list(order_low)
            floods = pyurd.floodPseudotime(
                urd,
                root_cells=root_idx,
                n=kwargs.pop('flood_n', 30),
                minimum_cells_flooded=1,
                seed=kwargs.pop('seed', 42),
            )
            res = pyurd.floodPseudotimeProcess(floods, max_frac_NA=0.9, stability_div=3)
            pt_series = res['pseudotime'] if isinstance(res, dict) else getattr(res, 'pseudotime', None)
            pt = np.full(len(self.adata), np.nan)
            for idx, c in enumerate(self.adata.obs_names):
                if c in pt_series.index:
                    pt[idx] = pt_series.loc[c]
            if np.all(np.isnan(pt)):
                pt = np.zeros_like(pt)
            else:
                pt = np.where(np.isnan(pt), np.nanmedian(pt), pt)
            self.adata.obs['urd_pseudotime'] = pt
            self.urd = urd
            add_reference(self.adata, 'URD', 'trajectory inference with URD')
        elif method == 'monocle3':
            # UMAP + principal-graph pseudotime (Cao 2019).
            # Backend: pymonocle3 — `pip install omicverse[trajectory]`
            import pymonocle3 as m3
            cds = self.adata.copy()
            if issparse(cds.X):
                cds.X = cds.X.toarray()
            m3.estimate_size_factors(cds)
            m3.preprocess_cds(cds, num_dim=kwargs.pop('num_dim', min(20, cds.n_vars - 1)))
            m3.reduce_dimension(cds,
                                max_components=kwargs.pop('max_components', 2),
                                umap_min_dist=kwargs.pop('umap_min_dist', 0.1),
                                umap_n_neighbors=kwargs.pop('umap_n_neighbors', 15))
            m3.cluster_cells(cds, resolution=kwargs.pop('cluster_resolution', 0.001))
            m3.learn_graph(cds, use_partition=kwargs.pop('use_partition', False))
            if self.origin is not None and self.groupby in self.adata.obs.columns:
                mask = self.adata.obs[self.groupby].astype(str) == str(self.origin)
                if mask.any():
                    root_cells = [self.adata.obs_names[np.where(mask.values)[0][0]]]
                else:
                    root_cells = self._monocle3_root_fallback(cds)
            else:
                root_cells = self._monocle3_root_fallback(cds)
            m3.order_cells(cds, root_cells=root_cells)
            # pymonocle3 writes to monocle3_pseudotime / monocle3_partitions
            pt_col = 'monocle3_pseudotime' if 'monocle3_pseudotime' in cds.obs \
                     else ('pseudotime' if 'pseudotime' in cds.obs else None)
            if pt_col is None:
                pt = np.zeros(cds.n_obs)
            else:
                pt = np.asarray(cds.obs[pt_col], dtype=float)
            finite = np.isfinite(pt)
            if finite.sum():
                pt = np.where(finite, pt, np.nanmedian(pt[finite]))
            else:
                pt = np.zeros_like(pt)
            if pt.max() > pt.min():
                pt = (pt - pt.min()) / (pt.max() - pt.min())
            self.adata.obs['monocle3_pseudotime'] = pt
            for part_col in ('monocle3_partitions', 'partition', 'monocle3_partition'):
                if part_col in cds.obs.columns:
                    self.adata.obs['monocle3_partition'] = cds.obs[part_col].astype(str).values
                    break
            for k in ('UMAP', 'X_umap'):
                if k in cds.obsm:
                    self.adata.obsm['X_monocle3'] = np.asarray(cds.obsm[k])
                    break
            self.monocle3 = cds
            add_reference(self.adata, 'Monocle3', 'trajectory inference with Monocle3')
        elif method == 'cytotrace':
            # Differentiation potential ranked by gene-count correlate (Gulati 2020).
            # Backend: pycytotrace — `pip install omicverse[trajectory]`
            import pycytotrace
            counts, _ = self._dense_counts()
            # pycytotrace expects genes × cells, like R upstream.
            res = pycytotrace.cytotrace_run(counts.T, **kwargs)
            stem = np.asarray(getattr(res, 'cytotrace', res), dtype=float)
            self.adata.obs['cytotrace_stemness'] = stem
            self.adata.obs['cytotrace_pseudotime'] = 1.0 - stem
            add_reference(self.adata, 'CytoTRACE', 'trajectory inference with CytoTRACE')
        else:
            print(
                'Please input the correct method name. Supported: '
                'palantir, diffusion_map, slingshot, sctour, stavia, '
                'scorpius, tscan, destiny, urd, monocle3, cytotrace'
            )
            return
        
    def palantir_plot_pseudotime(self, return_fig: bool = False, **kwargs):
        r"""Plot Palantir pseudotime results.
        
        Parameters
        ----------
        return_fig : bool, optional
            Whether to return the matplotlib figure object. By default the
            figure is drawn without being returned, which avoids duplicate
            notebook rendering.
        **kwargs
            Additional keyword arguments forwarded to
            ``plot_palantir_results`` such as ``embedding_basis``, ``cmap``,
            ``s``, ``n_cols`` or ``figsize``.

        Returns
        -------
        matplotlib.figure.Figure or None
        """
        palantir = _get_palantir_backend()
        fig = palantir["plot_palantir_results"](self.adata, **kwargs)
        if return_fig:
            return fig
        return None
        
    def palantir_cal_branch(
        self,
        plot_kwargs: dict | None = None,
        return_fig: bool = False,
        **kwargs,
    ):
        r"""Calculate and plot branch selection for Palantir results.
        
        Parameters
        ----------
        plot_kwargs : dict, optional
            Keyword arguments forwarded to ``plot_branch_selection`` such as
            ``figsize``, ``selected_color`` or ``deselected_color``.
        return_fig : bool, optional
            Whether to return the matplotlib figure object. By default the
            figure is drawn without being returned, which avoids duplicate
            notebook rendering.
        **kwargs
            Additional keyword arguments forwarded to
            ``select_branch_cells`` such as ``eps``, ``q`` or
            ``save_as_df``.

        Returns
        -------
        matplotlib.figure.Figure or None
        """
        palantir = _get_palantir_backend()
        palantir["select_branch_cells"](self.adata, **kwargs)
        plot_kwargs = dict(plot_kwargs or {})
        fig = palantir["plot_branch_selection"](self.adata, **plot_kwargs)
        if return_fig:
            return fig
        return None

    def palantir_cal_gene_trends(self,layers:str="MAGIC_imputed_data"):
        r"""Calculate gene expression trends along Palantir trajectories.
        
        Parameters
        ----------
        layers : str
            Expression layer key used for trend modeling.
            
        Returns
        -------
        object
            Gene-trend result object containing trajectory-associated expression
            patterns.
        """
        palantir = _get_palantir_backend()
        gene_trends = palantir["compute_gene_trends"](
            self.adata,
            expression_key=layers,
        )
        return gene_trends
        
    def palantir_plot_gene_trends(
        self,
        genes,
        lineages=None,
        layers: str = "MAGIC_imputed_data",
        pseudo_time_key: str = "palantir_pseudotime",
        fate_prob_key: str = "palantir_fate_probabilities",
        use_raw: bool = False,
        fit_kwargs: dict | None = None,
        return_fig: bool = False,
        return_result: bool = False,
        **plot_kwargs,
    ):
        r"""Plot gene expression trends along Palantir trajectories.
        
        Parameters
        ----------
        genes : list
            Gene symbols to visualize along inferred trajectories.
        lineages : list, optional
            Optional subset of Palantir lineages to compare. By default all
            available fate-probability columns are used. Each lineage is fit as
            a separate weighted series using its fate probabilities.
        layers : str, optional
            Expression layer used for fitting when ``use_raw=False``. The
            default ``"MAGIC_imputed_data"`` requires that this layer already
            exists in ``adata.layers``.
        pseudo_time_key : str, optional
            Obs key containing Palantir pseudotime values.
        fate_prob_key : str, optional
            Obsm key containing Palantir lineage probabilities.
        use_raw : bool, optional
            Whether to read expression values from ``adata.raw`` instead of the
            specified layer or ``adata.X``.
        fit_kwargs : dict, optional
            Additional keyword arguments forwarded to
            ``ov.single.dynamic_features``. If ``plot_kwargs`` requests
            ``add_point=True``, ``store_raw=True`` is enabled automatically.
        return_fig : bool, optional
            Whether to return the created matplotlib figure.
        return_result : bool, optional
            Whether to also return the fitted dynamic-features result object.
        **plot_kwargs
            Additional keyword arguments forwarded to
            ``ov.pl.dynamic_trends``. When multiple lineages are present,
            ``compare_groups=True`` is enabled automatically unless explicitly
            overridden.
            
        Returns
        -------
        object
            By default, draws the plot and returns ``None``. When
            ``return_fig=True`` and/or ``return_result=True``, returns the
            requested plotting figure/result objects.
        """
        from ._dynamic_features import dynamic_features
        from ..pl._dynamic_trends import dynamic_trends
        from ..external.palantir.validation import _validate_obsm_key

        fate_probs, lineage_names = _validate_obsm_key(self.adata, fate_prob_key)
        if not use_raw and layers is not None and layers not in self.adata.layers:
            raise KeyError(
                f"Layer `{layers}` not found in adata.layers. "
                "Run the required imputation step first, choose an available layer, or set `use_raw=True`."
            )
        if lineages is None:
            lineages = list(lineage_names)
        else:
            lineages = [str(lineage) for lineage in lineages]
            missing = [lineage for lineage in lineages if lineage not in lineage_names]
            if missing:
                raise KeyError(
                    f"Requested lineages were not found in `{fate_prob_key}`: {missing}."
                )

        dataset_map = {lineage: self.adata for lineage in lineages}
        weights = {lineage: fate_probs[lineage].to_numpy(dtype=float) for lineage in lineages}
        fit_kwargs = dict(fit_kwargs or {})
        fit_kwargs.setdefault("store_raw", bool(plot_kwargs.get("add_point", False)))

        trend_res = dynamic_features(
            dataset_map,
            genes=genes,
            pseudotime=pseudo_time_key,
            layer=None if use_raw else layers,
            use_raw=use_raw,
            weights=weights,
            key_added=None,
            **fit_kwargs,
        )

        plot_kwargs = dict(plot_kwargs)
        plot_kwargs.setdefault("compare_groups", len(lineages) > 1)
        plot_output = dynamic_trends(
            trend_res,
            genes=genes,
            return_fig=return_fig,
            **plot_kwargs,
        )

        if return_fig and return_result:
            return plot_output, trend_res
        if return_result:
            return trend_res
        return plot_output
    
import networkx as nx
from scipy.sparse import csr_matrix

def construct_graph_(
    W: csr_matrix, directed: bool = False, adjust_weights: bool = True
) -> nx.Graph:
    """
    Convert sparse adjacency matrix into weighted NetworkX graph.

    Parameters
    ----------
    W : csr_matrix
        Sparse adjacency/connectivity matrix.
    directed : bool
        Whether to build a directed graph.
    adjust_weights : bool
        Whether to median-normalize and round edge weights.

    Returns
    -------
    nx.Graph
        Weighted NetworkX graph.
    """
    assert issparse(W), "W must be a scipy.sparse matrix"

    # 根据 adjust_weights 选项调整权重
    s, t = W.nonzero()
    w = W.data
    if not directed:
        mask = s < t
        s, t, w = s[mask], t[mask], w[mask]

    if adjust_weights:
        # 标准化并四舍五入到小数点后两位
        w = ((w / np.median(w)) * 100.0 + 0.5).astype(int) / 100.0
        mask = w > 0
        s, t, w = s[mask], t[mask], w[mask]

    # 构建 networkx 图
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(W.shape[0]))
    edges = zip(s.tolist(), t.tolist(), w.tolist())
    G.add_weighted_edges_from(edges)
    return G

def construct_graph(
    W, directed: bool = False, adjust_weights: bool = True
) -> "igraph":
    """
    Convert sparse adjacency matrix into weighted igraph object.

    Parameters
    ----------
    W : scipy.sparse matrix
        Sparse adjacency/connectivity matrix.
    directed : bool
        Whether to build a directed graph.
    adjust_weights : bool
        Whether to median-normalize and round edge weights.

    Returns
    -------
    igraph.Graph
        Weighted igraph graph.
    """

    assert issparse(W)

    s, t = W.nonzero()
    w = W.data

    if not directed:
        idx = s < t
        s = s[idx]
        t = t[idx]
        w = w[idx]

    if adjust_weights:
        w = ((w / np.median(w)) * 100.0 + 0.5).astype(
            int
        ) / 100.0  # round to 2 decimal points
        idx = w > 0.0
        if idx.sum() < w.size:
            s = s[idx]
            t = t[idx]
            w = w[idx]

    G = igraph.Graph(directed=directed)
    G.add_vertices(W.shape[0])
    G.add_edges(zip(s, t))
    G.es["weight"] = w

    return G

def calc_force_directed_layout(
    W,
    file_name,
    n_jobs,
    target_change_per_node,
    target_steps,
    is3d,
    memory,
    random_state,
    init=None,
    scaling_ratio=2.0, 
    gravity=1.0, 
    edge_weight_influence=1.0,
    use_gpu=False,  # New parameter to enable GPU acceleration
    optimized_threading=True,  # Enable optimized threading algorithm
    compile_cython=True,  # 是否尝试编译Cython模块以加速
):
    """
    Compute ForceAtlas2 layout from connectivity graph.

    Parameters
    ----------
    W : scipy.sparse matrix
        Connectivity graph matrix.
    file_name : str
        Temporary file name (kept for compatibility).
    n_jobs : int
        Number of CPU threads.
    target_change_per_node : float
        Target movement change threshold per node.
    target_steps : int
        Maximum ForceAtlas2 iterations.
    is3d : bool
        Whether to output 3D coordinates.
    memory : int
        Reserved memory parameter for compatibility.
    random_state : int
        Random seed.
    init : Any
        Optional initial coordinates.
    scaling_ratio : float
        ForceAtlas2 scaling ratio.
    gravity : float
        Gravity strength.
    edge_weight_influence : float
        Influence of edge weights in force computation.
    use_gpu : bool
        Whether to use GPU-accelerated force computation.
    optimized_threading : bool
        Whether to enable optimized multi-thread execution.
    compile_cython : bool
        Whether to try compiling Cython helpers for acceleration.

    Returns
    -------
    np.ndarray
        Layout coordinates with shape ``(n_cells, 2)`` or ``(n_cells, 3)``.
    """
    G = construct_graph(W)
    try:
        from ..external.forcedirect2.forceatlas2 import ForceAtlas2
        
        # 如果请求编译并且fa2util模块需要编译
        if compile_cython:
            try:
                import os
                import sys
                import importlib.util
                
                # 检查是否已经编译
                force_dir = os.path.dirname(os.path.abspath(importlib.util.find_spec("omicverse.external.forcedirect2.forceatlas2").origin))
                compiled_module_exists = any(f.startswith('fa2util.') and (f.endswith('.so') or f.endswith('.pyd')) 
                                           for f in os.listdir(force_dir))
                
                # 如果没有编译过，尝试编译
                if not compiled_module_exists:
                    print("检测到未编译的fa2util模块。尝试编译以获得10-100倍速度提升...")
                    compile_script = os.path.join(force_dir, "compile.py")
                    
                    # 如果compile.py存在，执行它
                    if os.path.exists(compile_script):
                        import subprocess
                        subprocess.run([sys.executable, compile_script], cwd=force_dir)
                    else:
                        print(f"未找到编译脚本: {compile_script}")
                        print("请手动编译fa2util模块以获得更好性能")
                        print("cd " + force_dir)
                        print("python setup.py build_ext --inplace")
            except Exception as e:
                print(f"尝试编译模块时出错: {str(e)}")
                print("将使用未编译的Python版本 (较慢)")
        
        # Initialize ForceAtlas2 with our parameters
        forceatlas2 = ForceAtlas2(
            # Behavior
            outboundAttractionDistribution=True,
            edgeWeightInfluence=edge_weight_influence,
            
            # Performance
            jitterTolerance=1.0,
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            multiThreaded=True if not use_gpu else False,  # Only use multithreading if not using GPU
            numThreads=n_jobs if n_jobs > 0 else None,  # Use specified number of threads
            useGPU=use_gpu,  # Enable GPU acceleration if requested
            optimizedThreading=optimized_threading,  # Enable optimized threading
            
            # Tuning
            scalingRatio=scaling_ratio,
            strongGravityMode=False,
            gravity=gravity,
            
            # Log
            verbose=True
        )
        
        # Convert igraph to adjacency matrix for forceatlas2
        import numpy as np
        from scipy.sparse import csr_matrix
        
        edges = G.get_edgelist()
        weights = G.es["weight"]
        n_vertices = G.vcount()
        
        rows = [edge[0] for edge in edges]
        cols = [edge[1] for edge in edges]
        
        # Create sparse adjacency matrix
        adj_matrix = csr_matrix((weights, (rows, cols)), shape=(n_vertices, n_vertices))
        
        # Make symmetric if graph is undirected
        if not G.is_directed():
            adj_matrix = adj_matrix + adj_matrix.T
        
        # Run ForceAtlas2
        positions = forceatlas2.forceatlas2(adj_matrix, iterations=target_steps)
        
        # If 3D layout is requested, add a third coordinate with zeros
        if is3d:
            return np.column_stack([positions, np.zeros(len(positions))])
        
        return np.array(positions)
        
    except ImportError:
        # Fallback to external forceatlas2 if our implementation is not available
        import sys
        print("Could not find custom ForceAtlas2 implementation!")
        sys.exit(-1)


def fle(
    data,
    file_name: str = None,
    n_jobs: int = -1,
    rep: str = "diffmap",
    rep_ncomps: int = None,
    K: int = 50,
    full_speed: bool = False,
    target_change_per_node: float = 2.0,
    target_steps: int = 5000,
    is3d: bool = False,
    memory: int = 8,
    random_state: int = 0,
    out_basis: str = "fle",
    fa2_scaling_ratio: float = 2.0,
    fa2_gravity: float = 1.0,
    fa2_edge_weight_influence: float = 1.0,
    use_gpu: bool = False,  # New parameter to enable GPU acceleration
    optimized_threading: bool = True,  # Enable optimized threading algorithm
    compile_cython: bool = True,  # 是否尝试编译Cython模块以加速
) -> None:
    """
    Construct force-directed layout embedding (FLE) from graph representation.

    This implementation uses our custom ForceAtlas2 implementation, which is a multilthreaded version
    of the original ForceAtlas2 algorithm.

    See [Jacomy14]_ for details on FLE.

    Parameters
    ----------
    data : AnnData-like
        Data object with graph/connectivity fields.

    file_name : str, optional
        Temporary file to store the coordinates as the input to forceatlas2. If ``None``, use ``tempfile.mkstemp`` to generate file name.

    n_jobs : int, optional
        Number of threads to use. If ``-1``, use all physical CPU cores.

    rep : str, optional
        Representation of data used for the calculation. By default, use Diffusion Map coordinates. If ``None``, use the count matrix ``data.X``.

    rep_ncomps : int, optional
        Number of components to be used in `rep`. If rep_ncomps == None, use all components; otherwise, use the minimum of rep_ncomps and rep's dimensions.

    K : int, optional
        Number of nearest neighbors to be considered during the computation.

    full_speed : bool, optional
        * If ``True``, use multiple threads in constructing ``hnsw`` index. However, the kNN results are not reproducible.
        * Otherwise, use only one thread to make sure results are reproducible.

    target_change_per_node : float, optional
        Target change per node to stop ForceAtlas2.

    target_steps : int, optional
        Maximum number of iterations before stopping the ForceAtlas2 algorithm.

    is3d : bool, optional
        If ``True``, calculate 3D force-directed layout.

    memory : int, optional
        Memory size in GB for the Java FA2 component. By default, use 8GB memory.

    random_state : int, optional
        Random seed set for reproducing results.

    out_basis : str, optional
        Key name for calculated FLE coordinates to store.
    
    fa2_scaling_ratio : float, optional
        Scaling ratio parameter for ForceAtlas2.
        
    fa2_gravity : float, optional
        Gravity parameter for ForceAtlas2.
        
    fa2_edge_weight_influence : float, optional
        Edge weight influence parameter for ForceAtlas2.
        
    use_gpu : bool, optional
        Whether to use GPU acceleration for force calculations. Requires PyTorch to be installed.
        
    optimized_threading : bool, optional
        Whether to use optimized threading algorithm for better parallel performance.
        
    compile_cython : bool, optional
        Whether to try compiling fa2util module with Cython for 10-100x speedup.

    Returns
    -------
    None
        Stores layout in ``data.obsm['X_' + out_basis]``.

    Examples
    --------
    >>> pg.fle(data)
    """

    if file_name is None:
        import tempfile

        _, file_name = tempfile.mkstemp()

    rep = update_rep(rep)
    n_jobs = eff_n_jobs(n_jobs)

    if ("W_" + rep) not in data.uns:
        neighbors(
            data,
            K=K,
            rep=rep,
            n_comps=rep_ncomps,
            n_jobs=n_jobs,
            random_state=random_state,
            full_speed=full_speed,
        )

    key = f"X_{out_basis}"
    data.obsm[key] = calc_force_directed_layout(
        W_from_rep(data, rep),
        file_name,
        n_jobs,
        target_change_per_node,
        target_steps,
        is3d,
        memory,
        random_state,
        scaling_ratio=fa2_scaling_ratio,
        gravity=fa2_gravity,
        edge_weight_influence=fa2_edge_weight_influence,
        use_gpu=use_gpu,
        optimized_threading=optimized_threading,
        compile_cython=compile_cython,
    )


def diffmap_fle(
    adata,
    use_rep: str = 'X_pca',
    n_pcs: int = 50,
    n_comps: int = 15,
    n_neighbors: int = 30,
    random_state: int = 0,
    out_basis: str = "fle",
    target_steps: int = 5000,
    use_gpu: bool = False,
    optimized_threading: bool = True,
    compile_cython: bool = True,
    fa2_scaling_ratio: float = 2.0,
    fa2_gravity: float = 1.0,
    fa2_edge_weight_influence: float = 1.0,
    n_jobs: int = -1,
) -> None:
    """
    Compute diffusion map first, then build ForceAtlas2-based FLE embedding.
    
    This function first computes diffusion maps using scanpy and then applies the optimized
    ForceAtlas2 algorithm on the diffusion map coordinates to create a force-directed layout.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with rows for cells and columns for genes.
        
    use_rep : str, optional
        The dimensional reduction to use for neighbors calculation before diffmap.
        
    n_pcs : int, optional
        Number of PCs to use for neighbors calculation.
        
    n_comps : int, optional
        Number of diffusion components to compute.
        
    n_neighbors : int, optional
        Number of neighbors for building the neighborhood graph.
        
    random_state : int, optional
        Random seed for reproducibility.
        
    out_basis : str, optional
        Key name for calculated FLE coordinates to store.
        
    target_steps : int, optional
        Maximum number of iterations for the ForceAtlas2 algorithm.
        
    use_gpu : bool, optional
        Whether to use GPU acceleration for force calculations.
        
    optimized_threading : bool, optional
        Whether to use optimized threading algorithm for better parallel performance.
        
    compile_cython : bool, optional
        Whether to try compiling fa2util module with Cython for speedup.
        
    fa2_scaling_ratio : float, optional
        Scaling ratio parameter for ForceAtlas2.
        
    fa2_gravity : float, optional
        Gravity parameter for ForceAtlas2.
        
    fa2_edge_weight_influence : float, optional
        Edge weight influence parameter for ForceAtlas2.
        
    n_jobs : int, optional
        Number of threads to use. If ``-1``, use all physical CPU cores.
        
    Returns
    -------
    None
        Updates ``adata.obsm['X_diffmap']`` and ``adata.obsm['X_' + out_basis]``.
        
    Examples
    --------
    >>> diffmap_fle(adata, use_gpu=True)
    """
    # Step 1: Compute diffusion map using scanpy
    import scanpy as sc
    
    print("Computing neighbors for diffusion map...")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, 
                   use_rep=use_rep, random_state=random_state)
    
    print(f"Computing diffusion map with {n_comps} components...")
    sc.tl.diffmap(adata, n_comps=n_comps)
    
    # Store the diffusion map connectivity matrix for FLE
    W_diffmap = adata.obsp['connectivities']
    
    # Step 2: Apply ForceAtlas2 on the diffusion map
    print("Calculating force-directed layout from diffusion map...")
    
    import tempfile
    _, file_name = tempfile.mkstemp()
    
    n_jobs = eff_n_jobs(n_jobs)
    
    key = f"X_{out_basis}"
    adata.obsm[key] = calc_force_directed_layout(
        W_diffmap,
        file_name,
        n_jobs,
        2.0,  # target_change_per_node
        target_steps,
        False,  # is3d
        8,  # memory
        random_state,
        scaling_ratio=fa2_scaling_ratio,
        gravity=fa2_gravity,
        edge_weight_influence=fa2_edge_weight_influence,
        use_gpu=use_gpu,
        optimized_threading=optimized_threading,
        compile_cython=compile_cython,
    )
    
    print(f"Force-directed layout calculated and stored in adata.obsm['X_{out_basis}']")
    return None

# -*- coding: utf-8 -*-
import math
import copy
import os
import torch
import random

import anndata
#import cmcrameri
import numpy as np
import scanpy as sc
import networkx as nx
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from torch_geometric.nn import GCNConv, DeepGraphInfomax
from sklearn.neighbors import kneighbors_graph
from ._util import sparse_mx_to_torch_edge_list, corruption

class SpaceFlow(object):
    """An object for analysis of spatial transcriptomics data.

    :param adata: the `anndata.AnnData` object as input, see `https://anndata.readthedocs.io/en/latest/` for more info about`anndata`.
    :type adata: class:`anndata.AnnData`
    :param count_matrix: count matrix of gene expression, 2D numpy array of size (n_cells, n_genes)
    :type count_matrix: class:`numpy.ndarray`
    :param spatial_locs: spatial locations of cells (or spots) match to rows of the count matrix, 1D numpy array of size (n_cells,)
    :type spatial_locs: class:`numpy.ndarray`
    :param sample_names: list of sample names in 1D numpy str array of size (n_cells,), optional
    :type sample_names: class:`numpy.ndarray` or `list` of `str`
    :param gene_names: list of gene names in 1D numpy str array of size (n_genes,), optional
    :type gene_names: class:`numpy.ndarray` or `list` of `str`

    """

    def __init__(self, adata=None, count_matrix=None, spatial_locs=None, sample_names=None, gene_names=None):
        """
        Inputs
        ------
        adata: an anndata.AnnData type object, optional (either input `adata` or both `count_matrix` and `spatial_locs`)
        count_matrix : count matrix of gene expression, 2D numpy array of size (n_cells, n_genes)
        spatial_locs : spatial locations of cells (or spots) match to rows of the count matrix, 1D numpy array of size (n_cells,)
        sample_names : list of sample names in 1D numpy str array of size (n_cells,), optional
        gene_names : list of gene names in 1D numpy str array of size (n_genes,), optional
        """
        if adata and isinstance(adata, anndata.AnnData):
            self.adata = adata
        elif count_matrix is not None and spatial_locs is not None:
            self.adata = anndata.AnnData(count_matrix.astype(float))
            self.adata.obsm['spatial'] = spatial_locs.astype(float)
            if gene_names:
                self.adata.var_names = np.array(gene_names).astype(str)
            if sample_names:
                self.adata.obs_names = np.array(sample_names).astype(str)
        else:
            print("Please input either an anndata.AnnData or both the count_matrix (count matrix of gene expression, 2D int numpy array of size (n_cells, n_genes)) and spatial_locs (spatial locations of cells (or spots) in 1D float numpy array of size (n_locations,)) to initiate SpaceFlow object.")
            exit(1)

    def plt_setting(self, fig_title_sz=30, font_sz=12, font_weight="bold", axes_title_sz=12, axes_label_sz=12, xtick_sz=10, ytick_sz=10, legend_font_sz=10):
        """
        Setting the plotting configuration
        :param fig_title_sz: fontsize of the figure title, default: 30
        :type fig_title_sz: int, optional
        :param font_sz: controls default text sizes, default: 12
        :type font_sz: int, optional
        :param font_weight: controls default text weights, default: 'bold'
        :type font_weight: str, optional
        :param axes_title_sz: fontsize of the axes title, default: 12
        :type axes_title_sz: int, optional
        :param axes_label_sz: fontsize of the x and y labels, default 12
        :type axes_label_sz: int, optional
        :param xtick_sz: fontsize of the x tick label, default 10
        :type xtick_sz: int, optional
        :param ytick_sz: fontsize of the y tick label, default 10
        :type ytick_sz: int, optional
        :param legend_font_sz: legend fontsize, default 10
        :type legend_font_sz: int, optional
        """
        plt.rc('figure', titlesize=fig_title_sz)  # fontsize of the figure title
        plt.rc('font', size=font_sz, weight=font_weight)  # controls default text sizes
        plt.rc('axes', titlesize=axes_title_sz)  # fontsize of the axes title
        plt.rc('axes', labelsize=axes_label_sz)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=xtick_sz)  # fontsize of the x tick label
        plt.rc('ytick', labelsize=ytick_sz)  # fontsize of the y tick label
        plt.rc('legend', fontsize=legend_font_sz)  # legend fontsize

    def prepare_figure(self, rsz=4., csz=4., wspace=.4, hspace=.5, left=0.125, right=0.9, bottom=0.1, top=0.9):
        """
        Prepare the figure and axes given the configuration
        :param rsz: row size of the figure in inches, default: 4.0
        :type rsz: float, optional
        :param csz: column size of the figure in inches, default: 4.0
        :type csz: float, optional
        :param wspace: the amount of width reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
        :type wspace: float, optional
        :param hspace: the amount of height reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
        :type hspace: float, optional
        :param left: the leftmost position of the subplots of the figure in fraction, default: 0.125
        :type left: float, optional
        :param right: the rightmost position of the subplots of the figure in fraction, default: 0.9
        :type right: float, optional
        :param bottom: the bottom position of the subplots of the figure in fraction, default: 0.1
        :type bottom: float, optional
        :param top: the top position of the subplots of the figure in fraction, default: 0.9
        :type top: float, optional
        """
        fig, axs = plt.subplots(1, 1, figsize=(csz, rsz))
        plt.subplots_adjust(wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom, top=top)
        return fig, axs

    def preprocessing_data(self, n_top_genes=None, n_neighbors=10):
        """
        Preprocessing the spatial transcriptomics data
        Generates:  `self.adata_filtered`: (n_cells, n_locations) `numpy.ndarray`
                    `self.spatial_graph`: (n_cells, n_locations) `numpy.ndarray`
        :param adata: the annData object for spatial transcriptomics data with adata.obsm['spatial'] set to be the spatial locations.
        :type adata: class:`anndata.annData`
        :param n_top_genes: the number of top highly variable genes
        :type n_top_genes: int, optional
        :param n_neighbors: the number of nearest neighbors for building spatial neighbor graph
        :type n_neighbors: int, optional
        :return: a preprocessed annData object of the spatial transcriptomics data
        :rtype: class:`anndata.annData`
        :return: a geometry-aware spatial proximity graph of the spatial spots of cells
        :rtype: class:`scipy.sparse.csr_matrix`
        """
        adata = self.adata
        if not adata:
            print("No annData object found, please run SpaceFlow.SpaceFlow(expr_data, spatial_locs) first!")
            return
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='cell_ranger', subset=True)
        sc.pp.pca(adata)
        spatial_locs = adata.obsm['spatial']
        spatial_graph = self.graph_alpha(spatial_locs, n_neighbors=n_neighbors)

        self.adata_preprocessed = adata
        self.spatial_graph = spatial_graph

    def graph_alpha(self, spatial_locs, n_neighbors=10):
        """
        Construct a geometry-aware spatial proximity graph of the spatial spots of cells by using alpha complex.
        :param adata: the annData object for spatial transcriptomics data with adata.obsm['spatial'] set to be the spatial locations.
        :type adata: class:`anndata.annData`
        :param n_neighbors: the number of nearest neighbors for building spatial neighbor graph based on Alpha Complex
        :type n_neighbors: int, optional, default: 10
        :return: a spatial neighbor graph
        :rtype: class:`scipy.sparse.csr_matrix`
        """
        import gudhi
        A_knn = kneighbors_graph(spatial_locs, n_neighbors=n_neighbors, mode='distance')
        estimated_graph_cut = A_knn.sum() / float(A_knn.count_nonzero())
        spatial_locs_list = spatial_locs.tolist()
        n_node = len(spatial_locs_list)
        alpha_complex = gudhi.AlphaComplex(points=spatial_locs_list)
        simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=estimated_graph_cut ** 2)
        skeleton = simplex_tree.get_skeleton(1)
        initial_graph = nx.Graph()
        initial_graph.add_nodes_from([i for i in range(n_node)])
        for s in skeleton:
            if len(s[0]) == 2:
                initial_graph.add_edge(s[0][0], s[0][1])

        extended_graph = nx.Graph()
        extended_graph.add_nodes_from(initial_graph)
        extended_graph.add_edges_from(initial_graph.edges)

        # Remove self edges
        for i in range(n_node):
            try:
                extended_graph.remove_edge(i, i)
            except:
                pass
        
        try:
            G = nx.to_scipy_sparse_matrix(extended_graph, format='csr')
        except AttributeError:
            G = nx.to_scipy_sparse_array(extended_graph, format='csr')
        
        return G

    def train(self, embedding_save_filepath="./embedding.tsv", spatial_regularization_strength=0.1, z_dim=50, lr=1e-3, epochs=1000, max_patience=50, min_stop=100, random_seed=42, gpu=0, regularization_acceleration=True, edge_subset_sz=1000000):
        if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs < 1:
            raise ValueError('epochs must be a positive integer.')
        if edge_subset_sz < 1:
            raise ValueError('edge_subset_sz must be positive.')
        adata_preprocessed, spatial_graph = self.adata_preprocessed, self.spatial_graph
        """
        Training the Deep GraphInfomax Model
        :param embedding_save_filepath: the default save path for the low-dimensional embeddings
        :type embedding_save_filepath: class:`str`
        :param spatial_regularization_strength: the strength for spatial regularization
        :type spatial_regularization_strength: float, optional, default: 0.1
        :param z_dim: the size of latent dimension
        :type z_dim: int, optional, default: 50
        :param lr: the learning rate for model optimization
        :type lr: float, optional, default: 1e-3
        :param epochs: the max epoch number 
        :type epochs: int, optional, default: 1000
        :param max_patience: the tolerance epoch number without training loss decrease
        :type max_patience: int, optional, default: 50
        :param min_stop: the minimum epoch number for training before any early stop
        :type min_stop: int, optional, default: 100
        :param random_seed: the random seed
        :type random_seed: int, optional, default: 42
        :param gpu: the index for gpu device that will be used for model training, if no gpu detected, cpu will be used.
        :type gpu: int, optional, default: 0
        :param regularization_acceleration: whether or not accelerate the calculation of regularization loss using edge subsetting strategy
        :type regularization_acceleration: bool, optional, default: True
        :param edge_subset_sz: the edge subset size for regularization acceleration
        :type edge_subset_sz: int, optional, default: 1000000
        :return: low dimensional embeddings for the ST data, shape: n_cells x z_dim
        :rtype: class:`numpy.ndarray`
        """
        if not adata_preprocessed:
            print("The data has not been preprocessed, please run preprocessing_data() method first!")
            return
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

        device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
        model = DeepGraphInfomax(
            hidden_channels=z_dim, encoder=GraphEncoder(adata_preprocessed.shape[1], z_dim),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption).to(device)

        expr = adata_preprocessed.X.todense() if type(adata_preprocessed.X).__module__ != np.__name__ else adata_preprocessed.X
        expr = torch.tensor(expr).float().to(device)

        edge_list = sparse_mx_to_torch_edge_list(spatial_graph).to(device)

        model.train()
        min_loss = np.inf
        patience = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_params = copy.deepcopy(model.state_dict())

        for epoch in range(epochs):
            train_loss = 0.0
            torch.set_grad_enabled(True)
            optimizer.zero_grad()
            z, neg_z, summary = model(expr, edge_list)
            loss = model.loss(z, neg_z, summary)

            coords = torch.tensor(adata_preprocessed.obsm['spatial']).float().to(device)
            if regularization_acceleration or adata_preprocessed.shape[0] > 5000:
                cell_random_subset_1, cell_random_subset_2 = torch.randint(0, z.shape[0], (edge_subset_sz,)).to(
                    device), torch.randint(0, z.shape[0], (edge_subset_sz,)).to(device)
                z1, z2 = torch.index_select(z, 0, cell_random_subset_1), torch.index_select(z, 0, cell_random_subset_2)
                c1, c2 = torch.index_select(coords, 0, cell_random_subset_1), torch.index_select(coords, 0,
                                                                                                 cell_random_subset_2)
                pdist = torch.nn.PairwiseDistance(p=2, eps=0)

                z_dists = pdist(z1, z2)
                z_dists = z_dists / torch.clamp(torch.max(z_dists), min=torch.finfo(z_dists.dtype).eps)

                sp_dists = pdist(c1, c2)
                sp_dists = sp_dists / torch.clamp(torch.max(sp_dists), min=torch.finfo(sp_dists.dtype).eps)
                n_items = z_dists.size(dim=0)
            else:
                z_dists = torch.cdist(z, z, p=2)
                z_dists = torch.div(
                    z_dists,
                    torch.clamp(torch.max(z_dists), min=torch.finfo(z_dists.dtype).eps),
                ).to(device)
                sp_dists = torch.cdist(coords, coords, p=2)
                sp_dists = torch.div(
                    sp_dists,
                    torch.clamp(torch.max(sp_dists), min=torch.finfo(sp_dists.dtype).eps),
                ).to(device)
                n_items = z.size(dim=0) * z.size(dim=0)

            penalty_1 = torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items).to(device)
            loss = loss + spatial_regularization_strength * penalty_1

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if train_loss > min_loss:
                patience += 1
            else:
                patience = 0
                min_loss = train_loss
                best_params = copy.deepcopy(model.state_dict())
            if epoch % 10 == 1:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {str(train_loss)}")
            if patience > max_patience and epoch > min_stop:
                break

        model.load_state_dict(best_params)

        z, _, _ = model(expr, edge_list)
        embedding = z.cpu().detach().numpy()
        save_dir = os.path.dirname(embedding_save_filepath)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savetxt(embedding_save_filepath, embedding[:, :], delimiter="\t")
        print(f"Training complete!\nEmbedding is saved at {embedding_save_filepath}")

        self.embedding = embedding
        return embedding

    def segmentation(self, domain_label_save_filepath="./domains.tsv", n_neighbors=50, resolution=1.0):
        """
        Perform domain segmentation for ST data using Leiden clustering with low-dimensional embeddings as input
        :param domain_label_save_filepath: the default save path for the domain labels
        :type domain_label_save_filepath: class:`str`, optional, default: "./domains.tsv"
        :param n_neighbors: The size of local neighborhood (in terms of number of neighboring data
        points) used for manifold approximation. See `https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.neighbors.html` for detail
        :type n_neighbors: int, optional, default: 50
        :param resolution: A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters. See `https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.leiden.html` for detail
        :type resolution: float, optional, default: 1.0
        """
        error_message = "No embedding found, please ensure you have run train() method before segmentation!"
        try:
            print("Performing domain segmentation")
            embedding_adata = anndata.AnnData(self.embedding)
            sc.pp.neighbors(embedding_adata, n_neighbors=n_neighbors, use_rep='X')
            sc.tl.leiden(embedding_adata, resolution=float(resolution))
            domains = embedding_adata.obs["leiden"].cat.codes

            save_dir = os.path.dirname(domain_label_save_filepath)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.savetxt(domain_label_save_filepath, domains, fmt='%d', header='', footer='', comments='')
            print(f"Segmentation complete, domain labels of cells or spots saved at {domain_label_save_filepath} !")
            self.domains = domains

        except NameError:
            print(error_message)
        except AttributeError:
            print(error_message)

    def plot_segmentation(self, segmentation_figure_save_filepath="./domain_segmentation.pdf", colormap="tab20", scatter_sz=1., rsz=4., csz=4., wspace=.4, hspace=.5, left=0.125, right=0.9, bottom=0.1, top=0.9):
        """
        Plot the domain segmentation for ST data in spatial
        :param segmentation_figure_save_filepath: the default save path for the figure
        :type segmentation_figure_save_filepath: class:`str`, optional, default: "./domain_segmentation.pdf"
        :param colormap: The colormap to use. See `https://matplotlib.org/stable/tutorials/colors/colormaps.html` for full list of colormaps
        :type colormap: str, optional, default: tab20
        :param scatter_sz: The marker size in points**2
        :type scatter_sz: float, optional, default: 1.0
        :param rsz: row size of the figure in inches, default: 4.0
        :type rsz: float, optional
        :param csz: column size of the figure in inches, default: 4.0
        :type csz: float, optional
        :param wspace: the amount of width reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
        :type wspace: float, optional
        :param hspace: the amount of height reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
        :type hspace: float, optional
        :param left: the leftmost position of the subplots of the figure in fraction, default: 0.125
        :type left: float, optional
        :param right: the rightmost position of the subplots of the figure in fraction, default: 0.9
        :type right: float, optional
        :param bottom: the bottom position of the subplots of the figure in fraction, default: 0.1
        :type bottom: float, optional
        :param top: the top position of the subplots of the figure in fraction, default: 0.9
        :type top: float, optional
        """
        error_message = "No segmentation data found, please ensure you have run the segmentation() method."
        try:
            fig, ax = self.prepare_figure(rsz=rsz, csz=csz, wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom, top=top)

            pred_clusters = np.array(self.domains).astype(int)
            uniq_pred = np.unique(pred_clusters)
            n_cluster = len(uniq_pred)
            x, y = self.adata_preprocessed.obsm["spatial"][:, 0], self.adata_preprocessed.obsm["spatial"][:, 1]
            cmap = plt.get_cmap(colormap)
            for cid, cluster in enumerate(uniq_pred):
                color = cmap((cid * (n_cluster / (n_cluster - 1.0))) / n_cluster)
                ind = pred_clusters == cluster
                ax.scatter(x[ind], y[ind], s=scatter_sz, color=color, label=cluster, marker=".")
            ax.set_facecolor("none")
            ax.invert_yaxis()
            ax.set_title("Domain Segmentation", fontsize=14)
            box = ax.get_position()
            height_ratio = .8
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * height_ratio])
            lgnd = ax.legend(loc='center left', fontsize=8, bbox_to_anchor=(1, 0.5), scatterpoints=1, handletextpad=0.1,
                             borderaxespad=.1, ncol=int(math.ceil(n_cluster/10)))
            for handle in lgnd.legendHandles:
                handle._sizes = [8]

            save_dir = os.path.dirname(segmentation_figure_save_filepath)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(segmentation_figure_save_filepath, dpi=300)
            print(f"Plotting complete, segmentation figure saved at {segmentation_figure_save_filepath} !")
            plt.close('all')
        except NameError:
            print(error_message)
        except AttributeError:
            print(error_message)

    def pseudo_Spatiotemporal_Map(self, pSM_values_save_filepath="./pSM_values.tsv", n_neighbors=20, resolution=1.0):
        """
        Perform pseudo-Spatiotemporal Map for ST data
        :param pSM_values_save_filepath: the default save path for the pSM values
        :type pSM_values_save_filepath: class:`str`, optional, default: "./pSM_values.tsv"
        :param n_neighbors: The size of local neighborhood (in terms of number of neighboring data
        points) used for manifold approximation. See `https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.neighbors.html` for detail
        :type n_neighbors: int, optional, default: 20
        :param resolution: A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters. See `https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.leiden.html` for detail
        :type resolution: float, optional, default: 1.0
        """
        error_message = "No embedding found, please ensure you have run train() method before calculating pseudo-Spatiotemporal Map!"
        max_cell_for_subsampling = 5000
        try:
            print("Performing pseudo-Spatiotemporal Map")
            adata = anndata.AnnData(self.embedding)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
            sc.tl.umap(adata)
            sc.tl.leiden(adata, resolution=resolution)
            sc.tl.paga(adata)
            if adata.shape[0] < max_cell_for_subsampling:
                sub_adata_x = adata.X
            else:
                indices = np.arange(adata.shape[0])
                selected_ind = np.random.choice(indices, max_cell_for_subsampling, False)
                sub_adata_x = adata.X[selected_ind, :]
            sum_dists = distance_matrix(sub_adata_x, sub_adata_x).sum(axis=1)
            root_in_subset = int(np.argmax(sum_dists))
            adata.uns['iroot'] = (
                root_in_subset
                if adata.shape[0] < max_cell_for_subsampling
                else int(selected_ind[root_in_subset])
            )
            sc.tl.diffmap(adata)
            sc.tl.dpt(adata)
            pSM_values = adata.obs['dpt_pseudotime'].to_numpy()
            save_dir = os.path.dirname(pSM_values_save_filepath)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.savetxt(pSM_values_save_filepath, pSM_values, fmt='%.5f', header='', footer='', comments='')
            print(f"pseudo-Spatiotemporal Map(pSM) calculation complete, pSM values of cells or spots saved at {pSM_values_save_filepath}!")
            self.pSM_values = pSM_values
        except NameError:
            print(error_message)
        except AttributeError:
            print(error_message)

    def plot_pSM(self, pSM_figure_save_filepath="./pseudo-Spatiotemporal-Map.pdf", colormap='roma', scatter_sz=1., rsz=4., csz=4., wspace=.4, hspace=.5, left=0.125, right=0.9, bottom=0.1, top=0.9):
        """
        Plot the domain segmentation for ST data in spatial
        :param pSM_figure_save_filepath: the default save path for the figure
        :type pSM_figure_save_filepath: class:`str`, optional, default: "./Spatiotemporal-Map.pdf"
        :param colormap: The colormap to use. See `https://www.fabiocrameri.ch/colourmaps-userguide/` for name list of colormaps
        :type colormap: str, optional, default: roma
        :param scatter_sz: The marker size in points**2
        :type scatter_sz: float, optional, default: 1.0
        :param rsz: row size of the figure in inches, default: 4.0
        :type rsz: float, optional
        :param csz: column size of the figure in inches, default: 4.0
        :type csz: float, optional
        :param wspace: the amount of width reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
        :type wspace: float, optional
        :param hspace: the amount of height reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
        :type hspace: float, optional
        :param left: the leftmost position of the subplots of the figure in fraction, default: 0.125
        :type left: float, optional
        :param right: the rightmost position of the subplots of the figure in fraction, default: 0.9
        :type right: float, optional
        :param bottom: the bottom position of the subplots of the figure in fraction, default: 0.1
        :type bottom: float, optional
        :param top: the top position of the subplots of the figure in fraction, default: 0.9
        :type top: float, optional
        """
        error_message = "No pseudo Spatiotemporal Map data found, please ensure you have run the pseudo_Spatiotemporal_Map() method."
        try:
            fig, ax = self.prepare_figure(rsz=rsz, csz=csz, wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom, top=top)
            x, y = self.adata_preprocessed.obsm["spatial"][:, 0], self.adata_preprocessed.obsm["spatial"][:, 1]
            st = ax.scatter(x, y, s=scatter_sz, c=self.pSM_values, cmap=f"cmc.{colormap}", marker=".")
            ax.invert_yaxis()
            clb = fig.colorbar(st)
            clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
            ax.set_title("pseudo-Spatiotemporal Map", fontsize=14)
            ax.set_facecolor("none")

            save_dir = os.path.dirname(pSM_figure_save_filepath)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(pSM_figure_save_filepath, dpi=300)
            print(f"Plotting complete, pseudo-Spatiotemporal Map figure saved at {pSM_figure_save_filepath} !")
            plt.close('all')
        except NameError:
            print(error_message)
        except AttributeError:
            print(error_message)

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphEncoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=False)
        self.prelu = nn.PReLU(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.prelu2 = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        x = self.conv2(x, edge_index)
        x = self.prelu2(x)
        return x

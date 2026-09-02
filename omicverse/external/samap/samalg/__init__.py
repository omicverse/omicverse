import gc
import numpy as np
from anndata import AnnData
import anndata
import scipy.sparse as sp
import time
from sklearn.preprocessing import Normalizer, StandardScaler
import pickle
import pandas as pd
from . import utilities as ut
import sklearn.manifold as man
import sklearn.utils.sparsefuncs as sf
from packaging import version
import warnings
from numba.core.errors import NumbaWarning
from sklearn.utils.extmath import svd_flip
from scipy.linalg import block_diag
import harmonypy
from ...._anndata_compat import anndata_version as _ad_version

warnings.filterwarnings("ignore", category=NumbaWarning)

__version__ = "1.0.0"

"""
Copyright 2018, Alexander J. Tarashansky, All rights reserved.
Email: <tarashan@stanford.edu>
"""

class SAM(object):
    """Self-Assembling Manifolds single-cell RNA sequencing analysis tool.

    SAM iteratively rescales the input gene expression matrix to emphasize
    genes that are spatially variable along the intrinsic manifold of the data.
    It outputs the gene weights, nearest neighbor matrix, and a 2D projection.

    Parameters
    ----------
    counts : tuple or list (scipy.sparse matrix, numpy.ndarray,numpy.ndarray),
        OR tuple or list (numpy.ndarray, numpy.ndarray,numpy.ndarray), OR
        pandas.DataFrame, OR anndata.AnnData

        If a tuple or list, it should contain the gene expression data
        (scipy.sparse or numpy.ndarray) matrix (cells x genes), numpy array of
        gene IDs, and numpy array of cell IDs in that order.

        If a pandas.DataFrame, it should be (cells x genes)

        Only use this argument if you want to pass in preloaded data. Otherwise
        use one of the load functions.


    Attributes
    ----------

    preprocess_args: dict
        Dictionary of arguments used for the 'preprocess_data' function.

    run_args: dict
        Dictionary of arguments used for the 'run' function.

    adata_raw: AnnData
        An AnnData object containing the raw, unfiltered input data.

    adata: AnnData
        An AnnData object containing all processed data and SAM outputs.

    """

    def __init__(self, counts=None, inplace=False):

        if isinstance(counts, tuple) or isinstance(counts, list):
            raw_data, all_gene_names, all_cell_names = counts
            if isinstance(raw_data, np.ndarray):
                raw_data = sp.csr_matrix(raw_data)

            self.adata_raw = AnnData(
                X=raw_data,
                obs={"obs_names": all_cell_names},
                var={"var_names": all_gene_names},
            )

        elif isinstance(counts, pd.DataFrame):
            raw_data = sp.csr_matrix(counts.values)
            all_gene_names = np.array(list(counts.columns.values))
            all_cell_names = np.array(list(counts.index.values))

            self.adata_raw = AnnData(
                X=raw_data,
                obs={"obs_names": all_cell_names},
                var={"var_names": all_gene_names},
            )

        elif isinstance(counts, AnnData):
            all_cell_names = np.array(list(counts.obs_names))
            all_gene_names = np.array(list(counts.var_names))
            if counts.is_view:
                counts = counts.copy()
                
            if inplace:
                self.adata_raw = counts
            else:
                self.adata_raw = counts.copy()            

        elif counts is not None:
            raise Exception(
                "'counts' must be either a tuple/list of "
                "(data,gene IDs,cell IDs) or a Pandas DataFrame of"
                "cells x genes"
            )

        if counts is not None:
            if np.unique(all_gene_names).size != all_gene_names.size:
                self.adata_raw.var_names_make_unique()
            if np.unique(all_cell_names).size != all_cell_names.size:
                self.adata_raw.obs_names_make_unique()

            if inplace:
                self.adata = self.adata_raw
            else:
                self.adata = self.adata_raw.copy()

            if "X_disp" not in self.adata_raw.layers.keys():
                self.adata.layers["X_disp"] = self.adata.X

        self.run_args = {}
        self.preprocess_args = {}

    def preprocess_data(
        self,
        div=1,
        downsample=0,
        sum_norm="cell_median",
        norm="log",
        min_expression=1,
        thresh_low=0.0,
        thresh_high=0.96,
        thresh = None,
        filter_genes=True,
    ):
        """Log-normalizes and filters the expression data.

        Parameters
        ----------

        div : float, optional, default 1
            The factor by which the gene expression will be divided prior to
            normalization (e.g. log normalization).

        downsample : float, optional, default 0
            The factor by which to randomly downsample the data. If 0, the
            data will not be downsampled.

        sum_norm : str or float, optional, default "cell_median"
            If a float, the total number of transcripts in each cell will be
            normalized to this value prior to normalization and filtering.
            Otherwise, nothing happens. If 'cell_median', each cell is
            normalized to have the median total read count per cell. If
            'gene_median', each gene is normalized to have the median total
            read count per gene.

        norm : str, optional, default 'log'
            If 'log', log-normalizes the expression data. If 'ftt', applies the
            Freeman-Tukey variance-stabilization transformation. If
            'multinomial', applies the Pearson-residual transformation (this is
            experimental and should only be used for raw, un-normalized UMI
            datasets). If None, the data is not normalized.

        min_expression : float, optional, default 1
            The threshold above which a gene is considered
            expressed. Gene expression values less than 'min_expression' are
            set to zero.

        thresh_low : float, optional, default 0.0
            Keep genes expressed in greater than 'thresh_low'*100 % of cells,
            where a gene is considered expressed if its expression value
            exceeds 'min_expression'.

        thresh_high : float, optional, default 0.96
            Keep genes expressed in less than 'thresh_high'*100 % of cells,
            where a gene is considered expressed if its expression value
            exceeds 'min_expression'.

        filter_genes : bool, optional, default True
            Setting this to False turns off filtering operations.

        """
        if thresh is not None:
            thresh_low = thresh
            thresh_high = 1-thresh

        self.preprocess_args = {
            "div": div,
            "sum_norm": sum_norm,
            "norm": norm,
            "min_expression": min_expression,
            "thresh_low": thresh_low,
            "thresh_high":thresh_high,
            "filter_genes": filter_genes,
        }
        
        self.run_args = self.adata.uns.get('run_args',{})
        # load data
        try:
            D = self.adata_raw.X
            self.adata = self.adata_raw.copy()

        except AttributeError:
            print("No data loaded")

        D = self.adata.X
        if isinstance(D, np.ndarray):
            D = sp.csr_matrix(D, dtype="float32")
        else:
            if str(D.dtype) != "float32":
                D = D.astype("float32")
            D.sort_indices()

        if D.getformat() == "csc":
            D = D.tocsr()

        # sum-normalize
        if sum_norm == "cell_median" and norm != "multinomial":
            s = D.sum(1).A.flatten()
            sum_norm = np.median(s)
            D = D.multiply(1 / s[:, None] * sum_norm).tocsr()
        elif sum_norm == "gene_median" and norm != "multinomial":
            s = D.sum(0).A.flatten()
            sum_norm = np.median(s[s > 0])
            s[s == 0] = 1
            D = D.multiply(1 / s[None, :] * sum_norm).tocsr()

        elif sum_norm is not None and norm != "multinomial":
            D = D.multiply(1 / D.sum(1).A.flatten()[:, None] * sum_norm).tocsr()

        # normalize
        self.adata.X = D
        if norm is None:
            D.data[:] = D.data / div

        elif norm.lower() == "log":
            D.data[:] = np.log2(D.data / div + 1)

        elif norm.lower() == "ftt":
            D.data[:] = np.sqrt(D.data / div) + np.sqrt(D.data / div + 1) - 1

        elif norm.lower() == "asin":
            D.data[:] = np.arcsinh(D.data / div)
        elif norm.lower() == "multinomial":
            ni = D.sum(1).A.flatten()  # cells
            pj = (D.sum(0) / D.sum()).A.flatten()  # genes
            col = D.indices
            row = []
            for i in range(D.shape[0]):
                row.append(i * np.ones(D.indptr[i + 1] - D.indptr[i]))
            row = np.concatenate(row).astype("int32")
            mu = sp.coo_matrix((ni[row] * pj[col], (row, col))).tocsr()
            mu2 = mu.copy()
            mu2.data[:] = mu2.data ** 2
            mu2 = mu2.multiply(1 / ni[:, None])
            mu.data[:] = (D.data - mu.data) / np.sqrt(mu.data - mu2.data)

            self.adata.X = mu
            if sum_norm is None:
                sum_norm = np.median(ni)
            D = D.multiply(1 / ni[:, None] * sum_norm).tocsr()
            D.data[:] = np.log2(D.data / div + 1)

        else:
            D.data[:] = D.data / div

        # zero-out low-expressed genes
        idx = np.where(D.data <= min_expression)[0]
        D.data[idx] = 0

        # filter genes
        idx_genes = np.arange(D.shape[1])
        if filter_genes:
            a, ct = np.unique(D.indices, return_counts=True)
            c = np.zeros(D.shape[1])
            c[a] = ct

            keep = np.where(
                np.logical_and(c / D.shape[0] > thresh_low, c / D.shape[0] <= thresh_high)
            )[0]

            idx_genes = np.array(list(set(keep) & set(idx_genes)))

        mask_genes = np.zeros(D.shape[1], dtype="bool")
        mask_genes[idx_genes] = True

        self.adata.X = self.adata.X.multiply(mask_genes[None, :]).tocsr()
        self.adata.X.eliminate_zeros()
        self.adata.var["mask_genes"] = mask_genes

        if norm == "multinomial":
            self.adata.layers["X_disp"] = D.multiply(mask_genes[None, :]).tocsr()
            self.adata.layers["X_disp"].eliminate_zeros()
        else:
            self.adata.layers["X_disp"] = self.adata.X

        self.calculate_mean_var()

        self.adata.uns["preprocess_args"] = self.preprocess_args
        self.adata.uns['run_args'] = self.run_args

    def calculate_mean_var(self,adata=None):
        if adata is None:
            adata = self.adata

        if sp.issparse(adata.X):
            mu,var = sf.mean_variance_axis(adata.X,axis=0)
        else:
            mu = adata.X.mean(0)
            var = adata.X.var(0)
            
        adata.var['means'] = mu
        adata.var['variances'] = var
        
    def get_avg_obsm(self, keym, keyl):
        clu = self.get_labels_un(keyl)
        cl = self.get_labels(keyl)
        x = []
        for i in range(clu.size):
            x.append(self.adata.obsm[keym][cl == clu[i]].mean(0))
        x = np.vstack(x)
        return x

    def get_labels_un(self, key):
        if key not in list(self.adata.obs.keys()):
            print("Key does not exist in `obs`.")
            return np.array([])
        else:
            return np.array(list(np.unique(self.adata.obs[key])))

    def get_labels(self, key):
        if key not in list(self.adata.obs.keys()):
            print("Key does not exist in `obs`.")
            return np.array([])
        else:
            return np.array(list(self.adata.obs[key]))

    def get_cells(self, label, key):
        """Retrieves cells of a particular annotation.

        Parameters
        ----------
        label - The annotation to retrieve
        key - The key in `obs` from which to retrieve the annotation.

        """
        if key not in list(self.adata.obs.keys()):
            print("Key does not exist in `obs`.")
            return np.array([])
        else:
            return np.array(
                list(self.adata.obs_names[np.array(list(self.adata.obs[key])) == label])
            )

    def load_data(
        self,
        filename,
        transpose=True,
        save_sparse_file=None,
        sep=",",
        calculate_avg=False,
        **kwargs
    ):
        """Loads the specified data file. The file can be a table of
        read counts (i.e. '.csv' or '.txt'), with genes as rows and cells
        as columns by default. The file can also be a pickle file (output from
        'save_sparse_data') or an h5ad file (output from 'save_anndata').

        This function that loads the file specified by 'filename'.

        Parameters
        ----------
        filename - string
            The path to the tabular raw expression counts file.

        sep - string, optional, default ','
            The delimeter used to read the input data table. By default
            assumes the input table is delimited by commas.

        save_sparse_file - str, optional, default None
            Path to which the sparse data will be saved ('.h5ad').
            Writes the SAM 'adata_raw' object to a h5ad file (the native AnnData
            file format) for faster loading in the future.

        transpose - bool, optional, default True
            By default, assumes file is (genes x cells). Set this to False if
            the file has dimensions (cells x genes).

        calculate_avg - bool, optional, default True
            If nearest neighbors are already calculated in the .h5ad file,
            setting this parameter to True performs knn averaging and stores
            the result in adata.layers['X_knn_avg']. This is a fairly dense
            matrix, so set this to False if you do not need the averaged
            expressions.

        """
        if filename.split(".")[-1] == "p":
            raw_data, all_cell_names, all_gene_names = pickle.load(open(filename, "rb"))

            if transpose:
                raw_data = raw_data.T
                if raw_data.getformat() == "csc":
                    print("Converting sparse matrix to csr format...")
                    raw_data = raw_data.tocsr()

            save_sparse_file = None
        elif filename.split(".")[-1] != "h5ad":
            df = pd.read_csv(filename, sep=sep, index_col=0, **kwargs)
            if transpose:
                dataset = df.T
            else:
                dataset = df

            raw_data = sp.csr_matrix(dataset.values)
            all_cell_names = np.array(list(dataset.index.values))
            all_gene_names = np.array(list(dataset.columns.values))

        if filename.split(".")[-1] != "h5ad":
            self.adata_raw = AnnData(
                X=raw_data,
                obs={"obs_names": all_cell_names},
                var={"var_names": all_gene_names},
            )

            if np.unique(all_gene_names).size != all_gene_names.size:
                self.adata_raw.var_names_make_unique()
            if np.unique(all_cell_names).size != all_cell_names.size:
                self.adata_raw.obs_names_make_unique()

            self.adata = self.adata_raw.copy()
            self.adata.layers["X_disp"] = raw_data

        else:
            self.adata = anndata.read_h5ad(filename, **kwargs)
            if self.adata.raw is not None:
                self.adata_raw = AnnData(X=self.adata.raw.X)
                self.adata_raw.var_names = self.adata.var_names
                self.adata_raw.obs_names = self.adata.obs_names
                self.adata_raw.obs = self.adata.obs

                if version.parse('.'.join(map(str, _ad_version()))) >= version.parse("0.7rc1"):
                    del self.adata.raw
                else:
                    self.adata.raw = None

                if (
                    "X_knn_avg" not in self.adata.layers.keys()
                    and "connectivities" in self.adata.obsp.keys()
                    and calculate_avg
                ):
                    self.dispersion_ranking_NN(save_avgs=True)
            else:
                self.adata_raw = self.adata

            if "X_disp" not in list(self.adata.layers.keys()):
                self.adata.layers["X_disp"] = self.adata.X
            save_sparse_file = None
        filename = '.'.join(filename.split('.')[:-1])+'.h5ad'
        self.adata.uns['path_to_file'] = filename
        self.adata_raw.uns['path_to_file'] = filename        
        if save_sparse_file is not None:
            if save_sparse_file.split(".")[-1] == "p":
                self.save_sparse_data(save_sparse_file)
            elif save_sparse_file.split(".")[-1] == "h5ad":
                self.save_anndata(save_sparse_file)

    def save_anndata(self, fname='', save_knn=False, **kwargs):
        """Saves `adata_raw` to a .h5ad file (AnnData's native file format).

        Parameters
        ----------
        fname - string, default ''
            The filename of the output file. If empty, defaults to `.adata.uns['path_to_file']`.

        save_knn - bool, optional, default = False
            If True, saves `.layers['X_knn_avg']`. If False, does not save
            this layer. Default value is set to False as the nearest-neighbor
            averaged expression values can be quite dense.

        """
        if not save_knn:
            try:
                Xknn = self.adata.layers["X_knn_avg"]
                del self.adata.layers["X_knn_avg"]
            except:
                0
        if fname == '':
            try:
                fname = self.adata.uns['path_to_file']
            except KeyError:
                raise KeyError('Path to file not known.')
                
        x = self.adata
        x.raw = self.adata_raw
        
        # fixing weird issues when index name is an integer.
        for y in [x.obs.columns,x.var.columns,
                  x.obs.index,x.var.index,
                  x.raw.var.index,x.raw.var.columns]:
            y.name = str(y.name) if y.name is not None else None

        x.write_h5ad(fname, **kwargs)
        if version.parse('.'.join(map(str, _ad_version()))) >= version.parse("0.7rc1"):
            del x.raw
        else:
            x.raw = None

        try:
            self.adata.layers["X_knn_avg"] = Xknn
        except:
            0

    def load_var_annotations(self, aname, sep=",", key_added="annotations"):
        """Loads gene annotations.

        Loads the gene annotations into .adata_raw.var and .adata.var. The keys
        added correspond to the column labels in the input table.

        Parameters
        ----------
        aname - string or pandas.DataFrame
            If string, it is the path to the annotations file, which should be
            a table with the first column being the gene IDs and the first row
            being the column names for the annotations. Alternatively, you can
            directly pass in a preloaded pandas.DataFrame.

        sep - string, default ','
            The delimeter used in the file. Ignored if passing in a preloaded
            pandas.DataFrame.

        """
        if isinstance(aname, pd.DataFrame):
            ann = aname
        else:
            ann = pd.read_csv(aname, sep=sep, index_col=0)

        for i in range(ann.shape[1]):
            self.adata_raw.var[ann.columns[i]] = ann[ann.columns[i]]
            self.adata.var[ann.columns[i]] = ann[ann.columns[i]]

    def load_obs_annotations(self, aname, sep=","):
        """Loads cell annotations.

        Loads the cell annotations into .adata_raw.obs and .adata.obs. The keys
        added correspond to the column labels in the input table.

        Parameters
        ----------
        aname - string or pandas.DataFrame
            If string, it is the path to the annotations file, which should be
            a table with the first column being the cell IDs and the first row
            being the column names for the annotations. Alternatively, you can
            directly pass in a preloaded pandas.DataFrame.

        sep - string, default ','
            The delimeter used in the file. Ignored if passing in a preloaded
            pandas.DataFrame.

        """
        if isinstance(aname, pd.DataFrame):
            ann = aname
        else:
            ann = pd.read_csv(aname, sep=sep, index_col=0)

        for i in range(ann.shape[1]):
            self.adata_raw.obs[ann.columns[i]] = ann[ann.columns[i]]
            self.adata.obs[ann.columns[i]] = ann[ann.columns[i]]

    def scatter(
        self,
        projection=None,
        c=None,
        colorspec=None,
        cmap="rainbow",
        linewidth=0.0,
        edgecolor="k",
        axes=None,
        colorbar=True,
        s=10,
        **kwargs
    ):

        """Display a scatter plot.
        Displays a scatter plot using the SAM projection or another input
        projection.

        Parameters
        ----------
        projection - string, numpy.ndarray, default None
            A case-sensitive string indicating the projection to display (a key
            in adata.obsm) or a 2D numpy array with cell coordinates. If None,
            projection defaults to UMAP.

        c - string, numpy.ndarray, default None
            Categorical data to be mapped to a colormap and overlaid on top of
            the projection. Can be a key from adata.obs or a 1D numpy array.

        colorspec - string, numpy.ndarray, default None
            A string specifying a color or an array specifying the color
            for each point (can be strings, RGBA, RGB, etc). Colorbar will be
            turned off if `colorspec` is not None.

        axes - matplotlib axis, optional, default None
            Plot output to the specified, existing axes. If None, create new
            figure window.

        **kwargs - all keyword arguments in matplotlib.pyplot.scatter are eligible.
        """

        try:
            import matplotlib.pyplot as plt

            if isinstance(projection, str):
                try:
                    dt = self.adata.obsm[projection]
                except KeyError:
                    print(
                        "Please create a projection first using run_umap or" "run_tsne"
                    )

            elif projection is None:
                try:
                    dt = self.adata.obsm["X_umap"]
                except KeyError:
                    try:
                        dt = self.adata.obsm["X_tsne"]
                    except KeyError:
                        print(
                            "Please create either a t-SNE or UMAP projection" "first."
                        )
                        return
            else:
                dt = projection

            if axes is None:
                plt.figure()
                axes = plt.gca()

            if colorspec is not None:
                axes.scatter(
                    dt[:, 0],
                    dt[:, 1],
                    s=s,
                    linewidth=linewidth,
                    edgecolor=edgecolor,
                    c=colorspec,
                    **kwargs
                )
            elif c is None:
                axes.scatter(
                    dt[:, 0],
                    dt[:, 1],
                    s=s,
                    linewidth=linewidth,
                    edgecolor=edgecolor,
                    **kwargs
                )
            else:

                if isinstance(c, str):
                    try:
                        c = self.get_labels(c)
                    except KeyError:
                        0  # do nothing

                if (isinstance(c[0], str) or isinstance(c[0], np.str_)) and (
                    isinstance(c, np.ndarray) or isinstance(c, list)
                ):
                    i = ut.convert_annotations(c)
                    ui, ai = np.unique(i, return_index=True)
                    cax = axes.scatter(
                        dt[:, 0],
                        dt[:, 1],
                        c=i,
                        cmap=cmap,
                        s=s,
                        linewidth=linewidth,
                        edgecolor=edgecolor,
                        **kwargs
                    )

                    if colorbar:
                        cbar = plt.colorbar(cax, ax=axes, ticks=ui)
                        cbar.ax.set_yticklabels(c[ai])
                else:
                    if not (isinstance(c, np.ndarray) or isinstance(c, list)):
                        colorbar = False
                    i = c

                    cax = axes.scatter(
                        dt[:, 0],
                        dt[:, 1],
                        c=i,
                        cmap=cmap,
                        s=s,
                        linewidth=linewidth,
                        edgecolor=edgecolor,
                        **kwargs
                    )

                    if colorbar:
                        plt.colorbar(cax, ax=axes)
            return axes
        except ImportError:
            print("matplotlib not installed!")

    def show_gene_expression(self, gene, avg=True, axes=None, **kwargs):
        """Display a gene's expressions.
        Displays a scatter plot using the SAM projection or another input
        projection with a particular gene's expressions overlaid.
        Parameters
        ----------
        gene - string
            a case-sensitive string indicating the gene expression pattern
            to display.
        avg - bool, optional, default True
            If True, the plots use the k-nearest-neighbor-averaged expression
            values to smooth out noisy expression patterns and improves
            visualization.
        axes - matplotlib axis, optional, default None
            Plot output to the specified, existing axes. If None, create new
            figure window.
        **kwargs - all keyword arguments in 'SAM.scatter' are eligible.
        """
        all_gene_names = np.array(list(self.adata.var_names))
        cell_names = np.array(list(self.adata.obs_names))
        all_cell_names = np.array(list(self.adata_raw.obs_names))
        idx2 = np.where(np.in1d(all_cell_names, cell_names))[0]
        idx = np.where(all_gene_names == gene)[0]
        name = gene
        if idx.size == 0:
            print(
                "Gene note found in the filtered dataset. Note that genes "
                "are case sensitive."
            )
            return

        if avg:
            a = self.adata.layers["X_knn_avg"][:, idx].toarray().flatten()
            if a.sum() == 0:
                a = self.adata_raw.X[:, idx].toarray().flatten()[idx2]
                try:
                    norm = self.preprocess_args["norm"]
                except KeyError:
                    norm = "log"
                if norm is not None:
                    if norm.lower() == "log":
                        a = np.log2(a + 1)

                    elif norm.lower() == "ftt":
                        a = np.sqrt(a) + np.sqrt(a + 1)
                    elif norm.lower() == "asin":
                        a = np.arcsinh(a)
        else:
            a = self.adata_raw.X[:, idx].toarray().flatten()[idx2]
            try:
                norm = self.preprocess_args["norm"]
            except KeyError:
                norm = "log"

            if norm is not None:
                if norm.lower() == "log":
                    a = np.log2(a + 1)

                elif norm.lower() == "ftt":
                    a = np.sqrt(a) + np.sqrt(a + 1)
                elif norm.lower() == "asin":
                    a = np.arcsinh(a)

        axes = self.scatter(c=a, axes=axes, **kwargs)
        axes.set_title(name)

        return axes, a

    def dispersion_ranking_NN(self, nnm=None, num_norm_avg=50, weight_mode='combined',save_avgs=False, adata=None):
        """Computes the spatial dispersion factors for each gene.

        Parameters
        ----------
        nnm - scipy.sparse, default None
            Square cell-to-cell nearest-neighbor matrix. If None, uses the
            nearest neighbor matrix in .adata.obsp['connectivities']

        num_norm_avg - int, optional, default 50
            The top 'num_norm_avg' dispersions are averaged to determine the
            normalization factor when calculating the weights. This ensures
            that outlier genes do not significantly skew the weight
            distribution.

        Returns:
        -------
        weights - ndarray, float
            The vector of gene weights.
        """
        if adata is None:
            adata = self.adata

        if nnm is None:
            nnm = adata.obsp["connectivities"]
        f = nnm.sum(1).A
        f[f==0]=1
        D_avg = (nnm.multiply(1 / f)).dot(adata.layers["X_disp"])

        if save_avgs:
            adata.layers["X_knn_avg"] = D_avg.copy()

        if sp.issparse(D_avg):
            mu, var = sf.mean_variance_axis(D_avg, axis=0)            
            if weight_mode == 'rms':
                D_avg.data[:]=D_avg.data**2
                mu,_ =sf.mean_variance_axis(D_avg, axis=0)
                mu=mu**0.5
                
            if weight_mode == 'combined':
                D_avg.data[:]=D_avg.data**2
                mu2,_ =sf.mean_variance_axis(D_avg, axis=0)
                mu2=mu2**0.5                
        else:
            mu = D_avg.mean(0)
            var = D_avg.var(0)
            if weight_mode == 'rms':
                mu =(D_avg**2).mean(0)**0.5
            if weight_mode == 'combined':
                mu2 =(D_avg**2).mean(0)**0.5
            
                
        if not save_avgs:        
            del D_avg
            gc.collect()

        if weight_mode == 'dispersion' or weight_mode == 'rms' or weight_mode == 'combined':
            dispersions = np.zeros(var.size)
            dispersions[mu > 0] = var[mu > 0] / mu[mu > 0]
            adata.var["spatial_dispersions"] = dispersions.copy()
            
            if weight_mode == 'combined':
                dispersions2 = np.zeros(var.size)
                dispersions2[mu2 > 0] = var[mu2 > 0] / mu2[mu2 > 0]
                

        elif weight_mode == 'variance':
            dispersions = var
            adata.var["spatial_variances"] = dispersions.copy()
        else:
            raise ValueError('`weight_mode` ',weight_mode,' not recognized.')

        ma = np.sort(dispersions)[-num_norm_avg:].mean()
        dispersions[dispersions >= ma] = ma

        weights = ((dispersions / dispersions.max()) ** 0.5).flatten()
        
        if weight_mode == 'combined':
            ma = np.sort(dispersions2)[-num_norm_avg:].mean()
            dispersions2[dispersions2 >= ma] = ma

            weights2 = ((dispersions2 / dispersions2.max()) ** 0.5).flatten()
            weights = np.vstack((weights,weights2)).max(0)

        return weights

    def run(
        self,
        max_iter=10,
        verbose=True,
        projection="umap",
        stopping_condition=1e-2,
        num_norm_avg=50,
        k=20,
        distance="cosine",
        preprocessing="StandardScaler",
        npcs=150,
        n_genes=3000,
        weight_PCs=False,
        sparse_pca=False,
        proj_kwargs={},
        seed = 0,
        weight_mode='rms',
        components=None,
        batch_key=None
    ):
        """Runs the Self-Assembling Manifold algorithm.

        Parameters
        ----------
        k - int, optional, default 20
            The number of nearest neighbors to identify for each cell.

        distance : string, optional, default 'correlation'
            The distance metric to use when identifying nearest neighbors.
            Can be any of the distance metrics supported by sklearn's 'pdist'.

        max_iter - int, optional, default 10
            The maximum number of iterations SAM will run.

        stopping_condition - float, optional, default 5e-3
            The stopping condition threshold for the RMSE between gene weights
            in adjacent iterations.

        verbose - bool, optional, default True
            If True, the iteration number and error between gene weights in
            adjacent iterations will be displayed.

        projection - str, optional, default 'umap'
            If 'tsne', generates a t-SNE embedding. If 'umap', generates a UMAP
            embedding. Otherwise, no embedding will be generated.

        preprocessing - str, optional, default 'StandardScaler'
            If 'Normalizer', use sklearn.preprocessing.Normalizer, which
            normalizes expression data prior to PCA such that each cell has
            unit L2 norm. If 'StandardScaler', use
            sklearn.preprocessing.StandardScaler, which normalizes expression
            data prior to PCA such that each gene has zero mean and unit
            variance. Otherwise, do not normalize the expression data. We
            recommend using 'StandardScaler' for large datasets and
            'Normalizer' otherwise.
        
        npcs - int, optional, default 150
            Sets the number of principal components to use.
            
        n_genes - int, optional, default 3000
            Sets the number of genes to include prior to PCA. Genes are selected
            based on their SAM weights. If `None`, all genes are selected.
            In this case, it is recommended to set `sparse_pca` equal to True.
            
        num_norm_avg - int, optional, default 50
            The top 'num_norm_avg' dispersions are averaged to determine the
            normalization factor when calculating the weights. This prevents
            genes with large spatial dispersions from skewing the distribution
            of weights.

        sparse_pca - bool, optional, default False
            If True, uses an implementation of PCA that accepts sparse inputs.
            This is worth setting True for large datasets, where memory
            constraints start becoming noticeable.

        weight_PCs - bool, optional, default False
            Scale the principal components by their eigenvalues. If many
            cell populations are expected, it is recommended to set this False
            as the populations may be found on lesser-varying PCs.
            
        weight_mode - str, optional, default 'combined'
            Can be one of 'dispersion', 'variance', 'rms', 'combined'.
                'dispersion' - Gene weights are calculated as the fano factors
                (variance / mean) of kNN-averaged expressions.
                
                'variance' - Gene weights are calculated as the variances of
                kNN-averaged expressions.
                
                'rms' - Gene weights are calculated as a modified fano factor
                (variance / root-mean square) of kNN-averaged expressions.
                This makes the weight more robust to genes with extremely low
                mean expressions.
                
                'combined' - Gene weights are calculated as the maximum of the 
                weights calculated from 'dispersion' and 'rms' modes.
                

        proj_kwargs - dict, optional, default {}
            A dictionary of keyword arguments to pass to the projection
            functions.
        """
        D = self.adata.X
        if k < 5:
            k = 5
        if k > D.shape[0] - 1:
            k = D.shape[0] - 2

        if preprocessing not in ["StandardScaler", "Normalizer", None, "None"]:
            raise ValueError(
                "preprocessing must be 'StandardScaler', 'Normalizer', or None"
            )
        if weight_mode not in ["dispersion", "variance", "rms", "combined"]:
            raise ValueError(
                "weight_mode must be 'dispersion', 'variance', 'rms', or 'combined'."
            )

        if self.adata.layers['X_disp'].min() < 0 and weight_mode == 'dispersion':
            print("`X_disp` layer contains negative values. Setting `weight_mode` to 'rms'.")
            weight_mode = 'rms'

        numcells = D.shape[0]

        if n_genes == None:
            n_genes = self.adata.shape[1]
            if not sparse_pca and numcells > 10000:
                warnings.warn('All genes are being used. It is recommended '
                              'to set `sparse_pca=False` to satisfy memory '
                              'constraints for datasets with more than '
                              '10,000 cells. Setting `sparse_pca` to True.')
                sparse_pca=True

        if not sparse_pca:
            n_genes = min(n_genes, (D.sum(0) > 0).sum())
            
        self.run_args = {
            "max_iter": max_iter,
            "verbose": verbose,
            "projection": projection,
            "stopping_condition": stopping_condition,
            "num_norm_avg": num_norm_avg,
            "k": k,
            "distance": distance,
            "preprocessing": preprocessing,
            "npcs": npcs,
            "n_genes": n_genes,
            "weight_PCs": weight_PCs,
            "proj_kwargs": proj_kwargs,
            "sparse_pca": sparse_pca,
            "weight_mode": weight_mode,
            "seed": seed,
            "components": components,
        }
        self.adata.uns["run_args"] = self.run_args
        
        
        tinit = time.time()
        np.random.seed(seed)

        if verbose:
            print("RUNNING SAM")

        W = np.ones(D.shape[1])
        self.adata.var['weights'] = W

        old = np.zeros(W.size)
        new = W

        i = 0
        err = ((new - old) ** 2).mean() ** 0.5

        if max_iter < 5:
            max_iter = 5

        nnas = num_norm_avg

        while i < max_iter and err > stopping_condition:

            conv = err
            if verbose:
                print("Iteration: " + str(i) + ", Convergence: " + str(conv))

            i += 1
            old = new
            if i == 1:                
                first=True
            else:
                first=False

            W = self.calculate_nnm(batch_key=batch_key,
                n_genes=n_genes, preprocessing=preprocessing, npcs=npcs, num_norm_avg=nnas,
                weight_PCs=weight_PCs, sparse_pca=sparse_pca,weight_mode=weight_mode,seed=seed,components=components,first=first
                )  
            gc.collect()
            new = W
            err = ((new - old) ** 2).mean() ** 0.5
            self.adata.var['weights']=W

        all_gene_names = np.array(list(self.adata.var_names))
        indices = np.argsort(-W)
        ranked_genes = all_gene_names[indices]

        self.adata.uns["ranked_genes"] = ranked_genes

        if projection == "tsne":
            if verbose:
                print("Computing the t-SNE embedding...")
            self.run_tsne(**proj_kwargs)
        elif projection == "umap":
            if verbose:
                print("Computing the UMAP embedding...")
            self.run_umap(seed=seed,**proj_kwargs)
        elif projection == "diff_umap":
            if verbose:
                print("Computing the diffusion UMAP embedding...")
            self.run_diff_umap(**proj_kwargs)

        elapsed = time.time() - tinit
        if verbose:
            print("Elapsed time: " + str(elapsed) + " seconds")

    def calculate_nnm(
        self, adata=None,batch_key=None,g_weighted=None, n_genes=3000, preprocessing='StandardScaler', npcs=150, num_norm_avg=50, weight_PCs=False, sparse_pca=False,
        update_manifold=True,weight_mode='dispersion',seed=0,components=None,first=False
    ):
        if adata is None:
            adata = self.adata

        numcells = adata.shape[0]
        k = adata.uns['run_args'].get("k", 20)
        distance = adata.uns['run_args'].get("distance", "correlation")
                
        D = adata.X
        W = adata.var["weights"].values


        if 'means' not in adata.var.keys() or 'variances' not in adata.var.keys():
            self.calculate_mean_var(adata)
        
        if n_genes is None:
            gkeep = np.arange(W.size)
        else:
            if first:
                mu = np.array(list(adata.var['means']))
                var = np.array(list(adata.var['variances']))
                mu[mu==0]=1
                dispersions = var/mu
                gkeep = np.sort(np.argsort(-dispersions)[:n_genes])
            else:    
                gkeep = np.sort(np.argsort(-W)[:n_genes])

        if g_weighted is None:
            if preprocessing == "Normalizer":
                Ds = D[:, gkeep]
                if sp.issparse(Ds) and not sparse_pca:
                    Ds = Ds.toarray()

                Ds = Normalizer().fit_transform(Ds)

            elif preprocessing == "StandardScaler":
                if not sparse_pca:
                    Ds = D[:, gkeep]
                    if sp.issparse(Ds):
                        Ds = Ds.toarray()

                    v = adata.var['variances'].values[gkeep]
                    m = adata.var['means'].values[gkeep]
                    v[v==0]=1
                    Ds = (Ds - m)/v**0.5

                    Ds[Ds > 10] = 10
                    Ds[Ds < -10] = -10
                else:
                    Ds = D[:, gkeep]
                    v = adata.var['variances'].values[gkeep]
                    v[v==0]=1
                    Ds = Ds.multiply(1/v**0.5).tocsr()

            else:
                Ds = D[:, gkeep].toarray()

            if sp.issparse(Ds):
                D_sub = Ds.multiply(W[gkeep]).tocsr()
            else:
                D_sub = Ds * (W[gkeep])

            if components is None:
                if not sparse_pca:
                    npcs = min(npcs, min((D.shape[0],gkeep.size)))
                    if numcells > 500:
                        g_weighted, pca = ut.weighted_PCA(
                            D_sub,
                            npcs=npcs,
                            do_weight=weight_PCs,
                            solver="auto",seed=seed
                        )
                    else:
                        g_weighted, pca = ut.weighted_PCA(
                            D_sub,
                            npcs=npcs,
                            do_weight=weight_PCs,
                            solver="full",seed=seed
                        )
                    pca_obj = pca
                    components = pca.components_

                else:
                    npcs=min(npcs, min((D.shape[0],gkeep.size)) - 1)
                    v = adata.var['variances'].values[gkeep]
                    v[v==0]=1
                    m = adata.var['means'].values[gkeep]*W[gkeep]
                    no = (m/v**0.5) if preprocessing=='StandardScaler' else D_sub.mean(0).A.flatten()
                    mean_correction = no
                    output = ut._pca_with_sparse(D_sub, npcs,mu = (no)[None,:], seed = seed)
                    components = output['components']
                    g_weighted = output['X_pca']

                    if weight_PCs:
                        ev = output['variance']
                        ev = ev / ev.max()
                        g_weighted = g_weighted * (ev ** 0.5)
            else:
                components = components[:,gkeep]
                v = adata.var['variances'].values[gkeep]
                v[v==0]=1
                m = adata.var['means'].values[gkeep]*W[gkeep]
                ns = (m/v**0.5) if preprocessing=='StandardScaler' else D_sub.mean(0).A.flatten()
                mean_correction = ns

                if sp.issparse(D_sub):
                    g_weighted = D_sub.dot(components.T) - ns.flatten().dot(components.T)
                else:
                    g_weighted = (D_sub-ns).dot(components.T)
                if weight_PCs:
                    ev = g_weighted.var(0) #fix this?
                    ev = ev / ev.max()
                    g_weighted = g_weighted * (ev ** 0.5)
            
            adata.varm["PCs"] = np.zeros(shape=(adata.n_vars, npcs))
            adata.varm["PCs"][gkeep] = components.T
            adata.obsm["X_processed"] = D_sub
            adata.uns["dimred_indices"] = gkeep
            if sparse_pca:
                mc = np.zeros(adata.shape[1])
                mc[gkeep] = mean_correction
                adata.var["mean_correction"] = mc

        if distance == "euclidean":
            pass;#g_weighted = Normalizer().fit_transform(g_weighted)

        if batch_key is not None:
            harmony_out = harmonypy.run_harmony(g_weighted, adata.obs, batch_key, verbose=False)
            g_weighted = harmony_out.Z_corr.T            


        if update_manifold:
            edm = ut.calc_nnm(g_weighted, k, distance)
            adata.obsp['distances'] = edm.tolil()
            adata.obsp['distances'].setdiag(0)
            adata.obsp['distances']=adata.obsp['distances'].tocsr()
            
            EDM = edm.copy()
            EDM.data[:] = 1
            EDM = EDM.tolil(); EDM.setdiag(1); EDM = EDM.tocsr();

            adata.obsp['connectivities'] = EDM

            if distance in ['correlation','cosine']: #keep edge weights and store in nnm if distance is bounded
                edm.data[:] = 1-edm.data
                edm = edm.tolil(); edm.setdiag(1); edm = edm.tocsr();
                edm.data[edm.data<0]=0.001 #if negative correlation, set close to zero but not zero to preserve kNN structure
                adata.obsp['nnm'] = edm
            else:
                adata.obsp['nnm'] = EDM
            W = self.dispersion_ranking_NN(EDM, weight_mode=weight_mode, num_norm_avg=num_norm_avg, adata=adata)
            adata.obsm["X_pca"] = g_weighted
            ge = np.array(list(adata.var_names[gkeep]))
            return W
        else:
            print('Not updating the manifold...')
            PCs = np.zeros(shape=(adata.n_vars, npcs))
            PCs[gkeep] = components.T
            return PCs,g_weighted

    def run_tsne(self, X=None, metric="correlation", **kwargs):
        """Wrapper for sklearn's t-SNE implementation.

        See sklearn for the t-SNE documentation. All arguments are the same
        with the exception that 'metric' is set to 'correlation' by default.
        """
        if X is not None:
            dt = man.TSNE(metric=metric, **kwargs).fit_transform(X)
            return dt

        else:
            distance = self.adata.uns['run_args'].get("distance", "correlation")
            dt = man.TSNE(metric=distance, **kwargs).fit_transform(
                self.adata.obsm["X_pca"]
            )
            tsne2d = dt
            self.adata.obsm["X_tsne"] = tsne2d

    def run_umap(self, X="X_pca", metric=None,seed = None, **kwargs):
        """Wrapper for umap-learn.

        See https://github.com/lmcinnes/umap sklearn for the documentation
        and source code.
        """

        import umap as umap

        if metric is None:
            metric = self.adata.uns['run_args'].get("distance", "correlation")
            
        if seed is None:
            seed = self.adata.uns['run_args'].get('seed',0)

        if type(X) is str:
            if X == "":
                X = self.adata.X
            else:
                X = self.adata.obsm[X]
            # print(X.shape)
            umap_obj = umap.UMAP(metric=metric,random_state=seed, **kwargs)
            umap2d = umap_obj.fit_transform(X)
            self.adata.obsm["X_umap"] = umap2d
            return umap_obj
        else:
            umap_obj = umap.UMAP(metric=metric,random_state=seed, **kwargs)
            dt = umap_obj.fit_transform(X)
            return dt, umap_obj

    def run_diff_umap(
        self, use_rep="X_pca", metric="euclidean", n_comps=15, method="gauss", **kwargs
    ):
        """
        Experimental -- running UMAP on the diffusion components. Requires scanpy.
        """
        import scanpy.api as sc

        k = self.adata.uns['run_args'].get("k", 20)
        distance = self.adata.uns['run_args'].get("distance", "correlation")
        sc.pp.neighbors(
            self.adata, use_rep=use_rep, n_neighbors=k, metric=distance, method=method
        )
        sc.tl.diffmap(self.adata, n_comps=n_comps)
        sc.pp.neighbors(
            self.adata,
            use_rep="X_diffmap",
            n_neighbors=k,
            metric="euclidean",
            method=method,
        )

        if "X_umap" in self.adata.obsm.keys():
            temp = self.adata.obsm["X_umap"].copy()

        sc.tl.umap(self.adata, min_dist=0.1, copy=False)
        temp2 = self.adata.obsm["X_umap"]
        self.adata.obsm["X_umap"] = temp
        self.adata.obsm["X_diff_umap"] = temp2

    def run_diff_map(
        self, use_rep="X_pca", metric="euclidean", n_comps=15, method="gauss", **kwargs
    ):
        import scanpy.api as sc

        k = self.adata.uns['run_args'].get("k", 20)
        distance = self.adata.uns['run_args'].get("distance", "correlation")
        sc.pp.neighbors(
            self.adata, use_rep=use_rep, n_neighbors=k, metric=distance, method=method
        )
        sc.tl.diffmap(self.adata, n_comps=n_comps + 1)
        self.adata.obsm["X_diffmap"] = self.adata.obsm["X_diffmap"][:, 1:]

    def density_clustering(self, X=None, eps=1, metric="euclidean", **kwargs):
        from sklearn.cluster import DBSCAN

        if X is None:
            X = self.adata.obsm["X_umap"]
            save = True
        else:
            save = False

        cl = DBSCAN(eps=eps, metric=metric, **kwargs).fit_predict(X)
        k = self.adata.uns['run_args'].get("k", 20)
        idx0 = np.where(cl != -1)[0]
        idx1 = np.where(cl == -1)[0]
        if idx1.size > 0 and idx0.size > 0:
            xcmap = ut.generate_euclidean_map(X[idx0, :], X[idx1, :])
            knn = np.argsort(xcmap.T, axis=1)[:, :k]
            nnm = np.zeros(xcmap.shape).T
            nnm[
                np.tile(np.arange(knn.shape[0])[:, None], (1, knn.shape[1])).flatten(),
                knn.flatten(),
            ] = 1
            nnmc = np.zeros((nnm.shape[0], cl.max() + 1))
            for i in range(cl.max() + 1):
                nnmc[:, i] = nnm[:, cl[idx0] == i].sum(1)

            cl[idx1] = np.argmax(nnmc, axis=1)

        if save:
            self.adata.obs["dbscan_clusters"] = pd.Categorical(cl)
        else:
            return cl

    def clustering(self, X=None, param=None, method="leiden"):
        """A wrapper for clustering the SAM output using various clustering
        algorithms

        Parameters
        ----------
        X - data, optional, default None
            Data to be passed into the selected clustering algorithm. If None,
            uses the data in the SAM AnnData object. Different clustering
            algorithms accept different types of data. For example, Louvain
            and Leiden accept a scipy.sparse adjacency matrix representing the
            nearest neighbor graph. Dbscan, kmeans, and Hdbscan accept
            coordinates (like UMAP coordinates or PCA coordinates). If None,
            cluster results are saved to the 'obs' attribute in the AnnData
            object. Otherwise, cluster results are returned.

        param : float, optional, default None
            The parameter used for the different clustering algorithms. For
            louvain and leiden, it is the resolution parameter. For dbscan, it
            is the distance parameter. For kmeans, it is the number of clusters.

        method : string, optional, default 'leiden'
            Determines which clustering method is run.
            'leiden' - Leiden clustering with modularity optimization
            'leiden_sig' - Leiden clustering with significance optimization
            'louvain' - Louvain clustering with modularity optimization
            'louvain_sig' - Louvain clustering with significance optimization
            'kmeans' - Kmeans clustering
            'dbscan' - DBSCAN clustering
            'hdbscan' - HDBSCAN clustering
            If X is None, cluster assignments are saved in '.adata.obs' with
            key name equal to method + '_clusters'. Otherwise, they are returned.
        """
        if method == "leiden":
            if param is None:
                param = 1
            cl = self.leiden_clustering(X=X, res=param, method="modularity")
        elif method == "leiden_sig":
            if param is None:
                param = 1
            cl = self.leiden_clustering(X=X, res=param, method="significance")
        elif method == "louvain":
            if param is None:
                param = 1
            cl = self.louvain_clustering(X=X, res=param, method="modularity")
        elif method == "louvain_sig":
            if param is None:
                param = 1
            cl = self.louvain_clustering(X=X, res=param, method="significance")
        elif method == "kmeans":
            if param is None:
                param = 6
            cl = self.kmeans_clustering(param, X=X)
        elif method == "hdbscan":
            if param is None:
                param = 25
            cl = self.hdbknn_clustering(npcs=param)
        elif method == "dbscan":
            if param is None:
                param = 0.5
            cl = self.density_clustering(eps=param)
        else:
            cl = None
        return cl

    def louvain_clustering(self, X=None, res=1, method="modularity"):
        """Runs Louvain clustering using the vtraag implementation. Assumes
        that 'louvain' optional dependency is installed.

        Parameters
        ----------
        res - float, optional, default 1
            The resolution parameter which tunes the number of clusters Louvain
            finds.

        method - str, optional, default 'modularity'
            Can be 'modularity' or 'significance', which are two different
            optimizing funcitons in the Louvain algorithm.

        """

        if X is None:
            X = self.adata.obsp["connectivities"]
            save = True
        else:
            if not sp.isspmatrix_csr(X):
                X = sp.csr_matrix(X)
            save = False

        import igraph as ig
        import louvain

        adjacency = X
        sources, targets = adjacency.nonzero()
        weights = adjacency[sources, targets]
        if isinstance(weights, np.matrix):
            weights = weights.A1
        g = ig.Graph(directed=True)
        g.add_vertices(adjacency.shape[0])
        g.add_edges(list(zip(sources, targets)))
        try:
            g.es["weight"] = weights
        except BaseException:
            pass

        if method == "significance":
            cl = louvain.find_partition(g, louvain.SignificanceVertexPartition)
        else:
            cl = louvain.find_partition(
                g, louvain.RBConfigurationVertexPartition, resolution_parameter=res
            )

        if save:
            if method == "modularity":
                self.adata.obs["louvain_clusters"] = pd.Categorical(
                    np.array(cl.membership)
                )
            elif method == "significance":
                self.adata.obs["louvain_sig_clusters"] = pd.Categorical(
                    np.array(cl.membership)
                )
        else:
            return np.array(cl.membership)

    def kmeans_clustering(self, numc, X=None, npcs=25):
        """Performs k-means clustering.

        Parameters
        ----------
        numc - int
            Number of clusters

        npcs - int, optional, default 25
            Number of principal components to use as inpute for k-means
            clustering.

        """

        from sklearn.cluster import KMeans

        if X is None:
            X = self.adata.obsm['X_pca']

        km = KMeans(n_clusters=numc)
        cl = km.fit_predict(Normalizer().fit_transform(X))

        self.adata.obs["kmeans_clusters"] = pd.Categorical(cl)
        return cl, km

    def leiden_clustering(self, X=None, res=1, method="modularity", seed = 0):

        if X is None:
            X = self.adata.obsp["connectivities"]
            save = True
        else:
            if not sp.isspmatrix_csr(X):
                X = sp.csr_matrix(X)
            save = False

        import igraph as ig
        import leidenalg

        adjacency = X
        sources, targets = adjacency.nonzero()
        weights = adjacency[sources, targets]
        if isinstance(weights, np.matrix):
            weights = weights.A1
        g = ig.Graph(directed=True)
        g.add_vertices(adjacency.shape[0])
        g.add_edges(list(zip(sources, targets)))
        try:
            g.es["weight"] = weights
        except BaseException:
            pass

        if method == "significance":
            cl = leidenalg.find_partition(g, leidenalg.SignificanceVertexPartition,seed=seed)
        else:
            cl = leidenalg.find_partition(
                g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=res,seed=seed
            )

        if save:
            if method == "modularity":
                self.adata.obs["leiden_clusters"] = pd.Categorical(
                    np.array(cl.membership)
                )
            elif method == "significance":
                self.adata.obs["leiden_sig_clusters"] = pd.Categorical(
                    np.array(cl.membership)
                )
        else:
            return np.array(cl.membership)

    def hdbknn_clustering(self, X=None, k=None, npcs=15, **kwargs):
        import hdbscan

        if X is None:
            X = self.adata.obsm['X_pca']
            X = Normalizer().fit_transform(X)
            save = True
        else:
            save = False

        if k is None:
            k = self.adata.uns['run_args'].get("k", 20)

        hdb = hdbscan.HDBSCAN(metric="euclidean", **kwargs)

        cl = hdb.fit_predict(X)

        idx0 = np.where(cl != -1)[0]
        idx1 = np.where(cl == -1)[0]
        if idx1.size > 0 and idx0.size > 0:
            xcmap = ut.generate_euclidean_map(X[idx0, :], X[idx1, :])
            knn = np.argsort(xcmap.T, axis=1)[:, :k]
            nnm = np.zeros(xcmap.shape).T
            nnm[
                np.tile(np.arange(knn.shape[0])[:, None], (1, knn.shape[1])).flatten(),
                knn.flatten(),
            ] = 1
            nnmc = np.zeros((nnm.shape[0], cl.max() + 1))
            for i in range(cl.max() + 1):
                nnmc[:, i] = nnm[:, cl[idx0] == i].sum(1)

            cl[idx1] = np.argmax(nnmc, axis=1)

        if save:
            self.adata.obs["hdbscan_clusters"] = pd.Categorical(cl)
        else:
            return cl

    def identify_marker_genes_rf(self, labels=None, clusters=None, n_genes=4000):
        """
        Ranks marker genes for each cluster using a random forest
        classification approach.

        Parameters
        ----------

        labels - numpy.array or str, optional, default None
            Cluster labels to use for marker gene identification.
            Can also be a string corresponding to any of the keys
            in adata.obs.

        clusters - int/string or array-like, default None
            A cluster ID (int or string depending on the labels used)
            or vector corresponding to the specific cluster ID(s) for
            which marker genes will be calculated. If None, marker genes
            will be computed for all clusters, and the result will be written
            to adata.uns.

        n_genes - int, optional, default 4000
            By default, trains the classifier on the top 4000 SAM-weighted
            genes.

        Returns
        -------
        (dictionary of markers for each cluster,
        dictionary of marker scores for each cluster)
        """

        if labels is None:
            try:
                keys = np.array(list(self.adata.obs_keys()))
                lbls = self.get_labels(ut.search_string(keys, "_clusters")[0][0])
            except KeyError:
                print(
                    "Please generate cluster labels first or set the "
                    "'labels' keyword argument."
                )
                return
        elif isinstance(labels, str):
            lbls = self.get_labels(labels)
        else:
            lbls = labels

        from sklearn.ensemble import RandomForestClassifier

        markers = {}
        markers_scores = {}
        if clusters == None:
            lblsu = np.unique(lbls)
        else:
            lblsu = np.unique(clusters)

        indices = np.argsort(-self.adata.var["weights"].values)
        X = self.adata.layers["X_disp"][:, indices[:n_genes]].toarray()
        for K in range(lblsu.size):
            # print(K)
            y = np.zeros(lbls.size)

            y[lbls == lblsu[K]] = 1

            clf = RandomForestClassifier(
                n_estimators=100, max_depth=None, random_state=0
            )

            clf.fit(X, y)

            idx = np.argsort(-clf.feature_importances_)

            markers[lblsu[K]] = self.adata.uns["ranked_genes"][idx]
            markers_scores[lblsu[K]] = clf.feature_importances_[idx]

        if clusters is None:
            if isinstance(labels, str):
                self.adata.uns["rf_" + labels] = markers
            else:
                self.adata.uns["rf"] = markers

        return markers, markers_scores

    def identify_marker_genes_sw(self, labels=None, clusters=None, inplace=True):
        """
        Ranks marker genes for each cluster using partial sums of spatial
        dispersions.

        Parameters
        ----------

        labels - numpy.array or str, optional, default None
            Cluster labels to use for marker gene identification.
            Can also be a string corresponding to any of the keys
            in adata.obs.

        clusters - int/string or array-like, default None
            A cluster ID (int or string depending on the labels used)
            or vector corresponding to the specific cluster ID(s) for
            which marker genes will be calculated. If None, marker genes
            will be computed for all clusters, and the result will be written
            to adata.uns.

        inplace - bool, default True
            If True, returns nothing and places marker scores in `var`

        """

        if labels is None:
            try:
                keys = np.array(list(self.adata.obs_keys()))
                lbls = self.get_labels(ut.search_string(keys, "_clusters")[0][0])
            except KeyError:
                print(
                    "Please generate cluster labels first or set the "
                    "'labels' keyword argument."
                )
                return
        elif isinstance(labels, str):
            lbls = self.get_labels(labels)
        else:
            lbls = labels

        markers_scores = []
        if clusters == None:
            lblsu = np.unique(lbls)
        else:
            lblsu = np.unique(clusters)

        if "X_knn_avg" not in list(self.adata.layers.keys()):
            print("Performing kNN-averaging...")
            self.dispersion_ranking_NN(save_avgs=True)
        l = self.adata.layers["X_knn_avg"]
        m = l.mean(0).A.flatten()
        cells = np.array(list(self.adata.obs_names))
        for K in range(lblsu.size):
            selected = np.where(np.in1d(cells, self.get_cells(lblsu[K], labels)))[0]
            ms = l[selected, :].mean(0).A.flatten()
            lsub = l[selected, :]
            lsub.data[:] = lsub.data ** 2
            ms2 = lsub.mean(0).A.flatten()
            v = ms2 - 2 * ms * m + m ** 2
            wmu = np.zeros(v.size)
            wmu[m > 0] = v[m > 0] / m[m > 0]
            markers_scores.append(wmu)
        A = pd.DataFrame(
            data=np.vstack(markers_scores), index=lblsu, columns=self.adata.var_names
        ).T
        if inplace:
            A.columns = labels + ";;" + A.columns.astype('str').astype('object')
            for Ac in A.columns:
                self.adata.var[Ac]=A[Ac]
        else:
            return A

    def identify_marker_genes_ratio(self, labels=None):
        """
        Ranks marker genes for each cluster using a SAM-weighted
        expression-ratio approach (works quite well).

        Parameters
        ----------

        labels - numpy.array or str, optional, default None
            Cluster labels to use for marker gene identification. If None,
            assumes that one of SAM's clustering algorithms has been run. Can
            be a string (i.e. 'louvain_clusters', 'kmeans_clusters', etc) to
            specify specific cluster labels in adata.obs.

        """
        if labels is None:
            try:
                keys = np.array(list(self.adata.obs_keys()))
                lbls = self.get_labels(ut.search_string(keys, "_clusters")[0][0])
            except KeyError:
                print(
                    "Please generate cluster labels first or set the "
                    "'labels' keyword argument."
                )
                return
        elif isinstance(labels, str):
            lbls = self.get_labels(labels)
        else:
            lbls = labels

        all_gene_names = np.array(list(self.adata.var_names))

        markers = {}

        s = np.array(self.adata.layers["X_disp"].sum(0)).flatten()
        lblsu = np.unique(lbls)
        for i in lblsu:
            d = np.array(self.adata.layers["X_disp"][lbls == i, :].sum(0)).flatten()
            rat = np.zeros(d.size)
            rat[s > 0] = (
                d[s > 0] ** 2 / s[s > 0] * self.adata.var["weights"].values[s > 0]
            )
            x = np.argsort(-rat)
            markers[i] = all_gene_names[x[:]]

        self.adata.uns["marker_genes_ratio"] = markers

        return markers

    def gui(self):
        if 'SamGui' in self.__dict__.keys():
            return self.SamGui.SamPlot
        else:
            from .gui import SAMGUI
            self.SamGui = SAMGUI(self)
            return self.SamGui.SamPlot
            
    def save(self,fn):
        import dill
        if len(fn.split('.pkl')) == 1:
            fn = fn + '.pkl'
        self.path_to_file = fn
        d = {}
        for k in self.__dict__.keys():
            d[k] = self.__dict__[k]
        if 'SamGui' in d.keys():
            del d['SamGui']
        with open(fn,'wb') as f:
            dill.dump(d,f)
            if 'SamGui' in self.__dict__.keys():
                dill.dump(self.SamGui,f)
            
    def load(self,fn):
        import dill
        if len(fn.split('.pkl')) == 1:
            fn = fn + '.pkl'    
        with open(fn,'rb') as f:
            self.__dict__ = dill.load(f)
            try:
                self.__dict__['SamGui'] = dill.load(f)
            except:
                pass;

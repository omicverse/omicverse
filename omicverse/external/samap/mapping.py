from scipy.stats import binned_statistic
import hnswlib
import sklearn.utils.sparsefuncs as sf
import typing
from numba import njit, prange
import os
from os import path
import gc
from .samalg import SAM
import time
from sklearn.preprocessing import StandardScaler

from . import q, ut, pd, sp, np, warnings, sc
from .analysis import _compute_csim
from .utils import prepend_var_prefix, to_vn, substr, sparse_knn, df_to_dict

from numba.core.errors import NumbaPerformanceWarning, NumbaWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
warnings.filterwarnings("ignore", category=NumbaWarning)


class SAMAP(object):
    def __init__(
        self,
        sams: dict,
        f_maps: typing.Optional[str] = "maps/",
        names: typing.Optional[dict] = None,
        keys: typing.Optional[dict] = None,
        resolutions: typing.Optional[dict] = None,
        gnnm: typing.Optional[tuple] = None,
        save_processed: typing.Optional[bool] = True,
        eval_thr: typing.Optional[float] = 1e-6
    ):

        """Initializes and preprocess data structures for SAMap algorithm.

        Parameters
        ----------
        sams : dict of string OR SAM
            Dictionary of (indexed by species IDs):
            The path to an unprocessed '.h5ad' `AnnData` object for organisms.
            OR
            A processed and already-run SAM object.

        f_maps : string, optional, default 'maps/'
            Path to the `maps` directory output by `map_genes.sh`.
            By default assumes it is in the local directory.

        names : dict of list of 2D tuples or Nx2 numpy.ndarray, optional, default None
            If BLAST was run on a transcriptome with Fasta headers that do not match
            the gene symbols used in the dataset, you can pass a list of tuples mapping
            the Fasta header name to the Dataset gene symbol:
            (Fasta header name , Dataset gene symbol). Transcripts with the same gene
            symbol will be collapsed into a single node in the gene homology graph.
            By default, the Fasta header IDs are assumed to be equivalent to the
            gene symbols used in the dataset.

            The above mapping should be contained in a dicitonary keyed by the corresponding species.
            For example, if we have `hu` and `mo` species and the `hu` BLAST results need to be translated,
            then `names = {'hu' : mapping}, where `mapping = [(Fasta header 1, Gene symbol 1), ... , (Fasta header n, Gene symbol n)]`.
        
        keys : dict, optional, default None
            Dictionary of obs keys indexed by species to use for determining maximum
            neighborhood size of each cell. 
        
        resolutions : dict, optional, default None
            Dictionary of leiden clustering resolutions indexed by species. This parameter is ignored if
            `keys` is set.

        gnnm : tuple(scipy.sparse.csr_matrix,numpy array, dict[numpy array])
            If the homology graph was already computed, you can pass it here in the form of a tuple:
            (sparse adjacency matrix, numpy array of genes, dictionary of species-specific genes).
            Note that all genes must be prefixed with their species IDs, e.g. `hu_SOX2` instead of `SOX2`.
            
            This is the tuple returned by `_calculate_blast_graph(...)` or `_coarsen_eggnog_graph(...)`.

        save_processed : bool, optional, default False
            If True saves the processed SAM objects corresponding to each species to an `.h5ad` file.
            This argument is unused if preloaded SAM objects are passed in to SAMAP.

        eval_thr : float, optional, default 1e-6
            E-value threshold above which BLAST results will be filtered out.
        """

        for key,data in zip(sams.keys(),sams.values()):
            if not (isinstance(data, str) or isinstance(data, SAM)):
                raise TypeError(f"Input data {key} must be either a path or a SAM object.")
        
        
        ids = list(sams.keys())
        
        if keys is None:
            keys = {}
            for sid in ids:
                keys[sid] = 'leiden_clusters'
        
        if resolutions is None:
            resolutions = {}
            for sid in ids:
                resolutions[sid] = 3


        for sid in ids:
            data = sams[sid]
            key = keys[sid]
            res = resolutions[sid]
            
            if isinstance(data, str):
                print("Processing data {} from:\n{}".format(sid,data))
                sam = SAM()
                sam.load_data(data)
                sam.preprocess_data(
                    sum_norm="cell_median",
                    norm="log",
                    thresh_low=0.0,
                    thresh_high=0.96,
                    min_expression=1,
                )
                sam.run(
                    preprocessing="StandardScaler",
                    npcs=100,
                    weight_PCs=False,
                    k=20,
                    n_genes=3000,
                    weight_mode='rms'
                )
            else:
                sam = data                
            
            if key == "leiden_clusters":
                sam.leiden_clustering(res=res)
                
            if "PCs_SAMap" not in sam.adata.varm.keys():
                prepare_SAMap_loadings(sam)  

            if save_processed and isinstance(data,str):
                sam.save_anndata(data.split('.h5ad')[0]+'_pr.h5ad')

            sams[sid] = sam             

        if gnnm is None:
            gnnm, gns, gns_dict = _calculate_blast_graph(
                ids, f_maps=f_maps, reciprocate=True, eval_thr=eval_thr
            )            
            if names is not None:
                gnnm, gns_dict, gns  = _coarsen_blast_graph(
                    gnnm, gns, names
                )

            gnnm = _filter_gnnm(gnnm, thr=0.25)
        else:
            gnnm, gns, gns_dict = gnnm

        gns_list=[]
        ges_list=[]
        for sid in ids:
            prepend_var_prefix(sams[sid], sid)
            ge = q(sams[sid].adata.var_names)
            gn = gns_dict[sid]
            gns_list.append(gn[np.in1d(gn, ge)])
            ges_list.append(ge)
        
        f = np.in1d(gns, np.concatenate(gns_list))
        gns = gns[f]
        gnnm = gnnm[f][:, f]
        A = pd.DataFrame(data=np.arange(gns.size)[None, :], columns=gns)
        ges = np.concatenate(ges_list)
        ges = ges[np.in1d(ges, gns)]
        ix = A[ges].values.flatten()
        gnnm = gnnm[ix][:, ix]
        gns = ges
        
        gns_dict = {}
        for i,sid in enumerate(ids):
            gns_dict[sid] = ges[np.in1d(ges,gns_list[i])]
        
            print(
                "{} `{}` gene symbols match between the datasets and the BLAST graph.".format(
                    gns_dict[sid].size, sid
                )
            )

        for sid in sams:
            if not sp.sparse.issparse(sams[sid].adata.X):
                sams[sid].adata.X = sp.sparse.csr_matrix(sams[sid].adata.X)

        smap = _Samap_Iter(sams, gnnm, gns_dict, keys=keys)
        self.sams = sams
        self.gnnm = gnnm
        self.gns_dict = gns_dict
        self.gns = gns
        self.ids = ids
        self.smap = smap

    def run(
        self,
        NUMITERS: typing.Optional[int] = 3,
        NHS: typing.Optional[dict] = None,
        crossK: typing.Optional[int] = 20,
        N_GENE_CHUNKS: typing.Optional[int] = 1,
        umap: typing.Optional[bool] = True,
        ncpus: typing.Optional[float] = os.cpu_count(),
        hom_edge_thr: typing.Optional[float] = 0,
        hom_edge_mode: typing.Optional[str] = "pearson",
        scale_edges_by_corr: typing.Optional[bool] = True,
        neigh_from_keys: typing.Optional[dict] = None,
        pairwise: typing.Optional[bool] = True
    ):
        """Runs the SAMap algorithm.

        Parameters
        ----------
        NUMITERS : int, optional, default 3
            Runs SAMap for `NUMITERS` iterations.        
            
        NHS : dict, optional, default None
            Dictionary of maximum neighborhood sizes indexed by species.
            Cells up to `NHS[species_id]` hops away from a particular cell in `species_id`
            will be included in its neighborhood.
            
            If None, defaults to 3 for all species.

        crossK : int, optional, default 20
            The number of cross-species edges to identify per cell.

        N_GENE_CHUNKS : int, optional, default 1
            When updating the edge weights in the BLAST homology graph, the operation
            will be split up into `N_GENE_CHUNKS` chunks. For large datasets
            (>50,000 cells), use more chunks (e.g. 4) to avoid running out of
            memory.
            
        umap : bool, optional, default True
            If True, performs UMAP on the combined manifold to generate a 2D visualization.
            If False, skips this step. 
            
        ncpus : int, optional, default `os.cpu_count()`
            The number of CPUs to use when computing gene-gene correlations.
            Defaults to using all available CPUs.
            
        hom_edge_thr : float, optional, default 0
            Edges with weight below `hom_edge_thr` in the homology graph will be set to zero.
            
        hom_edge_mode: str, optional, default "pearson"
            If "pearson", edge weights in the homology graph will be calculated using Pearson
            correlation. If "mutual_info", edge weights will be calculated using normalized
            mutual information. The latter requires package `fast-histogram` to be installed.
            
        scale_edges_by_corr: bool, optional, default True
            If True, rescale cell-cell cross-species edges by their expression similarities
            (correlations).
            
        neigh_from_keys : dict[bool], optional, default None
            Dictionary of booleans indexed by species IDs. If True, species neighborhoods
            are calculated directly from the chosen clustering (the `keys` parameter in `SAMAP(...)`).
            Cells within the same cluster belong to the same neighborhood.
            
        pairwise: bool, optional, default True
            If True, compute mutual nearest neighborhoods independently between each pair of species.
            If False, compute mutual nearest neighborhoods between each species and all other species.
            This parameter is ignored if there are only two species to be mapped.

           `pairwise=True` would prevent outgroup species from being unmapped.
            For example, if `pairwise=False`, when mapping three species like human, mouse, and zebrafish, human and mice
            will preferentially map to each other, leaving zebrafish unmapped.

            Set `pairwise=False` when you'd like to be able to tell which cell types are more or less similar 
            between different species. Set `pairwise=True` when you'd like to remove the effects of evolutionary
            distance. This is particularly useful when using multiple reference species to annotate an unlabeled
            dataset.
            
        Returns
        -------
        samap - Species-merged SAM object
        """
        self.pairwise = pairwise

        ids = self.ids
        sams = self.sams
        gnnm = self.gnnm
        gns_dict = self.gns_dict
        gns = self.gns
        smap = self.smap
        
        if NHS is None:
            NHS={}
            for sid in ids:
                NHS[sid] = 3
        if neigh_from_keys is None:
            neigh_from_keys={}
            for sid in ids:
                neigh_from_keys[sid] = False

        start_time = time.time()

        smap.run(
            NUMITERS=NUMITERS,
            NHS=NHS,
            K=crossK,
            NCLUSTERS=N_GENE_CHUNKS,
            ncpus=ncpus,
            THR=hom_edge_thr,
            corr_mode=hom_edge_mode,
            scale_edges_by_corr = scale_edges_by_corr,
            neigh_from_keys=neigh_from_keys,
            pairwise=pairwise
        )
        samap = smap.final_sam
        self.samap = samap
        self.ITER_DATA = smap.ITER_DATA

        if umap:
            print("Running UMAP on the stitched manifolds.")
            sc.tl.umap(self.samap.adata,min_dist=0.1,init_pos='random',maxiter = 500 if self.samap.adata.shape[0] <= 10000 else 200)
        
        
        ix = pd.Series(data = np.arange(samap.adata.shape[1]),index = samap.adata.var_names)[gns].values
        rixer = pd.Series(index =np.arange(gns.size), data = ix)
        
        try:
            hom_graph = smap.GNNMS_corr[-1]
            x,y = hom_graph.nonzero()
            d = hom_graph.data
            hom_graph = sp.sparse.coo_matrix((d,(rixer[x].values,rixer[y].values)),shape=(samap.adata.shape[1],)*2).tocsr()                    
            samap.adata.varp["homology_graph_reweighted"] = hom_graph
            self.gnnm_refined = hom_graph            
        except:
            pass
        
        x,y = gnnm.nonzero()
        d = gnnm.data
        gnnm = sp.sparse.coo_matrix((d,(rixer[x].values,rixer[y].values)),shape=(samap.adata.shape[1],)*2).tocsr()                            
        samap.adata.varp["homology_graph"] = gnnm
        samap.adata.uns["homology_gene_names_dict"] = gns_dict
        
        
        self.gnnm = gnnm
        self.gns = q(samap.adata.var_names)
        
        gns_dict = {}
        for sid in ids:
            gns_dict[sid] = self.gns[np.in1d(self.gns,q(self.sams[sid].adata.var_names))]
        self.gns_dict = gns_dict
        
        if umap:
            for sid in ids:
                sams[sid].adata.obsm['X_umap_samap'] = self.samap.adata[sams[sid].adata.obs_names].obsm['X_umap']     
        
        self.run_time = time.time() - start_time
        print("Elapsed time: {} minutes.".format(self.run_time / 60))
        return samap

    def run_umap(self):
        print("Running UMAP on the stitched manifolds.")
        ids = self.ids
        sams = self.sams
        sc.tl.umap(self.samap.adata,min_dist=0.1,init_pos='random', maxiter = 500 if self.samap.adata.shape[0] <= 10000 else 200)
        for sid in ids:
            sams[sid].adata.obsm['X_umap_samap'] = self.samap.adata[sams[sid].adata.obs_names].obsm['X_umap']               

    def plot_expression_overlap(self,gs,axes=None,#'#000098', COLOR2='#ffb900'
                                COLOR0='gray', COLORS=None, COLORC='#00ceb5',
                                s0 = 1, ss=None, sc = 10,
                                thr = 0.1,**kwargs):
        """Displays the expression overlap of two genes on the combined manifold.

        Parameters
        ----------
        gs : dict
            Dictionary of genes to display, keyed by species IDs. 
            For example, human ('hu') and mouse ('ms') genes: 
            gs = {'hu':'TOP2A','ms':'Top2a'}
                    
        axes : matplotlib.pyplot.Axes, optional, default None
            Displays the scatter plot on the provided axes if specified.
            Otherwise creates a new figure.
            
        COLOR0 : str, optional, default 'gray'
            The color for cells that do not express `g1` or `g2`.
        
        COLORS : dict, optional, default None
            Dictionary of colors (hex codes) for cells expressing the
            corresponding genes for each species. This dictionary is
            keyed by species IDs. If not set, colors are chosen randomly.
        
        COLORC : str, optional, default '#00ceb5'
            The color for cells that overlap in
            expression of the two genes.
        
        s0 : int, optional, default 1
            Marker size corresponding to `COLOR0`.
            
        ss : dict, optional, default None
            Dictionary of marker sizes corresponding to the colors in `COLORS`.
            If not set, marker sizes default to 3.
                        
        sc : int, optional, default 10
            Marker size corresponding to `COLORC`.
        
        thr : float, optional, default 0.1
            Threshold below which imputed expressions across species are zero'd out. 
        
        Keyword Arguments (**kwargs)
        ----------------------------
        Most arguments accepted by matplotlib.pyplot.scatter are available.
        

        Returns
        -------
        ax - matplotlib.pyplot.Axes
        """   


        if len(list(gs.keys()))<len(list(self.sams.keys())):
            samap = SAM(counts = self.samap.adata[np.in1d(self.samap.adata.obs['species'],list(gs.keys()))])
        else:
            samap=self.samap
                                
        if ss is None:
            ss={}
            for sid in self.ids:
                ss[sid] = 3
        
        if COLORS is None:
            COLORS={}
            for sid in self.ids:
                s = ''
                for i in range(6):
                    s+=hex(np.random.randint(16))[-1].upper()
                s='#'+s                
                COLORS[sid] = s
                
        def hex_to_rgb(value):
            value = value.lstrip('#')
            lv = len(value)
            lv = list(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
            lv = [x/255 for x in lv]
            return lv
        
        

        nnm = samap.adata.obsp['connectivities']
        su = nnm.sum(1).A.flatten()[:,None]
        su[su==0]=1

        nnm = nnm.multiply(1/su).tocsr()
        AS={}
        for sid in gs.keys():
            g = gs[sid]
            try:
                AS[sid] = self.sams[sid].adata[:,g].X.A.flatten()
            except KeyError:
                try:
                    AS[sid] = self.sams[sid].adata[:,sid+'_'+g].X.A.flatten()
                except KeyError:
                    raise KeyError(f'Gene not found in species {sid}')
            
        davgs={}
        for sid in gs.keys():
            d = np.zeros(samap.adata.shape[0])
            d[samap.adata.obs['species']==sid] = AS[sid]
            davg = nnm.dot(d).flatten()
            davg[davg<thr]=0
            davgs[sid] = davg
        davg = np.vstack(list(davgs.values())).min(0)
        ma = np.vstack(list(davgs.values())).max()
        for sid in gs.keys():
            if davgs[sid].max()>0:
                davgs[sid] = davgs[sid]/davgs[sid].max()
        if davg.max()>0:
            davg = davg/davg.max()
        
        cs={}
        for sid in gs.keys():
            c = hex_to_rgb(COLORS[sid])+[0.0]
            cs[sid] = np.vstack([c]*davg.size)
            cs[sid][:,-1] = davgs[sid]
        cc = hex_to_rgb(COLORC)+[0.0]
        cc = np.vstack([cc]*davg.size)
        cc[:,-1] = davg

        ax = samap.scatter(projection = 'X_umap', colorspec = COLOR0, axes=axes, s = s0)        
        
        for sid in gs.keys():            
            samap.scatter(projection = 'X_umap', c = cs[sid], axes = ax, s = ss[sid],colorbar=False,**kwargs)
        
        samap.scatter(projection = 'X_umap', c = cc, axes = ax, s = sc,colorbar=False,**kwargs)
        
        return ax    
    
    def query_gene_pairs(self,gene):
        """ Get BLAST and correlation scores of all genes connected
        to the query gene.

        Preferrably, genes are prepended with their species IDs.
        For example, "hu_SOX2" instead of "SOX2".
        
        Returns: Dictionary with "blast" and "correlation" keys with
        the BLAST and correlation scores respectively for the queried
        gene.
        """ 

        ids = self.ids
        qgene = None
        if (gene in self.gns):
            qgene = gene
        else:
            for sid in ids:
                if sid+'_'+gene in self.gns:
                    qgene = sid+'_'+gene
                    break
        if qgene is None:
            raise ValueError(f"Query gene {gene} not found in dataset.")

        a = self.gnnm[self.gns==qgene]
        b = self.gnnm_refined[self.gns==qgene]

        i1 = self.gns[a.nonzero()[1]]
        i2 = self.gns[b.nonzero()[1]]
        d1 = a.data
        d2 = b.data
        a = pd.Series(index=i1,data=d1)
        b = pd.Series(index=i2,data=d2)
        return {"blast":a,"correlation":b}    

    def query_gene_pair(self,gene1,gene2):
        """ Get BLAST and correlation score for a pair of genes.
        
        Preferrably, genes are prepended with their species IDs.
        For example, "hu_SOX2" instead of "SOX2".
        
        Returns: Dictionary with "blast" and "correlation" keys with
        the BLAST and correlation scores respectively for the queried
        gene pair.
        """
        ids = self.ids
        qgene1 = None
        if (gene1 in self.gns):
            qgene1 = gene1
        else:
            for sid in ids:
                if sid+'_'+gene1 in self.gns:
                    qgene1 = sid+'_'+gene1
                    break
        if qgene1 is None:
            raise ValueError(f"Query gene {gene1} not found in dataset.")

        qgene2 = None
        if (gene2 in self.gns):
            qgene2 = gene2
        else:
            for sid in ids:
                if sid+'_'+gene2 in self.gns:
                    qgene2 = sid+'_'+gene2
                    break
        if qgene2 is None:
            raise ValueError(f"Query gene {gene2} not found in dataset.")

        a = self.gnnm[self.gns==qgene1].A.flatten()[self.gns==qgene2][0]
        b = self.gnnm_refined[self.gns==qgene1].A.flatten()[self.gns==qgene2][0]
        return {"blast":a,"correlation":b}  

    def scatter(self,axes=None,COLORS=None,ss=None,**kwargs):  
        
        if ss is None:
            ss={}
            for sid in self.ids:
                ss[sid] = 3
        
        if COLORS is None:
            COLORS={}
            for sid in self.ids:
                s = ''
                for i in range(6):
                    s+=hex(np.random.randint(16))[-1].upper()
                s='#'+s                
                COLORS[sid] = s
                
        for sid in self.ids:            
            axes = self.sams[sid].scatter(projection = 'X_umap_samap', colorspec = COLORS[sid], axes = axes, s = ss[sid],colorbar=False,**kwargs)
        
        return axes    
        
    def gui(self):
        """Launches a SAMGUI instance containing the two SAM objects."""
        if 'SamapGui' not in self.__dict__:
            try:
                from .samalg.gui import SAMGUI
            except ImportError:
                raise ImportError('Please install SAMGUI dependencies. See the README in the SAM github repository.')

            sg = SAMGUI(sam = list(self.sams.values()), title = list(self.ids),default_proj='X_umap_samap')
            self.SamapGui = sg
            return sg.SamPlot
        else:
            return self.SamapGui.SamPlot
        
    def refine_homology_graph(self, THR=0, NCLUSTERS=1, ncpus = os.cpu_count(), corr_mode='pearson', wscale = False):
        gnnm = self.smap.refine_homology_graph(NCLUSTERS=NCLUSTERS, ncpus=ncpus, THR=THR, corr_mode=corr_mode, wscale=wscale)
        samap = self.smap.samap
        gns_dict = self.smap.gns_dict
        gns = []
        for sid in q(samap.adata.obs['species'])[np.sort(np.unique(samap.adata.obs['species'],return_index=True)[1])]:
            gns.extend(gns_dict[sid])
        gns=q(gns)
        ix = pd.Series(data = np.arange(samap.adata.shape[1]),index = samap.adata.var_names)[gns].values
        rixer = pd.Series(index =np.arange(gns.size), data = ix)         
        x,y = gnnm.nonzero()
        d = gnnm.data
        gnnm = sp.sparse.coo_matrix((d,(rixer[x].values,rixer[y].values)),shape=(samap.adata.shape[1],)*2).tocsr()                           
        return gnnm
        
class _Samap_Iter(object):
    def __init__(
        self, sams, gnnm, gns_dict, keys=None
    ):
        self.sams = sams
        self.gnnm = gnnm
        self.gnnmu = gnnm
        self.gns_dict = gns_dict
        
        if keys is None:
            keys = {}
            for sid in sams.keys():
                keys[sid] = 'leiden_clusters'
        
        self.keys = keys
        
        self.GNNMS_corr = []
        self.GNNMS_pruned = []
        self.GNNMS_nnm = []
        
        self.ITER_DATA = [
            self.GNNMS_nnm,
            self.GNNMS_corr,
            self.GNNMS_pruned,
        ]
        self.iter = 0

    def refine_homology_graph(self, NCLUSTERS=1, ncpus=os.cpu_count(), THR=0, corr_mode='pearson', wscale=False):
        if corr_mode=='mutual_info':
            try:
                from fast_histogram import histogram2d
            except:
                raise ImportError("Package `fast_histogram` must be installed for `corr_mode='mutual_info'`.");
        sams = self.sams
        gnnm = self.gnnm
        gns_dict = self.gns_dict
        gnnmu = self.gnnmu
        keys = self.keys
        sam4 = self.samap
                
        gnnmu = _refine_corr(
            sams,
            sam4,
            gnnm,
            gns_dict,
            THR=THR,
            use_seq=False,
            T1=0,
            NCLUSTERS=NCLUSTERS,
            ncpus=ncpus,
            corr_mode=corr_mode,
            wscale=wscale
        )
        return gnnmu

    def run(self, NUMITERS=3, NHS=None, K=20, corr_mode='pearson', NCLUSTERS=1,
                  scale_edges_by_corr=True, THR=0, neigh_from_keys=None, pairwise=True,
                  ncpus=os.cpu_count()):
        sams = self.sams
        gnnm = self.gnnm
        gns_dict = self.gns_dict
        gnnmu = self.gnnmu
        keys = self.keys
        
        if NHS is None:
            NHS={}
            for sid in sams.keys():
                NHS[sid] = 2
        if neigh_from_keys is None:
            neigh_from_keys={}
            for sid in sams:
                neigh_from_keys[sid] = False        
        gns = np.concatenate(list(gns_dict.values()))        

        if self.iter > 0:
            sam4 = self.samap

        for i in range(NUMITERS):
            if self.iter > 0 and i == 0:
                print("Calculating gene-gene correlations in the homology graph...")
                gnnmu = self.refine_homology_graph(ncpus = ncpus, NCLUSTERS = NCLUSTERS, THR=THR, corr_mode=corr_mode)

                self.GNNMS_corr.append(gnnmu)
                self.gnnmu = gnnmu

            gnnm2 = _get_pairs(sams, gnnmu, gns_dict, NOPs1=0, NOPs2=0)
            self.GNNMS_pruned.append(gnnm2)

            sam4 = _mapper(
                sams,
                gnnm2,
                gns,
                umap=False,
                K=K,
                NHS=NHS,
                coarsen=True,
                keys=keys,
                scale_edges_by_corr=scale_edges_by_corr,
                neigh_from_keys=neigh_from_keys,
                pairwise=pairwise
            )
            sam4.adata.uns['mapping_K'] = K
            self.samap = sam4
            self.GNNMS_nnm.append(sam4.adata.obsp["connectivities"])

            print("Iteration " + str(i+1) + " complete.")
            print("Alignment scores:")
            print(_avg_as(sam4))

            self.iter += 1
            if i < NUMITERS - 1:
                print("Calculating gene-gene correlations in the homology graph...")
                self.samap = sam4
                gnnmu = self.refine_homology_graph(ncpus = ncpus,  NCLUSTERS = NCLUSTERS,  THR=THR, corr_mode=corr_mode)

                self.GNNMS_corr.append(gnnmu)
                self.gnnmu = gnnmu

            gc.collect()

        self.final_sam = sam4


def _avg_as(s):
    x = q(s.adata.obs['species'])
    xu = np.unique(x)
    a = np.zeros((xu.size,xu.size))
    for i in range(xu.size):
        for j in range(xu.size):
            if i!=j:
                a[i,j] = s.adata.obsp['connectivities'][x==xu[i],:][:,x==xu[j]].sum(1).A.flatten().mean() / s.adata.uns['mapping_K']
    return pd.DataFrame(data=a,index=xu,columns=xu)
        
@njit(parallel=True)        
def _replace(X,xi,yi):
    data = np.zeros(xi.size)
    for i in prange(xi.size):
        x=X[xi[i]]
        y=X[yi[i]]
        data[i] = ((x-x.mean())*(y-y.mean()) / x.std() / y.std()).sum() / x.size
    return data
    
    
def _generate_coclustering_matrix(cl):
    cl = ut.convert_annotations(np.array(list(cl)))
    clu,cluc=np.unique(cl,return_counts=True)    
    v = np.zeros((cl.size,clu.size))
    v[np.arange(v.shape[0]),cl]=1
    v = sp.sparse.csr_matrix(v)
    return v

    
def _mapper(
    sams,
    gnnm=None,
    gn=None,
    NHS=None,
    umap=False,
    mdata=None,
    k=None,
    K=20,
    chunksize=20000,
    coarsen=True,
    keys=None,
    scale_edges_by_corr=False,
    neigh_from_keys=None,
    pairwise=True,
    **kwargs
):
    if NHS is None:
        NHS={}
        for sid in sams.keys():
            NHS[sid] = 3         
    
    if neigh_from_keys is None:
        neigh_from_keys={}
        for sid in sams.keys():
            neigh_from_keys[sid] = False    
    
    if mdata is None:
        mdata = _mapping_window(sams, gnnm, gn, K=K, pairwise=pairwise)

    k1 = K

    if keys is None:
        keys = {}
        for sid in sams.keys():
            keys[sid] = 'leiden_clusters'

    nnms_in={}
    nnms_in0={}
    flag=False
    species_indexer=[]
    for sid in sams.keys():
        print(f"Expanding neighbourhoods of species {sid}...")
        cl = sams[sid].get_labels(keys[sid])
        _, ix, cluc = np.unique(cl, return_counts=True, return_inverse=True)
        K = cluc[ix]
        nnms_in0[sid] = sams[sid].adata.obsp["connectivities"].copy()
        species_indexer.append(np.arange(sams[sid].adata.shape[0]))
        if not neigh_from_keys[sid]:
            nnm_in = _smart_expand(nnms_in0[sid], K, NH=NHS[sid])
            nnm_in.data[:] = 1
            nnms_in[sid]=nnm_in
        else:
            nnms_in[sid]=_generate_coclustering_matrix(cl)
            flag=True
    
    for i in range(1,len(species_indexer)):
        species_indexer[i] += species_indexer[i-1].max()+1
    
    if not flag:
        nnm_internal = sp.sparse.block_diag(list(nnms_in.values())).tocsr()
    nnm_internal0 = sp.sparse.block_diag(list(nnms_in0.values())).tocsr()    
    
    ovt = mdata["knn"]    
    ovt0 = ovt.copy()
    ovt0.data[:]=1

    B = ovt
    # already sum-normalized per species
    #s = B.sum(1).A
    #s[s == 0] = 1
    #B = B.multiply(1 / s).tocsr()

    print("Indegree coarsening")
    
    numiter = nnm_internal0.shape[0] // chunksize + 1

    D = sp.sparse.csr_matrix((0, nnm_internal0.shape[0]))
    if flag:
        Cs=[]
        for it,sid in enumerate(sams.keys()):
            nfk = neigh_from_keys[sid]
            if nfk:
                Cs.append(nnms_in[sid].dot(nnms_in[sid].T.dot(B.T[species_indexer[it]])))
            else:
                Cs.append(nnms_in[sid].dot(B.T[species_indexer[it]]))
        D = sp.sparse.vstack(Cs).T
        del Cs
        gc.collect()
    else:
        for bl in range(numiter):
            print(str(bl) + "/" + str(numiter), D.shape)
            C = B[bl * chunksize : (bl + 1) * chunksize].dot(nnm_internal.T)
            C.data[C.data < 0.1] = 0
            C.eliminate_zeros()

            D = sp.sparse.vstack((D, C))        
            del C
            gc.collect()

    D = D.multiply(D.T).tocsr()
    D.data[:] = D.data**0.5
    mdata["xsim"] = D    
    
    if scale_edges_by_corr:
        print('Rescaling edge weights by expression correlations.')                
        x,y = D.nonzero()
        vals = _replace(mdata["wPCA"],x,y)
        vals[vals<1e-3]=1e-3
        
        # make a copy and write correlations
        F = D.copy()
        F.data[:] = vals
        
        # normalize by maximum and rescale with tanh
        ma = F.max(1).toarray()
        ma[ma==0]=1
        F = F.multiply(1/ma).tocsr()
        F.data[:] = _tanh_scale(F.data,center=0.7,scale=10)
        
        # get max aligment score from before
        ma = D.max(1).toarray()
        ma[ma==0]=1
        
        # geometric mean expression correlation scores by alignment scores
        D = F.multiply(D).tocsr()
        D.data[:] = np.sqrt(D.data)
        
        # get new max scores
        ma2 = D.max(1).toarray()
        ma2[ma2==0]=1
        
        # change new max scores to old max scores
        D = D.multiply(ma/ma2).tocsr()
    
    species_list = []
    for sid in sams.keys():
        species_list += [sid]*sams[sid].adata.shape[0]    
    species_list = np.array(species_list)

    if not pairwise or len(sams.keys())==2:
        Dk = sparse_knn(D, k1).tocsr()
        denom = k1
    else:
        Dk=[]
        for sid1 in sams.keys():
            row=[]
            for sid2 in sams.keys():
                if sid1 != sid2:
                    Dsubk = sparse_knn(D[species_list==sid1][:,species_list==sid2], k1).tocsr()
                else:
                    Dsubk = sp.sparse.csr_matrix((sams[sid1].adata.shape[0],)*2)
                row.append(Dsubk)
            Dk.append(sp.sparse.hstack(row))
        Dk = sp.sparse.vstack(Dk).tocsr()
        denom = (k1 * (len(sams.keys())-1))
            
    sr = Dk.sum(1).A    
    
    x = 1 - sr.flatten() / denom
    
    sr[sr==0]=1
    st = Dk.sum(0).A.flatten()[None,:]
    st[st==0]=1
    proj = Dk.multiply(1 / sr).dot(Dk.multiply(1 / st)).tocsr()
    z = proj.copy()
    z.data[:] = 1
    idx = np.where(z.sum(1).A.flatten() >= k1)[0]
    
    omp = nnm_internal0
    omp.data[:]=1
    s = proj.max(1).toarray()
    s[s == 0] = 1
    proj = proj.multiply(1 / s).tocsr()    
    X, Y = omp.nonzero()
    X2 = X[np.in1d(X, idx)]
    Y2 = Y[np.in1d(X, idx)]

    omp = omp.tolil()
    omp[X2, Y2] = np.vstack((proj[X2, Y2].A.flatten(), np.ones(X2.size) * 0.3)).max(
        0
    )

    omp = nnm_internal0.tocsr()
    NNM = omp.multiply(x[:, None])
    NNM = (NNM+Dk).tolil()
    NNM.setdiag(0)
   
    print("Concatenating SAM objects...")
    sam3 = _concatenate_sam(sams, NNM)
    
    sam3.adata.obs["species"] = pd.Categorical(species_list)

    sam3.adata.uns["gnnm_corr"] = mdata.get("gnnm_corr",None)

    if umap:
        print("Computing UMAP projection...")
        sc.tl.umap(sam3.adata, min_dist=0.1, maxiter = 500 if sam3.adata.shape[0] <= 10000 else 200)
    return sam3

def _refine_corr(
    sams,
    st,
    gnnm,
    gns_dict,
    corr_mode="mutual_info",
    THR=0,
    use_seq=False,
    T1=0.25,
    NCLUSTERS=1,
    ncpus=os.cpu_count(),
    wscale=False
):
    # import networkx as nx
    gns = np.concatenate(list(gns_dict.values()))

    x, y = gnnm.nonzero()
    sam = list(sams.values())[0]
    cl = sam.leiden_clustering(gnnm, res=0.5)
    ix = np.argsort(cl)
    NGPC = gns.size // NCLUSTERS + 1

    ixs = []
    for i in range(NCLUSTERS):
        ixs.append(np.sort(ix[i * NGPC : (i + 1) * NGPC]))

    assert np.concatenate(ixs).size == gns.size

    GNNMSUBS = []
    GNSUBS = []
    for i in range(len(ixs)):
        ixs[i] = np.unique(np.append(ixs[i], gnnm[ixs[i], :].nonzero()[1]))
        gnnm_sub = gnnm[ixs[i], :][:, ixs[i]]
        gnsub = gns[ixs[i]]
        gns_dict_sub={}
        for sid in gns_dict.keys():
            gn = gns_dict[sid]
            gns_dict_sub[sid] = gn[np.in1d(gn,gnsub)]

        gnnm2_sub = _refine_corr_parallel(
            sams,
            st,
            gnnm_sub,
            gns_dict_sub,
            corr_mode=corr_mode,
            THR=THR,
            use_seq=use_seq,
            T1=T1,
            ncpus=ncpus,
            wscale=wscale
        )
        GNNMSUBS.append(gnnm2_sub)
        GNSUBS.append(gnsub)
        gc.collect()
    
    I = []
    P = []
    for i in range(len(GNNMSUBS)):
        I.append(
            np.unique(np.sort(np.vstack((GNNMSUBS[i].nonzero())).T, axis=1), axis=0)
        )
        P.append(GNSUBS[i][I[-1]])

    GNS = pd.DataFrame(data=np.arange(gns.size)[None, :], columns=gns)
    gnnm3 = sp.sparse.lil_matrix(gnnm.shape)
    for i in range(len(I)):
        x, y = GNS[P[i][:, 0]].values.flatten(), GNS[P[i][:, 1]].values.flatten()
        gnnm3[x, y] = GNNMSUBS[i][I[i][:, 0], I[i][:, 1]].A.flatten()

    gnnm3 = gnnm3.tocsr()
    x, y = gnnm3.nonzero()
    # gnnm3[y,x]=gnnm3.data
    gnnm3 = gnnm3.tolil()
    gnnm3[y, x] = gnnm3[x, y].A.flatten()
    gnnm3 = gnnm3.tocsr()
    return gnnm3

def _prepend_blast_prefix(data, pre):
    x = [str(x).split("_")[0] for x in data]
    vn = []
    for i,g in enumerate(data):
        if x[i] != pre:
            vn.append(pre+"_"+g)
        else:
            vn.append(g)
    return np.array(vn).astype('str').astype('object')

def _calculate_blast_graph(ids, f_maps="maps/", eval_thr=1e-6, reciprocate=False):
    gns = []
    Xs=[]
    Ys=[]
    Vs=[]
    
    for i in range(len(ids)):
        id1=ids[i]
        for j in range(i,len(ids)):
            id2=ids[j]
            if i!=j:
                if os.path.exists(f_maps + "{}{}".format(id1, id2)):
                    fA = f_maps + "{}{}/{}_to_{}.txt".format(id1, id2, id1, id2)
                    fB = f_maps + "{}{}/{}_to_{}.txt".format(id1, id2, id2, id1)
                elif os.path.exists(f_maps + "{}{}".format(id2, id1)):
                    fA = f_maps + "{}{}/{}_to_{}.txt".format(id2, id1, id1, id2)
                    fB = f_maps + "{}{}/{}_to_{}.txt".format(id2, id1, id2, id1)
                else:
                    raise FileExistsError(
                        "BLAST mapping tables with the input IDs ({} and {}) not found in the specified path.".format(
                            id1, id2
                        )
                    )

                A = pd.read_csv(fA, sep="\t", header=None, index_col=0)
                B = pd.read_csv(fB, sep="\t", header=None, index_col=0)

                A.columns = A.columns.astype("<U100")
                B.columns = B.columns.astype("<U100")

                A = A[A.index.astype("str") != "nan"]
                A = A[A.iloc[:, 0].astype("str") != "nan"]
                B = B[B.index.astype("str") != "nan"]
                B = B[B.iloc[:, 0].astype("str") != "nan"]

                A.index = _prepend_blast_prefix(A.index,id1)
                B[B.columns[0]] = _prepend_blast_prefix(B.iloc[:, 0].values.flatten(),id1)

                B.index = _prepend_blast_prefix(B.index,id2)
                A[A.columns[0]] = _prepend_blast_prefix(A.iloc[:, 0].values.flatten(),id2)

                i1 = np.where(A.columns == "10")[0][0]
                i3 = np.where(A.columns == "11")[0][0]

                inA = q(A.index)
                inB = q(B.index)

                inA2 = q(A.iloc[:, 0])
                inB2 = q(B.iloc[:, 0])
                gn1 = np.unique(np.append(inB2, inA))
                gn2 = np.unique(np.append(inA2, inB))
                gn = np.append(gn1, gn2)
                gnind = pd.DataFrame(data=np.arange(gn.size)[None, :], columns=gn)

                A.index = pd.Index(gnind[A.index].values.flatten())
                B.index = pd.Index(gnind[B.index].values.flatten())
                A[A.columns[0]] = gnind[A.iloc[:, 0].values.flatten()].values.flatten()
                B[B.columns[0]] = gnind[B.iloc[:, 0].values.flatten()].values.flatten()

                Arows = np.vstack((A.index, A.iloc[:, 0], A.iloc[:, i3])).T
                Arows = Arows[A.iloc[:, i1].values.flatten() <= eval_thr, :]
                gnnm1 = sp.sparse.lil_matrix((gn.size,) * 2)
                gnnm1[Arows[:, 0].astype("int32"), Arows[:, 1].astype("int32")] = Arows[
                    :, 2
                ]  # -np.log10(Arows[:,2]+1e-200)

                Brows = np.vstack((B.index, B.iloc[:, 0], B.iloc[:, i3])).T
                Brows = Brows[B.iloc[:, i1].values.flatten() <= eval_thr, :]
                gnnm2 = sp.sparse.lil_matrix((gn.size,) * 2)
                gnnm2[Brows[:, 0].astype("int32"), Brows[:, 1].astype("int32")] = Brows[
                    :, 2
                ]  # -np.log10(Brows[:,2]+1e-200)

                gnnm = (gnnm1 + gnnm2).tocsr()
                gnnms = (gnnm + gnnm.T) / 2
                if reciprocate:
                    gnnm.data[:] = 1
                    gnnms = gnnms.multiply(gnnm).multiply(gnnm.T).tocsr()
                gnnm = gnnms

                f1 = np.where(np.in1d(gn,gn1))[0]
                f2 = np.where(np.in1d(gn,gn2))[0]
                f = np.append(f1,f2)
                gn = gn[f]
                gnnm = gnnm[f,:][:,f]
                
                V = gnnm.data
                X,Y = gnnm.nonzero()
                
                Xs.extend(gn[X])
                Ys.extend(gn[Y])
                Vs.extend(V)
                gns.extend(gn)
    
    gns = np.unique(gns)
    gns_sp = np.array([x.split('_')[0] for x in gns])
    gns2 = []
    gns_dict={}
    for sid in ids:
        gns2.append(gns[gns_sp==sid])
        gns_dict[sid] = gns2[-1]
    gns = np.concatenate(gns2)
    indexer = pd.Series(index=gns,data=np.arange(gns.size))
    
    X = indexer[Xs].values
    Y = indexer[Ys].values
    gnnm = sp.sparse.coo_matrix((Vs,(X,Y)),shape=(gns.size,gns.size)).tocsr()
    
    return gnnm, gns, gns_dict

def _coarsen_blast_graph(gnnm, gns, names):
    # eliminate zero entries in gnnm
    gnnm=gnnm.tocsr()
    gnnm.eliminate_zeros()

    sps = np.array([x.split('_')[0] for x in gns])
    sids = np.unique(sps)
    ss=[]
    for sid in sids:
        n = names.get(sid,None)
        if n is not None:
            n = np.array(n)
            n = (sid+'_'+n.astype('object')).astype('str')
            s1 = pd.Series(index=n[:,0],data=n[:,1])
            g = gns[sps==sid]
            g = g[np.in1d(g,n[:,0],invert=True)]
            s2 = pd.Series(index=g,data = g)
            s = pd.concat([s1,s2])
        else:
            s = pd.Series(index=gns[sps==sid],data = gns[sps==sid])
        ss.append(s)
    ss = pd.concat(ss)
    ss = ss[np.unique(q(ss.index),return_index=True)[1]]
    x,y = gnnm.nonzero() #get nonzeros
    s = pd.Series(data=gns,index=np.arange(gns.size)) # convert indices to gene pairs
    xn,yn = s[x].values,s[y].values 
    xg,yg = ss[xn].values,ss[yn].values #convert gene pairs to translated

    da=gnnm.data

    zgu,ix,ivx,cu = np.unique(np.array([xg,yg]).astype('str'),axis=1,return_counts=True,return_index=True,return_inverse=True) # find unique pairs

    xgu,ygu = zgu[:,cu>1] # extract pairs that appear duplicated times
    xgyg=q(xg.astype('object')+';'+yg.astype('object'))
    xguygu=q(xgu.astype('object')+';'+ygu.astype('object'))

    filt = np.in1d(xgyg,xguygu)

    DF=pd.DataFrame(data=xgyg[filt][:,None],columns=['key'])
    DF['val']=da[filt]

    dic = df_to_dict(DF,key_key='key')

    xgu = q([x.split(';')[0] for x in dic.keys()])
    ygu = q([x.split(';')[1] for x in dic.keys()])
    replz = q([max(dic[x]) for x in dic.keys()])

    xgu1,ygu1 = zgu[:,cu==1] # get non-duplicate pairs
    xg = np.append(xgu1, xgu) # append duplicate pairs
    yg = np.append(ygu1, ygu)
    da = np.append(da[ix][cu==1],replz) # append duplicate scores to the non-duplicate scores
    gn = np.unique(np.append(xg,yg)) # get the unique genes

    s = pd.Series(data=np.arange(gn.size),index=gn) # create an indexer
    xn,yn = s[xg].values,s[yg].values # convert gene pairs to indexes
    gnnm = sp.sparse.coo_matrix((da,(xn,yn)),shape=(gn.size,)*2).tocsr() # create sparse matrix

    f = gnnm.sum(1).A.flatten() != 0 #eliminate zero rows/columns
    gn = gn[f]
    sps = np.array([x.split('_')[0] for x in gn])

    gns_dict={}
    for sid in sids:
        gns_dict[sid] = gn[sps==sid]

    return gnnm, gns_dict, gn


def prepare_SAMap_loadings(sam, npcs=300):
    """ Prepares SAM object to contain the proper PC loadings associated with its manifold.
    Deposits the loadings in `sam.adata.varm['PCs_SAMap']`.
    
    Parameters
    ----------    
    sam - SAM object
    
    npcs - int, optional, default 300
        The number of PCs to calculate loadings for.
    
    """
    ra = sam.adata.uns["run_args"]
    preprocessing = ra.get("preprocessing", "StandardScaler")
    weight_PCs = ra.get("weight_PCs", False)
    A, _ = sam.calculate_nnm(
        n_genes=sam.adata.shape[1],
        preprocessing=preprocessing,
        npcs=npcs,
        weight_PCs=weight_PCs,
        sparse_pca=True,
        update_manifold=False,
        weight_mode='dispersion'
    )
    sam.adata.varm["PCs_SAMap"] = A


def _concatenate_sam(sams, nnm):
    acns = []
    exps = []
    agns = []
    sps = []
    for i,sid in enumerate(sams.keys()):
        acns.append(q(sams[sid].adata.obs_names))
        sps.append([sid]*acns[-1].size)
        exps.append(sams[sid].adata.X)
        agns.append(q(sams[sid].adata.var_names))


    acn = np.concatenate(acns)
    agn = np.concatenate(agns)
    sps = np.concatenate(sps)

    xx = sp.sparse.block_diag(exps,format='csr')


    sam = SAM(counts=(xx, agn, acn))

    sam.adata.uns["neighbors"] = {}
    nnm = nnm.tocsr()
    nnm.eliminate_zeros()
    sam.adata.obsp["connectivities"] = nnm
    sam.adata.uns["neighbors"]["params"] = {
        "n_neighbors": 15,
        "method": "umap",
        "use_rep": "X",
        "metric": "euclidean",
    }
    for i in sams.keys():
        for k in sams[i].adata.obs.keys():
            if sams[i].adata.obs[k].dtype.name == "category":
                z = np.array(['unassigned']*sam.adata.shape[0],dtype='object')
                z[sps==i] = q(sams[i].adata.obs[k])
                sam.adata.obs[i+'_'+k] = pd.Categorical(z)

    a = []
    for i,sid in enumerate(sams.keys()):
        a.extend(["batch" + str(i + 1)] * sams[sid].adata.shape[0])
    sam.adata.obs["batch"] = pd.Categorical(np.array(a))
    sam.adata.obs.columns = sam.adata.obs.columns.astype("str")
    sam.adata.var.columns = sam.adata.var.columns.astype("str")

    for i in sam.adata.obs:
        sam.adata.obs[i] = sam.adata.obs[i].astype("str")
        
    return sam

def _map_features_un(A, B, sam1, sam2, thr=1e-6):
    i1 = np.where(A.columns == "10")[0][0]
    i3 = np.where(A.columns == "11")[0][0]

    inA = q(A.index)
    inB = q(B.index)

    gn1 = q(sam1.adata.var_names)
    gn2 = q(sam2.adata.var_names)

    gn1 = gn1[np.in1d(gn1, inA)]
    gn2 = gn2[np.in1d(gn2, inB)]

    A = A.iloc[np.in1d(inA, gn1), :]
    B = B.iloc[np.in1d(inB, gn2), :]

    inA2 = q(A.iloc[:, 0])
    inB2 = q(B.iloc[:, 0])

    A = A.iloc[np.in1d(inA2, gn2), :]
    B = B.iloc[np.in1d(inB2, gn1), :]

    gn = np.append(gn1, gn2)
    gnind = pd.DataFrame(data=np.arange(gn.size)[None, :], columns=gn)

    A.index = pd.Index(gnind[A.index].values.flatten())
    B.index = pd.Index(gnind[B.index].values.flatten())
    A.iloc[:, 0] = gnind[A.iloc[:, 0].values.flatten()].values.flatten()
    B.iloc[:, 0] = gnind[B.iloc[:, 0].values.flatten()].values.flatten()

    Arows = np.vstack((A.index, A.iloc[:, 0], A.iloc[:, i3])).T
    Arows = Arows[A.iloc[:, i1].values.flatten() <= thr, :]
    gnnm1 = sp.sparse.lil_matrix((gn.size,) * 2)
    gnnm1[Arows[:, 0].astype("int32"), Arows[:, 1].astype("int32")] = Arows[
        :, 2
    ]  # -np.log10(Arows[:,2]+1e-200)

    Brows = np.vstack((B.index, B.iloc[:, 0], B.iloc[:, i3])).T
    Brows = Brows[B.iloc[:, i1].values.flatten() <= thr, :]
    gnnm2 = sp.sparse.lil_matrix((gn.size,) * 2)
    gnnm2[Brows[:, 0].astype("int32"), Brows[:, 1].astype("int32")] = Brows[
        :, 2
    ]  # -np.log10(Brows[:,2]+1e-200)

    gnnm = (gnnm1 + gnnm2).tocsr()
    gnnms = (gnnm + gnnm.T) / 2
    gnnm.data[:] = 1
    gnnms = gnnms.multiply(gnnm).multiply(gnnm.T).tocsr()
    return gnnms, gn1, gn2


def _filter_gnnm(gnnm, thr=0.25):
    x, y = gnnm.nonzero()
    mas = gnnm.max(1).toarray().flatten()
    gnnm4 = gnnm.copy()
    gnnm4.data[gnnm4[x, y].A.flatten() < mas[x] * thr] = 0
    gnnm4.eliminate_zeros()
    x, y = gnnm4.nonzero()
    z = gnnm4.data
    gnnm4 = gnnm4.tolil()
    gnnm4[y, x] = z
    gnnm4 = gnnm4.tocsr()
    return gnnm4


def _get_pairs(sams, gnnm, gns_dict, NOPs1=0, NOPs2=0):
    # gnnm = filter_gnnm(gnnm)
    su = gnnm.max(1).toarray()
    su[su == 0] = 1
    gnnm = gnnm.multiply(1 / su).tocsr()
    Ws = {}
    for sid in sams.keys():
        Ws[sid] = sams[sid].adata.var["weights"][gns_dict[sid]].values
    
    W = np.concatenate(list(Ws.values()))
    W[W < 0.0] = 0
    W[W > 0.0] = 1

    B = gnnm.multiply(W[None, :]).multiply(W[:, None]).tocsr()
    B.eliminate_zeros()

    return B

@njit
def nb_unique1d(ar):
    """
    Find the unique elements of an array, ignoring shape.
    """
    ar = ar.flatten()

    optional_indices = True

    if optional_indices:
        perm = ar.argsort(kind='mergesort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    if aux.shape[0] > 0 and aux.dtype.kind in "cfmM" and np.isnan(aux[-1]):
        if aux.dtype.kind == "c":  # for complex all NaNs are considered equivalent
            aux_firstnan = np.searchsorted(np.isnan(aux), True, side='left')
        else:
            aux_firstnan = np.searchsorted(aux, aux[-1], side='left')
        mask[1:aux_firstnan] = (aux[1:aux_firstnan] != aux[:aux_firstnan - 1])
        mask[aux_firstnan] = True
        mask[aux_firstnan + 1:] = False
    else:
        mask[1:] = aux[1:] != aux[:-1]


    imask = np.cumsum(mask) - 1
    inv_idx = np.empty(mask.shape, dtype=np.intp)
    inv_idx[perm] = imask
    idx = np.append(np.nonzero(mask)[0],mask.size)

                     #idx      #inverse   #counts
    return aux[mask],perm[mask],inv_idx,np.diff(idx)

@njit
def _xicorr(X,Y):
    '''xi correlation coefficient'''
    n = X.size
    xi = np.argsort(X,kind='quicksort')
    Y = Y[xi]
    _,_,b,c = nb_unique1d(Y)
    r = np.cumsum(c)[b]
    _,_,b,c = nb_unique1d(-Y)
    l = np.cumsum(c)[b]
    denominator = (2*(l*(n-l)).sum())
    if denominator > 0:
        return 1 - n*np.abs(np.diff(r)).sum() / denominator
    else:
        return 0

@njit(parallel=True)
def _refine_corr_kernel(p, ps, sids, sixs, indptr,indices,data, n, corr_mode):
    p1 = p[:,0]
    p2 = p[:,1]

    ps1 = ps[:,0]
    ps2 = ps[:,1]

    d = {}
    for i in range(len(sids)):
        d[sids[i]] = sixs[i]
    
    res = np.zeros(p1.size)

    for j in prange(len(p1)):
        j1, j2 = p1[j], p2[j]
        pl1d = data[indptr[j1] : indptr[j1 + 1]]
        pl1i = indices[indptr[j1] : indptr[j1 + 1]]

        sc1d = data[indptr[j2] : indptr[j2 + 1]]
        sc1i = indices[indptr[j2] : indptr[j2 + 1]]

        x = np.zeros(n)
        x[pl1i] = pl1d
        y = np.zeros(n)
        y[sc1i] = sc1d

        a1, a2 = ps1[j], ps2[j]
        ix1 = d[a1]
        ix2 = d[a2]
        

        xa,xb,ya,yb = x[ix1],x[ix2],y[ix1],y[ix2]
        xx=np.append(xa,xb)
        yy=np.append(ya,yb)
        
        if corr_mode == "pearson":
            c = ((xx-xx.mean())*(yy-yy.mean()) / xx.std() / yy.std()).sum() / xx.size
        else:
            c = _xicorr(xx,yy)
        res[j] = c
    return res  

def _refine_corr_parallel(
    sams,
    st,
    gnnm,
    gns_dict,
    corr_mode="pearson",
    THR=0,
    use_seq=False,
    T1=0.0,
    ncpus=os.cpu_count(),
    wscale=False
):

    import scipy as sp

    gn = np.concatenate(list(gns_dict.values()))

    Ws = []
    ix = []
    for sid in sams.keys():
        Ws.append(sams[sid].adata.var["weights"][gns_dict[sid]].values)
        ix += [sid]*gns_dict[sid].size
    ix = np.array(ix)
    w = np.concatenate(Ws)

    w[w > T1] = 1
    w[w < 1] = 0

    gnO = gn[w > 0]
    ix = ix[w > 0]
    gns_dictO = {}
    for sid in gns_dict.keys():
        gns_dictO[sid] = gnO[ix==sid]

    gnnmO = gnnm[w > 0, :][:, w > 0]
    x, y = gnnmO.nonzero()

    pairs = np.unique(np.sort(np.vstack((x, y)).T, axis=1), axis=0)

    xs, ys = q([i.split('_')[0] for i in gnO[pairs[:,0]]]), q([i.split('_')[0] for i in gnO[pairs[:,1]]])    
    pairs_species = np.vstack((xs,ys)).T

    nnm = st.adata.obsp["connectivities"]
    xs = []
    nnms = []
    for i,sid in enumerate(sams.keys()):
        nnms.append(nnm[:,(st.adata.obs['batch'] == f'batch{i+1}').values])
        s1 = nnms[-1].sum(1).A
        s1[s1 < 1e-3] = 1
        s1 = s1.flatten()[:, None]  
        nnms[-1] = nnms[-1].multiply(1 / s1)

        xs.append(sams[sid].adata[:,gns_dictO[sid]].X.astype("float32"))

    Xs = sp.sparse.block_diag(xs).tocsc()
    nnms = sp.sparse.hstack(nnms).tocsr()
    Xavg = nnms.dot(Xs).tocsc()


    p = pairs
    ps = pairs_species

    gnnm2 = gnnm.multiply(w[:, None]).multiply(w[None, :]).tocsr()
    x, y = gnnm2.nonzero()
    pairs = np.unique(np.sort(np.vstack((x, y)).T, axis=1), axis=0)

    species = q(st.adata.obs['species'])
    sixs = []
    sidss = np.unique(species)
    for sid in sidss:
        sixs.append(np.where(species==sid)[0])
    
    vals = _refine_corr_kernel(p,ps,sidss,sixs,Xavg.indptr,Xavg.indices,Xavg.data,Xavg.shape[0], corr_mode)
    vals[np.isnan(vals)]=0

    CORR = dict(zip(to_vn(np.vstack((gnO[p[:,0]],gnO[p[:,1]])).T),vals))

    for k in CORR.keys():
        CORR[k] = 0 if CORR[k] < THR else CORR[k]
        if wscale:
            id1,id2 = [x.split('_')[0] for x in k.split(';')]
            weight1 = sams[id1].adata.var["weights"][k.split(';')[0]]
            weight2 = sams[id2].adata.var["weights"][k.split(';')[1]]
            CORR[k] = np.sqrt(CORR[k] * np.sqrt(weight1 * weight2))   

    CORR = np.array([CORR[x] for x in to_vn(gn[pairs])])    

    gnnm3 = sp.sparse.lil_matrix(gnnm.shape)

    if use_seq:
        gnnm3[pairs[:, 0], pairs[:, 1]] = (
            CORR * gnnm2[pairs[:, 0], pairs[:, 1]].A.flatten()
        )
        gnnm3[pairs[:, 1], pairs[:, 0]] = (
            CORR * gnnm2[pairs[:, 1], pairs[:, 0]].A.flatten()
        )
    else:
        gnnm3[pairs[:, 0], pairs[:, 1]] = CORR  # *gnnm2[x,y].A.flatten()
        gnnm3[pairs[:, 1], pairs[:, 0]] = CORR  # *gnnm2[x,y].A.flatten()

    gnnm3 = gnnm3.tocsr()
    gnnm3.eliminate_zeros()
    return gnnm3 

try:
    from fast_histogram import histogram2d
except:
    pass;

def hist2d(X,Y,bins=100,domain=None):
    if domain is None:
        xmin = X.min()
        xmax = X.max()
        ymin=Y.min()
        ymax=Y.max()    
        domain = [(xmin,xmax),(ymin,ymax)]
    return histogram2d(X,Y,bins,domain)

def calc_MI(X,Y,bins=100,cl=None,cs=None):    
    xmin = X.min()
    xmax = X.max()
    ymin=Y.min()
    ymax=Y.max()    
    domain = [(xmin,xmax),(ymin,ymax)]
    c_XY = hist2d(X,Y,bins=bins,domain=domain)
    c_X = c_XY.sum(1)
    c_Y = c_XY.sum(0)
    
    c = c_XY
    c_normalized = c / c.sum()
    c1 = c_normalized.sum(1)
    c2 = c_normalized.sum(0)
    c1[c1==0]=1
    c2[c2==0]=1
    c_normalized[c_normalized==0]=1
    H_X = -(c1*np.log2(c1)).sum()
    H_Y = -(c2*np.log2(c2)).sum()
    H_XY = -(c_normalized*np.log2(c_normalized)).sum()
    
    H_Y = 0 if H_Y < 0 else H_Y
    H_X = 0 if H_X < 0 else H_X
    H_XY = 0 if H_XY < 0 else H_XY

    MI = H_X + H_Y - H_XY
    if MI <= 0 or H_X <= 0 or H_Y <= 0:
        return 0
    else:
        return MI / np.sqrt(H_X*H_Y)

def _parallel_wrapper(j):
    j1, j2 = p[j, 0], p[j, 1]

    pl1d = Xavg.data[Xavg.indptr[j1] : Xavg.indptr[j1 + 1]]
    pl1i = Xavg.indices[Xavg.indptr[j1] : Xavg.indptr[j1 + 1]]

    sc1d = Xavg.data[Xavg.indptr[j2] : Xavg.indptr[j2 + 1]]
    sc1i = Xavg.indices[Xavg.indptr[j2] : Xavg.indptr[j2 + 1]]

    x = np.zeros(Xavg.shape[0])
    x[pl1i] = pl1d
    y = np.zeros(Xavg.shape[0])
    y[sc1i] = sc1d

    ha = gnsO[j1] + ";" + gnsO[j2]

    try:
        if corr_mode == 'xicorr':
            CORR[ha] = _xicorr(x,y)
        elif corr_mode == 'mutual_info':
            CORR[ha] = calc_MI(x,y,cl=cl,cs=cs)
        else:
            raise ValueError(f'`{corr_mode}` not recognized.')
    except:
        CORR[ha] = 0


def _united_proj(wpca1, wpca2, k=20, metric="cosine", ef=200, M=48):

    metric = 'l2' if metric == 'euclidean' else metric
    metric = 'cosine' if metric == 'correlation' else metric
    labels2 = np.arange(wpca2.shape[0])
    p2 = hnswlib.Index(space=metric, dim=wpca2.shape[1])
    p2.init_index(max_elements=wpca2.shape[0], ef_construction=ef, M=M)
    p2.add_items(wpca2, labels2)
    p2.set_ef(ef)
    idx1, dist1 = p2.knn_query(wpca1, k=k)

    if metric == 'cosine':
        dist1 = 1 - dist1
        dist1[dist1 < 1e-3] = 1e-3
        dist1 = dist1/dist1.max(1)[:,None]
        dist1 = _tanh_scale(dist1,scale=10, center=0.7)
    else:
        sigma1 = dist1[:,4]
        sigma1[sigma1<1e-3]=1e-3
        dist1 = np.exp(-dist1/sigma1[:,None])
        
    Sim1 = dist1  # np.exp(-1*(1-dist1)**2)
    knn1v2 = sp.sparse.lil_matrix((wpca1.shape[0], wpca2.shape[0]))
    x1 = np.tile(np.arange(idx1.shape[0])[:, None], (1, idx1.shape[1])).flatten()
    knn1v2[x1.astype('int32'), idx1.flatten().astype('int32')] = Sim1.flatten()
    return knn1v2.tocsr()

def _tanh_scale(x,scale=10,center=0.5):
    return center + (1-center) * np.tanh(scale * (x - center))

def _mapping_window(sams, gnnm=None, gns=None, K=20, pairwise=True):
    k = K
    output_dict = {}
    if gnnm is not None and gns is not None:
        print('Prepping datasets for translation.')        
        gnnm_corr = gnnm.copy()
        gnnm_corr.data[:] = _tanh_scale(gnnm_corr.data)

        std = StandardScaler(with_mean=False)    

        gs = {}
        adatas={}
        Ws={}
        ss={}
        As={}
        species_indexer = []   
        genes_indexer = [] 
        for sid in sams.keys():
            gs[sid] = gns[np.in1d(gns,q(sams[sid].adata.var_names))]
            adatas[sid] = sams[sid].adata[:,gs[sid]]
            Ws[sid] = adatas[sid].var["weights"].values
            ss[sid] = std.fit_transform(adatas[sid].X).multiply(Ws[sid][None,:]).tocsr()
            species_indexer.append(np.arange(ss[sid].shape[0]))
            genes_indexer.append(np.arange(gs[sid].size))

        for i in range(1,len(species_indexer)):
            species_indexer[i] = species_indexer[i]+species_indexer[i-1].max()+1
            genes_indexer[i] = genes_indexer[i]+genes_indexer[i-1].max()+1

        su = gnnm_corr.sum(0).A
        su[su==0]=1
        gnnm_corr = gnnm_corr.multiply(1/su).tocsr()
        
        X = sp.sparse.block_diag(list(ss.values())).tocsr()
        W = np.concatenate(list(Ws.values())).flatten()

        ttt=time.time()
        if pairwise:
            print('Translating feature spaces pairwise.')
            Xtr = []
            for i,sid1 in enumerate(sams.keys()):
                xtr = []
                for j,sid2 in enumerate(sams.keys()):
                    if i != j:
                        gnnm_corr_sub = gnnm_corr[genes_indexer[i]][:,genes_indexer[j]]
                        su = gnnm_corr_sub.sum(0).A
                        su[su==0]=1
                        gnnm_corr_sub = gnnm_corr_sub.multiply(1/su).tocsr()                        
                        xtr.append(X[species_indexer[i]][:,genes_indexer[i]].dot(gnnm_corr_sub))
                        xtr[-1] = std.fit_transform(xtr[-1]).multiply(W[genes_indexer[j]][None,:])
                    else:
                        xtr.append(sp.sparse.csr_matrix((species_indexer[i].size,genes_indexer[i].size)))
                Xtr.append(sp.sparse.hstack(xtr))
            Xtr = sp.sparse.vstack(Xtr)
        else:
            print('Translating feature spaces all-to-all.')    

            Xtr = []
            for i,sid in enumerate(sams.keys()):
                Xtr.append(X[species_indexer[i]].dot(gnnm_corr))
                Xtr[-1] = std.fit_transform(Xtr[-1]).multiply(W[None,:])
            Xtr = sp.sparse.vstack(Xtr)
        Xc = (X + Xtr).tocsr()

        mus = []        
        for i,sid in enumerate(sams.keys()):
            mus.append(Xc[species_indexer[i]].mean(0).A.flatten())

        gc.collect()   
        
        print('Projecting data into joint latent space.',time.time()-ttt) 
        C = sp.linalg.block_diag(*[adatas[sid].varm["PCs_SAMap"] for sid in sams.keys()])
        M = np.vstack(mus).dot(C)    
        ttt=time.time()    
        it = 0;
        PCAs=[]
        for sid in sams.keys():
            PCAs.append(Xc[:,it : it + gs[sid].size].dot(adatas[sid].varm["PCs_SAMap"]))
            it+=gs[sid].size
        wpca = np.hstack(PCAs)#Xc.dot(C)

        print('Correcting data with means.',time.time()-ttt)            
        for i,sid in enumerate(sams.keys()):
            ixq = species_indexer[i]
            wpca[ixq] -= M[i]       
        output_dict["gnnm_corr"] = gnnm_corr 
    else:
        std = StandardScaler(with_mean=False)    

        gs = {}
        adatas={}
        Ws={}
        ss={}
        As={}
        species_indexer = []    
        mus=[]
        for sid in sams.keys():
            adatas[sid] = sams[sid].adata
            Ws[sid] = adatas[sid].var["weights"].values
            ss[sid] = std.fit_transform(adatas[sid].X).multiply(Ws[sid][None,:]).tocsr()
            mus.append(ss[sid].mean(0).A.flatten())
            species_indexer.append(np.arange(ss[sid].shape[0]))        
        for i in range(1,len(species_indexer)):
            species_indexer[i] = species_indexer[i]+species_indexer[i-1].max()+1            
        X = sp.sparse.vstack(list(ss.values()))
        C = np.hstack([adatas[sid].varm["PCs_SAMap"] for sid in sams.keys()])
        wpca = X.dot(C)
        M = np.vstack(mus).dot(C)         
        for i,sid in enumerate(sams.keys()):
            ixq = species_indexer[i]
            wpca[ixq] -= M[i]            
        
    ixg = np.arange(wpca.shape[0])
    Xs = []
    Ys = []
    Vs = []
    for i,sid in enumerate(sams.keys()):
        ixq = species_indexer[i]
        query = wpca[ixq]          
        
        for j,sid2 in enumerate(sams.keys()):
            if i!=j:
                ixr = species_indexer[j]
                reference = wpca[ixr]

                b = _united_proj(query, reference, k=k)
                
                # sum-normalize each species individually.
                su = b.sum(1).A
                su[su==0]=1
                b = b.multiply(1/su).tocsr()

                A = pd.Series(index = np.arange(b.shape[0]), data = ixq)        
                B = pd.Series(index = np.arange(b.shape[1]), data = ixr)

                x,y = b.nonzero()
                x,y = A[x].values,B[y].values
                Xs.extend(x)
                Ys.extend(y)
                Vs.extend(b.data)
            
    knn = sp.sparse.coo_matrix((Vs,(Xs,Ys)),shape=(ixg.size,ixg.size))

    output_dict["knn"] = knn.tocsr()
    output_dict["wPCA"] = wpca
    return output_dict


def _sparse_knn_ks(D, ks):
    D1 = D.tocoo()
    idr = np.argsort(D1.row)
    D1.row[:] = D1.row[idr]
    D1.col[:] = D1.col[idr]
    D1.data[:] = D1.data[idr]

    row, ind = np.unique(D1.row, return_index=True)
    ind = np.append(ind, D1.data.size)
    for i in range(ind.size - 1):
        idx = np.argsort(D1.data[ind[i] : ind[i + 1]])
        k = ks[row[i]]
        if idx.size > k:
            if k != 0:
                idx = idx[:-k]
            else:
                idx = idx
            D1.data[np.arange(ind[i], ind[i + 1])[idx]] = 0
    D1.eliminate_zeros()
    return D1


def _smart_expand(nnm, K, NH=3):
    stage0 = nnm.copy()
    S = [stage0]
    running = stage0
    for i in range(1, NH + 1):
        stage = running.dot(stage0)
        running = stage
        stage = stage.tolil()
        for j in range(i):
            stage[S[j].nonzero()] = 0
        stage = stage.tocsr()
        S.append(stage)

    for i in range(len(S)):
        s = _sparse_knn_ks(S[i], K).tocsr()
        a, c = np.unique(s.nonzero()[0], return_counts=True)
        numnz = np.zeros(s.shape[0], dtype="int32")
        numnz[a] = c
        K = K - numnz
        K[K < 0] = 0
        S[i] = s
    res = S[0]
    for i in range(1, len(S)):
        res = res + S[i]
    return res

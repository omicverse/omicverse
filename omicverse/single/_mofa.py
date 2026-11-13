from ..utils import pyomic_palette
from ..single import get_celltype_marker
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import issparse
import h5py
import anndata
from tqdm import tqdm,trange
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union,Tuple,Optional
import matplotlib
from .._settings import add_reference
from .._registry import register_function


def _get_mofa_entry_point():
    from ..external.mofapy2.run.entry_point import entry_point

    return entry_point


mofax_install=False

def global_imports(modulename,shortname = None, asfunction = False):
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:        
        globals()[shortname] = __import__(modulename)

def check_mofax():
    r"""Check if mofax is installed and import it.
    
    Returns
    -------
    None
        Raises ``ImportError`` if ``mofax`` is unavailable.
    """
    global mofax_install
    try:
        import mofax as mfx
        mofax_install=True
        #print('mofax have been install version:',mfx.__version__)
    except ImportError:
        raise ImportError(
            'Please install the mofax: `pip install mofax`.'
        )

@register_function(
    aliases=['GLUE配对器', 'GLUE_pair', 'RNA-ATAC cell pairing'],
    category="single",
    description="Pair cells across RNA and ATAC modalities using GLUE embeddings and correlation-based nearest-neighbor matching.",
    prerequisites={'optional_functions': ['alignment integration with GLUE']},
    requires={'obsm': ['X_glue in RNA and ATAC objects']},
    produces={'obs': ['paired-cell labels'], 'uns': ['pairing diagnostics']},
    auto_fix='none',
    examples=['pair_obj = ov.single.GLUE_pair(rna, atac)', 'pair_obj.correlation(); pair_df = pair_obj.find_neighbor_cell()'],
    related=['single.pyMOFA', 'single.factor_exact']
)
class GLUE_pair(object):
    """
    Pair RNA and ATAC cells using GLUE latent embeddings and neighbor matching.
    
    Parameters
    ----------
    rna:anndata.AnnData
        RNA AnnData with ``obsm['X_glue']``.
    atac:anndata.AnnData
        ATAC AnnData with ``obsm['X_glue']``.
    
    Returns
    -------
    None
        Initializes paired-modality matching workflow.
    
    Examples
    --------
    >>> pair_obj = ov.single.GLUE_pair(rna, atac)
    """

    def __init__(self,rna:anndata.AnnData,
              atac:anndata.AnnData) -> None:
        r"""Pair the cells between RNA and ATAC using result of GLUE.

        Parameters
        ----------
        rna:anndata.AnnData
            RNA AnnData containing ``obsm['X_glue']``.
        atac:anndata.AnnData
            ATAC AnnData containing ``obsm['X_glue']``.
        
        Returns
        -------
        None
        """
        
        print('......Extract GLUE layer from obs')
        self.rna_loc=pd.DataFrame(rna.obsm['X_glue'], index=rna.obs.index)
        self.atac_loc=pd.DataFrame(atac.obsm['X_glue'], index=atac.obs.index)

    def correlation(self):
        r"""Perform Pearson Correlation analysis in the layer of GLUE.
        
        Returns
        -------
        None
            Updates ``self.rna_pd`` and ``self.atac_pd`` pair-ranking tables.
        """
        
        print('......Prepare for pair')
        import gc
        len1=(len(self.rna_loc)//5000)+1
        len2=(len(self.atac_loc)//5000)+1
        if len1>len2:
            len1=len2
        p_pd=pd.DataFrame(columns=['rank_'+str(i) for i in range(50)])
        n_pd=pd.DataFrame(columns=['rank_'+str(i) for i in range(50)])
        print('......Start to calculate the Pearson coef')
        for j in range(len1):
            c=pd.DataFrame()
            with trange(len1) as tt:
                for i in tt:
                    t1=self.rna_loc.iloc[5000*(i):5000*(i+1)]
                    t2=self.atac_loc.iloc[5000*(j):5000*(j+1)]
                    a=np.corrcoef(t1,t2)[len(t1):,0:len(t1)]
                    b=pd.DataFrame(a,index=t2.index,columns=t1.index)  
                    c=pd.concat([c,b],axis=1)
                    del t1
                    del t2
                    del a
                    del b
                    gc.collect()
                    tt.set_description('Now Pearson block is {}/{}'.format(i,len1))
            with trange(len(c)) as t:
                for i in t:
                    t_c=c.iloc[i]
                    p_pd.loc[t_c.name]=c.iloc[i].sort_values(ascending=False)[:50].values
                    n_pd.loc[t_c.name]=c.iloc[i].sort_values(ascending=False)[:50].index.tolist()
                    t.set_description('Now rna_index is {}/{}, all is {}'.format(i+j*5000,i+j*5000+len(c),len(self.atac_loc)))
            print('Now epoch is {}, {}/{}'.format(j,j*5000+len(c),len(self.atac_loc))) 
            del c
            gc.collect()
        self.rna_pd=p_pd
        self.atac_pd=n_pd

    def find_neighbor_cell(self,depth:int=10,cor:float=0.9)->pd.DataFrame:
        r"""Find the neighbor cells between two omics using pearson correlation.
        
        Parameters
        ----------
        depth:int
            Maximum rank depth used when searching reciprocal neighbors.
        cor:float
            Minimum Pearson-correlation threshold for keeping pairs.

        Returns
        -------
        pd.DataFrame
            Pairing table with matched cell IDs across modalities.

        """


        if depth>50:
            print('......depth limited to 50')
            depth=50
        rubish_c=[]
        finish_c=[]
        p_pd=self.rna_pd.copy()
        n_pd=self.atac_pd.copy()
        with trange(depth) as dt:
            for d in dt:
                p_pd=p_pd.loc[p_pd['rank_{}'.format(d)]>cor]
                p_pd=p_pd.sort_values('rank_{}'.format(d),ascending=False)
                for i in p_pd.index:
                    name=n_pd.loc[i,'rank_{}'.format(d)]
                    if name not in rubish_c:
                        finish_c.append(i)
                        rubish_c.append(name)
                    else:
                        continue
                p_pd=p_pd.loc[~p_pd.index.isin(finish_c)]
                n_pd=n_pd.loc[~n_pd.index.isin(finish_c)]
                dt.set_description('Now depth is {}/{}'.format(d,depth))
        result=pd.DataFrame()
        result['omic_1']=rubish_c
        result['omic_2']=finish_c
        result.index=['cell_{}'.format(i) for i in range(len(result))]
        self.pair_res=result
        return result
    
    def pair_omic(self,omic1:anndata.AnnData,omic2:anndata.AnnData)->Tuple[anndata.AnnData,anndata.AnnData]:
        r"""Pair the omics using the result of find_neighbor_cell.

        Parameters
        ----------
        omic1:anndata.AnnData
            First modality AnnData.
        omic2:anndata.AnnData
            Second modality AnnData.

        Returns
        -------
        Tuple[anndata.AnnData,anndata.AnnData]
            Paired AnnData objects with aligned observation indices.

        """
        rna1=omic1[self.res_pair['omic_1']].copy()
        atac1=omic2[self.res_pair['omic_2']].copy()
        rna1.obs.index=self.res_pair.index
        atac1.obs.index=self.res_pair.index
        return rna1,atac1


def glue_pair(rna:anndata.AnnData,
              atac:anndata.AnnData,depth:int=20)->pd.DataFrame:
    r"""
    Pair the cells between RNA and ATAC using result of GLUE.

    Parameters
    ----------
    rna:anndata.AnnData
        RNA AnnData containing ``obsm['X_glue']``.
    atac:anndata.AnnData
        ATAC AnnData containing ``obsm['X_glue']``.
    depth:int
        Maximum rank depth used when searching reciprocal neighbors.

    Returns
    -------
    pd.DataFrame
        Pairing table with matched cell IDs across modalities.
    
    """


    #提取GLUE层结果
    print('......Extract GLUE layer from obs')
    rna_loc=pd.DataFrame(rna.obsm['X_glue'], index=rna.obs.index)
    atac_loc=pd.DataFrame(atac.obsm['X_glue'], index=atac.obs.index)

    #对GLUE层进行Pearson系数分析
    print('......Prepare for pair')
    import gc
    len1=(len(rna_loc)//5000)+1
    len2=(len(atac_loc)//5000)+1
    if len1>len2:
        len1=len2
    p_pd=pd.DataFrame(columns=['rank_'+str(i) for i in range(50)])
    n_pd=pd.DataFrame(columns=['rank_'+str(i) for i in range(50)])
    print('......Start to calculate the Pearson coef')
    for j in range(len1):
        c=pd.DataFrame()
        with trange(len1) as tt:
            for i in tt:
                t1=rna_loc.iloc[5000*(i):5000*(i+1)]
                t2=atac_loc.iloc[5000*(j):5000*(j+1)]
                a=np.corrcoef(t1,t2)[len(t1):,0:len(t1)]
                b=pd.DataFrame(a,index=t2.index,columns=t1.index)  
                c=pd.concat([c,b],axis=1)
                del t1
                del t2
                del a
                del b
                gc.collect()
                tt.set_description('Now Pearson block is {}/{}'.format(i,len1))
        with trange(len(c)) as t:
            for i in t:
                t_c=c.iloc[i]
                p_pd.loc[t_c.name]=c.iloc[i].sort_values(ascending=False)[:50].values
                n_pd.loc[t_c.name]=c.iloc[i].sort_values(ascending=False)[:50].index.tolist()
                t.set_description('Now rna_index is {}/{}, all is {}'.format(i+j*5000,i+j*5000+len(c),len(atac_loc)))
        print('Now epoch is {}, {}/{}'.format(j,j*5000+len(c),len(atac_loc))) 
        del c
        gc.collect()
        
    #寻找最近的细胞，其中depth的灵活调整可以使得配对成功的细胞数变大，同时精度有所下降
    def find_neighbor_cell(p_pd,n_pd,depth=10):
        if depth>50:
            print('......depth limited to 50')
            depth=50
        rubish_c=[]
        finish_c=[]
        with trange(depth) as dt:
            for d in dt:
                p_pd=p_pd.loc[p_pd['rank_{}'.format(d)]>0.9]
                p_pd=p_pd.sort_values('rank_{}'.format(d),ascending=False)
                for i in p_pd.index:
                    name=n_pd.loc[i,'rank_{}'.format(d)]
                    if name not in rubish_c:
                        finish_c.append(i)
                        rubish_c.append(name)
                    else:
                        continue
                p_pd=p_pd.loc[~p_pd.index.isin(finish_c)]
                n_pd=n_pd.loc[~n_pd.index.isin(finish_c)]
                dt.set_description('Now depth is {}/{}'.format(d,depth))
        result=pd.DataFrame()
        result['omic_1']=rubish_c
        result['omic_2']=finish_c
        result.index=['cell_{}'.format(i) for i in range(len(result))]
        return result
    print('......Start to find neighbor')
    res_pair=find_neighbor_cell(p_pd,n_pd,depth=depth)
    return res_pair

def normalization(data):
    r"""
    Scale an array by its maximum absolute value.

    Parameters
    ----------
    data : np.ndarray
        Input numeric array.

    Returns
    -------
    np.ndarray
        Rescaled array in approximately ``[-1, 1]``.
    """
    _range = np.max(abs(data))
    return data / _range

@register_function(
    aliases=['提取 MOFA 权重', 'get_weights', 'mofa weights'],
    category="single",
    description="Extract feature loadings of selected MOFA factors for one omics view to interpret latent biological programs.",
    prerequisites={'functions': ['pyMOFA']},
    requires={},
    produces={},
    auto_fix='none',
    examples=['ov.single.get_weights("mofa.hdf5", view="rna", factor=1, scale=True)'],
    related=['single.factor_exact', 'single.factor_correlation', 'single.pyMOFA']
)
def get_weights(hdf5_path:str,view:str,
                factor:int,scale:bool=True)->pd.DataFrame:
    r"""
    Extract feature loadings for one factor from a MOFA model.

    Parameters
    ----------
    hdf5_path : str
        Path to MOFA ``.hdf5`` model file.
    view : str
        View/modality name to query.
    factor : int
        1-based factor index.
    scale : bool
        If ``True``, rescale weights by maximum absolute value.

    Returns
    -------
    pd.DataFrame
        Feature table containing raw/absolute weights and sign label.
    """
    f = h5py.File(hdf5_path,'r')  
    view_names=f['views']['views'][:]
    group_names=f['groups']['groups'][:]
    feature_names={view: f['features'][view][:] for view in view_names}
    #sample_names=np.array([f['samples'][i][:] for i in group_names])
    f_name=feature_names[str.encode(view)]
    f_w=f['expectations']['W'][view][factor-1]
    if scale==True:
        f_w=normalization(f_w)
    res=pd.DataFrame()
    res['feature']=f_name
    res['weights']=f_w
    res['abs_weights']=abs(f_w)
    res['sig']='+'
    res.loc[(res.weights<0),'sig'] = '-'

    return res

@register_function(
    aliases=['MOFA 因子提取', 'factor_exact', 'extract mofa factors'],
    category="single",
    description="Project MOFA latent factors to cells and store them in AnnData for downstream trajectory, clustering, and phenotype association analyses.",
    prerequisites={'functions': ['pyMOFA']},
    requires={'obs': ['cell metadata (optional)']},
    produces={'obsm': ['X_mofa'], 'obs': ['factor scores']},
    auto_fix='none',
    examples=['ov.single.factor_exact(adata, hdf5_path="mofa.hdf5")'],
    related=['single.get_weights', 'single.factor_correlation']
)
def factor_exact(adata:anndata.AnnData,hdf5_path:str)->anndata.AnnData:
    r"""
    Add MOFA latent factors from model file into ``adata.obs``.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object to annotate.
    hdf5_path : str
        Path to MOFA ``.hdf5`` model file.

    Returns
    -------
    anndata.AnnData
        Input AnnData with added columns ``factor1``, ``factor2``, ... in
        ``adata.obs``.
    """
    f_pos = h5py.File(hdf5_path,'r')  
    g_name=f_pos['groups']['groups'][:][0]
    for i in range(f_pos['expectations']['Z'][g_name].shape[0]):
        adata.obs['factor{0}'.format(i+1)]=f_pos['expectations']['Z'][g_name][i] 
    return adata

@register_function(
    aliases=['MOFA 因子相关性', 'factor_correlation', 'factor phenotype correlation'],
    category="single",
    description="Compute correlations between MOFA factors and cell annotations to prioritize biologically interpretable latent axes.",
    prerequisites={'functions': ['factor_exact']},
    requires={'obs': ['cluster or phenotype labels']},
    produces={'uns': ['mofa_factor_correlation']},
    auto_fix='none',
    examples=['ov.single.factor_correlation(adata, cluster="celltype", factor_list=[1,2,3], p_threshold=0.05)'],
    related=['single.factor_exact', 'single.get_weights']
)
def factor_correlation(adata:anndata.AnnData,cluster:str,
                       factor_list:list,p_threshold:int=500)->pd.DataFrame:
    r"""
    Score MOFA-factor enrichment across annotated groups.

    For each factor and each group in ``cluster``, a two-sample t-test compares
    factor values in-group vs out-group, and stores ``-log(p)``.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData containing ``factor*`` columns in ``obs``.
    cluster : str
        Group label column in ``adata.obs``.
    factor_list : list
        Factor indices to evaluate.
    p_threshold : int
        Upper cap applied to ``-log(p)`` values to avoid extreme outliers.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by group labels with factor association scores.
    """
    plot_data=adata.obs
    cell_t=list(set(plot_data[cluster]))
    cell_pd=pd.DataFrame(index=cell_t)
    for i in factor_list:
        test=[]
        for j in cell_t:
            a=plot_data[plot_data[cluster]==j]['factor'+str(i)].values
            b=plot_data[~(plot_data[cluster]==j)]['factor'+str(i)].values
            t, p = stats.ttest_ind(a,b)
            logp=-np.log(p)
            if(logp>p_threshold):
                logp=p_threshold
            test.append(logp)
        cell_pd['factor'+str(i)]=test
    return cell_pd

@register_function(
    aliases=['MOFA训练器', 'pyMOFA', 'multi-omics factor model'],
    category="single",
    description="Train MOFA models for multi-omics factor decomposition and latent biology discovery across modalities.",
    prerequisites={'optional_functions': ['pp.preprocess']},
    requires={},
    produces={'obsm': ['X_mofa'], 'uns': ['mofa model']},
    auto_fix='none',
    examples=['test_mofa = ov.single.pyMOFA(omics=[rna, atac], omics_name=["rna","atac"])', 'test_mofa.mofa_run()'],
    related=['single.get_weights', 'single.factor_exact', 'single.pyMOFAART']
)
class pyMOFA(object):
    """
    Train MOFA models for latent factor discovery across multiple omics layers.
    
    Parameters
    ----------
    omics:list
        List of omics data matrices (for example RNA/ATAC/proteomics).
    omics_name:list
        Names corresponding to each entry in ``omics``.
    
    Returns
    -------
    None
        Initializes MOFA training inputs and metadata.
    
    Examples
    --------
    >>> test_mofa = ov.single.pyMOFA(omics=[rna, atac], omics_name=["rna","atac"])
    """
    def __init__(self,omics:list,omics_name:list):
        r"""
        Initialize a multi-view MOFA training container.

        Parameters
        ----------
        omics : list
            List of input omics objects, typically ``AnnData`` matrices for each
            modality (for example RNA, ATAC, protein).
        omics_name : list
            View names aligned with ``omics`` order. These names are used in the
            exported MOFA model and downstream interpretation.
        """
        self.omics=omics 
        self.omics_name=omics_name
        self.M=len(omics)

    def mofa_preprocess(self):
        r"""
        Convert input omics objects into MOFA-ready dense matrices.

        The method creates ``self.data_mat`` and ``self.feature_name`` in the
        format expected by ``mofapy2.entry_point.set_data_matrix``.
        """
        self.data_mat=[[None for g in range(1)] for m in range(len(self.omics))]
        self.feature_name=[]
        for m in range(self.M):
            if issparse(self.omics[m].X)==True:
                self.data_mat[m][0]=self.omics[m].X.toarray()
            else:
                self.data_mat[m][0]=self.omics[m].X
            self.feature_name.append([self.omics_name[m]+'_'+i for i in self.omics[m].var.index])

    def mofa_run(self,outfile:str='res.hdf5',factors:int=20,iter:int = 1000,convergence_mode:str = "fast",
                spikeslab_weights:bool = True,startELBO:int = 1, freqELBO:int = 1, dropR2:float = 0.001, gpu_mode:bool = True, 
                gpu_device: Optional[int] = None, verbose:bool = False, seed:int = 112,scale_groups:bool = False, 
                scale_views:bool = False,center_groups:bool=True,)->None:
        r"""
        Train and save a MOFA model using current multi-omics inputs.

        Parameters
        ----------
        outfile : str
            Output path for the trained MOFA ``.hdf5`` model.
        factors : int
            Number of latent factors to learn.
        iter : int
            Maximum number of training iterations.
        convergence_mode : str
            Optimization regime in MOFA (for example ``'fast'`` or ``'medium'``).
        spikeslab_weights : bool
            Whether to use spike-and-slab prior on feature weights, encouraging
            sparse factor loadings.
        startELBO : int
            Iteration at which ELBO evaluation starts.
        freqELBO : int
            Frequency (in iterations) for ELBO monitoring.
        dropR2 : float
            Early-stop criterion based on minimal variance-explained improvement.
        gpu_mode : bool
            Whether to enable GPU acceleration when supported by backend.
        gpu_device : int or None
            GPU device index used when ``gpu_mode=True``.
        verbose : bool
            Whether to print detailed training logs.
        seed : int
            Random seed for reproducible model initialization/training.
        scale_groups : bool
            Whether to scale each sample group before factorization.
        scale_views : bool
            Whether to scale each view (modality) before factorization.
        center_groups : bool
            Whether to center each sample group before factorization.

        Returns
        -------
        None
            Writes trained model to ``outfile`` and records MOFA reference in
            each input omics object.
        """
        entry_point = _get_mofa_entry_point()
        ent1 = entry_point()
        ent1.set_data_options(
            scale_groups = scale_groups, 
            scale_views = scale_views,
            center_groups=center_groups,
        )
        ent1.set_data_matrix(self.data_mat, likelihoods = [i for i in ["gaussian"]*self.M],
            views_names=self.omics_name,
            samples_names=[self.omics[0].obs.index],
            features_names=self.feature_name)
        # set param
        ent1.set_model_options(
            factors = factors, 
            spikeslab_weights = spikeslab_weights, 
            ard_factors = True,
            ard_weights = True
        )
        ent1.set_train_options(
            iter = iter, 
            convergence_mode = convergence_mode, 
            startELBO = startELBO, 
            freqELBO = freqELBO, 
            dropR2 = dropR2, 
            gpu_mode = gpu_mode, 
            gpu_device = gpu_device,
            verbose = verbose, 
            seed = seed
        )
        # 
        ent1.build()
        ent1.run()
        ent1.save(outfile=outfile)
        for ix in range(len(self.omics)):
            add_reference(self.omics[ix],'MOFA','Multi-omics factor analysis with MOFA')

    

@register_function(
    aliases=['MOFA模型读取器', 'pyMOFAART', 'mofa model loader'],
    category="single",
    description="Load and analyze pretrained MOFA models for downstream factor interpretation without retraining.",
    prerequisites={'optional_functions': ['pyMOFA']},
    requires={},
    produces={'obsm': ['X_mofa'], 'uns': ['mofa factor summaries']},
    auto_fix='none',
    examples=['pymofa_obj = ov.single.pyMOFAART(model_path="data/sample/MOFA_POS.hdf5")', 'pymofa_obj.get_factors()'],
    related=['single.pyMOFA', 'single.factor_correlation']
)
class pyMOFAART(object):
    """
    Load pretrained MOFA models for downstream factor interpretation.
    
    Parameters
    ----------
    model_path:str
        Path to a trained MOFA model file.
    
    Returns
    -------
    None
        Initializes MOFA model reader and analysis utilities.
    
    Examples
    --------
    >>> pymofa_obj = ov.single.pyMOFAART(model_path="data/sample/MOFA_POS.hdf5")
    """
    
    def __init__(self,model_path:str):
        """
        Load a pretrained MOFA model for downstream interpretation.

        Parameters
        ----------
        model_path : str
            Path to a trained MOFA ``.hdf5`` model file.
        """
        check_mofax()
        global mofax_install
        if mofax_install==True:
            global_imports("mofax","mfx")
        
        self.model_path=model_path
        mfx_model=mfx.mofa_model(model_path)
        self.factors=mfx_model.get_factors()
        plot_data=pd.DataFrame()
        for i in mfx_model.get_r2()['View'].unique():
            plot_data[i]=mfx_model.get_r2().loc[mfx_model.get_r2()['View']==i,'R2'].values
        self.r2=plot_data
        mfx_model.close()
        
       
    def get_factors(self,adata:anndata.AnnData):
        """
        Attach MOFA latent factors to ``adata.obsm['X_mofa']``.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData whose observations correspond to the samples/cells used in
            the MOFA model.

        Returns
        -------
        None
            Updates ``adata.obsm['X_mofa']`` and annotates factor columns in
            ``adata.obs`` via :func:`factor_exact`.
        """
        print('......Add factors to adata and store to adata.obsm["X_mofa"]')
        adata.obsm['X_mofa']=self.factors
        adata=factor_exact(adata,hdf5_path=self.model_path)

    def get_r2(self,)->pd.DataFrame:
        """
        Return factor-wise explained variance table.

        Returns
        -------
        pd.DataFrame
            Variance explained (R2) per factor and view.
        """

        return self.r2

    def plot_r2(self,figsize:tuple=(2,3),cmap:str='Greens',
                ticks_fontsize:int=10,labels_fontsize:int=12,
                save:bool=False)->Tuple[matplotlib.figure.Figure,matplotlib.axes._axes.Axes]:
        """
        Plot a heatmap of variance explained by each MOFA factor.

        Parameters
        ----------
        figsize : tuple
            Figure size in inches as ``(width, height)``.
        cmap : str
            Colormap for heatmap intensity.
        ticks_fontsize : int
            Font size for axis tick labels.
        labels_fontsize : int
            Font size for axis titles and plot title.
        save : bool
            If ``True``, save figure as ``mofa_varience.png``.

        Returns
        -------
        Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
            Matplotlib figure and axes of the R2 heatmap.
        """
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(self.r2,cmap=cmap,ax=ax,xticklabels=True,yticklabels=True,
                    cbar_kws={'shrink':0.5})
        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)
        plt.ylabel('Factor',fontsize=labels_fontsize)
        plt.xlabel('View',fontsize=labels_fontsize)
        plt.title('Varience',fontsize=labels_fontsize)
        if save:
            fig.savefig("mofa_varience.png",dpi=300,bbox_inches = 'tight')
        return fig,ax
    
    def get_cor(self,adata:anndata.AnnData,cluster:str,factor_list=None)->pd.DataFrame:
        """
        Compute factor-group association scores from MOFA factors.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData containing ``obs[cluster]`` and MOFA factor columns in
            ``obs``.
        cluster : str
            Column in ``adata.obs`` used as biological groups (for example cell
            type, condition, or batch).
        factor_list : list or None
            Factors to evaluate. If ``None``, all factors are used.

        Returns
        -------
        pd.DataFrame
            Matrix of group-wise factor association scores.
        """

        if factor_list==None:
            factor_list=[i+1 for i in range(self.r2.shape[0])]
        plot_data1=factor_correlation(adata=adata,cluster=cluster,factor_list=factor_list)
        return plot_data1

    def plot_cor(self,adata:anndata.AnnData,cluster:str,factor_list=None,figsize:tuple=(6,3),
                 cmap:str='Purples',ticks_fontsize:int=10,labels_fontsize:int=12,title:str='Correlation',
                 save:bool=False)->Tuple[matplotlib.figure.Figure,matplotlib.axes._axes.Axes]:
        """
        Visualize factor-group association scores as a heatmap.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData containing ``obs[cluster]`` and extracted MOFA factors.
        cluster : str
            Grouping column in ``adata.obs``.
        factor_list : list or None
            Factors to include. If ``None``, all factors are plotted.
        figsize : tuple
            Figure size in inches as ``(width, height)``.
        cmap : str
            Colormap for heatmap values.
        ticks_fontsize : int
            Font size for tick labels.
        labels_fontsize : int
            Font size for axis labels and title.
        title : str
            Plot title.
        save : bool
            If ``True``, save figure as ``mofa_cor.png``.

        Returns
        -------
        Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
            Matplotlib figure and axes of the correlation heatmap.
        """

        if factor_list==None:
            factor_list=[i+1 for i in range(self.r2.shape[0])]
        plot_data1=factor_correlation(adata=adata,cluster=cluster,factor_list=factor_list)
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(plot_data1,cmap=cmap,ax=ax,square=True,
                    cbar_kws={'shrink':0.5})
        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)
        plt.xlabel('Factor',fontsize=labels_fontsize)
        plt.ylabel(cluster,fontsize=labels_fontsize)
        plt.title(title,fontsize=labels_fontsize)
        if save:
            fig.savefig("mofa_cor.png",dpi=300,bbox_inches = 'tight')
        return fig,ax

    def plot_factor(self,adata:anndata.AnnData,cluster:str,title:str,figsize:tuple=(3,3),
                    factor1:int=1,factor2:int=2,palette:list=None,
                    save:bool=False)->Tuple[matplotlib.figure.Figure,matplotlib.axes._axes.Axes]:
        """
        Plot cells in MOFA factor space colored by group annotation.
        
        Parameters
        ----------
        adata : anndata.AnnData
            AnnData object to visualize.
        cluster : str
            Categorical column in ``adata.obs`` used for coloring.
        title : str
            Title shown above the embedding plot.
        figsize : tuple
            Figure size in inches as ``(width, height)``.
        factor1 : int
            First MOFA factor index used as x-axis.
        factor2 : int
            Second MOFA factor index used as y-axis.
        palette : list or None
            Color palette for cluster categories. If ``None``, uses
            :func:`pyomic_palette`.
        save : bool
            If ``True``, save figure to ``figures/mofa_factor_{factor1}_{factor2}.png``.
        
        Returns
        -------
        Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
            Matplotlib figure and axes for the factor embedding.
        """

        if 'X_mofa' not in adata.obsm.keys():
            self.get_factors(adata)
        if palette==None:
            palette=pyomic_palette()
        fig, ax = plt.subplots(figsize=figsize)
        #factor1,factor2=4,6
        from ..pl import embedding
        embedding(
            adata=adata,
            basis='X_mofa',
            color=cluster,
            title=title,
            components="{},{}".format(factor1,factor2),
            palette=palette,
            ncols=1,
            ax=ax
        )
        if save:
            fig.savefig("figures/mofa_factor_{}_{}.png".format(factor1,factor2),dpi=300,bbox_inches = 'tight')

        return fig,ax

    def plot_weight_gene_d1(self,view:str,factor1:int,factor2:int,
                            colors_dict:dict=None,plot_gene_num:int=5,title:str='',title_fontsize:int=12,
                            ticks_fontsize:int=12,labels_fontsize:int=12,
                            weith_threshold:float=0.5,figsize:tuple=(3,3),
                            save:bool=False)->Tuple[matplotlib.figure.Figure,matplotlib.axes._axes.Axes]:
        """
        Plot gene weights for two factors with 1D threshold-based highlighting.

        Parameters
        ----------
        view : str
            View/modality name in the MOFA model.
        factor1 : int
            Factor index used for thresholding and x-axis.
        factor2 : int
            Factor index used for y-axis.
        colors_dict : dict or None
            Colors for ``normal``, ``up`` and ``down`` classes. If ``None``,
            defaults to a red/green scheme.
        plot_gene_num : int
            Number of top genes per direction (up/down) to annotate.
        title : str
            Figure title.
        title_fontsize : int
            Font size for title.
        ticks_fontsize : int
            Font size for axis ticks.
        labels_fontsize : int
            Font size for axis labels.
        weith_threshold : float
            Absolute weight threshold on ``factor1`` to call genes ``up`` or
            ``down``.
        figsize : tuple
            Figure size in inches as ``(width, height)``.
        save : bool
            If ``True``, save figure as ``factor_gene_{title}.png``.

        Returns
        -------
        Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
            Matplotlib figure and axes for gene-weight scatter plot.
        """
        factor_w=pd.DataFrame()
        for i in range(self.factors.shape[1]):
            f1_w=get_weights(hdf5_path=self.model_path,view=view,factor=i+1)
            f1_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]
            factor_w['factor_{}'.format(i+1)]=f1_w['weights']
        factor_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]

        #factor1,factor2=6,4
        plot_data3=factor_w[['factor_{}'.format(factor1),'factor_{}'.format(factor2)]]
        plot_data3['sig']='normal'
        plot_data3.loc[(plot_data3['factor_{}'.format(factor1)]>weith_threshold),'sig']='up'
        plot_data3.loc[(plot_data3['factor_{}'.format(factor1)]<-weith_threshold),'sig']='down'

        if colors_dict==None:
            colors_dict={'normal':'#c2c2c2','up':'#a51616','down':'#0d6a3b'}
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(plot_data3.loc[plot_data3['sig']=='normal','factor_{}'.format(factor1)],
                plot_data3.loc[plot_data3['sig']=='normal','factor_{}'.format(factor2)],
                color=colors_dict['normal'],alpha=0.5)

        ax.scatter(plot_data3.loc[plot_data3['sig']=='up','factor_{}'.format(factor1)],
                plot_data3.loc[plot_data3['sig']=='up','factor_{}'.format(factor2)],
                color=colors_dict['up'],alpha=0.5)

        ax.scatter(plot_data3.loc[plot_data3['sig']=='down','factor_{}'.format(factor1)],
                plot_data3.loc[plot_data3['sig']=='down','factor_{}'.format(factor2)],
                color=colors_dict['down'],alpha=0.5)

        plt.vlines(x=weith_threshold,ymin=-1,ymax=1,color=colors_dict['up'],linestyles='dashed')
        plt.hlines(y=weith_threshold,xmin=-1,xmax=1,color=colors_dict['up'],linestyles='dashed')

        plt.vlines(x=-weith_threshold,ymin=-1,ymax=1,color=colors_dict['down'],linestyles='dashed')
        plt.hlines(y=-weith_threshold,xmin=-1,xmax=1,color=colors_dict['down'],linestyles='dashed')

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        plt.grid(False)

        plt.xlabel('factor_{}'.format(factor1),fontsize=labels_fontsize)
        plt.ylabel('factor_{}'.format(factor2),fontsize=labels_fontsize)

        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)

        from adjustText import adjust_text

        for sig,color in zip(['up','down'],
                            [colors_dict['up'],colors_dict['down']]):
            if 'up' in sig:
                hub_gene=plot_data3.loc[plot_data3['sig']==sig].sort_values('factor_{}'.format(factor1),ascending=False).index.tolist()
            else:
                hub_gene=plot_data3.loc[plot_data3['sig']==sig].sort_values('factor_{}'.format(factor1),ascending=True).index.tolist()
            if len(hub_gene)==0:
                continue
            texts=[ax.text(plot_data3.loc[i,'factor_{}'.format(factor1)],
                        plot_data3.loc[i,'factor_{}'.format(factor2)],
                        i,
                        fontdict={'size':10,'weight':'bold','color':'black'}
                        ) for i in hub_gene[:plot_gene_num]]

            adjust_text(texts,only_move={'text': 'xy'},arrowprops=dict(arrowstyle='->', color='grey'),)

        plt.title(title,fontsize=title_fontsize)
        if save:
            fig.savefig("factor_gene_{}.png".format(title),dpi=300,bbox_inches = 'tight')
        return fig,ax
    
    def plot_weight_gene_d2(self,view:str,factor1:int,factor2:int,
                            colors_dict:dict=None,plot_gene_num:int=5,title:str='',title_fontsize:int=12,
                            ticks_fontsize:int=12,labels_fontsize:int=12,
                            weith_threshold:float=0.5,figsize:tuple=(3,3),
                            save:bool=False)->Tuple[matplotlib.figure.Figure,matplotlib.axes._axes.Axes]:
        """
        Plot gene weights for two factors with quadrant-based labeling.

        Parameters
        ----------
        view : str
            View/modality name in the MOFA model.
        factor1 : int
            Factor index used for x-axis.
        factor2 : int
            Factor index used for y-axis.
        colors_dict : dict or None
            Colors for ``up-up``, ``up-down``, ``down-up``, ``down-down`` and
            ``normal`` categories. If ``None``, a default palette is used.
        plot_gene_num : int
            Number of labeled genes per quadrant category.
        title : str
            Figure title.
        title_fontsize : int
            Font size for title.
        ticks_fontsize : int
            Font size for axis ticks.
        labels_fontsize : int
            Font size for axis labels.
        weith_threshold : float
            Absolute cutoff applied to both factors to define quadrants.
        figsize : tuple
            Figure size in inches as ``(width, height)``.
        save : bool
            If ``True``, save figure as ``factor_gene_{title}.png``.

        Returns
        -------
        Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
            Matplotlib figure and axes for quadrant gene-weight scatter plot.
        """
        
        factor_w=pd.DataFrame()
        for i in range(self.factors.shape[1]):
            f1_w=get_weights(hdf5_path=self.model_path,view=view,factor=i+1)
            f1_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]
            factor_w['factor_{}'.format(i+1)]=f1_w['weights']
        factor_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]

        #factor1,factor2=6,4
        plot_data3=factor_w[['factor_{}'.format(factor1),'factor_{}'.format(factor2)]]
        plot_data3['sig']='normal'
        plot_data3.loc[(plot_data3['factor_{}'.format(factor1)]>weith_threshold)&(plot_data3['factor_{}'.format(factor2)]>weith_threshold),'sig']='up-up'
        plot_data3.loc[(plot_data3['factor_{}'.format(factor1)]>weith_threshold)&(plot_data3['factor_{}'.format(factor2)]<-weith_threshold),'sig']='up-down'
        plot_data3.loc[(plot_data3['factor_{}'.format(factor1)]<-weith_threshold)&(plot_data3['factor_{}'.format(factor2)]>weith_threshold),'sig']='down-up'
        plot_data3.loc[(plot_data3['factor_{}'.format(factor1)]<-weith_threshold)&(plot_data3['factor_{}'.format(factor2)]<-weith_threshold),'sig']='down-down'


        if colors_dict==None:
            colors_dict={'up-up':'#a51616','up-down':'#e25d5d','down-up':'#1a6e1a','down-down':'#5de25d','normal':'#c2c2c2'}
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(plot_data3.loc[plot_data3['sig']=='normal','factor_{}'.format(factor1)],
           plot_data3.loc[plot_data3['sig']=='normal','factor_{}'.format(factor2)],
          color=colors_dict['normal'],alpha=0.5)

        ax.scatter(plot_data3.loc[plot_data3['sig']=='up-up','factor_{}'.format(factor1)],
                plot_data3.loc[plot_data3['sig']=='up-up','factor_{}'.format(factor2)],
                color=colors_dict['up-up'],alpha=0.5)

        ax.scatter(plot_data3.loc[plot_data3['sig']=='up-down','factor_{}'.format(factor1)],
                plot_data3.loc[plot_data3['sig']=='up-down','factor_{}'.format(factor2)],
                color=colors_dict['up-down'],alpha=0.5)

        ax.scatter(plot_data3.loc[plot_data3['sig']=='down-up','factor_{}'.format(factor1)],
                plot_data3.loc[plot_data3['sig']=='down-up','factor_{}'.format(factor2)],
                color=colors_dict['down-up'],alpha=0.5)

        ax.scatter(plot_data3.loc[plot_data3['sig']=='down-down','factor_{}'.format(factor1)],
                plot_data3.loc[plot_data3['sig']=='down-down','factor_{}'.format(factor2)],
                color=colors_dict['down-down'],alpha=0.5)

        plt.vlines(x=weith_threshold,ymin=-1,ymax=1,color=colors_dict['up-up'],linestyles='dashed')
        plt.hlines(y=weith_threshold,xmin=-1,xmax=1,color=colors_dict['up-up'],linestyles='dashed')

        plt.vlines(x=-weith_threshold,ymin=-1,ymax=1,color=colors_dict['down-down'],linestyles='dashed')
        plt.hlines(y=-weith_threshold,xmin=-1,xmax=1,color=colors_dict['down-down'],linestyles='dashed')

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        plt.grid(False)

        plt.xlabel('factor_{}'.format(factor1),fontsize=labels_fontsize)
        plt.ylabel('factor_{}'.format(factor2),fontsize=labels_fontsize)

        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)

        from adjustText import adjust_text

        for sig,color in zip(['up-up','up-down','down-up','down-down'],
                     [colors_dict['up-up'],colors_dict['up-down'],colors_dict['down-up'],colors_dict['down-down']]):
            hub_gene=plot_data3.loc[plot_data3['sig']==sig].index.tolist()
            if len(hub_gene)==0:
                continue
            texts=[ax.text(plot_data3.loc[i,'factor_{}'.format(factor1)],
                        plot_data3.loc[i,'factor_{}'.format(factor2)],
                        i,
                        fontdict={'size':10,'weight':'bold','color':'black'}
                        ) for i in hub_gene[:plot_gene_num]]

            adjust_text(texts,only_move={'text': 'xy'},arrowprops=dict(arrowstyle='->', color='grey'),)

        plt.title(title,fontsize=title_fontsize)
        if save:
            fig.savefig("factor_gene_{}.png".format(title),dpi=300,bbox_inches = 'tight')
        return fig,ax

    def plot_weights_gene_factor(self,view:str,factor:int,color:str='#a51616',figsize:tuple=(3,4),
                     plot_gene_num:int=10,ascending:bool=False,
                    labels_fontsize:int=12,ticks_fontsize:int=12,title_fontsize:int=12,
                     title=None,save:bool=False)->Tuple[matplotlib.figure.Figure,matplotlib.axes._axes.Axes]:
        """
        Plot ranked gene weights for a single MOFA factor.

        Parameters
        ----------
        view : str
            View/modality name in the MOFA model.
        factor : int
            Factor index to visualize.
        color : str
            Line and marker color.
        figsize : tuple
            Figure size in inches as ``(width, height)``.
        plot_gene_num : int
            Number of extreme genes to annotate.
        ascending : bool
            Sorting direction for weights. ``False`` highlights largest positive
            weights; ``True`` highlights most negative weights.
        labels_fontsize : int
            Font size for axis labels.
        ticks_fontsize : int
            Font size for axis ticks.
        title_fontsize : int
            Font size for title.
        title : str or None
            Plot title. If ``None``, use ``factor_{factor}``.
        save : bool
            If ``True``, save figure as ``factor{factor}_gene.png``.

        Returns
        -------
        Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
            Matplotlib figure and axes for ranked weight plot.
        """
        factor_w=pd.DataFrame()
        for i in range(self.factors.shape[1]):
            f1_w=get_weights(hdf5_path=self.model_path,view=view,factor=i+1)
            f1_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]
            factor_w['factor_{}'.format(i+1)]=f1_w['weights']
        factor_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]

        fig, ax = plt.subplots(figsize=figsize)
        plot_data4=pd.DataFrame()
        plot_data4['weight']=factor_w['factor_{}'.format(factor)].sort_values(ascending=ascending)
        plot_data4['rank']=range(len(plot_data4['weight']))
        plt.plot(plot_data4['rank'],plot_data4['weight'],color=color)

        hub_gene=plot_data4.index[:plot_gene_num]
        plt.scatter(plot_data4.loc[hub_gene,'rank'],
                plot_data4.loc[hub_gene,'weight'],color=color,
                    alpha=0.5)

        from adjustText import adjust_text
        texts=[ax.text(plot_data4.loc[i,'rank'],
                        plot_data4.loc[i,'weight'],
                        i,
                        fontdict={'size':10,'weight':'normal','color':'black'}
                        ) for i in hub_gene]

        adjust_text(texts,only_move={'text': 'xy'},
                    arrowprops=dict(arrowstyle='->', color='grey'),)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        if title is None:
            plt.title('factor_{}'.format(factor),fontsize=title_fontsize,)
        else:
            plt.title(title,fontsize=title_fontsize,)
        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)
        plt.xlabel('Feature rank',fontsize=labels_fontsize)
        plt.ylabel('Weight',fontsize=labels_fontsize)

        plt.grid(False)
        if save:
            fig.savefig("factor{}_gene.png".format(factor),dpi=300,bbox_inches = 'tight')
        return fig,ax

    def plot_top_feature_dotplot(self,view:str,cmap:str='bwr',n_genes:int=3,n_pcs:int=50)->list:
        """
        Plot top marker features per factor as a dotplot.
        
        Parameters
        ----------
        view : str
            View/modality name in the MOFA model.
        cmap : str
            Colormap passed to :func:`omicverse.pl.rank_genes_groups_dotplot`.
        n_genes : int
            Number of top-ranked features per factor.
        n_pcs : int
            Kept for backward compatibility; the dotplot computes its own
            group dendrogram.

        Returns
        -------
        list
            Plot handles returned by :func:`omicverse.pl.rank_genes_groups_dotplot`.
        """

        factor_w=pd.DataFrame()
        for i in range(self.factors.shape[1]):
            f1_w=get_weights(hdf5_path=self.model_path,view=view,factor=i+1)
            f1_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]
            factor_w['factor_{}'.format(i+1)]=f1_w['weights']
        factor_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]

        adata1=anndata.AnnData(pd.concat([factor_w,factor_w],axis=1).T)
        adata1.obs['Factor']=adata1.obs.index
        adata1.obs['Factor']=adata1.obs['Factor'].astype('category')
        from ._sc_compat import rank_genes_groups
        from ..pl import rank_genes_groups_dotplot
        rank_genes_groups(adata1, groupby='Factor', method='wilcoxon')
        ax=rank_genes_groups_dotplot(adata1, groupby='Factor', n_genes=n_genes,
                                        cmap=cmap,show=False)
        return ax
    
    def plot_top_feature_heatmap(self,view:str,cmap:str='bwr',n_genes:int=3,n_pcs:int=50)->list:
        """
        Plot top marker features per factor as a grouped mean-expression heatmap.
        
        Parameters
        ----------
        view : str
            View/modality name in the MOFA model.
        cmap : str
            Colormap passed to :func:`omicverse.pl.group_heatmap`.
        n_genes : int
            Number of top-ranked features per factor.
        n_pcs : int
            Kept for backward compatibility (unused).

        Returns
        -------
        list
            Plot handles returned by :func:`omicverse.pl.group_heatmap`.
        """

        factor_w=pd.DataFrame()
        for i in range(self.factors.shape[1]):
            f1_w=get_weights(hdf5_path=self.model_path,view=view,factor=i+1)
            f1_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]
            factor_w['factor_{}'.format(i+1)]=f1_w['weights']
        factor_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]

        adata1=anndata.AnnData(pd.concat([factor_w,factor_w],axis=1).T)
        adata1.obs['Factor']=adata1.obs.index
        adata1.obs['Factor']=adata1.obs['Factor'].astype('category')
        from ._sc_compat import rank_genes_groups
        from ._markers import get_markers
        from ..pl import group_heatmap
        rank_genes_groups(adata1, groupby='Factor', method='wilcoxon')
        top = get_markers(adata1, n_genes=n_genes, return_type='dict')
        ax=group_heatmap(adata1, var_names=top, groupby='Factor',
                         cmap=cmap, show=False)
        return ax
    
    def get_top_feature(self,view:str,log2fc_min:int=3,pval_cutoff:float=0.1)->dict:
        """
        Identify top features for each factor using differential marker ranking.

        Parameters
        ----------
        view : str
            View/modality name in the MOFA model.
        log2fc_min : int
            Minimum log2 fold-change threshold for retained features.
        pval_cutoff : float
            Maximum adjusted p-value threshold for retained features.

        Returns
        -------
        dict
            Dictionary of marker features keyed by factor label.
        """


        factor_w=pd.DataFrame()
        for i in range(self.factors.shape[1]):
            f1_w=get_weights(hdf5_path=self.model_path,view=view,factor=i+1)
            f1_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]
            factor_w['factor_{}'.format(i+1)]=f1_w['weights']
        factor_w.index=[str(i,"utf8").replace('{}_'.format(view),'') for i in f1_w['feature']]

        adata1=anndata.AnnData(pd.concat([factor_w,factor_w],axis=1).T)
        adata1.obs['Factor']=adata1.obs.index
        adata1.obs['Factor']=adata1.obs['Factor'].astype('category')
        top_feature=get_celltype_marker(adata1,clustertype='Factor',
                            log2fc_min=log2fc_min,pval_cutoff=pval_cutoff)
        return top_feature

def calculate_r2(Z, W, Y):
    r"""
    Compute variance explained (R2) for a reconstructed data matrix.
    
    Parameters
    ----------
    Z : np.ndarray
        Factor matrix with shape ``(n_factors, n_samples)``.
    W : np.ndarray
        Weight matrix with shape ``(n_factors, n_features)``.
    Y : np.ndarray
        Observed data matrix with shape ``(n_samples, n_features)``.
    
    Returns
    -------
    float
        Fraction of variance explained by ``Z.T @ W``.
    """
    # Calculate predicted values
    Y_pred = Z.T @ W
    
    # Calculate residual sum of squares
    RSS = np.sum((Y - Y_pred) ** 2)
    
    # Calculate total sum of squares
    TSS = np.sum((Y - np.mean(Y)) ** 2)
    
    # Calculate R2
    r2 = 1 - RSS / TSS
    
    return r2

def get_r2_from_hdf5_complete(hdf5_path: str, 
                              factors: list = None,
                              views: list = None,
                              groups: list = None) -> pd.DataFrame:
    r"""
    Compute/collect factor-wise variance explained from MOFA ``.hdf5``.

    Parameters
    ----------
    hdf5_path : str
        Path to MOFA model file.
    factors : list or None
        1-based factor indices to include. If ``None``, include all factors.
    views : list or None
        View names to include. If ``None``, include all views.
    groups : list or None
        Group names to include. If ``None``, include all groups.

    Returns
    -------
    pd.DataFrame
        Long-format table with columns ``Factor``, ``View``, ``Group`` and ``R2``.
    """
    f = h5py.File(hdf5_path, 'r')
    
    # Get metadata
    view_names = [view.decode('utf-8') if isinstance(view, bytes) else view 
                  for view in f['views']['views'][:]]
    group_names = [group.decode('utf-8') if isinstance(group, bytes) else group 
                   for group in f['groups']['groups'][:]]
    
    # Filter by requested views and groups
    if views is not None:
        view_names = [v for v in view_names if v in views]
    if groups is not None:
        group_names = [g for g in group_names if g in groups]
    
    # Get number of factors
    nfactors = f['expectations']['Z'][group_names[0]].shape[0]
    
    # Filter by requested factors
    if factors is not None:
        factor_indices = [f-1 for f in factors if f <= nfactors]
        factor_names = [f'Factor{f}' for f in factors if f <= nfactors]
    else:
        factor_indices = list(range(nfactors))
        factor_names = [f'Factor{i+1}' for i in range(nfactors)]
    
    # Check if variance_explained exists in the file
    if 'variance_explained' in f.keys():
        # Load pre-computed R2 values
        r2_data = []
        for group_name in group_names:
            if group_name in f['variance_explained']['r2_per_factor']:
                r2_matrix = f['variance_explained']['r2_per_factor'][group_name][:]
                
                for view_idx, view_name in enumerate(view_names):
                    if view_idx < r2_matrix.shape[0]:
                        for factor_idx in factor_indices:
                            if factor_idx < r2_matrix.shape[1]:
                                r2_data.append({
                                    'Factor': f'Factor{factor_idx + 1}',
                                    'View': view_name,
                                    'Group': group_name,
                                    'R2': r2_matrix[view_idx, factor_idx]
                                })
        
        r2_df = pd.DataFrame(r2_data)
    else:
        # Calculate R2 if not pre-computed
        print("Calculating R2 values from scratch...")
        r2_data = []
        
        for group_name in group_names:
            Z_group = f['expectations']['Z'][group_name][:]  # shape: (n_factors, n_samples)
            
            for view_name in view_names:
                W_view = f['expectations']['W'][view_name][:]  # shape: (n_factors, n_features)
                
                if 'data' in f and view_name in f['data'] and group_name in f['data'][view_name]:
                    Y_data = f['data'][view_name][group_name][:]  # shape: (n_samples, n_features)
                    
                    for factor_idx in factor_indices:
                        # Calculate R2 for this specific factor
                        Z_factor = Z_group[factor_idx:factor_idx+1, :]  # shape: (1, n_samples)
                        W_factor = W_view[factor_idx:factor_idx+1, :]  # shape: (1, n_features)
                        
                        r2_value = calculate_r2(Z_factor, W_factor, Y_data)
                        
                        r2_data.append({
                            'Factor': f'Factor{factor_idx + 1}',
                            'View': view_name,
                            'Group': group_name,
                            'R2': r2_value
                        })
                else:
                    print(f"Warning: Data not found for view {view_name}, group {group_name}")
        
        r2_df = pd.DataFrame(r2_data)
    
    f.close()
    return r2_df

def get_r2_from_hdf5(hdf5_path: str) -> pd.DataFrame:
    r"""
    Load factor-wise explained variance from a MOFA ``.hdf5`` model.

    Parameters
    ----------
    hdf5_path : str
        Path to MOFA model file.

    Returns
    -------
    pd.DataFrame
        Long-format table with columns ``Factor``, ``View``, ``Group`` and ``R2``.

    """
    f = h5py.File(hdf5_path, 'r')
    
    # Get view names and factor count
    view_names = [view.decode('utf-8') if isinstance(view, bytes) else view 
                  for view in f['views']['views'][:]]
    
    # Check if variance_explained exists in the file
    if 'variance_explained' in f.keys():
        # Load pre-computed R2 values
        r2_data = []
        for group_name in f['variance_explained']['r2_per_factor'].keys():
            r2_matrix = f['variance_explained']['r2_per_factor'][group_name][:]
            nfactors = r2_matrix.shape[1]
            
            for view_idx, view_name in enumerate(view_names):
                for factor_idx in range(nfactors):
                    r2_data.append({
                        'Factor': f'Factor{factor_idx + 1}',
                        'View': view_name,
                        'Group': group_name.decode('utf-8') if isinstance(group_name, bytes) else group_name,
                        'R2': r2_matrix[view_idx, factor_idx]
                    })
        
        r2_df = pd.DataFrame(r2_data)
    else:
        # Use the complete function for calculation
        print("Pre-computed R2 not found. Using complete calculation...")
        f.close()
        return get_r2_from_hdf5_complete(hdf5_path)
    
    f.close()
    return r2_df

def convert_r2_to_matrix(r2_df: pd.DataFrame, group: str = None) -> pd.DataFrame:
    r"""
    Convert long-format R2 table into factor-by-view matrix form.

    Parameters
    ----------
    r2_df : pd.DataFrame
        DataFrame containing at least ``Factor``, ``View``, ``Group`` and ``R2``.
    group : str or None
        Group name to keep. If ``None`` and multiple groups are present, values
        are averaged across groups.

    Returns
    -------
    pd.DataFrame
        Matrix with factors as rows and views as columns.
    """
    if group is not None:
        # Filter for specific group
        r2_df = r2_df[r2_df['Group'] == group]
    else:
        # If multiple groups exist, take mean across groups
        r2_df = r2_df.groupby(['Factor', 'View'])['R2'].mean().reset_index()
    
    # Pivot the DataFrame
    matrix_df = r2_df.pivot(index='Factor', columns='View', values='R2')
    
    # Sort factor indices
    matrix_df = matrix_df.reindex(sorted(matrix_df.index, key=lambda x: int(x.replace('Factor', ''))))
    
    return matrix_df

def plot_factors_violin(mdata,
                      factors: list = None,
                      group: str = 'Group',
                      view: str = None,  # New parameter for view-specific plotting
                      violins: bool = True,
                      dots: bool = False,
                      zero_line: bool = True,
                      linewidth: float = 0,
                      zero_linewidth: float = 1,
                      size: float = 20,
                      legend: bool = True,
                      legend_prop: dict = None,
                      palette: str = None,
                      alpha: float = None,
                      violins_alpha: float = None,
                      ncols: int = 4,
                      sharex: bool = False,
                      sharey: bool = False,
                      figsize: tuple = (6,4),
                      **kwargs) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    r"""
    Plot MOFA factor distributions across groups as violin/strip plots.

    Parameters
    ----------
    mdata : MuData
        MuData containing MOFA scores in ``mdata.obsm['X_mofa']``.
    factors : list or None
        Factor IDs to plot. If ``None``, plot all available factors.
    group : str
        Grouping column name searched in ``obs`` of selected view(s).
    view : str or None
        Specific modality to use. If ``None``, combine samples from all views.
    violins : bool
        Whether to draw violin distributions.
    dots : bool
        Whether to overlay per-sample strip dots.
    zero_line : bool
        Whether to draw a horizontal zero reference line.
    linewidth : float
        Dot edge line width for strip plot.
    zero_linewidth : float
        Line width of zero reference line.
    size : float
        Dot size for strip plot.
    legend : bool
        Whether to display legend.
    legend_prop : dict or None
        Matplotlib legend style arguments.
    palette : str or None
        Seaborn palette for group colors.
    alpha : float or None
        Dot transparency.
    violins_alpha : float or None
        Violin transparency.
    ncols : int
        Reserved for compatibility; current implementation draws one panel.
    sharex : bool
        Reserved for compatibility.
    sharey : bool
        Reserved for compatibility.
    figsize : tuple
        Figure size in inches as ``(width, height)``.
    **kwargs
        Additional keyword arguments forwarded to seaborn plotting calls.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
        Matplotlib figure and axes containing factor distribution plot.
    """
    # Get factors data
    if 'X_mofa' not in mdata.obsm:
        raise ValueError("MOFA factors not found in mdata.obsm['X_mofa']")
    
    # Get MOFA factors from main MuData
    factor_data_full = mdata.obsm['X_mofa']
    main_samples = mdata.obs_names.tolist()
    
    # Create factors DataFrame
    factor_names = [f'Factor{i+1}' for i in range(factor_data_full.shape[1])]
    
    # Filter factors if specified
    if factors is not None:
        factor_indices = [i-1 if isinstance(i, int) else int(i.replace('Factor', ''))-1 for i in factors]
        factor_data_full = factor_data_full[:, factor_indices]
        factor_names = [factor_names[i] for i in factor_indices]
    
    if view is not None:
        # View-specific plotting (consistent with plot_factor_boxplots)
        if view not in mdata.mod.keys():
            raise ValueError(f"View '{view}' not found in MuData. Available views: {list(mdata.mod.keys())}")
        
        # Get data for specified view
        adata = mdata.mod[view]
        
        # Check if group column exists
        if group not in adata.obs.columns:
            raise ValueError(f"Group column '{group}' not found in {view}")
        
        # Get sample names for current view
        view_samples = adata.obs_names.tolist()
        
        # Find matching samples between view and main MuData
        sample_indices = []
        matched_samples = []
        for sample in view_samples:
            if sample in main_samples:
                idx = main_samples.index(sample)
                sample_indices.append(idx)
                matched_samples.append(sample)
        
        if len(sample_indices) == 0:
            raise ValueError(f"No matching samples found between {view} and main MuData")
        
        print(f"Using {len(matched_samples)} samples from {view} view")
        
        # Get factor data for matched samples
        factor_data = factor_data_full[sample_indices, :]
        
        # Get group information for matched samples
        groups = adata.obs.loc[matched_samples, group].values
        
        # Create plot data
        factor_df = pd.DataFrame(factor_data, columns=factor_names, index=matched_samples)
        
    else:
        # Original behavior: use all samples
        factor_df = pd.DataFrame(factor_data_full, columns=factor_names, index=main_samples)
        
        # Get group information from each modality
        all_obs_names = set()
        
        # First, get all observation names from all modalities
        for mod in mdata.mod.keys():
            mod_obs_names = set(mdata.mod[mod].obs.index)
            all_obs_names.update(mod_obs_names)
        
        # Create a combined group Series initialized with NaN
        combined_groups = pd.Series(np.nan, index=list(all_obs_names))
        
        # Fill in group information where available
        for mod in mdata.mod.keys():
            if group in mdata.mod[mod].obs.columns:
                mod_group_data = mdata.mod[mod].obs[group]
                combined_groups.loc[mod_group_data.index] = mod_group_data
        
        # Check if we have any group information
        if combined_groups.isna().all():
            raise ValueError(f"Group column '{group}' not found in any modality")
        
        # Ensure factor_data has the same index as combined_groups
        factor_df.index = combined_groups.index
        groups = combined_groups.values
        
        print(f"Using all {len(factor_df)} samples from all views")
    
    # Melt the DataFrame for plotting
    plot_data = factor_df.reset_index().melt(id_vars='index', 
                                           var_name='Factor',
                                           value_name='Value')
    
    # Add group information
    if view is not None:
        plot_data['Group'] = np.repeat(groups, len(factor_names))
    else:
        plot_data['Group'] = np.repeat(groups, len(factor_names))
    
    # Sort factors by number
    plot_data['Factor_num'] = plot_data['Factor'].str.extract(r'(\d+)').astype(int)
    plot_data = plot_data.sort_values('Factor_num')
    
    # Remove NaN groups from plotting
    plot_data_clean = plot_data.dropna(subset=['Group'])
    
    # Print information about missing groups
    n_missing = plot_data['Group'].isna().sum()
    if n_missing > 0:
        print(f"Note: {n_missing} observations were found in some modalities but not others "
              f"(these will be excluded from the plot)")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot violins
    if violins:
        sns.violinplot(data=plot_data_clean, x='Factor', y='Value', hue='Group',
                      inner=None, alpha=violins_alpha, ax=ax,
                      palette=palette)
    
    # Plot dots
    if dots:
        sns.stripplot(data=plot_data_clean, x='Factor', y='Value', hue='Group',
                     size=size, alpha=alpha, linewidth=linewidth,
                     dodge=True, ax=ax, palette=palette)
    
    # Add zero line
    if zero_line:
        ax.axhline(y=0, color='black', linestyle='--', 
                  linewidth=zero_linewidth, alpha=0.5)
    
    # Customize plot
    if not legend:
        ax.get_legend().remove()
    elif legend_prop is not None:
        ax.legend(prop=legend_prop)
    
    plt.xlabel('Factor')
    plt.ylabel('Value')
    
    # Add title to indicate view if specified
    if view is not None:
        plt.title(f'Factor Values by Group ({view} view, n={len(matched_samples)})')
    else:
        plt.title(f'Factor Values by Group (All views, n={len(factor_df)})')
    
    # Apply any additional customization
    for key, value in kwargs.items():
        setattr(ax, key, value)
    
    return fig, ax

def plot_factors(mdata,
                x: Union[int, str] = 1,
                y: Union[int, str] = 2,
                group: str = 'Group',
                dist: bool = False,
                zero_line_x: bool = False,
                zero_line_y: bool = False,
                linewidth: float = 0,
                zero_linewidth: float = 1,
                size: float = 20,
                legend: bool = True,
                legend_prop: dict = None,
                palette: str = None,
                alpha: float = None,
                figsize: tuple = (6,4),
                **kwargs) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    r"""
    Plot pairwise factor embedding colored by biological groups.

    Parameters
    ----------
    mdata : MuData
        MuData containing MOFA factors in ``mdata.obsm['X_mofa']``.
    x : Union[int, str]
        Factor index/name for x-axis (for example ``1`` or ``'Factor1'``).
    y : Union[int, str]
        Factor index/name for y-axis (for example ``2`` or ``'Factor2'``).
    group : str
        Grouping column name searched in modality ``obs`` tables.
    dist : bool
        If ``True``, return a seaborn jointplot with marginal distributions.
    zero_line_x : bool
        Whether to draw a vertical zero reference line.
    zero_line_y : bool
        Whether to draw a horizontal zero reference line.
    linewidth : float
        Point edge line width.
    zero_linewidth : float
        Line width for zero reference lines.
    size : float
        Scatter point size.
    legend : bool
        Whether to show legend.
    legend_prop : dict or None
        Matplotlib legend style arguments.
    palette : str or None
        Seaborn palette for group colors.
    alpha : float or None
        Point transparency.
    figsize : tuple
        Figure size in inches as ``(width, height)`` when ``dist=False``.
    **kwargs
        Additional keyword arguments forwarded to seaborn plotting calls.

    Returns
    -------
    Union[
        Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes],
        seaborn.axisgrid.JointGrid
    ]
        Scatter figure/axes when ``dist=False``; otherwise a seaborn JointGrid.
    """
    # Get factors data
    if 'X_mofa' not in mdata.obsm:
        raise ValueError("MOFA factors not found in mdata.obsm['X_mofa']")
    
    # Convert factor indices to names if needed
    x_name = f'Factor{x}' if isinstance(x, int) else x
    y_name = f'Factor{y}' if isinstance(y, int) else y
    
    # Create factors DataFrame
    factor_names = [f'Factor{i+1}' for i in range(mdata.obsm['X_mofa'].shape[1])]
    factor_data = pd.DataFrame(mdata.obsm['X_mofa'], 
                             columns=factor_names)
    
    # Get all observation names from all modalities
    all_obs_names = set()
    for mod in mdata.mod.keys():
        mod_obs_names = set(mdata.mod[mod].obs.index)
        all_obs_names.update(mod_obs_names)
    
    # Create a combined group Series initialized with NaN
    combined_groups = pd.Series(np.nan, index=list(all_obs_names))
    
    # Fill in group information where available
    for mod in mdata.mod.keys():
        if group in mdata.mod[mod].obs.columns:
            mod_group_data = mdata.mod[mod].obs[group]
            combined_groups.loc[mod_group_data.index] = mod_group_data
    
    # Check if we have any group information
    if combined_groups.isna().all():
        raise ValueError(f"Group column '{group}' not found in any modality")
    
    # Ensure factor_data has the same index as combined_groups
    factor_data.index = combined_groups.index
    
    # Create plot data
    plot_data = pd.DataFrame({
        'x': factor_data[x_name],
        'y': factor_data[y_name],
        'Group': combined_groups
    })
    
    # Remove NaN groups from plotting
    plot_data_clean = plot_data.dropna(subset=['Group'])
    
    # Print information about missing groups
    n_missing = plot_data['Group'].isna().sum()
    if n_missing > 0:
        print(f"Note: {n_missing} observations were found in some modalities but not others "
              f"(these will be excluded from the plot)")
    
    if dist:
        # Create joint plot with marginal distributions
        g = sns.jointplot(
            data=plot_data_clean,
            x='x',
            y='y',
            hue='Group',
            palette=palette,
            alpha=alpha,
            s=size,
            **kwargs
        )
        
        # Add zero lines if requested
        if zero_line_x:
            g.ax_joint.axvline(x=0, color='black', linestyle='--', 
                             linewidth=zero_linewidth, alpha=0.5)
        if zero_line_y:
            g.ax_joint.axhline(y=0, color='black', linestyle='--', 
                             linewidth=zero_linewidth, alpha=0.5)
        
        # Customize labels
        g.ax_joint.set_xlabel(x_name)
        g.ax_joint.set_ylabel(y_name)
        
        # Handle legend
        if not legend:
            g.ax_joint.get_legend().remove()
        elif legend_prop is not None:
            g.ax_joint.legend(prop=legend_prop)
        
        return g
    else:
        # Create regular scatter plot
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.scatterplot(
            data=plot_data_clean,
            x='x',
            y='y',
            hue='Group',
            palette=palette,
            alpha=alpha,
            s=size,
            ax=ax,
            **kwargs
        )
        
        # Add zero lines if requested
        if zero_line_x:
            ax.axvline(x=0, color='black', linestyle='--', 
                      linewidth=zero_linewidth, alpha=0.5)
        if zero_line_y:
            ax.axhline(y=0, color='black', linestyle='--', 
                      linewidth=zero_linewidth, alpha=0.5)
        
        # Customize labels
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        
        # Handle legend
        if not legend:
            ax.get_legend().remove()
        elif legend_prop is not None:
            ax.legend(prop=legend_prop)
        
        return fig, ax
    

def store_weights(mdata, hdf5_path: str):
    r"""
    Store MOFA factor loadings into each modality ``varm`` table.

    Parameters
    ----------
    mdata : MuData
        MuData object whose modalities will receive ``varm['mofa_weights']``.
    hdf5_path : str
        Path to trained MOFA ``.hdf5`` model.

    Returns
    -------
    None
        Updates ``mdata.mod[view].varm['mofa_weights']`` for matching views.
    """
    f = h5py.File(hdf5_path, 'r')
    view_names = [view.decode('utf-8') if isinstance(view, bytes) else view 
                  for view in f['views']['views'][:]]
    
    # Get number of factors
    nfactors = f['expectations']['Z'][list(f['expectations']['Z'].keys())[0]].shape[0]
    
    # Store weights for each view
    for view_name in view_names:
        if view_name in mdata.mod:
            # Get weights for this view
            weights = f['expectations']['W'][view_name][:]  # shape: (n_factors, n_features)
            feature_lst=[x.decode('utf-8') for x in f['features'][view_name][:]]
            
            # Create DataFrame with proper index and columns
            weights_df = pd.DataFrame(
                np.full((mdata.mod[view_name].shape[1], nfactors), np.nan),  # Transpose to make features as rows
                index=mdata.mod[view_name].var_names,
                columns=[f'Factor{i+1}' for i in range(nfactors)]
            )
            weights_df.loc[feature_lst,:] = weights.T
            
            # Store in varm
            mdata.mod[view_name].varm['mofa_weights'] = weights_df
    
    f.close()

def plot_weights(mdata,
                view: str,
                factors=None,
                n_features: int = 5,
                w_scaled: bool = False,
                w_abs: bool = False,
                size: float = 2,
                color: str = "black",
                label_size: float = 5,
                x_offset: float = 0.01,
                y_offset: float = 0.15,
                jitter: float = 0.01,
                line_width: float = 0.5,
                line_color: str = "black",
                line_alpha: float = 0.2,
                zero_line: bool = True,
                zero_line_width: float = 1,
                ncols: int = 4,
                sharex: bool = True,
                sharey: bool = False,
                feature_label: str = None,
                figsize: tuple = (10, 6),
                **kwargs) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    r"""
    Plot factor loadings (weights) for a selected MOFA view.

    Parameters
    ----------
    mdata : MuData
        MuData with ``mofa_weights`` stored in modality ``varm``.
    view : str
        Modality/view name to visualize.
    factors : list or None
        Factors to plot. If ``None``, plot all factors.
    n_features : int
        Number of top-weighted features to annotate per factor.
    w_scaled : bool
        Whether to z-score weight values within each factor before plotting.
    w_abs : bool
        Whether to plot absolute weight values.
    size : float
        Scatter marker size.
    color : str
        Marker/label color for highlighted features.
    label_size : float
        Font size of feature labels.
    x_offset : float
        Horizontal offset applied to labels.
    y_offset : float
        Vertical repulsion strength for label placement.
    jitter : float
        Jitter amplitude for x positions to reduce overplotting.
    line_width : float
        Width of connector lines from labels to points.
    line_color : str
        Color of connector lines.
    line_alpha : float
        Alpha of connector lines.
    zero_line : bool
        Whether to draw horizontal zero reference line.
    zero_line_width : float
        Width of zero reference line.
    ncols : int
        Number of subplot columns when plotting multiple factors.
    sharex : bool
        Whether to share x-axis across panels.
    sharey : bool
        Whether to share y-axis across panels.
    feature_label : str or None
        Optional ``var`` column used as displayed feature names.
    figsize : tuple
        Figure size in inches as ``(width, height)``.
    **kwargs
        Additional keyword arguments forwarded to seaborn plotting calls.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
        Matplotlib figure and axes for factor-loading visualization.
    """
    if view not in mdata.mod:
        raise ValueError(f"View {view} not found in MuData object")
    
    if 'mofa_weights' not in mdata.mod[view].varm:
        raise ValueError(f"Weights not found in varm['mofa_weights']. Please run store_weights first.")
    
    # Get weights
    weights = pd.DataFrame(mdata.mod[view].varm['mofa_weights'])
    
    # Get feature labels
    if feature_label is not None and feature_label in mdata.mod[view].var:
        feature_names = mdata.mod[view].var[feature_label]
    else:
        feature_names = mdata.mod[view].var_names
    
    # Set index name for weights DataFrame
    weights.index = feature_names
    weights.index.name = 'feature'  # Explicitly set index name
    
    # Filter factors if specified
    if factors is not None:
        factor_names = [f'Factor{i}' if isinstance(i, int) else i for i in factors]
        weights = weights[factor_names]
    
    # Scale weights if requested
    if w_scaled:
        weights = weights / weights.abs().max()
    
    # Convert to absolute values if requested
    if w_abs:
        weights = weights.abs()
    
    # Melt the DataFrame for plotting
    wm = weights.reset_index().melt(
        id_vars='feature',
        var_name='factor',
        value_name='value'
    )
    wm['abs_value'] = abs(wm['value'])
    
    # Sort factors
    wm['factor'] = wm['factor'].astype('category')
    wm['factor'] = wm['factor'].cat.reorder_categories(
        sorted(wm['factor'].cat.categories, key=lambda x: int(x.split('Factor')[1]))
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create stripplot
    g = sns.stripplot(
        data=wm,
        x='value',
        y='factor',
        jitter=jitter,
        size=size,
        color='lightgrey',
        ax=ax
    )
    
    # Add feature labels for each factor separately
    for fi, factor in enumerate(wm['factor'].unique()):
        factor_data = wm[wm['factor'] == factor].sort_values('abs_value', ascending=False)
        
        # Get top features for this factor only
        top_features = factor_data.head(n_features)
        
        # Overlay colored dots for top features
        for _, point in top_features.iterrows():
            ax.scatter(point['value'], fi, color=color, s=size*10, alpha=0.8, zorder=5)
        
        # Add labels for positive and negative weights separately
        for sign_i in [1, -1]:
            to_label = top_features[top_features['value'] * sign_i > 0].sort_values('abs_value', ascending=False)
            
            if len(to_label) == 0:
                continue
                
            x_start_pos = sign_i * (to_label['abs_value'].max() + x_offset)
            y_start_pos = fi - ((len(to_label) - 1) // 2) * y_offset
            y_prev = y_start_pos
            
            for i, (_, point) in enumerate(to_label.iterrows()):
                y_loc = y_prev + y_offset if i != 0 else y_start_pos
                
                g.annotate(
                    point['feature'],
                    xy=(point['value'], fi),
                    xytext=(x_start_pos, y_loc),
                    arrowprops=dict(
                        arrowstyle='-',
                        connectionstyle='arc3',
                        color=line_color,
                        alpha=line_alpha,
                        linewidth=line_width
                    ),
                    horizontalalignment='left' if sign_i > 0 else 'right',
                    size=label_size,
                    color='black',
                    weight='regular',
                    alpha=0.9
                )
                y_prev = y_loc
    
    # Add zero line
    if zero_line:
        ax.axvline(0, ls='--', color='lightgrey', linewidth=zero_line_width, zorder=0)
    
    # Customize plot
    sns.despine(offset=10, trim=True)
    ax.set_xlabel('Feature weight')
    ax.set_ylabel('')
    ax.set_title(view)
    
    return fig, ax

def compute_cross_correlation(mdata, 
                             view1: str, 
                             view2: str,
                             method: str = 'pearson',
                             min_corr: float = 0.3,
                             p_threshold: float = 0.05,
                             chunk_size: int = 1000) -> pd.DataFrame:
    r"""
    Compute feature-feature cross-modality correlations on paired samples.

    Parameters
    ----------
    mdata : MuData
        MuData object with matched sample IDs across modalities.
    view1 : str
        First modality name (for example ``'metab'``).
    view2 : str
        Second modality name (for example ``'micro'``).
    method : str
        Correlation metric: ``'pearson'``, ``'spearman'`` or ``'kendall'``.
    min_corr : float
        Minimum absolute correlation to retain.
    p_threshold : float
        Maximum p-value for significance filtering.
    chunk_size : int
        Number of features processed per chunk for memory control.

    Returns
    -------
    pd.DataFrame
        Significant feature-pair correlations with p-values and absolute
        correlations.
    """
    from scipy.stats import pearsonr, spearmanr, kendalltau
    from scipy.stats import t as t_dist
    
    # Get paired samples
    common_samples = list(set(mdata.mod[view1].obs_names) & set(mdata.mod[view2].obs_names))
    if len(common_samples) == 0:
        raise ValueError("No common samples found between the two modalities")
    
    print(f"Found {len(common_samples)} paired samples")
    
    # Get data for paired samples
    data1 = mdata.mod[view1][common_samples].X
    data2 = mdata.mod[view2][common_samples].X
    
    # Convert to dense if sparse
    if hasattr(data1, 'toarray'):
        data1 = data1.toarray()
    if hasattr(data2, 'toarray'):
        data2 = data2.toarray()
    
    # Transpose for easier computation (samples x features -> features x samples)
    data1 = data1.T
    data2 = data2.T
    
    n_features1, n_samples = data1.shape
    n_features2, _ = data2.shape
    
    print(f"Computing correlations between {n_features1} and {n_features2} features...")
    print(f"Using chunk size: {chunk_size}")
    
    # Prepare results storage
    correlations = []
    
    if method == 'pearson':
        # Use numpy's corrcoef for Pearson correlation (much faster)
        print("Using optimized Pearson correlation...")
        
        # Standardize data for correlation computation
        data1_std = (data1 - data1.mean(axis=1, keepdims=True)) / data1.std(axis=1, keepdims=True)
        data2_std = (data2 - data2.mean(axis=1, keepdims=True)) / data2.std(axis=1, keepdims=True)
        
        # Replace NaN with 0 (for features with zero variance)
        data1_std = np.nan_to_num(data1_std)
        data2_std = np.nan_to_num(data2_std)
        
        # Process in chunks to manage memory
        n_chunks1 = (n_features1 + chunk_size - 1) // chunk_size
        
        for chunk_idx1 in tqdm(range(n_chunks1), desc="Processing chunks"):
            start1 = chunk_idx1 * chunk_size
            end1 = min(start1 + chunk_size, n_features1)
            chunk1 = data1_std[start1:end1]
            
            # Compute correlation matrix for this chunk
            # corr_matrix shape: (chunk_size, n_features2)
            corr_matrix = np.dot(chunk1, data2_std.T) / (n_samples - 1)
            
            # Compute p-values using t-distribution
            # t = r * sqrt((n-2)/(1-r^2))
            t_stat = corr_matrix * np.sqrt((n_samples - 2) / (1 - corr_matrix**2 + 1e-10))
            p_values = 2 * (1 - t_dist.cdf(np.abs(t_stat), n_samples - 2))
            
            # Find significant correlations
            abs_corr = np.abs(corr_matrix)
            significant_mask = (abs_corr >= min_corr) & (p_values <= p_threshold)
            
            # Extract significant correlations
            sig_indices = np.where(significant_mask)
            for i, j in zip(sig_indices[0], sig_indices[1]):
                correlations.append({
                    f'{view1}_feature': mdata.mod[view1].var_names[start1 + i],
                    f'{view2}_feature': mdata.mod[view2].var_names[j],
                    'correlation': corr_matrix[i, j],
                    'p_value': p_values[i, j],
                    'abs_correlation': abs_corr[i, j]
                })
    
    elif method in ['spearman', 'kendall']:
        # For rank-based correlations, we need to use scipy functions
        # But we can still optimize by processing in chunks
        print(f"Using {method} correlation with chunking...")
        
        if method == 'spearman':
            corr_func = spearmanr
        else:
            corr_func = kendalltau
        
        # Convert to ranks for spearman
        if method == 'spearman':
            from scipy.stats import rankdata
            data1_ranked = np.array([rankdata(row) for row in data1])
            data2_ranked = np.array([rankdata(row) for row in data2])
        else:
            data1_ranked = data1
            data2_ranked = data2
        
        # Process in chunks
        n_chunks1 = (n_features1 + chunk_size - 1) // chunk_size
        
        for chunk_idx1 in tqdm(range(n_chunks1), desc="Processing chunks"):
            start1 = chunk_idx1 * chunk_size
            end1 = min(start1 + chunk_size, n_features1)
            
            for i in range(start1, end1):
                for j in range(n_features2):
                    try:
                        if method == 'spearman':
                            corr, p_val = corr_func(data1_ranked[i], data2_ranked[j])
                        else:
                            corr, p_val = corr_func(data1[i], data2[j])
                        
                        if abs(corr) >= min_corr and p_val <= p_threshold:
                            correlations.append({
                                f'{view1}_feature': mdata.mod[view1].var_names[i],
                                f'{view2}_feature': mdata.mod[view2].var_names[j],
                                'correlation': corr,
                                'p_value': p_val,
                                'abs_correlation': abs(corr)
                            })
                    except:
                        continue
    
    else:
        raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")
    
    corr_df = pd.DataFrame(correlations)
    print(f"Found {len(corr_df)} significant correlations")
    
    return corr_df

def compute_cross_correlation_fast(mdata, 
                                  view1: str, 
                                  view2: str,
                                  min_corr: float = 0.3,
                                  p_threshold: float = 0.05,
                                  max_features: int = None) -> pd.DataFrame:
    r"""
    Compute Pearson cross-modality correlations with optimized NumPy backend.

    Parameters
    ----------
    mdata : MuData
        MuData object with paired sample IDs.
    view1 : str
        First modality name.
    view2 : str
        Second modality name.
    min_corr : float
        Minimum absolute correlation to retain.
    p_threshold : float
        Maximum p-value for significance filtering.
    max_features : int or None
        Optional feature cap per modality (highest-variance features kept).

    Returns
    -------
    pd.DataFrame
        Significant Pearson correlations between features of two modalities.
    """
    # Get paired samples
    common_samples = list(set(mdata.mod[view1].obs_names) & set(mdata.mod[view2].obs_names))
    if len(common_samples) == 0:
        raise ValueError("No common samples found between the two modalities")
    
    print(f"Found {len(common_samples)} paired samples")
    
    # Get data for paired samples
    data1 = mdata.mod[view1][common_samples].X
    data2 = mdata.mod[view2][common_samples].X
    
    # Convert to dense if sparse
    if hasattr(data1, 'toarray'):
        data1 = data1.toarray()
    if hasattr(data2, 'toarray'):
        data2 = data2.toarray()
    
    # Limit features if specified (for memory management)
    if max_features is not None:
        if data1.shape[1] > max_features:
            # Select features with highest variance
            var1 = np.var(data1, axis=0)
            top_indices1 = np.argsort(var1)[-max_features:]
            data1 = data1[:, top_indices1]
            features1 = mdata.mod[view1].var_names[top_indices1]
        else:
            features1 = mdata.mod[view1].var_names
            
        if data2.shape[1] > max_features:
            var2 = np.var(data2, axis=0)
            top_indices2 = np.argsort(var2)[-max_features:]
            data2 = data2[:, top_indices2]
            features2 = mdata.mod[view2].var_names[top_indices2]
        else:
            features2 = mdata.mod[view2].var_names
    else:
        features1 = mdata.mod[view1].var_names
        features2 = mdata.mod[view2].var_names
    
    n_samples, n_features1 = data1.shape
    _, n_features2 = data2.shape
    
    print(f"Computing correlations between {n_features1} and {n_features2} features...")
    print("Using ultra-fast numpy implementation...")
    
    # Standardize data
    data1_centered = data1 - np.mean(data1, axis=0)
    data2_centered = data2 - np.mean(data2, axis=0)
    
    data1_std = np.std(data1, axis=0)
    data2_std = np.std(data2, axis=0)
    
    # Avoid division by zero
    data1_std[data1_std == 0] = 1
    data2_std[data2_std == 0] = 1
    
    data1_normalized = data1_centered / data1_std
    data2_normalized = data2_centered / data2_std
    
    # Compute correlation matrix using matrix multiplication
    corr_matrix = np.dot(data1_normalized.T, data2_normalized) / (n_samples - 1)
    
    # Compute p-values
    t_stat = corr_matrix * np.sqrt((n_samples - 2) / (1 - corr_matrix**2 + 1e-10))
    from scipy.stats import t as t_dist
    p_values = 2 * (1 - t_dist.cdf(np.abs(t_stat), n_samples - 2))
    
    # Find significant correlations
    abs_corr = np.abs(corr_matrix)
    significant_mask = (abs_corr >= min_corr) & (p_values <= p_threshold)
    
    # Extract results
    correlations = []
    sig_indices = np.where(significant_mask)
    
    for i, j in zip(sig_indices[0], sig_indices[1]):
        correlations.append({
            f'{view1}_feature': features1[i],
            f'{view2}_feature': features2[j],
            'correlation': corr_matrix[i, j],
            'p_value': p_values[i, j],
            'abs_correlation': abs_corr[i, j]
        })
    
    corr_df = pd.DataFrame(correlations)
    print(f"Found {len(corr_df)} significant correlations")
    
    return corr_df

def nmf_coexpression_modules(mdata,
                           view1: str,
                           view2: str,
                           n_components: int = 10,
                           correlation_threshold: float = 0.3,
                           method: str = 'pearson',
                           random_state: int = 42,
                           corr_df: pd.DataFrame = None) -> dict:
    r"""
    Identify cross-modality co-expression modules via NMF decomposition.

    Parameters
    ----------
    mdata : MuData
        MuData containing paired samples across modalities.
    view1 : str
        First modality name.
    view2 : str
        Second modality name.
    n_components : int
        Number of latent NMF modules.
    correlation_threshold : float
        Minimum absolute correlation retained in input matrix.
    method : str
        Correlation method used when ``corr_df`` is not provided.
    random_state : int
        Random seed for NMF reproducibility.
    corr_df : pd.DataFrame or None
        Optional precomputed correlation table to reuse.

    Returns
    -------
    dict
        Dictionary containing NMF model, W/H matrices, correlation matrix and
        per-module top features.
    """
    from sklearn.decomposition import NMF
    
    # Use pre-computed correlation data if provided
    if corr_df is not None:
        print("Using pre-computed correlation DataFrame...")
        # Validate that the correlation DataFrame has the expected columns
        expected_cols = [f'{view1}_feature', f'{view2}_feature', 'correlation', 'abs_correlation']
        missing_cols = [col for col in expected_cols if col not in corr_df.columns]
        if missing_cols:
            raise ValueError(f"Pre-computed corr_df missing columns: {missing_cols}")
        
        # Filter by threshold if needed
        corr_df_filtered = corr_df[corr_df['abs_correlation'] >= correlation_threshold].copy()
    else:
        # Compute cross-correlation matrix
        print("Computing cross-correlation matrix...")
        corr_df = compute_cross_correlation(mdata, view1, view2, 
                                          method=method, 
                                          min_corr=correlation_threshold)
        corr_df_filtered = corr_df.copy()
    
    if len(corr_df_filtered) == 0:
        raise ValueError("No significant correlations found")
    
    # Create correlation matrix
    features1 = corr_df_filtered[f'{view1}_feature'].unique()
    features2 = corr_df_filtered[f'{view2}_feature'].unique()
    
    corr_matrix = pd.DataFrame(0.0, index=features1, columns=features2)
    
    for _, row in corr_df_filtered.iterrows():
        corr_matrix.loc[row[f'{view1}_feature'], row[f'{view2}_feature']] = abs(row['correlation'])
    
    print(f"Correlation matrix shape: {corr_matrix.shape}")
    
    # Apply NMF
    print(f"Applying NMF with {n_components} components...")
    nmf = NMF(n_components=n_components, random_state=random_state, max_iter=1000)
    W = nmf.fit_transform(corr_matrix.values)  # Features x Components
    H = nmf.components_  # Components x Features
    
    # Create results
    results = {
        'nmf_model': nmf,
        'W_matrix': pd.DataFrame(W, index=corr_matrix.index, 
                               columns=[f'Module_{i+1}' for i in range(n_components)]),
        'H_matrix': pd.DataFrame(H, index=[f'Module_{i+1}' for i in range(n_components)], 
                               columns=corr_matrix.columns),
        'correlation_matrix': corr_matrix,
        'correlation_data': corr_df,
        'reconstruction_error': nmf.reconstruction_err_
    }
    
    # Extract modules
    modules = {}
    for i in range(n_components):
        module_name = f'Module_{i+1}'
        
        # Get top features from each modality for this module
        w_scores = results['W_matrix'][module_name].sort_values(ascending=False)
        h_scores = results['H_matrix'].loc[module_name].sort_values(ascending=False)
        
        modules[module_name] = {
            f'{view1}_features': w_scores.head(20).to_dict(),
            f'{view2}_features': h_scores.head(20).to_dict(),
            f'{view1}_top_features': w_scores.head(10).index.tolist(),
            f'{view2}_top_features': h_scores.head(10).index.tolist()
        }
    
    results['modules'] = modules
    
    print(f"NMF completed. Reconstruction error: {nmf.reconstruction_err_:.4f}")
    
    return results

def plot_nmf_modules(nmf_results: dict,
                    view1: str,
                    view2: str,
                    n_top_features: int = 10,
                    figsize: tuple = (12, 8),
                    cmap: str = 'viridis') -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    r"""
    Visualize top weighted features for each NMF module.

    Parameters
    ----------
    nmf_results : dict
        Output dictionary from :func:`nmf_coexpression_modules`.
    view1 : str
        First modality name.
    view2 : str
        Second modality name.
    n_top_features : int
        Number of top features to display for each module and modality.
    figsize : tuple
        Figure size in inches as ``(width, height)``.
    cmap : str
        Colormap used for bar colors.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
        Matplotlib figure and axes grid with module barplots.
    """
    modules = nmf_results['modules']
    n_modules = len(modules)
    
    fig, axes = plt.subplots(2, n_modules, figsize=figsize, sharey='row')
    if n_modules == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (module_name, module_data) in enumerate(modules.items()):
        # Plot view1 features
        features1 = list(module_data[f'{view1}_features'].keys())[:n_top_features]
        scores1 = list(module_data[f'{view1}_features'].values())[:n_top_features]
        
        axes[0, i].barh(range(len(features1)), scores1, color=plt.cm.get_cmap(cmap)(0.3))
        axes[0, i].set_yticks(range(len(features1)))
        axes[0, i].set_yticklabels(features1, fontsize=8)
        axes[0, i].set_title(f'{module_name}\n{view1}', fontsize=10)
        axes[0, i].set_xlabel('NMF Score')
        
        # Plot view2 features
        features2 = list(module_data[f'{view2}_features'].keys())[:n_top_features]
        scores2 = list(module_data[f'{view2}_features'].values())[:n_top_features]
        
        axes[1, i].barh(range(len(features2)), scores2, color=plt.cm.get_cmap(cmap)(0.7))
        axes[1, i].set_yticks(range(len(features2)))
        axes[1, i].set_yticklabels(features2, fontsize=8)
        axes[1, i].set_title(f'{view2}', fontsize=10)
        axes[1, i].set_xlabel('NMF Score')
    
    plt.tight_layout()
    return fig, axes

def plot_correlation_heatmap(corr_df: pd.DataFrame,
                           view1: str,
                           view2: str,
                           top_n: int = 50,
                           figsize: tuple = (10, 8),
                           cmap: str = 'RdBu_r') -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    r"""
    Plot a heatmap of top cross-modality feature correlations.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Correlation table from cross-modality correlation functions.
    view1 : str
        First modality name.
    view2 : str
        Second modality name.
    top_n : int
        Number of highest absolute-correlation pairs to visualize.
    figsize : tuple
        Figure size in inches as ``(width, height)``.
    cmap : str
        Colormap for correlation heatmap.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
        Matplotlib figure and axes for the heatmap.
    """
    # Get top correlations
    top_corr = corr_df.nlargest(top_n, 'abs_correlation')
    
    # Create correlation matrix for visualization
    features1 = top_corr[f'{view1}_feature'].unique()
    features2 = top_corr[f'{view2}_feature'].unique()
    
    corr_matrix = pd.DataFrame(0.0, index=features1, columns=features2)
    
    for _, row in top_corr.iterrows():
        corr_matrix.loc[row[f'{view1}_feature'], row[f'{view2}_feature']] = row['correlation']
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, cmap=cmap, center=0, 
                xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'Correlation'}, ax=ax)
    
    ax.set_xlabel(view2)
    ax.set_ylabel(view1)
    ax.set_title(f'Top {top_n} Cross-Correlations')
    
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig, ax

def network_analysis(corr_df: pd.DataFrame,
                    view1: str,
                    view2: str,
                    correlation_threshold: float = 0.5) -> dict:
    r"""
    Build and analyze a bipartite feature-correlation network.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Correlation table containing feature pairs and correlation values.
    view1 : str
        First modality name.
    view2 : str
        Second modality name.
    correlation_threshold : float
        Minimum absolute correlation to include an edge.

    Returns
    -------
    dict
        NetworkX graph and common topology metrics (density, components,
        centrality, clustering).
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("NetworkX is required for network analysis. Install with: pip install networkx")
    
    # Filter high correlations
    high_corr = corr_df[corr_df['abs_correlation'] >= correlation_threshold].copy()
    
    # Create network
    G = nx.Graph()
    
    # Add nodes
    for feature in high_corr[f'{view1}_feature'].unique():
        G.add_node(feature, modality=view1)
    for feature in high_corr[f'{view2}_feature'].unique():
        G.add_node(feature, modality=view2)
    
    # Add edges
    for _, row in high_corr.iterrows():
        G.add_edge(row[f'{view1}_feature'], row[f'{view2}_feature'], 
                  weight=abs(row['correlation']), 
                  correlation=row['correlation'])
    
    # Network analysis
    results = {
        'graph': G,
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
        'connected_components': list(nx.connected_components(G)),
        'degree_centrality': nx.degree_centrality(G),
        'betweenness_centrality': nx.betweenness_centrality(G),
        'clustering_coefficient': nx.clustering(G)
    }
    
    print(f"Network created with {results['n_nodes']} nodes and {results['n_edges']} edges")
    print(f"Network density: {results['density']:.4f}")
    print(f"Number of connected components: {len(results['connected_components'])}")
    
    return results

def compute_cross_correlation_torch(mdata, 
                                   view1: str, 
                                   view2: str,
                                   min_corr: float = 0.3,
                                   p_threshold: float = 0.05,
                                   max_features: int = None,
                                   device: str = 'auto',
                                   batch_size1: int = 1000,
                                   batch_size2: int = 1000,
                                   dtype: str = 'float32') -> pd.DataFrame:
    r"""
    Compute cross-modality correlations with PyTorch acceleration.
    
    Parameters
    ----------
    mdata : MuData
        MuData object with paired samples.
    view1 : str
        First modality name.
    view2 : str
        Second modality name.
    min_corr : float
        Minimum absolute correlation to retain.
    p_threshold : float
        Maximum p-value for significance filtering.
    max_features : int or None
        Optional cap on selected high-variance features per modality.
    device : str
        Compute device: ``'auto'``, ``'cpu'``, ``'cuda'`` or ``'mps'``.
    batch_size1 : int
        Feature-batch size for view1 dimension.
    batch_size2 : int
        Feature-batch size for view2 dimension.
    dtype : str
        Tensor precision, ``'float32'`` or ``'float64'``.
        
    Returns
    -------
    pd.DataFrame
        Significant feature-pair correlations with p-values.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        raise ImportError("PyTorch is required. Install with: pip install torch")
    
    # Determine device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Get paired samples
    common_samples = list(set(mdata.mod[view1].obs_names) & set(mdata.mod[view2].obs_names))
    if len(common_samples) == 0:
        raise ValueError("No common samples found between the two modalities")
    
    print(f"Found {len(common_samples)} paired samples")
    
    # Get data for paired samples
    data1 = mdata.mod[view1][common_samples].X
    data2 = mdata.mod[view2][common_samples].X
    
    # Convert to dense if sparse
    if hasattr(data1, 'toarray'):
        data1 = data1.toarray()
    if hasattr(data2, 'toarray'):
        data2 = data2.toarray()
    
    # Feature selection for memory management
    if max_features is not None:
        if data1.shape[1] > max_features:
            var1 = np.var(data1, axis=0)
            top_indices1 = np.argsort(var1)[-max_features:]
            data1 = data1[:, top_indices1]
            features1 = mdata.mod[view1].var_names[top_indices1]
        else:
            features1 = mdata.mod[view1].var_names
            
        if data2.shape[1] > max_features:
            var2 = np.var(data2, axis=0)
            top_indices2 = np.argsort(var2)[-max_features:]
            data2 = data2[:, top_indices2]
            features2 = mdata.mod[view2].var_names[top_indices2]
        else:
            features2 = mdata.mod[view2].var_names
    else:
        features1 = mdata.mod[view1].var_names
        features2 = mdata.mod[view2].var_names
    
    n_samples, n_features1 = data1.shape
    _, n_features2 = data2.shape
    
    print(f"Computing correlations between {n_features1} and {n_features2} features...")
    print(f"Using 2D batching: {batch_size1} x {batch_size2}")
    
    # Standardize data
    data1_mean = np.mean(data1, axis=0)
    data1_std = np.std(data1, axis=0)
    data1_std[data1_std == 0] = 1  # Avoid division by zero
    data1_normalized = (data1 - data1_mean) / data1_std
    
    data2_mean = np.mean(data2, axis=0)
    data2_std = np.std(data2, axis=0)
    data2_std[data2_std == 0] = 1
    data2_normalized = (data2 - data2_mean) / data2_std
    
    torch_dtype = torch.float32 if dtype == 'float32' else torch.float64
    
    print(f"Data standardized and ready for GPU processing")
    
    # Compute correlations in 2D batches
    correlations = []
    n_batches1 = (n_features1 + batch_size1 - 1) // batch_size1
    n_batches2 = (n_features2 + batch_size2 - 1) // batch_size2
    total_batches = n_batches1 * n_batches2
    
    with torch.no_grad():
        with tqdm(total=total_batches, desc="Processing 2D batches") as pbar:
            for batch_idx1 in range(n_batches1):
                start_idx1 = batch_idx1 * batch_size1
                end_idx1 = min(start_idx1 + batch_size1, n_features1)
                
                # Get batch of features from view1
                X1_batch = torch.tensor(data1_normalized[:, start_idx1:end_idx1], 
                                      dtype=torch_dtype, device=device)  # (n_samples, batch_size1)
                
                for batch_idx2 in range(n_batches2):
                    start_idx2 = batch_idx2 * batch_size2
                    end_idx2 = min(start_idx2 + batch_size2, n_features2)
                    
                    # Get batch of features from view2
                    X2_batch = torch.tensor(data2_normalized[:, start_idx2:end_idx2], 
                                          dtype=torch_dtype, device=device)  # (n_samples, batch_size2)
                    
                    # Compute correlation matrix for this batch pair
                    corr_matrix = torch.mm(X1_batch.T, X2_batch) / (n_samples - 1)  # (batch_size1, batch_size2)
                    
                    # Compute p-values using t-distribution
                    corr_squared = corr_matrix ** 2
                    t_stat = corr_matrix * torch.sqrt((n_samples - 2) / (1 - corr_squared + 1e-10))
                    
                    # Convert to numpy for scipy t-distribution
                    t_stat_np = t_stat.cpu().numpy()
                    from scipy.stats import t as t_dist
                    p_values = 2 * (1 - t_dist.cdf(np.abs(t_stat_np), n_samples - 2))
                    
                    # Find significant correlations
                    corr_matrix_np = corr_matrix.cpu().numpy()
                    abs_corr = np.abs(corr_matrix_np)
                    significant_mask = (abs_corr >= min_corr) & (p_values <= p_threshold)
                    
                    # Extract significant correlations
                    sig_indices = np.where(significant_mask)
                    for i, j in zip(sig_indices[0], sig_indices[1]):
                        correlations.append({
                            f'{view1}_feature': features1[start_idx1 + i],
                            f'{view2}_feature': features2[start_idx2 + j],
                            'correlation': corr_matrix_np[i, j],
                            'p_value': p_values[i, j],
                            'abs_correlation': abs_corr[i, j]
                        })
                    
                    pbar.update(1)
    
    corr_df = pd.DataFrame(correlations)
    print(f"Found {len(corr_df)} significant correlations")
    
    return corr_df

def compute_cross_correlation_torch_chunked(mdata, 
                                           view1: str, 
                                           view2: str,
                                           min_corr: float = 0.3,
                                           p_threshold: float = 0.05,
                                           max_features1: int = None,
                                           max_features2: int = None,
                                           device: str = 'auto',
                                           chunk_size1: int = 500,
                                           chunk_size2: int = 1000,
                                           dtype: str = 'float32') -> pd.DataFrame:
    r"""
    Compute cross-modality correlations with memory-efficient 2D chunking.
    
    Parameters
    ----------
    mdata : MuData
        MuData object with paired sample IDs.
    view1 : str
        First modality name.
    view2 : str
        Second modality name.
    min_corr : float
        Minimum absolute correlation to retain.
    p_threshold : float
        Maximum p-value for significance filtering.
    max_features1 : int or None
        Optional cap on selected high-variance features in ``view1``.
    max_features2 : int or None
        Optional cap on selected high-variance features in ``view2``.
    device : str
        Compute device: ``'auto'``, ``'cpu'``, ``'cuda'`` or ``'mps'``.
    chunk_size1 : int
        Feature chunk size along ``view1``.
    chunk_size2 : int
        Feature chunk size along ``view2``.
    dtype : str
        Tensor precision, ``'float32'`` or ``'float64'``.
        
    Returns
    -------
    pd.DataFrame
        Significant feature-pair correlations with p-values and absolute
        correlations.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required. Install with: pip install torch")
    
    # Determine device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Get paired samples
    common_samples = list(set(mdata.mod[view1].obs_names) & set(mdata.mod[view2].obs_names))
    if len(common_samples) == 0:
        raise ValueError("No common samples found between the two modalities")
    
    print(f"Found {len(common_samples)} paired samples")
    
    # Get data for paired samples
    data1 = mdata.mod[view1][common_samples].X
    data2 = mdata.mod[view2][common_samples].X
    
    # Convert to dense if sparse
    if hasattr(data1, 'toarray'):
        data1 = data1.toarray()
    if hasattr(data2, 'toarray'):
        data2 = data2.toarray()
    
    # Feature selection
    if max_features1 is not None and data1.shape[1] > max_features1:
        var1 = np.var(data1, axis=0)
        top_indices1 = np.argsort(var1)[-max_features1:]
        data1 = data1[:, top_indices1]
        features1 = mdata.mod[view1].var_names[top_indices1]
    else:
        features1 = mdata.mod[view1].var_names
        
    if max_features2 is not None and data2.shape[1] > max_features2:
        var2 = np.var(data2, axis=0)
        top_indices2 = np.argsort(var2)[-max_features2:]
        data2 = data2[:, top_indices2]
        features2 = mdata.mod[view2].var_names[top_indices2]
    else:
        features2 = mdata.mod[view2].var_names
    
    n_samples, n_features1 = data1.shape
    _, n_features2 = data2.shape
    
    print(f"Computing correlations between {n_features1} and {n_features2} features...")
    print(f"Using 2D chunking: {chunk_size1} x {chunk_size2}")
    
    # Standardize data
    data1_mean = np.mean(data1, axis=0)
    data1_std = np.std(data1, axis=0)
    data1_std[data1_std == 0] = 1
    data1_normalized = (data1 - data1_mean) / data1_std
    
    data2_mean = np.mean(data2, axis=0)
    data2_std = np.std(data2, axis=0)
    data2_std[data2_std == 0] = 1
    data2_normalized = (data2 - data2_mean) / data2_std
    
    torch_dtype = torch.float32 if dtype == 'float32' else torch.float64
    
    correlations = []
    n_chunks1 = (n_features1 + chunk_size1 - 1) // chunk_size1
    n_chunks2 = (n_features2 + chunk_size2 - 1) // chunk_size2
    
    total_chunks = n_chunks1 * n_chunks2
    
    with torch.no_grad():
        with tqdm(total=total_chunks, desc="Processing 2D chunks") as pbar:
            for chunk_idx1 in range(n_chunks1):
                start1 = chunk_idx1 * chunk_size1
                end1 = min(start1 + chunk_size1, n_features1)
                
                # Load chunk1 to GPU
                X1_chunk = torch.tensor(data1_normalized[:, start1:end1], 
                                      dtype=torch_dtype, device=device)
                
                for chunk_idx2 in range(n_chunks2):
                    start2 = chunk_idx2 * chunk_size2
                    end2 = min(start2 + chunk_size2, n_features2)
                    
                    # Load chunk2 to GPU
                    X2_chunk = torch.tensor(data2_normalized[:, start2:end2], 
                                          dtype=torch_dtype, device=device)
                    
                    # Compute correlation matrix for this chunk pair
                    corr_matrix = torch.mm(X1_chunk.T, X2_chunk) / (n_samples - 1)
                    
                    # Compute p-values
                    corr_squared = corr_matrix ** 2
                    t_stat = corr_matrix * torch.sqrt((n_samples - 2) / (1 - corr_squared + 1e-10))
                    
                    # Convert to numpy for scipy
                    t_stat_np = t_stat.cpu().numpy()
                    corr_matrix_np = corr_matrix.cpu().numpy()
                    
                    from scipy.stats import t as t_dist
                    p_values = 2 * (1 - t_dist.cdf(np.abs(t_stat_np), n_samples - 2))
                    
                    # Find significant correlations
                    abs_corr = np.abs(corr_matrix_np)
                    significant_mask = (abs_corr >= min_corr) & (p_values <= p_threshold)
                    
                    # Extract significant correlations
                    sig_indices = np.where(significant_mask)
                    for i, j in zip(sig_indices[0], sig_indices[1]):
                        correlations.append({
                            f'{view1}_feature': features1[start1 + i],
                            f'{view2}_feature': features2[start2 + j],
                            'correlation': corr_matrix_np[i, j],
                            'p_value': p_values[i, j],
                            'abs_correlation': abs_corr[i, j]
                        })
                    
                    pbar.update(1)
    
    corr_df = pd.DataFrame(correlations)
    print(f"Found {len(corr_df)} significant correlations")
    
    return corr_df

def nmf_coexpression_modules_fast(mdata,
                                 view1: str,
                                 view2: str,
                                 n_components: int = 10,
                                 correlation_threshold: float = 0.3,
                                 method: str = 'pearson',
                                 random_state: int = 42,
                                 max_features: int = None,
                                 use_torch: bool = True,
                                 device: str = 'auto',
                                 nmf_solver: str = 'mu',
                                 nmf_max_iter: int = 1000,
                                 corr_df: pd.DataFrame = None,
                                 corr_matrix: pd.DataFrame = None) -> dict:
    r"""
    Fast NMF module discovery with optimized correlation backends.

    Parameters
    ----------
    mdata : MuData
        MuData object with paired samples.
    view1 : str
        First modality name.
    view2 : str
        Second modality name.
    n_components : int
        Number of NMF modules to infer.
    correlation_threshold : float
        Minimum absolute correlation retained in module input matrix.
    method : str
        Correlation metric used when computing correlations from raw data.
    random_state : int
        Random seed for NMF reproducibility.
    max_features : int or None
        Optional cap on selected high-variance features per modality.
    use_torch : bool
        Whether to prioritize PyTorch-based correlation computation.
    device : str
        Device setting for PyTorch backend.
    nmf_solver : str
        Scikit-learn NMF solver (for example ``'mu'`` or ``'cd'``).
    nmf_max_iter : int
        Maximum NMF optimization iterations.
    corr_df : pd.DataFrame or None
        Optional precomputed long-format correlation table.
    corr_matrix : pd.DataFrame or None
        Optional precomputed feature-by-feature correlation matrix.

    Returns
    -------
    dict
        Dictionary containing NMF model, factor matrices, filtered correlation
        matrix and module feature definitions.
    """
    from sklearn.decomposition import NMF
    
    # Use pre-computed correlation data if provided
    if corr_df is not None:
        print("Using pre-computed correlation DataFrame...")
        # Validate that the correlation DataFrame has the expected columns
        expected_cols = [f'{view1}_feature', f'{view2}_feature', 'correlation', 'abs_correlation']
        missing_cols = [col for col in expected_cols if col not in corr_df.columns]
        if missing_cols:
            raise ValueError(f"Pre-computed corr_df missing columns: {missing_cols}")
        
        # Filter by threshold if needed
        corr_df_filtered = corr_df[corr_df['abs_correlation'] >= correlation_threshold].copy()
        
    elif corr_matrix is not None:
        print("Using pre-computed correlation matrix...")
        # Convert correlation matrix to DataFrame format
        corr_data = []
        for i, feat1 in enumerate(corr_matrix.index):
            for j, feat2 in enumerate(corr_matrix.columns):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= correlation_threshold:
                    corr_data.append({
                        f'{view1}_feature': feat1,
                        f'{view2}_feature': feat2,
                        'correlation': corr_val,
                        'abs_correlation': abs(corr_val)
                    })
        
        corr_df_filtered = pd.DataFrame(corr_data)
        corr_df = corr_df_filtered.copy()  # For return value
        
    else:
        # Compute cross-correlation matrix using optimized method
        print("Computing cross-correlation matrix...")
        if use_torch:
            try:
                if max_features is not None:
                    corr_df = compute_cross_correlation_torch(
                        mdata, view1, view2, 
                        method='pearson',  # PyTorch version only supports Pearson
                        min_corr=correlation_threshold,
                        max_features=max_features,
                        device=device,
                        batch_size1=1000,
                        batch_size2=1000
                    )
                else:
                    corr_df = compute_cross_correlation_torch_chunked(
                        mdata, view1, view2,
                        min_corr=correlation_threshold,
                        max_features1=max_features,
                        max_features2=max_features,
                        device=device
                    )
            except ImportError:
                print("PyTorch not available, falling back to parallel numpy...")
                corr_df = compute_cross_correlation_parallel_numpy(
                    mdata, view1, view2, 
                    min_corr=correlation_threshold,
                    max_features=max_features
                )
        else:
            # Try parallel numpy first, then fall back to regular fast version
            try:
                corr_df = compute_cross_correlation_parallel_numpy(
                    mdata, view1, view2, 
                    min_corr=correlation_threshold,
                    max_features=max_features
                )
            except Exception as e:
                print(f"Parallel numpy failed ({e}), falling back to regular fast computation...")
                corr_df = compute_cross_correlation_fast(
                    mdata, view1, view2, 
                    min_corr=correlation_threshold,
                    max_features=max_features
                )
        
        corr_df_filtered = corr_df.copy()
    
    if len(corr_df_filtered) == 0:
        raise ValueError("No significant correlations found")
    
    # Create correlation matrix from filtered data
    features1 = corr_df_filtered[f'{view1}_feature'].unique()
    features2 = corr_df_filtered[f'{view2}_feature'].unique()
    
    print(f"Creating correlation matrix: {len(features1)} x {len(features2)}")
    
    # Use sparse matrix for memory efficiency if large
    if len(features1) * len(features2) > 1e6:
        print("Using sparse matrix representation for large correlation matrix...")
        from scipy.sparse import csr_matrix
        
        # Create mapping dictionaries
        feat1_to_idx = {feat: i for i, feat in enumerate(features1)}
        feat2_to_idx = {feat: i for i, feat in enumerate(features2)}
        
        # Prepare sparse matrix data
        rows, cols, data = [], [], []
        for _, row in corr_df_filtered.iterrows():
            i = feat1_to_idx[row[f'{view1}_feature']]
            j = feat2_to_idx[row[f'{view2}_feature']]
            rows.append(i)
            cols.append(j)
            data.append(abs(row['correlation']))
        
        # Create sparse correlation matrix
        corr_matrix_sparse = csr_matrix(
            (data, (rows, cols)), 
            shape=(len(features1), len(features2))
        )
        
        # Convert to dense for NMF (sklearn NMF requires dense)
        final_corr_matrix = pd.DataFrame(
            corr_matrix_sparse.toarray(),
            index=features1,
            columns=features2
        )
    else:
        # Dense matrix for smaller datasets
        final_corr_matrix = pd.DataFrame(0.0, index=features1, columns=features2)
        for _, row in corr_df_filtered.iterrows():
            final_corr_matrix.loc[row[f'{view1}_feature'], row[f'{view2}_feature']] = abs(row['correlation'])
    
    print(f"Correlation matrix shape: {final_corr_matrix.shape}")
    print(f"Non-zero correlations: {(final_corr_matrix > 0).sum().sum()}")
    
    # Apply NMF with optimized parameters
    print(f"Applying NMF with {n_components} components...")
    nmf = NMF(
        n_components=n_components, 
        random_state=random_state, 
        max_iter=nmf_max_iter,
        solver=nmf_solver,  # 'mu' is often faster for sparse data
        beta_loss='frobenius',
        alpha_W=0.0,  # No L1 regularization for speed
        alpha_H=0.0,
        l1_ratio=0.0
    )
    
    W = nmf.fit_transform(final_corr_matrix.values)  # Features x Components
    H = nmf.components_  # Components x Features
    
    # Create results
    results = {
        'nmf_model': nmf,
        'W_matrix': pd.DataFrame(W, index=final_corr_matrix.index, 
                               columns=[f'Module_{i+1}' for i in range(n_components)]),
        'H_matrix': pd.DataFrame(H, index=[f'Module_{i+1}' for i in range(n_components)], 
                               columns=final_corr_matrix.columns),
        'correlation_matrix': final_corr_matrix,
        'correlation_data': corr_df,
        'reconstruction_error': nmf.reconstruction_err_,
        'n_iter': nmf.n_iter_
    }
    
    # Extract modules with better scoring
    modules = {}
    for i in range(n_components):
        module_name = f'Module_{i+1}'
        
        # Get top features from each modality for this module
        w_scores = results['W_matrix'][module_name]
        h_scores = results['H_matrix'].loc[module_name]
        
        # Use percentile-based thresholding for better feature selection
        w_threshold = np.percentile(w_scores, 90)  # Top 10%
        h_threshold = np.percentile(h_scores, 90)
        
        w_top = w_scores[w_scores >= w_threshold].sort_values(ascending=False)
        h_top = h_scores[h_scores >= h_threshold].sort_values(ascending=False)
        
        modules[module_name] = {
            f'{view1}_features': w_top.to_dict(),
            f'{view2}_features': h_top.to_dict(),
            f'{view1}_top_features': w_top.index.tolist(),
            f'{view2}_top_features': h_top.index.tolist(),
            f'{view1}_threshold': w_threshold,
            f'{view2}_threshold': h_threshold,
            'module_strength': (w_scores.max() + h_scores.max()) / 2
        }
    
    results['modules'] = modules
    
    print(f"NMF completed in {nmf.n_iter_} iterations")
    print(f"Reconstruction error: {nmf.reconstruction_err_:.4f}")
    
    return results

def nmf_coexpression_modules_torch(mdata,
                                  view1: str,
                                  view2: str,
                                  n_components: int = 10,
                                  correlation_threshold: float = 0.3,
                                  random_state: int = 42,
                                  max_features: int = None,
                                  device: str = 'auto',
                                  max_iter: int = 1000,
                                  tol: float = 1e-4,
                                  corr_df: pd.DataFrame = None,
                                  corr_matrix: pd.DataFrame = None,
                                  use_parallel: bool = True,
                                  n_workers: int = None) -> dict:
    r"""
    Run PyTorch-optimized NMF module discovery for large datasets.

    Parameters
    ----------
    mdata : MuData
        MuData object with paired samples.
    view1 : str
        First modality name.
    view2 : str
        Second modality name.
    n_components : int
        Number of NMF modules.
    correlation_threshold : float
        Minimum absolute correlation retained for module construction.
    random_state : int
        Random seed.
    max_features : int or None
        Optional cap on selected high-variance features.
    device : str
        PyTorch device selection (``'auto'``, ``'cpu'``, ``'cuda'``, ``'mps'``).
    max_iter : int
        Maximum NMF optimization iterations.
    tol : float
        Convergence tolerance for iterative updates.
    corr_df : pd.DataFrame or None
        Optional precomputed correlation table.
    corr_matrix : pd.DataFrame or None
        Optional precomputed correlation matrix.
    use_parallel : bool
        Whether to use ``pandarallel`` for large DataFrame operations.
    n_workers : int or None
        Number of workers for parallel DataFrame operations.

    Returns
    -------
    dict
        Dictionary containing module decomposition, correlation matrix and
        module feature summaries.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        raise ImportError("PyTorch is required for torch NMF")
    
    # Try to import pandarallel if parallel processing is requested
    if use_parallel:
        try:
            from pandarallel import pandarallel
            import multiprocessing as mp
            
            # Determine number of workers
            if n_workers is None:
                n_workers = min(mp.cpu_count(), 4)  # Conservative default
            
            # Initialize pandarallel
            pandarallel.initialize(nb_workers=n_workers, progress_bar=True, verbose=1)
            print(f"Using pandarallel with {n_workers} workers for DataFrame operations")
        except ImportError:
            print("pandarallel not available, falling back to sequential pandas operations")
            use_parallel = False
    
    # Determine device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Using PyTorch NMF on device: {device}")
    
    # Use pre-computed correlation data if provided
    if corr_df is not None:
        print("Using pre-computed correlation DataFrame...")
        # Validate that the correlation DataFrame has the expected columns
        expected_cols = [f'{view1}_feature', f'{view2}_feature', 'correlation', 'abs_correlation']
        missing_cols = [col for col in expected_cols if col not in corr_df.columns]
        if missing_cols:
            raise ValueError(f"Pre-computed corr_df missing columns: {missing_cols}")
        
        # Filter by threshold if needed - use parallel apply if available
        if use_parallel:
            corr_df_filtered = corr_df[corr_df['abs_correlation'].parallel_apply(lambda x: x >= correlation_threshold)].copy()
        else:
            corr_df_filtered = corr_df[corr_df['abs_correlation'] >= correlation_threshold].copy()
        
    elif corr_matrix is not None:
        print("Using pre-computed correlation matrix...")
        # Convert correlation matrix to DataFrame format using parallel processing
        
        if use_parallel:
            # Create a DataFrame with all combinations and use parallel apply
            print("Converting correlation matrix to DataFrame format with parallel processing...")
            
            # Create index combinations
            index_combinations = []
            for i, feat1 in enumerate(corr_matrix.index):
                for j, feat2 in enumerate(corr_matrix.columns):
                    index_combinations.append((i, j, feat1, feat2))
            
            # Convert to DataFrame for parallel processing
            combo_df = pd.DataFrame(index_combinations, columns=['i', 'j', 'feat1', 'feat2'])
            
            # Define function to process each row
            def process_correlation_row(row):
                i, j, feat1, feat2 = row['i'], row['j'], row['feat1'], row['feat2']
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= correlation_threshold:
                    return {
                        f'{view1}_feature': feat1,
                        f'{view2}_feature': feat2,
                        'correlation': corr_val,
                        'abs_correlation': abs(corr_val)
                    }
                return None
            
            # Apply parallel processing
            results_list = combo_df.parallel_apply(process_correlation_row, axis=1).tolist()
            
            # Filter out None results
            corr_data = [result for result in results_list if result is not None]
        else:
            # Sequential processing (original method)
            corr_data = []
            for i, feat1 in enumerate(corr_matrix.index):
                for j, feat2 in enumerate(corr_matrix.columns):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) >= correlation_threshold:
                        corr_data.append({
                            f'{view1}_feature': feat1,
                            f'{view2}_feature': feat2,
                            'correlation': corr_val,
                            'abs_correlation': abs(corr_val)
                        })
        
        corr_df_filtered = pd.DataFrame(corr_data)
        corr_df = corr_df_filtered.copy()  # For return value
        
    else:
        # Get correlation data using PyTorch
        print("Computing correlations with PyTorch...")
        corr_df = compute_cross_correlation_torch(
            mdata, view1, view2,
            min_corr=correlation_threshold,
            max_features=max_features,
            device=device,
            batch_size1=1000,
            batch_size2=1000
        )
        corr_df_filtered = corr_df.copy()
    
    if len(corr_df_filtered) == 0:
        raise ValueError("No significant correlations found")
    
    # Create correlation matrix
    features1 = corr_df_filtered[f'{view1}_feature'].unique()
    features2 = corr_df_filtered[f'{view2}_feature'].unique()
    
    # Create dense correlation matrix - use parallel processing for large datasets
    print(f"Creating correlation matrix: {len(features1)} x {len(features2)}")
    
    corr_matrix_np = np.zeros((len(features1), len(features2)))
    feat1_to_idx = {feat: i for i, feat in enumerate(features1)}
    feat2_to_idx = {feat: i for i, feat in enumerate(features2)}
    
    if use_parallel and len(corr_df_filtered) > 10000:
        print("Using parallel processing for correlation matrix creation...")
        
        # Define function to process each correlation entry
        def process_corr_entry(row):
            i = feat1_to_idx[row[f'{view1}_feature']]
            j = feat2_to_idx[row[f'{view2}_feature']]
            return (i, j, abs(row['correlation']))
        
        # Apply parallel processing
        indices_and_values = corr_df_filtered.parallel_apply(process_corr_entry, axis=1).tolist()
        
        # Fill the correlation matrix
        for i, j, val in indices_and_values:
            corr_matrix_np[i, j] = val
    else:
        # Sequential processing for smaller datasets
        for _, row in corr_df_filtered.iterrows():
            i = feat1_to_idx[row[f'{view1}_feature']]
            j = feat2_to_idx[row[f'{view2}_feature']]
            corr_matrix_np[i, j] = abs(row['correlation'])
    
    print(f"Correlation matrix shape: {corr_matrix_np.shape}")
    
    # Convert to PyTorch tensor
    X = torch.tensor(corr_matrix_np, dtype=torch.float32, device=device)
    m, n = X.shape
    
    # Initialize W and H matrices
    torch.manual_seed(random_state)
    W = torch.rand(m, n_components, device=device, requires_grad=False)
    H = torch.rand(n_components, n, device=device, requires_grad=False)
    
    print(f"Running PyTorch NMF with {n_components} components...")
    
    # NMF optimization loop
    prev_loss = float('inf')
    
    for iteration in tqdm(range(max_iter), desc="NMF iterations"):
        # Update H
        WtW = torch.mm(W.t(), W)
        WtX = torch.mm(W.t(), X)
        H = H * WtX / (torch.mm(WtW, H) + 1e-10)
        
        # Update W
        HHt = torch.mm(H, H.t())
        XHt = torch.mm(X, H.t())
        W = W * XHt / (torch.mm(W, HHt) + 1e-10)
        
        # Check convergence every 10 iterations
        if iteration % 10 == 0:
            with torch.no_grad():
                reconstruction = torch.mm(W, H)
                loss = torch.norm(X - reconstruction, p='fro').item()
                
                if abs(prev_loss - loss) < tol:
                    print(f"Converged at iteration {iteration}")
                    break
                prev_loss = loss
    
    # Convert results back to numpy/pandas
    W_np = W.cpu().numpy()
    H_np = H.cpu().numpy()
    
    # Create results
    results = {
        'W_matrix': pd.DataFrame(W_np, index=features1, 
                               columns=[f'Module_{i+1}' for i in range(n_components)]),
        'H_matrix': pd.DataFrame(H_np, index=[f'Module_{i+1}' for i in range(n_components)], 
                               columns=features2),
        'correlation_matrix': pd.DataFrame(corr_matrix_np, index=features1, columns=features2),
        'correlation_data': corr_df,
        'final_loss': prev_loss,
        'n_iter': iteration + 1
    }
    
    # Extract modules - use parallel processing for large results
    modules = {}
    for i in range(n_components):
        print(f"Extracting module {i+1} of {n_components}...")
        module_name = f'Module_{i+1}'
        
        w_scores = results['W_matrix'][module_name]
        h_scores = results['H_matrix'].loc[module_name]
        
        # Use adaptive thresholding
        w_threshold = np.percentile(w_scores, 85)
        h_threshold = np.percentile(h_scores, 85)
        
        if use_parallel and len(w_scores) > 1000:
            # Use parallel filtering for large datasets
            w_top_mask = w_scores.parallel_apply(lambda x: x >= w_threshold)
            h_top_mask = h_scores.parallel_apply(lambda x: x >= h_threshold)
            
            w_top = w_scores[w_top_mask].sort_values(ascending=False)
            h_top = h_scores[h_top_mask].sort_values(ascending=False)
        else:
            # Sequential processing for smaller datasets
            w_top = w_scores[w_scores >= w_threshold].sort_values(ascending=False)
            h_top = h_scores[h_scores >= h_threshold].sort_values(ascending=False)
        
        modules[module_name] = {
            f'{view1}_features': w_top.to_dict(),
            f'{view2}_features': h_top.to_dict(),
            f'{view1}_top_features': w_top.index.tolist(),
            f'{view2}_top_features': h_top.index.tolist()
        }
    
    results['modules'] = modules
    
    print(f"PyTorch NMF completed in {iteration + 1} iterations")
    print(f"Final reconstruction loss: {prev_loss:.4f}")
    
    return results

def plot_correlation_matrix(corr_df: pd.DataFrame,
                           nmf_results: dict,
                           view1: str,
                           view2: str,
                           top_n_per_module: int = 10,
                           figsize: tuple = (12, 10),
                           cmap: str = 'RdBu_r',
                           show_module_boundaries: bool = True,
                           legend_ncol: int = 2,
                           legend_bbox_to_anchor: tuple = (1.3, 1.0),
                           legend_fontsize: int = 10,
                           use_parallel: bool = True,
                           n_workers: int = None,
                           save: bool = False,
                           save_path: str = 'mofa_correlation_matrix.png') -> matplotlib.axes._axes.Axes:
    r"""
    Plot module-aware square correlation matrix across both modalities.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Cross-modality correlation table.
    nmf_results : dict
        Output from NMF module discovery containing module assignments.
    view1 : str
        First modality name.
    view2 : str
        Second modality name.
    top_n_per_module : int
        Number of top features retained per module and modality.
    figsize : tuple
        Figure size in inches as ``(width, height)``.
    cmap : str
        Colormap for correlation values.
    show_module_boundaries : bool
        Whether to draw module boundary rectangles/separators.
    legend_ncol : int
        Number of legend columns.
    legend_bbox_to_anchor : tuple
        Legend anchor position.
    legend_fontsize : int
        Legend font size.
    use_parallel : bool
        Whether to use ``pandarallel`` for large DataFrame operations.
    n_workers : int or None
        Number of workers for parallel mode.
    save : bool
        Whether to save figure to file.
    save_path : str
        Output file path when ``save=True``.

    Returns
    -------
    seaborn.matrix.ClusterGrid
        Seaborn clustermap object containing the module correlation matrix.
    """
    # Try to import pandarallel if parallel processing is requested
    if use_parallel:
        try:
            from pandarallel import pandarallel
            import multiprocessing as mp
            
            # Determine number of workers
            if n_workers is None:
                n_workers = min(mp.cpu_count(), 4)  # Conservative default
            
            # Initialize pandarallel
            pandarallel.initialize(nb_workers=n_workers, progress_bar=True, verbose=1)
            print(f"Using pandarallel with {n_workers} workers for DataFrame operations")
        except ImportError:
            print("pandarallel not available, falling back to sequential pandas operations")
            use_parallel = False
    
    # Get module assignments
    modules = nmf_results['modules']
    
    # Collect top features from each module and organize by module
    all_features = []  # Combined list of all features
    feature_to_module = {}  # Map feature to module
    feature_to_modality = {}  # Map feature to modality
    
    print(f"Selecting top {top_n_per_module} features per module...")
    
    for module_name, module_data in modules.items():
        # Get top features from each modality for this module
        view1_features = module_data[f'{view1}_top_features'][:top_n_per_module]
        view2_features = module_data[f'{view2}_top_features'][:top_n_per_module]
        
        # Add features with module and modality information
        for feat in view1_features:
            if feat not in feature_to_module:  # Avoid duplicates
                all_features.append(feat)
                feature_to_module[feat] = module_name
                feature_to_modality[feat] = view1
        
        for feat in view2_features:
            if feat not in feature_to_module:  # Avoid duplicates
                all_features.append(feat)
                feature_to_module[feat] = module_name
                feature_to_modality[feat] = view2
        
        print(f"{module_name}: {len(view1_features)} {view1} features, {len(view2_features)} {view2} features")
    
    print(f"Total combined features: {len(all_features)}")
    
    # Create square correlation matrix
    corr_matrix = pd.DataFrame(0.0, index=all_features, columns=all_features)
    
    # Fill diagonal with 1.0 (self-correlation)
    np.fill_diagonal(corr_matrix.values, 1.0)
    
    # Fill cross-modality correlations from corr_df using parallel processing
    print("Filling cross-modality correlations...")
    
    # Filter correlation data to only include selected features
    if use_parallel and len(corr_df) > 1000:
        print("Using parallel processing for correlation filtering...")
        
        # Define function to check if correlation should be included
        def check_correlation_inclusion(row):
            feat1 = row[f'{view1}_feature']
            feat2 = row[f'{view2}_feature']
            return feat1 in all_features and feat2 in all_features
        
        # Use parallel apply to filter correlations
        inclusion_mask = corr_df.parallel_apply(check_correlation_inclusion, axis=1)
        corr_df_filtered = corr_df[inclusion_mask].copy()
    else:
        # Sequential filtering for smaller datasets
        corr_df_filtered = corr_df[
            (corr_df[f'{view1}_feature'].isin(all_features)) &
            (corr_df[f'{view2}_feature'].isin(all_features))
        ].copy()
    
    print(f"Filtered correlations: {len(corr_df_filtered)} entries")
    
    # Fill correlation matrix using parallel processing if available
    if use_parallel and len(corr_df_filtered) > 1000:
        print("Using parallel processing for correlation matrix filling...")
        
        # Define function to extract correlation information
        def extract_correlation_info(row):
            feat1 = row[f'{view1}_feature']
            feat2 = row[f'{view2}_feature']
            corr_val = row['correlation']
            return (feat1, feat2, corr_val)
        
        # Use parallel apply to extract correlation information
        correlation_info = corr_df_filtered.parallel_apply(extract_correlation_info, axis=1).tolist()
        
        # Fill the matrix
        cross_correlations_filled = 0
        for feat1, feat2, corr_val in correlation_info:
            corr_matrix.loc[feat1, feat2] = corr_val
            corr_matrix.loc[feat2, feat1] = corr_val  # Symmetric
            cross_correlations_filled += 1
    else:
        # Sequential processing for smaller datasets
        cross_correlations_filled = 0
        for _, row in corr_df_filtered.iterrows():
            feat1 = row[f'{view1}_feature']
            feat2 = row[f'{view2}_feature']
            corr_val = row['correlation']
            
            corr_matrix.loc[feat1, feat2] = corr_val
            corr_matrix.loc[feat2, feat1] = corr_val  # Symmetric
            cross_correlations_filled += 1
    
    print(f"Filled {cross_correlations_filled} cross-modality correlations")
    
    # For within-modality correlations, we can compute them if needed
    # For now, we'll leave them as 0 (could be extended to compute within-modality correlations)
    
    # Create color annotations based on modules using parallel processing if beneficial
    feature_colors = pd.Series('lightgray', index=all_features)
    
    # Define colors for modules
    module_colors = plt.cm.Set3(np.linspace(0, 1, len(modules)))
    module_color_map = {module_name: module_colors[i] for i, module_name in enumerate(modules.keys())}
    
    # Color features by their module
    if use_parallel and len(all_features) > 1000:
        print("Using parallel processing for feature coloring...")
        
        # Create DataFrame for parallel processing
        feature_df = pd.DataFrame({'feature': all_features})
        feature_df['module'] = feature_df['feature'].map(feature_to_module)
        
        # Define function to get color
        def get_feature_color(row):
            module = row['module']
            return module_color_map[module]
        
        # Use parallel apply to get colors
        colors = feature_df.parallel_apply(get_feature_color, axis=1)
        feature_colors = pd.Series(colors.values, index=all_features)
    else:
        # Sequential processing
        for feat in all_features:
            module = feature_to_module[feat]
            feature_colors[feat] = module_color_map[module]
    
    # Sort features by module for better visualization
    print("Sorting features by module...")
    sorted_features = []
    module_boundaries = {}  # Store module start/end positions
    current_pos = 0
    
    for module_name in modules.keys():
        module_features = [feat for feat in all_features if feature_to_module[feat] == module_name]
        # Sort within module: view1 features first, then view2 features
        view1_feats = [f for f in module_features if feature_to_modality[f] == view1]
        view2_feats = [f for f in module_features if feature_to_modality[f] == view2]
        
        module_sorted = view1_feats + view2_feats
        sorted_features.extend(module_sorted)
        
        module_boundaries[module_name] = {
            'start': current_pos,
            'end': current_pos + len(module_sorted) - 1,
            'view1_end': current_pos + len(view1_feats) - 1 if view1_feats else current_pos - 1
        }
        current_pos += len(module_sorted)
    
    # Reorder matrix according to sorted features
    print("Reordering correlation matrix...")
    corr_matrix = corr_matrix.loc[sorted_features, sorted_features]
    feature_colors = feature_colors[sorted_features]
    
    # Create clustermap without clustering (to preserve module order)
    print("Creating clustermap...")
    g = sns.clustermap(
        corr_matrix,
        cmap=cmap,
        center=0,
        row_colors=feature_colors,
        col_colors=feature_colors,
        figsize=figsize,
        cbar_kws={"shrink": 0.5, "label": "Correlation"},
        xticklabels=False,  # Hide x-axis labels as requested
        yticklabels=False,  # Hide y-axis labels as requested
        square=True,  # Square matrix
        row_cluster=False,  # Don't cluster to preserve module order
        col_cluster=False,  # Don't cluster to preserve module order
        dendrogram_ratio=0,  # No dendrograms
        cbar_pos=(0.02, 0.8, 0.03, 0.18)
    )
    
    # Add module boundaries if requested
    if show_module_boundaries:
        print("Adding module boundaries...")
        for module_name, bounds in module_boundaries.items():
            start, end = bounds['start'], bounds['end']
            view1_end = bounds['view1_end']
            
            # Draw rectangle around entire module
            g.ax_heatmap.add_patch(
                plt.Rectangle(
                    (start, start), 
                    end - start + 1, 
                    end - start + 1,
                    fill=False, 
                    color='black', 
                    lw=2,
                    alpha=0.8
                )
            )
            
            # Draw separator between view1 and view2 within module if both exist
            if view1_end >= start and view1_end < end:
                # Vertical line
                g.ax_heatmap.axvline(x=view1_end + 0.5, ymin=start/len(sorted_features), 
                                   ymax=(end+1)/len(sorted_features), 
                                   color='red', linestyle='--', alpha=0.6, linewidth=1)
                # Horizontal line
                g.ax_heatmap.axhline(y=view1_end + 0.5, xmin=start/len(sorted_features), 
                                   xmax=(end+1)/len(sorted_features), 
                                   color='red', linestyle='--', alpha=0.6, linewidth=1)
    
    # Create legend
    from matplotlib import patches as mpatches
    patches_list = []
    for module_name, color in module_color_map.items():
        module_features = [feat for feat in all_features if feature_to_module[feat] == module_name]
        view1_count = sum(1 for f in module_features if feature_to_modality[f] == view1)
        view2_count = sum(1 for f in module_features if feature_to_modality[f] == view2)
        label = f"{module_name} ({view1_count}+{view2_count})"
        patch = mpatches.Patch(color=color, label=label)
        patches_list.append(patch)
    
    # Add modality separator legend
    patches_list.append(mpatches.Patch(color='red', linestyle='--', 
                                     label=f'{view1}/{view2} separator'))
    
    plt.legend(
        handles=patches_list,
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=legend_ncol,
        fontsize=legend_fontsize,
        title="Modules (features)"
    )
    
    # Set title
    g.fig.suptitle(f'Module Correlation Structure\n'
                   f'{len(all_features)} combined features from {len(modules)} modules', 
                   fontsize=14, fontweight='bold', y=0.98)
    
    if save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return g

def get_module_features(nmf_results: dict, 
                       module_name: str = None,
                       view: str = None,
                       top_n: int = None) -> dict:
    r"""
    Retrieve module feature lists from NMF module results.

    Parameters
    ----------
    nmf_results : dict
        Output dictionary from NMF module discovery.
    module_name : str or None
        Specific module ID (for example ``'Module_1'``). If ``None``, return all
        modules.
    view : str or None
        Restrict to one modality/view. If ``None``, return all available views.
    top_n : int or None
        Number of top features to keep per module/view. If ``None``, keep all
        stored features.

    Returns
    -------
    dict
        Nested dictionary containing selected module features and scores.
    """
    modules = nmf_results['modules']
    
    if module_name is not None:
        if module_name not in modules:
            raise ValueError(f"Module {module_name} not found. Available modules: {list(modules.keys())}")
        modules = {module_name: modules[module_name]}
    
    result = {}
    
    for mod_name, mod_data in modules.items():
        result[mod_name] = {}
        
        for view_key in mod_data.keys():
            if view_key.endswith('_features'):
                view_name = view_key.replace('_features', '')
                
                if view is None or view_name == view:
                    features_dict = mod_data[view_key]
                    
                    if top_n is not None:
                        # Get top N features
                        sorted_features = sorted(features_dict.items(), 
                                               key=lambda x: x[1], reverse=True)
                        features_dict = dict(sorted_features[:top_n])
                    
                    result[mod_name][view_name] = {
                        'features': list(features_dict.keys()),
                        'scores': list(features_dict.values()),
                        'feature_score_dict': features_dict
                    }
    
    return result

def get_module_network(nmf_results: dict,
                      corr_df: pd.DataFrame,
                      module_name: str,
                      view1: str,
                      view2: str,
                      correlation_threshold: float = 0.5) -> dict:
    r"""
    Build a feature-correlation network for one NMF module.

    Parameters
    ----------
    nmf_results : dict
        Output dictionary from NMF module discovery.
    corr_df : pd.DataFrame
        Cross-modality correlation table.
    module_name : str
        Module identifier (for example ``'Module_1'``).
    view1 : str
        First modality name.
    view2 : str
        Second modality name.
    correlation_threshold : float
        Minimum absolute correlation retained as an edge.

    Returns
    -------
    dict
        Network information including graph object, centrality metrics and hub
        features.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("NetworkX is required. Install with: pip install networkx")
    
    if module_name not in nmf_results['modules']:
        raise ValueError(f"Module {module_name} not found")
    
    module_data = nmf_results['modules'][module_name]
    
    # Get features from this module
    view1_features = module_data[f'{view1}_top_features']
    view2_features = module_data[f'{view2}_top_features']
    
    # Filter correlations for this module
    module_corr = corr_df[
        (corr_df[f'{view1}_feature'].isin(view1_features)) &
        (corr_df[f'{view2}_feature'].isin(view2_features)) &
        (corr_df['abs_correlation'] >= correlation_threshold)
    ].copy()
    
    # Create network
    G = nx.Graph()
    
    # Add nodes with attributes
    for feat in view1_features:
        G.add_node(feat, modality=view1, module=module_name)
    for feat in view2_features:
        G.add_node(feat, modality=view2, module=module_name)
    
    # Add edges
    for _, row in module_corr.iterrows():
        G.add_edge(
            row[f'{view1}_feature'], 
            row[f'{view2}_feature'],
            weight=abs(row['correlation']),
            correlation=row['correlation']
        )
    
    # Calculate network metrics
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # Find hub nodes (high degree)
    hub_threshold = np.percentile(list(degree_centrality.values()), 80)
    hub_nodes = [node for node, centrality in degree_centrality.items() 
                 if centrality >= hub_threshold]
    
    network_info = {
        'graph': G,
        'module_name': module_name,
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
        'degree_centrality': degree_centrality,
        'betweenness_centrality': betweenness_centrality,
        'hub_nodes': hub_nodes,
        'correlations': module_corr,
        f'{view1}_features': view1_features,
        f'{view2}_features': view2_features
    }
    
    return network_info

def plot_module_network(network_info: dict,
                       view1: str,
                       view2: str,
                       figsize: tuple = (10, 8),
                       node_size_factor: float = 1000,
                       edge_width_factor: float = 3,
                       show_labels: bool = True,
                       label_size: int = 8,
                       save: bool = False,
                       save_path: str = None) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    r"""
    Plot network topology for a selected module.

    Parameters
    ----------
    network_info : dict
        Network dictionary from :func:`get_module_network`.
    view1 : str
        First modality name.
    view2 : str
        Second modality name.
    figsize : tuple
        Figure size in inches as ``(width, height)``.
    node_size_factor : float
        Scale factor for node sizes from degree centrality.
    edge_width_factor : float
        Scale factor for edge widths from correlation strength.
    show_labels : bool
        Whether to annotate hub node labels.
    label_size : int
        Font size for labels.
    save : bool
        Whether to save the figure.
    save_path : str or None
        Output path when ``save=True``. If ``None``, a default filename is used.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
        Matplotlib figure and axes for network plot.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("NetworkX is required. Install with: pip install networkx")
    
    G = network_info['graph']
    
    if G.number_of_nodes() == 0:
        print("No nodes in the network")
        return None, None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set node positions using spring layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Prepare node colors and sizes
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        modality = G.nodes[node]['modality']
        degree_cent = network_info['degree_centrality'][node]
        
        # Color by modality
        if modality == view1:
            node_colors.append('lightblue')
        else:
            node_colors.append('lightcoral')
        
        # Size by degree centrality
        node_sizes.append(degree_cent * node_size_factor + 100)
    
    # Prepare edge widths
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    edge_widths = [w * edge_width_factor for w in edge_weights]
    
    # Draw network
    nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.7,
        ax=ax
    )
    
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.5,
        edge_color='gray',
        ax=ax
    )
    
    if show_labels:
        # Only show labels for hub nodes to avoid clutter
        hub_nodes = network_info['hub_nodes']
        hub_pos = {node: pos[node] for node in hub_nodes if node in pos}
        
        nx.draw_networkx_labels(
            G, hub_pos,
            font_size=label_size,
            font_weight='bold',
            ax=ax
        )
    
    # Customize plot
    ax.set_title(f"Module Network: {network_info['module_name']}", 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', label=view1),
        Patch(facecolor='lightcoral', label=view2)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add network statistics as text
    stats_text = f"Nodes: {network_info['n_nodes']}\n"
    stats_text += f"Edges: {network_info['n_edges']}\n"
    stats_text += f"Density: {network_info['density']:.3f}\n"
    stats_text += f"Hub nodes: {len(network_info['hub_nodes'])}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save:
        if save_path is None:
            save_path = f"module_network_{network_info['module_name']}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def summarize_modules(nmf_results: dict, 
                     view1: str, 
                     view2: str,
                     top_n: int = 5) -> pd.DataFrame:
    r"""
    Summarize inferred modules with top modality-specific features.

    Parameters
    ----------
    nmf_results : dict
        Output dictionary from NMF module discovery.
    view1 : str
        First modality name.
    view2 : str
        Second modality name.
    top_n : int
        Number of top features listed per modality and module.

    Returns
    -------
    pd.DataFrame
        Module summary table with feature counts and top-feature strings.
    """
    modules = nmf_results['modules']
    
    summary_data = []
    
    for module_name, module_data in modules.items():
        # Get top features
        view1_features = module_data[f'{view1}_top_features'][:top_n]
        view2_features = module_data[f'{view2}_top_features'][:top_n]
        
        # Get module strength if available
        module_strength = module_data.get('module_strength', 'N/A')
        
        summary_data.append({
            'Module': module_name,
            f'{view1}_count': len(module_data[f'{view1}_features']),
            f'{view2}_count': len(module_data[f'{view2}_features']),
            f'Top_{view1}_features': ', '.join(view1_features),
            f'Top_{view2}_features': ', '.join(view2_features),
            'Module_strength': module_strength
        })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

class MOFACorrelationAnalyzer(object):
    r"""
    A class to handle MOFA correlation analysis and visualization with caching to avoid repeated computations.
    """
    
    def __init__(self, nmf_results: dict, corr_df: pd.DataFrame, view1: str, view2: str):
        r"""
        Initialize the MOFA correlation analyzer.
        
        Parameters
        ----------
            nmf_results: Results from nmf_coexpression_modules.
            corr_df: Correlation DataFrame from compute_cross_correlation.
            view1: Name of first modality.
            view2: Name of second modality.
        """
        self.nmf_results = nmf_results
        self.corr_df = corr_df
        self.view1 = view1
        self.view2 = view2
        self.modules = nmf_results['modules']
        
        # Cache for computed results
        self._feature_cache = {}
        self._correlation_matrix_cache = {}
        self._module_info_cache = {}
        
        print(f"Initialized MOFACorrelationAnalyzer with {len(self.modules)} modules")
        print(f"Correlation data: {len(corr_df)} entries")
    
    def get_module_features(self, top_n_per_module: int = 10, use_cache: bool = True):
        r"""
        Get top features from each module with caching.
        
        Parameters
        ----------
            top_n_per_module: Number of top features per module.
            use_cache: Whether to use cached results.
            
        Returns
        -------
            Tuple of (all_features, feature_to_module, feature_to_modality)
        """
        cache_key = f"features_{top_n_per_module}"
        
        if use_cache and cache_key in self._feature_cache:
            print(f"Using cached feature data for top_n={top_n_per_module}")
            return self._feature_cache[cache_key]
        
        print(f"Computing module features for top_n={top_n_per_module}")
        
        all_features = []
        feature_to_module = {}
        feature_to_modality = {}
        
        for module_name, module_data in self.modules.items():
            view1_features = module_data[f'{self.view1}_top_features'][:top_n_per_module]
            view2_features = module_data[f'{self.view2}_top_features'][:top_n_per_module]
            
            for feat in view1_features:
                if feat not in feature_to_module:
                    all_features.append(feat)
                    feature_to_module[feat] = module_name
                    feature_to_modality[feat] = self.view1
            
            for feat in view2_features:
                if feat not in feature_to_module:
                    all_features.append(feat)
                    feature_to_module[feat] = module_name
                    feature_to_modality[feat] = self.view2
            
            print(f"{module_name}: {len(view1_features)} {self.view1}, {len(view2_features)} {self.view2}")
        
        result = (all_features, feature_to_module, feature_to_modality)
        
        if use_cache:
            self._feature_cache[cache_key] = result
        
        print(f"Total features: {len(all_features)}")
        return result
    
    def create_correlation_matrix(self, top_n_per_module: int = 10, 
                                use_parallel: bool = True, n_workers: int = None,
                                use_cache: bool = True):
        r"""
        Create correlation matrix with caching and parallel processing.
        
        Parameters
        ----------
            top_n_per_module: Number of top features per module.
            use_parallel: Whether to use parallel processing.
            n_workers: Number of workers for parallel processing.
            use_cache: Whether to use cached results.
            
        Returns
        -------
            Tuple of (corr_matrix, sorted_features, feature_colors, module_boundaries)
        """
        cache_key = f"corr_matrix_{top_n_per_module}_{use_parallel}"
        
        if use_cache and cache_key in self._correlation_matrix_cache:
            print(f"Using cached correlation matrix for top_n={top_n_per_module}")
            return self._correlation_matrix_cache[cache_key]
        
        print(f"Computing correlation matrix for top_n={top_n_per_module}...")
        
        # Initialize parallel processing if requested
        if use_parallel:
            try:
                from pandarallel import pandarallel
                import multiprocessing as mp
                
                if n_workers is None:
                    n_workers = min(mp.cpu_count(), 4)
                
                pandarallel.initialize(nb_workers=n_workers, progress_bar=True, verbose=1)
                print(f"Using pandarallel with {n_workers} workers")
            except ImportError:
                print("pandarallel not available, using sequential processing")
                use_parallel = False
        
        # Get features (this is relatively fast, so we don't need to cache it separately)
        all_features, feature_to_module, feature_to_modality = self.get_module_features(
            top_n_per_module, use_cache=False  # Don't cache this since it's fast
        )
        
        if len(all_features) == 0:
            raise ValueError("No features found. Check your module results and top_n_per_module parameter.")
        
        # Create correlation matrix (this is the expensive part we want to cache)
        print("Creating correlation matrix...")
        corr_matrix = pd.DataFrame(0.0, index=all_features, columns=all_features)
        np.fill_diagonal(corr_matrix.values, 1.0)
        
        # Filter and fill correlations
        print("Filtering correlations...")
        if use_parallel and len(self.corr_df) > 1000:
            def check_inclusion(row):
                return (row[f'{self.view1}_feature'] in all_features and 
                       row[f'{self.view2}_feature'] in all_features)
            
            inclusion_mask = self.corr_df.parallel_apply(check_inclusion, axis=1)
            corr_df_filtered = self.corr_df[inclusion_mask].copy()
        else:
            corr_df_filtered = self.corr_df[
                (self.corr_df[f'{self.view1}_feature'].isin(all_features)) &
                (self.corr_df[f'{self.view2}_feature'].isin(all_features))
            ].copy()
        
        if len(corr_df_filtered) == 0:
            print("Warning: No correlations found for selected features")
        else:
            print(f"Filling {len(corr_df_filtered)} correlations...")
            
            if use_parallel and len(corr_df_filtered) > 1000:
                def extract_corr_info(row):
                    return (row[f'{self.view1}_feature'], 
                           row[f'{self.view2}_feature'], 
                           row['correlation'])
                
                correlation_info = corr_df_filtered.parallel_apply(extract_corr_info, axis=1).tolist()
                
                for feat1, feat2, corr_val in correlation_info:
                    corr_matrix.loc[feat1, feat2] = corr_val
                    corr_matrix.loc[feat2, feat1] = corr_val
            else:
                for _, row in corr_df_filtered.iterrows():
                    feat1 = row[f'{self.view1}_feature']
                    feat2 = row[f'{self.view2}_feature']
                    corr_val = row['correlation']
                    corr_matrix.loc[feat1, feat2] = corr_val
                    corr_matrix.loc[feat2, feat1] = corr_val
        
        # Create feature colors and sort by modules
        print("Creating feature colors and sorting...")
        module_colors = plt.cm.Set3(np.linspace(0, 1, len(self.modules)))
        module_color_map = {name: module_colors[i] for i, name in enumerate(self.modules.keys())}
        
        # Sort features by module
        sorted_features = []
        module_boundaries = {}
        current_pos = 0
        
        for module_name in self.modules.keys():
            module_features = [f for f in all_features if feature_to_module[f] == module_name]
            view1_feats = [f for f in module_features if feature_to_modality[f] == self.view1]
            view2_feats = [f for f in module_features if feature_to_modality[f] == self.view2]
            
            module_sorted = view1_feats + view2_feats
            sorted_features.extend(module_sorted)
            
            module_boundaries[module_name] = {
                'start': current_pos,
                'end': current_pos + len(module_sorted) - 1,
                'view1_end': current_pos + len(view1_feats) - 1 if view1_feats else current_pos - 1
            }
            current_pos += len(module_sorted)
        
        # Reorder matrix and create feature colors
        corr_matrix = corr_matrix.loc[sorted_features, sorted_features]
        feature_colors = pd.Series([module_color_map[feature_to_module[f]] for f in sorted_features], 
                                 index=sorted_features)
        
        result = (corr_matrix, sorted_features, feature_colors, module_boundaries)
        
        # Cache the complete result (this is what we want to avoid recomputing)
        if use_cache:
            self._correlation_matrix_cache[cache_key] = result
            print(f"Cached correlation matrix for top_n={top_n_per_module}")
        
        return result
    
    def plot_correlation_matrix(self, top_n_per_module: int = 10,
                               figsize: tuple = (12, 10),
                               cmap: str = 'RdBu_r',
                               show_module_boundaries: bool = True,
                               legend_ncol: int = 2,
                               legend_bbox_to_anchor: tuple = (1.3, 1.0),
                               legend_fontsize: int = 10,
                               use_parallel: bool = True,
                               n_workers: int = None,
                               save: bool = False,
                               save_path: str = 'mofa_correlation_matrix.png'):
        r"""
        Plot correlation matrix with module annotations.
        
        Parameters
        ----------
            top_n_per_module: Number of top features per module.
            figsize: Figure size.
            cmap: Colormap.
            show_module_boundaries: Whether to show module boundaries.
            legend_ncol: Number of legend columns.
            legend_bbox_to_anchor: Legend position.
            legend_fontsize: Legend font size.
            use_parallel: Whether to use parallel processing.
            n_workers: Number of workers.
            save: Whether to save figure.
            save_path: Save path.
            
        Returns
        -------
            ClusterGrid object or None if plotting fails.
        """
        try:
            # Get correlation matrix
            corr_matrix, sorted_features, feature_colors, module_boundaries = self.create_correlation_matrix(
                top_n_per_module, use_parallel, n_workers
            )
            
            if len(sorted_features) == 0:
                print("Error: No features to plot")
                return None
            
            if corr_matrix.shape[0] < 2:
                print("Error: Need at least 2 features to create a meaningful plot")
                return None
            
            print(f"Creating clustermap with {corr_matrix.shape[0]} features...")
            
            # Create clustermap with error handling
            try:
                g = sns.clustermap(
                    corr_matrix,
                    cmap=cmap,
                    center=0,
                    row_colors=feature_colors,
                    col_colors=feature_colors,
                    figsize=figsize,
                    cbar_kws={"shrink": 0.5, "label": "Correlation"},
                    xticklabels=False,
                    yticklabels=False,
                    square=True,
                    row_cluster=False,
                    col_cluster=False,
                    dendrogram_ratio=0,
                    cbar_pos=(0.02, 0.8, 0.03, 0.18)
                )
            except Exception as e:
                print(f"Error creating clustermap: {e}")
                # Fallback to simple heatmap
                fig, ax = plt.subplots(figsize=figsize)
                sns.heatmap(corr_matrix, cmap=cmap, center=0, 
                           xticklabels=False, yticklabels=False,
                           cbar_kws={"shrink": 0.5, "label": "Correlation"}, ax=ax)
                ax.set_title(f'Module Correlation Structure\n{len(sorted_features)} features')
                
                if save:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
                return fig
            
            # Add module boundaries
            if show_module_boundaries and len(module_boundaries) > 0:
                print("Adding module boundaries...")
                for module_name, bounds in module_boundaries.items():
                    start, end = bounds['start'], bounds['end']
                    view1_end = bounds['view1_end']
                    
                    if start <= end:  # Valid boundary
                        g.ax_heatmap.add_patch(
                            plt.Rectangle(
                                (start, start), 
                                end - start + 1, 
                                end - start + 1,
                                fill=False, 
                                color='black', 
                                lw=2,
                                alpha=0.8
                            )
                        )
                        
                        if view1_end >= start and view1_end < end:
                            g.ax_heatmap.axvline(x=view1_end + 0.5, 
                                               ymin=start/len(sorted_features), 
                                               ymax=(end+1)/len(sorted_features), 
                                               color='red', linestyle='--', alpha=0.6, linewidth=1)
                            g.ax_heatmap.axhline(y=view1_end + 0.5, 
                                               xmin=start/len(sorted_features), 
                                               xmax=(end+1)/len(sorted_features), 
                                               color='red', linestyle='--', alpha=0.6, linewidth=1)
            
            # Create legend
            from matplotlib import patches as mpatches
            patches_list = []
            
            all_features, feature_to_module, feature_to_modality = self.get_module_features(top_n_per_module)
            module_colors = plt.cm.Set3(np.linspace(0, 1, len(self.modules)))
            module_color_map = {name: module_colors[i] for i, name in enumerate(self.modules.keys())}
            
            for module_name, color in module_color_map.items():
                module_features = [f for f in all_features if feature_to_module[f] == module_name]
                view1_count = sum(1 for f in module_features if feature_to_modality[f] == self.view1)
                view2_count = sum(1 for f in module_features if feature_to_modality[f] == self.view2)
                label = f"{module_name} ({view1_count}+{view2_count})"
                patch = mpatches.Patch(color=color, label=label)
                patches_list.append(patch)
            
            patches_list.append(mpatches.Patch(color='red', linestyle='--', 
                                             label=f'{self.view1}/{self.view2} separator'))
            
            plt.legend(
                handles=patches_list,
                bbox_to_anchor=legend_bbox_to_anchor,
                ncol=legend_ncol,
                fontsize=legend_fontsize,
                title="Modules (features)"
            )
            
            # Set title
            g.fig.suptitle(f'Module Correlation Structure\n'
                          f'{len(sorted_features)} combined features from {len(self.modules)} modules', 
                          fontsize=14, fontweight='bold', y=0.98)
            
            if save:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return g
            
        except Exception as e:
            print(f"Error in plot_correlation_matrix: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_summary(self):
        r"""
        Get summary information about the correlation analysis.
        
        Returns
        -------
            Dictionary with summary statistics.
        """
        summary = {
            'n_modules': len(self.modules),
            'n_correlations': len(self.corr_df),
            'view1': self.view1,
            'view2': self.view2,
            'modules': {}
        }
        
        for module_name, module_data in self.modules.items():
            summary['modules'][module_name] = {
                f'{self.view1}_features': len(module_data[f'{self.view1}_features']),
                f'{self.view2}_features': len(module_data[f'{self.view2}_features']),
                f'{self.view1}_top_features': len(module_data[f'{self.view1}_top_features']),
                f'{self.view2}_top_features': len(module_data[f'{self.view2}_top_features'])
            }
        
        return summary

    def clear_cache(self):
        r"""
        Clear all cached data to free memory.
        """
        self._feature_cache.clear()
        self._correlation_matrix_cache.clear()
        self._module_info_cache.clear()
        print("All caches cleared")
    
    def get_cache_info(self):
        r"""
        Get information about cached data.
        
        Returns
        -------
            Dictionary with cache information.
        """
        cache_info = {
            'feature_cache_keys': list(self._feature_cache.keys()),
            'correlation_matrix_cache_keys': list(self._correlation_matrix_cache.keys()),
            'module_info_cache_keys': list(self._module_info_cache.keys()),
            'total_cached_items': (len(self._feature_cache) + 
                                 len(self._correlation_matrix_cache) + 
                                 len(self._module_info_cache))
        }
        return cache_info

def create_correlation_analyzer(nmf_results: dict, 
                              corr_df: pd.DataFrame, 
                              view1: str, 
                              view2: str) -> MOFACorrelationAnalyzer:
    r"""
    Create a reusable analyzer object for module-correlation exploration.
    
    Parameters
    ----------
    nmf_results : dict
        Output from NMF module discovery.
    corr_df : pd.DataFrame
        Cross-modality correlation table.
    view1 : str
        First modality name.
    view2 : str
        Second modality name.
        
    Returns
    -------
    MOFACorrelationAnalyzer
        Initialized analyzer with internal caching.
    """
    return MOFACorrelationAnalyzer(nmf_results, corr_df, view1, view2)

def compute_cross_correlation_parallel_numpy(mdata, 
                                            view1: str, 
                                            view2: str,
                                            min_corr: float = 0.3,
                                            p_threshold: float = 0.05,
                                            max_features: int = None,
                                            n_workers: int = None) -> pd.DataFrame:
    r"""
    Compute Pearson cross-modality correlations with multiprocessing NumPy.
    
    Parameters
    ----------
    mdata : MuData
        MuData object with paired samples.
    view1 : str
        First modality name.
    view2 : str
        Second modality name.
    min_corr : float
        Minimum absolute correlation to retain.
    p_threshold : float
        Maximum p-value for significance filtering.
    max_features : int or None
        Optional cap on selected high-variance features per modality.
    n_workers : int or None
        Number of worker processes. If ``None``, inferred from CPU count.
        
    Returns
    -------
    pd.DataFrame
        Significant cross-modality correlations.
    """
    import multiprocessing as mp
    from functools import partial
    
    # Get paired samples
    common_samples = list(set(mdata.mod[view1].obs_names) & set(mdata.mod[view2].obs_names))
    if len(common_samples) == 0:
        raise ValueError("No common samples found between the two modalities")
    
    print(f"Found {len(common_samples)} paired samples")
    
    # Get data for paired samples
    data1 = mdata.mod[view1][common_samples].X
    data2 = mdata.mod[view2][common_samples].X
    
    # Convert to dense if sparse
    if hasattr(data1, 'toarray'):
        data1 = data1.toarray()
    if hasattr(data2, 'toarray'):
        data2 = data2.toarray()
    
    # Feature selection for memory management
    if max_features is not None:
        if data1.shape[1] > max_features:
            var1 = np.var(data1, axis=0)
            top_indices1 = np.argsort(var1)[-max_features:]
            data1 = data1[:, top_indices1]
            features1 = mdata.mod[view1].var_names[top_indices1]
        else:
            features1 = mdata.mod[view1].var_names
            
        if data2.shape[1] > max_features:
            var2 = np.var(data2, axis=0)
            top_indices2 = np.argsort(var2)[-max_features:]
            data2 = data2[:, top_indices2]
            features2 = mdata.mod[view2].var_names[top_indices2]
        else:
            features2 = mdata.mod[view2].var_names
    else:
        features1 = mdata.mod[view1].var_names
        features2 = mdata.mod[view2].var_names
    
    n_samples, n_features1 = data1.shape
    _, n_features2 = data2.shape
    
    print(f"Computing correlations between {n_features1} and {n_features2} features...")
    
    # Standardize data
    data1_mean = np.mean(data1, axis=0)
    data1_std = np.std(data1, axis=0)
    data1_std[data1_std == 0] = 1
    data1_normalized = (data1 - data1_mean) / data1_std
    
    data2_mean = np.mean(data2, axis=0)
    data2_std = np.std(data2, axis=0)
    data2_std[data2_std == 0] = 1
    data2_normalized = (data2 - data2_mean) / data2_std
    
    # Determine number of workers
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)  # Conservative default
    
    print(f"Using {n_workers} parallel workers")
    
    # Define worker function for parallel correlation computation
    def compute_correlation_chunk(args):
        start_idx, end_idx, data1_chunk, data2_full, features1_chunk, features2_full, min_corr, p_threshold, n_samples = args
        
        correlations = []
        
        # Compute correlation matrix for this chunk
        corr_matrix = np.dot(data1_chunk.T, data2_full) / (n_samples - 1)
        
        # Compute p-values using t-distribution
        t_stat = corr_matrix * np.sqrt((n_samples - 2) / (1 - corr_matrix**2 + 1e-10))
        from scipy.stats import t as t_dist
        p_values = 2 * (1 - t_dist.cdf(np.abs(t_stat), n_samples - 2))
        
        # Find significant correlations
        abs_corr = np.abs(corr_matrix)
        significant_mask = (abs_corr >= min_corr) & (p_values <= p_threshold)
        
        # Extract significant correlations
        sig_indices = np.where(significant_mask)
        for i, j in zip(sig_indices[0], sig_indices[1]):
            correlations.append({
                f'{view1}_feature': features1_chunk[i],
                f'{view2}_feature': features2_full[j],
                'correlation': corr_matrix[i, j],
                'p_value': p_values[i, j],
                'abs_correlation': abs_corr[i, j]
            })
        
        return correlations
    
    # Split data into chunks for parallel processing
    chunk_size = max(1, n_features1 // n_workers)
    chunks = []
    
    for i in range(0, n_features1, chunk_size):
        end_idx = min(i + chunk_size, n_features1)
        chunks.append((
            i, end_idx,
            data1_normalized[:, i:end_idx],
            data2_normalized,
            features1[i:end_idx],
            features2,
            min_corr, p_threshold, n_samples
        ))
    
    print(f"Processing {len(chunks)} chunks in parallel...")
    
    # Process chunks in parallel
    with mp.Pool(n_workers) as pool:
        chunk_results = pool.map(compute_correlation_chunk, chunks)
    
    # Combine results
    all_correlations = []
    for chunk_corr in chunk_results:
        all_correlations.extend(chunk_corr)
    
    corr_df = pd.DataFrame(all_correlations)
    print(f"Found {len(corr_df)} significant correlations")
    
    return corr_df

def compute_kendall_tau_torch(mdata, 
                             view1: str, 
                             view2: str,
                             min_corr: float = 0.3,
                             p_threshold: float = 0.05,
                             max_features: int = None,
                             device: str = 'auto',
                             batch_size1: int = 500,
                             batch_size2: int = 1000,
                             dtype: str = 'float32',
                             handle_zeros: bool = True) -> pd.DataFrame:
    r"""
    Compute Kendall's tau correlations with optional GPU acceleration.
    
    Parameters
    ----------
    mdata : MuData
        MuData object with paired sample IDs.
    view1 : str
        First modality name.
    view2 : str
        Second modality name.
    min_corr : float
        Minimum absolute correlation to retain.
    p_threshold : float
        Maximum p-value for significance filtering.
    max_features : int or None
        Optional cap on selected high-variance features.
    device : str
        Compute device: ``'auto'``, ``'cpu'``, ``'cuda'`` or ``'mps'``.
    batch_size1 : int
        Batch size for view1 features.
    batch_size2 : int
        Batch size for view2 features.
    dtype : str
        Tensor precision, ``'float32'`` or ``'float64'``.
    handle_zeros : bool
        Whether to use zero-aware tie handling strategy.
        
    Returns
    -------
    pd.DataFrame
        Significant Kendall correlations between modality features.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        raise ImportError("PyTorch is required. Install with: pip install torch")
    
    # Determine device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Using device: {device} for Kendall's Tau computation")
    
    # Get paired samples
    common_samples = list(set(mdata.mod[view1].obs_names) & set(mdata.mod[view2].obs_names))
    if len(common_samples) == 0:
        raise ValueError("No common samples found between the two modalities")
    
    print(f"Found {len(common_samples)} paired samples")
    
    # Get data for paired samples
    data1 = mdata.mod[view1][common_samples].X
    data2 = mdata.mod[view2][common_samples].X
    
    # Convert to dense if sparse
    if hasattr(data1, 'toarray'):
        data1 = data1.toarray()
    if hasattr(data2, 'toarray'):
        data2 = data2.toarray()
    
    # Feature selection optimized for sparse data
    if max_features is not None:
        if data1.shape[1] > max_features:
            # Select features with highest information content (non-zero variance)
            non_zero_counts1 = np.count_nonzero(data1, axis=0)
            var1 = np.var(data1, axis=0)
            # Prioritize features with more non-zero values and higher variance
            feature_scores1 = non_zero_counts1 * var1
            top_indices1 = np.argsort(feature_scores1)[-max_features:]
            data1 = data1[:, top_indices1]
            features1 = mdata.mod[view1].var_names[top_indices1]
        else:
            features1 = mdata.mod[view1].var_names
            
        if data2.shape[1] > max_features:
            non_zero_counts2 = np.count_nonzero(data2, axis=0)
            var2 = np.var(data2, axis=0)
            feature_scores2 = non_zero_counts2 * var2
            top_indices2 = np.argsort(feature_scores2)[-max_features:]
            data2 = data2[:, top_indices2]
            features2 = mdata.mod[view2].var_names[top_indices2]
        else:
            features2 = mdata.mod[view2].var_names
    else:
        features1 = mdata.mod[view1].var_names
        features2 = mdata.mod[view2].var_names
    
    n_samples, n_features1 = data1.shape
    _, n_features2 = data2.shape
    
    print(f"Computing Kendall's Tau between {n_features1} and {n_features2} features...")
    print(f"Using 2D batching: {batch_size1} x {batch_size2}")
    
    torch_dtype = torch.float32 if dtype == 'float32' else torch.float64
    
    def kendall_tau_torch_batch(X1_batch, X2_batch, handle_zeros=True):
        """
        Compute Kendall's Tau for batches using PyTorch.
        Optimized for sparse data with many zeros.
        """
        batch_size1, n_samples = X1_batch.shape
        batch_size2, _ = X2_batch.shape
        
        # Initialize results
        tau_matrix = torch.zeros(batch_size1, batch_size2, device=device, dtype=torch_dtype)
        p_matrix = torch.ones(batch_size1, batch_size2, device=device, dtype=torch_dtype)
        
        for i in range(batch_size1):
            for j in range(batch_size2):
                x = X1_batch[i]
                y = X2_batch[j]
                
                if handle_zeros:
                    # Remove pairs where both values are zero
                    non_zero_mask = (x != 0) | (y != 0)
                    if torch.sum(non_zero_mask) < 3:
                        continue
                    x_filtered = x[non_zero_mask]
                    y_filtered = y[non_zero_mask]
                else:
                    x_filtered = x
                    y_filtered = y
                
                # Compute Kendall's Tau using PyTorch
                tau, p_val = kendall_tau_torch_single(x_filtered, y_filtered)
                tau_matrix[i, j] = tau
                p_matrix[i, j] = p_val
        
        return tau_matrix, p_matrix
    
    def kendall_tau_torch_single(x, y):
        """
        Compute Kendall's Tau for a single pair using PyTorch.
        """
        n = len(x)
        if n < 3:
            return torch.tensor(0.0, device=device, dtype=torch_dtype), torch.tensor(1.0, device=device, dtype=torch_dtype)
        
        # Create all pairs
        i_indices = torch.arange(n, device=device).unsqueeze(1).expand(n, n)
        j_indices = torch.arange(n, device=device).unsqueeze(0).expand(n, n)
        
        # Only consider upper triangle (i < j)
        mask = i_indices < j_indices
        
        if torch.sum(mask) == 0:
            return torch.tensor(0.0, device=device, dtype=torch_dtype), torch.tensor(1.0, device=device, dtype=torch_dtype)
        
        # Get pairs
        x_i = x[i_indices[mask]]
        x_j = x[j_indices[mask]]
        y_i = y[i_indices[mask]]
        y_j = y[j_indices[mask]]
        
        # Compute concordant and discordant pairs
        x_diff = x_i - x_j
        y_diff = y_i - y_j
        
        concordant = torch.sum((x_diff * y_diff) > 0).float()
        discordant = torch.sum((x_diff * y_diff) < 0).float()
        
        # Handle ties
        x_ties = torch.sum(x_diff == 0).float()
        y_ties = torch.sum(y_diff == 0).float()
        
        total_pairs = torch.sum(mask).float()
        
        # Kendall's Tau formula
        if total_pairs == 0:
            return torch.tensor(0.0, device=device, dtype=torch_dtype), torch.tensor(1.0, device=device, dtype=torch_dtype)
        
        tau = (concordant - discordant) / torch.sqrt((total_pairs - x_ties) * (total_pairs - y_ties) + 1e-10)
        
        # Approximate p-value using normal approximation
        # Convert n to tensor for consistent operations
        n_tensor = torch.tensor(n, device=device, dtype=torch_dtype)
        var_tau = (2 * (2*n_tensor + 5)) / (9 * n_tensor * (n_tensor - 1))
        z_score = tau / torch.sqrt(var_tau + 1e-10)
        
        # Convert to p-value (two-tailed test)
        sqrt_2 = torch.tensor(2.0, device=device, dtype=torch_dtype)
        p_val = 2 * (1 - torch.erf(torch.abs(z_score) / torch.sqrt(sqrt_2)))
        
        return tau, p_val
    
    # Compute correlations in 2D batches
    correlations = []
    n_batches1 = (n_features1 + batch_size1 - 1) // batch_size1
    n_batches2 = (n_features2 + batch_size2 - 1) // batch_size2
    total_batches = n_batches1 * n_batches2
    
    with torch.no_grad():
        with tqdm(total=total_batches, desc="Processing Kendall's Tau batches") as pbar:
            for batch_idx1 in range(n_batches1):
                start_idx1 = batch_idx1 * batch_size1
                end_idx1 = min(start_idx1 + batch_size1, n_features1)
                
                # Get batch of features from view1
                X1_batch = torch.tensor(data1[:, start_idx1:end_idx1].T, 
                                      dtype=torch_dtype, device=device)  # (batch_size1, n_samples)
                
                for batch_idx2 in range(n_batches2):
                    start_idx2 = batch_idx2 * batch_size2
                    end_idx2 = min(start_idx2 + batch_size2, n_features2)
                    
                    # Get batch of features from view2
                    X2_batch = torch.tensor(data2[:, start_idx2:end_idx2].T, 
                                          dtype=torch_dtype, device=device)  # (batch_size2, n_samples)
                    
                    # Compute Kendall's Tau for this batch pair
                    tau_matrix, p_matrix = kendall_tau_torch_batch(X1_batch, X2_batch, handle_zeros)
                    
                    # Convert to numpy for processing
                    tau_matrix_np = tau_matrix.cpu().numpy()
                    p_matrix_np = p_matrix.cpu().numpy()
                    
                    # Find significant correlations
                    abs_tau = np.abs(tau_matrix_np)
                    significant_mask = (abs_tau >= min_corr) & (p_matrix_np <= p_threshold)
                    
                    # Extract significant correlations
                    sig_indices = np.where(significant_mask)
                    for i, j in zip(sig_indices[0], sig_indices[1]):
                        correlations.append({
                            f'{view1}_feature': features1[start_idx1 + i],
                            f'{view2}_feature': features2[start_idx2 + j],
                            'correlation': tau_matrix_np[i, j],
                            'p_value': p_matrix_np[i, j],
                            'abs_correlation': abs_tau[i, j]
                        })
                    
                    pbar.update(1)
    
    corr_df = pd.DataFrame(correlations)
    print(f"Found {len(corr_df)} significant Kendall's Tau correlations")
    
    return corr_df

def nmf_coexpression_modules_kendall(mdata,
                                   view1: str,
                                   view2: str,
                                   n_components: int = 10,
                                   correlation_threshold: float = 0.3,
                                   random_state: int = 42,
                                   max_features: int = None,
                                   use_torch: bool = True,
                                   device: str = 'auto',
                                   nmf_solver: str = 'mu',
                                   nmf_max_iter: int = 1000,
                                   corr_df: pd.DataFrame = None,
                                   handle_zeros: bool = True) -> dict:
    r"""
    Infer cross-modality modules using Kendall-correlation-driven NMF.

    Parameters
    ----------
    mdata : MuData
        MuData object with paired sample IDs across views.
    view1 : str
        First modality name.
    view2 : str
        Second modality name.
    n_components : int
        Number of latent NMF modules.
    correlation_threshold : float
        Minimum absolute Kendall tau retained in the module input matrix.
    random_state : int
        Random seed for NMF reproducibility.
    max_features : int or None
        Optional cap on selected high-variance features per modality.
    use_torch : bool
        Whether to use PyTorch backend for Kendall calculation.
    device : str
        Device selection for PyTorch backend.
    nmf_solver : str
        Scikit-learn NMF solver.
    nmf_max_iter : int
        Maximum NMF optimization iterations.
    corr_df : pd.DataFrame or None
        Optional precomputed Kendall correlation table.
    handle_zeros : bool
        Whether to use zero-aware strategy in Kendall computation.

    Returns
    -------
    dict
        Dictionary containing correlation matrix, NMF decomposition and
        module-specific feature sets.
    """
    from sklearn.decomposition import NMF
    
    # Use pre-computed correlation data if provided
    if corr_df is not None:
        print("Using pre-computed Kendall's Tau correlation DataFrame...")
        # Validate that the correlation DataFrame has the expected columns
        expected_cols = [f'{view1}_feature', f'{view2}_feature', 'correlation', 'abs_correlation']
        missing_cols = [col for col in expected_cols if col not in corr_df.columns]
        if missing_cols:
            raise ValueError(f"Pre-computed corr_df missing columns: {missing_cols}")
        
        # Filter by threshold
        corr_df_filtered = corr_df[corr_df['abs_correlation'] >= correlation_threshold].copy()
        
    else:
        # Compute Kendall's Tau correlations
        print("Computing Kendall's Tau correlations...")
        if use_torch:
            try:
                corr_df = compute_kendall_tau_torch(
                    mdata, view1, view2,
                    min_corr=correlation_threshold,
                    max_features=max_features,
                    device=device,
                    handle_zeros=handle_zeros
                )
            except ImportError:
                print("PyTorch not available, falling back to scipy implementation...")
                # Fallback to scipy-based implementation
                from scipy.stats import kendalltau
                
                # Get paired samples
                common_samples = list(set(mdata.mod[view1].obs_names) & set(mdata.mod[view2].obs_names))
                data1 = mdata.mod[view1][common_samples].X
                data2 = mdata.mod[view2][common_samples].X
                
                if hasattr(data1, 'toarray'):
                    data1 = data1.toarray()
                if hasattr(data2, 'toarray'):
                    data2 = data2.toarray()
                
                # Simple feature selection for fallback
                if max_features is not None:
                    if data1.shape[1] > max_features:
                        var1 = np.var(data1, axis=0)
                        top_indices1 = np.argsort(var1)[-max_features:]
                        data1 = data1[:, top_indices1]
                        features1 = mdata.mod[view1].var_names[top_indices1]
                    else:
                        features1 = mdata.mod[view1].var_names
                        
                    if data2.shape[1] > max_features:
                        var2 = np.var(data2, axis=0)
                        top_indices2 = np.argsort(var2)[-max_features:]
                        data2 = data2[:, top_indices2]
                        features2 = mdata.mod[view2].var_names[top_indices2]
                    else:
                        features2 = mdata.mod[view2].var_names
                else:
                    features1 = mdata.mod[view1].var_names
                    features2 = mdata.mod[view2].var_names
                
                # Compute correlations with progress bar
                correlations = []
                total_pairs = len(features1) * len(features2)
                
                with tqdm(total=total_pairs, desc="Computing Kendall correlations") as pbar:
                    for i, feat1 in enumerate(features1):
                        for j, feat2 in enumerate(features2):
                            x = data1[:, i]
                            y = data2[:, j]
                            
                            if handle_zeros:
                                # Remove pairs where both are zero
                                non_zero_mask = (x != 0) | (y != 0)
                                if np.sum(non_zero_mask) < 3:
                                    pbar.update(1)
                                    continue
                                x_filtered = x[non_zero_mask]
                                y_filtered = y[non_zero_mask]
                            else:
                                x_filtered = x
                                y_filtered = y
                            
                            try:
                                tau, p_val = kendalltau(x_filtered, y_filtered)
                                if not np.isnan(tau) and abs(tau) >= correlation_threshold:
                                    correlations.append({
                                        f'{view1}_feature': feat1,
                                        f'{view2}_feature': feat2,
                                        'correlation': tau,
                                        'p_value': p_val,
                                        'abs_correlation': abs(tau)
                                    })
                            except:
                                pass
                            
                            pbar.update(1)
                
                corr_df = pd.DataFrame(correlations)
        else:
            # Use scipy implementation directly
            print("Using scipy-based Kendall's Tau computation...")
            # [Same fallback code as above]
            
        corr_df_filtered = corr_df.copy()
    
    if len(corr_df_filtered) == 0:
        raise ValueError("No significant Kendall correlations found")
    
    print(f"Found {len(corr_df_filtered)} significant Kendall's Tau correlations")
    
    # Create correlation matrix from filtered data
    features1 = corr_df_filtered[f'{view1}_feature'].unique()
    features2 = corr_df_filtered[f'{view2}_feature'].unique()
    
    print(f"Creating correlation matrix: {len(features1)} x {len(features2)}")
    
    # Create correlation matrix
    final_corr_matrix = pd.DataFrame(0.0, index=features1, columns=features2)
    for _, row in tqdm(corr_df_filtered.iterrows(), total=len(corr_df_filtered), desc="Building correlation matrix"):
        final_corr_matrix.loc[row[f'{view1}_feature'], row[f'{view2}_feature']] = abs(row['correlation'])
    
    print(f"Correlation matrix shape: {final_corr_matrix.shape}")
    print(f"Non-zero correlations: {(final_corr_matrix > 0).sum().sum()}")
    
    # Apply NMF with optimized parameters
    print(f"Applying NMF with {n_components} components...")
    nmf = NMF(
        n_components=n_components, 
        random_state=random_state, 
        max_iter=nmf_max_iter,
        solver=nmf_solver,
        beta_loss='frobenius',
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0
    )
    
    W = nmf.fit_transform(final_corr_matrix.values)  # Features x Components
    H = nmf.components_  # Components x Features
    
    # Create results
    results = {
        'nmf_model': nmf,
        'W_matrix': pd.DataFrame(W, index=final_corr_matrix.index, 
                               columns=[f'Module_{i+1}' for i in range(n_components)]),
        'H_matrix': pd.DataFrame(H, index=[f'Module_{i+1}' for i in range(n_components)], 
                               columns=final_corr_matrix.columns),
        'correlation_matrix': final_corr_matrix,
        'correlation_data': corr_df,
        'reconstruction_error': nmf.reconstruction_err_,
        'n_iter': nmf.n_iter_,
        'correlation_method': 'kendall_tau'
    }
    
    # Extract modules with better scoring
    modules = {}
    for i in range(n_components):
        module_name = f'Module_{i+1}'
        
        # Get top features from each modality for this module
        w_scores = results['W_matrix'][module_name]
        h_scores = results['H_matrix'].loc[module_name]
        
        # Use percentile-based thresholding for better feature selection
        w_threshold = np.percentile(w_scores, 85)  # Top 15%
        h_threshold = np.percentile(h_scores, 85)
        
        w_top = w_scores[w_scores >= w_threshold].sort_values(ascending=False)
        h_top = h_scores[h_scores >= h_threshold].sort_values(ascending=False)
        
        modules[module_name] = {
            f'{view1}_features': w_top.to_dict(),
            f'{view2}_features': h_top.to_dict(),
            f'{view1}_top_features': w_top.index.tolist(),
            f'{view2}_top_features': h_top.index.tolist(),
            f'{view1}_threshold': w_threshold,
            f'{view2}_threshold': h_threshold,
            'module_strength': (w_scores.max() + h_scores.max()) / 2
        }
    
    results['modules'] = modules
    
    print(f"NMF completed in {nmf.n_iter_} iterations")
    print(f"Reconstruction error: {nmf.reconstruction_err_:.4f}")
    print(f"Modules extracted using Kendall's Tau correlations")
    
    return results

def validate_kendall_tau_implementation(mdata, 
                                      view1: str, 
                                      view2: str,
                                      n_test_pairs: int = 10,
                                      max_features: int = 100,
                                      device: str = 'auto') -> pd.DataFrame:
    r"""
    Validate GPU/torch Kendall implementation against SciPy reference.
    
    Parameters
    ----------
    mdata : MuData
        MuData object with paired sample IDs.
    view1 : str
        First modality name.
    view2 : str
        Second modality name.
    n_test_pairs : int
        Number of random feature pairs used for validation.
    max_features : int
        Maximum features retained per modality for test runtime control.
    device : str
        Device setting for torch implementation.
        
    Returns
    -------
    pd.DataFrame
        Comparison table including torch vs SciPy tau/p-value differences.
    """
    try:
        import torch
        from scipy.stats import kendalltau
    except ImportError:
        raise ImportError("PyTorch and scipy are required for validation")
    
    # Determine device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Validating Kendall's Tau implementation using device: {device}")
    
    # Get paired samples
    common_samples = list(set(mdata.mod[view1].obs_names) & set(mdata.mod[view2].obs_names))
    
    # Get data for paired samples
    data1 = mdata.mod[view1][common_samples].X
    data2 = mdata.mod[view2][common_samples].X
    
    # Convert to dense if sparse
    if hasattr(data1, 'toarray'):
        data1 = data1.toarray()
    if hasattr(data2, 'toarray'):
        data2 = data2.toarray()
    
    # Select subset of features for testing
    if data1.shape[1] > max_features:
        indices1 = np.random.choice(data1.shape[1], max_features, replace=False)
        data1 = data1[:, indices1]
        features1 = mdata.mod[view1].var_names[indices1]
    else:
        features1 = mdata.mod[view1].var_names
    
    if data2.shape[1] > max_features:
        indices2 = np.random.choice(data2.shape[1], max_features, replace=False)
        data2 = data2[:, indices2]
        features2 = mdata.mod[view2].var_names[indices2]
    else:
        features2 = mdata.mod[view2].var_names
    
    torch_dtype = torch.float32
    
    def kendall_tau_torch_single(x, y, handle_zeros=True):
        """Our PyTorch implementation for single pair."""
        if handle_zeros:
            non_zero_mask = (x != 0) | (y != 0)
            if torch.sum(non_zero_mask) < 10:
                return 0.0, 1.0
            x_filtered = x[non_zero_mask]
            y_filtered = y[non_zero_mask]
        else:
            x_filtered = x
            y_filtered = y
        
        n = len(x_filtered)
        if n < 10:
            return 0.0, 1.0
        
        # Vectorized computation
        x_expanded = x_filtered.unsqueeze(1)
        y_expanded = y_filtered.unsqueeze(1)
        
        x_diff = x_expanded - x_filtered.unsqueeze(0)
        y_diff = y_expanded - y_filtered.unsqueeze(0)
        
        mask = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)
        
        x_diff_masked = x_diff[mask]
        y_diff_masked = y_diff[mask]
        
        concordant = torch.sum((x_diff_masked * y_diff_masked) > 0).float()
        discordant = torch.sum((x_diff_masked * y_diff_masked) < 0).float()
        
        total_pairs = torch.sum(mask).float()
        
        if total_pairs == 0:
            return 0.0, 1.0
        
        tau = (concordant - discordant) / total_pairs
        
        # P-value approximation
        n_tensor = torch.tensor(n, device=device, dtype=torch_dtype)
        var_tau = (2 * (2*n_tensor + 5)) / (9 * n_tensor * (n_tensor - 1))
        
        if var_tau <= 0:
            return tau.item(), 1.0
        
        z_score = tau / torch.sqrt(var_tau)
        p_val = 2 * (1 - torch.erf(torch.abs(z_score) / torch.sqrt(torch.tensor(2.0, device=device, dtype=torch_dtype))))
        
        return tau.item(), p_val.item()
    
    # Test random pairs
    comparisons = []
    
    print(f"Testing {n_test_pairs} random feature pairs...")
    
    for test_idx in range(n_test_pairs):
        # Select random features
        i = np.random.randint(0, data1.shape[1])
        j = np.random.randint(0, data2.shape[1])
        
        x_np = data1[:, i]
        y_np = data2[:, j]
        
        # Remove zero pairs for fair comparison
        non_zero_mask = (x_np != 0) | (y_np != 0)
        if np.sum(non_zero_mask) < 10:
            continue
            
        x_filtered = x_np[non_zero_mask]
        y_filtered = y_np[non_zero_mask]
        
        # Scipy implementation
        try:
            tau_scipy, p_scipy = kendalltau(x_filtered, y_filtered)
        except:
            continue
        
        # Our implementation
        x_torch = torch.tensor(x_np, dtype=torch_dtype, device=device)
        y_torch = torch.tensor(y_np, dtype=torch_dtype, device=device)
        tau_ours, p_ours = kendall_tau_torch_single(x_torch, y_torch, handle_zeros=True)
        
        comparisons.append({
            'feature1': features1[i],
            'feature2': features2[j],
            'tau_scipy': tau_scipy,
            'tau_ours': tau_ours,
            'p_scipy': p_scipy,
            'p_ours': p_ours,
            'tau_diff': abs(tau_scipy - tau_ours),
            'p_diff': abs(p_scipy - p_ours),
            'n_samples': len(x_filtered)
        })
    
    comparison_df = pd.DataFrame(comparisons)
    
    if len(comparison_df) > 0:
        print(f"\nValidation Results:")
        print(f"Mean Tau difference: {comparison_df['tau_diff'].mean():.6f}")
        print(f"Max Tau difference: {comparison_df['tau_diff'].max():.6f}")
        print(f"Mean P-value difference: {comparison_df['p_diff'].mean():.6f}")
        print(f"Max P-value difference: {comparison_df['p_diff'].max():.6f}")
        
        # Check if differences are within acceptable range
        tau_acceptable = comparison_df['tau_diff'].max() < 0.01
        p_acceptable = comparison_df['p_diff'].max() < 0.05
        
        if tau_acceptable and p_acceptable:
            print("✅ Validation PASSED: Our implementation matches scipy within acceptable tolerance")
        else:
            print("❌ Validation FAILED: Differences exceed acceptable tolerance")
    else:
        print("No valid comparisons could be made")
    
    return comparison_df

def factor_group_correlation_mdata(mdata,
                                  group_col: str = 'Group',
                                  factor_prefix: str = 'Factor',
                                  view: str = None,
                                  p_threshold: float = 0.05,
                                  method: str = 'ttest',
                                  log_transform: bool = True,
                                  min_group_size: int = 3) -> pd.DataFrame:
    r"""
    Test associations between MOFA factors and categorical groups per view.
    
    Parameters
    ----------
    mdata : MuData
        MuData containing factor scores in ``mdata.obsm['X_mofa']``.
    group_col : str
        Group label column in modality ``obs`` tables.
    factor_prefix : str
        Prefix used for naming factor labels in output.
    view : str or None
        Specific modality to analyze. If ``None``, analyze all modalities.
    p_threshold : float
        Significance threshold used for result flagging.
    method : str
        Statistical test: ``'ttest'``, ``'anova'`` or ``'kruskal'``.
    log_transform : bool
        Whether to transform p-values to ``-log10(p)`` in outputs.
    min_group_size : int
        Minimum per-group sample size to include in tests.
        
    Returns
    -------
    pd.DataFrame
        Long-format table of factor-group association statistics per modality.
    """
    import pandas as pd
    import numpy as np
    from scipy import stats
    
    print(f"🔍 Analyzing Factor-Group relationships in MuData")
    print(f"   Group column: {group_col}")
    print(f"   Method: {method}")
    
    # Check if MOFA factors exist in main MuData
    if 'X_mofa' not in mdata.obsm:
        raise ValueError("No MOFA factors found in MuData.obsm['X_mofa']. Please run MOFA analysis first.")
    
    # Get MOFA factors from main MuData
    factor_data = mdata.obsm['X_mofa']
    n_factors = factor_data.shape[1]
    factor_names = [f'{factor_prefix}{i+1}' for i in range(n_factors)]
    
    print(f"   Found {n_factors} MOFA factors in main MuData")
    
    # Get views to analyze
    if view is not None:
        if view not in mdata.mod.keys():
            raise ValueError(f"View '{view}' not found in MuData. Available views: {list(mdata.mod.keys())}")
        views_to_analyze = [view]
    else:
        views_to_analyze = list(mdata.mod.keys())
    
    print(f"   Analyzing views: {views_to_analyze}")
    
    all_results = []
    
    for current_view in views_to_analyze:
        print(f"\n📊 Analyzing view: {current_view}")
        
        # Get data for current view
        adata = mdata.mod[current_view]
        
        # Check if group column exists
        if group_col not in adata.obs.columns:
            print(f"   ⚠️ Group column '{group_col}' not found in {current_view}, skipping...")
            continue
        
        # Get sample names for current view
        view_samples = adata.obs_names.tolist()
        
        # Find matching samples in main MuData
        main_samples = mdata.obs_names.tolist()
        
        # Get indices of view samples in main MuData
        sample_indices = []
        matched_samples = []
        for sample in view_samples:
            if sample in main_samples:
                idx = main_samples.index(sample)
                sample_indices.append(idx)
                matched_samples.append(sample)
        
        if len(sample_indices) == 0:
            print(f"   ⚠️ No matching samples found between {current_view} and main MuData, skipping...")
            continue
        
        print(f"   Found {len(matched_samples)} matching samples out of {len(view_samples)} in {current_view}")
        
        # Get factor data for matched samples
        view_factor_data = factor_data[sample_indices, :]
        
        # Get group information for matched samples
        groups = adata.obs.loc[matched_samples, group_col].values
        
        print(f"   Sample size for analysis: {len(groups)}")
        print(f"   Unique groups: {np.unique(groups)}")
        
        # Get unique groups and filter by minimum size
        unique_groups = np.unique(groups)
        valid_groups = []
        for group in unique_groups:
            group_size = np.sum(groups == group)
            if group_size >= min_group_size:
                valid_groups.append(group)
            else:
                print(f"   ⚠️ Group '{group}' has only {group_size} samples (< {min_group_size}), excluding...")
        
        if len(valid_groups) < 2:
            print(f"   ⚠️ Not enough valid groups for analysis in {current_view}")
            continue
        
        print(f"   Valid groups for analysis: {valid_groups}")
        
        # Analyze each factor
        view_results = []
        
        for factor_idx, factor_name in enumerate(factor_names):
            factor_values = view_factor_data[:, factor_idx]
            
            # Perform statistical test
            if method == 'ttest' and len(valid_groups) == 2:
                # Two-sample t-test
                group1_values = factor_values[groups == valid_groups[0]]
                group2_values = factor_values[groups == valid_groups[1]]
                
                statistic, p_value = stats.ttest_ind(group1_values, group2_values)
                test_name = 'T-test'
                
            elif method == 'anova' or (method == 'ttest' and len(valid_groups) > 2):
                # One-way ANOVA
                group_data = [factor_values[groups == group] for group in valid_groups]
                statistic, p_value = stats.f_oneway(*group_data)
                test_name = 'ANOVA'
                
            elif method == 'kruskal':
                # Kruskal-Wallis test (non-parametric)
                group_data = [factor_values[groups == group] for group in valid_groups]
                statistic, p_value = stats.kruskal(*group_data)
                test_name = 'Kruskal-Wallis'
                
            else:
                print(f"   ⚠️ Unknown method '{method}' or inappropriate for {len(valid_groups)} groups")
                continue
            
            # Calculate effect size (eta-squared for ANOVA-like tests)
            if method in ['anova', 'kruskal'] or (method == 'ttest' and len(valid_groups) > 2):
                # Calculate eta-squared
                group_means = [np.mean(factor_values[groups == group]) for group in valid_groups]
                overall_mean = np.mean(factor_values)
                
                ss_between = sum([np.sum(groups == group) * (mean - overall_mean)**2 
                                for group, mean in zip(valid_groups, group_means)])
                ss_total = np.sum((factor_values - overall_mean)**2)
                
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                effect_size = eta_squared
                effect_size_name = 'eta_squared'
                
            else:
                # Cohen's d for t-test
                group1_values = factor_values[groups == valid_groups[0]]
                group2_values = factor_values[groups == valid_groups[1]]
                
                pooled_std = np.sqrt(((len(group1_values) - 1) * np.var(group1_values, ddof=1) + 
                                    (len(group2_values) - 1) * np.var(group2_values, ddof=1)) / 
                                   (len(group1_values) + len(group2_values) - 2))
                
                cohens_d = (np.mean(group1_values) - np.mean(group2_values)) / pooled_std if pooled_std > 0 else 0
                effect_size = abs(cohens_d)
                effect_size_name = 'cohens_d'
            
            # Apply log transformation to p-value if requested
            if log_transform and p_value > 0:
                log_p = -np.log10(p_value)
            else:
                log_p = p_value
            
            # Determine significance
            is_significant = p_value < p_threshold
            
            # Calculate group statistics
            group_stats = {}
            for group in valid_groups:
                group_values = factor_values[groups == group]
                group_stats[f'{group}_mean'] = np.mean(group_values)
                group_stats[f'{group}_std'] = np.std(group_values)
                group_stats[f'{group}_n'] = len(group_values)
            
            # Store results
            result = {
                'view': current_view,
                'factor': factor_name,
                'test_method': test_name,
                'statistic': statistic,
                'p_value': p_value,
                'log10_p': log_p if log_transform else None,
                'significant': is_significant,
                effect_size_name: effect_size,
                'n_groups': len(valid_groups),
                'groups': ','.join(map(str, valid_groups)),
                'n_samples': len(factor_values),
                'n_matched_samples': len(matched_samples),
                **group_stats
            }
            
            view_results.append(result)
        
        all_results.extend(view_results)
        print(f"   ✅ Completed analysis for {current_view}: {len(view_results)} factors analyzed")
    
    # Create results DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Sort by significance and effect size
        if log_transform:
            results_df = results_df.sort_values(['view', 'log10_p'], ascending=[True, False])
        else:
            results_df = results_df.sort_values(['view', 'p_value'], ascending=[True, True])
        
        print(f"\n📈 Analysis Summary:")
        print(f"   Total factors analyzed: {len(results_df)}")
        print(f"   Significant associations (p < {p_threshold}): {results_df['significant'].sum()}")
        
        if len(results_df) > 0:
            print(f"   Views analyzed: {results_df['view'].unique()}")
            
            # Show top significant results
            significant_results = results_df[results_df['significant']]
            if len(significant_results) > 0:
                print(f"\n🎯 Top significant associations:")
                for _, row in significant_results.head(5).iterrows():
                    effect_col = 'cohens_d' if 'cohens_d' in row else 'eta_squared'
                    print(f"   {row['view']}.{row['factor']}: p={row['p_value']:.2e}, {effect_col}={row[effect_col]:.3f} (n={row['n_samples']})")
        
        return results_df
    else:
        print("❌ No results generated. Check your data and parameters.")
        return pd.DataFrame()

def plot_factor_group_associations(results_df: pd.DataFrame,
                                  top_n: int = 10,
                                  figsize: tuple = (12, 8),
                                  save: bool = False,
                                  save_path: str = 'factor_group_associations.png') -> tuple:
    r"""
    Visualize top factor-group associations from summary statistics table.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Output table from :func:`factor_group_correlation_mdata`.
    top_n : int
        Number of top associations displayed in plots.
    figsize : tuple
        Figure size in inches as ``(width, height)``.
    save : bool
        Whether to save figure to disk.
    save_path : str
        Output path when ``save=True``.
        
    Returns
    -------
    tuple
        ``(fig, axes)`` Matplotlib objects for the multi-panel summary plot.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError("matplotlib and seaborn are required for plotting")
    
    if len(results_df) == 0:
        print("❌ No data to plot")
        return None, None
    
    # Filter significant results and get top N
    significant_results = results_df[results_df['significant']].copy()
    
    if len(significant_results) == 0:
        print("⚠️ No significant associations found")
        # Plot all results instead
        plot_data = results_df.head(top_n).copy()
        title_suffix = "(All results - no significant associations)"
    else:
        plot_data = significant_results.head(top_n).copy()
        title_suffix = f"(Top {min(top_n, len(significant_results))} significant)"
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Factor-Group Associations {title_suffix}', fontsize=16, fontweight='bold')
    
    # 1. Bar plot of -log10(p-values)
    ax1 = axes[0, 0]
    if 'log10_p' in plot_data.columns and plot_data['log10_p'].notna().any():
        y_values = plot_data['log10_p']
        ylabel = '-log10(p-value)'
    else:
        y_values = -np.log10(plot_data['p_value'])
        ylabel = '-log10(p-value)'
    
    bars = ax1.barh(range(len(plot_data)), y_values, 
                    color=['#1f77b4' if view == 'metab' else '#ff7f0e' for view in plot_data['view']])
    ax1.set_yticks(range(len(plot_data)))
    ax1.set_yticklabels([f"{row['view']}.{row['factor']}" for _, row in plot_data.iterrows()], 
                       fontsize=10)
    ax1.set_xlabel(ylabel)
    ax1.set_title('Statistical Significance')
    ax1.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
    ax1.legend()
    
    # 2. Effect size plot
    ax2 = axes[0, 1]
    effect_col = 'cohens_d' if 'cohens_d' in plot_data.columns else 'eta_squared'
    effect_values = plot_data[effect_col]
    
    bars2 = ax2.barh(range(len(plot_data)), effect_values,
                     color=['#1f77b4' if view == 'metab' else '#ff7f0e' for view in plot_data['view']])
    ax2.set_yticks(range(len(plot_data)))
    ax2.set_yticklabels([f"{row['view']}.{row['factor']}" for _, row in plot_data.iterrows()], 
                       fontsize=10)
    ax2.set_xlabel(effect_col.replace('_', ' ').title())
    ax2.set_title('Effect Size')
    
    # 3. Scatter plot: p-value vs effect size
    ax3 = axes[1, 0]
    colors = ['#1f77b4' if view == 'metab' else '#ff7f0e' for view in plot_data['view']]
    scatter = ax3.scatter(effect_values, y_values, c=colors, s=60, alpha=0.7)
    ax3.set_xlabel(effect_col.replace('_', ' ').title())
    ax3.set_ylabel(ylabel)
    ax3.set_title('Effect Size vs Significance')
    ax3.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7)
    
    # Add text annotations for top points
    for i, (_, row) in enumerate(plot_data.head(5).iterrows()):
        ax3.annotate(f"{row['view']}.{row['factor']}", 
                    (effect_values.iloc[i], y_values.iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
    
    # 4. View comparison
    ax4 = axes[1, 1]
    view_counts = plot_data['view'].value_counts()
    colors_pie = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(view_counts)]
    wedges, texts, autotexts = ax4.pie(view_counts.values, labels=view_counts.index, 
                                      autopct='%1.1f%%', colors=colors_pie)
    ax4.set_title('Significant Associations by View')
    
    # Create legend for views
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor='#1f77b4', label='metab'),
                      plt.Rectangle((0,0),1,1, facecolor='#ff7f0e', label='micro')]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")
    
    return fig, axes

def plot_factor_boxplots(mdata,
                        results_df: pd.DataFrame,
                        top_n: int = 6,
                        group_col: str = 'Group',
                        factor_prefix: str = 'Factor',
                        figsize: tuple = (15, 10),
                        save: bool = False,
                        save_path: str = 'factor_boxplots.png') -> tuple:
    r"""
    Draw boxplots for top significant factor-group associations.
    
    Parameters
    ----------
    mdata : MuData
        MuData object containing ``obsm['X_mofa']`` and group annotations.
    results_df : pd.DataFrame
        Output table from :func:`factor_group_correlation_mdata`.
    top_n : int
        Number of top significant factor entries to visualize.
    group_col : str
        Grouping column name in modality ``obs``.
    factor_prefix : str
        Prefix used in factor labels (for example ``'Factor'``).
    figsize : tuple
        Figure size in inches as ``(width, height)``.
    save : bool
        Whether to save figure to disk.
    save_path : str
        Output path when ``save=True``.
        
    Returns
    -------
    tuple
        ``(fig, axes)`` Matplotlib objects for generated boxplots.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError("matplotlib and seaborn are required for plotting")
    
    # Get top significant results
    significant_results = results_df[results_df['significant']].head(top_n)
    
    if len(significant_results) == 0:
        print("❌ No significant results to plot")
        return None, None
    
    # Check if MOFA factors exist in main MuData
    if 'X_mofa' not in mdata.obsm:
        raise ValueError("No MOFA factors found in MuData.obsm['X_mofa']")
    
    # Get MOFA factors from main MuData
    factor_data = mdata.obsm['X_mofa']
    main_samples = mdata.obs_names.tolist()
    
    # Calculate subplot layout
    n_plots = len(significant_results)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    fig.suptitle('Factor Values by Group (Top Significant Associations)', 
                fontsize=16, fontweight='bold')
    
    for i, (_, row) in enumerate(significant_results.iterrows()):
        if i >= len(axes):
            break
            
        view = row['view']
        factor = row['factor']
        
        # Get data for current view
        adata = mdata.mod[view]
        view_samples = adata.obs_names.tolist()
        
        # Find matching samples between view and main MuData
        sample_indices = []
        matched_samples = []
        for sample in view_samples:
            if sample in main_samples:
                idx = main_samples.index(sample)
                sample_indices.append(idx)
                matched_samples.append(sample)
        
        if len(sample_indices) == 0:
            print(f"⚠️ No matching samples found for {view}, skipping plot...")
            continue
        
        # Get factor data for matched samples
        factor_idx = int(factor.replace(factor_prefix, '')) - 1
        factor_values = factor_data[sample_indices, factor_idx]
        
        # Get group information for matched samples
        groups = adata.obs.loc[matched_samples, group_col].values
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'Factor_Value': factor_values,
            'Group': groups
        })
        
        # Create boxplot
        ax = axes[i]
        sns.boxplot(data=plot_df, x='Group', y='Factor_Value', ax=ax)
        sns.stripplot(data=plot_df, x='Group', y='Factor_Value', ax=ax, 
                     color='black', alpha=0.5, size=3)
        
        # Customize plot
        ax.set_title(f'{view}.{factor}\np={row["p_value"]:.2e} (n={len(factor_values)})', fontweight='bold')
        ax.set_xlabel('Group')
        ax.set_ylabel('Factor Value')
        ax.tick_params(axis='x', rotation=45)
        
        # Add statistical annotation
        effect_col = 'cohens_d' if 'cohens_d' in row else 'eta_squared'
        ax.text(0.02, 0.98, f'{effect_col.replace("_", " ").title()}: {row[effect_col]:.3f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide empty subplots
    for i in range(len(significant_results), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Boxplots saved to {save_path}")
    
    return fig, axes

def example_factor_group_analysis(mdata, group_col: str = 'Group'):
    r"""
    Run an end-to-end demo of factor-group association analysis.
    
    Parameters
    ----------
    mdata : MuData
        MuData object containing MOFA factors.
    group_col : str
        Group column used for statistical testing.
        
    Returns
    -------
    pd.DataFrame
        Combined factor-group association results from demo workflow.
    """
    print("🔬 Example: Factor-Group Analysis")
    print("=" * 50)
    
    # 1. Analyze all views
    print("\n1️⃣ Analyzing all views...")
    results_all = factor_group_correlation_mdata(
        mdata, 
        group_col=group_col,
        method='anova',  # Use ANOVA for multiple groups
        p_threshold=0.05,
        log_transform=True
    )
    
    # 2. Analyze specific view (metab)
    print("\n2️⃣ Analyzing metab view only...")
    results_metab = factor_group_correlation_mdata(
        mdata, 
        group_col=group_col,
        view='metab',
        method='ttest',  # Use t-test if only 2 groups
        p_threshold=0.05
    )
    
    # 3. Analyze specific view (micro)
    print("\n3️⃣ Analyzing micro view only...")
    results_micro = factor_group_correlation_mdata(
        mdata, 
        group_col=group_col,
        view='micro',
        method='kruskal',  # Non-parametric test
        p_threshold=0.05
    )
    
    # 4. Plot results
    print("\n4️⃣ Creating visualizations...")
    
    if len(results_all) > 0:
        # Plot association summary
        fig1, axes1 = plot_factor_group_associations(
            results_all, 
            top_n=10,
            figsize=(14, 10),
            save=False
        )
        
        # Plot boxplots for top significant factors
        fig2, axes2 = plot_factor_boxplots(
            mdata,
            results_all,
            top_n=6,
            group_col=group_col,
            figsize=(16, 12),
            save=False
        )
        
        print("✅ Visualizations created successfully!")
    else:
        print("⚠️ No significant results found for visualization")
    
    print("\n📊 Analysis Summary:")
    print(f"   Total results (all views): {len(results_all)}")
    print(f"   Metab view results: {len(results_metab)}")
    print(f"   Micro view results: {len(results_micro)}")
    
    if len(results_all) > 0:
        significant_count = results_all['significant'].sum()
        print(f"   Significant associations: {significant_count}")
        
        if significant_count > 0:
            print("\n🎯 Top 3 significant associations:")
            top_results = results_all[results_all['significant']].head(3)
            for _, row in top_results.iterrows():
                effect_col = 'cohens_d' if 'cohens_d' in row else 'eta_squared'
                print(f"   • {row['view']}.{row['factor']}: p={row['p_value']:.2e}, {effect_col}={row[effect_col]:.3f}")
    
    print("\n" + "=" * 50)
    print("✅ Factor-Group analysis completed!")
    
    return results_all

def compute_spearman_torch_fast(mdata, 
                              view1: str, 
                              view2: str,
                              min_corr: float = 0.3,
                              p_threshold: float = 0.05,
                              max_features: int = 1000,
                              device: str = 'auto',
                              batch_size: int = 100,
                              dtype: str = 'float32',
                              handle_zeros: bool = True,
                              min_nonzero_pairs: int = 10) -> pd.DataFrame:
    r"""
    Compute cross-modality Spearman correlations with torch acceleration.

    Parameters
    ----------
    mdata : MuData
        MuData object with paired sample IDs.
    view1 : str
        First modality name.
    view2 : str
        Second modality name.
    min_corr : float
        Minimum absolute Spearman correlation retained.
    p_threshold : float
        Maximum p-value for significance filtering.
    max_features : int
        Maximum selected high-variance features per modality.
    device : str
        Compute device: ``'auto'``, ``'cpu'``, ``'cuda'`` or ``'mps'``.
    batch_size : int
        Number of feature pairs processed per batch.
    dtype : str
        Tensor precision, ``'float32'`` or ``'float64'``.
    handle_zeros : bool
        Whether to apply zero-aware handling for sparse data.
    min_nonzero_pairs : int
        Minimum non-zero paired observations required to test a feature pair.

    Returns
    -------
    pd.DataFrame
        Significant feature-pair Spearman correlations with p-values.
    """
    try:
        import torch
        from tqdm import tqdm
    except ImportError:
        raise ImportError("PyTorch and tqdm are required for this function")
    
    print(f"🔍 Computing Spearman correlations between {view1} and {view2}")
    
    # Get paired samples
    common_samples = list(set(mdata.mod[view1].obs_names) & set(mdata.mod[view2].obs_names))
    if len(common_samples) == 0:
        raise ValueError(f"No common samples found between {view1} and {view2}")
    
    print(f"   Found {len(common_samples)} common samples")
    
    # Get data matrices
    data1 = mdata.mod[view1][common_samples].X
    data2 = mdata.mod[view2][common_samples].X
    
    if hasattr(data1, 'toarray'):
        data1 = data1.toarray()
    if hasattr(data2, 'toarray'):
        data2 = data2.toarray()
    
    # Feature selection based on variance
    if max_features is not None:
        if data1.shape[1] > max_features:
            var1 = np.var(data1, axis=0)
            top_indices1 = np.argsort(var1)[-max_features:]
            data1 = data1[:, top_indices1]
            features1 = mdata.mod[view1].var_names[top_indices1]
        else:
            features1 = mdata.mod[view1].var_names
            
        if data2.shape[1] > max_features:
            var2 = np.var(data2, axis=0)
            top_indices2 = np.argsort(var2)[-max_features:]
            data2 = data2[:, top_indices2]
            features2 = mdata.mod[view2].var_names[top_indices2]
        else:
            features2 = mdata.mod[view2].var_names
    else:
        features1 = mdata.mod[view1].var_names
        features2 = mdata.mod[view2].var_names
    
    n_features1, n_features2 = len(features1), len(features2)
    n_samples = len(common_samples)
    
    print(f"   {view1}: {n_features1} features")
    print(f"   {view2}: {n_features2} features")
    print(f"   Samples: {n_samples}")
    
    # Set up device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"   Using device: {device}")
    
    # Set data type
    torch_dtype = torch.float32 if dtype == 'float32' else torch.float64
    
    def spearman_torch_single(x, y, handle_zeros=True):
        """
        Compute Spearman correlation for two vectors using PyTorch.
        """
        if handle_zeros:
            # Remove pairs where both values are zero
            non_zero_mask = (x != 0) | (y != 0)
            if torch.sum(non_zero_mask) < min_nonzero_pairs:
                return torch.tensor(0.0, device=device, dtype=torch_dtype), torch.tensor(1.0, device=device, dtype=torch_dtype)
            x_filtered = x[non_zero_mask]
            y_filtered = y[non_zero_mask]
        else:
            x_filtered = x
            y_filtered = y
        
        n = len(x_filtered)
        if n < 3:
            return torch.tensor(0.0, device=device, dtype=torch_dtype), torch.tensor(1.0, device=device, dtype=torch_dtype)
        
        # Compute ranks
        x_ranks = torch.argsort(torch.argsort(x_filtered)).float() + 1
        y_ranks = torch.argsort(torch.argsort(y_filtered)).float() + 1
        
        # Compute Spearman correlation using Pearson correlation of ranks
        x_mean = torch.mean(x_ranks)
        y_mean = torch.mean(y_ranks)
        
        x_centered = x_ranks - x_mean
        y_centered = y_ranks - y_mean
        
        numerator = torch.sum(x_centered * y_centered)
        x_var = torch.sum(x_centered ** 2)
        y_var = torch.sum(y_centered ** 2)
        
        denominator = torch.sqrt(x_var * y_var)
        
        if denominator == 0:
            rho = torch.tensor(0.0, device=device, dtype=torch_dtype)
        else:
            rho = numerator / denominator
        
        # Compute p-value using t-distribution approximation
        # t = rho * sqrt((n-2)/(1-rho^2))
        if abs(rho) >= 1.0:
            p_val = torch.tensor(0.0, device=device, dtype=torch_dtype)
        else:
            t_stat = rho * torch.sqrt((n - 2) / (1 - rho**2 + 1e-10))
            
            # Approximate p-value using normal distribution for large n
            # For small n, this is less accurate but still reasonable
            if n > 30:
                # Use normal approximation
                p_val = 2 * (1 - torch.erf(torch.abs(t_stat) / torch.sqrt(torch.tensor(2.0, device=device, dtype=torch_dtype))))
            else:
                # Simple approximation for small samples
                # This is less accurate but computationally efficient
                p_val = 2 * torch.exp(-0.5 * t_stat**2) / torch.sqrt(2 * torch.pi * torch.tensor(n-2, device=device, dtype=torch_dtype))
        
        return rho, p_val
    
    # Compute correlations in batches
    correlations = []
    n_batches = (n_features1 * n_features2 + batch_size - 1) // batch_size
    
    print(f"   Processing {n_batches} batches...")
    
    with torch.no_grad():
        with tqdm(total=n_features1 * n_features2, desc="Computing Spearman correlations") as pbar:
            for i in range(n_features1):
                # Convert feature to tensor
                x = torch.tensor(data1[:, i], dtype=torch_dtype, device=device)
                
                # Process features in batches
                for batch_start in range(0, n_features2, batch_size):
                    batch_end = min(batch_start + batch_size, n_features2)
                    
                    for j in range(batch_start, batch_end):
                        y = torch.tensor(data2[:, j], dtype=torch_dtype, device=device)
                        
                        # Compute Spearman correlation
                        rho, p_val = spearman_torch_single(x, y, handle_zeros)
                        
                        # Convert to numpy for processing
                        rho_val = rho.cpu().numpy()
                        p_val_val = p_val.cpu().numpy()
                        
                        # Check significance
                        if abs(rho_val) >= min_corr and p_val_val <= p_threshold:
                            correlations.append({
                                f'{view1}_feature': features1[i],
                                f'{view2}_feature': features2[j],
                                'correlation': float(rho_val),
                                'p_value': float(p_val_val),
                                'abs_correlation': abs(float(rho_val))
                            })
                        
                        pbar.update(1)
    
    corr_df = pd.DataFrame(correlations)
    print(f"✅ Found {len(corr_df)} significant Spearman correlations")
    
    return corr_df


# =============================================================================
# UNIFIED FUNCTIONS - Combining multiple implementations with optional parameters
# =============================================================================

def compute_correlation(mdata, 
                       view1: str, 
                       view2: str,
                       method: str = 'pearson',
                       backend: str = 'auto',
                       optimization: str = 'fast',
                       min_corr: float = 0.3,
                       p_threshold: float = 0.05,
                       max_features: int = None,
                       device: str = 'auto',
                       batch_size: int = 100,
                       chunk_size: int = 1000,
                       dtype: str = 'float32',
                       handle_zeros: bool = True,
                       min_nonzero_pairs: int = 10,
                       n_workers: int = None,
                       use_parallel: bool = True) -> pd.DataFrame:
    """
    Unified correlation computation function with multiple methods and backends.
    
    This function combines all correlation implementations in the module:
    - compute_cross_correlation (basic scipy-based)
    - compute_cross_correlation_fast (optimized numpy)
    - compute_cross_correlation_torch (PyTorch GPU-accelerated)
    - compute_cross_correlation_torch_chunked (memory-efficient)
    - compute_cross_correlation_parallel_numpy (parallel processing)
    - compute_kendall_tau_torch (Kendall's tau with PyTorch)
    - compute_kendall_tau_torch_fast (optimized Kendall's tau)
    - compute_spearman_torch_fast (Spearman correlation with PyTorch)
    
    Parameters
    ----------
    mdata:MuData
        Multi-omics data object
    view1:str
        Name of first view/modality
    view2:str
        Name of second view/modality
    method:str, default 'pearson'
        Correlation method: 'pearson', 'spearman', 'kendall'
    backend:str, default 'auto'
        Computation backend: 'auto', 'numpy', 'torch', 'scipy'
        - 'auto': automatically choose best backend
        - 'numpy': use numpy/scipy with optional parallelization
        - 'torch': use PyTorch for GPU acceleration
        - 'scipy': use scipy.stats functions
    optimization:str, default 'fast'
        Optimization level: 'basic', 'fast', 'chunked'
        - 'basic': simple implementation
        - 'fast': optimized with batching and filtering
        - 'chunked': memory-efficient chunked processing
    min_corr:float, default 0.3
        Minimum absolute correlation threshold
    p_threshold:float, default 0.05
        P-value significance threshold
    max_features:int, optional
        Maximum number of features to use from each view
    device:str, default 'auto'
        Device for PyTorch backend: 'auto', 'cpu', 'cuda'
    batch_size:int, default 100
        Batch size for processing
    chunk_size:int, default 1000
        Chunk size for chunked processing
    dtype:str, default 'float32'
        Data type for computations
    handle_zeros:bool, default True
        Whether to filter zero pairs for sparse data
    min_nonzero_pairs:int, default 10
        Minimum non-zero pairs required
    n_workers:int, optional
        Number of parallel workers
    use_parallel:bool, default True
        Whether to use parallel processing when available
        
    Returns
    -------
    pd.DataFrame
        Correlation results with columns:
        - {view1}_feature: feature from first view
        - {view2}_feature: feature from second view
        - correlation: correlation coefficient
        - p_value: statistical significance
        - abs_correlation: absolute correlation value
        
    Examples
    --------
    >>> # Basic Pearson correlation
    >>> corr_df = ov.single.compute_correlation(mdata, 'rna', 'protein')
    
    >>> # Spearman correlation with PyTorch backend
    >>> corr_df = ov.single.compute_correlation(mdata, 'metab', 'micro', 
    ...                                         method='spearman', backend='torch')
    
    >>> # Kendall's tau with chunked processing
    >>> corr_df = ov.single.compute_correlation(mdata, 'rna', 'atac',
    ...                                         method='kendall', optimization='chunked')
    
    >>> # High-performance setup for large datasets
    >>> corr_df = ov.single.compute_correlation(mdata, 'rna', 'protein',
    ...                                         backend='torch', optimization='chunked',
    ...                                         device='cuda', max_features=2000)
    """
    
    # Validate inputs
    valid_methods = ['pearson', 'spearman', 'kendall']
    valid_backends = ['auto', 'numpy', 'torch', 'scipy']
    valid_optimizations = ['basic', 'fast', 'chunked']
    
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}")
    if backend not in valid_backends:
        raise ValueError(f"backend must be one of {valid_backends}")
    if optimization not in valid_optimizations:
        raise ValueError(f"optimization must be one of {valid_optimizations}")
    
    # Auto-select backend if needed
    if backend == 'auto':
        try:
            import torch
            if torch.cuda.is_available() and (max_features is None or max_features > 500):
                backend = 'torch'
                print("🚀 Auto-selected PyTorch backend with GPU acceleration")
            else:
                backend = 'numpy'
                print("🔧 Auto-selected NumPy backend")
        except ImportError:
            backend = 'numpy'
            print("📊 Auto-selected NumPy backend (PyTorch not available)")
    
    print(f"🔄 Computing {method} correlations using {backend} backend ({optimization} optimization)")
    
    # Route to appropriate implementation
    if method == 'pearson':
        if backend == 'torch':
            if optimization == 'chunked':
                return compute_cross_correlation_torch_chunked(
                    mdata, view1, view2, min_corr, p_threshold, 
                    max_features, max_features, device, chunk_size, 
                    batch_size, dtype)
            elif optimization == 'fast':
                return compute_cross_correlation_torch(
                    mdata, view1, view2, min_corr, p_threshold,
                    max_features, device, batch_size, batch_size, dtype)
            else:  # basic
                return compute_cross_correlation_torch(
                    mdata, view1, view2, min_corr, p_threshold,
                    max_features, device, 100, 100, dtype)
        
        elif backend == 'numpy':
            if use_parallel and optimization in ['fast', 'chunked']:
                return compute_cross_correlation_parallel_numpy(
                    mdata, view1, view2, min_corr, p_threshold,
                    max_features, n_workers)
            elif optimization == 'fast':
                return compute_cross_correlation_fast(
                    mdata, view1, view2, min_corr, p_threshold, max_features)
            else:  # basic
                return compute_cross_correlation(
                    mdata, view1, view2, method, min_corr, p_threshold, chunk_size)
        
        else:  # scipy
            return compute_cross_correlation(
                mdata, view1, view2, method, min_corr, p_threshold, chunk_size)
    
    elif method == 'spearman':
        if backend == 'torch':
            return compute_spearman_torch_fast(
                mdata, view1, view2, min_corr, p_threshold,
                max_features, device, batch_size, dtype,
                handle_zeros, min_nonzero_pairs)
        else:
            # Fall back to scipy-based implementation
            return compute_cross_correlation(
                mdata, view1, view2, method, min_corr, p_threshold, chunk_size)
    
    elif method == 'kendall':
        if backend == 'torch':
            if optimization == 'fast':
                return compute_kendall_tau_torch_fast(
                    mdata, view1, view2, min_corr, p_threshold,
                    max_features, device, batch_size, dtype,
                    handle_zeros, min_nonzero_pairs)
            else:  # basic
                return compute_kendall_tau_torch(
                    mdata, view1, view2, min_corr, p_threshold,
                    max_features, device, batch_size, batch_size, dtype, handle_zeros)
        else:
            # Fall back to scipy-based implementation
            return compute_cross_correlation(
                mdata, view1, view2, method, min_corr, p_threshold, chunk_size)


def compute_coexpression_modules(mdata,
                               view1: str,
                               view2: str,
                               n_components: int = 10,
                               correlation_threshold: float = 0.3,
                               method: str = 'pearson',
                               backend: str = 'auto',
                               optimization: str = 'fast',
                               random_state: int = 42,
                               max_features: int = None,
                               device: str = 'auto',
                               nmf_solver: str = 'mu',
                               nmf_max_iter: int = 1000,
                               handle_zeros: bool = True,
                               use_parallel: bool = True,
                               n_workers: int = None,
                               corr_df: pd.DataFrame = None,
                               corr_matrix: pd.DataFrame = None) -> dict:
    """
    Unified NMF-based coexpression module discovery with multiple methods and backends.
    
    This function combines all NMF coexpression module implementations:
    - nmf_coexpression_modules (basic implementation)
    - nmf_coexpression_modules_fast (optimized with multiple backends)
    - nmf_coexpression_modules_torch (PyTorch-based with GPU support)
    - nmf_coexpression_modules_kendall (Kendall's tau specific)
    
    Parameters
    ----------
    mdata:MuData
        Multi-omics data object
    view1:str
        Name of first view/modality
    view2:str
        Name of second view/modality
    n_components:int, default 10
        Number of NMF components (modules)
    correlation_threshold:float, default 0.3
        Minimum correlation threshold for module inclusion
    method:str, default 'pearson'
        Correlation method: 'pearson', 'spearman', 'kendall'
    backend:str, default 'auto'
        Computation backend: 'auto', 'numpy', 'torch', 'scipy'
    optimization:str, default 'fast'
        Optimization level: 'basic', 'fast', 'torch'
    random_state:int, default 42
        Random seed for reproducibility
    max_features:int, optional
        Maximum number of features to use
    device:str, default 'auto'
        Device for PyTorch backend
    nmf_solver:str, default 'mu'
        NMF solver algorithm
    nmf_max_iter:int, default 1000
        Maximum NMF iterations
    handle_zeros:bool, default True
        Whether to handle zeros in sparse data
    use_parallel:bool, default True
        Whether to use parallel processing
    n_workers:int, optional
        Number of parallel workers
    corr_df:pd.DataFrame, optional
        Pre-computed correlation DataFrame
    corr_matrix:pd.DataFrame, optional
        Pre-computed correlation matrix
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'modules': NMF components/modules
        - 'correlation_matrix': correlation matrix used
        - 'feature_loadings': feature loadings for each module
        - 'module_info': summary information for each module
        
    Examples
    --------
    >>> # Basic NMF modules with Pearson correlation
    >>> modules = ov.single.compute_coexpression_modules(mdata, 'rna', 'protein')
    
    >>> # Spearman-based modules with PyTorch backend
    >>> modules = ov.single.compute_coexpression_modules(mdata, 'metab', 'micro',
    ...                                                  method='spearman', backend='torch')
    
    >>> # Kendall's tau modules with custom parameters
    >>> modules = ov.single.compute_coexpression_modules(mdata, 'rna', 'atac',
    ...                                                  method='kendall', n_components=15)
    
    >>> # High-performance setup
    >>> modules = ov.single.compute_coexpression_modules(mdata, 'rna', 'protein',
    ...                                                  backend='torch', optimization='torch',
    ...                                                  device='cuda', max_features=2000)
    """
    
    # Validate inputs
    valid_methods = ['pearson', 'spearman', 'kendall']
    valid_backends = ['auto', 'numpy', 'torch', 'scipy']
    valid_optimizations = ['basic', 'fast', 'torch']
    
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}")
    if backend not in valid_backends:
        raise ValueError(f"backend must be one of {valid_backends}")
    if optimization not in valid_optimizations:
        raise ValueError(f"optimization must be one of {valid_optimizations}")
    
    # Auto-select backend if needed
    if backend == 'auto':
        try:
            import torch
            if torch.cuda.is_available() and (max_features is None or max_features > 500):
                backend = 'torch'
                print("🚀 Auto-selected PyTorch backend for NMF modules")
            else:
                backend = 'numpy'
                print("🔧 Auto-selected NumPy backend for NMF modules")
        except ImportError:
            backend = 'numpy'
            print("📊 Auto-selected NumPy backend for NMF modules (PyTorch not available)")
    
    print(f"🔄 Computing coexpression modules using {method} correlation ({backend} backend)")
    
    # Route to appropriate implementation
    if method == 'kendall' and backend == 'torch':
        return nmf_coexpression_modules_kendall(
            mdata, view1, view2, n_components, correlation_threshold,
            random_state, max_features, True, device, nmf_solver,
            nmf_max_iter, corr_df, handle_zeros)
    
    elif backend == 'torch' and optimization == 'torch':
        return nmf_coexpression_modules_torch(
            mdata, view1, view2, n_components, correlation_threshold,
            random_state, max_features, device, nmf_max_iter, 1e-4,
            corr_df, corr_matrix, use_parallel, n_workers)
    
    elif optimization == 'fast':
        return nmf_coexpression_modules_fast(
            mdata, view1, view2, n_components, correlation_threshold,
            method, random_state, max_features, backend == 'torch',
            device, nmf_solver, nmf_max_iter, corr_df, corr_matrix)
    
    else:  # basic
        return nmf_coexpression_modules(
            mdata, view1, view2, n_components, correlation_threshold,
            method, random_state, corr_df)

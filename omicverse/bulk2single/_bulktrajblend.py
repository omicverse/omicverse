
import pandas as pd
import numpy as np
import anndata

import torch
from typing import Union, Optional, Any

from ..bulk import data_drop_duplicates_index, deseq2_normalize
from ..bulk2single import Bulk2Single
from ..single import scnocd
from ._vae import train_vae, generate_vae, load_vae
from .._registry import register_function

@register_function(
    aliases=['Bulk轨迹融合', 'BulkTrajBlend', 'bulk trajectory blend'],
    category="bulk2single",
    description="Integrate bulk and single-cell signals to infer transitional cell states and trajectory dynamics through bulk-to-single reconstruction plus graph modeling.",
    prerequisites={'functions': ['Bulk2Single']},
    requires={'obs': ['celltype labels']},
    produces={'obs': ['trajectory-related annotations'], 'uns': ['trajectory blend results']},
    auto_fix='none',
    examples=['bulktb = ov.bulk2single.BulkTrajBlend(bulk_seq=bulk, single_seq=adata, celltype_key="celltype")', 'adata1 = bulktb.train(spot_num=100, cell_num=10)'],
    related=['bulk2single.Bulk2Single', 'utils.cal_paga', 'utils.plot_paga']
)
class BulkTrajBlend(object):
    """
    Integrate bulk and single-cell information to infer transitional cell-state trajectories.

    Parameters
    ----------
    bulk_seq:pd.DataFrame
        Bulk expression matrix with genes in rows and samples in columns.
    single_seq:anndata.AnnData
        Reference single-cell dataset used to define cell identities.
    celltype_key:str
        Column name in ``single_seq.obs`` containing cell-type labels.
    bulk_group:Optional[Any]
        Optional grouping key/list for averaging bulk replicates.
    max_single_cells:int
        Maximum number of reference cells retained for model fitting.
    top_marker_num:int
        Number of top marker genes used in Bulk2Single preparation.
    ratio_num:int
        Ratio controlling generated cell numbers per cell type.
    gpu:Union[int,str]
        Compute device specification (CUDA index, ``'mps'``, or CPU fallback).
    
    Returns
    -------
    None
        Initializes bulk-trajectory blending workflow.
    
    Examples
    --------
    >>> bulktb = ov.bulk2single.BulkTrajBlend(bulk_seq=bulk, single_seq=adata, celltype_key="celltype")
    """

    def __init__(self, bulk_seq: pd.DataFrame, single_seq: anndata.AnnData,
                 celltype_key: str, bulk_group: Optional[Any] = None, max_single_cells: int = 5000,
                 top_marker_num: int = 500, ratio_num: int = 1, gpu: Union[int, str] = 0) -> None:
        r"""
        Initialize BulkTrajBlend for trajectory inference and cell blending.

        Parameters
        ----------
        bulk_seq:pd.DataFrame
            Bulk RNA-seq matrix with genes as rows and samples as columns.
        single_seq:anndata.AnnData
            Single-cell reference AnnData used for cell-state prior information.
        celltype_key:str
            Column in ``single_seq.obs`` that stores cell-type annotation.
        bulk_group:Optional[Any]
            Optional grouping used to aggregate bulk replicates.
        max_single_cells:int
            Maximum number of single cells used in internal Bulk2Single model.
        top_marker_num:int
            Number of marker genes per cell type used during preparation.
        ratio_num:int
            Cell-number ratio used when converting fractions into target counts.
        gpu:Union[int,str]
            Compute device selector for VAE/GNN workflows.

        Returns
        -------
            None
        """

        self.bulk_seq = bulk_seq.copy()
        self.single_seq = single_seq.copy()
        self.celltype_key=celltype_key
        self.top_marker_num=top_marker_num
        self.ratio_num=ratio_num
        self.gpu=gpu
        self.group=bulk_group
        self.max_single_cells=max_single_cells
        if gpu=='mps' and torch.backends.mps.is_available():
            print('Note that mps may loss will be nan, used it when torch is supported')
            self.used_device = torch.device("mps")
        else:
            self.used_device = torch.device(f"cuda:{gpu}") if gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
        self.history=[]
        data_dg_v=self.bulk_seq.mean(axis=1)
        data_dg=pd.DataFrame(index=data_dg_v.index)
        data_dg['group']=data_dg_v
        self.bulk_seq_group=data_dg
        pass

    def bulk_preprocess_lazy(self,)->None:
        r"""
        Preprocess bulk RNA-seq data for trajectory analysis.
        
        Performs normalization, log transformation, and optional group averaging
        of bulk data for downstream trajectory inference.

        Parameters
        ----------
        None
            
        Returns
        -------
        None
            Updates ``self.bulk_seq`` and ``self.bulk_seq_group`` in place.
        """

        print("......drop duplicates index in bulk data")
        self.bulk_seq=data_drop_duplicates_index(self.bulk_seq)
        print("......deseq2 normalize the bulk data")
        self.bulk_seq=deseq2_normalize(self.bulk_seq)
        print("......log10 the bulk data")
        self.bulk_seq=np.log10(self.bulk_seq+1)
        print("......calculate the mean of each group")
        if self.group is None:
            return None
        else:
            data_dg_v=self.bulk_seq[self.group].mean(axis=1)
            data_dg=pd.DataFrame(index=data_dg_v.index)
            data_dg['group']=data_dg_v
            self.bulk_seq_group=data_dg
        return None
    
    def single_preprocess_lazy(self,target_sum:int=1e4)->None:
        r"""
        Preprocess single-cell reference data for trajectory analysis.
        
        Normalizes single-cell data and makes cell/gene names unique for
        consistent integration with bulk data.

        Parameters
        ----------
        target_sum:int
            Library-size target used for total-count normalization.
            
        Returns
        -------
        None
            Updates ``self.single_seq`` in place.
        """

        print("......normalize the single data")
        self.single_seq.obs_names_make_unique()
        self.single_seq.var_names_make_unique()
        from ..pp._preprocess import normalize_total,log1p
        normalize_total(self.single_seq, target_sum=target_sum)
        print("......log1p the single data")
        log1p(self.single_seq)
        return None
    
    def vae_configure(self, cell_target_num: Optional[int] = None, **kwargs: Any) -> None:
        r"""
        Configure the VAE model for bulk-to-single-cell generation.

        Sets up the Bulk2Single model with cell-type target numbers either from
        deconvolution prediction or manual specification.

        Parameters
        ----------
        cell_target_num:Optional[int]
            Fixed number of generated cells per cell type. If ``None``, cell
            fractions are first estimated from bulk data.
        **kwargs:Any
            Extra keyword arguments forwarded to ``predicted_fraction``.

        Returns
        -------
            None
        """
        self.vae_model=Bulk2Single(bulk_data=self.bulk_seq,single_data=self.single_seq,
                                   celltype_key=self.celltype_key,bulk_group=self.group,
                                      max_single_cells=self.max_single_cells,
                 top_marker_num=self.top_marker_num,ratio_num=self.ratio_num,gpu=self.gpu)
        if cell_target_num!=None:
            self.vae_model.cell_target_num=dict(zip(list(set(self.single_seq.obs[self.celltype_key])),
                                                [cell_target_num]*len(list(set(self.single_seq.obs[self.celltype_key])))))
        else:
            self.cellfract=self.vae_model.predicted_fraction(**kwargs)
        
        self.sc_ref=self.vae_model.sc_ref.copy()
        self.bulk_ref=self.vae_model.bulk_data.T.copy()

        self.vae_model.bulk_preprocess_lazy()
        self.vae_model.single_preprocess_lazy()
        self.vae_model.prepare_input()

        

    def vae_train(self,
                  vae_save_dir:str='save_model',
            vae_save_name:str='vae',
            generate_save_dir:str='output',
            generate_save_name:str='output',
            batch_size:int=512,
            learning_rate:float=1e-4,
            hidden_size:int=256,
            epoch_num:int=5000,
            patience:int=50,save:bool=True):
        r"""
        Train the VAE model for trajectory-aware single-cell generation.
        
        Trains the underlying Bulk2Single VAE model to generate synthetic single
        cells that preserve trajectory information from bulk data.

        Parameters
        ----------
        vae_save_dir:str
            Directory where trained VAE weights will be saved.
        vae_save_name:str
            Filename prefix for the saved VAE checkpoint.
        generate_save_dir:str
            Reserved output directory for generated single-cell matrices.
        generate_save_name:str
            Reserved output filename prefix for generated matrices.
        batch_size:int
            Mini-batch size for VAE optimization.
        learning_rate:float
            Learning rate for VAE optimizer.
        hidden_size:int
            Hidden dimension of encoder/decoder blocks.
        epoch_num:int
            Maximum training epochs.
        patience:int
            Early-stopping patience.
        save:bool
            Whether to persist trained model and metadata.
            
        Returns
        -------
        None
            Updates ``self.vae_net`` with the trained model.
        """
        
        self.vae_net=self.vae_model.train(
                batch_size=batch_size,
                learning_rate=learning_rate,
                hidden_size=hidden_size,
                epoch_num=epoch_num,
                vae_save_dir=vae_save_dir,
                vae_save_name=vae_save_name,
                generate_save_dir=generate_save_dir,
                generate_save_name=generate_save_name,
                patience=patience,save=save)
        
        
    def vae_load(self,vae_load_dir:str,hidden_size:int=256):
        r"""
        Load a pre-trained VAE model for trajectory analysis.
        
        Loads a previously trained VAE model for generating trajectory-aware
        single-cell data.

        Parameters
        ----------
        vae_load_dir:str
            Path to a saved VAE checkpoint.
        hidden_size:int
            Hidden dimension used to reconstruct model architecture.
            
        Returns
        -------
        None
            Loads weights into the internal VAE model.
        """

        print(f'loading model from {vae_load_dir}')
        vae_net = self.vae_model.load(vae_load_dir,hidden_size=hidden_size)
        self.vae_net=vae_net

    def vae_generate(self,highly_variable_genes:bool=True,max_value:float=10,
                     n_comps:int=100,svd_solver:str='auto',leiden_size:int=50)->anndata.AnnData:
        r"""
        Generate trajectory-aware single-cell data with quality filtering.
        
        Uses the trained VAE to generate synthetic single cells and applies
        quality control filtering to remove noisy clusters.

        Parameters
        ----------
        highly_variable_genes:bool
            Whether to restrict clustering to highly variable genes.
        max_value:float
            Clipping threshold used during scaling.
        n_comps:int
            Number of principal components for graph construction.
        svd_solver:str
            SVD solver used in PCA.
        leiden_size:int
            Minimum accepted Leiden cluster size; smaller clusters are removed.

        Returns
        -------
        anndata.AnnData
            Generated AnnData after filtering noisy clusters.
        """
        from ..pp import highly_variable_genes as hvg
        from .._sc_compat import scale_X, pca_X, neighbors, leiden

        generate_adata=self.vae_model.generate()
        self.generate_adata_raw=generate_adata.copy()
        generate_adata.raw = generate_adata
        if highly_variable_genes:
            hvg(generate_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
            generate_adata = generate_adata[:, generate_adata.var.highly_variable]
        scale_X(generate_adata, max_value=max_value)
        pca_X(generate_adata, n_comps=n_comps, svd_solver=svd_solver)
        neighbors(generate_adata, use_rep="X_pca")
        leiden(generate_adata)
        filter_leiden=list(generate_adata.obs['leiden'].value_counts()[generate_adata.obs['leiden'].value_counts()<leiden_size].index)
        generate_adata.uns['noisy_leiden']=filter_leiden
        print("The filter leiden is ",filter_leiden)
        generate_adata=generate_adata[~generate_adata.obs['leiden'].isin(filter_leiden)]
        self.generate_adata=generate_adata.copy()
        return generate_adata.raw.to_adata()
    
    def gnn_configure(self, use_rep: str = 'X', neighbor_rep: str = 'X_pca',
                      gpu: Union[int, str] = 0, hidden_size: int = 128,
                     weight_decay: float = 1e-2,
                     dropout: float = 0.5,
                     batch_norm: bool = True,
                     lr: float = 1e-3,
                     max_epochs: int = 500,
                     display_step: int = 25,
                     balance_loss: bool = True,
                     stochastic_loss: bool = True,
                     batch_size: int = 2000, num_workers: int = 5) -> None:
        r"""
        Configure Graph Neural Network for trajectory and transition state analysis.

        Sets up the NOCD (Non-Overlapping Cell-type Decomposition) GNN model
        for identifying cellular trajectories and intermediate states.

        Parameters
        ----------
        use_rep:str
            Representation key in ``adata`` used as GNN input features.
        neighbor_rep:str
            Representation key used to build neighborhood graph.
        gpu:Union[int,str]
            Device selector for GNN training.
        hidden_size:int
            Hidden dimension of GNN layers.
        weight_decay:float
            L2 regularization strength.
        dropout:float
            Dropout probability.
        batch_norm:bool
            Whether to apply batch normalization.
        lr:float
            Learning rate for GNN optimization.
        max_epochs:int
            Maximum number of GNN training epochs.
        display_step:int
            Epoch interval for progress logging.
        balance_loss:bool
            Whether to balance positive/negative terms in the objective.
        stochastic_loss:bool
            Whether to enable stochastic objective approximation.
        batch_size:int
            Batch size for edge/node sampling.
        num_workers:int
            Number of workers for preprocessing data loaders.

        Returns
        -------
            None
        """
        nocd_obj=scnocd(self.generate_adata,use_rep=use_rep,
                        neighbor_rep=neighbor_rep,gpu=gpu)
        #nocd_obj.device = torch.device(f"cuda:{gpu}") if gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
        nocd_obj.matrix_transform(clustertype=self.celltype_key)
        nocd_obj.matrix_normalize()
        nocd_obj.GNN_configure(hidden_size=hidden_size,weight_decay=weight_decay,
                             dropout=dropout,batch_norm=batch_norm,lr=lr,
                             max_epochs=max_epochs,display_step=display_step,
                             balance_loss=balance_loss,stochastic_loss=stochastic_loss,
                             batch_size=batch_size)
        nocd_obj.GNN_preprocess(num_workers=num_workers)

        self.nocd_obj=nocd_obj
    
    def gnn_train(self,thresh:float=0.5,gnn_save_dir:str='save_model',
            gnn_save_name:str='gnn'):
        r"""
        Train the GNN model for trajectory and transition state inference.
        
        Trains the NOCD model to identify cellular trajectories and overlapping
        cell communities representing transition states.

        Parameters
        ----------
        thresh:float
            Probability threshold for assigning community membership.
        gnn_save_dir:str
            Directory where trained GNN model is saved.
        gnn_save_name:str
            Filename prefix for saved GNN checkpoint.
            
        Returns
        -------
        None
            Fits the NOCD model and stores inferred communities.
        """
        self.nocd_obj.GNN_model()
        self.nocd_obj.GNN_result(thresh=thresh)
        self.nocd_obj.cal_nocd()
        self.nocd_obj.save(gnn_save_dir=gnn_save_dir,gnn_save_name=gnn_save_name)

    def gnn_load(self,gnn_load_dir:str,thresh:float=0.5,):
        r"""
        Load a pre-trained GNN model for trajectory analysis.
        
        Loads a previously trained NOCD model and computes trajectory results.

        Parameters
        ----------
        gnn_load_dir:str
            Path to directory containing saved GNN checkpoint.
        thresh:float
            Probability threshold for assigning community membership.
            
        Returns
        -------
        None
            Loads trained model and recomputes community assignments.
        """
        self.nocd_obj.load(gnn_load_dir)
        self.nocd_obj.GNN_result(thresh=thresh)
        self.nocd_obj.cal_nocd()
    
    def gnn_generate(self)->pd.DataFrame:
        r"""
        Generate overlapping cell communities representing transition states.
        
        Identifies and names cell communities based on trajectory analysis,
        creating binary matrix indicating community membership for each cell.

        Parameters
        ----------
        None
            
        Returns
        -------
        pd.DataFrame
            Binary matrix of NOCD community assignments per cell.
        """
        '''
        pair_dict_r={}
        for i in range(self.nocd_obj.K):
            j=0
            while 1:
                if self.nocd_obj.adata[self.nocd_obj.adata.obs['nocd_n']==str(i)].shape[0]==0:
                    break
                if j>=len(self.nocd_obj.adata[self.nocd_obj.adata.obs['nocd_n']==str(i)].obs.value_counts(self.celltype_key).index):
                    pair_dict_r[str(i)]=self.nocd_obj.adata[self.nocd_obj.adata.obs['nocd_n']==str(i)].obs.value_counts(self.celltype_key).index[j-1]+'_'+str(j)
                    break
                if self.nocd_obj.adata[self.nocd_obj.adata.obs['nocd_n']==str(i)].obs.value_counts(self.celltype_key).index[j] not in list(pair_dict_r.values()):
                    pair_dict_r[str(i)]=self.nocd_obj.adata[self.nocd_obj.adata.obs['nocd_n']==str(i)].obs.value_counts(self.celltype_key).index[j]
                    break
                else:
                    j+=1
        pair_dict_r
        '''
        unique_adata=self.nocd_obj.adata[~self.nocd_obj.adata.obs['nocd_n'].str.contains('-')]
        pair_dict_r={}
        repeat_celltype=dict(zip(list(set(unique_adata.obs[self.celltype_key])),np.zeros(len(list(set(unique_adata.obs[self.celltype_key]))))))
        for nocd_class in list(set(unique_adata.obs['nocd_n'])):
            now_celltype=unique_adata[unique_adata.obs['nocd_n']==nocd_class].obs.value_counts(self.celltype_key).index[0]
            if (now_celltype in pair_dict_r.values()):
                #print(now_celltype)
                pair_dict_r[str(nocd_class)]=now_celltype+'_'+str(int(repeat_celltype[now_celltype]))
                repeat_celltype[now_celltype]+=1
            else:
                pair_dict_r[str(nocd_class)]=now_celltype
                repeat_celltype[now_celltype]+=1
        
        # Ensure all keys from 0 to K-1 exist in pair_dict_r to prevent KeyError
        for j in range(self.nocd_obj.K):
            if str(j) not in pair_dict_r:
                pair_dict_r[str(j)] = f'unknown_{j}'

        def li_range(li,max_len):
            r=[0]*max_len   
            for i in li:
                r[int(i)]=1
            return r
        
        res_li=[li_range(i.split('-'),self.nocd_obj.K) for i in self.nocd_obj.adata.obs['nocd_n']]
        res_pd=pd.DataFrame(res_li,index=self.nocd_obj.adata.obs.index,columns=['nocd_'+i for i in [pair_dict_r[str(j)] for j in range(self.nocd_obj.K)]])
        print("The nocd result is ",res_pd.sum(axis=0))
        print("The nocd result has been added to adata.obs['nocd_']")
        self.nocd_obj.adata.obs=pd.concat([self.nocd_obj.adata.obs,res_pd],axis=1)
        return res_pd 
    
    def interpolation(self,celltype:str,adata:anndata.AnnData=None,)->anndata.AnnData:
        r"""
        Interpolate trajectory communities back to original data space.
        
        Integrates identified cell communities from generated data back with
        original single-cell reference data for downstream analysis.

        Parameters
        ----------
        celltype:str
            Community label suffix used in ``obs['nocd_<celltype>']``.
        adata:anndata.AnnData or None
            Target AnnData to merge with inferred transition-state cells. If
            ``None``, ``self.single_seq`` is used.

        Returns
        -------
        anndata.AnnData
            Concatenated AnnData containing selected transition cells and source data.
        """
        if adata is None:
            adata=self.single_seq
        test_adata=self.nocd_obj.adata[self.nocd_obj.adata.obs['nocd_{}'.format(celltype)]==1].raw.to_adata()
        if test_adata.shape[0]!=0:
            adata1=anndata.concat([test_adata,
                        adata],merge='same')
        else:
            adata1=adata 
            print("The cell type {} is not in the nocd result".format(celltype))
        return adata1
    

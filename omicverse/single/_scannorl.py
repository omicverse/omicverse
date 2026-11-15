import pandas as pd
import numpy as np
import anndata
import torch
from typing import Union

from ._nocd import scnocd

class scANNORL(object):
    
    def __init__(self,adata,cluster_key):
        
        self.adata=adata
        self.cluster_key=cluster_key
        
    def gnn_configure(self,gpu=0,hidden_size:int=128,
                     weight_decay:int=1e-2,
                     dropout:float=0.5,
                     batch_norm:bool=True,
                     lr:int=1e-3,
                     max_epochs:int=300,
                     display_step:int=25,
                     balance_loss:bool=True,
                     stochastic_loss:bool=True,
                     batch_size:int=2000,num_workers:int=5,):
        """
        Configure the GNN model of BulkTrajBlend.

        Parameters
        ----------
            gpu: The GPU ID for training the GNN model. Default is 0.
            hidden_size: The hidden size for the GNN model. Default is 128.
            weight_decay: The weight decay for the GNN model. Default is 1e-2.
            dropout: The dropout for the GNN model. Default is 0.5.
            batch_norm: Whether to use batch normalization for the GNN model. Default is True.
            lr: The learning rate for the GNN model. Default is 1e-3.
            max_epochs: The maximum epoch number for training the GNN model. Default is 500.
            display_step: The display step for training the GNN model. Default is 25.
            balance_loss: Whether to use the balance loss for training the GNN model. Default is True.
            stochastic_loss: Whether to use the stochastic loss for training the GNN model. Default is True.
            batch_size: The batch size for training the GNN model. Default is 2000.
            num_workers: The number of workers for training the GNN model. Default is 5.

        Returns
        -------
        None
        """
        nocd_obj=ov.single.scnocd(self.adata,gpu=gpu)
        #nocd_obj.device = torch.device(f"cuda:{gpu}") if gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
        nocd_obj.matrix_transform(clustertype=self.cluster_key)
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
        """
        Train the GNN model of BulkTrajBlend.

        Parameters
        ----------
            thresh: The threshold for the GNN model. Default is 0.5.
            gnn_save_dir: The directory for saving the GNN model. Default is 'save_model'.
            gnn_save_name: The name for saving the GNN model. Default is 'gnn'.

        Returns
        -------
        None
        """
        self.nocd_obj.GNN_model()
        self.nocd_obj.GNN_result(thresh=thresh)
        self.nocd_obj.save(gnn_save_dir=gnn_save_dir,gnn_save_name=gnn_save_name)

    def gnn_load(self,gnn_load_dir:str,thresh:float=0.5,):
        """
        Load the GNN model of BulkTrajBlend.

        Parameters
        ----------
            gnn_load_dir: The directory for loading the GNN model.
            thresh: The threshold for the GNN model. Default is 0.5.

        Returns
        -------
        None
        """
        self.nocd_obj.load(gnn_load_dir)
        self.nocd_obj.GNN_result(thresh=thresh)
    
    def cluster(self):
        
        self.nocd_obj.cal_nocd()
        self.nocd_obj.adata.obs['nocd_cluster']=self.nocd_obj.adata.obs['nocd_n'].copy()
        self.nocd_obj.adata.obs['nocd_cluster']=self.nocd_obj.adata.obs['nocd_cluster'].astype(str)
        self.nocd_obj.adata.obs.loc[self.nocd_obj.adata.obs['nocd_n'].str.contains('-'),'nocd_cluster']='Unknown'
        self.nocd_obj.adata.obs['nocd_cluster']=self.nocd_obj.adata.obs['nocd_cluster'].astype(str)
        self.nocd_obj.adata.obs['scANNORL_cluster']=self.nocd_obj.adata.obs['nocd_cluster'].copy()
        test_=self.nocd_obj.adata.obs.loc[self.nocd_obj.adata.obs['nocd_cluster']!='Unknown','nocd_cluster'].value_counts()
        pair_dict=dict(zip(test_.index,[ i for i in range(len(test_.index))]))
        for i in test_.index:
            self.nocd_obj.adata.obs.loc[self.nocd_obj.adata.obs['nocd_cluster']==i,'scANNORL_cluster']=str(pair_dict[i])
        
        return self.nocd_obj.adata
        
    


import random
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad

import os
import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
#from torch.utils.tensorboard import SummaryWriter
import time
import anndata
from .._settings import add_reference
from .._registry import register_function


def _get_tosica_backend():
    from ..external.tosica import fit_model
    from ..external.tosica.TOSICA_model import scTrans_model as create_model
    from ..external.tosica.pre import predict, predicted, todense
    from ..external.tosica.train import (
        MyDataSet,
        create_pathway_mask,
        evaluate,
        get_gmt,
        read_gmt,
        set_seed,
        splitDataSet,
        train_one_epoch,
    )

    return {
        "fit_model": fit_model,
        "create_model": create_model,
        "set_seed": set_seed,
        "splitDataSet": splitDataSet,
        "get_gmt": get_gmt,
        "read_gmt": read_gmt,
        "create_pathway_mask": create_pathway_mask,
        "train_one_epoch": train_one_epoch,
        "evaluate": evaluate,
        "MyDataSet": MyDataSet,
        "predict": predict,
        "predicted": predicted,
        "todense": todense,
    }



@register_function(
    aliases=['TOSICA注释器', 'pyTOSICA', 'transformer cell annotation'],
    category="single",
    description="TOSICA wrapper for pathway-informed transformer-based cell-type annotation and transfer learning.",
    prerequisites={'optional_functions': ['utils.download_tosica_gmt', 'pp.preprocess']},
    requires={'obs': ['cell type labels (training mode)'], 'var': ['gene symbols']},
    produces={'obs': ['predicted cell types'], 'uns': ['tosica model outputs']},
    auto_fix='none',
    examples=['tosica_obj = ov.single.pyTOSICA(adata=ref_adata, project_path="./tosica")', 'tosica_obj.train()'],
    related=['single.pySCSA', 'utils.download_tosica_gmt']
)
class pyTOSICA(object):
    """
    TOSICA wrapper for pathway-informed transformer-based cell-type annotation.
    
    Parameters
    ----------
    adata:anndata.AnnData
        Training/reference AnnData with labels.
    project_path:str
        Output directory for TOSICA checkpoints and logs.
    gmt_path:str|None, optional, default=None
        Pathway GMT file path. If ``None``, default gene-set resources are used.
    label_name:str, optional, default='Celltype'
        Label column in ``adata.obs``.
    mask_ratio:float, optional, default=0.015
        Ratio of masked genes/tokens used for training regularization.
    max_g:int, optional, default=300
        Maximum number of genes used per pathway/tokenization unit.
    max_gs:int, optional, default=300
        Maximum number of gene sets used in the model.
    n_unannotated:int, optional, default=1
        Number of unlabeled classes reserved during training.
    embed_dim:int, optional, default=48
        Transformer embedding dimension.
    depth:int, optional, default=1
        Number of transformer encoder layers.
    num_heads:int, optional, default=4
        Number of attention heads.
    batch_size:int, optional, default=8
        Mini-batch size used during training/inference.
    device:str, optional, default='cuda:0'
        Device used for model training/inference.
    
    Returns
    -------
    None
        Initializes TOSICA model configuration and training resources.
    
    Examples
    --------
    >>> tosica_obj = ov.single.pyTOSICA(adata=ref_adata, project_path="./tosica")
    """

    def __init__(self,adata:anndata.AnnData,project_path:str,gmt_path=None,
                 label_name:str='Celltype',mask_ratio:float=0.015,
                 max_g:int=300,max_gs:int=300,n_unannotated:int= 1,
                 embed_dim:int=48,depth:int=1,num_heads:int=4,batch_size:int=8,
                 device:str='cuda:0',auto_download:bool=True,
                 ) -> None:
        r"""Initialize a pyTOSICA object for cell type classification.

        Parameters
        ----------
        adata:anndata.AnnData
            Training/reference AnnData for TOSICA.
        project_path:str
            Directory used to save masks, labels, checkpoints, and logs.
        gmt_path:str or None
            Pathway GMT identifier/path. If ``None``, full-connection mask is used.
        label_name:str
            Label column in ``adata.obs``.
        mask_ratio:float
            Random mask ratio used when pathway mask is unavailable.
        max_g:int
            Maximum number of genes per pathway.
        max_gs:int
            Maximum number of pathway tokens used by the model.
        n_unannotated:int
            Number of extra unannotated tokens appended to pathway mask.
        embed_dim:int
            Transformer embedding dimension.
        depth:int
            Number of transformer layers.
        num_heads:int
            Number of attention heads.
        batch_size:int
            Training batch size.
        device:str
            Preferred device string (for example ``'cuda:0'``).

        
        """

        tosica_backend = _get_tosica_backend()

        self.adata=adata 
        self.gmt_path=gmt_path
        self.project_path=project_path
        self.label_name=label_name
        self.n_unannotated=n_unannotated
        self.embed_dim=embed_dim

        GLOBAL_SEED = 1
        tosica_backend["set_seed"](GLOBAL_SEED)
        from torch.utils.tensorboard import SummaryWriter
        #device = 'cuda:0'
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.device = device
        print(device)
        today = time.strftime('%Y%m%d',time.localtime(time.time()))
        #train_weights = os.getcwd()+"/weights%s"%today
        #project = project or gmt_path.replace('.gmt','')+'_%s'%today
        #project_path = os.getcwd()+'/%s'%project
        if os.path.exists(project_path) is False:
            os.makedirs(project_path)
        self.tb_writer = SummaryWriter()
        exp_train, label_train, exp_valid, label_valid, inverse,genes = tosica_backend["splitDataSet"](adata,label_name)
        if gmt_path is None:
            mask = np.random.binomial(1,mask_ratio,size=(len(genes), max_gs))
            pathway = list()
            for i in range(max_gs):
                x = 'node %d' % i
                pathway.append(x)
            print('Full connection!')
        else:
            if '.gmt' in gmt_path:
                gmt_path = gmt_path
            else:
                resolved = tosica_backend["get_gmt"](gmt_path)
                if resolved == "Error":
                    if auto_download:
                        print("   - TOSICA GMT not found locally; auto-downloading via "
                              "ov.utils.download_tosica_gmt()...")
                        from ..utils._data import download_tosica_gmt
                        download_tosica_gmt()
                        resolved = tosica_backend["get_gmt"](gmt_path)
                        if resolved == "Error":
                            raise FileNotFoundError(
                                f"TOSICA GMT '{gmt_path}' still missing after "
                                "auto-download. Pass an explicit .gmt path."
                            )
                    else:
                        raise FileNotFoundError(
                            f"TOSICA GMT '{gmt_path}' not found. Call "
                            "ov.utils.download_tosica_gmt() first or pass an "
                            "explicit .gmt path."
                        )
                gmt_path = resolved
            
            reactome_dict = tosica_backend["read_gmt"](gmt_path, min_g=0, max_g=max_g)
            mask,pathway = tosica_backend["create_pathway_mask"](feature_list=genes,
                                            dict_pathway=reactome_dict,
                                            add_missing=n_unannotated,
                                            fully_connected=True)
            pathway = pathway[np.sum(mask,axis=0)>4]
            mask = mask[:,np.sum(mask,axis=0)>4]
            #print(mask.shape)
            pathway = pathway[sorted(np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):])]
            mask = mask[:,sorted(np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):])]
            #print(mask.shape)
            print('Mask loaded!')
        np.save(project_path+'/mask.npy',mask)
        pd.DataFrame(pathway).to_csv(project_path+'/pathway.csv') 
        pd.DataFrame(inverse,columns=[label_name]).to_csv(project_path+'/label_dictionary.csv', quoting=None)
        self.pathway=pathway 
        self.mask=mask

        num_classes = np.int64(torch.max(label_train)+1)

        train_dataset = tosica_backend["MyDataSet"](exp_train, label_train)
        valid_dataset = tosica_backend["MyDataSet"](exp_valid, label_valid)
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                pin_memory=True,drop_last=True)
        self.valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                pin_memory=True,drop_last=True)

        self.model = tosica_backend["create_model"](num_classes=num_classes, 
                                  num_genes=len(exp_train[0]),  
                                  mask = mask,embed_dim=embed_dim,
                                  depth=depth,num_heads=num_heads,
                                  has_logits=False).to(device) 
        pass

    def train(self,pre_weights:str='',lr:float=0.001, epochs:int= 10, lrf:float=0.01):
        r"""Train the TOSICA model for cell type classification.

        Parameters
        ----------
        pre_weights:str
            Optional path to pretrained weights.
        lr:float
            Initial learning rate.
        epochs:int
            Number of training epochs.
        lrf:float
            Final learning-rate factor used in cosine schedule.

        Returns
        -------
        torch.nn.Module
            Trained TOSICA model.
        
        """
        tosica_backend = _get_tosica_backend()
        if pre_weights != "":
            assert os.path.exists(pre_weights), "pre_weights file: '{}' not exist.".format(pre_weights)
            preweights_dict = torch.load(pre_weights, map_location=self.device)
            print(self.model.load_state_dict(preweights_dict, strict=False))
        #for name, param in model.named_parameters():
        #    if param.requires_grad:
        #        print(name) 
        print('Model builded!')
        pg = [p for p in self.model.parameters() if p.requires_grad]  
        optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5) 
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        for epoch in range(epochs):
            train_loss, train_acc = tosica_backend["train_one_epoch"](model=self.model,
                                                    optimizer=optimizer,
                                                    data_loader=self.train_loader,
                                                    device=self.device,
                                                    epoch=epoch)
            scheduler.step() 
            val_loss, val_acc = tosica_backend["evaluate"](model=self.model,
                                        data_loader=self.valid_loader,
                                        device=self.device,
                                        epoch=epoch)
            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            self.tb_writer.add_scalar(tags[0], train_loss, epoch)
            self.tb_writer.add_scalar(tags[1], train_acc, epoch)
            self.tb_writer.add_scalar(tags[2], val_loss, epoch)
            self.tb_writer.add_scalar(tags[3], val_acc, epoch)
            self.tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
            torch.save(self.model.state_dict(), self.project_path+"/model-{}.pth".format(epoch))
        print('Training finished!')
        add_reference(self.adata,'TOSICA','cell type classification with TOSICA')
        return self.model
    
    def save(self,save_path=None):
        r"""Save the trained TOSICA model.

        Parameters
        ----------
        save_path:str or None
            Checkpoint output path. If ``None``, defaults to
            ``project_path/model-best.pth``.
        
        """
        if save_path==None:
            save_path = self.project_path+'/model-best.pth'
        torch.save(self.model.state_dict(), save_path)
        print('Model saved!')

    def load(self,load_path=None):
        r"""Load a pre-trained TOSICA model.

        Parameters
        ----------
        load_path:str or None
            Checkpoint path to load. If ``None``, defaults to
            ``project_path/model-best.pth``.

        """
        if load_path==None:
            load_path = self.project_path+'/model-best.pth'
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        print('Model loaded!')
        
        
    def predicted(self,pre_adata:anndata.AnnData,laten:bool=False,n_step:int=10000,cutoff:float=0.1,
        batch_size:int=50,):
        r"""Predict cell types for new single-cell data.

        Parameters
        ----------
        pre_adata:anndata.AnnData
            Query AnnData used for inference.
        laten:bool
            Whether to export latent embedding rather than pathway attention.
        n_step:int
            Number of cells processed per outer step.
        cutoff:float
            Prediction confidence cutoff.
        batch_size:int
            Inference batch size.

        Returns
        -------
        anndata.AnnData
            Query AnnData augmented with predicted labels and model outputs.
        
        """
        tosica_backend = _get_tosica_backend()

        mask_path = os.getcwd()+'/%s'%self.project_path+'/mask.npy'
        dictionary = pd.read_table(self.project_path+'/label_dictionary.csv', sep=',',header=0,index_col=0)
        self.pathway=pd.read_csv(self.project_path+'/pathway.csv', index_col=0)
        n_c = len(dictionary)
        dic = {}
        for i in range(len(dictionary)):
            dic[i] = dictionary[self.label_name][i]
        self.model.eval()
        parm={}
        for name,parameters in self.model.named_parameters():
            #print(name,':',parameters.size())
            parm[name]=parameters.detach().cpu().numpy()
        gene2token = parm['feature_embed.fe.weight']
        gene2token = gene2token.reshape((int(gene2token.shape[0]/self.embed_dim),self.embed_dim,pre_adata.shape[1]))
        gene2token = abs(gene2token)
        gene2token = np.max(gene2token,axis=1)
        gene2token = pd.DataFrame(gene2token)
        gene2token.columns=pre_adata.var_names
        gene2token.index = self.pathway['0']
        gene2token.to_csv(self.project_path+'/gene2token_weights.csv')
        latent = torch.empty([0,self.embed_dim]).cpu()
        att = torch.empty([0,(len(self.pathway))]).cpu()
        predict_class = np.empty(shape=0)
        pre_class = np.empty(shape=0)      
        latent = torch.squeeze(latent).cpu().numpy()
        l_p = np.c_[latent, predict_class,pre_class]
        att = np.c_[att, predict_class,pre_class]
        all_line = pre_adata.shape[0]
        n_line = 0
        adata_list = []
        while (n_line) <= all_line:
            if (all_line-n_line)%batch_size != 1:
                expdata = pd.DataFrame(tosica_backend["todense"](pre_adata[n_line:n_line+min(n_step,(all_line-n_line))]),
                                       index=np.array(pre_adata[n_line:n_line+min(n_step,(all_line-n_line))].obs_names).tolist(), 
                                       columns=np.array(pre_adata.var_names).tolist())
                print(n_line)
                n_line = n_line+n_step
            else:
                expdata = pd.DataFrame(tosica_backend["todense"](pre_adata[n_line:n_line+min(n_step,(all_line-n_line-2))]),
                                       index=np.array(pre_adata[n_line:n_line+min(n_step,(all_line-n_line-2))].obs_names).tolist(), 
                                       columns=np.array(pre_adata.var_names).tolist())
                n_line = (all_line-n_line-2)
                print(n_line)
            expdata = np.array(expdata)
            expdata = torch.from_numpy(expdata.astype(np.float32))
            data_loader = torch.utils.data.DataLoader(expdata,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    pin_memory=True)
            with torch.no_grad():
                # predict class
                for step, data in enumerate(data_loader):
                    #print(step)
                    exp = data
                    lat, pre, weights = self.model(exp.to(self.device))
                    pre = torch.squeeze(pre).cpu()
                    pre = F.softmax(pre,1)
                    predict_class = np.empty(shape=0)
                    pre_class = np.empty(shape=0) 
                    for i in range(len(pre)):
                        if torch.max(pre, dim=1)[0][i] >= cutoff: 
                            predict_class = np.r_[predict_class,torch.max(pre, dim=1)[1][i].numpy()]
                        else:
                            predict_class = np.r_[predict_class,n_c]
                        pre_class = np.r_[pre_class,torch.max(pre, dim=1)[0][i]]     
                    l_p = torch.squeeze(lat).cpu().numpy()
                    att = torch.squeeze(weights).cpu().numpy()
                    meta = np.c_[predict_class,pre_class]
                    meta = pd.DataFrame(meta)
                    meta.columns = ['Prediction','Probability']
                    meta.index = meta.index.astype('str')
                    if laten:
                        l_p = l_p.astype('float32')
                        new = ad.AnnData(l_p, obs=meta)
                    else:
                        att = att[:,0:(len(self.pathway)-self.n_unannotated)]
                        att = att.astype('float32')
                        varinfo = pd.DataFrame(self.pathway.iloc[0:len(self.pathway)-self.n_unannotated,0].values,
                                               index=self.pathway.iloc[0:len(self.pathway)-self.n_unannotated,0],columns=['pathway_index'])
                        new = ad.AnnData(att, obs=meta, var = varinfo)
                    adata_list.append(new)
        print(all_line)
        new = ad.concat(adata_list)
        new.obs.index = pre_adata.obs.index
        new.obs['Prediction'] = new.obs['Prediction'].map(dic)
        new.obs[pre_adata.obs.columns] = pre_adata.obs[pre_adata.obs.columns].values    
        add_reference(self.adata,'TOSICA','cell type classification with TOSICA')

        return(new)


def train(adata, gmt_path, project=None,pre_weights='', 
          label_name='Celltype',max_g=300,max_gs=300,
          mask_ratio =0.015, n_unannotated = 1,
          batch_size=8, embed_dim=48,depth=2,
          num_heads=4,lr=0.001, epochs= 10, lrf=0.01):
    r"""
    Fit the model with reference data
    
    Parameters
    ----------
        adatas: Single-cell datasets
        gmt_path: The name (human_gobp; human_immune; human_reactome; human_tf; mouse_gobp; mouse_reactome and mouse_tf) or path of mask to be used.
        project: The name of project. Default: gmt_path_today.
        pre_weights: The path to the pre-trained weights. If pre_weights = '', the model will be trained from scratch.
        label_name: The column name of the label you want to prediect. Should in adata.obs.columns.
        max_g: The max of gene number belong to one pathway.
        max_gs: The max of pathway/token number.
        mask_ratio: The ratio of the connection reserved when there is no available mask.
        n_unannotated: The number of fully connected tokens to be added.
        batch_size: The number of cells for training in one epoch.
        embed_dim: The dimension of pathway/token embedding.
        depth: The number of multi-head self-attention layer.
        num_heads: The number of head in one self-attention layer.
        lr: Learning rate.
        epochs: The number of epoch will be trained.
        lrf: The hyper-parameter of Cosine Annealing.
    
    Returns
    -------
        ./mask.npy: Mask matrix
        ./pathway.csv: Gene set list
        ./label_dictionary.csv: Label list
        ./weights20220603/: Weights

    """
    tosica_backend = _get_tosica_backend()
    tosica_backend["fit_model"](adata, gmt_path, project=project,pre_weights=pre_weights, label_name=label_name,
              max_g=max_g,max_gs=max_gs,mask_ratio=mask_ratio, n_unannotated = n_unannotated,batch_size=batch_size, 
              embed_dim=embed_dim,depth=depth,num_heads=num_heads,lr=lr, epochs= epochs, lrf=lrf)


def pre(adata,model_weight_path,project,laten=False,save_att = 'X_att', 
        save_lantent = 'X_lat',n_step=10000,cutoff=0.1,
        n_unannotated = 1,batch_size=50,embed_dim=48,depth=2,num_heads=4):
    r"""
    Prediect query data with the model and pre-trained weights.
    
    Parameters
    ----------
        adatas: Query single-cell datasets.
        model_weight_path: The path to the pre-trained weights.
        mask_path: The path to the mask matrix.
        project: The name of project.
        laten: Get laten output.
        save_att: The name of the attention matrix to be added in the adata.obsm.
        save_lantent: The name of the laten matrix to be added in the adata.obsm.
        max_gs: The max of pathway/token number.
        n_step: The number of cells load into memory at the same time.
        cutoff: Unknown cutoff.
        n_unannotated: The number of fully connected tokens to be added. Should be the same as train.
        batch_size: The number of cells for training in one epoch.
        embed_dim: The dimension of pathway/token embedding. Should be the same as train.
        depth: The number of multi-head self-attention layer. Should be the same as train.
        num_heads: The number of head in one self-attention layer. Should be the same as train.
    
    Returns
    -------
        adata: adata.X : Attention matrix
        adata.obs['Prediction'] : Predicted labels
        adata.obs['Probability'] : Probability of the prediction
        adata.var['pathway_index'] : Gene set of each colume
    """
    
    mask_path = os.getcwd()+'/%s'%project+'/mask.npy'
    tosica_backend = _get_tosica_backend()
    adata = tosica_backend["predict"](adata,model_weight_path,project=project,mask_path = mask_path,laten=laten,
             save_att = save_att, save_lantent = save_lantent,n_step=n_step,cutoff=cutoff,n_unannotated = n_unannotated,batch_size=batch_size,embed_dim=embed_dim,depth=depth,num_heads=num_heads)
    return(adata)

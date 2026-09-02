import numpy as np
import torch
import os
import timeit
import copy
import argparse
import anndata
from anndata import AnnData
import scanpy as sc
from sklearn.neighbors import kneighbors_graph
import gc
from .fast_generator import *
from typing import Union, Tuple
import pickle

import warnings
import logging
warnings.filterwarnings("ignore")
import warnings
import logging
import pytorch_lightning as pl
from ...._anndata_compat import as_dtype as _as_dtype
#logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)


#from .vaegan import *

def setdata(name:str,sid:str,device:str='cuda:0',k:int=15,diagw:float=1.0)  -> Tuple[anndata.AnnData, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Wrapper function for any preparations that need to be done for a anndata.AnnData object before sending to the model. 

    Parameters
    ----------
    name
        The porject name.
    sid
        Sample ID of the sample to be prepared
    device
        CPU or GPU for model training
    k
        Number of neighbors to consider in the cell graph
    diagw
        The weight of the original cell when agregating the information

    Returns
    -------
    adata
        The augmented anndata object. 
    adj
        The adjacency matrix of the cell neighbor graph.
    variances
        Variances of the features
    pseudobulk
        pseudobulk data of the sample
    geneset_len
        Length of the gene set score features
    """
    
    adata = anndata.read_h5ad(name + '/sample_sc/' + sid + '.h5ad') 
    
    # load geneset
    if 'geneset_scores' in os.listdir(name):    
        sample_geneset = np.load(name + '/geneset_scores/'+sid+'.npy')
        setmask = np.load(name + '/hvset.npy')
        sample_geneset = sample_geneset[:,setmask]
        sample_geneset = sample_geneset.astype('float32')
        geneset_len = sample_geneset.shape[1]

        features = np.concatenate([adata.X,sample_geneset],1)
        bdata = anndata.AnnData(_as_dtype(features, 'float32'))
        bdata.obs = adata.obs
        bdata.obsm = adata.obsm
        bdata.uns = adata.uns
        adata = bdata.copy()
    else:
        geneset_len = 0
    
    # adj for cell graph
    adj = adata.obsm['adj']
    adj = torch.from_numpy(adj.astype('float32'))
    
    # variances
    variances = torch.tensor(adata.uns['feature_var'])
    variances = variances.to(device)
    
    #pseudobulk
    pseudobulk = np.array(adata.X.mean(axis=0)).reshape((-1))
    fastgenerator.setup_anndata(adata)

    return adata,adj,variances,pseudobulk,geneset_len




def fastrecon(name:str, sid:str, device:str='cuda:0',k:int=15,diagw:float=1.0,vaesteps:int=100,gansteps:int=100,lr:float = 1e-3,save:bool=True,path:str=None) -> fastgenerator:
    """
    Accelerated version of pretrain 1 reconstruction.
    
    Parameters
    ----------
    name
        The porject name.
    sid
        Sample ID of the sample to be prepared
    device
        CPU or GPU for model training
    k
        Number of neighbors to consider in the cell graph
    diagw
        The weight of the original cell when agregating the information
    vaestep:
        Steps for training the generator
    ganstep:
        Steps for joint training the generator and the discriminator.
    lr
        Learning rate
    save
        Saving the model or not
    path
        Path for saving the model
    
    Returns
    -------
    model
        The trained model.
    """
    
    #set data
    adata,adj,variances,bulk,geneset_len = setdata(name,sid,device,k,diagw)
    #print(0)
    #print(variances.shape)
    #print(adata)
    
    # train
    model = fastgenerator(variances,None,geneset_len,adata,n_hidden=256,n_latent=32,dropout_rate=0)

    #model.train(max_epochs=vaesteps, plan_kwargs={'lr':lr,'lr2':0,'kappa':4.0},use_gpu=device)
    #model.train(max_epochs=gansteps*3, plan_kwargs={'lr':lr,'lr2':lr,'kappa':4.0},use_gpu=device)
    steps=3
    model.train(max_epochs=vaesteps, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4},use_gpu=device)
    print('start pertrain 1 adversarial training:')
    for i in range(gansteps):
        print(i,end=', ')
        model.train(max_epochs=steps, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4},gan = True,use_gpu=device)
        model.train(max_epochs=steps, plan_kwargs={'lr':1e-10,'lr2':lr,'kappa':4},gan = True,use_gpu=device)
       
        
    # save model
    if save == True:
        if path == None:
            if (os.path.isdir(name + '/models')) == False:
                os.system('mkdir '+ name + '/models')
            path = name + '/models/fast_reconst1_'+sid
        torch.save(model.module.state_dict(), path)
    
    
    with open(name+'/history/pretrain1_' + sid + '.pkl', 'wb') as pickle_file:
                        pickle.dump(model.history, pickle_file)
            
    return model


# reconst stage 2
def reconst_pretrain2(name:str, sid:str ,premodel:Union[str,fastgenerator],device='cuda:0',k=15,diagw=1.0,vaesteps=50,gansteps=50,lr=1e-4,save=True,path=None)->fastgenerator:
    """
    Accelerated version of pretrain 2 reconstruction.
    
    Parameters
    ----------
    name
        The porject name.
    sid
        Sample ID of the sample to be prepared
    premodel
        Pretrained model or path to pretrained model
    device
        CPU or GPU for model training
    k
        Number of neighbors to consider in the cell graph
    diagw
        The weight of the original cell when agregating the information
    vaestep:
        Steps for training the generator
    ganstep:
        Steps for joint training the generator and the discriminator.
    lr
        Learning rate
    save
        Saving the model or not
    path
        Path for saving the model
    
    Returns
    -------
    model
        The trained model.
    """
    
    adata,adj,variances,bulk,geneset_len = setdata(name,sid,device,k,diagw)
    
    #(4) bulk
    bulk = (np.array(adata.X)).mean(axis=0)
    bulk = bulk.reshape((-1))
    bulk = torch.tensor(bulk).to(device)
    
    #(5) reconstruct pretrain
    fastgenerator.setup_anndata(adata)
    model = fastgenerator(variances = variances,bulk=bulk,geneset_len = geneset_len,adata=adata,\
                n_hidden=256,n_latent=32,dropout_rate=0,countbulkweight=1,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,\
                power=2,corrbulkweight=0)
    
    if type(premodel) == type(None):
        pass
    else:
        model.module.load_state_dict(premodel.module.state_dict())
    

    #model.train(max_epochs=vaesteps, plan_kwargs={'lr':lr,'lr2':0,'kappa':4.0},use_gpu=device)
    #model.train(max_epochs=gansteps*3, plan_kwargs={'lr':lr,'lr2':lr,'kappa':4.0},use_gpu=device)
    steps=3
    model.train(max_epochs=vaesteps, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4},use_gpu=device)
    print('start pertrain 2 adversarial training:')
    for i in range(gansteps):
        print(i,end=', ')
        model.train(max_epochs=steps, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4},gan = True,use_gpu=device)
        model.train(max_epochs=steps, plan_kwargs={'lr':1e-10,'lr2':lr,'kappa':4},gan = True,use_gpu=device)
    
    
    
    if save == True:
        if path == None:
            path = name + '/models/fastreconst2_' + sid
        torch.save(model.module.state_dict(), path)
    
    with open(name+'/history/pretrain2_' + sid + '.pkl', 'wb') as pickle_file:
                        pickle.dump(model.history, pickle_file)
    
    return model





def unisemi0(name,adata,adj,variances,geneset_len,bulk,batch_size,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0,lr = 2e-4,epochs=150):
    model0 = fastgenerator(adata=adata,variances=variances,geneset_len=geneset_len,\
                      bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight =1*8,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                      power=2,upperbound=99999)
    if type(premodel)==type('string'):
        model0.module.load_state_dict(torch.load(premodel))
    else:
        model0.module.load_state_dict(premodel.module.state_dict())
    model0.train(max_epochs=epochs, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model0.module.state_dict(), name+'/tmp/model0')
    return model0.history


def unisemi1(name,adata,adj,variances,geneset_len,bulk,batch_size,upperbound,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0,lr = 2e-4,epochs=150):
    model1 = fastgenerator(adata=adata,variances=variances,geneset_len=geneset_len,\
                      bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 4*8,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbound)
    model1.module.load_state_dict(torch.load(name+'/tmp/model0'))
    model1.train(max_epochs=epochs, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model1.module.state_dict(), name+'/tmp/model1')
    return model1.history

def unisemi2(name,adata,adj,variances,geneset_len,bulk,batch_size,upperbound,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0,lr = 2e-4,epochs=150):
    model2 = fastgenerator(adata=adata,variances=variances,geneset_len=geneset_len,\
                      bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 16*8,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbound)
    model2.module.load_state_dict(torch.load(name+'/tmp/model1'))
    model2.train(max_epochs=epochs, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model2.module.state_dict(), name+'/tmp/model2')
    return model2.history

def unisemi3(name,adata,adj,variances,geneset_len,bulk,batch_size,upperbound,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0,lr = 2e-4,epochs=150):
    model3 = fastgenerator(adata=adata,variances=variances,geneset_len=geneset_len,\
                      bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 64*8,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbound)
    model3.module.load_state_dict(torch.load(name+'/tmp/model2'))
    model3.train(max_epochs=epochs, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model3.module.state_dict(), name+'/tmp/model3')
    return model3.history

def unisemi4(name,adata,adj,variances,geneset_len,bulk,batch_size,upperbound,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0,lr = 2e-4,epochs=150):
    model4 = fastgenerator(adata=adata,variances=variances,geneset_len=geneset_len,\
                      bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 128*8,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbound)
    model4.module.load_state_dict(torch.load(name+'/tmp/model3'))
    model4.train(max_epochs=epochs, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model4.module.state_dict(), name+'/tmp/model4')
    return model4.history

def unisemi5(adata,adj,variances,geneset_len,bulk,batch_size,upperbound,reprepid,tgtpid,premodel,device='cuda:5',k=15,diagw=1.0,lr = 2e-4,epochs=150):
    model = fastgenerator(adata=adata,variances=variances,geneset_len=geneset_len,\
                      bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 512*8,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbound)
    model.module.load_state_dict(torch.load(name+'/tmp/model4'))
    model.train(max_epochs=epochs, plan_kwargs={'lr':lr,'lr2':1e-10,'kappa':4040*1e-10},use_gpu=device,batch_size=batch_size)
    torch.save(model.module.state_dict(), name+'/tmp/model')
    return model.history

def fast_semi(name:str,reprepid:int,tgtpid:int,premodel:Union[fastgenerator,str],device:str='cuda:0',k:int=15,diagw:float=1.0,bulktype='pseudobulk',pseudocount=0.1,lr = 2e-4,epochs=150,ministages = 5) -> Tuple[ dict, np.array, fastgenerator]:
    """
    Accelerated version of single-cell inference for the target sample.
    
    Parameters
    ----------
    name
        The porject name.
    reprepid
        Sample ID (number) of the representative
    tgtpid
        Sample ID (number) of the target sample
    premodel
        Pretrained model or path to pretrained model
    device
        CPU or GPU for model training
    k
        Number of neighbors to consider in the cell graph
    diagw
        The weight of the original cell when agregating the information
    bulktype
        'real' or 'pseudobulk'
    pseudocount
    	Pseudocount value used when simulating pseudobulk using real bulk
    lr
        Learning rate
    epochs
        Epochs for each mini-stage in the inference process
        
    Returns
    -------
    histdic
        A dictionary containing the training history information
    xsemi
        Inferred single-cell data
    model
        Trained inference model
        
    
    """
    
    
    sids = []
    f = open(name + '/sids.txt','r')
    lines = f.readlines()
    for l in lines:
        sids.append(l.strip())
    f.close()
    
    adata,adj,variances,reprepseudobulk,geneset_len = setdata(name,sids[reprepid],device=device,k=k,diagw=diagw)
    
    varainces = None
    
    maxexpr = adata.X.max()
    upperbounds = [maxexpr/2, maxexpr/4, maxexpr/8, maxexpr/(8*np.sqrt(2)),maxexpr/16, maxexpr/32,maxexpr/64]     
    
    genelen = len(np.load(name+'/hvgenes.npy',allow_pickle=True))
    
    #(5) tgt bulk
    if bulktype == 'real':
        tgtbulkdata = anndata.read_h5ad(name + '/processed_bulkdata.h5ad')
        tgtbulk = np.exp(tgtbulkdata.X[tgtpid]) - 1
        repbulk = np.exp(tgtbulkdata.X[reprepid]) - 1
        tgtrealbulk = np.array(tgtbulk).reshape((1,-1))  # target real bulk
        reprealbulk = np.array(repbulk).reshape((1,-1))  # representative real bulk
        
        pseudobulk = (np.array(adata.X)).mean(axis=0)
        pseudobulk = pseudobulk.reshape((1,-1))
        ratio = np.array((tgtrealbulk+pseudocount)/(reprealbulk+pseudocount))
        #ratio = np.array((tgtrealbulk+1)/(reprealbulk+1))
        ratio = np.concatenate([ratio,np.ones((1, pseudobulk.shape[1]-ratio.shape[1]))],axis=1)
        bulk = pseudobulk * ratio
        bulk = torch.tensor(bulk).to(device)
        
    elif (bulktype == 'pseudobulk') or (bulktype == 'pseudo'):
        bulkdata = anndata.read_h5ad(name + '/processed_bulkdata.h5ad')
        tgtbulk = np.exp(bulkdata.X[tgtpid]) - 1
        tgtbulk = np.array(tgtbulk).reshape((1,-1))
        bulk = adata.X.mean(axis=0)
        bulk = np.array(bulk).reshape((1,-1))
        bulk[:,:tgtbulk.shape[1]] = tgtbulk
        bulk = torch.tensor(bulk).to(device)
    
    else:
        print('Error. Please specify bulktype as "pseudobulk" or "real".')
        return
    
    batch_size=int(np.min([adata.X.shape[0],9000]))
    
    
    #(6) semiprofiling
    fastgenerator.setup_anndata(adata)

    
    hist = unisemi0(name,adata,adj,variances,geneset_len,bulk,batch_size,reprepid,tgtpid,premodel,device=device,k=k,diagw=1.0,lr=lr,epochs=epochs)
    histdic={}
    histdic['total0'] = hist['train_loss_epoch']
    histdic['bulk0'] = hist['kl_global_train']
    #del premodel
    gc.collect()
    torch.cuda.empty_cache() 
    #import time
    #time.sleep(10)
    
    
    hist = unisemi1(name,adata,adj,variances,geneset_len,bulk,batch_size,upperbounds[0],reprepid,tgtpid,premodel,device=device,k=k,diagw=1.0,lr=lr,epochs=epochs)

    histdic['total1'] = hist['train_loss_epoch']
    histdic['bulk1'] = hist['kl_global_train']
    #del model0
    gc.collect()
    torch.cuda.empty_cache() 
    hist = unisemi2(name,adata,adj,variances,geneset_len,bulk,batch_size,upperbounds[1],reprepid,tgtpid,premodel,device=device,k=k,diagw=1.0,lr=lr,epochs=epochs)

    histdic['total2'] = hist['train_loss_epoch']
    histdic['bulk2'] = hist['kl_global_train']
    #del model1
    gc.collect()

    #time.sleep(10)
    torch.cuda.empty_cache() 
    
    if ministages>3:
        hist = unisemi3(name,adata,adj,variances,geneset_len,bulk,batch_size,upperbounds[2],reprepid,tgtpid,premodel,device=device,k=k,diagw=1.0,lr=lr,epochs=epochs)

        histdic['total3'] = hist['train_loss_epoch']
        histdic['bulk3'] = hist['kl_global_train']
    
    
    #del model2
    gc.collect()
    torch.cuda.empty_cache() 
    #time.sleep(10)
    if ministages >4:
        hist = unisemi4(name,adata,adj,variances,geneset_len,bulk,batch_size,upperbounds[3],reprepid,tgtpid,premodel,device=device,k=k,diagw=1.0,lr=lr,epochs=epochs)

        histdic['total4'] = hist['train_loss_epoch']
        histdic['bulk4'] = hist['kl_global_train']
    
    #del model3
    gc.collect()
    torch.cuda.empty_cache() 
    #time.sleep(10)
    #hist = unisemi5(adata,adj,variances,geneset_len,bulk,batch_size,upperbounds[4],reprepid,tgtpid,premodel,device=device,k=15,diagw=1.0)
    #histdic['total'] = hist['train_loss_epoch']
    #histdic['bulk'] = hist['kl_global_train']


    model = fastgenerator(adata=adata,variances=variances,geneset_len=geneset_len,\
                     bulk=bulk,n_hidden=256,n_latent=32,\
                     dropout_rate=0,countbulkweight = 512,logbulkweight=0,absbulkweight=0,abslogbulkweight=0,corrbulkweight=0,\
                     power=2,upperbound=upperbounds[3])
    model.module.load_state_dict(torch.load(name+'/tmp/model'+str(ministages-1)))

    # inference
    xsemi = []
    scdl = model._make_data_loader(
            adata=adata,batch_size=batch_size
    )
    for tensors in scdl:
        samples = model.module.sample(tensors, n_samples=1)
        xsemi.append(samples)
    
    # save inferred data
    xsemi = np.array(torch.cat(xsemi))[:,:genelen]
    torch.save(model.module.state_dict(), name+'/models/semi_'+sids[reprepid]+"_to_"+sids[tgtpid])
    xsemi = xsemi*(xsemi>10)
    np.save(name + '/inferreddata/'+ sids[reprepid]+'_to_'+sids[tgtpid],xsemi)
    
    # save training history
    with open(name+'/history/inference_' + sids[reprepid] + '_to_' + sids[tgtpid] + '.pkl', 'wb') as pickle_file:
                    pickle.dump(histdic, pickle_file)
    
    
    gc.collect()
    torch.cuda.empty_cache() 
    
    
    return histdic,xsemi,model


def tgtinfer(name:str, representative:Union[str,int],target:Union[str,int],bulktype:str='pseudobulk',
    lambdad:float = 4.0,
    pretrain1batch:int = 128,
    pretrain1lr:float = 1e-3,
    pretrain1vae:int = 100,
    pretrain1gan:int = 100,
    lambdabulkr:float = 1,
    pretrain2lr:float = 1e-4,
    pretrain2vae:int = 50,
    pretrain2gan:int = 50,
    inferepochs:int = 150,
    lambdabulkt:float = 8.0,
    inferlr:float = 2e-4,
    pseudocount:float = 0.1,
    k:int = 15,
    device:str = 'cuda:0') -> None:
    """
    Computationally infer the single-cell data of a single non-representative target sample based on a representatives' single-cell data and bulk data of both samples. 
    
    Parameters
    ----------
    name
        The project name.
    representative
        The representative. Either indicated using sample ID (str) or the i-th (int) sample.
    target
        The target sample. Either indicated using sample ID (str) or the i-th (int) sample.
    bulktype
        Pseudobulk or real bulk data
    lambdad
        Scaling factor for the discriminator loss.
    pretrain1batch 
        The mini-batch size during the first pretrain stage.
    pretrain1lr 
        The learning rate used in the first pretrain stage.
    pretrain1vae 
        The number of epochs for training the VAE during the first pretrain stage.
    pretrain1gan 
        The number of iterations for training GAN during the first pretrain stage.
    lambdabulkr 
        Scaling factor for represenatative bulk loss for pretrain 2.
    pretrain2lr 
        Pretrain 2 learning rate.
    pretrain2vae 
        The number of epochs for training the VAE during the second pretrain stage.
    pretrain2gan 
        The number of iterations for training the GAN during the second pretrain stage.
    inferepochs 
        The number of epochs used for each mini-stage during inference.
    lambdabulkt 
        Scaling factor for the initial target bulk loss.
    inferlr 
        Infer stage learning rate.
    k
        The number of nearest neighbors used in cell graph.
    device 
        Which device to use, e.g. 'cpu', 'cuda:0'.
    pseudocount
        Pseudocount used when converting real bulk to pseudobulk space
        
    Returns
    -------
        None

    Example
    -------
    >>> name = 'project_name'
    >>> scSemiProfiler.tgtinfer(name = name, representatives = 6, target = 7, bulktype = 'real')

    """
    
    
    
    if (os.path.isdir(name + '/inferreddata')) == False:
        os.system('mkdir ' + name + '/inferreddata')
    if (os.path.isdir(name + '/models')) == False:
        os.system('mkdir ' + name + '/models')
    if (os.path.isdir(name + '/tmp')) == False:
        os.system('mkdir ' + name + '/tmp')
    if (os.path.isdir(name + '/history')) == False:
        os.system('mkdir '+ name + '/history')
    
    device = device
    
    

    diagw = 1.0

    sids = []
    f = open(name + '/sids.txt','r')
    lines = f.readlines()
    for l in lines:
        sids.append(l.strip())
    f.close()


    if type(representative) == type(123):
        rp = sids[representative]
    else:
        rp = representative
    
    if type(target) == type(123):
        tgt = sids[target]
    else:
        tgt = target
    tgtpid = sids.index(tgt)
    reprepid = sids.index(rp)
        
    print('pretrain 1: representative reconstruction')
            
    # if exists, load model
    modelfile = 'fast_reconst1_' + rp
    path = name + '/models/fast_reconst1_' + rp
    if modelfile in os.listdir(name + '/models'):
        print('load existing pretrain 1 reconstruction model for ' + rp)
        adata,adj,variances,bulk,geneset_len = setdata(name,rp,device,k,diagw)
        model = fastgenerator(variances,None,geneset_len,adata,n_hidden=256,n_latent=32,dropout_rate=0)
        model.module.load_state_dict(torch.load(path))
        repremodel = model
        #continue
    else:
        # otherwise, train model
        repremodel = fastrecon(name=name,sid=rp,device=device,k=15,diagw=1,vaesteps=int(pretrain1vae),gansteps=int(pretrain1gan),save=True,path=None)\
                    

        
    print('pretrain2: reconstruction with representative bulk loss')
    modelfile = 'fastreconst2_' + rp
    path = name + '/models/fastreconst2_' + rp 
    if modelfile in os.listdir(name + '/models'):
        print('load existing pretrain 2 model for ' + rp)
        adata,adj,variances,bulk,geneset_len = setdata(name,rp,device,k,diagw)
        model = fastgenerator(variances,None,geneset_len,adata,n_hidden=256,n_latent=32,dropout_rate=0)
        model.module.load_state_dict(torch.load(path))
        repremodels2 = model
    else:
        repremodels2 = (reconst_pretrain2(name,rp,repremodel,device,k=15,diagw=1.0,vaesteps=int(pretrain2vae),gansteps=int(pretrain2gan),save=True))

    

    fname = rp + '_to_' + tgt + '.npy'
    if fname in os.listdir(name+'/inferreddata/'):
        print('Inference for '+tgt+' has been finished previously. Skip.')

    premodel = repremodels2
    histdic,xsemi,infer_model  = fast_semi(name,reprepid,tgtpid,premodel,device=device,k=15,diagw=1.0, bulktype = bulktype,lr=inferlr,epochs=inferepochs,pseudocount=pseudocount)
    print('Finished target sample single-cell inference')
    return


def scinfer(name:str, representatives:str,cluster:str,bulktype:str='pseudobulk',
    lambdad:float = 4.0,
    pretrain1batch:int = 128,
    pretrain1lr:float = 1e-3,
    pretrain1vae:int = 100,
    pretrain1gan:int = 100,
    lambdabulkr:float = 1,
    pretrain2lr:float = 1e-4,
    pretrain2vae:int = 50,
    pretrain2gan:int = 50,
    inferepochs:int = 150,
    lambdabulkt:float = 8.0,
    inferlr:float = 2e-4,
    pseudocount:float = 0.1,
    ministages:int = 5,
    k:int = 15,
    device:str = 'cuda:0') -> None:
    """
    Computationally infer the single-cell data of all non-representative samples (target samples) based on the cohort's bulk data and the representatives' single-cell data
    
    Parameters
    ----------
    name
        The project name.
    representatives
        Path to a "txt" file containing the representative sample IDs (number)
    cluster
        Path to a "txt" file containing the cluster label information
    bulktype
        Pseudobulk or real bulk data
    lambdad
        Scaling factor for the discriminator loss.
    pretrain1batch 
        The mini-batch size during the first pretrain stage.
    pretrain1lr 
        The learning rate used in the first pretrain stage.
    pretrain1vae 
        The number of epochs for training the VAE during the first pretrain stage.
    pretrain1gan 
        The number of iterations for training GAN during the first pretrain stage.
    lambdabulkr 
        Scaling factor for represenatative bulk loss for pretrain 2.
    pretrain2lr 
        Pretrain 2 learning rate.
    pretrain2vae 
        The number of epochs for training the VAE during the second pretrain stage.
    pretrain2gan 
        The number of iterations for training the GAN during the second pretrain stage.
    inferepochs 
        The number of epochs used for each mini-stage during inference.
    lambdabulkt 
        Scaling factor for the initial target bulk loss.
    inferlr 
        Infer stage learning rate.
    ministages
        Number of ministages during inference
    k
        The number of nearest neighbors used in cell graph.
    device 
        Which device to use, e.g. 'cpu', 'cuda:0'.
    pseudocount:
        Pseudocount used when converting data from real bulk space to pseudobulk space
        
    Returns
    -------
        None

    Example
    -------
    >>> name = 'project_name'
    >>> representatives = name + '/status/init_representatives.txt'
    >>> cluster = name + '/status/init_cluster_labels.txt'
    >>> scSemiProfiler.scinfer(name = name, representatives = representatives, cluster = cluster, bulktype = 'pseudobulk')

    """
    
    
    if (os.path.isdir(name + '/inferreddata')) == False:
        os.system('mkdir ' + name + '/inferreddata')
    if (os.path.isdir(name + '/models')) == False:
        os.system('mkdir ' + name + '/models')
    if (os.path.isdir(name + '/tmp')) == False:
        os.system('mkdir ' + name + '/tmp')
    if (os.path.isdir(name + '/history')) == False:
        os.system('mkdir '+ name + '/history')
    
    device = device
    
    

    diagw = 1.0

    
    print('Start single-cell inference in cohort mode')

    sids = []
    f = open(name + '/sids.txt','r')
    lines = f.readlines()
    for l in lines:
        sids.append(l.strip())
    f.close()


    repres = []
    f=open(representatives,'r')
    lines = f.readlines()
    f.close()
    for l in lines:
        repres.append(int(l.strip()))

    cluster_labels = []
    f=open(cluster,'r')
    lines = f.readlines()
    f.close()
    for l in lines:
        cluster_labels.append(int(l.strip()))

    #timing
    pretrain1start = timeit.default_timer()

    print('pretrain 1: representative reconstruction')
    repremodels = []
    for rp in repres:
        sid = sids[rp]

        # if exists, load model
        modelfile = 'fast_reconst1_' + sid
        path = name + '/models/fast_reconst1_'+sid
        if modelfile in os.listdir(name + '/models'):
            print('load existing pretrain 1 reconstruction model for '+sid)
            adata,adj,variances,bulk,geneset_len = setdata(name,sid,device,k,diagw)
            model = fastgenerator(variances,None,geneset_len,adata,n_hidden=256,n_latent=32,dropout_rate=0)
            model.module.load_state_dict(torch.load(path))
            repremodels.append(model)
            #continue
        else:
            # otherwise, train model
            repremodels.append(\
                               fastrecon(name=name,sid=sid, \
                                         device=device,k=15,\
                                         diagw=1,vaesteps=int(pretrain1vae),\
                                         gansteps=int(pretrain1gan),save=True,path=None)\
                              )

    # timing
    pretrain1end = timeit.default_timer()
    f=open('pretrain1time.txt','w')
    f.write(str(pretrain1end-pretrain1start))
    f.close()

    #timing
    pretrain2start = timeit.default_timer()

    print('pretrain2: reconstruction with representative bulk loss')
    repremodels2=[]
    i=0
    for rp in repres:
        sid = sids[rp]
        # if exists, load model
        print('load existing model')
        modelfile = 'fastreconst2_' + sid
        path = name + '/models/fastreconst2_' + sid 
        if modelfile in os.listdir(name + '/models'):
            print('load existing pretrain 2 model for ' + sid)
            adata,adj,variances,bulk,geneset_len = setdata(name,sid,device,k,diagw)
            model = fastgenerator(variances,None,geneset_len,adata,n_hidden=256,n_latent=32,dropout_rate=0)
            model.module.load_state_dict(torch.load(path))
            repremodels2.append(model)
            #continue
        else: repremodels2.append(reconst_pretrain2(name,sid,repremodels[i],\
                                                    device,k=15,diagw=1.0,vaesteps=int(pretrain2vae), gansteps=int(pretrain2gan),save=True))
        i=i+1

    #timing
    #pretrain2end = timeit.default_timer()
    #f=open('pretrain2time.txt','w')
    #f.write(str(pretrain2end-pretrain2start))
    #f.close()


    #timing
    #f = open('infertime.txt','w')

    print('inference')
    for i in range(len(sids)):
        if i not in repres:
            #timing
            inferstart = timeit.default_timer()

            tgtpid = i
            reprepid = repres[cluster_labels[i]]

            fname = sids[reprepid]+'_to_'+sids[tgtpid]+'.npy'
            if fname in os.listdir(name+'/inferreddata/'):
                print('Inference for '+sids[i]+' has been finished previously. Skip.')
                continue

            premodel = repremodels2[cluster_labels[i]]
            histdic,xsemi,infer_model  = fast_semi(name,reprepid,tgtpid,premodel,device=device,k=15,diagw=1.0,bulktype=bulktype,lr=inferlr,epochs=inferepochs,pseudocount=pseudocount,ministages = ministages)


            #timing
            inferend = timeit.default_timer()
            #f.write(str(inferend-inferend)+'\n')
    #timing
    #f.close()
        

    
    print('Finished single-cell inference')
    return




def main():
    parser = argparse.ArgumentParser(description="scSemiProfiler scinfer")
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    required.add_argument('--name',required=True,help="Project name (same as previous steps).")
    
    required.add_argument('--representatives',required=True,help="Either a txt file including all the IDs of the representatives used in the current round of semi-profiling when running in cohort mode, or a single sample ID when running in single-sample mode.")
    
    optional.add_argument('--cluster',required=False,default='na', help="A txt file specifying the cluster membership. Required when running in cohort mode.")
    
    optional.add_argument('--targetid',required=False, default='na', help="Sample ID of the target sample when running in single-sample mode.")
    
    optional.add_argument('--bulktype',required=False, default='real', help="Specify 'pseudo' for pseudobulk or 'real' for real bulk data. (Default: real)")
    
    optional.add_argument('--lambdad',required=False, default='4.0', help="Scaling factor for the discriminator loss for training the VAE generator. (Default: 4.0)")
    
    optional.add_argument('--pretrain1batch',required=False, default='128', help="Sample Batch Size of the first pretrain stage. (Default: 128)")
    
    optional.add_argument('--pretrain1lr',required=False, default='1e-3', help="Learning rate of the first pretrain stage. (Default: 1e-3)")
    
    optional.add_argument('--pretrain1vae',required=False, default='100', help = "The number of epochs for training the VAE generator during the first pretrain stage. (Default: 100)")
    
    optional.add_argument('--pretrain1gan',required=False, default='100', help="The number of iterations for training the generator and discriminator jointly during the first pretrain stage. (Default: 100)")
    
    optional.add_argument('--lambdabulkr',required=False, default='1.0', help="Scaling factor for the representative bulk loss. (Default: 1.0)")
    
    optional.add_argument('--pretrain2lr',required=False, default='1e-4', help="The number of epochs for training the VAE generator during the second pretrain stage. (Default: 50)")
    
    optional.add_argument('--pretrain2vae',required=False, default='50', help="Sample ID of the target sample when running in single-sample mode.")
    
    optional.add_argument('--pretrain2gan',required=False, default='50', help="The number of iterations for training the generator and discriminator jointly during the second pretrain stage. (Default: 50)")
    
    optional.add_argument('--inferepochs',required=False, default='150', help="The number of epochs for training the generator in each mini-stage during the inference. (Default: 150)")
    
    optional.add_argument('--lambdabulkt',required=False, default='8.0', help="Scaling factor for the intial target bulk loss. (Default: 8.0)")
    
    optional.add_argument('--inferlr',required=False, default='2e-4', help="Learning rate during the inference stage. (Default: 2e-4)")
    
    
    
    
    args = parser.parse_args()
    
    name = args.name
    representatives = args.representatives
    cluster = args.cluster
    targetid = args.targetid
    bulktype = args.bulktype
    lambdad = float(args.lambdad)
    
    pretrain1batch = int(args.pretrain1batch)
    pretrain1lr = float(args.pretrain1lr)
    pretrain1vae = int(args.pretrain1vae)
    pretrain1gan = int(args.pretrain1gan)
    lambdabulkr = float(args.lambdabulkr)
    
    pretrain2lr = float(args.pretrain2lr)
    pretrain2vae = int(args.pretrain2vae)
    pretrain2gan = int(args.pretrain2gan)
    inferepochs = int(args.inferepochs)
    lambdabulkt = float(args.lambdabulkt)
    
    inferlr = float(args.inferlr)
    
    
    
    scinfer(name, representatives,cluster,targetid,bulktype,lambdad,pretrain1batch,pretrain1lr,pretrain1vae,pretrain1gan,lambdabulkr,pretrain2lr, pretrain2vae,pretrain2gan,inferepochs,lambdabulkt,inferlr)

if __name__=="__main__":
    main()

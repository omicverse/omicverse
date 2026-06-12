import pdb,sys,os
import timeit
import warnings
warnings.filterwarnings('ignore')
import argparse
import numpy as np
from sklearn.neural_network import MLPClassifier
import anndata
import scanpy as sc
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sn
from scipy import sparse
from typing import Union
from mpmath import *
mp.dps = 200
import scipy.stats as stats
import copy
from typing import Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from .inference import *
import random
import faiss
from sklearn.decomposition import PCA

def set_representatives(name:str, representatives_list:list, iteration:int = 0):
    """
    Specify the reference samples for single-cell deconvolution. 
    
    Parameters
    ----------
    name
        Project name. 
    batch 
        Representative selection batch size. 
    
    Returns
    -------
        None
    
    Example
    --------
    >>> scSemiProfiler.set_representatives(name = 'deconvolution_example', representatives_list = [6])

    
    """
    if iteration == 0:
        repfile = 'init_representatives.txt'
    else:
        repfile = 'eer_representatives_' + str(iteration) + '.txt'
    f = open(name + '/status/' + repfile,'w')
    for rep in representatives_list:
        f.write(str(rep) + '\n')
    f.close()
    
    return

def get_cohort_sc(name:str, path:str, cohort_name:str, representative_ids:list) -> None:
    """
    Get representatives' single-cell data from a given preprocessed cohort (COVID-19, colorectal cancer, iMGL) and store the data as cohort_name.h5ad under the project's directory
    
    Parameters
    ----------
    name 
        Project name
    path 
        Path to the cohort dataset h5ad file
    cohort_name 
        The output single-cell data file name
    representative_ids 
        The sample IDs of the selected representatives
        
    Returns
    -------
    None

    Example
    -------
    >>> covidpath = '../preprocessed_data/preprocessed_covid-19_sc.h5ad'
    >>> name = 'semi-profile-covid-19'
    >>> cohort_name = 'covid-19'
    >>> representative_ids['MH9143273', 'BGCV09_CV0279', 'MH9143276', 'MH8919282']
    >>> get_cohort_sc(name=name, path=path, cohort_name= cohort_name, representative_ids=representative_ids)
    """
    
    scdata = anndata.read_h5ad(path)
    sids = []
    f = open(name + '/sids.txt', 'r')
    lines = f.readlines()
    for l in lines:
        sids.append(l.strip())
    f.close()
    
    rsids = representative_ids
    rmask=[]    
    for i in range(len(scdata.obs.index)):
        sid = scdata.obs['sample_ids'][i]
        if sid in rsids:
            rmask.append(True)
        else:
            rmask.append(False)
    rmask = np.array(rmask)
    
    repredata = scdata[rmask,:]
    
    X = repredata.X
    if X.max() < 20:
        if scipy.sparse.issparse(X):
            X = np.array(X.todense())
        X = np.exp(X) - 1
        X = sparse.csr_matrix(X)
    adata = anndata.AnnData(X)
    adata.obs = repredata.obs
    adata.var = repredata.var
    
    adata.write(name + '/' + cohort_name + '.h5ad')
    
    print('Obtained single-cell data for representatives from the cohort.')
    
    return






def estimate_cost(total_samples:int,n_representatives:int) -> Tuple[float,float]:
    """
    Estimate the cost of semi-profiling and real-profiling.

    Parameters
    ----------
    total_samples
        Total number of samples
    n_representatives:
        Number of representatives
        
    Returns
    -------
    semicost
        Cost of semi-profiling
    realcost
        Cost of real-profiling

    Example
    -------
    >>> semicost, realcost = estimate_cost(12,2)
    """
    
    bulkcost = 7000 + total_samples * 110
    sccost = 5000 * 0.3 * n_representatives
    
    semicost = bulkcost + sccost
    realcost = 5000 * 0.3 * total_samples
    
    pct = round((realcost - semicost)/realcost * 100,1)
    print()
    print('Estimated semi-profiling cost: $'+str(semicost))
    print('Estimated cost if conducting real single-cell profiling: $'+str(realcost))
    print('Percentage saved: ' + str(pct) + '%')
    return semicost,realcost


# visualizing reconstruction performance of a representative
def visualize_recon(name:str, representative:Union[int,str], save = None) -> None:
    """
    Visualize the performance of reconstruction by plotting the original and reconstructed data in the same UMAP.

    Parameters
    ----------
    name
        Project name
    representative
        Representative sample ID (string or int)
    save
        Path within the 'figures' folder to save the plot
        
    Returns
    -------
        None

    Example
    -------
    >>> name = 'project name'
    >>> visualize_recon(name, 6)

    """
    
    sids = []
    f = open(name+'/sids.txt','r')
    lines = f.readlines()
    for l in lines:
        sids.append(l.strip())
    f.close()

    if type(representative) == type(1):
        sid = sids[representative]
    else:
        sid = representative
    repredata = anndata.read_h5ad(name + '/sample_sc/'+sid+'.h5ad')
    x0 = repredata.X
    genelen = x0.shape[1]
    
    
    device = 'cpu'
    k=15
    batch_size = x0.shape[0]
    diagw=1
    adata,adj,variances,bulk,geneset_len = setdata(name,sid,device,k,diagw)
    model = fastgenerator(variances,None,geneset_len,adata,n_hidden=256,n_latent=32,dropout_rate=0)
    path = name + '/models/fastreconst2_' + sid 
    model.module.load_state_dict(torch.load(path))
    
    x1 = []
    scdl = model._make_data_loader(
            adata=adata,batch_size=batch_size
    )
    for tensors in scdl:
        samples = model.module.sample(tensors, n_samples=1)
        x1.append(samples)
    
    # save inferred data
    x1 = np.array(torch.cat(x1))[:,:genelen]
    
    vdata = anndata.AnnData(np.concatenate([x0,x1],axis=0))
    rc = []
    for i in range(x0.shape[0]):
        rc.append('Representative')
    for i in range(x1.shape[0]):
        rc.append('Reconstructed')

    vdata.obs['reconstruction'] = rc
    sc.pp.log1p(vdata)
    sc.tl.pca(vdata)
    sc.pp.neighbors(vdata)
    sc.tl.umap(vdata)
    
    palette = {'Representative':'blue','Reconstructed':'gray'}
    sc.pl.umap(vdata,color='reconstruction',alpha=0.5,palette=palette,save=save)

# visualizing inference performance for a target sample
def visualize_inferred(name:str, target:int, representatives:list, cluster_labels:list, realdata_path:str = 'example_data/scdata.h5ad', n_comps:int=100, save = None) -> None:
    """
    Visualize the inference performance by plotting the representative, inferred target, and target ground truth in the same UMAP.

    Parameters
    ----------
    name
        Project name
    target:
        Target sanmple ID (number)
    representatives:
        Representatives sample IDs (int)
    cluster_labels:
        Cluster labels
    realdata_path:
        Path to the real-profiled single-cell data
    n_comps:
        The number of PCs in PCA
    save
        Path within the 'figures' folder to save the plot
        
    Returns
    -------
        None
        
    Example
    -------
    >>> name = 'project name'
    >>> 
    >>> # load representatives and cluster labels lists
    >>> repres = []
    >>> f=open(name + '/status/init_representatives.txt','r')
    >>> lines = f.readlines()
    >>> f.close()
    >>> for l in lines:
    >>>     repres.append(int(l.strip()))
    >>> 
    >>> cl = []
    >>> f=open(name + '/status/init_cluster_labels.txt','r')
    >>> lines = f.readlines()
    >>> f.close()
    >>> for l in lines:
    >>>     cl.append(int(l.strip()))
    >>>
    >>> visualize_inferred(name, 0, repres, cl)
    """
    
    sids = []
    f = open(name+'/sids.txt','r')
    lines = f.readlines()
    for l in lines:
        sids.append(l.strip())
    f.close()
    representative = representatives[cluster_labels[target]]
    repredata = anndata.read_h5ad(name + '/sample_sc/'+sids[representative]+'.h5ad')
    x0 = repredata.X
    genelen = x0.shape[1]
    
    xsem = np.load(name + '/inferreddata/' + sids[representative] + '_to_' + sids[target]+'.npy' )
    x1 = xsem[:,:genelen]
    alldata = anndata.read_h5ad(realdata_path)
    tgtdata = alldata[alldata.obs['sample_ids']==sids[target]]
    
    
    x2 = tgtdata.X
    if scipy.sparse.issparse(x2):
        x2 = x2.todense()
    x2 = np.array(x2)
    
    if x2.max()<20:
        x2 = np.exp(x2)-1
    
    vdata = anndata.AnnData(np.concatenate([x0,x1,x2],axis=0))
    rc = []
    for i in range(x0.shape[0]):
        rc.append('Representative')
    for i in range(x1.shape[0]):
        rc.append('Target inferred')
    for i in range(x2.shape[0]):
        rc.append('Target ground truth')
    vdata.obs['inference'] = rc
    sc.pp.log1p(vdata)
    sc.tl.pca(vdata,n_comps=n_comps)
    sc.pp.neighbors(vdata)
    sc.tl.umap(vdata)
    
    palette = {'Representative':'blue','Target inferred':'yellow','Target ground truth':'red'}
    sc.pl.umap(vdata,color='inference',alpha=0.5,palette=palette,save=save)
    
    return

def loss_curve(name:str, reprepid:int=None,tgtpid:int=None,stage:int=1, save = None)->None:
    """
    Visualize the training loss curves

    Parameters
    ----------
    name
        Project name
    reprepid:
        Representative sanmple ID 
    tgtpid:
        target sample IDs 
    stage:
        The training stage to visualize, 1: pretrain1; 2: pretrain2; 3: inference 
    save
        Path within the 'figures' folder to save the plot
        
    Returns
    -------
        None
        
    Example
    -------
    >>> name = 'project name'
    >>> loss_curve(name, reprepid='BGCV09_CV0279',tgtpid=None,stage=1) # or loss_curve(name, sids, reprepid=6,tgtpid=None,stage=1)

        
    """
    
    sids = []
    f = open(name+'/sids.txt','r')
    lines = f.readlines()
    for l in lines:
        sids.append(l.strip())
    f.close()
    
    if type(reprepid) == type(1):
        reprepid = sids[reprepid]
    if type(tgtpid) != type(None):
        if type(tgtpid) == type(1):
            tgtpid = sids[tgtpid]
            
    if (stage == 1) or (stage == 'pretrain1') or (stage == 'pretrain 1'): 
        fname = name+'/history/pretrain1_' + reprepid + '.pkl'
        with open(fname, 'rb') as pickle_file:
            history = pickle.load(pickle_file)
        plt.plot(history['train_loss_epoch'])
        plt.title('Pretrain 1 Loss Curve')
        
    elif (stage == 2) or (stage == 'pretrain2') or (stage == 'pretrain 2'):
        fname = name+'/history/pretrain2_' + reprepid + '.pkl'
        with open(fname, 'rb') as pickle_file:
            history = pickle.load(pickle_file)
        plt.plot(history['train_loss_epoch'])
        plt.title('Pretrain 2 Loss Curve')
        
    elif (stage == 3) or (stage == 'inference'):
        fname = name+'/history/inference_' + reprepid + '_to_' + tgtpid + '.pkl'
        with open(fname, 'rb') as pickle_file:
            history = pickle.load(pickle_file)
        fig, axs = plt.subplots(5,1,figsize=(2,8))
        axs[0].plot(history['bulk0'])
        axs[1].plot(history['bulk1'])
        axs[2].plot(history['bulk2'])
        axs[3].plot(history['bulk3'])
        axs[4].plot(history['bulk4'])
        plt.title('Inference Loss')
    
    if save!= None:
        plt.savefig(name + 'figures/'+save,dpi=600,bbox_inches='tight')
    
    return


def assemble_cohort(name:str,
                    representatives:Union[list,str],
                    cluster_labels:Union[list,str],
                    celltype_key:str = 'celltypes',
                    sample_info_keys:list = ['states_collection_sum'],
                    bulkpath:str = 'example_data/bulkdata.h5ad',
                    annotator = MLPClassifier(hidden_layer_sizes=(200,))):
    """
    Assemble inferred sample data and representative sample data into semi-profiled cohort and annotate the celltype. 
    
    Parameters
    ----------
    name: 
        Project name
    representatives:
        Either a list of representatives or path to a txt file specifying the representative information
    cluster_labels:
        Either a list of sample cluster labels or path to a txt file specifying the sample cluster label information
    celltype_key:
        The key in .obs specifying the cell type information
    sample_info_keys:
        Keys for other sample-level information to be stored in the assembled dataset
    bulkpath:
        Path to the bulk data
    
    Returns
    -------
    semidata:
        The assembled and annotated semi-profiled dataset

    Example
    -------
    >>> semisdata = assemble_cohort(name,
    >>>                 repre,
    >>>                 cls,
    >>>                 celltype_key = 'celltypes',
    >>>                 sample_info_keys = ['states_collection_sum'],
    >>>                 bulkpath = 'example_data/bulkdata.h5ad')


    """
    
    
    print('Start assembling semi-profiled cohort.')

    # read sample ids
    sids = []
    f = open(name + '/sids.txt', 'r')
    lines = f.readlines()
    for l in lines:
        sids.append(l.strip())
    f.close()
    
    # read representatives and clustering info
    if type(representatives) == type(''):
        rs = []
        f = open(representatives, 'r')
        lines = f.readlines()
        for l in lines:
            rs.append(int(l.strip()))
        f.close()
        representatives = rs
    if type(cluster_labels) == type(''):
        cl = []
        f = open(cluster_labels, 'r')
        lines = f.readlines()
        for l in lines:
            cl.append(int(l.strip()))
        f.close()
        cluster_labels = cl
    
    
    
    ### read expression data
    # read representative single-cell h5ad
    xtrain = []
    ytrain = []
    for i in range(len(representatives)):
        sid = sids[representatives[i]]
        adata = anndata.read_h5ad(name + '/sample_sc/' + sid + '.h5ad')
        xtrain.append(np.array(adata.X))
        sample_celltype = list(adata.obs[celltype_key])
        ytrain = ytrain + sample_celltype
    xtrain = np.concatenate(xtrain, axis=0)
    if xtrain.max()>10:
        xtrain = np.log1p(xtrain)
    ytrain = np.array(ytrain)
    
    
    # train annotator
    print('Training cell type annotator.')
    st =  timeit.default_timer()
    #annotator = MLPClassifier(hidden_layer_sizes=(200,)) 
    annotator.fit((xtrain),ytrain)
    ed =  timeit.default_timer()
    print('Finished. Cost ' + str(ed-st) + ' seconds.')
    
    
    # assemble cohort
    print('Generating semi-profiled cohort data.')
    X = [] # expression matrix
    source = [] # inferred or representative
    semicelltype = [] # cell type
    semiids = [] # sample ids
    
    sampleinfo = {} #sample information to be preserved
    for k in sample_info_keys: # other sample information to be preserved
        sampleinfo[k] = []
    
    # read expression data
    
    for i in range(len(sids)):
        if i in representatives:
            sid = sids[i]
            adata = anndata.read_h5ad(name + '/sample_sc/' + sid + '.h5ad')
            
            # expression
            X.append(np.array(adata.X))
            
            # essential keys
            semicelltype = semicelltype + list(adata.obs[celltype_key])
            #semiids = semiids + list(adata.obs['sample_ids'])
            for j in range(adata.X.shape[0]):
                semiids.append(sids[i])
                source.append('Representative')
                
            ## other keys
            bulkdata = anndata.read_h5ad(bulkpath)
            for k in sample_info_keys:
                for j in range(adata.X.shape[0]):
                    sampleinfo[k].append(bulkdata.obs[k][i])
                    
        else: # inferred target sample
            # expression
            x = np.load(name + '/inferreddata/' + \
                        sids[representatives[cluster_labels[i]]] + \
                        '_to_' + sids[i] + '.npy')
            X.append(np.array(x))
            
            # essential keys
            predicted_celltype = annotator.predict(np.log1p(x))
            semicelltype = semicelltype + list(predicted_celltype)
            for j in range(x.shape[0]):
                semiids.append(sids[i])
                source.append('Inferred')
                
            ## other keys
            bulkdata = anndata.read_h5ad(bulkpath)
            for k in sample_info_keys:
                for j in range(x.shape[0]):
                    sampleinfo[k].append(bulkdata.obs[k][i])
                    
    semidata = anndata.AnnData(np.array(np.concatenate(X)))
    semidata.obs['source'] = source
    semidata.obs[celltype_key] = semicelltype
    semidata.obs['sample_ids'] = semiids
    semidata.var = adata.var
    
    for k in sample_info_keys:
        semidata.obs[k] = sampleinfo[k]
    
    semidata.write(name + '/semidata.h5ad')
    
    print('Finished assembling semi-profiled cohort. Output as semidata.h5ad')
    return semidata


def assemble_representatives(name:str,celltype_key:str='celltypes',sample_info_keys:list = ['states_collection_sum'],rnd:int=2,batch:int=2,gtsc:str = 'example_data/scdata.h5ad', gtbulk:str = 'example_data/bulkdata.h5ad') -> Tuple[anndata.AnnData, anndata.AnnData]:
    """
    Assemble previous round of inferred representative data and annotate the cell type. The real-profiled representatives in the current round is also provided for comparison.
    
    Parameters
    ----------
    name: 
        Project name
    celltype_key:
        The key in .obs specifying the cell type information
    sample_info_keys:
        Keys for other sample-level information to be stored in the assembled dataset
    rnd:
        The round of semi-profiling to assemble. For example, select the second round (2 batches of representatives) using rnd = 2
    batch:
        The representative selection batch size
    gtsc:
        Path to ground truth data
    gtbulk: 
        Path to bulk data
        
    Returns
    -------
    realrepdata:
        The real-profiled representative dataset
    infrepdata:
        The inferred representative dataset

    Example
    -------
    >>> real_rep, inferred_rep = assemble_representatives(name,celltype_key='celltypes',sample_info_keys = ['states_collection_sum'],rnd=2,batch=2)


    """
    
    sids=[]
    f = open(name+'/sids.txt','r')
    lines=f.readlines()
    for l in lines:
        sids.append(l.strip())
    f.close()
    
    
    reps = []
    f=open(name + '/status/eer_representatives_' + str(rnd) + '.txt','r')
    lines = f.readlines()
    for l in lines:
        reps.append(int(l.strip()))
    f.close()


    newcl=[]
    f=open(name + '/status/eer_cluster_labels_' + str(rnd) + '.txt','r')
    lines = f.readlines()
    for l in lines:
        newcl.append(int(l.strip()))
    f.close()

    newreps = reps[-batch:]
    
    alldata = anndata.read_h5ad(gtsc)
    
    genelen = len(alldata.var.index)
    
    newrepsids = []
    for i in newreps:
        newrepsids.append(sids[i])
    
    repmask =[] 
    for i in range(alldata.X.shape[0]):
        if alldata.obs['sample_ids'][i] in newrepsids:
            repmask.append(True)
        else:
            repmask.append(False)
    repmask = np.array(repmask)
    
    realrepdata = alldata[repmask]
    
    
    
    
    ### load semi profiled data 
    oldreps = reps[:-batch]
    
    if rnd==2:
        clname = name + '/status/init_cluster_labels.txt'
    else:
        clname = name + '/status/eer_cluster_labels_'+str(rnd-1)+'.txt'
    
    oldcl = []
    f = open(clname, 'r')
    lines = f.readlines()
    for l in lines:
        oldcl.append(int(l.strip()))
    f.close()
    
    
    ### train annotator using the previous representatives
    xtrain = []
    ytrain = []
    for i in range(len(oldreps)):
        sid = sids[oldreps[i]]
        adata = anndata.read_h5ad(name + '/sample_sc/' + sid + '.h5ad')
        x = np.array(adata.X)
        if x.max() > 20:
            x = np.log1p(x)
        xtrain.append(x)
        sample_celltype = list(adata.obs[celltype_key])
        ytrain = ytrain + sample_celltype
    xtrain = np.concatenate(xtrain, axis=0)
    ytrain = np.array(ytrain)
    # train annotator
    print('Training cell type annotator.')
    st =  timeit.default_timer()
    annotator = MLPClassifier() #hidden_layer_sizes=(200,)) 
    annotator.fit((xtrain),ytrain)
    ed =  timeit.default_timer()
    print('Finished. Cost ' + str(ed-st) + ' seconds.')
    
    
    semicelltype = [] # cell type
    semiids = [] # sample ids
    sampleinfo = {} #sample information to be preserved
    for k in sample_info_keys: # other sample information to be preserved
        sampleinfo[k] = []
    xinf = []
    for i in range(batch):
        # expression
        sid = sids[newreps[i]]
        repsid = sids[oldreps[oldcl[newreps[i]]]]
        x = np.load(name + '/inferreddata/' + repsid + '_to_' + sid + '.npy')
        xinf.append(x)

        # essential keys
        if x.max()>20:
            x = np.log1p(x)
        predicted_celltype = annotator.predict(x)
        semicelltype = semicelltype + list(predicted_celltype)
        for j in range(x.shape[0]):
            semiids.append(sids[i])
            
        ## other keys
        bulkdata = anndata.read_h5ad(gtbulk)
        for k in sample_info_keys:
            for j in range(x.shape[0]):
                sampleinfo[k].append(bulkdata.obs[k][i])
                    
                    
    xinf = np.concatenate(xinf, axis=0)
    xinf = xinf[:,:genelen]
    
    infrepdata = anndata.AnnData(np.array(xinf))
    infrepdata.obs[celltype_key] = semicelltype
    infrepdata.obs['sample_ids'] = semiids
    infrepdata.var = bulkdata.var
    
    for k in sample_info_keys:
        infrepdata.obs[k] = sampleinfo[k]
    
    infrepdata.write(name + '/infrepdata_' + str(rnd) + '.h5ad')
    realrepdata.write(name + '/realrepdata_' + str(rnd) + '.h5ad')

    return realrepdata, infrepdata


def compare_umaps(
                    semidata:anndata.AnnData,
                    gtdata:anndata.AnnData,
                    name:str = 'testexample',
                    representatives:str = 'testexample/status/init_representatives.txt',
                    cluster_labels:str = 'testexample/status/init_cluster_labels.txt',
                    celltype_key:str = 'celltypes',
                    save = None
                 ) -> Tuple[anndata.AnnData, anndata.AnnData, anndata.AnnData]:
    """
    Compare the real-profiled and semi-profiled datasets by plotting them in a same UMAP
    
    Parameters
    ----------
    semidata:
        Semi-profiled dataset
    gtdata:
        Real-profiled dataset
    name: 
        Project name
    representatives:
        Path to the txt file storing the representative information
    cluster_labels:
        Path to the txt file storing the cluster label information
    celltype_key:
        The key in .obs specifying the cell type information
    save
        Path within the 'figures' folder to save the plot

    Returns
    -------
    combdata
        Combined dataset, with real-profiled cells in the front
    gtdata
        Real-profiled dataset
    semidata
        Semi-profiled dataset
        
    Example
    -------
    >>> combined_data,gtdata,semidata = compare_umaps(
    >>>             semidata = semisdata, # assembled semi-profiled dataset
    >>>             name = name,
    >>>             representatives = name + '/status/init_representatives.txt',
    >>>             cluster_labels = name + '/status/init_cluster_labels.txt',
    >>>             celltype_key = 'celltypes'
    >>>             )
    
    """
    
    # visualize UMAPs of real-profiled and semi-profiled data for comparison
    # return both datasets with PCA and UMAP coordinates added
    #gtdata = anndata.read_h5ad('example_data/scdata.h5ad')
    if type(gtdata.X) != np.ndarray:
        x0 = np.array(gtdata.X.todense())
    else:
        x0 = np.array(gtdata.X)
    if type(semidata.X) != np.ndarray:
        x1 = np.array(semidata.X.todense())
    else:
        x1 = np.array(semidata.X)
    if x0.max()<20:
        x0 = np.exp(x0)-1
    if x1.max()<20:
        x1 = np.exp(x1)-1
    
    combdata = anndata.AnnData(np.concatenate([x0,x1],axis=0))
    combdata.var = gtdata.var
    combdata.obs[celltype_key] = list(gtdata.obs[celltype_key]) + list(semidata.obs[celltype_key])
    
    cohort = []
    for i in range(gtdata.X.shape[0]):
        cohort.append('real-profiled')
    for i in range(semidata.X.shape[0]):
        cohort.append('semi-profiled')
    combdata.obs['cohort'] = cohort
    
    sc.pp.log1p(combdata)
    sc.tl.pca(combdata,n_comps=100)
    sc.pp.neighbors(combdata,n_neighbors=50)
    sc.tl.umap(combdata)
    
    if save!=None:
        savecomb = save + 'real_vs_semi'
        savereal = save + 'real'
        savesemi = save + 'semi'
    else:
        savecomb = None
        savereal = None
        savesemi = None
    sc.pl.umap(combdata,color = 'cohort', title = 'Real-profiled VS Semi-profiled', save = savecomb)
    
    
    semidata.obsm['X_umap'] = combdata.obsm['X_umap'][gtdata.X.shape[0]:]
    gtdata.obsm['X_umap'] = combdata.obsm['X_umap'][:gtdata.X.shape[0]]
    
    sc.pl.umap(gtdata,color = 'celltypes',title = 'Real-profiled', save = savereal)
    sc.pl.umap(semidata,color = 'celltypes',title = 'Semi-profiled', save = savesemi)
    
    return combdata,gtdata,semidata


def compare_adata_umaps(
    semidata:anndata.AnnData,
    gtdata:anndata.AnnData,                
    name:str = 'testexample',
    celltype_key:str = 'celltypes', save = None
    ) -> Tuple[anndata.AnnData, anndata.AnnData, anndata.AnnData]:
    """
    Compare the real-profiled and semi-profiled datasets by plotting them in a same UMAP
    
    Parameters
    ----------
    semidata:
        Semi-profiled dataset
    gtdata:
        Real-profiled dataset
    name: 
        Project name
    celltype_key:
        The key in .obs specifying the cell type information
    save
        Path within the 'figures' folder to save the plot
        
    Returns
    -------
    combdata
        Combined dataset, with real-profiled cells in the front
    gtdata
        Real-profiled dataset
    semidata
        Semi-profiled dataset
        
    Example
    -------
    >>> combdata, gtdata, semidata = compare_adata_umaps(
    >>> inferred_rep, # inferred representatives from the last round
    >>> real_rep,     # real-profiled representatives from the current round          
    >>> name = name,  # project name
    >>> celltype_key = 'celltypes'
    >>> )
        
    """
    
    x0 = gtdata.X
    if type(x0) != np.ndarray:#== scipy.sparse.csr_matrix:
        x0 = x0.todense()
    x1 = semidata.X
    if type(x1) != np.ndarray:#scipy.sparse.csr_matrix:
        x1 = x1.todense()
    
    x0 = np.array(x0)
    x1 = np.array(x1)
    
    if x1.max()>20:    
        x1 = np.log1p(x1)
    if x0.max()>20:    
        x0 = np.log1p(x0)
    combdata = anndata.AnnData(np.concatenate([x0,x1],axis=0))
    combdata.var = gtdata.var
    combdata.obs[celltype_key] = list(gtdata.obs[celltype_key]) + list(semidata.obs[celltype_key])
    
    cohort = []
    for i in range(gtdata.X.shape[0]):
        cohort.append('real-profiled')
    for i in range(semidata.X.shape[0]):
        cohort.append('semi-profiled')
    combdata.obs['cohort'] = cohort
    
    #sc.pp.log1p(combdata)
    sc.tl.pca(combdata,n_comps=100)
    sc.pp.neighbors(combdata,n_neighbors=50)
    sc.tl.umap(combdata)
    
    sc.pl.umap(combdata,color = 'cohort', title = 'Real-profiled VS Semi-profiled')
    
    
    semidata.obsm['X_umap'] = combdata.obsm['X_umap'][gtdata.X.shape[0]:]
    gtdata.obsm['X_umap'] = combdata.obsm['X_umap'][:gtdata.X.shape[0]]
    
    if save!=None:
        savereal = save+'read'
        savesemi = save+'semi'
    else:
        savereal = None
        savesemi = None
    sc.pl.umap(gtdata,color = celltype_key,title = 'Real-profiled',save=savereal)
    sc.pl.umap(semidata,color = celltype_key,title = 'Semi-profiled',save=savesemi)
    
    return combdata,gtdata,semidata



def celltype_proportion(adata:anndata.AnnData,totaltypes:Union[np.array,list]) -> np.array:
    """
    Compute the cell type proportion in a dataset
    
    Parameters
    ----------
    adata:
        The dataset to investigate
    totaltypes:
        The total cell types to consider

    Returns
    -------
    prop
        Cell type proportion
        
    Example
    -------
    >>> real_prop = celltype_proportion(real_rep,totaltypes)
    """
    
    prop = np.zeros(len(totaltypes))
    for i in range(len(totaltypes)):
        prop[i] += (adata.obs['celltypes'] == totaltypes[i]).sum()
    
    # norm
    prop = prop/prop.sum()
    
    return prop




def composition_by_group(
    adata:anndata.AnnData,
    colormap:Union[str,list] = None,
    groupby:str = None,
    title:str = 'Cell type composition'
    , save = None
    ) -> None:
    """
    Visualizing the cell type composition in each group.
    
    Parameters
    ----------
    adata:
        The dataset to investigate
    colormap:
        The colormap for visualization
    groupby:
        The key in .obs specifying groups.
    title:
        Plot title
    save
        Path within the 'figures' folder to save the plot

    Returns
    -------
        None

    Example
    -------
    >>> groupby = 'states_collection_sum'
    >>> composition_by_group(
    >>>     adata = gtdata,
    >>>     groupby = groupby,
    >>>     title = 'Ground truth'
    >>>     )

        
    """
    
    totaltypes = np.array(adata.obs['celltypes'].cat.categories)
    
    if colormap == None:
        colormap = adata.uns['celltypes_colors']
    
    conditions = np.unique(adata.obs[groupby])
    
    n = conditions.shape[0]

    percentages = []
    for i in range(conditions.shape[0]):
        condition_prop = celltype_proportion(adata[adata.obs[groupby] == conditions[i]],totaltypes)
        percentages.append(condition_prop)
        
    fig, axs = plt.subplots(n,1,figsize=(n,1))
    axs[0].set_title(title)
    
    for j in range(n):
        for i in range(len(totaltypes)):
            axs[j].barh(conditions[j],percentages[j][i], left = sum(percentages[j][:i]),color = colormap[i])
            axs[j].set_xlim([0, 1])
            axs[j].set_yticklabels([])
            axs[j].yaxis.set_tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
            
            if j != n:
                axs[j].set_xticklabels([])
                
        axs[j].text(-0.01, 0, conditions[j], ha='right', va='center')
                
    patches = []
    for i in range(len(totaltypes)):
        patches.append(mpatches.Patch(color=colormap[i], label=totaltypes[i]))
    axs[-1].legend(handles=patches, loc='center left', bbox_to_anchor=(1.1, n))
    
    
    plt.xlabel('Proportion')
    
    if save != None:
        plt.savefig(name + 'figures/'+save,dpi=600,bbox_inches='tight')
        
    return



def geneset_pattern(
    adata: anndata.AnnData,
    genes: list,
    condition_key: str,
    celltype_key: str,
    totaltypes: Union[np.array,list] = None,
    baseline:str = None,
    save = None
    ) -> np.array:
    """
    Generate heatmaps for visualizing gene set activation pattern in a dataset.
    
    Parameters
    ----------
    adata:
        The dataset to investigate
    genes:
        The list of genes in the gene set
    condition_key:
        The key in .obs specifying different sample conditions.
    celltype_key:
        The key in .obs specifying cell type information
    totaltypes:
        Total cell type names
    baseline:
        Baseline condition
    save
        Path within the 'figures' folder to save the plot
    
    Returns
    -------
    pattern
        np.array
        
    Example
    -------
    >>> gtmtx = geneset_pattern(gtdata,IFN_genes,'states_collection_sum','celltypes')
    """
    
    sc.tl.score_genes(adata, genes, ctrl_size=50, gene_pool=None, n_bins=25, score_name='geneset', random_state=0, copy=False, use_raw=None)
    
    conditions = np.unique(adata.obs[condition_key])
    if type(totaltypes) == type(None):
        totaltypes = np.unique(adata.obs[celltype_key])
    
    scores = adata.obs['geneset']

    pattern = np.zeros((len(conditions),len(totaltypes)))
    
    for i in range(len(conditions)):
        condition = conditions[i]
        condition_mask = (adata.obs[condition_key] == condition)
        for j in range(len(totaltypes)):
            celltype = totaltypes[j]
            celltype_mask = (adata.obs[celltype_key] == celltype)
            relevantscores = scores[np.logical_and(condition_mask, celltype_mask)]
            pattern[i,j] = relevantscores.mean()
            
    sn.heatmap(pattern,xticklabels = totaltypes,yticklabels=conditions,cmap="coolwarm",square = True)
    if (save != None) and (save!=False):
        plt.savefig(save)
    
    
    return pattern





# dot plot
def celltype_signature_comparison(gtdata:anndata.AnnData,semisdata:anndata.AnnData,celltype_key:str, top_n_degs:int = 3, save = None) -> Tuple[float, float, list]:
    """
    Use dotplot to compare the cell type signatures found using the real-profiled dataset and the semi-profiled datset.
    
    Parameters
    ----------
    gtdata:
        The real-profiled dataset
    semisdata:
        The semi-profiled dataset
    celltype_key:
        The key in .obs specifying the cell type labels
    save
        Path within the 'figures' folder to save the plot
    top_n_degs
        Top n DEGs used for each cell type in the dotplot
    
    Returns
    -------
        Tuple[float, float, List]: A tuple containing two floats and a list: Pearson correlation of dot sizes, dot colors, and a list of signature genes.

    Example
    -------
    >>> celltype_signature_comparison(gtdata=gtdata,semisdata=semisdata,celltype_key='celltypes')
    """
    totaltypes = np.unique(gtdata.obs[celltype_key])
    
    sc.tl.rank_genes_groups(gtdata, celltype_key, method='t-test')
    
    signatures = []
    for j in range(totaltypes.shape[0]):
        typede = []
        for i in range(top_n_degs):
            g = gtdata.uns['rank_genes_groups']['names'][i][j]
            typede.append(g)
        signatures = signatures + typede
    
    
    if save!=None:
        savegt = save + 'gt'
        savesemi = save + 'semi'
    else:
        savegt = None
        savesemi = None
    sc.pl.dotplot(gtdata, signatures, groupby = celltype_key, save = savegt)
    sc.pl.dotplot(semisdata, signatures, groupby = celltype_key,save=savesemi)
    
    dpgt = sc.pl.dotplot(gtdata, signatures, groupby=celltype_key,return_fig=True)
    dpgtcolor = (dpgt.dot_color_df.to_numpy())
    dpgtsize = (dpgt.dot_size_df.to_numpy())
    
    dpsemi = sc.pl.dotplot(semisdata, signatures, groupby=celltype_key,return_fig=True)
    dpsemicolor = (dpsemi.dot_color_df.to_numpy())
    dpsemisize = (dpsemi.dot_size_df.to_numpy())
    
    print('Expression fraction (size) similarity between real and semi-profiled:')
    print(scipy.stats.pearsonr(dpgtsize.flatten(), dpsemisize.flatten()))
    print('Expression intensity (color) similarity between real and semi-profiled')
    print(scipy.stats.pearsonr(dpgtcolor.flatten(), dpsemicolor.flatten()))
    return scipy.stats.pearsonr(dpgtsize.flatten(), dpsemisize.flatten())[0], scipy.stats.pearsonr(dpgtcolor.flatten(), dpsemicolor.flatten())[0], signatures



### statistics utils and rrho
def comb(a:int,b:int) -> mp.mpf:
    """
    Combination number
    
    Parameters
    ----------
    a:
        The total number of choice
    b:
        The number of elements to choose.

    Returns
    -------
    cad
        Choose a from b
        
    Example
    -------
    >>> print(comb(5,3))
    """
    
    a=mp.mpf(a)
    b=mp.mpf(b)
    cab = fac(a)/fac(a-b)/fac(b)
    return cab

def hyperp(N:int,n1:int,n2:int,k:int) -> mp.mpf:
    """
    Returns the pmf of a hypergeometric test.
    
    Parameters
    ----------
    N:
        Population size. In our case this is the total number of gene
    n1:
        The number of element in the first set
    n2:
        The number of element in the second set
    k:
        The number of overlap

    Returns
    -------
    p
        cdf
        
    Example
    -------
    >>> print(hyperp(6000,100,100,97))
    
    """
    #cdf
    
    N = mp.mpf(N)
    n1 = mp.mpf(n1)
    n2 = mp.mpf(n2)
    k = mp.mpf(k)
    p = comb(n2,k)*comb(N-n2,n1-k)/comb(N,n1)
    return p 

def hypert(N:int,n1:int,n2:int,k:int) -> mp.mpf:
    """
    Returns the p-value of a hypergeometric test.
    
    Parameters
    ----------
    N:
        Population size. In our case this is the total number of gene
    n1:
        The number of element in the first set
    n2:
        The number of element in the second set
    k:
        The number of overlap

    Returns
    -------
    pval
        p-value
    
    Example
    -------
    >>> print(hypert(6000,100,100,97))
    
    """
    cdf = mp.mpf(0)
    for i in range(0,int(k)+1):
        cdf += hyperp(N,n1,n2,i)
       # print()
    return (1-cdf)

def rrho_plot(name:str, list1:list, list2:list, list3:list, list4:list, celltype:str, population:int ,upperbound:int=50, save = None) -> Tuple[np.array,np.array,np.array,np.array]:
    """
    Generates data for RRHO graph used to compare the positive and negative markers found using real-profiled and semi-profiled datasets.
    
    Parameters
    ----------
    name
         project name
    list1:
         Positive markers found using the real-profiled dataset
    list2:
        Positive markers found using the semi-profiled dataset
    list3:
        Negative markers found using the real-profiled dataset
    list4:
        Negative markers fuond using the semi-profiled dataset
    celltype:
        The selected cell type to analyze 
    population:
        The population size in hypergeometric test for evaluating the overlap between two gene lists. In our case this will be to total number of genes used.
    upperbound:
        The upperbound for negative log p-value for visualization
    save
        Path within the 'figures' folder to save the plot

    Returns
    -------
    rrho_matrix1
        Values for the first quadrant
    rrho_matrix2
        Values for the second quadrant
    rrho_matrix3
        Values for the third quadrant
    rrho_matrix4
        Values for the forth quadrant
        
    """
    
    n1 = len(list1) # gt pos
    n2 = len(list2) # semi pos
    n3 = len(list3) # gt neg
    n4 = len(list4) # semi neg

    # Calculate the maximum rank to consider
    max_rank1 = max(n3, n2)
    max_rank2 = max(n3, n4)
    max_rank3 = max(n1, n2)
    max_rank4 = max(n1, n4)

    # Initialize the RRHO plot matrix
    rrho_matrix1 = np.zeros((max_rank1, max_rank1))
    rrho_matrix2 = np.zeros((max_rank2, max_rank2))
    rrho_matrix3 = np.zeros((max_rank3, max_rank3))
    rrho_matrix4 = np.zeros((max_rank4, max_rank4))
    
    # Iterate over different cutoff ranks
    upper=1e-50
    ## ax1 
    for rank1 in range(1, max_rank1 + 1):
        for rank2 in range(1, max_rank1 + 1):
            # Get the top genes up to the cutoff ranks
            top_genes1 = set(list1[:rank1])
            top_genes2 = set(list4[:rank2])
            union = np.unique(list(top_genes1) + list(top_genes2))
            # Calculate the overlap between the two gene sets
            overlap = len(top_genes1.intersection(top_genes2))
            # Calculate the hypergeometric p-value for the overlap
            p_value = hypert(population,rank1,rank2,overlap)#1 - scipy.stats.hypergeom.cdf(overlap, 6000, rank1, rank2)
            #print(p_value)
            p_value = float(p_value)
            if p_value < upper:
                p_value = upper
            if np.isnan(p_value):
                print(overlap, len(union), rank1, rank2)
            # Store the negative logarithm of the p-value in the RRHO matrix
            rrho_matrix1[rank1 - 1, rank2 - 1] = -np.log10(p_value)
    #ax2
    for rank1 in range(1, max_rank2 + 1):
        for rank2 in range(1, max_rank2 + 1):
            # Get the top genes up to the cutoff ranks
            top_genes1 = set(list1[:rank1])
            top_genes2 = set(list2[:rank2])
            union = np.unique(list(top_genes1) + list(top_genes2))
            # Calculate the overlap between the two gene sets
            overlap = len(top_genes1.intersection(top_genes2))
            # Calculate the hypergeometric p-value for the overlap
            p_value = hypert(population,rank1,rank2,overlap)#1 - scipy.stats.hypergeom.cdf(overlap, 6000, rank1, rank2)
            #print(p_value)
            p_value = float(p_value)
            if p_value < upper:
                p_value = upper
            if np.isnan(p_value):
                print(overlap, len(union), rank1, rank2)
            # Store the negative logarithm of the p-value in the RRHO matrix
            rrho_matrix2[rank1 - 1, rank2 - 1] = -np.log10(p_value)
            
    for rank1 in range(1, max_rank3 + 1):
        for rank2 in range(1, max_rank3 + 1):
            # Get the top genes up to the cutoff ranks
            top_genes1 = set(list3[:rank1])
            top_genes2 = set(list4[:rank2])
            union = np.unique(list(top_genes1) + list(top_genes2))
            # Calculate the overlap between the two gene sets
            overlap = len(top_genes1.intersection(top_genes2))
            # Calculate the hypergeometric p-value for the overlap
            p_value = hypert(population,rank1,rank2,overlap)#1 - scipy.stats.hypergeom.cdf(overlap, 6000, rank1, rank2)
            #print(p_value)
            p_value = float(p_value)
            if p_value < upper:
                p_value = upper
            if np.isnan(p_value):
                print(overlap, len(union), rank1, rank2)
            # Store the negative logarithm of the p-value in the RRHO matrix
            rrho_matrix3[rank1 - 1, rank2 - 1] = -np.log10(p_value)
            
    for rank1 in range(1, max_rank4 + 1):
        for rank2 in range(1, max_rank4 + 1):
            # Get the top genes up to the cutoff ranks
            top_genes1 = set(list3[:rank1])
            top_genes2 = set(list2[:rank2])
            union = np.unique(list(top_genes1) + list(top_genes2))
            # Calculate the overlap between the two gene sets
            overlap = len(top_genes1.intersection(top_genes2))
            # Calculate the hypergeometric p-value for the overlap
            p_value = hypert(population,rank1,rank2,overlap)#1 - scipy.stats.hypergeom.cdf(overlap, 6000, rank1, rank2)
            #print(p_value)
            p_value = float(p_value)
            if p_value < upper:
                p_value = upper
            if np.isnan(p_value):
                print(overlap, len(union), rank1, rank2)
            # Store the negative logarithm of the p-value in the RRHO matrix
            rrho_matrix4[rank1 - 1, rank2 - 1] = -np.log10(p_value)
            
            
    fig, axs = plt.subplots(2, 2,figsize=(6,6))
    cmap = 'magma'
    
    rrho_matrix2 = np.flip(rrho_matrix2,axis=1)
    rrho_matrix3 = np.flip(rrho_matrix3,axis=0)
    rrho_matrix4 = np.flip(np.flip(rrho_matrix4,axis=0),axis=1)
    
    vmx = 50
    # Plot heatmap in the first quadrant
    im1 = axs[0, 0].imshow(rrho_matrix1, cmap=cmap,vmax=vmx, aspect='auto')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    # Plot heatmap in the second quadrant
    im2 = axs[0, 1].imshow(rrho_matrix2, cmap=cmap,vmax=vmx, aspect='auto')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    # Plot heatmap in the third quadrant
    im3 = axs[1, 0].imshow(rrho_matrix3, cmap=cmap,vmax=vmx, aspect='auto')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    # Plot heatmap in the fourth quadrant
    im4 = axs[1, 1].imshow(rrho_matrix4, cmap=cmap,vmax=vmx, aspect='auto')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    
    cbar_ax = fig.add_axes([1, 0.025, 0.04, 0.95])
    cbar = fig.colorbar(im4, ax=axs[1, 1], fraction=0.046 * 2, pad=0.04,cax=cbar_ax,label='-log10(p)')
    plt.tight_layout()
    if save!= None:
        plt.savefig(name + '/figures/'+save,dpi=600,bbox_inches='tight')
    plt.show()
    
    
    return rrho_matrix1,rrho_matrix2,rrho_matrix3,rrho_matrix4




def rrho(name,gtdata:anndata.AnnData,semisdata:anndata.AnnData,celltype_key:str,celltype:str, save = None) -> Tuple[np.array,np.array,np.array,np.array]:
    """
    Use RRHO graph to compare the positive and negative markers found using real-profiled and semi-profiled datasets.
    
    Parameters
    ----------
    name
        Project name
    gtdata:
        Real-profiled (ground truth) data
    semisdata:
        Semi-profiled dataset
    celltype_key:
        The key in anndata.AnnData.obs for storing the cell type information
    celltype:
        The selected cell type to analyze 
    save
        Path within the 'figures' folder to save the plot
    
    Returns
    -------
    rrho_matrix1
        Values for the first quadrant
    rrho_matrix2
        Values for the second quadrant
    rrho_matrix3
        Values for the third quadrant
    rrho_matrix4
        Values for the forth quadrant

    Example
    -------
    >>> _ = rrho(gtdata=gtdata,semisdata=semisdata,celltype_key='celltypes',celltype='CD4')
    """
    
    print('Plotting RRHO for comparing ' + str(celltype) + ' markers.')
    sc.tl.rank_genes_groups(gtdata, celltype_key, method='t-test',rankby_abs=True)
    sc.tl.rank_genes_groups(semisdata, celltype_key, method='t-test',rankby_abs=True)
    
    population = semisdata.X.shape[1]
    
    nummarkers = 500
    gt_posmarkers = []
    gt_negmarkers = []
    semi_posmarkers = []
    semi_negmarkers = []
    totaltypes = (gtdata.uns['rank_genes_groups']['names'].dtype).names
    totaltypes = list(totaltypes)
    j = totaltypes.index(celltype)
    # gt
    for i in range(nummarkers):
        g = gtdata.uns['rank_genes_groups']['names'][i][j]
        score = gtdata.uns['rank_genes_groups']['scores'][i][j]
        if score > 0:
            gt_posmarkers.append(g)
        else:
            gt_negmarkers.append(g)
    #semii
    for i in range(nummarkers):
        g = semisdata.uns['rank_genes_groups']['names'][i][j]
        score = semisdata.uns['rank_genes_groups']['scores'][i][j]
        if score > 0:
            semi_posmarkers.append(g)
        else:
            semi_negmarkers.append(g)

    ngenes = 50
    gt_posmarkers = gt_posmarkers[:ngenes]
    semi_posmarkers = semi_posmarkers[:ngenes]
    gt_negmarkers = gt_negmarkers[:ngenes]
    semi_negmarkers = semi_negmarkers[:ngenes]
    rmat = rrho_plot(name = name,\
                     list1 = gt_posmarkers, \
                     list2 = semi_posmarkers,\
                     list3 = gt_negmarkers,\
                     list4 = semi_negmarkers,\
                     celltype = celltype, population=population, upperbound = 50,save=save)
    
    return rmat





def enrichment_comparison(name:str, gtdata:anndata.AnnData, semisdata:anndata.AnnData, celltype_key:str, selectedtype:str, save = None)-> Tuple[np.array,np.array,np.array,np.array]:
    """
    Compare the enrichment analysis results using the real-profiled and semi-profiled datasets. 
    
    Parameters
    ----------
    name:
        Project name
    gtdata:
        Real-profiled (ground truth) data
    semisdata:
        Semi-profiled dataset
    celltype_key:
        The key in anndata.AnnData.obs for storing the cell type information
    selectedtype:
        The selected cell type to analyze 
    save
        Path within the 'figures' folder to save the plot
    
    Returns
    -------
    CommonDEGs
        The number of overlapping DEGs between real and semi-profiled data 
    HypergeometricP
        P-value of hypergeometric test examining the overlap between two versions of DEGs
    PearsonR
        Pearson correlation between bar lengths in real-profiled and semi-profiled bar plots
    PearsonP
        P-value of the Pearson correlation test

    Example
    -------
    >>> _ = enrichment_comparison(name, gtdata, semisdata, celltype_key = 'celltypes', selectedtype = 'CD4')

    """
    
    totaltypes = np.unique(gtdata.obs[celltype_key])
    sc.tl.rank_genes_groups(gtdata, celltype_key, method='t-test')
    
    typededic = {}
    for j in range(totaltypes.shape[0]):
        celltype = totaltypes[j]
        typede = []
        for i in range(100):
            g = gtdata.uns['rank_genes_groups']['names'][i][j]
            typede.append(g)
        typededic[celltype] = typede


    sc.tl.rank_genes_groups(semisdata, celltype_key, method='t-test')
    semitypededic = {}
    for j in range(totaltypes.shape[0]):
        celltype = totaltypes[j]
        typede = []
        for i in range(100):
            g = semisdata.uns['rank_genes_groups']['names'][i][j]
            typede.append(g)
        semitypededic[celltype] = typede
    
    
    
    c=0
    gtdeg = typededic[selectedtype]
    semideg = semitypededic[selectedtype]
    for i in semideg:
        if i in gtdeg:
            c+=1
    
    hyperpval = hypert(semisdata.X.shape[1],100,100,c)
    
    print('p-value of hypergeometric test for overlapping DEGs:', str(float(hyperpval)))
    
    if (os.path.isdir(name + '/gseapygt')) == False:
        os.system('mkdir ' + name + '/gseapygt')
        
    from ....bulk._ora import enrichr
    results = enrichr(gene_list=gtdeg, gene_sets='GO_Biological_Process_2021',outdir=name + '/gseapygt')
    f=open(name + '/gseapygt/GO_Biological_Process_2021.human.enrichr.reports.txt','r')
    lines=f.readlines()
    f.close()

    gtsets=[]
    gtps=[]
    gtdic={}
    for l in lines[1:]:
        term = l.split('\t')[1]
        p = float(l.split('\t')[4])
        gtsets.append(term)
        gtps.append(p)
        gtdic[term] = p

    results = enrichr(gene_list=semideg, gene_sets='GO_Biological_Process_2021',outdir=name + '/gseapysemi')
    f=open(name + '/gseapysemi/GO_Biological_Process_2021.human.enrichr.reports.txt','r')
    lines=f.readlines()
    f.close()
    
    
    semisets=[]
    semips=[]
    semidic={}
    for l in lines[1:]:
        term = l.split('\t')[1]
        p = float(l.split('\t')[4])
        semisets.append(term)
        semips.append(p)
        semidic[term]=p
    terms = copy.deepcopy(gtsets[:10])
    real_data = copy.deepcopy(gtps[:10])
    sim_data = []
    for i in range(10):
        gtterm = semisets[i]
        if gtterm not in semidic.keys():
            sim_data.append(1)
        else:
            sim_data.append(semidic[gtterm])


    for i in range(10):
        if semisets[i] in terms:
            continue
        terms.append(semisets[i])
        sim_data.append(semips[i])
        if semisets[i] not in gtdic.keys():
            real_data.append(1)
        else:
            real_data.append(gtdic[semisets[i]])
    real_data = np.flip(real_data)
    sim_data = np.flip(sim_data)
    terms = np.flip(terms)
    sim_bar_lengths = [-np.log10(p) for p in sim_data]
    real_bar_lengths = [-np.log10(p) for p in real_data]
    
    

    res = scipy.stats.pearsonr(np.array(sim_bar_lengths),np.array(real_bar_lengths))
    print('Significance correlation:',res)

    
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 5))
    bar_width = 0.4
    y = np.arange(len(sim_data))+1
    ax1.barh(y, real_bar_lengths, height=bar_width, color='green', label='Real')
    ax1.set_xlabel('-log10(p)')
    ax1.set_ylabel('Term')
    ax1.set_title('Real Data ('+str(len(semideg))+' DEGs)')
    ax2.barh(y, sim_bar_lengths, height=bar_width, color='blue', label='Simulated')
    ax2.set_xlabel('-log10(p)')
    ax2.set_title('Semi-profiled Data('+str(len(gtdeg))+' DEGs)')
    max_val = max(max(sim_bar_lengths), max(real_bar_lengths))
    ax1.set_xlim(0,max_val + 1)
    ax2.set_xlim(0, max_val + 1)
    ax1.invert_xaxis()
    ax1.set_yticks(y)
    ax2.set_yticklabels(terms)
    fig.suptitle(selectedtype + ' GO ('+str(c)+' Overlap DEGs)')
    if save!= None:
        plt.savefig(name + '/figures/'+save+selectedtype + ' GO.pdf',bbox_inches='tight')
        plt.savefig(name + '/figures/'+save+selectedtype + ' GO.jpg',dpi=600,bbox_inches='tight')
    plt.show()

    return c, float(hyperpval), res[0], res[1]
    
    
    
    
def faiss_knn(query:np.array, x:np.array, n_neighbors:int=1) -> np.array:
    """
    Compute distances from a vector to its K-nearest neighbros in a matrix. 
    
    Parameters
    ----------
    query:
        The query vector
    X:
        The data matrix
    n_neighbors:
        How many neighbors to consider?
    
    Returns
    -------
    weights: distances

    Example
    -------
    >>> faiss_knn(ma,mb,n_neighbors=1)
    """
    
    
    n_samples = x.shape[0]
    n_features = x.shape[1]
    x = np.ascontiguousarray(x)
    
    index = faiss.IndexFlatL2(n_features)
    #index = faiss.IndexFlatIP(n_features)
                  
    index.add(x)
    
    if n_neighbors < 2:
        neighbors = 2
    else: 
        neighbors = n_neighbors
    
    weights, targets = index.search(query, neighbors)

    #sources = np.repeat(np.arange(n_samples), neighbors)
    #targets = targets.flatten()
    #weights = weights.flatten()
    weights = weights[:,:n_neighbors]
    if -1 in targets:
        raise InternalError("Not enough neighbors were found. Please consider "
                            "reducing the number of neighbors.")
    return weights



def get_error(name:str,scpath:str = 'example_data/scdata.h5ad')->Tuple[list,list,list,list]:
    """
    Conclude the semi-profiling history of a project and output the erros, upperbounds, and lower bounds, which are necessary for overall performance evaluation.

    Parameters
    ----------
    name:
        Project name
    scpath:
        Path to single-cell data

    Returns
    -------
    upperbounds
        The error upper bounds calculated in each round
    lowerbounds
        The error lower bounds calculated in each round
    semierrors
        The errors of semi-profiling
    naiveerrors
        The errors of the selection-only method 

    Example
    -------
    >>> upperbounds, lowerbounds, semierrors, naiveerrors = get_error(name)

    """
    
    
    
    # load gound truth
    print('loading and processing ground truth data.')
    st = timeit.default_timer()
    gtdata = anndata.read_h5ad(scpath)
    sids = []
    
    f = open(name + '/sids.txt','r')
    lines = f.readlines()
    for l in lines:
        sids.append(l.strip())
    f.close()
    
    gts = []
    for i in range(len(sids)):
        sid = sids[i]
        adata = gtdata[gtdata.obs['sample_ids'] == sid]
        x = adata.X
        if scipy.sparse.issparse(x):
            x = x.todense()
        x = np.array(x)
        if x.max()>20:
            x = np.log1p(x)
        gts.append(np.array(x))
    ed = timeit.default_timer()
    print('finished processing ground truth',str(ed-st),' seconds')
    
    # compute semi-profile error for each round
    upperbounds = []
    lowerbounds = []
    naiveerrors = []
    semierrors = []

    print('computing error for each round')
    for rnd in range(1,len(os.listdir(name+'/status'))//2+1):
        print('round ',str(rnd))
        # read representative and cluster labels info
        if rnd == 1:
            reprefile = name + '/status/init_representatives.txt'
            clusterfile = name + '/status/init_cluster_labels.txt'
        else:
            reprefile = name + '/status/eer_representatives_' + str(rnd) + '.txt'
            clusterfile = name + '/status/eer_cluster_labels_' + str(rnd) + '.txt'
        representatives = []
        f = open(reprefile,'r')
        lines = f.readlines()
        for l in lines:
            representatives.append(int(l.strip()))
        f.close()
        cluster_labels = []
        f = open(clusterfile,'r')
        lines = f.readlines()
        for l in lines:
            cluster_labels.append(int(l.strip()))
        f.close()
        
        print('loading semi-profiled cohort')
        st = timeit.default_timer()
        # semi-profiled cohort
        semis = []
        for i in range(len(sids)):
            if i in representatives:
                semis.append(gts[i])
            else:
                sid = sids[i]
                represid = sids[representatives[cluster_labels[i]]]
                xinf = np.load(name + '/inferreddata/' + represid + '_to_' + sid + '.npy')
                if xinf.max() > 20:
                    xinf = np.log1p(xinf)
                semis.append(xinf)
        ed = timeit.default_timer()
        print(str(ed-st),'for loading semi-profiled cohort.')
        
        
        print('pca')
        # pca 
        t_start = timeit.default_timer()
        X = np.concatenate([np.concatenate(gts,axis=0),np.concatenate(semis,axis=0)],axis=0)
        #X = np.log(X+1)
        reducer =  PCA(n_components = 100)#PCA(n_components = 100)#,svd_solver = 'randomized')#randomized_svd(n_components=100)  #PCA(n_components=100)#
        X_reduced = reducer.fit_transform(X)
        t_end = timeit.default_timer()
        print(str(t_end-t_start),'for pca')
        
        ### reduced data 
        xdimgts=[]
        xdimsemis=[]
        offset=0
        xused = X_reduced#X_UMAP # X_PCA
        for i in range(len(sids)):
            xdimgts.append(xused[offset:(offset+gts[i].shape[0]),:])
            offset = offset+gts[i].shape[0]
        lengt = offset
        for i in range(len(sids)):
            xdimsemis.append(xused[offset:(offset+semis[i].shape[0]),:])
            offset = offset+semis[i].shape[0]
        
        # error
        print('computing errors')
        st = timeit.default_timer()
        # lowerbound
        lbgt = copy.deepcopy(X_reduced)
        np.random.shuffle(lbgt)
        lbgt1 = lbgt[:lbgt.shape[0]//2,:]
        lbgt2 = lbgt[lbgt.shape[0]//2:,:]
        ma = np.array(lbgt1).copy(order='C')
        mb = np.array(lbgt2).copy(order='C')
        lowerbound = list(faiss_knn(ma,mb,n_neighbors=1)) + list(faiss_knn(mb,ma,n_neighbors=1))
        lowerbound = np.array(lowerbound)
        lowerbound = lowerbound.mean()
        lowerbounds.append(lowerbound)
        
        #upperbound
        ubscores = []
        for i in range(len(sids)):
            gt = xdimgts[i]
            randomidx = np.random.randint(0,len(sids))
            gtr = xdimgts[randomidx]
            ma = np.array(gt).copy(order='C')
            mb = np.array(gtr).copy(order='C')
            ubscore = list(faiss_knn(ma,mb,n_neighbors=1)) + list(faiss_knn(mb,ma,n_neighbors=1))
            ubscores.append(np.array(ubscore).mean())
        upperbound = np.array(ubscores).mean()
        upperbounds.append(upperbound)
        
        #semi error
        scores = []
        for i in range(len(sids)):
            pid = sids[i]
            if i in representatives:
                scores.append(lowerbound)
                continue
            gt = xdimgts[i]
            xs = xdimsemis[i]
            ma = np.array(gt).copy(order='C')
            mb = np.array(xs).copy(order='C')
            err1 = faiss_knn(ma,mb,n_neighbors=1)
            err2 = faiss_knn(mb,ma,n_neighbors=1)
            err = list(err1) + list(err2)
            err = (np.array(err)).mean()
            scores.append(err)
        semierror = np.array(scores).mean()
        semierrors.append(semierror)
        
        #naive error
        naivescores = []
        for i in range(len(sids)):
            pid = sids[i]
            if i in representatives:
                naivescores.append(lowerbound)
                continue
            gt = xdimgts[i]
            repre = representatives[cluster_labels[i]]
            xs = xdimgts[repre]
            ma = np.array(gt).copy(order='C')
            mb = np.array(xs).copy(order='C')
            err1 = faiss_knn(ma,mb,n_neighbors=1)
            err2 = faiss_knn(mb,ma,n_neighbors=1)
            err = list(err1) + list(err2)
            err = (np.array(err)).mean()
            naivescores.append(err)
        naiveerror = np.array(naivescores).mean()
        naiveerrors.append(naiveerror)
        ed = timeit.default_timer()
        print(str(ed-st),'for computing error.')
        
    return upperbounds, lowerbounds, semierrors, naiveerrors


def errorcurve(upperbounds:list, lowerbounds:list, semierrors:list, naiveerrors:list, batch:int=2,total_samples:int = 12) -> None:
    """
    Visualize the error and cost as more representatives are sequenced.

    Parameters
    ----------
    upperbounds
        The error upper bounds calculated in each round
    lowerbounds
        The error lower bounds calculated in each round
    semierrors
        The errors of semi-profiling
    naiveerrors
        The errors of the selection-only method 
    batch
        Representative selection batch size
    total_samples
        The total number of samples in the cohort

    Returns
    -------
        None

    Example
    -------
    >>> errorcurve(upperbounds, lowerbounds, semierrors, naiveerrors, batch=2,total_samples = 12)

    """
    
    ub = np.mean(upperbounds)
    lb = np.mean(lowerbounds)
    
    semi = (np.array(semierrors) - lb)/(ub-lb)
    naive = (np.array(naiveerrors) - lb)/(ub-lb)
    
    y1 = naive
    y2 = semi
    
    x = np.linspace(batch,batch*len(y1),len(y1))
    
    cost = []
    for i in range(len(y1)):
        bulkcost = 7000 + total_samples * 110
        sccost = 5000 * 0.3 * batch * i
        semicost = bulkcost + sccost
        cost.append(semicost/1000)
        
    
    # Create the main plot
    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, color='tab:blue', label='Selection-only')
    ax1.set_xlabel('Number of Samples Sequenced')
    ax1.set_ylabel('Normalized Error')
    #ax1.tick_params(axis='y', labelcolor='tab:blue')


    ax1.plot(x, y2, color='tab:orange', label='Semi-profiling')
    ax1.set_ylabel('Normalized Error')
    #ax1.tick_params(axis='y', labelcolor='tab:red')

    # Create a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(x, cost, color='tab:red', label='cost')
    ax2.set_ylabel('Cost (in k USD)')
    #ax2.tick_params(axis='y', labelcolor='tab:red')

    # Add legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center')

    plt.title('Error and Cost as More Representatives Sequenced')
    
    return 




def prop(gttypes ,semitypes, totaltypes):
    pgt = np.zeros(totaltypes.shape)
    psemi = np.zeros(totaltypes.shape)
    for i in range(len(totaltypes)):
        pgt[i] += (np.array(gttypes) == totaltypes[i]).sum()
        psemi[i] += (np.array(semitypes) == totaltypes[i]).sum()  ## celltypes2
    numgt=pgt
    numsemi=psemi
    pgt = pgt/pgt.sum()
    psemi = psemi/psemi.sum()
    pcor,pval = scipy.stats.pearsonr(pgt,psemi)
    return pcor,pval,pgt,psemi,numgt,numsemi

def RMSE(p1,p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    mse = ((p1-p2)**2).mean()
    rmse = mse**0.5
    return rmse

def CCCscore(y_pred, y_true, mode='all'):
    # adapated from Yanshuo et.al's work TAPE
    # pred: shape{n sample, m cell}
    if mode == 'all':
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
    elif mode == 'avg':
        pass
    ccc_value = 0
    for i in range(y_pred.shape[1]):
        r = np.corrcoef(y_pred[:, i], y_true[:, i])[0, 1]
        # Mean
        mean_true = np.mean(y_true[:, i])
        mean_pred = np.mean(y_pred[:, i])
        # Variance
        var_true = np.var(y_true[:, i])
        var_pred = np.var(y_pred[:, i])
        # Standard deviation
        sd_true = np.std(y_true[:, i])
        sd_pred = np.std(y_pred[:, i])
        # Calculate CCC
        numerator = 2 * r * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        ccc = numerator / denominator
        ccc_value += ccc
    return ccc_value / y_pred.shape[1]



def inspect_data(bulk:str,logged:bool=False,normed:bool = True, geneselection:Union[bool,int]=True, save=None) -> None:
    """
    To accommodate users who prefer all-in-one-batch profiling, we have developed hits new function. This function allows users to assess data heterogeneity (using Silhouette scores and bulk data visualization) and determine an approximate number of representative clusters needed for their dataset. Once the number of clusters is established, users can sequence all representatives in one batch to minimize potential batch effects. 

    Parameters
    ----------
    name
        Project name. 
    bulk
        Path to bulk data as an h5ad file. Sample IDs should be stored in adata.obs['sample_ids'] and gene names should be stored in adata.var.index. 
    logged
        Whether the data has been logged or not
    normed
        Whether the library size has been normalized or not
    geneselection
        Either a boolean value indicating whether to perform gene selection using the bulk data or not, or a integer specifying the number of highly variable genes should be selected.
    save
        Specify a folder for saving figures

    Returns
    -------
        None
    
    Example
    --------
    >>> import scSemiProfiler as semi
    >>> from scSemiProfiler.utils import *
    >>> bulk = 'example_data/bulkdata.h5ad'
    >>> logged = False
    >>> normed = True
    >>> geneselection = False
    >>> inspect_data(bulk,logged,normed,geneselection)

    """
    
    
    print('Presenting dataset heterogeneity information to decide sample batch size')
    
    if save != None:
        if (os.path.isdir(save)) == False:
            os.system('mkdir '+save)

        
    bulkdata = anndata.read_h5ad(bulk)
    
    
    if normed == False:
        if logged == True:
            print('Bad data preprocessing. Please normalize the library size before log-transformation.')
            return
        sc.pp.normalize_total(bulkdata, target_sum=1e4)
    
    if logged == False:
        sc.pp.log1p(bulkdata)
        
    # write sample ids
    sids = list(bulkdata.obs['sample_ids'])
    
    
    if geneselection == False:
        hvgenes = np.array(bulkdata.var.index)
    elif geneselection == True:
        sc.pp.highly_variable_genes(bulkdata, n_top_genes=6000)
        #sc.pp.highly_variable_genes(bulkdata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        bulkdata = bulkdata[:, bulkdata.var.highly_variable]
        hvgenes = (np.array(bulkdata.var.index))[bulkdata.var.highly_variable]
    else:
        sc.pp.highly_variable_genes(bulkdata, n_top_genes = int(geneselection))
        #sc.pp.highly_variable_genes(bulkdata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        bulkdata = bulkdata[:, bulkdata.var.highly_variable]
        hvgenes = (np.array(bulkdata.var.index))[bulkdata.var.highly_variable]

    #dim reduction and clustering
    if bulkdata.X.shape[0]>100:
        n_comps = 100
    else:
        n_comps = bulkdata.X.shape[0]-1
    
    sc.tl.pca(bulkdata,n_comps=n_comps)
    
    sc.pp.neighbors(bulkdata)
    sc.tl.umap(bulkdata)
    sc.pl.umap(bulkdata)
    
    avgs = []
    xaxis = []
    for i in range(2,len(sids)):
        n_clusters = i# max(2,batch*i + batch)
        xaxis.append(n_clusters)
        clusterer = KMeans(n_clusters=n_clusters, random_state=i)
        X = bulkdata.obsm['X_pca']
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        avgs.append(silhouette_avg)
    
    plt.xlabel('Number of Clusters')
    plt.plot(xaxis,avgs)
    plt.title('Average Silhoette Scores')
    return



def global_stop_checking(name:str, representatives:list, n_targets:int = 2, celltype_key:str='celltypes' ) -> Tuple[list,list]:
    """
    Checking if further representative selection is needed for the global mode. 
    n_target samples will be randomly selected from the representatives as the dummy target samples. 
    Other representatives will be used for infer the dummy target samples. A list of inferred samples and 
    the corresponding round truth will be returned for the user to examine the similarity and decide if further
    representative selection is needed. Information regarding cell type deconvolution performance will also be presented.
    
    Parameters
    ----------
    name
        Project name. 
    representatives
        List of representative samples.
    cluster_labels
        List of cluster labels.
    n_target
        The number of dummy targets.
    celltype_key
        The key in adata.obs for storing cell types
    Returns
    -------
    real_target_list
        List of real-profiled target ground truth
    inferred_target_list
        List of inferred target data for comparison
    
    Example
    --------
    >>> import scSemiProfiler as semi
    >>> from scSemiProfiler.utils import *
    >>> name = 'project_name'
    >>> representatives = [6, 10, 5, 7]
    >>> real_target_list, inferred_target_list = global_stop_checking(name = name, representatives =representatives, n_target = 2 )
    >>> # downstram analysis comparing real targets and inferred targets
    """
    
    sids = []
    f = open(name + '/sids.txt','r')
    lines = f.readlines()
    for l in lines:
        sids.append(l.strip())
    f.close()
    
    
    repsdata = []
    for i in range(len(representatives)):
        r = representatives[i]
        repsdata.append(anndata.read_h5ad(name + '/sample_sc/'+sids[r]+'.h5ad'))
    
    repspseudobulk = []
    for adata in repsdata:
        repspseudobulk.append( adata.X.mean(axis=0) )
    repspseudobulk = np.array(repspseudobulk)
    repspseudobulk = repspseudobulk.reshape((-1, adata.X.shape[1]))
    
    repspseudobulk = anndata.AnnData(repspseudobulk)
    if repspseudobulk.X.max()>20:
        sc.pp.log1p(repspseudobulk)
    sc.tl.pca(repspseudobulk)
    dismtx = repspseudobulk.obsm['X_pca']
    
    real_target_list = []
    inferred_target_list = []
    targets = random.sample(representatives,2)

    # find ref
    refs=[]
    for i in range(n_targets):
        t = dismtx[i]
        dists = ((dismtx - t)**2).sum(axis=1)
        sorted_indices = np.argsort(dists)
        index_second_smallest = sorted_indices[1]
        refs.append(representatives[index_second_smallest])
        real_target_list.append(repsdata[index_second_smallest])
    
    
    # infer
    for i in range(n_targets):
        tgt = targets[i]
        rep = refs[i]
        inferredtgt = tgtinfer(name = name, representative = rep, target = tgt, device = 'cuda:0') 
        
    for i in range(n_targets):
        tgt = targets[i]
        adata = anndata.read_h5ad(name + '/sample_sc/'+sids[tgt]+'.h5ad')
        inferred_target_list.append(adata)
    
    xtrain = []
    ytrain = []
    for i in range(len(real_target_list)):
        adata = real_target_list[i]
        xtrain.append(np.array(adata.X))
        sample_celltype = list(adata.obs[celltype_key])
        ytrain = ytrain + sample_celltype
    xtrain = np.concatenate(xtrain, axis=0)
    if xtrain.max()>10:
        xtrain = np.log1p(xtrain)
    ytrain = np.array(ytrain)
    totaltypes = np.unique(ytrain)
    
    annotator = MLPClassifier(hidden_layer_sizes=(200,)) 
    annotator.fit((xtrain),ytrain)

    print('Pearson correlation between cell type proportions in inferred and ground truth samples:')
    for i in range(n_targets):
        x = inferred_target_list[i].X
        if x.max()>20:
            x= np.log1p(x)
        predicted_celltype = annotator.predict(x)
        inferred_target_list[i].obs[celltype_key] = predicted_celltype
        realtypes = celltype_proportion(real_target_list[i],totaltypes)
        semitypes = celltype_proportion(inferred_target_list[i],totaltypes)
        print(scipy.stats.pearsonr(realtypes ,semitypes ))
    
    return real_target_list, inferred_target_list





def main():
    parser=argparse.ArgumentParser(description="scSemiProfiler assemble_cohort")
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    required.add_argument('--name',required=True, help="Project name.")
    
    required.add_argument('--representatives',required=True, help="Either a txt file including all representatives or a list of representatives.")
    required.add_argument('--cluster_labels',required=True, help="Either a txt file including cluster label information or a list of labels.")
    
    optional.add_argument('--celltype_key',required=False, default='celltypes', help="The key in the adata.obs indicating the cell type labels (Default: 'celltypes')") ###
    
    

    args = parser.parse_args()
    name = args.name
    representatives = args.representatives
    cluster_labels = args.cluster_labels
    celltype_key = args.celltype_key

    
    assemble_cohort(name,
                    representatives,
                    cluster_labels,
                    celltype_key)

if __name__=="__main__":
    main()

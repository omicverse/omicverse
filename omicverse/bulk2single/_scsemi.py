import anndata


import numpy as np
import os
from sklearn.cluster import KMeans
from typing import Union
import scipy
import scipy.sparse

class scSemiProfiler:
    def __init__(self,
                 name:str,
                 single_data:anndata.AnnData,
                 bulk_data:anndata.AnnData,
                 bulk_group:str=None,
                 ):
        self.name = name
        self.single_data = single_data
        self.bulk_data = bulk_data

    def init_bulk(self,
        #name:str, 
        logged:bool=False,
        normed:bool = True, 
        geneselection:Union[bool,int]=True,
        sample_id='sample_ids',
        batch:int=4) -> None:
        """
        Initial setup of the semi-profiling pipeline, including processing the bulk data, clustering for finding the initial representatives. Bulk data should be provided as an 'h5ad' file. Sample IDs should be stored in adata.obs['sample_ids'] and gene names should be stored in adata.var.index. If not using active learning for iterative representative selection, directly set the batch size to be the total number of representatives desired.
        
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
        batch 
            Representative selection batch size. 
        
        Returns
        -------
            None
        
        Example
        --------
        >>> import scSemiProfiler
        >>> name = 'runexample'
        >>> bulk = 'example_data/bulkdata.h5ad'
        >>> logged = False
        >>> normed = True
        >>> geneselection = False
        >>> batch = 2
        >>> scSemiProfiler.initsetup(name, bulk,logged,normed,geneselection,batch)

        """
        name = self.name
        print('Start initial setup')

        
        if (os.path.isdir(name)) == False:
            os.system('mkdir '+name)
        else:
            print(name + ' exists. Please choose another name.')
            return
        
        if (os.path.isdir(name+'/figures')) == False:
            os.system('mkdir '+name+'/figures')
        
        #bulkdata = anndata.read_h5ad(bulk)
        bulkdata = self.bulk_data
        
        if normed == False:
            if logged == True:
                print('Bad data preprocessing. Please normalize the library size before log-transformation.')
                return
            from ..pp._preprocess import normalize_total
            normalize_total(bulkdata, target_sum=1e4)
        
        if logged == False:
            from ..pp._preprocess import log1p
            log1p(bulkdata)
            
        # write sample ids
        sids = list(bulkdata.obs[sample_id])
        f = open(name+'/sids.txt','w')
        for sid in sids:
            f.write(sid+'\n')
        f.close()
        
        from ..pp import highly_variable_genes
        from .._sc_compat import pca_X
        if geneselection == False:
            hvgenes = np.array(bulkdata.var.index)
        elif geneselection == True:
            highly_variable_genes(bulkdata, n_top_genes=6000)
            bulkdata = bulkdata[:, bulkdata.var.highly_variable]
            hvgenes = (np.array(bulkdata.var.index))[bulkdata.var.highly_variable]
        else:
            highly_variable_genes(bulkdata, n_top_genes = int(geneselection))
            bulkdata = bulkdata[:, bulkdata.var.highly_variable]
            hvgenes = (np.array(bulkdata.var.index))[bulkdata.var.highly_variable]
        np.save(name+'/hvgenes.npy',hvgenes)
        
        #dim reduction and clustering
        
        if bulkdata.X.shape[0]>100:
            n_comps = 100
        else:
            n_comps = bulkdata.X.shape[0]-1
        
        pca_X(bulkdata,n_comps=n_comps)
        
        bulkdata.write(name + '/processed_bulkdata.h5ad')
        
        #cluster
        BATCH_SIZE = batch
        kmeans = KMeans(n_clusters=BATCH_SIZE, random_state=1).fit(bulkdata.obsm['X_pca'])
        cluster_labels = kmeans.labels_
        #find representatives and cluster labels
        pnums = []
        for i in range(len(bulkdata.X)):
            pnums.append(i)
        pnums=np.array(pnums)
        centers=[]
        representatives=[]
        repredic={}
        for i in range(len(np.unique(cluster_labels))):
            mask = (cluster_labels==i)
            cluster = bulkdata.obsm['X_pca'][mask]
            cluster_patients = pnums[mask]
            center = cluster.mean(axis=0)
            centers.append(center)
            # find the closest patient
            sqdist = ((cluster - center)**2).sum(axis=1)
            cluster_representative = cluster_patients[np.argmin(sqdist)]
            representatives.append(cluster_representative)
            repredic[i] = cluster_representative
        centers = np.array(centers)
        #store representatives cluster labels
        if (os.path.isdir(name + '/status')) == False:
            os.system('mkdir ' + name + '/status')
        

        f=open(name + '/status/init_cluster_labels.txt','w')
        for i in range(len(cluster_labels)):
            f.write(str(cluster_labels[i])+'\n')
        f.close()

        f=open(name + '/status/init_representatives.txt','w')
        for i in range(len(representatives)):
            f.write(str(representatives[i])+'\n')
        f.close()
        
        print('Initial setup finished. Among ' + str(len(sids)) + ' total samples, selected '+str(batch)+' representatives:')
        for i in range(batch):
            print(sids[representatives[i]])
        
        return

    def init_single(self,
        #name:str,
        #singlecell:str, 
        logged:bool = False, normed:bool = True, 
        cellfilter:bool = False, threshold:float = 1e-3, 
        geneset:Union[bool,str] = True, 
        sample_id='sample_ids',
        weight:float=0.5, k:int=15) -> None:
        """
        Process the reprsentatives' single-cell data, including preprocessing and feature augmentations. 

        Parameters
        ----------
        name
            Project name.
        singlecell
            Path to representatives' single-cell data.
        logged
            Whether the data has been logged or not
        normed
            Whether the library size has been normalized or not
        cellfilter
            Whether to perform standard cell filtering.
        threshold
            Threshold for background noise removal.
        geneset
            Whether to use gene set to augment gene expression features or no.
        weight
            The proportion of top features to increase importance weight.
        k
            K for the K-NN graph built for cells.
            
        Returns
        -------
            None
            
        Example
        -------
        >>> scSemiProfiler.scprocess(name = 'project_name', singlecell = name+'/representative_sc.h5ad', logged = False, normed = True, cellfilter = False, threshold=1e-3, geneset=True, weight = 0.5, k = 15)
        
        
        """
        from ..external.bulk2single.scSemiProfiler.singlecell_process import hamster_to_human,getGeneSetMatrix,fast_cellgraph
        from .. import pp as _pp
        print('Processing representative single-cell data')
        
        #scdata = anndata.read_h5ad(singlecell)
        name = self.name
        scdata = self.single_data
        sids = np.unique(scdata.obs[sample_id])

        
        
        # cell filtering
        if cellfilter == True:
            print('Filtering cells')
            _pp.filter_cells(scdata, min_genes=200)
            scdata.var['mt'] = scdata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
            _pp.calculate_qc_metrics(scdata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
            scdata = scdata[scdata.obs.n_genes_by_counts < 2500, :]
            scdata = scdata[scdata.obs.pct_counts_mt < 5, :]
        
        if logged == True:
            print('recovering log-transformed data to count data')
            adata = scdata
            bdata = anndata.AnnData((np.exp(adata.X)-1))
            bdata.obs = adata.obs
            bdata.var = adata.var
            bdata.obsm = adata.obsm
            bdata.uns = adata.uns
            scdata = bdata
        
        if normed == False:
            print('Library size normalization.')
            _pp.normalize_total(scdata, target_sum=1e4)
        
        # convert to dense if sparse
        if scipy.sparse.issparse(scdata.X):
            X = np.array(scdata.X.todense())
            tempdata = anndata.AnnData(X)
            tempdata.obs = scdata.obs
            tempdata.var = scdata.var
            scdata = tempdata
        
        
        # norm remove noise
        if float(threshold) > 0:
            print('Removing background noise')
            X = np.array(scdata.X)
            cutoff = 1e4*threshold
            X = X * np.array(X>cutoff)
            nscdata = anndata.AnnData(X)
            nscdata.obs = scdata.obs
            nscdata.obsm = scdata.obsm
            nscdata.var = scdata.var
            nscdata.uns = scdata.uns
            scdata = nscdata
        
        
        
        # store singlecell data, geneset score
        if (os.path.isdir(name + '/sample_sc')) == False:
            os.system('mkdir ' + name + '/sample_sc')

        
        
        
        
        if geneset != False:
            if (os.path.isdir(name + '/geneset_scores')) == False:
                os.system('mkdir ' + name + '/geneset_scores')
            
            prior_name = "c2.cp.v7.4.symbols.gmt" 
            if (geneset == True) or (geneset == 'human'):
                print('Computing human geneset scores')
            elif geneset == 'hamster':
                print('Computing hamster geneset scores')
            zps=[]
            for sid in sids:
                adata = scdata[scdata.obs[sample_id] == sid]
                X = adata.X

                gene_sets_path = "genesets/"
                genes = list(adata.var.index)
                
                if geneset == 'hamster':
                    genes = hamster_to_human(genes)
                genes_upper = [g.upper() for g in genes]
                N = adata.X.shape[0]
                G = len(genes_upper)
                gene_set_matrix, keys_all = getGeneSetMatrix(prior_name, genes_upper, gene_sets_path)

                zp = X.dot(np.array(gene_set_matrix).T)
                eps = 1e-6
                den = (np.array(gene_set_matrix.sum(axis=1))+eps)
                zp = (zp+eps)/den
                zp = zp - eps/den
                np.save(name + '/geneset_scores/' + sid,zp)
                zps.append(zp)

            if 'hvset.npy' not in os.listdir(name):
                zps=np.concatenate(zps,axis=0)
                zdata = anndata.AnnData(zps)
                _pp.log1p(zdata)
                _pp.highly_variable_genes(zdata)
                hvset = zdata.var.highly_variable
                np.save(name + '/hvset.npy',hvset)
            
            # select highly variable genes (genes in preprocessed bulk data)
            hvgenes = np.load(name + '/hvgenes.npy', allow_pickle = True)
            
            for g in hvgenes:
                if g not in scdata.var.index:
                    print('Error. Bulk data contains genes that are not in single-cell data. Please remove those genes from the bulk data and try again.')
                    return
            
            hvmask = []
            for i in scdata.var.index:
                if i in hvgenes:
                    hvmask.append(True)
                else:
                    hvmask.append(False)
            hvmask = np.array(hvmask)
            scdata = scdata[:,hvmask]
            np.save(name + '/hvmask.npy',hvmask)
        

        print('Augmenting and saving single-cell data.')
        for sid in sids:
            adata = scdata[scdata.obs[sample_id] == sid]
            
            # gcn
            adata.obs['cellidx']=range(len(adata.obs))
            adata,adj = fast_cellgraph(adata,k=k,diagw=1.0)
            
            
            if geneset == True:
                # # importance weight
                sample_geneset = np.load(name + '/geneset_scores/'+sid+'.npy')
                setmask = np.load(name + '/hvset.npy')
                sample_geneset = sample_geneset[:,setmask]
                sample_geneset = sample_geneset.astype('float32')

                features = np.concatenate([adata.X,sample_geneset],1)
            else:
                features = adata.X
                
            variances = np.var(features,axis=0)
            adata.uns['feature_var'] = variances
            
            adata.write(name + '/sample_sc/' + sid + '.h5ad')
        
        print('Finished processing representative single-cell data')
        return 

    def infer(
        self,
        representatives:str,cluster:str,bulktype:str='pseudobulk',
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
        device:str = 'cuda:0'
    ):
        from ..external.bulk2single.scSemiProfiler.inference import scinfer
        scinfer(
            self.name, representatives,cluster,
            bulktype,lambdad,pretrain1batch,pretrain1lr,
            pretrain1vae,pretrain1gan,lambdabulkr,pretrain2lr,
            pretrain2vae,pretrain2gan,inferepochs,lambdabulkt,
            inferlr,pseudocount,ministages,k,device
        )

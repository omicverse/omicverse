import argparse, os, sys
from os import listdir
from os.path import isfile, join
from ipywidgets import widgets

import pickle
import math
import collections
import time

import numpy as np
import pandas as pd
import warnings
from .._registry import register_function
warnings.filterwarnings('ignore')
from sklearn.metrics import silhouette_score
import multiprocess as mp
from functools import partial
import seaborn as sns
from matplotlib import pyplot as plt
from .._settings import add_reference


import time
@register_function(
    aliases=['自动分辨率选择', 'autoResolution', 'optimal leiden resolution'],
    category="single",
    description="Search clustering resolutions and select a stable/optimal Leiden resolution for biologically coherent cluster granularity.",
    prerequisites={'functions': ['pp.neighbors']},
    requires={'obsp': ['connectivities'], 'uns': ['neighbors']},
    produces={'obs': ['leiden'], 'uns': ['resolution_search']},
    auto_fix='auto',
    examples=['ov.single.autoResolution(adata, cpus=8)'],
    related=['pp.leiden', 'utils.cluster']
)
def autoResolution(adata,cpus=4):
    r"""Automatically determine clustering resolution.

    Parameters
    ----------
    adata:anndata.AnnData
        Input single-cell AnnData for louvain resolution search.
    cpus:int
        Number of worker processes used for subsampling evaluations.
    
    Returns
    -------
    Tuple[anndata.AnnData,float,pd.DataFrame]
        Updated AnnData, selected resolution, and silhouette-score table.
    """
    print("Automatically determine clustering resolution...")
    from ..pp import louvain
    start = time.time()
    def subsample_clustering(adata, sample_n, subsample_n, resolution, subsample):
        subadata = adata[subsample]
        louvain(subadata, resolution=resolution)
        cluster = subadata.obs['louvain'].tolist()
        
        subsampling_n = np.zeros((sample_n, sample_n), dtype=bool)
        coclustering_n = np.zeros((sample_n, sample_n), dtype=bool)
        
        for i in range(subsample_n):
            for j in range(subsample_n):
                x = subsample[i]
                y = subsample[j]
                subsampling_n[x][y] = True
                if cluster[i] == cluster[j]:
                    coclustering_n[x][y] = True
        return (subsampling_n, coclustering_n)
    rep_n = 5
    subset = 0.8
    sample_n = len(adata.obs)
    subsample_n = int(sample_n * subset)
    resolutions = np.linspace(0.4, 1.4, 6)
    silhouette_avg = {}
    np.random.seed(1)
    best_resolution = 0
    highest_sil = 0
    for r in resolutions:
        r = np.round(r, 1)
        print("Clustering test: resolution = ", r)
        sub_start = time.time()
        subsamples = [np.random.choice(sample_n, subsample_n, replace=False) for t in range(rep_n)]
        p = mp.Pool(cpus)
        func = partial(subsample_clustering, adata, sample_n, subsample_n, r)
        resultList = p.map(func, subsamples)
        p.close()
        p.join()
        
        subsampling_n = sum([result[0] for result in resultList])
        coclustering_n = sum([result[1] for result in resultList])
        
        subsampling_n[np.where(subsampling_n == 0)] = 1e6
        distance = 1.0 - coclustering_n / subsampling_n
        np.fill_diagonal(distance, 0.0)
        
        louvain(adata, resolution=r, key_added = 'louvain_r' + str(r))
        silhouette_avg[str(r)] = silhouette_score(distance, adata.obs['louvain_r' + str(r)], metric="precomputed")
        if silhouette_avg[str(r)] > highest_sil:
            highest_sil = silhouette_avg[str(r)]
            best_resolution = r
        print("robustness score = ", silhouette_avg[str(r)])
        sub_end = time.time()
        print('time: {}', sub_end - sub_start)
        print()
    adata.obs['louvain'] = adata.obs['louvain_r' + str(best_resolution)]
    print("resolution with highest score: ", best_resolution)
    res = best_resolution
    # write silhouette record to uns and remove the clustering results except for the one with the best resolution
    adata.uns['sihouette score'] = silhouette_avg
    # draw lineplot
    df_sil = pd.DataFrame(silhouette_avg.values(), columns=['silhouette score'], index=[float(x) for x in silhouette_avg.keys()])
    df_sil.plot.line(style='.-', color='green', title='Auto Resolution', xticks=resolutions, xlabel='resolution', ylabel='silhouette score', legend=False)
    #pp.savefig()
    #plt.close()
    end = time.time()
    print('time: {}', end-start)
    return adata, res, df_sil

def writeGEP(adata_GEP,path):
    r"""Write the gene expression profile to a file.

    Parameters
    ----------
    adata_GEP:anndata.AnnData
        AnnData containing expression matrix and ``obs['louvain']`` labels.
    path:str
        Output directory path for ``GEP.txt``.
    
    Returns
    -------
        None

    """
    print('Exporting GEP...')
    from ..pp import normalize_total
    normalize_total(adata_GEP, target_sum=1e6)
    mat = adata_GEP.X.transpose()
    if type(mat) is not np.ndarray:
        mat = mat.toarray()
    GEP_df = pd.DataFrame(mat, index=adata_GEP.var.index)
    GEP_df.columns = adata_GEP.obs['louvain'].tolist()
    # GEP_df = GEP_df.loc[adata.var.index[adata.var.highly_variable==True]]
    GEP_df.dropna(axis=1, inplace=True)
    GEP_df.to_csv(os.path.join(path, 'GEP.txt'), sep='\t')
    
@register_function(
    aliases=['药物响应预测器', 'Drug_Response', 'single-cell drug response'],
    category="single",
    description="Predict drug sensitivity from single-cell transcriptomes using CaDRReS and pharmacogenomic reference resources.",
    prerequisites={'optional_functions': ['utils.download_CaDRReS_model', 'utils.download_GDSC_data']},
    requires={'var': ['gene symbols']},
    produces={'obs': ['drug response scores'], 'uns': ['drug response ranking']},
    auto_fix='escalate',
    examples=['job = ov.single.Drug_Response(adata, scriptpath="CaDRReS-Sc")', 'res = job.main()'],
    related=['utils.download_CaDRReS_model', 'utils.download_GDSC_data']
)
class Drug_Response:
    """
    Predict drug sensitivity from single-cell transcriptomes using CaDRReS models.

    Parameters
    ----------
    adata:AnnData
        Query single-cell AnnData.
    scriptpath:str
        Path to CaDRReS-Sc scripts.
    modelpath:str
        Path to pretrained pharmacogenomic model/data resources.
    output:str, optional
        Output directory for prediction tables and plots.
    model:{'GDSC', 'PRISM'}, optional
        Pharmacogenomic reference model.
    clusters:str, optional
        Cluster subset to analyze (``'All'`` uses all cells).
    cell:str, optional
        Cell-line context used by the model.
    cpus:int, optional
        CPU threads used by downstream steps.
    n_drugs:int, optional
        Number of top drugs to report/plot.

    Returns
    -------
    None
        Initializes drug-response prediction workflow state.
    
    Examples
    --------
    >>> job = ov.single.Drug_Response(adata, scriptpath="CaDRReS-Sc")
    """
    def __init__(self,adata,scriptpath,modelpath,output='./',model='GDSC',clusters='All',
                 cell='A549',cpus=4,n_drugs=10):
        
        r"""Initialize the Drug_Response class.

        Parameters
        ----------
        adata:anndata.AnnData
            Input AnnData used for single-cell drug-response prediction.
        scriptpath:str
            Path to cloned ``CaDRReS-Sc`` script directory.
        modelpath:str
            Path containing pretrained CaDRReS model/data files.
        output:str
            Output directory for prediction tables and figures.
        model:str
            Pharmacogenomic reference model, typically ``'GDSC'`` or ``'PRISM'``.
        clusters:str
            Comma-separated louvain cluster IDs, or ``'All'``.
        cell:str
            Cell-line context label used by CaDRReS.
        cpus:int
            Number of CPUs used by downstream routines.
        n_drugs:int
            Number of top drugs displayed in output figures.

        Returns
        -------
            None
        """

        self.model = model
        self.adata=adata
        self.clusters=clusters
        self.output=output
        self.n_drugs=n_drugs
        self.modelpath=modelpath

        self.scriptpath = scriptpath
        sys.path.append(os.path.abspath(scriptpath))

        from cadrres_sc import pp, model, evaluation, utility
        
        self.load_model()
        self.drug_info()
        self.bulk_exp()
        self.sc_exp()
        self.kernel_feature_preparartion()
        self.sensitivity_prediction()
        if self.model == 'GDSC':
            self.masked_drugs = list(pd.read_csv(self.modelpath+'masked_drugs.csv')['GDSC'].dropna().astype('int64').astype('str'))
            self.cell_death_proportion()
        else:
            self.masked_drugs = list(pd.read_csv(self.modelpath+'masked_drugs.csv')['PRISM'])
        self.output_result()
        self.figure_output()

    def load_model(self):
        r"""Load the pre-trained model.

        Returns
        -------
            None

        """
        from cadrres_sc import pp, model, evaluation, utility
        ### IC50/AUC prediction
        ## Read pre-trained model
        #model_dir = '/Users/fernandozeng/Desktop/analysis/scDrug/CaDRReS-Sc-model/'
        model_dir = self.modelpath
        obj_function = widgets.Dropdown(options=['cadrres-wo-sample-bias', 'cadrres-wo-sample-bias-weight'], description='Objetice function')
        self.model_spec_name = obj_function.value
        if self.model == 'GDSC':
            model_file = model_dir + '{}_param_dict_all_genes.pickle'.format(self.model_spec_name)
        elif self.model == 'PRISM':
            model_file = model_dir + '{}_param_dict_prism.pickle'.format(self.model_spec_name)
        else:
            sys.exit('Wrong model name.')
        self.cadrres_model = model.load_model(model_file)

    def drug_info(self):
        r"""
        read the drug information.

        """
        ## Read drug information
        if self.model == 'GDSC':
            self.drug_info_df = pd.read_csv(self.scriptpath + '/preprocessed_data/GDSC/drug_stat.csv', index_col=0)
            self.drug_info_df.index = self.drug_info_df.index.astype(str)
        else:
            self.drug_info_df = pd.read_csv(self.scriptpath + '/preprocessed_data/PRISM/PRISM_drug_info.csv', index_col='broad_id')
        
    def bulk_exp(self):
        r"""
        extract the bulk gene expression data.

        """
        ## Read test data
        if self.model == 'GDSC':
            #GDSC_exp exists in the data folder
            files=os.listdir(self.scriptpath + '/data/GDSC')
            if 'GDSC_exp.tsv' not in files:
                self.gene_exp_df = pd.read_csv(self.modelpath + 'GDSC_exp.tsv.gz', sep='\t', index_col=0)
                self.gene_exp_df = self.gene_exp_df.groupby(self.gene_exp_df.index).mean()
            else:
                self.gene_exp_df = pd.read_csv(self.scriptpath + '/data/GDSC/GDSC_exp.tsv', sep='\t', index_col=0)
                self.gene_exp_df = self.gene_exp_df.groupby(self.gene_exp_df.index).mean()
        else:
            self.gene_exp_df = pd.read_csv(self.scriptpath + '/data/CCLE/CCLE_expression.csv', low_memory=False, index_col=0).T
            self.gene_exp_df.index = [gene.split(sep=' (')[0] for gene in self.gene_exp_df.index]

    def sc_exp(self):
        r"""
        Load cluster-specific gene expression profile
        """
        ## Load cluster-specific gene expression profile
        if self.clusters == 'All':
            clusters = sorted(self.adata.obs['louvain'].unique(), key=int)
        else:
            clusters = [x.strip() for x in self.clusters.split(',')]

        self.cluster_norm_exp_df = pd.DataFrame(columns=clusters, index=self.adata.raw.var.index)
        for cluster in clusters:
            self.cluster_norm_exp_df[cluster] =  self.adata.raw.X[self.adata.obs['louvain']==cluster].mean(axis=0).T \
                                                 if np.sum(self.adata.raw.X[self.adata.obs['louvain']==cluster]) else 0.0

    def kernel_feature_preparartion(self):
        r"""
        kernel feature preparation

        """
        from cadrres_sc import pp, model, evaluation, utility
        ## Read essential genes list
        if self.model == 'GDSC':
            ess_gene_list = self.gene_exp_df.index.dropna().tolist()
        else:
            ess_gene_list = utility.get_gene_list(self.scriptpath + '/preprocessed_data/PRISM/feature_genes.txt')

        ## Calculate fold-change
        cell_line_log2_mean_fc_exp_df, cell_line_mean_exp_df = pp.gexp.normalize_log2_mean_fc(self.gene_exp_df)
            
        self.adata_exp_mean = pd.Series(self.adata.raw.X.mean(axis=0).tolist()[0], index=self.adata.raw.var.index)
        cluster_norm_exp_df = self.cluster_norm_exp_df.sub(self.adata_exp_mean, axis=0)

        ## Calculate kernel feature
        self.test_kernel_df = pp.gexp.calculate_kernel_feature(cluster_norm_exp_df, cell_line_log2_mean_fc_exp_df, ess_gene_list)
    
    def sensitivity_prediction(self):
        r"""
        Predict drug sensitivity
        
        """
        from cadrres_sc import pp, model, evaluation, utility
        ## Drug response prediction
        if self.model == 'GDSC':
            print('...Predicting drug response for using CaDRReS(GDSC): {}'.format(self.model_spec_name))
            self.pred_ic50_df, P_test_df= model.predict_from_model(self.cadrres_model, self.test_kernel_df, self.model_spec_name)
            print('...done!')
        else:
            print('...Predicting drug response for using CaDRReS(PRISM): {}'.format(self.model_spec_name))
            self.pred_auc_df, P_test_df= model.predict_from_model(self.cadrres_model, self.test_kernel_df, self.model_spec_name)
            print('...done!')
        add_reference(self.adata,'scDrug','drug response prediction with CaDRReS')

    def cell_death_proportion(self):
        r"""
        Predict cell death proportion and cell death percentage at the ref_type dosage

        """
        ### Drug kill prediction
        ref_type = 'log2_median_ic50'
        self.drug_list = [x for x in self.pred_ic50_df.columns if not x in self.masked_drugs]
        self.drug_info_df = self.drug_info_df.loc[self.drug_list]
        self.pred_ic50_df = self.pred_ic50_df.loc[:,self.drug_list]

        ## Predict cell death percentage at the ref_type dosage
        pred_delta_df = pd.DataFrame(self.pred_ic50_df.values - self.drug_info_df[ref_type].values, columns=self.pred_ic50_df.columns)
        pred_cv_df = 100 / (1 + (np.power(2, -pred_delta_df)))
        self.pred_kill_df = 100 - pred_cv_df
    
    def output_result(self):
        """
        Export predicted drug response tables to CSV files.

        Returns
        -------
        None
            Writes normalized prediction files to ``self.output``:
            ``IC50_prediction.csv``/``drug_kill_prediction.csv`` for GDSC, or
            ``PRISM_prediction.csv`` for PRISM.

        Examples
        --------
        >>> dr.sensitivity_prediction()
        >>> dr.cell_death_proportion()
        >>> dr.output_result()
        """
        if self.model == 'GDSC':
            drug_df = pd.DataFrame({'Drug ID': self.drug_list, 
                                    'Drug Name': [self.drug_info_df.loc[drug_id]['Drug Name'] for drug_id in self.drug_list]})
            self.pred_ic50_df = (self.pred_ic50_df.T-self.pred_ic50_df.min(axis=1))/(self.pred_ic50_df.max(axis=1)-self.pred_ic50_df.min(axis=1))
            self.pred_ic50_df = self.pred_ic50_df.T
            self.pred_ic50_df.columns = pd.MultiIndex.from_frame(drug_df)
            self.pred_ic50_df.round(3).to_csv(os.path.join(self.output, 'IC50_prediction.csv'))
            self.pred_kill_df.columns = pd.MultiIndex.from_frame(drug_df)
            self.pred_kill_df.round(3).to_csv(os.path.join(self.output, 'drug_kill_prediction.csv'))
        else:
            drug_list = list(self.pred_auc_df.columns)
            drug_list  = [d for d in drug_list if d not in self.masked_drugs]
            drug_df = pd.DataFrame({'Drug ID':drug_list,
                                    'Drug Name':[self.drug_info_df.loc[d, 'name'] for d in drug_list]})
            self.pred_auc_df = self.pred_auc_df.loc[:,drug_list].T
            self.pred_auc_df = (self.pred_auc_df-self.pred_auc_df.min())/(self.pred_auc_df.max()-self.pred_auc_df.min())
            self.pred_auc_df = self.pred_auc_df.T
            self.pred_auc_df.columns = pd.MultiIndex.from_frame(drug_df)
            self.pred_auc_df.round(3).to_csv(os.path.join(self.output, 'PRISM_prediction.csv'))
    
    def draw_plot(self, df, n_drug=10, name='', figsize=()):
        r"""
        plot heatmap of drug response prediction

        Parameters
        ----------
        df:pd.DataFrame
            Drug-response matrix to visualize.
        n_drug:int
            Number of top drugs shown in heatmap.
        name:str
            Output figure filename prefix.
        figsize:tuple
            Figure size for heatmap.
        """
        def select_drug(df, n_drug):
            selected_drugs = []
            df_tmp = df.reset_index().set_index('Drug Name').iloc[:, 1:]
            for cluster in sorted([x for x in df_tmp.columns], key=int):
                for drug_name in df_tmp.sort_values(by=cluster, ascending=False).index[:n_drug].values:
                    if drug_name not in selected_drugs:
                        selected_drugs.append(drug_name)
            df_tmp = df_tmp.loc[selected_drugs, :]
            return df_tmp

        if self.model == 'GDSC':
            fig, ax = plt.subplots(figsize=figsize) 
            sns.heatmap(df.iloc[:n_drug,:-1], cmap='Blues', \
                        linewidths=0.5, linecolor='lightgrey', cbar=True, cbar_kws={'shrink': .2, 'label': 'Drug Sensitivity'}, ax=ax)
            ax.set_xlabel('Cluster', fontsize=20)
            ax.set_ylabel('Drug', fontsize=20)
            ax.figure.axes[-1].yaxis.label.set_size(20)
            for _, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_color('lightgrey') 
            plt.savefig(os.path.join(self.output, '{}.png'.format(name)), bbox_inches='tight', dpi=200)
            plt.close()

        else:
            fig, ax = plt.subplots(figsize=(df.shape[1], int(n_drug*df.shape[1]/5))) 
            sns.heatmap(select_drug(df, n_drug), cmap='Reds', \
                        linewidths=0.5, linecolor='lightgrey', cbar=True, cbar_kws={'shrink': .2, 'label': 'Drug Sensitivity'}, ax=ax, vmin=0, vmax=1)
            ax.set_xlabel('Cluster', fontsize=20)
            ax.set_ylabel('Drug', fontsize=20)
            ax.figure.axes[-1].yaxis.label.set_size(20)
            for _, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_color('lightgrey') 
            plt.savefig(os.path.join(self.output, '{}.png'.format(name)), bbox_inches='tight', dpi=200)
            plt.close()

    def figure_output(self):
        r"""
        plot figures

        """
        print('...Ploting figures...')
        ## GDSC figures
        if self.model == 'GDSC':
            tmp_pred_ic50_df = self.pred_ic50_df.T
            tmp_pred_ic50_df = tmp_pred_ic50_df.assign(sum=tmp_pred_ic50_df.sum(axis=1)).sort_values(by='sum', ascending=True)
            self.draw_plot(tmp_pred_ic50_df, name='GDSC prediction', figsize=(12,40))
            tmp_pred_kill_df = self.pred_kill_df.T
            tmp_pred_kill_df = tmp_pred_kill_df.loc[(tmp_pred_kill_df>=50).all(axis=1)]
            tmp_pred_kill_df = tmp_pred_kill_df.assign(sum=tmp_pred_kill_df.sum(axis=1)).sort_values(by='sum', ascending=False)
            self.draw_plot(tmp_pred_kill_df, n_drug=10, name='predicted cell death', figsize=(12,8))

        ## PRISM figures
        else:
            tmp_pred_auc_df = self.pred_auc_df.T
            #tmp_pred_auc_df = tmp_pred_auc_df.assign(sum=tmp_pred_auc_df.sum(axis=1)).sort_values(by='sum', ascending=True)
            self.draw_plot(tmp_pred_auc_df, n_drug=self.n_drugs, name='PRISM prediction')  
        print('done!')  

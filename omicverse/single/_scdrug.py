import os
from typing import Optional, Sequence

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from ipywidgets import widgets

from .._registry import register_function
from .._settings import Colors, EMOJI, add_reference
from ..report._provenance import tracked, note


@register_function(
    aliases=['自动分辨率选择', 'autoResolution', 'optimal leiden resolution'],
    category="single",
    description=(
        "Pick the most stable Leiden resolution by subsample-ARI "
        "stability: for each candidate resolution, recluster N "
        "subsamples and score the mean Adjusted Rand Index against the "
        "reference (full-data) labels. The resolution with the highest "
        "mean ARI subject to a minimum cluster count is selected."
    ),
    prerequisites={'functions': ['pp.neighbors']},
    requires={'obsp': ['connectivities'], 'uns': ['neighbors']},
    produces={'obs': ['leiden'], 'uns': ['autoResolution']},
    auto_fix='auto',
    examples=[
        'res, df = ov.single.autoResolution(adata)',
        'ov.single.autoResolution(adata, resolutions=np.arange(0.2, 2.0, 0.1))',
    ],
    related=['pp.leiden'],
)
@tracked('autoResolution', 'ov.single.autoResolution')
def autoResolution(
    adata: anndata.AnnData,
    resolutions: Optional[Sequence[float]] = None,
    *,
    n_subsamples: int = 5,
    subsample_frac: float = 0.8,
    min_clusters: int = 3,
    key_added: str = 'leiden',
    random_state: int = 0,
    verbose: bool = True,
):
    r"""Pick the most stable Leiden resolution via subsample-bootstrap ARI.

    Algorithm
    ---------
    For each candidate resolution :math:`r`:

    1. Run Leiden on the **full** AnnData → reference labels at ``r``.
       These will be the labels the user actually keeps if ``r`` wins.
    2. Take :paramref:`n_subsamples` random subsamples without
       replacement, each of size ``subsample_frac × n_obs``.
    3. For each subsample, run Leiden on the induced subgraph
       (``adata.obsp['connectivities']`` is sliced automatically).
    4. Compute Adjusted Rand Index (ARI) between the reference labels
       restricted to the subsample and the bootstrap's labels.
    5. The stability score for ``r`` is the **mean ARI** across the
       :paramref:`n_subsamples` bootstraps.

    The resolution with the highest mean ARI is chosen, **subject to**
    producing at least :paramref:`min_clusters` clusters on the full
    data — otherwise a degenerate "everything in one cluster" trivially
    wins on stability.

    Why mean-ARI bootstrap instead of silhouette on a co-clustering
    distance matrix:

    - Silhouette on a precomputed :math:`n \times n` co-clustering
      distance is :math:`O(n^2)` memory and conceptually double-dips —
      the labels you score and the distance you score against are both
      derived from the same clustering procedure on the same data.
    - Mean ARI to a reference clustering is :math:`O(n)` per bootstrap,
      well-defined for any partition pair, and directly answers the
      question we actually care about: *would small perturbations of
      the data produce roughly the same clusters?*

    Parameters
    ----------
    adata
        AnnData with a precomputed neighbor graph
        (``adata.obsp['connectivities']``).
    resolutions
        Candidate resolutions to test. Defaults to
        ``np.round(np.arange(0.2, 1.6, 0.1), 2)``.
    n_subsamples
        Number of bootstrap subsamples per resolution. 5 is usually
        enough; 10–20 trades runtime for tighter ARI estimates.
    subsample_frac
        Fraction of cells in each bootstrap (``0 < f < 1``). Default 0.8.
    min_clusters
        Lower bound on the number of clusters the chosen resolution
        must produce on the full data. Resolutions producing fewer
        clusters are excluded from the argmax to avoid the
        "trivially-stable single cluster" pitfall.
    key_added
        ``adata.obs`` column to write the chosen resolution's labels to.
    random_state
        Seed for both subsample selection and Leiden RNG.
    verbose
        Print per-resolution scores as the search progresses.

    Returns
    -------
    Tuple[anndata.AnnData, float, pandas.DataFrame]
        ``(adata, best_resolution, scores_df)`` where ``scores_df`` is
        indexed by resolution with columns ``mean_ari``, ``std_ari``,
        ``n_clusters``. Also writes ``adata.obs[key_added]`` and
        ``adata.uns['autoResolution']``.

    Raises
    ------
    ValueError
        If the AnnData has fewer than ~50 cells (subsampling becomes
        meaningless), or no neighbor graph is present.
    RuntimeError
        If no resolution produced ``≥ min_clusters`` clusters.
    """
    from sklearn.metrics import adjusted_rand_score
    from ..pp import leiden as _leiden  # tracked; nesting guard silences it

    n_obs = adata.n_obs
    if n_obs < 50:
        raise ValueError(
            f"autoResolution needs at least 50 cells; got {n_obs}."
        )
    if 'connectivities' not in adata.obsp:
        raise ValueError(
            "autoResolution requires a precomputed neighbor graph "
            "(adata.obsp['connectivities']); run ov.pp.neighbors first."
        )
    if not (0.0 < subsample_frac < 1.0):
        raise ValueError(
            f"subsample_frac must be in (0, 1); got {subsample_frac}."
        )

    if resolutions is None:
        resolutions = list(np.round(np.arange(0.2, 1.6, 0.1), 2))
    resolutions = [float(np.round(r, 3)) for r in resolutions]

    rng = np.random.default_rng(random_state)
    sub_n = max(int(round(n_obs * subsample_frac)), 30)
    subsamples = [
        np.sort(rng.choice(n_obs, size=sub_n, replace=False))
        for _ in range(n_subsamples)
    ]

    if verbose:
        print(
            f"{EMOJI['start']} autoResolution: testing "
            f"{len(resolutions)} resolutions × {n_subsamples} subsamples "
            f"(subsample_frac={subsample_frac}, n_cells={n_obs})"
        )

    # Reference labels at each r (kept so we can write the winner back
    # to adata.obs at the end without re-clustering). Stored as a
    # categorical-friendly string array.
    ref_labels: dict[float, np.ndarray] = {}
    rows = []

    REF_KEY = '_autores_ref'
    SUB_KEY = '_autores_sub'

    for r in resolutions:
        _leiden(adata, resolution=r, key_added=REF_KEY,
                random_state=random_state)
        ref = adata.obs[REF_KEY].astype(str).values.copy()
        n_clusters = int(pd.Series(ref).nunique())
        ref_labels[r] = ref

        if n_clusters < min_clusters:
            rows.append({
                'resolution': r,
                'mean_ari': np.nan,
                'std_ari':  np.nan,
                'n_clusters': n_clusters,
            })
            if verbose:
                print(f"  r={r:.2f}: clusters={n_clusters} "
                       f"(below min_clusters={min_clusters}, skipped)")
            continue

        aris = []
        for idx in subsamples:
            sub = adata[idx].copy()
            _leiden(sub, resolution=r, key_added=SUB_KEY,
                    random_state=random_state)
            ari = adjusted_rand_score(
                ref[idx],
                sub.obs[SUB_KEY].astype(str).values,
            )
            aris.append(ari)

        mean_ari = float(np.mean(aris))
        std_ari = float(np.std(aris))
        rows.append({
            'resolution': r,
            'mean_ari': mean_ari,
            'std_ari':  std_ari,
            'n_clusters': n_clusters,
        })
        if verbose:
            print(f"  r={r:.2f}: clusters={n_clusters:3d}  "
                   f"mean ARI={mean_ari:.3f} ± {std_ari:.3f}")

    # Drop temporary obs columns we leaked while testing.
    for col in (REF_KEY, SUB_KEY):
        if col in adata.obs.columns:
            del adata.obs[col]

    df = pd.DataFrame(rows).set_index('resolution').sort_index()
    eligible = df[df['n_clusters'] >= min_clusters]
    if eligible.empty:
        raise RuntimeError(
            f"No resolution produced >= {min_clusters} clusters; consider "
            "lowering `min_clusters` or extending `resolutions`."
        )
    best = float(eligible['mean_ari'].idxmax())
    best_n_clusters = int(df.loc[best, 'n_clusters'])
    best_ari = float(df.loc[best, 'mean_ari'])

    adata.obs[key_added] = pd.Categorical(ref_labels[best])
    adata.uns['autoResolution'] = {
        'best_resolution': best,
        'scores': df.reset_index().to_dict('list'),
        'n_subsamples': int(n_subsamples),
        'subsample_frac': float(subsample_frac),
        'method': 'bootstrap-ARI',
    }
    add_reference(
        adata, 'autoResolution',
        f'auto-selected leiden resolution={best} via subsample-ARI stability',
    )

    if verbose:
        print(
            f"{EMOJI['done']} chosen resolution: {best} "
            f"({best_n_clusters} clusters, mean ARI {best_ari:.3f})"
        )

    note(
        backend=f'omicverse · ARI-stability · {n_subsamples} subsamples',
        viz=[
            {'function': 'ov.pl.cluster_sizes_bar',
              'kwargs': {'groupby': key_added}},
            *([{'function': 'ov.pl.embedding',
                 'kwargs': {'basis': 'X_umap', 'color': key_added,
                            'frameon': 'small'}}]
               if 'X_umap' in adata.obsm else []),
        ],
    )

    return adata, best, df

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
    sc.pp.normalize_total(adata_GEP, target_sum=1e6)
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

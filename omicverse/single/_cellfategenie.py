import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
import anndata
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from .._registry import register_function

class PyTorchRidge: # 保持不变， 不需要修改
    def __init__(self, alpha=1.0, device='cpu'):
        self.alpha = alpha
        self.device = device
        self.weights = None
        self.coef_ = None # 初始化 coef_ 属性

    def fit(self, X_train, y_train):
        X_train_tensor = torch.tensor(X_train.values if isinstance(X_train, pd.DataFrame) else X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train.values if isinstance(y_train, pd.Series) else y_train, dtype=torch.float32).to(self.device)

        X_t_X = torch.matmul(X_train_tensor.T, X_train_tensor)
        I = torch.eye(X_train_tensor.shape[1]).to(self.device) # 单位矩阵
        A = X_t_X + self.alpha * I
        b = torch.matmul(X_train_tensor.T, y_train_tensor)

        # 使用 torch.linalg.solve 解线性方程组 Aw = b
        self.weights = torch.linalg.solve(A, b)
        self.coef_ = self.weights.cpu().numpy().flatten() # 在 fit 后将系数赋值给 coef_ 属性

    def predict(self, X_test):
        X_test_tensor = torch.tensor(X_test.values if isinstance(X_test, pd.DataFrame) else X_test, dtype=torch.float32).to(self.device)
        return torch.matmul(X_test_tensor, self.weights).cpu().numpy()

# 数据增强函数 (可以添加到 Fate 类外部或内部，这里放在外部更模块化)
def augment_pseudotime_jitter(y_train, jitter_std=0.05):
    """
    对 pseudotime 数据添加高斯噪声 (抖动).

    Parameters
    ----------
    y_train:pd.Series
        Pseudotime values in training set.
    jitter_std:float
        Standard deviation of Gaussian jitter.

    Returns
    -------
    pd.Series
        Jittered pseudotime series.
    """
    noise = np.random.normal(0, jitter_std, size=y_train.shape)
    return y_train + noise

def augment_gene_expression_noise(X_train, noise_std=0.01):
    """
    对基因表达数据添加高斯噪声.

    Parameters
    ----------
    X_train:pd.DataFrame
        Training-set gene expression matrix.
    noise_std:float
        Standard deviation of additive Gaussian noise.

    Returns
    -------
    pd.DataFrame
        Noise-augmented gene expression matrix.
    """
    noise = np.random.normal(0, noise_std, size=X_train.shape)
    return X_train + noise

@register_function(
    aliases=["细胞命运预测", "Fate", "time_fate_kernel", "TimeFateKernel", "时间命运核"],
    category="single",
    description="TimeFateKernel: Adaptive ridge regression model to identify timing-associated genes in single-cell pseudotime analysis",
    examples=[
        "# Initialize Fate analysis",
        "fate_obj = ov.single.Fate(adata, pseudotime='palantir_pseudotime')",
        "# Initialize model and run ATR",
        "fate_obj.model_init()",
        "fate_obj.ATR(stop=500)",
        "# Fit the model",
        "result = fate_obj.model_fit()",
        "# Plot filtering results",
        "fig, ax = fate_obj.plot_filtering()",
        "# Calculate lineage scores",
        "fate_obj.lineage_score(cluster_key='leiden', lineage=['0', '1'])"
    ],
    related=["pp.leiden", "utils.plot_heatmap", "pl.embedding"]
)
class Fate(object):
    """
    Adaptive ridge-regression framework for pseudotime-associated gene discovery.

    Parameters
    ----------
    adata:anndata.AnnData
        AnnData object containing pseudotime in ``obs`` and gene expression features.
    pseudotime:str
        ``adata.obs`` key storing pseudotime values.

    Returns
    -------
    None
        Initializes Fate model state.
    
    Examples
    --------
    >>> # Initialize Fate analysis
    """

    def __init__(self,adata:anndata.AnnData,pseudotime:str):
        """
        Initialize TimeFate model.

        Parameters
        ----------
        adata:anndata.AnnData
            Input AnnData containing expression matrix and pseudotime metadata.
        pseudotime:str
            Column name in ``adata.obs`` storing pseudotime values.

        """
        self.adata=adata
        self.pseudotime=pseudotime

    def model_init(self, test_size:float=0.3,
                   random_state:int=112, alpha:float=0.1,
                   use_data_augmentation: bool = False, # 新增参数：是否使用数据增强
                   augmentation_strategy: str = 'jitter_pseudotime_noise', # 新增参数：数据增强策略
                   augmentation_intensity: float = 0.05 # 新增参数：增强强度
                   )->pd.DataFrame:
        """
        Initialize the model

        Parameters
        ----------
        test_size:float
            Proportion of samples used as test set.
        random_state:int
            Random seed for train/test split.
        alpha:float
            Ridge regularization strength.
        use_data_augmentation:bool
            Whether to apply augmentation to training data.
        augmentation_strategy:str
            Augmentation mode: ``'jitter_pseudotime_noise'``,
            ``'gene_expression_noise'``, ``'both'`` or ``'none'``.
        augmentation_intensity:float
            Noise/jitter intensity used by selected strategy.

        Returns
        -------
        pd.DataFrame
            Per-gene coefficient table from initial ridge model.

        """
        X = self.adata.to_df()
        y = self.adata.obs.loc[:,self.pseudotime]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state)

        # 数据增强 (如果 use_data_augmentation 为 True)
        if use_data_augmentation:
            print(f"Applying data augmentation: {augmentation_strategy} with intensity {augmentation_intensity}")
            if augmentation_strategy == 'jitter_pseudotime_noise':
                y_train = augment_pseudotime_jitter(y_train, jitter_std=augmentation_intensity)
            elif augmentation_strategy == 'gene_expression_noise':
                X_train = augment_gene_expression_noise(X_train, noise_std=augmentation_intensity)
            elif augmentation_strategy == 'both': # 同时增强 pseudotime 和 基因表达
                y_train = augment_pseudotime_jitter(y_train, jitter_std=augmentation_intensity)
                X_train = augment_gene_expression_noise(X_train, noise_std=augmentation_intensity)
            elif augmentation_strategy == 'none':
                pass # 不进行数据增强
            else:
                raise ValueError(f"Unknown augmentation strategy: {augmentation_strategy}. Choose from 'jitter_pseudotime_noise', 'gene_expression_noise', 'both', 'none'.")

        # 初始化Ridge模型并拟合训练数据
        if torch.cuda.is_available():
            self.ridge = PyTorchRidge(alpha=alpha, device='cuda')
        else:
            self.ridge = PyTorchRidge(alpha=alpha, device='cpu')
       # PyTorchRidge(alpha=alpha, device='cuda' if torch.cuda.is_available() else 'cpu') # 使用 GPU 如果可用
        #self.ridge = Ridge(alpha=alpha)
        self.ridge.fit(X_train, y_train)

        # 预测测试集并计算均方误差
        y_pred = self.ridge.predict(X_test)
        self.y_test_r=y_test
        self.y_pred_r=y_pred

        # 计算均方误差（MSE）
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        self.raw_mse=mse
        self.raw_rmse=rmse
        self.raw_mae=mae
        self.raw_r2=r2
        print("$MSE|RMSE|MAE|R^2$:{:.2}|{:.2}|{:.2}|{:.2}".format(mse,rmse,mae,r2))

        res_pd_ievt=pd.DataFrame(index=self.adata.to_df().columns)
        res_pd_ievt['coef']=self.ridge.coef_
        res_pd_ievt['abs(coef)']=abs(self.ridge.coef_)
        res_pd_ievt['values']=self.adata.to_df().mean(axis=0)
        res_pd_ievt=res_pd_ievt.sort_values('abs(coef)',ascending=False)

        self.coef=res_pd_ievt
        return res_pd_ievt

    def atac_init(self,columns,gene_name='neargene'): # 保持不变， 不需要修改
        """
        Initialize the atac model

        if you want to use atac data to fit the model, you should use this function first

        Parameters
        ----------
            columns: list, the columns of atac data
            gene_name: str, the column name of gene name in adata.var

        """
        self.atac_gene_name=gene_name
        self.peak_pd=self.adata.var[columns].copy()

    #peak_pd=adata.var[['peaktype','neargene']].copy()
    def get_related_peak(self,peak): # 保持不变， 不需要修改
        """
        Get the related peak of gene

        Parameters
        ----------
            peak: str, the peak name

        """
        related_genes=self.peak_pd.loc[peak,self.atac_gene_name].unique()
        return self.peak_pd.loc[self.peak_pd[self.atac_gene_name].isin(related_genes)].index.tolist()

    def ATR(self, test_size:float=0.4, random_state:int=112,
            alpha:float=0.1, stop:int=100, flux=0.01, related=False,
            use_data_augmentation: bool = False, # 新增参数：是否使用数据增强
            augmentation_strategy: str = 'jitter_pseudotime_noise', # 新增参数：数据增强策略
            augmentation_intensity: float = 0.05 # 新增参数：增强强度
            )->pd.DataFrame:
        """
        Adaptive Threshold Regression

        Parameters
        ----------
        test_size:float
            Proportion of samples used as test set in each iteration.
        random_state:int
            Random seed for split reproducibility.
        alpha:float
            Ridge regularization strength.
        stop:int
            Maximum number of threshold iterations.
        flux:float
            Allowed R2 drop from raw model when selecting threshold.
        related:bool
            Whether to extend selected features to related ATAC peaks.
        use_data_augmentation:bool
            Whether to apply data augmentation in each ATR iteration.
        augmentation_strategy:str
            Augmentation mode: ``'jitter_pseudotime_noise'``,
            ``'gene_expression_noise'``, ``'both'`` or ``'none'``.
        augmentation_intensity:float
            Noise/jitter intensity used by selected strategy.

        Returns
        -------
        pd.DataFrame
            Iteration table with coefficient threshold and corresponding R2.

        """
        res_pd=pd.DataFrame()
        coef_threshold_li=[]
        r2_li=[]
        k=0
        from tqdm import tqdm
        for i in tqdm(self.coef['abs(coef)'].values[1:]):
            coef_threshold_li.append(i)
            train_idx=self.coef.loc[self.coef['abs(coef)']>=i].index.values
            if related == True:
                train_idx=self.get_related_peak(train_idx)

            adata_t=self.adata[:,train_idx]

            X = adata_t.to_df()
            y = adata_t.obs.loc[:,self.pseudotime]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                random_state=random_state)

            # 数据增强 (在每次 ATR 迭代中，如果 use_data_augmentation 为 True)
            if use_data_augmentation:
                if augmentation_strategy == 'jitter_pseudotime_noise':
                    y_train = augment_pseudotime_jitter(y_train, jitter_std=augmentation_intensity)
                elif augmentation_strategy == 'gene_expression_noise':
                    X_train = augment_gene_expression_noise(X_train, noise_std=augmentation_intensity)
                elif augmentation_strategy == 'both': # 同时增强 pseudotime 和 基因表达
                    y_train = augment_pseudotime_jitter(y_train, jitter_std=augmentation_intensity)
                    X_train = augment_gene_expression_noise(X_train, noise_std=augmentation_intensity)
                elif augmentation_strategy == 'none':
                    pass # 不进行数据增强

            # 初始化Ridge模型并拟合训练数据
            # 初始化Ridge模型并拟合训练数据
            if torch.cuda.is_available():
                self.ridge_t = PyTorchRidge(alpha=alpha, device='cuda')
            else:
                self.ridge_t = PyTorchRidge(alpha=alpha, device='cpu')
            #self.ridge_t = Ridge(alpha=alpha)
            self.ridge_t.fit(X_train, y_train)

            # 预测测试集并计算均方误差
            y_pred = self.ridge_t.predict(X_test)

            # 计算均方误差（MSE）
            #mse = mean_squared_error(y_test, y_pred)
            #rmse = root_mean_squared_error(y_test, y_pred)
            #mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            r2_li.append(r2)
            k+=1
            if k==stop:
                break

        res_pd['coef_threshold']=coef_threshold_li
        res_pd['r2']=r2_li

        for i in res_pd.index:
            if res_pd.loc[i,'r2']>=self.raw_r2-flux:
                self.coef_threshold=res_pd.loc[i,'coef_threshold']
                print("coef_threshold:{}, r2:{}".format(res_pd.loc[i,'coef_threshold'],res_pd.loc[i,'r2']))
                break

        self.max_threshold=res_pd
        return res_pd

    def model_fit(self, test_size:float=0.3,
                   random_state:int=112,
                   alpha:float=0.1, related=False,
                   use_data_augmentation: bool = False, # 新增参数：是否使用数据增强
                   augmentation_strategy: str = 'jitter_pseudotime_noise', # 新增参数：数据增强策略
                   augmentation_intensity: float = 0.05 # 新增参数：增强强度
                   )->pd.DataFrame:
        """
        Fit the model

        Parameters
        ----------
        test_size:float
            Proportion of samples used as test set.
        random_state:int
            Random seed for split reproducibility.
        alpha:float
            Ridge regularization strength in filtered model.
        related:bool
            Whether to extend selected features to related ATAC peaks.
        use_data_augmentation:bool
            Whether to augment training data.
        augmentation_strategy:str
            Augmentation mode: ``'jitter_pseudotime_noise'``,
            ``'gene_expression_noise'``, ``'both'`` or ``'none'``.
        augmentation_intensity:float
            Noise/jitter intensity used by selected strategy.

        Returns
        -------
        pd.DataFrame
            Per-gene coefficient table from filtered model.

        """
        train_idx=self.coef.loc[self.coef['abs(coef)']>=self.coef_threshold].index.values
        if related == True:
            train_idx=self.get_related_peak(train_idx)
        adata_t=self.adata[:,train_idx]
        X = adata_t.to_df()
        y = adata_t.obs.loc[:,self.pseudotime]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state)

        # 数据增强 (如果 use_data_augmentation 为 True)
        if use_data_augmentation:
            print(f"Applying data augmentation: {augmentation_strategy} with intensity {augmentation_intensity}")
            if augmentation_strategy == 'jitter_pseudotime_noise':
                y_train = augment_pseudotime_jitter(y_train, jitter_std=augmentation_intensity)
            elif augmentation_strategy == 'gene_expression_noise':
                X_train = augment_gene_expression_noise(X_train, noise_std=augmentation_intensity)
            elif augmentation_strategy == 'both': # 同时增强 pseudotime 和 基因表达
                y_train = augment_pseudotime_jitter(y_train, jitter_std=augmentation_intensity)
                X_train = augment_gene_expression_noise(X_train, noise_std=augmentation_intensity)
            elif augmentation_strategy == 'none':
                pass # 不进行数据增强

        # 初始化Ridge模型并拟合训练数据
        self.ridge_f = Ridge(alpha=alpha)
        self.ridge_f.fit(X_train, y_train)

        # 预测测试集并计算均方误差
        y_pred = self.ridge_f.predict(X_test)
        self.y_test_f=y_test
        self.y_pred_f=y_pred

        # 计算均方误差（MSE）
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        self.filter_mse=mse
        self.filter_rmse=rmse
        self.filter_mae=mae
        r2 = r2_score(y_test, y_pred)
        self.filter_r2=r2
        print("$MSE|RMSE|MAE|R^2$:{:.2}|{:.2}|{:.2}|{:.2}".format(mse,rmse,mae,r2))

        res_pd_ievt=pd.DataFrame(index=adata_t.to_df().columns)
        res_pd_ievt['coef']=self.ridge_f.coef_
        res_pd_ievt['abs(coef)']=abs(self.ridge_f.coef_)
        res_pd_ievt['values']=adata_t.to_df().mean(axis=0)
        res_pd_ievt=res_pd_ievt.sort_values('abs(coef)',ascending=False)

        self.filter_coef=res_pd_ievt
        return res_pd_ievt

    def kendalltau_filter(self): # 保持不变， 不需要修改
        """
        Compute Kendall's tau between filtered gene trends and pseudotime.

        Returns
        -------
        pandas.DataFrame
            Per-gene correlation statistics with columns ``kendalltau_sta`` and ``pvalue``.

        Examples
        --------
        >>> kt = fate.kendalltau_filter()
        """
        import pandas as pd
        from scipy.stats import kendalltau
        test_pd=pd.DataFrame()
        mk_sta_li=[]
        mk_p_li=[]
        t_series=self.adata.obs[self.pseudotime].sort_values()
        for gene in self.filter_coef.index.tolist():
            test_x=self.adata[t_series.index,gene].X.toarray().reshape(-1)
            statistic, p_value = kendalltau(t_series.values,test_x)
            mk_sta_li.append(statistic)
            mk_p_li.append(p_value)
        test_pd['kendalltau_sta']=mk_sta_li
        test_pd['pvalue']=mk_p_li
        test_pd.index=self.filter_coef.index.tolist()
        self.kendalltau_filter=test_pd
        return test_pd

    def low_density(self, # 保持不变， 不需要修改
                    n_components: int = 10,
                    knn: int = 30,
                    alpha: float = 0,
                    seed = 0,
                    pca_key: str = "X_pca",
                    kernel_key: str = "DM_Kernel",
                    sim_key: str = "DM_Similarity",
                    eigval_key: str = "DM_EigenValues",
                    eigvec_key: str = "DM_EigenVectors",):
        """
        Estimate manifold density on diffusion-map coordinates.

        Parameters
        ----------
        n_components:int, optional
            Number of diffusion components.
        knn:int, optional
            Number of neighbors for diffusion map construction.
        alpha:float, optional
            Diffusion alpha parameter.
        seed:int, optional
            Random seed.
        pca_key:str, optional
            ``adata.obsm`` key containing PCA coordinates.
        kernel_key:str, optional
            Key used to store diffusion kernel.
        sim_key:str, optional
            Key used to store diffusion similarity.
        eigval_key:str, optional
            Key used to store diffusion eigenvalues.
        eigvec_key:str, optional
            Key used to store diffusion eigenvectors.

        Returns
        -------
        None
            Stores low-dimensional density in ``adata.obs['mellon_log_density_lowd']``.

        Examples
        --------
        >>> fate.low_density(n_components=10, knn=30)
        """
        try:
            import mellon
        except:
            print("Please install mellon package first using ``pip install mellon``")
        from ..external.palantir.utils import run_diffusion_maps
        run_diffusion_maps(self.adata,n_components=n_components,knn=knn,alpha=alpha,seed=seed,
                           pca_key=pca_key,kernel_key=kernel_key,sim_key=sim_key,
                           eigval_key=eigval_key,eigvec_key=eigvec_key)

        model = mellon.DensityEstimator(d_method="fractal")
        log_density = model.fit_predict(self.adata.obsm["DM_EigenVectors"])
        self.adata.obs["mellon_log_density_lowd"] = log_density

    def lineage_score(self, # 保持不变， 不需要修改
                    cluster_key:str,lineage=None,
                    cell_mask= "specification",
                    density_key: str = "mellon_log_density_lowd",
                    localvar_key: str = "local_variability",
                    #score_key: str = "low_density_gene_variability",
                    expression_key: str = "MAGIC_imputed_data",
                    distances_key: str = "distances",
                    ):
        """
        Compute lineage-specific change scores using low-density variability.

        Parameters
        ----------
        cluster_key:str
            ``adata.obs`` key containing cluster or lineage labels.
        lineage:sequence of str, optional
            Cluster labels defining the lineage-of-interest.
        cell_mask:str, optional
            ``adata.obsm`` key used as lineage mask.
        density_key:str, optional
            ``adata.obs`` key containing estimated density.
        localvar_key:str, optional
            ``adata.layers`` key for local variability matrix.
        expression_key:str, optional
            Layer key used for expression in variability estimation.
        distances_key:str, optional
            ``adata.obsp`` key containing neighborhood distances.

        Returns
        -------
        None
            Writes lineage scores to ``adata.var['change_scores_lineage']``.

        Examples
        --------
        >>> fate.lineage_score(cluster_key='celltype', lineage=['Prog', 'Diff'])
        """
        from ..external.palantir.utils import run_low_density_variability,run_local_variability

        if localvar_key not in self.adata.layers.keys():
            print("Run low_density first")
            run_local_variability(self.adata,expression_key=expression_key,
                                  distances_key=distances_key,localvar_key=localvar_key)
        import pandas as pd
        specification_cells = (
            self.adata.obs[cluster_key].isin(lineage)
        )
        self.adata.obsm["specification"] = pd.DataFrame({"lineage": specification_cells})
        print("Calculating lineage score")
        run_low_density_variability(
                                    self.adata,
                                    cell_mask=cell_mask,
                                    density_key=density_key,
                                    score_key="change_scores",
                                )
        print(f"The lineage score stored in adata.var['change_scores_lineage']")

    def get_coef(self,type:str='raw')->pd.DataFrame: # 保持不变， 不需要修改
        """
        Get the coef of model

        Parameters
        ----------
        type:str
            Coefficient table type: ``'raw'`` or ``'filter'``.

        Returns
        -------
        pd.DataFrame
            Selected coefficient table.

        """

        if type=='raw':
            return self.coef
        elif type=='filter':
            return self.filter_coef

    def get_r2(self,type:str='raw')->float: # 保持不变， 不需要修改
        """
        Get the r2 of model

        Parameters
        ----------
        type:str
            Metric type: ``'raw'`` or ``'filter'``.

        Returns
        -------
        float
            R2 score of selected model.

        """
        if type=='raw':
            return self.raw_r2
        elif type=='filter':
            return self.filter_r2

    def get_mse(self,type:str='raw')->pd.DataFrame: # 保持不变， 不需要修改
        """
        Get the mse of model

        Parameters
        ----------
        type:str
            Metric type: ``'raw'`` or ``'filter'``.

        Returns
        -------
        float
            MSE of selected model.

        """
        if type=='raw':
            return self.raw_mse
        elif type=='filter':
            return self.filter_mse

    def get_rmse(self,type:str='raw')->pd.DataFrame: # 保持不变， 不需要修改
        """
        Get the rmse of model

        Parameters
        ----------
        type:str
            Metric type: ``'raw'`` or ``'filter'``.

        Returns
        -------
        float
            RMSE of selected model.

        """
        if type=='raw':
            return self.raw_rmse
        elif type=='filter':
            return self.filter_rmse

    def get_mae(self,type:str='raw')->pd.DataFrame: # 保持不变， 不需要修改
        """
        Get the mae of model

        Parameters
        ----------
        type:str
            Metric type: ``'raw'`` or ``'filter'``.

        Returns
        -------
        float
            MAE of selected model.

        """
        if type=='raw':
            return self.raw_mae
        elif type=='filter':
            return self.filter_mae

    def plot_filtering(self,figsize:tuple=(3,3),color:str='#5ca8dc', # 保持不变， 不需要修改
                    fontsize:int=12,alpha:float=0.8)->tuple:
        """
        Plot the filtering result

        Parameters
        ----------
        figsize:tuple
            Figure size.
        color:str
            Scatter color.
        fontsize:int
            Axis/annotation font size.
        alpha:float
            Scatter transparency.

        Returns
        -------
        Tuple[matplotlib.figure.Figure,matplotlib.axes.Axes]
            Figure and axes of ATR filtering plot.

        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(self.max_threshold['coef_threshold'],
                    self.max_threshold['r2'],color=color,alpha=alpha)
        ax.axhline(y=self.raw_r2, c="red")
        ax.text(self.max_threshold['coef_threshold'].max(),self.raw_r2,
                '$r^2:{:.2}$'.format(self.raw_r2),
                 fontsize=12,horizontalalignment='right')
        ax.axvline(x=self.coef_threshold, c="red")
        ax.text(self.coef_threshold,self.max_threshold['r2'].min(),
                '$ATR:{:.2}$'.format(self.coef_threshold),
                 fontsize=12,horizontalalignment='left')
        ax.spines['left'].set_position(('outward', 20))
        ax.spines['bottom'].set_position(('outward', 20))
        ax.set_xlabel('Coef threshold',fontsize=fontsize)
        ax.set_ylabel('$r^2$',fontsize=fontsize)
        #plt.ylim(0,1)
        #plt.xlim(0,1)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.grid(False)
        #设置spines可视化情况
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        return fig,ax

    def plot_fitting(self,type:str='raw', # 保持不变， 不需要修改
                     figsize:tuple=(3,3),color:str='#0d6a3b',
                    fontsize:int=12)->tuple:
        """
        Plot the fitting result

        Parameters
        ----------
        type:str
            Model type: ``'raw'`` or ``'filter'``.
        figsize:tuple
            Figure size.
        color:str
            Regression/scatter color.
        fontsize:int
            Axis/title font size.

        Returns
        -------
        Tuple[matplotlib.figure.Figure,matplotlib.axes.Axes]
            Figure and axes of fit-quality plot.

        """
        import seaborn as sns
        fig, ax = plt.subplots(figsize=figsize)
        if type=='raw':
            y_test=self.y_test_r
            y_pred=self.y_pred_r
        elif type=='filter':
            y_test=self.y_test_f
            y_pred=self.y_pred_f
        sns.regplot(x=y_test,y=y_pred,ax=ax,line_kws={'color':color},
                color=color)
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        ax.set_xlabel('Raw',fontsize=fontsize)
        ax.set_ylabel('Predicted',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.grid(False)
        #设置spines可视化情况
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        if type=='filter':
            ax.set_title(f'Dimension: {self.filter_coef.shape[0]}',
                         fontsize=fontsize)
        elif type=='raw':
            ax.set_title(f'Dimension: {self.coef.shape[0]}',
                         fontsize=fontsize)

        return fig,ax

    def plot_color_fitting(self,type:str='raw',cluster_key:str='clusters', # 保持不变， 不需要修改
                     figsize:tuple=(3,3),color:str='#6BBBA0',
                    fontsize:int=12,legend_loc:list=[0.2,0.1,0],omics='RNA')->tuple:
        """
        Plot the colorful of clusters fitting result

        Parameters
        ----------
        type:str
            Model type: ``'raw'`` or ``'filter'``.
        cluster_key:str
            ``adata.obs`` key used for cluster-wise coloring.
        figsize:tuple
            Figure size.
        color:str
            Base color for regression line/confidence region.
        fontsize:int
            Axis/title font size.
        legend_loc:list
            Text placement coordinates for metric summary.
        omics:str
            Omics label displayed in metric summary.

        Returns
        -------
        Tuple[matplotlib.figure.Figure,matplotlib.axes.Axes]
            Figure and axes of colored fit-quality plot.

        """
        #fontsize=13
        fig, ax = plt.subplots(figsize=figsize)
        if type=='raw':
            y_test=self.y_test_r
            y_pred=pd.Series(self.y_pred_r)
            y_pred.index=y_test.index
        elif type=='filter':
            y_test=self.y_test_f
            y_pred=pd.Series(self.y_pred_f)
            y_pred.index=y_test.index

        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(y_test, y_pred)
        line = slope * y_test + intercept

        # 计算置信区间的上界和下界
        confidence_interval = 1.96 * std_err  # 95% 置信区间

        upper_bound = line + confidence_interval
        lower_bound = line - confidence_interval

        #color_dict
        self.adata.obs[cluster_key]=self.adata.obs[cluster_key].astype('category')
        if '{}_colors'.format(cluster_key) in self.adata.uns.keys():
            color_dict=dict(zip(self.adata.obs[cluster_key].cat.categories.tolist(),
                            self.adata.uns['{}_colors'.format(cluster_key)]))
        else:
            if len(self.adata.obs[cluster_key].cat.categories)>28:
                from ._sc_compat import palette_102
                color_dict=dict(zip(self.adata.obs[cluster_key].cat.categories,palette_102()))
            else:
                from ._sc_compat import palette_28
                color_dict=dict(zip(self.adata.obs[cluster_key].cat.categories,palette_28()))

        for i in self.adata.obs[cluster_key].cat.categories:
            ax.scatter(y_test[list(set(self.adata.obs.loc[self.adata.obs[cluster_key]==i].index)&set(y_test.index))],
                    y_pred[list(set(self.adata.obs.loc[self.adata.obs[cluster_key]==i].index)&set(y_pred.index))],
                    color=color_dict[i])
        ax.plot(y_test, line, color=color,
                label='Fit: y = {:.2f}x + {:.2f}'.format(slope, intercept),
            linewidth=3)
        ax.fill_between(y_test, lower_bound, upper_bound,
                        color='grey', alpha=0.2, label='95% Confidence Interval')

        #sns.regplot(x=y_test,y_pred=y_pred,ax=ax,line_kws={'color':color},
        #        color=color)
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        ax.set_xlabel('True pseudotime',fontsize=fontsize)
        ax.set_ylabel('Predicted pseudotime',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.grid(False)
        #设置spines可视化情况
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        ax.text(1,legend_loc[0],'$r^2={:.2}$'.format(r2),fontsize=fontsize+1,horizontalalignment='right')
        ax.text(1,legend_loc[1],'$MSE={:.2}$'.format(mse),fontsize=fontsize+1,horizontalalignment='right')
        ax.text(1,legend_loc[2],'$MAE={:.2}$'.format(mae),fontsize=fontsize+1,horizontalalignment='right')

        if type=='filter':
            ax.set_title(f'Regression {omics}\nDimension: {self.filter_coef.shape[0]}',
                         fontsize=fontsize+1)
        elif type=='raw':
            ax.set_title(f'Regression {omics}\nDimension: {self.coef.shape[0]}',
                         fontsize=fontsize+1)
        return fig,ax

@register_function(
    aliases=['基因趋势分析器', 'gene_trends', 'pseudotime gene trends'],
    category="single",
    description="Model and visualize gene-set or gene-level trends along pseudotime to identify dynamic programs during differentiation.",
    prerequisites={'optional_functions': ['single.VIA', 'utils.cal_paga']},
    requires={'obs': ['pseudotime'], 'var': ['genes/features']},
    produces={'uns': ['gene trend fits']},
    auto_fix='none',
    examples=['gt_obj = ov.single.gene_trends(adata_aucs, "pt_via", var_name)', 'gt_obj.calculate()'],
    related=['single.Fate', 'utils.plot_paga']
)
class gene_trends(object):
    """
    Fit and visualize smooth feature trends along pseudotime.

    Parameters
    ----------
    adata:AnnData
        AnnData containing pseudotime and feature matrix.
    pseudotime:str
        ``adata.obs`` key used for temporal ordering.
    var_names:sequence of str
        Feature names to model along pseudotime.

    Returns
    -------
    None
        Initializes trend analysis configuration.
    
    Examples
    --------
    >>> gt_obj = ov.single.gene_trends(adata_aucs, "pt_via", var_name)
    """

    def __init__(self,adata,pseudotime,var_names):
        """
        Initialize the gene_trends analysis based on pseudotime

        Parameters
        ----------
            adata: AnnData object
            pseudotime: str, the column name of pseudotime in adata.obs
            var_names: list, the list of gene name to calculate
        
        """
        self.adata=adata
        self.pseudotime=pseudotime
        self.var_names=var_names


    def calculate(self,n_convolve=None):
        """
        Calculate the trends of gene with pseudotime

        Parameters
        ----------
            n_convolve: int, the number of convolve to smooth the trends
        
        """
        import numpy as np
        from scipy.spatial.distance import euclidean
        
        
        from scipy.sparse import issparse

        adata=self.adata
        pseudotime=self.pseudotime
        var_names=self.var_names
        
        time = adata.obs[pseudotime].values
        time = time[np.isfinite(time)]
        X = (
            adata[:, var_names].X
        )
        if issparse(X):
            X = X.A
        df = pd.DataFrame(X[np.argsort(time)], columns=var_names)
        

        if n_convolve is not None:
            weights = np.ones(n_convolve) / n_convolve
            for gene in var_names:
                # TODO: Handle exception properly
                try:
                    df[gene] = np.convolve(df[gene].values, weights, mode="same")
                except ValueError as e:
                    print(f"Skipping variable {gene}: {e}")
                    pass  # e.g. all-zero counts or nans cannot be convolved
        
        max_sort = np.argsort(np.argmax(df.values, axis=0))
        df = pd.DataFrame(df.values[:, max_sort], columns=df.columns[max_sort])
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(df)
        self.normalized_pd=pd.DataFrame(normalized_data,columns=df.columns,index=adata.obs[pseudotime].sort_values().index)
        self.normalized_data=normalized_data
        from statsmodels.tsa.stattools import adfuller
    
        # 生成示例时间序列数据
        np.random.seed(0)
        #time_series = np.random.rand(20)
        
        # 执行Cox-Stuart检验
        max_avg_li=[]
        for data_array in normalized_data:
            # 找到值大于 0.8 的元素的索引
            indices = np.where(data_array > np.max(data_array)*0.8)
            
            # 计算索引的平均值
            average_index = np.mean(indices)
            #print(average_index)
            max_avg_li.append(average_index)

        from scipy.stats import kendalltau,linregress
        self.max_avg_li=max_avg_li
        self.kt=kendalltau(range(len(max_avg_li)),np.array(max_avg_li))
        self.lr=linregress(range(len(max_avg_li)),np.array(max_avg_li))

    def get_heatmap(self):
        """
        Get the data of heatmap of trends
        
        """
        return self.normalized_data
    
    def get_kendalltau(self):
        """
        Get the kendalltau of trends
        
        """
        return self.kt
    
    def get_linregress(self):
        """
        Get the linregress of trends
        
        """
        return self.lr
    
    def cal_border_cell(self,adata:anndata.AnnData,
                        pseudotime:str,cluster_key:str,
                        threshold:float=0.1):
        """
        Calculate the border cell of each cluster

        Parameters
        ----------
            adata: AnnData object
            pseudotime: str, the column name of pseudotime in adata.obs
            cluster_key: str, the column name of cluster in adata.obs
            threshold: float, the threshold of border cell
        
        """
        adata.obs[cluster_key]=adata.obs[cluster_key].astype('category')
        adata.obs['border']=False
        adata.obs['border_type']='normal'
        for cluster in adata.obs[cluster_key].cat.categories:
            cluster_obs=adata.obs.loc[adata.obs[cluster_key]==cluster,:]
            pseudotime_min=np.min(adata.obs.loc[adata.obs[cluster_key]==cluster,pseudotime])
            pseudotime_max=np.max(adata.obs.loc[adata.obs[cluster_key]==cluster,pseudotime])
            ## set smaller than 10% and larger than 90% as border cells
            border_idx=cluster_obs.loc[(cluster_obs[pseudotime]<pseudotime_min+threshold*(pseudotime_max-pseudotime_min))|
                                        (cluster_obs[pseudotime]>=pseudotime_max-threshold*(pseudotime_max-pseudotime_min)),:].index
            adata.obs.loc[border_idx,'border']=True

            low_border_idx=cluster_obs.loc[(cluster_obs[pseudotime]<pseudotime_min+threshold*(pseudotime_max-pseudotime_min)),:].index
            high_border_idx=cluster_obs.loc[(cluster_obs[pseudotime]>=pseudotime_max-threshold*(pseudotime_max-pseudotime_min)),:].index
            adata.obs.loc[low_border_idx,'border_type']='low'
            adata.obs.loc[high_border_idx,'border_type']='high'
        print("adding ['border','border_type'] annotation to adata.obs")
    
    def get_border_gene(self,adata:anndata.AnnData,
                        cluster_key:str,cluster1:str,cluster2:str,
                        num_gene:int=10,threshold=None):
        """
        Get the border gene between two clusters

        Parameters
        ----------
            adata: AnnData object
            cluster_key: str, the column name of cluster in adata.obs
            cluster1: str, the name of cluster1
            cluster2: str, the name of cluster2
            num_gene: int, the number of border gene
            threshold: float, the threshold of border gene

        Returns
        -------
            border_gene: list, the list of border gene
        
        """
        if threshold is None:
            threshold=self.normalized_pd.mean().mean()
        cluster1_mean=np.mean(adata.obs.loc[adata.obs[cluster_key]==cluster1,self.pseudotime])
        cluster2_mean=np.mean(adata.obs.loc[adata.obs[cluster_key]==cluster2,self.pseudotime])
        if cluster1_mean>cluster2_mean:
            cluster1,cluster2=cluster2,cluster1
        max_cell_idx=adata.obs[(adata.obs[cluster_key]==cluster1)&(adata.obs['border_type']=='high')].index.tolist()
        min_cell_idx=adata.obs[(adata.obs[cluster_key]==cluster2)&(adata.obs['border_type']=='low')].index.tolist()
        #cell_idx=adata.obs[(adata.obs[cluster_key].isin([cluster1,cluster2])&(adata.obs['border']==True))].index
        data=self.normalized_pd.loc[min_cell_idx+max_cell_idx,:]
        #border_gene=data.mean().sort_values(ascending=False).index[:num_gene]
        # border_gene must larger than threshold
        border_gene=data.mean()[data.mean()>=threshold].sort_values(ascending=False).index[:num_gene]
        return border_gene
        
    def get_multi_border_gene(self,adata:anndata.AnnData,
                        cluster_key:str,
                        num_gene:int=10,threshold=None):
        """
        Get the border gene between two clusters for all clusters

        Parameters
        ----------
            adata: AnnData object
            cluster_key: str, the column name of cluster in adata.obs
            num_gene: int, the number of border gene
            threshold: float, the threshold of border gene

        Returns
        -------
            border_gene_dict: dict, the dict of border gene
        
        """
        border_gene_dict={}
        for cluster1 in adata.obs[cluster_key].cat.categories:
            for cluster2 in adata.obs[cluster_key].cat.categories:
                if f"{cluster2}_{cluster1}" in border_gene_dict.keys():
                    continue
                else:
                    if cluster1!=cluster2:
                        border_gene_dict[cluster1+'_'+cluster2]=self.get_border_gene(adata,
                            cluster_key,cluster1,cluster2,
                            num_gene=num_gene,threshold=threshold)
        return border_gene_dict
    
    def get_special_border_gene(self, adata:anndata.AnnData,
                                cluster_key:str,cluster1:str,cluster2:str,):
        """
        Get the special border gene between two clusters

        Parameters
        ----------
            adata: AnnData object
            cluster_key: str, the column name of cluster in adata.obs
            cluster1: str, the name of cluster1
            cluster2: str, the name of cluster2

        Returns
        -------
            border_gene: list, the list of border gene
        
        """
        # the border gene can't appear in other cluster
        border_gene_dict=self.get_multi_border_gene(adata,cluster_key,num_gene=10)
        cluster_name=f"{cluster1}_{cluster2}"
        if cluster_name not in border_gene_dict.keys():
            cluster_name=f"{cluster2}_{cluster1}"

        border_genes=border_gene_dict[cluster_name]
        for cluster in border_gene_dict.keys():
            if (cluster!=cluster1+'_'+cluster2)&(cluster!=cluster2+'_'+cluster1):
                for border_gene in border_gene_dict[cluster]:
                    if border_gene in border_genes:
                        border_genes=border_genes.drop(border_gene)
        return border_genes
    
    def get_kernel_gene(self,adata:anndata.AnnData,cluster_key:str,cluster:str,
                        num_gene:int=10,threshold=None):
        """
        Get the kernel gene of cluster

        Parameters
        ----------
            adata: AnnData object
            cluster_key: str, the column name of cluster in adata.obs
            cluster: str, the name of cluster
            num_gene: int, the number of kernel gene
            threshold: float, the threshold of kernel gene

        Returns
        -------
            kernel_gene: list, the list of kernel gene
        
        """
        if threshold is None:
            threshold=self.normalized_pd.mean().mean()
        cell_idx=adata.obs[(adata.obs[cluster_key].isin([cluster])&(adata.obs['border']==False))].index
        data=self.normalized_pd.loc[cell_idx,:]
        #border_gene=data.mean().sort_values(ascending=False).index[:num_gene]
        # border_gene must larger than threshold
        border_gene=data.mean()[data.mean()>=threshold].sort_values(ascending=False).index[:num_gene]
        return border_gene
    
    def get_multi_kernel_gene(self,adata:anndata.AnnData,
                        cluster_key:str,num_gene:int=10,threshold=None):
        """
        Get the kernel gene of cluster for all clusters

        Parameters
        ----------
            adata: AnnData object
            cluster_key: str, the column name of cluster in adata.obs
            num_gene: int, the number of kernel gene
            threshold: float, the threshold of kernel gene

        Returns
        -------
            kernel_gene_dict: dict, the dict of kernel gene
        
        """
        kernel_gene_dict={}
        for cluster in adata.obs[cluster_key].cat.categories:
            kernel_gene_dict[cluster]=self.get_kernel_gene(adata,
                            cluster_key,cluster,
                            num_gene=num_gene,threshold=threshold)
            
        return kernel_gene_dict
    
    def get_special_kernel_gene(self, adata:anndata.AnnData,
                                cluster_key:str,cluster:str,num_gene:int=10,):
        """
        Get the special kernel gene of cluster

        Parameters
        ----------
            adata: AnnData object
            cluster_key: str, the column name of cluster in adata.obs
            cluster: str, the name of cluster
            num_gene: int, the number of kernel gene

        Returns
        -------
            kernel_gene: list, the list of kernel gene
        """
        # the border gene can't appear in other cluster
        kernel_gene_dict=self.get_multi_kernel_gene(adata,cluster_key,num_gene=num_gene)
        kernel_genes=kernel_gene_dict[cluster]
        for cluster in kernel_gene_dict.keys():
            if cluster!=cluster:
                for kernel_gene in kernel_gene_dict[cluster]:
                    if kernel_gene in kernel_genes:
                        kernel_genes=kernel_genes.drop(kernel_gene)
        return kernel_genes


    def plot_trend(self,figsize:tuple=(3,3),max_threshold:float=0.8,
                   color:str='#a51616',xlabel:str='pseudotime',
                  ylabel:str='Genes',fontsize:int=12):
        """
        Plot the trends of gene with pseudotime

        Parameters
        ----------
            figsize: tuple, the size of figure
            max_threshold: float, the threshold of max value
            color: str, the color of scatter
            xlabel: str, the label of x axis
            ylabel: str, the label of y axis
            fontsize: int, the size of text

        Returns
        -------
            fig: matplotlib.pyplot.figure, the figure of trends
            ax: matplotlib.pyplot.axis, the axis of trends
        
        """
        fig, ax = plt.subplots(figsize=figsize)
        # 执行Cox-Stuart检验
        max_avg_li=[]
        for data_array in self.normalized_data:
            # 找到值大于 0.8 的元素的索引
            indices = np.where(data_array >= np.max(data_array)*max_threshold)
            
            # 计算索引的平均值
            average_index = np.mean(indices)
            #print(average_index)
            max_avg_li.append(average_index)
        ax.scatter(range(len(max_avg_li)),max_avg_li,color=color)
        ax.spines['left'].set_position(('outward', 20))
        ax.spines['bottom'].set_position(('outward', 20))
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        plt.grid(False)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        ax.set_ylabel(ylabel,fontsize=fontsize+1)
        ax.set_xlabel(xlabel,fontsize=fontsize+1)
        return fig,ax


def mellon_density(adata,
                    n_components: int = 10,
                    knn: int = 30,
                    alpha: float = 0,
                    seed = 0,
                    pca_key: str = "X_pca",
                    kernel_key: str = "DM_Kernel",
                    sim_key: str = "DM_Similarity",
                    eigval_key: str = "DM_EigenValues",
                    eigvec_key: str = "DM_EigenVectors",):
        try:
            import mellon
        except:
            print("Please install mellon package first using ``pip install mellon``")
        from ..external.palantir.utils import run_diffusion_maps
        run_diffusion_maps(adata,n_components=n_components,knn=knn,alpha=alpha,seed=seed,
                           pca_key=pca_key,kernel_key=kernel_key,sim_key=sim_key,
                           eigval_key=eigval_key,eigvec_key=eigvec_key)
        
        model = mellon.DensityEstimator(d_method="fractal")
        log_density = model.fit_predict(adata.obsm["DM_EigenVectors"])
        adata.obs["mellon_log_density_lowd"] = log_density

#聚类
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import scanpy as sc
import pandas as pd
import numpy as np
import anndata
from .._settings import add_reference
from .._registry import register_function
from ..external._pymclustr import fit_pymclustr

mira_install=False
def global_imports(modulename,shortname = None, asfunction = False):
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:        
        globals()[shortname] = __import__(modulename)

#初始化聚类位置，这个很重要
def get_initial_means(X, n_components,init_params, r):
    # Run a GaussianMixture with max_iter=0 to output the initialization means
    gmm = GaussianMixture(
        n_components=n_components, init_params=init_params, tol=1e-9, max_iter=0, random_state=r
    ).fit(X)
    return gmm.means_


def mclust_py(adata,  n_components=None,use_rep:str='X_pca',
              modelNames='EEE',  random_seed=2020):
    r"""Cluster cells with a Gaussian mixture model.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing a low-dimensional embedding in
        ``adata.obsm[use_rep]``.
    n_components : int or None, default=None
        Number of Gaussian components (clusters). Required for model fitting.
    use_rep : str, default='X_pca'
        Key in ``adata.obsm`` used as input features for clustering.
    modelNames : str, default='EEE'
        Covariance structure code inspired by R ``mclust`` naming.
        Supported mappings include ``EEE``, ``VVV``, ``EEV``, and ``VVI``.
    random_seed : int, default=2020
        Random seed for reproducible model initialization and fitting.

    Returns
    -------
    anndata.AnnData
        Input object with cluster assignments stored in
        ``adata.obs['mclust']`` and ``adata.obs['gmm_cluster']``.
    """

    if n_components is None:
        print('You need to input the `n_components` when methods is `GMM`')
        return
    print(f"""running GaussianMixture clustering""")
    # Extract the data to be clustered
    data = adata.obsm[use_rep]
    
    import numpy as np
    np.random.seed(random_seed)
    
    # Extract the data to be clustered
    data = adata.obsm[use_rep]
    
    # Map modelNames to scikit-learn covariance_type
    covariance_type_map = {
        'EEE': 'spherical',  # Equal volume, shape, and orientation (spherical)
        'VVV': 'full',       # Variable volume, shape, and orientation
        'EEV': 'tied',       # Equal volume and shape, variable orientation (tied)
        'VVI': 'diag',       # Variable volume and shape, equal orientation (diag)
        # Add more mappings as needed
    }
    
    covariance_type = covariance_type_map.get(modelNames, 'full')
    
    # Initialize and fit the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_seed)
    gmm.fit(data)
    
    # Get the cluster labels
    mclust_res = gmm.predict(data)
    
    # Add the cluster labels to adata.obs
    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    adata.obs['gmm_cluster'] = adata.obs['mclust']
    
    return adata


@register_function(
    aliases=["聚类", "cluster", "clustering", "细胞聚类", "单细胞聚类"],
    category="utils",
    description="Perform clustering using various algorithms including Leiden, Louvain, GMM, K-means, and scICE",
    prerequisites={
        'functions': ['neighbors'],
        'optional_functions': ['pca']
    },
    requires={
        'obsp': ['connectivities', 'distances'],
        'obsm': []
    },
    produces={
        'obs': ['leiden', 'louvain', 'kmeans', 'mclust', 'scICE_cluster', 'cellcharter']
    },
    auto_fix='escalate',
    examples=[
        "# Leiden clustering (recommended)",
        "sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)",
        "ov.utils.cluster(adata, method='leiden', resolution=1.0)",
        "# Gaussian Mixture Model clustering",
        "ov.utils.cluster(adata, method='GMM', n_components=10,",
        "                 use_rep='X_pca', covariance_type='full')",
        "# scICE ensemble clustering with stability analysis",
        "model = ov.utils.cluster(adata, method='scICE',",
        "                         resolution_range=(5,20), n_boot=50)",
        "# K-means clustering",
        "ov.utils.cluster(adata, method='kmeans', n_components=8)",
        "# Louvain clustering",
        "ov.utils.cluster(adata, method='louvain', resolution=0.8)",
        "# CellCharter-style spatial clustering",
        "model = ov.utils.cluster(adata, method='cellcharter',",
        "                         n_components=8, use_rep='X_pca')",
    ],
    related=["pp.neighbors", "pl.embedding", "utils.refine_label"]
)
def cluster(adata:anndata.AnnData,method:str='leiden',
            use_rep:str='X_pca',random_state:int=1024,
            n_components=None, **kwargs):
    r"""Run a selected clustering backend on single-cell data.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix to be clustered.
    method : str, default='leiden'
        Clustering backend. Supported values include ``'leiden'``,
        ``'louvain'``, ``'kmeans'``, ``'GMM'``, ``'mclust'``,
        ``'pymclustR'``, ``'schist'``, ``'scICE'``, and ``'cellcharter'``.
        ``'pymclustR'`` is the pure-Python re-implementation of CRAN
        ``mclust`` (the legacy ``'mclust_R'`` rpy2 backend has been
        removed).
    use_rep : str, default='X_pca'
        Key in ``adata.obsm`` used for embedding-based methods such as GMM,
        K-means, and scICE.
    random_state : int, default=1024
        Random seed used by stochastic clustering methods.
    n_components : int or None, default=None
        Number of clusters/components for ``'kmeans'``, ``'GMM'``, and
        ``'mclust'``.
    **kwargs
        Extra keyword arguments forwarded to the selected backend. For
        ``method='pymclustR'``, ``key_added`` selects the output column;
        the legacy spelling ``add_key`` is also accepted.

    Returns
    -------
    object or None
        Returns a fitted scICE or CellCharter model instance for the
        corresponding methods. Other methods write labels to ``adata.obs``
        and return ``None``.

    Examples
    --------
    >>> sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
    >>> cluster(adata, method='leiden', resolution=1.0)
    >>> cluster(adata, method='GMM', n_components=10, use_rep='X_pca')
    >>> scice_model = cluster(adata, method='scICE', use_rep='X_pca')
    >>> cc_model = cluster(adata, method='cellcharter', n_components=8, use_rep='X_pca')
    """

    if method=='leiden':
        sc.tl.leiden(adata,**kwargs)
        add_reference(adata,'leiden','clustering with Leiden')
    elif method=='louvain':
        sc.tl.louvain(adata,**kwargs)
        add_reference(adata,'louvain','clustering with Louvain')
    elif method=='kmeans':
        if n_components is None:
            print('You need to input the `n_components` when method is `kmeans`')
            return
        print(f"""running KMeans clustering""")
        kmeans = KMeans(n_clusters=n_components, random_state=random_state, **kwargs)
        kmeans_res = kmeans.fit_predict(adata.obsm[use_rep])
        adata.obs['kmeans'] = kmeans_res
        adata.obs['kmeans'] = adata.obs['kmeans'].astype('int')
        adata.obs['kmeans'] = adata.obs['kmeans'].astype('category')
        print(f"""finished: found {n_components} clusters and added
    'kmeans', the cluster labels (adata.obs, categorical)""")
        add_reference(adata,'kmeans','clustering with K-means')
    elif method=='GMM':
        mclust_py(adata, n_components=n_components,use_rep=use_rep,random_seed=random_state,**kwargs)
        print(f"""finished: found {n_components} clusters and added
    'mclust', the cluster labels (adata.obs, categorical)""")
        add_reference(adata,'GMM','clustering with Gaussian Mixture Model')
    elif method=='mclust':
        mclust_py(adata, n_components=n_components,use_rep=use_rep,
                  random_seed=random_state,**kwargs)
        print(f"""finished: found {n_components} clusters and added
    'mclust', the cluster labels (adata.obs, categorical)""")
        add_reference(adata,'mclust','clustering with Gaussian Mixture Model')
    elif method=='schist':
        try:
            import schist
        except ImportError:
            raise ImportError(
                'Please install the schist using conda `conda install -c conda-forge schist` \nor `pip install git+https://github.com/dawe/schist.git`'
            )
        schist.inference.nested_model(adata, **kwargs)
        add_reference(adata,'schist','clustering with schist')
    elif method=='pymclustR':
        if n_components is None:
            raise ValueError("n_components must be provided for method='pymclustR'")
        key_added = kwargs.pop('key_added', None)
        add_key = kwargs.pop('add_key', None)
        if key_added is not None and add_key is not None and key_added != add_key:
            raise ValueError(
                "key_added and add_key must match when both are provided"
            )
        output_key = (
            key_added
            if key_added is not None
            else add_key if add_key is not None else method
        )
        if not isinstance(output_key, str) or not output_key:
            raise ValueError("key_added must be a non-empty string")
        model_names = kwargs.pop('modelNames', 'EEE')
        labels, fit = fit_pymclustr(
            adata.obsm[use_rep],
            n_components=n_components,
            model_names=model_names,
            random_state=random_state,
            **kwargs,
        )
        adata.obs[output_key] = labels
        adata.obs[output_key] = adata.obs[output_key].astype('category')
        adata.uns.setdefault('pymclustR', {})[use_rep] = {
            'modelName': fit.model_name,
            'G': fit.G,
            'loglik': fit.loglik,
            'bic': fit.bic,
        }
        print(f"""finished: found {n_components} clusters and added
    '{output_key}', the cluster labels (adata.obs, categorical)
    [model={fit.model_name}, loglik={fit.loglik:.4f}, BIC={fit.bic:.4f}]""")
        add_reference(adata, 'pymclustR',
                      'clustering with pymclustR (Python re-implementation of CRAN mclust)')
    elif method=='scICE':
        from ._scice import scICE
        scice = scICE(n_jobs=-1,use_gpu=False)
        # Run clustering ensemble
        results = scice.fit(
            adata, 
            use_rep=use_rep,
            **kwargs
        )
        scice.add_to_adata(adata)
        print('scICE_cluster has been added to adata.obs')
        add_reference(adata,'scICE','clustering with scICE')
        return scice
    elif method=='cellcharter':
        if n_components is None:
            print('You need to input the `n_components` when method is `cellcharter`')
            return
        from ..space._cellcharter import cellcharter as run_cellcharter

        model = run_cellcharter(
            adata,
            n_clusters=n_components,
            use_rep=use_rep,
            random_state=random_state,
            **kwargs,
        )
        print(f"""finished: found {n_components} clusters and added
    'cellcharter', the cluster labels (adata.obs, categorical)""")
        add_reference(adata,'cellcharter','clustering with CellCharter')
        return model
    else:
        if method == 'mclust_R':
            raise ValueError(
                "method='mclust_R' (rpy2 bridge to CRAN mclust) was removed. "
                "Use method='pymclustR' — same 14 covariance parameterizations, "
                "no R / rpy2 dependency. Install with: pip install pymclustR"
            )
        raise ValueError(
            f"Unknown clustering method: {method!r}. Supported: 'leiden', "
            "'louvain', 'kmeans', 'GMM', 'mclust', 'pymclustR', 'schist', "
            "'scICE', 'cellcharter'."
        )



@register_function(
    aliases=["精化标签", "refine_label", "label_refinement", "标签优化", "邻域投票"],
    category="utils",
    description="Optimize cluster labels by majority voting in spatial neighborhood",
    prerequisites={
        'functions': [],
        'optional_functions': ['leiden', 'cluster']
    },
    requires={
        'obsm': ['spatial'],
        'obs': []
    },
    produces={
        'obs': ['label_refined']
    },
    auto_fix='none',
    examples=[
        "# Basic label refinement for spatial data",
        "adata.obs['refined_clusters'] = ov.utils.refine_label(",
        "    adata, radius=50, key='leiden')",
        "# Custom spatial representation",
        "adata.obs['refined_stagate'] = ov.utils.refine_label(",
        "    adata, use_rep='STAGATE', radius=30, key='mclust')",
        "# Fine-grained refinement",
        "adata.obs['refined_labels'] = ov.utils.refine_label(",
        "    adata, use_rep='spatial', radius=20, key='graphst_clusters')",
        "# After spatial clustering",
        "adata.obs['mclust_GraphST'] = ov.utils.refine_label(",
        "    adata, radius=50, key='mclust')"
    ],
    related=["utils.cluster", "space.merge_cluster", "space.clusters"]
)
def refine_label(adata, use_rep='spatial',radius=50, key='label'):
    """Refine labels with neighborhood majority voting.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing labels and coordinates.
    use_rep : str, default='spatial'
        Key in ``adata.obsm`` containing coordinates for neighborhood search.
    radius : int, default=50
        Number of nearest neighbors used for voting (excluding the cell itself).
    key : str, default='label'
        Column name in ``adata.obs`` containing original labels.

    Returns
    -------
    list of str
        Refined labels for all cells in ``adata``.
    """
    from scipy.spatial import distance
    from tqdm import tqdm
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm[use_rep]
    dist_matrix = distance.cdist(position, position, metric="euclidean")

    n_cell = dist_matrix.shape[0]

    for i in tqdm(range(n_cell)):
        vec = dist_matrix[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs['label_refined'] = np.array(new_type)

    return new_type
        
def filtered(adata:anndata.AnnData,
             cluster_key:str,
             cluster_minsize:int=10):
    """Collapse small clusters to an outlier label.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with categorical cluster labels in ``adata.obs``.
    cluster_key : str
        Column in ``adata.obs`` to be filtered by cluster size.
    cluster_minsize : int, default=10
        Minimum number of cells required to keep a cluster label.
        Clusters smaller than this threshold are reassigned to ``'-1'``.

    Returns
    -------
    None
        Operates in place by updating ``adata.obs[cluster_key]``.
    """
    new_num=adata.obs[cluster_key].value_counts()[adata.obs[cluster_key].value_counts()<cluster_minsize].shape[0]
    adata.obs['gmm_cluster']=adata.obs['gmm_cluster'].astype(str)
    adata.obs.loc[adata.obs[cluster_key].isin(adata.obs[cluster_key].value_counts()[adata.obs[cluster_key].value_counts()<cluster_minsize].index.tolist()),cluster_key]='-1'
    adata.obs[cluster_key]=adata.obs[cluster_key].astype('category')
    adata.obs['gmm_cluster'] = adata.obs['gmm_cluster'].cat.rename_categories(pd.Index(list(range(len(adata.obs['gmm_cluster'].cat.categories)))))
    print(f"""filtered {new_num} clusters and changed the cluster labels to '-1'(adata.obs, categorical)""")


@register_function(
    aliases=["LDA主题模型", "LDA_topic", "topic_model", "主题建模", "潜在狄利克雷分配"],
    category="utils", 
    description="Latent Dirichlet Allocation (LDA) topic modeling for single-cell data using MIRA",
    examples=[
        "# Basic LDA topic modeling",
        "LDA_obj = ov.utils.LDA_topic(adata, feature_type='expression',",
        "                           highly_variable_key='highly_variable_features',",
        "                           layers='counts')",
        "# Determine optimal number of topics",
        "LDA_obj.plot_topic_contributions(6)",
        "# Fit model and predict topics",
        "LDA_obj.predicted(13)",
        "# Advanced classification with Random Forest",
        "LDA_obj.get_results_rfc(adata, use_rep='X_pca',",
        "                        LDA_threshold=0.4, num_topics=13)",
        "# Plot topic compositions on embedding",
        "ov.pl.embedding(adata, basis='X_umap', color=LDA_obj.model.topic_cols,",
        "                cmap='BuPu', ncols=4)"
    ],
    related=["utils.cluster", "single.cNMF", "pl.embedding"]
)
class LDA_topic(object):
    r"""Latent Dirichlet Allocation (LDA) topic modeling for single-cell data using MIRA.

    Parameters
    ----------
    adata : anndata.AnnData
        Single-cell data object.
    feature_type : str
        Feature modality passed to MIRA topic model.
    highly_variable_key : str
        Var column indicating highly variable features.
    layers : str
        Layer key containing count matrix.
    batch_key : str or None
        Optional obs covariate for batch-aware modeling.
    learning_rate : float
        Learning rate for model optimization.
    ondisk : bool
        Whether to use on-disk dataset mode for large datasets.

    Returns
    -------
    LDA_topic
        Topic-model wrapper object.

    Examples:
        >>> import omicverse as ov
        >>> # Basic LDA topic modeling
        >>> LDA_obj = ov.utils.LDA_topic(adata, feature_type='expression',
        ...                              layers='counts')
        >>> # Determine optimal number of topics
        >>> LDA_obj.plot_topic_contributions(6)
        >>> # Fit model and predict topics
        >>> LDA_obj.predicted(13)
        >>> # Advanced classification with Random Forest
        >>> LDA_obj.get_results_rfc(adata, use_rep='X_pca', LDA_threshold=0.4)
    """

    def _apply_torch_compatibility_fix(self):
        """
        Apply compatibility fix for PyTorch versions missing torch._subclasses.schema_check_mode.
        
        This error commonly occurs when using mira-multiome with certain PyTorch versions.
        The fix creates a mock module to prevent ImportError.
        """
        try:
            import torch._subclasses.schema_check_mode
        except (ImportError, ModuleNotFoundError):
            try:
                import torch
                import sys
                from types import ModuleType
                
                # Create mock schema_check_mode module
                mock_module = ModuleType('schema_check_mode')
                
                # Add basic attributes that might be expected
                mock_module.__file__ = '<mock>'
                mock_module.__path__ = []
                
                # Create the submodules structure if needed
                if not hasattr(torch, '_subclasses'):
                    torch._subclasses = ModuleType('_subclasses')
                    sys.modules['torch._subclasses'] = torch._subclasses
                
                # Add the mock module
                torch._subclasses.schema_check_mode = mock_module
                sys.modules['torch._subclasses.schema_check_mode'] = mock_module
                
                print("Applied PyTorch compatibility fix for missing torch._subclasses.schema_check_mode")
                
            except Exception as e:
                print(f"Warning: Could not apply PyTorch compatibility fix: {e}")
                print("This may cause issues with mira functionality. Consider updating PyTorch.")

    def __init__(self,adata,feature_type='expression',
                  highly_variable_key='highly_variable_features',
                 layers='counts',batch_key=None,learning_rate=1e-3,ondisk=False):
        global mira_install
        
        # Apply PyTorch compatibility fix for missing torch._subclasses.schema_check_mode
        self._apply_torch_compatibility_fix()
        
        try:
            import mira
            mira_install=True
            print('mira have been install version:',mira.__version__)
        except ImportError:
            raise ImportError(
                """Please install the mira: `conda install -c bioconda mira-multiome` or 
                `pip install mira-multiome`.'"""
            )
        if mira_install==True:
            global_imports("mira")
        self.adata=adata
        self.model = mira.topics.make_model(
            adata.n_obs, adata.n_vars, # helps MIRA choose reasonable values for some hyperparameters which are not tuned.
            feature_type = feature_type,
            highly_variable_key=highly_variable_key,
            counts_layer=layers,
            categorical_covariates=batch_key
        )
        self.ondisk=ondisk
        if self.ondisk!=True:
        
            self.model.get_learning_rate_bounds(adata)
            self.model.set_learning_rates(learning_rate, 0.25) # for larger datasets, the default of 1e-3, 0.1 usually works well.
        else:
            train, test = self.model.train_test_split(adata)
            import os 
            os.mkdir('topic_train')
            os.mkdir('topic_test')
            self.model.write_ondisk_dataset(train, dirname='topic_train')
            self.model.write_ondisk_dataset(test, dirname='topic_test')
            self.model.get_learning_rate_bounds('topic_train')
            self.model.set_learning_rates(learning_rate, 0.25) # for larger datasets, the default of 1e-3, 0.1 usually works well.

        self.model.plot_learning_rate_bounds(figsize=(6,3))
        
    def plot_topic_contributions(self,num_topics=6):
        NUM_TOPICS = num_topics
        if self.ondisk!=True:
            topic_contributions = mira.topics.gradient_tune(self.model, self.adata)
        else:
            topic_contributions = mira.topics.gradient_tune(self.model, 'topic_train')
        mira.pl.plot_topic_contributions(topic_contributions, NUM_TOPICS)
        
    def predicted(self,num_topics=6):
        print(f"""running LDA topic predicted""")
        self.model = self.model.set_params(num_topics = num_topics).fit(self.adata)
        self.model.predict(self.adata)
        # 找到每行中最大值所在的列（topic）
        df=self.adata.obs[self.model.topic_cols].copy()
        max_topic = df.idxmax(axis=1)
        # 将结果添加到DataFrame中
        self.adata.obs['LDA_cluster'] = max_topic
        print(f"""finished: found {num_topics} clusters and added
    'LDA_cluster', the cluster labels (adata.obs, categorical)""")
        
    def get_results_rfc(self,adata,use_rep='X_pca',LDA_threshold=0.5,num_topics=6):
        import pandas as pd
        print(f"""running LDA topic predicted""")
        #self.model = self.model.set_params(num_topics = num_topics).fit(self.adata)
        #self.model.predict(self.adata)
        # 找到每行中最大值所在的列（topic）
        #df=self.adata.obs[self.model.topic_cols].copy()
        
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        new_array = []
        class_array = []
        for i in range(0, num_topics):
            data = adata[adata.obs[f'topic_{i}'] > LDA_threshold].obsm[use_rep].toarray()
            new_array.append(data)
            class_array.append(np.full(data.shape[0], i))

        new_array = np.concatenate(new_array, axis=0)
        class_array = np.concatenate(class_array)

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(new_array,class_array,test_size=0.3)
        clf = DecisionTreeClassifier(random_state=0)
        rfc = RandomForestClassifier(random_state=0)
        clf = clf.fit(Xtrain,Ytrain)
        rfc = rfc.fit(Xtrain,Ytrain)
        #查看模型效果
        score_c = clf.score(Xtest,Ytest)
        score_r = rfc.score(Xtest,Ytest)
        #打印最后结果
        print("Single Tree:",score_c)
        print("Random Forest:",score_r)

        adata.obs['LDA_cluster_rfc']=[str(i) for i in rfc.predict(adata.obsm[use_rep])]
        adata.obs['LDA_cluster_clf']=[str(i) for i in clf.predict(adata.obsm[use_rep])]
        print('LDA_cluster_rfc is added to adata.obs')
        print('LDA_cluster_clf is added to adata.obs')

# Differential celltype abundance analysis using pertpy
from anndata import AnnData
import numpy as np
from scipy.sparse import issparse
import pandas as pd

from .._settings import EMOJI,add_reference
from .._registry import register_function

@register_function(
    aliases=["差异组成分析", "DCT", "differential_abundance", "差异丰度分析", "细胞组成分析"],
    category="single",
    description="Differential cell type composition analysis using scCODA or Milo (edgepy) methods for abundance testing",
    prerequisites={
        'functions': ['leiden'],
        'optional_functions': ['pca', 'neighbors']
    },
    requires={
        'obs': []
    },
    produces={
        'uns': ['DCT_results']
    },
    auto_fix='escalate',
    examples=[
        "# Initialize DCT with scCODA",
        "dct_obj = ov.single.DCT(adata, condition='treatment', ctrl_group='Control',",
        "                        test_group='Treatment', cell_type_key='cell_type', method='sccoda')",
        "# Run scCODA analysis",
        "dct_obj.run(num_samples=5000, num_warmup=500)",
        "# Get results",
        "results = dct_obj.get_results()",
        "# Set FDR threshold",
        "dct_obj.model.set_fdr(dct_obj.sccoda_data, modality_key='coda', est_fdr=0.4)",
        "# Initialize DCT with Milo (OmicVerse's pure Python implementation)",
        "dct_obj = ov.single.DCT(adata, condition='condition', ctrl_group='Control',",
        "                        test_group='Disease', cell_type_key='celltype', sample_key='sample',",
        "                        method='milo', use_rep='X_pca')",
        "# Run Milo analysis (uses edgepy solver)",
        "dct_obj.run()",
        "# Get Milo results with mixing threshold",
        "milo_results = dct_obj.get_results(mix_threshold=0.6)"
    ],
    related=["single.DEG", "external.pertpy", "pp.neighbors"]
)
class DCT:
    """Differential cell-type abundance testing wrapper.

    Parameters
    ----------
    adata : AnnData
        Input single-cell AnnData.
    condition : str
        Condition column in ``adata.obs``.
    ctrl_group : str
        Control-group label.
    test_group : str
        Test-group label.
    cell_type_key : str
        Cell-type annotation column.
    method : str, default='sccoda'
        Differential-abundance backend.
    """
    def __init__(self, 
                 adata: AnnData, 
                 condition: str,
                 ctrl_group: str,
                 test_group: str,
                 cell_type_key: str,
                 method: str='sccoda',
                 sample_key=None,
                 use_rep=None,
    ):
        """
        Initialize differential cell-type abundance analysis.

        Parameters
        ----------
        adata : AnnData
            Single-cell AnnData object.
        condition : str
            Obs column containing condition labels.
        ctrl_group : str
            Control condition label.
        test_group : str
            Test condition label.
        cell_type_key : str
            Obs column containing cell-type annotations.
        method : str, default='sccoda'
            Differential abundance method: ``'sccoda'``, ``'milo'``, or ``'milopy'``.
        sample_key : str or None, default=None
            Obs column containing sample IDs (required for Milo methods).
        use_rep : str or None, default=None
            Embedding key in ``adata.obsm`` used for neighborhood graph in Milo.

        Returns
        -------
        None
        """
        # filter adata for condition and test group
        self.adata = adata
        self.condition = condition
        self.ctrl_group = ctrl_group
        self.test_group = test_group
        self.adata.obs[condition] = self.adata.obs[condition].astype('category')
        self.adata=self.adata[self.adata.obs[condition].isin([ctrl_group, test_group])]
        self.adata.obs[condition] = self.adata.obs[condition].cat.reorder_categories([ctrl_group, test_group])
        self.method = method
        self.cell_type_key = cell_type_key

        # Print EMOJI
        print(f"{EMOJI['check_mark']} Differential cell type abundance analysis initialized")
        print(f"{EMOJI['bar']} DCT analysis using {self.method} method")
        print(f"{EMOJI['bar']} Condition: {self.condition}, Control group: {self.ctrl_group}, Test group: {self.test_group}")

        if method == 'sccoda':
            # Import pertpy only for sccoda
            import pertpy as pt
            self.model = pt.tl.Sccoda()
            self.sccoda_data = self.model.load(
                adata,
                type="cell_level",
                #generate_sample_level=True,
                cell_type_identifier=cell_type_key,
                sample_identifier=sample_key,
                covariate_obs=[condition],
            )
            self.sccoda_data = self.model.prepare(
                self.sccoda_data,
                modality_key="coda",
                formula="condition",
                #reference_cell_type="Goblet",
            )
        elif method == 'milopy':
            #check if use_rep is provided
            if use_rep is None:
                raise ValueError("use_rep must be provided for milo")
            elif use_rep not in adata.obsm.keys():
                raise ValueError("use_rep must be a valid embedding in adata.obsm")

            # Use OmicVerse's Milo implementation (pure Python, uses edgepy)
            from ._milo_dev import Milo
            self.model = Milo()
            self.mdata = self.model.load(adata)
            from ..pp import neighbors
            neighbors(self.mdata["rna"], use_rep=use_rep, n_neighbors=150)
            self.model.make_nhoods(self.mdata["rna"], prop=0.1)
            self.mdata = self.model.count_nhoods(self.mdata, sample_col=sample_key)
        elif method == 'milo':
            # Use pertpy's Milo implementation
            import pertpy as pt
            self.model = pt.tl.Milo()
            self.mdata = self.model.load(adata)
            from ..pp import neighbors
            neighbors(self.mdata["rna"], use_rep=use_rep, n_neighbors=150)
            self.model.make_nhoods(self.mdata["rna"], prop=0.1)
            self.mdata = self.model.count_nhoods(self.mdata, sample_col=sample_key)
        else:
            raise ValueError(f"Method {self.method} not supported")


    def run(self,**kwargs):
        """
        Run differential abundance testing.

        Parameters
        ----------
        **kwargs
            Additional method-specific keyword arguments.

        Returns
        -------
        None
        """
        if self.method == 'sccoda':
            self.model.run_nuts(self.sccoda_data, modality_key="coda", **kwargs)
            self.model.credible_effects(self.sccoda_data, modality_key="coda")
            print(f"{EMOJI['check_mark']} {self.method} DCT analysis completed")
            add_reference(self.adata,'Sccoda','differential cell type abundance analysis with Sccoda')
            add_reference(self.adata,'pertpy','Sccoda is a part of pertpy')
        elif self.method == 'milopy':
            # Use edgepy solver (pure Python implementation)
            self.model.da_nhoods(
                self.mdata,
                design=f"~{self.condition}",
                model_contrasts=f"{self.condition}[{self.test_group}]-{self.condition}[{self.ctrl_group}]",
                solver="edger"
            )
            self.model.build_nhood_graph(self.mdata)
            self.model.annotate_nhoods(self.mdata, anno_col=self.cell_type_key)
            print(f"{EMOJI['check_mark']} {self.method} DCT analysis completed")
            add_reference(self.adata,'Milo','differential cell type abundance analysis with Milo')
            add_reference(self.adata,'OmicVerse','Milo implementation uses OmicVerse edgepy solver')
        elif self.method == 'milo':
            self.model.da_nhoods(self.mdata, design=f"~{self.condition}", model_contrasts=f"{self.condition}{self.test_group}-{self.condition}{self.ctrl_group}")
            self.model.build_nhood_graph(self.mdata)
            self.model.annotate_nhoods(self.mdata, anno_col=self.cell_type_key)
            print(f"{EMOJI['check_mark']} {self.method} DCT analysis completed")
            add_reference(self.adata,'Milo','differential cell type abundance analysis with Milo')
            add_reference(self.adata,'pertpy','Milo is a part of pertpy')
        else:
            raise ValueError(f"Method {self.method} not supported")

    def get_results(self,mix_threshold: float=0.6):
        """
        Retrieve differential abundance results.

        Parameters
        ----------
        mix_threshold : float, default=0.6
            Minimum neighborhood purity for Milo annotations; lower-purity
            neighborhoods are labeled ``'Mixed'``.

        Returns
        -------
        pandas.DataFrame
            Differential abundance result table.
        """
        if self.method == 'sccoda':
            return self.model.get_effect_df(self.sccoda_data, modality_key="coda")
        elif self.method == 'milo':
            self.mdata["milo"].var[ "nhood_annotation"]=self.mdata["milo"].var[ "nhood_annotation"].astype(str)
            self.mdata["milo"].var.loc[self.mdata["milo"].var["nhood_annotation_frac"] < mix_threshold, "nhood_annotation"] = "Mixed"
            return self.mdata["milo"].var
        elif self.method == 'milopy':
            self.mdata["milo"].var['nhood_annotation']=self.mdata["milo"].var['nhood_annotation'].astype(str)
            self.mdata["milo"].var.loc[self.mdata["milo"].var["nhood_annotation_frac"] < mix_threshold, "nhood_annotation"] = "Mixed"
            return self.mdata["milo"].var
        else:
            raise ValueError(f"Method {self.method} not supported")
        
            

@register_function(
    aliases=["差异表达分析", "DEG", "differential_expression", "差异基因分析", "单细胞差异表达"],
    category="single",
    description="Differential gene expression analysis for single-cell data using Wilcoxon, t-test, or memento methods",
    prerequisites={
        'optional_functions': ['preprocess', 'leiden']
    },
    requires={
        'obs': []  # Dynamic: requires condition and celltype_key columns (user-specified)
    },
    produces={
        'uns': ['rank_genes_groups']  # For wilcoxon/t-test methods
    },
    auto_fix='none',
    examples=[
        "# Initialize DEG with Wilcoxon test",
        "deg_obj = ov.single.DEG(adata, condition='condition', ctrl_group='Control',",
        "                        test_group='Treatment', method='wilcoxon')",
        "# Run DEG analysis for specific cell types",
        "deg_obj.run(celltype_key='cell_label', celltype_group=['T_cells', 'B_cells'])",
        "# Get results",
        "results = deg_obj.get_results()",
        "# Initialize with t-test method",
        "deg_obj = ov.single.DEG(adata, condition='condition', ctrl_group='Control',",
        "                        test_group='Disease', method='t-test')",
        "# Run for all cell types",
        "deg_obj.run(celltype_key='celltype', celltype_group=None)",
        "# Initialize with memento method",
        "deg_obj = ov.single.DEG(adata, condition='status', ctrl_group='healthy',",
        "                        test_group='diseased', method='memento-de')",
        "# Run memento analysis",
        "deg_obj.run(celltype_key='cell_type', celltype_group=['neurons'],",
        "           capture_rate=0.07, num_cpus=8, num_boot=5000)"
    ],
    related=["single.DCT", "bulk.pyDEG", "tl.rank_genes_groups"]
)
class DEG:
    """Differential gene-expression testing wrapper for single-cell datasets.

    Parameters
    ----------
    adata : AnnData
        Input single-cell AnnData.
    condition : str
        Condition column in ``adata.obs``.
    ctrl_group : str
        Control-group label.
    test_group : str
        Test-group label.
    method : str, default='wilcoxon'
        DEG method name.
    use_raw : bool or None, default=None
        Whether to use ``adata.raw`` as expression matrix.
    """
    def __init__(self,
                 adata: AnnData,
                 condition: str,
                 ctrl_group: str,
                 test_group: str,
                 method: str='wilcoxon',
                 use_raw: bool=None,
                 ):
        """
        Initialize differential expression analysis.

        Parameters
        ----------
        adata : AnnData
            Single-cell AnnData object.
        condition : str
            Obs column containing condition labels.
        ctrl_group : str
            Control condition label.
        test_group : str
            Test condition label.
        method : str, default='wilcoxon'
            DEG method: ``'wilcoxon'``, ``'t-test'``, or ``'memento-de'``.
        use_raw : bool or None, default=None
            Whether to use ``adata.raw`` as expression matrix. If ``None``,
            auto-detects based on availability.

        Returns
        -------
        None
        """
        # Auto-detect use_raw
        if use_raw is None:
            if adata.raw is not None:
                use_raw = True
                print(f"{EMOJI['bar']} Auto-detected adata.raw, will use raw data for DEG analysis")
            else:
                use_raw = False

        self.use_raw = use_raw

        # If using raw, create a copy with raw data as X
        if self.use_raw and adata.raw is not None:
            # Create a new AnnData with raw data
            self.adata = AnnData(
                X=adata.raw.X,
                obs=adata.obs.copy(),
                var=adata.raw.var.copy(),
            )
            print(f"{EMOJI['bar']} Using raw data with {self.adata.shape[1]} genes")
        else:
            self.adata = adata

        self.condition=condition
        self.ctrl_group=ctrl_group
        self.test_group=test_group
        self.method=method

        from scipy.sparse import csr_matrix
        try:
            assert type(self.adata.X) == csr_matrix
        except:
            self.adata.X = csr_matrix(self.adata.X)

        #print EMOJI
        print(f"{EMOJI['check_mark']} Differential expression analysis initialized")
        print(f"{EMOJI['bar']} DEG analysis using {self.method} method")
        #print the condition and ctrl_group and test_group
        print(f"{EMOJI['bar']} Condition: {self.condition}, Control group: {self.ctrl_group}, Test group: {self.test_group}")
        
        
    def run(self,
            celltype_key: str,
            celltype_group=None,
            max_cells: int=100000,
            **kwargs):
        """
        Run differential expression testing.

        Parameters
        ----------
        celltype_key : str
            Obs column containing cell-type labels.
        celltype_group : list or None, default=None
            Cell types to include. If ``None``, uses all cell types.
        max_cells : int, default=100000
            Maximum number of cells used for testing (downsamples if exceeded).
        **kwargs
            Additional method-specific arguments, e.g. memento parameters
            ``capture_rate``, ``num_cpus``, and ``num_boot``.

        Returns
        -------
        None
        """
        if celltype_group is None:
            celltype_group = self.adata.obs[celltype_key].unique()
        else:
            celltype_group = celltype_group

        self.adata_test=self.adata[self.adata.obs[celltype_key].isin(celltype_group)]

        if max_cells is None:
            max_cells = self.adata_test.shape[0]

        print(f"{EMOJI['bar']} Celltype key: {celltype_key}, Celltype group: {celltype_group}")
        print(f"Total cells: {self.adata_test.shape[0]} will be used for DEG analysis")
        if self.adata_test.shape[0] == 0:
            raise ValueError(f"No cells found for DEG analysis")
        elif self.adata_test.shape[0] > max_cells:
            print(f"{EMOJI['warning']} Total cells: {self.adata_test.shape[0]} is too large, will be downsampled to {max_cells}")
            print(f"If you want to keep all cells, please set max_cells to None")
            from ..pp import subsample
            try:
                subsample(self.adata_test, n_obs=max_cells)
            except Exception as e:
                raise ValueError(f"Failed to downsample the data, please check the data") from e
                

  

        if self.method == 'wilcoxon' or self.method == 't-test':
            self.adata_test=self.adata_test[self.adata_test.obs[self.condition].isin([self.ctrl_group, self.test_group])]

            if is_all_integer(self.adata_test.X):
                from ..pp import normalize_total, log1p
                self.adata_test = self.adata_test.copy()
                normalize_total(self.adata_test, inplace=True)
                log1p(self.adata_test)

            from ._sc_compat import rank_genes_groups
            rank_genes_groups(
                self.adata_test,  
                groupby=self.condition, 
                groups=[self.test_group, self.ctrl_group], 
                reference=self.ctrl_group, 
                n_genes=self.adata_test.shape[1], 
                method=self.method 
            ) 
            print(f"{EMOJI['check_mark']} {self.method} DEG analysis completed")
            add_reference(self.adata,'Wilcoxon','differential expression analysis with Wilcoxon')
            add_reference(self.adata,'T-test','differential expression analysis with T-test')
            add_reference(self.adata,'scanpy','differential expression analysis with SCANPY')
        elif self.method == 'memento-de':
            import memento
            self.adata_test=self.adata_test[self.adata_test.obs[self.condition].isin([self.ctrl_group, self.test_group])]
            self.adata_test.obs['stim'] = self.adata_test.obs[self.condition].apply(lambda x: 0 if x == self.ctrl_group else 1)

            # check counts in adata_test.layers['counts']
            if is_all_integer(self.adata_test.X):
                pass
            elif 'counts' in self.adata_test.layers:
                self.adata_test.X = self.adata_test.layers['counts']
            else:
                from ..pp import recover_counts
                from scipy.sparse import issparse, csr_matrix

                if issparse(self.adata_test.X):
                    pass 
                else:
                    self.adata_test.X=csr_matrix(self.adata_test.X)
                #detect the lognorm 10e4 or other value use max value
                if self.adata_test.X.max() < np.log1p(1e4):
                    lognorm = 1e4
                elif self.adata_test.X.max() < np.log1p(10*1e4):
                    lognorm = 10*1e4
                elif self.adata_test.X.max() < np.log1p(50*1e4):
                    lognorm = 50*1e4
                elif self.adata_test.X.max() < np.log1p(100*1e4):
                    lognorm = 100*1e4
                else:
                    lognorm = self.adata_test.X.max()
                print('Recover the counts matrix from log-normalized data.')
                X_counts_recovered, size_factors_sub=recover_counts(self.adata_test.X, lognorm, 
                                                                    lognorm*10,  log_base=None, 
                                                          chunk_size=10000)
                self.adata_test.X=X_counts_recovered

            result_1d = memento.binary_test_1d(
                adata=self.adata_test, 
                treatment_col='stim', 
                **kwargs
            )
            self.result_1d = result_1d
            print(f"{EMOJI['check_mark']} {self.method} DEG analysis completed")
            add_reference(self.adata,'memento','differential expression analysis with memento-de')
        else:
            raise ValueError(f"Method {self.method} not supported")
 
            
        
    def get_results(self):
        """
        Get DEG result table.

        Returns
        -------
        pandas.DataFrame
            Differential expression results in OmicVerse DEG format.
        """
        if self.method == 'wilcoxon' or self.method == 't-test':
            from ._sc_compat import rank_genes_groups_df
            md_d = (
                rank_genes_groups_df(self.adata_test, group=self.test_group)
                .set_index("names", drop=False)
            )
            res=pd.DataFrame(index=md_d['names'].tolist())
            res['log2FC']=md_d['logfoldchanges'].tolist()
            res['pvalue']=md_d['pvals'].tolist()
            res['padj']=md_d['pvals_adj'].tolist()
            res['qvalue']=md_d['pvals_adj'].tolist()
            res['size']=np.abs(res['log2FC'])/10
            res['sig']='normal'
            res.loc[(res['padj'] < 0.05) & (res['log2FC'] > 0), 'sig'] = 'up'
            res.loc[(res['padj'] < 0.05) & (res['log2FC'] < 0), 'sig'] = 'down'
            res['-log(pvalue)'] = -np.log10(res['pvalue'])
            res['-log(qvalue)'] = -np.log10(res['qvalue'])
            #calculate the Mean of the self.adata_test's genes(var)
            res['baseMean']=np.mean(self.adata_test.to_df(),axis=0).loc[res.index]

            # Calculate expression percentage for ctrl and test groups
            print(f"{EMOJI['bar']} Calculating expression percentages for {self.ctrl_group} and {self.test_group}...")
            ctrl_cells = self.adata_test[self.adata_test.obs[self.condition] == self.ctrl_group]
            test_cells = self.adata_test[self.adata_test.obs[self.condition] == self.test_group]

            # Get expression matrix for each group
            if issparse(ctrl_cells.X):
                ctrl_expr = ctrl_cells.X.toarray()
            else:
                ctrl_expr = ctrl_cells.X

            if issparse(test_cells.X):
                test_expr = test_cells.X.toarray()
            else:
                test_expr = test_cells.X

            # Calculate percentage of cells expressing each gene (expression > 0)
            pct_ctrl = np.sum(ctrl_expr > 0, axis=0) / ctrl_cells.shape[0] * 100
            pct_test = np.sum(test_expr > 0, axis=0) / test_cells.shape[0] * 100

            # Map to gene names in results
            gene_to_idx = {gene: idx for idx, gene in enumerate(ctrl_cells.var_names)}
            res['pct_ctrl'] = [pct_ctrl[gene_to_idx[gene]] if gene in gene_to_idx else 0 for gene in res.index]
            res['pct_test'] = [pct_test[gene_to_idx[gene]] if gene in gene_to_idx else 0 for gene in res.index]
            res['pct_diff'] = res['pct_test'] - res['pct_ctrl']

            print(f"{EMOJI['check_mark']} Expression percentages calculated successfully")

            self.result=res
            return res
        elif self.method == 'memento-de':
            self.result=self.result_1d
            self.result['baseMean']=np.mean(self.adata_test.to_df(),axis=0).iloc[self.result.index.to_list()].to_list()

            # Calculate expression percentage for ctrl and test groups
            print(f"{EMOJI['bar']} Calculating expression percentages for {self.ctrl_group} and {self.test_group}...")
            ctrl_cells = self.adata_test[self.adata_test.obs[self.condition] == self.ctrl_group]
            test_cells = self.adata_test[self.adata_test.obs[self.condition] == self.test_group]

            # Get expression matrix for each group
            if issparse(ctrl_cells.X):
                ctrl_expr = ctrl_cells.X.toarray()
            else:
                ctrl_expr = ctrl_cells.X

            if issparse(test_cells.X):
                test_expr = test_cells.X.toarray()
            else:
                test_expr = test_cells.X

            # Calculate percentage of cells expressing each gene (expression > 0)
            pct_ctrl = np.sum(ctrl_expr > 0, axis=0) / ctrl_cells.shape[0] * 100
            pct_test = np.sum(test_expr > 0, axis=0) / test_cells.shape[0] * 100

            # Map to gene names in results
            gene_to_idx = {gene: idx for idx, gene in enumerate(ctrl_cells.var_names)}
            self.result['pct_ctrl'] = [pct_ctrl[gene_to_idx[gene]] if gene in gene_to_idx else 0 for gene in self.result.index]
            self.result['pct_test'] = [pct_test[gene_to_idx[gene]] if gene in gene_to_idx else 0 for gene in self.result.index]
            self.result['pct_diff'] = self.result['pct_test'] - self.result['pct_ctrl']

            print(f"{EMOJI['check_mark']} Expression percentages calculated successfully")

            return self.result
        else:
            raise ValueError(f"Method {self.method} not supported")




def is_all_integer(data, allow_negative=True, tol=0):
    """
    判断矩阵 data 中所有元素是否为整数。

    参数
    ----
    data : np.ndarray 或 scipy.sparse 矩阵
        待检测的数据矩阵。
    allow_negative : bool, default True
        如果为 False，则遇到任何负值直接返回 False。
    tol : float, default 0
        允许的浮点数误差容限。如果你的数据理论上是整数但由于浮点运算会出现 1e-16 之类的小偏差，
        可以把 tol 设为 1e-8 之类的值。

    返回
    ----
    bool
        如果所有值都是整数（且符合 allow_negative 要求），返回 True；否则 False。
    """
    # 1) 取出实际的 ndarray
    if issparse(data):
        arr = data.data
    else:
        arr = np.asarray(data)

    # 2) 可选：是否允许出现负数
    if not allow_negative and np.signbit(arr).any():
        return False

    # 3) 判断"小数部分"是否都在 tol 以内
    #    np.modf(arr) 会返回 (frac_part, int_part)
    frac_part, _ = np.modf(arr)
    return np.all(np.abs(frac_part) <= tol)

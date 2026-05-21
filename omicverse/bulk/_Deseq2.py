r"""
Different Expression analysis in Python
"""

import numpy as np
import pandas as pd

import statsmodels.api as sm
import anndata as ad
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib
from typing import Union, Tuple, Optional, Any
from ..utils import plot_boxplot
from .._registry import register_function
from ..pl import volcano

@register_function(
    aliases=["基因ID映射", "gene_id_mapping", "id_mapping", "基因符号转换", "gene_symbol_mapping"],
    category="bulk",
    description="Map gene IDs to gene symbols using a reference table for bulk RNA-seq data",
    examples=[
        "ov.bulk.Matrix_ID_mapping(data, gene_ref_path='gene_reference.txt')",
        "ov.bulk.Matrix_ID_mapping(data, gene_ref_path='gene_ref.tsv', keep_unmapped=False)"
    ],
    related=["bulk.deseq2_normalize", "utils.gene_symbol_to_ensembl", "pp.filter_genes"]
)
def Matrix_ID_mapping(data:pd.DataFrame,gene_ref_path:str,keep_unmapped:bool=True,
                      auto_download:bool=True)->pd.DataFrame:
    r"""Map gene IDs in the input data to gene symbols using a reference table.

    Arguments:
        data: The input data containing gene IDs as index.
        gene_ref_path: The path to the reference table containing the mapping from gene IDs to gene symbols.
        keep_unmapped: Whether to keep genes that are not found in the mapping table. If True, unmapped genes retain their original IDs. If False, unmapped genes are removed (original behavior). Default: True.
        auto_download: If the reference at ``gene_ref_path`` is missing AND its
            basename matches a known omicverse pair (``pair_GRCh38.tsv``,
            ``pair_GRCh37.tsv``, ``pair_GRCm39.tsv``, ``pair_danRer11.tsv``),
            call :func:`ov.utils.download_geneid_annotation_pair` to fetch the
            standard pairs into ``./genesets/`` and resolve from there.

    Returns:
        data: The input data with gene IDs mapped to gene symbols.

    """

    import os
    if auto_download and not os.path.exists(gene_ref_path):
        known_pairs = {
            'pair_GRCh38.tsv', 'pair_GRCh37.tsv',
            'pair_GRCm39.tsv', 'pair_danRer11.tsv',
        }
        basename = os.path.basename(gene_ref_path)
        if basename in known_pairs:
            from ..utils._data import download_geneid_annotation_pair
            print(f"......Gene-ID pair '{basename}' missing locally; "
                  f"auto-downloading via ov.utils.download_geneid_annotation_pair()...")
            download_geneid_annotation_pair()
            cand = os.path.join('./genesets', basename)
            if os.path.exists(cand):
                gene_ref_path = cand

    pair=pd.read_csv(gene_ref_path,sep='\t',index_col=0)
    
    if keep_unmapped:
        # Keep all genes, map those that exist in the reference
        all_genes = data.index.tolist()
        mapped_genes = list(set(all_genes) & set(pair.index.tolist()))
        unmapped_genes = list(set(all_genes) - set(pair.index.tolist()))
        
        new_index = []
        
        # Process mapped genes
        for gene in all_genes:
            if gene in pair.index:
                symbol = pair.loc[gene, 'symbol']
                if str(symbol) == 'nan':
                    new_index.append(gene)  # Keep original ID if symbol is NaN
                else:
                    new_index.append(symbol)
            else:
                new_index.append(gene)  # Keep original ID for unmapped genes
        
        data.index = new_index
        print(f"......Mapped {len(mapped_genes)} genes to symbols, kept {len(unmapped_genes)} unmapped genes with original IDs")
    else:
        # Original behavior: only keep genes that can be mapped
        original_genes = data.index.tolist()
        ret_gene=list(set(original_genes) & set(pair.index.tolist()))
        data=data.loc[ret_gene]
        new_index=[]
        for i in ret_gene:
            a=pair.loc[i,'symbol']
            if str(a)=='nan':
                new_index.append(i)
            else:
                new_index.append(a)
        data.index=new_index
        print(f"......Mapped {len(ret_gene)} genes to symbols, removed {len(original_genes) - len(ret_gene)} unmapped genes")
    
    return data


def deseq2_normalize(data:pd.DataFrame)->pd.DataFrame:
    r"""Normalize the data using DESeq2 method.

    Arguments:
        data: The data to be normalized.
    
    Returns:
        data: The normalized data.

    """
    avg1=data.apply(np.log,axis=1).mean(axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    data1=data.loc[avg1.index]
    data_log=data1.apply(np.log,axis=1)
    scale=data_log.sub(avg1.values,axis=0).median(axis=0).apply(np.exp)
    return data/scale


def normalize_bulk(df_counts: pd.DataFrame, df_lengths: pd.DataFrame, normalization_type: str) -> pd.DataFrame:
    r"""Normalize the count data.

    Arguments:
        df_counts: Gene expression count matrix (number of cells x number of genes).
        df_lengths: Vector of gene lengths.
        normalization_type: Type of normalization (e.g., 'CPM', 'TPM', 'FPKM', 'RPKM').

    Returns:
        normalized_data: Normalized data as DataFrame
    """
    counts = df_counts.values
    lengths = df_lengths['feature_length'].astype(float).values.reshape(1, -1)  # Ensure lengths is a column vector
    
    if normalization_type == 'CPM':
        # Counts Per Million
        counts_per_million = counts / counts.sum(axis=1, keepdims=True) * 1e6
        return pd.DataFrame(counts_per_million, index=df_counts.index, columns=df_counts.columns)
    
    elif normalization_type == 'TPM':
        # Transcripts Per Million
        rate = counts / lengths
        tpm = rate / rate.sum(axis=1, keepdims=True) * 1e6
        return pd.DataFrame(tpm, index=df_counts.index, columns=df_counts.columns)
    
    elif normalization_type == 'FPKM' or normalization_type == 'RPKM':
        # Fragments Per Kilobase of transcript per Million mapped reads
        total_counts = counts.sum(axis=1, keepdims=True)
        fpkm = (counts / lengths) / total_counts * 1e9
        return pd.DataFrame(fpkm, index=df_counts.index, columns=df_counts.columns)
    
    else:
        raise ValueError("Unsupported normalization type. Choose from 'CPM', 'TPM', 'FPKM', 'RPKM'.")


def estimateSizeFactors(data:pd.DataFrame)->pd.Series:
    r"""Estimate size factors for data normalization.

    Arguments:
        data: A pandas DataFrame of gene expression data where rows correspond to samples and columns correspond to genes.
    
    Returns:
        scale: A pandas Series of size factors, one for each sample.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> import omicverse as ov
        >>> data = pd.DataFrame(np.random.rand(100, 10), columns=list('abcdefghij'))
        >>> size_factors = ov.bulk.estimateSizeFactors(data)
    """
    avg1=data.apply(np.log,axis=1).mean(axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    data1=data.loc[avg1.index]
    data_log=data1.apply(np.log,axis=1)
    scale=data_log.sub(avg1.values,axis=0).median(axis=0).apply(np.exp)
    return scale


def estimateDispersions(counts:pd.DataFrame)->pd.Series:
    r"""Estimate the dispersion parameter of the Negative Binomial distribution for each gene in the input count matrix.

    Arguments:
        counts: Input count matrix with shape (n_genes, n_samples).

    Returns:
        disp: Array of dispersion values for each gene in the input count matrix.
    """
    # Step 1: Calculate mean and variance of counts for each gene
    mean_counts = np.mean(counts, axis=1)
    var_counts = np.var(counts, axis=1)
    
    # Step 2: Fit trend line to variance-mean relationship using GLM
    mean_expr = sm.add_constant(np.log(mean_counts))
    mod = sm.GLM(np.log(var_counts), mean_expr, family=sm.families.Gamma())
    res = mod.fit()
    fitted_var = np.exp(res.fittedvalues)
    
    # Step 3: Calculate residual variance for each gene
    disp = fitted_var / var_counts
    
    return disp

def data_drop_duplicates_index(data:pd.DataFrame)->pd.DataFrame:
    r"""Drop the duplicated index of data.

    Arguments:
        data: The data to be processed.

    Returns:
        data: The data after dropping the duplicated index.
    """
    # Sort the data by the sum of counts in descending order
    data = data.loc[data.sum(axis=1).sort_values(ascending=False).index]
    
    # Drop duplicates, keeping the first occurrence (which is the highest due to sorting)
    data = data.loc[~data.index.duplicated(keep='first')]
    return data

@register_function(
    aliases=["差异表达分析", "DEG", "differential_expression", "差异基因分析", "pyDEG"],
    category="bulk", 
    description="Python implementation of differential expression analysis for bulk RNA-seq data",
    examples=[
        "# Initialize with raw count data",
        "dds = ov.bulk.pyDEG(raw_count_data)",
        "# Remove duplicate gene IDs",
        "dds.drop_duplicates_index()",
        "# Normalize using DESeq2 method",
        "dds.normalize()",
        "# Perform differential expression analysis",
        "dds.deg_analysis(treatment_groups, control_groups, method='DEseq2')",
        "# Set fold change thresholds",
        "dds.foldchange_set(fc_threshold=2, pval_threshold=0.05)",
        "# Visualize results",
        "dds.plot_volcano(title='DEG Analysis')",
        "dds.plot_boxplot(genes=['GENE1', 'GENE2'], treatment_groups, control_groups)",
        "# Prepare ranking for GSEA",
        "ranked_genes = dds.ranking2gsea()"
    ],
    related=["bulk.deseq2_normalize", "single.rank_genes_groups", "utils.volcano_plot"]
)
class pyDEG(object):
    """Differential-expression analysis helper for bulk RNA-seq count tables.

    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw count matrix with genes in rows and samples in columns.
    """


    def __init__(self,raw_data:pd.DataFrame) -> None:
        r"""Initialize the pyDEG class.

        Arguments:
            raw_data: The raw data to be processed.
        
        Returns:
            None
        """
        self.raw_data=raw_data
        self.data=raw_data.copy()
        
    def drop_duplicates_index(self)->pd.DataFrame:
        r"""Drop the duplicated index of data.

        Returns:
            data: The data after dropping the duplicated index.
        """
        self.data=data_drop_duplicates_index(self.data)
        return self.data

    def normalize(self)->pd.DataFrame:
        r"""Normalize the data using DESeq2 method.
        
        Returns:
            data: The normalized data.
        """
        self.size_factors=estimateSizeFactors(self.data)
        self.data=deseq2_normalize(self.data)
        return self.data
    
    def foldchange_set(self, fc_threshold: int = -1, pval_threshold: float = 0.05, logp_max: int = 6, fold_threshold: int = 0) -> None:
        r"""Set fold-change and p-value thresholds to classify differentially expressed genes as up-regulated, down-regulated, or not significant.

        Arguments:
            fc_threshold: Absolute fold-change threshold. If set to -1, the threshold is calculated based on the histogram of log2 fold-changes. (-1)
            pval_threshold: p-value threshold for determining significance. (0.05)
            logp_max: Maximum value for log-transformed p-values. (6)
            fold_threshold: Index of the histogram bin corresponding to the fold-change threshold (only applicable if fc_threshold=-1). (0)

        Returns:
            None
        """
        if fc_threshold==-1:
            foldp=np.histogram(self.result['log2FC'].dropna())
            foldchange=(foldp[1][np.where(foldp[1]>0)[0][fold_threshold]]+foldp[1][np.where(foldp[1]>0)[0][fold_threshold+1]])/2
        else:
            foldchange=fc_threshold
        print('... Fold change threshold: %s'%foldchange)
        fc_max,fc_min=foldchange,0-foldchange
        self.fc_max,self.fc_min=fc_max,fc_min
        self.pval_threshold=pval_threshold
        self.result['sig']='normal'
        self.result.loc[((self.result['log2FC']>fc_max)&(self.result['qvalue']<pval_threshold)),'sig']='up'
        self.result.loc[((self.result['log2FC']<fc_min)&(self.result['qvalue']<pval_threshold)),'sig']='down'
        self.result.loc[self.result['-log(qvalue)']>logp_max,'-log(qvalue)']=logp_max
        self.logp_max=logp_max
    

    def plot_volcano(self, figsize: tuple = (4, 4), pval_name: str = 'qvalue', fc_name: str = 'log2FC',
                     title: str = '', titlefont: dict = {'weight': 'normal', 'size': 14},
                     up_color: str = '#e25d5d', down_color: str = '#7388c1', normal_color: str = '#d7d7d7',
                     up_fontcolor: str = '#e25d5d', down_fontcolor: str = '#7388c1', normal_fontcolor: str = '#d7d7d7',
                     legend_bbox: tuple = (0.8, -0.2), legend_ncol: int = 2, legend_fontsize: int = 12,
                     plot_genes: Optional[list] = None, plot_genes_num: int = 10, plot_genes_fontsize: int = 10,
                     ticks_fontsize: int = 12, ax: Optional[matplotlib.axes._axes.Axes] = None) -> matplotlib.axes._axes.Axes:
        r"""Generate a volcano plot for the differential gene expression analysis results.

        Arguments:
            figsize: The size of the generated figure. ((4,4))
            pval_name: Column name for p-values. ('qvalue')
            fc_name: Column name for fold changes. ('log2FC')
            title: The title of the plot. ('')
            titlefont: A dictionary of font properties for the plot title. ({'weight':'normal','size':14,})
            up_color: The color of the up-regulated genes in the plot. ('#e25d5d')
            down_color: The color of the down-regulated genes in the plot. ('#7388c1')
            normal_color: The color of the non-significant genes in the plot. ('#d7d7d7')
            up_fontcolor: Font color for up-regulated gene labels. ('#e25d5d')
            down_fontcolor: Font color for down-regulated gene labels. ('#7388c1')
            normal_fontcolor: Font color for normal gene labels. ('#d7d7d7')
            legend_bbox: A tuple containing the coordinates of the legend's bounding box. ((0.8, -0.2))
            legend_ncol: The number of columns in the legend. (2)
            legend_fontsize: The font size of the legend. (12)
            plot_genes: A list of genes to be plotted on the volcano plot. (None)
            plot_genes_num: The number of genes to be plotted on the volcano plot. (10)
            plot_genes_fontsize: The font size of the genes to be plotted on the volcano plot. (10)
            ticks_fontsize: The font size of the ticks. (12)
            ax: Matplotlib axis object. (None)

        Returns:
            ax: The generated volcano plot.

        """
        
        ax=volcano(result=self.result,pval_name=pval_name,fc_name=fc_name,pval_max=self.logp_max,
                       figsize=figsize,title=title,titlefont=titlefont,
                       up_color=up_color,down_color=down_color,normal_color=normal_color,
                       up_fontcolor=up_fontcolor,down_fontcolor=down_fontcolor,normal_fontcolor=normal_fontcolor,
                       legend_bbox=legend_bbox,legend_ncol=legend_ncol,legend_fontsize=legend_fontsize,plot_genes=plot_genes,
                       plot_genes_num=plot_genes_num,plot_genes_fontsize=plot_genes_fontsize,
                       ticks_fontsize=ticks_fontsize,ax=ax,
                       pval_threshold=self.pval_threshold,fc_max=self.fc_max,fc_min=self.fc_min)
        return ax
        '''
        fig, ax = plt.subplots(figsize=figsize)
        result=self.result.copy()
        #首先绘制正常基因
        ax.scatter(x=result[result['sig']=='normal']['log2FC'],
                y=result[result['sig']=='normal']['-log(qvalue)'],
                color=normal_color,#颜色
                alpha=.5,#透明度
                )
        #接着绘制上调基因
        ax.scatter(x=result[result['sig']=='up']['log2FC'],
                y=result[result['sig']=='up']['-log(qvalue)'],
                color=up_color,#选择色卡第15个颜色
                alpha=.5,#透明度
                )
        #绘制下调基因
        ax.scatter(x=result[result['sig']=='down']['log2FC'],
                y=result[result['sig']=='down']['-log(qvalue)'],
                color=down_color,#颜色
                alpha=.5,#透明度
                )
        
        ax.plot([result['log2FC'].min(),result['log2FC'].max()],#辅助线的x值起点与终点
                [-np.log10(self.pval_threshold),-np.log10(self.pval_threshold)],#辅助线的y值起点与终点
                linewidth=2,#辅助线的宽度
                linestyle="--",#辅助线类型：虚线
                color='black'#辅助线的颜色
        )
        ax.plot([self.fc_max,self.fc_max],
                [result['-log(qvalue)'].min(),result['-log(qvalue)'].max()],
                linewidth=2, 
                linestyle="--",
                color='black')
        ax.plot([self.fc_min,self.fc_min],
                [result['-log(qvalue)'].min(),result['-log(qvalue)'].max()],
                linewidth=2, 
                linestyle="--",
                color='black')
        #设置横标签与纵标签
        ax.set_ylabel(r'$-log_{10}(qvalue)$',titlefont)                                    
        ax.set_xlabel(r'$log_{2}FC$',titlefont)
        #设置标题
        ax.set_title(title,titlefont)

        #绘制图注
        #legend标签列表，上面的color即是颜色列表
        labels = ['up:{0}'.format(len(result[result['sig']=='up'])),
                'down:{0}'.format(len(result[result['sig']=='down']))]  
        #用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend
        color = [up_color,down_color]
        patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i]) ) for i in range(len(color))] 

        ax.legend(handles=patches,
            bbox_to_anchor=legend_bbox, 
            ncol=legend_ncol,
            fontsize=legend_fontsize)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        from adjustText import adjust_text
        import adjustText
        
        if plot_genes is not None:
            hub_gene=plot_genes
        else:
            up_result=result.loc[result['sig']=='up']
            down_result=result.loc[result['sig']=='down']
            hub_gene=up_result.sort_values('qvalue').index[:plot_genes_num//2].tolist()+down_result.sort_values('qvalue').index[:plot_genes_num//2].tolist()

        color_dict={
        'up':up_fontcolor,
            'down':down_fontcolor,
            'normal':normal_fontcolor
        }

        texts=[ax.text(result.loc[i,'log2FC'], 
               result.loc[i,'-log(qvalue)'],
               i,
               fontdict={'size':plot_genes_fontsize,'weight':'bold','color':color_dict[result.loc[i,'sig']]}
               ) for i in hub_gene]
        
        if adjustText.__version__<='0.8':
            adjust_text(texts,only_move={'text': 'xy'},arrowprops=dict(arrowstyle='->', color='red'),)
        else:
            adjust_text(texts,only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
                        arrowprops=dict(arrowstyle='->', color='red'))

        ax.set_xticks([round(i,2) for i in ax.get_xticks()[1:-1]],#获取x坐标轴内容
              [round(i,2) for i in ax.get_xticks()[1:-1]],#更新x坐标轴内容
              fontsize=ticks_fontsize,
              fontweight='normal'
              )
        return fig,ax
        '''
    
    def plot_boxplot(self, genes: list, treatment_groups: list, control_groups: list,
                     log: bool = True,
                     treatment_name: str = 'Treatment', control_name: str = 'Control',
                     figsize: tuple = (4, 3), palette: list = ["#a64d79", "#674ea7"],
                     title: str = 'Gene Expression', fontsize: int = 12, legend_bbox: tuple = (1, 0.55), legend_ncol: int = 1,
                     **kwarg: Any) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
        r"""
        Plot the boxplot of genes from dds data

        Arguments:
            genes: The genes to plot.
            treatment_groups: The treatment groups.
            control_groups: The control groups.
            figsize: The figure size.
            palette: The color palette.
            title: The title of the plot.
            fontsize: The fontsize of the plot.
            legend_bbox: The bbox of the legend.
            legend_ncol: The number of columns of the legend.
            **kwarg: Other arguments for plot_boxplot function.
        
        Returns:
            fig: The figure of the plot.
            ax: The axis of the plot.
        """
        p_data=pd.DataFrame(columns=['Value','Gene','Type'])
        if log:
            for gene in genes:
                plot_data1=pd.DataFrame()
                plot_data1['Value']=np.log1p(self.data[treatment_groups].loc[gene].values)
                plot_data1['Gene']=gene
                plot_data1['Type']=treatment_name

                plot_data2=pd.DataFrame()
                plot_data2['Value']=np.log1p(self.data[control_groups].loc[gene].values)
                plot_data2['Gene']=gene
                plot_data2['Type']=control_name

                plot_data=pd.concat([plot_data1,plot_data2],axis=0)
                p_data=pd.concat([p_data,plot_data],axis=0)
        else:
            for gene in genes:
                plot_data1=pd.DataFrame()
                plot_data1['Value']=self.data[treatment_groups].loc[gene].values
                plot_data1['Gene']=gene
                plot_data1['Type']=treatment_name

                plot_data2=pd.DataFrame()
                plot_data2['Value']=self.data[control_groups].loc[gene].values
                plot_data2['Gene']=gene
                plot_data2['Type']=control_name

                plot_data=pd.concat([plot_data1,plot_data2],axis=0)
                p_data=pd.concat([p_data,plot_data],axis=0)

        fig,ax=plot_boxplot(p_data,hue='Type',x_value='Gene',y_value='Value',palette=palette,
                          figsize=figsize,fontsize=fontsize,title=title,
                          legend_bbox=legend_bbox,legend_ncol=legend_ncol, **kwarg)
        return fig,ax
    
    def ranking2gsea(self,rank_max:int=200,rank_min:int=274)->pd.DataFrame:
        r"""
        Ranking the result of dds data for gsea analysis

        Arguments:
            rank_max: The max rank of the result.
            rank_min: The min rank of the result.

        Returns:
            rnk: The ranking result.

        """


        result=self.result.copy()
        result['fcsign']=np.sign(result['log2FC'])
        result['logp']=-np.log10(result['pvalue'])
        result['metric']=result['logp']/result['fcsign']
        rnk=pd.DataFrame()
        rnk['gene_name']=result.index
        rnk['rnk']=result['metric'].values
        rnk=rnk.sort_values(by=['rnk'],ascending=False)
        k=1
        total=0
        for i in range(len(rnk)):
            if rnk.loc[i,'rnk']==np.inf: 
                total+=1
        #200跟274根据你的数据进行更改，保证inf比你数据最大的大，-inf比数据最小的小就好
        for i in range(len(rnk)):
            if rnk.loc[i,'rnk']==np.inf: 
                rnk.loc[i,'rnk']=rank_max+(total-k)
                k+=1
            elif rnk.loc[i,'rnk']==-np.inf: 
                rnk.loc[i,'rnk']=-(rank_min+k)
                k+=1
        return rnk

    def deg_analysis(self,group1:list,group2:list,
                 method:str='DEseq2',alpha:float=0.05,
                 multipletests_method:str='fdr_bh',n_cpus:int=8,
                 cooks_filter:bool=True, independent_filter:bool=True)->pd.DataFrame:
        r"""
        Differential expression analysis.

        Arguments:
            group1: The first group to be compared.
            group2: The second group to be compared.
            method: The method to be used for differential expression analysis.
                - `DEseq2`: PyDESeq2 negative-binomial Wald test (raw counts).
                - `edger`: edgeR quasi-likelihood F-test via the vendored
                  pure-Python `pyedger` port (R-parity tested; no R needed).
                - `limma`: limma-voom pipeline via the vendored pure-Python
                  `pylimma` port (voom mean-variance modelling + eBayes;
                  R-parity tested; no R needed).
                - `edgepy`: legacy edgeR-style GLM-LRT via the `inmoose`
                  package (requires `pip install inmoose`).
                - `ttest`: independent two-sample t-test.
                - `wilcox`: Wilcoxon rank-sum test (not implemented).
            alpha: The threshold of p-value.
            multipletests_method:
                - `bonferroni` : one-step correction
                - `sidak` : one-step correction
                - `holm-sidak` : step down method using Sidak adjustments
                - `holm` : step-down method using Bonferroni adjustments
                - `simes-hochberg` : step-up method  (independent)
                - `hommel` : closed method based on Simes tests (non-negative)
                - `fdr_bh` : Benjamini/Hochberg  (non-negative)
                - `fdr_by` : Benjamini/Yekutieli (negative)
                - `fdr_tsbh` : two stage fdr correction (non-negative)
                - `fdr_tsbky` : two stage fdr correction (non-negative)

        Returns
            result: The result of differential expression analysis.
        """
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
        from scipy.stats import ttest_ind
        from statsmodels.stats.multitest import multipletests
        print(f"⚙️ You are using {method} method for differential expression analysis.")
        if method=='ttest':
            
            data=self.data

            g1_mean=data[group1].mean(axis=1)
            g2_mean=data[group2].mean(axis=1)
            g=(g2_mean+g1_mean)/2
            g=g.loc[g>0].min()
            fold=(g1_mean+g)/(g2_mean+g)
            #log2fold=np.log2(fold)
            ttest = ttest_ind(data[group1].T.values, data[group2].T.values)
            pvalue=ttest[1]
            print(f"⏰ Start to calculate qvalue...")
            qvalue = multipletests(np.nan_to_num(np.array(pvalue),0), alpha=0.5, 
                               method=multipletests_method, is_sorted=False, returnsorted=False)
            #qvalue=fdrcorrection(np.nan_to_num(np.array(pvalue),0), alpha=0.05, method='indep', is_sorted=False)
            genearray = np.asarray(pvalue)
            result = pd.DataFrame({'pvalue':genearray,'qvalue':qvalue[1],'FoldChange':fold})
            result['MaxBaseMean']=np.max([g1_mean,g2_mean],axis=0)
            result['BaseMean']=(g1_mean+g2_mean)/2
            result['log2(BaseMean)']=np.log2((g1_mean+g2_mean)/2)
            result['log2FC'] = np.log2(result['FoldChange'])
            result['abs(log2FC)'] = abs(np.log2(result['FoldChange']))
            result['size']  =np.abs(result['FoldChange'])/10
            result=result.loc[~result['pvalue'].isnull()]
            result['-log(pvalue)'] = -np.log10(result['pvalue'])
            result['-log(qvalue)'] = -np.log10(result['qvalue'])
            #max mean of between each value in group1 and group2
            #result=result[result['padj']<alpha]
            result['sig']='normal'
            result.loc[(result['qvalue'] < alpha) & (result['log2FC'] > 0), 'sig'] = 'up'
            result.loc[(result['qvalue'] < alpha) & (result['log2FC'] < 0), 'sig'] = 'down'
            print(f"✅ Differential expression analysis completed.")
            
            self.result=result
            return result
        elif method=='wilcox':
            raise ValueError('The method is not supported.')
            print(f"⚙️ You are using {method} method for differential expression analysis.")
        elif method=='DEseq2':
            import pydeseq2
            counts_df = self.data[group1+group2].T
            clinical_df = pd.DataFrame(index=group1+group2)

            clinical_df['condition'] = ['Treatment'] * len(group1) + ['Control'] * len(group2)
            print(f"⏰ Start to create DeseqDataSet...")
            # Determine pydeseq2 version and create the DeseqDataSet accordingly
            if pydeseq2.__version__ <= '0.3.5':
                dds = DeseqDataSet(
                    counts=counts_df,
                    clinical=clinical_df,
                    design_factors="condition",  # compare samples based on "condition"
                    ref_level=["condition", "Control"],
                    refit_cooks=True,
                    n_cpus=n_cpus,
                )
            elif pydeseq2.__version__ <= '0.4.1':
                if ad.__version__ > '0.10.8':
                    raise ImportError(
                        'Please install the 0.10.8 version of anndata: `pip install anndata==0.10.8`.'
                    )
                dds = DeseqDataSet(
                    counts=counts_df,
                    metadata=clinical_df,
                    design_factors="condition",
                    refit_cooks=True,
                    n_cpus=n_cpus,
                )
            else:
                from pydeseq2.default_inference import DefaultInference
                inference = DefaultInference(n_cpus=n_cpus)
                dds = DeseqDataSet(
                    counts=counts_df,
                    metadata=clinical_df,
                    design_factors="condition",
                    refit_cooks=True,
                    inference=inference,
                )
        
            dds.fit_size_factors()
            dds.fit_genewise_dispersions()
            dds.fit_dispersion_trend()
            dds.fit_dispersion_prior()
            print(f"logres_prior={dds.uns['_squared_logres']}, sigma_prior={dds.uns['prior_disp_var']}")
            dds.fit_MAP_dispersions()
            dds.fit_LFC()
            dds.calculate_cooks()
            if dds.refit_cooks:
                dds.refit()
        
            # Add the 'contrast' parameter here:
        # FIX: Adding version check for DeseqStats constructor
            if pydeseq2.__version__<='0.3.5':
                stat_res = DeseqStats(dds, alpha=alpha, cooks_filter=cooks_filter, independent_filter=independent_filter)
            elif pydeseq2.__version__ <= '0.4.1':
                # For newer PyDESeq2 versions that require the contrast parameter
                stat_res = DeseqStats(dds, contrast=["condition", "Treatment", "Control"], 
                                    alpha=alpha, cooks_filter=cooks_filter, independent_filter=independent_filter)
                stat_res.run_wald_test()
                if stat_res.cooks_filter:
                    stat_res._cooks_filtering()
                    
                if stat_res.independent_filter:
                    stat_res._independent_filtering()
                else:
                    stat_res._p_value_adjustment()
            else:
                stat_res=DeseqStats(
                            dds,
                            contrast=["condition", "Treatment", "Control"], 
                            alpha=alpha,
                            cooks_filter=cooks_filter,
                            independent_filter=independent_filter,
                        )
                stat_res.run_wald_test()
                if stat_res.cooks_filter:
                    stat_res._cooks_filtering()
                    
                if stat_res.independent_filter:
                    stat_res._independent_filtering()
                else:
                    stat_res._p_value_adjustment()

                    
            self.stat_res = stat_res
            stat_res.summary()
            result = stat_res.results_df
            result['qvalue'] = result['padj']
            result['-log(pvalue)'] = -np.log10(result['pvalue'])
            result['-log(qvalue)'] = -np.log10(result['padj'])
            result['BaseMean'] = result['baseMean']
            result['log2(BaseMean)'] = np.log2(result['baseMean'] + 1)
            result['log2FC'] = result['log2FoldChange']
            result['abs(log2FC)'] = abs(result['log2FC'])
            result['sig'] = 'normal'
            result.loc[(result['qvalue'] < alpha) & (result['log2FC'] > 0), 'sig'] = 'up'
            result.loc[(result['qvalue'] < alpha) & (result['log2FC'] < 0), 'sig'] = 'down'
            self.result = result
            print(f"✅ Differential expression analysis completed.")
            return result
            
        
        elif method == 'edgepy':
            try:
                from inmoose.data.pasilla import pasilla
                from inmoose.edgepy import DGEList, glmLRT, topTags
                from patsy import dmatrix
            except:
                raise ImportError('Please install inmoose: `pip install inmoose`')
            print(f"⏰ Start to create DGEList...")
            anno1=pd.DataFrame(
                index=group1+group2
            )
            anno1['condition']=['treatment' for i in group1]+['control' for i in group2]
            var=pd.DataFrame(index=self.data.index)
            var.index.name='gene_id'

            # build a DGEList object
            dge_list = DGEList(
                counts=self.data[group1+group2].values, 
                samples=anno1, 
                group_col="condition", 
                genes=var
            )
            design1 = dmatrix("~condition", data=anno1)
            dge_list.estimateGLMCommonDisp(design=design1)
            fit = dge_list.glmFit(design=design1)
            lrt = glmLRT(fit)
            lrt.index=var.index

            #	log2FoldChange	lfcSE	logCPM	stat	pvalue		
            # 
            pvalue=lrt['pvalue'].values.reshape(-1)
            qvalue = multipletests(np.nan_to_num(np.array(pvalue),0), alpha=0.5, 
                               method=multipletests_method, is_sorted=False, returnsorted=False)
            
            g1_mean=self.data[group1].mean(axis=1)
            g2_mean=self.data[group2].mean(axis=1)
            g=(g2_mean+g1_mean)/2
            g=g.loc[g>0].min()
            fold=(g1_mean+g)/(g2_mean+g)
            print(f"⏰ Start to calculate qvalue...")

            result = pd.DataFrame({'pvalue':pvalue,'qvalue':qvalue[1],'FoldChange':fold})
            result['MaxBaseMean']=np.max([g1_mean,g2_mean],axis=0)
            result['BaseMean']=(g1_mean+g2_mean)/2
            result['log2(BaseMean)']=np.log2((g1_mean+g2_mean)/2)
            result['log2FC'] = np.log2(result['FoldChange'])
            result['abs(log2FC)'] = abs(result['log2FC'])
            result['size']  =np.abs(result['log2FC'])/10
            result=result.loc[~result['pvalue'].isnull()]
            result['-log(pvalue)'] = -np.log10(result['pvalue'])
            result['-log(qvalue)'] = -np.log10(result['qvalue'])
            #max mean of between each value in group1 and group2
            #result=result[result['padj']<alpha]
            result['sig']='normal'
            result.loc[(result['qvalue'] < alpha) & (result['log2FC'] > 0), 'sig'] = 'up'
            result.loc[(result['qvalue'] < alpha) & (result['log2FC'] < 0), 'sig'] = 'down'
            self.result=result
            print(f"✅ Differential expression analysis completed.")
            return result

        elif method == 'edger':
            # edgeR quasi-likelihood pipeline via the vendored pure-Python
            # port `pyedger` (omicverse.external.pyedger), R-parity tested
            # against Bioconductor edgeR. No R / inmoose needed.
            from ..external import pyedger as _edger
            print("⏰ Start edgeR quasi-likelihood (QL) pipeline (pyedger)...")
            counts = self.data[group1 + group2]
            groups = ['treatment'] * len(group1) + ['control'] * len(group2)
            # control < treatment alphabetically → control is the reference
            # level, so coef 'treatment' tests treatment vs control.
            y = _edger.calcNormFactors(_edger.DGEList(counts, group=groups))
            _edger.estimateDisp(y)
            fit = _edger.glmQLFit(y, legacy=False)
            qlf = _edger.glmQLFTest(fit, coef='treatment')
            tab = qlf.table                       # gene-ordered: logFC/logCPM/F/PValue

            pvalue = tab['PValue'].to_numpy()
            print(f"⏰ Start to calculate qvalue...")
            qvalue = multipletests(np.nan_to_num(np.array(pvalue), nan=1.0), alpha=alpha,
                                   method=multipletests_method, is_sorted=False,
                                   returnsorted=False)[1]
            g1_mean = self.data[group1].mean(axis=1)
            g2_mean = self.data[group2].mean(axis=1)

            result = pd.DataFrame({'pvalue': pvalue, 'qvalue': qvalue},
                                  index=self.data.index)
            result['log2FC'] = tab['logFC'].to_numpy()        # edgeR logFC is log2
            result['abs(log2FC)'] = result['log2FC'].abs()
            result['BaseMean'] = (g1_mean + g2_mean) / 2
            result['MaxBaseMean'] = np.max([g1_mean, g2_mean], axis=0)
            result['log2(BaseMean)'] = np.log2(result['BaseMean'] + 1)
            result['logCPM'] = tab['logCPM'].to_numpy()
            result['F'] = tab['F'].to_numpy()
            result['size'] = result['abs(log2FC)'] / 10
            result = result.loc[~result['pvalue'].isnull()]
            result['-log(pvalue)'] = -np.log10(result['pvalue'])
            result['-log(qvalue)'] = -np.log10(result['qvalue'])
            result['sig'] = 'normal'
            result.loc[(result['qvalue'] < alpha) & (result['log2FC'] > 0), 'sig'] = 'up'
            result.loc[(result['qvalue'] < alpha) & (result['log2FC'] < 0), 'sig'] = 'down'
            self.result = result
            print(f"✅ Differential expression analysis completed.")
            return result

        elif method == 'limma':
            # limma-voom pipeline via the vendored pure-Python port `pylimma`
            # (omicverse.external.pylimma), R-parity tested against
            # Bioconductor limma. voom() models the count mean-variance trend
            # — the statistically correct way to run limma on RNA-seq counts.
            from ..external import pylimma as _limma
            print("⏰ Start limma-voom pipeline (pylimma)...")
            counts = self.data[group1 + group2]
            n1, n2 = len(group1), len(group2)
            # ~0 + condition design; contrast [-1, 1] = treatment − control.
            design = pd.DataFrame(
                {'control':   [0] * n1 + [1] * n2,
                 'treatment': [1] * n1 + [0] * n2},
                index=group1 + group2,
            )
            v = _limma.voom(counts, design.values)
            fit = _limma.lmFit(v.E, design.values, weights=v.weights)
            fit = _limma.contrasts_fit(fit, np.array([-1.0, 1.0]))
            print(f"⏰ Start to adjust pvalue...")
            fit = _limma.eBayes(fit)

            log2FC = np.asarray(fit.coefficients)[:, 0]      # gene-ordered
            pvalue = np.asarray(fit.p_value)[:, 0]
            tstat = np.asarray(fit.t)[:, 0]
            print(f"⏰ Start to calculate qvalue...")
            qvalue = multipletests(np.nan_to_num(np.array(pvalue), nan=1.0), alpha=alpha,
                                   method=multipletests_method, is_sorted=False,
                                   returnsorted=False)[1]
            g1_mean = self.data[group1].mean(axis=1)
            g2_mean = self.data[group2].mean(axis=1)

            result = pd.DataFrame({'pvalue': pvalue, 'qvalue': qvalue},
                                  index=self.data.index)
            result['log2FC'] = log2FC                        # voom logFC is log2
            result['abs(log2FC)'] = result['log2FC'].abs()
            result['BaseMean'] = (g1_mean + g2_mean) / 2
            result['MaxBaseMean'] = np.max([g1_mean, g2_mean], axis=0)
            result['log2(BaseMean)'] = np.log2(result['BaseMean'] + 1)
            result['AveExpr'] = np.asarray(fit.Amean)
            result['t'] = tstat
            result['size'] = result['abs(log2FC)'] / 10
            result = result.loc[~result['pvalue'].isnull()]
            result['-log(pvalue)'] = -np.log10(result['pvalue'])
            result['-log(qvalue)'] = -np.log10(result['qvalue'])
            result['sig'] = 'normal'
            result.loc[(result['qvalue'] < alpha) & (result['log2FC'] > 0), 'sig'] = 'up'
            result.loc[(result['qvalue'] < alpha) & (result['log2FC'] < 0), 'sig'] = 'down'
            self.result = result
            print(f"✅ Differential expression analysis completed.")
            return result

        else:  # This is where the "method" check (not pydeseq2 version check) ends
            raise ValueError('The method is not supported.')

    def continuous_deg(self, trait, covariates=None, alpha: float = 0.05,
                       multipletests_method: str = 'fdr_bh',
                       data_type: str = 'auto') -> pd.DataFrame:
        r"""Differential expression against a CONTINUOUS sample-level trait.

        Where :meth:`deg_analysis` compares two groups, this models each
        gene's expression as a linear function of a **continuous**
        predictor — disease duration, age, dose, a severity score — via
        limma-voom + eBayes (the vendored pure-Python
        ``omicverse.external.pylimma``). It is the first-class tool for
        "which genes are associated with / track / correlate with a
        continuous variable": ``eBayes`` moderation borrows variance
        across genes (a per-gene Spearman/Pearson scan has none, and is
        noisiest in the tails that hold the top hits), and ``voom``
        models the count mean-variance trend.

        Arguments:
            trait: Per-sample value of the continuous trait — a
                ``pandas.Series`` indexed by sample, a ``dict``
                ``{sample: value}``, or an array aligned to the count
                columns. Samples with a missing (NaN) trait are dropped.
            covariates: Optional ``pandas.DataFrame`` (samples × columns)
                of nuisance variables to hold fixed — RIN, age, sex,
                batch, sequencing platform. Numeric columns enter the
                design directly; non-numeric columns are one-hot encoded.
            alpha: FDR threshold for the ``sig`` column (default 0.05).
            multipletests_method: ``statsmodels`` multiple-testing
                method for the q-value (default ``'fdr_bh'``).
            data_type: Nature of the expression matrix —
                ``'counts'`` (RNA-seq raw counts: ``voom`` removes the
                mean-variance trend, then weighted ``lmFit``);
                ``'continuous'`` (TPM / log-CPM / microarray or any
                already-normalized, log-scaled matrix: ``lmFit`` is run
                directly, no ``voom``, no weights); ``'auto'`` (default
                — an all-non-negative, near-integer matrix is treated
                as ``'counts'``, anything else as ``'continuous'``).
                Match this to your input: feeding a TPM / log matrix to
                the ``voom`` path is wrong — pass ``data_type='continuous'``
                (or raw counts) rather than hand-rolling a per-gene model.

        Returns:
            A genes × stats DataFrame (also stored on ``self.result``):
            ``log2FC`` here is the **slope** — log2 expression change
            per unit of the trait, covariates held fixed — plus
            ``pvalue``, ``qvalue``, ``t``, ``AveExpr`` and ``sig``
            (``up`` / ``down`` / ``normal`` by ``qvalue < alpha`` and
            slope sign). Column names match :meth:`deg_analysis` so
            ``foldchange_set`` / ``plot_volcano`` work unchanged.
        """
        from ..external import pylimma as _limma
        from statsmodels.stats.multitest import multipletests
        print("⏰ Start continuous-trait limma-voom pipeline (pylimma)...")

        samples = list(self.data.columns)

        # --- align the trait to the count-matrix columns --------------
        if isinstance(trait, dict):
            trait = pd.Series(trait)
        if isinstance(trait, pd.Series):
            trait_s = trait.reindex(samples).astype(float)
        else:
            trait_arr = np.asarray(trait, dtype=float).ravel()
            if len(trait_arr) != len(samples):
                raise ValueError(
                    f"trait has {len(trait_arr)} values but the count "
                    f"matrix has {len(samples)} samples; pass a Series "
                    f"indexed by sample to align unambiguously.")
            trait_s = pd.Series(trait_arr, index=samples)

        use_samples = [s for s in samples if pd.notna(trait_s.get(s))]
        if len(use_samples) < 3:
            raise ValueError(
                f"Only {len(use_samples)} samples have a non-missing trait "
                f"value — too few for a linear model.")
        trait_s = trait_s.loc[use_samples]

        # --- design matrix: Intercept + covariates + trait (trait last) -
        design = pd.DataFrame({'Intercept': 1.0}, index=use_samples)
        if covariates is not None:
            cov = covariates.reindex(use_samples)
            for col in cov.columns:
                series = cov[col]
                if pd.api.types.is_numeric_dtype(series):
                    design[col] = series.astype(float)
                else:  # one-hot encode a categorical covariate
                    dummies = pd.get_dummies(series.astype('category'),
                                             prefix=str(col), drop_first=True)
                    for d in dummies.columns:
                        design[d] = dummies[d].astype(float)
        trait_col = 'trait'
        while trait_col in design.columns:
            trait_col += '_'
        design[trait_col] = trait_s.astype(float)

        if bool(design.isnull().any().any()):
            bad = design.columns[design.isnull().any()].tolist()
            raise ValueError(
                f"design matrix has missing values in {bad} — covariate "
                f"values must be present for every used sample.")

        counts = self.data[use_samples]

        # --- resolve counts vs already-normalized (continuous) input --
        dt = str(data_type).lower().strip()
        if dt == 'auto':
            arr = np.asarray(counts.values, dtype=float)
            finite = arr[np.isfinite(arr)]
            is_counts = (finite.size > 0 and (finite >= 0).all()
                         and np.allclose(finite, np.round(finite)))
            dt = 'counts' if is_counts else 'continuous'
        if dt not in ('counts', 'continuous'):
            raise ValueError(
                f"data_type must be 'counts', 'continuous' or 'auto', "
                f"got {data_type!r}")

        # --- lmFit + eBayes -------------------------------------------
        if dt == 'counts':
            # RNA-seq counts: voom removes the mean-variance trend.
            print("⏰ Start continuous-trait limma-voom pipeline (pylimma)...")
            v = _limma.voom(counts, design.values)
            fit = _limma.lmFit(v.E, design.values, weights=v.weights)
        else:
            # Already-normalized / log-scaled (TPM, log-CPM, microarray):
            # lmFit directly — no voom, no precision weights.
            print("⏰ Start continuous-trait limma pipeline (continuous input)...")
            fit = _limma.lmFit(np.asarray(counts.values, dtype=float),
                               design.values)
        print("⏰ Start to adjust pvalue...")
        fit = _limma.eBayes(fit)

        trait_idx = list(design.columns).index(trait_col)
        log2FC = np.asarray(fit.coefficients)[:, trait_idx]
        pvalue = np.asarray(fit.p_value)[:, trait_idx]
        tstat = np.asarray(fit.t)[:, trait_idx]
        print("⏰ Start to calculate qvalue...")
        qvalue = multipletests(np.nan_to_num(np.asarray(pvalue, dtype=float), nan=1.0),
                               alpha=alpha, method=multipletests_method,
                               is_sorted=False, returnsorted=False)[1]

        base_mean = self.data[use_samples].mean(axis=1)
        result = pd.DataFrame({'pvalue': pvalue, 'qvalue': qvalue},
                              index=self.data.index)
        result['log2FC'] = log2FC          # slope: log2 change per unit trait
        result['abs(log2FC)'] = result['log2FC'].abs()
        result['BaseMean'] = base_mean
        result['MaxBaseMean'] = base_mean  # no two groups; kept for column parity
        result['log2(BaseMean)'] = np.log2(base_mean + 1)
        result['AveExpr'] = np.asarray(fit.Amean)
        result['t'] = tstat
        result['size'] = result['abs(log2FC)'] / 10
        result = result.loc[~result['pvalue'].isnull()]
        result['-log(pvalue)'] = -np.log10(result['pvalue'])
        result['-log(qvalue)'] = -np.log10(result['qvalue'])
        result['sig'] = 'normal'
        result.loc[(result['qvalue'] < alpha) & (result['log2FC'] > 0), 'sig'] = 'up'
        result.loc[(result['qvalue'] < alpha) & (result['log2FC'] < 0), 'sig'] = 'down'
        self.result = result
        print(f"✅ Continuous-trait DE complete: {len(use_samples)} samples, "
              f"{int((result['sig'] != 'normal').sum())} genes at q<{alpha}.")
        return result

    @staticmethod
    def _moderated_f_subset(fit, cols, df_total):
        r"""Moderated F-test over a SUBSET of design coefficients.

        pylimma's ``topTable`` only returns a per-coefficient moderated
        *t*-test; it does not expose a multi-column F-test. limma's
        ``classifyTestsF`` / ``topTableF`` build the F by *whitening* the
        moderated t-statistics with the eigen-decomposition of the
        coefficient *correlation* matrix and summing the squares. This
        helper replicates that procedure, but restricted to the columns
        in ``cols`` — i.e. the moderated F for "any of these coefficients
        is non-zero", which is exactly the time-course question (a block
        of spline / factor columns, or a block of interaction columns).

        It mirrors :func:`omicverse.external.pylimma.fit._classify_tests_f_stat`
        but slices ``fit.t`` and ``fit.cov_coefficients`` to ``cols``.

        Arguments:
            fit: An ``MArrayLM`` after ``eBayes`` (needs ``t`` and
                ``cov_coefficients``).
            cols: Integer column indices of the coefficient block to test.
            df_total: Per-gene total (moderated) residual d.f.
                (``fit.df_total``).

        Returns:
            ``(F, pvalue, df1)`` — per-gene moderated F statistic, its
            F-distribution survival p-value, and the numerator d.f.
        """
        from scipy import stats as _stats
        cols = list(cols)
        tsub = np.asarray(fit.t, dtype=float)[:, cols]      # genes × |cols|
        ntests = tsub.shape[1]
        if ntests == 1:
            f_stat = np.squeeze(tsub * tsub)
            df1 = 1
        else:
            cov = np.asarray(fit.cov_coefficients, dtype=float)
            cov = cov[np.ix_(cols, cols)].copy()
            diag = np.diag(cov)
            diag = np.where(diag == 0, 1.0, diag)
            cor = cov / np.sqrt(np.outer(diag, diag))
            vals, vecs = np.linalg.eigh(cor)
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            r = int(np.sum(vals / vals[0] > 1e-8)) if vals[0] > 0 else 1
            # q already carries the 1/sqrt(r) factor, so sum(z**2) == F.
            q = (vecs[:, :r] / np.sqrt(vals[:r])[None, :]) / np.sqrt(r)
            z = tsub @ q
            f_stat = np.sum(z * z, axis=1)
            df1 = r
        f_stat = np.asarray(f_stat, dtype=float)
        pvalue = _stats.f.sf(f_stat, df1, np.asarray(df_total, dtype=float))
        return f_stat, pvalue, df1

    def timecourse_deg(self, time, group=None, block=None, covariates=None,
                       time_basis: str = 'auto', spline_df: int = 3,
                       data_type: str = 'auto', alpha: float = 0.05,
                       multipletests_method: str = 'fdr_bh') -> pd.DataFrame:
        r"""Time-course / longitudinal differential-expression analysis.

        Where :meth:`deg_analysis` compares two groups and
        :meth:`continuous_deg` regresses on a single continuous slope,
        this method answers the three classic time-course questions with
        a *moderated F-test over a whole block of design columns* — the
        statistically correct way to ask "does expression change over
        time" without committing to a single shape. It is built on the
        vendored pure-Python limma (``omicverse.external.pylimma``)
        and is isomorphic to :meth:`continuous_deg`: same design-matrix
        construction, same ``lmFit → eBayes`` core, same
        ``deg_analysis``-shaped return so ``foldchange_set`` /
        ``plot_volcano`` keep working.

        The method (a port of splineTimeR's design) accepts **two kinds
        of input**, switched by ``data_type``:

        * RNA-seq **counts** — the mean-variance trend is removed with
          ``voom``, which feeds precision weights into ``lmFit``.
        * **continuous** expression — microarray log-ratios, log-CPM, or
          any already-normalized / log-scaled matrix. splineTimeR was
          originally a *microarray* time-course method, so this path
          runs ``lmFit`` directly on the expression matrix with no
          ``voom`` and no weights.

        Three first-class paths, selected by the arguments:

        1. **Temporal regulation** (``group=None``). Fit
           ``~ 1 + covariates + time_basis`` and report a moderated
           F-test over the *whole block* of time-basis columns — "which
           genes are temporally regulated", regardless of trajectory
           shape.
        2. **Group × time interaction** (``group=`` given). Fit
           ``~ group * time_basis`` and report the moderated F-test over
           the *interaction columns only* — "do the trajectories differ
           between groups" (e.g. treated vs control time courses).
        3. **Repeated measures** (``block=`` given). When the same
           subject is sampled at several time points, the within-subject
           correlation is estimated with ``duplicateCorrelation`` and
           passed as ``block`` / ``correlation`` to ``lmFit`` (limma's
           two-round longitudinal workflow). Combines with paths 1 or 2.

        Arguments:
            time: Per-sample time value — a ``pandas.Series`` indexed by
                sample, a ``dict`` ``{sample: time}``, or an array
                aligned to the count columns. Samples with a missing
                (NaN) time are dropped.
            group: Optional per-sample categorical group label (Series /
                dict / aligned array). When given, the analysis tests the
                group×time interaction (path 2). When ``None``, the
                single-group temporal test (path 1).
            block: Optional per-sample subject ID (Series / dict /
                aligned array) for repeated-measures designs — the same
                subject sampled at multiple times. Triggers the
                ``duplicateCorrelation`` workflow (path 3).
            covariates: Optional ``pandas.DataFrame`` (samples × columns)
                of nuisance variables to hold fixed. Numeric columns
                enter the design directly; non-numeric columns are
                one-hot encoded. Same handling as :meth:`continuous_deg`.
            time_basis: How to encode time —
                ``'spline'`` (natural cubic / restricted spline of
                ``spline_df`` degrees of freedom, ``patsy``'s
                ``cr(time, df=...)``); ``'factor'`` (one-hot of the
                discrete time points, drop-first); ``'auto'`` (default —
                ``factor`` if ≤4 unique time values, else ``spline``).
            spline_df: Degrees of freedom of the natural cubic spline
                when ``time_basis`` resolves to ``'spline'`` (default 3).
            data_type: Nature of the expression matrix —
                ``'counts'`` (RNA-seq raw counts: ``voom`` removes the
                mean-variance trend, then ``lmFit`` with precision
                weights); ``'continuous'`` (microarray log-ratios,
                log-CPM, or any already-normalized / log-scaled matrix:
                ``lmFit`` is run directly, no ``voom``, no weights);
                ``'auto'`` (default — inspect the matrix: an all
                non-negative, near-integer matrix is treated as
                ``'counts'``, anything with negative or non-integer
                values as ``'continuous'``).
            alpha: FDR threshold for the ``sig`` column (default 0.05).
            multipletests_method: ``statsmodels`` multiple-testing
                method for the q-value (default ``'fdr_bh'``).

        Returns:
            A genes × stats DataFrame (also stored on ``self.result``):
            ``F``, ``pvalue``, ``qvalue`` (the moderated F-test over the
            time block, or the interaction block); ``AveExpr``,
            ``BaseMean``, ``log2(BaseMean)``; ``-log(pvalue)`` /
            ``-log(qvalue)``; ``sig`` ∈ {``temporal``, ``normal``} by
            ``qvalue < alpha`` (an F-test has no direction, so the
            ``up`` / ``down`` of :meth:`deg_analysis` is replaced by
            ``temporal`` / ``normal``); plus the per-time-point (factor)
            or per-spline-term ``log2FC_<term>`` coefficient columns so
            the *shape* of the trajectory is recoverable. A plain
            ``log2FC`` column (the largest-magnitude time-block
            coefficient) is also added so ``plot_volcano`` works.
            Rows with a missing p-value are dropped.
        """
        from ..external import pylimma as _limma
        from statsmodels.stats.multitest import multipletests

        # --- resolve counts vs continuous input ------------------------
        if data_type not in ('auto', 'counts', 'continuous'):
            raise ValueError(
                f"data_type must be 'auto', 'counts' or 'continuous', "
                f"got {data_type!r}.")
        dt = data_type
        if dt == 'auto':
            vals = np.asarray(self.data.values, dtype=float)
            finite = vals[np.isfinite(vals)]
            near_int = (finite.size > 0
                        and np.allclose(finite, np.round(finite)))
            all_nonneg = finite.size > 0 and (finite >= 0).all()
            dt = 'counts' if (near_int and all_nonneg) else 'continuous'
            print(f"   data_type='auto' resolved to {dt!r} "
                  f"({'non-negative integers' if dt == 'counts' else 'has negative / non-integer values'}).")
        if dt == 'counts':
            print("⏰ Start time-course limma-voom pipeline (pylimma)...")
        else:
            print("⏰ Start time-course limma pipeline "
                  "(continuous input, no voom)...")

        samples = list(self.data.columns)

        def _align(x, name, dtype=None):
            """Align a per-sample variable to the count-matrix columns."""
            if x is None:
                return None
            if isinstance(x, dict):
                x = pd.Series(x)
            if isinstance(x, pd.Series):
                s = x.reindex(samples)
            else:
                arr = np.asarray(x).ravel()
                if len(arr) != len(samples):
                    raise ValueError(
                        f"{name} has {len(arr)} values but the count matrix "
                        f"has {len(samples)} samples; pass a Series indexed "
                        f"by sample to align unambiguously.")
                s = pd.Series(arr, index=samples)
            if dtype is not None:
                s = s.astype(dtype)
            return s

        time_s = _align(time, 'time', dtype=float)
        group_s = _align(group, 'group')
        block_s = _align(block, 'block')

        # --- keep samples with a non-missing time (and group/block) ----
        use_samples = [s for s in samples if pd.notna(time_s.get(s))]
        if group_s is not None:
            use_samples = [s for s in use_samples if pd.notna(group_s.get(s))]
        if block_s is not None:
            use_samples = [s for s in use_samples if pd.notna(block_s.get(s))]
        if len(use_samples) < 4:
            raise ValueError(
                f"Only {len(use_samples)} samples usable — too few for a "
                f"time-course model.")
        time_s = time_s.loc[use_samples]
        if group_s is not None:
            group_s = group_s.loc[use_samples].astype('category')
        if block_s is not None:
            block_s = block_s.loc[use_samples]

        # --- resolve the time basis -----------------------------------
        n_time = int(time_s.nunique())
        basis = time_basis
        if basis == 'auto':
            basis = 'factor' if n_time <= 4 else 'spline'
        if basis not in ('factor', 'spline'):
            raise ValueError(
                f"time_basis must be 'auto', 'factor' or 'spline', "
                f"got {time_basis!r}.")
        if n_time < 2:
            raise ValueError("Need at least 2 distinct time points.")

        if basis == 'factor':
            time_dum = pd.get_dummies(
                pd.Categorical(time_s, categories=sorted(time_s.unique())),
                prefix='time', drop_first=True)
            time_dum.index = use_samples
            time_block = time_dum.astype(float)
        else:  # natural cubic / restricted spline
            import patsy
            sdf = int(spline_df)
            if sdf >= n_time:
                sdf = max(1, n_time - 1)
                print(f"   spline_df reduced to {sdf} (only {n_time} "
                      f"distinct time points).")
            # patsy's cr() restricted-cubic-spline span *includes* the
            # constant: cr(t, df=k) has k effective df total, one of which
            # is the intercept. Request df=sdf+1 and drop patsy's
            # intercept so the remaining columns carry exactly `sdf`
            # shape degrees of freedom on top of our own Intercept.
            dm = patsy.dmatrix(f"cr(_t, df={sdf + 1})",
                               {'_t': time_s.values},
                               return_type='dataframe')
            if 'Intercept' in dm.columns:
                dm = dm.drop(columns='Intercept')
            dm.columns = [f"time_s{i}" for i in range(dm.shape[1])]
            dm.index = use_samples
            time_block = dm.astype(float)
        time_cols = list(time_block.columns)

        # Drop any time-basis column that is collinear with the constant
        # plus the columns already kept (a restricted-cubic-spline basis
        # can carry a hidden constant component). This keeps exactly the
        # independent time degrees of freedom for the F-test.
        kept, kmat = [], np.ones((len(use_samples), 1))
        for c in time_cols:
            cand = np.hstack([kmat, time_block[[c]].values])
            if np.linalg.matrix_rank(cand) > kmat.shape[1]:
                kept.append(c)
                kmat = cand
        if len(kept) < len(time_cols):
            dropped = [c for c in time_cols if c not in kept]
            print(f"   dropped {len(dropped)} collinear time-basis "
                  f"column(s): {dropped}")
        time_block = time_block[kept]
        time_cols = kept
        if not time_cols:
            raise ValueError("time basis has no independent columns.")

        # --- assemble the design matrix -------------------------------
        design = pd.DataFrame({'Intercept': 1.0}, index=use_samples)

        # covariates: numeric in directly, categorical one-hot.
        if covariates is not None:
            cov = covariates.reindex(use_samples)
            for col in cov.columns:
                series = cov[col]
                if pd.api.types.is_numeric_dtype(series):
                    design[col] = series.astype(float)
                else:
                    dummies = pd.get_dummies(series.astype('category'),
                                             prefix=str(col), drop_first=True)
                    for d in dummies.columns:
                        design[d] = dummies[d].astype(float)

        if group_s is None:
            # path 1 — single-group temporal test.
            for c in time_cols:
                design[c] = time_block[c]
            test_cols = list(time_cols)
            mode = 'temporal'
        else:
            # path 2 — group × time interaction test.
            grp_dum = pd.get_dummies(group_s, prefix='group', drop_first=True)
            grp_dum.index = use_samples
            grp_cols = list(grp_dum.columns)
            for c in grp_cols:
                design[c] = grp_dum[c].astype(float)
            for c in time_cols:
                design[c] = time_block[c]
            # interaction columns: each group dummy × each time-basis col.
            inter_cols = []
            for gc in grp_cols:
                for tc in time_cols:
                    name = f"{gc}:{tc}"
                    design[name] = (design[gc] * design[tc]).astype(float)
                    inter_cols.append(name)
            test_cols = inter_cols
            mode = 'interaction'

        if bool(design.isnull().any().any()):
            bad = design.columns[design.isnull().any()].tolist()
            raise ValueError(
                f"design matrix has missing values in {bad} — covariate / "
                f"group / time values must be present for every used sample.")
        # guard against a rank-deficient design (e.g. confounded covariate).
        if np.linalg.matrix_rank(design.values) < design.shape[1]:
            raise ValueError(
                "design matrix is rank-deficient — a covariate or group is "
                "confounded with time; drop the offending term.")

        counts = self.data[use_samples]
        test_idx = [list(design.columns).index(c) for c in test_cols]

        # --- fit the linear model -------------------------------------
        # 'counts'     : voom removes the mean-variance trend, then lmFit
        #                runs with the voom precision weights.
        # 'continuous' : the matrix is already log-scaled / normalized —
        #                lmFit runs directly on it, no voom, no weights.
        if dt == 'counts':
            if block_s is not None:
                block_arr = block_s.values
                print("⏰ Repeated-measures design — estimating "
                      "within-subject correlation (duplicateCorrelation)...")
                # Round 1: voom without weights-aware corr, estimate corr.
                v = _limma.voom(counts, design.values)
                dc = _limma.duplicateCorrelation(v.E, design.values,
                                                 block=block_arr,
                                                 weights=v.weights)
                corr = float(dc.consensus_correlation)
                # Round 2: voom with the block correlation, re-estimate.
                try:
                    v = _limma.voom(counts, design.values,
                                    block=block_arr, correlation=corr)
                    dc = _limma.duplicateCorrelation(v.E, design.values,
                                                     block=block_arr,
                                                     weights=v.weights)
                    corr = float(dc.consensus_correlation)
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"   voom(block=...) round 2 skipped ({exc}); "
                          f"using round-1 correlation.")
                print(f"   consensus within-subject correlation = {corr:.4f}")
                fit = _limma.lmFit(v.E, design.values, weights=v.weights,
                                   block=block_arr, correlation=corr)
                self.timecourse_correlation = corr
            else:
                v = _limma.voom(counts, design.values)
                fit = _limma.lmFit(v.E, design.values, weights=v.weights)
        else:
            # continuous expression — lmFit directly on the matrix.
            expr = counts.astype(float)
            if block_s is not None:
                block_arr = block_s.values
                print("⏰ Repeated-measures design — estimating "
                      "within-subject correlation (duplicateCorrelation)...")
                dc = _limma.duplicateCorrelation(expr.values, design.values,
                                                 block=block_arr)
                corr = float(dc.consensus_correlation)
                print(f"   consensus within-subject correlation = {corr:.4f}")
                fit = _limma.lmFit(expr.values, design.values,
                                   block=block_arr, correlation=corr)
                self.timecourse_correlation = corr
            else:
                fit = _limma.lmFit(expr.values, design.values)

        print("⏰ Start to adjust pvalue (eBayes)...")
        fit = _limma.eBayes(fit)

        # --- moderated F-test over the tested coefficient block --------
        f_stat, pvalue, df1 = self._moderated_f_subset(
            fit, test_idx, fit.df_total)
        f_stat = np.asarray(f_stat, dtype=float)
        pvalue = np.asarray(pvalue, dtype=float)

        print("⏰ Start to calculate qvalue...")
        qvalue = multipletests(np.nan_to_num(pvalue, nan=1.0),
                               alpha=alpha, method=multipletests_method,
                               is_sorted=False, returnsorted=False)[1]

        base_mean = self.data[use_samples].mean(axis=1)
        result = pd.DataFrame({'F': f_stat, 'pvalue': pvalue, 'qvalue': qvalue},
                              index=self.data.index)
        result['AveExpr'] = np.asarray(fit.Amean)
        result['BaseMean'] = base_mean
        result['MaxBaseMean'] = base_mean      # kept for column parity
        result['log2(BaseMean)'] = np.log2(base_mean + 1)

        # per-term coefficients of the tested block — the trajectory shape.
        coefs = np.asarray(fit.coefficients)
        for c, idx in zip(test_cols, test_idx):
            result[f"log2FC_{c}"] = coefs[:, idx]
        # a single log2FC (largest-magnitude tested coefficient) so the
        # existing volcano plotting still has something to draw.
        block_coefs = coefs[:, test_idx]
        amax = np.argmax(np.abs(block_coefs), axis=1)
        result['log2FC'] = block_coefs[np.arange(block_coefs.shape[0]), amax]
        result['abs(log2FC)'] = result['log2FC'].abs()
        result['size'] = result['abs(log2FC)'] / 10

        result = result.loc[~result['pvalue'].isnull()]
        result['-log(pvalue)'] = -np.log10(result['pvalue'])
        result['-log(qvalue)'] = -np.log10(result['qvalue'])
        result['sig'] = 'normal'
        result.loc[result['qvalue'] < alpha, 'sig'] = 'temporal'

        self.result = result
        n_hit = int((result['sig'] == 'temporal').sum())
        if mode == 'interaction':
            print(f"✅ Time-course DE (group×time interaction) complete: "
                  f"{len(use_samples)} samples, {basis} basis ({len(test_cols)} "
                  f"interaction df), {n_hit} genes with differing trajectories "
                  f"at q<{alpha}.")
        else:
            print(f"✅ Time-course DE (temporal regulation) complete: "
                  f"{len(use_samples)} samples, {basis} basis ({len(test_cols)} "
                  f"time df), {n_hit} temporally-regulated genes at q<{alpha}.")
        return result


def _temporal_knee(xs, ys):
    """Knee of a curve — the point farthest from the chord joining its
    first and last points (orientation-agnostic elbow detection)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) <= 2:
        return int(xs[0])
    x = (xs - xs.min()) / (np.ptp(xs) or 1.0)
    y = (ys - ys.min()) / (np.ptp(ys) or 1.0)
    dist = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y
                  + x[-1] * y[0] - y[-1] * x[0])
    return int(xs[int(np.argmax(dist))])


@register_function(
    aliases=["temporal_clusters", "时序聚类", "temporal_clustering",
             "soft_temporal_clusters"],
    category="bulk",
    description=(
        "Soft-cluster the temporal expression trajectories of time-course "
        "genes by fuzzy c-means (Mfuzz). Pairs with pyDEG.timecourse_deg — "
        "after finding which genes are temporally regulated, this groups "
        "them by trajectory shape (monotone rise / fall, transient). "
        "Backed by the pure-Python pymfuzz port of Bioconductor Mfuzz."
    ),
    examples=[
        "ov.bulk.temporal_clusters(data, time, genes=deg.index[deg['sig']=='temporal'])",
        "ov.bulk.temporal_clusters(data, time, n_clusters=9, plot=True)",
    ],
    related=["bulk.pyDEG"],
)
def temporal_clusters(data, time, *, genes=None, n_clusters="auto",
                      m="auto", agg="mean", crange=None, seed=0,
                      plot=False):
    r"""Soft-cluster temporal expression trajectories by fuzzy c-means.

    Companion to :meth:`pyDEG.timecourse_deg`. ``timecourse_deg`` answers
    *which* genes change over time; ``temporal_clusters`` answers *what
    shape* those changes take — grouping the temporally regulated genes
    into soft clusters of co-trending trajectories (monotone rise / fall,
    transient up-then-down, ...) via Mfuzz fuzzy c-means on the
    z-standardised, replicate-averaged time profiles.

    Arguments:
        data: Expression matrix — a genes x samples ``pandas.DataFrame``
            (normalised expression is recommended), or an ``AnnData``
            (samples x genes, transposed internally).
        time: Per-sample time value — a ``pandas.Series`` indexed by
            sample, a ``dict`` ``{sample: time}``, or an array aligned to
            the sample columns.
        genes: Optional subset of genes to cluster — typically the
            temporally regulated genes from ``timecourse_deg``
            (``deg.index[deg['sig'] == 'temporal']``). Default: all genes.
        n_clusters: Number of soft clusters — an int, or ``'auto'`` to
            pick the knee of the Mfuzz ``Dmin`` curve.
        m: Fuzzifier — a float, or ``'auto'`` for Mfuzz ``mestimate``.
        agg: How to collapse replicate samples sharing a time point —
            ``'mean'`` (default) or ``'median'``.
        crange: Candidate cluster counts scanned when ``n_clusters='auto'``
            (default ``range(2, 13)``).
        seed: RNG seed for the fuzzy c-means initialisation.
        plot: If True, draw the Mfuzz soft-cluster trajectory grid.

    Returns:
        A genes x stats ``pandas.DataFrame`` indexed by gene: ``cluster``
        (1-based hard assignment) and ``membership`` (the gene's
        membership in its assigned cluster). ``.attrs`` carries
        ``centers`` (clusters x time points — the trajectory templates),
        ``membership_matrix`` (genes x clusters), ``m`` and ``n_clusters``.
    """
    try:
        import pymfuzz
    except ImportError as exc:
        raise ImportError(
            "temporal_clusters needs the 'pymfuzz' package — "
            "install it with:  pip install pymfuzz"
        ) from exc

    # --- coerce expression to a genes x samples DataFrame --------------
    if isinstance(data, ad.AnnData):
        expr = data.to_df().T          # AnnData is samples x genes
    else:
        expr = pd.DataFrame(data).copy()
    samples = list(expr.columns)

    # --- align the time vector to the sample columns -------------------
    if isinstance(time, dict):
        time = pd.Series(time)
    if isinstance(time, pd.Series):
        time_s = time.reindex(samples).astype(float)
    else:
        arr = np.asarray(time, dtype=float).ravel()
        if len(arr) != len(samples):
            raise ValueError(
                f"time has {len(arr)} values but data has {len(samples)} "
                f"samples; pass a Series indexed by sample to align.")
        time_s = pd.Series(arr, index=samples)
    keep = list(time_s.dropna().index)
    expr, time_s, samples = expr[keep], time_s.loc[keep], keep

    # --- restrict to the genes of interest ----------------------------
    if genes is not None:
        genes = [g for g in pd.Index(genes) if g in expr.index]
        if len(genes) < 2:
            raise ValueError("Need at least 2 of the requested genes "
                             "present in data.")
        expr = expr.loc[genes]

    # --- collapse replicates -> one mean profile per time point -------
    times = sorted(time_s.unique())
    if len(times) < 3:
        raise ValueError("Temporal clustering needs >=3 distinct time points.")
    reducer = np.nanmedian if agg == "median" else np.nanmean
    traj = pd.DataFrame(
        {t: reducer(expr[[s for s in samples if time_s[s] == t]]
                    .to_numpy(dtype=float), axis=1)
         for t in times},
        index=expr.index,
    )
    traj = traj.loc[traj.notna().all(axis=1) & (traj.std(axis=1) > 0)]
    if traj.shape[0] < 2:
        raise ValueError("Too few genes with a usable (non-flat) trajectory.")

    # --- standardise (z-score each gene), estimate the fuzzifier ------
    z = pymfuzz.standardise(traj)
    m_val = pymfuzz.mestimate(z) if m == "auto" else float(m)

    # --- choose the cluster count -------------------------------------
    if isinstance(n_clusters, str) and n_clusters == "auto":
        cr = list(crange) if crange is not None else list(range(2, 13))
        cr = [c for c in cr if 2 <= c < traj.shape[0]]
        dmin = np.asarray(
            pymfuzz.Dmin(z, m=m_val, crange=cr, repeats=3, visu=False,
                         random_state=seed), dtype=float)
        c_val = _temporal_knee(cr, dmin)
    else:
        c_val = int(n_clusters)
    print(f"⏰ Mfuzz fuzzy c-means: {traj.shape[0]} genes, "
          f"{len(times)} time points, c={c_val}, m={m_val:.3f}")

    # --- fuzzy c-means soft clustering --------------------------------
    fc = pymfuzz.mfuzz(z, c=c_val, m=m_val, random_state=seed)
    membership = np.asarray(fc.membership, dtype=float)   # genes x clusters
    hard = membership.argmax(axis=1) + 1                  # 1-based
    cl_cols = [f"cluster_{i}" for i in fc.cluster_names]

    out = pd.DataFrame(
        {"cluster": hard, "membership": membership.max(axis=1)},
        index=list(fc.gene_names),
    )
    out.attrs["centers"] = pd.DataFrame(
        np.asarray(fc.centers, dtype=float),
        index=cl_cols, columns=list(fc.time_names))
    out.attrs["membership_matrix"] = pd.DataFrame(
        membership, index=list(fc.gene_names), columns=cl_cols)
    out.attrs["m"] = m_val
    out.attrs["n_clusters"] = c_val

    if plot:
        pymfuzz.mfuzz_plot(z, fc)
    return out

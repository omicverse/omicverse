import math
import requests
from time import sleep
import numpy as np
import pandas as pd
import scipy.stats as stats
import statistics
import sys
import warnings
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram, fcluster
from scipy.cluster.hierarchy import to_tree,set_link_color_palette
from scipy.stats import t
from scipy.stats import rankdata
from statsmodels.formula.api import ols
from matplotlib import colors as mcolors
from sklearn.impute import KNNImputer
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx

from ..gseapy.plot import dotplot


from pandas.api.types import is_numeric_dtype

from .geneExp import *
from ..._registry import register_function



# remove runtime warning (divided by zero)
np.seterr(divide='ignore', invalid='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")


# public values
networkTypes = ["unsigned", "signed", "signed hybrid"]
adjacencyTypes = ["unsigned", "signed", "signed hybrid"]
TOMTypes = ["unsigned", "signed"]
TOMDenoms = ["min", "mean"]

# bcolors
HEADER = "\033[95m"
OKBLUE = "\033[94m"
OKCYAN = "\033[96m"
OKGREEN = "\033[92m"
WARNING = "\033[93m"
FAIL = "\033[91m"
ENDC = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"

@register_function(
    aliases=["WGCNA分析", "pyWGCNA", "wgcna_analysis", "加权基因共表达网络"],
    category="bulk",
    description="Weighted Gene Co-expression Network Analysis (WGCNA) for identifying gene modules",
    examples=[
        "# Initialize WGCNA analysis",
        "wgcna = ov.bulk.pyWGCNA(name='MyAnalysis', species='mus musculus', geneExp=data.T)",
        "# Preprocessing workflow",
        "wgcna.preprocess()",
        "# Calculate soft threshold",
        "wgcna.calculate_soft_threshold()",
        "# Calculate adjacency matrix",
        "wgcna.calculating_adjacency_matrix()",
        "# Calculate TOM similarity matrix",
        "wgcna.calculating_TOM_similarity_matrix()",
        "# Build co-expression network",
        "wgcna.calculate_geneTree()",
        "wgcna.calculate_dynamicMods()",
        "wgcna.calculate_gene_module()",
        "# Visualize network",
        "wgcna.plot_matrix()",
        "# Save and load analysis",
        "wgcna.saveWGCNA()",
        "# Get submodules and networks",
        "sub_genes = wgcna.get_sub_module(['lightgreen'], mod_type='module_color')",
        "sub_network = wgcna.get_sub_network(['lightgreen'], correlation_threshold=0.2)",
        "wgcna.plot_sub_network(['lightgreen'], figsize=(8,8))",
        "# Sample information and analysis",
        "wgcna.updateSampleInfo('sample_info.csv')",
        "wgcna.setMetadataColor('condition', {'Control': 'blue', 'Treatment': 'red'})",
        "wgcna.analyseWGCNA()",
        "wgcna.plotModuleEigenGene('lightgreen', metadata)",
        "wgcna.barplotModuleEigenGene('lightgreen', metadata)",
        "# Find hub genes",
        "hub_genes = wgcna.top_n_hub_genes('lightgreen', n=10)"
    ],
    related=["bulk.readWGCNA", "utils.read", "pl.network"]
)
class pyWGCNA(GeneExp):
    """
    A class used to do weighted gene co-expression network analysis.

    :param name: name of the WGCNA we used to visualize data (default: 'WGCNA')
    :type name: str
    :param save: indicate if you want to save result of important steps in a figure directory (default: False)
    :type save: bool
    :param species: species of the data you use i.e mouse, human
    :type species: str
    :param level: which type of data you use including gene, transcript (default: gene)
    :type level: str
    :param outputPath: path you want to save all you figures and object (default: '', where you rau your script)
    :type outputPath: str
    :param anndata: if the expression data is in anndata format you should pass it through this parameter. X should be expression matrix. var is a gene information and obs is a sample information.
    :type anndata: anndata
    :param geneExp: expression matrix which genes are in the rows and samples are columns
    :type geneExp: pandas dataframe
    :param geneExpPath: path of expression matrix
    :type geneExpPath: str
    :param sep: separation symbol to use for reading data in geneExpPath properly
    :type sep: str
    :param geneInfo: dataframe that contains genes information it should have a same index as gene expression column names (gene/transcript ID)
    :type geneInfo: pandas dataframe
    :param sampleInfo: dataframe that contains samples information it should have a same index as gene expression index (sample ID)
    :type sampleInfo: pandas dataframe
    :param TPMcutoff: cut off for removing genes that expressed under this number along samples
    :type TPMcutoff: int
    :param cut: number to remove outlier sample (default: 'inf') By default we don't remove any sample by hierarchical clustering
    :type cut: float
    :param powers: different powers to test to have scale free network (default: [1:10, 11:21:2])
    :type powers: list of int
    :param RsquaredCut: R squaered cut to choose power for having scale free network; between 0 to 1 (default: 0.9)
    :type RsquaredCut: float
    :param MeanCut: mean connectivity to choose power for having scale free network (default: 100)
    :type MeanCut: int
    :param networkType: Type of network we can create including "unsigned", "signed" and "signed hybrid" (default: "signed hybrid")
    :type networkType: str
    :param TOMType: Type of topological overlap matrix(TOM) including "unsigned", "signed" (default: "signed")
    :type TOMType: str
    :param minModuleSize: We like large modules, so we set the minimum module size relatively high (default: 50)
    :type minModuleSize: int
    :param naColor: color we used to identify genes we don't find any cluster for them (default: "grey")
    :type naColor: str
    :param MEDissThres:  diss similarity threshold (default: 0.2)
    :type MEDissThres: float
    :param figureType: extension of figure (default: "pdf")
    :type figureType: str
    :param MEs: eigengenes
    :type MEs: ndarray
    :param geneExpr: gene expression object that contains raw gene expression along with gene and sample information.
    :type geneExpr: geneExp class
    :param datExpr: data expression data that contains preprocessed data
    :type datExpr: anndata
    :param dynamicMods: name of modules by clustering similar genes together
    :type dynamicMods: list
    :param TOM: topological overlap measure using average linkage hierarchical clustering which inputs a measure of interconnectedness
    :param TOM: ndarray
    :param adjacency: adjacency matrix calculating base of the type of network
    :type adjacency: ndarray
    :param geneTree: average hierarchical clustering of dissTOM matrix
    :type geneTree: ndarray
    :param power: power to have scale free network (default: 6)
    :type power: int
    :param sft: soft threshold table which has information for each powers
    :type sft: pandas dataframe
    :param datME:
    :type datME: pandas dataframe
    :param signedKME:(signed) eigengene-based connectivity (module membership)
    :type signedKME: pandas dataframe
    :param moduleTraitCor: correlation between each module and metadata
    :type moduleTraitCor: pandas dataframe
    :param moduleTraitPvalue: p-value of correlation between each module and metadata
    :type moduleTraitPvalue: pandas dataframe
    """

    def __init__(self, name='WGCNA',
                 TPMcutoff=1,
                 powers=None, RsquaredCut=0.85, MeanCut=100,
                 networkType="signed hybrid", TOMType="signed",
                 minModuleSize=50, naColor="grey", cut=float('inf'),
                 MEDissThres=0.2,
                 species=None, level='gene', anndata=None, geneExp=None,
                 geneExpPath=None, sep=',', geneInfo=None, sampleInfo=None,
                 save=False, outputPath=None, figureType='pdf'):

        super().__init__(species=species, level=level, anndata=anndata, geneExp=geneExp,
                         geneExpPath=geneExpPath, sep=sep, geneInfo=geneInfo, sampleInfo=sampleInfo)

        if powers is None:
            powers = list(range(1, 11)) + list(range(11, 21, 2))

        self.name = name

        self.save = save
        self.outputPath = f"{os.getcwd()}/" if outputPath is None else outputPath

        self.TPMcutoff = TPMcutoff
        self.cut = cut

        self.datExpr = self.geneExpr.copy()

        self.metadataColors = {}

        self.networkType = networkType

        # Choose a set of soft-thresholding powers
        self.RsquaredCut = RsquaredCut
        self.MeanCut = MeanCut
        self.powers = powers
        self.power = 0
        self.sft = None

        self.geneTree = None
        self.adjacency = None
        self.TOMType = TOMType
        self.TOM = None
        self.minModuleSize = minModuleSize
        self.dynamicMods = [None]
        self.naColor = naColor
        self.MEs = None
        self.MEDissThres = MEDissThres

        self.datME = None
        self.signedKME = None

        self.moduleTraitCor = None
        self.moduleTraitPvalue = None

        if self.save:
            print(f"{OKGREEN}Saving data to be True, checking requirements ...{ENDC}")
            if not os.path.exists(self.outputPath + 'figures/'):
                print(f"{WARNING}Figure directory does not exist!\nCreating figure directory!{ENDC}")
                os.makedirs(self.outputPath + 'figures/')

        self.figureType = figureType

    def preprocess(self, show=True):
        """
        Preprocessing PyWGCNA object including removing obvious outlier on genes and samples

        :param show: indicate if you want to show your plot or not (if you put this to False it will not either show and save the plot)
        :type show: bool
        """
        print(f"{BOLD}{OKBLUE}Pre-processing...{ENDC}")

        # Prepare and clean data
        # Remove cols with less than 1 TPM
        self.datExpr = self.datExpr[:, (self.datExpr.to_df() > self.TPMcutoff).any(axis=0)]

        # Check that all genes and samples have sufficiently low numbers of missing values.
        goodGenes, goodSamples, allOK = pyWGCNA.goodSamplesGenes(self.datExpr.to_df().T)
        # if not okay
        if not allOK:
            # Optionally, print the gene and sample names that were removed:
            if np.size(goodGenes) - np.count_nonzero(goodGenes) > 0:
                print(
                    f"{OKGREEN} {np.size(goodGenes) - np.count_nonzero(goodGenes)} gene(s) detected as an outlier!{ENDC}")
                print(
                    f"{OKGREEN}Removing genes: {', '.join(self.datExpr.var.index[[not goodGene for goodGene in goodGenes]].tolist())}{ENDC}")
            if np.size(goodSamples) - np.count_nonzero(goodSamples) > 0:
                print(
                    f"{OKGREEN} {np.size(goodSamples) - np.count_nonzero(goodSamples)} sample(s) detected as an outlier!{ENDC}")
                print(
                    f"{OKGREEN}Removing samples: {', '.join(self.datExpr.obs.index[[not goodSample for goodSample in goodSamples]].tolist())}{ENDC}")
            # Remove the offending genes and samples from the data:
            self.datExpr = self.datExpr[goodSamples, goodGenes]

        # Clustering
        sampleTree = pyWGCNA.hclust(pdist(self.datExpr.to_df()), method="average")

        if show:
            plt.figure(figsize=(max(25, round(self.datExpr.X.shape[0] / 20)), 10), facecolor='white')
            dendrogram(sampleTree, color_threshold=self.cut, labels=self.datExpr.to_df().index, leaf_rotation=90,
                       leaf_font_size=8)
            plt.axhline(y=self.cut, c='grey', lw=1, linestyle='dashed')
            plt.title('Sample clustering to detect outliers')
            plt.xlabel('Samples')
            plt.ylabel('Distances')
            plt.tight_layout()
            if self.save:
                plt.savefig(f"{self.outputPath}figures/sample_clustering_cleaning.{self.figureType}")

        # Determine cluster under the line
        clust = pyWGCNA.cutree(sampleTree, cutHeight=self.cut)
        # clust 0 contains the samples we want to keep.
        clust = clust.T.tolist()[0]
        index = [index for index, element in enumerate(clust) if element == 0]

        self.datExpr = self.datExpr[index, :]

        print("\tDone pre-processing..\n")

    def findModules(self, kwargs_function={'cutreeHybrid': {'deepSplit': 2, 'pamRespectsDendro': False}},
                    colorlist=None, max_block_size=None):
        """
        Clustering genes through original WGCNA pipeline: 1.pick soft threshold 2.calculating adjacency matrix 3.calculating TOM similarity matrix 4.cluster genes base of dissTOM 5.merge similar cluster dynamically

        :param kwargs_function: dictionary where the keys are the name of the function and values are the dictionary contains parameter you want to change within function
        :type kwargs_function: dict
        :param max_block_size: if set and the gene count exceeds it, run the
            memory-bounded blockwise pipeline (R WGCNA ``blockwiseModules``
            style) instead of one full gene x gene adjacency/TOM. The single
            pipeline holds ~5-6 dense copies of an N x N matrix, so for
            N >> 10000 genes pass e.g. ``max_block_size=5000``.
        :type max_block_size: int or None
        """
        n_genes = self.datExpr.shape[1]
        if max_block_size is not None and n_genes > max_block_size:
            return self.calculate_blockwise_modules(
                max_block_size=max_block_size, kwargs_function=kwargs_function
            )

        print(f"{BOLD}{OKBLUE}Run pyWGCNA...{ENDC}")

        # Call the network topology analysis function
        kwargs = dict()
        if 'pickSoftThreshold' in list(kwargs_function.keys()):
            kwargs = kwargs_function['pickSoftThreshold']
        self.power, self.sft = pyWGCNA.pickSoftThreshold(self.datExpr.to_df(), RsquaredCut=self.RsquaredCut,
                                                       MeanCut=self.MeanCut, powerVector=self.powers,
                                                       networkType=self.networkType, **kwargs)

        fig, ax = plt.subplots(ncols=2, figsize=(8, 4), facecolor='white')
        ax[0].plot(self.sft['Power'], -1 * np.sign(self.sft['slope']) * self.sft['SFT.R.sq'], 'o')
        for i in range(len(self.powers)):
            ax[0].text(self.sft.loc[i, 'Power'],
                       -1 * np.sign(self.sft.loc[i, 'slope']) * self.sft.loc[i, 'SFT.R.sq'],
                       str(self.sft.loc[i, 'Power']), ha="center", va="center", color='black', weight='bold')
        ax[0].axhline(0.9, color='r')
        ax[0].set_xlabel("Soft Threshold (power)")
        ax[0].set_ylabel("Scale Free Topology Model Fit,signed R^2")
        ax[0].title.set_text('Scale independence')

        ax[1].plot(self.sft['Power'], self.sft['mean(k)'], 'o')
        for i in range(len(self.powers)):
            ax[1].text(self.sft.loc[i, 'Power'], self.sft.loc[i, 'mean(k)'],
                       str(self.sft.loc[i, 'Power']), ha="center", va="center", color='r', weight='bold')
        ax[1].set_xlabel("Soft Threshold (power)")
        ax[1].set_ylabel("Mean Connectivity")
        ax[1].title.set_text('Mean connectivity')

        fig.tight_layout()
        if self.save:
            fig.savefig(f"{self.outputPath}figures/summary_power.{self.figureType}")

        # Set Power
        kwargs = dict()
        if 'adjacency' in list(kwargs_function.keys()):
            kwargs = kwargs_function['adjacency']
        self.adjacency = pyWGCNA.adjacency(self.datExpr.to_df(), power=self.power, adjacencyType=self.networkType, **kwargs)
        self.adjacency = pd.DataFrame(self.adjacency,
                                      columns=self.datExpr.to_df().columns,
                                      index=self.datExpr.to_df().columns)

        # Turn adjacency into topological overlap
        kwargs = dict()
        if 'TOMsimilarity' in list(kwargs_function.keys()):
            kwargs = kwargs_function['TOMsimilarity']
        self.TOM = pyWGCNA.TOMsimilarity(self.adjacency.to_numpy(), TOMType=self.TOMType, **kwargs)
        self.TOM.columns = self.datExpr.to_df().columns
        self.TOM.index = self.datExpr.to_df().columns
        dissTOM = 1 - self.TOM
        dissTOM = dissTOM.round(decimals=8)

        a = squareform(dissTOM.values, checks=False)
        # Call the hierarchical clustering function
        self.geneTree = linkage(a, method="average")

        # Module identification using dynamic tree cut
        kwargs = dict()
        if 'cutreeHybrid' in list(kwargs_function.keys()):
            kwargs = kwargs_function['cutreeHybrid']
        #dynamicMods = pyWGCNA.cutreeHybrid(dendro=self.geneTree, distM=dissTOM, minClusterSize=self.minModuleSize, **kwargs)

        from ...bulk._dynamicTree import cutreeHybrid
        print(f"{OKCYAN}Going through the merge tree...{ENDC}")
        dynamicMods_cut = cutreeHybrid(self.geneTree, distM=a, minClusterSize=self.minModuleSize, **kwargs)
        self.dynamicMods=pd.DataFrame({"Name": dynamicMods_cut['labels'], "Value": dynamicMods_cut['labels']})

        # Convert numeric labels into colors
        kwargs = dict()
        if 'labels2colors' in list(kwargs_function.keys()):
            kwargs = kwargs_function['labels2colors']
        self.datExpr.var['dynamicColors'] = pyWGCNA.labels2colors(labels=self.dynamicMods, **kwargs)


        # Calculate eigengenes
        kwargs = dict()
        if 'moduleEigengenes' in list(kwargs_function.keys()):
            kwargs = kwargs_function['moduleEigengenes']
        MEList = pyWGCNA.moduleEigengenes(expr=self.datExpr.to_df(), colors=self.datExpr.var['dynamicColors'], **kwargs)

        self.MEs = MEList['eigengenes']
        if 'MEgrey' in self.MEs.columns:
            self.MEs.drop(['MEgrey'], axis=1, inplace=True)
        # Calculate dissimilarity of module eigengenes
        MEDiss = pd.DataFrame(1 - np.corrcoef(self.MEs, rowvar=False), index=self.MEs.columns, columns=self.MEs.columns)
        if MEDiss.shape != (1, 1):
            # Cluster module eigengenes
            a = squareform(MEDiss, checks=False)

            METree = pyWGCNA.hclust(a, method="average")
            
            plt.figure(figsize=(max(20, round(MEDiss.shape[1] / 20)), 10), facecolor='white')
            dendrogram(METree, color_threshold=self.MEDissThres, labels=MEDiss.columns, leaf_rotation=90,
                       leaf_font_size=8)
            plt.axhline(y=self.MEDissThres, c='grey', lw=1, linestyle='dashed')
            plt.title('Clustering of module eigengenes')
            plt.xlabel('')
            plt.ylabel('')
            plt.tight_layout()
            if self.save:
                plt.savefig(f"{self.outputPath}figures/eigenesgenes.{self.figureType}")
            

            # Call an automatic merging function
            kwargs = dict()
            if 'mergeCloseModules' in list(kwargs_function.keys()):
                kwargs = kwargs_function['mergeCloseModules']
            merge = pyWGCNA.mergeCloseModules(self.datExpr.to_df(), self.datExpr.var['dynamicColors'],
                                            cutHeight=self.MEDissThres, **kwargs)
            # The merged module colors; Rename to moduleColors
            self.datExpr.var['moduleColors'] = merge['colors']
        else:
            self.datExpr.var['moduleColors'] = ["black"] * self.datExpr.shape[1]
        


        # Construct numerical labels corresponding to the colors
        colorOrder = np.unique(self.datExpr.var['moduleColors']).tolist()
        self.datExpr.var['moduleLabels'] = [colorOrder.index(x) if x in colorOrder else None for x in
                                            self.datExpr.var['moduleColors']]
        # Eigengenes of the new merged modules:
        self.MEs = merge['newMEs']

        # Recalculate MEs with color labels
        kwargs = dict()
        if 'moduleEigengenes' in list(kwargs_function.keys()):
            kwargs = kwargs_function['moduleEigengenes']
        self.datME = pyWGCNA.moduleEigengenes(self.datExpr.to_df(), self.datExpr.var['moduleColors'], **kwargs)['eigengenes']
        if 'MEgrey' in self.datME.columns:
            self.datME.drop(['MEgrey'], axis=1, inplace=True)

        kwargs = dict()
        if 'orderMEs' in list(kwargs_function.keys()):
            kwargs = kwargs_function['orderMEs']
        self.MEs = pyWGCNA.orderMEs(self.datME, **kwargs)

        # Visualize the module
        plt.figure(figsize=(25, 8))
        grid = plt.GridSpec(10, 1, wspace=0.5, hspace=0.1)

        plt.subplot(grid[0:6,0])
        set_link_color_palette(['#000000'])
        dn=dendrogram(self.geneTree,color_threshold=0.5, above_threshold_color='black')
        plt.tick_params( \
            axis='x',
            which='both',
            bottom='off',
            top='off',
            labelbottom='off')
        #mol
        #ivl=dn['ivl']
        mod=[self.dynamicMods['Value'][int(x)] for x in dn['ivl']]
        plot_mod=np.array([mod])
        gene_name=[self.adjacency.index[int(x)] for x in dn['ivl']]
        mol=pd.DataFrame(columns=['ivl','module','name'])
        mol['ivl']=dn['ivl']
        mol['module']=mod
        mol['name']=gene_name
        mol['dynamo_color']=mol['name'].map(dict(zip(gene_name,self.datExpr.var.loc[gene_name,'dynamicColors'])))
        mol['module_color']=mol['name'].map(dict(zip(gene_name,self.datExpr.var.loc[gene_name,'moduleColors'])))
        self.mol=mol

        # 将颜色字符串转换为 RGBA 格式
        dynamo_colors = mol['dynamo_color'].map(mcolors.CSS4_COLORS.get).map(mcolors.to_rgba).tolist()
        module_colors = mol['module_color'].map(mcolors.CSS4_COLORS.get).map(mcolors.to_rgba).tolist()

        # 绘制 dynamo_color 颜色条
        plt.subplot(grid[6:8, 0])
        dynamo_colors = np.array(dynamo_colors).reshape(1, -1, 4)
        plt.imshow(dynamo_colors, aspect='auto', origin='lower')
        plt.axis('off')

        # 绘制 module_color 颜色条
        plt.subplot(grid[8:, 0])
        module_colors = np.array(module_colors).reshape(1, -1, 4)
        plt.imshow(module_colors, aspect='auto', origin='lower')
        plt.axis('off')
        if self.save:
            plt.savefig(f"{self.outputPath}figures/eigenesgenes_genetree.{self.figureType}")

        print("\tDone running pyWGCNA..\n")

    @staticmethod
    def _blockwise_gene_clusters(expr_df, max_block_size, random_state=0):
        """Partition genes into co-expression blocks, each <= max_block_size.

        Genes are clustered by their z-scored expression profile and the gene
        set is recursively k-means-split until every block fits the cap. This
        keeps co-expressed genes together (so few modules straddle block
        boundaries) while guaranteeing a per-block adjacency / TOM never
        exceeds ``max_block_size**2`` entries — the whole point of the
        blockwise approach. Returns a list of integer index arrays, sorted
        largest block first (matching R ``blockwiseModules``).
        """
        from sklearn.cluster import KMeans

        X = np.asarray(expr_df.to_numpy(), dtype=float).T          # genes x samples
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd                                          # z-score per gene
        n_genes = Z.shape[0]

        blocks = []
        stack = [np.arange(n_genes)]
        while stack:
            idx = stack.pop()
            if len(idx) <= max_block_size:
                blocks.append(idx)
                continue
            k = int(np.ceil(len(idx) / max_block_size))
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            sub = km.fit_predict(Z[idx])
            children = [idx[sub == c] for c in range(k) if (sub == c).any()]
            # Degenerate k-means (no size reduction) -> hard consecutive chunks.
            if max(len(c) for c in children) == len(idx):
                children = [idx[i:i + max_block_size]
                            for i in range(0, len(idx), max_block_size)]
            stack.extend(children)

        blocks.sort(key=len, reverse=True)
        return blocks

    def calculate_blockwise_modules(
        self,
        max_block_size=5000,
        kwargs_function=None,
        random_state=0,
    ):
        """Memory-bounded module detection — the R ``blockwiseModules`` analogue.

        The standard pipeline (:meth:`findModules`) builds a dense gene x gene
        adjacency *and* TOM; at peak ~5-6 copies of an ``N x N`` matrix exist,
        so it OOMs for large gene counts (N ~ 20000 needs tens of GB).

        This method instead:

        1. picks the soft-threshold power on the full matrix, streaming the
           gene-gene correlation in chunks of ``max_block_size`` so this step
           too stays memory-bounded;
        2. pre-clusters genes into co-expression blocks each <= ``max_block_size``;
        3. builds the adjacency / TOM / gene tree and runs the dynamic tree cut
           **one block at a time**, freeing each block's matrices before the
           next — so peak memory is bounded by ``max_block_size**2``, not ``N**2``;
        4. merges modules whose eigengenes are close *across all blocks*
           (``mergeCloseModules`` works on the eigengene matrix, samples x
           modules, which is small).

        Populates the same attributes as :meth:`findModules`
        (``datExpr.var['dynamicColors' / 'moduleColors' / 'moduleLabels']``,
        ``MEs``, ``datME``, ``power``, ``sft``), plus ``self.blocks`` (the
        per-block gene-index arrays) and ``self.blockGeneTrees`` (per-block
        linkage matrices). ``self.geneTree`` is set to ``None`` — there is no
        single global dendrogram in blockwise mode.

        :param max_block_size: maximum genes per block (R default 5000).
        :type max_block_size: int
        :param kwargs_function: same per-function kwargs dict as findModules.
        :type kwargs_function: dict
        :param random_state: seed for the gene pre-clustering k-means.
        :type random_state: int
        """
        from ...bulk._dynamicTree import cutreeHybrid

        if kwargs_function is None:
            kwargs_function = {'cutreeHybrid': {'deepSplit': 2,
                                                'pamRespectsDendro': False}}
        expr = self.datExpr.to_df()                       # samples x genes
        genes = list(expr.columns)
        n_genes = len(genes)
        print(f"{BOLD}{OKBLUE}Run pyWGCNA (blockwise, max_block_size="
              f"{max_block_size})...{ENDC}")

        # --- 1. soft threshold ----------------------------------------------
        # pickSoftThreshold streams genes in chunks of ``blockSize``; cap it at
        # max_block_size so its peak footprint stays O(nGenes x max_block_size).
        kwargs = dict(kwargs_function.get('pickSoftThreshold', {}))
        kwargs.setdefault('blockSize', max_block_size)
        self.power, self.sft = pyWGCNA.pickSoftThreshold(
            expr, RsquaredCut=self.RsquaredCut, MeanCut=self.MeanCut,
            powerVector=self.powers, networkType=self.networkType, **kwargs)

        # --- 2. pre-cluster genes into size-capped blocks -------------------
        blocks = pyWGCNA._blockwise_gene_clusters(expr, max_block_size,
                                                  random_state=random_state)
        print(f"{OKCYAN}Partitioned {n_genes} genes into {len(blocks)} "
              f"block(s) of sizes {[len(b) for b in blocks]}.{ENDC}")

        # --- 3. per-block adjacency -> TOM -> tree -> dynamic cut -----------
        adj_kwargs = kwargs_function.get('adjacency', {})
        tom_kwargs = kwargs_function.get('TOMsimilarity', {})
        cut_kwargs = kwargs_function.get('cutreeHybrid',
                                         {'deepSplit': 2, 'pamRespectsDendro': False})

        block_labels = np.zeros(n_genes, dtype=int)       # 0 == grey / unassigned
        self.blockGeneTrees = []
        label_offset = 0
        for bi, block_idx in enumerate(blocks):
            block_genes = [genes[i] for i in block_idx]
            sub = expr[block_genes]
            print(f"{OKCYAN}  block {bi + 1}/{len(blocks)}: "
                  f"{len(block_genes)} genes — adjacency/TOM/treecut...{ENDC}")
            adj = pyWGCNA.adjacency(sub, power=self.power,
                                    adjacencyType=self.networkType, **adj_kwargs)
            tom = pyWGCNA.TOMsimilarity(np.asarray(adj), TOMType=self.TOMType,
                                        **tom_kwargs)
            diss = squareform(
                (1 - np.asarray(tom)).round(decimals=8), checks=False)
            tree = linkage(diss, method="average")
            self.blockGeneTrees.append(tree)
            cut = cutreeHybrid(tree, distM=diss,
                               minClusterSize=self.minModuleSize, **cut_kwargs)
            # cutreeHybrid returns a bare zeros-array (not the result dict)
            # when a block has no merges below the cut, i.e. no modules —
            # in that case every gene in the block is grey / unassigned.
            if isinstance(cut, dict):
                lab = np.asarray(cut['labels'], dtype=int)
            else:
                lab = np.zeros(len(block_idx), dtype=int)
            # Offset non-grey labels so they are globally unique across blocks.
            block_labels[block_idx] = np.where(lab > 0, lab + label_offset, 0)
            if (lab > 0).any():
                label_offset = int(block_labels.max())
            del adj, tom, diss, tree                      # free per-block memory

        # --- 4. combined dynamic colors ------------------------------------
        self.dynamicMods = pd.DataFrame({"Name": block_labels,
                                         "Value": block_labels})
        kwargs = kwargs_function.get('labels2colors', {})
        self.datExpr.var['dynamicColors'] = pyWGCNA.labels2colors(
            labels=self.dynamicMods, **kwargs)

        # --- 5. merge close modules across all blocks ----------------------
        # mergeCloseModules operates on the eigengene matrix (samples x
        # modules) — small, so this step is memory-safe regardless of N.
        n_mod = len(set(self.datExpr.var['dynamicColors']) - {"grey"})
        if n_mod >= 2:
            kwargs = kwargs_function.get('mergeCloseModules', {})
            merge = pyWGCNA.mergeCloseModules(
                expr, self.datExpr.var['dynamicColors'],
                cutHeight=self.MEDissThres, **kwargs)
            self.datExpr.var['moduleColors'] = merge['colors']
        else:
            self.datExpr.var['moduleColors'] = list(
                self.datExpr.var['dynamicColors'])

        colorOrder = np.unique(self.datExpr.var['moduleColors']).tolist()
        self.datExpr.var['moduleLabels'] = [
            colorOrder.index(x) for x in self.datExpr.var['moduleColors']]

        # --- 6. module eigengenes ------------------------------------------
        kwargs = kwargs_function.get('moduleEigengenes', {})
        self.datME = pyWGCNA.moduleEigengenes(
            expr, self.datExpr.var['moduleColors'], **kwargs)['eigengenes']
        if 'MEgrey' in self.datME.columns:
            self.datME.drop(['MEgrey'], axis=1, inplace=True)
        kwargs = kwargs_function.get('orderMEs', {})
        self.MEs = pyWGCNA.orderMEs(self.datME, **kwargs)

        # No single global dendrogram in blockwise mode.
        self.geneTree = None
        self.blocks = blocks
        print(f"{OKCYAN}Blockwise pyWGCNA done — "
              f"{len(colorOrder)} modules across {len(blocks)} blocks.{ENDC}")
        return self

    def runWGCNA(self):
        """
        Preprocess and find modules
        """
        pyWGCNA.preprocess(self)

        pyWGCNA.findModules(self)

        return self
    
    def calculate_soft_threshold(self,
                    colorlist=None,**kwargs):
        print(f"{BOLD}{OKBLUE}Run pyWGCNA...{ENDC}")

        # Call the network topology analysis function
        self.power, self.sft = pyWGCNA.pickSoftThreshold(self.datExpr.to_df(), RsquaredCut=self.RsquaredCut,
                                                       MeanCut=self.MeanCut, powerVector=self.powers,
                                                       networkType=self.networkType, **kwargs)

        fig, ax = plt.subplots(ncols=2, figsize=(8, 4), facecolor='white')
        ax[0].plot(self.sft['Power'], -1 * np.sign(self.sft['slope']) * self.sft['SFT.R.sq'], 'o')
        for i in range(len(self.powers)):
            ax[0].text(self.sft.loc[i, 'Power'],
                       -1 * np.sign(self.sft.loc[i, 'slope']) * self.sft.loc[i, 'SFT.R.sq'],
                       str(self.sft.loc[i, 'Power']), ha="center", va="center", color='black', weight='bold')
        ax[0].axhline(0.9, color='r')
        ax[0].set_xlabel("Soft Threshold (power)")
        ax[0].set_ylabel("Scale Free Topology Model Fit,signed R^2")
        ax[0].title.set_text('Scale independence')

        ax[1].plot(self.sft['Power'], self.sft['mean(k)'], 'o')
        for i in range(len(self.powers)):
            ax[1].text(self.sft.loc[i, 'Power'], self.sft.loc[i, 'mean(k)'],
                       str(self.sft.loc[i, 'Power']), ha="center", va="center", color='r', weight='bold')
        ax[1].set_xlabel("Soft Threshold (power)")
        ax[1].set_ylabel("Mean Connectivity")
        ax[1].title.set_text('Mean connectivity')

        fig.tight_layout()
        if self.save:
            fig.savefig(f"{self.outputPath}figures/summary_power.{self.figureType}")

    def calculating_adjacency_matrix(self,**kwargs):
        # Set Power
        self.adjacency = pyWGCNA.adjacency(self.datExpr.to_df(), power=self.power, adjacencyType=self.networkType, **kwargs)
        self.adjacency = pd.DataFrame(self.adjacency,
                                      columns=self.datExpr.to_df().columns,
                                      index=self.datExpr.to_df().columns)
        
    def calculating_TOM_similarity_matrix(self,**kwargs):
        # Turn adjacency into topological overlap
        self.TOM = pyWGCNA.TOMsimilarity(self.adjacency.to_numpy(), TOMType=self.TOMType, **kwargs)
        self.TOM.columns = self.datExpr.to_df().columns
        self.TOM.index = self.datExpr.to_df().columns
    
    def calculate_geneTree(self):
        dissTOM = 1 - self.TOM
        dissTOM = dissTOM.round(decimals=8)

        self.a = squareform(dissTOM.values, checks=False)
        # Call the hierarchical clustering function
        self.geneTree = linkage(self.a, method="average")

    def calculate_dynamicMods(self,kwargs_function={'cutreeHybrid': {'deepSplit': 2, 'pamRespectsDendro': False}}):
        # Module identification using dynamic tree cut
        from ...bulk._dynamicTree import cutreeHybrid
        print(f"{OKCYAN}Going through the merge tree...{ENDC}")
        kwargs = dict()
        if 'cutreeHybrid' in list(kwargs_function.keys()):
            kwargs = kwargs_function['cutreeHybrid']
        dynamicMods_cut = cutreeHybrid(self.geneTree, distM=self.a, minClusterSize=self.minModuleSize, **kwargs)
        self.dynamicMods=pd.DataFrame({"Name": dynamicMods_cut['labels'], "Value": dynamicMods_cut['labels']})
        # Convert numeric labels into colors
        # Convert numeric labels into colors
        kwargs = dict()
        if 'labels2colors' in list(kwargs_function.keys()):
            kwargs = kwargs_function['labels2colors']
        self.datExpr.var['dynamicColors'] = pyWGCNA.labels2colors(labels=self.dynamicMods, **kwargs)

    def calculate_gene_module(self,kwargs_function={'cutreeHybrid': {'deepSplit': 2, 'pamRespectsDendro': False}}):
        # Calculate eigengenes
        kwargs = dict()
        if 'moduleEigengenes' in list(kwargs_function.keys()):
            kwargs = kwargs_function['moduleEigengenes']
        MEList = pyWGCNA.moduleEigengenes(expr=self.datExpr.to_df(), colors=self.datExpr.var['dynamicColors'], **kwargs)

        self.MEs = MEList['eigengenes']
        if 'MEgrey' in self.MEs.columns:
            self.MEs.drop(['MEgrey'], axis=1, inplace=True)
        # Calculate dissimilarity of module eigengenes
        MEDiss = pd.DataFrame(1 - np.corrcoef(self.MEs, rowvar=False), index=self.MEs.columns, columns=self.MEs.columns)
        if MEDiss.shape != (1, 1):
            # Cluster module eigengenes
            a = squareform(MEDiss, checks=False)

            METree = pyWGCNA.hclust(a, method="average")
            # Call an automatic merging function
            kwargs = dict()
            if 'mergeCloseModules' in list(kwargs_function.keys()):
                kwargs = kwargs_function['mergeCloseModules']
            merge = pyWGCNA.mergeCloseModules(self.datExpr.to_df(), self.datExpr.var['dynamicColors'],
                                            cutHeight=self.MEDissThres, **kwargs)
            # The merged module colors; Rename to moduleColors
            self.datExpr.var['moduleColors'] = merge['colors']
        else:
            self.datExpr.var['moduleColors'] = ["black"] * self.datExpr.shape[1]
        


        # Construct numerical labels corresponding to the colors
        colorOrder = np.unique(self.datExpr.var['moduleColors']).tolist()
        self.datExpr.var['moduleLabels'] = [colorOrder.index(x) if x in colorOrder else None for x in
                                            self.datExpr.var['moduleColors']]
        # Eigengenes of the new merged modules:
        self.MEs = merge['newMEs']

        # Recalculate MEs with color labels
        kwargs = dict()
        if 'moduleEigengenes' in list(kwargs_function.keys()):
            kwargs = kwargs_function['moduleEigengenes']
        self.datME = pyWGCNA.moduleEigengenes(self.datExpr.to_df(), self.datExpr.var['moduleColors'], **kwargs)['eigengenes']
        if 'MEgrey' in self.datME.columns:
            self.datME.drop(['MEgrey'], axis=1, inplace=True)

        kwargs = dict()
        if 'orderMEs' in list(kwargs_function.keys()):
            kwargs = kwargs_function['orderMEs']
        self.MEs = pyWGCNA.orderMEs(self.datME, **kwargs)

        # Visualize the module
        plt.figure(figsize=(25, 8))
        grid = plt.GridSpec(10, 1, wspace=0.5, hspace=0.1)

        plt.subplot(grid[0:6,0])
        set_link_color_palette(['#000000'])
        dn=dendrogram(self.geneTree,color_threshold=0.5, above_threshold_color='black')
        plt.tick_params( \
            axis='x',
            which='both',
            bottom='off',
            top='off',
            labelbottom='off')
        #mol
        #ivl=dn['ivl']
        mod=[self.dynamicMods['Value'][int(x)] for x in dn['ivl']]
        plot_mod=np.array([mod])
        gene_name=[self.adjacency.index[int(x)] for x in dn['ivl']]
        mol=pd.DataFrame(columns=['ivl','module','name'])
        mol['ivl']=dn['ivl']
        mol['module']=mod
        mol['name']=gene_name
        mol['dynamo_color']=mol['name'].map(dict(zip(gene_name,self.datExpr.var.loc[gene_name,'dynamicColors'])))
        mol['module_color']=mol['name'].map(dict(zip(gene_name,self.datExpr.var.loc[gene_name,'moduleColors'])))
        self.mol=mol

        # 将颜色字符串转换为 RGBA 格式
        dynamo_colors = mol['dynamo_color'].map(mcolors.CSS4_COLORS.get).map(mcolors.to_rgba).tolist()
        module_colors = mol['module_color'].map(mcolors.CSS4_COLORS.get).map(mcolors.to_rgba).tolist()

        # 绘制 dynamo_color 颜色条
        plt.subplot(grid[6:8, 0])
        dynamo_colors = np.array(dynamo_colors).reshape(1, -1, 4)
        plt.imshow(dynamo_colors, aspect='auto', origin='lower')
        plt.axis('off')

        # 绘制 module_color 颜色条
        plt.subplot(grid[8:, 0])
        module_colors = np.array(module_colors).reshape(1, -1, 4)
        plt.imshow(module_colors, aspect='auto', origin='lower')
        plt.axis('off')
        if self.save:
            plt.savefig(f"{self.outputPath}figures/eigenesgenes_genetree.{self.figureType}")

    def visual_gene_module(self,figsize=(25,8)):
        # Visualize the module
        plt.figure(figsize=figsize)
        grid = plt.GridSpec(10, 1, wspace=0.5, hspace=0.1)

        plt.subplot(grid[0:6,0])
        set_link_color_palette(['#000000'])
        dn=dendrogram(self.geneTree,color_threshold=0.5, above_threshold_color='black')
        plt.tick_params( \
            axis='x',
            which='both',
            bottom='off',
            top='off',
            labelbottom='off')
        #mol
        #ivl=dn['ivl']
        mod=[self.dynamicMods['Value'][int(x)] for x in dn['ivl']]
        plot_mod=np.array([mod])
        gene_name=[self.adjacency.index[int(x)] for x in dn['ivl']]
        mol=pd.DataFrame(columns=['ivl','module','name'])
        mol['ivl']=dn['ivl']
        mol['module']=mod
        mol['name']=gene_name
        mol['dynamo_color']=mol['name'].map(dict(zip(gene_name,self.datExpr.var.loc[gene_name,'dynamicColors'])))
        mol['module_color']=mol['name'].map(dict(zip(gene_name,self.datExpr.var.loc[gene_name,'moduleColors'])))

        self.mol=mol
        
        # 将颜色字符串转换为 RGBA 格式
        dynamo_colors = mol['dynamo_color'].map(mcolors.CSS4_COLORS.get).map(mcolors.to_rgba).tolist()
        module_colors = mol['module_color'].map(mcolors.CSS4_COLORS.get).map(mcolors.to_rgba).tolist()

        # 绘制 dynamo_color 颜色条
        plt.subplot(grid[6:8, 0])
        dynamo_colors = np.array(dynamo_colors).reshape(1, -1, 4)
        plt.imshow(dynamo_colors, aspect='auto', origin='lower')
        plt.axis('off')

        # 绘制 module_color 颜色条
        plt.subplot(grid[8:, 0])
        module_colors = np.array(module_colors).reshape(1, -1, 4)
        plt.imshow(module_colors, aspect='auto', origin='lower')
        plt.axis('off')
        if self.save:
            plt.savefig(f"{self.outputPath}figures/eigenesgenes_genetree.{self.figureType}")

    
    def plot_matrix(self,cmap='RdBu_r',save:bool=True,figsize:tuple=(8,9),
                    legene_ncol:int=2,legene_bbox_to_anchor:tuple=(5, 2.95),legene_fontsize:int=12,
                    color_type='dynamicColors',**kwargs):
        """plot the matrix of correlation

        Arguments:
            cmap: The color of matrix
            save: Whether to save the figure
            figsize: The size of figure
            legene_ncol: The number of column of legene
            legene_bbox_to_anchor: The position of legene
            legene_fontsize: The size of legene

        Returns:
            ax: The axis of figure
        """

        module_color=self.mol.copy()
        #module_color.index=module_color['name']
        a=sns.clustermap(self.TOM,#standard_scale=1,
                            cmap=cmap,yticklabels=False,xticklabels=False,#method='ward',metric='euclidean',
                            #row_cluster=False,col_cluster=False,
                            col_linkage=self.geneTree,row_linkage=self.geneTree,
                            col_colors=self.datExpr.var.loc[self.TOM.index,color_type].values,
                            row_colors=self.datExpr.var.loc[self.TOM.index,color_type].values,
                            cbar_kws={"shrink": .5},square=True,
                            dendrogram_ratio=(.1, .2),
                            cbar_pos=(-.1, .32, .03, .2),figsize=figsize,**kwargs
                )
        #add the color of the module
        for i in (set(module_color['module'])):
            med=np.median(module_color.loc[module_color['module']==i].index)
            med_shape=module_color.loc[module_color['module']==i].shape
            a.ax_heatmap.add_patch(plt.Rectangle((med-med_shape[0]//2,med-med_shape[0]//2),
                                                med_shape[0],med_shape[0],fill=False,color='black',lw=1))

        original_list = self.datExpr.var['dynamicColors'].tolist()
        unique_list = list(dict.fromkeys(original_list))

        from matplotlib import patches as mpatches
        #用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend
        patches = [ mpatches.Patch(color=mcolors.CSS4_COLORS[i], label=f"ME{i}" ) for i in unique_list ] 

        plt.legend(handles=patches,bbox_to_anchor=legene_bbox_to_anchor, ncol=legene_ncol,fontsize=legene_fontsize)
        if save==True:
            plt.savefig(f"{self.outputPath}figures/module_matrix.{self.figureType}",
                        dpi=300,bbox_inches = 'tight')
        return a

    def get_sub_module(self,mod_list:list,mod_type='module_color')->pd.DataFrame:
        '''
        Get sub-module of a module

        Arguments:
            mod_list: module number

        Returns:
            sub_module:sub-module of a module
        '''
        mol=self.mol
        return mol[mol[mod_type].isin(mod_list)]
    
    def get_sub_network(self,mod_list:list,mod_type='module_color',
                        correlation_threshold=0.95,)->nx.Graph:
        '''
        Get sub-network of a module

        Arguments:
            mod_list: module number

        Returns:
            sub_network:sub-network of a module
        '''
        
        module1=self.get_sub_module(mod_list=mod_list,mod_type=mod_type)
        gene_net1=abs(self.TOM.loc[module1.name.tolist(),module1.name.tolist()])
        # 获取 DataFrame 的索引
        idx = np.diag_indices_from(gene_net1)

        # 将对角线元素置为 0
        gene_net1.values[idx] = 0

        G = nx.Graph()
        for i in gene_net1.index:
            for j in gene_net1.columns:
                if gene_net1.loc[i,j]>correlation_threshold:
                    G.add_weighted_edges_from([(i, j, gene_net1.loc[i,j])])
        return G
    
    def plot_sub_network(self,mod_list:list,mod_type='module_color',
                         correlation_threshold:float=0.95,
                         plot_genes=None,
                         plot_gene_num:int=5,**kwargs):
        '''
        plot sub-network of a module

        Arguments:
            mod_list: module number
            correlation_threshold: correlation threshold
            plot_genes: genes to plot in the sub-network. If None, the hub genes will be ploted
            plot_gene_num: number of genes to plot

        Returns:
            fig: figure
            ax: axis
        '''

        module1=self.get_sub_module(mod_list=mod_list,mod_type=mod_type)
        G_type_dict=dict(zip(module1['name'],[str(i) for i in module1[mod_type]]))
        G_color_dict=dict(zip(module1['name'],module1[mod_type].map(mcolors.CSS4_COLORS.get).map(mcolors.to_rgba).tolist()))
        G=self.get_sub_network(mod_list=mod_list,mod_type=mod_type,correlation_threshold=correlation_threshold)
        G_color_dict=dict(zip(G.nodes,[G_color_dict[i] for i in G.nodes]))
        G_type_dict=dict(zip(G.nodes,[G_type_dict[i] for i in G.nodes]))

        degree_dict = dict(G.degree(G.nodes()))
        de_pd=pd.DataFrame(degree_dict.values(),index=degree_dict.keys(),columns=['Degree'])
        hub_gene=[]
        if plot_genes!=None:
            hub_gene=plot_genes
        else:
            for i in mod_list:
                ret_gene=list(set(de_pd.index) & set(module1.loc[module1['module']==i]['name'].tolist()))
                hub_gene1=de_pd.loc[ret_gene,'Degree'].sort_values(ascending=False)[:plot_gene_num].index.tolist()
                hub_gene+=hub_gene1
        from ...utils import plot_network
        fig,ax=plot_network(G,G_type_dict,G_color_dict,plot_node=hub_gene,**kwargs)
        return fig,ax

    def analyseWGCNA(self, order=None, geneList=None, show=True, alternative='two-sided'):
        """
        Analysing results: 1.calculating module trait relationship 2.plotting module heatmap eigengene 3.finding GO term for each module

        :param order: indicate in which order metadata will show up in plots (should same as metadata name in anndata)
        :type order: list
        :param geneList: genes information you want to add (keep in mind you can not have multiple row for same gene)
        :type geneList: pandas dataframe
        :param show: indicate if you want to see plots in when you run your code
        :type show: bool
        :param alternative: Defines the alternative hypothesis for calculating correlation for module-trait relationship. Default is ‘two-sided’. The following options are available: 'two-sided’: the correlation is nonzero, ‘less’: the correlation is negative (less than zero), ‘greater’: the correlation is positive (greater than zero)
        :type alternative: str
        """
        print(f"{BOLD}{OKBLUE}Analysing pyWGCNA...{ENDC}")

        print(f"{OKCYAN}Calculating module trait relationship ...{ENDC}")

        self.module_trait_relationships_heatmap(metaData=self.datExpr.obs.columns.tolist(),
                                                alternative=alternative,
                                                show=show,
                                                file_name='module-traitRelationships')
        print("\tDone..\n")

        print(f"{OKCYAN}Adding (signed) eigengene-based connectivity (module membership) ...{ENDC}")
        self.CalculateSignedKME()
        print("\tDone..\n")

        if geneList is not None:
            print(f"{OKCYAN}Updating gene information based on given gene list ...{ENDC}")
            self.updateGeneInfo(geneInfo=geneList)
            print("\tDone..\n")
        return 'Done'
        # Select module probes
        modules = self.datExpr.var.moduleColors.unique().tolist()
        metadata = self.datExpr.obs.columns.tolist()
        if order is not None:
            if all(item in metadata for item in order):
                metadata = order
            else:
                sys.exit("Given order is not valid!")

        if self.save:
            print(f"{OKCYAN}plotting module heatmap eigengene...{ENDC}")
            for module in modules:
                self.plotModuleEigenGene(module, metadata, show=show)
            print("\tDone..\n")

        if self.save:
            print(f"{OKCYAN}plotting module barplot eigengene...{ENDC}")
            if self.datExpr.obs.shape[1] == 0:
                print(
                    f"{WARNING}There is not any sample information in given object! Skip running barplotModuleEigenGene() function.{ENDC}")
            else:
                
                for module in modules:
                   self.barplotModuleEigenGene(module, metadata, colorBar=metadata[-1], show=show)
                print("\tDone..\n")
        '''
        if self.save:
            print(f"{OKCYAN}doing Enrichr GO analysis for each module...{ENDC}")
            if 'gene_name' not in self.datExpr.var.columns:
                print(
                    f"{WARNING}\tgene name didn't found in gene information!\n\t Go term analysis can not be done{ENDC}")
            else:
                for module in modules:
                    self.functional_enrichment_analysis(type="GO",
                                                        moduleName=module)
            print("\tDone..\n")
        '''

    @staticmethod
    def replaceMissing(x, replaceWith):
        """
        Replacing missing (NA) value with appropriate value (for integer number replace with 0 and for string replace with "")

        :param x: value want to replace (single item)
        :type x: object
        :param replaceWith: define character you want to replace na value by looking at type of data
        :type replaceWith: object

        :return: object without any missing (NA) value
        """
        if replaceWith:
            if x.isnumeric():
                replaceWith = 0
            elif x.isalpha():
                replaceWith = ""
            else:
                sys.exit("Need 'replaceWith'.")

        x = x.fillna(replaceWith)
        return x

    @staticmethod
    def checkAndScaleWeights(weights, expr, scaleByMax=True):
        """
        check and scale weights of gene expression
        :param weights: weights of gene expression
        :type weights: pandas dataframe
        :param expr: gene expression matrix
        :type expr: pandas dataframe
        :param scaleByMax: if you want to scale your weights by diving to max
        :type scaleByMax: boll

        :return: processed weights of gene expression
        :rtype: pandas dataframe
        """
        if weights is None:
            return weights

        weights = np.asmatrix(weights)
        if expr.shape != weights.shape:
            sys.exit("When 'weights' are given, they must have the same dimensions as 'expr'.")
        if (weights < 0).any():
            sys.exit("Found negative weights. All weights must be non-negative.")

        nf = np.isinf(weights)
        if any(nf):
            print(f"{WARNING}Found non-finite weights. The corresponding data points will be removed.{ENDC}")
            weights[nf] = None

        if scaleByMax:
            maxw = np.amax(weights, axis=0)
            maxw[maxw == 0] = 1
            weights = weights / np.reshape(maxw, weights.shape)

        return weights

    # Check that all genes and samples have sufficiently low numbers of missing values.
    @staticmethod
    def goodSamplesGenes(datExpr, weights=None, minFraction=1 / 2, minNSamples=4, minNGenes=4, tol=None,
                         minRelativeWeight=0.1):
        """
        Checks data for missing entries, entries with weights below a threshold, and zero-variance genes. If necessary, the filtering is iterated.

        :param datExpr:expression data. A data frame in which columns are samples and rows are gene.
        :type datExpr: pandas dataframe
        :param weights: optional observation weights in the same format (and dimensions) as datExpr.
        :type weights: pandas dataframe
        :param minFraction: minimum fraction of non-missing samples for a gene to be considered good. (default = 1/2)
        :type minFraction: float
        :param minNSamples: minimum number of non-missing samples for a gene to be considered good. (default = 4)
        :type minNSamples: int
        :param minNGenes: minimum number of good genes for the data set to be considered fit for analysis. If the actual number of good genes falls below this threshold, an error will be issued. (default = 4)
        :type minNGenes: int
        :param tol: An optional 'small' number to compare the variance against
        :type tol: float
        :param minRelativeWeight: observations whose relative weight is below this threshold will be considered missing. Here relative weight is weight divided by the maximum weight in the column (gene). (default = 0.1)
        :type minRelativeWeight: float

        :return: A triple containing (goodGenes, goodSamples, allOK) goodSamples: A logical vector with one entry per sample that is TRUE if the sample is considered good and FALSE otherwise. goodGenes: A logical vector with one entry per gene that is TRUE if the gene is considered good and FALSE otherwise. allOK: if everything is okay
        :rtype: list, list, bool
        """
        goodGenes = None
        goodSamples = None
        nBadGenes = 0
        nBadSamples = 0
        changed = True
        iter = 1
        print("\tDetecting genes and samples with too many missing values...", flush=True)
        while changed:
            goodGenes = pyWGCNA.goodGenesFun(datExpr, weights, goodSamples, goodGenes, minFraction=minFraction,
                                           minNSamples=minNSamples, minNGenes=minNGenes,
                                           minRelativeWeight=minRelativeWeight,
                                           tol=tol)
            goodSamples = pyWGCNA.goodSamplesFun(datExpr, weights, goodSamples, goodGenes, minFraction=minFraction,
                                               minNSamples=minNSamples, minNGenes=minNGenes,
                                               minRelativeWeight=minRelativeWeight)
            changed = np.logical_or((np.logical_not(goodGenes).sum() > nBadGenes),
                                    (np.logical_not(goodSamples).sum() > nBadSamples))
            nBadGenes = np.logical_not(goodGenes).sum()
            nBadSamples = np.logical_not(goodSamples).sum()
            iter = iter + 1

        allOK = (nBadGenes + nBadSamples == 0)

        return goodGenes, goodSamples, allOK

    # Filter genes with too many missing entries
    @staticmethod
    def goodGenesFun(datExpr, weights=None, useSamples=None, useGenes=None, minFraction=1 / 2,
                     minNSamples=4, minNGenes=4, tol=None, minRelativeWeight=0.1):
        """
        Check data for missing entries and returns a list of genes that have non-zero variance

        :param datExpr:expression data. A data frame in which columns are genes and rows ar samples.
        :type datExpr: pandas dataframe
        :param weights: optional observation weights in the same format (and dimensions) as datExpr.
        :type weights: pandas dataframe
        :param useSamples: optional specifications of which samples to use for the check (Defaults to using all samples)
        :type useSamples: list of bool
        :param useGenes: optional specifications of genes for which to perform the check (Defaults to using all genes)
        :type useGenes: list of bool
        :param minFraction: minimum fraction of non-missing samples for a gene to be considered good. (default = 1/2)
        :type minFraction: float
        :param minNSamples: minimum number of non-missing samples for a gene to be considered good. (default = 4)
        :type minNSamples: int
        :param minNGenes: minimum number of good genes for the data set to be considered fit for analysis. If the actual number of good genes falls below this threshold, an error will be issued. (default = 4)
        :type minNGenes: int
        :param tol: An optional 'small' number to compare the variance against
        :type tol: float
        :param minRelativeWeight: observations whose relative weight is below this threshold will be considered missing. Here relative weight is weight divided by the maximum weight in the column (gene). (default = 0.1)
        :type minRelativeWeight: float

        :return: A logical list with one entry per gene that is TRUE if the gene is considered good and FALSE otherwise. Note that all genes excluded by useGenes are automatically assigned FALSE.
        :rtype: list of bool
        """
        if not datExpr.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).all():
            sys.exit("datExpr must contain numeric data.")

        weights = pyWGCNA.checkAndScaleWeights(weights, datExpr, scaleByMax=True)

        if tol is None:
            tol = 1e-10 * datExpr.abs().max().max()
        if useGenes is None:
            useGenes = np.repeat(True, datExpr.shape[0])
        if useSamples is None:
            useSamples = np.repeat(True, datExpr.shape[1])

        if len(useGenes) != datExpr.shape[0]:
            sys.exit("Length of nGenes is not compatible with number of columns in datExpr.")
        if len(useSamples) != datExpr.shape[1]:
            sys.exit("Length of nSamples is not compatible with number of rows in datExpr.")

        nSamples = sum(useSamples)
        nGenes = sum(useGenes)
        if weights is None:
            nPresent = datExpr.loc[useGenes, useSamples].notna().sum(axis=1)
        else:
            nPresent = (datExpr.loc[useGenes, useSamples].notna() and
                        weights.loc[useGenes, useSamples] > minRelativeWeight).sum(axis=1)

        gg = useGenes.copy()
        if len(useGenes) != len(nPresent):
            tmp = list(useGenes.copy())
            ind = [i for i in range(len(tmp)) if tmp[i]]
            tmp = pd.Series(data=tmp)
            index = np.asarray(tmp.index.tolist(), dtype=object)
            index[ind] = nPresent.index.tolist()
            tmp.index = index
            tmp.loc[nPresent.index] = nPresent.values
            nPresent = tmp.copy()

        gg[np.logical_and(useGenes, nPresent < minNSamples)] = False

        if weights is None:
            var = np.var(datExpr.loc[gg, useSamples], axis=1)
            # var = var.sort_index(inplace=True)
        else:
            # need to be fix
            # TODO:colWeightedVars
            var = np.var(datExpr, w=weights)

        var[pd.isnull(var)] = 0
        nNAsGenes = datExpr.loc[gg, useSamples].isna().sum(axis=1)
        gg[gg] = np.logical_and(np.logical_and(nNAsGenes < (1 - minFraction) * nSamples, var > tol ** 2),
                                nSamples - nNAsGenes >= minNSamples)

        if sum(gg) < minNGenes:
            sys.exit("Too few genes with valid expression levels in the required number of samples.")
        if nGenes - sum(gg) > 0:
            print("\n\n  ..Excluding", nGenes - sum(gg),
                  "genes from the calculation due to too many missing samples or zero variance.\n\n", flush=True)

        return gg

    # Filter samples with too many missing entries
    @staticmethod
    def goodSamplesFun(datExpr, weights=None, useSamples=None, useGenes=None, minFraction=1 / 2,
                       minNSamples=4, minNGenes=4, minRelativeWeight=0.1):
        """
        Check data for missing entries and returns a list of samples that have non-zero variance

        :param datExpr:expression data. A data frame in which columns are genes and rows ar samples.
        :type datExpr: pandas dataframe
        :param weights: optional observation weights in the same format (and dimensions) as datExpr.
        :type weights: pandas dataframe
        :param useSamples: optional specifications of which samples to use for the check (Defaults to using all samples)
        :type useSamples: list of bool
        :param useGenes: optional specifications of genes for which to perform the check (Defaults to using all genes)
        :type useGenes: list of bool
        :param minFraction: minimum fraction of non-missing samples for a gene to be considered good. (default = 1/2)
        :type minFraction: float
        :param minNSamples: findModulesminimum number of non-missing samples for a gene to be considered good. (default = 4)
        :type minNSamples: int
        :param minNGenes: minimum number of good genes for the data set to be considered fit for analysis. If the actual number of good genes falls below this threshold, an error will be issued. (default = 4)
        :type minNGenes: int
        :param minRelativeWeight: observations whose relative weight is below this threshold will be considered missing. Here relative weight is weight divided by the maximum weight in the column (gene). (default = 0.1)
        :type minRelativeWeight: float

        :return: A logical list with one entry per sample that is TRUE if the sample is considered good and FALSE otherwise. Note that all samples excluded by useSamples are automatically assigned FALSE.
        :rtype: list of bool
        """
        if useGenes is None:
            useGenes = np.repeat(True, datExpr.shape[0])

        if useSamples is None:
            useSamples = np.repeat(True, datExpr.shape[1])

        if len(useGenes) != datExpr.shape[0]:
            sys.exit("Length of nGenes is not compatible with number of columns in datExpr.")

        if len(useSamples) != datExpr.shape[1]:
            sys.exit("Length of nSamples is not compatible with number of rows in datExpr.")

        weights = pyWGCNA.checkAndScaleWeights(weights, datExpr, scaleByMax=True)
        nSamples = sum(useSamples)
        nGenes = sum(useGenes)
        if weights is None:
            nNAsSamples = np.sum((datExpr.loc[useGenes, useSamples]).isnull(), axis=0)
        else:
            nNAsSamples = np.sum(np.logical_or(datExpr[useGenes, useSamples],
                                               pyWGCNA.replaceMissing(weights[useGenes, useSamples] < minRelativeWeight,
                                                                    True))
                                 .isnull(), axis=0)

        goodSamples = useSamples
        goodSamples[useSamples] = np.logical_and((nNAsSamples < (1 - minFraction) * nGenes),
                                                 (nGenes - nNAsSamples >= minNGenes))

        if sum(goodSamples) < minNSamples:
            sys.exit("Too few samples with valid expression levels for the required number of genes.")

        if nSamples - sum(goodSamples) > 0:
            print("  ..Excluding", nSamples - sum(goodSamples),
                  "samples from the calculation due to too many missing genes.", flush=True)

        return goodSamples

    @staticmethod
    def hclust(d, method="complete"):
        """
        Hierarchical cluster analysis on a set of dissimilarities and methods for analyzing it.

        :param d: a dissimilarity structure as produced by 'pdist'.
        :type d: ndarray
        :param method: The linkage algorithm to use. (default = complete)
        :type method: str

        :return: The hierarchical clustering encoded as a linkage matrix.
        :rtype: ndarray
        """
        METHODS = ["single", "complete", "average", "weighted", "centroid"]

        if method not in METHODS:
            sys.exit("Invalid clustering method.")

        if method == -1:
            sys.exit("Ambiguous clustering method.")

        dendrogram = linkage(d, method=method)

        return dendrogram

    # Determine cluster under the line
    @staticmethod
    def cutree(sampleTree, cutHeight=50000.0):
        """
        Given a linkage matrix Z, return the cut tree. remove samples/genes/modules base on hierarchical clustering

        :param sampleTree: The linkage matrix.
        :type sampleTree: scipy.cluster.linkage array
        :param cutHeight: A optional height at which to cut the tree (default = 50000)
        :type cutHeight: array_like

        :return: An array indicating group membership at each agglomeration step. I.e., for a full cut tree, in the first column each data point is in its own cluster. At the next step, two nodes are merged. Finally, all singleton and non-singleton clusters are in one group. If n_clusters or height are given, the columns correspond to the columns of n_clusters or height.
        :rtype: array
        """
        cutTree = cut_tree(sampleTree, height=cutHeight)

        return cutTree

    # Call the network topology analysis function
    @staticmethod
    def pickSoftThreshold(data, dataIsExpr=True, weights=None, RsquaredCut=0.85, MeanCut=100, powerVector=None,
                          nBreaks=10,
                          blockSize=None, corOptions=None, networkType="unsigned", moreNetworkConcepts=False,
                          gcInterval=None):
        """
        Analysis of scale free topology for multiple soft thresholding powers.

        :param data: expression data in a matrix or data frame. Rows correspond to samples and columns to genes.
        :param data: pandas dataframe
        :param dataIsExpr: should the data be interpreted as expression (or other numeric) data, or as a similarity matrix of network nodes?
        :type dataIsExpr: bool
        :param weights: optional observation weights for data to be used in correlation calculation. A matrix of the same dimensions as datExpr, containing non-negative weights. Only used with Pearson correlation.
        :type weights: pandas dataframe
        :param RsquaredCut: desired minimum scale free topology fitting index (R^2). (default = 0.9)
        :type RsquaredCut: float
        :param MeanCut: desired maximum mean connectivity scale free topology fitting index. (default = 100)
        :type MeanCut: int
        :param powerVector: A list of soft thresholding powers for which the scale free topology fit indices are to be calculated.
        :type powerVector: list of int
        :param nBreaks: number of bins in connectivity histograms (default = 10)
        :type nBreaks: int
        :param blockSize: block size into which the calculation of connectivity should be broken up. If not given, a suitable value will be calculated using function blockSize and printed if verbose>0. If R runs into memory problems, decrease this value.
        :type blockSize: int
        :param corOptions: a list giving further options to the correlation function specified in corFnc.
        :type corOptions: list
        :param networkType: network type. Allowed values are (unique abbreviations of) "unsigned", "signed", "signed hybrid". (default = unsigned)
        :type networkType: str
        :param moreNetworkConcepts: should additional network concepts be calculated? If TRUE, the function will calculate how the network density, the network heterogeneity, and the network centralization depend on the power. For the definition of these additional network concepts, see Horvath and Dong (2008). PloS Comp Biol.
        :type moreNetworkConcepts: bool
        :param gcInterval: a number specifying in interval (in terms of individual genes) in which garbage collection will be performed. The actual interval will never be less than blockSize.
        :type gcInterval: int

        :return: tuple including powerEstimate: estimate of an appropriate soft-thresholding power which is the lowest power for which the scale free topology fit \(R^2\) exceeds RsquaredCut and conectivity is less than MeanCut. If \(R^2\) is below RsquaredCut for all powers maximum will re returned and datout which is a data frame containing the fit indices for scale free topology. The columns contain the soft-thresholding power, adjusted \(R^2\) for the linear fit, the linear coefficient, adjusted \(R^2\) for a more complicated fit models, mean connectivity, median connectivity and maximum connectivity. If input moreNetworkConcepts is TRUE, 3 additional columns containing network density, centralization, and heterogeneity.
        :type: int and pandas dataframe
        """
        if powerVector is None:
            powerVector = list(range(1, 11)) + list(range(1, 21, 2))
        powerVector = np.sort(powerVector)
        intType = networkTypes.index(networkType)
        if intType is None:
            sys.exit(("Unrecognized 'networkType'. Recognized values are", str(networkTypes)))

        nGenes = data.shape[1]
        if nGenes < 3:
            sys.exit("The input data data contain fewer than 3 rows (nodes).\n"
                     "This would result in a trivial correlation network.")

        print(f"{OKCYAN}pickSoftThreshold: calculating connectivity for given powers...{ENDC}")

        if not dataIsExpr:
            pyWGCNA.checkSimilarity(data)
            if any(np.diag(data) != 1):
                data = np.where(np.diag(data), 1)

        if blockSize is None:
            blockSize = pyWGCNA.calBlockSize(nGenes, rectangularBlocks=True, maxMemoryAllocation=2 ** 30)
            print("will use block size ", blockSize, flush=True)

        if gcInterval is None or len(gcInterval) == 0:
            gcInterval = 4 * blockSize

        colname1 = ["Power", "SFT.R.sq", "slope", "truncated R.sq", "mean(k)", "median(k)", "max(k)"]

        if moreNetworkConcepts:
            colname1 = colname1.append(["Density", "Centralization", "Heterogeneity"])

        datout = pd.DataFrame(np.full((len(powerVector), len(colname1)), 666), columns=colname1, dtype=object)
        datout['Power'] = powerVector

        datk = np.zeros((nGenes, len(powerVector)))
        nPowers = len(powerVector)
        startG = 0
        lastGC = 0
        if corOptions is None:
            corOptions = pd.DataFrame()
            corOptions['x'] = [data]
        else:
            corOptions['x'] = [data]

        if weights is not None:
            if not dataIsExpr:
                sys.exit("Weights can only be used when 'data' represents expression data ('dataIsExpr' must be TRUE).")
            if data.shape != weights.shape:
                sys.exit("When 'weights' are given, dimensions of 'data' and 'weights' must be the same.")
            corOptions['weights.x'] = weights

        while startG < nGenes:
            endG = min(startG + blockSize, nGenes)

            useGenes = list(range(startG, endG))
            nGenes1 = len(useGenes)
            if dataIsExpr:
                corOptions['y'] = [data.iloc[:, useGenes]]
                if weights is not None:
                    corOptions['weights.y'] = [weights.iloc[:, useGenes]]
                # Direct cross-correlation cor(all genes, this block): the
                # standardised dot product equals Pearson r exactly, but its
                # peak footprint is (nGenes x blockSize) instead of the
                # (nGenes + blockSize)^2 matrix np.corrcoef would build over
                # the concatenation — which is what makes large gene counts OOM.
                _Xf = np.asarray(corOptions.x[0], dtype=float)   # samples x nGenes
                _Yb = np.asarray(corOptions.y[0], dtype=float)   # samples x block
                _Xc = _Xf - _Xf.mean(axis=0, keepdims=True)
                _Yc = _Yb - _Yb.mean(axis=0, keepdims=True)
                _Xn = _Xc / np.sqrt((_Xc ** 2).sum(axis=0, keepdims=True))
                _Yn = _Yc / np.sqrt((_Yc ** 2).sum(axis=0, keepdims=True))
                corx = _Xn.T @ _Yn                               # nGenes x block
                del _Xf, _Yb, _Xc, _Yc, _Xn, _Yn
                if intType == 0:
                    corx = abs(corx)
                elif intType == 1:
                    corx = (1 + corx) / 2
                elif intType == 2:
                    corx[corx < 0] = 0

                if np.count_nonzero(np.isnan(corx)) != 0:
                    print(
                        f"{WARNING}Some correlations are NA in block {str(startG)} : {str(endG)}.{ENDC}")
            else:
                corx = data.iloc[:, useGenes].to_numpy()

            corx[useGenes, list(range(len(useGenes)))] = 1
            datk_local = np.empty((nGenes1, nPowers))
            datk_local[:] = np.nan
            corxPrev = np.ones(corx.shape)
            powerVector1 = [0]
            powerVector1.extend(powerVector[:-1])
            powerSteps = powerVector - powerVector1
            uniquePowerSteps = np.unique(powerSteps)

            def func(power):
                return corx ** power

            corxPowers = pd.DataFrame()
            for p in uniquePowerSteps:
                corxPowers[p] = [func(p)]

            for j in range(nPowers):
                corxCur = corxPrev * corxPowers[powerSteps[j]][0]
                datk_local[:, j] = np.nansum(corxCur, axis=0) - 1
                corxPrev = corxCur

            datk[startG:endG, :] = datk_local

            startG = endG
            if 0 < gcInterval < startG - lastGC:
                lastGC = startG

        for i in range(len(powerVector)):
            khelp = datk[:, i]
            SFT1 = pyWGCNA.scaleFreeFitIndex(k=khelp, nBreaks=nBreaks)
            datout.loc[i, 'SFT.R.sq'] = SFT1.loc[0, 'Rsquared.SFT']
            datout.loc[i, 'slope'] = SFT1.loc[0, 'slope.SFT']
            datout.loc[i, 'truncated R.sq'] = SFT1.loc[0, 'truncatedExponentialAdjRsquared']
            datout.loc[i, 'mean(k)'] = statistics.mean(khelp)
            datout.loc[i, 'median(k)'] = statistics.median(khelp)
            datout.loc[i, 'max(k)'] = max(khelp)

            if moreNetworkConcepts:
                Density = sum(khelp) / (nGenes * (nGenes - 1))
                datout.loc[i, 'Density'] = Density
                Centralization = nGenes * (max(khelp) - statistics.mean(khelp)) / ((nGenes - 1) * (nGenes - 2))
                datout.loc[i, 'Centralization'] = Centralization
                Heterogeneity = np.sqrt(nGenes * sum(khelp ^ 2) / sum(khelp) ^ 2 - 1)
                datout.loc[i, 'Heterogeneity'] = Heterogeneity

        print(datout)

        # Scale-free topology selection must use the SIGNED fit index,
        # -sign(slope) * R^2 — not the raw R^2. A scale-free degree
        # distribution has a NEGATIVE slope; at power 1-2 the fit often
        # has a POSITIVE slope (not scale-free at all) yet a high raw
        # R^2. Selecting on raw R^2 therefore collapses to power 1, a
        # degenerate network where most genes fall into one giant
        # 'hairball' module.
        sft_r2 = datout['SFT.R.sq'].astype(float).to_numpy()
        slope = datout['slope'].astype(float).to_numpy()
        meanK = datout['mean(k)'].astype(float).to_numpy()
        signedR2 = -np.sign(slope) * sft_r2
        powerArr = np.asarray(powerVector)

        ind = np.logical_and(signedR2 >= RsquaredCut, meanK <= MeanCut)
        if np.sum(ind) > 0:
            powerEstimate = int(np.min(powerArr[ind]))
            print(f"{OKGREEN}Selected power to have scale free network is {str(powerEstimate)}.{ENDC}")
        else:
            # No power reaches the scale-free fit cut. Do NOT fall back
            # to the raw-R^2 argmax — that routinely returns power 1-2
            # and a degenerate hairball. Use the Langfelder sample-size
            # default (WGCNA FAQ "Choosing the soft-thresholding
            # power"), the recommended power when the fit cut is unmet.
            nSamples = data.shape[0]
            signed = str(networkType).lower() in ("signed", "signed hybrid")
            if signed:
                _default = 18 if nSamples < 20 else 16 if nSamples < 30 else 14 if nSamples < 40 else 12
            else:
                _default = 9 if nSamples < 20 else 8 if nSamples < 30 else 7 if nSamples < 40 else 6
            powerEstimate = int(min(max(_default, int(powerArr.min())), int(powerArr.max())))
            try:
                _best = float(np.nanmax(signedR2))
            except ValueError:
                _best = float('nan')
            print(f"{WARNING}No soft-threshold power reached the scale-free "
                  f"fit cut (best signed R^2 = {_best:.2f} < {RsquaredCut}). "
                  f"Falling back to the sample-size default power = "
                  f"{powerEstimate} ({networkType}, n={nSamples}). A weak "
                  f"scale-free fit usually means residual batch effect or "
                  f"heterogeneous samples in the expression matrix.{ENDC}")

        return powerEstimate, datout

    @staticmethod
    def checkSimilarity(adjMat, min=-1, max=1):
        """
        check similarity matrix format is correct

        :param adjMat: data we want to be checked
        :type adjMat: pandas dataframe
        :param min: minimum value to be allowed for data (default = 0)
        :type min: int
        :param max: maximum value to be allowed for data (default = 1)
        :type max: int

        :raises exit: if format is not correct
        """
        dim = adjMat.shape
        if dim is None or len(dim) != 2:
            sys.exit("adjacency is not two-dimensional")

        if not (all(np.array_equal(adjMat[ele], adjMat[ele].astype(float)) for ele in adjMat)):
            sys.exit("adjacency is not numeric")

        if dim[0] != dim[1]:
            sys.exit("adjacency is not square")

        if all(np.max(np.abs(adjMat - adjMat.transpose())) > 1e-12):
            sys.exit("adjacency is not symmetric")

        if all(np.min(adjMat) < min) or all(np.max(adjMat) > max):
            sys.exit(("some entries are not between", min, "and", max))

    @staticmethod
    def calBlockSize(matrixSize, rectangularBlocks=True, maxMemoryAllocation=None, overheadFactor=3):
        """
        find suitable block size for calculating soft power threshold
        """
        try:
            import resource
        except ImportError:
            try:
                import rsrc as resource
            except ImportError:
                sys.exit("resource or rsrc package is not installed!")
        if maxMemoryAllocation is None:
            maxAlloc = resource.getrlimit(resource.RLIMIT_AS)[1]
        else:
            maxAlloc = maxMemoryAllocation / 8

        maxAlloc = maxAlloc / overheadFactor

        if rectangularBlocks:
            blockSz = math.floor(maxAlloc / matrixSize)
        else:
            blockSz = math.floor(math.sqrt(maxAlloc))

        return min(matrixSize, blockSz)

    # Calculation of fitting statistics for evaluating scale free topology fit.
    @staticmethod
    def scaleFreeFitIndex(k, nBreaks=10):
        """
        calculates several indices (fitting statistics) for evaluating scale free topology fit.

        :param k: numeric list whose components contain non-negative values
        :type k: list
        :param nBreaks: (default = 10)
        :type nBreaks: int
        """
        df = pd.DataFrame({'data': k})
        df['discretized_k'] = pd.cut(df['data'], nBreaks)
        dk = df.groupby('discretized_k').mean()  # tapply(k, discretized_k, mean)
        dk = pd.DataFrame(dk.reset_index())
        dk.columns = ['discretized_k', 'dk']
        p_dk = df['discretized_k'].value_counts() / len(k)  # as.vector(tapply(k, discretized.k, length)/length(k))
        p_dk = pd.DataFrame(p_dk.reset_index())
        p_dk.columns = ['discretized_k', 'p_dk']
        breaks1 = np.linspace(start=min(k), stop=max(k), num=nBreaks + 1)
        y, edges = np.histogram(df['data'], bins=breaks1)
        dk2 = 0.5 * (edges[1:] + edges[:-1])
        df = pd.merge(dk, p_dk, on='discretized_k')
        if df['dk'].isnull().values.any():
            df.loc[df['dk'].isnull().values, 'dk'] = dk2[df['dk'].isnull().values]
        if np.any(df['dk'] == 0):
            df.loc[df['dk'] == 0, 'dk'] = dk2[df['dk'] == 0]
        if df['p_dk'].isnull().values.any():
            df.loc[df['p_dk'].isnull().values, 'p_dk'] = 0
        df['log_dk'] = np.log10(df['dk'])
        df['log_p_dk'] = np.log10(df['p_dk'] + 1e-09)
        df['log_p_dk_10'] = np.power(10, df['log_dk'])

        model1 = ols(formula='log_p_dk ~ log_dk', data=df).fit()
        model2 = ols(formula='log_p_dk ~ log_dk + log_p_dk_10', data=df).fit()
        dfout = pd.DataFrame({'Rsquared.SFT': [model1.rsquared],
                              'slope.SFT': [model1.params.values[1]],
                              'truncatedExponentialAdjRsquared': [model2.rsquared_adj]})
        return dfout

    @staticmethod
    def adjacency(datExpr, selectCols=None, adjacencyType="unsigned", power=6, corOptions=pd.DataFrame(), weights=None,
                  weightArgNames=None):
        """
        Calculates (correlation or distance) network adjacency from given expression data or from a similarity

        :param datExpr: data frame containing expression data. Columns correspond to genes and rows to samples.
        :type datExpr: pandas dataframe
        :param selectCols: for correlation networks only; can be used to select genes whose adjacencies will be calculated. Should be either a numeric list giving the indices of the genes to be used, or a boolean list indicating which genes are to be used.
        :type selectCols: list
        :param adjacencyType: adjacency network type. Allowed values are (unique abbreviations of) "unsigned", "signed", "signed hybrid". (default = unsigned)
        :type adjacencyType: str
        :param power: soft thresholding power.
        :type power: int
        :param corOptions: specifying additional arguments to be passed to the function given by corFnc.
        :type corOptions: pandas dataframe
        :param weights: optional observation weights for datExpr to be used in correlation calculation. A matrix of the same dimensions as datExpr, containing non-negative weights. Only used with Pearson correlation.
        :type weights: pandas dataframe
        :param weightArgNames: character list of length 2 giving the names of the arguments to corFnc that represent weights for variable x and y. Only used if weights are non-NULL.
        :type weightArgNames: list

        :return: Adjacency matrix
        :rtype: pandas dataframe
        """
        print(f"{OKCYAN}calculating adjacency matrix ...{ENDC}")
        if weightArgNames is None:
            weightArgNames = ["weights.x", "weights.y"]
        intType = adjacencyTypes.index(adjacencyType)
        if intType is None:
            sys.exit(("Unrecognized 'type'. Recognized values are", str(adjacencyTypes)))
        weights = pyWGCNA.checkAndScaleWeights(weights, datExpr, scaleByMax=False)
        if weights is not None:
            if selectCols is None:
                if isinstance(corOptions, pd.DataFrame):
                    weightOpt = pd.DataFrame({'weights.x': weights})
                    weightOpt.index = weightArgNames[0]
                else:
                    weightOpt = weightArgNames[0] + " = weights"
            else:
                if isinstance(corOptions, pd.DataFrame):
                    weightOpt = pd.DataFrame({'weights.x': weights, 'weights.y': weights[:, selectCols]})
                    weightOpt.index = weightArgNames[0:2]
                else:
                    weightOpt = weightArgNames[1] + " = weights, " + weightArgNames[2] + " = weights[, selectCols]"
        else:
            if isinstance(corOptions, pd.DataFrame):
                weightOpt = pd.DataFrame()
            else:
                weightOpt = ""

        if selectCols is None:
            cor_mat = np.corrcoef(
                datExpr.T)  # cor_mat = do.call(corFnc.fnc, c(list(x = datExpr), weightOpt, corOptions))
        else:
            cor_mat = np.corrcoef(x=datExpr, y=datExpr[:, selectCols])  # , weightOpt, corOptions)

        if intType == 0:
            cor_mat = abs(cor_mat)
        elif intType == 1:
            cor_mat = (1 + cor_mat) / 2
        elif intType == 2:
            cor_mat[cor_mat < 0] = 0

        print("\tDone..\n")

        return cor_mat ** power

    @staticmethod
    def checkAdjMat(adjMat, min=0, max=1):
        """
        check adjacency matrix format is correct

        :param adjMat: data we want to be checked
        :type adjMat: pandas dataframe
        :param min: minimum value to be allowed for data (default = 0)
        :type min: int
        :param max: maximum value to be allowed for data (default = 1)
        :type max: int

        :raises exit: if format is not correct
        """
        shape = adjMat.shape
        if shape is None or len(shape) != 2:
            sys.exit("adjacency is not two-dimensional")

        if not issubclass(adjMat.dtype.type, np.floating):
            sys.exit("adjacency is not numeric")
        if shape[0] != shape[1]:
            sys.exit("adjacency is not square")
        if np.max(np.fabs(np.subtract(adjMat, adjMat.T))) > 1e-12:
            sys.exit("adjacency is not symmetric")
        if np.min(adjMat) < min or np.max(adjMat) > max:
            sys.exit(("some entries are not between", min, "and", max))

    @staticmethod
    def TomSimilarityFromAdj(adjMat, TOMDenom, TOMType):
        # Prepare adjacency
        np.fill_diagonal(adjMat, 0)
        # Compute TOM
        L = adjMat @ adjMat
        ki = adjMat.sum(axis=1)
        kj = adjMat.sum(axis=0)
        if TOMDenom == 0:  # min
            MINK = np.array([np.minimum(ki_, kj) for ki_ in ki])
        else:  # mean
            MINK = np.array([((ki_ + kj) / 2) for ki_ in ki])
        if TOMType == 0:  # unsigned
            tom = (L + adjMat) / (MINK + 1 - adjMat)
        else:  # signed
            tom = np.fabs((L + adjMat)) / (MINK + 1 - np.fabs(adjMat))
        np.fill_diagonal(tom, 1)
        return tom

    @staticmethod
    def TOMsimilarity(adjMat, TOMType="signed", TOMDenom="min"):
        """
        Calculation of the topological overlap matrix, and the corresponding dissimilarity, from a given adjacency matrix

        :param adjMat: adjacency matrix, that is a square, symmetric matrix with entries between 0 and 1 (negative values are allowed if TOMType=="signed").
        :type adjMat: pandas dataframe
        :param TOMType: one of "unsigned", "signed"
        :type TOMType: str
        :param TOMDenom: a character string specifying the TOM variant to be used. Recognized values are "min" giving the standard TOM described in Zhang and Horvath (2005), and "mean" in which the min function in the denominator is replaced by mean. The "mean" may produce better results but at this time should be considered experimental.
        :type TOMDenom: str

        :return: A matrix holding the topological overlap.
        :rtype: pandas dataframe
        """
        TOMTypeC = TOMTypes.index(TOMType)
        if TOMTypeC is None:
            sys.exit(f"Invalid 'TOMType'. Recognized values are {', '.join(TOMTypes)}")
        TOMDenomC = TOMDenoms.index(TOMDenom)
        if TOMDenomC is None:
            sys.exit(f"Invalid 'TOMDenom'. Recognized values are {', '.join(TOMDenoms)}")

        min = 0
        if TOMTypeC == 1:
            min = -1
        pyWGCNA.checkAdjMat(adjMat, min=min, max=1)
        np.nan_to_num(adjMat, copy=False, nan=0)

        print(f"{OKCYAN}calculating TOM similarity matrix ...{ENDC}")

        tom = pyWGCNA.TomSimilarityFromAdj(adjMat, TOMDenomC, TOMTypeC)

        print("\tDone..\n")

        return pd.DataFrame(tom)

    @staticmethod
    def interpolate(data, index):
        i = round(index)
        n = len(data)
        if i < 1:
            return data[1]
        if i >= n:
            return data[n]
        r = index - i
        return data[i] * (1 - r) + data[i + 1] * r

    @staticmethod
    def coreSizeFunc(BranchSize, minClusterSize):
        BaseCoreSize = minClusterSize / 2 + 1
        if BaseCoreSize < BranchSize:
            CoreSize = int(BaseCoreSize + math.sqrt(BranchSize - BaseCoreSize))
        else:
            CoreSize = BranchSize

        return CoreSize

    @staticmethod
    def get_heights(Z):

        clusternode = to_tree(Z, True)
        height = np.array([c.dist for c in clusternode[1] if c.is_leaf() is not True])

        return height

    @staticmethod
    def get_merges(z):
        n = z.shape[0]
        merges = np.zeros((z.shape[0], 2), dtype=int)

        for i in range(z.shape[0]):
            for j in range(2):
                if z[i][j] <= n:
                    merges[i][j] = -(z[i][j] + 1)
                else:
                    cluster = z[i][j] - n
                    merges[i][j] = cluster

        return merges

    @staticmethod
    def cutreeHybrid(dendro, distM, cutHeight=None, minClusterSize=20, deepSplit=1,
                     maxCoreScatter=None, minGap=None, maxAbsCoreScatter=None,
                     minAbsGap=None, minSplitHeight=None, minAbsSplitHeight=None,
                     externalBranchSplitFnc=None, nExternalSplits=0, minExternalSplit=None,
                     externalSplitOptions=pd.DataFrame(), externalSplitFncNeedsDistance=None,
                     assumeSimpleExternalSpecification=True, pamStage=True,
                     pamRespectsDendro=True, useMedoids=False, maxPamDist=None,
                     respectSmallClusters=True):
        """
        Detect clusters in a dendorgram produced by the function hclust.

        :param dendro: a hierarchical clustering dendorgram such as one returned by hclust.
        :type dendro: ndarray
        :param distM: Distance matrix that was used as input to hclust.
        :type distM: pandas dataframe
        :param cutHeight: Maximum joining heights that will be considered. It defaults to 99of the range between the 5th percentile and the maximum of the joining heights on the dendrogram.
        :type cutHeight: int
        :param minClusterSize: Minimum cluster size. (default = 20)
        :type minClusterSize: int
        :param deepSplit: Either logical or integer in the range 0 to 4. Provides a rough control over sensitivity to cluster splitting. The higher the value, the more and smaller clusters will be produced. (default = 1)
        :type deepSplit: int or bool
        :param maxCoreScatter: Maximum scatter of the core for a branch to be a cluster, given as the fraction of cutHeight relative to the 5th percentile of joining heights.
        :type maxCoreScatter: int
        :param minGap: Minimum cluster gap given as the fraction of the difference between cutHeight and the 5th percentile of joining heights.
        :type minGap: int
        :param maxAbsCoreScatter: Maximum scatter of the core for a branch to be a cluster given as absolute heights. If given, overrides maxCoreScatter.
        :type maxAbsCoreScatter: int
        :param minAbsGap: Minimum cluster gap given as absolute height difference. If given, overrides minGap.
        :type minAbsGap: int
        :param minSplitHeight: Minimum split height given as the fraction of the difference between cutHeight and the 5th percentile of joining heights. Branches merging below this height will automatically be merged. Defaults to zero but is used only if minAbsSplitH
        :type minSplitHeight: int
        :param minAbsSplitHeight: Minimum split height given as an absolute height. Branches merging below this height will automatically be merged. If not given (default), will be determined from minSplitHeight above.
        :type minAbsSplitHeight: int
        :param externalBranchSplitFnc: Optional function to evaluate split (dissimilarity) between two branches. Either a single function or a list in which each component is a function.
        :param minExternalSplit: Thresholds to decide whether two branches should be merged. It should be a numeric list of the same length as the number of functions in externalBranchSplitFnc above.
        :type minExternalSplit: list
        :param externalSplitOptions: Further arguments to function externalBranchSplitFnc. If only one external function is specified in externalBranchSplitFnc above, externalSplitOptions can be a named list of arguments or a list with one component.
        :type externalSplitOptions: pandas dataframe
        :param externalSplitFncNeedsDistance: Optional specification of whether the external branch split functions need the distance matrix as one of their arguments. Either NULL or a logical list with one element per branch
        :type externalSplitFncNeedsDistance: pandas dataframe
        :param assumeSimpleExternalSpecification: when minExternalSplit above is a scalar (has length 1), should the function assume a simple specification of externalBranchSplitFnc and externalSplitOptions. (default = True)
        :type assumeSimpleExternalSpecification: bool
        :param pamStage: If TRUE, the second (PAM-like) stage will be performed. (default = True)
        :type pamStage: bool
        :param pamRespectsDendro: If TRUE, the PAM stage will respect the dendrogram in the sense an object can be PAM-assigned only to clusters that lie below it on the branch that the object is merged into. (default = True)
        :type pamRespectsDendro: bool
        :param useMedoids: if TRUE, the second stage will be use object to medoid distance; if FALSE, it will use average object to cluster distance. (default = False)
        :param maxPamDist: Maximum object distance to closest cluster that will result in the object assigned to that cluster. Defaults to cutHeight.
        :type maxPamDist: float
        :param respectSmallClusters: If TRUE, branches that failed to be clusters in stage 1 only because of insufficient size will be assigned together in stage 2. If FALSE, all objects will be assigned individually. (default = False)
        :type respectSmallClusters: bool

        :return: list detailing the deteced branch structure.
        :rtype: list
        """
        dendro_height = pyWGCNA.get_heights(dendro)
        dendro_merge = pyWGCNA.get_merges(dendro)

        chunkSize = dendro.shape[0]

        if maxPamDist is None:
            maxPamDist = cutHeight

        nMerge = dendro.shape[0]
        if nMerge < 1:
            sys.exit("The given dendrogram is suspicious: number of merges is zero.")
        if distM is None:
            sys.exit("distM must be non-NULL")
        if distM.shape is None:
            sys.exit("distM must be a matrix.")
        if distM.shape[0] != nMerge + 1 or distM.shape[1] != nMerge + 1:
            sys.exit("distM has incorrect dimensions.")
        if pamRespectsDendro and not respectSmallClusters:
            print("cutreeHybrid Warning: parameters pamRespectsDendro (TRUE) "
                  "and respectSmallClusters (FALSE) imply contradictory intent.\n"
                  "Although the code will work, please check you really intented "
                  "these settings for the two arguments.", flush=True)

        print(f"{OKCYAN}Going through the merge tree...{ENDC}")

        if any(np.diag(distM) != 0):
            np.fill_diagonal(distM, 0)
        refQuantile = 0.05
        refMerge = round(nMerge * refQuantile) - 1
        if refMerge < 0:
            refMerge = 0
        refHeight = dendro[refMerge, 2]
        if cutHeight is None:
            cutHeight = 0.99 * (np.max(dendro_height) - refHeight) + refHeight
            print("..cutHeight not given, setting it to", round(cutHeight, 3),
                  " ===>  99% of the (truncated) height range in dendro.", flush=True)
        else:
            if cutHeight > np.max(dendro_height):
                cutHeight = np.max(dendro_height)
        if maxPamDist is None:
            maxPamDist = cutHeight
        nMergeBelowCut = np.count_nonzero(dendro_height <= cutHeight)
        if nMergeBelowCut < minClusterSize:
            print("cutHeight set too low: no merges below the cut.", flush=True)
            return pd.DataFrame({'labels': np.repeat(0, nMerge + 1, axis=0)})

        if externalBranchSplitFnc is not None:
            nExternalSplits = len(externalBranchSplitFnc)
            if len(minExternalSplit) < 1:
                sys.exit("'minExternalBranchSplit' must be given.")
            if assumeSimpleExternalSpecification and nExternalSplits == 1:
                externalSplitOptions = pd.DataFrame(externalSplitOptions)
            # TODO: externalBranchSplitFnc = lapply(externalBranchSplitFnc, match.fun)
            for es in range(nExternalSplits):
                externalSplitOptions['tree'][es] = dendro
                if len(externalSplitFncNeedsDistance) == 0 or externalSplitFncNeedsDistance[es]:
                    externalSplitOptions['dissimMat'][es] = distM

        MxBranches = nMergeBelowCut
        branch_isBasic = np.repeat(True, MxBranches, axis=0)
        branch_isTopBasic = np.repeat(True, MxBranches, axis=0)
        branch_failSize = np.repeat(False, MxBranches, axis=0)
        branch_rootHeight = np.repeat(np.nan, MxBranches, axis=0)
        branch_size = np.repeat(2, MxBranches, axis=0)
        branch_nMerge = np.repeat(1, MxBranches, axis=0)
        branch_nSingletons = np.repeat(2, MxBranches, axis=0)
        branch_nBasicClusters = np.repeat(0, MxBranches, axis=0)
        branch_mergedInto = np.repeat(0, MxBranches, axis=0)
        branch_attachHeight = np.repeat(np.nan, MxBranches, axis=0)
        branch_singletons = [np.nan] * MxBranches
        branch_basicClusters = [np.nan] * MxBranches
        branch_mergingHeights = [np.nan] * MxBranches
        branch_singletonHeights = [np.nan] * MxBranches
        nBranches = 0

        defMCS = [0.64, 0.73, 0.82, 0.91, 0.95]
        defMG = [(1.0 - defMC) * 3.0 / 4.0 for defMC in defMCS]
        nSplitDefaults = len(defMCS)
        if isinstance(deepSplit, bool):
            deepSplit = pd.to_numeric(deepSplit) * (nSplitDefaults - 2)
        if deepSplit < 0 or deepSplit > nSplitDefaults:
            msg = "Parameter deepSplit (value" + str(deepSplit) + \
                  ") out of range: allowable range is 0 through", str(nSplitDefaults - 1)
            sys.exit(msg)
        if maxCoreScatter is None:
            maxCoreScatter = pyWGCNA.interpolate(defMCS, deepSplit)
        if minGap is None:
            minGap = pyWGCNA.interpolate(defMG, deepSplit)
        if maxAbsCoreScatter is None:
            maxAbsCoreScatter = refHeight + maxCoreScatter * (cutHeight - refHeight)
        if minAbsGap is None:
            minAbsGap = minGap * (cutHeight - refHeight)
        if minSplitHeight is None:
            minSplitHeight = 0
        if minAbsSplitHeight is None:
            minAbsSplitHeight = refHeight + minSplitHeight * (cutHeight - refHeight)
        nPoints = nMerge + 1
        IndMergeToBranch = np.repeat(0, nMerge, axis=0)
        onBranch = np.repeat(0, nPoints, axis=0)
        RootBranch = 0

        mergeDiagnostics = pd.DataFrame({'smI': np.repeat(np.nan, nMerge, axis=0),
                                         'smSize': np.repeat(np.nan, nMerge, axis=0),
                                         'smCrSc': np.repeat(np.nan, nMerge, axis=0),
                                         'smGap': np.repeat(np.nan, nMerge, axis=0),
                                         'lgI': np.repeat(np.nan, nMerge, axis=0),
                                         'lgSize': np.repeat(np.nan, nMerge, axis=0),
                                         'lgCrSc': np.repeat(np.nan, nMerge, axis=0),
                                         'lgGap': np.repeat(np.nan, nMerge, axis=0),
                                         'merged': np.repeat(np.nan, nMerge, axis=0)})
        if externalBranchSplitFnc is not None:
            externalMergeDiags = pd.DataFrame(np.nan, index=list(range(nMerge)), columns=list(range(nExternalSplits)))

        extender = np.repeat(0, chunkSize, axis=0)

        for merge in range(nMerge):
            if dendro_height[merge] <= cutHeight:
                if dendro_merge[merge, 0] < 0 and dendro_merge[merge, 1] < 0:
                    nBranches = nBranches + 1
                    branch_isBasic[nBranches - 1] = True
                    branch_isTopBasic[nBranches - 1] = True
                    branch_singletons[nBranches - 1] = np.append(-1 * dendro_merge[merge, :], extender)
                    branch_basicClusters[nBranches - 1] = extender
                    branch_mergingHeights[nBranches - 1] = np.append(np.repeat(dendro_height[merge], 2), extender)
                    branch_singletonHeights[nBranches - 1] = np.append(np.repeat(dendro_height[merge], 2), extender)
                    IndMergeToBranch[merge] = nBranches
                    RootBranch = nBranches
                elif np.sign(dendro_merge[merge, 0]) * np.sign(dendro_merge[merge, 1]) < 0:
                    clust = IndMergeToBranch[int(np.max(dendro_merge[merge, :])) - 1]

                    if clust == 0:
                        sys.exit("Internal error: a previous merge has no associated cluster. Sorry!")

                    gene = -1 * int(np.min(dendro_merge[merge, :]))
                    ns = branch_nSingletons[clust - 1] + 1
                    nm = branch_nMerge[clust - 1] + 1

                    if branch_isBasic[clust - 1]:
                        if ns > len(branch_singletons[clust - 1]):
                            branch_singletons[clust - 1] = np.append(branch_singletons[clust - 1], extender)
                            branch_singletonHeights[clust - 1] = np.append(branch_singletonHeights[clust - 1], extender)
                        branch_singletons[clust - 1][ns - 1] = gene
                        branch_singletonHeights[clust - 1][ns - 1] = dendro_height[merge]
                    else:
                        onBranch[int(gene) - 1] = clust

                    if nm >= len(branch_mergingHeights[clust - 1]):
                        branch_mergingHeights[clust - 1] = np.append(branch_mergingHeights[clust - 1], extender)
                    branch_mergingHeights[clust - 1][nm - 1] = dendro_height[merge]
                    branch_size[clust - 1] = branch_size[clust - 1] + 1
                    branch_nMerge[clust - 1] = nm
                    branch_nSingletons[clust - 1] = ns
                    IndMergeToBranch[merge] = clust
                    RootBranch = clust
                else:
                    clusts = IndMergeToBranch[dendro_merge[merge, :].astype(int) - 1]
                    sizes = branch_size[clusts - 1]
                    rnk = rankdata(sizes, method="ordinal")
                    small = clusts[rnk[0] - 1]
                    large = clusts[rnk[1] - 1]
                    sizes = sizes[rnk - 1]

                    if branch_isBasic[small - 1]:
                        coresize = pyWGCNA.coreSizeFunc(branch_nSingletons[small - 1], minClusterSize)
                        Core = np.array(branch_singletons[small - 1][np.arange(coresize)] - 1, dtype=int)
                        SmAveDist = np.mean(distM.iloc[Core, Core].sum() / coresize - 1)
                    else:
                        SmAveDist = 0

                    if branch_isBasic[large - 1]:
                        coresize = pyWGCNA.coreSizeFunc(branch_nSingletons[large - 1], minClusterSize)
                        Core = np.array(branch_singletons[large - 1][np.arange(int(coresize))] - 1, dtype=int)
                        LgAveDist = np.mean(distM.iloc[Core, Core].sum() / coresize - 1)
                    else:
                        LgAveDist = 0

                    mergeDiagnostics.loc[merge, :] = [small,
                                                      branch_size[small - 1],
                                                      SmAveDist,
                                                      dendro_height[merge] - SmAveDist,
                                                      large,
                                                      branch_size[large - 1],
                                                      LgAveDist,
                                                      dendro_height[merge] - LgAveDist,
                                                      None]
                    SmallerScores = [branch_isBasic[small - 1],
                                     branch_size[small - 1] < minClusterSize,
                                     SmAveDist > maxAbsCoreScatter,
                                     dendro_height[merge] - SmAveDist < minAbsGap,
                                     dendro_height[merge] < minAbsSplitHeight]
                    if SmallerScores[0] * np.count_nonzero(SmallerScores[1:]) > 0:
                        DoMerge = True
                        SmallerFailSize = ~np.logical_or(SmallerScores[2], SmallerScores[3])
                    else:
                        LargerScores = [branch_isBasic[large - 1],
                                        branch_size[large - 1] < minClusterSize,
                                        LgAveDist > maxAbsCoreScatter,
                                        dendro_height[merge] - LgAveDist < minAbsGap,
                                        dendro_height[merge] < minAbsSplitHeight]
                        if LargerScores[0] * np.count_nonzero(LargerScores[1:]) > 0:
                            DoMerge = True
                            SmallerFailSize = ~np.logical_or(LargerScores[2], LargerScores[3])
                            x = small
                            small = large
                            large = x
                            sizes = np.flip(sizes)
                        else:
                            DoMerge = False

                    if DoMerge:
                        mergeDiagnostics['merged'][merge] = 1

                    if not DoMerge and nExternalSplits > 0 and branch_isBasic[small - 1] and branch_isBasic[large - 1]:
                        branch1 = branch_singletons[large - 1][np.arange(sizes[1])]
                        branch2 = branch_singletons[small - 1][np.arange(sizes[0])]
                        es = 0
                        while es < nExternalSplits and not DoMerge:
                            es = es + 1
                            args = pd.DataFrame({'externalSplitOptions': externalSplitOptions[es - 1],
                                                 'branch1': branch1, 'branch2': branch2})
                            # TODO: extSplit = do.call(externalBranchSplitFnc[[es]], args)
                            extSplit = None
                            DoMerge = extSplit < minExternalSplit[es - 1]
                            externalMergeDiags[merge, es - 1] = extSplit
                            mergeDiagnostics['merged'][merge] = 0
                            if DoMerge:
                                mergeDiagnostics['merged'][merge] = 2

                    if DoMerge:
                        branch_failSize[small - 1] = SmallerFailSize
                        branch_mergedInto[small - 1] = large
                        branch_attachHeight[small - 1] = dendro_height[merge]
                        branch_isTopBasic[small - 1] = False
                        nss = branch_nSingletons[small - 1]
                        nsl = branch_nSingletons[large - 1]
                        ns = nss + nsl

                        if branch_isBasic[large - 1]:
                            nExt = np.ceil((ns - len(branch_singletons[large - 1])) / chunkSize)
                            if nExt > 0:
                                branch_singletons[large - 1] = np.append(branch_singletons[large - 1],
                                                                         np.repeat(extender, nExt))
                                branch_singletonHeights[large - 1] = np.append(branch_singletonHeights[large - 1],
                                                                               np.repeat(extender, nExt))

                            branch_singletons[large - 1][np.arange(nsl, ns)] = branch_singletons[small - 1][
                                np.arange(nss)]
                            branch_singletonHeights[large - 1][np.arange(nsl, ns)] = branch_singletonHeights[small - 1][
                                np.arange(nss)]
                            branch_nSingletons[large] = ns
                        else:
                            if not branch_isBasic[small - 1]:
                                sys.exit("Internal error: merging two composite clusters. Sorry!")

                            onBranch[branch_singletons[small - 1][branch_singletons[small - 1] != 0] - 1] = large

                        nm = branch_nMerge[large - 1] + 1

                        if nm > len(branch_mergingHeights[large - 1]):
                            branch_mergingHeights[large - 1] = np.append(branch_mergingHeights[large - 1], extender)

                        branch_mergingHeights[large - 1][nm - 1] = dendro_height[merge]
                        branch_nMerge[large - 1] = nm
                        branch_size[large - 1] = branch_size[small - 1] + branch_size[large - 1]
                        IndMergeToBranch[merge] = large
                        RootBranch = large
                    else:
                        if branch_isBasic[large - 1] and not branch_isBasic[small - 1]:
                            x = large
                            large = small
                            small = x
                            sizes = np.flip(sizes)

                        if branch_isBasic[large - 1] or (pamStage and pamRespectsDendro):
                            nBranches = nBranches + 1
                            branch_attachHeight[[large - 1, small - 1]] = dendro_height[merge]
                            branch_mergedInto[[large - 1, small - 1]] = nBranches
                            if branch_isBasic[small - 1]:
                                addBasicClusters = small
                            else:
                                addBasicClusters = branch_basicClusters[small - 1]
                            if branch_isBasic[large - 1]:
                                addBasicClusters = np.append(addBasicClusters, large)
                            else:
                                addBasicClusters = np.append(addBasicClusters, branch_basicClusters[large - 1])
                            branch_isBasic[nBranches - 1] = False
                            branch_isTopBasic[nBranches - 1] = False
                            branch_basicClusters[nBranches - 1] = addBasicClusters
                            branch_mergingHeights[nBranches - 1] = np.append(np.repeat(dendro_height[merge], 2),
                                                                             extender)
                            branch_nMerge[nBranches - 1] = 2
                            branch_size[nBranches - 1] = np.sum(sizes)
                            branch_nBasicClusters[nBranches - 1] = len(addBasicClusters)
                            IndMergeToBranch[merge] = nBranches
                            RootBranch = nBranches
                        else:
                            addBasicClusters = [small] if branch_isBasic[small - 1] else branch_basicClusters[small - 1]

                            nbl = branch_nBasicClusters[large - 1]
                            nb = branch_nBasicClusters[large - 1] + len(addBasicClusters)

                            if nb > len(branch_basicClusters[large - 1]):
                                nExt = np.ceil((nb - len(branch_basicClusters[large - 1])) / chunkSize)
                                branch_basicClusters[large - 1] = np.append(branch_basicClusters[large - 1],
                                                                            np.repeat(extender, nExt))

                            branch_basicClusters[large - 1][np.arange(nbl, nb)] = addBasicClusters
                            branch_nBasicClusters[large - 1] = nb
                            branch_size[large - 1] = branch_size[large - 1] + branch_size[small - 1]
                            nm = branch_nMerge[large - 1] + 1

                            if nm > len(branch_mergingHeights[large - 1]):
                                branch_mergingHeights[large - 1] = np.append(branch_mergingHeights[large - 1], extender)

                            branch_mergingHeights[large - 1][nm - 1] = dendro_height[merge]
                            branch_nMerge[large - 1] = nm
                            branch_attachHeight[small - 1] = dendro_height[merge]
                            branch_mergedInto[small - 1] = large
                            IndMergeToBranch[merge] = large
                            RootBranch = large

        isCluster = np.repeat(False, nBranches)
        SmallLabels = np.repeat(0, nPoints)

        for clust in range(nBranches):
            if np.isnan(branch_attachHeight[clust]):
                branch_attachHeight[clust] = cutHeight
            if branch_isTopBasic[clust]:
                coresize = pyWGCNA.coreSizeFunc(branch_nSingletons[clust], minClusterSize)
                Core = np.array(branch_singletons[clust][np.arange(coresize)] - 1, dtype=int)
                CoreScatter = np.mean(distM.iloc[Core, Core].sum() / (coresize - 1))
                isCluster[clust] = (branch_isTopBasic[clust] and branch_size[clust] >= minClusterSize and
                                    CoreScatter < maxAbsCoreScatter and branch_attachHeight[
                                        clust] - CoreScatter > minAbsGap)
            else:
                CoreScatter = 0
            if branch_failSize[clust]:
                SmallLabels[branch_singletons[clust].astype(int) - 1] = clust + 1

        if not respectSmallClusters:
            SmallLabels = np.repeat(0, nPoints)

        Colors = np.zeros((nPoints,))
        coreLabels = np.zeros((nPoints,))
        clusterBranches = np.where(isCluster)[0].tolist()
        branchLabels = np.zeros((nBranches,))
        color = 0

        for clust in clusterBranches:
            color = color + 1
            Colors[branch_singletons[clust][branch_singletons[clust] != 0] - 1] = color
            SmallLabels[branch_singletons[clust][branch_singletons[clust] != 0] - 1] = 0
            coresize = pyWGCNA.coreSizeFunc(branch_nSingletons[clust], minClusterSize)
            Core = np.array(branch_singletons[clust][np.arange(coresize)] - 1, dtype=int)
            coreLabels[Core] = color
            branchLabels[clust] = color

        Labeled = np.where(Colors != 0)[0].tolist()
        Unlabeled = np.where(Colors == 0)[0].tolist()
        nUnlabeled = len(Unlabeled)
        UnlabeledExist = nUnlabeled > 0

        if len(Labeled) > 0:
            LabelFac = pd.Categorical(Colors[Labeled])
            nProperLabels = len(LabelFac.categories)
        else:
            nProperLabels = 0

        if pamStage and UnlabeledExist and nProperLabels > 0:
            nPAMed = 0
            if useMedoids:
                Medoids = np.repeat(0, nProperLabels)
                ClusterRadii = np.repeat(0, nProperLabels)
                for cluster in range(1, nProperLabels + 1):
                    InCluster = np.arange(1, nPoints + 1)[Colors == cluster]
                    DistInCluster = distM.iloc[InCluster - 1, InCluster - 1]
                    DistSums = DistInCluster.sum(axis=0)
                    Medoids[cluster - 1] = InCluster[DistSums.idxmin()]
                    ClusterRadii[cluster - 1] = np.max(DistInCluster[:, DistSums.idxmin()])

                if respectSmallClusters:
                    FSmallLabels = pd.Categorical(SmallLabels)
                    SmallLabLevs = pd.to_numeric(FSmallLabels.categories)
                    nSmallClusters = len(FSmallLabels.categories) - (SmallLabLevs[1] == 0)

                    if nSmallClusters > 0:
                        for sclust in SmallLabLevs[SmallLabLevs != 0]:
                            InCluster = np.where(SmallLabels == sclust)[0].tolist()
                            if pamRespectsDendro:
                                onBr = np.unique(onBranch[InCluster])
                                if len(onBr) > 1:
                                    msg = "Internal error: objects in a small cluster are marked to belong\n " \
                                          "to several large branches:" + str(onBr)
                                    sys.exit(msg)

                                if onBr > 0:
                                    basicOnBranch = branch_basicClusters[onBr[0] - 1]
                                    labelsOnBranch = branchLabels[basicOnBranch - 1]
                                else:
                                    labelsOnBranch = None
                            else:
                                labelsOnBranch = list(range(1, nProperLabels + 1))

                            DistInCluster = distM.iloc[InCluster, InCluster]

                            if len(labelsOnBranch) > 0:
                                if len(InCluster) > 1:
                                    DistSums = DistInCluster.sum(axis=1)
                                    smed = InCluster[DistSums.idxmin()]
                                    DistToMeds = distM.iloc[Medoids[labelsOnBranch - 1] - 1, smed]
                                    closest = DistToMeds.idxmin()
                                    DistToClosest = DistToMeds[closest]
                                    closestLabel = labelsOnBranch[closest]
                                    if DistToClosest < ClusterRadii[closestLabel - 1] or DistToClosest < maxPamDist:
                                        Colors[InCluster] = closestLabel
                                        nPAMed = nPAMed + len(InCluster)
                                else:
                                    Colors[InCluster] = -1
                            else:
                                Colors[InCluster] = -1

                Unlabeled = np.where(Colors == 0)[0].tolist()
                if len(Unlabeled > 0):
                    for obj in Unlabeled:
                        if pamRespectsDendro:
                            onBr = onBranch[obj]
                            if onBr > 0:
                                basicOnBranch = branch_basicClusters[onBr - 1]
                                labelsOnBranch = branchLabels[basicOnBranch - 1]
                            else:
                                labelsOnBranch = None
                        else:
                            labelsOnBranch = np.arange(nProperLabels)

                        if labelsOnBranch is not None:
                            UnassdToMedoidDist = distM.iloc[Medoids[labelsOnBranch - 1] - 1, obj]
                            nearest = UnassdToMedoidDist.idxmin()
                            NearestCenterDist = UnassdToMedoidDist[nearest]
                            nearestMed = labelsOnBranch[nearest]
                            if NearestCenterDist < ClusterRadii[nearestMed - 1] or NearestCenterDist < maxPamDist:
                                Colors[obj] = nearestMed
                                nPAMed = nPAMed + 1
                    UnlabeledExist = (sum(Colors == 0) > 0)
            else:
                ClusterDiam = np.zeros((nProperLabels,))
                for cluster in range(nProperLabels):
                    InCluster = np.where(Colors == cluster)[0].tolist()
                    nInCluster = len(InCluster)
                    DistInCluster = distM.iloc[InCluster, InCluster]
                    if nInCluster > 1:
                        AveDistInClust = DistInCluster.sum(axis=1) / (nInCluster - 1)
                        AveDistInClust.reset_index(drop=True, inplace=True)
                        ClusterDiam[cluster] = AveDistInClust.max()
                    else:
                        ClusterDiam[cluster] = 0

                ColorsX = Colors.copy()
                if respectSmallClusters:
                    FSmallLabels = pd.Categorical(SmallLabels)
                    SmallLabLevs = pd.to_numeric(FSmallLabels.categories)
                    nSmallClusters = len(FSmallLabels.categories) - (SmallLabLevs[0] == 0)
                    if nSmallClusters > 0:
                        if pamRespectsDendro:
                            for sclust in SmallLabLevs[SmallLabLevs != 0]:
                                InCluster = list(range(nPoints))[SmallLabels == sclust]
                                onBr = pd.unique(onBranch[InCluster])
                                if len(onBr) > 1:
                                    msg = "Internal error: objects in a small cluster are marked to belong\n" \
                                          "to several large branches:" + str(onBr)
                                    sys.exit(msg)
                                if onBr > 0:
                                    basicOnBranch = branch_basicClusters[onBr[0] - 1]
                                    labelsOnBranch = branchLabels[basicOnBranch - 1]
                                    useObjects = ColorsX in np.unique(labelsOnBranch)
                                    DistSClustClust = distM.iloc[InCluster, useObjects]
                                    MeanDist = DistSClustClust.mean(axis=0)
                                    useColorsFac = ColorsX[useObjects]  # pd.Categorical(ColorsX[useObjects])
                                    MeanDist = pd.DataFrame({'MeanDist': MeanDist, 'useColorsFac': useColorsFac})
                                    MeanMeanDist = MeanDist.groupby(
                                        'useColorsFac').mean()  # tapply(MeanDist, useColorsFac, mean)
                                    nearest = MeanMeanDist[['MeanDist']].idxmin().astype(int) - 1
                                    NearestDist = MeanMeanDist[['MeanDist']].min()
                                    if np.logical_or(np.all(NearestDist < ClusterDiam[nearest]),
                                                     NearestDist < maxPamDist).tolist()[0]:
                                        Colors[InCluster] = nearest
                                        nPAMed = nPAMed + len(InCluster)
                                    else:
                                        Colors[InCluster] = -1
                        else:
                            labelsOnBranch = list(range(nProperLabels))
                            useObjects = np.where(ColorsX != 0)[0].tolist()
                            for sclust in SmallLabLevs[SmallLabLevs != 0]:
                                InCluster = np.where(SmallLabels == sclust)[0].tolist()
                                DistSClustClust = distM.iloc[InCluster, useObjects]
                                MeanDist = DistSClustClust.mean(axis=0)
                                useColorsFac = ColorsX[useObjects]  # pd.Categorical(ColorsX[useObjects])
                                MeanDist = pd.DataFrame({'MeanDist': MeanDist, 'useColorsFac': useColorsFac})
                                MeanMeanDist = MeanDist.groupby(
                                    'useColorsFac').mean()  # tapply(MeanDist, useColorsFac, mean)
                                nearest = MeanMeanDist[['MeanDist']].idxmin().astype(int) - 1
                                NearestDist = MeanMeanDist[['MeanDist']].min()
                                if np.logical_or(np.all(NearestDist < ClusterDiam[nearest]),
                                                 NearestDist < maxPamDist).tolist()[0]:
                                    Colors[InCluster] = nearest
                                    nPAMed = nPAMed + len(InCluster)
                                else:
                                    Colors[InCluster] = -1

                Unlabeled = np.where(Colors == 0)[0].tolist()
                if len(Unlabeled) > 0:
                    if pamRespectsDendro:
                        unlabOnBranch = Unlabeled[onBranch[Unlabeled] > 0]
                        for obj in unlabOnBranch:
                            onBr = onBranch[obj]
                            basicOnBranch = branch_basicClusters[onBr - 1]
                            labelsOnBranch = branchLabels[basicOnBranch]
                            useObjects = ColorsX in np.unique(labelsOnBranch)
                            useColorsFac = ColorsX[useObjects]  # pd.Categorical(ColorsX[useObjects])
                            UnassdToClustDist = distM.iloc[useObjects, obj].groupby(
                                'useColorsFac').mean()  # tapply(distM[useObjects, obj], useColorsFac, mean)
                            nearest = UnassdToClustDist.idxmin().astype(int) - 1
                            NearestClusterDist = UnassdToClustDist[nearest]
                            nearestLabel = pd.to_numeric(useColorsFac.categories[nearest])
                            if np.logical_or(np.all(NearestClusterDist < ClusterDiam[nearest]),
                                             NearestClusterDist < maxPamDist).tolist()[0]:
                                Colors[obj] = nearest
                                nPAMed = nPAMed + 1
                    else:
                        useObjects = np.where(ColorsX != 0)[0].tolist()
                        useColorsFac = ColorsX[useObjects]  # pd.Categorical(ColorsX[useObjects])
                        tmp = pd.DataFrame(distM.iloc[useObjects, Unlabeled])
                        tmp['group'] = useColorsFac
                        UnassdToClustDist = tmp.groupby(
                            ['group']).mean()  # apply(distM[useObjects, Unlabeled], 2, tapply, useColorsFac, mean)
                        nearest = np.subtract(UnassdToClustDist.idxmin(axis=0),
                                              np.ones(UnassdToClustDist.shape[1])).astype(
                            int)  # apply(UnassdToClustDist, 2, which.min)
                        nearestDist = UnassdToClustDist.min(axis=0)  # apply(UnassdToClustDist, 2, min)
                        nearestLabel = nearest + 1
                        sumAssign = np.sum(np.logical_or(nearestDist < ClusterDiam[nearest], nearestDist < maxPamDist))
                        assign = np.where(np.logical_or(nearestDist < ClusterDiam[nearest], nearestDist < maxPamDist))[
                            0].tolist()
                        tmp = [Unlabeled[x] for x in assign]
                        Colors[tmp] = [nearestLabel.iloc[x] for x in assign]
                        nPAMed = nPAMed + sumAssign

        Colors[np.where(Colors < 0)[0].tolist()] = 0
        UnlabeledExist = (np.count_nonzero(Colors == 0) > 0)
        NumLabs = list(map(int, Colors.copy()))
        Sizes = pd.DataFrame(NumLabs).value_counts().sort_index()
        OrdNumLabs = pd.DataFrame({"Name": NumLabs, "Value": np.repeat(1, len(NumLabs))})

        if UnlabeledExist:
            if len(Sizes) > 1:
                SizeRank = np.insert(stats.rankdata(-1 * Sizes[1:len(Sizes)], method='ordinal') + 1, 0, 1)
            else:
                SizeRank = [1]
            for i in range(len(NumLabs)):
                OrdNumLabs.Value[i] = SizeRank[NumLabs[i] - 1]
        else:
            SizeRank = stats.rankdata(-1 * Sizes, method='ordinal')
            for i in range(len(NumLabs)):
                OrdNumLabs.Value[i] = SizeRank[NumLabs[i] - 1]

        print("\tDone..\n")

        OrdNumLabs.Value = OrdNumLabs.Value - UnlabeledExist
        return OrdNumLabs

    @staticmethod
    def labels2colors(labels, zeroIsGrey=True, colorSeq=None, naColor="grey"):
        """
        Converts a vector or array of numerical labels into a corresponding vector or array of colors corresponding to the labels.

        :param labels: list or matrix of non-negative integer or other (such as character) labels.
        :type labels: list or matrix
        :param zeroIsGrey: If TRUE, labels 0 will be assigned color grey. Otherwise, labels below 1 will trigger an error. (default = True)
        :type zeroIsGrey: bool
        :param colorSeq: Color sequence corresponding to labels. If not given, a standard sequence will be used.
        :type colorSeq: list or matrix
        :param naColor: Color that will encode missing values.
        :type naColor: str

        :return: An array of character strings of the same length or dimensions as labels.
        :rtype: ndarray
        """
        if colorSeq is None:
            colors = mcolors.CSS4_COLORS
            #naColor = "white"  # Example naColor, you can set it to the actual naColor you have
            colorSeq = create_custom_color_seq(colors, naColor)
            colorSeq = interleave_colors(colorSeq)

        if all(isinstance(x, int) for x in labels.Value):
            if zeroIsGrey:
                minLabel = 0
            else:
                minLabel = 1
            if np.any(labels.Value < 0):
                minLabel = np.min(labels.Value)
            nLabels = labels.copy()
        else:
            factors = pd.Categorical(labels.Value)
            nLabels = factors.codes

        if np.max(nLabels.Value) > len(colorSeq):
            nRepeats = int((np.max(labels.Value) - 1) / len(colorSeq)) + 1
            print(f"{WARNING}labels2colors: Number of labels exceeds number of avilable colors.\n"
                  f"Some colors will be repeated {str(nRepeats)} times.{ENDC}")
            extColorSeq = colorSeq
            for rep in range(nRepeats):
                tmp = [str(item) + "." + str(rep) for item in colorSeq]
                extColorSeq = np.concatenate((extColorSeq, tmp), axis=None)
        else:
            nRepeats = 1
            extColorSeq = colorSeq

        colors = np.empty(nLabels.shape[0], dtype=object)
        fin = [v is not None for v in nLabels.Value]
        fin = np.atleast_1d(fin)
        colors[~fin] = naColor
        finLabels = nLabels.loc[fin, :]
        colors[fin] = [extColorSeq[x] for x in finLabels.Value]

        return colors

    @staticmethod
    def moduleEigengenes(expr, colors, impute=True, nPC=1, align="along average", excludeGrey=False, grey="grey",
                         subHubs=True, softPower=6, scaleVar=True, trapErrors=False):
        """
        Calculates module eigengenes (1st principal component) of modules in a given single dataset.

        :param expr: Expression data for a single set in the form of a data frame where rows are samples and columns are genes (probes).
        :type expr: pandas dataframe
        :param colors: A list of the same length as the number of probes in expr, giving module color for all probes (genes). Color "grey" is reserved for unassigned genes.
        :type colors: list
        :param impute: If TRUE, expression data will be checked for the presence of NA entries and if the latter are present, numerical data will be imputed. (defualt = True)
        :type impute: bool
        :param nPC: Number of principal components and variance explained entries to be calculated. Note that only the first principal component is returned; the rest are used only for the calculation of proportion of variance explained. If given nPC is greater than 10, a warning is issued. (default = 1)
        :type nPC: int
        :param align: Controls whether eigengenes, whose orientation is undetermined, should be aligned with average expression (align = "along average") or left as they are (align = ""). Any other value will trigger an error. (default = "along average")
        :type align: str
        :param excludeGrey: Should the improper module consisting of 'grey' genes be excluded from the eigengenes (default = False)
        :type excludeGrey: bool
        :param grey: Value of colors designating the improper module. Note that if colors is a factor of numbers, the default value will be incorrect. (default = grey)
        :type grey: str
        :param subHubs: Controls whether hub genes should be substituted for missing eigengenes. If TRUE, each missing eigengene (i.e., eigengene whose calculation failed and the error was trapped) will be replaced by a weighted average of the most connected hub genes in the corresponding module. If this calculation fails, or if subHubs==FALSE, the value of trapErrors will determine whether the offending module will be removed or whether the function will issue an error and stop. (default = True)
        :type subHubs: bool
        :param softPower: The power used in soft-thresholding the adjacency matrix. Only used when the hubgene approximation is necessary because the principal component calculation failed. It must be non-negative. The default value should only be changed if there is a clear indication that it leads to incorrect results. (default = 6)
        :type softPower: int
        :param trapErrors: Controls handling of errors from that may arise when there are too many NA entries in expression data. If TRUE, errors from calling these functions will be trapped without abnormal exit. If FALSE, errors will cause the function to stop. Note, however, that subHubs takes precedence in the sense that if subHubs==TRUE and trapErrors==FALSE, an error will be issued only if both the principal component and the hubgene calculations have failed. (default = False)
        :type trapErrors: bool
        :param scaleVar: can be used to turn off scaling of the expression data before calculating the singular value decomposition. The scaling should only be turned off if the data has been scaled previously, in which case the function can run a bit faster. Note however that the function first imputes, then scales the expression data in each module. If the expression contain missing data, scaling outside of the function and letting the function impute missing data may lead to slightly different results than if the data is scaled within the function. (default = True)
        :type scaleVar: bool

        :return: A dictionary containing: "eigengenes": Module eigengenes in a dataframe, with each column corresponding to one eigengene. The columns are named by the corresponding color with an "ME" prepended, e.g., MEturquoise etc. If returnValidOnly==FALSE, module eigengenes whose calculation failed have all components set to NA. "averageExpr": If align == "along average", a dataframe containing average normalized expression in each module. The columns are named by the corresponding color with an "AE" prepended, e.g., AEturquoise etc. "varExplained": A dataframe in which each column corresponds to a module, with the component varExplained[PC, module] giving the variance of module module explained by the principal component no. PC. The calculation is exact irrespective of the number of computed principal components. At most 10 variance explained values are recorded in this dataframe. "nPC": A copy of the input nPC. "validMEs": A boolean vector. Each component (corresponding to the columns in data) is TRUE if the corresponding eigengene is valid, and FALSE if it is invalid. Valid eigengenes include both principal components and their hubgene approximations. When returnValidOnly==FALSE, by definition all returned eigengenes are valid and the entries of validMEs are all TRUE. "validColors": A copy of the input colors with entries corresponding to invalid modules set to grey if given, otherwise 0 if colors is numeric and "grey" otherwise. "allOK": Boolean flag signalling whether all eigengenes have been calculated correctly, either as principal components or as the hubgene average approximation. "allPC": Boolean flag signalling whether all returned eigengenes are principal components. "isPC": Boolean vector. Each component (corresponding to the columns in eigengenes) is TRUE if the corresponding eigengene is the first principal component and FALSE if it is the hubgene approximation or is invalid. "isHub": Boolean vector. Each component (corresponding to the columns in eigengenes) is TRUE if the corresponding eigengene is the hubgene approximation and FALSE if it is the first principal component or is invalid. "validAEs": Boolean vector. Each component (corresponding to the columns in eigengenes) is TRUE if the corresponding module average expression is valid. "allAEOK": Boolean flag signalling whether all returned module average expressions contain valid data. Note that returnValidOnly==TRUE does not imply allAEOK==TRUE: some invalid average expressions may be returned if their corresponding eigengenes have been calculated correctly.
        :rtype: dict
        """
        print(f"{OKCYAN}Calculating {len(pd.Categorical(colors).categories)} module eigengenes in given set...{ENDC}")
        check = True
        pc = None
        returnValidOnly = trapErrors
        if all(isinstance(x, int) for x in colors):
            grey = 0
        if expr is None:
            sys.exit("moduleEigengenes: Error: expr is NULL.")
        if colors is None:
            sys.exit("moduleEigengenes: Error: colors is NULL.")
        if isinstance(expr, dict):
            expr = expr['data']
        if expr.shape is None or len(expr.shape) != 2:
            sys.exit("moduleEigengenes: Error: expr must be two-dimensional.")
        if expr.shape[1] != len(colors):
            sys.exit("moduleEigengenes: Error: ncol(expr) and length(colors) must be equal (one color per gene).")
        if len(pd.Categorical(colors).categories) == 1:
            sys.exit(
                f"{WARNING}All your genes ended up in one modules! Reconsider your input parameters in order to get more modules.{ENDC}")
        # TODO: "Argument 'colors' contains unused levels (empty modules). Use colors[, drop=TRUE] to get rid of them."
        if softPower < 0:
            sys.exit("softPower must be non-negative")
        maxVarExplained = 10
        if nPC > maxVarExplained:
            print(f"{WARNING}Given nPC is too large. Will use value {str(maxVarExplained)}{ENDC}")
        nVarExplained = min(nPC, maxVarExplained)
        modlevels = pd.Categorical(colors).categories
        if excludeGrey:
            if len(np.where(modlevels != grey)) > 0:
                modlevels = modlevels[np.where(modlevels != grey)]
            else:
                sys.exit("Color levels are empty. Possible reason: the only color is grey and grey module is excluded "
                         "from the calculation.")
        PrinComps = np.empty((expr.shape[0], len(modlevels)))
        PrinComps[:] = np.nan
        PrinComps = pd.DataFrame(PrinComps)
        averExpr = np.empty((expr.shape[0], len(modlevels)))
        averExpr[:] = np.nan
        averExpr = pd.DataFrame(averExpr)
        varExpl = np.empty((nVarExplained, len(modlevels)))
        varExpl[:] = np.nan
        varExpl = pd.DataFrame(varExpl)
        validMEs = np.repeat(True, len(modlevels))
        validAEs = np.repeat(False, len(modlevels))
        isPC = np.repeat(True, len(modlevels))
        isHub = np.repeat(False, len(modlevels))
        validColors = colors
        PrinComps.columns = ["ME" + str(modlevel) for modlevel in modlevels]
        averExpr.columns = ["AE" + str(modlevel) for modlevel in modlevels]
        if expr.index is not None:
            PrinComps.index = expr.index
            averExpr.index = expr.index
        for i in range(len(modlevels)):
            modulename = modlevels[i]
            restrict1 = (colors == modulename)
            datModule = expr.loc[:, restrict1].T
            n = datModule.shape[0]
            p = datModule.shape[1]
            try:
                if datModule.shape[0] > 1 and impute:
                    seedSaved = True
                    if datModule.isnull().values.any():
                        # define imputer
                        imputer = KNNImputer(n_neighbors=np.min(10, datModule.shape[0] - 1))
                        # fit on the dataset
                        imputer.fit(datModule)
                        # transform the dataset
                        datModule = imputer.transform(
                            datModule)  # datModule = impute.knn(datModule, k = min(10, nrow(datModule) - 1))
                if scaleVar:
                    datModule = pd.DataFrame(scale(datModule.T).T, index=datModule.index, columns=datModule.columns)
                u, d, v = np.linalg.svd(datModule)
                u = u[:, 0:min(n, p, nPC)]
                v = v[0:min(n, p, nPC), :]
                tmp = datModule.copy()
                tmp = pd.concat([tmp, pd.DataFrame(v[0:min(n, p, nVarExplained), :],
                                                   columns=tmp.columns.tolist())],
                                ignore_index=True)
                veMat = pd.DataFrame(np.corrcoef(tmp.values)).iloc[:-1, -1].T
                varExpl.iloc[0:min(n, p, nVarExplained), i] = (veMat ** 2).mean(axis=0)
                pc = v[0].tolist()
            except:
                if not subHubs:
                    sys.exit("Error!")
                if subHubs:
                    print(" ..principal component calculation for module", modulename,
                          "failed with the following error:", flush=True)
                    print("     ..hub genes will be used instead of principal components.", flush=True)

                    isPC[i] = False
                    check = True
                    try:
                        scaledExpr = pd.DataFrame(scale(datModule.T).T, index=datModule.index,
                                                  columns=datModule.columns)
                        covEx = np.cov(scaledExpr)
                        covEx[~ np.isfinite(covEx)] = 0
                        modAdj = np.abs(covEx) ** softPower
                        kIM = (modAdj.mean(axis=0)) ** 3
                        if np.max(kIM) > 1:
                            kIM = kIM - 1
                        kIM[np.where(kIM is None)] = 0
                        hub = np.argmax(kIM)
                        alignSign = np.sign(covEx[:, hub])
                        alignSign[np.where(alignSign is None)] = 0
                        isHub[i] = True
                        tmp = np.array(kIM * alignSign)
                        tmp.shape = scaledExpr.shape
                        pcxMat = scaledExpr * tmp / sum(kIM)
                        pcx = pcxMat.mean(axis=0)
                        varExpl[0, i] = np.mean(np.corrcoef(pcx, datModule.transpose()) ** 2)
                        pc = pcx
                    except:
                        check = False
            if not check:
                if not trapErrors:
                    sys.exit("Error!")
                print(" ..ME calculation of module", modulename, "failed with the following error:", flush=True)
                print("     ", pc, " ..the offending module has been removed.", flush=True)
                print(
                    f"{WARNING}Eigengene calculation of module {modulename} failed with the following error \n"
                    f"{pc} The offending module has been removed.{ENDC}")
                validMEs[i] = False
                isPC[i] = False
                isHub[i] = False
                validColors[restrict1] = grey
            else:
                PrinComps.iloc[:, i] = pc
                try:
                    if isPC[i]:
                        scaledExpr = scale(datModule.T)
                    averExpr.iloc[:, i] = scaledExpr.mean(axis=1)
                    if align == "along average":
                        corAve = np.corrcoef(averExpr.iloc[:, i], PrinComps.iloc[:, i])[0, 1]
                        if not np.isfinite(corAve):
                            corAve = 0
                        if corAve < 0:
                            PrinComps.iloc[:, i] = -1 * PrinComps.iloc[:, i]
                    validAEs[i] = True
                except:
                    if not trapErrors:
                        sys.exit("Error!")
                    print(" ..Average expression calculation of module", modulename,
                          "failed with the following error:", flush=True)
                    print(" ..the returned average expression vector will be invalid.", flush=True)

                    print(f"{WARNING}Average expression calculation of module {modulename} "
                          f"failed with the following error.\nThe returned average expression vector will "
                          f"be invalid.\n{ENDC}")

        allOK = (sum(np.logical_not(validMEs)) == 0)
        if returnValidOnly and sum(np.logical_not(validMEs)) > 0:
            PrinComps = PrinComps[:, validMEs]
            averExpr = averExpr[:, validMEs]
            varExpl = varExpl[:, validMEs]
            validMEs = np.repeat(True, PrinComps.shape[1])
            isPC = isPC[validMEs]
            isHub = isHub[validMEs]
            validAEs = validAEs[validMEs]

        allPC = (sum(np.logical_not(isPC)) == 0)
        allAEOK = (sum(np.logical_not(validAEs)) == 0)

        print("\tDone..\n")

        return {"eigengenes": PrinComps, "averageExpr": averExpr, "varExplained": varExpl, "nPC": nPC,
                "validMEs": validMEs, "validColors": validColors, "allOK": allOK, "allPC": allPC, "isPC": isPC,
                "isHub": isHub, "validAEs": validAEs, "allAEOK": allAEOK}

    @staticmethod
    def permissiveDim(x):
        d = x.shape
        if d is None:
            return [len(x), 1]
        return d

    @staticmethod
    def checkSets(data, checkStructure=False, useSets=None):
        """
        Checks whether given sets have the correct format and retrieves dimensions.

        :param data: A dict of lists; in each list there must be a component named data whose content is a matrix or dataframe or array of dimension 2.
        :type data: dict
        :param checkStructure: If FALSE, incorrect structure of data will trigger an error. If TRUE, an appropriate flag (see output) will be set to indicate whether data has correct structure. (default = False)
        :type checkStructure: bool
        :param useSets: Optional specification of entries of the list data that are to be checked. Defaults to all components. This may be useful when data only contains information for some of the sets.
        :type useSets: list

        :return: a dictionary contains: "nSets": Number of sets (length of the vector data). "nGenes": Number of columns in the data components in the lists. This number must be the same for all sets. "nSamples": A vector of length nSets giving the number of rows in the data components. "structureOK": Only set if the argument checkStructure equals TRUE. The value is TRUE if the paramter data passes a few tests of its structure, and FALSE otherwise. The tests are not exhaustive and are meant to catch obvious user errors rather than be bulletproof.
        :rtype: dict
        """
        if isinstance(data, pd.DataFrame):
            nSets = data.shape[1]
        else:
            nSets = len(data)
        if useSets is None:
            useSets = list(range(nSets))
        if nSets <= 0:
            sys.exit("No data given.")
        structureOK = True
        if not (isinstance(data, list) or isinstance(data, dict)):
            if checkStructure:
                structureOK = False
                nGenes = 0
                nSamples = 0
            else:
                sys.exit("data does not appear to have the correct format. "
                         "Consider using fixDataStructure or setting checkStructure = TRUE when calling this function.")
        elif isinstance(data, dict):
            nSamples = np.zeros(nSets)
            nGenes = pyWGCNA.permissiveDim(data[useSets[0]]['data'])[1]
            for set in useSets:
                if nGenes != pyWGCNA.permissiveDim(data[set]['data'])[1]:
                    if checkStructure:
                        structureOK = False
                    else:
                        sys.exit(("Incompatible number of genes in set 1 and", str(set)))
                nSamples[set] = pyWGCNA.permissiveDim(data[set]['data'])[0]
        else:
            nSamples = np.zeros(nSets)
            nGenes = pyWGCNA.permissiveDim(data[useSets[0]])[1]
            for set in useSets:
                if nGenes != pyWGCNA.permissiveDim(data[set])[1]:
                    if checkStructure:
                        structureOK = False
                    else:
                        sys.exit(("Incompatible number of genes in set 1 and", str(set)))
                nSamples[set] = pyWGCNA.permissiveDim(data[set])[0]
        return {"nSets": nSets, "nGenes": nGenes, "nSamples": nSamples, "structureOK": structureOK}

    @staticmethod
    def fixDataStructure(data):
        """
        Encapsulates single-set data in a wrapper that makes the data suitable for functions working on multiset data collections.

        :param data: A dataframe, matrix or array with two dimensions to be encapsulated.
        :type data: pandas dataframe ot dict

        :return: input data in a format suitable for functions operating on multiset data collections.
        :rtype: dict
        """
        if not isinstance(data, list):
            print("fixDataStructure: data is not a Dictionary: converting it into one.")
            x = data.copy()
            data = {0: {'data': x}}
        return data

    @staticmethod
    def multiSetMEs(exprData, colors, universalColors=None, useSets=None, useGenes=None, impute=True, nPC=1,
                    align="along average", excludeGrey=False, subHubs=True, trapErrors=False, softPower=6,
                    grey=None):
        """
        Calculates module eigengenes for several sets.

        :param exprData: Expression data in a multi-set format
        :type exprData: pandas dataframe
        :param colors: A list of the same length as the number of probes in expr, giving module color for all probes (genes). Color "grey" is reserved for unassigned genes.
        :type colors: list
        :param universalColors: Alternative specification of module assignment
        :type universalColors: list
        :param useSets: If calculations are requested in (a) selected set(s) only, the set(s) can be specified here. Defaults to all sets.
        :type useSets: list
        :param useGenes: Can be used to restrict calculation to a subset of genes
        :type useGenes: list
        :param impute: If TRUE, expression data will be checked for the presence of NA entries and if the latter are present, numerical data will be imputed. (defualt = True)
        :type impute: bool
        :param nPC: Number of principal components and variance explained entries to be calculated. Note that only the first principal component is returned; the rest are used only for the calculation of proportion of variance explained. If given nPC is greater than 10, a warning is issued. (default = 1)
        :type nPC: int
        :param align: Controls whether eigengenes, whose orientation is undetermined, should be aligned with average expression (align = "along average") or left as they are (align = ""). Any other value will trigger an error. (default = "along average")
        :type align: str
        :param excludeGrey: Should the improper module consisting of 'grey' genes be excluded from the eigengenes (default = False)
        :type excludeGrey: bool
        :param subHubs: Controls whether hub genes should be substituted for missing eigengenes. If TRUE, each missing eigengene (i.e., eigengene whose calculation failed and the error was trapped) will be replaced by a weighted average of the most connected hub genes in the corresponding module. If this calculation fails, or if subHubs==FALSE, the value of trapErrors will determine whether the offending module will be removed or whether the function will issue an error and stop. (default = True)
        :type subHubs: bool
        :param trapErrors: Controls handling of errors from that may arise when there are too many NA entries in expression data. If TRUE, errors from calling these functions will be trapped without abnormal exit. If FALSE, errors will cause the function to stop. Note, however, that subHubs takes precedence in the sense that if subHubs==TRUE and trapErrors==FALSE, an error will be issued only if both the principal component and the hubgene calculations have failed. (default = False)
        :type trapErrors: bool
        :param softPower: The power used in soft-thresholding the adjacency matrix. Only used when the hubgene approximation is necessary because the principal component calculation failed. It must be non-negative. The default value should only be changed if there is a clear indication that it leads to incorrect results. (default = 6)
        :type softPower: int
        :param grey: Value of colors or universalColors (whichever applies) designating the improper module
        :type grey: str

        :return: A dictionary similar in spirit to the input exprData
        :rtype: dict
        """
        returnValidOnly = trapErrors
        if grey is None:
            if universalColors is None:
                if isinstance(int, colors):
                    grey = 0
                else:
                    grey = "grey"
            elif isinstance(int, universalColors):
                grey = 0
            else:
                grey = "grey"
        nSets = len(exprData)
        setsize = pyWGCNA.checkSets(exprData, useSets=useSets)
        nGenes = setsize['nGenes']
        nSamples = setsize['nSamples']
        print("multiSetMEs: Calculating module MEs.", flush=True)
        MEs = {}
        consValidMEs = None
        if universalColors is not None:
            consValidColors = universalColors
        if useSets is None:
            useSets = list(range(nSets))
        if useGenes is None:
            for set in useSets:
                print("  Working on set", str(set + 1), "...", flush=True)
                if universalColors is None:
                    setColors = colors[:, set]
                else:
                    setColors = universalColors
                setMEs = pyWGCNA.moduleEigengenes(expr=exprData[set], colors=setColors, impute=impute, nPC=nPC,
                                                align=align, excludeGrey=excludeGrey, grey=grey, trapErrors=trapErrors,
                                                subHubs=subHubs, softPower=softPower)
                if universalColors is not None and not setMEs['allOK']:
                    if consValidMEs is None:
                        consValidMEs = setMEs['validMEs']
                    else:
                        consValidMEs = consValidMEs * setMEs['validMEs']
                    consValidColors[setMEs['validColors'] != universalColors] = setMEs['validColors'][
                        setMEs['validColors'] != universalColors]
                setMEs['data'] = setMEs.pop("eigengenes")
                MEs[set] = setMEs
        else:
            for set in useSets:
                print("  Working on set", str(set), "...", flush=True)
                if universalColors is None:
                    setColors = colors[useGenes, set]
                else:
                    setColors = universalColors[useGenes]
                setMEs = pyWGCNA.moduleEigengenes(expr=exprData[[set]][:, useGenes], colors=setColors, impute=impute,
                                                nPC=nPC, align=align, excludeGrey=excludeGrey, grey=grey,
                                                trapErrors=trapErrors, subHubs=subHubs, softPower=softPower)
                if universalColors is not None and not setMEs['allOK']:
                    if consValidMEs is None:
                        consValidMEs = setMEs['validMEs']
                    else:
                        consValidMEs = consValidMEs * setMEs['validMEs']
                    consValidColors[setMEs['validColors'] != universalColors[useGenes]] = \
                        setMEs['validColors'][setMEs['validColors'] != universalColors[useGenes]]
                setMEs['data'] = setMEs.pop("eigengenes")
                MEs[set] = setMEs
        if universalColors is not None:
            for set in range(nSets):
                if consValidMEs is not None:
                    MEs[set]['validMEs'] = consValidMEs
                MEs[set]['validColors'] = consValidColors
        for set in range(nSets):
            MEs[set]['allOK'] = (sum(np.logical_not(MEs[set]['validMEs'])) == 0)
            if returnValidOnly:
                valid = (MEs[set]['validMEs'] > 0)
                MEs[set]['data'] = MEs[set]['data'][:, valid]
                MEs[set]['averageExpr'] = MEs[set]['averageExpr'][:, valid]
                MEs[set]['varExplained'] = MEs[set]['varExplained'][:, valid]
                MEs[set]['isPC'] = MEs[set]['isPC'][valid]
                MEs[set]['allPC'] = (sum(np.logical_not(MEs[set]['isPC'])) == 0)
                MEs[set]['isHub'] = MEs[set]['isHub'][valid]
                MEs[set]['validAEs'] = MEs[set]['validAEs'][valid]
                MEs[set]['allAEOK'] = (sum(np.logical_not(MEs[set]['validAEs'])) == 0)
                MEs[set]['validMEs'] = np.repeat(True, MEs[set]['data'].shape[1])
        # names(MEs) = names(exprData)
        return MEs

    @staticmethod
    def consensusMEDissimilarityMajor(MEs, useAbs=False, useSets=None, method="consensus"):
        """
        Calculates consensus dissimilarity (1-cor) of given module eigengenes realized in several sets.
        """
        methods = ["consensus", "majority"]
        m = methods.index(method)
        if m is None:
            sys.exit(f"Unrecognized method given. Recognized values are {', '.join(methods)}")
        nSets = len(MEs)
        MEDiss = {}
        if useSets is None:
            useSets = list(range(nSets))
        for set in useSets:
            if useAbs:
                diss = 1 - np.abs(np.corrcoef(MEs[set]['data'], rowvar=False))
            else:
                diss = 1 - np.corrcoef(MEs[set]['data'], rowvar=False)
            diss = pd.DataFrame(diss, index=MEs[set]['data'].columns, columns=MEs[set]['data'].columns)
            MEDiss[set] = {}
            MEDiss[set]['Diss'] = diss
        for set in useSets:
            if set == useSets[0]:
                ConsDiss = MEDiss[set]['Diss']
            else:
                if m == 0:
                    ConsDiss = pd.concat([ConsDiss, MEDiss[set]['Diss']]).max(
                        level=0)  # pmax(ConsDiss, MEDiss[[set]]['Diss'])
                else:
                    ConsDiss = ConsDiss + MEDiss[set]['Diss']
        if m == 1:
            ConsDiss = ConsDiss / nSets
        ConsDiss = pd.DataFrame(ConsDiss, index=np.unique(MEs[useSets[0]]['data'].columns),
                                columns=MEs[useSets[0]]['data'].columns)

        return ConsDiss

    @staticmethod
    def clustOrder(distM, greyLast=True, greyName="MEgrey"):
        distNames = distM.index
        # distM = distM.values
        greyInd = np.where(greyName == distNames)[0].tolist()
        if len(greyInd) == 0:
            greyInd = None
        else:
            greyInd = int(greyInd[0])
        if greyLast and greyInd is not None:
            clusterMEs = np.where(greyName != distNames)[0].tolist()
            if len(clusterMEs) > 1:
                h = pyWGCNA.hclust(pdist(distM.iloc[clusterMEs, clusterMEs]), method="average")
                order = dendrogram(h, no_plot=True)['leaves']  # order
                if len(np.where(np.array(order) >= greyInd)[0].tolist()) > 0:
                    for x in np.where(np.array(order) >= greyInd)[0].tolist():
                        order[x] = order[x] + 1
                order.append(greyInd)
            elif distM.shape[1] > 1:
                if greyInd == 1:
                    order = [1, 0]
                else:
                    order = [0, 1]
            else:
                order = 1
        else:
            if len(distM) > 1:
                h = pyWGCNA.hclust(pdist(distM), method="average")
                order = dendrogram(h, no_plot=True)['leaves']  # order
            else:
                order = 1
        return order

    @staticmethod
    def orderMEs(MEs, greyLast=True, greyName="MEgrey", orderBy=0, order=None, useSets=None):
        """
        Reorder given (eigen-)vectors such that similar ones (as measured by correlation) are next to each other.

        :param MEs: Module eigengenes in a multi-set format.
        :type MEs: dict
        :param greyLast: Normally the color grey is reserved for unassigned genes; hence the grey module is not a proper module and it is conventional to put it last. If this is not desired, set the parameter to FALSE. (default = True)
        :type greyLast: bool
        :param greyName: Name of the grey module eigengene. (default = "MEgrey")
        :type greyName: str
        :param orderBy: Specifies the set by which the eigengenes are to be ordered (in all other sets as well). Defaults to the first set in useSets (or the first set, if useSets is not given). (defualt = 0)
        :type orderBy: int
        :param order: Allows the user to specify a custom ordering.
        :type order: list
        :param useSets: Allows the user to specify for which sets the eigengene ordering is to be performed.
        :type useSets: list

        :return: A dictionary of the same type as MEs containing the re-ordered eigengenes.
        :rtype: dict
        """
        if "eigengenes" in MEs.keys():
            if order is None:
                print("orderMEs: order not given, calculating using given set", str(orderBy), flush=True)
                disPC = 1 - np.corrcoef(MEs['eigengenes'], rowvar=False)
                disPC = pd.DataFrame(disPC, index=MEs['eigengenes'].index, columns=MEs['eigengenes'].index)
                order = pyWGCNA.clustOrder(disPC, greyLast=greyLast, greyName=greyName)
            if len(order) != MEs['eigengenes'].shape[1]:
                sys.exit("orderMEs: given MEs and order have incompatible dimensions.")
            orderedMEs = MEs.copy()
            orderedMEs['eigengenes'] = pd.DataFrame(MEs['eigengenes'][:, order])
            orderedMEs['eigengenes'].columns = MEs['eigengenes'].columns[order]
            if MEs['averageExpr'] is not None:
                orderedMEs['averageExpr'] = pd.DataFrame(MEs['averageExpr'][:, order])
                orderedMEs['averageExpr'].columns = MEs['eigengenes'].columns[order]

            if MEs['varExplained'] is not None:
                orderedMEs['varExplained'] = pd.DataFrame(MEs['varExplained'][:, order])
                orderedMEs['varExplained'].columns = MEs['eigengenes'].columns[order]
            return orderedMEs
        else:
            check = pyWGCNA.checkSets(MEs, checkStructure=True, useSets=useSets)
            if check['structureOK']:
                multiSet = True
            else:
                multiSet = False
                MEs = pyWGCNA.fixDataStructure(MEs)
                useSets = None
                orderBy = 0
            if useSets is not None:
                if useSets.index(orderBy) is None:
                    orderBy = useSets[1]
            if order is None:
                print("orderMEs: order not given, calculating using given set", str(orderBy), flush=True)
                disPC = 1 - np.corrcoef(MEs[orderBy]['data'], rowvar=False)
                disPC = pd.DataFrame(disPC, index=MEs[orderBy]['data'].columns, columns=MEs[orderBy]['data'].columns)
                order = pyWGCNA.clustOrder(disPC, greyLast=greyLast, greyName=greyName)
            if len(order) != MEs[orderBy]['data'].shape[1]:
                sys.exit("orderMEs: given MEs and order have incompatible dimensions.")
            nSets = len(MEs)
            orderedMEs = MEs.copy()
            if useSets is None:
                useSets = list(range(nSets))
            for set in useSets:
                orderedMEs[set]['data'] = MEs[set]['data'].iloc[:, order]
                if "averageExpr" in MEs[set].keys():
                    if MEs[set]['averageExpr'] is not None:
                        orderedMEs[set]['averageExpr'] = MEs[set]['averageExpr'].iloc[:, order]
                if "varExplained" in MEs[set].keys():
                    if MEs[set]['varExplained'] is not None:
                        orderedMEs[set]['varExplained'] = MEs[set]['varExplained'].iloc[:, order]
            if multiSet:
                return orderedMEs
            else:
                return orderedMEs[0]['data']

    @staticmethod
    def consensusOrderMEs(MEs, useAbs=False, useSets=None, greyLast=True, greyName="MEgrey", method="consensus"):
        """
        Reorder given (eigen-)vectors such that similar ones (as measured by correlation) are next to each other.

        :param MEs: Module eigengenes of several sets in a multi-set format
        :type MEs: dict
        :param useAbs: Controls whether vector similarity should be given by absolute value of correlation or plain correlation. (defualt = False)
        :type useAbs: bool
        :param useSet: Allows the user to specify for which sets the eigengene ordering is to be performed.
        :type useSets: list
        :param greyLast: Normally the color grey is reserved for unassigned genes; hence the grey module is not a proper module and it is conventional to put it last. If this is not desired, set the parameter to FALSE. (defualt = True)
        :type greyLast: bool
        :param greyName: Name of the grey module eigengene. (defualt = "MEgrey")
        :type greyName: str
        :param method: A character string giving the method to be used calculating the consensus dissimilarity. Allowed values are (abbreviations of) "consensus" and "majority". The consensus dissimilarity is calculated as the maximum of given set dissimilarities for "consensus" and as the average for "majority".
        :type method: str

        :return: A dictionary of the same type as MEs containing the re-ordered eigengenes
        :rtype: dict
        """
        Diss = pyWGCNA.consensusMEDissimilarityMajor(MEs, useAbs=useAbs, useSets=useSets, method=method)
        order = pyWGCNA.clustOrder(Diss, greyLast, greyName)
        MEs = pyWGCNA.orderMEs(MEs, greyLast=greyLast, greyName=greyName, order=order, useSets=useSets)
        return MEs

    @staticmethod
    def equalizeQuantilesFun(data, summaryType=["median", "mean"]):
        # summaryType = match.arg(summaryType)
        data_sorted = pd.DataFrame(np.sort(data, axis=0))  # apply(data, 2, sort)
        if summaryType == "median":
            refSample = data_sorted.median(axis=0)
        elif summaryType == "mean":
            refSample = data_sorted.mean(axis=0)
        ranks = round(pd.DataFrame(data).rank(axis=1))
        out = pd.DataFrame(refSample[ranks], index=data.index, columns=data.columns)
        # dim(out = dim(data)
        # dimnames(out) = dimnames(data)
        return out

    @staticmethod
    def consensusMEDissimilarity(multiMEs, useSets=None, equalizeQuantiles=False, quantileSummary="mean",
                                 corOptions={}, consensusQuantile=0, useAbs=False, greyName="ME0"):
        nSets = pyWGCNA.checkSets(multiMEs)['nSets']
        useMEs = np.where(multiMEs[0]['data'].columns != greyName)[0].tolist()
        useNames = multiMEs[0]['data'].columns[useMEs]
        nUseMEs = len(useMEs)
        if useSets is None:
            useSets = list(range(nSets))
        nUseSets = len(useSets)
        MEDiss = np.zeros((nUseMEs, nUseMEs, nUseSets))
        MEDiss[:, :, :] = np.nan
        for set in useSets:
            corOptions['x'] = multiMEs[set]['data'].iloc[:, useMEs]
            if useAbs:
                diss = 1 - np.abs(np.corrcoef(corOptions['x'], rowvar=False))
            else:
                diss = 1 - np.corrcoef(corOptions['x'], rowvar=False)
            diss = pd.DataFrame(diss, index=corOptions['x'].columns, columns=corOptions['x'].columns)
            MEDiss[:, :, set] = diss
        if equalizeQuantiles:
            distMat = pd.DataFrame()
            for i in range(MEDiss.shape[2]):
                distMat[i] = pdist(MEDiss[:, :, i])
            # distMat = apply(MEDiss, 3, function(x) { as.numeric( as.dist(x))})
            # TODO: checkdistMat.shape = [nUseMEs * (nUseMEs - 1) / 2, nUseSets]
            normalized = pyWGCNA.equalizeQuantilesFun(distMat, summaryType=quantileSummary)
            # TODO:apply(normalized, 2, .turnDistVectorIntoMatrix, size = nUseMEs, Diag = FALSE, Upper = FALSE, diagValue = 0)
            MEDiss = normalized
            np.fill_diagonal(normalized, 0)
        ConsDiss = pd.DataFrame(np.quantile(MEDiss, q=1 - consensusQuantile, axis=2), index=useNames.unique(),
                                columns=useNames.unique())
        return ConsDiss

    @staticmethod
    def moduleNumber(dendro, cutHeight=0.9, minSize=50):
        Branches = pyWGCNA.cutree(dendro, cutHeight=cutHeight)[:, 0].tolist()
        NOnBranches = pd.DataFrame(Branches).value_counts()
        TrueBranch = NOnBranches >= minSize
        if len(np.where(np.logical_not(TrueBranch.iloc[Branches]))[0].tolist()) != 0:
            Branches[np.where(np.logical_not(TrueBranch.iloc[Branches]))[0].tolist()] = 0
        return Branches

    @staticmethod
    def mergeCloseModules(exprData, colors, MEs=None, useSets=None, impute=True, checkDataFormat=True,
                          unassdColor="grey", useAbs=False, equalizeQuantiles=False, quantileSummary="mean",
                          consensusQuantile=0, cutHeight=0.2, iterate=True, relabel=False, colorSeq=None,
                          getNewMEs=True, getNewUnassdME=True, trapErrors=False):
        """
        Merges modules in gene expression networks that are too close as measured by the correlation of their eigengenes.

        :param exprData: Expression data, either a single data frame with rows corresponding to samples and columns to genes, or in a multi-set format.
        :type exprData: pandas dataframe
        :param colors: A list (numeric, character or a factor) giving module colors for genes. The method only makes sense when genes have the same color label in all sets, hence a single list.
        :type colors: list
        :param MEs: If module eigengenes have been calculated before, the user can save some computational time by inputting them. MEs should have the same format as exprData. If they are not given, they will be calculated.
        :type MEs: dict
        :param useSets: A list of scalar allowing the user to specify which sets will be used to calculate the consensus dissimilarity of module eigengenes. Defaults to all given sets.
        :type useSets: list
        :param impute: Should missing values be imputed in eigengene calculation? If imputation is disabled, the presence of NA entries will cause the eigengene calculation to fail and eigengenes will be replaced by their hubgene approximation. (defualt = True)
        :type impute: bool
        :param checkDataFormat: If TRUE, the function will check exprData and MEs for correct multi-set structure. If single set data is given, it will be converted into a format usable for the function. If FALSE, incorrect structure of input data will trigger an error. (defualt = True)
        :type checkDataFormat: bool
        :param unassdColor: Specifies the string that labels unassigned genes. Module of this color will not enter the module eigengene clustering and will not be merged with other modules. (default = "grey")
        :type unassdColor: str
        :param useAbs: Specifies whether absolute value of correlation or plain correlation (of module eigengenes) should be used in calculating module dissimilarity. (defualt = False)
        :type useAbs: bool
        :param equalizeQuantiles: should quantiles of the eigengene dissimilarity matrix be equalized ("quantile normalized")? The default is FALSE for reproducibility of old code; when there are many eigengenes (e.g., at least 50), better results may be achieved if quantile equalization is used. (defualt = False)
        :type equalizeQuantiles: bool
        :param quantileSummary: One of "mean" or "median". Controls how a reference dissimilarity is computed from the input ones (using mean or median, respectively). (default = "mean")
        :type quantileSummary: str
        :param consensusQuantile: A number giving the desired quantile to use in the consensus similarity calculation. (defualt = 0)
        :type consensusQuantile: int
        :param cutHeight: Maximum dissimilarity (i.e., 1-correlation) that qualifies modules for merging. (defualt = 0.2)
        :type cutHeight: float
        :param iterate: Controls whether the merging procedure should be repeated until there is no change. If FALSE, only one iteration will be executed. (defualt = True)
        :type iterate: bool
        :param relabel: Controls whether, after merging, color labels should be ordered by module size. (defualt = False)
        :type relabel: bool
        :param colorSeq: Color labels to be used for relabeling. Defaults to the standard color order used in this package if colors are not numeric, and to integers starting from 1 if colors is numeric.
        :type colorSeq: list
        :param getNewMEs: Controls whether module eigengenes of merged modules should be calculated and returned. (defualt = True)
        :type getNewMEs: bool
        :param getNewUnassdME: When doing module eigengene manipulations, the function does not normally calculate the eigengene of the 'module' of unassigned ('grey') genes. Setting this option to TRUE will force the calculation of the unassigned eigengene in the returned newMEs, but not in the returned oldMEs. (defualt = True)
        :type getNewUnassdME: bool
        :param trapErrors: Controls whether computational errors in calculating module eigengenes, their dissimilarity, and merging trees should be trapped. If TRUE, errors will be trapped and the function will return the input colors. If FALSE, errors will cause the function to stop. (defualt = False)

        :return: A dictionaty contains: "colors": Color labels for the genes corresponding to merged modules. The function attempts to mimic the mode of the input colors: if the input colors is numeric, character and factor, respectively, so is the output. Note, however, that if the fnction performs relabeling, a standard sequence of labels will be used: integers starting at 1 if the input colors is numeric, and a sequence of color labels otherwise. "dendro": Hierarchical clustering dendrogram (average linkage) of the eigengenes of the most recently computed tree. If iterate was set TRUE, this will be the dendrogram of the merged modules, otherwise it will be the dendrogram of the original modules. "oldDendro": Hierarchical clustering dendrogram (average linkage) of the eigengenes of the original modules. "cutHeight": The input cutHeight. "oldMEs": Module eigengenes of the original modules in the sets given by useSets. "newMEs": Module eigengenes of the merged modules in the sets given by useSets. "allOK": A boolean set to TRUE.
        :raises trapErrors==TRUE: A dictionaty contains: "colors": A copy of the input colors. "allOK": a boolean set to FALSE.
        :rtype: dict
        """
        if all(isinstance(x, int) for x in colors):
            unassdColor = 0
        MEsInSingleFrame = False
        origColors = colors
        greyName = "ME" + unassdColor
        print("mergeCloseModules: Merging modules whose distance is less than", str(cutHeight), flush=True)
        if not pyWGCNA.checkSets(exprData, checkStructure=True, useSets=useSets)['structureOK']:
            if checkDataFormat:
                exprData = pyWGCNA.fixDataStructure(exprData)
                MEsInSingleFrame = True
            else:
                sys.exit("Given exprData appear to be misformatted.")
        setsize = pyWGCNA.checkSets(exprData, useSets=useSets)
        nSets = setsize['nSets']
        if MEs is not None:
            checkMEs = pyWGCNA.checkSets(MEs, checkStructure=True, useSets=useSets)
            if checkMEs['structureOK']:
                if setsize['nSets'] != checkMEs['nSets']:
                    sys.exit("Input error: numbers of sets in exprData and MEs differ.")
                for set in range(nSets):
                    if checkMEs['nSamples'][set] != setsize['nSamples'][set]:
                        sys.exit(("Number of samples in MEs is incompatible with subset length for set", str(set)))
            else:
                if MEsInSingleFrame:
                    MEs = pyWGCNA.fixDataStructure(MEs)
                    checkMEs = pyWGCNA.checkSets(MEs)
                else:
                    sys.exit("MEs do not have the appropriate structure (same as exprData). ")
        if setsize['nGenes'] != len(colors):
            sys.exit("Number of genes in exprData is different from the length of original colors. They must equal.")
        if cutHeight < 0 or cutHeight > 1 + int(useAbs):
            sys.exit(("Given cutHeight is out of sensible range between 0 and", 1 + int(useAbs)))
        done = False
        iteration = 1
        MergedColors = colors
        try:
            while not done:
                if MEs is None:
                    MEs = pyWGCNA.multiSetMEs(exprData, colors=None, universalColors=colors, useSets=useSets,
                                            impute=impute, subHubs=True, trapErrors=False, excludeGrey=True,
                                            grey=unassdColor)
                    MEs = pyWGCNA.consensusOrderMEs(MEs, useAbs=useAbs, useSets=useSets, greyLast=False)
                elif len(pd.Categorical(colors).codes) != checkMEs['nGenes']:
                    if iteration == 1:
                        print(
                            "Number of given module colors does not match number of given MEs => recalculating the MEs.",
                            flush=True)
                    MEs = pyWGCNA.multiSetMEs(exprData, colors=None, universalColors=colors, useSets=useSets,
                                            impute=impute, subHubs=True, trapErrors=False, excludeGrey=True,
                                            grey=unassdColor)
                    MEs = pyWGCNA.consensusOrderMEs(MEs, useAbs=useAbs, useSets=useSets, greyLast=False)
                if iteration == 1:
                    oldMEs = MEs
                colLevs = pd.Categorical(colors)
                if len(colLevs[colLevs != str(unassdColor)]) < 2:
                    print("mergeCloseModules: less than two proper modules.", flush=True)
                    print(" ..color levels are", colLevs, flush=True)
                    print(" ..there is nothing to merge.", flush=True)
                    MergedNewColors = colors
                    MergedColors = colors
                    nOldMods = 1
                    nNewMods = 1
                    oldTree = None
                    Tree = None
                    break
                nOldMods = len(pd.Categorical(colors).categories)
                ConsDiss = pyWGCNA.consensusMEDissimilarity(MEs, equalizeQuantiles=equalizeQuantiles,
                                                          quantileSummary=quantileSummary,
                                                          consensusQuantile=consensusQuantile, useAbs=useAbs,
                                                          useSets=useSets, greyName=greyName)
                a = squareform(ConsDiss, checks=False)
                Tree = pyWGCNA.hclust(a, method="average")
                if iteration == 1:
                    oldTree = Tree
                TreeBranches = pyWGCNA.cutree(Tree, cutHeight=cutHeight)[:, 0]
                UniqueBranches = pd.Categorical(TreeBranches)
                nBranches = len(UniqueBranches.categories)
                NumberOnBranch = pd.DataFrame(TreeBranches).value_counts().sort_index()
                MergedColors = colors
                TreeBranches = pd.DataFrame(TreeBranches, index=ConsDiss.index).T
                for branch in range(nBranches):
                    if NumberOnBranch[branch] > 1:
                        ModulesOnThisBranch = TreeBranches.columns[
                            np.where(TreeBranches == UniqueBranches[branch])[1].tolist()]
                        ColorsOnThisBranch = [x[2:] for x in ModulesOnThisBranch]
                        if all(isinstance(x, int) for x in origColors):
                            ColorsOnThisBranch = [int(x) for x in ColorsOnThisBranch]
                        for color in range(1, len(ColorsOnThisBranch)):
                            MergedColors[MergedColors == ColorsOnThisBranch[color]] = ColorsOnThisBranch[0]
                # MergedColors = MergedColors[:, drop = TRUE]
                nNewMods = len(pd.Categorical(MergedColors).categories)
                if nNewMods < nOldMods and iterate:
                    colors = MergedColors
                    MEs = None
                else:
                    done = True
                iteration = iteration + 1
            if relabel:
                RawModuleColors = pd.Categorical(MergedColors).codes
                if colorSeq is None:
                    if isinstance(int, origColors):
                        colorSeq = list(range(len(pd.DataFrame(origColors).value_counts())))
                    else:
                        nNewColors = len(RawModuleColors)
                        colorSeq = pyWGCNA.labels2colors(list(range(nNewColors)))
                nGenesInModule = pd.DataFrame(MergedColors).value_counts()
                SortedRawModuleColors = RawModuleColors[nGenesInModule.sort(reverse=True)]
                MergedNewColors = MergedColors
                if isinstance(pd.Categorical, MergedNewColors):
                    MergedNewColors = str(MergedNewColors)
                rank = 0
                for color in range(len(SortedRawModuleColors)):
                    if SortedRawModuleColors[color] != unassdColor:
                        rank = rank + 1
                        MergedNewColors[MergedColors == SortedRawModuleColors[color]] = colorSeq[rank]
                if isinstance(pd.Categorical, MergedColors):
                    MergedNewColors = pd.Categorical(MergedNewColors)
            else:
                MergedNewColors = MergedColors
            # MergedNewColors = MergedNewColors[, drop = TRUE]
            if getNewMEs:
                if nNewMods < nOldMods or relabel or getNewUnassdME:
                    print("  Calculating new MEs...", flush=True)
                    NewMEs = pyWGCNA.multiSetMEs(exprData, colors=None, universalColors=MergedNewColors, useSets=useSets,
                                               impute=impute, subHubs=True, trapErrors=False,
                                               excludeGrey=not getNewUnassdME, grey=unassdColor)
                    newMEs = pyWGCNA.consensusOrderMEs(NewMEs, useAbs=useAbs, useSets=useSets, greyLast=True,
                                                     greyName=greyName)
                    ConsDiss = pyWGCNA.consensusMEDissimilarity(newMEs, equalizeQuantiles=equalizeQuantiles,
                                                              quantileSummary=quantileSummary,
                                                              consensusQuantile=consensusQuantile,
                                                              useAbs=useAbs, useSets=useSets, greyName=greyName)
                    if len(ConsDiss) > 1:
                        Tree = pyWGCNA.hclust(pdist(ConsDiss), method="average")
                    else:
                        Tree = None
                else:
                    newMEs = MEs
            else:
                newMEs = None
            if MEsInSingleFrame:
                newMEs = newMEs[0]  # $data
                oldMEs = oldMEs[0]  # $data
        except:
            if not trapErrors:
                sys.exit("Error!")
            print("Warning: merging of modules failed", flush=True)
            print(" --> returning unmerged modules and *no* eigengenes.", flush=True)
            print(
                f"{WARNING}mergeCloseModules: merging of modules failed --> returning unmerged modules and *no* "
                f"eigengenes.\n{ENDC}")
            return {"colors": origColors, "allOK": False}
        else:
            return {"colors": MergedNewColors, "dendro": Tree, "oldDendro": oldTree, "cutHeight": cutHeight,
                    "oldMEs": oldMEs['data'], "newMEs": newMEs['data'], "allOK": True}

    def saveWGCNA(self):
        """
        Saves the current WGCNA in pickle format with the .p extension
        """
        print(f"{BOLD}{OKBLUE}Saving WGCNA as {self.name}.p{ENDC}")

        picklefile = open(self.outputPath + self.name + '.p', 'wb')
        pickle.dump(self, picklefile)
        picklefile.close()

    def getDatTraits(self, metaData):
        """
        get data trait module base on samples information

        :return: a dataframe contains information in suitable format for plotting module trait relationship heatmap
        :rtype: pandas dataframe
        """
        data = self.datExpr.obs.copy()[metaData]
        datTraits = pd.DataFrame(index=data.index)
        for i in range(data.shape[1]):
            if is_numeric_dtype(data.iloc[:, i].dtypes):
                datTraits[data.columns[i]] = data.iloc[:, i]
                continue
            data.iloc[:, i] = data.iloc[:, i].astype(str)
            if len(np.unique(data.iloc[:, i])) == 2:
                datTraits[data.columns[i]] = data.iloc[:, i]
                org = np.unique(data.iloc[:, i]).tolist()
                rep = list(range(len(org)))
                datTraits.replace(to_replace=org, value=rep,
                                  inplace=True)
            elif len(np.unique(data.iloc[:, i])) > 2:
                for name in np.unique(data.iloc[:, i]):
                    datTraits[name] = data.iloc[:, i]
                    org = np.unique(data.iloc[:, i])
                    rep = np.repeat(0, len(org))
                    rep[np.where(org == name)] = 1
                    org = org.tolist()
                    rep = rep.tolist()
                    datTraits.replace(to_replace=org, value=rep, inplace=True)

        return datTraits

    def module_trait_relationships_heatmap(self,
                                           metaData,
                                           alternative='two-sided',
                                           figsize=None,
                                           show=True,
                                           file_name='module-traitRelationships'):
        """
        plot topic-trait relationship heatmap

        :param metaData: traits you would like to see the relationship with topics (must be column name of datExpr.obs)
        :type metaData: list
        :param alternative: Defines the alternative hypothesis for calculating correlation for module-trait relationship. Default is ‘two-sided’. The following options are available: 'two-sided’: the correlation is nonzero, ‘less’: the correlation is negative (less than zero), ‘greater’: the correlation is positive (greater than zero)
        :type alternative: str
        :param figsize: indicate the size of plot
        :type figsize: tuple of float
        :param show: indicate if you want to show the plot or not (default: True)
        :type show: bool
        :param file_name: name and path of the plot use for save (default: topic-traitRelationships)
        :type file_name: str
        """
        datTraits = self.getDatTraits(metaData)

        moduleTraitCor = pd.DataFrame(index=self.MEs.columns,
                                      columns=datTraits.columns,
                                      dtype="float")
        moduleTraitPvalue = pd.DataFrame(index=self.MEs.columns,
                                         columns=datTraits.columns,
                                         dtype="float")

        for i in self.MEs.columns:
            for j in datTraits.columns:
                tmp = stats.pearsonr(self.MEs[i], datTraits[j], alternative=alternative)
                moduleTraitCor.loc[i, j] = tmp[0]
                moduleTraitPvalue.loc[i, j] = tmp[1]

        if set(metaData) == set(self.datExpr.obs.columns.tolist()):
            self.moduleTraitCor = moduleTraitCor
            self.moduleTraitPvalue = moduleTraitPvalue

        if figsize is None:
            figsize = (max(20, int(moduleTraitPvalue.shape[0] * 1.5)),
                       moduleTraitPvalue.shape[1] * 1.5)
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        # names
        xlabels = []
        for label in self.MEs.columns:
            xlabels.append(label[2:].capitalize() + '(' + str(sum(self.datExpr.var['moduleColors'] == label[2:])) + ')')
        ylabels = datTraits.columns

        # Loop over data dimensions and create text annotations.
        tmp_cor = moduleTraitCor.T.round(decimals=2)
        tmp_pvalue = moduleTraitPvalue.T.round(decimals=3)
        labels = (np.asarray(["{0}\n({1})".format(cor, pvalue)
                              for cor, pvalue in zip(tmp_cor.values.flatten(),
                                                     tmp_pvalue.values.flatten())])) \
            .reshape(moduleTraitCor.T.shape)

        sns.set(font_scale=1.5)
        res = sns.heatmap(moduleTraitCor.T, annot=labels, fmt="", cmap='RdBu_r',
                          vmin=-1, vmax=1, ax=ax, annot_kws={'size': 20, "weight": "bold"},
                          xticklabels=xlabels, yticklabels=ylabels)
        res.set_xticklabels(res.get_xmajorticklabels(), fontsize=20, fontweight="bold", rotation=90)
        res.set_yticklabels(res.get_ymajorticklabels(), fontsize=20, fontweight="bold")
        plt.yticks(rotation=0)
        ax.set_title(f"Module-trait Relationships heatmap for {self.name}",
                     fontsize=30, fontweight="bold")
        ax.set_facecolor('white')
        fig.tight_layout()
        if not show:
            plt.close(fig)
        if self.save:
            fig.savefig(f"{self.outputPath}figures/{file_name}.{self.figureType}")

    def getModuleName(self):
        """
        get names of modules

        :return: name of modules
        :rtype: ndarray
        """
        return np.unique(self.datExpr.var['moduleColors']).tolist()

    def getGeneModule(self, moduleName):
        """
        get list of genes corresponding to modules

        :param moduleName: name of modules
        :type moduleName: list

        :return: A dictionary contains list of genes for requested module(s)
        :rtype: dict
        """
        output = {}
        moduleColors = np.unique(self.datExpr.var['moduleColors']).tolist()
        if moduleName not in moduleColors:
            print(f"{WARNING}Module name(s) does not exist in {ENDC}")
            return None
        for color in moduleColors:
            if color in moduleName:
                output[color] = self.datExpr.var[self.datExpr.var['moduleColors'] == color]
        return output

    def getModulesGene(self, geneIds):
        """
        get list of modules corresponding to gene(s)

        :param geneIds: gene id
        :type geneIds: list or str

        :return: A list contains name of module(s) for requested gene(s)
        :rtype: list or str
        """
        if isinstance(geneIds, str):
            geneIds = [geneIds]

        modules = []
        for geneId in geneIds:
            modules.append(self.datExpr.var.moduleColors[self.datExpr.var.gene_id == geneId])

        if len(modules) == 1:
            modules = modules[0]

        return modules

    def setMetadataColor(self, col, cmap):
        """
        set color pallete for each group of metadata

        :param col: name of metadata
        :type col: str
        :param cmap: color pallet
        :type cmap: list
        """
        # check if obs_col is even there
        if col not in self.datExpr.obs.columns.tolist():
            print(f"{WARNING}Metadata column {col} not found!{ENDC}")
            return None
        self.metadataColors[col] = cmap

    def plotModuleEigenGene(self, moduleName, metadata, show=True):
        """
        plot module eigen gene figure in given module

        :param moduleName: module name
        :type moduleName: str
        :param metadata: list of metadata you want to be plotted
        :type metadata: list
        :param show: indicate if you want to see plots in when you run your code
        :type show: bool
        """
        sampleInfo = self.datExpr.obs

        height_ratios = []
        for m in metadata:
            if isinstance(self.metadataColors[m], dict):
                height_ratios.append(len(list(self.metadataColors[m].keys())))
            else:
                height_ratios.append(1)
        height_ratios.reverse()

        modules = np.unique(self.datExpr.var['moduleColors']).tolist()
        if np.all(moduleName not in modules):
            print(f"{WARNING}Module name does not exist in {ENDC}")
            return None
        else:
            heatmap = self.datExpr[:, self.datExpr.var['moduleColors'] == moduleName].to_df()
            heatmap = (heatmap - heatmap.min(axis=0)) / (heatmap.max(axis=0) - heatmap.min(axis=0))
            heatmap = heatmap.T
            a = pdist(heatmap)
            np.nan_to_num(a, copy=False)
            Z = pyWGCNA.hclust(a, method="average")
            # Clusterize the data
            labels = fcluster(Z, t=0.8, criterion='distance')
            # Keep the indices to sort labels
            labels_order = np.argsort(labels)
            heatmap = heatmap.iloc[labels_order, :]
            ME = pd.DataFrame(self.datME["ME" + moduleName].values, columns=['eigengeneExp'])
            ME['sample_name'] = self.datME.index
            ind = [i + 0.5 for i in range(ME.shape[0])]

            if len(metadata) != 0:
                fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(ME.shape[0], len(metadata) * 5),
                                        sharex='col', gridspec_kw={
                        'height_ratios': [len(metadata) * 0.4, len(metadata) * 0.5, len(metadata) * 1.5],
                        'width_ratios': [20, 3]})

                gs = axs[0, 1].get_gridspec()
                # remove the underlying axes
                for ax in axs[:, 1]:
                    ax.remove()
                ax_legend = fig.add_subplot(gs[:, 1])
                ax_legend.axis('off')
                axs_legend = gridspec.GridSpecFromSubplotSpec(len(metadata), 1,
                                                              subplot_spec=ax_legend.get_subplotspec(),
                                                              height_ratios=height_ratios)

                for m in metadata:
                    handles = []
                    x = ind
                    y = np.repeat(3000 * metadata.index(m), len(ind))
                    color = sampleInfo[m].values
                    if type(self.metadataColors[m]) == dict:
                        for n in list(self.metadataColors[m].keys()):
                            color = np.where(color == n, self.metadataColors[m][n], color)
                            patch = mpatches.Patch(color=self.metadataColors[m][n], label=n)
                            handles.append(patch)
                        axs[0, 0].scatter(x, y, c=color, s=1600, marker='s')
                        ax_legend = plt.Subplot(fig, axs_legend[len(metadata) - 1 - metadata.index(m)])
                        ax_legend.legend(title=m, handles=handles)
                        ax_legend.axis('off')
                        fig.add_subplot(ax_legend)
                    else:
                        axs[0, 0].scatter(x, y, c=color, cmap=self.metadataColors[m].get_cmap(),
                                              s=1600, marker='s')

                        ax_legend = plt.Subplot(fig, axs_legend[len(metadata) - 1 - metadata.index(m)])
                        clb = fig.colorbar(mappable=self.metadataColors[m],
                                               ax=ax_legend,
                                               orientation='horizontal',
                                               fraction=0.9)
                        clb.ax.set_title(m.capitalize())
                        ax_legend.axis('off')
                        fig.add_subplot(ax_legend)

                axs[0, 0].set_title(f"Module Eigengene for {moduleName}", size=28, fontweight="bold")
                axs[0, 0].set_ylim(-2000, np.max(y) + 2000)
                axs[0, 0].grid(False)
                axs[0, 0].axis('off')

                axs[1, 0].bar(ind, ME.eigengeneExp, align='center', color='black')
                axs[1, 0].set_ylabel('eigengeneExp')
                axs[1, 0].set_facecolor('white')

                cmap = sns.color_palette("dark:red", as_cmap=True)
                sns.heatmap(heatmap, cmap=cmap,
                            cbar=False,  # cbar_ax=axs[2,1],
                            yticklabels=False, xticklabels=False,
                            ax=axs[2, 0])
            else:
                fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(ME.shape[0], 20),
                                        sharex='col', gridspec_kw={
                        'height_ratios': [1, 1.5]}, facecolor='white')

                axs[0].set_title(f"Module Eigengene for {moduleName}", size=28, fontweight="bold")
                axs[0].bar(ind, ME.eigengeneExp, align='center', color='black')
                axs[0].set_ylabel('eigengeneExp')

                cmap = sns.color_palette("dark:red", as_cmap=True)
                sns.heatmap(heatmap, cmap=cmap,
                            cbar=False,  # cbar_ax=axs[2,1],
                            yticklabels=False, xticklabels=False,
                            ax=axs[1])

            if self.save:
                fig.savefig(f"{self.outputPath}figures/module_heatmap_eigengene_{moduleName}.{self.figureType}")
            if not show:
                plt.close(fig)
            else:
                return axs

        return None

    def barplotModuleEigenGene(self, moduleName, metadata, combine=True, colorBar=None, show=True):
        """
        bar plot of module eigen gene figure in given module

        :param moduleName: module name
        :type moduleName: str
        :param metadata: list of metadata you want to be plotted
        :type metadata: list
        :param combine: indicate if you want to combine all metadata to show them together
        :type combine: bool
        :praram colorBar: metadata you want to use to color bar plot with
        :type colorBar: str
        :param show: indicate if you want to see plots in when you run your code
        :type show: bool
        """
        sampleInfo = self.datExpr.obs
        if sampleInfo.shape[1] == 0:
            print(
                f"{WARNING}There is not any sample information in given object! Skip running barplotModuleEigenGene() function.{ENDC}")
            return None

        height_ratios = []
        for m in metadata:
            if isinstance(self.metadataColors[m], dict):
                height_ratios.append(len(list(self.metadataColors[m].keys())))
            else:
                height_ratios.append(1)
        height_ratios.reverse()

        modules = np.unique(self.datExpr.var['moduleColors']).tolist()
        if np.all(moduleName not in modules):
            print(f"{WARNING}Module name does not exist in {ENDC}")
            return None
        else:
            ME = pd.DataFrame(self.datME["ME" + moduleName].values, columns=['eigengeneExp'])

            if combine:
                df = ME.copy(deep=True)
                df['all'] = ''
                for m in metadata:
                    df[m] = sampleInfo[m].values
                    df['all'] = df['all'] + '_' + df[m].astype(str)
                df['all'] = df['all'].apply(lambda x: x[1:])
                cat = pd.DataFrame(pd.unique(df['all']), columns=['all'])
                cat[metadata] = cat['all'].str.split('_', expand=True)
                ybar = df[['all', 'eigengeneExp']].groupby(['all']).mean()['eigengeneExp']
                ebar = df[['all', 'eigengeneExp']].groupby(['all']).std()['eigengeneExp']
                ybar = ybar.loc[cat['all']]
                ebar = ebar.loc[cat['all']]
                label = list(ybar.index)
                dot = df[['all', 'eigengeneExp']].copy()
                ind = {}
                for i in range(cat.shape[0]):
                    ind[cat.loc[i, 'all']] = cat.index[i]
                dot.replace(ind, inplace=True)
                xdot = dot['all']
                ydot = dot['eigengeneExp']

                if colorBar is None:
                    palette = "lightblue"
                else:
                    palette = cat[[colorBar]].copy()
                    palette.replace(self.metadataColors[colorBar], inplace=True)
                    palette = palette[colorBar].values

                fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(cat.shape[0] + 2, len(metadata) * 4),
                                        sharex='col', gridspec_kw={
                        'height_ratios': [len(metadata) * 0.4, len(metadata) * 0.6],
                        'width_ratios': [cat.shape[0], 3]})

                gs = axs[0, 1].get_gridspec()
                # remove the underlying axes
                for ax in axs[:, 1]:
                    ax.remove()
                ax_legend = fig.add_subplot(gs[:, 1])
                ax_legend.axis('off')
                axs_legend = gridspec.GridSpecFromSubplotSpec(len(metadata), 1, 
                                                              subplot_spec=ax_legend.get_subplotspec(),
                                                              height_ratios=height_ratios)

                ind = [i for i in range(cat.shape[0])]
                for m in metadata:
                    handles = []
                    x = ind
                    y = np.repeat(3000 * metadata.index(m), len(ind))
                    color = cat[m].values
                    if type(self.metadataColors[m]) == dict:
                        for n in list(self.metadataColors[m].keys()):
                            color = np.where(color == n, self.metadataColors[m][n], color)
                            patch = mpatches.Patch(color=self.metadataColors[m][n], label=n)
                            handles.append(patch)
                        if m != colorBar:
                            axs[0, 0].scatter(x, y, c=color, s=1600, marker='s')
                        ax_legend = plt.Subplot(fig, axs_legend[len(metadata) - 1 - metadata.index(m)])
                        ax_legend.legend(title=m, handles=handles)
                        ax_legend.axis('off')
                        fig.add_subplot(ax_legend)
                    else:
                        cat[m] = cat[m].astype('float')
                        color = cat[m].values
                        axs[0, 0].scatter(x, y, c=color,
                                          cmap=self.metadataColors[m].get_cmap(),
                                          s=1600, marker='s')

                        ax_legend = plt.Subplot(fig, axs_legend[len(metadata) - 1 - metadata.index(m)])
                        clb = fig.colorbar(mappable=self.metadataColors[m],
                                           ax=ax_legend,
                                           orientation='horizontal',
                                           fraction=0.9)
                        clb.ax.set_title(m.capitalize())
                        ax_legend.axis('off')
                        fig.add_subplot(ax_legend)
                    # if m != colorBar:
                    #     axs[0, 0].scatter(x, y, c=color, s=1600, marker='s')
                    # ax_legend = plt.Subplot(fig, axs_legend[len(metadata) - 1 - metadata.index(m)])
                    # ax_legend.legend(title=m, handles=handles)
                    # ax_legend.axis('off')
                    # fig.add_subplot(ax_legend)

                axs[0, 0].set_title(f"Module Eigengene for {moduleName}", size=28, fontweight="bold")
                axs[0, 0].set_ylim(-2000, np.max(y) + 2000)
                axs[0, 0].grid(False)
                axs[0, 0].axis('off')

                ind = [i for i in range(cat.shape[0])]
                axs[1, 0].bar(ind, ybar, align='center', color=palette)
                axs[1, 0].errorbar(ind, ybar, yerr=ebar, fmt="o", color="r")
                axs[1, 0].scatter(xdot, ydot, c='black', alpha=0.5)
                axs[1, 0].set_xticks(np.arange(len(ind)))
                axs[1, 0].set_xticklabels(label, rotation=90)
                axs[1, 0].set_ylabel('eigengeneExp')
                axs[1, 0].set_facecolor('white')
                fig.subplots_adjust(bottom=0.3)
                fig.savefig(f"{self.outputPath}figures/module_barplot_eigengene_{moduleName}.{self.figureType}")
                if not show:
                    plt.close(fig)

            else:
                fig, axs = plt.subplots(nrows=1, ncols=len(metadata), figsize=(5 * len(metadata), 5))

                for i in range(len(metadata)):
                    df = ME.copy(deep=True)
                    df[metadata[i]] = sampleInfo[metadata[i]].values
                    palette = self.metadataColors[metadata[i]]
                    bar = sns.barplot(x=metadata[i], y="eigengeneExp", data=df, palette=palette, ci='sd', capsize=0.1,
                                      ax=axs[i])
                    if i != 0:
                        bar.set(ylabel=None)

                if self.save:
                    fig.savefig(f"{self.outputPath}figures/module_barplot_eigengene_{moduleName}.{self.figureType}")
                if not show:
                    plt.close(fig)
                else:
                    return axs

    def functional_enrichment_analysis(self, type, moduleName, sets=None, p_value=1, file_name=None, **kwargs):
        """
        Doing functional enrichment analysis including GO, KEGG and REACTOME

        :param type: indicate the type of databases which it should be one of "GO", "KEGG", "REACTOME"
        :type type: str
        :param moduleName: module name
        :type moduleName: str
        :param sets: str, list, tuple of Enrichr Library name(s). or custom defined gene_sets (dict, or gmt file) (you can add any Enrichr Libraries from here: https://maayanlab.cloud/Enrichr/#stats) only need to fill if the type is GO or KEGG
        :type sets: str, list, tuple
        :param p_value: Defines the pValue threshold. (default: 0.05)
        :type p_value: float
        :param file_name: name of the file you want to use to save plot (default is moduleName)
        :type file_name: str
        :param kwargs: Other keyword arguments are passed through to the underlying gseapy.enrichr() finction
        :type kwargs: key, value pairings
        """
        if type not in ["GO", "KEGG", "REACTOME"]:
            sys.exit("Type is not valid! it should be one of them GO, KEGG, REACTOME")

        if file_name is None:
            file_name = moduleName

        modules = np.unique(self.datExpr.var['moduleColors']).tolist()
        if np.all(moduleName not in modules):
            print(f"{WARNING}Module name does not exist in {ENDC}")
            return

        if not os.path.exists(f"{self.outputPath}figures/{type}"):
            print(f"{WARNING}{type} directory does not exist!\nCreating {type} directory!{ENDC}")
            os.makedirs(f"{self.outputPath}figures/{type}")

        if type == "GO" and sets is None:
            sets = ["GO_Biological_Process_2021"]
        elif type == "KEGG" and sets is None:
            sets = ["KEGG_2016"]

        geneModule = self.datExpr.var.gene_name[self.datExpr.var.moduleColors == moduleName]
        geneModule = geneModule.fillna("").values.tolist()

        if type in ["GO", "KEGG"]:
            try:
                from ..gseapy import enrichr
                enr = enrichr(gene_list=geneModule,
                                 gene_sets=sets,
                                 organism=self.species,
                                 outdir=f"{self.outputPath}figures/{type}/{file_name}",
                                 cutoff=p_value,
                                 **kwargs)
                dotplot(enr.res2d,
                        title=f"Gene ontology in {moduleName} module",
                        cmap='viridis_r',
                        cutoff=p_value,
                        ofname=f"{self.outputPath}figures/{type}/{file_name}.{self.figureType}")
            except:
                print(f"No enrich terms when cutoff = {p_value} in module {moduleName}")
        else:
            from reactome2py import analysis
            numGeneModule = len(geneModule)
            geneModule = ",".join(geneModule)
            result = analysis.identifiers(ids=geneModule,
                                          species=self.species,
                                          p_value=str(p_value))
            token = result['summary']['token']
            analysis.report(token,
                            path=f"{self.outputPath}figures/{type}/",
                            file=f"{file_name}.{self.figureType}",
                            number='50',
                            species=self.species)
            token_result = analysis.token(token,
                                          species=self.species,
                                          p_value=str(p_value))

            print(
                f"{numGeneModule - token_result['identifiersNotFound']} out of {numGeneModule} genes (identifiers) in the sample were found in Reactome.")
            print(
                f"{token_result['resourceSummary'][0]['pathways']} pathways were hit by at least one of them, which {len(token_result['pathways'])} of them have p-value less than {p_value}.")
            print(f"Report was saved {self.outputPath}figures/{type}/{file_name}.{self.figureType}!")
            print(f"For more information please visit https://reactome.org/PathwayBrowser/#/DTAB=AN&ANALYSIS={token}")

    def updateGeneInfo(self, geneInfo=None, path=None, sep=','):
        """
        add/update genes info in datExpr and geneExpr anndata

        :param geneInfo: gene information table you want to add to your data
        :type geneInfo: pandas dataframe
        :param path: path of geneInfo
        :type path: str
        :param sep: separation symbol to use for reading data in path properly (default: ",")
        :type sep: str
        """

        self.geneExpr = GeneExp.updateGeneInfo(self.geneExpr, geneInfo, path, sep)
        self.datExpr = GeneExp.updateGeneInfo(self.datExpr, geneInfo, path, sep)

    def updateSampleInfo(self, sampleInfo=None, path=None, sep=','):
        """
        add/update metadata in datExpr and geneExpr anndata

        :param sampleInfo: Sample information table you want to add to your data
        :type sampleInfo: pandas dataframe
        :param path: path of metaData
        :type path: str
        :param sep: separation symbol to use for reading data in path properly (default: ",")
        :type sep: str
        """

        self.geneExpr = GeneExp.updateSampleInfo(self.geneExpr, sampleInfo, path, sep)
        self.datExpr = GeneExp.updateSampleInfo(self.datExpr, sampleInfo, path, sep)

    @staticmethod
    def softConnectivity(datExpr, corOptions=pd.DataFrame(), weights=None, type="unsigned", power=6, blockSize=1500,
                         minNSamples=None):
        """
        Given expression data or a similarity, the function constructs the adjacency matrix and for each node calculates its connectivity, that is the sum of the adjacency to the other nodes.

        :param datExpr: a data frame containing the expression data, with rows corresponding to samples and columns to genes.
        :type datExpr: pandas dataframe
        :param corOptions: character string giving further options to be passed to the correlation function.
        :type corOptions: pandas dataframe
        :param weights: optional observation weights for datExpr to be used in correlation calculation. A matrix of the same dimensions as datExpr, containing non-negative weights. Only used with Pearson correlation.
        :type weights: pandas dataframe
        :param type: network type. Allowed values are (unique abbreviations of) "unsigned", "signed", "signed hybrid".
        :type type: str
        :param power: soft thresholding power.
        :type power: int
        :param blockSize: block size in which adjacency is to be calculated. Too low (say below 100) may make the calculation inefficient, while too high may cause R to run out of physical memory and slow down the computer. Should be chosen such that an array of doubles of size (number of genes) * (block size) fits into available physical memory.
        :type blockSize: int
        :param minNSamples: minimum number of samples available for the calculation of adjacency for the adjacency to be considered valid. If not given, defaults to the greater of ..minNSamples (currently 4) and number of samples divided by 3. If the number of samples falls below this threshold, the connectivity of the corresponding gene will be returned as NA.
        :type minNSamples: int

        :return: A list with one entry per gene giving the connectivity of each gene in the weighted network.
        :rtype: ndarray
        """
        try:
            import resource
        except ImportError:
            try:
                import rsrc as resource
            except ImportError:
                sys.exit("resource or rsrc package is not installed!")
        if type == "signed":
            power = 15

        nGenes = datExpr.shape[1]
        nSamples = datExpr.shape[0]

        if blockSize * nGenes > resource.getrlimit(resource.RLIMIT_AS)[1]:
            blockSize = int(resource.getrlimit(resource.RLIMIT_AS)[1] / nGenes)

        if minNSamples is None:
            minNSamples = max(4, nSamples / 3)

        if nGenes < 10 or nSamples < minNSamples:
            sys.exit(
                f"Error: Something seems to be wrong. \n Make sure that the input data frame has genes as rows and "
                f"array samples as columns.\n Alternatively, there seem to be fewer than 10 genes or fewer than "
                f"{minNSamples} samples.")
        if nGenes < nSamples:
            print(
                f"{WARNING}Warning: There are fewer genes than samples in the function softConnectivity. Maybe you "
                f"should transpose the data?{ENDC}")

        k = np.zeros((nGenes,))
        start = 1
        while start < nGenes:
            end = min(start + blockSize - 1, nGenes)
            index1 = range(start, end)
            ad1 = pyWGCNA.adjacency(datExpr, weights=weights, selectCols=index1, power=power, adjacencyType=type,
                                  corOptions=corOptions)
            k[index1] = ad1.sum(axis=1) - 1
            # If fewer than minNSamples contain gene expression information for a given
            # gene, then we set its connectivity to 0.
            NoSamplesAvailable = datExpr.iloc[:, index1].notna().sum(axis=1)
            k[index1][NoSamplesAvailable < minNSamples] = np.nan
            start = end + 1
        return k

    @staticmethod
    def intramodularConnectivity(mat, colors, scaleByMax=False, index=None):
        """
        Calculates intramodular connectivity, i.e., connectivity of nodes to other nodes within the same module.

        :param mat: adjacency which should be a square, symmetric matrix with entries between 0 and 1.
        :type mat: ndarray
        :param colors: module labels. A list of length ncol(adjMat) giving a module label for each gene (node) of the network.
        :type colors: list
        :param scaleByMax: should intramodular connectivity be scaled by the maximum IM connectivity in each module?
        :type scaleByMax: bool
        :param index: gene id or name of mat index
        :type index: ndarray

        :return: If input getWholeNetworkConnectivity is TRUE, a data frame with 4 columns giving the total connectivity, intramodular connectivity, extra-modular connectivity, and the difference of the intra- and extra-modular connectivities for all genes; otherwise a vector of intramodular connectivities
        :rtype: pandas dataframe

        """
        if mat.shape[0] != mat.shape[1]:
            sys.exit("input matrix is not a square matrix.")

        if mat.shape[0] != len(colors):
            sys.exit("Dimensions of matrix (number of genes) and length of 'colors' differ.")

        nNodes = len(colors)
        colorLevels = pd.Categorical(colors).categories
        nLevels = len(colorLevels)
        kWithin = np.zeros((nNodes,))

        np.fill_diagonal(mat, 0)
        mat = np.nan_to_num(mat)
        mat = pd.DataFrame(mat)
        for i in range(nLevels):
            rest1 = np.where(colors == colorLevels[i])[0].tolist()
            if len(rest1) < 3:
                kWithin[colors == colorLevels[i]] = 0
            else:
                kWithin[rest1] = mat.iloc[rest1, rest1].sum(axis=1)
                if scaleByMax:
                    kWithin[rest1] = kWithin[rest1] / max(kWithin[rest1])
        kTotal = mat.sum(axis=1)  # apply(adjMat, 2, sum, na.rm = TRUE)
        kOut = np.subtract(kTotal, kWithin)
        if scaleByMax:
            kOut = np.zeros((nNodes,))
        kDiff = np.subtract(kWithin, kOut)
        res = pd.DataFrame({'kTotal': kTotal, 'kWithin': kWithin, 'kOut': kOut, 'kDiff': kDiff})
        if index is not None:
            res.index = index
        return res

    def CalculateSignedKME(self, exprWeights=None, MEWeights=None):
        """
        Calculation of (signed) eigengene-based connectivity, also known as module membership.

        :param exprWeights: optional weight matrix of observation weights for datExpr, of the same dimensions as datExpr
        :type exprWeights: pandas dataframe
        :param MEWeights: optional weight matrix of observation weights for datME, of the same dimensions as datME
        :type MEWeights: pandas dataframe

        :return: A data frame in which rows correspond to input genes and columns to module eigengenes, giving the signed eigengene-based connectivity of each gene with respect to each eigengene.
        :rtype: pandas dataframe
        """
        corOptions = []
        if self.datME.shape[0] != self.datExpr.to_df().shape[0]:
            sys.exit("Number of samples (rows) in 'datExpr' and 'datME' must be the same.")
        datExpr = self.datExpr.to_df()
        datME = self.datME
        if exprWeights is not None:
            exprWeights = pyWGCNA.checkAndScaleWeights(exprWeights, datExpr, scaleByMax=False)
        if MEWeights is not None:
            MEWeights = pyWGCNA.checkAndScaleWeights(exprWeights, datME, scaleByMax=False)

        varianceZeroIndicatordatExpr = sum(datExpr.var(axis=0) == 0)
        varianceZeroIndicatordatME = sum(datME.var(axis=0) == 0)
        if varianceZeroIndicatordatExpr > 0:
            print(f"{WARNING}Some genes are constant. Hint: consider removing constant columns from datExpr.{ENDC}")
        if varianceZeroIndicatordatME > 0:
            print(f"{WARNING}Some module eigengenes are constant, which is suspicious. Hint: consider removing "
                  f"constant columns from datME.{ENDC}")
        if MEWeights is not None:
            corOptions.append("weights.y = MEWeights, ")
        if exprWeights is not None:
            corOptions.append("weights.x = exprWeights, ")

        output = np.corrcoef(datExpr.T, datME.T)
        tmp = len(datME.columns)
        col = ["k" + datME.columns[i] for i in range(tmp)]
        self.signedKME = pd.DataFrame(output[0:datExpr.shape[1], datExpr.shape[1]:],
                                      index=datExpr.columns, columns=col)

    def CoexpressionModulePlot(self, modules, numGenes=10, numConnections=100, minTOM=0, filters=None, file_name=None):
        """
        plot Coexpression for given module

        :param modules: name of modules you like to plot
        :type modules: list of str
        :param numGenes: number of genes you want to show for each module
        :type numGenes: int
        :param numConnections: number of connection you want to show for each module
        :type numConnections: int
        :param minTOM: minimum TOM to keep connections
        :type minTOM: float
        :param filters: Dictionary which keys are columns names of datExpr.var that you want to filter the genes based on it and values are determining which rows you want to keep
        :type filters: dict
        :param file_name: name of the html output file (default: module names or network if there is more than 3 modules for input)
        :type file_name: str

        :return: save a html file with name of modules in figures directory
        """
        from pyvis.network import Network
        if self.signedKME is None:
            print("signedKME is empty! call signedKME() to calculate it")

        if not os.path.exists(f"{self.outputPath}figures/network/"):
            print(f"{WARNING}Network directory does not exist!\nCreating network directory!{ENDC}")
            os.makedirs(f"{self.outputPath}figures/network/")

        gene_id = self.datExpr.var.loc[self.datExpr.var.moduleColors.isin(modules), :]
        if gene_id.shape[0] == 0:
            sys.exit(f"There is no gene assign to {','.join(modules)} module(s)!")
        if filters is not None:
            for key in filters.keys():
                gene_id = gene_id.loc[gene_id[key].isin(filters[key]), :]
        if gene_id.shape[0] < numGenes:
            numGenes = gene_id.shape[0]
            numConnections = numGenes * (numGenes - 1)
        genes = []
        for module in modules:
            mat = self.signedKME.loc[gene_id.index].sort_values(["kME" + module], ascending=False)
            mat = mat.iloc[:numGenes, :]
            genes = genes + mat.index.tolist()

        adj = self.TOM.loc[genes, genes]
        adj[adj < minTOM] = 0
        adj.where(np.triu(np.ones(adj.shape)).astype(bool), inplace=True)
        adj.where(adj.values != np.diag(adj), 0, inplace=True)
        adj.where(adj.values != np.flipud(adj).diagonal(0), 0, inplace=True)
        adj = adj.stack().nlargest(numConnections)

        net = Network()
        gene_id = list(adj.index.get_level_values(0)) + list(adj.index.get_level_values(1))
        gene_id = np.unique(gene_id)
        nodes = self.datExpr.var.loc[gene_id, :]
        node_color = nodes["moduleColors"].tolist()
        nodes.drop(["dynamicColors", "moduleColors", "moduleLabels"], axis=1, inplace=True)
        title = nodes.index.tolist()
        for i in range(nodes.shape[0]):
            for j in range(nodes.shape[1]):
                if not pd.isna(nodes.iloc[i, j]):
                    title[i] = title[i] + "\n" + str(nodes.columns[j]) + ": " + str(nodes.iloc[i, j])

        net.add_nodes(nodes.index.tolist(),
                      title=title,
                      label=nodes.index.tolist(),
                      color=node_color)

        for i in range(len(adj)):
            if adj[i] != 0:
                net.add_edge(adj.index.get_level_values(0)[i],
                             adj.index.get_level_values(1)[i],
                             weight=adj[i])

        if file_name is None:
            if len(modules) <= 3:
                file_name = '_'.join(modules)
            else:
                file_name = "network"

        net.show(f"{self.outputPath}figures/network/{file_name}.html")

    def PPI_network(self, species, moduleName=None, geneList=None, output_format="image"):
        """
        retrieve an image of a STRING network of a neighborhood surrounding one or more proteins or ask STRING to show only the network of interactions between your input proteins.

        :param species: NCBI taxon identifiers (e.g. Human is 9606, see: https://string-db.org/cgi/input.pl?input_page_active_form=organisms).
        :type species: int
        :param moduleName: name of module you want to find PPI
        :type moduleName: str
        :param geneList: list of genes you want to find PPI
        :type geneList: list
        :param output_format: format of output which can be "image", "highres_image", "svg" (default: "image")
        :type output_format: str

        :return: dataframe contains genes along with interaction with their scores
        :rtype: pandas dataframe
        """

        if moduleName is not None:
            genes = self.datExpr.var[self.datExpr.var.moduleColors == moduleName]
            geneList = genes.index.astype(str).tolist()
        elif geneList is None:
            sys.exit("geneList or moduleName should be empty at the same time!")

        if not os.path.exists(f"{self.outputPath}figures/PPI/"):
            print(f"{WARNING}PPI directory does not exist!\nCreating PPI directory!{ENDC}")
            os.makedirs(f"{self.outputPath}figures/PPI/")

        res = pyWGCNA.request_PPI(genes=geneList, species=species)
        res.sort_values(by=["gene1"], axis=0, inplace=True)

        string_api_url = "https://version-11-5.string-db.org/api"
        method = "network"

        request_url = "/".join([string_api_url, output_format, method])

        for gene in res.gene1.unique():
            tmp = res[res.gene1 == gene]
            genes = tmp.gene2.tolist()
            genes.append(tmp.gene1.tolist()[0])
            params = {
                "identifiers": "\r".join(genes),  # your protein
                "species": species,  # species NCBI identifier
                "add_white_nodes": 15,  # add 15 white nodes to my protein
                "network_flavor": "confidence",  # show confidence links
                "caller_identity": "www.awesome_app.org"  # your app name

            }
            pyWGCNA.request_PPI_image(params=params,
                                    genes=genes,
                                    file_name=f"{self.outputPath}figures/PPI/{tmp.gene1.tolist()[0]}_PPI.png",
                                    request_url=request_url)
        return res

    @staticmethod
    def request_PPI_subset(params,
                           request_url='https://version-11-5.string-db.org/api/tsv-no-header/interaction_partners'):
        """
        request STRING to find genes interact with our gene list base

        :param request_url: suitable url for using STRING API
        :type request_url: str
        :param params: parameters for requesting
        :type params: dict

        :return: dataframe contains genes interact with each other
        :rtype: pandas dataframe
        """
        response = requests.post(request_url, data=params)

        res = pd.DataFrame(columns=["gene1", "gene2", "score"])
        if response.status_code == 200:
            for line in response.text.strip().split("\n"):
                l = line.strip().split("\t")

                tmp = pd.DataFrame({'gene1': l[2], 'gene2': l[3], 'score': l[5]}, index=[0])
                res = pd.concat([res, tmp], ignore_index=True)

        sleep(1)

        return res

    @staticmethod
    def request_PPI(genes,
                    species):
        """
        Getting all the STRING interaction partners of the protein set

        :param genes: list of genes you want to find interaction for
        :type genes: list
        :param species: NCBI taxon identifiers (e.g. Human is 9606, see: https://string-db.org/cgi/input.pl?input_page_active_form=organisms).
        :type species: int

        :return: dataframe contains genes interact with each other
        :rtype: pandas dataframe
        """

        string_api_url = "https://version-11-5.string-db.org/api"
        output_format = "tsv-no-header"
        method = "interaction_partners"

        request_url = "/".join([string_api_url, output_format, method])

        count = 0
        res = pd.DataFrame(columns=["gene1", "gene2", "score"])
        while count < len(genes):

            if count + 2000 > len(genes):
                my_genes = genes[count:]
            else:
                my_genes = genes[count:count + 2000]

            params = {
                "identifiers": "%0d".join(my_genes),  # your protein
                "species": species,  # species NCBI identifier
                "limit": 5,
                "caller_identity": "www.awesome_app.org"  # your app name
            }

            tmp = pyWGCNA.request_PPI_subset(params, request_url)
            count = count + 2000
            res = pd.concat([res, tmp], ignore_index=True)

        print(f"{len(res.gene1.unique().tolist())} genes out of {len(genes)} genes have PPI.")
        return res

    @staticmethod
    def request_PPI_image(params,
                          genes,
                          file_name,
                          request_url='https://version-11-5.string-db.org/api/image/network'):
        """
        plot PPI interaction along with link that direct you to the STRING webpage

        :param params: parameters for requesting
        :type params: dict
        :param genes: list of genes you want to find interaction for
        :type genes: list
        :param file_name: name of the output file
        :type file_name: str
        :param request_url: suitable url for using STRING API
        :type request_url: str

        """
        response = requests.post(request_url, data=params)

        if response.status_code == 200:
            print("Saving interaction network to %s" % file_name)

            with open(file_name, 'wb') as fh:
                fh.write(response.content)

            print("For having interactive network, click line below:")
            tmp = "%0d".join(genes)
            print(f"https://string-db.org/api/tsv/get_link?identifiers={tmp}&species={params['species']}")

        sleep(1)

    def top_n_hub_genes(self, moduleName, n=10):
        """
        find top n hub genes based on connectivity in given module

        :param moduleName: name of module you want to top n hub genes
        :type moduleName: str
        :param n: number of top hub genes
        :type n: int

        :return: dataframe contains top n hun genes along with connectivity score and additional gene information you added to your expression matrix
        :rtype: pandas dataframe
        """

        modules = np.unique(self.datExpr.var['moduleColors']).tolist()
        if np.all(moduleName not in modules):
            print(f"{WARNING}Module name does not exist in {ENDC}")
            return

        datExpr = self.datExpr.copy()
        datExpr = datExpr[:, datExpr.var['moduleColors'] == moduleName]
        adj = pyWGCNA.adjacency(datExpr.to_df(), power=self.power, adjacencyType=self.networkType)
        adj = pd.DataFrame(adj,
                           columns=datExpr.to_df().columns,
                           index=datExpr.to_df().columns)
        hub_genes = adj.sum().sort_values(ascending=False)
        hub_genes = pd.DataFrame(hub_genes,
                                 columns=["connectivity"])

        hub_genes = pd.concat([hub_genes, datExpr.var], axis=1)
        hub_genes = hub_genes.iloc[:n, :]

        return hub_genes

import matplotlib.colors as mcolors
import pandas as pd
import itertools

def create_custom_color_seq(colors, naColor):
    # Convert colors to DataFrame
    colors_df = pd.DataFrame({"name": list(colors.keys()),
                              "color": list(colors.values())})
    colors_df.drop_duplicates(subset='color', keep="last", inplace=True)
    
    # Sort colors by hue, saturation, value and name
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors_df.itertuples(index=False))
    
    # Create a dictionary to hold color groups
    grouped_colors = {}
    
    # Group colors by hue
    for hsv, name in by_hsv:
        hue = hsv[0]
        if hue not in grouped_colors:
            grouped_colors[hue] = []
        grouped_colors[hue].append(name)
    
    # Remove naColor from all groups
    for hue in grouped_colors:
        if naColor in grouped_colors[hue]:
            grouped_colors[hue].remove(naColor)
    
    # Create the custom color sequence
    colorSeq = []
    max_len = max(len(group) for group in grouped_colors.values())
    for i in range(max_len):
        for hue in sorted(grouped_colors.keys()):
            if i < len(grouped_colors[hue]):
                colorSeq.append(grouped_colors[hue][i])
    
    return colorSeq

def interleave_colors(color_list):
    # Convert the color list to DataFrame to get HSV values
    colors_df = pd.DataFrame({"name": color_list})
    colors_df["color"] = colors_df["name"].apply(lambda name: mcolors.CSS4_COLORS[name])
    colors_df["hsv"] = colors_df["color"].apply(lambda color: mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3]))

    # Sort colors by hue
    colors_df = colors_df.sort_values(by="hsv", key=lambda x: x.apply(lambda hsv: hsv[0]))
    #return colors_df
    # Split the sorted colors into several groups
    num_groups = 12
    grouped_colors = [colors_df.iloc[i::num_groups] for i in range(num_groups)]
    #print(colors_df)

    # Interleave the colors from different groups
    interleaved_colors = []
    for group in grouped_colors:
        #for color in group:
        #print(group)
        #print(group['color'])
        interleaved_colors+=group['name'].tolist()
    
    return interleaved_colors


"""
__author__ = "Peizhuo Wang"
__email__ = "wangpeizhuo_37@163.com"
__citation__ = Wang, P., Wen, X., Li, H. et al. Deciphering driver regulators of cell fate decisions from single-cell transcriptomics data with CEFCON. Nat Commun 14, 8459 (2023). https://doi.org/10.1038/s41467-023-44103-3
"""


from os import fspath
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from itertools import permutations, product
from typing import Optional
import requests
import os
import scanpy as sc
import zipfile
from .._registry import register_function


def _get_cefcon_backend():
    from ..external.CEFCON.cell_lineage_GRN import NetModel
    from ..external.CEFCON.utils import data_preparation

    return NetModel, data_preparation

biomart_install = False

@register_function(
    aliases=["小鼠造血干细胞数据集", "mouse_hsc", "nestorowa16", "造血数据", "HSC数据集"],
    category="single",
    description="Load mouse hematopoietic stem cell dataset from Nestorowa et al. (2016) with lineage information for CEFCON analysis",
    examples=[
        "# Load mouse HSC dataset (default v0)",
        "adata = ov.single.mouse_hsc_nestorowa16()",
        "# Load specific version",
        "adata = ov.single.mouse_hsc_nestorowa16(version='v0')",
        "# Custom file path",
        "adata = ov.single.mouse_hsc_nestorowa16(fpath='./my_data/hsc_data.h5ad')"
    ],
    related=["single.pyCEFCON", "single.load_human_prior_interaction_network", "pp.preprocess"]
)
def mouse_hsc_nestorowa16(fpath: Optional[str] = './data_cache/mouse_hsc_nestorowa16_v0.h5ad', version: Optional[str] = 'v0'):
    """Load Nestorowa16 mouse HSC reference data used by CEFCON.

    Parameters
    ----------
    fpath : str, optional
        Local cache path for the dataset file.
    version : {'v0', 'v1'}, default='v0'
        Dataset version tag.

    Returns
    -------
    AnnData
        Mouse HSC AnnData object.
    """
    if version=='v0':
        fpath = './data_cache/mouse_hsc_nestorowa16_v0.h5ad'
        url = 'https://zenodo.org/record/8013900/files/mouse_hsc_nestorowa16_v0.h5ad'
        print('Load mouse_hsc_nestorowa16_v0.h5ad')
    elif version=='v1':
        fpath = './data_cache/mouse_hsc_nestorowa16_v1.h5ad'
        url = 'https://zenodo.org/record/8013900/files/mouse_hsc_nestorowa16_v1.h5ad'
        print('Load mouse_hsc_nestorowa16_v1.h5ad')
    else:
        print('Wrong data!')
    adata = sc.read(fpath, backup_url=url, sparse=True, cache=True)
    adata.var_names_make_unique()
    return adata


def _download_from_url(file_url: str, save_path: Path):
    try:
        response = requests.get(file_url, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f'Download error: {e}')
        return

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

    download_file = save_path.parent / file_url.split('/')[-1]
    with open(download_file, 'wb') as file:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))

    progress_bar.close()

    if str(download_file).endswith('.zip'):
        with zipfile.ZipFile(download_file, 'r') as zip_file:
            zip_file.extractall(os.path.dirname(save_path))
            zip_file.close()
        os.remove(download_file)

    print(f'Ths data has been downloaded to `{save_path}`.')


@register_function(
    aliases=["人类先验网络", "load_prior_network", "human_network", "基因调控网络", "prior_interaction_network"],
    category="single",
    description="Load human prior gene interaction network datasets for CEFCON analysis",
    examples=[
        "# Load default NicheNet network",
        "network = ov.single.load_human_prior_interaction_network()",
        "# Load PathwayCommons network",
        "network = ov.single.load_human_prior_interaction_network(dataset='pathwaycommons')",
        "# Load InBioMap network",
        "network = ov.single.load_human_prior_interaction_network(dataset='inbiomap')",
        "# Force download even if file exists",
        "network = ov.single.load_human_prior_interaction_network(force_download=True)"
    ],
    related=["single.convert_human_to_mouse_network", "single.pyCEFCON", "single.mouse_hsc_nestorowa16"]
)
def load_human_prior_interaction_network(dataset: str = 'nichenet',
                                         only_directed: bool = False,
                                         force_download: bool = False):
    """Load one of the packaged human prior interaction networks.

    Parameters
    ----------
    dataset : str, default='nichenet'
        Network source name.
    only_directed : bool, default=False
        Keep only directed edges when edge-direction information exists.
    force_download : bool, default=False
        Force redownload even when local cache exists.

    Returns
    -------
    pandas.DataFrame
        Two-column edge table with ``from`` and ``to`` gene symbols.
    """

    # The URL for every dataset. These datasets are stored at zenodo (https://doi.org/10.5281/zenodo.7564872).
    urls = {
        'nichenet': 'https://zenodo.org/record/8013900/files/NicheNet_human.zip',
        'pathwaycommons': 'https://zenodo.org/record/8013900/files/PathwayCommons12.All.hgnc.zip',
        'inbiomap': 'https://zenodo.org/record/8013900/files/InBioMap.zip',
        'harmonizome': 'https://zenodo.org/record/8013900/files/Harmonizome_nichenet.zip',
        'omnipath_interactions': 'https://zenodo.org/record/8013900/files/Omnipath_interaction.zip',
    }
    filenames = {
        'nichenet': 'NicheNet_human.csv',
        'pathwaycommons': 'PathwayCommons12.All.hgnc.sif',
        'inbiomap': 'InBioMap.csv',
        'harmonizome': 'Harmonizome_nichenet.csv',
        'omnipath_interactions': 'Omnipath_interaction.csv',
    }

    # Download if the file does not exist
    data_path = Path('./data_cache') / filenames[dataset]
    if force_download or not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        _download_from_url(urls[dataset], data_path)

    if dataset == 'nichenet':  # 5,583,023
        prior_net = pd.read_csv(data_path, index_col=None, header=0)

    elif dataset == 'pathwaycommons':  # 1,200,159
        prior_net = pd.read_csv(data_path, sep='\t',
                                names=['from', 'type', 'to'])
        undirected_type = ['interacts-with', 'in-complex-with']
        type_mapper = dict({v: 1 for v in list(set(prior_net.type.unique()) - set(undirected_type))},
                           **{v: 0 for v in undirected_type})
        prior_net['is_directed'] = prior_net['type'].map(type_mapper)
        prior_net = prior_net.loc[~prior_net['from'].str.startswith('CHEBI'), :]
        prior_net = prior_net.loc[~prior_net['to'].str.startswith('CHEBI'), :]
        del prior_net['type']

    elif dataset == 'inbiomap':  # 625,641
        prior_net = pd.read_csv(data_path)
        prior_net.rename(columns={'genesymbol_a': 'from', 'genesymbol_b': 'to'}, inplace=True)
        prior_net = prior_net.loc[:, ['from', 'to']]

    elif dataset == 'harmonizome':  # 3,418,949
        prior_net = pd.read_csv(data_path, sep='\t')
        prior_net = prior_net.loc[:, ['from', 'to']]

    elif dataset == 'omnipath_interactions':  # 525,430
        prior_net = pd.read_csv(data_path)
        prior_net.rename(columns={'source_genesymbol': 'from', 'target_genesymbol': 'to'}, inplace=True)
        complex_idx = prior_net['source'].str.startswith('COMPLEX') | \
                      prior_net['target'].str.startswith('COMPLEX')
        prior_net_genes_only = prior_net.loc[~complex_idx, ['from', 'to', 'is_directed']]
        prior_net_genes_complex = prior_net.loc[complex_idx, ['from', 'to', 'is_directed']]
        # process complex items
        temp_edge = []
        temp_edge_type = []
        for source, target in zip(prior_net_genes_complex['from'], prior_net_genes_complex['to']):
            source = source.split('_')
            target = target.split('_')
            # inside the complex
            intra = list(permutations(source, r=2)) + list(permutations(target, r=2))
            temp_edge += intra
            temp_edge_type += [0] * len(intra)
            # between two complexes
            inter = list(product(source, target))
            temp_edge += inter
            temp_edge_type += [1] * len(inter)
        prior_net_complex = pd.DataFrame(temp_edge, columns=['from', 'to'])
        prior_net_complex['is_directed'] = temp_edge_type
        prior_net = pd.concat([prior_net_genes_only, prior_net_complex], axis=0)
        prior_net = prior_net.dropna()
        prior_net = prior_net.drop_duplicates()

    else:
        print(f"Value error. {dataset} is not available.")
        print("Available option: {'nichenet', 'pathwaycommons', 'inbiomap', 'harmonizome', 'omnipath_interactions'}")

    if ('is_directed' in prior_net) and only_directed:
        prior_net = prior_net[prior_net['is_directed'] == 1]

    prior_net = prior_net[['from', 'to']].drop_duplicates().astype(str)
    print(f"Load the prior gene interaction network: {dataset}. "
          f"#Genes: {len(np.unique(prior_net.iloc[:, [0, 1]]))}, #Edges: {len(prior_net)}")

    return prior_net


@register_function(
    aliases=["人类小鼠网络转换", "convert_network", "human_to_mouse", "基因符号转换", "species_conversion"],
    category="single",
    description="Convert human gene symbols in prior interaction network to mouse gene symbols using biomart",
    examples=[
        "# Convert human network to mouse with default server",
        "mouse_network = ov.single.convert_human_to_mouse_network(human_network)",
        "# Use European server",
        "mouse_network = ov.single.convert_human_to_mouse_network(human_network, server_name='europe')",
        "# Use US server",
        "mouse_network = ov.single.convert_human_to_mouse_network(human_network, server_name='useast')"
    ],
    related=["single.load_human_prior_interaction_network", "single.pyCEFCON", "single.mouse_hsc_nestorowa16"]
)
def convert_human_to_mouse_network(net: pd.DataFrame):
    """Convert a human-symbol interaction network to mouse symbols.

    Parameters
    ----------
    net : pd.DataFrame
        Human network edge table with ``from`` and ``to`` columns.
    server_name : str, default='asia'
        Ensembl Biomart server preference.

    Returns
    -------
    pandas.DataFrame
        Converted mouse interaction edge table.
    """
    global gseapy_install
    try:
        from gseapy import Biomart
        gseapy_install=True
    except ImportError:
        raise ImportError(
            'Please install the gseapy: `pip install -U gseapy`.'
            )


    print('Convert genes of the prior interaction network to mouse gene symbols:')
    with tqdm(total=10, desc='Processing', miniters=1) as outer_bar:
        outer_bar.update()

        bm = None
        try:
            bm = Biomart()
            print(f'Server is OK')
        except Exception as e:
            print(f"Connect failed!")

        human_dataset_name = 'hsapiens_gene_ensembl'
        mouse_dataset_name = 'mmusculus_gene_ensembl'
        outer_bar.update()

        human_attributes = ['ensembl_gene_id', 'hgnc_symbol']
        mouse_attributes = ['ensembl_gene_id', 'mgi_symbol']
        to_homolog_attribute = 'mmusculus_homolog_ensembl_gene'

        # ========== 第一段查询：人 Ensembl ID + symbol
        df_human = bm.query(
            dataset=human_dataset_name,
            attributes=human_attributes
        )
        df_human.columns = ['human_ensembl_id', 'hgnc_symbol']
        outer_bar.update(2)

        # ========== 第二段查询：人ID → 小鼠同源ID
        df_homolog = bm.query(
            dataset=human_dataset_name,
            attributes=['ensembl_gene_id', to_homolog_attribute]
        )
        df_homolog.columns = ['human_ensembl_id', 'mouse_ensembl_id']
        from2to = df_homolog.merge(df_human, how='outer', on='human_ensembl_id')
        outer_bar.update()

        # ========== 第三段查询：小鼠Ensembl ID → mgi_symbol
        df_mouse = bm.query(
            dataset=mouse_dataset_name,
            attributes=mouse_attributes
        )
        df_mouse.columns = ['mouse_ensembl_id', 'mgi_symbol']
        target = df_mouse.merge(from2to, how='outer', on='mouse_ensembl_id')
        outer_bar.update(2)

        # Gene mapper of the network
        query_genes = np.unique(net.loc[:, ['from', 'to']].astype(str))
        mapper = target.loc[target['hgnc_symbol'].isin(query_genes), ['hgnc_symbol', 'mgi_symbol']].copy()
        mapper = mapper.dropna()
        mapper = mapper.drop_duplicates()
        mapper = mapper.loc[mapper['mgi_symbol'] != '', :]

        # Process ambiguous (1-to-many) and unambiguous (1-to-1 and many-to-1) genes separately
        human_gene_value_counts = mapper.loc[:, 'hgnc_symbol'].value_counts()
        unambiguous_genes = human_gene_value_counts[human_gene_value_counts == 1].index.tolist()
        ambiguous_genes = human_gene_value_counts[human_gene_value_counts > 1].index.tolist()
        outer_bar.update()

        # Directly convert the interactions with unambiguous genes
        net_una = net.loc[net['from'].isin(unambiguous_genes) & net['to'].isin(unambiguous_genes), ['from', 'to']]
        converted_network_unambiguous = pd.DataFrame()
        mapper_una_dict = mapper.loc[mapper['hgnc_symbol'].isin(unambiguous_genes), :].set_index(['hgnc_symbol'])[
            'mgi_symbol'].to_dict()
        converted_network_unambiguous['from'] = net_una['from'].map(mapper_una_dict)
        converted_network_unambiguous['to'] = net_una['to'].map(mapper_una_dict)
        outer_bar.update()

        # Process interactions where one gene is ambiguous and another is unambiguous
        net_a = net.loc[(net['from'].isin(ambiguous_genes) & net['to'].isin(unambiguous_genes)) |
                        (net['to'].isin(ambiguous_genes) & net['from'].isin(unambiguous_genes)), ['from', 'to']]
        mapper_a = mapper.loc[mapper['hgnc_symbol'].isin(ambiguous_genes), :]
        temp_edge = []
        with tqdm(total=len(net_a),
                  desc='Converting ambiguous gene symbols',
                  leave=False,
                  miniters=1,
                  ) as pbar:
            for source, target in zip(net_a['from'], net_a['to']):
                if source in unambiguous_genes:
                    source_convert = [mapper_una_dict[source]]
                    target_convert = mapper_a[mapper_a['hgnc_symbol'] == target]['mgi_symbol'].tolist()
                else:
                    source_convert = mapper_a[mapper_a['hgnc_symbol'] == source]['mgi_symbol'].tolist()
                    target_convert = [mapper_una_dict[target]]
                temp_edge += list(product(source_convert, target_convert))
                pbar.update(1)
        converted_network_ambiguous = pd.DataFrame(temp_edge, columns=['from', 'to'])
        outer_bar.update()

    # Combine the converted network
    prior_net_converted = pd.concat([converted_network_unambiguous, converted_network_ambiguous], axis=0)
    prior_net_converted = prior_net_converted.drop_duplicates()

    print(f"The converted prior gene interaction network: "
          f"#Genes: {len(np.unique(prior_net_converted.iloc[:, [0, 1]]))}, "
          f"#Edges: {len(prior_net_converted)}")

    return prior_net_converted

@register_function(
    aliases=["CEFCON分析", "pyCEFCON", "driver_regulators", "细胞命运调控", "gene_regulatory_network"],
    category="single",
    description="CEFCON: Computational tool for deciphering driver regulators of cell fate decisions from single-cell RNA-seq data",
    examples=[
        "# Initialize CEFCON with basic parameters",
        "cefcon = ov.single.pyCEFCON(adata, prior_network)",
        "# CEFCON with custom parameters and GUROBI solver",
        "cefcon = ov.single.pyCEFCON(adata, prior_network, repeats=5, solver='GUROBI')",
        "# Run complete CEFCON analysis",
        "cefcon = ov.single.pyCEFCON(adata, network)",
        "cefcon.preprocess()",
        "cefcon.train()",
        "cefcon.predicted_driver_regulators()",
        "cefcon.predicted_RGM()"
    ],
    related=["single.mouse_hsc_nestorowa16", "single.load_human_prior_interaction_network", "single.convert_human_to_mouse_network"]
)
class pyCEFCON(object):
    """CEFCON workflow wrapper for driver-regulator discovery.

    Parameters
    ----------
    input_expData : str or AnnData or pd.DataFrame
        Input expression data with optional lineage metadata.
    input_priorNet : str or pd.DataFrame
        Prior interaction network.
    input_genesDE : str or pd.DataFrame, optional
        Differential-expression score table.
    repeats : int, default=5
        Number of model repeats.
    solver : {'GUROBI', 'SCIP'}, default='GUROBI'
        ILP solver used for driver-regulator selection.
    """

    def __init__(self,
             # New arguments
             input_expData,
             input_priorNet,
             input_genesDE=None, # genesDE
             additional_edges_pct=0.01,
             cuda=0,# -1: cpu; 0: gpu ; 1,2,3 ...: specify gpu
             seed=2023,
             hidden_dim=128,
             output_dim=64,
             heads=4,
             attention='COS',
             miu=0.5,
             epochs=350,
             repeats=5,
             edge_threshold_param=8,
             remove_self_loops=False,
             topK_drivers=100,
            solver = 'GUROBI',
        #     out_dir='./output'
            ):
        """
        Parameters
        ----------
            input_expData (str or sc.AnnData or pd.DataFrame): input gene expression data. It can be the path to a csv file, an AnnData object, or a pandas dataframe. If the input is an AnnData object, the lineage name must be contained in AnnData.uns['lineages'], and the lineage information (can be the pseudotime, where non-NA data denotes cells in the lineage) must be contained in AnnData.obs. If no lineage information is detected, all cell expressions will be regarded as one lineage, which will be named 'all' by default.
            input_priorNet (str or pd.DataFrame): input prior gene interaction network. It can be the path to a csv file or a pandas dataframe
            input_genesDE (str or pd.DataFrame): input gene differential expression score. It can be the path to a csv file or a pandas dataframe
            additional_edges_pct (float, optional): proportion of high co-expression interactions to be added (default: 0.01)
            cuda (int, optional): an integer greater than -1 indicates the GPU device number and -1 indicates the CPU device
            seed (int, optional): random seed (set to -1 means no random seed is assigned)
            hidden_dim (int, optional): hidden dimension of the GNN encoder (default: 128)
            output_dim (int, optional): output dimension of the GNN encoder (default: 64)
            heads (int, optional): number of heads for the multi-head attention (default: 4)
            attention (str, optional): type of attention scoring function ('COS', 'SD', 'AD') (default: 'COS')
            miu (float, optional): parameter (0~1) for considering the importance of attention coefficients of the first GNN layer (default: 0.5)
            epochs (int, optional): number of epochs for one repeat (default: 350)
            repeats (int, optional): number of repeats (default: 5)
            edge_threshold_param (int, optional): threshold for selecting top-weighted edges (larger values mean more edges). This parameter corresponds to the average degree of the constructed GRN (default: 8)
            remove_self_loops (bool, optional): whether to remove all self-loops (default: True)
            topK_drivers (int, optional): number of top-ranked candidate driver genes according to their influence scores (default: 100)
            solver (str, optional): Solver ('GUROBI', 'SCIP') for solving the integer linear programming problems (for identifying drive regulators) (default: 'GUROBI')

        Returns
        -------
        None
        """   
        self.input_expData = input_expData
        self.input_priorNet = input_priorNet
        self.input_genesDE = input_genesDE
        self.additional_edges_pct = additional_edges_pct
        self.cuda = cuda
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.heads = heads
        self.attention = attention
        self.miu = miu
        self.epochs = epochs
        self.repeats = repeats
        self.edge_threshold_param = edge_threshold_param
        self.remove_self_loops = remove_self_loops
        self.topK_drivers = topK_drivers
        self.solver = solver

        NetModel, _ = _get_cefcon_backend()
        self.cefcon_GRN_model = NetModel(hidden_dim=self.hidden_dim,
                                output_dim=self.output_dim,
                                heads=self.heads,
                                attention_type=self.attention,
                                miu=self.miu,
                                epochs=self.epochs,
                                repeats=self.repeats,
                                seed=self.seed,
                                cuda=self.cuda,
                                )

    def preprocess(self):
        print('Start data preparation\n')
        _, data_preparation = _get_cefcon_backend()
        self.data = data_preparation(self.input_expData, self.input_priorNet, genes_DE=self.input_genesDE,
                            additional_edges_pct=self.additional_edges_pct)
        

    def train(self):
        print('Start model training\n')
        self.cefcon_results_dict = {}
        for lineage, data_lineage in self.data.items():
            self.cefcon_GRN_model.run(data_lineage)
            cefcon_results = self.cefcon_GRN_model.get_cefcon_results(edge_threshold_avgDegree=8)
            self.cefcon_results_dict[lineage] = cefcon_results
            del cefcon_results

        print('Finish model training\n')
    

    def predicted_driver_regulators(self):
        for lineage, result_lineage in self.cefcon_results_dict.items():
            print(f'Start predict lineage - {lineage}:')
            print(f'Start calculate gene influence score - {lineage}:')
            result_lineage.gene_influence_score()

            print(f'Start calculate gene driver regulators - {lineage}:')
            result_lineage.driver_regulators(solver = self.solver)
            self.cefcon_results_dict[lineage] = result_lineage
            del result_lineage
    

    def predicted_RGM(self):

        for lineage, result_lineage in self.cefcon_results_dict.items():
            print(f'Start calculate regulon-like gene modules - {lineage}:')
            result_lineage.RGM_activity()
            self.cefcon_results_dict[lineage] = result_lineage
            del result_lineage
        print('Finish predicted\n')
            


def global_imports_members(modulename, members=None, asfunction=False):
    if members is None:
        members = [modulename]  # Default to importing the entire module

    imported_module = __import__(modulename, fromlist=members)

    if asfunction:
        for member in members:
            globals()[member] = getattr(imported_module, member)
    else:
        globals()[modulename] = imported_module

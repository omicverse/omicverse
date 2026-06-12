r"""
Pyomic data (Pyomic.utils._data)
"""


import time
import requests
import os
import pandas as pd
from ._genomics import read_gtf,Gtf
import anndata
import numpy as np
from typing import Callable, List, Mapping, Optional,Dict
from ._enum import ModeEnum
from scipy.sparse import diags, issparse, spmatrix, csr_matrix, isspmatrix_csr
import warnings
from scipy.stats import norm
from .._settings import Colors, EMOJI  # Import Colors and EMOJI from settings
from .._registry import register_function
from ..datasets import download_data_requests
from ..datasets import load_signatures_from_file, predefined_signatures
from ..io.general import load, read_csv, save
from ..io.single import (
    convert_adata_for_rust,
    convert_to_pandas,
    read,
    read_10x_h5,
    read_10x_mtx,
    read_h5ad,
    wrap_dataframe,
)


DATA_DOWNLOAD_LINK_DICT = {
    'cadrres-wo-sample-bias_output_dict_all_genes':{
        'figshare':'https://figshare.com/ndownloader/files/39753568',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/cadrres-wo-sample-bias_output_dict_all_genes.pickle',
    },
    'cadrres-wo-sample-bias_output_dict_prism':{
        'figshare':'https://figshare.com/ndownloader/files/39753571',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/cadrres-wo-sample-bias_output_dict_prism.pickle',
    },
    'cadrres-wo-sample-bias_param_dict_all_genes':{
        'figshare':'https://figshare.com/ndownloader/files/39753574',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/cadrres-wo-sample-bias_param_dict_all_genes.pickle',
    },
    'cadrres-wo-sample-bias_param_dict_prism':{
        'figshare':'https://figshare.com/ndownloader/files/39753577',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/cadrres-wo-sample-bias_param_dict_prism.pickle',
    },
    'GDSC_exp':{
        'figshare':'https://figshare.com/ndownloader/files/39753580',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/GDSC_exp.tsv.gz',
    },
    'masked_drugs':{
        'figshare':'https://figshare.com/ndownloader/files/39753583',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/masked_drugs.csv',
    },
    'GO_Biological_Process_2021':{
        'figshare':'https://figshare.com/ndownloader/files/39820720',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/GO_Biological_Process_2021.txt',
    },
    'GO_Cellular_Component_2021':{
        'figshare':'https://figshare.com/ndownloader/files/39820714',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/GO_Cellular_Component_2021.txt',
    },
    'GO_Molecular_Function_2021':{
        'figshare':'https://figshare.com/ndownloader/files/39820711',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/GO_Molecular_Function_2021.txt',
    },
    'WikiPathway_2021_Human':{
        'figshare':'https://figshare.com/ndownloader/files/39820705',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/WikiPathway_2021_Human.txt',
    },
    'WikiPathways_2019_Mouse':{
        'figshare':'https://figshare.com/ndownloader/files/39820717',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/WikiPathways_2019_Mouse.txt',
    },
    'Reactome_2022':{
        'figshare':'https://figshare.com/ndownloader/files/39820702',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/Reactome_2022.txt',
    },
    'pair_GRCm39':{
        'figshare':'https://figshare.com/ndownloader/files/39820684',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/pair_GRCm39.tsv',
    },
    'pair_T2TCHM13':{
        'figshare':'https://figshare.com/ndownloader/files/39820687',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/pair_T2TCHM13.tsv',
    },
    'pair_GRCh38':{
        'figshare':'https://figshare.com/ndownloader/files/39820690',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/pair_GRCh38.tsv',
    },
    'pair_GRCh37':{
        'figshare':'https://figshare.com/ndownloader/files/39820693',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/pair_GRCh37.tsv',
    },
    'pair_danRer11':{
        'figshare':'https://figshare.com/ndownloader/files/39820696',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/pair_danRer11.tsv',
    },
    'pair_danRer7':{
        'figshare':'https://figshare.com/ndownloader/files/39820699',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/pair_danRer7.tsv',
    },
    'GO_bp':{
        'figshare':'https://figshare.com/ndownloader/files/41460072',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/GO_bp.gmt',
    },
    'TF':{
        'figshare':'https://figshare.com/ndownloader/files/41460066',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/TF.gmt',
    },
    'reactome':{
        'figshare':'https://figshare.com/ndownloader/files/41460051',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/reactome.gmt',
    },
    'm_GO_bp':{
        'figshare':'https://figshare.com/ndownloader/files/41460060',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/m_GO_bp.gmt',
    },
    'm_TF':{
        'figshare':'https://figshare.com/ndownloader/files/41460057',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/m_TF.gmt',
    },
    'm_reactome':{
        'figshare':'https://figshare.com/ndownloader/files/41460054',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/m_reactome.gmt',
    },
    'immune':{
        'figshare':'https://figshare.com/ndownloader/files/41460049',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/immune.gmt',
    },

}


def get_utils_dataset_url(dataset_name: str, prefer_stanford: bool = True) -> str:
    """Get URL for a dataset by name, preferring Stanford over Figshare.

    Parameters
    ----------
    dataset_name : str
        Dataset key in ``DATA_DOWNLOAD_LINK_DICT``.
    prefer_stanford : bool
        Whether Stanford mirror is preferred over Figshare mirror.

    Returns
    -------
    str
        Download URL for requested dataset.

    Raises
    ------
    ValueError
        If dataset name does not exist in ``DATA_DOWNLOAD_LINK_DICT``.
    """
    if dataset_name not in DATA_DOWNLOAD_LINK_DICT:
        raise ValueError(f"Dataset '{dataset_name}' not found in DATA_DOWNLOAD_LINK_DICT")

    dataset_urls = DATA_DOWNLOAD_LINK_DICT[dataset_name]

    if prefer_stanford and 'stanford' in dataset_urls:
        print(f"{Colors.CYAN}Using Stanford mirror for {dataset_name}{Colors.ENDC}")
        return dataset_urls['stanford']
    elif 'figshare' in dataset_urls:
        if prefer_stanford:
            print(f"{Colors.WARNING}{EMOJI['warning']} Stanford link not available for {dataset_name}, using Figshare{Colors.ENDC}")
        return dataset_urls['figshare']
    else:
        raise ValueError(f"No valid URL found for dataset '{dataset_name}'")


# Internal debug logger (opt-in via env OV_DEBUG/OMICVERSE_DEBUG)
def _ov_debug_enabled():
    try:
        val = os.environ.get("OV_DEBUG") or os.environ.get("OMICVERSE_DEBUG")
        if val is None:
            return False
        return str(val).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        return False


def _dbg(msg):
    try:
        if _ov_debug_enabled():
            print(msg)
    except Exception:
        pass




# Deprecated: data_downloader has been replaced by download_data_requests from omicverse.datasets
# All download functions now use download_data_requests for better error handling and progress display

@register_function(
    aliases=['下载 CaDRReS 模型', 'download_CaDRReS_model', 'CaDRReS model download'],
    category="utils",
    description="Download pretrained CaDRReS drug-response models used by single-cell drug sensitivity prediction workflows.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['ov.utils.download_CaDRReS_model()'],
    related=['utils.download_GDSC_data', 'single.Drug_Response']
)
def download_CaDRReS_model():
    r"""
    Download pretrained CaDRReS model parameter/output files.

    Returns
    -------
    None
        Downloads model files into local ``./models`` directory.
    """
    _datasets = [
        'cadrres-wo-sample-bias_output_dict_all_genes',
        'cadrres-wo-sample-bias_output_dict_prism',
        'cadrres-wo-sample-bias_param_dict_all_genes',
        'cadrres-wo-sample-bias_param_dict_prism',
    ]
    for datasets_name in _datasets:
        print(f'{Colors.CYAN}......CaDRReS model download start: {datasets_name}{Colors.ENDC}')
        url = get_utils_dataset_url(datasets_name)
        model_path = download_data_requests(url=url, file_path=f'{datasets_name}.pickle', dir='./models')
    print(f'{Colors.GREEN}{EMOJI["done"]} CaDRReS model download finished!{Colors.ENDC}')

@register_function(
    aliases=['下载 GDSC 数据', 'download_GDSC_data', 'GDSC data download'],
    category="utils",
    description="Download GDSC pharmacogenomic response matrices and annotation files for drug-response modeling.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['ov.utils.download_GDSC_data()'],
    related=['utils.download_CaDRReS_model', 'single.Drug_Response']
)
def download_GDSC_data():
    r"""
    Download GDSC expression and drug mask tables.

    Returns
    -------
    None
        Downloads data files into local ``./models`` directory.
    """
    _datasets = {
        'masked_drugs': '.csv',
        'GDSC_exp': '.tsv.gz',
    }
    for datasets_name, ext in _datasets.items():
        print(f'{Colors.CYAN}......GDSC data download start: {datasets_name}{Colors.ENDC}')
        url = get_utils_dataset_url(datasets_name)
        download_data_requests(url=url, file_path=f'{datasets_name}{ext}', dir='./models')
    print(f'{Colors.GREEN}{EMOJI["done"]} GDSC data download finished!{Colors.ENDC}')

@register_function(
    aliases=["下载通路数据库", "download_pathway_database", "download_genesets", "通路数据下载"],
    category="utils",
    description="Download pathway and gene set databases for enrichment analysis",
    examples=[
        "ov.utils.download_pathway_database()",
        "# Downloads the following databases:",
        "# - GO_Biological_Process_2021",
        "# - GO_Cellular_Component_2021", 
        "# - GO_Molecular_Function_2021",
        "# - WikiPathway_2021_Human",
        "# - WikiPathways_2019_Mouse",
        "# - Reactome_2022"
    ],
    related=["utils.geneset_prepare", "bulk.geneset_enrichment", "bulk.pyGSEA"]
)
def download_pathway_database():
    r"""Download pathway and gene set databases for enrichment analysis.

    Returns
    -------
    None
        Downloads pathway resources to local ``./genesets`` directory.
    """
    _datasets = [
        'GO_Biological_Process_2021',
        'GO_Cellular_Component_2021',
        'GO_Molecular_Function_2021',
        'WikiPathway_2021_Human',
        'WikiPathways_2019_Mouse',
        'Reactome_2022',
    ]

    for datasets_name in _datasets:
        print(f'{Colors.CYAN}......Pathway Geneset download start: {datasets_name}{Colors.ENDC}')
        url = get_utils_dataset_url(datasets_name)
        download_data_requests(url=url, file_path=f'{datasets_name}.txt', dir='./genesets')
    print(f'{Colors.GREEN}{EMOJI["done"]} Pathway Geneset download finished!{Colors.ENDC}')
    print(f'{Colors.CYAN}......Other Genesets can be downloaded from https://maayanlab.cloud/Enrichr/#libraries{Colors.ENDC}')

@register_function(
    aliases=["下载基因ID注释", "download_geneid_annotation_pair", "download_gene_mapping", "基因ID映射下载"],
    category="utils",
    description="Download gene ID annotation mapping files for various organisms",
    examples=[
        "ov.utils.download_geneid_annotation_pair()",
        "# Files downloaded to genesets/ directory:",
        "# - pair_GRCm39.tsv (Mouse)",
        "# - pair_GRCh38.tsv (Human)",
        "# - pair_GRCh37.tsv (Human legacy)",
        "# - pair_danRer11.tsv (Zebrafish)"
    ],
    related=["bulk.Matrix_ID_mapping", "utils.geneset_prepare"]
)
def download_geneid_annotation_pair():
    r"""Download gene ID annotation mapping files for various organisms.

    Returns
    -------
    None
        Downloads gene ID mapping tables to local ``./genesets`` directory.
    """
    _datasets = [
        'pair_GRCm39',
        'pair_T2TCHM13',
        'pair_GRCh38',
        'pair_GRCh37',
        'pair_danRer11',
        'pair_danRer7',
    ]

    # Add special handling for pair_hgnc_all
    _special_datasets = {
        'pair_hgnc_all': 'https://github.com/Starlitnightly/omicverse/files/14664966/pair_hgnc_all.tsv.tar.gz'
    }

    for datasets_name in _datasets:
        print(f'{Colors.CYAN}......Geneid Annotation Pair download start: {datasets_name}{Colors.ENDC}')
        url = get_utils_dataset_url(datasets_name)
        download_data_requests(url=url, file_path=f'{datasets_name}.tsv', dir='./genesets')

    # Handle special datasets not in DATA_DOWNLOAD_LINK_DICT
    for datasets_name, url in _special_datasets.items():
        print(f'{Colors.CYAN}......Geneid Annotation Pair download start: {datasets_name}{Colors.ENDC}')
        import tarfile
        tar_path = download_data_requests(url=url, file_path=f'{datasets_name}.tar.gz', dir='./genesets')
        # Extract the TSV file from tar.gz
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path='genesets/')
        print(f'{Colors.GREEN}......Extracted {datasets_name}.tsv from tar.gz{Colors.ENDC}')

    print(f'{Colors.GREEN}{EMOJI["done"]} Geneid Annotation Pair download finished!{Colors.ENDC}')

@register_function(
    aliases=["GTF转换", "gtf_to_pair_tsv", "gtf_to_mapping", "GTF基因映射", "convert_gtf"],
    category="utils",
    description="Convert GTF file to gene ID mapping pairs TSV format for Matrix_ID_mapping",
    examples=[
        "# Convert GTF to mapping pairs",
        "gene_count = ov.utils.gtf_to_pair_tsv('genes.gtf', 'gene_pairs.tsv')",
        "# Keep version numbers in gene IDs",
        "ov.utils.gtf_to_pair_tsv('genes.gtf', 'gene_pairs.tsv', gene_id_version=True)",
        "# Remove version numbers from gene IDs", 
        "ov.utils.gtf_to_pair_tsv('genes.gtf', 'gene_pairs.tsv', gene_id_version=False)",
        "# Use converted file for gene mapping",
        "data = ov.bulk.Matrix_ID_mapping(data, 'gene_pairs.tsv')"
    ],
    related=["bulk.Matrix_ID_mapping", "utils.download_geneid_annotation_pair", "utils.read_gtf"]
)
def gtf_to_pair_tsv(gtf_path, output_path, gene_id_version=True):
    r"""Convert GTF file to gene ID mapping pairs TSV format.

    Parameters
    ----------
    gtf_path : str
        Path to input GTF file.
    output_path : str
        Path for output TSV file.
    gene_id_version : bool
        Whether to keep version numbers in gene IDs.

    Returns
    -------
    int
        Number of unique genes written to output file.

    Examples
    --------
        >>> import omicverse as ov
        >>> # Convert GTF to mapping pairs
        >>> gene_count = ov.utils.gtf_to_pair_tsv('genes.gtf', 'gene_pairs.tsv')
        >>> # Use converted file for gene mapping
        >>> data = ov.bulk.Matrix_ID_mapping(data, 'gene_pairs.tsv')
    """
    import pandas as pd
    from ._genomics import read_gtf
    
    print(f'......Reading GTF file: {gtf_path}')
    
    # Read GTF file using existing reader
    gtf = read_gtf(gtf_path)
    
    # Filter for gene features only
    gene_features = gtf[gtf['feature'] == 'gene'].copy()
    print(f'......Found {len(gene_features)} gene features')
    
    if len(gene_features) == 0:
        raise ValueError("No gene features found in GTF file!")
    
    # Split attributes to extract gene_id and gene_name
    gene_features = gene_features.split_attribute()
    
    # Check required columns
    if 'gene_id' not in gene_features.columns:
        raise ValueError("gene_id not found in GTF attributes!")
    
    # Extract gene_id and gene_name
    gene_pairs = []
    for idx, row in gene_features.iterrows():
        gene_id = str(row['gene_id'])
        
        # Handle version numbers in gene IDs
        if not gene_id_version and '.' in gene_id:
            gene_id = gene_id.split('.')[0]
        
        # Use gene_name if available, otherwise use gene_id as symbol
        if 'gene_name' in gene_features.columns and pd.notna(row['gene_name']) and str(row['gene_name']) != '.':
            symbol = str(row['gene_name'])
        else:
            symbol = gene_id
            
        gene_pairs.append([gene_id, symbol])
    
    # Create DataFrame and remove duplicates
    df = pd.DataFrame(gene_pairs, columns=['gene_id', 'symbol'])
    df = df.drop_duplicates(subset=['gene_id'], keep='first')
    
    print(f'......Processed {len(df)} unique genes')
    
    # Save to TSV
    df.to_csv(output_path, sep='\t', index=False)
    print(f'......Gene mapping pairs saved to: {output_path}')
    
    return len(df)

@register_function(
    aliases=['下载 TOSICA 基因集', 'download_tosica_gmt', 'tosica gmt'],
    category="utils",
    description="Download curated GMT pathway/gene-set files required by TOSICA-based single-cell annotation workflows.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['ov.utils.download_tosica_gmt()'],
    related=['single.pyTOSICA', 'single.pathway_enrichment']
)
def download_tosica_gmt():
    r"""
    Download curated GMT files used by TOSICA workflows.

    Returns
    -------
    None
        Downloads GMT files into local ``./genesets`` directory.
    """
    _datasets = [
        'GO_bp',
        'TF',
        'reactome',
        'm_GO_bp',
        'm_TF',
        'm_reactome',
        'immune',
    ]

    for datasets_name in _datasets:
        print(f'{Colors.CYAN}......TOSICA gmt dataset download start: {datasets_name}{Colors.ENDC}')
        url = get_utils_dataset_url(datasets_name)
        download_data_requests(url=url, file_path=f'{datasets_name}.gmt', dir='./genesets')
    print(f'{Colors.GREEN}{EMOJI["done"]} TOSICA gmt dataset download finished!{Colors.ENDC}')

@register_function(
    aliases=["基因集准备", "geneset_prepare", "pathway_prepare", "基因集加载", "load_geneset"],
    category="utils",
    description="Load and prepare gene sets from GMT/TXT files for enrichment analysis",
    examples=[
        "# Load human gene sets",
        "geneset_dict = ov.utils.geneset_prepare('KEGG_pathways.gmt', organism='Human')",
        "# Load mouse gene sets",
        "geneset_dict = ov.utils.geneset_prepare('GO_biological_process.txt', organism='Mouse')",
        "# Use with enrichment analysis",
        "geneset_dict = ov.utils.geneset_prepare('c2.cp.kegg.v7.4.symbols.gmt')",
        "enrich_res = ov.bulk.geneset_enrichment(gene_list, geneset_dict)"
    ],
    related=["bulk.geneset_enrichment", "utils.download_tosica_gmt", "single.pathway_enrichment"]
)
def _download_enrichr_library(name, dir='./genesets'):
    r"""Fetch a gene-set library from Enrichr by name, cache it, return its path.

    Enrichr hosts ~200 curated gene-set libraries — MSigDB Hallmark, KEGG,
    GO, Reactome, WikiPathways, cell-type and TF/kinase sets — listed at
    https://maayanlab.cloud/Enrichr/#libraries . This lets
    :func:`geneset_prepare` resolve any of them on demand instead of
    failing when the local file is absent.

    Parameters
    ----------
    name : str
        Enrichr library name, e.g. ``"MSigDB_Hallmark_2020"``,
        ``"KEGG_2021_Human"``, ``"GO_Biological_Process_2021"``.
    dir : str, default='./genesets'
        Directory to cache the downloaded ``.gmt`` into.

    Returns
    -------
    str or None
        Local path to the cached library, or ``None`` if the name is
        unknown to Enrichr or the download fails (e.g. offline).
    """
    import requests
    os.makedirs(dir, exist_ok=True)
    out = os.path.join(dir, f'{name}.gmt')
    if os.path.exists(out):
        return out
    url = ('https://maayanlab.cloud/Enrichr/geneSetLibrary'
           f'?mode=text&libraryName={name}')
    try:
        print(f"   - Geneset '{name}' missing locally; fetching from Enrichr ...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        text = resp.text
        # Enrichr returns an empty body / an HTML page for an unknown name.
        if (not text.strip()) or text.lstrip()[:1] == '<' or '\t' not in text:
            print(f"   - Enrichr has no library named '{name}'.")
            return None
        with open(out, 'w', encoding='utf-8') as fh:
            fh.write(text)
        n_terms = sum(1 for ln in text.splitlines() if ln.strip())
        print(f"   - Downloaded Enrichr library '{name}' "
              f"({n_terms} gene sets) -> {out}")
        return out
    except Exception as exc:  # network error, bad name, etc.
        print(f"   - Enrichr download failed for '{name}': {exc}")
        return None


def geneset_prepare(geneset_path,organism='Human',auto_download=True):
    r"""Load and prepare gene sets from GMT/TXT files for enrichment analysis.

    Parameters
    ----------
    geneset_path : str
        Path to geneset file.
    organism : str
        Organism name used for gene-symbol case normalization.
    auto_download : bool, default=True
        If the path does not exist AND its basename matches a known
        omicverse pathway resource (``GO_*_2021``, ``Reactome_2022``,
        ``WikiPathway_2021_Human``, ``WikiPathways_2019_Mouse``), call
        :func:`download_pathway_database` to fetch all standard genesets
        into ``./genesets/`` and retry. Disable to fail fast offline.

    Returns
    -------
    dict
        Dictionary where keys are pathway names and values are gene symbol lists.
    """
    result_dict = {}
    file_path=geneset_path
    if auto_download and not os.path.exists(file_path):
        known = {
            'GO_Biological_Process_2021', 'GO_Cellular_Component_2021',
            'GO_Molecular_Function_2021', 'WikiPathway_2021_Human',
            'WikiPathways_2019_Mouse', 'Reactome_2022',
        }
        stem = os.path.splitext(os.path.basename(file_path))[0]
        resolved = None
        # Reuse a previously downloaded copy under ./genesets/ instead of
        # re-fetching on every call. Both standard (download_pathway_database
        # -> .txt) and Enrichr (.gmt) caches land here.
        for ext in ('.txt', '.gmt'):
            cand = os.path.join('./genesets', f'{stem}{ext}')
            if os.path.exists(cand):
                resolved = cand
                break
        if resolved is None and stem in known:
            print(f"   - Geneset '{stem}' missing locally; auto-downloading "
                  f"via ov.utils.download_pathway_database()...")
            download_pathway_database()
            # download_pathway_database writes to ./genesets/
            cand = os.path.join('./genesets', f'{stem}.txt')
            if os.path.exists(cand):
                resolved = cand
        if resolved is None:
            # Any other name is treated as an Enrichr gene-set library and
            # fetched on demand — MSigDB_Hallmark_2020, KEGG_2021_Human,
            # Reactome_2022, and ~200 more. The caller never has to
            # pre-stage a database file or have internet arranged manually.
            resolved = _download_enrichr_library(stem)
        if resolved is not None and os.path.exists(resolved):
            file_path = resolved

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"geneset_prepare: '{geneset_path}' not found locally and could "
            f"not be auto-downloaded. Pass a path to a .gmt/.txt file, or an "
            f"Enrichr library name such as 'MSigDB_Hallmark_2020' or "
            f"'KEGG_2021_Human' — browse names at "
            f"https://maayanlab.cloud/Enrichr/#libraries"
        )

    with open(file_path, 'r', encoding='utf-8') as file:
        for idx,line in enumerate(file):
            line = line.strip()
            if not line:
                continue

            # 自动检测第一个分隔符
            if idx==0:
                first_delimiter=None
                delimiters = ['\t\t',',', '\t', ';', ' ',]
                for delimiter in delimiters:
                    if delimiter in line:
                        first_delimiter = delimiter
                        break
            
            if first_delimiter is None:
                # 如果找不到分隔符，跳过这行
                continue
            
            # 使用第一个分隔符分割行
            parts = line.split(first_delimiter, 1)
            if len(parts) != 2:
                continue
            
            key = parts[0].strip()
            # 使用剩余部分的第一个字符作为分隔符来分割
            value = parts[1].strip().split()

            # 将键值对添加到字典中
            result_dict[key] = value
    go_bio_dict=result_dict

    if (organism == 'Mouse') or (organism == 'mouse') or (organism == 'mm'):
        for key in go_bio_dict:
            go_bio_dict[key]=[i.lower().capitalize() for i in go_bio_dict[key]]
    elif (organism == 'Human') or (organism == 'human') or (organism == 'hs'):
        for key in go_bio_dict:
            go_bio_dict[key]=[i.upper() for i in go_bio_dict[key]]
    else:
        for key in go_bio_dict:
            go_bio_dict[key]=[i for i in go_bio_dict[key]]

    return go_bio_dict

def geneset_prepare_old(geneset_path,organism='Human'):
    r"""
    Legacy geneset loader for old double-tab GMT-like files.

    Parameters
    ----------
    geneset_path : str
        Path to geneset file.
    organism : str
        Organism name used for gene-symbol case normalization.

    Returns
    -------
    dict
        Dictionary of pathway-to-genes mapping.
    """
    go_bio_geneset=pd.read_csv(geneset_path,sep='\t\t',header=None)
    go_bio_dict={}
    if (organism == 'Mouse') or (organism == 'mouse') or (organism == 'mm'):
        for i in go_bio_geneset.index:
            go_bio_dict[go_bio_geneset.loc[i,0]]=[i.lower().capitalize() for i in go_bio_geneset.loc[i,1].split('\t')]
    elif (organism == 'Human') or (organism == 'human') or (organism == 'hs'):
        for i in go_bio_geneset.index:
            go_bio_dict[go_bio_geneset.loc[i,0]]=[i.upper() for i in go_bio_geneset.loc[i,1].split('\t')]
    else:
        for i in go_bio_geneset.index:
            go_bio_dict[go_bio_geneset.loc[i,0]]=[i for i in go_bio_geneset.loc[i,1].split('\t')]
    return go_bio_dict

@register_function(
    aliases=['基因注释映射', 'get_gene_annotation', 'gtf annotation mapping'],
    category="utils",
    description="Map transcript/gene identifiers to annotation fields (e.g., symbol, biotype) using GTF metadata and store into adata.var.",
    prerequisites={},
    requires={'var': ['gene identifiers']},
    produces={'var': ['gene annotation columns']},
    auto_fix='none',
    examples=['ov.utils.get_gene_annotation(adata, var_by="gene_id", gtf="genes.gtf", gtf_by="gene_id")'],
    related=['generate_reference_table', 'utils.read_csv']
)
def get_gene_annotation(
        adata: anndata.AnnData, var_by: str = None,
        gtf: os.PathLike = None, gtf_by: str = None,
        by_func: Optional[Callable] = None
) -> None:
    r"""
    Annotate ``adata.var`` by merging with gene-level GTF attributes.

    Parameters
    ----------
        adata : Input dataset.
        var_by : Specify a column in ``adata.var`` used to merge with GTF attributes,
            otherwise ``adata.var_names`` is used by default.
        gtf : Path to the GTF file.
        gtf_by : Specify a field in the GTF attributes used to merge with ``adata.var``,
            e.g. "gene_id", "gene_name".
        by_func : Specify an element-wise function used to transform merging fields,
            e.g. removing suffix in gene IDs.

    Note:
        The genomic locations are converted to 0-based as specified
        in bed format rather than 1-based as specified in GTF format.

    """
    if gtf is None:
        raise ValueError("Missing required argument `gtf`!")
    if gtf_by is None:
        raise ValueError("Missing required argument `gtf_by`!")
    var_by = adata.var_names if var_by is None else adata.var[var_by]
    gtf = read_gtf(gtf).query("feature == 'gene'").split_attribute()
    if by_func:
        by_func = np.vectorize(by_func)
        var_by = by_func(var_by)
        gtf[gtf_by] = by_func(gtf[gtf_by])  # Safe inplace modification
    gtf = gtf.sort_values("seqname").drop_duplicates(
        subset=[gtf_by], keep="last"
    )  # Typically, scaffolds come first, chromosomes come last
    merge_df = pd.concat([
        pd.DataFrame(gtf.to_bed(name=gtf_by)),
        pd.DataFrame(gtf).drop(columns=Gtf.COLUMNS)  # Only use the splitted attributes
    ], axis=1).set_index(gtf_by).reindex(var_by).set_index(adata.var.index)
    adata.var = adata.var.assign(**merge_df)
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    Literal,
    TypeVar,
    Callable,
    Hashable,
    Iterable,
    Optional,
    Sequence,
)

class TestMethod(ModeEnum):  # noqa
    FISHER = "fisher"
    PERM_TEST = "perm_test"



def _mat_mat_corr_sparse(
    X: csr_matrix,
    Y: np.ndarray,
) -> np.ndarray:
    n = X.shape[1]

    X_bar = np.reshape(np.array(X.mean(axis=1)), (-1, 1))
    X_std = np.reshape(
        np.sqrt(np.array(X.power(2).mean(axis=1)) - (X_bar**2)), (-1, 1)
    )

    y_bar = np.reshape(np.mean(Y, axis=0), (1, -1))
    y_std = np.reshape(np.std(Y, axis=0), (1, -1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (X @ Y - (n * X_bar * y_bar)) / ((n - 1) * X_std * y_std)

def correlation_pseudotime(
    X: Union[np.ndarray, spmatrix],
    Y: np.ndarray,
    method: TestMethod = TestMethod.FISHER,
    n_perms: Optional[int] = None,
    seed: Optional[int] = None,
    confidence_level: float = 0.95,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the correlation between rows in matrix ``X`` columns of matrix ``Y``.

    Parameters
    ----------
    X
        Array or matrix of `(M, N)` elements.
    Y
        Array of `(N, K)` elements.
    method
        Method for p-value calculation.
    n_perms
        Number of permutations if ``method='perm_test'``.
    seed
        Random seed if ``method = 'perm_test'``.
    confidence_level
        Confidence level for the confidence interval calculation. Must be in `[0, 1]`.
    kwargs
        Keyword arguments for :func:`cellrank._utils._parallelize.parallelize`.

    Returns
    -------
        Correlations, p-values, corrected p-values, lower and upper bound of 95% confidence interval.
        Each array if of shape ``(n_genes, n_lineages)``.
    """

    def perm_test_extractor(
        res: Sequence[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pvals, corr_bs = zip(*res)
        pvals = np.sum(pvals, axis=0) / float(n_perms)

        corr_bs = np.concatenate(corr_bs, axis=0)
        corr_ci_low, corr_ci_high = np.quantile(corr_bs, q=ql, axis=0), np.quantile(
            corr_bs, q=qh, axis=0
        )

        return pvals, corr_ci_low, corr_ci_high

    if not (0 <= confidence_level <= 1):
        raise ValueError(
            f"Expected `confidence_level` to be in interval `[0, 1]`, found `{confidence_level}`."
        )

    n = X.shape[1]  # genes x cells
    ql = 1 - confidence_level - (1 - confidence_level) / 2.0
    qh = confidence_level + (1 - confidence_level) / 2.0

    if issparse(X) and not isspmatrix_csr(X):
        X = csr_matrix(X)

    corr = _mat_mat_corr_sparse(X, Y) if issparse(X) else _mat_mat_corr_dense(X, Y)

    if method == TestMethod.FISHER:
        # see: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Using_the_Fisher_transformation
        mean, se = np.arctanh(corr), 1.0 / np.sqrt(n - 3)
        z_score = (np.arctanh(corr) - np.arctanh(0)) * np.sqrt(n - 3)

        z = norm.ppf(qh)
        corr_ci_low = np.tanh(mean - z * se)
        corr_ci_high = np.tanh(mean + z * se)
        pvals = 2 * norm.cdf(-np.abs(z_score))
    else:
        raise NotImplementedError(method)
    '''
    elif method == TestMethod.PERM_TEST:
        if not isinstance(n_perms, int):
            raise TypeError(
                f"Expected `n_perms` to be an integer, found `{type(n_perms).__name__}`."
            )
        if n_perms <= 0:
            raise ValueError(f"Expcted `n_perms` to be positive, found `{n_perms}`.")


        pvals, corr_ci_low, corr_ci_high = parallelize(
            _perm_test,
            np.arange(n_perms),
            as_array=False,
            unit="permutation",
            extractor=perm_test_extractor,
            **kwargs,
        )(corr, X, Y, seed=seed)
    '''
    

    return corr, pvals, corr_ci_low, corr_ci_high

def _np_apply_along_axis(func1d, axis: int, arr: np.ndarray) -> np.ndarray:
    """
    Apply a reduction function over a given axis.

    Parameters
    ----------
    func1d
        Reduction function that operates only on 1 dimension.
    axis
        Axis over which to apply the reduction.
    arr
        The array to be reduced.

    Returns
    -------
    The reduced array.
    """

    assert arr.ndim == 2
    assert axis in [0, 1]

    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
        return result

    result = np.empty(arr.shape[0])
    for i in range(len(result)):
        result[i] = func1d(arr[i, :])

    return result

def np_mean(array: np.ndarray, axis: int) -> np.ndarray:  # noqa
    return _np_apply_along_axis(np.mean, axis, array)

def np_std(array: np.ndarray, axis: int) -> np.ndarray:  # noqa
    return _np_apply_along_axis(np.std, axis, array)

def _mat_mat_corr_dense(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    #from cellrank.kernels._utils import np_std, np_mean

    n = X.shape[1]

    X_bar = np.reshape(np_mean(X, axis=1), (-1, 1))
    X_std = np.reshape(np_std(X, axis=1), (-1, 1))

    y_bar = np.reshape(np_mean(Y, axis=0), (1, -1))
    y_std = np.reshape(np_std(Y, axis=0), (1, -1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (X @ Y - (n * X_bar * y_bar)) / ((n - 1) * X_std * y_std)


def _perm_test(
    ixs: np.ndarray,
    corr: np.ndarray,
    X: Union[np.ndarray, spmatrix],
    Y: np.ndarray,
    seed: Optional[int] = None,
    queue=None,
) -> Tuple[np.ndarray, np.ndarray]:
    rs = np.random.RandomState(None if seed is None else seed + ixs[0])
    cell_ixs = np.arange(X.shape[1])
    pvals = np.zeros_like(corr, dtype=np.float64)
    corr_bs = np.zeros((len(ixs), X.shape[0], Y.shape[1]))  # perms x genes x lineages

    mmc = _mat_mat_corr_sparse if issparse(X) else _mat_mat_corr_dense

    for i, _ in enumerate(ixs):
        rs.shuffle(cell_ixs)
        corr_i = mmc(X, Y[cell_ixs, :])
        pvals += np.abs(corr_i) >= np.abs(corr)

        bootstrap_ixs = rs.choice(cell_ixs, replace=True, size=len(cell_ixs))
        corr_bs[i, :, :] = mmc(X[:, bootstrap_ixs], Y[bootstrap_ixs, :])

        if queue is not None:
            queue.put(1)

    if queue is not None:
        queue.put(None)

    return pvals, corr_bs

def anndata_sparse(adata):
    """
    Convert ``adata.X`` to CSR sparse matrix.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.

    Returns
    -------
    AnnData
        AnnData with ``X`` converted to CSR format.
    """

    from scipy.sparse import csr_matrix
    x = csr_matrix(adata.X.copy())
    adata.X=x
    return adata

@register_function(
    aliases=["存储层数据", "store_layers", "save_layers", "层数据存储", "保存层"],
    category="utils",
    description="Store the X matrix of AnnData in adata.uns for later retrieval",
    examples=[
        "# Store current X matrix as 'counts'",
        "ov.utils.store_layers(adata, layers='counts')",
        "# Store normalized data",
        "ov.utils.store_layers(adata, layers='normalized')",
        "# Use with preprocessing pipeline",
        "ov.utils.store_layers(adata, layers='raw')",
        "adata = ov.pp.preprocess(adata)",
        "ov.utils.retrieve_layers(adata, layers='raw')"
    ],
    related=["utils.retrieve_layers", "pp.preprocess", "pp.scale"]
)
def store_layers(adata,layers='counts'):
    """Store the X matrix of AnnData in adata.uns for later retrieval.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing single-cell data.
    layers : str
        Layer name used for stored snapshot.

    Returns
    -------
    None
        Stores current ``adata.X`` snapshot into ``adata.uns``.

    Examples
    --------
        >>> import omicverse as ov
        >>> # Store original counts before preprocessing
        >>> ov.utils.store_layers(adata, layers='raw_counts')
        >>> # Apply preprocessing
        >>> adata = ov.pp.preprocess(adata)
        >>> # Retrieve original data if needed
        >>> ov.utils.retrieve_layers(adata, layers='raw_counts')
    """


    if issparse(adata.X) and not isspmatrix_csr(adata.X):
        adata.uns['layers_{}'.format(layers)]=anndata.AnnData(csr_matrix(adata.X.copy()),
                                           obs=pd.DataFrame(index=adata.obs.index),
                                          var=pd.DataFrame(index=adata.var.index),)
    elif issparse(adata.X):
        adata.uns['layers_{}'.format(layers)]=anndata.AnnData(adata.X.copy(),
                                           obs=pd.DataFrame(index=adata.obs.index),
                                           var=pd.DataFrame(index=adata.var.index),)
    else:
        adata.uns['layers_{}'.format(layers)]=anndata.AnnData(csr_matrix(adata.X.copy()),
                                           obs=pd.DataFrame(index=adata.obs.index),
                                          var=pd.DataFrame(index=adata.var.index),)
    print('......The X of adata have been stored in {}'.format(layers))

@register_function(
    aliases=["检索层数据", "retrieve_layers", "get_layers", "层数据检索", "获取层"],
    category="utils",
    description="Retrieve previously stored X matrix from adata.uns and restore to adata.X",
    examples=[
        "# Retrieve stored counts data",
        "ov.utils.retrieve_layers(adata, layers='counts')",
        "# Retrieve raw data after preprocessing",
        "ov.utils.retrieve_layers(adata, layers='raw')",
        "# Complete workflow example",
        "ov.utils.store_layers(adata, layers='original')",
        "adata = ov.pp.preprocess(adata)",
        "ov.utils.retrieve_layers(adata, layers='original')"
    ],
    related=["utils.store_layers", "pp.preprocess", "pp.scale"]
)
def retrieve_layers(adata,layers='counts'):
    """Retrieve previously stored X matrix from adata.uns and restore to adata.X.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing single-cell data.
    layers : str
        Layer name used for stored snapshot retrieval.

    Returns
    -------
    None
        Restores stored matrix into ``adata.X``.

    Examples
    --------
        >>> import omicverse as ov
        >>> # Store original data before preprocessing
        >>> ov.utils.store_layers(adata, layers='raw_counts')
        >>> # Apply preprocessing
        >>> adata = ov.pp.preprocess(adata)
        >>> # Retrieve original data
        >>> ov.utils.retrieve_layers(adata, layers='raw_counts')
    """

    adata_test=adata.uns['layers_{}'.format(layers)].copy()
    adata_test=adata_test[adata.obs.index,adata.var.index]
    
    if issparse(adata.X) and not isspmatrix_csr(adata.X):
        adata.uns['layers_raw'.format(layers)]=anndata.AnnData(csr_matrix(adata.X.copy()),
                                           obs=pd.DataFrame(index=adata.obs.index),
                                          var=pd.DataFrame(index=adata.var.index),)
    elif issparse(adata.X):
        adata.uns['layers_raw'.format(layers)]=anndata.AnnData(adata.X.copy(),
                                           obs=pd.DataFrame(index=adata.obs.index),
                                           var=pd.DataFrame(index=adata.var.index),)
    else:
        adata.uns['layers_raw'.format(layers)]=anndata.AnnData(csr_matrix(adata.X.copy()),
                                           obs=pd.DataFrame(index=adata.obs.index),
                                          var=pd.DataFrame(index=adata.var.index),)
    print('......The X of adata have been stored in raw')
    adata.X=adata_test.X.copy()
    print('......The layers {} of adata have been retreved'.format(layers))
    del adata_test


class easter_egg(object):

    def __init__(self,):
        print('Easter egg is ready to be hatched!')

    def O(self):
        print('尊嘟假嘟')


# Note: save/load/read-conversion routines were moved to `omicverse.io`.


from pathlib import Path
import re

def split_pattern(name: str):
    """
    把文件名拆成:
    prefix + number + suffix
    例如:
    CellOverlay_F001.jpg -> ('CellOverlay_F', 1, '.jpg')
    """
    m = re.match(r"^(.*?)(\d+)(\.[^.]+)$", name)
    if m:
        prefix, num, suffix = m.groups()
        return prefix, int(num), suffix
    return None

def compress_files(files):
    """
    将同类文件压缩成:
    [首个文件, '...', 末个文件]
    """
    groups = {}
    others = []

    for f in files:
        pat = split_pattern(f.name)
        if pat is None:
            others.append(f)
        else:
            key = (pat[0], pat[2])  # prefix, suffix
            groups.setdefault(key, []).append((pat[1], f))

    result = []

    for key in sorted(groups):
        items = sorted(groups[key], key=lambda x: x[0])
        only_files = [f for _, f in items]
        if len(only_files) <= 2:
            result.extend(only_files)
        else:
            result.extend([only_files[0], "...", only_files[-1]])

    result.extend(sorted(others, key=lambda x: x.name.lower()))
    return result

def print_tree(path: Path, prefix: str = ""):
    print(prefix + path.name + "/")

    dirs = sorted([p for p in path.iterdir() if p.is_dir()], key=lambda x: x.name.lower())
    files = sorted([p for p in path.iterdir() if p.is_file()], key=lambda x: x.name.lower())
    files = compress_files(files)

    children = dirs + files

    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        branch = "└── " if is_last else "├── "
        next_prefix = prefix + ("    " if is_last else "│   ")

        if child == "...":
            print(prefix + branch + "...")
        elif isinstance(child, Path) and child.is_dir():
            print_tree(child, next_prefix)
        else:
            print(prefix + branch + child.name)
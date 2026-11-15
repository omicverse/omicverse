import ntpath
import os
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import requests
from typing import Optional
from tqdm import tqdm

import pandas as pd
import numpy as np
from anndata import AnnData, read_h5ad
try:  # anndata >= 0.11 moved the readers to anndata.io
    from anndata.io import read_loom
except ImportError:  # pragma: no cover - older anndata
    from anndata import read_loom
import warnings
from .._registry import register_function

# Import omicverse color settings
try:
    from .._settings import Colors, EMOJI
except ImportError:
    # Fallback if settings not available
    class Colors:
        HEADER = '\033[95m'
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
    
    EMOJI = {
        "start": "🔍",
        "done": "✅",
        "error": "❌",
        "warning": "⚠️",
    }


DATA_DOWNLOAD_LINK_DICT = {
    'neuron_splicing':{
        'figshare':'https://figshare.com/ndownloader/files/47439605',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/neuron_splicing.h5ad',
    },
    'neuron_labeling':{
        'figshare':'https://figshare.com/ndownloader/files/47439629',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/neuron_labeling.h5ad',
    },
    'zebrafish':{
        'figshare':'https://figshare.com/ndownloader/files/47420257',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/zebrafish.h5ad',
    },
    'bone_marrow':{
        'figshare':'https://figshare.com/ndownloader/files/35826944',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/setty_bone_marrow.h5ad',
    },
    'human_tfs':{
        'figshare':'https://figshare.com/ndownloader/files/47439617',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/human_tfs.txt',
    },
    'onefilepercell_A1_unique_and_others_J2CH1':{
        'figshare':'https://figshare.com/ndownloader/files/47439620',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/onefilepercell_A1_unique_and_others_J2CH1.loom',
    },
    '10X_multiome_mouse_brain':{
        'figshare':'https://figshare.com/ndownloader/files/54153947',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/10X_multiome_mouse_brain.loom',
    },
    'cell_annotations':{
        'figshare':'https://figshare.com/ndownloader/files/54154376',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/cell_annotations.tsv',
    },
    'dentategyrus_scv':{
        'figshare':'https://figshare.com/ndownloader/files/47439623',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/dentategyrus_scv.h5ad',
    },
    'hematopoiesis_raw':{
        'figshare':'https://figshare.com/ndownloader/files/47439626',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/hematopoiesis_raw.h5ad',
    },
    'rpe1':{
        'figshare':'https://figshare.com/ndownloader/files/47439641',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/rpe1.h5ad',
    },
    'organoid':{
        'figshare':'https://figshare.com/ndownloader/files/47439632',
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/organoid.h5ad',
    },
    'hematopoiesis':{
        'figshare':'https://figshare.com/ndownloader/files/47439635',  
        'stanford':'https://stacks.stanford.edu/file/sh696dv4420/hematopoiesis.h5ad',
    },
    'COVID_PBMC_bulk':{
        'figshare':'https://figshare.com/ndownloader/files/59192924',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/COVID_PBMC_bulk_GSE152418.h5ad',
    },
    'COVID_PBMC_single':{
        'figshare':'https://figshare.com/ndownloader/files/59192927',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/COVID_PBMC_sc_ref.h5ad',
    },
    'pbmc8k':{
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/pbmc8k.h5ad',
        'figshare':'',
    },
    'seqfish':{
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/seqfish.h5ad',
        'figshare':'https://exampledata.scverse.org/squidpy/figshare/seqfish.h5ad',
    },
}

@register_function(
    aliases=['下载数据', 'download_data', 'dataset downloader'],
    category="datasets",
    description="Download OmicVerse tutorial datasets or resources to local cache with progress and integrity checks.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['ov.datasets.download_data(url, file_path="pbmc3k.h5ad", dir="data")'],
    related=['datasets.pancreatic_endocrinogenesis', 'datasets.sc_ref_Lymph_Node']
)
def download_data(url: str, file_path: Optional[str] = None, dir: str = "./data") -> str:
    """Download a dataset file to local storage.

    Parameters
    ----------
    url:str
        Source URL of the dataset file.
    file_path:Optional[str]
        Target filename. If ``None``, the basename of ``url`` is used.
    dir:str
        Output directory where file is stored.

    Returns
    -------
    str
        Absolute/relative local path of downloaded file.
    """
    file_path = ntpath.basename(url) if file_path is None else file_path
    file_path = os.path.join(dir, file_path)
    print(f"{Colors.BLUE}{EMOJI['start']} Downloading data to {file_path}{Colors.ENDC}")

    if not os.path.exists(file_path):
        parent_dir = os.path.dirname(file_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        # download the data with colored progress bar
        class ColoredTqdm(tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                
        with ColoredTqdm(
            unit='B', 
            unit_scale=True, 
            desc=f"{Colors.GREEN}Downloading{Colors.ENDC}",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            colour='green'
        ) as t:
            def report_progress(block_num, block_size, total_size):
                if t.total != total_size:
                    t.total = total_size
                downloaded = block_num * block_size
                if downloaded <= total_size:
                    t.update(downloaded - t.n)
            urlretrieve(url, file_path, reporthook=report_progress)
        print(f"{Colors.GREEN}{EMOJI['done']} Download completed{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}{EMOJI['warning']} File {file_path} already exists{Colors.ENDC}")

    return file_path


def get_dataset_url(dataset_name: str, prefer_stanford: bool = True) -> str:
    """Get URL for a dataset by name, preferring Stanford over Figshare.

    Parameters
    ----------
    dataset_name:str
        Dataset key in ``DATA_DOWNLOAD_LINK_DICT``.
    prefer_stanford:bool
        Whether Stanford mirror is preferred over Figshare mirror.

    Returns
    -------
    str
        Download URL for the requested dataset.

    Raises:
        ValueError: If dataset name is not found.
    """
    if dataset_name not in DATA_DOWNLOAD_LINK_DICT:
        raise ValueError(f"Dataset '{dataset_name}' not found in available datasets")

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


@register_function(
    aliases=['get_adata', '加载AnnData', 'download and load adata'],
    category="datasets",
    description="Download a dataset file and load it into AnnData with automatic format detection.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.get_adata(url, filename="dataset.h5ad")'],
    related=['datasets.download_data_requests', 'datasets.download_data']
)
def get_adata(url: str, filename: Optional[str] = None) -> Optional[AnnData]:
    """Download example data to local folder.

    Parameters
    ----------
    url:str
        Download URL of an ``.h5ad`` or ``.loom`` dataset file.
    filename:Optional[str]
        Local filename used for caching.

    Returns
    -------
    Optional[AnnData]
        Loaded AnnData object; ``None`` if loading fails.
    """

    try:
        file_path = download_data_requests(url, filename)
        print(f"{Colors.CYAN} Loading data from {file_path}{Colors.ENDC}")

        if Path(file_path).suffixes[-1][1:] == "loom":
            adata = read_loom(filename=file_path)
        elif Path(file_path).suffixes[-1][1:] == "h5ad":
            adata = read_h5ad(filename=file_path)
        else:
            print(f"{Colors.FAIL}{EMOJI['error']} REPORT THIS: Unknown filetype ({file_path}){Colors.ENDC}")
            return None

        adata.var_names_make_unique()
        print(f"{Colors.GREEN}{EMOJI['done']} Successfully loaded: {adata.n_obs} cells × {adata.n_vars} genes{Colors.ENDC}")

    except OSError:
        # Usually occurs when download is stopped before completion then attempted again.
        file_path = os.path.join('./data', filename)
        print(f"{Colors.WARNING}{EMOJI['warning']} Corrupted file. Deleting {file_path} then redownloading...{Colors.ENDC}")
        # Half-downloaded file cannot be read due to corruption so it's better to delete it.
        # Potential issue: user have a file with duplicate name but is not sample data (this will overwrite file).
        try:
            os.remove(file_path)
        except:
            pass
        adata = get_adata(url, filename)
    except Exception as e:
        print(f"{Colors.FAIL}{EMOJI['error']} REPORT THIS: {str(e)}{Colors.ENDC}")
        adata = None

    return adata


@register_function(
    aliases=['download_data_requests', 'requests下载数据', 'robust downloader'],
    category="datasets",
    description="Download dataset files with HTTP headers and streaming to avoid mirror access failures.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['path = ov.datasets.download_data_requests(url, file_path="pbmc3k.h5ad")'],
    related=['datasets.download_data', 'datasets.get_adata']
)
def download_data_requests(url: str, file_path: Optional[str] = None, dir: str = "./data") -> str:
    """Download data with custom headers to reduce HTTP 403 failures.

    Parameters
    ----------
    url:str
        Source URL of the dataset file.
    file_path:Optional[str]
        Target filename. If ``None``, the basename of ``url`` is used.
    dir:str
        Output directory where file is stored.

    Returns
    -------
    str
        Local path to downloaded (or cached) file.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)

    file_name = os.path.basename(url) if file_path is None else file_path
    file_path = os.path.join(dir, file_name)

    if os.path.exists(file_path):
        print(f"{Colors.WARNING}{EMOJI['warning']} File {file_path} already exists{Colors.ENDC}")
        return file_path

    print(f"{Colors.BLUE}{EMOJI['start']} Downloading data to {file_path}...{Colors.ENDC}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://cf.10xgenomics.com/",
    }

    try:
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('Content-Length', 0))
            chunk_size = 8192
            
            with open(file_path, 'wb') as f:
                with tqdm(
                    total=total_size, 
                    unit='B', 
                    unit_scale=True, 
                    desc=f"{Colors.GREEN}Downloading{Colors.ENDC}",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                    colour='blue'
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        print(f"{Colors.GREEN}{EMOJI['done']} Download completed{Colors.ENDC}")
                            
    except Exception as e:
        print(f"{Colors.FAIL}{EMOJI['error']} Download failed: {e}{Colors.ENDC}")
        raise

    return file_path


# add our toy sample data
@register_function(
    aliases=['gillespie', 'gillespie_dataset', '占位数据集_gillespie'],
    category="datasets",
    description="Placeholder loader for Gillespie simulation dataset (currently not implemented).",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['ov.datasets.gillespie()'],
    related=['datasets.toggleswitch', 'datasets.krumsiek11']
)
def gillespie():
    """TODO: add data here"""
    pass


@register_function(
    aliases=['hl60', 'hl60_dataset', '占位数据集_hl60'],
    category="datasets",
    description="Placeholder loader for HL60 dataset (currently not implemented).",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['ov.datasets.hl60()'],
    related=['datasets.hematopoiesis', 'datasets.pbmc3k']
)
def hl60():
    """TODO: add data here"""
    pass


@register_function(
    aliases=['nascseq', 'nascent_seq', '占位数据集_nascseq'],
    category="datasets",
    description="Placeholder loader for NASC-seq dataset (currently not implemented).",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['ov.datasets.nascseq()'],
    related=['datasets.scslamseq', 'datasets.scifate']
)
def nascseq():
    """TODO: add data here"""
    pass


@register_function(
    aliases=['scslamseq', 'scSLAM-seq', '占位数据集_scslamseq'],
    category="datasets",
    description="Placeholder loader for scSLAM-seq dataset (currently not implemented).",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['ov.datasets.scslamseq()'],
    related=['datasets.nascseq', 'datasets.scifate']
)
def scslamseq():
    """TODO: add data here"""
    pass


@register_function(
    aliases=['scifate', 'sci-fate', '占位数据集_scifate'],
    category="datasets",
    description="Placeholder loader for sci-fate dataset (currently not implemented).",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['ov.datasets.scifate()'],
    related=['datasets.nascseq', 'datasets.scslamseq']
)
def scifate():
    """TODO: add data here"""
    pass


@register_function(
    aliases=['神经元剪接数据', 'scnt_seq_neuron_splicing', 'neuron_splicing_dataset'],
    category="datasets",
    description="Load scNT-seq neuron splicing dataset for RNA velocity/splicing dynamics analysis.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.scnt_seq_neuron_splicing()'],
    related=['datasets.scnt_seq_neuron_labeling', 'datasets.get_adata']
)
def scnt_seq_neuron_splicing(
    filename: str = "neuron_splicing.h5ad",
) -> AnnData:
    """The neuron splicing data is from Qiu, et al (2020).

    This data consists of 44,021 genes across 13,476 cells.
    """
    url = get_dataset_url("neuron_splicing")
    adata = get_adata(url, filename)

    return adata


@register_function(
    aliases=['神经元标记数据', 'scnt_seq_neuron_labeling', 'neuron_labeling_dataset'],
    category="datasets",
    description="Load scNT-seq neuron labeling dataset for RNA labeling kinetics analysis.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.scnt_seq_neuron_labeling()'],
    related=['datasets.scnt_seq_neuron_splicing', 'datasets.get_adata']
)
def scnt_seq_neuron_labeling(
    filename: str = "neuron_labeling.h5ad",
) -> AnnData:
    """The neuron splicing data is from Qiu, et al (2020).

    This data consists of 24, 078 genes across 3,060 cells.
    """
    url = get_dataset_url("neuron_labeling")
    adata = get_adata(url, filename)

    return adata


# 10x Genomics PBMC CITE-seq runs (TotalSeq-B), filtered for doublets and outliers
# as in the totalVI paper (Gayoso et al. 2021); the same files scvi-tools'
# ``pbmcs_10x_cite_seq`` reads.  RNA counts in X (dense float32), ADT counts in
# obsm['protein_expression'] and protein names in uns['protein_names'].
CITE_SEQ_PBMC_FILES = (
    ("PBMC10k", "pbmc_10k_protein_v3.h5ad",
     "https://exampledata.scverse.org/scvi-tools/pbmc_10k_protein_v3.h5ad"),
    ("PBMC5k", "pbmc_5k_protein_v3.h5ad",
     "https://exampledata.scverse.org/scvi-tools/pbmc_5k_protein_v3.h5ad"),
)


def _cite_seq_run(path: str, batch: str) -> AnnData:
    """One CITE-seq run as AnnData: CSR counts, ADT counts as a DataFrame."""
    from scipy import sparse
    a = read_h5ad(path)
    names = [str(n) for n in np.asarray(a.uns.pop("protein_names"))]
    a.X = sparse.csr_matrix(np.asarray(a.X, dtype=np.float32)) \
        if not sparse.issparse(a.X) else a.X.tocsr().astype(np.float32)
    a.obsm["protein_expression"] = pd.DataFrame(
        np.asarray(a.obsm["protein_expression"], dtype=np.float32),
        index=a.obs_names, columns=names)
    a.obs["batch"] = batch
    return a


@register_function(
    aliases=['cite_seq', 'CITE-seq', 'pbmcs_10x_cite_seq', 'pbmc_cite_seq', 'CITE-seq数据'],
    category="datasets",
    description="Load the 10x PBMC 5k + 10k CITE-seq runs (RNA counts + 14 ADT proteins, batch column) for totalVI / multimodal tutorials.",
    prerequisites={},
    requires={},
    produces={'layers': ['counts'], 'obsm': ['protein_expression'], 'obs': ['batch']},
    auto_fix='none',
    examples=['adata = ov.datasets.cite_seq()',
              "ov.single.batch_correction(adata, batch_key='batch', methods='totalVI', protein_expression_obsm_key='protein_expression')"],
    related=['datasets.pbmc3k', 'datasets.pbmc8k', 'vi.TOTALVI']
)
def cite_seq(protein_join: str = "inner", dir: str = "./data") -> AnnData:
    """10x Genomics PBMC CITE-seq (5k + 10k runs, TotalSeq-B antibodies).

    The two public 10x runs ``pbmc_10k_protein_v3`` and ``5k_pbmc_protein_v3``
    after the doublet / outlier filtering of the totalVI paper — the same data
    as scvi-tools' ``scvi.data.pbmcs_10x_cite_seq`` (no scvi-tools needed).
    Files are downloaded once to ``dir`` and read from there afterwards.

    Parameters
    ----------
    protein_join : str
        ``'inner'`` keeps the 14 antibodies measured in both runs; ``'outer'``
        keeps all 29 (absent proteins are 0 in the run that lacks them —
        ``ov.vi.TOTALVI`` treats them as missing in that batch).
    dir : str
        Download / cache directory.

    Returns
    -------
    AnnData
        10 849 cells x 15 792 genes (genes shared by both runs):
        ``X`` and ``layers['counts']`` raw RNA counts (CSR float32);
        ``obsm['protein_expression']`` ADT counts (cells x proteins DataFrame);
        ``uns['protein_names']``; ``obs['batch']`` in {``'PBMC10k'``, ``'PBMC5k'``}.
        Cell names get a ``-PBMC10k`` / ``-PBMC5k`` suffix (the barcodes overlap).

    References
    ----------
    Gayoso A, Steier Z, Lopez R, et al. Joint probabilistic modeling of
    single-cell multi-omic data with totalVI. Nat Methods 18, 272-282 (2021).
    """
    import anndata
    if protein_join not in ("inner", "outer"):
        raise ValueError("protein_join must be 'inner' or 'outer'")
    print(f"{Colors.HEADER}Loading 10x PBMC CITE-seq (5k + 10k){Colors.ENDC}")
    runs = [_cite_seq_run(download_data_requests(url, fname, dir=dir), batch)
            for batch, fname, url in CITE_SEQ_PBMC_FILES]
    genes = runs[0].var_names.intersection(runs[1].var_names)
    runs = [a[:, genes] for a in runs]
    adata = anndata.concat(runs, join=protein_join,
                           keys=[a.obs["batch"].iloc[0] for a in runs], index_unique="-")
    pro = adata.obsm["protein_expression"].fillna(0).astype(np.float32)
    adata.obsm["protein_expression"] = pro
    adata.uns["protein_names"] = np.asarray(pro.columns, dtype=object)
    adata.obs["batch"] = pd.Categorical(adata.obs["batch"],
                                        categories=[b for b, _, _ in CITE_SEQ_PBMC_FILES])
    adata.layers["counts"] = adata.X.copy()
    print(f"{Colors.GREEN}{EMOJI['done']} {adata.n_obs} cells x {adata.n_vars} genes, "
          f"{pro.shape[1]} proteins{Colors.ENDC}")
    return adata


@register_function(
    aliases=['zebrafish', '斑马鱼数据', 'zebrafish_dataset'],
    category="datasets",
    description="Load zebrafish single-cell dataset for developmental trajectory studies.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.zebrafish()'],
    related=['datasets.dentate_gyrus', 'datasets.hematopoiesis']
)
def zebrafish(
    filename: str = "zebrafish.h5ad",
) -> AnnData:
    """The zebrafish is from Saunders, et al (2019).

    This data consists of 16,940 genes across 4,181 cells.
    """
    url = get_dataset_url("zebrafish")
    adata = get_adata(url, filename)

    return adata


@register_function(
    aliases=['dentate_gyrus', '齿状回数据', 'DG_loom_dataset'],
    category="datasets",
    description="Load Dentate Gyrus loom dataset commonly used in RNA velocity analyses.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.dentate_gyrus()'],
    related=['datasets.dentate_gyrus_scvelo', 'datasets.get_adata']
)
def dentate_gyrus(
    url: str = "http://pklab.med.harvard.edu/velocyto/DentateGyrus/DentateGyrus.loom",
    filename: Optional[str] = None,
) -> AnnData:
    """The Dentate Gyrus dataset used in https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb.

    This data consists of 27,998 genes across 18,213 cells.
    Note this one http://pklab.med.harvard.edu/velocyto/DG1/10X43_1.loom: a subset of the above data.
    """
    adata = get_adata(url, filename)

    return adata


@register_function(
    aliases=['bone_marrow', '骨髓数据', 'bone_marrow_dataset'],
    category="datasets",
    description="Load bone marrow single-cell dataset for hematopoietic analysis.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.bone_marrow()'],
    related=['datasets.hematopoiesis', 'datasets.bm']
)
def bone_marrow(
    filename: str = "bone_marrow.h5ad",
) -> AnnData:
    """The bone marrow dataset used in

    This data consists of 27,876 genes across 5,780 cells.
    """
    url = get_dataset_url("bone_marrow")
    adata = get_adata(url, filename)

    return adata


@register_function(
    aliases=['haber', 'Haber_dataset', '小肠上皮数据'],
    category="datasets",
    description="Load Haber et al. intestinal epithelium dataset with cell-cycle gene annotations.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.haber()'],
    related=['datasets.hg_forebrain_glutamatergic', 'datasets.get_adata']
)
def haber(
    url: str = "http://pklab.med.harvard.edu/velocyto/Haber_et_al/Haber_et_al.loom",
    filename: Optional[str] = None,
) -> AnnData:
    """The Haber dataset used in https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/Haber_et_al.ipynb

    This data consists of 27,998 genes across 7,216 cells.
    """
    adata = get_adata(url, filename)
    urlretrieve(
        "http://pklab.med.harvard.edu/velocyto/Haber_et_al/goatools_cellcycle_genes.txt",
        "data/goatools_cellcycle_genes.txt",
    )
    cell_cycle_genes = open("data/goatools_cellcycle_genes.txt").read().split()
    adata.var.loc[:, "cell_cycle_genes"] = adata.var.index.isin(cell_cycle_genes)

    return adata


@register_function(
    aliases=['hg_forebrain_glutamatergic', '前脑谷氨酸能数据', 'hgForebrainGlut'],
    category="datasets",
    description="Load human forebrain glutamatergic loom dataset for velocity benchmark analyses.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.hg_forebrain_glutamatergic()'],
    related=['datasets.haber', 'datasets.chromaffin']
)
def hg_forebrain_glutamatergic(
    url: str = "http://pklab.med.harvard.edu/velocyto/hgForebrainGlut/hgForebrainGlut.loom",
    filename: Optional[str] = None,
) -> AnnData:
    """The hgForebrainGlutamatergic dataset used in https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/hgForebrainGlutamatergic.ipynb

    This data consists of 32,738 genes across 1,720 cells.
    """
    adata = get_adata(url, filename)
    urlretrieve(
        "http://pklab.med.harvard.edu/velocyto/Haber_et_al/goatools_cellcycle_genes.txt",
        "data/goatools_cellcycle_genes.txt",
    )
    cell_cycle_genes = open("data/goatools_cellcycle_genes.txt").read().split()
    adata.var.loc[:, "cell_cycle_genes"] = adata.var.index.isin(cell_cycle_genes)

    return adata


@register_function(
    aliases=['chromaffin', '嗜铬细胞数据', 'chromaffin_dataset'],
    category="datasets",
    description="Load chromaffin-cell loom dataset used in velocity and lineage studies.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.chromaffin()'],
    related=['datasets.bm', 'datasets.get_adata']
)
def chromaffin(
    filename: str = "onefilepercell_A1_unique_and_others_J2CH1.loom",
) -> AnnData:  #
    """The chromaffin dataset used in http://pklab.med.harvard.edu/velocyto/notebooks/R/chromaffin2.nb.html

    This data consists of 32,738 genes across 1,720 cells.
    """

    url = get_dataset_url("onefilepercell_A1_unique_and_others_J2CH1")
    adata = get_adata(url, filename)

    adata.var_names_make_unique()
    return adata


@register_function(
    aliases=['bm', 'mouse_bm_dataset', '小鼠骨髓loom'],
    category="datasets",
    description="Load mouse bone marrow loom dataset for differentiation trajectory analysis.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.bm()'],
    related=['datasets.bone_marrow', 'datasets.hematopoiesis']
)
def bm(
    url: str = "http://pklab.med.harvard.edu/velocyto/mouseBM/SCG71.loom",
    filename: Optional[str] = None,
) -> AnnData:
    """The BM dataset used in http://pklab.med.harvard.edu/velocyto/notebooks/R/SCG71.nb.html

    This data consists of 24,421genes across 6,667 cells.
    """

    adata = get_adata(url, filename)

    return adata


@register_function(
    aliases=['胰腺内分泌发生数据', 'pancreatic_endocrinogenesis', 'pancreas development dataset'],
    category="datasets",
    description="Load or download the pancreatic endocrinogenesis reference dataset for developmental trajectory and lineage analyses.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.pancreatic_endocrinogenesis()'],
    related=['datasets.download_data', 'utils.cal_paga', 'pp.preprocess']
)
def pancreatic_endocrinogenesis(
    url: str = "https://github.com/theislab/scvelo_notebooks/raw/master/data/Pancreas/endocrinogenesis_day15.h5ad",
    filename: Optional[str] = None,
) -> AnnData:
    """Pancreatic endocrinogenesis. Data from scvelo.

    Pancreatic epithelial and Ngn3-Venus fusion (NVF) cells during secondary transition / embryonic day 15.5.
    https://dev.biologists.org/content/146/12/dev173849
    """

    adata = get_adata(url, filename)

    return adata

@register_function(
    aliases=['pancreas_cellrank', 'cellrank_pancreas', '胰腺cellrank数据'],
    category="datasets",
    description="Load CellRank pancreas benchmark dataset for fate and lineage analyses.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.pancreas_cellrank()'],
    related=['datasets.pancreatic_endocrinogenesis', 'datasets.get_adata']
)
def pancreas_cellrank(
    url: str = "https://github.com/theislab/scvelo_notebooks/raw/master/data/Pancreas/endocrinogenesis_day15.h5ad",
    filename: str = "pancreas_cellrank.h5ad",
) -> AnnData:
    """The pancreas cellrank dataset used in https://github.com/theislab/scvelo_notebooks/tree/master/data/Pancreas.

    This data consists of 13,913 genes across 2,930 cells.

    Note: The previous figshare URL (files/25060877) returned a 2KB HTML error
    page instead of the dataset, causing an infinite redownload loop.
    Switched the default to the scvelo_notebooks GitHub raw URL, which is the
    upstream source that scvelo / cellrank also use.
    """
    adata = get_adata(url, filename)
    return adata



@register_function(
    aliases=['dentate_gyrus_scvelo', 'DG_scvelo', '齿状回scvelo数据'],
    category="datasets",
    description="Load processed Dentate Gyrus dataset from scVelo examples.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.dentate_gyrus_scvelo()'],
    related=['datasets.dentate_gyrus', 'datasets.scnt_seq_neuron_splicing']
)
def dentate_gyrus_scvelo(
    filename: str = "dentategyrus_scv.h5ad",
) -> AnnData:
    """The Dentate Gyrus dataset used in https://github.com/theislab/scvelo_notebooks/tree/master/data/DentateGyrus.

    This data consists of 13,913 genes across 2,930 cells. Note this dataset is the same processed dataset from the
    excellent scVelo package, which is a subset of the DentateGyrus dataset.
    """
    url = get_dataset_url("dentategyrus_scv")
    adata = get_adata(url, filename)

    return adata


@register_function(
    aliases=['sceu_seq_rpe1', 'rpe1_dataset', 'scEU_seq_RPE1'],
    category="datasets",
    description="Load scEU-seq RPE1 dataset for RNA labeling kinetics analyses.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.sceu_seq_rpe1()'],
    related=['datasets.sceu_seq_organoid', 'datasets.get_adata']
)
def sceu_seq_rpe1(
    filename: str = "rpe1.h5ad",
):
    """Download rpe1 dataset from Battich, et al (2020) via a figshare link.

    This data consists of 13,913 genes across 2,930 cells.
    """
    print(f"{Colors.HEADER}{EMOJI['start']} Downloading scEU_seq data{Colors.ENDC}")
    url = get_dataset_url("rpe1")
    adata = get_adata(url, filename)
    return adata


@register_function(
    aliases=['sceu_seq_organoid', 'organoid_dataset', 'scEU_seq_organoid'],
    category="datasets",
    description="Load scEU-seq organoid dataset for temporal transcription dynamics studies.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.sceu_seq_organoid()'],
    related=['datasets.sceu_seq_rpe1', 'datasets.get_adata']
)
def sceu_seq_organoid(
    filename: str = "organoid.h5ad",
):
    """Download organoid dataset from Battich, et al (2020) via a figshare link.

    This data consists of 9,157 genes across 3,831 cells.
    """
    print(f"{Colors.HEADER}{EMOJI['start']} Downloading scEU_seq data{Colors.ENDC}")
    url = get_dataset_url("organoid")
    adata = get_adata(url, filename)
    return adata


@register_function(
    aliases=['hematopoiesis', '造血数据', 'hematopoiesis_processed'],
    category="datasets",
    description="Load processed hematopoiesis dataset for trajectory and lineage benchmarking.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.hematopoiesis()'],
    related=['datasets.hematopoiesis_raw', 'datasets.bone_marrow']
)
def hematopoiesis(
    filename: str = "hematopoiesis.h5ad",
) -> AnnData:
    """Processed dataset originally from https://pitt.box.com/v/hematopoiesis-processed."""
    print(f"{Colors.HEADER}🧬 Downloading processed hematopoiesis adata{Colors.ENDC}")
    url = get_dataset_url("hematopoiesis")
    adata = get_adata(url, filename)
    return adata


@register_function(
    aliases=['multi_brain_5k', '10x_multiome_brain_5k', '小鼠脑multiome'],
    category="datasets",
    description="Download and assemble 10x E18 mouse brain 5k multiome dataset resources.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['mdata = ov.datasets.multi_brain_5k()'],
    related=['datasets.download_data_requests', 'multi.read_10x_multiome_h5']
)
def multi_brain_5k():
    """Processed dataset originally from https://pitt.box.com/v/hematopoiesis-processed."""
    print(f"{Colors.HEADER}🧠 Downloading raw Fresh Embryonic E18 Mouse Brain (5k){Colors.ENDC}")
    print(f"{Colors.CYAN}Epi Multiome ATAC + Gene Expression dataset{Colors.ENDC}")

    h5_url='https://cf.10xgenomics.com/samples/cell-arc/1.0.0/e18_mouse_brain_fresh_5k/e18_mouse_brain_fresh_5k_filtered_feature_bc_matrix.h5'
    fragment_url='https://cf.10xgenomics.com/samples/cell-arc/1.0.0/e18_mouse_brain_fresh_5k/e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz'
    fragment_tbi_url='https://cf.10xgenomics.com/samples/cell-arc/1.0.0/e18_mouse_brain_fresh_5k/e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz.tbi'
    peak_annotation_url='https://cf.10xgenomics.com/samples/cell-arc/1.0.0/e18_mouse_brain_fresh_5k/e18_mouse_brain_fresh_5k_atac_peak_annotation.tsv'
    velocyto_url=get_dataset_url("10X_multiome_mouse_brain")
    anontation_url=get_dataset_url("cell_annotations")

    h5_path = download_data_requests(h5_url, 'filtered_feature_bc_matrix.h5', dir='./data/multi_brain_5k')
    fragment_path = download_data_requests(fragment_url, 'fragments.tsv.gz', dir='./data/multi_brain_5k')
    fragment_tbi_path = download_data_requests(fragment_tbi_url, 'fragments.tsv.gz.tbi', dir='./data/multi_brain_5k')
    peak_annotation_path = download_data_requests(peak_annotation_url, 'peak_annotation.tsv', dir='./data/multi_brain_5k')
    velocyto_path = download_data_requests(velocyto_url, '10X_multiome_mouse_brain.loom', dir='./data/multi_brain_5k/velocyto')
    annotation_path = download_data_requests(anontation_url, 'cell_annotations.tsv', dir='./data/multi_brain_5k')

    analysis_url='https://cf.10xgenomics.com/samples/cell-arc/1.0.0/e18_mouse_brain_fresh_5k/e18_mouse_brain_fresh_5k_analysis.tar.gz'
    analysis_path = download_data_requests(analysis_url, 'e18_mouse_brain_fresh_5k_analysis.tar.gz', dir='./data/multi_brain_5k')
    # Extract the tar.gz file
    import tarfile
    with tarfile.open(analysis_path, "r:gz") as tar:
        tar.extractall(path='./data/multi_brain_5k/')
    # Remove the tar.gz file after extraction
    os.remove(analysis_path)

    try:
        from ..multi import read_10x_multiome_h5
        mdata=read_10x_multiome_h5(multiome_base_path='./data/multi_brain_5k',
                                                       rna_splicing_loom='velocyto/10X_multiome_mouse_brain.loom',
                                                      cellranger_path_structure=False)
        cell_annot = pd.read_csv('./data/multi_brain_5k/cell_annotations.tsv', sep='\t', index_col=0)
        cell_annot.index=[i.split('-')[0] for i in cell_annot.index]
        ret_index=list(set(cell_annot.index) & set(mdata.obs.index))
        cell_annot=cell_annot.loc[ret_index]
        mdata.update()
        mdata = mdata[ret_index]
        mdata['rna'].obs['celltype'] = cell_annot['celltype'].tolist()
        return mdata
    except ImportError:
        print(f"{Colors.WARNING}{EMOJI['warning']} multi module not available, returning None{Colors.ENDC}")
        return None


@register_function(
    aliases=['hematopoiesis_raw', '造血原始数据', 'hematopoiesis_unprocessed'],
    category="datasets",
    description="Load raw hematopoiesis dataset for custom preprocessing pipelines.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.hematopoiesis_raw()'],
    related=['datasets.hematopoiesis', 'pp.preprocess']
)
def hematopoiesis_raw(
    filename: str = "hematopoiesis_raw.h5ad",
) -> AnnData:
    """Processed dataset originally from https://pitt.box.com/v/hematopoiesis-processed."""
    print(f"{Colors.HEADER}🧬 Downloading raw hematopoiesis adata{Colors.ENDC}")
    url = get_dataset_url("hematopoiesis_raw")
    adata = get_adata(url, filename)
    return adata


@register_function(
    aliases=['human_tfs', '人类转录因子列表', 'human_transcription_factors'],
    category="datasets",
    description="Download curated human transcription factor table.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['tfs = ov.datasets.human_tfs()'],
    related=['single.get_celltype_marker', 'bulk.pyDEG']
)
def human_tfs(
    filename: str = "human_tfs.txt",
) -> pd.DataFrame:
    """Download human transcription factors."""
    url = get_dataset_url("human_tfs")
    file_path = download_data_requests(url, filename)
    tfs = pd.read_csv(file_path, sep="\t")
    return tfs


# Scanpy-inspired datasets with dynamo pattern
@register_function(
    aliases=['blobs', 'gaussian_blobs', '模拟聚类数据'],
    category="datasets",
    description="Generate Gaussian blob synthetic AnnData for clustering benchmarks.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.blobs(n_centers=5, n_observations=1000)'],
    related=['datasets.create_mock_dataset', 'datasets.krumsiek11']
)
def blobs(
    n_variables: int = 11,
    n_centers: int = 5,
    cluster_std: float = 1.0,
    n_observations: int = 640,
    random_state: int = 0,
) -> AnnData:
    """Gaussian Blobs dataset.

    Parameters
    ----------
    n_variables:int
        Feature dimension of generated data.
    n_centers:int
        Number of Gaussian cluster centers.
    cluster_std:float
        Standard deviation of each Gaussian cluster.
    n_observations:int
        Number of synthetic observations (cells).
    random_state:int
        Random seed passed to sklearn data generation.

    Returns
    -------
    AnnData
        Synthetic AnnData with ``obs['blobs']`` storing cluster labels.
    """
    print(f"{Colors.HEADER}🎯 Generating Gaussian Blobs dataset{Colors.ENDC}")
    
    try:
        import sklearn.datasets
        
        X, y = sklearn.datasets.make_blobs(
            n_samples=n_observations,
            n_features=n_variables,
            centers=n_centers,
            cluster_std=cluster_std,
            random_state=random_state,
        )
        adata = AnnData(X, obs=dict(blobs=y.astype(str)))
        print(f"{Colors.GREEN}{EMOJI['done']} Generated blobs: {n_observations} cells × {n_variables} features, {n_centers} centers{Colors.ENDC}")
        return adata
    except ImportError:
        print(f"{Colors.WARNING}{EMOJI['warning']} sklearn not available, generating mock blobs{Colors.ENDC}")
        return create_mock_dataset(n_cells=n_observations, n_genes=n_variables, n_cell_types=n_centers, with_clustering=False)


@register_function(
    aliases=['burczynski06', 'UC_CD_bulk', '炎症性肠病bulk数据'],
    category="datasets",
    description="Load Burczynski06 UC/CD PBMC bulk expression dataset.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.burczynski06()'],
    related=['datasets.bhattacherjee', 'datasets.get_adata']
)
def burczynski06(
    url: str = "ftp://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS1nnn/GDS1615/soft/GDS1615_full.soft.gz",
    filename: str = "GDS1615_full.soft.gz",
) -> AnnData:
    """Bulk data with conditions ulcerative colitis (UC) and Crohn's disease (CD).

    The study assesses transcriptional profiles in peripheral blood mononuclear
    cells from 42 healthy individuals, 59 CD patients, and 26 UC patients.

    This data consists of 127 samples × 22283 genes.
    """
    print(f"{Colors.HEADER}🩸 Downloading Burczynski06 UC/CD dataset{Colors.ENDC}")
    adata = get_adata(url, filename)
    return adata


@register_function(
    aliases=['moignard15', 'embryo_hematopoiesis', '小鼠胚胎造血数据'],
    category="datasets",
    description="Load Moignard15 embryo hematopoiesis single-cell qRT-PCR dataset.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.moignard15()'],
    related=['datasets.paul15', 'datasets.hematopoiesis']
)
def moignard15(
    url: str = "https://static-content.springer.com/esm/art%3A10.1038%2Fnbt.3154/MediaObjects/41587_2015_BFnbt3154_MOESM4_ESM.xlsx",
    filename: str = "nbt.3154-S3.xlsx",
) -> AnnData:
    """Hematopoiesis in early mouse embryos (Moignard et al. 2015).

    The data was obtained using qRT–PCR.
    Contains normalized dCt values with experimental groups:
    "primitive streak" (PS), "neural plate" (NP), "head fold (HF),
    "four somite" blood/GFP⁺ (4SG), and "four somite" endothelial/GFP¯ (4SFG).
    
    This data consists of 3934 cells × 42 genes.
    """
    print(f"{Colors.HEADER}🐭 Downloading Moignard15 mouse embryo hematopoiesis{Colors.ENDC}")
    try:
        adata = get_adata(url, filename)
        if adata is not None:
            # Apply some basic processing similar to scanpy version
            import numpy as np
            # filter out 4 genes as in original processing
            if adata.n_vars > 42:
                gene_subset = ~np.isin(adata.var_names, ["Eif2b1", "Mrpl19", "Polr2a", "Ubc"])
                adata = adata[:, gene_subset].copy()
            
            # Add experimental groups if not present
            if "exp_groups" not in adata.obs.columns and adata.obs_names is not None:
                groups = {"HF": "#D7A83E", "NP": "#7AAE5D", "PS": "#497ABC", "4SG": "#AF353A", "4SFG": "#765099"}
                adata.obs["exp_groups"] = [
                    next((gname for gname in groups if str(sname).startswith(gname)), "Unknown")
                    for sname in adata.obs_names
                ]
                adata.uns["exp_groups_colors"] = list(groups.values())
                adata.uns["iroot"] = 532
            return adata
    except Exception as e:
        print(f"{Colors.WARNING}{EMOJI['warning']} Failed to load Moignard15: {e}{Colors.ENDC}")
        print(f"{Colors.WARNING}🔄 Generating mock hematopoiesis data{Colors.ENDC}")
        return create_mock_dataset(n_cells=3934, n_genes=42, n_cell_types=5, with_clustering=True)


@register_function(
    aliases=['paul15', 'myeloid_progenitors', '髓系祖细胞数据'],
    category="datasets",
    description="Load Paul15 myeloid progenitor dataset with curated lineage labels.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.paul15()'],
    related=['datasets.krumsiek11', 'datasets.moignard15']
)
def paul15(
    url: str = "https://falexwolf.de/data/paul15.h5",
    filename: str = "paul15.h5",
) -> AnnData:
    """Development of Myeloid Progenitors (Paul et al. 2015).

    Non-logarithmized raw data of myeloid progenitor development.
    This data consists of 2730 cells × 3451 genes.
    """
    print(f"{Colors.HEADER}🧬 Downloading Paul15 myeloid progenitors{Colors.ENDC}")
    try:
        import h5py
        file_path = download_data(url, filename)
        
        with h5py.File(file_path, "r") as f:
            X = f["data.debatched"][()].astype(np.float32)
            gene_names = f["data.debatched_rownames"][()].astype(str)
            cell_names = f["data.debatched_colnames"][()].astype(str)
            clusters = f["cluster.id"][()].flatten().astype(int)
            infogenes_names = f["info.genes_strings"][()].astype(str)
            
        # each row corresponds to observation, therefore transpose
        adata = AnnData(X.transpose())
        adata.var_names = gene_names
        adata.obs_names = cell_names
        
        # names reflecting cell type identifications
        cell_type = 6 * ["Ery"]
        cell_type += "MEP Mk GMP GMP DC Baso Baso Mo Mo Neu Neu Eos Lymph".split()
        adata.obs["paul15_clusters"] = [f"{i}{cell_type[i - 1]}" for i in clusters]
        
        # just keep the first of the two equivalent names per gene
        adata.var_names = [gn.split(";")[0] for gn in adata.var_names]
        
        # remove corrupted gene names
        infogenes_names = np.intersect1d(infogenes_names, adata.var_names)
        adata = adata[:, infogenes_names].copy()
        
        adata.uns["iroot"] = 840
        print(f"{Colors.GREEN}{EMOJI['done']} Loaded Paul15: {adata.n_obs} cells × {adata.n_vars} genes{Colors.ENDC}")
        return adata
        
    except Exception as e:
        print(f"{Colors.WARNING}{EMOJI['warning']} Failed to load Paul15: {e}{Colors.ENDC}")
        print(f"{Colors.WARNING}🔄 Generating mock myeloid data{Colors.ENDC}")
        return create_mock_dataset(n_cells=2730, n_genes=3451, n_cell_types=13, with_clustering=True)


@register_function(
    aliases=['pbmc8k', 'pbmc_8k', 'PBMC8K数据集'],
    category="datasets",
    description="Load 10x PBMC 8k dataset for immune-cell method benchmarking.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.pbmc8k()'],
    related=['datasets.pbmc3k', 'datasets.seqfish']
)
def pbmc8k(
    url: str = "https://stacks.stanford.edu/file/cv694yk7414/pbmc8k.h5ad", 
    filename: str = "pbmc8k.h5ad"
) -> AnnData:
    """PBMC 8k dataset from 10x Genomics.
    """
    print(f"{Colors.HEADER}🩸 Downloading PBMC 8k dataset{Colors.ENDC}")
    url = get_dataset_url("pbmc8k")
    adata = get_adata(url, filename)
    return adata

@register_function(
    aliases=['seqfish', 'seqFISH数据', 'spatial_seqfish_dataset'],
    category="datasets",
    description="Load seqFISH spatial transcriptomics dataset for spatial algorithm demos.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.seqfish()'],
    related=['space.svg', 'space.clusters', 'datasets.pbmc8k']
)
def seqfish(
    url: str = "https://stacks.stanford.edu/file/cv694yk7414/seqfish.h5ad", 
    filename: str = "seqfish.h5ad"
) -> AnnData:
    """SeqFish dataset from 10x Genomics.
    """
    print(f"{Colors.HEADER}🐟 Downloading SeqFish dataset{Colors.ENDC}")
    url = get_dataset_url("seqfish")
    adata = get_adata(url, filename)
    return adata


@register_function(
    aliases=['toggleswitch', 'toggle_switch_simulation', '开关模型模拟数据'],
    category="datasets",
    description="Generate two-gene toggle-switch simulation data.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.toggleswitch()'],
    related=['datasets.krumsiek11', 'datasets.blobs']
)
def toggleswitch(
    filename: str = "toggleswitch.txt"
) -> AnnData:
    """Simulated toggleswitch data.

    Data obtained simulating a simple toggleswitch system.
    This data consists of 200 cells × 2 genes.
    """
    print(f"{Colors.HEADER}⚖️ Loading toggleswitch simulation data{Colors.ENDC}")
    try:
        # Try to create simple toggleswitch-like data
        np.random.seed(0)
        n_cells = 200
        
        # Simple toggleswitch: two genes with anti-correlated expression
        t = np.linspace(0, 4*np.pi, n_cells)
        gene1 = np.maximum(0, np.sin(t) + 0.1*np.random.randn(n_cells))
        gene2 = np.maximum(0, np.cos(t) + 0.1*np.random.randn(n_cells))
        
        X = np.column_stack([gene1, gene2])
        adata = AnnData(X)
        adata.var_names = ['Gene1', 'Gene2'] 
        adata.obs_names = [f'Cell_{i+1}' for i in range(n_cells)]
        adata.uns["iroot"] = 0
        
        print(f"{Colors.GREEN}{EMOJI['done']} Generated toggleswitch: {n_cells} cells × 2 genes{Colors.ENDC}")
        return adata
        
    except Exception as e:
        print(f"{Colors.WARNING}{EMOJI['warning']} Error generating toggleswitch: {e}{Colors.ENDC}")
        return create_mock_dataset(n_cells=200, n_genes=2, n_cell_types=2, with_clustering=False)


@register_function(
    aliases=['krumsiek11', 'myeloid_simulation', '髓系分化模拟数据'],
    category="datasets",
    description="Generate Krumsiek11-inspired myeloid differentiation simulation dataset.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.krumsiek11()'],
    related=['datasets.toggleswitch', 'datasets.paul15']
)
def krumsiek11() -> AnnData:
    """Simulated myeloid progenitors (Krumsiek et al. 2011).

    The literature-curated boolean network was used to simulate the data. 
    It describes development to four cell fates:
    "monocyte" (Mo), "erythrocyte" (Ery), "megakaryocyte" (Mk) and "neutrophil" (Neu).
    
    This data consists of 640 cells × 11 genes.
    """
    print(f"{Colors.HEADER}🧬 Loading Krumsiek11 myeloid progenitor simulation{Colors.ENDC}")
    try:
        # Generate simple simulated myeloid development data
        np.random.seed(42)
        n_cells = 640
        n_genes = 11
        
        # Create differentiation trajectory
        X = np.random.lognormal(0, 1, (n_cells, n_genes)).astype(np.float32)
        
        # Add structure for different cell fates
        cell_type = pd.array(["progenitor"]).repeat(n_cells)
        cell_type[80:160] = "Mo"     # Monocyte
        cell_type[240:320] = "Ery"   # Erythrocyte  
        cell_type[400:480] = "Mk"    # Megakaryocyte
        cell_type[560:640] = "Neu"   # Neutrophil
        
        adata = AnnData(X)
        adata.var_names = [f'Gene_{i+1}' for i in range(n_genes)]
        adata.obs_names = [f'Cell_{i+1}' for i in range(n_cells)]
        adata.obs["cell_type"] = cell_type
        
        # Add trajectory information
        adata.uns["iroot"] = 0
        fate_labels = {0: "Stem", 159: "Mo", 319: "Ery", 459: "Mk", 619: "Neu"}
        adata.uns["highlights"] = fate_labels
        
        print(f"{Colors.GREEN}{EMOJI['done']} Generated Krumsiek11: {n_cells} cells × {n_genes} genes{Colors.ENDC}")
        return adata
        
    except Exception as e:
        print(f"{Colors.WARNING}{EMOJI['warning']} Error generating Krumsiek11: {e}{Colors.ENDC}")
        return create_mock_dataset(n_cells=640, n_genes=11, n_cell_types=5, with_clustering=True)


# Mock datasets (following dynamo pattern)
@register_function(
    aliases=['create_mock_dataset', 'mock_adata', '模拟单细胞数据'],
    category="datasets",
    description="Create configurable mock single-cell AnnData for pipeline testing.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.create_mock_dataset(n_cells=1000, n_cell_types=5)'],
    related=['datasets.pbmc3k', 'datasets.blobs']
)
def create_mock_dataset(
    n_cells: int = 2000,
    n_genes: int = 1500, 
    n_cell_types: int = 6,
    with_clustering: bool = False,
    random_state: int = 42
) -> AnnData:
    """
    Create a mock single-cell dataset for testing statistical functions.
    
    Parameters
    ----------
    n_cells:int
        Number of cells simulated in mock dataset.
    n_genes:int
        Number of genes simulated in mock dataset.
    n_cell_types:int
        Number of synthetic cell types.
    with_clustering:bool
        Whether to run lightweight preprocessing and clustering-like fields.
    random_state:int
        Random seed for deterministic dataset generation.

    Returns
    -------
    AnnData
        Simulated AnnData with expression matrix and basic metadata fields.
    """
    
    np.random.seed(random_state)
    
    # Generate mock expression data
    # Create some structure with different expression patterns per cell type
    X = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes)).astype(np.float32)
    
    # Add some structure: different cell types have different expression patterns
    cell_type_labels = np.random.choice(range(n_cell_types), n_cells)
    
    for ct in range(n_cell_types):
        ct_mask = cell_type_labels == ct
        if np.sum(ct_mask) > 0:
            # Make some genes more highly expressed in this cell type
            high_genes_size = min(100, n_genes // 2)  # Use at most half the genes, max 100
            high_genes = np.random.choice(n_genes, size=high_genes_size, replace=False)
            X[ct_mask][:, high_genes] *= np.random.uniform(2, 5)
    
    # Create gene names
    gene_names = [f"Gene_{i+1:04d}" for i in range(n_genes)]
    
    # Create cell names  
    cell_names = [f"Cell_{i+1:04d}" for i in range(n_cells)]
    
    # Create AnnData object
    adata = AnnData(X=X)
    adata.var_names = gene_names
    adata.obs_names = cell_names
    
    # Add mock metadata
    adata.obs['cell_type'] = [f'CellType_{i+1}' for i in cell_type_labels]
    
    # Add sample information
    n_samples = max(2, n_cell_types // 2)
    sample_labels = np.random.choice([f'Sample_{i+1}' for i in range(n_samples)], n_cells)
    adata.obs['sample_id'] = sample_labels
    
    # Add conditions
    conditions = np.random.choice(['Control', 'Treatment'], n_cells, p=[0.5, 0.5])
    adata.obs['condition'] = conditions
    
    # Add tissue types for odds ratio testing
    tissues = np.random.choice(['Blood', 'Normal', 'Tumor'], n_cells, p=[0.33, 0.33, 0.34])
    adata.obs['tissue'] = tissues
    
    # Add some basic gene information
    adata.var['gene_symbols'] = gene_names
    adata.var['highly_variable'] = np.random.choice([True, False], n_genes, p=[0.2, 0.8])
    
    # Add clustering preprocessing if requested (without scanpy dependency)
    if with_clustering:
        try:
            # Simple normalization (total count normalization + log1p)
            X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
            
            # Total count normalization
            cell_sums = np.sum(X, axis=1, keepdims=True)
            X_norm = X / cell_sums * 1e4
            
            # Log1p transformation
            X_log = np.log1p(X_norm)
            
            # Simple highly variable gene selection (top variance genes)
            gene_vars = np.var(X_log, axis=0)
            n_hvg = min(2000, int(n_genes * 0.1))  # Top 10% or 2000 genes
            top_genes = np.argsort(gene_vars)[::-1][:n_hvg]
            
            # Mark highly variable genes
            adata.var['highly_variable'] = False
            adata.var.iloc[top_genes, adata.var.columns.get_loc('highly_variable')] = True
            
            # Store raw data and subset to HVGs
            adata.raw = adata.copy()
            adata = adata[:, adata.var.highly_variable].copy()
            
            # Update X with normalized data
            adata.X = X_log[np.ix_(np.arange(n_cells), top_genes)]
            
            # Simple scaling (z-score, clipped at 10)
            X_scaled = (adata.X - np.mean(adata.X, axis=0)) / (np.std(adata.X, axis=0) + 1e-8)
            X_scaled = np.clip(X_scaled, -10, 10)
            adata.X = X_scaled
            
            # Simple PCA (using SVD)
            n_comps = min(50, adata.n_vars - 1, adata.n_obs - 1, 50)
            if n_comps > 0:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=n_comps, random_state=42)
                X_pca = pca.fit_transform(X_scaled)
                adata.obsm['X_pca'] = X_pca
                
                # Simple UMAP-like embedding (just first 2 PCs with some noise)
                umap_coords = X_pca[:, :2] + np.random.normal(0, 0.1, (n_cells, 2))
                adata.obsm['X_umap'] = umap_coords
            
            # Simple clustering (just assign random clusters based on cell types)
            n_clusters = min(8, n_cell_types + 2)
            cluster_labels = np.random.choice(range(n_clusters), n_cells)
            adata.obs['leiden'] = [str(i) for i in cluster_labels]
            
        except Exception as e:
            warnings.warn(f"Mock clustering preprocessing failed: {e}")
    
    print(f"{Colors.GREEN}{EMOJI['done']} Created mock dataset: {n_cells} cells, {n_genes} genes, {n_cell_types} cell types{Colors.ENDC}")
    return adata


@register_function(
    aliases=['DeCOV bulk COVID', 'decov_bulk_covid_bulk', 'covid bulk dataset'],
    category="datasets",
    description="Load the bulk RNA-seq cohort used in DeCOV spatial deconvolution examples for cross-platform reference mapping.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['bulk_df = ov.datasets.decov_bulk_covid_bulk()'],
    related=['datasets.decov_bulk_covid_single', 'space.Deconvolution']
)
def decov_bulk_covid_bulk(
    filename: str = "COVID_PBMC_bulk.h5ad"
) -> AnnData:
    """COVID-19 PBMC bulk data from Decov et al. 2020.

    This data consists of 10,000 cells × 15,000 genes.
    """
    print(f"{Colors.HEADER}🧬 Loading COVID-19 PBMC bulk data{Colors.ENDC}")
    url = get_dataset_url("COVID_PBMC_bulk")
    adata = get_adata(url, filename)
    return adata

@register_function(
    aliases=['DeCOV single-cell COVID', 'decov_bulk_covid_single', 'covid scRNA dataset'],
    category="datasets",
    description="Load the single-cell reference cohort paired with DeCOV bulk data for cell-type signature derivation and transfer.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata_sc = ov.datasets.decov_bulk_covid_single()'],
    related=['datasets.decov_bulk_covid_bulk', 'space.calculate_gene_signature']
)
def decov_bulk_covid_single(
    filename: str = "COVID_PBMC_single.h5ad"
) -> AnnData:
    """COVID-19 PBMC single-cell data from Decov et al. 2020.

    This data consists of 10,000 cells × 15,000 genes.
    """
    print(f"{Colors.HEADER}🧬 Loading COVID-19 PBMC single-cell data{Colors.ENDC}")
    url = get_dataset_url("COVID_PBMC_single")
    adata = get_adata(url, filename)
    return adata

@register_function(
    aliases=['淋巴结单细胞参考', 'sc_ref_Lymph_Node', 'lymph node reference'],
    category="datasets",
    description="Load a lymph-node single-cell reference dataset for immune-cell annotation and spatial deconvolution benchmarking.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata_ref = ov.datasets.sc_ref_Lymph_Node()'],
    related=['datasets.download_data', 'single.scanpy_cellanno_from_dict']
)
def sc_ref_Lymph_Node(
    url: str = "https://cell2location.cog.sanger.ac.uk/paper/integrated_lymphoid_organ_scrna/RegressionNBV4Torch_57covariates_73260cells_10237genes/sc.h5ad",
    filename: str = "sc_ref_Lymph_Node.h5ad"
) -> AnnData:
    """SC reference data for Lymph Node.
    
    This data consists of 10,000 cells × 15,000 genes.
    """
    print(f"{Colors.HEADER}🧬 Loading SC reference data for Lymph Node{Colors.ENDC}")
    adata = get_adata(url, filename)
    return adata


@register_function(
    aliases=['淋巴结Visium', 'visium_lymph_node', 'V1_Human_Lymph_Node', '10x lymph node visium'],
    category="datasets",
    description="Download and load the 10x V1 Human Lymph Node Visium spatial transcriptomics sample (paired with sc_ref_Lymph_Node for deconvolution benchmarking).",
    prerequisites={},
    requires={},
    produces={'obsm': ['spatial']},
    auto_fix='none',
    examples=['adata_sp = ov.datasets.visium_lymph_node()'],
    related=['datasets.sc_ref_Lymph_Node', 'io.spatial.read_visium', 'space.Deconvolution']
)
def visium_lymph_node(dir: str = "./data") -> AnnData:
    r"""Download + load the 10x V1 Human Lymph Node Visium dataset.

    Bundles the two artefacts the 10x sample page ships separately:

    - ``V1_Human_Lymph_Node_filtered_feature_bc_matrix.h5`` — gene × spot
      counts filtered to in-tissue spots,
    - ``V1_Human_Lymph_Node_spatial.tar.gz`` — tissue image, fiducial
      JSON, and per-spot pixel coordinates.

    Both come from the public 10x CDN
    (https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Human_Lymph_Node/);
    extraction puts them in the layout that ``ov.io.spatial.read_visium``
    expects (``filtered_feature_bc_matrix.h5`` + ``spatial/``).

    Parameters
    ----------
    dir : str
        Local directory under which the dataset is cached (``./data`` by
        default). Re-downloads are skipped when the artefacts already exist.

    Returns
    -------
    AnnData
        Visium AnnData with raw counts on ``X``, spot coordinates on
        ``obsm['spatial']``, and the tissue image / scalefactors registered
        under ``uns['spatial']``.

    Examples
    --------
    >>> adata_sc = ov.datasets.sc_ref_Lymph_Node()
    >>> adata_sp = ov.datasets.visium_lymph_node()
    >>> decov = ov.space.Deconvolution(adata_sc=adata_sc, adata_sp=adata_sp)
    >>> decov.deconvolution(method='RCTD', celltype_key_sc='Subset')
    """
    import tarfile
    print(f"{Colors.HEADER}🧬 Loading 10x V1 Human Lymph Node Visium data{Colors.ENDC}")

    base = "https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Human_Lymph_Node"
    sample_dir = os.path.join(dir, "V1_Human_Lymph_Node")
    os.makedirs(sample_dir, exist_ok=True)

    # 10x labels every artefact with the sample prefix; `read_visium` expects
    # `filtered_feature_bc_matrix.h5` literally, so download the prefixed file
    # straight under that canonical filename.
    h5_path = os.path.join(sample_dir, "filtered_feature_bc_matrix.h5")
    if not os.path.exists(h5_path):
        download_data(
            f"{base}/V1_Human_Lymph_Node_filtered_feature_bc_matrix.h5",
            file_path="filtered_feature_bc_matrix.h5",
            dir=sample_dir,
        )

    # Spatial side: tarball expands into `spatial/` next to the matrix.
    spatial_dir = os.path.join(sample_dir, "spatial")
    if not os.path.isdir(spatial_dir) or not os.listdir(spatial_dir):
        tar_local = download_data(
            f"{base}/V1_Human_Lymph_Node_spatial.tar.gz",
            file_path="V1_Human_Lymph_Node_spatial.tar.gz",
            dir=sample_dir,
        )
        with tarfile.open(tar_local, "r:gz") as tar:
            tar.extractall(sample_dir)
        print(f"{Colors.GREEN}{EMOJI['done']} Extracted spatial assets to {spatial_dir}{Colors.ENDC}")

    from ..io.spatial import read_visium
    adata = read_visium(sample_dir)
    adata.var_names_make_unique()
    return adata


@register_function(
    aliases=['pbmc3k', 'PBMC3K', 'pbmc_3k_dataset'],
    category="datasets",
    description="Load PBMC3k dataset with optional processed/raw mode and mock fallback.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.pbmc3k(processed=True)'],
    related=['datasets.pbmc8k', 'datasets.create_mock_dataset']
)
def pbmc3k(processed: bool = False) -> AnnData:
    """
    Load PBMC 3k dataset from URL.
    
    3k PBMCs from 10x Genomics. Downloads directly from public URLs,
    falls back to mock data generation if URLs are unavailable.
    
    Parameters
    ----------
    processed:bool
        If ``True``, load processed PBMC3k file; otherwise load raw matrix.
    
    Returns
    -------
    AnnData
        PBMC3k AnnData, with mock fallback if remote download fails.
    """
    try:
        if processed:
            # Use the official processed PBMC3k from scanpy/cellxgene
            url = "https://raw.githubusercontent.com/chanzuckerberg/cellxgene/main/example-dataset/pbmc3k.h5ad"
            filename = "pbmc3k_processed.h5ad"
        else:
            # Use the raw PBMC3k data
            url = "https://falexwolf.de/data/pbmc3k_raw.h5ad"
            filename = "pbmc3k_raw.h5ad"
        
        print(f"{Colors.HEADER} Loading PBMC 3k dataset ({'processed' if processed else 'raw'}){Colors.ENDC}")
        adata = get_adata(url, filename)
        
        if adata is not None:
            return adata
        else:
            print(f"{Colors.WARNING}{EMOJI['warning']} Failed to load from URL, generating mock data...{Colors.ENDC}")
            return create_mock_dataset(n_cells=2700, n_genes=32738, n_cell_types=8, with_clustering=processed)
            
    except Exception as e:
        print(f"{Colors.FAIL}{EMOJI['error']} Error loading PBMC3k: {e}{Colors.ENDC}")
        print(f"{Colors.WARNING}🔄 Generating mock data as fallback...{Colors.ENDC}")
        return create_mock_dataset(n_cells=2700, n_genes=32738, n_cell_types=8, with_clustering=processed)


@register_function(
    aliases=['bhattacherjee', 'cocaine_pfc_dataset', 'bhattacherjee_pfc'],
    category="datasets",
    description="Load Bhattacherjee mouse PFC cocaine-administration scRNA-seq dataset.",
    prerequisites={},
    requires={},
    produces={},
    auto_fix='none',
    examples=['adata = ov.datasets.bhattacherjee(processed=True)'],
    related=['datasets.pbmc3k', 'datasets.create_mock_dataset']
)
def bhattacherjee(processed: bool = True) -> AnnData:
    """Processed single-cell data PFC adult mice under cocaine self-administration.

    Adult mice were subject to cocaine self-administration, samples were
    collected at three time points: Maintenance, 48h after cocaine withdrawal and
    15 days after cocaine withdrawal.

    Parameters
    ----------
    processed:bool
        Whether to return processed-style fallback mock data on failure.

    References:
        Bhattacherjee A, Djekidel MN, Chen R, Chen W, Tuesta LM, Zhang Y. Cell
        type-specific transcriptional programs in mouse prefrontal cortex during
        adolescence and addiction. Nat Commun. 2019 Sep 13;10(1):4169.
        doi: 10.1038/s41467-019-12054-3. PMID: 31519873; PMCID: PMC6744514.

    Returns
    -------
    AnnData
        Single-cell RNA-seq dataset from Bhattacherjee et al. or mock fallback.

    Examples:
        >>> import omicverse as ov
        >>> adata = ov.datasets.bhattacherjee()
        >>> print(adata)
    """
    try:
        url = "https://exampledata.scverse.org/pertpy/bhattacher_rna.h5ad"
        filename = "bhattacherjee_rna.h5ad"

        print(f"{Colors.HEADER} Loading Bhattacherjee et al. dataset{Colors.ENDC}")
        adata = get_adata(url, filename)

        if adata is not None:
            return adata
        else:
            print(f"{Colors.WARNING}{EMOJI['warning']} Failed to load from URL, generating mock data...{Colors.ENDC}")
            return create_mock_dataset(n_cells=5000, n_genes=2000, n_cell_types=10, with_clustering=processed)

    except Exception as e:
        print(f"{Colors.FAIL}{EMOJI['error']} Error loading Bhattacherjee dataset: {e}{Colors.ENDC}")
        print(f"{Colors.WARNING}🔄 Generating mock data as fallback...{Colors.ENDC}")
        return create_mock_dataset(n_cells=5000, n_genes=2000, n_cell_types=10, with_clustering=processed)


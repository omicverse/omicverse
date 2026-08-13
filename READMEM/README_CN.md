<img src="https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/firstpage4.jpg" >


<div align="center">
  <a href="../README.md">English</a> | <a href="README_ES.md">Español</a> | <a href="README_JP.md">日本語</a> | <a href="README_DE.md">Deutsch</a> | <a href="README_FR.md">Français</a> | <a href="README_KR.md">한국어</a>

|         |                                                                                                                                                  |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| CI/CD   |  [![pre-commit.ci status][precommit-badge]][precommit-link]               |
| Docs    | [![Documentation Status][readthedocs-badge]][readthedocs-link] [![Preprint][preprint-badge]][preprint-link]                                            |
| Package | [![PyPI - Version][pypi-badge]][pypi-link] [![pypi download][pypi-download-badge]][pypi-download-link] [![Conda-forge badge][conda-forge-badge]][anaconda-link] [![Docker image][docker-badge]][docker-link] |
| Meta    | [![scverse-badge]][scverse-link] [![License][license-badge]][license-link] [![Star][star-badge]][star-link] [![Citations][citation-badge]][citation-link]           |


[precommit-badge]: https://github.com/Starlitnightly/omicverse/workflows/py310|py311/badge.svg
[precommit-link]:https://github.com/Starlitnightly/omicverse/
[readthedocs-badge]:https://readthedocs.org/projects/omicverse/badge/?version=latest
[readthedocs-link]:https://omicverse.readthedocs.io/en/latest/?badge=latest
[preprint-badge]: https://img.shields.io/badge/DOI-10.1038/s41467.024.50194.3-368650.svg
[preprint-link]: https://doi.org/10.1038/s41467-024-50194-3
[pypi-badge]: https://img.shields.io/pypi/v/omicverse
[pypi-link]: https://pypi.org/project/omicverse
[pypi-download-badge]:https://static.pepy.tech/badge/omicverse
[pypi-download-link]:https://pepy.tech/project/omicverse
[conda-forge-badge]: https://img.shields.io/conda/dn/conda-forge/omicverse?logo=Anaconda
[anaconda-link]: https://anaconda.org/conda-forge/omicverse
[docker-badge]: https://img.shields.io/badge/docker-image-blue?logo=docker
[docker-link]: https://img.shields.io/docker/pulls/starlitnightly/omicverse
[license-badge]: https://img.shields.io/badge/license-GNU-blue
[license-link]: https://img.shields.io/apm/l/vim-mode
[scverse-badge]: https://img.shields.io/badge/scverse-ecosystem-blue.svg?labelColor=yellow
[scverse-link]: https://scverse.org/
[star-badge]:https://img.shields.io/github/stars/Starlitnightly/omicverse
[star-link]:https://github.com/Starlitnightly/omicverse
[citation-badge]:https://citations.njzjz.win/10.1038/s41467-024-50194-3
[citation-link]:https://doi.org/10.1038/s41467-024-50194-3
</div>


**`OmicVerse v2`** 是一个面向现代转录组学与多组学分析的统一 Python 项目，覆盖 bulk RNA-seq、单细胞、空间转录组、可视化、模型分析与 AI 辅助工作流。了解更多背景，请阅读我们的论文：[OmicVerse: a framework for bridging and deepening insights across bulk and single-cell sequencing](https://www.nature.com/articles/s41467-024-50194-3)

> [!IMPORTANT]
>
> **给我们点个星**, 您将及时收到GitHub的所有发布通知 ~ ⭐️
>
> 如果您喜欢 **OmicVerse** 并想支持我们的使命，请考虑[💗捐赠](https://ifdian.net/a/starlitnightly)来支持我们的努力。

> [!Note]
> 
> 为了更好地服务中文用户社区，我们创建了微信群，欢迎在issue中扫描二维码加入：https://github.com/Starlitnightly/omicverse/issues/294

<details>
  <summary><kbd>Star历史</kbd></summary>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://star-history.dera.page/svg?repos=Starlitnightly%2Fomicverse&theme=dark&type=Date">
    <img width="100%" src="https://star-history.dera.page/svg?repos=Starlitnightly%2Fomicverse&type=Date">
  </picture>
</details>

## `1` [简介][docs-feat-provider]

**OmicVerse v2** 是一个面向现代转录组学与多组学分析的统一 Python 项目。它将 bulk RNA-seq、单细胞分析、空间转录组、可视化、基于模型的分析，以及 AI 辅助工作流整合到同一个包和文档体系中。

> [!NOTE]
> OmicVerse v2 现在更强调一个完整的分析平台，而不再只是单一方法集合。除了核心分析模块之外，它还包含通过 **J.A.R.V.I.S.** 提供的 agent 式工作流、面向 AI 客户端的 MCP 工具服务，以及持续扩展的 `omicverse_guide` 教程与文档系统。

![omicverse-light](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/background_light.png#gh-light-mode-only)
![omicverse-dark](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/background_dark.png#gh-dark-mode-only)

## `2` [目录结构](#)

````shell
.
├── omicverse                  # 主要Python包
├── omicverse_guide            # 文档文件
├── omicverse_web              # Web分析平台
├── sample                     # 一些测试数据
├── LICENSE
└── README.md
````

## `3` [开始使用](#)

OmicVerse可以通过conda或pypi安装，您需要首先安装`pytorch`。请参考[安装教程](https://starlitnightly.github.io/omicverse/Installation_guild/)了解更详细的安装步骤和不同平台的适配（`Windows`、`Linux`或`Mac OS`）。

您可以使用`conda install omicverse -c conda-forge`或`pip install -U omicverse`进行安装。

请查看[omicverse页面](https://starlitnightly.github.io/omicverse/)或[omicverse.readthedocs.io](https://omicverse.readthedocs.io/en/latest/index.html)上的文档和教程。

## `4` [J.A.R.V.I.S 入门](#)

### 4.1 OpenClaw

OmicVerse 现在可以与 OpenClaw 项目配合进行直接交互式分析，例如：

```bash
omicverse claw 'help me annotate the lung scrna-seq'
```

这个模块由 `ov.Agent` 提供支持，适合通过对话方式分析 `AnnData`。

完整教程可见 [这里](https://omicverse.readthedocs.io/en/latest/Tutorials-llm/t_ov_agent_pbmc3k/)

### 4.2 MCP Server（Model Context Protocol）

OmicVerse 提供了一个 MCP 服务器，可通过标准 [Model Context Protocol](https://modelcontextprotocol.io/) 将已注册的分析工具暴露给 Claude Code 等 AI 助手。

```bash
# 安装 MCP 依赖
pip install -e "omicverse[mcp]"

# 启动服务器（stdio transport）
python -m omicverse.mcp        # 或者：omicverse-mcp
python -m omicverse.mcp --phase P0   # 仅启用核心流程工具
```

完整教程可见 [这里](https://omicverse.readthedocs.io/en/latest/Tutorials-llm/t_mcp_guide/)

### 4.3 J.A.R.V.I.S 消息系统

如果你想通过手机等消息端分析 `AnnData`，可以尝试：

```bash
# 安装 jarvis 依赖
pip install "omicverse[jarvis]"

# 通过 Telegram 启动 J.A.R.V.I.S
omicverse jarvis --channel telegram --token "$TELEGRAM_BOT_TOKEN"
```

完整教程可见 [这里](https://omicverse.readthedocs.io/en/latest/Tutorials-jarvis/t_msg_bot_overview/)


## `5` [数据框架和参考](#)

omicverse基于以下四种数据结构实现。

<div align="center">
<table>
  <tr>
    <td> <a href="https://github.com/pandas-dev/pandas">pandas</a></td>
    <td> <a href="https://github.com/scverse/anndata">anndata</a></td>
    <td> <a href="https://github.com/numpy/numpy">numpy</a></td>
    <td> <a href="https://github.com/scverse/mudata">mudata</a></td>
  </tr>
</table>
</div>

---

该表包含已发布的工具

<div align="center">
<table>

  <tr>
    <td align="center">Scanpy<br><a href="https://github.com/scverse/scanpy">📦</a> <a href="https://link.springer.com/article/10.1186/s13059-017-1382-0">📖</a></td>
    <td align="center">dynamicTreeCut<br><a href="https://github.com/kylessmith/dynamicTreeCut">📦</a> <a href="https://academic.oup.com/bioinformatics/article/24/5/719/200751">📖</a></td>
    <td align="center">scDrug<br><a href="https://github.com/ailabstw/scDrug">📦</a> <a href="https://www.sciencedirect.com/science/article/pii/S2001037022005505">📖</a></td>
    <td align="center">MOFA<br><a href="https://github.com/bioFAM/mofapy2">📦</a> <a href="https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02015-1">📖</a></td>
    <td align="center">COSG<br><a href="https://github.com/genecell/COSG">📦</a> <a href="https://academic.oup.com/bib/advance-article-abstract/doi/10.1093/bib/bbab579/6511197?redirectedFrom=fulltext">📖</a></td>
    <td align="center">CellPhoneDB<br><a href="https://github.com/ventolab/CellphoneDB">📦</a> <a href="https://www.nature.com/articles/s41596-020-0292-x">📖</a></td>
    </tr>

  <tr>
    <td align="center">AUCell<br><a href="https://github.com/aertslab/AUCell">📦</a> <a href="https://bioconductor.org/packages/AUCell">📖</a></td>
    <td align="center">Bulk2Space<br><a href="https://github.com/ZJUFanLab/bulk2space">📦</a> <a href="https://www.nature.com/articles/s41467-022-34271-z">📖</a></td>
    <td align="center">SCSA<br><a href="https://github.com/bioinfo-ibms-pumc/SCSA">📦</a> <a href="https://doi.org/10.3389/fgene.2020.00490">📖</a></td>
    <td align="center">WGCNA<br><a href="http://www.genetics.ucla.edu/labs/horvath/CoexpressionNetwork/Rpackages/WGCNA">📦</a> <a href="https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-559">📖</a></td>
    <td align="center">StaVIA<br><a href="https://github.com/ShobiStassen/VIA">📦</a> <a href="https://www.nature.com/articles/s41467-021-25773-3">📖</a></td>
    <td align="center">PyDESeq2<br><a href="https://github.com/owkin/PyDESeq2">📦</a> <a href="https://academic.oup.com/bioinformatics/article/39/9/btad547/7260507">📖</a></td>
</tr>

  <tr>
    <td align="center">NOCD<br><a href="https://github.com/shchur/overlapping-community-detection">📦</a> <a href="https://arxiv.org/abs/1909.12201">📖</a></td>
    <td align="center">SIMBA<br><a href="https://github.com/pinellolab/simba">📦</a> <a href="https://www.nature.com/articles/s41592-023-01899-8">📖</a></td>
    <td align="center">GLUE<br><a href="https://github.com/gao-lab/GLUE">📦</a> <a href="https://www.nature.com/articles/s41587-022-01284-4">📖</a></td>
    <td align="center">MetaTiME<br><a href="https://github.com/yi-zhang/MetaTiME">📦</a> <a href="https://www.nature.com/articles/s41467-023-38333-8">📖</a></td>
    <td align="center">TOSICA<br><a href="https://github.com/JackieHanLab/TOSICA">📦</a> <a href="https://doi.org/10.1038/s41467-023-35923-4">📖</a></td>
    <td align="center">Harmony<br><a href="https://github.com/slowkow/harmonypy/">📦</a> <a href="https://www.nature.com/articles/s41592-019-0619-0">📖</a></td>
  </tr>

  <tr>
    <td align="center">Scanorama<br><a href="https://github.com/brianhie/scanorama">📦</a> <a href="https://www.nature.com/articles/s41587-019-0113-3">📖</a></td>
    <td align="center">Combat<br><a href="https://github.com/epigenelabs/pyComBat/">📦</a> <a href="https://doi.org/10.1101/2020.03.17.995431">📖</a></td>
    <td align="center">TAPE<br><a href="https://github.com/poseidonchan/TAPE">📦</a> <a href="https://doi.org/10.1038/s41467-022-34550-9">📖</a></td>
    <td align="center">SEACells<br><a href="https://github.com/dpeerlab/SEACells">📦</a> <a href="https://www.nature.com/articles/s41587-023-01716-9">📖</a></td>
    <td align="center">Palantir<br><a href="https://github.com/dpeerlab/Palantir">📦</a> <a href="https://doi.org/10.1038/s41587-019-0068-49">📖</a></td>
    <td align="center">STAGATE<br><a href="https://github.com/QIFEIDKN/STAGATE_pyG">📦</a> <a href="https://www.nature.com/articles/s41467-022-29439-6">📖</a></td>
  </tr>

  <tr>
    <td align="center">scVI<br><a href="https://github.com/scverse/scvi-tools">📦</a> <a href="https://doi.org/10.1038/s41587-021-01206-w">📖</a></td>
    <td align="center">MIRA<br><a href="https://github.com/cistrome/MIRA">📦</a> <a href="https://www.nature.com/articles/s41592-022-01595-z">📖</a></td>
    <td align="center">Tangram<br><a href="https://github.com/broadinstitute/Tangram/">📦</a> <a href="https://www.nature.com/articles/s41592-021-01264-7">📖</a></td>
    <td align="center">STAligner<br><a href="https://github.com/zhoux85/STAligner">📦</a> <a href="https://doi.org/10.1038/s43588-023-00528-w">📖</a></td>
    <td align="center">CEFCON<br><a href="https://github.com/WPZgithub/CEFCON">📦</a> <a href="https://www.nature.com/articles/s41467-023-44103-3">📖</a></td>
    <td align="center">PyComplexHeatmap<br><a href="https://github.com/DingWB/PyComplexHeatmap">📦</a> <a href="https://doi.org/10.1002/imt2.115">📖</a></td>
      </tr>

  <tr>
    <td align="center">STT<br><a href="https://github.com/cliffzhou92/STT/">📦</a> <a href="https://www.nature.com/articles/s41592-024-02266-x#Sec2">📖</a></td>
    <td align="center">SLAT<br><a href="https://github.com/gao-lab/SLAT">📦</a> <a href="https://www.nature.com/articles/s41467-023-43105-5">📖</a></td>
    <td align="center">GPTCelltype<br><a href="https://github.com/Winnie09/GPTCelltype">📦</a> <a href="https://www.nature.com/articles/s41592-024-02235-4">📖</a></td>
    <td align="center">PROST<br><a href="https://github.com/Tang-Lab-super/PROST">📦</a> <a href="https://doi.org/10.1038/s41467-024-44835-w">📖</a></td>
    <td align="center">CytoTRACE 2<br><a href="https://github.com/digitalcytometry/cytotrace2">📦</a> <a href="https://doi.org/10.1101/2024.03.19.585637">📖</a></td>
    <td align="center">GraphST<br><a href="https://github.com/JinmiaoChenLab/GraphST">📦</a> <a href="https://www.nature.com/articles/s41467-023-36796-3#citeas">📖</a></td>
  </tr>

  <tr>
    <td align="center">COMPOSITE<br><a href="https://github.com/CHPGenetics/COMPOSITE/">📦</a> <a href="https://www.nature.com/articles/s41467-024-49448-x#Abs1">📖</a></td>
    <td align="center">mellon<br><a href="https://github.com/settylab/mellon">📦</a> <a href="https://www.nature.com/articles/s41592-024-02302-w">📖</a></td>
    <td align="center">starfysh<br><a href="https://github.com/azizilab/starfysh">📦</a> <a href="http://dx.doi.org/10.1038/s41587-024-02173-8">📖</a></td>
    <td align="center">COMMOT<br><a href="https://github.com/zcang/COMMOT">📦</a> <a href="https://www.nature.com/articles/s41592-022-01728-4">📖</a></td>
    <td align="center">flowsig<br><a href="https://github.com/axelalmet/flowsig">📦</a> <a href="https://doi.org/10.1038/s41592-024-02380-w">📖</a></td>
    <td align="center">pyWGCNA<br><a href="https://github.com/mortazavilab/PyWGCNA">📦</a> <a href="https://doi.org/10.1093/bioinformatics/btad415">📖</a></td>
  </tr>

  <tr>
    <td align="center">CAST<br><a href="https://github.com/wanglab-broad/CAST">📦</a> <a href="https://www.nature.com/articles/s41592-024-02410-7">📖</a></td>
    <td align="center">scMulan<br><a href="https://github.com/SuperBianC/scMulan">📦</a> <a href="https://link.springer.com/chapter/10.1007/978-1-0716-3989-4_57">📖</a></td>
    <td align="center">cellANOVA<br><a href="https://github.com/Janezjz/cellanova">📦</a> <a href="https://www.nature.com/articles/s41587-024-02463-1">📖</a></td>
    <td align="center">BINARY<br><a href="https://github.com/senlin-lin/BINARY/">📦</a> <a href="https://www.sciencedirect.com/science/article/pii/S2666979X24001319">📖</a></td>
    <td align="center">GASTON<br><a href="https://github.com/raphael-group/GASTON">📦</a> <a href="https://www.nature.com/articles/s41592-024-02503-3">📖</a></td>
    <td align="center">pertpy<br><a href="https://github.com/scverse/pertpy">📦</a> <a href="https://www.biorxiv.org/content/early/2024/08/07/2024.08.04.606516">📖</a></td>
  </tr>

  <tr>
    <td align="center">inmoose<br><a href="https://github.com/epigenelabs/inmoose">📦</a> <a href="https://www.nature.com/articles/s41598-025-03376-y">📖</a></td>
    <td align="center">memento<br><a href="https://github.com/yelabucsf/scrna-parameter-estimation">📦</a> <a href="https://www.cell.com/cell/fulltext/S0092-8674(24)01144-9">📖</a></td>
    <td align="center">GSEApy<br><a href="https://github.com/zqfang/GSEApy">📦</a> <a href="https://academic.oup.com/bioinformatics/article-abstract/39/1/btac757/6847088">📖</a></td>
    <td align="center">marsilea<br><a href="https://github.com/Marsilea-viz/marsilea/">📦</a> <a href="https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03469-3">📖</a></td>
    <td align="center">scICE<br><a href="https://github.com/Mathbiomed/scICE">📦</a> <a href="https://www.nature.com/articles/s41467-025-60702-8">📖</a></td>
    <td align="center">sude<br><a href="https://github.com/ZPGuiGroupWhu/sude">📦</a> <a href="https://www.nature.com/articles/s42256-025-01112-9">📖</a></td>
  </tr>

  <tr>
    <td align="center">Geneformer<br><a href="https://huggingface.co/ctheodoris/Geneformer">📦</a> <a href="https://www.nature.com/articles/s41586-023-06139-9">📖</a></td>
    <td align="center">scGPT<br><a href="https://github.com/bowang-lab/scGPT">📦</a> <a href="https://www.nature.com/articles/s41592-024-02201-0">📖</a></td>
    <td align="center">scFoundation<br><a href="https://github.com/biomap-research/scFoundation">📦</a> <a href="https://www.nature.com/articles/s41592-024-02305-7">📖</a></td>
    <td align="center">UCE<br><a href="https://github.com/snap-stanford/UCE">📦</a> <a href="https://www.biorxiv.org/content/10.1101/2023.11.28.568918v1.full.pdf">📖</a></td>
    <td align="center">CellPLM<br><a href="https://github.com/OmicsML/CellPLM">📦</a> <a href="https://www.biorxiv.org/content/10.1101/2023.10.03.560734v1">📖</a></td>
    <td align="center">kb-python<br><a href="https://github.com/pachterlab/kb_python">📦</a> <a href="https://doi.org/10.1038/s41596-024-01057-0">📖</a></td>

  </tr>

  <tr>
    <td align="center">Scaden<br><a href="https://github.com/KevinMenden/scaden">📦</a> <a href="https://www.science.org/doi/10.1126/sciadv.aba2619">📖</a></td>
    <td align="center">BayesPrism<br><a href="https://github.com/Danko-Lab/BayesPrism">📦</a> <a href="https://github.com/ziluwang829/pyBayesPrism">📦</a> <a href="https://www.nature.com/articles/s43018-022-00356-3">📖</a></td>
    <td align="center">InstaPrism<br><a href="https://github.com/humengying0907/InstaPrism">📦</a> <a href="https://academic.oup.com/bioinformatics/article/40/7/btae440/7708397">📖</a></td>
    <td align="center">CellTypist<br><a href="https://github.com/Teichlab/celltypist">📦</a> <a href="https://www.science.org/doi/10.1126/science.abl5197">📖</a></td>
    <td align="center">latentvelo<br><a href="https://github.com/Spencerfar/LatentVelo">📦</a> <a href="https://www.cell.com/cell-reports-methods/fulltext/S2667-2375(23)00225-4">📖</a></td>
    <td align="center">graphvelo<br><a href="https://github.com/xing-lab-pitt/GraphVelo">📦</a> <a href="https://www.nature.com/articles/s41467-025-62784-w">📖</a></td>

  </tr>

  <tr>
    <td align="center">scvelo<br><a href="https://github.com/theislab/scvelo">📦</a> <a href="http://dx.doi.org/10.1038/s41587-020-0591-3">📖</a></td>
    <td align="center">Dynamo<br><a href="https://github.com/aristoteleo/dynamo-release">📦</a> <a href="https://www.sciencedirect.com/science/article/pii/S0092867421015774">📖</a></td>
    <td align="center">CONCORD<br><a href="https://github.com/Gartner-Lab/Concord/">📦</a> <a href="https://www.nature.com/articles/s41587-025-02950-z">📖</a></td>
    <td align="center">FlashDeconv<br><a href="https://github.com/cafferychen777/FlashDeconv">📦</a> <a href="https://doi.org/10.64898/2025.12.22.696108">📖</a></td>
    <td align="center">Hotspot<br><a href="https://github.com/yoseflab/hotspot">📦</a> <a href="https://www.sciencedirect.com/science/article/pii/S2405471221001149">📖</a></td>
    <td align="center">Banksy<br><a href="https://github.com/prabhakarlab/Banksy_py">📦</a> <a href="https://www.nature.com/articles/s41588-024-01664-3#citeas">📖</a></td>

  </tr>
  <tr>
    <td align="center">STAR<br><a href="https://github.com/alexdobin/STAR">📦</a> <a href="https://pubmed.ncbi.nlm.nih.gov/23104886/">📖</a></td>
    <td align="center">fastp<br><a href="https://github.com/OpenGene/fastp">📦</a> <a href="https://doi.org/10.1093/bioinformatics/bty560">📖</a></td>
    <td align="center">featureCounts<br><a href="https://github.com/ShiLab-Bioinformatics/subread">📦</a> <a href="https://pubmed.ncbi.nlm.nih.gov/24227677/">📖</a></td>
    <td align="center">edgeR<br><a href="https://bioconductor.org/packages/edgeR">📦</a> <a href="https://academic.oup.com/nar/article/doi/10.1093/nar/gkaf018/7973897">📖</a></td>
    <td align="center">spaco<br><a href="https://github.com/BrainStOrmics/Spaco">📦</a> <a href="https://www.cell.com/patterns/fulltext/S2666-3899(23)00324-0">📖</a></td>
    <td align="center">gsMap<br><a href="https://yanglab.westlake.edu.cn/gsmap/document/software">📦</a> <a href="https://www.nature.com/articles/s41586-025-08757-x">📖</a></td>

  </tr>
  <tr>
    <td align="center">Monocle 2<br><a href="https://github.com/cole-trapnell-lab/monocle-release">📦</a> <a href="https://doi.org/10.1038/nmeth.4402">📖</a></td>
    <td align="center">cell2location<br><a href="https://github.com/BayraktarLab/cell2location">📦</a> <a href="https://doi.org/10.1038/s41587-021-01139-4">📖</a></td>
    <td align="center">bin2cell<br><a href="https://github.com/Teichlab/bin2cell">📦</a> <a href="https://doi.org/10.1093/bioinformatics/btae546">📖</a></td>
    <td align="center">CellCharter<br><a href="https://github.com/CSOgroup/cellcharter">📦</a> <a href="https://doi.org/10.1038/s41588-023-01588-4">📖</a></td>
    <td align="center">SpaceFlow<br><a href="https://github.com/hongleir/SpaceFlow">📦</a> <a href="https://www.nature.com/articles/s41467-022-31739-w">📖</a></td>
    <td align="center">SpatialDE<br><a href="https://github.com/Teichlab/SpatialDE">📦</a> <a href="https://doi.org/10.1038/nmeth.4636">📖</a></td>
  </tr>
  <tr>
    <td align="center">DoubletFinder<br><a href="https://github.com/chris-mcginnis-ucsf/DoubletFinder">📦</a> <a href="https://www.cell.com/cell-systems/fulltext/S2405-4712(19)30073-0">📖</a></td>
    <td align="center">scTenifoldKnk<br><a href="https://github.com/cailab-tamu/scTenifoldKnk">📦</a> <a href="https://doi.org/10.1016/j.patter.2022.100434">📖</a></td>
    <td align="center">scFEA<br><a href="https://github.com/changwn/scFEA">📦</a> <a href="https://doi.org/10.1101/gr.271205.120">📖</a></td>
    <td align="center">scMetabolism<br><a href="https://github.com/wu-yc/scMetabolism">📦</a> <a href="https://doi.org/10.1158/2159-8290.cd-21-0316">📖</a></td>
    <td align="center">MEBOCOST<br><a href="https://github.com/kaifuchenlab/MEBOCOST">📦</a> <a href="https://doi.org/10.1093/nar/gkaf569">📖</a></td>
    <td align="center">Compass<br><a href="https://github.com/YosefLab/Compass">📦</a> <a href="https://doi.org/10.1016/j.cell.2021.05.045">📖</a></td>
  </tr>
</table>
</div>

---

**包含的包未发布或预印本**

- [1] [Cellula](https://github.com/andrecossa5/Cellula/) 是用于探索scRNA-seq的工具包。这些工具执行常见的单细胞分析任务
- [2] [pegasus](https://github.com/lilab-bcb/pegasus/) 是分析数百万个单细胞转录组的工具。它是一个命令行工具、一个python包和基于云的分析工作流的基础。
- [3] [cNMF](https://github.com/dylkot/cNMF) 是从单细胞RNA-Seq (scRNA-Seq)数据推断基因表达程序的分析管道。

## `6` [联系方式](#)

- Zehua Zeng ([starlitnightly@gmail.com](mailto:starlitnightly@gmail.com) 或 [zehuazeng@xs.ustb.edu.cn](mailto:zehuazeng@xs.ustb.edu.cn))
- Lei Hu ([hulei@westlake.edu.cn](mailto:hulei@westlake.edu.cn))

## `7` [开发者指南和贡献](#)

如果您想为omicverse贡献代码，请参考我们的[开发者文档](https://omicverse.readthedocs.io/en/latest/Developer_guild/)。

### 本地运行测试

安装测试依赖并使用 `pytest` 运行测试：

```bash
pip install -e .[tests]
# 或安装最新固定依赖
pip install -r requirements-latest.txt

pytest
```

可选的 `tests` extra 和 `requirements-latest.txt` 文件已经包含 `pytest-asyncio>=0.23`，这是 `tests/utils/` 目录下异步流式测试所必需的。

<table align="center">
    <tr>
      <th colspan="2">
        <br><img src="https://contrib.rocks/image?repo=Starlitnightly/omicverse"><br><br>
      </th>
    </tr>
</table>

> [!IMPORTANT]  
> 我们要感谢以下微信公众号推广Omicverse。
> <p align="center"> <a href="https://mp.weixin.qq.com/s/egAnRfr3etccU_RsN-zIlg" target="_blank" rel="noreferrer"> <img src="../README.assets/image-20230701163953794.png" alt="linux" width="50" height="50"/> </a> <a href="https://zhuanlan.zhihu.com/c_1257815636945915904?page=3" target="_blank" rel="noreferrer"> <img src="../README.assets/WechatIMG688.png" alt="linux" width="50" height="50"/> </a> </p>

## `8` [引用](https://doi.org/10.1038/s41467-024-50194-3)

如果您在工作中使用`omicverse`，请按以下方式引用`omicverse`论文：

> **OmicVerse: a framework for bridging and deepening insights across bulk and single-cell sequencing**
>
> Zeng, Z., Ma, Y., Hu, L. et al.
>
> _Nature Communication_ 2024年7月16日. doi: [10.1038/s41467-024-50194-3](https://doi.org/10.1038/s41467-024-50194-3).

以下是一些其他相关包，如果您使用它们，请随时引用它们！

> **CellOntologyMapper: Consensus mapping of cell type annotation**
>
> Zeng, Z., Wang, X., Du, H. et al.
>
> _imetaomics_ 2025年11月6日. doi: [10.1002/imo2.70064](https://doi.org/10.1002/imo2.70064).

## `9` [其他](#)

如果您想赞助我们项目的开发，可以前往爱发电网站 (https://ifdian.net/a/starlitnightly) 赞助我们。

版权所有 © 2024 [112 Lab](https://112lab.asia/)。<br />
本项目采用[GPL3.0](../LICENSE)许可证。

<!-- LINK GROUP -->
[docs-feat-provider]: https://starlitnightly.github.io/omicverse/ 

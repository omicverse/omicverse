<img src="https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/firstpage4.jpg" >


<div align="center">
  <a href="../README.md">English</a> | <a href="README_CN.md">中文</a> | <a href="README_ES.md">Español</a> | <a href="README_DE.md">Deutsch</a> | <a href="README_FR.md">Français</a> | <a href="README_KR.md">한국어</a>

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


**`OmicVerse v2`** は、現代的なトランスクリプトミクスとマルチオミクス解析のための統合 Python プロジェクトです。bulk RNA-seq、シングルセル解析、空間トランスクリプトミクス、可視化、モデルベース解析、AI 支援ワークフローまでをカバーします。詳細については、私たちの論文をお読みください：[OmicVerse: a framework for bridging and deepening insights across bulk and single-cell sequencing](https://www.nature.com/articles/s41467-024-50194-3)

> [!IMPORTANT]
>
> **私たちにスターを**, GitHubからの全ての新リリース通知を遅延なく受け取れます ~ ⭐️
>
> **OmicVerse** をお気に入りいただき、私たちの使命をサポートしたい場合は、[💗寄付](https://ifdian.net/a/starlitnightly)をご検討ください。

<details>
  <summary><kbd>スター履歴</kbd></summary>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://star-history.dera.page/svg?repos=Starlitnightly%2Fomicverse&theme=dark&type=Date">
    <img width="100%" src="https://star-history.dera.page/svg?repos=Starlitnightly%2Fomicverse&type=Date">
  </picture>
</details>

## `1` [はじめに][docs-feat-provider]

**OmicVerse v2** は、現代的なトランスクリプトミクスおよびマルチオミクス解析のための統合 Python プロジェクトです。bulk RNA-seq、シングルセル解析、空間トランスクリプトミクス、可視化、モデルベース解析、さらに AI 支援ワークフローまでを、1 つのパッケージとドキュメント体系にまとめています。

> [!NOTE]
> OmicVerse v2 は、もはや単一手法のパッケージではなく、より広い解析プラットフォームとして構成されています。コア解析モジュールに加えて、**J.A.R.V.I.S.** によるエージェント型ワークフロー、AI クライアント向けの MCP ベースのツール提供、そして `omicverse_guide` 以下で拡張され続けるチュートリアル／ドキュメント体系を含んでいます。

![omicverse-light](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/background_light.png#gh-light-mode-only)
![omicverse-dark](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/background_dark.png#gh-dark-mode-only)

## `2` [ディレクトリ構造](#)

````shell
.
├── omicverse                  # メインPythonパッケージ
├── omicverse_guide            # ドキュメントファイル
├── omicverse_web              # Web解析プラットフォーム
├── sample                     # いくつかのテストデータ
├── LICENSE
└── README.md
````

## `3` [使い始める](#)

OmicVerseはcondaまたはpypiを通じてインストールでき、最初に`pytorch`をインストールする必要があります。詳細なインストール手順と異なるプラットフォーム（`Windows`、`Linux`、`Mac OS`）への適応については、[インストールチュートリアル](https://starlitnightly.github.io/omicverse/Installation_guild/)をご参照ください。

インストールには`conda install omicverse -c conda-forge`または`pip install -U omicverse`を使用できます。

[omicverseページ](https://starlitnightly.github.io/omicverse/)または[omicverse.readthedocs.io](https://omicverse.readthedocs.io/en/latest/index.html)でドキュメントとチュートリアルをご確認ください。

## `4` [J.A.R.V.I.S 入門](#)

### 4.1 OpenClaw

OmicVerse は OpenClaw プロジェクトと連携した対話的解析をサポートしています。たとえば次のように使えます。

```bash
omicverse claw 'help me annotate the lung scrna-seq'
```

このモジュールは `ov.Agent` によって支えられており、対話形式で `AnnData` を解析するのに適しています。

完全なチュートリアルは [こちら](https://omicverse.readthedocs.io/en/latest/Tutorials-llm/t_ov_agent_pbmc3k/)

### 4.2 MCP Server（Model Context Protocol）

OmicVerse は、登録済み解析ツールを [Model Context Protocol](https://modelcontextprotocol.io/) を通じて Claude Code などの AI アシスタントへ提供する MCP サーバーを備えています。

```bash
# MCP 依存関係付きでインストール
pip install -e "omicverse[mcp]"

# サーバー起動（stdio transport）
python -m omicverse.mcp        # または: omicverse-mcp
python -m omicverse.mcp --phase P0   # コアパイプラインツールのみ
```

完全なチュートリアルは [こちら](https://omicverse.readthedocs.io/en/latest/Tutorials-llm/t_mcp_guide/)

### 4.3 J.A.R.V.I.S メッセージシステム

携帯電話やメッセージチャネルから `AnnData` を解析したい場合は、次を試すことができます。

```bash
# jarvis 依存関係付きでインストール
pip install "omicverse[jarvis]"

# Telegram で J.A.R.V.I.S を起動
omicverse jarvis --channel telegram --token "$TELEGRAM_BOT_TOKEN"
```

完全なチュートリアルは [こちら](https://omicverse.readthedocs.io/en/latest/Tutorials-jarvis/t_msg_bot_overview/)


## `5` [データフレームワークと参考文献](#)

omicverseは以下の4つのデータ構造に基づくインフラストラクチャとして実装されています。

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

この表には公開されたツールが含まれています

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

**含まれているパッケージが未公開またはプレプリント**

- [1] [Cellula](https://github.com/andrecossa5/Cellula/) は、scRNA-seqの探索のためのツールキットを提供します。これらのツールは一般的な単一細胞解析タスクを実行します
- [2] [pegasus](https://github.com/lilab-bcb/pegasus/) は、数百万の単一細胞の転写産物を解析するためのツールです。これは、コマンドラインツール、pythonパッケージ、およびクラウドベースの解析ワークフローの基盤です。
- [3] [cNMF](https://github.com/dylkot/cNMF) は、単一細胞RNA-Seq (scRNA-Seq)データから遺伝子発現プログラムを推論するための解析パイプラインです。

## `6` [お問い合わせ](#)

- Zehua Zeng ([starlitnightly@gmail.com](mailto:starlitnightly@gmail.com) または [zehuazeng@xs.ustb.edu.cn](mailto:zehuazeng@xs.ustb.edu.cn))
- Lei Hu ([hulei@westlake.edu.cn](mailto:hulei@westlake.edu.cn))

## `7` [開発者ガイドと貢献](#)

omicverseに貢献したい場合は、[開発者ドキュメント](https://omicverse.readthedocs.io/en/latest/Developer_guild/)をご参照ください。

### ローカルでテストを実行する

テスト依存関係をインストールし、`pytest` でテストスイートを実行します：

```bash
pip install -e .[tests]
# または固定された最新要件をインストール
pip install -r requirements-latest.txt

pytest
```

オプションの `tests` extra と `requirements-latest.txt` ファイルにはすでに `pytest-asyncio>=0.23` が含まれており、`tests/utils/` 配下の非同期ストリーミングテストに必要です。

<table align="center">
    <tr>
      <th colspan="2">
        <br><img src="https://contrib.rocks/image?repo=Starlitnightly/omicverse"><br><br>
      </th>
    </tr>
</table>

> [!IMPORTANT]  
> Omicverseを推進してくださった以下のWeChat公式アカウントに感謝いたします。
> <p align="center"> <a href="https://mp.weixin.qq.com/s/egAnRfr3etccU_RsN-zIlg" target="_blank" rel="noreferrer"> <img src="../README.assets/image-20230701163953794.png" alt="linux" width="50" height="50"/> </a> <a href="https://zhuanlan.zhihu.com/c_1257815636945915904?page=3" target="_blank" rel="noreferrer"> <img src="../README.assets/WechatIMG688.png" alt="linux" width="50" height="50"/> </a> </p>

## `8` [引用](https://doi.org/10.1038/s41467-024-50194-3)

作業で`omicverse`を使用する場合は、以下のように`omicverse`の出版物を引用してください：

> **OmicVerse: a framework for bridging and deepening insights across bulk and single-cell sequencing**
>
> Zeng, Z., Ma, Y., Hu, L. et al.
>
> _Nature Communication_ 2024年7月16日. doi: [10.1038/s41467-024-50194-3](https://doi.org/10.1038/s41467-024-50194-3).

以下は他の関連パッケージです。使用する場合は自由に参照してください！

> **CellOntologyMapper: Consensus mapping of cell type annotation**
>
> Zeng, Z., Wang, X., Du, H. et al.
>
> _imetaomics_ 2025年11月6日. doi: [10.1002/imo2.70064](https://doi.org/10.1002/imo2.70064).

## `9` [その他](#)

プロジェクトの開発をスポンサーしたい場合は、afdianウェブサイト（https://ifdian.net/a/starlitnightly）にアクセスしてスポンサーしてください。

Copyright © 2024 [112 Lab](https://112lab.asia/)。<br />
このプロジェクトは[GPL3.0](../LICENSE)ライセンスです。

<!-- LINK GROUP -->
[docs-feat-provider]: https://starlitnightly.github.io/omicverse/ 

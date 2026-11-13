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
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Starlitnightly%2Fomicverse&theme=dark&type=Date">
    <img width="100%" src="https://api.star-history.com/svg?repos=Starlitnightly%2Fomicverse&type=Date">
  </picture>
</details>

## `1` [简介][docs-feat-provider]

**OmicVerse v2** 是一个面向现代转录组学与多组学分析的统一 Python 项目。它将 bulk RNA-seq、单细胞分析、空间转录组、可视化、基于模型的分析，以及 AI 辅助工作流整合到同一个包和文档体系中。

> [!NOTE]
> OmicVerse v2 现在更强调一个完整的分析平台，而不再只是单一方法集合。除了核心分析模块之外，它还包含通过 **J.A.R.V.I.S.** 提供的 agent 式工作流、面向 AI 客户端的 MCP 工具服务，以及持续扩展的 `omicverse_guide` 教程与文档系统。

![omicverse-light](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/background_light.png#gh-light-mode-only)
![omicverse-dark](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/background_dark.png#gh-dark-mode-only)

## `2` [开始使用](#)

OmicVerse可以通过conda或pypi安装，您需要首先安装`pytorch`。请参考[安装教程](https://starlitnightly.github.io/omicverse/Installation_guild/)了解更详细的安装步骤和不同平台的适配（`Windows`、`Linux`或`Mac OS`）。

您可以使用`conda install omicverse -c conda-forge`或`pip install -U omicverse`进行安装。

请查看[omicverse页面](https://starlitnightly.github.io/omicverse/)或[omicverse.readthedocs.io](https://omicverse.readthedocs.io/en/latest/index.html)上的文档和教程。

智能体工作流（OpenClaw、MCP 服务器和 J.A.R.V.I.S. 消息机器人）的说明见 [OmicClaw](https://github.com/omicverse/omicclaw)。

## `3` [功能覆盖](https://omicverse.readthedocs.io/en/latest/Feature_coverage/)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../README.assets/feature_coverage_dark.png">
  <img src="../README.assets/feature_coverage_light.png" width="90%" alt="Feature coverage of omicverse, the scverse ecosystem and ten other frameworks">
</picture>

omicverse、scverse 生态与另外十个框架在 75 项经逐项核对的分析功能上的覆盖对比。带逐格证据的完整功能表、数据框架以及 OmicVerse 收录的已发表工具清单见[文档](https://omicverse.readthedocs.io/en/latest/Feature_coverage/)。

## `4` [联系方式](#)

- Zehua Zeng ([starlitnightly@gmail.com](mailto:starlitnightly@gmail.com) 或 [steorra@omicos.cn](mailto:steorra@omicos.cn))
- Meng Xu ([mengxu@omicos.cn](mailto:mengxu@omicos.cn))

## `5` [开发者指南和贡献](#)

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

## `6` [引用](https://doi.org/10.1038/s41467-024-50194-3)

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

## `7` [其他](#)

如果您想赞助我们项目的开发，可以前往爱发电网站 (https://ifdian.net/a/starlitnightly) 赞助我们。

版权所有 © 2024 [112 Lab](https://112lab.asia/)。<br />
本项目采用[GPL3.0](../LICENSE)许可证。

<!-- LINK GROUP -->
[docs-feat-provider]: https://starlitnightly.github.io/omicverse/ 

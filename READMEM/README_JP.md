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
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Starlitnightly%2Fomicverse&theme=dark&type=Date">
    <img width="100%" src="https://api.star-history.com/svg?repos=Starlitnightly%2Fomicverse&type=Date">
  </picture>
</details>

## `1` [はじめに][docs-feat-provider]

**OmicVerse v2** は、現代的なトランスクリプトミクスおよびマルチオミクス解析のための統合 Python プロジェクトです。bulk RNA-seq、シングルセル解析、空間トランスクリプトミクス、可視化、モデルベース解析、さらに AI 支援ワークフローまでを、1 つのパッケージとドキュメント体系にまとめています。

> [!NOTE]
> OmicVerse v2 は、もはや単一手法のパッケージではなく、より広い解析プラットフォームとして構成されています。コア解析モジュールに加えて、**J.A.R.V.I.S.** によるエージェント型ワークフロー、AI クライアント向けの MCP ベースのツール提供、そして `omicverse_guide` 以下で拡張され続けるチュートリアル／ドキュメント体系を含んでいます。

![omicverse-light](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/background_light.png#gh-light-mode-only)
![omicverse-dark](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/background_dark.png#gh-dark-mode-only)

## `2` [使い始める](#)

OmicVerseはcondaまたはpypiを通じてインストールでき、最初に`pytorch`をインストールする必要があります。詳細なインストール手順と異なるプラットフォーム（`Windows`、`Linux`、`Mac OS`）への適応については、[インストールチュートリアル](https://starlitnightly.github.io/omicverse/Installation_guild/)をご参照ください。

インストールには`conda install omicverse -c conda-forge`または`pip install -U omicverse`を使用できます。

[omicverseページ](https://starlitnightly.github.io/omicverse/)または[omicverse.readthedocs.io](https://omicverse.readthedocs.io/en/latest/index.html)でドキュメントとチュートリアルをご確認ください。

エージェントワークフロー（OpenClaw、MCP サーバー、J.A.R.V.I.S. メッセージボット）は [OmicClaw](https://github.com/omicverse/omicclaw) を参照してください。

## `3` [機能カバレッジ](https://omicverse.readthedocs.io/en/latest/Feature_coverage/)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../README.assets/feature_coverage_dark.png">
  <img src="../README.assets/feature_coverage_light.png" width="90%" alt="Feature coverage of omicverse, the scverse ecosystem and ten other frameworks">
</picture>

75 項目の分析機能（逐一検証済み）における omicverse、scverse エコシステムと他の 10 フレームワークの機能カバレッジ比較です。セルごとの根拠を含む完全な表、データフレームワーク、OmicVerse に含まれる公開ツールの一覧は[ドキュメント](https://omicverse.readthedocs.io/en/latest/Feature_coverage/)を参照してください。

## `4` [お問い合わせ](#)

- Zehua Zeng ([starlitnightly@gmail.com](mailto:starlitnightly@gmail.com) または [steorra@omicos.cn](mailto:steorra@omicos.cn))
- Meng Xu ([mengxu@omicos.cn](mailto:mengxu@omicos.cn))

## `5` [開発者ガイドと貢献](#)

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

## `6` [引用](https://doi.org/10.1038/s41467-024-50194-3)

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

## `7` [その他](#)

プロジェクトの開発をスポンサーしたい場合は、afdianウェブサイト（https://ifdian.net/a/starlitnightly）にアクセスしてスポンサーしてください。

Copyright © 2024 [112 Lab](https://112lab.asia/)。<br />
このプロジェクトは[GPL3.0](../LICENSE)ライセンスです。

<!-- LINK GROUP -->
[docs-feat-provider]: https://starlitnightly.github.io/omicverse/ 

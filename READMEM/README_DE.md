<img src="https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/firstpage4.jpg" >

<div align="center">
  <a href="../README.md">English</a> | <a href="README_CN.md">中文</a> | <a href="README_ES.md">Español</a> | <a href="README_JP.md">日本語</a> | <a href="README_FR.md">Français</a> | <a href="README_KR.md">한국어</a>

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

**`OmicVerse v2`** ist ein einheitliches Python-Projekt für moderne Transkriptomik- und Multi-Omics-Analysen. Es umfasst Bulk RNA-seq, Einzelzellanalysen, räumliche Transkriptomik, Visualisierung, modellbasierte Analyse und KI-gestützte Workflows. Für weitere Hintergründe lesen Sie bitte unser Paper: [OmicVerse: a framework for bridging and deepening insights across bulk and single-cell sequencing](https://www.nature.com/articles/s41467-024-50194-3)

> [!IMPORTANT]
>
> **Geben Sie uns einen Stern**, Sie erhalten alle Release-Benachrichtigungen von GitHub ohne Verzögerung ~ ⭐️
>
> Wenn Sie **OmicVerse** mögen und unsere Mission unterstützen möchten, erwägen Sie bitte eine [💗Spende](https://ifdian.net/a/starlitnightly), um unsere Bemühungen zu unterstützen.

<details>
  <summary><kbd>Stern-Historie</kbd></summary>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Starlitnightly%2Fomicverse&theme=dark&type=Date">
    <img width="100%" src="https://api.star-history.com/svg?repos=Starlitnightly%2Fomicverse&type=Date">
  </picture>
</details>

## `1` [Einführung][docs-feat-provider]

**OmicVerse v2** ist ein einheitliches Python-Projekt für moderne Transkriptomik- und Multi-Omics-Analysen. Es bündelt Bulk-RNA-seq, Einzelzellanalysen, räumliche Transkriptomik, Visualisierung, modellbasierte Analyse und KI-gestützte Workflows in einem gemeinsamen Paket- und Dokumentationssystem.

> [!NOTE]
> OmicVerse v2 ist heute als breitere Analyseplattform organisiert und nicht mehr nur als einzelnes Methodenpaket. Neben den Kernmodulen für Datenanalyse umfasst es inzwischen auch agentische Workflows über **J.A.R.V.I.S.**, MCP-basiertes Tool-Serving für KI-Clients sowie ein wachsendes Tutorial- und Dokumentationssystem unter `omicverse_guide`.

![omicverse-light](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/background_light.png#gh-light-mode-only)
![omicverse-dark](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/background_dark.png#gh-dark-mode-only)

## `2` [Erste Schritte](#)

OmicVerse kann über conda oder pypi installiert werden und Sie müssen zuerst `pytorch` installieren. Bitte schauen Sie sich das [Installationstutorial](https://starlitnightly.github.io/omicverse/Installation_guild/) für detailliertere Installationsschritte und Anpassungen für verschiedene Plattformen (`Windows`, `Linux` oder `Mac OS`) an.

Sie können `conda install omicverse -c conda-forge` oder `pip install -U omicverse` für die Installation verwenden.

Bitte schauen Sie sich die Dokumentation und Tutorials auf der [omicverse-Seite](https://starlitnightly.github.io/omicverse/) oder [omicverse.readthedocs.io](https://omicverse.readthedocs.io/en/latest/index.html) an.

Agenten-Workflows – OpenClaw, der MCP-Server und die J.A.R.V.I.S.-Nachrichtenbots – sind in [OmicClaw](https://github.com/omicverse/omicclaw) dokumentiert.

## `3` [Funktionsumfang](https://omicverse.readthedocs.io/en/latest/Feature_coverage/)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../README.assets/feature_coverage_dark.png">
  <img src="../README.assets/feature_coverage_light.png" width="90%" alt="Feature coverage of omicverse, the scverse ecosystem and ten other frameworks">
</picture>

Funktionsumfang von omicverse im Vergleich zum scverse-Ökosystem und zehn weiteren Frameworks über 75 geprüfte Analysefunktionen. Die vollständige Tabelle mit Belegen je Zelle, das Daten-Framework und die Liste der in OmicVerse enthaltenen publizierten Tools finden sich in der [Dokumentation](https://omicverse.readthedocs.io/en/latest/Feature_coverage/).

## `4` [Kontakt](#)

- Zehua Zeng ([starlitnightly@gmail.com](mailto:starlitnightly@gmail.com) oder [steorra@omicos.cn](mailto:steorra@omicos.cn))
- Meng Xu ([mengxu@omicos.cn](mailto:mengxu@omicos.cn))

## `5` [Entwicklerhandbuch und Beitrag](#)

Wenn Sie zu omicverse beitragen möchten, schauen Sie sich bitte unsere [Entwicklerdokumentation](https://omicverse.readthedocs.io/en/latest/Developer_guild/) an.

### Tests lokal ausführen

Installieren Sie die Test-Abhängigkeiten und führen Sie die Suite mit `pytest` aus:

```bash
pip install -e .[tests]
# oder installieren Sie die zuletzt festgelegten Anforderungen
pip install -r requirements-latest.txt

pytest
```

Das optionale Extra `tests` und die Datei `requirements-latest.txt` enthalten bereits `pytest-asyncio>=0.23`, das für die asynchronen Streaming-Tests unter `tests/utils/` erforderlich ist.

<table align="center">
    <tr>
      <th colspan="2">
        <br><img src="https://contrib.rocks/image?repo=Starlitnightly/omicverse"><br><br>
      </th>
    </tr>
</table>

> [!IMPORTANT]  
> Wir möchten den folgenden offiziellen WeChat-Accounts für die Bewerbung von Omicverse danken.
> <p align="center"> <a href="https://mp.weixin.qq.com/s/egAnRfr3etccU_RsN-zIlg" target="_blank" rel="noreferrer"> <img src="../README.assets/image-20230701163953794.png" alt="linux" width="50" height="50"/> </a> <a href="https://zhuanlan.zhihu.com/c_1257815636945915904?page=3" target="_blank" rel="noreferrer"> <img src="../README.assets/WechatIMG688.png" alt="linux" width="50" height="50"/> </a> </p>

## `6` [Zitation](https://doi.org/10.1038/s41467-024-50194-3)

Wenn Sie `omicverse` in Ihrer Arbeit verwenden, zitieren Sie bitte die `omicverse`-Publikation wie folgt:

> **OmicVerse: a framework for bridging and deepening insights across bulk and single-cell sequencing**
>
> Zeng, Z., Ma, Y., Hu, L. et al.
>
> _Nature Communication_ 16. Juli 2024. doi: [10.1038/s41467-024-50194-3](https://doi.org/10.1038/s41467-024-50194-3).

Hier sind einige andere verwandte Pakete, die Sie gerne referenzieren können, wenn Sie sie verwenden!

> **CellOntologyMapper: Consensus mapping of cell type annotation**
>
> Zeng, Z., Wang, X., Du, H. et al.
>
> _imetaomics_ 6. November 2025. doi: [10.1002/imo2.70064](https://doi.org/10.1002/imo2.70064).

## `7` [Sonstiges](#)

Wenn Sie die Entwicklung unseres Projekts sponsern möchten, können Sie zur afdian-Website (https://ifdian.net/a/starlitnightly) gehen und uns sponsern.

Copyright © 2024 [112 Lab](https://112lab.asia/). <br />
Dieses Projekt ist unter [GPL3.0](../LICENSE) lizenziert.

<!-- LINK GROUP -->
[docs-feat-provider]: https://starlitnightly.github.io/omicverse/ 

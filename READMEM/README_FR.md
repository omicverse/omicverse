<img src="https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/firstpage4.jpg" >


<div align="center">
  <a href="../README.md">English</a> | <a href="README_CN.md">中文</a> | <a href="README_ES.md">Español</a> | <a href="README_JP.md">日本語</a> | <a href="README_DE.md">Deutsch</a> | <a href="README_KR.md">한국어</a>

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


**`OmicVerse v2`** est un projet Python unifié pour la transcriptomique moderne et l'analyse multi-omique. Il couvre le bulk RNA-seq, l'analyse single-cell, la transcriptomique spatiale, la visualisation, l'analyse fondée sur des modèles et les workflows assistés par l'IA. Pour plus de contexte, veuillez lire notre article : [OmicVerse: a framework for bridging and deepening insights across bulk and single-cell sequencing](https://www.nature.com/articles/s41467-024-50194-3)

> [!IMPORTANT]
>
> **Donnez-nous une étoile**, vous recevrez toutes les notifications de version de GitHub sans délai ~ ⭐️
>
> Si vous aimez **OmicVerse** et souhaitez soutenir notre mission, veuillez envisager de faire un [💗don](https://ifdian.net/a/starlitnightly) pour soutenir nos efforts.

<details>
  <summary><kbd>Historique des Étoiles</kbd></summary>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Starlitnightly%2Fomicverse&theme=dark&type=Date">
    <img width="100%" src="https://api.star-history.com/svg?repos=Starlitnightly%2Fomicverse&type=Date">
  </picture>
</details>

## `1` [Introduction][docs-feat-provider]

**OmicVerse v2** est un projet Python unifié pour l'analyse moderne de la transcriptomique et des données multi-omiques. Il rassemble le bulk RNA-seq, l'analyse single-cell, la transcriptomique spatiale, la visualisation, l'analyse fondée sur des modèles et les workflows assistés par l'IA dans un même paquet et un même système de documentation.

> [!NOTE]
> OmicVerse v2 est désormais organisé comme une plateforme d'analyse plus large, et non plus comme un simple paquet centré sur une méthode. En plus des modules d'analyse principaux, il inclut des workflows de type agent via **J.A.R.V.I.S.**, un service d'outils basé sur MCP pour les clients IA, ainsi qu'un système de documentation et de tutoriels en expansion dans `omicverse_guide`.

![omicverse-light](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/background_light.png#gh-light-mode-only)
![omicverse-dark](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/background_dark.png#gh-dark-mode-only)

## `2` [Commencer](#)

OmicVerse peut être installé via conda ou pypi et vous devez installer `pytorch` en premier. Veuillez consulter le [tutoriel d'installation](https://starlitnightly.github.io/omicverse/Installation_guild/) pour des étapes d'installation plus détaillées et des adaptations pour différentes plateformes (`Windows`, `Linux` ou `Mac OS`).

Vous pouvez utiliser `conda install omicverse -c conda-forge` ou `pip install -U omicverse` pour l'installation.

Veuillez consulter la documentation et les tutoriels sur la [page omicverse](https://starlitnightly.github.io/omicverse/) ou [omicverse.readthedocs.io](https://omicverse.readthedocs.io/en/latest/index.html).

Les workflows d’agents — OpenClaw, le serveur MCP et les bots de messagerie J.A.R.V.I.S. — sont documentés dans [OmicClaw](https://github.com/omicverse/omicclaw).

## `3` [Couverture fonctionnelle](https://omicverse.readthedocs.io/en/latest/Feature_coverage/)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../README.assets/feature_coverage_dark.png">
  <img src="../README.assets/feature_coverage_light.png" width="90%" alt="Feature coverage of omicverse, the scverse ecosystem and ten other frameworks">
</picture>

Couverture fonctionnelle d’omicverse comparée à l’écosystème scverse et à dix autres frameworks sur 75 fonctions d’analyse vérifiées. Le tableau complet avec les preuves par cellule, le cadre de données et la liste des outils publiés intégrés à OmicVerse sont dans la [documentation](https://omicverse.readthedocs.io/en/latest/Feature_coverage/).

## `4` [Contact](#)

- Zehua Zeng ([starlitnightly@gmail.com](mailto:starlitnightly@gmail.com) ou [steorra@omicos.cn](mailto:steorra@omicos.cn))
- Meng Xu ([mengxu@omicos.cn](mailto:mengxu@omicos.cn))

## `5` [Guide du Développeur et Contribution](#)

Si vous souhaitez contribuer à omicverse, veuillez consulter notre [documentation développeur](https://omicverse.readthedocs.io/en/latest/Developer_guild/).

### Exécuter les tests localement

Installez les dépendances de test et exécutez la suite avec `pytest` :

```bash
pip install -e .[tests]
# ou installez les dépendances les plus récentes fixées
pip install -r requirements-latest.txt

pytest
```

L'extra optionnel `tests` et le fichier `requirements-latest.txt` incluent déjà `pytest-asyncio>=0.23`, requis pour les tests de streaming asynchrones dans `tests/utils/`.

<table align="center">
    <tr>
      <th colspan="2">
        <br><img src="https://contrib.rocks/image?repo=Starlitnightly/omicverse"><br><br>
      </th>
    </tr>
</table>

> [!IMPORTANT]  
> Nous aimerions remercier les comptes officiels WeChat suivants pour la promotion d'Omicverse.
> <p align="center"> <a href="https://mp.weixin.qq.com/s/egAnRfr3etccU_RsN-zIlg" target="_blank" rel="noreferrer"> <img src="../README.assets/image-20230701163953794.png" alt="linux" width="50" height="50"/> </a> <a href="https://zhuanlan.zhihu.com/c_1257815636945915904?page=3" target="_blank" rel="noreferrer"> <img src="../README.assets/WechatIMG688.png" alt="linux" width="50" height="50"/> </a> </p>

## `6` [Citation](https://doi.org/10.1038/s41467-024-50194-3)

Si vous utilisez `omicverse` dans votre travail, veuillez citer la publication `omicverse` comme suit :

> **OmicVerse: a framework for bridging and deepening insights across bulk and single-cell sequencing**
>
> Zeng, Z., Ma, Y., Hu, L. et al.
>
> _Nature Communication_ 16 Juil 2024. doi: [10.1038/s41467-024-50194-3](https://doi.org/10.1038/s41467-024-50194-3).

Voici quelques autres packages connexes, n'hésitez pas à les référencer si vous les utilisez !

> **CellOntologyMapper: Consensus mapping of cell type annotation**
>
> Zeng, Z., Wang, X., Du, H. et al.
>
> _imetaomics_ 6 Juin 2025. doi: [10.1002/imo2.70064](https://doi.org/10.1002/imo2.70064).

## `7` [Autre](#)

Si vous souhaitez parrainer le développement de notre projet, vous pouvez aller sur le site web afdian (https://ifdian.net/a/starlitnightly) et nous parrainer.

Copyright © 2024 [112 Lab](https://112lab.asia/). <br />
Ce projet est sous licence [GPL3.0](../LICENSE).

<!-- LINK GROUP -->
[docs-feat-provider]: https://starlitnightly.github.io/omicverse/ 

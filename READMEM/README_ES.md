<img src="https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/firstpage4.jpg" >


<div align="center">
  <a href="../README.md">English</a> | <a href="README_CN.md">中文</a> | <a href="README_JP.md">日本語</a> | <a href="README_DE.md">Deutsch</a> | <a href="README_FR.md">Français</a> | <a href="README_KR.md">한국어</a>

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


**`OmicVerse v2`** es un proyecto unificado de Python para transcriptómica y análisis multi-ómicos modernos. Cubre RNA-seq bulk, análisis de célula única, transcriptómica espacial, visualización, análisis basados en modelos y flujos de trabajo asistidos por IA. Para más contexto, lee nuestro artículo: [OmicVerse: a framework for bridging and deepening insights across bulk and single-cell sequencing](https://www.nature.com/articles/s41467-024-50194-3)

> [!IMPORTANT]
>
> **Danos una estrella**, recibirás todas las notificaciones de lanzamiento de GitHub sin demora ~ ⭐️
>
> Si te gusta **OmicVerse** y quieres apoyar nuestra misión, considera hacer una [💗donación](https://ifdian.net/a/starlitnightly) para apoyar nuestros esfuerzos.

<details>
  <summary><kbd>Historial de Estrellas</kbd></summary>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Starlitnightly%2Fomicverse&theme=dark&type=Date">
    <img width="100%" src="https://api.star-history.com/svg?repos=Starlitnightly%2Fomicverse&type=Date">
  </picture>
</details>

## `1` [Introducción][docs-feat-provider]

**OmicVerse v2** es un proyecto unificado de Python para análisis modernos de transcriptómica y multi-ómica. Reúne RNA-seq bulk, análisis de célula única, transcriptómica espacial, visualización, análisis basados en modelos y flujos de trabajo asistidos por IA en un mismo paquete y sistema de documentación.

> [!NOTE]
> OmicVerse v2 está organizado ahora como una plataforma de análisis más amplia, no como un paquete de un solo método. Además de los módulos principales de análisis, incorpora flujos de trabajo tipo agente mediante **J.A.R.V.I.S.**, servicio de herramientas basado en MCP para clientes de IA y un sistema creciente de documentación y tutoriales en `omicverse_guide`.

![omicverse-light](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/background_light.png#gh-light-mode-only)
![omicverse-dark](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/background_dark.png#gh-dark-mode-only)

## `2` [Comenzando](#)

OmicVerse se puede instalar a través de conda o pypi y necesitas instalar `pytorch` primero. Consulta el [tutorial de instalación](https://starlitnightly.github.io/omicverse/Installation_guild/) para pasos de instalación más detallados y adaptaciones para diferentes plataformas (`Windows`, `Linux` o `Mac OS`).

Puedes usar `conda install omicverse -c conda-forge` o `pip install -U omicverse` para la instalación.

Consulta la documentación y tutoriales en [página de omicverse](https://starlitnightly.github.io/omicverse/) o [omicverse.readthedocs.io](https://omicverse.readthedocs.io/en/latest/index.html).

Los flujos de trabajo con agentes —OpenClaw, el servidor MCP y los bots de mensajería J.A.R.V.I.S.— se documentan en [OmicClaw](https://github.com/omicverse/omicclaw).

## `3` [Cobertura de funciones](https://omicverse.readthedocs.io/en/latest/Feature_coverage/)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../README.assets/feature_coverage_dark.png">
  <img src="../README.assets/feature_coverage_light.png" width="90%" alt="Feature coverage of omicverse, the scverse ecosystem and ten other frameworks">
</picture>

Cobertura de funciones de omicverse frente al ecosistema scverse y otros diez frameworks en 75 funciones de análisis verificadas. La tabla completa con evidencia por celda, el marco de datos y la lista de herramientas publicadas incluidas en OmicVerse están en la [documentación](https://omicverse.readthedocs.io/en/latest/Feature_coverage/).

## `4` [Contacto](#)

- Zehua Zeng ([starlitnightly@gmail.com](mailto:starlitnightly@gmail.com) o [steorra@omicos.cn](mailto:steorra@omicos.cn))
- Meng Xu ([mengxu@omicos.cn](mailto:mengxu@omicos.cn))

## `5` [Guía del Desarrollador y Contribución](#)

Si deseas contribuir a omicverse, consulta nuestra [documentación para desarrolladores](https://omicverse.readthedocs.io/en/latest/Developer_guild/).

### Ejecutar pruebas localmente

Instala las dependencias de prueba y ejecuta la suite con `pytest`:

```bash
pip install -e .[tests]
# o instala los requisitos más recientes fijados
pip install -r requirements-latest.txt

pytest
```

El extra opcional `tests` y el archivo `requirements-latest.txt` ya incluyen `pytest-asyncio>=0.23`, que es necesario para las pruebas asíncronas de streaming en `tests/utils/`.

<table align="center">
    <tr>
      <th colspan="2">
        <br><img src="https://contrib.rocks/image?repo=Starlitnightly/omicverse"><br><br>
      </th>
    </tr>
</table>

> [!IMPORTANT]  
> Nos gustaría agradecer a las siguientes cuentas oficiales de WeChat por promover Omicverse.
> <p align="center"> <a href="https://mp.weixin.qq.com/s/egAnRfr3etccU_RsN-zIlg" target="_blank" rel="noreferrer"> <img src="../README.assets/image-20230701163953794.png" alt="linux" width="50" height="50"/> </a> <a href="https://zhuanlan.zhihu.com/c_1257815636945915904?page=3" target="_blank" rel="noreferrer"> <img src="../README.assets/WechatIMG688.png" alt="linux" width="50" height="50"/> </a> </p>

## `6` [Cita](https://doi.org/10.1038/s41467-024-50194-3)

Si usas `omicverse` en tu trabajo, cita la publicación de `omicverse` de la siguiente manera:

> **OmicVerse: a framework for bridging and deepening insights across bulk and single-cell sequencing**
>
> Zeng, Z., Ma, Y., Hu, L. et al.
>
> _Nature Communication_ 16 Jul 2024. doi: [10.1038/s41467-024-50194-3](https://doi.org/10.1038/s41467-024-50194-3).

Aquí hay algunos otros paquetes relacionados, ¡siéntete libre de referenciarlos si los usas!

> **CellOntologyMapper: Consensus mapping of cell type annotation**
>
> Zeng, Z., Wang, X., Du, H. et al.
>
> _imetaomics_ 6 Nov 2025. doi: [10.1002/imo2.70064](https://doi.org/10.1002/imo2.70064).

## `7` [Otros](#)

Si deseas patrocinar el desarrollo de nuestro proyecto, puedes ir al sitio web de afdian (https://ifdian.net/a/starlitnightly) y patrocinarnos.

Copyright © 2024 [112 Lab](https://112lab.asia/). <br />
Este proyecto tiene licencia [GPL3.0](../LICENSE).

<!-- LINK GROUP -->
[docs-feat-provider]: https://starlitnightly.github.io/omicverse/ 

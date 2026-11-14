<img src="https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/firstpage4.jpg" >


<div align="center">
  <a href="../README.md">English</a> | <a href="README_CN.md">中文</a> | <a href="README_ES.md">Español</a> | <a href="README_JP.md">日本語</a> | <a href="README_DE.md">Deutsch</a> | <a href="README_FR.md">Français</a>

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


**`OmicVerse v2`**는 현대 전사체학 및 멀티오믹스 분석을 위한 통합 Python 프로젝트입니다. bulk RNA-seq, 단일세포 분석, 공간 전사체학, 시각화, 모델 기반 분석, 그리고 AI 보조 워크플로까지 포괄합니다. 자세한 내용은 우리의 논문을 참고하세요: [OmicVerse: a framework for bridging and deepening insights across bulk and single-cell sequencing](https://www.nature.com/articles/s41467-024-50194-3)

> [!IMPORTANT]
>
> **별표를 주세요**, GitHub의 모든 릴리스 알림을 지연 없이 받으실 수 있습니다 ~ ⭐️
>
> **OmicVerse**가 마음에 드시고 우리의 사명을 지원하고 싶으시면, 우리의 노력을 지원하기 위해 [💗기부](https://ifdian.net/a/starlitnightly)를 고려해 주세요.

<details>
  <summary><kbd>별표 히스토리</kbd></summary>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Starlitnightly%2Fomicverse&theme=dark&type=Date">
    <img width="100%" src="https://api.star-history.com/svg?repos=Starlitnightly%2Fomicverse&type=Date">
  </picture>
</details>

## `1` [소개][docs-feat-provider]

**OmicVerse v2**는 현대 전사체학 및 멀티오믹스 분석을 위한 통합 Python 프로젝트입니다. bulk RNA-seq, 단일세포 분석, 공간 전사체학, 시각화, 모델 기반 분석, 그리고 AI 보조 워크플로를 하나의 패키지와 문서 체계 안에 함께 제공합니다.

> [!NOTE]
> OmicVerse v2는 이제 단일 방법 중심 패키지라기보다 더 넓은 분석 플랫폼으로 구성됩니다. 핵심 분석 모듈 외에도 **J.A.R.V.I.S.** 기반의 에이전트형 워크플로, AI 클라이언트를 위한 MCP 기반 도구 서빙, 그리고 `omicverse_guide` 아래에서 계속 확장되는 문서 및 튜토리얼 시스템을 포함합니다.

![omicverse-light](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/background_light.png#gh-light-mode-only)
![omicverse-dark](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/background_dark.png#gh-dark-mode-only)

## `2` [시작하기](#)

OmicVerse는 conda 또는 pypi를 통해 설치할 수 있으며 먼저 `pytorch`를 설치해야 합니다. 더 자세한 설치 단계와 다양한 플랫폼(`Windows`, `Linux` 또는 `Mac OS`)에 대한 적응은 [설치 튜토리얼](https://starlitnightly.github.io/omicverse/Installation_guild/)을 참조하세요.

설치를 위해 `conda install omicverse -c conda-forge` 또는 `pip install -U omicverse`를 사용할 수 있습니다.

[omicverse 페이지](https://starlitnightly.github.io/omicverse/) 또는 [omicverse.readthedocs.io](https://omicverse.readthedocs.io/en/latest/index.html)에서 문서와 튜토리얼을 확인하세요.

에이전트 워크플로(OpenClaw, MCP 서버, J.A.R.V.I.S. 메시지 봇)는 [OmicClaw](https://github.com/omicverse/omicclaw)에서 안내합니다.

## `3` [기능 범위](https://omicverse.readthedocs.io/en/latest/Feature_coverage/)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../README.assets/feature_coverage_dark.png">
  <img src="../README.assets/feature_coverage_light.png" width="90%" alt="Feature coverage of omicverse, the scverse ecosystem and ten other frameworks">
</picture>

검증된 75개 분석 기능에서 omicverse, scverse 생태계와 다른 10개 프레임워크의 기능 범위 비교입니다. 셀별 근거가 포함된 전체 표, 데이터 프레임워크, OmicVerse에 포함된 공개 도구 목록은 [문서](https://omicverse.readthedocs.io/en/latest/Feature_coverage/)에서 확인할 수 있습니다.

## `4` [연락처](#)

- Zehua Zeng ([starlitnightly@gmail.com](mailto:starlitnightly@gmail.com) 또는 [steorra@omicos.cn](mailto:steorra@omicos.cn))
- Meng Xu ([mengxu@omicos.cn](mailto:mengxu@omicos.cn))

## `5` [개발자 가이드 및 기여](#)

omicverse에 기여하고 싶으시면, [개발자 문서](https://omicverse.readthedocs.io/en/latest/Developer_guild/)를 참조하세요.

### 로컬에서 테스트 실행하기

테스트 의존성을 설치하고 `pytest`로 테스트를 실행합니다:

```bash
pip install -e .[tests]
# 또는 고정된 최신 요구 사항 설치
pip install -r requirements-latest.txt

pytest
```

선택적 `tests` extra 및 `requirements-latest.txt` 파일에는 이미 `pytest-asyncio>=0.23`가 포함되어 있으며, 이는 `tests/utils/` 디렉토리의 비동기 스트리밍 테스트에 필요합니다.

<table align="center">
    <tr>
      <th colspan="2">
        <br><img src="https://contrib.rocks/image?repo=Starlitnightly/omicverse"><br><br>
      </th>
    </tr>
</table>

> [!IMPORTANT]  
> Omicverse를 홍보해주신 다음 WeChat 공식 계정들에게 감사드립니다.
> <p align="center"> <a href="https://mp.weixin.qq.com/s/egAnRfr3etccU_RsN-zIlg" target="_blank" rel="noreferrer"> <img src="../README.assets/image-20230701163953794.png" alt="linux" width="50" height="50"/> </a> <a href="https://zhuanlan.zhihu.com/c_1257815636945915904?page=3" target="_blank" rel="noreferrer"> <img src="../README.assets/WechatIMG688.png" alt="linux" width="50" height="50"/> </a> </p>

## `6` [인용](https://doi.org/10.1038/s41467-024-50194-3)

작업에서 `omicverse`를 사용하시면, 다음과 같이 `omicverse` 발행물을 인용해 주세요:

> **OmicVerse: a framework for bridging and deepening insights across bulk and single-cell sequencing**
>
> Zeng, Z., Ma, Y., Hu, L. et al.
>
> _Nature Communication_ 2024년 7월 16일. doi: [10.1038/s41467-024-50194-3](https://doi.org/10.1038/s41467-024-50194-3).

다른 관련 패키지들도 있습니다. 사용하시면 언제든지 참조해 주세요!

> **CellOntologyMapper: Consensus mapping of cell type annotation**
>
> Zeng, Z., Wang, X., Du, H. et al.
>
> _imetaomics_ 2025년 11월 6일. doi: [10.1002/imo2.70064](https://doi.org/10.1002/imo2.70064).

## `7` [기타](#)

우리 프로젝트의 개발을 후원하고 싶으시면, afdian 웹사이트(https://ifdian.net/a/starlitnightly)로 가서 후원해 주세요.

Copyright © 2024 [112 Lab](https://112lab.asia/). <br />
이 프로젝트는 [GPL3.0](../LICENSE) 라이센스를 따릅니다.

<!-- LINK GROUP -->
[docs-feat-provider]: https://starlitnightly.github.io/omicverse/ 

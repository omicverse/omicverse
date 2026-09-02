r"""Bonsai integration — driver only, no upstream source.

Bonsai reconstructs a maximum-likelihood *tree* over cells instead of a 2-D
embedding: cells are leaves, internal vertices are inferred ancestral states,
and branch lengths are distances in expression space. It consumes the
measurement error on each feature, so a poorly-determined cell sits on a short
branch near its parent rather than being scattered by noise.

What is in this directory
-------------------------
Only omicverse's own driver code. **No Bonsai source is vendored here**, unlike
most of ``omicverse/external/``: upstream is released under **CC BY-NC 4.0**
(Attribution-NonCommercial), an additional restriction that omicverse's GPL-3.0
licence cannot carry, so redistributing it inside this package is not available
to us. ``_run.py`` drives a copy the user installs themselves, the arrangement
omicverse already uses for Spateo.

Installing Bonsai
-----------------
Bonsai is not on PyPI (the PyPI package named ``bonsai`` is unrelated)::

    git clone https://github.com/dhdegroot/Bonsai-data-representation.git
    cd Bonsai-data-representation && pip install -r requirements.txt
    export BONSAI_HOME=$PWD

**Commercial users cannot use Bonsai under CC BY-NC 4.0** and should contact the
authors (Biozentrum, University of Basel) first.

Public API
----------
:func:`omicverse.tl.bonsai` and :func:`omicverse.pl.bonsai`.

Reference
---------
de Groot, D. H., Morillo Leonardo, S. X., Pachkov, M., & van Nimwegen, E. (2026).
Bonsai-data-representation (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.20370956
"""
from ._run import run_bonsai
from ._layout import equal_angle_layout, scout_layout

__all__ = ["run_bonsai", "scout_layout", "equal_angle_layout"]

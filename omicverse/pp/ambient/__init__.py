r"""Ambient / contamination-RNA removal for droplet scRNA-seq — ``ov.pp.ambient``.

Ambient ("soup") RNA — cell-free transcripts released by lysed cells —
contaminates every droplet during library prep. Left uncorrected it
inflates marker genes in cell types that never expressed them and biases
downstream DE, annotation and trajectory inference. ``ov.pp.ambient`` is
the QC step that strips that contamination back out.

The sub-package threads four native, pure-Python R-parity backends and
two optional deep-learning wrappers behind one ``method=`` dispatcher:

============  ==============  ==========================================
method        backend         input requirement
============  ==============  ==========================================
``soupx``     ``pysoupx``      raw unfiltered matrix + empty droplets
``fastcar``   ``pyfastcar``    raw unfiltered matrix + empty droplets
``decontx``   ``pydecontx``    filtered, *clustered* matrix
``sccdc``     ``pysccdc``      filtered, *clustered* matrix
``cellbender`` (CellBender)    optional heavyweight DL — not bundled
``scar``       (scAR)          optional heavyweight DL — not bundled
============  ==============  ==========================================

Install the native backends with::

    pip install omicverse[ambient]

The deep-learning backends (CellBender, scAR) are heavyweight and must be
installed directly; ``ov.pp.ambient`` raises a clean, actionable
:class:`ImportError` if a method is requested without its backend.

Public API
----------
Correction    :func:`remove_ambient`, :func:`estimate_contamination`
Diagnostics   :func:`ambient_negative_marker_check`,
              :func:`count_integrity_check`, :func:`contamination_report`
Plotting      :func:`plot_contamination`

Quick-start
-----------
>>> import omicverse as ov
>>> # SoupX / FastCAR — need the raw unfiltered droplets
>>> ov.pp.ambient.remove_ambient(adata, method='soupx', raw=raw_adata)
>>> # DecontX / scCDC — need a clustered filtered matrix
>>> ov.pp.ambient.remove_ambient(adata, method='decontx', cluster_key='leiden')
>>> # diagnostics
>>> ov.pp.ambient.count_integrity_check(adata.layers['ambient_raw'], adata.X)
>>> ov.pp.ambient.contamination_report(adata)
"""
from __future__ import annotations

from ._ambient import remove_ambient, estimate_contamination
from ._diagnostics import (
    ambient_negative_marker_check,
    count_integrity_check,
    contamination_report,
)
from ._plotting import plot_contamination

__all__ = [
    "remove_ambient",
    "estimate_contamination",
    "ambient_negative_marker_check",
    "count_integrity_check",
    "contamination_report",
    "plot_contamination",
]

__version__ = "0.1.0"

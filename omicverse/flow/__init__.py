r"""``ov.flow`` — flow, spectral and mass cytometry.

Cytometry is a modality, not a sub-analysis of single-cell: an event is a cell,
but the file format, the display scalings, the compensation model and above all
the sequential gating hierarchy have no analogue anywhere else in omicverse.

WHAT LIVES HERE
    display transforms  logicle / hyperlog / asinh / log / linear, derived from
                        the GatingML 2.0 + Parks 2006 specifications and
                        verified bit-for-bit against the reference C
                        implementation as a black-box oracle.
    gate geometry       rectangle / polygon / ellipsoid / quadrant / boolean.
    gating strategy     the TREE, owned by the analysis rather than by a
                        sample, so one strategy applies to a whole batch.
    plots               biaxial density, histograms, back-gating, the hierarchy
                        and the spillover matrix — drawn in the gate's OWN
                        transform, because a gate is only checkable by looking
                        at it. See ``plotting.py``.

WHAT DOES NOT
    Reading FCS files — that is ``ov.io.read_fcs``, because it is I/O. This
    module never becomes a second reader.
    Single-EV proteomics — ``ov.single.ev`` owns ``uns['ev']``, the MISEV
    panels and the vesicle vocabulary. Nothing here reads or writes them.
    Unsupervised clustering — ``ov.pp`` and ``ov.single`` already do it, and
    the pure-numpy SOM behind ``flowsom`` is shared rather than duplicated.

WHY IT IS WRITTEN RATHER THAN WRAPPED
    ``flowutils`` >= 1.2 requires ``numpy>=2`` while other cytometry packages
    require ``numpy<2``; a hard dependency would force a numpy major on every
    omicverse user. Its logicle is a C extension whose published root finder
    derives from Numerical Recipes, which cannot be redistributed. And a
    spec-derived core can be reimplemented again in the browser for an
    interactive gate editor with no notice obligation — a wrapper cannot.
    See ``_transforms.py`` for the full argument and the parity evidence.
"""

from ._cluster import flowsom
from ._compensate import (
    compensate,
    spillover_spreading_matrix,
    spillover_to_compensation,
)
from ._gates import (
    BooleanGate,
    EllipsoidGate,
    Gate,
    PolygonGate,
    QuadrantGate,
    RectangleGate,
    gate_from_dict,
    points_in_polygon,
)
from ._som import SOM, som_metacluster
from ._gatingml import (
    from_gatingml,
    is_valid_xml_id,
    read_gatingml,
    sanitize_id,
    to_gatingml,
    write_gatingml,
)
from ._strategy import GatingResult, GatingStrategy
from .plotting import (
    backgate,
    biaxial,
    flowsom_heatmap,
    hierarchy,
    histogram,
    spillover_heatmap,
)
from ._transforms import (
    Asinh,
    Hyperlog,
    Linear,
    Log,
    Logicle,
    Transform,
    make_transform,
    transform_from_dict,
)

__all__ = [
    # transforms
    "Transform", "Linear", "Log", "Asinh", "Logicle", "Hyperlog",
    "make_transform", "transform_from_dict",
    # gates
    "Gate", "RectangleGate", "PolygonGate", "EllipsoidGate", "QuadrantGate",
    "BooleanGate", "gate_from_dict", "points_in_polygon",
    # strategy
    "GatingStrategy", "GatingResult",
    # compensation
    "compensate", "spillover_to_compensation", "spillover_spreading_matrix",
    # interchange
    "to_gatingml", "from_gatingml", "write_gatingml", "read_gatingml",
    "is_valid_xml_id", "sanitize_id",
    # clustering
    "flowsom", "SOM", "som_metacluster",
    # plotting
    "biaxial", "histogram", "backgate", "hierarchy",
    "spillover_heatmap", "flowsom_heatmap",
]

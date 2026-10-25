"""Geometry rather than expression — ``ov.space.geom``.

Align serial sections into one frame, stack them into a volume, fill the gaps::

    aligned = ov.space.geom.align([s1, s2, s3])
    volume  = ov.space.geom.stack(aligned, z_spacing=10.0)
    mid     = ov.space.geom.interpolate(aligned[0], aligned[1])

Match single cells between two segmentations of the same section (e.g. DAPI vs
H&E), one-to-one, through a global similarity plus non-rigid deformation::

    pairs = ov.space.geom.match_cells(dapi_xy, he_xy)
    nuclei = ov.space.geom.hematoxylin_channel(he_rgb)   # H&E -> nuclear image

``pySTAligner`` and ``CAST`` align expression and give a shared embedding; this
gives a shared *coordinate frame*, which is what a 3-D reconstruction needs and an
embedding cannot provide.
"""
from ._geom import align, align_pairwise, interpolate, stack
from ._cellmatch import hematoxylin_channel, match_cells, nn_spacing

__all__ = ["align", "align_pairwise", "stack", "interpolate",
           "match_cells", "hematoxylin_channel", "nn_spacing"]

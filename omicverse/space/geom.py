"""Geometry rather than expression — ``ov.space.geom``.

Align serial sections into one frame, stack them into a volume, fill the gaps::

    aligned = ov.space.geom.align([s1, s2, s3])
    volume  = ov.space.geom.stack(aligned, z_spacing=10.0)
    mid     = ov.space.geom.interpolate(aligned[0], aligned[1])

``pySTAligner`` and ``CAST`` align expression and give a shared embedding; this
gives a shared *coordinate frame*, which is what a 3-D reconstruction needs and an
embedding cannot provide.
"""
from ._geom import align, align_pairwise, interpolate, stack

__all__ = ["align", "align_pairwise", "stack", "interpolate"]

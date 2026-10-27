r"""``ov.radiology`` — reading clinical DICOM studies (MRI / CT).

WHAT LIVES HERE
    study geometry   ``read_dicom_study``: series, imaging plane from the slice
                     normal, contrast class, slice order, localizer flag,
                     laterality, pixel <-> patient (mm) mapping, nearest
                     instance of every series to a point (cross-referencing
                     and image citations).
    measurement      ``landmark_measurement``: landmark pixels -> patient mm,
                     two-point distance, midpoint-to-reference distance.
    rendering        ``render_slices`` (aspect-correct, gridded, orientation-
                     labelled crops side by side across sequences) and
                     ``montage`` (instance-labelled overview) — images sized
                     for visual reading.
    profiles         ``path_profile`` / ``abnormal_segments``: signal along a
                     tendon / ligament / nerve / fascia, against its own intact
                     portion, on several sequences at once.

WHAT DOES NOT
    Radiomics feature extraction from a mask (PyRadiomics) and segmentation
    models. No anatomy, diagnosis or threshold tables: what is abnormal is read
    from the images by the caller.
"""

from ._dicom import (DicomSeries, DicomStudy, landmark_measurement, plane_of,
                     read_dicom_study, sequence_weighting)
from ._profile import abnormal_segments, path_profile
from ._render import montage, render_slices

__all__ = [
    "DicomSeries", "DicomStudy", "read_dicom_study", "plane_of", "sequence_weighting",
    "landmark_measurement", "render_slices", "montage", "path_profile", "abnormal_segments",
]

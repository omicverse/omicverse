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
                     ``montage`` (instance-labelled overview) and
                     ``region_sheets`` (each region of the field of view
                     magnified over all slices) — images sized for visual
                     reading.
    profiles         ``path_profile`` / ``abnormal_segments`` / ``profile_slices``: signal along a
                     tendon / ligament / nerve / fascia, against its own intact
                     portion, on several sequences at once.
    defects          ``measure_band_defect``: tear gap and midpoint-to-attachment
                     distance of a band, on coil-shading-normalised slices
                     (``band_flatfield``), on the slice where the band is most
                     conspicuous, ends at each side's own half-way level.

    survey           ``structure_survey``: deterministic, bounded reading sheets +
                     manifest (regional survey per plane; per-structure
                     consecutive-slice crops on every series);
                     ``sheet_cell_to_patient``; ``ledger_add_read`` (reader text
                     -> ledger), ``ledger_hotspots`` (rank locations),
                     ``ledger_concordance``.

WHAT DOES NOT
    Radiomics feature extraction from a mask (PyRadiomics) and segmentation
    models. No anatomy, diagnosis or threshold tables: what is abnormal is read
    from the images by the caller.
"""

from ._dicom import (DicomSeries, DicomStudy, landmark_measurement, plane_of,
                     read_dicom_study, sequence_weighting)
from ._profile import abnormal_segments, measure_band_gap, path_profile, profile_slices
from ._defect import band_flatfield, measure_band_defect
from ._render import montage, region_sheets, render_slices
from ._survey import (DEFAULT_SEVERITY_TERMS, ledger_add_read, ledger_concordance, ledger_hotspots,
                      ledger_principal, draft_principal,
                      sheet_cell_to_patient, structure_survey)

__all__ = [
    "DicomSeries", "DicomStudy", "read_dicom_study", "plane_of", "sequence_weighting",
    "landmark_measurement", "render_slices", "montage", "path_profile", "abnormal_segments", "profile_slices", "measure_band_gap",
    "band_flatfield", "measure_band_defect",
    "region_sheets", "structure_survey", "sheet_cell_to_patient", "ledger_concordance",
    "ledger_add_read", "ledger_hotspots", "ledger_principal", "draft_principal", "DEFAULT_SEVERITY_TERMS",
]

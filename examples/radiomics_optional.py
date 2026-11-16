"""Compare the optional facade and independent package on the same small array.

Use an environment where this OmicVerse checkout is installed or importable.
Before registry publication, install the accepted local wheel for your platform:

    python -m pip install '<path-to-accepted-ov-radiomics-wheel.whl>'

The radiomics extra is a commented metadata placeholder. After registry
publication and activation of the extra, the future command will be:

    pip install 'omicverse[radiomics]'

This example requires the Rust-backed ov-radiomics wheel and does not read or
download patient images, import DICOM readers, or construct an AnnData object.
"""

import numpy as np
import ov_radiomics
from pandas.testing import assert_frame_equal

from omicverse import radiomics


def extract_small_array(api):
    """Use identical identity, geometry, mask and extraction parameters."""
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    image = api.image.from_array(values, affine=np.eye(4), spatial_unit="millimeter")
    mask = api.mask.from_array(
        np.ones(values.shape, dtype=np.uint32), geometry=image.geometry
    )
    sample = api.ImageSample(
        {"sample_id": "small-array", "patient_id": "synthetic"}, image, mask
    )
    config = {
        "profile": "python_golden_v1",
        "discretization": {"method": "fixed_count", "count": 4},
        "aggregation": "direction_mean",
        "gldm_alpha": 0,
        "voxel_array_shift": 0.0,
    }
    return sample.extract(config, families=["firstorder"]).features


if __name__ == "__main__":
    direct = extract_small_array(ov_radiomics)
    adapted = extract_small_array(radiomics)
    assert_frame_equal(direct, adapted, check_exact=True)
    print(direct.to_string(index=False))
    print("The independent package and omicverse.radiomics tables are equal.")

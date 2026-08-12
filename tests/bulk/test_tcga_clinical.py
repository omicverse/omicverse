from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from omicverse.bulk._tcga import pyTCGA


def _tcga_with(clinical_sheet: pd.DataFrame, case_ids=("P1", "P2", "P3")):
    tcga = pyTCGA.__new__(pyTCGA)
    tcga.clinical_sheet = clinical_sheet
    tcga.adata = AnnData(
        X=np.ones((len(case_ids), 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"Case ID": list(case_ids)},
            index=[f"sample_{i}" for i in range(len(case_ids))],
        ),
    )
    return tcga


def test_survial_init_accepts_current_gdc_prefixed_columns():
    clinical = pd.DataFrame(
        {
            "case_submitter_id": ["P1", "P1", "P2"],
            "demographic.vital_status": ["Alive", "Alive", "Dead"],
            "demographic.days_to_death": [np.nan, np.nan, "410"],
            "diagnoses.days_to_last_follow_up": ["100", "250", "--"],
            "demographic.age_at_index": [50, 50, 60],
            "diagnoses.tumor_grade": ["G2", "G2", "G3"],
        }
    )
    tcga = _tcga_with(clinical)

    result = tcga.survial_init()

    assert result is None
    assert tcga.adata.obs_names.tolist() == ["sample_0", "sample_1"]
    assert tcga.adata.obs["vital_status"].tolist() == ["Alive", "Dead"]
    assert tcga.adata.obs["days"].tolist() == pytest.approx([250.0, 410.0])
    assert tcga.s_pd.index.tolist() == ["P1", "P2"]
    assert tcga.s_pd.loc["P1", "days_to_last_follow_up"] == 250.0


def test_survial_init_accepts_explicit_column_mapping_and_obs_key():
    clinical = pd.DataFrame(
        {
            "participant": ["P1", "P2"],
            "status": ["Alive", "Dead"],
            "follow_up": [365, np.nan],
            "death": [np.nan, 730],
        }
    )
    tcga = _tcga_with(clinical, case_ids=("P1", "P2"))
    tcga.adata.obs.rename(columns={"Case ID": "patient"}, inplace=True)

    tcga.survial_init(
        clinical_columns={
            "case_submitter_id": "participant",
            "vital_status": "status",
            "days_to_last_follow_up": "follow_up",
            "days_to_death": "death",
        },
        obs_case_id="patient",
    )

    assert tcga.adata.obs["days"].tolist() == pytest.approx([365.0, 730.0])


def test_survial_init_preserves_legacy_unprefixed_columns():
    clinical = pd.DataFrame(
        {
            "case_submitter_id": ["P1", "P2"],
            "vital_status": ["Alive", "Dead"],
            "days_to_last_follow_up": [100, np.nan],
            "days_to_death": [np.nan, 200],
            "age_at_index": [50, 60],
            "tumor_grade": ["G1", "G2"],
        }
    )
    tcga = _tcga_with(clinical, case_ids=("P1", "P2"))

    tcga.survial_init()

    assert tcga.adata.obs["days"].tolist() == pytest.approx([100.0, 200.0])
    assert tcga.s_pd["tumor_grade"].tolist() == ["G1", "G2"]


def test_survial_init_reports_missing_required_clinical_field():
    tcga = _tcga_with(pd.DataFrame({"case_submitter_id": ["P1"]}))

    with pytest.raises(KeyError, match="vital_status.*clinical_columns"):
        tcga.survial_init()

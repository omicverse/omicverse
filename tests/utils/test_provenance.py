"""ov.utils provenance of validated defaults (synthetic)."""
import warnings
import numpy as np
import pandas as pd
import pytest

from omicverse.utils._provenance import (DefaultOverrideWarning, validated_defaults, provenance_log,
                                         provenance_report, clear_provenance, write_provenance)


@validated_defaults("alpha", note="validated on a toy generator")
def _f(x, *, alpha=0.75, beta=1):
    return pd.DataFrame({"x": [x * alpha * beta]})


def test_default_kept_records_nothing():
    clear_provenance()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _f(2.0, beta=3)                     # beta is not validated
    assert "ov_provenance" not in out.attrs and provenance_log() == []
    assert "all validated defaults kept" in provenance_report()


def test_override_without_reason_warns_and_records(tmp_path):
    clear_provenance()
    with pytest.warns(DefaultOverrideWarning, match="alpha=1.0"):
        out = _f(2.0, alpha=1.0)
    rec = out.attrs["ov_provenance"][0]
    assert rec["parameter"] == "alpha" and rec["default"] == 0.75 and rec["reason"] is None
    assert "NO REASON GIVEN" in provenance_report()
    p = write_provenance(str(tmp_path / "prov.json"))
    assert "alpha" in open(p).read()


def test_override_with_reason_is_silent_but_recorded():
    clear_provenance()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _f(2.0, alpha=1.0, override_reason="spike-in ground truth")
    assert out.attrs["ov_provenance"][0]["reason"] == "spike-in ground truth"
    assert "0 unexplained / 1" in provenance_report()


def test_library_function_is_wrapped():
    from omicverse.tl._channel_qc import channel_qc_table
    assert "flat_min_sec" in channel_qc_table.__validated_defaults__ if hasattr(channel_qc_table, "__validated_defaults__") \
        else True
    clear_provenance()
    x = np.random.default_rng(0).normal(scale=6, size=(4, 256 * 60))
    with pytest.warns(DefaultOverrideWarning):
        q = channel_qc_table(x, 256.0, flat_min_sec=10.0)
    assert q.attrs["ov_provenance"][0]["parameter"] == "flat_min_sec"


# ── deliverable drift ────────────────────────────────────────────────────────

def _labeler_output(result, args):
    return result, ["cohort", "sample"], list(args["fields"])


@validated_defaults("alpha", output=_labeler_output)
def _labeler(n, fields=("disease",), *, alpha=0.5):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"cohort": ["A"] * n, "sample": [f"s{i}" for i in range(n)]})
    for f in fields:
        df[f] = np.where(rng.random(n) < alpha, "case", "control")
    return df


def test_check_deliverable_match_and_unexplained(tmp_path, monkeypatch):
    from omicverse.utils._provenance import check_deliverable, declare_departure
    monkeypatch.setenv("OV_PROVENANCE_DIR", str(tmp_path / "ledger"))
    lib = _labeler(40, fields=("disease", "sex"))
    p = tmp_path / "deliv.tsv"
    lib.to_csv(p, sep="\t", index=False)
    res = check_deliverable(str(p))
    assert set(res.status) == {"match"} and len(res) == 2
    # hand-written replacement of one column
    own = lib.copy()
    own.loc[:9, "disease"] = np.where(own.loc[:9, "disease"] == "case", "control", "case")
    own.to_csv(p, sep="\t", index=False)
    res = check_deliverable(str(p))
    row = res.set_index("column").loc["disease"]
    assert row.status == "UNEXPLAINED" and abs(row.agreement - 0.75) < 1e-9
    assert "UNEXPLAINED" in res.attrs["summary"]
    assert res.set_index("column").loc["sex"].status == "match"
    declare_departure("_labeler", "external assay on 10 samples", columns=["disease"])
    res = check_deliverable(own)
    assert res.set_index("column").loc["disease"].status == "departs"


def test_check_deliverable_unrelated_file(tmp_path, monkeypatch):
    from omicverse.utils._provenance import check_deliverable
    monkeypatch.setenv("OV_PROVENANCE_DIR", str(tmp_path / "ledger"))
    _labeler(5)
    res = check_deliverable(pd.DataFrame({"image_id": [1, 2], "identity": ["a", "b"]}))
    assert res.empty and "no recorded validated output" in res.attrs["summary"]


def test_declare_departure_needs_reason(tmp_path, monkeypatch):
    from omicverse.utils._provenance import declare_departure
    monkeypatch.setenv("OV_PROVENANCE_DIR", str(tmp_path / "ledger"))
    with pytest.raises(ValueError):
        declare_departure("x", "  ")

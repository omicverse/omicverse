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

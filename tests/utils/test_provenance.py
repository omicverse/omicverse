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
    # records follow the contract schema
    from omicverse.utils._provenance import _read_ledger
    r = _read_ledger("outputs.jsonl")[-1]
    assert r["schema"] == "verse.provenance.v1" and r["output_hash"] and r["inputs"]["n"]


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


# ── compare_to_default ───────────────────────────────────────────────────────

def _confounded(seed=0, n=240):
    rng = np.random.default_rng(seed)
    batch = rng.integers(0, 4, n)
    p = np.array([0.9, 0.85, 0.15, 0.1])[batch]          # label coupled with batch
    y = (rng.random(n) < p).astype(int)
    x = y + rng.normal(scale=0.9, size=n)                 # weak but real biology
    return x, y, batch


def test_compare_to_default_rejects_batch_reader(tmp_path, monkeypatch):
    from sklearn.linear_model import LogisticRegression
    from omicverse.utils._provenance import compare_to_default
    monkeypatch.setenv("OV_PROVENANCE_DIR", str(tmp_path / "ledger"))
    x, y, b = _confounded()

    def default(tr, te):
        return LogisticRegression().fit(x[tr, None], y[tr]).predict(x[te, None])

    def batch_prior(tr, te):        # "predict the label from the technical batch"
        maj = pd.Series(y[tr]).groupby(b[tr]).mean()
        glob = y[tr].mean()
        return np.array([int(maj.get(v, glob) >= 0.5) for v in b[te]])

    res = compare_to_default(y, default, batch_prior, groups=b, confounder=b, n_candidates=5,
                             function="toy_classifier")
    assert res["verdict"] == "keep_default"
    assert res["default_score"] > res["override_score"]


def test_compare_to_default_winners_curse(tmp_path, monkeypatch):
    """The best of 30 noise 'models' picked on the reference must not pass."""
    from omicverse.utils._provenance import compare_to_default
    monkeypatch.setenv("OV_PROVENANCE_DIR", str(tmp_path / "ledger"))
    rng = np.random.default_rng(5)
    n = 200
    y = rng.integers(0, 2, n)
    g = rng.integers(0, 4, n)
    cands = [rng.integers(0, 2, n) for _ in range(30)]
    best = max(cands, key=lambda c: (c == y).mean())          # selected on the same data
    res = compare_to_default(y, lambda tr, te: np.full(len(te), 0), lambda tr, te: best[te],
                             groups=g, n_candidates=30)
    assert res["verdict"] == "keep_default"


def test_compare_to_default_accepts_real_improvement(tmp_path, monkeypatch):
    from sklearn.linear_model import LogisticRegression
    from omicverse.utils._provenance import compare_to_default, _read_ledger
    monkeypatch.setenv("OV_PROVENANCE_DIR", str(tmp_path / "ledger"))
    x, y, b = _confounded(1, 400)

    def constant(tr, te):
        return np.full(len(te), int(round(y[tr].mean())))

    def logistic(tr, te):
        return LogisticRegression().fit(x[tr, None], y[tr]).predict(x[te, None])

    res = compare_to_default(y, constant, logistic, groups=b, confounder=b, function="toy_classifier",
                             columns=["label"])
    assert res["verdict"] == "override_supported"
    a = _read_ledger("audits.jsonl")[-1]
    assert a["function"] == "toy_classifier" and a["verdict"] == "override_supported" and a["digest"]


def test_omicos_envelopes_and_audit_override(tmp_path, monkeypatch):
    """omicos.provenance/1: signed library_default + override_audit bound to the file hash."""
    import hashlib, hmac, json, os
    from sklearn.linear_model import LogisticRegression
    from omicverse.utils._provenance import audit_override
    monkeypatch.setenv("OMICOS_PROVENANCE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("OMICOS_PROVENANCE_KEY", "k3y")
    lib = _labeler(30)
    recs = [json.load(open(tmp_path / "store" / "records" / f))
            for f in os.listdir(tmp_path / "store" / "records")]
    env = recs[0]
    assert env["schema"] == "omicos.provenance/1"
    assert env["mac"] == hmac.new(b"k3y", env["payload"].encode(), hashlib.sha256).hexdigest()
    pl = json.loads(env["payload"])
    assert pl["kind"] == "library_default" and pl["output"]["table"]["key"] == ["cohort", "sample"]
    assert (tmp_path / "store" / "objects" / pl["output"]["sha256"]).exists()
    x, y, b = _confounded(2)
    cand = tmp_path / "cand.csv"
    lib.to_csv(cand, index=False)
    res = audit_override(pl["id"], str(cand), y=y, groups=b, confounder=b,
                         fit_predict_default=lambda tr, te: LogisticRegression().fit(x[tr, None], y[tr]).predict(x[te, None]),
                         fit_predict_override=lambda tr, te: np.full(len(te), 1))
    assert res["verdict"] == "keep_default"
    au = [json.loads(json.load(open(tmp_path / "store" / "records" / f))["payload"])
          for f in os.listdir(tmp_path / "store" / "records")]
    au = [a for a in au if a["kind"] == "override_audit"][0]
    assert au["verdict"] == "fail" and au["default_id"] == pl["id"]
    assert au["candidate"]["sha256"] == hashlib.sha256(cand.read_bytes()).hexdigest()

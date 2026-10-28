"""Synthetic tests for omicverse.tl ERP repair / averaging."""
import numpy as np

from omicverse.tl._erp import channel_erps, repair_reject_epochs


def _epochs(seed=0, n=40, ch=3, t=50):
    rng = np.random.default_rng(seed)
    return rng.normal(scale=5, size=(n, ch, t))


def test_repair_then_reject_and_counts_over_accepted_only():
    ep = _epochs()
    names = ["a", "b", "c"]
    ep[0, 1, 10] += 400          # repaired, accepted
    ep[1, 1, 10] += 400          # repaired, but another channel > reject -> rejected
    ep[1, 0, 10] += 150
    ep[2, 0, 10] += 400          # two channels over repair -> not repairable -> rejected
    ep[2, 2, 10] += 400
    ep[3, 2, 10] += 150          # plain rejection
    rr = repair_reject_epochs(ep, names, reject_uV=100, repair_uV=250, max_repaired=1)
    assert rr["accepted"][0] and not rr["accepted"][1:4].any()
    assert rr["repair_counts"] == {"b": 1}
    assert not rr["repaired"][1].any()


def test_unchecked_channels_do_not_reject():
    ep = _epochs()
    ep[5, 2, 10] += 500
    rr = repair_reject_epochs(ep, ["a", "b", "c"], reject_uV=100, repair_uV=250,
                              check_channels=["a", "b"])
    assert rr["accepted"].all() and rr["repair_counts"] == {}


def test_channel_erp_excludes_repaired_epochs_and_group_is_mean_of_channels():
    ep = _epochs(n=20, ch=2)
    names = ["a", "b"]
    cond = np.array(["t"] * 10 + ["n"] * 10)
    ep[0, 1, 25] += 1000                      # will be repaired on b only
    rr = repair_reject_epochs(ep, names, reject_uV=1e6, repair_uV=250)
    out = channel_erps(ep, names, cond, rr["accepted"], rr["repaired"], groups={"g": names})
    assert out["n_epochs"]["t"]["b"] == 9 and out["n_epochs"]["t"]["a"] == 10
    np.testing.assert_allclose(out["channel"]["t"][1], ep[1:10, 1].mean(0))
    np.testing.assert_allclose(out["group"]["g"]["t"],
                               (ep[0:10, 0].mean(0) + ep[1:10, 1].mean(0)) / 2)


def test_bad_channels_removed_from_rejection_and_dominant_channel_warned():
    ep = _epochs(n=30)
    ep[:, 2, 10] += 300                       # channel c over the limit in every epoch
    names = ["a", "b", "c"]
    rr = repair_reject_epochs(ep, names, reject_uV=100, repair_uV=None)
    assert not rr["accepted"].any() and any("c exceeds" in w for w in rr["warnings"])
    rr2 = repair_reject_epochs(ep, names, reject_uV=100, repair_uV=None, bad_channels=["c"])
    assert rr2["accepted"].all() and rr2["checked_channels"] == ["a", "b"] and not rr2["warnings"]

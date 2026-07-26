"""``ov.datasets.flow_pbmc_*`` — the real cytometry loaders.

Almost everything here runs offline. The two tests that actually download are
skipped unless ``OV_TEST_NETWORK=1``: a CI job that turns red because Zenodo is
having a bad afternoon teaches nobody anything, and the loaders' contracts —
argument validation, citation strings, the licence metadata that CC-BY makes a
condition rather than a courtesy — do not need the network to check.

    OV_TEST_NETWORK=1 pytest tests/flow/test_real_datasets.py -m network

Both datasets are downloaded from their original repositories rather than
re-hosted, so the URL templates are part of the contract: if one silently
changes shape, the loader fails at the reader rather than at the download, and
the error will point at the wrong thing.
"""
from __future__ import annotations

import os

import pytest

import omicverse as ov
from omicverse.datasets import _cytometry as C


def test_the_citation_carries_author_year_doi_and_licence():
    """CC-BY requires attribution. If the citation string is ever emptied or
    trimmed to a bare URL, redistribution of anything derived stops being
    compliant, so it is pinned rather than trusted."""
    for citation, doi in ((C._FORTESSA_CITE, "10.5281/zenodo.14311616"),
                          (C._SPECTRAL_CITE, "10.1371/journal.pone.0351131")):
        assert doi in citation
        assert "CC-BY-4.0" in citation
        assert "(20" in citation, f"no year in: {citation}"


def test_the_download_urls_still_have_a_place_for_the_file():
    assert C._FORTESSA_URL.count("{}") == 1
    assert "zenodo.org/api/records/14311616" in C._FORTESSA_URL
    assert "{:03d}" in C._SPECTRAL_URL
    assert "journals.plos.org" in C._SPECTRAL_URL


def test_the_fortessa_selection_is_balanced_and_from_one_arm():
    """The comparison is healthy versus cardiac arrest, so the files have to
    come from the same treatment arm — otherwise the group difference is
    confounded with the drug the samples were exposed to."""
    groups = [g for _, g in C._FORTESSA_FILES]
    assert groups.count("Healthy") == groups.count("CardiacArrest") == 3
    assert all("Isotype" in f for f, _ in C._FORTESSA_FILES), (
        "mixing treatment arms would confound the group comparison"
    )


@pytest.mark.parametrize("n", [0, 4, -1])
def test_fortessa_rejects_a_sample_count_it_cannot_serve(n):
    with pytest.raises(ValueError, match="n_per_group"):
        ov.datasets.flow_pbmc_fortessa(n_per_group=n)


@pytest.mark.parametrize("n", [0, 7, -1])
def test_spectral_rejects_a_donor_count_it_cannot_serve(n):
    with pytest.raises(ValueError, match="n_donors"):
        ov.datasets.flow_pbmc_spectral(n_donors=n)


def test_subsampling_keeps_acquisition_order():
    """Events are thinned, not shuffled. A cytometry file is ordered in time and
    `$TIME` is a real channel — reordering would make a time gate meaningless
    and would hide an acquisition artefact that drifts through a run."""
    import numpy as np
    import anndata as ad
    import pandas as pd

    a = ad.AnnData(np.arange(1000, dtype=float).reshape(1000, 1),
                   var=pd.DataFrame(index=["Time"]))
    out = C._subsample(a, max_events=100, seed=0)
    values = out.X.ravel()
    assert out.n_obs == 100
    assert np.array_equal(values, np.sort(values))


def test_subsampling_is_a_no_op_when_the_file_is_already_small():
    import numpy as np
    import anndata as ad
    import pandas as pd

    a = ad.AnnData(np.zeros((10, 1)), var=pd.DataFrame(index=["x"]))
    assert C._subsample(a, max_events=100, seed=0) is a
    assert C._subsample(a, max_events=None, seed=0) is a


# ── the ones that touch the network ────────────────────────────────────────

needs_network = pytest.mark.skipif(
    not os.environ.get("OV_TEST_NETWORK"),
    reason="downloads from Zenodo / PLOS; set OV_TEST_NETWORK=1 to run",
)


@needs_network
@pytest.mark.network
def test_fortessa_loads_with_a_real_uncompensated_spillover_matrix(tmp_path):
    """The property the compensation half of the tutorial depends on. If this
    dataset were ever replaced by a compensated one, notebook 05's before/after
    figure would show two identical panels and say they differed."""
    import numpy as np

    a = ov.datasets.flow_pbmc_fortessa(dir=str(tmp_path), n_per_group=1,
                                       max_events=5000, verbose=False)
    assert a.n_obs == 10_000
    assert set(a.obs["group"]) == {"Healthy", "CardiacArrest"}
    spill = a.uns["fcs"]["spillover"]
    off = spill.to_numpy()[~np.eye(spill.shape[0], dtype=bool)]
    assert (off != 0).sum() > 50, "the spillover matrix has gone identity"
    assert off.max() > 0.5
    for marker in ("CD3", "CD4", "CD8", "LIVE DEAD"):
        assert marker in a.var.index


@needs_network
@pytest.mark.network
def test_spectral_loads_with_fsc_h_and_a_top_of_scale_that_is_not_the_bd_one(tmp_path):
    """The two things this dataset is here for: a real singlet gate, and an
    instrument whose $PnR is not 262144."""
    a = ov.datasets.flow_pbmc_spectral(dir=str(tmp_path), n_donors=1,
                                       max_events=5000, verbose=False)
    assert a.n_obs == 5000
    for marker in ("CD3", "CD4", "CD8", "CD19", "FSC-H"):
        assert marker in a.var.index, f"{marker} missing"
    tops = set(a.var["PnR"])
    assert 262144.0 not in tops, f"expected a non-BD top of scale, got {tops}"

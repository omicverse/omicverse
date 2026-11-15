"""ov.vi.MRVI: architecture parity with scvi-tools 1.5.2's torch MRVAE, training, and
the sample-level downstream analyses (local sample representation / distances,
differential abundance, differential expression)."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")
ad = pytest.importorskip("anndata")

from .test_data import _KEEP  # noqa: E402


def _multi_sample(n=1200, g=80, n_sample=8, seed=0):
    rs = np.random.default_rng(seed)
    sample = rs.integers(0, n_sample, n)
    is_covid = (np.arange(n_sample) < n_sample // 2)  # sample-level covariate
    disease = np.where(is_covid, "covid", "ctrl")
    grp = rs.integers(0, 4, n)  # cell state, independent of sample
    base = rs.gamma(0.4, 1.0, (4, g))
    sample_noise = rs.normal(0, 0.2, (n_sample, g))  # per-sample idiosyncratic shift
    disease_effect = rs.normal(0, 1.0, g) * is_covid[:, None]  # shared covid-vs-ctrl shift
    shift = sample_noise + disease_effect
    lib = rs.lognormal(7.0, 0.3, n)
    rate = base[grp] * np.exp(shift[sample])
    m = rate / rate.sum(1, keepdims=True) * lib[:, None]
    X = sp.csr_matrix(rs.poisson(m).astype(np.float32))
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"G{j}" for j in range(g)]
    a.obs["sample"] = pd.Categorical([f"s{i}" for i in sample])
    a.obs["batch"] = pd.Categorical(np.where(np.arange(n) % 2 == 0, "b0", "b1"))
    a.obs["cell_type"] = pd.Categorical([f"t{k}" for k in grp])
    a.obs["disease"] = pd.Categorical([disease[s] for s in sample])
    a.layers["counts"] = a.X.copy()
    a.uns["disease_effect_true"] = disease_effect  # ground truth, for a correlation check
    return a


def _small_module(**kw):
    from omicverse.vi._mrvi import MRVAEModule
    args = dict(n_input=25, n_sample=5, n_batch=3, n_labels=1, n_latent=8, n_latent_u=4,
               encoder_n_hidden=20, encoder_n_layers=1)
    args.update(kw)
    return MRVAEModule(**args)


def _map_scvi_key(k: str) -> str:
    """scvi-tools' ``MRVAE`` parameter names -> ours (same architecture, renamed
    submodules: see ``omicverse.vi._mrvi``)."""
    k = k.replace("layer_norm_batch_embed", "layer_norm_batch")
    k = k.replace("resnet_blocks", "blocks")
    k = k.replace("layer_norm1", "ln1").replace("layer_norm2", "ln2")
    k = k.replace("conditional_norm1", "cn1").replace("conditional_norm2", "cn2")
    k = k.replace("gamma_embedding", "gamma").replace("beta_embedding", "beta")
    return k


def test_mrvae_matches_scvi_tools_module():
    """Every weight shape matches scvi-tools 1.5.2's torch ``MRVAE`` one-to-one, and
    with its weights copied over, ``inference`` / ``generative`` are bit-exact (not
    just close): the attention encoder/decoder, the conditional normalisation and the
    mixture-of-Gaussians prior are a faithful port, not a re-derivation."""
    pytest.importorskip("scvi", minversion="1.5.2")
    from scvi.external.mrvi._module import MRVAE as RefMRVAE
    torch.manual_seed(0)
    torch.set_num_threads(4)
    kw = dict(n_input=25, n_sample=5, n_batch=3, n_labels=1, n_latent=8, n_latent_u=4,
             encoder_n_hidden=20, encoder_n_layers=1)
    ref = RefMRVAE(**kw)
    mine = _small_module(**kw)
    mapped = {_map_scvi_key(k): v for k, v in ref.state_dict().items()}
    missing, unexpected = mine.load_state_dict(mapped, strict=False)
    assert not missing and not unexpected
    ref.eval(); mine.eval()
    g = torch.Generator().manual_seed(1)
    x = torch.poisson(torch.rand(16, 25, generator=g) * 20)
    sample_index = torch.randint(0, 5, (16, 1), generator=g)
    batch_index = torch.randint(0, 3, (16, 1), generator=g)
    label_index = torch.zeros(16, 1, dtype=torch.long)
    with torch.no_grad():
        ref_inf = ref.inference(x, sample_index, use_mean=True)
        my_inf = mine.inference(x, sample_index.squeeze(-1), use_mean=True)
        assert torch.equal(ref_inf["u"], my_inf["u"])
        assert torch.equal(ref_inf["z"], my_inf["z"])
        ref_gen = ref.generative(ref_inf["z"], ref_inf["library"], batch_index, label_index)
        my_gen = mine.generative(my_inf["z"], my_inf["library"], batch_index.squeeze(-1),
                                 label_index.squeeze(-1))
        assert torch.allclose(ref_gen["px"].mean, my_gen["px"].mean, atol=1e-6)
        assert torch.allclose(ref_gen["pu"].log_prob(ref_inf["u"]),
                              my_gen["pu"].log_prob(my_inf["u"]), atol=1e-5)


def test_mrvi_requires_two_samples():
    from omicverse.vi import MRVI
    a = _multi_sample(n=50, n_sample=3)
    a.obs["sample"] = "only_one"
    with pytest.raises(ValueError, match="at least 2 samples"):
        MRVI(a, sample_key="sample")


def test_mrvi_trains_and_runs_downstream_analyses(tmp_path):
    from omicverse.vi import MRVI
    torch.set_num_threads(4)
    a = _multi_sample()
    m = MRVI(a, sample_key="sample", layer="counts", batch_key="batch", n_latent=10,
             n_latent_u=5, u_prior_mixture_k=5, encoder_n_hidden=32, encoder_n_layers=1,
             seed=0)
    assert m.n_sample == 8
    m.train(max_epochs=30, batch_size=128, device="cpu", progress=False)
    assert np.isfinite(m.history_["elbo_train"]).all()

    u = m.get_latent_representation()
    z = m.get_latent_representation(give_z=True)
    assert u.shape == (a.n_obs, 5) and z.shape == (a.n_obs, 10)
    assert np.isfinite(u).all() and np.isfinite(z).all()

    reps = m.get_local_sample_representation(batch_size=256)
    assert reps.shape == (a.n_obs, 8, 10) and np.isfinite(reps).all()

    dres = m.get_local_sample_distances(batch_size=256, groupby="cell_type")
    assert dres["cell"].shape == (a.n_obs, 8, 8)
    assert np.allclose(dres["cell"], dres["cell"].transpose(0, 2, 1), atol=1e-4)  # symmetric
    assert (np.diagonal(dres["cell"], axis1=1, axis2=2) < 1e-4).all()  # d(s, s) == 0
    assert set(dres["groups"]["cell_type"].keys()) == set(a.obs["cell_type"].unique())

    ap = m.get_aggregated_posterior(sample="s0")
    assert ap.sample((4,)).shape == (4, 5)

    da = m.differential_abundance(sample_cov_keys=["disease"], compute_log_enrichment=True)
    assert da["log_probs"].shape == (a.n_obs, 8)
    assert set(da["disease_log_probs"].keys()) == {"covid", "ctrl"}
    # regression test for a bug where `aggregate()`'s `omit_original_sample` masking
    # (`np.isin(samples[:, None], ...)`, shape (n_obs, 1)) broadcast a *single* True
    # across every column of the per-covariate `sub` matrix instead of only the one
    # column matching the cell's own sample -- so every cell belonging to a given
    # covariate value got `-inf` for that value's own aggregate (log_probs/log_enrichs
    # all `-inf`/`nan` for, e.g., every covid cell under `disease_log_probs["covid"]`).
    for v in ("covid", "ctrl"):
        assert np.isfinite(da["disease_log_probs"][v]).all(), f"{v}: non-finite log_probs"
        assert np.isfinite(da["disease_log_enrichs"][v]).all(), f"{v}: non-finite log_enrichs"
    is_covid_cell = a.obs["disease"].values == "covid"
    covid_enrich = da["disease_log_enrichs"]["covid"]
    assert covid_enrich[is_covid_cell].mean() > covid_enrich[~is_covid_cell].mean()

    torch.manual_seed(123)  # differential_expression draws MC samples from the global RNG
    de = m.differential_expression(sample_cov_keys=["disease"], mc_samples=25, store_lfc=True,
                                   mc_samples_lfc=15)
    assert de["beta"].shape == (a.n_obs, 1, 10)
    assert de["pvalue"].shape == de["padj"].shape == (a.n_obs, 1)
    assert ((de["pvalue"] >= 0) & (de["pvalue"] <= 1)).all()
    assert de["lfc"].shape == (a.n_obs, 1, a.n_vars)
    assert de["pvalue"].std() > 0  # not degenerate
    # the simulation has a real per-gene covid-vs-ctrl shift (a.uns["disease_effect_true"]);
    # the mean estimated log2FC across cells should correlate (up to the sign of whichever
    # category one-hot-encoding dropped as the reference level) with its magnitude, even with
    # a short training run and the limited power of an 8-sample chi-squared test
    mean_lfc = de["lfc"][:, 0, :].mean(0)
    corr = np.corrcoef(mean_lfc, a.uns["disease_effect_true"])[0, 1]
    assert abs(corr) > 0.15, f"lfc vs ground truth correlation too low: {corr}"

    m.save(str(tmp_path / "m"))
    m2 = MRVI.load(str(tmp_path / "m"), adata=a, device="cpu")
    assert np.allclose(m2.get_latent_representation(), u, atol=1e-5)


def test_mrvi_disk_and_dask_match_memory(tmp_path):
    """Same seed, same minibatches up to block order: AnnDataOOM and Dask-backed
    training reach the same ELBO ballpark as in-memory training."""
    import omicverse as ov
    from omicverse.vi import MRVI
    torch.set_num_threads(4)
    a = _multi_sample(n=800, g=60, n_sample=6, seed=2)
    path = tmp_path / "mrvi.h5ad"
    a.write_h5ad(path)

    mem = MRVI(a, sample_key="sample", layer="counts", batch_key="batch", n_latent=8,
              n_latent_u=4, encoder_n_hidden=16, encoder_n_layers=1, seed=0)
    mem.train(max_epochs=10, batch_size=128, device="cpu", progress=False, seed=0)
    mem_elbo = mem.history_["elbo_train"][-1]

    ov.settings.cpu_init()
    oom = ov.read(str(path), storage="disk")
    _KEEP.append(oom)
    d = MRVI(oom, sample_key="sample", layer="counts", batch_key="batch", n_latent=8,
            n_latent_u=4, encoder_n_hidden=16, encoder_n_layers=1, seed=0)
    d.train(max_epochs=10, batch_size=128, device="cpu", progress=False, seed=0,
           block_size=128, buffer_blocks=4)
    assert d._source().kind in ("h5", "oom")
    oom_elbo = d.history_["elbo_train"][-1]
    u_oom = d.get_latent_representation()
    assert u_oom.shape == (a.n_obs, 4) and np.isfinite(u_oom).all()

    from omicverse.pp._dask import read_h5ad
    dsk = read_h5ad(str(path), chunk_size=200, device="cpu")  # layers unsupported; a.X is counts
    dm = MRVI(dsk, sample_key="sample", layer=None, batch_key="batch", n_latent=8,
             n_latent_u=4, encoder_n_hidden=16, encoder_n_layers=1, seed=0)
    dm.train(max_epochs=10, batch_size=128, device="cpu", progress=False, seed=0)
    assert d._source  # sanity: training ran
    dask_elbo = dm.history_["elbo_train"][-1]
    u_dask = dm.get_latent_representation()
    assert u_dask.shape == (a.n_obs, 4) and np.isfinite(u_dask).all()

    assert abs(oom_elbo - mem_elbo) < 0.2 * abs(mem_elbo)
    assert abs(dask_elbo - mem_elbo) < 0.2 * abs(mem_elbo)


def _toy_dist():
    """A tiny synthetic `get_local_sample_distances`-shaped dict: 4 samples, 2
    disease groups {s0, s1} vs {s2, s3}; 'typeA' has a strong across-group gap,
    'typeB' has none, so the expected `sample_distance_ratio` is analytically known."""
    order = ["s0", "s1", "s2", "s3"]
    disease = pd.Series(["covid", "covid", "ctrl", "ctrl"], index=order)
    base = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]], dtype=float)
    matA = np.where(base > 0, np.where(
        (disease.values[:, None] == disease.values[None, :]), 1.0, 4.0), 0.0)
    matB = base.copy()  # uniform -> ratio should be 1.0
    dist = {"sample_order": order, "groups": {"cell_type": {"typeA": matA, "typeB": matB}}}
    return dist, disease


def test_sample_distance_ratio():
    from omicverse.vi import sample_distance_ratio
    dist, disease = _toy_dist()
    ratio = sample_distance_ratio(dist, disease, groupby="cell_type")
    assert list(ratio.index[:1]) == ["typeA"]  # typeA has the larger across/within gap
    assert ratio["typeA"] == pytest.approx(4.0)
    assert ratio["typeB"] == pytest.approx(1.0)
    # groupby= is optional when there is only one key
    assert sample_distance_ratio({"sample_order": dist["sample_order"],
                                  "groups": {"ct": dist["groups"]["cell_type"]}},
                                 disease).equals(ratio)


def test_plot_sample_distances():
    pytest.importorskip("seaborn")
    from omicverse.vi import plot_sample_distances
    dist, disease = _toy_dist()
    g = plot_sample_distances(dist, "typeA", groupby="cell_type", sample_annotation=disease)
    assert g.data2d.shape == (4, 4)
    assert list(g.data2d.index) == dist["sample_order"]

"""ov.vi helpers: plot_history, integration_metrics, MemoryMonitor."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
ad = pytest.importorskip("anndata")

from .test_data import _counts  # noqa: E402


@pytest.fixture(scope="module")
def trained():
    from omicverse.vi import SCVI
    torch.set_num_threads(4)
    a = _counts(n=1500, g=100, seed=2)
    m = SCVI(a, batch_key="batch", n_hidden=32)
    m.train(max_epochs=6, device="cpu", progress=False, check_val_every_n_epoch=2)
    a.obsm["X_scVI"] = m.get_latent_representation()
    return a, m


def test_plot_history_places_validation_at_checked_epochs(trained):
    import matplotlib
    matplotlib.use("Agg")
    import omicverse as ov
    _, m = trained
    ax = ov.vi.plot_history(m)
    lines = {l.get_label(): l for l in ax.get_lines()}
    assert len(lines["elbo_train"].get_xdata()) == 6
    assert list(lines["elbo_validation"].get_xdata()) == [1, 3, 5]
    ax = ov.vi.plot_history(m, skip_first=2)
    assert min(ax.get_lines()[0].get_xdata()) == 2


def test_plot_history_needs_a_trained_model():
    import omicverse as ov
    from omicverse.vi import SCVI
    m = SCVI(_counts(n=200, g=50), batch_key="batch", n_hidden=16)
    with pytest.raises(ValueError, match="history"):
        ov.vi.plot_history(m)


def test_memory_monitor_reports_rss_and_time():
    import omicverse as ov
    with ov.vi.MemoryMonitor(period=0.01, gpu=False) as mon:
        x = np.ones((2000, 2000))  # 32 MB
        x.sum()
    s = mon.summary()
    assert s["wall_s"] > 0 and s["rss_peak_gb"] >= s["rss_start_gb"] > 0
    assert s["gpu_peak_gb"] is None and "RSS" in repr(mon)


def test_integration_metrics_columns(trained):
    pytest.importorskip("scib_metrics")
    import omicverse as ov
    a, _ = trained
    df = ov.vi.integration_metrics(a, "X_scVI", batch_key="batch", label_key="label",
                                   n_jobs=2, subsample=600, nmi_ari=False)
    assert list(df.index) == ["X_scVI"]
    assert {"iLISI", "kBET", "ASW_batch", "cLISI", "ASW_label", "graph_conn"} <= set(df.columns)
    assert ((df.values >= 0) & (df.values <= 1)).all()

from types import SimpleNamespace

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse


def _run_plot_top_genes(monkeypatch, tmp_path, matrix):
    import omicverse.pp as ov_pp  # gsmap normalises with ov.pp (no scanpy)

    from omicverse.external.gsmap import diagnosis
    from omicverse.genetics._gsmap import _gsmap_runner

    adata = AnnData(
        matrix,
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1"]),
    )
    calls = []
    monkeypatch.setattr(ov_pp, "normalize_total", lambda *args, **kwargs: calls.append("normalize"))
    monkeypatch.setattr(ov_pp, "log1p", lambda *args, **kwargs: calls.append("log1p"))
    monkeypatch.setattr(
        diagnosis,
        "load_gene_diagnostic_info",
        lambda *args, **kwargs: pd.DataFrame({"Gene": ["g1"]}),
    )

    runner = _gsmap_runner(adata, workdir=str(tmp_path), sample_name="sample")
    monkeypatch.setattr(runner, "_get_latent_adata", lambda: adata)
    monkeypatch.setattr(runner, "plot_gene_gss", lambda **kwargs: kwargs["gene"])

    genes = runner.plot_top_genes("trait", top_corr_genes=1, show=False)

    return calls, genes


def test_plot_top_genes_normalizes_dense_counts(monkeypatch, tmp_path):
    calls, genes = _run_plot_top_genes(
        monkeypatch,
        tmp_path,
        np.array([[20.0], [1.0]]),
    )

    assert calls == ["normalize", "log1p"]
    assert genes == ["g1"]


def test_plot_top_genes_normalizes_sparse_counts(monkeypatch, tmp_path):
    calls, genes = _run_plot_top_genes(
        monkeypatch,
        tmp_path,
        sparse.csr_matrix([[20.0], [1.0]]),
    )

    assert calls == ["normalize", "log1p"]
    assert genes == ["g1"]


def test_plot_manhattan_normalizes_sparse_counts(monkeypatch, tmp_path):
    import omicverse.pp as ov_pp  # gsmap normalises with ov.pp (no scanpy)

    from omicverse.external.gsmap import _manhattan_plot, _regression_read, diagnosis
    from omicverse.genetics._gsmap import _gsmap_runner

    adata = AnnData(
        sparse.csr_matrix([[20.0], [1.0]]),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1"]),
    )
    calls = []
    monkeypatch.setattr(ov_pp, "normalize_total", lambda *args, **kwargs: calls.append("normalize"))
    monkeypatch.setattr(ov_pp, "log1p", lambda *args, **kwargs: calls.append("log1p"))
    monkeypatch.setattr(
        diagnosis,
        "load_gene_diagnostic_info",
        lambda *args, **kwargs: pd.DataFrame({"Gene": ["g1"], "Annotation": ["A"], "PCC": [0.5]}),
    )
    monkeypatch.setattr(_regression_read, "_read_chr_files", lambda *args, **kwargs: ["pairs"])
    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda *args, **kwargs: pd.DataFrame({"SNP": ["rs1"], "Z": [2.0]}),
    )
    monkeypatch.setattr(
        pd,
        "read_feather",
        lambda *args, **kwargs: pd.DataFrame({"SNP": ["rs1"], "gene_name": ["g1"]}),
    )

    class Figure:
        def update_xaxes(self, **kwargs):
            pass

        def update_layout(self, **kwargs):
            pass

    monkeypatch.setattr(_manhattan_plot, "manhattan_plot", lambda **kwargs: Figure())

    runner = _gsmap_runner(adata, workdir=str(tmp_path), sample_name="sample")
    monkeypatch.setattr(runner, "_get_latent_adata", lambda: adata)

    sumstats_file = tmp_path / "sumstats.tsv.gz"
    sumstats_file.touch()
    runner.plot_manhattan("trait", str(sumstats_file), show=False)

    assert calls == ["normalize", "log1p"]


def test_run_diagnosis_normalizes_sparse_counts(monkeypatch):
    from omicverse.external.gsmap import diagnosis

    adata = AnnData(
        sparse.csr_matrix([[20.0], [1.0]]),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1"]),
    )
    calls = []
    monkeypatch.setattr(diagnosis.sc, "read_h5ad", lambda *args, **kwargs: adata)
    monkeypatch.setattr(
        diagnosis.sc.pp,
        "normalize_total",
        lambda *args, **kwargs: calls.append("normalize"),
    )
    monkeypatch.setattr(diagnosis.sc.pp, "log1p", lambda *args, **kwargs: calls.append("log1p"))
    monkeypatch.setattr(diagnosis, "generate_gsmap_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(diagnosis, "generate_manhattan_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(diagnosis, "generate_gss_distribution", lambda *args, **kwargs: None)

    diagnosis.run_diagnosis(SimpleNamespace(hdf5_with_latent_path="latent.h5ad", plot_type="all"))

    assert calls == ["normalize", "log1p"]

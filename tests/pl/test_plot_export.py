import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData

from omicverse.pl import _scatterplot_backend as scatterplot


def test_continuous_embedding_does_not_use_axes_grid1_inset_axes(monkeypatch):
    import mpl_toolkits.axes_grid1.inset_locator as inset_locator

    adata = AnnData(np.ones((5, 2)))
    adata.obsm["X_umap"] = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
    adata.obs["score"] = np.linspace(0, 1, adata.n_obs)

    def forbidden(*args, **kwargs):
        raise AssertionError("axes_grid1.inset_axes must not be used for continuous embeddings")

    monkeypatch.setattr(inset_locator, "inset_axes", forbidden)
    fig = scatterplot.embedding(adata, "umap", color="score", return_fig=True)

    assert len(fig.axes[0].child_axes) == 1
    plt.close(fig)


def test_continuous_embedding_public_save_writes_pdf_on_its_own_canvas(tmp_path):
    adata = AnnData(np.ones((5, 2)))
    adata.obsm["X_umap"] = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
    adata.obs["score"] = np.linspace(0, 1, adata.n_obs)
    output = tmp_path / "umap.pdf"

    scatterplot.embedding(adata, "umap", color="score", show=False, save=str(output))

    assert output.read_bytes().startswith(b"%PDF")

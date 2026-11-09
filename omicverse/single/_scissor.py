"""Optional Scissor cell selection, using the independent scissor_py backend.

OmicVerse adapter; GPL-3.0-or-later backend installed separately.
Added 2026-10-10. No solver, preprocessing, graph construction or download is
implemented in this wrapper. Importing this module does not import scissor_py.
"""
from .._registry import register_function


def _backend():
    try:
        from scissor_py import scissor_from_anndata
    except ModuleNotFoundError as error:
        if error.name != "scissor_py":
            raise
        raise ImportError(
            "Scissor requires the optional scissor_py backend (Python >=3.10). "
            "Install it in this Python environment with: "
            "python -m pip install git+https://github.com/omicverse/py-Scissors.git@cc630ae828149202b553fb7dfdaeda2d0b94d1d7 — "
            "The unrelated PyPI "
            "package pyscissor is not this backend."
        ) from error
    return scissor_from_anndata


@register_function(
    aliases=["Scissor", "phenotype-associated cells", "表型关联细胞"],
    category="single",
    description=("Select phenotype-associated cells with an explicit bulk phenotype, "
                 "expression layer and cell graph. Optional R-free scissor_py backend; "
                 "official/corrected numerical semantics remain explicit."),
    requires={"obsp": ["an explicitly chosen cell graph"]},
    produces={"obs": ["scissor", "scissor_coef"], "uns": ["scissor"]},
    auto_fix="none",
    examples=["model = ov.single.Scissor(adata, bulk=bulk, phenotype=phenotype, "
              "family='binomial', layer='lognorm', graph_key='connectivities', "
              "positive_label='mutant', semantics='corrected'); "
              "result = model.run(alpha=[.05, .2, .6], foldid=foldid)"],
)
class Scissor:
    """Thin, explicit phenotype-guided cell selection adapter.

    Parameters
    ----------
    adata
        In-memory, non-view AnnData with unique cell/gene IDs. The selected
        layer has cells × genes orientation; no input normalization is guessed.
    bulk
        pandas DataFrame with genes × bulk samples, using unique IDs.
    phenotype
        Sample-ID-indexed Series for Gaussian/Binomial, or a DataFrame with
        exactly ``time`` and ``status`` for Cox (1 event, 0 right-censored).
    family
        ``'gaussian'``, ``'binomial'`` or ``'cox'``.
    layer
        Explicit layer name; None means ``adata.X``. No fallback to other layers.
    graph_key
        Existing ``adata.obsp`` key. Build the graph explicitly before use;
        this wrapper never downloads data or constructs/changes a graph.
    positive_label
        Required for binary classes other than numeric 0/1. Scissor+ points
        toward this class; Cox Scissor+ points toward greater log hazard.
    semantics
        ``'official'`` (default) preserves defined upstream behavior and rejects
        unsafe official Gaussian/Cox graph indexing. ``'corrected'`` must be
        selected explicitly; it changes graph/curvature/stopping/path semantics.
    key_added, overwrite
        Output namespace (default scissor); collisions fail by default. After
        successful fitting and serialization, write ``obs[key_added]`` with
        Scissor+/Scissor-/background, ``obs[key_added + '_coef']``, and an audit
        record in ``uns[key_added]``. Use overwrite=True only intentionally.

    Notes
    -----
    This class does not alter expression, graph, sample IDs or phenotype
    encoding. The backend aligns IDs and jointly quantile-normalizes common
    genes. Graphs created by Scanpy need not match historical Seurat SNN graphs.
    ``result_`` is replaced only after a successful run. The returned
    ScissorResult retains the backend's ``Background`` label; AnnData uses the
    documented display label ``background``. Metadata records that mapping.
    """
    def __init__(self, adata, *, bulk, phenotype, family, layer=None, graph_key,
                 positive_label=None, semantics="official", key_added="scissor",
                 overwrite=False):
        self.adata = adata
        self.bulk = bulk
        self.phenotype = phenotype
        self.family = family
        self.layer = layer
        self.graph_key = graph_key
        self.positive_label = positive_label
        self.semantics = semantics
        self.key_added = key_added
        self.overwrite = overwrite
        self.result_ = None

    def run(self, *, alpha=None, cutoff=.2, random_state=123, foldid=None,
            nfolds=None, lambdas=None, nlambda=100, rlambda=None, thresh=1e-7,
            maxit=100000, engine="numpy", max_dense_bytes=1_000_000_000):
        """Fit ordered alpha/CV/refit and annotate only after complete success.

        Explicit foldid is required for cross-language fold reproducibility.
        The first alpha with selected_fraction < cutoff is returned; otherwise
        the last successful alpha reports cutoff_reached=False. No convergence
        failure is converted to success. Return the backend ScissorResult.
        """
        backend = _backend()
        if self.layer == "X":
            raise ValueError("Use layer=None to select adata.X; the name 'X' is reserved by the backend, so explicitly rename an actual layer named 'X'.")
        if not isinstance(self.graph_key, str) or self.graph_key not in self.adata.obsp:
            raise ValueError(
                "graph_key must name an existing adata.obsp cell graph. Run your "
                "chosen preprocessing/neighbors workflow first and pass its key; "
                "Scissor does not automatically build a graph."
            )
        result = backend(
            self.adata, self.bulk, self.phenotype,
            layer="X" if self.layer is None else self.layer, graph_key=self.graph_key,
            family=self.family, positive_label=self.positive_label,
            semantics=self.semantics, writeback=True, key_added=self.key_added,
            overwrite=self.overwrite, alpha=alpha, cutoff=cutoff,
            random_state=random_state, foldid=foldid, nfolds=nfolds,
            lambdas=lambdas, nlambda=nlambda, rlambda=rlambda, thresh=thresh,
            maxit=maxit, engine=engine, max_dense_bytes=max_dense_bytes,
        )
        self.result_ = result
        return result

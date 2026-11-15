"""Multi-condition expression modelling using pyLemur."""
from __future__ import annotations

from .._optional import import_optional_module
from .._registry import register_function
from .._backend_validation import expression, kwargs_copy, output_key, write_embedding


@register_function(aliases=["pyLemur", "LEMUR"], category="single",
                   description="Fit multi-condition expression regression with pyLemur")
def lemur(adata, *, design, n_embedding=15, layer=None, align=False,
          key_added="X_lemur", model_kwargs=None, fit_kwargs=None,
          alignment_kwargs=None, overwrite=False):
    """Fit pyLemur to variance-stabilized expression and return its model.

    ``design`` is a formula such as ``~ condition``. Counts must be normalized
    and transformed by the caller; this function never silently normalizes.
    Fits on a copy to keep the input expression unchanged. Optional Harmony
    alignment runs only with ``align=True``. The embedding is written in input
    cell order. pyLemur does not implement R lemur's neighborhood/pseudobulk
    hypothesis testing: returned predictions must not be called p-values.
    """
    matrix = expression(adata, layer)
    if not isinstance(design, str) or not design.strip():
        raise ValueError("design must be a nonempty model formula.")
    if isinstance(n_embedding, bool) or not isinstance(n_embedding, int) or not 0 < n_embedding < min(adata.shape):
        raise ValueError("n_embedding must be a positive integer smaller than both data dimensions.")
    output_key(adata, key_added, overwrite=overwrite)
    model_args = kwargs_copy(model_kwargs, reserved=("adata", "design", "n_embedding", "layer", "copy"))
    fit = kwargs_copy(fit_kwargs)
    alignment = kwargs_copy(alignment_kwargs)
    if alignment and not align:
        raise ValueError("alignment_kwargs requires align=True.")
    backend = import_optional_module("pylemur.tl", feature="LEMUR", dependencies=("pylemur",),
                                     install_hint="Install pylemur.")
    work = adata.copy()
    work.X = matrix.copy()
    model = backend.LEMUR(work, design=design, n_embedding=n_embedding, **model_args)
    model.fit(**fit)
    if align:
        model.align_with_harmony(**alignment)
    write_embedding(adata, key_added, model.embedding)
    return model

r"""``ov.es`` — enrichment / gene-set scoring.

Vendored fork of `decoupler.mt
<https://decoupler-py.readthedocs.io/en/latest/api/mt.html>`_'s scoring
kernels (`aucell` / `gsea` / `gsva` / `mdt` / `mlm` / `ora` / `udt` /
`ulm` / `viper` / `waggr` / `zscore`), plus `decouple` / `consensus` /
`query_set`. The decoupler source was copied in tree (under
``omicverse.es._*``) so each kernel can be modified independently — the
plan is to ship per-method GPU accelerations without depending on
upstream decoupler releases. Imports of ``decoupler.*`` were rewritten
to ``omicverse.es.*``; original copyright remains with the decoupler
authors (GPL-3.0).

The public surface accepts **dict-style signatures** (omicverse's
convention) and converts them to decoupler's ``net`` DataFrame
internally — you don't see the ``source/target/weight`` long format
unless you want to.

Example
-------
>>> import omicverse as ov
>>> sigs = {
...     'HALLMARK_INTERFERON_ALPHA': ['IFI6', 'ISG15', 'MX1'],
...     'HALLMARK_INFLAMMATORY':     ['IL6', 'TNF', 'CXCL8'],
... }
>>> ov.es.aucell(adata, signatures=sigs)
>>> adata.obsm['score_aucell']        # cells × signatures DataFrame
>>>
>>> # Weighted / signed signatures (viper / mlm / zscore care about sign):
>>> regulons = {
...     'NFKB': {'TNF': 1.0, 'IL6': 1.0, 'IL10': -1.0},
... }
>>> ov.es.viper(adata, signatures=regulons)

Power users can still pass a raw ``net`` DataFrame via the ``net=``
keyword — the dict path is just the default, more ergonomic option.

Notes
-----
``ov.single.aucell`` (the SCENIC/ctxcore-based legacy path) is retained
for back-compat with pySCENIC workflows that depend on its exact
numerical output or its weighted-regulon / leading-edge side products.
New code should prefer ``ov.es.aucell`` — it is ~15-20× faster
single-threaded and shares preprocessing with the other scoring
methods.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Sequence, Union

import pandas as pd

# omicverse's coloured pre/post-call summary box ("Duration / Shape /
# CHANGES DETECTED" — same one `ov.pp.qc` etc. emit). Wraps each scoring
# call so users see what got added to ``adata.obsm`` for free.
from .._monitor import monitor as _monitor

# Per-method public functions. Each module exposes `<name>()` directly
# (ov.pp-style: decorator stack + Google docstring + inline dispatch).
# The vendored numba CPU + torch GPU kernels live alongside them as
# private helpers (`_func_<name>` / `_func_<name>_torch`).
from ._aucell    import aucell
from ._ucell     import ucell
from ._gsea      import gsea
from ._gsva      import gsva
from ._mdt       import mdt
from ._mlm       import mlm
from ._ora       import ora
from ._udt       import udt
from ._ulm       import ulm
from ._viper     import viper
from ._waggr     import waggr
from ._zscore    import zscore
from ._decouple  import decouple  as _decouple_fn
from ._consensus import consensus as _consensus_fn
from ._query_set import query_set as _query_set_fn

SignatureValue = Union[Sequence[str], Mapping[str, float]]
Signatures = Mapping[str, SignatureValue]


def signatures_to_net(
    signatures: Signatures,
    default_weight: float = 1.0,
) -> pd.DataFrame:
    r"""Convert dict-of-genes to decoupler's long ``net`` DataFrame.

    Accepts two value shapes per signature:

    * ``list``/``tuple``/``set`` of gene names — binary set, all genes
      get ``weight = default_weight``.
    * ``dict`` mapping ``gene → weight`` (float, sign-aware) — passed
      through unchanged. Required for signed regulons (viper / mlm /
      zscore use the sign to distinguish activators from repressors).

    Parameters
    ----------
    signatures
        Mapping ``name → list[str] | dict[str, float]``.
    default_weight
        Weight applied when the value is an unweighted iterable.

    Returns
    -------
    pandas.DataFrame
        Long-format ``net`` with columns ``source``, ``target``,
        ``weight`` — the shape every vendored kernel consumes.
    """
    rows = []
    for name, item in signatures.items():
        if isinstance(item, Mapping):
            for g, w in item.items():
                rows.append({'source': name, 'target': str(g), 'weight': float(w)})
        elif isinstance(item, (list, tuple, set, frozenset)):
            for g in item:
                rows.append(
                    {'source': name, 'target': str(g), 'weight': float(default_weight)}
                )
        else:
            raise TypeError(
                f"signatures[{name!r}] must be list / tuple / set / dict, "
                f"got {type(item).__name__}"
            )
    if not rows:
        raise ValueError("`signatures` is empty.")
    return pd.DataFrame(rows)


def _resolve_net(signatures, net):
    """Either ``signatures`` (dict) or ``net`` (DataFrame), not both."""
    if signatures is not None and net is not None:
        raise ValueError("pass either `signatures` or `net`, not both")
    if signatures is not None:
        return signatures_to_net(signatures)
    if net is None:
        raise ValueError("must pass `signatures` (dict) or `net` (DataFrame)")
    return net


@_monitor
def decouple(
    data,
    signatures=None,
    *,
    net=None,
    methods=None,
    args=None,
    cons: bool = True,
    **kwargs,
):
    """Run multiple scoring kernels in one pass; optional consensus.

    Equivalent of ``decoupler.mt.decouple`` with dict signature input.
    See the original ``decouple`` docstring for ``methods`` / ``args``
    semantics.
    """
    resolved_net = _resolve_net(signatures, net)
    return _decouple_fn(
        data, net=resolved_net, methods=methods, args=args, cons=cons, **kwargs,
    )


_SINGLE_METHOD_DISPATCH = {
    'aucell': lambda: aucell, 'gsea':   lambda: gsea,
    'gsva':   lambda: gsva,   'mdt':    lambda: mdt,
    'mlm':    lambda: mlm,    'ora':    lambda: ora,
    'udt':    lambda: udt,    'ulm':    lambda: ulm,
    'viper':  lambda: viper,  'waggr':  lambda: waggr,
    'zscore': lambda: zscore,
}


try:
    from .._registry import register_function as _register_function
except ImportError:  # pragma: no cover — keep es importable without registry
    def _register_function(*_a, **_k):
        return lambda f: f


@_register_function(
    aliases=[
        "enrichment", "es",
        "decoupler", "score_signatures",
        "通路打分", "富集打分", "基因集打分",
    ],
    category="enrichment",
    description=(
        "Unified ov.es enrichment-score API. The ``method`` argument "
        "picks one of 11 GPU-accelerated scoring kernels — aucell, gsea, "
        "gsva, ora, ulm, mlm, waggr, zscore, viper, mdt, udt — each a "
        "drop-in replacement of the corresponding ``decoupler.mt`` kernel "
        "with an added ``engine='auto' | 'cpu' | 'gpu'`` selector. Writes "
        "scores to ``adata.obsm['score_<method>']`` and (when applicable) "
        "p-values to ``adata.obsm['padj_<method>']``."
    ),
    prerequisites={
        "optional_functions": ["preprocess", "geneset_prepare"],
    },
    requires={
        "var": ["gene symbols matching the signature dict keys"],
    },
    produces={
        "obsm": ["score_<method>", "padj_<method>"],
    },
    auto_fix="none",
    examples=[
        "ov.es.decoupler(adata, signatures=sigs, method='aucell')",
        "ov.es.decoupler(adata, signatures=sigs, method='gsea', engine='gpu', times=1)",
        "ov.es.decoupler(adata, signatures=sigs, method='ulm', engine='cpu', tmin=3)",
        "ov.es.decoupler(adata, signatures=sigs, method='viper', pleiotropy=False)",
    ],
    related=[
        "aucell", "gsea", "gsva", "ora", "ulm", "mlm",
        "waggr", "zscore", "viper", "mdt", "udt",
        "decouple", "consensus",
    ],
)
def decoupler(
    data,
    signatures=None,
    *,
    net=None,
    method: str = 'aucell',
    engine: str = 'auto',
    **kwargs,
):
    r"""Unified dispatcher for the eleven ``ov.es`` scoring methods.

    ``ov.es.decoupler(adata, signatures=sigs, method='aucell', engine='gpu')``
    is equivalent to ``ov.es.aucell(adata, signatures=sigs, engine='gpu')``
    — same kwargs, same outputs in ``adata.obsm[f'score_<method>']``. The
    indirection is useful when the scoring choice should be a parameter
    (sweeps, agent tool calls, configuration files).

    Parameters
    ----------
    data
        AnnData (or DataFrame) to score.
    signatures
        Mapping ``{name → list[gene]}`` or ``{name → dict[gene, weight]}``.
        Mutually exclusive with ``net``.
    net
        Long-format ``source / target / weight`` DataFrame (decoupler
        convention). Power-user escape hatch; ``signatures`` is the default.
    method
        Which kernel to run. One of: ``aucell``, ``gsea``, ``gsva``, ``ora``,
        ``ulm``, ``mlm``, ``waggr``, ``zscore``, ``viper``, ``mdt``, ``udt``.
    engine
        ``'auto'`` (default) | ``'cpu'`` | ``'gpu'``.
    **kwargs
        Forwarded to the chosen kernel.

    Examples
    --------
    >>> ov.es.decoupler(adata, signatures=sigs, method='aucell')
    >>> ov.es.decoupler(adata, signatures=sigs, method='gsea', engine='gpu', times=1)
    """
    if method not in _SINGLE_METHOD_DISPATCH:
        raise ValueError(
            f"method must be one of {sorted(_SINGLE_METHOD_DISPATCH)}, got {method!r}"
        )
    fn = _SINGLE_METHOD_DISPATCH[method]()
    return fn(data, signatures=signatures, net=net, engine=engine, **kwargs)


def consensus(result, verbose: bool = False):
    """Build a consensus score across per-method outputs (Stouffer-like).

    Pass the dict returned by ``ov.es.decouple(..., cons=False)``.
    """
    return _consensus_fn(result, verbose=verbose)


def query_set(
    features,
    signatures=None,
    *,
    net=None,
    alternative: str = 'two-sided',
    n_bg: int = 1000,
    ha_corr: str = 'BH',
    tmin: int = 5,
    verbose: bool = False,
):
    """Hypergeometric-style enrichment test of ``features`` in each signature."""
    resolved_net = _resolve_net(signatures, net)
    return _query_set_fn(
        features,
        net=resolved_net,
        alternative=alternative,
        n_bg=n_bg,
        ha_corr=ha_corr,
        tmin=tmin,
        verbose=verbose,
    )


__all__ = [
    'aucell', 'ucell', 'gsea', 'gsva', 'mdt', 'mlm', 'ora',
    'udt', 'ulm', 'viper', 'waggr', 'zscore',
    'decouple', 'decoupler', 'consensus', 'query_set',
    'signatures_to_net',
]

from __future__ import annotations

from typing import Mapping

import anndata
import pandas as pd


_DEFAULT_RESULT_UNS_KEYS = (
    "liana_res",
    "cpdb_results",
    "cellphonedb_results",
    "cpdb_res",
    "comm_adata",
)
_LIANA_PRIMARY_COLUMNS = {"source", "target", "ligand_complex", "receptor_complex"}


def _is_comm_adata(adata: anndata.AnnData) -> bool:
    return (
        isinstance(adata, anndata.AnnData)
        and {"sender", "receiver"}.issubset(adata.obs.columns)
        and {"means", "pvalues"}.issubset(adata.layers.keys())
    )


def _looks_like_liana_results(value) -> bool:
    return isinstance(value, pd.DataFrame) and _LIANA_PRIMARY_COLUMNS.issubset(value.columns)


def _looks_like_cpdb_results(value) -> bool:
    if not isinstance(value, Mapping):
        return False
    if not isinstance(value.get("means"), pd.DataFrame):
        return False
    return any(
        isinstance(value.get(key), pd.DataFrame)
        for key in ("pvalues", "relevant_interactions")
    )


def _resolve_uns_key(adata: anndata.AnnData, requested_key: str | None) -> str | None:
    if requested_key is not None and requested_key in adata.uns:
        return requested_key
    for candidate in _DEFAULT_RESULT_UNS_KEYS:
        if candidate in adata.uns:
            return candidate
    return None


def to_comm_adata(
    adata: anndata.AnnData | None = None,
    *,
    data=None,
    result_uns_key: str | None = None,
    score_key: str = "specificity_rank",
    pvalue_key: str = "specificity_rank",
    inverse_score: bool = True,
    inverse_pvalue: bool = False,
    classification: str | Mapping[str, str] | None = None,
    classification_reference: str | pd.DataFrame | None = "cellchat",
    classification_fallback: str | None = "family",
    separator: str = "|",
) -> anndata.AnnData:
    """
    Convert a supported CCC result object into OmicVerse communication AnnData.

    Parameters
    ----------
    adata
        Input AnnData. May already be a communication AnnData or may contain
        supported CCC results in ``adata.uns``.
    data
        Explicit result object to convert. Supported inputs are:
        communication AnnData, LIANA result DataFrame, or CellPhoneDB result dict.
    result_uns_key
        Preferred ``adata.uns`` key to inspect when ``data`` is omitted.
    """
    from ._cpdb import format_cpdb_results
    from ._liana import format_liana_results

    if data is not None:
        if isinstance(data, anndata.AnnData) and _is_comm_adata(data):
            return data
        if _looks_like_liana_results(data):
            return format_liana_results(
                liana_res=data,
                score_key=score_key,
                pvalue_key=pvalue_key,
                inverse_score=inverse_score,
                inverse_pvalue=inverse_pvalue,
                classification=classification,
                classification_reference=classification_reference,
                classification_fallback=classification_fallback,
            )
        if _looks_like_cpdb_results(data):
            return format_cpdb_results(data, separator=separator)
        raise TypeError(
            "`data` must be a communication AnnData, LIANA result DataFrame, or "
            "CellPhoneDB result dict."
        )

    if adata is None:
        raise ValueError("Provide either `adata` or `data`.")

    if _is_comm_adata(adata):
        return adata

    resolved_key = _resolve_uns_key(adata, result_uns_key)
    if resolved_key is None:
        raise ValueError(
            "Could not find a supported communication result in `adata.uns`. "
            f"Tried: {list(_DEFAULT_RESULT_UNS_KEYS)}."
        )

    value = adata.uns[resolved_key]
    if isinstance(value, anndata.AnnData) and _is_comm_adata(value):
        return value
    if _looks_like_liana_results(value):
        return format_liana_results(
            adata=adata,
            uns_key=resolved_key,
            score_key=score_key,
            pvalue_key=pvalue_key,
            inverse_score=inverse_score,
            inverse_pvalue=inverse_pvalue,
            classification=classification,
            classification_reference=classification_reference,
            classification_fallback=classification_fallback,
        )
    if _looks_like_cpdb_results(value):
        return format_cpdb_results(value, separator=separator)

    raise ValueError(
        f"`adata.uns['{resolved_key}']` is not a supported communication result. "
        "Expected LIANA results, CellPhoneDB results, or a communication AnnData."
    )


def extract_comm_adata(
    adata: anndata.AnnData | None = None,
    *,
    data=None,
    result_uns_key: str | None = None,
    score_key: str = "specificity_rank",
    pvalue_key: str = "specificity_rank",
    inverse_score: bool = True,
    inverse_pvalue: bool = False,
    classification: str | Mapping[str, str] | None = None,
    classification_reference: str | pd.DataFrame | None = "cellchat",
    classification_fallback: str | None = "family",
    separator: str = "|",
) -> anndata.AnnData:
    """Alias of :func:`to_comm_adata` for users who prefer extraction wording."""
    return to_comm_adata(
        adata=adata,
        data=data,
        result_uns_key=result_uns_key,
        score_key=score_key,
        pvalue_key=pvalue_key,
        inverse_score=inverse_score,
        inverse_pvalue=inverse_pvalue,
        classification=classification,
        classification_reference=classification_reference,
        classification_fallback=classification_fallback,
        separator=separator,
    )

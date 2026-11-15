from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import anndata

from .._registry import register_function


_LIANA_PRIMARY_COLUMNS = ("source", "target", "ligand_complex", "receptor_complex")
_LIANA_CELLCHAT_REFERENCE = {
    "cellchat_human": "cellchat_interactions_and_tfs_human.csv",
    "cellchat_mouse": "cellchat_interactions_and_tfs_mouse.csv",
}


def _get_liana_res(
    *,
    adata=None,
    liana_res: pd.DataFrame | None = None,
    uns_key: str = "liana_res",
) -> pd.DataFrame:
    if liana_res is not None:
        return liana_res.copy()
    if adata is None:
        raise ValueError("Provide either `liana_res` or `adata` with LIANA results in `adata.uns`.")
    if uns_key not in adata.uns:
        raise KeyError(f"`adata.uns['{uns_key}']` not found.")

    value = adata.uns[uns_key]
    if isinstance(value, pd.DataFrame):
        return value.copy()

    raise TypeError(
        f"`adata.uns['{uns_key}']` must be a pandas DataFrame, got {type(value).__name__}."
    )


def _validate_liana_res(liana_res: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in _LIANA_PRIMARY_COLUMNS if column not in liana_res.columns]
    if missing:
        raise ValueError(
            "LIANA results are missing required columns: "
            f"{missing}. Expected {list(_LIANA_PRIMARY_COLUMNS)}."
        )

    result = liana_res.copy()
    for column in _LIANA_PRIMARY_COLUMNS:
        result[column] = result[column].astype(str)

    duplicate_subset = list(_LIANA_PRIMARY_COLUMNS)
    for candidate in ("sample", "context", "condition", "dataset"):
        if candidate in result.columns:
            duplicate_subset.append(candidate)
            result[candidate] = result[candidate].astype(str)
            break

    duplicated = result.duplicated(subset=duplicate_subset, keep=False)
    if duplicated.any():
        duplicate_count = int(duplicated.sum())
        raise ValueError(
            "LIANA results contain duplicated source-target-ligand-receptor combinations "
            f"({duplicate_count} duplicated rows). Aggregate or deduplicate before formatting."
        )
    return result


def _complex_tokens(value, *, uppercase: bool = True) -> list[str]:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    text = text.replace("(", "").replace(")", "").replace(" ", "").replace("+", "_")
    tokens = []
    for token in text.split("_"):
        normalized = re.sub(r"[^A-Za-z0-9-]", "", token)
        if not normalized:
            continue
        tokens.append(normalized.upper() if uppercase else normalized)
    return tokens


def _normalize_complex(value) -> str:
    tokens = _complex_tokens(value, uppercase=True)
    if not tokens:
        return ""
    return "_".join(sorted(dict.fromkeys(tokens)))


def _normalized_lr_key(ligand, receptor) -> str:
    ligand_key = _normalize_complex(ligand)
    receptor_key = _normalize_complex(receptor)
    if not ligand_key or not receptor_key:
        return ""
    return f"{ligand_key}||{receptor_key}"


def _parse_interaction_name_2(value) -> tuple[str, str]:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "", ""
    parts = re.split(r"\s+-\s+", text, maxsplit=1)
    if len(parts) != 2:
        return "", ""
    return parts[0].strip(), parts[1].strip()


def _first_mode(values: pd.Series):
    clean = values.dropna().astype(str).str.strip()
    clean = clean.loc[clean.ne("")]
    if clean.empty:
        return np.nan
    counts = clean.value_counts()
    top = counts.loc[counts == counts.max()].index.tolist()
    return sorted(top)[0]


def _infer_cellchat_reference_name(liana_res: pd.DataFrame) -> str:
    human_like = 0
    mouse_like = 0
    for column in ("ligand_complex", "receptor_complex"):
        for value in liana_res[column].astype(str):
            for token in _complex_tokens(value, uppercase=False):
                letters = re.sub(r"[^A-Za-z]", "", token)
                if not letters:
                    continue
                if letters.isupper():
                    human_like += 1
                elif letters[:1].isupper() and letters[1:] == letters[1:].lower():
                    mouse_like += 1
    return "cellchat_mouse" if mouse_like > human_like else "cellchat_human"


@lru_cache(maxsize=4)
def _load_builtin_classification_reference(name: str) -> pd.DataFrame:
    if name not in _LIANA_CELLCHAT_REFERENCE:
        available = sorted(["cellchat", *_LIANA_CELLCHAT_REFERENCE.keys()])
        raise ValueError(
            f"Unsupported `classification_reference={name!r}`. Available built-ins: {available}."
        )
    base_dir = Path(__file__).resolve().parents[1] / "datasets" / "data_files" / "TF"
    return pd.read_csv(base_dir / _LIANA_CELLCHAT_REFERENCE[name], index_col=0)


def _prepare_reference_frame(reference: pd.DataFrame) -> pd.DataFrame:
    ref = reference.copy()
    classification_column = None
    for candidate in ("classification", "pathway_name", "signaling"):
        if candidate in ref.columns:
            classification_column = candidate
            break
    if classification_column is None:
        raise ValueError(
            "Classification reference must contain one of: "
            "`classification`, `pathway_name`, or `signaling`."
        )

    ligand = pd.Series("", index=ref.index, dtype=object)
    receptor = pd.Series("", index=ref.index, dtype=object)
    if "interaction_name_2" in ref.columns:
        parsed = ref["interaction_name_2"].apply(_parse_interaction_name_2)
        ligand = parsed.map(lambda pair: pair[0])
        receptor = parsed.map(lambda pair: pair[1])
    has_parsed_pairs = "interaction_name_2" in ref.columns
    if {"ligand_complex", "receptor_complex"}.issubset(ref.columns):
        ligand = ligand.where(ligand.astype(str).str.strip() != "", ref["ligand_complex"])
        receptor = receptor.where(receptor.astype(str).str.strip() != "", ref["receptor_complex"])
    elif {"ligand", "receptor"}.issubset(ref.columns):
        ligand = ligand.where(ligand.astype(str).str.strip() != "", ref["ligand"])
        receptor = receptor.where(receptor.astype(str).str.strip() != "", ref["receptor"])
    elif not has_parsed_pairs:
        raise ValueError(
            "Classification reference must contain `interaction_name_2`, "
            "`ligand_complex`/`receptor_complex`, or `ligand`/`receptor` columns."
        )

    prepared = pd.DataFrame(index=ref.index)
    prepared["classification_value"] = ref[classification_column].astype(str)
    prepared["normalized_key"] = [
        _normalized_lr_key(ligand_value, receptor_value)
        for ligand_value, receptor_value in zip(ligand, receptor)
    ]
    prepared = prepared.loc[
        (prepared["normalized_key"].astype(str) != "")
        & (prepared["classification_value"].astype(str).str.strip() != "")
    ]
    return prepared


def _reference_classification_series(
    liana_res: pd.DataFrame,
    *,
    classification_reference: str | pd.DataFrame,
) -> tuple[pd.Series, str]:
    if isinstance(classification_reference, str):
        reference_name = classification_reference
        if reference_name == "cellchat":
            reference_name = _infer_cellchat_reference_name(liana_res)
        reference = _load_builtin_classification_reference(reference_name)
    elif isinstance(classification_reference, pd.DataFrame):
        reference_name = "custom_reference"
        reference = classification_reference
    else:
        raise TypeError(
            "`classification_reference` must be a built-in reference name or a pandas DataFrame."
        )

    prepared_reference = _prepare_reference_frame(reference)
    mapping = prepared_reference.groupby("normalized_key", observed=True)["classification_value"].agg(_first_mode)
    pair_keys = pd.Series(
        [
            _normalized_lr_key(ligand, receptor)
            for ligand, receptor in zip(
                liana_res["ligand_complex"].astype(str),
                liana_res["receptor_complex"].astype(str),
            )
        ],
        index=liana_res.index,
        dtype=object,
    )
    return pair_keys.map(mapping), reference_name


def _infer_family_label(ligand_complex, receptor_complex) -> str:
    tokens = _complex_tokens(ligand_complex, uppercase=True) + _complex_tokens(
        receptor_complex, uppercase=True
    )
    if not tokens:
        return "Unclassified"

    token_set = set(tokens)

    def startswith_any(prefixes) -> bool:
        return any(any(token.startswith(prefix) for prefix in prefixes) for token in token_set)

    if startswith_any(("CXCL", "CXCR", "CX3CL", "CX3CR", "ACKR")):
        return "CXCL"
    if startswith_any(("CCL", "CCR")):
        return "CCL"
    if startswith_any(("XCL", "XCR")):
        return "XCL"
    if startswith_any(("TNF", "TNFR", "TNFRSF", "TNFSF", "FAS", "FASLG", "TRAIL", "LTA", "LTB")):
        return "TNF"
    if startswith_any(("TGFB", "TGFBR")):
        return "TGFb"
    if startswith_any(("BMP", "BMPR", "ACVR", "GDF", "LEFTY", "INHBA", "INHBB")):
        return "BMP"
    if startswith_any(("WNT", "FZD", "LRP", "ROR")):
        return "WNT"
    if startswith_any(("DLL", "JAG", "NOTCH")):
        return "NOTCH"
    if startswith_any(("FGF", "FGFR")):
        return "FGF"
    if startswith_any(("EGF", "EGFR", "ERBB", "HBEGF", "EREG", "NRG", "BTC", "AREG")):
        return "EGF"
    if startswith_any(("VEGF", "FLT", "KDR")):
        return "VEGF"
    if startswith_any(("PDGF", "PDGFR")):
        return "PDGF"
    if startswith_any(("SEMA", "PLXN", "NRP")):
        return "Semaphorin/Plexin"
    if startswith_any(("EFN", "EPH")):
        return "Ephrin/EPH"
    if startswith_any(("LGALS",)):
        return "GALECTIN"
    if startswith_any(("ANXA", "FPR", "F2R")):
        return "ANNEXIN"
    if startswith_any(("CD74", "HLA-D", "H2-AA", "H2-AB", "H2-EB", "H2-DM", "H2-O")):
        return "MHC-II"
    if startswith_any(("B2M", "HLA-A", "HLA-B", "HLA-C", "HLA-E", "HLA-F", "HLA-G", "H2-K", "H2-D", "H2-L")):
        return "MHC-I"
    if startswith_any(("HLA", "H2-")):
        return "Antigen presentation"
    if startswith_any(
        (
            "PDCD1",
            "CD274",
            "PDCD1LG",
            "CTLA4",
            "CD80",
            "CD86",
            "LAG3",
            "HAVCR2",
            "TIGIT",
            "PVR",
            "NECTIN",
            "VSIR",
            "SIRPA",
            "CD47",
            "CD200",
            "CD200R",
        )
    ):
        return "Immune checkpoint"
    if startswith_any(("C1Q", "C2", "C3", "C4", "C5", "CFH", "CFB")):
        return "COMPLEMENT"
    if startswith_any(("FN1",)):
        return "FN1"
    if startswith_any(("LAMA", "LAMB", "LAMC")):
        return "LAMININ"
    if startswith_any(("COL",)):
        return "COLLAGEN"
    if startswith_any(("THBS",)):
        return "THBS"
    if startswith_any(("SPP1",)):
        return "SPP1"
    if startswith_any(("VCAN",)):
        return "VCAN"
    if startswith_any(("MK",)):
        return "MK"
    if startswith_any(("PTN",)):
        return "PTN"
    if startswith_any(("ITGA", "ITGB", "SDC", "CD44", "TNC")):
        return "ECM/Adhesion"
    if startswith_any(("IFN", "IFNAR", "IFNGR", "IL", "ILR", "OSM", "LIF", "CSF", "CSFR", "CNTF")):
        return "Interleukin/Cytokine"
    if "MIF" in token_set:
        return "MIF"
    return "Unclassified"


def _resolve_classification(
    liana_res: pd.DataFrame,
    *,
    classification: str | Mapping[str, str] | None = None,
    classification_reference: str | pd.DataFrame | None = "cellchat",
    classification_fallback: str | None = "family",
) -> tuple[pd.Series, pd.Series, str | None]:
    labels = pd.Series(np.nan, index=liana_res.index, dtype=object)
    sources = pd.Series(np.nan, index=liana_res.index, dtype=object)
    reference_name = None
    pair_labels = (
        liana_res["ligand_complex"].astype(str) + "-" + liana_res["receptor_complex"].astype(str)
    )

    def fill(values: pd.Series, source: str) -> None:
        valid = values.notna() & (values.astype(str).str.strip() != "")
        mask = labels.isna() & valid
        if mask.any():
            labels.loc[mask] = values.loc[mask].astype(str)
            sources.loc[mask] = source

    if isinstance(classification, str):
        if classification not in liana_res.columns:
            raise KeyError(f"`classification='{classification}'` not found in LIANA result columns.")
        fill(liana_res[classification], f"column:{classification}")
    elif isinstance(classification, Mapping):
        fill(pair_labels.map(classification), "mapping")
    elif classification is not None:
        raise TypeError("`classification` must be a column name, a mapping, or None.")

    if classification is None:
        for candidate in ("classification", "pathway_name", "signaling"):
            if candidate in liana_res.columns:
                fill(liana_res[candidate], f"column:{candidate}")
                break

    if classification_reference is not None and labels.isna().any():
        reference_labels, reference_name = _reference_classification_series(
            liana_res,
            classification_reference=classification_reference,
        )
        fill(reference_labels, f"reference:{reference_name}")

    if labels.isna().any():
        if classification_fallback == "family":
            family_labels = pd.Series(
                [
                    _infer_family_label(ligand, receptor)
                    for ligand, receptor in zip(
                        liana_res["ligand_complex"].astype(str),
                        liana_res["receptor_complex"].astype(str),
                    )
                ],
                index=liana_res.index,
                dtype=object,
            )
            fill(family_labels.where(family_labels.ne("Unclassified")), "family")
        elif isinstance(classification_fallback, str) and classification_fallback.strip():
            if classification_fallback == "unclassified":
                pass
            elif classification_fallback.startswith("label:"):
                fallback_label = classification_fallback.split(":", 1)[1].strip()
                if not fallback_label:
                    raise ValueError(
                        "`classification_fallback='label:<name>'` requires a non-empty fallback label."
                    )
                fill(
                    pd.Series(fallback_label, index=liana_res.index, dtype=object),
                    f"fallback:{fallback_label}",
                )
            else:
                raise ValueError(
                    "`classification_fallback` must be one of: 'family', 'unclassified', "
                    "`label:<name>`, or None."
                )
        elif classification_fallback is None:
            pass
        else:
            raise TypeError(
                "`classification_fallback` must be a string strategy or None."
            )

    labels = labels.fillna("Unclassified").astype(str)
    sources = sources.fillna("unclassified").astype(str)
    return labels, sources, reference_name


def _prepare_score_series(series: pd.Series, *, invert: bool, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().all():
        raise ValueError(
            f"Selected LIANA column `{label}` contains no numeric values after coercion. "
            "For LIANA rank_aggregate outputs, `specificity_rank` is usually the safest "
            "consensus column. `magnitude_rank` may be empty on some datasets / methods."
        )

    if invert:
        finite = values[np.isfinite(values)]
        if finite.empty:
            return pd.Series(0.0, index=series.index, dtype=float)

        if float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
            return (1.0 - values).fillna(0.0).clip(lower=0.0, upper=1.0).astype(float)

        min_value = float(finite.min())
        max_value = float(finite.max())
        if np.isclose(max_value, min_value):
            transformed = pd.Series(1.0, index=series.index, dtype=float)
        else:
            transformed = 1.0 - ((values - min_value) / (max_value - min_value))
        transformed = transformed.fillna(0.0).clip(lower=0.0)
        return transformed.astype(float)

    return values.fillna(0.0).astype(float)


@register_function(
    aliases=["LIANA运行", "run_liana", "LIANA+"],
    category="single",
    description="Run LIANA / LIANA+ ligand-receptor inference and store the unified result table in `adata.uns`.",
    prerequisites={"functions": [], "optional_functions": []},
    requires={"obs": ["groupby"]},
    produces={"uns": ["liana_res"]},
    auto_fix="none",
    examples=[
        "ov.single.run_liana(adata, groupby='cell_type')",
        "ov.single.run_liana(adata, groupby='cell_type', method='cellchat', key_added='cpdb_like_res')",
    ],
    related=["single.format_liana_results_for_viz", "pl.ccc_heatmap", "pl.ccc_network_plot", "pl.ccc_stat_plot"],
)
def run_liana(
    adata,
    *,
    groupby: str,
    method: str = "rank_aggregate",
    key_added: str = "liana_res",
    inplace: bool = True,
    **kwargs,
):
    """
    Run LIANA ligand-receptor inference on an AnnData object.

    Parameters
    ----------
    adata
        AnnData object used by LIANA.
    groupby
        Observation column with cell-type labels.
    method
        LIANA method name under ``liana.mt``. Defaults to ``'rank_aggregate'``.
    key_added
        Key used to store LIANA results in ``adata.uns`` when ``inplace=True``.
    inplace
        Whether to write results back to ``adata.uns``.
    **kwargs
        Forwarded to the selected LIANA method.

    Returns
    -------
    pandas.DataFrame or AnnData
        LIANA result table when ``inplace=False``; otherwise returns ``adata``.
    """
    try:
        import liana as li
    except ImportError as exc:  # pragma: no cover - exercised in optional dependency environments
        raise ImportError(
            "ov.single.run_liana requires the optional dependency `liana`. "
            "Install it with `pip install liana` in a Python 3.10-3.13 environment."
        ) from exc

    if not hasattr(li.mt, method):
        available = sorted(name for name in dir(li.mt) if not name.startswith("_"))
        raise ValueError(f"Unknown LIANA method `{method}`. Available methods include: {available}.")

    method_fn = getattr(li.mt, method)
    result = method_fn(
        adata,
        groupby=groupby,
        key_added=key_added,
        inplace=inplace,
        **kwargs,
    )
    return adata if inplace else result


@register_function(
    aliases=["LIANA结果格式化", "LIANA可视化格式化", "format_liana_results", "format_liana_results_for_viz", "LIANA转通信AnnData"],
    category="single",
    description="Convert LIANA aggregate results into OmicVerse communication AnnData compatible with `ov.pl.ccc_*`.",
    prerequisites={"functions": [], "optional_functions": ["run_liana"]},
    requires={"uns": ["liana_res"]},
    produces={
        "layers": ["means", "pvalues"],
        "obs": ["sender", "receiver", "cell_type_pair"],
        "var": ["interacting_pair", "classification", "gene_a", "gene_b"],
    },
    auto_fix="none",
    examples=[
        "comm_adata = ov.single.format_liana_results(adata)",
        "comm_adata = ov.single.format_liana_results(adata, score_key='specificity_rank', pvalue_key='specificity_rank')",
    ],
    related=["single.run_liana", "pl.ccc_heatmap", "pl.ccc_network_plot", "pl.ccc_stat_plot"],
)
def format_liana_results(
    adata=None,
    *,
    liana_res: pd.DataFrame | None = None,
    uns_key: str = "liana_res",
    score_key: str = "specificity_rank",
    pvalue_key: str = "specificity_rank",
    inverse_score: bool = True,
    inverse_pvalue: bool = False,
    classification: str | Mapping[str, str] | None = None,
    classification_reference: str | pd.DataFrame | None = "cellchat",
    classification_fallback: str | None = "family",
):
    """
    Format LIANA results into the communication AnnData expected by ``ov.pl.ccc_*``.

    Parameters
    ----------
    adata
        AnnData containing ``adata.uns[uns_key]``.
    liana_res
        LIANA result table. If omitted, it is read from ``adata.uns[uns_key]``.
    uns_key
        Key in ``adata.uns`` that stores the LIANA result DataFrame.
    score_key
        Column used to populate ``layers['means']``. For current aggregated LIANA
        results, ``'specificity_rank'`` is the safest default because
        ``'magnitude_rank'`` can be empty on some datasets.
    pvalue_key
        Column used to populate ``layers['pvalues']``. For aggregated LIANA
        results, ``'specificity_rank'`` is recommended.
    inverse_score
        Whether smaller values in ``score_key`` should be transformed to larger
        communication strengths. This should stay ``True`` for rank-based scores.
    inverse_pvalue
        Whether smaller values in ``pvalue_key`` should be inverted before being
        written to ``layers['pvalues']``. Keep ``False`` for rank / p-value-like
        columns where smaller values indicate stronger support.
    classification
        Optional column name or ligand-receptor mapping used to populate
        ``var['classification']``.
    classification_reference
        Optional built-in reference (``'cellchat'``, ``'cellchat_human'``,
        ``'cellchat_mouse'``) or a custom DataFrame used to backfill pathway
        annotations for interactions not covered by ``classification``.
    classification_fallback
        Fallback strategy for interactions still not covered after reference
        matching. Use ``'family'`` for coarse signaling-family hints, a custom
        string for a fixed fallback label, or ``None`` to keep
        ``'Unclassified'``.

    Returns
    -------
    anndata.AnnData
        Communication AnnData with ``obs`` cell-type pairs and ``var`` LR pairs.
    """
    result = _validate_liana_res(_get_liana_res(adata=adata, liana_res=liana_res, uns_key=uns_key))

    for key in (score_key, pvalue_key):
        if key not in result.columns:
            raise KeyError(f"`{key}` not found in LIANA result columns.")

    result = result.copy()
    result["score_value"] = _prepare_score_series(result[score_key], invert=inverse_score, label=score_key)
    result["pvalue_value"] = _prepare_score_series(result[pvalue_key], invert=inverse_pvalue, label=pvalue_key)
    result["cell_type_pair"] = result["source"].astype(str) + "|" + result["target"].astype(str)
    sample_column = None
    for candidate in ("sample", "context", "condition", "dataset"):
        if candidate in result.columns:
            sample_column = candidate
            result["cell_type_pair"] = (
                result["cell_type_pair"].astype(str) + "|" + candidate + "=" + result[candidate].astype(str)
            )
            break
    result["interaction_key"] = (
        result["ligand_complex"].astype(str) + " -> " + result["receptor_complex"].astype(str)
    )
    result["interacting_pair"] = (
        result["ligand_complex"].astype(str) + "_" + result["receptor_complex"].astype(str)
    )
    result["pair_lr"] = (
        result["ligand_complex"].astype(str) + "-" + result["receptor_complex"].astype(str)
    )
    (
        result["classification_resolved"],
        result["classification_source"],
        reference_name,
    ) = _resolve_classification(
        result,
        classification=classification,
        classification_reference=classification_reference,
        classification_fallback=classification_fallback,
    )

    obs_index = pd.Index(sorted(result["cell_type_pair"].unique()))
    var_index = pd.Index(sorted(result["interaction_key"].unique()))

    obs = pd.DataFrame(index=obs_index)
    obs["sender"] = obs.index.to_series().str.split("|").str[0].astype(str)
    obs["receiver"] = obs.index.to_series().str.split("|").str[1].astype(str)
    obs["cell_type_pair"] = obs.index.astype(str)
    if sample_column is not None:
        sample_values = obs.index.to_series().str.extract(rf"\|{re.escape(sample_column)}=(.*)$", expand=False)
        obs[sample_column] = sample_values.fillna("").astype(str)

    var = (
        result.loc[
            :,
            [
                "interaction_key",
                "interacting_pair",
                "pair_lr",
                "ligand_complex",
                "receptor_complex",
                "classification_resolved",
                "classification_source",
            ],
        ]
        .drop_duplicates("interaction_key")
        .set_index("interaction_key")
        .reindex(var_index)
    )
    var["interaction_name"] = var["interacting_pair"].astype(str)
    var["interaction_name_2"] = (
        var["ligand_complex"].astype(str) + " - " + var["receptor_complex"].astype(str)
    )
    var["classification"] = var["classification_resolved"].astype(str)
    var["pathway_name"] = var["classification"].astype(str)
    var["signaling"] = var["classification"].astype(str)
    var["gene_a"] = var["ligand_complex"].astype(str)
    var["gene_b"] = var["receptor_complex"].astype(str)
    var["ligand"] = var["ligand_complex"].astype(str)
    var["receptor"] = var["receptor_complex"].astype(str)
    var["annotation_strategy"] = "liana_aggregate"
    var["classification_source"] = var["classification_source"].astype(str)
    var = var.loc[
        :,
        [
            "interacting_pair",
            "pair_lr",
            "interaction_name",
            "interaction_name_2",
            "classification",
            "pathway_name",
            "signaling",
            "gene_a",
            "gene_b",
            "ligand",
            "receptor",
            "annotation_strategy",
            "classification_source",
        ],
    ]

    score_matrix = (
        result.pivot(index="cell_type_pair", columns="interaction_key", values="score_value")
        .reindex(index=obs_index, columns=var_index)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    pvalue_matrix = (
        result.pivot(index="cell_type_pair", columns="interaction_key", values="pvalue_value")
        .reindex(index=obs_index, columns=var_index)
        .fillna(1.0)
        .to_numpy(dtype=float)
    )

    comm_adata = anndata.AnnData(X=score_matrix, obs=obs, var=var)
    comm_adata.layers["means"] = score_matrix.copy()
    comm_adata.layers["pvalues"] = pvalue_matrix.copy()

    # Preserve the original LIANA metrics for advanced downstream use.
    metric_columns = [
        column
        for column in result.columns
        if column
        not in {
            "cell_type_pair",
            "interaction_key",
            "score_value",
            "pvalue_value",
            "classification_resolved",
        }
        and pd.api.types.is_numeric_dtype(result[column])
    ]
    for column in metric_columns:
        matrix = (
            result.pivot(index="cell_type_pair", columns="interaction_key", values=column)
            .reindex(index=obs_index, columns=var_index)
            .fillna(np.nan)
            .to_numpy(dtype=float)
        )
        comm_adata.layers[column] = matrix

    comm_adata.uns["liana_score_key"] = score_key
    comm_adata.uns["liana_pvalue_key"] = pvalue_key
    comm_adata.uns["liana_uns_key"] = uns_key
    comm_adata.uns["liana_sample_key"] = sample_column or "none"
    comm_adata.uns["liana_classification_reference"] = reference_name or "none"
    comm_adata.uns["liana_classification_fallback"] = (
        classification_fallback if classification_fallback is not None else "Unclassified"
    )
    comm_adata.uns["liana_classification_source_counts"] = (
        var["classification_source"].value_counts().to_dict()
    )
    return comm_adata


def format_liana_results_for_viz(*args, **kwargs):
    """Backward-compatible alias of :func:`format_liana_results`."""
    return format_liana_results(*args, **kwargs)

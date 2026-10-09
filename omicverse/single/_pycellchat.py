"""Adapt shaoleishen/pycellchat results without importing its inference backend."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import anndata
import numpy as np
import pandas as pd


def _pycellchat_result(value):
    """Unwrap the public CellChat.cc property or an AnnData result namespace."""
    if isinstance(value, anndata.AnnData):
        return value.uns.get("cellchat")
    if isinstance(value, Mapping):
        return value
    return getattr(value, "cc", None)


def _looks_like_pycellchat_results(value) -> bool:
    result = _pycellchat_result(value)
    return isinstance(result, Mapping) and {"net", "LR", "idents"}.issubset(result)


def _labels(values, *, name: str, unique: bool = True) -> list[str]:
    values = np.asarray(values, dtype=object)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence of labels.")
    if pd.isna(values).any():
        raise ValueError(f"{name} must not contain missing labels.")
    labels = [str(value) for value in values]
    if any(not label.strip() for label in labels):
        raise ValueError(f"{name} must not contain empty labels.")
    if unique and len(set(labels)) != len(labels):
        raise ValueError(f"{name} must contain unique labels after string conversion.")
    return labels


def format_pycellchat_results(data, separator: str = "|") -> anndata.AnnData:
    """Convert a pycellchat result to communication AnnData for ``ov.pl.ccc_*``.

    Supports the single-dataset schema of ``shaoleishen/pycellchat`` 0.1.0.
    This is a result adapter only; it neither runs inference nor imports
    pycellchat (or its Rust backend).

    Parameters
    ----------
    data
        A pycellchat ``CellChat`` object, its ``.cc`` result mapping, or its
        source AnnData with results in ``uns['cellchat']``. The mapping must
        contain ``net['prob']`` and ``net['pval']`` arrays with shape
        (sender groups, receiver groups, L-R interactions), ``idents['names']``
        in tensor-axis order, and a ``LR['LRsig']`` DataFrame in interaction
        order with ``ligand`` and ``receptor`` columns. ``interaction_name``
        identifies interactions when present; otherwise the DataFrame index
        is used. Merged, pathway-only, and summary-only results are unsupported.
    separator
        Separator for observation names. Must be nonempty and absent from
        group labels (choose another separator if needed).

    Returns
    -------
    anndata.AnnData
        All ordered sender-receiver pairs, including self-pairs, as observations
        and L-R interactions as variables. ``X`` and ``layers['means']`` contain
        the original communication probabilities, not expression means.
        ``layers['pvalues']`` contains the original statistical p-values;
        missing p-values are an error. No thresholding, aggregation, or
        significance imputation is performed. LR metadata is preserved, with
        ``classification`` taken from ``pathway_name`` (or ``Unclassified``),
        and ``gene_a`` / ``gene_b`` from ligand / receptor (including complexes).
        The input is not modified and output arrays are independent copies.

    Examples
    --------
    >>> comm = ov.single.format_pycellchat_results(cellchat)
    >>> comm = ov.single.to_comm_adata(cellchat.adata)
    >>> ov.pl.ccc_network_plot(comm, plot_type="circle", show=False)
    """
    result = _pycellchat_result(data)
    if not isinstance(result, Mapping):
        raise TypeError(
            "Expected a pycellchat CellChat object, result mapping, or AnnData."
        )
    for key in ("net", "LR", "idents"):
        if key not in result:
            raise ValueError(f"pycellchat results require a '{key}' mapping.")
        if not isinstance(result[key], Mapping):
            raise TypeError(f"pycellchat '{key}' must be a mapping.")
    options = result.get("options", {})
    if not isinstance(options, Mapping):
        raise TypeError("pycellchat 'options' must be a mapping.")
    if options.get("mode", "single") != "single":
        raise ValueError(
            "Only single-dataset pycellchat results are supported; convert each dataset separately."
        )

    net, lr, idents = result["net"], result["LR"], result["idents"]
    if "prob" not in net or "pval" not in net:
        raise ValueError(
            "pycellchat net requires both 'prob' and 'pval'; p-values cannot be inferred."
        )
    prob, pval = np.asarray(net["prob"]), np.asarray(net["pval"])
    for name, tensor in (("prob", prob), ("pval", pval)):
        if tensor.ndim != 3:
            raise ValueError(
                f"pycellchat net['{name}'] must be a 3D (sender, receiver, interaction) array."
            )
        if tensor.dtype.kind not in "fiu" or not np.isfinite(tensor).all():
            raise ValueError(
                f"pycellchat net['{name}'] must contain finite real numbers."
            )
        if (tensor < 0).any() or (name == "pval" and (tensor > 1).any()):
            raise ValueError(
                f"pycellchat net['{name}'] contains values outside its valid range."
            )
    if prob.shape != pval.shape:
        raise ValueError(
            "pycellchat net['prob'] and net['pval'] must have the same shape."
        )
    if "names" not in idents:
        raise ValueError("pycellchat idents requires 'names' in tensor-axis order.")
    groups = _labels(idents["names"], name="pycellchat idents['names']")
    if prob.shape[:2] != (len(groups), len(groups)):
        raise ValueError(
            "pycellchat tensor sender/receiver dimensions must match idents['names']."
        )
    if not isinstance(separator, str) or not separator:
        raise ValueError("separator must be a nonempty string.")
    if any(separator in group for group in groups):
        raise ValueError(
            "separator occurs in pycellchat group labels; choose another separator."
        )

    lr_sig = lr.get("LRsig")
    if not isinstance(lr_sig, pd.DataFrame):
        raise TypeError(
            "pycellchat LR['LRsig'] must be a DataFrame in tensor interaction order."
        )
    if not lr_sig.columns.is_unique:
        raise ValueError("pycellchat LR['LRsig'] must have unique column names.")
    if len(lr_sig) != prob.shape[2]:
        raise ValueError(
            "pycellchat LR['LRsig'] row count must match the tensor interaction dimension."
        )
    missing = {"ligand", "receptor"}.difference(lr_sig.columns)
    if missing:
        raise ValueError(
            f"pycellchat LR['LRsig'] is missing required columns: {sorted(missing)}."
        )
    interaction_names = _labels(
        lr_sig.get("interaction_name", lr_sig.index),
        name="pycellchat interaction names",
    )
    # Upstream also stores this redundant list. Reject inconsistent order rather
    # than silently assigning L-R metadata to the wrong tensor slices.
    if "interaction_name" in lr:
        stored_names = _labels(
            lr["interaction_name"], name="pycellchat LR['interaction_name']"
        )
        if stored_names and stored_names != interaction_names:
            raise ValueError(
                "pycellchat LR['interaction_name'] must match LRsig interaction order."
            )
    ligands = _labels(lr_sig["ligand"], name="pycellchat ligands", unique=False)
    receptors = _labels(lr_sig["receptor"], name="pycellchat receptors", unique=False)

    var = lr_sig.copy(deep=True)
    var.index = pd.Index(interaction_names)
    var["interaction_name"] = interaction_names
    var["gene_a"], var["gene_b"] = ligands, receptors
    var["interacting_pair"] = [
        f"{ligand}_{receptor}" for ligand, receptor in zip(ligands, receptors)
    ]
    var["pair_lr"] = [
        f"{ligand}-{receptor}" for ligand, receptor in zip(ligands, receptors)
    ]
    if "pathway_name" in var:
        pathways = var["pathway_name"].astype(object)
        valid = pathways.notna() & pathways.astype(str).str.strip().ne("")
        var["classification"] = pathways.where(valid, "Unclassified").astype(str)
    else:
        var["classification"] = "Unclassified"

    pairs = [(sender, receiver) for sender in groups for receiver in groups]
    pair_names = [separator.join(pair) for pair in pairs]
    if len(set(pair_names)) != len(pair_names):
        raise ValueError(
            "pycellchat sender-receiver names collide; choose another separator."
        )
    obs = pd.DataFrame(
        pairs, columns=["sender", "receiver"], index=pd.Index(pair_names)
    )
    obs["cell_type_pair"] = pair_names
    shape = (len(pairs), len(interaction_names))
    # C order keeps receiver varying fastest within each sender; no transpose.
    means = prob.reshape(shape).copy()
    comm = anndata.AnnData(X=means.copy(), obs=obs, var=var)
    comm.layers["means"] = means
    comm.layers["pvalues"] = pval.reshape(shape).copy()
    comm.uns["comm_source"] = "pycellchat"
    comm.uns["support_kind"] = "pvalue"
    comm.uns["pvalues_are_statistical"] = True
    comm.uns["pycellchat_separator"] = separator
    comm.uns["pycellchat_group_names"] = groups
    comm.uns["pycellchat_options"] = deepcopy(dict(options))
    if "column" in idents:
        comm.uns["pycellchat_groupby"] = str(idents["column"])
    return comm

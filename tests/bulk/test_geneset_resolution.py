"""Tests for `_resolve_genesets` — the gene-set argument normaliser that
lets the enrichment entry points (`geneset_enrichment`, `pyGSEA`,
`pyGSE`, `geneset_enrichment_GSEA`) accept a prepared dict, a local
`.gmt`/`.txt` path, or an Enrichr library name.

Network-free: only the dict-passthrough, error, and local-path branches
are exercised. Resolving an Enrichr library name downloads from Enrichr
and is covered by `geneset_prepare`'s own tests.
"""

import inspect

import pandas as pd
import pytest

from omicverse.bulk._Enrichment import (
    _resolve_genesets,
    geneset_enrichment_GSEA,
    pyGSE,
    pyGSEA,
)


@pytest.fixture
def geneset_txt(tmp_path):
    """A tiny Enrichr-format gene-set file: `name<tab>gene<tab>gene...`."""
    p = tmp_path / "sets.txt"
    p.write_text("PATHWAY_A\tTP53\tBRCA1\tATM\n"
                 "PATHWAY_B\tEGFR\tMYC\tPTEN\n")
    return str(p)


def test_resolve_dict_passthrough():
    d = {"PATHWAY_A": ["TP53", "BRCA1"]}
    assert _resolve_genesets(d) is d


def test_resolve_bad_type_raises():
    with pytest.raises(TypeError, match="dict.*or a str"):
        _resolve_genesets(123)


def test_resolve_local_path(geneset_txt):
    res = _resolve_genesets(geneset_txt, organism="Human")
    assert res == {"PATHWAY_A": ["TP53", "BRCA1", "ATM"],
                   "PATHWAY_B": ["EGFR", "MYC", "PTEN"]}


def test_pyGSEA_accepts_path_and_dict(geneset_txt):
    rnk = pd.DataFrame({"gene_name": ["TP53", "EGFR", "MYC"],
                        "rnk": [3.0, 1.0, -2.0]})
    # path -> resolved dict
    g = pyGSEA(rnk, geneset_txt)
    assert g.pathways_dict["PATHWAY_A"] == ["TP53", "BRCA1", "ATM"]
    # dict -> unchanged
    d = {"P": ["TP53"]}
    assert pyGSEA(rnk, d).pathways_dict is d


def test_pyGSE_accepts_path(geneset_txt):
    e = pyGSE(["TP53", "EGFR"], geneset_txt)
    assert e.pathways_dict["PATHWAY_B"] == ["EGFR", "MYC", "PTEN"]


def test_organism_param_present():
    # organism was added to the two entry points that lacked it.
    assert "organism" in inspect.signature(pyGSEA.__init__).parameters
    assert "organism" in inspect.signature(geneset_enrichment_GSEA).parameters

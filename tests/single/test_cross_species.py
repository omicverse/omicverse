"""Ortholog-based cross-species integration (issue #856).

BioMart connectivity itself is exercised live during development; here we mock
the network layer and test OUR logic: species resolution, ortholog-table
filtering/caching, gene renaming/collapsing, and the end-to-end integrate
(alignment → shared-gene concat → batch backend) on small synthetic AnnData.
"""
import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")

from omicverse.single import _cross_species as cs


# ---- a fake BioMart: human symbols -> mouse orthologs -----------------------
_ONE2ONE = [(f"GENE{c}", f"Gen{c.lower()}") for c in "ABCDEFGHIJ"]
_FAKE_TSV = "\n".join(
    [f"{s}\t{t}\tortholog_one2one" for s, t in _ONE2ONE] + [
        "DUPSRC1\tSharedmm\tortholog_one2one",
        "DUPSRC2\tSharedmm\tortholog_one2one",   # two src -> same target
        "MANYA\tManya1\tortholog_one2many",       # dropped when one2one
        "\tOrphan\tortholog_one2one",             # empty source -> dropped
        "NOSYM\t\tortholog_one2one",              # empty target -> dropped
    ])


@pytest.fixture
def fake_biomart(monkeypatch):
    def _fake_query(dataset, attributes, **kw):
        df = pd.read_csv(pd.io.common.StringIO(_FAKE_TSV), sep="\t",
                         header=None, names=list(attributes), dtype=str)
        return df
    monkeypatch.setattr(cs, "_biomart_query", _fake_query)


def _adata(genes, n=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(2.0, size=(n, len(genes))).astype(np.float32)
    return anndata.AnnData(X, var=pd.DataFrame(index=list(genes)))


def test_resolve_dataset_variants():
    assert cs._resolve_dataset("human") == "hsapiens_gene_ensembl"
    assert cs._resolve_dataset("Mouse") == "mmusculus_gene_ensembl"
    assert cs._resolve_dataset("hsapiens") == "hsapiens_gene_ensembl"
    assert cs._resolve_dataset("sscrofa_gene_ensembl") == "sscrofa_gene_ensembl"
    with pytest.raises(ValueError):
        cs._resolve_dataset("nonexistent-species")


def test_get_orthologs_one2one_filter_and_cache(fake_biomart, tmp_path):
    df = cs.get_orthologs("human", "mouse", cache_dir=str(tmp_path))
    assert list(df.columns) == ["source_symbol", "target_symbol", "orthology_type"]
    # one2many dropped, empties dropped
    assert (df["orthology_type"] == "ortholog_one2one").all()
    assert "MANYA" not in set(df["source_symbol"])
    assert "" not in set(df["source_symbol"]) and "" not in set(df["target_symbol"])
    # cache file written and reused (fast path, no query)
    cache_files = list(tmp_path.glob("*.tsv"))
    assert len(cache_files) == 1
    df2 = cs.get_orthologs("human", "mouse", cache_dir=str(tmp_path))
    pd.testing.assert_frame_equal(df, df2)


def test_get_orthologs_all_keeps_one2many(fake_biomart, tmp_path):
    df = cs.get_orthologs("human", "mouse", orthology_type="all",
                          cache_dir=str(tmp_path))
    assert "MANYA" in set(df["source_symbol"])


def test_map_var_to_orthologs_renames_and_collapses(fake_biomart, tmp_path):
    ad = _adata(["GENEA", "GENEB", "DUPSRC1", "DUPSRC2", "UNMAPPED"])
    out = cs.map_var_to_orthologs(ad, "human", "mouse", cache_dir=str(tmp_path))
    # UNMAPPED dropped; DUPSRC1/2 collapse into a single 'Sharedmm'
    assert set(out.var_names) == {"Gena", "Genb", "Sharedmm"}
    assert not out.var_names.duplicated().any()
    # collapse == sum: Sharedmm equals DUPSRC1 + DUPSRC2
    col = out[:, "Sharedmm"].X
    col = np.asarray(col.todense()).ravel() if hasattr(col, "todense") else np.asarray(col).ravel()
    expected = ad[:, "DUPSRC1"].X.ravel() + ad[:, "DUPSRC2"].X.ravel()
    assert np.allclose(col, expected)
    # provenance kept
    assert "source_symbol" in out.var.columns


def test_map_var_raises_when_no_overlap(fake_biomart, tmp_path):
    ad = _adata(["ZZZ1", "ZZZ2"])
    with pytest.raises(ValueError, match="No genes"):
        cs.map_var_to_orthologs(ad, "human", "mouse", cache_dir=str(tmp_path))


def test_cross_species_integrate_end_to_end(fake_biomart, tmp_path):
    # human ref keeps its own symbols; mouse gets mapped human->mouse? No —
    # ref is human, so the mouse dataset must be mapped mouse->human. Provide a
    # reverse fake by symmetry: reuse the same table but query mouse->human.
    human_syms = [s for s, _ in _ONE2ONE]          # GENEA..GENEJ
    mouse_syms = [t for _, t in _ONE2ONE]           # Gena..Genj
    human = _adata(human_syms, n=30, seed=1)
    # mouse dataset already on mouse symbols; integrate with ref='mouse' so the
    # human set is mapped human->mouse (our fake table direction).
    mouse = _adata(mouse_syms, n=25, seed=2)
    out = cs.cross_species_integrate(
        [human, mouse], ["human", "mouse"], ref_species="mouse",
        method="harmony", n_top_genes=8, n_pcs=5,
        cache_dir=str(tmp_path))
    # concatenated across species on the shared mouse-symbol axis
    assert out.n_obs == 55
    assert set(out.obs["species"].unique()) == {"human", "mouse"}
    assert out.uns["cross_species"]["ref_species"] == "mouse"
    assert out.uns["cross_species"]["n_common_orthologs"] == len(_ONE2ONE)
    # harmony wrote its embedding
    assert "X_pca_harmony" in out.obsm

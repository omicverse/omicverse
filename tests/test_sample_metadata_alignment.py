"""Tests for `omicverse.utils.preflight_alignment` and friends.

The cases exercise the three motivating bug classes:

1. Pandas auto-rename of duplicate column labels (most common silent
   failure mode — counts CSVs with two columns sharing a sample ID).
2. Samples present in the matrix but absent from metadata, and vice
   versa.
3. Mixed in-memory DataFrame / file-path / AnnData inputs.

No network calls; everything uses tmp_path fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_pair(tmp_path: Path):
    """Counts (genes × samples) + meta (samples × phenotype), all aligned."""
    samples = ["s1", "s2", "s3", "s4"]
    counts = pd.DataFrame(
        {s: [10 * i + j for j in range(3)] for i, s in enumerate(samples)},
        index=["GENE_A", "GENE_B", "GENE_C"],
    )
    counts.index.name = "gene"
    counts_path = tmp_path / "counts.csv"
    counts.to_csv(counts_path)

    meta = pd.DataFrame(
        {"sample_id": samples, "condition": ["A", "A", "B", "B"], "batch": [1, 2, 1, 2]}
    )
    meta_path = tmp_path / "meta.csv"
    meta.to_csv(meta_path, index=False)
    return counts_path, meta_path


@pytest.fixture
def dup_pair(tmp_path: Path):
    """Counts CSV where two columns share the same sample ID — the
    pandas-auto-rename case. Naive `pd.read_csv` plus `.duplicated()`
    would report 0 here; the helper must catch it via raw-header read."""
    counts_path = tmp_path / "counts_dup.csv"
    counts_path.write_text(
        "gene,s1,s2,s2,s3\n"
        "GENE_A,1,2,3,4\n"
        "GENE_B,5,6,7,8\n"
    )

    meta = pd.DataFrame({"sample_id": ["s1", "s2", "s3"], "condition": ["A", "B", "C"]})
    meta_path = tmp_path / "meta_dup.csv"
    meta.to_csv(meta_path, index=False)
    return counts_path, meta_path


@pytest.fixture
def missing_meta_pair(tmp_path: Path):
    """One matrix sample is absent from metadata."""
    counts_path = tmp_path / "counts_miss.csv"
    counts_path.write_text(
        "gene,s1,s2,s3,s4\n"
        "GENE_A,1,2,3,4\n"
    )
    meta = pd.DataFrame({"sample_id": ["s1", "s2", "s3"], "condition": ["A", "B", "C"]})
    meta_path = tmp_path / "meta_miss.csv"
    meta.to_csv(meta_path, index=False)
    return counts_path, meta_path


# ---------------------------------------------------------------------------
# preflight_alignment
# ---------------------------------------------------------------------------


def test_preflight_clean(clean_pair):
    from omicverse.utils import preflight_alignment

    counts_path, meta_path = clean_pair
    r = preflight_alignment(counts_path, meta_path)
    assert r.is_clean
    assert not r.needs_alignment
    assert r.n_dup_in_matrix == 0
    assert r.n_dup_in_meta == 0
    assert r.n_missing_from_meta == 0
    assert r.n_missing_from_matrix == 0
    assert r.sample_col_used == "sample_id"
    assert r.matrix_sample_axis == "columns"


def test_preflight_catches_pandas_renamed_duplicates(dup_pair):
    """The canonical bug: pandas reads the CSV with two `s2` columns as
    `s2` + `s2.1`. A naive `df.columns.duplicated().sum()` returns 0.
    The helper reads the raw header and reports the real count."""
    from omicverse.utils import preflight_alignment

    counts_path, meta_path = dup_pair
    r = preflight_alignment(counts_path, meta_path)
    assert r.needs_alignment, r
    # Two `s2` columns → 1 ID appears more than once → counter > 1 once.
    assert r.n_dup_in_matrix == 1, r

    # Sanity: naive pandas check would NOT have caught this.
    naive = pd.read_csv(counts_path).columns.duplicated().sum()
    assert naive == 0, "this test asserts pandas's silent rename behavior"


def test_preflight_catches_missing_from_meta(missing_meta_pair):
    from omicverse.utils import preflight_alignment

    counts_path, meta_path = missing_meta_pair
    r = preflight_alignment(counts_path, meta_path)
    assert r.needs_alignment
    assert r.n_dup_in_matrix == 0
    assert r.n_missing_from_meta == 1  # `s4` in matrix, not in meta
    assert r.n_missing_from_matrix == 0


def test_preflight_accepts_dataframe_inputs(clean_pair):
    """In-memory DataFrames go through the same code path."""
    from omicverse.utils import preflight_alignment

    counts_path, meta_path = clean_pair
    counts = pd.read_csv(counts_path, index_col=0)
    meta = pd.read_csv(meta_path)
    r = preflight_alignment(counts, meta)
    assert r.is_clean


def test_preflight_explicit_sample_col(missing_meta_pair):
    """User-supplied `sample_col` overrides auto-detect."""
    from omicverse.utils import preflight_alignment

    counts_path, meta_path = missing_meta_pair
    r = preflight_alignment(counts_path, meta_path, sample_col="sample_id")
    assert r.sample_col_used == "sample_id"


def test_preflight_no_overlap_raises(tmp_path: Path):
    """When no metadata column overlaps the matrix sample axis, fail
    clearly rather than producing a garbage diff."""
    from omicverse.utils import preflight_alignment

    counts = tmp_path / "counts.csv"
    counts.write_text("gene,s1,s2\nA,1,2\n")
    meta = tmp_path / "meta.csv"
    meta.write_text("subject,age\nfoo,10\nbar,20\n")
    with pytest.raises(ValueError, match="overlaps"):
        preflight_alignment(counts, meta)


def test_preflight_str_summary_includes_status(dup_pair):
    from omicverse.utils import preflight_alignment

    counts_path, meta_path = dup_pair
    r = preflight_alignment(counts_path, meta_path)
    s = str(r)
    assert "needs alignment" in s
    assert "dup_matrix=1" in s


def test_preflight_summary_dict_is_json_serializable(dup_pair):
    from omicverse.utils import preflight_alignment

    counts_path, meta_path = dup_pair
    r = preflight_alignment(counts_path, meta_path)
    d = r.summary_dict()
    assert "n_dup_in_matrix" in d
    assert "matrix_sample_ids" not in d  # the heavy lists are kept off
    json.dumps(d)  # raises on non-serializable values


# ---------------------------------------------------------------------------
# align_to_common
# ---------------------------------------------------------------------------


def test_align_drops_pandas_renamed_duplicates(dup_pair):
    """After alignment, the matrix has no duplicates and matches meta."""
    from omicverse.utils import align_to_common, preflight_alignment

    counts_path, meta_path = dup_pair
    r = preflight_alignment(counts_path, meta_path)
    mat, meta = align_to_common(counts_path, meta_path, r)
    # Both `s2` columns dropped (default `keep=False`); only s1 and s3 remain.
    assert list(mat.columns) == ["s1", "s3"]
    assert list(meta.index) == ["s1", "s3"]


def test_align_drops_missing_from_meta(missing_meta_pair):
    from omicverse.utils import align_to_common, preflight_alignment

    counts_path, meta_path = missing_meta_pair
    r = preflight_alignment(counts_path, meta_path)
    mat, meta = align_to_common(counts_path, meta_path, r)
    # `s4` (matrix-only) dropped.
    assert list(mat.columns) == ["s1", "s2", "s3"]
    assert list(meta.index) == ["s1", "s2", "s3"]


def test_align_samples_one_shot(missing_meta_pair):
    from omicverse.utils import align_samples

    counts_path, meta_path = missing_meta_pair
    mat, meta, r = align_samples(counts_path, meta_path)
    assert list(mat.columns) == ["s1", "s2", "s3"]
    assert list(meta.index) == ["s1", "s2", "s3"]
    assert r.n_missing_from_meta == 1


def test_align_clean_pair_is_passthrough(clean_pair):
    from omicverse.utils import align_samples

    counts_path, meta_path = clean_pair
    mat, meta, r = align_samples(counts_path, meta_path)
    assert r.is_clean
    assert sorted(mat.columns) == ["s1", "s2", "s3", "s4"]


# ---------------------------------------------------------------------------
# Misc edge cases
# ---------------------------------------------------------------------------


def test_preflight_handles_int_sample_ids(tmp_path: Path):
    """CSV readers can infer one side as int64; the helper must coerce
    both to str before comparing."""
    from omicverse.utils import preflight_alignment

    counts = tmp_path / "counts.csv"
    counts.write_text("gene,1,2,3\nA,1,2,3\n")
    meta = tmp_path / "meta.csv"
    meta.write_text("sample_id,condition\n1,A\n2,B\n3,C\n")  # int-looking
    r = preflight_alignment(counts, meta)
    assert r.is_clean


def test_preflight_tsv(tmp_path: Path):
    """Tab-separated text works without explicit `sep=`."""
    from omicverse.utils import preflight_alignment

    counts = tmp_path / "counts.tsv"
    counts.write_text("gene\ts1\ts2\nA\t1\t2\n")
    meta = tmp_path / "meta.tsv"
    meta.write_text("sample_id\tcondition\ns1\tA\ns2\tB\n")
    r = preflight_alignment(counts, meta)
    assert r.is_clean


def test_align_to_common_int_sample_ids(tmp_path: Path):
    """`align_to_common` resolves the common set when a CSV reader infers
    the metadata sample column as int64. Regression: the string-typed
    `common` set was indexed against an int64 metadata index, raising a
    spurious KeyError."""
    from omicverse.utils import align_samples

    counts = tmp_path / "counts.csv"
    counts.write_text("gene,1,2,3\nGENE_A,5,6,7\nGENE_B,8,9,10\n")
    meta = tmp_path / "meta.csv"
    meta.write_text("sample_id,condition\n1,A\n2,B\n3,C\n")  # int-looking
    mat, aligned_meta, _ = align_samples(counts, meta)
    assert sorted(map(str, mat.columns)) == ["1", "2", "3"]
    assert sorted(aligned_meta.index) == ["1", "2", "3"]


def test_align_collapses_exact_duplicate_meta_rows(tmp_path: Path):
    """A metadata sheet that lists a sample twice with identical values is
    collapsed to one row — the sample is kept, not dropped as an ambiguous
    ID collision (keep=False applies only to genuinely conflicting rows)."""
    from omicverse.utils import align_samples

    counts = tmp_path / "counts.csv"
    counts.write_text("gene,s1,s2,s3\nGENE_A,1,2,3\n")
    meta = tmp_path / "meta.csv"
    meta.write_text("sample_id,condition\ns1,A\ns2,B\ns2,B\ns3,C\n")  # s2 twice
    mat, aligned_meta, r = align_samples(counts, meta)
    assert r.n_dup_in_meta == 1
    assert sorted(aligned_meta.index) == ["s1", "s2", "s3"]
    assert sorted(map(str, mat.columns)) == ["s1", "s2", "s3"]


# ---------------------------------------------------------------------------
# HDF5 matrix inputs — a `.h5` that is NOT an AnnData h5ad
# ---------------------------------------------------------------------------


def _write_rhdf5_matrix(path, genes, samples, dense, *, sparse: bool) -> None:
    """Write an R/rhdf5-style HDF5 count matrix: a `.<name>_dimnames`
    group plus a dense 2-D dataset and, optionally, a CSC sparse triple.
    rhdf5 stores an R (genes × samples) matrix transposed on disk."""
    import h5py
    import numpy as np
    import scipy.sparse as sp

    with h5py.File(path, "w") as f:
        dn = f.create_group(".gene_counts_dimnames")
        dn.create_dataset("1", data=np.array(genes, dtype="S25"))
        dn.create_dataset("2", data=np.array(samples, dtype="S9"))
        f.create_dataset("gene_counts", data=dense.T)
        if sparse:
            csc = sp.csc_matrix(dense)
            sm = f.create_group("sparse_matrix")
            sm.create_dataset(
                "dimensions", data=np.array([len(genes), len(samples)], dtype="int32")
            )
            sm.create_dataset("i_indices", data=csc.indices.astype("int32"))
            sm.create_dataset("j_ptr", data=csc.indptr.astype("int32"))
            sm.create_dataset("values", data=csc.data.astype("float64"))


@pytest.fixture
def rhdf5_matrix_pair(tmp_path: Path):
    """An R/rhdf5-style HDF5 count matrix and its metadata sheet."""
    import numpy as np

    genes = [f"GENE{i}" for i in range(6)]
    samples = ["s1", "s2", "s3", "s4"]
    dense = np.arange(len(genes) * len(samples), dtype="float64").reshape(
        len(genes), len(samples)
    )
    meta = pd.DataFrame({"sample_id": samples, "condition": ["A", "A", "B", "B"]})
    meta_path = tmp_path / "meta.csv"
    meta.to_csv(meta_path, index=False)
    return tmp_path, genes, samples, dense, meta_path


@pytest.mark.parametrize("sparse", [True, False])
def test_align_samples_reads_rhdf5_h5_matrix(rhdf5_matrix_pair, sparse):
    """A `.h5` that is a plain rhdf5 matrix (not an h5ad) is detected,
    read into an AnnData, and oriented with the sample axis on `.obs`."""
    import numpy as np

    from omicverse.utils import align_samples

    tmp_path, genes, samples, dense, meta_path = rhdf5_matrix_pair
    h5_path = tmp_path / "counts.h5"
    _write_rhdf5_matrix(h5_path, genes, samples, dense, sparse=sparse)

    mat, aligned_meta, r = align_samples(h5_path, meta_path, sample_col="sample_id")
    assert r.matrix_kind == "h5matrix"
    assert list(mat.obs_names) == samples       # samples on obs
    assert list(mat.var_names) == genes         # genes on var
    assert list(aligned_meta.index) == samples
    X = mat.X.toarray() if hasattr(mat.X, "toarray") else np.asarray(mat.X)
    np.testing.assert_array_equal(X, dense.T)   # values survive the round-trip


def test_preflight_unparseable_h5_raises_clear_error(tmp_path: Path):
    """A `.h5` that is neither an h5ad nor a recognisable rhdf5 matrix
    raises an actionable ValueError, not an opaque AnnData TypeError."""
    import h5py
    import numpy as np

    from omicverse.utils import preflight_alignment

    bad = tmp_path / "mystery.h5"
    with h5py.File(bad, "w") as f:
        f.create_dataset("payload", data=np.zeros((3, 3)))
    meta = tmp_path / "meta.csv"
    meta.write_text("sample_id,condition\ns1,A\ns2,B\n")
    with pytest.raises(ValueError, match="dimnames"):
        preflight_alignment(bad, meta, sample_col="sample_id")

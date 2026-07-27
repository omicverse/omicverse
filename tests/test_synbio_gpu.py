"""GPU smoke test for ov.synbio layer B — protein/enzyme design.

Intended for a CUDA node: it downloads and runs ESM-2 650M, ESM-1v, ESMFold and
ProteinMPNN. Every test gates itself on an actual CUDA device rather than on a
``-m`` filter, because CI runs a bare ``pytest`` and would otherwise try to pull
several GB of checkpoints onto a CPU runner.

Was ``tests/synbio_gpu_smoke.py``, a top-level script pytest never collected
(``python_files = ["test_*.py"]``).

Run it where it belongs with::

    pytest tests/test_synbio_gpu.py -v
"""
import pytest

torch = pytest.importorskip("torch", reason="ov.synbio layer B needs torch")
pytest.importorskip("esm", reason="needs omicverse[synbio] (fair-esm)")

import omicverse as ov

sb = ov.synbio

pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_gpu,
    pytest.mark.requires_network,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="ov.synbio layer B needs CUDA; these models are far too slow on CPU",
    ),
]

SEQ = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
       "QTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFG")
SHORT = SEQ[:60]
DEV = "cuda"


def test_protein_embed_shape():
    embeddings = sb.protein_embed([SEQ, SHORT], model="esm2_t33_650M", device=DEV)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 1280, "esm2_t33_650M emits 1280-d embeddings"


def test_variant_effect_scans_every_substitution():
    scores = sb.variant_effect(SHORT, model="esm1v", device=DEV)
    assert len(scores) == len(SHORT) * 19, "expected a full saturation scan (19 subs/site)"
    assert scores.iloc[0]["score"] >= scores.iloc[-1]["score"], "must be ranked"


def test_predict_structure_then_inverse_design(tmp_path):
    pdb = tmp_path / "esmfold.pdb"
    pred = sb.predict_structure(SHORT, device=DEV, out_path=str(pdb))
    assert pdb.is_file() and pdb.stat().st_size > 0
    assert 0 <= pred.mean_plddt <= 100

    designs = sb.inverse_design(str(pdb), num_sequences=4, device=DEV)
    assert len(designs) >= 2, "ProteinMPNN returns the native plus the designs"
    assert all(len(d.sequence) == len(SHORT) for d in designs)


def test_stability_ddg_scores_each_requested_mutation(tmp_path):
    pdb = tmp_path / "esmfold.pdb"
    sb.predict_structure(SHORT, device=DEV, out_path=str(pdb))
    mutations = ["A2V", "K3E", "T4A"]
    ddg = sb.stability_ddg(str(pdb), mutations=mutations, device=DEV)
    assert list(ddg["mutation"]) == mutations


def test_enzyme_kcat_is_sequence_sensitive():
    """A point mutation must move the predicted turnover number."""
    wild = sb.enzyme_kcat(SHORT, "C(C(=O)O)N", device=DEV)
    mutant = sb.enzyme_kcat(SHORT[:30] + "A" + SHORT[31:], "C(C(=O)O)N", device=DEV)
    assert wild.kcat > 0 and mutant.kcat > 0
    assert wild.kcat != mutant.kcat, (
        "identical kcat for different sequences means the model is not reading "
        "the sequence at all"
    )


def test_enzyme_function_assigns_an_ec_number():
    prediction = sb.enzyme_function(SHORT, device=DEV)
    assert prediction.top_ec, "expected a top EC assignment"
    assert prediction.predictions

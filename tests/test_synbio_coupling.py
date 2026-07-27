"""ov.synbio layer A, and the A↔B hinge that is the module's signature.

    enzyme_kcat(enzyme_seq, substrate)   ->  kcat            (protein layer B)
              |
    ec_model(GEM, {reaction: kcat})      ->  EC-constrained model (layer A)
              |
    fba(EC model)                        ->  growth / yield recomputed

"Edit one enzyme -> the metabolic network re-solves its yield."
Runs on a real small model (e_coli_core) + a central-metabolism enzyme (PFK).

Was ``tests/synbio_coupling_demo.py``, a top-level script pytest never
collected (``python_files = ["test_*.py"]``) — which is how an unresolvable
``[synbio]`` extra survived unnoticed from #887 until #912. Everything here
gates itself at runtime instead of relying on ``-m`` filters, because CI runs
a bare ``pytest``.
"""
import pytest

# The whole module is layer A: skipped entirely unless omicverse[synbio] is in.
cobra = pytest.importorskip("cobra", reason="needs omicverse[synbio] (cobra)")

import omicverse as ov

sb = ov.synbio

# A real E. coli phosphofructokinase (PfkA) fragment as the enzyme under
# design; the substrate is fructose-6-phosphate (its physiological substrate).
PFK_SEQ = ("MIKKIGVLTSGGDAPGMNAAIRGVVRSALTEGLEVMGIYDGYLGLYEDRMVQLDRYSVSDMINRGGTFLGSARFPEFRD"
           "ENIRAVAIENLKKRGIDALVVIGGDGSYMGAMRLTEMGFPCIGLPGTIDNDIKGTDYTIGFFTALSTVVEAIDRLRDT")
SUBSTRATE_F6P = "OCC1OC(O)(COP(=O)(O)O)C(O)C1O"


@pytest.fixture(scope="module")
def core_model():
    """The *E. coli* core model (95 reactions).

    ``cobra.io.load_model`` fetches it from BiGG (~35 kB) and caches it, so
    this needs the network on a cold runner — verified in a clean Linux
    container. Skips rather than fails when it is unreachable.
    """
    try:
        return sb.load_gem("e_coli_core")
    except Exception as exc:
        pytest.skip(f"e_coli_core unavailable: {exc}")


@pytest.mark.requires_network
def test_fba_solves_wild_type_growth(core_model):
    solution = sb.fba(core_model)
    assert solution.objective_value == pytest.approx(0.8739, abs=1e-3), (
        "e_coli_core's textbook max growth rate is ~0.874 /h"
    )


@pytest.mark.requires_network
def test_ec_model_constrains_growth_and_respects_the_protein_budget(core_model):
    """The layer-A half of the hinge: a kcat becomes an enzyme-capacity bound.

    Uses a fixed kcat so this runs without the protein layer — the point is
    that under one shared protein budget a slower enzyme must not out-grow a
    faster one, and that kcat actually moves the solution at all.
    """
    kcat_wt = 100.0
    ecm = sb.ec_model(core_model, {"PFK": kcat_wt})
    pool = ecm.synbio_ec["total_protein"]
    growth_wt = sb.fba(ecm).objective_value
    assert growth_wt > 0, "an enzyme-constrained model must still be feasible"

    slower = sb.fba(sb.ec_model(core_model, {"PFK": kcat_wt * 0.3},
                                total_protein=pool)).objective_value
    faster = sb.fba(sb.ec_model(core_model, {"PFK": kcat_wt * 10.0},
                                total_protein=pool)).objective_value

    assert slower <= growth_wt + 1e-9, "a slower enzyme cannot raise the yield"
    assert faster >= growth_wt - 1e-9, "a faster enzyme cannot lower the yield"
    assert slower < faster, (
        "kcat must actually move the flux solution — if these come out equal "
        "the enzyme constraint is not binding and the hinge is decorative"
    )


@pytest.mark.slow
@pytest.mark.requires_network
def test_full_kcat_to_flux_hinge(core_model):
    """The real hinge, end to end. Needs DLKcat (cloned + weights on first use)."""
    try:
        kcat = sb.enzyme_kcat(PFK_SEQ, SUBSTRATE_F6P, verbose=False)
    except Exception as exc:
        pytest.skip(f"enzyme_kcat backend unavailable: {exc}")

    assert kcat.kcat > 0, "a turnover number must be positive"
    ecm = sb.ec_model(core_model, {"PFK": kcat.kcat})
    assert sb.fba(ecm).objective_value > 0

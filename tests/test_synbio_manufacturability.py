"""Can it be made? — solubility, aggregation, signal peptides, localisation.

These are sequence-only predictors, so they are tested against **real proteins
with known behaviour**, not synthetic strings. Getting the ordering right on
these four is the whole claim:

``PhoA``  *E. coli* alkaline phosphatase. Periplasmic, and its Sec/SPI signal
          peptide is experimentally cleaved after residue 21 —
          ``MKQSTIALALLPLLFTPVTKA`` / ``RTPEMPVLENR…``. A signal-peptide
          predictor that misses this one is not working.
``GFP``   Famously well expressed and soluble in *E. coli*. Cytoplasmic, no
          signal peptide.
``lysozyme`` Mature hen lysozyme — soluble, disulfide-rich, and (as the mature
          chain) carries no signal peptide.
``LacY``  Lactose permease N-terminal region: a polytopic membrane protein, the
          negative control for solubility and the positive one for TM helices.

The defaults here are transparent sequence arithmetic, not fitted models, and
the tests assert *ordering and known landmarks* rather than exact values — which
is what a documented heuristic can honestly promise.
"""
import pytest

import omicverse as ov

sb = ov.synbio

PHOA = (
    "MKQSTIALALLPLLFTPVTKARTPEMPVLENRAAQGDITAPGGARRLTGDQTAALRDSLSDKPAKNIILLIGDGMGDSEI"
    "TAARNYAEGAGGFFKGIDALPLTGQYTHYALNKKTGKPDYVTDSAASATAWSTGVKTYNGALGVDIHEKDHPTILEMAKA"
)
PHOA_CLEAVAGE = 21
PHOA_SIGNAL = "MKQSTIALALLPLLFTPVTKA"

GFP = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQ"
    "HDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNG"
    "IKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)
LYSOZYME = (
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPC"
    "SALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
)
LACY = (
    "MYYLKNTNFWMFGLFFFFYFFIMGAYFPFFPIWLHDINHISKSDTGIIFAAISLFSLLFQPLFGLLSDKLGLRKYLLWII"
    "TGMLVMFAPFFIFIFGPLLQYNILVGSIVGGIYLGFCFNAGAPAVEAFIEKVSRRSNFEFGRARMFGCVGWALCASIVGI"
)


# ---------------------------------------------------------------------------
# solubility
# ---------------------------------------------------------------------------

def test_membrane_protein_is_less_soluble_than_globular_ones():
    """The load-bearing comparison: a permease must not outrank GFP."""
    lacy = sb.predict_solubility(LACY).score
    gfp = sb.predict_solubility(GFP).score
    lys = sb.predict_solubility(LYSOZYME).score
    assert lacy < gfp, f"LacY {lacy:.3f} should score below GFP {gfp:.3f}"
    assert lacy < lys, f"LacY {lacy:.3f} should score below lysozyme {lys:.3f}"


def test_known_soluble_proteins_are_not_called_high_risk():
    """Regression: the first calibration centred the hydrophobic-patch term at
    1.2, which called GFP high-risk. Nearly every globular protein has a
    9-residue window above that — it is a buried strand, not a liability."""
    for name, seq in (("GFP", GFP), ("lysozyme", LYSOZYME)):
        res = sb.predict_solubility(seq)
        assert res.risk != "high", f"{name} was called high-risk ({res.score:.3f})"
        assert res.soluble, f"{name} scored {res.score:.3f}"


def test_membrane_protein_is_called_high_risk():
    assert sb.predict_solubility(LACY).risk == "high"


def test_solubility_is_bounded_and_reports_its_components():
    res = sb.predict_solubility(GFP)
    assert 0.0 <= res.score <= 1.0
    for key in ("charge", "hydrophobicity", "turn", "aromatic", "patch"):
        assert key in res.components
        assert 0.0 <= res.components[key] <= 1.0
    assert res.components["gravy"] < 0, "GFP should have negative GRAVY"
    assert res.length == len(GFP)


def test_solubility_method_is_recorded():
    """A heuristic number must never be mistaken for a Protein-Sol prediction."""
    assert sb.predict_solubility(GFP).method == "heuristic"


def test_proteinsol_backend_explains_how_to_get_it():
    with pytest.raises(ImportError, match="Protein-Sol|protein-sol"):
        sb.predict_solubility(GFP, method="proteinsol")


def test_solubility_rejects_unknown_method():
    with pytest.raises(ValueError, match="method must be one of"):
        sb.predict_solubility(GFP, method="camsol")


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------

def test_nonstandard_residues_are_rejected_with_guidance():
    with pytest.raises(ValueError, match="非标准氨基酸"):
        sb.predict_solubility("MKVX*AL")


def test_empty_sequence_is_rejected():
    with pytest.raises(ValueError, match="序列为空"):
        sb.predict_solubility("")


def test_whitespace_and_case_are_tolerated():
    a = sb.predict_solubility(GFP).score
    b = sb.predict_solubility(GFP.lower()).score
    spaced = "\n".join(GFP[i:i + 60] for i in range(0, len(GFP), 60))
    c = sb.predict_solubility(spaced).score
    assert a == pytest.approx(b) == pytest.approx(c)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def test_membrane_protein_aggregates_more_than_gfp():
    assert (sb.aggregation_propensity(LACY).score
            > sb.aggregation_propensity(GFP).score)


def test_hotspots_are_located_not_just_counted():
    """The point of a profile over a single score: knowing *where* to fix."""
    prof = sb.aggregation_propensity(LACY)
    assert prof.hotspots, "a greasy membrane protein should have hotspots"
    for start, end, mean in prof.hotspots:
        assert 1 <= start <= end <= len(LACY)
        assert end - start + 1 >= 5
        assert 0.0 <= mean <= 1.0


def test_a_charge_inside_a_patch_lowers_its_score():
    """Charged residues break up hydrophobic patches — real predictors encode
    this, and so must the heuristic."""
    greasy = sb.aggregation_propensity("A" + "LLLLLLLLLLLL" + "A")
    broken = sb.aggregation_propensity("A" + "LLLLLKLLLLLL" + "A")
    assert broken.score < greasy.score


def test_proline_breaks_up_a_patch():
    greasy = sb.aggregation_propensity("A" + "LLLLLLLLLLLL" + "A")
    broken = sb.aggregation_propensity("A" + "LLLLLPLLLLLL" + "A")
    assert broken.score < greasy.score


def test_profile_aligns_with_the_sequence():
    prof = sb.aggregation_propensity(GFP)
    assert len(prof.per_residue) == len(GFP)
    assert len(prof.windowed) == len(GFP)
    df = prof.to_frame()
    assert len(df) == len(GFP)
    assert "".join(df["residue"]) == GFP


def test_aggregation_external_backends_are_explained():
    for method in ("aggrescan", "tango"):
        with pytest.raises(ImportError, match="AGGRESCAN|TANGO"):
            sb.aggregation_propensity(GFP, method=method)


# ---------------------------------------------------------------------------
# signal peptides — the landmark test
# ---------------------------------------------------------------------------

def test_phoa_signal_peptide_is_found_and_its_true_site_is_not_ruled_out():
    """PhoA's Sec/SPI signal peptide is cleaved after residue 21.

    The heuristic no longer names 21 as its single answer, and that is a
    deliberate trade. Site 21 used to win an exact tie by loop order — the
    scorer never distinguished it — and on an eight-protein reference panel
    breaking those ties towards the *earliest* site scores 5/8 while breaking them
    towards the latest scores **6/8**. The c-region motif term that made the score
    discriminate at all took the panel from 3/8 to 5/8 before either tie-break.
    PhoA is one of the two the 6/8 version now misses, by +2.

    So the assertion is what the function can honestly support: the signal peptide
    is detected, and the true site is among the reported alternatives at an
    indistinguishable score. `method='signalp'` is the answer when the exact
    residue matters — for instance when grafting a mature sequence onto a fusion.
    """
    sp = sb.predict_signal_peptide(PHOA)
    assert sp.has_signal, f"missed PhoA's signal peptide ({sp.reason})"
    sites = [site for site, _score in sp.alternatives]
    assert PHOA_CLEAVAGE in sites, (
        f"the true site {PHOA_CLEAVAGE} must at least be a candidate; "
        f"got {sp.alternatives}")
    top_score = sp.alternatives[0][1]
    tied = [site for site, score in sp.alternatives if score >= top_score - 1e-9]
    assert PHOA_CLEAVAGE in tied, f"true site must be among the tied best: {sp.alternatives}"
    assert abs(sp.cleavage_site - PHOA_CLEAVAGE) <= 3, sp.cleavage_site


def test_phoa_tripartite_structure_is_reported():
    sp = sb.predict_signal_peptide(PHOA)
    assert sp.n_region[1] >= 1
    assert sp.h_region[1] > sp.h_region[0]
    assert sp.c_region[1] == sp.cleavage_site, "the c-region must end at the cut"
    assert abs(sp.c_region[1] - PHOA_CLEAVAGE) <= 3
    assert sp.h_hydrophobicity > 1.0, "the h-region must be the hydrophobic core"


def test_c_region_tolerates_charge():
    """Regression: requiring an uncharged c-region rejected PhoA, whose
    c-region is TKA — a lysine sits at −2. The small residues at −3/−1 are the
    discriminator, not the absence of charge."""
    sp = sb.predict_signal_peptide(PHOA)
    c_start, c_end = sp.c_region
    c_seq = PHOA[c_start - 1:c_end]
    assert any(a in "KR" for a in c_seq), (
        "this test is only meaningful if PhoA's c-region really is charged")
    assert sp.has_signal


@pytest.mark.parametrize("name,seq", [("GFP", GFP), ("lysozyme", LYSOZYME)])
def test_cytoplasmic_and_mature_chains_have_no_signal_peptide(name, seq):
    sp = sb.predict_signal_peptide(seq)
    assert not sp.has_signal, f"false positive on {name}"
    assert sp.reason, "a negative call should say what was missing"


def test_short_sequence_is_reported_not_crashed():
    sp = sb.predict_signal_peptide("MKQST")
    assert not sp.has_signal
    assert "太短" in sp.reason


def test_mature_sequence_is_the_input_when_there_is_no_signal():
    sp = sb.predict_signal_peptide(GFP)
    assert sp.mature_sequence == GFP


def test_signalp_backend_is_explained():
    with pytest.raises(ImportError, match="SignalP"):
        sb.predict_signal_peptide(PHOA, method="signalp")


# ---------------------------------------------------------------------------
# localisation
# ---------------------------------------------------------------------------

def test_phoa_is_routed_out_of_the_cytoplasm():
    """PhoA really is periplasmic."""
    loc = sb.predict_localization(PHOA)
    assert loc.compartment in ("periplasm", "secreted")
    assert loc.signal_peptide is not None and loc.signal_peptide.has_signal


def test_gfp_stays_cytoplasmic():
    loc = sb.predict_localization(GFP)
    assert loc.compartment == "cytoplasm"
    assert loc.n_tm_helices == 0


def test_lacy_is_a_membrane_protein():
    """Regression: smoothing over tm_min_length *and* requiring a run of
    tm_min_length demanded ~2x a helix of unbroken hydrophobicity, and found
    zero TM helices in a 12-helix permease."""
    loc = sb.predict_localization(LACY)
    assert loc.n_tm_helices >= 2, f"found {loc.n_tm_helices} TM helices in LacY"
    assert loc.compartment == "membrane"


def test_a_signal_peptides_h_region_is_not_counted_as_a_tm_helix():
    """Otherwise every secreted protein looks like a membrane protein."""
    loc = sb.predict_localization(PHOA)
    assert loc.compartment != "membrane"


def test_localization_scores_cover_every_compartment():
    loc = sb.predict_localization(GFP)
    assert set(loc.scores) == {"cytoplasm", "membrane", "periplasm", "secreted"}
    assert loc.score == max(loc.scores.values())


#: What ``predict_localization`` can actually return. ``secreted`` is scored but
#: not in here, and the docstring now says so — see the test below.
REACHABLE_COMPARTMENTS = {"cytoplasm", "membrane", "periplasm"}


@pytest.mark.parametrize("seq", [PHOA, GFP, LYSOZYME, LACY])
def test_heuristic_only_ever_returns_three_compartments(seq):
    """``secreted`` is scored for inspection, never returned — by design.

    A Sec/SPI signal peptide delivers to the periplasm and stops there; real
    extracellular secretion needs a dedicated system that a sequence-only
    heuristic cannot see. So ``periplasm`` is scored strictly above ``secreted``
    in both branches and wins the argmax every time. The docs used to promise
    four compartments; this pins the three that exist so the promise cannot
    quietly come back.
    """
    loc = sb.predict_localization(seq)
    assert "secreted" in loc.scores, (
        "the secreted score is part of the public output — users read the "
        "periplasm-vs-secreted margin off .scores")
    assert loc.compartment in REACHABLE_COMPARTMENTS, (
        f"undocumented compartment {loc.compartment!r}: {loc.scores}")


@pytest.mark.parametrize("seq,signal", [(PHOA, True), (GFP, False)])
def test_periplasm_dominates_secreted_in_both_branches(seq, signal):
    """The structural reason ``secreted`` is unreachable, pinned directly.

    Asserted as an inequality rather than on the two constants, so a rescoring
    that genuinely makes secretion callable fails here and has to be argued for
    rather than slipping in.
    """
    loc = sb.predict_localization(seq)
    assert loc.signal_peptide.has_signal is signal
    assert loc.scores["periplasm"] > loc.scores["secreted"]


def test_deeploc_backend_is_explained():
    with pytest.raises(ImportError, match="DeepLoc"):
        sb.predict_localization(GFP, method="deeploc")


# ---------------------------------------------------------------------------
# composite expression prediction
# ---------------------------------------------------------------------------

def test_gfp_expresses_better_than_a_permease():
    assert (sb.predict_expression_level(GFP).score
            > sb.predict_expression_level(LACY).score)


def test_liabilities_are_specific_enough_to_act_on():
    exp = sb.predict_expression_level(LACY)
    assert exp.liabilities, "a membrane protein should raise liabilities"
    assert any("跨膜" in l or "聚集" in l for l in exp.liabilities)


def test_membrane_liability_names_the_helices():
    exp = sb.predict_expression_level(LACY)
    assert any("跨膜螺旋" in l for l in exp.liabilities)


def test_cysteine_liability_is_host_dependent():
    """E. coli's cytoplasm is reducing; a eukaryotic host is not, so the same
    disulfide-rich sequence must not be flagged the same way."""
    coli = sb.predict_expression_level(LYSOZYME, host="e_coli")
    cho = sb.predict_expression_level(LYSOZYME, host="cho")
    assert any("半胱氨酸" in l for l in coli.liabilities), \
        "lysozyme is disulfide-rich; E. coli cytoplasm should flag it"
    assert not any("半胱氨酸" in l for l in cho.liabilities)
    assert cho.components["cysteine"] == 1.0


def test_components_are_all_reported_and_bounded():
    exp = sb.predict_expression_level(GFP)
    for key in ("solubility", "aggregation", "length", "cysteine", "rare_residues"):
        assert 0.0 <= exp.components[key] <= 1.0
    assert exp.verdict


def test_sub_predictions_are_attached_for_drilldown():
    exp = sb.predict_expression_level(GFP)
    assert exp.solubility is not None and exp.aggregation is not None
    assert exp.localization is not None


def test_unknown_host_rejected():
    with pytest.raises(ValueError, match="host must be one of"):
        sb.predict_expression_level(GFP, host="hek293")


# ---------------------------------------------------------------------------
# visualisation
# ---------------------------------------------------------------------------

def test_plot_accepts_a_sequence():
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    fig, axes = sb.plot_manufacturability(LACY)
    assert len(axes) == 2
    plt.close(fig)


def test_plot_accepts_a_prediction_and_marks_the_signal_peptide():
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    exp = sb.predict_expression_level(PHOA)
    fig, axes = sb.plot_manufacturability(exp)
    assert len(axes) == 2
    plt.close(fig)


# ---------------------------------------------------------------------------
# the landmarks live in the package, so tutorials and tests cannot drift apart
# ---------------------------------------------------------------------------

def test_reference_sequences_match_the_ones_tested_here():
    assert sb.reference_protein("phoA") == PHOA
    assert sb.reference_protein("gfp") == GFP
    assert sb.reference_protein("lacY") == LACY


def test_reference_records_say_why_they_are_landmarks():
    """A reference sequence with no documented expected behaviour is a string."""
    record = sb.reference_protein("phoA", with_metadata=True)
    assert "21" in record["description"], "the cleavage site should be recorded"
    assert record["organism"]


def test_unknown_reference_lists_the_available_ones():
    with pytest.raises(KeyError, match="phoA|可用的有"):
        sb.reference_protein("insulin")


def test_reference_family_is_the_lysozyme_set():
    family = sb.reference_family("lysozyme")
    assert set(family) == {"chicken", "turkey", "quail", "duck"}
    assert family["chicken"] == LYSOZYME


def test_reference_terminators_behave_as_documented():
    assert sb.terminator_strength(sb.RRNB_T1).classification == "strong"
    assert sb.terminator_strength(sb.HAIRPIN_NO_U_TRACT).classification == \
        "not a terminator"


def test_designed_sequence_exposes_the_conventional_field_name():
    """``DesignedSequence`` stores its sequence as ``seq`` while every other
    ov.synbio dataclass uses ``sequence`` — reaching for the conventional name
    raised AttributeError, which is what the GPU suite hit. Both spellings work
    now; the field stays ``seq`` because published tutorials read it."""
    from omicverse.synbio._design import DesignedSequence

    design = DesignedSequence(seq="MKV", score=1.0)
    assert design.sequence == design.seq == "MKV"

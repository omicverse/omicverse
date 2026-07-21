"""ov.synbio A↔B coupling demo — the moat feature.

    enzyme_kcat(enzyme_seq, substrate)   ->  kcat            (protein layer B)
              |
    ec_model(GEM, {reaction: kcat})      ->  EC-constrained model (layer A)
              |
    fba(EC model)                        ->  growth / yield recomputed

"Edit one enzyme -> the metabolic network re-solves its yield."
Run on a real small model (e_coli_core) + a central-metabolism enzyme (PFK).
"""
import os, torch
import omicverse as ov

sb = ov.synbio
dev = "cuda" if torch.cuda.is_available() else "cpu"

# a real E. coli phosphofructokinase (PfkA) fragment as the enzyme under design;
# the substrate is fructose-6-phosphate (its physiological substrate).
PFK_SEQ = ("MIKKIGVLTSGGDAPGMNAAIRGVVRSALTEGLEVMGIYDGYLGLYEDRMVQLDRYSVSDMINRGGTFLGSARFPEFRD"
           "ENIRAVAIENLKKRGIDALVVIGGDGSYMGAMRLTEMGFPCIGLPGTIDNDIKGTDYTIGFFTALSTVVEAIDRLRDT")
SUBSTRATE_F6P = "OCC1OC(O)(COP(=O)(O)O)C(O)C1O"

m = sb.load_gem("e_coli_core")
wt = sb.fba(m).objective_value
print(f"[0] wild-type max growth (no enzyme constraint): {wt:.4f} /h\n")

# --- protein layer: predict a turnover number for the enzyme ---------------
k_wt = sb.enzyme_kcat(PFK_SEQ, SUBSTRATE_F6P, device=dev, verbose=False)
print(f"[1] enzyme_kcat(PfkA, F6P)  ->  kcat = {k_wt.kcat:.2f} /s  (engine={k_wt.engine})")

# fix a protein budget from the wild-type enzyme so variants compare fairly
ecm = sb.ec_model(m, {"PFK": k_wt.kcat})
pool = ecm.synbio_ec["total_protein"]
g_wt = sb.fba(ecm).objective_value
print(f"[2] ec_model(GEM, PFK={k_wt.kcat:.2f})  ->  fba growth = {g_wt:.4f} /h "
      f"(protein pool fixed at {pool:.3g})\n")

# --- engineer a faster and a slower enzyme variant, same protein budget -----
print("[3] 'Edit the enzyme' -> re-solve yield under the SAME protein budget:")
print(f"    {'kcat (/s)':>12} {'growth (/h)':>14}   {'vs WT':>8}")
for label, kcat in [("slower 0.3x", k_wt.kcat*0.3),
                    ("wild-type", k_wt.kcat),
                    ("faster 3x", k_wt.kcat*3.0),
                    ("faster 10x", k_wt.kcat*10.0)]:
    ecm_v = sb.ec_model(m, {"PFK": kcat}, total_protein=pool)
    g = sb.fba(ecm_v).objective_value
    print(f"    {kcat:12.2f} {g:14.4f}   {g-g_wt:+8.4f}   ({label})")

print("\n=== COUPLING DEMO DONE: enzyme kcat -> enzyme-constrained FBA -> yield ===")

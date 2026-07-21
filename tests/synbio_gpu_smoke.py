"""GPU smoke test for ov.synbio layer B (run on a CUDA node)."""
import os, time, json
import torch
import omicverse as ov

sb = ov.synbio
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

SEQ = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
       "QTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFG")
results = {}

def mem():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()/1024**3
    return 0.0

# 1) protein_embed
torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
t=time.time(); X = sb.protein_embed([SEQ, SEQ[:60]], model="esm2_t33_650M", device=dev)
results["protein_embed"] = {"shape": list(X.shape), "sec": round(time.time()-t,2), "gpu_gb": round(mem(),2)}
print("protein_embed:", results["protein_embed"])

# 2) variant_effect (saturation scan on shorter seq)
torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
short = SEQ[:60]
t=time.time(); df = sb.variant_effect(short, model="esm1v", device=dev)
results["variant_effect"] = {"rows": len(df), "sec": round(time.time()-t,2), "gpu_gb": round(mem(),2),
                             "top": df.iloc[0].to_dict(), "bottom": df.iloc[-1].to_dict()}
print("variant_effect:", len(df), "muts | top", df.iloc[0]['mutation'], round(df.iloc[0]['score'],2),
      "| worst", df.iloc[-1]['mutation'], round(df.iloc[-1]['score'],2), "|", results["variant_effect"]["sec"],"s")

# 3) predict_structure (ESMFold)
torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
t=time.time(); pred = sb.predict_structure(short, device=dev, out_path="/tmp/synbio_esmfold.pdb")
results["predict_structure"] = {"len": len(short), "mean_plddt": round(pred.mean_plddt,1),
                                "device": pred.device, "sec": round(time.time()-t,2), "gpu_gb": round(mem(),2),
                                "pdb": pred.path}
print("predict_structure:", results["predict_structure"])

# 4) inverse_design (ProteinMPNN) on the folded backbone
t=time.time(); designs = sb.inverse_design("/tmp/synbio_esmfold.pdb", num_sequences=4, device=dev)
results["inverse_design"] = {"n": len(designs), "sec": round(time.time()-t,2),
                             "native_score": round(designs[0].score,3),
                             "best_design_score": round(designs[1].score,3) if len(designs)>1 else None}
print("inverse_design:", results["inverse_design"])

# 5) stability_ddg (ProteinMPNN proxy) — a few mutations
t=time.time(); ddg = sb.stability_ddg("/tmp/synbio_esmfold.pdb",
                                      mutations=["A2V","K3E","T4A"], device=dev)
results["stability_ddg"] = {"rows": len(ddg), "sec": round(time.time()-t,2),
                            "vals": ddg[["mutation","ddg_proxy"]].to_dict("records")}
print("stability_ddg:", results["stability_ddg"])

# 6) enzyme_kcat baseline (sequence-sensitive)
t=time.time(); k1 = sb.enzyme_kcat(short, "C(C(=O)O)N", device=dev)
k2 = sb.enzyme_kcat(short[:30]+"A"+short[31:], "C(C(=O)O)N", device=dev)
results["enzyme_kcat"] = {"kcat_wt": round(k1.kcat,3), "kcat_mut": round(k2.kcat,3),
                          "sec": round(time.time()-t,2)}
print("enzyme_kcat:", results["enzyme_kcat"])

# 7) enzyme_function (ESM knn vs bundled EC reference)
t=time.time(); ef = sb.enzyme_function(short, device=dev)
results["enzyme_function"] = {"top_ec": ef.top_ec, "preds": ef.predictions, "sec": round(time.time()-t,2)}
print("enzyme_function:", results["enzyme_function"])

results["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
json.dump(results, open("/tmp/synbio_gpu_results.json","w"), indent=2, default=str)
print("\n=== GPU SMOKE DONE ===")
print(json.dumps(results, indent=2, default=str))

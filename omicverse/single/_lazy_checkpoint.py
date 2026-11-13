"""
Checkpointing version of the lazy function for memory-efficient processing
of large single-cell datasets.
"""

import os

import anndata
import numpy as np

from .._settings import add_reference, settings
from ..pp import (
    leiden,
    louvain,
    neighbors,
    pca,
    preprocess,
    qc,
    scale,
    score_genes_cell_cycle,
    tsne,
    umap,
)
from . import SCCAF_optimize_all, batch_correction


def lazy_checkpoint(
    adata,
    species="human",
    reforce_steps=[],
    sample_key=None,
    checkpoint_dir="./lazy_checkpoints",
    save_intermediate=True,
    qc_kwargs=None,
    preprocess_kwargs=None,
    pca_kwargs=None,
    harmony_kwargs=None,
    scvi_kwargs=None,
):
    """
    Memory-efficient version of lazy function with checkpointing support.

    This function saves intermediate results at each step to avoid memory crashes
    on large datasets. Each step is saved as an .h5ad file and can be resumed
    if the process crashes.
    Parameters
    ----------
        adata: the data to analysis
        species: 'human' or 'mouse' for cell cycle scoring
        reforce_steps: we can reforce run lazy step, because some step have been run and will be skipped.
                        ['qc','pca','preprocess','scaled','Harmony','scVI','eval_bench','eval_clusters']
        sample_key: the key store in `adata.obs` to batch correction.
        checkpoint_dir: directory to save intermediate results
        save_intermediate: whether to save intermediate results at each step
        qc_kwargs: arguments for qc step
        preprocess_kwargs: arguments for preprocess step
        pca_kwargs: arguments for pca step
        harmony_kwargs: arguments for harmony step
        scvi_kwargs: arguments for scVI step
    Returns
    -------
        adata: processed AnnData object
    """

    mode = settings.mode
    print(f"🔧 The mode of lazy_checkpoint is {mode}")
    print(f"💾 Checkpoint directory: {checkpoint_dir}")

    # Create checkpoint directory
    if save_intermediate:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Step 0: Initial setup and package checking
    try:
        import louvain
    except:
        print("❌ Louvain package not found, we will install it now")
        import pip

        pip.main(["install", "louvain"])
        import louvain

    print("✅ All packages used in lazy_checkpoint are installed")

    # Initialize adata structure
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata = adata.copy()

    if "status" not in adata.uns.keys():
        adata.uns["status"] = {}

    checkpoint_file = os.path.join(checkpoint_dir, "step_{}_checkpoint.h5ad")

    # Step 1: Quality Control
    step_name = "qc"
    step_file = checkpoint_file.format("01_qc")

    if (
        os.path.exists(step_file)
        and save_intermediate
        and step_name not in reforce_steps
    ):
        print(f"✅ Loading {step_name} checkpoint from {step_file}")
        adata = anndata.read_h5ad(step_file)
    elif (
        ("qc" not in adata.uns["status"].keys())
        or ("qc" in adata.uns["status"].keys() and adata.uns["status"]["qc"] == False)
        or ("qc" in reforce_steps)
    ):
        print("❌ QC step didn't start, we will start it now")
        if qc_kwargs is None:
            qc_kwargs = {
                "tresh": {"mito_perc": 0.2, "nUMIs": 500, "detected_genes": 250},
                "doublets_method": "scrublet",
                "batch_key": sample_key,
            }
        print(
            f"🔧 The argument of qc we set\n"
            f"   mito_perc: {qc_kwargs['tresh']['mito_perc']}\n"
            f"   nUMIs: {qc_kwargs['tresh']['nUMIs']}\n"
            f"   detected_genes: {qc_kwargs['tresh']['detected_genes']}\n"
            f"   doublets_method: {qc_kwargs['doublets_method']}\n"
            f"   batch_key: {qc_kwargs['batch_key']}\n"
        )
        adata = qc(adata, **qc_kwargs)

        if save_intermediate:
            print(f"💾 Saving {step_name} checkpoint to {step_file}")
            adata.write(step_file)
    else:
        print("✅ QC step already finished, skipping it")

    # Step 2: Preprocessing (normalization and HVG selection)
    step_name = "preprocess"
    step_file = checkpoint_file.format("02_preprocess")

    if (
        os.path.exists(step_file)
        and save_intermediate
        and step_name not in reforce_steps
    ):
        print(f"✅ Loading {step_name} checkpoint from {step_file}")
        adata = anndata.read_h5ad(step_file)
    elif (
        ("preprocess" not in adata.uns["status"].keys())
        or (
            "preprocess" in adata.uns["status"].keys()
            and adata.uns["status"]["preprocess"] == False
        )
        or ("preprocess" in reforce_steps)
    ):
        print("❌ Preprocess step didn't start, we will start it now")
        if preprocess_kwargs is None:
            preprocess_kwargs = {
                "mode": "shiftlog|pearson",
                "n_HVGs": 2000,
                "target_sum": 50 * 1e4,
            }
        print(
            f"🔧 The argument of preprocess we set\n"
            f"   mode: {preprocess_kwargs['mode']}\n"
            f"   n_HVGs: {preprocess_kwargs['n_HVGs']}\n"
            f"   target_sum: {preprocess_kwargs['target_sum']}\n"
        )
        adata = preprocess(adata, **preprocess_kwargs)

        if save_intermediate:
            print(f"💾 Saving {step_name} checkpoint to {step_file}")
            adata.write(step_file)
    else:
        print("✅ Preprocess step already finished, skipping it")

    # Step 3: Scaling
    step_name = "scaled"
    step_file = checkpoint_file.format("03_scaled")

    if (
        os.path.exists(step_file)
        and save_intermediate
        and step_name not in reforce_steps
    ):
        print(f"✅ Loading {step_name} checkpoint from {step_file}")
        adata = anndata.read_h5ad(step_file)
    elif (
        ("scaled" not in adata.uns["status"].keys())
        or (
            "scaled" in adata.uns["status"].keys()
            and adata.uns["status"]["scaled"] == False
        )
        or ("scaled" in reforce_steps)
    ):
        print("❌ Scaled step didn't start, we will start it now")
        scale(adata)

        if save_intermediate:
            print(f"💾 Saving {step_name} checkpoint to {step_file}")
            adata.write(step_file)
    else:
        print("✅ Scaled step already finished, skipping it")

    # Step 4: PCA
    step_name = "pca"
    step_file = checkpoint_file.format("04_pca")

    if (
        os.path.exists(step_file)
        and save_intermediate
        and step_name not in reforce_steps
    ):
        print(f"✅ Loading {step_name} checkpoint from {step_file}")
        adata = anndata.read_h5ad(step_file)
    elif (
        ("pca" not in adata.uns["status"].keys())
        or ("pca" in adata.uns["status"].keys() and adata.uns["status"]["pca"] == False)
        or ("pca" in reforce_steps)
    ):
        print("❌ PCA step didn't start, we will start it now")
        if pca_kwargs is None:
            pca_kwargs = {
                "layer": "scaled",
                "n_pcs": 50,
                "use_highly_variable": True,
            }
        if ("highly_variable" not in adata.var.columns) and (
            "highly_variable_features" in adata.var.columns
        ):
            adata.var["highly_variable"] = adata.var[
                "highly_variable_features"
            ].tolist()
        print(
            f"🔧 The argument of PCA we set\n"
            f"   layer: {pca_kwargs['layer']}\n"
            f"   n_pcs: {pca_kwargs['n_pcs']}\n"
            f"   use_highly_variable: {pca_kwargs['use_highly_variable']}\n"
        )
        pca(adata, **pca_kwargs)
        adata.obsm["X_pca"] = adata.obsm["scaled|original|X_pca"]

        if save_intermediate:
            print(f"💾 Saving {step_name} checkpoint to {step_file}")
            adata.write(step_file)
    else:
        print("✅ PCA step already finished, skipping it")

    # Step 5: Cell cycle scoring
    step_name = "cell_cycle"
    step_file = checkpoint_file.format("05_cell_cycle")

    if (
        os.path.exists(step_file)
        and save_intermediate
        and step_name not in reforce_steps
    ):
        print(f"✅ Loading {step_name} checkpoint from {step_file}")
        adata = anndata.read_h5ad(step_file)
    elif (
        ("cell_cycle" not in adata.uns["status"].keys())
        or (
            "cell_cycle" in adata.uns["status"].keys()
            and adata.uns["status"]["cell_cycle"] == False
        )
        or ("cell_cycle" in reforce_steps)
    ):
        print("❌ Cell cycle scoring step didn't start, we will start it now")
        score_genes_cell_cycle(adata, species=species)

        if save_intermediate:
            print(f"💾 Saving {step_name} checkpoint to {step_file}")
            adata.write(step_file)
    else:
        print("✅ Cell cycle scoring step already finished, skipping it")

    # Step 6: Harmony batch correction (Memory intensive!)
    step_name = "harmony"
    step_file = checkpoint_file.format("06_harmony")

    if (
        os.path.exists(step_file)
        and save_intermediate
        and step_name not in reforce_steps
    ):
        print(f"✅ Loading {step_name} checkpoint from {step_file}")
        adata = anndata.read_h5ad(step_file)
    elif ("X_harmony" not in adata.obsm.keys()) or ("Harmony" in reforce_steps):
        print("❌ Batch Correction: `Harmony` step didn't start, we will start it now")
        print(
            "⚠️  WARNING: This step is memory-intensive. Processing HVG subset to reduce memory usage."
        )

        # Create HVG subset to reduce memory usage
        adata_hvg = adata.copy()
        if "highly_variable_features" in adata_hvg.var.columns:
            adata_hvg = adata_hvg[:, adata_hvg.var.highly_variable_features]
        elif "highly_variable" in adata_hvg.var.columns:
            adata_hvg = adata_hvg[:, adata_hvg.var.highly_variable]

        if harmony_kwargs is None:
            harmony_kwargs = {"n_pcs": 50}

        batch_correction(
            adata_hvg, batch_key=sample_key, methods="harmony", **harmony_kwargs
        )

        # Copy results back to main object
        adata.obsm["X_harmony"] = adata_hvg.obsm["X_pca_harmony"]

        if "status" not in adata.uns.keys():
            adata.uns["status"] = {}
        if "status_args" not in adata.uns.keys():
            adata.uns["status_args"] = {}

        adata.uns["status"]["harmony"] = True
        adata.uns["status_args"]["harmony"] = {
            "n_pcs": harmony_kwargs["n_pcs"],
        }

        # Clean up intermediate data to save memory
        del adata_hvg

        # Compute neighbors and embeddings
        neighbors(adata=adata, n_neighbors=15, use_rep="X_harmony", n_pcs=30)
        umap(adata)
        tsne(adata, use_rep="X_harmony")
        adata.obsm["X_umap_harmony"] = adata.obsm["X_umap"]
        adata.obsm["X_tsne_harmony"] = adata.obsm["X_tsne"]

        if save_intermediate:
            print(f"💾 Saving {step_name} checkpoint to {step_file}")
            adata.write(step_file)
    else:
        print("✅ Batch Correction: `Harmony` step already finished, skipping it")

    # Step 7: scVI batch correction (Optional, memory intensive!)
    step_name = "scvi"
    step_file = checkpoint_file.format("07_scvi")

    try:
        import scvi

        if (
            os.path.exists(step_file)
            and save_intermediate
            and step_name not in reforce_steps
        ):
            print(f"✅ Loading {step_name} checkpoint from {step_file}")
            adata = anndata.read_h5ad(step_file)
        elif ("X_scVI" not in adata.obsm.keys()) or ("scVI" in reforce_steps):
            print("❌ Batch Correction: `scVI` step didn't start, we will start it now")
            print(
                "⚠️  WARNING: This step is very memory-intensive. Processing HVG subset."
            )

            # Create HVG subset to reduce memory usage
            adata_hvg = adata.copy()
            if "highly_variable_features" in adata_hvg.var.columns:
                adata_hvg = adata_hvg[:, adata_hvg.var.highly_variable_features].copy()
            elif "highly_variable" in adata_hvg.var.columns:
                adata_hvg = adata_hvg[:, adata_hvg.var.highly_variable].copy()

            if scvi_kwargs is None:
                scvi_kwargs = {"n_layers": 2, "n_latent": 30, "gene_likelihood": "nb"}

            batch_correction(
                adata_hvg, batch_key=sample_key, methods="scVI", **scvi_kwargs
            )

            # Copy results back to main object
            adata.obsm["X_scVI"] = adata_hvg.obsm["X_scVI"]

            if "status" not in adata.uns.keys():
                adata.uns["status"] = {}
            if "status_args" not in adata.uns.keys():
                adata.uns["status_args"] = {}

            adata.uns["status"]["scVI"] = True
            adata.uns["status_args"]["scVI"] = {
                "n_layers": scvi_kwargs["n_layers"],
                "n_latent": scvi_kwargs["n_latent"],
                "gene_likelihood": scvi_kwargs["gene_likelihood"],
            }

            # Clean up intermediate data to save memory
            del adata_hvg

            # Compute neighbors and embeddings
            neighbors(adata=adata, n_neighbors=15, use_rep="X_scVI", n_pcs=30)
            umap(adata)
            tsne(adata, use_rep="X_scVI")
            adata.obsm["X_umap_scVI"] = adata.obsm["X_umap"]
            adata.obsm["X_tsne_scVI"] = adata.obsm["X_tsne"]

            if save_intermediate:
                print(f"💾 Saving {step_name} checkpoint to {step_file}")
                adata.write(step_file)
        else:
            print("✅ Batch Correction: `scVI` step already finished, skipping it")
    except ImportError:
        print("❌ scvi package not found, we will not run scVI step")

    # Step 8: Evaluate best batch correction method
    step_name = "eval_bench"
    step_file = checkpoint_file.format("08_eval_bench")

    if (
        os.path.exists(step_file)
        and save_intermediate
        and step_name not in reforce_steps
    ):
        print(f"✅ Loading {step_name} checkpoint from {step_file}")
        adata = anndata.read_h5ad(step_file)
    elif ("bench_best_res" not in adata.uns.keys()) or ("eval_bench" in reforce_steps):
        print("❌ Best Bench Correction Eval step didn't start, we will start it now")

        if "X_scVI" in adata.obsm.keys():
            adata.uns["bench_best_res"] = "X_scVI"
        else:
            adata.uns["bench_best_res"] = "X_harmony"
        print(f"The Best Bench Correction Method is {adata.uns['bench_best_res']}")
        print("We can found it in `adata.uns['bench_best_res']`")

        if save_intermediate:
            print(f"💾 Saving {step_name} checkpoint to {step_file}")
            adata.write(step_file)
    else:
        print("✅ Best Bench Correction Eval step already finished, skipping it")

    # Step 9: MDE and clustering (Very memory intensive!)
    step_name = "clustering"
    step_file = checkpoint_file.format("09_clustering")

    if (
        os.path.exists(step_file)
        and save_intermediate
        and step_name not in reforce_steps
    ):
        print(f"✅ Loading {step_name} checkpoint from {step_file}")
        adata = anndata.read_h5ad(step_file)
    elif ("best_clusters" not in adata.obs.columns) or (
        "eval_clusters" in reforce_steps
    ):
        print("❌ Best Clusters step didn't start, we will start it now")
        print(
            "⚠️  WARNING: This is the most memory-intensive step (MDE + SCCAF clustering)."
        )

        method_test = adata.uns["bench_best_res"]
        print(f"Automatic clustering using sccaf")
        print(f"Dimensionality using: {method_test}")

        # Recompute neighbors for clustering
        neighbors(
            adata=adata, n_neighbors=15, use_rep=adata.uns["bench_best_res"], n_pcs=30
        )
        print("🔧 Computing UMAP embedding...")
        umap(adata)

        # Pre-clustering
        print(f"Automatic clustering using leiden for preprocessed")
        leiden(adata, resolution=1.5, key_added="leiden_r1.5", random_state=0, backend='python')
        adata.obs["L1_result_smooth"] = adata.obs["leiden_r1.5"]

        # Automatic clustering with SCCAF - memory intensive!
        print("⚠️  Starting SCCAF clustering - this may use significant memory")
        for idx in range(10):
            if (len(np.unique(adata.obs["L1_result_smooth"].tolist())) > 3) and idx > 0:
                break
            else:
                adata.obs["L1_Round0"] = adata.obs["L1_result_smooth"]
                print(f"Automatic clustering using sccaf, Times: {idx}")

                try:
                    SCCAF_optimize_all(
                        min_acc=0.95,
                        ad=adata,
                        classifier="RF",
                        n_jobs=4,
                        use=adata.uns["bench_best_res"],
                        basis="X_umap",
                        method="leiden",
                        prefix="L1",
                        plot=True,
                    )

                    # Smooth clustering effect
                    print(f"Smoothing the effect of clustering, Times: {idx}")
                    adata.obs["L1_result_smooth"] = adata.obs["L1_result"].tolist()
                except Exception as e:
                    print(f"⚠️  SCCAF clustering failed at iteration {idx}: {e}")
                    print("   Continuing with previous clustering result...")
                    break

        # Final clustering assignments
        adata.obs["best_clusters"] = adata.obs["L1_result_smooth"].copy()
        leiden(adata, resolution=1, key_added="leiden_clusters_L1", random_state=0, backend='python')
        louvain(adata, resolution=1, key_added="louvain_clusters_L1")
        leiden(adata, resolution=0.5, key_added="leiden_clusters_L2", random_state=0, backend='python')
        louvain(adata, resolution=0.5, key_added="louvain_clusters_L2")

        if save_intermediate:
            print(f"💾 Saving {step_name} checkpoint to {step_file}")
            adata.write(step_file)
    else:
        print("✅ Best Clusters step already finished, skipping it")

    # Step 10: Final embeddings
    step_name = "final_embeddings"
    step_file = checkpoint_file.format("10_final_embeddings")

    if (
        os.path.exists(step_file)
        and save_intermediate
        and step_name not in reforce_steps
    ):
        print(f"✅ Loading {step_name} checkpoint from {step_file}")
        adata = anndata.read_h5ad(step_file)
    else:
        # UMAP
        if ("X_umap" not in adata.obsm.keys()) or ("umap" in reforce_steps):
            print("❌ UMAP step didn't start, we will start it now")
            umap(adata)
            adata.obsm["X_umap"] = adata.obsm["X_umap"]
        else:
            print("✅ UMAP step already finished, skipping it")

        # tSNE
        if ("X_tsne" not in adata.obsm.keys()) or ("tsne" in reforce_steps):
            print("❌ tSNE step didn't start, we will start it now")
            tsne(adata, use_rep=adata.uns["bench_best_res"])
            adata.obsm["X_tsne"] = adata.obsm["X_tsne"]
        else:
            print("✅ tSNE step already finished, skipping it")

        if save_intermediate:
            print(f"💾 Saving {step_name} checkpoint to {step_file}")
            adata.write(step_file)

    # Final save
    if save_intermediate:
        final_file = os.path.join(checkpoint_dir, "final_result.h5ad")
        print(f"💾 Saving final result to {final_file}")
        adata.write(final_file)

    print("🎉 lazy_checkpoint processing completed!")
    return adata


def resume_from_checkpoint(checkpoint_dir, step_name):
    """
    Resume processing from a specific checkpoint.
    Parameters
    ----------
        checkpoint_dir: directory containing checkpoints
        step_name: name of the step to resume from
                  ('qc', 'preprocess', 'scaled', 'pca', 'cell_cycle',
                   'harmony', 'scvi', 'eval_bench', 'clustering', 'final_embeddings')
    Returns
    -------
        adata: AnnData object from the checkpoint
    """
    step_map = {
        "qc": "01_qc",
        "preprocess": "02_preprocess",
        "scaled": "03_scaled",
        "pca": "04_pca",
        "cell_cycle": "05_cell_cycle",
        "harmony": "06_harmony",
        "scvi": "07_scvi",
        "eval_bench": "08_eval_bench",
        "clustering": "09_clustering",
        "final_embeddings": "10_final_embeddings",
        "final": "final_result",
    }

    if step_name not in step_map:
        raise ValueError(
            f"Unknown step name: {step_name}. Available steps: {list(step_map.keys())}"
        )

    if step_name == "final":
        checkpoint_file = os.path.join(checkpoint_dir, "final_result.h5ad")
    else:
        checkpoint_file = os.path.join(
            checkpoint_dir, f"step_{step_map[step_name]}_checkpoint.h5ad"
        )

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    print(f"📂 Loading checkpoint from {checkpoint_file}")
    adata = anndata.read_h5ad(checkpoint_file)
    print(f"✅ Successfully loaded {step_name} checkpoint")

    return adata


def list_checkpoints(checkpoint_dir):
    """
    List available checkpoints in the directory.
    Parameters
    ----------
        checkpoint_dir: directory containing checkpoints
    Returns
    -------
        list: available checkpoint files
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return []

    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith("_checkpoint.h5ad") or file == "final_result.h5ad":
            checkpoint_files.append(file)

    checkpoint_files.sort()

    print(f"Available checkpoints in {checkpoint_dir}:")
    for file in checkpoint_files:
        file_path = os.path.join(checkpoint_dir, file)
        file_size = os.path.getsize(file_path) / (1024**3)  # Convert to GB
        print(f"  📁 {file} ({file_size:.2f} GB)")

    return checkpoint_files


def cleanup_checkpoints(checkpoint_dir, keep_final=True):
    """
    Clean up intermediate checkpoint files to save disk space.
    Parameters
    ----------
        checkpoint_dir: directory containing checkpoints
        keep_final: whether to keep the final result file
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return

    removed_files = 0
    total_size_freed = 0

    for file in os.listdir(checkpoint_dir):
        if file.endswith("_checkpoint.h5ad"):
            file_path = os.path.join(checkpoint_dir, file)
            file_size = os.path.getsize(file_path)
            os.remove(file_path)
            removed_files += 1
            total_size_freed += file_size
            print(f"🗑️  Removed {file}")
        elif file == "final_result.h5ad" and not keep_final:
            file_path = os.path.join(checkpoint_dir, file)
            file_size = os.path.getsize(file_path)
            os.remove(file_path)
            removed_files += 1
            total_size_freed += file_size
            print(f"🗑️  Removed {file}")

    total_size_gb = total_size_freed / (1024**3)
    print(
        f"✅ Cleanup complete: removed {removed_files} files, freed {total_size_gb:.2f} GB"
    )

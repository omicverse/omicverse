"""
Step-by-step processing functions for memory-efficient single-cell analysis.
This allows users to run each step individually and save results between steps.
"""

import os

import anndata
import numpy as np

from .._settings import add_reference, settings
from ..pp import (
    leiden,
    louvain,
    mde,
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


def lazy_step_qc(
    adata, species="human", sample_key=None, output_path=None, **qc_kwargs
):
    """
    Step 1: Quality control and doublet detection.
    Parameters
    ----------
        adata: AnnData object
        species: 'human' or 'mouse'
        sample_key: batch key for QC
        output_path: path to save the result (optional)
        **qc_kwargs: additional arguments for qc function
    Returns
    -------
        adata: processed AnnData object
    """
    print("🔧 Step 1/10: Quality Control and Doublet Detection")

    if "tresh" not in qc_kwargs:
        qc_kwargs["tresh"] = {"mito_perc": 0.2, "nUMIs": 500, "detected_genes": 250}
    if "doublets_method" not in qc_kwargs:
        qc_kwargs["doublets_method"] = "scrublet"
    if "batch_key" not in qc_kwargs:
        qc_kwargs["batch_key"] = sample_key

    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata = adata.copy()

    if "status" not in adata.uns.keys():
        adata.uns["status"] = {}

    print(
        f"🔧 QC parameters:\n"
        f"   mito_perc: {qc_kwargs['tresh']['mito_perc']}\n"
        f"   nUMIs: {qc_kwargs['tresh']['nUMIs']}\n"
        f"   detected_genes: {qc_kwargs['tresh']['detected_genes']}\n"
        f"   doublets_method: {qc_kwargs['doublets_method']}\n"
        f"   batch_key: {qc_kwargs['batch_key']}\n"
    )

    adata = qc(adata, **qc_kwargs)

    if output_path:
        print(f"💾 Saving QC result to {output_path}")
        adata.write(output_path)

    print("✅ Step 1/10 completed: Quality Control")
    return adata


def lazy_step_preprocess(adata, output_path=None, **preprocess_kwargs):
    """
    Step 2: Normalization and highly variable gene selection.
    Parameters
    ----------
        adata: AnnData object
        output_path: path to save the result (optional)
        **preprocess_kwargs: additional arguments for preprocess function
    Returns
    -------
        adata: processed AnnData object
    """
    print("🔧 Step 2/10: Normalization and HVG Selection")

    if "mode" not in preprocess_kwargs:
        preprocess_kwargs["mode"] = "shiftlog|pearson"
    if "n_HVGs" not in preprocess_kwargs:
        preprocess_kwargs["n_HVGs"] = 2000
    if "target_sum" not in preprocess_kwargs:
        preprocess_kwargs["target_sum"] = 50 * 1e4

    print(
        f"🔧 Preprocess parameters:\n"
        f"   mode: {preprocess_kwargs['mode']}\n"
        f"   n_HVGs: {preprocess_kwargs['n_HVGs']}\n"
        f"   target_sum: {preprocess_kwargs['target_sum']}\n"
    )

    adata = preprocess(adata, **preprocess_kwargs)

    if output_path:
        print(f"💾 Saving preprocess result to {output_path}")
        adata.write(output_path)

    print("✅ Step 2/10 completed: Preprocessing")
    return adata


def lazy_step_scale(adata, output_path=None):
    """
    Step 3: Data scaling.
    Parameters
    ----------
        adata: AnnData object
        output_path: path to save the result (optional)
    Returns
    -------
        adata: processed AnnData object
    """
    print("🔧 Step 3/10: Data Scaling")

    scale(adata)

    if output_path:
        print(f"💾 Saving scaling result to {output_path}")
        adata.write(output_path)

    print("✅ Step 3/10 completed: Data Scaling")
    return adata


def lazy_step_pca(adata, output_path=None, **pca_kwargs):
    """
    Step 4: Principal Component Analysis.
    Parameters
    ----------
        adata: AnnData object
        output_path: path to save the result (optional)
        **pca_kwargs: additional arguments for PCA
    Returns
    -------
        adata: processed AnnData object
    """
    print("🔧 Step 4/10: Principal Component Analysis")

    if "layer" not in pca_kwargs:
        pca_kwargs["layer"] = "scaled"
    if "n_pcs" not in pca_kwargs:
        pca_kwargs["n_pcs"] = 50
    if "use_highly_variable" not in pca_kwargs:
        pca_kwargs["use_highly_variable"] = True

    if ("highly_variable" not in adata.var.columns) and (
        "highly_variable_features" in adata.var.columns
    ):
        adata.var["highly_variable"] = adata.var[
            "highly_variable_features"
        ].tolist()

    print(
        f"🔧 PCA parameters:\n"
        f"   layer: {pca_kwargs['layer']}\n"
        f"   n_pcs: {pca_kwargs['n_pcs']}\n"
        f"   use_highly_variable: {pca_kwargs['use_highly_variable']}\n"
    )

    pca(adata, **pca_kwargs)
    adata.obsm["X_pca"] = adata.obsm["scaled|original|X_pca"]

    if output_path:
        print(f"💾 Saving PCA result to {output_path}")
        adata.write(output_path)

    print("✅ Step 4/10 completed: Principal Component Analysis")
    return adata


def lazy_step_cell_cycle(adata, species="human", output_path=None):
    """
    Step 5: Cell cycle scoring.
    Parameters
    ----------
        adata: AnnData object
        species: 'human' or 'mouse'
        output_path: path to save the result (optional)
    Returns
    -------
        adata: processed AnnData object
    """
    print("🔧 Step 5/10: Cell Cycle Scoring")

    score_genes_cell_cycle(adata, species=species)

    if output_path:
        print(f"💾 Saving cell cycle result to {output_path}")
        adata.write(output_path)

    print("✅ Step 5/10 completed: Cell Cycle Scoring")
    return adata


def lazy_step_harmony(adata, sample_key, output_path=None, **harmony_kwargs):
    """
    Step 6: Harmony batch correction (memory intensive).
    Parameters
    ----------
        adata: AnnData object
        sample_key: batch key for correction
        output_path: path to save the result (optional)
        **harmony_kwargs: additional arguments for harmony
    Returns
    -------
        adata: processed AnnData object
    """
    print("🔧 Step 6a/10: Harmony Batch Correction")
    print("⚠️  WARNING: This step is memory-intensive")

    if "n_pcs" not in harmony_kwargs:
        harmony_kwargs["n_pcs"] = 50

    # Use only HVG subset to reduce memory usage
    print("🔧 Processing HVG subset to reduce memory usage...")
    adata_hvg = adata.copy()
    if "highly_variable_features" in adata_hvg.var.columns:
        adata_hvg = adata_hvg[:, adata_hvg.var.highly_variable_features]
    elif "highly_variable" in adata_hvg.var.columns:
        adata_hvg = adata_hvg[:, adata_hvg.var.highly_variable]

    print(f"🔧 Harmony parameters:\n   n_pcs: {harmony_kwargs['n_pcs']}")

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
    adata.uns["status_args"]["harmony"] = {"n_pcs": harmony_kwargs["n_pcs"]}

    # Clean up to save memory
    del adata_hvg

    # Compute neighbors and embeddings for Harmony
    print("🔧 Computing neighbors and embeddings for Harmony...")
    neighbors(adata=adata, n_neighbors=15, use_rep="X_harmony", n_pcs=30)
    umap(adata)
    tsne(adata, use_rep="X_harmony")
    adata.obsm["X_umap_harmony"] = adata.obsm["X_umap"]
    adata.obsm["X_tsne_harmony"] = adata.obsm["X_tsne"]

    if output_path:
        print(f"💾 Saving Harmony result to {output_path}")
        adata.write(output_path)

    print("✅ Step 6a/10 completed: Harmony Batch Correction")
    return adata


def lazy_step_scvi(adata, sample_key, output_path=None, **scvi_kwargs):
    """
    Step 6b: scVI batch correction (memory intensive, optional).
    Parameters
    ----------
        adata: AnnData object
        sample_key: batch key for correction
        output_path: path to save the result (optional)
        **scvi_kwargs: additional arguments for scVI
    Returns
    -------
        adata: processed AnnData object
    """
    print("🔧 Step 6b/10: scVI Batch Correction")
    print("⚠️  WARNING: This step is very memory-intensive")

    try:
        import scvi
    except ImportError:
        print("❌ scvi package not found, skipping scVI step")
        return adata

    if "n_layers" not in scvi_kwargs:
        scvi_kwargs["n_layers"] = 2
    if "n_latent" not in scvi_kwargs:
        scvi_kwargs["n_latent"] = 30
    if "gene_likelihood" not in scvi_kwargs:
        scvi_kwargs["gene_likelihood"] = "nb"

    # Use only HVG subset to reduce memory usage
    print("🔧 Processing HVG subset to reduce memory usage...")
    adata_hvg = adata.copy()
    if "highly_variable_features" in adata_hvg.var.columns:
        adata_hvg = adata_hvg[:, adata_hvg.var.highly_variable_features].copy()
    elif "highly_variable" in adata_hvg.var.columns:
        adata_hvg = adata_hvg[:, adata_hvg.var.highly_variable].copy()

    print(
        f"🔧 scVI parameters:\n"
        f"   n_layers: {scvi_kwargs['n_layers']}\n"
        f"   n_latent: {scvi_kwargs['n_latent']}\n"
        f"   gene_likelihood: {scvi_kwargs['gene_likelihood']}\n"
    )

    batch_correction(adata_hvg, batch_key=sample_key, methods="scVI", **scvi_kwargs)

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

    # Clean up to save memory
    del adata_hvg

    # Compute neighbors and embeddings for scVI
    print("🔧 Computing neighbors and embeddings for scVI...")
    neighbors(adata=adata, n_neighbors=15, use_rep="X_scVI", n_pcs=30)
    umap(adata)
    tsne(adata, use_rep="X_scVI")
    adata.obsm["X_umap_scVI"] = adata.obsm["X_umap"]
    adata.obsm["X_tsne_scVI"] = adata.obsm["X_tsne"]

    if output_path:
        print(f"💾 Saving scVI result to {output_path}")
        adata.write(output_path)

    print("✅ Step 6b/10 completed: scVI Batch Correction")
    return adata


def lazy_step_select_best_method(adata, output_path=None):
    """
    Step 7: Select best batch correction method.
    Parameters
    ----------
        adata: AnnData object
        output_path: path to save the result (optional)
    Returns
    -------
        adata: processed AnnData object
    """
    print("🔧 Step 7/10: Select Best Batch Correction Method")

    if "X_scVI" in adata.obsm.keys():
        adata.uns["bench_best_res"] = "X_scVI"
        print("🏆 Selected method: scVI")
    else:
        adata.uns["bench_best_res"] = "X_harmony"
        print("🏆 Selected method: Harmony")

    print(f"Best batch correction method: {adata.uns['bench_best_res']}")

    if output_path:
        print(f"💾 Saving method selection result to {output_path}")
        adata.write(output_path)

    print("✅ Step 7/10 completed: Method Selection")
    return adata


def lazy_step_mde(adata, output_path=None):
    """
    Step 8: MDE embedding computation (memory intensive).
    Parameters
    ----------
        adata: AnnData object
        output_path: path to save the result (optional)
    Returns
    -------
        adata: processed AnnData object
    """
    print("🔧 Step 8/10: MDE Embedding")
    print("⚠️  WARNING: This step is memory-intensive")

    method_test = adata.uns["bench_best_res"]
    print(f"🔧 Using dimensionality reduction: {method_test}")

    # MDE computation - memory intensive!
    mde(
        adata,
        embedding_dim=2,
        n_neighbors=15,
        basis="X_mde",
        n_pcs=30,
        use_rep=adata.uns["bench_best_res"],
    )

    if output_path:
        print(f"💾 Saving MDE result to {output_path}")
        adata.write(output_path)

    print("✅ Step 8/10 completed: MDE Embedding")
    return adata


def lazy_step_clustering(adata, output_path=None, max_iterations=10):
    """
    Step 9: Automated clustering with SCCAF (memory intensive).
    Parameters
    ----------
        adata: AnnData object
        output_path: path to save the result (optional)
        max_iterations: maximum SCCAF iterations
    Returns
    -------
        adata: processed AnnData object
    """
    print("🔧 Step 9/10: Automated Clustering")
    print("⚠️  WARNING: This step is very memory-intensive (SCCAF clustering)")

    # Recompute neighbors for clustering
    neighbors(
        adata=adata, n_neighbors=15, use_rep=adata.uns["bench_best_res"], n_pcs=30
    )

    # Pre-clustering
    print("🔧 Pre-clustering with Leiden...")
    leiden(adata, resolution=1.5, key_added="leiden_r1.5", random_state=0, backend='python')
    adata.obs["L1_result_smooth"] = adata.obs["leiden_r1.5"]

    # Automatic clustering with SCCAF
    print(f"🔧 Starting SCCAF clustering (max {max_iterations} iterations)...")
    for idx in range(max_iterations):
        if (len(np.unique(adata.obs["L1_result_smooth"].tolist())) > 3) and idx > 0:
            print(f"✅ Clustering converged after {idx} iterations")
            break
        else:
            adata.obs["L1_Round0"] = adata.obs["L1_result_smooth"]
            print(f"🔧 SCCAF iteration {idx + 1}/{max_iterations}")

            try:
                SCCAF_optimize_all(
                    min_acc=0.95,
                    ad=adata,
                    classifier="RF",
                    n_jobs=4,
                    use=adata.uns["bench_best_res"],
                    basis="X_mde",
                    method="leiden",
                    prefix="L1",
                    plot=True,
                )

                # Smooth clustering effect
                print(f"🔧 Smoothing clustering results...")
                adata.obs["L1_result_smooth"] = adata.obs["L1_result"].tolist()
            except Exception as e:
                print(f"⚠️  SCCAF clustering failed at iteration {idx + 1}: {e}")
                print("   Using previous clustering result...")
                break

    # Final clustering assignments
    adata.obs["best_clusters"] = adata.obs["L1_result_smooth"].copy()

    # Additional clustering resolutions
    print("🔧 Computing additional clustering resolutions...")
    leiden(adata, resolution=1, key_added="leiden_clusters_L1", random_state=0, backend='python')
    louvain(adata, resolution=1, key_added="louvain_clusters_L1")
    leiden(adata, resolution=0.5, key_added="leiden_clusters_L2", random_state=0, backend='python')
    louvain(adata, resolution=0.5, key_added="louvain_clusters_L2")

    if output_path:
        print(f"💾 Saving clustering result to {output_path}")
        adata.write(output_path)

    print("✅ Step 9/10 completed: Automated Clustering")
    return adata


def lazy_step_final_embeddings(adata, output_path=None):
    """
    Step 10: Final UMAP and tSNE embeddings.
    Parameters
    ----------
        adata: AnnData object
        output_path: path to save the result (optional)
    Returns
    -------
        adata: processed AnnData object
    """
    print("🔧 Step 10/10: Final Embeddings")

    # Final UMAP
    print("🔧 Computing final UMAP...")
    umap(adata)
    adata.obsm["X_umap"] = adata.obsm["X_umap"]

    # Final tSNE
    print("🔧 Computing final tSNE...")
    tsne(adata, use_rep=adata.uns["bench_best_res"])
    adata.obsm["X_tsne"] = adata.obsm["X_tsne"]

    if output_path:
        print(f"💾 Saving final result to {output_path}")
        adata.write(output_path)

    print("✅ Step 10/10 completed: Final Embeddings")
    print("🎉 All processing steps completed!")
    return adata


def lazy_step_by_step_guide():
    """
    Print a guide for step-by-step processing.
    """
    guide = """
    🔧 OmicVerse Lazy Step-by-Step Processing Guide

    For large datasets that cause memory issues, you can run each step individually:

    # Example workflow:
    import omicverse as ov

    # Load your data
    adata = ov.read('your_data.h5ad')

    # Step 1: Quality Control
    adata = ov.single.lazy_step_qc(adata, sample_key='batch',
                                   output_path='step1_qc.h5ad')

    # Step 2: Preprocessing
    adata = ov.single.lazy_step_preprocess(adata,
                                          output_path='step2_preprocess.h5ad')

    # Step 3: Scaling
    adata = ov.single.lazy_step_scale(adata,
                                     output_path='step3_scale.h5ad')

    # Step 4: PCA
    adata = ov.single.lazy_step_pca(adata,
                                   output_path='step4_pca.h5ad')

    # Step 5: Cell cycle
    adata = ov.single.lazy_step_cell_cycle(adata, species='human',
                                          output_path='step5_cell_cycle.h5ad')

    # Step 6a: Harmony batch correction (memory intensive!)
    adata = ov.single.lazy_step_harmony(adata, sample_key='batch',
                                       output_path='step6a_harmony.h5ad')

    # Step 6b: scVI batch correction (optional, very memory intensive!)
    # adata = ov.single.lazy_step_scvi(adata, sample_key='batch',
    #                                 output_path='step6b_scvi.h5ad')

    # Step 7: Select best method
    adata = ov.single.lazy_step_select_best_method(adata,
                                                  output_path='step7_best_method.h5ad')

    # Step 8: MDE embedding (memory intensive!)
    adata = ov.single.lazy_step_mde(adata,
                                   output_path='step8_mde.h5ad')

    # Step 9: Clustering (very memory intensive!)
    adata = ov.single.lazy_step_clustering(adata,
                                          output_path='step9_clustering.h5ad')

    # Step 10: Final embeddings
    adata = ov.single.lazy_step_final_embeddings(adata,
                                                output_path='step10_final.h5ad')

    💡 Tips:
    - Save results after each step to avoid losing progress
    - For memory-intensive steps (6a, 6b, 8, 9), monitor your system memory
    - You can skip scVI (step 6b) if memory is very limited
    - If clustering fails, try reducing max_iterations parameter
    - Between steps, you can restart Python to free memory

    ⚠️  Memory-intensive steps:
    - Step 6a: Harmony batch correction
    - Step 6b: scVI batch correction (optional)
    - Step 8: MDE embedding computation
    - Step 9: SCCAF clustering (most intensive)
    """
    print(guide)

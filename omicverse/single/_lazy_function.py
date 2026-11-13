from ..pp import (
    leiden,
    louvain,
    qc,
    preprocess,
    scale,
    pca,
    score_genes_cell_cycle,
    neighbors,
    umap,
    tsne,
)
import numpy as np

from .._settings import add_reference, settings
from .._registry import register_function


@register_function(
    aliases=['一键单细胞流程', 'lazy', 'lazy scRNA pipeline', 'automatic single-cell pipeline'],
    category="single",
    description="Execute an end-to-end single-cell analysis pipeline with automatic QC, normalization, integration, clustering, and annotation helpers.",
    prerequisites={'optional_functions': ['pp.qc']},
    requires={'layers': ['counts (recommended)']},
    produces={'obsm': ['X_pca', 'X_umap'], 'obsp': ['connectivities'], 'uns': ['neighbors', 'lazy_pipeline']},
    auto_fix='auto',
    examples=['ov.single.lazy(adata, species="human", sample_key="batch")'],
    related=['pp.preprocess', 'single.generate_scRNA_report', 'single.gptcelltype']
)
def lazy(
    adata,
    species="human",
    reforce_steps=[],
    sample_key=None,
    qc_kwargs=None,
    preprocess_kwargs=None,
    pca_kwargs=None,
    harmony_kwargs=None,
    scvi_kwargs=None,
):
    """
    Run a one-click single-cell analysis pipeline with resumable steps.

    Parameters
    ----------
    adata:AnnData
        Input AnnData to be processed.
    species:str
        Species name used by downstream annotation helpers.
    reforce_steps:list
        Step names forced to rerun even if status flags exist. Common values:
        ``['qc', 'pca', 'preprocess', 'scaled', 'Harmony', 'scVI', 'eval_bench', 'eval_clusters']``.
    sample_key:str or None
        Column in ``adata.obs`` used as batch key for correction/integration.
    qc_kwargs:dict or None
        Keyword arguments passed to ``ov.pp.qc``.
    preprocess_kwargs:dict or None
        Keyword arguments passed to ``ov.pp.preprocess``.
    pca_kwargs:dict or None
        Keyword arguments passed to PCA routine.
    harmony_kwargs:dict or None
        Keyword arguments for Harmony integration block.
    scvi_kwargs:dict or None
        Keyword arguments for scVI integration block.

    Returns
    -------
    AnnData
        Updated AnnData with QC, preprocessing, embeddings, and clustering results.

    """
    mode = settings.mode
    print(f"🔧 The mode of lazy is {mode}")
    # step 0: check packages:
    try:
        import louvain
    except:
        print("❌ Louvain package not found, we will install it now")
        import pip

        pip.main(["install", "louvain"])
        import louvain

    try:
        import louvain
    except:
        print("❌ louvain package not found, we will install it now")
        import pip

        pip.main(["install", "louvain"])
        import louvain

    print("✅ All packages used in lazy are installed")

    # step 1: qc:
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata.copy()
    # check adata.uns['status']
    if "status" not in adata.uns.keys():
        adata.uns["status"] = {}

    if (
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
    else:
        print("✅ QC step already finished, skipping it")
        print(
            f"🔧 The argument of qc we set\n"
            f"   mito_perc: {adata.uns['status_args']['qc']['mito_perc']}\n"
            f"   nUMIs: {adata.uns['status_args']['qc']['nUMIs']}\n"
            f"   detected_genes: {adata.uns['status_args']['qc']['detected_genes']}\n"
            f"   doublets_method: {adata.uns['status_args']['qc']['doublets_method']}\n"
            f"   batch_key: {adata.uns['status_args']['qc']['batch_key']}\n"
        )

    # step 2: normalization and highly variable genes:
    if (
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
    else:
        print("✅ Preprocess step already finished, skipping it")
        print(
            f"🔧 The argument of preprocess in data\n"
            f"   mode: {adata.uns['status_args']['preprocess']['mode']}\n"
            f"   n_HVGs: {adata.uns['status_args']['preprocess']['n_HVGs']}\n"
            f"   target_sum: {adata.uns['status_args']['preprocess']['target_sum']}\n"
        )

    if (
        ("scaled" not in adata.uns["status"].keys())
        or (
            "scaled" in adata.uns["status"].keys()
            and adata.uns["status"]["scaled"] == False
        )
        or ("scaled" in reforce_steps)
    ):
        print("❌ Scaled step didn't start, we will start it now")
        scale(adata)
    else:
        print("✅ Scaled step already finished, skipping it")

    # step 3: PCA:
    if (
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
    else:
        print("✅ PCA step already finished, skipping it")

    # step 4 Score cell cycle:
    if (
        ("cell_cycle" not in adata.uns["status"].keys())
        or (
            "cell_cycle" in adata.uns["status"].keys()
            and adata.uns["status"]["cell_cycle"] == False
        )
        or ("cell_cycle" in reforce_steps)
    ):
        print("❌ Cell cycle scoring step didn't start, we will start it now")
        score_genes_cell_cycle(adata, species=species)
    else:
        print("✅ Cell cycle scoring step already finished, skipping it")

    # step 5 batch remove:
    adata_hvg = adata.copy()
    if "highly_variable_features" in adata_hvg.var.columns:
        adata_hvg = adata_hvg[:, adata_hvg.var.highly_variable_features]
    elif "highly_variable" in adata_hvg.var.columns:
        adata_hvg = adata_hvg[:, adata_hvg.var.highly_variable]
    from . import batch_correction

    if ("X_harmony" not in adata.obsm.keys()) or ("Harmony" in reforce_steps):
        print("❌ Batch Correction: `Harmony` step didn't start, we will start it now")
        if harmony_kwargs is None:
            harmony_kwargs = {"n_pcs": 50}
        batch_correction(
            adata_hvg, batch_key=sample_key, methods="harmony", **harmony_kwargs
        )
        adata.obsm["X_harmony"] = adata_hvg.obsm["X_pca_harmony"]
        if "status" not in adata.uns.keys():
            adata.uns["status"] = {}
        if "status_args" not in adata.uns.keys():
            adata.uns["status_args"] = {}

        adata.uns["status"]["harmony"] = True
        adata.uns["status_args"]["harmony"] = {
            "n_pcs": harmony_kwargs["n_pcs"],
        }
        neighbors(adata=adata, n_neighbors=15, use_rep="X_harmony", n_pcs=30)
        umap(adata)
        tsne(adata, use_rep="X_harmony")
        adata.obsm["X_umap_harmony"] = adata.obsm["X_umap"]
        adata.obsm["X_tsne_harmony"] = adata.obsm["X_tsne"]
    else:
        print("✅ Batch Correction: `Harmony` step already finished, skipping it")

    try:
        import scvi

        if ("X_scVI" not in adata.obsm.keys()) or ("scVI" in reforce_steps):
            print("❌ Batch Correction: `scVI` step didn't start, we will start it now")
            if scvi_kwargs is None:
                scvi_kwargs = {"n_layers": 2, "n_latent": 30, "gene_likelihood": "nb"}
            batch_correction(
                adata_hvg, batch_key=sample_key, methods="scVI", **scvi_kwargs
            )
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
            neighbors(adata=adata, n_neighbors=15, use_rep="X_scVI", n_pcs=30)
            umap(adata)
            tsne(adata, use_rep="X_scVI")
            adata.obsm["X_umap_scVI"] = adata.obsm["X_umap"]
            adata.obsm["X_tsne_scVI"] = adata.obsm["X_tsne"]
        else:
            print("✅ Batch Correction: `scVI` step already finished, skipping it")
        del adata_hvg
    except:
        print("❌ scvi package not found, we will not run scVI step")

    if ("bench_best_res" not in adata.uns.keys()) or ("eval_bench" in reforce_steps):
        print("❌ Best Bench Correction Eval step didn't start, we will start it now")
        """
        from scib_metrics.benchmark import Benchmarker

        emb_keys=["X_harmony",'X_scVI']
        bm = Benchmarker(
            adata,
            batch_key=sample_key,
            label_key="phase",
            embedding_obsm_keys=emb_keys,
            pre_integrated_embedding_obsm_key="X_pca",
            n_jobs=-1,
        )
        bm.benchmark()
        bench_res = bm.get_results(min_max_scale=False)
        adata.uns['bench_res']=bench_res.loc[emb_keys]
        adata.uns['bench_best_res']=bench_res.loc[emb_keys,'Batch correction'].sort_values().index[-1]
        for col in adata.uns['bench_res']:
            adata.uns['bench_res'][col]=adata.uns['bench_res'][col].astype(float)
        import matplotlib.pyplot as plt
        bm.plot_results_table(min_max_scale=False,show=False)
        """
        if "X_scVI" in adata.obsm.keys():
            adata.uns["bench_best_res"] = "X_scVI"
        else:
            adata.uns["bench_best_res"] = "X_harmony"
        print(f"The Best Bench Correction Method is {adata.uns['bench_best_res']}")
        print("We can found it in `adata.uns['bench_best_res']`")
    else:
        print("✅ Best Bench Correction Eval step already finished, skipping it")

    # step 6 clusters:
    if ("best_clusters" not in adata.obs.columns) or ("eval_clusters" in reforce_steps):
        print("❌ Best Clusters step didn't start, we will start it now")
        method_test = adata.uns["bench_best_res"]
        print(f"Automatic clustering using sccaf")
        print(f"Dimensionality using :{method_test}")
        neighbors(
            adata=adata, n_neighbors=15, use_rep=adata.uns["bench_best_res"], n_pcs=30
        )
        umap(adata)
        # 预聚类
        print(f"Automatic clustering using leiden for preprocessed")
        leiden(adata, resolution=1.5, key_added="leiden_r1.5", random_state=0, backend='python')
        # self.adata.obs['L1_Round0'] = self.adata.obs['leiden_r1.5']
        adata.obs["L1_result_smooth"] = adata.obs["leiden_r1.5"]
        # 自动聚类
        for idx in range(10):
            if (len(np.unique(adata.obs["L1_result_smooth"].tolist())) > 3) and idx > 0:
                break
            else:
                adata.obs["L1_Round0"] = adata.obs["L1_result_smooth"]
                print(f"Automatic clustering using sccaf, Times: {idx}")
                from . import SCCAF_optimize_all

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
                # 平滑聚类效果
                print(f"Smoothing the effect of clustering, Times: {idx}")
                adata.obs["L1_result_smooth"] = adata.obs["L1_result"].tolist()

        # 获取最佳聚类
        adata.obs["best_clusters"] = adata.obs["L1_result_smooth"].copy()
        leiden(adata, resolution=1, key_added="leiden_clusters_L1", random_state=0, backend='python')
        louvain(adata, resolution=1, key_added="louvain_clusters_L1")
        leiden(adata, resolution=0.5, key_added="leiden_clusters_L2", random_state=0, backend='python')
        louvain(adata, resolution=0.5, key_added="louvain_clusters_L2")
    else:
        print("✅ Best Clusters step already finished, skipping it")

    # step 7 UMAP:
    if ("X_umap" not in adata.obsm.keys()) or ("umap" in reforce_steps):
        print("❌ UMAP step didn't start, we will start it now")
        umap(adata)
        adata.obsm["X_umap"] = adata.obsm["X_umap"]
    else:
        print("✅ UMAP step already finished, skipping it")

    # step 8 tsne:
    if ("X_tsne" not in adata.obsm.keys()) or ("tsne" in reforce_steps):
        print("❌ tSNE step didn't start, we will start it now")
        tsne(adata, use_rep=adata.uns["bench_best_res"])
        adata.obsm["X_tsne"] = adata.obsm["X_tsne"]
    else:
        print("✅ tSNE step already finished, skipping it")

    return adata

    # step 7 anno celltype automatically:
    #step 8 tsne:
    if ('X_tsne' not in adata.obsm.keys()) or ('tsne' in reforce_steps):
        print('❌ tSNE step didn\'t start, we will start it now')
        tsne(adata,use_rep=adata.uns['bench_best_res'])
        adata.obsm['X_tsne']=adata.obsm['X_tsne']
    else:
        print('✅ tSNE step already finished, skipping it')

    return adata

    #step 7 anno celltype automatically:

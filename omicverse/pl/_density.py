import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from .._registry import register_function
from ._plotdata import get_values



@register_function(
    aliases=["基因密度计算", "calculate_gene_density", "gene_density", "密度计算", "表达密度"],
    category="pl",
    description="Calculate weighted kernel density estimates for gene expression on 2D embeddings",
    examples=[
        "# Basic gene density calculation",
        "ov.pl.calculate_gene_density(adata, features=['CD3D', 'CD8A'])",
        "# Custom parameters",
        "ov.pl.calculate_gene_density(adata, features=['marker_gene'],",
        "                             basis='X_tsne', adjust=0.5)",
        "# Multiple genes with threshold",
        "ov.pl.calculate_gene_density(adata, features=marker_genes,",
        "                             min_expr=0.2, adjust=1.5)"
    ],
    related=["pl.embedding_density", "pl.add_density_contour", "pl.embedding"]
)
def calculate_gene_density(
    adata,
    features,
    basis="X_umap",
    dims=(0, 1),
    adjust=1,
    min_expr=0.1,          # NEW: minimal raw expression to keep as weight > 0
    layer=None,
    use_raw=None,
):
    r"""
    Calculate weighted kernel density estimates for gene expression on 2D embeddings.
    
    Computes KDE for each feature using expression values as weights and stores
    density values in adata.obs as 'density_{feature}' columns.
    
    Args:
        adata: Annotated data object with embedding coordinates
        features: List of gene names or feature names to process
        basis: Key in adata.obsm containing 2D embedding coordinates ('X_umap')
        dims: Embedding dimensions to use as (x_dim, y_dim) ((0, 1))
        adjust: Bandwidth scaling factor for KDE (1)
        min_expr: Minimum expression threshold for including cells (0.1)
        layer: Read expression from `adata.layers[layer]` instead of `.X` (None)
        use_raw: Force (`True`) or forbid (`False`) reading from `adata.raw`.
            `None` reads `.X` and falls back to `.raw` only for names that are
            absent from `.var_names` — see `ov.pl.get_values` (None)

    Returns:
        None: Updates adata.obs with density_{feature} columns
    """
    if len(dims) != 2:
        raise ValueError("`dims` must have length 2")
    if basis not in adata.obsm:
        raise ValueError(f"Embedding '{basis}' not found.")

    emb_all = adata.obsm[basis][:, dims]          # (n_cells, 2)

    for feat in features:
        # ----- fetch raw weights -------------------------------------------
        w_raw = get_values(adata, feat, layer=layer, use_raw=use_raw)

        # ----- validity mask: finite coords & finite expr -------------------
        mask_finite = np.isfinite(w_raw) & np.all(np.isfinite(emb_all), axis=1)

        # ----- NEW: expression threshold -----------------------------------
        mask_expr   = w_raw > min_expr
        mask_train  = mask_finite & mask_expr

        emb_train   = emb_all[mask_train]
        w_train_raw = w_raw[mask_train]

        if emb_train.shape[0] < 5:
            print(f"[{feat}] too few cells above threshold; skipping KDE.")
            adata.obs[f"density_{feat}"] = np.nan
            continue

        # ----- min–max scale to 0-1 ----------------------------------------
        w_min, w_max = w_train_raw.min(), w_train_raw.max()
        if w_max == w_min:
            # every selected cell carries the same value, so the min-max scale
            # is 0/0. The limiting case is an unweighted KDE of those cells —
            # which is the honest answer, not a matrix full of NaN handed to
            # scipy to fail on several frames later.
            print(f"[{feat}] constant across the selected cells; "
                  f"using an unweighted KDE.")
            w_train = np.ones_like(w_train_raw, dtype=float)
        else:
            w_train = (w_train_raw - w_min) / (w_max - w_min)

        # ----- KDE fit ------------------------------------------------------
        kde = gaussian_kde(emb_train.T, weights=w_train, bw_method=adjust)

        # ----- evaluate on ALL cells ---------------------------------------
        density = kde(emb_all.T)
        adata.obs[f"density_{feat}"] = density

        print(f"✅ density_{feat} written (train cells = {emb_train.shape[0]})")


import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt

@register_function(
    aliases=["添加密度等高线", "add_density_contour", "density_contour", "密度等高线", "等高线添加"],
    category="pl",
    description="Add KDE-based density contours to existing matplotlib plot",
    examples=[
        "# Basic density contour",
        "import matplotlib.pyplot as plt",
        "fig, ax = plt.subplots()",
        "ax.scatter(embeddings[:, 0], embeddings[:, 1])",
        "ov.pl.add_density_contour(ax, embeddings, weights)",
        "# Customized contours",
        "ov.pl.add_density_contour(ax, embeddings, expression_values,",
        "                          levels='quantile', n_quantiles=10,",
        "                          cmap_contour='Blues', fill=True)",
        "# With embedding plot",
        "ax = ov.pl.embedding(adata, basis='X_umap', color='cell_type', show=False)",
        "ov.pl.add_density_contour(ax, adata.obsm['X_umap'], adata.obs['marker_gene'])"
    ],
    related=["pl.calculate_gene_density", "pl.embedding_density", "pl.embedding"]
)
def add_density_contour(
    ax,
    embeddings,              # (n_cells, 2) array
    weights,                 # 1-D array, will be min-max scaled
    levels="quantile",       # "quantile" or a numeric list, see below
    n_quantiles=5,
    bw_adjust=0.3,
    cmap_contour="Greys",
    linewidth=1.0,
    zorder=10,
    fill=False,
    alpha=0.4,
):
    r"""
    Add KDE-based density contours to an existing matplotlib plot.
    
    Args:
        ax: matplotlib.axes.Axes object to draw contours on
        embeddings: 2D coordinate array with shape (n_cells, 2)
        weights: 1D weight array for KDE, will be min-max normalized
        levels: Contour level specification - 'quantile' or list of values ('quantile')
        n_quantiles: Number of quantile levels when levels='quantile' (5)
        bw_adjust: Bandwidth adjustment factor for KDE (0.3)
        cmap_contour: Colormap for contour lines ('Greys')
        linewidth: Width of contour lines (1.0)
        zorder: Drawing order for contours (10)
        fill: Whether to fill contours (False for lines, True for filled)
        alpha: Transparency for filled contours (0.4)
        
    Returns:
        cs: matplotlib contour object for potential colorbar addition
    """
    # ---------- fit KDE ----------------------------------------------------
    w_min, w_max = weights.min(), weights.max()
    w_norm = None if w_max == w_min else (weights - w_min) / (w_max - w_min)
    kde = gaussian_kde(embeddings.T, weights=w_norm, bw_method=bw_adjust)

    # ---------- prepare evaluation grid -----------------------------------
    xmin, xmax = embeddings[:, 0].min(), embeddings[:, 0].max()
    ymin, ymax = embeddings[:, 1].min(), embeddings[:, 1].max()
    xx, yy = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]   # 300×300 grid
    grid = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    # ---------- determine contour levels ----------------------------------
    if levels == "quantile":
        qs = np.linspace(0, 1, n_quantiles + 2)[1:-1]   # drop 0 & 1
        levels = np.quantile(grid, qs)

    # ---------- draw -------------------------------------------------------
    if fill:
        cs = ax.contourf(
            xx, yy, grid,
            levels=levels,
            cmap=cmap_contour,
            alpha=alpha,
            zorder=zorder,
        )
    else:
        cs = ax.contour(
            xx, yy, grid,
            levels=levels,
            cmap=cmap_contour,
            linewidths=linewidth,
            zorder=zorder,
        )
    return cs   # so you can add a colorbar if desired

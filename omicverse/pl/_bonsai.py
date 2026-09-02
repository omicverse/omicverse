r"""``ov.pl.bonsai`` — draw the Bonsai tree computed by :func:`omicverse.tl.bonsai`.

Bonsai returns a tree, not coordinates, so the layout is computed at draw time by
:func:`omicverse.external.bonsai.equal_angle_layout`. Branch lengths survive the
layout, which is the point: distance along the drawing is distance in the model,
unlike a UMAP where the gap between two clusters carries no meaning.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from .._registry import register_function

__all__ = ["bonsai"]


@register_function(
    aliases=["bonsai plot", "Bonsai树图", "plot bonsai", "画Bonsai树"],
    category="pl",
    description="Draw the Bonsai tree from ov.tl.bonsai, colouring leaves by an "
                "obs column.",
    examples=["ov.pl.bonsai(adata, color='leiden')"],
    related=["ov.tl.bonsai", "ov.pl.embedding"],
    requires={"uns": ["bonsai"]},
)
def bonsai(
    adata,
    color: Optional[str] = None,
    *,
    key: str = "bonsai",
    style: str = "scout",
    layer: Optional[str] = None,
    use_raw: Optional[bool] = None,
    gene_symbols: Optional[str] = None,
    ax=None,
    figsize=(4.6, 4.6),
    radius: Optional[float] = None,
    edge_color: str = "#b4b4b4",
    edge_width: float = 0.55,
    edge_style: str = "--",
    disk: bool = True,
    spread: float = 0.8,
    cache_layout: bool = True,
    palette: Optional[Sequence[str]] = None,
    cmap: Optional[str] = None,
    color_map: Optional[str] = None,
    na_color: str = "lightgray",
    title: Optional[str] = None,
    legend_loc: Optional[str] = "right margin",
    legend_fontsize: Optional[float] = None,
    legend_fontweight: str = "bold",
    legend_fontoutline: Optional[float] = None,
    colorbar_loc: Optional[str] = "right",
    frameon: Union[bool, str] = "small",
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
):
    r"""Draw the Bonsai tree stored in ``adata.uns[key]``.

    Every cell is a leaf and every internal vertex an inferred ancestral state.
    Branches are drawn at their inferred lengths, so a long branch means a cell
    whose state really is far from its neighbours rather than one that a
    projection happened to push aside.

    Parameters
    ----------
    adata
        Annotated data matrix carrying ``adata.uns[key]`` from
        :func:`omicverse.tl.bonsai`.
    color
        What to colour the leaves by: a column of ``adata.obs`` **or a gene**,
        resolved exactly as :func:`omicverse.pl.embedding` resolves it (obs, then
        ``var_names``, then ``raw``). Categorical columns
        take ``palette`` (default :data:`omicverse.pl.sc_color`), numeric columns
        take ``cmap`` and get a colourbar. ``None`` draws the bare tree.
    key
        Key under ``adata.uns`` written by :func:`omicverse.tl.bonsai`.
    layer, use_raw, gene_symbols
        Where to read a gene from, as in :func:`omicverse.pl.embedding`.
    ax
        Existing axes to draw into; a new figure is created when omitted.
    style
        ``'scout'`` reproduces Bonsai-scout's own view: equal-angle then
        equal-daylight layout, mapped into the Poincaré disk. ``'equal_angle'``
        is the plain Euclidean fallback.
    figsize, edge_color, edge_width, edge_style
        Figure size, and the colour, width and dash of the branches.
    radius
        Leaf marker radius, **in data units** — not ``ov.pl.embedding``'s
        ``size``, which is a marker area in points². The difference is
        deliberate: drawing the cells at a fixed radius in the tree's own
        coordinates is what makes a dense clade read as one blob and keeps it
        that way when the figure is resized. Defaults to upstream's ``0.015``.
    disk
        Draw the bounding circle. Only meaningful for ``style='scout'``, where
        the circle is the boundary of the Poincaré disk.
    cache_layout
        Keep the computed coordinates in ``adata.uns[key]['layout']`` and reuse
        them when ``style`` and ``spread`` are unchanged, so plotting the same
        tree by a different ``color`` is instant.
    spread
        How far the tree reaches towards the rim: the zoom is chosen so this
        fraction of vertices sits inside that same radius. ``0.8`` is upstream's
        framing; raise it to fill more of the disk, lower it to pull in.
    palette
        Colours for a categorical ``color``.
    cmap, color_map
        Colormap for a numeric ``color``; ``ov.pl.embedding`` accepts both names,
        so both are taken here. Defaults to ``'RdBu_r'``.
    na_color
        Colour for a category with no palette entry.
    title
        Axes title; defaults to ``color``.
    legend_loc, legend_fontsize, legend_fontweight, legend_fontoutline
        As in :func:`omicverse.pl.embedding` — ``'right margin'`` (default),
        ``'on data'``, or ``'none'``/``None``. Placement, the column rule and the
        outline handling follow that function so the two look like one family.
    colorbar_loc
        Where the colourbar goes for a numeric ``color``; ``None`` omits it.
    frameon
        ``False`` strips the axes entirely; ``'small'`` matches the reduced frame
        :func:`omicverse.pl.embedding` uses.
    show
        Call :func:`matplotlib.pyplot.show` and return ``None``. Defaults to
        ``True`` only when ``ax`` was not supplied.
    save
        Path to write the figure to, or ``True`` for ``'bonsai.png'``.

    Returns
    -------
    :class:`matplotlib.axes.Axes` or None
        The axes, unless ``show`` resolved to ``True``.

    Examples
    --------
    >>> import omicverse as ov
    >>> ov.tl.bonsai(adata, use_rep='scaled|original|X_pca')
    >>> ov.pl.bonsai(adata, color='leiden')

    Draw it beside a UMAP to compare what each representation preserves:

    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    >>> ov.pl.umap(adata, color='leiden', ax=axes[0], show=False)
    >>> ov.pl.bonsai(adata, color='leiden', ax=axes[1], show=False)
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import EllipseCollection, LineCollection

    from ..external.bonsai import equal_angle_layout, scout_layout

    if key not in adata.uns:
        raise KeyError(
            f"adata.uns['{key}'] not found — run ov.tl.bonsai(adata, ...) first."
        )
    res = adata.uns[key]

    # adata.uns travels through slicing untouched, but leaf_of_obs is indexed by
    # cell position -- so on a subset it silently refers to the wrong cells. The
    # raw failure is an IndexError from a mismatched boolean mask, which says
    # nothing about the cause.
    n_map = len(np.asarray(res.get("leaf_of_obs", ())))
    if n_map != adata.n_obs:
        raise ValueError(
            f"adata.uns['{key}'] was computed for {n_map} cells but this AnnData "
            f"has {adata.n_obs}. The tree does not survive subsetting -- slice "
            f"first, then run ov.tl.bonsai on the subset."
        )

    edges = np.asarray(res["edges"], dtype=int)
    lengths = np.asarray(res["edge_lengths"], dtype=float)
    n_vert = int(max(int(edges.max()), int(np.asarray(res["vert_ind"]).max()))) + 1

    if style not in ("scout", "equal_angle"):
        raise ValueError(f"style must be 'scout' or 'equal_angle', got {style!r}")

    # Cache the layout on the AnnData. The daylight pass costs ~3 s at 1600
    # vertices and ~31 s at 12800, so recomputing it for every `color` a user
    # tries would dominate; a backbone-scale tree makes that unusable. The
    # signature keys the cache to the parameters that change the coordinates.
    sig = (style, float(spread), int(n_vert), int(len(edges)))
    cached = res.get("layout") if isinstance(res, dict) else None
    if cache_layout and cached is not None and tuple(cached.get("sig", ())) == sig:
        pos = np.asarray(cached["pos"], dtype=float)
    else:
        if style == "scout":
            pos = scout_layout(edges, lengths, n_vert,
                               frac_within=spread, within_radius=spread)[0]
        else:
            pos = equal_angle_layout(edges, lengths, n_vert)[0]
        if cache_layout and isinstance(res, dict):
            res["layout"] = {"sig": sig, "pos": pos}

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show = True if show is None else show
    else:
        fig = ax.figure
        show = False if show is None else show

    # Upstream's own styling (bonsai_scout_helpers.py): a white disk bounded by a
    # grey circle, dashed grey branches behind it, and cells drawn as circles of a
    # fixed radius in DATA units -- that last part is what makes dense clades read
    # as solid blobs instead of dissolving when the figure is resized.
    span = float(np.abs(pos).max()) or 1.0
    if disk and style == "scout":
        ax.add_patch(plt.Circle((0, 0), 1.0, facecolor="white",
                                edgecolor="#a9a9a9", linewidth=1.1, zorder=-3))
        lim = 1.06
    else:
        lim = span * 1.06

    ax.add_collection(LineCollection(
        [[pos[a_], pos[b_]] for a_, b_ in edges],
        colors=edge_color, linewidths=edge_width,
        linestyles=edge_style, zorder=1))

    leaf_of_obs = np.asarray(res["leaf_of_obs"], dtype=int)
    keep = leaf_of_obs >= 0
    pts = pos[leaf_of_obs[keep]]
    radius = ((0.015 if style == "scout" else 0.015 * span)
              if radius is None else radius)

    def _blobs(xy, colors, **kw):
        """Cells as data-unit circles, so their size means the same everywhere."""
        coll = EllipseCollection(
            widths=2 * radius, heights=2 * radius, angles=0, units="xy",
            offsets=xy, offset_transform=ax.transData,
            facecolors=colors, edgecolors="#16161e", linewidths=0.35,
            zorder=5, **kw)
        ax.add_collection(coll)
        return coll

    if color is None:
        _blobs(pts, "#7d7d85")
    else:
        # ov.pl.embedding resolves `color` against obs, then var, then raw, and
        # honours layer/use_raw/gene_symbols. Reusing its resolver is what makes
        # `color='CST3'` work here exactly as it does there, instead of only
        # accepting obs columns.
        from ._scatterplot_backend import _get_color_source_vector

        vals = _get_color_source_vector(
            adata, color,
            use_raw=(False if use_raw is None else use_raw),
            gene_symbols=gene_symbols, layer=layer)
        if vals is None:
            raise KeyError(
                f"color='{color}' is neither a column of adata.obs nor a gene "
                "in adata.var_names")
        vals = np.asarray(vals)[keep]
        is_cat = hasattr(vals, "categories") or getattr(vals, "dtype", None) == object
        if is_cat:
            import pandas as pd
            from matplotlib.lines import Line2D
            cats = list(pd.Categorical(vals).categories)
            if palette is None:
                from ._palette import sc_color
                palette = sc_color
            cols = {c: palette[i % len(palette)] for i, c in enumerate(cats)}
            _blobs(pts, [cols.get(v, na_color) for v in vals])
            if legend_loc in (None, "none"):
                pass
            elif legend_loc == "right margin":
                # Same placement and column rule as ov.pl.embedding, so the two
                # read as one family. EllipseCollection carries no label, hence
                # the empty proxy scatters.
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.91, box.height])
                for c in cats:
                    ax.scatter([], [], c=cols.get(c, na_color), label=c)
                ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5),
                          ncol=(1 if len(cats) <= 14 else 2 if len(cats) <= 30 else 3),
                          fontsize=legend_fontsize)
            elif legend_loc == "on data":
                from matplotlib import patheffects
                fx = ([patheffects.withStroke(linewidth=legend_fontoutline,
                                              foreground="white")]
                      if legend_fontoutline is not None else None)
                for c in cats:
                    m = np.asarray(vals == c)
                    if m.any():
                        # Median, not mean: one straggler on a long branch would
                        # otherwise drag the label off its own clade.
                        ax.text(np.median(pts[m, 0]), np.median(pts[m, 1]), str(c),
                                weight=legend_fontweight, ha="center", va="center",
                                fontsize=legend_fontsize, path_effects=fx)
        else:
            v = np.asarray(vals, dtype=float)
            coll = _blobs(pts, None, array=v, cmap=(cmap or color_map or "RdBu_r"))
            if colorbar_loc is not None:
                fig.colorbar(coll, ax=ax, shrink=0.6, label=color,
                             location=colorbar_loc)

    ax.set_title(title if title is not None else (color or "Bonsai"), fontsize=13)
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    if frameon is False or (frameon == "small" and style == "scout"):
        # The disk is the frame; axes on top of it are noise.
        ax.set_axis_off()
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if frameon == "small":
            for s in ("left", "bottom"):
                ax.spines[s].set_visible(False)

    if save:
        plt.savefig(save if isinstance(save, str) else "bonsai.png",
                    bbox_inches="tight")
    if show:
        plt.show()
        return None
    return ax

from ._scatterplot_backend import _embedding
from .._registry import register_function
from ._plotdata import get_values
from ._plotdata import accepts_frame
from ._plot_backend import style_axes
import collections.abc as cabc
from copy import copy
from numbers import Integral
from itertools import combinations, product
import matplotlib
from typing import (
    Collection,
    Union,
    Optional,
    Sequence,
    Any,
    Mapping,
    List,
    Tuple,
    Literal,
)
from warnings import warn

import matplotlib.patches as mpatches
from scipy.stats import kruskal
import numpy as np
import pandas as pd
from anndata import AnnData
from cycler import Cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas.api.types import is_categorical_dtype
from matplotlib import pyplot as pl, colors, colormaps
from matplotlib import rcParams
from matplotlib import patheffects
from matplotlib.colors import Colormap, Normalize
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns

from ._scanpy_compat import (
    _FontWeight,
    _FontSize,
    ColorLike,
    VBound,
    default_palette,
    embedding_density as compute_embedding_density,
)

def mde(adata: AnnData,convert=True, **kwargs):
    r"""
    Plot MDE embedding.

    Args:
        adata: Annotated data matrix.
        **kwargs: Additional keyword arguments.

    Returns:
        fig: figure and axis
        ax: axis
    """
    if 'X_mde' not in adata.obsm.keys():
        raise ValueError('X_mde not found in adata.obsm. Please run ov.pp.mde first.')
    if convert:
        adata.obsm['MDE'] = adata.obsm['X_mde']
        print(f"X_mde converted to MDE to visualize and saved to adata.obsm['MDE']")
        print(f"if you want to use X_mde, please set convert=False")
    else:
        print(f"if you want to use MDE to visualize, please set convert=True")
    return embedding(adata, basis='MDE', **kwargs)

@register_function(
    aliases=["tsne图", "tsne", "tSNE可视化", "tsne_plot", "t-SNE"],
    category="pl",
    description="Plot t-SNE embedding with omicverse styling",
    examples=[
        "# Basic t-SNE plot colored by cell type",
        "ov.pl.tsne(adata, color='leiden')",
        "# Multiple colors",
        "ov.pl.tsne(adata, color=['leiden', 'cell_type'], ncols=2)",
        "# Customize point size and colormap",
        "ov.pl.tsne(adata, color='leiden', size=10, frameon=False)",
    ],
    related=["pl.umap", "pl.embedding", "pl.mde", "pp.tsne"]
)
def tsne(adata: AnnData,convert=True, **kwargs):
    r"""
    Plot t-SNE embedding.

    Args:
        adata: Annotated data matrix.
        **kwargs: Additional keyword arguments.

    Returns:
        fig: figure and axis
        ax: axis
    """
    if 'X_tsne' not in adata.obsm.keys():
        raise ValueError('X_tsne not found in adata.obsm. Please run ov.pp.tsne first.')
    if convert:
        adata.obsm['TSNE'] = adata.obsm['X_tsne']
        print(f"X_tsne converted to TSNE to visualize and saved to adata.obsm['TSNE']")
        print(f"if you want to use X_tsne, please set convert=False")
    else:
        print(f"if you want to use TSNE to visualize, please set convert=True")
    return embedding(adata, basis='TSNE', **kwargs)

def pca(adata: AnnData,convert=True, **kwargs):
    r"""
    Plot PCA embedding.

    Args:
        adata: Annotated data matrix.
        **kwargs: Additional keyword arguments.
    """
    if 'X_pca' not in adata.obsm.keys() and 'scaled|original|X_pca' not in adata.obsm.keys():
        raise ValueError('X_pca not found in adata.obsm. Please run ov.pp.pca first.')
    if 'scaled|original|X_pca' in adata.obsm.keys():
        adata.obsm['X_pca'] = adata.obsm['scaled|original|X_pca']
    if convert:
        adata.obsm['PCA'] = adata.obsm['X_pca']
        print(f"X_pca converted to PCA to visualize and saved to adata.obsm['PCA']")
        print(f"if you want to use X_pca, please set convert=False")
    else:
        print(f"if you want to use PCA to visualize, please set convert=True")
    return embedding(adata, basis='PCA', **kwargs)

@register_function(
    aliases=["umap图", "umap", "UMAP可视化", "umap_plot", "UMAP"],
    category="pl",
    description="Plot UMAP embedding with omicverse styling",
    examples=[
        "# Basic UMAP plot colored by cell type",
        "ov.pl.umap(adata, color='leiden')",
        "# Multiple colors side by side",
        "ov.pl.umap(adata, color=['leiden', 'cell_type'], ncols=2)",
        "# Customize point size and remove frame",
        "ov.pl.umap(adata, color='leiden', size=5, frameon=False)",
    ],
    related=["pl.tsne", "pl.embedding", "pl.mde", "pp.umap"]
)
def umap(adata: AnnData,convert=True, **kwargs):
    r"""
    Plot UMAP embedding.

    Args:
        adata: Annotated data matrix.
        color: Keys for annotations of observations/cells or variables/genes. (None)
        gene_symbols: Key for field in `.var` that stores gene symbols. (None)
        use_raw: Use `.raw` attribute of `adata` if present. (None)
        sort_order: For continuous annotations used as color parameter, plot data points with higher values on top of others. (True)
        edges: Show edges between cells. (False)
        edges_width: Width of edges. (0.1)
        edges_color: Color of edges. ('grey')
        neighbors_key: Key to use for neighbors. (None)
        arrows: Show arrows for velocity. (False)
        arrows_kwds: Keyword arguments for arrow plots. (None)
        groups: Groups to highlight. (None)
        components: Components to plot. (None)
        dimensions: Dimensions to plot. (None)
        layer: Name of the layer to use for coloring. (None)
        projection: Type of projection ('2d' or '3d'). ('2d')
        scale_factor: Scaling factor for sizes. (None)
        color_map: Colormap to use for continuous variables. (None)
        cmap: Colormap to use for continuous variables. (None)
        palette: Colors to use for categorical variables. (None)
        na_color: Color to use for NaN values. ('lightgray')
        na_in_legend: Include NaN values in legend. (True)
        size: Size of the dots. (None)
        frameon: Draw a frame around the plot. ('small')
        legend_fontsize: Font size for legend. (None)
        legend_fontweight: Font weight for legend. ('bold')
        legend_loc: Location of legend. ('right margin')
        legend_fontoutline: Outline width for legend text. (None)
        colorbar_loc: Location of colorbar. ('right')
        vmax: Maximum value for colorbar. (None)
        vmin: Minimum value for colorbar. (None)
        vcenter: Center value for colorbar. (None)
        norm: Normalization for colorbar. (None)
        add_outline: Add outline to points. (False)
        outline_width: Width of outline. ((0.3, 0.05))
        outline_color: Color of outline. (('black', 'white'))
        ncols: Number of columns for subplots. (4)
        hspace: Height spacing between subplots. (0.25)
        wspace: Width spacing between subplots. (None)
        title: Title for the plot. (None)
        show: Show the plot. (None)
        save: Save the plot. (None)
        ax: Matplotlib axes object. (None)
        return_fig: Return figure object. (None)
        marker: Marker style. ('.')
        **kwargs: Additional keyword arguments.

    Returns:
        fig: figure and axis
        ax: axis
    """
    if 'X_umap' not in adata.obsm.keys():
        raise ValueError('X_umap not found in adata.obsm. Please run ov.pp.umap first.')
    if convert:
        adata.obsm['UMAP'] = adata.obsm['X_umap']
        print(f"X_umap converted to UMAP to visualize and saved to adata.obsm['UMAP']")
        print(f"if you want to use X_umap, please set convert=False")
    else:
        print(f"if you want to use UMAP to visualize, please set convert=True")
    return embedding(adata, basis='UMAP', **kwargs)

@register_function(
    aliases=["细胞嵌入图", "embedding", "scatter_plot", "降维可视化", "嵌入图"],
    category="pl",
    description="Scatter plot for user specified embedding basis (UMAP, t-SNE, PCA, etc)",
    examples=[
        "# Basic UMAP plot",
        "ov.pl.embedding(adata, basis='X_umap', color='cell_type')",
        "# Multiple genes visualization",
        "ov.pl.embedding(adata, basis='X_umap', color=['CD3D', 'CD8A', 'CD4'])",
        "# Customized embedding plot",
        "ov.pl.embedding(adata, basis='X_tsne', color='leiden',",
        "                palette=ov.pl.red_color, frameon='small')",
        "# 3D embedding",
        "ov.pl.embedding(adata, basis='X_umap', color='cell_type',",
        "                projection='3d', ncols=2)"
    ],
    related=["pp.umap", "pp.tsne", "pp.pca", "pl.umap", "pl.tsne", "pl.pca"]
)
def embedding(
    adata: AnnData,
    basis: str,
    *,
    color: Union[str, Sequence[str], None] = None,
    gene_symbols: Optional[str] = None,
    use_raw: Optional[bool] = None,
    sort_order: bool = True,
    edges: bool = False,
    edges_width: float = 0.1,
    edges_color: Union[str, Sequence[float], Sequence[str]] = 'grey',
    neighbors_key: Optional[str] = None,
    arrows: bool = False,
    arrows_kwds: Optional[Mapping[str, Any]] = None,
    groups: Optional[str] = None,
    components: Union[str, Sequence[str]] = None,
    dimensions: Optional[Union[Tuple[int, int], Sequence[Tuple[int, int]]]] = None,
    layer: Optional[str] = None,
    projection: Literal['2d', '3d'] = '2d',
    scale_factor: Optional[float] = None,
    color_map: Union[Colormap, str, None] = None,
    cmap: Union[Colormap, str, None] = None,
    palette: Union[str, Sequence[str], Cycler, None] = None,
    na_color: ColorLike = "lightgray",
    na_in_legend: bool = True,
    size: Union[float, Sequence[float], None] = None,
    frameon: Optional[bool] = 'small',
    legend_fontsize: Union[int, float, _FontSize, None] = None,
    legend_fontweight: Union[int, _FontWeight] = 'bold',
    legend_loc: str = 'right margin',
    legend_fontoutline: Optional[int] = None,
    colorbar_loc: Optional[str] = "right",
    vmax: Union[VBound, Sequence[VBound], None] = None,
    vmin: Union[VBound, Sequence[VBound], None] = None,
    vcenter: Union[VBound, Sequence[VBound], None] = None,
    norm: Union[Normalize, Sequence[Normalize], None] = None,
    add_outline: Optional[bool] = False,
    outline_width: Tuple[float, float] = (0.3, 0.05),
    outline_color: Tuple[str, str] = ('black', 'white'),
    ncols: int = 4,
    hspace: float = 0.25,
    wspace: Optional[float] = None,
    title: Union[str, Sequence[str], None] = None,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    ax: Optional[Axes] = None,
    return_fig: Optional[bool] = None,
    marker: Union[str, Sequence[str]] = '.',
    **kwargs,
) -> Union[Figure, Axes, None]:
    r"""
    Scatter plot for user specified embedding basis (e.g. umap, pca, etc).


    Args:
        adata: Annotated data matrix.
        basis: Name of the `obsm` basis to use.
        color: Keys for annotations of observations/cells or variables/genes. (None)
        gene_symbols: Key for field in `.var` that stores gene symbols. (None)
        use_raw: Use `.raw` attribute of `adata` if present. (None)
        sort_order: For continuous annotations used as color parameter, plot data points with higher values on top of others. (True)
        edges: Show edges between cells. (False)
        edges_width: Width of edges. (0.1)
        edges_color: Color of edges. ('grey')
        neighbors_key: Key to use for neighbors. (None)
        arrows: Show arrows for velocity. (False)
        arrows_kwds: Keyword arguments for arrow plots. (None)
        groups: Groups to highlight. (None)
        components: Components to plot. (None)
        dimensions: Dimensions to plot. (None)
        layer: Name of the layer to use for coloring. (None)
        projection: Type of projection ('2d' or '3d'). ('2d')
        scale_factor: Scaling factor for sizes. (None)
        color_map: Colormap to use for continuous variables. (None)
        cmap: Colormap to use for continuous variables. (None)
        palette: Colors to use for categorical variables. (None)
        na_color: Color to use for NaN values. ('lightgray')
        na_in_legend: Include NaN values in legend. (True)
        size: Size of the dots. (None)
        frameon: Draw a frame around the plot. ('small')
        legend_fontsize: Font size for legend. (None)
        legend_fontweight: Font weight for legend. ('bold')
        legend_loc: Location of legend. ('right margin')
        legend_fontoutline: Outline width for legend text. (None)
        colorbar_loc: Location of colorbar. ('right')
        vmax: Maximum value for colorbar. (None)
        vmin: Minimum value for colorbar. (None)
        vcenter: Center value for colorbar. (None)
        norm: Normalization for colorbar. (None)
        add_outline: Add outline to points. (False)
        outline_width: Width of outline. ((0.3, 0.05))
        outline_color: Color of outline. (('black', 'white'))
        ncols: Number of columns for subplots. (4)
        hspace: Height spacing between subplots. (0.25)
        wspace: Width spacing between subplots. (None)
        title: Title for the plot. (None)
        show: Show the plot. (None)
        save: Save the plot. (None)
        ax: Matplotlib axes object. (None)
        return_fig: Return figure object. (None)
        marker: Marker style. ('.')
        **kwargs: Additional keyword arguments.


    Returns:
        ax: If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.
    """

    return _embedding(adata=adata, basis=basis, color=color, 
                     gene_symbols=gene_symbols, use_raw=use_raw, 
                     sort_order=sort_order, edges=edges, 
                     edges_width=edges_width, edges_color=edges_color, 
                     neighbors_key=neighbors_key, arrows=arrows, 
                     arrows_kwds=arrows_kwds, groups=groups, 
                     components=components, dimensions=dimensions, 
                     layer=layer, projection=projection, scale_factor=scale_factor,
                       color_map=color_map, cmap=cmap, palette=palette, 
                       na_color=na_color, na_in_legend=na_in_legend, 
                       size=size, frameon=frameon, legend_fontsize=legend_fontsize, 
                       legend_fontweight=legend_fontweight, legend_loc=legend_loc, 
                       legend_fontoutline=legend_fontoutline, colorbar_loc=colorbar_loc, 
                       vmax=vmax, vmin=vmin, vcenter=vcenter, norm=norm, 
                       add_outline=add_outline, outline_width=outline_width, 
                       outline_color=outline_color, ncols=ncols, hspace=hspace,
                         wspace=wspace, title=title, show=show, save=save, ax=ax,
                           return_fig=return_fig, marker=marker, **kwargs)



@register_function(
    aliases=["细胞比例图", "cellproportion", "cell_proportion", "细胞组成", "比例图"],
    category="pl",
    description="Plot cell proportion of each cell type in each visual cluster",
    examples=[
        "# Basic cell proportion plot",
        "ov.pl.cellproportion(adata, celltype_clusters='cell_type', groupby='leiden')",
        "# Horizontal bar plot",
        "ov.pl.cellproportion(adata, celltype_clusters='cell_type', groupby='leiden',",
        "                     transpose=True, figsize=(6,4))",
        "# Custom group order",
        "ov.pl.cellproportion(adata, celltype_clusters='cell_type', groupby='leiden',",
        "                     groupby_li=['0', '1', '2'], legend=True)"
    ],
    related=["pl.embedding", "tl.leiden"]
)
@accepts_frame
def cellproportion(adata:AnnData,celltype_clusters:str,groupby:str,
                       groupby_li=None,figsize:tuple=(4,6),
                       ticks_fontsize:int=12,labels_fontsize:int=12,ax=None,
                       legend:bool=False,legend_awargs=None,transpose:bool=False,
                       save:str=None,**kwargs):
    r"""Plot cell proportion of each cell type in each visual cluster.

    Args:
        adata: AnnData object.
        celltype_clusters: Cell type clusters.
        groupby: Visual clusters.
        groupby_li: Visual cluster list. (None)
        figsize: Figure size. ((4,6))
        ticks_fontsize: Ticks fontsize. (12)
        labels_fontsize: Labels fontsize. (12)
        ax: Matplotlib axes object. (None)
        legend: Whether to show legend. (False)
        legend_awargs: Legend arguments. ({'ncol':1})
        transpose: Whether to transpose the plot (horizontal bars). (False)
    
    Returns:
        None
    
    """

    b=pd.DataFrame(columns=['cell_type','value','Week'])
    visual_clusters=groupby
    visual_li=groupby_li
    if visual_li==None:
        adata.obs[visual_clusters]=adata.obs[visual_clusters].astype('category')
        visual_li=adata.obs[visual_clusters].cat.categories
    
    # Ensure color palettes exist for the requested categories
    import matplotlib
    if f'{celltype_clusters}_colors' not in adata.uns:
        palette = matplotlib.colormaps.get_cmap('tab20')
        colors = [matplotlib.colors.to_hex(palette(i % palette.N)) for i in range(len(adata.obs[celltype_clusters].cat.categories))]
        adata.uns[f'{celltype_clusters}_colors'] = colors
    if f'{visual_clusters}_colors' not in adata.uns:
        palette = matplotlib.colormaps.get_cmap('tab20')
        colors = [matplotlib.colors.to_hex(palette(i % palette.N)) for i in range(len(adata.obs[visual_clusters].cat.categories))]
        adata.uns[f'{visual_clusters}_colors'] = colors

    for i in visual_li:
        b1=pd.DataFrame()
        test=adata.obs.loc[adata.obs[visual_clusters]==i,celltype_clusters].value_counts()
        b1['cell_type']=test.index
        b1['value']=test.values/test.sum()
        b1['Week']=i.replace('Retinoblastoma_','')
        b=pd.concat([b,b1])
    
    plt_data2=adata.obs[celltype_clusters].value_counts()
    plot_data2_color_dict=dict(zip(adata.obs[celltype_clusters].cat.categories,
                                   adata.uns['{}_colors'.format(celltype_clusters)]))
    plt_data3=adata.obs[visual_clusters].value_counts()
    plot_data3_color_dict=dict(zip([i.replace('Retinoblastoma_','') for i in adata.obs[visual_clusters].cat.categories],adata.uns['{}_colors'.format(visual_clusters)]))
    b['cell_type_color'] = b['cell_type'].map(plot_data2_color_dict)
    b['stage_color']=b['Week'].map(plot_data3_color_dict)
    
    if legend_awargs is None:
        legend_awargs = {'ncol':1}

    created_ax = ax is None
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize)
    #用ax控制图片
    #sns.set_theme(style="whitegrid")
    #sns.set_theme(style="ticks")
    n=0
    all_celltype=adata.obs[celltype_clusters].cat.categories
    for i in all_celltype:
        if n==0:
            test1=b[b['cell_type']==i]
            if transpose:
                ax.barh(y=test1['Week'],width=test1['value'],height=0.8,color=list(set(test1['cell_type_color']))[0], label=i)
            else:
                ax.bar(x=test1['Week'],height=test1['value'],width=0.8,color=list(set(test1['cell_type_color']))[0], label=i)
            bottoms=test1['value'].values
        else:
            test2=b[b['cell_type']==i]
            if transpose:
                ax.barh(y=test2['Week'],width=test2['value'],left=bottoms,height=0.8,color=list(set(test2['cell_type_color']))[0], label=i)
            else:
                ax.bar(x=test2['Week'],height=test2['value'],bottom=bottoms,width=0.8,color=list(set(test2['cell_type_color']))[0], label=i)
            test1=test2
            bottoms+=test1['value'].values
        n+=1
    if legend!=False:
        # Merge defaults with user-supplied legend kwargs to avoid duplicate bbox_to_anchor
        legend_kw = {'bbox_to_anchor': (1.05, -0.05), 'loc': 3, 'borderaxespad': 0, 'fontsize': 10}
        legend_kw.update(legend_awargs or {})
        plt.legend(**legend_kw)
    
    plt.grid(False)
    
    plt.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # 设置左边和下边的坐标刻度为透明色
    #ax.yaxis.tick_left()
    #ax.xaxis.tick_bottom()
    #ax.xaxis.set_tick_params(color='none')
    #ax.yaxis.set_tick_params(color='none')

    # 设置左边和下边的坐标轴线为独立的线段
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    if transpose:
        plt.yticks(fontsize=ticks_fontsize,rotation=0)
        plt.xticks(fontsize=ticks_fontsize)
        plt.ylabel(groupby,fontsize=labels_fontsize)
        plt.xlabel('Cells per Stage',fontsize=labels_fontsize)
    else:
        plt.xticks(fontsize=ticks_fontsize,rotation=90)
        plt.yticks(fontsize=ticks_fontsize)
        plt.xlabel(groupby,fontsize=labels_fontsize)
        plt.ylabel('Cells per Stage',fontsize=labels_fontsize)
    #fig.tight_layout()
    if save:
        plt.savefig(save, bbox_inches="tight")
    if created_ax:
        return fig,ax
    


@register_function(
    aliases=["细胞类型嵌入图", "embedding_celltype", "celltype_embedding", "嵌入细胞类型", "细胞类型可视化"],
    category="pl",
    description="Plot embedding with celltype color visualization",
    examples=[
        "# Basic celltype embedding plot",
        "ov.pl.embedding_celltype(adata, celltype_key='cell_type')",
        "# Custom parameters",
        "ov.pl.embedding_celltype(adata, basis='tsne', figsize=(8,6),",
        "                         celltype_key='leiden', title='Cell Types')",
        "# Adjust range parameters",
        "ov.pl.embedding_celltype(adata, celltype_range=(1,8),",
        "                         embedding_range=(2,12))"
    ],
    related=["pl.embedding", "pl.cellproportion"]
)
def embedding_celltype(adata:AnnData,figsize:tuple=(6,4),basis:str='umap',
                            celltype_key:str='major_celltype',title:str=None,
                            celltype_range:tuple=(2,9),
                            embedding_range:tuple=(3,10),
                            xlim:int=-1000)->tuple:
    r"""
    Plot embedding with celltype color by omicverse.

    Args:
        adata: AnnData object  
        figsize: tuple, optional (default=(6,4))
            Figure size
        basis: str, optional (default='umap')
            Embedding method
        celltype_key: str, optional (default='major_celltype')
            Celltype key in adata.obs
        title: str, optional (default=None)
            Figure title
        celltype_range: tuple, optional (default=(2,9))
            Celltype range to plot
        embedding_range: tuple, optional (default=(3,10))
            Embedding range to plot
        xlim: int, optional (default=-1000)
            X axis limit

    Returns:
        fig: figure and axis
        ax: axis
    """

    adata.obs[celltype_key]=adata.obs[celltype_key].astype('category')
    cell_counts=adata.obs[celltype_key].value_counts()
    cell_num_pd=pd.DataFrame({celltype_key: cell_counts.values}, index=cell_counts.index)
    
    if '{}_colors'.format(celltype_key) in adata.uns.keys():
        cell_color_dict=dict(zip(adata.obs[celltype_key].cat.categories.tolist(),
                        adata.uns['{}_colors'.format(celltype_key)]))
    else:
        if len(adata.obs[celltype_key].cat.categories)>28:
            cell_color_dict=dict(zip(adata.obs[celltype_key].cat.categories, default_palette(len(adata.obs[celltype_key].cat.categories))))
        else:
            cell_color_dict=dict(zip(adata.obs[celltype_key].cat.categories, default_palette(len(adata.obs[celltype_key].cat.categories))))

    if figsize==None:
        if len(adata.obs[celltype_key].cat.categories)<10:
            fig = plt.figure(figsize=(6,4))
        else:
            print('The number of cell types is too large, please set the figsize parameter')
            return
    else:
        fig = plt.figure(figsize=figsize)
    grid = plt.GridSpec(10, 10)
    ax1 = fig.add_subplot(grid[:, embedding_range[0]:embedding_range[1]])       # 占据第一行的所有列
    ax2 = fig.add_subplot(grid[celltype_range[0]:celltype_range[1], :2]) 
    # 定义子图的大小和位置
         # 占据第二行的前两列
    #ax3 = fig.add_subplot(grid[1:, 2])      # 占据第二行及以后的最后一列
    #ax4 = fig.add_subplot(grid[2, 0])       # 占据最后一行的第一列
    #ax5 = fig.add_subplot(grid[2, 1])       # 占据最后一行的第二列

    embedding(
        adata,
        basis=basis,
        color=[celltype_key],
        title='',
        frameon=False,
        ncols=3,
        ax=ax1,
        legend_loc=False,
        show=False
    )



    for idx,cell in zip(range(cell_num_pd.shape[0]),
                        adata.obs[celltype_key].cat.categories):
        ax2.scatter(100,
                cell,c=cell_color_dict[cell],s=50)
        ax2.plot((100,cell_num_pd.loc[cell,celltype_key]),(idx,idx),
                c=cell_color_dict[cell],lw=4)
        ax2.text(100,idx+0.2,
                cell+'('+str("{:,}".format(cell_num_pd.loc[cell,celltype_key]))+')',fontsize=11)
    ax2.set_xlim(xlim,cell_num_pd.iloc[1].values[0]) 
    ax2.text(xlim,idx+1,title,fontsize=12)
    ax2.grid(False)
    #ax2.legend(bbox_to_anchor=(1.05, -0.05), loc=3, borderaxespad=0,fontsize=10,**legend_awargs)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.axis('off')

    # ——关键：确保 ax2 没有图例——
    if ax1.get_legend() is not None:   # 如果有，就移除
        ax1.get_legend().remove()
    if ax2.get_legend() is not None:   # 如果有，就移除
        ax2.get_legend().remove()

    return fig,[ax1,ax2]


@register_function(
    aliases=["凸包", "ConvexHull", "convex_hull", "凸包图", "轮廓图"],
    category="pl",
    description="Plot the ConvexHull for a cluster in embedding visualization",
    examples=[
        "# Basic ConvexHull plot",
        "import matplotlib.pyplot as plt",
        "fig, ax = plt.subplots()",
        "ov.pl.embedding(adata, basis='X_umap', color='leiden', ax=ax, show=False)",
        "ov.pl.ConvexHull(adata, basis='X_umap', cluster_key='leiden',",
        "                 hull_cluster='0', ax=ax)",
        "# Multiple clusters",
        "for cluster in ['0', '1', '2']:",
        "    ov.pl.ConvexHull(adata, basis='X_umap', cluster_key='leiden',",
        "                     hull_cluster=cluster, ax=ax, alpha=0.3)"
    ],
    related=["pl.embedding", "pl.contour"]
)
def ConvexHull(adata:AnnData,basis:str,cluster_key:str,
                    hull_cluster:str,ax,color=None,alpha:float=0.2):
    r"""
    Plot the ConvexHull for a cluster in embedding.

    Args:
        adata: AnnData object
        basis: str
            Embedding method in adata.obsm
        cluster_key: str
            Cluster key in adata.obs
        hull_cluster: str
            Cluster to plot for ConvexHull
        ax: matplotlib.axes.Axes
            Axes object
        color: str, optional (default=None)
            Color for ConvexHull
        alpha: float, optional (default=0.2)
            Alpha for ConvexHull

    Returns:
        ax: matplotlib.axes.Axes
            Modified axes object
    """
    from scipy.spatial import ConvexHull
    adata.obs[cluster_key]=adata.obs[cluster_key].astype('category')
    if '{}_colors'.format(cluster_key) in adata.uns.keys():
        print('{}_colors'.format(cluster_key))
        type_color_all=dict(zip(adata.obs[cluster_key].cat.categories,adata.uns['{}_colors'.format(cluster_key)]))
    else:
        if len(adata.obs[cluster_key].cat.categories)>28:
            type_color_all=dict(zip(adata.obs[cluster_key].cat.categories, default_palette(len(adata.obs[cluster_key].cat.categories))))
        else:
            type_color_all=dict(zip(adata.obs[cluster_key].cat.categories, default_palette(len(adata.obs[cluster_key].cat.categories))))
    
    #color_dict=dict(zip(adata.obs[cluster_key].cat.categories,adata.uns[f'{cluster_key}_colors']))
    points=adata[adata.obs[cluster_key]==hull_cluster].obsm[basis]
    hull = ConvexHull(points)
    vert = np.append(hull.vertices, hull.vertices[0])  # close the polygon by appending the first point at the end
    if color==None:
        ax.plot(points[vert, 0], points[vert, 1], '--', c=type_color_all[hull_cluster])
        ax.fill(points[vert, 0], points[vert, 1], c=type_color_all[hull_cluster], alpha=alpha)
    else:
        ax.plot(points[vert, 0], points[vert, 1], '--', c=color)
        ax.fill(points[vert, 0], points[vert, 1], c=color, alpha=alpha)
    return ax


@register_function(
    aliases=["嵌入标签调整", "embedding_adjust", "adjust_labels", "标签调整", "嵌入图调整"],
    category="pl",
    description="Get locations of cluster median and adjust text labels accordingly on embedding",
    examples=[
        "# Basic label adjustment",
        "ov.pl.embedding_adjust(adata, groupby='leiden')",
        "# Exclude specific clusters",
        "ov.pl.embedding_adjust(adata, groupby='cell_type', exclude=('Unknown',))",
        "# Custom parameters",
        "ov.pl.embedding_adjust(adata, groupby='leiden', basis='X_tsne',",
        "                       adjust_kwargs={'avoid_text': True},",
        "                       text_kwargs={'fontsize': 12, 'weight': 'bold'})"
    ],
    related=["pl.embedding", "pl.embedding_celltype"]
)
def embedding_adjust(
    adata, groupby, exclude=(), 
    basis='X_umap',ax=None, adjust_kwargs=None, text_kwargs=None
):
    r"""
    Get locations of cluster median and adjust text labels accordingly.
    
    Borrowed from scanpy github forum.

    Args:
        adata: AnnData object
        groupby: str
            Key in adata.obs for grouping
        exclude: tuple, optional (default=())
            Groups to exclude from labeling
        basis: str, optional (default='X_umap')
            Embedding basis key in adata.obsm
        ax: matplotlib.axes.Axes, optional (default=None)
            Axes object to plot on
        adjust_kwargs: dict, optional (default=None)
            Arguments for adjust_text function
        text_kwargs: dict, optional (default=None)
            Arguments for text annotation

    Returns:
        medians: dict
            Dictionary of median positions for each group
    """
    if adjust_kwargs is None:
        adjust_kwargs = {"text_from_points": False}
    if text_kwargs is None:
        text_kwargs = {}

    medians = {}

    for g, g_idx in adata.obs.groupby(groupby).groups.items():
        if g in exclude:
            continue
        medians[g] = np.median(adata[g_idx].obsm[basis], axis=0)

    if ax is None:
        texts = [
            plt.text(x=x, y=y, s=k, **text_kwargs) for k, (x, y) in medians.items()
        ]
    else:
        texts = [ax.text(x=x, y=y, s=k, **text_kwargs) for k, (x, y) in medians.items()]
    from adjustText import adjust_text
    adjust_text(texts, **adjust_kwargs)
    return texts


@register_function(
    aliases=["嵌入密度图", "embedding_density", "density_embedding", "密度嵌入图", "嵌入密度可视化"],
    category="pl",
    description="Create density visualization for specific clusters on embedding",
    examples=[
        "# Basic embedding density plot",
        "ov.pl.embedding_density(adata, basis='X_umap', groupby='leiden',",
        "                        target_clusters=['0', '1'])",
        "# Multiple clusters",
        "ov.pl.embedding_density(adata, basis='X_tsne', groupby='cell_type',",
        "                        target_clusters=['T_cell', 'B_cell'])",
        "# Custom visualization parameters",
        "ov.pl.embedding_density(adata, basis='X_umap', groupby='leiden',",
        "                        target_clusters=['2'], color='viridis')"
    ],
    related=["pl.embedding", "pl.calculate_gene_density", "pl.add_density_contour"]
)
def embedding_density(adata,basis,groupby,target_clusters,**kwargs):
    """Plot cluster-specific density on an existing embedding.

    Parameters
    ----------
    adata : AnnData
        AnnData containing embedding coordinates and group labels.
    basis : str
        Embedding key (e.g., ``'X_umap'``).
    groupby : str
        Observation column defining cluster labels.
    target_clusters : str or list
        Cluster label(s) to highlight in density map.
    **kwargs
        Extra plotting arguments forwarded to ``embedding``.

    Returns
    -------
    Any
        Return value of ``embedding(...)`` with temporary density color.
    """
    if 'X_' in basis:
        basis1=basis.split('_')[1]
    compute_embedding_density(adata,
                              basis=basis1,
                              groupby=groupby,
                              key_added='temp_density')
    adata.obs.loc[adata.obs[groupby]!=target_clusters,'temp_density']=0
    return embedding(adata,
                  basis=basis,
                  color=['temp_density'],
                    title=target_clusters,
                   **kwargs
                 )

@register_function(
    aliases=["柱点图", "bardotplot", "bar_dot_plot", "柱状点图", "条形点图"],
    category="pl",
    description="Create combined bar plot and dot plot for categorical data visualization",
    examples=[
        "# Basic bar dot plot",
        "ov.pl.bardotplot(adata, groupby='cell_type', color='CD3D')",
        "# Custom styling",
        "ov.pl.bardotplot(adata, groupby='leiden', color='marker_gene',",
        "                 figsize=(10,4), fontsize=14)",
        "# Return values for further processing",
        "values = ov.pl.bardotplot(adata, groupby='cluster', color='gene',",
        "                          return_values=True)"
    ],
    related=["pl.violin", "pl.dotplot", "pl.boxplot"]
)
def bardotplot(adata,groupby,color,figsize=(8,3),return_values=False,
               fontsize=12,xlabel='',ylabel='',xticks_rotation=90,ax=None,
               bar_kwargs=None,scatter_kwargs=None):
    """Create a combined bar-and-dot summary plot by groups.

    Parameters
    ----------
    adata : AnnData
        AnnData containing expression/metadata used for plotting.
    groupby : str
        Grouping column in ``adata.obs``.
    color : str
        Feature in ``adata.var_names`` or ``adata.obs`` to summarize.
    figsize : tuple, default=(8, 3)
        Figure size.
    return_values : bool, default=False
        Whether to return computed grouped values instead of plotting.
    fontsize : int, default=12
        Font size.
    xlabel : str, default=''
        X-axis label.
    ylabel : str, default=''
        Y-axis label.
    xticks_rotation : int, default=90
        Rotation angle of x tick labels.
    ax : matplotlib.axes.Axes, optional
        Existing axis for plotting.
    bar_kwargs : dict, optional
        Keyword arguments forwarded to ``plt.bar``.
    scatter_kwargs : dict, optional
        Keyword arguments forwarded to ``plt.scatter``.

    Returns
    -------
    pandas.DataFrame or tuple
        Grouped values when ``return_values=True``; otherwise plotting handles.
    """
    
    if bar_kwargs is None:
        bar_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}
    
    var_ticks=False
    obs_ticks=False
    plot_text_=color
    if (plot_text_ in adata.var_names):
        adata1=adata
        var_ticks=True
    elif plot_text_ in adata.obs.columns:
        adata1=adata
        obs_ticks=True
    elif (adata.raw!=None) and (plot_text_ in adata.raw.var_names):
        adata1=adata1.raw.to_adata()
        var_ticks=True
    else:
        print(f'Please check the `{color}` key in adata.obs or adata.var')
        return
    adata1.obs[groupby]=adata1.obs[groupby].astype('category')
    
    if var_ticks==True:
        plot_data=pd.DataFrame()
        max_len=0
        for group in adata1.obs[groupby].cat.categories:
            if max_len<len(adata1[adata1.obs[groupby]==group,plot_text_].to_df().values.reshape(-1)):
                max_len=len(adata1[adata1.obs[groupby]==group,plot_text_].to_df().values.reshape(-1))
        for group in adata1.obs[groupby].cat.categories:
            t_data1=list(adata1[adata1.obs[groupby]==group,plot_text_].to_df().values.reshape(-1))
            while len(t_data1)<max_len:
                t_data1.append(np.nan)
            plot_data[group]=t_data1
    elif obs_ticks==True:
        plot_data=pd.DataFrame()
        max_len=0
        for group in adata1.obs[groupby].cat.categories:
            if max_len<len(adata1.obs.loc[adata1.obs[groupby]==group,plot_text_].values.reshape(-1)):
                max_len=len(adata1.obs.loc[adata1.obs[groupby]==group,plot_text_].values.reshape(-1))
        for group in adata1.obs[groupby].cat.categories:
            t_data1=list(adata1.obs.loc[adata1.obs[groupby]==group,plot_text_].values.reshape(-1))
            while len(t_data1)<max_len:
                t_data1.append(np.nan)
            plot_data[group]=t_data1
    
    if return_values==True:
        return plot_data
    
    
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize)
        
    xbar = np.arange(len(plot_data.columns.to_numpy()))
    #color_list_dot=[ov.utils.green_color[0],ov.utils.green_color[1],ov.utils.red_color[0],'#EC9DC5','#5BC23D']
    #color_list_dot=adata.uns['clusters_colors']
    if '{}_colors'.format(groupby) in adata.uns.keys():
        color_list_dot=adata.uns['{}_colors'.format(groupby)]
    else:
        if len(adata.obs[groupby].cat.categories)>28:
            color_list_dot=default_palette(len(adata.obs[groupby].cat.categories))
        else:
            color_list_dot=default_palette(len(adata.obs[groupby].cat.categories))
            
    plt.bar(x=plot_data.columns, 
            height=plot_data.describe().loc['mean'], 
            yerr=plot_data.sem(), 
            color=color_list_dot, 
            zorder=1, #fill=False,
            edgecolor=color_list_dot,
            error_kw={'elinewidth': None, 'capthick': None},**bar_kwargs)
    bw=0.4
    for cols in range(len(plot_data.columns.to_numpy())):
        # get markers from here https://matplotlib.org/3.1.1/api/markers_api.html
        plt.scatter(x=np.linspace(xbar[cols]-bw/2, xbar[cols]+bw/2, int(plot_data.describe().loc['count'][cols])),
                   y=plot_data[plot_data.columns[cols]].dropna(), 
                    color=color_list_dot[cols], zorder=1, 
                   **scatter_kwargs)


    style_axes(ax)

    plt.xticks(rotation=xticks_rotation,fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize+1)
    plt.ylabel(ylabel,fontsize=fontsize+1)
    plt.title(plot_text_,fontsize=fontsize+1)
    if ax==None:
        return fig,ax
    
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import kruskal

@register_function(
    aliases=["单组箱线图", "single_group_boxplot", "single_boxplot", "单个组箱图", "单组箱形图"],
    category="pl",
    description="Create boxplot for single group comparison with statistical analysis",
    examples=[
        "# Basic single group boxplot",
        "ov.pl.single_group_boxplot(adata, groupby='cell_type', color='CD3D')",
        "# Custom styling with color dictionary",
        "colors = {'T_cell': 'red', 'B_cell': 'blue', 'NK_cell': 'green'}",
        "ov.pl.single_group_boxplot(adata, groupby='cell_type', color='marker_gene',",
        "                           type_color_dict=colors, title='Expression Comparison')",
        "# Multiple comparisons",
        "ov.pl.single_group_boxplot(adata, groupby='leiden', color='gene_expression',",
        "                           ylabel='Expression Level')"
    ],
    related=["pl.boxplot", "pl.violin", "pl.bardotplot"]
)
def single_group_boxplot(adata,
                         groupby: str = '',
                         color: str = '',
                         type_color_dict: dict = None,
                         title: str = '',
                         ylabel: str = '',
                         kruskal_test: bool = False,
                         figsize: tuple = (4, 4),
                         x_ticks_plot: bool = False,
                         legend_plot: bool = True,
                         bbox_to_anchor: tuple = (1, 0.55),
                         save: bool = False,
                         point_number: int = 5,
                         save_pathway: str = '',
                         sort: bool = True,
                         scatter_kwargs: dict = None,
                         ax = None,
                         fontsize = 12,
                        ):
    """
    adata (AnnData object): The data object containing the information for plotting.
    groupby (str): The variable used for grouping the data.
    color (str): The variable used for coloring the data points.
    type_color_dict (dict): A dictionary mapping group categories to specific colors.
    title (str): The title for the plot.
    ylabel (str): The label for the y-axis.
    kruskal_test (bool): Whether to perform a Kruskal-Wallis test and display the p-value on the plot.
    figsize (tuple): The size of the plot figure in inches (width, height).
    x_ticks_plot (bool): Whether to display x-axis tick labels.
    legend_plot (bool): Whether to display a legend for the groups.
    bbox_to_anchor (tuple): The position of the legend bbox (x, y) in axes coordinates.
    save (bool): Whether to save the plot to a file.
    point_number (int): The number of data points to be plotted for each group.
    save_pathway (str): The file path for saving the plot (if save is True).
    sort (bool): Whether to sort the groups based on their mean values.
    scatter_kwargs (dict): Additional keyword arguments for customizing the scatter plot.
    ax (matplotlib.axes.Axes): A pre-existing axes object for plotting (optional).
    
    Examples:
        ov.pl.single_group_boxplot(adata,groupby='clusters',
             color='Sox_aucell',
             type_color_dict=dict(zip(pd.Categorical(adata.obs['clusters']).categories, adata.uns['clusters_colors'])),
             x_ticks_plot=True,
             figsize=(5,4),
             kruskal_test=True,
             ylabel='Sox_aucell',
             legend_plot=False,
             bbox_to_anchor=(1,1),
             title='Expression',
             scatter_kwargs={'alpha':0.8,'s':10,'marker':'o'},
                 point_number=15,
                 sort=False,
                 save=False,
                 )
    """

    if scatter_kwargs is None:
        scatter_kwargs = {}

    # Create an empty dictionary to store results
    plot_data = {}

    var_ticks = False
    obs_ticks = False
    plot_text_ = color
    if (plot_text_ in adata.var_names):
        adata1 = adata.copy()
        var_ticks = True
    elif plot_text_ in adata.obs.columns:
        adata1 = adata.copy()
        obs_ticks = True
    elif (adata.raw is not None) and (plot_text_ in adata.raw.var_names):
        adata1 = adata.raw.to_adata().copy()
        var_ticks = True
    else:
        print(f'Please check the `{color}` key in adata.obs or adata.var')
        return
    adata1.obs[groupby] = adata1.obs[groupby].astype('category')

    if var_ticks == True:
        adata1.obs[color] = adata1[:, plot_text_].to_df().values.flatten()
        #print(adata1.obs[color])

    # Categorize by groups
    
    for group in adata1.obs[groupby].cat.categories:
        plot_data[group] = np.array(adata1.obs.loc[adata1.obs[groupby] == group, color].tolist())

    if sort == True:
        sorted_keys = sorted(plot_data.keys(), key=lambda k: np.mean(plot_data[k]))
        sorted_plot_data = {key: plot_data[key] for key in sorted_keys}
        plot_data = sorted_plot_data

        sorted_colors = [type_color_dict[key] for key in sorted_keys]
        sc_color = sorted_colors
    else:
        sc_color = [type_color_dict[key] for key in plot_data.keys()]

    shake_dict = {}

    for group in adata1.obs[groupby].cat.categories:
        data_list = []
        gene_data = adata1.obs.loc[adata1.obs[groupby] == group, color].tolist()
        if len(gene_data) > point_number:
            bootstrap_data = np.random.choice(gene_data, size=point_number, replace=False)
        else:
            bootstrap_data = gene_data
        shake_dict[group] = np.array(bootstrap_data)

    if ax==None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot boxplots
    width = 0.8
    ticks = np.arange(len(plot_data))
    positions = np.arange(len(ticks))

    for num, (hue_data, hue_color) in enumerate(zip(plot_data.keys(), sc_color)):
        position = positions[num]
        b1 = ax.boxplot(plot_data[hue_data],
                        positions=[position],
                        sym='',
                        widths=width,
                        patch_artist=True
                        )
        plt.scatter(np.random.normal(position, 0.12, point_number),
                    shake_dict[hue_data],
                    c=hue_color, zorder=1, **scatter_kwargs)
        box = b1['boxes'][0]
        light_hue_color = tuple((min(1, c + 0.5 * (1 - c))) for c in plt.cm.colors.to_rgb(hue_color))
        box.set(facecolor=light_hue_color, edgecolor=hue_color, linewidth=2)
        plt.setp(b1['whiskers'], color=hue_color, linewidth=2)
        plt.setp(b1['caps'], color=hue_color, linewidth=2)
        plt.setp(b1['medians'], color=hue_color, linewidth=3)

    # Axis labels and title
    

    if x_ticks_plot == True:
        ax.set_xticks(positions)
        ax.set_xticklabels(plot_data.keys(), rotation=90, fontsize=fontsize)
    else:
        ax.set_xticklabels([])

    yticks = ax.get_yticks()
    ax.set_title(title, fontsize=fontsize+1,)
    plt.ylabel(ylabel, fontsize=fontsize+1, )
    style_axes(ax)

    if legend_plot == True:
        labels = list(plot_data.keys())
        patches = [mpatches.Patch(color=sc_color[i], label="{:s}".format(labels[i])) for i in range(len(plot_data))]
        ax.legend(handles=patches, bbox_to_anchor=bbox_to_anchor, ncol=1, fontsize=fontsize)

    if kruskal_test == True:
        data_list = [plot_data[key] for key in plot_data]
        statistic, p_value = kruskal(*data_list)

        if p_value < 0.0001:
            formatted_p_value = "{:.2e}".format(p_value)
        else:
            formatted_p_value = "{:.4f}".format(p_value)
        if p_value < 2.2e-16:
            formatted_p_value = 2.2e-16
            text = f"Kruskal-Wallis: P < {formatted_p_value}"
        else:
            text = f"Kruskal-Wallis: P = {formatted_p_value}"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=fontsize, fontweight='bold', verticalalignment='top',
                 bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.5'))

    if save == True:
        plt.savefig(save_pathway, dpi=300, bbox_inches='tight')

    if ax is None:
        return fig, ax

@register_function(
    aliases=["等高线图", "contour", "contour_plot", "轮廓线", "密度等高线"],
    category="pl",
    description="Add density contour lines for specific clusters on embedding plot",
    examples=[
        "# Basic contour plot",
        "import matplotlib.pyplot as plt",
        "fig, ax = plt.subplots()",
        "ov.pl.embedding(adata, basis='X_umap', color='leiden', ax=ax, show=False)",
        "ov.pl.contour(ax, adata, groupby='leiden', clusters=['0', '1'])",
        "# Custom parameters",
        "ov.pl.contour(ax, adata, groupby='cell_type', clusters=['T_cell'],",
        "              basis='X_tsne', grid_density=150, contour_threshold=0.05)",
        "# Multiple cluster contours",
        "for cluster in ['0', '1', '2']:",
        "    ov.pl.contour(ax, adata, groupby='leiden', clusters=[cluster])"
    ],
    related=["pl.embedding", "pl.ConvexHull", "pl.add_density_contour"]
)
def contour(ax,adata,groupby,clusters,basis='X_umap',
            grid_density=100,contour_threshold=0.1,
           **kwargs):
    """Overlay a KDE contour for selected clusters on embedding axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis where contour lines are drawn.
    adata : AnnData
        AnnData containing embedding coordinates and cluster labels.
    groupby : str
        Observation column for cluster filtering.
    clusters : list
        Cluster labels to include in contour estimation.
    basis : str, default='X_umap'
        Embedding key in ``adata.obsm``.
    grid_density : int, default=100
        Resolution of contour estimation grid.
    contour_threshold : float, default=0.1
        Relative density threshold used for outer contour level.
    **kwargs
        Additional arguments forwarded to ``ax.contour``.

    Returns
    -------
    matplotlib.axes.Axes
        Axis with contour overlay.
    """
    from scipy.stats import gaussian_kde
    umap_embedding=adata[adata.obs[groupby].isin(clusters)].obsm[basis]
    kde = gaussian_kde(umap_embedding.T)
    # 生成网格
    x_grid = np.linspace(min(umap_embedding[:, 0]), max(umap_embedding[:, 0]), grid_density)
    y_grid = np.linspace(min(umap_embedding[:, 1]), max(umap_embedding[:, 1]), grid_density)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
    Z = np.reshape(kde(positions).T, X_grid.shape)
    
    # 找到最外层轮廓的阈值
    threshold = np.max(Z) * contour_threshold  # 通过调整阈值的倍数来控制轮廓的密度

    # 绘制最外层轮廓
    contour = ax.contour(X_grid, Y_grid, Z, 
                         levels=[threshold],
                        **kwargs)

    # 获取最外层轮廓的路径
    #outer_contour_path = contour.collections[0].get_paths()[0]
    #outer_contour_x, outer_contour_y = outer_contour_path.vertices[:, 0], outer_contour_path.vertices[:, 1]

    return ax


def plot_boxplots(  # pragma: no cover
        data,
        feature_name: str,
        modality_key: str = "coda",
        y_scale = "relative",
        plot_facets: bool = False,
        add_dots: bool = False,
        cell_types=None,
        args_boxplot=None,
        args_swarmplot= None,
        palette= "Blues",
        show_legend= True,
        level_order=None,
        figsize  = None,
        dpi= 100,
        return_fig= None,
        ax = None,
        show = None,
        save = None,
    ):
        """Grouped boxplot visualization.

         The cell counts for each cell type are shown as a group of boxplots
         with intra--group separation by a covariate from data.obs.

        Args:
            data: AnnData object or MuData object
            feature_name: The name of the feature in data.obs to plot
            modality_key: If data is a MuData object, specify which modality to use.
            y_scale: Transformation to of cell counts. Options: "relative" - Relative abundance, "log" - log(count),
                     "log10" - log10(count), "count" - absolute abundance (cell counts).
            plot_facets: If False, plot cell types on the x-axis. If True, plot as facets.
            add_dots: If True, overlay a scatterplot with one dot for each data point.
            cell_types: Subset of cell types that should be plotted.
            args_boxplot: Arguments passed to sns.boxplot.
            args_swarmplot: Arguments passed to sns.swarmplot.
            figsize: Figure size.
            dpi: Dpi setting.
            palette: The seaborn color map for the barplot.
            show_legend: If True, adds a legend.
            level_order: Custom ordering of bars on the x-axis.

        Returns:
            Depending on `plot_facets`, returns a :class:`~matplotlib.axes.Axes` (`plot_facets = False`)
            or: class:`~sns.axisgrid.FacetGrid` (`plot_facets = True`) object

        Examples:
            >>> import pertpy as pt
            >>> haber_cells = pt.dt.haber_2017_regions()
            >>> sccoda = pt.tl.Sccoda()
            >>> mdata = sccoda.load(haber_cells, type="cell_level", generate_sample_level=True, cell_type_identifier="cell_label", \
                sample_identifier="batch", covariate_obs=["condition"])
            >>> sccoda.plot_boxplots(mdata, feature_name="condition", add_dots=True)

            Preview:
                .. image:: /_static/docstring_previews/sccoda_boxplots.png
        """
        if args_boxplot is None:
            args_boxplot = {}
        if args_swarmplot is None:
            args_swarmplot = {}
        #if isinstance(data, MuData):
        #    data = data[modality_key]
        #if isinstance(data, AnnData):
        #    data = data
        # y scale transformations
        if y_scale == "relative":
            sample_sums = np.sum(data.X, axis=1, keepdims=True)
            X = data.X / sample_sums
            value_name = "Proportion"
        # add pseudocount 0.5 if using log scale
        elif y_scale == "log":
            X = data.X.copy()
            X[X == 0] = 0.5
            X = np.log(X)
            value_name = "log(count)"
        elif y_scale == "log10":
            X = data.X.copy()
            X[X == 0] = 0.5
            X = np.log(X)
            value_name = "log10(count)"
        elif y_scale == "count":
            X = data.X
            value_name = "count"
        else:
            raise ValueError("Invalid y_scale transformation")

        count_df = pd.DataFrame(X, columns=data.var.index, index=data.obs.index).merge(
            data.obs[feature_name], left_index=True, right_index=True
        )
        plot_df = pd.melt(count_df, id_vars=feature_name, var_name="Cell type", value_name=value_name)
        if cell_types is not None:
            plot_df = plot_df[plot_df["Cell type"].isin(cell_types)]

        # Currently disabled because the latest statsannotations does not support the latest seaborn.
        # We had to drop the dependency.
        # Get credible effects results from model
        # if draw_effects:
        #     if model is not None:
        #         credible_effects_df = model.credible_effects(data, modality_key).to_frame().reset_index()
        #     else:
        #         print("[bold yellow]Specify a tasCODA model to draw effects")
        #     credible_effects_df[feature_name] = credible_effects_df["Covariate"].str.removeprefix(f"{feature_name}[T.")
        #     credible_effects_df[feature_name] = credible_effects_df[feature_name].str.removesuffix("]")
        #     credible_effects_df = credible_effects_df[credible_effects_df["Final Parameter"]]

        # If plot as facets, create a FacetGrid and map boxplot to it.
        if plot_facets:
            if level_order is None:
                level_order = pd.unique(plot_df[feature_name])

            K = X.shape[1]

            if figsize is not None:
                height = figsize[0]
                aspect = np.round(figsize[1] / figsize[0], 2)
            else:
                height = 3
                aspect = 2

            g = sns.FacetGrid(
                plot_df,
                col="Cell type",
                sharey=False,
                col_wrap=int(np.floor(np.sqrt(K))),
                height=height,
                aspect=aspect,
            )
            g.map(
                sns.boxplot,
                feature_name,
                value_name,
                palette=palette,
                order=level_order,
                **args_boxplot,
            )

            if add_dots:
                if "hue" in args_swarmplot:
                    hue = args_swarmplot.pop("hue")
                else:
                    hue = None

                if hue is None:
                    g.map(
                        sns.swarmplot,
                        feature_name,
                        value_name,
                        color="black",
                        order=level_order,
                        **args_swarmplot,
                    ).set_titles("{col_name}")
                else:
                    g.map(
                        sns.swarmplot,
                        feature_name,
                        value_name,
                        hue,
                        order=level_order,
                        **args_swarmplot,
                    ).set_titles("{col_name}")

            if save:
                plt.savefig(save, bbox_inches="tight")
            if show:
                plt.show()
            if return_fig:
                return plt.gcf()
            if not (show or save):
                return g
            return None

        # If not plot as facets, call boxplot to plot cell types on the x-axis.
        else:
            if level_order:
                args_boxplot["hue_order"] = level_order
                args_swarmplot["hue_order"] = level_order

            _, ax = plt.subplots(figsize=figsize, dpi=dpi)

            ax = sns.boxplot(
                x="Cell type",
                y=value_name,
                hue=feature_name,
                data=plot_df,
                fliersize=1,
                palette=palette,
                ax=ax,
                **args_boxplot,
            )

            # Currently disabled because the latest statsannotations does not support the latest seaborn.
            # We had to drop the dependency.
            # if draw_effects:
            #     pairs = [
            #         [(row["Cell Type"], row[feature_name]), (row["Cell Type"], "Control")]
            #         for _, row in credible_effects_df.iterrows()
            #     ]
            #     annot = Annotator(ax, pairs, data=plot_df, x="Cell type", y=value_name, hue=feature_name)
            #     annot.configure(test=None, loc="outside", color="red", line_height=0, verbose=False)
            #     annot.set_custom_annotations([row[feature_name] for _, row in credible_effects_df.iterrows()])
            #     annot.annotate()

            if add_dots:
                sns.swarmplot(
                    x="Cell type",
                    y=value_name,
                    data=plot_df,
                    hue=feature_name,
                    ax=ax,
                    dodge=True,
                    palette="dark:black",
                    **args_swarmplot,
                )

            cell_types = pd.unique(plot_df["Cell type"])
            ax.set_xticklabels(cell_types, rotation=90)

            if show_legend:
                handles, labels = ax.get_legend_handles_labels()
                handout = []
                labelout = []
                for h, l in zip(handles, labels, strict=False):
                    if l not in labelout:
                        labelout.append(l)
                        handout.append(h)
                ax.legend(
                    handout,
                    labelout,
                    loc="upper left",
                    bbox_to_anchor=(1, 1),
                    ncol=1,
                    title=feature_name,
                )

            if save:
                plt.savefig(save, bbox_inches="tight")
            if show:
                plt.show()
            if return_fig:
                return plt.gcf()
            if not (show or save):
                return ax
            return None
        

@accepts_frame
def cellstackarea(adata,celltype_clusters:str,groupby:str,
                       groupby_li=None,figsize:tuple=(4,6),
                       ticks_fontsize:int=12,labels_fontsize:int=12,ax=None,
                       legend:bool=False,legend_awargs={},text_show=False,):
    """
    Plot the cell type percentage in each groupby category
    
    """
    df = adata.obs[[groupby, celltype_clusters]]

    # 计算每个样本类型中每个细胞类型的数量
    count_df = df.groupby([groupby, celltype_clusters]).size().reset_index(name='count')
    
    # 计算每个样本类型中的总数
    total_count_df = count_df.groupby(groupby)['count'].sum().reset_index(name='total_count')
    
    # 将总数合并回原数据框
    count_df = count_df.merge(total_count_df, on=groupby)
    
    # 计算百分比
    count_df['percentage'] = count_df['count'] / count_df['total_count'] * 100
    
    # 将数据从长格式转换为宽格式，以便绘制面积图
    pivot_df = count_df.pivot(index=groupby, columns=celltype_clusters, values='percentage').fillna(0)
    if groupby_li!=None:
        pivot_df=pivot_df.loc[groupby_li]

    
    # 使用 matplotlib 绘制面积图
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # 为每种细胞类型绘制面积图
    cell_types = pivot_df.columns
    bottom = pd.Series([0] * len(pivot_df), index=pivot_df.index)

    adata.obs[celltype_clusters]=adata.obs[celltype_clusters].astype('category')
    if '{}_colors'.format(celltype_clusters) in adata.uns.keys():
        print('{}_colors'.format(celltype_clusters))
        type_color_all=dict(zip(adata.obs[celltype_clusters].cat.categories,adata.uns['{}_colors'.format(celltype_clusters)]))
    else:
        if len(adata.obs[celltype_clusters].cat.categories)>28:
            type_color_all=dict(zip(adata.obs[celltype_clusters].cat.categories, default_palette(len(adata.obs[celltype_clusters].cat.categories))))
        else:
            type_color_all=dict(zip(adata.obs[celltype_clusters].cat.categories, default_palette(len(adata.obs[celltype_clusters].cat.categories))))
    
    
    
    for cell_type in cell_types:
        ax.fill_between(pivot_df.index, bottom, bottom + pivot_df[cell_type], label=cell_type,
                       color=type_color_all[cell_type])
        max_index = pivot_df[cell_type].idxmax()
        if text_show==True:
            ax.text(max_index,bottom[max_index]+pivot_df.loc[max_index,cell_type]/2,cell_type,fontsize=ticks_fontsize-1)
        
        bottom += pivot_df[cell_type]
    
    if legend!=False:
        plt.legend(bbox_to_anchor=(1.05, -0.05), loc=3, borderaxespad=0,
                   fontsize=labels_fontsize,**legend_awargs)
    
    plt.grid(False)
    
    plt.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # 设置左边和下边的坐标刻度为透明色
    #ax.yaxis.tick_left()
    #ax.xaxis.tick_bottom()
    #ax.xaxis.set_tick_params(color='none')
    #ax.yaxis.set_tick_params(color='none')

    # 设置左边和下边的坐标轴线为独立的线段
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    plt.xticks(fontsize=ticks_fontsize,rotation=90)
    plt.yticks(fontsize=ticks_fontsize)
    plt.xlabel(groupby,fontsize=labels_fontsize)
    plt.ylabel('Cells per Stage',fontsize=labels_fontsize)
    #fig.tight_layout()
    if ax==None:
        return fig,ax



def violin_old(adata,keys=None,groupby=None,ax=None,figsize=(4,4),fontsize=13,
           ticks_fontsize=None,rotation=90,**kwargs):
    warn(
        "`ov.pl.violin_old` is deprecated and will be removed in omicverse 2.2; "
        "use `ov.pl.violin` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from ._violin import violin

    if ax==None:
        fig, ax = plt.subplots(figsize=figsize)
    violin(
        adata,
        keys=keys,
        groupby=groupby,
        ax=ax,
        figsize=figsize,
        show=False,
        fontsize=fontsize,
        **kwargs,
    )
    style_axes(ax)
    if ticks_fontsize==None:
        ticks_fontsize=fontsize-1

    plt.xticks(fontsize=ticks_fontsize,rotation=rotation)
    plt.yticks(fontsize=ticks_fontsize)
    plt.xlabel(groupby,fontsize=fontsize)
    plt.ylabel(keys,fontsize=fontsize)

    if ax==None:
        return fig,ax

    #plt.xticks(fontsize=ticks_fontsize,rotation=90)
    #plt.yticks(fontsize=ticks_fontsize)

def half_violin_boxplot(adata, keys, groupby, ax=None, figsize=(4,4), show=True, max_strip_points=1000):
    import colorcet
    from scipy.sparse import issparse  
    
    # 获取 y 数据
    y = np.asarray(get_values(adata, keys), dtype=float).reshape(-1)
    
    # 获取 x 数据
    x = adata.obs[groupby].values.reshape(-1)
    
    # 创建绘图数据
    plot_data = pd.DataFrame({groupby: x, keys: y})
    
    # 创建图形和轴
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # 获取或设置颜色
    from ._palette import palette_28,palette_56,palette_112,sc_color
    if f'{groupby}_colors' not in adata.uns or adata.uns[f'{groupby}_colors'] is None:
        #colors = ['#%02x%02x%02x' % tuple([int(k * 255) for k in i]) for i in colorcet.glasbey_bw_minc_20_maxl_70]
        if len(adata.obs[groupby].unique())>56:
            colors=palette_112
        elif len(adata.obs[groupby].unique())>28:
            colors=palette_56
        else:
            colors=sc_color
        adata.uns[f'{groupby}_colors'] = colors[:len(adata.obs[groupby].unique())]
    
    # 绘制小提琴图
    sns.violinplot(x=groupby, y=keys, data=plot_data, hue=groupby, dodge=False,
                   palette=adata.uns[f'{groupby}_colors'], scale="width", inner=None, ax=ax,
                   legend=False)
    
    # 调整小提琴图
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
    
    
    
    # 限制 stripplot 的数据点数量
    if len(plot_data) > max_strip_points:
        plot_data = plot_data.sample(max_strip_points)
    
    # 绘制 stripplot
    old_len_collections = len(ax.collections)
    sns.stripplot(x=groupby, y=keys, data=plot_data, hue=groupby,
                  palette=adata.uns[f'{groupby}_colors'], dodge=False, ax=ax,
                  )
    
    # 调整 stripplot 点的位置
    for dots in ax.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([0.12, 0]))

    # 绘制箱线图
    sns.boxplot(x=groupby, y=keys, data=plot_data, saturation=1, showfliers=False,
                width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax,
                )
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    #ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    #remove legend
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    #ax.legend().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    #

    if show:
        plt.show()
    
    return ax


def violin_box(adata, keys, groupby, ax=None, figsize=(4,4), show=True, max_strip_points=1000):
    warn(
        "`ov.pl.violin_box` is deprecated and will be removed in omicverse 2.2; "
        "use `ov.pl.half_violin_boxplot` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return half_violin_boxplot(
        adata,
        keys=keys,
        groupby=groupby,
        ax=ax,
        figsize=figsize,
        show=show,
        max_strip_points=max_strip_points,
    )


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

def dotplot_doublegroup(adata, gene, group1, group2, cmap='Reds',
                       standard_scale = 'group',figsize=(6, 4),layer=None):
    # 检查输入
    if gene not in adata.var_names:
        raise ValueError(f"Gene '{gene}' is not in the provided AnnData object.")
        
    if group1 not in adata.obs.columns:
        raise ValueError(f"Group '{group1}' is not in the provided AnnData object.")
        
    if group2 not in adata.obs.columns:
        raise ValueError(f"Group '{group2}' is not in the provided AnnData object.")
    
    # 将分组列转换为类别类型
    adata.obs[group1] = adata.obs[group1].astype('category')
    adata.obs[group2] = adata.obs[group2].astype('category')
    group1s = adata.obs[group1].cat.categories
    group2s = adata.obs[group2].cat.categories

    group_li_exp_mean_pd = pd.DataFrame(np.zeros((len(group1s), len(group2s))), index=group1s, columns=group2s)
    group_li_exp_size_pd = pd.DataFrame(np.zeros((len(group1s), len(group2s))), index=group1s, columns=group2s)

    for group1_ in group1s:
        adata1 = adata[adata.obs[group1] == group1_, [gene]]
        for group2_ in group2s:
            if layer is None:
                exp = adata1[adata1.obs[group2] == group2_, [gene]].to_df().values.reshape(-1)
            elif layer in adata1.layers.keys():
                exp = adata1[adata1.obs[group2] == group2_, [gene]].layers[layer].to_df().values.reshape(-1)
            else:
                raise ValueError(f"Layer '{layer}' is not in the provided AnnData object.")
            exp_larger_zero = exp[exp > 0]
            if len(exp) != 0:
                group_li_exp_size_pd.loc[group1_, group2_] = len(exp_larger_zero) / len(exp)
                group_li_exp_mean_pd.loc[group1_, group2_] = np.mean(exp)
            else:
                group_li_exp_size_pd.loc[group1_, group2_] = 0

    dot_color_df = group_li_exp_mean_pd
    
    if standard_scale == "group":
        dot_color_df = dot_color_df.sub(dot_color_df.min(1), axis=0)
        dot_color_df = dot_color_df.div(dot_color_df.max(1), axis=0).fillna(0)
    elif standard_scale == "var":
        dot_color_df -= dot_color_df.min(0)
        dot_color_df = (dot_color_df / dot_color_df.max(0)).fillna(0)
    else:
        pass
    
    # 设置常量
    dot_min = 0
    dot_max = 1
    size_exponent = 1
    largest_dot = 100
    smallest_dot = 10
    dot_edge_lw = 1
    size_title = 'Fraction of cells\nin group (%)'

    # 创建图形和网格布局
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax = fig.add_subplot(gs[0])

    # 绘制圆点
    for y, (label, row_mean) in enumerate(dot_color_df.iterrows()):
        for x, (column, value_mean) in enumerate(row_mean.items()):
            value_size = group_li_exp_size_pd.loc[label, column]
            size = value_size * 500  # 调整大小
            color = plt.get_cmap(cmap)(value_mean / dot_color_df.values.max())
            ax.scatter(x, y, s=size, color=color, alpha=1, 
                       edgecolors='w', linewidth=0.5)

    # 设置轴
    ax.set_xticks(range(len(group_li_exp_mean_pd.columns)))
    ax.set_xticklabels(group_li_exp_mean_pd.columns, rotation=45)
    ax.set_yticks(range(len(group_li_exp_mean_pd.index)))
    ax.set_yticklabels(group_li_exp_mean_pd.index)
    ax.set_title('Dot Plot')

    # 添加颜色图例
    ax1 = fig.add_subplot(gs[1])
    legend_gs = ax1.get_subplotspec().subgridspec(10, 1)

    color_legend_ax = fig.add_subplot(legend_gs[7])
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=group_li_exp_mean_pd.values.min(),
                                                                   vmax=group_li_exp_mean_pd.values.max()))
    plt.colorbar(mappable, cax=color_legend_ax, orientation="horizontal")
    color_legend_ax.set_title('Mean expression\nin group', fontsize="small")
    color_legend_ax.xaxis.set_tick_params(labelsize="small")

    # 添加点大小图例
    size_legend_ax = fig.add_subplot(legend_gs[2:5])
    diff = dot_max - dot_min
    step = 0.1 if 0.3 < diff <= 0.6 else 0.05 if diff <= 0.3 else 0.2

    size_range = np.arange(dot_max, dot_min, step * -1)[::-1]
    if dot_min != 0 or dot_max != 1:
        dot_range = dot_max - dot_min
        size_values = (size_range - dot_min) / dot_range
    else:
        size_values = size_range

    size = size_values ** size_exponent
    size = size * (largest_dot - smallest_dot) + smallest_dot

    size_legend_ax.scatter(
        np.arange(len(size)) + 0.5,
        np.repeat(0, len(size)),
        s=size * 5,
        color="gray",
        edgecolor="black",
        linewidth=dot_edge_lw,
        zorder=100,
    )
    size_legend_ax.set_xticks(np.arange(len(size)) + 0.5)
    labels = [f"{np.round((x * 100), decimals=0).astype(int)}" for x in size_range]
    size_legend_ax.set_xticklabels(labels, fontsize="small")

    size_legend_ax.tick_params(
        axis="y", left=False, labelleft=False, labelright=False
    )

    size_legend_ax.spines["right"].set_visible(False)
    size_legend_ax.spines["top"].set_visible(False)
    size_legend_ax.spines["left"].set_visible(False)
    size_legend_ax.spines["bottom"].set_visible(False)
    size_legend_ax.grid(visible=False)

    ymax = size_legend_ax.get_ylim()[1]
    size_legend_ax.set_ylim(-1.05 - largest_dot * 0.003, 4)
    size_legend_ax.set_title(size_title, y=ymax + 0.45, size="small")

    xmin, xmax = size_legend_ax.get_xlim()
    size_legend_ax.set_xlim(xmin - 0.15, xmax + 0.5)

    ax.grid(False)
    ax1.grid(False)
    ax1.axis(False)
    plt.show()


def add_arrow(ax,adata,basis,fontsize=12,
              x_label=None,y_label=None,
              arrow_scale=10,arrow_width=0.01):
    r"""
    Add arrow and label to the axis
    Args:
        ax: matplotlib.axes.Axes
            The axis to add the arrow and label to.
        adata: AnnData
            The AnnData object to add the arrow and label to.
        basis: str
            The basis to add the arrow and label to.
        fontsize: int
            The fontsize of the label.
        arrow_scale: float
            The scale of the arrow.
        arrow_width: float
            The width of the arrow.
        x_label: str
            The label of the x-axis.
        y_label: str
            The label of the y-axis.
    """
    if x_label is None:
        x_label=basis+'1'
    if y_label is None:
        y_label=basis+'2'

    obsm_key = basis if basis in adata.obsm else f"X_{basis}"
    coords = adata.obsm[obsm_key]

    x_range=(coords[:,0].max()-coords[:,0].min())/6
    y_range=(coords[:,1].max()-coords[:,1].min())/6
    x_min=coords[:,0].min()
    y_min=coords[:,1].min()
    ax.arrow(x=x_min-x_range/5,y=y_min,dx=x_range+x_range/arrow_scale,dy=0, 
            width=arrow_width, color="k", 
                head_width=y_range*2/arrow_scale, head_length=x_range*2/arrow_scale, overhang=0.5)

    ax.arrow(x=x_min,y=y_min-y_range/5,dx=0,dy=y_range+y_range/arrow_scale, 
            width=arrow_width, color="k", 
                head_width=x_range*2/arrow_scale, head_length=y_range*2/arrow_scale, overhang=0.5)
    ax.text(x=x_min,y=y_min-y_range/2,s=x_label,fontsize=fontsize,multialignment='center',
            verticalalignment='center')
    ax.text(x=x_min-x_range/2,y=y_min,s=y_label,fontsize=fontsize,rotation='vertical',multialignment='center',
           horizontalalignment='center')

"""Module providing encapsulation of SpatRio for spatial cell mapping."""
from typing import Any
import pandas as pd
import numpy as np
from .._registry import register_function


def _get_spatrio_functions():
    from ..external.spatrio.spatrio import assign_coord, ot_alignment

    return ot_alignment, assign_coord

@register_function(
    aliases=["CellMap", "空间映射", "细胞映射到空间", "spatrio_cellmap", "OT映射"],
    category="space",
    description="Map single-cell profiles to spatial spots with SpatRio optimal transport",
    prerequisites={
        "optional_functions": ["pp.pca", "pp.neighbors"]
    },
    requires={
        "obsm": ["X_pca", "spatial"]
    },
    produces={
        "obs": ["Cell_xcoord", "Cell_ycoord", "spot", "spot_type", "spot_value"],
        "obsm": ["spatial"]
    },
    auto_fix="none",
    examples=[
        "mapper = ov.space.CellMap(adata_sc, adata_sp, use_rep_sc='X_pca', use_rep_sp='X_pca')",
        "mapper.map(sc_type='celltype', sp_type='leiden')",
        "adata_sc_spatial = mapper.assign_coord()",
    ],
    related=["space.CellLoc", "space.Deconvolution", "space.pySTAligner"],
)
class CellMap(object):
    """
    SpatRio CellMap class for mapping single cells to spatial coordinates.

    This class implements optimal transport-based mapping of single cells to spatial
    coordinates using expression similarity and spatial awareness. It provides methods
    for both mapping and coordinate assignment.

    Parameters
    ----------
    adata_sc : anndata.AnnData
        Single-cell reference AnnData.
    adata_sp : anndata.AnnData
        Spatial AnnData to receive mapped cells.
    use_rep_sc : str, default="X_pca"
        Representation key in ``adata_sc.obsm`` used for transport.
    use_rep_sp : str, default="X_pca"
        Representation key in ``adata_sp.obsm`` used for transport.

    Attributes:
        adata_sc: AnnData
            Single-cell RNA sequencing data.
        adata_sp: AnnData
            Spatial transcriptomics data.
        use_rep_sc: str
            Representation key for single-cell data.
        use_rep_sp: str
            Representation key for spatial data.
        spatrio_decon: pandas.DataFrame
            Deconvolution results after mapping.
        spatrio_map: pandas.DataFrame
            Coordinate assignment results.

    Examples:
        >>> import omicverse as ov
        >>> # Load data
        >>> adata_sc = ov.read('single_cell.h5ad')
        >>> adata_sp = ov.read('spatial.h5ad')
        >>> # Initialize CellMap
        >>> cm = ov.space.CellMap(
        ...     adata_sc=adata_sc,
        ...     adata_sp=adata_sp,
        ...     use_rep_sc='X_pca'
        ... )
    """
    def __init__(self,
                    adata_sc,
                    adata_sp,
                    use_rep_sc='X_pca',
                    use_rep_sp='X_pca',
                 ) -> None:
        """Initialize CellMap with single-cell and spatial references.

        Parameters
        ----------
        adata_sc : anndata.AnnData
            Single-cell reference object to be projected into spatial coordinates.
        adata_sp : anndata.AnnData
            Spatial transcriptomics object providing spot locations.
        use_rep_sc : str, default="X_pca"
            Embedding key in ``adata_sc.obsm`` used for alignment.
        use_rep_sp : str, default="X_pca"
            Embedding key in ``adata_sp.obsm`` used for alignment.
        """
        self.adata_sc=adata_sc
        self.adata_sp=adata_sp
        self.use_rep_sc=use_rep_sc
        self.use_rep_sp=use_rep_sp
        self.spatrio_decon=None
        self.spatrio_map=None
        self.adata_sp.obsm['spatial']=pd.DataFrame(np.array(self.adata_sp.obsm['spatial']),
                                    columns=['x','y'],index=self.adata_sp.obs.index)
        self.adata_sc.obsm['reduction']=pd.DataFrame(self.adata_sc.obsm[self.use_rep_sc],
                                    index=self.adata_sc.obs.index)

    def map(self,
            sc_type: str = 'celltype',
            sp_type: str = 'leiden',
            alpha: float = 0.1,
            aware_power: int = 2,
            resolution: int = 1,
            aware_spatial: bool = True,
            aware_multi: bool = True,
            use_gpu: bool = True,
            **kwargs: Any
        ) -> pd.DataFrame:
        """Map single cells to spatial spots by optimal transport.

        Parameters
        ----------
        sc_type : str, default="celltype"
            Cell-type column in ``adata_sc.obs``. Set to ``None`` to disable
            cell-type-aware constraints.
        sp_type : str, default="leiden"
            Spatial domain column in ``adata_sp.obs``. If ``'leiden'`` or
            ``'louvain'`` is requested but missing, it will be computed.
        alpha : float, default=0.1
            Tradeoff between expression and spatial terms in transport cost.
        aware_power : int, default=2
            Power parameter for distance-based penalty.
        resolution : int, default=1
            Resolution for on-the-fly clustering when ``sp_type`` is missing.
        aware_spatial : bool, default=True
            Whether to incorporate spatial-domain prior during alignment.
        aware_multi : bool, default=True
            Whether to incorporate cell-type prior during alignment.
        use_gpu : bool, default=True
            Whether to use GPU in SpatRio backend.
        **kwargs : Any
            Additional options forwarded to ``ot_alignment``.

        Returns
        -------
        pandas.DataFrame
            Transport assignments with spot, cell and mapping score columns.
        """
        ot_alignment, _ = _get_spatrio_functions()

        ##spatial type
        if sp_type=='leiden' and 'leiden' not in self.adata_sp.obs.columns:
            from .._sc_compat import neighbors, leiden
            neighbors(self.adata_sp,n_neighbors=15,use_rep=self.use_rep_sp)
            leiden(self.adata_sp,resolution=resolution)
            self.adata_sp.obs['type']=self.adata_sp.obs[sp_type].tolist()
        elif sp_type=='louvain' and 'louvain' not in self.adata_sp.obs.columns:
            from .._sc_compat import neighbors
            from ..pp import louvain
            neighbors(self.adata_sp,n_neighbors=15,use_rep=self.use_rep_sp)
            louvain(self.adata_sp)
            self.adata_sp.obs['type']=self.adata_sp.obs[sp_type].tolist()
        elif sp_type is None:
            aware_spatial=False
        else:
            self.adata_sp.obs['type']=self.adata_sp.obs[sp_type].tolist()

        ##single cell type
        if sc_type is None:
            aware_multi=False
        else:
            self.adata_sc.obs['type']=self.adata_sc.obs[sc_type].tolist()

        spatrio_decon = ot_alignment(adata1 = self.adata_sp, adata2 = self.adata_sc,
                                     alpha = alpha, aware_power = aware_power,
                aware_spatial = aware_spatial,
                aware_multi=aware_multi,use_gpu=use_gpu,**kwargs)
        self.spatrio_decon=spatrio_decon
        return spatrio_decon

    def assign_coord(self,**kwargs):
        """Assign concrete spatial coordinates to mapped single cells.

        Parameters
        ----------
        **kwargs : Any
            Extra keyword arguments forwarded to SpatRio ``assign_coord``.

        Returns
        -------
        anndata.AnnData
            Filtered copy of ``adata_sc`` containing mapped cells with
            coordinates in ``obsm['spatial']`` and mapping metadata in ``obs``.
        """
        _, assign_coord = _get_spatrio_functions()
        spatrio_map = assign_coord(adata1 = self.adata_sp,adata2 = self.adata_sc,
                                   out_data = self.spatrio_decon,**kwargs)
        #self.adata_sp.obs=self.adata_sp.obs.join(spatrio_map.set_index('cell'))

        self.spatrio_map=spatrio_map

        loc1=pd.DataFrame(spatrio_map[['Cell_xcoord','Cell_ycoord','spot','spot_type',
                                       'value']].values,
                          columns=['Cell_xcoord','Cell_ycoord','spot','spot_type','value'],
                          index=spatrio_map['cell'].tolist())
        print('...assigning spatial coordinates to single cell data')
        adata_sc_copy=self.adata_sc.copy()
        adata_sc_copy=adata_sc_copy[loc1.index.tolist()]
        adata_sc_copy.obs['Cell_xcoord']=loc1.loc[adata_sc_copy.obs.index,'Cell_xcoord'].tolist()
        adata_sc_copy.obs['Cell_ycoord']=loc1.loc[adata_sc_copy.obs.index,'Cell_ycoord'].tolist()
        adata_sc_copy.obs['spot_type']=loc1.loc[adata_sc_copy.obs.index,'spot_type'].tolist()
        adata_sc_copy.obs['spot']=loc1.loc[adata_sc_copy.obs.index,'spot'].tolist()
        adata_sc_copy.obs['spot_value']=loc1.loc[adata_sc_copy.obs.index,'value'].tolist()

        print('...adding spatial coordinates to single cell data')
        adata_sc_copy.obsm['spatial']=loc1.loc[adata_sc_copy.obs.index,
                                               ['Cell_xcoord','Cell_ycoord']].values
        adata_sc_copy.uns['spatial']=self.adata_sp.uns['spatial'].copy()
        return adata_sc_copy
    # End-of-file (EOF)

@register_function(
    aliases=["CellLoc", "概率定位", "细胞空间定位", "spatrio_cellloc", "空间概率映射"],
    category="space",
    description="Probabilistic spatial localization of single cells with SpatRio",
    prerequisites={
        "optional_functions": ["pp.pca", "pp.neighbors"]
    },
    requires={
        "obsm": ["X_pca", "spatial"]
    },
    produces={
        "obs": ["Cell_xcoord", "Cell_ycoord", "spot", "spot_type", "spot_value"],
        "obsm": ["spatial"]
    },
    auto_fix="none",
    examples=[
        "loc = ov.space.CellLoc(adata_sc, adata_sp)",
        "loc.loc_map(sc_type='celltype', sp_type='leiden')",
        "adata_sc_spatial = loc.loc_assign()",
    ],
    related=["space.CellMap", "space.Deconvolution"],
)
class CellLoc(object):
    """
    SpatRio CellLoc class for probabilistic cell localization.

    This class extends CellMap with probabilistic filtering based on cell type
    proportions for more accurate spatial localization. It provides methods for
    mapping, saving/loading results, and probabilistic assignment.

    Parameters
    ----------
    adata_sc : anndata.AnnData
        Single-cell reference AnnData.
    adata_sp : anndata.AnnData
        Spatial AnnData to receive probabilistic localization.
    use_rep_sc : str, default="X_pca"
        Representation key in ``adata_sc.obsm`` used for transport.
    use_rep_sp : str, default="X_pca"
        Representation key in ``adata_sp.obsm`` used for transport.

    Attributes:
        adata_sc: AnnData
            Single-cell RNA sequencing data.
        adata_sp: AnnData
            Spatial transcriptomics data.
        use_rep_sc: str
            Representation key for single-cell data.
        use_rep_sp: str
            Representation key for spatial data.
        spatrio_decon: pandas.DataFrame
            Deconvolution results after mapping.
        spatrio_map: pandas.DataFrame
            Coordinate assignment results.

    Examples:
        >>> import omicverse as ov
        >>> # Load data
        >>> adata_sc = ov.read('single_cell.h5ad')
        >>> adata_sp = ov.read('spatial.h5ad')
        >>> # Initialize CellLoc
        >>> cl = ov.space.CellLoc(
        ...     adata_sc=adata_sc,
        ...     adata_sp=adata_sp
        ... )
    """
    def __init__(self,
                    adata_sc,
                    adata_sp,
                    use_rep_sc='X_pca',
                    use_rep_sp='X_pca',
                 ) -> None:
        """Initialize CellLoc for probabilistic cell localization.

        Parameters
        ----------
        adata_sc : anndata.AnnData
            Single-cell reference object to be localized.
        adata_sp : anndata.AnnData
            Spatial transcriptomics object containing spot coordinates.
        use_rep_sc : str, default="X_pca"
            Embedding key in ``adata_sc.obsm`` used for mapping.
        use_rep_sp : str, default="X_pca"
            Embedding key in ``adata_sp.obsm`` used for mapping.
        """
        self.adata_sc=adata_sc
        self.adata_sp=adata_sp
        self.use_rep_sc=use_rep_sc
        self.use_rep_sp=use_rep_sp
        self.spatrio_decon=None
        self.spatrio_map=None
        self.adata_sp.obsm['spatial']=pd.DataFrame(np.array(self.adata_sp.obsm['spatial']),
                                    columns=['x','y'],index=self.adata_sp.obs.index)
        self.adata_sc.obsm['reduction']=pd.DataFrame(self.adata_sc.obsm[self.use_rep_sc],
                                    index=self.adata_sc.obs.index)

    def loc_map(self,
            sc_type: str = 'celltype',
            sp_type: str = 'leiden',
            alpha: float = 0.1,
            aware_power: int = 2,
            resolution: int = 1,
            aware_spatial: bool = True,
            aware_multi: bool = True,
            use_gpu: bool = True,
            **kwargs: Any
        ) -> pd.DataFrame:
        """Run SpatRio mapping for probabilistic localization workflow.

        Parameters
        ----------
        sc_type : str, default="celltype"
            Cell-type column in ``adata_sc.obs``; set ``None`` to disable.
        sp_type : str, default="leiden"
            Spatial-domain column in ``adata_sp.obs``; computed if missing.
        alpha : float, default=0.1
            Tradeoff between expression and spatial constraints.
        aware_power : int, default=2
            Exponent for spatial-aware penalty term.
        resolution : int, default=1
            Clustering resolution when computing missing Leiden labels.
        aware_spatial : bool, default=True
            Whether to enforce spatial-domain prior.
        aware_multi : bool, default=True
            Whether to enforce cell-type prior.
        use_gpu : bool, default=True
            Whether to use GPU in backend alignment.
        **kwargs : Any
            Additional options passed to ``ot_alignment``.

        Returns
        -------
        pandas.DataFrame
            Raw cell-to-spot transport mapping table.
        """
        ot_alignment, _ = _get_spatrio_functions()

        ##spatial type
        if sp_type=='leiden' and 'leiden' not in self.adata_sp.obs.columns:
            from .._sc_compat import neighbors, leiden
            neighbors(self.adata_sp,n_neighbors=15,use_rep=self.use_rep_sp)
            leiden(self.adata_sp,resolution=resolution)
            self.adata_sp.obs['type']=self.adata_sp.obs[sp_type].tolist()
        elif sp_type=='louvain' and 'louvain' not in self.adata_sp.obs.columns:
            from .._sc_compat import neighbors
            from ..pp import louvain
            neighbors(self.adata_sp,n_neighbors=15,use_rep=self.use_rep_sp)
            louvain(self.adata_sp)
            self.adata_sp.obs['type']=self.adata_sp.obs[sp_type].tolist()
        elif sp_type is None:
            aware_spatial=False
        else:
            self.adata_sp.obs['type']=self.adata_sp.obs[sp_type].tolist()

        ##single cell type
        if sc_type is None:
            aware_multi=False
        else:
            self.adata_sc.obs['type']=self.adata_sc.obs[sc_type].tolist()

        spatrio_decon = ot_alignment(adata1 = self.adata_sp, adata2 = self.adata_sc,
                                     alpha = alpha, aware_power = aware_power,
                aware_spatial = aware_spatial,
                aware_multi=aware_multi,use_gpu=use_gpu,**kwargs)
        self.spatrio_decon=spatrio_decon
        return spatrio_decon

    def load_map(self,map_info,
                 sc_type: str = 'celltype',
                 sp_type: str = 'leiden',
                 resolution: int = 1,
                 aware_spatial: bool = True,
            aware_multi: bool = True,):
        """Load precomputed transport map into the CellLoc instance.

        Parameters
        ----------
        map_info : pandas.DataFrame
            Precomputed mapping table with spot/cell/value columns.
        sc_type : str, default="celltype"
            Cell-type column in ``adata_sc.obs`` for downstream filtering.
        sp_type : str, default="leiden"
            Spatial-domain column in ``adata_sp.obs`` for downstream filtering.
        resolution : int, default=1
            Resolution used when deriving missing cluster labels.
        aware_spatial : bool, default=True
            Indicates whether spatial prior should be considered.
        aware_multi : bool, default=True
            Indicates whether cell-type prior should be considered.

        Returns
        -------
        pandas.DataFrame
            Stored mapping table.
        """
        ##spatial type
        if sp_type=='leiden' and 'leiden' not in self.adata_sp.obs.columns:
            from .._sc_compat import neighbors, leiden
            neighbors(self.adata_sp,n_neighbors=15,use_rep=self.use_rep_sp)
            leiden(self.adata_sp,resolution=resolution)
            self.adata_sp.obs['type']=self.adata_sp.obs[sp_type].tolist()
        elif sp_type=='louvain' and 'louvain' not in self.adata_sp.obs.columns:
            from .._sc_compat import neighbors
            from ..pp import louvain
            neighbors(self.adata_sp,n_neighbors=15,use_rep=self.use_rep_sp)
            louvain(self.adata_sp)
            self.adata_sp.obs['type']=self.adata_sp.obs[sp_type].tolist()
        elif sp_type is None:
            aware_spatial=False
        else:
            self.adata_sp.obs['type']=self.adata_sp.obs[sp_type].tolist()

        ##single cell type
        if sc_type is None:
            aware_multi=False
        else:
            self.adata_sc.obs['type']=self.adata_sc.obs[sc_type].tolist()
        self.spatrio_decon=map_info
        return map_info

    def save_map(self,path):
        """Save current mapping table as CSV.

        Parameters
        ----------
        path : str
            Output CSV path.
        """
        self.spatrio_decon.to_csv(path)
    
    def loc_prob(self,spot_cell_prob,
                 sc_type: str = 'celltype',
                 sc_prop: float = 0.5,
                 n_cpu=6,):
        """Filter transport assignments by spot-level cell-type probabilities.

        Parameters
        ----------
        spot_cell_prob : pandas.DataFrame
            Spot-by-celltype probability matrix used as prior constraints.
        sc_type : str, default="celltype"
            Cell-type label column in ``adata_sc.obs``.
        sc_prop : float, default=0.5
            Minimum probability required for keeping a mapping.
        n_cpu : int, default=6
            Reserved CPU worker count parameter.

        Returns
        -------
        None
            Updates ``self.spatrio_decon`` in place after probabilistic filtering.
        """
        map_info=self.spatrio_decon.copy()
        from tqdm import tqdm
        df=spot_cell_prob

        # 创建字典
        first_max_dict = {}
        first_max_value_dict = {}
        
        # 找到每个索引得分第一高的列
        print('...finding spot prop')
        for idx in tqdm(df.index):
            series = df.loc[idx]
            sorted_series = series.sort_values(ascending=False)
            third_max_value = sorted_series.iloc[0]  # 第三大的值
            third_max_column = list(df)[list(series).index(third_max_value)]  # 对应的列名
            first_max_dict[idx] = third_max_column
            first_max_value_dict[idx]=third_max_value

        map_info['spot_prop1']=map_info['spot'].map(first_max_dict)
        map_info['spot_prop_value1']=map_info['spot'].map(first_max_value_dict)

        # 创建字典
        second_max_dict = {}
        second_max_value_dict = {}
        
        # 找到每个索引得分第二高的列
        print('...finding spot prop')
        for idx in tqdm(df.index):
            series = df.loc[idx]
            sorted_series = series.sort_values(ascending=False)
            third_max_value = sorted_series.iloc[1]  # 第三大的值
            third_max_column = list(df)[list(series).index(third_max_value)]  # 对应的列名
            second_max_dict[idx] = third_max_column
            second_max_value_dict[idx]=third_max_value
        map_info['spot_prop2']=map_info['spot'].map(second_max_dict)
        map_info['spot_prop_value2']=map_info['spot'].map(second_max_value_dict)

        # 创建字典
        third_max_dict = {}
        third_max_value_dict = {}

        # 找到每个索引得分第三高的列
        print('...finding spot prop')
        for idx in tqdm(df.index):
            series = df.loc[idx]
            sorted_series = series.sort_values(ascending=False)
            third_max_value = sorted_series.iloc[2]  # 第三大的值
            third_max_column = list(df)[list(series).index(third_max_value)]  # 对应的列名
            third_max_dict[idx] = third_max_column
            third_max_value_dict[idx]=third_max_value

     
        map_info['spot_prop3']=map_info['spot'].map(third_max_dict)
        map_info['spot_prop_value3']=map_info['spot'].map(third_max_value_dict)

        print('...adding spot prop to map info')

        map_info['cell_type']=map_info['cell'].map(dict(zip(
            self.adata_sc.obs.index.tolist(),
            self.adata_sc.obs[sc_type].tolist()   
        )))
        print('...adding cell type to map info')

        cond1 = (map_info['cell_type'] == map_info['spot_prop1']) & (map_info['spot_prop_value1'] > sc_prop)
        print('... finish spot type 1 filter')
        cond2 = (map_info['cell_type'] == map_info['spot_prop2']) & (map_info['spot_prop_value2'] > sc_prop)
        print('... finish spot type 2 filter')
        cond3 = (map_info['cell_type'] == map_info['spot_prop3']) & (map_info['spot_prop_value3'] > sc_prop)
        print('... finish spot type 3 filter')
        mask = cond1 | cond2 | cond3

        print('...filtering map info')
        # 应用过滤条件 ----------------------------------------------------------
        map_info.loc[~mask, 'value'] = -1
        map_info=map_info[map_info['value']!=-1]
        self.spatrio_decon = map_info[['spot', 'cell', 'value']]



    def loc_assign(self,**kwargs):
        """Assign coordinates to filtered CellLoc mappings.

        Parameters
        ----------
        **kwargs : Any
            Additional options forwarded to SpatRio ``assign_coord``.

        Returns
        -------
        anndata.AnnData
            Single-cell AnnData with mapped spatial coordinates and spot metadata.
        """
        _, assign_coord = _get_spatrio_functions()
        spatrio_map = assign_coord(adata1 = self.adata_sp,adata2 = self.adata_sc,
                                   out_data = self.spatrio_decon,**kwargs)
        #self.adata_sp.obs=self.adata_sp.obs.join(spatrio_map.set_index('cell'))

        self.spatrio_map=spatrio_map

        loc1=pd.DataFrame(spatrio_map[['Cell_xcoord','Cell_ycoord','spot','spot_type',
                                       'value']].values,
                          columns=['Cell_xcoord','Cell_ycoord','spot','spot_type','value'],
                          index=spatrio_map['cell'].tolist())
        print('...assigning spatial coordinates to single cell data')
        adata_sc_copy=self.adata_sc.copy()
        adata_sc_copy=adata_sc_copy[loc1.index.tolist()]
        adata_sc_copy.obs['Cell_xcoord']=loc1.loc[adata_sc_copy.obs.index,'Cell_xcoord'].tolist()
        adata_sc_copy.obs['Cell_ycoord']=loc1.loc[adata_sc_copy.obs.index,'Cell_ycoord'].tolist()
        adata_sc_copy.obs['spot_type']=loc1.loc[adata_sc_copy.obs.index,'spot_type'].tolist()
        adata_sc_copy.obs['spot']=loc1.loc[adata_sc_copy.obs.index,'spot'].tolist()
        adata_sc_copy.obs['spot_value']=loc1.loc[adata_sc_copy.obs.index,'value'].tolist()

        print('...adding spatial coordinates to single cell data')
        adata_sc_copy.obsm['spatial']=loc1.loc[adata_sc_copy.obs.index,
                                               ['Cell_xcoord','Cell_ycoord']].values
        adata_sc_copy.uns['spatial']=self.adata_sp.uns['spatial'].copy()
        adata_sc_copy.obsm['spatial']=adata_sc_copy.obsm['spatial'].astype(float)
        del adata_sc_copy.obsm['reduction']
        
        return adata_sc_copy

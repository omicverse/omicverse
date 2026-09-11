from .._settings import add_reference, EMOJI, Colors
import numpy as np
import scanpy as sc
import pandas as pd

from .._registry import register_function


def _validate_count_layer(adata, requested_layer, label, method='cell2location'):
    """Resolve a count layer and reject continuous/invalid count inputs."""
    from scipy import sparse

    if requested_layer is not None and requested_layer in adata.layers:
        layer = requested_layer
        matrix = adata.layers[layer]
    elif requested_layer in (None, 'counts'):
        layer = None
        matrix = adata.X
    else:
        raise KeyError(
            f"Requested {label} count layer {requested_layer!r} was not found."
        )

    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    values = np.asarray(values)
    for start in range(0, values.size, 1_000_000):
        chunk = values[start:start + 1_000_000]
        if not np.isfinite(chunk).all():
            raise ValueError(f"{label} count input contains NaN or infinite values.")
        if np.any(chunk < 0):
            raise ValueError(f"{label} count input contains negative values.")
        if not np.allclose(chunk, np.rint(chunk), rtol=0, atol=1e-6):
            source = f"adata.layers[{layer!r}]" if layer is not None else "adata.X"
            layer_hint = (
                f" and pass counts_layer_{'sc' if label == 'reference' else 'sp'}="
                "'your_layer'"
                if method == 'cell2location'
                else ""
            )
            raise ValueError(
                f"{method} requires raw integer-like counts for {label}; {source} "
                f"contains continuous values. Preserve raw counts in a layer{layer_hint}."
            )
    return layer


def _starfysh_visium_metadata(adata):
    """Build Starfysh metadata from canonical Visium ``(x, y)`` coordinates."""
    spatial_mapping = adata.uns.get('spatial')
    if not isinstance(spatial_mapping, dict) or not spatial_mapping:
        raise KeyError("Starfysh requires a non-empty adata.uns['spatial'] mapping.")
    if len(spatial_mapping) != 1:
        raise ValueError(
            "The Starfysh adapter currently accepts one Visium library at a time. "
            "Subset the requested library before deconvolution."
        )
    sample_id = next(iter(spatial_mapping))
    if 'spatial' not in adata.obsm:
        raise KeyError("Starfysh requires canonical coordinates in adata.obsm['spatial'].")
    coords = np.asarray(adata.obsm['spatial'], dtype=np.float64)
    if coords.ndim != 2 or coords.shape != (adata.n_obs, 2):
        raise ValueError("adata.obsm['spatial'] must have shape (n_obs, 2).")
    if not np.isfinite(coords).all():
        raise ValueError("adata.obsm['spatial'] contains non-finite coordinates.")
    required_obs = ['array_row', 'array_col']
    missing_obs = [column for column in required_obs if column not in adata.obs]
    if missing_obs:
        raise KeyError(f"Starfysh Visium metadata is missing obs columns: {missing_obs}.")

    spatial_info = spatial_mapping[sample_id]
    images = spatial_info.get('images', {})
    scalefactors = spatial_info.get('scalefactors')
    if 'hires' not in images or scalefactors is None:
        raise KeyError(
            "Starfysh requires a hires image and Visium scalefactors in "
            "adata.uns['spatial'][library_id]."
        )

    map_info = pd.DataFrame(
        {
            'array_row': adata.obs['array_row'].to_numpy(),
            'array_col': adata.obs['array_col'].to_numpy(),
            # AnnData/Visium convention: spatial[:, 0] is x/image column and
            # spatial[:, 1] is y/image row. Starfysh multiplies these full-res
            # coordinates by tissue_hires_scalef when extracting image patches.
            'imagerow': coords[:, 1],
            'imagecol': coords[:, 0],
            'sample': sample_id,
        },
        index=adata.obs_names.copy(),
    )
    return sample_id, {
        'img': images['hires'],
        'scalefactor': scalefactors,
        'map_info': map_info,
    }


@register_function(
    aliases=["空间解卷积", "spatial deconvolution", "Deconvolution", "cell type mapping", "空间细胞类型映射"],
    category="space",
    description="Class for transferring single-cell cell-type information onto spatial transcriptomics spots using Tangram/cell2location/Starfysh/FlashDeconv/RCTD backends.",
    prerequisites={
        'optional_functions': ['pp.preprocess', 'space.svg']
    },
    requires={
        'obs': ['reference cell-type labels in adata_sc.obs'],
        'layers': ['counts (recommended for both adata_sc and adata_sp)']
    },
    produces={
        'obsm': ['prop_celltypes / method-specific proportion matrices'],
        'layers': ['cell-type-specific imputed layers (cell2location mode)'],
        'uns': ['method-specific model metadata']
    },
    auto_fix='escalate',
    examples=[
        "decov = ov.space.Deconvolution(adata_sp=adata_sp, adata_sc=adata_sc)",
        "decov.preprocess_sc(mode='shiftlog|pearson', n_HVGs=3000)",
        "decov.preprocess_sp(mode='pearsonr', n_svgs=3000)",
        "decov.deconvolution(method='Tangram', celltype_key_sc='cell_type')",
        "decov.deconvolution(method='cell2location', celltype_key_sc='cell_type')",
        "decov.deconvolution(method='RCTD', celltype_key_sc='cell_type')",
    ],
    related=['space.calculate_gene_signature', 'space.svg', 'single.get_celltype_marker']
)
class Deconvolution(object):
    """
    Spatial deconvolution pipeline that aligns scRNA-seq references with spatial transcriptomics.
    
    Parameters
    ----------
    adata_sp:AnnData
        Spatial transcriptomics AnnData object (spots x genes).
    adata_sc:AnnData or None
        Single-cell reference AnnData (cells x genes) containing cell-type labels.
    
    Returns
    -------
    None
        Initializes the deconvolution manager and backend placeholders.
    
    Examples
    --------
    >>> decov = ov.space.Deconvolution(adata_sp=adata_sp, adata_sc=adata_sc)
    >>> decov.deconvolution(method='Tangram', celltype_key_sc='cell_type')
    """

    def __init__(self, adata_sp,adata_sc=None, ):
        r"""Initialize Deconvolution object.
        Parameters
        ----------
        adata_sp:AnnData
            Spatial transcriptomics data.
        adata_sc:AnnData or None
            Single-cell RNA-seq reference. If ``None``, only spatial object is initialized.
        """
        self.adata_sc = adata_sc
        self.adata_sp = adata_sp
        self.mod_sc=None
        self.mod_sp=None
        self.adata_cell2location=None
        self.adata_impute=None
        self.tangram_model=None
        self.flashdeconv_params=None
        self.method=None
        if adata_sc is not None:
            self._check_layer()


    def _check_layer(self):

        #Check the raw counts layer
        if 'counts' in self.adata_sc.layers.keys():
            print(f"{Colors.GREEN}[OK] Existing 'counts' layer in scRNA-seq data{Colors.ENDC}")
        if 'counts' in self.adata_sp.layers.keys():
            print(f"{Colors.GREEN}[OK] Existing 'counts' layer in spatial transcriptomics data{Colors.ENDC}")

        # Check the normalized expression layer 
        log1p_1e4_warning = False
        if self.adata_sc.X.max()<np.log1p(1e4):
            print(f"{Colors.GREEN}[OK] scRNA-seq data appears log-normalized{Colors.ENDC}")
            log1p_1e4_warning = True
        if self.adata_sp.X.max()<np.log1p(1e4):
            print(f"{Colors.GREEN}[OK] spatial transcriptomics data appears log-normalized{Colors.ENDC}")
            log1p_1e4_warning = True
        
        if log1p_1e4_warning:   
            print(f"{Colors.WARNING}[WARN] 1e4 is a common target sum for `scanpy`{Colors.ENDC}")

        log1p_50_warning = False
        if self.adata_sc.X.max()<np.log1p(50*1e4):
            print(f"{Colors.GREEN}[OK] scRNA-seq values are within the expected numeric range{Colors.ENDC}")
            log1p_50_warning = True
        if self.adata_sp.X.max()<np.log1p(50*1e4):
            print(f"{Colors.GREEN}[OK] spatial values are within the expected numeric range{Colors.ENDC}")
            log1p_50_warning = True

        if log1p_50_warning:   
            print(f"{Colors.WARNING}[WARN] 50*1e4 is a common OmicVerse target sum{Colors.ENDC}")

    @staticmethod
    def _default_cell2location_device_kwargs():
        try:
            import torch
        except ImportError:
            return {}

        if torch.cuda.is_available():
            return {"accelerator": "gpu", "device": 1}
        return {}

    def _with_cell2location_device_defaults(self, train_kwargs):
        train_kwargs = {} if train_kwargs is None else dict(train_kwargs)
        device_keys = {"accelerator", "device", "devices", "use_gpu"}
        if not any(key in train_kwargs for key in device_keys):
            train_kwargs.update(self._default_cell2location_device_kwargs())
        return train_kwargs

    def preprocess_sc(self,mode='shiftlog|pearson',n_HVGs=3000,target_sum=1e4,**kwargs):
        """
        Preprocess the scRNA-seq reference before spatial mapping.

        Parameters
        ----------
        mode:str
            Preprocessing recipe used by ``ov.pp.preprocess``.
        n_HVGs:int
            Number of highly variable genes to retain.
        target_sum:float
            Library-size normalization target sum.
        **kwargs
            Additional keyword arguments passed to ``ov.pp.preprocess``.

        Returns
        -------
        None
            Updates ``self.adata_sc`` in-place and subsets to HVGs.

        Examples
        --------
        >>> decov.preprocess_sc(mode='shiftlog|pearson', n_HVGs=3000, target_sum=1e4)
        """
        from ..pp import preprocess
        self.adata_sc=preprocess(self.adata_sc,mode=mode,n_HVGs=n_HVGs,target_sum=target_sum,**kwargs)
        self.adata_sc.raw = self.adata_sc
        self.adata_sc = self.adata_sc[:, self.adata_sc.var.highly_variable_features]
        print(f"{Colors.GREEN}[OK] scRNA-seq data is preprocessed{Colors.ENDC}")
        #return self.adata_sc

    def preprocess_sp(self,
        mode='pearsonr',n_svgs=3000,target_sum=50*1e4,
        platform="visium",mt_startwith='MT-',
        subset_genes=True,
        **kwargs):
        """
        Preprocess spatial transcriptomics data and select spatially variable genes.

        Parameters
        ----------
        mode:str
            SVG selection mode passed to ``ov.space.svg``.
        n_svgs:int
            Number of spatially variable genes to keep.
        target_sum:float
            Library-size normalization target sum for spatial data.
        platform:str
            Spatial platform identifier.
        mt_startwith:str
            Prefix used to identify mitochondrial genes.
        subset_genes:bool
            Whether to subset ``self.adata_sp`` to selected SVGs.
        **kwargs
            Additional keyword arguments forwarded to ``ov.space.svg``.

        Returns
        -------
        None
            Updates ``self.adata_sp`` and ``self.adata_sp_raw`` in-place.

        Examples
        --------
        >>> decov.preprocess_sp(mode='pearsonr', n_svgs=3000, platform='visium')
        """

        from ._svg import svg
        self.adata_sp_raw=self.adata_sp.copy()
        self.adata_sp=svg(self.adata_sp,mode=mode,n_svgs=n_svgs,
                    target_sum=target_sum,platform=platform,
                    mt_startwith=mt_startwith,**kwargs)
        both_genes=np.intersect1d(self.adata_sp_raw.var_names, self.adata_sp.var_names)
        self.adata_sp_raw=self.adata_sp_raw[:, both_genes]
        self.adata_sp=self.adata_sp[:, both_genes]
        self.adata_sp_raw.var['space_variable_features']=self.adata_sp.var['space_variable_features']
        self.adata_sp_raw.var['highly_variable']=self.adata_sp.var['highly_variable']
        self.adata_sp.raw = self.adata_sp
        self.adata_sp.var['highly_variable']=self.adata_sp.var['space_variable_features']
        if subset_genes:
            self.adata_sp = self.adata_sp[:, self.adata_sp.var.space_variable_features]
        else:
            self.adata_sp = self.adata_sp
        print(f"{Colors.GREEN}[OK] spatial transcriptomics data is preprocessed{Colors.ENDC}")
        #return self.adata_sp

    def deconvolution(
        self,
        method='Tangram',
        celltype_key_sc='cell_type',
        batch_key_sc=None,
        batch_key_sp=None,
        tangram_kwargs=None,
        cell2location_scrna_kwargs=None,
        cell2location_spatial_kwargs=None,
        N_cells_per_location=30,
        detection_alpha=200,
        sample_kwargs=None,
        # FlashDeconv parameters
        flashdeconv_kwargs=None,
        starfysh_kwargs=None,
        # RCTD parameters
        rctd_kwargs=None,
        spatial_type='visium',
        gene_sig=None,
        categorical_covariate_keys_sc=None,
        counts_layer_sc='counts',
        counts_layer_sp='counts',
    ):
        """
        Infer spot-level cell-type composition from single-cell references.

        Parameters
        ----------
        method:{'Tangram', 'cell2location', 'FlashDeconv', 'starfysh', 'RCTD'}
            Deconvolution backend.
        celltype_key_sc:str
            Cell-type label key in ``adata_sc.obs``.
        batch_key_sc:str or None
            Batch key in scRNA-seq reference for batch-aware models.
        batch_key_sp:str or None
            Batch key in spatial data.
        tangram_kwargs:dict or None
            Keyword arguments for Tangram model training.
        cell2location_scrna_kwargs:dict or None
            Keyword arguments for cell2location reference regression model.
        cell2location_spatial_kwargs:dict or None
            Keyword arguments for cell2location spatial model.
        N_cells_per_location:int
            Expected number of cells per spatial location (cell2location).
        detection_alpha:float
            Detection alpha hyper-parameter used by cell2location.
        sample_kwargs:dict or None
            Posterior sampling options for cell2location.
        counts_layer_sc, counts_layer_sp:str or None
            Raw count layers for the single-cell reference and spatial object.
            If the requested ``'counts'`` layer is absent, ``.X`` is accepted
            only when it is finite, non-negative, and integer-like.
        flashdeconv_kwargs:dict or None
            Additional parameters for FlashDeconv.
        starfysh_kwargs:dict or None
            Additional parameters for Starfysh.
        rctd_kwargs:dict or None
            Additional parameters for RCTD. Recognised keys: ``mode``
            (``'full'`` / ``'doublet'`` / ``'multi'``, default ``'full'``),
            ``cell_min``, ``n_max_cells``, ``min_UMI`` (forwarded to
            ``rctd.Reference``), ``batch_size``, ``sigma_override``
            (forwarded to ``rctd.run_rctd``), and ``config``
            (a dict of ``RCTDConfig`` overrides).
        spatial_type:str
            Spatial platform type used by backend-specific wrappers.
        gene_sig:pandas.DataFrame or None
            Gene-signature table used by Starfysh.
        categorical_covariate_keys_sc:list[str] or None
            Categorical covariates used by cell2location regression.

        Returns
        -------
        object
            Backend-specific trained model object for methods that explicitly return one
            (for example Tangram); otherwise results are stored in object attributes.

        Examples
        --------
        >>> decov.deconvolution(method='Tangram', celltype_key_sc='cell_type')
        >>> decov.deconvolution(method='cell2location', celltype_key_sc='cell_type')
        >>> decov.deconvolution(method='RCTD', celltype_key_sc='cell_type',
        ...                     rctd_kwargs={'mode': 'full'})
        """
        canonical_methods = {
            'tangram': 'Tangram',
            'cell2location': 'cell2location',
            'flashdeconv': 'FlashDeconv',
            'starfysh': 'starfysh',
            'rctd': 'RCTD',
        }
        normalized_method = str(method).lower()
        if normalized_method not in canonical_methods:
            raise ValueError(
                f"Method {method!r} is not supported. Choose from: "
                f"{list(canonical_methods.values())}."
            )
        method = canonical_methods[normalized_method]

        if method=='Tangram':
            self.method='Tangram'
            from ._tangram import Tangram
            if tangram_kwargs is None:
                tangram_kwargs={'mode':'clusters','num_epochs':500,'device':'cuda:0'}
            tangram=Tangram(self.adata_sc,self.adata_sp,clusters=celltype_key_sc)
            tangram.train(**tangram_kwargs)
            self.adata_cell2location=tangram.cell2location()
            print(f"{Colors.GREEN}[OK] Tangram projection is done{Colors.ENDC}")
            print(f"The cell2location result is saved in self.adata_cell2location")
            #self.adata_impute=tangram.impute()
            self.tangram_model=tangram
            #print(f"{Colors.GREEN}✓ Tangram impute is done{Colors.ENDC}")
            #print(f"The impute result is saved in self.adata_impute")
            return tangram
        elif method=='cell2location':
            self.method='cell2location'
            from ..external.space.cell2location.models import Cell2location, RegressionModel
            from ..external.space.cell2location.plt import plot_spatial
            from ..external.space.cell2location.utils import select_slide
            from ..external.space.cell2location.utils.filtering import filter_genes

            reference_count_layer = _validate_count_layer(
                self.adata_sc,
                counts_layer_sc,
                'reference',
            )
            spatial_count_layer = _validate_count_layer(
                self.adata_sp,
                counts_layer_sp,
                'spatial',
            )

            original_reference_x = self.adata_sc.X
            if reference_count_layer is not None:
                self.adata_sc.X = self.adata_sc.layers[reference_count_layer]
            try:
                selected = filter_genes(
                    self.adata_sc,
                    cell_count_cutoff=5,
                    cell_percentage_cutoff2=0.03,
                    nonz_mean_cutoff=1.12,
                )
            finally:
                self.adata_sc.X = original_reference_x

            # filter the object
            self.adata_sc = self.adata_sc[:, selected].copy()

            # prepare anndata for the regression model
            RegressionModel.setup_anndata(
                adata=self.adata_sc,
                layer=reference_count_layer,
                # 10X reaction / sample / batch
                batch_key=batch_key_sc,
                # cell type, covariate used for constructing signatures
                labels_key=celltype_key_sc,
                # multiplicative technical effects (platform, 3' vs 5', donor effect)
                categorical_covariate_keys=categorical_covariate_keys_sc,
            )
            self.mod_sc = RegressionModel(self.adata_sc)

            # Use all data for training (validation not implemented yet, train_size=1)
            if cell2location_scrna_kwargs is None:
                cell2location_scrna_kwargs={'max_epochs':250,'batch_size':2500,'train_size':1,'lr':0.002}
            cell2location_scrna_kwargs = self._with_cell2location_device_defaults(cell2location_scrna_kwargs)
            self.mod_sc.train(
                **cell2location_scrna_kwargs
            )

            if sample_kwargs is None:
                sample_kwargs={"num_samples": 1000, "batch_size": 2500}

            self.adata_sc = self.mod_sc.export_posterior(
                self.adata_sc,
                sample_kwargs=sample_kwargs,
            )


            # export estimated expression in each cluster
            if "means_per_cluster_mu_fg" in self.adata_sc.varm.keys():
                inf_aver = self.adata_sc.varm["means_per_cluster_mu_fg"][
                    [f"means_per_cluster_mu_fg_{i}" for i in self.adata_sc.uns["mod"]["factor_names"]]
                ].copy()
            else:
                inf_aver = self.adata_sc.var[
                    [f"means_per_cluster_mu_fg_{i}" for i in self.adata_sc.uns["mod"]["factor_names"]]
                ].copy()
            inf_aver.columns = self.adata_sc.uns["mod"]["factor_names"]

            self.inf_aver=inf_aver

            intersect = np.intersect1d(self.adata_sp.var_names, inf_aver.index)
            self.adata_sp = self.adata_sp[:, intersect].copy()
            self.inf_aver = self.inf_aver.loc[intersect, :].copy()

            print(f"Total number of genes both in the scRNA-seq data and the spatial transcriptomics data: {len(intersect)}")

            Cell2location.setup_anndata(
                adata=self.adata_sp,
                layer=spatial_count_layer,
                batch_key=batch_key_sp,
            )


            self.mod_sp = Cell2location(
                self.adata_sp,
                cell_state_df=self.inf_aver,
                # the expected average cell abundance: tissue-dependent
                # hyper-prior which can be estimated from paired histology:
                N_cells_per_location=N_cells_per_location,
                # hyperparameter controlling normalisation of
                # within-experiment variation in RNA detection (using default here):
                detection_alpha=detection_alpha,
            )

            if cell2location_spatial_kwargs is None:
                cell2location_spatial_kwargs={'max_epochs':30000,'batch_size':None,'train_size':1}
            cell2location_spatial_kwargs = self._with_cell2location_device_defaults(cell2location_spatial_kwargs)

            self.mod_sp.train(
                **cell2location_spatial_kwargs
            )

            self.adata_sp = self.mod_sp.export_posterior(
                self.adata_sp,
                sample_kwargs=sample_kwargs,
            )

            # 3) 绝对丰度 → 比例
            abund = self.adata_sp.obsm['q05_cell_abundance_w_sf']
            self.adata_sp.obsm['prop_celltypes'] = abund.div(abund.sum(axis=1).clip(lower=1e-9), axis=0)

            adata_cell2location=sc.AnnData(self.adata_sp.obsm['prop_celltypes'])
            adata_cell2location.var.index=self.adata_sp.uns["mod"]["factor_names"]
            adata_cell2location.var_names=self.adata_sp.uns["mod"]["factor_names"]
            
            adata_cell2location.obsm=self.adata_sp.obsm.copy()
            adata_cell2location.obs=self.adata_sp.obs.copy()
            adata_cell2location.obsp=self.adata_sp.obsp.copy()
            adata_cell2location.uns=self.adata_sp.uns.copy()
            self.adata_cell2location=adata_cell2location

            print(f"{Colors.GREEN}[OK] cell2location is done{Colors.ENDC}")
            print(f"The cell2location result is saved in self.adata_cell2location")

        elif method=='FlashDeconv':
            self.method='FlashDeconv'
            try:
                import flashdeconv as fd
            except ImportError:
                raise ImportError(
                    "FlashDeconv is not installed. Install it with: pip install flashdeconv"
                )

            # Set default kwargs for FlashDeconv
            if flashdeconv_kwargs is None:
                flashdeconv_kwargs = {
                    'sketch_dim': 512,
                    'lambda_spatial': 5000.0,
                    'n_hvg': 2000,
                    'n_markers_per_type': 50,
                }

            # Determine spatial coordinate key
            spatial_key = 'spatial'
            if spatial_key not in self.adata_sp.obsm:
                # Try alternative keys
                for alt_key in ['X_spatial', 'spatial_coords']:
                    if alt_key in self.adata_sp.obsm:
                        spatial_key = alt_key
                        break

            # Run FlashDeconv deconvolution
            print(f"Running FlashDeconv with parameters: {flashdeconv_kwargs}")
            fd.tl.deconvolve(
                self.adata_sp,
                self.adata_sc,
                cell_type_key=celltype_key_sc,
                spatial_key=spatial_key,
                key_added='flashdeconv',
                **flashdeconv_kwargs
            )

            # Convert results to match omicverse API format
            # Create adata_cell2location from the deconvolution results
            proportions_df = self.adata_sp.obsm['flashdeconv']
            adata_cell2location = sc.AnnData(proportions_df)
            adata_cell2location.var_names = proportions_df.columns.tolist()
            adata_cell2location.obs_names = proportions_df.index.tolist()

            # Copy spatial information
            adata_cell2location.obsm = self.adata_sp.obsm.copy()
            adata_cell2location.obs = self.adata_sp.obs.copy()
            if hasattr(self.adata_sp, 'obsp') and self.adata_sp.obsp:
                adata_cell2location.obsp = self.adata_sp.obsp.copy()
            adata_cell2location.uns = self.adata_sp.uns.copy()

            self.adata_cell2location = adata_cell2location
            self.flashdeconv_params = self.adata_sp.uns.get('flashdeconv_params', flashdeconv_kwargs)

            print(f"{Colors.GREEN}[OK] FlashDeconv deconvolution is done{Colors.ENDC}")
            print(f"The deconvolution result is saved in self.adata_cell2location")
            print(f"Cell type proportions are also stored in self.adata_sp.obsm['flashdeconv']")

        elif method=='starfysh':
            if spatial_type != 'visium':
                raise NotImplementedError(
                    "The Starfysh adapter currently supports only spatial_type='visium'."
                )
            if gene_sig is None:
                raise ValueError("Starfysh requires a gene signature via `gene_sig=`.")
            from ..external.starfysh import (AA, utils, plot_utils, post_analysis)
            from ..external.starfysh import _starfysh as sf_model
            self.method='starfysh'

            import torch

            starfysh_default_kwargs={
                'n_repeats':3,
                'epochs':200,
                'device':None,
                'batch_size':32,
                'alpha_mul':50,
                'lr':1e-4,
                'poe':False,
                'verbose':True,
                'n_anchors':60,
                'window_size':3,
            }

            #check the starfysh_kwargs if ignore the None values
            if starfysh_kwargs is None:
                starfysh_kwargs=starfysh_default_kwargs
            else:
                # Update default kwargs with user provided kwargs
                # This ensures all keys exist, using defaults for missing ones
                # and user values for provided ones
                full_kwargs = starfysh_default_kwargs.copy()
                full_kwargs.update(starfysh_kwargs)
                starfysh_kwargs = full_kwargs

            if spatial_type=='visium':
                sample_id, img_metadata = _starfysh_visium_metadata(self.adata_sp)
                # Prepare raw counts adata for starfysh (expects raw counts as first arg)
                if 'counts' in self.adata_sp.layers:
                    adata_raw = self.adata_sp.copy()
                    adata_raw.X = adata_raw.layers['counts'].copy()
                else:
                    adata_raw = self.adata_sp.copy()

                # Parameters for training
                visium_args = utils.VisiumArguments(adata_raw,
                                                    self.adata_sp,
                                                    gene_sig,
                                                    img_metadata,
                                                    n_anchors=starfysh_kwargs['n_anchors'],
                                                    window_size=starfysh_kwargs['window_size'],
                                                    sample_id=sample_id
                                                )

                adata, adata_normed = visium_args.get_adata()
                anchors_df = visium_args.get_anchors()
                adata.obs['log library size']=visium_args.log_lib
                adata.obs['windowed log library size']=visium_args.win_loglib

                aa_model = AA.ArchetypalAnalysis(adata_orig=adata_normed)
                archetype, arche_dict, major_idx, evs = aa_model.compute_archetypes(cn=40)
                # (1). Find archetypal spots & archetypal clusters
                arche_df = aa_model.find_archetypal_spots(major=True)

                # (2). Find marker genes associated with each archetypal cluster
                markers_df = aa_model.find_markers(n_markers=30, display=False)

                # (3). Map archetypes to closest anchors (1-1 per cell type)
                map_df, map_dict = aa_model.assign_archetypes(anchors_df)

                # (4). Optional: Find the most distant archetypes that are not assigned to any annotated cell types
                distant_arches = aa_model.find_distant_archetypes(anchors_df, n=3)

                visium_args = utils.refine_anchors(
                    visium_args,
                    aa_model,
                    #thld=0.7,  # alignment threshold
                    n_genes=5,
                    #n_iters=1
                )

                # Get updated adata & signatures
                adata, adata_normed = visium_args.get_adata()
                gene_sig = visium_args.gene_sig
                cell_types = gene_sig.columns

                

                n_repeats = starfysh_kwargs['n_repeats']
                epochs = starfysh_kwargs['epochs']
                device = starfysh_kwargs['device']
                if device is None:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                batch_size = starfysh_kwargs['batch_size']
                alpha_mul = starfysh_kwargs['alpha_mul']
                lr = starfysh_kwargs['lr']
                poe = starfysh_kwargs['poe']
                verbose = starfysh_kwargs['verbose']
                # Run models
                model, loss = utils.run_starfysh(visium_args,
                                                n_repeats=n_repeats,
                                                epochs=epochs,
                                                device=device,
                                                batch_size=batch_size,
                                                alpha_mul=alpha_mul,
                                                lr=lr,
                                                poe=poe,
                                                verbose=verbose
                                                )

                adata, adata_normed = visium_args.get_adata()
                inference_outputs, generative_outputs,adata_ = sf_model.model_eval(model,
                                                                            adata,
                                                                            visium_args,
                                                                            poe=poe,
                                                                            device=device)

                def cell2proportion(adata):
                    adata_plot=sc.AnnData(adata.X)
                    adata_plot.obs=utils.extract_feature(adata, 'qc_m').obs.copy()
                    adata_plot.var=adata.var.copy()
                    adata_plot.obsm=adata.obsm.copy()
                    adata_plot.obsp=adata.obsp.copy()
                    adata_plot.uns=adata.uns.copy()
                    return adata_plot
                adata_plot=cell2proportion(adata_)
                self.adata_cell2location=adata_plot
                self.adata_sp=adata_normed
                self.starfysh_model=model 
                print(f"{Colors.GREEN}[OK] starfysh is done{Colors.ENDC}")
                print(f"The starfysh result is saved in self.adata_cell2location")
                print(f"The starfysh model is saved in self.starfysh_model")
            

        elif method == 'RCTD':
            self.method = 'RCTD'
            try:
                from rctd import RCTD, Reference, RCTDConfig, run_rctd
            except ImportError:
                raise ImportError(
                    "rctd-py is not installed. Install it with: pip install rctd-py "
                    "(see https://github.com/p-gueguen/rctd-py for details)."
                )

            # ── 1. RCTD needs RAW counts on .X for both adata_sc and adata_sp.
            # The omicverse preprocessing pipeline writes raw counts into
            # `layers['counts']` and replaces .X with normalized values, so we
            # build temporary count-matrix views without mutating the user's
            # AnnData. If `layers['counts']` is missing we fall back to .X
            # (with a sanity check) — that handles the case where the user
            # passes raw-counts AnnData directly without going through
            # `preprocess_sc` / `preprocess_sp`.
            def _counts_view(adata, label, requested_layer):
                requested_layer = _validate_count_layer(
                    adata,
                    requested_layer,
                    label,
                    method='RCTD',
                )
                counts = adata.layers[requested_layer] if requested_layer is not None else adata.X
                view = adata.copy()
                view.X = counts
                return view

            adata_sc_counts = _counts_view(
                self.adata_sc,
                'reference',
                counts_layer_sc,
            )
            adata_sp_counts = _counts_view(
                self.adata_sp,
                'spatial',
                counts_layer_sp,
            )

            # ── 2. Set defaults; let user override via rctd_kwargs.
            rctd_defaults = {
                'mode': 'full',           # 'full' | 'doublet' | 'multi'
                # Reference kwargs
                'cell_min': 25,
                'n_max_cells': 10000,
                'min_UMI': 100,
                # Run kwargs
                'batch_size': 'auto',
                'sigma_override': None,
                # RCTDConfig overrides — passed straight to RCTDConfig(**...)
                'config': {},
            }
            user_kwargs = dict(rctd_kwargs) if rctd_kwargs else {}
            cfg_overrides = user_kwargs.pop('config', None) or {}
            rctd_defaults['config'] = {**rctd_defaults['config'], **cfg_overrides}
            rctd_defaults.update(user_kwargs)

            mode = rctd_defaults['mode']
            print(f"{Colors.CYAN}Running RCTD (mode={mode!r}) — this may take a while on full Visium slides.{Colors.ENDC}")

            ref = Reference(
                adata_sc_counts,
                cell_type_col=celltype_key_sc,
                cell_min=rctd_defaults['cell_min'],
                n_max_cells=rctd_defaults['n_max_cells'],
                min_UMI=rctd_defaults['min_UMI'],
            )

            cfg = RCTDConfig(**rctd_defaults['config']) if rctd_defaults['config'] else RCTDConfig()

            result = run_rctd(
                adata_sp_counts,
                ref,
                mode=mode,
                config=cfg,
                batch_size=rctd_defaults['batch_size'],
                sigma_override=rctd_defaults['sigma_override'],
            )
            self.rctd_result = result

            # ── 3. Convert RCTD result → omicverse `adata_cell2location` schema:
            # an AnnData of shape (kept_pixels × cell_types) holding row-normalised
            # proportions. Only pixels that pass RCTD's UMI filter are kept; the
            # boolean `pixel_mask` aligns indices back to the original adata_sp.
            pixel_mask = np.asarray(result.pixel_mask)
            kept_idx = np.where(pixel_mask)[0]
            cell_type_names = list(result.cell_type_names)

            if mode in ('full', 'multi'):
                weights = np.asarray(result.weights, dtype=np.float32)
            elif mode == 'doublet':
                # `weights_doublet` is (n_pixels, 2) — only two cell types per
                # pixel are non-zero. Materialise into the full (n_pixels, n_types)
                # matrix by scattering using `first_type` / `second_type` indices.
                weights_doublet = np.asarray(result.weights_doublet, dtype=np.float32)
                first_type = np.asarray(result.first_type, dtype=np.int64)
                second_type = np.asarray(result.second_type, dtype=np.int64)
                n_kept = len(first_type)
                weights = np.zeros((n_kept, len(cell_type_names)), dtype=np.float32)
                rows = np.arange(n_kept)
                weights[rows, first_type] += weights_doublet[:, 0]
                weights[rows, second_type] += weights_doublet[:, 1]
            else:
                raise ValueError(f"Unknown RCTD mode {mode!r}; expected 'full' / 'doublet' / 'multi'.")

            # Row-normalise to proportions (RCTD weights are unnormalised abundances)
            row_sum = weights.sum(axis=1, keepdims=True)
            proportions = np.divide(
                weights, row_sum, out=np.zeros_like(weights), where=row_sum > 0
            )

            adata_cell2location = sc.AnnData(proportions)
            adata_cell2location.var_names = cell_type_names
            adata_cell2location.var.index = pd.Index(cell_type_names)

            # Slice the spatial-object metadata to the kept pixels
            adata_cell2location.obs_names = self.adata_sp.obs_names[kept_idx]
            adata_cell2location.obs = self.adata_sp.obs.iloc[kept_idx].copy()
            for k, v in self.adata_sp.obsm.items():
                adata_cell2location.obsm[k] = v[kept_idx]
            adata_cell2location.uns = self.adata_sp.uns.copy()

            # Mirror omicverse's existing convention: also stash the proportions
            # in adata_sp.obsm so existing plotting helpers (e.g. matplotlib /
            # ov.pl.spatial(color=cell_type)) can consume them as a column source.
            prop_df = pd.DataFrame(
                proportions, index=adata_cell2location.obs_names, columns=cell_type_names
            )
            self.adata_sp.obsm['rctd_proportions'] = prop_df.reindex(self.adata_sp.obs_names).fillna(0.0)
            self.adata_cell2location = adata_cell2location

            print(f"{Colors.GREEN}[OK] RCTD deconvolution is done{Colors.ENDC}")
            print(f"The deconvolution result is saved in self.adata_cell2location")
            print(f"Cell type proportions are also stored in self.adata_sp.obsm['rctd_proportions']")
            print(f"The full RCTD result object is saved in self.rctd_result")

        else:  # guarded above
            raise RuntimeError(f"Unhandled deconvolution method {method!r}.")

    def impute(self,method='Tangram'):
        """
        Generate spot-level imputation outputs from a fitted spatial deconvolution model.

        Parameters
        ----------
        method:{'Tangram', 'cell2location'}
            Imputation backend. ``'Tangram'`` runs the loaded Tangram model and stores
            the imputed AnnData in ``self.adata_impute``. ``'cell2location'`` computes
            expected cell-type-specific expression and writes each cell type into
            ``self.adata_sp.layers``.

        Returns
        -------
        None
            Results are stored in object attributes in-place.

        Examples
        --------
        >>> deconv.impute(method='Tangram')
        >>> deconv.impute(method='cell2location')
        """
        normalized_method = str(method).lower()
        if normalized_method == 'tangram':
            self.method='Tangram'
            from ._tangram import Tangram
            self.adata_impute=self.tangram_model.impute()
            print(f"{Colors.GREEN}[OK] Tangram impute is done{Colors.ENDC}")
            print(f"The impute result is saved in self.adata_impute")
        elif normalized_method == 'cell2location':
            self.method='cell2location'
            from ..external.space.cell2location.models import Cell2location
            expected_dict = self.mod_sp.module.model.compute_expected_per_cell_type(
                self.mod_sp.samples["post_sample_q05"], self.mod_sp.adata_manager
            )
            # Add to anndata layers
            for i, n in enumerate(self.mod_sp.factor_names_):
                self.adata_sp.layers[n] = expected_dict["mu"][i]
            print(f"{Colors.GREEN}[OK] cell2location impute is done{Colors.ENDC}")
            print(f"Compare with the tangram impute result, cell2location's impute stores in self.adata_sp.layers")
        else:
            raise ValueError("method must be 'Tangram' or 'cell2location'.")
    
    def load_cell2location_model(self,mod_sp_path):
        """
        Load a previously trained cell2location spatial model.

        Parameters
        ----------
        mod_sp_path:str
            Path to a serialized model object produced by ``ov.utils.save``.

        Returns
        -------
        None
            Stores loaded model in ``self.mod_sp``.

        Examples
        --------
        >>> decov.load_cell2location_model('results/mod_sp.pkl')
        """
        self.method='cell2location'
        from ..utils import load
        #self.mod_sc=load(mod_sc_path)
        self.mod_sp=load(mod_sp_path)
        print(f"{Colors.GREEN}[OK] cell2location model is loaded{Colors.ENDC}")
        print(f"The cell2location model is saved in self.mod_sc and self.mod_sp")

    def cell2location_inference(self,sample_kwargs=None):
        """
        Export cell2location posterior and derive normalized cell-type proportions.

        Parameters
        ----------
        sample_kwargs:dict or None
            Parameters passed to ``self.mod_sp.export_posterior``.
            Defaults to ``{'num_samples': 1000, 'batch_size': 2500}``.

        Returns
        -------
        None
            Writes posterior outputs to ``self.adata_sp`` and stores normalized proportion
            AnnData in ``self.adata_cell2location``.

        Examples
        --------
        >>> decov.cell2location_inference()
        >>> decov.cell2location_inference(sample_kwargs={'num_samples': 500, 'batch_size': 2048})
        """
        if sample_kwargs is None:
            sample_kwargs={"num_samples": 1000, "batch_size": 2500}
        
        self.adata_sp = self.mod_sp.export_posterior(
            self.adata_sp,
            sample_kwargs=sample_kwargs,
        )
        # 3) 绝对丰度 → 比例
        abund = self.adata_sp.obsm['q05_cell_abundance_w_sf']
        self.adata_sp.obsm['prop_celltypes'] = abund.div(abund.sum(axis=1).clip(lower=1e-9), axis=0)

        adata_cell2location=sc.AnnData(self.adata_sp.obsm['prop_celltypes'])
        adata_cell2location.var.index=self.adata_sp.uns["mod"]["factor_names"]
        adata_cell2location.var_names=self.adata_sp.uns["mod"]["factor_names"]
        
        adata_cell2location.obsm=self.adata_sp.obsm.copy()
        adata_cell2location.obs=self.adata_sp.obs.copy()
        adata_cell2location.obsp=self.adata_sp.obsp.copy()
        adata_cell2location.uns=self.adata_sp.uns.copy()
        self.adata_cell2location=adata_cell2location

        print(f"{Colors.GREEN}[OK] cell2location is done{Colors.ENDC}")
        print(f"The cell2location result is saved in self.adata_cell2location")

    def load_tangram_model(self,model_path):
        """
        Load a pre-trained Tangram model for downstream projection/inference.

        Parameters
        ----------
        model_path:str
            Path to a serialized Tangram model object.

        Returns
        -------
        None
            Stores loaded model in ``self.tangram_model``.

        Examples
        --------
        >>> decov.load_tangram_model('results/tangram_model.pkl')
        """
        self.method='Tangram'
        #from ._tangram import Tangram
        from ..utils import load
        
        self.tangram_model=load(model_path)
        print(f"{Colors.GREEN}[OK] Tangram model is loaded{Colors.ENDC}")
        print(f"The Tangram model is saved in self.tangram")

    def tangram_inference(self,sample_kwargs=None):
        """
        Infer spot-level cell-type proportions using a loaded Tangram model.

        Parameters
        ----------
        sample_kwargs:dict or None
            Reserved argument for API consistency. Current implementation does not use it.

        Returns
        -------
        None
            Stores inferred proportions in ``self.adata_cell2location``.

        Examples
        --------
        >>> decov.tangram_inference()
        """
        self.adata_cell2location=self.tangram_model.cell2location()
        print(f"{Colors.GREEN}[OK] Tangram is done{Colors.ENDC}")
        print(f"The Tangram result is saved in self.adata_cell2location")


@register_function(
    aliases=['计算细胞类型签名', 'calculate_gene_signature', 'celltype_marker_signature', '构建细胞类型特征基因'],
    category="space",
    description="Generate per-cell-type marker gene signatures from scRNA-seq references for spatial deconvolution methods such as Starfysh.",
    prerequisites={'optional_functions': ['single.get_celltype_marker']},
    requires={'obs': ['clustertype labels'], 'var': ['gene symbols']},
    produces={},
    auto_fix='none',
    examples=[
        "gene_sig = ov.space.calculate_gene_signature(adata_sc, clustertype='celltype', topgenenumber=50)",
        "decov.deconvolution(method='starfysh', gene_sig=gene_sig, celltype_key_sc='celltype')",
    ],
    related=['space.Deconvolution', 'single.get_celltype_marker']
)
def calculate_gene_signature(
    adata_sc,clustertype,
    rank=True,
    key='rank_genes_groups',
    foldchange=2,
    topgenenumber=20 
):
    """
    Build a marker-gene signature table for each cell type in a reference scRNA-seq dataset.

    Parameters
    ----------
    adata_sc:AnnData
        Single-cell reference AnnData.
    clustertype:str
        Cell-type label key in ``adata_sc.obs``.
    rank:bool
        Whether to use ranked markers from differential expression analysis.
    key:str
        ``adata.uns`` key for ranked genes (used when ``rank=True``).
    foldchange:float
        Fold-change threshold for marker selection.
    topgenenumber:int
        Number of top marker genes retained per cell type.

    Returns
    -------
    pandas.DataFrame
        Gene-signature matrix where each column corresponds to one cell type and each
        cell stores a marker gene (padded with NA when needed).

    Examples
    --------
    >>> gene_sig = ov.space.calculate_gene_signature(adata_sc, clustertype='celltype', topgenenumber=50)
    """
    # 1) 用“已注释细胞类型”作为分组
    from ..single import get_celltype_marker
    all_markers = get_celltype_marker(
        adata_sc,
        clustertype=clustertype,
        rank=rank,
        key=key,
        foldchange=foldchange,
        topgenenumber=topgenenumber
    )
    # 2) dict -> 表格（列=celltype，行=genes）
    max_len = max(len(v) for v in all_markers.values())
    gene_sig = pd.DataFrame({
        k: list(v) + [pd.NA] * (max_len - len(v))
        for k, v in all_markers.items()
    })

    return gene_sig

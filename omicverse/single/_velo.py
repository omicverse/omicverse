from .._settings import Colors, EMOJI
from .._monitor import monitor
from .._registry import register_function
from .._optional import build_optional_dependency_error

@register_function(
    aliases=['RNA velocity分析器', 'Velo', 'RNA velocity pipeline', 'dynamo scvelo regvelo wrapper', '细胞状态转变速度分析'],
    category="single",
    description="Unified RNA velocity workflow supporting dynamo, scvelo, latentvelo, graphvelo and RegVelo to infer transcriptional state transitions and directional trajectories.",
    prerequisites={'optional_functions': ['pp.preprocess', 'pp.neighbors']},
    requires={'layers': ['spliced/unspliced or counts'], 'obsm': ['X_umap (recommended)']},
    produces={'layers': ['velocity-related layers'], 'obsm': ['velocity embeddings'], 'uns': ['velocity graphs']},
    auto_fix='escalate',
    examples=['velo_obj = ov.single.Velo(adata)', 'velo_obj.cal_velocity(method="dynamo")', 'velo_obj.cal_velocity(method="regvelo", prior_grn=adata.uns["skeleton"])'],
    related=['single.Velo', 'pl.add_streamplot', 'utils.cal_paga']
)
class Velo:
    """
    RNA velocity analysis wrapper for directional cell-state transition inference.
    
    Parameters
    ----------
    adata : AnnData
        AnnData containing spliced/unspliced layers (or backend-compatible count layers)
        and low-dimensional embeddings.
    
    Returns
    -------
    None
        Initializes velocity workflow state.
    
    Examples
    --------
    >>> velo_obj = ov.single.Velo(adata)
    """

    def __init__(self, adata):
        self.adata = adata
        print(f"{Colors.WARNING}In Velo module, you should keep all genes' expression not normalized.{Colors.ENDC}")

    def run(self):
        """
        Print a quick diagnostic summary for velocity input data.

        Returns
        -------
        None
            Prints basic matrix size and expression statistics.
        """
        print(f"{Colors.HEADER}{Colors.BOLD}{EMOJI['start']} Vela Analysis Initialization:{Colors.ENDC}")
        print(f"   {Colors.CYAN}Input data shape: {Colors.BOLD}{self.adata.shape[0]} cells × {self.adata.shape[1]} genes{Colors.ENDC}")
        print(f"   {Colors.BLUE}Total UMI counts: {Colors.BOLD}{self.adata.X.sum():,.0f}{Colors.ENDC}")
        print(f"   {Colors.BLUE}Mean genes per cell: {Colors.BOLD}{self.adata.X.mean():,.1f}{Colors.ENDC}")
        print(f"   {Colors.GREEN}Vela Analysis Completed:{Colors.ENDC}")

    def filter_genes(self,min_shared_counts=20,**kwargs):
        """
        Filter genes for velocity modeling using scVelo shared-count criteria.

        Parameters
        ----------
        min_shared_counts : int
            Minimum shared spliced/unspliced counts required to keep a gene.
        **kwargs
            Additional keyword arguments forwarded to ``scvelo.pp.filter_genes``.

        Returns
        -------
        None
            Updates ``self.adata`` in-place.

        Examples
        --------
        >>> velo.filter_genes(min_shared_counts=20)
        """
        from scvelo.pp import filter_genes
        filter_genes(self.adata,min_shared_counts=min_shared_counts,**kwargs)

    def preprocess(self, recipe='monocle',
                   n_neighbors=30,
                   n_pcs=30,
                   **kwargs):
        """
        Preprocess expression data before velocity estimation.

        Parameters
        ----------
        recipe : str
            Dynamo preprocessing recipe (for example ``'monocle'``).
        n_neighbors : int
            Number of neighbors for the graph built after preprocessing.
        n_pcs : int
            Number of principal components for neighbor graph construction.
        **kwargs
            Additional keyword arguments passed to ``dynamo.pp.Preprocessor``.

        Returns
        -------
        None
            Writes preprocessing outputs to ``self.adata``.

        Examples
        --------
        >>> velo.preprocess(recipe='monocle', n_neighbors=30, n_pcs=30)
        """
        import dynamo as dyn 
        preprocessor = dyn.pp.Preprocessor(cell_cycle_score_enable=True,**kwargs)
        preprocessor.preprocess_adata(self.adata, recipe=recipe)
        from ..pp import neighbors
        neighbors(self.adata,n_neighbors=n_neighbors,n_pcs=n_pcs,use_rep='X_pca')

    def moments(self,backend='dynamo',n_pcs=30,n_neighbors=30,**kwargs):
        """
        Compute neighborhood moments required by RNA velocity models.

        Parameters
        ----------
        backend : {'dynamo', 'scvelo'}
            Backend used to compute moments.
        n_pcs : int
            Number of principal components.
        n_neighbors : int
            Number of neighbors used in moment estimation.
        **kwargs
            Additional backend-specific options.

        Returns
        -------
        None
            Stores moments in ``adata.layers``.

        Examples
        --------
        >>> velo.moments(backend='dynamo')
        """
        if backend == 'dynamo':
            import dynamo as dyn 
            dyn.tl.moments(self.adata,n_pca_components=n_pcs,n_neighbors=n_neighbors,**kwargs)
            self.adata.layers['Ms'] = self.adata.layers['M_s']
            self.adata.layers['Mu'] = self.adata.layers['M_u']
        elif backend == 'scvelo':
            import scvelo as scv 
            scv.pp.moments(self.adata, n_pcs=n_pcs, n_neighbors=n_neighbors,**kwargs)
        else:
            raise ValueError(f"Backend {backend} not supported")

    def prepare_regvelo(
        self,
        prior_grn,
        regulators=None,
        tf_key='is_tf',
        n_neighbors=30,
        n_pcs=50,
        moment_backend='scvelo',
        prior_orientation='regulator_by_target',
        use_ov_neighbors=True,
        preprocess_kwargs=None,
        set_prior_kwargs=None,
        neighbors_kwargs=None,
        moments_kwargs=None,
    ):
        """
        Prepare AnnData for ``cal_velocity(method='regvelo')``.

        This method wraps the common upstream RegVelo preparation pattern in an
        OmicVerse-facing step: build a neighbor graph, compute ``Ms``/``Mu``
        moment layers, run RegVelo gene preprocessing, align the prior GRN to
        the retained genes, and store it in ``adata.uns['skeleton']``.

        Parameters
        ----------
        prior_grn : pandas.DataFrame or array-like
            Prior GRN used by RegVelo. The RegVelo tutorial dataset returns a
            regulator-by-target matrix, so ``prior_orientation`` defaults to
            ``'regulator_by_target'`` and the matrix is transposed before
            calling ``regvelo.pp.set_prior_grn``. Use
            ``'target_by_regulator'`` when rows are already target genes and
            columns are regulators. DataFrame edge lists with TF/source/regulator
            and target columns are also accepted and are interpreted from their
            explicit column names.
        regulators : sequence of str or None
            Regulator names. If ``None`` and ``adata.var[tf_key]`` exists, TFs
            are inferred before preprocessing and intersected with retained
            genes.
        tf_key : str
            Column in ``adata.var`` used to store TF/regulator flags.
        n_neighbors, n_pcs : int
            Neighbor/moment parameters.
        moment_backend : {'scvelo', 'dynamo'}
            Backend used by :meth:`moments`.
        prior_orientation : {'regulator_by_target', 'target_by_regulator'}
            Orientation of ``prior_grn`` before passing to RegVelo.
        use_ov_neighbors : bool
            Whether to run ``ov.pp.neighbors`` before computing moments.
        preprocess_kwargs, set_prior_kwargs, neighbors_kwargs, moments_kwargs
            Additional options for RegVelo preprocessing, RegVelo prior-GRN
            alignment, OmicVerse neighbors, and moment computation.

        Returns
        -------
        tuple
            ``(prior_grn, regulators)`` aligned to the prepared AnnData.
        """
        _, _, rgv = self._import_regvelo()

        if regulators is None and tf_key in self.adata.var:
            regulators = self.adata.var_names[self.adata.var[tf_key].astype(bool)].tolist()
        elif regulators is not None:
            regulators = list(regulators)

        if use_ov_neighbors:
            from ..pp import neighbors as ov_neighbors
            neighbor_params = {'n_neighbors': n_neighbors, 'n_pcs': n_pcs}
            if neighbors_kwargs:
                neighbor_params.update(neighbors_kwargs)
            ov_neighbors(self.adata, **neighbor_params)

        moment_params = {'n_neighbors': n_neighbors, 'n_pcs': n_pcs}
        if moments_kwargs:
            moment_params.update(moments_kwargs)
        self.moments(backend=moment_backend, **moment_params)

        preprocess_params = {}
        if preprocess_kwargs:
            preprocess_params.update(preprocess_kwargs)
        prepared = rgv.pp.preprocess_data(self.adata, **preprocess_params)
        if prepared is not None:
            self.adata = prepared

        prior_for_regvelo = self._orient_prior_for_regvelo(prior_grn, prior_orientation)
        set_prior_params = {}
        if set_prior_kwargs:
            set_prior_params.update(set_prior_kwargs)
        prepared = rgv.pp.set_prior_grn(self.adata, prior_for_regvelo, **set_prior_params)
        if prepared is not None:
            self.adata = prepared

        if regulators is None and tf_key in self.adata.var:
            regulators = self.adata.var_names[self.adata.var[tf_key].astype(bool)].tolist()
        elif regulators is not None:
            regulators = sorted(set(regulators).intersection(self.adata.var_names))
            self.adata.var[tf_key] = self.adata.var_names.isin(regulators)

        self.adata.uns['regvelo_prepare'] = {
            'n_neighbors': n_neighbors,
            'n_pcs': n_pcs,
            'moment_backend': moment_backend,
            'prior_orientation': prior_orientation,
            'tf_key': tf_key,
            'n_regulators': None if regulators is None else len(regulators),
        }
        self.adata.uns['regvelo_regulators'] = regulators
        return self.adata.uns['skeleton'], regulators
    
    def dynamics(self,backend='dynamo',**kwargs):
        """
        Fit transcriptional dynamics parameters for velocity inference.

        Parameters
        ----------
        backend : {'dynamo', 'scvelo'}
            Backend used to estimate kinetics.
        **kwargs
            Additional arguments passed to backend fitting functions.

        Returns
        -------
        None
            Stores fitted parameters in ``self.adata``.

        Examples
        --------
        >>> velo.dynamics(backend='scvelo')
        """
        if backend == 'dynamo':
            import dynamo as dyn 
            dyn.tl.dynamics(self.adata,**kwargs)
        elif backend == 'scvelo':
            import scvelo as scv 
            scv.tl.recover_dynamics(self.adata,**kwargs)
        else:
            raise ValueError(f"Backend {backend} not supported")
    
    
    def cal_velocity(
        self,
        method='dynamo',
        batch_key=None,
        celltype_key=None,
        velocity_key='velocity_S',
        n_jobs=1,
        n_top_genes=2000,
        param_name_key='tmp/latentvelo_params',
        latentvelo_VAE_kwargs={},
        prior_grn=None,
        regulators=None,
        spliced_layer='Ms',
        unspliced_layer='Mu',
        n_samples=30,
        batch_size=None,
        model_load_path=None,
        model_save_path=None,
        model_overwrite=False,
        reuse_regvelo_output=False,
        regvelo_kwargs=None,
        train_kwargs=None,
        compute_velocity_graph=False,
        compute_velocity_embedding=False,
        basis='umap',
        graph_kwargs=None,
        embedding_kwargs=None,
        **kwargs
    ):
        """
        Estimate RNA velocity vectors and write them into AnnData.

        Parameters
        ----------
        method : {'dynamo', 'scvelo', 'latentvelo', 'graphvelo', 'regvelo'}
            Velocity estimation strategy.
        batch_key : str or None
            Batch key used by latentvelo.
        celltype_key : str or None
            Cell-type key used by latentvelo.
        velocity_key : str
            Output velocity layer key.
        n_jobs : int
            Number of jobs used by graphvelo.
        n_top_genes : int
            Number of top genes used by latentvelo.
        param_name_key : str
            Directory/key to store latentvelo parameters.
        latentvelo_VAE_kwargs : dict
            Extra arguments for latentvelo VAE construction.
        prior_grn : pandas.DataFrame, array-like, torch.Tensor or None
            Prior GRN used by RegVelo. If ``None``, ``adata.uns['skeleton']``
            is used when available. DataFrames can be square adjacency
            matrices or edge lists with TF/source/regulator and target columns.
        regulators : list of str or None
            Regulator names used by RegVelo. If ``None`` and
            ``adata.var['is_tf']`` exists, TF genes are inferred from it.
        spliced_layer, unspliced_layer : str
            Layers passed to ``REGVELOVI.setup_anndata`` for RegVelo.
        n_samples : int
            Posterior samples passed to ``regvelo.tl.set_output``.
        batch_size : int or None
            Batch size passed to ``regvelo.tl.set_output``. Defaults to
            ``adata.n_obs`` for RegVelo.
        model_load_path : str or None
            Optional directory of an existing RegVelo model. When provided,
            OmicVerse loads the model and skips training.
        model_save_path : str or None
            Optional directory where the trained RegVelo model is saved.
        model_overwrite : bool
            Whether to overwrite ``model_save_path`` if it already exists.
        reuse_regvelo_output : bool
            Whether to reuse existing ``adata.layers[velocity_key]`` or
            ``adata.layers['velocity']`` and skip the expensive RegVelo output
            export step. This is useful when loading a model for an AnnData
            object that already has RegVelo velocities.
        regvelo_kwargs, train_kwargs : dict or None
            Extra arguments passed to ``REGVELOVI`` and ``REGVELOVI.train``.
        compute_velocity_graph : bool
            Whether to run ``scvelo.tl.velocity_graph`` after RegVelo.
        compute_velocity_embedding : bool
            Whether to project RegVelo velocities to ``basis`` after building
            the velocity graph.
        basis : str
            Embedding basis used for optional RegVelo graph/embedding output.
        graph_kwargs, embedding_kwargs : dict or None
            Extra arguments forwarded to ``velocity_graph`` and
            ``velocity_embedding`` when the optional RegVelo downstream steps
            are enabled.
        **kwargs
            Additional backend-specific options.

        Returns
        -------
        None
            Writes velocity outputs to ``self.adata.layers``/``.obsm``/``.var``.

        Examples
        --------
        >>> velo.cal_velocity(method='dynamo')
        >>> velo.cal_velocity(method='graphvelo', n_jobs=4)
        >>> velo.cal_velocity(method='regvelo', prior_grn=adata.uns['skeleton'])
        """
        
        if method == 'dynamo':
            import dynamo as dyn 
            dyn.tl.cell_velocities(self.adata,**kwargs)
            self.adata.var[f'{velocity_key}_genes']=self.adata.var['use_for_transition']

        elif method == 'scvelo':
            import scvelo as scv 
            scv.tl.velocity(self.adata,**kwargs)
            self.adata.layers[velocity_key] = self.adata.layers['velocity']
            self.adata.var[f'{velocity_key}_genes']=self.adata.var['velocity_genes']

        elif method == 'latentvelo':
            self._latentvelo_cal(
                velocity_key=velocity_key,
                celltype_key=celltype_key,
                batch_key=batch_key,
                latentvelo_VAE_kwargs=latentvelo_VAE_kwargs,
                param_name_key=param_name_key,
                **kwargs
            )
        elif method == 'graphvelo':
            dynamo_flag = False
            try:
                import dynamo as dyn 
                dyn.tl.neighbors(self.adata)
                dyn.tl.cell_velocities(self.adata,**kwargs)
                #self.adata.var[f'{velocity_key}_genes']=self.adata.var['use_for_transition']
                self._graphvelo_cal(backend='dynamo',xkey='Ms',vkey='velocity_S',n_jobs=n_jobs,**kwargs)
                dynamo_flag = True
            except:
                print(f"{Colors.WARNING}dynamo run failed.{Colors.ENDC}")
            if dynamo_flag==False:
                try:
                    import scvelo as scv
                    scv.tl.velocity(self.adata,**kwargs)
                    #self.adata.layers[velocity_key] = self.adata.layers['velocity']
                    #self.adata.var[f'{velocity_key}_genes']=self.adata.var['velocity_genes']

                    self._graphvelo_cal(backend='scvelo',xkey='Ms',vkey='velocity',
                    n_jobs=n_jobs,**kwargs)
                except:
                    print(f"{Colors.WARNING}scvelo run failed.{Colors.ENDC}")
                    raise ValueError("scvelo also run failed.")
            
        elif method == 'regvelo':
            self._regvelo_cal(
                velocity_key=velocity_key,
                prior_grn=prior_grn,
                regulators=regulators,
                spliced_layer=spliced_layer,
                unspliced_layer=unspliced_layer,
                n_samples=n_samples,
                batch_size=batch_size,
                model_load_path=model_load_path,
                model_save_path=model_save_path,
                model_overwrite=model_overwrite,
                reuse_regvelo_output=reuse_regvelo_output,
                regvelo_kwargs=regvelo_kwargs,
                train_kwargs=train_kwargs,
                compute_velocity_graph=compute_velocity_graph,
                compute_velocity_embedding=compute_velocity_embedding,
                basis=basis,
                graph_kwargs=graph_kwargs,
                embedding_kwargs=embedding_kwargs,
                **kwargs
            )
            


        else:
            raise ValueError(f"Method {method} not supported")
        return self.adata

    def graphvelo(
        self,xkey='Ms',vkey='velocity_S',
        n_jobs=1,
        basis_keys=['X_umap','X_pca'],
        gene_subset=None,
        **kwargs
    ):
        """
        Refine velocity vectors with GraphVelo and project to selected embeddings.

        Parameters
        ----------
        xkey : str
            Layer key containing expression moments used as model input.
        vkey : str
            Layer key containing initial velocity vectors.
        n_jobs : int
            Number of CPU jobs for GraphVelo training.
        basis_keys : list of str
            Embedding keys in ``adata.obsm`` to project refined velocity onto.
        gene_subset : list of str or None
            Gene subset used by GraphVelo. If ``None``, all genes are used.
            Unknown or empty subsets raise before model training.
        **kwargs
            Additional arguments passed to ``GraphVelo``.

        Returns
        -------
        None
            Stores refined velocity in ``adata.layers['velocity_gv']`` and projected vectors in ``adata.obsm``.

        Examples
        --------
        >>> velo.graphvelo(xkey='Ms', vkey='velocity_S', basis_keys=['X_umap'])
        """
        if gene_subset is None:
            selected_genes = self.adata.var_names.tolist()
        else:
            if isinstance(gene_subset, str):
                gene_subset = [gene_subset]
            selected_genes = list(dict.fromkeys(gene_subset))
            if not selected_genes:
                raise ValueError("`gene_subset` must contain at least one gene.")
            missing_genes = [
                gene for gene in selected_genes if gene not in self.adata.var_names
            ]
            if missing_genes:
                raise KeyError(
                    "Some genes in `gene_subset` are not present in `adata.var_names`: "
                    f"{missing_genes}."
                )
        from ..external.graphvelo.graph_velocity import GraphVelo
        from ..external.graphvelo.utils import adj_to_knn
        indices, _ = adj_to_knn(self.adata.obsp['connectivities'])
        self.adata.uns['neighbors']['indices'] = indices
        gv=GraphVelo(
            self.adata,
            xkey=xkey,
            vkey=vkey,
            gene_subset=selected_genes,
            **kwargs,
        )
        gv.train(n_jobs=n_jobs)
        self.adata.layers['velocity_gv'] = gv.project_velocity(self.adata.layers[xkey])

        self.adata.var['velocity_gv_genes'] = self.adata.var_names.isin(selected_genes)
        if issparse(self.adata.layers['velocity_gv']):
            self.adata.layers['velocity_gv'] = self.adata.layers['velocity_gv'].toarray()
        for basis_key in basis_keys:
            self.adata.obsm[f'gv_{basis_key}'] = gv.project_velocity(self.adata.obsm[basis_key])



    def velocity_graph(self,basis='umap',vkey='velocity_S',**kwargs):
        """
        Build a velocity transition graph from precomputed velocity vectors.

        Parameters
        ----------
        basis : str
            Embedding basis used by scVelo graph construction.
        vkey : str
            Velocity layer key.
        **kwargs
            Additional arguments forwarded to ``scvelo.tl.velocity_graph``.

        Returns
        -------
        None
            Writes velocity graph to ``adata.uns``/``adata.obsp``.

        Examples
        --------
        >>> velo.velocity_graph(basis='umap', vkey='velocity_S')
        """
        import scvelo as scv
        scv.tl.velocity_graph(self.adata, vkey=vkey, **kwargs)
    
    def velocity_embedding(self,basis='umap',vkey='velocity_S',**kwargs):   
        """
        Project velocity vectors onto a low-dimensional embedding.

        Parameters
        ----------
        basis : str
            Embedding basis name (for example ``'umap'``).
        vkey : str
            Velocity layer key.
        **kwargs
            Additional arguments forwarded to ``scvelo.tl.velocity_embedding``.

        Returns
        -------
        None
            Writes projected vectors to ``adata.obsm``.

        Examples
        --------
        >>> velo.velocity_embedding(basis='umap', vkey='velocity_S')
        """
        import scvelo as scv
        scv.tl.velocity_embedding(self.adata, basis=basis, vkey=vkey, **kwargs)
        #return self.adata

    def velocity_streamplot(
        self,
        basis='umap',
        velocity_key='velocity_S',
        color=None,
        ax=None,
        show=False,
        size=100,
        alpha=0.3,
        embedding_kwargs=None,
        stream_kwargs=None,
        title=None,
    ):
        """
        Plot cells and velocity streamlines with OmicVerse plotting helpers.

        Parameters
        ----------
        basis : str
            Embedding basis, either ``'umap'`` or an AnnData key such as
            ``'X_umap'``.
        velocity_key : str
            Projected velocity key in ``adata.obsm``. For a layer
            ``'velo_regvelo'`` projected on UMAP, this is usually
            ``'velo_regvelo_umap'``.
        color : str or None
            Observation/feature key used to color cells. If ``None``, a
            sensible observation key such as ``cell_type`` or ``clusters`` is
            selected when available.
        ax : matplotlib Axes or None
            Axis to draw on.
        show : bool
            Whether to show the figure immediately.
        size, alpha : float
            Cell point size and transparency.
        embedding_kwargs, stream_kwargs : dict or None
            Extra keyword arguments forwarded to ``ov.pl.embedding`` and
            ``ov.pl.add_streamplot``.
        title : str or None
            Optional plot title.

        Returns
        -------
        matplotlib Axes
            Axis containing the plot.
        """
        from .. import pl as _pl
        from .. import plt as _plt

        basis_key = basis if str(basis).startswith('X_') else f'X_{basis}'
        if basis_key not in self.adata.obsm:
            raise KeyError(f"Could not find embedding `{basis_key}` in adata.obsm")
        if velocity_key not in self.adata.obsm:
            raise KeyError(f"Could not find projected velocity `{velocity_key}` in adata.obsm")

        if color is None:
            color = self._default_velocity_color_key()

        if ax is None:
            _, ax = _plt.subplots(figsize=(4, 4))

        embedding_params = {'show': False, 'size': size, 'alpha': alpha}
        if embedding_kwargs:
            embedding_params.update(embedding_kwargs)
        _pl.embedding(
            self.adata,
            basis=basis_key,
            color=color,
            ax=ax,
            **embedding_params,
        )

        stream_params = {}
        if stream_kwargs:
            stream_params.update(stream_kwargs)
        _pl.add_streamplot(
            self.adata,
            basis=basis_key,
            velocity_key=velocity_key,
            ax=ax,
            **stream_params,
        )

        if title is not None:
            ax.set_title(title)
        if show:
            _plt.show()
        return ax

    def cellrank_fate(
        self,
        velocity_key='velocity_S',
        xkey='Ms',
        cluster_key=None,
        terminal_states=None,
        n_states=8,
        n_cells=30,
        connectivity_weight=0.2,
        compute_fate_probabilities=False,
        fate_kwargs=None,
        clean=False,
        plot=False,
        basis='umap',
        **kwargs,
    ):
        """
        Run a CellRank fate-analysis step from OmicVerse velocity output.

        This mirrors the RegVelo reproducibility workflow, where the RegVelo
        velocity field is converted into a CellRank velocity kernel and
        optionally mixed with a connectivity kernel.

        Parameters
        ----------
        velocity_key : str
            Velocity layer used to construct the CellRank velocity kernel.
        xkey : str
            Expression or moment layer used by the velocity kernel.
        cluster_key : str or None
            Observation column used to constrain macrostate discovery.
        terminal_states : str, sequence of str or None
            Terminal states to set after macrostate discovery. Missing states
            are skipped with a warning.
        n_states : int
            Number of GPCCA macrostates to compute.
        n_cells : int or None
            Number of representative cells per macrostate.
        connectivity_weight : float
            Weight of the connectivity kernel mixed into the velocity kernel.
        compute_fate_probabilities : bool
            Whether to compute CellRank fate probabilities.
        fate_kwargs : dict or None
            Extra arguments forwarded to
            ``estimator.compute_fate_probabilities``.
        clean : bool
            Whether to sanitize lineage probabilities after fate computation.
        plot : bool
            Whether to plot terminal states with ``ov.pl.cell_fate``.
        basis : str
            Embedding basis for optional plotting.
        **kwargs
            Additional arguments forwarded to ``estimator.compute_macrostates``.

        Returns
        -------
        cellrank.estimators.GPCCA
            Fitted GPCCA estimator. It is also stored as
            ``self.cellrank_estimator`` and in
            ``adata.uns['velocity_cellrank']['estimator']``.
        """
        try:
            import cellrank as cr
        except ImportError as exc:
            raise build_optional_dependency_error(
                "omicverse.single.Velo.cellrank_fate",
                ("cellrank",),
                install_hint="Install with `pip install cellrank`.",
            ) from exc

        vk = cr.kernels.VelocityKernel(self.adata, xkey=xkey, vkey=velocity_key)
        vk.compute_transition_matrix()
        if connectivity_weight:
            ck = cr.kernels.ConnectivityKernel(self.adata).compute_transition_matrix()
            kernel = (1 - connectivity_weight) * vk + connectivity_weight * ck
        else:
            kernel = vk

        estimator = cr.estimators.GPCCA(kernel)
        macrostate_kwargs = {'n_states': n_states}
        if n_cells is not None:
            macrostate_kwargs['n_cells'] = n_cells
        if cluster_key is not None:
            macrostate_kwargs['cluster_key'] = cluster_key
        macrostate_kwargs.update(kwargs)
        estimator.compute_macrostates(**macrostate_kwargs)
        requested_terminal_states = terminal_states
        missing_terminal_states = []
        terminal_states_used = None
        if terminal_states is not None:
            if isinstance(terminal_states, str):
                requested = [terminal_states]
            else:
                requested = list(terminal_states)
            available = state_names(getattr(estimator, 'macrostates', None))
            if not available:
                terminal_states_used = requested
            else:
                available_set = set(available)
                terminal_states_used = [state for state in requested if state in available_set]
                missing_terminal_states = [
                    state for state in requested if state not in available_set
                ]
                if missing_terminal_states:
                    warnings.warn(
                        "Some requested terminal states are not CellRank macrostates "
                        f"and will be skipped: {missing_terminal_states}. "
                        f"Valid macrostates are: {available}.",
                        UserWarning,
                        stacklevel=2,
                    )
            if terminal_states_used:
                estimator.set_terminal_states(terminal_states_used)
            else:
                warnings.warn(
                    "None of the requested terminal states were found in the "
                    "CellRank macrostates. Skipping `set_terminal_states()`.",
                    UserWarning,
                        stacklevel=2,
                    )
        if compute_fate_probabilities:
            if terminal_states is not None and not terminal_states_used:
                warnings.warn(
                    "Skipping `compute_fate_probabilities()` because no valid "
                    "terminal states were selected.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                if fate_kwargs is None:
                    fate_kwargs = {}
                estimator.compute_fate_probabilities(**fate_kwargs)
                if clean:
                    clean_lineages(self.adata)
        if plot:
            if terminal_states is not None and not terminal_states_used:
                warnings.warn(
                    "Skipping terminal-state plot because no valid terminal "
                    "states were selected.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                from .. import pl as _pl
                _pl.cell_fate(
                    estimator,
                    which='terminal',
                    basis=basis,
                    legend_loc='right',
                    s=100,
                )

        self.cellrank_kernel = kernel
        self.cellrank_estimator = estimator
        self.adata.uns['velocity_cellrank'] = {
            'velocity_key': velocity_key,
            'xkey': xkey,
            'cluster_key': cluster_key,
            'terminal_states': terminal_states_used,
            'requested_terminal_states': requested_terminal_states,
            'missing_terminal_states': missing_terminal_states,
            'n_states': n_states,
            'n_cells': n_cells,
            'connectivity_weight': connectivity_weight,
            'fate_kwargs': fate_kwargs or {},
            'clean': clean,
            'estimator': estimator,
            'kernel': kernel,
        }
        return estimator

    def regvelo_perturb(
        self,
        tf,
        model=None,
        adata=None,
        effects=0,
        cutoff=0.001,
        batch_size=None,
        **kwargs,
    ):
        """
        Run RegVelo's native in-silico TF regulon blockade from a Velo object.

        Parameters
        ----------
        tf : str or list of str
            Transcription factor(s) to perturb.
        model : str or RegVelo model or None
            Saved model path or in-memory model. If ``None``, the method uses
            ``adata.uns['regvelo_model_path']`` or ``adata.uns['regvelo_model']``.
        adata : AnnData or None
            AnnData used for perturbation. Defaults to ``self.adata``.
        effects, cutoff, batch_size
            Arguments forwarded to ``regvelo.tl.in_silico_block_simulation``.
        **kwargs
            Additional arguments forwarded to RegVelo.

        Returns
        -------
        tuple
            ``(perturbed_adata, perturbed_model)`` returned by RegVelo.
        """
        _, _, rgv = self._import_regvelo()
        model = self._resolve_regvelo_model(model)
        adata = self.adata if adata is None else adata
        if batch_size is None:
            batch_size = adata.n_obs

        perturb = getattr(getattr(rgv, 'tl', None), 'in_silico_block_simulation', None)
        if perturb is None:
            perturb = getattr(getattr(rgv, 'tools', None), 'in_silico_block_simulation', None)
        if perturb is None:
            raise AttributeError(
                "regvelo.tl.in_silico_block_simulation is required for RegVelo perturbation"
            )

        if isinstance(model, (str, bytes)) or hasattr(model, '__fspath__'):
            result = perturb(
                model=model,
                adata=adata,
                TF=tf,
                effects=effects,
                cutoff=cutoff,
                batch_size=batch_size,
                **kwargs,
            )
        elif hasattr(model, 'save'):
            import tempfile

            with tempfile.TemporaryDirectory(prefix='ov_regvelo_perturb_') as tmpdir:
                model.save(tmpdir, overwrite=True)
                result = perturb(
                    model=tmpdir,
                    adata=adata,
                    TF=tf,
                    effects=effects,
                    cutoff=cutoff,
                    batch_size=batch_size,
                    **kwargs,
                )
        else:
            raise TypeError(
                "RegVelo perturbation requires a saved model path or a saveable "
                "RegVelo model object. Pass `model='path/to/model'` or run "
                "`cal_velocity(method='regvelo', model_save_path=...)` first."
            )
        self.adata.uns.setdefault('regvelo_perturbations', {})[str(tf)] = {
            'effects': effects,
            'cutoff': cutoff,
            'batch_size': batch_size,
        }
        return result

    def perturbation_effect(
        self,
        perturbed_adata,
        terminal_states,
        method='regvelo',
        key_prefix='perturbation effect on ',
        **kwargs,
    ):
        """
        Write single-cell perturbation effects back to ``adata.obs``.

        Currently ``method='regvelo'`` wraps RegVelo's fate-probability
        perturbation effect helper while keeping the public OmicVerse API
        method-agnostic for future perturbation backends.

        Parameters
        ----------
        perturbed_adata : anndata.AnnData
            Perturbed object containing fate probabilities.
        terminal_states : str or sequence of str
            Terminal state(s) for which perturbation effects are computed.
        method : {'regvelo'}
            Perturbation backend.
        key_prefix : str
            Prefix for columns written to ``adata.obs``.
        **kwargs
            Additional arguments forwarded to RegVelo's perturbation-effect
            helper.

        Returns
        -------
        anndata.AnnData
            Baseline object with perturbation-effect columns in ``obs``.
        """
        if method != 'regvelo':
            raise NotImplementedError(
                "Only method='regvelo' is currently supported for perturbation_effect()."
            )
        _, _, rgv = self._import_regvelo()

        effect = getattr(getattr(rgv, 'tools', None), 'perturbation_effect', None)
        if effect is None:
            effect = getattr(getattr(rgv, 'tl', None), 'perturbation_effect', None)
        if effect is None:
            raise AttributeError(
                "regvelo.tools.perturbation_effect is required for perturbation_effect()"
            )

        result = effect(
            adata_perturb=perturbed_adata,
            adata=self.adata,
            terminal_state=terminal_states,
            **kwargs,
        )
        if result is not None:
            self.adata = result

        default_prefix = 'perturbation effect on '
        if key_prefix != default_prefix:
            states = [terminal_states] if isinstance(terminal_states, str) else list(terminal_states)
            for state in states:
                default_key = f'{default_prefix}{state}'
                custom_key = f'{key_prefix}{state}'
                if default_key in self.adata.obs and custom_key != default_key:
                    self.adata.obs[custom_key] = self.adata.obs[default_key]
                    del self.adata.obs[default_key]
        return self.adata

    def cell_fate_perturbation(
        self,
        perturbed,
        terminal_states=None,
        method='regvelo',
        score_method='likelihood',
        solver='gmres',
        **kwargs,
    ):
        """
        Summarize perturbation effects on terminal cell fates.

        The returned table is also stored in
        ``adata.uns['cell_fate_perturbation']``.

        Parameters
        ----------
        perturbed : anndata.AnnData or dict
            Perturbed object, or mapping of perturbation names to objects.
        terminal_states : str, sequence of str or None
            Terminal states to summarize.
        method : {'regvelo'}
            Perturbation backend.
        score_method : str
            Scoring method forwarded to RegVelo.
        solver : str
            Linear solver forwarded to RegVelo.
        **kwargs
            Additional arguments forwarded to RegVelo's
            ``cellfate_perturbation`` function.

        Returns
        -------
        pandas.DataFrame
            Fate perturbation summary table.
        """
        if method != 'regvelo':
            raise NotImplementedError(
                "Only method='regvelo' is currently supported for cell_fate_perturbation()."
            )
        _, _, rgv = self._import_regvelo()

        metrics = getattr(rgv, 'metrics', None)
        if metrics is None:
            metrics = getattr(rgv, 'mt', None)
        cellfate_perturbation = getattr(metrics, 'cellfate_perturbation', None)
        if cellfate_perturbation is None:
            raise AttributeError(
                "regvelo.metrics.cellfate_perturbation is required for "
                "cell_fate_perturbation()"
            )

        if not isinstance(perturbed, dict):
            perturbed = {'perturbation': perturbed}

        result = cellfate_perturbation(
            perturbed=perturbed,
            baseline=self.adata,
            terminal_state=terminal_states,
            method=score_method,
            solver=solver,
            **kwargs,
        )
        self.adata.uns['cell_fate_perturbation'] = result
        return result

    def velocity_effect(
        self,
        perturbed_adata,
        baseline_velocity_key='velo_regvelo',
        perturbed_velocity_key='velocity',
        effect_key=None,
        target=None,
    ):
        """
        Compute per-cell velocity direction change after perturbation.

        The effect is ``1 - cosine_similarity`` between baseline and perturbed
        velocity vectors and is written to ``adata.obs[effect_key]``. Cells and
        genes are aligned by ``obs_names`` and ``var_names`` before comparison;
        both objects must contain the same unique labels.

        Parameters
        ----------
        perturbed_adata : anndata.AnnData
            Perturbed object containing a velocity layer.
        baseline_velocity_key : str
            Velocity layer in ``self.adata``.
        perturbed_velocity_key : str
            Velocity layer in ``perturbed_adata``.
        effect_key : str or None
            Observation column for the output. If ``None``, a key is derived
            from ``target``.
        target : str, sequence of str or None
            Perturbation label used when deriving ``effect_key``.

        Returns
        -------
        pandas.Series
            Per-cell velocity direction-change score.
        """
        import numpy as _np
        from scipy.sparse import issparse as _issparse

        if baseline_velocity_key not in self.adata.layers:
            raise KeyError(f"adata.layers has no baseline velocity key {baseline_velocity_key!r}.")
        if perturbed_velocity_key not in perturbed_adata.layers:
            raise KeyError(
                f"perturbed_adata.layers has no velocity key {perturbed_velocity_key!r}."
            )

        def _as_dense(value):
            return value.toarray() if _issparse(value) else _np.asarray(value)

        for axis_name, baseline_names, perturbed_names in (
            ('obs', self.adata.obs_names, perturbed_adata.obs_names),
            ('var', self.adata.var_names, perturbed_adata.var_names),
        ):
            if not baseline_names.is_unique or not perturbed_names.is_unique:
                raise ValueError(
                    f"Baseline and perturbed `{axis_name}_names` must be unique "
                    "to align velocity vectors safely."
                )
            missing = baseline_names.difference(perturbed_names)
            extra = perturbed_names.difference(baseline_names)
            if len(missing) or len(extra):
                raise ValueError(
                    f"Baseline and perturbed `{axis_name}_names` must contain the "
                    "same labels; "
                    f"missing from perturbed: {missing.tolist()}, "
                    f"extra in perturbed: {extra.tolist()}."
                )

        obs_indexer = perturbed_adata.obs_names.get_indexer(self.adata.obs_names)
        var_indexer = perturbed_adata.var_names.get_indexer(self.adata.var_names)
        baseline_velocity = _as_dense(self.adata.layers[baseline_velocity_key])
        perturbed_velocity = perturbed_adata.layers[perturbed_velocity_key]
        perturbed_velocity = perturbed_velocity[obs_indexer, :][:, var_indexer]
        perturbed_velocity = _as_dense(perturbed_velocity)
        if baseline_velocity.shape != perturbed_velocity.shape:
            raise ValueError(
                "Baseline and perturbed velocity layers must have the same shape; "
                f"got {baseline_velocity.shape} and {perturbed_velocity.shape}."
            )

        numerator = _np.sum(baseline_velocity * perturbed_velocity, axis=1)
        denominator = (
            _np.linalg.norm(baseline_velocity, axis=1)
            * _np.linalg.norm(perturbed_velocity, axis=1)
        )
        values = 1 - numerator / _np.maximum(denominator, 1e-12)

        if effect_key is None:
            if target is None:
                effect_key = 'velocity_effect'
            elif isinstance(target, (list, tuple, set)):
                effect_key = f'{"+".join(map(str, target))}_velocity_effect'
            else:
                effect_key = f'{target}_velocity_effect'

        self.adata.obs[effect_key] = values
        return self.adata.obs[effect_key]


    def _graphvelo_cal(self,backend='dynamo',xkey='Ms',vkey='velocity_S',n_jobs=1,**kwargs):
        """
        Internal helper to run GraphVelo refinement from selected backend outputs.

        Parameters
        ----------
        backend : {'dynamo', 'scvelo'}
            Source backend used to initialize velocity inputs.
        xkey : str
            Expression layer key used by GraphVelo.
        vkey : str
            Velocity layer key to overwrite with refined velocity.
        n_jobs : int
            Number of CPU jobs for GraphVelo training.
        **kwargs
            Additional keyword arguments forwarded to ``GraphVelo``.
        """
        from ..external.graphvelo.graph_velocity import GraphVelo
        from ..external.graphvelo.utils import mack_score, adj_to_knn
        if backend == 'dynamo':
            gv=GraphVelo(self.adata, xkey=xkey, vkey=vkey,**kwargs)
            gv.train(n_jobs=n_jobs)
        elif backend == 'scvelo':
            indices, _ = adj_to_knn(self.adata.obsp['connectivities'])
            self.adata.uns['neighbors']['indices'] = indices
            gene_subset=self.adata.var.loc[self.adata.var['velocity_genes']].index.tolist()
            gv=GraphVelo(self.adata, xkey=xkey, vkey=vkey,gene_subset=gene_subset,**kwargs)
            gv.train(n_jobs=n_jobs)
        else:
            raise ValueError(f"Backend {backend} not supported")
        
        self.adata.layers[vkey] = gv.project_velocity(self.adata.layers['M_s'])
        self.adata.obsm['gv_pca'] = gv.project_velocity(self.adata.obsm['X_pca'])
        self.adata.obsm['gv_umap'] = gv.project_velocity(self.adata.obsm['X_umap'])


    def _latentvelo_cal(
        self,param_name_key='tmp/latentvelo_params',
        velocity_key='velocity_S',
        celltype_key=None,
        batch_key=None,
        latentvelo_VAE_kwargs={},
        use_rep=None,
        **kwargs):
        """
        Internal helper to train latentvelo and export velocity outputs.

        Parameters
        ----------
        param_name_key : str
            Directory used to store latentvelo training artifacts.
        velocity_key : str
            Velocity key prefix used for marker-gene flags.
        celltype_key : str or None
            Optional cell-type annotation key for AnnotVAE mode.
        batch_key : str or None
            Optional batch key used in latentvelo preprocessing.
        latentvelo_VAE_kwargs : dict
            Additional keyword arguments for VAE/AnnotVAE initialization.
        use_rep : str or None
            Representation key used in latentvelo preprocessing.
        **kwargs
            Additional training arguments passed to latentvelo ``train``.
        """
        try:
            import torchdiffeq
        except:
            print(f"{Colors.WARNING}torchdiffeq not installed, please install it with 'pip install torchdiffeq'.{Colors.ENDC}")
            raise ValueError("torchdiffeq not installed")
        import os
        os.makedirs(param_name_key, exist_ok=True)
        # latentvelo
        from ..external.latentvelo.models.vae_model import VAE
        from ..external.latentvelo.models.annot_vae_model import AnnotVAE
        from ..external.latentvelo.train import train
        from ..external.latentvelo.utils import standard_clean_recipe, anvi_clean_recipe
        # Optional device override for latentvelo stack
        device_override = kwargs.pop('device', None)
        if device_override is not None:
            from ..external.latentvelo import trainer as lv_trainer
            from ..external.latentvelo import trainer_anvi as lv_trainer_anvi
            from ..external.latentvelo import trainer_atac as lv_trainer_atac
            from ..external.latentvelo import output_results as lv_out_mod
            from ..external.latentvelo import utils as lv_utils
            for m in (lv_trainer, lv_trainer_anvi, lv_trainer_atac, lv_out_mod, lv_utils):
                if hasattr(m, 'set_device'):
                    m.set_device(device_override)

        # Shared preprocessing
        if celltype_key == None:
            self.adata = standard_clean_recipe(self.adata, batch_key=batch_key, 
                        celltype_key=celltype_key, r2_adjust=True,use_rep=use_rep)

            self.model = VAE(**latentvelo_VAE_kwargs)
            epochs, vae, val_traj = train(self.model,self.adata,name=param_name_key,**kwargs)
        else:
            self.adata=anvi_clean_recipe(self.adata, celltype_key=celltype_key,
                        batch_key=batch_key,r2_adjust=True,use_rep=use_rep)
            # Get required parameters from adata
            observed = self.adata.n_vars
            celltypes = len(self.adata.obs[celltype_key].unique())
            self.model = AnnotVAE(observed=observed, celltypes=celltypes, **latentvelo_VAE_kwargs)
            epochs, vae, val_traj = train(self.model,self.adata,name=param_name_key,**kwargs)
        self.adata.uns['latentvelo_train_params'] = {
                    'epochs': epochs,
                    'vae': vae,
                    'val_traj': val_traj
                }
        from ..external.latentvelo.output_results import output_results as lv_output
        latent_data, adta = lv_output(self.model,self.adata,gene_velocity=True,)
        self.adata.var[f'{velocity_key}_genes'] = self.adata.var['velocity_genes']
        #covert to csr
        import scipy as sp
        if not issparse(adta.layers['velo_s']):
            self.adata.layers['velocity_S'] = sp.sparse.csr_matrix(adta.layers['velo_s'])
        else:
            self.adata.layers['velocity_S'] = adta.layers['velo_s']
        if not issparse(adta.layers['velo_u']):
            self.adata.layers['velocity_U'] = sp.sparse.csr_matrix(adta.layers['velo_u'])
        else:
            self.adata.layers['velocity_U'] = adta.layers['velo_u']
        self.adata.obsm['X_latentvelo'] = latent_data.X
        self.adata.obsm['X_latentvelo_velo_s'] = latent_data.layers['spliced_velocity']
        self.adata.obsm['X_latentvelo_velo_u'] = latent_data.layers['unspliced_velocity']

    
    def _regvelo_cal(
        self,
        velocity_key='velocity_S',
        prior_grn=None,
        regulators=None,
        spliced_layer='Ms',
        unspliced_layer='Mu',
        n_samples=30,
        batch_size=None,
        model_load_path=None,
        model_save_path=None,
        model_overwrite=False,
        reuse_regvelo_output=False,
        regvelo_kwargs=None,
        train_kwargs=None,
        compute_velocity_graph=False,
        compute_velocity_embedding=False,
        basis='umap',
        graph_kwargs=None,
        embedding_kwargs=None,
        **kwargs):
        """Run RegVelo and export velocity outputs into ``self.adata``."""
        torch, REGVELOVI, rgv = self._import_regvelo()
        self._validate_regvelo_layers(spliced_layer, unspliced_layer)

        model_kwargs = {}
        if regvelo_kwargs:
            model_kwargs.update(regvelo_kwargs)
        model_kwargs.update(kwargs)

        train_params = {}
        if train_kwargs:
            train_params.update(train_kwargs)

        if model_load_path is not None:
            load_params = {}
            for key in ('accelerator', 'device'):
                if key in train_params:
                    load_params[key] = train_params[key]
            if 'device' not in load_params and 'devices' in train_params:
                load_params['device'] = train_params['devices']
            reg_vae = REGVELOVI.load(model_load_path, adata=self.adata, **load_params)
            self.adata.uns['regvelo_model_path'] = model_load_path
            regulators = self._prepare_regvelo_regulators(regulators)
        else:
            prior_matrix = self._prepare_regvelo_prior_grn(prior_grn, torch)
            regulators = self._prepare_regvelo_regulators(regulators)

            REGVELOVI.setup_anndata(
                self.adata,
                spliced_layer=spliced_layer,
                unspliced_layer=unspliced_layer,
            )
            reg_vae = REGVELOVI(self.adata, W=prior_matrix, regulators=regulators, **model_kwargs)
            reg_vae.train(**train_params)

            if model_save_path is not None:
                reg_vae.save(model_save_path, overwrite=model_overwrite)
                self.adata.uns['regvelo_model_path'] = model_save_path
            else:
                self.adata.uns['regvelo_model'] = reg_vae

        output_batch_size = batch_size if batch_size is not None else self.adata.n_obs
        reused_output = self._reuse_regvelo_output(velocity_key) if reuse_regvelo_output else False
        if not reused_output:
            set_output = getattr(getattr(rgv, 'tl', None), 'set_output', None)
            if set_output is None:
                raise AttributeError("regvelo.tl.set_output is required to export RegVelo outputs")
            output = set_output(
                self.adata,
                reg_vae,
                n_samples=n_samples,
                batch_size=output_batch_size,
            )
            if output is not None:
                self.adata = output

        if 'velocity' not in self.adata.layers:
            raise ValueError("RegVelo completed but did not write adata.layers['velocity']")
        self.adata.layers[velocity_key] = self.adata.layers['velocity']

        velocity_genes_key = f'{velocity_key}_genes'
        if regulators is None:
            self.adata.var[velocity_genes_key] = True
        else:
            regulator_set = set(regulators)
            self.adata.var[velocity_genes_key] = [
                gene in regulator_set for gene in self.adata.var_names
            ]

        self.adata.uns['regvelo'] = {
            'spliced_layer': spliced_layer,
            'unspliced_layer': unspliced_layer,
            'n_samples': n_samples,
            'batch_size': output_batch_size,
            'velocity_key': velocity_key,
            'n_regulators': None if regulators is None else len(regulators),
            'model_load_path': model_load_path,
            'model_save_path': model_save_path,
            'model_overwrite': model_overwrite,
            'reuse_regvelo_output': reuse_regvelo_output,
            'reused_regvelo_output': reused_output,
        }

        if compute_velocity_graph or compute_velocity_embedding:
            graph_params = {}
            if graph_kwargs:
                graph_params.update(graph_kwargs)
            graph_params.setdefault('xkey', spliced_layer)
            self.velocity_graph(basis=basis, vkey=velocity_key, **graph_params)

        if compute_velocity_embedding:
            embedding_params = {}
            if embedding_kwargs:
                embedding_params.update(embedding_kwargs)
            self.velocity_embedding(basis=basis, vkey=velocity_key, **embedding_params)

    def _import_regvelo(self):
        try:
            import torch
            from regvelo import REGVELOVI
            import regvelo as rgv
        except ImportError as exc:
            raise build_optional_dependency_error(
                "omicverse.single.Velo.cal_velocity(method='regvelo')",
                ("regvelo", "torch", "scvi"),
                install_hint="Install with `pip install regvelo scvi-tools`.",
            ) from exc
        return torch, REGVELOVI, rgv

    def _validate_regvelo_layers(self, spliced_layer, unspliced_layer):
        missing = [
            layer
            for layer in (spliced_layer, unspliced_layer)
            if layer not in self.adata.layers
        ]
        if missing:
            raise ValueError(
                "RegVelo requires spliced/unspliced moment layers. "
                f"Missing layer(s): {', '.join(missing)}"
            )

    def _prepare_regvelo_prior_grn(self, prior_grn, torch):
        if prior_grn is None:
            if 'skeleton' not in self.adata.uns:
                raise ValueError(
                    "RegVelo requires a prior GRN. Pass `prior_grn` or set "
                    "`adata.uns['skeleton']`."
                )
            prior_grn = self.adata.uns['skeleton']

        if hasattr(torch, 'is_tensor') and torch.is_tensor(prior_grn):
            prior_tensor = prior_grn
        else:
            prior_array = self._coerce_regvelo_prior_grn_array(prior_grn)
            tensor_kwargs = {}
            if hasattr(torch, 'float32'):
                tensor_kwargs['dtype'] = torch.float32
            if hasattr(torch, 'as_tensor'):
                prior_tensor = torch.as_tensor(prior_array, **tensor_kwargs)
            else:
                prior_tensor = torch.tensor(prior_array, **tensor_kwargs)

        if getattr(prior_tensor, 'shape', None) != (self.adata.n_vars, self.adata.n_vars):
            raise ValueError(
                "RegVelo prior GRN must be a square matrix aligned to "
                f"adata.var_names with shape {(self.adata.n_vars, self.adata.n_vars)}; "
                f"got {getattr(prior_tensor, 'shape', None)}."
            )
        return prior_tensor.T

    def _coerce_regvelo_prior_grn_array(self, prior_grn):
        import numpy as _np
        import pandas as _pd

        if isinstance(prior_grn, _pd.DataFrame):
            genes = list(self.adata.var_names)
            if set(genes).issubset(prior_grn.index) and set(genes).issubset(prior_grn.columns):
                return prior_grn.loc[genes, genes].to_numpy()
            return self._regvelo_edgelist_to_matrix(prior_grn)
        return _np.asarray(prior_grn)

    def _regvelo_edgelist_to_matrix(self, edgelist):
        import numpy as _np

        source_candidates = ('TF', 'tf', 'source', 'regulator')
        target_candidates = ('target', 'Target')
        weight_candidates = ('weight', 'importance', 'coef_abs', 'coef_mean', 'score')

        source_col = next((col for col in source_candidates if col in edgelist.columns), None)
        target_col = next((col for col in target_candidates if col in edgelist.columns), None)
        if source_col is None or target_col is None:
            raise ValueError(
                "RegVelo prior GRN DataFrame must be either a square adjacency "
                "matrix or an edge list with TF/source/regulator and target columns."
            )

        weight_col = next((col for col in weight_candidates if col in edgelist.columns), None)
        gene_to_idx = {gene: idx for idx, gene in enumerate(self.adata.var_names)}
        matrix = _np.zeros((self.adata.n_vars, self.adata.n_vars), dtype=_np.float32)
        for _, row in edgelist.iterrows():
            source = row[source_col]
            target = row[target_col]
            if source not in gene_to_idx or target not in gene_to_idx:
                continue
            weight = row[weight_col] if weight_col is not None else 1.0
            matrix[gene_to_idx[target], gene_to_idx[source]] = weight
        return matrix

    def _prepare_regvelo_regulators(self, regulators):
        if regulators is not None:
            return list(regulators)
        if 'is_tf' in self.adata.var:
            return self.adata.var_names[self.adata.var['is_tf'].astype(bool)].tolist()
        return None

    def _orient_prior_for_regvelo(self, prior_grn, prior_orientation):
        if self._is_regvelo_prior_edgelist(prior_grn):
            import pandas as _pd

            matrix = self._regvelo_edgelist_to_matrix(prior_grn)
            return _pd.DataFrame(
                matrix,
                index=self.adata.var_names,
                columns=self.adata.var_names,
            )
        if prior_orientation == 'regulator_by_target':
            if hasattr(prior_grn, 'T'):
                return prior_grn.T
            import numpy as _np
            return _np.asarray(prior_grn).T
        if prior_orientation == 'target_by_regulator':
            return prior_grn
        raise ValueError(
            "prior_orientation must be 'regulator_by_target' or 'target_by_regulator'"
        )

    def _is_regvelo_prior_edgelist(self, prior_grn):
        if not hasattr(prior_grn, 'columns'):
            return False
        source_candidates = ('TF', 'tf', 'source', 'regulator')
        target_candidates = ('target', 'Target')
        has_source = any(col in prior_grn.columns for col in source_candidates)
        has_target = any(col in prior_grn.columns for col in target_candidates)
        return has_source and has_target

    def _resolve_regvelo_model(self, model=None):
        if model is not None:
            return model
        if 'regvelo_model_path' in self.adata.uns:
            return self.adata.uns['regvelo_model_path']
        if 'regvelo_model' in self.adata.uns:
            return self.adata.uns['regvelo_model']
        raise ValueError(
            "No RegVelo model found. Pass `model`, or run "
            "`cal_velocity(method='regvelo', model_save_path=...)` first."
        )

    def _reuse_regvelo_output(self, velocity_key):
        if velocity_key in self.adata.layers:
            self.adata.layers['velocity'] = self.adata.layers[velocity_key]
            return True
        if 'velocity' in self.adata.layers:
            return True
        return False

    def _default_velocity_color_key(self):
        for key in ('cell_type', 'clusters', 'stage', 'leiden', 'louvain'):
            if key in self.adata.obs:
                return key
        if len(self.adata.obs.columns) > 0:
            return self.adata.obs.columns[0]
        raise KeyError("No observation columns are available for coloring cells")


def velocity(adata, **kwargs):
    """
    AnnData-first convenience wrapper for :meth:`Velo.cal_velocity`.

    Results are written into ``adata`` and the same AnnData object is returned.
    This keeps the velocity workflow aligned with other OmicVerse functions
    that use AnnData as the primary result container.

    Parameters
    ----------
    adata : anndata.AnnData
        Input object. Velocity results are written in place.
    **kwargs
        Arguments forwarded to :meth:`Velo.cal_velocity`.

    Returns
    -------
    anndata.AnnData or None
        Return value from :meth:`Velo.cal_velocity`.
    """
    velo = Velo(adata)
    return velo.cal_velocity(**kwargs)


def cellrank_fate(adata, **kwargs):
    """
    AnnData-first convenience wrapper for :meth:`Velo.cellrank_fate`.

    The CellRank estimator and kernel are stored in
    ``adata.uns['velocity_cellrank']`` for reuse by ``ov.pl.cell_fate(adata)``.

    Parameters
    ----------
    adata : anndata.AnnData
        Object containing velocity layers.
    **kwargs
        Arguments forwarded to :meth:`Velo.cellrank_fate`.

    Returns
    -------
    cellrank.estimators.GPCCA
        Fitted CellRank estimator.
    """
    velo = Velo(adata)
    return velo.cellrank_fate(**kwargs)


def state_names(estimator):
    """
    Return state names from a CellRank estimator or state container.

    Parameters
    ----------
    estimator
        CellRank estimator, categorical state object, Lineage-like object or
        array-like object with state labels.

    Returns
    -------
    list of str
        State names. Returns an empty list if names cannot be inferred.
    """
    states = getattr(estimator, "macrostates", estimator)
    if states is None:
        return []
    if hasattr(states, "cat"):
        return [str(state) for state in list(states.cat.categories)]
    names = getattr(states, "names", None)
    if names is not None:
        return [str(state) for state in list(names)]
    try:
        import pandas as _pd
        return sorted(map(str, _pd.unique(states)))
    except Exception:
        return []


def clean_lineages(adata, key="lineages_fwd"):
    """
    Make lineage probabilities finite, non-negative, and row-normalized.

    Parameters
    ----------
    adata : anndata.AnnData
        Object containing CellRank lineage probabilities in ``adata.obsm``.
    key : str
        Key in ``adata.obsm`` containing a CellRank ``Lineage`` object or a
        numeric lineage-probability matrix.

    Returns
    -------
    cellrank.Lineage
        Cleaned lineage probabilities written back to ``adata.obsm[key]``.
    """
    import numpy as _np

    if key not in adata.obsm:
        raise KeyError(f"adata.obsm has no lineage key {key!r}.")

    lineages = adata.obsm[key]
    values = _np.asarray(lineages, dtype=float).copy()
    names = list(getattr(lineages, "names", [f"lineage_{i}" for i in range(values.shape[1])]))
    colors = getattr(lineages, "colors", None)

    changed = not _np.isfinite(values).all() or _np.any(values < 0)
    values = _np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values[values < 0] = 0.0

    row_sums = values.sum(axis=1, keepdims=True)
    nonzero = row_sums[:, 0] > 0
    if _np.any(nonzero):
        values[nonzero] = values[nonzero] / row_sums[nonzero]
    if _np.any(~nonzero):
        values[~nonzero] = 1.0 / values.shape[1]
        changed = True

    try:
        from cellrank import Lineage
        adata.obsm[key] = Lineage(values, names=names, colors=colors)
    except ImportError as exc:
        raise build_optional_dependency_error(
            "omicverse.single.clean_lineages",
            ("cellrank",),
            install_hint="Install with `pip install cellrank`.",
        ) from exc

    if changed:
        adata.uns.setdefault("lineage_clean", {})[key] = {
            "finite": True,
            "non_negative": True,
            "row_normalized": True,
        }
    return adata.obsm[key]


def perturbation_effect(adata, perturbed_adata, **kwargs):
    """
    AnnData-first convenience wrapper for :meth:`Velo.perturbation_effect`.

    Parameters
    ----------
    adata : anndata.AnnData
        Baseline object containing fate probabilities.
    perturbed_adata : anndata.AnnData
        Perturbed object containing fate probabilities.
    **kwargs
        Arguments forwarded to :meth:`Velo.perturbation_effect`.

    Returns
    -------
    anndata.AnnData
        Baseline object with perturbation-effect columns added to ``obs``.
    """
    velo = Velo(adata)
    return velo.perturbation_effect(perturbed_adata, **kwargs)


def cell_fate_perturbation(adata, perturbed, **kwargs):
    """
    AnnData-first convenience wrapper for :meth:`Velo.cell_fate_perturbation`.

    Parameters
    ----------
    adata : anndata.AnnData
        Baseline object containing CellRank fate probabilities.
    perturbed : anndata.AnnData or dict
        Perturbed object or mapping of perturbation names to perturbed objects.
    **kwargs
        Arguments forwarded to :meth:`Velo.cell_fate_perturbation`.

    Returns
    -------
    pandas.DataFrame
        Fate perturbation summary table returned by RegVelo.
    """
    velo = Velo(adata)
    return velo.cell_fate_perturbation(perturbed, **kwargs)


def velocity_effect(adata, perturbed_adata, **kwargs):
    """
    AnnData-first convenience wrapper for :meth:`Velo.velocity_effect`.

    Parameters
    ----------
    adata : anndata.AnnData
        Baseline object containing a velocity layer.
    perturbed_adata : anndata.AnnData
        Perturbed object containing a velocity layer.
    **kwargs
        Arguments forwarded to :meth:`Velo.velocity_effect`.

    Returns
    -------
    pandas.Series
        Per-cell velocity direction-change score written to ``adata.obs``.
    """
    velo = Velo(adata)
    return velo.velocity_effect(perturbed_adata, **kwargs)



import warnings

import numpy as np
from scipy.sparse import issparse




# TODO: Addd docstrings
def quiver_autoscale(X_emb, V_emb):
    """Estimate a quiver scale factor from embedding coordinates and vectors.

    Parameters
    ----------
    X_emb : array-like
        Embedding coordinates with shape ``(n_cells, 2)``.
    V_emb : array-like
        Velocity vectors projected into embedding space.

    Returns
    -------
    float
        Suggested quiver scale normalized to embedding magnitude.
    """
    import matplotlib.pyplot as pl

    scale_factor = np.abs(X_emb).max()  # just so that it handles very large values
    fig, ax = pl.subplots()
    Q = ax.quiver(
        X_emb[:, 0] / scale_factor,
        X_emb[:, 1] / scale_factor,
        V_emb[:, 0],
        V_emb[:, 1],
        angles="xy",
        scale_units="xy",
        scale=None,
    )
    Q._init()
    fig.clf()
    pl.close(fig)
    return Q.scale / scale_factor


def velocity_embedding(
    data,
    basis=None,
    vkey="velocity",
    scale=10,
    self_transitions=True,
    use_negative_cosines=True,
    direct_pca_projection=None,
    retain_scale=False,
    autoscale=True,
    all_comps=True,
    T=None,
    copy=False,
):
    r"""Projects the single cell velocities into any embedding.

    Given normalized difference of the embedding positions

    .. math::
        \tilde \delta_{ij} = \frac{x_j-x_i}{\left\lVert x_j-x_i \right\rVert},

    the projections are obtained as expected displacements with respect to the
    transition matrix :math:`\tilde \pi_{ij}` as

    .. math::
        \tilde \nu_i = E_{\tilde \pi_{i\cdot}} [\tilde \delta_{i \cdot}]
        = \sum_{j \neq i} \left( \tilde \pi_{ij} - \frac1n \right) \tilde
        \delta_{ij}.


    Parameters
    ----------
    data : AnnData
        AnnData containing precomputed velocity layers and embeddings.
    basis : str or None
        Embedding basis key without ``X_`` prefix (for example ``'umap'``).
    vkey : str
        Layer key storing velocity matrix.
    scale : int
        Gaussian-kernel scale used in transition matrix construction.
    self_transitions : bool
        Whether to allow self transitions in transition matrix.
    use_negative_cosines : bool
        Whether negative cosine transitions contribute opposite-direction vectors.
    direct_pca_projection : bool or None
        Whether to directly project velocity to PCA space without graph-based transport.
    retain_scale : bool
        Whether to keep high-dimensional scale in projected velocities.
    autoscale : bool
        Whether to automatically rescale projected velocity vectors.
    all_comps : bool
        Whether to use all embedding components.
    T : csr_matrix or None
        Optional precomputed transition matrix.
    copy : bool
        Whether to return a copied AnnData instead of in-place update.

    Returns
    -------
    velocity_umap: `.obsm`
        coordinates of velocity projection on embedding (e.g., basis='umap')
    """
    from scvelo import logging as logg
    from scvelo import settings
    from scvelo.core import l2_norm
    from scvelo.tools.transition_matrix import transition_matrix
    adata = data.copy() if copy else data

    if basis is None:
        keys = [
            key for key in ["pca", "tsne", "umap"] if f"X_{key}" in adata.obsm.keys()
        ]
        if len(keys) > 0:
            basis = "pca" if direct_pca_projection else keys[-1]
        else:
            raise ValueError("No basis specified")

    if f"X_{basis}" not in adata.obsm_keys():
        raise ValueError("You need to compute the embedding first.")

    if direct_pca_projection and "pca" in basis:
        logg.warn(
            "Directly projecting velocities into PCA space is for exploratory analysis "
            "on principal components.\n"
            "         It does not reflect the actual velocity field from high "
            "dimensional gene expression space.\n"
            "         To visualize velocities, consider applying "
            "`direct_pca_projection=False`.\n"
        )

    logg.info("computing velocity embedding", r=True)

    if issparse(adata.layers[vkey]):
        V=adata.layers[vkey].toarray()
    else:
        V = adata.layers[vkey]
    vgenes = np.ones(adata.n_vars, dtype=bool)
    if f"{vkey}_genes" in adata.var.keys():
        vgenes &= np.array(adata.var[f"{vkey}_genes"], dtype=bool)
    vgenes &= ~np.isnan(V.sum(0))
    V = V[:, vgenes]

    if direct_pca_projection and "pca" in basis:
        PCs = adata.varm["PCs"] if all_comps else adata.varm["PCs"][:, :2]
        PCs = PCs[vgenes]

        X_emb = adata.obsm[f"X_{basis}"]
        V_emb = (V - V.mean(0)).dot(PCs)

    else:
        X_emb = (
            adata.obsm[f"X_{basis}"] if all_comps else adata.obsm[f"X_{basis}"][:, :2]
        )
        V_emb = np.zeros(X_emb.shape)

        T = (
            transition_matrix(
                adata,
                vkey=vkey,
                scale=scale,
                self_transitions=self_transitions,
                use_negative_cosines=use_negative_cosines,
            )
            if T is None
            else T
        )
        T.setdiag(0)
        T.eliminate_zeros()

        densify = adata.n_obs < 1e4
        TA = T.toarray() if densify else None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(adata.n_obs):
                indices = T[i].indices
                dX = X_emb[indices] - X_emb[i, None]  # shape (n_neighbors, 2)
                if not retain_scale:
                    dX /= l2_norm(dX)[:, None]
                dX[np.isnan(dX)] = 0  # zero diff in a steady-state
                probs = TA[i, indices] if densify else T[i].data
                V_emb[i] = probs.dot(dX) - probs.mean() * dX.sum(0)

        if retain_scale:
            X = (
                adata.layers["Ms"]
                if "Ms" in adata.layers.keys()
                else adata.layers["spliced"]
            )
            delta = T.dot(X[:, vgenes]) - X[:, vgenes]
            if issparse(delta):
                delta = delta.toarray()
            cos_proj = (V * delta).sum(1) / l2_norm(delta)
            V_emb *= np.clip(cos_proj[:, None] * 10, 0, 1)

    if autoscale:
        V_emb /= 3 * quiver_autoscale(X_emb, V_emb)

    if f"{vkey}_params" in adata.uns.keys():
        adata.uns[f"{vkey}_params"]["embeddings"] = (
            []
            if "embeddings" not in adata.uns[f"{vkey}_params"]
            else list(adata.uns[f"{vkey}_params"]["embeddings"])
        )
        adata.uns[f"{vkey}_params"]["embeddings"].extend([basis])

    vkey += f"_{basis}"
    adata.obsm[vkey] = V_emb

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint("added\n" f"    '{vkey}', embedded velocity vectors (adata.obsm)")

    return adata if copy else None

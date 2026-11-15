from .._settings import Colors
import numpy as np
import anndata as ad
from anndata import AnnData
from .._registry import register_function
from ..report._provenance import tracked, note


@register_function(
    aliases=[
        "参考映射注释",
        "reference mapping annotation",
        "细胞类型参考映射注释",
        "cell type reference mapping annotation",
        "细胞类型图谱注释",
        "cell type atlas annotation",
        "AnnotationRef",
        "reference transfer annotation",
        "query to reference label transfer",
    ],
    category="single",
    description="Reference-to-query label transfer class that integrates two AnnData objects and assigns cell types to query cells using weighted kNN in an integrated latent space.",
    prerequisites={
        'optional_functions': ['pp.preprocess', 'single.batch_correction']
    },
    requires={'var': ['shared genes between query and reference'], 'obs': ['reference celltype labels']},
    produces={
        'obsm': ['X_pca_harmony_anno', 'X_scVI_anno', 'X_scanorama_anno'],
        'obs': ['harmony_prediction', 'scVI_prediction', 'scanorama_prediction']
    },
    auto_fix='escalate',
    examples=[
        "ref_anno = ov.single.AnnotationRef(adata_query, adata_ref, celltype_key='celltype')",
        "ref_anno.train(method='harmony')",
        "adata_query = ref_anno.predict(method='harmony', n_neighbors=15)"
    ],
    related=[
        "single.Annotation",
        "single.batch_correction",
        "utils.weighted_knn_trainer",
        "utils.weighted_knn_transfer",
    ]
)
class AnnotationRef(object):
    """
    Reference-based label transfer helper for single-cell annotation.
    
    Parameters
    ----------
    adata_query : AnnData
        Query AnnData that needs cell-type annotation.
    adata_ref : AnnData
        Reference AnnData with known cell-type labels.
    celltype_key : str
        Column name in ``adata_ref.obs`` containing reference cell-type labels.
    
    Returns
    -------
    None
        Initializes concatenated query/reference data and checks feature overlap.
    """
    def __init__(self, adata_query: AnnData, adata_ref: AnnData, celltype_key: str = 'celltype'):
        self.adata_query = adata_query
        self.adata_ref = adata_ref
        self.celltype_key = celltype_key
        # TOSICA-trained classifier (populated by train(method='TOSICA'))
        self._tosica_obj = None

        #check var names 
        both_var_names = list(set(adata_query.var_names) & set(adata_ref.var_names))
        if len(both_var_names) == 0:
            raise ValueError("Query and reference adata have no common var names")
        else:
            self.adata_query = adata_query[:, both_var_names]
            self.adata_ref = adata_ref[:, both_var_names]

        #check values is log normalized or not
        if self.adata_query.X.max() < np.log1p(1e4):
            print(f"{Colors.WARNING}Query adata is log normalized{Colors.ENDC}")
            print(f"{Colors.WARNING}Please run `ov.pp.recover_counts` to recover the counts before concatenation{Colors.ENDC}")
        if self.adata_ref.X.max() < np.log1p(1e4):
            print(f"{Colors.WARNING}Reference adata is log normalized{Colors.ENDC}")
            print(f"{Colors.WARNING}Please run `ov.pp.recover_counts` to recover the counts before concatenation{Colors.ENDC}")
        
        
        self.adata_new=ad.concat(
            {'ref':self.adata_ref,
            'query':self.adata_query},
            label='integrate_batch'
        )
        print(f"Concatenated adata saved to self.adata_new")

    def preprocess(self,mode='shiftlog|pearson',n_HVGs=3000,batch_key='integrate_batch'):
        """
        Preprocess concatenated query/reference data for robust label transfer.

        Parameters
        ----------
        mode : str
            Preprocessing mode string passed to ``ov.pp.preprocess``.
        n_HVGs : int
            Number of highly variable genes retained.
        batch_key : str
            Batch key used for HVG selection and integration.

        Returns
        -------
        None
            Updates ``self.adata_new`` with HVG selection, scaling, and PCA.

        Examples
        --------
        >>> ar.preprocess(mode='shiftlog|pearson', n_HVGs=3000)
        """
        from ..pp._preprocess import preprocess,scale,pca
        self.adata_new=preprocess(self.adata_new,mode=mode,
                       n_HVGs=n_HVGs,batch_key=batch_key)
        self.adata_new = self.adata_new[:, self.adata_new.var.highly_variable_features]
        scale(self.adata_new)
        pca(self.adata_new,layer='scaled',n_pcs=50)

    @tracked("AnnotationRef.train", "ov.single.AnnotationRef.train",
             adata_attr="adata_query")
    def train(
        self,method='harmony',
        **kwargs
    ):
        """
        Train/compute an integrated embedding used for reference label transfer.

        Parameters
        ----------
        method : {'harmony', 'scVI', 'scanorama'}
            Integration backend used to create shared latent space for transfer.
        **kwargs
            Additional arguments forwarded to ``single.batch_correction``.

        Returns
        -------
        anndata.AnnData
            Query AnnData with integrated embedding copied to ``.obsm``.

        Examples
        --------
        >>> ar.train(method='harmony')
        """
        # train() builds an integrated embedding under a fixed obsm key
        # (X_<method>_anno on the query). Declare viz/backend up front;
        # nested batch_correction() is silenced by the @tracked stack.
        _basis_for_train = {
            "harmony":   "X_pca_harmony_anno",
            "scVI":      "X_scVI_anno",
            "scanorama": "X_scanorama_anno",
        }.get(method)
        if _basis_for_train is not None:
            note(backend=f"AnnotationRef · {method}",
                 viz=[{"function": "ov.pl.embedding",
                        "kwargs": {"basis": _basis_for_train,
                                    "color": "integrate_batch",
                                    "frameon": "small"}}])

        from ._batch import batch_correction
        if method=='harmony':
            batch_correction(self.adata_new,batch_key='integrate_batch',methods='harmony',**kwargs)
            self.adata_query.obsm['X_pca_harmony_anno']=self.adata_new[self.adata_query.obs.index].obsm['X_pca_harmony']
            self.adata_ref.obsm['X_pca_harmony_anno']=self.adata_new[self.adata_ref.obs.index].obsm['X_pca_harmony']
            print(f"Harmony integrated embeddings saved to self.adata_query.obsm['X_pca_harmony'] and self.adata_ref.obsm['X_pca_harmony']")
        elif method=='scVI':
            batch_correction(self.adata_new,batch_key='integrate_batch',methods='scVI',**kwargs)
            self.adata_query.obsm['X_scVI_anno']=self.adata_new[self.adata_query.obs.index].obsm['X_scVI']
            self.adata_ref.obsm['X_scVI_anno']=self.adata_new[self.adata_ref.obs.index].obsm['X_scVI']
            print(f"scVI integrated embeddings saved to self.adata_query.obsm['X_scVI'] and self.adata_ref.obsm['X_scVI']")
        elif method=='scanorama':
            batch_correction(self.adata_new,batch_key='integrate_batch',methods='scanorama',**kwargs)
            self.adata_query.obsm['X_scanorama_anno']=self.adata_new[self.adata_query.obs.index].obsm['X_scanorama']
            self.adata_ref.obsm['X_scanorama_anno']=self.adata_new[self.adata_ref.obs.index].obsm['X_scanorama']
            print(f"Scanorama integrated embeddings saved to self.adata_query.obsm['X_scanorama'] and self.adata_ref.obsm['X_scanorama']")
        elif method=='TOSICA':
            # TOSICA is a transformer-based ref classifier — no shared
            # embedding step. Train on the reference; the trained model
            # is stashed on `self` so a subsequent predict(method='TOSICA')
            # can use it without further setup.
            from ._tosica import pyTOSICA
            gmt_path = kwargs.pop('gmt_path', None)
            project_path = kwargs.pop('project_path', 'tosica_project')
            label_name = kwargs.pop('label_name', self.celltype_key)
            epochs = kwargs.pop('epochs', 5)
            lr = kwargs.pop('lr', 0.001)
            lrf = kwargs.pop('lrf', 0.01)
            pre_weights = kwargs.pop('pre_weights', '')
            save_path = kwargs.pop('save_path', None)
            # Pass-through TOSICA constructor knobs
            tosica_ctor_keys = {
                'mask_ratio', 'max_g', 'max_gs', 'n_unannotated',
                'embed_dim', 'depth', 'num_heads', 'batch_size',
                'device', 'auto_download',
            }
            ctor_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in tosica_ctor_keys}
            self._tosica_obj = pyTOSICA(
                adata=self.adata_ref,
                project_path=project_path,
                gmt_path=gmt_path,
                label_name=label_name,
                **ctor_kwargs,
            )
            self._tosica_obj.train(
                pre_weights=pre_weights, lr=lr, epochs=epochs, lrf=lrf,
            )
            if save_path:
                self._tosica_obj.save(save_path=save_path)
            print(
                f"TOSICA model trained on reference "
                f"(epochs={epochs}); "
                "call predict(method='TOSICA') to score the query."
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
        return self.adata_query

    @tracked("AnnotationRef.predict", "ov.single.AnnotationRef.predict",
             adata_attr="adata_query")
    def predict(self,method='harmony',n_neighbors=15,pred_key=None,uncert_key=None):
        """
        Transfer reference labels to query cells using weighted kNN.

        Parameters
        ----------
        method : {'harmony', 'scVI', 'scanorama', 'TOSICA'}
            Label-transfer backend. ``harmony``/``scVI``/``scanorama``
            run weighted kNN in a shared integrated embedding; ``TOSICA``
            uses the transformer classifier trained by
            ``train(method='TOSICA')`` to predict directly.
        n_neighbors : int
            Number of neighbors in the weighted kNN model.
        pred_key : str or None
            Output ``obs`` key for predicted labels.
        uncert_key : str or None
            Output ``obs`` key for uncertainty scores.
        
        Returns
        -------
        anndata.AnnData
            Query AnnData with predicted labels and uncertainties in ``.obs``.

        Examples
        --------
        >>> adata_q = ar.predict(method='harmony', n_neighbors=15)
        """
        # Resolve the obs columns the predict will write so the report's
        # viz spec can be declared before the actual transfer runs.
        _pred = pred_key or {
            "harmony":   "harmony_prediction",
            "scVI":      "scVI_prediction",
            "scanorama": "scanorama_prediction",
        }.get(method, f"{method}_prediction")
        note(backend=f"AnnotationRef · method={method}",
             viz=([{"function": "ov.pl.embedding",
                     "kwargs": {"basis": "X_umap",
                                 "color": _pred,
                                 "frameon": "small"}}]
                   if "X_umap" in self.adata_query.obsm else []))

        if method=='harmony':
            if pred_key is None:
                pred_key='harmony_prediction'
            if uncert_key is None:
                uncert_key='harmony_uncertainty'
            emb_key='X_pca_harmony_anno'
        elif method=='scVI':
            if pred_key is None:
                pred_key='scVI_prediction'
            if uncert_key is None:
                uncert_key='scVI_uncertainty'
            emb_key='X_scVI_anno'
        elif method=='scanorama':
            if pred_key is None:
                pred_key='scanorama_prediction'
            if uncert_key is None:
                uncert_key='scanorama_uncertainty'
            emb_key='X_scanorama_anno'
        elif method=='TOSICA':
            # TOSICA uses a trained classifier rather than weighted kNN
            # over a shared embedding. The trained model is on
            # `self._tosica_obj` (set by `train(method='TOSICA')`); call
            # its `predicted()` and merge labels back into adata_query.
            if not hasattr(self, '_tosica_obj') or self._tosica_obj is None:
                raise RuntimeError(
                    "TOSICA model not trained. Call "
                    "`train(method='TOSICA', ...)` first."
                )
            if pred_key is None:
                pred_key = 'TOSICA_prediction'
            if uncert_key is None:
                uncert_key = 'TOSICA_probability'
            new = self._tosica_obj.predicted(pre_adata=self.adata_query)
            # `new.obs` is indexed by adata_query.obs_names — copy the
            # two columns of interest back to the original query AnnData.
            self.adata_query.obs[pred_key] = (
                new.obs['Prediction']
                   .reindex(self.adata_query.obs_names)
                   .astype('category')
            )
            self.adata_query.obs[uncert_key] = (
                new.obs['Probability']
                   .reindex(self.adata_query.obs_names)
                   .astype(float)
            )
            print(f"{pred_key} saved to adata.obs['{pred_key}']")
            print(f"{uncert_key} saved to adata.obs['{uncert_key}']")
            return self.adata_query
        else:
            raise ValueError(f"Unsupported method: {method}")

        from ..utils._knn import weighted_knn_trainer,weighted_knn_transfer
        knn_transformer=weighted_knn_trainer(
            train_adata=self.adata_ref,
            train_adata_emb=emb_key,
            n_neighbors=n_neighbors,
        )
        labels,uncert=weighted_knn_transfer(
            query_adata=self.adata_query,
            query_adata_emb=emb_key,
            label_keys=self.celltype_key,
            knn_model=knn_transformer,
            ref_adata_obs=self.adata_ref.obs,
        )
        self.adata_query.obs[pred_key]=labels.loc[self.adata_query.obs.index,self.celltype_key]
        self.adata_query.obs[uncert_key]=uncert.loc[self.adata_query.obs.index,self.celltype_key]
        print(f"{pred_key} saved to adata.obs['{pred_key}']")
        print(f"{uncert_key} saved to adata.obs['{uncert_key}']")
        return self.adata_query
    
    

    

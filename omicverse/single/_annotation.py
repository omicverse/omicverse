import itertools
import json
import os
import re
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import requests
from anndata import AnnData

from ..datasets import download_data
from .._registry import register_function
from ..report._provenance import tracked, note


_PROMPT_DESCRIPTION_LIMIT = 400
_CELLXGENE_SYSTEM_PROMPT = (
    "You are a senior single-cell analysis assistant. "
    "Given candidate CellxGene collections, select the ones that best match the user's dataset description. "
    "Only choose from the provided candidates and respond with strict JSON following the requested schema."
)

_CELLTYPIST_SYSTEM_PROMPT = (
    "You are assisting with automated cell-type annotation. "
    "Given candidate CellTypist pre-trained models, choose those that best match the user's dataset description. "
    "Only select from the provided models and respond with strict JSON following the requested schema."
)

_CELLTYPIST_MODELS_URL = "https://celltypist.cog.sanger.ac.uk/models/models.json"


class _ConsoleSpinner:
    """Lightweight terminal spinner to indicate progress during LLM calls."""

    def __init__(self, message: str = "Processing...", interval: float = 0.15) -> None:
        self.message = message
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self):
        sys.stdout.write(f"{self.message} ")
        sys.stdout.flush()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        sys.stdout.write('\r' + ' ' * (len(self.message) + 4) + '\r')
        sys.stdout.flush()

    def _spin(self) -> None:
        for ch in itertools.cycle('|/-\\'):
            if self._stop_event.is_set():
                break
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(self.interval)
            sys.stdout.write('\b')


class LLMTableSelector:
    """Generic helper to select relevant table rows using an LLM."""

    def __init__(
        self,
        table: pd.DataFrame,
        id_column: str,
        description_column: str,
        name_column: Optional[str] = None,
        url_column: Optional[str] = None,
        table_label: str = 'entry',
        json_root_key: str = 'collections',
        json_id_key: Optional[str] = None,
        json_url_key: Optional[str] = None,
        json_reason_keys: Optional[List[str]] = None,
        max_prompt_rows: Optional[int] = None,
        extra_columns: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize a reusable LLM-based table selector.

        Parameters
        ----------
        table : pd.DataFrame
            Candidate table to select from.
        id_column : str
            Unique identifier column used to map LLM selections back to rows.
        description_column : str
            Text column containing biological/context descriptions.
        name_column : str or None
            Optional display name column.
        url_column : str or None
            Optional URL column for returned entries.
        table_label : str
            Human-readable record type used in prompts (for example
            ``'CellxGene collection'``).
        json_root_key : str
            Root key expected in LLM JSON response.
        json_id_key : str or None
            JSON field name that carries selected record IDs.
        json_url_key : str or None
            JSON field name that carries selected URLs.
        json_reason_keys : list or None
            Candidate JSON keys searched to extract explanation text.
        max_prompt_rows : int or None
            Maximum number of candidate rows included in one prompt.
        extra_columns : list or None
            Additional columns included in prompts/results.
        """
        self.table = table.reset_index(drop=True).copy()
        self.id_column = id_column
        self.description_column = description_column
        self.name_column = name_column
        self.url_column = url_column
        self.table_label = table_label
        self.json_root_key = json_root_key
        self.json_id_key = json_id_key or id_column
        self.json_url_key = json_url_key or url_column
        self.json_reason_keys = json_reason_keys or ['reason', 'explanation', 'note']
        self.max_prompt_rows = max_prompt_rows
        self.extra_columns = [col for col in (extra_columns or []) if col in self.table.columns]

        self.last_candidates: Optional[pd.DataFrame] = None
        self.last_prompt: Optional[str] = None
        self.last_raw_response: Optional[str] = None

    def select(
        self,
        query_text: str,
        system_prompt: str,
        llm_model: str,
        llm_api_key: str,
        llm_provider: str = 'openai',
        llm_base_url: str = 'https://api.openai.com/v1',
        llm_extra_params: Optional[Dict[str, Any]] = None,
        client_factory: Optional[Callable[..., Any]] = None,
        fallback_reason: str = 'Fallback to first entry (LLM unavailable or returned no results).',
        max_prompt_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run LLM-based candidate selection and return matched rows.

        Parameters
        ----------
        query_text : str
            Free-text dataset/task description used for matching.
        system_prompt : str
            System instruction controlling model behavior and output schema.
        llm_model : str
            Model name for the chosen provider.
        llm_api_key : str
            API key for provider authentication.
        llm_provider : str
            Provider identifier (for example ``'openai'`` or ``'ollama'``).
        llm_base_url : str
            Base URL for OpenAI-compatible providers.
        llm_extra_params : dict or None
            Additional provider-specific generation parameters.
        client_factory : callable or None
            Optional client factory returning ``(client, runtime_config)``.
        fallback_reason : str
            Reason text stored when fallback row selection is used.
        max_prompt_rows : int or None
            Per-call override of maximum prompt candidate rows.

        Returns
        -------
        pd.DataFrame
            Selected candidate rows with ``llm_reason`` column.
        """
        if self.table.empty:
            print("⚠️ No entries available for LLM selection.")
            return pd.DataFrame(columns=self._result_columns())

        limit = max_prompt_rows or self.max_prompt_rows
        candidates = self.table.head(limit) if limit else self.table
        candidates = candidates.reset_index(drop=True)
        self.last_candidates = candidates

        prompt = _build_table_prompt(
            query_text=query_text,
            candidates_df=candidates,
            id_column=self.id_column,
            description_column=self.description_column,
            name_column=self.name_column,
            url_column=self.url_column,
            json_root_key=self.json_root_key,
            json_id_key=self.json_id_key,
            json_url_key=self.json_url_key,
            table_label=self.table_label,
            extra_columns=self.extra_columns,
        )
        self.last_prompt = prompt

        llm_params = dict(llm_extra_params or {})

        try:
            if client_factory is not None:
                llm_client, runtime_config = client_factory(
                    provider=llm_provider,
                    api_key=llm_api_key,
                    model=llm_model,
                    base_url=llm_base_url,
                    extra_params=llm_params,
                )
            else:
                resolved_key = _resolve_api_key(llm_provider, llm_api_key)
                llm_client, runtime_config = _setup_llm_client(
                    llm_provider, resolved_key, llm_model, llm_base_url, llm_params
                )
        except ValueError as exc:
            print(f"⚠️ LLM setup failed ({exc}). {fallback_reason}")
            self.last_raw_response = None
            return self._fallback_dataframe(candidates, fallback_reason)

        try:
            spinner_message = f"🔍 Contacting LLM for {self.table_label} selection..."
            with _ConsoleSpinner(spinner_message):
                llm_raw_response = _call_llm_for_collections(
                    llm_client,
                    runtime_config,
                    system_prompt,
                    prompt,
                )
            self.last_raw_response = llm_raw_response
        except Exception as exc:  # pragma: no cover - runtime call errors
            print(f"⚠️ LLM query failed ({exc}). {fallback_reason}")
            self.last_raw_response = None
            return self._fallback_dataframe(candidates, fallback_reason)

        selections = _parse_llm_collection_response(self.last_raw_response, root_key=self.json_root_key)
        if not selections:
            print(f"⚠️ LLM did not return structured selections. {fallback_reason}")
            return self._fallback_dataframe(candidates, fallback_reason)

        result_df = _build_selection_dataframe(
            candidates,
            selections,
            id_column=self.id_column,
            name_column=self.name_column,
            description_column=self.description_column,
            url_column=self.url_column,
            json_id_key=self.json_id_key,
            json_url_key=self.json_url_key,
            json_reason_keys=self.json_reason_keys,
            extra_columns=self.extra_columns,
        )

        if result_df.empty:
            print(f"⚠️ LLM returned IDs that did not match candidates. {fallback_reason}")
            return self._fallback_dataframe(candidates, fallback_reason)

        return result_df

    def _result_columns(self) -> List[str]:
        cols = [self.id_column]
        if self.url_column and self.url_column not in cols:
            cols.append(self.url_column)
        if self.name_column and self.name_column not in cols:
            cols.append(self.name_column)
        if self.description_column and self.description_column not in cols:
            cols.append(self.description_column)
        for extra in self.extra_columns:
            if extra not in cols:
                cols.append(extra)
        cols.append('llm_reason')
        return cols

    def _fallback_dataframe(self, candidates: pd.DataFrame, reason: str) -> pd.DataFrame:
        if candidates.empty:
            return pd.DataFrame(columns=self._result_columns())

        row = candidates.iloc[0]
        entry: Dict[str, Any] = {self.id_column: row.get(self.id_column)}
        if self.url_column:
            entry[self.url_column] = row.get(self.url_column)
        if self.name_column:
            entry[self.name_column] = row.get(self.name_column)
        if self.description_column:
            entry[self.description_column] = row.get(self.description_column)
        for extra in self.extra_columns:
            entry[extra] = row.get(extra)
        entry['llm_reason'] = reason

        return pd.DataFrame([entry], columns=self._result_columns())


@register_function(
    aliases=[
        "单细胞自动注释",
        "single cell automatic annotation",
        "细胞类型自动注释",
        "cell type automatic annotation",
        "Annotation",
        "annotation manager",
        "reference annotation selector",
        "celltypist annotation",
        "scsa annotation",
        "gpt4celltype annotation",
        'annotate single cell',
        'annotate single cell type',
    ],
    category="single",
    description="Single-cell annotation manager supporting CellTypist, SCSA, GPT-based marker annotation, and reference-mapping workflows.",
    prerequisites={
        'optional_functions': ['pp.preprocess', 'pp.neighbors', 'pp.leiden']
    },
    requires={
        'var': ['gene identifiers'],
        'obs': ['cluster labels (recommended)']
    },
    produces={
        'obs': ['celltypist_prediction', 'scsa_prediction', 'gpt4celltype_prediction'],
        'obsm': ['celltypist_decision_matrix', 'celltypist_probability_matrix']
    },
    auto_fix='escalate',
    examples=[
        "anno = ov.single.Annotation(adata)",
        "anno.query_reference(source='celltypist', data_desc='human PBMC scRNA-seq')",
        "anno.annotate(method='celltypist', cluster_key='leiden')"
    ],
    related=[
        "single.AnnotationRef",
        "single.pySCSA",
        "single.get_celltype_marker",
        "single.gptcelltype",
    ]
)
class Annotation(object):
    """
    Unified single-cell annotation manager for cell-type labeling.
    
    Parameters
    ----------
    adata : AnnData
        Query single-cell AnnData to annotate.
    
    Returns
    -------
    None
        Initializes annotation manager state and reference caches.
    """

    def __init__(self, adata: AnnData,):
        """
        Initialize annotation manager with a query AnnData object.

        Parameters
        ----------
        adata : AnnData
            Query dataset to annotate.
        """
        self.adata = adata

        self.cellxgene_desc_df = None
        self.celltypist_models_df = None
        self._llm_client = None
        self.scsa_db_path = None
        self.scmulan_ckpt_path: Optional[str] = None
        self._llm_runtime_config: Optional[Dict[str, Any]] = None
        self._llm_config_signature: Optional[str] = None

        # Tracking last query context for inspection/debugging
        self.last_reference_query: Optional[str] = None
        self.last_reference_candidates: Optional[pd.DataFrame] = None
        self.last_reference_matches: Optional[pd.DataFrame] = None
        self.last_reference_llm_raw: Optional[str] = None
        self.last_reference_prompt: Optional[str] = None

    @tracked("Annotation.annotate", "ov.single.Annotation.annotate",
             adata_attr="adata")
    def annotate(
        self,
        method='celltypist',
        cluster_key='leiden',
        **kwargs
    ):
        """
        Annotate cells or clusters using the selected annotation backend.

        Parameters
        ----------
        method : {'celltypist', 'scsa', 'gpt4celltype', 'scMulan', 'MetaTiME', 'harmony', 'scVI', 'scanorama', 'TOSICA'}
            Annotation backend.
        cluster_key : str
            Cluster label key used by marker-based methods (SCSA/GPT4CellType).
        **kwargs
            Additional backend-specific keyword arguments. For ``scMulan``
            the most useful ones are ``ckp_path`` (override the local
            checkpoint), ``uniform_genes`` (default ``True``;
            run :func:`omicverse.external.scMulan.GeneSymbolUniform`
            so the input gene panel matches what the model expects),
            ``smoothing_threshold`` (default ``0.1``; pass ``None`` to
            skip the post-hoc smoothing pass), and ``parallel`` /
            ``n_process`` forwarded to scMulan's inference loop.

        Returns
        -------
        object
            Returns backend-specific results (for example a prediction table or updated AnnData).
            Predictions are also written into ``self.adata.obs``/``.obsm``.

        Examples
        --------
        >>> anno.annotate(method='celltypist')
        >>> anno.annotate(method='gpt4celltype', cluster_key='leiden')
        >>> anno.download_scmulan_ckpt()
        >>> anno.annotate(method='scMulan', smoothing_threshold=0.1)
        """
        # Predictions land deterministically at obs['<method>_prediction'].
        # Declare the report's viz/backend up front; @tracked discards the
        # entry on exception so failed calls never leak.
        _pred_col = f"{method}_prediction"
        note(backend=f"omicverse · method={method}",
             viz=([{"function": "ov.pl.embedding",
                     "kwargs": {"basis": "X_umap",
                                "color": _pred_col,
                                "frameon": "small"}}]
                   if "X_umap" in self.adata.obsm else []))

        if method=='celltypist':
            import celltypist
            predictions = celltypist.annotate(
                self.adata, model = self.pkl_ref,
                majority_voting = True,
                **kwargs
            )
            #return predictions
            
            self.adata.obs['celltypist_prediction'] = predictions.predicted_labels['majority_voting'].astype(str)
            self.adata.obsm['celltypist_decision_matrix']=predictions.decision_matrix
            self.adata.obsm['celltypist_probability_matrix']=predictions.probability_matrix
            print(f"Celltypist prediction saved to adata.obs['celltypist_prediction']")
            print(f"Celltypist decision matrix saved to adata.obsm['celltypist_decision_matrix']")
            print(f"Celltypist probability matrix saved to adata.obsm['celltypist_probability_matrix']")
        elif method=='scsa':
            from ._anno import pySCSA
            if self.scsa_db_path is None:
                self.scsa_db_path = self.download_scsa_db()
            scsa=pySCSA(adata=self.adata,
                        model_path=self.scsa_db_path,
                        **kwargs
            )
            if cluster_key not in self.adata.obs.columns:
                resolution = 1
                from ..pp import leiden
                leiden(self.adata, resolution=resolution, key_added=cluster_key, random_state=0, backend='python')

            anno=scsa.cell_anno(clustertype=cluster_key,
               cluster='all',rank_rep=True)
            result=scsa.cell_auto_anno(self.adata, key='scsa_prediction')
            return result
        elif method=='gpt4celltype':
            from ._anno import get_celltype_marker
            from ._gptcelltype import gptcelltype
            all_markers=get_celltype_marker(
                self.adata,clustertype=cluster_key,rank=True,
                    key='rank_genes_groups',
                    foldchange=2,topgenenumber=10
            )
            result = gptcelltype(all_markers, **kwargs)
            #return result
            #new_result={}
            #for key in result.keys():
            #    new_result[key]=result[key].split(': ')[-1].split(' (')[0].split('. ')[1]
            self.adata.obs['gpt4celltype_prediction'] = self.adata.obs[cluster_key].map(result).astype('category')
            print(f"GPT4celltype prediction saved to adata.obs['gpt4celltype_prediction']")
            return result
        elif method=='MetaTiME':
            # MetaTiME (Yan et al. *Nat Comm* 2023) projects cells onto a
            # bank of pre-trained meta-components and assigns a TME
            # cell-state label per overclustered group. The class is
            # defined in `_anno.py` and gated by `check_metatime()` at
            # __init__ — so the import is cheap even when the optional
            # `metatime` dependency isn't installed.
            from ._anno import MetaTiME

            mode = kwargs.pop('mode', 'table')
            resolution = kwargs.pop('resolution', 8)
            overcluster_key = kwargs.pop('overcluster_key', 'overcluster')
            save_obs_name = kwargs.pop('save_obs_name', 'MetaTiME')
            random_state = kwargs.pop('random_state', 0)

            time_obj = MetaTiME(self.adata, mode=mode)
            time_obj.overcluster(
                resolution=resolution,
                clustercol=overcluster_key,
                random_state=random_state,
            )
            time_obj.predictTiME(save_obs_name=save_obs_name)

            # Mirror the *_prediction convention used by celltypist / scsa
            # / gpt4celltype / scMulan so downstream tools (CellVote) can
            # treat MetaTiME like any other method without special-casing
            # the column name.
            self.adata.obs['MetaTiME_prediction'] = (
                self.adata.obs[save_obs_name].astype('category')
            )
            major_col = f'Major_{save_obs_name}'
            if major_col in self.adata.obs.columns:
                self.adata.obs['MetaTiME_major_prediction'] = (
                    self.adata.obs[major_col].astype('category')
                )
            print(
                f"MetaTiME prediction saved to adata.obs['{save_obs_name}'] "
                "(alias: adata.obs['MetaTiME_prediction'])"
            )
            return time_obj
        elif method=='scMulan':
            # Lazy-import: scMulan pulls torch + a large checkpoint loader,
            # so we only touch it when this branch is actually requested.
            from ..external import scMulan as _scmulan_mod
            from scipy.sparse import csc_matrix, issparse

            # Resolve checkpoint path (kwarg overrides instance attr).
            ckp_path = kwargs.pop('ckp_path', None) or self.scmulan_ckpt_path
            if ckp_path is None or not os.path.exists(ckp_path):
                raise FileNotFoundError(
                    "scMulan checkpoint not found at "
                    f"{ckp_path!r}. Pass `ckp_path=...` or call "
                    "`obj.download_scmulan_ckpt()` first to fetch the "
                    "pretrained weights from the Tsinghua mirror."
                )

            uniform_genes = kwargs.pop('uniform_genes', True)
            output_dir = kwargs.pop('output_dir', './data')
            output_prefix = kwargs.pop('output_prefix', 'scmulan_input')
            parallel = kwargs.pop('parallel', True)
            n_process = kwargs.pop('n_process', 1)
            smoothing_threshold = kwargs.pop('smoothing_threshold', 0.1)

            # scMulan expects CSC sparse counts on the gene panel it was
            # trained against. Do not mutate self.adata — keep a working
            # copy that we map predictions back from.
            adata_for_scml = self.adata
            if uniform_genes:
                adata_for_scml = self.adata.copy()
                if not issparse(adata_for_scml.X) or not isinstance(
                    adata_for_scml.X, csc_matrix,
                ):
                    adata_for_scml.X = csc_matrix(adata_for_scml.X)
                adata_for_scml = _scmulan_mod.GeneSymbolUniform(
                    input_adata=adata_for_scml,
                    output_dir=output_dir,
                    output_prefix=output_prefix,
                )

            # The model expects log1p-normalised counts to 1e4. Skip if
            # the matrix is already in that scale.
            try:
                _xmax = float(adata_for_scml.X.max())
            except (TypeError, ValueError):
                _xmax = None
            if _xmax is not None and _xmax > 10:
                from ..pp import normalize_total, log1p
                normalize_total(adata_for_scml, target_sum=1e4)
                log1p(adata_for_scml)

            scml = _scmulan_mod.model_inference(
                ckp_path, adata_for_scml, **kwargs,
            )
            try:
                scml.cuda_count()
            except Exception:
                # cuda_count is informational — not having CUDA is OK
                pass
            scml.get_cell_types_and_embds_for_adata(
                parallel=parallel, n_process=n_process,
            )

            # Map predictions back to self.adata.obs by obs_names.
            pred = pd.Series(
                scml.adata.obs['cell_type_from_scMulan'].values,
                index=scml.adata.obs_names,
            ).reindex(self.adata.obs_names)
            self.adata.obs['scMulan_prediction'] = pred.astype('category')
            print(
                "scMulan prediction saved to "
                "adata.obs['scMulan_prediction']"
            )

            # Optional smoothing pass — writes a separate column so the
            # raw vs smoothed labels can be compared. scMulan stores its
            # learned embedding under `obsm['X_scMulan']`, not the
            # default `X_pca` that `cell_type_smoothing` expects, so
            # point it at the right rep explicitly.
            if smoothing_threshold is not None:
                smoothing_rep = (
                    'X_scMulan'
                    if 'X_scMulan' in scml.adata.obsm
                    else 'X_pca'
                )
                _scmulan_mod.cell_type_smoothing(
                    scml.adata,
                    threshold=smoothing_threshold,
                    use_rep=smoothing_rep,
                )
                smooth_col = 'cell_type_from_mulan_smoothing'
                if smooth_col in scml.adata.obs.columns:
                    smoothed = pd.Series(
                        scml.adata.obs[smooth_col].values,
                        index=scml.adata.obs_names,
                    ).reindex(self.adata.obs_names)
                    self.adata.obs[
                        'scMulan_smoothed_prediction'
                    ] = smoothed.astype('category')
                    print(
                        "scMulan smoothed prediction saved to "
                        "adata.obs['scMulan_smoothed_prediction']"
                    )

            return scml.adata
        elif method=='harmony':
            from ._annotation_ref import AnnotationRef
            annotation_ref=AnnotationRef(adata_query=self.adata,adata_ref=self.adata_ref,celltype_key=self.celltype_key)
            annotation_ref.train(method='harmony',**kwargs)
            annotation_ref.predict(method='harmony',n_neighbors=15,
                pred_key='harmony_prediction',uncert_key='harmony_uncertainty',**kwargs)
            return annotation_ref.adata_query
        elif method=='scVI':
            from ._annotation_ref import AnnotationRef
            annotation_ref=AnnotationRef(adata_query=self.adata,adata_ref=self.adata_ref,celltype_key=self.celltype_key)
            annotation_ref.train(method='scVI',**kwargs)
            annotation_ref.predict(method='scVI',n_neighbors=15,
                pred_key='scVI_prediction',uncert_key='scVI_uncertainty',**kwargs)
            return annotation_ref.adata_query
        elif method=='scanorama':
            from ._annotation_ref import AnnotationRef
            annotation_ref=AnnotationRef(adata_query=self.adata,adata_ref=self.adata_ref,celltype_key=self.celltype_key)
            annotation_ref.train(method='scanorama',**kwargs)
            annotation_ref.predict(method='scanorama',n_neighbors=15,
                pred_key='scanorama_prediction',uncert_key='scanorama_uncertainty',**kwargs)
            return annotation_ref.adata_query
        elif method=='TOSICA':
            # TOSICA is a transformer classifier — no shared embedding
            # step. AnnotationRef.train(method='TOSICA') consumes the
            # TOSICA-specific kwargs (gmt_path / project_path / epochs /
            # lr / etc.) and stashes the trained model; .predict scores
            # the query.
            from ._annotation_ref import AnnotationRef
            annotation_ref = AnnotationRef(
                adata_query=self.adata,
                adata_ref=self.adata_ref,
                celltype_key=self.celltype_key,
            )
            # Split kwargs: train vs predict. predict takes only
            # pred_key/uncert_key here.
            pred_key = kwargs.pop('pred_key', 'TOSICA_prediction')
            uncert_key = kwargs.pop('uncert_key', 'TOSICA_probability')
            annotation_ref.train(method='TOSICA', **kwargs)
            annotation_ref.predict(
                method='TOSICA',
                pred_key=pred_key,
                uncert_key=uncert_key,
            )
            self.adata.obs[pred_key] = annotation_ref.adata_query.obs[pred_key]
            self.adata.obs[uncert_key] = annotation_ref.adata_query.obs[uncert_key]
            return annotation_ref.adata_query
        else:
            raise ValueError(f"Unsupported method: {method}")
                


    def add_reference_sc(self, reference: AnnData, celltype_key: str = 'celltype'):
        """
        Register a single-cell reference atlas for transfer-based annotation.

        Parameters
        ----------
        reference : AnnData
            Reference dataset containing annotated cell types.
        celltype_key : str
            Column in ``reference.obs`` containing cell-type labels.

        Returns
        -------
        None
            Stores reference AnnData and label key on the object.

        Examples
        --------
        >>> anno.add_reference_sc(adata_ref, celltype_key='celltype')
        """
        self.adata_ref=reference
        self.celltype_key=celltype_key


    def add_reference_pkl(self, reference: str):
        """
        Register a CellTypist model file for supervised annotation.

        Parameters
        ----------
        reference : str
            Path to a CellTypist ``.pkl`` model.

        Returns
        -------
        None
            Loads and stores the model for ``method='celltypist'`` annotation.

        Examples
        --------
        >>> anno.add_reference_pkl('Immune_All_Low.pkl')
        """
        self.pkl_ref=reference

        from celltypist import models
        self.model = models.Model.load(model = self.pkl_ref)

    def add_reference_scsa_db(self, reference: str):
        """
        Register a local SCSA marker database file.

        Parameters
        ----------
        reference : str
            Path to SCSA SQLite database file.

        Returns
        -------
        None
            Stores database path for SCSA-based annotation.

        Examples
        --------
        >>> anno.add_reference_scsa_db('temp/pySCSA_2023_v2_plus.db')
        """
        self.scsa_db_path = reference


    def download_scsa_db(self, save_path: str = 'temp/pySCSA_2023_v2_plus.db'):
        """Download SCSA database with fallback mirrors.

        Tries Stanford repository first, falls back to Figshare if download fails.

        Parameters
        ----------
        save_path : str
            Path where the downloaded SCSA SQLite database will be saved.

        Returns
        -------
        str
            Local path of downloaded database file.

        Raises
        ------
        Exception
            If download fails from all configured mirrors.
        """
        # Define download mirrors (Stanford preferred, Figshare as fallback)
        mirrors = [
            {
                'name': 'Stanford',
                'url': 'https://stacks.stanford.edu/file/cv694yk7414/pySCSA_2023_v2_plus.db'
            },
            {
                'name': 'Figshare',
                'url': 'https://figshare.com/ndownloader/files/41369037'
            }
        ]

        parent_dir = os.path.dirname(save_path)
        file_name = os.path.basename(save_path)
        target_dir = parent_dir if parent_dir else '.'

        # Try each mirror in order
        last_error = None
        for mirror in mirrors:
            try:
                print(f"Trying to download from {mirror['name']}...")
                # download_data will handle directory creation automatically
                download_data(
                    url=mirror['url'],
                    file_path=file_name,
                    dir=target_dir
                )
                print(f"SCSA database saved to {save_path}")
                return save_path
            except Exception as e:
                last_error = e
                print(f"Failed to download from {mirror['name']}: {str(e)}")
                # Try next mirror
                continue

        # If all mirrors failed, raise the last error
        raise Exception(
            f"Failed to download SCSA database from all mirrors. "
            f"Last error: {str(last_error)}"
        )

    def download_scmulan_ckpt(
        self,
        save_path: str = './ckpt/ckpt_scMulan.pt',
        force_download: bool = False,
    ) -> str:
        """Download the pretrained scMulan checkpoint.

        Fetches ``ckpt_scMulan.pt`` from the Tsinghua mirror and stores
        the local path on ``self.scmulan_ckpt_path`` so
        :meth:`annotate(method='scMulan')` can pick it up automatically.

        Parameters
        ----------
        save_path : str, default ``'./ckpt/ckpt_scMulan.pt'``
            Where the checkpoint will be written.
        force_download : bool, default ``False``
            If ``False`` and the file already exists, the cached copy
            is reused. Set ``True`` to overwrite.

        Returns
        -------
        str
            Resolved local path of the checkpoint.
        """
        url = 'https://cloud.tsinghua.edu.cn/f/2250c5df51034b2e9a85/?dl=1'

        if os.path.exists(save_path) and not force_download:
            print(f"scMulan checkpoint already present at {save_path}; "
                   "pass force_download=True to overwrite.")
            self.scmulan_ckpt_path = save_path
            return save_path

        parent_dir = os.path.dirname(save_path)
        file_name = os.path.basename(save_path)
        target_dir = parent_dir if parent_dir else '.'
        download_data(url=url, file_path=file_name, dir=target_dir)
        print(f"scMulan checkpoint saved to {save_path}")
        self.scmulan_ckpt_path = save_path
        return save_path

    def download_reference_pkl(
        self, 
        reference_name: str, 
        save_path: str,
        force_download: bool = False) -> str:
        """
        Download a CellTypist model by model name and store it locally.

        Parameters
        ----------
        reference_name : str
            Model name (or filename) from CellTypist model registry.
        save_path : str
            Local output path of downloaded ``.pkl`` file.
        force_download : bool
            If ``True``, remove existing file and re-download.

        Returns
        -------
        str
            Absolute local path of downloaded model file.
        """

        if not reference_name or not str(reference_name).strip():
            raise ValueError("Please provide a valid `reference_name` that matches the CellTypist model list.")

        if self.celltypist_models_df is None:
            self.celltypist_models_df = _celltypist_models_description()
            print("CellTypist model table saved to self.celltypist_models_df")

        ref = str(reference_name).strip()
        df = self.celltypist_models_df

        matched = df[df['model'] == ref]
        if matched.empty and 'filename' in df.columns:
            matched = df[df['filename'] == ref]
        if matched.empty and 'model' in df.columns:
            matched = df[df['model'].str.lower() == ref.lower()]

        if matched.empty:
            available = df['model'].tolist() if 'model' in df.columns else []
            preview = available[:10]
            suffix = '...' if len(available) > 10 else ''
            raise ValueError(
                f"Model '{reference_name}' not found in CellTypist registry. "
                f"Available models include: {preview}{suffix}"
            )

        row = matched.iloc[0]
        url = row.get('url')
        if not url:
            raise RuntimeError(f"Selected model '{reference_name}' does not include a download URL.")

        abs_path = os.path.abspath(save_path)
        target_dir = os.path.dirname(abs_path)
        if not target_dir:
            target_dir = '.'
        
        if force_download:
            os.remove(abs_path)

        downloaded_path = download_data(url, file_path=os.path.basename(abs_path), dir=target_dir)
        downloaded_path = os.path.abspath(downloaded_path)
        print(url)
        print(f"✓ Model saved to {downloaded_path}")
        self.pkl_ref = downloaded_path
        return downloaded_path

    def query_reference(
        self,
        source='cellxgene',
        data_desc: str = None,
        llm_model='gpt-4o-mini',
        llm_api_key='sk*',
        llm_provider='openai',
        llm_base_url='https://api.openai.com/v1',
        llm_extra_params: Optional[Dict[str, Any]] = None,
    ):
        """Use an LLM to identify relevant reference resources based on a description.

        Parameters
        ----------
        source : str
            Data source identifier. Supported values: ``'cellxgene'`` and ``'celltypist'``.
        data_desc : str
            Free-text description of the dataset you are trying to match.
        llm_model : str
            Model name passed to the selected LLM provider.
        llm_api_key : str
            API key for the LLM provider. If left as a placeholder (e.g. ``'sk*'``),
            the method will attempt to read provider-specific environment variables.
        llm_provider : str
            LLM provider identifier. Implemented providers: ``'openai'``,
            ``'custom_openai'`` (OpenAI-compatible), ``'doubao'``, ``'anthropic'`` and ``'ollama'``.
        llm_base_url : str
            Base URL for OpenAI-compatible endpoints. Ignored for providers that do
            not use it.
        llm_extra_params : dict or None
            Additional parameters forwarded directly to the LLM API call.

        Returns
        -------
        pandas.DataFrame
            Selected entries with source-specific columns (e.g. ``collection_id``/``collection_url``
            for CellxGene, ``model`` for CellTypist) plus ``llm_reason``.
        """

        source_lower = (source or 'cellxgene').strip().lower()

        if data_desc is None or not str(data_desc).strip():
            raise ValueError("Please provide a non-empty description via `data_desc`.")

        query_text = str(data_desc).strip()
        if source_lower == 'cellxgene':
            if self.cellxgene_desc_df is None:
                self.cellxgene_desc_df = _cellxgene_scrape_with_api()
                if self.cellxgene_desc_df is None or self.cellxgene_desc_df.empty:
                    raise RuntimeError("Failed to retrieve CellxGene collections from the API.")
                print("CellxGene description dataframe saved to self.cellxgene_desc_df")

            cellxgene_table = self.cellxgene_desc_df[['collection_id', 'collection_url', 'name', 'description']].copy()
            selector = LLMTableSelector(
                table=cellxgene_table,
                id_column='collection_id',
                description_column='description',
                name_column='name',
                url_column='collection_url',
                table_label='CellxGene collection',
                json_root_key='collections',
                json_id_key='collection_id',
                json_url_key='collection_url',
            )
            system_prompt = _CELLXGENE_SYSTEM_PROMPT
            fallback_reason = 'Fallback to first collection (LLM unavailable or returned no results).'

        elif source_lower == 'celltypist':
            if self.celltypist_models_df is None:
                self.celltypist_models_df = _celltypist_models_description()
                print("CellTypist model table saved to self.celltypist_models_df")

            preferred_display_cols = [
                'model', 'description', 'url', 'version', 'No_celltypes', 'source', 'date', 'default'
            ]
            display_cols = [col for col in preferred_display_cols if col in self.celltypist_models_df.columns]
            if not display_cols:
                display_cols = ['model', 'description']
            celltypist_table = self.celltypist_models_df[display_cols].copy()

            extra_cols = [col for col in display_cols if col not in {'model', 'description', 'url'}]
            selector = LLMTableSelector(
                table=celltypist_table,
                id_column='model',
                description_column='description',
                name_column='model',
                url_column=None,
                table_label='CellTypist model',
                json_root_key='models',
                json_id_key='model',
                json_url_key=None,
                max_prompt_rows=None,
                extra_columns=extra_cols,
            )
            system_prompt = _CELLTYPIST_SYSTEM_PROMPT
            fallback_reason = 'Fallback to first model (LLM unavailable or returned no results).'

        else:
            raise NotImplementedError(f"Unsupported source '{source}'. Supported values: 'cellxgene', 'celltypist'.")

        result_df = selector.select(
            query_text=query_text,
            system_prompt=system_prompt,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_provider=llm_provider,
            llm_base_url=llm_base_url,
            llm_extra_params=llm_extra_params,
            client_factory=self._ensure_llm_client,
            fallback_reason=fallback_reason,
        )

        self.last_reference_query = query_text
        self.last_reference_candidates = selector.last_candidates
        self.last_reference_matches = result_df
        self.last_reference_llm_raw = selector.last_raw_response
        self.last_reference_prompt = selector.last_prompt

        label_plural = f"{selector.table_label}s"
        if not result_df.empty:
            print(f"✓ LLM-selected {label_plural}:")
            id_col = selector.id_column
            name_col = selector.name_column or selector.id_column
            url_col = selector.url_column
            for _, row in result_df.iterrows():
                identifier = row.get(id_col, 'N/A')
                display_name = row.get(name_col, identifier)
                url_value = row.get(url_col) if url_col else None
                if url_value:
                    print(f"  - {identifier}: {display_name} ({url_value})")
                else:
                    print(f"  - {identifier}: {display_name}")
        else:
            print(f"⚠️ No relevant {label_plural} found even after fallback.")

        return result_df

    

    def _ensure_llm_client(
        self,
        provider: str,
        api_key: Optional[str],
        model: str,
        base_url: Optional[str],
        extra_params: Optional[Dict[str, Any]],
    ):
        """
        Build or reuse an LLM client based on current runtime configuration.

        Parameters
        ----------
        provider : str
            LLM provider identifier.
        api_key : str or None
            Provider API key.
        model : str
            Model name.
        base_url : str or None
            Optional OpenAI-compatible endpoint base URL.
        extra_params : dict or None
            Extra generation/runtime parameters.

        Returns
        -------
        tuple
            ``(client, runtime_config)`` for downstream LLM calls.
        """
        provider = (provider or 'openai').lower()
        resolved_key = _resolve_api_key(provider, api_key)

        if provider != 'ollama' and not resolved_key:
            raise ValueError(
                f"Missing API key for LLM provider '{provider}'. Provide via `llm_api_key` "
                "or set the corresponding environment variable."
            )

        extra_signature = json.dumps(extra_params or {}, sort_keys=True, default=str)
        key_signature = hash(resolved_key) if resolved_key is not None else None
        config_signature = json.dumps(
            {
                'provider': provider,
                'model': model,
                'base_url': base_url or '',
                'extra': extra_signature,
                'api_key': key_signature,
            },
            sort_keys=True,
        )

        if self._llm_config_signature != config_signature:
            client, runtime_config = _setup_llm_client(provider, resolved_key, model, base_url, extra_params or {})
            self._llm_client = client
            self._llm_runtime_config = runtime_config
            self._llm_config_signature = config_signature

        return self._llm_client, self._llm_runtime_config



def _resolve_api_key(provider: str, provided_key: Optional[str]) -> Optional[str]:
    """
    Resolve provider API key from explicit input or environment variables.

    Parameters
    ----------
    provider : str
        LLM provider identifier.
    provided_key : str or None
        User-provided API key candidate.

    Returns
    -------
    str or None
        Resolved API key, or ``None`` if unavailable.
    """
    if provided_key:
        trimmed = provided_key.strip()
        if trimmed and not trimmed.endswith('*') and trimmed not in {'sk*', 'sk-***'}:
            return trimmed

    env_map = {
        'openai': 'OPENAI_API_KEY',
        'custom_openai': 'OPENAI_API_KEY',
        'doubao': 'DOUBAO_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
    }
    env_key = env_map.get(provider)
    if env_key:
        env_value = os.getenv(env_key)
        if env_value:
            return env_value.strip()
    return None


def _setup_llm_client(
    provider: str,
    api_key: Optional[str],
    model: str,
    base_url: Optional[str],
    extra_params: Dict[str, Any],
):
    """
    Initialize provider-specific LLM client and runtime config.

    Parameters
    ----------
    provider : str
        LLM provider identifier.
    api_key : str or None
        Provider API key.
    model : str
        Model name.
    base_url : str or None
        Optional custom endpoint URL for OpenAI-compatible providers.
    extra_params : dict
        Additional generation/runtime parameters.

    Returns
    -------
    tuple
        ``(client, runtime_config)`` used by internal LLM call wrappers.
    """
    provider = provider.lower()
    extra_params = extra_params or {}
    runtime_config = {
        'provider': provider,
        'model': model,
        'base_url': base_url,
        'extra_params': extra_params,
    }

    if provider in {'openai', 'custom_openai', 'doubao'}:
        if not api_key:
            raise ValueError(f"Missing API key for provider '{provider}'.")
        import openai

        client_kwargs: Dict[str, Any] = {}
        if base_url:
            client_kwargs['base_url'] = base_url
        elif provider == 'doubao':
            client_kwargs['base_url'] = 'https://ark.cn-beijing.volces.com/api/v3'
        elif provider == 'custom_openai':
            raise ValueError("`llm_base_url` is required when using the 'custom_openai' provider.")

        client = openai.OpenAI(api_key=api_key, **client_kwargs)
        runtime_config['base_url'] = client_kwargs.get('base_url', base_url)
        return client, runtime_config

    if provider == 'anthropic':
        if not api_key:
            raise ValueError("Missing API key for provider 'anthropic'.")
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        return client, runtime_config

    if provider == 'ollama':
        runtime_config['base_url'] = base_url or 'http://localhost:11434'
        return None, runtime_config

    raise ValueError(
        f"Unsupported llm_provider '{provider}'. Supported providers: openai, custom_openai, doubao, anthropic, ollama."
    )


def _call_llm_for_collections(
    client,
    config: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
) -> str:
    """
    Call configured LLM backend and return raw text response.

    Parameters
    ----------
    client : Any
        Provider client object or ``None`` for backends using raw HTTP calls.
    config : dict
        Runtime configuration dictionary from ``_setup_llm_client``.
    system_prompt : str
        System-level instruction text.
    user_prompt : str
        User prompt containing candidate table and JSON output schema.

    Returns
    -------
    str
        Raw model response text.
    """
    provider = config.get('provider')
    model = config.get('model')
    extra_params = dict(config.get('extra_params') or {})

    if provider in {'openai', 'custom_openai', 'doubao'}:
        request_payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
        }
        request_payload.update(extra_params)
        response = client.chat.completions.create(**request_payload)
        return response.choices[0].message.content.strip()

    if provider == 'anthropic':
        anthropic_payload = {
            'model': model,
            'system': system_prompt,
            'messages': [{'role': 'user', 'content': user_prompt}],
        }
        anthropic_payload.update(extra_params)
        response = client.messages.create(**anthropic_payload)
        text_chunks: List[str] = []
        for block in getattr(response, 'content', []):
            if hasattr(block, 'text'):
                text_chunks.append(block.text)
        return ''.join(text_chunks).strip()

    if provider == 'ollama':
        import requests as _requests

        payload: Dict[str, Any] = {
            'model': model,
            'prompt': f"{system_prompt}\n\nUser: {user_prompt}",
            'stream': False,
        }
        options = extra_params.pop('options', None)
        if options is not None:
            payload['options'] = options
        if extra_params:
            payload.update(extra_params)

        response = _requests.post(
            f"{config.get('base_url', 'http://localhost:11434').rstrip('/')}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return (data.get('response') or '').strip()

    raise ValueError(f"Unsupported llm_provider '{provider}' for runtime call.")


def _strip_code_fence(text: str) -> str:
    """
    Remove surrounding Markdown code-fence wrappers from text.

    Parameters
    ----------
    text : str
        Raw model response text.

    Returns
    -------
    str
        Unfenced text content.
    """
    match = re.match(r"^```[a-zA-Z0-9_-]*\n?(.*?)\n?```$", text.strip(), flags=re.DOTALL)
    if match:
        return match.group(1)
    return text


def _parse_llm_collection_response(raw_text: Optional[str], root_key: str = 'collections') -> List[Dict[str, Any]]:
    """
    Parse LLM JSON response into a list of selection dictionaries.

    Parameters
    ----------
    raw_text : str or None
        Raw response text from LLM.
    root_key : str
        Expected top-level JSON key containing selection list.

    Returns
    -------
    List[Dict[str, Any]]
        Parsed selection items.
    """
    if not raw_text:
        return []

    cleaned = _strip_code_fence(raw_text.strip())
    if not cleaned:
        return []

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            return []
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return []

    if isinstance(data, dict):
        payload = data.get(root_key) if root_key in data else data.get('collections')
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    return []


def _build_selection_dataframe(
    candidates_df: pd.DataFrame,
    selections: List[Dict[str, Any]],
    id_column: str,
    name_column: Optional[str],
    description_column: Optional[str],
    url_column: Optional[str],
    json_id_key: str,
    json_url_key: Optional[str],
    json_reason_keys: Optional[List[str]] = None,
    extra_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a DataFrame of selected rows from parsed LLM selections.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        Candidate table used in prompting.
    selections : list
        Parsed selection dictionaries from LLM response.
    id_column : str
        Candidate ID column name.
    name_column : str or None
        Optional display-name column.
    description_column : str or None
        Optional description column.
    url_column : str or None
        Optional URL column.
    json_id_key : str
        Key used by selections to identify candidate IDs.
    json_url_key : str or None
        Optional key used by selections to provide URL values.
    json_reason_keys : list or None
        Candidate keys used to extract rationale text.
    extra_columns : list or None
        Additional columns retained in output.

    Returns
    -------
    pd.DataFrame
        Selection table with standardized output columns and ``llm_reason``.
    """
    reason_keys = json_reason_keys or ['reason', 'explanation', 'note']
    extra_columns = [col for col in (extra_columns or []) if col]
    core_columns = [col for col in [id_column, url_column, name_column, description_column] if col]
    ordered_columns: List[str] = []
    for col in core_columns + extra_columns:
        if col and col not in ordered_columns:
            ordered_columns.append(col)
    columns = ordered_columns + ['llm_reason']
    if candidates_df is None or candidates_df.empty:
        return pd.DataFrame(columns=columns)

    results: List[Dict[str, Any]] = []
    seen_ids = set()

    for selection in selections:
        if not isinstance(selection, dict):
            continue
        collection_id = selection.get(json_id_key) or selection.get('id') or selection.get('collection_id')
        if not collection_id or collection_id in seen_ids:
            continue

        matched = candidates_df[candidates_df[id_column] == collection_id]
        if matched.empty:
            continue
        row = matched.iloc[0]

        entry: Dict[str, Any] = {id_column: row.get(id_column)}
        if url_column:
            entry[url_column] = row.get(url_column) or (selection.get(json_url_key) if json_url_key else None)
        if name_column:
            entry[name_column] = row.get(name_column)
        if description_column:
            entry[description_column] = row.get(description_column)
        for extra_col in extra_columns:
            entry[extra_col] = row.get(extra_col)

        entry['llm_reason'] = next((selection.get(key) for key in reason_keys if selection.get(key)), None)

        results.append(entry)
        seen_ids.add(collection_id)

    return pd.DataFrame(results, columns=columns)


def _truncate_text(text: Optional[str], limit: int) -> str:
    """
    Truncate text to a maximum length with ellipsis.

    Parameters
    ----------
    text : str or None
        Input text.
    limit : int
        Maximum character length.

    Returns
    -------
    str
        Truncated text.
    """
    if not text:
        return ''
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + '...'


def _build_table_prompt(
    query_text: str,
    candidates_df: pd.DataFrame,
    id_column: str,
    description_column: str,
    name_column: Optional[str] = None,
    url_column: Optional[str] = None,
    json_root_key: str = 'collections',
    json_id_key: str = 'collection_id',
    json_url_key: Optional[str] = 'collection_url',
    table_label: str = 'entry',
    extra_columns: Optional[List[str]] = None,
) -> str:
    """
    Build structured table-selection prompt for LLM ranking.

    Parameters
    ----------
    query_text : str
        Free-text description of user dataset/task.
    candidates_df : pd.DataFrame
        Candidate table rows to include in prompt.
    id_column : str
        Candidate ID column.
    description_column : str
        Candidate description column.
    name_column : str or None
        Optional candidate display-name column.
    url_column : str or None
        Optional candidate URL column.
    json_root_key : str
        Expected output JSON root key.
    json_id_key : str
        Expected output JSON ID key.
    json_url_key : str or None
        Optional expected output JSON URL key.
    table_label : str
        Human-readable row type label.
    extra_columns : list or None
        Additional candidate columns rendered in prompt.

    Returns
    -------
    str
        Prompt text containing candidate records and strict JSON schema.
    """
    extra_columns = [col for col in (extra_columns or []) if col not in {id_column, name_column, description_column, url_column}]
    lines = [
        f"We need to select the most relevant {table_label} records for the analysis.",
        "Match them to the following dataset description:",
        f"\"\"\"{query_text}\"\"\"",
        "",
        "Candidates (ordered as provided):",
    ]

    for idx, (_, row) in enumerate(candidates_df.iterrows(), start=1):
        identifier = row.get(id_column, 'Unknown')
        description = _truncate_text(row.get(description_column), _PROMPT_DESCRIPTION_LIMIT)
        if not description:
            description = 'No description provided.'

        lines.append(f"{idx}. ID: {identifier}")
        if name_column and pd.notna(row.get(name_column)):
            lines.append(f"   Name: {row.get(name_column)}")
        if url_column and pd.notna(row.get(url_column)):
            lines.append(f"   URL: {row.get(url_column)}")
        for extra_col in extra_columns:
            value = row.get(extra_col)
            if pd.notna(value):
                lines.append(f"   {extra_col}: {value}")
        lines.append(f"   Description: {description}")
        lines.append("")

    lines.extend(
        [
            "Respond with strict JSON following this schema (no commentary):",
            f'{{"{json_root_key}": ['
            f"\n  {{\"{json_id_key}\": \"...\""
            + (f", \"{json_url_key}\": \"...\"" if json_url_key else "")
            + ', "reason": "why this fits"}'
            '\n] }',
            f'Only use {json_id_key} values from the candidates. '
            f'If none are suitable, respond with {{{json_root_key}: []}}.',
        ]
    )

    return '\n'.join(lines)

# 方案3: 直接访问 API (如果可用)
def _cellxgene_scrape_with_api():
    """
    Fetch CellxGene collection metadata directly from public API.

    Returns
    -------
    pd.DataFrame or None
        Collection metadata table when request succeeds, otherwise ``None``.
    """
    api_url = "https://api.cellxgene.cziscience.com/curation/v1/collections"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    print(f"正在访问 API: {api_url}")
    response = requests.get(api_url, headers=headers, timeout=30)

    if response.status_code == 200:
        print(f"✓ API 访问成功 (状态码: {response.status_code})")
        data = response.json()

        # 解析 JSON 数据为 DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'collections' in data:
            df = pd.DataFrame(data['collections'])
        else:
            df = pd.json_normalize(data)

        return df
    else:
        print(f"✗ API 访问失败 (状态码: {response.status_code})")
        return None


def _celltypist_models_description() -> pd.DataFrame:
    """
    Download and normalize CellTypist model registry metadata.

    Returns
    -------
    pd.DataFrame
        Standardized model table with ``model``, ``description`` and URL fields.
    """
    response = requests.get(_CELLTYPIST_MODELS_URL, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch CellTypist model metadata (status {response.status_code})."
        )

    payload = response.json()
    models_list = payload.get('models')
    if not isinstance(models_list, list) or not models_list:
        raise RuntimeError("CellTypist model list is empty or malformed.")

    df = pd.DataFrame(models_list)
    if df.empty:
        raise RuntimeError("CellTypist model description dataframe is empty.")

    if 'filename' not in df.columns or 'details' not in df.columns or 'url' not in df.columns:
        raise RuntimeError("CellTypist model metadata missing required keys (expected 'filename', 'details', 'url').")

    df = df.copy()
    df['model'] = df['filename']
    df['description'] = df['details']

    preferred_order = [
        'model', 'description', 'url', 'version', 'date', 'No_celltypes', 'source', 'default',
    ]
    ordered_cols = [col for col in preferred_order if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in ordered_cols]
    df = df[ordered_cols + remaining_cols]

    return df.reset_index(drop=True)

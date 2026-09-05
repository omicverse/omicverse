"""Small AnnData-first runners for distinct COMMOT and FlowSig tasks."""
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd
from scipy import sparse

from .._registry import register_function


def _versions():
    result = {}
    for package in ('omicverse', 'numpy', 'anndata', 'causaldag'):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = 'not-installed'
    return result


def _single_library(adata, library_key):
    if library_key is not None:
        if library_key not in adata.obs or adata.obs[library_key].isna().any():
            raise ValueError('library_key must exist and contain no missing labels.')
        if adata.obs[library_key].nunique() != 1:
            raise ValueError('Run each library separately; subset observations and spatial metadata first.')
    if len(adata.uns.get('spatial', {})) > 1:
        raise ValueError('Multiple image libraries: subset observations and spatial metadata first.')
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError('Observation and gene names must be unique.')


def _expression(adata, layer):
    if layer is not None and layer not in adata.layers:
        raise KeyError(f'Expression layer {layer!r} does not exist.')
    matrix = adata.X if layer is None else adata.layers[layer]
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Expression must be finite and non-negative; scaled residuals are not supported.')
    return matrix


@register_function(aliases=['run_commot', '空间通讯推断'], category='space',
                   description='Run COMMOT on one library with explicit expression and coordinate inputs.')
def run_commot(adata, *, database_name, df_ligrec, dis_thr, distance_unit,
               layer=None, spatial_key='spatial', library_key=None, pathway_sum=True,
               heteromeric=True, inplace=True, overwrite=False, max_distance_matrix_mb=1024,
               **kwargs):
    """Run real COMMOT and return AnnData, retaining native commot-* outputs.

    distance_unit describes the supplied coordinates; no implicit pixel/micron
    conversion is performed. Supply non-negative expression using layer (None
    means X); no normalization is performed. df_ligrec requires named ligand,
    receptor and pathway columns. Run independent libraries separately.
    The dense distance matrix alone must fit max_distance_matrix_mb; this is a
    lower bound on backend memory, not a total memory guarantee.
    inplace=False returns an independent copy. Failures do not write partial
    outputs to the input. Existing output namespace requires overwrite=True.
    """
    _single_library(adata, library_key)
    expression = _expression(adata, layer)
    if not isinstance(database_name, str) or not database_name:
        raise ValueError('database_name must be a non-empty string.')
    if distance_unit not in ('micron', 'pixel', 'arbitrary'):
        raise ValueError("distance_unit must be 'micron', 'pixel', or 'arbitrary'.")
    if not np.isfinite(dis_thr) or dis_thr <= 0:
        raise ValueError('dis_thr must be finite and positive in coordinate units.')
    if spatial_key not in adata.obsm:
        raise KeyError(f'Missing spatial coordinates: {spatial_key}.')
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2 or not np.isfinite(coords).all():
        raise ValueError('Spatial coordinates must be finite with at least two columns.')
    required_mb = adata.n_obs ** 2 * 8 / 1024 ** 2
    if max_distance_matrix_mb <= 0 or required_mb > max_distance_matrix_mb:
        raise MemoryError(f'Dense distance matrix needs at least {required_mb:.1f} MiB; '
                          'subset explicitly or raise max_distance_matrix_mb.')
    columns = ['ligand', 'receptor', 'pathway']
    if not isinstance(df_ligrec, pd.DataFrame) or not set(columns).issubset(df_ligrec.columns):
        raise ValueError('df_ligrec requires named ligand, receptor, pathway columns.')
    if df_ligrec[columns].isna().any().any():
        raise ValueError('Database identities and pathways cannot be missing.')
    database = df_ligrec[columns].astype(str).drop_duplicates().copy()
    genes = set(adata.var_names)
    delimiter = kwargs.get('heteromeric_delimiter', '_')
    if not delimiter:
        raise ValueError('heteromeric_delimiter cannot be empty.')
    def present(name):
        return all(gene in genes for gene in name.split(delimiter)) if heteromeric else name in genes
    database = database.loc[database.ligand.map(present) & database.receptor.map(present)].copy()
    if database.empty:
        raise ValueError('No complete ligand-receptor pair is present in expression genes.')
    if any(key in kwargs for key in ('copy', 'adata')):
        raise ValueError('Use inplace to control copies; do not pass backend copy/adata.')
    prefix = f'commot-{database_name}-'
    for key in adata.uns:
        if key.startswith('commot-') and key.endswith('-info'):
            other = key[7:-5]
            if other != database_name and (other.startswith(database_name + '-') or database_name.startswith(other + '-')):
                raise ValueError('COMMOT database names have overlapping output prefixes; choose a distinct name.')
    if not overwrite and any(key.startswith(prefix) for slot in (adata.uns, adata.obsm, adata.obsp) for key in slot):
        raise ValueError(f'Output namespace {prefix} exists; use overwrite=True explicitly.')
    from ..external.commot.tools import spatial_communication
    work = adata.copy()
    work.X = expression.copy()
    work.obsm['spatial'] = coords.copy()
    # Coordinates are the single distance source; never reuse a stale distance matrix.
    work.obsp.pop('spatial_distance', None)
    for slot in (work.uns, work.obsm, work.obsp):
        for key in list(slot):
            if key.startswith(prefix):
                del slot[key]
    spatial_communication(work, database_name=database_name, df_ligrec=database,
                          dis_thr=dis_thr, pathway_sum=pathway_sum, heteromeric=heteromeric, **kwargs)
    work.uns[prefix + 'info']['omicverse_run'] = {
        'method': 'commot', 'layer': layer or 'X', 'spatial_key': spatial_key,
        'distance_unit': distance_unit, 'distance_threshold': float(dis_thr),
        'database_sha256': hashlib.sha256(database.to_csv(index=False).encode()).hexdigest(),
        'parameters_json': json.dumps(kwargs, default=str, sort_keys=True),
        'pathway_sum': bool(pathway_sum), 'heteromeric': bool(heteromeric),
        'versions': _versions(), 'backend_source': 'omicverse.external.commot',
    }
    output = adata if inplace else adata.copy()
    for source, target in ((work.uns, output.uns), (work.obsm, output.obsm), (work.obsp, output.obsp)):
        if overwrite:
            for key in list(target):
                if key.startswith(prefix):
                    del target[key]
        for key in source:
            if key.startswith(prefix):
                target[key] = source[key]
    return output


@register_function(aliases=['run_flowsig', '空间信号网络推断'], category='space',
                   description='Run FlowSig from existing COMMOT marginals and supplied GEMs on one library.')
def run_flowsig(adata, *, commot_output_key, gem_expr_key='X_gem', block_key,
                layer=None, library_key=None, key_added='flowsig_network',
                flow_expr_key='X_flow', n_bootstraps=100, n_jobs=1,
                alpha_ci=1e-3, edge_threshold=0.8, inplace=True, overwrite=False):
    """Construct FlowSig variables and learn a block-resampled spatial network.

    Supply GEMs explicitly: this runner does not train or substitute GEM methods.
    All constructed variables are retained (no implicit Moran selection).
    GEM scaling uses the backend's scale_gem_expr=False mode; X/layer provides
    non-negative ligand expression. Raw, biologically oriented and filtered
    adjacency matrices remain in uns[key_added]['network']. Edge weights are
    bootstrap frequencies, not p-values or experimentally established causality.
    The backend uses bootstrap-index seeds; no unsupported random seed is implied.
    inplace=False returns a copy; backend failure leaves the input unchanged.
    """
    _single_library(adata, library_key)
    expression = _expression(adata, layer)
    if block_key not in adata.obs or adata.obs[block_key].isna().any():
        raise ValueError('block_key must contain non-missing spatial block labels.')
    if adata.n_obs < 4:
        raise ValueError('FlowSig needs at least four observations.')
    if gem_expr_key not in adata.obsm:
        raise KeyError(f'Missing GEM expression {gem_expr_key}.')
    gems = np.asarray(adata.obsm[gem_expr_key])
    if gems.ndim != 2 or gems.shape[1] == 0 or not np.isfinite(gems).all() or (gems < 0).any() or (gems.sum(axis=0) <= 0).any():
        raise ValueError('GEMs must be finite, non-negative, with no zero-sum columns.')
    for suffix in ('-sum-receiver', '-sum-sender'):
        key = commot_output_key + suffix
        if key not in adata.obsm:
            raise KeyError(f'Missing COMMOT marginal {key}.')
        marginal = adata.obsm[key]
        if not isinstance(marginal, pd.DataFrame) or not marginal.index.equals(adata.obs_names):
            raise ValueError('COMMOT marginals must be barcode-indexed DataFrames aligned to observations.')
    if not 0 <= edge_threshold <= 1:
        raise ValueError('edge_threshold must lie in [0, 1].')
    if not overwrite and (key_added in adata.uns or flow_expr_key in adata.obsm):
        raise ValueError('FlowSig output exists; use overwrite=True explicitly.')
    try:
        import causaldag  # noqa: F401
    except ImportError as exc:
        raise ImportError('run_flowsig requires causaldag; install omicverse[flowsig].') from exc
    from ..external.flowsig.preprocessing import construct_flows_from_commot
    from ..external.flowsig.tools import (
        apply_biological_flow,
        filter_low_confidence_edges,
        learn_intercellular_flows,
    )
    work = adata.copy()
    work.X = expression.copy()
    construct_flows_from_commot(work, commot_output_key, gem_expr_key=gem_expr_key,
                               scale_gem_expr=False, flowsig_network_key=key_added, flowsig_expr_key=flow_expr_key)
    learn_intercellular_flows(work, flowsig_key=key_added, flow_expr_key=flow_expr_key,
                             use_spatial=True, block_key=block_key, n_jobs=n_jobs,
                             n_bootstraps=n_bootstraps, alpha_ci=alpha_ci)
    apply_biological_flow(work, flowsig_network_key=key_added)
    filter_low_confidence_edges(work, edge_threshold=edge_threshold, flowsig_network_key=key_added,
                                adjacency_key='adjacency_validated')
    work.uns[key_added]['omicverse_run'] = {
        'method': 'flowsig', 'commot_output_key': commot_output_key, 'gem_expr_key': gem_expr_key,
        'block_key': block_key, 'layer': layer or 'X', 'n_bootstraps': n_bootstraps,
        'alpha_ci': alpha_ci, 'edge_threshold': edge_threshold, 'variable_selection': 'all',
        'scale_gem_expr': False, 'seed_policy': 'backend_bootstrap_index',
        'versions': _versions(), 'backend_source': 'omicverse.external.flowsig',
    }
    output = adata if inplace else adata.copy()
    output.uns[key_added] = work.uns[key_added]
    output.obsm[flow_expr_key] = work.obsm[flow_expr_key]
    return output

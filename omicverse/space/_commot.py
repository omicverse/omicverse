"""Select and summarize COMMOT outputs with explicit statistical meaning."""
import warnings
from itertools import product

import anndata as ad
import numpy as np
import pandas as pd

from .._registry import register_function


def _get_summarize_cluster_gpu():
    from ..external.commot.tools._spatial_communication import summarize_cluster_gpu
    return summarize_cluster_gpu


def _result_metadata(adata, database_name=None):
    available = [key[7:-5] for key in adata.uns
                 if key.startswith('commot-') and key.endswith('-info')
                 and isinstance(adata.uns[key], dict) and 'df_ligrec' in adata.uns[key]]
    if database_name is None:
        if len(available) != 1:
            raise ValueError(f'Choose database_name explicitly; available databases: {available}.')
        database_name = available[0]
    if database_name not in available:
        raise ValueError(f'Missing commot-{database_name}-info with df_ligrec; preserve backend metadata.')
    db = adata.uns[f'commot-{database_name}-info']['df_ligrec']
    if not {'ligand', 'receptor', 'pathway'}.issubset(db.columns):
        raise ValueError('df_ligrec requires ligand, receptor, pathway columns.')
    prefix = f'commot-{database_name}-'
    records = {}

    def add(key, record):
        if key in records and records[key] != record:
            raise ValueError(f'Ambiguous COMMOT result key {key!r}; database identities collide.')
        records[key] = record

    def record(name, level, pathway, ligand='', receptor=''):
        return dict(interacting_pair=name, level=level, classification=pathway,
                    gene_a=ligand, gene_b=receptor, partner_a=ligand, partner_b=receptor,
                    secreted='Unknown', is_integrin='Unknown',
                    annotation_strategy='commot_database', directionality='ligand-receptor')

    for (ligand, receptor), rows in db.groupby(['ligand', 'receptor'], sort=False, dropna=False):
        if pd.isna(ligand) or pd.isna(receptor):
            raise ValueError('Missing ligand/receptor identity.')
        pathways = sorted(set(rows['pathway'].dropna().astype(str)))
        entry = record(f'{ligand}-{receptor}', 'lr', ';'.join(pathways) or 'Unknown',
                       str(ligand), str(receptor))
        for field in ('secreted', 'is_integrin'):
            values = rows[field].dropna().unique() if field in rows else []
            entry[field] = str(values[0]) if len(values) == 1 else 'Unknown'
        add(prefix + f'{ligand}-{receptor}', entry)
    for pathway in db['pathway'].dropna().astype(str).unique():
        add(prefix + pathway, record(pathway, 'pathway', pathway))
    add(prefix + 'total-total', record('total-total', 'total', 'Total'))
    for key, entry in records.items():
        entry['id_cp_interaction'] = key
    return database_name, records


def _summaries(adata, clustering_column, n_permutations, database_name, level, statistic, use_gpu, seed):
    if level not in ('lr', 'pathway', 'total', 'all'):
        raise ValueError("level must be 'lr', 'pathway', 'total', or 'all'.")
    if statistic not in ('sum', 'mean'):
        raise ValueError("statistic must be 'sum' or 'mean'.")
    if not isinstance(n_permutations, int) or n_permutations < 1:
        raise ValueError('n_permutations must be a positive integer.')
    if clustering_column not in adata.obs or adata.obs[clustering_column].isna().any():
        raise ValueError('clustering_column must exist and contain no missing labels.')
    raw_labels = adata.obs[clustering_column]
    labels = raw_labels.astype(str).to_numpy()
    if len(pd.unique(raw_labels)) != len(np.unique(labels)):
        raise ValueError('Cell-type labels collide after conversion to strings.')
    celltypes = sorted(set(labels))
    if any('|' in name for name in celltypes):
        raise ValueError("Cell-type labels cannot contain the plotting separator '|'.")
    database_name, metadata = _result_metadata(adata, database_name)
    selected = {key: value for key, value in metadata.items()
                if key in adata.obsp and (level == 'all' or value['level'] == level)}
    if not selected:
        raise ValueError(f'No {level} result matrices found for {database_name}.')
    if level == 'all':
        warnings.warn('LR, pathway and total overlap; select one level before aggregating columns.',
                      UserWarning, stacklevel=3)
    summarize = _get_summarize_cluster_gpu()
    results = {}
    for key in selected:
        matrix = adata.obsp[key]
        values = matrix.data if hasattr(matrix, 'tocsr') else np.asarray(matrix)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'Invalid non-finite or negative scores in {key}.')
        # Legacy backend uses the global NumPy RNG; always restore caller state.
        state = np.random.get_state()
        try:
            if seed is not None:
                np.random.seed(seed)
            score, pvalue = summarize(matrix, labels, celltypes, n_permutations,
                                      use_gpu=use_gpu, scale_factor='sum' if statistic == 'sum' else None)
        finally:
            np.random.set_state(state)
        results[key] = {'communication': score, 'pvalue': pvalue}
    return database_name, selected, celltypes, labels, results


@register_function(aliases=['通信AnnData', 'create_communication_anndata'], category='space',
                   description='Summarize a selected COMMOT database into plotting AnnData.',
                   examples=["comm = ov.space.create_communication_anndata(adata, 'cell_type', level='lr')"])
def create_communication_anndata(adata, clustering_column, n_permutations=100, *,
                                database_name=None, level='all', statistic='sum', use_gpu=False, seed=0):
    """Return cell-type-pair by interaction AnnData from existing COMMOT results.

    database_name can be omitted only for one database metadata entry.
    level='all' preserves legacy selection with a warning because the columns
    overlap; prefer explicit level='lr'. layers['means'] retains its plotting
    name but contains the requested statistic, recorded in uns['commot_summary'].
    P-values test labels on a fixed graph, not experimental condition effects.
    Unknown database annotations remain 'Unknown', never invented booleans.
    """
    database_name, metadata, celltypes, labels, results = _summaries(
        adata, clustering_column, n_permutations, database_name, level, statistic, use_gpu, seed)
    pairs = list(product(celltypes, repeat=2))
    names = [f'{a}|{b}' for a, b in pairs]
    obs = pd.DataFrame({'sender': [a for a, _ in pairs], 'receiver': [b for _, b in pairs],
                        'cell_type_pair': names,
                        'n_sender': [int((labels == a).sum()) for a, _ in pairs],
                        'n_receiver': [int((labels == b).sum()) for _, b in pairs]}, index=names)
    var = pd.DataFrame.from_dict(metadata, orient='index')
    scores = np.column_stack([r['communication'].to_numpy().ravel() for r in results.values()])
    pvalues = np.column_stack([r['pvalue'].to_numpy().ravel() for r in results.values()])
    output = ad.AnnData(scores, obs=obs, var=var)
    output.layers['means'] = scores.copy()
    output.layers['pvalues'] = pvalues
    output.uns['commot_summary'] = dict(database_name=database_name, level=level, statistic=statistic,
        score_layer='means', pvalue_method='cell_type_label_permutation_fixed_graph',
        n_permutations=n_permutations, seed=seed if seed is not None else 'unspecified')
    return output


def process_all_commot(adata, clustering_column, n_permutations=100, return_format='anndata', **kwargs):
    """Return selected summaries in AnnData or legacy dictionary format."""
    if return_format == 'anndata':
        return create_communication_anndata(adata, clustering_column, n_permutations, **kwargs)
    if return_format != 'dict':
        raise ValueError("return_format must be 'anndata' or 'dict'.")
    options = dict(database_name=None, level='all', statistic='sum', use_gpu=False, seed=0)
    options.update(kwargs)
    return _summaries(adata, clustering_column, n_permutations, **options)[-1]


def quick_demo(adata, clustering_column, max_pathways=5):
    """Return a small LR summary; this does not run inference."""
    result = create_communication_anndata(adata, clustering_column, n_permutations=10, level='lr')
    return result[:, :max_pathways].copy()


def example_usage():
    """Print a minimal example."""
    print("comm = ov.space.create_communication_anndata(adata, 'cell_type', level='lr')")


@register_function(aliases=['更新通信分类', 'update_classification_from_database'], category='space',
                   description='Update COMMOT annotations using exact database identities.')
def update_classification_from_database(comm_adata, adata_with_db, *, database_name=None):
    """Update known annotations without parsing hyphenated molecular identifiers."""
    saved = comm_adata.uns.get('commot_summary', {}).get('database_name')
    if database_name is not None and saved is not None and database_name != saved:
        raise ValueError('database_name differs from the database used to create this summary.')
    _, metadata = _result_metadata(adata_with_db, database_name or saved)
    for key in comm_adata.var_names:
        if key in metadata:
            for field, value in metadata[key].items():
                comm_adata.var.loc[key, field] = value
    return comm_adata

# !/usr/bin/env python
# coding: utf-8
# @Time    : 2022/9/23 13:25
# @Author  : Z.-H. G.
# @Email_0 : guozhenhao17@mails.ucas.ac.cn
# @Email_1 : guozhenhao@tongji.edu.cn
# @File    : preprocess_untils.py
# @IDE     : PyCharm

import numpy as np
import pandas as pd
import scanpy as sc
import os
import logging
from typing import Sequence, Optional, Callable


def get_file_name(path='./', file_type='.h5ad'):
    """
    Checking the .h5ad file from the path
    :param path:
    :param file_type:
    :return:
    """
    #
    # read path
    all_file_name = os.listdir(path)
    file_name = []
    for name in all_file_name:
        if name.endswith(file_type):
            file_name.append(name)
    return file_name


def get_balanced_dataset(data_name_list, ref_name=None):
    """
    Remove cell types that are not in the reference dataset
    :param data_name_list:
    :param ref_name:
    :return:
    """
    if ref_name is None:
        ref_name = data_name_list[0]
        data_name_list = data_name_list[1:]
    else:
        data_name_list.remove(ref_name)
    ref_data = sc.read_h5ad(ref_name)
    ref_cell_name = ref_data.obs.loc[:, 'cell_ontology_class'].unique()
    query_data = [sc.read_h5ad(name) for name in data_name_list]
    query_data_balanced = [data[data.obs.loc[:, 'cell_ontology_class'].isin(ref_cell_name), :] for data in query_data]
    return data_name_list, query_data, query_data_balanced


def downsample_dataset(data_name_list, ration=0.2):
    # load
    adata_list = [sc.read_h5ad(name) for name in data_name_list]
    [sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True) for adata in adata_list]
    # down sample
    adata_list_down = [
        sc.pp.downsample_counts(adata, total_counts=int(ration * adata.obs.loc[:, 'total_counts'].sum()), copy=True) for
        adata in adata_list]
    [sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True) for adata in adata_list_down]
    return data_name_list, adata_list, adata_list_down


def preprocess(data):
    sc.pp.calculate_qc_metrics(data, percent_top=None, log1p=False, inplace=True)
    sc.pp.filter_cells(data, min_genes=200)
    sc.pp.filter_cells(data, max_genes=20000)
    sc.pp.filter_genes(data, min_cells=3)
    data = data[data.obs.n_genes_by_counts < 2500, :]
    return data


def get_counts_from_dataset(dataset, dataset_name):
    """
    Get statistics of the dataset
    :param dataset:
    :param dataset_name:
    :return:
    """
    dataset = preprocess(dataset)
    pd_statistics = dataset.obs.loc[:, ['cell_ontology_class', 'total_counts', 'n_genes_by_counts']]
    pd_statistics['dataset_name'] = dataset_name
    return pd_statistics


def quick_preprocess(adata, n_top_genes=2000):
    adata = adata.copy()
    adata.layers['counts'] = adata.X
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key='batch')
    # adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    return adata


def get_marker_info_table(
        adata, groups=None, key='rank_genes_groups',
        cut_padj: Optional[float] = 0.05,
        cut_logfc: Optional[float] = 0.25,
        cut_pts: Optional[float] = None,
):
    result = adata.uns[key]
    if groups is None:
        groups = result['names'].dtype.names

    dfs = []
    cols = ['names', 'logfoldchanges', 'pvals', 'pvals_adj', 'scores', ]
    # cols = [c for c in cols if c in result.keys()]
    flag_pts = 'pts' in result.keys()

    for group in groups:
        _df = pd.DataFrame({
            key: result[key][group] for key in cols
        })
        _df['group'] = group

        if flag_pts:
            # expression proportions, avoid row mismatching
            _df['pts'] = _df['names'].map(result['pts'][group])
            _df['pts_rest'] = _df['names'].map(result['pts_rest'][group])
            if cut_pts is not None:
                _df = _df[_df['pts'] >= cut_pts]
        if cut_padj is not None:
            _df = _df[_df['pvals_adj'] <= cut_padj]
        if cut_logfc is not None:
            _df = _df[_df['logfoldchanges'] >= cut_logfc]
        dfs.append(_df.copy())  # [['group'] + cols])
    df = pd.concat(dfs, axis=0, keys=groups)
    if flag_pts:
        cols += ['pts', 'pts_rest']
    return df[['group'] + cols]


def top_markers_from_info(
        df_info, n=5, groups=None, unique=True,
        col_group='group',
        col_name='names') -> list:
    """
    df_info: DEG-info table that can be take from `top_markers_from_adata`
    """
    # if groups is not None:
    #     df_info = df_info[df_info[col_group].isin(groups)]
    subdf = df_info.groupby(col_group).apply(lambda x: x.head(n))
    if groups is not None:
        subdf = subdf.loc[groups]  # filter, or keep the group orders
    names = subdf[col_name]
    return names.unique().tolist() if unique else names.tolist()


def top_markers_from_adata(adata: sc.AnnData,
                           n: Optional[int] = 5,
                           groups: Optional[Sequence] = None,
                           unique=True,
                           cut_padj=0.05,
                           cut_logfc=0.25,
                           cut_pts=0.1,
                           key='rank_genes_groups'):
    df_info = get_marker_info_table(
        adata, groups, key=key,
        cut_padj=cut_padj, cut_logfc=cut_logfc, cut_pts=cut_pts)

    if n is None:
        if unique:
            return df_info['names'].unique().tolist()
        return df_info['names'].tolist()
    else:
        # df = get_marker_name_table(adata, key=key)
        # return top_markers_from_df(df, n=n, groups=groups, unique=unique)
        if groups is None:
            _groupby = adata.uns[key]['params']['groupby']
            try:
                # to keep the group order
                groups = adata.obs[_groupby].cat.categories
            except:
                pass
        return top_markers_from_info(df_info, n=n, groups=groups, unique=unique)


def normalize_default(adata: sc.AnnData,
                      target_sum=None,
                      copy: bool = False,
                      log_only: bool = False,
                      force_return: bool = False, ):
    """ Normalizing datasets with default settings (total-counts normalization
    followed by log(x+1) transform).

    Parameters
    ----------
    adata
        ``AnnData`` object
    target_sum
        scale factor of total-count normalization
    copy
        whether to copy the dataset
    log_only
        whether to skip the "total-counts normalization" and only perform
        log(x+1) transform
    force_return
        whether to return the data, even if changes are made for the
        original object

    Returns
    -------
    ``AnnData`` or None

    """
    if copy:
        adata = adata.copy()
        logging.info('A copy of AnnData made!')
    else:
        logging.info('No copy was made, the input AnnData will be changed!')
    logging.info('normalizing datasets with default settings.')
    if not log_only:
        logging.info(
            f'performing total-sum normalization, target_sum={target_sum}...')
        sc.pp.normalize_total(adata, target_sum=target_sum)
    else:
        logging.info('skipping total-sum normalization')
    sc.pp.log1p(adata)
    return adata if copy or force_return else None


def change_names(names: Sequence,
                 foo_change: Callable,
                 **kwargs):
    """
    Parameters
    ----------
    names
        a list of names to be modified
    foo_change: function to map a name-string to a new one
    **kwargs: other kwargs for foo_change
    """
    return list(map(foo_change, names, **kwargs))


def take_group_labels(labels: Sequence, group_names: Sequence,
                      indicate=False, remove=False):
    """
    Parameters
    ----------
    labels: list-like
    group_names:
        names of groups that you want to take out
    indicate: bool
        if True, return a Series of bool indicators of the groups
        else, return the labels.
    remove: bool
        False by default, set as True if you want to keep groups that
        NOT in the `given group_names`
    """
    if isinstance(group_names, (str, int)):
        group_names = [group_names]
    if remove:
        indicators = change_names(labels, lambda x: x not in group_names)
    else:
        indicators = change_names(labels, lambda x: x in group_names)
    if indicate:
        return np.array(indicators)
    else:
        return np.array(labels)[indicators]


def remove_small_groups(labels, min_samples=10,
                        indicate=False, ):
    """ return labels with small groups removed
    """
    vcnts = pd.value_counts(labels)
    #    print(vcnts)
    groups_rmv = list(vcnts[vcnts <= min_samples].index)
    logging.info('groups to be removed:\n\t%s', groups_rmv)
    return take_group_labels(labels, group_names=groups_rmv,
                             indicate=indicate, remove=True)


def remove_adata_small_groups(adata: sc.AnnData,
                              key: str,
                              min_samples=10):
    """ return adata with small groups removed, grouped by `key`
    """
    indicators = remove_small_groups(adata.obs[key],
                                     min_samples=min_samples,
                                     indicate=True, )
    return adata[indicators, :].copy()


def compute_and_get_DEGs(adata: sc.AnnData,
                         groupby: str,
                         n=50,
                         groups=None,
                         unique=True,
                         force_redo=False,
                         key_added='rank_genes_groups',
                         inplace=True,
                         do_normalize=False,
                         method='t-test',
                         return_info=False,
                         cuts={},
                         **kwds):
    """ Compute and get DEGs from ``sc.AnnData`` object

    By default, assume that the counts in adata has been normalized.
    If `force_redo`: re-compute DEGs and the original DEGs in adata will be ignored

    cuts: dict with keys 'cut_padj', 'cut_pts', and 'cut_logfc'

    """
    if not inplace:
        logging.info('making a copy')
        adata = adata.copy()
    adata.obs[groupby] = adata.obs[groupby].astype('category')
    if force_redo or key_added not in adata.uns.keys():
        logging.info(f'computing differentially expressed genes using {method}')
        if do_normalize:
            # todo: ``target_sum`` used to be 1e4
            normalize_default(adata, target_sum=None, )
        else:
            logging.info(
                'computing differential expression genes using default settings'
                '(assume that the expression values are already normalized)')
        if True:  # TODO: singletons will raise error
            adata = remove_adata_small_groups(adata, key=groupby, min_samples=1)
        sc.tl.rank_genes_groups(adata, groupby=groupby,
                                key_added=key_added,
                                pts=True,
                                method=method,
                                **kwds)
    if return_info:
        return get_marker_info_table(adata, key=key_added, **cuts)
    return top_markers_from_adata(adata, n=n,
                                  groups=groups,
                                  unique=unique,
                                  key=key_added, **cuts)


if __name__ == '__main__':
    print('hello world')

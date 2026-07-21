# !/usr/bin/env python
# coding: utf-8
# @Time    : 2022/7/17 14:26
# @Author  : Z.-H. G.
# @Email_0 : guozhenhao17@mails.ucas.ac.cn
# @Email_1 : guozhenhao@tongji.edu.cn
# @File    : base2.py
# @IDE     : PyCharm

import torch
import dgl
import copy
import scanpy as sc
import numpy as np
import pandas as pd
import networkx as nx

from scipy import sparse
from scipy.sparse import csr_matrix

from .train_untils import z_score, integrate_feature_gene, subset_matches
from .preprocess_untils import quick_preprocess

np.random.seed(42)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)


class Dataset(object):
    """
    1，load data
    2，qc
    3，preprocess
    4，generate dgl data
    """

    def __init__(self, **kwargs):
        # params
        self.params = kwargs

        # preprocess
        # load data
        self.data_dict, self.data_dict_whole = self.load_data()  # {dataset_name: dataset_adata}
        # get many-to-many homologous  relationship
        self.relationship_multiple = self.load_relationship()  # {relationship_name: [src, rel, dst]}
        self.relationship_single = self.load_relationship_1v1()  # pd.DataFrame

        # generate feature and graph
        # generate feature gene
        self.feature_gene, self.feature_gene_raw_1v1 = self.generate_feature_gene()
        # generate node gene
        self.generate_node_gene()
        # generate node feature
        self.generate_feature()
        # generate index to find gene or cell node from DGL
        self.generate_index()
        # generate cell label
        self.cl_cell_type = self.generate_cl_cell_type()
        self.generate_cell_label()
        # cluster label
        self.generate_cluster_label()
        # graph in data[dataset].uns[graph]
        self.generate_graph()
        # generate dgl with graph and feature
        self.dgl_data = self.generate_dgl_data()

        # use feature gene to generate adata_whole
        self.adata_whole = self.generate_adata_whole()  #
        print()

    def load_data(self):
        """
        load_data
        :return:
        """
        # Check whether the data sets are existed and connected to each other
        self._check_dataset()
        # get_dataset_name
        dataset_name = self._get_dataset_name()
        # load_dataset
        dataset_dict = self._load_dataset(dataset_name)
        # preprocess including dowmsample and get balanced dataset
        dataset_dict = self._preprocess(dataset_dict, dataset_name)
        # qc
        dataset_dict = {k: self._qc_dataset(v) for k, v in dataset_dict.items()}
        # combat
        # dataset_dict = self._combat(dataset_dict)   # TODO
        # get_describe
        self._get_describe(dataset_dict)

        # raw
        dataset_dict_raw = {k: copy.deepcopy(v.uns['whole']) for k, v in dataset_dict.items()}  # raw
        # del
        for name, data in dataset_dict.items():
            del data.uns['whole']
        return dataset_dict, dataset_dict_raw

    def _get_describe(self, dataset_dict):
        # get_describe
        ref_desc = [data.uns['dataset_description'] for data in dataset_dict.values() if
                    data.uns['dataset_type'] == 'reference']
        query_desc = [data.uns['dataset_description'] for data in dataset_dict.values() if
                      data.uns['dataset_type'] == 'query']
        ref_desc.extend(query_desc)
        dataset_descriptions_list = ref_desc
        dataset_descriptions = pd.concat(dataset_descriptions_list, axis=1)
        self.data_description = dataset_descriptions
        print(dataset_descriptions)

    def _check_dataset(self):
        self._check_dataset_exist()
        self._check_dataset_graph()

    def _check_dataset_exist(self):
        import os
        dataset_file = self.params['dataset_file']
        dataset = dataset_file.loc[:, ['source', 'relationship', 'destination']].to_numpy()
        path = self.params['path']
        for item in dataset:
            os.path.exists(path + item[0])  # data1
            os.path.exists(path + item[1])  # gene
            os.path.exists(path + item[2])  # data2
            temp = pd.read_csv(path + item[1], index_col=None, header=0, nrows=10)
            assert temp.columns[0] == item[0]
            assert temp.columns[1] == item[2]

    def _check_dataset_graph(self):
        """
        Check whether the data set is connected
        We need to link all datasets together using many-to-many homology relationships

        :return:
        """
        dataset_file = self.params['dataset_file']
        edges = dataset_file.loc[:, ['source', 'destination']].to_numpy()
        dataset_graph = nx.Graph()
        dataset_graph.add_edges_from(edges)
        for item in list(dataset_graph.degree()):
            if item[1] == 0:
                print('dataset_graph is not connected!')
                raise TypeError
        return

    def _get_dataset_name(self):
        """
        get_dataset_name
        :param
        :return:
        """
        dataset_file = self.params['dataset_file']
        # get_dataset_name
        dataset_all = {}
        for item in dataset_file.to_numpy():
            if item[1] is True:
                dataset_all.setdefault(item[0], 'reference')
            else:
                dataset_all.setdefault(item[0], 'query')
            if item[4] is True:
                dataset_all.setdefault(item[3], 'reference')
            else:
                dataset_all.setdefault(item[3], 'query')
        return dataset_all

    def _load_dataset(self, dataset_name):
        # load dataset
        path = self.params['path']
        dataset_dict = {k[: -5]: sc.read_h5ad(path + k) for k, v in dataset_name.items()}  # {name: adata}，no ".h5ad"
        # dataset_type: ref or query
        for name, data in dataset_dict.items():
            data.uns['dataset_type'] = dataset_name[f'{name}.h5ad']
            data.uns['dataset_name'] = name
        return dataset_dict

    def _preprocess(self, dataset_dict, dataset_name):
        # 1，balance dataset
        if self.params['get_balance'] == 'True':
            ref_names = [name for name, dataset_type in dataset_name.items() if dataset_type == 'reference']
            if len(ref_names) > 0:
                ref_cell_type = np.unique(
                    np.concatenate([dataset_dict[name[:-5]].obs.loc[:, 'cell_ontology_class'].unique()
                                    for name in ref_names]))
                #
                dataset_dict = {name: data[data.obs.loc[:, 'cell_ontology_class'].isin(ref_cell_type), :]
                                for name, data in dataset_dict.items()}
        # 2，downsample counts，
        if 0 < self.params['sample_ratio'] < 1:
            [sc.pp.downsample_counts(data, total_counts=int(self.params['sample_ratio'] * data.X.toarray().sum()),
                                     replace=True) for data in dataset_dict.values()]
        elif 0 < self.params['sample_ratio'] == 1:
            pass
        else:
            print('there must be sth. wrong in sample_ratio and we set sample_ratio to 1')
            pass
        return dataset_dict

    @staticmethod
    def _qc_dataset(dataset):
        #
        if isinstance(dataset.X, csr_matrix):
            pass
        else:
            dataset.X = csr_matrix(dataset.X)

        # unique barcode and gene
        dataset = dataset[:, ~dataset.var.index.duplicated(keep='first')]
        # dataset = dataset[~dataset.obs.index.duplicated(keep='first'), :]

        # # filter 1
        # sc.pp.filter_cells(dataset, min_genes=200)
        # sc.pp.filter_genes(dataset, min_cells=3)

        # filter 2，
        dataset_description = dataset.obs.loc[:, 'cell_ontology_class'].value_counts()
        # #
        # major_classes = dataset_description[dataset_description > 5].index.tolist()  # get the minority classes
        # dataset = dataset[dataset.obs.loc[:, 'cell_ontology_class'].isin(major_classes)]
        #
        dataset_description = pd.DataFrame(dataset_description).astype(int)
        dataset_description.columns = [f'{dataset.uns["dataset_name"]}: {dataset.uns["dataset_type"]}']
        dataset.uns['dataset_description'] = dataset_description

        # whole
        dataset.uns['whole'] = dataset.copy()

        # normalization
        sc.pp.normalize_total(dataset, target_sum=None)
        sc.pp.log1p(dataset)

        # hvg
        sc.pp.highly_variable_genes(dataset, n_top_genes=2000)
        # raw
        dataset.raw = dataset

        # subset
        dataset = dataset[:, dataset.var['highly_variable']]

        # z-score
        sc.pp.scale(dataset, zero_center=True)

        # n_comps = 30
        n_comps = min(30, len(dataset) - 1)
        n_neighbors = min(15, n_comps)
        sc.tl.pca(dataset, n_comps=n_comps)

        if dataset.uns['dataset_type'] == 'reference':
            # #
            # dataset.obs['clust_lbs'] = dataset.obs['cell_ontology_class']
            # cell_type = np.unique(dataset.obs['clust_lbs'].to_numpy())
            # cell_type_dict = {cell_type[i]: str(i) for i in range(len(cell_type))}  #
            # dataset.obs['clust_lbs'] = dataset.obs['clust_lbs'].apply(lambda x: cell_type_dict[x])
            # sc.pp.neighbors(dataset, n_pcs=n_comps, n_neighbors=n_neighbors, use_rep='X_pca', key_added='clust', random_state=0)
            #
            sc.pp.neighbors(dataset, use_rep='X_pca', n_pcs=n_comps, n_neighbors=n_neighbors, key_added='clust', random_state=0, metric='cosine') # TODO
            sc.tl.leiden(dataset, resolution=0.5, neighbors_key='clust', key_added='clust_lbs', random_state=0)
        else:
            #
            sc.pp.neighbors(dataset, use_rep='X_pca', n_pcs=n_comps, n_neighbors=n_neighbors, key_added='clust', random_state=0, metric='cosine')
            sc.tl.leiden(dataset, resolution=0.5, neighbors_key='clust', key_added='clust_lbs')  #

        # DEG
        sc.tl.rank_genes_groups(dataset, groupby='clust_lbs', method='t-test', key_added='rank_genes_groups', pts=True)
        deg_all = pd.DataFrame(dataset.uns['rank_genes_groups']['names'])
        deg_selected = pd.unique(deg_all.iloc[0: 50, :].values.T.flatten())

        # save
        dataset.uns['deg'] = deg_selected
        dataset.uns['hvg'] = dataset.var_names
        hvg_deg = np.unique(np.concatenate((dataset.var_names.to_numpy(), deg_selected)))
        dataset.uns['hvg_deg'] = hvg_deg
        dataset.uns['gene_raw'] = dataset.raw.var_names
        return dataset

    def load_relationship(self):
        # path
        path = self.params['path']
        # get relationship
        relationship_name = self._get_relationship_name()
        # dict
        # relationship_dict = {item[1][: -4]: {'src': item[0][: -5], 'mul_to_mul': pd.read_csv(path + item[1]),
        #                                      'dst': item[2][: -5]} for item in relationship_name}
        # to avoid name error, we change the src and dst name as the 1 and 2 column name
        relationship_dict = {}
        for item in relationship_name:
            rel_temp = pd.read_csv(path + item[1])  # load relationship
            rel_temp.columns = [item[0], item[2]]  # change columns of relationship
            relationship_dict[item[1][: -4]] = {'src': item[0][: -5], 'mul_to_mul': rel_temp, 'dst': item[2][: -5]}
        return relationship_dict

    def _get_relationship_name(self):
        dataset_file = self.params['dataset_file']
        relationship_name = dataset_file.loc[:, ['source', 'relationship', 'destination']].to_numpy().tolist()
        return relationship_name

    def load_relationship_1v1(self):
        """

        :return:
        """
        # path
        path = self.params['path']
        # dataset_file
        dataset_file = self.params['dataset_file']
        # dataset_graph
        edges = dataset_file.loc[:, ['source', 'destination']].to_numpy()
        dataset_graph = nx.Graph()
        dataset_graph.add_edges_from(edges)
        # dfs or bfs on dataset_graph, the result is oriented [(s, d), (s, d)], and merge in that order
        result = list(nx.traversal.bfs_edges(dataset_graph, list(dataset_graph.nodes)[0]))  # (graph, root)
        gene_map_1v1_list = []
        for item in result:
            # Because dfs or bfs results are oriented, (s, d) or (d, s) are in the original data
            # If (s, d) is in the original data
            condition1 = (dataset_file.iloc[:, 0] == item[0]) & (dataset_file.iloc[:, 3] == item[1])
            # If (d, s) is in the original data
            condition2 = (dataset_file.iloc[:, 0] == item[1]) & (dataset_file.iloc[:, 3] == item[0])
            if len(dataset_file[condition1]) == 1:
                # gene_map_1v1_list.extend(dataset_file[condition1].iloc[:, -1].to_numpy().tolist())    # 1v1
                gene_map_1v1_list.extend(dataset_file[condition1].loc[:, 'relationship'].to_numpy().tolist())
            elif len(dataset_file[condition2]) == 1:
                # gene_map_1v1_list.extend(dataset_file[condition2].iloc[:, -1].to_numpy().tolist())    # 1v1
                gene_map_1v1_list.extend(dataset_file[condition2].loc[:, 'relationship'].to_numpy().tolist())
            else:
                print('there must be sth. wrong')
                # raise TypeError
                continue

        # load the first
        gene_map_1v1_all = self.read_csv_drop(path + gene_map_1v1_list[0])
        # merge the rest
        for i in range(1, len(gene_map_1v1_list)):
            gene_map_1v1_all = gene_map_1v1_all.merge(self.read_csv_drop(path + gene_map_1v1_list[i]))
        return gene_map_1v1_all

    def read_csv_drop(self, name):
        # method 1 drop_duplicates，
        df = pd.read_csv(name)
        df = df.drop_duplicates(subset=[df.columns[0]], keep=False).drop_duplicates(subset=[df.columns[1]], keep=False)
        df = df.reset_index(drop=True)

        # # TODO method 2 select one2one
        # df = pd.read_csv(name)
        # df = df.loc[df.iloc[:, 2] == 'ortholog_one2one', :].reset_index(drop=True)
        return df

    def generate_feature_gene(self):
        # feature gene is common to all datasets and is used to generate features of cells and genes
        raw_gene_list = []
        deg_list = []
        for item in self.relationship_single.columns:
            raw_gene_list.append(self.data_dict[item[: -5]].uns['gene_raw'])  #
            deg_list.append(self.data_dict[item[: -5]].uns['deg'])  #
        feature_gene_raw_1v1 = integrate_feature_gene(self.relationship_single, raw_gene_list, union=False)
        feature_gene = integrate_feature_gene(feature_gene_raw_1v1, deg_list, union=True)
        return feature_gene, feature_gene_raw_1v1

    @staticmethod
    def integrate_feature_gene(gene_map, deg_list, union: bool = False):
        """
        Enter 1v1 homologous gene and deg of each dataset
        :param gene_map:
        :param deg_list:
        :param union:
        :return:
        """
        # TODO
        # Enter in a certain order, or determine the name of the dataset and the name of the columns
        # Or combine all 1v1 homologous genes into a single table
        temp = None
        cols = gene_map.columns
        for i in range(len(deg_list)):
            if i == 0:  # i == 0
                temp = gene_map[cols[i]].isin(deg_list[i]).to_frame(cols[i])
            else:  #
                temp[cols[i]] = gene_map[cols[i]].isin(deg_list[i])

        keep = temp.max(1) if union else temp.min(1)  #
        return gene_map[keep]

    def generate_node_gene(self):
        # node gene
        # select
        self._select_node_gene()
        # integrate
        self._integrate_node_gene()

    def _integrate_node_gene(self, mod='union'):
        assert mod in {'intersect', 'union'}
        for name, dataset in self.data_dict.items():
            node_gene_temp = pd.Series()
            for k, v in dataset.uns.items():
                if k[0: 9] == 'node_gene':
                    if len(node_gene_temp) == 0:
                        node_gene_temp = pd.Series(v)
                    elif mod == 'union':
                        node_gene_temp = pd.concat([pd.Series(node_gene_temp), pd.Series(v)]).unique()
                    else:
                        raise NotImplementedError
            node_gene_list = node_gene_temp.tolist()
            dataset.uns['node_gene'] = node_gene_list
            dataset.uns['node_gene_dict'] = {node_gene_list[i]: i for i in range(len(node_gene_list))}  # num: str

    def _select_node_gene(self):
        for rel, rel_details in self.relationship_multiple.items():
            gene_raw_src = self.data_dict[rel_details['src']].uns['gene_raw']  # all expressed gene
            gene_raw_dst = self.data_dict[rel_details['dst']].uns['gene_raw']
            hvg_deg_src = self.data_dict[rel_details['src']].uns['hvg_deg']  # hvg union deg
            hvg_deg_dst = self.data_dict[rel_details['dst']].uns['hvg_deg']

            # The intersection of genes in raw and homologous genes, i.e. homologous genes expressed in the data set
            sub_map = subset_matches(rel_details['mul_to_mul'], gene_raw_src, gene_raw_dst, union=False)
            # The combination of expressed homologous genes and HVG+DEG in the dataset
            sub_map = subset_matches(sub_map, hvg_deg_src, hvg_deg_dst, union=True)
            # self.node_gene_pair = sub_map
            rel_details['sub_map'] = sub_map
            # The above results were combined with HVG+DEG, that is, some homologous genes of opposite species were mapped on the basis of DEG+HVG
            # Submaps are obtained from many-to-many-homologous genes, and there may be duplicates in each column, namely nodes1 and nodes2, so the effect is better if they are not taken
            node_gene_src = pd.concat([sub_map.iloc[:, 0], pd.Series(hvg_deg_src)]).unique().tolist()
            node_gene_dst = pd.concat([sub_map.iloc[:, 1], pd.Series(hvg_deg_dst)]).unique().tolist()
            # node_gene_src = list(set(sub_map.iloc[:, 0]))
            # node_gene_dst = list(set(sub_map.iloc[:, 1]))
            self.data_dict[rel_details['src']].uns['node_gene' + '_' + rel] = node_gene_src  # candidate
            self.data_dict[rel_details['dst']].uns['node_gene' + '_' + rel] = node_gene_dst  # candidate

    def generate_feature(self):

        # combat
        adata_dict = {}
        for item in self.feature_gene.columns:
            gene = self.feature_gene.loc[:, item].to_numpy()
            cell_feature = self.data_dict[item[: -5]].raw[:, gene].X.toarray()
            adata_dict[item] = sc.AnnData(X=cell_feature)
        adata_all = sc.concat([v for v in adata_dict.values()], label='batch', keys=list(adata_dict.keys()))
        sc.pp.combat(adata_all, key='batch')
        feature_dict = {name: adata_all[adata_all.obs.loc[:, 'batch'] == name, :].X.astype(np.float32)
                        for name, data in adata_dict.items()}

        for item in self.feature_gene.columns:  # columns are each data set
            if self.params['feature_gene'] == 'HIG':
                # use HIG genes (DEG+HVG)
                # feature_gene
                gene = self.feature_gene.loc[:, item].to_numpy()
                self.data_dict[item[: -5]].uns['feature_gene'] = gene

                # cell_feature
                try:
                    # cell_feature = self.data_dict[item[: -5]].raw[:, gene].X.toarray()
                    cell_feature = feature_dict[item]     # TODO combat
                except:
                    cell_feature = z_score(self.data_dict[item[: -5]].raw[:, gene].X.toarray())
                # # TODO
                # from sklearn.decomposition import PCA
                # self.n_pcs = 30
                # pca = PCA(n_components=self.n_pcs, svd_solver="arpack", random_state=0)
                # cell_feature = pca.fit_transform(cell_feature)
                # # TODO
                self.data_dict[item[: -5]].uns[item[: -5] + 'cell'] = cell_feature  # z_score(cell_feature)
                # self.data_dict[item[: -5]].uns[item[: -5] + 'cell'] = z_score(cell_feature)  # z_score(cell_feature)

                # gene_feature
                num_gene = len(self.data_dict[item[: -5]].uns['node_gene'])
                num_dim = len(cell_feature[0])
                gene_feature = np.zeros((num_gene, num_dim), dtype=np.float32)
                self.data_dict[item[: -5]].uns[item[: -5] + 'gene'] = gene_feature
            elif self.params['feature_gene'] == 'all':
                # use all genes
                # feature_gene
                gene = self.feature_gene_raw_1v1.loc[:, item].to_numpy()
                self.data_dict[item[: -5]].uns['feature_gene'] = gene

                # cell_feature
                cell_feature = self.data_dict[item[: -5]].raw[:, gene].X.toarray()
                self.data_dict[item[: -5]].uns[item[: -5] + 'cell'] = z_score(cell_feature)

                # gene_feature
                num_gene = len(self.data_dict[item[: -5]].uns['node_gene'])
                num_dim = len(self.feature_gene_raw_1v1)
                gene_feature = np.zeros((num_gene, num_dim), dtype=np.float32)
                self.data_dict[item[: -5]].uns[item[: -5] + 'gene'] = gene_feature
            else:
                raise NotImplementedError

    def generate_index(self):
        # generate node index
        # cell_index and gene_index
        for name, data in self.data_dict.items():
            # The cell feature is generated based on data.obs_names
            cell_index = data.obs_names
            data.uns[name + 'cell_index'] = {i: cell_index[i] for i in range(len(cell_index))}
            # gene_feature
            data.uns[name + 'gene_index'] = {i: data.uns['node_gene'][i] for i in range(len(data.uns['node_gene']))}

    def generate_cl_cell_type(self):
        # generate separately
        cl_cell_type_dict = {'reference': {}, 'query': {}}
        cl_cell_type_reference = []
        cl_cell_type_query = []
        for name, data in self.data_dict.items():
            if data.uns['dataset_type'] == 'reference':
                temp = data.obs['cell_ontology_class'].to_numpy().tolist()
                cl_cell_type_dict['reference'][name] = temp
                cl_cell_type_reference.extend(temp)
            else:
                temp = data.obs['cell_ontology_class'].to_numpy().tolist()
                cl_cell_type_dict['query'][name] = temp
                cl_cell_type_query.extend(temp)

        cl_cell_type_reference = np.unique(np.array(cl_cell_type_reference))
        cl_cell_type_query = np.unique(np.array(cl_cell_type_query))

        # If the cell_type of query exists in reference, use reference; otherwise, set it to unknown
        # reference comes before, query comes after. self.data_description.index is the order
        describe_dict = {item: i for i, item in enumerate(self.data_description.index)}
        cell_type_dict_reference = {item: describe_dict[item] for item in cl_cell_type_reference}
        # unknown
        cell_type_dict_reference.setdefault('unknown', len(cell_type_dict_reference))
        # save
        cl_cell_type_dict['cell_type_dict_reference'] = copy.deepcopy(cell_type_dict_reference)
        # query
        cell_type_dict_query = {cl_cell_type_query[i]: i for i in range(len(cl_cell_type_query))}
        # save
        cl_cell_type_dict['cell_type_dict_query'] = copy.deepcopy(cell_type_dict_query)  # query

        counter = len(cell_type_dict_reference)
        cell_type_dict_all = copy.deepcopy(cell_type_dict_reference)
        for cell_type in cell_type_dict_query.keys():
            if cell_type not in cell_type_dict_all.keys():
                cell_type_dict_all.setdefault(cell_type, counter)
                counter += 1

        cl_cell_type_dict['cell_type_dict_all'] = copy.deepcopy(cell_type_dict_all)  # ref + query

        return cl_cell_type_dict

    def generate_cell_label(self):
        # Convert the cell body type to numeric type
        # stored in data.obs['cell_ontology_class_num']
        cl_cell_type_dict_all = self.cl_cell_type['cell_type_dict_all']
        for name, data in self.data_dict.items():
            data.uns['cell_label_cl'] = [cl_cell_type_dict_all[cell_type] for cell_type in
                                         data.obs['cell_ontology_class'].to_numpy()]
            data.obs.loc[:, ['cell_class']] = [item if item in self.cl_cell_type['cell_type_dict_reference'].keys()
                                               else 'unknown' for item in
                                               data.obs.loc[:, 'cell_ontology_class'].to_numpy()]
            data.obs.loc[:, ['cell_class_num']] = data.obs.loc[:, 'cell_class'].map(
                self.cl_cell_type['cell_type_dict_reference']).astype(int)
            data.uns['cell_label_class'] = data.obs.loc[:, 'cell_class_num'].to_numpy()

    def generate_cluster_label(self):
        for name, data in self.data_dict.items():
            data.uns['cluster_label'] = [int(cluster_label) for cluster_label in data.obs['clust_lbs'].to_numpy()]

    def generate_graph(self):
        """
        After merging different types of graph_gene_candidate to obtain the complete graph_gene, a diagram of the cell-gene is generated
        """
        self._generate_dataset_graph()
        self._generate_relationship_graph()

    def _generate_dataset_graph(self):
        for name, dataset in self.data_dict.items():
            dataset.uns['graph'] = {}  # graph for each dataset
            # 1 cell_knn_cell
            # sc.pp.neighbors(dataset, n_neighbors=30)
            sc.pp.neighbors(dataset, n_pcs=30, n_neighbors=20)   # TODO
            dataset.uns['graph'][tuple(f'{name}cell_knn_{name}cell'.split('_'))] = dataset.obsp['connectivities']

            # 2 cell_selfloop_cell
            node_cell = dataset.obs_names.to_numpy()
            data = np.ones(len(node_cell), dtype=int)
            nodes = np.arange(len(data))
            cell_selfloop_cell = sparse.csr_matrix((data, (nodes, nodes)))
            dataset.uns['graph'][tuple(f'{name}cell_selfloop_{name}cell'.split('_'))] = cell_selfloop_cell

            # 3 gene_selfloop_gene
            node_gene = dataset.uns['node_gene']
            data = np.ones(len(node_gene), dtype=int)
            nodes = np.arange(len(data))
            gene_selfloop_gene = sparse.csr_matrix((data, (nodes, nodes)))
            dataset.uns['graph'][tuple(f'{name}gene_homo_{name}gene'.split('_'))] = gene_selfloop_gene

            # 4 cell_express_gene
            node_gene = dataset.uns['node_gene']
            cell_express_gene = dataset.raw[:, node_gene].X
            dataset.uns['graph'][tuple(f'{name}cell_express_{name}gene'.split('_'))] = cell_express_gene

            # 5 gene_expressedby_cell
            gene_expressedby_cell = cell_express_gene.T
            dataset.uns['graph'][tuple(f'{name}gene_expressedby_{name}cell'.split('_'))] = gene_expressedby_cell

    def _generate_relationship_graph(self):
        for rel, rel_details in self.relationship_multiple.items():
            # graph_gene_homo_gene
            sub_map = rel_details['sub_map']
            src_name = sub_map.iloc[:, 0].to_numpy()
            dst_name = sub_map.iloc[:, 1].to_numpy()

            # gene_homo_gene
            #
            # src to dst
            dict_source = self.data_dict[rel_details['src']].uns['node_gene_dict']
            dict_target = self.data_dict[rel_details['dst']].uns['node_gene_dict']

            src = [dict_source[src_name[i]] for i in range(len(src_name))]
            dst = [dict_target[dst_name[i]] for i in range(len(dst_name))]
            # TODO directed or undirected
            data = np.ones(sub_map.shape[0], dtype=int)
            gene_homo_gene = sparse.csr_matrix((data, (src, dst)))
            # dataset_namegene homo dataset_namegene
            rel_details['graph'] = {}
            rel_details['graph'][
                tuple(f'{rel_details["src"]}gene_homo_{rel_details["dst"]}gene'.split('_'))] = gene_homo_gene
            if self.params['graph_mode'] == 'undirected':
                temp = sparse.csr_matrix((data, (dst, src)))  # reverse graph
                rel_details['graph'][tuple(f'{rel_details["dst"]}gene_homo_{rel_details["src"]}gene'.split('_'))] = temp
            else:
                pass

    def generate_dgl_data(self):
        dgl_graph = self.generate_dgl_graph()  # {relationship: graph}
        data_mode = self.generate_data_mode()  #
        class_num = self.generate_class_num(data_mode)
        data_type = {name: data.uns['dataset_type'] for name, data in self.data_dict.items()}
        cell_type = self.cl_cell_type['reference']
        dgl_data_dict = {'graph': dgl_graph, 'data_mode': data_mode, 'data_type': data_type, 'class_num': class_num,
                         'cell_type': cell_type,
                         'node_gene': {name: data.uns['node_gene'] for name, data in self.data_dict.items()}}

        if self.params['feature_gene'] == 'HIG':
            dgl_data_dict['feature_dim'] = len(self.feature_gene)
            # dgl_data_dict['feature_dim'] = self.n_pcs
            dgl_data_dict['feature_gene'] = {name: data.uns['feature_gene'] for name, data in self.data_dict.items()}
        elif self.params['feature_gene'] == 'all':
            dgl_data_dict['feature_dim'] = len(self.feature_gene_raw_1v1)
            dgl_data_dict['feature_gene'] = {name: data.uns['feature_gene'] for name, data in self.data_dict.items()}

        dgl_data = DGLData(**dgl_data_dict)
        return dgl_data

    def generate_class_num(self, data_mode):
        # annotation
        if data_mode == 'all query':
            cluster_len_list = []
            for name, data in self.data_dict.items():
                cluster_len = len(set(data.obs['clust_lbs'].tolist()))
                cluster_len_list.append(cluster_len)
            class_num = max(cluster_len_list)
        else:
            class_num = len(self.cl_cell_type['cell_type_dict_reference'])
        return class_num

    def generate_data_mode(self):
        # Count the number of reference and query datasets
        mode = None
        reference = 0
        query = 0
        for name, data in self.data_dict.items():
            if data.uns['dataset_type'] == 'reference':
                reference += 1
            elif data.uns['dataset_type'] == 'query':
                query += 1
            else:
                pass

        if reference > 0 and query > 0:
            mode = 'reference and query'
        elif reference > 0 and query == 0:
            mode = 'all reference'
        elif reference == 0 and query > 0:
            mode = 'all query'
        else:
            pass
        return mode

    def generate_dgl_graph(self):
        # edge
        dgl_graph = self._generate_dgl_edge()
        # feature and label
        dgl_graph = self._generate_dgl_feature_and_label(dgl_graph)
        return dgl_graph

    def _generate_dgl_feature_and_label(self, dgl_graph):
        dgl_graph = dgl_graph
        # Assign features and labels to dgl_graph
        for name, data in self.data_dict.items():
            cell = name + 'cell'
            gene = name + 'gene'
            assert cell in dgl_graph.ntypes
            assert gene in dgl_graph.ntypes

            # cell
            # feature
            dgl_graph.nodes[name + 'cell'].data['feature'] = torch.tensor(data.uns[name + 'cell'])
            # index
            dgl_graph.nodes[name + 'cell'].data['index'] = torch.tensor(list(data.uns[name + 'cell_index'].keys()))
            # cell_label_cl
            dgl_graph.nodes[name + 'cell'].data['cell_label_cl'] = torch.tensor(data.uns['cell_label_cl'])
            # cell_label_class
            dgl_graph.nodes[name + 'cell'].data['cell_label_class'] = torch.tensor(data.uns['cell_label_class'])
            # cluster_label
            dgl_graph.nodes[name + 'cell'].data['cluster_label'] = torch.tensor(data.uns['cluster_label'])

            # gene
            # feature
            dgl_graph.nodes[name + 'gene'].data['feature'] = torch.tensor(data.uns[name + 'gene'])
            # index
            dgl_graph.nodes[name + 'gene'].data['index'] = torch.tensor(list(data.uns[name + 'gene_index'].keys()))

        return dgl_graph

    def _generate_dgl_edge(self):
        # Aggregates the edges in the data set and the edges in the relationship together
        graph = {}
        # data graph
        for data_name, data in self.data_dict.items():
            graph.update(data.uns['graph'])

        # relationship graph
        for data_name, data in self.relationship_multiple.items():
            graph.update(data['graph'])

        # After converting the csr sparse matrix in the graph to coo matrix, row col is turned into a tuple
        for k, v in graph.items():
            # coo
            v = v.tocoo()
            # tuple
            row = torch.from_numpy(v.row).to(torch.long)
            col = torch.from_numpy(v.col).to(torch.long)
            # update
            graph[k] = tuple([row, col])

        # dgl
        dgl_graph = dgl.heterograph(graph)
        return dgl_graph

    def generate_adata_whole(self):

        # 1, raw data
        data_dict_raw = copy.deepcopy(self.data_dict_whole)

        # 2, convert gene dict
        gene_convert_dict = {name: {item[0]: item[1]
                                    for item in self.relationship_single.loc[:,
                                                [name, self.relationship_single.columns[0]]].to_numpy()}
                             for name in self.relationship_single.columns}

        # 3 preprocess
        data_dict_preprocess = {}
        for name, data in data_dict_raw.items():
            adata = data.copy()
            # Take only the genes that have been expressed
            condition_gene = adata.var_names.map(gene_convert_dict[name + '.h5ad']).notnull()
            adata = adata[:, condition_gene]
            # map
            adata.var.index = adata.var.index.map(gene_convert_dict[name + '.h5ad'])
            # get hvg and raw
            raw = adata.copy()
            # adata = quick_preprocess(adata)   # TODO
            adata.raw = raw
            # add key
            data_dict_preprocess[name] = adata

        # 4, concat
        data_whole = sc.concat([data for data in data_dict_preprocess.values()], label='batch')   # TODO
        data_whole = quick_preprocess(data_whole, n_top_genes=2000)
        data_order = list(data_dict_preprocess.keys())
        data_whole.obs.loc[:, 'batch'] = data_whole.obs.loc[:, 'batch'].apply(lambda x: data_order[int(x)])  # name
        data_whole.uns['data_order'] = data_order

        # 5，add cell label
        # mapping
        #
        data_whole.obs.loc[:, ['cell_ontology_class_num']] = data_whole.obs.loc[:, 'cell_ontology_class']. \
            map(self.cl_cell_type['cell_type_dict_all']).astype(int)
        # 计算细胞类型
        cell_class = [item if item in self.cl_cell_type['cell_type_dict_reference'].keys() else 'unknown'
                      for item in data_whole.obs.loc[:, 'cell_ontology_class'].to_numpy()]
        data_whole.obs.loc[:, ['cell_class']] = cell_class
        data_whole.obs.loc[:, ['cell_class_num']] = data_whole.obs.loc[:, 'cell_class']. \
            map(self.cl_cell_type['cell_type_dict_reference']).astype(int)

        # 5, postprocess
        sc.pp.regress_out(data_whole, ['total_counts'])
        sc.pp.scale(data_whole, max_value=10)
        sc.tl.pca(data_whole, svd_solver='arpack')
        sc.pp.neighbors(data_whole, n_neighbors=10)
        # sc.pp.neighbors(data_whole, n_neighbors=10, n_pcs=40)

        # 6, uns
        data_whole.uns['dataset_type'] = {name: data.uns['dataset_type'] for name, data in self.data_dict.items()}
        data_whole.uns['dataset_description'] = self.data_description
        data_whole.uns['cell_type'] = self.cl_cell_type['cell_type_dict_all']
        data_whole.uns['hvg'] = data_whole.var.index.tolist()

        return data_whole


class DGLData(object):
    def __init__(self, **kwargs):
        self.graph = kwargs['graph']

        self.data_mode = kwargs['data_mode']
        self.data_type = kwargs['data_type']
        self.feature_dim = kwargs['feature_dim']
        self.class_num = kwargs['class_num']
        self.node_gene = kwargs['node_gene']
        self.feature_gene = kwargs['feature_gene']

    def to_device(self, device):
        self.graph = self.graph.to(device)

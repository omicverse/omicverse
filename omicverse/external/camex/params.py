# !/usr/bin/env python
# coding: utf-8
# @Time    : 2022/7/17 13:54
# @Author  : Z.-H. G.
# @Email_0 : guozhenhao17@mails.ucas.ac.cn
# @Email_1 : guozhenhao@tongji.edu.cn
# @File    : params.py
# @IDE     : PyCharm


import pandas as pd

# TODO dict? yaml?


# all query
PARAMS = \
    {'preprocess':
        {
            'path': './dataset/kidney/',
            'dataset_file': pd.DataFrame(
                data=[['raw-kidney-human-Wu.h5ad', True, 'gene_matches_human2mouse.csv',
                       'raw-kidney-mouse-Adam.h5ad', False],
                      # ['raw-kidney-human-Wu.h5ad', True, 'gene_matches_human2zebrafish.csv',  #
                      #  'raw-kidney-zebrafish-Alemany.h5ad', False],
                      ],
                columns=['source', 'source label', 'relationship', 'destination', 'destination label']),
            'graph_mode': 'undirected',
            # 'feature_gene': 'all',  # can impute all_1v1_gene
            'feature_gene': 'HIG',  # can impute HIG
        },

        'train': {
            'train_mode': 'mini_batch',
            # 'train_mode': 'full_batch',
            'dim_hidden': 64,
            'batch_size': 1024,
            'gnn_layer_num': 2,
            'epoch_pretrain': 20,
            'epoch_train': 20,
            # 'encoder': 'GCN',
            'encoder': 'GAT',
            'decoder': 'Dot',
            # 'decoder': 'GAT',
            # 'decoder': 'Dot_GAT',
           'classifier': 'GAT',
            'use_domain': 'False',

        },

        'postprocess': {}
    }


# # all reference
# PARAMS = \
#     {'preprocess':
#         {
#             'path': './dataset/brain/',
#             'dataset_file': pd.DataFrame(
#                 data=[['raw-brain-human-Lake.h5ad', True, 'gene_matches_human2lizard.csv',
#                        'raw-brain-lizard-Tosches.h5ad', True],
#                       ['raw-brain-human-Lake.h5ad', True, 'gene_matches_human2turtle.csv',
#                        'raw-brain-turtle-Tosches.h5ad', True],
#                       ],
#                 columns=['source', 'source label', 'relationship', 'destination', 'destination label']),
#             'graph_mode': 'undirected',
#             # 'feature_gene': 'all',  # can impute all_1v1_gene
#             'feature_gene': 'HIG',  # can impute HIG
#         },
#
#         'train': {
#             'train_mode': 'mini_batch',
#             # 'train_mode': 'full_batch',
#             'dim_hidden': 64,
#             'batch_size': 1024,
#             'gnn_layer_num': 2,
#             'epoch_pretrain': 2,
#             'epoch_train': 2,
#             # 'encoder': 'GCN',
#             'encoder': 'GAT',
#             'decoder': 'Dot',
#             # 'decoder': 'GAT',
#             # 'decoder': 'Dot_GAT',
#             'classifier': 'GAT',
#             'use_domain': 'False',
#
#         },
#
#         'postprocess': {}
#     }



# # all query
# PARAMS = \
#     {'preprocess':
#         {
#             'path': './dataset/brain/',
#             'dataset_file': pd.DataFrame(
#                 data=[['raw-brain-human-Lake.h5ad', False, 'gene_matches_human2lizard.csv',
#                        'raw-brain-lizard-Tosches.h5ad', False, 'gene_matches_1v1_human2lizard.csv'],
#                       ['raw-brain-human-Lake.h5ad', False, 'gene_matches_human2turtle.csv',
#                        'raw-brain-turtle-Tosches.h5ad', False, 'gene_matches_1v1_human2turtle.csv'],
#                       ['raw-brain-mouse-Chen.h5ad', False, 'gene_matches_mouse2lizard.csv',
#                        'raw-brain-lizard-Tosches.h5ad', False, 'gene_matches_1v1_mouse2lizard.csv'],
#                       ['raw-brain-mouse-Chen.h5ad', False, 'gene_matches_mouse2turtle.csv',
#                        'raw-brain-turtle-Tosches.h5ad', False, 'gene_matches_1v1_mouse2turtle.csv'],
#                       ],
#                 columns=['source', 'source label', 'relationship', 'destination', 'destination label', '1v1test']),
#             'graph_mode': 'undirected',
#             # 'feature_gene': 'all',  # can impute all_1v1_gene
#             'feature_gene': 'HIG',  # can impute HIG
#         },
#
#         'train': {
#             'train_mode': 'mini_batch',
#             # 'train_mode': 'full_batch',
#             'dim_hidden': 64,
#             'batch_size': 1024,
#             'gnn_layer_num': 2,
#             'epoch_pretrain': 20,
#             'epoch_train': 20,
#             # 'encoder': 'GCN',
#             'encoder': 'GAT',
#             'decoder': 'Dot',
#             # 'decoder': 'GAT',
#             # 'decoder': 'Dot_GAT',
#             'classifier': 'GAT',
#             'use_domain': 'False',
#
#         },
#
#         'postprocess': {}
#     }
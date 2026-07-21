#!/usr/bin/python3.8+
# -*- coding: utf-8 -*-
# @Time    : 2022/2/26 21:53
# @Author  : Z.-H. G.
# @Email_0 : guozhenhao17@mails.ucas.ac.cn
# @Email_1 : guozhenhao@tongji.edu.cn
# @File    : script.py
# @IDE: PyCharm

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn import metrics
import numpy as np
from sklearn.metrics import balanced_accuracy_score

np.random.seed(42)


def subset_matches(df_match, left, right, union: bool = False):
    """ Take a subset of token matches (e.g., gene homologies)

    Parameters
    ----------
    df_match: pd.DataFrame
        a dataframe with at least 2 columns
    left:
        list-like, for extracting elements in the first column.
    right:
        list-like, for extracting elements in the second column.
    union:
        whether to take union of both sides of genes
    """

    # TODO
    cols = df_match.columns[: 2]

    c1, c2 = cols
    tmp = df_match[c1].isin(left).to_frame(c1)
    tmp[c2] = df_match[c2].isin(right)

    keep = tmp.max(1) if union else tmp.min(1)

    return df_match[keep]


def integrate_feature_gene(gene_map, deg_list, union: bool = False):
    """
    # Enter 1v1 homologous gene and deg of each dataset
    :param gene_map: 同源基因
    :param args: hvg, deg or hvg+deg
    :param union: True or False
    :return:
    """

    # TODO
    # Enter in a certain order or determine the name of the dataset and the name of the columns
    # Or combine all 1v1 homologous genes into one table
    temp = None
    cols = gene_map.columns
    for i in range(len(deg_list)):
        if i == 0:  #
            temp = gene_map[cols[i]].isin(deg_list[i]).to_frame(cols[i])
        else:  #
            temp[cols[i]] = gene_map[cols[i]].isin(deg_list[i])

    keep = temp.max(1) if union else temp.min(1)  # 取交或并
    return gene_map[keep]


def get_embedding(dataset_dict, hidden):
    for name, feature in hidden.items():
        if name[-4:] == 'cell':
            dataset_dict[name[0: -4]].embedding['cell'] = feature.detach().to('cpu').numpy()  #
        elif name[-4:] == 'gene':
            dataset_dict[name[0: -4]].embedding['gene'] = feature.detach().to('cpu').numpy()  #
        else:
            print('there must be sth. wrong!')
    return


class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        if reduction not in {'sum', 'mean'}:
            raise ValueError('`reduction` should be either "sum" or "mean",'
                             f'got {reduction}')
        self.reduction = reduction

    def forward(self, output, target, weight=None):
        c = output.size()[-1]
        # F.cross_entropy() combines `log_softmax` and `nll_loss` in a single function.
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = - log_preds.sum()
        else:
            loss = - log_preds.sum(dim=-1)
            loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=weight)


def cross_entropy_loss(
        logits, labels, weight=None,
        smooth_eps=0.1,
        reduction='mean',
):
    # take out representations of nodes that have labels
    # F.cross_entropy() combines `log_softmax` and `nll_loss` in a single function
    if smooth_eps > 0.:
        loss = LabelSmoothingCrossEntropy(eps=smooth_eps, reduction=reduction)(
            logits, labels, weight=weight
        )
    else:
        loss = F.cross_entropy(logits, labels, weight=weight, reduction=reduction)
    return loss


def multilabel_binary_cross_entropy(
        logits: torch.Tensor,
        labels: torch.Tensor,
        weight=None,
        reduction: str = 'mean',
):
    """ multi-label binary cross-entropy

    Parameters
    ----------
    logits
        model output logits, without softmax
    labels
        two-dimensional one-hot labels
    weight
        class weights
    reduction
        'mean' or 'sum'

    Returns
    -------
    loss
    """
    # probas = F.sigmoid(logits)
    # loss = F.binary_cross_entropy(probas, labels, weight=weight)
    loss = F.binary_cross_entropy_with_logits(
        logits, labels.float(), weight=weight, reduction=reduction)
    return loss


def loss_classification(predict_prob, labels, labels_1hot=None, weight=None, smooth_eps=0.1, reduction='mean', beta=1.):
    loss = cross_entropy_loss(
        predict_prob,
        labels,
        weight=weight,
        smooth_eps=smooth_eps,
        reduction=reduction,
    )

    if labels_1hot is not None and beta > 0.:
        loss += multilabel_binary_cross_entropy(
            predict_prob,
            labels_1hot,
            weight=weight,
            reduction=reduction,
        ) * beta
    return loss


def z_score(data, with_mean=True, scale=True):
    """ For each column of X, do centering (z-scoring)
    """
    # code borrowed from `scanpy.pp._simple`
    scaler = StandardScaler(with_mean=with_mean, copy=True).partial_fit(data)
    if scale:
        # user R convention (unbiased estimator)
        e_adjust = np.sqrt(data.shape[0] / (data.shape[0] - 1))
        scaler.scale_ *= e_adjust
    else:
        scaler.scale_ = np.array([1] * data.shape[1])
    data_new = scaler.transform(data)
    return data_new


def tensor_to_numpy(dict_in: dict):
    dict_out = {k: v.detach().clone().cpu().numpy() for k, v in dict_in.items()}
    return dict_out


def cal_pred_with_unknown(y_prob: torch.tensor, unknown_num: int, threshold=0.5):
    """
    Reject the prediction probability less than threshold and set it to unknown, accept all when threshold=0
    When unknown is 0, the following can be used instead
    y_predict_train = tensor_to_numpy({dataset_name: torch.max(label, -1)[1] * (torch.max(label, -1)[0] > 0.5) for
                                           dataset_name, label in y_predict_prob_train.items()})
    y_predict_test = tensor_to_numpy({dataset_name: torch.max(label, -1)[1] * (torch.max(label, -1)[0] > 0.5) for
                                          dataset_name, label in y_predict_prob_test.items()})
    :param y_prob:
    :param unknown_num:
    :return:
    """
    y_pred = tensor_to_numpy({dataset_name: torch.max(prob, -1)[1] for dataset_name, prob in y_prob.items()})
    y_prob_condition = tensor_to_numpy({dataset_name: torch.max(prob, -1)[0] > threshold
                                        for dataset_name, prob in y_prob.items()})
    y_predict = {dataset_name: [item if y_prob_condition[dataset_name][num] else unknown_num for
                                num, item in enumerate(y_pred[dataset_name])] for dataset_name, prob in y_prob.items()}

    return y_predict


def get_acc(y_true: dict, y_predict: dict, r=4) -> dict:
    """

    :param y_true:
    :param y_predict:
    :param r:
    :return:
    """
    acc_dict = {}
    for dataset_name, true_label in y_predict.items():
        # method 1 acc
        correct = np.sum(y_true[dataset_name] == y_predict[dataset_name])
        acc = correct / len(y_true[dataset_name])

        # # method 2 balanced_accuracy
        # acc = round(balanced_accuracy_score(y_true[dataset_name], y_predict[dataset_name]), r)

        # acc_dict
        acc_dict[dataset_name + '_acc'] = round(acc, r)  #
    return acc_dict


def get_f1(y_true, y_predict, average='micro', r=4):
    """
    :param y_true:
    :param y_predict:
    :param average: micro: microF1, macro: macroF1, weighted: weightedF1
                    ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    :param r:
    :return:
    """
    f1_dict = {}
    for dataset_name, true_label in y_predict.items():
        f1 = metrics.f1_score(y_true[dataset_name], y_predict[dataset_name], average=average)
        f1_dict[dataset_name + f'_{average}F1'] = round(f1, r)  # 保留有效数字
    return f1_dict


# —————— data integration and batch correlation metric
def get_ami(y_true, y_predict, suffix='_ami', r=4):
    """
    :param y_true: cluster label, not cell label
    :param y_predict:
    :param r:
    :return:
    """
    ami_dict = {}
    for dataset_name, true_label in y_predict.items():
        ami = metrics.adjusted_mutual_info_score(y_true[dataset_name], y_predict[dataset_name])
        ami_dict[dataset_name + suffix] = round(ami, r)  #
    return ami_dict

# more metric ref: Learning interpretable cellular and gene signature embeddings from single-cell transcript data

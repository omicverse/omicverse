import numpy as np
import torch
import torch.nn.functional as F
from typing import List
from torchmetrics.functional.classification import multiclass_f1_score, multiclass_accuracy, multiclass_precision, multiclass_recall 

def aggregate_eval_results(scores: List[dict]):
    scores_new = {}
    for k in scores[0].keys():
        scores_new[k] = []
        for t in scores:
            scores_new[k].append(t[k])
        scores_new[k] = sum(scores_new[k]) / len(scores_new[k])
    scores = scores_new
    return scores

def downstream_eval(task, pred_labels, true_labels, num_classes=None, eval_mask=None, dim=1, 
                    normalize=True, top_de_dict=None, batch_labels=None, control_level=None,
                    topk=20, **kwargs):
    if task == 'annotation':
        return annotation_eval(pred_labels, true_labels, num_classes)
    elif task == 'denoising':
        return denoising_eval(pred_labels, true_labels, eval_mask, normalize)
    elif task == 'imputation':
        return imputation_eval(pred_labels, true_labels, dim)
    elif task == 'perturbation_prediction':
        raise NotImplementedError("For simplicity, the perturbation evaluation is removed from the current release.")
    else:
        raise NotImplementedError(f"{task} should be chosen from ['annotation', 'denoising', 'imputation', 'perturbation_prediction']")

def CountCorr(y_true, y_pred):
    y_true = torch.log1p(y_true)
    y_pred = torch.log1p(y_pred)
    y_true_c = y_true - torch.mean(y_true, 1)[:, None]
    y_pred_c = y_pred - torch.mean(y_pred, 1)[:, None]
    pearson = torch.mean(torch.sum(y_true_c * y_pred_c, 1) / torch.sqrt(torch.sum(y_true_c * y_true_c, 1)) / torch.sqrt(
        torch.sum(y_pred_c * y_pred_c, 1)))
    return pearson

def PearsonCorr(y_true, y_pred):
    assert len(y_true.shape) == 2
    y_true_c = y_true - torch.mean(y_true, 1)[:, None]
    y_pred_c = y_pred - torch.mean(y_pred, 1)[:, None]
    pearson = torch.mean(torch.sum(y_true_c * y_pred_c, 1) / torch.sqrt(torch.sum(y_true_c * y_true_c, 1)) 
                         / torch.sqrt(torch.sum(y_pred_c * y_pred_c, 1)))
    return pearson

def PearsonCorr1d(y_true, y_pred):
    assert len(y_true.shape) == 1
    y_true_c = y_true - torch.mean(y_true)
    y_pred_c = y_pred - torch.mean(y_pred)
    pearson = torch.mean(torch.sum(y_true_c * y_pred_c) / torch.sqrt(torch.sum(y_true_c * y_true_c)) 
                         / torch.sqrt(torch.sum(y_pred_c * y_pred_c)))
    return pearson


def clustering_eval(adata, cluster_key='leiden', label_key='cell_type'):
    raise NotImplementedError("For simplicity, rapids_singlecell was removed from the dependency. Therefore currently the clustering evaluation is not available.")
    import rapids_singlecell as rsc
    from scib.metrics.ari import ari
    from scib.metrics.nmi import nmi
    print('Start building knn.')
    rsc.pp.neighbors(adata, use_rep='X_cellbert')
    best_ari = -1
    best_nmi = -1
    for res in range(1, 15, 1):
        res = res / 10
        rsc.tl.leiden(adata, resolution=res, key_added=cluster_key)
        ari_score = ari(adata, cluster_key=cluster_key, label_key=label_key)
        if ari_score > best_ari:
            best_ari = ari_score
        nmi_score = nmi(adata, cluster_key=cluster_key, label_key=label_key)
        if nmi_score > best_nmi:
            best_nmi = nmi_score
    return {'ari': best_ari, 'nmi':best_nmi}

def minimum_eval(adata):
    raise NotImplementedError("For simplicity, scib was removed from the dependency. Therefore currently the scib evaluation is not available.")
    import scib
    import rapids_singlecell as rsc
    print('Start building knn.')
    rsc.pp.neighbors(adata, use_rep='X_cellbert')
    return scib.metrics.metrics(adata, adata, "batch", "cell_type", embed='X_cellbert', cluster_key="cluster",
                         #organism='human', ari_=True, nmi_=True, pcr_=True, graph_conn_=True)
    organism = 'human', graph_conn_ = True)

def annotation_eval(pred_labels, true_labels, num_classes=None):
    num_classes = len(true_labels.unique()) if num_classes is None else num_classes
    acc = multiclass_accuracy(pred_labels, true_labels, num_classes).cpu().item()
    f1_score = multiclass_f1_score(pred_labels, true_labels, num_classes).cpu().item()
    precision = multiclass_precision(pred_labels, true_labels, num_classes).cpu().item()
    recall = multiclass_recall(pred_labels, true_labels, num_classes).cpu().item()
    return {'acc': acc, 'f1_score': f1_score, 'precision': precision, 'recall': recall}

def normalize_counts(counts):
    counts = F.relu(counts / counts.sum(1, keepdim=True) * 1e4)
    return torch.log1p(counts)

def denoising_eval(pred_labels, true_labels, eval_mask=None, normalize=True):
    if normalize:
        true_labels = normalize_counts(true_labels)
        pred_labels = normalize_counts(pred_labels)
    if eval_mask is not None:
        true_labels = true_labels[eval_mask]
        pred_labels = pred_labels[eval_mask]
        nz_idx = torch.nonzero(true_labels, as_tuple=True)
        true_labels = true_labels#[nz_idx]
        pred_labels = pred_labels#[nz_idx]
        corr = PearsonCorr1d(pred_labels, true_labels).item()
        cos = F.cosine_similarity(pred_labels, true_labels, dim=0).item()
    else:
        corr = PearsonCorr(pred_labels, true_labels).item()
        cos = F.cosine_similarity(pred_labels, true_labels, dim=1).mean().item()
    mse = F.mse_loss(pred_labels, true_labels).item()
    rmse = np.sqrt(mse)
    mae = F.l1_loss(pred_labels, true_labels).item()
    return {'mse': mse, 'rmse':rmse, 'mae':mae, 'corr':corr, 'cos': cos}

def imputation_eval(pred_labels, true_labels, dim=1):
    mse = []
    rmse = []
    rmsle = []
    mae = []
    corr = []
    cos = []
    # if ((true_labels - true_labels.int().float())<1e-6).all():
    #     print('Lognorm')
    #     true_labels = torch.log1p(true_labels)
    #     pred_labels = torch.log1p(pred_labels)
    for i in range(true_labels.shape[dim]):
        true_vec = true_labels[i] if dim == 0 else true_labels[:, i]
        pred_vec = F.relu(pred_labels[i]) if dim == 0 else F.relu(pred_labels[:, i])
        (nz_idx,) = torch.nonzero(true_vec, as_tuple=True)
        true_nz = true_vec#[nz_idx]
        pred_nz = pred_vec#[nz_idx]
        mse.append(F.mse_loss(pred_nz, true_nz).item())
        rmse.append(np.sqrt(mse))
        # rmsle.append(np.sqrt(F.mse_loss(torch.log(pred_nz + 1), torch.log(true_nz + 1)).item()))
        mae.append(F.l1_loss(pred_nz, true_nz).item())
        corr.append(PearsonCorr1d(pred_nz, true_nz).item())
        cos.append(F.cosine_similarity(pred_nz, true_nz, dim=0).item())
    rmse = np.concatenate(rmse)
    return {
        'mse': sum(mse) / len(mse),
        'rmse': sum(rmse) / len(rmse),
        # 'rmsle': sum(rmsle) / len(rmsle),
        'mae': sum(mae) / len(mae),
        'corr': sum(corr) / len(corr),
        'cos': sum(cos) / len(cos),
    }

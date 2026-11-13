import torch
import torch.nn as nn
import numpy as np
import anndata as ad
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from tqdm import tqdm
from copy import deepcopy
from ..utils.eval import downstream_eval, aggregate_eval_results
from ..utils.data import XDict, TranscriptomicDataset
from typing import List, Union
from .experimental import symbol_to_ensembl
from torch.utils.data import DataLoader
import warnings
from . import Pipeline, load_pretrain

CellTypeAnnotationDefaultModelConfig = {
    'drop_node_rate': 0.3,
    'dec_layers': 1,
    'model_dropout': 0.5,
    'mask_node_rate': 0.75,
    'mask_feature_rate': 0.25,
    'dec_mod': 'mlp',
    'latent_mod': 'ae',
    'head_type': 'annotation',
    'max_batch_size': 70000,
}

CellTypeAnnotationDefaultPipelineConfig = {
    'es': 200,
    'lr': 5e-3,
    'wd': 1e-7,
    'scheduler': 'plat',
    'epochs': 2000,
    'max_eval_batch_size': 100000,
    'hvg': 3000,
    'patience': 25,
    'workers': 0,
}
def inference(model, dataloader, split, device, batch_size, eval_dict, label_fields=None, order_required=False):
    if order_required and split:
        warnings.warn('When cell order required to be preserved, dataset split will be ignored.')

    with torch.no_grad():
        model.eval()
        epoch_loss = []
        order_list = []
        pred = []
        label = []
        for i, data_dict in enumerate(dataloader):
            if not order_required and split and np.sum(data_dict['split'] == split) == 0:
                continue

            idx = torch.arange(data_dict['x_seq'].shape[0])
            if split:
                data_dict['loss_mask'] = torch.from_numpy((data_dict['split'] == split).values).bool()
            else:
                data_dict['loss_mask'] = torch.ones(data_dict['x_seq'].shape[0]).bool()
            if label_fields:
                data_dict['label'] = data_dict[label_fields[0]]
            for j in range(0, len(idx), batch_size):
                if len(idx) - j < batch_size:
                    cur = idx[j:]
                else:
                    cur = idx[j:j + batch_size]
                input_dict = {}
                for k in data_dict:
                    if k =='x_seq':
                        input_dict[k] = data_dict[k].index_select(0, cur).to(device)
                    elif k not in ['gene_list', 'split']:
                        input_dict[k] = data_dict[k][cur].to(device)
                x_dict = XDict(input_dict)
                out_dict, loss = model(x_dict, data_dict['gene_list'])
                if 'label' in input_dict:
                    epoch_loss.append(loss.item())
                    label.append(out_dict['label'])
                if order_required:
                    order_list.append(input_dict['order_list'])
                pred.append(out_dict['pred'])

        pred = torch.cat(pred)
        if order_required:
            order = torch.cat(order_list)
            order.scatter_(0, order.clone(), torch.arange(order.shape[0]).to(order.device))
            pred = pred[order]

        if len(epoch_loss) == 0:
            return {'pred': pred}
        else:
            scores = downstream_eval('annotation', pred, torch.cat(label),
                                           **eval_dict)
            return {'pred': pred,
                    'loss': sum(epoch_loss)/len(epoch_loss),
                    'metrics': scores}

class CellTypeAnnotationPipeline(Pipeline):
    def __init__(self,
                 pretrain_prefix: str,
                 overwrite_config: dict = CellTypeAnnotationDefaultModelConfig,
                 pretrain_directory: str = './ckpt',
                 ):
        assert 'out_dim' in overwrite_config, '`out_dim` must be provided in `overwrite_config` for initializing a cell type annotation pipeline. '
        super().__init__(pretrain_prefix, overwrite_config, pretrain_directory)
        self.eval_dict = {'num_classes': overwrite_config['out_dim']}
        self.label_encoders = None

    def fit(self, adata: ad.AnnData,
            train_config: dict = None,
            split_field: str = None,
            train_split: str = 'train',
            valid_split: str = 'valid',
            covariate_fields: List[str] = None,
            label_fields: List[str] = None,
            batch_gene_list: dict = None,
            ensembl_auto_conversion: bool = True,
            device: Union[str, torch.device] = 'cpu',
            ):
        config = CellTypeAnnotationDefaultPipelineConfig.copy()
        if train_config:
            config.update(train_config)
        self.model.to(device)
        assert not self.fitted, 'Current pipeline is already fitted and does not support continual training. Please initialize a new pipeline.'
        if batch_gene_list is not None:
            raise NotImplementedError('Batch specific gene set is not implemented for cell type annotation pipeline. Please raise an issue on Github for further support.')
        if len(label_fields) != 1:
            raise NotImplementedError(f'`label_fields` containing multiple labels (f{len(label_fields)}) is not implemented for cell type annotation pipeline. Please raise an issue on Github for further support.')
        assert (split_field and train_split and valid_split), '`train_split` and `valid_split` must be specified.'
        adata = self.common_preprocess(adata, config['hvg'], covariate_fields, ensembl_auto_conversion)
        print(f'After filtering, {adata.shape[1]} genes remain.')
        dataset = TranscriptomicDataset(adata, split_field, covariate_fields, label_fields)
        self.label_encoders = dataset.label_encoders
        dataloader = DataLoader(dataset, batch_size=None, shuffle=True, num_workers=config['workers'])
        optim = torch.optim.AdamW([
            {'params': list(self.model.embedder.parameters()), 'lr': config['lr'] * 0.1,
             'weight_decay': 1e-10},
            {'params': list(self.model.encoder.parameters()) + list(self.model.head.parameters()) + list(
                self.model.latent.parameters()), 'lr': config['lr'],
             'weight_decay': config['wd']},
        ])
        if config['scheduler'] == 'plat':
            scheduler = ReduceLROnPlateau(optim, 'min', patience=config['patience'], factor=0.95)
        else:
            scheduler = None

        train_loss = []
        valid_loss = []
        valid_metric = []
        final_epoch = -1
        best_dict = None

        for epoch in tqdm(range(config['epochs'])):
            self.model.train()
            epoch_loss = []
            train_scores = []

            if epoch < 30:
                for param_group in optim.param_groups[1:]:
                    param_group['lr'] = config['lr'] * (epoch + 1) / 30

            for i, data_dict in enumerate(dataloader):
                input_dict = data_dict.copy()
                del input_dict['gene_list'], input_dict['split']
                input_dict['loss_mask'] = torch.from_numpy((data_dict['split'] == train_split).values).bool()
                input_dict['label'] = input_dict[label_fields[0]] # Currently only support annotating one label
                for k in input_dict:
                    input_dict[k] = input_dict[k].to(device)
                x_dict = XDict(input_dict)
                out_dict, loss = self.model(x_dict, data_dict['gene_list'])
                with torch.no_grad():
                    train_scores.append(
                        downstream_eval('annotation', out_dict['pred'], out_dict['label'], **self.eval_dict))

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                optim.step()
                epoch_loss.append(loss.item())

                if config['scheduler'] == 'plat':
                    scheduler.step(loss.item())

            train_loss.append(sum(epoch_loss) / len(epoch_loss))
            train_scores = aggregate_eval_results(train_scores)
            result_dict = inference(self.model, dataloader, valid_split, device,
                                                config['max_eval_batch_size'], self.eval_dict, label_fields)
            valid_scores = result_dict['metrics']
            valid_loss.append(result_dict['loss'])
            valid_metric.append(valid_scores['f1_score'])

            print(f'Epoch {epoch} | Train loss: {train_loss[-1]:.4f} | Valid loss: {valid_loss[-1]:.4f}')
            print(
                f'Train ACC: {train_scores["acc"]:.4f} | Valid ACC: {valid_scores["acc"]:.4f} | '
                f'Train f1: {train_scores["f1_score"]:.4f} | Valid f1: {valid_scores["f1_score"]:.4f} | '
                f'Train pre: {train_scores["precision"]:.4f} | Valid pre: {valid_scores["precision"]:.4f}')

            if max(valid_metric) == valid_metric[-1]:
                best_dict = deepcopy(self.model.state_dict())
                final_epoch = epoch

            if max(valid_metric) != max(valid_metric[-config['es']:]):
                print(f'Early stopped. Best validation performance achieved at epoch {final_epoch}.')
                break

        assert best_dict, 'Best state dict was not stored. Please report this issue on Github.'
        self.model.load_state_dict(best_dict)
        self.fitted = True
        return self

    def predict(self, adata: ad.AnnData,
                inference_config: dict = None,
                covariate_fields: List[str] = None,
                batch_gene_list: dict = None,
                ensembl_auto_conversion: bool = True,
                device: Union[str, torch.device] = 'cpu',
                ):
        config = CellTypeAnnotationDefaultPipelineConfig.copy()
        if inference_config:
            config.update(inference_config)
        self.model.to(device)
        assert self.fitted, 'Cell type annotation pipeline does not support zero shot setting. Please fine-tune the model on downstream datasets before inference.'
        if batch_gene_list is not None:
            raise NotImplementedError('Batch specific gene set is not implemented for cell type annotation pipeline. Please raise an issue on Github for further support.')
        adata = self.common_preprocess(adata, config['hvg'], covariate_fields, ensembl_auto_conversion)
        print(f'After filtering, {adata.shape[1]} genes remain.')
        dataset = TranscriptomicDataset(adata, None, covariate_fields, order_required=True)
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=config['workers'])
        return inference(self.model, dataloader, None, device,
                  config['max_eval_batch_size'], self.eval_dict, order_required=True)['pred']

    def score(self, adata: ad.AnnData,
              evaluation_config: dict = None,
              split_field: str = None,
              target_split: str = 'test',
              covariate_fields: List[str] = None,
              label_fields: List[str] = None,
              batch_gene_list: dict = None,
              ensembl_auto_conversion: bool = True,
              device: Union[str, torch.device] = 'cpu',
              ):
        config = CellTypeAnnotationDefaultPipelineConfig.copy()
        if evaluation_config:
            config.update(evaluation_config)
        self.model.to(device)
        assert self.fitted, 'Cell type annotation pipeline does not support zero shot setting. Please fine-tune the model on downstream datasets before inference.'
        if batch_gene_list is not None:
            raise NotImplementedError('Batch specific gene set is not implemented for cell type annotation pipeline. Please raise an issue on Github for further support.')
        if len(label_fields) != 1:
            raise NotImplementedError(
                f'`label_fields` containing multiple labels (f{len(label_fields)}) is not implemented for cell type annotation pipeline. Please raise an issue on Github for further support.')
        if target_split:
            assert split_field, '`split_filed` must be provided when `target_split` is specified.'
        adata = self.common_preprocess(adata, config['hvg'], covariate_fields, ensembl_auto_conversion, )
        print(f'After filtering, {adata.shape[1]} genes remain.')
        dataset = TranscriptomicDataset(adata, split_field, covariate_fields, label_fields, label_encoders=self.label_encoders)
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=config['workers'])
        return inference(self.model, dataloader, target_split, device,
                  config['max_eval_batch_size'], self.eval_dict, label_fields)['metrics']
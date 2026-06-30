import pandas as pd
import random

import numpy as np
import torch
#from deepforest import CascadeForestClassifier
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import solve
from sklearn.neighbors import KDTree
from scipy.optimize import nnls
import anndata
import os
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm,trange
from torch.utils.data import TensorDataset, DataLoader



def create_st(generate_sc_data, generate_sc_meta, spot_num, cell_num, gene_num, marker_used):
    r"""
    Create synthetic spatial transcriptomics data from single-cell data.
    
    Generates synthetic spots by randomly sampling and aggregating single cells
    to create spatial transcriptomics-like expression profiles.

    Arguments:
        generate_sc_data: Single-cell expression data matrix
        generate_sc_meta: Single-cell metadata with cell types
        spot_num: Number of synthetic spots to create
        cell_num: Number of cells per spot
        gene_num: Number of marker genes to use
        marker_used: Whether to use marker genes for spot creation

    Returns:
        tuple: (sc_data, sc_meta, spots_data, spots_meta)
    """
    sc = generate_sc_data
    sc_ct = generate_sc_meta
    cell_name = sorted(list(set(sc_ct.Cell)))

    last_cell_pool = []
    spots = pd.DataFrame()
    meta = pd.DataFrame(columns=['Cell', 'Celltype', 'Spot'])
    sc_ct.index = sc_ct['Cell']
    for i in range(spot_num):
        cell_pool = random.sample(cell_name, cell_num)
        while set(cell_pool) == set(last_cell_pool):
            cell_pool = random.sample(cell_name, cell_num)
        last_cell_pool = cell_pool
        syn_spot = sc[cell_pool].sum(axis=1)
        if syn_spot.sum() > 25000:
            syn_spot *= 20000 / syn_spot.sum()
        spot_name = f'spot_{i + 1}'
        spots.insert(len(spots.columns), spot_name, syn_spot)

        for cell in cell_pool:
            celltype = sc_ct.loc[cell, 'Cell_type']
            row = {'Cell': cell, 'Celltype': celltype, 'Spot': spot_name}
            #meta = pd.concat([meta,pd.DataFrame(row)],axis=0)
            meta.loc[len(meta)]=pd.Series(row)
            #meta = meta.append(row, ignore_index=True)

    if marker_used:
        import scanpy as scp
        adata = scp.AnnData(sc.T)

        adata.obs = sc_ct[['Cell_type']]
        scp.tl.rank_genes_groups(adata, 'Cell_type', method='wilcoxon')
        marker_df = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(gene_num)
        marker_array = np.array(marker_df)
        marker_array = np.ravel(marker_array)
        marker_array = np.unique(marker_array)
        marker = list(marker_array)
        sc = sc.loc[marker, :]
        spots = spots.loc[marker, :]
        
    return sc, sc_ct, spots, meta


def create_sample(sc, st, meta, multiple):
    r"""
    Create positive and negative training samples for mapping model.
    
    Generates training data pairs of single cells and spatial spots with
    positive (correct) and negative (incorrect) associations.

    Arguments:
        sc: Single-cell expression data
        st: Spatial transcriptomics expression data
        meta: Metadata linking cells to spots
        multiple: Multiplier for negative samples

    Returns:
        tuple: (positive_features, negative_features)
    """
    cell_name = meta.Cell.values.tolist()
    spot_name = meta.Spot.values.tolist()

    # get wrong spot name for negative data
    all_spot = list(set(meta.Spot))
    wrong_spot_name = []
    for sn in spot_name:
        last_spot = all_spot.copy()  # --
        last_spot.remove(sn)  # --
        mul_wrong = random.sample(last_spot, multiple)
        wrong_spot_name.extend(mul_wrong)

    cfeat_p_list, cfeat_n_list = [], []

    for c in cell_name:
        cell_feat = sc[c].values.tolist()

        cfeat_p_list.append(cell_feat)
        cfeat_m = [cell_feat * multiple]
        cfeat_n_list.extend(cfeat_m)

    cfeat_p = np.array(cfeat_p_list)  # [n, d]
    cfeat_n = np.array(cfeat_n_list).reshape(-1, cfeat_p.shape[1])  # [n*m, d]

    # positive spot features
    sfeat_p_list = []
    for s in spot_name:
        spot_feat = st[s].values.tolist()
        sfeat_p_list.append(spot_feat)
        # sfeat_p = np.vstack((sfeat_p, spot_feat))
    sfeat_p = np.array(sfeat_p_list)  # [n, d]

    mfeat_p = sfeat_p - cfeat_p
    feat_p = np.hstack((cfeat_p, sfeat_p))
    feat_p = np.hstack((feat_p, mfeat_p))
    print('sucessfully create positive data')

    # negative spot features
    sfeat_n_list = []
    for s in wrong_spot_name:
        spot_feat = st[s].values.tolist()
        sfeat_n = sfeat_n_list.append(spot_feat)
    sfeat_n = np.array(sfeat_n_list)

    mfeat_n = sfeat_n - cfeat_n
    feat_n = np.hstack((cfeat_n, sfeat_n))
    feat_n = np.hstack((feat_n, mfeat_n))
    print('sucessfully create negative data')

    return feat_p, feat_n


def get_data(pos, neg):
    r"""
    Combine positive and negative samples into training dataset.
    
    Concatenates positive and negative feature matrices with corresponding labels.

    Arguments:
        pos: Positive sample features
        neg: Negative sample features

    Returns:
        tuple: (combined_features, labels)
    """
    X = np.vstack((pos, neg))
    y = np.concatenate((np.ones(pos.shape[0]), np.zeros(neg.shape[0])))

    return X, y

def create_data_pyomic(single_data:anndata.AnnData,
                       spatial_data:anndata.AnnData,
                       celltype_key:str,spot_key:list=['xcoord','ycoord'],
                       ):
    r"""
    Prepare integrated dataset from single-cell and spatial data.
    
    Aligns and formats single-cell and spatial transcriptomics data for
    mapping analysis by finding common genes and organizing metadata.

    Arguments:
        single_data: Single-cell RNA-seq data as AnnData object
        spatial_data: Spatial transcriptomics data as AnnData object
        celltype_key: Column name for cell type annotations
        spot_key: Column names for spatial coordinates (['xcoord','ycoord'])

    Returns:
        dict: Dictionary containing aligned data matrices and metadata
    """
    print("...loading data")
    input_data = {}
    sc_gene=single_data.var.index.values.tolist()
    st_gene=spatial_data.var.index.values.tolist()
    intersection_genes=[]
    for i in sc_gene:
        if i in st_gene:
            intersection_genes.append(i)

    intersect_gene = intersection_genes
    generate_sc_data = single_data[:,intersect_gene].to_df().T
    st_data = spatial_data[:,intersect_gene].to_df().T
    ism=pd.DataFrame(index=single_data.obs.index)
    ism['Cell']=single_data.obs.index
    ism['Cell_type']=single_data.obs[celltype_key].values
    generate_sc_meta = ism
    ssm=pd.DataFrame(index=spatial_data.obs.index)
    ssm['Spot']=spatial_data.obs.index
    ssm['xcoord']=spatial_data.obs[spot_key[0]].values
    ssm['ycoord']=spatial_data.obs[spot_key[1]].values

    input_data["input_sc_meta"] = ism
    input_data["input_sc_data"] = generate_sc_data
    input_data["input_st_data"] = st_data
    input_data["input_st_meta"] = ssm
    input_data["sc_gene"]=sc_gene
    input_data["st_gene"]=st_gene
    input_data["intersect_gene"]=intersect_gene

    ssm=pd.DataFrame(index=spatial_data.obs.index)
    ssm['Spot']=spatial_data.obs.index
    ssm['xcoord']=spatial_data.obs[spot_key[0]].values
    ssm['ycoord']=spatial_data.obs[spot_key[1]].values

    return input_data


    


def create_data(generate_sc_meta, generate_sc_data, st_data, spot_num, cell_num, top_marker_num, marker_used,
                mul_train):
    r"""
    Create training data for single-cell to spatial mapping.
    
    Generates synthetic training data by creating artificial spots and
    positive/negative sample pairs for model training.

    Arguments:
        generate_sc_meta: Single-cell metadata with cell types
        generate_sc_data: Single-cell expression matrix
        st_data: Spatial transcriptomics reference data
        spot_num: Number of synthetic spots to create
        cell_num: Number of cells per spot
        top_marker_num: Number of top marker genes to use
        marker_used: Whether to use marker gene selection
        mul_train: Multiplier for training sample generation

    Returns:
        tuple: (training_features, training_labels)
    """
    sc_gene = generate_sc_data.index.values.tolist()
    st_gene = st_data.index.values.tolist()

    intersection_genes=[]
    for i in sc_gene:
        if i in st_gene:
            intersection_genes.append(i)

    intersect_gene = intersection_genes
    generate_sc_data = generate_sc_data.loc[intersect_gene]
    st_data = st_data.loc[intersect_gene]
    sc_train, _, st_train, meta_train = create_st(generate_sc_data, generate_sc_meta,
                                                  spot_num, cell_num,
                                                  top_marker_num, marker_used)
    pos_train, neg_train = create_sample(sc_train, st_train, meta_train, mul_train)
    xtrain, ytrain = get_data(pos_train, neg_train)
    
    return xtrain, ytrain


def creat_pre_data(st, cell_name, spot_name, spot_indx, cfeat, return_np=False):
    spot = spot_name[spot_indx]
    spot_feat = st[spot].values
    tlist = np.isnan(spot_feat).tolist()
    tlist = [i for i, x in enumerate(tlist) if x == True]
    assert len(tlist) == 0

    sfeat = np.tile(spot_feat, (len(cell_name), 1))
    mfeat = sfeat - cfeat
    feat = np.hstack((cfeat, sfeat))
    feat = np.hstack((feat, mfeat))
    if not return_np:
        feat = torch.from_numpy(feat).type(torch.FloatTensor)

    tlist = np.isnan(sfeat).tolist()
    tlist = [i for i, x in enumerate(tlist) if x == True]
    assert len(tlist) == 0

    tlist = np.isnan(cfeat).tolist()
    tlist = [i for i, x in enumerate(tlist) if x == True]
    assert len(tlist) == 0

    return feat

def creat_pre_data_batch(st, cell_name, spot_name, start_indx, end_indx, cfeat, return_np=False):
    feats = []
    for i in range(start_indx, end_indx):
        spot = spot_name[i]
        spot_feat = st[spot].values
        tlist = np.isnan(spot_feat).tolist()
        tlist = [j for j, x in enumerate(tlist) if x == True]
        assert len(tlist) == 0

        sfeat = np.tile(spot_feat, (len(cell_name), 1))
        mfeat = sfeat - cfeat
        feat = np.hstack((cfeat, sfeat))
        feat = np.hstack((feat, mfeat))
        feats.append(feat)

    feats = np.array(feats)
    if not return_np:
        feats = torch.from_numpy(feats).type(torch.FloatTensor)

    tlist = np.isnan(sfeat).tolist()
    tlist = [i for i, x in enumerate(tlist) if x == True]
    assert len(tlist) == 0

    tlist = np.isnan(cfeat).tolist()
    tlist = [i for i, x in enumerate(tlist) if x == True]
    assert len(tlist) == 0

    return feats


def predict_for_one_spot(model, st_test, cell_name, spot_name, spot_indx, cfeat):
    feats = creat_pre_data(st_test, cell_name, spot_name, spot_indx, cfeat, return_np=True)
    outputs = model.predict_proba(feats)[:, 1]
    # outputs = np.where(outputs>0.5, 1, 0)
    predict = outputs.tolist()
    return spot_indx, predict


def predict_for_one_spot_svm(model, st_test, cell_name, spot_name, spot_indx, cfeat):
    feats = creat_pre_data(st_test, cell_name, spot_name, spot_indx, cfeat, return_np=True)
    outputs = model(torch.tensor(feats)).detach().numpy()[:, 1]
    # outputs = np.where(outputs>0.5, 1, 0)
    predict = outputs.tolist()
    return spot_indx, predict

def predict_for_one_spot_net(model, st_test, cell_name, spot_name, spot_indx, cfeat):
    feats = creat_pre_data(st_test, cell_name, spot_name, spot_indx, cfeat, return_np=True)
    outputs = model(torch.tensor(feats)).detach().numpy().reshape(-1)
    # outputs = np.where(outputs>0.5, 1, 0)
    predict = outputs.tolist()
    return spot_indx, predict

def predict_for_one_spot_net_batch(model, st_test, cell_name, spot_name, spot_indx, cfeat):
    feats = creat_pre_data_batch(st_test, cell_name, spot_name, spot_indx, cfeat, return_np=True)
    outputs = model(torch.tensor(feats)).detach().numpy().reshape(-1,len(cell_name))
    return outputs
    # outputs = np.where(outputs>0.5, 1, 0)
    predict = outputs.tolist()
    return spot_indx, predict



# Define the model
class SVM(nn.Module):
    r"""
    Simple linear Support Vector Machine model.
    
    PyTorch implementation of SVM for binary classification in mapping tasks.
    """
    
    def __init__(self, X_train,num_classes=2):
        r"""
        Initialize SVM with input dimensions.

        Arguments:
            X_train: Training data to determine input dimensions
            num_classes: Number of output classes (2)
            
        Returns:
            None
        """
        super().__init__()
        torch.manual_seed(2)
        self.linear = nn.Linear(X_train.shape[1], num_classes)

    def forward(self, x):
        r"""
        Forward pass through linear layer.

        Arguments:
            x: Input tensor
            
        Returns:
            torch.Tensor: Linear transformation output
        """
        return self.linear(x)
    
# 定义神经网络模型
class Net(nn.Module):
    r"""
    Neural network for single-cell to spatial mapping.
    
    Simple feedforward network with one hidden layer for predicting
    cell-spot associations in spatial mapping tasks.
    """
    
    def __init__(self,size):
        r"""
        Initialize neural network with specified input size.

        Arguments:
            size: Input feature dimension
            
        Returns:
            None
        """
        super(Net, self).__init__()
        torch.manual_seed(2)
        self.linear1 = nn.Linear(size, 256)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        r"""
        Forward pass through neural network.

        Arguments:
            x: Input feature tensor
            
        Returns:
            torch.Tensor: Sigmoid-activated output probabilities
        """
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

class DFRunner:
    r"""
    Deep learning model runner for single-cell to spatial mapping.
    
    Manages the training and inference pipeline for mapping single cells to
    spatial coordinates using neural network models. Handles data preprocessing,
    model training, and result generation.
    """
    
    def __init__(self,
                 generate_sc_data,
                 generate_sc_meta,
                 st_data, st_meta,
                 marker_used,
                 top_marker_num,
                 random_seed=0,
                 n_jobs=1,device=None,):
        r"""
        Initialize DFRunner with data and configuration.

        Arguments:
            generate_sc_data: Single-cell expression data matrix
            generate_sc_meta: Single-cell metadata with cell types
            st_data: Spatial transcriptomics expression data
            st_meta: Spatial metadata with coordinates
            marker_used: Whether to use marker gene selection
            top_marker_num: Number of top marker genes to select
            random_seed: Random seed for reproducibility (0)
            n_jobs: Number of parallel jobs (1)
            device: Computation device (None)
            
        Returns:
            None
        """
        import scanpy as sc

        self.sc_test_allgene = generate_sc_data  # pandas.DataFrame, generated gene-cell expression data
        self.cell_type = generate_sc_meta  # pandas.DataFrame, cell type
        self.st_data = st_data  # pandas.DataFrame, gene-sport expression data
        self.meta_test = st_meta  # pandas.DataFrame, spot coordinate.
        self.n_jobs=n_jobs

        self.sc_test = self.sc_test_allgene  # pd.DataFrame, test cell-gene expression data.
        self.st_test = self.st_data  # pd.DataFrame, test spot-gene expression data.
        sc_gene = self.sc_test.index.values.tolist()
        st_gene = self.st_test.index.values.tolist()
        
        intersect_gene = list(set(sc_gene).intersection(set(st_gene)))
        self.sc_test = self.sc_test.loc[intersect_gene]
        self.st_test = self.st_test.loc[intersect_gene]
        self.used_device=device
        

        if marker_used:
            print('select top %d marker genes of each cell type...' % top_marker_num)

            scdata = sc.AnnData(self.sc_test.T)
            scdata.obs = self.cell_type[['Cell_type']]
            sc.tl.rank_genes_groups(scdata, 'Cell_type', method='wilcoxon')
            marker_df = pd.DataFrame(scdata.uns['rank_genes_groups']['names']).head(top_marker_num)
            marker_array = np.array(marker_df)
            marker_array = np.ravel(marker_array)
            marker_array = np.unique(marker_array)
            marker = list(marker_array)
            self.sc_test = self.sc_test.loc[marker, :]
            self.st_test = self.st_test.loc[marker, :]

        #self.model = CascadeForestClassifier(random_state=random_seed, n_jobs=n_jobs,
        #                                     verbose=0)
        #self.model = KNN(k=2)
        

        breed = self.cell_type['Cell_type']
        breed_np = breed.values
        breed_set = set(breed_np)
        self.id2label = sorted(list(breed_set))
        self.label2id = {label: idx for idx, label in enumerate(self.id2label)}
        self.cell2label = dict()
        self.label2cell = defaultdict(set)
        for row in self.cell_type.itertuples():
            cell_name = getattr(row, 'Cell')
            cell_type = self.label2id[getattr(row, 'Cell_type')]
            self.cell2label[cell_name] = cell_type
            self.label2cell[cell_type].add(cell_name)

    def run(self, xtrain, ytrain, max_cell_in_diff_spot_ratio, k, save_dir, save_name, load_path=None,
            num_epochs=1000,batch_size=1000,predicted_size=32):
        
        self.model = Net(xtrain.shape[1]).to(self.used_device)
        print('...The model size is %d' % xtrain.shape[1])
        if load_path is None:
            print('...Net training')
            #xtrain = torch.tensor(xtrain, dtype=torch.float32).to(self.used_device)
            #ytrain = torch.tensor(ytrain, dtype=torch.float32).to(self.used_device)
            # 定义损失函数和优化器
            criterion = nn.BCELoss()
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)

            inputs = torch.tensor(xtrain, dtype=torch.float32).to(self.used_device)
            labels = torch.tensor(ytrain.reshape(len(ytrain),1), dtype=torch.float32).to(self.used_device)


            # 将数据分成小批量
            batch_size = batch_size
            dataset = TensorDataset(inputs, labels)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # 训练模型
            with trange(num_epochs) as t:
                for epoch in t:
                    running_loss = 0.0
                    num_samples = 0
                    for i, data in enumerate(dataloader):
                        # 从数据集中获取输入和标签
                        inputs, labels = data

                        # 将输入和标签转换为 PyTorch 的变量
                        inputs = inputs.requires_grad_()
                        labels = labels.float()

                        # 计算模型输出和损失
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)

                        # 反向传播和优化
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # 记录损失值
                        running_loss += loss.item()*inputs.size(0)
                        num_samples += inputs.size(0)

                    # 输出平均损失值
                    average_loss = running_loss / num_samples
                    t.set_description('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, average_loss))

            '''  
            # 训练模型
            num_epochs = 10000
            for epoch in range(num_epochs):
                # 计算模型输出和损失
                output_tensor = self.model(inputs)
                loss = criterion(output_tensor, labels)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if epoch % 100 == 0:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
            '''
            """
            self.model = SVM(xtrain,2).to(self.used_device)

            # Define the loss function
            criterion = nn.MultiMarginLoss()

            # Define the optimizer
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)

            # Train the model
            for epoch in range(10000):
                optimizer.zero_grad()
                y_pred = self.model(xtrain)
                loss = criterion(y_pred, ytrain)
                loss.backward()
                optimizer.step()
                if epoch % 100 == 0:
                    print("Epoch: %d, Loss: %.4f" % (epoch, loss.item()))
"""
            #self.model.fit(xtrain, ytrain)  # train model
            print('...Net training done!')
            
            path_save = os.path.join(save_dir, f"{save_name}.pth")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(self.model.state_dict(), path_save)
            print(f"...save trained net in {path_save}.")

        else:
            self.model.load_state_dict(torch.load(load_path,map_location=self.used_device))
            #self._load_model(load_path)

        #return self.cre_csv(max_cell_in_diff_spot_ratio, k)
        df_meta, df_spot = self.cre_csv(max_cell_in_diff_spot_ratio, k,predicted_size)  # save predicted results.
        return df_meta, df_spot

    def load(self, load_path,modelsize,max_cell_in_diff_spot_ratio, k,predicted_size):
        
        self.model = Net(modelsize).to(self.used_device)
        self.model.load_state_dict(torch.load(load_path,map_location=self.used_device))
        #self._load_model(load_path

        df_meta, df_spot = self.cre_csv(max_cell_in_diff_spot_ratio, k,predicted_size)  # save predicted results.
        return df_meta, df_spot


    def cre_csv(self, max_cell_in_diff_spot_ratio, k,batch_size):
        # data
        cell_name = self.sc_test.columns.values.tolist()
        spot_name = self.st_test.columns.tolist()  # list of spot name ['spot_1', 'spot_2', ...]
        cfeat = self.sc_test.values.T
        cell_num = len(cell_name)
        spot_num = len(spot_name)
        batch_size=batch_size
        if max_cell_in_diff_spot_ratio is None:
            max_cell_in_diff_spot = None
        else:
            max_cell_in_diff_spot = int(max_cell_in_diff_spot_ratio * k * spot_num / cell_num)

        def joint_predict(ratio):            
            score_triple_list = list()
            spot2cell = defaultdict(set)
            cell2spot = defaultdict(set)
            spot2ratio = dict()
            
            print('Calculating scores...')
            #batch_size = batch_size  # adjust this value based on the available memory and the size of the data
            batch_list = [(i, min(i+batch_size, spot_num)) for i in range(0, spot_num, batch_size)]

            with trange(len(batch_list)) as t:
                for batch_indx in t:
                    start_indx, end_indx = batch_list[batch_indx]
                    feat=creat_pre_data_batch(self.st_test, cell_name, spot_name, start_indx,end_indx, cfeat, return_np=True)
                    predict=self.model(torch.tensor(feat, dtype=torch.float32).to(self.used_device)).cpu().detach().numpy().reshape(-1,len(cell_name))
                    for spot_indx,spot_p_num in zip(range(start_indx, end_indx),range(len(spot_name[start_indx:end_indx]))):
                        spot = spot_name[spot_indx]  # spotname
                        for c, p in zip(cell_name,predict[spot_p_num]):
                            score_triple_list.append((c, spot, p))
                        spot2ratio[spot] = np.round(ratio[spot_indx] * k)
                        t.set_description('Now calculating scores for spot %d/%d' % (spot_indx + 1, len(spot_name)))

            print('Calculating scores done.')

            # spot2ratio: map spot to cell type ratio in it.
            score_triple_list = sorted(score_triple_list, key=lambda x: x[2], reverse=True)
            # sort by score
            for c, spt, score in score_triple_list:
                # cell name, spot name, score
                if max_cell_in_diff_spot is not None and len(cell2spot[c]) == max_cell_in_diff_spot:
                    # The number of this cell in different spots reaches a maximum
                    continue
                if len(spot2cell[spt]) == k:
                    # The maximum number of cells in this spot
                    continue
                cell_class = self.cell2label.get(c)
                if cell_class is None:
                    continue
                if spot2ratio[spt][cell_class] > 0:
                    # Put this cell in this spot
                    spot2ratio[spt][cell_class] -= 1
                    spot2cell[spt].add(c)
                    cell2spot[c].add(spt)
                else:
                    continue

            cell_list, spot_list, spot_len = list(), list(), list()
            df_spots = pd.DataFrame()

            order_list = spot_name
            for spot in order_list:
                if spot2cell.get(spot):
                    cells = spot2cell.get(spot)
                    cell_num = len(cells)
                    cell_list.extend(sorted(list(cells)))
                    spot_list.extend([spot] * cell_num)
                    spot_len.append(cell_num)
                    cell_pool = list(cells)
                    cell_pool.sort()

                    predict_spot = self.sc_test_allgene[cell_pool]
                    df_spots = pd.concat([df_spots, predict_spot], axis=1)
            return cell_list, spot_list, spot_len, df_spots

        ratio = self.__calc_ratio()  # [spot_num, class_num]

        cell_list, spot_list, spot_len, df_spots = joint_predict(ratio)
        #return joint_predict(ratio)
        meta = {'Cell': cell_list, 'Spot': spot_list}
        df = pd.DataFrame(meta)
        self.cell_type = self.cell_type.reset_index(drop=True)
        df_meta = pd.merge(df, self.cell_type, how='left')
        df_meta = df_meta[['Cell', 'Cell_type', 'Spot']]
        df_meta = pd.merge(df_meta, self.meta_test, how='inner')

        df_meta = df_meta.rename(columns={'xcoord': 'Spot_xcoord', 'ycoord': 'Spot_ycoord'})

        coord = self.meta_test[['xcoord', 'ycoord']].to_numpy()
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(coord)
        distances, indices = nbrs.kneighbors(coord)
        radius = distances[:, -1] / 2
        radius = radius.tolist()
        all_coord = df_meta[['Spot_xcoord', 'Spot_ycoord']].to_numpy()
        all_radius = list()

        for i in range(len(spot_len)):
            a = [radius[i]] * spot_len[i]
            all_radius.extend(a)

        length = np.random.uniform(0, all_radius)
        angle = np.pi * np.random.uniform(0, 2, all_coord.shape[0])

        x = all_coord[:, 0] + length * np.cos(angle)
        y = all_coord[:, 1] + length * np.sin(angle)
        cell_coord = {'Cell_xcoord': np.around(x, 2).tolist(), 'Cell_ycoord': np.around(y, 2).tolist()}
        df_cc = pd.DataFrame(cell_coord)
        df_meta = pd.concat([df_meta, df_cc], axis=1)

        cell_rename = [f'C_{i}' for i in range(1, df_spots.shape[1] + 1)]
        df_spots.columns = cell_rename
        df_meta = df_meta.drop(['Cell'], axis=1)
        df_meta.insert(0, "Cell", cell_rename)

        return df_meta, df_spots

    def __calc_ratio(self):

        label_devide_data = dict()
        for label, cells in self.label2cell.items():
            label_devide_data[label] = self.sc_test[list(cells)]

        single_cell_splitby_breed_np = {}
        for key in label_devide_data.keys():
            single_cell_splitby_breed_np[key] = label_devide_data[key].values  # [gene_num, cell_num]
            single_cell_splitby_breed_np[key] = single_cell_splitby_breed_np[key].mean(axis=1)

        max_decade = len(single_cell_splitby_breed_np.keys())
        single_cell_matrix = []
        
        for i in range(max_decade):
            single_cell_matrix.append(single_cell_splitby_breed_np[i].tolist())

        single_cell_matrix = np.array(single_cell_matrix)
        single_cell_matrix = np.transpose(single_cell_matrix)  # (gene_num, label_num)

        num_spot = self.st_test.values.shape[1]

        spot_ratio_values = np.zeros((num_spot, max_decade))  # (spot_num, label_num)
        spot_values = self.st_test.values  # (gene_num, spot_num)

        for i in range(num_spot):
            ratio_list = [0 for x in range(max_decade)]
            spot_rep = spot_values[:, i].reshape(-1, 1)
            spot_rep = spot_rep.reshape(spot_rep.shape[0], )

            ratio = nnls(single_cell_matrix, spot_rep)[0]

            ratio_list = [r for r in ratio]
            ratio_list = (ratio_list / np.sum([ratio_list], axis=1)[0]).tolist()

            for j in range(max_decade):
                spot_ratio_values[i, j] = ratio_list[j]
                
        return spot_ratio_values

    def _load_model(self, save_path):
        self.model.load(save_path)
        print(f"loading model from {save_path}")

    def _save_model(self, save_path):
        self.model.save(save_path)
        print(f"saving model to {save_path}")
        return save_path


def aprior(gamma_hat, axis=None):
    m = np.mean(gamma_hat, axis=axis)
    s2 = np.var(gamma_hat, ddof=1, axis=axis)
    
    return (2 * s2 + np.power(m, 2)) / s2


def bprior(gamma_hat, axis=None):
    m = np.mean(gamma_hat, axis=axis)
    s2 = np.var(gamma_hat, ddof=1, axis=axis)
    
    return (m * s2 + np.power(m, 3)) / s2


def postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2 + a - 1)


def it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001):
    n = np.sum(~np.isnan(sdat), axis=1)
    g_old = g_hat
    d_old = d_hat
    change = 1
    count = 0
    while change > conv:
        g_new = postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = np.nansum(np.power(sdat - np.dot(np.expand_dims(g_new, axis=1), np.ones((1, sdat.shape[1]))), 2), axis=1)
        d_new = postvar(sum2, n, a, b)
        change = np.max((np.max(np.abs(g_new - g_old) / g_old), np.max(np.abs(d_new - d_old) / d_old)))
        g_old = g_new
        d_old = d_new
        count += 1
        
    return np.concatenate((np.expand_dims(g_new, axis=1), np.expand_dims(d_new, axis=1)), axis=1)

def joint_analysis(dat, batch, mod=None, par_prior=True, proir_plots=False, mean_only=False, ref_batch=None):
    r"""
    Perform batch effect correction using empirical Bayes methods.
    
    Implements ComBat-style batch correction to remove systematic differences
    between batches while preserving biological variation.

    Arguments:
        dat: Expression data matrix with genes as rows
        batch: Batch assignments for each sample
        mod: Model matrix for covariates (None)
        par_prior: Whether to use parametric priors (True)
        proir_plots: Whether to generate prior plots (False)
        mean_only: Whether to adjust means only (False)
        ref_batch: Reference batch for correction (None)

    Returns:
        pd.DataFrame: Batch-corrected expression data
    """
    rownames = dat.index
    colnames = dat.columns
    dat = np.array(dat)
    batch_levels = batch.drop_duplicates()

    batches = []
    ref_index = 0
    zero_rows_list = []
    for i, batch_level in enumerate(batch_levels):
        idx = batch.isin([batch_level])
        if batch_level == ref_batch:
            ref_index = i
        batches.append(idx.reset_index().loc[lambda d: d.Batch == True].index)
        batch_dat = dat[:, idx]
        for row in range(np.size(batch_dat, 0)):
            if np.var(batch_dat[row, :], ddof=1) == 0:
                zero_rows_list.append(row)
    zero_rows = list(set(zero_rows_list))
    keep_rows = list(set(range(dat.shape[0])).difference(set(zero_rows_list)))
    dat_origin = dat
    dat = dat[keep_rows, :]
    batchmod = pd.get_dummies(batch, drop_first=False, prefix='batch')
    batchmod['batch_' + ref_batch] = 1
    ref = batchmod.columns.get_loc('batch_' + ref_batch)
    design = np.array(batchmod)
    n_batch = batch_levels.shape[0]
    n_batches = [len(x) for x in batches]

    B_hat = solve(np.dot(design.T, design), np.dot(design.T, dat.T))
    grand_mean = np.expand_dims(B_hat[ref, :], axis=1)
    ref_dat = dat[:, batches[ref_index]]
    var_pool = np.expand_dims(np.dot(np.square(ref_dat - np.dot(design[batches[ref_index], :], \
                                                                B_hat).T),
                                     np.ones(n_batches[ref_index]).T * 1 / n_batches[ref_index]), axis=1)
    stand_mean = np.dot(grand_mean, np.ones((1, batch.shape[0])))
    s_data = (dat - stand_mean) / np.dot(np.sqrt(var_pool), np.ones((1, batch.shape[0])))
    batch_design = design
    gamma_hat = solve(np.dot(batch_design.T, batch_design), np.dot(batch_design.T, s_data.T))
    delta_hat = np.empty([0, s_data.shape[0]])
    for i in batches:
        row_vars = np.expand_dims(np.nanvar(s_data[:, i], axis=1, ddof=1), axis=0)
        delta_hat = np.concatenate((delta_hat, row_vars), axis=0)
    gamma_bar = np.mean(gamma_hat, axis=1)
    t2 = np.var(gamma_hat, axis=1, ddof=1)
    a_prior = aprior(delta_hat, axis=1)
    b_prior = bprior(delta_hat, axis=1)
    results = []
    gamma_star = np.empty((n_batch, s_data.shape[0]))
    delta_star = np.empty((n_batch, s_data.shape[0]))
    for j, batch_level in enumerate(batch_levels):
        i = batchmod.columns.get_loc('batch_' + batch_level)
        results.append(it_sol(s_data[:, batches[j]], gamma_hat[i, :], delta_hat[j, :], gamma_bar[i], t2[i], a_prior[j],
                              b_prior[j]).T)
    for j, batch_level in enumerate(batch_levels):
        gamma_star[j, :] = results[j][0]
        delta_star[j, :] = results[j][1]
    gamma_star[ref_index, :] = 0
    delta_star[ref_index, :] = 1
    bayesdata = s_data
    for i, batch_index in enumerate(batches):
        bayesdata[:, batch_index] = (bayesdata[:, batch_index] - np.dot(batch_design[batch_index, :], gamma_star).T) / \
                                    np.dot(np.sqrt(np.expand_dims(delta_star[i], axis=1)), np.ones((1, n_batches[i])))

    bayesdata = (bayesdata * np.dot(np.sqrt(var_pool), np.ones((1, dat.shape[1])))) + stand_mean
    bayesdata[:, batches[ref_index]] = dat[:, batches[ref_index]]
    if len(zero_rows) > 0:
        dat_origin[keep_rows, :] = bayesdata
        bayesdata = pd.DataFrame(dat_origin, index=rownames, columns=colnames)
    bayesdata[bayesdata < 0] = 0
    
    return bayesdata


def knn(data, query, k):
    r"""
    Find k-nearest neighbors using KDTree.
    
    Efficient k-nearest neighbor search for spatial mapping applications.

    Arguments:
        data: Reference data points for neighbor search
        query: Query points to find neighbors for
        k: Number of nearest neighbors to find

    Returns:
        tuple: (distances, indices) of k-nearest neighbors
    """
    tree = KDTree(data)
    dist, ind = tree.query(query, k)
    
    return dist, ind

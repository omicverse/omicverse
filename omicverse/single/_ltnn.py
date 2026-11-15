import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import scipy

from scipy.stats import norm
from scipy.stats import dweibull
from tqdm import tqdm


class scLTNN(object):

    def __init__(self,adata,basis,input_dim,cpu='cuda:0'):
        self.adata=adata
        self.basis=basis
        self.input_dim=input_dim
        #self.model=model
        if cpu=='cpu':
            self.cpu=True 
        else:
            self.cpu=False
        self.device = torch.device(cpu)

    def ANNmodel_init(self,pseudotime,batch_size,):

        self.model = RadioModel(self.input_dim).to(self.device)
        #self.adata=adata

        # Assuming you have your X_train, Y_train, X_test, and Y_test tensors
        ran=np.random.choice(self.adata.obs.index.tolist(),8*(len(self.adata.obs.index.tolist())//10))
        ran_r=list(set(self.adata.obs.index.tolist())-set(ran))
        
        X_train=self.adata[ran].obsm[self.basis]
        Y_train=self.adata.obs.loc[ran,pseudotime]
        X_test=self.adata[ran_r].obsm[self.basis]
        Y_test=self.adata.obs.loc[ran_r,pseudotime]
        # Convert X_train and y_train into PyTorch tensors
        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.Y_train = torch.tensor(Y_train.values.reshape(len(Y_train.values),1), dtype=torch.float32).to(self.device)
        
        # Convert x_test and y_test into PyTorch tensors and move to GPU
        self.X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.Y_test = torch.tensor(Y_test.values.reshape(len(Y_test.values),1), dtype=torch.float32).to(self.device)
        
        # Create a DataLoader for batching
        batch_size = batch_size
        self.train_dataset = TensorDataset(self.X_train, self.Y_train)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

    def ANNmodel_train(self,n_epochs):
        train_losses = []
        val_losses = []
        train_maes = []
        val_maes = []
        # Training loop
        criterion = nn.MSELoss()  # Mean squared error loss
        optimizer = optim.Adam(self.model.parameters())

        num_epochs = n_epochs  # Adjust as needed
        with tqdm(total=num_epochs, desc="Pre-ANN model") as pbar:
            for epoch in range(num_epochs):
                for batch_X, batch_y in self.train_loader:
                    optimizer.zero_grad()  # Clear gradients
            
                    # Forward pass
                    outputs = self.model(batch_X)
            
                    # Compute loss
                    loss = criterion(outputs, batch_y)
            
                    # Backpropagation
                    loss.backward()
            
                    # Update weights
                    optimizer.step()

                train_losses.append(loss.item())
                train_mae = torch.mean(torch.abs(outputs - batch_y))
                train_maes.append(train_mae.item())
                
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(self.X_test)
                    val_loss = criterion(outputs, self.Y_test)
                    val_losses.append(val_loss.item())
                    val_mae = torch.mean(torch.abs(outputs - self.Y_test))
                    val_maes.append(val_mae.item())
                pbar.set_postfix({'val loss, val mae':'{0:1.5f}, {0:1.5f}'.format(val_loss,val_mae)})  # 输入一个字典，显示实验指标
                pbar.update(1)

        self.train_losses=train_losses
        self.val_losses=val_losses
        self.train_maes=train_maes
        self.val_maes=val_maes

    def ANNmodel_predicted(self,x):
        return self.model(x)


    def ANNmodel_save(self,save_path):
        # Save the entire model (architecture and parameters)
        torch.save(self.model.state_dict(), save_path)

    def ANNmodel_load(self,load_path):
        self.model = RadioModel(self.input_dim).to(self.device)
        if self.cpu==False:
            self.model.load_state_dict(torch.load(load_path))
        else:
            self.model.load_state_dict(torch.load(load_path,map_location=torch.device('cpu')))

    def cal_paga(self,use_rep='X_pca',n_neighbors=15, n_pcs=50,resolution=1.0):
        print('......calculate paga')
        from ..pp import neighbors, leiden
        from ._sc_compat import paga
        neighbors(self.adata, n_neighbors=15, n_pcs=50,
               use_rep=use_rep)
        leiden(self.adata, resolution=resolution, random_state=0, backend='python')
        paga(self.adata, groups='leiden')

    def cal_model_time(self):
        r"""
        predict the latent time by primriary ANN model
    Parameters
    ----------
        
        """
        print('......predict model_time')
        
        X_val=self.adata.obsm[self.basis]
        X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        predicted=self.model(X_val)
        y_pred=predicted.cpu().detach().numpy().reshape(-1)
        self.adata.obs['p_time']=y_pred
        self.adata.obs['p_time_r']=1-y_pred


    def cal_exp_gene_value(self,mode='exp',rev=False):
        r"""
        Calculated the gene with the same trend in the expression gene amount of cell
        
        Parameters
    ----------
        """
        res_pd=find_related_gene(self.adata)
        h_gene=res_pd.loc[res_pd['cor']>0.5,'gene'].values
        if len(h_gene)!=0:
            RPS_s=self.adata[self.adata.obs['p_time']<0.2,h_gene].X.mean()
            RPS_l=self.adata[self.adata.obs['p_time']>0.8,h_gene].X.mean()
            if RPS_s==0:
                RPS_s=self.adata[self.adata.obs['p_time']<0.4,h_gene].X.mean()
            if RPS_l==0:
                RPS_l=self.adata[self.adata.obs['p_time']>0.6,h_gene].X.mean()
            print('gene in p_time_low',RPS_s,'gene in p_time_high',RPS_l)
        else:
            RPS_s=self.adata.obs.loc[self.adata.obs['p_time']<0.2,'n_genes'].mean()
            RPS_l=self.adata.obs.loc[self.adata.obs['p_time']>0.8,'n_genes'].mean()
            if RPS_s==0:
                RPS_s=self.adata.obs.loc[self.adata.obs['p_time']<0.4,'n_genes'].mean()
            if RPS_l==0:
                RPS_l=self.adata.obs.loc[self.adata.obs['p_time']>0.6,'n_genes'].mean()
        if ((RPS_s>RPS_l)):
            self.adata.obs['p_latent_time']=self.adata.obs['p_time']
        else:
            self.adata.obs['p_latent_time']=self.adata.obs['p_time_r']
        if rev==True:
            self.adata.obs['p_latent_time']=1-self.adata.obs['p_latent_time']
        if mode=='exp':
            data1=1-self.adata.obs['n_genes'].values
            min_vals = np.min(data1, axis=0)
            max_vals = np.max(data1, axis=0)

            # Perform Min-Max normalization
            normalized_data = (data1 - min_vals) / (max_vals - min_vals)
            if ((RPS_s>RPS_l)):
                self.adata.obs['p_latent_time']=(normalized_data+self.adata.obs['p_time'].values)/2
            else:
                self.adata.obs['p_latent_time']=(normalized_data+self.adata.obs['p_time_r'].values)/2
            if rev==True:
                self.adata.obs['p_latent_time']=1-normalized_data

            
    def cal_dpt_pseudotime(self,leiden_range_start=0.01,leiden_range_end=0.01,leiden_range_mid=0.05,rev=False):
        r"""
        calculate the diffusion pseudotime of anndata by start node selected automatically
    Parameters
    ----------
            leiden_range_start: the range of start node
            leiden_range_end: the range of end node
            leiden_range_mid: the range of middle node
            rev: test function to use the end node

        """

        print('......calculate dpt_pseudotime')
        leiden_pd=pd.DataFrame(columns=['Time_value'])
        for i in set(self.adata.obs['leiden']):
            leiden_pd.loc[i]={'Time_value':self.adata.obs.loc[self.adata.obs['leiden']==i,'p_latent_time'].mean()}
        leiden_pd=leiden_pd.sort_values('Time_value')
        self.leiden_pd=leiden_pd
        leiden_start=leiden_pd.index[0]
        leiden_end=leiden_pd.index[-1]

        self.leiden_start=leiden_pd.loc[leiden_pd['Time_value']<leiden_pd.loc[leiden_pd.index[0]].values[0]+leiden_range_start].index.tolist()
        self.leiden_end=leiden_pd.loc[leiden_pd['Time_value']>leiden_pd.loc[leiden_pd.index[-1]].values[0]-leiden_range_end].index.tolist()
        
        
        #prev
        self.adata.uns['iroot'] = np.flatnonzero(self.adata.obs['leiden'].isin(self.leiden_start))[0]
        from ..pp import diffmap, dpt
        diffmap(self.adata)
        dpt(self.adata)
        self.adata.obs['dpt_pseudotime_p']=self.adata.obs['dpt_pseudotime'].values

        #middle
        leiden_dpt_pd=pd.DataFrame(columns=['Time_value'])
        for i in set(self.adata.obs['leiden']):
            leiden_dpt_pd.loc[i]={'Time_value':self.adata.obs.loc[self.adata.obs['leiden']==i,'dpt_pseudotime'].mean()}
        leiden_dpt_pd=leiden_dpt_pd.sort_values('Time_value') 
        self.leiden_dpt_pd=leiden_dpt_pd
        leiden_sum=len(leiden_dpt_pd)
        leiden_middle=leiden_sum//2
        
        leiden_middle_index=leiden_dpt_pd.iloc[leiden_middle].name
        leiden_middle_value=leiden_dpt_pd.iloc[leiden_middle].values[0]

        self.leiden_middle=leiden_dpt_pd.loc[(leiden_dpt_pd['Time_value']<leiden_middle_value+leiden_range_mid)&
                        (leiden_dpt_pd['Time_value']>leiden_middle_value-leiden_range_mid)].index.tolist()

        self.leiden_start=list(set(self.leiden_start).difference(set(self.leiden_middle)))
        self.leiden_end=list(set(self.leiden_end).difference(set(self.leiden_middle)))
        
        print('......leiden_start:',self.leiden_start)
        print('......leiden_middle',self.leiden_middle)
        print('......leiden_end',self.leiden_end)
        
        
        #rev
        if rev==True:
            self.adata.uns['iroot'] = np.flatnonzero(self.adata.obs['leiden'].isin(self.leiden_end))[0]
            diffmap(self.adata)
            dpt(self.adata)
            self.adata.obs['dpt_pseudotime_r']=np.max(self.adata.obs['dpt_pseudotime'].values)-self.adata.obs['dpt_pseudotime'].values

            self.adata.obs['dpt_pseudotime']=(self.adata.obs['dpt_pseudotime_p'].values+self.adata.obs['dpt_pseudotime_r'].values)/2
        else:
            self.adata.obs['dpt_pseudotime']=self.adata.obs['dpt_pseudotime_p']
        
        
    def ANN(self,batch_size=30,n_epochs=200,verbose=0,mode='p_time'):
        r"""
        regression of latent time by start and end node using ANN model
    Parameters
    ----------
            batch_size: the batch_size of ANN model
            epochs: the epochs of ANN model
            verbose: the visualization of ANN model summary
            mode: the calculation mode of ANN model, if we want to use the diffusion time to regression the ANN model, we can set the model 'dpt_time'
        
        """
        print('......ANN')
        plot_pd=pd.DataFrame()
        plot_pd['cell_id']=self.adata.obs.index
        plot_pd['leiden']=self.adata.obs.loc[plot_pd['cell_id'],'leiden'].values
        plot_pd['dpt_time']=self.adata.obs['dpt_pseudotime'].values
        plot_pd['p_latent_time']=self.adata.obs['p_latent_time'].values
        a1=plot_pd.loc[plot_pd['leiden'].isin(self.leiden_start)].copy()
        a1.sort_values(by=['dpt_time'],inplace=True)
        #a1['p_time']=np.random.normal(0.07, 0.03, len(a1))
        a2=plot_pd.loc[plot_pd['leiden'].isin(self.leiden_end)].copy()
        a2.sort_values(by=['dpt_time'],inplace=True)
        #a2['p_time']=np.random.normal(0.93, 0.03, len(a2))
        a3=plot_pd.loc[plot_pd['leiden'].isin(self.leiden_middle)].copy()
        a3.sort_values(by=['dpt_time'],inplace=True)

        a1['p_time']=np.sort(np.random.normal(0.07, 0.03, len(a1)))
        a2['p_time']=np.sort(np.random.normal(0.93, 0.03, len(a2)))
        a3['p_time']=np.sort(np.random.normal(0.5, 0.05, len(a3)))

        train_pd=pd.concat([a1,a3,a2])
        train_pd.index=train_pd['cell_id']



        ran=np.random.choice(train_pd.index.tolist(),8*(len(train_pd)//10))
        ran_r=list(set(train_pd.index.tolist())-set(ran))

        if mode=='p_time':
            X_train=self.adata[ran].obsm[self.basis]
            #Y_train=adata_test.obs.loc[ran,'p_latent_time']
            Y_train=train_pd.loc[ran,'p_time']
            X_test=self.adata[ran_r].obsm[self.basis]
            #Y_test=adata_test.obs.loc[ran_r,'p_latent_time']
            Y_test=train_pd.loc[ran_r,'p_time']
        elif mode=='dpt_time':
            X_train=self.adata[ran].obsm[self.basis]
            #Y_train=adata_test.obs.loc[ran,'p_latent_time']
            Y_train=train_pd.loc[ran,'dpt_time']
            X_test=self.adata[ran_r].obsm[self.basis]
            #Y_test=adata_test.obs.loc[ran_r,'p_latent_time']
            Y_test=train_pd.loc[ran_r,'dpt_time']

        # Convert X_train and y_train into PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        Y_train = torch.tensor(Y_train.values.reshape(len(Y_train.values),1), dtype=torch.float32).to(self.device)
        
        # Convert x_test and y_test into PyTorch tensors and move to GPU
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        Y_test = torch.tensor(Y_test.values.reshape(len(Y_test.values),1), dtype=torch.float32).to(self.device)

        model = RadioModel(self.adata.obsm[self.basis].shape[1]).to(self.device)
        # Create a DataLoader for batching
        batch_size = batch_size
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        train_losses = []
        val_losses = []
        train_maes = []
        val_maes = []
        # Training loop
        criterion = nn.MSELoss()  # Mean squared error loss
        optimizer = optim.Adam(model.parameters())
    
        num_epochs = n_epochs  # Adjust as needed
        with tqdm(total=num_epochs, desc="ANN model") as pbar:
            for epoch in range(num_epochs):
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()  # Clear gradients
            
                    # Forward pass
                    outputs = model(batch_X)
            
                    # Compute loss
                    loss = criterion(outputs, batch_y)
            
                    # Backpropagation
                    loss.backward()
            
                    # Update weights
                    optimizer.step()

                train_losses.append(loss.item())
                train_mae = torch.mean(torch.abs(outputs - batch_y))
                train_maes.append(train_mae.item())
                
                model.eval()
                with torch.no_grad():
                    outputs = model(X_test)
                    val_loss = criterion(outputs, Y_test)
                    val_losses.append(val_loss.item())
                    val_mae = torch.mean(torch.abs(outputs - Y_test))
                    val_maes.append(val_mae.item())
                pbar.set_postfix({'val loss, val mae':'{0:1.5f}, {0:1.5f}'.format(val_loss,val_mae)})  # 输入一个字典，显示实验指标
                pbar.update(1)

        self.train_losses=train_losses
        self.val_losses=val_losses
        self.train_maes=train_maes
        self.val_maes=val_maes

        X_val=self.adata.obsm[self.basis]
        X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        predicted=model(X_val)
        y_pred=predicted.cpu().detach().numpy().reshape(-1)
        self.adata.obs['p_ANN_time']=y_pred

    def cal_distrubute(self) -> None:
        r"""
        
        calculate the distribution of ANN time and diffusion pseudotime
    Parameters
    ----------
        
        """

        from distfit import distfit
        print('......Dweibull analysis')
        dist1 = distfit(todf=True)
        dist1.fit_transform(self.adata.obs[['dpt_pseudotime']].values)
        loc_dweibull=dist1.summary.loc[dist1.summary['name']=='dweibull','loc'].values[0]
        scale_dweibull=dist1.summary.loc[dist1.summary['name']=='dweibull','scale'].values[0]
        c_dweibull=dist1.summary.loc[dist1.summary['name']=='dweibull','arg'].values[0][0]
        self.dweibull_arg={'c':c_dweibull,'loc':loc_dweibull,'scale':scale_dweibull}
        
        print('......Norm analysis')
        dist2 = distfit(todf=True)
        dist2.fit_transform(self.adata.obs[['p_ANN_time']].values)
        loc_norm=dist2.summary.loc[dist2.summary['name']=='norm','loc'].values[0]
        scale_norm=dist2.summary.loc[dist2.summary['name']=='norm','scale'].values[0]
        self.norm_arg={'loc':loc_norm,'scale':scale_norm}
        self.dist1=dist1
        self.dist2=dist2

    def distribute_fun(self,mode,x1,x2):
        r"""
        calculate the distribution of model
    Parameters
    ----------
            mode:
                Norm: the normal module of scLTNN distribution
                Best_all: the best distribution to use in scLTNN time calcultion
                Best_dweibull: the best distribution of ANN model time to use in  scLTNN time calcultion
            x1: the dpt_pseudotime of adata
            x2: the p_ANN_time of adata
    Returns
    -------
        x: the composition of x1 and x2 by special distribution
        
        """
        if mode=='Norm':
            a1=(norm.pdf(x1,loc=self.norm_arg['loc'],scale=self.norm_arg['scale']))
            a2=(dweibull.pdf(x2,self.dweibull_arg['c'],loc=self.dweibull_arg['loc'],scale=self.dweibull_arg['scale']))
        elif mode=='Best_all':
            a1=(self.dist2.model['name'].pdf(x1,*self.dist2.model['params']))
            a2=(self.dist1.model['name'].pdf(x2,*self.dist1.model['params']))
        elif mode=='Best_dweibull':
            a1=(self.dist2.model['name'].pdf(x1,*self.dist2.model['params']))
            a2=(dweibull.pdf(x2,self.dweibull_arg['c'],loc=self.dweibull_arg['loc'],scale=self.dweibull_arg['scale']))
        
        return x1*(a1/(a1+a2))+x2*(a2/(a1+a2))

    def cal_scLTNN_time(self,mode='Norm'):
        r"""
        calcualte the scLTNN time of anndata
    Parameters
    ----------
            mode:
                'Norm': the normal module of scLTNN distribution
                'Best_all' the best distribution to use in scLTNN time calcultion
                'Best_dweibull': the best distribution of ANN model time to use in  scLTNN time calcultion
        
        """
        print('......calculate scLTNN time')
        new_x=[]
        for i in self.adata.obs.index:
            x1=self.adata.obs.loc[i,'dpt_pseudotime']
            x2=self.adata.obs.loc[i,'p_ANN_time']
            x=self.distribute_fun(mode,x1,x2)
            new_x.append(x)
        self.adata.obs['LTNN_time']=new_x
        self.adata.obs['LTNN_time_r']=1-np.array(new_x)

def find_related_gene(adata):
    r"""
    Find out the gene with postivate relation of the amounts of cells
    Parameters
    ----------
        adata: the anndata performed LTNN analysis
    Returns
    -------
        res_pd: the pandas of correlation with genes and the amounts of cells

    """
    adata_copy=adata.copy()
    from ..pp import filter_cells, filter_genes
    filter_cells(adata_copy, min_genes=200)
    filter_genes(adata_copy, min_cells=3)
    t1=np.array(adata_copy.obs['n_genes'].values)
    if len(adata_copy.var)%5000==0:
        len1=(len(adata_copy.var)//5000)
    else:
        len1=(len(adata_copy.var)//5000)+1
    res_pd=pd.DataFrame(columns=['gene','cor'])
    for i in range(len1):
        if scipy.sparse.issparse(adata.X):
            t2=np.array(adata_copy[:,:].X.toarray().T[5000*(i):5000*(i+1)])
        else:
            t2=np.array(adata_copy[:,:].X.T[5000*(i):5000*(i+1)])
        cor=np.corrcoef(t1,t2)
        cor_pd=pd.DataFrame()
        cor_pd['gene']=adata_copy.var.index[5000*(i):5000*(i+1)]
        cor_pd['cor']=cor[0,1:]
        res_pd=pd.concat([res_pd,cor_pd])
    res_pd=res_pd.sort_values('cor',ascending=False)
    return res_pd



class ANNmodel(object):

    def __init__(self,adata,basis,pseudotime,input_dim,batch_size,cpu):
        device = torch.device(cpu)
        self.model = RadioModel(input_dim).to(device)
        self.adata=adata

        # Assuming you have your X_train, Y_train, X_test, and Y_test tensors
        ran=np.random.choice(adata.obs.index.tolist(),8*(len(adata.obs.index.tolist())//10))
        ran_r=list(set(adata.obs.index.tolist())-set(ran))
        
        X_train=adata[ran].obsm[basis]
        Y_train=adata.obs.loc[ran,pseudotime]
        X_test=adata[ran_r].obsm[basis]
        Y_test=adata.obs.loc[ran_r,pseudotime]
        # Convert X_train and y_train into PyTorch tensors
        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        self.Y_train = torch.tensor(Y_train.values.reshape(len(Y_train.values),1), dtype=torch.float32).to(device)
        
        # Convert x_test and y_test into PyTorch tensors and move to GPU
        self.X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        self.Y_test = torch.tensor(Y_test.values.reshape(len(Y_test.values),1), dtype=torch.float32).to(device)
        
        # Create a DataLoader for batching
        batch_size = batch_size
        self.train_dataset = TensorDataset(self.X_train, self.Y_train)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)


    def train(self,n_epochs):
        train_losses = []
        val_losses = []
        train_maes = []
        val_maes = []
        # Training loop
        criterion = nn.MSELoss()  # Mean squared error loss
        optimizer = optim.Adam(self.model.parameters())

        num_epochs = n_epochs  # Adjust as needed
        for epoch in range(num_epochs):
            for batch_X, batch_y in self.train_loader:
                optimizer.zero_grad()  # Clear gradients
        
                # Forward pass
                outputs = self.model(batch_X)
        
                # Compute loss
                loss = criterion(outputs, batch_y)
        
                # Backpropagation
                loss.backward()
        
                # Update weights
                optimizer.step()

            train_losses.append(loss.item())
            train_mae = torch.mean(torch.abs(outputs - batch_y))
            train_maes.append(train_mae.item())
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(self.X_test)
                val_loss = criterion(outputs, self.Y_test)
                val_losses.append(val_loss.item())
                val_mae = torch.mean(torch.abs(outputs - self.Y_test))
                val_maes.append(val_mae.item())

        self.train_losses=train_losses
        self.val_losses=val_losses
        self.train_maes=train_maes
        self.val_maes=val_maes

    def predicted(self,x):
        return self.model(x)


    def save(self,save_path):
        # Save the entire model (architecture and parameters)
        torch.save(self.model.state_dict(), save_path)

    def load(self,load_path):
        self.model.load_state_dict(torch.load(load_path))
        
class RadioModel(nn.Module):
    def __init__(self, input_dim):
        super(RadioModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        #self.fc3 = nn.Linear(512, 50)
        self.fc4 = nn.Linear(512, 1)
    
    def forward(self, x):

        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=F.relu(x)
        x=self.fc3(x)
        x=F.relu(x)
        x = self.fc4(x)
        return x
    

def find_related_gene(adata):
    r"""
    Find out the gene with postivate relation of the amounts of cells
    Parameters
    ----------
        adata: the anndata performed LTNN analysis
    Returns
    -------
        res_pd: the pandas of correlation with genes and the amounts of cells

    """
    adata_copy=adata.copy()
    from ..pp import filter_cells, filter_genes
    filter_cells(adata_copy, min_genes=200)
    filter_genes(adata_copy, min_cells=3)
    t1=np.array(adata_copy.obs['n_genes'].values)
    if len(adata_copy.var)%5000==0:
        len1=(len(adata_copy.var)//5000)
    else:
        len1=(len(adata_copy.var)//5000)+1
    res_pd=pd.DataFrame(columns=['gene','cor'])
    for i in range(len1):
        if scipy.sparse.issparse(adata.X):
            t2=np.array(adata_copy[:,:].X.toarray().T[5000*(i):5000*(i+1)])
        else:
            t2=np.array(adata_copy[:,:].X.T[5000*(i):5000*(i+1)])
        cor=np.corrcoef(t1,t2)
        cor_pd=pd.DataFrame()
        cor_pd['gene']=adata_copy.var.index[5000*(i):5000*(i+1)]
        cor_pd['cor']=cor[0,1:]
        res_pd=pd.concat([res_pd,cor_pd])
    res_pd=res_pd.sort_values('cor',ascending=False)
    return res_pd


def plot_origin_tesmination(adata,basis,origin,tesmination,figsize=(4,4),**kwargs):
    r"""
    plot the origin and tesmination cell of scRNA-seq
    Parameters
    ----------
        adata: the anndata performed LTNN analysis
        origin: the origin cell list/numpy.nparray
        tesmination: the tesmination cell list/numpy.nparray
    Returns
    -------
        ax: the axex subplot of heatmap
    
    """
    import matplotlib.pyplot as plt
    start_mao=[]
    start_mao_name=[]
    for i in adata.obs['leiden']:
        if i in origin:
            start_mao.append('-1')
            start_mao_name.append('Origin')
        elif i in tesmination:
            start_mao.append('1')
            start_mao_name.append('Tesmination')
        else:
            start_mao.append('0')
            start_mao_name.append('Other')
    adata.obs['mao']=start_mao
    adata.obs['mao']=adata.obs['mao'].astype('category')
    adata.obs['mao_name']=start_mao_name
    adata.obs['mao_name']=adata.obs['mao_name'].astype('category')
    nw=adata.obs['mao_name'].cat.categories
    mao_color={
        'Origin':'#e25d5d',
        'Other':'white',
        'Tesmination':'#a51616'
    }
    adata.uns['mao_name_colors'] = nw.map(mao_color)
    #return adata
    fig,ax=plt.subplots(figsize=figsize)
    from ..pl import embedding
    embedding(adata,basis=basis,show=False,ax=ax,color=['mao_name'],**kwargs)
    #t.plot([0,10],[0,10])

    circle1_loc=adata[adata.obs['mao']=='-1'].obsm[basis].mean(axis=0)
    circle1_max=adata[adata.obs['mao']=='-1'].obsm[basis].max(axis=0)
    circle1_r=circle1_loc[0]-circle1_max[0]
    circle1 = plt.Circle(circle1_loc, circle1_r*1.2, color='#e25d5d',fill=False,ls='--',lw=2)

    circle2_loc=adata[adata.obs['mao']=='1'].obsm[basis].mean(axis=0)
    circle2_max=adata[adata.obs['mao']=='1'].obsm[basis].max(axis=0)
    circle2_r=circle2_loc[0]-circle2_max[0]
    circle2 = plt.Circle(circle2_loc, circle2_r*1.2, color='#a51616',fill=False,ls='--',lw=2)

    ax.add_patch(circle1)
    ax.add_patch(circle2)
    return fig,ax

r"""
Copy from scglue: https://github.com/gao-lab/GLUE/
"""

import pandas as pd
import numpy as np
import anndata as ad
from anndata import AnnData
import scipy.sparse
from sklearn.preprocessing import normalize
from typing import Optional
import sklearn.utils.extmath
from typing import Optional, Union

Array = Union[np.ndarray, scipy.sparse.spmatrix]

def tfidf(X: Array) -> Array:
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    Parameters
    ----------
        X: Input matrix
    Returns
    -------
    X_tfidf: TF-IDF normalized matrix
    
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf

def lsi(
        adata: AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)

    Parameters
    ----------
    adata
        Input dataset
    n_components
        Number of dimensions to use
    use_highly_variable
        Whether to use highly variable features only, stored in
        ``adata.var['highly_variable']``. By default uses them if they
        have been determined beforehand.
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.utils.extmath.randomized_svd`
    """
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0  # Keep deterministic as the default behavior
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = normalize(X, norm="l1")
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi

  

def find_high_correlation_gene(adata,rev=False):

    r"""Calculate the Pearson Correlation between gene and LTNN_time
    Arguments
    ---------
    adata
        the anndata performed LTNN analysis
    rev
        the selection of LTNN_time or LTNN_time_r
    Returns
    -------
    LTNN_time_Pearson
        the pandas of LTNN_time_Pearson 
    adata
        the anndata calculated by find_high_correlation_gene
    """

    """
    # Extract data from count matrix
    """
    if rev==True:
        pd1=adata.obs.loc[:,['LTNN_time_r']]
    else:
        pd1 = adata.obs.loc[:,['LTNN_time']]
    pd2 = pd.DataFrame(adata.X.toarray(),columns = adata.var_names,index = adata.obs_names )

    """
    # Calculate the Pearson Correlation
    """
    from scipy import stats
    LTNN_time_Cor = np.arange(len(adata.var.index),dtype=float)  
    for i in range(len(pd2.columns)):
        res = stats.pearsonr(pd1.to_numpy().flatten(),pd2.iloc[:,i].to_numpy())
        LTNN_time_Cor[i] = float(res[0])

    """
    # Assign Pearson_Correlation to adata
    """
    LTNN_time_Pearson = pd.DataFrame(LTNN_time_Cor,index=pd2.columns)
    adata.var.loc[:,'Pearson_correlation'] = LTNN_time_Pearson.iloc[:,0].to_list()
    """
    # Extract the Pearson Correlation
    """
    LTNN_time_Pearson['feautre'] = LTNN_time_Pearson.index
    LTNN_time_Pearson.columns = ['correlation','feature']
    LTNN_time_Pearson['abs_correlation'] = LTNN_time_Pearson['correlation'].abs()
    LTNN_time_Pearson['sig']='+'
    LTNN_time_Pearson.loc[(LTNN_time_Pearson.correlation<0),'sig'] = '-'
    LTNN_time_Pearson=LTNN_time_Pearson.sort_values('correlation',ascending=False)
    return LTNN_time_Pearson,adata

def find_related_gene(adata):
    r"""Find out the gene with postivate relation of the amounts of cells
    Arguments
    ---------
    adata
        the anndata performed LTNN analysis

    Returns
    -------
    res_pd
        the pandas of correlation with genes and the amounts of cells

    """
    adata_copy=adata.copy()
    from ..pp import filter_cells, filter_genes
    filter_cells(adata_copy, min_genes=200)
    filter_genes(adata_copy, min_cells=3)
    t1=np.array(adata_copy.obs['n_genes'].values)
    if len(adata_copy.var)%5000==0:
        len1=(len(adata_copy.var)//5000)
    else:
        len1=(len(adata_copy.var)//5000)+1
    res_pd=pd.DataFrame(columns=['gene','cor'])
    for i in range(len1):
        if scipy.sparse.issparse(adata.X):
            t2=np.array(adata_copy[:,:].X.toarray().T[5000*(i):5000*(i+1)])
        else:
            t2=np.array(adata_copy[:,:].X.T[5000*(i):5000*(i+1)])
        cor=np.corrcoef(t1,t2)
        cor_pd=pd.DataFrame()
        cor_pd['gene']=adata_copy.var.index[5000*(i):5000*(i+1)]
        cor_pd['cor']=cor[0,1:]
        res_pd=pd.concat([res_pd,cor_pd])
    res_pd=res_pd.sort_values('cor',ascending=False)
    return res_pd
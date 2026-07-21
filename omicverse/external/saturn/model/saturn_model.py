'''
Created on Nov 7, 2022

@author: Yanay Rosen
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi.distributions import ZeroInflatedNegativeBinomial
from sklearn.cluster import KMeans
from scipy.stats import rankdata
import scanpy as sc
import numpy as np


def full_block(in_features, out_features, p_drop=0.1):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.LayerNorm(out_features),
        nn.ReLU(),
        nn.Dropout(p=p_drop),
    )


class SATURNPretrainModel(torch.nn.Module):
    def __init__(self, gene_scores, dropout=0, hidden_dim=128, embed_dim=10, species_to_gene_idx={}, vae=False, random_weights=False, 
                 sorted_batch_labels_names=None, l1_penalty=0.1, pe_sim_penalty=1.0):
        super().__init__()
        
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        self.sorted_batch_labels_names = sorted_batch_labels_names
        if self.sorted_batch_labels_names is not None:
            self.num_batch_labels = len(sorted_batch_labels_names)
        else:
            self.num_batch_labels = 0
        
        self.num_gene_scores = len(gene_scores)
        self.num_species = len(species_to_gene_idx)
        self.species_to_gene_idx = species_to_gene_idx
        self.sorted_species_names = sorted(self.species_to_gene_idx.keys())
        
        self.num_genes = 0
        self.vae = vae
        for k,v in self.species_to_gene_idx.items():
            self.num_genes = max(self.num_genes, v[1])

        self.p_weights = nn.Parameter(gene_scores.float().t().log())
        if random_weights: # for the genes to centroids weights
            nn.init.xavier_uniform_(self.p_weights, gain=nn.init.calculate_gain('relu'))
        self.num_cl = gene_scores.shape[1]
            
        self.cl_layer_norm = nn.LayerNorm(self.num_cl)
        self.expr_filler = nn.Parameter(torch.zeros(self.num_genes), requires_grad=False) # pad exprs with zeros        
        
        if self.vae:
            # Z Encoder
            self.encoder = nn.Sequential(
                    full_block(self.num_cl, hidden_dim, self.dropout),
            )
            self.fc_var = nn.Linear(self.hidden_dim, self.embed_dim)
            self.fc_mu = nn.Linear(self.hidden_dim, self.embed_dim)
            
        else:
            self.encoder = nn.Sequential(
                    full_block(self.num_cl, self.hidden_dim, self.dropout),
                    full_block(self.hidden_dim, self.embed_dim, self.dropout),
            )
            
        # Decoder
        self.px_decoder = nn.Sequential(
            full_block(self.embed_dim + self.num_species + self.num_batch_labels, self.hidden_dim, self.dropout),
        )
        
        self.cl_scale_decoder = full_block(self.hidden_dim, self.num_cl)
        
        self.px_dropout_decoders = nn.ModuleDict({
            species: nn.Sequential(
                nn.Linear(self.hidden_dim, gene_idxs[1] - gene_idxs[0]) 
            ) for species, gene_idxs in species_to_gene_idx.items()}
        )
        
        self.px_rs = nn.ParameterDict({
            species: torch.nn.Parameter(torch.randn(gene_idxs[1] - gene_idxs[0]))
            for species, gene_idxs in species_to_gene_idx.items()}
        )
        
        self.metric_learning_mode = False
        # Gene to Macrogene modifiers
        self.l1_penalty = l1_penalty
        self.pe_sim_penalty = pe_sim_penalty
        
        self.p_weights_embeddings = nn.Sequential(
            full_block(self.num_cl, 256, self.dropout) # This embedding layer will be used in metric learning to encode
                                                       # similarity in the protein embedding space
        )

    def forward(self, inp, species, batch_labels=None):
        batch_size = inp.shape[0]
        
        # Pad the appened expr with 0s to fill all gene nodes
        expr = torch.zeros(batch_size, self.num_genes).to(inp.device)
        filler_idx = self.species_to_gene_idx[species]
        expr[:, filler_idx[0]:filler_idx[1]] = inp
        expr = torch.log(expr + 1)
        
        # concatenate the gene embeds with the expression as the last item in the embed
        expr = expr.unsqueeze(1)
        
        # GNN and cluster weights
        clusters = []
        expr_and_genef = expr 

        x = nn.functional.linear(expr_and_genef.squeeze(), self.p_weights.exp())
        x = self.cl_layer_norm(x)
        x = F.relu(x) # all pos
        x = F.dropout(x, self.dropout)
            
        encoder_input = x.squeeze()
        encoded = self.encoder(encoder_input)
        
        if self.vae:
            # VAE 
            mu = self.fc_mu(encoded)
            log_var = self.fc_var(encoded)
        
            encoded = self.reparameterize(mu, log_var)
        else:
            mu = None
            log_var = None
        
        spec_1h = torch.zeros(batch_size, self.num_species).to(inp.device)
        #spec_idx = np.argmax(np.array(self.sorted_species_names) == species) # Fix for one hot
        spec_idx = 0
        spec_1h[:, spec_idx] = 1.
        
        if self.num_batch_labels > 0:
            # construct the one hot encoding of the batch labels
            # also a categorical covariate
            batch_1h = torch.zeros(batch_size, self.num_batch_labels).to(inp.device)
            batch_idx = np.argmax(np.array(self.sorted_batch_labels_names) == batch_labels)
            batch_1h[:, batch_idx] = 1.
            spec_1h = torch.hstack((spec_1h, batch_1h)) # should already be one hotted
        
        
        
        if encoded.dim() != 2:
            encoded = encoded.unsqueeze(0)
        
        if self.metric_learning_mode:
            # Return Encoding if in metric learning mode (encoder only)
            return encoded
        
        decoded = self.px_decoder(torch.hstack((encoded, spec_1h)))
        
        library = torch.log(inp.sum(1)).unsqueeze(1)
        
        # modfiy
        cl_scale = self.cl_scale_decoder(decoded) # num_cl output
        # index genes for mu
        idx = self.species_to_gene_idx[species]
        
        cl_to_px = nn.functional.linear(cl_scale.unsqueeze(0), self.p_weights.exp().t())[:, :, idx[0]:idx[1]]
        # distribute the means by cluster
        px_scale_decode = nn.Softmax(-1)(cl_to_px.squeeze())
        
        px_drop = self.px_dropout_decoders[species](decoded)
        px_rate =  torch.exp(library) * px_scale_decode
        px_r = torch.exp(self.px_rs[species])        
        
        if self.metric_learning_mode:
            return encoded
        
        return encoder_input, encoded, mu, log_var, px_rate, px_r, px_drop
    
    
    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout):
        '''https://github.com/scverse/scvi-tools/blob/master/scvi/module/_vae.py'''
        return -ZeroInflatedNegativeBinomial(
                                                mu=px_rate, theta=px_r, zi_logits=px_dropout
                                            ).log_prob(x).sum(dim=-1)
    
    def loss_vae(self, inp, mu, log_var, kld_weight, px_rate, px_r, px_drop, weights=None):
        if weights is None:
            recons_loss = torch.sum(self.get_reconstruction_loss(inp, px_rate, px_r, px_drop))
        else:
            recons_loss = torch.sum(self.get_reconstruction_loss(inp, px_rate, px_r, px_drop) * weights) # weight by CT abundancy
        
        loss = recons_loss
        if self.vae:
            kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

            loss = loss + (kld_weight * kld_loss)
            return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
        else:
            return {'loss': loss}
            
    
    def lasso_loss(self, weights):
        # Lasso Loss used to regularize the gene to macrogene weights
        loss = torch.nn.L1Loss(reduction="sum")
        return loss(weights, torch.zeros_like(weights))
    
    
    def gene_weight_ranking_loss(self, weights, embeddings):
        # weights is M x G
        x1 = self.p_weights_embeddings(weights.t())
        # genes x 256
        loss = nn.MSELoss(reduction="sum")
        similarity = torch.nn.CosineSimilarity()
        
        idx1 = torch.randint(low=0, high=x1.shape[0], size=(x1.shape[0],))      
        x2 = x1[idx1, :]
        target = similarity(embeddings, embeddings[idx1, :])
        
        return loss(similarity(x1, x2), target)
        
    
    def loss_ae(self, pred, y):
        loss = F.mse_loss(pred, y) # MSE loss for numerical values
        return loss
    
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    
class SATURNMetricModel(torch.nn.Module):
    def __init__(self, input_dim=2000, dropout=0, hidden_dim=128, embed_dim=10, species_to_gene_idx={}, vae=False):
        super().__init__()
        
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        self.num_species = len(species_to_gene_idx)
        self.species_to_gene_idx = species_to_gene_idx
        self.sorted_species_names = sorted(self.species_to_gene_idx.keys())
        
        self.num_genes = 0
        self.vae = vae
        for k,v in self.species_to_gene_idx.items():
            self.num_genes = max(self.num_genes, v[1])
            
            
        self.cl_layer_norm = nn.LayerNorm(self.input_dim)
        
        if self.vae:
            # Z Encoder
            self.encoder = nn.Sequential(
                    full_block(self.input_dim, hidden_dim, self.dropout),
            )
            self.fc_var = nn.Linear(self.hidden_dim, self.embed_dim)
            self.fc_mu = nn.Linear(self.hidden_dim, self.embed_dim)
            
        else:
            self.encoder = nn.Sequential(
                    full_block(self.input_dim, self.hidden_dim, self.dropout),
                    full_block(self.hidden_dim, self.embed_dim, self.dropout),
            )
        

    def forward(self, inp, species=None):
        batch_size = inp.shape[0]
        
        # input is now the anchor values themselves
        encoded = self.encoder(inp)
        
        if self.vae:
            # VAE 
            mu = self.fc_mu(encoded)
            return mu
        else:
            return encoded
    
    
    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout):
        '''https://github.com/scverse/scvi-tools/blob/master/scvi/module/_vae.py'''
        return -ZeroInflatedNegativeBinomial(
                                                mu=px_rate, theta=px_r, zi_logits=px_dropout
                                            ).log_prob(x).sum(dim=-1)
    
    def loss_vae(self, inp, mu, log_var, kld_weight, px_rate, px_r, px_drop):
        recons_loss = torch.sum(self.get_reconstruction_loss(inp, px_rate, px_r, px_drop))
        if self.vae:
            kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

            loss = recons_loss + kld_weight * kld_loss
            return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
        else:
            return {'loss': recons_loss}
            

    def loss_ae(self, pred, y):
        loss = F.mse_loss(pred, y) # MSE loss for numerical values
        return loss
    
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    

def make_centroids(embeds, species_gene_names, num_centroids=2000, normalize=False, seed=0, score_function="default"):
    print("Making Centroids")
    if normalize:
        row_sums = embeds.sum(axis=1)
        embeds = embeds / row_sums[:, np.newaxis]
    kmeans_obj = KMeans(n_clusters=num_centroids, random_state=seed).fit(embeds)
    # dd is distance frome each gene to centroid
    dd = kmeans_obj.transform(embeds)
    
    if score_function == "default":
        to_scores = default_centroids_scores(dd)
    elif score_function == "one_hot":
        to_scores = one_hot_centroids_scores(dd)
    elif score_function == "smoothed":
        to_scores = smoothed_centroids_score(dd)
    
    species_genes_scores = {}
    for i, gene_species_name in enumerate(species_gene_names):
        species_genes_scores[gene_species_name] = to_scores[i, :]
    return species_genes_scores


def default_centroids_scores(dd):
    """
    Convert KMeans distances to centroids to scores.
    :param dd: distances from gene to centroid.
    """
    ranked = rankdata(dd, axis=1) # rank 1 is close rank 2000 is far

    to_scores = np.log1p(1 / ranked) # log 1 is close log 1/2000 is far

    to_scores = ((to_scores) ** 2)  * 2
    return to_scores


def one_hot_centroids_scores(dd):
    """
    Convert KMeans distances to centroids to scores. All or nothing, so closest centroid has score 1, others have score 0.
    :param dd: distances from gene to centroid.
    """
    ranked = rankdata(dd, axis=1) # rank 1 is close rank 2000 is far
    
    to_scores = (ranked == 1).astype(float) # true, which is rank 1, is highest, everything else is 0
    return to_scores


def smoothed_centroids_score(dd):
    """
    Convert KMeans distances to centroids to scores. Smoothed version of original function, so later ranks have larger values.
    :param dd: distances from gene to centroid.
    """
    ranked = rankdata(dd, axis=1) # rank 1 is close rank 2000 is far
    to_scores = 1 / ranked # 1/1 is highest, 1/2 is higher than before, etc.
    return to_scores


### ABLATION (INPUT IS ORTHOLOG GENES) ###

class OrthologPretrainModel(torch.nn.Module):
    def __init__(self, input_dim, dropout=0, hidden_dim=256, embed_dim=256, species_names=[], vae=False):
        super().__init__()
        
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # self.num_gene_scores = len(gene_scores)
        self.num_species = len(species_names)
        # self.species_to_gene_idx = species_to_gene_idx
        self.sorted_species_names = sorted(species_names)
        
        self.num_genes = 0
        self.vae = vae
        self.num_genes = input_dim
        
        
        if self.vae:
            # Z Encoder
            self.encoder = nn.Sequential(
                    full_block(self.num_genes, hidden_dim, self.dropout),
            )
            self.fc_var = nn.Linear(self.hidden_dim, self.embed_dim)
            self.fc_mu = nn.Linear(self.hidden_dim, self.embed_dim)
            
        else:
            self.encoder = nn.Sequential(
                    full_block(self.num_genes, self.hidden_dim, self.dropout),
                    full_block(self.hidden_dim, self.embed_dim, self.dropout),
            )
            
        # Decoder
        self.px_decoder = nn.Sequential(
            full_block(self.embed_dim + self.num_species, self.hidden_dim, self.dropout),
        )
        
        self.px_scale_decoder = full_block(self.hidden_dim, self.num_genes)
        
        self.px_dropout_decoders =  nn.Sequential(
                nn.Linear(self.hidden_dim, self.num_genes)
        )
        
        self.px_rs = torch.nn.Parameter(torch.randn(self.num_genes))
        

    def forward(self, inp, species):
        batch_size = inp.shape[0]
        
        # Pad the appened expr with 0s to fill all gene nodes
        #expr = torch.zeros(batch_size, self.num_genes).to(inp.device)
        expr = torch.log(inp + 1)
        
        # concatenate the gene embeds with the expression as the last item in the embed
        expr = expr.unsqueeze(1)
        
        # GNN and cluster weights
        clusters = []
        expr_and_genef = expr
            
        encoder_input = expr.squeeze()
        encoded = self.encoder(encoder_input)
        
        if self.vae:
            # VAE 
            mu = self.fc_mu(encoded)
            log_var = self.fc_var(encoded)
        
            encoded = self.reparameterize(mu, log_var)
        else:
            mu = None
            log_var = None
        
        spec_1h = torch.zeros(batch_size, self.num_species).to(inp.device)
        spec_idx = np.argmax(self.sorted_species_names == species)
        spec_1h[:, spec_idx] = 1.
        
        if encoded.dim() != 2:
            encoded = encoded.unsqueeze(0)
        
        decoded = self.px_decoder(torch.hstack((encoded, spec_1h)))
        
        library = torch.log(inp.sum())
        
        # modfiy                
        px_scale = self.px_scale_decoder(decoded)
        px_scale_decode = nn.Softmax(-1)(px_scale.squeeze())
        
        px_drop = self.px_dropout_decoders(decoded)
        px_rate =  torch.exp(library) * px_scale_decode
        px_r = torch.exp(self.px_rs)        
        
        return encoder_input, encoded, mu, log_var, px_rate, px_r, px_drop
    
    
    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout):
        '''https://github.com/scverse/scvi-tools/blob/master/scvi/module/_vae.py'''
        return -ZeroInflatedNegativeBinomial(
                                                mu=px_rate, theta=px_r, zi_logits=px_dropout
                                            ).log_prob(x).sum(dim=-1)
    
    def loss_vae(self, inp, mu, log_var, kld_weight, px_rate, px_r, px_drop):
        recons_loss = torch.sum(self.get_reconstruction_loss(inp, px_rate, px_r, px_drop))
        if self.vae:
            kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

            loss = recons_loss + kld_weight * kld_loss
            return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
        else:
            return {'loss': recons_loss}
            

    def loss_ae(self, pred, y):
        loss = F.mse_loss(pred, y) # MSE loss for numerical values
        return loss
    
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    
    
class OrthologMetricModel(torch.nn.Module):
    def __init__(self, input_dim=2000, dropout=0, hidden_dim=256, embed_dim=256, species_names=[], vae=False):
        super().__init__()
        
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        self.num_species = len(species_names)
        self.sorted_species_names = sorted(species_names)
        
        self.num_genes = 0
        self.vae = vae
        self.num_genes = input_dim
        
        if self.vae:
            # Z Encoder
            self.encoder = nn.Sequential(
                    full_block(self.input_dim, hidden_dim, self.dropout),
            )
            self.fc_var = nn.Linear(self.hidden_dim, self.embed_dim)
            self.fc_mu = nn.Linear(self.hidden_dim, self.embed_dim)
            
        else:
            self.encoder = nn.Sequential(
                    full_block(self.input_dim, self.hidden_dim, self.dropout),
                    full_block(self.hidden_dim, self.embed_dim, self.dropout),
            )
        

    def forward(self, inp, species):
        batch_size = inp.shape[0]
        
        # input is now the anchor values themselves
        encoded = self.encoder(inp)
        
        if self.vae:
            # VAE 
            mu = self.fc_mu(encoded)
            return mu
        else:
            return encoded
    
    
    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout):
        '''https://github.com/scverse/scvi-tools/blob/master/scvi/module/_vae.py'''
        return -ZeroInflatedNegativeBinomial(
                                                mu=px_rate, theta=px_r, zi_logits=px_dropout
                                            ).log_prob(x).sum(dim=-1)
    
    def loss_vae(self, inp, mu, log_var, kld_weight, px_rate, px_r, px_drop):
        recons_loss = torch.sum(self.get_reconstruction_loss(inp, px_rate, px_r, px_drop))
        if self.vae:
            kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

            loss = recons_loss + kld_weight * kld_loss
            return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
        else:
            return {'loss': recons_loss}
            

    def loss_ae(self, pred, y):
        loss = F.mse_loss(pred, y) # MSE loss for numerical values
        return loss
    
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
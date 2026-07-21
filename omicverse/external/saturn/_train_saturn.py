'''
Created on Nov 7, 2022

@author: Yanay Rosen
'''


import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"


import os
import scanpy as sc
import pandas as pd
from anndata import AnnData
import warnings
from builtins import int
warnings.filterwarnings('ignore')
from . import losses, miners, distances, reducers, testers
import torch
import torch.optim as optim
import numpy as np
from .utils import logging_presets as logging_presets
try:
    import record_keeper
except Exception:  # optional dependency, only needed for logging hooks
    record_keeper = None
from .data.gene_embeddings import load_gene_embeddings_adata
from .data.multi_species_data import ExperimentDatasetMulti, multi_species_collate_fn, ExperimentDatasetMultiEqualCT
from .data.multi_species_data import ExperimentDatasetMultiEqual

from .model.saturn_model import SATURNPretrainModel, SATURNMetricModel, make_centroids
import torch.nn.functional as F
from tqdm import trange, tqdm
from .pretrain_utils import *

import argparse
from datetime import datetime
from pathlib import Path
import sys
sys.path.append('../')

# SATURN
from sklearn.cluster import KMeans
from scipy.stats import rankdata
import pickle
from copy import deepcopy
from .utils import stop_conditions
import random


def get_all_scores(*args, **kwargs):
    """Lazy proxy for :func:`score_adata.get_all_scores`.

    ``score_adata`` pulls in heavy optional plotting dependencies
    (plotly, seaborn, networkx). It is only needed when ``score_adatas=True``,
    so we import it on demand to keep ``import`` of this package cheap and robust.
    """
    from .score_adata import get_all_scores as _gas
    return _gas(*args, **kwargs)


def train(model, loss_func, mining_func, device,
                train_loader, optimizer, epoch, mnn, 
          sorted_species_names, use_ref_labels=False, indices_counts={}, equalize_triplets_species=False):
    '''
    Train one epoch for a SATURN model with Metric Learning
    
    Keyword arguments:
    model -- the pretrain model, class is saturnMetricModel
    loss_func -- the loss function, cosine similarity distance
    mining_func -- mining function, triplet margin miner
    device -- the current torch device
    train_loader -- train loader, returns the macrogene values and label categorical codes
    optimizer -- torch optimizer for model
    epoch -- current epoch
    mnn -- use mutual nearest neighbors for metric learning mining
    sorted_species_names -- names of the species that are being aligned
    use_ref_labels -- if metric learning should increase supervision by using shared coarse labels, stored in the ref labels values
    equalize_triplets_species -- if metric learning should mine triples in a balanced manner from each species
    
    '''
    
    model.train()
    torch.autograd.set_detect_anomaly(True)
    for batch_idx, batch_dict in enumerate(train_loader):
        optimizer.zero_grad()
        embs = []
        labs = []
        spec = []
        ref_labs = []
        for species, (data, labels, ref_labels, _) in batch_dict.items():
            if data is None:
                continue
            data, labels, ref_labels = data.to(device), labels.to(device), ref_labels.to(device)
            
            embeddings = model(data, species)
            embeddings = F.normalize(embeddings)
            embs.append(embeddings)
            labs.append(labels)
            ref_labs.append(ref_labels)
            spec.append(np.argmax(np.array(sorted_species_names) == species) * torch.ones_like(labels))
            
        embeddings = torch.cat(embs)
        labels = torch.cat(labs)
        ref_labels = torch.cat(ref_labs)
        
        species = torch.cat(spec)
        if use_ref_labels:
            indices_tuple = mining_func(embeddings, labels, species, mnn=mnn, ref_labels=ref_labels)
        else:
            indices_tuple = mining_func(embeddings, labels, species, mnn=mnn)
            
        indices_mapped = [labels[i] for i in indices_tuple] # map to labels for only the purpose of writing to triplets file
        
        for j in range(len(indices_mapped[0])):
            key = f"{indices_mapped[0][j]},{indices_mapped[1][j]},{indices_mapped[2][j]}"
            indices_counts[key] = indices_counts.get(key, 0) + 1
        loss = loss_func(embeddings, labels, indices_tuple, embs_list=embs)
        
        if equalize_triplets_species:
            species_mapped = [species[i] for i in indices_tuple] # a,p,n species vectors
            a_spec = species_mapped[0]
            p_spec = species_mapped[1]
            a_uq, a_inv, a_ct = torch.unique(a_spec, return_counts=True, return_inverse=True)
            p_uq, p_inv, p_ct = torch.unique(p_spec, return_counts=True, return_inverse=True)
            
            a_prop = a_ct / torch.sum(a_ct) # Proportions of total ie 1/4
            p_prop = p_ct / torch.sum(p_ct) # Proportions of total ie 3/4
            
            a_balance = torch.reciprocal(a_prop) / len(a_prop) # balancing ie * 4 / 1, then divide by num species
            p_balance = torch.reciprocal(p_prop) / len(p_prop) # balancing ie 4 / 3, then divide by num species        
            
            a_bal_inv = a_balance[a_inv]
            p_bal_inv = p_balance[p_inv]
            
            # WEIGHT THE LOSS BY THE NUMBER OF TRIPLETS MINED PER SPECIES
            loss = torch.mul(torch.mul(loss, a_bal_inv), p_bal_inv).mean()                                         
                                                       
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print("Epoch {} Iteration {}: Loss = {}, Number of mined triplets "
                  "= {}".format(epoch, batch_idx, loss,
                                mining_func.num_triplets))

def pretrain_saturn(model, pretrain_loader, optimizer, device, nepochs, 
                       sorted_species_names, balance=False, use_batch_labels=False, embeddings_tensor=None):
    '''
    Pretrain a SATURN model with a conditional autoencoder

    Keyword arguments:
    model -- the pretrain model, class is saturnPretrainModel
    pretrain_loader -- train loader, returns the count values
    optimizer -- torch optimizer for model
    device -- the current torch device
    nepochs -- how many epochs to pretrain for
    sorted_species_names -- names of the species that are being aligned
    balance -- if we should balance the loss by cell label abundancy
    use_batch_labels -- if we add batch labels as a categorical covariate
    embeddings_tensor -- dictionary containing species:protein embeddings
    '''
    
    
    print('Pretraining...')
    model.train();
    
    if balance:
        all_labels = []
        # Count the label frequency
        for batch_idx, batch_dict in enumerate(pretrain_loader):
            for species in sorted_species_names:
                (_, labels, _, _) = batch_dict[species]
                if labels is None:
                    continue
                labels = labels.cpu()
                all_labels.append(labels)
        all_labels = torch.cat(all_labels)
        unique_labels, label_counts = torch.unique(all_labels, return_counts=True)

        label_weights = label_counts / torch.sum(label_counts) # frequencies
        label_weights = 1 / label_weights # inverse frequencies
        
        max_weight = torch.tensor(unique_labels.shape[0])
        
        label_weights = torch.min(max_weight, label_weights) / max_weight # cap the inverse frequency 

        label_weights = label_weights[unique_labels] # make sure it is in the right order
                
    
    pbar = tqdm(np.arange(1, nepochs+1))
    all_ave_losses = {}
    for species in sorted_species_names:
        all_ave_losses[species] = []
    for epoch in pbar:
        model.train();
        epoch_ave_losses = {}
        for species in sorted_species_names:
            epoch_ave_losses[species] = []
        if model.vae:
            kld_weight = get_kld_cycle(epoch - 1, period=50)
        else:
            kld_weight = 0
        epoch_triplet_loss = []

        for batch_idx, batch_dict in enumerate(pretrain_loader):
            optimizer.zero_grad()

            batch_loss = 0
            for species in np.random.choice(sorted_species_names, size=len(sorted_species_names), replace=False):
                (data, labels, refs, batch_labels) = batch_dict[species]
                
                if data is None:
                    continue
                spec_loss = 0
                if use_batch_labels:
                    data, labels, refs, batch_labels = data.to(device), labels.to(device),\
                                                        refs.to(device), batch_labels.to(device)
                    encoder_input, encoded, mus, log_vars, px_rates, px_rs, px_drops = model(data, species, batch_labels)
                else:
                    data, labels, refs = data.to(device), labels.to(device), refs.to(device)
                
                    encoder_input, encoded, mus, log_vars, px_rates, px_rs, px_drops = model(data, species)

                if model.vae:
                    if mus.dim() != 2:
                        mus = mus.unsqueeze(0)
                    if log_vars.dim() != 2:
                        log_vars = log_vars.unsqueeze(0)
                if px_rates.dim() != 2:
                    px_rates = px_rates.unsqueeze(0)
                if px_rs.dim() != 2:
                    px_rs = px_rs.unsqueeze(0)
                if px_drops.dim() != 2:
                    px_drops = px_drops.unsqueeze(0)
                
                gene_weights = model.p_weights.exp()
                if balance:
                    batch_weights = label_weights[labels].to(device)
                    l = model.loss_vae(data, mus, 
                           log_vars, kld_weight, 
                           px_rates, px_rs, px_drops, batch_weights) # This loss also works for non vae loss
                else:
                    l = model.loss_vae(data, mus, 
                           log_vars, kld_weight, 
                           px_rates, px_rs, px_drops) # This loss also works for non vae loss

                spec_loss = l["loss"] / data.shape[0]
                epoch_ave_losses[species].append(float(spec_loss.detach().cpu()))
                batch_loss += spec_loss
            
            
            l1_loss = model.l1_penalty * model.lasso_loss(model.p_weights.exp())
            rank_loss = model.pe_sim_penalty * model.gene_weight_ranking_loss(model.p_weights.exp(), embeddings_tensor)
           
            batch_loss += l1_loss + rank_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        if model.vae:
            loss_string = [f"Epoch {epoch}: KLD weight: {kld_weight}: L1 Loss {l1_loss.detach()} Rank Loss {rank_loss.detach()}"]
        else:
            loss_string = [f"Epoch {epoch}: L1 Loss {l1_loss.detach()} Rank Loss {rank_loss.detach()}"]

        for species in sorted_species_names:
            loss_string += [f"Avg Loss {species}: {round(np.average(epoch_ave_losses[species]))}"]
            
        loss_string = ", ".join(loss_string)

        pbar.set_description(loss_string)
        for species in sorted_species_names:
            all_ave_losses[species].append(np.mean(epoch_ave_losses[species]))
    return model


def get_all_embeddings(dataset, model, device, use_batch_labels=False):
    '''
    Get the embeddings and other metadata for a pretraining model.

    Keyword arguments:
    model -- the pretrain model, class is saturnPretrainModel
    dataset -- count values
    use_batch_labels -- if we add batch labels as a categorical covariate
    '''
    test_loader = torch.utils.data.DataLoader(dataset, collate_fn=multi_species_collate_fn,
                                        batch_size=1024, shuffle=False)

    model.eval()
    embs = []
    macrogenes = []
    labs = []
    spec = []
    refs = []
    if use_batch_labels:
        batch_labs = []
    
    with torch.no_grad():
        for batch_idx, batch_dict in tqdm(enumerate(test_loader), total=len(test_loader)):
            for species, (data, labels, ref_labels, batch_labels) in batch_dict.items():
                if data is None:
                    continue
                if use_batch_labels:
                    data, labels, ref_labels, batch_labels = data.to(device), labels.to(device), \
                                                             ref_labels.to(device), batch_labels.to(device)
                    encoder_inputs, encodeds, mus, log_var, px_rate, px_r, px_drop = model(data, species, batch_labels)
                else:
                    data, labels, ref_labels = data.to(device), labels.to(device), ref_labels.to(device)

                    encoder_inputs, encodeds, mus, log_var, px_rate, px_r, px_drop = model(data, species)
                if model.vae:
                    for mu in mus:
                        embs.append(mu.detach().cpu())
                else:
                    for encoded in encodeds:
                        embs.append(encoded.detach().cpu())
                for encoder_input in encoder_inputs:
                    macrogenes.append(encoder_input.detach().cpu())
                spec += [species] * data.shape[0]
                labs = labs + list(labels.cpu().numpy())
                refs = refs + list(ref_labels.cpu().numpy())
                
                if use_batch_labels:
                    batch_labs = batch_labs + list(batch_labels.cpu().numpy())
    if use_batch_labels:
        return torch.stack(embs).cpu().numpy(), np.array(labs), np.array(spec), torch.stack(macrogenes),\
                                                                np.array(refs), np.array(batch_labs)
    else:
        return torch.stack(embs).cpu().numpy(), np.array(labs), np.array(spec), torch.stack(macrogenes), np.array(refs)

def get_all_embeddings_metric(dataset, model, device, use_batch_labels=False):
    test_loader = torch.utils.data.DataLoader(dataset, collate_fn=multi_species_collate_fn,
                                        batch_size=1024, shuffle=False)
    '''
    Get the embeddings and other metadata for a trained SATURN model.

    Keyword arguments:
    model -- the trained model, class is saturnMetricModel
    dataset -- macrogene values
    use_batch_labels -- if we add batch labels as a categorical covariate
    '''
    model.eval()
    embs = []
    macrogenes = []
    labs = []
    spec = []
    refs = []
    
    if use_batch_labels:
        batch_labs = []
        
    with torch.no_grad():
        for batch_idx, batch_dict in tqdm(enumerate(test_loader), total=len(test_loader)):
            for species, (data, labels, ref_labels, batch_labels) in batch_dict.items():
                if data is None:
                    continue
                if use_batch_labels:
                    data, labels, ref_labels, batch_labels = data.to(device), labels.to(device), \
                                                             ref_labels.to(device), batch_labels.to(device)
                else:
                    data, labels, ref_labels = data.to(device), labels.to(device), ref_labels.to(device)

                encodeds = model(data, species)
               
                for encoded in encodeds:
                    embs.append(encoded.detach().cpu())
                # encoder input is just the data
                for encoder_input in data:
                    macrogenes.append(encoder_input.detach().cpu())
                spec += [species] * data.shape[0]
                labs = labs + list(labels.cpu().numpy())
                refs = refs + list(ref_labels.cpu().numpy())
                if use_batch_labels:
                    batch_labs = batch_labs + list(batch_labels.cpu().numpy())
    if use_batch_labels:
        return torch.stack(embs).cpu().numpy(), np.array(labs), np.array(spec), torch.stack(macrogenes),\
                                                                np.array(refs), np.array(batch_labs)
    else:
        return torch.stack(embs).cpu().numpy(), np.array(labs), np.array(spec), torch.stack(macrogenes), np.array(refs)           
    

def create_output_anndata(train_emb, train_lab, train_species, train_macrogenes, train_ref, celltype_id_map, reftype_id_map, use_batch_labels=False, batchtype_id_map=None, train_batch=None, obs_names=None):
    '''
    Create an AnnData from SATURN results
    '''
    adata = AnnData(train_emb)
    labels = train_lab.squeeze()
    id2cell_type = {v:k for k,v in celltype_id_map.items()}
    adata.obs['labels'] = pd.Categorical([id2cell_type[int(l)]
                                           for l in labels])
    adata.obs['labels2'] = pd.Categorical([l.split('_')[-1]
                                           for l in adata.obs['labels']])
    
    
    ref_labels = train_ref.squeeze()
    id2ref_type = {v:k for k,v in reftype_id_map.items()}
    adata.obs['ref_labels'] = pd.Categorical([id2ref_type[int(l)]
                                           for l in ref_labels])
    
    adata.obs['species'] = pd.Categorical(train_species)
    
    adata.obsm["macrogenes"] = train_macrogenes
    if use_batch_labels:
        batch_labels = train_batch.squeeze()
        id2batch_type = {v:k for k,v in batchtype_id_map.items()}
        adata.obs['batch_labels'] = pd.Categorical([id2batch_type[int(l)]
                                               for l in batch_labels])
    if obs_names is not None:
        adata.obs_names = obs_names
    return adata

def trainer(args):
    '''
    Runs the SATURN pipeline
    '''
    data_df = pd.read_csv(args.in_data, index_col="species")       
    # data_df should have columns for df location
    species_to_path = data_df.to_dict()["path"]
    species_to_adata = {species:sc.read(path) for species,path in species_to_path.items()}
    species_to_embedding_paths = {species:None for species in species_to_path.keys()} # default to use existing paths
    
    if "embedding_path" in data_df.columns:
        species_to_embedding_paths = {species:path for species,path in data_df.to_dict()["embedding_path"].items()}
        
    species_to_drop = []
    if args.tissue_subset is not None:
        for species in species_to_adata.keys():
            ad = species_to_adata[species]
            species_to_adata[species] = ad[ad.obs[args.tissue_column] == args.tissue_subset]
            if species_to_adata[species].X.shape[0] == 0:
                print(f"Dropping {species}")
                species_to_drop.append(species)
    for species in species_to_drop:
        species_to_adata.pop(species)
        species_to_path.pop(species)
    
    if args.in_label_col is None:
        # Assume in_label_col is set as the in_label_col column in the run DF 
        species_to_label = data_df.to_dict()["in_label_col"]
        species_to_label_col = {species:col for species,col in species_to_label.items()}
        
    #
    all_obs_names = []
    # Add species to celltype name
    for species, adata in species_to_adata.items():
        adata_label = args.in_label_col
        if args.in_label_col is None:
            adata_label = species_to_label_col[species]
        species_str = pd.Series([species] * adata.obs.shape[0])
        species_str.index = adata.obs[adata_label].index
        adata.obs["species"] = species_str
        species_specific_celltype = species_str.str.cat(adata.obs[adata_label], sep="_")
        adata.obs["species_type_label"] = species_specific_celltype
        all_obs_names += list(adata.obs_names)
    
    # Create mapping from cell type to ID
    
    unique_cell_types = set()
    for adata in species_to_adata.values():
        unique_cell_types = (unique_cell_types | set(adata.obs["species_type_label"]))
    
    unique_cell_types = sorted(unique_cell_types)
    
    celltype_id_map = {cell_type: index for index, cell_type in enumerate(unique_cell_types)}

    for adata in species_to_adata.values():
        adata.obs["truth_labels"] = pd.Categorical(
            values=[celltype_id_map[cell_type] for cell_type in adata.obs["species_type_label"]]
        )
    
    num_batch_labels = 0
    use_batch_labels = args.non_species_batch_col is not None
    
    
    if args.score_ref_labels:
        score_column = "ref_labels"
    else:
        score_column = "labels2"
        
        
    # If we are using batch labels, add them as a column in our output anndatas and pass them as a categorical covariate to pretraining
    if use_batch_labels:
        
        unique_batch_types = set()
        for adata in species_to_adata.values():
            unique_batch_types = (unique_batch_types | set(adata.obs[args.non_species_batch_col]))

        unique_batch_types = sorted(unique_batch_types)
        batchtype_id_map = {batch_type: index for index, batch_type in enumerate(unique_batch_types)}
        for adata in species_to_adata.values():
            adata.obs["batch_labels"] = pd.Categorical(
                values=[batchtype_id_map[batch_type] for batch_type in adata.obs[args.non_species_batch_col]]
            )
        num_batch_labels = len(unique_batch_types)
        print(f"Using Batch Labels, {num_batch_labels}")
                
    # make the ref labels column categorical and mapped
    unique_ref_types = set()
    for adata in species_to_adata.values():
        unique_ref_types = (unique_ref_types | set(adata.obs[args.ref_label_col]))

    unique_ref_types = sorted(unique_ref_types)

    reftype_id_map = {ref_type: index for index, ref_type in enumerate(unique_ref_types)}

    
    for adata in species_to_adata.values():
        adata.obs["ref_labels"] = pd.Categorical(
            values=[reftype_id_map[ref_type] for ref_type in adata.obs[args.ref_label_col]]
        )
  
    # Load gene embeddings (which also requires filtering data genes to those with embeddings)
    species_to_gene_embeddings = {}
    for species, adata in species_to_adata.items():
        adata, species_gene_embeddings = load_gene_embeddings_adata(
            adata=adata,
            species=[species],
            embedding_model=args.embedding_model,
            embedding_path=species_to_embedding_paths[species]
        )

        species_to_gene_embeddings.update(species_gene_embeddings)
        species_to_adata[species] = adata
        print("After loading the anndata", species, adata)
    
    sorted_species_names = sorted(species_to_gene_embeddings.keys())
    
    # Get highly variable genes and subset the adatas and embeddings
    species_to_gene_idx_hv = {}
    ct = 0
    for species in sorted_species_names:
        adata = species_to_adata[species]
        if use_batch_labels:
            sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=args.hv_genes, \
                                        batch_key=args.non_species_batch_col, span=args.hv_span)  # Expects Count Data
        else:
            sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=args.hv_genes)  # Expects Count Data
        hv_index = adata.var["highly_variable"]
        species_to_adata[species] = adata[:, hv_index]
        species_to_gene_embeddings[species] = species_to_gene_embeddings[species][hv_index]
        species_to_gene_idx_hv[species] = (ct, ct+species_to_gene_embeddings[species].shape[0])
        ct+=species_to_gene_embeddings[species].shape[0]
        
        
    # List of species_ + gene_name for macrogenes
    all_gene_names = []
    for species in sorted_species_names:
        adata = species_to_adata[species]
        species_str = pd.Series([species] * adata.var_names.shape[0])
        gene_names = pd.Series(adata.var_names)
        all_gene_names += list(species_str.str.cat(gene_names, sep="_"))
    
    # stacked embeddings
    X = torch.cat([species_to_gene_embeddings[species] for species in sorted_species_names])
    
    # Create or load the centroid weights
    if args.centroids_init_path is not None:
        # Try to load the centroids, if not, create it.
        try:
            with open(args.centroids_init_path, "rb") as f:
                species_genes_scores = pickle.load(f)
            print("Loaded centroids")
        except:
            # create the centroids and save them to that location. Centroids are the unmodified macrogene weight values.
            # We also save the modified macrogene values to a different file.
            species_genes_scores = make_centroids(X, all_gene_names, args.num_macrogenes, seed=args.seed, score_function=args.centroid_score_func)
            with open(args.centroids_init_path, "wb") as f:
                pickle.dump(species_genes_scores, f, protocol=4) # Save the scores dict
            print(f"Saved centroids to {args.centroids_init_path}")
    else:
        # create the centroids and don't save them
        species_genes_scores = make_centroids(X, all_gene_names, args.num_macrogenes, seed=args.seed, score_function=args.centroid_score_func)
    
    # Initialize macrogenes
    centroid_weights = []
    all_species_gene_names = []
    for species in sorted_species_names:
        adata = species_to_adata[species]
        species_str = pd.Series([species] * adata.var_names.shape[0])
        gene_names = pd.Series(adata.var_names)
        species_gene_names = list(species_str.str.cat(gene_names, sep="_"))
        all_species_gene_names = all_species_gene_names + species_gene_names
        for sgn in species_gene_names:
            centroid_weights.append(torch.tensor(species_genes_scores[sgn]))
    centroid_weights =  torch.stack(centroid_weights)
    
    # Make the train loader
    if use_batch_labels: # we have a batch column to use for the pretrainer
        dataset = ExperimentDatasetMultiEqual(
            all_data = species_to_adata,
            all_ys = {species:adata.obs["truth_labels"] for (species, adata) in species_to_adata.items()},
            all_refs = {species:adata.obs["ref_labels"] for (species, adata) in species_to_adata.items()},
            all_batch_labs = {species:adata.obs["batch_labels"] for (species, adata) in species_to_adata.items()}
        )
        
        test_dataset = ExperimentDatasetMulti(
            all_data = species_to_adata,
            all_ys = {species:adata.obs["truth_labels"] for (species, adata) in species_to_adata.items()},
            all_refs = {species:adata.obs["ref_labels"] for (species, adata) in species_to_adata.items()},
            all_batch_labs = {species:adata.obs["batch_labels"] for (species, adata) in species_to_adata.items()}
        )
    else:
        dataset = ExperimentDatasetMultiEqual(
            all_data = species_to_adata,
            all_ys = {species:adata.obs["truth_labels"] for (species, adata) in species_to_adata.items()},
            all_refs = {species:adata.obs["ref_labels"] for (species, adata) in species_to_adata.items()},
            all_batch_labs = {}
        )
        
        test_dataset = ExperimentDatasetMulti(
            all_data = species_to_adata,
            all_ys = {species:adata.obs["truth_labels"] for (species, adata) in species_to_adata.items()},
            all_refs = {species:adata.obs["ref_labels"] for (species, adata) in species_to_adata.items()},
            all_batch_labs = {}
        )

    hooks = logging_presets.get_hook_container(record_keeper)
    tester = testers.GlobalEmbeddingSpaceTester(
        end_of_testing_hook=hooks.end_of_testing_hook,
        dataloader_num_workers=1,
        use_trunk_output=True)

    device = torch.device(args.device)
    dt = str(datetime.now())[5:19].replace(' ', '_').replace(':', '-')
    
    args.dir_ = args.work_dir + args.log_dir + \
                'test'+str(args.model_dim)+\
                '_data_'
    
    df_names = []
    for species in sorted_species_names:
        p = species_to_path[species]
        df_names += [p.split('/')[-1].split('.h5ad')[0]]
    args.dir_ += '_'.join(df_names) + \
                ('_org_'+ str(args.org) if args.org is not None else '') +\
                ('_'+dt if args.time_stamp else '') +\
                ('_'+args.tissue_subset if args.tissue_subset else '') +\
                ('_seed_'+str(args.seed))

    model_dim = args.model_dim
    hidden_dim = args.hidden_dim    
    
    
    if use_batch_labels:
        sorted_batch_labels_names=list(unique_batch_types)
    else:
        sorted_batch_labels_names = None
    
    
    #### Pretraining ####
    pretrain_model = SATURNPretrainModel(gene_scores=centroid_weights, 
                                   hidden_dim=hidden_dim, embed_dim=model_dim, 
                                   dropout=0.1, species_to_gene_idx=species_to_gene_idx_hv, 
                                   vae=args.vae, sorted_batch_labels_names=sorted_batch_labels_names, 
                                   l1_penalty=args.l1_penalty, pe_sim_penalty=args.pe_sim_penalty).to(device)
    
    
    if args.pretrain:
        optim_pretrain = optim.Adam(pretrain_model.parameters(), lr=args.pretrain_lr)
        pretrain_loader = torch.utils.data.DataLoader(dataset, collate_fn=multi_species_collate_fn,
                                        batch_size=args.pretrain_batch_size, shuffle=True)
        
        pretrain_model = pretrain_saturn(pretrain_model, pretrain_loader, optim_pretrain, 
                                            args.device, args.pretrain_epochs, 
                                            sorted_species_names, balance=args.balance_pretrain, 
                                            use_batch_labels=use_batch_labels, embeddings_tensor=X.to(device))
    
        if args.pretrain_model_path != None:
            # Save the pretraining model if asked to
            print(f"Saved Pretrain Model to {args.pretrain_model_path}")
            torch.save(pretrain_model.state_dict(), args.pretrain_model_path)
            
            
            
    if args.pretrain_model_path != None:
        pretrain_model.load_state_dict(torch.load(args.pretrain_model_path))
        print("Loaded Pretrain Model")
      
    
    
    # write the weights
    final_scores = pretrain_model.p_weights.exp().detach().cpu().numpy().T
    species_genes_scores_final = {}
    for i, gene_species_name in enumerate(all_species_gene_names):
        species_genes_scores_final[gene_species_name] = final_scores[i, :]
    
    metric_dir = Path(args.work_dir) / 'saturn_results'
    metric_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.dir_.split(args.log_dir)[-1]
    
    with open(metric_dir / f'{run_name}_genes_to_macrogenes.pkl', "wb") as f:
        pickle.dump(species_genes_scores_final, f, protocol=4) # Save the scores dict
    
    # create the pretrain adata
    print("Saving Pretrain AnnData")
    if use_batch_labels:
        train_emb, train_lab, train_species, train_macrogenes, train_ref, train_batch = get_all_embeddings(test_dataset, pretrain_model, device, use_batch_labels)
    
        adata = create_output_anndata(train_emb, train_lab, train_species, 
                                        train_macrogenes.cpu().numpy(), train_ref, 
                                        celltype_id_map, reftype_id_map, use_batch_labels, batchtype_id_map, train_batch, obs_names=all_obs_names)  
    else:
        train_emb, train_lab, train_species, train_macrogenes, train_ref = get_all_embeddings(test_dataset, pretrain_model, device, use_batch_labels)
    
        adata = create_output_anndata(train_emb, train_lab, train_species, 
                                        train_macrogenes.cpu().numpy(), train_ref, 
                                        celltype_id_map, reftype_id_map, obs_names=all_obs_names)        
    
    pretrain_adata_path = metric_dir / f'{run_name}_pretrain.h5ad'
    adata.write(pretrain_adata_path)

    # Score the pretraining Dataset
    if args.score_adatas:
        print("*****PRETRAIN SCORES*****")
        get_all_scores(pretrain_adata_path, args.ct_map_path, score_column, 
                       sorted_species_names[0], sorted_species_names[1], num_scores=1)
    print("-----------------------------")
    
    #### Metric Learning ####
    print("***STARTING METRIC LEARNING***")
    if args.unfreeze_macrogenes:
        print("***MACROGENE WEIGHTS UNFROZEN***")
    # Start Metric Learning 
    
    metric_dataset = dataset
    test_metric_dataset = test_dataset
    if (not args.unfreeze_macrogenes):
        # metric_dataset = test_dataset
        
        # Create the new dataset with the macrogenes as input
        metric_dataset.num_genes = {species:train_macrogenes.shape[1] for species in sorted_species_names}
        
        ct = 0
        for species in sorted_species_names:
            species_ct = metric_dataset.xs[species]
            n_cells = species_ct.shape[0]
            species_macrogenes = train_macrogenes[ct:(ct+n_cells), :]
            ct += n_cells
            metric_dataset.xs[species] = species_macrogenes
            
        test_metric_dataset.num_genes = {species:train_macrogenes.shape[1] for species in sorted_species_names}

        ct = 0
        for species in sorted_species_names:
            species_ct = test_metric_dataset.xs[species]
            n_cells = species_ct.shape[0]
            species_macrogenes = train_macrogenes[ct:(ct+n_cells), :]
            ct += n_cells
            test_metric_dataset.xs[species] = species_macrogenes
    
    
    
        pretrain_model = pretrain_model.cpu()

        # metric model will have params copied over, initialize it, only takes
        # macrogene values as inputs since we have frozen them
        metric_model = SATURNMetricModel(input_dim=train_macrogenes.shape[1], 
                                       hidden_dim=hidden_dim, embed_dim=model_dim, 
                                       dropout=0.1, 
                                       species_to_gene_idx=species_to_gene_idx_hv, 
                                       vae=args.vae)
        # Copy over the pretrain model parameters to the metric model
        if pretrain_model.vae:
            metric_model.fc_mu = deepcopy(pretrain_model.fc_mu)
            metric_model.fc_var = deepcopy(pretrain_model.fc_var)
        metric_model.cl_layer_norm = deepcopy(pretrain_model.cl_layer_norm)
        metric_model.encoder = deepcopy(pretrain_model.encoder)
    else:
        metric_model = pretrain_model
        metric_model.metric_learning_mode = True
    
    metric_model.to(device) 
    optimizer = optim.Adam(metric_model.parameters(), lr=args.metric_lr)
    
    #### START METRIC LEARNING ####
    ### pytorch-metric-learning stuff ###
    train_loader = torch.utils.data.DataLoader(metric_dataset, collate_fn=multi_species_collate_fn,
                                        batch_size=args.batch_size, shuffle=True)
    
    
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low = 0)
    
    
    # TripletMarginMMDLoss
    loss_func = losses.TripletMarginLoss(margin = 0.2,
                            distance = distance, reducer = reducer)

    mining_func = miners.TripletMarginMiner(margin = 0.2,
                            distance = distance, type_of_triplets = "semihard",
                            miner_type = "cross_species")
    print("***STARTING METRIC TRAINING***")
    all_indices_counts = pd.DataFrame(columns=["Epoch", "Triplet", "Count"])
    
    scores_df = pd.DataFrame(columns=["epoch", "score", "type"] + list(sorted_species_names))
    batch_size_multiplier = 1
    for epoch in range(1, args.epochs+1):
        epoch_indices_counts = {}
        
               
        train(metric_model, loss_func, mining_func, device,
              train_loader, optimizer, epoch, args.mnn, 
              sorted_species_names, use_ref_labels=args.use_ref_labels, indices_counts=epoch_indices_counts, 
              equalize_triplets_species = args.equalize_triplets_species)
        epoch_df = pd.DataFrame.from_records(list(epoch_indices_counts.items()), columns=["Triplet", "Count"])
        epoch_df["Epoch"] = epoch
        all_indices_counts = pd.concat((all_indices_counts, epoch_df))
        
        if epoch%args.polling_freq==0:
            if use_batch_labels:
                train_emb, train_lab, train_species, train_macrogenes, train_ref, train_batch = get_all_embeddings_metric(\
                    test_dataset, metric_model, device, use_batch_labels)
    
                adata = create_output_anndata(train_emb, train_lab, train_species, 
                                        train_macrogenes.cpu().numpy(), train_ref, 
                                        celltype_id_map, reftype_id_map, use_batch_labels, batchtype_id_map, train_batch, obs_names=all_obs_names)  
            else:
                train_emb, train_lab, train_species, train_macrogenes, train_ref = get_all_embeddings_metric(test_metric_dataset, \
                                                                                     metric_model, device, use_batch_labels)
                adata = create_output_anndata(train_emb, train_lab, train_species, 
                                                    train_macrogenes.cpu().numpy(), train_ref, 
                                                    celltype_id_map, reftype_id_map, obs_names=all_obs_names)
            if args.score_adatas:
                lr_row = stop_conditions.logreg_epoch_score(adata, epoch)
                scores_df = pd.concat((scores_df, pd.DataFrame([lr_row])), ignore_index=True)
            mmd_row = stop_conditions.median_min_distance_score(adata, epoch)
            scores_df = pd.concat((scores_df, pd.DataFrame([mmd_row])), ignore_index=True)
                    
        if epoch%args.polling_freq==0:
            if use_batch_labels:
                train_emb, train_lab, train_species, train_macrogenes, train_ref, train_batch = get_all_embeddings_metric( \
                                                                       test_dataset, metric_model, device, use_batch_labels)
    
                adata = create_output_anndata(train_emb, train_lab, train_species, 
                                        train_macrogenes.cpu().numpy(), train_ref, 
                                        celltype_id_map, reftype_id_map, use_batch_labels, batchtype_id_map, train_batch, obs_names=all_obs_names)  
            else:
                train_emb, train_lab, train_species, train_macrogenes, train_ref = get_all_embeddings_metric(test_metric_dataset, \
                                                                                     metric_model, device, use_batch_labels)
                adata = create_output_anndata(train_emb, train_lab, train_species, 
                                                    train_macrogenes.cpu().numpy(), train_ref, 
                                                    celltype_id_map, reftype_id_map, obs_names=all_obs_names)
            ml_intermediate_path = metric_dir / f'{run_name}_ep_{epoch}.h5ad'
            adata.write(ml_intermediate_path)
            
            if args.score_adatas:
                print(f"***Metric Learning Epoch {epoch} Scores***")
                lr_cross_row = {}
                lr_cross_scores = get_all_scores(ml_intermediate_path, args.ct_map_path, score_column, 
                               sorted_species_names[0], sorted_species_names[1], num_scores=1)
                lr_cross_row["epoch"] = epoch
                lr_cross_row["type"] = "cross_lr"
                lr_cross_row["score"] = lr_cross_scores["species_2_logreg_accuracy"]
                scores_df = pd.concat((scores_df, pd.DataFrame([lr_cross_row])), ignore_index=True)
            
     # Write outputs to file
    print("Saving Final AnnData")
    if use_batch_labels:
        train_emb, train_lab, train_species, train_macrogenes, train_ref, train_batch = get_all_embeddings_metric(\
            test_dataset, metric_model, device, use_batch_labels)

        adata = create_output_anndata(train_emb, train_lab, train_species, 
                                train_macrogenes.cpu().numpy(), train_ref, 
                                celltype_id_map, reftype_id_map, use_batch_labels, batchtype_id_map, train_batch, obs_names=all_obs_names)  
    else:
        train_emb, train_lab, train_species, train_macrogenes, train_ref = get_all_embeddings_metric(test_metric_dataset, \
                                                                             metric_model, device, use_batch_labels)
        adata = create_output_anndata(train_emb, train_lab, train_species, 
                                            train_macrogenes.cpu().numpy(), train_ref, 
                                            celltype_id_map, reftype_id_map, obs_names=all_obs_names)
    final_path = metric_dir / f'{run_name}.h5ad'
    adata.write(final_path)
    
    final_path_triplets = metric_dir / f'{run_name}_triplets.csv'
    all_indices_counts.to_csv(final_path_triplets, index=False)
    
    final_path_epoch_scores = metric_dir / f'{run_name}_epoch_scores.csv'
    scores_df.to_csv(final_path_epoch_scores, index=False)
    
    final_path_ctid = metric_dir / f'{run_name}_celltype_id.pkl'
    with open(final_path_ctid, "wb+") as f:
        pickle.dump(celltype_id_map, f)
    
    if args.score_adatas:
        print(f"***Final Scores***")
        get_all_scores(final_path, args.ct_map_path, score_column, 
                       sorted_species_names[0], sorted_species_names[1], num_scores=1)
    print(f"Final AnnData Path: {final_path}")
    print(f"Final Triplets csv Path: {final_path_triplets}")
    print(f"Final Epoch scores csv Path: {final_path_epoch_scores}")
    print(f"Final celltype_id Path: {final_path_ctid}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Set model hyperparametrs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Run Setup
    parser.add_argument('--in_data', type=str,
                        help='Path to csv containing input datas and species')
    parser.add_argument('--device', type=str,
                    help='Set GPU/CPU')
    parser.add_argument('--device_num', type=int,
                        help='Set GPU Number')
    parser.add_argument('--time_stamp', type=bool,
                        help='Add time stamp in file name')
    parser.add_argument('--org', type=str,
                        help='Add organization to filename')
    parser.add_argument('--log_dir', type=str,
                        help='Log directory')
    parser.add_argument('--work_dir', type=str,
                        help='Working directory')
    parser.add_argument('--seed', type=int,
                        help='Init Seed')
    parser.add_argument('--in_label_col', type=str,
                        help='Label column for input data')
    parser.add_argument('--ref_label_col', type=str,
                        help='Reference label column for input data')
    parser.add_argument('--tissue_subset', type=str,
                        help='Subset the input anndatas by the column args.tissue_column to just be this tissue')
    parser.add_argument('--tissue_column', type=str,
                        help='When subsetting the input anndatas by the column, use this column name.')
    
    # SATURN Setup
    parser.add_argument('--hv_genes', type=int,
                        help='Number of highly variable genes')
    parser.add_argument('--hv_span', type=float,
                        help='Fraction of cells to use when calculating highly variable genes, scanpy defeault is 0.3.')
    parser.add_argument('--num_macrogenes', type=int,
                        help='Number of macrogenes')
    parser.add_argument('--centroids_init_path', type=str,
                        help='Path to existing centroids pretraining weights, or location to save to.')
    parser.add_argument('--embedding_model', type=str, choices=['ESM1b', 'MSA1b', 'protXL', 'ESM1b_protref', 'ESM2'],
                    help='Gene embedding model whose embeddings should be loaded if using gene_embedding_method')
    parser.add_argument('--centroid_score_func', type=str, choices=['default', 'one_hot', 'smoothed'],
                    help='Gene embedding model whose embeddings should be loaded if using gene_embedding_method')
    
    
    # Model Setup
    parser.add_argument('--vae', type=bool, nargs='?', const=True,
                        help='Set the embedding layer to be a VAE.')
    parser.add_argument('--hidden_dim', type=int,
                        help='Model first layer hidden dimension')
    parser.add_argument('--model_dim', type=int,
                        help='Model latent space dimension')
    
    
    # Expression Modifications
    parser.add_argument('--binarize_expression', type=bool, nargs='?', const=True,
                        help='Whether to binarize the gene expression matrix')
    parser.add_argument('--scale_expression', type=bool, nargs='?', const=True,
                        help='Whether to scale the gene expression to zero mean and unit variance')
    
    
    # Pretrain Arguments
    parser.add_argument('--pretrain', type=bool, nargs='?', const=True,
                    help='Pretrain the model')
    parser.add_argument('--pretrain_model_path', type=str,
                        help='Path to save/load a pretraining model to')
    parser.add_argument('--pretrain_lr', type=float,
                        help='Pre training Learning learning rate')    
    parser.add_argument('--pretrain_batch_size', type=int,
                        help='pretrain batch size')
    parser.add_argument('--l1_penalty', type=float,
                        help='L1 Penalty hyperparameter Default is 0.')
    parser.add_argument('--pe_sim_penalty', type=float,
                        help='Protein Embedding similarity to Macrogene loss, weight hyperparameter. Default is 1.0')
    
    # Metric Learning Arguments
    parser.add_argument('--pretrain_epochs', type=int,
                        help='Number of pretraining epochs')
    parser.add_argument('--unfreeze_macrogenes', type=bool, nargs='?', const=True,
                        help='Let Metric Learning Modify macrogene weights')
    parser.add_argument('--mnn', type=bool, nargs='?', const=True, 
                        help='Use mutual nearest neighbors')
    parser.add_argument('--use_ref_labels', type=bool, nargs='?', const=True, 
                    help='Use reference labels when aligning')
    parser.add_argument('--batch_size', type=int,
                        help='batch size')
    parser.add_argument('--metric_lr', type=float,
                        help='Metric Learning learning rate')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs for metric learning')
    
    # Balancing arguments
    parser.add_argument('--balance_pretrain', type=bool, nargs='?', const=True,
                        help='Balance cell types\' weighting in the pretraining model')
    parser.add_argument('--equalize_triplets_species', type=bool, nargs='?', const=True,
                        help='Balance species\' weighting in the metric learning model')
    parser.add_argument('--balance_species_cells', type=bool, nargs='?', const=True,
                        help='Balance species\' number of cells in all steps by resampling randomly.')  
    
    
    # Extra Batch Correction Arguments
    parser.add_argument('--non_species_batch_col', type=str,
                        help='Extra column for batch correction for pretraining. For example, the tissue column in atlas data.')
    
    
    # Scoring argumetns
    parser.add_argument('--polling_freq', type=int,
                        help='Epoch Frequency of scoring during metric learning')
    parser.add_argument('--score_adatas', type=bool, nargs='?', const=True, 
                help='Score the pretraining and metric learning adatas.')
    parser.add_argument('--ct_map_path', type=str,
                        help='Path to csv containing label column mappings between species')
    parser.add_argument('--score_ref_labels', type=bool, nargs='?', const=True,
                        help='Use the ref labels to score instead of the labels.')
    # Defaults
    parser.set_defaults(
        in_data=None,
        org='saturn',
        in_label_col=None,
        ref_label_col="CL_class_coarse",
        non_species_batch_col=None,
        ct_map_path='/dfs/project/cross-species/yanay/fz_true_ct.csv',
        device= torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        device_num=0,
        pretrain_batch_size=4096,
        batch_size=4096,
        model_dim=256,
        hidden_dim=256,
        hv_genes=8000,
        num_macrogenes=2000,
        epochs=50,
        metric_lr=0.001,
        pretrain_epochs=200,
        log_dir='tboard_log/',
        work_dir='./out/',
        time_stamp=False,
        mnn=True,
        pretrain=True,
        vae=False,
        use_ref_labels=False,
        embedding_model='ESM1b',
        gene_embedding_method=None,
        centroids_init_path=None,
        binarize_expression=False,
        scale_expression=False,
        score_adatas=False,
        seed=0,
        pretrain_model_path=None,
        unfreeze_macrogenes=False,
        balance_pretrain=False,
        polling_freq=25,
        equalize_triplets_species=False,
        balance_species_cells=False,
        pretrain_lr=0.0005,
        score_ref_labels=False,
        tissue_subset=None,
        tissue_column="tissue_type",
        l1_penalty=0.0,
        pe_sim_penalty=0.2,
        hv_span=0.3,
        centroid_score_func="default"
    )

    args = parser.parse_args()
    torch.cuda.set_device(args.device_num)
    print(f"Using Device {args.device_num}")
    # Numpy seed
    np.random.seed(args.seed)
    # Torch Seed
    torch.manual_seed(args.seed)
    # Default random seed
    random.seed(args.seed)
    
    print(f"Set seed to {args.seed}")
    if not args.balance_species_cells:
        # Don't Balance the species numbers
        ExperimentDatasetMultiEqual = ExperimentDatasetMulti

    trainer(args)


import math

import numpy as np
import torch

from ..utils import common_functions as c_f


# input must be 2D
def logsumexp(x, keep_mask=None, add_one=True, dim=1):
    if keep_mask is not None:
        x = x.masked_fill(~keep_mask, c_f.neg_inf(x.dtype))
    if add_one:
        zeros = torch.zeros(x.size(dim - 1), dtype=x.dtype, device=x.device).unsqueeze(
            dim
        )
        x = torch.cat([x, zeros], dim=dim)

    output = torch.logsumexp(x, dim=dim, keepdim=True)
    if keep_mask is not None:
        output = output.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)
    return output


def meshgrid_from_sizes(x, y, dim=0):
    a = torch.arange(x.size(dim)).to(x.device)
    b = torch.arange(y.size(dim)).to(y.device)
    return torch.meshgrid(a, b)


def get_all_pairs_indices(labels, ref_labels=None):
    """
    Given a tensor of labels, this will return 4 tensors.
    The first 2 tensors are the indices which form all positive pairs
    The second 2 tensors are the indices which form all negative pairs
    """
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    return a1_idx, p_idx, a2_idx, n_idx


def convert_to_pairs(indices_tuple, labels):
    """
    This returns anchor-positive and anchor-negative indices,
    regardless of what the input indices_tuple is
    Args:
        indices_tuple: tuple of tensors. Each tensor is 1d and specifies indices
                        within a batch
        labels: a tensor which has the label for each element in a batch
    """
    if indices_tuple is None:
        return get_all_pairs_indices(labels)
    elif len(indices_tuple) == 4:
        return indices_tuple
    else:
        a, p, n = indices_tuple
        return a, p, a, n


def convert_to_pos_pairs_with_unique_labels(indices_tuple, labels):
    a, p, _, _ = convert_to_pairs(indices_tuple, labels)
    _, unique_idx = np.unique(labels[a].cpu().numpy(), return_index=True)
    return a[unique_idx], p[unique_idx]


def pos_pairs_from_tuple(indices_tuple):
    return indices_tuple[:2]


def neg_pairs_from_tuple(indices_tuple):
    return indices_tuple[2:]


def get_all_triplets_indices(labels, ref_labels=None):
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
    return torch.where(triplets)
    
# sample triplets, with a weighted distribution if weights is specified.

"""
This function gets a set of triplets based on the species
"""
# sample triplets, with a weighted distribution if weights is specified.
def get_species_triplet_indices_coarse( # returns indices of triplet
    labels, species, embeddings, distance, metadata=None, ref_labels=None, t_per_anchor=None, weights=None, mnn=True
):
    """
    Args: 
        * labels: torch.Tensor, of integer labels
        * species: torch.Tensor, tensor of integers representing species
        * embeddings: torch.Tensor, model embeddings of the data, (1024 (batch size), 256 (embedding_length))
        * distance: CosineSimilarity obj, matrix of distances (CosineSimilarity) between embeddings
        * ref_labels: torch.Tensor, same as labels since ref_labels is None
        * t_per_anchor: int, number of samples per anchor?
        
    """
    
    # obtain intersection between two lists
    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))

    a_idx, p_idx, n_idx = [], [], []
    labels_device = labels.device # q: what are the 'labels'? 
    ref_labels = labels if ref_labels is None else ref_labels
    unique_labels = torch.unique(labels) # around 75-80 unique labels
    unique_species = torch.unique(species)
    
    # in general case, ref_labels is just `labels`. Q: why do we need this var?
    # assume label is a cell type
    for label in unique_labels:
        # Get indices of positive samples for this label.
        
        # torch.where outputs a tuple, so we do [0] to get the actual tensor
        p_inds = torch.where(ref_labels == label)[0] # get indices of this cell type
        p_species = species[p_inds] # the species of each one
        p_metadata = metadata[p_inds.tolist()] if metadata is not None else None # list, all the same value
        
        # cross-species, only use the ones where the coarse_label (if i have access to it, are equal) are the same
        if len(torch.unique(p_species))==1: # if only of one species
            
            # we need to infer positive labels between species
            curr_species = p_species[0]
            curr_coarse_label = p_metadata[0] if p_metadata is not None else None 
            
            ### ADDING EXTRA STEP TO ADD MORE POSITIVE PAIRINGS FROM SAME COARSE_LABEL ###
            # get indices of other cells that have the same coarse_label and species but not the same label
#             extra_coarse_inds = [i for i, coarse_label in enumerate(metadata) 
#                                  if coarse_label == curr_coarse_label and ref_labels[i] != label and species[i] == curr_species]
            
#             n = int(0.15 * len(extra_coarse_inds)) # TODO: this number should be variable
#             extra_coarse_inds_sample = random.sample(extra_coarse_inds, n) 
#             p_inds_list = p_inds.tolist()
            
#             p_inds = p_inds_list + extra_coarse_inds_sample
#             p_inds = torch.LongTensor(p_inds)
#             p_inds = p_inds.to(labels_device)
            
#             print("p_inds: ", p_inds)
            ### ADDING EXTRA STEP TO ADD MORE POSITIVE PAIRINGS FROM SAME COARSE_LABEL ###
            
                        
            # mat is distance from each sample, for each of the `p_inds`
            mat = distance(embeddings, embeddings[p_inds]) # shape: (1024 - batch size, len(p_inds))
        
            sp_inds = torch.where(species == curr_species)[0] # get all indices for this specific species
       
            # do we not care about the distances? 
            # interpretation: for all the samples with the same species, we don't care
            mat[sp_inds, :] = -1
            
            # for each of the p_inds (samples with this label), give me the sample of the OTHER SPECIES that is closest
            closest = torch.max(mat, 0)[1] # index location of each maximum value, shape: (len(p_inds),)
            
            if not mnn:
                p_inds = torch.cat([p_inds, closest])
            else:
                #check whether neighbors agree
                
                # for each of the closest, give me distance to everything else
                # basically checking if the closest index
                # indices in closest are of the OPPOSITE SPECIES
                mat = distance(embeddings, embeddings[closest]) # (1024, len(closest))
                sp_inds = torch.where(species != curr_species)[0] # indices with other species
            
                mat[sp_inds, :] = -1
                other_closest = torch.max(mat, 0)[1] # for each of the closest from this species, which is the closest from the 
                
                closest_mnn = closest[torch.where(ref_labels[other_closest]==label)[0]]
                
                # use p_metadata[0] because all elements are identical since.strip('][').split(', ') they share same label
                p_inds_cross = [] if metadata is not None else closest_mnn.tolist()
#                 print('closest metadata: ', metadata[closest_mnn.tolist()])
                if metadata is not None: 
                    for opp_species_idx in closest_mnn: 
                            # metadata right now is coarse_labels
                            opp_coarse_label = metadata[opp_species_idx]
                            curr_coarse_label = p_metadata[0]

                            # convert strings rep of coarse labels array to list if necessary
                            opp_coarse_label = opp_coarse_label.strip('][').split(', ')
                            curr_coarse_label = curr_coarse_label.strip('][').split(', ')

                            coarse_label_intersection = intersection(opp_coarse_label, curr_coarse_label)

                            if len(coarse_label_intersection) >= 1: # at least one matching coarse label 
                                p_inds_cross.append(opp_species_idx) 
                                  
                p_inds_cross = torch.Tensor(p_inds_cross).to(p_inds.device)
                p_inds = torch.cat([p_inds, p_inds_cross])
                
        if ref_labels is labels:
            a_inds = p_inds
        else:
            a_inds = torch.where(labels == label)[0]
          
        # create negative indices within species 
        n_inds = []
        for curr_species in unique_species: # within species
            n_inds1 = torch.where(ref_labels != label)[0] # get the ones that are of a different label 
            n_inds2 = torch.where(species == curr_species)[0] 
            # intersection
            combined = torch.cat((n_inds1, n_inds2))
            uniques, counts = combined.unique(return_counts=True)
            n_inds.append(uniques[counts > 1])
        n_inds = torch.cat(n_inds)
        
        n_a = len(a_inds)
        n_p = len(p_inds)
        min_required_p = 2 if ref_labels is labels else 1
        if (n_p < min_required_p) or (len(n_inds) < 1):
            continue

        k = n_p if t_per_anchor is None else t_per_anchor
        num_triplets = n_a * k
        p_inds_ = p_inds.expand((n_a, n_p))
        # Remove anchors from list of possible positive samples.
        if ref_labels is labels:
            p_inds_ = p_inds_[~torch.eye(n_a).bool()].view((n_a, n_a - 1))
        # Get indices of indices of k random positive samples for each anchor.
        p_ = torch.randint(0, p_inds_.shape[1], (num_triplets,))
        # Get indices of indices of corresponding anchors.
        a_ = torch.arange(n_a).view(-1, 1).repeat(1, k).view(num_triplets)
        p = p_inds_[a_, p_]
        a = a_inds[a_]
        
        # Get indices of negative samples for this label.
        if weights is not None:
            w = weights[:, n_inds][a]
            non_zero_rows = torch.where(torch.sum(w, dim=1) > 0)[0]
            if len(non_zero_rows) == 0:
                continue
            w = w[non_zero_rows]
            a = a[non_zero_rows]
            p = p[non_zero_rows]
            # Sample the negative indices according to the weights.
            if w.dtype == torch.float16:
                # special case needed due to pytorch cuda bug
                # https://github.com/pytorch/pytorch/issues/19900
                w = w.type(torch.float32)
            n_ = torch.multinomial(w, 1, replacement=True).flatten()
        else:
            # Sample the negative indices uniformly.
            n_ = torch.randint(0, len(n_inds), (num_triplets,))
            
        n = n_inds[n_]
        
        a_idx.append(a) # anchor
        p_idx.append(p) # pos
        n_idx.append(n) # neg
        

    # return results
    if len(a_idx) > 0:
        # concatenate results from each label iteration
        a_idx = torch.cat(a_idx).to(labels_device).long()
        p_idx = torch.cat(p_idx).to(labels_device).long()
        n_idx = torch.cat(n_idx).to(labels_device).long()
        assert len(a_idx) == len(p_idx) == len(n_idx)
        
        # return anchor indices, positive indices, and negative indices
        return a_idx, p_idx, n_idx
    else:
        empty = torch.LongTensor([]).to(labels_device)
        return empty.clone(), empty.clone(), empty.clone()
    
    
def get_species_triplet_indices( # returns indices of triplet
    labels, species, embeddings, distance, metadata=None, ref_labels=None, t_per_anchor=None, weights=None, mnn=True
):
    a_idx, p_idx, n_idx = [], [], []
    labels_device = labels.device
    ref_labels = labels if ref_labels is None else ref_labels
    unique_labels = torch.unique(labels)
    unique_species = torch.unique(species)
    
    for label in unique_labels:
        # Get indices of positive samples for this label.
        p_inds = torch.where(labels == label)[0]
        p_species = species[p_inds]
        if len(torch.unique(p_species))==1: # defaulted to true 
            # we need to infer positive labels between species
            curr_species = p_species[0]
            
            #for other_species in unique_species:
                #if curr_species == other_species:
                    #continue
            mat = distance(embeddings, embeddings[p_inds])
            sp_inds = torch.where(species == curr_species)[0]
            mat[sp_inds, :] = -1
            closest = torch.max(mat,0)[1]

            if not mnn:
                p_inds = torch.cat([p_inds, closest])
            else:
                #check whether neighbors agree
                mat = distance(embeddings, embeddings[closest])
                sp_inds = torch.where(species != curr_species)[0]
                mat[sp_inds, :] = -1
                other_closest = torch.max(mat,0)[1]
                p_inds = torch.cat([p_inds, closest[torch.where(labels[other_closest]==label)[0]]])
                
        a_inds = p_inds
        
        n_inds = []
        for curr_species in unique_species:
            n_inds1 = torch.where(labels != label)[0]
            n_inds2 = torch.where(species == curr_species)[0]
            # intersection
            combined = torch.cat((n_inds1, n_inds2))
            uniques, counts = combined.unique(return_counts=True)
            n_inds.append(uniques[counts > 1])
            
        n_inds = torch.cat(n_inds)
        
        n_a = len(a_inds)
        n_p = len(p_inds)
        
        min_required_p = 2 
        if (n_p < min_required_p) or (len(n_inds) < 1):
            continue

        k = n_p if t_per_anchor is None else t_per_anchor
        num_triplets = n_a * k
        #p_inds_ = p_inds.expand((n_a, n_p))
        # Remove anchors from list of possible positive samples.
        # Yanay new change
        # keep anchors as including other species nn
        # but sample no cross species neighbors
        
        a = []
        p = []
        #print(f"Before: {n_a}, {n_p}")
        for sp in unique_species:
            a_sp = a_inds[torch.where(species[a_inds] == sp)[0]]       
            
            n_a_sp = a_sp.shape[0]
            p_not_sp = p_inds[torch.where(species[p_inds] != sp)[0]]
            n_p_sp = p_not_sp.shape[0]
            #print(f"for species {sp}, {n_a_sp}, {n_p_sp}")
            if n_p_sp > 0:
                k = n_p_sp
                num_triplets_sp = n_a_sp * k    
                a_ = torch.arange(n_a_sp).view(-1, 1).repeat(1, k).view(num_triplets_sp)
                p_not_sp_ = p_not_sp.expand((n_a_sp, n_p_sp))
                p_ = torch.randint(0, n_p_sp, (num_triplets_sp,))
                p.append(p_not_sp_[a_, p_])
                a.append(a_sp[a_])
                #print(f"Adding {a_inds[a_].shape}, {p_inds_[a_, p_].shape}")
                # print(p_inds_[a_, p_].shape)
                
        a = torch.cat(a)
        p = torch.cat(p)
        
        
        # Get indices of negative samples for this label.
        if weights is not None:
            w = weights[:, n_inds][a]
            non_zero_rows = torch.where(torch.sum(w, dim=1) > 0)[0]
            if len(non_zero_rows) == 0:
                continue
            w = w[non_zero_rows]
            a = a[non_zero_rows]
            p = p[non_zero_rows]
            # Sample the negative indices according to the weights.
            if w.dtype == torch.float16:
                # special case needed due to pytorch cuda bug
                # https://github.com/pytorch/pytorch/issues/19900
                w = w.type(torch.float32)
            n_ = torch.multinomial(w, 1, replacement=True).flatten()
        else:
            # Sample the negative indices uniformly.
            # n_ = torch.randint(0, len(n_inds), (num_triplets,)) # with replacement=True original
            # new
            
            n_ = torch.zeros_like(a)
            
            a_l = labels[a]
            p_l = labels[p]
            n_l = labels[n_inds]
            
            n_s = species[n_inds]
            
            unique_a_l = torch.unique(a_l)
            unique_a_p = torch.unique(p_l)
            
            
            
            # NEW IDEA:
            # FOR 3 SPECIES
            # BEFORE: NEGATIVE FROM ANY OTHER LABEL
            # NOW: NEGATIVE FROM ANY OTHER LABEL FROM THE ANCHOR AND POSITIVE'S SPECIES
            
            for al in unique_a_l:
                for pl in unique_a_p:
                    a_s = species[labels == al][0] # should just be the same species
                    p_s = species[labels == pl][0]
                    poss_n_ind = torch.where((n_l != al) \
                                             # Anchor label is not equal to negative label
                                             & (n_l != pl) \
                                             # Positive label is not equal to negative label
                                             & ((a_s == n_s) | (p_s == n_s)) \
                                             # Negative is same species as the pos or anchor
                                             )[0]
                    set_idx = torch.where((a_l == al) & (p_l == pl))[0]
                    choice_size = len(set_idx)
                    if choice_size != 0:
                        n_rand = torch.randint(0, len(poss_n_ind), (choice_size,))
                        n_[set_idx] = n_inds[poss_n_ind[n_rand]]
            
        n = n_            
   
        a_idx.append(a)
        p_idx.append(p)
        n_idx.append(n)
        
    if len(a_idx) > 0:
        a_idx = torch.cat(a_idx).to(labels_device).long()
        p_idx = torch.cat(p_idx).to(labels_device).long()
        n_idx = torch.cat(n_idx).to(labels_device).long()
        assert len(a_idx) == len(p_idx) == len(n_idx)
        return a_idx, p_idx, n_idx
    else:
        empty = torch.LongTensor([]).to(labels_device)
        return empty.clone(), empty.clone(), empty.clone()

def get_species_triplet_indices_local(
    labels, species, embeddings, distance, ref_labels=None, t_per_anchor=None, weights=None, mnn=True):
    
    a_idx, p_idx, n_idx = [], [], []
    labels_device = labels.device
    ref_labels = labels if ref_labels is None else ref_labels
    unique_labels = torch.unique(labels)
    unique_species = torch.unique(species)
    
    k = 1
    
    for label in unique_labels:
        # Get indices of positive samples for this label.
        
        p_inds_all = torch.where(ref_labels == label)[0]
        if p_inds_all.shape[0]==1:
            continue
        p_species = species[p_inds_all]
        
        mat = distance(embeddings[p_inds_all], embeddings[p_inds_all])
        curr_k = k+1 if k<mat.shape[0] else mat.shape[0]
        knn = torch.topk(mat,k=curr_k,dim=1)[1]
        knn = torch.transpose(knn,0,1)[1:,:]
        p_inds = torch.reshape(p_inds_all[knn], (-1,))
        a_inds = p_inds_all.repeat(curr_k-1)
        curr_species = p_species[0]
        
        #print(len(p_inds_all))
        #print(len(a_inds))
            
        k2 = 1
        mat = distance(embeddings, embeddings[p_inds_all])
        sp_inds = torch.where(species == curr_species)[0]
        mat[sp_inds, :] = -1
        closest = torch.topk(mat,k=k2,dim=0)[1]
        closest = torch.reshape(closest, (-1,))
        
        mat = distance(embeddings, embeddings[closest])
        sp_inds = torch.where(species != curr_species)[0]
        mat[sp_inds, :] = -1
        other_closest = torch.max(mat,0)[1]
        cross_species_idx = torch.where(ref_labels[other_closest]==label)[0]
        p_inds = torch.cat([p_inds, closest[cross_species_idx]])
        a_inds2 = p_inds_all.repeat(k2)
        a_inds = torch.cat([a_inds, a_inds2[cross_species_idx]])
        
        #print("***")
        
        #mat = distance(embeddings, embeddings[p_inds_all])
        #sp_inds = torch.where(species == curr_species)[0]
        #mat[sp_inds, :] = -1
        #closest = torch.max(mat,0)[1]
        #mat = distance(embeddings, embeddings[closest])
        #sp_inds = torch.where(species != curr_species)[0]
        #mat[sp_inds, :] = -1
        #other_closest = torch.max(mat,0)[1]
        #cross_species_idx = torch.where(ref_labels[other_closest]==label)[0]
        #p_inds = torch.cat([p_inds, closest[cross_species_idx]])
        #a_inds = torch.cat([a_inds, p_inds_all[cross_species_idx]])
        
        n_a = len(a_inds)
        n_p = len(p_inds)
        min_required_p = 2 if ref_labels is labels else 1
        if (n_p < min_required_p):
            continue
        num_triplets = torch.max(torch.tensor(n_a*n_p), torch.tensor(5000))
        all_inds = torch.randint(0, len(a_inds), (num_triplets,))
        a_inds = a_inds[all_inds]
        p_inds = p_inds[all_inds]
        
        n_inds = []
        for curr_species in unique_species:
            n_inds1 = torch.where(ref_labels != label)[0]
            n_inds2 = torch.where(species == curr_species)[0]
            # intersection
            combined = torch.cat((n_inds1, n_inds2))
            uniques, counts = combined.unique(return_counts=True)
            n_inds.append(uniques[counts > 1])
        n_inds = torch.cat(n_inds)
        
        if (len(n_inds) < 1):
            continue
        
        # Sample the negative indices uniformly
        p = p_inds
        a = a_inds
        n_ = torch.randint(0, len(n_inds), (num_triplets,))
        n = n_inds[n_]
    
        a_idx.append(a)
        p_idx.append(p)
        n_idx.append(n)

    if len(a_idx) > 0:
        a_idx = torch.cat(a_idx).to(labels_device).long()
        p_idx = torch.cat(p_idx).to(labels_device).long()
        n_idx = torch.cat(n_idx).to(labels_device).long()
        assert len(a_idx) == len(p_idx) == len(n_idx)
        return a_idx, p_idx, n_idx
    else:
        empty = torch.LongTensor([]).to(labels_device)
        return empty.clone(), empty.clone(), empty.clone()

# sample triplets, with a weighted distribution if weights is specified.
def get_random_triplet_indices(
    labels, ref_labels=None, t_per_anchor=None, weights=None
):
    a_idx, p_idx, n_idx = [], [], []
    labels_device = labels.device
    ref_labels = labels if ref_labels is None else ref_labels
    unique_labels = torch.unique(labels)
    for label in unique_labels:
        # Get indices of positive samples for this label.
        p_inds = torch.where(ref_labels == label)[0]
        if ref_labels is labels:
            a_inds = p_inds
        else:
            a_inds = torch.where(labels == label)[0]
        n_inds = torch.where(ref_labels != label)[0]
        n_a = len(a_inds)
        n_p = len(p_inds)
        min_required_p = 2 if ref_labels is labels else 1
        if (n_p < min_required_p) or (len(n_inds) < 1):
            continue

        k = n_p if t_per_anchor is None else t_per_anchor
        num_triplets = n_a * k
        p_inds_ = p_inds.expand((n_a, n_p))
        # Remove anchors from list of possible positive samples.
        if ref_labels is labels:
            p_inds_ = p_inds_[~torch.eye(n_a).bool()].view((n_a, n_a - 1))
        # Get indices of indices of k random positive samples for each anchor.
        p_ = torch.randint(0, p_inds_.shape[1], (num_triplets,))
        # Get indices of indices of corresponding anchors.
        a_ = torch.arange(n_a).view(-1, 1).repeat(1, k).view(num_triplets)
        p = p_inds_[a_, p_]
        a = a_inds[a_]

        # Get indices of negative samples for this label.
        if weights is not None:
            w = weights[:, n_inds][a]
            non_zero_rows = torch.where(torch.sum(w, dim=1) > 0)[0]
            if len(non_zero_rows) == 0:
                continue
            w = w[non_zero_rows]
            a = a[non_zero_rows]
            p = p[non_zero_rows]
            # Sample the negative indices according to the weights.
            if w.dtype == torch.float16:
                # special case needed due to pytorch cuda bug
                # https://github.com/pytorch/pytorch/issues/19900
                w = w.type(torch.float32)
            n_ = torch.multinomial(w, 1, replacement=True).flatten()
        else:
            # Sample the negative indices uniformly.
            n_ = torch.randint(0, len(n_inds), (num_triplets,))
        n = n_inds[n_]
        a_idx.append(a)
        p_idx.append(p)
        n_idx.append(n)

    if len(a_idx) > 0:
        a_idx = torch.cat(a_idx).to(labels_device).long()
        p_idx = torch.cat(p_idx).to(labels_device).long()
        n_idx = torch.cat(n_idx).to(labels_device).long()
        assert len(a_idx) == len(p_idx) == len(n_idx)
        return a_idx, p_idx, n_idx
    else:
        empty = torch.LongTensor([]).to(labels_device)
        return empty.clone(), empty.clone(), empty.clone()


def repeat_to_match_size(smaller_set, larger_size, smaller_size):
    num_repeat = math.ceil(float(larger_size) / float(smaller_size))
    return smaller_set.repeat(num_repeat)[:larger_size]


def matched_size_indices(curr_p_idx, curr_n_idx):
    num_pos_pairs = len(curr_p_idx)
    num_neg_pairs = len(curr_n_idx)
    if num_pos_pairs > num_neg_pairs:
        n_idx = repeat_to_match_size(curr_n_idx, num_pos_pairs, num_neg_pairs)
        p_idx = curr_p_idx
    else:
        p_idx = repeat_to_match_size(curr_p_idx, num_neg_pairs, num_pos_pairs)
        n_idx = curr_n_idx
    return p_idx, n_idx


def convert_to_triplets(indices_tuple, labels, t_per_anchor=100):
    """
    This returns anchor-positive-negative triplets
    regardless of what the input indices_tuple is
    """
    if indices_tuple is None:
        if t_per_anchor == "all":
            return get_all_triplets_indices(labels)
        else:
            return get_random_triplet_indices(labels, t_per_anchor=t_per_anchor)
    elif len(indices_tuple) == 3:
        return indices_tuple
    else:
        a_out, p_out, n_out = [], [], []
        a1, p, a2, n = indices_tuple
        empty_output = [torch.tensor([]).to(labels.device)] * 3
        if len(a1) == 0 or len(a2) == 0:
            return empty_output
        for i in range(len(labels)):
            pos_idx = torch.where(a1 == i)[0]
            neg_idx = torch.where(a2 == i)[0]
            if len(pos_idx) > 0 and len(neg_idx) > 0:
                p_idx = p[pos_idx]
                n_idx = n[neg_idx]
                p_idx, n_idx = matched_size_indices(p_idx, n_idx)
                a_idx = torch.ones_like(c_f.longest_list([p_idx, n_idx])) * i
                a_out.append(a_idx)
                p_out.append(p_idx)
                n_out.append(n_idx)
        try:
            return [torch.cat(x, dim=0) for x in [a_out, p_out, n_out]]
        except RuntimeError:
            # assert that the exception was caused by disjoint a1 and a2
            # otherwise something has gone wrong
            assert len(np.intersect1d(a1, a2)) == 0
            return empty_output


def convert_to_weights(indices_tuple, labels, dtype):
    """
    Returns a weight for each batch element, based on
    how many times they appear in indices_tuple.
    """
    weights = torch.zeros_like(labels).type(dtype)
    if indices_tuple is None:
        return weights + 1
    if all(len(x) == 0 for x in indices_tuple):
        return weights + 1
    indices, counts = torch.unique(torch.cat(indices_tuple, dim=0), return_counts=True)
    counts = counts.type(dtype) / torch.sum(counts)
    weights[indices] = counts / torch.max(counts)
    return weights
#!/usr/bin/env python

# ================================
# @auther: Rongbin Zheng
# @email: Rongbin.Zheng@childrens.harvard.edu
# @date: May 2022
# ================================

import os,sys
import pandas as pd
import numpy as np
import json
import pickle as pk
import collections
import scanpy as sc
from scipy.stats.mstats import gmean
from datetime import datetime
from scipy import sparse


"""
estimate metabolite level from single cell, three ways can be used:
1) scFEA, a published software, analyzed cell-wise flux and metabolite balance
2) Compass, a published software, predicted cell-wise reaction flux, uptake, and secretion
3) estimate relative metabolite level using expression of enzymes related to metabolic reactions
for flux result, we can impute metabolite level by summing the positive reaction and substracting the negative reaction
"""

def info(string):
    """
    print information
    """
    today = datetime.today().strftime("%B %d, %Y")
    now = datetime.now().strftime("%H:%M:%S")
    current_time = today + ' ' + now
    print("[{}]: {}".format(current_time, string))



## scFEA-flux
def _scFEA_flux_est_(scFEA_pred, scFEA_info, hmdb_info):
    """
    This function can summarize metabolite relative abundance taking scFEA flux rate result as input

    Params
    ------
    scFEA_pred
        a data frame, rows are cells, columns are metabolite modules in scFEA
    scFEA_info
        a data frame, the metabolite annotation file, can be found in scFEA database in github at https://github.com/changwn/scFEA/blob/master/data/Human_M168_information.symbols.csv
    hmdb_info
        a data frame, summarized metabolite annotation information, here needs metabolite HMDB id and KEGG compund id in the columns named as HMDB_ID and Kegg_ID, respectively
    
    Returns
    ------
    Returns a data frame, the columns are cells, rows are metabolite HMDB id, value indicates the estimated metabolite abundance

    """
    ## the scFEA flux result predicts flux rate in module (reaction)
    ## prepare scFEA in and out module
    met_m_out = pd.Series()
    met_m_in = pd.Series()
    for i, line in scFEA_info.iterrows():
        m_out = line['Compound_OUT_ID'].split('+')
        m_in = line['Compound_IN_ID'].split('+')
        for m in m_out: ## out module
            if m in met_m_out:
                met_m_out[m] += '; '+i
            else:
                met_m_out[m] = i
        for m in m_in: ## in module
            if m in met_m_in:
                met_m_in[m] += '; '+i
            else:
                met_m_in[m] = i
    ## summarized module flux to metabolite level
    met_summ = {}
    ma = list(set(met_m_in.index.tolist()) & set(met_m_out.index.tolist())) ## all modules for iterate
    for m in ma:
        m_out = list(set(met_m_out[m].split('; ')) & set(scFEA_pred.columns.tolist())) ## interect with scFEA result
        m_in = list(set(met_m_in[m].split('; ')) & set(scFEA_pred.columns.tolist()))
        m_level = pd.Series()
        if m_out:
            m_level = scFEA_pred[m_out].T.sum()
        if m_in:
            m_level -= scFEA_pred[m_in].T.sum()
        if len(m_level) != 0:
            met_summ[m] = m_level
    met_summ = pd.DataFrame(met_summ) ## the estimated metabolite matrix, cell x met (kegg_id)
    ## we use hmdb id as accession, so match to that
    kegg_id_hmdb = {}
    for k, h in hmdb_info[['HMDB_ID', 'Kegg_ID']].values.tolist():
        kegg_id_hmdb[k] = h
    ## rename met_summ matrix by hmdb id
    met_summ = met_summ.loc[:,met_summ.columns.isin(kegg_id_hmdb.keys())].T
#     met_summ.columns = [kegg_id_hmdb.get(x) for x in met_summ.columns.tolist()()]
    ## transpose
    indexer = [kegg_id_hmdb.get(x) for x in met_summ.columns.tolist()()]
    columns = met_summ.columns
    met_summ = sparse.csc_matrix(met_summ)
    return(met_summ, indexer, columns)

## scFEA-balance
def _scFEA_balance_est_(scFEA_pred, scFEA_info, hmdb_info):
    """
    scFEA predicts metabolite balance result, usually the rows are cells, columns are metabolite name,
    this function can match the metabolite name to HMDB id which is the unique accession here

    Params
    -----
    scFEA_pred
        a data frame, rows are cells, columns are metabolite names in scFEA
    scFEA_info
        a data frame, the metabolite annotation file, can be found in scFEA database in github at https://github.com/changwn/scFEA/blob/master/data/Human_M168_information.symbols.csv
    hmdb_info
        a data frame, summarized metabolite annotation information, here needs metabolite HMDB id and KEGG compund id in the columns named as HMDB_ID and Kegg_ID, respectively
   
    Return
    -----
    Returns a data frame, columns are cells, rows are metabolite HMDB id

    """
    ## metabolite name to kegg compound id in scFEA info
    met_to_kegg = {}
    for i, line in scFEA_info.iterrows():
        metabolite = line['Compound_OUT_name'].split(' + ')
        kegg_id = line['Compound_OUT_ID'].split('+')
        for x in range(len(kegg_id)):
            met_to_kegg[metabolite[x].strip().upper()] = kegg_id[x]
    ## kegg compound id to HMDB id
    kegg_id_hmdb = {}
    for k, h in hmdb_info[['HMDB_ID', 'Kegg_ID']].values.tolist():
        kegg_id_hmdb[k] = h
    ## rename column name for scFEA pred to kegg id
    scFEA_pred.columns = scFEA_pred.columns.str.upper()
    scFEA_pred = scFEA_pred.loc[:,scFEA_pred.columns.isin(met_to_kegg.keys())]
    scFEA_pred.columns = [met_to_kegg.get(x) for x in scFEA_pred.columns.tolist()]
    ## renamce kegg id to hmdb id
    scFEA_pred = scFEA_pred.loc[:, scFEA_pred.columns.isin(kegg_id_hmdb.keys())].T
#     scFEA_pred.columns = [kegg_id_hmdb.get(x) for x in scFEA_pred.columns.tolist()()]
    ## transpose
    indexer = [kegg_id_hmdb.get(x) for x in scFEA_pred.columns.tolist()()]
    columns = scFEA_pred.columns
    scFEA_pred = sparse.csc_matrix(scFEA_pred)
    return(scFEA_pred)

## Compass-reaction
def _compass_react_est_(compass_pred, compass_react_ann, compass_met_ann, hmdb_info):
    """
    Compass predicts metabolic reaction flux, usually rows are reactions (pos and neg), columns are cells
    this function can estimate metabolite relative abundance based on the reaction happening predicted by Compass

    Params
    -----
    compass_pred
        a data frame, output from Compass software, rows are reactions, columns are cells
    compass_react_ann
        a data frame, containing reaction annotation of Compass used, can be found at https://github.com/YosefLab/Compass/blob/master/compass/Resources/Recon2_export/rxn_md.csv
    compass_met_ann
        a data frame, containing metabolite annotation of Compass used, can be found at https://github.com/YosefLab/Compass/blob/master/compass/Resources/Recon2_export/met_md.csv
    hmdb_info
        a data frame, summarized metabolite annotation information, here needs metabolite HMDB primary id and secondary id in the columns named as HMDB_ID and Secondary_HMDB_ID, respectively
   
    Return
    -----
    Returns a data frame, columns are cells, rows are metabolite HMDB id
    """
    ## Compass reaction predicts flux for reaction happening in positive or negative (reverse)
    ## separate pos and neg flux, the met level could be the difference between pos and neg
    compass_pred_pos = compass_pred.loc[compass_pred.index.str.endswith('_pos'),:]
    compass_pred_pos.index = compass_pred.index.str.replace('_pos', '')
    compass_pred_neg = compass_pred.loc[compass_pred.index.str.endswith('_neg'),:]
    compass_pred_neg.index = compass_pred.index.str.replace('_neg', '')

    ## get in and out met based on reaction formula in compass
    compass_met_re_in = {}
    compass_met_re_out = {}
    for i, line in compass_react_ann.iterrows():
        in_ms = line['rxn_formula'].split(' --> ')[0].split(' + ')
        in_ms = list(set([j.split(' [')[0].split(' * ')[-1].strip() for j in in_ms]))
        out_ms = line['rxn_formula'].split(' --> ')[-1].replace('\nNo genes', '').split(' + ')
        out_ms = list(set([j.split(' [')[0].split(' * ')[-1].strip() for j in out_ms]))
        for m in in_ms:
            if m in compass_met_re_in:
                compass_met_re_in[m] += '; '+i
            else:
                compass_met_re_in[m] = i
        for m in out_ms:
            if m in compass_met_re_out:
                compass_met_re_out[m] += '; '+i
            else:
                compass_met_re_out[m] = i
    ## estimate met level from reaction
    ## four situiations for a met in reaction direction
    ## cases: in-pos, in-neg, out-pos, out-neg
    ## wights: -1, +1, +1, -1
    met_all = list(set(list(compass_met_re_in.keys())+list(compass_met_re_out.keys())))
    met_from_react = {}
    for m in met_all:
        ## reaction of in met
        m_in_r = compass_met_re_in[m].split('; ') if m in compass_met_re_in else []
        ## positive flux for in met
        m_in_r_pos = list(set(m_in_r) & set(compass_pred_pos.index.tolist()))
        m_in_r_neg = list(set(m_in_r) & set(compass_pred_neg.index.tolist()))
        ## reaction of out met
        m_out_r = compass_met_re_out[m].split('; ') if m in compass_met_re_out else []
        ## positive flux of out met
        m_out_r_pos = list(set(m_out_r) & set(compass_pred_pos.index.tolist()))
        m_out_r_neg = list(set(m_out_r) & set(compass_pred_neg.index.tolist()))
        ##
        met_level = pd.Series()
        if m_out_r_pos:
            met_level = compass_pred_pos.loc[m_out_r_pos,:].sum() ## +1 for out-pos
        if m_in_r_neg:
            met_level += compass_pred_neg.loc[m_in_r_neg,:].sum() ## +1 for in-neg
        if m_in_r_pos:
            met_level -= compass_pred_pos.loc[m_in_r_pos,:].sum() ## -1 for in-pos
        if m_out_r_neg:
            met_level -= compass_pred_neg.loc[m_out_r_neg,:].sum() ## -1 for out-neg
        if len(met_level) != 0:
            met_from_react[m] = met_level
        met_from_react = pd.DataFrame(met_from_react) ## the estimated met matrix, rows are cells, columns are met
    ## rename metabolite name to HMDB id
    compass_met_ann = pd.merge(compass_met_ann, hmdb_info, left_on = 'hmdbID', right_on = 'Secondary_HMDB_ID', how = 'left')
    met_to_hmdbid = {}
    for i, line in compass_met_ann[['metName', 'HMDB_ID']].dropna().iterrows():
        met, hmdbid = line['metName'], line['HMDB_ID']
        met_to_hmdbid[met] = hmdbid
    ## rename
    met_from_react = met_from_react.loc[:, met_from_react.columns.isin(met_to_hmdbid.keys())].T
#     met_from_react.columns = [met_to_hmdbid.get(x) for x in met_from_react.columns.tolist()]
    ## transpose
    indexer = [met_to_hmdbid.get(x) for x in met_from_react.columns.tolist()]
    columns = met_from_react.columns
    met_from_react = sparse.csc_matrix(met_from_react)
    return(met_from_react)

## compass-uptake or compass-secretion
def _compass_uptake_secrete_est_(compass_pred, compass_met_ann, hmdb_info):
    """
    Compass software predicts uptake and secretion flux as well as reaction, such estimation can be directly used
    usually, it is a matrix that rows are metabolites, columns are cells
    
    Params
    -----
    compass_pred
        a data frame, output from Compass software (uptake or secretion), rows are metabolite, columns are cells
    compass_met_ann
        a data frame, containing metabolite annotation of Compass used, can be found at https://github.com/YosefLab/Compass/blob/master/compass/Resources/Recon2_export/met_md.csv
    hmdb_info
        a data frame, summarized metabolite annotation information, here needs compass metabolite id and HMDB id in the columns named as met and HMDB_ID, respectively

    Return
    -----
    Returns a data frame, columns are cells, rows are metabolite HMDB id
    """ 
    ## rename metabolite name to HMDB id, compass_pred columns are metId in compass
    compass_met_ann = pd.merge(compass_met_ann, hmdb_info, left_on = 'hmdbID', right_on = 'Secondary_HMDB_ID', how = 'left')
    met_to_hmdbid = {}
    for i, line in compass_met_ann[['met', 'HMDB_ID']].dropna().iterrows():
        met, hmdbid = line['met'], line['HMDB_ID']
        met_to_hmdbid[met] = hmdbid
    ## rename
    compass_pred = compass_pred.loc[compass_pred.index.isin(met_to_hmdbid.keys()),:].T
#     compass_pred.index = [met_to_hmdbid.get(x) for x in compass_pred.index.tolist()]
    ## transpose
    indexer = [met_to_hmdbid.get(x) for x in compass_pred.index.tolist()]
    columns = compass_pred.columns
    compass_pred = sparse.csc_matrix(compass_pred) ## rows are met, columns are cells
    return(compass_pred)

# estimate from enzyme gene expression
def _met_est_input_dataframe_(exp_mat, mIdList, met_gene, method):
    """
    expression matrix is a data frame
    """
    met_from_gene = pd.DataFrame()
    with_exp_gene_m = [] ## only met-enzyme gene in the matrix can be estimated
    for mId in mIdList:
        ## genes for the reaction of producing the metabolite
        gene_pos = met_gene[(met_gene['HMDB_ID'] == mId) & (met_gene['direction'] == 'product')]['gene_name'].tolist()
        gene_pos = list(set(gene_pos) & set(exp_mat.index.tolist()))
        ## genes for the reaction of taking the metabolite as substrate
        gene_neg = met_gene[(met_gene['HMDB_ID'] == mId) & (met_gene['direction'] == 'substrate')]['gene_name'].tolist()
        gene_neg = list(set(gene_neg) & set(exp_mat.index.tolist())) if gene_neg else []
        ## estimate by aggerating gene_pos and gene_neg
        ## only estimate when there are genes of positive reactons
        if len(gene_pos) != 0:
            with_exp_gene_m.append(mId)
            ## if neg genes, do subraction from pos genes
            if not gene_neg:
                if method == 'mean':
                    m_from_enzyme = exp_mat.loc[exp_mat.index.isin(gene_pos),:].mean()
                elif method == 'median':
                    m_from_enzyme = exp_mat.loc[exp_mat.index.isin(gene_pos),:].median()
                elif method == 'max':
                    m_from_enzyme = exp_mat.loc[exp_mat.index.isin(gene_pos),:].max()
                elif method == 'gmean':
                    m_from_enzyme = exp_mat.loc[exp_mat.index.isin(gene_pos),:].apply(lambda col: gmean(col))
                else:
                    raise KeyError('method error')
            else:
                if method == 'mean':
                    pos = exp_mat.loc[exp_mat.index.isin(gene_pos),:].mean()
                    neg = exp_mat.loc[exp_mat.index.isin(gene_neg),:].mean()
                elif method == 'median':
                    pos = exp_mat.loc[exp_mat.index.isin(gene_pos),:].median()
                    neg = exp_mat.loc[exp_mat.index.isin(gene_neg),:].median()
                elif method == 'max':
                    pos = exp_mat.loc[exp_mat.index.isin(gene_pos),:].max()
                    neg = exp_mat.loc[exp_mat.index.isin(gene_neg),:].max()
                elif method == 'gmean':
                    pos = exp_mat.loc[exp_mat.index.isin(gene_pos),:].apply(lambda col: gmean(col))
                    neg = exp_mat.loc[exp_mat.index.isin(gene_neg),:].apply(lambda col: gmean(col))
                else:
                    raise KeyError('method error')
                m_from_enzyme = pos - neg
            met_from_gene = pd.concat([met_from_gene,
                                             m_from_enzyme], axis = 1)
    met_from_gene.columns = with_exp_gene_m
    ## tranpose
    met_from_gene = met_from_gene.T ## rows = metabolite (HMDB ID), columns = cells
    return(met_from_gene)

def met_est_input_adata_(adata, mIdList, met_gene, method):
    """
    expression data in scanpy adata object
    """
    ## check the adata object
    ngene = len(adata.var_names)
    ncell = len(adata.obs_names)
#     info('We are receiving expression data with {n1} genes and {n2} cells.'.format(n1 = ngene, n2 = ncell))
#     if ngene < 10000:
#         info('scanpy object contains less than 10000 genes, please make sure you are using raw.to_adata()')
    ## estimating for each met
    met_from_gene = pd.DataFrame()
    with_exp_gene_m = []
    for mId in mIdList:
        ## genes for the reaction of producing the metabolite
        gene_pos = met_gene[(met_gene['HMDB_ID'] == mId) & (met_gene['direction'] == 'product')]['gene_name'].tolist()
        gene_pos = list(set(gene_pos) &
                       set(adata.var_names.tolist()))
        ## genes for the reaction of taking the metabolite as substrate
        gene_neg = met_gene[(met_gene['HMDB_ID'] == mId) & (met_gene['direction'] == 'substrate')]['gene_name'].tolist()
        gene_neg = list(set(gene_neg) & set(adata.var_names.tolist())) if gene_neg else []
        ## estimate by aggerating gene_pos and gene_neg
        ## only estimate when there are genes of positive reactons
        if len(gene_pos) != 0:
            with_exp_gene_m.append(mId)
            ## gene pos matrix
            pos_g_index = np.where(adata.var_names.isin(gene_pos))
            pos_exp = pd.DataFrame(adata.T[pos_g_index].X.toarray(), 
                                index = adata.var_names[pos_g_index].tolist(),
                                columns = adata.obs_names.tolist())

            ## if neg genes, do subraction from pos genes
            if not gene_neg:
                if method == 'mean':
                    m_from_enzyme = pos_exp.mean()
                elif method == 'median':
                    m_from_enzyme = pos_exp.median()
                elif method == 'max':
                    m_from_enzyme = pos_exp.max()
                elif method == 'gmean':
                    m_from_enzyme = pos_exp.apply(lambda col: gmean(col))
                else:
                    continue
            else:
                ## gene neg matrix
                neg_g_index = np.where(adata.var_names.isin(gene_neg))
                neg_exp = pd.DataFrame(adata.T[neg_g_index].X.toarray(), 
                                index = adata.var_names[neg_g_index].tolist(),
                                columns = adata.obs_names.tolist())
                if method == 'mean':
                    pos = pos_exp.mean()
                    neg = neg_exp.mean()
                elif method == 'median':
                    pos = pos_exp.median()
                    neg = neg_exp.median()
                elif method == 'max':
                    pos = pos_exp.max()
                    neg = neg_exp.max()
                elif method == 'gmean':
                    pos = pos_exp.apply(lambda col: gmean(col))
                    neg = neg_exp.apply(lambda col: gmean(col), axis = 1)
                else:
                    raise ValueError('method should be one of [mean, gmean, median, max]')
                m_from_enzyme = pos - neg
            met_from_gene = pd.concat([met_from_gene,
                                             m_from_enzyme], axis = 1)
    met_from_gene.columns = with_exp_gene_m
    ## tranpose
    met_from_gene = met_from_gene.T ## rows = metabolite (HMDB ID), columns = cells
    return(met_from_gene)


def _met_from_enzyme_dataframe_adata_(exp_mat=None, adata=None, mIdList=[], met_gene=pd.DataFrame, method = 'mean'):
    """
    This function takes expression of metabolic reaction related genes (enzyme) as input, and estimate the relative abundance of metabolite in cells,
    the idea is that the metabolite accumulation can be a reaction happening balance between the one of taking the metabolite as substrates and the one of producing it.
    
    Params
    -----
    exp_mat
        a data frame, single cell expression matrix, rows are genes, columns are cells
        'exp_mat' is exclusive parameter to 'adata'
    adata
        a scanpy adata object, the expression will be extracted from the adata to estimate metabolite level
        'adata' is exclusive parameter to 'exp_mat'
    mIdList
        a list of the HMDB ID, the given HMDB ID will be estimated if there are associated genes available, such as:
        ['HMDB0000017', 'HMDB0000026', 'HMDB0000033', ...]
    met_gene
        a data frame, containing the curated metabolite related genes, the data frame at least includes three columns representing HMDB ID, gene, and direction (product or substrate)
        for example:
        HMDB0003944 ACOX1[Unknown]  product
        HMDB0003944 ACOX3[Unknown]  product
        HMDB0003944 ACADSB[Unknown] product
        HMDB0003944 MECR[Unknown]   substrate
        HMDB0006529 PECR[Enzyme]    substrate
        ....
    method
        the way to aggerate expression of metabolite related multiple genes or enzymes, should be one of [mean, gmean, median, max],
        mean for taking arithmetic mean, gmean for taking geomatrix mean, median for taking median value, max for taking maximum value across the genes.
        By default, we set mean which is arithmetic mean.


    Return
    -----
    Returns a data frame, columns are cells, rows are metabolite HMDB id

    """
    ## metabolite related genes to data frame
    met_gene_new = []
    for i, line in met_gene.iterrows():
        genes = line['gene'].split('; ')
        for g in genes:
            tmp = line.copy()
            tmp['gene'] = g
            met_gene_new.append(tmp)
    met_gene = pd.DataFrame(met_gene_new) ## each row is the related gene annotation for metabolite
    met_gene['gene_name'] = met_gene['gene'].apply(lambda x: x.split('[')[0])
    if len(mIdList) == 0:
        mIdList = met_gene['HMDB_ID'].unique().tolist()
    ## check input and load right estimator
    if (type(exp_mat) == type(pd.DataFrame())) and (type(adata) == type(None)):
        info('Receiving input from a data frame')
        met_from_gene = _met_est_input_dataframe_(exp_mat = exp_mat, mIdList = mIdList, met_gene = met_gene, method = method)
        
    elif (type(exp_mat) == type(None)) and (type(adata) != type(None)):
        info('Receiving input from a scanpy adata object')
        try:
            cells = adata.obs_names
            genes = adata.var_names
        except:
            raise ValueError('adata prolem! cannot extract adata.obs_names or adata.var_names, may be it is not a scanpy object.')
        met_from_gene = met_est_input_adata_(adata = adata, mIdList = mIdList, met_gene = met_gene, method = method)
    else:
        raise ValueError('please either provide expression matrix through exp_mat or adata!')

    return(met_from_gene)

def _met_est_input_sparse_(exp_mat, indexer, columns, mIdList, met_gene, method):
    """
    expression matrix is a data frame
    """
    met_from_gene = np.empty(shape=(0,len(columns)))
    with_exp_gene_m = [] ## only met-enzyme gene in the matrix can be estimated
    for mId in mIdList:
        ## genes for the reaction of producing the metabolite
        gene_pos = met_gene[(met_gene['HMDB_ID'] == mId) & (met_gene['direction'] == 'product')]['gene_name'].tolist()
        gene_pos_loc = [i for i, g in enumerate(indexer) if g in gene_pos]
        ## genes for the reaction of taking the metabolite as substrate
        gene_neg = met_gene[(met_gene['HMDB_ID'] == mId) & (met_gene['direction'] == 'substrate')]['gene_name'].tolist()
        gene_neg_loc = [i for i, g in enumerate(indexer) if g in gene_neg] if gene_neg else []
        ## estimate by aggerating gene_pos and gene_neg
        ## only estimate when there are genes of positive reactons
        if len(gene_pos_loc) != 0:
            with_exp_gene_m.append(mId)
            ## if neg genes, do subraction from pos genes
            if not gene_neg_loc:
                if method == 'mean':
                    m_from_enzyme = exp_mat[gene_pos_loc].mean(axis = 0)
                elif method == 'gmean':
                    dense_matrix = exp_mat[gene_pos_loc].toarray()
                    m_from_enzyme = np.array([np.prod(dense_matrix, axis=0) ** (1.0 / dense_matrix.shape[0])])
                else:
                    raise KeyError('method error')
            else:
                if method == 'mean':
                    pos = exp_mat[gene_pos_loc].mean(axis = 0)
                    neg = exp_mat[gene_neg_loc].mean(axis = 0)
                elif method == 'gmean':
                    pos_matrix = exp_mat[gene_pos_loc].toarray()
                    pos = np.array([np.prod(pos_matrix, axis=0) ** (1.0 / pos_matrix.shape[0])])
                    neg_matrix = exp_mat[gene_neg_loc].toarray()
                    neg = np.array([np.prod(neg_matrix, axis=0) ** (1.0 / neg_matrix.shape[0])])
                else:
                    raise KeyError('method error')
                m_from_enzyme = pos - neg
            met_from_gene = np.concatenate((met_from_gene, m_from_enzyme), axis = 0)

    met_from_gene = sparse.csc_matrix(met_from_gene)
    met_indexer = with_exp_gene_m
    met_columns = columns.copy()
    return(met_from_gene, met_indexer, met_columns)


def _react_agg_enzyme_(exp_mat, react, method1):
    genes = react['gene_name'].tolist()
    gene_loc = [i for i, g in enumerate(indexer) if g in genes]
    if method1 == 'mean':
        agg_exp = exp_mat[gene_loc].mean(axis = 0)
    elif method1 == 'gmean':
        dense_matrix = exp_mat[gene_loc].toarray()
        agg_exp = np.array([np.prod(dense_matrix, axis=0) ** (1.0 / dense_matrix.shape[0])])
    else:
        raise KeyError('method1 error')
    return(agg_exp)

def _met_est_input_sparse_by_reaction_(exp_mat, indexer, columns, mIdList, met_gene, method1='gmean', method2 = 'mean'):
    """
    expression matrix is a data frame
    """
    met_from_gene = np.empty(shape=(0,len(columns)))
    with_exp_gene_m = [] ## only met-enzyme gene in the matrix can be estimated
    reaction_exp = met_gene.groupby(['HMDB_ID', 'reaction', 'direction']).apply(lambda react: _react_agg_enzyme_(exp_mat, react, method1 = method1))
    for mId in mIdList:
        ## genes for the reaction of producing the metabolite
        gene_pos = reaction_exp[(reaction_exp['HMDB_ID'] == mId) & (reaction_exp['direction'] == 'product')]
        ## genes for the reaction of taking the metabolite as substrate
        gene_neg = met_gene[(met_gene['HMDB_ID'] == mId) & (met_gene['direction'] == 'substrate')]
        ## estimate by aggerating gene_pos and gene_neg
        ## only estimate when there are genes of positive reactons
        if gene_pos.shape[0] != 0:
            with_exp_gene_m.append(mId)
            ## if neg genes, do subraction from pos genes
            if gene_neg.shape[0] == 0:
                if method2 == 'mean':
                    m_from_enzyme = np.array(gene_pos['react_value']).mean(axis = 0)
                elif method2 == 'gmean':
                    m_from_enzyme = np.prod(np.array(gene_pos['react_value']), axis = 0) ** (1 / gene_pos.shape[0])
                else:
                    raise KeyError('method error')
            else:
                if method2 == 'mean':
                    pos = np.array(gene_pos['react_value']).mean(axis = 0)
                    neg = np.array(gene_neg['react_value']).mean(axis = 0)
                elif method2 == 'gmean':
                    pos = np.prod(np.array(gene_pos['react_value']), axis = 0) ** (1 / gene_pos.shape[0])
                    neg = np.prod(np.array(gene_neg['react_value']), axis = 0) ** (1 / gene_pos.shape[0])
                else:
                    raise KeyError('method error')
                m_from_enzyme = pos - neg
            met_from_gene = np.concatenate((met_from_gene, m_from_enzyme), axis = 0)

    met_from_gene = sparse.csc_matrix(met_from_gene)
    met_indexer = with_exp_gene_m
    met_columns = columns.copy()
    return(met_from_gene, met_indexer, met_columns)


def _met_from_enzyme_est_(exp_mat, indexer, columns, mIdList=[], met_gene=pd.DataFrame, method = 'mean'):
    """
    This function takes expression of metabolic reaction related genes (enzyme) as input, and estimate the relative abundance of metabolite in cells,
    the idea is that the metabolite accumulation can be a reaction happening balance between the one of taking the metabolite as substrates and the one of producing it.
    
    Params
    -----
    exp_mat
        a data frame, single cell expression matrix, rows are genes, columns are cells
        'exp_mat' is exclusive parameter to 'adata'
    adata
        a scanpy adata object, the expression will be extracted from the adata to estimate metabolite level
        'adata' is exclusive parameter to 'exp_mat'
    mIdList
        a list of the HMDB ID, the given HMDB ID will be estimated if there are associated genes available, such as:
        ['HMDB0000017', 'HMDB0000026', 'HMDB0000033', ...]
    met_gene
        a data frame, containing the curated metabolite related genes, the data frame at least includes three columns representing HMDB ID, gene, and direction (product or substrate)
        for example:
        HMDB0003944 ACOX1[Unknown]  product
        HMDB0003944 ACOX3[Unknown]  product
        HMDB0003944 ACADSB[Unknown] product
        HMDB0003944 MECR[Unknown]   substrate
        HMDB0006529 PECR[Enzyme]    substrate
        ....
    method
        the way to aggerate expression of metabolite related multiple genes or enzymes, should be one of [mean, gmean, median, max],
        mean for taking arithmetic mean, gmean for taking geomatrix mean, median for taking median value, max for taking maximum value across the genes.
        By default, we set mean which is arithmetic mean.


    Return
    -----
    Returns a data frame, columns are cells, rows are metabolite HMDB id

    """
    ## metabolite related genes to data frame
    met_gene_new = []
    for i, line in met_gene.iterrows():
        genes = line['gene'].split('; ')
        for g in genes:
            tmp = line.copy()
            tmp['gene'] = g
            met_gene_new.append(tmp)
    met_gene = pd.DataFrame(met_gene_new) ## each row is the related gene annotation for metabolite
    met_gene['gene_name'] = met_gene['gene'].apply(lambda x: x.split('[')[0])
    if len(mIdList) == 0:
        mIdList = met_gene['HMDB_ID'].unique().tolist()
    
    met_from_gene, met_indexer, met_columns = _met_est_input_sparse_(exp_mat = exp_mat, indexer = indexer,
                                              columns = columns, mIdList = mIdList, 
                                              met_gene = met_gene, method = method)
    # met_from_gene, met_indexer, met_columns = _met_est_input_sparse_by_reaction_(exp_mat = exp_mat, indexer = indexer,
    #                                           columns = columns, mIdList = mIdList, 
    #                                           met_gene = met_gene, method1 = 'gmean', method2 = 'mean')
    return(met_from_gene, met_indexer, met_columns)







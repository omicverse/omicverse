#!/usr/bin/env python

# ================================
# @auther: Rongbin Zheng
# @email: Rongbin.Zheng@childrens.harvard.edu
# @date: Oct 2024
# ================================

import os,sys
import time
import pickle as pk
from datetime import datetime
import numpy as np
import pandas as pd
import traceback
from operator import itemgetter
import statsmodels
from statsmodels import api as sm
import scipy
from scipy import sparse
import collections
import multiprocessing
from functools import partial

"""
main functions of differential cross talk calculator
v1.2.2: fix the bug for differential analysis using log2FC

"""


def info(string):
    """
    print information
    """
    today = datetime.today().strftime("%B %d, %Y")
    now = datetime.now().strftime("%H:%M:%S")
    current_time = today + ' ' + now
    print("[{}]: {}".format(current_time, string))

def _testing_(real_score, bg_values, method = 'ztest', statv = 'Log2FC'):
    """
    run hypothesis testing
    real_score: the observed value
    bg_value: a array-like or list of background value
    """
    stat, pval = 0, 1
    if statv == 'Log2FC':
        sep = 0
    else:
        sep = 1
        
    if method == 'ztest':
        stat, pval = sm.stats.ztest(bg_values,
                                value = real_score, 
                                alternative = "smaller" if real_score > sep else "larger")
    elif method == 'ttest':
        stat, pval, df = sm.stats.ttest_ind(x1 = bg_values,
                                x2 = [real_score], 
                                alternative = "smaller" if real_score > sep else "larger")
    elif method == 'ranksum_test':
        stat, pval = scipy.stats.ranksums(x = bg_values, y = [real_score], 
                                          alternative = "less" if real_score > sep else "greater")
    # elif method == 'sign_test':
    #     stat, pval = statsmodels.stats.descriptivestats.sign_test(bg_values, real_score)
    elif method == 'permutation_test':
        ### greater
        bg_larger = 0
        if real_score > sep:
            for i in bg_values:
                if i > real_score:
                    bg_larger += 1
        else:
            for i in bg_values:
                if i < real_score:
                    bg_larger += 1
        stat, pval = bg_larger, bg_larger / len(bg_values)
    else:
        print('+++ unknown testing')
        
    return(stat, pval)

def cummin( x):
    """A python implementation of the cummin function in R"""
    for i in range(1, len(x)):
        if x[i-1] < x[i]:
            x[i] = x[i-1]
    return x

def bh_fdr(pval):
    """A python implementation of the Benjamani-Hochberg FDR method.
    This code should always give precisely the same answer as using
    p.adjust(pval, method="BH") in R.
    Parameters
    ----------
    pval : list or array
        list/array of p-values
    Returns
    -------
    pval_adj : np.array
        adjusted p-values according the benjamani-hochberg method
    """
    pval_array = np.array(pval)
    sorted_order = np.argsort(pval_array)
    original_order = np.argsort(sorted_order)
    pval_array = pval_array[sorted_order]

    # calculate the needed alpha
    n = float(len(pval))
    pval_adj = np.zeros(int(n))
    i = np.arange(1, int(n)+1, dtype=float)[::-1]  # largest to smallest
    pval_adj = np.minimum(1, cummin(n/i * pval_array[::-1]))[::-1]
    return pval_adj[original_order]

def _fdr_(pvalue_res,
      testing_method = ['ztest', 'ttest', 'ranksum_test', 'permutation_test']):
    """
    calculate FDR correction
    """
    method = ['permutation_test']
    pvalue_res = pvalue_res.fillna(1) ## if NA
    for method in testing_method:
        fdr = bh_fdr(pvalue_res[method+'_pval'])
        pvalue_res[method+'_fdr'] = fdr
        del fdr
    return(pvalue_res)
    
def computeDiffcomm(comparison, 
                    cell_ann,
                    condition_col,
                    group_col,
                    original_result,
                    prop_cut,
                    commu_bg,
                    commu_res=pd.DataFrame(),
                   thread = 1):
    """
    compute and test for differential mCCC
    """
    info('Compute diff mCCC')
    cond1, cond2 = comparison.split('_vs_')
    info('Cond1 is {c1}, Cond2 is {c2}'.format(c1=cond1, c2=cond2))
    conds = cell_ann[condition_col].unique()
    cellgroup1 = cell_ann[cell_ann[condition_col] == cond1][group_col].unique()
    cellgroup2 = cell_ann[cell_ann[condition_col] == cond2][group_col].unique()
    cellgroups = np.intersect1d(cellgroup1, cellgroup2)

    if cond1 not in conds or cond2 not in conds:
        info('Error: condition name cannot be found')
        return 
        
    if len(cellgroups) == 0:
        info('Error: no common cellgroup identified between conditions')
        return       

    ## stat test for each diff 
    info('Compute fold change')
    ## === new
    ms_cellpair = original_result[['Metabolite_Name', 'Metabolite', 'Sensor', 'Sender', 'Receiver', 'Commu_Score', 'met_in_sender', 'sensor_in_receiver']]
    ms_cellpair.index = range(ms_cellpair.shape[0])
    ms_cellpair['cond'], ms_cellpair['Sender'] = np.array(ms_cellpair['Sender'].str.split(' ~ ').tolist()).T
    ## only the right cond
    ms_cellpair = ms_cellpair[ms_cellpair['cond'].isin([cond1, cond2])]
    ms_cellpair['Receiver'] = np.array(ms_cellpair['Receiver'].str.split(' ~ ').tolist())[:,1]
    ms_cellpair = ms_cellpair[ms_cellpair['Sender'].isin(cellgroups) & ms_cellpair['Receiver'].isin(cellgroups)]
    
    if commu_res.shape[0] > 0:
        commu_res['Sender'] = [x1.replace(x2+' ~ ', '') for x1, x2 in commu_res[['Sender', 'Condition']].values.tolist()]
        commu_res['Receiver'] = [x1.replace(x2+' ~ ', '') for x1, x2 in commu_res[['Receiver', 'Condition']].values.tolist()]
        sig_mccc = commu_res[['Metabolite_Name', 'Sensor', 'Sender', 'Receiver']].apply(lambda row: '~'.join(row.tolist()), axis = 1)
        uniq_mccc = ms_cellpair[['Metabolite_Name', 'Sensor', 'Sender', 'Receiver']].apply(lambda row: '~'.join(row.tolist()), axis = 1)
        ms_cellpair = ms_cellpair.loc[uniq_mccc.isin(sig_mccc),:]
    else:
        indices = (original_result['Commu_Score'] > 0) & (original_result['metabolite_prop_in_sender'] > prop_cut) & (original_result['sensor_prop_in_receiver'] > prop_cut)
        original_result = original_result.loc[indices,:]
        original_result['Sender'] = [x1.replace(x2+' ~ ', '') for x1, x2 in original_result[['Sender', 'Condition']].values.tolist()]
        original_result['Receiver'] = [x1.replace(x2+' ~ ', '') for x1, x2 in original_result[['Receiver', 'Condition']].values.tolist()]
        sig_mccc = original_result[['Metabolite_Name', 'Sensor', 'Sender', 'Receiver']].apply(lambda row: '~'.join(row.tolist()), axis = 1)
        uniq_mccc = ms_cellpair[['Metabolite_Name', 'Sensor', 'Sender', 'Receiver']].apply(lambda row: '~'.join(row.tolist()), axis = 1)
        ms_cellpair = ms_cellpair.loc[uniq_mccc.isin(sig_mccc),:]
    
    ms_cellpair = ms_cellpair.pivot_table(index=['Metabolite_Name', 'Metabolite', 'Sensor', 'Sender', 'Receiver'], 
                                 columns='cond', 
                                 values=['Commu_Score', 'met_in_sender', 'sensor_in_receiver']).reset_index()
    ms_cellpair.columns = ['_'.join(col).strip() if col[1] else col[0] for col in ms_cellpair.columns.values]
    ## remove NA, maybe only found in one condition
    ms_cellpair = ms_cellpair[~pd.isna(ms_cellpair['Commu_Score_'+cond1]) & ~pd.isna(ms_cellpair['Commu_Score_'+cond2])]
    # ms_cellpair = ms_cellpair.dropna()

    ##  Symmetric Log Fold Change, adding minimal as pseudo to avoid nan and neg
    ep = 1e-06
    amin = abs(min([ms_cellpair['Commu_Score_'+cond1].min(), ms_cellpair['Commu_Score_'+cond2].min()]))+ep
    v1, v2 = (ms_cellpair['Commu_Score_'+cond1]+amin), (ms_cellpair['Commu_Score_'+cond2]+amin)
    ms_cellpair['Scaled_Commu_Score_'+cond1] = v1
    ms_cellpair['Scaled_Commu_Score_'+cond2] = v2
    ms_cellpair['FC'] = v1 / v2
    ms_cellpair['Log2FC'] = np.log2(v1) - np.log2(v2)

    ## compute background fold change
    info('Estimate background diff')
    ms_pairs = ms_cellpair[['Metabolite_Name', 'Metabolite', 'Sensor']].drop_duplicates()
    sender_cg = ms_cellpair['Sender'].unique().tolist()
    receiver_cg = ms_cellpair['Receiver'].unique().tolist()

    ## null distribution
    # amin_bg = abs(min([commu_bg[x].drop(['N_permut', 'Sender_'], axis = 1).min().min() for x in commu_bg]))
    bgfc_res = collections.defaultdict()
    for i, line in ms_pairs.iterrows():
        m, hmid, s = line.tolist()
        bg = commu_bg[hmid+'~'+s]
        cond, cellgroup = np.array(bg['Sender_'].str.split(' ~ ').tolist()).T
        bg['cond'] = cond
        bg['cg'] = cellgroup
        tmp = pd.DataFrame()
        ## permutation was done on each m-s, so pseudo value on the base of m-s
        bmin = abs(bg[receiver_cg].min().min())+ep
        for x in sender_cg:
            df1 = bg[(bg['cg'] == x) & (bg['cond'] == cond1)][receiver_cg]
            df2 = bg[(bg['cg'] == x) & (bg['cond'] == cond2)][receiver_cg]
            df1.index = range(df1.shape[0])
            df2.index = range(df2.shape[0])
            # ttmp = (df1+amin).div((df2+amin))
            # ttmp = np.log2(df1+amin) - np.log2(df2+amin)
            ttmp = np.log2(df1+bmin) - np.log2(df2+bmin)
            ttmp['Sender_'] = x
            tmp = pd.concat([tmp, ttmp])
        bgfc_res[m+'~'+s] = tmp
        del tmp
        del ttmp

    info('Test significance')
    methods = ['permutation_test', 'ttest']
    if isinstance(thread, int) and thread > 1:
        func_input = []
        for i, line in ms_cellpair.iterrows():
            m, s, sender, receiver = line['Metabolite_Name'], line['Sensor'], line['Sender'], line['Receiver']
            fc = line['Log2FC']
            bg_v = bgfc_res[m+'~'+s].query('Sender_ == @sender')[receiver].dropna()
            func_input.append((fc, bg_v))

        for method in methods:
            with multiprocessing.Pool(thread) as pool:
                func = partial(
                    _testing_, 
                    method = method,
                    statv = 'Log2FC'
                )
                results = pool.starmap(func, func_input)
    
            ms_cellpair[method+'_stat'] = [x[0] for x in results]
            ms_cellpair[method+'_pval'] = [x[1] for x in results]
    else:
        fc_res = []
        for i, line in ms_cellpair.iterrows():
            m, s, sender, receiver = line['Metabolite_Name'], line['Sensor'], line['Sender'], line['Receiver']
            fc = line['Log2FC']
            bg_v = bgfc_res[m+'~'+s].query('Sender_ == @sender')[receiver].dropna()
            for method in methods:
                stat, pval = _testing_(fc, bg_v, method = method, statv = 'Log2FC')
                line[method+'_stat'] = stat
                line[method+'_pval'] = pval
            fc_res.append(line)
        ms_cellpair = pd.DataFrame(fc_res)
        
    ms_cellpair = _fdr_(ms_cellpair, testing_method=['permutation_test', 'ttest'])
    ms_cellpair = ms_cellpair.sort_values('permutation_test_fdr')
    return(ms_cellpair)

def DiffComm(cell_ann,
            condition_col,
            group_col,
            original_result,
            commu_bg, 
            prop_cut,
            comparison,
            commu_res = pd.DataFrame(),
            thread = 1):
    """
    handle differential mCCC test
    """
    diff_res = computeDiffcomm(comparison, 
                            cell_ann,
                            condition_col,
                            group_col,
                            original_result,
                            prop_cut,
                            commu_bg,
                            commu_res = commu_res,
                            thread = thread)

    return(diff_res)







import os,sys
import time, re
import pickle as pk
from datetime import datetime
import numpy as np
import pandas as pd
from operator import itemgetter
import scipy
from scipy import sparse
import scanpy as sc
import collections
import multiprocessing
import configparser
import tracemalloc
import warnings
import importlib



def info(string):
    """
    print information
    """
    today = datetime.today().strftime("%B %d, %Y")
    now = datetime.now().strftime("%H:%M:%S")
    current_time = today + ' ' + now
    print("[{}]: {}".format(current_time, string))

def _get_one_compass_(compass_folder):
    
    if os.path.exists(compass_folder):
        uptake_path = os.path.join(compass_folder, 'uptake.tsv')
        secret_path = os.path.join(compass_folder, 'secretions.tsv')
        if os.path.exists(uptake_path) and os.path.exists(secret_path):
            uptake = pd.read_csv(uptake_path, index_col = 0, sep = '\t')
            secretion = pd.read_csv(secret_path, index_col = 0, sep = '\t')
        else:
            uptake_path = os.path.join(compass_folder, 'uptake.tsv.gz')
            secret_path = os.path.join(compass_folder, 'secretions.tsv.gz')
            if os.path.exists(uptake_path) and os.path.exists(secret_path):
                uptake = pd.read_csv(uptake_path, index_col = 0, sep = '\t')
                secretion = pd.read_csv(secret_path, index_col = 0, sep = '\t')
            else:
                raise ValueError('Failed to identify COMPASS output files')
    else:
        raise ValueError('compass_folder path does not exist')
    return(uptake, secretion)

def _get_compass_flux_(compass_folder, compass_met_ann_path, met_ann):  
    
    if isinstance(compass_folder, dict):
        uptake = pd.DataFrame()
        secretion = pd.DataFrame()
        for cond in compass_folder:
            # try:
            uptake_tmp, secretion_tmp = _get_one_compass_(compass_folder[cond])
            # except:
            #     continue
            uptake_tmp.columns = [cond + ' ~ ' + x for x in uptake_tmp.columns.tolist()] 
            secretion_tmp.columns = [cond + ' ~ ' + x for x in secretion_tmp.columns.tolist()] 
            uptake = pd.concat([uptake, uptake_tmp], axis = 1)
            secretion = pd.concat([secretion, secretion_tmp], axis = 1)
    else:
        uptake, secretion = _get_one_compass_(compass_folder)
                    
    ## load compass annotation
    compass_met_ann = pd.read_csv(compass_met_ann_path) #_read_config(self.config_path)['common']['compass_met_ann_path'])
    # compass_rxn_ann = pd.read_csv(_read_config(self.config_path)['common']['compass_rxt_ann_path'])
    ## annotate compass result
    efflux_mat = pd.merge(secretion, compass_met_ann[['met', 'hmdbID']],
                            left_index = True, right_on = 'met').dropna()
    efflux_mat = pd.merge(efflux_mat, met_ann[['Secondary_HMDB_ID', 'metabolite']],
                            left_on = 'hmdbID', right_on = 'Secondary_HMDB_ID')
    efflux_mat = efflux_mat.drop(['met','hmdbID','Secondary_HMDB_ID'], axis = 1).groupby('metabolite').max()
    influx_mat = pd.merge(uptake, compass_met_ann[['met', 'hmdbID']],
                            left_index = True, right_on = 'met').dropna()
    influx_mat = pd.merge(influx_mat, met_ann[['Secondary_HMDB_ID', 'metabolite']],
                            left_on = 'hmdbID', right_on = 'Secondary_HMDB_ID')
    influx_mat = influx_mat.drop(['met','hmdbID','Secondary_HMDB_ID'], axis = 1).groupby('metabolite').max()
    return(efflux_mat, influx_mat)

        

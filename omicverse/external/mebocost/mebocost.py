#!/usr/bin/env python

# ================================
# @auther: Rongbin Zheng
# @email: Rongbin.Zheng@childrens.harvard.edu
# @date: July 2025
# ================================

import os,sys
import time, re
import pickle as pk
import _pickle as cPickle
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
import copy

import importlib
from . import MetEstimator as ME
from . import crosstalk_calculator as CC
from . import crosstalk_plots as CP
from . import fba_handler as FBA
from . import crosstalk_diff as CD


from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
## disable warnings
import warnings
warnings.filterwarnings("ignore")


plt.rcParams.update(plt.rcParamsDefault)
rc={"axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "figure.titleweight":"bold", #"font.size":14,
    "figure.figsize":(5.5,4.2), "font.weight":"regular", "legend.fontsize":10,
    'axes.labelpad':8, 'figure.dpi':300}
plt.rcParams.update(**rc)

"""
linking input and out 
"""

def info(string):
    """
    print information
    """
    today = datetime.today().strftime("%B %d, %Y")
    now = datetime.now().strftime("%H:%M:%S")
    current_time = today + ' ' + now
    print("[{}]: {}".format(current_time, string))

def __version__():
    info('Version: 1.2.2')

def concat_obj(obj1, obj2, cond1='cond1', cond2='cond2'):
    """
    Concatenate two mebocost objects without mutating the originals.  
    Performs copying only for attributes that need modification.
    Returns a new merged object.
    """
    # 1) Merge cell annotations (copy only those DataFrames)
    ca1 = obj1.cell_ann.copy()
    ca1['Condition'] = cond1
    ca1.index = ca1.index + ' ~ ' + cond1
    ca2 = obj2.cell_ann.copy()
    ca2['Condition'] = cond2
    ca2.index = ca2.index + ' ~ ' + cond2
    cell_ann = pd.concat([ca1, ca2])
    cell_ann['group_col_concat'] = cell_ann[obj1.group_col].tolist()
    cell_ann['cell_group'] = (
        cell_ann['Condition'].str.replace('~', '_') + ' ~ ' +
        cell_ann['cell_group'].str.replace('~', '_')
    )

    # 2) Merge communication results
    cr1 = obj1.commu_res.copy()
    cr1['Condition'] = cond1
    cr1['Sender'] = cond1 + ' ~ ' + cr1['Sender']
    cr1['Receiver'] = cond1 + ' ~ ' + cr1['Receiver']
    cr2 = obj2.commu_res.copy()
    cr2['Condition'] = cond2
    cr2['Sender'] = cond2 + ' ~ ' + cr2['Sender']
    cr2['Receiver'] = cond2 + ' ~ ' + cr2['Receiver']
    commu_res = pd.concat([cr1, cr2])

    # Original results
    if isinstance(obj1.original_result, pd.DataFrame):
        or1 = obj1.original_result.copy()
        or1['Condition'] = cond1
        or1['Sender'] = cond1 + ' ~ ' + or1['Sender']
        or1['Receiver'] = cond1 + ' ~ ' + or1['Receiver']
        or2 = obj2.original_result.copy()
        or2['Condition'] = cond2
        or2['Sender'] = cond2 + ' ~ ' + or2['Sender']
        or2['Receiver'] = cond2 + ' ~ ' + or2['Receiver']
        original_result = pd.concat([or1, or2])
    else:
        original_result = None

    # 3) Merge background tables
    commu_bg = collections.defaultdict()
    for key in set(obj1.commu_bg) & set(obj2.commu_bg):
        bg1 = obj1.commu_bg[key].copy()
        bg1['Sender_'] = cond1 + ' ~ ' + bg1['Sender_']
        bg2 = obj2.commu_bg[key].copy()
        bg2['Sender_'] = cond2 + ' ~ ' + bg2['Sender_']
        commu_bg[key] = pd.concat([bg1, bg2])

    # 4) Merge properties
    mp1 = obj1.met_prop.copy(); mp1.index = cond1 + ' ~ ' + mp1.index
    mp2 = obj2.met_prop.copy(); mp2.index = cond2 + ' ~ ' + mp2.index
    met_prop = pd.concat([mp1, mp2])
    ep1 = obj1.exp_prop.copy(); ep1.index = cond1 + ' ~ ' + ep1.index
    ep2 = obj2.exp_prop.copy(); ep2.index = cond2 + ' ~ ' + ep2.index
    exp_prop = pd.concat([ep1, ep2])

    # 5) Build combined matrices without mutating originals
    def combine_matrix(mat1, idx1, cols1, mat2, idx2, cols2, prefix1, prefix2):
        df1 = pd.DataFrame(mat1.toarray(), index=idx1,
                           columns=[f"{c} ~ {prefix1}" for c in cols1])
        df2 = pd.DataFrame(mat2.toarray(), index=idx2,
                           columns=[f"{c} ~ {prefix2}" for c in cols2])
        return sparse.csc_matrix(pd.concat([df1, df2]).fillna(0))

    exp_mat = combine_matrix(obj1.exp_mat, obj1.exp_mat_indexer, obj1.exp_mat_columns,
                             obj2.exp_mat, obj2.exp_mat_indexer, obj2.exp_mat_columns,
                             cond1, cond2)
    avg_exp = combine_matrix(obj1.avg_exp, obj1.avg_exp_indexer, obj1.avg_exp_columns,
                              obj2.avg_exp, obj2.avg_exp_indexer, obj2.avg_exp_columns,
                              cond1, cond2)
    met_mat = combine_matrix(obj1.met_mat, obj1.met_mat_indexer, obj1.met_mat_columns,
                             obj2.met_mat, obj2.met_mat_indexer, obj2.met_mat_columns,
                             cond1, cond2)
    avg_met = combine_matrix(obj1.avg_met, obj1.avg_met_indexer, obj1.avg_met_columns,
                              obj2.avg_met, obj2.avg_met_indexer, obj2.avg_met_columns,
                              cond1, cond2)

    # 6) Efflux and influx if present
    def maybe_combine_df(d1, d2, prefix1, prefix2):
        if isinstance(d1, pd.DataFrame) and isinstance(d2, pd.DataFrame):
            c1 = d1.copy(); c1.columns = [f"{c} ~ {prefix1}" for c in c1.columns]
            c2 = d2.copy(); c2.columns = [f"{c} ~ {prefix2}" for c in c2.columns]
            return sparse.csc_matrix(pd.concat([c1, c2], axis=1).fillna(0))
        return None

    efflux_mat = maybe_combine_df(obj1.efflux_mat, obj2.efflux_mat, cond1, cond2)
    influx_mat = maybe_combine_df(obj1.influx_mat, obj2.influx_mat, cond1, cond2)

    # 7) Shallow copy obj1, assign merged data
    merged = copy.copy(obj1)
    merged.group_col = 'group_col_concat'
    merged.condition_col = 'Condition'
    merged.cutoff_exp  = max(obj1.cutoff_exp, obj2.cutoff_exp)
    merged.cutoff_met  = max(obj1.cutoff_met, obj2.cutoff_met)
    merged.cutoff_prop = max(obj1.cutoff_prop, obj2.cutoff_prop)

    merged.cell_ann         = cell_ann
    merged.commu_res        = commu_res
    merged.original_result  = original_result
    merged.commu_bg         = commu_bg
    merged.met_prop         = met_prop
    merged.exp_prop         = exp_prop
    merged.exp_mat          = exp_mat
    merged.exp_mat_indexer  = cell_ann.index[:exp_mat.shape[0]]
    merged.exp_mat_columns  = exp_mat.shape[1] and exp_mat.indices
    merged.avg_exp          = avg_exp
    merged.met_mat          = met_mat
    merged.avg_met          = avg_met
    merged.efflux_mat       = efflux_mat
    merged.influx_mat       = influx_mat

    return merged
    
def _correct_colname_meta_(scRNA_meta, cellgroup_col=[]):
    """
    sometime the column names have different
    """
#     print(scRNA_meta)
    if scRNA_meta is None or scRNA_meta is pd.DataFrame:
        raise KeyError('Please provide cell_ann data frame!')
    
    if cellgroup_col:
        ## check columns names
        for x in cellgroup_col:
            if x not in scRNA_meta.columns.tolist():
                info('ERROR: given cell group identifier {} is not in meta table columns'.format(x))
                raise ValueError('given cell group identifier {} is not in meta table columns'.format(x))
        ## get cell group name
        scRNA_meta['cell_group'] = scRNA_meta[cellgroup_col].astype('str').apply(lambda row: '_'.join(row), axis = 1).tolist()
    else:
        info('no cell group given, try to search cluster and cell_type')
        col_names = scRNA_meta.columns.tolist()
        if 'cell_type' in col_names:
            pass
        elif 'cell_type' not in col_names and 'Cell_Type' in col_names:
            scRNA_meta.columns = ['cell_type' if x.upper() == 'CELL_TYPE' else x for x in col_names]
        elif 'cell_type' not in col_names and 'celltype' in col_names:
            scRNA_meta.columns = ['cell_type' if x.upper() == 'CELLTYPE' else x for x in col_names]
        elif 'cell_type' not in col_names and 'CellType' in col_names:
            scRNA_meta.columns = ['cell_type' if x.upper() == 'CELL TYPE' else x for x in col_names]
        else:
            info('ERROR: "cell_type" not in scRNA meta column names, will try cluster')
            if 'cluster' not in col_names and 'Cluster' in col_names:
                scRNA_meta.columns = ['cluster' if x.upper() == 'CLUSTER' else x for x in col_names]
            else:
                raise KeyError('cluster cannot find in the annotation, and cell_group does not specified')
            raise KeyError('cell_type cannot find in the annotation, and cell_group does not specified'.format(x))
        
        if 'cell_type' in scRNA_meta.columns.tolist():
            scRNA_meta['cell_group'] = scRNA_meta['cell_type'].tolist()
        elif 'cluster' in scRNA_meta.columns.tolist():
            scRNA_meta['cell_group'] = scRNA_meta['cluster'].tolist()
        else:
            raise KeyError('Please a group_col to group single cell')
    return(scRNA_meta)

def _read_config(conf_path):
    """
    read config file
    """
    #read config
    cf = configparser.ConfigParser()
    cf.read(conf_path)
    config = cf._sections
    # remove the annotation:
    for firstLevel in config.keys():
        for secondLevel in config[firstLevel]:
            if '#' in config[firstLevel][secondLevel]:
                config[firstLevel][secondLevel] = config[firstLevel][secondLevel][:config[firstLevel][secondLevel].index('#')-1].rstrip()
    return(config)


def load_obj(path):
    """
    read mebocost object
    """
    try:
        file = open(path,'rb')
        dataPickle = file.read()
        file.close()
        obj_vars = cPickle.loads(dataPickle)
    except:
        obj_vars = pd.read_pickle(path)
    ## check for group_col: for v1 it accepts list, for v2, only string
    ## so need to check
    if isinstance(obj_vars['group_col'], list):
        if 'cell_group' in obj_vars:
            obj_vars['group_col'] = "cell_group"
        else:
            # obj_vars['cell_ann']['cell_group'] = obj_vars['cell_ann'][obj_vars['group_col'][0]].astype(str)+' ~ '+obj_vars['cell_ann'][obj_vars['group_col'][1]].astype(str)
            obj_vars['cell_ann']['cell_group'] = obj_vars['cell_ann'][obj_vars['group_col']].apply(lambda row: ' ~ '.join(row.tolist()), axis = 1)
            obj_vars['group_col'] = "cell_group"
            
    mebocost_obj = create_obj(exp_mat = obj_vars['exp_mat'] if 'exp_mat' in obj_vars else None,
                        adata = obj_vars['adata'] if 'adata' in obj_vars else None,
                        cell_ann = obj_vars['cell_ann'] if 'cell_ann' in obj_vars else None,
                        group_col = obj_vars['group_col'] if 'group_col' in obj_vars else None,
                        condition_col = obj_vars['condition_col'] if 'condition_col' in obj_vars else None,
                        config_path = obj_vars['config_path'] if 'config_path' in obj_vars else None,
                       )
    mebocost_obj.__dict__ = obj_vars

    return mebocost_obj


def save_obj(obj, path = 'mebocost_result.pk', filetype = 'pickle'):
    """
    save object to pickle
    """
    file = open(path, 'wb')
    file.write(cPickle.dumps(obj.__dict__))
    file.close()

def _check_exp_mat_(exp_mat):
    """
    check if the expression matrix are all numerical
    """
    str_cols = exp_mat.apply(lambda col: np.array_equal(col, col.astype(str)))
    str_rows = exp_mat.apply(lambda row: np.array_equal(row, row.astype(str)), axis = 1)
    if np.any(str_cols == True):
        warnings.warn("%s column is a str, will be removed as only int or float accepted in expression matrix"%(exp_mat.columns[str_cols]))
    if np.any(str_rows == True):
        warnings.warn("%s row is a str, will be removed as only int or float accepted in expression matrix"%(exp_mat.columns[str_cols]))
    exp_mat = exp_mat.loc[~str_rows, ~str_cols]
    return(exp_mat)


class create_obj:
    """
    MEBOCOST for predicting metabolite-based cell-cell communication (mCCC). The modules of the package include communication inference, communication visualization, and differential communication analysis.

    Params
    -------
    exp_mat
        python pandas data frame, single cell expression matrix, rows are genes, columns are cells
        'exp_mat' is a exclusive parameter to 'adata'
    adata
        scanpy adata object, the expression will be extracted, 'adata' is an exclusive parameter to 'exp_mat'
    cell_ann
        data frame, cell annotation information, cells in row names
    group_col
        a list, specify the column names in 'cell_ann' for grouping cells, by default 'cell_type' or 'cluster' will be detected and used
    condition_col
        a list, specify the column names in 'cell_ann' for running mCCC in different samples/conditions, e.g., control, treatment
    species
        human or mouse, this determines which database will be used in our collection

    met_est
        the method for estimating metabolite enzyme expression in cell, should be one of:
        mebocost: estimated by the enzyme network related to the metabolite
        scFEA-flux: flux result of published software scFEA (https://pubmed.ncbi.nlm.nih.gov/34301623/)
        scFEA-balance: balance result of published software scFEA (https://pubmed.ncbi.nlm.nih.gov/34301623/)
        compass-reaction: reaction result of published software Compass (https://pubmed.ncbi.nlm.nih.gov/34216539/)
        compass-uptake: uptake result of published software Compass (https://pubmed.ncbi.nlm.nih.gov/34216539/)
        compass-secretion: secretion result of published software Compass (https://pubmed.ncbi.nlm.nih.gov/34216539/)
    met_pred
        data frame, if scFEA or Compass is used to impute the metabolite enzyme expression in cells, please provide the original result from scFEA or Compass, cells in row names, metabolite/reaction/module in column names, 
        Noted that this parameter will be ignored if 'met_est' was set as mebocost.

    config_path
        str, the path for a config file containing the path of files for metabolite annotation, enzyme, sensor, scFEA annotation, compass annotation. These can also be specified separately by paramters as following:

        if config_path not given, please set:
    met_enzyme
        data frame, metabolite and gene (enzyme) relationships, required columns include HMDB_ID, gene, direction, for instance:
        
        HMDB_ID     gene                                                direction
        HMDB0003375 Cyp2c54[Unknown]; Cyp2c38[Unknown]; Cyp2c50[Un...   substrate
        HMDB0003375 Cyp2c54[Unknown]; Cyp2c38[Unknown]; Cyp2c50[Un...   substrate
        HMDB0003375 Cyp2c54[Unknown]; Cyp2c38[Unknown]; Cyp2c50[Un...   substrate
        HMDB0003450 Cyp2c54[Unknown]; Cyp2c38[Unknown]; Cyp2c50[Un...   product
        HMDB0003948 Tuba8[Unknown]; Ehhadh[Unknown]; Echs1[Enzyme]...   product

    met_sensor
        data frame, metabolite sensor information, each row is a pair of metabolite and sensor, must include columns  HMDB_ID, Gene_name, Annotation, for instance:
        
        HMDB_ID Gene_name   Annotation
        HMDB0006247 Abca1   Transporter
        HMDB0000517 Slc7a1  Transporter
        HMDB0000030 Slc5a6  Transporter
        HMDB0000067 Cd36    Transporter
        
    met_ann:
        data frame, the annotation of metabolite collected from HMDB website, these are basic annotation info including HMDB_ID, Kegg_ID, metabolite, etc

    scFEA_ann
        data frame, module annotation of metabolite flux in scFEA, usually is the file at https://github.com/changwn/scFEA/blob/master/data/Human_M168_information.symbols.csv

    compass_met_ann
        data frame, the metabolite annotation used in Compass software, usually is the file at https://github.com/YosefLab/Compass/blob/master/compass/Resources/Recon2_export/met_md.csv

    compass_rxn_ann
        data frame, the reaction annotation used in Compass software, usually is the file at https://github.com/YosefLab/Compass/blob/master/compass/Resources/Recon2_export/rxn_md.csv

    cutoff_exp
        auto or float, used to filter out cells which are lowly expressed for the given gene, by default is auto, meaning that automatically decide cutoffs for sensor expression to exclude the lowly 25% non-zeros across all sensor or metabolites in all cells in addition to zeros 

    cutoff_met
        auto or float, used to filter out cells which are lowly abundant of the given metabolite, by default is auto, meaning that automatically decide cutoffs for metabolite presence to exclude the lowly 25% non-zeros across all sensor or metabolites in all cells in addition to zeros 

    cutoff_prop
        float from 0 to 1, used to filter out metabolite or genes if the proportion of their abundant cells less than the cutoff

    sensor_type
        All or a list, the list set to focus on several sensor type in the communication modeling, must be one or more from ['receptor', 'transporter', 'enzyme', 'channel'], default is All

    thread
        int, number of cores used for running job, default 1
        
    """
    def __init__(self,  
                exp_mat=None, 
                adata=None, 
                cell_ann=None,
                group_col='celltype',
                condition_col=None,
                species = 'human',
                met_est=None,
                met_pred=pd.DataFrame(), 
                config_path=None,
                met_enzyme=pd.DataFrame(),
                met_sensor=pd.DataFrame(),
                met_ann=pd.DataFrame(), 
                scFEA_ann=pd.DataFrame(),
                compass_met_ann=pd.DataFrame(),
                compass_rxn_ann=pd.DataFrame(),
                cutoff_exp='auto',
                cutoff_met='auto',
                cutoff_prop=0.15,
                sensor_type='All',
                thread = 1
                ):
        tic = time.time()

        self.exp_mat = exp_mat
        self.adata = adata
        ## check cell group information
        ## add a column "cell_group" if successfull
        if (self.exp_mat is None and cell_ann is None) and (self.adata is not None):
            cell_ann = adata.obs.copy()
        if group_col not in cell_ann.columns.tolist():
            raise KeyError('group_col: %s is not in cell_ann columns, it should be one of %s'%(group_col, cell_ann.columns.tolist()))
        else:
            self.group_col = group_col
            # cell_ann['cell_group'] = cell_ann[group_col].tolist()

        if not condition_col or condition_col in cell_ann.columns.tolist():
            self.condition_col = condition_col
        else:
            raise KeyError('condition_col: %s is not in cell_ann columns, it should be one of %s'%(condition_col, cell_ann.columns.tolist()))
        
        # self.cell_ann = _correct_colname_meta_(cell_ann, cellgroup_col = self.group_col)
        self.cell_ann = cell_ann
        self.species = species

        self.met_est = 'mebocost' if not met_est else met_est # one of [scFEA-flux, scFEA-balance, compass-reaction, compass-uptake, compass-secretion]
        self.met_pred = met_pred

        ## the path of config file
        self.config_path = config_path
        ## genes (enzyme) related to met
        self.met_enzyme = met_enzyme
        ## gene name in metaboltie sensor
        self.met_sensor = met_sensor
        ## met basic ann
        self.met_ann = met_ann
        ## software ann
        self.scFEA_ann = scFEA_ann
        self.compass_met_ann = compass_met_ann
        self.compass_rxn_ann = compass_rxn_ann
        ## gene network
        if not self.config_path and (self.met_sensor is None or self.met_sensor.shape[0] == 0):
            raise KeyError('Please either provide config_path or a data frame of met_enzyme, met_sensor, met_ann, etc')

        ## cutoff for expression, metabolite, and proportion of cells
        self.cutoff_exp = cutoff_exp
        self.cutoff_met = cutoff_met
        self.cutoff_prop = cutoff_prop
        self.sensor_type = sensor_type
        self.thread = thread
        self.mode = 'scrna'
        
        ## ============== initial ===========

        if self.exp_mat is None and self.adata is None:
            raise ValueError('ERROR: please provide expression matrix either from exp_mat or adata (scanpy object)')  
        elif self.exp_mat is None and self.adata is not None:
            ## check the adata object
            ngene = len(self.adata.var_names)
            ncell = len(self.adata.obs_names)
            info('We get expression data with {n1} genes and {n2} cells.'.format(n1 = ngene, n2 = ncell))
            if ngene < 5000:
                info('scanpy object contains less than 5000 genes, please make sure you are using raw.to_adata()')
            self.exp_mat = sparse.csc_matrix(self.adata.X.T)
            self.exp_mat_indexer = self.adata.var_names
            self.exp_mat_columns = self.adata.obs_names
            self.adata = None
        else:
            if 'scipy.sparse' in str(type(self.exp_mat)):
                ## since the scipy version problem leads to the failure of using sparse.issparse
                ## use a simple way to check!!!
                #sparse.issparse(self.exp_mat):
                pass 
            elif type(self.exp_mat) is type(pd.DataFrame()):
                ## check if the exp_mat values are all int or float
                self.exp_mat = _check_exp_mat_(self.exp_mat)
                self.exp_mat_indexer = self.exp_mat.index ## genes
                self.exp_mat_columns = self.exp_mat.columns ## columns
                self.exp_mat = sparse.csc_matrix(self.exp_mat)
                ngene, ncell = self.exp_mat.shape
                info('We get expression data with {n1} genes and {n2} cells.'.format(n1 = ngene, n2 = ncell))
            else:
                info('ERROR: cannot read the expression matrix, please provide pandas dataframe or scanpy adata')
        if self.condition_col not in ['', False, 'NA', None, 'None']:
            group_names = self.cell_ann[self.condition_col].astype('str').str.replace('~', '_')+' ~ '+self.cell_ann[self.group_col].astype('str').str.replace('~', '_')
        else:
            group_names = self.cell_ann[self.group_col]
        self.cell_ann['cell_group'] = group_names.copy()
        self.group_names = group_names.unique().tolist()
        
        ## end preparation
        toc = time.time()
        info('Data Preparation Done in {:.4f} seconds'.format(toc-tic))


    def _load_config_(self):
        """
        load config and read data from the given path based on given species
        """
        ## the path of config file
        info('Load config and read data based on given species [%s].'%(self.species))
        if self.config_path:
            if not os.path.exists(self.config_path):
                raise KeyError('ERROR: the config path is not exist!')
            config = _read_config(conf_path = self.config_path)
            ## common
            self.met_ann = pd.read_csv(config['common']['hmdb_info_path'], sep = '\t')
            if self.met_est.startswith('scFEA'):
                    self.scFEA_ann = pd.read_csv(config['common']['scfea_info_path'], index_col = 0)
            if self.met_est.startswith('compass'):
                self.compass_met_ann = pd.read_csv(config['common']['compass_met_ann_path'])
                self.compass_rxn_ann = pd.read_csv(config['common']['compass_rxt_ann_path'])
            ## depends on species
            if self.species == 'human':
                self.met_enzyme = pd.read_csv(config['human']['met_enzyme_path'], sep = '\t')
                met_sensor = pd.read_csv(config['human']['met_sensor_path'], sep = '\t')
                self.met_sensor = met_sensor

            elif self.species == 'mouse':
                self.met_enzyme = pd.read_csv(config['mouse']['met_enzyme_path'], sep = '\t')
                met_sensor = pd.read_csv(config['mouse']['met_sensor_path'], sep = '\t')
                self.met_sensor = met_sensor
            else:
                raise KeyError('Species should be either human or mouse!')
            ## check row and columns, we expect rows are genes, columns are cells
            if len(set(self.met_sensor['Gene_name'].tolist()) & set(self.exp_mat_indexer.tolist())) < 10 and len(set(self.met_sensor['Gene_name'].tolist()) & set(self.exp_mat_columns.tolist())) < 10:
                raise KeyError('it looks like that both the row and columns are not matching to gene name very well, please check the provided matrix or species!')
            if len(set(self.met_sensor['Gene_name'].tolist()) & set(self.exp_mat_indexer.tolist())) < 10 and len(set(self.met_sensor['Gene_name'].tolist()) & set(self.exp_mat_columns.tolist())) > 10:
                info('it is likely the columns of the exp_mat are genes, will transpose the matrix')
                self.exp_mat = self.exp_mat.T
                columns = self.exp_mat_indexer.copy()
                index = self.exp_mat_columns.copy()
                self.exp_mat_indexer = index
                self.exp_mat_columns = columns
        else:
            info('please provide config path')

    def estimator(self):
        """
        estimate of metabolite enzyme expression in cells using the expression of related enzymes
        """
        info('Estimtate metabolite enzyme expression using %s'%self.met_est)
        mtd = self.met_est

        if mtd == 'mebocost':
            met_mat, met_indexer, met_columns = ME._met_from_enzyme_est_(exp_mat=self.exp_mat, 
                                                   indexer = self.exp_mat_indexer,
                                                   columns = self.exp_mat_columns,
                                                    met_gene=self.met_enzyme, 
                                                    method = 'mean')
        elif mtd == 'scFEA-flux':
            met_mat = ME._scFEA_flux_est_(scFEA_pred = self.met_pred, 
                                            scFEA_info=self.scFEA_ann, 
                                            hmdb_info=self.met_ann)
        elif mtd == 'scFEA-balance':
            met_mat = ME._scFEA_balance_est_(scFEA_pred = self.met_pred, 
                                                scFEA_info=self.scFEA_ann, 
                                                hmdb_info=self.met_ann)
        elif mtd == 'compass-reaction':
            met_mat = ME._compass_react_est_(compass_pred=self.met_pred, 
                                                compass_react_ann=self.compass_rxn_ann, 
                                                compass_met_ann=self.compass_met_ann, 
                                                hmdb_info=self.met_ann)
        else:
            raise KeyError('Please specify "met_est" to be one of [mebocost, scFEA-flux, scFEA-balance, compass-reaction, compass-uptake, compass-secretion]')
        
        self.met_mat = sparse.csc_matrix(met_mat)
        self.met_mat_indexer = np.array(met_indexer)
        self.met_mat_columns = np.array(met_columns)
#         return met_mat


    def infer(self, met_mat=pd.DataFrame(), n_shuffle = 1000, seed = 12345, thread = None):
        """
        excute communication prediction
        met_mat
            data frame, columns are cells and rows are metabolites
        """
        info('Infer communications')
        if met_mat.shape[0] != 0: ## if given met_mat in addition
            self.met_mat_indexer = np.array(met_mat.index)
            self.met_mat_columns = np.array(met_mat.columns)
            self.met_mat = sparse.csc_matrix(met_mat)
        ## focus on met and gene of those are in the data matrix
        met_sensor = self.met_sensor[self.met_sensor['Gene_name'].isin(self.exp_mat_indexer) & 
                                     self.met_sensor['HMDB_ID'].isin(self.met_mat_indexer)]
        self.met_sensor = met_sensor

        ## init
        cobj = CC.InferComm(exp_mat = self.exp_mat,
                            exp_mat_indexer = self.exp_mat_indexer, 
                            exp_mat_columns = self.exp_mat_columns,
                            avg_exp = self.avg_exp,
                            avg_exp_indexer = self.avg_exp_indexer,
                            avg_exp_columns = self.avg_exp_columns,
                            met_mat = self.met_mat,
                            met_mat_indexer = self.met_mat_indexer,
                            met_mat_columns = self.met_mat_columns,
                            avg_met = self.avg_met,
                            avg_met_indexer = self.avg_met_indexer,
                            avg_met_columns = self.avg_met_columns,
                            cell_ann = self.cell_ann,
                            group_col = self.group_col,
                            condition_col = self.condition_col,
                            met_sensor = self.met_sensor,
                            sensor_type = self.sensor_type,
                            thread = thread
                           )

        commu_res_df, commu_res_bg = cobj.pred(n_shuffle = n_shuffle, seed = seed)
    
        ## add metabolite name
        hmdbid_to_met = {}
        for Id, met in self.met_ann[['HMDB_ID', 'metabolite']].values.tolist():
            hmdbid_to_met[Id] = met
        ## add name
        commu_res_df['Metabolite_Name'] = list(map(lambda x: hmdbid_to_met.get(x) if x in hmdbid_to_met else None,
                                                   commu_res_df['Metabolite']))

        ## add annotation
        sensor_to_ann = {}
        for s, a in self.met_sensor[['Gene_name', 'Annotation']].values.tolist():
            sensor_to_ann[s] = a
        commu_res_df['Annotation'] = list(map(lambda x: sensor_to_ann.get(x) if x in sensor_to_ann else None,
                                              commu_res_df['Sensor']))
        
        return commu_res_df, commu_res_bg


    def _filter_lowly_aboundant_(self, 
                                 pvalue_res,
                                 cutoff_prop,
                                 met_prop=None,
                                 exp_prop=None,
                                 pval_method='permutation_test_fdr',
                                 pval_cutoff=0.05,
                                 min_cell_number=50,
                                 return_signi_only = False
                                ):
        """
        change p value to 1 if either metabolite_prop or transporter_prop less than the cutoff 
        (meaning that no enough metabolite or sensor present in the cell group)
        -------
         pvalue_res,
         cutoff_prop,
         met_prop=None,
         exp_prop=None,
         pval_method='permutation_test_fdr',
         pval_cutoff=0.05,
         min_cell_number=50,
         return_signi_only = False
        """
        res = pvalue_res.copy()
        ## add the metabolite abudance proportion
        if met_prop is not None:
            res['metabolite_prop_in_sender'] = [met_prop.loc[s, m] for s, m in res[['Sender', 'Metabolite']].values.tolist()]
        ## add the metabolite abudance proportion
        if exp_prop is not None:
            res['sensor_prop_in_receiver'] = [exp_prop.loc[r, s] for r, s in res[['Receiver', 'Sensor']].values.tolist()]
        
        if 'original_result' not in list(vars(self)):
            self.original_result = res.copy()
        ## minimum cell number
        cell_count = pd.Series(dict(collections.Counter(self.cell_ann['cell_group'].tolist())))
        bad_cellgroup = cell_count[cell_count<min_cell_number].index.tolist() 
        
        info('Set p value and fdr to 1 if sensor or metaboltie expressed cell proportion less than {}'.format(cutoff_prop))
        bad_index = np.where((res['metabolite_prop_in_sender'] <= cutoff_prop) |
                             (res['sensor_prop_in_receiver'] <= cutoff_prop) |
                             (res['Commu_Score'] < 0) |
                             (res['Sender'].isin(bad_cellgroup)) | 
                             (res['Receiver'].isin(bad_cellgroup))
                            )[0]
        if len(bad_index) > 0:
            pval_index = np.where(res.columns.str.endswith('_pval'))[0]
            res.iloc[bad_index, pval_index] = 1 # change to 1
            fdr_index = np.where(res.columns.str.endswith('_fdr'))[0]
            res.iloc[bad_index, fdr_index] = 1 # change to 1
        
        if return_signi_only:
            ## filter out non-significant pairs
            if pval_method in res.columns.tolist():
                res = res[res[pval_method]<pval_cutoff]
            else:
                warnings.warn('%s is not in pvalue_res table columns, so will skip p value filter'%pval_method)

        if 'Condition' in res.columns.tolist():
            ## reorder columns
            columns = ['Sender', 'Receiver', 'Condition',
                        'Metabolite', 'Metabolite_Name', 'Sensor', 
                        'Annotation', 'Commu_Score', 'Norm_Commu_Score',
                       'met_in_sender', 'sensor_in_receiver',
                       'metabolite_prop_in_sender', 'sensor_prop_in_receiver', 
                       'ttest_stat','ttest_pval', 'ranksum_test_stat', 'ranksum_test_pval',
                       'permutation_test_stat', 'permutation_test_pval',
                       'ttest_fdr', 'ranksum_test_fdr',
                       'permutation_test_fdr']
        else:
            columns = ['Sender', 'Receiver',
                        'Metabolite', 'Metabolite_Name', 'Sensor', 
                        'Annotation', 'Commu_Score', 'Norm_Commu_Score',
                       'met_in_sender', 'sensor_in_receiver',
                       'metabolite_prop_in_sender', 'sensor_prop_in_receiver', 
                        'ttest_stat','ttest_pval', 'ranksum_test_stat', 'ranksum_test_pval',
                       'permutation_test_stat', 'permutation_test_pval',
                       'ttest_fdr', 'ranksum_test_fdr',
                       'permutation_test_fdr']
        get_columns = [x for x in columns if x in res.columns.tolist()]
        res = res.reindex(columns = get_columns).sort_values('permutation_test_fdr')
        ## record updated parameters
        self.cutoff_prop = cutoff_prop
        self.pval_method = pval_method
        self.pval_cutoff = pval_cutoff
        self.min_cell_number = min_cell_number
        return(res)

    def _auto_cutoff_(self, mat, q = 0.25):
        """
        given a matrix, such as gene-by-cell matrix,
        find 25% percentile value as a cutoff
        meaning that, for example, sensor in cell with lowest 25% expression will be discarded, by default.
        """
        v = []
        for x in mat:
            if np.all(x.toarray() <= 0):
                continue
            xx = x.toarray()
            xx = xx[xx>0]
            v.extend(xx.tolist())
        v = np.array(sorted(v))
        c = np.quantile(v, q)
        return(c)


    def _check_aboundance_(self, cutoff_exp=None, cutoff_met=None):
        """
        check the aboundance of metabolite or transporter expression in cell clusters,
        return the percentage of cells that meet the given cutoff
        by default, cutoff for metabolite aboundance is 0, expression of transporter is 0
        """
        info('Calculating metabolite presence and sensor expression in cell groups')
        ## this will re-write the begin values
        j1 = cutoff_exp is None or cutoff_exp is False
        j2 = self.cutoff_exp is None or self.cutoff_exp is False
        j3 = self.cutoff_exp == 'auto'
        j4 = isinstance(self.cutoff_exp, float) or isinstance(self.cutoff_exp, int)

        if cutoff_exp == 'auto':
            # decide cutoff by taking 75% percentile across all sensor in all cells
            sensor_loc = np.where(self.exp_mat_indexer.isin(self.met_sensor['Gene_name']))[0]
            sensor_mat = self.exp_mat[sensor_loc,:]
            cutoff_exp = self._auto_cutoff_(mat = sensor_mat)
            self.cutoff_exp = cutoff_exp
            info('automated cutoff for sensor expression, cutoff=%s'%cutoff_exp)
        elif j1 and (j2 or j3):
            ## decide cutoff by taking 75% percentile across all sensor in all cells
            sensor_loc = np.where(self.exp_mat_indexer.isin(self.met_sensor['Gene_name']))[0]
            sensor_mat = self.exp_mat[sensor_loc,:]
            cutoff_exp = self._auto_cutoff_(mat = sensor_mat)
            self.cutoff_exp = cutoff_exp
            info('automated cutoff for sensor expression, cutoff=%s'%cutoff_exp)
        elif j1 and j4:
            cutoff_exp = self.cutoff_exp 
            info('provided cutoff for sensor expression, cutoff=%s'%cutoff_exp)
        elif j1 and j2:
            cutoff_exp = 0
            info('cutoff for sensor expression, cutoff=%s'%cutoff_exp)
        else:
            cutoff_exp = 0 if not cutoff_exp else cutoff_exp
            info('cutoff for sensor expression, cutoff=%s'%cutoff_exp)
        ## met 
        j1 = cutoff_met is None or cutoff_met is False
        j2 = self.cutoff_met is None or self.cutoff_met is False
        j3 = self.cutoff_met == 'auto'
        j4 = isinstance(self.cutoff_met, float) or isinstance(self.cutoff_met, int)

        if cutoff_met == 'auto':
            ## decide cutoff by taking 75% percentile across all sensor in all cells
            cutoff_met = self._auto_cutoff_(mat = self.met_mat)
            self.cutoff_met = cutoff_met
            info('automated cutoff for metabolite presence, cutoff=%s'%cutoff_met)
        elif j1 and (j2 or j3):
            ## decide cutoff by taking 75% percentile across all sensor in all cells
            cutoff_met = self._auto_cutoff_(mat = self.met_mat)
            self.cutoff_met = cutoff_met
            info('automated cutoff for metabolite presence, cutoff=%s'%cutoff_met)
        elif j1 and j4:
            cutoff_met = self.cutoff_met 
            info('provided cutoff for metabolite presence, cutoff=%s'%cutoff_met)
        elif j1 and j2:
            cutoff_met = 0
            info('cutoff for metabolite presence, cutoff=%s'%cutoff_met)
        else:
            cutoff_met = 0 if not cutoff_met else cutoff_met
            info('cutoff for metabolite presence, cutoff=%s'%cutoff_met)

        ## expression for all transporters
        sensors = self.met_sensor['Gene_name'].unique().tolist()
        info('cutoff_exp: {}'.format(cutoff_exp))
        
        sensor_loc = {g:i for i,g in enumerate(self.exp_mat_indexer) if g in sensors}
        exp_prop = {}
        for x in self.cell_ann['cell_group'].unique().tolist():
            cells = self.cell_ann[self.cell_ann['cell_group'] == x].index.tolist()
            cell_loc = [i for i, c in enumerate(self.exp_mat_columns) if c in cells]
            s = self.exp_mat[list(sensor_loc.values()),:][:,cell_loc]
            exp_prop[x] = pd.Series([v[v>cutoff_exp].shape[1] / v.shape[1] for v in s],
                                   index = list(sensor_loc.keys()))
        exp_prop = pd.DataFrame.from_dict(exp_prop, orient = 'index')
         
        # ====================== #
        info('cutoff_metabolite: {}'.format(cutoff_met))
        ## metabolite aboundance
        metabolites = self.met_sensor['HMDB_ID'].unique().tolist()
        met_prop = {}
        for x in self.cell_ann['cell_group'].unique().tolist():
            cells = self.cell_ann[self.cell_ann['cell_group'] == x].index.tolist()
            cell_loc = [i for i, c in enumerate(self.met_mat_columns) if c in cells]
            m = self.met_mat[:,cell_loc]
            met_prop[x] = pd.Series([v[v>cutoff_met].shape[1] / v.shape[1] for v in m],
                                   index = self.met_mat_indexer.tolist())
        met_prop = pd.DataFrame.from_dict(met_prop, orient = 'index')
        
        self.cutoff_exp = cutoff_exp
        self.cutoff_met = cutoff_met
        self.exp_prop = exp_prop
        self.met_prop = met_prop ## cell_group x sensor gene, cell_group x metabolite
    
    def _get_gene_exp_(self):
        """
        only sensor and enzyme gene expression are needed for each cells
        """
        sensors = self.met_sensor['Gene_name'].unique().tolist()
        enzymes = []
        for x in self.met_enzyme['gene'].tolist():
            enzymes.extend([i.split('[')[0] for i in x.split('; ')])
        genes = list(set(sensors+enzymes))
        ## gene loc
        gene_loc = np.where(pd.Series(self.exp_mat_indexer).isin(genes))[0]
        
        gene_dat = self.exp_mat[gene_loc].copy()
        ## update the exp_mat and indexer
        self.exp_mat = sparse.csr_matrix(gene_dat)
        self.exp_mat_indexer = self.exp_mat_indexer[gene_loc]
                                   
    def _avg_by_group_(self):
        ## avg exp by cell_group for met sensor
        group_names = self.cell_ann['cell_group'].unique().tolist()
        avg_exp = np.empty(shape = (self.exp_mat.shape[0],0)) ## save exp data

        for x in group_names:
            cells = self.cell_ann[self.cell_ann['cell_group'] == x].index.tolist()
            cell_loc = np.where(pd.Series(self.exp_mat_columns).isin(cells))[0]
            # arithmatic mean
            avg_exp = np.concatenate((avg_exp, self.exp_mat[:,cell_loc].mean(axis = 1)), axis = 1)
        
        self.avg_exp = sparse.csr_matrix(avg_exp)
        self.avg_exp_indexer = np.array(self.exp_mat_indexer)
        self.avg_exp_columns = np.array(group_names)
    
    
    def _avg_met_group_(self):
        """
        take average of sensor expression and metabolite by cell groups
        """
        ## avg met by cell_group for met
        avg_met = np.empty(shape = (self.met_mat.shape[0],0)) ## save exp data
        group_names = self.cell_ann['cell_group'].unique().tolist()

        for x in group_names:
            cells = self.cell_ann[self.cell_ann['cell_group'] == x].index.tolist()
            cell_loc = np.where(pd.Series(self.met_mat_columns).isin(cells))[0]
            ## mean
            avg_met = np.concatenate((avg_met, self.met_mat[:,cell_loc].mean(axis = 1)), axis = 1)

        self.avg_met = sparse.csr_matrix(avg_met)
        self.avg_met_indexer = np.array(self.met_mat_indexer)
        self.avg_met_columns = group_names
            
## ============================== constrain by flux ============================
    def _matchMetName_(self, met_name_list = []):
        """
        met_name is a list of metabolite names to match with HMDB standard names
        """
        met_ann = self.met_ann[['metabolite', 'synonyms_name']]
        met_syn_dict = {}
        ## build a dict key is alias and value is standard name
        for mn, sn in met_ann.dropna().values.tolist():
            for ms in sn.split('; '):
                met_syn_dict[ms] = mn
        ## match names
        met_matched = {m:met_syn_dict.get(m) for m in met_name_list if m in met_syn_dict}
        return(met_matched)
            
    def _ConstrainFluxFromAnyTool_(self,
                            efflux_mat,
                            influx_mat,
                            efflux_cut = 'auto', 
                            influx_cut = 'auto',
                            norm=False,
                            inplace=True
                            ):
        """
        constraint efflux and influx for mCCC events based FBA results from any tools
        efflux_mat and influx_mat should be provided as data frames with rows for metabolite names, columns for cell group
        """
        comm_res = self.commu_res.sort_values(['Sender', 'Receiver', 'Metabolite', 'Sensor']).copy()
        comm_res.index = range(comm_res.shape[0])
        cg_all = list(set(comm_res['Sender'].tolist()+comm_res['Receiver'].tolist()))
        info('Match cell groups')
        influx_cg = influx_mat.columns.tolist()
        ## check if all cell group in mCCC table in influx or efflux table
        influx_cg_check = np.all([x in influx_mat.columns.tolist() for x in cg_all])
        efflux_cg = efflux_mat.columns.tolist()
        efflux_cg_check = np.all([x in efflux_mat.columns.tolist() for x in cg_all])
        if not (influx_cg_check and efflux_cg_check):
            raise KeyError('Error: the cell group in commu_res does match in efflux or influx matrix')
    
        info('Match metabolites')
        influx_met = influx_mat.index.tolist()
        efflux_met = efflux_mat.index.tolist()
        influx_met_match = self._matchMetName_(met_name_list = influx_met)
        efflux_met_match = self._matchMetName_(met_name_list = efflux_met)
        if len(influx_met_match) == 0:
            raise KeyError('Error: No metabolite in influx matrix can match with commu_res')
        if len(efflux_met_match) == 0:
            raise KeyError('Error: No metabolite in efflux matrix can match with commu_res')
        ## rename efflux and influx matrix by standard name
        met_known = self.met_ann['metabolite'].unique().tolist()
        efflux_mat.index = [efflux_met_match.get(x, x) if x not in met_known else x for x in efflux_mat.index.tolist()]
        influx_mat.index = [influx_met_match.get(x, x) if x not in met_known else x for x in influx_mat.index.tolist()]
        ## concate to commu_res
        x1 = 'sender_transport_flux'
        x2 = 'receiver_transport_flux'
        if norm:
            flux_norm = lambda x: (x/np.abs(x)) * np.sqrt(np.abs(x)) if x != 0 else 0
            comm_res[x1] = [flux_norm(efflux_mat.loc[m,c].max()) if m in efflux_mat.index.tolist() else np.nan for c, m in comm_res[['Sender', 'Metabolite_Name']].values.tolist()]
            comm_res[x2] = [flux_norm(influx_mat.loc[m,c].max()) if m in influx_mat.index.tolist() else np.nan for c, m in comm_res[['Receiver', 'Metabolite_Name']].values.tolist()]
            if efflux_cut == 'auto':
                all_efflux = [flux_norm(efflux_mat.loc[m,c].max()) if m in efflux_mat.index.tolist() else np.nan for c, m in self.original_result[['Sender', 'Metabolite_Name']].values.tolist()]
                efflux_cut = np.nanpercentile(all_efflux, 25)
            if influx_cut == 'auto':
                all_influx = [flux_norm(influx_mat.loc[m,c].max()) if m in influx_mat.index.tolist() else np.nan for c, m in self.original_result[['Receiver', 'Metabolite_Name']].values.tolist()]
                influx_cut = np.nanpercentile(all_influx, 25)
        else:
            comm_res[x1] = [efflux_mat.loc[m,c].max() if m in efflux_mat.index.tolist() else np.nan for c, m in comm_res[['Sender', 'Metabolite_Name']].values.tolist()]
            comm_res[x2] = [influx_mat.loc[m,c].max() if m in influx_mat.index.tolist() else np.nan for c, m in comm_res[['Receiver', 'Metabolite_Name']].values.tolist()]
            if efflux_cut == 'auto':
                all_efflux = [efflux_mat.loc[m,c].max() if m in efflux_mat.index.tolist() else np.nan for c, m in self.original_result[['Sender', 'Metabolite_Name']].values.tolist()]
                efflux_cut = np.nanpercentile(all_efflux, 25)
            if influx_cut == 'auto':
                all_influx = [influx_mat.loc[m,c].max() if m in influx_mat.index.tolist() else np.nan for c, m in self.original_result[['Receiver', 'Metabolite_Name']].values.tolist()]
                influx_cut = np.nanpercentile(all_influx, 25)        
        print('efflux_cut:', efflux_cut)
        print('influx_cut:', influx_cut)
        
        ## base_efflux_influx_cut
        tmp_na = comm_res[pd.isna(comm_res[x1]) | pd.isna(comm_res[x2])] # some metabolite not in flux result, retain
        ## non-receptor sensor, as long as it not simply as receptor
        tmp1 = comm_res.query('Annotation != "Receptor"').copy()
        ## receptor sensor, when it is only as receptor
        tmp2 = comm_res.query('Annotation == "Receptor"').copy()
        ## apply filter
        tmp1_new = tmp1[(tmp1[x1]>efflux_cut) & (tmp1[x2]>influx_cut)] ## non-receptor sensors, consider influx and efflux
        tmp2_new = tmp2[(tmp2[x1]>efflux_cut)] ## receptor sensors, only consider efflux
        indexs = tmp_na.index.tolist()+tmp1_new.index.tolist()+tmp2_new.index.tolist()
        #tmp1 = tmp1[(tmp1[x1]>efflux_cut) & (tmp1[x2]>influx_cut)] ## non-receptor sensors, consider influx and efflux
        #tmp2 = tmp2[(tmp2[x1]>efflux_cut)] ## receptor sensors, only consider efflux
        #indexs = tmp_na.index.tolist()+tmp1.index.tolist()+tmp2.index.tolist()
        tmp_other = comm_res[~comm_res.index.isin(indexs)] ## other mCCC: significant coexpression but inferred flux does not pass the cutoff
        ## add labels for Flux_PASS
        tmp_na['Flux_PASS'] = 'N/A'
        #tmp1['Flux_PASS'] = 'PASS'
        #tmp2['Flux_PASS'] = 'PASS'
        tmp1_new['Flux_PASS'] = 'PASS'
        tmp2_new['Flux_PASS'] = 'PASS'
        tmp_other['Flux_PASS'] = 'UNPASS'
        ## re-concat result
        update_commu_res = pd.concat([tmp1_new, tmp2_new, tmp_na, tmp_other])
        if inplace:
            self.efflux_mat = efflux_mat
            self.influx_mat = influx_mat
            self.commu_res = update_commu_res.copy()
        else:
            return(update_commu_res)

    def _ConstrainCompassFlux_(self, compass_folder, efflux_cut = 'auto', influx_cut='auto', inplace=True):
        """
        a function to filter out communications with low efflux and influx rates based on COMPASS output, the commu_res will be replaced by updated table
        Params
        -----
        compass_folder: a string for folder path or a dict {condition: path}. The path indicates COMPASS output folder. The output folder should include secretions.tsv and uptake.tsv for cell group level.
        efflux_cut: a numeric efflux threshold to indicate active efflux event. Default sets to 'auto', which determines the threshold by taking 25th percentile of COMPASS values after square root transfermation ((x/np.abs(x)) * np.sqrt(np.abs(x)))
        influx_cut: a numeric ifflux threshold to indicate active influx event. Default sets to 'auto', which determines the threshold by taking 25th percentile of COMPASS values after square root transfermation ((x/np.abs(x)) * np.sqrt(np.abs(x)))
        inplace: True for updating the commu_res in the object, False for return the updated communication table without changing the mebo_obj
        """
        comm_res = self.commu_res.sort_values(['Sender', 'Receiver', 'Metabolite', 'Sensor'])
        comm_res.index = range(comm_res.shape[0])
        ## compass
        efflux_mat, influx_mat = FBA._get_compass_flux_(compass_folder=compass_folder, 
                                           compass_met_ann_path=_read_config(self.config_path)['common']['compass_met_ann_path'], 
                                                                  met_ann=self.met_ann)
        # self._get_compass_flux_(compass_folder = compass_folder)
        x1 = 'sender_transport_flux'
        x2 = 'receiver_transport_flux'
        comm_res[x1] = [efflux_mat.loc[m,c] if m in efflux_mat.index.tolist() else np.nan for c, m in comm_res[['Sender', 'Metabolite_Name']].values.tolist()]
        comm_res[x2] = [influx_mat.loc[m,c] if m in influx_mat.index.tolist() else np.nan for c, m in comm_res[['Receiver', 'Metabolite_Name']].values.tolist()]
        flux_norm = lambda x: (x/np.abs(x)) * np.sqrt(np.abs(x)) if x != 0 else 0
        comm_res[x1] = [flux_norm(x) for x in comm_res[x1].tolist()]
        comm_res[x2] = [flux_norm(x) for x in comm_res[x2].tolist()]
        if efflux_cut == 'auto':
            all_efflux = [flux_norm(efflux_mat.loc[m,c]) if m in efflux_mat.index.tolist() else np.nan for c, m in self.original_result[['Sender', 'Metabolite_Name']].values.tolist()]
            efflux_cut = np.nanpercentile(all_efflux, 25)
        if influx_cut == 'auto':
            all_influx = [flux_norm(influx_mat.loc[m,c]) if m in influx_mat.index.tolist() else np.nan for c, m in self.original_result[['Receiver', 'Metabolite_Name']].values.tolist()]
            influx_cut = np.nanpercentile(all_influx, 25)
        print('efflux_cut:', efflux_cut)
        print('influx_cut:', influx_cut)
        ## base_efflux_influx_cut
        tmp_na = comm_res[pd.isna(comm_res[x1]) | pd.isna(comm_res[x2])] # some metabolite not in flux result, retain
        ## non-receptor sensor, as long as it not simply as receptor
        tmp1 = comm_res.query('Annotation != "Receptor"').copy()
        ## receptor sensor, when it is only as receptor
        tmp2 = comm_res.query('Annotation == "Receptor"').copy()
        ## apply filter
        tmp1_new = tmp1[(tmp1[x1]>efflux_cut) & (tmp1[x2]>influx_cut)] ## non-receptor sensors, consider influx and efflux
        tmp2_new = tmp2[(tmp2[x1]>efflux_cut)] ## receptor sensors, only consider efflux
        indexs = tmp_na.index.tolist()+tmp1_new.index.tolist()+tmp2_new.index.tolist()
        #tmp1 = tmp1[(tmp1[x1]>efflux_cut) & (tmp1[x2]>influx_cut)] ## non-receptor sensors, consider influx and efflux
        #tmp2 = tmp2[(tmp2[x1]>efflux_cut)] ## receptor sensors, only consider efflux
        #indexs = tmp_na.index.tolist()+tmp1.index.tolist()+tmp2.index.tolist()
        tmp_other = comm_res[~comm_res.index.isin(indexs)] ## other mCCC: significant coexpression but inferred flux does not pass the cutoff
        ## add labels for Flux_PASS
        tmp_na['Flux_PASS'] = 'N/A'
        #tmp1['Flux_PASS'] = 'PASS'
        #tmp2['Flux_PASS'] = 'PASS'
        tmp1_new['Flux_PASS'] = 'PASS'
        tmp2_new['Flux_PASS'] = 'PASS'
        tmp_other['Flux_PASS'] = 'UNPASS'
        ## re-concat result
        update_commu_res = pd.concat([tmp1_new, tmp2_new, tmp_na, tmp_other])
        if inplace:
            self.efflux_mat = efflux_mat
            self.influx_mat = influx_mat
            self.commu_res = update_commu_res.copy()
        else:
            return(update_commu_res)
            

    def infer_commu(self, 
                      n_shuffle = 1000,
                      seed = 12345, 
                      Return = True, 
                      thread = None,
                      save_permuation = False,
                      pval_method='permutation_test_fdr',
                      pval_cutoff=0.05,
                      min_cell_number = 50
                     ):
        """
        execute mebocost to infer communications

        Params
        -----
        n_shuffle
            int, number of cell label shuffling for generating null distribution when calculating p-value
            
        seed
            int, a random seed for shuffling cell labels, set seed to get reproducable shuffling result 
            
        Return
            True or False, set True to return the communication event in a data frame
            
        thread
            int, the number of cores used in the computing, default None, thread set when create the object has the highest priority to be considered, so only set thread here if you want to make a change
            
        save_permuation
            True or False, set True to save the communication score for each permutation, this could occupy a higher amount of space when saving out, so default is False
        pval_method
            should be one of ['ttest_pval', 'ranksum_test_pval', 'permutation_test_pval', 'ttest_fdr', 'ranksum_test_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        pval_cutoff
            float, set to filter out non-significant communication events
        min_cell_number
            int, the cell groups will be excluded and p-value will be replaced to 1 if there are not enough number of cells (less than min_cell_number), default is 50

        """
        tic = time.time()
        today = datetime.today().strftime("%B %d, %Y")
        now = datetime.now().strftime("%H:%M:%S")
        current_time = today + ' ' + now
        self.commu_time_stamp = current_time
        self.pval_method = pval_method
        self.pval_cutoff = pval_cutoff
        self.min_cell_number = min_cell_number
        tracemalloc.start()
        ## load config
        self._load_config_()
        
        ## take average by cell group, this must be done before extract sensor and enzyme gene expression of cells
        self._avg_by_group_()
        
        ## extract exp data for sensor and enzyme genes for all cells
        self._get_gene_exp_()
        
        ## estimate metabolite
        self.estimator()
        
        ## avg met mat
        self._avg_met_group_()
    
        # running communication inference
        commu_res_df, commu_res_bg = self.infer(
                                                n_shuffle = n_shuffle, 
                                                seed = seed,
                                                thread = self.thread if thread is None else thread ## allow to set thread in this function
                                                )
        ## update self
        self.commu_res = commu_res_df
        if save_permuation:
            self.commu_bg = commu_res_bg
        
        ## check cell proportion
        self._check_aboundance_()

        ## check low and set p val to 1
        commu_res_df_updated = self._filter_lowly_aboundant_(pvalue_res = commu_res_df,
                                                             cutoff_prop = self.cutoff_prop,
                                                             met_prop=self.met_prop, 
                                                             exp_prop=self.exp_prop,
                                                             pval_method='permutation_test_fdr',
                                                             pval_cutoff=0.05,
                                                             min_cell_number = min_cell_number)
        ## update self
        self.commu_res = commu_res_df_updated[(commu_res_df_updated[pval_method]<pval_cutoff)]
        
        current, peak = tracemalloc.get_traced_memory()
        
        # stopping the library
        tracemalloc.stop()
        
        toc = time.time()
        info('Prediction Done in {:.4f} seconds'.format(toc-tic))
        info('Memory Usage in Peak {:.2f} GB'.format(peak / 1024 / 1024 / 1024))
        if Return:
            return(self.commu_res)

    ## ============================== examining bais to abundant metabolites in the blood ============================
    def _blood_correct_test_(self, 
                            met_cont_file, 
                            commu_score_col = 'Commu_Score', 
                            title = '',
                            show_plot = False,
                            pdf = False):
        """
        A function to regress out effect of high abundant metabolites in blood system. Use this be cautious only when you know it.
        """
        from sklearn.linear_model import LinearRegression
        from scipy.stats import pearsonr, spearmanr
        blood_cont = pd.read_table(met_cont_file, index_col = 0).iloc[:,0]
        commu_res = self.commu_res.copy()
        commu_res['blood_level'] = [np.nan if x not in blood_cont.index.tolist() else blood_cont[x] for x in commu_res['Metabolite_Name'].tolist()]
        commu_res['blood_level'] = np.log(commu_res['blood_level'])
        commu_res['blood_level'] = (commu_res['blood_level'] - commu_res['blood_level'].min()) / (commu_res['blood_level'].max() - commu_res['blood_level'].min())
        commu_res = commu_res[~pd.isna(commu_res['blood_level'])]#[['blood_level', 'Commu_Score']]
        plotm = commu_res.drop_duplicates(['Sender', 'Metabolite_Name'])
        r1, p1 = pearsonr(commu_res['blood_level'], commu_res['Commu_Score'])
        sr1, sp1 = spearmanr(commu_res['blood_level'], commu_res['Commu_Score'])
        model = LinearRegression(fit_intercept = True)
        model.fit(commu_res[['blood_level']], commu_res['Commu_Score'])
        commu_res['pred'] = model.predict(commu_res[['blood_level']])
        commu_res['Corrected_Commu_Score'] = commu_res['Commu_Score'] - commu_res['pred']
        r2, p2 = pearsonr(commu_res['blood_level'], commu_res['Corrected_Commu_Score'])
        sr2, sp2 = spearmanr(commu_res['blood_level'], commu_res['Corrected_Commu_Score'])
        if show_plot:
            fig, ax = plt.subplots(figsize = (10, 4), nrows = 1, ncols = 2)
            sns.regplot(data = commu_res,
                        x = 'blood_level', y = 'Commu_Score', ci = False, ax = ax[0],
                       scatter_kws={'alpha':.5})
            # ax[0].set_title('PCC: %.2f, p-val: %.2e\nSp Rho: %.2f, p-val:%.2e'%(r1, p1, sr1, sp1))
            ax[0].set_title('PCC: %.2f, p-val: %.2e'%(r1, p1))
            ax[0].set_xlabel('Metabolite level in blood')
            ax[0].set_ylabel('mCCC score')
            sns.regplot(data = commu_res,
                        x = 'blood_level', y = 'Corrected_Commu_Score', ci = False, ax = ax[1],
                       scatter_kws={'alpha':.5})
            # ax[1].set_title('PCC: %.2f, p-val: %.2eSp Rho: %.2f, p-val:%.2e'%(r2, p2, sr2, sp2))
            ax[1].set_title('PCC: %.2f, p-val: %.2e'%(r2, p2))
            ax[1].set_xlabel('Metabolite level in blood')
            ax[1].set_ylabel('Corrected mCCC score')
            # ax.set_ylabel('Communication score')
            fig.suptitle(title)
            sns.despine()
            plt.tight_layout()
            pdf.savefig(fig) if pdf else plt.show()
            plt.close()
        return(commu_res)
        
## ============================= differential communication analyisis =====================
    def CommDiff(self,
                comps=[],
                sig_mccc_only = True,
                flux_pass = True,
                thread = 8,
                Return=False
                ):
        """
        differential communication function

        Param:
        -----
        comps: a list, format should be cond1_vs_cond2, for example, Tumor_vs_Normal will indicate
                         to compare mCCC using Tumor vs. Normal.
        sig_mccc_only
            set True to focus on significant mCCC events in either condition for differential analysis.
        flux_pass
            set True to focus on mCCC that pass the flux constrain in the differential analysis, this is effective only when sig_mccc_only is True. Set False to plot all mCCC with significant enzyme-sensor coexpression. Default True.
        thread
            number of threads used in computing, default is 8

        Return
        ----
        if Return set to True, a dict will be return to show the diff mCCC table in each comparison
        """
        tracemalloc.start()
        tic = time.time()
        ## check original_result, commu_res, and commu_bg in the object
        try:
            original_result=self.original_result
            commu_bg=self.commu_bg
        except:
            raise KeyError('cannot find original_result or commu_bg in object, please re-run the infer_commu function and set save_permuation=True')
        if len(self.commu_bg) == 0:
            raise KeyError('commu_bg is empty, please re-run the infer_commu function and set save_permuation=True')
            
        diff_res = collections.defaultdict()
        for comp in comps:
            info('Diff Comm for {}'.format(comp))
            if '_VS_' in comp:
                comp = comp.replace('_VS_', '_vs_')
            ## 
            commu_res = self.commu_res.copy()
            if sig_mccc_only:
                if flux_pass == True:
                    if 'Flux_PASS' in commu_res.columns.tolist():
                        commu_res = commu_res[commu_res['Flux_PASS'] == 'PASS']
                    else:
                        info('flux_pass set to True but NO Flux_PASS column, so skip.')
            else:
                commu_res = pd.DataFrame()
                    
            diff_res[comp] = CD.DiffComm(cell_ann=self.cell_ann,
                                        condition_col=self.condition_col,
                                        group_col=self.group_col,
                                        original_result=self.original_result,
                                        commu_res = commu_res,
                                        commu_bg=self.commu_bg,
                                        prop_cut = self.cutoff_prop,
                                        comparison=comp, thread = thread)
        self.diffcomm_res = diff_res
        
        # stopping the library
        toc = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        info('Diff mCCC Analysis Done in {:.4f} seconds'.format(toc-tic))
        info('Memory Usage in Peak {:.2f} GB'.format(peak / 1024 / 1024 / 1024))
        
        if Return:
            return(diff_res)
            
    def filtermccc(self, and_or,
                mccc_df,
                metabolite_focus,
                sender_focus,
                receiver_focus,
                sensor_focus,
                ):
        mccc_df = pd.DataFrame(mccc_df)
        if and_or == 'and':
            if sender_focus:
                mccc_df = mccc_df[(mccc_df['Sender'].isin(sender_focus))]
            if receiver_focus:
                mccc_df = mccc_df[(mccc_df['Receiver'].isin(receiver_focus))]
            if metabolite_focus:
                mccc_df = mccc_df[(mccc_df['Metabolite_Name'].isin(metabolite_focus))]
            if sensor_focus:
                mccc_df = mccc_df[(mccc_df['Sensor'].isin(sensor_focus))]
        else:
            if sender_focus or receiver_focus or metabolite_focus or sensor_focus:
                mccc_df = mccc_df[(mccc_df['Sender'].isin(sender_focus)) |
                                         (mccc_df['Receiver'].isin(receiver_focus)) |
                                         (mccc_df['Metabolite_Name'].isin(metabolite_focus)) |
                                         (mccc_df['Sensor'].isin(sensor_focus))]
        #comp_res = [line.to_dict() for i, line in mccc_df.iterrows()]
    
        return(mccc_df)

    def DiffSummaryPlot(self, comp_cond, 
                        pval_method='permutation_test_fdr',
                        pval_cutoff=0.05,
                        Log2FC_threshold = 0,
                        sender_focus = [],
                        metabolite_focus = [],
                        sensor_focus = [],
                        receiver_focus = [],
                        and_or = 'and',
                        numtop_bar = 5, 
                        save=None, 
                        return_fig = False
                        ):
        """
        Summary plot to show top sender, receiver, and metabolite with the highest number of up or down differential mCCC events.

        Params
        ------
        comp_cond
            a string, comparison between two conditions, e.g. cond1_vs_cond2
        pval_method
            should be one of ['ttest_pval', 'permutation_test_pval', 'ttest_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        pval_cutoff
            pvalue cutoff for pval_method to define a significant differential event, default 0.05
        Log2FC_threshold
            cutoff of log2 fold change to define up and down differential event, defualt 0
        sender_focus
            a list, set a list of sender cells to be focused, only plot related communications
        metabolite_focus
            a list, set a list of metabolites to be focused, only plot related communications
        sensor_focus
            a list, set a list of sensors to be focused, only plot related communications
        receiver_focus
            a list, set a list of receiver cells to be focused, only plot related communications
        and_or
            eithor 'and' or 'or', 'and' for finding communications that meet to all focus, 'or' for union
        numtop_bar
            number of top ranked sender, receiver, and metabolite with the highest number of differential mCCC events to be shown, default is 5.
        """

        comp_plot_df = self.diffcomm_res[comp_cond].copy()
        cond1, cond2 = comp_cond.split('_vs_') 
    
        ## update comp_df
        if metabolite_focus or sender_focus or receiver_focus or sensor_focus:
            comp_plot_df = self.filtermccc(and_or=and_or,
                                    mccc_df=comp_plot_df,
                                    metabolite_focus=metabolite_focus,
                                    sender_focus=sender_focus,
                                    receiver_focus=receiver_focus,
                                    sensor_focus=sensor_focus,
                                    )        
        ## plot number of cell type and metabolite with diff mCCC events
        # --- Up-regulated events (Log2FC > 0) ---
        Log2FC_threshold = abs(Log2FC_threshold)
        up_mccc = comp_plot_df[(comp_plot_df[pval_method] < pval_cutoff) & (comp_plot_df['Log2FC'] > Log2FC_threshold)]
        
        sender_up = up_mccc.groupby('Sender').size()
        receiver_up = up_mccc.groupby('Receiver').size()
        met_up = up_mccc.groupby('Metabolite_Name').size()

        sender_up = sender_up.sort_values().tail(numtop_bar)
        receiver_up = receiver_up.sort_values().tail(numtop_bar)
        met_up = met_up.sort_values().tail(numtop_bar)

        # --- Down-regulated events (Log2FC < 0) ---
        down_mccc = comp_plot_df[(comp_plot_df[pval_method] < pval_cutoff) & (comp_plot_df['Log2FC'] < -Log2FC_threshold)]

        sender_down = down_mccc.groupby('Sender').size()
        receiver_down = down_mccc.groupby('Receiver').size()
        met_down = down_mccc.groupby('Metabolite_Name').size()

        sender_down = sender_down.sort_values().tail(numtop_bar)
        receiver_down = receiver_down.sort_values().tail(numtop_bar)
        met_down = met_down.sort_values().tail(numtop_bar)

        # Create a figure with 2 rows and 3 columns (one column for each category)
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 6+(numtop_bar/5)))

        # --- Top Row: Up-regulated (light blue) ---
        # Sender
        axes[0, 0].barh(sender_up.index, sender_up.values, color='salmon')
        axes[0, 0].set_title('Up-regulated: Sender cells')
        axes[0, 0].set_xlabel('# diff mCCC events')
        for bar in axes[0, 0].containers[0]:
            width = bar.get_width()
            axes[0, 0].text(width + 0.1, bar.get_y() + bar.get_height()/2,
                            f'{int(width)}', va='center', ha='left', fontsize=10, color='red')

        # Receiver
        axes[0, 1].barh(receiver_up.index, receiver_up.values, color='salmon')
        axes[0, 1].set_title('Up-regulated: Receiver cells')
        for bar in axes[0, 1].containers[0]:
            width = bar.get_width()
            axes[0, 1].text(width + 0.1, bar.get_y() + bar.get_height()/2,
                            f'{int(width)}', va='center', ha='left', fontsize=10, color='red')

        # Metabolite
        axes[0, 2].barh(met_up.index, met_up.values, color='salmon')
        axes[0, 2].set_title('Up-regulated: Metabolites')
        for bar in axes[0, 2].containers[0]:
            width = bar.get_width()
            axes[0, 2].text(width + 0.1, bar.get_y() + bar.get_height()/2,
                            f'{int(width)}', va='center', ha='left', fontsize=10, color='red')

        # --- Bottom Row: Down-regulated (salmon) ---
        # Sender
        axes[1, 0].barh(sender_down.index, sender_down.values, color='lightblue')
        axes[1, 0].set_title('Down-regulated: Sender cells')
        axes[1, 0].set_xlabel('# diff mCCC events')
        for bar in axes[1, 0].containers[0]:
            width = bar.get_width()
            axes[1, 0].text(width + 0.1, bar.get_y() + bar.get_height()/2,
                            f'{int(width)}', va='center', ha='left', fontsize=10, color='blue')

        # Receiver
        axes[1, 1].barh(receiver_down.index, receiver_down.values, color='lightblue')
        axes[1, 1].set_title('Down-regulated: Receiver cells')
        for bar in axes[1, 1].containers[0]:
            width = bar.get_width()
            axes[1, 1].text(width + 0.1, bar.get_y() + bar.get_height()/2,
                            f'{int(width)}', va='center', ha='left', fontsize=10, color='blue')

        # Metabolite
        axes[1, 2].barh(met_down.index, met_down.values, color='lightblue')
        axes[1, 2].set_title('Down-regulated: Metabolites')
        for bar in axes[1, 2].containers[0]:
            width = bar.get_width()
            axes[1, 2].text(width + 0.1, bar.get_y() + bar.get_height()/2,
                            f'{int(width)}', va='center', ha='left', fontsize=10, color='blue')

        sns.despine()
        plt.tight_layout()
        
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
            Pdf.savefig(fig)
            Pdf.close()
            plt.close()
        if return_fig:
            return(fig)


    def DiffFlowPlot(self,
                    comp_cond,
                    pval_method='permutation_test_fdr',
                    pval_cutoff=0.05,
                    Log2FC_threshold = 0,
                    sender_focus = [],
                    metabolite_focus = [],
                    sensor_focus = [],
                    receiver_focus = [],
                    remove_unrelevant = False,
                    and_or = 'and',
                    node_label_size = 8,
                    node_alpha = .8,
                    figsize = 'auto',
                    node_cmap = 'Set1',
                    line_color_col = 'Log2FC',
                    line_cmap = 'coolwarm',
                    line_cmap_vmin = None,
                    line_cmap_vmax = None,
                    line_cmap_center = None,
                    linewidth = 1.5,
                    node_size_norm = (10, 150),
                    node_value_range = None,
                    save=None, 
                    save_plot = False, 
                    show_plot = True,
                    text_outline = False,
                    return_fig = False):
        """
        plot diff flow plot, the line color reflects log2 fold change, by default, only significant diff mCCC be plotted

        Params
        ------
        comp_cond
            a string, comparison between two conditions, e.g. cond1_vs_cond2
        pval_method
            should be one of ['ttest_pval', 'permutation_test_pval', 'ttest_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        pval_cutoff
            float, set to filter out non-significant communication events
        Log2FC_threshold
            cutoff of log2 fold change to define up and down differential event, defualt 0
        sender_focus
            a list, set a list of sender cells to be focused, only plot related communications
        metabolite_focus
            a list, set a list of metabolites to be focused, only plot related communications
        sensor_focus
            a list, set a list of sensors to be focused, only plot related communications
        receiver_focus
            a list, set a list of receiver cells to be focused, only plot related communications
        conditions
            a list of condition name, set to focus on mCCC in those conditions. Only use this when you run multiple samples with condition labels.
        remove_unrelevant
            True or False, set True to hide unrelated nodes 
        and_or
            eithor 'and' or 'or', 'and' for finding communications that meet to all focus, 'or' for union
        node_label_size
            float, font size of text label on node, default will be 8
        node_alpha
            float, set to transparent node color
        figsize
            auto or a tuple of float such as (5.5, 4.2), defualt will be automatically estimate
        node_cmap
            node color map or a four-element list, used to color sender, metabolite, sensor, receiver, set one from https://matplotlib.org/stable/tutorials/colors/colormaps.html
        line_color_col
            set a column with floats to draw line colors, default Log2FC.
        line_cmap
            line color map, used to indicate the communication score, set one from https://matplotlib.org/stable/tutorials/colors/colormaps.html
        node_size_norm
            two values in a tuple, used to normalize the dot size, such as (10, 150)
        linewidth_norm
            two values in a tuple, used to normalize the line width, such as (0.1, 1)
        save
            str, the file name to save the figure
        show_plot
            True or False, whether print the figure on the screen
        comm_score_col
            column name of communication score, can be Commu_Score
        comm_score_cutoff
            a float, set a cutoff so only communications with score greater than the cutoff will be focused
        cutoff_prop
            a float between 0 and 1, set a cutoff to filter out lowly abundant cell populations by the fraction of cells expressed sensor genes or metabolite, Note that this parameter will lost the function if cutoff_prop was set lower than the one user set at begaining of running mebocost.infer_commu or preparing mebocost object. This parameter were designed to further strengthen the filtering.
        return_fig:
            True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself
        """
        comm_res = self.diffcomm_res[comp_cond].copy()
        cond1, cond2 = comp_cond.split('_vs_')        
        ## pdf
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None

        fig = CP._DiffFlowPlot_(comm_res = comm_res, pval_method=pval_method, pval_cutoff=pval_cutoff, Log2FC_threshold=Log2FC_threshold,
                    sender_focus = sender_focus, metabolite_focus = metabolite_focus, sensor_focus = sensor_focus,
                    receiver_focus = receiver_focus, remove_unrelevant = remove_unrelevant, and_or = and_or,
                    node_label_size = node_label_size, node_alpha = node_alpha, figsize = figsize,
                    node_cmap = node_cmap, line_color_col = line_color_col, line_cmap = line_cmap, 
                    line_cmap_vmin = line_cmap_vmin, line_cmap_vmax = line_cmap_vmax, line_cmap_center = line_cmap_center,
                    linewidth = linewidth, node_size_norm = node_size_norm, node_value_range = node_value_range,
                    pdf=Pdf, save_plot = save_plot, show_plot = show_plot, text_outline = text_outline, return_fig = return_fig)
        
        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        if return_fig:
            return(fig)
            
    def CompScatterPlot(self, comp_cond, 
                        pval_method = 'permutation_test_fdr',
                        pval_threshold = 0.05,
                        Log2FC_threshold = 0,
                        sender_focus = [],
                        metabolite_focus = [],
                        sensor_focus = [],
                        receiver_focus = [],
                        and_or = 'and',
                        show_plot = True,
                        figsize = (5.5, 4),
                        return_fig = False,
                        save = None
                       ):
        """
        generate a scatter plot to compare communication scores of two samples

        Params
        ------
        comp_cond
            a string, comparison between two conditions, e.g. cond1_vs_cond2
        pval_method
            should be one of ['ttest_pval', 'permutation_test_pval', 'ttest_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        pval_cutoff
            float, set to filter out non-significant communication events
        Log2FC_threshold
            cutoff of log2 fold change to define up and down differential event, defualt 0
        sender_focus
            a list, set a list of sender cells to be focused, only plot related communications
        metabolite_focus
            a list, set a list of metabolites to be focused, only plot related communications
        sensor_focus
            a list, set a list of sensors to be focused, only plot related communications
        receiver_focus
            a list, set a list of receiver cells to be focused, only plot related communications
        and_or
            eithor 'and' or 'or', 'and' for finding communications that meet to all focus, 'or' for union
        show_plot
            True or False, whether print the figure on the screen
        figsize
            auto or a tuple of float such as (5.5, 4.2)
        return_fig:
            True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself
        save
            str, the file name to save the figure
        """
        if comp_cond not in self.diffcomm_res:
            raise KeyError('%s is not in the comparision results!'%comp_cond)

        ## pdf
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None
        
        if '_VS_' in comp_cond:
            comp_cond = comp_cond.replace('_VS_', '_vs_')
        cond1, cond2 = comp_cond.split('_vs_')
        
        comm_res = self.diffcomm_res[comp_cond].copy()
        if metabolite_focus or sender_focus or receiver_focus or sensor_focus:
            comm_res = self.filtermccc(and_or=and_or,
                                    mccc_df=comm_res,
                                    metabolite_focus=metabolite_focus,
                                    sender_focus=sender_focus,
                                    receiver_focus=receiver_focus,
                                    sensor_focus=sensor_focus,
                                    )
            
        fig = CP.CompScatterPlot(comm_res, cond1 = cond1, cond2 = cond2, pval_method = pval_method,
                    pval_threshold = pval_threshold, Log2FC_threshold = Log2FC_threshold, 
                    show_plot = show_plot, figsize = figsize,
                    return_fig = return_fig, save = Pdf, title = comp_cond)
        
        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        if return_fig:
            return(fig)
 
## ============================== communication plot functions ============================
    def eventnum_bar(self,
                    sender_focus = [],
                    metabolite_focus = [],
                    sensor_focus = [],
                    receiver_focus = [],
                    conditions = [],
                    and_or = 'and',
                    xorder = [],
                    flux_pass = True,
                    pval_method = 'permutation_test_fdr',
                    pval_cutoff = 0.05,
                    comm_score_col = 'Commu_Score',
                    comm_score_cutoff = None,
                    cutoff_prop = None,
                    figsize = 'auto',
                    save = None,
                    show_plot = True,
                    show_num = True,
                    include = ['sender-receiver', 'sensor', 'metabolite', 'metabolite-sensor'],
                    group_by_cell = True,
                    colorcmap = 'tab20',
                    return_fig = False
                  ):
        """
        this function summarize the number of communication events
        
        Params
        ------
        sender_focus
            a list, set a list of sender cells to be focused, only plot related communications
        metabolite_focus
            a list, set a list of metabolites to be focused, only plot related communications
        sensor_focus
            a list, set a list of sensors to be focused, only plot related communications
        receiver_focus
            a list, set a list of receiver cells to be focused, only plot related communications
        conditions
            a list of condition name, set to focus on mCCC in those conditions. Only use this when you run multiple samples with condition labels.
        and_or
            eithor 'and' or 'or', 'and' for finding communications that meet to all focus, 'or' for union
        xorder
            a list to order the x axis
        flux_pass
            set True to focus on mCCC that pass the flux constrain, set False to plot all mCCC. Default True.
        pval_method
            should be one of ['ttest_pval', 'permutation_test_pval', 'ttest_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        pval_cutoff
            float, set to filter out non-significant communication events
        figsize
            auto or a tuple of float such as (5.5, 4.2), defualt will be automatically estimate
        save
            str, the file name to save the figure
        show_plot
             True or False, whether print the figure on the screen
        show_num
            True or False, whether label y-axis value to the top of each bar
        comm_score_col
            column name of communication score, can be Commu_Score
        comm_score_cutoff
            a float, set a cutoff so only communications with score greater than the cutoff will be focused
        cutoff_prop
            a float between 0 and 1, set a cutoff to further filter out lowly abundant cell populations by the fraction of cells expressed sensor genes or metabolite, Note that this parameter will lost the function if cutoff_prop was set lower than the one user set at begaining of running mebocost.infer_commu or preparing mebocost object. This parameter were designed to further strengthen the filtering.
        include
            a list, contains one or more elements from ['sender-receiver', 'sensor', 'metabolite', 'metabolite-sensor'], we try to summarize the number of communications grouping by the given elements, if return_fig set to be True, only provide one for each run.
        group_by_cell
            True or False, only effective for metabolite and sensor summary, True to further label number of communications in cell groups, False to do not do that
        colormap
            only effective when group_by_cell is True, should be a python camp str, default will be 'tab20', or can be a dict where keys are cell group, values are RGB readable color
        return_fig:
            True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself.
            
        """
        
#         if show_plot is None and self.show_plot is not None:
#             show_plot = self.show_plot
        
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None

        commu_res = self.commu_res.copy()
        if flux_pass == True:
            if 'Flux_PASS' in commu_res.columns.tolist():
                commu_res = commu_res[commu_res['Flux_PASS'] == 'PASS']
            else:
                info('flux_pass set to True but NO Flux_PASS column, so skip.')
            
        if conditions:
            indices = commu_res['Condition'].isin(conditions)
            commu_res = commu_res.loc[indices,:]
            
        fig = CP._eventnum_bar_(commu_res = commu_res,
                    sender_focus = sender_focus,
                    metabolite_focus = metabolite_focus,
                    sensor_focus = sensor_focus,
                    receiver_focus = receiver_focus,
                    and_or = and_or,
                    xorder = xorder,
                    pval_method = pval_method,
                    pval_cutoff = pval_cutoff,
                    comm_score_col = comm_score_col,
                    comm_score_cutoff = comm_score_cutoff,
                    cutoff_prop = cutoff_prop,
                    figsize = figsize,
                    pdf = Pdf,
                    show_plot = show_plot,
                    show_num = show_num,
                    include = include,
                    group_by_cell = group_by_cell,
                    colorcmap = colorcmap,
                    return_fig = return_fig
                  )
        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        if return_fig:
            return(fig)
    
    def commu_dotmap(self,
                sender_focus = [],
                metabolite_focus = [],
                sensor_focus = [],
                receiver_focus = [],
                conditions = [],
                and_or = 'and',
                flux_pass = True,
                pval_method='permutation_test_fdr',
                pval_cutoff=0.05, 
                figsize = 'auto',
                cmap = 'Reds',
                cmap_vmin = None,
                cmap_vmax = None,
                cellpair_order = [],
                met_sensor_order = [],
                dot_size_norm = (10, 150),
                save = None, 
                show_plot = True,
                comm_score_col = 'Commu_Score',
                comm_score_range = None,
                comm_score_cutoff = None,
                cutoff_prop = None,
                swap_axis = False,
                return_fig = False):
        """
        commu_dotmap to show all significant communication events
        
        Params
        -----
        sender_focus
            a list, set a list of sender cells to be focused, only plot related communications
        metabolite_focus
            a list, set a list of metabolites to be focused, only plot related communications
        sensor_focus
            a list, set a list of sensors to be focused, only plot related communications
        receiver_focus
            a list, set a list of receiver cells to be focused, only plot related communications
        conditions
            a list of condition name, set to focus on mCCC in those conditions. Only use this when you run multiple samples with condition labels.
        and_or
            eithor 'and' or 'or', 'and' for finding communications that meet to all focus, 'or' for union
        flux_pass
            set True to focus on mCCC that pass the flux constrain, set False to plot all mCCC. Default True.
        pval_method
            should be one of ['ttest_pval', 'permutation_test_pval', 'ttest_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        pval_cutoff
            float, set to filter out non-significant communication events
        figsize
            auto or a tuple of float such as (5.5, 4.2), defualt will be automatically estimate
        cmap
            colormap for dot color, default is Reds
        node_size_norm
            two values in a tuple, used to normalize the dot size, such as (10, 150)
        save
            str, the file name to save the figure
        show_plot
             True or False, whether print the figure on the screen
        comm_score_col
            column name of communication score, can be Commu_Score
        comm_score_cutoff
            a float, set a cutoff so only communications with score greater than the cutoff will be focused
        cutoff_prop
            a float between 0 and 1, set a cutoff to filter out lowly abundant cell populations by the fraction of cells expressed sensor genes or metabolite, Note that this parameter will lost the function if cutoff_prop was set lower than the one user set at begaining of running mebocost.infer_commu or preparing mebocost object. This parameter were designed to further strengthen the filtering.
        return_fig:
            True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself
        """
        
#         if show_plot is None and self.show_plot is not None:
#             show_plot = self.show_plot

        commu_res = self.commu_res.copy()
        if flux_pass == True:
            if 'Flux_PASS' in commu_res.columns.tolist():
                commu_res = commu_res[commu_res['Flux_PASS'] == 'PASS']
            else:
                info('flux_pass set to True but NO Flux_PASS column, so skip.')
                
        ## pdf
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None
        if conditions:
            indices = commu_res['Condition'].isin(conditions)
            commu_res = commu_res.loc[indices,:]
            
        fig = CP._commu_dotmap_(comm_res=commu_res, 
                     sender_focus = sender_focus,
                     metabolite_focus = metabolite_focus,
                     sensor_focus = sensor_focus,
                     receiver_focus = receiver_focus,
                     and_or = and_or,
                     pval_method=pval_method, 
                     pval_cutoff=pval_cutoff,
                     cmap_vmin = cmap_vmin,
                     cmap_vmax = cmap_vmax,
                     cellpair_order = cellpair_order,
                     met_sensor_order = met_sensor_order,
                     figsize = figsize, 
                     comm_score_col = comm_score_col,
                     comm_score_range = comm_score_range,
                     comm_score_cutoff = comm_score_cutoff,
                     cutoff_prop = cutoff_prop,
                     cmap = cmap,
                     dot_size_norm = dot_size_norm,
                     pdf = Pdf, 
                     show_plot = show_plot,
                     swap_axis = swap_axis,
                     return_fig = return_fig
                    )
        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        if return_fig:
            return(fig)
        
    def FlowPlot(self, 
                pval_method='permutation_test_fdr',
                pval_cutoff=0.05,
                sender_focus = [],
                metabolite_focus = [],
                sensor_focus = [],
                receiver_focus = [],
                conditions = [],
                remove_unrelevant = True,
                and_or = 'and',
                flux_pass = True,
                node_label_size = 8,
                node_alpha = .8,
                figsize = 'auto',
                node_cmap = 'Set1',
                line_cmap = 'bwr',
                line_cmap_vmin = None,
                line_cmap_vmax = None,
                linewidth_norm = (0.1, 1),
                linewidth_value_range = None,
                node_size_norm = (10, 150),
                node_value_range = None,
                save=None, 
                show_plot = True,
                comm_score_col = 'Commu_Score',
                comm_score_cutoff = None,
                cutoff_prop = None,
                text_outline = False,
                return_fig = False):
        """
        Flow plot to show the communication connections from sender to metabolite, to sensor, to receiver

        Params
        ------
        pval_method
            should be one of ['ttest_pval', 'permutation_test_pval', 'ttest_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        pval_cutoff
            float, set to filter out non-significant communication events
        sender_focus
            a list, set a list of sender cells to be focused, only plot related communications
        metabolite_focus
            a list, set a list of metabolites to be focused, only plot related communications
        sensor_focus
            a list, set a list of sensors to be focused, only plot related communications
        receiver_focus
            a list, set a list of receiver cells to be focused, only plot related communications
        conditions
            a list of condition name, set to focus on mCCC in those conditions. Only use this when you run multiple samples with condition labels.
        remove_unrelevant
            True or False, set True to hide unrelated nodes 
        and_or
            eithor 'and' or 'or', 'and' for finding communications that meet to all focus, 'or' for union
        flux_pass
            set True to focus on mCCC that pass the flux constrain, set False to plot all mCCC. Default True.
        node_label_size
            float, font size of text label on node, default will be 8
        node_alpha
            float, set to transparent node color
        figsize
            auto or a tuple of float such as (5.5, 4.2), defualt will be automatically estimate
        node_cmap
            node color map or a four-element list, used to color sender, metabolite, sensor, receiver, set one from https://matplotlib.org/stable/tutorials/colors/colormaps.html
        line_cmap
            line color map, used to indicate the communication score, set one from https://matplotlib.org/stable/tutorials/colors/colormaps.html
        node_size_norm
            two values in a tuple, used to normalize the dot size, such as (10, 150)
        linewidth_norm
            two values in a tuple, used to normalize the line width, such as (0.1, 1)
        save
            str, the file name to save the figure
        show_plot
            True or False, whether print the figure on the screen
        comm_score_col
            column name of communication score, can be Commu_Score
        comm_score_cutoff
            a float, set a cutoff so only communications with score greater than the cutoff will be focused
        cutoff_prop
            a float between 0 and 1, set a cutoff to filter out lowly abundant cell populations by the fraction of cells expressed sensor genes or metabolite, Note that this parameter will lost the function if cutoff_prop was set lower than the one user set at begaining of running mebocost.infer_commu or preparing mebocost object. This parameter were designed to further strengthen the filtering.
        return_fig:
            True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself
        """
        
#         if show_plot is None and self.show_plot is not None:
#             show_plot = self.show_plot

        commu_res = self.commu_res.copy()
        if flux_pass == True:
            if 'Flux_PASS' in commu_res.columns.tolist():
                commu_res = commu_res[commu_res['Flux_PASS'] == 'PASS']
            else:
                info('flux_pass set to True but NO Flux_PASS column, so skip.')
                
        ## pdf
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None
        if conditions:
            indices = commu_res['Condition'].isin(conditions)
            commu_res = commu_res.loc[indices,:]
        fig = CP._FlowPlot_(comm_res=commu_res, pval_method=pval_method, pval_cutoff=pval_cutoff, 
                      sender_focus = sender_focus, metabolite_focus = metabolite_focus,
                      sensor_focus = sensor_focus, receiver_focus = receiver_focus, 
                      remove_unrelevant = remove_unrelevant, and_or = and_or,
                      node_label_size = node_label_size, node_alpha = node_alpha, figsize = figsize, 
                      node_cmap = node_cmap, line_cmap = line_cmap, line_cmap_vmin = line_cmap_vmin,
                      line_cmap_vmax = line_cmap_vmax, linewidth_norm = linewidth_norm, 
                      linewidth_value_range = linewidth_value_range, node_value_range = node_value_range,
                      node_size_norm = node_size_norm, pdf=Pdf, show_plot = show_plot, 
                      comm_score_col = comm_score_col, comm_score_cutoff = comm_score_cutoff, cutoff_prop = cutoff_prop,
                      text_outline = text_outline, return_fig = return_fig)
        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        if return_fig:
            return(fig)
            
    def count_dot_plot(self, 
                    conditions = [],
                    flux_pass = True,
                    pval_method='permutation_test_pval', 
                    pval_cutoff=0.05, 
                    cmap='RdBu_r', 
                    figsize = 'auto',
                    save = None,
                    dot_size_norm = (5, 100),
                    dot_value_range = None,
                    dot_color_vmin = None,
                    dot_color_vmax = None,
                    show_plot = True,
                    comm_score_col = 'Commu_Score',
                    comm_score_cutoff = None,
                    cutoff_prop = None,
                    dendrogram_cluster = True,
                    sender_order = [],
                    receiver_order = [],
                    return_fig = False):
        """
        dot plot to show the summary of communication numbers between sender and receiver 

        Params
        -----
        conditions
            a list of condition name, set to focus on mCCC in those conditions. Only use this when you run multiple samples with condition labels.
        flux_pass
            set True to focus on mCCC that pass the flux constrain, set False to plot all mCCC. Default True.
        pval_method
            should be one of ['zztest_pval', 'ttest_pval', 'ranksum_test_pval', 'permutation_test_pval', 'zztest_fdr', 'ttest_fdr', 'ranksum_test_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        pval_cutoff
            float, set to filter out non-significant communication events
        cmap
            color map to set dot color 
        figsize
            auto or a tuple of float such as (5.5, 4.2), defualt will be automatically estimate
        save
            str, the file name to save the figure
        dot_size_norm
            two values in a tuple, used to normalize the dot size, such as (10, 150)
        dot_color_vmin
            float, the value limits the color map in maximum
        dot_color_vmax
            float, the value limits the color map in minimum
        show_plot
            True or False, whether print the figure on the screen
        comm_score_col
            column name of communication score, can be Commu_Score
        comm_score_cutoff
            a float, set a cutoff so only communications with score greater than the cutoff will be focused
        cutoff_prop
            a float between 0 and 1, set a cutoff to filter out lowly abundant cell populations by the fraction of cells expressed sensor genes or metabolite, Note that this parameter will lost the function if cutoff_prop was set lower than the one user set at begaining of running mebocost.infer_commu or preparing mebocost object. This parameter were designed to further strengthen the filtering.
        return_fig:
            True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself
        """
        
        
#         if show_plot is None and self.show_plot is not None:
#             show_plot = self.show_plot

        commu_res = self.commu_res.copy()
        if flux_pass == True:
            if 'Flux_PASS' in commu_res.columns.tolist():
                commu_res = commu_res[commu_res['Flux_PASS'] == 'PASS']
            else:
                info('flux_pass set to True but NO Flux_PASS column, so skip.')
            
        ## pdf
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None
        if conditions:
            indices = commu_res['Condition'].isin(conditions)
            commu_res = commu_res.loc[indices,:]
        fig = CP._count_dot_plot_(commu_res=commu_res, pval_method = pval_method, pval_cutoff = pval_cutoff, 
                        cmap = cmap, figsize = figsize, pdf = Pdf, dot_size_norm = dot_size_norm, dot_value_range = dot_value_range,
                        dot_color_vmin = dot_color_vmin, dot_color_vmax = dot_color_vmax, show_plot = show_plot,
                        comm_score_col = comm_score_col, comm_score_cutoff = comm_score_cutoff, cutoff_prop = cutoff_prop,
                        dendrogram_cluster = dendrogram_cluster,
                        sender_order = sender_order, receiver_order = receiver_order,
                        return_fig = return_fig)
        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        if return_fig:
            return(fig)

    def commu_network_plot(self,
                        sender_focus = [],
                        metabolite_focus = [],
                        sensor_focus = [],
                        receiver_focus = [],
                        conditions = [],
                        flux_pass = True,
                        remove_unrelevant = False,
                        and_or = 'and',
                        pval_method = 'permutation_test_fdr',
                        pval_cutoff = 0.05,
                        node_cmap = 'tab20',
                        figsize = 'auto',
                        line_cmap = 'RdBu_r',
                        line_color_vmin = None,
                        line_color_vmax = None,
                        linewidth_value_range = None,
                        linewidth_norm = (0.1, 1),
                        node_size_norm = (50, 300),
                        node_value_range = None,
                        adjust_text_pos_node = True,
                        node_text_hidden = False,
                        node_text_font = 10,
                        save = None,
                        show_plot = True,
                        comm_score_col = 'Commu_Score',
                        comm_score_cutoff = None,
                        cutoff_prop = None,
                        text_outline = False,
                        return_fig = False):

        """
        Network plot to show the communications between cell groups

        Params
        ------
        sender_focus
            a list, set a list of sender cells to be focused, only plot related communications
        metabolite_focus
            a list, set a list of metabolites to be focused, only plot related communications
        sensor_focus
            a list, set a list of sensors to be focused, only plot related communications
        receiver_focus
            a list, set a list of receiver cells to be focused, only plot related communications
        conditions
            a list of condition name, set to focus on mCCC in those conditions. Only use this when you run multiple samples with condition labels.
        flux_pass
            set True to focus on mCCC that pass the flux constrain, set False to plot all mCCC. Default True.
        remove_unrelevant
            True or False, set True to hide unrelated nodes
        and_or
            eithor 'and' or 'or', 'and' for finding communications that meet to all focus, 'or' for union
        pval_method
            should be one of ['zztest_pval', 'ttest_pval', 'ranksum_test_pval', 'permutation_test_pval', 'zztest_fdr', 'ttest_fdr', 'ranksum_test_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        pval_cutoff
            float, set to filter out non-significant communication events
        node_cmap
            node color map, used to indicate different cell groups, set one from https://matplotlib.org/stable/tutorials/colors/colormaps.html
        figsize
            auto or a tuple of float such as (5.5, 4.2), defualt will be automatically estimate
        line_cmap
            line color map, used to indicate number of communication events, set one from https://matplotlib.org/stable/tutorials/colors/colormaps.html
        line_color_vmin
            float, the value limits the line color map in minimum
        line_color_vmax
            float, the value limits the line color map in maximum
        linewidth_norm
            two values in a tuple, used to normalize the dot size, such as (0.1, 1)
        node_size_norm
            two values in a tuple, used to normalize the node size, such as (50, 300)
        adjust_text_pos_node 
            True or Flase, whether adjust the text position to avoid overlapping automatically
        node_text_font
            float, font size for node text annotaion
        save
            str, the file name to save the figure
        show_plot
            True or False, whether print the figure on the screen
        comm_score_col
            column name of communication score, can be Commu_Score
        comm_score_cutoff
            a float, set a cutoff so only communications with score greater than the cutoff will be focused
        cutoff_prop
            a float between 0 and 1, set a cutoff to filter out lowly abundant cell populations by the fraction of cells expressed sensor genes or metabolite, Note that this parameter will lost the function if cutoff_prop was set lower than the one user set at begaining of running mebocost.infer_commu or preparing mebocost object. This parameter were designed to further strengthen the filtering.
        return_fig:
            True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself
        """
        
        commu_res = self.commu_res.copy()
        if flux_pass == True:
            if 'Flux_PASS' in commu_res.columns.tolist():
                commu_res = commu_res[commu_res['Flux_PASS'] == 'PASS']
            else:
                info('flux_pass set to True but NO Flux_PASS column, so skip.')
            
        ## pdf
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None
        if conditions:
            indices = commu_res['Condition'].isin(conditions)
            commu_res = commu_res.loc[indices,:]
        fig = CP._commu_network_plot_(commu_res=commu_res, sender_focus = sender_focus, metabolite_focus = metabolite_focus, 
                            sensor_focus = sensor_focus, receiver_focus = receiver_focus, and_or = and_or, 
                            pval_method = pval_method, remove_unrelevant = remove_unrelevant,
                            pval_cutoff = pval_cutoff, node_cmap = node_cmap, figsize = figsize, line_cmap = line_cmap, 
                            line_color_vmin = line_color_vmin, line_color_vmax = line_color_vmax,
                            linewidth_norm = linewidth_norm, linewidth_value_range = linewidth_value_range, node_text_hidden = node_text_hidden,
                            node_size_norm = node_size_norm, node_value_range = node_value_range, adjust_text_pos_node = adjust_text_pos_node, 
                            comm_score_col = comm_score_col, comm_score_cutoff = comm_score_cutoff, cutoff_prop = cutoff_prop,
                            node_text_font = node_text_font, pdf = Pdf, show_plot = show_plot, text_outline = text_outline,
                            return_fig = return_fig)
        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        
        if return_fig:
            return(fig)
            
    def violin_plot(self,
                    sensor_or_met,
                    cell_focus = [],
                    cell_order = [],
                    conditions = [],
                    row_zscore = False,
                    cmap = None,
                    vmin = None,
                    vmax = None,
                    figsize = 'auto',
                    cbar_title = '',
                    save = None,
                    show_plot = True,
                    return_fig = False):
        """
        Violin plot to show the distribution of sensor expression or metabolite enzyme expression across cell groups

        Params
        -----
        sensor_or_met
            a list, provide a list of sensor gene name or metabolite name
        cell_focus
            a list, provide a list of cell type that you want to focus, otherwise keep empty
        conditions
            a list of condition name, set to focus on mCCC in those conditions. Only use this when you run multiple samples with condition labels.
        cmap
            the color map used to draw the violin
        vmin
            float, maximum value for the color map
        vmin
            float, minimum value for the color map
        figsize
            auto or a tuple of float such as (5.5, 4.2), defualt will be automatically estimate
        title
            str, figure title on the top
        save
            str, the file name to save the figure
        show_plot
            True or False, whether print the figure on the screen
        comm_score_col
            column name of communication score, can be Commu_Score
        comm_score_cutoff
            a float, set a cutoff so only communications with score greater than the cutoff will be focused
        return_fig:
            True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself
        """
        ## cell group
        cell_ann = self.cell_ann.copy()
        if conditions:
            cell_ann = cell_ann.loc[cell_ann[self.condition_col].isin(conditions),:]
            
        if 'cell_group' not in cell_ann.columns.tolist():
            raise ValueError('ERROR: "cell_group" not in cell_ann column names!')
        ### extract expression for sensor
        sensors = []
        if self.exp_mat is not None and self.exp_mat_indexer is not None:
            sensor_loc = np.where(pd.Series(self.exp_mat_indexer).isin(sensor_or_met))
            #[i for i,j in enumerate(self.exp_mat_indexer.tolist()) if j in sensor_or_met]
            sensors = self.exp_mat_indexer[sensor_loc]
            #[j for i,j in enumerate(self.exp_mat_indexer.tolist()) if j in sensor_or_met]
            exp_dat = pd.DataFrame(self.exp_mat[sensor_loc].toarray(),
                                   index = sensors,
                                   columns = self.exp_mat_columns)
            
            if len(sensors) > 0:
                info('Find genes %s to plot violin'%(sensors))
                ## expression
                if save is not None and save is not False and isinstance(save, str):
                    save = save.replace('.pdf', '_sensor_exp.pdf')
                    Pdf = PdfPages(save)
                else:
                    Pdf = None
                if cmap is None:
                    ccmap = 'Reds'
                else:
                    ccmap = cmap

                if cbar_title == '':
                    if row_zscore:
                        sensor_cbar_title = 'Mean Z score of sensor expression'
                    else:
                        sensor_cbar_title = 'Mean sensor expression'
                else:
                    sensor_cbar_title = cbar_title
                ## data mat for plot
                dat_mat = pd.merge(exp_dat.T, cell_ann[['cell_group']], left_index = True, right_index = True).dropna()
                fig = CP._violin_plot_(dat_mat=dat_mat, sensor_or_met=list(sensors),
                                       cell_focus = cell_focus, cell_order = cell_order, 
                                       cmap = ccmap, row_zscore = row_zscore,
                                       vmin = vmin, vmax = vmax, figsize = figsize, 
                                       cbar_title = sensor_cbar_title, pdf = Pdf,
                                       show_plot = show_plot, return_fig = return_fig)

                if save is not None and save is not False and isinstance(save, str):
                    Pdf.close()
                if return_fig:
                    return(fig)
            else:
                info('Warnings: no sensors to plot')
        else:
            info('Warnings: failed to load metabolite data matrix')
            
        ### extract metabolite level
        metabolites = list(set(sensor_or_met) - set(sensors))
        metabolites = list(set(metabolites) & set(self.met_ann['metabolite'].unique().tolist()))
        if metabolites:
            # to HMDBID
            met_name_to_id = {}
            for m, iD in self.met_ann[['metabolite', 'HMDB_ID']].values.tolist():
                met_name_to_id[m] = iD
            metaboliteIds = {x: met_name_to_id.get(x) for x in metabolites}
            ## metabolite matrix
            if self.met_mat is not None and self.met_mat_indexer is not None:
                met_loc = np.where(pd.Series(self.met_mat_indexer).isin(list(metaboliteIds.values())))[0]
                met_Ids = self.met_mat_indexer[met_loc]
                met_names = [list(metaboliteIds.keys())[list(metaboliteIds.values()).index(x)] for x in met_Ids]
                met_dat = pd.DataFrame(self.met_mat[met_loc].toarray(),
                                   index = met_names,
                                   columns = self.met_mat_columns)
                dat_mat = pd.merge(met_dat.T, cell_ann[['cell_group']], left_index = True, right_index = True).dropna()
                if len(met_names) > 0:
                    info("Find metabolites %s to plot violin"%metabolites)
                    ## expression
                    if save is not None and save is not False and isinstance(save, str):
                        save = save.replace('.pdf', '_metabolite.pdf')
                        Pdf = PdfPages(save)
                    else:
                        Pdf = None
                    if cmap is None:
                        ccmap = 'Purples'
                    else:
                        ccmap = cmap
                    if cbar_title == '':
                        if row_zscore:
                            met_cbar_title = 'Mean Z score of\n aggregated enzyme expression'
                        else:
                            met_cbar_title = 'Mean aggregated enzyme expression'
                    else:
                        met_cbar_title = cbar_title
                        
                    fig = CP._violin_plot_(dat_mat=dat_mat, sensor_or_met=list(metaboliteIds.keys()),
                                     cell_focus = cell_focus, cmap = ccmap,
                                     cell_order = cell_order, row_zscore = row_zscore, 
                                    vmin = vmin, vmax = vmax, figsize = figsize,
                                    cbar_title = met_cbar_title, pdf = Pdf,
                                    show_plot = show_plot, return_fig = return_fig)

                    if save is not None and save is not False and isinstance(save, str):
                        Pdf.close()
                    if return_fig:
                        return(fig)
                else:
                    info('Warnings: no metabolites to plot')
            else:
                info('Warnings: failed to load metabolite data matrix')
        else:
            info('Warnings: no metabolites to plot')

# ============ notebook ==========
    def communication_in_notebook(self,
                                  pval_method = 'permutation_test_fdr',
                                  pval_cutoff = 0.05,
                                  comm_score_col = 'Commu_Score',
                                  comm_score_cutoff = None, 
                                  cutoff_prop = None
                                 ):

        # some handy functions to use along widgets
        from IPython.display import display, Markdown, clear_output, HTML
        import ipywidgets as widgets
        import functools

        outt = widgets.Output()

        df = self.commu_res.copy()
        
        if not comm_score_cutoff:
            comm_score_cutoff = 0
        if not cutoff_prop:
            cutoff_prop = 0
        ## basic filter
        df = df[(df[pval_method] <= pval_cutoff) & 
                (df[comm_score_col] >= comm_score_cutoff) &
                (df['metabolite_prop_in_sender'] >= cutoff_prop) &
                (df['sensor_prop_in_receiver'] >= cutoff_prop)
                ]
        
        senders = ['All']+sorted(list(df['Sender'].unique()))
        receivers = ['All']+sorted(list(df['Receiver'].unique()))
        metabolites = ['All']+sorted(list(df['Metabolite_Name'].unique()))
        transporters = ['All']+sorted(list(df['Sensor'].unique()))
        
        logic_butt = widgets.RadioButtons(
                            options=['and', 'or'],
                            description='Logic',
                            disabled=False
                        )

        sender_sel = widgets.SelectMultiple(description='Sender:',
                                            options=senders,
                                            layout=widgets.Layout(width='30%'))
        receiver_sel = widgets.SelectMultiple(description='Receiver:',
                                              options=receivers,
                                              layout=widgets.Layout(width='30%'))
        metabolite_sel = widgets.SelectMultiple(description='Metabolite:',
                                                options=metabolites,
                                                layout=widgets.Layout(width='30%'))
        sensor_sel = widgets.SelectMultiple(description='Sensor:',
                                                 options=transporters,
                                                layout=widgets.Layout(width='30%'))
        
        flux_butt = widgets.Button(description='Communication Flow (FlowPlot)',
                              layout=widgets.Layout(width='100%'))
        net_butt = widgets.Button(description='Communication Network (CirclePlot)',
                              layout=widgets.Layout(width='100%'))
        dotHeatmap_butt = widgets.Button(description='Communication Details (Dot-shaped Heatmap)',
                              layout=widgets.Layout(width='100%'))
        violin_butt = widgets.Button(description='ViolinPlot to show metabolite or sensor level in cell groups',
                              layout=widgets.Layout(width='100%'))

        def _flowplot_filter_(b):
            with outt:
                clear_output()
                print('+++++++++++++++++++++++++++ Running, Please Wait +++++++++++++++++++++++++++ ')
                print('[Selection]: Sender{}; Metabolite{}; Transporter{}; Receiver{}'.format(sender_sel.value,
                                                                                                  metabolite_sel.value,
                                                                                                  sensor_sel.value,
                                                                                                  receiver_sel.value))
                and_or = logic_butt.value
        
                self.FlowPlot(pval_method=pval_method,
                            pval_cutoff=pval_cutoff,
                            sender_focus = [x for x in sender_sel.value if x != 'All'],
                            metabolite_focus = [x for x in metabolite_sel.value if x != 'All'],
                            sensor_focus = [x for x in sensor_sel.value if x != 'All'],
                            receiver_focus = [x for x in receiver_sel.value if x != 'All'],
                            remove_unrelevant = True,
                            and_or = 'and',
                            flux_pass = True,
                            node_label_size = 8,
                            node_alpha = .8,
                            figsize = 'auto',
                            node_cmap = 'Set1',
                            line_cmap = 'bwr',
                            line_cmap_vmin = None,
                            line_cmap_vmax = None,
                            linewidth_norm = (0.1, 1),
                            linewidth_value_range = None,
                            node_size_norm = (10, 150),
                            node_value_range = None,
                            save=None, 
                            show_plot = True,
                            comm_score_col = comm_score_col,
                            comm_score_cutoff = comm_score_cutoff,
                            cutoff_prop = cutoff_prop,
                            text_outline = False,
                            return_fig = False)
                
                
        def _networkplot_filter_(b):
            with outt:
                clear_output()
                print('+++++++++++++++++++++++++++ Running, Please Wait +++++++++++++++++++++++++++ ')
                print('[Selection]: Sender{}; Metabolite{}; Transporter{}; Receiver{}'.format(sender_sel.value,
                                                                                                  metabolite_sel.value,
                                                                                                  sensor_sel.value,
                                                                                                  receiver_sel.value))
                and_or = logic_butt.value
                self.commu_network_plot(
                                sender_focus = [x for x in sender_sel.value if x != 'All'],
                                metabolite_focus = [x for x in metabolite_sel.value if x != 'All'],
                                sensor_focus = [x for x in sensor_sel.value if x != 'All'],
                                receiver_focus = [x for x in receiver_sel.value if x != 'All'],
                                remove_unrelevant = False,
                                and_or = and_or,
                                pval_method = pval_method,
                                pval_cutoff = pval_cutoff,
                                node_cmap = 'tab20',
                                figsize = 'auto',
                                line_cmap = 'RdBu_r',
                                line_color_vmin = None,
                                line_color_vmax = None,
                                linewidth_norm = (0.1, 1),
                                node_size_norm = (50, 300),
                                adjust_text_pos_node = False,
                                node_text_font = 10,
                                save = None,
                                show_plot = True,
                                comm_score_col = comm_score_col,
                                comm_score_cutoff = comm_score_cutoff,
                                cutoff_prop = cutoff_prop,
                                text_outline = False
                                )
        def _dotHeatmapPlot_(b):
            with outt:
                clear_output()
                print('+++++++++++++++++++++++++++ Running, Please Wait +++++++++++++++++++++++++++ ')
                print('[Selection]: Sender{}; Metabolite{}; Transporter{}; Receiver{}'.format(sender_sel.value,
                                                                                                  metabolite_sel.value,
                                                                                                  sensor_sel.value,
                                                                                                  receiver_sel.value))
                and_or = logic_butt.value
                self.commu_dotmap(
                            sender_focus = [x for x in sender_sel.value if x != 'All'],
                            metabolite_focus = [x for x in metabolite_sel.value if x != 'All'],
                            sensor_focus = [x for x in sensor_sel.value if x != 'All'],
                            receiver_focus = [x for x in receiver_sel.value if x != 'All'],
                            and_or = and_or,
                            pval_method=pval_method,
                            pval_cutoff=pval_cutoff, 

                            flux_pass = True,
                            figsize = 'auto',
                            cmap = 'bwr',
                            cmap_vmin = None,
                            cmap_vmax = None,
                            cellpair_order = [],
                            met_sensor_order = [],
                            dot_size_norm = (10, 150),
                            save = None, 
                            show_plot = True,
                            comm_score_col = comm_score_col,
                            comm_score_range = None,
                            comm_score_cutoff = comm_score_cutoff,
                            cutoff_prop = cutoff_prop,
                            swap_axis = False,
                            return_fig = False
                            
                )

        def _violinPlot_(b):
            with outt:
                clear_output()
                print('+++++++++++++++++++++++++++ Running, Please Wait +++++++++++++++++++++++++++ ')
                print('[Selection]: Sender{}; Metabolite{}; Transporter{}; Receiver{}'.format(sender_sel.value,
                                                                                                  metabolite_sel.value,
                                                                                                  sensor_sel.value,
                                                                                                  receiver_sel.value))
                
                self.violin_plot(
                                sensor_or_met = [x for x in metabolite_sel.value + sensor_sel.value if x != 'All'],
                                cell_focus = [x for x in sender_sel.value + receiver_sel.value if x != 'All'],
                                cmap = None,
                                vmin = None,
                                vmax = None,
                                figsize = 'auto',
                                cbar_title = '',
                                save = None,
                                show_plot = True)
                
                
        flux_butt.on_click(_flowplot_filter_)
        net_butt.on_click(_networkplot_filter_)
        dotHeatmap_butt.on_click(_dotHeatmapPlot_)
        violin_butt.on_click(_violinPlot_)


        h1 = widgets.HBox([sender_sel, metabolite_sel, sensor_sel, receiver_sel])
        h2 = widgets.VBox([flux_butt, net_butt, dotHeatmap_butt, violin_butt])

        mk = Markdown("""<b>Select and Click button to visulize</b>""")
        display(mk, widgets.VBox([logic_butt, h1, h2, outt]))

            
            
            
            
            
            
            

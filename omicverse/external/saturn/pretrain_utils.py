
import scanpy as sc
import pandas as pd
from anndata import AnnData
import warnings
from builtins import int
warnings.filterwarnings('ignore')
from . import losses, miners, distances, reducers, testers
from .utils.transformations import *
from .utils.accuracy_calculator import AccuracyCalculator
import torch
import torch.optim as optim
from torch import nn
import numpy as np
from .utils import logging_presets as logging_presets
try:
    import record_keeper
except Exception:  # optional dependency, only needed for logging hooks
    record_keeper = None

import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

import random 

#from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime
import sys


# https://www.microsoft.com/en-us/research/blog/less-pain-more-gain-a-simple-method-for-vae-training-with-less-of-that-kl-vanishing-agony/
def get_kld_cycle(epoch, period=20):
    '''
    0-10: 0 to 1
    10-20 1
    21-30 0 to 1
    30-40 1
    '''
    ct = epoch % period
    pt = epoch % (period//2)
    if ct >= period//2:
        return 1
    else:
        
        return min(1, (pt) / (period//2))    

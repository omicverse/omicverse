# -*- coding: utf-8 -*-
"""
scFEA: single-cell Flux Estimation Analysis.

Vendored into omicverse from https://github.com/changwn/scFEA
(commit 4c1fb76d52f07bafad84ce7686ad7c3acfcf0126).

Original author: wnchang@iu.edu
Reference: Alghamdi et al. (2021), Genome Research.

This module keeps the original GNN architecture, loss and training loop
bit-faithful to the upstream ``main()``. Only the I/O has been refactored:
the expression matrix is accepted as an in-memory pandas DataFrame and the
results are returned as DataFrames instead of being read from / written to
CSV files. The bundled model files are loaded from the vendored ``data/``
directory rather than from a user-supplied ``--data_dir`` argument.
"""

# system lib
import os
import time
import warnings

# tools
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
from tqdm import tqdm

# scFEA lib
from .ClassFlux import FLUX  # Flux class network
from .util import pearsonr
from .DatasetFlux import MyDataset


# hyper parameters
LEARN_RATE = 0.008
LAMB_BA = 1
LAMB_NG = 1
LAMB_CELL = 1
LAMB_MOD = 1e-2


# directory holding the vendored scFEA model files
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# bundled M168 asset sets, keyed by species
_SPECIES_FILES = {
    'human': {
        'moduleGene_file': 'module_gene_m168.csv',
        'stoichiometry_matrix': 'cmMat_c70_m168.csv',
        'cName_file': 'cName_c70_m168.csv',
    },
    'mouse': {
        'moduleGene_file': 'module_gene_complete_mouse_m168.csv',
        'stoichiometry_matrix': 'cmMat_complete_mouse_c70_m168.csv',
        'cName_file': 'cName_complete_mouse_c70_m168.csv',
    },
}


def myLoss(m, c, lamb1=0.2, lamb2=0.2, lamb3=0.2, lamb4=0.2,
           geneScale=None, moduleScale=None):

    # balance constrain
    total1 = torch.pow(c, 2)
    total1 = torch.sum(total1, dim=1)

    # non-negative constrain
    error = torch.abs(m) - m
    total2 = torch.sum(error, dim=1)

    # sample-wise variation constrain
    diff = torch.pow(torch.sum(m, dim=1) - geneScale, 2)
    if sum(diff > 0) == m.shape[0]:  # solve Nan after several iteraions
        total3 = torch.pow(diff, 0.5)
    else:
        total3 = diff

    # module-wise variation constrain
    if lamb4 > 0:
        corr = torch.FloatTensor(np.ones(m.shape[0]))
        for i in range(m.shape[0]):
            corr[i] = pearsonr(m[i, :], moduleScale[i, :])
        corr = torch.abs(corr)
        penal_m_var = torch.FloatTensor(np.ones(m.shape[0])) - corr
        total4 = penal_m_var
    else:
        total4 = torch.FloatTensor(np.zeros(m.shape[0]))

    # loss
    loss1 = torch.sum(lamb1 * total1)
    loss2 = torch.sum(lamb2 * total2)
    loss3 = torch.sum(lamb3 * total3)
    loss4 = torch.sum(lamb4 * total4)
    loss = loss1 + loss2 + loss3 + loss4
    return loss, loss1, loss2, loss3, loss4


def run_scfea(expr, *, species='human', sc_imputation=False, n_epoch=100,
              verbose=True, device=None):
    """Estimate single-cell metabolic flux with scFEA.

    Parameters
    ----------
    expr : pandas.DataFrame
        Gene-expression matrix, cells x genes (rows = cells, columns = gene
        symbols). May be raw or normalised counts; a log2 transform is applied
        automatically when the maximum value exceeds 50, mirroring upstream.
    species : {'human', 'mouse'}, default 'human'
        Selects the bundled M168 module / stoichiometry / compound-name files.
    sc_imputation : bool, default False
        Whether to MAGIC-impute the expression matrix before flux estimation
        (recommended for sparse 10x data). Requires the ``magic-impute``
        package; it is imported only when this is True.
    n_epoch : int, default 100
        Number of training epochs for the flux GNN. Must be > 0.
    verbose : bool, default True
        Print progress messages and show the training progress bar.
    device : str or torch.device, optional
        Compute device. Defaults to ``cuda:0`` if available, else ``cpu``.

    Returns
    -------
    dict
        ``{'flux': DataFrame (cells x modules),
        'balance': DataFrame (cells x metabolites),
        'predictions': dict}`` where ``flux`` corresponds to scFEA's ``setF``
        output, ``balance`` to its ``setB`` output, and ``predictions`` bundles
        the raw numpy arrays plus cell / module / compound name indices.
    """
    if n_epoch <= 0:
        raise NameError('n_epoch must greater than 1!')

    if species not in _SPECIES_FILES:
        raise ValueError(
            "species must be one of %s" % sorted(_SPECIES_FILES))

    if not isinstance(expr, pd.DataFrame):
        raise TypeError('expr must be a pandas DataFrame (cells x genes).')

    files = _SPECIES_FILES[species]
    moduleGene_file = files['moduleGene_file']
    cm_file = files['stoichiometry_matrix']
    cName_file = files['cName_file']

    # choose cpu or gpu automatically
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # ------------------------------------------------------------------
    # read data (expression supplied directly as a cells x genes DataFrame)
    # ------------------------------------------------------------------
    if verbose:
        print("Starting load data...")
    geneExpr = expr.copy()
    geneExpr = geneExpr * 1.0
    if sc_imputation is True:
        import magic  # deferred import: only needed for imputation
        magic_operator = magic.MAGIC()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            geneExpr = magic_operator.fit_transform(geneExpr)
    if geneExpr.max().max() > 50:
        geneExpr = (geneExpr + 1).apply(np.log2)
    geneExprSum = geneExpr.sum(axis=1)
    stand = geneExprSum.mean()
    geneExprScale = geneExprSum / stand
    geneExprScale = torch.FloatTensor(geneExprScale.values).to(device)

    BATCH_SIZE = geneExpr.shape[0]

    moduleGene = pd.read_csv(
        os.path.join(_DATA_DIR, moduleGene_file),
        sep=',',
        index_col=0)
    moduleLen = [moduleGene.iloc[i, :].notna().sum()
                 for i in range(moduleGene.shape[0])]
    moduleLen = np.array(moduleLen)

    # find existing gene
    module_gene_all = []
    for i in range(moduleGene.shape[0]):
        for j in range(moduleGene.shape[1]):
            if pd.isna(moduleGene.iloc[i, j]) is False:
                module_gene_all.append(moduleGene.iloc[i, j])
    module_gene_all = set(module_gene_all)
    data_gene_all = set(geneExpr.columns)
    gene_overlap = list(data_gene_all.intersection(module_gene_all))  # fix
    gene_overlap.sort()

    cmMat = pd.read_csv(
        os.path.join(_DATA_DIR, cm_file),
        sep=',',
        header=None)
    cmMat = cmMat.values
    cmMat = torch.FloatTensor(cmMat).to(device)

    cName = None
    if cName_file != 'noCompoundName':
        if verbose:
            print("Load compound name file, the balance output will have "
                  "compound name.")
        cName = pd.read_csv(
            os.path.join(_DATA_DIR, cName_file),
            sep=',',
            header=0)
        cName = cName.columns
    if verbose:
        print("Load data done.")

    if verbose:
        print("Starting process data...")
    emptyNode = []
    # extract overlap gene
    geneExpr = geneExpr[gene_overlap]
    gene_names = geneExpr.columns
    cell_names = geneExpr.index.astype(str)
    n_modules = moduleGene.shape[0]
    n_genes = len(gene_names)
    n_cells = len(cell_names)
    n_comps = cmMat.shape[0]
    _module_frames = []
    for i in range(n_modules):
        genes = moduleGene.iloc[i, :].values.astype(str)
        genes = [g for g in genes if g != 'nan']
        if not genes:
            emptyNode.append(i)
            continue
        temp = geneExpr.copy()
        temp.loc[:, [g for g in gene_names if g not in genes]] = 0
        temp = temp.T
        temp['Module_Gene'] = ['%02d_%s' % (i, g) for g in gene_names]
        _module_frames.append(temp)
    geneExprDf = pd.concat(_module_frames, ignore_index=True, sort=False)
    geneExprDf = geneExprDf[['Module_Gene'] + list(cell_names)]
    geneExprDf.index = geneExprDf['Module_Gene']
    geneExprDf.drop('Module_Gene', axis='columns', inplace=True)
    X = geneExprDf.values.T
    X = torch.FloatTensor(X.astype(float)).to(device)

    # prepare data for constraint of module variation based on gene
    df = geneExprDf
    df.index = [i.split('_')[0] for i in df.index]
    # must change type to ensure correct order, T column name order change!
    df.index = df.index.astype(int)
    module_scale = df.groupby(df.index).sum().T
    module_scale = torch.FloatTensor(module_scale.values / moduleLen)
    if verbose:
        print("Process data done.")

    # =====================================================================
    # NN
    torch.manual_seed(16)
    net = FLUX(X, n_modules, f_in=n_genes, f_out=1).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARN_RATE)

    # Dataloader
    dataloader_params = {'batch_size': BATCH_SIZE,
                         'shuffle': False,
                         'num_workers': 0,
                         'pin_memory': False}

    dataSet = MyDataset(X, geneExprScale, module_scale)
    train_loader = torch.utils.data.DataLoader(dataset=dataSet,
                                               **dataloader_params)

    # =====================================================================
    if verbose:
        print("Starting train neural network...")
    start = time.time()
    # training
    loss_v = []
    loss_v1 = []
    loss_v2 = []
    loss_v3 = []
    loss_v4 = []
    net.train()
    epoch_iter = tqdm(range(n_epoch)) if verbose else range(n_epoch)
    for epoch in epoch_iter:
        loss, loss1, loss2, loss3, loss4 = 0, 0, 0, 0, 0

        for i, (X, X_scale, m_scale) in enumerate(train_loader):

            X_batch = Variable(X.float().to(device))
            X_scale_batch = Variable(X_scale.float().to(device))
            m_scale_batch = Variable(m_scale.float().to(device))

            out_m_batch, out_c_batch = net(X_batch, n_modules, n_genes,
                                           n_comps, cmMat)
            (loss_batch, loss1_batch, loss2_batch, loss3_batch,
             loss4_batch) = myLoss(
                out_m_batch, out_c_batch,
                lamb1=LAMB_BA, lamb2=LAMB_NG, lamb3=LAMB_CELL, lamb4=LAMB_MOD,
                geneScale=X_scale_batch, moduleScale=m_scale_batch)

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

            loss += loss_batch.cpu().data.numpy()
            loss1 += loss1_batch.cpu().data.numpy()
            loss2 += loss2_batch.cpu().data.numpy()
            loss3 += loss3_batch.cpu().data.numpy()
            loss4 += loss4_batch.cpu().data.numpy()

        loss_v.append(loss)
        loss_v1.append(loss1)
        loss_v2.append(loss2)
        loss_v3.append(loss3)
        loss_v4.append(loss4)

    # =====================================================================
    end = time.time()
    if verbose:
        print("Training time: ", end - start)

    # Dataloader
    dataloader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': 0,
                         'pin_memory': False}

    dataSet = MyDataset(X, geneExprScale, module_scale)
    test_loader = torch.utils.data.DataLoader(dataset=dataSet,
                                              **dataloader_params)

    # testing
    fluxStatuTest = np.zeros((n_cells, n_modules), dtype='f')  # float32
    balanceStatus = np.zeros((n_cells, n_comps), dtype='f')
    net.eval()
    for epoch in range(1):
        for i, (X, X_scale, _) in enumerate(test_loader):

            X_batch = Variable(X.float().to(device))
            out_m_batch, out_c_batch = net(X_batch, n_modules, n_genes,
                                           n_comps, cmMat)

            # save data
            fluxStatuTest[i, :] = out_m_batch.detach().cpu().numpy()
            balanceStatus[i, :] = out_c_batch.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # assemble results as DataFrames (mirrors scFEA setF / setB outputs)
    # ------------------------------------------------------------------
    setF = pd.DataFrame(fluxStatuTest)
    setF.columns = moduleGene.index
    setF.index = geneExpr.index.tolist()

    setB = pd.DataFrame(balanceStatus)
    setB.rename(columns=lambda x: x + 1)
    setB.index = setF.index
    if cName_file != 'noCompoundName':
        setB.columns = cName

    if verbose:
        print("scFEA job finished.")

    predictions = {
        'flux': fluxStatuTest,
        'balance': balanceStatus,
        'cell_names': list(setF.index),
        'module_names': list(moduleGene.index),
        'compound_names': list(cName) if cName is not None else None,
        'loss': loss_v,
    }

    return {'flux': setF, 'balance': setB, 'predictions': predictions}

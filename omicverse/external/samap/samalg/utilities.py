import pandas as pd
import numpy as np
import scipy as sp
import os
import errno
from sklearn.decomposition import PCA
import umap.distances as dist
from sklearn.utils.extmath import svd_flip
from sklearn.utils import check_array, check_random_state
from scipy import sparse
import sklearn.utils.sparsefuncs as sf
from umap.umap_ import nearest_neighbors
__version__ = "0.8.7"


def find_corr_genes(sam, input_gene):
    """Rank genes by their spatially averaged expression pattern correlations to
    a desired gene.

    Parameters
    ----------

    sam - SAM
        The analyzed SAM object

    input_gene - string
        The gene ID with respect to which correlations will be computed.

    Returns
    -------
    A ranked list of gene IDs based on correlation to the input gene.
    """
    all_gene_names = np.array(list(sam.adata.var_names))

    D_avg = sam.adata.layers["X_knn_avg"]

    input_gene = np.where(all_gene_names == input_gene)[0]

    if input_gene.size == 0:
        print(
            "Gene note found in the filtered dataset. Note "
            "that genes are case sensitive."
        )
        return

    pw_corr = generate_correlation_map(D_avg.T.A, D_avg[:, input_gene].T.A)
    return all_gene_names[np.argsort(-pw_corr.flatten())]

def _pca_with_sparse(X, npcs, solver='arpack', mu=None, seed=0, mu_axis=0):
    random_state = check_random_state(seed)
    np.random.set_state(random_state.get_state())
    random_init = np.random.rand(np.min(X.shape))
    X = check_array(X, accept_sparse=['csr', 'csc'])

    if mu is None:
        if mu_axis == 0:
            mu = X.mean(0).A.flatten()[None, :]
        else:
            mu = X.mean(1).A.flatten()[:, None]
    
    if mu_axis == 0:
        mdot = mu.dot
        mmat = mdot
        mhdot = mu.T.dot
        mhmat = mu.T.dot
        Xdot = X.dot
        Xmat = Xdot
        XHdot = X.T.conj().dot
        XHmat = XHdot
        ones = np.ones(X.shape[0])[None, :].dot

        def matvec(x):
            return Xdot(x) - mdot(x)

        def matmat(x):
            return Xmat(x) - mmat(x)

        def rmatvec(x):
            return XHdot(x) - mhdot(ones(x))

        def rmatmat(x):
            return XHmat(x) - mhmat(ones(x))

    else:
        mdot = mu.dot
        mmat = mdot
        mhdot = mu.T.dot
        mhmat = mu.T.dot
        Xdot = X.dot
        Xmat = Xdot
        XHdot = X.T.conj().dot
        XHmat = XHdot
        ones = np.ones(X.shape[1])[None, :].dot        

        def matvec(x):
            return Xdot(x) - mdot(ones(x))

        def matmat(x):
            return Xmat(x) - mmat(ones(x))

        def rmatvec(x):
            return XHdot(x) - mhdot(x)

        def rmatmat(x):
            return XHmat(x) - mhmat(x)

    XL = sp.sparse.linalg.LinearOperator(
        matvec=matvec,
        dtype=X.dtype,
        matmat=matmat,
        shape=X.shape,
        rmatvec=rmatvec,
        rmatmat=rmatmat,
    )

    u, s, v = sp.sparse.linalg.svds(XL, solver=solver, k=npcs, v0=random_init)
    u, v = svd_flip(u, v)
    idx = np.argsort(-s)
    v = v[idx, :]

    X_pca = (u * s)[:, idx]
    ev = s[idx] ** 2 / (X.shape[0] - 1)

    total_var = sf.mean_variance_axis(X, axis=0)[1].sum()
    ev_ratio = ev / total_var

    output = {
        'X_pca': X_pca,
        'variance': ev,
        'variance_ratio': ev_ratio,
        'components': v,
    }
    return output



def nearest_neighbors_wrapper(X,n_neighbors=15,metric='correlation',metric_kwds={},angular=True,random_state=0):
    random_state=np.random.RandomState(random_state)
    return nearest_neighbors(X,n_neighbors,metric,metric_kwds,angular,random_state)[:2]

def knndist(nnma):
    x, y = nnma.nonzero()
    data = nnma.data
    knn = y.reshape((nnma.shape[0], nnma[0, :].data.size))
    val = data.reshape(knn.shape)
    return knn, val


def save_figures(filename, fig_IDs=None, **kwargs):
    """
    Save figures.

    Parameters
    ----------
    filename - str
        Name of output file

    fig_IDs - int, numpy.array, list, optional, default None
        A list of open figure IDs or a figure ID that will be saved to a
        pdf/png file respectively.

    **kwargs -
        Extra keyword arguments passed into 'matplotlib.pyplot.savefig'.

    """
    import matplotlib.pyplot as plt

    if fig_IDs is not None:
        if type(fig_IDs) is list:
            savetype = "pdf"
        else:
            savetype = "png"
    else:
        savetype = "pdf"

    if savetype == "pdf":
        from matplotlib.backends.backend_pdf import PdfPages

        if len(filename.split(".")) == 1:
            filename = filename + ".pdf"
        else:
            filename = ".".join(filename.split(".")[:-1]) + ".pdf"

        pdf = PdfPages(filename)

        if fig_IDs is None:
            figs = [plt.figure(n) for n in plt.get_fignums()]
        else:
            figs = [plt.figure(n) for n in fig_IDs]

        for fig in figs:
            fig.savefig(pdf, format="pdf", **kwargs)
        pdf.close()
    elif savetype == "png":
        plt.figure(fig_IDs).savefig(filename, **kwargs)


def weighted_PCA(mat, do_weight=True, npcs=None, solver="auto",seed = 0):
    # mat = (mat - np.mean(mat, axis=0))
    if do_weight:
        if min(mat.shape) >= 10000 and npcs is None:
            print(
                "More than 10,000 cells. Running with 'npcs' set to < 1000 is"
                " recommended."
            )

        if npcs is None:
            ncom = min(mat.shape)
        else:
            ncom = min((min(mat.shape), npcs))
            
        pca = PCA(svd_solver=solver, n_components=ncom,random_state=check_random_state(seed))
        reduced = pca.fit_transform(mat)
        scaled_eigenvalues = pca.explained_variance_
        scaled_eigenvalues = scaled_eigenvalues / scaled_eigenvalues.max()
        reduced_weighted = reduced * scaled_eigenvalues[None, :] ** 0.5
    else:
        pca = PCA(n_components=npcs, svd_solver=solver,random_state=check_random_state(seed))
        reduced = pca.fit_transform(mat)
        if reduced.shape[1] == 1:
            pca = PCA(n_components=2, svd_solver=solver,random_state=check_random_state(seed))
            reduced = pca.fit_transform(mat)
        reduced_weighted = reduced

    return reduced_weighted, pca


def transform_wPCA(mat, pca):
    mat = mat - pca.mean_
    reduced = mat.dot(pca.components_.T)
    v = pca.explained_variance_  # .var(0)
    scaled_eigenvalues = v / v.max()
    reduced_weighted = np.array(reduced) * scaled_eigenvalues[None, :] ** 0.5
    return reduced_weighted

def search_string(vec, s, case_sensitive=False, invert=False):
    vec = np.array(vec)


    if isinstance(s,list):
        S = s
    else:
        S = [s]

    V=[]; M=[]
    for s in S:
        m = []
        if not case_sensitive:
            s = s.lower()
        for i in range(len(vec)):
            if case_sensitive:
                st = vec[i]
            else:
                st = vec[i].lower()
            b = st.find(s)
            if not invert and b != -1 or invert and b == -1:
                m.append(i)
        if len(m) > 0:
            V.append(vec[np.array(m)]); M.append(np.array(m))
    if len(V)>0:
        i = len(V)
        if not invert:
            V = np.concatenate(V); M = np.concatenate(M);
            if i > 1:
                ix = np.sort(np.unique(M,return_index=True)[1])
                V=V[ix]; M=M[ix];
        else:
            for i in range(len(V)):
                V[i]=list(set(V[i]).intersection(*V))
            V = vec[np.in1d(vec,np.unique(np.concatenate(V)))]
            M = np.array([np.where(vec==x)[0][0] for x in V])
        return V,M
    else:
        return -1,-1


def distance_matrix_error(dist1, dist2):
    s = 0
    for k in range(dist1.shape[0]):
        s += np.corrcoef(dist1[k, :], dist2[k, :])[0, 1]
    return 1 - s / dist1.shape[0]


def generate_euclidean_map(A, B):
    a = (A ** 2).sum(1).flatten()
    b = (B ** 2).sum(1).flatten()
    x = a[:, None] + b[None, :] - 2 * np.dot(A, B.T)
    x[x < 0] = 0
    return np.sqrt(x)


def generate_correlation_map(x, y):
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError("x and y must " + "have the same number of timepoints.")
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    s_x[s_x == 0] = 1
    s_y[s_y == 0] = 1
    cov = np.dot(x, y.T) - n * np.dot(mu_x[:, None], mu_y[None, :])
    return cov / np.dot(s_x[:, None], s_y[None, :])


def extract_annotation(cn, x, c="_"):
    m = []
    if x is not None:
        for i in range(cn.size):
            f = cn[i].split(c)
            x = min(len(f) - 1, x)
            m.append(f[x])
        return np.array(m)
    else:
        ms = []
        ls = []
        for i in range(cn.size):
            f = cn[i].split(c)
            m = []
            for x in range(len(f)):
                m.append(f[x])
            ms.append(m)
            ls.append(len(m))
        ml = max(ls)
        for i in range(len(ms)):
            ms[i].extend([""] * (ml - len(ms[i])))
            if ml - len(ms[i]) > 0:
                ms[i] = np.concatenate(ms[i])
        ms = np.vstack(ms)
        MS = []
        for i in range(ms.shape[1]):
            MS.append(ms[:, i])
        return MS


def isolate(dt, x1, x2, y1, y2):
    return np.where(
        np.logical_and(
            np.logical_and(dt[:, 0] > x1, dt[:, 0] < x2),
            np.logical_and(dt[:, 1] > y1, dt[:, 1] < y2),
        )
    )[0]


def to_lower(y):
    x = y.copy().flatten()
    for i in range(x.size):
        x[i] = x[i].lower()
    return x


def to_upper(y):
    x = y.copy().flatten()
    for i in range(x.size):
        x[i] = x[i].upper()
    return x


def create_folder(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def convert_annotations(A):
    x = np.unique(A)
    y = np.zeros(A.size)
    z = 0
    for i in x:
        y[A == i] = z
        z += 1

    return y.astype("int")


def nearest_neighbors_hnsw(x,ef=200,M=48,n_neighbors = 100):
    import hnswlib
    labels = np.arange(x.shape[0])
    p = hnswlib.Index(space = 'cosine', dim = x.shape[1])
    p.init_index(max_elements = x.shape[0], ef_construction = ef, M = M)
    p.add_items(x, labels)
    p.set_ef(ef)
    idx, dist = p.knn_query(x, k = n_neighbors)
    return idx,dist


def calc_nnm(g_weighted, k, distance=None):
    if g_weighted.shape[0] > 0:
        if distance == 'cosine':
            nnm, dists = nearest_neighbors_hnsw(g_weighted, n_neighbors=k)
        else:
            nnm, dists = nearest_neighbors_wrapper(g_weighted, n_neighbors=k, metric=distance)
        EDM = gen_sparse_knn(nnm, dists)
        EDM = EDM.tocsr()        
    return EDM


def compute_distances(A, dm):
    if dm == "euclidean":
        m = np.dot(A, A.T)
        h = np.diag(m)
        x = h[:, None] + h[None, :] - 2 * m
        x[x < 0] = 0
        dist = np.sqrt(x)
    elif dm == "correlation":
        dist = 1 - np.corrcoef(A)
    else:
        dist = sp.spatial.distance.squareform(sp.spatial.distance.pdist(A, metric=dm))
    return dist


def dist_to_nn(d, K):  # , offset = 0):
    E = d.copy()
    np.fill_diagonal(E, -1)
    M = np.max(E) * 2
    x = np.argsort(E, axis=1)[:, :K]  # offset:K+offset]
    E[
        np.tile(
            np.arange(E.shape[0]).reshape(E.shape[0], -1), (1, x.shape[1])
        ).flatten(),
        x.flatten(),
    ] = M

    E[E < M] = 0
    E[E > 0] = 1
    return E  # ,x


def to_sparse_knn(D1, k):
    for i in range(D1.shape[0]):
        x = D1.data[D1.indptr[i] : D1.indptr[i + 1]]
        idx = np.argsort(x)
        if idx.size > k:
            x[idx[:-k]] = 0
        D1.data[D1.indptr[i] : D1.indptr[i + 1]] = x
    D1.eliminate_zeros()
    return D1


def gen_sparse_knn(knni, knnd, shape=None):
    if shape is None:
        shape = (knni.shape[0], knni.shape[0])

    D1 = sp.sparse.lil_matrix(shape)

    D1[
        np.tile(np.arange(knni.shape[0])[:, None], (1, knni.shape[1])).flatten().astype('int32'),
        knni.flatten().astype('int32'),
    ] = knnd.flatten()
    D1 = D1.tocsr()
    return D1

from . import np, ut, pd

def save_samap(sm,fn):
    import dill
    if len(fn.split('.pkl')) == 1:
        fn = fn + '.pkl'
    sm.path_to_file = fn
    
    #TODO: Clean this up...
    try:
        del sm.sam1.SamGui
    except:
        pass
    try:
        del sm.sam2.SamGui
    except:
        pass
    try:
        del sm.samap.SamGui
    except:
        pass    
    try:
        del sm.SamapGui
    except:
        pass    
    try:
        del sm.sam1.umap_obj
    except:
        pass;
    try:
        del sm.sam2.umap_obj
    except:
        pass;
    
    with open(fn,'wb') as f:
        dill.dump(sm,f)
        
def load_samap(fn):
    import dill
    if len(fn.split('.pkl')) == 1:
        fn = fn + '.pkl'    
    with open(fn,'rb') as f:
        sm = dill.load(f)
    return sm
        
def prepend_var_prefix(s, pre):
    x = [x.split("_")[0] for x in s.adata.var_names]
    vn = []
    for i,g in enumerate(s.adata.var_names):
        if x[i] != pre:
            vn.append(pre+"_"+g)
        else:
            vn.append(g)
    s.adata.var_names = pd.Index(vn)
    s.adata_raw.var_names = pd.Index(vn)


def df_to_dict(DF, key_key=None, val_key=[]):
    if key_key is None:
        index = list(DF.index)
    else:
        index = list(DF[key_key].values)

    if len(val_key) == 0:
        val_key = list(DF.columns)

    a = []
    b = []
    for key in val_key:
        if key != key_key:
            a.extend(index)
            b.extend(list(DF[key].values))
    a = np.array(a)
    b = np.array(b)

    idx = np.argsort(a)
    a = a[idx]
    b = b[idx]
    bounds = np.where(a[:-1] != a[1:])[0] + 1
    bounds = np.append(np.append(0, bounds), a.size)
    bounds_left = bounds[:-1]
    bounds_right = bounds[1:]
    slists = [b[bounds_left[i] : bounds_right[i]] for i in range(bounds_left.size)]
    d = dict(zip(np.unique(a), slists))
    return d


def to_vn(op):
    return np.array(list(op[:, 0].astype("object") + ";" + op[:, 1].astype("object")))


def to_vo(op):
    return np.vstack((ut.extract_annotation(op, None, ";"))).T


def substr(x, s="_", ix=None, obj=False):
    m = []
    if ix is not None:
        for i in range(len(x)):
            f = x[i].split(s)
            ix = min(len(f) - 1, ix)
            m.append(f[ix])
        return np.array(m).astype("object") if obj else np.array(m)
    else:
        ms = []
        ls = []
        for i in range(len(x)):
            f = x[i].split(s)
            m = []
            for ix in range(len(f)):
                m.append(f[ix])
            ms.append(m)
            ls.append(len(m))
        ml = max(ls)
        for i in range(len(ms)):
            ms[i].extend([""] * (ml - len(ms[i])))
            if ml - len(ms[i]) > 0:
                ms[i] = np.concatenate(ms[i])
        ms = np.vstack(ms)
        if obj:
            ms = ms.astype("object")
        MS = []
        for i in range(ms.shape[1]):
            MS.append(ms[:, i])
        return MS


def sparse_knn(D, k):
    D1 = D.tocoo()
    idr = np.argsort(D1.row)
    D1.row[:] = D1.row[idr]
    D1.col[:] = D1.col[idr]
    D1.data[:] = D1.data[idr]

    _, ind = np.unique(D1.row, return_index=True)
    ind = np.append(ind, D1.data.size)
    for i in range(ind.size - 1):
        idx = np.argsort(D1.data[ind[i] : ind[i + 1]])
        if idx.size > k:
            idx = idx[:-k]
            D1.data[np.arange(ind[i], ind[i + 1])[idx]] = 0
    D1.eliminate_zeros()
    return D1

"""Public adapter routing and input/output contracts, independent of backends."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import anndata as ad
import pytest
from scipy import sparse
from omicverse import _roadmap_ports as ports

def test_ruv_arguments_are_forwarded(monkeypatch):
    received={}
    def call(*args,**kwargs):received.update(kwargs);return 'result'
    monkeypatch.setattr(ports,'_backend',lambda name: SimpleNamespace(ruvg=call))
    assert ports.ruvg(np.ones((4,6)),[0,1],drop=0,is_log=True)=='result'
    assert received['is_log'] and received['round_counts']

def test_spark_sparse_orientation_and_gene_names(monkeypatch):
    a=ad.AnnData(sparse.csr_matrix([[1,0,2],[2,0,1],[3,0,4],[4,0,3]]))
    a.var_names=['a','zero','b'];a.layers['counts']=a.X.copy();a.obsm['spatial']=np.ones((4,2))
    original=a.copy()
    def call(x,loc,cov,option):
        assert sparse.issparse(x) and x.shape==(3,4)
        table=pd.DataFrame({'value':[1.,2.]})
        return {'statistics':table.copy(),'pvalues':table.copy(),'results':table.copy(),
                'keep_genes':np.array([True,False,True]),'keep_spots':np.ones(4,dtype=bool)}
    monkeypatch.setattr(ports,'_backend',lambda name:SimpleNamespace(sparkx=call))
    result=ports.sparkx(a)
    assert result['results'].index.tolist()==['a','b']
    assert (a.X!=original.X).nnz==0
    with pytest.raises(ValueError,match='obs_names'):
        ports.sparkx(a,covariates=pd.DataFrame({'x':[1,2,3,4]}))

def test_scrans_restricted_path_and_no_mutation(monkeypatch):
    a=ad.AnnData(np.ones((240,10)));a.layers['counts']=a.X.copy()
    a.obs['cluster']=np.repeat(['a','b'],120)
    def call(work,**kwargs):
        assert kwargs['algorithm']=='QR' and kwargs['max_size'] is None
        assert not kwargs['parallelize'] and not kwargs['plotting']
        work.obs['size_factors']=np.ones(240)
        work.X[:]=99
    monkeypatch.setattr(ports,'_backend',lambda name:SimpleNamespace(compute_sum_factors=call))
    out=ports.scran_pooling(a,cluster_key='cluster')
    np.testing.assert_array_equal(out,1)
    np.testing.assert_array_equal(a.X,1)
    assert 'size_factors' not in a.obs
    a.layers['counts']=sparse.csr_matrix(a.layers['counts'])
    with pytest.raises(ValueError,match='dense'):ports.scran_pooling(a)

def test_nebulosa_requires_explicit_bandwidth(monkeypatch):
    def call(w,x,**kwargs):
        assert kwargs['bandwidth']==(.3,.4) and kwargs['method']=='wkde'
        return np.ones(len(w))
    monkeypatch.setattr(ports,'_backend',lambda name:SimpleNamespace(calculate_density=call))
    np.testing.assert_array_equal(ports.nebulosa_density([1,2],[[0,0],[1,1]],bandwidth=(.3,.4)),1)
    with pytest.raises(TypeError):ports.nebulosa_density([1,2],[[0,0],[1,1]])

def test_muscat_preserves_design_contrast_and_sample_contract(monkeypatch):
    design=pd.DataFrame({'intercept':[1,1,1,1],'condition':[0,0,1,1]},index=list('abcd'))
    contrasts=pd.DataFrame({'stim':[0,1]},index=design.columns)
    pb=object()
    def call(data,d,c,**kwargs):
        assert data is pb and d is design and c is contrasts
        assert kwargs=={'method':'limma-voom','min_cells':12,'filter':'genes'}
        return {'skipped':{'rare':'insufficient replication'}}
    monkeypatch.setattr(ports,'_backend',lambda name:SimpleNamespace(pb_ds=call))
    result=ports.muscat_pb_ds(pb,design,contrasts,method='limma-voom',min_cells=12,filter='genes')
    assert 'rare' in result['skipped']

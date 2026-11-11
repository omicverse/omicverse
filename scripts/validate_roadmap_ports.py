"""Real backend acceptance against a checkout layout matching the adapter guide."""
from pathlib import Path
import sys
import argparse
import json
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import omicverse as ov

parser=argparse.ArgumentParser()
parser.add_argument('--ports-root',type=Path,required=True)
parser.add_argument('--report',type=Path)
args=parser.parse_args();root=args.ports_root
passed=[]
p=root/'py-RUVSeq/data'
g=np.loadtxt(p/'ruvg/counts_input.csv',delimiter=',')
g_result=ov.bulk.ruvg(g,range(8),k=2)
np.testing.assert_array_equal(g_result['normalized_counts'],np.loadtxt(p/'ruvg/counts_corrected.csv',delimiter=','))
x=np.loadtxt(p/'input.csv',delimiter=',');e=np.loadtxt(p/'residuals.csv',delimiter=',')
for name,result in [('r',ov.bulk.ruvr(x,range(12),2,e,round_counts=False)),
                    ('s',ov.bulk.ruvs(x,range(12),2,np.arange(16).reshape(4,4),round_counts=False))]:
    np.testing.assert_allclose(result['normalized_counts'],np.loadtxt(p/f'{name}_corrected.csv',delimiter=','),atol=1e-8,rtol=1e-8)
passed.append('RUVg/RUVr/RUVs')
p=root/'py-SPARKX/data'
x=np.loadtxt(p/'input.csv',delimiter=',');a=ad.AnnData(x.T)
a.layers['counts']=a.X.copy();a.obsm['spatial']=np.loadtxt(p/'coordinates.csv',delimiter=',')
result=ov.space.sparkx(a)
np.testing.assert_allclose(result['pvalues'],np.loadtxt(p/'mixture_p.csv',delimiter=','),atol=1e-6,rtol=0)
assert result['results'].index.equals(a.var_names)
passed.append('SPARK-X')
p=root/'py-MuSiC/data'
x=pd.read_csv(p/'reference.csv',index_col=0);bulk=pd.read_csv(p/'bulk.csv',index_col=0)
meta=pd.read_csv(p/'metadata.csv',index_col=0)
result=ov.bulk.music_prop(bulk,x,meta.ct,meta.donor)
np.testing.assert_allclose(result['proportions'],pd.read_csv(p/'default_Est.prop.weighted.csv',index_col=0),atol=1e-8,rtol=1e-8)
passed.append('MuSiC')
p=root/'py-muscat/data'
x=pd.read_csv(p/'public_counts.csv',index_col=0);meta=pd.read_csv(p/'public_metadata.csv',index_col=0)
a=ad.AnnData(x.T.to_numpy(),obs=meta,var=pd.DataFrame(index=x.index));a.layers['counts']=a.X.copy()
pb=ov.single.muscat_aggregate(a)
d=pd.read_csv(p/'public_design.csv',index_col=0);c=pd.read_csv(p/'public_contrast.csv',index_col=0)
for method in ['edgeR','limma-voom']:
    result=ov.single.muscat_pb_ds(pb,d,c,method=method)['table'].set_index(['cluster_id','gene']).sort_index()
    ref=pd.read_csv(p/f'public_{method}.csv').set_index(['cluster_id','gene']).sort_index()
    np.testing.assert_allclose(result.logFC,ref.logFC,atol=1e-8,rtol=1e-8)
    assert spearmanr(result.p_val,ref.p_val).statistic>=.99
passed.append('muscat edgeR/limma-voom')
p=root/'references/scranPY/tests/fixtures'
x=np.loadtxt(p/'scran_counts.csv',delimiter=',').T;a=ad.AnnData(x.copy());a.layers['counts']=a.X.copy()
a.obs['cluster']=np.repeat(['a','b'],120)
result=ov.pp.scran_pooling(a,cluster_key='cluster')
np.testing.assert_allclose(result,np.loadtxt(p/'scran_factors.csv',delimiter=','),atol=1e-8,rtol=1e-8)
np.testing.assert_array_equal(a.X,x)
passed.append('scranPY QR')
p=root/'references/pyNebulosa/tests/fixtures/r_grid'
result=ov.pl.nebulosa_density(np.loadtxt(p/'weights.csv',delimiter=','),np.loadtxt(p/'coordinates.csv',delimiter=','),
                              bandwidth=tuple(np.loadtxt(p/'bandwidth.csv',delimiter=',')))
np.testing.assert_allclose(result,np.loadtxt(p/'cells.csv',delimiter=','),atol=1e-12,rtol=1e-10)
passed.append('Nebulosa fixed bandwidth')
report={'passed':passed,'status':'passed','scope':'Fixture-level acceptance of the documented subsets.'}
print(json.dumps(report,indent=2))
if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')

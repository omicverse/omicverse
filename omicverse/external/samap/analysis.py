import sklearn.utils.sparsefuncs as sf
from . import q, ut, pd, sp, np, warnings, sc
from .utils import to_vo, to_vn, substr, df_to_dict, sparse_knn, prepend_var_prefix
from .samalg import SAM
from scipy.stats import rankdata
import networkx as nx

def _log_factorial(n):
    return np.log(np.arange(1,n+1)).sum()
def _log_binomial(n,k):
    return _log_factorial(n) - (_log_factorial(k) + _log_factorial(n-k))

def GOEA(target_genes,GENE_SETS,df_key='GO',goterms=None,fdr_thresh=0.25,p_thresh=1e-3): 
    """Performs GO term Enrichment Analysis using the hypergeometric distribution.
    
    Parameters
    ----------
    target_genes - array-like
        List of target genes from which to find enriched GO terms.
    GENE_SETS - dictionary or pandas.DataFrame
        Dictionary where the keys are GO terms and the values are lists of genes associated with each GO term.
        Ex: {'GO:0000001': ['GENE_A','GENE_B'],
             'GO:0000002': ['GENE_A','GENE_C','GENE_D']}
        Make sure to include all available genes that have GO terms in your dataset.
        
        ---OR---
        
        Pandas DataFrame with genes as the index and GO terms values.
        Ex: 'GENE_A','GO:0000001',
            'GENE_A','GO:0000002',
            'GENE_B','GO:0000001',
            'GENE_B','GO:0000004',
            ...
        If `GENE_SETS` is a pandas DataFrame, the `df_key` parameter should be the name of the column in which
        the GO terms are stored.       
    df_key - str, optional, default 'GO'
        The name of the column in which GO terms are stored. Only used if `GENE_SETS` is a DataFrame.
    goterms - array-list, optional, default None
        If provided, only these GO terms will be tested.
    fdr_thresh - float, optional, default 0.25
        Filter out GO terms with FDR q value greater than this threshold.
    p_thresh - float, optional, default 1e-3
        Filter out GO terms with p value greater than this threshold.
        
    Returns:
    -------
    enriched_goterms - pandas.DataFrame
        A Pandas DataFrame of enriched GO terms with FDR q values, p values, and associated genes provided.
    """    
    
    # identify all genes found in `GENE_SETS`
    
    if isinstance(GENE_SETS,pd.DataFrame):
        print('Converting DataFrame into dictionary')
        genes = np.array(list(GENE_SETS.index))
        agt = np.array(list(GENE_SETS[df_key].values))
        idx = np.argsort(agt)
        genes = genes[idx]
        agt = agt[idx]
        bounds = np.where(agt[:-1]!=agt[1:])[0]+1
        bounds = np.append(np.append(0,bounds),agt.size)
        bounds_left=bounds[:-1]
        bounds_right=bounds[1:]
        genes_lists = [genes[bounds_left[i]:bounds_right[i]] for i in range(bounds_left.size)]
        GENE_SETS = dict(zip(np.unique(agt),genes_lists))
    all_genes = np.unique(np.concatenate(list(GENE_SETS.values())))
    all_genes = np.array(all_genes)
    
    # if goterms is None, use all the goterms found in `GENE_SETS`
    if goterms is None:
        goterms = np.unique(list(GENE_SETS.keys()))
    else:
        goterms = goterms[np.in1d(goterms,np.unique(list(GENE_SETS.keys())))]
    
    # ensure that target genes are all present in `all_genes`
    _,ix = np.unique(target_genes,return_index=True)
    target_genes=target_genes[np.sort(ix)]
    target_genes = target_genes[np.in1d(target_genes,all_genes)]
    
    # N -- total number of genes
    N = all_genes.size

    probs=[]
    probs_genes=[]
    counter=0
    # for each go term,
    for goterm in goterms:
        if counter%1000==0:
            pass; #print(counter)
        counter+=1
        
        # identify genes associated with this go term
        gene_set = np.array(GENE_SETS[goterm])
        
        # B -- number of genes associated with this go term
        B = gene_set.size
        
        # b -- number of genes in target associated with this go term
        gene_set_in_target = gene_set[np.in1d(gene_set,target_genes)]
        b = gene_set_in_target.size        
        if b != 0:
            # calculate the enrichment probability as the cumulative sum of the tail end of a hypergeometric distribution
            # with parameters (N,B,n,b)
            n = target_genes.size
            num_iter = min(n,B)
            rng = np.arange(b,num_iter+1)
            probs.append(sum([np.exp(_log_binomial(n,i)+_log_binomial(N-n,B-i) - _log_binomial(N,B)) for i in rng]))
        else:
            probs.append(1.0)
        
        #append associated genes to a list
        probs_genes.append(gene_set_in_target)
        
    probs = np.array(probs)    
    probs_genes = np.array([';'.join(x) for x in probs_genes])
    
    # adjust p value to correct for multiple testing
    fdr_q_probs = probs.size*probs / rankdata(probs,method='ordinal')
    
    # filter out go terms based on the FDR q value and p value thresholds
    filt = np.logical_and(fdr_q_probs<fdr_thresh,probs<p_thresh)
    enriched_goterms = goterms[filt]
    p_values = probs[filt]
    fdr_q_probs = fdr_q_probs[filt]    
    probs_genes=probs_genes[filt]
    
    # construct the Pandas DataFrame
    gns = probs_genes
    enriched_goterms = pd.DataFrame(data=fdr_q_probs,index=enriched_goterms,columns=['fdr_q_value'])
    enriched_goterms['p_value'] = p_values
    enriched_goterms['genes'] = gns
    
    # sort in ascending order by the p value
    enriched_goterms = enriched_goterms.sort_values('p_value')   
    return enriched_goterms

_KOG_TABLE = dict(A = "RNA processing and modification",
                 B = "Chromatin structure and dynamics",
                 C = "Energy production and conversion",
                 D = "Cell cycle control, cell division, chromosome partitioning",
                 E = "Amino acid transport and metabolism",
                 F = "Nucleotide transport and metabolism",
                 G = "Carbohydrate transport and metabolism",
                 H = "Coenzyme transport and metabolism",
                 I = "Lipid transport and metabolism",
                 J = "Translation, ribosomal structure and biogenesis",
                 K = "Transcription",
                 L = "Replication, recombination, and repair",
                 M = "Cell wall membrane/envelope biogenesis",
                 N = "Cell motility",
                 O = "Post-translational modification, protein turnover, chaperones",
                 P = "Inorganic ion transport and metabolism",
                 Q = "Secondary metabolites biosynthesis, transport and catabolism",
                 R = "General function prediction only",
                 S = "Function unknown",
                 T = "Signal transduction mechanisms",
                 U = "Intracellular trafficking, secretion, and vesicular transport",
                 V = "Defense mechanisms",
                 W = "Extracellular structures",
                 Y = "Nuclear structure",
                 Z = "Cytoskeleton")

import gc
from collections.abc import Iterable
class FunctionalEnrichment(object):    
    def __init__(self,sm, DFS, col_key, keys, delimiter = '', align_thr = 0.1, limit_reference = False, n_top = 0):
        """Performs functional enrichment analysis on gene pairs enriched
        in mapped cell types using functional annotations output by Eggnog.
        
        Parameters
        ----------
        sm - SAMAP object.
        
        DFS - dictionary of pandas.DataFrame functional annotations keyed by species present in the input `SAMAP` object.
        
        col_key - str
            The column name with functional annotations in the annotation DataFrames.
        
        keys - dictionary of column keys from `.adata.obs` DataFrames keyed by species present in the input `SAMAP` object.
            Cell type mappings will be computed between these annotation vectors.
                        
        delimiter - str, optional, default ''
            Some transcripts may have multiple functional annotations (e.g. GO terms or KOG terms) separated by
            a delimiter. For KOG terms, this is typically no delimiter (''). For GO terms, this is usually a comma
            (',').
            
        align_thr - float, optional, default 0.1
            The alignment score below which to filter out cell type mappings
            
        limit_reference - bool, optional, default False
            If True, limits the background set of genes to include only those that are enriched in any cell type mappings
            If False, the background set of genes will include all genes present in the input dataframes.
            
        n_top: int, optional, default 0
            If `n_top` is 0, average the alignment scores for all cells in a pair of clusters.
            Otherwise, average the alignment scores of the top `n_top` cells in a pair of clusters.
            Set this to non-zero if you suspect there to be subpopulations of your cell types mapping
            to distinct cell types in the other species.

        """
        # get dictionary of sam objects

        SAMS=sm.sams
        
        # link up SAM memories.
        for sid in sm.ids:
            sm.sams[sid] = SAMS[sid]
            gc.collect()
            
        for k in DFS.keys():
            DFS[k].index = k+'_'+DFS[k].index

        # concatenate DFS
        A = pd.concat(list(DFS.values()),axis=0)
        RES = pd.DataFrame(A[col_key])
        RES.columns=['GO']    
        RES = RES[(q(RES.values.flatten())!='nan')]
        
        # EXPAND RES
        data = []
        index = []
        for i in range(RES.shape[0]):
            if delimiter == '':
                l = list(RES.values[i][0])
                l = np.array([str(x) if str(x).isalpha() else '' for x in l])
                l = l[l!= '']
                l = list(l)
            else:
                l = RES.values[i][0].split(delimiter)
                
            data.extend(l)
            index.extend([RES.index[i]]*len(l))
        
        RES = pd.DataFrame(index = index,data = data,columns = ['GO'])
        genes = np.array(list(RES.index))
        agt = np.array(list(RES['GO'].values))
        idx = np.argsort(agt)
        genes = genes[idx]
        agt = agt[idx]
        bounds = np.where(agt[:-1]!=agt[1:])[0]+1
        bounds = np.append(np.append(0,bounds),agt.size)
        bounds_left=bounds[:-1]
        bounds_right=bounds[1:]
        genes_lists = [genes[bounds_left[i]:bounds_right[i]] for i in range(bounds_left.size)]
        GENE_SETS = dict(zip(np.unique(agt),genes_lists)) 
        for cc in GENE_SETS.keys():
            GENE_SETS[cc]=np.unique(GENE_SETS[cc])
        
        G = []
        
        print(f'Finding enriched gene pairs...')
        gpf = GenePairFinder(sm,keys=keys)
        gene_pairs = gpf.find_all(thr=align_thr,n_top=n_top)   
        
        self.DICT = {}
        for c in gene_pairs.columns:
            if "_pval1" not in c and "_pval2" not in c:
                x = q(gene_pairs[c].values.flatten()).astype('str')
                ff = x!='nan'
                if ff.sum()>0:
                    self.DICT[c] = x[ff]

        if limit_reference:
            all_genes = np.unique(np.concatenate(substr(np.concatenate(list(self.DICT.values())),';')))
        else:
            all_genes = np.unique(np.array(list(A.index)))
        
        for d in GENE_SETS.keys():
            GENE_SETS[d] = GENE_SETS[d][np.in1d(GENE_SETS[d],all_genes)]

        self.gene_pairs = gene_pairs
        self.CAT_NAMES = np.unique(q(RES['GO']))
        self.GENE_SETS = GENE_SETS
        self.RES = RES
        
    def calculate_enrichment(self,verbose=False):
        """ Calculates the functional enrichment.
        
        Parameters
        ----------
        verbose - bool, optional, default False
            If False, function does not log progress to output console.
            
        Returns
        -------
        ENRICHMENT_SCORES - pandas.DataFrame (cell types x function categories)
            Enrichment scores (-log10 p-value) for each function in each cell type.
        
        NUM_ENRICHED_GENES - pandas.DataFrame (cell types x function categories)
            Number of enriched genes for each function in each cell type.        
        
        ENRICHED_GENES - pandas.DataFrame (cell types x function categories)
            The IDs of enriched genes for each function in each cell type.
        """
        DICT = self.DICT
        RES = self.RES
        CAT_NAMES = self.CAT_NAMES
        GENE_SETS = self.GENE_SETS
        pairs = np.array(list(DICT.keys()))
        all_nodes = np.unique(np.concatenate(substr(pairs,';')))

        CCG={}
        P=[]
        for ik in range(len(all_nodes)):
            genes=[]
            nodes = all_nodes[ik]
            for j in range(len(pairs)):
                n1,n2 = pairs[j].split(';')
                if n1 == nodes or n2 == nodes:
                    g1,g2 = substr(DICT[pairs[j]],';')
                    genes.append(np.append(g1,g2))
            if len(genes) > 0:
                genes = np.concatenate(genes)
                genes = np.unique(genes)
            else:
                genes = np.array([])
            CCG[all_nodes[ik]] = genes

        HM = np.zeros((len(CAT_NAMES),len(all_nodes)))
        HMe = np.zeros((len(CAT_NAMES),len(all_nodes)))
        HMg = np.zeros((len(CAT_NAMES),len(all_nodes)),dtype='object')
        for ii,cln in enumerate(all_nodes):
            if verbose:
                print(f'Calculating functional enrichment for cell type {cln}')
                
            g = CCG[cln]    

            if g.size > 0:
                gi = g[np.in1d(g,q(RES.index))]
                ix = np.where(np.in1d(q(RES.index),gi))[0]
                res = RES.iloc[ix]
                goterms = np.unique(q(res['GO']))
                goterms = goterms[goterms!='S'].flatten()
                if goterms.size > 0:
                    result = GOEA(gi,GENE_SETS,goterms=goterms,fdr_thresh=100,p_thresh=100)

                    lens = np.array([len(np.unique(x.split(';'))) for x in result['genes'].values])
                    F = -np.log10(result['p_value'])
                    gt,vals = F.index,F.values
                    Z = pd.DataFrame(data=np.arange(CAT_NAMES.size)[None,:],columns=CAT_NAMES)
                    if gt.size>0:
                        HM[Z[gt].values.flatten(),ii] = vals
                        HMe[Z[gt].values.flatten(),ii] = lens
                        HMg[Z[gt].values.flatten(),ii] = [';'.join(np.unique(x.split(';'))) for x in result['genes'].values]

        #CAT_NAMES = [_KOG_TABLE[x] for x in CAT_NAMES]
        SC = pd.DataFrame(data = HM,index=CAT_NAMES,columns=all_nodes).T
        SCe = pd.DataFrame(data = HMe,index=CAT_NAMES,columns=all_nodes).T
        SCg = pd.DataFrame(data = HMg,index=CAT_NAMES,columns=all_nodes).T
        SCg.values[SCg.values==0]=''
        
        self.ENRICHMENT_SCORES = SC
        self.NUM_ENRICHED_GENES = SCe
        self.ENRICHED_GENES = SCg
        
        return self.ENRICHMENT_SCORES,self.NUM_ENRICHED_GENES,self.ENRICHED_GENES
    
    def plot_enrichment(self,cell_types = [], pval_thr=2.0,msize = 50):
        """Create a plot summarizing the functional enrichment analysis.
        
        Parameters
        ----------
        cell_types - list, default []
            A list of cell types for which enrichment scores will be plotted. If empty (default),
            all cell types will be plotted.
            
        pval_thr - float, default 2.0
            -log10 p-values < 2.0 will be filtered from the plot.
            
        msize - float, default 50
            The marker size in pixels for the dot plot.
        
        Returns
        -------
        fig - matplotlib.pyplot.Figure
        ax - matplotlib.pyplot.Axes
        """
        import colorsys
        import seaborn as sns
        import matplotlib
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle
        from matplotlib import cm,colors
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import linkage, dendrogram

        SC = self.ENRICHMENT_SCORES
        SCe = self.NUM_ENRICHED_GENES
        SCg = self.ENRICHED_GENES

        if len(cell_types) > 0:
            SC = SC.T[cell_types].T
            SCe = SCe.T[cell_types].T
            SCg = SCg.T[cell_types].T

        CAT_NAMES = self.CAT_NAMES
        gc_names = np.array(CAT_NAMES)

        SC.values[SC.values<pval_thr]=0
        SCe.values[SC.values<pval_thr]=0
        SCg.values[SC.values<pval_thr]=''
        SCg=SCg.astype('str')
        SCg.values[SCg.values=='nan']=''

        ixrow = np.array(dendrogram(linkage(SC.values.T,method='ward',metric='euclidean'),no_plot=True)['ivl']).astype('int')
        ixcol = np.array(dendrogram(linkage(SC.values,method='ward',metric='euclidean'),no_plot=True)['ivl']).astype('int')    

        SC = SC.iloc[ixcol].iloc[:,ixrow]
        SCe = SCe.iloc[ixcol].iloc[:,ixrow]
        SCg = SCg.iloc[ixcol].iloc[:,ixrow]


        SCgx = SCg.values.copy()


        for i in range(SCgx.shape[0]):
            idn = SCg.index[i].split('_')[0]

            for j in range(SCgx.shape[1]):
                genes = np.array(SCgx[i,j].split(';'))    
                SCgx[i,j] = ';'.join(genes[np.array([x.split('_')[0] for x in genes]) == idn])


        x,y=np.tile(np.arange(SC.shape[0]),SC.shape[1]),np.repeat(np.arange(SC.shape[1]),SC.shape[0])    
        co = SC.values[x,y].flatten()#**0.5
        ms = SCe.values[x,y].flatten()
        ms=ms/ms.max()
        x=x.max()-x #
        ms = ms*msize
        ms[np.logical_and(ms<0.15,ms>0)]=0.15

        fig,ax = plt.subplots();
        fig.set_size_inches((7*SC.shape[0]/SC.shape[1],7)) 

        scat=ax.scatter(x,y,c=co,s=ms,cmap='seismic',edgecolor='k',linewidth=0.5,vmin=3)
        cax = fig.colorbar(scat,pad=0.02);
        ax.set_yticks(np.arange(SC.shape[1]))
        ax.set_yticklabels(SC.columns,ha='right',rotation=0)
        ax.set_xticks(np.arange(SC.shape[0]))
        ax.set_xticklabels(SC.index[::-1],ha='right',rotation=45)
        ax.invert_yaxis()
        ax.invert_xaxis()
        #ax.figure.tight_layout()
        return fig,ax
    
def sankey_plot(M,species_order=None,align_thr=0.1,**params):
    """Generate a sankey plot
    
    Parameters
    ----------
    M: pandas.DataFrame
        Mapping table output from `get_mapping_scores` (second output).

    align_thr: float, optional, default 0.1
        The alignment score threshold below which to remove cell type mappings.
    
    species_order: list, optional, default None
        Specify the order of species (left-to-right) in the sankey plot.
        For example, `species_order=['hu','le','ms']`.

    Keyword arguments
    -----------------
    Keyword arguments will be passed to `sankey.opts`.
    """    
    if species_order is not None:
        ids = np.array(species_order)
    else:
        ids = np.unique([x.split('_')[0] for x in M.index])

    if len(ids)>2:
        d = M.values.copy()
        d[d<align_thr]=0
        x,y = d.nonzero()
        x,y = np.unique(np.sort(np.vstack((x,y)).T,axis=1),axis=0).T
        values = d[x,y]
        nodes = q(M.index)

        node_pairs = nodes[np.vstack((x,y)).T]
        sn1 = q([xi.split('_')[0] for xi in node_pairs[:,0]])
        sn2 = q([xi.split('_')[0] for xi in node_pairs[:,1]])
        filt = np.logical_or(
            np.logical_or(np.logical_and(sn1==ids[0],sn2==ids[1]),np.logical_and(sn1==ids[1],sn2==ids[0])),
            np.logical_or(np.logical_and(sn1==ids[1],sn2==ids[2]),np.logical_and(sn1==ids[2],sn2==ids[1]))
        )
        x,y,values=x[filt],y[filt],values[filt]
        
        d=dict(zip(ids,list(np.arange(len(ids)))))        
        depth_map = dict(zip(nodes,[d[xi.split('_')[0]] for xi in nodes]))
        data =  nodes[np.vstack((x,y))].T
        for i in range(data.shape[0]):
            if d[data[i,0].split('_')[0]] > d[data[i,1].split('_')[0]]:
                data[i,:]=data[i,::-1]
        R = pd.DataFrame(data = data,columns=['source','target'])
        
        R['Value'] = values       
    else:
        d = M.values.copy()
        d[d<align_thr]=0
        x,y = d.nonzero()
        x,y = np.unique(np.sort(np.vstack((x,y)).T,axis=1),axis=0).T
        values = d[x,y]
        nodes = q(M.index)
        R = pd.DataFrame(data = nodes[np.vstack((x,y))].T,columns=['source','target'])
        R['Value'] = values
        depth_map=None
    
    try:
        from holoviews import dim
        #from bokeh.models import Label
        import holoviews as hv
        hv.extension('bokeh',logo=False)
        hv.output(size=100)        
    except:
        raise ImportError('Please install holoviews-samap with `!pip install holoviews-samap`.')

    def f(plot,element):
        plot.handles['plot'].sizing_mode='scale_width'    
        plot.handles['plot'].x_range.start = -600    
        #plot.handles['plot'].add_layout(Label(x=plot.handles['plot'].x_range.end*0.78, y=plot.handles['plot'].y_range.end*0.96, text=id2))
        plot.handles['plot'].x_range.end = 1500    
        #plot.handles['plot'].add_layout(Label(x=0, y=plot.handles['plot'].y_range.end*0.96, text=id1))


    sankey1 = hv.Sankey(R, kdims=["source", "target"])#, vdims=["Value"])

    cmap = params.get('cmap','Colorblind')
    label_position = params.get('label_position','outer')
    edge_line_width = params.get('edge_line_width',0)
    show_values = params.get('show_values',False)
    node_padding = params.get('node_padding',4)
    node_alpha = params.get('node_alpha',1.0)
    node_width = params.get('node_width',40)
    node_sort = params.get('node_sort',True)
    frame_height = params.get('frame_height',1000)
    frame_width = params.get('frame_width',800)
    bgcolor = params.get('bgcolor','snow')
    apply_ranges = params.get('apply_ranges',True)


    sankey1.opts(cmap=cmap,label_position=label_position, edge_line_width=edge_line_width, show_values=show_values,
                 node_padding=node_padding,node_cmap=depth_map, node_alpha=node_alpha, node_width=node_width,
                 node_sort=node_sort,frame_height=frame_height,frame_width=frame_width,bgcolor=bgcolor,
                 apply_ranges=apply_ranges,hooks=[f])

    return sankey1

def chord_plot(A,align_thr=0.1):
    """Generate a chord plot
    
    Parameters
    ----------
    A: pandas.DataFrame
        Mapping table output from `get_mapping_scores` (second output).

    align_thr: float, optional, default 0.1
        The alignment score threshold below which to remove cell type mappings.
    """        
    try:
        from holoviews import dim, opts
        import holoviews as hv
        hv.extension('bokeh',logo=False)
        hv.output(size=300)        
    except:
        raise ImportError('Please install holoviews-samap with `!pip install holoviews-samap`.')

    xx=A.values.copy()
    xx[xx<align_thr]=0
    x,y = xx.nonzero()
    z=xx[x,y]
    x,y = A.index[x],A.columns[y]
    links=pd.DataFrame(data=np.array([x,y,z]).T,columns=['source','target','value'])
    links['edge_grp'] = [x.split('_')[0]+y.split('_')[0] for x,y in zip(links['source'],links['target'])]
    links['value']*=100
    f = links['value'].values
    z=((f-f.min())/(f.max()-f.min())*0.99+0.01)*100
    links['value']=z
    links['value']=np.round([x for x in links['value'].values]).astype('int')
    clu=np.unique(A.index)
    clu = clu[np.in1d(clu,np.unique(np.array([x,y])))]
    links = hv.Dataset(links)
    nodes = hv.Dataset(pd.DataFrame(data=np.array([clu,clu,np.array([x.split('_')[0] for x in clu])]).T,columns=['index','name','group']),'index')
    chord = hv.Chord((links, nodes),kdims=["source", "target"], vdims=["value","edge_grp"])#.select(value=(5, None))
    chord.opts(
        opts.Chord(cmap='Category20', edge_cmap='Category20',edge_color=dim('edge_grp'),
                   labels='name', node_color=dim('group').str()))    
    return chord


class GenePairFinder(object):
    def __init__(self, sm, keys=None):
        """Find enriched gene pairs in cell type mappings.
        
        sm: SAMAP object

        keys: dict of str, optional, default None
            Keys corresponding to the annotations vectors in the AnnData's keyed by species ID.
            By default, will use the leiden clusters, e.g. {'hu':'leiden_clusters','ms':'leiden_clusters'}.

        """
        if keys is None:
            keys={}
            for sid in sm.sams.keys():
                keys[sid] = 'leiden_clusters'
        self.sm = sm
        self.sams = sm.sams
        self.s3 = sm.samap
        self.gns = q(sm.samap.adata.var_names)
        self.gnnm = sm.samap.adata.varp['homology_graph_reweighted']
        self.gns_dict = sm.gns_dict

        self.ids = sm.ids
        
        mus={}
        stds={}
        for sid in self.sams.keys():
            self.sams[sid].adata.obs[keys[sid]] = self.sams[sid].adata.obs[keys[sid]].astype('str')
            mu, var = sf.mean_variance_axis(self.sams[sid].adata[:, self.gns_dict[sid]].X, axis=0)
            var[var == 0] = 1
            var = var ** 0.5
            mus[sid]=pd.Series(data=mu,index=self.gns_dict[sid])
            stds[sid]=pd.Series(data=var,index=self.gns_dict[sid])

        self.mus = mus
        self.stds = stds
        self.keys = keys
        self.find_markers()

    def find_markers(self):
        for sid in self.sams.keys():
            print(
                "Finding cluster-specific markers in {}:{}.".format(
                    sid, self.keys[sid]
                )
            )        
            import gc
            if self.keys[sid]+'_scores' not in self.sams[sid].adata.varm.keys():
                find_cluster_markers(self.sams[sid], self.keys[sid])
                gc.collect()
                    
    def find_all(self,n=None,align_thr=0.1,n_top=0,**kwargs):
        """Find enriched gene pairs in all pairs of mapped cell types.
        
        Parameters
        ----------
        n: str, optional, default None
            If passed, find enriched gene pairs of all cell types connected to `n`.
            
        thr: float, optional, default 0.2
            Alignment score threshold above which to consider cell type pairs mapped.
        
        n_top: int, optional, default 0
            If `n_top` is 0, average the alignment scores for all cells in a pair of clusters.
            Otherwise, average the alignment scores of the top `n_top` cells in a pair of clusters.
            Set this to non-zero if you suspect there to be subpopulations of your cell types mapping
            to distinct cell types in the other species.        
            
        Keyword arguments
        -----------------
        Keyword arguments to `find_genes` accepted here.
        
        Returns
        -------
        Table of enriched gene pairs for each cell type pair
        """        

        _,M = get_mapping_scores(self.sm, self.keys, n_top = n_top)
        ax = q(M.index)
        data = M.values.copy()
        data[data<align_thr]=0
        x,y = data.nonzero()
        ct1,ct2 = ax[x],ax[y]
        if n is not None:
            f1 = ct1==n
            f2 = ct2==n
            f = np.logical_or(f1,f2)
        else:
            f = np.array([True]*ct2.size)
        
        ct1=ct1[f]
        ct2=ct2[f]
        ct1,ct2 = np.unique(np.sort(np.vstack((ct1,ct2)).T,axis=1),axis=0).T
        res={}
        for i in range(ct1.size):
            a = '_'.join(ct1[i].split('_')[1:])
            b = '_'.join(ct2[i].split('_')[1:])
            print('Calculating gene pairs for the mapping: {};{} to {};{}'.format(ct1[i].split('_')[0],a,ct2[i].split('_')[0],b))
            res['{};{}'.format(ct1[i],ct2[i])] = self.find_genes(ct1[i],ct2[i],**kwargs)
        
        cols = []
        col_names = []
        for k in res:
            col_names.append(k)
            col_names.append(k+"_pval1")
            col_names.append(k+"_pval2")
            cols.append(res[k][0])
            cols.append(res[k][-2])
            cols.append(res[k][-1])
        res = pd.DataFrame(cols,index=col_names).fillna(np.nan).T            
        return res
        
    def find_genes(
        self,
        n1,
        n2,
        w1t=0.2,
        w2t=0.2,
        n_genes=1000,
        thr=1e-2,
    ):
        """Find enriched gene pairs in a particular pair of cell types.
        
        n1: str, cell type ID from species 1
        
        n2: str, cell type ID from species 2
        
        w1t & w2t: float, optional, default 0.2
            SAM weight threshold for species 1 and 2. Genes with below this threshold will not be
            included in any enriched gene pairs.
        
        n_genes: int, optional, default 1000
            Takes the top 1000 ranked gene pairs before filtering based on differential expressivity and
            SAM weights.
        
        thr: float, optional, default 0.01
            Excludes genes with greater than 0.01 differential expression p-value.
            
        Returns
        -------
        G - Enriched gene pairs
        G1 - Genes from species 1 involved in enriched gene pairs
        G2 - Genes from species 2 involved in enriched gene pairs
        pvals1 - pvalues for genes from species 1 involved in enriched gene pairs
        pvals2 - pvalues for genes from species 2 involved in enriched gene pairs
        """
        n1 = str(n1)
        n2 = str(n2)
        id1,id2 = n1.split('_')[0],n2.split('_')[0]
        sam1,sam2=self.sams[id1],self.sams[id2]

        n1,n2 = '_'.join(n1.split('_')[1:]),'_'.join(n2.split('_')[1:])
        assert n1 in q(self.sams[id1].adata.obs[self.keys[id1]])
        assert n2 in q(self.sams[id2].adata.obs[self.keys[id2]])
        
        m,gpairs = self._find_link_genes_avg(n1, n2, id1,id2, w1t=w1t, w2t=w2t, expr_thr=0.05)

        self.gene_pair_scores = pd.Series(index=gpairs, data=m)

        G = q(gpairs[np.argsort(-m)[:n_genes]])
        G1 = substr(G, ";", 0)
        G2 = substr(G, ";", 1)
        pvals1 = q(sam1.adata.varm[self.keys[id1] + "_pvals"][n1][G1])
        pvals2 = q(sam2.adata.varm[self.keys[id2] + "_pvals"][n2][G2])
        filt = np.logical_and(pvals1 < thr,pvals2 < thr)
        G = q(
            G[filt]
        )
        G1 = substr(G, ";", 0)
        G2 = substr(G, ";", 1)
        _, ix1 = np.unique(G1, return_index=True)
        _, ix2 = np.unique(G2, return_index=True)
        G1 = G1[np.sort(ix1)]
        G2 = G2[np.sort(ix2)]
        return G, G1, G2, pvals1[filt], pvals2[filt]

    def _find_link_genes_avg(self, c1, c2, id1, id2, w1t=0.35, w2t=0.35, expr_thr=0.05):
        mus = self.mus
        stds = self.stds
        sams=self.sams

        keys=self.keys
        sam3=self.s3
        gnnm = self.gnnm
        gns = self.gns
        
        xs = []
        for sid in [id1,id2]:
            xs.append(sams[sid].get_labels(keys[sid]).astype('str').astype('object'))
        x1,x2 = xs
        g1, g2 = gns[np.vstack(gnnm.nonzero())]
        gs1,gs2 = q([x.split('_')[0] for x in g1]),q([x.split('_')[0] for x in g2])
        filt = np.logical_and(gs1==id1,gs2==id2)
        g1=g1[filt]
        g2=g2[filt]
        sam1,sam2 = sams[id1],sams[id2]
        mu1,std1,mu2,std2 = mus[id1][g1].values,stds[id1][g1].values,mus[id2][g2].values,stds[id2][g2].values

        X1 = _sparse_sub_standardize(sam1.adata[:, g1].X[x1 == c1, :], mu1, std1)
        X2 = _sparse_sub_standardize(sam2.adata[:, g2].X[x2 == c2, :], mu2, std2)
        a, b = sam3.adata.obsp["connectivities"][sam3.adata.obs['species']==id1,:][:,sam3.adata.obs['species']==id2][
            x1 == c1, :][:, x2 == c2].nonzero()
        c, d = sam3.adata.obsp["connectivities"][sam3.adata.obs['species']==id2,:][:,sam3.adata.obs['species']==id1][
            x2 == c2, :][:, x1 == c1].nonzero()            

        pairs = np.unique(np.vstack((np.vstack((a, b)).T, np.vstack((d, c)).T)), axis=0)

        av1 = X1[np.unique(pairs[:, 0]), :].mean(0).A.flatten()
        av2 = X2[np.unique(pairs[:, 1]), :].mean(0).A.flatten()
        sav1 = (av1 - av1.mean()) / av1.std()
        sav2 = (av2 - av2.mean()) / av2.std()
        sav1[sav1 < 0] = 0
        sav2[sav2 < 0] = 0
        val = sav1 * sav2 / sav1.size
        X1.data[:] = 1
        X2.data[:] = 1
        min_expr = (X1.mean(0).A.flatten() > expr_thr) * (
            X2.mean(0).A.flatten() > expr_thr
        )

        w1 = sam1.adata.var["weights"][g1].values.copy()
        w2 = sam2.adata.var["weights"][g2].values.copy()
        w1[w1 < 0.2] = 0
        w2[w2 < 0.2] = 0
        w1[w1 > 0] = 1
        w2[w2 > 0] = 1
        return val * w1 * w2 * min_expr, to_vn(np.array([g1,g2]).T)

def find_cluster_markers(sam, key, inplace=True):
    """ Finds differentially expressed genes for provided cell type labels.
    
    Parameters
    ----------
    sam - SAM object
    
    key - str
        Column in `sam.adata.obs` for which to identifying differentially expressed genes.
        
    inplace - bool, optional, default True
        If True, deposits enrichment scores in `sam.adata.varm[f'{key}_scores']`
        and p-values in `sam.adata.varm[f'{key}_pvals']`.
        
        Otherwise, returns three pandas.DataFrame objects (genes x clusters).
            NAMES - the gene names
            PVALS - the p-values
            SCORES - the enrichment scores
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   
        
        a,c = np.unique(q(sam.adata.obs[key]),return_counts=True)
        t = a[c==1]

        adata = sam.adata[np.in1d(q(sam.adata.obs[key]),a[c==1],invert=True)].copy()
        sc.tl.rank_genes_groups(
            adata,
            key,
            method="wilcoxon",
            n_genes=sam.adata.shape[1],
            use_raw=False,
            layer=None,
        )

        sam.adata.uns['rank_genes_groups'] = adata.uns['rank_genes_groups']

        NAMES = pd.DataFrame(sam.adata.uns["rank_genes_groups"]["names"])
        PVALS = pd.DataFrame(sam.adata.uns["rank_genes_groups"]["pvals"])
        SCORES = pd.DataFrame(sam.adata.uns["rank_genes_groups"]["scores"])
        if not inplace:
            return NAMES, PVALS, SCORES
        dfs1 = []
        dfs2 = []
        for i in range(SCORES.shape[1]):
            names = NAMES.iloc[:, i]
            scores = SCORES.iloc[:, i]
            pvals = PVALS.iloc[:, i]
            pvals[scores < 0] = 1.0
            scores[scores < 0] = 0
            pvals = q(pvals)
            scores = q(scores)

            dfs1.append(pd.DataFrame(
                data=scores[None, :], index = [SCORES.columns[i]], columns=names
            )[sam.adata.var_names].T)
            dfs2.append(pd.DataFrame(
                data=pvals[None, :], index = [SCORES.columns[i]], columns=names
            )[sam.adata.var_names].T)
        df1 = pd.concat(dfs1,axis=1)
        df2 = pd.concat(dfs2,axis=1)

        try:
            sam.adata.varm[key+'_scores'] = df1
            sam.adata.varm[key+'_pvals'] = df2
        except:
            sam.adata.varm.dim_names = sam.adata.var_names
            sam.adata.varm.dim_names = sam.adata.var_names
            sam.adata.varm[key+'_scores'] = df1
            sam.adata.varm[key+'_pvals'] = df2        

        for i in range(t.size):
            sam.adata.varm[key+'_scores'][t[i]]=0
            sam.adata.varm[key+'_pvals'][t[i]]=1



def ParalogSubstitutions(sm, ortholog_pairs, paralog_pairs=None, psub_thr = 0.3):
    """Identify paralog substitutions. 
    
    For all genes in `ortholog_pairs` and `paralog_pairs`, this function expects the genes to
    be prepended with their corresponding species IDs.
    
    Parameters
    ----------
    sm - SAMAP object
    
    ortholog_pairs - n x 2 numpy array of ortholog pairs
    
    paralog_pairs - n x 2 numpy array of paralog pairs, optional, default None
        If None, assumes every pair in the homology graph that is not an ortholog is a paralog.
        Note that this would essentially result in the more generic 'homolog substitutions' rather
        than paralog substitutions.
        
        The paralogs can be either cross-species, within-species, or a mix of both. 
        
    psub_thr - float, optional, default 0.3
        Threshold for correlation difference between paralog pairs and ortholog pairs.
        Paralog pairs that do not have greater than `psub_thr` correlation than their 
        corresponding ortholog pairs are filtered out.
        
    Returns
    -------
    RES - pandas.DataFrame
        A table of paralog substitutions.
        
    """
    if paralog_pairs is not None:
        ids1 = np.array([x.split('_')[0] for x in paralog_pairs[:,0]])
        ids2 = np.array([x.split('_')[0] for x in paralog_pairs[:,1]])
        ix = np.where(ids1==ids2)[0]
        ixnot = np.where(ids1!=ids2)[0]
        
        if ix.size > 0:
            pps = paralog_pairs[ix]
            
            ZZ1 = {}
            ZZ2 = {}
            for i in range(pps.shape[0]):    
                L = ZZ1.get(pps[i,0],[])
                L.append(pps[i,1])
                ZZ1[pps[i,0]]=L

                L = ZZ2.get(pps[i,1],[])
                L.append(pps[i,0])
                ZZ2[pps[i,1]]=L      

            keys = list(ZZ1.keys())
            for k in keys:
                L = ZZ2.get(k,[])
                L.extend(ZZ1[k])
                ZZ2[k] = list(np.unique(L))

            ZZ = ZZ2

            L1=[]
            L2=[]
            for i in range(ortholog_pairs.shape[0]):
                try:
                    x = ZZ[ortholog_pairs[i,0]]
                except:
                    x = []
                L1.extend([ortholog_pairs[i,1]]*len(x))
                L2.extend(x)

                try:
                    x = ZZ[ortholog_pairs[i,1]]
                except:
                    x = []
                L1.extend([ortholog_pairs[i,0]]*len(x))
                L2.extend(x)        

            L = np.vstack((L2,L1)).T
            pps = np.unique(np.sort(L,axis=1),axis=0)
            
            paralog_pairs = np.unique(np.sort(np.vstack((pps,paralog_pairs[ixnot])),axis=1),axis=0)
        
        
    smp = sm.samap
    
    gnnm = smp.adata.varp["homology_graph_reweighted"]
    gn = q(smp.adata.var_names)
    
    ortholog_pairs = np.sort(ortholog_pairs,axis=1)

    ortholog_pairs = ortholog_pairs[np.logical_and(np.in1d(ortholog_pairs[:,0],gn),np.in1d(ortholog_pairs[:,1],gn))]
    if paralog_pairs is None:
        paralog_pairs = gn[np.vstack(smp.adata.varp["homology_graph"].nonzero()).T]
    else:
        paralog_pairs = paralog_pairs[np.logical_and(np.in1d(paralog_pairs[:,0],gn),np.in1d(paralog_pairs[:,1],gn))]
        
    paralog_pairs = np.sort(paralog_pairs,axis=1)        

    paralog_pairs = paralog_pairs[
        np.in1d(to_vn(paralog_pairs), np.append(to_vn(ortholog_pairs),to_vn(ortholog_pairs[:,::-1])), invert=True)
    ]

    A = pd.DataFrame(data=np.arange(gn.size)[None, :], columns=gn)
    xp, yp = (
        A[paralog_pairs[:, 0]].values.flatten(),
        A[paralog_pairs[:, 1]].values.flatten(),
    )
    xp, yp = np.unique(
        np.vstack((np.vstack((xp, yp)).T, np.vstack((yp, xp)).T)), axis=0
    ).T

    xo, yo = (
        A[ortholog_pairs[:, 0]].values.flatten(),
        A[ortholog_pairs[:, 1]].values.flatten(),
    )
    xo, yo = np.unique(
        np.vstack((np.vstack((xo, yo)).T, np.vstack((yo, xo)).T)), axis=0
    ).T
    A = pd.DataFrame(data=np.vstack((xp, yp)).T, columns=["x", "y"])
    pairdict = df_to_dict(A, key_key="x", val_key="y")
    Xp = []
    Yp = []
    Xo = []
    Yo = []
    for i in range(xo.size):
        try:
            y = pairdict[xo[i]]
        except KeyError:
            y = np.array([])
        Yp.extend(y)
        Xp.extend([xo[i]] * y.size)
        Xo.extend([xo[i]] * y.size)
        Yo.extend([yo[i]] * y.size)

    orths = to_vn(gn[np.vstack((np.array(Xo), np.array(Yo))).T])
    paras = to_vn(gn[np.vstack((np.array(Xp), np.array(Yp))).T])
    orth_corrs = gnnm[Xo, Yo].A.flatten()
    par_corrs = gnnm[Xp, Yp].A.flatten()
    diff_corrs = par_corrs - orth_corrs

    RES = pd.DataFrame(
        data=np.vstack((orths, paras)).T, columns=["ortholog pairs", "paralog pairs"]
    )
    RES["ortholog corrs"] = orth_corrs
    RES["paralog corrs"] = par_corrs
    RES["corr diff"] = diff_corrs
    RES = RES.sort_values("corr diff", ascending=False)
    RES = RES[RES["corr diff"] > psub_thr]
    orths = RES['ortholog pairs'].values.flatten()
    paras = RES['paralog pairs'].values.flatten()
    if orths.size > 0:
        orthssp = np.vstack([np.array([x.split('_')[0] for x in xx]) for xx in to_vo(orths)])
        parassp = np.vstack([np.array([x.split('_')[0] for x in xx]) for xx in to_vo(paras)])
        filt=[]
        for i in range(orthssp.shape[0]):
            filt.append(np.in1d(orthssp[i],parassp[i]).mean()==1.0)
        filt=np.array(filt)
        return RES[filt]
    else:
        return RES


def convert_eggnog_to_homologs(sm, EGGs, og_key = 'eggNOG_OGs', taxon=2759):
    """Gets an n x 2 array of homologs at some taxonomic level based on Eggnog results.
    
    Parameters
    ----------
    smp: SAMAP object
    
    EGGs: dict of pandas.DataFrame, Eggnog output tables keyed by species IDs

    og_key: str, optional, default 'eggNOG_OGs'
        The column name of the orthology group mapping results in the Eggnog output tables.

    taxon: int, optional, default 2759
        Taxonomic ID corresponding to the level at which genes with overlapping orthology groups
        will be considered homologs. Defaults to the Eukaryotic level.
        
    Returns
    -------
    homolog_pairs: n x 2 numpy array of homolog pairs.
    """
    smp = sm.samap
    
    taxon = str(taxon)
    EGGs = dict(zip(list(EGGs.keys()),list(EGGs.values()))) #copying
    for k in EGGs.keys():
        EGGs[k] = EGGs[k].copy()

    Es=[]    
    for k in EGGs.keys():
        A=EGGs[k]
        A.index=k+"_"+A.index
        Es.append(A)
    
    A = pd.concat(Es, axis=0)
    gn = q(smp.adata.var_names)
    A = A[np.in1d(q(A.index), gn)]

    orthology_groups = A[og_key]
    og = q(orthology_groups)
    x = np.unique(",".join(og).split(","))
    D = pd.DataFrame(data=np.arange(x.size)[None, :], columns=x)

    for i in range(og.size):
        n = orthology_groups[i].split(",")
        taxa = substr(substr(n, "@", 1),'|',0)
        if (taxa == "2759").sum() > 1 and taxon == '2759':
            og[i] = ""
        else:
            og[i] = "".join(np.array(n)[taxa == taxon])

    A[og_key] = og

    og = q(A[og_key].reindex(gn))
    og[og == "nan"] = ""

    X = []
    Y = []
    for i in range(og.size):
        x = og[i]
        if x != "":
            X.extend(D[x].values.flatten())
            Y.extend([i])

    X = np.array(X)
    Y = np.array(Y)
    B = sp.sparse.lil_matrix((og.size, D.size))
    B[Y, X] = 1
    B = B.tocsr()
    B = B.dot(B.T)
    B.data[:] = 1
    pairs = gn[np.vstack((B.nonzero())).T]
    pairssp = np.vstack([q([x.split('_')[0] for x in xx]) for xx in pairs])
    return np.unique(np.sort(pairs[pairssp[:,0]!=pairssp[:,1]],axis=1),axis=0)


def CellTypeTriangles(sm,keys, align_thr=0.1):
    """Outputs a table of cell type triangles.
    
    Parameters
    ----------
    sm: SAMAP object - assumed to contain at least three species.
       
    keys: dictionary of annotation keys (`.adata.obs[key]`) keyed by  species.

    align_thr: float, optional, default, 0.1
        Only keep triangles with minimum `align_thr` alignment score.        
    """
    
    D,A = get_mapping_scores(sm,keys=keys)
    x,y = A.values.nonzero()
    all_pairsf = np.array([A.index[x],A.columns[y]]).T.astype('str')
    alignmentf = A.values[x,y].flatten()

    alignment = alignmentf.copy()
    all_pairs = all_pairsf.copy()
    all_pairs = all_pairs[alignment > align_thr]
    alignment = alignment[alignment > align_thr]
    all_pairs = to_vn(np.sort(all_pairs, axis=1))

    x, y = substr(all_pairs, ";")
    ctu = np.unique(np.concatenate((x, y)))
    Z = pd.DataFrame(data=np.arange(ctu.size)[None, :], columns=ctu)
    nnm = sp.sparse.lil_matrix((ctu.size,) * 2)
    nnm[Z[x].values.flatten(), Z[y].values.flatten()] = alignment
    nnm[Z[y].values.flatten(), Z[x].values.flatten()] = alignment
    nnm = nnm.tocsr()

    import networkx as nx

    G = nx.Graph()
    gps=ctu[np.vstack(nnm.nonzero()).T]
    G.add_edges_from(gps)
    alignment = pd.Series(index=to_vn(gps),data=nnm.data)
    all_triangles = [c for c in nx.enumerate_all_cliques(G) if len(c)==3]
    Z = np.sort(np.vstack(all_triangles), axis=1)
    DF = pd.DataFrame(data=Z, columns=[x.split("_")[0] for x in Z[0]])
    for i,sid1 in enumerate(sm.ids):
        for sid2 in sm.ids[i:]:
            if sid1!=sid2:
                DF[sid1+';'+sid2] = [alignment[x] for x in DF[sid1].values.astype('str').astype('object')+';'+DF[sid2].values.astype('str').astype('object')]
    DF = DF[sm.ids]
    return DF

def GeneTriangles(sm,orth,keys=None,compute_markers=True,corr_thr=0.3, psub_thr = 0.3, pval_thr=1e-10):
    """Outputs a table of gene triangles.
    
    Parameters
    ----------
    sm: SAMAP object which contains at least three species
    
    orths: (n x 2) ortholog pairs
    
    keys: dict of strings corresponding to each species annotation column keyed by species, optional, default None
        If you'd like to include information about where each gene is differentially expressed, you can specify the
        annotation column to compute differential expressivity from for each species.

    compute_markers: bool, optional, default True
        Set this to False if you already precomputed differential expression for the input keys.
        
    corr_thr: float, optional, default, 0.3
        Only keep triangles with minimum `corr_thr` correlation.
        
    pval_thr: float, optional, defaul, 1e-10
        Consider cell types as differentially expressed if their p-values are less than `pval_thr`.
    """
    FINALS = []

    orth = np.sort(orth,axis=1)
    orthsp = np.vstack([q([x.split('_')[0] for x in xx]) for xx in orth])

    RES = ParalogSubstitutions(sm, orth, psub_thr = psub_thr)
    if RES.shape[0] > 0:
        op = to_vo(q(RES['ortholog pairs']))
        pp = to_vo(q(RES['paralog pairs']))
        ops = np.vstack([q([x.split('_')[0] for x in xx]) for xx in op])
        pps = np.vstack([q([x.split('_')[0] for x in xx]) for xx in pp])
        doPsubsAll=True
    else:
        doPsubsAll=False
    gnnm = sm.samap.adata.varp["homology_graph_reweighted"]
    gn = q(sm.samap.adata.var_names)
    gnsp = q([x.split('_')[0] for x in gn])

    import itertools
    combs = list(itertools.combinations(sm.ids,3))
    for comb in combs:
        A,B,C = comb
        smp1 = SAM(counts=sm.samap.adata[np.logical_or(sm.samap.adata.obs['species']==A,sm.samap.adata.obs['species']==B)])
        smp2 = SAM(counts=sm.samap.adata[np.logical_or(sm.samap.adata.obs['species']==A,sm.samap.adata.obs['species']==C)])
        smp3 = SAM(counts=sm.samap.adata[np.logical_or(sm.samap.adata.obs['species']==B,sm.samap.adata.obs['species']==C)])

        sam1=sm.sams[A]
        sam2=sm.sams[B]
        sam3=sm.sams[C]
        A1,A2=A,B
        B1,B2=A,C
        C1,C2=B,C

        f1 = ((orthsp[:,0]==A1) * (orthsp[:,1]==A2) + (orthsp[:,0]==A2) * (orthsp[:,1]==A1)) > 0
        f2 = ((orthsp[:,0]==B1) * (orthsp[:,1]==B2) + (orthsp[:,0]==B2) * (orthsp[:,1]==B1)) > 0
        f3 = ((orthsp[:,0]==C1) * (orthsp[:,1]==C2) + (orthsp[:,0]==C2) * (orthsp[:,1]==C1)) > 0
        orth1 = orth[f1]
        orth2 = orth[f2]
        orth3 = orth[f3]


        gnnm1 = sp.sparse.vstack((
                                    sp.sparse.hstack((sp.sparse.csr_matrix(((gnsp==A1).sum(),)*2),gnnm[gnsp==A1,:][:,gnsp==A2])),
                                    sp.sparse.hstack((gnnm[gnsp==A2,:][:,gnsp==A1],sp.sparse.csr_matrix(((gnsp==A2).sum(),)*2)))
                                )).tocsr()
        gnnm2 = sp.sparse.vstack((
                                    sp.sparse.hstack((sp.sparse.csr_matrix(((gnsp==B1).sum(),)*2),gnnm[gnsp==B1,:][:,gnsp==B2])),
                                    sp.sparse.hstack((gnnm[gnsp==B2,:][:,gnsp==B1],sp.sparse.csr_matrix(((gnsp==B2).sum(),)*2)))
                                )).tocsr()
        gnnm3 = sp.sparse.vstack((
                                    sp.sparse.hstack((sp.sparse.csr_matrix(((gnsp==C1).sum(),)*2),gnnm[gnsp==C1,:][:,gnsp==C2])),
                                    sp.sparse.hstack((gnnm[gnsp==C2,:][:,gnsp==C1],sp.sparse.csr_matrix(((gnsp==C2).sum(),)*2)))
                                )).tocsr()                                                                
        gn1 = np.append(gn[gnsp==A1],gn[gnsp==A2])
        gn2 = np.append(gn[gnsp==B1],gn[gnsp==B2])
        gn3 = np.append(gn[gnsp==C1],gn[gnsp==C2])


        if doPsubsAll:
            f1 = np.logical_and(((ops[:,0]==A1) * (ops[:,1]==A2) + (ops[:,0]==A2) * (ops[:,1]==A1)) > 0,
                                ((pps[:,0]==A1) * (pps[:,1]==A2) + (pps[:,0]==A2) * (pps[:,1]==A1)) > 0)
            f2 = np.logical_and(((ops[:,0]==B1) * (ops[:,1]==B2) + (ops[:,0]==B2) * (ops[:,1]==B1)) > 0,
                                ((pps[:,0]==B1) * (pps[:,1]==B2) + (pps[:,0]==B2) * (pps[:,1]==B1)) > 0)
            f3 = np.logical_and(((ops[:,0]==C1) * (ops[:,1]==C2) + (ops[:,0]==C2) * (ops[:,1]==C1)) > 0,
                                ((pps[:,0]==C1) * (pps[:,1]==C2) + (pps[:,0]==C2) * (pps[:,1]==C1)) > 0)
            doPsubs = f1.sum() > 0 and f2.sum() > 0 and f3.sum() > 0
        else:
            doPsubs = False

        if doPsubs:
            RES1=RES[f1]
            RES2=RES[f2]
            RES3=RES[f3]


            op1 = to_vo(q(RES1["ortholog pairs"]))
            op2 = to_vo(q(RES2["ortholog pairs"]))
            op3 = to_vo(q(RES3["ortholog pairs"]))
            pp1 = to_vo(q(RES1["paralog pairs"]))
            pp2 = to_vo(q(RES2["paralog pairs"]))
            pp3 = to_vo(q(RES3["paralog pairs"]))


            # suppress warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                T1 = pd.DataFrame(data=np.arange(gn1.size)[None, :], columns=gn1)
                x, y = T1[op1[:, 0]].values.flatten(), T1[op1[:, 1]].values.flatten()
                gnnm1[x, y] = gnnm1[x, y]
                gnnm1[y, x] = gnnm1[y, x]

                T1 = pd.DataFrame(data=np.arange(gn2.size)[None, :], columns=gn2)
                x, y = T1[op2[:, 0]].values.flatten(), T1[op2[:, 1]].values.flatten()
                gnnm2[x, y] = gnnm2[x, y]
                gnnm2[y, x] = gnnm2[y, x]

                T1 = pd.DataFrame(data=np.arange(gn3.size)[None, :], columns=gn3)
                x, y = T1[op3[:, 0]].values.flatten(), T1[op3[:, 1]].values.flatten()
                gnnm3[x, y] = gnnm3[x, y]
                gnnm3[y, x] = gnnm3[y, x]

            gnnm1.data[gnnm1.data==0]=1e-4
            gnnm2.data[gnnm2.data==0]=1e-4
            gnnm3.data[gnnm3.data==0]=1e-4

        pairs1 = gn1[np.vstack(gnnm1.nonzero()).T]
        pairs2 = gn2[np.vstack(gnnm2.nonzero()).T]
        pairs3 = gn3[np.vstack(gnnm3.nonzero()).T]
        data = np.concatenate((gnnm1.data, gnnm2.data, gnnm3.data))

        CORR1 = pd.DataFrame(data=gnnm1.data[None, :], columns=to_vn(pairs1))
        CORR2 = pd.DataFrame(data=gnnm2.data[None, :], columns=to_vn(pairs2))
        CORR3 = pd.DataFrame(data=gnnm3.data[None, :], columns=to_vn(pairs3))

        pairs = np.vstack((pairs1, pairs2, pairs3))
        all_genes = np.unique(pairs.flatten())
        Z = pd.DataFrame(data=np.arange(all_genes.size)[None, :], columns=all_genes)
        x, y = Z[pairs[:, 0]].values.flatten(), Z[pairs[:, 1]].values.flatten()
        GNNM = sp.sparse.lil_matrix((all_genes.size,) * 2)
        GNNM[x, y] = data
        GNNM=GNNM.tocsr()
        GNNM.data[GNNM.data<corr_thr]=0
        GNNM.eliminate_zeros()

        G = nx.from_scipy_sparse_array(GNNM, create_using=nx.Graph)
        all_triangles = [c for c in nx.enumerate_all_cliques(G) if len(c)==3]
        Z = all_genes[np.sort(np.vstack(all_triangles), axis=1)] 
        
        DF = pd.DataFrame(data=Z, columns=[x.split("_")[0] for x in Z[0]])
        DF = DF[[A, B, C]]

        orth1DF = pd.DataFrame(data=orth1, columns=[x.split("_")[0] for x in orth1[0]])[
            [A, B]
        ]
        orth2DF = pd.DataFrame(data=orth2, columns=[x.split("_")[0] for x in orth2[0]])[
            [A, C]
        ]
        orth3DF = pd.DataFrame(data=orth3, columns=[x.split("_")[0] for x in orth3[0]])[
            [B, C]
        ]

        if doPsubs:
            ps1DF = pd.DataFrame(
                data=np.sort(pp1, axis=1),
                columns=[x.split("_")[0] for x in np.sort(pp1, axis=1)[0]],
            )[[A, B]]
            ps2DF = pd.DataFrame(
                data=np.sort(pp2, axis=1),
                columns=[x.split("_")[0] for x in np.sort(pp2, axis=1)[0]],
            )[[A, C]]
            ps3DF = pd.DataFrame(
                data=np.sort(pp3, axis=1),
                columns=[x.split("_")[0] for x in np.sort(pp3, axis=1)[0]],
            )[[B, C]]

            A_AB = pd.DataFrame(data=to_vn(op1)[None, :], columns=to_vn(ps1DF.values))
            A_AC = pd.DataFrame(data=to_vn(op2)[None, :], columns=to_vn(ps2DF.values))
            A_BC = pd.DataFrame(data=to_vn(op3)[None, :], columns=to_vn(ps3DF.values))
        else:
            ps1DF,ps2DF,ps3DF = None,None,None
            A_AB,A_AC,A_BC = None,None,None

        AB = to_vn(DF[[A, B]].values)
        AC = to_vn(DF[[A, C]].values)
        BC = to_vn(DF[[B, C]].values)

        AVs = []
        CATs = []
        CORRs = []
        for i, X, O, P, Z, R in zip(
            [0, 1, 2],
            [AB, AC, BC],
            [orth1DF, orth2DF, orth3DF],
            [ps1DF, ps2DF, ps3DF],
            [A_AB, A_AC, A_BC],
            [CORR1, CORR2, CORR3],
        ):
            cat = q(["homolog"] * X.size).astype("object")
            cat[np.in1d(X, to_vn(O.values))] = "ortholog"
            AV = np.zeros(X.size, dtype="object")
            if doPsubs:
                ff = np.in1d(X, to_vn(P.values))
                cat[ff] = "substitution"

                z = Z[X[ff]] #problem line here
                x = X[ff]
                av = np.zeros(x.size, dtype="object")
                for ai in range(x.size):
                    v=pd.DataFrame(z[x[ai]]) #get ortholog pairs - paralog pairs dataframe
                    vd=v.values.flatten() #get ortholog pairs
                    vc=q(';'.join(v.columns).split(';')) # get paralogous genes
                    temp = np.unique(q(';'.join(vd).split(';'))) #get orthologous genes
                    av[ai] = ';'.join(temp[np.in1d(temp,vc,invert=True)]) #get orthologous genes not present in paralogous genes
                AV[ff] = av
            corr = R[X].values.flatten()
            AVs.append(AV)
            CATs.append(cat)
            CORRs.append(corr)

        tri_pairs = np.vstack((AB, AC, BC)).T
        cat_pairs = np.vstack(CATs).T
        corr_pairs = np.vstack(CORRs).T
        homology_triangles = DF.values
        substituted_genes = np.vstack(AVs).T
        substituted_genes[substituted_genes == 0] = "N.S."
        data = np.hstack(
            (
                homology_triangles.astype("object"),
                substituted_genes.astype("object"),
                tri_pairs.astype("object"),
                corr_pairs.astype("object"),
                cat_pairs.astype("object"),
            )
        )

        FINAL = pd.DataFrame(data = data, columns = [f'{A} gene',f'{B} gene',f'{C} gene',
                                                    f'{A}/{B} subbed',f'{A}/{C} subbed',f'{B}/{C} subbed',
                                                    f'{A}/{B}',f'{A}/{C}',f'{B}/{C}',
                                                    f'{A}/{B} corr',f'{A}/{C} corr',f'{B}/{C} corr',
                                                    f'{A}/{B} type',f'{A}/{C} type',f'{B}/{C} type'])
        FINAL['#orthologs'] = (cat_pairs=='ortholog').sum(1)
        FINAL['#substitutions'] = (cat_pairs=='substitution').sum(1)    
        FINAL = FINAL[(FINAL['#orthologs']+FINAL['#substitutions'])==3]
        x = FINAL[[f'{A}/{B} corr',f'{A}/{C} corr',f'{B}/{C} corr']].min(1)
        FINAL['min_corr'] = x
        FINAL = FINAL[x>corr_thr]
        if keys is not None:
            keys = [keys[A],keys[B],keys[C]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if keys is not None:
                for i,sam,n in zip([0,1,2],[sam1,sam2,sam3],[A,B,C]):
                    if compute_markers:
                        find_cluster_markers(sam,keys[i])
                    a = sam.adata.varm[keys[i]+'_scores'].T[q(FINAL[n+' gene'])].T
                    p = sam.adata.varm[keys[i]+'_pvals'].T[q(FINAL[n+' gene'])].T.values
                    p[p>pval_thr]=1
                    p[p<1]=0
                    p=1-p
                    f = a.columns[a.values.argmax(1)]
                    res=[]
                    for i in range(p.shape[0]):
                        res.append(';'.join(np.unique(np.append(f[i],a.columns[p[i,:]==1]))))            
                    FINAL[n+' cell type'] = res
        FINAL = FINAL.sort_values('min_corr',ascending=False)
        FINALS.append(FINAL)
    FINAL = pd.concat(FINALS,axis=0)
    return FINAL

def _compute_csim(samap, key, X=None, prepend=True, n_top = 0):
    splabels = q(samap.adata.obs['species'])
    skeys = splabels[np.sort(np.unique(splabels,return_index=True)[1])]

    cl = []
    clu = []
    for sid in skeys:
        if prepend:
            cl.append(sid+'_'+q(samap.adata.obs[key])[samap.adata.obs['species']==sid].astype('str').astype('object'))
        else:
            cl.append(q(samap.adata.obs[key])[samap.adata.obs['species']==sid])            
        clu.append(np.unique(cl[-1]))

    clu = np.concatenate(clu)
    cl = np.concatenate(cl)

    CSIM = np.zeros((clu.size, clu.size))
    if X is None:
        X = samap.adata.obsp["connectivities"].copy()

    xi,yi = X.nonzero()
    spxi = splabels[xi]
    spyi = splabels[yi]

    filt = spxi!=spyi
    di = X.data[filt]
    xi = xi[filt]
    yi = yi[filt]

    px,py = xi,cl[yi]
    p = px.astype('str').astype('object')+';'+py.astype('object')

    A = pd.DataFrame(data=np.vstack((p, di)).T, columns=["x", "y"])
    valdict = df_to_dict(A, key_key="x", val_key="y")   
    cell_scores = [valdict[k].sum() for k in valdict.keys()]
    ixer = pd.Series(data=np.arange(clu.size),index=clu)
    if len(valdict.keys())>0:
        xc,yc = substr(list(valdict.keys()),';')
        xc = xc.astype('int')
        yc=ixer[yc].values
        cell_cluster_scores = sp.sparse.coo_matrix((cell_scores,(xc,yc)),shape=(X.shape[0],clu.size)).A

        for i, c in enumerate(clu):
            if n_top > 0:
                CSIM[i, :] = np.sort(cell_cluster_scores[cl==c],axis=0)[-n_top:].mean(0)
            else:
                CSIM[i, :] = cell_cluster_scores[cl==c].mean(0)

        CSIM = np.stack((CSIM,CSIM.T),axis=2).max(2)
        CSIMth = CSIM / samap.adata.uns['mapping_K']
        return CSIMth,clu
    else:
        return np.zeros((clu.size, clu.size)), clu

def get_mapping_scores(sm, keys, n_top = 0):
    """Calculate mapping scores
    Parameters
    ----------
    sm: SAMAP object
    
    keys: dict, annotation vector keys for at least two species with species identifiers as the keys
        e.g. {'pl':'tissue','sc':'tissue'}
    
    n_top: int, optional, default 0
        If `n_top` is 0, average the alignment scores for all cells in a pair of clusters.
        Otherwise, average the alignment scores of the top `n_top` cells in a pair of clusters.
        Set this to non-zero if you suspect there to be subpopulations of your cell types mapping
        to distinct cell types in the other species.
    Returns
    -------
    D - table of highest mapping scores for cell types 
    A - pairwise table of mapping scores between cell types across species
    """
    

    if len(list(keys.keys()))<len(list(sm.sams.keys())):
        samap = SAM(counts = sm.samap.adata[np.in1d(sm.samap.adata.obs['species'],list(keys.keys()))])
    else:
        samap=sm.samap
    
    clusters = []
    ix = np.unique(samap.adata.obs['species'],return_index=True)[1]
    skeys = q(samap.adata.obs['species'])[np.sort(ix)]
    
    for sid in skeys:
        clusters.append(q([sid+'_'+str(x) for x in sm.sams[sid].adata.obs[keys[sid]]]))
    
    cl = np.concatenate(clusters)
    l = "{}_mapping_scores".format(';'.join([keys[sid] for sid in skeys]))
    samap.adata.obs[l] = pd.Categorical(cl)
    
    CSIMth, clu = _compute_csim(samap, l, n_top = n_top, prepend = False)

    A = pd.DataFrame(data=CSIMth, index=clu, columns=clu)
    i = np.argsort(-A.values.max(0).flatten())
    H = []
    C = []
    for I in range(A.shape[1]):
        x = A.iloc[:, i[I]].sort_values(ascending=False)
        H.append(np.vstack((x.index, x.values)).T)
        C.append(A.columns[i[I]])
        C.append(A.columns[i[I]])
    H = np.hstack(H)
    D = pd.DataFrame(data=H, columns=[C, ["Cluster","Alignment score"]*(H.shape[1]//2)])
    return D, A


def _knndist(nnma, k):
    x, y = nnma.nonzero()
    data = nnma.data
    xc, cc = np.unique(x, return_counts=True)
    cc2 = np.zeros(nnma.shape[0], dtype="int")
    cc2[xc] = cc
    cc = cc2
    newx = []
    newdata = []
    for i in range(nnma.shape[0]):
        newx.extend([i] * k)
        newdata.extend(list(data[x == i]) + [0] * (k - cc[i]))
    data = np.array(newdata)
    val = data.reshape((nnma.shape[0], k))
    return val


def _sparse_sub_standardize(X, mu, var, rows=False):
    x, y = X.nonzero()
    if not rows:
        Xs = X.copy()
        Xs.data[:] = (X.data - mu[y]) / var[y]
    else:
        mu, var = sf.mean_variance_axis(X, axis=1)
        var = var ** 0.5
        var[var == 0] = 1
        Xs = X.copy()
        Xs.data[:] = (X.data - mu[x]) / var[x]
    Xs.data[Xs.data < 0] = 0
    Xs.eliminate_zeros()
    return Xs

def _get_mu_std(sam3, sam1, sam2, knn=False):
    g1, g2 = ut.extract_annotation(sam3.adata.uns['gene_pairs'], 0, ";"), ut.extract_annotation(
        sam3.adata.uns['gene_pairs'], 1, ";"
    )
    if knn:
        mu1, var1 = sf.mean_variance_axis(sam1.adata[:, g1].layers["X_knn_avg"], axis=0)
        mu2, var2 = sf.mean_variance_axis(sam2.adata[:, g2].layers["X_knn_avg"], axis=0)
    else:
        mu1, var1 = sf.mean_variance_axis(sam1.adata[:, g1].X, axis=0)
        mu2, var2 = sf.mean_variance_axis(sam2.adata[:, g2].X, axis=0)
    var1[var1 == 0] = 1
    var2[var2 == 0] = 1
    var1 = var1 ** 0.5
    var2 = var2 ** 0.5
    return mu1, var1, mu2, var2    

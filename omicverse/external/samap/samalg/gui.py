import pickle
import colorlover as cl
import numpy as np
import scipy.sparse as sp
import ipywidgets as widgets
import plotly.graph_objs as go
from . import SAM
from . import utilities as ut
import pandas as pd

from ipyevents import Event
from ipywidgets import Widget, Layout
import warnings
from numba.core.errors import NumbaWarning
warnings.filterwarnings("ignore", category=NumbaWarning)

__version__ = "0.8.7"


class SAMGUI(object):
    def __init__(self, sam=None,title=None, close_all_widgets=False, default_proj='X_umap',height=600):
        self.ACTION_LOG = []
        
        if close_all_widgets:
            Widget.close_all()
        self.widget_height = height
        self.log('Setting widget height to {}'.format(height))
        
        self.default_proj = default_proj
        self.titles = []
        self.stab = widgets.Tab(layout={"height": "95%", "width": "50%"})
        self.stab.observe(self.on_switch_tabs, "selected_index")

        if sam is not None:
            self.log('Initializing GUI from input SAM object.')
            if not isinstance(sam,list):
                sams=[sam]
            else:
                sams = sam
                sam = sams[0] 
            if title is not None:
                if not isinstance(title,list):
                    titles=[title]
                else:
                    titles = title
                    title = titles[0]
            else:
                titles = [None]*len(sams)
            
            self.init_from_sam(sam)
            if titles[0] is None:
                title = 'Full dataset'
            else:
                title = titles[0]
                
            self.create_plot(0, title)
            items = [self.stab, self.tab]
            self.titles = [title]
            for i in range(1,len(sams)):
                self.sams.append(sams[i])
                self.selected.append(np.ones(sams[i].adata.shape[0]).astype("bool"))
                self.selected_cells.append(np.array(list(sams[i].adata.obs_names)))
                self.active_labels.append(np.zeros(sams[i].adata.shape[0]))
                self.dd_opts.append(["Toggle cluster"])
                self.gene_expressions.append(np.zeros(sams[i].adata.shape[0]))
                self.marker_genes.append(
                    np.array(list(sams[i].adata.var_names))[
                        np.argsort(-sams[i].adata.var["weights"].values)
                    ]
                )
                self.marker_genes_tt.append("Genes ranked by SAM weights.")
                self.ds.append(0)  
                if titles[i] is None:
                    title = 'Full dataset {}'.format(i)
                else:
                    title = titles[i]
                self.titles.append(title)
                self.create_plot(i, title)
                self.update_dropdowns(i)            
        else:
            self.log('Initializing blank GUI.')
            tab = widgets.Tab(layout={"height": "95%", "width": "50%"})
            load_box = self.init_load()
            children = [load_box]
            self.SAM_LOADED = False
            tab.children = children
            tab.set_title(0, "LOAD_SAM")
            self.create_plot(0, "SAM not loaded")
            self.titles = ["SAM not loaded"]
            items = [self.stab, tab]

        self.current_tab = 0
        self.current_sam = sam

        self.SamPlot = widgets.HBox(
            items, layout=Layout(width="auto", height=str(height) + "px")
        )

        x = self.stab.children[self.stab.selected_index]
        f = x.layout.xaxis.range
        x.layout.xaxis.range=(-1,1)
        x.layout.xaxis.range=f        

        f = x.layout.yaxis.range
        x.layout.yaxis.range=(-1,1)
        x.layout.yaxis.range=f
        
    def __setstate__(self,d):       
        self.__init__(sam = d['sams'],title=d['titles'],
                      height=d['widget_height'],default_proj=d['default_proj'])
        
        for k in d.keys():
            self.__dict__[k] = d[k]
        self.__dict__['ds'] = [0]*len(self.sams)

        for i in range(len(self.sams)):
            self.create_plot(i,self.titles[i])
        
    def __getstate__(self):
        d = {}
        exclude = ['SamPlot','children','ps_dict','rb_dict','cs_dict','stab','rs_box','cs_box',
                   'pp_box','out','tab','ds']
        for k in self.__dict__.keys():
            if k not in exclude:
                d[k] = self.__dict__[k]
        return d
        


    def log(self,s):
        self.ACTION_LOG.append(s)
        
    def init_from_sam(self, sam):
        self.current_sam = sam
        tab = widgets.Tab(layout={"height": "95%", "width": "50%"})
        self.load_vars_from_sam(sam)
        self.pp_box = self.init_preprocess()
        self.rs_box = self.init_run_sam()
        self.cs_box = self.init_cs()
        
        self.SAM_LOADED = True
        self.out = widgets.Output(
            layout={"border": "1px solid black", "height": "95%", "width": "100%"}
        )
        children = [self.cs_box, self.rs_box, self.pp_box, self.out]
        self.children = children
        names = ["Interact (0)", "Analysis (0)", "Preprocess (0)", "Output (0)"]
        self.names = ["Interact", "Analysis", "Preprocess", "Output"]
        tab.children = children
        tab.set_trait("selected_index", 0)
        for i in range(len(children)):
            tab.set_title(i, names[i])
        tab.set_trait("selected_index", 0)
        self.tab = tab

    def close_all_tabs(self):
        self.log('Closing all tabs except tab 0.')
        self.stab.set_trait("selected_index", 0)
        I = 1
        while len(self.sams) > 1:
            self.ds[I].close()
            self.stab.children[I].close()
            del self.ds[I]
            del self.sams[I]
            del self.selected[I]
            del self.selected_cells[I]
            del self.active_labels[I]
            del self.dd_opts[I]
            del self.marker_genes[I]
            del self.marker_genes_tt[I]
            del self.gene_expressions[I]
        self.stab.children = (self.stab.children[0],)

    def get_scatter(self, i):
        return self.stab.children[i].data[0]

    def load_vars_from_sam(self, sam):
        self.sams = [sam]
        self.GENE_KEY = ""
        self.selected = [np.zeros(sam.adata.shape[0], dtype="bool")]
        self.selected[0][:] = True
        self.active_labels = [np.zeros(self.selected[0].size, dtype="int")]
        self.dd_opts = [["Toggle cluster"]]
        try:
            self.marker_genes = [
                np.array(list(sam.adata.var_names))[
                    np.argsort(-sam.adata.var["weights"].values)
                ]
            ]
            self.marker_genes_tt = ["Genes ranked by SAM weights."]
        except KeyError:
            self.marker_genes = [np.array(list(sam.adata.var_names))]
            self.marker_genes_tt = ["Unranked genes (SAM not run)."]

        self.selected_cells = [np.array(list(sam.adata.obs_names))]
        self.ds = [0]
        self.gene_expressions = [np.zeros(sam.adata.shape[0])]
        try:
            self.preprocess_args = sam.adata.uns["preprocess_args"].copy()
            for i in list(self.preprocess_args.keys()):
                if isinstance(self.preprocess_args[i], np.ndarray):
                    self.preprocess_args[i] = self.preprocess_args[i][0]
        except:
            self.preprocess_args = {}

        rm = list(invalidArgs(self.sams[0].preprocess_data,self.preprocess_args))
        for k in rm:
            del self.preprocess_args[k]            
            
        self.preprocess_args['thresh'] = self.preprocess_args.get('thresh_low',0.0)
        self.preprocess_args_init = self.preprocess_args.copy()

        try:
            self.run_args = sam.adata.uns["run_args"].copy()
            for i in list(self.run_args.keys()):
                if isinstance(self.run_args[i], np.ndarray):
                    self.run_args[i] = self.run_args[i][0]
        except:
            self.run_args = {}
        
        rm = list(invalidArgs(self.sams[0].run,self.run_args))
        for k in rm:
            del self.run_args[k]
        self.run_args_init = self.run_args.copy()
        


    def create_plot(self, i, title):
        if self.SAM_LOADED:
            projs = list(self.sams[i].adata.obsm.keys())
            if self.default_proj in projs:
                p = self.default_proj
            elif len(projs) > 1:
                p = np.array(projs)[np.where(np.array(projs) != "X_pca")[0][0]]
            else:
                p = "X_pca"

            if p not in projs:
                xdata = []
                ydata = []
            else:
                xdata = self.sams[i].adata.obsm[p][:, 0]
                ydata = self.sams[i].adata.obsm[p][:, 1]

            if i < len(self.stab.children):
                f1 = self.stab.children[i]
                f1.update_layout(autosize=False)
                f1.data = []
            else:
                f1 = go.FigureWidget()

            f1.add_scattergl(x=xdata, y=ydata)

            f1.for_each_trace(self.init_graph)
            f1.update_layout(
                margin_l=0,
                margin_r=0,
                margin_t=40,
                margin_b=0,
                xaxis_ticks="",
                xaxis_showticklabels=False,
                height=self.widget_height * 0.85,
                title="",
                # xaxis_showgrid=False,
                # xaxis_zeroline=False,
                # yaxis_showgrid=False,
                # yaxis_zeroline=False,
                yaxis_ticks="",
                yaxis_showticklabels=False,
                autosize=True,
                dragmode="select",
            )
            f1.update_yaxes(autorange=True)
            f1.update_xaxes(autorange=True)

            f1.data[0].text = list(self.active_labels[i])
            f1.data[0].hoverinfo = "text"
            f1.set_trait(
                "_config",
                {
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "edits": {"titleText": False},
                },
            )

            if i >= len(self.stab.children):
                self.stab.children += (f1,)
                # self.stab.set_trait('selected_index',i)

            if type(self.ds[i]) is int:
                d = Event(source=f1, watched_events=["keydown"])
                self.ds[i] = d
                d.on_dom_event(self.handle_events)

            slider = self.cs_dict["THR"]
            slider.set_trait("min", slider.value)
            slider.set_trait("max", slider.value)

        else:
            f1 = go.FigureWidget()
            f1.add_scattergl(x=[], y=[])
            f1.update_layout(
                margin_l=0,
                margin_r=0,
                margin_t=40,
                margin_b=0,
                height=self.widget_height * 0.85,
                xaxis_ticks="",
                xaxis_showticklabels=False,
                yaxis_ticks="",
                yaxis_showticklabels=False,
                autosize=True,
                dragmode="select",
            )
            f1.set_trait(
                "_config",
                {
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "edits": {"titleText": False},
                },
            )
            self.stab.children += (f1,)

        self.stab.set_title(i, title)

    def handle_events(self, event):
        if event["type"] == "keydown":
            key = event["key"]
            if key == "ArrowRight":
                self.log('Scrolling ranked genes slider one to the right.')
                self.cs_dict["RGENES"].set_trait(
                    "value", self.cs_dict["RGENES"].value + 1
                )
            elif key == "ArrowLeft":
                self.log('Scrolling ranked genes slider one to the left.')
                x = self.cs_dict["RGENES"].value - 1
                if x < 0:
                    x = 0
                self.cs_dict["RGENES"].set_trait("value", x)

            elif key == "Shift":
                self.irm_genes(None)
            elif key == "Enter":
                self.ism_genes(None)
            elif key == "x":
                self.unselect_all(None)
            elif key == "c":
                self.select_all(None)
            elif key == "v":
                self.reset_view(None)
            elif key == "a":
                self.log('Turning off k-nn averaging of gene expression values.')
                self.cs_dict["AVG"].set_trait("value", not self.cs_dict["AVG"].value)

    def close_tab(self, event):
        I = self.stab.selected_index
        if I > 0:
            self.log('Closing tab {}'.format(I))
            self.stab.set_trait("selected_index", I - 1)

            titles = []
            for i in range(len(self.stab.children)):
                titles.append(self.stab.get_title(i))
            del titles[I]

            self.ds[I].close()
            self.stab.children[I].close()
            t = list(self.stab.children)
            del t[I]
            self.stab.children = t
            del self.ds[I]
            del self.sams[I]
            del self.selected[I]
            del self.selected_cells[I]
            del self.gene_expressions[I]
            del self.active_labels[I]
            del self.dd_opts[I]
            del self.marker_genes[I]
            del self.marker_genes_tt[I]

            for i in range(1, len(self.stab.children)):
                self.stab.set_title(i, titles[i])

    """ BEGIN PREPROCESS INIT"""

    def init_preprocess(self):

        pdata = widgets.Button(
            description = 'Preprocess',
            disabled = False,
            tooltip = 'Preprocess data',
            icon = ''
        )
        pdata.on_click(self.preprocess_sam)

        fgenes = widgets.Checkbox(
            value=bool(self.preprocess_args.get("filter_genes", True)),
            description="Filter genes",
        )
        fgenes.observe(self.pp_filtergenes, names="value")
        dfts = widgets.Button(
            description="Set defaults",
            disabled=False,
            tooltip="Set default options",
            icon="",
        )
        dfts.on_click(self.set_pp_defaults)
        
        l1 = widgets.Label("Expr threshold:")
        expr_thr = widgets.FloatSlider(
            value=float(self.preprocess_args.get("thresh", 0.01)),
            min=0,
            max=0.1,
            step=0.005,
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="2f",
            style={"description_width": "initial"},
        )
        expr_thr.observe(self.et_update, names="value")

        l2 = widgets.Label("Min expr:")
        min_expr = widgets.FloatSlider(
            value=float(self.preprocess_args.get("min_expression", 1)),
            min=0,
            max=6.0,
            step=0.02,
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="1f",
            style={"description_width": "initial"},
        )
        min_expr.observe(self.me_update, names="value")

        init = self.preprocess_args.get("sum_norm", "None")
        if isinstance(init, np.ndarray):
            init = init[0]
        if init is None:
            init = "None"
        sumnorm = widgets.Dropdown(
            options=["cell_median", "gene_median", "None"],
            value=init,
            description="Library normalization:",
            disabled=False,
            style={"description_width": "initial"},
        )
        sumnorm.observe(self.sumnorm_submit, "value")

        init = self.preprocess_args.get("norm", "log")
        if isinstance(init, np.ndarray):
            init = init[0]
        if init is None:
            init = "None"
        norm = widgets.Dropdown(
            options=["log", "ftt", "None"],
            value=init,
            description="Data normalization:",
            disabled=False,
            style={"description_width": "initial"},
        )
        norm.observe(self.norm_submit, "value")

        load = widgets.Button(
            description="Load data",
            tooltip="Enter the path to the desired data file you wish to "
            "load. Accepted filetypes are .csv (comma), .txt (tab), .h5ad "
            "(AnnData), and .p (pickled SAM dictionary).",
            disabled=False,
        )
        load.on_click(self.load_data)
        load_data = widgets.Text(value="", layout={"width": "100%%"})

        loada = widgets.Button(
            description="Load annotations",
            tooltip="Enter the path to the desired annotations file you wish to "
            "load. Accepted filetypes are .csv (comma) or .txt (tab).",
            disabled=False,
        )
        loada.on_click(self.load_ann)
        load_dataa = widgets.Text(value="", layout={"width": "100%%"})

        loadv = widgets.Button(
            description="Load gene annotations",
            tooltip="Enter the path to the desired gene annotations file "
            "you wish to load. Accepted filetypes are .csv (comma) or .txt (tab).",
            disabled=False,
        )
        loadv.on_click(self.load_vann)
        load_datav = widgets.Text(value="", layout={"width": "100%%"})

        self.ps_dict = {}
        self.ps_dict["PDATA"] = pdata
        self.ps_dict["DFTS"] = dfts
        self.ps_dict["FGENES"] = fgenes
        self.ps_dict["NORM"] = norm
        self.ps_dict["SUMNORM"] = sumnorm
        self.ps_dict["L1"] = l1
        self.ps_dict["EXPR_THR"] = expr_thr
        self.ps_dict["L2"] = l2
        self.ps_dict["MIN_EXPR"] = min_expr
        self.ps_dict["LOAD"] = load
        self.ps_dict["LOAD_DATA"] = load_data
        self.ps_dict["LOADA"] = loada
        self.ps_dict["LOAD_DATAA"] = load_dataa
        self.ps_dict["LOADV"] = loadv
        self.ps_dict["LOAD_DATAV"] = load_datav

        pp = widgets.VBox(
            [   pdata,
                widgets.HBox([dfts, fgenes]),
                norm,
                sumnorm,
                widgets.HBox([l1, expr_thr]),
                widgets.HBox([l2, min_expr]),
                widgets.HBox([load, load_data]),
                widgets.HBox([loada, load_dataa]),
                widgets.HBox([loadv, load_datav]),
            ],
            layout={"height": "95%", "width": "100%"},
        )
        return pp

    def load_ann(self, event):
        path = self.ps_dict["LOAD_DATAA"].value
        self.log('Loading cell annotation file from {}'.format(path))        
        try:
            for i in range(len(self.stab.children)):
                self.sams[i].load_obs_annotations(path)
                self.update_dropdowns(i)
        except:
            with self.out:
                print("Annotation file not found or was improperly formatted.")

    def load_vann(self, event):
        path = self.ps_dict["LOAD_DATAV"].value
        self.log('Loading gene annotation file from {}'.format(path))
        try:
            for i in range(len(self.stab.children)):
                self.sams[i].load_var_annotations(path)
                self.update_dropdowns(i)
        except:
            with self.out:
                print("Annotation file not found or was improperly formatted.")

    def init_load(self):
        load = widgets.Button(
            description="Load data",
            tooltip="Enter the path to the desired data file you wish to "
            "load. Accepted filetypes are .csv (comma), .txt (tab), .h5ad "
            "(AnnData), and .p (pickled SAM dictionary).",
            disabled=False,
        )
        load.on_click(self.load_data)
        load_data = widgets.Text(value="", layout={"width": "100%"})
        return widgets.HBox([load, load_data], layout={"width": "50%"})

    def load_data(self, event):
        try:
            if not self.SAM_LOADED:
                path = self.SamPlot.children[1].children[0].children[1].value
            else:
                path = self.ps_dict["LOAD_DATA"].value
            self.log('Loading data file from {}'.format(path))
            filetype = path.split(".")[-1]
            if filetype == "gz":
                filetype = path.split(".")[-2]

            sam = SAM()
            if filetype == "h5ad" or filetype == "csv":
                sam.load_data(path)
            elif filetype == "p":
                try:
                    sam.load(path)
                except:
                    sam.load_data(path)
            else:
                sam.load_data(path, sep="\t")

            if not self.SAM_LOADED:
                self.SamPlot.children[1].children[0].children[0].close()
                self.SamPlot.children[1].children[0].children[1].close()
                self.SamPlot.children[1].children[0].close()
                self.SamPlot.children[1].close()
                self.init_from_sam(sam)
                self.SamPlot.children = [self.stab, self.tab]
                self.tab.set_trait("selected_index", 1)
            else:
                self.close_all_tabs()
                self.ds[0].close()
                self.ds[0] = 0
                self.load_vars_from_sam(sam)
            self.create_plot(0, "Full dataset")
            if len(self.titles) == 0:
                self.titles = ["Full dataset"]
            else:
                self.titles[0] = "Full dataset"
        except FileNotFoundError:
            0
            # do nothing

    def me_update(self, val):
        self.log('Setting min_expression to be {}'.format(val['new']))
        self.preprocess_args["min_expression"] = val["new"]

    def et_update(self, val):
        self.log('Setting the threshold value to be {}'.format(val['new']))
        self.preprocess_args["thresh"] = val["new"]

    def sumnorm_submit(self, txt):
        self.log('Setting sum_norm value to be {}'.format(txt['new']))
        if txt["new"] == "None":
            self.preprocess_args["sum_norm"] = None
        else:
            self.preprocess_args["sum_norm"] = txt["new"]

    def set_pp_defaults(self, event):
        self.log('Setting preprocessing parameters to defaults.')
        self.preprocess_args = self.preprocess_args_init.copy()
        fgenes = self.ps_dict["FGENES"]  # checkbox
        norm = self.ps_dict["NORM"]
        sumnorm = self.ps_dict["SUMNORM"]
        expr_thr = self.ps_dict["EXPR_THR"]  # expr_threshold
        min_expr = self.ps_dict["MIN_EXPR"]  # min_expr

        init = float(self.preprocess_args.get("min_expression", 1))
        min_expr.set_trait("value", init)

        init = float(self.preprocess_args.get("thresh", 0.01))
        expr_thr.set_trait("value", init)

        init = self.preprocess_args.get("sum_norm", "None")
        if isinstance(init, np.ndarray):
            init = init[0]
        if init is None:
            sumnorm.set_trait("value", "None")
        else:
            sumnorm.set_trait("value", init)

        init = self.preprocess_args.get("norm", "log")
        if isinstance(init, np.ndarray):
            init = init[0]
        if init is None:
            norm.set_trait("value", "None")
        else:
            norm.set_trait("value", init)

        init = bool(self.preprocess_args.get("filter_genes", True))
        fgenes.set_trait("value", True)
        self.preprocess_args["filter_genes"] = init

    def norm_submit(self, txt):
        self.log('Setting norm parameter to be {}'.format(txt['new']))
        if txt["new"] == "None":
            self.preprocess_args["norm"] = None
        else:
            self.preprocess_args["norm"] = txt["new"]

    def pp_filtergenes(self, event):
        t = bool(self.preprocess_args.get("filter_genes", True))
        self.log('Setting filter_genes parameter to be {}'.format(not t))
        self.preprocess_args["filter_genes"] = not t

    def preprocess_sam(self, event):
        i = self.stab.selected_index
        self.log('Preprocessing SAM object in tab {}'.format(i))
        self.sams[i].preprocess_data(**self.preprocess_args)

    """ END PREPROCESS INIT"""
    
    """ BEGIN RUN INIT"""

    def init_run_sam(self):
        cpm = widgets.Dropdown(
            options=["UMAP", "t-SNE", "Diffusion map", "Diffusion UMAP"],
            value="UMAP",
            description="",
            layout={"width": "60%"},
        )

        cp = widgets.Button(
            description="Compute projection",
            tooltip="Compute a 2D projection using the selected method in the dropdown menu to the right.",
            layout={"width": "40%"},
        )
        cp.on_click(self.compute_projection)

        clm = widgets.Dropdown(
            options=[
                "Louvain cluster",
                "Density cluster",
                "Hdbscan cluster",
                "Kmeans cluster",
                "Leiden cluster",
            ],
            value="Leiden cluster",
            description="",
            layout={"width": "60%"},
        )
        clm.observe(self.rewire_cluster_slider, "value")
        cl = widgets.Button(
            description="Cluster",
            tooltip="Cluster the data using the selected method in the dropdown menu to the right.",
            layout={"width": "40%"},
        )
        cl.on_click(self.cluster_data)

        l = widgets.Label("Leiden 'res'", layout={"width": "20%"})
        cslider = widgets.FloatSlider(
            value=1,
            min=0.1,
            max=10,
            step=0.1,
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="2f",
            description="",
            style={"description_width": "initial"},
            layout={"width": "80%"},
        )

        runb = widgets.Button(
            description="Run SAM",
            disabled=False,
            tooltip="Run SAM on the currently selected cells. Enter the title of the new tab in the text box on the right",
            icon="",
        )
        runb.on_click(self.subcluster)
        title = widgets.Text(value="",)
        wpca = widgets.Checkbox(
            value=bool(self.run_args.get("weight_PCs", True)), description="Weight PCs"
        )
        wpca.observe(self.weightpcs)


        spca = widgets.Checkbox(
            value=bool(self.run_args.get("sparse_pca", True)), description="Sparse PCA"
        )
        spca.observe(self.sparsepca)
        

        dfts = widgets.Button(
            description="Set defaults",
            disabled=False,
            tooltip="Set default options",
            icon="",
        )
        dfts.on_click(self.set_run_defaults)

        l1 = widgets.Button(
            description="# genes:",
            disabled=True,
            tooltip="The number of highest-weighted genes to select each iteration of SAM.",
            icon="",
        )
        init = self.run_args.get("n_genes", 3000)
        if isinstance(init, np.ndarray):
            init = init[0]
        if init is None:
            init = 3000
        ngenes = widgets.FloatSlider(
            value=init,
            min=min(100, self.sams[0].adata.shape[1]),
            max=self.sams[0].adata.shape[1],
            step=100,
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="5d",
            style={"description_width": "initial"},
        )
        ngenes.observe(self.ngenes_update, names="value")

        l2 = widgets.Button(
            description="# PCs:",
            disabled=True,
            tooltip="The number of principal components to select in PCA.",
            icon="",
        )
        init = self.run_args.get("npcs", 150)
        if isinstance(init, np.ndarray):
            init = init[0]
        if init is None:
            init = 150
        npcs = widgets.FloatSlider(
            value=init,
            min=10,
            max=500,
            step=1,
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style={"description_width": "initial"},
        )
        npcs.observe(self.npcs_update, names="value")

        l3 = widgets.Button(
            description="# neighbors:",
            disabled=True,
            tooltip="The number of nearest neighbors to identify for each cell.",
            icon="",
        )
        init = self.run_args.get("k", 20)
        if isinstance(init, np.ndarray):
            init = init[0]
        if init is None:
            init = 20
        knn = widgets.FloatSlider(
            value=init,
            min=5,
            max=200,
            step=1,
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style={"description_width": "initial"},
        )
        knn.observe(self.knn_update, names="value")

        l4 = widgets.Button(
            description="num_norm_avg:",
            disabled=True,
            tooltip="The top 'num_norm_avg' dispersions are averaged to determine the "
            "normalization factor when calculating the weights. This prevents "
            "genes with large spatial dispersions from skewing the distribution "
            "of weights.",
            icon="",
        )
        init = self.run_args.get("num_norm_avg", 50)
        if isinstance(init, np.ndarray):
            init = init[0]
        if init is None:
            init = 50
        nna = widgets.FloatSlider(
            value=init,
            min=1,
            max=200,
            step=1,
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            style={"description_width": "initial"},
        )
        nna.observe(self.nna_update, names="value")

        init = self.run_args.get("preprocessing", "Normalizer")
        if isinstance(init, np.ndarray):
            init = init[0]
        if init is None:
            init = "None"
        norm = widgets.Dropdown(
            options=["StandardScaler", "Normalizer", "None"],
            value=init,
            description="Preprocessing:",
            disabled=False,
            style={"description_width": "initial"},
        )
        norm.observe(self.rnorm_update, "value")

        init = self.run_args.get("distance", "correlation")
        if isinstance(init, np.ndarray):
            init = init[0]
        if init is None:
            init = "correlation"
        distance = widgets.Dropdown(
            options=["correlation", "euclidean", "cosine"],
            value=init,
            description="Distance:",
            disabled=False,
            style={"description_width": "initial"},
        )
        distance.observe(self.dist_update, "value")

        init = self.run_args.get("projection", "umap")
        if isinstance(init, np.ndarray):
            init = init[0]
        if init is None:
            init = "umap"
        proj = widgets.Dropdown(
            options=["umap", "tsne", "diff_umap"],
            value=init,
            description="Projection:",
            disabled=False,
            style={"description_width": "initial"},
        )
        proj.observe(self.proj_update, "value")

        self.rb_dict = {}
        self.rb_dict['SPCA'] = spca
        self.rb_dict["RUNB"] = runb
        self.rb_dict["TITLE"] = title
        self.rb_dict["DFTS"] = dfts
        self.rb_dict["WPCA"] = wpca
        self.rb_dict["L3"] = l3
        self.rb_dict["L4"] = l4
        self.rb_dict["NNA"] = nna
        self.rb_dict["KNN"] = knn
        self.rb_dict["NORM"] = norm
        self.rb_dict["DISTANCE"] = distance
        self.rb_dict["PROJ"] = proj
        self.rb_dict["L1"] = l1
        self.rb_dict["L2"] = l2
        self.rb_dict["NGENES"] = ngenes
        self.rb_dict["NPCS"] = npcs
        self.rb_dict["CP"] = cp
        self.rb_dict["CPM"] = cpm
        self.rb_dict["CL"] = cl
        self.rb_dict["CLM"] = clm
        self.rb_dict["L"] = l
        self.rb_dict["CSLIDER"] = cslider

        rs = widgets.VBox(
            [
                widgets.HBox([runb, title]),
                widgets.HBox([dfts, wpca, spca]),
                widgets.HBox([l3, knn]),
                widgets.HBox([l4, nna]),
                norm,
                distance,
                proj,
                widgets.HBox([l1, ngenes]),
                widgets.HBox([l2, npcs]),
                widgets.HBox([cp, cpm]),
                widgets.HBox([cl, clm]),
                widgets.HBox([l, cslider]),
            ],
            layout={"height": "95%", "width": "100%"},
        )

        return rs

    def set_run_defaults(self, event):
        self.log('Setting run parameters to defaults.')
        self.run_args = self.run_args_init.copy()
        # run
        # defaults,wpca
        # ,ngenes
        # ,npcs
        # ,knn
        # ,nna
        # norm,dist,proj
        wpca = self.rb_dict["WPCA"]
        ngenes = self.rb_dict["NGENES"]
        npcs = self.rb_dict["NPCS"]
        knn = self.rb_dict["KNN"]
        nna = self.rb_dict["NNA"]
        rnorm = self.rb_dict["NORM"]
        dist = self.rb_dict["DISTANCE"]
        proj = self.rb_dict["PROJ"]

        init = self.run_args.get("num_norm_avg", 50)
        nna.set_trait("value", init)
        init = self.run_args.get("k", 20)
        knn.set_trait("value", init)

        init = self.run_args.get("npcs", 150)
        if init is None:
            init = 150
        npcs.set_trait("value", init)

        init = self.run_args.get("n_genes", 3000)
        if init is None:
            init = 3000
        ngenes.set_trait("value", init)

        init = self.run_args.get("preprocessing", "Normalizer")
        if init is None:
            init = ""
        rnorm.set_trait("value", init)
        init = self.run_args.get("distance", "correlation")
        dist.set_trait("value", init)
        init = self.run_args.get("projection", "umap")
        proj.set_trait("value", init)

        init = bool(self.run_args.get("weight_PCs", True))
        wpca.set_trait("value", init)
        self.run_args["weight_PCs"] = init

    def sam_weights(self, event):
        self.log('Setting gene rankings to be ordered by the SAM weights.')
        s = self.sams[self.stab.selected_index]
        self.marker_genes[self.stab.selected_index] = np.array(
            list(s.adata.var_names[np.argsort(-s.adata.var["weights"].values)])
        )
        self.marker_genes_tt[self.stab.selected_index] = "Genes ranked by SAM weights."
        self.cs_dict["LGENES"].set_trait(
            "tooltip", self.marker_genes_tt[self.stab.selected_index]
        )

        if self.cs_dict["RGENES"].value != 0:
            self.cs_dict["RGENES"].set_trait("value", 0)
        else:
            self.show_expression(str(self.marker_genes[self.stab.selected_index][0]))

    def weightpcs(self, event):
        t = self.run_args.get("weight_PCs", True)
        self.log('Setting weight_PCs to be {}'.format(not t))
        self.run_args["weight_PCs"] = not t

    def sparsepca(self, event):
        t = self.run_args.get("sparse_pca", True)
        self.log('Setting sparse_pca to be {}'.format(not t))
        self.run_args["sparse_pca"] = not t
        
    def npcs_update(self, val):
        self.log('Setting npcs to be {}'.format(int(val['new'])))
        self.run_args["npcs"] = int(val["new"])

    def nna_update(self, val):
        self.log('Setting num_norm_avg to be {}'.format(int(val['new'])))
        self.run_args["num_norm_avg"] = int(val["new"])

    def knn_update(self, val):
        self.log('Setting k to be {}'.format(int(val['new'])))
        self.run_args["k"] = int(val["new"])

    def ngenes_update(self, val):
        self.log('Setting n_genes to be {}'.format(int(val['new'])))
        self.run_args["n_genes"] = int(val["new"])

    def rnorm_update(self, txt):
        self.log('Setting preprocessing to be {}'.format(txt['new']))
        if txt["new"] == "None":
            self.run_args["preprocessing"] = None
        else:
            self.run_args["preprocessing"] = txt["new"]

    def proj_update(self, txt):
        self.log('Setting projection to be {}'.format(txt['new']))
        self.run_args["projection"] = txt["new"]

    def dist_update(self, txt):
        self.log('Setting distance to be {}'.format(txt['new']))
        self.run_args["distance"] = txt["new"]

    def subcluster(self, event):

        execute = False
        i = self.stab.selected_index

        selected = self.selected[i]
        selected_cells = self.selected_cells[i]

        sam = self.sams[i]
        if not np.all(selected) and selected.sum() > 0:
            sam_subcluster = SAM(counts=sam.adata[selected_cells, :].copy())
            sam_subcluster.adata_raw = sam.adata_raw[selected_cells].copy()

            sam_subcluster.adata_raw.obs = sam_subcluster.adata.obs.copy()
            sam_subcluster.adata_raw.obsm = sam_subcluster.adata.obsm.copy()

            #sam_subcluster.preprocess_data(**self.preprocess_args)
            
            self.out.clear_output()
            with self.out:
                sam_subcluster.run(**self.run_args)
            self.sams.append(sam_subcluster)
            self.selected.append(np.ones(sam_subcluster.adata.shape[0]).astype("bool"))
            self.selected_cells.append(np.array(list(sam_subcluster.adata.obs_names)))
            self.active_labels.append(np.zeros(sam_subcluster.adata.shape[0]))
            self.dd_opts.append(["Toggle cluster"])
            self.gene_expressions.append(np.zeros(sam_subcluster.adata.shape[0]))
            self.marker_genes.append(
                np.array(list(sam_subcluster.adata.var_names))[
                    np.argsort(-sam_subcluster.adata.var["weights"].values)
                ]
            )
            self.marker_genes_tt.append("Genes ranked by SAM weights.")
            self.ds.append(0)
            i = len(self.sams) - 1
            execute = True

        elif np.all(selected):
            sam.preprocess_data(**self.preprocess_args)

            self.out.clear_output()
            with self.out:
                sam.run(**self.run_args)
            self.marker_genes[i] = np.array(list(sam.adata.var_names))[
                np.argsort(-sam.adata.var["weights"].values)
            ]
            self.marker_genes_tt[i] = "Genes ranked by SAM weights."
            execute = True
            self.current_sam = sam

        if execute:
            self.log(('Subclustering SAM object in tab {}'.format(i),selected_cells))
            title = self.rb_dict["TITLE"].value
            if title == "" and i == 0:
                title = "Full dataset"
            elif title == "":
                title = "Subcluster " + str(i)
            self.create_plot(i, title)
            try:
                self.titles[i] = title
            except:
                self.titles.append(title)
                
            self.update_dropdowns(i)

    """ END RUN INIT"""

    """ BEGIN CS INIT"""

    def init_cs(self):
        initl = list(self.sams[0].adata.obsm.keys())
        initl = [""] + initl
        if self.default_proj in initl:
            init = self.default_proj
        else:
            init = initl[0]

        dpm = widgets.Dropdown(
            options=initl, value=init, description="", layout={"width": "60%"}
        )
        dp = widgets.Button(
            description="Display projection",
            tooltip="Display the 2D projection selected in the dropdown menu to the right.",
            layout={"width": "40%"},
        )
        dp.on_click(self.display_projection)

        initl = list(self.sams[0].adata.obs.keys())
        initl = [""] + initl
        dam = widgets.Dropdown(
            options=initl, value=initl[0], description="", layout={"width": "60%"}
        )
        da = widgets.Button(
            description="Display cell ann.",
            tooltip="Overlay the annotations selected in the dropdown menu to the right.",
            layout={"width": "40%"},
        )
        da.on_click(self.display_annotation)

        dv = widgets.Button(
            description="Display gene ann.",
            tooltip="Include the 'var' annotation for the corresponding gene in the title of the gene expression plots.",
            layout={"width": "40%"},
        )
        dv.on_click(self.display_var_annotation)

        initl = list(self.sams[0].adata.var.keys())
        initl = [""] + initl
        dvm = widgets.Dropdown(
            options=initl, value=initl[0], description="", layout={"width": "60%"}
        )

        irm = widgets.Button(
            description="Find markers (RF)",
            tooltip="Identify marker genes of selected cells using a random forest classifier.",
            layout={"width": "34%"},
        )
        irm.on_click(self.irm_genes)
        ism = widgets.Button(
            description="Find markers (SW)",
            tooltip="Identify marker genes of selected cells by finding genes with large SAM weights among the selected cells.",
            layout={"width": "33%"},
        )
        ism.on_click(self.ism_genes)

        sws = widgets.Button(
            description="SAM ranking",
            tooltip="Ranks genes by SAM weights.",
            layout={"width": "33%"},
        )
        sws.on_click(self.sam_weights)

        us = widgets.Button(
            description="Unselect all (x)",
            tooltip="Unselect all cells. Pressing 'x' on the keyboard while hovering over the scatter plot will do the same thing.",
            layout={"width": "34%"},
        )
        usa = widgets.Button(
            description="Select all (c)",
            tooltip="Select all cells. Pressing 'c' on the keyboard while hovering over the scatter plot will do the same thing.",
            layout={"width": "33%"},
        )
        res = widgets.Button(
            description="Reset view (v)",
            tooltip="Resets the current plot to the defautl view. Pressing 'v' on the keyboard while hovering over the scatter plot"
            " will do the same thing.",
            layout={"width": "33%"},
        )
        us.on_click(self.unselect_all)
        usa.on_click(self.select_all)
        res.on_click(self.reset_view)

        avg = widgets.Checkbox(
            indent=False, value=True, description="Avg expr", layout={"width": "20%"}
        )

        log = widgets.Checkbox(
            indent=False, value=False, description="Log cbar", layout={"width": "20%"}
        )

        show_colorbar = widgets.Checkbox(
            indent=False, value=False, description="Show cbar", layout={"width": "20%"}
        )
        lann = widgets.Button(
            description="Annotate",
            tooltip=(
                "Enter the key of the 'obs' annotation vector you wish to modify."
                " If the key does not exist, a new vector in 'obs' will be created. Enter the"
                " label you want to annotate the selected cells with in the right."
            ),
            disabled=True,
            layout={"width": "20%"},
        )
        anno_name = widgets.Text(
            value="", placeholder="Annotation column name", layout={"width": "40%"}
        )
        anno = widgets.Text(value="", placeholder="Annotation", layout={"width": "40%"})
        anno.on_submit(self.annotate_pop)

        lgsm = widgets.Button(
            description="Get similar genes",
            tooltip="Rank genes in order of decreasing spatially averaged correlation with respect to the input gene.",
            disabled=True,
            layout={"width": "30%"},
        )
        gsm = widgets.Text(
            value="",
            placeholder=self.sams[0].adata.var_names[0],
            layout={"width": "50%"},
        )
        gsm.on_submit(self.get_similar_genes)
        lshg = widgets.Button(
            description="Show expressions",
            tooltip="Overlay the input gene expressions over the scatter plot. Only a substring of the gene need be provided.",
            disabled=True,
            layout={"width": "30%"},
        )
        shg = widgets.Text(
            value="",
            placeholder=self.sams[0].adata.var_names[0],
            layout={"width": "50%"},
        )
        shg.on_submit(self.show_expression)

        lgenes = widgets.Button(
            description="Ranked genes",
            tooltip="Genes ranked by SAM weights.",
            disabled=True,
            layout={"width": "30%"},
        )
        rgenes = widgets.IntSlider(
            value=0,
            min=0,
            max=200,
            step=1,
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            description="",
            style={"description_width": "initial"},
            layout={"width": "70%"},
        )
        rgenes.observe(self.gene_update, "value")

        lthr = widgets.Button(
            description="Threshold expr",
            tooltip="Select genes with greater than the chosen expression threshold.",
            disabled=True,
            layout={"width": "30%"},
        )
        thr = widgets.FloatSlider(
            value=0,
            min=0,
            max=0,
            step=0.05,
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            description="",
            style={"description_width": "initial"},
            layout={"width": "70%"},
        )
        thr.observe(self.threshold_selection, "value")

        lsf = widgets.Button(
            description="Save",
            tooltip="Save the current SAM object. Filenames should end with .h5ad.",
            disabled=True,
            layout={"width": "30%"},
        )

        sf = widgets.Text(value="", layout={"width": "70%"})
        sf.on_submit(self.save_data)

        close = widgets.Button(
            description="Close current tab",
            tooltip="Closes the currently open subclustering tab.",
            disabled=False,
        )  # layout={'width':'30%'})
        close.on_click(self.close_tab)

        hotkeys = widgets.Button(
            description="ReadMe",
            tooltip="""While hovering over the scatter plot, the following keyboard inputs are available:
                - Left/Right arrows: Scroll through & display ranked genes
                - Shift: Random Forest classifier marker gene identification of selected cells
                - Enter: SAM weights-based approach to marker gene identification of selected cells
                - x: Unselect all cells
                - c: Select all cells
                - v: Reset view
                - a: Toggle expression spatial averaging ON/OFF

                Also note that clicking a cell will select/unselect all cells sharing its label.""",
            disabled=True,
        )  # layout={'width':'30%'})

        amslider = widgets.Button(
            description="Opacity",
            tooltip="Changes the opacities of unselected points in the current plot.",
            disabled=True,
            layout={"width": "30%"},
        )
        aslider = widgets.FloatSlider(
            value=0.05,
            min=0.0,
            max=1,
            step=0.01,
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="2f",
            description="",
            style={"description_width": "initial"},
            layout={"width": "70%"},
        )
        aslider.observe(self.change_alpha, "value")

        lmslider = widgets.Button(
            description="Marker size",
            tooltip="Changes the marker sizes in the current plot.",
            disabled=True,
            layout={"width": "30%"},
        )
        mslider = widgets.FloatSlider(
            value=5,
            min=0.1,
            max=20,
            step=0.1,
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="2f",
            description="",
            style={"description_width": "initial"},
            layout={"width": "70%"},
        )
        mslider.observe(self.change_msize, "value")

        acc = widgets.Dropdown(value="Toggle cluster", options=["Toggle cluster"])
        acc.observe(self.pick_cells_dd, "value")

        self.cs_dict = {}
        self.cs_dict["DP"] = dp
        self.cs_dict["DPM"] = dpm
        self.cs_dict["DA"] = da
        self.cs_dict["DAM"] = dam
        self.cs_dict["ACC"] = acc
        self.cs_dict["DV"] = dv
        self.cs_dict["DVM"] = dvm
        self.cs_dict["IRM"] = irm
        self.cs_dict["ISM"] = ism
        self.cs_dict["SWS"] = sws
        self.cs_dict["US"] = us
        self.cs_dict["USA"] = usa
        self.cs_dict["RES"] = res
        self.cs_dict["LANN"] = lann
        self.cs_dict["ANNO_NAME"] = anno_name
        self.cs_dict["ANNO"] = anno
        self.cs_dict["SHOW_COLORBAR"] = show_colorbar
        self.cs_dict["LGSM"] = lgsm
        self.cs_dict["GSM"] = gsm
        self.cs_dict["AVG"] = avg
        self.cs_dict["LSHG"] = lshg
        self.cs_dict["SHG"] = shg
        self.cs_dict["LOG"] = log
        self.cs_dict["LGENES"] = lgenes
        self.cs_dict["RGENES"] = rgenes
        self.cs_dict["LTHR"] = lthr
        self.cs_dict["THR"] = thr
        self.cs_dict["LSF"] = lsf
        self.cs_dict["SF"] = sf
        self.cs_dict["AMSLIDER"] = amslider
        self.cs_dict["ASLIDER"] = aslider
        self.cs_dict["LMSLIDER"] = lmslider
        self.cs_dict["MSLIDER"] = mslider
        self.cs_dict["CLOSE"] = close
        self.cs_dict["HOTKEYS"] = hotkeys

        return widgets.VBox(
            [
                widgets.HBox([dp, dpm]),
                widgets.HBox([da, dam, acc]),
                widgets.HBox([dv, dvm]),
                widgets.HBox([irm, ism, sws]),
                widgets.HBox([us, usa, res]),
                widgets.HBox([lann, anno_name, anno,]),
                widgets.HBox([lgsm, gsm,]),
                widgets.HBox([lshg, shg,]),
                widgets.HBox([lgenes, rgenes]),
                widgets.HBox([lthr, thr]),
                widgets.HBox([amslider, aslider]),
                widgets.HBox([show_colorbar, avg, log]),
                widgets.HBox([lmslider, mslider]),
                widgets.HBox([lsf, sf]),
                widgets.HBox([close, hotkeys]),
            ],
            layout={"height": "95%", "width": "100%"},
        )

    def reset_view(self, event):
        i = self.stab.selected_index
        self.log('Resetting view for tab {}'.format(i))        
        self.create_plot(i, self.stab.get_title(i))
        self.marker_genes[i] = np.array(list(self.sams[i].adata.var_names))[
            np.argsort(-self.sams[i].adata.var["weights"].values)
        ]
        self.marker_genes_tt[i] = "Genes ranked by SAM weights."
        self.cs_dict["LGENES"].set_trait("tooltip", self.marker_genes_tt[i])

    def save_data(self, path):
        path = path.value
        self.log('Saving data from tab {} to {}'.format(self.stab.selected_index,path))
        if path != "":
            if path.split(".")[-1] == "h5ad":
                s = self.sams[self.stab.selected_index]
                s.save_anndata(path)
            elif (
                path.split(".")[-1] == "png"
                or path.split(".")[-1] == "pdf"
                or path.split(".")[-1] == "eps"
                or path.split(".")[-1] == "jpg"
            ):
                if len(path.split("/")) > 1:
                    ut.create_folder("/".join(path.split("/")[:-1]))
                self.stab.children[self.stab.selected_index].write_image(path)

    def show_expression(self, text):
        if type(text) is not str:
            gene = text.value
        else:
            gene = text

        s = self.sams[self.stab.selected_index]
        if gene != "":

            try:
                if not (gene in s.adata.var_names):
                    genes = ut.search_string(
                        np.array(list(s.adata.var_names)), gene, case_sensitive=True
                    )[0]
                    if genes is not -1:
                        gene = genes[0]

                if self.cs_dict["AVG"].value:
                    if "X_knn_avg" not in s.adata.layers.keys():
                        s.dispersion_ranking_NN(save_avgs=True)

                    x = s.adata[:, gene].layers["X_knn_avg"]
                    if sp.issparse(x):
                        a = x.A.flatten()
                    else:
                        a = x.flatten()
                else:
                    x = s.adata[:, gene][s.adata.obs_names, :].X
                    if sp.issparse(x):
                        a = x.A.flatten()
                    else:
                        a = x.flatten()

                if self.GENE_KEY != "":
                    title = gene + "; " + str(s.adata.var[self.GENE_KEY].T[gene])
                else:
                    title = gene

                # self.select_all(None)
                self.update_colors_expr(a, title)
                self.log('Showing gene expression values for {} in tab {}'.format(title,self.stab.selected_index))
            except:
                if self.cs_dict["AVG"].value:
                    if not (gene in s.adata.var_names):
                        with self.out:
                            print(
                                "X_knn_avg does not exist. Either run"
                                " sam.dispersion_ranking_NN() or uncheck"
                                " `avg` in the control panel."
                            )
                elif not (gene in s.adata.var_names):
                    with self.out:
                        print(
                            "Gene not found. Check your spelling and " "capitalization."
                        )

    def get_similar_genes(self, txt):
        gene = txt.value
        if gene != "":
            s = self.sams[self.stab.selected_index]
            if "X_knn_avg" not in s.adata.layers.keys():
                s.dispersion_ranking_NN()

            genes = ut.search_string(
                np.array(list(s.adata.var_names)), gene, case_sensitive=True
            )[0]
            if genes is not -1:
                gene = genes[0]
            else:
                return
                # quit

            markers = ut.find_corr_genes(s, gene).flatten()
            _, i = np.unique(markers, return_index=True)
            markers = markers[np.sort(i)]
            self.marker_genes[self.stab.selected_index] = markers

            self.cs_dict["RGENES"].set_trait("value", 0)
            self.marker_genes_tt[self.stab.selected_index] = (
                "Ranked genes from most to least spatially correlated with "
                + gene
                + "."
            )
            self.cs_dict["LGENES"].set_trait(
                "tooltip", self.marker_genes_tt[self.stab.selected_index]
            )
            self.log('Getting genes with similar expression patterns to {}'.format(gene))
            self.show_expression(str(gene))
            

    def annotate_pop(self, text):
        text = text.value
        text_name = self.cs_dict["ANNO_NAME"].value
        selected = self.selected[self.stab.selected_index]
        selected_cells = self.selected_cells[self.stab.selected_index]
        if text != "" and text_name != "" and selected.sum() != selected.size:
            self.log(('Annotating cells with the {} label in the {} key.'.format(text,text_name),selected_cells))
            for it, s in enumerate(self.sams):
                try:
                    x1 = np.array(list(s.adata.obs_names))
    
                    if text_name in list(s.adata.obs.keys()):
                        a = s.get_labels(text_name).copy().astype("<U100")
                        a[np.in1d(x1, selected_cells)] = text
                        s.adata.obs[text_name] = pd.Categorical(a)
    
                    else:
                        a = np.zeros(s.adata.shape[0], dtype="<U100")
                        a[:] = ""
                        a[np.in1d(x1, selected_cells)] = text
                        s.adata.obs[text_name] = pd.Categorical(a)
                except:
                    pass
                
                self.update_dropdowns(it)

    def unselect_all(self, event):
        self.log('Unselecting all cells.')
        self.selected[self.stab.selected_index][:] = False
        self.selected_cells[self.stab.selected_index] = []
        self.stab.children[self.stab.selected_index].data[0].selectedpoints = np.array(
            []
        )
        self.stab.children[self.stab.selected_index].data[0].unselected = {
            "marker": {"opacity": self.cs_dict["ASLIDER"].value}
        }

    def select_all(self, event):
        self.log('Selecting all cells.')
        self.selected[self.stab.selected_index][:] = True
        self.selected_cells[self.stab.selected_index] = np.array(
            list(self.sams[self.stab.selected_index].adata.obs_names)
        )
        self.stab.children[self.stab.selected_index].data[0].selectedpoints = list(
            np.arange(self.selected[self.stab.selected_index].size)
        )
        self.stab.children[self.stab.selected_index].data[0].marker.opacity = 1

    def ism_genes(self, event):
        selected = self.selected[self.stab.selected_index]
        s = self.sams[self.stab.selected_index]
        if not np.all(selected) and selected.sum() > 0:
            self.log(('Calculating marker genes using SW method in tab {}'.format(self.stab.selected_index),self.selected_cells[self.stab.selected_index]))           

            if "X_knn_avg" not in s.adata.layers.keys():
                s.dispersion_ranking_NN(save_avgs=True)                
            l = s.adata.layers["X_knn_avg"]
            m = l.mean(0).A.flatten()
            ms = l[selected, :].mean(0).A.flatten()
            lsub = l[selected, :]
            lsub.data[:] = lsub.data ** 2
            ms2 = lsub.mean(0).A.flatten()
            v = ms2 - 2 * ms * m + m ** 2
            wmu = np.zeros(v.size)
            wmu[m > 0] = v[m > 0] / m[m > 0]

            self.marker_genes[self.stab.selected_index] = np.array(
                list(s.adata.var_names[np.argsort(-wmu)])
            )
            if self.cs_dict["RGENES"].value != 0:
                self.cs_dict["RGENES"].set_trait("value", 0)
            else:
                self.show_expression(
                    str(self.marker_genes[self.stab.selected_index][0])
                )
            self.marker_genes_tt[
                self.stab.selected_index
            ] = "Ranked genes according to spatial SAM weights among selected cells."
            self.cs_dict["LGENES"].set_trait(
                "tooltip", self.marker_genes_tt[self.stab.selected_index]
            )

    def irm_genes(self, event):
        selected = self.selected[self.stab.selected_index]
        s = self.sams[self.stab.selected_index]
        if not np.all(selected) and selected.sum() > 0:
            self.log(('Calculating marker genes using RF method in tab {}'.format(self.stab.selected_index),self.selected_cells[self.stab.selected_index]))            
            a = np.zeros(s.adata.shape[0])
            a[selected] = 1
            markers, _ = s.identify_marker_genes_rf(labels=a, clusters=1)
            self.marker_genes[self.stab.selected_index] = markers[1]
            if self.cs_dict["RGENES"].value != 0:
                self.cs_dict["RGENES"].set_trait("value", 0)
            else:
                self.show_expression(
                    str(self.marker_genes[self.stab.selected_index][0])
                )
            self.marker_genes_tt[
                self.stab.selected_index
            ] = "Ranked genes according to random forest classification."
            self.cs_dict["LGENES"].set_trait(
                "tooltip", self.marker_genes_tt[self.stab.selected_index]
            )

    def gene_update(self, val):
        val = val["new"]
        markers = self.marker_genes[self.stab.selected_index]

        if int(val) < markers.size:
            gene = markers[int(val)]
        else:
            gene = markers[-1]
        self.cs_dict["SHG"].set_trait("value", gene)
        self.show_expression(str(gene))

    def update_dropdowns(self, i):        
        s = self.sams[i]
        self.cs_dict["DPM"].options = [""] + list(s.adata.obsm.keys())
        self.cs_dict["DAM"].options = [""] + list(s.adata.obs.keys())
        self.cs_dict["DVM"].options = [""] + list(s.adata.var.keys())
        self.cs_dict["ACC"].options = self.dd_opts[i]

    def on_switch_tabs(self, event):
        self.log('Switching to tab {}'.format(self.stab.selected_index))
        self.update_dropdowns(self.stab.selected_index)
        self.cs_dict["LGENES"].set_trait(
            "tooltip", self.marker_genes_tt[self.stab.selected_index]
        )
        self.current_tab = self.stab.selected_index
        self.current_sam = self.sams[self.current_tab]
        for i in range(len(self.children)):
            self.tab.set_title(
                i, self.names[i] + " (" + str(self.stab.selected_index) + ")"
            )

        x = self.stab.children[self.stab.selected_index]
        f = x.layout.xaxis.range
        x.layout.xaxis.range=(-1,1)
        x.layout.xaxis.range=f        

        f = x.layout.yaxis.range
        x.layout.yaxis.range=(-1,1)
        x.layout.yaxis.range=f

            

    def display_var_annotation(self, event):
        self.GENE_KEY = self.cs_dict["DVM"].value
        self.log('Setting the var annotations to be from key {}'.format(self.GENE_KEY))

    def display_annotation(self, event):
        key = self.cs_dict["DAM"].value
        self.log('Displaying cell annotations from obs key {}'.format(key))
        if key != "":
            labels = np.array(list(self.sams[self.stab.selected_index].get_labels(key)))

            self.active_labels[self.stab.selected_index] = labels
            self.update_colors_anno(labels)

    def update_colors_expr(self, a, title):
        if self.cs_dict["LOG"].value:
            a = np.log2(a + 1)
        if self.cs_dict["SHOW_COLORBAR"].value:
            showscale = True
        else:
            showscale = False
        self.gene_expressions[self.stab.selected_index] = a

        f1 = self.stab.children[self.stab.selected_index]
        f1.update_traces(
            marker=dict(
                color=a,
                colorscale="spectral",
                reversescale=True,
                showscale=showscale,
                colorbar_ticks="outside",
                colorbar_tickmode="auto",
                colorbar_title="",
                opacity=1,
            )
        )

        f1.update_layout(hovermode="closest", title=title)
        self.stab.children[self.stab.selected_index].data[0].text = list(a)
        self.stab.children[self.stab.selected_index].data[0].hoverinfo = "text"

        slider = self.cs_dict["THR"]
        slider.set_trait("min", 0)
        slider.set_trait("max", a.max() + (a.max() - a.min()) / 100)
        slider.set_trait("step", (a.max() - a.min()) / 100)
        slider.set_trait("value", 0)

    def update_colors_anno(self, labels):
        nlabels = np.unique(labels).size
        title = self.cs_dict["DAM"].value.split("_clusters")[0]

        if issubclass(labels.dtype.type, np.number):
            if issubclass(labels.dtype.type, np.float):
                self.update_colors_expr(labels, title)
                return

        if nlabels == 1:
            x = "spectral"
        elif nlabels <= 2:
            x = cl.scales["3"]["qual"]["Paired"][:nlabels]
        elif nlabels <= 11:
            nlabels = str(nlabels)
            x = cl.scales[nlabels]["qual"]["Paired"]
        else:
            nlabels = "11"
            x = cl.scales[nlabels]["qual"]["Paired"]
            x = cl.interp(x, int(nlabels))

        lbls, inv = np.unique(labels, return_inverse=True)
        self.dd_opts[self.stab.selected_index] = ["Toggle cluster"] + list(lbls)
        self.cs_dict["ACC"].options = self.dd_opts[self.stab.selected_index]

        if issubclass(labels.dtype.type, np.character):
            tickvals = np.arange(lbls.size)
            ticktext = list(lbls)
        else:
            idx = np.round(np.linspace(0, len(lbls) - 1, 6)).astype(int)
            tickvals = list(idx)
            ticktext = lbls[tickvals]

        if self.cs_dict["SHOW_COLORBAR"].value:
            showscale = True
        else:
            showscale = False

        f1 = self.stab.children[self.stab.selected_index]
        f1.update_traces(
            marker=dict(
                color=inv,
                colorscale=x,
                colorbar_ticks="outside",
                colorbar_tickmode="array",
                colorbar_title="",
                colorbar_tickvals=tickvals,
                showscale=showscale,
                colorbar_ticktext=ticktext,
                opacity=1,
            )
        )
        self.stab.children[self.stab.selected_index].layout.title = title
        self.stab.children[self.stab.selected_index].data[0].text = list(lbls[inv])
        self.stab.children[self.stab.selected_index].data[0].hoverinfo = "text"
        self.stab.children[self.stab.selected_index].layout.hovermode = "closest"

        slider = self.cs_dict["THR"]
        slider.set_trait("min", slider.value)
        slider.set_trait("max", slider.value)

    def threshold_selection(self, val):
        val = val["new"]
        self.log('Setting gene expression selection threshold to be {}'.format(val))
        s = self.sams[self.stab.selected_index]

        self.selected[self.stab.selected_index][:] = False
        self.selected[self.stab.selected_index][
            self.gene_expressions[self.stab.selected_index] >= val
        ] = True
        self.selected_cells[self.stab.selected_index] = np.array(
            list(s.adata.obs_names)
        )[self.selected[self.stab.selected_index]]
        self.stab.children[self.stab.selected_index].data[0].selectedpoints = np.where(
            self.selected[self.stab.selected_index]
        )[0]

    def rewire_cluster_slider(self, event):
        x = self.rb_dict["CSLIDER"]
        l = self.rb_dict["L"]
        val = event["new"]
        if val == "Kmeans cluster":
            x.set_trait("min", 2)
            x.set_trait("max", 200)
            x.set_trait("value", 6)
            x.set_trait("step", 1)
            l.set_trait("value", "Kmeans 'k'")

        elif val == "Louvain cluster":
            x.set_trait("min", 0.1)
            x.set_trait("max", 10)
            x.set_trait("value", 1)
            x.set_trait("step", 0.1)
            l.set_trait("value", "Louvain 'res'")

        elif val == "Density cluster":
            x.set_trait("min", 0.1)
            x.set_trait("max", 2)
            x.set_trait("value", 0.5)
            x.set_trait("step", 0.01)
            l.set_trait("value", "Density 'eps'")

        elif val == "Hdbscan cluster":
            l.set_trait("value", "Hdbscan 'N/A'")

        elif val == "Leiden cluster":
            x.set_trait("min", 0.1)
            x.set_trait("max", 10)
            x.set_trait("value", 1)
            x.set_trait("step", 0.1)
            l.set_trait("value", "Leiden 'res'")

    def cluster_data(self, event):
        s = self.sams[self.stab.selected_index]
        val = self.rb_dict["CLM"].value
        eps = self.rb_dict["CSLIDER"].value
        self.log('Clustering data using the {} method with parameter {} in tab {}'.format(val,eps,self.stab.selected_index))
        if val == "Kmeans cluster":
            s.kmeans_clustering(int(eps))

        elif val == "Louvain cluster":
            s.louvain_clustering(res=eps)

        elif val == "Density cluster":
            s.density_clustering(eps=eps)

        elif val == "Hdbscan cluster":
            s.hdbknn_clustering(k=int(eps))

        elif val == "Leiden cluster":
            s.leiden_clustering(res=eps)

        self.cs_dict["DAM"].options = [""] + list(s.adata.obs.keys())

    def display_projection(self, event):
        s = self.sams[self.stab.selected_index]
        key = self.cs_dict["DPM"].value
        self.log('Displaying projection from obsm key {} in tab {}'.format(key,self.stab.selected_index))
        if key != "":
            X = s.adata.obsm[key][:, :2]
            self.stab.children[self.stab.selected_index].data[0]["x"] = X[:, 0]
            self.stab.children[self.stab.selected_index].data[0]["y"] = X[:, 1]

    def compute_projection(self, event):
        i = self.stab.selected_index
        s = self.sams[i]
        val = self.rb_dict["CPM"].value
        if val == "UMAP":
            s.run_umap()
        elif val == "t-SNE":
            s.run_tsne()
        elif val == "Diffusion map":
            s.run_diff_map()
        elif val == "Diffusion UMAP":
            s.run_diff_umap()
        self.log('Computing 2D projection using the {} method'.format(val))
        self.update_dropdowns(i)

    """ END CS INIT"""

    def select(self, trace, points, selector):
        if np.array(points.point_inds).size > 0:
            self.selected[self.stab.selected_index][np.array(points.point_inds)] = True
            self.selected_cells[self.stab.selected_index] = np.array(
                list(
                    self.sams[self.stab.selected_index].adata.obs_names[
                        self.selected[self.stab.selected_index]
                    ]
                )
            )
        trace.selectedpoints = list(
            np.where(self.selected[self.stab.selected_index])[0]
        )
        trace.unselected = {"marker": {"opacity": self.cs_dict["ASLIDER"].value}}

    def pick_cells_dd(self, txt):
        if txt["new"] != "Toggle cluster":
            al = str(txt["new"])
            sel = self.selected[self.stab.selected_index]
            als = self.active_labels[self.stab.selected_index].astype("str")
            ratio = sel[als == al].sum() / sel[als == al].size
            if ratio >= 0.5:
                sel[als == al] = False
            else:
                sel[als == al] = True

            self.selected_cells[self.stab.selected_index] = np.array(
                list(self.sams[self.stab.selected_index].adata.obs_names[sel])
            )

            self.stab.children[self.stab.selected_index].data[0].selectedpoints = list(
                np.where(sel)[0]
            )
            self.log('Toggling cluster {} in tab {}'.format(al,self.stab.selected_index))

            self.cs_dict["ACC"].value = "Toggle cluster"

    def pick_cells(self, trace, points, selector):
        tf = self.selected[self.stab.selected_index][points.point_inds[0]]
        als = self.active_labels[self.stab.selected_index]
        al = als[points.point_inds[0]]

        if tf:
            self.selected[self.stab.selected_index][als == al] = False
        else:
            self.selected[self.stab.selected_index][als == al] = True

        self.selected_cells[self.stab.selected_index] = np.array(
            list(
                self.sams[self.stab.selected_index].adata.obs_names[
                    self.selected[self.stab.selected_index]
                ]
            )
        )
        trace.selectedpoints = list(
            np.where(self.selected[self.stab.selected_index])[0]
        )

    def change_alpha(self, val):
        # m = self.stab.children[self.stab.selected_index].data[0].marker.copy()
        # m['opacity'] = val['new']
        self.stab.children[self.stab.selected_index].data[0].unselected = {
            "marker": {"opacity": val["new"]}
        }
        self.log('Changing alpha (opacity) to be {}'.format(val['new']))

    def change_msize(self, val):
        val = val["new"]
        markers = self.stab.children[self.stab.selected_index].data[0].marker
        markers.size = val
        self.log('Changing marker size to be {}'.format(val))

    def init_graph(self, trace):
        trace.mode = "markers"
        trace.marker.size = 5
        trace.on_selection(self.select)
        trace.on_click(self.pick_cells)

    
def invalidArgs(func, argdict):
    import inspect
    args = inspect.getfullargspec(func).args
    return set(argdict) - set(args)
    

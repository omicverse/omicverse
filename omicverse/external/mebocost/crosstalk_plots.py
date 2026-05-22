#!/usr/bin/env python

# ================================
# @auther: Rongbin Zheng
# @email: Rongbin.Zheng@childrens.harvard.edu
# @date: May 2024
# ================================

import os,sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
import traceback
import matplotlib
import collections
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx
## disable warnings
import warnings
warnings.filterwarnings("ignore")


plt.rcParams.update(plt.rcParamsDefault)
rc={"axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "figure.titleweight":"bold", #"font.size":14,
    "figure.figsize":(5.5,4.2), "font.weight":"regular", "legend.fontsize":10,
    'axes.labelpad':8, 'figure.dpi':300}
plt.rcParams.update(**rc)


def info(string):
    """
    print information
    """
    today = datetime.today().strftime("%B %d, %Y")
    now = datetime.now().strftime("%H:%M:%S")
    current_time = today + ' ' + now
    print("[{}]: {}".format(current_time, string))
    


def _histogram_(commu_mat,
                 commu_bg, 
                 title = '', 
                 pdf = False, 
                 show_plot = False,
                 bins = 100,
                 alpha = .6,
                 bg_color = 'grey',
                 obs_color = 'red',
                 figsize = (5.5, 4.2),
                 comm_score_col = 'Commu_Score',
                 return_fig = False):
    """
    plot histogram to show the distribution of communication in given pair of metabolite and sensor
    """
    ## visulize
    bg_values = []
    for x in commu_bg.iloc[:,2:commu_bg.shape[1]].values:
        bg_values.extend(x)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(bg_values, bins = bins, #density = True,
            label = 'Background', alpha = alpha, color = bg_color)
    ymin, ymax = ax.get_ylim()
    n = 0
    for i in commu_mat[comm_score_col].tolist():
        ax.vlines(x = i, ymin = ymin, ymax = ymax*0.1, colors = obs_color,
                 alpha = alpha, label = 'Observed' if n == 0 else None)
        n += 1
#     ax.hist(commu_mat[comm_score_col], bins = bins, density = True,
#             label = 'Observed C', alpha = alpha, color = obs_color)
    ax.set_ylabel('Count')
    ax.set_xlabel('Communication Score')
    ax.set_title(title)
    ax.legend()
    sns.despine()
    plt.tight_layout()
    pdf.savefig(fig) if pdf else None
    if show_plot:
        plt.show()
    plt.close()
    if return_fig:
        return(fig)

def _matrix_commu_(comm_res, pval_method, pval_cutoff, 
                   comm_score_col = 'Commu_Score',
                   comm_score_cutoff = None,
                   cutoff_prop = None 
                  ):
    cols = ['Metabolite_Name', 'Sensor', 'Sender', 'Receiver',
            comm_score_col, pval_method, 'metabolite_prop_in_sender', 'sensor_prop_in_receiver']
    df = comm_res[cols]
    ## especially for permutation test, p value could be 0, so set to 2.2e-16
    df.loc[df[pval_method]==0,pval_method] = 2.2e-16
    # print(communication_res_tidy.head())
    ## focus on significant ones, and communication score should be between 0 and 1
    if not comm_score_cutoff:
        comm_score_cutoff = 0
    if not cutoff_prop:
        cutoff_prop = 0
    df = df[(df[pval_method] < pval_cutoff) &
                        (df[comm_score_col] > comm_score_cutoff) &
                        (df['metabolite_prop_in_sender'] > cutoff_prop) &
                        (df['sensor_prop_in_receiver'] > cutoff_prop)
                       ]
    df['Signal_Pair'] = df['Metabolite_Name'] + '~' + df['Sensor']
    df['Cell_Pair'] = df['Sender'] + '→' + df['Receiver']
    df['-log10(pvalue)'] = -np.log10(df[pval_method])
    return df

def _commu_dotmap_(comm_res, 
            sender_focus = [],
            metabolite_focus = [],
            sensor_focus = [],
            receiver_focus = [],
            and_or = 'and',
            pval_method='permutation_test_fdr',
            pval_cutoff=0.05, 
            figsize = 'auto',
            cmap = 'Reds',
            cmap_vmin = None,
            cmap_vmax = None,
            cellpair_order = [],
            met_sensor_order = [],
            dot_size_norm = (10, 150),
            pdf = False, 
            show_plot = False,
            comm_score_col = 'Commu_Score',
            comm_score_range = None,
            comm_score_cutoff = None,
            cutoff_prop = None,
            swap_axis = False,
            return_fig = False):
    """
    plot all significant communications in heatmap.
    -----------------
    comm_res: a dict, keys are metabolite-sensor pairs,
            values are data frame
    pval_method: a string, default using permutation_test_fdr, but ttest_pval was also supported 
    pval_cutoff: a float, cutoff for filtering significant events, decault using 0.05
    bad_prediction: a list containing the name of metabolites, 
            these metabolites will be considered to be removed out
    prefix: a string, specefy where to save the figure, default the current directory
    """
    info('plot heatmap for significant result')
    ## clean
    plt.close()
    
    ## get matrix
    ptmp = _matrix_commu_(comm_res, pval_method, pval_cutoff, comm_score_col, comm_score_cutoff, cutoff_prop)
    
    ## stop plotting if nothing significant
    if ptmp.shape[0] == 0:
        info('Sorry, nothing to plot!')
        return

    ## focus 
    if and_or == 'and':
        if sender_focus:
            index = [x.split('→')[0] in sender_focus for x in ptmp['Cell_Pair'].tolist()]
            ptmp = ptmp.loc[index,:]
        if metabolite_focus:
            index = [x.split('~')[0] in metabolite_focus for x in ptmp['Signal_Pair'].tolist()]
            ptmp = ptmp.loc[index,:]    
        if sensor_focus:
            index = [x.split('~')[1] in sensor_focus for x in ptmp['Signal_Pair'].tolist()]
            ptmp = ptmp.loc[index,:] 
        if receiver_focus:
            index = [x.split('→')[1] in receiver_focus for x in ptmp['Cell_Pair'].tolist()]
            ptmp = ptmp.loc[index,:]
    else:
        if sender_focus or metabolite_focus or receiver_focus or sensor_focus:
            ptmp['sender'] = [x.split('→')[0] for x in ptmp['Cell_Pair'].tolist()]
            ptmp['receiver'] = [x.split('→')[1] for x in ptmp['Cell_Pair'].tolist()]
            ptmp['metabolite'] = [x.split('~')[0] for x in ptmp['Signal_Pair'].tolist()]
            ptmp['sensor'] = [x.split('~')[1] for x in ptmp['Signal_Pair'].tolist()]
#             index = [(x.split('→')[0] in sender_focus) or (x.split('→')[0] in metabolite_focus) or (x.split('~')[1] in sensor_focus) or (x.split('→')[1] in receiver_focus) for x in ptmp['Cell_Pair'].tolist()]
            ptmp = ptmp[ptmp['sender'].isin(sender_focus) |
                                    ptmp['receiver'].isin(receiver_focus) |
                                    ptmp['metabolite'].isin(metabolite_focus) |
                                    ptmp['sensor'].isin(sensor_focus)]
    ## if no data
    if ptmp.shape[0] == 0 or type(ptmp) != type(pd.DataFrame()):
        info('No enough data to show with your filtering!')
        return
    
    ## plot
    ncols = ptmp['Cell_Pair'].unique().shape[0] if not swap_axis else ptmp['Signal_Pair'].unique().shape[0]
    nrows = ptmp['Signal_Pair'].unique().shape[0] if not swap_axis else ptmp['Cell_Pair'].unique().shape[0]
    if figsize == 'auto' or not figsize:
        if swap_axis:
            figsize = (6+ncols*0.2, 5+nrows*0.2)
        else:
            figsize = (5+ncols*0.25, 6+nrows*0.2)

    ## format plotting data frame
    ptmp = ptmp[['Cell_Pair', 'Signal_Pair', '-log10(pvalue)', comm_score_col]]
    ptmp.index = ptmp['Cell_Pair'].astype('str')+'~'+ptmp['Signal_Pair'].astype('str')
    sr = ptmp['Signal_Pair'].unique().tolist()
    cg = ptmp['Cell_Pair'].unique().tolist()
    ptmp =  ptmp.T.to_dict()
    for i in cg:
        for j in sr:
            mccc = i+'~'+j
            if mccc not in ptmp:
                ptmp[mccc] = {'Cell_Pair':i, 'Signal_Pair':j, comm_score_col:np.nan, '-log10(pvalue)':np.nan}
    ptmp = pd.DataFrame.from_dict(ptmp).T  

    fig, ax = plt.subplots(figsize = figsize)
    plt.grid(zorder = 0, color = '#E8E8E8')
    ## default ordering
    ptmp = ptmp.sort_values(['Signal_Pair'])
    
    ## given ordering for cell pair
    ptmp['mCCC'] = ptmp['Cell_Pair'].astype('str')+'~'+ptmp['Signal_Pair'].astype('str')

    if cellpair_order and not met_sensor_order:
        ptmp = ptmp[ptmp['Cell_Pair'].isin(cellpair_order)]
        ptmp['Cell_Pair'] = ptmp['Cell_Pair'].astype('category').cat.set_categories(cellpair_order)
        ptmp = ptmp.sort_values('Cell_Pair')
    ## given ordering for met sensor
    elif met_sensor_order and not cellpair_order:
        ptmp = ptmp[ptmp['Signal_Pair'].isin(met_sensor_order)]
        ptmp['Signal_Pair'] = ptmp['Signal_Pair'].astype('category').cat.set_categories(met_sensor_order)
        ptmp = ptmp.sort_values('Signal_Pair')
    elif cellpair_order and met_sensor_order:
        ptmp = ptmp[ptmp['Signal_Pair'].isin(met_sensor_order) & ptmp['Cell_Pair'].isin(cellpair_order)]
        ptmp['Cell_Pair'] = ptmp['Cell_Pair'].astype('category').cat.set_categories(cellpair_order)
        ptmp['Signal_Pair'] = ptmp['Signal_Pair'].astype('category').cat.set_categories(met_sensor_order)
        ptmp = ptmp.sort_values(['Cell_Pair', 'Signal_Pair'])
    else:
        pass

    ## norm the dot size
    if type(comm_score_range) is type(()) and len(comm_score_range) == 2:
        dot_min, dot_max = comm_score_range
        ## legend for dot size
        dot_ann = sorted(list(set([dot_min, np.median(range(dot_min, dot_max)), dot_max])))
        if dot_min > dot_max:
            info('Warnings: the comm_score_range should be a typle (smaller value, larger value)')
    else:
        dot_min, dot_max = min(ptmp[comm_score_col].dropna()), max(ptmp[comm_score_col].dropna())
        ## legend for dot size
        dot_ann = sorted(list(set([np.min(ptmp[comm_score_col].dropna()), np.median(ptmp[comm_score_col].dropna()), np.max(ptmp[comm_score_col].dropna())])))
    ## define dot size norm function
    dot_size_norm_fun = lambda x: dot_size_norm[0]+((x-dot_min) / (dot_max - dot_min) * (dot_size_norm[1]-dot_size_norm[0])) if dot_max != dot_min else dot_size_norm[0]+((x-dot_min) / dot_max * (node_size_norm[1]-node_size_norm[0])) 
    
    ## dot color
    if not cmap_vmin:
        cmap_vmin = np.percentile(ptmp['-log10(pvalue)'].dropna(), 0)
    if not cmap_vmax:
        cmap_vmax = np.percentile(ptmp['-log10(pvalue)'].dropna(), 100)

    sp = ax.scatter(x = ptmp['Cell_Pair'] if not swap_axis else ptmp['Signal_Pair'],
               y = ptmp['Signal_Pair'] if not swap_axis else ptmp['Cell_Pair'],
               s = [dot_size_norm_fun(x) for x in ptmp[comm_score_col].tolist()],
               c = ptmp['-log10(pvalue)'].tolist(),
               vmax = cmap_vmax,
               vmin = cmap_vmin,
               cmap=cmap, zorder = 2)
    ax.tick_params(axis = 'x', which = 'major',
                   rotation = 90)
    ax.set_xlabel('')
    ax.set_ylabel('')

    for label in dot_ann:
        ax.scatter([], [], 
                   color = 'black',
                   facecolors='none',
                   s = dot_size_norm_fun(label),
                   label = round(label, 4))
    ax.legend(title = 'Communication Score', 
              loc='center left', 
              bbox_to_anchor=(1, 0.92),
              fontsize = 10,
              frameon = False)
    ## legend for color
    cbar=plt.colorbar(sp, ax = ax, shrink = .5)
    cbar.set_label(label = '-log10(p-value)', fontsize = 10)
    ax.set_xlim(-0.5, len(ptmp['Cell_Pair'].unique()) if not swap_axis else len(ptmp['Signal_Pair'].unique()))
    ax.set_ylim(-0.5, len(ptmp['Signal_Pair'].unique()) if not swap_axis else len(ptmp['Cell_Pair'].unique()))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tight_layout()
    pdf.savefig(fig) if pdf else None
    if show_plot:
        plt.show()
    plt.close()
    if return_fig:
        return(fig)

def CompScatterPlot(comm_res, 
                    cond1, 
                    cond2,
                    pval_method = 'permutation_test_fdr',
                    pval_threshold = 0.05,
                    Log2FC_threshold = 0,
                    figsize = (5.5, 4),
                    show_plot = False,
                    title = '',
                    return_fig = False,
                    save = False
                   ):
        """
        generate a scatter plot to show the difference
        """
        plot_df = comm_res.copy()
        plot_df['sig'] = (plot_df[pval_method] < pval_threshold)
        cols = []
        for lfc, sig in plot_df[['Log2FC', 'sig']].values.tolist():
            if lfc < -Log2FC_threshold and sig == True:
                cols.append('#1E90FF')
            elif lfc > Log2FC_threshold and sig == True:
                cols.append('#FF4040')
            else:
                cols.append('lightgrey')
        sizes = []
        for lfc in plot_df['Log2FC'].abs().tolist():
            if lfc > 1:
                sizes.append(50)
            elif lfc > 0.58:
                sizes.append(30)
            else:
                sizes.append(5)
        plot_df['cols'] = cols
        plot_df['sizes'] = sizes
        
        sig_df = plot_df.query('sig == True')
        nonsig_df = plot_df.query('sig != True')
        
        fig, ax = plt.subplots(figsize = figsize)
        ax.scatter(x = nonsig_df['Commu_Score_'+cond1], y = nonsig_df['Commu_Score_'+cond2],
                  s = nonsig_df['sizes'], edgecolor = 'none',
                  c = nonsig_df['cols'], alpha = .8)
        
        ax.scatter(x = sig_df['Commu_Score_'+cond1], y = sig_df['Commu_Score_'+cond2],
                  s = sig_df['sizes'], edgecolor = 'none',
                  c = sig_df['cols'], alpha = .6)
        xmin, xmax = plot_df['Commu_Score_'+cond1].min(), plot_df['Commu_Score_'+cond1].max()
        ymin, ymax = plot_df['Commu_Score_'+cond2].min(), plot_df['Commu_Score_'+cond2].max()
        plt.plot([xmin, xmax], [xmin, xmax], color='black', linestyle = 'dashed', linewidth = .5) 
        ax.set(xlabel = cond1, ylabel = cond2, title = title)
        colmap = {'Up': '#FF4040', 'NS': 'lightgrey', 'Down': '#1E90FF'}
        patches = [plt.Line2D([], [], marker = 'o', mec=None, label = key, color = colmap[key], lw=0) for key in colmap]
        legend1 = ax.legend(handles=patches, bbox_to_anchor=(0.95, 0.8), loc='center left', title = 'Signif.', frameon=False)
        
        ax.add_artist(legend1)
        
        size_dict = {1:50, 0.58:30, 0:5}
        patches = [plt.Line2D([], [], marker = 'o', mec=None, label = '> '+str(key),
                              color = 'grey', lw=0, markersize = size_dict[key]/10) for key in size_dict]
        legend2 = ax.legend(handles=patches, bbox_to_anchor=(0.95, 0.45), loc='center left', title = '|log2FC|', frameon=False)
        
        plt.tight_layout()
        plt.show() if show_plot else None
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
            Pdf.savefig(fig)
            Pdf.close()
        if return_fig:
            return(fig)
        return 

    
def _DiffFlowPlot_(comm_res, 
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
                line_cmap = 'spring_r',
                line_cmap_vmin = None,
                line_cmap_vmax = None,
                line_cmap_center = None,
                linewidth = 2,
                node_size_norm = (10, 150),
                node_value_range = None,
                pdf=None, 
                save_plot = True, 
                show_plot = False,
                text_outline = False,
                return_fig = False):
    """
    plot the flux plot to connect sender - metabolite - sensor - receiver
    ------------
    commu_res: a data frame containing all the communication events
    pval_method: a string, default using permutation_test_fdr, but ttest_pval was also supported 
    cutoff: the p value cutoff, the default will be 0.05
    node_label_size: the font size to label the node
    node_alpha: transpency for the node dots
    figsize: the figsize in tuple format
    node_cmap: color map used to draw node
    line_cmap: color map used to distinguish communication significance in line color
    line_cmap_vmin and line_vmax: set the colormap scale for line color, default both None
    node_size_norm: node size tells the connection numbers, can be set to normalize into a range
    linewidth_norm: linewidth to show the 
    prefix: a string, specefy where to save the figure, default the current directory
    """
    ## clean
    plt.close()
    
    info('plot flow plot to show the communications from Sender -> Metabolite -> Sensor -> Receiver')
    Sender_col = 'Sender'
    Receiver_col = 'Receiver'
    metabolite_col = 'Metabolite_Name'
    sensor_col = 'Sensor'

    ## get matrix
    ptmp = comm_res.loc[(comm_res[pval_method] < pval_cutoff) & (comm_res['Log2FC'].abs() > Log2FC_threshold),]
    
    ## if want to focus on showing some communications
    focus_plot = ptmp.copy()
    if and_or == 'and':
        if sender_focus:
            focus_plot = focus_plot[(focus_plot[Sender_col].isin(sender_focus))]
        if receiver_focus:
            focus_plot = focus_plot[(focus_plot[Receiver_col].isin(receiver_focus))]
        if metabolite_focus:
            focus_plot = focus_plot[(focus_plot[metabolite_col].isin(metabolite_focus))]
        if sensor_focus:
            focus_plot = focus_plot[(focus_plot[sensor_col].isin(sensor_focus))]
    else:
        if sender_focus or receiver_focus or metabolite_focus or sensor_focus:
            focus_plot = focus_plot[(focus_plot[Sender_col].isin(sender_focus)) |
                                     (focus_plot[Receiver_col].isin(receiver_focus)) |
                                     (focus_plot[metabolite_col].isin(metabolite_focus)) |
                                     (focus_plot[sensor_col].isin(sensor_focus))]
    
    if remove_unrelevant:
        ptmp = focus_plot.copy()
    if ptmp.shape[0] == 0:
        info('No mCCC identified based on your filtering')
        return

    ## get count to define dot size
    sender_count = dict(collections.Counter(ptmp['Sender']))
    receiver_count = dict(collections.Counter(ptmp['Receiver']))
    metabolite_count = dict(collections.Counter(ptmp['Metabolite_Name']))
    sensor_count = dict(collections.Counter(ptmp['Sensor']))
    ## all count to set dot size scale
    all_count = list(sender_count.values())+list(receiver_count.values())+list(metabolite_count.values())+list(sensor_count.values())
    # sender, receiver, uniquely
    senders = sorted(ptmp['Sender'].unique().tolist())
    receivers = sorted(ptmp['Receiver'].unique().tolist())
    metabolites = sorted(ptmp['Metabolite_Name'].unique().tolist())
    sensors = sorted(ptmp['Sensor'].unique().tolist())
    ## y maximum
    max_y = max([len(senders), len(receivers), len(metabolites), len(sensors)])
    senders_inti = int(abs(max_y-len(senders))/2) if abs(max_y-len(senders)) != 0 else 0
    receivers_inti = int(abs(max_y-len(receivers))/2) if abs(max_y-len(receivers)) != 0 else 0
    metabolites_inti = int(abs(max_y-len(metabolites))/2) if abs(max_y-len(metabolites)) != 0 else 0
    sensors_inti = int(abs(max_y-len(sensors))/2) if abs(max_y-len(sensors)) != 0 else 0

    # node size shows the connection number
    if type(node_value_range) is type(()) and len(node_value_range) == 2:
        dot_min, dot_max = node_value_range
        ## legend for dot size
        dot_ann = sorted(list(set([dot_min, int(np.median(range(dot_min, dot_max))), dot_max])))
        if dot_min > dot_max:
            info('Warnings: the node_value_range should be a typle (smaller value, larger value)')
    else:
        dot_min, dot_max = min(all_count), max(all_count)
            ## node size
        dot_ann = sorted(list(set([min(all_count), int(np.median(all_count)), max(all_count)])))

    node_size_norm_fun = lambda x: node_size_norm[0]+((x-dot_min) / (dot_max - dot_min) * (node_size_norm[1]-node_size_norm[0])) if dot_max != dot_min else node_size_norm[0]+((x-dot_min) / dot_max * (node_size_norm[1]-node_size_norm[0])) 

    # figsize
    if figsize == 'auto' or not figsize:
        figsize = (10, 5+max_y*0.07)
    # build up figure layout
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    subfigs = fig.add_gridspec(3, 4, height_ratios=[5, 1, 0.1],
                              hspace = 0, wspace = 0)
    main = fig.add_subplot(subfigs[0, :])
    bottom_left = fig.add_subplot(subfigs[1, 0])
    bottom_leftb = fig.add_subplot(subfigs[2, 0])
    mid = fig.add_subplot(subfigs[1:2, 1])
    bottom_mid = fig.add_subplot(subfigs[1:2, 2])
    bottom_right = fig.add_subplot(subfigs[1:2, 3])
    ## hiden axis
    main.axis('off')
    bottom_left.axis('off')
    mid.axis('off')
    bottom_leftb.axis('off')
    bottom_mid.axis('off')
    bottom_right.axis('off')
    
    ## node color
    if type(node_cmap) == type(list()) and len(node_cmap) == 4:
        sc, mc, ssc, rrc = node_cmap
    else:
        sc, mc, ssc, rrc = plt.cm.get_cmap(node_cmap)(1), plt.cm.get_cmap(node_cmap)(2), plt.cm.get_cmap(node_cmap)(3), plt.cm.get_cmap(node_cmap)(4)
    
    def _loc_(i, r, a):
        if a == 1:
            return([i])
        l = [i]
        for x in range(1, a):
            i += r
            l.append(i)
        return(l)

    ## draw dot for sender
    m_to_sender_ratio = int(.5*len(metabolites)/len(senders))
    m_to_sender_ratio = 1 if m_to_sender_ratio < 1 else m_to_sender_ratio
    
    sender_loc = _loc_(senders_inti, m_to_sender_ratio, len(senders))
    main.scatter([0]*len(senders),
                 sender_loc,
#                  range(senders_inti, senders_inti+len(senders)),
              s = [node_size_norm_fun(sender_count[x]) for x in senders], alpha = node_alpha,
              facecolor = sc, edgecolor = 'none')
    ## label sender
    for s in senders:
        txt = main.text(0, sender_loc[senders.index(s)], s, fontsize = node_label_size, ha = 'center', zorder = 2)
        if text_outline:
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])


    ## draw dot for metabolite
    main.scatter([1]*len(metabolites), range(metabolites_inti, metabolites_inti+len(metabolites)),
              s = [node_size_norm_fun(metabolite_count[x]) for x in metabolites], alpha = node_alpha,
              facecolor = mc, edgecolor = 'none')
    ## label metabolite
    for m in metabolites:
        txt = main.text(1, metabolites_inti+metabolites.index(m), m, fontsize = node_label_size, ha = 'center', zorder = 2)
        if text_outline:
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

    ## draw dot for sensor
    main.scatter([2]*len(sensors), range(sensors_inti, sensors_inti+len(sensors)),
              s = [node_size_norm_fun(sensor_count[x]) for x in sensors], alpha = node_alpha,
              facecolor = ssc, edgecolor = 'none')
    ## label sensor
    for t in sensors:
        txt = main.text(2, sensors_inti+sensors.index(t), t, fontsize = node_label_size, ha = 'center', zorder = 2)
        if text_outline:
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

    ## draw dot for receiver
    s_to_receiver_ratio = int(.5*len(sensors)/len(receivers))
    s_to_receiver_ratio = 1 if s_to_receiver_ratio < 1 else s_to_receiver_ratio
    receiver_loc = _loc_(receivers_inti, s_to_receiver_ratio, len(receivers))
    main.scatter([3]*len(receivers), 
                 receiver_loc,
#                  range(receivers_inti, receivers_inti+len(receivers)),
              s = [node_size_norm_fun(receiver_count[x]) for x in receivers], alpha = node_alpha,
              facecolor = ssc, edgecolor = 'none')
    ## label sender
    for r in receivers:
        txt = main.text(3, receiver_loc[receivers.index(r)], r, fontsize = node_label_size, ha = 'center', zorder = 2)
        if text_outline:
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

    my_cmap = plt.cm.get_cmap(line_cmap)

    ## line color scale
    if line_cmap_vmin is None:
        line_cmap_vmin = np.min(ptmp[line_color_col])

    if line_cmap_vmax is None:
        line_cmap_vmax = np.max(ptmp[line_color_col])
                
    if line_cmap_center is None:
        # print(line_cmap_center)
        line_cmap_center = np.mean([line_cmap_vmin, line_cmap_vmax])
    
    # print(line_cmap_vmin, line_cmap_center, line_cmap_vmax)
    
    if not line_cmap_vmin < line_cmap_center < line_cmap_vmax:
        info('color scale resorted')
        line_cmap_vmin, line_cmap_center, line_cmap_vmax = sorted([line_cmap_vmin, line_cmap_center, line_cmap_vmax])
        if not line_cmap_vmin < line_cmap_center < line_cmap_vmax:
            info('To default color scale')
            line_cmap_vmin, line_cmap_center, line_cmap_vmax = -1.5, 0, 1.5

    norm = matplotlib.colors.TwoSlopeNorm(vmin = line_cmap_vmin,
                                         vmax = line_cmap_vmax,
                                         vcenter = line_cmap_center)

    ## draw lines 
    n = 0
    for i,line in focus_plot.iterrows(): 
        m = metabolites_inti+metabolites.index(line['Metabolite_Name'])
        t = sensors_inti+sensors.index(line['Sensor'])
        S = sender_loc[senders.index(line['Sender'])]
        R = receiver_loc[receivers.index(line['Receiver'])]
        logFC = line[line_color_col]
        ## 
        linecolor = my_cmap(norm(logFC))
        ## between sender and metabolite
        main.plot([0, 1],[S, m], color = linecolor, zorder = 0, linewidth = linewidth, alpha = .2)
        ## between sender and metabolite
        main.plot([1, 2],[m, t], color = linecolor, zorder = 0, linewidth = linewidth, alpha = .2)
        ## between sender and metabolite
        main.plot([2, 3],[t, R], color = linecolor, zorder = 0, linewidth = linewidth, alpha = .2)
        n += 1

    ## label category
    main.text(0, -2, 'Sender', weight = 'bold', ha = 'center')
    main.text(1, -2, 'Metabolite', weight = 'bold', ha = 'center')
    main.text(2, -2, 'Sensor', weight = 'bold', ha = 'center')
    main.text(3, -2, 'Receiver', weight = 'bold', ha = 'center')

    ## add direction legend
    main.plot([0.4, 0.6], [-1.9, -1.9], color = 'lightgrey', linestyle = '-')
    main.plot([0.6], [-1.9], color = 'lightgrey', marker = '>')
    main.plot([1.4, 1.6], [-1.9, -1.9], color = 'lightgrey', linestyle = '-')
    main.plot([1.6], [-1.9], color = 'lightgrey', marker = '>')
    main.plot([2.4, 2.6], [-1.9, -1.9], color = 'lightgrey', linestyle = '-')
    main.plot([2.6], [-1.9], color = 'lightgrey', marker = '>')

    sm = matplotlib.cm.ScalarMappable(cmap=my_cmap, norm = norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax = bottom_left,
                        shrink = .9,
                        location = None)
    cbar.set_label(label=line_color_col, fontsize = 10)


    for label in dot_ann:
        bottom_mid.scatter([], [], 
                          color = 'black',
                          facecolors='none',
                          s = node_size_norm_fun(label),
                         label = int(label))
    bottom_mid.legend(title = '# of Connection', 
                     loc='center',
                     frameon = False)

    # plt.tight_layout()
    pdf.savefig(fig) if pdf else None
    if show_plot:
        plt.show()
    plt.close()
    if return_fig:
        return(fig)


def _FlowPlot_(comm_res, 
                pval_method='permutation_test_fdr',
                pval_cutoff=0.05,
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
                line_cmap = 'spring_r',
                line_cmap_vmin = None,
                line_cmap_vmax = None,
                linewidth_norm = (0.1, 1),
                linewidth_value_range = None,
                node_size_norm = (10, 150),
                node_value_range = None,
                pdf=None, 
                save_plot = True, 
                show_plot = False,
                comm_score_col = 'Commu_Score',
                comm_score_cutoff = None,
                cutoff_prop = None,
                text_outline = False,
                return_fig = False):
    """
    plot the flux plot to connect sender - metabolite - sensor - receiver
    ------------
    commu_res: a data frame containing all the communication events
    pval_method: a string, default using permutation_test_fdr, but ttest_pval was also supported 
    cutoff: the p value cutoff, the default will be 0.05
    node_label_size: the font size to label the node
    node_alpha: transpency for the node dots
    figsize: the figsize in tuple format
    node_cmap: color map used to draw node
    line_cmap: color map used to distinguish communication significance in line color
    line_cmap_vmin and line_vmax: set the colormap scale for line color, default both None
    node_size_norm: node size tells the connection numbers, can be set to normalize into a range
    linewidth_norm: linewidth to show the 
    prefix: a string, specefy where to save the figure, default the current directory
    """
    ## clean
    plt.close()
    
    info('plot flow plot to show the communications from Sender -> Metabolite -> Sensor -> Receiver')
    Sender_col = 'Sender'
    Receiver_col = 'Receiver'
    metabolite_col = 'Metabolite_Name'
    sensor_col = 'Sensor'
    line_color_col = '-log10(pvalue)'


    ## get matrix
    ptmp = _matrix_commu_(comm_res, pval_method, pval_cutoff, comm_score_col, comm_score_cutoff, cutoff_prop)
    
    ## if want to focus on showing some communications
    focus_plot = ptmp.copy()
    if and_or == 'and':
        if sender_focus:
            focus_plot = focus_plot[(focus_plot[Sender_col].isin(sender_focus))]
        if receiver_focus:
            focus_plot = focus_plot[(focus_plot[Receiver_col].isin(receiver_focus))]
        if metabolite_focus:
            focus_plot = focus_plot[(focus_plot[metabolite_col].isin(metabolite_focus))]
        if sensor_focus:
            focus_plot = focus_plot[(focus_plot[sensor_col].isin(sensor_focus))]
    else:
        if sender_focus or receiver_focus or metabolite_focus or sensor_focus:
            focus_plot = focus_plot[(focus_plot[Sender_col].isin(sender_focus)) |
                                     (focus_plot[Receiver_col].isin(receiver_focus)) |
                                     (focus_plot[metabolite_col].isin(metabolite_focus)) |
                                     (focus_plot[sensor_col].isin(sensor_focus))]
    
    if remove_unrelevant:
        ptmp = focus_plot.copy()
    if ptmp.shape[0] == 0:
        info('No mCCC identified based on your filtering')
        return

    ## get count to define dot size
    sender_count = dict(collections.Counter(ptmp['Sender']))
    receiver_count = dict(collections.Counter(ptmp['Receiver']))
    metabolite_count = dict(collections.Counter(ptmp['Metabolite_Name']))
    sensor_count = dict(collections.Counter(ptmp['Sensor']))
    ## all count to set dot size scale
    all_count = list(sender_count.values())+list(receiver_count.values())+list(metabolite_count.values())+list(sensor_count.values())
    # sender, receiver, uniquely
    senders = sorted(ptmp['Sender'].unique().tolist())
    receivers = sorted(ptmp['Receiver'].unique().tolist())
    metabolites = sorted(ptmp['Metabolite_Name'].unique().tolist())
    sensors = sorted(ptmp['Sensor'].unique().tolist())
    ## y maximum
    max_y = max([len(senders), len(receivers), len(metabolites), len(sensors)])
    senders_inti = int(abs(max_y-len(senders))/2) if abs(max_y-len(senders)) != 0 else 0
    receivers_inti = int(abs(max_y-len(receivers))/2) if abs(max_y-len(receivers)) != 0 else 0
    metabolites_inti = int(abs(max_y-len(metabolites))/2) if abs(max_y-len(metabolites)) != 0 else 0
    sensors_inti = int(abs(max_y-len(sensors))/2) if abs(max_y-len(sensors)) != 0 else 0

    # node size shows the connection number
    if type(node_value_range) is type(()) and len(node_value_range) == 2:
        dot_min, dot_max = node_value_range
        ## legend for dot size
        dot_ann = sorted(list(set([dot_min, int(np.median(range(dot_min, dot_max))), dot_max])))
        if dot_min > dot_max:
            info('Warnings: the node_value_range should be a typle (smaller value, larger value)')
    else:
        dot_min, dot_max = min(all_count), max(all_count)
            ## node size
        dot_ann = sorted(list(set([min(all_count), int(np.median(all_count)), max(all_count)])))

    node_size_norm_fun = lambda x: node_size_norm[0]+((x-dot_min) / (dot_max - dot_min) * (node_size_norm[1]-node_size_norm[0])) if dot_max != dot_min else node_size_norm[0]+((x-dot_min) / dot_max * (node_size_norm[1]-node_size_norm[0])) 

    ## line width range
    if type(linewidth_value_range) is type(()) and len(linewidth_value_range) == 2:
        line_min, line_max = linewidth_value_range
        ## legend for line width
        line_ann = sorted(list(set([line_min, np.median(range(line_min, line_max)), line_max])))
        if line_min > line_max:
            info('Warnings: the linewidth_value_range should be a typle (smaller value, larger value)')
    else:
        line_min, line_max = min(ptmp[comm_score_col]), max(ptmp[comm_score_col])
        ## line width
        line_ann = sorted(list(set([np.min(ptmp[comm_score_col]), np.median(ptmp[comm_score_col]), np.max(ptmp[comm_score_col])])))
    
    linewidth_norm_fun = lambda x: linewidth_norm[0]+((x-line_min) / (line_max - line_min) * (linewidth_norm[1]-linewidth_norm[0])) if line_max != line_min else linewidth_norm[0]+((x-line_min) / line_max * (linewidth_norm[1]-linewidth_norm[0]))

    # figsize
    if figsize == 'auto' or not figsize:
        figsize = (12, 5+max_y*0.07)
    # build up figure layout
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    subfigs = fig.add_gridspec(3, 4, height_ratios=[5, 1, 0.1],
                              hspace = 0, wspace = 0)
    main = fig.add_subplot(subfigs[0, :])
    bottom_left = fig.add_subplot(subfigs[1, 0])
    bottom_leftb = fig.add_subplot(subfigs[2, 0])
    mid = fig.add_subplot(subfigs[1:2, 1])
    bottom_mid = fig.add_subplot(subfigs[1:2, 2])
    bottom_right = fig.add_subplot(subfigs[1:2, 3])
    ## hiden axis
    main.axis('off')
    bottom_left.axis('off')
    mid.axis('off')
    bottom_leftb.axis('off')
    bottom_mid.axis('off')
    bottom_right.axis('off')
    
    ## node color
    if type(node_cmap) == type(list()) and len(node_cmap) == 4:
        sc, mc, ssc, rrc = node_cmap
    else:
        sc, mc, ssc, rrc = plt.cm.get_cmap(node_cmap)(1), plt.cm.get_cmap(node_cmap)(2), plt.cm.get_cmap(node_cmap)(3), plt.cm.get_cmap(node_cmap)(4)
    
    def _loc_(i, r, a):
        if a == 1:
            return([i])
        l = [i]
        for x in range(1, a):
            i += r
            l.append(i)
        return(l)

    ## draw dot for sender
    m_to_sender_ratio = int(.5*len(metabolites)/len(senders))
    m_to_sender_ratio = 1 if m_to_sender_ratio < 1 else m_to_sender_ratio
    
    sender_loc = _loc_(senders_inti, m_to_sender_ratio, len(senders))
    main.scatter([0]*len(senders),
                 sender_loc,
#                  range(senders_inti, senders_inti+len(senders)),
              s = [node_size_norm_fun(sender_count[x]) for x in senders], alpha = node_alpha,
              facecolor = sc, edgecolor = 'none')
    ## label sender
    for s in senders:
        txt = main.text(0, sender_loc[senders.index(s)], s, fontsize = node_label_size, ha = 'center', zorder = 2)
        if text_outline:
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])


    ## draw dot for metabolite
    main.scatter([1]*len(metabolites), range(metabolites_inti, metabolites_inti+len(metabolites)),
              s = [node_size_norm_fun(metabolite_count[x]) for x in metabolites], alpha = node_alpha,
              facecolor = mc, edgecolor = 'none')
    ## label metabolite
    for m in metabolites:
        txt = main.text(1, metabolites_inti+metabolites.index(m), m, fontsize = node_label_size, ha = 'center', zorder = 2)
        if text_outline:
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

    ## draw dot for sensor
    main.scatter([2]*len(sensors), range(sensors_inti, sensors_inti+len(sensors)),
              s = [node_size_norm_fun(sensor_count[x]) for x in sensors], alpha = node_alpha,
              facecolor = ssc, edgecolor = 'none')
    ## label sensor
    for t in sensors:
        txt = main.text(2, sensors_inti+sensors.index(t), t, fontsize = node_label_size, ha = 'center', zorder = 2)
        if text_outline:
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

    ## draw dot for receiver
    s_to_receiver_ratio = int(.5*len(sensors)/len(receivers))
    s_to_receiver_ratio = 1 if s_to_receiver_ratio < 1 else s_to_receiver_ratio
    receiver_loc = _loc_(receivers_inti, s_to_receiver_ratio, len(receivers))
    main.scatter([3]*len(receivers), 
                 receiver_loc,
#                  range(receivers_inti, receivers_inti+len(receivers)),
              s = [node_size_norm_fun(receiver_count[x]) for x in receivers], alpha = node_alpha,
              facecolor = ssc, edgecolor = 'none')
    ## label sender
    for r in receivers:
        txt = main.text(3, receiver_loc[receivers.index(r)], r, fontsize = node_label_size, ha = 'center', zorder = 2)
        if text_outline:
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

    my_cmap = plt.cm.get_cmap(line_cmap)

    ## line color scale
    if not line_cmap_vmin:
        line_cmap_vmin = np.min(ptmp[line_color_col])
    if not line_cmap_vmax:
        line_cmap_vmax = np.max(ptmp[line_color_col])
        
    if line_cmap_vmin>=line_cmap_vmax:
        line_cmap_vmin = line_cmap_vmax - abs(line_cmap_vmax-line_cmap_vmin)
        
    norm = matplotlib.colors.Normalize(vmin = line_cmap_vmin, vmax = line_cmap_vmax)
    # linecolor = np.array(list(map(norm, ptmp['-log10(pvalue)'])))

    ## draw lines 
    n = 0
    for i,line in focus_plot.iterrows(): 
        m = metabolites_inti+metabolites.index(line['Metabolite_Name'])
        t = sensors_inti+sensors.index(line['Sensor'])
        S = sender_loc[senders.index(line['Sender'])]
        R = receiver_loc[receivers.index(line['Receiver'])]
        comm_score = line[comm_score_col]
        significance = line['-log10(pvalue)']
        ## 
        linecolor = my_cmap(norm(significance))
        linewidth = linewidth_norm_fun(comm_score)
        ## between sender and metabolite
        main.plot([0, 1],[S, m], color = linecolor, zorder = 0, linewidth = linewidth, alpha = .2)
        ## between sender and metabolite
        main.plot([1, 2],[m, t], color = linecolor, zorder = 0, linewidth = linewidth, alpha = .2)
        ## between sender and metabolite
        main.plot([2, 3],[t, R], color = linecolor, zorder = 0, linewidth = linewidth, alpha = .2)
        n += 1

    ## label category
    main.text(0, -2, 'Sender', weight = 'bold', ha = 'center')
    main.text(1, -2, 'Metabolite', weight = 'bold', ha = 'center')
    main.text(2, -2, 'Sensor', weight = 'bold', ha = 'center')
    main.text(3, -2, 'Receiver', weight = 'bold', ha = 'center')

    ## add direction legend
    main.plot([0.4, 0.6], [-1.9, -1.9], color = 'lightgrey', linestyle = '-')
    main.plot([0.6], [-1.9], color = 'lightgrey', marker = '>')
    main.plot([1.4, 1.6], [-1.9, -1.9], color = 'lightgrey', linestyle = '-')
    main.plot([1.6], [-1.9], color = 'lightgrey', marker = '>')
    main.plot([2.4, 2.6], [-1.9, -1.9], color = 'lightgrey', linestyle = '-')
    main.plot([2.6], [-1.9], color = 'lightgrey', marker = '>')

    sm = matplotlib.cm.ScalarMappable(cmap=my_cmap, norm = norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax = bottom_left,
                        shrink = .9,
                        location = None)
    cbar.set_label(label='-log10(p-value)', fontsize = 10)


    for label in dot_ann:
        bottom_mid.scatter([], [], 
                          color = 'black',
                          facecolors='none',
                          s = node_size_norm_fun(label),
                         label = int(label))
    bottom_mid.legend(title = '# of Connection', 
                     loc='center',
                     frameon = False)

    for label in line_ann:
        bottom_right.plot([], [], 'g',
                          color = 'black',
                          linewidth = linewidth_norm_fun(label),
                         label = round(label, 4))
    bottom_right.legend(title = 'Communication Score', 
                     loc='center',
                     frameon = False)

    # plt.tight_layout()
    pdf.savefig(fig) if pdf else None
    if show_plot:
        plt.show()
    plt.close()
    if return_fig:
        return(fig)

def _make_cmat_(commu_res,
                        pval_method = 'permutation_test_fdr',
                        pval_cutoff = 0.05,
                        comm_score_col = 'Commu_Score',
                        comm_score_cutoff = None,
                        cutoff_prop = None
                     ):
    ptmp = _matrix_commu_(commu_res, pval_method, pval_cutoff, comm_score_col, comm_score_cutoff, cutoff_prop)
    if ptmp.shape[0] == 0:
        info('No communication events under pval_cutoff:{} and comm_score_cutoff:{}, try to tune them!'.format(pval_cutoff, comm_score_cutoff))
        return None
    ## visulize the communication frequency between cells
    count_df = pd.DataFrame(ptmp.groupby('Cell_Pair')['Cell_Pair'].count())
    count_df.index = count_df.index.tolist()
    count_df = pd.concat([count_df, 
                          pd.DataFrame(count_df.index.str.split('→').tolist(), index = count_df.index)], axis = 1)
    count_df.columns = ['Count', 'Sender', 'Receiver']
    ## communicate event summary
    cmat = {}
    for x in ptmp['Cell_Pair'].tolist():
        tmp = ptmp[ptmp['Cell_Pair']==x]['-log10(pvalue)'] # all the p values
        cmat[x] = [len(tmp), -sum(-tmp)] ## sum of log p = the product of p
    ## all cell group, this will enables manually ordering
    sg = list(set(ptmp['Sender'].tolist()))
    rg= list(set(ptmp['Receiver'].tolist()))
    for i in sg:
        for j in rg:
            cc = i+'→'+j
            if cc not in cmat:
                cmat[cc] = [np.nan, np.nan]
    cmat = pd.DataFrame.from_dict(cmat, orient = 'index')
    cmat = pd.concat([cmat, pd.DataFrame(cmat.index.str.split('→').tolist(), index = cmat.index)], axis = 1)
    cmat.columns = ['Count', '-log10(pvalue)', 'Sender', 'Receiver']
    return(cmat)

def _count_dot_plot_(commu_res, 
                        pval_method='permutation_test_fdr', 
                        pval_cutoff=0.05, 
                        cmap='RdBu_r', 
                        figsize = 'auto',
                        pdf = None,
                        dot_size_norm = (5, 100),
                        dot_value_range = None,
                        dot_color_vmin = None,
                        dot_color_vmax = None,
                        show_plot = False,
                        comm_score_col = 'Commu_Score',
                        comm_score_cutoff = None,
                        cutoff_prop = None,
                        dendrogram_cluster = True,
                        sender_order = [],
                        receiver_order = [],
                        return_fig = False):
    """
    plot dot plot where y axis is receivers, x axis is senders
    -----------
    commu_res: data frame 
        Cell_Pair  -log10(pvalue)
        Malignant_0->Malignant_0    1.378965
        Malignant_12->Malignant_0   1.378965
        Malignant_1->Malignant_0    1.359778
        CD8Tex_3->Malignant_0   1.347078

    cmap: colormap for dot color representing overall score of communication between cells
    """
    ## clean
    plt.close()
    
    info('plot dot plot to show communication in cell type level')
    cmat = _make_cmat_(commu_res = commu_res,
                                    pval_method = pval_method,
                                    pval_cutoff = pval_cutoff,
                                    comm_score_col = comm_score_col,
                                    comm_score_cutoff = comm_score_cutoff,
                                    cutoff_prop = cutoff_prop
                                  )
    if cmat is None:
        return
    ## plot setting
    if not dot_color_vmin:
        dot_color_vmin = np.percentile(cmat['-log10(pvalue)'].dropna(), 0)
    if not dot_color_vmax:
        dot_color_vmax = np.percentile(cmat['-log10(pvalue)'].dropna(), 100)

    ## plot dot plot with dendrogram
    if figsize == 'auto':
        snum = cmat['Sender'].unique().shape[0]
        rnum = cmat['Receiver'].unique().shape[0]
        figsize = (6.5+rnum*0.27, 5+snum*0.17)
    df = cmat.pivot_table(index = 'Sender', columns = 'Receiver', values = ['Count']).fillna(0)
    fig, axes = plt.subplots(nrows=2, ncols=4, 
                             gridspec_kw={'height_ratios': [.8, 5],
                                        'width_ratios':[5, 1.8, 0.2, 1.8],
                                        'hspace':0,
                                        'wspace':0},
                            figsize=figsize)
    leftupper = fig.add_subplot(axes[0, 0])
    leftlower = fig.add_subplot(axes[1, 0])
    midupper = fig.add_subplot(axes[0, 1])
    midlower = fig.add_subplot(axes[1, 1])
    rightupper = fig.add_subplot(axes[0, 2])
    rightlower = fig.add_subplot(axes[1, 2])
    sideupper = fig.add_subplot(axes[0, 3])
    sidelower = fig.add_subplot(axes[1, 3])

    ## off axis
    midupper.axis('off')
    rightupper.axis('off')
    rightlower.axis('off')
    leftupper.axis('off')
    midlower.axis('off')
    sideupper.axis('off')
    sidelower.axis('off')

    ## order x and y by user setting or dendrogram
    cmat['sender_receiver'] = cmat['Sender'].astype('str')+'~'+cmat['Receiver'].astype('str')
    if sender_order and not receiver_order:
        cmat = cmat[cmat['Sender'].isin(sender_order)]
        cmat['Sender'] = cmat['Sender'].astype('category').cat.set_categories(sender_order)
        cmat = cmat.sort_values(['Sender'])
        dendrogram_cluster = False

    elif receiver_order and not sender_order:
        cmat = cmat[cmat['Receiver'].isin(receiver_order)]
        cmat['Receiver'] = cmat['Receiver'].astype('category').cat.set_categories(receiver_order)
        cmat = cmat.sort_values(['Receiver'])
        dendrogram_cluster = False

    elif sender_order and receiver_order:
        cmat = cmat[cmat['Sender'].isin(sender_order) & cmat['Receiver'].isin(receiver_order)]
        cmat['Sender'] = cmat['Sender'].astype('category').cat.set_categories(sender_order)
        cmat['Receiver'] = cmat['Receiver'].astype('category').cat.set_categories(receiver_order)
        cmat = cmat.sort_values(['Sender', 'Receiver'])
        dendrogram_cluster = False

    else:
        pass

    if dendrogram_cluster:
        ## dendrogram for row and columns
        try:
            linkage_top = linkage(df)
            linkage_side = linkage(df.T)
            ## draw dendrogram
            plt.rcParams['lines.linewidth'] = 1
            den_top = dendrogram(linkage_top, 
                                 ax=leftupper, color_threshold=0, above_threshold_color='k')
            leftupper.tick_params(axis = 'x', rotation = 90)
            ## side dendrogram
            den_side = dendrogram(linkage_side,
                                  ax=midlower, color_threshold=0, above_threshold_color='k',
                      orientation='right')

            ## extract orders
            den_top_order = df.index[[int(x) for x in den_top['ivl']]]
            den_side_order = df.columns.get_level_values(1)[[int(x) for x in den_side['ivl']]]
            
            ## force orders
            cmat['Sender'] = cmat['Sender'].astype('category').cat.set_categories(den_top_order)
            cmat['Receiver'] = cmat['Receiver'].astype('category').cat.set_categories(den_side_order)
            cmat = cmat.sort_values(['Sender', 'Receiver'])
        except:
            info('dendrogram_cluster problem, so set to default order')

    ## main scatter
    ## norm the dot size
    if type(dot_value_range) is type(()) and len(dot_value_range) == 2:
        dot_min, dot_max = dot_value_range
        ## legend for dot size
        dot_values = sorted(list(set([dot_min, int(np.median(range(dot_min, dot_max))), dot_max])))
        if dot_min > dot_max:
            info('Warnings: the dot_value_range should be a typle (smaller value, larger value)')
    else:
        dot_min, dot_max = min(cmat['Count'].dropna()), max(cmat['Count'].dropna())
        ## legend for dot size
            ## legend for dot size
        dot_values = sorted(list(set([np.min(cmat['Count'].dropna()), int(np.median(cmat['Count'].dropna())), np.max(cmat['Count'].dropna())])))

    dot_size_norm_fun = lambda x: dot_size_norm[0]+((x-dot_min) / (dot_max - dot_min) * (dot_size_norm[1]-dot_size_norm[0])) if dot_max != dot_min else dot_size_norm[0]+((x-dot_min) / dot_max * (dot_size_norm[1]-dot_size_norm[0])) 
    dot_size = [dot_size_norm_fun(x) for x in cmat['Count'].tolist()]

    sp = leftlower.scatter(x = cmat['Sender'], 
                           y = cmat['Receiver'],
                           c = cmat['-log10(pvalue)'].tolist(), 
                           s = dot_size,
                           vmax = dot_color_vmax,
                           vmin = dot_color_vmin,
                           cmap = cmap)
    leftlower.tick_params(axis = 'x', rotation = 90)
    leftlower.set_xlabel('Sender')
    leftlower.set_ylabel('Receiver')

    ## legend for color
    cbar = plt.colorbar(sp, ax = sidelower,
                        location = 'top',
                       shrink = .7)
    cbar.set_label(label = 'Overall Score', 
                   fontsize = 10)

    for label in dot_values:
        sidelower.scatter([],[],
                color = 'black',
                facecolors='none',
                s = dot_size_norm_fun(label),
                label=int(label))
    sidelower.legend(title = '# of Communication Event', 
                     loc='upper center',
                     fontsize = 10,
                     frameon = False)
    nrow = len(cmat['Sender'].unique())
    ncol = len(cmat['Receiver'].unique())
    leftlower.set_xlim(-0.5, nrow-0.5)
    leftlower.set_ylim(-0.5, ncol-0.5)

    plt.tight_layout()
    pdf.savefig(fig) if pdf else None
    if show_plot:
        plt.show()
    plt.close()
    
    if return_fig:
        return(fig)

def _commu_network_plot_(commu_res,
                        sender_focus = [],
                        metabolite_focus = [],
                        sensor_focus = [],
                        receiver_focus = [],
                        remove_unrelevant = False,
                        and_or = 'and',
                        pval_method = 'permutation_test_fdr',
                        pval_cutoff = 0.05,
                        node_cmap = 'tab20',
                        figsize = 'auto',
                        line_cmap = 'RdBu_r',
                        line_color_vmin = None,
                        line_color_vmax = None,
                        line_width_col = 'Count',
                        linewidth_norm = (0.1, 1),
                        linewidth_value_range = None,
                        node_size_norm = (50, 300),
                        node_value_range = None,
                        adjust_text_pos_node = True,
                        node_text_hidden = False,
                        node_text_font = 10,
                        pdf = None,
                        save_plot = True,
                        show_plot = False,
                        comm_score_col = 'Commu_Score',
                        comm_score_cutoff = None,
                        cutoff_prop = None,
                        text_outline = False,
                        return_fig = False):
    """
    plot network figure to show the interactions between cells
    --------------
    cmat: a data frame with the format like this:
                                    Count   -log10(pvalue)  Sender  Receiver
        Malignant_0->Malignant_0    13  17.864293   Malignant_0 Malignant_0
        Malignant_12->Malignant_0   16  21.788151   Malignant_12    Malignant_0
        Malignant_1->Malignant_0    10  13.598459   Malignant_1 Malignant_0

    line_cmap: line color map, usually for the overall score, a sum of -log10(pvalue) for all metabolite-sensor communications to the connection
    node_cmap: node color map, usually for different type of cells
    figsize: a tuple to indicate the width and height for the figure, default is automatically estimate
    sender_col: column names for sender cells
    receiver_col: column names for sender cells
    """
    info('show communication in cells by network plot')

    ## clean
    plt.close()
    
    sender_col = 'Sender'
    receiver_col = 'Receiver'
    metabolite_col = 'Metabolite_Name'
    sensor_col = 'Sensor'
    line_color_col = '-log10(pvalue)'
    
    ## adjust by filter
    focus_commu = commu_res.copy()
    if and_or == 'and':
        if sender_focus:
            focus_commu = focus_commu[(focus_commu[sender_col].isin(sender_focus))]
        if receiver_focus:
            focus_commu = focus_commu[(focus_commu[receiver_col].isin(receiver_focus))]
        if metabolite_focus:
            focus_commu = focus_commu[(focus_commu[metabolite_col].isin(metabolite_focus))]
        if sensor_focus:
            focus_commu = focus_commu[(focus_commu[sensor_col].isin(sensor_focus))]
    else:
        if sender_focus or receiver_focus or metabolite_focus or sensor_focus:
            focus_commu = focus_commu[(focus_commu[sender_col].isin(sender_focus)) |
                                     (focus_commu[receiver_col].isin(receiver_focus)) |
                                     (focus_commu[metabolite_col].isin(metabolite_focus)) |
                                     (focus_commu[sensor_col].isin(sensor_focus))]
    if focus_commu.shape[0] == 0:
        info('No communication detected under the filtering')
                                     
    if remove_unrelevant is True:
        ## make cmat
        cmat = _make_cmat_(
                                commu_res = focus_commu,
                                pval_method = pval_method,
                                pval_cutoff = pval_cutoff,
                                comm_score_col = comm_score_col,
                                comm_score_cutoff = comm_score_cutoff,
                                cutoff_prop = cutoff_prop
                            )
    else:
        ## make cmat
        cmat = _make_cmat_(
                                commu_res = commu_res,
                                pval_method = pval_method,
                                pval_cutoff = pval_cutoff,
                                comm_score_col = comm_score_col,
                                comm_score_cutoff = comm_score_cutoff,
                                cutoff_prop = cutoff_prop
                            )
    
    if cmat is None:
        return
    
    cmat = cmat.dropna()

    if figsize == 'auto' or not figsize:
        node_num = len(set(cmat[sender_col].tolist()+cmat[receiver_col].tolist()))
        figsize = (3.2+node_num*0.2, 1.8+node_num * 0.1)

    fig = plt.figure(constrained_layout=True, figsize=figsize)

    subfigs = fig.add_gridspec(2, 3, width_ratios=[5.2, .7, .7])
    leftfig = fig.add_subplot(subfigs[:, 0])
    midfig = [fig.add_subplot(subfigs[0, 1]), fig.add_subplot(subfigs[1, 1])]
    rightfig = fig.add_subplot(subfigs[:, 2])

    ## get the node cells
    total_count = collections.Counter(cmat[receiver_col].tolist()+cmat[sender_col].tolist())
    total_count_value = list(total_count.values())
    G = nx.DiGraph(directed = True)

    ## add node
    for n in sorted(list(total_count.keys())):
        G.add_node(n)
        
    ## node size and color
    if type(node_value_range) is type(()) and len(node_value_range) == 2:
        dot_min, dot_max = node_value_range
        ## legend for dot size
        node_size_ann = sorted(list(set([dot_min, int(np.median(range(dot_min, dot_max))), dot_max])))
        if dot_min > dot_max:
            info('Warnings: the node_value_range should be a typle (smaller value, larger value)')
    else:
        dot_min, dot_max = min(total_count_value), max(total_count_value)
            ## node size
        node_size_ann = sorted(list(set([min(total_count_value), int(np.median(total_count_value)), max(total_count_value)])))

    node_size_norm_fun = lambda x: node_size_norm[0]+((x-dot_min) / (dot_max - dot_min) * (node_size_norm[1]-node_size_norm[0])) if dot_max != dot_min else node_size_norm[0]+((x-dot_min) / dot_max * (node_size_norm[1]-node_size_norm[0])) 
    
    node_size = [node_size_norm_fun(total_count.get(x, 0)) for x in G.nodes()] 

    ## node color
    if type(node_cmap) == type(dict()):
        node_col = np.array([node_cmap.get(x) for x in G.nodes()])
    else:
        cm = plt.cm.get_cmap(node_cmap)
        NUM_COLORS = len(G.nodes())
        node_col = np.array([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    ## line width
    if type(linewidth_value_range) is type(()) and len(linewidth_value_range) == 2:
        line_min, line_max = linewidth_value_range
        ## legend for line width
        line_ann = sorted(list(set([line_min, int(np.median(range(line_min, line_max))), line_max])))
        if line_min > line_max:
            info('Warnings: the linewidth_value_range should be a typle (smaller value, larger value)')
    else:
        line_min, line_max = min(cmat[line_width_col]), max(cmat[line_width_col])
        ## line width
        line_ann = sorted(list(set([np.min(cmat[line_width_col]), int(np.median(cmat[line_width_col])), np.max(cmat[line_width_col])])))

    linewidth_norm_fun = lambda x: linewidth_norm[0]+((x-line_min) / (line_max - line_min) * (linewidth_norm[1]-linewidth_norm[0])) if line_max != line_min else linewidth_norm[0]+((x-line_min) / line_max * (linewidth_norm[1]-linewidth_norm[0]))
    
    edge_color_norm = matplotlib.colors.Normalize(vmin = line_color_vmin if line_color_vmin else cmat[line_color_col].min(), vmax = line_color_vmax if line_color_vmax else cmat[line_color_col].max())
    rad = .3
    conn_style = f'arc3,rad={rad}'
    if focus_commu.shape[0] == 0:
        # Custom the nodes:
        pos = nx.circular_layout(G)
        nx.draw(G, pos, #with_labels=with_labels,
                font_size = node_text_font, node_color=node_col,
                node_size=node_size, connectionstyle=conn_style,
                cmap = node_cmap, ax = leftfig, alpha = .9)
    else:
        cmat_filter = _make_cmat_(
                                        commu_res = focus_commu,
                                        pval_method = pval_method,
                                        pval_cutoff = pval_cutoff,
                                        comm_score_col = comm_score_col,
                                        comm_score_cutoff = comm_score_cutoff
                                    )
        if cmat_filter is None:
            return
        
        cmat_filter = cmat_filter.dropna()

        for i,line in cmat_filter.iterrows():
            sender = line[sender_col]
            receiver = line[receiver_col]
            G.add_edge(sender, receiver)

        # Custom the nodes:
        pos = nx.circular_layout(G)

        if not line_color_vmin:
            line_color_vmin = np.percentile(cmat[line_color_col], 0)
        if not line_color_vmax:
            line_color_vmax = np.percentile(cmat[line_color_col], 100)

        edge_color = [plt.cm.get_cmap(line_cmap)(edge_color_norm(cmat[(cmat[sender_col]==x[0]) &
                                  (cmat[receiver_col]==x[1])].loc[:, line_color_col])) for x in G.edges]

        linewidth = [linewidth_norm_fun(cmat[(cmat[sender_col]==x[0]) & 
                                                   (cmat[receiver_col]==x[1])].loc[:, line_width_col]) for x in G.edges]
        nx.draw(G, pos, 
#                 with_labels=with_labels,
                arrows = True, 
                arrowstyle = '-|>',
                font_size = node_text_font,
                node_color=node_col,
                node_size=node_size, 
                edge_color=edge_color, 
                width=linewidth,
                connectionstyle=conn_style,
#                 cmap = node_cmap, 
                ax = leftfig, alpha = .8)
        
    if node_text_hidden is False or node_text_hidden is None:
        if adjust_text_pos_node:
            from adjustText import adjust_text  # deferred — optional dep
            text = []
            for x in pos.keys():
                txt = leftfig.text(pos[x][0], pos[x][1], x, size = node_text_font)
                if text_outline:
                    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
                text.append(txt)

            adjust_text(text, 
                        # arrowprops=dict(arrowstyle="-", color = 'k'),
                        ax = leftfig
                        )   
        else:
            for x in pos.keys():
                txt = leftfig.text(pos[x][0], pos[x][1], x, size = node_text_font, ha = 'center')
                if text_outline:
                    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

    ## legend for connected node size
    for label in node_size_ann:
        midfig[0].scatter([],[],
                color = 'black',
                facecolors='none',
                s = node_size_norm_fun(label),
                label=int(label))

    for label in line_ann:
        midfig[1].plot([],[],'g',
                color = 'black',
                linewidth = linewidth_norm_fun(label),
                label=int(label))
    midfig[0].axis('off')
    midfig[1].axis('off')
    midfig[0].legend(title = '# of\nConnected\nNodes', loc='center', frameon = False)
    midfig[1].legend(title = '# of\nCommunication\nEvents', loc='center', frameon = False)
    ## legend for communication confidence, line color
    sm = matplotlib.cm.ScalarMappable(cmap=plt.cm.get_cmap(line_cmap), norm = edge_color_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax = rightfig, shrink = .7, location = 'left')
    cbar.set_label(label='Overall Score',fontsize = 10)
    rightfig.axis('off')
    pdf.savefig(fig) if pdf else None
    if show_plot:
        plt.show()
    plt.close()
    if return_fig:
        return(fig)


def _violin_plot_(dat_mat,
                sensor_or_met,
                cell_focus = [],
                cell_order = [],
                cmap = 'Blues',
                vmin = None,
                vmax = None,
                figsize = 'auto',
                cbar_title = '',
                row_zscore = False,
                pdf = None,
                show_plot = False,
                return_fig = False):
    """
    violin plot to show the expression of sensor or metabolite level
    """
    ## clean
    plt.close()
    

    if row_zscore:
        cell_group = dat_mat['cell_group'].tolist()
        dat_mat = dat_mat[sensor_or_met].apply(lambda col: (col-np.mean(col))/np.std(col))
        dat_mat['cell_group'] = cell_group

    plot_df = dat_mat[sensor_or_met+['cell_group']]
    plot_df_group_mean = plot_df.groupby('cell_group').mean()
    
    if plot_df_group_mean.shape[0] == 0:
        info('No data found by the given sensor_or_met and cell_focus, please check!')
        return

    if vmax == None:
        vmax = np.percentile(plot_df_group_mean.max(), 95)
    if vmin == None:
        vmin = np.percentile(plot_df_group_mean.min(), 5)

    my_cmap = plt.cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax)

    if cell_focus:
        plot_df_group_mean = plot_df_group_mean.loc[plot_df_group_mean.index.isin(cell_focus),:]
        if plot_df_group_mean.shape[0] == 0:
            info('No data found by the given sensor_or_met and cell_focus, please check!')
            return
        
    if figsize == 'auto':
        figsize = (3.5+len(dat_mat['cell_group'].unique())*0.25, 3.5 + len(sensor_or_met)*0.35)

    fig = plt.figure(figsize=figsize)
    kw = dict(
            wspace=0,
            hspace=0,
            width_ratios=[1, .1]
        )

    subfigs = matplotlib.gridspec.GridSpec(len(sensor_or_met), 2, **kw)

    colbar = fig.add_subplot(subfigs[:,1])

    uniq_celltype = plot_df_group_mean.index.tolist()#sorted(plot_df['cell_group'].unique().tolist())
    if cell_order:
        cell_order = [x for x in cell_order if x in uniq_celltype]
        uniq_celltype = cell_order

    for i in range(len(sensor_or_met)):
        row_ax = fig.add_subplot(subfigs[i, 0])
        row_norm = list(map(norm, plot_df_group_mean.reindex(uniq_celltype)[sensor_or_met[i]]))
        row_color = [my_cmap(x) for x in row_norm]
        
        df = plot_df.groupby('cell_group')[sensor_or_met[i]].apply(lambda x: list(x))[uniq_celltype]
        vp = row_ax.violinplot(df, 
                  range(len(uniq_celltype)), widths = 1,
                 showmeans=False, showmedians=False, showextrema=False)
        
        for pc in range(len(vp['bodies'])):
            vp['bodies'][pc].set_facecolor(row_color[pc])
            vp['bodies'][pc].set_edgecolor('grey')
            vp['bodies'][pc].set_linewidth(.5)
            vp['bodies'][pc].set_alpha(1)

        row_ax.set_xlabel('')
        row_ax.set_ylabel(sensor_or_met[i], rotation=0,
                         fontsize = 8, 
#                           labelpad = 5, 
                          ha = 'right')
        row_ax.tick_params(axis = 'y', labelsize = 6)
        if i != len(sensor_or_met)-1:
            row_ax.set_xticks([])

    row_ax.tick_params(axis = 'x', rotation = 90, labelsize = 10)
    plt.xticks(range(len(uniq_celltype)), uniq_celltype)
    sm = matplotlib.cm.ScalarMappable(cmap=my_cmap,norm=norm)
    cbar=plt.colorbar(sm, ax = colbar,
                      location = 'right',
                      label='',
                      shrink=1)
    cbar.set_label(label = cbar_title,
                   fontsize = 8)
    colbar.axis('off')
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    pdf.savefig(fig) if pdf else None
    if show_plot:
        plt.show()
    plt.close()
    if return_fig:
        return(fig)
    
def _count_stacked_bar_(tdf, figsize, title='', legend_loc='upper right',
                        ylabel = 'Number of Metabolite-Sensor Communication',
                       colorcmap = 'tab20', pdf = None, show_plot = False,
                       return_fig = False):
    xorder = tdf.sum().sort_values(ascending = False).index.tolist()
    tdf = tdf.reindex(columns = xorder)
    groups = tdf.index.tolist()
    ## color for groups
    if type(colorcmap) == type(dict()):
        colormap = colorcmap.copy()
    else:
        colormap = {groups[i]: plt.cm.get_cmap(colorcmap)(i) for i in range(len(groups))}

    fig, ax = plt.subplots(figsize = figsize)

    xtick = list(np.arange(0, len(xorder) * 2, 2))
    for i in range(len(groups)):
        c = groups[i]
        if i == 0:
            ax.bar(xtick, tdf.loc[c].tolist(), label = c, color = colormap.get(c))
        else:
            bottom = tdf.iloc[:i,].sum()
            ax.bar(xtick, tdf.loc[c], bottom = bottom, 
                   label = c, color = colormap.get(c))

    for i in range(len(xtick)):
        ax.text(x=xtick[i], y=tdf.iloc[:,i].sum(),
                s=int(tdf.iloc[:,i].sum()),
                ha = 'center',
                va = 'bottom',
                fontsize = 8
               )

    plt.xticks(ticks = xtick,
                labels = tdf.columns.tolist())

    ax.set_title(title,
            pad = 10, fontweight = 'bold')

    ax.tick_params(axis = 'x', which = 'major',
                   rotation = 90, size = 12)

    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.set_xlim(-1, len(xtick)*2+1)

    sns.despine(trim = True)
    ax.legend(ncol = 2 if len(groups) > 10 else 1, loc = legend_loc)
#     ax.legend()

    plt.tight_layout()
    pdf.savefig(fig) if pdf else None
    plt.show() if show_plot else None
    plt.close()
    if return_fig:
        return(fig)

def _count_bar_(tdf, figsize, title='', color='pink',
                ylabel = 'Number of mCCC event', 
                pdf=None, show_plot = False, return_fig = False):
    fig, ax = plt.subplots(figsize = figsize)
    xtick = list(np.arange(0, len(tdf) * 2, 2))
    ax.bar(xtick, tdf.tolist(), color = color)

    for i in range(len(xtick)):
        ax.text(x=xtick[i], y=tdf.tolist()[i],
                s=int(tdf.tolist()[i]),
                ha = 'center',
                va = 'bottom',
                fontsize = 8
               )

    plt.xticks(ticks = xtick,
                labels = tdf.index.tolist())

    ax.set_title(title,
            pad = 10, fontweight = 'bold')

    ax.tick_params(axis = 'x', which = 'major',
                   rotation = 90, size = 12)

    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.set_xlim(-1, len(xtick)*2+1)

    sns.despine(trim = True)
    plt.tight_layout()
    pdf.savefig(fig) if pdf else None
    plt.show() if show_plot else None
    plt.close()
    if return_fig:
        return(fig)
    
def _eventnum_bar_(commu_res,
                    sender_focus = [],
                    metabolite_focus = [],
                    sensor_focus = [],
                    receiver_focus = [],
                    xorder = [],
                    and_or = 'and',
                    pval_method = 'permutation_test_fdr',
                    pval_cutoff = 0.05,
                    comm_score_col = 'Commu_Score',
                    comm_score_cutoff = None,
                    cutoff_prop = None,
                    figsize = 'auto',
                    pdf = None,
                    show_num = True,
                    show_plot = True,
                    include = ['sender-receiver', 'sensor', 'metabolite'],
                    group_by_cell = False,
                    colorcmap = 'tab20',
                    return_fig = False
                  ):
    """
    the bar plot to show the total event number
    """
    ##clean
    plt.close()  
    ## adjust by filter
    sender_col = 'Sender'
    receiver_col = 'Receiver'
    metabolite_col = 'Metabolite_Name'
    sensor_col = 'Sensor'
    line_color_col = '-log10(pvalue)'
    
    focus_commu = commu_res.copy()
    if and_or == 'and':
        if sender_focus:
            focus_commu = focus_commu[(focus_commu[sender_col].isin(sender_focus))]
        if receiver_focus:
            focus_commu = focus_commu[(focus_commu[receiver_col].isin(receiver_focus))]
        if metabolite_focus:
            focus_commu = focus_commu[(focus_commu[metabolite_col].isin(metabolite_focus))]
        if sensor_focus:
            focus_commu = focus_commu[(focus_commu[sensor_col].isin(sensor_focus))]
    else:
        if sender_focus or receiver_focus or metabolite_focus or sensor_focus:
            focus_commu = focus_commu[(focus_commu[sender_col].isin(sender_focus)) |
                                     (focus_commu[receiver_col].isin(receiver_focus)) |
                                     (focus_commu[metabolite_col].isin(metabolite_focus)) |
                                     (focus_commu[sensor_col].isin(sensor_focus))]
    if focus_commu.shape[0] == 0:
        info('No communication detected under the filtering')
                                    
    ##
    ptmp = _matrix_commu_(focus_commu, pval_method, pval_cutoff, comm_score_col, comm_score_cutoff, cutoff_prop)

    if 'sender-receiver' in include:
        ## sender and receiver count
        sender_n = ptmp.groupby('Sender')['Sender'].count()
        receiver_n = ptmp.groupby('Receiver')['Receiver'].count()

        ## plot sender and receiver number
        df = pd.concat([sender_n, receiver_n], axis = 1).fillna(0)
        df = df.sort_values('Sender', ascending = False)

        if xorder:
            xorder = [x for x in xorder if x in df.index.tolist()]
            df = df.loc[xorder,:]

        nrow, ncol = df.shape

        if figsize == 'auto':
            Figsize = (3+nrow*0.25, 2.5+nrow*0.15)
        else:
            Figsize = figsize

        fig, ax = plt.subplots(figsize = Figsize)

        colmap = {'Sender':'purple', 'Receiver':'darkorange'}

        xtick = list(np.arange(0, nrow * 3, 3))
        xtick1 = list(np.arange(0, nrow * 3, 3) - 0.5)
        xtick2 = list(np.arange(0, nrow * 3, 3) + 0.5)

        ax.bar(xtick1, df['Sender'], 
                    color = colmap['Sender'],
                   label = 'Sender')
        ax.bar(xtick2, df['Receiver'], 
                    color = colmap['Receiver'],
                   label = 'Receiver')
        if show_num is True:
            for i in range(len(xtick1)):
                ax.text(x=xtick1[i], y=df.iloc[i]['Sender'],
                        s=int(df.iloc[i]['Sender']),
                        rotation = 90, 
                        ha = 'center',
                        va = 'bottom',
                        fontsize = 8
                       )
            for i in range(len(xtick2)):
                ax.text(x=xtick2[i], y=df.iloc[i]['Receiver'],
                        s=int(df.iloc[i]['Receiver']),
                        rotation = 90, 
                        ha = 'center',
                        va = 'bottom',
                        fontsize = 8
                       )
        plt.xticks(ticks = xtick,
                         labels = df.index.tolist())

        ax.tick_params(axis = 'x', which = 'major',
                       rotation = 90, size = 10)
        ax.tick_params(axis = 'y', which = 'major',
                       rotation = 0, size = 10)
        ax.set_xlabel('')
        ax.set_ylabel('Number of mCCC event', size = 12)
        # ax.set_title('Event number of cell group as sender or receiver',
        #             pad = 10, fontweight = 'bold')
        sns.despine(trim = True)
        ax.legend(#bbox_to_anchor=(1, .9),
                     frameon = False,
                  fontsize = 10)
        ax.set_xlim(-1.5, nrow*3+1)
        plt.tight_layout()
        pdf.savefig(fig) if pdf else None

        if show_plot:
            plt.show()
        plt.close()
        
        if return_fig:
            return(fig)
        
    if 'metabolite-sensor' in include:
        tmp_n = pd.Series(dict(collections.Counter(ptmp['Metabolite_Name'] + ' ~ ' + ptmp['Sensor'])))
        tmp_n = tmp_n.sort_values(ascending = False)
        if xorder:
            xorder = [x for x in xorder if x in tmp_n.index.tolist()]
            tmp_n = tmp_n[xorder]

        if figsize == 'auto':
            Figsize = (4+len(tmp_n)*0.17, 7)
        else:
            Figsize = figsize
        color = 'pink'
        fig = _count_bar_(tdf=tmp_n, figsize=Figsize,
                    title='Number of mCCC event', 
                    ylabel = 'Number of Sender-Receiver Pairs',
                    color=color, pdf=pdf, show_plot = show_plot)        
        if return_fig:
            return(fig)
    
    if 'metabolite' in include:
        if group_by_cell:
            met_n = ptmp.groupby(['Sender', 'Metabolite_Name']).apply(lambda x: x.shape[0]).reset_index()
            met_n.columns = ['Sender', 'Metabolite_Name', 'Count']

            tdf = met_n.pivot_table(index = 'Sender', columns = 'Metabolite_Name').fillna(0).astype('int')
            tdf.columns = tdf.columns.get_level_values(1).tolist()

            if xorder:
                xorder = [x for x in xorder if x in tdf.columns.tolist()]
                tdf = tdf[xorder]

            nrow, ncol = tdf.shape

            if figsize == 'auto':
                Figsize = (4+ncol*0.25, 7)
            else:
                Figsize = figsize
            fig = _count_stacked_bar_(tdf=tdf, figsize=Figsize, 
                                    title='Communication Number of Metabolite in Sender', 
                                    legend_loc='upper right',
                                    ylabel = 'Number of mCCC event',
                                    colorcmap = colorcmap, pdf = pdf, 
                                    show_plot = show_plot)
        else:
            met_n = ptmp.groupby('Metabolite_Name')['Metabolite_Name'].count().sort_values(ascending = False)

            if xorder:
                xorder = [x for x in xorder if x in met_n.index.tolist()]
                met_n = met_n[xorder]

            if figsize == 'auto':
                Figsize = (4+ncol*0.25, 6)
            else:
                Figsize = figsize
            fig = _count_bar_(tdf=met_n, figsize=Figsize, 
                            title='Communication Number of Metabolite in Sender',
                            ylabel = 'Number of mCCC event',
                            color=color, pdf=None, show_plot = False)
        if return_fig:
            return(fig)
        
    if 'sensor' in include:
        if group_by_cell:
            sensor_n = ptmp.groupby(['Receiver', 'Sensor']).apply(lambda x: x.shape[0]).reset_index()
            sensor_n.columns = ['Receiver', 'Sensor', 'Count']

            tdf = sensor_n.pivot_table(index = 'Receiver', columns = 'Sensor').fillna(0).astype('int')
            tdf.columns = tdf.columns.get_level_values(1).tolist()

            if xorder:
                xorder = [x for x in xorder if x in tdf.columns.tolist()]
                tdf = tdf[xorder]

            nrow, ncol = tdf.shape
            if figsize == 'auto':
                Figsize = (4+ncol*0.25, 7)
            else:
                Figsize = figsize
            fig = _count_stacked_bar_(tdf=tdf, figsize=Figsize, 
                                    title='Communication Number of Sensor in Receiver', 
                                    legend_loc='upper right',
                                    ylabel = 'Number of mCCC event',
                                    colorcmap = colorcmap, pdf = pdf, 
                                    show_plot = show_plot)
        else:
            sensor_n = ptmp.groupby(['Receiver', 'Sensor']).apply(lambda x: x.shape[0]).reset_index()

            if xorder:
                xorder = [x for x in xorder if x in sensor_n.index.tolist()]
                sensor_n = sensor_n[xorder]

            if figsize == 'auto':
                Figsize = (4+ncol*0.25, 6)
            else:
                Figsize = figsize
            fig = _count_bar_(tdf=sensor_n, figsize=Figsize, 
                            title='Communication Number of Sensor in Receiver',
                            ylabel = 'Number of mCCC event',
                            color=color, pdf=None, show_plot = False)
    
        if return_fig:
            return(fig)


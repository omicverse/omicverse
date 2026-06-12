"""Self-contained enrichment plots — the running-ES GSEA curve, dot-plot and
bar-plot used by :mod:`omicverse.bulk` enrichment.

These were previously imported from the vendored ``gseapy`` package
(``omicverse.external.gseapy.plot``). gseapy has been removed in favour of
omicverse's own NumPy GSEA/ORA backends (:mod:`omicverse.bulk._gsea_numpy`,
:mod:`omicverse.bulk._ora`); this module keeps the matplotlib drawing code so
``pyGSEA.plot_gsea`` and PyWGCNA's module dot-plot keep working unchanged.
"""
# -*- coding: utf-8 -*-
import sys

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms


def unique(seq):
    """Remove duplicates from a list while preserving order."""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def isfloat(x):
    try:
        float(x)
    except Exception:
        return False
    return True


class MidpointNormalize(Normalize):
    """Normalize a colormap so its midpoint sits at a chosen value (0 for ES)."""

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class GSEAPlot(object):
    """Reproduce the classic GSEA running-enrichment-score figure.

    Drop-in for the former ``gseapy.plot.GSEAPlot``; accepts the same
    ``rank_metric / term / hit_indices / nes / pval / fdr / RES`` keywords that
    :attr:`PrerankResult.results` provides.
    """

    def __init__(self, rank_metric, term, hit_indices, nes, pval, fdr, RES,
                 pheno_pos='', pheno_neg='', figsize=(6, 5.5),
                 cmap='seismic', ofname=None, **kwargs):
        self._norm = MidpointNormalize(midpoint=0)
        self._x = np.arange(len(rank_metric))
        self.rankings = np.asarray(rank_metric.values)
        self.RES = RES
        self._im_matrix = np.tile(self.rankings, (2, 1))

        self.figsize = figsize
        self.term = term
        self.cmap = cmap
        self.ofname = ofname

        self._pos_label = pheno_pos
        self._neg_label = pheno_neg
        self._zero_score_ind = np.abs(self.rankings).argmin()
        self._z_score_label = 'Zero score at ' + str(self._zero_score_ind)
        self._hit_indices = hit_indices
        self.module = 'tmp' if ofname is None else ofname.split(".")[-2]
        if self.module == 'ssgsea':
            self._nes_label = 'ES: ' + "{:.3f}".format(float(nes))
            self._pval_label = 'Pval: invalid for ssgsea'
            self._fdr_label = 'FDR: invalid for ssgsea'
        else:
            self._nes_label = 'NES: ' + "{:.3f}".format(float(nes))
            self._pval_label = 'Pval: ' + "{:.3f}".format(float(pval))
            self._fdr_label = 'FDR: ' + "{:.3f}".format(float(fdr))

        plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})

        if hasattr(sys, 'ps1') and (self.ofname is None):
            self.fig = plt.figure(figsize=self.figsize)
        else:
            self.fig = Figure(figsize=self.figsize)
            self._canvas = FigureCanvas(self.fig)

        self.fig.suptitle(self.term, fontsize=16, fontweight='bold')

    def axes_rank(self, rect):
        ax1 = self.fig.add_axes(rect, sharex=self.ax)
        if self.module == 'ssgsea':
            ax1.fill_between(self._x, y1=np.log(self.rankings), y2=0, color='#C9D3DB')
            ax1.set_ylabel("log ranked metric", fontsize=14)
        else:
            ax1.fill_between(self._x, y1=self.rankings, y2=0, color='#C9D3DB')
            ax1.set_ylabel("Ranked list metric", fontsize=14)

        ax1.text(.05, .9, self._pos_label, color='red',
                 horizontalalignment='left', verticalalignment='top',
                 transform=ax1.transAxes)
        ax1.text(.95, .05, self._neg_label, color='Blue',
                 horizontalalignment='right', verticalalignment='bottom',
                 transform=ax1.transAxes)
        trans1 = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)
        ax1.vlines(self._zero_score_ind, 0, 1, linewidth=.5,
                   transform=trans1, linestyles='--', color='grey')

        hap = self._zero_score_ind / max(self._x)
        if hap < 0.25:
            ha = 'left'
        elif hap > 0.75:
            ha = 'right'
        else:
            ha = 'center'
        ax1.text(hap, 0.5, self._z_score_label,
                 horizontalalignment=ha, verticalalignment='center',
                 transform=ax1.transAxes)
        ax1.set_xlabel("Rank in Ordered Dataset", fontsize=14)
        ax1.spines['top'].set_visible(False)
        ax1.tick_params(axis='both', which='both', top=False, right=False, left=False)
        ax1.locator_params(axis='y', nbins=5)
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda tick_loc, tick_num: '{:.1f}'.format(tick_loc)))

    def axes_hits(self, rect):
        ax2 = self.fig.add_axes(rect, sharex=self.ax)
        trans2 = transforms.blended_transform_factory(ax2.transData, ax2.transAxes)
        ax2.vlines(self._hit_indices, 0, 1, linewidth=.5, transform=trans2)
        ax2.spines['bottom'].set_visible(False)
        ax2.tick_params(axis='both', which='both', bottom=False, top=False,
                        right=False, left=False, labelbottom=False, labelleft=False)

    def axes_cmap(self, rect):
        ax3 = self.fig.add_axes(rect, sharex=self.ax)
        ax3.imshow(self._im_matrix, aspect='auto', norm=self._norm,
                   cmap=self.cmap, interpolation='none')
        ax3.spines['bottom'].set_visible(False)
        ax3.tick_params(axis='both', which='both', bottom=False, top=False,
                        right=False, left=False, labelbottom=False, labelleft=False)

    def axes_stat(self, rect):
        ax4 = self.fig.add_axes(rect)
        ax4.plot(self._x, self.RES, linewidth=4, color='#88C544')
        ax4.text(.1, .1, self._fdr_label, transform=ax4.transAxes)
        ax4.text(.1, .2, self._pval_label, transform=ax4.transAxes)
        ax4.text(.1, .3, self._nes_label, transform=ax4.transAxes)

        trans4 = transforms.blended_transform_factory(ax4.transAxes, ax4.transData)
        ax4.hlines(0, 0, 1, linewidth=.5, transform=trans4, color='grey')
        ax4.set_ylabel("Enrichment Score", fontsize=14)
        ax4.tick_params(axis='both', which='both', bottom=False, top=False,
                        right=False, labelbottom=False)
        ax4.locator_params(axis='y', nbins=5)
        ax4.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda tick_loc, tick_num: '{:.1f}'.format(tick_loc)))
        self.ax = ax4

    def add_axes(self):
        self.axes_stat([0.1, 0.5, 0.8, 0.4])
        self.axes_hits([0.1, 0.45, 0.8, 0.05])
        self.axes_cmap([0.1, 0.40, 0.8, 0.05])
        self.axes_rank([0.1, 0.1, 0.8, 0.3])

    def savefig(self, bbox_inches='tight', dpi=300):
        if hasattr(sys, 'ps1') and (self.ofname is not None):
            self.fig.savefig(self.ofname, bbox_inches=bbox_inches, dpi=dpi)
        elif self.ofname is None:
            return
        else:
            self._canvas.print_figure(self.ofname, bbox_inches=bbox_inches, dpi=300)


def gseaplot(rank_metric, term, hit_indices, nes, pval, fdr, RES,
             pheno_pos='', pheno_neg='', figsize=(6, 5.5),
             cmap='seismic', ofname=None, **kwargs):
    """Functional wrapper around :class:`GSEAPlot` (build + save in one call)."""
    g = GSEAPlot(rank_metric, term, hit_indices, nes, pval, fdr, RES,
                 pheno_pos, pheno_neg, figsize, cmap, ofname)
    g.add_axes()
    g.savefig()


def adjust_spines(ax, spines):
    """Keep only the named spines/ticks (e.g. ``['left', 'bottom']``)."""
    for loc, spine in ax.spines.items():
        if loc in spines:
            continue
        spine.set_color('none')
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.xaxis.set_ticks([])


def dotplot(df, column='Adjusted P-value', title='', cutoff=0.05, top_term=10,
            sizes=None, norm=None, legend=True, figsize=(6, 5.5),
            cmap='RdBu_r', ofname=None, **kwargs):
    """Dot-plot of enrichment results (term vs -log10 p, sized by hits)."""
    colname = column
    if colname in ['Adjusted P-value', 'P-value']:
        can_be_coerced = df[colname].map(isfloat)
        if np.sum(~can_be_coerced) > 0:
            raise ValueError('some value in %s could not be typecast to `float`' % colname)
        df.loc[:, colname] = df[colname].map(float)
        df = df[df[colname] <= cutoff]
        if len(df) < 1:
            return "Warning: No enrich terms when cutoff = %s" % cutoff
        df = df.assign(logAP=lambda x: - x[colname].apply(np.log10))
        colname = 'logAP'
    df = df.sort_values(by=colname).iloc[-top_term:, :]
    temp = df['Overlap'].str.split("/", expand=True).astype(int)
    df = df.assign(Hits=temp.iloc[:, 0], Background=temp.iloc[:, 1])
    df = df.assign(Hits_ratio=lambda x: x.Hits / x.Background)
    x = df.loc[:, colname].values
    combined_score = df['Combined Score'].round().astype('int')
    y = [i for i in range(0, len(df))]
    ylabels = df['Term'].values

    levels = numbers = np.sort(df.Hits.unique())
    if norm is None:
        norm = Normalize()
    elif isinstance(norm, tuple):
        norm = Normalize(*norm)
    elif not isinstance(norm, Normalize):
        raise ValueError("``size_norm`` must be None, tuple, or Normalize object.")
    min_width, max_width = np.r_[20, 100] * plt.rcParams["lines.linewidth"]
    norm.clip = True
    if not norm.scaled():
        norm(np.asarray(numbers))
    scl = norm(numbers)
    widths = np.asarray(min_width + scl * (max_width - min_width))
    if scl.mask.any():
        widths[scl.mask] = 0
    sizes = dict(zip(levels, widths))
    df['sizes'] = df.Hits.map(sizes)
    area = df['sizes'].values

    if hasattr(sys, 'ps1') and (ofname is None):
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = Figure(figsize=figsize)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
    vmin = np.percentile(combined_score.min(), 2)
    vmax = np.percentile(combined_score.max(), 98)
    sc = ax.scatter(x=x, y=y, s=area, edgecolors='face', c=combined_score,
                    cmap=cmap, vmin=vmin, vmax=vmax)

    if column in ['Adjusted P-value', 'P-value']:
        xlabel = "-log$_{10}$(%s)" % column
    else:
        xlabel = column
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.yaxis.set_major_locator(plt.FixedLocator(y))
    ax.yaxis.set_major_formatter(plt.FixedFormatter(ylabels))
    ax.set_yticklabels(ylabels, fontsize=16)
    ax.grid()
    cax = fig.add_axes([0.95, 0.20, 0.03, 0.22])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.ax.tick_params(right=True)
    cbar.ax.set_title('Combined\nScore', loc='left', fontsize=12)

    if len(df) >= 3:
        idx = [area.argmax(), np.abs(area - area.mean()).argmin(), area.argmin()]
        idx = unique(idx)
    else:
        idx = range(len(df))
    label = df.iloc[idx, df.columns.get_loc('Hits')]

    if legend:
        legend_markers = []
        for ix in idx:
            legend_markers.append(ax.scatter([], [], s=area[ix], c='b'))
        ax.legend(legend_markers, label, title='Hits')
    ax.set_title(title, fontsize=20, fontweight='bold')

    if ofname is not None:
        fig.savefig(ofname, bbox_inches='tight', dpi=300)
        return
    return ax


def barplot(df, column='Adjusted P-value', title="", cutoff=0.05, top_term=10,
            figsize=(6.5, 6), color='salmon', ofname=None, **kwargs):
    """Horizontal bar-plot of the top enrichment terms by -log10 p."""
    colname = column
    if colname in ['Adjusted P-value', 'P-value']:
        can_be_coerced = df[colname].map(isfloat)
        if np.sum(~can_be_coerced) > 0:
            raise ValueError('some value in %s could not be typecast to `float`' % colname)
        df.loc[:, colname] = df[colname].map(float)
        df = df[df[colname] <= cutoff]
        if len(df) < 1:
            return "Warning: No enrich terms using library %s when cutoff = %s" % (title, cutoff)
        df = df.assign(logAP=lambda x: - x[colname].apply(np.log10))
        colname = 'logAP'

    dd = df.sort_values(by=colname).iloc[-top_term:, :]
    if hasattr(sys, 'ps1') and (ofname is None):
        fig = plt.figure(figsize=figsize)
    else:
        fig = Figure(figsize=figsize)
        canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    bar = dd.plot.barh(x='Term', y=colname, color=color, alpha=0.75,
                       fontsize=16, ax=ax)

    if column in ['Adjusted P-value', 'P-value']:
        xlabel = "-log$_{10}$(%s)" % column
    else:
        xlabel = column
    bar.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    bar.set_ylabel("")
    bar.set_title(title, fontsize=24, fontweight='bold')
    bar.xaxis.set_major_locator(MaxNLocator(integer=True))
    bar.legend_.remove()
    adjust_spines(ax, spines=['left', 'bottom'])

    if hasattr(sys, 'ps1') and (ofname is not None):
        fig.savefig(ofname, bbox_inches='tight', dpi=300)
        return
    elif ofname is None:
        return
    else:
        canvas.print_figure(ofname, bbox_inches='tight', dpi=300)
        return
    return ax

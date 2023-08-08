# -*- encoding: utf8 -*-

# Author: Nobuhito Manome
# License: BSD 3 clause

import os
import numpy as np
from matplotlib import pyplot as plt

def savefig_line2D(data, xlim=None, ylim=None, xscale=None, yscale=None, markevery=1000, labels=None, xlabel=None, ylabel=None, title=None, path='output/', filename='figure', extension='png', is_show=True):
    '''
    Save line2D figure.
    ----------
    Parameters
    ----------
    data : array-like, shape = [n_models, steps]
        data.
        n_models ≤ 20.
    xlim : list of float optional (default=None)
        xlim.
    ylim : list of float optional (default=None)
        ylim.
    xscale: str, optional (default=None)
        set the xaxis' scale.
    yscale: str, optional (default=None)
        set the yaxis' scale.
    markevery : int (default=1000)
        markevery.
    labels: list of str, optional (default=None)
        labels.
    xlabel: str, optional (default=None)
        xlabel.
    ylabel: str, optional (default=None)
        ylabel.
    title : str, optional (default=None)
        title.
    path: str, optional (default='output/')
        path.
    filename: str, optional (default='figure')
        filename.
    extension: str, optional (default='png')
        extension.
    is_show: bool, optional (default=True)
        if is_show: show figure.
    '''
    # Make directory
    os.makedirs(path, exist_ok=True)
    # Make figure
    plt.figure(figsize=(10, 5.5))
    plt.rcParams['font.size'] = 20
    plt.grid(color = 'gray', linestyle='--')
    linewidth = 4 # 2
    linestyle = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
    markersize = 12 # 8
    marker = ['o', '^', 'v', '>', '<', 's', 'D', '*', '2', '1', '3', '4', 'p', 'h', 'H', 'd', '$◎$', 'x', '$∴$', '+']
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#393B79', '#6B6ECF', '#637939', '#B5CF6B', '#8C6D31', '#E7BA52', '#843C39', '#D6616B', '#7B4173', '#CE6DBD']
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    if xscale is not None: plt.xscale(xscale)
    if yscale is not None: plt.yscale(yscale)
    for idx, elem in enumerate(data):
        plt.plot(elem, linewidth=linewidth, linestyle=linestyle[idx%len(linestyle)], markersize=markersize, markevery=int(markevery), marker=marker[idx%len(marker)], color=color[idx%len(color)])
    if title is not None: plt.title(title, color='#000000')
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    if labels is not None: plt.legend(labels, loc='upper left', bbox_to_anchor=(1.02, 1.02))
    plt.savefig('{}{}.{}'.format(path, filename, extension), bbox_inches='tight')
    if is_show: plt.show()
    plt.close()
    print('save line2D figure (file: {}{}.{})'.format(path, filename, extension))

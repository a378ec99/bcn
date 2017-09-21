"""Visualization module.

Notes
-----
This module defines several functions that can generate different types of visualizations for diagnostic purposes.
"""
from __future__ import division, absolute_import


__all__ = ['visualize_dependences', 'visualize_correlations']

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as pl
import seaborn.apionly as sb


def visualize_dependences(data, space='sample', file_name='test_dependences', truth_available=True, format='.png', max_plots=10, max_points=20): # '.eps'
    """Visualizes dependences to diagnose how the original, guess and reconstruction looks like on that backgrounnd.

    Parameters
    ----------
    data : dict
    
    space : str, optional (default = 'sample')
        
    file_name : str, optional (default = 'test_dependences')

    blind_recovery : bool

    format : str

    max_plots : int

    max_points : int
    """
    n_subplots = min(max_plots, len(data[space]['estimated_pairs']))
    n_points = min(max_points, data[space]['shape'][1])

    #print 'a', data['sample']['estimated_stds'] # TODO WHY IS A different from a?
    #print 'b', data['sample']['true_stds']

    if truth_available is False:
        indices = []
        for index, item in enumerate(data[space]['true_pairs']):
            if item in data[space]['estimated_pairs']:
                indices.append(index)
    else:
        indices = range(len(data[space]['true_pairs']))
        
    fig = pl.figure(figsize=(10, 10 * n_subplots))
    
    for k, i in zip(range(n_subplots), indices):
        ax = fig.add_subplot(n_subplots, 1, k + 1)
        for j in xrange(n_points):
            
            if truth_available is False:
                ax.plot(data[space]['mixed'][data[space]['estimated_pairs'][i][1]][j], data[space]['mixed'][data[space]['estimated_pairs'][i][0]][j], 'o', color='red', alpha=0.6)
                ax.plot(data[space]['estimated_signal'][data[space]['estimated_pairs'][i][1]][j], data[space]['estimated_signal'][data[space]['estimated_pairs'][i][0]][j], 'D', color='blue', alpha=0.6)
                ax.plot(data[space]['guess_X'][data[space]['estimated_pairs'][i][1]][j], data[space]['guess_X'][data[space]['estimated_pairs'][i][0]][j], 'o', color='brown', alpha=0.6)
            else:
                ax.plot(data[space]['mixed'][data[space]['true_pairs'][i][1]][j], data[space]['mixed'][data[space]['true_pairs'][i][0]][j], 'o', color='red', alpha=0.6)
                ax.plot(data[space]['estimated_signal'][data[space]['true_pairs'][i][1]][j], data[space]['estimated_signal'][data[space]['true_pairs'][i][0]][j], 'D', color='blue', alpha=0.6)
                ax.plot(data[space]['guess_X'][data[space]['true_pairs'][i][1]][j], data[space]['guess_X'][data[space]['true_pairs'][i][0]][j], 'o', color='brown', alpha=0.6)
                ax.plot(data[space]['signal'][data[space]['true_pairs'][i][1]][j], data[space]['signal'][data[space]['true_pairs'][i][0]][j], 'o', color='green', alpha=0.6)
                ax.plot([data[space]['signal'][data[space]['true_pairs'][i][1]][j], data[space]['mixed'][data[space]['true_pairs'][i][1]][j]], [data[space]['signal'][data[space]['true_pairs'][i][0]][j], data[space]['mixed'][data[space]['true_pairs'][i][0]][j]], '-', color='red', alpha=0.6)
                ax.plot([data[space]['signal'][data[space]['true_pairs'][i][1]][j], data[space]['estimated_signal'][data[space]['true_pairs'][i][1]][j]], [data[space]['signal'][data[space]['true_pairs'][i][0]][j], data[space]['estimated_signal'][data[space]['true_pairs'][i][0]][j]], '-', color='blue', zorder=10, alpha=0.6)

                #print data[space]['true_directions']
                #print data[space]['true_stds']

                # NOTE Show estimated linear dependency
                direction = data[space]['true_directions'][i]
                std_b, std_a = data[space]['true_stds'][i]
                std_b = std_b * -1 * direction
                m = -std_b / float(std_a)
                ax.plot(list(ax.get_xlim()), [m * p + 0.0 for p in ax.get_xlim()], '-', color='orange', alpha=0.6)

                #print data[space]['estimated_directions']
                #print data[space]['estimated_stds']
                
                direction = data[space]['estimated_directions'][i]
                std_b, std_a = data[space]['estimated_stds'][i]
                std_b = std_b * -1 * direction
                m = -std_b / float(std_a)
                ax.plot(list(ax.get_xlim()), [m * p + 0.0 for p in ax.get_xlim()], '--', color='black', alpha=0.6)

                # TODO Plot mean, plus 1 std in both directions marked on Axes both for true and for estimated! Then see if given corret mean is better?

                
        # NOTE Make plot pretty
        #ax.spines['right'].set_color('none')
        #ax.spines['top'].set_color('none')
        #ax.spines['left'].set_smart_bounds(True)
        #ax.spines['bottom'].set_smart_bounds(True)

        #ax.xaxis.set_ticks_position('bottom')
        #ax.yaxis.set_ticks_position('left')

        #ax.yaxis.set_tick_params(width=4, which='major')
        #ax.yaxis.set_tick_params(size=20, which='major')
        #ax.xaxis.set_tick_params(width=4, which='major')
        #ax.xaxis.set_tick_params(size=20, which='major')

        #ax.yaxis.set_tick_params(width=4, which='minor')
        #ax.yaxis.set_tick_params(size=10, which='minor')
        #ax.xaxis.set_tick_params(width=4, which='minor')
        #ax.xaxis.set_tick_params(size=20, which='minor')

        #for axis in ['bottom', 'left']:
        #    ax.spines[axis].set_linewidth(4)

        #ax.xaxis.set_major_formatter(pl.NullFormatter())
        #ax.yaxis.set_major_formatter(pl.NullFormatter())
        ax.axis('equal')

    fig.savefig(file_name + format)


def visualize_correlations(data, space='sample', file_name='test_correlations', blind=False, format='.png'):
    """Visualize estimated or true correlation matrices.

    Parameters
    ----------
    data : dict

    space : str, (default = 'sample')

    file_name : str, (default = 'test_correlations')

    blind : bool

    format : str
    """
    if blind:
        correlations = data[space]['estimated_correlations']
    else:
        correlations = data[space]['true_correlations']

    # NOTE Plot correlation matrix
    fig = pl.figure(figsize=(10, 20))
    ax = fig.add_subplot(211)
    sb.heatmap(correlations, cmap=pl.cm.viridis, square=True, ax=ax, cbar_kws={'shrink': 0.5}, xticklabels=False, yticklabels=False)
    ax.set_xlabel(space)
    ax.set_ylabel(space)

    # NOTE Plot threshold vs. number of correlations
    ax = fig.add_subplot(212)
    trimmed = np.trim_zeros(np.sort(np.tril(np.absolute(correlations), -1).ravel()))
    ax.set_xlabel('threshold')
    ax.set_ylabel('n pairs')
    ax.plot(trimmed, np.arange(1, len(trimmed) + 1)[::-1], '-', alpha=0.8)

    fig.savefig(file_name + format)


def visualize_performance(errors, x, y, x_name, y_name, file_name, format='.png'):
    """
    """
    xlabel = x_name
    ylabel = y_name
    xticklabels = x
    yticklabels = y

    if np.sum(np.asarray(xticklabels, dtype=int) - xticklabels) != 0:
        xticklabels = ['{:.2f}'.format(i) for i in xticklabels]
    if np.sum(np.asarray(yticklabels, dtype=int) - yticklabels) != 0:
        yticklabels = ['{:.2f}'.format(i) for i in yticklabels]

    fig = pl.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_title('Recovery Performance')
    ax_seaborn = sb.heatmap(np.log2(errors), vmin=-2.5, vmax=2.5, cmap=pl.cm.inferno, xticklabels=xticklabels, yticklabels=yticklabels, ax=ax, cbar_kws={'shrink': 0.5})
    ax_seaborn.set_ylabel(ylabel)
    ax_seaborn.set_xlabel(xlabel)

    # NOTE Colorbar
    cbar = ax_seaborn.collections[0].colorbar
    cbar.set_ticks([-2.0, -1.0, 0.0, 1.0, 2.0])
    cbar.set_ticklabels([r'$>4$x better', r'  $2$x better', '  No correction', r'  $2$x worse', r'$>4$x worse'])

    ax.invert_yaxis()
    ax.tick_params(axis='both', which='both', length=0)

    fig.savefig(file_name + format)



        
# visualize batch effects (plus annotation) for blind and add contrast if known

# visualize in PCA (plus annotation) for blind and add contrast if known


















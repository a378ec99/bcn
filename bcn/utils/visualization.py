"""Visualization of data and recovery.

Notes
-----
Defines several functions that can generate different types of visualizations for diagnostic purposes.
"""
from __future__ import division, absolute_import

import numpy as np

import pylab as pl
import seaborn as sb


def show_absolute(signal, kind, unshuffled=False, map_backward=None, vmin=-4, vmax=4):
    cmap = sb.diverging_palette(250, 15, s=75, l=40, as_cmap=True, center="dark")
    indices_x = np.arange(signal.shape[0], dtype=int)
    indices_y = np.arange(signal.shape[1], dtype=int)
    fig = pl.figure(figsize=(6 * (signal.shape[1] / signal.shape[0]), 6))
    ax = fig.add_subplot(111)
    if unshuffled:
        ax.set_title('{}'.format(kind))
        indices_x = np.asarray([map_backward['sample'][i] for i in indices_x])
        indices_y = np.asarray([map_backward['feature'][i] for i in indices_y])
    else:
        ax.set_title('{} (shuffled)'.format(kind))
    ax_seaborn = sb.heatmap(signal, vmin=vmin, vmax=vmax, cmap=cmap, ax=ax, cbar_kws={'shrink': 0.5}, xticklabels=indices_y, yticklabels=indices_x)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xlabel('Features')
    ax.set_ylabel('Samples')

    
def show_dependences(signal, pairs, space, n_pairs=5, n_points=50):
    cmap = sb.diverging_palette(250, 15, s=75, l=40, n=10, center="dark")
    if space == 'feature':
        shape = signal.T.shape
    if space == 'sample':
        shape = signal.shape
    pairs = pairs[space]
    for n, i in enumerate(np.random.choice(np.arange(len(pairs), dtype=int), min(n_pairs, len(pairs)), replace=False)):
        j = np.random.choice(np.arange(shape[1], dtype=int), min(n_points, shape[1]), replace=False)
        if space == 'sample':
            grid = sb.jointplot(signal[np.atleast_2d(pairs[i][1]), j], signal[np.atleast_2d(pairs[i][0]), j], ylim=(-4, 4), xlim=(-4, 4), alpha=0.6, stat_func=None, color='black')
            grid.set_axis_labels('Sample {}'.format(pairs[i][1]), 'Sample {}'.format(pairs[i][0]))
        if space == 'feature':
            grid = sb.jointplot(signal[j[:, None], pairs[i][1]], signal[j[:, None], pairs[i][0]], ylim=(-4, 4), xlim=(-4, 4), alpha=0.6, stat_func=None, color='black')
            grid.set_axis_labels('Feature {}'.format(pairs[i][1]), 'Feature {}'.format(pairs[i][0]))
        pl.setp(grid.ax_marg_y.patches, color=cmap[2])
        pl.setp(grid.ax_marg_x.patches, color=cmap[-2])

        
def show_independences(signal, pairs, space, n_pairs=5, n_points=50):
    if space == 'feature':
        shape = signal.T.shape
    if space == 'sample':
        shape = signal.shape
    true_pairs = set()
    for pair in pairs[space]:
        true_pairs.add((pair[0], pair[1]))
        true_pairs.add((pair[1], pair[0]))
    all_pairs = set()
    for i in xrange(shape[0]):
        for j in range(shape[0]):
            all_pairs.add((i, j))
            all_pairs.add((j, i))
    non_pairs = all_pairs - true_pairs
    pairs = {space: np.asarray(list(non_pairs), dtype=int)}
    show_dependences(signal, pairs, space, n_pairs=n_pairs, n_points=n_points)


def show_dependence_structure(correlations, space, unshuffled=False, map_backward=None):
    cmap = sb.diverging_palette(250, 15, s=75, l=40, as_cmap=True, center="dark")
    indices = np.arange(correlations[space].shape[0], dtype=int)
    if space == 'feature':
        size = 6 * (correlations['feature'].shape[0] / correlations['sample'].shape[0])
    if space == 'sample':
        size = 6
    fig = pl.figure(figsize=(size, size))
    ax = fig.add_subplot(111)
    if unshuffled:
        ax.set_title('Correlations')
        indices = np.asarray([map_backward[space][i] for i in indices])
    else:
        ax.set_title('Correlations (shuffled)')
    sb.heatmap(correlations[space], cmap=cmap, vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={'shrink': 0.5}, xticklabels=indices, yticklabels=indices)
    if space == 'feature':
        ax.set_xlabel('Features')
        ax.set_ylabel('Features')
    if space == 'sample':
        ax.set_xlabel('Samples')
        ax.set_ylabel('Samples')

        
def show_threshold(correlations, threshold, space):
    cmap = sb.diverging_palette(250, 15, s=75, l=40, n=10, center="dark")
    fig = pl.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    trimmed = np.trim_zeros(np.sort(np.tril(np.absolute(correlations[space]), -1).ravel()))
    ax.set_xlabel('Threshold')
    ax.set_ylabel('# pairs')
    x = trimmed
    y = np.arange(1, len(trimmed) + 1)
    ax.plot(x, y[::-1], '-', alpha=0.8, color='black')
    ax.axvline(threshold, min(x), max(x), linestyle='dashed', color=cmap[2])
    sb.despine()


    
'''
from __future__ import division, absolute_import

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as pl
import seaborn.apionly as sb


def pair_index(pairs, pair):
    """ Determines the index of a pair in a list of pairs.

    Parameters
    ----------
    pairs : list
        Sequence of pairs (e.g. tuples).
    pair : list or tuple, len = 2
        The pair to be looked up.

    Returns
    -------
    index : int
        The index of the pair looked up.
    """
    index = np.where(np.all(pairs == pair, axis=1)) #[i for i, item in enumerate(pairs) if ((item[0] == pair[0]) and (item[1] == pair[1])) or ((item[0] == pair[1]) and (item[1] == pair[0]))]
    try:
        index = index[0][0]
    except IndexError:
        index = None
    return index

    
def pair_overlap(true_pairs, estimated_pairs):
    """ Determines the possible 3 groups of overlap (only_in_estimated, only_in_true, in both)

    Parameters
    ----------
    true_pairs : list
        Sequence of pairs (e.g. tuples).
    estimated_pairs : list
        Sequence of pairs (e.g. tuples).
    
    Returns
    -------
    d : dict
        Dictionary of the 3 groups of overlap.
    """
    only_in_estimated = []
    for pair in estimated_pairs:
        if pair_index(true_pairs, pair) is None:
            only_in_estimated.append(pair)

    only_in_true = []
    for pair in true_pairs:
        if pair_index(estimated_pairs, pair) is None:
            only_in_true.append(pair)

    in_both = []
    for pair in set([(item[0], item[1]) for item in estimated_pairs] + [(item[0], item[1]) for item in true_pairs]):
        pair = np.asarray(pair)
        if (pair_index(true_pairs, pair) is not None) and (pair_index(estimated_pairs, pair) is not None):
            in_both.append(pair)

    d = {'only_in_estimated': np.asarray(only_in_estimated), 'only_in_true': np.asarray(only_in_true), 'in_both': np.asarray(in_both)}
    
    return d

    
def visualize_absolute(data, space='sample', file_name='../../out/test_absolute', format='.png', recovered=False):
    """Visualize absolute values of the dataset, with signal, mixed and bias shown.

    Parameters
    ----------
    data : Data object
        The dataset to be visualized.
    space : str
        Space of array to be visualized, e.g. sample or feature.
    file_name : str
        Name and path of figure output.
    format : str
        Format of figure.
    recovered : bool
        If recovery has been performed; then show also the results.
    """

    if recovered == True:
        fig = pl.figure(figsize=(10 * 2, 10 * 3))

        ax = fig.add_subplot(3, 2, 1)
        ax.set_title('Signal')
        ax_seaborn = sb.heatmap(data.d[space]['signal'], vmin=-2.5, vmax=2.5, cmap=pl.cm.inferno, ax=ax, cbar_kws={'shrink': 0.5}, xticklabels=False, yticklabels=False, robust=True)
        ax.tick_params(axis='both', which='both', length=0)

        ax = fig.add_subplot(3, 2, 2)
        ax.set_title('Recovered Signal')
        ax_seaborn = sb.heatmap(data.d[space]['estimated_signal'], vmin=-2.5, vmax=2.5, cmap=pl.cm.inferno, ax=ax, cbar_kws={'shrink': 0.5}, xticklabels=False, yticklabels=False, robust=True)
        ax.tick_params(axis='both', which='both', length=0)

        ax = fig.add_subplot(3, 2, 3)
        ax.set_title('Mixed')
        ax_seaborn = sb.heatmap(data.d[space]['mixed'], vmin=-2.5, vmax=2.5, cmap=pl.cm.inferno, ax=ax, cbar_kws={'shrink': 0.5}, xticklabels=False, yticklabels=False, robust=True)
        ax.tick_params(axis='both', which='both', length=0)

        ax = fig.add_subplot(3, 2, 4)
        ax.set_title('Guess')
        ax_seaborn = sb.heatmap(data.d[space]['guess_X'], vmin=-2.5, vmax=2.5, cmap=pl.cm.inferno, ax=ax, cbar_kws={'shrink': 0.5}, xticklabels=False, yticklabels=False, robust=True)
        ax.tick_params(axis='both', which='both', length=0)

        ax = fig.add_subplot(3, 2, 5)
        ax.set_title('Bias (noise_amplitude {})'.format(data.noise_amplitude))
        ax_seaborn = sb.heatmap(data.d[space]['true_bias'], vmin=-2.5, vmax=2.5, cmap=pl.cm.inferno, ax=ax, cbar_kws={'shrink': 0.5}, xticklabels=False, yticklabels=False, robust=True)
        ax.tick_params(axis='both', which='both', length=0)
        
        ax = fig.add_subplot(3, 2, 6)
        ax.set_title('Recovered Bias)'.format(data.noise_amplitude))
        ax_seaborn = sb.heatmap(data.d[space]['estimated_bias'], vmin=-2.5, vmax=2.5, cmap=pl.cm.inferno, ax=ax, cbar_kws={'shrink': 0.5}, xticklabels=False, yticklabels=False, robust=True)
        ax.tick_params(axis='both', which='both', length=0)

        fig.savefig(file_name + format)

    else:
        fig = pl.figure(figsize=(10, 10 * 3))

        ax = fig.add_subplot(3, 1, 1)
        ax.set_title('Signal')
        ax_seaborn = sb.heatmap(data.d[space]['signal'], vmin=-2.5, vmax=2.5, cmap=pl.cm.inferno, ax=ax, cbar_kws={'shrink': 0.5}, xticklabels=False, yticklabels=False, robust=True)
        ax.tick_params(axis='both', which='both', length=0)

        ax = fig.add_subplot(3, 1, 2)
        ax.set_title('Mixed')
        ax_seaborn = sb.heatmap(data.d[space]['mixed'], vmin=-2.5, vmax=2.5, cmap=pl.cm.inferno, ax=ax, cbar_kws={'shrink': 0.5}, xticklabels=False, yticklabels=False, robust=True)
        ax.tick_params(axis='both', which='both', length=0)

        ax = fig.add_subplot(3, 1, 3)
        ax.set_title('Bias (noise_amplitude {})'.format(data.noise_amplitude))
        ax_seaborn = sb.heatmap(data.d[space]['true_bias'], vmin=-2.5, vmax=2.5, cmap=pl.cm.inferno, ax=ax, cbar_kws={'shrink': 0.5}, xticklabels=False, yticklabels=False, robust=True)
        ax.tick_params(axis='both', which='both', length=0)

        fig.savefig(file_name + format)
        

def visualize_dependences(data, space='sample', file_name='../../out/test_dependences', truth_available=True, estimate_available=True, recovery_available=True, format='.png', max_plots=10, max_points=40): # '.eps'
    """Visualizes dependences to diagnose how the original, guess and reconstruction looks like on that backgrounnd.

    Parameters
    ----------
    data : Data object
        The dataset to be visualized.
    space : str
        Space of array to be visualized, e.g. sample or feature.
    file_name : str
        Name and path of figure output.
    truth_available : bool
        If ground truth avaiable set here.
    estimate_available: bool
        If recovery data is available in Data object.
    recovery_available: bool
        If recovery data is available in Data object.
    format : str
        Format of figure.
    max_plots : int
        Maximum number of plots (features/samples) used.
    max_points : int
        Maximum number of points to be plotted.
    """
    if truth_available and estimate_available:

        d = pair_overlap(data.d[space]['true_pairs'], data.d[space]['estimated_pairs'])
        
        for pairs_name, pairs in d.iteritems():

            if len(pairs) == 0:
                continue

            n_subplots = min(max_plots, len(pairs))
            n_points = min(max_points, data.d[space]['shape'][1])

            indices = range(len(pairs))

            fig = pl.figure(figsize=(10, 10 * n_subplots))

            for k, i in zip(range(n_subplots), indices):
                ax = fig.add_subplot(n_subplots, 1, k + 1)
                for j in xrange(n_points):
                    ax.plot(data.d[space]['mixed'][pairs[i][1]][j], data.d[space]['mixed'][pairs[i][0]][j], 'o', color='red', alpha=0.6)
                    if recovery_available:
                        ax.plot(data.d[space]['estimated_signal'][pairs[i][1]][j], data.d[space]['estimated_signal'][pairs[i][0]][j], 'D', color='blue', alpha=0.6)
                        ax.plot(data.d[space]['guess_X'][pairs[i][1]][j], data.d[space]['guess_X'][pairs[i][0]][j], 'o', color='brown', alpha=0.6)
                    ax.plot(data.d[space]['signal'][pairs[i][1]][j], data.d[space]['signal'][pairs[i][0]][j], 'o', color='green', alpha=0.6)
                    ax.plot([data.d[space]['signal'][pairs[i][1]][j], data.d[space]['mixed'][pairs[i][1]][j]], [data.d[space]['signal'][pairs[i][0]][j], data.d[space]['mixed'][pairs[i][0]][j]], '-', color='red', alpha=0.6)
                    if recovery_available:
                        ax.plot([data.d[space]['signal'][pairs[i][1]][j], data.d[space]['estimated_signal'][pairs[i][1]][j]], [data.d[space]['signal'][pairs[i][0]][j], data.d[space]['estimated_signal'][pairs[i][0]][j]], '-', color='blue', alpha=0.6) # , zorder=10

                if pairs_name == 'only_in_true' or pairs_name == 'in_both':
                    direction = data.d[space]['true_directions'][pair_index(data.d[space]['true_pairs'], pairs[i])]
                    std_b, std_a = data.d[space]['true_stds'][pair_index(data.d[space]['true_pairs'], pairs[i])]
                    std_b = std_b * -1 * direction
                    m = -std_b / float(std_a)
                    ax.plot(list(ax.get_xlim()), [m * p + 0.0 for p in ax.get_xlim()], '-', color='orange', alpha=0.6)

                if pairs_name == 'only_in_estimated' or pairs_name == 'in_both':
                    direction = data.d[space]['estimated_directions'][pair_index(data.d[space]['estimated_pairs'], pairs[i])]
                    std_b, std_a = data.d[space]['estimated_stds'][pair_index(data.d[space]['estimated_pairs'], pairs[i])]
                    std_b = std_b * -1 * direction
                    m = -std_b / float(std_a)
                    ax.plot(list(ax.get_xlim()), [m * p + 0.0 for p in ax.get_xlim()], '--', color='black', alpha=0.6)

                ax.axis('equal')
            fig.savefig(file_name + '_' + pairs_name + format)

    
    if truth_available and not estimate_available and not recovery_available:

        n_subplots = min(max_plots, len(data.d[space]['true_pairs']))
        n_points = min(max_points, data.d[space]['shape'][1])

        indices = range(len(data.d[space]['true_pairs']))

        fig = pl.figure(figsize=(10, 10 * n_subplots))

        for k, i in zip(range(n_subplots), indices):
            ax = fig.add_subplot(n_subplots, 1, k + 1)
            for j in xrange(n_points):

                ax.plot(data.d[space]['mixed'][data.d[space]['true_pairs'][i][1]][j], data.d[space]['mixed'][data.d[space]['true_pairs'][i][0]][j], 'o', color='red', alpha=0.6)
                ax.plot(data.d[space]['signal'][data.d[space]['true_pairs'][i][1]][j], data.d[space]['signal'][data.d[space]['true_pairs'][i][0]][j], 'o', color='green', alpha=0.6)
                ax.plot([data.d[space]['signal'][data.d[space]['true_pairs'][i][1]][j], data.d[space]['mixed'][data.d[space]['true_pairs'][i][1]][j]], [data.d[space]['signal'][data.d[space]['true_pairs'][i][0]][j], data.d[space]['mixed'][data.d[space]['true_pairs'][i][0]][j]], '-', color='red', alpha=0.6)

            direction = data.d[space]['true_directions'][i]
            std_b, std_a = data.d[space]['true_stds'][i]
            std_b = std_b * -1 * direction
            m = -std_b / float(std_a)
            ax.plot(list(ax.get_xlim()), [m * p + 0.0 for p in ax.get_xlim()], '-', color='orange', alpha=0.6)

            ax.axis('equal')
        fig.savefig(file_name + format)

    
    if not truth_available and estimate_available and recovery_available:

        n_subplots = min(max_plots, len(data.d[space]['estimated_pairs']))
        n_points = min(max_points, data.d[space]['shape'][1])

        indices = range(len(data.d[space]['estimated_pairs']))

        fig = pl.figure(figsize=(10, 10 * n_subplots))

        for k, i in zip(range(n_subplots), indices):
            ax = fig.add_subplot(n_subplots, 1, k + 1)
            for j in xrange(n_points):

                ax.plot(data.d[space]['mixed'][data.d[space]['estimated_pairs'][i][1]][j], data.d[space]['mixed'][data.d[space]['estimated_pairs'][i][0]][j], 'o', color='red', alpha=0.6)
                ax.plot(data.d[space]['estimated_signal'][data.d[space]['estimated_pairs'][i][1]][j], data.d[space]['estimated_signal'][data.d[space]['estimated_pairs'][i][0]][j], 'D', color='blue', alpha=0.6)
                ax.plot(data.d[space]['guess_X'][data.d[space]['estimated_pairs'][i][1]][j], data.d[space]['guess_X'][data.d[space]['estimated_pairs'][i][0]][j], 'o', color='brown', alpha=0.6)

            direction = data.d[space]['estimated_directions'][i]
            std_b, std_a = data.d[space]['estimated_stds'][i]
            std_b = std_b * -1 * direction
            m = -std_b / float(std_a)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5) # WARNING hardcoded.
            ax.plot(list(ax.get_xlim()), [m * p + 0.0 for p in ax.get_xlim()], '--', color='black', alpha=0.6)

            ax.axis('equal')
        fig.savefig(file_name + format)
    

    if not truth_available and estimate_available and not recovery_available:

        n_subplots = min(max_plots, len(data.d[space]['estimated_pairs']))
        n_points = min(max_points, data.d[space]['shape'][1])

        indices = range(len(data.d[space]['estimated_pairs']))

        fig = pl.figure(figsize=(10, 10 * n_subplots))

        for k, i in zip(range(n_subplots), indices):
            ax = fig.add_subplot(n_subplots, 1, k + 1)
            for j in xrange(n_points):

                ax.plot(data.d[space]['mixed'][data.d[space]['estimated_pairs'][i][1]][j], data.d[space]['mixed'][data.d[space]['estimated_pairs'][i][0]][j], 'o', color='red', alpha=0.6)

            direction = data.d[space]['estimated_directions'][i]
            std_b, std_a = data.d[space]['estimated_stds'][i]
            std_b = std_b * -1 * direction
            m = -std_b / float(std_a)
            ax.plot(list(ax.get_xlim()), [m * p + 0.0 for p in ax.get_xlim()], '--', color='black', alpha=0.6)

            ax.axis('equal')
        fig.savefig(file_name + format)
        

def visualize_correlations(data, space='sample', file_name='../../out/test_correlations', truth_available=True, format='.png'):
    """Visualize estimated or true correlation matrices.

    Parameters
    ----------
    data : Data object
        The dataset to be visualized.
    space : str
        Space of array to be visulizaed, e.g. sample or feature.
    file_name : str
        Name and path of figure output.
    truth_available : bool
        If ground truth avaiable set here.
    format : str
        Format of figure.
    """
    if truth_available:
        correlations = data.d[space]['true_correlations']
    else:
        correlations = data.d[space]['estimated_correlations']

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
    

def visualize_performance(errors, x, y, x_name, y_name, file_name='../../out/test_performance', format='.png'):
    """Visualizes the overall performance of multiple runs in the standard diagrams for compressed sensing recovery.

    Parameters
    ----------
    errors : ndarray, 2D
        Relative or absolute error matrix.
    x : ndarray, 1D
        Basically xticklabels.
    y : ndarray, 1D
        Basically yticklabels.
    x_name : str
        Plot xlabel.
    y_name : str
        Plot ylabel.
    file_name : str
        Name and path of figure output.
    format : str
        Format of figure.
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


'''

















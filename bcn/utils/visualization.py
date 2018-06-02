"""Plotting functions.
"""
from __future__ import division, absolute_import

import numpy as np

import pylab as pl
import seaborn as sb

sb.set(font_scale=0.7, style='white')


def recovery_performance(mixed, cost_func, true_bias, estimated_signal, true_signal, estimated_bias):
    """
    Print recovery performance statistics.

    Parameters
    ----------
    mixed : numpy.ndarray, shape=(n_samples, n_features)
        Corrupted signal to be cleaned.
    cost_func : func
        Objective function for evaluation of current bias matrix estimate.
    true_bias : numpy.ndarray, shape=(n_samples, n_features)
        True bias matrix.
    estimated_signal : numpy.ndarray, shape=(n_samples, n_features)
        Estimated signal matrix.
    true_signal : numpy.ndarray, shape=(n_samples, n_features)
        True signal matrix.
    estimated_bias: numpy.ndarray, shape=(n_samples, n_features)
        Estimated bias matrix.

    Returns
    -------
    d : dict
        Performance metrics.
    """
    error_cost_func_true_bias = cost_func(true_bias)
    error_cost_func_estimated_bias = cost_func(estimated_bias)
    d['Error cost function (true bias)'] = error_cost_func_true_bias
    d['Error cost function (estimated bias)'] = error_cost_func_estimated_bias
    divisor = np.sum(~np.isnan(mixed))
    d['Number of valid values in corrupted signal'] = divisor
    mean_absolute_error_true_signal = np.nansum(
        np.absolute(true_signal - (mixed - true_bias))) / divisor
    mean_absolute_error_estimated_signal = np.nansum(
        np.absolute(true_signal - estimated_signal)) / divisor
    d['Mean absolute error (true_signal)'] = mean_absolute_error_true_signal
    d['Mean absolute error (estimated_signal)'] = mean_absolute_error_estimated_signal
    mean_absolute_error_zeros = np.nansum(
        np.absolute(true_signal - mixed)) / divisor
    d['Mean absolute error (zeros)'] = mean_absolute_error_zeros
    ratio_estimated_signal_to_zeros = mean_absolute_error_estimated_signal / \
        mean_absolute_error_zeros
    d['Ratio mean absolute error (estimated signal / zeros)'] = ratio_estimated_signal_to_zeros
    return d

    
def show_absolute(signal, kind, unshuffled=False, unshuffle=False, map_backward=None, vmin=-4, vmax=4):
    """
    Plot the absolute values of the given signal matrix.

    Parameters
    ----------
    signal : numpy.ndarray, shape=(n_samples, n_features)
        True signal matrix.
    kind : str, values=('Bias', 'Signal')
        Type of absolute value matrix to be shown (used as annotation on plot).
    unshuffled : bool
        If the input data is unshuffled.
    unshuffle : bool
        If to unshuffle the input data.
    map_backward : dict, value=('feature', 'sample'), values=dict
        Map from new annotation to old annotion.
    vmin : int
        Minimum absolute value on color scale.
    vmax : int
        Maximum absolute value on color scale.
    """
    cmap = sb.diverging_palette(
        250, 15, s=75, l=40, as_cmap=True, center="dark")
    indices_x = np.arange(signal.shape[0], dtype=int)
    indices_y = np.arange(signal.shape[1], dtype=int)
    fig = pl.figure(figsize=(7 * (signal.shape[1] / signal.shape[0]), 7))
    ax = fig.add_subplot(111)
    if unshuffle:
        ax.set_title('{} (unshuffled)'.format(kind))
        indices_x = np.asarray([map_backward['sample'][i] for i in indices_x])
        indices_y = np.asarray([map_backward['feature'][i] for i in indices_y])
        signal = signal[indices_x]
        signal = signal[:, indices_y]
    if unshuffled:
        ax.set_title('{} (unshuffled)'.format(kind))
        indices_x = np.asarray([map_backward['sample'][i] for i in indices_x])
        indices_y = np.asarray([map_backward['feature'][i] for i in indices_y])
    else:
        ax.set_title('{}'.format(kind))
    ax_seaborn = sb.heatmap(signal, vmin=vmin, vmax=vmax, cmap=cmap, ax=ax, cbar_kws={
                            'shrink': 0.5}, xticklabels=indices_y, yticklabels=indices_x)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xlabel('Features')
    ax.set_ylabel('Samples')


def show_dependences(signal, pairs, space, n_pairs=5, n_points=50):
    """
    Plot the signal dependences for a subset of correlated pairs.

    Parameters
    ----------
    signal : numpy.ndarray, shape=(n_samples, n_features)
        True signal matrix.
    pairs : dict, keys=('feature', 'sample'), values=numpy.ndarray, shape=(n, 2)
        Correlated pair indices.
    space : str, values=('feature', 'sample')
        Feature or sample space.
    n_pairs : int
        Number of correlated pairs to show.
    n_points : int
        Number of data point to show.
    """
    cmap = sb.diverging_palette(250, 15, s=75, l=40, n=10, center="dark")
    if space == 'feature':
        shape = signal.T.shape
    if space == 'sample':
        shape = signal.shape
    pairs = pairs[space]
    for n, i in enumerate(np.random.choice(np.arange(len(pairs), dtype=int), min(n_pairs, len(pairs)), replace=False)):
        j = np.random.choice(np.arange(shape[1], dtype=int), min(
            n_points, shape[1]), replace=False)
        if space == 'sample':
            grid = sb.jointplot(signal[np.atleast_2d(pairs[i][1]), j], signal[np.atleast_2d(
                pairs[i][0]), j], ylim=(-4, 4), xlim=(-4, 4), alpha=0.6, size=5, stat_func=None, color='black')
            grid.set_axis_labels('Sample {}'.format(
                pairs[i][1]), 'Sample {}'.format(pairs[i][0]))
        if space == 'feature':
            grid = sb.jointplot(signal[j[:, None], pairs[i][1]], signal[j[:, None], pairs[i][0]], ylim=(
                -4, 4), xlim=(-4, 4), alpha=0.6, size=5, stat_func=None, color='black')
            grid.set_axis_labels('Feature {}'.format(
                pairs[i][1]), 'Feature {}'.format(pairs[i][0]))
        pl.setp(grid.ax_marg_y.patches, color=cmap[2])
        pl.setp(grid.ax_marg_x.patches, color=cmap[-2])


def show_recovery(mixed, guess_X, true_signal, estimated_signal, true_pairs, estimated_pairs, true_stds, estimated_stds, true_directions, estimated_directions, n_pairs=5, n_points=50):
    """
    Plot the signal dependences for a subset of correlated pairs overlayed with the estimated and true values. 

    Parameters
    ----------
    mixed : numpy.ndarray, shape=(n_samples, n_features)
        Corrupted signal to be cleaned.
    guess_X : numpy.ndarray, shape=(n_samples, n_features)
        Initial guess used for the final solution.
    true_signal : numpy.ndarray, shape=(n_samples, n_features)
        True signal matrix.
    estimated_signal : numpy.ndarray, shape=(n_samples, n_features)
        Estimated signal matrix.
    true_pairs : numpy.ndarray, shape=(n, 2)
        True correlated pairs.
    estimated_pairs : numpy.ndarray, shape=(n, 2)
        Estimated correlated pairs.
    true_stds : numpy.ndarray, len=n
        True standard deviations.
    estimated_stds : numpy.ndarray, len=n
        Estimated standard deviations.
    true_directions : numpy.ndarray, len=n
        True directions.
    estimated_directions : numpy.ndarray, len=n
        Estimated directions.
    n_pairs : int
        Number of correlated pairs to show.
    n_points : int
        Number of data point to show.
    """
    def pair_index(pairs, pair):
        index = np.where(np.all(pairs == pair, axis=1))
        try:
            index = index[0][0]
        except IndexError:
            index = None
        return index

    fig = pl.figure(figsize=(5, 5 * n_pairs))
    pairs = np.vstack([true_pairs, estimated_pairs])
    np.random.shuffle(pairs)

    for i in xrange(n_pairs):
        ax = fig.add_subplot(n_pairs, 1, i + 1)
        for j in xrange(n_points):
            ax.plot(mixed[pairs[i][1]][j], mixed[pairs[i][0]]
                    [j], 'o', color='red', alpha=0.6)
            ax.plot(estimated_signal[pairs[i][1]][j],
                    estimated_signal[pairs[i][0]][j], 'D', color='blue', alpha=0.6)
            ax.plot(guess_X[pairs[i][1]][j], guess_X[pairs[i][0]]
                    [j], 'o', color='brown', alpha=0.6)
            ax.plot(true_signal[pairs[i][1]][j], true_signal[pairs[i]
                                                             [0]][j], 'o', color='green', alpha=0.6)
            ax.plot([true_signal[pairs[i][1]][j], mixed[pairs[i][1]][j]], [
                    true_signal[pairs[i][0]][j], mixed[pairs[i][0]][j]], '-', color='red', alpha=0.6)
            ax.plot([true_signal[pairs[i][1]][j], estimated_signal[pairs[i][1]][j]], [
                    true_signal[pairs[i][0]][j], estimated_signal[pairs[i][0]][j]], '-', color='blue', alpha=0.6)
        if pairs[i] in true_pairs:
            direction = true_directions[pair_index(true_pairs, pairs[i])]
            std_b, std_a = true_stds[pair_index(true_pairs, pairs[i])]
            std_b = std_b * -1 * direction
            m = -std_b / float(std_a)
            ax.plot(list(ax.get_xlim()), [
                    m * p + 0.0 for p in ax.get_xlim()], '-', color='orange', alpha=0.6)
        if pairs[i] in estimated_pairs:
            direction = estimated_directions[pair_index(
                estimated_pairs, pairs[i])]
            std_b, std_a = estimated_stds[pair_index(
                estimated_pairs, pairs[i])]
            std_b = std_b * -1 * direction
            m = -std_b / float(std_a)
            ax.plot(list(ax.get_xlim()), [
                    m * p + 0.0 for p in ax.get_xlim()], '--', color='black', alpha=0.6)
        sb.despine()
        ax.set_xlabel('Sample {}'.format(pairs[i][1]))
        ax.set_ylabel('Sample {}'.format(pairs[i][0]))
        ax.set_ylim(-4, 4)
        ax.set_xlim(-4, 4)


def show_independences(signal, pairs, space, n_pairs=5, n_points=50):
    """
    Plot the signal dependences for a subset of uncorrelated pairs.

    Parameters
    ----------
    signal : numpy.ndarray, shape=(n_samples, n_features)
        True signal matrix.
    pairs : dict, keys=('feature', 'sample'), values=numpy.ndarray, shape=(n, 2)
        Correlated pair indices.
    space : str, values=('feature', 'sample')
        Feature or sample space.
    n_pairs : int
        Number of correlated pairs to show.
    n_points : int
        Number of data point to show.
    """
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
    identical = set([(i, i) for i in range(shape[0])])
    non_pairs = all_pairs - true_pairs - identical
    pairs = {space: np.asarray(list(non_pairs), dtype=int)}
    show_dependences(signal, pairs, space, n_pairs=n_pairs, n_points=n_points)


def show_dependence_structure(correlations, space, unshuffled=False, map_backward=None):
    """
    Plot a correlation matrix.

    Parameters
    ----------
    correlations : dict, keys=('feature', 'sample'), values=numpy.ndarray, shape=(n_samples, n_samples) or (n_features, n_features)
        Correlation matrix.
    space : str, values=('feature', 'sample')
        Feature or sample space.
    unshuffled : bool
        If the input data is unshuffled.
    map_backward : dict, value=('feature', 'sample'), values=dict
        Map from new annotation to old annotion.
    """
    cmap = sb.diverging_palette(
        250, 15, s=75, l=40, as_cmap=True, center="dark")
    indices = np.arange(correlations[space].shape[0], dtype=int)
    if space == 'feature':
        size = 7 * (correlations['feature'].shape[0] /
                    correlations['sample'].shape[0])
    if space == 'sample':
        size = 7
    fig = pl.figure(figsize=(size, size))
    ax = fig.add_subplot(111)
    if unshuffled:
        ax.set_title('Correlations (unshuffled)')
        indices = np.asarray([map_backward[space][i] for i in indices])
    else:
        ax.set_title('Correlations')
    sb.heatmap(correlations[space], cmap=cmap, vmin=-1, vmax=1, square=True,
               ax=ax, cbar_kws={'shrink': 0.5}, xticklabels=indices, yticklabels=indices)
    if space == 'feature':
        ax.set_xlabel('Features')
        ax.set_ylabel('Features')
    if space == 'sample':
        ax.set_xlabel('Samples')
        ax.set_ylabel('Samples')


def show_threshold(correlations, threshold, space):
    """
    Plot the number of estimated pairs at a particular correlation threshold.

    Parameters
    ----------
    correlations : dict, keys=('feature', 'sample'), values=numpy.ndarray, shape=(n_samples, n_samples) or (n_features, n_features)
        Correlation matrix.
    threshold : float
        Correlation threshold at which to cut-off (used to draw vertical line)
    space : str, values=('feature', 'sample')
        Feature or sample space.
    """
    cmap = sb.diverging_palette(250, 15, s=75, l=40, n=10, center="dark")
    fig = pl.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    trimmed = np.trim_zeros(
        np.sort(np.tril(np.absolute(correlations[space]), -1).ravel()))
    ax.set_xlabel('Threshold')
    ax.set_ylabel('# pairs')
    x = trimmed
    y = np.arange(1, len(trimmed) + 1)
    ax.plot(x, y[::-1], '-', alpha=0.8, color='black')
    ax.axvline(threshold, min(x), max(x), linestyle='dashed', color=cmap[2])
    sb.despine()

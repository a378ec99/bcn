"""Classification based evaluation of bias recovery on high-throughput data.

Notes
-----
Defines a class that can be used to evaluate bias recovery on high-throughput data via classification.
"""
from __future__ import division, absolute_import


__all__ = ['visualize_threshold', 'reduce_dimensions', 'performance_evaluation', 'bias_correction']

import sys # WARNING remove in final version
sys.path.append('/home/sohse/projects/PUBLICATION/GITssh/bcn')

import cPickle

import numpy as np
import pylab as pl
import scipy.stats as spst
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score

import seaborn.apionly as sb
from scipy import interp

from bcn.data import DataBlind
from bcn.bias import guess_func
from bcn.solvers import ConjugateGradientSolver
from bcn.linear_operators import LinearOperatorCustom, possible_measurement_range
from bcn.cost import Cost
from bcn.utils.visualization import visualize_dependences, visualize_correlations

def visualize_threshold(X):
    '''
    Visualizes the number of pairs for different correlation cut-off thresholds.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        The dataset that the correlation matrix is done for (sample space only ?).

    Returns a plot, with the name ``threshold_tuning''.
    '''
    mixed_masked = np.ma.masked_less_equal(X, 0.0)
    correlations = np.ma.corrcoef(mixed_masked)
    out = np.tril(np.absolute(correlations), -1).ravel()
    out_sorted = np.sort(out)
    trimmed = np.trim_zeros(out_sorted)
    
    fig = pl.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_title('Help for setting the threshold')
    ax.set_xlabel('threshold')
    ax.set_ylabel('n pairs')
    ax.plot(trimmed, np.arange(1, len(trimmed) + 1)[::-1], '-', color='red', alpha=0.8)
    fig.savefig('threshold_tuning')


def reduce_dimensions(X, model='tSNE'):
    '''
    Does a low-dimensional embedding to 2D.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        The input dataset to the PCA or tSNE algorithm.
    model = str, {'PCA', 'tSNE'}
        The algorithm to be used for the embedding.

    Returns
    -------
    X : nd_array, (n_samples, 2)
        A reduced dataset to two components.
    '''
    
    # NOTE Correct for nans.
    X = np.nan_to_num(X)
    if model == 'tSNE':
        tsne = TSNE(n_components=2)
        X = tsne.fit_transform(X)
    if model == 'PCA':
        pca = PCA(n_components=2, svd_solver='randomized')
        X = pca.fit_transform(X)
    return X


def performance_evaluation(X, y, batch, model='naiveBayes', file_name='test'):
    '''
    Evaluation the performance of a particular dataset with a specific calssification algorithm.

    Parameters
    ----------
    reduced : ndarray, (n_samples, 2)
        Dataset with samples as rows and reduced features as columns.
    y : list (of str)
        Labels of the samples, e.g. tissue types.
    model : str {'RF', 'naiveBayes', 'linearSVM'}
        The classification algorithm to be used. At the moment only random forrests (RF).
    file_name : str
        String to be added to the file name describing the dataset used.
    Results
    -------
    best : str ?
        Performance of the classification evaluated with the specific metric.
    '''

    if model == 'RF':
        classifier = RandomForestClassifier()
    if model == 'naiveBayes':
        classifier = GaussianNB()
    if model == 'linearSVM':
        classifier = SVC(kernel='linear', probability=True)

    classifier.fit(X, y)
    # NOTE Create a mesh (http://scikit-learn.org/0.16/auto_examples/svm/plot_iris.html).
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    fig = pl.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=pl.cm.jet, alpha=0.8) # Paired
    indices_1 = np.argwhere(batch == 'GPL1261')
    indices_2 = np.argwhere(batch == 'GPL570')
    ax.scatter(X[indices_1, 0], X[indices_1, 1], c=y[indices_1], cmap=pl.cm.jet, marker='o') # Paired
    ax.scatter(X[indices_2, 0], X[indices_2, 1], c=y[indices_2], cmap=pl.cm.jet, marker='s')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('{} Multi-Class Classification on 2 PCs'.format(model))
    fig.savefig('../../out/classification_' + file_name)
    
    scores = cross_val_score(classifier, X, y, cv=5)
    print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))


def bias_correction(X, ranks, thresholds, n_restarts=3):
    '''
    Blind bias correction with BCN testing different ranks and thresholds.
    
    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        Matrix consiting of a mixture of signal and bias.
    ranks : sequence of int
        Ranks to be tested for the low rank matrix modelling the bias to be recovered.
    thresholds : float
        Threshold used to select correlated pairs (both sample and feature space).
    n_restarts : int (default = 3)
        The number of restarts of the conjugate gradient solver.

    Returns
    -------
    corrected : ndarray (n_samples, n_features)
        Bias corrected matrix (ideally pure signal).
    '''
    errors = []
    results = []
    
    for t, threshold in enumerate(thresholds):
        for r, rank in enumerate(ranks):
            print 'threshold', threshold, '-', t + 1, 'of', len(thresholds)
            print 'rank', rank, '-', r + 1, 'of', len(ranks)
            blind = DataBlind(X, rank, correlation_threshold=threshold) # 0.85
            blind.estimate()
            #visualize_correlations(blind, file_name='correlation_thresholds', truth_available=False)
            missing_fraction = np.isnan(X).sum() / X.size
            print possible_measurement_range(X.shape, missing_fraction)
            n_measurements = 500 # len(blind.d['sample']['estimated_pairs']) * (blind.d['sample']['shape'][1] - missing_fraction * blind.d['sample']['shape'][1]) + len(blind.d['feature']['estimated_pairs']) * (blind.d['sample']['shape'][0] - missing_fraction * blind.d['sample']['shape'][0])
            print n_measurements
            
            # NOTE Construction of the measurement operator and measurements from the data.
            operator = LinearOperatorCustom(blind, int(n_measurements)).generate()
            A = operator['A']
            y = operator['y']
            cost = Cost(A, y)

            # NOTE Setup and run of the recovery with the standard solver.
            solver = ConjugateGradientSolver(cost.cost_func, guess_func, blind, rank, n_restarts, verbosity=1)
            
            result = solver.recover()
            results.append(result)
            error = result.d['sample']['final_cost']
            errors.append(error)
            
    index = np.argmin(errors)
    corrected = results[index].d['sample']['estimated_signal']
    visualize_dependences(results[index], file_name='../../out/recovery_blind_rank_{}'.format(rank), truth_available=False, estimate_available=True, recovery_available=True)

    return corrected


if __name__ == '__main__':

    y = cPickle.load(open('../../data/y.pickle', 'r'))
    scan_X = cPickle.load(open('../../data/X.pickle', 'r'))
    batches = cPickle.load(open('../../data/batches.pickle', 'r'))
    gsms = cPickle.load(open('../../data/gsms.pickle', 'r'))
    ensembls = cPickle.load(open('../../data/ensembls.pickle', 'r'))
    rank = 8
    threshold = 0.92 # NOTE Same as in preprocessing.
    mixed = scan_X

    #print 'Threshold visualization...'
    #visualize_threshold(mixed)
    
    print 'mixed.shape', mixed.shape

    le = LabelEncoder()
    y = le.fit_transform(y)

    print 'Low dimensional visualization...'
    reduced = reduce_dimensions(mixed, model='tSNE')

    print 'Performance evaluation no correction...'
    performance_evaluation(reduced, y, batches, model='naiveBayes', file_name='before_correction_{}'.format(rank))

    print 'Bias correction...'
    corrected = bias_correction(mixed, [rank], [threshold])

    print 'Low dimensional visualization...'
    reduced = reduce_dimensions(corrected, model='tSNE')

    print 'Performance evaluation correction...'
    performance_evaluation(reduced, y, batches, model='naiveBayes', file_name='after_correction_rank_{}'.format(rank))

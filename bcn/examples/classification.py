"""Classification based evaluation of bias recovery on high-throughput data.

Notes
-----
Defines a class that can be used to evaluate bias recovery on high-throughput data via classification.
"""
from __future__ import division, absolute_import


__all__ = ['visualize_threshold', 'visualize_2D', 'performance_evaluation']

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


def reduce_dimensions(X, y, model='tSNE'):
    '''
    Visualizes the dataset with a low-dimensional embedding given batch and tissue annotations and outputs this.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        The input dataset to the PCA or tSNE algorithm.
    y : sequence (of str)
        The labels with respect to tissue type (not digitized).
    batch = sequence (of str)
        The labels with respect to batch, e.g. platform type.
    model = str, {'PCA', 'tSNE'}
        The algorithm to be used for the embedding.
        
    Returns a plot, with the name ``pca''.

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

    """
    fig = pl.figure()
    ax = fig.add_subplot(111)
    
    for n, (sample, label_tissue, label_batch) in enumerate(zip(X, y, batch)):
        if label_batch == 'GPL1261':
            marker = 'D'
        if label_batch == 'GPL570':
            marker = 'o'
        if n == 1 or n == (len(y) - 1):
            label = label_batch
        else:
            label = None
        color = pl.cm.Paired(label_tissue) 
        ax.plot([sample[0]], [sample[1]], marker=marker, alpha=0.8, color=color, label=label)
        
    if model == 'PCA':
        ax.set_title('PCA')
        ax.set_xlabel('PC 1 ({}% variance)'.format(int(100 * pca.explained_variance_ratio_[0])))
        ax.set_ylabel('PC 2 ({}% variance)'.format(int(100 * pca.explained_variance_ratio_[1])))
    if model == 'tSNE':
        ax.set_title('tSNE')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
    ax.legend(loc='best', numpoints=1)
    fig.savefig('visualization3')
    """
    return X


def performance_evaluation(X, y, batch, model='naiveBayes'):
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
    fig.savefig('classification3')
    
    scores = cross_val_score(classifier, X, y, cv=5)
    print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))


def bias_correction(X, ranks, thresholds, n_restarts=3): # 10
    '''
    '''
    errors = []
    results = []
    
    for threshold in thresholds:
        for rank in ranks:
            print threshold, 'of', len(thresholds)
            print rank, 'of', len(ranks)
            blind = DataBlind(X, rank, correlation_threshold=threshold) # 0.85
            blind.estimate()
            visualize_correlations(blind, file_name='correlation_thresholds', truth_available=False)
            missing_fraction = np.isnan(X).sum() / X.size
            possible_measurement_range(X.shape, missing_fraction)
            n_measurements = len(blind.d['sample']['estimated_pairs']) * (blind.d['sample']['shape'][1] - missing_fraction * blind.d['sample']['shape'][1]) + len(blind.d['feature']['estimated_pairs']) * (blind.d['sample']['shape'][0] - missing_fraction * blind.d['sample']['shape'][0])

            print n_measurements
            
            # NOTE Construction of the measurement operator and measurements from the data.
            operator = LinearOperatorCustom(blind, int(n_measurements)).generate()
            A = operator['A']
            y = operator['y']
            cost = Cost(A, y)

            # NOTE Setup and run of the recovery with the standard solver.
            solver = ConjugateGradientSolver(cost.cost_func, guess_func, blind, rank, n_restarts, verbosity=0)
            
            result = solver.recover()
            results.append(result)
            error = result.d['sample']['final_cost']
            errors.append(error)
            
    index = np.argmin(errors)
    corrected = results[index].d[self.space]['estimated_signal']
    return corrected


if __name__ == '__main__':

    ys = cPickle.load(open('../../data/ys.pickle', 'r'))
    scan_Xs = cPickle.load(open('../../data/Xs.pickle', 'r'))
    batches = cPickle.load(open('../../data/batches.pickle', 'r'))
    #gsms = cPickle.load(open('../../data/gsms.pickle', 'r'))
    #ensembls = cPickle.load(open('../../data/ensembls.pickle', 'r'))

    ranks = np.arange(1, 10)
    thresholds = np.linspace(0.65, 0.95, 5)
    mixed = scan_Xs[0] # scan_corrected_Xs
    '''
    scan_corrected_Xs = []
    for scan_X in scan_Xs:
        scan_corrected_Xs.append(bias_correction(scan_X, ranks, thresholds))
    scan_corrected_Xs[0]
    '''
    y = ys[0]
    batch = batches[0]
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    print 'Low dimensional visualization...'
    reduced = reduce_dimensions(mixed, y, model='tSNE') # model='PCA'

    #print 'Threshold visualization...'
    #visualize_threshold(mixed)

    print 'Performance evaluation...'
    performance_evaluation(reduced, y, batch, model='naiveBayes') # model='RF'
    
    
    corrected = bias_correction(mixed, ranks, thresholds) # [:20, :10]


 











"""



    # NOTE Get an overview of the data, then set these parameters and correct.

    # TODO Plot the correlation structure at different thresholds.

    # TODO Plot the absolute look of the data.

    # TODO show with real batch annotation in PCA (visualize full)! Get rid of d and use what is given Xs, ys, etc.

    # TODO docstrings

    # TODO Plot some dependencies.

    # TODO Could also plot unannotated, since using that for bias recovery anyhow!



    

def select_subset(X, index_dict, annotation, max_size=100):
    temp = []
    temp_annotation = []
    temp_annotation_2 = []
    for key in sorted(index_dict):
        for n, index in enumerate(index_dict[key]):
            if n == max_size:
                break
            temp.append(X[index])
            temp_annotation.append(annotation[index])
            temp_annotation_2.append(key)
    X = np.vstack(temp)
    annotation = np.vstack(temp_annotation).ravel()
    annotation_2 = np.vstack(temp_annotation_2).ravel()
    return X, annotation, annotation_2



def select_subset_both_dimensions(X, index_dict_x, index_dict_y, annotation_x, annotation_y):
    X, annotation_x, annotation_x_2 = select_subset(X, index_dict_x, annotation_x)
    Xt, annotation_y, annotation_y_2 = select_subset(X.T, index_dict_y, annotation_y)
    X = Xt.T
    return X, annotation_x, annotation_x_2, annotation_y, annotation_y_2





    
visualize_threshold(X_scan_subset, 'SCAN_sample')
visualize_threshold(X_scan_subset.T, 'SCAN_feature') # WARNING doesn't work


visualize_full(scan_X, d_samples, 'sample', 'SCAN_centered', batches_scatter, format='png')

batches_scatter_feature = {'o': range(len(scan_X.T)), '+': []}

visualize_full(scan_X.T, d_features, 'feature', 'SCAN_centered', batches_scatter_feature, format='png')


c = Classification()

X_scan_subset, X_scan_subset_annotation_x, X_scan_subset_annotation_x_2, X_scan_subset_annotation_y, X_scan_subset_annotation_y_2 = select_subset_both_dimensions(scan_X, d_samples, d_features, samples, features)

X_scan_subset_annotation_x_numeric_2 = label_encoder(X_scan_subset_annotation_x_2, tissues)
X_scan_subset_annotation_y_numeric_2 = label_encoder(X_scan_subset_annotation_y_2, GO_terms)

print 'SCAN', 'sample', 'accuracy', c.best_score(X_scan_subset, X_scan_subset_annotation_x_numeric_2, metric='accuracy')
print 'SCAN', 'feature', 'accuracy', c.best_score(X_scan_subset.T, X_scan_subset_annotation_y_numeric_2, metric='accuracy')
print 'SCAN', 'sample', 'F1', c.best_score(X_scan_subset, X_scan_subset_annotation_x_numeric_2)
print 'SCAN', 'feature', 'F1', c.best_score(X_scan_subset.T, X_scan_subset_annotation_y_numeric_2)




    
for rank in range(1, 5):
    print rank
    data = DataBlind(X_scan_subset, rank, noise_amplitude=1.0) #NOTE # noise_amplitude is just for guess!
    data.estimate()
    n_measurements = 1000 # TODO max and min non-redundant? Note that random sampling will always yield some ducpicates.
    operator = LinearOperatorCustom(data.d, n_measurements).generate()
    A = operator['A']
    y = operator['y']
    cost = Cost(A, y)
    solver = ConjugateGradientSolver(cost.cost_function, data, rank, 10, verbosity=0)
    data.d = solver.recover()
    #visualize_dependences(data.d, file_name='test_dependences_estimated_real_data_' + str(rank))
    X_scan_subset_clean = data.d['sample']['mixed'] - data.d['sample']['estimated_bias']

    print X_scan_subset_clean.shape

    print 'SCAN', 'sample', 'accuracy', c.best_score(X_scan_subset_clean, X_scan_subset_annotation_x_numeric_2, metric='accuracy')
    #print 'SCAN', 'feature', 'accuracy', c.best_score(X_scan_subset_clean.T, X_scan_subset_annotation_y_numeric_2, metric='accuracy')
    print 'SCAN', 'sample', 'F1', c.best_score(X_scan_subset_clean, X_scan_subset_annotation_x_numeric_2)
    #print 'SCAN', 'feature', 'F1', c.best_score(X_scan_subset_clean.T, X_scan_subset_annotation_y_numeric_2)

    # TODO Stop doing the subset thing. scan_X needs to be the cleaned version.

    #visualize_full(scan_X, d_samples, 'sample', 'SCAN_centered_' + str(rank), batches_scatter, format='png')

# TODO increase size and only do with those samples / features that have high correlation... could just do it with same samples but features with high correlation!
"""





    
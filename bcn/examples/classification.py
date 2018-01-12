"""Classification based evaluation of bias recovery on high-throughput data.

Notes
-----
Defines a class that can be used to evaluate bias recovery on high-throughput data via classification.
"""
from __future__ import division, absolute_import


__all__ = ['Classification']

import sys # WARNING remove in final version
sys.path.append('/home/sohse/projects/PUBLICATION/GITssh/bcn')

import numpy as np
import pylab as pl
import scipy.stats as spst
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import seaborn.apionly as sb

from bcn.data import DataBlind
from bcn.solvers import ConjugateGradientSolver
from bcn.linear_operators import LinearOperatorCustom
from bcn.cost import Cost
from bcn.utils.visualization import visualize_dependences


def label_encoder(annotation, classes):
    out = []
    for name in annotation:
        index = np.argwhere(np.asarray(classes) == name).ravel()
        out.append(index)
    return np.asarray(out).ravel()

    
class Classification(object):
    '''
    '''
    def __init__(self, model):
        if model == 'RF':
            self.clf = RandomForestClassifier()

    def best_score(self, X, y, metric='f1_micro'):
        X = np.nan_to_num(X)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        param_dist = {'max_depth': [3, None],
                    'max_features': spst.randint(1, 10),
                    'min_samples_split': spst.randint(2, 10),
                    'min_samples_leaf': spst.randint(1, 10),
                    'bootstrap': [True, False],
                    'criterion': ['gini', 'entropy'],
                    'n_estimators': spst.randint(1, 40)}
        random_search = RandomizedSearchCV(self.clf, param_distributions=param_dist, scoring=metric, n_jobs=-1, n_iter=200)
        random_search.fit(X, y)
        return random_search.best_score_


def visualize_threshold(X, name):
    '''
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
    fig.savefig('threhold_' + str(name))


def visualize_full(X, d, space, normalization, batches_scatter, format='eps'):
    '''
    '''
    # NOTE Correct for nans
    X = np.nan_to_num(X)

    pca = PCA(n_components=2, svd_solver='randomized')
    X_reduced = pca.fit_transform(X)

    # Plot classifiert in PCA space
    fig = pl.figure()
    ax = fig.add_subplot(111)
    
    ax.scatter(X_reduced[batches_scatter['+'], 0], X_reduced[batches_scatter['+'], 1], marker='D', alpha=0.05, s=40, c='gray', edgecolors='none', zorder=1, label=None)
    ax.scatter(X_reduced[batches_scatter['o'], 0], X_reduced[batches_scatter['o'], 1], marker='o', alpha=0.05, s=40, c='gray', edgecolors='none', zorder=1, label=None)
    
    
    label_names = sorted(d.keys())
    N = len(label_names)

    colors = sb.color_palette('plasma', n_colors=N) # 'viridis' # colors = ['orange', 'gold', 'green', 'blue'] #
    sizes = np.linspace(30, 100, N).tolist()[::-1] # 60 - 200 #  [200, 160, 120, 80]

    for label, size, color in zip(label_names, sizes, colors):
        indices = np.asarray(d[label])

        subset_indices_batch_1 = np.asarray([index for index in indices if index in batches_scatter['+']])

        #print subset_indices_batch_1.shape
        #print subset_indices_batch_1
        #print X_reduced.shape

        if subset_indices_batch_1.size != 0:
            ax.scatter(X_reduced[subset_indices_batch_1, 0], X_reduced[subset_indices_batch_1, 1], s=size, marker='D', alpha=0.3, edgecolors=color, facecolors='none', zorder=5, linewidths=2, label=label)

        subset_indices_batch_2 = np.asarray([index for index in indices if index in batches_scatter['o']])
        if subset_indices_batch_2.size != 0:
            ax.scatter(X_reduced[subset_indices_batch_2, 0], X_reduced[subset_indices_batch_2, 1], s=size, marker='o', alpha=0.3, edgecolors=color, facecolors='none', zorder=5, linewidths=2, label=label)

    ax.set_title(normalization + ' ' + space)

    ax.set_xlabel('PC 1 ({}% variance)'.format(int(100 * pca.explained_variance_ratio_[0])))
    ax.set_ylabel('PC 2 ({}% variance)'.format(int(100 * pca.explained_variance_ratio_[1])))
    ax.legend(loc='best', numpoints=1, shadow=False)

    fig.savefig('test_{}_{}.{}'.format(space, normalization, format))




# NOTE Seperate the data set creation and the classification. Dataset should just be imported at the begnnning. then also don't need all the mapping crap!



def overview(mixed)
    '''
    '''
    #TODO Plot the correlation structure at different thresholds.

    #TODO Plot the absolute look of the data.

    #TODO Plot a PCA of the data.
    
    #TODO Plot some dependencies.

    pass


def bias_correction(mixed, ranks, thresholds):
    '''
    '''
    for threshold in thresholds:
        for rank in ranks:
            pass

            # mixed
            
    # TODO Select best of them based on?
    
    return corrected


def performance_evaluation(mixed, mixed_batch_labels):
    '''
    '''
    c = Classification('RF')
    return c.best_score(mixed, mixed_batch_labels, metric='accuracy')


    


    


    

if __name__ == '__main__':

    #raw =
    #scan =
    #mixed_batch_labels =

    for mixed in [raw, scan]:
        overview(mixed)


    #NOTE Then set these parameters and correct.

    ranks = np.arange(1, 10)
    thresholds = np.linspace(0.65,0.95, 5)

    scan_corrected = bias_correction(scan, ranks, thresholds)

    #NOTE test different settings for performance evaluation.

    for mixed in [raw, scan, scan_corrected]:
        performance_evaluation(mixed, mixed_batch_labels)





 











"""


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





    
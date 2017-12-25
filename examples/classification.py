"""Classification on high-throughput data.

Notes
-----
Defines two classes that can be used to test bias recovery on high-throughput data.
"""
from __future__ import division, absolute_import


#__all__ = ['Classification', 'DataReal']

import numpy as np
import pylab as pl

import sys
sys.path.append('/home/sohse/projects/PUBLICATION/annotation')
from database import Annotation
from os import path
import h5py

import scipy.stats as spst

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

import seaborn.apionly as sb
import re

import cPickle

from bcn.data import DataBlind
from bcn.solvers import ConjugateGradientSolver
from bcn.linear_operators import LinearOperatorBlind
from bcn.cost import Cost
from bcn.utils.visualization import visualize_dependences


class Classification(object):

    def __init__(self):
        self.clf = RandomForestClassifier() # NOTE build a classifier

    def best_score(self, X, y, metric='f1_micro'):
        X = np.nan_to_num(X)
        # NOTE specify parameters and distributions to sample from
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

        
def clean(names):
    '''
    Clean up a list of file names or features by removing for example "_at" or ".CEL" at the end.
    '''
    cleaned  = [re.split(r'[_|.]', name)[0] for name in names]
    return cleaned

    
def plot_threshold_to_pairs(X, name):
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

    
def label_encoder(annotation, classes):
    out = []
    for name in annotation:
        index = np.argwhere(np.asarray(classes) == name).ravel()
        out.append(index)
    return np.asarray(out).ravel()

    
def select_subset_both_dimensions(X, index_dict_x, index_dict_y, annotation_x, annotation_y):
    X, annotation_x, annotation_x_2 = select_subset(X, index_dict_x, annotation_x)
    Xt, annotation_y, annotation_y_2 = select_subset(X.T, index_dict_y, annotation_y)
    X = Xt.T
    return X, annotation_x, annotation_x_2, annotation_y, annotation_y_2


def map_to_orthologs(organism, features):

    a = dict(np.genfromtxt('data/GPL570_GPL1261_ortho.txt', delimiter=',', dtype=str))

    temp = []
    for feature in features:
        try:
            mouse_name = a[feature]
            temp.append(mouse_name)
        except KeyError:
            temp.append('')
    orthologs = np.asarray(temp)
    return orthologs
    

def align_features(a, b): # TODO might have to do cross-species with mapping_8-5-17.txt
    intersection = set(a) & set(b)
    #print difference = set(a) ^ set(b)
    a_indices = []
    b_indices = []
    for feature in intersection:
        if 'AFFX' not in feature:
            a_index = np.argwhere(a == feature)
            b_index = np.argwhere(b == feature)
            if len(a_index) == 1 and len(b_index) == 1:
                a_indices.append(a_index[0])
                b_indices.append(b_index[0])
            else:
                print 'Warning'
    return np.hstack(a_indices), np.hstack(b_indices)


def compute_batch_effects_features(GC, features, pickle_name='data/mapping_features_batch.pickle'):
    '''
    NOTE Map the 'scan_features' to ENSEMBL human and GC, length, location (?). Return as 'batch_mapping_d-a-t-e.txt'.

    with open('features.txt', 'w') as f:
        for id_ in features:
            f.write(id_ + '\n')
    '''
    if path.exists(pickle_name):
        d_GC = cPickle.load(open(pickle_name, 'r'))
    else:
        feature_batch = dict(zip(np.genfromtxt('data/mapping_batch_14-5-17.csv', delimiter=',', usecols=[0], skip_header=1, dtype=str), np.genfromtxt('mapping_batch_14-5-17.csv', delimiter=',', usecols=[1], skip_header=1, dtype=float))) # different batch effects in different columns...
        out = []
        for feature in features:
            out.append(feature_batch.get(feature, np.nan))
        out = np.asarray(out)

        d_GC = {}
        for GC_range in GC:
            low, high = np.asarray(GC_range.split('-'), dtype=float)
            indices = np.argwhere((low < out) & (out <= high)).ravel()
            d_GC[GC_range] = indices
        cPickle.dump(d_GC, open(pickle_name, 'w'))
    return d_GC


def compute_batch_effects_samples(batches, samples, pickle_name='data/mapping_tissues_batch.pickle', redo=False):

    if path.exists(pickle_name) and redo == False:
        d_batches = cPickle.load(open(pickle_name, 'r'))

    else:
        db = Annotation('/home/sohse/projects/TEMP/Hauke/GEOmetadb.sqlite')

        d_batches = {}
        for batch in batches:

            if batch is 'all_batches':
                indices = range(len(samples))

            else:
                samples = list(samples)
                out = db.query('gsm', rows=samples, cols=['gsm', 'extract_protocol_ch1']) # 'extract_protocol_ch1', 'label_protocol_ch1', 'hyb_protocol', 'label_ch1' # , 'extract_protocol_ch1'
                result = db.search(batch, out)
                indices = []
                for i, gsm in enumerate(samples):
                    if gsm in result:
                        indices.append(i)
                indices = indices

            d_batches[batch] = indices
        cPickle.dump(d_batches, open(pickle_name, 'w'))
    return d_batches


def compute_annotation_features(GO_terms, features, pickle_name='data/mapping_features.pickle', redo=False): # TODO limit here to interesting features
    '''
    NOTE Map the 'scan_features' to ENSEMBL human and GO terms with description. Return as 'mapping_d-a-t-e.txt'.

    with open('features.txt', 'w') as f:
        for id_ in features:
            f.write(id_ + '\n')
    '''
    if path.exists(pickle_name) and redo == False:
        d_features = cPickle.load(open(pickle_name, 'r'))

    else:
        gene_GO = np.genfromtxt('data/mapping_8-5-17.txt', delimiter=',', usecols=[3, 1], skip_header=1, dtype=str)
        #print 'n GO terms', len(set(gene_GO[:, 1]))
        #### want_set = set(gene_GO[:, 1]) - set([''])

        d_features = {}
        for GO_term in GO_terms:
            indices = np.argwhere(gene_GO[:, 1] == GO_term).ravel().tolist()
            temp = []
            for gene in set(gene_GO[indices, 0]):
                if gene in features: #NOTE Should not be needed... but is!
                    temp.extend(np.argwhere(gene == features)[0])
            indices = temp
            d_features[GO_term] = indices
        cPickle.dump(d_features, open(pickle_name, 'w'))
    return d_features


def compute_annotation_samples(tissues, samples, pickle_name='data/mapping_tissues.pickle', redo=False):

    if path.exists(pickle_name) and redo == False:
        d_tissues = cPickle.load(open(pickle_name, 'r'))

    else:
        db = Annotation('/home/sohse/projects/TEMP/Hauke/GEOmetadb.sqlite')

        d_tissues = {}
        for tissue in tissues:

            if tissue is 'all_tissues':
                indices = range(len(samples))

            else:
                samples = list(samples)
                out = db.query('gsm', rows=samples, cols=['gsm', 'description']) # 'extract_protocol_ch1', 'label_protocol_ch1', 'hyb_protocol', 'label_ch1'
                result = db.search(tissue, out)
                indices = []
                for i, gsm in enumerate(samples):
                    if gsm in result:
                        indices.append(i)
                indices = indices

            d_tissues[tissue] = indices
        cPickle.dump(d_tissues, open(pickle_name, 'w'))
    return d_tissues


def use_only_unique(d): # NOTE Sort of validated!
    # NOTE could also find the set of overlapping values and then just remove those from each!
    """
    overlapping_set = d.values() #TODO make % over many sets!

    d_new = {}

    for k in d:
        values = d[k]
        d_new[k] = []
        for item in values:
            if item not in overlapping_set:
                d_new[k].append(item)

    return d_new
    """
    inv_d = {}
    for k, v in d.iteritems():
        for i in v:
            inv_d[i] = inv_d.get(i, [])
            inv_d[i].append(k)

    d_len1 = {}
    for k, v in inv_d.iteritems():
        if len(v) == 1:
            d_len1[k] = v

    revert_d = {}
    for k, v in d_len1.iteritems():
        for i in v:
            revert_d[i] = revert_d.get(i, [])
            revert_d[i].append(k)

    return revert_d


def visualize_full(X, d, space, normalization, batches_scatter, format='eps'):

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

    
if __name__ == '__main__':
    #load_data()

    #class DataReal(DataBlindLarge):

    #def load_data():

    # NOTE Load data

    np.random.seed(seed=42)

        
    gpl = 'GPL1261'

    with h5py.File('/home/sohse/projects/NCBI/Normalization_SCAN_{gpl}.h5'.format(gpl=gpl), 'r') as f:
        scan_features1 = np.asarray(clean(np.asarray(f['features'])))
        scan_samples1 = np.asarray(clean(np.asarray(f['samples'])))[::40]
        scan_X1 = np.asarray(f['SCAN'])

    gpl = 'GPL570'

    with h5py.File('/home/sohse/projects/NCBI/Normalization_SCAN_{gpl}.h5'.format(gpl=gpl), 'r') as f:
        scan_features2 = np.asarray(clean(np.asarray(f['features'])))
        scan_samples2 = np.asarray(clean(np.asarray(f['samples'])))[::100]
        scan_X2 = np.asarray(f['SCAN'])

    batches_scatter = {'+': range(len(scan_samples1)), 'o': (np.arange(len(scan_samples2)) + len(scan_samples1)).tolist()} # GPL1261, GPL570
    samples = np.hstack([scan_samples1, scan_samples2])

    with open('data/features_GPL1261.txt', 'w') as f:
        for id_ in scan_features1:
            f.write(id_ + '\n')

    with open('data/features_GPL570.txt', 'w') as f:
        for id_ in scan_features2:
            f.write(id_ + '\n')

    scan_features1_orthologs = scan_features1 # map_to_orthologs('mouse', scan_features1)
    scan_features2_orthologs = map_to_orthologs('human', scan_features2)

    scan1_indices, scan2_indices = align_features(scan_features1_orthologs, scan_features2_orthologs)
    scan_features1_orthologs = scan_features1_orthologs[scan1_indices]
    scan_features2_orthologs = scan_features2_orthologs[scan2_indices]

    np.testing.assert_array_equal(scan_features1_orthologs, scan_features2_orthologs)

    features = scan_features1_orthologs

    scan_X1_full = scan_X1[:, scan1_indices]
    scan_X2_full = scan_X2[:, scan2_indices]

    scan_X_full = np.vstack([scan_X1_full, scan_X2_full])
    print 'scan_X_full.shape', scan_X_full.shape

    scan_X1 = scan_X1[::40, scan1_indices]
    scan_X2 = scan_X2[::100, scan2_indices]

    scan_X = np.vstack([scan_X1, scan_X2])
    print 'scan_X.shape', scan_X.shape

    GO_terms = sorted(['GO:0007283', 'GO:0016020', 'GO:0045095', 'GO:0016491'])
    tissues = sorted(['liver', 'kidney']) #, 'lung', 'skin', 'muscle', 'brain' # , 'all_tissues']
    d_samples = use_only_unique(compute_annotation_samples(tissues, samples, redo=False))
    d_features = use_only_unique(compute_annotation_features(GO_terms, scan_features1[scan1_indices], redo=False))
    
    #X =
    #y =
    #return X, y

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
    
    plot_threshold_to_pairs(X_scan_subset, 'SCAN_sample')
    #plot_threshold_to_pairs(X_scan_subset.T, 'SCAN_feature') # WARNING doesn't work

    for rank in range(1, 5):
        print rank
        data = DataBlind(X_scan_subset, rank, noise_amplitude=1.0) #NOTE # noise_amplitude is just for guess!
        data.estimate()
        n_measurements = 1000 # TODO max and min non-redundant? Note that random sampling will always yield some ducpicates.
        operator = LinearOperatorBlind(data.d, n_measurements).generate()
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














    
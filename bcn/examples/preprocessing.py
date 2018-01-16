"""Preprocesing and creation of the high-throughput data.

Notes
-----
Defines a classes that can be used to annotate high-throughput data and conducts preprocessing.
"""
from __future__ import division, absolute_import


__all__ = ['Annotation']

import sys # WARNING remove in final version
sys.path.append('/home/sohse/projects/PUBLICATION/GITssh/bcn')

from os import path
import re
import cPickle
import sqlite3

import h5py
import numpy as np


class Annotation(object):
    '''
    Allows queries to the SQLite database from http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2639278/.

    Web interface @ http://gbnci.abcc.ncifcrf.gov/geo/.
    '''
    def __init__(self, database='../../data/GEOmetadb.sqlite'):
        self.connection = sqlite3.connect(database)

        def dict_factory(cursor, row):
            d = {}
            for idx, col in enumerate(cursor.description):
                d[col[0]] = row[idx]
            return d

        self.connection.row_factory = dict_factory
        self.connection.text_factory = str
        self.cursor = self.connection.cursor()

    def view(self):
        '''
        Lists all tables and corresponding columns in the database.
        '''
        tables = self.cursor.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()
        for table in tables:
            columns = self.cursor.execute('SELECT * FROM {table}'.format(table=table['name'])).fetchone()
            print '\nTable:', table['name'],
            print '\nColumns:', columns.keys()

    def query(self, table, cols='*', rows=None):
        '''
        Queries database for a list of columns (or all, e.g. *) in a table and return dictionary with result. # WARNING Assumes GSM of each row is unique.
        '''
        if rows:
            self.cursor.execute('SELECT {cols} FROM {table} WHERE gsm in ("{rows}")'.format(cols=', '.join(cols), table=table, rows='", "'.join(rows)))
        else:
            self.cursor.execute('SELECT {cols} FROM {table}'.format(cols=', '.join(cols), table=table))
        result = self.cursor.fetchall()
        return result

    def search(self, pattern, result):
        '''
        Checks for occurance of a regular expression pattern in a query result. Ignores case and returns matches if >= 1 are found. # WARNING record must have GSM key.
        '''
        matches = []
        for record in result:
            for value in record.values():
                if re.search(pattern, str(value), re.IGNORECASE): # len(re.findall(pattern, value, re.IGNORECASE))
                    matches.append(record['gsm'])
        return matches


def clean(names):
    '''
    Clean up a list of file names or features by removing for example "_at" or ".CEL" at the end.

    Parameters
    ----------
    names : list (of str)
        A list of to be cleaned file or feature names.
        
    Returns
    -------
    cleaned : list (of str)
        A list of cleaned file nor feature names.
    '''
    cleaned  = [re.split(r'[_|.]', name)[0] for name in names]
    return cleaned


def human_to_mouse(features):
    '''
    Maps human ENSEMBL IDs from platform GPL570 to mouse ENSEMBL IDs from platform GPL1261.

    Parameters
    ----------
    features : list (of str)
        A list of ENSEMBL IDs from human (subset of GPL570)
        
    Returns
    -------
    mapped : list (of str)
        A list of mapped ENSEMBL IDs usable for mouse (subset of GPL1261)
    '''
    mapping = dict(np.genfromtxt('../../data/GPL570_GPL1261_ortho.txt', delimiter=',', dtype=str))
    temp = []
    for feature in features:
        try:
            mouse = mapping[feature]
            temp.append(mouse)
        except KeyError:
            temp.append('')
    mapped = np.asarray(temp)
    return mapped


def overlapping_feature_indices(a, b): # TODO might have to do cross-species with mapping_8-5-17.txt
    '''
    Selects an overlap of features and returns the corresponding indices.

    Parameters
    ----------
    a : sequence of str
        A sequence of feature names.
    b : sequence of str
        A sequence of feature names.
        
    Returns
    -------
    a_indices, b_indices : tuple (of 1darray of int)
        Two arrays of the overlapping features indices.
    '''
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
                print 'Warning: Multiple or missing matches.', a_index, b_index
    a_indices = np.hstack(a_indices)
    b_indices = np.hstack(b_indices)
    return a_indices, b_indices 

                    
def compute_annotation(labels, items, column_names=['gsm', 'description'], pickle_file_name='../../data/mapping_tissues.pickle', recompute=False):
    '''
    Parameters
    ----------
    labels : list(of str)  or array
        A list of tissue or GO term names used as labels for the subsequent classification.
    data : Sequence (of str)
        A list of sample or feature names to be used.
    column_names : list (or str)
        List of sql database ``cols'' to be serached for the labels.
    pickle_file_name : str
        File name of the pickle file from which a precomputed dictionary can be loaded if recompute=False. 
    recompute : bool
        Switch weather to recompute the dictionary.
    
    Returns
    -------
    d : dict
        A dictionary containing the sample names ('lung', 'liver') or feature names ()
    '''
    if path.exists(pickle_file_name) and recompute == False:
        d = cPickle.load(open(pickle_file_name, 'r'))
    else:
        db = Annotation('../../data/GEOmetadb.sqlite')
        d = {}
        for label in labels:
            out = db.query('gsm', rows=list(items), cols=column_names) # 'extract_protocol_ch1', 'label_protocol_ch1', 'hyb_protocol', 'label_ch1'
            result = db.search(label, out)
            indices = []
            for i, item in enumerate(items):
                if item in result:
                    indices.append(i)
            d[label] = indices

    d = keep_unique_values(d)
    cPickle.dump(d, open(pickle_file_name, 'w'))
    return d


def keep_unique_values(d): # NOTE Sort of validated!
    '''
    Removes non-unique items in the values from a dictionary.

    Parameters
    ----------
    d : dict
        A dictionary containing keys equal to tissue or GO labels and values that contain indices to the respective samples/features.

    Returns
    -------
    unique : dict
        A dictionary with non-unique items in the values removed.
    '''
    inv_d = {}
    for k, v in d.iteritems():
        for i in v:
            inv_d[i] = inv_d.get(i, [])
            inv_d[i].append(k)
    d_len1 = {}
    for k, v in inv_d.iteritems():
        if len(v) == 1:
            d_len1[k] = v
    unique = {}
    for k, v in d_len1.iteritems():
        for i in v:
            unique[i] = unique.get(i, [])
            unique[i].append(k)
    return unique


if __name__ == '__main__':

    np.random.seed(seed=42)

    gpl = 'GPL1261'
    with h5py.File('/home/sohse/projects/NCBI/Normalization_SCAN_{gpl}.h5'.format(gpl=gpl), 'r') as f:
        scan_features1 = np.asarray(clean(np.asarray(f['features'])))
        scan_samples1 = np.asarray(clean(np.asarray(f['samples'])))
        scan_X1 = np.asarray(f['SCAN'])

    gpl = 'GPL570'
    with h5py.File('/home/sohse/projects/NCBI/Normalization_SCAN_{gpl}.h5'.format(gpl=gpl), 'r') as f:
        scan_features2 = np.asarray(clean(np.asarray(f['features'])))
        scan_samples2 = np.asarray(clean(np.asarray(f['samples'])))
        scan_X2 = np.asarray(f['SCAN'])

    '''
    # NOTE The storage in between in order to do the online mapping of features to orthologues.
    
    with open('../../data/features_GPL1261.txt', 'w') as f:
        for id_ in scan_features1:
            f.write(id_ + '\n')

    with open('../../data/features_GPL570.txt', 'w') as f:
        for id_ in scan_features2:
            f.write(id_ + '\n')
    '''

    # NOTE Feature merging based on orthologues.
    scan_features2_mapped = human_to_mouse(scan_features2)
    indices_1, indices_2 = overlapping_feature_indices(scan_features1, scan_features2_mapped)
    scan_features1_subset = scan_features1[indices_1]
    scan_features2_subset = scan_features2_mapped[indices_2]
    np.testing.assert_array_equal(scan_features1_subset, scan_features2_subset)

    # NOTE Reduction of the dataset to approximately the same size for each plattform and ~1000 each.
    subset_1 = 4 # 40
    subset_2 = 10 # 100
    scan_X1 = scan_X1[::subset_1, indices_1]
    scan_X2 = scan_X2[::subset_2, indices_2] 
    scan_X = np.vstack([scan_X1, scan_X2])
    #print 'scan_X.shape', scan_X.shape
    labels_batches = np.asarray(['+'] * len(scan_samples1[::subset_1]) + ['o'] * len(scan_samples2[::subset_2])) # GPL1261, GPL570
    labels_gsms = np.hstack([scan_samples1[::subset_1], scan_samples2[::subset_2]])
    
    labels_tissues_2 = sorted(['liver', 'kidney'])
    labels_tissues_6 = sorted(['liver', 'kidney', 'lung', 'skin', 'muscle', 'brain'])

    Xs, ys, batches, gsms = [], [], [], []
    for d_samples, labels in zip([compute_annotation(labels_tissues_2, labels_gsms, recompute=True), compute_annotation(labels_tissues_6, labels_gsms, recompute=True)], [labels_tissues_2, labels_tissues_6]):
        Xs.append(np.vstack([scan_X[d_samples[item]] for item in labels]))
        ys.append(np.hstack([len(d_samples[item]) * [item] for item in labels]))
        batches.append(np.hstack([np.asarray(labels_batches)[d_samples[item]] for item in labels]))
        gsms.append(np.hstack([labels_gsms[d_samples[item]] for item in labels]))

    cPickle.dump(ys, open('../../data/ys.pickle', 'w'))
    cPickle.dump(Xs, open('../../data/Xs.pickle', 'w'))
    cPickle.dump(batches, open('../../data/batches.pickle', 'w'))
    cPickle.dump(gsms, open('../../data/gsms.pickle', 'w'))

    # TODO Estimate stds, pairs and directions (e.g. correlations) on large database and use the best know correlations together with the rows/columns of interest here to estimate batch effects and correct.
    # TODO Might have to optimize solver, e.g. sparsity stuff and randomization properly.
    # TODO get also batch effect annotations from mining, not just platforms

















    
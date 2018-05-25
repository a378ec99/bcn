"""Linear operator tests.
"""
from __future__ import division, absolute_import

import unittest

import numpy as np

from bcn.linear_operators import LinearOperatorEntry, LinearOperatorDense, LinearOperatorKsparse, LinearOperatorCustom, integer_to_matrix, sample_n_choose_k, choose_random_matrix_elements
from bcn.data import DataSimulated, estimate_partial_signal_characterists


class TestSimple(unittest.TestCase):
    """Test to verify shapes and outputs of different linear operators are correct.
    """
    def setUp(self):
        self.n_samples = 10
        self.n_features = 9
        self.sparsity = 2
        self.n = 90
        self.signal = np.asarray(np.arange(90), dtype=float).reshape((self.n_samples, self.n_features))
        self.signal_with_nan = np.array(self.signal)
        self.signal_with_nan[0, 0] = np.nan
        self.signal_with_nan[0, 1] = np.nan
        
    def test_entry(self):
        operator = LinearOperatorEntry(self.n)
        out = operator.generate(self.signal)
        assert len(out['y']) == self.n
        assert len(out['A'][0]['value']) == 1
        out = operator.generate(self.signal_with_nan)
        assert len(out['y']) == self.n - 2

    def test_custom(self):
        np.random.seed(42)
        rank = 2
        correlation_threshold = 0.8
        data = DataSimulated((self.n_samples, self.n_features), rank, missing_type='no-missing', m_blocks_size=5)
        true_correlations = {'sample': data.d['sample']['true_correlations'], 'feature': data.d['feature']['true_correlations']}
        true_pairs = {'sample': data.d['sample']['true_pairs'], 'feature': data.d['feature']['true_pairs']}
        true_directions = {'sample': data.d['sample']['true_directions'], 'feature': data.d['feature']['true_directions']}
        true_stds = {'sample': data.d['sample']['true_stds'], 'feature': data.d['feature']['true_stds']}
        mixed = data.d['sample']['mixed']
        estimated = estimate_partial_signal_characterists(mixed, correlation_threshold,  true_pairs=true_pairs, true_directions=true_directions, true_stds=true_stds, true_correlations=true_correlations)
        operator = LinearOperatorCustom(self.n)
        out = operator.generate(estimated)
        assert len(out['A'][0]['value']) == 2
        assert len(out['y']) == self.n
        mixed[0, 0] = np.nan
        operator = LinearOperatorCustom(self.n)
        out = operator.generate(estimated)
        assert len(out['A'][0]['value']) == 2
        assert len(out['y']) == self.n - 3     
        
    def test_dense(self):
        operator = LinearOperatorDense(self.n)
        out = operator.generate(self.signal)
        assert len(out['y']) == self.n
        assert len(out['A'][0]['value']) == self.n_samples * self.n_features
        out = operator.generate(self.signal_with_nan)
        assert len(out['y']) == self.n
        assert np.isfinite(out['y']).all()
        
    def test_ksparse(self):
        operator = LinearOperatorKsparse(self.n, self.sparsity)
        out = operator.generate(self.signal)
        assert len(out['y']) == self.n
        assert len(out['A'][0]['value']) == self.sparsity
        np.random.seed(37)
        out = operator.generate(self.signal_with_nan)
        assert len(out['y']) == self.n
        assert np.isfinite(out['y']).all()
        np.random.seed(38)   
        out = operator.generate(self.signal_with_nan)
        assert len(out['y']) == self.n - 1
        assert np.isfinite(out['y']).all()

    def test_random_sparse_matrices(self):
        n_samples = 3
        n_features = 2
        max_ = 15
        samples = sample_n_choose_k(n_samples, n_features, self.sparsity, max_)
        assert len(set(samples)) == max_ 
        A = [integer_to_matrix(i, self.sparsity, n_samples, n_features) for i in samples]
        flattened = [tuple(a.flatten()) for a in A]
        assert len(flattened) == len(set(flattened))

    def test_random_elements(self):
        element_indices = choose_random_matrix_elements((self.n_samples, self.n_features), self.n)
        assert self.n == len(element_indices)
        element_indices_tuple = [(index_pair[0], index_pair[1]) for index_pair in element_indices]
        assert len(set(element_indices_tuple)) == self.n
        assert element_indices.shape == (self.n, 2)
        assert np.isfinite(element_indices).all() == True
        assert type(element_indices) == np.ndarray
            

if __name__ == '__main__':
    unittest.main()







        

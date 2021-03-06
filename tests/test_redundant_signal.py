"""Redundant signal matrix tests.
"""
from __future__ import division, absolute_import

import unittest

import numpy as np

from bcn.redundant_signal import RedundantSignal, _generate_square_blocks_matrix, _generate_pairs, _generate_stds, _generate_directions, _generate_covariance, _generate_matrix_normal_sample


class Test_std_consistency(unittest.TestCase):
    """ Test the consistency of standard deviations.
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.m_blocks = 10
        self.shape = (250, 250)
        
    def test(self):
        self.correlation_matrix_U = _generate_square_blocks_matrix(self.shape[0], self.m_blocks, r=1.0)
        self.stds_U, scaling_factor_sample = _generate_stds(self.shape[0], 'random', normalize=True)
        self.U = _generate_covariance(self.correlation_matrix_U, self.stds_U) # sample
        print 'trace U', np.trace(self.U)
        assert np.allclose(np.sum(self.stds_U**2), 1.0, rtol=1e-5, atol=1e-5)
        pairs_sample = _generate_pairs(self.correlation_matrix_U)
        
        self.correlation_matrix_V = _generate_square_blocks_matrix(self.shape[1], self.m_blocks, r=1.0)
        self.stds_V, scaling_factor_feature = _generate_stds(self.shape[1], 'random', normalize=True) 
        self.V = _generate_covariance(self.correlation_matrix_V, self.stds_V) # feature
        print 'trace V', np.trace(self.V)
        assert np.allclose(np.sum(self.stds_U**2), 1.0, rtol=1e-5, atol=1e-5)
        pairs_feature = _generate_pairs(self.correlation_matrix_V)
        
        temp = []
        for q in xrange(100):
            self.sample = _generate_matrix_normal_sample(self.U, self.V)
            temp.append(self.sample)
        self.sample = np.mean(temp, axis=0)
        print self.sample.shape
        sample_stds = np.std(self.sample, axis=1)
        feature_stds = np.std(self.sample, axis=0)

        A = (feature_stds[pairs_feature][:, 0] / feature_stds[pairs_feature][:, 1])[:4]
        B = (sample_stds[pairs_sample][:, 0] / sample_stds[pairs_sample][:, 1])[:4]
        true_A = (self.stds_V[pairs_feature][:, 0] / self.stds_V[pairs_feature][:, 1])[:4]
        true_B = (self.stds_U[pairs_sample][:, 0] / self.stds_U[pairs_sample][:, 1])[:4]

        assert np.allclose(A, true_A, rtol=1e-5, atol=1e-5)
        assert np.allclose(B, true_B, rtol=1e-5, atol=1e-5)
        
        print 'feature mean stds and raw', np.mean(feature_stds), np.mean(self.stds_V) # NOTE To get macht, should not normalize.
        print 'sample mean stds and raw', np.mean(sample_stds), np.mean(self.stds_U)
        print 'difference feature stds and raw', np.mean(feature_stds - self.stds_V)
        print 'difference sample stds and raw', np.mean(sample_stds - self.stds_U)
        
      
class Test_generate_covariance(unittest.TestCase):
    """Test to verify that covariances are generated correctly.

    Attributes
    ----------
    seed : int, default = 42
        Random seed of the whole test.
    correlation_matrix : ndarray, shape (n_samples, n_samples) or (n_features, n_features)
        Correlation matrix on which the covariance matrix is based.
    stds : ndarray, (dimensions = 1, e.g. like list)
        Standard deviations which are used in combination with the correlation amtrix to generate the covariance matrix.
    n : int
        Dimensions of the square block correlation matrix (n, n).
    m_blocks : int, default = 4
        The number of square blocks in the correlation matrix. Does not have to be an even fit into `n`.
    shape : tuple of int, computed automatically as (n, n)
        Shape of the covariance matrix.
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.n = 15
        self.m_blocks = 4
        self.shape = (self.n, self.n)
        self.correlation_matrix = _generate_square_blocks_matrix(self.n, self.m_blocks, r=1.0)
        self.stds, self.scaling_factor = _generate_stds(self.n, 'random', normalize=True)
        self.covariance_matrix = _generate_covariance(self.correlation_matrix, self.stds)
        
    def _assert_finite(self, covariance_matrix):
        assert np.isfinite(covariance_matrix).all() == True

    def _assert_shape(self, covariance_matrix):
        assert covariance_matrix.shape == self.shape

    def _assert_ndarray(self, covariance_matrix):
        assert type(covariance_matrix) == np.ndarray

    def test(self):
        covariance_matrix = _generate_covariance(self.correlation_matrix, self.stds)
        self._assert_finite(covariance_matrix)
        self._assert_shape(covariance_matrix)
        self._assert_ndarray(covariance_matrix)

        
class Test_generate_directions(unittest.TestCase):
    """Test to verify that directions are generated correctly.

    Attributes
    ----------
    seed : int, default = 42
        Random seed of the whole test.
    correlation_matrix : ndarray, shape (n_samples, n_samples) or (n_features, n_features)
        Signed correlation matrix from which the signs are extracted as direcitons.
    pairs : ndarray, (dimensions = 1, e.g. like list)
        The pairs of interest for which the directions are to be extraced.
    n : int
        Dimensions of the square block correlation matrix (n, n).
    m_blocks : int, default = 4
        The number of square blocks in the correlation matrix. Does not have to be an even fit into `n`.
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.n = 15
        self.m_blocks = 4
        self.correlation_matrix = _generate_square_blocks_matrix(self.n, self.m_blocks, r=1.0)
        self.pairs = _generate_pairs(self.correlation_matrix)

    def _assert_finite(self, directions):
        assert np.isfinite(directions).all() == True

    def _assert_shape(self, directions, pairs):
        assert directions.shape[0] == len(pairs)

    def _assert_ndarray(self, directions):
        assert type(directions) == np.ndarray
        
    def test(self):
        directions = _generate_directions(self.correlation_matrix, self.pairs)
        self._assert_finite(directions)
        self._assert_shape(directions, self.pairs)
        self._assert_ndarray(directions)

        
class Test_generate_stds(unittest.TestCase):
    """Test to verify that standard deviations are generated correctly.

    Attributes
    ----------
    seed : int, default = 42
        Random seed of the whole test.
    n : int, default = 19
        Feature/sample dimensions of the matrix for which the standard deviations are computed.
    model : {'random', 'constant'}
        The type of model to generate standard deviations. Constant uses the `std_value`, whereas `random` samples from a uniform distribution between 0.1 and 2.
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.n = 19
        
    def _assert_finite(self, stds):
        assert np.isfinite(stds).all() == True

    def _assert_float(self, scaling_factor):
        assert type(scaling_factor) == float

    def _assert_shape(self, stds, n):
        assert stds.shape[0] == n

    def _assert_ndarray(self, stds):
        assert type(stds) == np.ndarray
        
    def test_random(self):
        stds, scaling_factor = _generate_stds(self.n, 'random', normalize=True)
        self._assert_finite(stds)
        self._assert_float(scaling_factor)
        self._assert_shape(stds, self.n)
        self._assert_ndarray(stds)
        
    def test_constant(self):
        stds, scaling_factor = _generate_stds(self.n, 'constant', normalize=True)
        self._assert_finite(stds)
        self._assert_float(scaling_factor)
        self._assert_shape(stds, self.n)
        self._assert_ndarray(stds)

    
class Test_generate_square_blocks_matrix(unittest.TestCase):
    """Test to verify that the square block matrices are generated correctly.
    
    Attributes
    ----------
    seed : int, default = 42
        Random seed of the whole test.
    n : int
        Dimensions of the square block matrix (n, n).
    shape :tuple of int, computed from `n` (no need to set)
        Shape of the square block matrix.
    m_blocks : int, default = 4
        Number of blocks in the correlation matix of features or samples that are varying together (with differences only in degree, direction and scale). Fewer blocks are better for bias recovery.
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.n = 17
        self.shape = (self.n, self.n)
        self.m_blocks = 4

    def _assert_finite(self, square_blocks_matrix):
        assert np.isfinite(square_blocks_matrix).all() == True

    def _assert_shape(self, square_blocks_matrix):
        assert square_blocks_matrix.shape == self.shape

    def _assert_ndarray(self, square_blocks_matrix):
        assert type(square_blocks_matrix) == np.ndarray
      
    def test(self):
        square_blocks_matrix = _generate_square_blocks_matrix(self.n, self.m_blocks)
        self._assert_finite(square_blocks_matrix)
        self._assert_shape(square_blocks_matrix)
        self._assert_ndarray(square_blocks_matrix)


class Test_generate_pairs(unittest.TestCase):
    """Test to verfiy that indicies of correlated pairs are generated correctly.

    Attributes
    ----------
    seed : int, default = 42
        Random seed of the whole test.
    n : int
        Dimensions of the square block correlation matrix (n, n).
    correlation_matrix: ndarray, shape (n_features. n_features), computed from `n` (no need to set)
        Correlation matrix with a square block structure.
    m_blocks : int, default = 4
        The number of square blocks in the correlation matrix. Does not have to be an even fit into `n`.
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.n = 15
        self.m_blocks = 4
        self.correlation_matrix = _generate_square_blocks_matrix(self.n, self.m_blocks, r=1.0)
        
    def _assert_int(self, pairs):
        assert pairs.dtype == int
        
    def _assert_finite(self, pairs):
        assert np.isfinite(pairs).all() == True

    def _assert_shape(self, pairs):
        assert pairs.shape[1] == 2

    def _assert_ndarray(self, pairs):
        assert type(pairs) == np.ndarray

    def test(self):
        pairs = _generate_pairs(self.correlation_matrix)
        self._assert_finite(pairs)
        self._assert_shape(pairs)
        self._assert_ndarray(pairs)
        self._assert_int(pairs)


class Test_generate_matrix_normal_sample(unittest.TestCase):
    """Test to verify that the sample from the matrix variate normal distribution is generated correctly.

    Attributes
    ----------
    seed : int, default = 42
        Random seed of the whole test.
    shape : tuple of int, (n_samples, n_features)
        Shape of the matrix.
    m_blocks : int
        The number of square blocks in the correlation matrix. Does not have to be an even fit into `n`.
    U : ndarray, shape (n_samples, n_samples)
        Covariance matrix among samples (rows).
    V : ndarray, shape (n_features, n_features)
        Covariance matrix among features (columns).
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.shape = (25, 35)
        self.m_blocks = 4
        self.correlation_matrix_U = _generate_square_blocks_matrix(self.shape[0], self.m_blocks, r=1.0)
        self.stds_U = _generate_stds(self.shape[0], 'random', normalize=False)[0]
        self.U = _generate_covariance(self.correlation_matrix_U, self.stds_U)
        self.correlation_matrix_V = _generate_square_blocks_matrix(self.shape[1], self.m_blocks, r=1.0)
        self.stds_V = _generate_stds(self.shape[1], 'random', normalize=False)[0]
        self.V = _generate_covariance(self.correlation_matrix_V, self.stds_V)
        
    def _assert_finite(self, sample):
        assert np.isfinite(sample).all() == True

    def _assert_shape(self, sample):
        assert sample.shape == self.shape

    def _assert_ndarray(self, sample):
        assert type(sample) == np.ndarray
        
    def test(self):
        self.sample = _generate_matrix_normal_sample(self.U, self.V)
        self._assert_finite(self.sample)
        self._assert_shape(self.sample)
        self._assert_ndarray(self.sample)

    
class TestRedundantSignal(unittest.TestCase):
    """Test to verify that all signal outputs are finite consistent ndarrays.

    Attributes
    ----------
    seed : int, default = 42
        Random seed of the whole test.
    shape : tuple of int, default = (50, 60)
        Shape of the output signal matrix in the form of (n_samples, n_features).
    m_blocks : int, default = 4
        Number of blocks in the correlation matix of features or samples that are varying together (with differences only in degree, direction and scale). Fewer blocks are better for bias recovery.
    correlation_strength : float, default = 1.0
        The strength of correlations between features or samples. Stronger correlations are better for bias recovery.
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.shape = (50, 60)
        self.m_blocks = 4
        self.correlation_strength = 1.0
        
    def _assert_shape(self, signal):
        assert signal['X'].shape == self.shape

    def _assert_finite(self, signal):
        assert np.isfinite(signal['X']).all() == True

    def _assert_ndarray(self, signal):
        assert type(signal['X']) == np.ndarray

    def _assert_dict(self, signal):
        assert type(signal) == dict

    def test_constant(self):
        signal = RedundantSignal(self.shape, 'constant', self.m_blocks, self.correlation_strength, std_value=-1.5).generate()
        self._assert_dict(signal)
        self._assert_finite(signal)
        self._assert_ndarray(signal)
        self._assert_shape(signal)
      
    def test_random(self):
        signal = RedundantSignal(self.shape, 'random', self.m_blocks, self.correlation_strength).generate()
        self._assert_dict(signal)
        self._assert_finite(signal)
        self._assert_ndarray(signal)
        self._assert_shape(signal)

        
if __name__ == '__main__':
    unittest.main()



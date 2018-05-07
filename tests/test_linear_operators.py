"""Linear operator and measurement testing.

Notes
-----
Defines a test class that asserts the functioning of the `linear_operator` module and its related functions.
"""
from __future__ import division, absolute_import

import unittest
import hashlib

import numpy as np

from bcn.linear_operators import _choose_random_matrix_elements, LinearOperatorEntry, LinearOperatorDense, LinearOperatorKsparse, LinearOperatorCustom
from bcn.data import DataSimulated
from bcn.utils.testing import assert_consistency


class Test_choose_random_matrix_elements(unittest.TestCase):
    """Test to verify that random matrix elements are chosen correctly.

    Attributes
    ----------
    seed : int, default = 42
        Random seed of the whole test.
    shape : tuple of int, no need to set with default = (50, 60)
        Shape of the matrix of which to sample the random elements from.
    n : int
        Number of elements to choose (max. determined by shape).
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.shape = (50, 60)
        
    def _assert_n(self, n, element_indices):
        assert n == len(element_indices)
        
    def _assert_no_duplicates(self, n, element_indices):
        element_indices = [(index_pair[0], index_pair[1]) for index_pair in element_indices]
        assert len(set(element_indices)) == n

    def _assert_shape(self, n, element_indices):
        assert element_indices.shape == (n, 2)

    def _assert_finite(self, element_indices):
        assert np.isfinite(element_indices).all() == True

    def _assert_ndarray(self, element_indices):
        assert type(element_indices) == np.ndarray

    def _assert_consistency(self, element_indices, true_md5):
        m = hashlib.md5()
        element_indices = np.asarray(element_indices, order='C') # NOTE Interesting case of ValueError: ndarray is not C-contiguous. Why?
        m.update(element_indices) 
        current_md5 = m.hexdigest()
        assert current_md5 == true_md5
        
    def test_sample_no_duplicates_all(self):
        n = self.shape[0] * self.shape[1]
        element_indices = _choose_random_matrix_elements(self.shape, n)
        self._assert_n(n, element_indices)
        self._assert_no_duplicates(n, element_indices)
        self._assert_shape(n, element_indices)
        self._assert_finite(element_indices)
        self._assert_ndarray(element_indices)
        self._assert_consistency(element_indices, '9447d323a3d696d8ba0022cd34462789')
        
    def test_sample_no_duplicates_subset(self):
        n = 15
        element_indices = _choose_random_matrix_elements(self.shape, n)
        self._assert_n(n, element_indices)
        self._assert_no_duplicates(n, element_indices)
        self._assert_shape(n, element_indices)
        self._assert_finite(element_indices)
        self._assert_ndarray(element_indices)
        self._assert_consistency(element_indices, '8c3ef5a1fdcda8a1f5071dd734319c2a')

    def test_sample(self):
        n = self.shape[0] * self.shape[0] 
        element_indices = _choose_random_matrix_elements(self.shape, n, duplicates=True)
        self._assert_n(n, element_indices)
        self._assert_shape(n, element_indices)
        self._assert_finite(element_indices)
        self._assert_ndarray(element_indices)
        self._assert_consistency(element_indices, '9436c2e0a181dcb9c65873950b807371')

        
class TestLinearOperatorEntry(unittest.TestCase):
    """Test to verify that the entry based linear operator and its measurements are generated correctly.

    Attributes
    ----------
    data : Data object
        Contains a dictionary with all the data that is needed for the linear operator and measurment creation.
    n_measurements : int (default = 101)
        Number of linear operators and measurements to be generated.
    shape : tuple of int, optional (unless mixed == None), default = (30, 40)
        Shape of the mixed, signal, bias and missing matrix in the form of (n_samples, n_features) of the data object.
    rank : int, (default = 2)
        Rank of the low-rank decomposition for the data object.
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.shape = (50, 60)
        self.n_measurements = 1000
        self.rank = 2
        data = DataSimulated(self.shape, self.rank)
        self.data = data
        
    def _assert_shape(self, A, y):
        assert A.shape == (self.n_measurements, self.shape[0], self.shape[1])
        assert y.shape[0] == self.n_measurements
        
    def _assert_finite(self, A, y):
        assert np.isfinite(A).all() == True
        assert np.isfinite(y).all() == True

    def _assert_ndarray(self, A, y):
        assert type(A) == np.ndarray
        assert type(y) == np.ndarray

    def _assert_nonzero_sum(self, A, y):
        assert np.sum(A) == self.n_measurements
        assert np.sum(y) != 0.0
    
    def test(self):
        operator = LinearOperatorEntry(self.data, self.n_measurements).generate()
        A = operator['A']
        y = operator['y']
        self._assert_shape(A, y)
        self._assert_finite(A, y)
        self._assert_ndarray(A, y)
        self._assert_nonzero_sum(A, y)
        assert_consistency(A, '538a3ea661a347917553bef1fb05d972')

        
class TestLinearOperatorDense(unittest.TestCase):
    """Test to verify that the dense based linear operator and its measurements are generated correctly.

    Attributes
    ----------
    data : Data object
        Contains a dictionary with all the data that is needed for the linear operator and measurment creation.
    n_measurements : int (default = 101)
        Number of linear operators and measurements to be generated.
    shape : tuple of int, optional (unless mixed == None), default = (30, 40)
        Shape of the mixed, signal, bias and missing matrix in the form of (n_samples, n_features) of the data object.
    rank : int, (default = 2)
        Rank of the low-rank decomposition for the data object.
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.shape = (50, 60)
        self.n_measurements = 1000
        self.rank = 2
        data = DataSimulated(self.shape, self.rank)
        self.data = data

    def _assert_shape(self, A, y):
        assert A.shape == (self.n_measurements, self.shape[0], self.shape[1])
        assert y.shape[0] == self.n_measurements

    def _assert_finite(self, A, y):
        assert np.isfinite(A).all() == True
        assert np.isfinite(y).all() == True

    def _assert_ndarray(self, A, y):
        assert type(A) == np.ndarray
        assert type(y) == np.ndarray

    def _assert_nonzero_sum(self, A, y):
        assert np.sum(A) != 0.0
        assert np.sum(y) != 0.0

    def test(self):
        operator = LinearOperatorDense(self.data, self.n_measurements).generate()
        A = operator['A']
        y = operator['y']
        self._assert_shape(A, y)
        self._assert_finite(A, y)
        self._assert_ndarray(A, y)
        self._assert_nonzero_sum(A, y)
        assert_consistency(A, '12ce54cc5d20ea2629c13891fc3065ed')


class TestLinearOperatorKsparse(unittest.TestCase):
    """Test to verify that the sparse linear operator and its measurements are generated correctly.

    Attributes
    ----------
    data : Data object
        Contains a dictionary with all the data that is needed for the linear operator and measurment creation.
    n_measurements : int (default = 101)
        Number of linear operators and measurements to be generated.
    shape : tuple of int, optional (unless mixed == None), default = (30, 40)
        Shape of the mixed, signal, bias and missing matrix in the form of (n_samples, n_features) of the data object.
    rank : int, (default = 2)
        Rank of the low-rank decomposition for the data object.
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.shape = (50, 60)
        self.n_measurements = 1000
        self.rank = 2
        data = DataSimulated(self.shape, self.rank)
        self.data = data
        
    def _assert_shape(self, A, y):
        assert A.shape == (self.n_measurements, self.shape[0], self.shape[1])
        assert y.shape[0] == self.n_measurements

    def _assert_finite(self, A, y):
        assert np.isfinite(A).all() == True
        assert np.isfinite(y).all() == True

    def _assert_ndarray(self, A, y):
        assert type(A) == np.ndarray
        assert type(y) == np.ndarray

    def _assert_nonzero_sum(self, A, y):
        assert np.sum(A) != 0.0
        assert np.sum(y) != 0.0
        
    def test_k1(self):
        operator = LinearOperatorKsparse(self.data, self.n_measurements, 1).generate()
        A = operator['A']
        y = operator['y']
        self._assert_shape(A, y)
        self._assert_finite(A, y)
        self._assert_ndarray(A, y)
        self._assert_nonzero_sum(A, y)
        assert_consistency(A, '2294b09ca9bec5295989b90558d646db')
    
    def test_k2(self):
        operator = LinearOperatorKsparse(self.data, self.n_measurements, 2).generate()
        A = operator['A']
        y = operator['y']
        self._assert_shape(A, y)
        self._assert_finite(A, y)
        self._assert_ndarray(A, y)
        self._assert_nonzero_sum(A, y)
        assert_consistency(A, 'f0ced32671c642581475304a0c3eddfb')
       
    def test_k3(self):
        operator = LinearOperatorKsparse(self.data, self.n_measurements, 3).generate()
        A = operator['A']
        y = operator['y']
        self._assert_shape(A, y)
        self._assert_finite(A, y)
        self._assert_ndarray(A, y)
        self._assert_nonzero_sum(A, y)
        assert_consistency(A, 'd2f329357b12e01488d03df4c8415735')


class TestLinearOperatorKsparse(unittest.TestCase):
    """Test to verify that the sparse linear operator and its measurements are generated correctly.

    Attributes
    ----------
    data : Data object
        Contains a dictionary with all the data that is needed for the linear operator and measurment creation.
    n_measurements : int (default = 101)
        Number of linear operators and measurements to be generated.
    shape : tuple of int, optional (unless mixed == None), default = (30, 40)
        Shape of the mixed, signal, bias and missing matrix in the form of (n_samples, n_features) of the data object.
    rank : int, (default = 2)
        Rank of the low-rank decomposition for the data object.
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.shape = (50, 60)
        self.n_measurements = 1000
        self.rank = 2
        data = DataSimulated(self.shape, self.rank)
        self.data = data

    def _assert_shape(self, A, y):
        assert A.shape == (self.n_measurements, self.shape[0], self.shape[1])
        assert y.shape[0] == self.n_measurements

    def _assert_finite(self, A, y):
        assert np.isfinite(A).all() == True
        assert np.isfinite(y).all() == True

    def _assert_ndarray(self, A, y):
        assert type(A) == np.ndarray
        assert type(y) == np.ndarray

    def _assert_nonzero_sum(self, A, y):
        assert np.sum(A) != 0.0
        assert np.sum(y) != 0.0

    def test_k1(self):
        operator = LinearOperatorKsparse(self.data, self.n_measurements, 1).generate()
        A = operator['A']
        y = operator['y']
        self._assert_shape(A, y)
        self._assert_finite(A, y)
        self._assert_ndarray(A, y)
        self._assert_nonzero_sum(A, y)
        assert_consistency(A, '6d7a25377414360e85f9ea63a3fd1c82')

    def test_k2(self):
        operator = LinearOperatorKsparse(self.data, self.n_measurements, 2).generate()
        A = operator['A']
        y = operator['y']
        self._assert_shape(A, y)
        self._assert_finite(A, y)
        self._assert_ndarray(A, y)
        self._assert_nonzero_sum(A, y)
        assert_consistency(A, '4462732b6b354fde758586eec5930a54')

    def test_k3(self):
        operator = LinearOperatorKsparse(self.data, self.n_measurements, 3).generate()
        A = operator['A']
        y = operator['y']
        self._assert_shape(A, y)
        self._assert_finite(A, y)
        self._assert_ndarray(A, y)
        self._assert_nonzero_sum(A, y)
        assert_consistency(A, 'd9bb35c7903d0713d54e2c8a7b703279')

        
class TestLinearOperatorCustom(unittest.TestCase):
    """Test to verify that the blind linear operator and its measurements are generated correctly.

    Attributes
    ----------
    data : Data object
        Contains a dictionary with all the data that is needed for the linear operator and measurment creation.
    n_measurements : int (default = 901)
        Number of linear operators and measurements to be generated.
    shape : tuple of int, optional (unless mixed == None), default = (30, 40)
        Shape of the mixed, signal, bias and missing matrix in the form of (n_samples, n_features) of the data object.
    rank : int, (default = 2)
        Rank of the low-rank decomposition for the data object.
    """
    def setUp(self, seed=42):
        np.random.seed(seed)
        self.shape = (50, 60)
        self.n_measurements = 1000
        self.rank = 2
        #mixed = np.random.normal(0, 1, size=self.shape)
        #mixed[mixed < -1.0] = np.nan
        #self.mixed = mixed
        data = DataSimulated(self.shape, self.rank)
        data.estimate()
        self.data = data

    def _assert_shape(self, A, y):
        assert A.shape == (self.n_measurements, self.shape[0], self.shape[1])
        assert y.shape[0] == self.n_measurements

    def _assert_finite(self, A, y):
        assert np.isfinite(A).all() == True
        assert np.isfinite(y).all() == True

    def _assert_ndarray(self, A, y):
        assert type(A) == np.ndarray
        assert type(y) == np.ndarray

    def _assert_nonzero_sum(self, A, y):
        assert np.sum(A) != 0.0
        assert np.sum(y) != 0.0
        
    def test(self):
        operator = LinearOperatorCustom(self.data, self.n_measurements).generate()
        A = operator['A']
        y = operator['y']
        self._assert_shape(A, y)
        self._assert_finite(A, y)
        self._assert_ndarray(A, y)
        self._assert_nonzero_sum(A, y)
        assert_consistency('A', '7fc56270e7a70fa81a5935b72eacbe29')
        
if __name__ == '__main__':
    unittest.main()








        

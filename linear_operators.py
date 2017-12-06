"""Linear operator and measurement generation module.

Notes
-----
This module defines a class that generates the measurement operator A and measurement y from the input of the `data` module.
"""
from __future__ import division, absolute_import


__all__ = ['LinearOperatorEntry', 'LinearOperatorDense', 'LinearOperatorKsparse', 'LinearOperatorCustom', 'min_measurements', 'max_measurements']

import abc
import numpy as np





def min_measurements(shape):
    '''
    Computes the minimum number of measurements that are possible in an ideal case where all the underlying pairs can be detected correctly but the block structure is minimal (m_bocks=dimension/2). Both feature and sample space together!
    '''
    assert shape[0] % 2 == 0
    assert shape[1] % 2 == 0
    a, b = shape
    n = ((a / 2) * b) + ((b / 2) * a)
    return n


def max_measurements(shape):
    '''
    Computes the maximum number of measurements that are possible in an ideal case where all the underlying pairs can be detected correctly and the block structure is maximal (m_bocks=2). Both feature and sample space together!
    '''
    assert shape[0] % 2 == 0
    assert shape[1] % 2 == 0
    a, b = shape
    a_pairs = (a / 2)**2 - (a / 2)
    b_pairs = (b / 2)**2 - (b / 2)
    n = int((a_pairs * b) + (b_pairs * a))
    return n

    
def _print_size(name, X):
    """Print the memory footprint in MB of a particular data matrix.

    Parameters
    ----------
    name : str
        Name of data matrix.
    X : ndarray, shape (n_samples, n_features)
        Input data matrix.
    """
    print str(name), X.nbytes * 1.0e-6, 'MB'
    

def _pop_pairs_with_indices_randomly(n_pairs, n):
    """ Does popping randomdly and without replacement. WARNING Need to make sure that not using more unique indices then possible.
    """
    random_indices = np.arange(n * n_pairs, dtype=int)
    np.random.shuffle(random_indices)
    
    pair_indices = np.repeat(np.arange(n_pairs, dtype=int), n) # sample until randomly choosen all samples from it.
    sample_indices = np.tile(np.arange(n, dtype=int), n_pairs) # sample until empty.

    for i in random_indices:
        pair_index = pair_indices[i]
        sample_index = sample_indices[i]
        yield pair_index, sample_index

    
def _choose_random_matrix_elements(shape, n, duplicates=False):
    """Choose `n` random matrix element indices for a matrix with a given `shape`.

    Random element indices are never chosen more than once and the maximum number is the total number of elements in shape. The returned random element indices are always shuffled.

    If duplicates, or sparsity is != 1, then elements can/are chosen more than once. WARNING make consitent with all the other non-duplicate samplings!
    
    Parameters
    ----------
    shape : tuple of int
        Shape of the matrix of which to sample the random element indices from.
    n : int
        Number of element indices to choose (max. determined by shape)

    Returns
    -------
    element_indices : ndarray, shape (n, 2)
        Matrix indices of `n` elements.
    """
    size = shape[0] * shape[1]

    if duplicates == False:

        assert n <= size
        
        if n == size:
            element_indices = np.nonzero(np.ones(shape))
            element_indices = np.vstack(element_indices).T
            np.random.shuffle(element_indices)

        if n < size:
            temp = np.zeros(size)
            temp[:n] = 1
            np.random.shuffle(temp)
            temp = temp.reshape(shape)
            element_indices = np.nonzero(temp)
            element_indices = np.vstack(element_indices).T
            np.random.shuffle(element_indices)

    else:
        element_indices = []
        for i in xrange(n):
            x = np.random.randint(0, shape[0])
            y = np.random.randint(0, shape[1])
            element_indices.append([x, y])
        element_indices = np.asarray(element_indices)
        
    return element_indices

    
class LinearOperator(object):

    __metaclass__ = abc.ABCMeta
        
    def __init__(self, data):
        """Abstract base class for the generation of linear operators and their measurements.

        Parameters
        ----------
        data : Data object
                Contains a dictionary with all the data that is needed for the linear operator and measurment creation.
                
        """
        self.data = data

    @abc.abstractmethod
    def generate(self):
        """Generates the linear operators and corresponding measurements.

        Returns
        -------
        A : ndarray, shape (n_operators, n_samples, n_features)
            Linear operators.
        y : ndarray, shape (n_operators)
            Measurements.
        """
        pass

    
class LinearOperatorEntry(LinearOperator):

    def __init__(self, data, n_measurements):
        super(LinearOperatorEntry, self).__init__(data)
        self.n_measurements = n_measurements
        
    def generate(self):
        """Generate linear operators A and measurements y from entries of a signal matrix.

        Parameters
        ----------
        data : Data object
                Contains a dictionary with all the data that is needed for the linear operator and measurement creation.
        n_measurements : int
            Number of linear operators and measurements to be generated.

        Returns
        -------
        out : dict, {A: ndarray, shape (n_operators, n_samples, n_features)
                        Linear operators.
                     y : ndarray, shape (n_operators)
                        Measurements.}

        Note
        ----
        Does not work with missing values.
        """
        true_bias = self.data.d['sample']['true_bias']
        shape = true_bias.shape
        A = np.zeros((self.n_measurements, shape[0], shape[1]))
        A = np.array(A, dtype=int)
        y = np.zeros(self.n_measurements)
        random_element_indices = _choose_random_matrix_elements(true_bias.shape, self.n_measurements, duplicates=False) # WARNING Also here same element can be sampled multiple times if duplicatese is True. Handy for comparisons. But doesn'T make sense.
        for n, element_index in enumerate(random_element_indices):
            A[n, element_index[0], element_index[1]] = 1
            y[n] = true_bias[element_index[0], element_index[1]]
        _print_size('A', A)
        _print_size('y', y)
        out = {'A': A, 'y': y}
        return out


class LinearOperatorDense(LinearOperator):

    def __init__(self, data, n_measurements):
        super(LinearOperatorDense, self).__init__(data)
        self.n_measurements = n_measurements

    def generate(self):
        """Generate linear operators A and measurements y from a dense sampling of a signal matrix.

        Parameters
        ----------
        data : Data object
            Contains a dictionary with all the data that is needed for the linear operator and measurement creation.
        n_measurements : int
            Number of linear operators and measurements to be generated.

        Returns
        -------
        out : dict, {A: ndarray, shape (n_operators, n_samples, n_features)
                        Linear operators.
                     y : ndarray, shape (n_operators)
                        Measurements.}

        Note
        ----
        Does not work with missing values.
        """
        true_bias = self.data.d['sample']['true_bias']
        shape = true_bias.shape
        A = np.zeros((self.n_measurements, shape[0], shape[1]))
        A = np.array(A, dtype=float)
        y = np.zeros(self.n_measurements)
        for n in xrange(self.n_measurements):
            A_i = np.random.normal(0.0, 2.0, size=shape) # WARNING, could be too small or too large for float and subsequent computations, but unlikely.
            y_i = np.sum(A_i * true_bias)
            A[n, :, :] = A_i 
            y[n] = y_i
        _print_size('A', A)
        _print_size('y', y)
        out = {'A': A, 'y': y}
        return out


class LinearOperatorKsparse(LinearOperator):

    def __init__(self, data, n_measurements, sparsity):
        super(LinearOperatorKsparse, self).__init__(data)
        self.n_measurements = n_measurements
        self.sparsity = sparsity
        
    def generate(self):
        """Generate linear operators A and measurements y from a dense sampling of a signal matrix.

        Parameters
        ----------
        data : Data object
            Contains a dictionary with all the data that is needed for the linear operator and measurement creation.
        n_measurements : int
            Number of linear operators and measurements to be generated.
        sparsity : int
            Sparsity of the measuremnt operator, e.g. if 2-sparse then only 2 entries in `A_i` are non-zero.
            
        Returns
        -------
        out : dict, {A: ndarray, shape (n_operators, n_samples, n_features)
                        Linear operators.
                     y : ndarray, shape (n_operators)
                        Measurements.}

        Note
        ----
        # Does not work with missing values. Same pair can be sampled multiple times (but within a pair elements can't be the same).

        # TODO shuffle the pairs and still do estimation (make sure those are then pairs which are false or that give them completly new A_i matrices.
        # TODO add noise to y to simulate the redundancy.
        """
        true_bias = self.data.d['sample']['true_bias']
        shape = true_bias.shape
        A = np.zeros((self.n_measurements, shape[0], shape[1]))
        A = np.array(A, dtype=float)
        y = np.zeros(self.n_measurements)
        for n in xrange(self.n_measurements):
            A_i = np.zeros(shape)
            indices = _choose_random_matrix_elements(shape, self.sparsity, duplicates=False) # Do not want to implement possible pairs for all sparsities; therefore just using random sampling here.
            values = np.random.normal(0.0, 2.0, size=self.sparsity)
            for k in xrange(self.sparsity):
                A_i[indices[k][0], indices[k][1]] = values[k]
            y_i = np.sum(A_i * true_bias)
            A[n, :, :] = A_i
            y[n] = y_i
        _print_size('A', A)
        _print_size('y', y)
        out = {'A': A, 'y': y}
        return out

        
class LinearOperatorCustom(LinearOperator):

    def __init__(self, data, n_measurements):
        super(LinearOperatorCustom, self).__init__(data)
        self.n_measurements = n_measurements
        
    def _solve(self, a, b, d):
        """Solve a linear equation ay + bx + c for c.
        """
        c = -(a * d[0] + b * d[1])
        return c

    def _distance(self, a, b, x0, y0):
        """Distance from point (x0, y0) to line y = mx + 0.0

        Source
        ------
        https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        """
        m = -b / float(a)
        x = (x0 + m * y0) / float(m**2 + 1)
        y = m * x
        d = np.asarray([y - y0, x - x0])
        return d

    def _construct_measurement(self, pair, j, stds, direction, space):
        """Workhorse functon to create linear operator A and measurement y.
        """
        A = np.zeros(self.data.d[space]['shape'])
        std_b, std_a = stds
        signed_std_b = -direction * std_b
        y0, x0 = self.data.d[space]['mixed'][pair, j]
        d = self._distance(std_a, signed_std_b, x0, y0)
        c = self._solve(std_a, signed_std_b, d)
        A[pair[0], j] = std_a
        A[pair[1], j] = signed_std_b
        y = c
        if space == 'feature':
            A = A.T
        return A, y
        
    def generate(self):
        """Generate linear operators A and measurements y from a dense sampling of a signal matrix.

        Parameters
        ----------
        data : Data object
            Contains a dictionary with all the data that is needed for the linear operator and measurement creation.
        n_measurements : int
            Number of linear operators and measurements to be generated.

        Returns
        -------
        out : dict, {A: ndarray, shape (n_operators, n_samples, n_features)
                        Linear operators.
                     y : ndarray, shape (n_operators)
                        Measurements.}

        Note
        ----
        Does not work with missing values and only makes sense for sparsity = 2.
        # TODO fast multiplication and addition and low-memory storage of sparse operators!
        # TODO Incorrect stuff, e.g. return a proportion of incorrect and correct pairs.
        # TODO Initial case of non-perfect correlations (set during signal construction!)
        """
        A = np.zeros((self.n_measurements, self.data.d['sample']['shape'][0], self.data.d['sample']['shape'][1]))
        A = np.array(A, dtype=float)
        y = np.zeros(self.n_measurements)

        indices = {'sample': _pop_pairs_with_indices_randomly(len(self.data.d['sample']['estimated_pairs']), self.data.d['sample']['shape'][1]), 'feature': _pop_pairs_with_indices_randomly(len(self.data.d['feature']['estimated_pairs']), self.data.d['feature']['shape'][1])}
        
        for n in xrange(self.n_measurements):
            
            space = np.random.choice(['feature', 'sample']) # TODO here pop those that have not been samples randomly only. REally can do randomly and just pop downstream.

            #index = np.random.choice(np.arange(len(self.data.d[space]['estimated_pairs'])))# TODO here pop those that have not been samples randomly only.

            index, j = indices[space].next()
            
            pair = self.data.d[space]['estimated_pairs'][index]
            stds = self.data.d[space]['estimated_stds'][index]
            direction = self.data.d[space]['estimated_directions'][index]

            #j = np.random.choice(np.arange(self.data.d[space]['mixed'].shape[1])) # TODO here pop those that have not been samples randomly only.

            assert np.isfinite(stds[0])
            assert np.isfinite(stds[1])
            assert np.isfinite(direction)
            if np.isfinite(self.data.d[space]['mixed'][pair[0], j]) == False:
                continue
            if np.isfinite(self.data.d[space]['mixed'][pair[1], j]) == False:
                continue
            A_i, y_i = self._construct_measurement(pair, j, stds, direction, space)
            A[n, :, :] = A_i # TODO could be made into sparse matrix and used like that with solver to save memory. Check duplicates?
            y[n] = y_i
        _print_size('A', A)
        _print_size('y', y)
        out = {'A': A, 'y': y}
        return out


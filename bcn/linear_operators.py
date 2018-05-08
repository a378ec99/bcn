"""Linear operator and measurement generation.

Notes
-----
Defines classes that generate the measurement operator A and measurement y from the input of a Data object.
"""
from __future__ import division, absolute_import

import abc
import numpy as np

from scipy.sparse import issparse, coo_matrix
from scipy.special import comb


def sample_n_choose_k(n_samples, n_features, k, n):
    """
    Source
    ------
    https://stats.stackexchange.com/questions/315087/how-can-one-generate-a-sequence-of-unique-k-sparse-matrices-without-rejection-sa
    """
    assert n_samples * n_features >= k
    max_ = comb(n_samples * n_features, k, exact=True)
    assert max_ >= n
    samples = []
    while len(samples) < n:
        i = np.random.randint(max_)
        if not i in samples:
            samples.append(i)
    return samples

    
def integer_to_matrix(i, k, n_samples, n_features):
    """
    Use the combinatorial number system to map from i \in {1,2,..., N_choose_k} to corresponding N-dimensional binary vector with k ones.

    See: https://en.wikipedia.org/wiki/Combinatorial_number_system#Finding_the_k-combination_for_a_given_number

    Source
    ------
    https://stats.stackexchange.com/questions/315087/how-can-one-generate-a-sequence-of-unique-k-sparse-matrices-without-rejection-sa
    """
    flat_matrix = np.zeros(n_samples * n_features, dtype=int)
    for h in range(k, 0, -1): # h means how many entries still to set to 1
        j = h - 1
        found = False
        while not found:
            testvalue = comb(j, h, exact=True) # Its first value is 0 == ((h-1) choose h)
            if testvalue == i:
                found = True
                flat_matrix[j] = 1
                i = i - testvalue
            elif testvalue > i:
                found = True
                flat_matrix[j - 1] = 1
                i = i - testvalue_prev
            else:
                j = j + 1
                testvalue_prev = testvalue
    matrix = flat_matrix.reshape((n_samples, n_features))
    return matrix

        
def possible_measurement_range(shape, missing_fraction):
    '''
    Computes the range of the possible number of measurements for a particular shaped matrix.

    Parameters
    ----------
    shape : (int, int)
        Dimensions of the array to be recovered.
    missing_fraction : float
        Fraction of missing values in mixed matrix to be used for recovery.
    Returns
    -------
    n_worst_case, n_best_case : (int, int)
        Number of measurements possible in the worst case and the best case.
    
    Note
    ----
    Possible measurements are defined by the block structure of the correlation matrix. Worst case block structure (m_bocks = dimension/2) and best case block structure (m_blocks = 2).
    Optimal detection of all potential pairs is assumed and both feature and sample space are summed for the final number of possible measurements.
    '''
    assert shape[0] % 2 == 0
    assert shape[1] % 2 == 0
    a, b = shape

    n_worst_case = int(((a / 2) * b) + ((b / 2) * a))
    n_worst_case  = n_worst_case - (missing_fraction * n_worst_case)
    
    a_pairs = (a / 2)**2 - (a / 2)
    b_pairs = (b / 2)**2 - (b / 2)
    n_best_case = int((a_pairs * b) + (b_pairs * a))
    n_best_case = n_best_case - (missing_fraction * n_best_case)
    
    return n_worst_case, n_best_case


def _print_size(name, X):
    """Print the memory footprint in MB of a particular data matrix.

    Parameters
    ----------
    name : str
        Name of data matrix.
    X : ndarray, shape (n_samples, n_features)
        Input data matrix.
    """
    if issparse(X):
        print '{}: {} MB'.format(name, (X.data.nbytes + X.row.nbytes + X.col.nbytes) * 1.0e-6)
    else:
        print '{}: {} MB'.format(name, X.nbytes * 1.0e-6)


def _pop_pairs_with_indices_randomly(n_pairs, n, max_pops):
    """ Does popping of indices to create measurements randomly and without replacement.

    Parameters
    ----------
    n_pairs : int
        Number of pairs in feature or sample space.
    n : int
        Number of entries for a particular pair.
    max_pops : int
        Number of maximal calls to this generator. Need to make sure that not trying to pop more unique indices then possible.
        
    Yields
    ------
    pair_index, sample_index : (int, int)
        Index for a particular pair and a particular entry for that pair.
    """
    assert max_pops <= n * n_pairs
    
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

    Parameters
    ----------
    shape : tuple of int
        Shape of the matrix of which to sample the random element indices from.
    n : int
        Number of element indices to choose (max. determined by shape)
    duplicates : bool
        Allow duplicate elements or not. If duplicates, or sparsity is != 1, then elements can be chosen more than once.
    
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
        
    def __init__(self):
        """Abstract base class for the generation of linear operators and corresponding targets.
        """
        pass
    
    @abc.abstractmethod
    def generate(self):
        """Generates linear operators and corresponding targets.

        Returns
        -------
        A : dict, elements=list, len=n_measurements
            Linear operator stored as sparse matrices.
        y : list, elements=float, len=n_measurements
            Target vector.
        """
        pass

    
class LinearOperatorEntry(LinearOperator):

    def __init__(self, n):
        """
        Parameters
        ----------
        n : int
            Number of linear operators / targets to be generated, e.g. measurements.
        """
        self.n = n
        
    def generate(self, signal):
        """Generate linear operators A and targets y from entry sampling of a clean signal matrix.

        Parameters
        ----------
        signal : numpy.ndarray, shape=(n_samples, n_features)
            Clean signal matrix to sample the known entries from.
            
        Returns
        -------
        measurements : dict, elements={A : dict, elements=list, len=n
                                        Linear operator stored as sparse matrices.
                                       y : list, elements=float, len=n
                                        Target vector.}

        Note
        ----
        Allows for nan values in signal matrix.
        """
        A, y = [], []
        indices = _choose_random_matrix_elements(signal.shape, self.n, duplicates=False) # NOTE Also here same element can be sampled multiple times if duplicatese is True.
        for index in indices:
            entry = signal[index[0], index[1]]
            if np.isfinite(entry):
                A.append({'row':[index[0]], 'col': [index[1]], 'value': [1.0]})
                y.append(entry)
        measurements = {'A': A, 'y': y}
        return measurements


class LinearOperatorDense(LinearOperator):

    def __init__(self, n):
        """
        Parameters
        ----------
        n : int
            Number of linear operators / targets to be generated, e.g. measurements.
        """   
        self.n = n

    def generate(self, signal):
        """Generate linear operators A and targets y from dense sampling of a clean signal matrix.

        Parameters
        ----------
        signal : numpy.ndarray, shape=(n_samples, n_features)
            Clean signal matrix to sample the known entries from.

        Returns
        -------
        measurements : dict, elements={A : dict, elements=list, len=n
                                        Linear operator stored as sparse matrices.
                                       y : list, elements=float, len=n
                                        Target vector.}

        Note
        ----
        Allows for nan values in signal matrix.
        """
        A, y = [], []
        for n in xrange(self.n):
            A_i_original = np.random.standard_normal(signal.shape)
            A_i = coo_matrix(A_i_original, copy=True)
            A.append({'row': list(A_i.row), 'col': list(A_i.col), 'value': list(A_i.data)})
            y_i = np.nansum(A_i_original * signal)
            y.append(y_i)
        measurements = {'A': A, 'y': y}
        return measurements


class LinearOperatorKsparse(LinearOperator):

    def __init__(self, n, sparsity):
        """
        Parameters
        ----------
        n : int
            Number of linear operators and measurements to be generated.
        sparsity : int
            Sparsity of the measuremnt operator, e.g. if 2-sparse then only 2 entries in `A_i` are non-zero.
        """
        self.n = n
        self.sparsity = sparsity
        
    def generate(self, signal):
        """Generate linear operators A and measurements y from a dense sampling of a signal matrix.

        Parameters
        ----------
        signal : numpy.ndarray, shape=(n_samples, n_features)
            Clean signal matrix to sample the known entries from.
            
        Returns
        -------
        measurements : dict, elements={A : dict, elements=list, len=n
                                        Linear operator stored as sparse matrices.
                                       y : list, elements=float, len=n
                                        Target vector.}
        """
        A, y = [], []
        for i in sample_n_choose_k(signal.shape[0], signal.shape[1], self.sparsity, self.n):
            A_i = integer_to_matrix(i, self.sparsity, signal.shape[0], signal.shape[1])
            A_i = coo_matrix(A_i)
            if np.isfinite(signal[A_i.row, A_i.col]).any():
                value = np.random.standard_normal(self.sparsity)
                A.append({'row': list(A_i.row), 'col': list(A_i.col), 'value': value})
                y_i = np.nansum(value * signal[A_i.row, A_i.col])
                y.append(y_i)
            else:
                continue
        measurements = {'A': A, 'y': y}
        return measurements

        
class LinearOperatorCustom(LinearOperator):

    def __init__(self, data, n_measurements):
        super(LinearOperatorCustom, self).__init__(data)
        self.n_measurements = n_measurements
        
    def _solve(self, a, b, d):
        """Solve a linear equation ay + bx + c for c.

        Parameters
        ----------
        a : float
            First variable of the linear equation based on the standard deviation of a.
        b : float
            Second variable of the linear equation based on the signed standard deviation of b.
        d : (float, float)
            Encoded by the distance d is the value for x and y needed to selvoe the equation.

        Returns
        -------
        c : float
            Variable solved for.
        """
        c = -(a * d[0] + b * d[1])
        return c

    def _distance(self, a, b, x0, y0):
        """Distance from point (x0, y0) to line y = mx + 0.0

        Parameters
        ----------
        a : float
            Standard deviation of feature/sample a.
        b : float
            Standard deviation of feature/sample b with sign (direction).
        x0 : float
            The x-coordinate of a point somewhere off the ideal line.
        y0 : float
            The y-coordinate of a point somewhere off the ideal line.

        Returns
        -------
        d : float
            Distance from point (x0, y0) to line y = mx + 0.0.
            
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
        """Creates linear operator A and measurement y in the framework of compressed sensing.

        Parameters
        ----------
        pair : (int, int)
            A pair of features or samples that are strongly dependent.
        j : int
            A particular entry of the pair.
        stds :
            Standard deviations of the pair.
        direction :
            Direction of the dependency, e.g. -1 anticorrelated.
        space : {'feature', 'sample'}
            The space of the array to be used.

        Returns
        -------
        A, y : 2d-array, 1d-array
            Measurment operator, measurements.
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
        out : dict, {A: ndarray, shape (n_measurements, n_samples, n_features)
                        Linear operators.
                     y : ndarray, shape (n_measurements)
                        Measurements.}
        """
        A = []
        y = []

        indices = {'sample': _pop_pairs_with_indices_randomly(len(self.data.d['sample']['estimated_pairs']), self.data.d['sample']['shape'][1], self.n_measurements // 2), 'feature': _pop_pairs_with_indices_randomly(len(self.data.d['feature']['estimated_pairs']), self.data.d['feature']['shape'][1], self.n_measurements // 2)}
        
        for n in xrange(self.n_measurements):
            space = np.random.choice(['feature', 'sample'])
            index, j = indices[space].next()
            pair = self.data.d[space]['estimated_pairs'][index]
            stds = self.data.d[space]['estimated_stds'][index]
            direction = self.data.d[space]['estimated_directions'][index]

            assert np.isfinite(stds[0])
            assert np.isfinite(stds[1])
            assert np.isfinite(direction)

            # NOTE Checking for nan and inf values in measured data matrix.
            if np.isfinite(self.data.d[space]['mixed'][pair[0], j]) == False:
                continue
            if np.isfinite(self.data.d[space]['mixed'][pair[1], j]) == False:
                continue
            
            A_i, y_i = self._construct_measurement(pair, j, stds, direction, space)
            A_i = coo_matrix(A_i)
            A.append({'row': list(A_i.row), 'col': list(A_i.col), 'value': list(A_i.data)})
            y.append(y_i)
        out = {'A': A, 'y': y}
        return out


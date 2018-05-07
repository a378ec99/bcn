"""Linear operator and measurement generation.

Notes
-----
Defines classes that generate the measurement operator A and measurement y from the input of a Data object.
"""
from __future__ import division, absolute_import

import abc
import numpy as np

from scipy.sparse import issparse, coo_matrix


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
        A : ndarray, shape (n_measurements, n_samples, n_features)
            Linear operators.
        y : ndarray, shape (n_measurements)
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
        out : dict, {A: ndarray, shape (n_measurements, n_samples, n_features)
                        Linear operators.
                     y : ndarray, shape (n_measurements)
                        Measurements.}

        Note
        ----
        Does not work with missing values.
        """
        true_bias = self.data.d['sample']['true_bias']
        shape = true_bias.shape
        A = []
        y = []
        random_element_indices = _choose_random_matrix_elements(true_bias.shape, self.n_measurements, duplicates=False) # NOTE Also here same element can be sampled multiple times if duplicatese is True. That setting might be handy for comparisons, but doesn't make sense in the bigger picture.
        for element_index in enumerate(random_element_indices):
            A.append({'row':[element_index[0]], 'col': [element_index[1]], 'value': [1.0]})
            y.append(true_bias[element_index[0], element_index[1]])
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
        out : dict, {A: ndarray, shape (n_measurements, n_samples, n_features)
                        Linear operators.
                     y : ndarray, shape (n_measurements)
                        Measurements.}

        Note
        ----
        Does not work with missing values.
        """
        true_bias = self.data.d['sample']['true_bias']
        shape = true_bias.shape
        A = []
        y = []
        for n in xrange(self.n_measurements):
            A_i = np.random.normal(0.0, 2.0, size=shape) # WARNING, could be too small or too large for float and subsequent computations, but unlikely.
            A_i = coo_matrix(A_i)
            A.append({'row': list(A_i.row), 'col': list(A_i.col), 'value': list(A_i.data)})
            y_i = np.sum(A_i * true_bias)
            y.append(y_i)
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
        out : dict, {A: ndarray, shape (n_measurements, n_samples, n_features)
                        Linear operators.
                     y : ndarray, shape (n_measurements)
                        Measurements.}

        Note
        ----
        Does not consider missing values. Directly based on true bias. Same pair can be sampled multiple times (but within a pair elements can't be the same).
        """
        true_bias = self.data.d['sample']['true_bias']
        shape = true_bias.shape
        A = []
        y = []
        for n in xrange(self.n_measurements):            
            indices = _choose_random_matrix_elements(shape, self.sparsity, duplicates=False)
            values = np.random.normal(0.0, 2.0, size=self.sparsity)
            A_i_row = list(indices[:, 0])
            A_i_col = list(indices[:, 1])
            A_i_data = list(values)
            A.append({'row': A_i_row, 'col': A_i_col, 'value': A_i_data})
            y_i = np.sum(A_i * true_bias)
            y.append(y_i)
        out = {'A': A, 'y': y}
        return out

        
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


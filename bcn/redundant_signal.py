"""Redundant signal matrix.
"""
from __future__ import division, absolute_import

import numpy as np


def _convert_space_to_shape_index(space):
    """Map from `feature` to 0 and `sample` to 1 (for appropriate shape indexing).

    Parameters
    ----------
    space : str, values=('feature', 'sample')
        Feature or sample space.
        
    Returns
    -------
    return : int, values=(0, 1)
        Index location of the desired space.
    """
    if space == 'feature':
        return 1
        
    if space == 'sample':
        return 0
        

def _generate_square_blocks_matrix(n, m_blocks, r=1.0, step=0):
    """Generates a square block type correlation matrix.

    Parameters
    ----------
    n : int
        Dimensions of the output matrix (n, n).
    m_blocks : int
        The number of square blocks in the correlation matrix. Does not have to be an even fit into n.
    r : float
        Correlation strength.
    step : int, values=(0, 1)
        Indicates the depth of recursion to create a proper square blocks matrix; internal use only.
    
    Note
    ----
    Blocks contain half and half correlated (r) and anticorrelated (-r) features and samples.

    Returns
    -------
    block_matrix : numpy.ndarray, shape=(n, n)
        Square blocks matrix.
    """
    assert r >= 0
    assert n >= m_blocks

    blocksizes = np.repeat(n // m_blocks, m_blocks) # WARNING some problem if block size == n
    blocksizes[-1] = blocksizes[-1] + (n % m_blocks)

    if step == 0:
        block_matrix = np.zeros((n, n))
    if step == 1:
        block_matrix = -1 * np.ones((n, n)) * r

    for i, size in enumerate(blocksizes):
        if step == 0:
            square = _generate_square_blocks_matrix(size, 2, r=r, step=1)
        if step == 1:
            square = np.ones((size, size)) * r
            di = np.diag_indices(size)
            square[di] = 1.0
        indices = np.indices(square.shape)
        indices = np.vstack([indices[0].ravel(), indices[1].ravel()]).T
        for index in indices:
            block_matrix[index[0] + (i * blocksizes[0]), index[1] + (i * blocksizes[0])] = square[index[0], index[1]]

    return block_matrix

    
def _generate_pairs(correlation_matrix):
    """Generate index pairs for correlated features.

    Parameters
    ----------
    correlation_matrix: numpy.ndarray, shape=(n_features, n_features) or (n_samples, n_samples)
        Correlation matrix with a square block structure.

    Returns
    -------
    pairs : numpy.ndarray, shape=(<= n_features, 2)
        Shuffled indices of correlated features.
    """
    pairs = np.vstack(np.nonzero(np.tril(correlation_matrix, -1))).T
    indices = np.arange(len(pairs))
    np.random.shuffle(indices)
    pairs = np.asarray(pairs)
    
    assert len(indices) <= len(pairs)
    
    pairs = pairs[indices]
    return pairs

    
def _generate_matrix_normal_sample(U, V):
    """Generate a sample from a matrix variate normal distribution.

    Parameters
    ----------
    U : numpy.ndarray, shape=(n_samples, n_samples)
        Covariance matrix among samples (rows).
    V : numpy.ndarray, shape=(n_features, n_features)
        Covariance matrix among features (columns).
        
    Returns
    -------
    Y : numpy.ndarray, shape=(n_samples, n_features)
        A sample from a matrix variate normal distribution.

    Source
    ------
    https://en.wikipedia.org/wiki/Matrix_normal_distribution#Drawing_values_from_the_distribution
    """
    n = len(U)
    m = len(V)
    X = np.random.standard_normal((n, m))
    u, s, vt = np.linalg.svd(U)
    Us = np.dot(u, np.diag(np.sqrt(s)))
    u, s, vt = np.linalg.svd(V)
    Vs = np.dot(u, np.diag(np.sqrt(s)))
    Y = np.dot(np.dot(Us, X), Vs.T)
    return Y


def _generate_stds(n, model, std_value=1.5, normalize=False):
    """Generate standard deviations according to two models.

    Parameters
    ----------
    n : int
        Feature/sample dimensions of the matrix for which the standard deviations are computed.
    model : str, values=('random', 'constant')
        The type of model to generate standard deviations. Constant uses the std_value, whereas random samples from a uniform distribution between 0.1 and 2.
    std_value : float
        If model == 'constant' then this defines the standard deviation of all features/samples.
    normalize : bool
        Whether to normalize the sum of standard deviations to one, so that the generation of the matrix variate normal sample is simpler.

    Returns
    -------
    stds, scaling_factor : numpy.ndarray, float
        A tuple of scaled standard deviations and the corresponding scaling factor (if normalize == True).
    """
    if model == 'random':
        stds = np.random.uniform(0.1, 2.0, n) # 0.01, 20.0
    if model == 'constant':
        stds = np.repeat(std_value, n)
    if normalize:
        scaling_factor = float(np.sqrt(1 / float(np.sum(stds**2))))
        stds = stds * scaling_factor
        return stds, scaling_factor
    else:
        return stds, None

    
def _generate_directions(correlation_matrix, pairs):
    """Generate directions from a signed correlation matrix.
    
    Parameters
    ----------
    correlation_matrix : numpy.ndarray, shape=(n_samples, n_samples) or (n_features, n_features)
        Signed correlation matrix from which the signs are extracted as directions.
    pairs : numpy.ndarray, len=n_pairs
        The pairs of interest for which the directions are to be extraced.
        
    Returns
    -------
    directions : numpy.ndarray, len=n_pairs
        The directions of the correlations based on the signs given by the correlation matrix for the pairs of interest.
    """
    directions = np.sign(correlation_matrix[pairs[:, 0], pairs[:, 1]])
    return directions

    
def _generate_covariance(correlation_matrix, stds):
    """Generate a covariance matrix based on a given correlation matrix and the corresponing feature/sample standard deviations.

    Parameters
    ----------
    correlation_matrix : numpy.ndarray, shape=(n_samples, n_samples) or (n_features, n_features)
        Correlation matrix on which the covariance matrix is based.
    stds : numpy.ndarray, len=n_stds
        Standard deviations which are used in combination with the correlation amtrix to generate the covariance matrix.
        
    Returns
    -------
    covariance_matrix : numpy.ndarray, shape=(n_samples, n_samples) or (n_features, n_features)
    """
    covariance_matrix = np.outer(stds, stds) * correlation_matrix
    return covariance_matrix

    
class RedundantSignal(object):

    def __init__(self, shape, model, m_blocks, correlation_strength, std_value=None, normalize_stds=False):
        """Generate a high-dimensional signal with a particular redundancy.

        Parameters
        ----------
        shape : tuple of int
            Shape of the output signal matrix in the form of (n_samples, n_features).
        model : str, values=('random', 'constant')
            The redundancy model specifies if the standard deviations in feature and sample space are random, or a specific constant.
        m_blocks : int
            Number of blocks in the correlation matix of features or samples that are varying together (with differences only in degree, direction and scale). Fewer blocks are better for bias recovery.
        correlation_strength : float 
            The strength of correlations between features or samples. Stronger correlations are better for bias recovery.
        std_value : float, optional unless model == 'constant'.
            Value to use as a constant standard deviation for both sample and feature space.
        normalize_stds : bool
            Whether to normalize the standard deviations to sum to one in feature and sample space, respetively. This makes the generation of the matrix-variate normal sample simpler.
        """
        self.shape = shape
        self.model = model
        self.m_blocks = m_blocks
        self.correlation_strength = correlation_strength
        self.std_value = std_value
        self.normalize_stds = normalize_stds
        
        assert self.model in ['random', 'constant']
        
    def generate(self):
        """Generate a high-dimensional signal with a particular redundancy.

        Returns
        -------
        signal : dict, {'X': numpy.ndarray, shape=(n_sample, n_features),
                        'features: { 'pairs': ndarray, shape (n_pairs, 2),
                                     'correlation_matrix': numpy.ndarray, shape=(n_features, n_features)
                                     'stds': numpy.ndarray, len=n_pairs
                                     'directions': numpy.ndarray, len=n_pairs
                                     'covariance_matrix': numpy.ndarray, shape=(n_features, n_features)
                                     'scaling_factor_stds': float
                        'samples': ...
                       }
            Contains signal matrix X and the corresponding details for its creation.
        """
        signal = {}
        for space in ['feature', 'sample']:
            correlation_matrix = _generate_square_blocks_matrix(self.shape[_convert_space_to_shape_index(space)], self.m_blocks, r=self.correlation_strength)
            pairs = _generate_pairs(correlation_matrix)
            directions = _generate_directions(correlation_matrix, pairs)
            stds, scaling_factor = _generate_stds(self.shape[_convert_space_to_shape_index(space)], self.model, std_value=self.std_value, normalize=self.normalize_stds)
            covariance_matrix = _generate_covariance(correlation_matrix, stds)
            signal[space] = {'pairs': pairs,
                             'correlation_matrix': correlation_matrix,
                             'stds': stds,
                             'directions': directions,
                             'covariance_matrix': covariance_matrix,
                             'scaling_factor': scaling_factor}
        U, V = signal['sample']['covariance_matrix'], signal['feature']['covariance_matrix']
        X = _generate_matrix_normal_sample(U, V)
        signal['X'] = X
        return signal

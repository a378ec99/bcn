"""The Data class is defined here.

Notes
-----
Defines a class that is used thoughout for storage, processing and visualization of all data.
"""
from __future__ import division, absolute_import

import abc
from copy import deepcopy

import numpy as np
from scipy.stats import pearsonr

from bcn.bias import BiasLowRank, BiasUnconstrained
from bcn.redundant_signal import RedundantSignal
from bcn.missing import Missing


def estimate_partial_signal_characterists(mixed, correlation_threshold, true_pairs=None, true_directions=None, true_stds=None, true_correlations=None):
    """Estimate correlations, pairs, directions and strandard deviations from a corrupted signal.

    Parameters
    ----------
    mixed : numpy.ndarray, shape=(n_samples, n_features)
        The bias corrupted low-rank matrix from which the bias is to be recovered.
    correlation_threshold : float
        The threshold to use when estimating pairs from a correlation matrix (the higher the fewer pairs).
    true_pairs : numpy.ndarray, elements=int, shape=(n, 2)
        Sequence of true pairs given as tuples.
    true_directions : numpy.ndarray, elements=int, len=n
        Sequence of true directions, e.g. -1, +1.
    true_stds : numpy.ndarray, elements=int, shape=(n, 2)
        Sequence of true standard deviations of each pair.
    """
    estimates = {'feature': {'mixed': mixed.T, 'shape': mixed.T.shape}, 'sample': {'mixed': mixed, 'shape': mixed.shape}}
    
    for space in ['feature', 'sample']: 
        if true_correlations is not None:
            estimates[space]['estimated_correlations'] = true_correlations[space]
        else:
            estimates[space]['estimated_correlations'] = estimate_correlations(estimates[space]['mixed'])
        if true_pairs is not None:
            estimates[space]['estimated_pairs'] = true_pairs[space]
        else:
            estimates[space]['estimated_pairs'] = estimate_pairs(estimates[space]['estimated_correlations'], correlation_threshold)
        if true_stds is not None:
            estimates[space]['estimated_stds'] = true_stds[space]
        else:
            estimates[space]['estimated_stds'] = estimate_stds(estimates[space]['mixed'], estimates[space]['estimated_pairs'])
        if true_directions is not None:
            estimates[space]['estimated_directions'] = true_directions[space]
        else:
            estimates[space]['estimated_directions'] = estimate_directions(estimates[space]['estimated_correlations'], estimates[space]['estimated_pairs'])

    return estimates


def transpose_view(X, space):
    """Small helper function to return transpose of in imput matrix if required.

    Parameters
    ----------
    X: ndarray, shape (n_samples, n_features)
        A matrix that may need to be transposed (view only).
    space : {'sample', 'feature'}
        The space the matrix should be for (determines if transpossed or not).

    Returns
    -------
    X_transpose : ndarray, shape (n_features, n_samples) or (n_samples, n_features)
        The possibly transposed inpute matrix `X`.
    """
    if space == 'feature':
        X_transpose = X.T

    if space == 'sample':
        X_transpose = X
        
    return X_transpose

    
def opposite(space):
    """Convert to oppostie dimension.

    Parameters
    ----------
    space : str, {'feature', 'sample'}
        Dimension. 

    Returns
    -------
    space : {'feature', 'sample'}
        Dimension.
    """
    if space == 'feature':
        return 'sample'
    if space == 'sample':
        return 'feature'
        
    
def estimate_pairs(correlations, threshold=0.8):
    """Estimate pairs from a correlation matrix.

    Parameters
    ----------
    correlations : ndarray, shape (n_samples, n_samples)
        A correlation matrix. Can contain `nan` values.
    threshold : float
        The threshold below which correlations are not considered as pairs.

    Returns
    -------
    pairs : ndarray, shape (n <= n_samples, 2)
        A sequence of pairs which contain the indices of samples that are strongly correlated (as determined by the threshold).
    """
    correlations = np.nan_to_num(correlations)
    correlations[np.absolute(correlations) < threshold] = 0
    pairs = np.vstack(np.nonzero(np.tril(correlations, -1))).T
    indices = np.arange(len(pairs))
    np.random.shuffle(indices)
    pairs = np.asarray(pairs)
    pairs = pairs[indices]
    return pairs

    
def estimate_correlations(mixed):
    """Estimate correlations from a `mixed` matrix.

    Parameters
    ----------
    mixed : ndarray, shape (n_samples, n_features)
        A matrix that requires bias removal. Can contain `nan` values.

    Returns
    -------
    correlations : ndarray, shape (n_samples, n_samples)
    """
    correlations = np.zeros((mixed.shape[0], mixed.shape[0])) * np.nan
    for i, a in enumerate(mixed):
        bool_indices_a = np.isfinite(a)
        for j, b in enumerate(mixed):
            if i == j:
                correlations[i, j] = 1
            else:
                bool_indices_b = np.isfinite(b)
                bool_indices = np.logical_and(bool_indices_a, bool_indices_b)
                if np.sum(bool_indices) < 3:
                    continue
                else:
                    r = pearsonr(a[bool_indices], b[bool_indices])[0]
                    correlations[i, j] = r
    return correlations


def estimate_directions(correlations, pairs):
    """Estimate directions from a correlation matrix for specific pairs.

    Parameters
    ----------
    correlations : ndarray (n_samples, n_samples)
        A correlation matrix. Can contain nan values.
    pairs : ndarray, shape (n < n_samples, 2)
        A sequence of pairs which contain the indices of samples that are strongly correlated.

    Returns
    -------
    directions : ndarray, shape (n < n_samples)
        A sequence of -1/+1 which indicates the direction of the correlation (e.g. anti or normal).
    """
    directions = np.sign(correlations[pairs[:, 0], pairs[:, 1]])
    return directions

    
def estimate_stds(mixed, pairs):
    """Estimate standard deviations from a mixed` matrix for specific pairs.

    Parameters
    ----------
    mixed : ndarray, shape (n_samples, n_features)
        A matrix that requires bias removal. Can contain `nan` values.
    pairs : ndarray, shape (n < n_samples, 2)
        A sequence of pairs which contain the indices of samples that are strongly correlated.

    Returns
    -------
    stds : ndarray, shape (n < n_samples)
        A sequence of estimated standard deviations.
    """
    stds = []
    for pair in pairs:
        bool_indices_a = np.isfinite(mixed[pair[0]])
        std_a = np.std(mixed[pair[0]][bool_indices_a])
        if np.sum(bool_indices_a) < 3: # NOTE No need to check because there would be no pair if there were not 3 overlapping finite values for the pair (see estimate correlations).
            std_a = np.nan
        bool_indices_b = np.isfinite(mixed[pair[1]])
        std_b = np.std(mixed[pair[1]][bool_indices_b])
        if np.sum(bool_indices_b) < 3:
            std_b = np.nan
        stds.append([std_a, std_b])
    stds = np.vstack(stds)
    return stds


def random_permutation(shape):
    a = np.arange(shape[0], dtype=int)
    b = np.arange(shape[1], dtype=int)
    new_a = np.random.permutation(shape[0])
    new_b = np.random.permutation(shape[1])
    d = {'feature': dict(zip(b, new_b)), 'sample': dict(zip(a, new_a))}
    inverse = {'feature': dict(zip(new_b, b)), 'sample': dict(zip(new_a, a))}
    return d, inverse

    
def shuffle_matrix(matrix, d_sample, d_feature=None):
    if d_feature is None:
        d_feature = d_sample
    x_indices = np.asarray([d_sample[i] for i in xrange(matrix.shape[0])])
    y_indices = np.asarray([d_feature[i] for i in xrange(matrix.shape[1])])
    new_matrix = matrix[x_indices]
    new_matrix = new_matrix[:, y_indices]
    return new_matrix

    
def shuffle_pairs(pairs, d):
    new_pairs = np.zeros_like(pairs, dtype=int)
    for i in xrange(pairs.shape[0]):
        for j in xrange(pairs.shape[1]):
            new_pairs[i, j] = d[pairs[i, j]]
    return new_pairs



            
class DataSubset(object):

    def __init__(self):
        self.mixed =  None,
        self.large_scale_mixed = None

        self.subset_indices = None
        self.subset_shape = None
        self.shape = None
        self.annotation = None
        self.annotation_batch = None

        self.guess_X = None
        self.guess_usvt = None
        self.estimated_bias = None
        self.estimated_signal = None
        self.estimated_pairs = None
        self.estimated_stds = None
        self.estimated_directions = None
        self.estimated_correlations = None

        self.signal = None
        self.true_missing = None
        self.true_bias = None
        self.true_pairs = None
        self.true_stds = None
        self.true_directions = None
        self.true_correlations = None

        self.correlation_threshold = None
        self.subset_factor = None
    
    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):    
        return repr(self.__dict__) 


class Data(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, seed=None):
        """Abstract base class for the data containers.

        Parameters
        ----------
        seed : int, optional, default == None
            The seed to initialize np.random.seed with.
            
        Attributes
        ----------
        d : dict
            Dictionary containing all the intial data of a bias recovery run (including randomly created signal, bias, missing matrices).
        rank : int
            Rank of the matrix to be recovered.
        noise_amplitude :
            Noise level of the bias.
        """
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed
        self.d = {'sample': DataSubset(), 'feature': DataSubset()}
        self.rank = None
        self.noise_amplitude = None
        
    @abc.abstractmethod
    def estimate(self):
        """Estimate sucessively correlations, pairs, directions and strandard deviations from a `mixed` matrix.
        """
        pass

    
class DataSimulated(Data):

    def __init__(self, shape, rank, bias_model='gaussian', correlation_threshold=0.7, m_blocks_size=2, noise_amplitude=1.0, correlation_strength=1.0, missing_type='MAR', missing_fraction=0.1, feature_annotation=None, sample_annotation=None, feature_annotation_batch=None, sample_annotation_batch=None, seed=None, image_source='../../tests/trump.png'):
        """Creates (simulates) and stores all the data of a bias recovery experiment.

        Parameters
        ----------
        shape : tuple of int
            Shape of the mixed, signal, bias and missing matrix in the form of (n_samples, n_features).
        rank : int
            Rank of the low-rank decomposition.
        correlation_threshold : float
            The threshold to use when estimating pairs from a correlation matrix (the higher the fewer pairs).
        m_blocks_size : int, default = 2
            Size of each block (e.g. number of pairs). Factor to determine the number of blocks in the correlation matix of features or samples that are varying together (with differences only in degree, direction and scale). Fewer blocks are better for bias recovery.
        noise_amplitude : float, default = None
            Scale/amptitude of the bias (noise).
        missing_type : {'MAR', 'NMAR', 'no-missing', 'SCAN'}
            The type if missing values, from none to censored or SCAN based.
        feature_annotation : list, optional
            List of str that annotates the features in `mixed`.
        sample_annotation : list, optional 
            List of str that annotates the samples in `mixed`.
        feature_annotation_batch : list, optional
            List of str that annotates the features in `mixed` for the batch they are from, e.g. measurement technology/platform.
        sample_annotation_batch : list, optional 
            List of str that annotates the samples in `mixed` for the batch they are from, e.g. measurement technology/platform.
        seed : int
            Seed for the random number generator.
        """
        super(DataSimulated, self).__init__(seed)
        self.shape = shape
        self.rank = rank
        self.correlation_threshold = correlation_threshold
        self.m_blocks_size = m_blocks_size
        self.noise_amplitude = noise_amplitude
        self.correlation_strength = correlation_strength
        self.missing_type = missing_type
        self.feature_annotation = feature_annotation
        self.sample_annotation = sample_annotation
        self.feature_annotation_batch = feature_annotation_batch
        self.sample_annotation_batch = sample_annotation_batch
        self.missing_fraction = missing_fraction
        self.bias_model = bias_model
        self.image_source = image_source

        m_blocks = self.shape[0] // self.m_blocks_size # NOTE using the sample space to determine the m_blocks here.

        #print 'm_blocks', m_blocks
        #print 'shape', shape

        bias_unshuffled = BiasLowRank(self.shape, self.rank, bias_model=self.bias_model, noise_amplitude=self.noise_amplitude, image_source=self.image_source).generate() # BiasUnconstrained(self.shape, bias_model='gaussian', noise_amplitude=1.0).generate()
        self.map_forward_bias, self.map_backward_bias = random_permutation(bias_unshuffled['X'].shape)
        bias = shuffle_matrix(bias_unshuffled['X'], self.map_forward_bias['sample'], self.map_forward_bias['feature'])
        
        missing = Missing(self.shape, self.missing_type, p_random=self.missing_fraction).generate()['X']

        signal_unshuffled = RedundantSignal(self.shape, 'random', m_blocks, self.correlation_strength).generate()
        self.map_forward, self.map_backward = random_permutation(signal_unshuffled['X'].shape)
        signal = shuffle_matrix(signal_unshuffled['X'], self.map_forward['sample'], self.map_forward['feature'])
        
        mixed = signal + bias + missing
        
        for space in ['sample', 'feature']:
            self.d[space]['mixed'] = transpose_view(mixed, space)
            self.d[space]['shape'] = self.d[space]['mixed'].shape
            self.d[space]['signal_unshuffled'] = transpose_view(signal_unshuffled['X'], space)
            self.d[space]['signal'] = transpose_view(signal, space)
            self.d[space]['true_missing'] = transpose_view(missing, space)
            self.d[space]['true_bias_unshuffled'] = transpose_view(bias_unshuffled['X'], space)
            self.d[space]['true_bias'] = transpose_view(bias, space)
            self.d[space]['true_correlations_unshuffled'] = signal_unshuffled[space]['correlation_matrix']
            self.d[space]['true_correlations'] = shuffle_matrix(signal_unshuffled[space]['correlation_matrix'], self.map_forward[space])
            self.d[space]['true_pairs_unshuffled'] = signal_unshuffled[space]['pairs']
            self.d[space]['true_pairs'] = shuffle_pairs(signal_unshuffled[space]['pairs'], self.map_backward[space])
            self.d[space]['true_stds'] = signal_unshuffled[space]['stds'][signal_unshuffled[space]['pairs']] # WARNING Could be wrong!
            self.d[space]['true_directions'] = signal_unshuffled[space]['directions'] # WARNING Could be wrong!
            self.d[space]['correlation_threshold'] = correlation_threshold
            
    def estimate(self, true_pairs=None, true_directions=None, true_stds=None, true_correlations=None):
        """Estimate sucessively correlations, pairs, directions and strandard deviations from a `mixed` matrix.

        Parameters
        ----------
        true_pairs : ndarray, int (n, 2)
            Sequence of true pairs given as tuples.
        true_directions : ndarray, int, n
            Sequence of true directions, e.g. -1, +1.
        true_stds : ndarray, int (n, 2)
            Sequence of true standard deviations of each pair.
        """
        for space in ['feature', 'sample']:
            assert self.d[space]['correlation_threshold'] is not None
            assert self.d[space]['mixed'] is not None
            if true_correlations is not None:
                self.d[space]['estimated_correlations'] = true_correlations[space]
            else:
                self.d[space]['estimated_correlations'] = estimate_correlations(self.d[space]['mixed'])
            if true_pairs is not None:
                self.d[space]['estimated_pairs'] = true_pairs[space]
            else:
                self.d[space]['estimated_pairs'] = estimate_pairs(self.d[space]['estimated_correlations'], self.d[space]['correlation_threshold'])
            if true_stds is not None:
                self.d[space]['estimated_stds'] = true_stds[space]
            else:
                self.d[space]['estimated_stds'] = estimate_stds(self.d[space]['mixed'], self.d[space]['estimated_pairs'])
            if true_directions is not None:
                self.d[space]['estimated_directions'] = true_directions[space]
            else:
                self.d[space]['estimated_directions'] = estimate_directions(self.d[space]['estimated_correlations'], self.d[space]['estimated_pairs'])



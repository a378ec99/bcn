"""Data class definition module.

Notes
-----
This module defines a class that is used thoughout for storage, processing and visualization of all data.
"""
from __future__ import division, absolute_import


__all__ = ['DataSimulated', 'DataSimulatedLarge', 'DataBlind', 'DataBlindLarge', 'estimate_pairs', 'estimate_correlations', 'estimate_directions', 'estimate_stds', 'pair_subset']

from copy import deepcopy
import numpy as np
import abc
from scipy.stats import pearsonr
from bias import BiasLowRank
from redundant_signal import RedundantSignal
from missing import Missing


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
    threshold : int, default 0.7
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


def pair_subset(pairs, subset_indices, mode='subset_pairs'):
    """Creates new indices for pairs and the corresponding mappings and selection for the subset matrix from the large_scale_matrix.

    Parameters
    ----------
    pairs : ndarray, shape (n, 2)
        Pairs found at the large_scale_matrix aas determined by thresholding the correation matrix.
    subset_indices : ndarray, m
        Indices used to select a subset of the large_scale_matrix in sample/feature space.
    mode : {'selection', 'subset_pairs', 'subset_pairs_remapped'}, default = 'subset_pairs'
        The type of remapping to return
        
    Returns
    -------
    subset_pairs : ndarray, int shape (n <= m, 2)
        The subset of pairs which contain indices in `subset_indices_sample` and `subset_indices_feature`
    subset_pairs_remapped : ndarray, int shape (n <= m, 2)
        The `subset_pairs` remapped to the smaller matrix with shape (len(subset_indices_sample), len(subset_indices_feature))
    selection : ndarray, int (n <= m)
        Indices of pairs which are in the subset (to go from original pairs to ony those on the subset quickly).
    """
    if mode == 'selection':
        subset_pairs = np.asarray([pair for pair in pairs if pair[0] in subset_indices and pair[1] in subset_indices])
        selection = np.asarray([pairs.tolist().index(pair.tolist()) for pair in subset_pairs])
        return selection
    if mode == 'subset_pairs':
        subset_pairs = np.asarray([pair for pair in pairs if pair[0] in subset_indices and pair[1] in subset_indices])
        return subset_pairs
    if mode == 'subset_pairs_remapped':
        mapping = dict(zip(subset_indices, range(len(subset_indices))))
        subset_pairs = np.asarray([pair for pair in pairs if pair[0] in subset_indices and pair[1] in subset_indices])
        subset_pairs_remapped = np.asarray([(mapping[pair[0]], mapping[pair[1]]) for pair in subset_pairs])
        return subset_pairs_remapped

        
def pair_subset_back_map(pairs, subset_indices):
    """Map back from indices of pairs of the subset matrix to indices of the large_scale_matrix.

    Parameters
    ----------
    pairs : ndarray, shape (n, 2)
        Pairs of indices of the subset matrix.
    subset_indices : ndarray, shape (n)
        Indices for the larg_scale_matrix to produce the subset matrix.

    Returns
    -------
    remapped : ndarray, shape (n, 2)
        Pairs of indices converted to fir with the large_scale_matrix.
    """
    mapping = dict(zip(range(len(subset_indices)), subset_indices))
    remapped = np.asarray([(mapping[pair[0]], mapping[pair[1]]) for pair in pairs])
    return remapped
    

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

    @abc.abstractmethod
    def guess(self):
        """Generate an initial bias guess for the solver to start at.
        """
        pass

    def guess(self, rank=None):
        """Generate an initial bias guess for the solver to start at.
        """
        if rank is None:
            rank = self.rank
        for space in ['sample', 'feature']:
            assert self.d[space]['mixed'] is not None
            bias = BiasLowRank(self.d[space]['mixed'].shape, 'gaussian', rank, noise_amplitude=self.noise_amplitude).generate() # WARNING need to guess appropriately if different type of bias manifold or source.
            self.d[space]['guess_X'] = bias['X']
            self.d[space]['guess_usvt'] = bias['usvt']    

    
class DataSimulated(Data):

    def __init__(self, shape, rank, correlation_threshold=0.7, m_blocks_factor=4, noise_amplitude=1.0, feature_annotation=None, sample_annotation=None, feature_annotation_batch=None, sample_annotation_batch=None, seed=None):
        """Creates (simulates) and stores all the data of a bias recovery experiment.

        Parameters
        ----------
        shape : tuple of int
            Shape of the mixed, signal, bias and missing matrix in the form of (n_samples, n_features).
        rank : int
            Rank of the low-rank decomposition.
        correlation_threshold : float, default = 0.7
            The threshold to use when estimating pairs from a correlation matrix (the higher the fewer pairs).
        m_blocks_factor : int, default = 4
            Factor to determine the number of blocks in the correlation matix of features or samples that are varying together (with differences only in degree, direction and scale). Fewer blocks are better for bias recovery.
        feature_annotation : list, optional
            List of str that annotates the features in `mixed`.
        sample_annotation : list, optional 
            List of str that annotates the samples in `mixed`.
        feature_annotation_batch : list, optional
            List of str that annotates the features in `mixed` for the batch they are from, e.g. measurement technology/platform.
        sample_annotation_batch : list, optional 
            List of str that annotates the samples in `mixed` for the batch they are from, e.g. measurement technology/platform.
        noise_amplitude : float, default = None
            Scale/amptitude of the bias (noise).
        """
        super(DataSimulated, self).__init__(seed)
        self.shape = shape
        self.rank = rank
        self.correlation_threshold = correlation_threshold
        self.m_blocks_factor = m_blocks_factor
        self.feature_annotation = feature_annotation
        self.sample_annotation = sample_annotation
        self.feature_annotation_batch = feature_annotation_batch
        self.sample_annotation_batch = sample_annotation_batch
        self.noise_amplitude = noise_amplitude
        
        m_blocks = self.shape[0] // self.m_blocks_factor # NOTE using the sample space to determine the m_blocks here.
        bias = BiasLowRank(self.shape, 'gaussian', self.rank, noise_amplitude=self.noise_amplitude).generate()
        missing = missing = Missing(self.shape, 'no-missing', p_random=0.1).generate()
        signal = RedundantSignal(self.shape, 'random', m_blocks, 1.0).generate()
        mixed = signal['X'] + bias['X'] + missing['X']

        for space in ['sample', 'feature']:
            self.d[space]['mixed'] = transpose_view(mixed, space)
            self.d[space]['shape'] = self.d[space]['mixed'].shape
            self.d[space]['signal'] = transpose_view(signal['X'], space)
            self.d[space]['true_missing'] = transpose_view(missing['X'], space)
            self.d[space]['true_bias'] = transpose_view(bias['X'], space)
            self.d[space]['true_correlations'] = signal[space]['correlation_matrix']
            self.d[space]['true_pairs'] = signal[space]['pairs']
            self.d[space]['true_stds'] = signal[space]['stds'][signal[space]['pairs']]
            self.d[space]['true_directions'] = signal[space]['directions'] # TODO assert that these are indeed using the same pairs as used in true_stds.
            self.d[space]['correlation_threshold'] = correlation_threshold
            
    def estimate(self, true_pairs=None, true_directions=None, true_stds=None):
        """Estimate sucessively correlations, pairs, directions and strandard deviations from a `mixed` matrix.
        """
        for space in ['feature', 'sample']:
            assert self.d[space]['correlation_threshold'] is not None
            assert self.d[space]['mixed'] is not None
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

                
class DataSimulatedLarge(Data):

    def __init__(self, large_scale_shape, rank, correlation_threshold=0.7, m_blocks_factor=4, subset_factor=4, noise_amplitude=1.0, feature_annotation=None, sample_annotation=None, feature_annotation_batch=None, sample_annotation_batch=None, seed=None):
        """Creates (simulates) and stores all the data of a large bias recovery experiment.

        Parameters
        ----------
        large_scale_shape : tuple of int
            Shape of the large mixed, signal, bias and missing matrix in the form of (n_samples, n_features).
        rank : int
            Rank of the low-rank decomposition.
        correlation_threshold : float, default = 0.7
            The threshold to use when estimating pairs from a correlation matrix (the higher the fewer pairs).
        m_blocks_factor : int, default = 4
            Factor to determine the number of blocks in the correlation matix of features or samples that are varying together (with differences only in degree, direction and scale). Fewer blocks are better for bias recovery.
        feature_annotation : list, optional
            List of str that annotates the features in `mixed`.
        sample_annotation : list, optional
            List of str that annotates the samples in `mixed`.
        feature_annotation_batch : list, optional
            List of str that annotates the features in `mixed` for the batch they are from, e.g. measurement technology/platform.
        sample_annotation_batch : list, optional
            List of str that annotates the samples in `mixed` for the batch they are from, e.g. measurement technology/platform.
        """
        super(DataSimulatedLarge, self).__init__(seed)
        self.large_scale_shape = large_scale_shape
        self.rank = rank
        self.correlation_threshold = correlation_threshold
        self.m_blocks_factor = m_blocks_factor
        self.subset_factor = subset_factor
        self.feature_annotation = feature_annotation
        self.sample_annotation = sample_annotation
        self.feature_annotation_batch = feature_annotation_batch
        self.sample_annotation_batch = sample_annotation_batch
        self.noise_amplitude = noise_amplitude
        
        m_blocks = self.large_scale_shape[0] // self.m_blocks_factor # NOTE using the sample space to determine the m_blocks here.
        large_scale_bias = BiasLowRank(self.large_scale_shape, 'gaussian', self.rank, noise_amplitude=self.noise_amplitude).generate()
        large_scale_missing = Missing(self.large_scale_shape, 'no-missing', p_random=0.1).generate()
        large_scale_signal = RedundantSignal(self.large_scale_shape, 'random', m_blocks, 1.0).generate()
        large_scale_mixed = large_scale_signal['X'] + large_scale_bias['X'] + large_scale_missing['X']
            
        self.d['sample']['subset_indices'] = np.random.choice(range(large_scale_mixed.shape[0]), size=large_scale_mixed.shape[0] // self.subset_factor, replace=False)
        self.d['feature']['subset_indices'] = np.random.choice(range(large_scale_mixed.shape[1]), size=large_scale_mixed.shape[1] // self.subset_factor, replace=False)
        
        for space in ['sample', 'feature']:
            #print space
            self.d[space]['large_scale_mixed'] = transpose_view(large_scale_mixed, space)
            self.d[space]['mixed'] = transpose_view(large_scale_mixed, space)[self.d[space]['subset_indices'][:, None], self.d[opposite(space)]['subset_indices'][None, :]]
            #print 'A', self.d[space]['mixed'].shape
            self.d[space]['shape'] = self.d[space]['mixed'].shape
            XXX = transpose_view(large_scale_signal['X'], space)
            #print 'Z', XXX.shape
            self.d[space]['signal'] = XXX[self.d[space]['subset_indices'][:, None], self.d[opposite(space)]['subset_indices'][None, :]]
            #print 'ZZ', XXX[self.d[space]['subset_indices'][:, None], self.d[opposite(space)]['subset_indices'][None, :]].shape
            #print self.d[space]['subset_indices'][None].shape, self.d[space]['subset_indices'].shape
            #print self.d[opposite(space)]['subset_indices'][:, None].shape, self.d[opposite(space)]['subset_indices'].shape
            
            #print 'B', self.d[space]['signal'].shape, len(self.d[space]['subset_indices']), len(self.d[opposite(space)]['subset_indices'])
            self.d[space]['true_bias'] = transpose_view(large_scale_bias['X'], space)[self.d[space]['subset_indices'][:, None], self.d[opposite(space)]['subset_indices'][None, :]]
            #print 'C', self.d[space]['true_bias'].shape
            
            self.d[space]['true_missing'] = transpose_view(large_scale_missing['X'], space)[self.d[space]['subset_indices'][:, None], self.d[opposite(space)]['subset_indices'][None, :]]
            self.d[space]['true_correlations'] = large_scale_signal[space]['correlation_matrix'][self.d[space]['subset_indices'][:, None], self.d[space]['subset_indices'][None, :]]
            self.d[space]['true_pairs'] = pair_subset(large_scale_signal[space]['pairs'], self.d[space]['subset_indices'], mode='subset_pairs_remapped')
            self.d[space]['true_directions'] = large_scale_signal[space]['directions'][pair_subset(large_scale_signal[space]['pairs'], self.d[space]['subset_indices'], mode='selection')]
            self.d[space]['true_stds'] = large_scale_signal[space]['stds'][pair_subset(large_scale_signal[space]['pairs'], self.d[space]['subset_indices'], mode='subset_pairs')] # # [:, 0] [self.d[space]['subset_indices']][self.d[space]['true_pairs']] should equal selection
            self.d[space]['subset_factor'] = self.subset_factor
            self.d[space]['correlation_threshold'] = self.correlation_threshold
            
    def estimate(self, true_pairs=None, true_directions=None, true_stds=None):
        """Estimate sucessively correlations, pairs, directions and strandard deviations from a `large_scale_mixed` matrix.
        """
        for space in ['feature', 'sample']:
            assert self.d[space]['correlation_threshold'] is not None
            assert self.d[space]['large_scale_mixed'] is not None
            self.d[space]['estimated_correlations'] = estimate_correlations(self.d[space]['large_scale_mixed'])[self.d[space]['subset_indices'][:, None], self.d[space]['subset_indices'][None, :]]

            if true_pairs is not None:
                self.d[space]['estimated_pairs'] = true_pairs[space]
            else:
                self.d[space]['estimated_pairs'] = estimate_pairs(self.d[space]['estimated_correlations'], self.d[space]['correlation_threshold'])
            if true_directions is not None:
                self.d[space]['estimated_directions'] = true_directions[space]
            else:
                self.d[space]['estimated_directions'] = estimate_directions(self.d[space]['estimated_correlations'], self.d[space]['estimated_pairs'])
            if true_stds is not None:
                self.d[space]['estimated_stds'] = true_stds[space]
            else:
                self.d[space]['estimated_stds'] = estimate_stds(self.d[space]['large_scale_mixed'], pair_subset_back_map(self.d[space]['estimated_pairs'], self.d[space]['subset_indices'])) 


class DataBlind(Data): # TODO add stds and correlation (direction, pairs) information from larger matrix directly into this here... # NOTE NO need for estimate but all given! OR special large scale estimate?

    def __init__(self, mixed, rank, correlation_threshold=0.7, feature_annotation=None, noise_amplitude=1.0, sample_annotation=None, feature_annotation_batch=None, sample_annotation_batch=None, seed=None):
        """Creates (simulates) and stores all the data of a bias recovery experiment.

        Parameters
        ----------
        mixed : ndarray, shape (n_samples, n_features)
            The bias corrupted low-rank matrix from which the bias is to be recovered.
        rank : int
            The rank to use for the intial guess of the bias (for the solver).
        correlation_threshold : float, default = 0.7
            The threshold to use when estimating pairs from a correlation matrix (the higher the fewer pairs).
        feature_annotation : list, optional
            List of str that annotates the features in `mixed`.
        sample_annotation : list, optional
            List of str that annotates the samples in `mixed`.
        feature_annotation_batch : list, optional
            List of str that annotates the features in `mixed` for the batch they are from, e.g. measurement technology/platform.
        sample_annotation_batch : list, optional
            List of str that annotates the samples in `mixed` for the batch they are from, e.g. measurement technology/platform.
        """
        super(DataBlind, self).__init__(seed)
        self.mixed = mixed
        self.rank = rank
        self.correlation_threshold = correlation_threshold
        self.feature_annotation = feature_annotation
        self.sample_annotation = sample_annotation
        self.feature_annotation_batch = feature_annotation_batch
        self.sample_annotation_batch = sample_annotation_batch
        self.noise_amplitude = noise_amplitude
        
        for space in ['sample', 'feature']:
            self.d[space]['mixed'] = transpose_view(mixed, space)
            self.d[space]['shape'] = self.d[space]['mixed'].shape
            self.d[space]['correlation_threshold'] = correlation_threshold

    def estimate(self, true_pairs=None, true_directions=None, true_stds=None):
        """Estimate sucessively correlations, pairs, directions and strandard deviations from a `mixed` matrix.
        """
        for space in ['feature', 'sample']:
            assert self.d[space]['correlation_threshold'] is not None
            assert self.d[space]['mixed'] is not None
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

                
class DataBlindLarge(Data):

    def __init__(self, large_scale_mixed, rank, correlation_threshold=0.7, subset_factor=4, noise_amplitude=1.0, feature_annotation=None, sample_annotation=None, feature_annotation_batch=None, sample_annotation_batch=None, seed=None):
        """Creates (simulates) and stores all the data of a bias recovery experiment.

        Parameters
        ----------
        large_scale_mixed : ndarray, shape (n_samples, n_features)449
            The bias corrupted large low-rank matrix from which the bias is to be recovered.
        rank : int
            The rank to use for the intial guess of the bias (for the solver).
        correlation_threshold : float, default = 0.7
            The threshold to use when estimating pairs from a correlation matrix (the higher the fewer pairs).
        feature_annotation : list, optional
            List of str that annotates the features in `mixed`.
        sample_annotation : list, optional
            List of str that annotates the samples in `mixed`.
        feature_annotation_batch : list, optional
            List of str that annotates the features in `mixed` for the batch they are from, e.g. measurement technology/platform.
        sample_annotation_batch : list, optional
            List of str that annotates the samples in `mixed` for the batch they are from, e.g. measurement technology/platform.

            #TODO get subset_indices from __init__ becquse want to specifically select here certain genes and samples?!
        """
        super(DataBlindLarge, self).__init__(seed)
        self.large_scale_mixed = large_scale_mixed
        self.rank = rank
        self.correlation_threshold = correlation_threshold
        self.subset_factor = subset_factor
        self.feature_annotation = feature_annotation
        self.sample_annotation = sample_annotation
        self.feature_annotation_batch = feature_annotation_batch
        self.sample_annotation_batch = sample_annotation_batch
        self.noise_amplitude = noise_amplitude
        
        self.d['sample']['subset_indices'] = np.random.choice(range(large_scale_mixed.shape[0]), size=large_scale_mixed.shape[0] // self.subset_factor, replace=False)
        self.d['feature']['subset_indices'] = np.random.choice(range(large_scale_mixed.shape[1]), size=large_scale_mixed.shape[1] // self.subset_factor, replace=False)
            
        for space in ['sample', 'feature']:
            self.d[space]['large_scale_mixed'] = transpose_view(large_scale_mixed, space)
            self.d[space]['mixed'] = transpose_view(large_scale_mixed, space)[self.d[space]['subset_indices'][:, None], self.d[opposite(space)]['subset_indices'][None, :]]
            self.d[space]['shape'] = self.d[space]['mixed'].shape
            self.d[space]['subset_factor'] = self.subset_factor
            self.d[space]['correlation_threshold'] = self.correlation_threshold
            
    def estimate(self):
        """Estimate sucessively correlations, pairs, directions and strandard deviations from a `mixed` matrix.
        """
        for space in ['feature', 'sample']:
            assert self.d[space]['correlation_threshold'] is not None
            assert self.d[space]['large_scale_mixed'] is not None
            self.d[space]['estimated_correlations'] = estimate_correlations(self.d[space]['large_scale_mixed'])[self.d[space]['subset_indices'][:, None], self.d[space]['subset_indices'][None, :]]
            self.d[space]['estimated_pairs'] = estimate_pairs(self.d[space]['estimated_correlations'], self.d[space]['correlation_threshold'])
            self.d[space]['estimated_directions'] = estimate_directions(self.d[space]['estimated_correlations'], self.d[space]['estimated_pairs'])
            self.d[space]['estimated_stds'] = estimate_stds(self.d[space]['large_scale_mixed'], pair_subset_back_map(self.d[space]['estimated_pairs'], self.d[space]['subset_indices']))
            


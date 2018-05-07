"""Missing values generation.

Notes
-----
Defines a class that can generate different types of issing values.
"""
from __future__ import division, absolute_import

import numpy as np


class Missing(object):

    def __init__(self, shape, model, p_random=None, p_censored=None):
        """Generate missing values according to a specific model.

        Parameters
        ----------
        shape : tuple of int
            Shape of the output bias matrix in the form of (n_samples, n_features).
        model : {'MAR', 'NMAR', 'SCAN', 'no-missing'}
            Four missing values models are supported, `MAR` which is based on random missing values, `NMAR` which is based on non-random missing values, `SCAN` which is based on the signal-to-noise ratio of 1 output by the SCAN algorithm (all those values <1 are counted as missing due to noise being dominating), and `no-missing` which returns simply a zero matrix.
        p_random : float, optional unless model `MAR`
            Probability that a value is missing.
        p_censored : float, optional unless model `NMAR`
            Probability that a row or column is censored (missing) completely.

        Notes
        -----
        # NOTE The `SCAN` model is currently not yet implemented.
        """
        self.shape = tuple(shape)
        self.model = model
        self.p_random = p_random
        self.p_censored = p_censored 

        assert self.model in ['MAR', 'NMAR', 'SCAN', 'no-missing']

    def generate(self):
        """Generate missing values according to particular model.

        Returns
        -------
        missing : dict, {'X': ndarray, shape (n_sample, n_features)}
            Contains a missing values matrix `X` (missing values as np.nan and others as zeros).
        """
        if self.model == 'MAR':
            missing = np.zeros(self.shape)
            q = list(np.ndindex(self.shape))
            indices = np.random.choice(
                np.arange(len(q)), replace=False, size=int(missing.size * self.p_random))
            for index in indices:
                missing[q[index]] = np.nan
            missing = {'X': missing}
        if self.model == 'NMAR':
            censored = np.zeros(self.shape)
            indices = np.random.choice(
                np.arange(self.shape[0]), replace=False, size=int(self.shape[0] * self.p_censored))
            for index in indices:
                start_ = np.random.randint(0, self.shape[1])
                len_ = np.random.randint(0, self.shape[1] - start_) // 2
                censored[index, start_:start_ + len_] = np.nan
            indices = np.random.choice(
                np.arange(self.shape[1]), replace=False, size=int(self.shape[1] * self.p_censored))
            for index in indices:
                start_ = np.random.randint(0, self.shape[0])
                len_ = np.random.randint(0, self.shape[0] - start_) // 2
                censored[start_:start_ + len_, index] = np.nan
            missing = {'X': censored}
        if self.model == 'SCAN':
            raise NotImplementedError
        if self.model == 'no-missing':
            missing = {'X': np.zeros(self.shape)}
        return missing

        
    

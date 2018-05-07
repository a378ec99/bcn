"""Cost function construction.

Notes
-----
Defines a class that can generate a cost function.
"""
from __future__ import division, absolute_import

import autograd.numpy as ag


class Cost(object):

    def __init__(self, A, y):
        """Creates a cost function based on autograd with given linear operator A and measurements y.
        
        Parameters
        ----------
        A : numpy.ndarray; shape=(n_measurements, n_samples, n_features); dtype=float 
            Linear operator.
        y : numpy.ndarray; shape=(n_measuremnts), dtype=int
            Measurement vector.

        #TODO Allow input as csr_matrix and convert to indices and values for quick multiplication with X.
        """
        self.A = A
        self.y = y

    def cost_func(self, X):
        """Cost function for evaluationg linear operator A and measurements y at X.

        Parameters
        ----------
        X : numpy.ndarray; shape=(n_samples, n_features)
            Measured data matrix with mixed signal and noise.

        Returns
        -------
        error : float
            Squared mean error.

        Note
        ----
        Size scaling with ag.mean is not nessesary for convergence.
        
        #TODO Make nan safe.
        """
        if len(X) == 3:
            usvt = X
            X = ag.dot(usvt[0], ag.dot(ag.diag(usvt[1]), usvt[2]))
        sum_ = ag.sum(self.A * X, axis=(1, 2))
        print sum_.shape, sum_ 
        error = ag.mean((sum_ - self.y)**2)
        return error


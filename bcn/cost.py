"""Cost function construction.

Notes
-----
Defines a class that can generate a cost function.
"""
from __future__ import division, absolute_import

import numpy as np
import autograd.numpy as ag


class Cost(object):

    def __init__(self, A, y):
        """Creates a cost function based on autograd with a linear operator A and target y.
        
        Parameters
        ----------
        A : list; elements=dict, len=n_measurements
            Linear operator stored as sparse matrices.
        y : list; elements=float, len=n_measuremnts
            Target vector.
        """
        self.A = A
        self.y = np.array(y)

    def cost_func(self, X):
        """Cost function for evaluationg linear operator A and measurements y at X.

        Parameters
        ----------
        X : numpy.ndarray; shape=(n_samples, n_features)
            Measured data matrix containing mixed signal and noise.

        Returns
        -------
        error : float
            Squared mean error.

        Note
        ----
        Size scaling with ag.mean is not nessesary for convergence.
        """
        
        if isinstance(X, tuple):
            usvt = X
            X = ag.dot(usvt[0], ag.dot(ag.diag(usvt[1]), usvt[2]))

        if np.isnan(X).any():
            print 'Warning: X contains nan values. Converting those to zero.'
            X = np.nan_to_num(X, copy=False)
            
        y_est = []
        for measurement in self.A:
            sum_ = 0.0
            for i, j, value in zip(measurement['row'], measurement['col'], measurement['value']):
                sum_ = sum_ + value * X[i, j]
            y_est.append(sum_)
            
        y_est = np.array(y_est)
        error = ag.mean((y_est - self.y)**2)
        return error


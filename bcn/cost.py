"""Cost function construction.

Notes
-----
Defines a class that can generate a cost function.
"""
from __future__ import division, absolute_import

import numpy as np
import autograd.numpy as ag


class Cost(object):

    def __init__(self, A, y, sparsity=1):
        """Creates a cost function based on autograd with a linear operator A and target y.
        
        Parameters
        ----------
        A : list; elements=dict, len=n_measurements
            Linear operator stored as sparse matrices.
        y : list; elements=float, len=n_measuremnts
            Target vector.
        """
        self.sparsity = sparsity
        self.A_rows = ag.array([a['row'] for a in A])
        self.A_cols = ag.array([a['col'] for a in A])
        self.A_values = ag.array([a['value'] for a in A])
        self.y = ag.array(y)
        
    def cost_func(self, X):
        """Cost function for evaluationg linear operator A and targets y at X.

        Parameters
        ----------
        X : numpy.ndarray; shape=(n_samples, n_features)
            Estimated bias matrix.

        Returns
        -------
        error : float
            Squared mean error.

        Note
        ----        
        Size scaling with ag.mean is not nessesary for convergence.
        """
        if len(X) == 3: # WARNING type changes from list to autograd.builtins.SequenceBox after first run.
            usvt = X
            X = ag.dot(usvt[0], ag.dot(ag.diag(usvt[1]), usvt[2]))
        else:
            assert np.isfinite(X).all()
        y_est = ag.sum(self.A_values * X[self.A_rows, self.A_cols].reshape(-1, self.sparsity), axis=1)
        error = ag.mean((y_est - self.y)**2)
        return error


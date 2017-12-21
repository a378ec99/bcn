"""Cost function construction.

Notes
-----
Defines a class that can generate a cost function.
"""
from __future__ import division, absolute_import


__all__ = ['Cost']

import autograd.numpy as ag


class Cost(object):

    def __init__(self, A, y):
        """Creates a cost function based on autograd with given linear operator A and measurements y.

        Parameters
        ----------
        A : 2d-array (generally sparse)
            Linear operator.
        y : 1d-array
            Measurement vector.
        """
        self.A = A
        self.y = y

    def cost_func(self, X):
        """Cost function for evaluationg linear operator A and measurements y at X.

        Parameters
        ----------
        X : 2d-array
            Given data matrix (mixed).

        Returns
        -------
        error : float
            The squared mean error of y - y_est, where y_est = A * X summed.

        Note
        ----
        The size scaling is not nessesary for convergence. # TODO Use ag.sqrt to get RMSE?
        """
        if len(X) == 3:
            usvt = X
            X = ag.dot(usvt[0], ag.dot(ag.diag(usvt[1]), usvt[2]))
        error = ag.mean((ag.sum(self.A * X, axis=(1, 2)) - self.y)**2) 
        return error


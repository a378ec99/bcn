"""Cost function construction module.

Notes
-----
This module defines a class that can generate a cost function.
"""
from __future__ import division, absolute_import


__all__ = ['Cost']

import autograd.numpy as ag


class Cost(object):

    def __init__(self, A, y):
        """Creates a cost function based on autograd with given linear operator A and measurements y.
        """
        self.A = A
        self.y = y

    def cost_function(self, X):
        """Cost function for evaluationg linear operator A and measurements y at X.
        """
        if len(X) == 3:
            usvt = X
            X = ag.dot(usvt[0], ag.dot(ag.diag(usvt[1]), usvt[2]))
        error = ag.sum((ag.sum(self.A * X, axis=(1, 2)) - self.y)**2) / self.y.size # NOTE The size scaling is not nessesary for convergence.
        return error


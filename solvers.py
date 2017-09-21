"""Solver construction module.

Notes
-----
This module defines a class that can solve a pymanopt `problem`. Currently only supports ConjugateGradient.
"""
from __future__ import division, absolute_import


__all__ = ['ConjugateGradientSolver']

import numpy as np
from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient # ,SteepestDescent
from pymanopt.manifolds import FixedRankEmbedded


class ConjugateGradientSolver(object):

    def __init__(self, cost, data, rank, n_restarts, seed=None, verbosity=2):
        """Create a solver to recovery bias given observations `X` and a cost function.

        Parameters
        ----------
        cost_function : func
            Cost function that has information about linear operator `A` and measurement `y` and takes observation `X` as input.
        data : dict
            Dictionary containing all the intial data of a bias recovery run (including randomly created signal, bias, missing matrices).
        rank : int
            Rank of the manifold (presumably the same rank of the true bias matrix to be recovered) and rank of the initial guess.
        n_restarts : int
            Number of restats of the solver (more needed the larger the bias matrix).
        seed : int, default = 42
            The random see with wich to run the recovery.
        verbosiy : {0, 1, 2}
            Level of information the gets printed during solver run. A higher number means more.

        # TODO Introduce the option to use a specified inital guess.
        """
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        self.data = data
        self.rank = rank
        self.n_restarts = n_restarts
        self.space = 'sample' # TODO Better way?
        self.manifold = FixedRankEmbedded(self.data.d[self.space]['mixed'].shape[0], self.data.d[self.space]['mixed'].shape[1], self.rank)
        self.problem = Problem(manifold=self.manifold, cost=cost, verbosity=verbosity)
        self.solver = ConjugateGradient(logverbosity=2, maxiter=1000, maxtime=100, mingradnorm=1e-12, minstepsize=1e-12) # , maxiter=None, maxtime=None, mingradnorm=None, minstepsize=None
        
    def solve(self, guess):
        usvt, optlog = self.solver.solve(self.problem, x=guess)
        stopping_reason = optlog['stoppingreason']
        final_cost = optlog['final_values']['f(x)']
        X = usvt[0].dot(np.diag(usvt[1])).dot(usvt[2])
        return X, stopping_reason, final_cost

    def recover(self):
        estimates, errors, guesses_X, guesses_usvt = [], [], [], []
        for k in xrange(self.n_restarts):
            self.data.guess(rank=self.rank)
            X, stopping_reason, final_cost = self.solve(self.data.d[self.space]['guess_usvt'])
            print final_cost
            estimates.append(X)
            guesses_X.append(self.data.d[self.space]['guess_X'])
            guesses_usvt.append(self.data.d[self.space]['guess_usvt'])
            errors.append(final_cost)
        index = np.argmin(errors)

        error = errors[index]
        estimated_bias = estimates[index] 
        guess_X = guesses_X[index]
        guess_usvt = guesses_usvt[index]

        self.data.d[self.space]['guess_X'] = guess_X
        self.data.d[self.space]['guess_usvt'] = guess_usvt
        self.data.d[self.space]['estimated_bias'] = estimated_bias
        estimated_signal = self.data.d[self.space]['mixed'] - estimated_bias
        self.data.d[self.space]['estimated_signal'] = estimated_signal

        return self.data.d

        
# NOTE
# https://github.com/ContinuumIO/anaconda-issues/issues/695
# print 'Likely np.linalg.LinAlgError:', traceback.format_exc()







        
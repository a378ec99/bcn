"""Solver construction.

Notes
-----
Defines a class that can solve a pymanopt `problem`. Currently only supports ConjugateGradient.
"""
from __future__ import division, absolute_import

import numpy as np

from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient # ,SteepestDescent
from pymanopt.manifolds import FixedRankEmbedded


class ConjugateGradientSolver(object):

    def __init__(self, cost_func, guess_func, data, rank, n_restarts, noise_amplitude=5.0, maxiter=1000, space='sample', seed=None, verbosity=2):
        """Create a solver to recovery bias given observations `X` and cost/guess functions.

        Parameters
        ----------
        cost_func : func
            Cost function that has information about linear operator `A` and measurement `y` and takes observation `X` as input.
        guess_func : func
            Guess function that guesses an intial point for the solver to start optimizing at.
        data : Data object
            Contains a dictionary with all the data for the intial bias recovery run (including randomly created signal, bias, missing matrices).
        rank : int
            Rank of the manifold (presumably the same rank of the true bias matrix to be recovered) and rank of the initial guess.
        n_restarts : int
            Number of restats of the solver (more needed the larger the bias matrix).
        noise_amplitude : float
            Noise level needed to make a guess
        space = str, {feature, sample}
            To be used as reference for the input mixed.
        seed : int, default = 42
            The random see with wich to run the recovery.
        verbosiy : {0, 1, 2}
            Level of information the gets printed during solver run. A higher number means more.
        """
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        self.data = data
        self.space = space
        self.rank = rank
        self.n_restarts = n_restarts
        self.maxiter = maxiter
        self.mixed = self.data.d[self.space]['mixed']
        self.shape = self.mixed.shape
        self.manifold = FixedRankEmbedded(self.shape[0], self.shape[1], self.rank)
        self.guess_func = guess_func
        self.problem = Problem(manifold=self.manifold, cost=cost_func, verbosity=verbosity)
        self.solver = ConjugateGradient(logverbosity=2, maxiter=self.maxiter, maxtime=100, mingradnorm=1e-12, minstepsize=1e-12) # , maxiter=None, maxtime=None, mingradnorm=None, minstepsize=None
        self.n_retries_svd = 10
        self.noise_amplitude = noise_amplitude
        
    def solve(self, guess):
        """ Solve a particular recovery problem based on the given initial guess.
        
        Parameters
        ----------
        guess : tuple, (u, s, vt)
            Descomposed random low-rank matrix as initial guess.
            
        Returns
        -------
        X : ndarray, (n_samples, n_features)
            Solution of the recovery problem.
        stopping_reason : str
            Why the solver finished, e.g. out of time, out of steps, etc.
        final_cost : float
            Final value of the objective function.
        """
        worked = False
        for n in xrange(self.n_retries_svd):
            try:
                usvt, optlog = self.solver.solve(self.problem, x=guess)
                worked = True
                break
            except np.linalg.LinAlgError:
                continue
        if worked == False:
            raise Exception('Not enough SVD restarts.')
        stopping_reason = optlog['stoppingreason']
        final_cost = optlog['final_values']['f(x)']
        X = usvt[0].dot(np.diag(usvt[1])).dot(usvt[2])
        return X, stopping_reason, final_cost

    def recover(self):
        """
        Runs the solver n_restarts times and picks the best run.
        
        Returns
        -------
        self.data : Data object
            Updated data object with solution from solver and intial guess, plus estimates of correlations, directions and standard deviations.
        """
        estimates, errors, guesses_X, guesses_usvt = [], [], [], []
        for k in xrange(self.n_restarts):
            guess = self.guess_func(self.shape, self.rank, noise_amplitude=self.noise_amplitude)
            X, stopping_reason, final_cost = self.solve(guess['usvt'])
            #print final_cost
            estimates.append(X)
            guesses_X.append(guess['X'])
            guesses_usvt.append(guess['usvt'])
            errors.append(final_cost)
        index = np.argmin(errors)

        error = errors[index]
        estimated_bias = estimates[index] 
        guess_X = guesses_X[index]
        guess_usvt = guesses_usvt[index]

        self.data.d[self.space]['guess_X'] = guess_X
        self.data.d[self.space]['guess_usvt'] = guess_usvt
        self.data.d[self.space]['estimated_bias'] = estimated_bias
        self.data.d[self.space]['estimated_signal'] = self.mixed - estimated_bias
        self.data.d[self.space]['final_cost'] = error
        return self.data

        

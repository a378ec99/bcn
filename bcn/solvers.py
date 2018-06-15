"""Solvers for matrix recovery.

Note
----
Currently only supports conjugate gradient methods.
"""
from __future__ import division, absolute_import

import numpy as np

from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient, SteepestDescent
from pymanopt.manifolds import FixedRankEmbedded


class ConjugateGradientSolver(object):

    def __init__(self, mixed, cost_func, guess_func, rank, n_restarts=10, guess_noise_amplitude=5.0, maxiter=1000, maxtime=100, mingradnorm=1e-12, minstepsize=1e-12, n_retries_svd=10, verbosity=2):
        """Solver for matrix recovery.

        Parameters
        ----------
        mixed : numpy.ndarray, shape=(n_samples, n_features)
            Corrupted signal to be cleaned.
        cost_func : func
            Cost function based on linear operators A and targets y.
        guess_func : func
            Guess function that guesses an intial point for the solver to start optimizing at.
        rank : int
            Rank of the matrix to be recovered and of the initial guess.
        n_restarts : int
            Number of restats of the solver with a fresh initial guess.
        guess_noise_amplitude : float
            Noise amplitude for the random low-rank initial guess.
        maxiter : int
            Maximum number of iterations of solver.
        maxtime : int
            Maximum run time of solver in seconds.
        mingradnorm : float
            Minimal gradient norm of solver (before stopping).
        minstepsize : int
            Minimal step size of solver (before stopping).
        n_retries_svd : int
            Number of retries when LinAlgError.
        verbosiy : int, values=(0, 1, 2)
            Higher verbosity means more information printed.
        """
        self.mixed = mixed
        self.shape = self.mixed.shape
        self.guess_func = guess_func
        self.cost_func = cost_func
        self.rank = rank
        self.n_restarts = n_restarts
        self.guess_noise_amplitude = guess_noise_amplitude
        self.manifold = FixedRankEmbedded(
            self.shape[0], self.shape[1], self.rank)
        self.problem = Problem(manifold=self.manifold,
                               cost=self.cost_func, verbosity=verbosity)
        self.maxiter = maxiter
        self.maxtime = maxtime
        self.mingradnorm = mingradnorm
        self.minstepsize = minstepsize
        self.solver = ConjugateGradient(logverbosity=2, maxiter=self.maxiter, maxtime=self.maxtime,
                                        mingradnorm=self.mingradnorm, minstepsize=self.minstepsize)
        self.n_retries_svd = n_retries_svd

    def solve(self, guess):
        """ Solve a matrix recovery problem based on the given constraints and an initial guess.

        Parameters
        ----------
        guess : tuple, values=(u, s, vt)
            Descomposed random low-rank matrix.

        Returns
        -------
        X : numpy.ndarray, shape=(n_samples, n_features)
            Solution of the recovery problem.
        stopping_reason : str
            Why the solver finished, e.g. out of time, out of steps, etc.
        final_cost : float
            Final value of the cost function.
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
        results : dict
            Results of the recovery with initial guess, estimated signal, estimated bias and final cost.
        """
        estimates, errors, guesses_X, guesses_usvt = [], [], [], []
        for k in xrange(self.n_restarts):
            guess = self.guess_func(
                self.shape, self.rank, noise_amplitude=self.guess_noise_amplitude)
            X, stopping_reason, final_cost = self.solve(guess['usvt'])
            estimates.append(X)
            guesses_X.append(guess['X'])
            guesses_usvt.append(guess['usvt'])
            errors.append(final_cost)
        index = np.argmin(errors)
        error = errors[index]
        estimated_bias = estimates[index]
        guess_X = guesses_X[index]
        guess_usvt = guesses_usvt[index]
        results = {'guess_X': guess_X,
                   'guess_usvt': guess_usvt,
                   'estimated_bias': estimated_bias,
                   'estimated_signal': self.mixed - estimated_bias,
                   'final_cost': error}
        return results



class SteepestDescentSolver(object):

    def __init__(self, mixed, cost_func, guess_func, rank, n_restarts=10, guess_noise_amplitude=5.0, maxiter=1000, maxtime=100, mingradnorm=1e-12, minstepsize=1e-12, n_retries_svd=10, verbosity=2):
        """Solver for matrix recovery.

        Parameters
        ----------
        mixed : numpy.ndarray, shape=(n_samples, n_features)
            Corrupted signal to be cleaned.
        cost_func : func
            Cost function based on linear operators A and targets y.
        guess_func : func
            Guess function that guesses an intial point for the solver to start optimizing at.
        rank : int
            Rank of the matrix to be recovered and of the initial guess.
        n_restarts : int
            Number of restats of the solver with a fresh initial guess.
        guess_noise_amplitude : float
            Noise amplitude for the random low-rank initial guess.
        maxiter : int
            Maximum number of iterations of solver.
        maxtime : int
            Maximum run time of solver in seconds.
        mingradnorm : float
            Minimal gradient norm of solver (before stopping).
        minstepsize : int
            Minimal step size of solver (before stopping).
        n_retries_svd : int
            Number of retries when LinAlgError.
        verbosiy : int, values=(0, 1, 2)
            Higher verbosity means more information printed.
        """
        self.mixed = mixed
        self.shape = self.mixed.shape
        self.guess_func = guess_func
        self.cost_func = cost_func
        self.rank = rank
        self.n_restarts = n_restarts
        self.guess_noise_amplitude = guess_noise_amplitude
        self.manifold = FixedRankEmbedded(
            self.shape[0], self.shape[1], self.rank)
        self.problem = Problem(manifold=self.manifold,
                               cost=self.cost_func, verbosity=verbosity)
        self.maxiter = maxiter
        self.maxtime = maxtime
        self.mingradnorm = mingradnorm
        self.minstepsize = minstepsize
        self.solver = SteepestDescent(logverbosity=2, maxiter=self.maxiter, maxtime=self.maxtime,
                                        mingradnorm=self.mingradnorm, minstepsize=self.minstepsize)
        self.n_retries_svd = n_retries_svd

    def solve(self, guess):
        """ Solve a matrix recovery problem based on the given constraints and an initial guess.

        Parameters
        ----------
        guess : tuple, values=(u, s, vt)
            Descomposed random low-rank matrix.

        Returns
        -------
        X : numpy.ndarray, shape=(n_samples, n_features)
            Solution of the recovery problem.
        stopping_reason : str
            Why the solver finished, e.g. out of time, out of steps, etc.
        final_cost : float
            Final value of the cost function.
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
        results : dict
            Results of the recovery with initial guess, estimated signal, estimated bias and final cost.
        """
        estimates, errors, guesses_X, guesses_usvt = [], [], [], []
        for k in xrange(self.n_restarts):
            guess = self.guess_func(
                self.shape, self.rank, noise_amplitude=self.guess_noise_amplitude)
            X, stopping_reason, final_cost = self.solve(guess['usvt'])
            estimates.append(X)
            guesses_X.append(guess['X'])
            guesses_usvt.append(guess['usvt'])
            errors.append(final_cost)
        index = np.argmin(errors)
        error = errors[index]
        estimated_bias = estimates[index]
        guess_X = guesses_X[index]
        guess_usvt = guesses_usvt[index]
        results = {'guess_X': guess_X,
                   'guess_usvt': guess_usvt,
                   'estimated_bias': estimated_bias,
                   'estimated_signal': self.mixed - estimated_bias,
                   'final_cost': error}
        return results
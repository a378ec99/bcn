"""Parallel computation of figures.

Notes
-----
This module defines classes that are used to run multiple bias recoveries in parallel to generate different figures.
"""
from __future__ import division, absolute_import


__all__ = ['Figure1, Figure2, Figure3, Figure4, Figure5, Figure6, Figure7, Figure8', 'shuffle_some_pairs', 'generate_random_pairs']

from abc import ABCMeta, abstractmethod
import subprocess
import json
from popen2 import popen2
from copy import deepcopy

import numpy as np

from data import DataSimulated
from visualization import visualize_performance, visualize_dependences
from solvers import ConjugateGradientSolver
from cost import Cost
from linear_operators import LinearOperatorCustom, LinearOperatorKsparse, min_measurements, max_measurements
from bias import guess_func




class TaskPull(object):
    """
    Abstract class that denotes API to taskpull.py and taskpull_local.py.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def allocate(self):
        pass

    @abstractmethod
    def create_tasks(self):
        pass

    @abstractmethod
    def work(self):
        pass

    @abstractmethod
    def store(self):
        pass

    @abstractmethod
    def postprocessing(self):
        pass

'''        
def shuffle_some_pairs(pairs, random_fraction, max_indices):
    """
    Shuffles a fraction of pairs randomly.

    Parameters
    ----------
    pairs : ndarray (n, 2)
        Sequence of pairs used as integer indices to an array.
    random_fraction : float
        Fraction of pairs that are to be randomized.
    max_indices : int
        Maximum value of indices that are allowed for the pairs.
        
    Returns
    -------
    pairs_new : ndarray (n, 2)
        Sequence of pairs shuffled accordingly.
    """
    n = int(random_fraction * len(pairs))
    if n == len(pairs):
        pairs_new = generate_random_pairs(n, max_indices)
    elif n == 0:
        pairs_new = pairs
    else:
        pairs_new = deepcopy(pairs)
        x = generate_random_pairs(n, max_indices)
        pairs_new[-n:, :] = x
    return pairs_new
'''

def shuffle_some_pairs(pairs, n_random, max_indices):
    """
    Shuffles a fraction of pairs randomly.

    Parameters
    ----------
    pairs : ndarray (n, 2)
        Sequence of pairs used as integer indices to an array.
    random_fraction : float
        Fraction of pairs that are to be randomized.
    max_indices : int
        Maximum value of indices that are allowed for the pairs.

    Returns
    -------
    pairs_new : ndarray (n, 2)
        Sequence of pairs shuffled accordingly.
    """
    n = n_random
    if n == len(pairs):
        pairs_new = generate_random_pairs(n, max_indices)
    elif n == 0:
        pairs_new = pairs
    else:
        pairs_new = deepcopy(pairs)
        x = generate_random_pairs(n, max_indices)
        pairs_new[-n:, :] = x
    return pairs_new
    
    
def generate_random_pairs(n_pairs, max_indices):
    """
    Generates a sequence of random pairs.

    Parameters
    ----------
    n_pairs : int
        Number of random pairs to generate.
    max_indices : int
        Maximum value of indices that are allowed for the pairs.
        
    Returns
    -------
    pairs : ndarray (n_pairs, 2)
        Sequence of random pairs.
    """
    pairs = []
    indices = np.arange(0, max_indices, dtype=int)
    for i in xrange(n_pairs):
        pair = tuple(np.random.choice(indices, size=2, replace=False))
        pairs.append(pair)
    pairs = np.asarray(pairs)
    return pairs


class Figure1(TaskPull):

    def __init__(self, measurements=None, sparsities=None, seed=None, noise_amplitude=None, file_name=None):
        """
        """
        self.rank = 2
        self.measurements = measurements
        self.seed = seed
        self.n_restarts = 10
        self.verbosity = 1
        self.shape = (100, 110)
        self.noise_amplitude = noise_amplitude
        self.file_name = file_name
        self.sparsities = sparsities

    def allocate(self):
        self.true_errors = np.ones((len(self.measurements), len(self.sparsities))) * np.nan
        self.solver_errors = np.ones((len(self.measurements), len(self.sparsities))) * np.nan

    def create_tasks(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        for i, m in enumerate(self.measurements):
            for j, sparsity in enumerate(self.sparsities):
                secondary_seed = np.random.randint(0, 1e8)
                task = i, j, m, sparsity, secondary_seed
                print task
                yield task

    def work(self, task):
        i, j, m, sparsity, secondary_seed = task
        np.random.seed(secondary_seed)

        data = DataSimulated(self.shape, self.rank, noise_amplitude=self.noise_amplitude)
        data.estimate(true_stds={'sample': data.d['sample']['true_stds'], 'feature': data.d['feature']['true_stds']}, true_pairs={'sample': data.d['sample']['true_pairs'], 'feature': data.d['feature']['true_pairs']}, true_directions={'sample': data.d['sample']['true_directions'], 'feature': data.d['feature']['true_directions']})
        operator = LinearOperatorKsparse(data, m, sparsity).generate()
        A, y = operator['A'], operator['y']
        cost = Cost(A, y)
        solver = ConjugateGradientSolver(cost.cost_func, guess_func, data, self.rank, self.n_restarts, noise_amplitude=self.noise_amplitude, verbosity=self.verbosity)
        data = solver.recover()

        error_solver = cost.cost_func(data.d['sample']['estimated_bias'])
        divisor = np.sum(~np.isnan(data.d['sample']['mixed']))
        error = np.nansum(np.absolute(data.d['sample']['signal'] - (data.d['sample']['mixed'] - data.d['sample']['estimated_bias']))) / divisor
        zero_error = np.nansum(np.absolute(data.d['sample']['signal'] - data.d['sample']['mixed'])) / divisor
        error_true = error / zero_error

        #print error_true, error_solver, i, j, 'error_true, error_solver, i, j'
        return error_true, error_solver, i, j

    def store(self, result):
        error_true, error_solver, i, j = result
        self.true_errors[i, j] = error_true
        self.solver_errors[i, j] = error_solver

    def postprocessing(self):
        visualize_performance(self.true_errors, self.sparsities, self.measurements, 'sparsities', 'measurements', file_name=self.file_name + '_error_true')
        visualize_performance(self.solver_errors, self.sparsities, self.measurements, 'sparsities', 'measurements', file_name=self.file_name + '_error_solver')


class Figure2(TaskPull):

    def __init__(self, measurement_percentages=None, dimensions=None, seed=None, noise_amplitude=None, file_name=None):
        """
        """
        self.dimensions = dimensions
        self.measurement_percentages = measurement_percentages
        self.seed = seed
        self.n_restarts = 10
        self.verbosity = 1
        self.noise_amplitude = noise_amplitude
        self.file_name = file_name
        self.sparsity = 2
        self.rank = 2

    def allocate(self):
        self.true_errors = np.ones((len(self.measurement_percentages), len(self.dimensions))) * np.nan
        self.solver_errors = np.ones((len(self.measurement_percentages), len(self.dimensions))) * np.nan

    def create_tasks(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        for i, m in enumerate(self.measurement_percentages):
            for j, dimension in enumerate(self.dimensions):
                secondary_seed = np.random.randint(0, 1e8)
                task = i, j, m, dimension, secondary_seed
                print task
                yield task

    def work(self, task):
        i, j, m, dimension, secondary_seed = task
        np.random.seed(secondary_seed)

        m = int(m * dimension[0] * dimension[1])
        m = min(m, 4000) # WARNING Hardcoded max

        data = DataSimulated(dimension, self.rank, noise_amplitude=self.noise_amplitude)
        data.estimate(true_stds={'sample': data.d['sample']['true_stds'], 'feature': data.d['feature']['true_stds']}, true_pairs={'sample': data.d['sample']['true_pairs'], 'feature': data.d['feature']['true_pairs']}, true_directions={'sample': data.d['sample']['true_directions'], 'feature': data.d['feature']['true_directions']})
        operator = LinearOperatorKsparse(data, m, self.sparsity).generate()
        A, y = operator['A'], operator['y']
        cost = Cost(A, y)
        solver = ConjugateGradientSolver(cost.cost_func, guess_func, data, self.rank, self.n_restarts, noise_amplitude=self.noise_amplitude, verbosity=self.verbosity)
        data = solver.recover()

        error_solver = cost.cost_func(data.d['sample']['estimated_bias'])
        divisor = np.sum(~np.isnan(data.d['sample']['mixed']))
        error = np.nansum(np.absolute(data.d['sample']['signal'] - (data.d['sample']['mixed'] - data.d['sample']['estimated_bias']))) / divisor
        zero_error = np.nansum(np.absolute(data.d['sample']['signal'] - data.d['sample']['mixed'])) / divisor
        error_true = error / zero_error

        #print error_true, error_solver, i, j, 'error_true, error_solver, i, j'
        return error_true, error_solver, i, j

    def store(self, result):
        error_true, error_solver, i, j = result
        self.true_errors[i, j] = error_true
        self.solver_errors[i, j] = error_solver

    def postprocessing(self):
        visualize_performance(self.true_errors, self.dimensions, self.measurement_percentages, 'dimensions', 'measurement_percentages', file_name=self.file_name + '_error_true')
        visualize_performance(self.solver_errors, self.dimensions, self.measurement_percentages, 'dimensions', 'measurement_percentages', file_name=self.file_name + '_error_solver')


class Figure3(TaskPull):

    def __init__(self, measurements=None, ranks=None, seed=None, noise_amplitude=None, file_name=None):
        """
        """
        self.ranks = ranks
        self.measurements = measurements
        self.seed = seed
        self.n_restarts = 10
        self.verbosity = 1
        self.shape = (100, 110)
        self.noise_amplitude = noise_amplitude
        self.file_name = file_name
        self.sparsity = 2
        
    def allocate(self):
        self.true_errors = np.ones((len(self.measurements), len(self.ranks))) * np.nan
        self.solver_errors = np.ones((len(self.measurements), len(self.ranks))) * np.nan

    def create_tasks(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        for i, m in enumerate(self.measurements):
            for j, rank in enumerate(self.ranks):
                secondary_seed = np.random.randint(0, 1e8)
                task = i, j, m, rank, secondary_seed
                print task
                yield task

    def work(self, task):
        i, j, m, rank, secondary_seed = task
        np.random.seed(secondary_seed)

        data = DataSimulated(self.shape, rank, noise_amplitude=self.noise_amplitude) 
        data.estimate(true_stds={'sample': data.d['sample']['true_stds'], 'feature': data.d['feature']['true_stds']}, true_pairs={'sample': data.d['sample']['true_pairs'], 'feature': data.d['feature']['true_pairs']}, true_directions={'sample': data.d['sample']['true_directions'], 'feature': data.d['feature']['true_directions']})
        operator = LinearOperatorKsparse(data, m, self.sparsity).generate()
        A, y = operator['A'], operator['y']
        cost = Cost(A, y)
        solver = ConjugateGradientSolver(cost.cost_func, guess_func, data, rank, self.n_restarts, noise_amplitude=self.noise_amplitude, verbosity=self.verbosity)
        data = solver.recover()

        error_solver = cost.cost_func(data.d['sample']['estimated_bias'])
        divisor = np.sum(~np.isnan(data.d['sample']['mixed']))
        error = np.nansum(np.absolute(data.d['sample']['signal'] - (data.d['sample']['mixed'] - data.d['sample']['estimated_bias']))) / divisor
        zero_error = np.nansum(np.absolute(data.d['sample']['signal'] - data.d['sample']['mixed'])) / divisor
        error_true = error / zero_error

        #print error_true, error_solver, i, j, 'error_true, error_solver, i, j'
        return error_true, error_solver, i, j

    def store(self, result):
        error_true, error_solver, i, j = result
        self.true_errors[i, j] = error_true
        self.solver_errors[i, j] = error_solver

    def postprocessing(self):
        visualize_performance(self.true_errors, self.ranks, self.measurements, 'rank', 'measurements', file_name=self.file_name + '_error_true')
        visualize_performance(self.solver_errors, self.ranks, self.measurements, 'rank', 'measurements', file_name=self.file_name + '_error_solver')


class Figure4(TaskPull):

    def __init__(self, measurements=None, ranks=None, seed=None, noise_amplitude=None, file_name=None):
        """
        """
        self.ranks = ranks
        self.measurements = measurements
        self.seed = seed
        self.n_restarts = 10
        self.verbosity = 1
        self.shape = (100, 110)
        self.noise_amplitude = noise_amplitude
        self.file_name = file_name

    def allocate(self):
        self.true_errors = np.ones((len(self.measurements), len(self.ranks))) * np.nan
        self.solver_errors = np.ones((len(self.measurements), len(self.ranks))) * np.nan

    def create_tasks(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        for i, m in enumerate(self.measurements):
            for j, rank in enumerate(self.ranks):
                secondary_seed = np.random.randint(0, 1e8)
                task = i, j, m, rank, secondary_seed
                print task
                yield task

    def work(self, task):
        i, j, m, rank, secondary_seed = task
        np.random.seed(secondary_seed)

        data = DataSimulated(self.shape, rank, noise_amplitude=self.noise_amplitude)
        data.estimate(true_stds={'sample': data.d['sample']['true_stds'], 'feature': data.d['feature']['true_stds']}, true_pairs={'sample': data.d['sample']['true_pairs'], 'feature': data.d['feature']['true_pairs']}, true_directions={'sample': data.d['sample']['true_directions'], 'feature': data.d['feature']['true_directions']})
        operator = LinearOperatorCustom(data, m).generate()
        A, y = operator['A'], operator['y']
        cost = Cost(A, y)
        solver = ConjugateGradientSolver(cost.cost_func, guess_func, data, rank, self.n_restarts, noise_amplitude=self.noise_amplitude, verbosity=self.verbosity)
        data = solver.recover()

        error_solver = cost.cost_func(data.d['sample']['estimated_bias'])
        divisor = np.sum(~np.isnan(data.d['sample']['mixed']))
        error = np.nansum(np.absolute(data.d['sample']['signal'] - (data.d['sample']['mixed'] - data.d['sample']['estimated_bias']))) / divisor
        zero_error = np.nansum(np.absolute(data.d['sample']['signal'] - data.d['sample']['mixed'])) / divisor
        error_true = error / zero_error

        #print error_true, error_solver, i, j, 'error_true, error_solver, i, j'
        return error_true, error_solver, i, j

    def store(self, result):
        error_true, error_solver, i, j = result
        self.true_errors[i, j] = error_true
        self.solver_errors[i, j] = error_solver

    def postprocessing(self):
        visualize_performance(self.true_errors, self.ranks, self.measurements, 'rank', 'measurements', file_name=self.file_name + '_error_true')
        visualize_performance(self.solver_errors, self.ranks, self.measurements, 'rank', 'measurements', file_name=self.file_name + '_error_solver')
      

class Figure5(TaskPull):

    def __init__(self, measurements=None, y_noises=None, seed=None, noise_amplitude=None, file_name=None):
        """
        """
        self.y_noises = y_noises
        self.measurements = measurements
        self.seed = seed
        self.n_restarts = 10
        self.verbosity = 1
        self.shape = (100, 110)
        self.noise_amplitude = noise_amplitude
        self.file_name = file_name
        self.sparsity = 2
        self.rank = 2
        
    def allocate(self):
        self.true_errors = np.ones((len(self.measurements), len(self.y_noises))) * np.nan
        self.solver_errors = np.ones((len(self.measurements), len(self.y_noises))) * np.nan

    def create_tasks(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        for i, m in enumerate(self.measurements):
            for j, y_noise in enumerate(self.y_noises):
                secondary_seed = np.random.randint(0, 1e8)
                task = i, j, m, y_noise, secondary_seed
                print task
                yield task

    def work(self, task):
        i, j, m, y_noise, secondary_seed = task
        np.random.seed(secondary_seed)

        data = DataSimulated(self.shape, self.rank, noise_amplitude=self.noise_amplitude) 
        data.estimate(true_stds={'sample': data.d['sample']['true_stds'], 'feature': data.d['feature']['true_stds']}, true_pairs={'sample': data.d['sample']['true_pairs'], 'feature': data.d['feature']['true_pairs']}, true_directions={'sample': data.d['sample']['true_directions'], 'feature': data.d['feature']['true_directions']})
        operator = LinearOperatorKsparse(data, m, self.sparsity).generate()
        A, y = operator['A'], operator['y']
        y = y + np.random.normal(0.0, y_noise, size=y.shape) # NOTE
        cost = Cost(A, y)
        solver = ConjugateGradientSolver(cost.cost_func, guess_func, data, self.rank, self.n_restarts, noise_amplitude=self.noise_amplitude, verbosity=self.verbosity)
        data = solver.recover()

        error_solver = cost.cost_func(data.d['sample']['estimated_bias'])
        divisor = np.sum(~np.isnan(data.d['sample']['mixed']))
        error = np.nansum(np.absolute(data.d['sample']['signal'] - (data.d['sample']['mixed'] - data.d['sample']['estimated_bias']))) / divisor
        zero_error = np.nansum(np.absolute(data.d['sample']['signal'] - data.d['sample']['mixed'])) / divisor
        error_true = error / zero_error

        #print error_true, error_solver, i, j, 'error_true, error_solver, i, j'
        return error_true, error_solver, i, j

    def store(self, result):
        error_true, error_solver, i, j = result
        self.true_errors[i, j] = error_true
        self.solver_errors[i, j] = error_solver

    def postprocessing(self):
        visualize_performance(self.true_errors, self.y_noises, self.measurements, 'y_noises', 'measurements', file_name=self.file_name + '_error_true')
        visualize_performance(self.solver_errors, self.y_noises, self.measurements, 'y_noises', 'measurements', file_name=self.file_name + '_error_solver')


class Figure6(TaskPull):

    def __init__(self, measurements=None, y_noises=None, seed=None, noise_amplitude=None, file_name=None):
        """
        """
        self.y_noises = y_noises
        self.measurements = measurements
        self.seed = seed
        self.n_restarts = 10
        self.verbosity = 1
        self.shape = (100, 110)
        self.noise_amplitude = noise_amplitude
        self.file_name = file_name
        self.rank = 2

    def allocate(self):
        self.true_errors = np.ones((len(self.measurements), len(self.y_noises))) * np.nan
        self.solver_errors = np.ones((len(self.measurements), len(self.y_noises))) * np.nan

    def create_tasks(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        for i, m in enumerate(self.measurements):
            for j, y_noise in enumerate(self.y_noises):
                secondary_seed = np.random.randint(0, 1e8)
                task = i, j, m, y_noise, secondary_seed
                print task
                yield task

    def work(self, task):
        i, j, m, y_noise, secondary_seed = task
        np.random.seed(secondary_seed)

        data = DataSimulated(self.shape, self.rank, noise_amplitude=self.noise_amplitude)
        data.estimate(true_stds={'sample': data.d['sample']['true_stds'], 'feature': data.d['feature']['true_stds']}, true_pairs={'sample': data.d['sample']['true_pairs'], 'feature': data.d['feature']['true_pairs']}, true_directions={'sample': data.d['sample']['true_directions'], 'feature': data.d['feature']['true_directions']})
        operator = LinearOperatorCustom(data, m).generate()
        A, y = operator['A'], operator['y']
        y = y + np.random.normal(0.0, y_noise, size=y.shape) # NOTE
        cost = Cost(A, y)
        solver = ConjugateGradientSolver(cost.cost_func, guess_func, data, self.rank, self.n_restarts, noise_amplitude=self.noise_amplitude, verbosity=self.verbosity)
        data = solver.recover()

        error_solver = cost.cost_func(data.d['sample']['estimated_bias'])
        divisor = np.sum(~np.isnan(data.d['sample']['mixed']))
        error = np.nansum(np.absolute(data.d['sample']['signal'] - (data.d['sample']['mixed'] - data.d['sample']['estimated_bias']))) / divisor
        zero_error = np.nansum(np.absolute(data.d['sample']['signal'] - data.d['sample']['mixed'])) / divisor
        error_true = error / zero_error

        #print error_true, error_solver, i, j, 'error_true, error_solver, i, j'
        return error_true, error_solver, i, j

    def store(self, result):
        error_true, error_solver, i, j = result
        self.true_errors[i, j] = error_true
        self.solver_errors[i, j] = error_solver

    def postprocessing(self):
        visualize_performance(self.true_errors, self.y_noises, self.measurements, 'y_noises', 'measurements', file_name=self.file_name + '_error_true')
        visualize_performance(self.solver_errors, self.y_noises, self.measurements, 'y_noises', 'measurements', file_name=self.file_name + '_error_solver')

        
class Figure7(TaskPull):

    def __init__(self, measurements=None, random_fractions=None, seed=None, noise_amplitude=None, file_name=None):
        """
        """
        self.measurements = measurements
        self.seed = seed
        self.n_restarts = 5
        self.verbosity = 1
        self.shape = (100, 110)
        self.noise_amplitude = noise_amplitude
        self.file_name = file_name
        self.sparsity = 2
        self.rank = 2
        self.random_fractions = random_fractions
        
    def allocate(self):
        self.true_errors = np.ones((len(self.measurements), len(self.random_fractions))) * np.nan
        self.solver_errors = np.ones((len(self.measurements), len(self.random_fractions))) * np.nan

    def create_tasks(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        for i, m in enumerate(self.measurements):
            for j, random_fraction in enumerate(self.random_fractions):
                secondary_seed = np.random.randint(0, 1e8)
                task = i, j, m, random_fraction, secondary_seed
                print task
                yield task

    def work(self, task):
        i, j, m, random_fraction, secondary_seed = task
        np.random.seed(secondary_seed)

        data = DataSimulated(self.shape, self.rank, noise_amplitude=self.noise_amplitude)
        operator = LinearOperatorKsparse(data, m, self.sparsity).generate()
        A, y = operator['A'], operator['y']
        shuffle_indices = int(random_fraction * len(y))
        shuffle_indices_subset = deepcopy(y[:shuffle_indices])
        np.random.shuffle(shuffle_indices_subset)
        y[:shuffle_indices] = shuffle_indices_subset
        cost = Cost(A, y)
        solver = ConjugateGradientSolver(cost.cost_func, guess_func, data, self.rank, self.n_restarts, noise_amplitude=self.noise_amplitude, verbosity=self.verbosity)
        data = solver.recover()

        error_solver = cost.cost_func(data.d['sample']['estimated_bias'])
        divisor = np.sum(~np.isnan(data.d['sample']['mixed']))
        error = np.nansum(np.absolute(data.d['sample']['signal'] - (data.d['sample']['mixed'] - data.d['sample']['estimated_bias']))) / divisor
        zero_error = np.nansum(np.absolute(data.d['sample']['signal'] - data.d['sample']['mixed'])) / divisor
        error_true = error / zero_error

        #print error_true, error_solver, i, j, 'error_true, error_solver, i, j'
        return error_true, error_solver, i, j

    def store(self, result):
        error_true, error_solver, i, j = result
        self.true_errors[i, j] = error_true
        self.solver_errors[i, j] = error_solver

    def postprocessing(self):
        visualize_performance(self.true_errors, self.random_fractions, self.measurements, 'random_fractions', 'measurements', file_name=self.file_name + '_error_true')
        visualize_performance(self.solver_errors, self.random_fractions, self.measurements, 'random_fractions', 'measurements', file_name=self.file_name + '_error_solver')
        

class Figure8(TaskPull):

    def __init__(self, measurements=None, random_fractions=None, seed=None, noise_amplitude=None, file_name=None):
        """
        """
        self.measurements = measurements
        self.seed = seed
        self.n_restarts = 5
        self.verbosity = 1
        self.shape = (100, 110)
        self.noise_amplitude = noise_amplitude
        self.file_name = file_name
        self.rank = 2
        self.random_fractions = random_fractions
        self.m_blocks_factor = self.shape[0]
        
    def allocate(self):
        self.true_errors = np.ones((len(self.measurements), len(self.random_fractions))) * np.nan
        self.solver_errors = np.ones((len(self.measurements), len(self.random_fractions))) * np.nan

    def create_tasks(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        for i, m in enumerate(self.measurements):
            for j, random_fraction in enumerate(self.random_fractions):
                secondary_seed = np.random.randint(0, 1e8)
                task = i, j, m, random_fraction, secondary_seed
                print task
                yield task

    def work(self, task):
        i, j, m, random_fraction, secondary_seed = task
        np.random.seed(secondary_seed)

        data = DataSimulated(self.shape, self.rank, m_blocks_factor=self.m_blocks_factor, noise_amplitude=self.noise_amplitude)

        sample_pairs = shuffle_some_pairs(data.d['sample']['true_pairs'], random_fraction, data.d['sample']['shape'][0])
        feature_pairs = shuffle_some_pairs(data.d['feature']['true_pairs'], random_fraction, data.d['feature']['shape'][0])
        
        print "random_fraction", random_fraction # "random_fraction * data.d['sample']['shape'][0]", random_fraction * data.d['sample']['shape'][0]
        print "data.d['sample']['true_pairs'][:5]", "sample_pairs[:5]", sample_pairs[:5], data.d['sample']['true_pairs'][:5]
        print "data.d['sample']['true_pairs'][-5:]", "sample_pairs[-5:]", sample_pairs[-5:], data.d['sample']['true_pairs'][-5:]
        
        data.estimate(true_pairs={'sample': sample_pairs, 'feature': feature_pairs}, true_directions={'sample': data.d['sample']['true_directions'], 'feature': data.d['feature']['true_directions']}, true_stds={'sample': data.d['sample']['true_stds'], 'feature': data.d['feature']['true_stds']})
        #data.estimate(true_pairs={'sample': data.d['sample']['true_pairs'], 'feature': data.d['feature']['true_pairs']}, true_directions={'sample': data.d['sample']['true_directions'], 'feature': data.d['feature']['true_directions']}, true_stds={'sample': data.d['sample']['true_stds'], 'feature': data.d['feature']['true_stds']})
        operator = LinearOperatorCustom(data, m).generate()
        A, y = operator['A'], operator['y']
        cost = Cost(A, y)
        solver = ConjugateGradientSolver(cost.cost_func, guess_func, data, self.rank, self.n_restarts, noise_amplitude=self.noise_amplitude, verbosity=self.verbosity)
        data = solver.recover()

        error_solver = cost.cost_func(data.d['sample']['estimated_bias'])
        divisor = np.sum(~np.isnan(data.d['sample']['mixed']))
        error = np.nansum(np.absolute(data.d['sample']['signal'] - (data.d['sample']['mixed'] - data.d['sample']['estimated_bias']))) / divisor
        zero_error = np.nansum(np.absolute(data.d['sample']['signal'] - data.d['sample']['mixed'])) / divisor
        error_true = error / zero_error

        #print error_true, error_solver, i, j, 'error_true, error_solver, i, j'
        return error_true, error_solver, i, j

    def store(self, result):
        error_true, error_solver, i, j = result
        self.true_errors[i, j] = error_true
        self.solver_errors[i, j] = error_solver

    def postprocessing(self):
        visualize_performance(self.true_errors, self.random_fractions, self.measurements, 'random_fractions', 'measurements', file_name=self.file_name + '_error_true')
        visualize_performance(self.solver_errors, self.random_fractions, self.measurements, 'random_fractions', 'measurements', file_name=self.file_name + '_error_solver')


def submit(kwargs, run_class, mode='local', ppn=1, hours=10000, nodes=1, path='/home/sohse/projects/PUBLICATION/GITssh/bcn'):
    """Submits jobs to a SunGRID engine or Torque.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments to run_class.
    run_class : class
        The class to run, e.g. Figure 1-8.
    mode : str, {'local', 'parallel'}
        Run locally (for debugging purposes) or in parallel on cluster.
    ppn : int
        Number of processes per node.
    hours : int
        Maximum number of hours to run.
    nodes : int
        Number of nodes to use.
    path : str
        Path to work in (e.g. store logs, output and use bcn.py from).
    """
    if mode == 'local':
        subprocess.call(['python', path + '/taskpull_local.py', run_class, json.dumps(kwargs)])

    if mode == 'parallel':
        output, input_ = popen2('qsub')
        job = """#!/bin/bash
                 #PBS -S /bin/bash
                 #PBS -l nodes={nodes}:ppn={ppn},walltime={hours}:00:00
                 #PBS -N {jobname}
                 #PBS -o {path}/logs/{jobname}.out
                 #PBS -e {path}/logs/{jobname}.err
                 export LD_LIBRARY_PATH="${{LD_LIBRARY_PATH}}"
                 export PATH=/home/sohse/anaconda2/bin:$PATH
                 echo $PATH
                 echo ""
                 echo -n "This script is running on: "
                 hostname -f
                 date
                 echo ""
                 echo "PBS_NODEFILE (${{PBS_NODEFILE}})"
                 echo ""
                 cat ${{PBS_NODEFILE}}
                 echo ""
                 cd $PBS_O_WORKDIR
                 /opt/openmpi/1.6.5/gcc/bin/mpirun python {path}/taskpull.py {run_class} '{json}'
                 """.format(run_class=run_class, nodes=nodes, jobname=run_class, json=json.dumps(kwargs), ppn=ppn, hours=hours, path=path)
        input_.write(job)
        input_.close()
        print 'Submitted {output}'.format(output=output.read())




if __name__ == '__main__':
    '''
    run_class = 'Figure3'
    file_name = 'out/' + run_class
    kwargs = {'measurements': list(np.asarray(np.logspace(np.log10(1e2), np.log10(1e3), 8), dtype=int)), 'ranks': list(np.asarray(np.linspace(1, 8, 8), dtype=int)), 'seed': 42, 'noise_amplitude': 5.0, 'file_name': file_name}
    submit(kwargs, mode='parallel', run_class=run_class)
    print run_class, kwargs
    
    run_class = 'Figure4'
    file_name = 'out/' + run_class
    kwargs = {'measurements': list(np.asarray(np.logspace(np.log10(1e2), np.log10(1e3), 8), dtype=int)), 'ranks': list(np.asarray(np.linspace(1, 8, 8), dtype=int)), 'seed': 42, 'noise_amplitude': 5.0, 'file_name': file_name}
    submit(kwargs, mode='parallel', run_class=run_class)
    print run_class, kwargs

    
    run_class = 'Figure2' # WARNING Max. of 2 processes per node, otherwise memory error. # WARNING Also hardcoded number of measurements for solver convergence issues.
    file_name = 'out/' + run_class
    kwargs = {'measurement_percentages': list(np.asarray(np.logspace(np.log10(0.01), np.log10(1.0), 8))), 'dimensions': [(10, 10), (25, 25), (50, 50), (100, 100), (200, 200)], 'seed': 42, 'noise_amplitude': 5.0, 'file_name': file_name} # , (400, 400), (800, 800)
    submit(kwargs, mode='parallel', run_class=run_class)
    print run_class, kwargs

    
    run_class = 'Figure1'
    file_name = 'out/' + run_class
    kwargs = {'measurements': list(np.asarray(np.logspace(np.log10(300), np.log10(1900), 8), dtype=int)), 'sparsities': list(np.asarray(np.arange(1, 9), dtype=int)), 'seed': 42, 'noise_amplitude': 5.0, 'file_name': file_name}
    submit(kwargs, mode='parallel', run_class=run_class)
    print run_class, kwargs

    
    run_class = 'Figure5'
    file_name = 'out/' + run_class
    kwargs = {'measurements': list(np.asarray(np.logspace(np.log10(500), np.log10(1300), 8), dtype=int)), 'y_noises': list(np.logspace(np.log10(0.001), np.log10(0.2), 8)), 'seed': 42, 'noise_amplitude': 5.0, 'file_name': file_name}
    submit(kwargs, mode='parallel', run_class=run_class)
    print run_class, kwargs

    run_class = 'Figure6'
    file_name = 'out/' + run_class
    kwargs = {'measurements': list(np.asarray(np.logspace(np.log10(500), np.log10(1300), 8), dtype=int)), 'y_noises': list(np.logspace(np.log10(0.001), np.log10(0.2), 8)), 'seed': 42, 'noise_amplitude': 5.0, 'file_name': file_name}
    submit(kwargs, mode='parallel', run_class=run_class)
    print run_class, kwargs
    
    
    run_class = 'Figure7' 
    file_name = 'out/' + run_class
    kwargs = {'measurements': list(np.asarray(np.logspace(np.log10(1e2), np.log10(1e4), 8), dtype=int)), 'random_fractions': list(np.logspace(np.log10(0.01), np.log10(1.0), 8)), 'seed': 42, 'noise_amplitude': 5.0, 'file_name': file_name + 'fast'}
    submit(kwargs, mode='parallel', nodes=12, ppn=6, run_class=run_class)
    print run_class, kwargs
    '''
    run_class = 'Figure8'
    file_name = 'out/' + run_class
    kwargs = {'measurements': list(np.asarray(np.logspace(np.log10(1e2), np.log10(5e4), 8), dtype=int)), 'random_fractions': [0.0, 1.0/50**2, 2.0/50**2, 3.0/50**2, 5.0/50**2, 10.0/50**2, 25.0/50**2, 1.0], 'seed': 42, 'noise_amplitude': 5.0, 'file_name': file_name + '_log'}
    # list(np.logspace(np.log10(0.001), np.log10(0.5), 8))[:2] # list(np.logspace(np.log10(0.0001), np.log10(0.4), 8))
    submit(kwargs, mode='parallel', ppn=2, nodes=20, run_class=run_class)
    print run_class, kwargs
   
    run_class = 'Figure8'
    file_name = 'out/' + run_class
    kwargs = {'measurements': [200, 400, 800, 1000, 1200, 2000, 3000, 10000], 'random_fractions': [0, 1, 2, 3, 4, 5, 6, 10], 'seed': 42, 'noise_amplitude': 5.0, 'file_name': file_name + '_custom'}
    # list(np.logspace(np.log10(0.001), np.log10(0.5), 8))[:2] # list(np.logspace(np.log10(0.0001), np.log10(0.4), 8))
    submit(kwargs, mode='parallel', ppn=2, nodes=20, run_class=run_class)
    print run_class, kwargs
   













































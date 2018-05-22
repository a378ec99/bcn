"""Parallel computation of figures.
Notes
-----
Defines classes that are used to run multiple bias recoveries in parallel to generate different figures.
"""
from __future__ import division, absolute_import

from abc import ABCMeta, abstractmethod
import subprocess
import json
from popen2 import popen2
from copy import deepcopy

import numpy as np

from bcn.data import DataSimulated
from bcn.utils.visualization import visualize_performance, visualize_dependences
from bcn.solvers import ConjugateGradientSolver
from bcn.cost import Cost
from bcn.linear_operators import LinearOperatorCustom, LinearOperatorKsparse
from bcn.bias import guess_func


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


def shuffle_some_pairs(pairs, n, max_indices):
    """
    Shuffles a fraction of pairs randomly.
    Parameters
    ----------
    pairs : ndarray (n, 2)
        Sequence of pairs used as integer indices to an array.
    n : int
        Number of pairs that are to be randomized.
    max_indices : int
        Maximum value of indices that are allowed for the pairs.
    Returns
    -------
    pairs_new : ndarray (n, 2)
        Sequence of pairs shuffled accordingly.
    """
    if n == len(pairs):
        pairs_new = generate_random_pairs(n, max_indices)
    elif n == 0:
        pairs_new = pairs
    else:
        pairs_new = deepcopy(pairs)
        x = generate_random_pairs(n, max_indices)
        pairs_new[-n:, :] = x
    return pairs_new


def generate_random_pairs(n, max_indices):
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
    for i in xrange(n):
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

        return error_true, error_solver, i, j

    def store(self, result):
        error_true, error_solver, i, j = result
        self.true_errors[i, j] = error_true
        self.solver_errors[i, j] = error_solver

    def postprocessing(self):
        visualize_performance(self.true_errors, self.sparsities, self.measurements, 'sparsities', 'measurements', file_name=self.file_name + '_error_true')
        visualize_performance(self.solver_errors, self.sparsities, self.measurements, 'sparsities', 'measurements', file_name=self.file_name + '_error_solver')


def submit(kwargs, run_class, mode='local', ppn=1, hours=10000, nodes=1, path='/home/sohse/projects/bcn'):
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
        subprocess.call(['python', path + '/bcn/utils/taskpull_local.py', run_class, json.dumps(kwargs)])

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
                 /opt/openmpi/1.6.5/gcc/bin/mpirun python {path}/bcn/utils/taskpull.py {run_class} '{json}'
                 """.format(run_class=run_class, nodes=nodes, jobname=run_class, json=json.dumps(kwargs), ppn=ppn, hours=hours, path=path)
        input_.write(job)
        input_.close()
        print 'Submitted {output}'.format(output=output.read())




if __name__ == '__main__':
    run_class = 'Figure1'
    file_name = '../out/' + run_class
    kwargs = {'measurements': list(np.asarray(np.logspace(np.log10(1e2), np.log10(1e3), 8), dtype=int)), 'ranks': list(np.asarray(np.linspace(1, 8, 8), dtype=int)), 'seed': 42, 'noise_amplitude': 5.0, 'file_name': file_name}
    submit(kwargs, mode='parallel', run_class=run_class)
    print run_class, kwargs

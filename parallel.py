"""Parallel computations module.

Notes
-----
This module defines a class that can be used to run multiple bias recoveries in parallel.

# TODO brainstorm what want to use it for?
# TODO what should its output be, visualize?
# TODO combine with plotting visualize ... make simpler?
# TODO set parameters for a run somewhere...

# TODO put cost class into solvers class!

"""
from __future__ import division, absolute_import


__all__ = ['Simulation', 'BlindRecovery']

from abc import ABCMeta, abstractmethod
import subprocess
import json
from popen2 import popen2
import numpy as np
from data import DataSimulated
from visualization import visualize_performance, visualize_dependences
from solvers import ConjugateGradientSolver
from cost import Cost
from linear_operators import LinearOperatorBlind


class TaskPull(object):
    '''
    Abstract class that denotes API to taskpull.py and taskpull_local.py.
    '''
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

        
class Simulation(TaskPull):
    
    def __init__(self, measurements=None, ranks=None, seed=None, noise_amplitude=None, file_name='test_performance_visualization'):
        """Simulates the signal and bias to be recoverd and does recovery for different rank and measurements settings.
        """
        #self.replicates = None
        self.ranks = ranks
        self.measurements = measurements
        self.seed = seed
        self.n_restarts = 8
        self.verbosity = 1
        self.shape = shape=(50, 60)
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
        
        data = DataSimulated(self.shape, rank, noise_amplitude=self.noise_amplitude) # true_stds={'sample': data.d['sample']['true_stds'], 'feature': data.d['feature']['true_stds']}, 
        data.estimate(true_pairs={'sample': data.d['sample']['true_pairs'], 'feature': data.d['feature']['true_pairs']}, true_directions={'sample': data.d['sample']['true_directions'], 'feature': data.d['feature']['true_directions']})
        operator = LinearOperatorBlind(data.d, m).generate()
        A, y = operator['A'], operator['y']
        cost = Cost(A, y)
        solver = ConjugateGradientSolver(cost.cost_function, data, rank, self.n_restarts, verbosity=self.verbosity)
        data.d = solver.recover()
        
        visualize_dependences(data.d, file_name=self.file_name + '_dependences_{}_{}'.format(m, rank))

        error_solver = cost.cost_function(data.d['sample']['estimated_bias'])
        # TODO implement nan checks!
        divisor = np.sum(~np.isnan(data.d['sample']['mixed']))
        error = np.nansum(np.absolute(data.d['sample']['signal'] - (data.d['sample']['mixed'] - data.d['sample']['estimated_bias']))) / divisor
        zero_error = np.nansum(np.absolute(data.d['sample']['signal'] - data.d['sample']['mixed'])) / divisor
                
        error_true = error / zero_error #np.sqrt(np.mean((data.d['sample']['estimated_bias'] - data.d['sample']['true_bias'])**2))
        # plotting option for diagnostics for each run? That's why self.free_x_name, self.free_y_name may have to be passed as arguments

        print error_true, error_solver, i, j, 'error_true, error_solver, i, j'
        return error_true, error_solver, i, j

    def store(self, result):
        error_true, error_solver, i, j = result
        self.true_errors[i, j] = error_true
        self.solver_errors[i, j] = error_solver

    def postprocessing(self):
        visualize_performance(self.true_errors, self.measurements, self.ranks, 'measurements', 'rank', self.file_name + '_error_true')
        visualize_performance(self.solver_errors, self.measurements, self.ranks, 'measurements', 'rank', self.file_name + '_error_solver')
        
        #if unittest == True:
        #    np.save(self.parameters['name'] + '_' + self.parameters['mode'], self.X)
        #    with open(self.parameters['name'] + '_complete.token', 'w') as f:
        #        pass


def submit(kwargs, run_class='Simulation', mode='local', ppn=12, hours=10000, nodes=1, path='/home/sohse/projects/PUBLICATION/GITrefactored/bcn'):
    """Submit jobs to the SunGRID engine / Torque.
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
                 """.format(run_class=run_class, nodes=nodes, jobname='test', json=json.dumps(kwargs), ppn=ppn, hours=hours, path=path)
        input_.write(job)
        input_.close()
        print 'Submitted {output}'.format(output=output.read())




if __name__ == '__main__':

    file_name = 'parallel_estimate_estimate_std'
    for noise_amplitude in [0.1, 1.0, 10.0, 20.0, 50.0, 100.0]:
        name = file_name + '_' + str(noise_amplitude).split('.')[0]
        kwargs = {'measurements': [2000], 'ranks': [2], 'seed': 42, 'noise_amplitude': noise_amplitude, 'file_name': name} # 100, 500, 1000, 2000, 5000 # 1, 2, 3, 4, 5
        submit(kwargs, mode='parallel')

    
'''

    def signal_to_noise(self, signal, noise):
        # NOTE Sum of std(clean signal) / std(true noise) for each
        # feature/sample column.
        snr = np.mean(np.std(signal['X'][:, signal['sample']['pairs'].ravel()], axis=0) / np.mean(np.absolute(noise['X'][:, signal['sample']['pairs'].ravel()]), axis=0)) + \
            np.mean(np.std(signal['X'][signal['feature']['pairs'].ravel()], axis=1) / np.mean(
                np.absolute(noise['X'][signal['feature']['pairs'].ravel()]), axis=1))
        return snr
'''
                
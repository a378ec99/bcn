"""Benchmarking of different solvers for the case of imperfect correlations.
"""
from __future__ import division, absolute_import

import numpy as np

from bcn.bias import guess_func
from bcn.data import DataSimulated, estimate_partial_signal_characterists
from bcn.cost import Cost, CostMiniBatch
from bcn.solvers import ConjugateGradientSolver
from bcn.linear_operators import LinearOperatorCustom, possible_measurements
from bcn.utils.visualization import recovery_performance

from joblib import Parallel, delayed


def test(correlation_strength, solver='minibatch'):

    # Setup of general parameters for the recovery experiment.
    n_restarts = 10
    rank = 6
    n_measurements = 2800
    shape = (50, 70) # samples, features
    missing_fraction = 0.1
    noise_amplitude = 2.0
    m_blocks_size = 5 # size of each block
    correlation_threshold = 0.75
    correlation_strength = correlation_strength
    bias_model = 'image'

    # Creation of the true signal for both datasets.
    truth = DataSimulated(shape, rank, bias_model=bias_model, m_blocks_size=m_blocks_size, noise_amplitude=noise_amplitude, correlation_strength=correlation_strength, missing_fraction=missing_fraction)

    true_correlations = {'sample': truth.d['sample']['true_correlations'], 'feature': truth.d['feature']['true_correlations']}
    true_pairs = {'sample': truth.d['sample']['true_pairs'], 'feature': truth.d['feature']['true_pairs']}
    true_directions = {'sample': truth.d['sample']['true_directions'], 'feature': truth.d['feature']['true_directions']}
    true_stds = {'sample': truth.d['sample']['true_stds'], 'feature': truth.d['feature']['true_stds']}

    mixed = truth.d['sample']['mixed']

    # Prior information estimated from the corrputed signal and used for blind recovery.
    signal_characterists = estimate_partial_signal_characterists(mixed, correlation_threshold, true_correlations=true_correlations, true_pairs=true_pairs, true_directions=true_directions, true_stds=true_stds)

    # Construct measurements from corrupted signal and its estimated partial characteristics.
    operator = LinearOperatorCustom(n_measurements)
    measurements = operator.generate(signal_characterists)

    # Construct cost function.

    if solver == 'minibatch':
        cost = CostMiniBatch(measurements['A'], measurements['y'], sparsity=2, batch_size=10)
    else:
        cost = Cost(measurements['A'], measurements['y'], sparsity=2)

    # Recover the bias.
    solver = ConjugateGradientSolver(mixed, cost.cost_func, guess_func, rank, guess_noise_amplitude=noise_amplitude, verbosity=0)
    results = solver.recover()
    
    d = recovery_performance(mixed, cost.cost_func, truth.d['sample']['true_bias'], results['estimated_signal'], truth.d['sample']['signal'], results['estimated_bias'])
    error = d['Mean absolute error (estimated_signal)']
    
    return error


r = Parallel(n_jobs=2)(delayed(test)(correlation_strength) for correlation_strength in np.linspace(0.9, 1.0, 10))
print r



#if __name__ == '__main__':
#    pass
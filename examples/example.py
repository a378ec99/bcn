"""Example scipt showcasing blind compressive normalization on simulated data.
"""

from __future__ import division, absolute_import


import sys # WARNING remove in final version
sys.path.append('/home/sohse/projects/PUBLICATION/GITssh/bcn')

import numpy as np

from bcn.bias import guess_func
from bcn.data import DataSimulated, DataBlind
from bcn.cost import Cost
from bcn.solvers import ConjugateGradientSolver
from bcn.linear_operators import LinearOperatorCustom, possible_measurement_range
from bcn.utils.visualization import visualize_dependences, visualize_correlations, visualize_absolute


if __name__ == '__main__':
    
    np.random.seed(seed=42)

    # General setup of the recovery experiment.
    n_restarts = 10 # 15
    rank = 6
    n_measurements = 10000    
    shape = (100, 110) #NOTE samples, features
    missing_fraction = 0.1
    noise_amplitude = 30.0
    print 'possible_measurement_range', possible_measurement_range(shape, missing_fraction)

    # Creation of the test data and blind estimation of the dependency structure.
    truth = DataSimulated(shape, rank, model='image', correlation_threshold=0.9, m_blocks_factor=shape[0] // 2, noise_amplitude=noise_amplitude)
    #FIXME visualize_absolute(truth, file_name='../out/test_absolute_truth_{}'.format(n_measurements))
    mixed = truth.d['sample']['mixed']
    blind = DataBlind(mixed, rank, correlation_threshold=0.9) # 0.85
    blind.estimate() # true_pairs={'sample': truth.d['sample']['true_pairs'], 'feature':truth.d['feature']['true_pairs']}, true_directions={'sample': truth.d['sample']['true_directions'], 'feature': truth.d['feature']['true_directions']}, true_stds={'sample': truth.d['sample']['true_stds'], 'feature': truth.d['feature']['true_stds']}
    visualize_correlations(blind, file_name='../out/test_image_correlations_estimated_blind_{}'.format(n_measurements), truth_available=False)

    # Construction of the measurement operator and measurements from the data.
    operator = LinearOperatorCustom(blind, n_measurements).generate()
    A = operator['A']
    y = operator['y']
    cost = Cost(A, y)
    
    # Setup and run of the recovery with the standard solver.
    solver = ConjugateGradientSolver(cost.cost_func, guess_func, blind, rank, n_restarts, verbosity=0)
    recovered = solver.recover()

    # Add information about true signal to visualize recovery performance.
    for space in ['sample', 'feature']:
        recovered.d[space]['true_pairs'] = truth.d[space]['true_pairs']
        recovered.d[space]['signal'] = truth.d[space]['signal']
        recovered.d[space]['true_missing'] = truth.d[space]['true_missing']
        recovered.d[space]['true_bias'] = truth.d[space]['true_bias']
        recovered.d[space]['true_correlations'] = truth.d[space]['true_correlations']
        recovered.d[space]['true_pairs'] = truth.d[space]['true_pairs']
        recovered.d[space]['true_stds'] = truth.d[space]['true_stds']
        recovered.d[space]['true_directions'] = truth.d[space]['true_directions']
        recovered.noise_amplitude = noise_amplitude
    visualize_dependences(recovered, file_name='../out/test_image_dependences_blind_{}'.format(n_measurements), truth_available=True, estimate_available=True, recovery_available=True)
    
    # Print and visualize the recovery performance statistics.
    error_solver = cost.cost_func(recovered.d['sample']['estimated_bias'])
    divisor = np.sum(~np.isnan(recovered.d['sample']['mixed']))
    error_ideal = np.nansum(np.absolute(recovered.d['sample']['signal'] - (recovered.d['sample']['mixed'] - recovered.d['sample']['true_bias']))) / divisor
    error = np.nansum(np.absolute(recovered.d['sample']['signal'] - (recovered.d['sample']['mixed'] - recovered.d['sample']['estimated_bias']))) / divisor
    zero_error = np.nansum(np.absolute(recovered.d['sample']['signal'] - recovered.d['sample']['mixed'])) / divisor
    error_true = error / zero_error #np.sqrt(np.mean((recovered.d['sample']['estimated_bias'] - recovered.d['sample']['true_bias'])**2))
    
    print 'error_ideal', error_ideal
    print 'zero_error', zero_error
    print 'error_true_absolute', error
    print 'error_zero_absolute', zero_error
    print 'error_true_ratio', error_true
    print 'error_solver', error_solver

    visualize_absolute(recovered, file_name='../out/test_image_absolute_recovered_{}'.format(n_measurements), recovered=True)

















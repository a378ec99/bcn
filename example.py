"""Example scipt showcasing blind compressive normalization on simulated data.
"""

from __future__ import division, absolute_import

import sys
sys.path.append('/home/sohse/projects/PUBLICATION/GITssh/bcn')

import numpy as np

from bias import guess_func
from data import DataSimulated, DataBlind
from cost import Cost
from solvers import ConjugateGradientSolver
from linear_operators import LinearOperatorCustom, min_measurements, max_measurements
from visualization import visualize_dependences, visualize_correlations, visualize_absolute


if __name__ == '__main__':

    np.random.seed(seed=42)
    
    rank = 2
    n_measurements = 10000 # 16000 # NOTE that at some points wil be just repeating the same mesurements... need to see how many unique are possible and likely! Depends on correlation structure, e.g. if few blocks then lots of possiblities.

    #for rank, shape in zip(range(5), [int(i) for i in np.linspace(100, 1000, 5)]):

    shape = (100, 110) # (150, 160) # NOTE sample, feature

    print 'min', min_measurements(shape), 50 * 105 * 2 # m_blocks factor is is 2 # 155 # 70 #- 10% missing # - 20% not estimated propoerly!
    print 'max', max_measurements(shape), 2500 * 105 * 2 # m_blocks factor is is 2 # 155 # 5600 #  - 10% missing # - 20 % not estaimted properly!
    
    truth = DataSimulated(shape, rank, noise_amplitude=30.0) # 100
    #visualize_dependences(truth, file_name='out/test_dependences_truth', truth_available=True, estimate_available=False, recovery_available=False)
    visualize_correlations(truth, file_name='out/test_correlations_truth_{}'.format(n_measurements), truth_available=True)

    truth.estimate()
    #visualize_dependences(truth, file_name='out/test_dependences_truth_estimate', truth_available=True, estimate_available=True, recovery_available=False)
    visualize_correlations(truth, file_name='out/test_correlations_estimated_truth_{}'.format(n_measurements), truth_available=False)
    visualize_absolute(truth, file_name='out/test_absolute_truth_{}'.format(n_measurements))




    mixed = truth.d['sample']['mixed']

    blind = DataBlind(mixed, rank)
    #visualize_dependences(blind, file_name='out/test_dependences_blind', truth_available=False, estimate_available=False, recovery_available=False)

    
    blind.estimate(true_pairs={'sample': truth.d['sample']['true_pairs'], 'feature':truth.d['feature']['true_pairs']}, true_directions={'sample': truth.d['sample']['true_directions'], 'feature': truth.d['feature']['true_directions']}, true_stds={'sample': truth.d['sample']['true_stds'], 'feature': truth.d['feature']['true_stds']}) # 
    #visualize_dependences(blind, file_name='out/test_dependences_blind_estimate', truth_available=False, estimate_available=True, recovery_available=False)
    visualize_correlations(blind, file_name='out/test_correlations_estimated_blind_{}'.format(n_measurements), truth_available=False)

    n_restarts = 10

    operator = LinearOperatorCustom(blind, n_measurements).generate()
    A = operator['A']
    y = operator['y']
    cost = Cost(A, y)

    solver = ConjugateGradientSolver(cost.cost_func, guess_func, blind, rank, n_restarts, verbosity=0)
    recovered = solver.recover()

    #visualize_dependences(recovered, file_name='out/test_dependences_blind_recovered', truth_available=False, estimate_available=True, recovery_available=True)

    for space in ['sample', 'feature']:
        recovered.d[space]['true_pairs'] = truth.d[space]['true_pairs']
        recovered.d[space]['signal'] = truth.d[space]['signal']
        recovered.d[space]['true_missing'] = truth.d[space]['true_missing']
        recovered.d[space]['true_bias'] = truth.d[space]['true_bias']
        recovered.d[space]['true_correlations'] = truth.d[space]['true_correlations']
        recovered.d[space]['true_pairs'] = truth.d[space]['true_pairs']
        recovered.d[space]['true_stds'] = truth.d[space]['true_stds']
        recovered.d[space]['true_directions'] = truth.d[space]['true_directions']

    #visualize_dependences(recovered, file_name='out/test_dependences_blind_truth_recovered', truth_available=True, estimate_available=True, recovery_available=True)

    #for space in ['sample', 'feature']:
    #    recovered.d[space]['estimated_directions'] = truth.d[space]['estimated_directions']
    #    recovered.d[space]['estimated_pairs'] = truth.d[space]['estimated_pairs']
    #    recovered.d[space]['estimated_stds'] = truth.d[space]['estimated_stds']

    visualize_dependences(recovered, file_name='out/test_dependences_blind_{}'.format(n_measurements), truth_available=True, estimate_available=True, recovery_available=True)

    
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



















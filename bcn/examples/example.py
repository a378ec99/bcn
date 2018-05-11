"""Example scipt showcasing blind compressive normalization on simulated data.
"""

from __future__ import division, absolute_import

import numpy as np

from bcn.bias import guess_func
from bcn.data import DataSimulated, DataBlind
from bcn.cost import Cost
from bcn.solvers import ConjugateGradientSolver
from bcn.linear_operators import LinearOperatorCustom, possible_measurements
#from bcn.utils.visualization import visualize_dependences, visualize_correlations, visualize_absolute

import pylab as pl

if __name__ == '__main__':
    
    np.random.seed(seed=42)

    fig = pl.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot([1,2,3,4], [5, 6, 7, 8], '.')
    ax.set_xlabel('features')
    ax.set_ylabel('samples')
    fig.savefig('test')
    
    # Setup of general parameters for the recovery experiment.
    sparsity = 2
    n_restarts = 10
    rank = 6
    n_measurements = 100
    shape = (100, 110) # samples, features
    missing_fraction = 0.1
    noise_amplitude = 30.0
    m_blocks_size = 50 # size of each block
    correlation_threshold = 0.9
    bias_model = 'image'
    
    print possible_measurements(shape, missing_fraction, m_blocks_size=m_blocks_size)

    # Creation of the corrupted signal.
    truth = DataSimulated(shape, rank, bias_model=bias_model, correlation_threshold=correlation_threshold, m_blocks_size=m_blocks_size, noise_amplitude=noise_amplitude, missing_fraction=missing_fraction)
    mixed = truth.d['sample']['mixed']

    # print mixed, mixed.dtype

    # WARNING Not identical to notebook!
    
    blind = DataBlind(mixed, rank, correlation_threshold=0.9) # 0.85
    blind.estimate()
    ###visualize_correlations(blind, file_name='../../out/test_image_correlations_estimated_blind_{}'.format(n_measurements), truth_available=False)

    estimated = {'sample': {'estimated_stds': blind.d['sample']['estimated_stds'],
                            'estimated_directions': blind.d['sample']['estimated_directions'],
                            'estimated_pairs': blind.d['sample']['estimated_pairs'],
                            'shape': blind.d['sample']['shape']},
                'feature': {'estimated_stds': blind.d['feature']['estimated_stds'],
                            'estimated_directions': blind.d['feature']['estimated_directions'],
                            'estimated_pairs': blind.d['feature']['estimated_pairs'],
                            'shape': blind.d['feature']['shape']}
            }

    # Construction of the measurement operator and measurements from the data.
    operator = LinearOperatorCustom(n_measurements)
    out = operator.generate(mixed, estimated)
    cost = Cost(out['A'], out['y'], sparsity)
    
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
    ###visualize_dependences(recovered, file_name='../../out/test_image_dependences_blind_{}'.format(n_measurements), truth_available=True, estimate_available=True, recovery_available=True)
    
    # Print and visualize the recovery performance statistics.
    error_cost_func_true_bias = cost.cost_func(recovered.d['sample']['true_bias'])
    error_cost_func_estimated_bias = cost.cost_func(recovered.d['sample']['estimated_bias'])
    print 'error_cost_func_true_bias', error_cost_func_true_bias
    print 'error_cost_func_estimated_bias', error_cost_func_estimated_bias
    
    divisor = np.sum(~np.isnan(recovered.d['sample']['mixed']))
    print 'number of valid values', divisor
    mean_absolute_error_true_signal = np.nansum(np.absolute(recovered.d['sample']['signal'] - (recovered.d['sample']['mixed'] - recovered.d['sample']['true_bias']))) / divisor
    mean_absolute_error_estimated_signal = np.nansum(np.absolute(recovered.d['sample']['signal'] - (recovered.d['sample']['mixed'] - recovered.d['sample']['estimated_bias']))) / divisor
    print 'mean_absolute_error_true_signal', mean_absolute_error_true_signal
    print 'mean_absolute_error_estimated_signal', mean_absolute_error_estimated_signal
    
    mean_absolute_error_zeros = np.nansum(np.absolute(recovered.d['sample']['signal'] - recovered.d['sample']['mixed'])) / divisor
    print 'mean_absolute_error_zeros', mean_absolute_error_zeros
    
    ratio_estimated_signal_to_zeros = mean_absolute_error_estimated_signal / mean_absolute_error_zeros
    print 'ratio_estimated_signal_to_zeros', ratio_estimated_signal_to_zeros
    
    ###visualize_absolute(recovered, file_name='../../out/test_image_absolute_recovered_{}'.format(n_measurements), recovered=True)



    

    # Repeat with known values (optimal estimates)

    estimated = {'sample': {'estimated_stds': truth.d['sample']['true_stds'],
                        'estimated_directions': truth.d['sample']['true_directions'],
                        'estimated_pairs': truth.d['sample']['true_pairs'],
                        'shape': truth.d['sample']['shape']},
            'feature': {'estimated_stds': truth.d['feature']['true_stds'],
                        'estimated_directions': truth.d['feature']['true_directions'],
                        'estimated_pairs': truth.d['feature']['true_pairs'],
                        'shape': truth.d['feature']['shape']}
        }

    # Construction of the measurement operator and measurements from the data.
    operator = LinearOperatorCustom(n_measurements)
    out = operator.generate(mixed, estimated)
    cost = Cost(out['A'], out['y'], sparsity)

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
    ###visualize_dependences(recovered, file_name='../../out/test_image_dependences_blind_{}'.format(n_measurements), truth_available=True, estimate_available=True, recovery_available=True)

    # Print and visualize the recovery performance statistics.
    error_cost_func_true_bias = cost.cost_func(recovered.d['sample']['true_bias'])
    error_cost_func_estimated_bias = cost.cost_func(recovered.d['sample']['estimated_bias'])
    print 'error_cost_func_true_bias', error_cost_func_true_bias
    print 'error_cost_func_estimated_bias', error_cost_func_estimated_bias

    divisor = np.sum(~np.isnan(recovered.d['sample']['mixed']))
    print 'number of valid values', divisor
    mean_absolute_error_true_signal = np.nansum(np.absolute(recovered.d['sample']['signal'] - (recovered.d['sample']['mixed'] - recovered.d['sample']['true_bias']))) / divisor
    mean_absolute_error_estimated_signal = np.nansum(np.absolute(recovered.d['sample']['signal'] - (recovered.d['sample']['mixed'] - recovered.d['sample']['estimated_bias']))) / divisor
    print 'mean_absolute_error_true_signal', mean_absolute_error_true_signal
    print 'mean_absolute_error_estimated_signal', mean_absolute_error_estimated_signal

    mean_absolute_error_zeros = np.nansum(np.absolute(recovered.d['sample']['signal'] - recovered.d['sample']['mixed'])) / divisor
    print 'mean_absolute_error_zeros', mean_absolute_error_zeros

    ratio_estimated_signal_to_zeros = mean_absolute_error_estimated_signal / mean_absolute_error_zeros
    print 'ratio_estimated_signal_to_zeros', ratio_estimated_signal_to_zeros



    # TODO Visualize and show difference between ingoing estimates and the results. Then improve them. Show maximum possible, with lots of restarts.

    # TODO Also do some absolute shifts of the data (no centering anymore)

    # TODO also do some different initializations...

    # TODO Why first correlation threshold no effect on performance?












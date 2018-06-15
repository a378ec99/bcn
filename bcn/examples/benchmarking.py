"""Benchmarking of different solvers for the case of imperfect correlations.
"""
from __future__ import division, absolute_import

import numpy as np
import seaborn as sb
import pandas as pd

from joblib import Parallel, delayed

from bcn.bias import guess_func
from bcn.data import DataSimulated, estimate_partial_signal_characterists
from bcn.cost import Cost
from bcn.solvers import ConjugateGradientSolver, SteepestDescentSolver
from bcn.linear_operators import LinearOperatorCustom, possible_measurements
from bcn.utils.visualization import recovery_performance



def benchmark(correlation_strength=0.97, n_restarts=3, solver='ConjugateGradient'):

    # Setup of general parameters for the recovery experiment.
    rank = 6
    n_measurements = 3000 # 2800
    shape = (50, 70) # samples, features
    missing_fraction = 0.01
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

    cost = Cost(measurements['A'], measurements['y'], sparsity=2)

    # Recover the bias.
    if solver == 'ConjugateGradient':
        Solver = ConjugateGradientSolver
    if solver == 'SteepestDescent':
        Solver = SteepestDescentSolver
        
    Solver = Solver(mixed, cost.cost_func, guess_func, rank, guess_noise_amplitude=noise_amplitude, verbosity=0, mingradnorm=1e-99, minstepsize=1e-99)
    results = Solver.recover()
    
    d = recovery_performance(mixed, cost.cost_func, truth.d['sample']['true_bias'], results['estimated_signal'], truth.d['sample']['signal'], results['estimated_bias'])
    d['solver'] = solver
    d['n_restarts'] = n_restarts
    d['correlation_strength'] = correlation_strength
    
    return d

# TODO Start up instance with lots of cores to run all at once!
# TODO Generate all parameters that want. 
# TODO Convergences at different maxiter=1000, maxtime=100, mingradnorm=1e-12, minstepsize=1e-12.


parameters = []

for correlation_strength in [0.5, 0.9, 0.97, 0.98, 0.99, 1.0]:
    for solver in ['ConjugateGradient', 'SteepestDescent']:
        for n_restarts in [1, 3, 10]:
            for maxiter, maxtime in zip([100, 1000, 5000, 1000000, 1000000], [1000000, 1000000, 1000000, 100, 1000]):
                parameters.append({'n_restarts': n_restarts, 'correlation_strength': correlation_strength, 'solver': solver, 'maxiter': maxiter, 'maxtime': maxtime})
            
        
print 'Performing Experiments'

experiments = Parallel(n_jobs=18)(delayed(benchmark)(**kwargs) for kwargs in parameters)

df = pd.DataFrame(experiments)

cm = sb.light_palette("green", as_cmap=True)

df = df.drop('Number of valid values in corrupted signal', axis=1)

html = df.style.background_gradient(cmap=cm).format({'Number of valid values in corrupted signal': "{:d}",
                                                     'n_restarts': "{:d}",
                                                     'correlation_strength': "{:.2E}",
                                                     'solver': "{}",
                                                     'Error cost function (true bias)': "{:.2E}",
                                                     'Error cost function (estimated bias)': "{:.2E}",
                                                     'Mean absolute error (true_signal)': "{:.2E}",
                                                     'Mean absolute error (estimated_signal)': "{:.2E}",
                                                     'Mean absolute error (zeros)': "{:.2E}",
                                                     'Ratio mean absolute error (estimated signal / zeros)': "{:.2E}",
                                                     'maxiter': "{:d}",
                                                     'maxtime': "{:d}"}).render()
    
with open("experiments_style.html","w") as f:
    f.write(html)
    





#if __name__ == '__main__':
#    pass
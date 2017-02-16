import unittest

import numpy as np
import numpy.testing as np_testing

from bcn.utils import submit

class TestSubmitCustom(unittest.TestCase):
    def setUp(self):
        seed = 42
        run = 'unittest-custom'
        parameters = {'class': 'Experiment',
                      'name': run,
                      'mode': None,
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 1,
                      'shape': (100, 110),
                      'correlation_strength': 1.0,
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 1,
                      'noise_amplitude': 5.0,
                      'p_random': 0.05,
                      'known_correlations': 1.0,
                      'operator_name': 'custom',
                      'additive_noise_y_std': None,
                      'additive_noise_A_std': None,
                      'm_blocks': 15,
                      'save_signal': False,
                      'sparsity': 2,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'free_x': ('rank', list(np.asarray(np.linspace(1, 5, 5), dtype=int))),
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(3000), np.log10(1.1e4), 5), dtype=int)))}
        
    def test_local(self):
        parameters['mode'] = 'local'
        submit(parameters)
        np_testing.assert_almost_equal()

    def test_parallel(self):
        parameters['mode'] = 'parallel'
        submit(parameters)
        np_testing.assert_almost_equal()
        
        
       

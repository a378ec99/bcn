import unittest
import time
import os

import numpy as np
import numpy.testing as np_testing

import sys
sys.path.append('/home/sohse/projects/PUBLICATION/GIT/bcn')
from utils import submit #from ..utils import submit

class TestSubmitDense(unittest.TestCase):
    def setUp(self):
        seed = 42
        run = 'unittest-dense'
        self.parameters = {'run_class': 'Simulation',
                      'name': run,
                      'mode': None,
                      'visualization_extension': '.png',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'double_blind': False,
                      'replicates': 1,
                      'shape': (10, 20),
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
                      'operator_name': 'dense',
                      'additive_noise_y_std': None,
                      'additive_noise_A_std': None,
                      'm_blocks': 5,
                      'save_run': False,
                      'sparsity': 2,
                      'verbosity': 0,
                      'logverbosity': 2,
                      'maxiter': 1000,
                      'maxtime': 100,
                      'mingradnorm': 1e-12,
                      'minstepsize': 1e-12,
                      'incorrect_A_std': 1.0,
                      'unittest': True,
                      'save_visualize': False,
                      'free_x': ('rank', list(np.asarray(np.linspace(1, 5, 2), dtype=int))),
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(10), np.log10(10*20), 2), dtype=int)))}
    
    def test_local(self):
        self.parameters['mode'] = 'local'
        submit(self.parameters)
        os.remove(self.parameters['name'] + '_complete.token')
        file_name = self.parameters['name'] + '_' + self.parameters['mode']
        X = np.load(file_name + '.npy')
        true_X = np.array([[  1.45509927e+00,   6.75724850e-13], [  1.34796093e+00,   3.71185767e-11]])
        np_testing.assert_almost_equal(X, true_X, decimal=4)
     
    def test_parallel(self):
        self.parameters['mode'] = 'parallel'
        submit(self.parameters)
        token = 'missing'
        while token == 'missing':
            try:
                open(self.parameters['name'] + '_complete.token', 'r')
                token = 'found'
                os.remove(self.parameters['name'] + '_complete.token')
            except IOError: time.sleep(5)
        file_name = self.parameters['name'] + '_' + self.parameters['mode']
        X = np.load(file_name + '.npy')
        true_X = np.array([[  1.45509927e+00,   6.02572450e-13], [  1.49368043e+00,   1.03385684e-11]])
        np_testing.assert_almost_equal(X, true_X, decimal=4)
    

if __name__ == '__main__':
    unittest.main()
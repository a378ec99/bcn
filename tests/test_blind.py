import unittest
import time
import os

import numpy as np
import numpy.testing as np_testing

import sys
sys.path.append('/home/sohse/projects/PUBLICATION/GIT/bcn')
from utils import submit #from ..utils import submit

class TestSubmitBlind(unittest.TestCase):
    def setUp(self):
        seed = 42
        run = 'unittest-blind'
        self.parameters = {'run_class': 'BlindCompressiveNormalization',
                        'name': run,
                        'mode': None,
                        'seed': seed,
                        'visualization_extension': '.png',
                        'figure_size': (8, 8),
                        'shape': (50, 50),
                        'signal_model': ('random', 'random'),
                        'correlation_strength': 1.0,
                        'normalize_stds': True,
                        'noise_amplitude': 1.0,
                        'noise_model': 'low-rank',
                        'm_blocks': int(50 / 2.0),
                        'restarts': 1,
                        'operator_name': 'custom',
                        'incorrect_A_std': None,
                        'measurements': np.inf,
                        'estimate': (True, True),
                        'sparsity': 2,
                        'verbosity': 1,
                        'logverbosity': 2,
                        'maxiter': 1000,
                        'maxtime': 100,
                        'mingradnorm': 1e-12,
                        'minstepsize': 1e-12, #  * 1e-8
                        'unittest': True,
                        'save_run': False,
                        'save_visualize': False,
                        'mixed': 'mixed.npy', #'threshold': 0.95, #'rank': 2,
                        'free_x': ('rank', list(np.asarray(np.linspace(1, 5, 2), dtype=int))),
                        'free_y': ('threshold', list(np.asarray(np.logspace(np.log10(0.5), np.log10(0.95), 2))))}
                  
    def test_local(self):
        self.parameters['mode'] = 'local'
        submit(self.parameters)
        os.remove(self.parameters['name'] + '_complete.token')
        file_name = self.parameters['name'] + '_' + self.parameters['mode']
        X = np.load(file_name + '.npy')
        true_X = np.array([[  4.75e-03,   2.65e-06], [  2.45e-03,   7.07e-11]])
        np_testing.assert_almost_equal(X, true_X, decimal=2)

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
        true_X = np.array([[  4.75e-03,   2.65e-06], [  2.45e-03,   7.50e-10]])
        np_testing.assert_almost_equal(X, true_X, decimal=2)


if __name__ == '__main__':
    unittest.main()
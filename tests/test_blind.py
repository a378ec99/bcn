import unittest
import time
import os

import numpy as np
import numpy.testing as np_testing

import sys
sys.path.append('/home/sohse/projects/PUBLICATION/GIT/bcn')
from utils import submit #from ..utils import submit
from bcn import Signal, Noise, Missing

class TestSubmitBlind(unittest.TestCase):
    def setUp(self):
        seed = 42
        run = 'unittest-blind'
        np.random.seed(seed)
        self.parameters = {'run_class': 'Simulation',
                        'name': run,
                        'mode': None,
                        'seed': seed,
                        'double_blind': True,
                        'replicates': 1,
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
                        'missing_model': 'MAR',
                        'p_random': 0.1,
                        'mixed': run + '_mixed.npy',
                        'free_x': ('rank', list(np.asarray(np.linspace(1, 5, 2), dtype=int))),
                        'free_y': ('threshold', list(np.asarray(np.logspace(np.log10(0.5), np.log10(0.95), 2))))}

        signal = Signal(self.parameters['shape'], self.parameters['signal_model'], self.parameters['m_blocks'], self.parameters['correlation_strength'], self.parameters['normalize_stds']).generate()
        noise = Noise(self.parameters['shape'], self.parameters['noise_model']).generate(self.parameters['noise_amplitude'], 2)
        missing = Missing(self.parameters['shape'], self.parameters['missing_model']).generate(p_random=self.parameters['p_random'])
        mixed = signal['X'] + noise['X'] + missing['X']
        np.save(self.parameters['name'] + '_mixed', mixed)
        
    def test_local(self):
        self.parameters['mode'] = 'local'
        submit(self.parameters)
        os.remove(self.parameters['name'] + '_complete.token')
        file_name = self.parameters['name'] + '_' + self.parameters['mode']
        X = np.load(file_name + '.npy')
        true_X = np.array([[  4.8186e-03,   1.4304e-07], [  2.1629e-03,   7.9500e-10]])
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
        true_X = np.array([[  4.8185547e-03,   1.5772426e-07], [  2.1629426e-03,   2.2993649e-10]])
        np_testing.assert_almost_equal(X, true_X, decimal=4)
    

if __name__ == '__main__':
    unittest.main()
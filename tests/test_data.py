"""Simulated data tests.
"""
from __future__ import division, absolute_import

import unittest

import numpy as np

from bcn.data import DataSimulated, estimate_partial_signal_characterists

        
class TestDataSimulated(unittest.TestCase):
    """Test to verify that the inital data is created correctly.
    """
    def setUp(self):
        self.shape = (50, 60)
        self.rank = 2
        
    def test(self):
        data = DataSimulated(shape=self.shape, rank=self.rank)
        mixed = data.d['sample']['mixed']
        correlation_threshold = 0.7
        estimates = estimate_partial_signal_characterists(mixed, correlation_threshold, true_pairs={'sample': data.d['sample']['true_pairs'], 'feature': data.d['feature']['true_pairs']})
        for space in ['feature', 'sample']:
            print estimates[space].keys()
            assert estimates[space]['estimated_correlations'].shape == (data.d[space]['shape'][0], data.d[space]['shape'][0])
            assert estimates[space]['estimated_pairs'].shape[1] == 2
            assert estimates[space]['estimated_directions'].shape[0] == data.d[space]['true_pairs'].shape[0]
            assert estimates[space]['estimated_stds'].shape[0] == data.d[space]['true_pairs'].shape[0]
            assert estimates[space]['estimated_directions'].ndim == 1

            
if __name__ == '__main__':
    unittest.main()



    













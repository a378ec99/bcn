import unittest

import numpy as np
import numpy.testing as np_testing

from bcn.utils import submit

class TestLocal(unittest.TestCase):
    def setUp(self):
        seed = 42
        np.random.seed(seed)
        
    def test_submit(self):
        parameters = 
        submit(parameters['mode'], parameters['class'], parameters, # TODO Just submit parameters, not specific items too.
           path='PUBLICATION/bcn/bcn') # TODO Change path to full.
        np_testing.assert_almost_equal()
           
 class TestParallel(unittest.TestCase):
    def setUp(self):
        seed = 42
        np.random.seed(seed)
        
    def test_submit(self):
        parameters = 
        submit(parameters['mode'], parameters['class'], parameters, # TODO Just submit parameters, not specific items too.
           path='PUBLICATION/bcn/bcn') # TODO Change path to full.
        np_testing.assert_almost_equal()

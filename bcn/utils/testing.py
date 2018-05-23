"""Unittest utility function.
"""
from __future__ import division, absolute_import

import hashlib


def assert_consistency(X, true_md5):
    """
    Asserts the consistency between two function outputs based on a hash.

    Parameters
    ----------
    X : numpy.ndarray
        Array to be hashed.
    true_md5 : str
        Expected hash.
    """
    m = hashlib.md5()
    m.update(X)
    current_md5 = m.hexdigest()
    print
    print current_md5, true_md5
    print
    assert current_md5 == true_md5

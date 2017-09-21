"""Bias generation module.

Notes
-----
This module defines two classes that can generate different types of bias.
"""
from __future__ import division, absolute_import


__all__ = ['BiasLowRank', 'BiasUnconstrained']

import numpy as np

from scipy.misc import imread, imresize
from sklearn.datasets import make_checkerboard
from pymanopt.manifolds import FixedRankEmbedded, Euclidean


class BiasLowRank(object):

    def __init__(self, shape, model, rank, noise_amplitude=None, image_source=None, n_clusters=None):
        """Generate bias according to a low-rank (sparse) model.

        Parameters
        ----------
        shape : tuple of int
            Shape of the output bias matrix in the form of (n_samples, n_features).
        model : {'image', 'bicluster', 'gaussian'}
            Three bias models are supported, `gaussian` which is based on a QR decomposition of a random Gaussian matrix, `image` which is based on a prespecified image that is then rank reduced, and `bicluster` which is based on `sklearn's` checkerboard function that is then rank reduced.
        rank : int
            Rank of the low-rank decomposition.
        noise_amplitude : float, optional unless model `gaussian`
            Sets the level of the bias.
        image_source: str, optional unless model `image`
            File location of the image to be used for the model `image`.
        n_clusters: tuple of int, optional unless model `bicluster`
            Number of clusters for the model `bicluster` in the form of (n_sample_clusters, n_column_clusters).

        Notes
        -----
        # TODO Check that n_samples, n_features is correct and do so for all other similar instances in this module.
        """
        self.shape = shape
        self.model = model
        self.rank = rank
        self.noise_amplitude = noise_amplitude
        self.image_source = image_source
        self.n_clusters = n_clusters
        
        assert self.model in ['image', 'bicluster', 'gaussian']
        
    def generate(self):
        """Generate bias according to a low-rank (sparse) model.

        Returns
        -------
        bias : dict, {'X': ndarray, shape (n_sample, n_features), 'usvt': tuple of ndarray, (U, S, Vt), shape ((n_samples, rank), rank, (rank, n_samples))}
            Contains low-rank bias matrix `X` and it's corresponding decomposition `usvt`.
        """
        if self.model == 'gaussian':
            usvt = FixedRankEmbedded(self.shape[0], self.shape[1], self.rank).rand()
            # NOTE Eigenvalues are normalized so that the bias level is approximately consistent over differing rank matrices.
            usvt = usvt[0], (usvt[1] / np.sum(np.absolute(usvt[1]))), usvt[2]
            usvt = usvt[0], usvt[1] * self.noise_amplitude, usvt[2]
            X = np.dot(np.dot(usvt[0], np.diag(usvt[1])), usvt[2])

        if self.model == 'image':
            X = imread(self.image_source, flatten=True, mode='L')
            if X.shape != self.shape:
                X = imresize(X, self.shape)
            # TODO Need this normalization below? Can do better?
            X = (X / X.max()) - 0.5
            usvt = np.linalg.svd(X)
            usvt = usvt[0][:, :self.rank], usvt[1][:self.rank], usvt[2][:self.rank, :]
            X = np.dot(np.dot(usvt[0], np.diag(usvt[1])), usvt[2])
            
        if self.model == 'bicluster':
            X, rows, columns = make_checkerboard(shape=self.shape, n_clusters=self.n_clusters, noise=0, shuffle=False)
            # TODO Need this normalization below? Can do better?
            X = (X / X.max()) - 0.5
            usvt = np.linalg.svd(X)
            usvt = usvt[0][:, :self.rank], usvt[1][:self.rank], usvt[2][:self.rank, :]
            X = np.dot(np.dot(usvt[0], np.diag(usvt[1])), usvt[2])
        
        return {'X': X, 'usvt': usvt}
        

class BiasUnconstrained(object):

    def __init__(self, shape, model, noise_amplitude=None, fill_value=None):
        """Generate bias according to an unconstrained (non-sparse) model.

        Parameters
        ----------
        shape : tuple of int
            Shape of the output bias matrix in the form of (n_samples, n_features).
        model : {'gaussian', 'uniform'}
            Two bias models are supported, `gaussian` which is based on random sampling of a Gaussian matrix and `uniform` which is based on repetition of a prespecified fill value.
        amplitude : float, optional unless model `gaussian`
            Sets the level of the bias.
        fill_value : float, optional unless model `uniform`
            Sets the fill value for the uniform bias model.
        """
        self.shape = shape
        self.model = model
        self.noise_amplitude = noise_amplitude
        self.fill_value = fill_value
        
        assert self.model in ['gaussian', 'uniform']
        
    def generate(self):
        """Generate bias according to an unconstrained (non-sparse) model.

        Returns
        -------
        bias : dict, {'X': ndarray, shape (n_sample, n_features)}
            Contains low-rank bias matrix `X` and it's corresponding decomposition `usvt`.
        """
        if self.model == 'gaussian':
            X = Euclidean(self.shape[0], self.shape[1]).rand()
            X = X * self.noise_amplitude
            
        if self.model == 'uniform':
            X = np.full(self.shape, self.fill_value)

        return {'X': X}


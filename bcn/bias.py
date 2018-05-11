"""Bias generation.

Notes
-----
Defines two classes that can generate different types of bias and a bias guess function.
"""
from __future__ import division, absolute_import

import numpy as np

from skimage.transform import resize
from skimage.io import imread
from sklearn.datasets import make_checkerboard
from pymanopt.manifolds import FixedRankEmbedded, Euclidean


def guess_func(shape, rank, **kwargs):
    """Generate an initial bias guess for the solver to start at.

    Parameters
    ----------
    shape : (int, int)
        Dimensions of the array to be recoved.
    rank : int
        Rank of the bias to be recovered (estimate or truth).
    kwargs : dict
        Additional arguments to be passed to the BiasLowRank class.

    Returns
    -------
    guess : dict
        Initial guess for the solver to be used, containing X and the decomposed usvt.

    Notes
    -----
    The guess function needs to use the class that is matched to the according underlying bias.
    """
    bias = BiasLowRank(shape, rank, **kwargs).generate()
    guess = {'X': bias['X'], 'usvt': bias['usvt']}
    return guess


class BiasLowRank(object):

    def __init__(self, shape, rank, bias_model='gaussian', noise_amplitude=1.0, n_clusters=2, image_source=None):
        """Generate bias according to a low-rank (sparse) model.

        Parameters
        ----------
        shape : tuple of int
            Shape of the output bias matrix in the form of (n_samples, n_features).
        rank : int
            Rank of the low-rank decomposition.
        bias_model : {'image', 'bicluster', 'gaussian'}
            Three bias models are supported, `gaussian` which is based on a QR decomposition of a random Gaussian matrix, `image` which is based on a prespecified image that is then rank reduced, and `bicluster` which is based on `sklearn's` checkerboard function that is then rank reduced.
        noise_amplitude : float, optional unless model `gaussian`
            Sets the level of the bias.
        n_clusters: tuple of int, optional unless model `bicluster`
            Number of clusters for the model `bicluster` in the form of (n_sample_clusters, n_column_clusters).
        image_source: str, optional unless model `image`
            File location of the image to be used for the model `image`.
        """
        self.shape = shape
        self.bias_model = bias_model
        self.rank = rank
        self.noise_amplitude = noise_amplitude
        self.image_source = image_source
        self.n_clusters = n_clusters

        assert self.bias_model in ['image', 'bicluster', 'gaussian']

    def generate(self):
        """Generate bias according to a low-rank (sparse) model.

        Returns
        -------
        bias : dict, {'X': ndarray, shape (n_sample, n_features), 'usvt': tuple of ndarray, (U, S, Vt), shape ((n_samples, rank), rank, (rank, n_samples))}
            Contains low-rank bias matrix `X` and it's corresponding decomposition `usvt`.
        """
        if self.bias_model == 'gaussian':
            usvt = FixedRankEmbedded(self.shape[0], self.shape[1], self.rank).rand()
            # NOTE Eigenvalues are normalized so that the bias level is
            # approximately consistent over differing rank matrices.
            usvt = usvt[0], (usvt[1] / np.sum(np.absolute(usvt[1]))), usvt[2]
            usvt = usvt[0], usvt[1] * self.noise_amplitude, usvt[2]
            X = np.dot(np.dot(usvt[0], np.diag(usvt[1])), usvt[2])

        if self.bias_model == 'image':
            X = imread(self.image_source, flatten=True, mode='L')
            if X.shape != self.shape:
                X = resize(X, self.shape)
            X = 0.5 * ((X / np.absolute(X).max()) - 0.5)
            usvt = np.linalg.svd(X)
            usvt = usvt[0][:, :self.rank], usvt[1][
                :self.rank], usvt[2][:self.rank, :]
            X = np.dot(np.dot(usvt[0], np.diag(usvt[1])), usvt[2])

        if self.bias_model == 'bicluster':
            X, rows, columns = make_checkerboard(
                shape=self.shape, n_clusters=self.n_clusters, noise=0, shuffle=False)
            X = (X / X.max()) - 0.5
            usvt = np.linalg.svd(X)
            usvt = usvt[0][:, :self.rank], usvt[1][
                :self.rank], usvt[2][:self.rank, :]
            X = np.dot(np.dot(usvt[0], np.diag(usvt[1])), usvt[2])

        bias = {'X': X, 'usvt': usvt}
        return bias


class BiasUnconstrained(object):

    def __init__(self, shape, bias_model='gaussian', noise_amplitude=1.0, fill_value=42):
        """Generate bias according to an unconstrained (non-sparse) model.

        Parameters
        ----------
        shape : tuple of int
            Shape of the output bias matrix in the form of (n_samples, n_features).
        bias_model : {'gaussian', 'uniform'}
            Two bias models are supported, `gaussian` which is based on random sampling of a Gaussian matrix and `uniform` which is based on repetition of a prespecified fill value.
        noise_amplitude : float, optional unless model `gaussian`
            Sets the level of the bias.
        fill_value : float, optional unless model `uniform`
            Sets the fill value for the uniform bias model.
        """
        self.shape = shape
        self.bias_model = bias_model
        self.noise_amplitude = noise_amplitude
        self.fill_value = fill_value

        assert self.bias_model in ['gaussian', 'uniform']

    def generate(self):
        """Generate bias according to an unconstrained (non-sparse) model.

        Returns
        -------
        bias : dict, {'X': ndarray, shape (n_sample, n_features)}
            Contains low-rank bias matrix `X` and it's corresponding decomposition `usvt`.
        """
        if self.bias_model == 'gaussian':
            X = Euclidean(self.shape[0], self.shape[1]).rand()
            X = X * self.noise_amplitude

        if self.bias_model == 'uniform':
            X = np.full(self.shape, self.fill_value)

        bias = {'X': X}
        return bias

from __future__ import division

import sys

from abc import ABCMeta, abstractmethod
from utils import submit, Visualize, square_blocks_matrix

import autograd.numpy as ag
import numpy as np

import scipy.misc as spmi
from sklearn.datasets import make_checkerboard

from pymanopt.manifolds import FixedRankEmbedded, Euclidean
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, ConjugateGradient, TrustRegions, NelderMead, ParticleSwarm

import traceback
import bottleneck as bn


class TaskPull(object):
    '''
    Abstract class that denotes API to taskpull.py and taskpull_local.py.
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def allocate(self):
        pass

    @abstractmethod
    def create_tasks(self):
        pass

    @abstractmethod
    def work(self):
        pass

    @abstractmethod
    def store(self):
        pass

    @abstractmethod
    def postprocessing(self):
        pass


class Experiment(TaskPull):
    '''
    Evaluation of two free parameters over their specified range.
    '''

    def __init__(self, parameters):
        self.parameters = parameters
        self.free_x = self.parameters['free_x']
        self.free_y = self.parameters['free_y']
        self.replicates = self.parameters['replicates']
        self.file_name = self.parameters['name']
        self.visualization_extension = self.parameters[
            'visualization_extension']
        self.figure_size = self.parameters['figure_size']

    def allocate(self):
        self.X = np.empty((self.replicates, len(
            self.free_x[1]), len(self.free_y[1])))

    def create_tasks(self):
        for h in xrange(self.replicates):
            for i, x in enumerate(self.free_x[1]):
                for j, y in enumerate(self.free_y[1]):
                    task = h, i, j, x, y, self.free_x[0], self.free_y[0]
                    print task
                    yield task

    def work(self, task):
        h, i, j, x, y, x_name, y_name = task
        self.parameters[x_name] = x
        self.parameters[y_name] = y
        r = Recovery(self.parameters)
        error = r.run()
        return h, i, j, error

    def store(self, result):
        h, i, j, error = result
        self.X[h, i, j] = error

    def postprocessing(self):
        self.X = bn.nanmean(self.X, axis=0)
        np.save(self.file_name, self.X)
        v = Visualize(self.X.T, file_name=self.file_name +
                      self.visualization_extension, size=self.figure_size)
        v.recovery_performance(xlabel=self.free_x[0], ylabel=self.free_y[
                               0], xticklabels=self.free_x[1], yticklabels=self.free_y[1]) # vmin=0, vmax=2.0, 


class Missing(object):
    '''
    Generate missing values according to different models:

        - MAR (missing at random)
        - MNAR (missing not at random)
        - No missing values
    '''

    def __init__(self, shape, model):
        self.shape = shape
        self.model = model
        assert self.model in ['MAR', 'NMAR', 'no-missing', 'SCAN']

    def generate(self, X=None, p_random=0.2, p_censored=0.1):
        if self.model == 'MAR':
            missing = np.zeros(self.shape)
            q = list(np.ndindex(self.shape))
            indices = np.random.choice(
                np.arange(len(q)), replace=False, size=int(missing.size * p_random))
            for index in indices:
                missing[q[index]] = np.nan
            missing = {'X': missing}
        if self.model == 'NMAR':
            censored = np.zeros(self.shape)
            q = list(np.ndindex(self.shape))
            indices = np.random.choice(
                np.arange(len(q)), replace=False, size=int(censored.size * p_random))
            for index in indices:
                censored[q[index]] = np.nan
            indices = np.random.choice(
                np.arange(self.shape[0]), replace=False, size=int(self.shape[0] * p_censored))
            for index in indices:
                start_ = np.random.randint(0, self.shape[1])
                len_ = np.random.randint(0, self.shape[1] - start_) // 2
                censored[index, start_:start_ + len_] = np.nan
            indices = np.random.choice(
                np.arange(self.shape[1]), replace=False, size=int(self.shape[1] * p_censored))
            for index in indices:
                start_ = np.random.randint(0, self.shape[0])
                len_ = np.random.randint(0, self.shape[0] - start_) // 2
                censored[start_:start_ + len_, index] = np.nan
            missing = {'X': censored}
        if self.model == 'SCAN':
            raise NotImplementedError
        if self.model == 'no-missing':
            missing = {'X': np.zeros(self.shape)}
        return missing


class Noise(object):
    '''
    Generate noise according to different models:

        - Low-rank
        - Gaussian
        - Prespecified image
        - Biclusters
    '''

    def __init__(self, shape, model):
        self.shape = shape
        self.model = model
        assert self.model in ['low-rank', 'euclidean', 'image', 'biclusters']

    def generate(self, noise_amplitude=1.0, rank=None, image_file_name='trump.png'):
        if self.model == 'low-rank':
            # NOTE Eigenvalues are normalized to 1.0 so that noise is consistent
            # with an increasing rank.
            usvt = FixedRankEmbedded(self.shape[0], self.shape[1], rank).rand()
            usvt = usvt[0], (usvt[1] / np.sum(np.absolute(usvt[1]))), usvt[2]
            usvt = usvt[0], usvt[1] * noise_amplitude, usvt[2]
            X = np.dot(np.dot(usvt[0], np.diag(usvt[1])), usvt[2])
            noise = {'X': X, 'usvt': usvt}
            return noise
        if self.model == 'euclidean':
            X = Euclidean(self.shape[0], self.shape[1]).rand()
            noise = {'X': X}
            return noise
        if self.model == 'image':
            X = spmi.imread(image_file_name, flatten=True, mode='L')
            X = (X / X.max()) - 0.5
            usvt = np.linalg.svd(X)
            usvt = usvt[0][:, :rank], usvt[1][:rank], usvt[2][:rank, :]
            X = np.dot(np.dot(usvt[0], np.diag(usvt[1])), usvt[2])
            noise = {'X': X, 'usvt': usvt}
            return noise
        if self.model == 'biclusters':
            data, rows, columns = make_checkerboard(shape=self.shape, n_clusters=(
                4, 3), noise=0, shuffle=False, random_state=42)
            data = (data / data.max()) - 0.5
            usvt = np.linalg.svd(data)
            usvt = usvt[0][:, :rank], usvt[1][:rank], usvt[2][:rank, :]
            X = np.dot(np.dot(usvt[0], np.diag(usvt[1])), usvt[2])
            X = X / 50.0  # WARNING Dirty hack.
            noise = {'X': X, 'usvt': usvt}
            return noise


class Signal(object):
    '''
    Generate a signal (observation matrix) of a complex system with certain redundancy.
    '''

    def __init__(self, shape, model, m_blocks, correlation_strength, normalize_stds):
        self.shape = {'feature': shape[0], 'sample': shape[1]}
        self.model = model
        self.m_blocks = m_blocks
        self.normalize_stds = normalize_stds
        assert self.model[0] in ['random', 'unity', 'constant']
        assert self.model[1] in ['random', '-', '+']
        self.correlation_strength = correlation_strength

    def _generate_pairs(self, correlations):
        pairs = np.vstack(np.nonzero(np.tril(correlations, -1))).T
        # NOTE Shuffling the order of the pairs.
        indices = np.arange(len(pairs))
        np.random.shuffle(indices)
        pairs = np.asarray(pairs)
        pairs = pairs[indices]
        return pairs

    def _generate_stds(self, n, pairs, space):
        if self.model[0] == 'random':
            stds = np.random.uniform(0.1, 2, n)
        if self.model[0] == 'unity':
            stds = np.ones(n)
        if self.model[0] == 'constant':  # WARNING Something fishy!
            stds = np.random.uniform(0.1, 2, n)
            for pair in pairs:
                i, j = pair
                std = np.random.uniform(0.1, 2)
                if space == 'feature':
                    factor = np.random.choice([2.0, 0.5])
                if space == 'sample':
                    factor = np.random.choice([2.0, 0.5])
                stds[i] = std * factor
                stds[j] = std

        if self.normalize_stds:
            scaling_factor_stds = np.sqrt(1 / float(np.sum(stds**2)))
            stds = stds * scaling_factor_stds
        return stds, scaling_factor_stds

    def _generate_directions(self, correlations, pairs):
        directions = np.sign(correlations[pairs[:, 0], pairs[:, 1]])
        return directions

    def _generate_correlations(self, n, correlation_strength, m_blocks):
        correlations = square_blocks_matrix(
            n, m_blocks, r=correlation_strength)
        return correlations

    def _generate_covariance(self, correlations, stds):
        covariance = np.outer(stds, stds) * correlations
        return covariance

    def _sample_matrix_normal(self, U, V):
        n = len(U)
        m = len(V)
        # NOTE Could also use np.random.multivariate_normal(np.zeros(n),
        # np.eye(n), m).T, but no advantage and likely slower.
        X = np.random.standard_normal((n, m))
        u, s, vt = np.linalg.svd(U)
        # s[s < 1.0e-8] = 0 # WARNING Needed?
        Us = np.dot(u, np.diag(np.sqrt(s)))
        u, s, vt = np.linalg.svd(V)
        # s[s < 1.0e-8] = 0 # WARNING Needed?
        Vs = np.dot(u, np.diag(np.sqrt(s)))
        Y = np.dot(np.dot(Us, X), Vs.T)
        return Y

    def generate(self):
        result = None
        while result is None:
            try:
                signal = {}
                for space in ['feature', 'sample']:
                    correlations = self._generate_correlations(
                        self.shape[space], self.correlation_strength, self.m_blocks)
                    pairs = self._generate_pairs(correlations)
                    directions = self._generate_directions(correlations, pairs)
                    stds, scaling_factor_stds = self._generate_stds(self.shape[space], pairs, space)
                    print 'scaling_factor_stds', scaling_factor_stds
                    covariance = self._generate_covariance(correlations, stds)
                    signal[space] = {'pairs': pairs,
                                     'correlations': correlations,
                                     'stds': stds,
                                     'directions': directions,
                                     'covariance': covariance,
                                     'scaling_factor_stds': scaling_factor_stds}
                U, V = signal['feature']['covariance'], signal[
                    'sample']['covariance']
                X = self._sample_matrix_normal(U, V)
                signal['X'] = X
                result = True
            except Exception:
                # NOTE
                # https://github.com/ContinuumIO/anaconda-issues/issues/695
                print 'Likely np.linalg.LinAlgError:', traceback.format_exc()
        return signal


class Cost(object):
    '''
    Generate an objective function based on A and y constructed by LinearOperator.
    '''

    def __init__(self, operator_name, A, Y, pairs=None):
        self.operator_name = operator_name
        self.A = A
        self.Y = Y
        self.pairs = pairs
        assert self.operator_name in ['entry', 'dense', 'custom']

    def evaluate(self, X):
        if len(X) == 3:
            usvt = X
            X = ag.dot(usvt[0], ag.dot(ag.diag(usvt[1]), usvt[2]))
        if self.operator_name == 'entry':
            error = ag.sum((self.A * X - self.Y)**2)
        # TODO Just use the three arrays that csr_matrix uses and use indices
        # and elements to then quickly evaluate those for sparse matrices! Look
        # inside to see how they are doing it. E.g. Also sparse matrix
        # multiolication...
        if self.operator_name == 'dense':
            #error = ag.sum((ag.sum(self.A * X[:, :, None], axis=(0, 1)) - self.Y)**2)
            error = 0.0
            for i in xrange(self.A.shape[0]):
                error = error + (ag.sum(X[ag.asarray(self.A[i, 1, :], dtype=int), ag.asarray(self.A[i, 2, :], dtype=int)] * self.A[i, 0, :]) - self.Y[i])**2  # TODO Make without for loop, only numpy operations!
        if self.operator_name == 'custom':
            X = {'feature': X, 'sample': X.T}
            error = 0.0
            for space in ['feature', 'sample']:
                Y_estimate = self.A[space][::2] * X[space][self.pairs[space][:, 0]
                                                           ] + self.A[space][1::2] * X[space][self.pairs[space][:, 1]]  # NOTE Could try to extent to clusters of arbitrary size (e.g. more than 2 pairs) but then still same size... could compute Operator here? Nah.
                error = error + ag.sum((Y_estimate - self.Y[space])**2)
        return error


class LinearOperator(object):
    '''
    Generate a linear measurement operator A and the corresponding y.
    '''

    def __init__(self, operator_name):
        self.operator_name = operator_name
        assert self.operator_name in ['entry', 'dense', 'custom']

    def _solve(self, a, b, d):
        # NOTE Solve ay + bx + c for c.
        c = -(a * d[0] + b * d[1])
        return c

    def _distance(self, a, b, x0, y0):
        # NOTE Distance from point (x0, y0) to line y = mx + 0.0;
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        m = -b / float(a)
        x = (x0 + m * y0) / float(m**2 + 1)
        y = m * x
        d = np.asarray([y - y0, x - x0])
        return d

    def _estimate_correlations_direction_pairs(self, mixed_masked, threshold):
        # WARNING Not yet live for estimation from real data.
        correlations = np.ma.corrcoef(mixed_masked)
        correlations[np.absolute(correlations) < threshold] = 0
        pairs = np.vstack(np.nonzero(np.tril(correlations, -1))).T
        indices = np.arange(len(pairs))
        np.random.shuffle(indices)
        pairs = np.asarray(pairs)
        pairs = pairs[indices]
        directions = np.sign(correlations[pairs[:, 0], pairs[:, 1]])
        return correlations, directions, pairs

    def _estimate_stds(self, mixed_masked):
        stds = np.ma.filled(np.ma.std(mixed_masked, axis=1), np.nan)
        return stds
        
    def _construct_measurements(self, pairs, stds, directions, mixed_masked, inexact, incorrect, scaling_factor_stds, incorrect_A_std):
        A = np.zeros((2 * len(pairs), mixed_masked.shape[1]))  # * np.nan
        Y = np.zeros((len(pairs), mixed_masked.shape[1]))
        for i, pair in enumerate(pairs):
            direction = directions[i]
            b, a = stds[pair]
            if i in inexact:
                s = 0.1 # NOTE ~10%
                b, a = b - (s * np.random.choice([-1, 1]) * b), a - (s * np.random.choice([-1, 1]) * a)
            if i in incorrect:
                direction = np.random.choice([-1, 1])
                b, a = np.random.uniform(0.1, 2, 2) * scaling_factor_stds # np.absolute(np.random.normal(0.0, incorrect_A_std, 2)) # NOTE All pairs stay the same, just the stds change.
            b = -direction * b
            for j in xrange(mixed_masked.shape[1]):
                if (mixed_masked.mask[pair[0], j] == False) and (mixed_masked.mask[pair[1], j] == False):
                    y0, x0 = mixed_masked[pair, j]
                    if np.isfinite([y0, x0]).all():
                        d = self._distance(a, b, x0, y0)
                        c = self._solve(a, b, d)
                        A[2 * i, j] = a
                        A[2 * i + 1, j] = b
                        Y[i, j] = c
                    else:
                        raise Exception('Point not finite.')
        return A, Y

    def generate(self, incorrect_A_std=None, mixed=None, X=None, true_noise=None, true_signal=None, estimate_pairs=False, estimate_stds=False, estimate_noise=False, measurements=None, constrain_samples=False, known_correlations=None, sparsity=None, additive_noise_y_std=None, additive_noise_A_std=None):  # TODO remove estimate noise.
        if self.operator_name == 'entry':
            assert sparsity == 1
            max_measurements = true_noise['X'].shape[
                0] * true_noise['X'].shape[1]
            if measurements > max_measurements:
                measurements = max_measurements
                print 'WARNING: specified too many measurements.'
            A = np.zeros_like(true_noise['X'])
            A = np.array(A, dtype=int)
            Y = np.zeros_like(true_noise['X'])
            random_pairs = np.random.choice(np.arange(0, true_noise['X'].shape[
                                            0]), measurements), np.random.choice(np.arange(0, true_noise['X'].shape[1]), measurements)
            random_pairs = np.vstack(random_pairs).T
            for r_p in random_pairs:
                A[r_p[0], r_p[1]] = 1
                Y[r_p[0], r_p[1]] = true_noise['X'][r_p[0], r_p[1]]
            print "A", A.nbytes * 1.0e-6, 'MB'
            print "Y", Y.nbytes * 1.0e-6, 'MB'
            return A, Y, None
        if self.operator_name == 'dense':
            inexact = []
            #inexact = np.random.choice(np.arange(0, n_pairs, dtype=int),size=n_pairs_estimated_incorrectly, replace=False) # TODO Which pairs sampled from all possible pairs and given the number wanted should be modified?
            incorrect = np.random.choice(np.arange(0, measurements, dtype=int), size=int((1 - known_correlations) * measurements), replace=False)
            # TODO Try low-rank approach for most dense matrices.
            assert sparsity <= true_noise['X'].shape[
                0] * true_noise['X'].shape[1]
            A = np.empty((measurements, 3, sparsity))
            Y = np.empty(measurements)
            for i in xrange(measurements):
                A[i, 0, :] = np.random.uniform(0.1, 2, sparsity) * true_signal['feature']['scaling_factor_stds']
                A[i, 1, :] = np.random.choice(
                    np.arange(0, true_noise['X'].shape[0]), sparsity, replace=False)
                A[i, 2, :] = np.random.choice(
                    np.arange(0, true_noise['X'].shape[1]), sparsity, replace=False)
                Y[i] = np.sum(A[i, 0, :] * true_noise['X'][np.asarray(A[i, 1, :], dtype=int), np.asarray(A[i, 2, :], dtype=int)])
                if i in incorrect:
                    A[i, 0, :] = np.random.uniform(0.1, 2, sparsity) * true_signal['feature']['scaling_factor_stds'] # np.absolute(np.random.normal(0.0, incorrect_A_std, sparsity)) 
            if additive_noise_y_std is not None:
                print '------------------------->', additive_noise_y_std
                Y = Y + np.random.normal(0.0, additive_noise_y_std, size=Y.shape)
            #else:
            #    sys.exit()
            #    # WARNING DANGER
            if additive_noise_A_std:
                A[:, 0, :] = A[:, 0, :] + np.random.normal(0.0, additive_noise_A_std, size=A[:, 0, :].shape)
            print "A", A.nbytes * 1.0e-6, 'MB'
            print "Y", Y.nbytes * 1.0e-6, 'MB'
            return A, Y, None
        if self.operator_name == 'custom':
            #assert measurements % 2 == 0 # WARNING Might be needed; otherwise might be less measurements then thought!
            assert sparsity == 2
            A, Y, pairs_all = {}, {}, {}
            for space in ['feature', 'sample']:
                min_required_pairs = ((measurements // 2) // mixed[space].shape[1]) + 1  # WARNING Dirty hack.
                n_pairs = min_required_pairs
                n_pairs_estimated_incorrectly = n_pairs - int(known_correlations * n_pairs)
                inexact = []
                #inexact = np.random.choice(np.arange(0, n_pairs, dtype=int),size=n_pairs_estimated_incorrectly, replace=False) # TODO Which pairs sampled from all possible pairs and given the number wanted should be modified?
                incorrect = np.random.choice(np.arange(0, n_pairs, dtype=int),size=n_pairs_estimated_incorrectly, replace=False)
                size = int((1 - known_correlations) * measurements)
                mixed_masked = np.ma.masked_invalid(mixed[space])
                pairs = true_signal[space]['pairs'][:n_pairs]
                directions = true_signal[space]['directions'][:n_pairs]
                stds = true_signal[space]['stds']
                temp = self._construct_measurements(pairs, stds, directions, mixed_masked, inexact, incorrect, true_signal[space]['scaling_factor_stds'], incorrect_A_std)
                A[space] = temp[0]
                Y[space] = temp[1]
                pairs_all[space] = pairs
            print "A['feature']", A['feature'].nbytes * 1.0e-6, 'MB'
            print "Y['feature']", Y['feature'].nbytes * 1.0e-6, 'MB'
            return A, Y, pairs_all
            # TODO simple way just zero some so that measurements are exact?
            # TODO Fix memory error.


class Recovery(object):
    '''
    Low-rank matrix recovery via compressed sensing of a bias matrix.
    '''

    def __init__(self, parameters):
        self.parameters = parameters

    def run(self):
        signal, noise, missing, mixed, problem = self.setup()
        estimates, errors, guesses, true_errors = [], [], [], []
        for k in xrange(self.parameters['restarts']):
            X_estimate, X0, final_cost = self.solve(problem)
            divisor = np.sum(~np.isnan(mixed['feature']))
            # NOTE Size corresponds to non-nan elements.
            error = np.nansum(np.absolute(
                signal['X'] - (mixed['feature'] - X_estimate))) / divisor
            zero_error = np.nansum(np.absolute(
                signal['X'] - (mixed['feature']))) / divisor
            X0_error = np.nansum(np.absolute(
                signal['X'] - (mixed['feature'] - X0))) / divisor
            estimates.append(X_estimate)
            guesses.append(X0)
            #errors.append(error / zero_error)
            # NOTE Can't estimate better than that if don't know answer.
            errors.append(final_cost)
            true_errors.append(error / zero_error) 
            print k, '0', zero_error
            print k, 'X0', 'absolute:', X0_error, 'relative:', X0_error / zero_error
            print k, 'RMSE', 'absolute:', error, 'relative:', error / zero_error
            print k, 'final cost', final_cost
            if self.parameters['operator_name'] == 'custom':
                print k, 'SNR', self.signal_to_noise(signal, noise)
        index = np.argmin(errors)
        error = true_errors[index]
        estimate = estimates[index]
        if self.parameters['save_signal']:
            self.save(signal, noise, missing, mixed['feature'], guesses[index])
        return error

    # TODO visualize with figure4.py right away?
    def save(self, signal, noise, missing, mixed, guess):
        np.save(self.parameters['name'] + '_guess', guess)
        np.save(self.parameters['name'] + '_signal', signal['X'])
        np.save(self.parameters['name'] + '_mixed', mixed)
        np.save(self.parameters['name'] + '_noise', noise['X'])
        for space in ['sample', 'feature']:
            np.save(self.parameters['name'] +
                    '_stds_' + space, signal[space]['stds'])
            np.save(self.parameters[
                    'name'] + '_directions_' + space, signal[space]['directions'])
            np.save(self.parameters['name'] +
                    '_pairs_' + space, signal[space]['pairs'])

    def setup(self):
        signal = Signal(self.parameters['shape'], self.parameters['signal_model'], self.parameters[
                        'm_blocks'], self.parameters['correlation_strength'], self.parameters['normalize_stds']).generate()
        noise = Noise(self.parameters['shape'], self.parameters['noise_model']).generate(
            noise_amplitude=self.parameters['noise_amplitude'], rank=self.parameters['rank'])
        missing = Missing(self.parameters['shape'], self.parameters[
                          'missing_model']).generate(p_random=self.parameters['p_random'])
        mixed = signal['X'] + noise['X'] + missing['X']
        mixed = {'feature': mixed, 'sample': mixed.T}
        operator = LinearOperator(self.parameters['operator_name'])
        if self.parameters['operator_name'] == 'custom':
            A, Y, pairs = operator.generate(incorrect_A_std=self.parameters['incorrect_A_std'], mixed=mixed, true_signal=signal, measurements=self.parameters['measurements'], estimate_pairs=self.parameters['estimate'][
                                            0], estimate_stds=self.parameters['estimate'][1], known_correlations=self.parameters['known_correlations'], sparsity=self.parameters['sparsity'])
        else:
            A, Y, pairs = operator.generate(incorrect_A_std=self.parameters['incorrect_A_std'], true_signal=signal, true_noise=noise, measurements=self.parameters['measurements'], sparsity=self.parameters[
                                            'sparsity'], additive_noise_A_std=self.parameters['additive_noise_A_std'], additive_noise_y_std=self.parameters['additive_noise_y_std'], known_correlations=self.parameters['known_correlations'])
        cost = Cost(self.parameters['operator_name'], A, Y, pairs)
        manifold = FixedRankEmbedded(self.parameters['shape'][0], self.parameters[
                                     'shape'][1], self.parameters['rank'])
        print "cost.evaluate(noise['X'])", cost.evaluate(noise['X'])
        problem = Problem(manifold=manifold, cost=cost.evaluate,
                          verbosity=self.parameters['verbosity'])
        return signal, noise, missing, mixed, problem

    def solve(self, problem):
        solver = ConjugateGradient(logverbosity=self.parameters['logverbosity'], maxiter=self.parameters['maxiter'], maxtime=self.parameters[
                                   'maxtime'], mingradnorm=self.parameters['mingradnorm'], minstepsize=self.parameters['minstepsize'])
        result = None
        while result is None:
            try:
                noise0 = Noise(self.parameters['shape'], 'low-rank').generate(
                    rank=self.parameters['rank'], noise_amplitude=self.parameters['noise_amplitude'])
                usvt, optlog = solver.solve(problem, x=noise0['usvt'])
                #stopping_reason = optlog['stoppingreason']
                # WARNING Don't know true signal in general.
                final_cost = optlog['final_values']['f(x)']
                X = usvt[0].dot(np.diag(usvt[1])).dot(usvt[2])
                result = True
            except Exception:
                # NOTE
                # https://github.com/ContinuumIO/anaconda-issues/issues/695
                print 'Likely np.linalg.LinAlgError:', traceback.format_exc()
        return X, noise0['X'], final_cost

    def signal_to_noise(self, signal, noise):
        # NOTE Sum of std(clean signal) / std(true noise) for each
        # feature/sample column.
        snr = np.mean(np.std(signal['X'][:, signal['sample']['pairs'].ravel()], axis=0) / np.mean(np.absolute(noise['X'][:, signal['sample']['pairs'].ravel()]), axis=0)) + \
            np.mean(np.std(signal['X'][signal['feature']['pairs'].ravel()], axis=1) / np.mean(
                np.absolute(noise['X'][signal['feature']['pairs'].ravel()]), axis=1))
        return snr




# TODO make nice (PEP8 REcovery)
# TODO Run the different simulations and make plots.
# TODO Run on the real data to make figure 4 and 5.


# NOTE Higher dimensions make things exponentially better?
# WARNING must shuffle noise, e.g. if biclustered because signal is not shuffled (difficult to do to get correlation structure to stay simple...); Actually, should be simple., because don'T need to shuffle the correlations, just the rows and columns... but can't if correlations in both spaces?
# TODO constrain samples, get more accuracy for various estimations since know that have to be absolutely same and not just relative.
# TODO implement low memory footprint option with storing operators with memory map? And only one problem per node (e.g. with dummies?)
# TODO load in the real data!
# TODO work on correlations estimation function...
# TODO could take only the best runs!? But then might signelct for noise and signal that are more ameanable, e.g. fluke.
# TODO Check that all maitrices, e.g. V, U are finite before SVD is computed.
# NOTE Not having completely random signal is fine since that is never the case anaywys in real applications where the signal contains potentially redundant information?
# NOTE free_x and free_y should be integers!


#shape = 100, 120
#measurements = 1000
# min_block size = 2
# possible pairs are 100/2 = 50 + 120/2 = 60 = 110; and each of theses
# actually has 50 * 120 + 60 * 100 = 6000 + 6000 = 12000 measurements with
# sparsity 2 up to a max. of 50**2 * 120 + 60**2 * 100 = 300000 + 360000 =
# 660000 measurements (but they are not totally random or independent)


"""
# NOTE could do 3D plots that keep rank information!!! Do all small scale! Theory for Gaussian random matrices has much fewer values then entry based sensing?


# Use percentage for the size and keep sizes constant unless themselves are the contrast of interest.


# Dense measurement matrix for different signal redundancies and noise ranks and matrix sizes. In the end choose one.
# Signal redundancies are always different depending on the number of measurments (e.g. they are the min. required). # NOTE This is here too to keep it consistent; since matrices are still random but structured random in a certain way!

class Additive_Noise_A_vs_Measurements # Set sparsity=2, same as custom, and no noise.

class Additive_Noise_y_vs_Measurements # Set sparsity=2, same as custom, and no noise. # NOTE Maybe just need one class.

class Wrong_A_vs_Measurements # Fill or exchange A with stds from same distribution.

class Sparsity_vs_Measurements # Include the entry operator here.

# NOTE DO 3D one plot of the two below.

class Size_vs_Measurements # Set sparsity=2, same as custom, and no noise. # NOTE Max # of measurement if sparsity=1 is size**2; if 2 then all number of possible pairs including onse-sided overlaps.

# Again but this time doing the noise rank and none of the variations above (e.g. ideal). Set sparsity=2, same as custom, and no noise.

class Rank_vs_Measurements

# Custom measurement matrix for different signal redundancies and noise ranks and matrix sizes. In the end choose one.
# Signal redundancies are always different depending on the number of measurments (e.g. they are the min. required).
# All should start out with the ideal condition in the bottom middle (left too few measurements and right too many); this row should look the same everywhere.

class Inexact_Std_vs_Measurements

class Wrong_Pairs_vs_Measurements # In final operator construction just exchange a all those wrong pairs with new stds sampled from same distribution.

class Inexact_Points_vs_Measurements

# Maybe no need to plot. It's inherent. But stress that do not need to throw everything out just because one value missing. Also no need to impute with 0s or other computationally intensive way!
# class Missing_Values_vs_Measurements # Replenish with additional measurments? No need to set to zero! So fine as long as enough data!


# NOTE DO 3D one plot of the two below.

class Size_vs_Measurements

# Again but this time doing the noise rank and none of the variations above (e.g. ideal).

# Custom measurement matrix for different signal redundancies and matrix sizes. In the end choose one.
# Signal redundancies are always different depending on the number of measurments (e.g. they are the min. required).
# Also check different types of redundancies, e.g. random, unity or constant, and noise amplitude or model combinations.

class Rank_vs_Measurements

# Highlight perfect recovery stunning result somehow visually?!?


# TODO proper correlation estimation? Simple. Just threshold. Directions are included too.
# TODO std estimation. Simple.
"""

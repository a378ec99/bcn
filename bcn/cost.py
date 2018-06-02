"""Cost function.
"""
from __future__ import division, absolute_import

import autograd.numpy as ag


class CostMiniBatch(object):

    def __init__(self, A, y, sparsity=1, batch_size=10):
        """Creates a cost function with autograd based on linear operators A and targets y.

        Parameters
        ----------
        A : list; elements=dict, len=n_measurements
            Linear operators stored as sparse matrices.
        y : list; elements=float, len=n_measuremnts
            Target vector.
        sparsity : int, values=(1=entry sensing, 2=blind recovery, A.size=dense sensing)
            Level of sparsity of the linear operators.
        """
        self.batch_size = batch_size
        self.sparsity = sparsity
        self.A_rows = ag.array([a['row'] for a in A])
        self.A_cols = ag.array([a['col'] for a in A])
        self.A_values = ag.array([a['value'] for a in A])
        self.y = ag.array(y)
        self.indices = np.random.randint(self.y.size, size=self.batch_size)
        
    def cost_func(self, X):
        """Cost function for evaluation at X.

        Parameters
        ----------
        X : numpy.ndarray; shape=(n_samples, n_features)
            Estimated bias matrix.

        Returns
        -------
        error : float
            Squared mean error.

        Note
        ----
        Size scaling with ag.mean is not nessesary for convergence but useful for comparison of error magnitude.
        """
        if len(X) == 3:  # WARNING type changes from list to autograd.builtins.SequenceBox after first run.
            usvt = X
            X = ag.dot(usvt[0], ag.dot(ag.diag(usvt[1]), usvt[2]))
        else:
            assert ag.isfinite(X).all()
        y_est = ag.sum(
            self.A_values[self.indices] * X[self.A_rows, self.A_cols].reshape(-1, self.sparsity), axis=1)
        error = ag.mean((y_est - self.y[self.indices])**2)
        return error

        
class Cost(object):

    def __init__(self, A, y, sparsity=1):
        """Creates a cost function with autograd based on linear operators A and targets y.

        Parameters
        ----------
        A : list; elements=dict, len=n_measurements
            Linear operators stored as sparse matrices.
        y : list; elements=float, len=n_measuremnts
            Target vector.
        sparsity : int, values=(1=entry sensing, 2=blind recovery, A.size=dense sensing)
            Level of sparsity of the linear operators.
        """
        self.sparsity = sparsity
        self.A_rows = ag.array([a['row'] for a in A])
        self.A_cols = ag.array([a['col'] for a in A])
        self.A_values = ag.array([a['value'] for a in A])
        self.y = ag.array(y)

    def cost_func(self, X):
        """Cost function for evaluation at X.

        Parameters
        ----------
        X : numpy.ndarray; shape=(n_samples, n_features)
            Estimated bias matrix.

        Returns
        -------
        error : float
            Squared mean error.

        Note
        ----        
        Size scaling with ag.mean is not nessesary for convergence but useful for comparison of error magnitude.
        """
        if len(X) == 3:  # WARNING type changes from list to autograd.builtins.SequenceBox after first run.
            usvt = X
            X = ag.dot(usvt[0], ag.dot(ag.diag(usvt[1]), usvt[2]))
        else:
            assert ag.isfinite(X).all()
        y_est = ag.sum(
            self.A_values * X[self.A_rows, self.A_cols].reshape(-1, self.sparsity), axis=1)
        error = ag.mean((y_est - self.y)**2)
        return error

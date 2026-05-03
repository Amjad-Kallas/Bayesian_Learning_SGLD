import numpy as np

class BayesianLogisticRegression:

    def __init__(self, dim):
        self.theta = np.random.normal(0, 2, size=dim)

    def sigmoid(self, z):
        z = np.clip(z, -250, 250) # to prevents overflow in exp for early runs
        return 1 / (1 + np.exp(-z))

    def gradient(self, X_batch, y_batch, N):

        batch_size = X_batch.shape[0]

        z = y_batch * (X_batch @ self.theta)

        likelihood_grad = (self.sigmoid(-z) * y_batch)[:, None] * X_batch

        likelihood_grad = likelihood_grad.sum(axis=0)

        likelihood_grad *= (N / batch_size)

        prior_grad = -np.sign(self.theta)

        return prior_grad + likelihood_grad

    def log_joint(self, X, y):
    
        z = y * (X @ self.theta)

        # we add 1e-15 to prevent potential log(0)
        log_likelihood = np.sum(np.log(self.sigmoid(z) + 1e-15))

        log_prior = -np.sum(np.abs(self.theta))   # Laplace prior

        return log_likelihood + log_prior


class ICAModel:
    def __init__(self, lambda_reg=1.0):
        self.lambda_reg = lambda_reg

    def sample_langevin_noise(self, W, eps):
        M = W.T @ W

        # stabilize eigenvalues (robust way)
        eigvals, eigvecs = np.linalg.eigh(M)

        # clamp to ensure positive definiteness
        eigvals = np.clip(eigvals, 1e-6, None)

        sqrt_M = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        Z = np.random.randn(*W.shape)

        noise = Z @ sqrt_M * np.sqrt(eps)

        return noise

    def natural_grad_and_noise(self, W, X_batch, N, eps):
        n, D = X_batch.shape
        Y = X_batch @ W.T
        G = np.tanh(Y / 2.0)

        # 1. Natural gradient approximation (Equation 15 in the paper)
        term1 = N * np.eye(D) - (N / n) * (G.T @ Y)
        grad_likelihood = term1 @ W
        grad_prior = -self.lambda_reg * W @ W.T @ W

        grad_total = grad_likelihood + grad_prior

        # 2. Preconditioned Noise (Equation 16 in paper)
        WTW = W.T @ W
        # Using SVD/Eigendecomposition for numerical stability of matrix square root
        eigvals, eigvecs = np.linalg.eigh(WTW)
        eigvals = np.maximum(eigvals, 0.0)

        noise = self.sample_langevin_noise(W, eps)

        return grad_total, noise

    def full_natural_grad(self, W, X):
        """
        Computes the exact natural gradient over the entire dataset (no mini-batching, no noise).
        """
        N, D = X.shape
        Y = X @ W.T
        G = np.tanh(Y / 2.0)

        # Equation 15 exactly, over the full dataset X
        term1 = N * np.eye(D) - (G.T @ Y)
        grad_likelihood = term1 @ W
        grad_prior = -self.lambda_reg * W @ W.T @ W

        return grad_likelihood + grad_prior

    def log_posterior(self, W, X):
        """
        Computes the unnormalized log posterior for the Metropolis-Hastings acceptance step.
        """
        N, D = X.shape
        Y = X @ W.T

        # 1. Log Determinant term: N * log|det(W)|
        # slogdet returns (sign, logdet), we only need the logdet
        sign, log_det_W = np.linalg.slogdet(W)
        if sign == 0:  # Handle singular matrix edge-case
            return -np.inf

        ll_det = N * log_det_W

        # 2. Log Likelihood of the marginals: sum(log p_i(y_i))
        # The paper uses p(y) = 1 / (4 * cosh^2(y/2))
        # log p(y) = -log(4) - 2 * log(cosh(y/2))
        # For numerical stability with large values of Y, we use logaddexp
        def stable_logcosh(x):
            return np.logaddexp(x, -x) - np.log(2.0)

        ll_pdf = np.sum(-np.log(4.0) - 2.0 * stable_logcosh(Y / 2.0))

        # 3. Log Prior of W
        # Based on the natural gradient prior (-lambda * W @ W.T @ W),
        # the standard gradient is -lambda * W, meaning the log prior is proportional to -lambda/2 * ||W||_F^2
        log_prior = -(self.lambda_reg / 2.0) * np.sum(W ** 2)

        return ll_det + ll_pdf + log_prior

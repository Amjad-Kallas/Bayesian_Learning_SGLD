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



import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from src.optimizers import SGLD


class SimpleDemo:
    def __init__(self, N=100, seed=0):
        """Create a toy mixture-model demo with synthetic data and simulation methods."""
        self.N = N
        self.seed = seed
        self.X = self._generate_data()

    # ==========================================
    # Data + Model
    # ==========================================
    def _generate_data(self):
        """Generate synthetic mixture data used by the demo."""
        np.random.seed(self.seed)

        theta1_true = 0.0
        theta2_true = 1.0
        sigma_x = np.sqrt(2)

        X = []
        for _ in range(self.N):
            if np.random.rand() < 0.5:
                x = np.random.normal(theta1_true, sigma_x)
            else:
                x = np.random.normal(theta1_true + theta2_true, sigma_x)
            X.append(x)

        return np.array(X)

    def grad_log_prior(self, theta):
        """Compute the gradient of the Gaussian prior for the model parameters."""
        theta1, theta2 = theta
        return np.array([-theta1 / 10.0, -theta2 / 1.0])

    def log_prior(self, theta):
        """Evaluate the log prior probability for the model parameters."""
        theta1, theta2 = theta
        return -0.5 * (theta1**2 / 10.0 + theta2**2)

    def grad_log_likelihood(self, theta, x):
        """Compute the gradient of the log likelihood for a single observation."""
        theta1, theta2 = theta
        sigma2 = 2.0

        mu1 = theta1
        mu2 = theta1 + theta2

        p1 = np.exp(-0.5 * (x - mu1)**2 / sigma2)
        p2 = np.exp(-0.5 * (x - mu2)**2 / sigma2)

        Z = p1 + p2

        d_theta1 = (p1 * (x - mu1) / sigma2 + p2 * (x - mu2) / sigma2) / Z
        d_theta2 = (p2 * (x - mu2) / sigma2) / Z

        return np.array([d_theta1, d_theta2])

    def log_likelihood(self, theta):
        """Compute the log likelihood of the full dataset under the mixture model."""
        theta1, theta2 = theta
        sigma2 = 2.0

        mu1 = theta1
        mu2 = theta1 + theta2

        norm_const = 1.0 / np.sqrt(2 * np.pi * sigma2)
        p1 = norm_const * np.exp(-0.5 * (self.X - mu1)**2 / sigma2)
        p2 = norm_const * np.exp(-0.5 * (self.X - mu2)**2 / sigma2)

        return np.sum(np.log(0.5 * p1 + 0.5 * p2))

    def full_log_posterior(self, theta):
        """Return the sum of the log prior and the log likelihood."""
        return self.log_prior(theta) + self.log_likelihood(theta)

    def full_grad_log_posterior(self, theta):
        """Compute the gradient of the full log posterior across the dataset."""
        g_prior = self.grad_log_prior(theta)
        g_lik = np.sum([self.grad_log_likelihood(theta, x) for x in self.X], axis=0)
        return g_prior + g_lik

    # ==========================================
    # Experiments
    # ==========================================
    def run_sgld(self, total_iterations=1000000, burn_in=10000):
        """Run SGLD sampling and return posterior samples plus noise diagnostics."""
        theta = np.array([0.0, 0.0])
        optimizer = SGLD(a=0.1934, b=231, gamma=0.55)

        samples = []
        track_iters, var_inj, var_g1, var_g2 = [], [], [], []

        for t in range(total_iterations):
            idx = np.random.randint(0, self.N)
            x = self.X[idx]

            g_prior = self.grad_log_prior(theta)
            g_lik = self.grad_log_likelihood(theta, x)
            stoch_grad = g_prior + self.N * g_lik

            theta, eps_t, _ = optimizer.step(theta, stoch_grad)

            if t > burn_in and t % 10 == 0:
                samples.append(theta.copy())

            if t % 100 == 0 or t < 100:
                var_inj.append(eps_t)

                all_g = np.array([
                    self.grad_log_likelihood(theta, xi) for xi in self.X
                ])
                stoch_all = g_prior + self.N * all_g

                var_grad = (0.5 * eps_t) ** 2 * np.var(stoch_all, axis=0)

                var_g1.append(var_grad[0])
                var_g2.append(var_grad[1])
                track_iters.append(t)

        return {
            "samples": np.array(samples),
            "iters": track_iters,
            "var_injected": var_inj,
            "var_g1": var_g1,
            "var_g2": var_g2,
        }

    def run_rejection(self, total_iterations=100000):
        """Run a Metropolis-style rejection sampler and return rejection statistics."""
        theta = np.array([0.0, 0.0])

        rejection_rates, step_sizes = [], []
        current = []

        for t in range(total_iterations):
            eps_t = 10 ** (-2 - 6 * (t / total_iterations))

            idx = np.random.randint(0, self.N)
            x = self.X[idx]

            g_prior = self.grad_log_prior(theta)
            g_curr = g_prior + self.N * self.grad_log_likelihood(theta, x)

            noise = np.random.normal(0, np.sqrt(eps_t), size=theta.shape)
            theta_next = theta + 0.5 * eps_t * g_curr + noise

            g_next = self.grad_log_prior(theta_next) + self.N * \
                     self.grad_log_likelihood(theta_next, x)

            log_q_fwd = -np.sum(noise**2) / (2 * eps_t)
            log_q_rev = -np.sum(
                (theta - (theta_next + 0.5 * eps_t * g_next))**2
            ) / (2 * eps_t)

            log_p_curr = self.full_log_posterior(theta)
            log_p_next = self.full_log_posterior(theta_next)

            log_alpha = log_p_next - log_p_curr + log_q_rev - log_q_fwd
            acc = min(1.0, np.exp(log_alpha))

            current.append(1.0 - acc)

            if (t + 1) % self.N == 0:
                rejection_rates.append(np.mean(current))
                step_sizes.append(eps_t)
                current = []

            theta = theta_next

        return {
            "step_sizes": step_sizes,
            "rejection_rates": rejection_rates,
        }

    # ==========================================
    # Plotting
    # ==========================================
    def plot_posterior(self, samples):
        """Plot the true posterior contour and the estimated SGLD posterior density."""
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        t1_grid = np.linspace(-1.5, 2.5, 100)
        t2_grid = np.linspace(-3, 3, 100)
        T1, T2 = np.meshgrid(t1_grid, t2_grid)
        Z = np.zeros_like(T1)

        for i in range(T1.shape[0]):
            for j in range(T1.shape[1]):
                Z[i, j] = self.full_log_posterior([T1[i, j], T2[i, j]])

        Z_prob = np.exp(Z - np.max(Z))

        ax1.contour(T1, T2, Z_prob, levels=6, cmap='jet', linewidths=1)
        ax1.set_xlim(-1.5, 2.5)
        ax1.set_ylim(-3, 3)
        ax1.set_title("True Posterior Distribution")

        xy = np.vstack([samples[:, 0], samples[:, 1]])
        kde = gaussian_kde(xy)
        Z_est = kde(np.vstack([T1.ravel(), T2.ravel()])).reshape(T1.shape)

        ax2.imshow(
            np.rot90(Z_est),
            cmap='Greys',
            extent=[-1.5, 2.5, -3, 3],
            aspect='auto'
        )
        ax2.set_xlim(-1.5, 2.5)
        ax2.set_ylim(-3, 3)
        ax2.set_title("Estimated Posterior (SGLD)")

        plt.suptitle("Figure 1")
        plt.show()

    def plot_diagnostics(self, sgld_res, rej_res):
        """Plot SGLD noise diagnostics and rejection rate versus step size."""
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax3.plot(sgld_res["iters"], sgld_res["var_g1"], label=r'$\nabla\theta_1$ noise', color='blue', alpha=0.7)
        ax3.plot(sgld_res["iters"], sgld_res["var_g2"], label=r'$\nabla\theta_2$ noise', color='green', alpha=0.7)
        ax3.plot(sgld_res["iters"], sgld_res["var_injected"], label='injected noise', color='red')

        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlim(1, 1000000)
        ax3.set_ylim(1e-6, 1e0)
        ax3.set_xlabel('iteration')
        ax3.set_ylabel('noise variance')
        ax3.legend()
        ax3.set_title("Variances of Noise")

        ax4.plot(rej_res["step_sizes"], rej_res["rejection_rates"], color='blue', alpha=0.8)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.set_xlim(1e-8, 1e-2)
        ax4.set_ylim(1e-3, 1e0)
        ax4.set_xlabel('step size')
        ax4.set_ylabel('average rejection rate')
        ax4.set_title("Rejection vs Step Size")

        plt.suptitle("Figure 2")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    exp = SimpleDemo(N=100)

    sgld_res = exp.run_sgld()
    rej_res = exp.run_rejection()

    exp.plot_posterior(sgld_res["samples"])
    exp.plot_diagnostics(sgld_res, rej_res)
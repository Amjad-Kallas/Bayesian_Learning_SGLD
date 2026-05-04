import numpy as np
from src.models import ICAModel
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

np.random.seed(1)

class ICA:
    """Bayesian ICA via SGLD and Corrected Langevin sampling."""

    def __init__(self, model=None):
        """Initialize with an optional ICAModel; defaults to a fresh ICAModel."""
        self.model = model or ICAModel()

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_artificial_data(N=1000, D=6):
        """Generate N mixed observations from D sources (3 Laplace + 3 Gaussian), whitened."""
        # 3 super-Gaussian + 3 Gaussian
        S = np.zeros((N, D))

        for i in range(3):
            S[:, i] = np.random.laplace(size=N)  # high kurtosis

        for i in range(3, D):
            S[:, i] = np.random.randn(N)

        A = np.random.randn(D, D)  # mixing matrix
        X = S @ A.T

        X = X - np.mean(X, axis=0)

        cov = np.cov(X, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)

        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + 1e-8))
        X = X @ eigvecs @ D_inv_sqrt @ eigvecs.T

        return X, A

    # ------------------------------------------------------------------
    # Samplers
    # ------------------------------------------------------------------

    def run_sgld(self, X, steps=50000, batch_size=100):
        """Run SGLD on X and return (final W, posterior samples, iteration indices)."""
        N, D = X.shape
        W = np.eye(D)

        # Paper's hyperparameters
        a = 4.0 / N
        gamma = 0.55
        b = 1000

        samples = []
        sample_iters = []  # tracks sample collection points for the Amari plot

        cum_eps = 0.0
        threshold = None

        for t in range(steps):
            eps = a * ((b + t) ** (-gamma))

            idx = np.random.choice(N, batch_size, replace=False)
            X_batch = X[idx]

            grad, noise = self.model.natural_grad_and_noise(W, X_batch, N, eps)

            W = W + 0.5 * eps * grad + noise

            # define threshold after burn-in
            if t == 10000:
                threshold = eps  # paper uses D0 ≈ ε_t at sampling start

            if t > 10000:
                cum_eps += eps

                if cum_eps >= threshold:
                    samples.append(W.copy())
                    sample_iters.append(t)
                    cum_eps = 0.0

        return W, np.array(samples), np.array(sample_iters)

    def run_corrected_langevin(self, X, W_init, steps=500000):
        """Run Metropolis-corrected Langevin MCMC, returning (final W, thinned samples)."""
        N, D = X.shape
        W = W_init.copy()

        eps = 0.25 / N

        samples = []
        accepts = 0

        grad_W = self.model.full_natural_grad(W, X)
        log_p_W = self.model.log_posterior(W, X)

        for t in range(steps):
            WTW = W.T @ W

            eigvals, eigvecs = np.linalg.eigh(WTW)
            eigvals = np.maximum(eigvals, 1e-6)  # clamp to prevent negative/zero
            sqrt_WTW = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            noise = np.random.randn(D, D) * np.sqrt(eps)
            W_star = W + 0.5 * eps * grad_W + noise @ sqrt_WTW

            grad_W_star = self.model.full_natural_grad(W_star, X)
            log_p_W_star = self.model.log_posterior(W_star, X)

            log_q_fwd = self._calc_log_q(W, W_star, grad_W, eps)
            log_q_rev = self._calc_log_q(W_star, W, grad_W_star, eps)

            # log p(accept) = log p(W*) + log q(W*->W) - log p(W) - log q(W->W*)
            log_alpha = log_p_W_star + log_q_rev - log_p_W - log_q_fwd

            if np.log(np.random.rand()) < log_alpha:
                W = W_star
                grad_W = grad_W_star
                log_p_W = log_p_W_star
                accepts += 1

            samples.append(W.copy())

            if t % 10000 == 0:
                print(f"Step {t}/{steps} | Acceptance Rate: {accepts / (t + 1):.4f}")

        print(f"Final Acceptance Rate: {accepts / steps:.4f}")
        samples = samples[10000:]
        samples = samples[::30]  # thinning
        sample_iters = np.arange(10000, 10000 + len(samples) * 30, 30)

        return W, np.array(samples), sample_iters

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_amari_distances(self, samples, A, sample_iters, experiment="", ax=None):
        """Plot raw and online-averaged Amari distances over sampling iterations."""
        distances = []

        for W_sample in samples:
            P = W_sample @ A
            P_abs = np.abs(P)

            max_row = np.max(P_abs, axis=1, keepdims=True)
            max_col = np.max(P_abs, axis=0, keepdims=True)

            term1 = np.sum(P_abs / max_row) - P.shape[0]
            term2 = np.sum(P_abs / max_col) - P.shape[1]

            d = (term1 + term2) / (2 * P.shape[0])
            distances.append(d)

        smoothed = self._moving_average(distances)
        online_avg = np.cumsum(distances) / np.arange(1, len(distances) + 1)

        standalone = ax is None
        if standalone:
            _, ax = plt.subplots()

        ax.plot(sample_iters[100:], online_avg[100:], linewidth=1.3, label="Online avg", color='g')
        ax.plot(sample_iters[100:len(smoothed)], smoothed[100:], alpha=0.7, linewidth=0.7, label="Raw", color='b')
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Amari Distance")
        ax.set_title(f"Amari Distance {experiment}")
        ax.legend()

        if standalone:
            plt.show()

    @staticmethod
    def plot_instability(samples, X, experiment, ax=None):
        """Plot per-component instability (Eq. 21) sorted descending."""
        var_W = np.var(samples, axis=0)  # Shape: (D, D)
        var_X = np.var(X, axis=0)        # Shape: (D,)

        # Equation 21
        instability = np.sum(var_W * var_X, axis=1)

        sorted_instability = np.sort(instability)[::-1]

        D = X.shape[1]

        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(6, 4))

        ax.bar(range(1, D + 1), sorted_instability, color='navy')
        ax.set_xlabel("Sorted Component ID")
        ax.set_ylabel("Instability Metric")
        ax.set_title(experiment)

        if standalone:
            plt.show()

    @staticmethod
    def plot_2d_pdfs(samples, experiment_name):
        """Plot KDE joint densities of W(1,1)/W(1,2) and W(1,1)/W(2,1) from posterior samples."""

        def kde_plot(x, y, ax, xlabel, ylabel, title):
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)

            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()

            xx, yy = np.meshgrid(
                np.linspace(xmin, xmax, 100),
                np.linspace(ymin, ymax, 100)
            )

            grid = np.vstack([xx.ravel(), yy.ravel()])
            zz = kde(grid).reshape(xx.shape)

            ax.imshow(
                zz,
                origin="lower",
                aspect="auto",
                extent=[xmin, xmax, ymin, ymax],
                cmap="gray_r"
            )

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)

        w11 = samples[:, 0, 0]
        w12 = samples[:, 0, 1]
        w21 = samples[:, 1, 0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        kde_plot(w11, w12, axes[0], "W(1,1)", "W(1,2)", f"PDF W(1,1) vs W(1,2) {experiment_name}")
        kde_plot(w11, w21, axes[1], "W(1,1)", "W(2,1)", f"PDF W(1,1) vs W(2,1) {experiment_name}")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _moving_average(x, window=100):
        """Return a uniform moving average of x with the given window size."""
        return np.convolve(x, np.ones(window) / window, mode='valid')

    @staticmethod
    def _calc_log_q(W_from, W_to, grad_from, eps):
        """Compute log proposal density log q(W_from -> W_to) via Eq. 18."""
        D = W_from.shape[0]
        delta_W = W_to - W_from
        mean_shift = 0.5 * eps * grad_from

        diff = delta_W - mean_shift

        WTW = W_from.T @ W_from

        WTW_inv_diff_T = np.linalg.solve(WTW + 1e-6 * np.eye(D), diff.T)

        exponent = (-1.0 / (2 * eps)) * np.trace(diff @ WTW_inv_diff_T)

        sign, logdet = np.linalg.slogdet(WTW)
        if sign <= 0:
            return -np.inf

        norm_const = -0.5 * D * logdet - 0.5 * (D * D) * np.log(2 * np.pi * eps)

        return exponent + norm_const

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self):
        """Generate data, run SGLD, and return (W, samples, A, X, sample_iters)."""
        X, A = self.generate_artificial_data()
        W, samples, sample_iters = self.run_sgld(X)
        return W, samples, A, X, sample_iters


if __name__ == "__main__":
    print("Generating artificial data...")
    # The paper used 1000 data points and 6 channels [cite: 282]
    X, A = ICA.generate_artificial_data(N=1000, D=6)

    ica = ICA()

    print("\n--- Running SGLD ---")
    # The paper used a batch size of 100 for 500,000 iterations [cite: 329]
    W_sgld, samples_sgld, iters_sgld = ica.run_sgld(X, steps=50000, batch_size=100)

    print("\n--- Running Corrected Langevin ---")
    # Initialize with the final W from SGLD to force them into the same local maximum [cite: 333]
    W_corr, samples_corr, iters_corr = ica.run_corrected_langevin(X, W_init=W_sgld, steps=50000)

    print("\n--- Plotting Amari Distances ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ica.plot_amari_distances(samples_sgld, A, iters_sgld, experiment="Stoc. Lan.", ax=axes[0])
    ica.plot_amari_distances(samples_corr, A, iters_corr, experiment="Corr. Lan.", ax=axes[1])
    plt.tight_layout()
    plt.show()

    print("\n--- Plotting Instability ---")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ICA.plot_instability(samples_sgld, X, "Instability SGLD", ax=axes[0])
    ICA.plot_instability(samples_corr, X, "Instability Corrected Lan.", ax=axes[1])
    plt.tight_layout()
    plt.show()

    # ICA.plot_2d_pdfs(samples_sgld, "Stoc. Lan.")
    # ICA.plot_2d_pdfs(samples_corr, "Corr. Lan.")

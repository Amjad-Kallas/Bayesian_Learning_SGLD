import numpy as np
from src.models import ICAModel
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import FastICA

def initialize_w_fastica(X):
    # Added max_iter=1000 just to ensure it converges safely on real data
    ica = FastICA(n_components=X.shape[1], whiten=False, random_state=0, max_iter=1000)
    
    ica.fit(X)
    W = ica.components_.copy()  # shape (n_components, n_features)

    W = W / np.linalg.norm(W, axis=1, keepdims=True)

    return W

def generate_artificial_data(N=1000, D=6):
    # 3 super-Gaussian + 3 Gaussian
    S = np.zeros((N, D))
    
    for i in range(3):
        S[:, i] = np.random.laplace(size=N)  # high kurtosis
    
    for i in range(3, D):
        S[:, i] = np.random.randn(N)

    A = np.random.randn(D, D)  # mixing matrix
    X = S @ A.T

    # standardize to prevent numerical explosion
    # X = X / np.std(X, axis=0)

    # apply whitening
    X = X - np.mean(X, axis=0)

    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + 1e-8))
    X = X @ eigvecs @ D_inv_sqrt @ eigvecs.T

    return X, A


def sgld_ica_artificial(X, model, steps=50000, batch_size=100):
    N, D = X.shape
    W = np.eye(D)
    
    # Paper's hyperparameters
    a = 4.0 / N
    gamma = 0.55
    b = 1000
    
    samples = []
    sample_iters = [] # this is to track the samples, used for the amari plot    

    cum_eps = 0.0
    threshold = None
    
    for t in range(steps):
        eps = a * ((b + t) ** (-gamma))        

        idx = np.random.choice(N, batch_size, replace=False)
        X_batch = X[idx]
        
        grad, noise = model.natural_grad_and_noise(W, X_batch, N, eps)

        # Clip gradients to safely navigate the initial burn-in phase
        # grad = np.clip(grad, -1000, 1000)
        
        W = W + 0.5 * eps * grad + noise
        
        # define threshold after burn-in
        if t == 10000:
            threshold = eps   # paper uses D0 ≈ ε_t at sampling start

        if t > 10000:
            cum_eps += eps

            if cum_eps >= threshold:
                samples.append(W.copy())
                sample_iters.append(t) 
                cum_eps = 0.0
            
    return W, np.array(samples), np.array(sample_iters)

def moving_average(x, window=100):
    return np.convolve(x, np.ones(window)/window, mode='valid')



def calc_log_q(W_from, W_to, grad_from, eps):
    """
    Computes the log proposal density log q(W_from -> W_to)
    using Equation 18 and the normalization constant.
    """
    D = W_from.shape[0]
    delta_W = W_to - W_from
    mean_shift = 0.5 * eps * grad_from
    
    # 1. Calculate diff first
    diff = delta_W - mean_shift
    
    # 2. Calculate WTW
    WTW = W_from.T @ W_from
    
    # 3. Solve for (WTW^-1 * diff.T) safely 
    WTW_inv_diff_T = np.linalg.solve(WTW + 1e-6 * np.eye(D), diff.T)
    
    # 4. Now calculate the exponent using the solved term
    exponent = (-1.0 / (2 * eps)) * np.trace(diff @ WTW_inv_diff_T)
    
    # 5. Normalization constant depends on det(W^T W)^D
    sign, logdet = np.linalg.slogdet(WTW)
    if sign <= 0:
        return -np.inf

    norm_const = -0.5 * D * logdet - 0.5 * (D * D) * np.log(2 * np.pi * eps)
    
    return exponent + norm_const

def corrected_langevin_ica_artificial(X, W_init, model, steps=500000):
    N, D = X.shape
    W = W_init.copy()
    
    
    eps = 0.25 / N 
    
    samples = []
    accepts = 0
    

    grad_W = model.full_natural_grad(W, X) 
    log_p_W = model.log_posterior(W, X)    

    for t in range(steps):
        WTW = W.T @ W

        eigvals, eigvecs = np.linalg.eigh(WTW)
        eigvals = np.maximum(eigvals, 1e-6) # clamp to prevent negative/zero
        sqrt_WTW = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
        noise = np.random.randn(D, D) * np.sqrt(eps)
        W_star = W + 0.5 * eps * grad_W + noise @ sqrt_WTW
        
        # 1. Evaluate gradient and probabilities at proposed state W*
        grad_W_star = model.full_natural_grad(W_star, X)
        log_p_W_star = model.log_posterior(W_star, X)
        
        # 2. Calculate proposal probabilities log q(W -> W*) and log q(W* -> W)
        log_q_fwd = calc_log_q(W, W_star, grad_W, eps)
        log_q_rev = calc_log_q(W_star, W, grad_W_star, eps)
        
        # 3. Metropolis-Hastings Accept/Reject (Log scale for numerical stability)
        # log p(accept) = log(p(W*)) + log(q(W* -> W)) - log(p(W)) - log(q(W -> W*))
        log_alpha = log_p_W_star + log_q_rev - log_p_W - log_q_fwd
        
        # Accept or reject
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

    # apply thinning
    samples = samples[::30]

    return W, np.array(samples)

def plot_amari_distances(samples, A, sample_iters):
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


    smoothed = moving_average(distances)
    online_avg = np.cumsum(distances) / np.arange(1, len(distances) + 1)

    
    plt.plot(sample_iters[100:], online_avg[100:], linewidth=1.3, label="Online avg", color='g')
    plt.plot(sample_iters[100:len(smoothed)], smoothed[100:], alpha=0.7, linewidth=0.7, label="Raw", color='b')
    plt.xlim(0, 100000)
    plt.xlabel("Iterations")
    plt.ylabel("Amari Distance")
    plt.title("ICA convergence (SGLD)")
    plt.show()
    


def plot_instability(samples, X, experiment):
    # Calculate variance of weights across collected posterior samples
    var_W = np.var(samples, axis=0)  # Shape: (D, D)
    var_X = np.var(X, axis=0)        # Shape: (D,)
    
    # Equation 21
    instability = np.sum(var_W * var_X, axis=1)
    
    # The paper plots these sorted descending
    sorted_instability = np.sort(instability)[::-1]
    
    D = X.shape[1]

    plt.figure(figsize=(6, 4))
    plt.bar(range(1, D + 1), sorted_instability, color='navy')
    plt.xlabel("Sorted Component ID")
    plt.ylabel("Instability Metric")
    plt.title(experiment)
    plt.show()


from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np


def plot_2d_pdfs(samples, experiment_name):
    """
    experiment_name: str
        e.g. "Stoc. Lan." or "Corr. Lan."
    """

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

    # -----------------------------
    # Extract entries
    # -----------------------------
    w11 = samples[:, 0, 0]
    w12 = samples[:, 0, 1]
    w21 = samples[:, 1, 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1
    kde_plot(
        w11, w12,
        axes[0],
        "W(1,1)", "W(1,2)",
        f"PDF W(1,1) vs W(1,2) {experiment_name}"
    )

    # Plot 2
    kde_plot(
        w11, w21,
        axes[1],
        "W(1,1)", "W(2,1)",
        f"PDF W(1,1) vs W(2,1) {experiment_name}"
    )

    plt.tight_layout()
    plt.show()


def run():
    X, A = generate_artificial_data()
    model = ICAModel()

    W, samples, sample_iters = sgld_ica_artificial(X, model)

    return W, samples, A, X, sample_iters


if __name__ == "__main__":
    print("Generating artificial data...")
    # The paper used 1000 data points and 6 channels [cite: 282]
    X, A = generate_artificial_data(N=1000, D=6)
    
    # Instantiate your model 
    # (Make sure ICAModel has natural_grad_and_noise, full_natural_grad, and log_posterior)
    model = ICAModel()

    print("\n--- Running SGLD ---")
    # The paper used a batch size of 100 for 500,000 iterations [cite: 329]
    W_sgld, samples_sgld, iters_sgld = sgld_ica_artificial(
        X, model, steps=200000, batch_size=100
    )

    # ---------------------------------------------------------
    # 2. Run Corrected Langevin (MCMC)
    # ---------------------------------------------------------
    print("\n--- Running Corrected Langevin ---")
    # Initialize with the final W from SGLD to force them into the same local maximum [cite: 333]
    W_corr, samples_corr = corrected_langevin_ica_artificial(
        X, W_init=W_sgld, model=model, steps=200000
    )


    #plot_amari_distances(samples_sgld, A, iters_sgld)

    iters_corr = np.arange(len(samples_corr))
    
    #print("-> Corrected Langevin Amari Distances")
    #plot_amari_distances(samples_corr, A, iters_corr)

    plot_instability(samples_sgld, X,  "sgld")
    plot_instability(samples_corr, X, "corr")

    #plot_2d_pdfs(samples_sgld)
    #plot_2d_pdfs(samples_corr)
    
    #plot_instability(samples, X)
    
    '''X = load_meg_data()
    W_init = initialize_w_fastica(X)
    model = ICAModel(lambda_reg=0.01)

    print("Running SGLD...")
    W, samples, sample_iters = sgld_ica_meg(X, model, W_init=W_init, steps=50000, a_num=0.1)
    
    print(f"Collected {len(samples)} samples. Plotting results...")
    
    # 1. Plot Instability (This is the primary metric for the MEG data)
    plot_instability(samples, X)

    print(X.shape)'''

    
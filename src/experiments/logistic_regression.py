import numpy as np
from src.dataset import load_a9a, add_bias, train_test_split, minibatch
from src.optimizers import SGLD
from src.models import BayesianLogisticRegression
import matplotlib.pyplot as plt

np.random.seed(1)



class LogisticRegression_SGLD:
    """Bayesian logistic regression workflow with SGLD training and plotting."""

    def __init__(self, a=0.01, b=1, gamma=0.55, batch_size=10, sweeps=10, seed=1):
        self.a = a
        self.b = b
        self.gamma = gamma
        self.batch_size = batch_size
        self.sweeps = sweeps
        self.seed = seed
        np.random.seed(self.seed)

        self.model = None
        self.optimizer = None

    def load_data(self, file_path):
        """Load the a9a dataset and add a bias feature."""
        X, y = load_a9a(file_path)
        X = add_bias(X)
        return X, y

    def split_data(self, X, y):
        """Split data into training and test sets."""
        return train_test_split(X, y)

    def build_model(self, d):
        """Initialize the Bayesian logistic regression model and its SGLD optimizer."""
        self.model = BayesianLogisticRegression(d)
        self.optimizer = SGLD(a=self.a, b=self.b, gamma=self.gamma)
        return self.model, self.optimizer

    def _accumulate_probs(self, cumulative_probs, sum_of_stepsizes, probs, eps):
        """Accumulate step-size weighted prediction probabilities."""
        cumulative_probs += eps * probs
        sum_of_stepsizes += eps
        return cumulative_probs, sum_of_stepsizes

    def _compute_accuracy(self, cumulative_probs, sum_of_stepsizes, y_true):
        """Compute accuracy from weighted Bayesian predictive probabilities."""
        mean_probs = cumulative_probs / sum_of_stepsizes
        bayesian_preds = np.where(mean_probs > 0.5, 1, -1)
        return np.mean(bayesian_preds == y_true)

    def plot_log_joint(self, mean_lj, std_lj=None):
        """Plot average log joint probability per sweep with optional std bands."""
        x_axis = np.linspace(0, self.sweeps, len(mean_lj))
        plt.figure(figsize=(6, 4))
        plt.plot(x_axis, mean_lj, 'b-', linewidth=0.9, label='Mean')
        if std_lj is not None:
            plt.plot(x_axis, mean_lj + std_lj, 'b:', linewidth=0.9, dashes=(1, 6), label='+1 Std Dev')
            plt.plot(x_axis, mean_lj - std_lj, 'b:', linewidth=0.9, dashes=(1, 6), label='-1 Std Dev')
        plt.xlabel('sweeps')
        plt.ylabel('Average log joint probability per data item')
        plt.title('Log Joint Probability')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_accuracy(self, mean_acc, std_acc=None):
        """Plot test-set accuracy over sweeps with optional std bands."""
        x_axis = np.linspace(0, self.sweeps, len(mean_acc))
        plt.figure(figsize=(6, 4))
        plt.plot(x_axis, mean_acc, 'b-', linewidth=0.9, label='Mean')
        if std_acc is not None:
            plt.plot(x_axis, mean_acc + std_acc, 'b:', linewidth=0.9, dashes=(1, 6), label='+1 Std Dev')
            plt.plot(x_axis, mean_acc - std_acc, 'b:', linewidth=0.9, dashes=(1, 6), label='-1 Std Dev')

        final_accuracy = mean_acc[-1]
        plt.axhline(y=final_accuracy, color='r', linestyle='--', linewidth=0.8,
                    dashes=(10, 6), label='Acc after {} sweeps'.format(self.sweeps))
        plt.xlabel('sweeps')
        plt.ylabel('accuracy on test set')
        plt.title('Test Accuracy')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlim(0, 3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_summary(self, mean_lj, std_lj, mean_acc, std_acc):
        """Plot log joint and accuracy side by side with std deviation shading."""
        x_axis_lj = np.linspace(0, self.sweeps, len(mean_lj))
        x_axis_acc = np.linspace(0, self.sweeps, len(mean_acc))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(x_axis_lj, mean_lj, 'b-', linewidth=0.9, label='Mean')
        ax1.plot(x_axis_lj, mean_lj + std_lj, 'b:', linewidth=0.9, dashes=(1, 6), label='+1 Std Dev')
        ax1.plot(x_axis_lj, mean_lj - std_lj, 'b:', linewidth=0.9, dashes=(1, 6), label='-1 Std Dev')
        ax1.set_xlabel('sweeps')
        ax1.set_ylabel('Average log joint probability per data item')
        ax1.set_title('Log Joint Probability')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()

        ax2.plot(x_axis_acc, mean_acc, 'b-', linewidth=0.9, label='Mean')
        ax2.plot(x_axis_acc, mean_acc + std_acc, 'b:', linewidth=0.9, dashes=(1, 6), label='+1 Std Dev')
        ax2.plot(x_axis_acc, mean_acc - std_acc, 'b:', linewidth=0.9, dashes=(1, 6), label='-1 Std Dev')
        final_accuracy = mean_acc[-1]
        ax2.axhline(y=final_accuracy, color='r', linestyle='--', linewidth=0.8,
                    dashes=(10, 6), label='Acc after {} sweeps'.format(self.sweeps))
        ax2.set_xlabel('sweeps')
        ax2.set_ylabel('accuracy on test set')
        ax2.set_title('Test Accuracy')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_xlim(0, 3)
        ax2.set_ylim(0.6, 0.9)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def run_sgld_experiment(self, X_train, y_train, X_test, y_test, batch_size=None, sweeps=None):
        """Run a single SGLD experiment and return log joint and accuracy histories."""
        batch_size = batch_size or self.batch_size
        sweeps = sweeps or self.sweeps

        N = X_train.shape[0]
        d = X_train.shape[1]
        self.build_model(d)

        updates_per_sweep = N // batch_size
        total_iterations = sweeps * updates_per_sweep
        eval_interval_lj = max(1, updates_per_sweep // 20)
        eval_interval_acc = max(1, updates_per_sweep // 100)

        history_log_joint = []
        history_accuracy = []
        cumulative_weighted_probs = np.zeros(X_test.shape[0])
        sum_of_stepsizes = 0.0

        for t in range(total_iterations):
            if t % eval_interval_lj == 0 or t == total_iterations - 1:
                history_log_joint.append(self.model.log_joint(X_train, y_train) / N)

            Xb, yb = minibatch(X_train, y_train, batch_size)
            grad = self.model.gradient(Xb, yb, N)

            eps = self.optimizer.lr()
            self.model.theta, _, _ = self.optimizer.step(self.model.theta, grad)

            z = X_test @ self.model.theta
            probs = self.model.sigmoid(z)
            cumulative_weighted_probs, sum_of_stepsizes = self._accumulate_probs(
                cumulative_weighted_probs, sum_of_stepsizes, probs, eps
            )

            if t % eval_interval_acc == 0 or t == total_iterations - 1:
                history_accuracy.append(
                    self._compute_accuracy(cumulative_weighted_probs, sum_of_stepsizes, y_test)
                )

        return history_log_joint, history_accuracy

    def run(self, X, y, batch_size=None, sweeps=None):
        """Prepare data splits and run the full SGLD experiment."""
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        return self.run_sgld_experiment(X_train, y_train, X_test, y_test,
                                       batch_size=batch_size, sweeps=sweeps)


# test
if __name__ == "__main__":
    FILE_PATH = "dataset/a9a.txt"
    
    experiment = LogisticRegression_SGLD()
    X, y = experiment.load_data(FILE_PATH)

    num_runs = 2
    sweeps = experiment.sweeps
    batch_size = experiment.batch_size

    print(f"Starting experiment: {num_runs} runs, {sweeps} sweeps per run")

    all_log_joints = []
    all_accuracies = []

    for i in range(num_runs):
        print(f"  Running experiment {i+1}/{num_runs}...")
        lj_hist, acc_hist = experiment.run(X, y, batch_size=batch_size, sweeps=sweeps)
        all_log_joints.append(lj_hist)
        all_accuracies.append(acc_hist)

    all_log_joints = np.array(all_log_joints)
    all_accuracies = np.array(all_accuracies)

    mean_lj = np.mean(all_log_joints, axis=0)
    std_lj = np.std(all_log_joints, axis=0)
    mean_acc = np.mean(all_accuracies, axis=0)
    std_acc = np.std(all_accuracies, axis=0)

    experiment.plot_summary(mean_lj, std_lj, mean_acc, std_acc)

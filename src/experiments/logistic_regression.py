import numpy as np
from src.dataset import load_a9a, add_bias, train_test_split, minibatch
from src.optimizers import SGLD
from src.models import BayesianLogisticRegression
import matplotlib.pyplot as plt

np.random.seed(1)

FILE_PATH = "dataset/a9a.txt"

def run_exp_LR(X, y, batch_size=10, sweeps=10):

    # train/test split: new random split for each run
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    N = X_train.shape[0]
    d = X_train.shape[1]

    # Bayesian LR model
    model = BayesianLogisticRegression(d)

    # SGLD optimizer
    optimizer = SGLD(a=0.01, b=1, gamma=0.55)

    updates_per_sweep = N // batch_size
    total_iterations = sweeps * updates_per_sweep


    eval_interval_lj = max(1, updates_per_sweep // 20)
    
    eval_interval_acc = max(1, updates_per_sweep // 100)

    history_log_joint = []
    history_accuracy = []

    # Variables for Step-Size Weighted Bayesian Model Averaging
    cumulative_weighted_probs = np.zeros(X_test.shape[0])
    sum_of_stepsizes = 0.0

    for t in range(total_iterations):
        # Record log joint probability before the parameters (theta) changes
        if t % eval_interval_lj == 0 or t == total_iterations - 1:
            # Average log joint probability per data item
            lj = model.log_joint(X_train, y_train) / N
            history_log_joint.append(lj)


        Xb, yb = minibatch(X_train, y_train, batch_size)
        grad = model.gradient(Xb, yb, N)
        
        # Get current step size before taking the step
        eps = optimizer.lr()
        model.theta, _, _ = optimizer.step(model.theta, grad)

        # Accumulate weighted probabilities immediately (no burn-in)
        z = X_test @ model.theta
        probs = model.sigmoid(z)
        
        cumulative_weighted_probs += eps * probs
        sum_of_stepsizes += eps



        # Record Test Accuracy
        if t % eval_interval_acc == 0 or t == total_iterations - 1:
            # Calculate accuracy using the weighted Bayesian prediction
            mean_probs = cumulative_weighted_probs / sum_of_stepsizes
            bayesian_preds = np.where(mean_probs > 0.5, 1, -1)
            acc = np.mean(bayesian_preds == y_test)
            
            history_accuracy.append(acc)

    return history_log_joint, history_accuracy

if __name__ == "__main__":
    # load data
    X, y = load_a9a(FILE_PATH)

    # add bias for LR
    X = add_bias(X)

    num_runs = 2
    sweeps = 10 # equivalent to epoch in modern ML
    batch_size = 10


    print(f"Starting experiment: {num_runs} runs, {sweeps} sweeps per run")

    all_log_joints = []
    all_accuracies = []

    for i in range(num_runs):
        print(f"  Running experiment {i+1}/{num_runs}...")
        lj_hist, acc_hist = run_exp_LR(X, y, batch_size=batch_size, sweeps=sweeps)
        
        all_log_joints.append(lj_hist)
        all_accuracies.append(acc_hist)

    # Convert to numpy arrays for mean/std computation
    all_log_joints = np.array(all_log_joints)
    all_accuracies = np.array(all_accuracies)

    # Calculate Mean and 1 Standard Deviation
    mean_lj = np.mean(all_log_joints, axis=0)
    std_lj = np.std(all_log_joints, axis=0)

    mean_acc = np.mean(all_accuracies, axis=0)
    std_acc = np.std(all_accuracies, axis=0)


    # Create the x-axis (sweeps)
    x_axis_lj = np.linspace(0, sweeps, len(mean_lj))
    x_axis_acc = np.linspace(0, sweeps, len(mean_acc))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left Plot: Log Joint Probability
    ax1.plot(x_axis_lj, mean_lj, 'b-', linewidth=0.9, label='Mean')
    ax1.plot(x_axis_lj, mean_lj + std_lj, 'b:', linewidth=0.9, dashes=(1,6), label='+1 Std Dev')
    ax1.plot(x_axis_lj, mean_lj - std_lj, 'b:', linewidth=0.9, dashes=(1,6), label='-1 Std Dev')
    ax1.set_xlabel('sweeps')
    ax1.set_ylabel('Average log joint probability per data item')
    ax1.set_title('Log Joint Probability')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Right Plot: Test Accuracy
    ax2.plot(x_axis_acc, mean_acc, 'b-', linewidth=0.9, label='Mean')
    ax2.plot(x_axis_acc, mean_acc + std_acc, 'b:', linewidth=0.9, dashes=(1,6), label='+1 Std Dev')
    ax2.plot(x_axis_acc, mean_acc - std_acc, 'b:', linewidth=0.9, dashes=(1,6), label='-1 Std Dev')

    # final accuracy after 10 sweeps
    final_accuracy = mean_acc[-1]
    ax2.axhline(y=final_accuracy, color='r', linestyle='--', linewidth=0.8, dashes=(10, 6), label='Acc after 10 sweeps')

    ax2.set_xlabel('sweeps')
    ax2.set_ylabel('accuracy on test set')
    ax2.set_title('Test Accuracy')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xlim(0, 3)

    plt.tight_layout()
    plt.show()

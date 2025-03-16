#should call generate.py first 
#calls simulate.py
#calls recover.py  AFTER simulate.py 

import numpy as np
from generate import forward_equations
from simulate import simulate_observed_data
from recover import inverse_equations

def compute_bias_and_error(true_values, estimated_values):
    """
    Compute bias and squared error between true and estimated parameters.

    Parameters:
    - true_values: tuple (true ν, α, τ)
    - estimated_values: tuple (estimated ν, α, τ)

    Returns:
    - bias: np.array of bias values
    - squared_error: sum of squared error values
    """
    bias = np.array(true_values) - np.array(estimated_values)
    squared_error = np.sum(bias ** 2)
    return bias, squared_error

def generate_true_parameters():
    """
    Generate random valid values for model parameters.
    """
    alpha = np.random.uniform(0.5, 2.0)  # Boundary separation
    nu = np.random.uniform(0.5, 2.0)  # Drift rate
    tau = np.random.uniform(0.1, 0.5)  # Non-decision time
    return alpha, nu, tau

def main():
    """
    Runs the full simulate-and-recover loop for sample sizes N = 10, 40, 4000.
    Computes bias and squared error for each sample size over multiple trials.
    """
    N_values = [10, 40, 4000]  # Sample sizes to test
    num_trials = 1000  # Number of iterations for averaging bias

    print("\n=== Running Simulate-and-Recover Experiment ===")

    for N in N_values:
        print(f"\nRunning {num_trials} trials for N = {N}...")

        # Store biases and squared errors
        total_bias = np.zeros(3)  # For averaging bias (ν, α, τ)
        total_squared_error = 0  # For averaging squared error

        for _ in range(num_trials):
            # Step 1: Generate random true parameters (α, ν, τ)
            alpha_true, nu_true, tau_true = generate_true_parameters()

            # Step 2: Compute predicted summary statistics (R_pred, M_pred, V_pred)
            R_pred, M_pred, V_pred = forward_equations(alpha_true, nu_true, tau_true)

            # Step 3: Simulate observed summary statistics (add noise)
            R_obs, M_obs, V_obs = simulate_observed_data(R_pred, M_pred, V_pred, N)

            # Step 4: Recover estimated parameters (ν_est, α_est, τ_est)
            nu_est, alpha_est, tau_est = inverse_equations(R_obs, M_obs, V_obs)

            # Step 5: Compute bias and squared error for this trial
            bias, squared_error = compute_bias_and_error(
                (nu_true, alpha_true, tau_true),
                (nu_est, alpha_est, tau_est)
            )

            # Debugging prints for individual trials
            print(f"Trial Results (N={N}):")
            print(f"  True:      α={alpha_true:.3f}, ν={nu_true:.3f}, τ={tau_true:.3f}")
            print(f"  Estimated: α={alpha_est:.3f}, ν={nu_est:.3f}, τ={tau_est:.3f}")
            print(f"  Bias: {bias}")
            print(f"  Squared Error: {squared_error:.6f}\n")

            # Accumulate total bias and squared error
            total_bias += bias
            total_squared_error += squared_error

        # Compute average bias and squared error
        avg_bias = total_bias / num_trials
        avg_squared_error = total_squared_error / num_trials

        # Print final results for N
        print(f"\nFinal Averages for N = {N}:")
        print(f"  Average Bias: {avg_bias}")
        print(f"  Average Squared Error: {avg_squared_error:.6f}")

if __name__ == "__main__":
    main()




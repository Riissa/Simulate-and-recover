#should call generate.py first 
#then call simulate.py
#then call recover.py  AFTER simulate.py 

import numpy as np
import os 
import sys
from generate import forward_equations
from simulate import simulate_observed_data
from recover import inverse_equations

def compute_bias_and_error(true_values, estimated_values):
    """
    Compute bias and squared error between true and estimated parameters.
    """
    bias = np.array(true_values) - np.array(estimated_values)
    squared_error = bias ** 2
    return bias, squared_error

def generate_true_parameters():
    """
    Generate random true parameters within valid ranges.
    """
    return (
        np.random.uniform(0.5, 2),  # α (boundary separation)
        np.random.uniform(0.5, 2),  # ν (drift rate)
        np.random.uniform(0.1, 0.5) # τ (non-decision time)
    )

def main():
    """
    Runs the full simulate-and-recover loop for sample sizes N = 10, 40, 4000.
    Computes bias and squared error for each sample size.
    """
    output_file = "results.txt"

    with open(output_file, "w") as file: 

        N_values = [10, 40, 4000]  # Different sample sizes to test
        num_trials = 1000  # Number of iterations

        print("\n=== Running Simulate-and-Recover Experiment ===\n")
        

        for N in N_values:

            print(f"Running {num_trials} trials for N = {N}...")

            total_bias = np.zeros(3)
            total_squared_error = np.zeros(3)  

            # Loop only once
            for i in range(num_trials):
                # Generate true parameters
                alpha_true, nu_true, tau_true = generate_true_parameters()

                #Compute predicted summary statistics (R_pred, M_pred, V_pred)
                R_pred, M_pred, V_pred = forward_equations(alpha_true, nu_true, tau_true)

                #Simulate observed summary statistics (add noise)
                R_obs, M_obs, V_obs = simulate_observed_data(R_pred, M_pred, V_pred, N, trial_num=i)

                # Print debug info **only for the first 3 iterations**
                #if i < 3:
                #   print(f"Debug (Generate Output - Trial {i+1}): R_pred={R_pred}, M_pred={M_pred}, V_pred={V_pred}")
                #  print(f"Debug (Simulate Before Noise - Trial {i+1}): R_pred={R_pred}, M_pred={M_pred}, V_pred={V_pred}, N={N}")
                # print(f"Debug (Simulate After Noise - Trial {i+1}): R_obs={R_obs}, M_obs={M_obs}, V_obs={V_obs}")

                # Step 4: Recover estimated parameters
                nu_est, alpha_est, tau_est = inverse_equations(R_obs, M_obs, V_obs)

                # Step 5: Compute bias and squared error
                bias, squared_error = compute_bias_and_error(
                    (nu_true, alpha_true, tau_true), (nu_est, alpha_est, tau_est)
                )

                total_bias += bias
                total_squared_error += squared_error

            # Step 6: Print final results
            #avg_bias = total_bias / num_trials
            #avg_squared_error = total_squared_error / num_trials
            #file.write(f"\nTrial Results (N={N}):")
            #file.write(f"  Avg Bias:    [ ν = {avg_bias[0]:.6f}, α = {avg_bias[1]:.6f}, τ = {avg_bias[2]:.6f} ]")
            #file.write(f"  Avg Squared Error: [ ν = {avg_squared_error[0]:.6f}, α = {avg_squared_error[1]:.6f}, τ = {avg_squared_error[2]:.6f} ]\n")
            avg_bias = total_bias / num_trials
            avg_squared_error = total_squared_error / num_trials

            with open("results.txt", "a") as file:  # Append mode to keep results for each N
                file.write(f"\nN={N}\n")
                file.write(f"Average Bias: [ ν = {avg_bias[0]:.6f}, α = {avg_bias[1]:.6f}, τ = {avg_bias[2]:.6f} ]\n")
                file.write(f"Average Squared Error: [ ν = {avg_squared_error[0]:.6f}, α = {avg_squared_error[1]:.6f}, τ = {avg_squared_error[2]:.6f} ]\n\n")

                print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()

   #Co-written w/ help of chatGPT



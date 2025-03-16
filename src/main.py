#should call generate.py first 
#calls simulate.py
#calls recover.py  AFTER simulate.py 

import numpy as np

from generate import forward_equations
from simulate import simulate_observed_data
from recover import inverse_equations

def compute_bias_and_error(true_values, estimated_values):
    bias = np.array(true_values) - np.array(estimated_values)
    squared_error = np.sum(bias ** 2)
    return bias, squared_error

def generate_true_parameters():
    return np.random.uniform(0.5, 2), np.random.uniform(0.5, 2), np.random.uniform(0.1, 0.5)

def main():
    N_values = [10, 40, 4000]  
    num_trials = 1000  

    for N in N_values:
        total_bias = np.zeros(3)
        total_squared_error = 0  

        for _ in range(num_trials):
            alpha_true, nu_true, tau_true = generate_true_parameters()
            R_pred, M_pred, V_pred = forward_equations(alpha_true, nu_true, tau_true)
            R_obs, M_obs, V_obs = simulate_observed_data(R_pred, M_pred, V_pred, N)
            nu_est, alpha_est, tau_est = inverse_equations(R_obs, M_obs, V_obs)

            bias, squared_error = compute_bias_and_error(
                (nu_true, alpha_true, tau_true), (nu_est, alpha_est, tau_est)
            )
            
            total_bias += bias
            total_squared_error += squared_error
            avg_bias = total_bias / num_trials ##
            avg_squared_error = total_squared_error / num_trials ##

        #print(f"N={N}, Avg Bias: {total_bias / num_trials}, Avg Squared Error: {total_squared_error / num_trials}")
        print(f"\nTrial Results (N={N}):")
        print(f"  Avg Bias:      ν = {avg_bias[0]:.6f}, α = {avg_bias[1]:.6f}, τ = {avg_bias[2]:.6f}")
        print(f"  Avg Squared Error: {avg_squared_error:.6f}\n")

if __name__ == "__main__":
    main() 



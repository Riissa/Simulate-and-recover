import numpy as np
from generate import forward_equations

def simulate_observed_data(R_pred, M_pred, V_pred, N):
    """
    Simulates observed summary statistics (R_obs, M_obs, V_obs)
    by adding noise to the predicted values.

    Parameters:
    - R_pred: Predicted accuracy rate
    - M_pred: Predicted mean reaction time
    - V_pred: Predicted variance of reaction time
    - N: Sample size (number of trials)

    Returns:
    - R_obs: Simulated observed accuracy rate
    - M_obs: Simulated observed mean reaction time
    - V_obs: Simulated observed variance of reaction time
    """
    # Step 1: Simulate correct responses based on R_pred
    k = np.random.binomial(N, R_pred)  # Number of correct responses
    R_obs = k / N  # Equation 7

    # Step 2: Add noise to mean reaction time
    sigma_M = np.sqrt(V_pred / N)  # Standard deviation for mean RT noise
    M_obs = M_pred + np.random.normal(0, sigma_M)  # Equation 8

    # Step 3: Add noise to variance of reaction time
    sigma_V = V_pred / np.sqrt(N)  # Standard deviation for variance RT noise
    V_obs = V_pred + np.random.normal(0, sigma_V)  # Equation 9

    return R_obs, M_obs, V_obs

import numpy as np
from generate import forward_equations

def simulate_observed_data(R_pred, M_pred, V_pred, N,trial_num):
   

    """
    Simulates observed summary statistics (R_obs, M_obs, V_obs) by adding noise.

    Parameters:
    - R_pred: Predicted accuracy rate
    - M_pred: Predicted mean reaction time
    - V_pred: Predicted variance of reaction time
    - N: Sample size (10, 40, 4000)

    Returns:
    - R_obs: Simulated observed accuracy rate
    - M_obs: Simulated observed mean reaction time
    - V_obs: Simulated observed variance of reaction time
    """
    if trial_num < 3:
        print(f"Debug (Simulate Before Noise): R_pred={R_pred}, M_pred={M_pred}, V_pred={V_pred}, N={N}")

    #Clamp R_pred before binomial sampling
    R_pred = min(max(R_pred, 0.01), 0.99)
    # Equation 7: Simulate observed accuracy rate (Binomial Distribution)
    T_obs = np.random.binomial(N, R_pred)  
    R_obs = T_obs / N 

    # Simulate correct responses safely
    k = np.random.binomial(N, R_pred)
    R_obs = min(max(k / N, 0.01), 0.99)

    #Equation 8: Simulate observed mean reaction time (Normal Distribution)
    ####sigma_M = np.sqrt(max(V_pred / N, 1e-6))  
    #####M_obs = M_pred + np.random.normal(0, sigma_M)
    sigma_M = np.sqrt(V_pred / N)  
    M_obs = np.random.normal(M_pred, sigma_M)

    ####sigma_V = max(V_pred / np.sqrt(N), 1e-6)  
    ####V_obs = V_pred + np.random.normal(0, sigma_V)
    # Equation 9: Simulate observed variance of reaction time (Gamma Distribution)
    if N > 1:
        shape_param = (N - 1) / 2
        scale_param = (2 * V_pred) / (N - 1)
        V_obs = np.random.gamma(shape_param, scale_param)  
    else:
        V_obs = V_pred  # Default to predicted variance if N=1
   
    return R_obs, M_obs, V_obs

    
    
    """# Step 1: Simulate correct responses based on R_pred
    k = np.random.binomial(N, R_pred)  # Number of correct responses
    R_obs = k / N  # Equation 7

    # Step 2: Add noise to mean reaction time
    sigma_M = np.sqrt(V_pred / N)  # Standard deviation for mean RT noise
    M_obs = M_pred + np.random.normal(0, sigma_M)  # Equation 8

    # Step 3: Add noise to variance of reaction time
    sigma_V = V_pred / np.sqrt(N)  # Standard deviation for variance RT noise
    V_obs = V_pred + np.random.normal(0, sigma_V)  # Equation 9

    return R_obs, M_obs, V_obs """

import numpy as np
from simulate import simulate_observed_data  

def inverse_equations(R_obs, M_obs, V_obs):
    """
    Recover estimated parameters using inverse EZ equations.
    """
    #if statement to make sure inverse_eqns raises ValueError if R_obs is not between 0 and 1
    if not (0 <= R_obs <= 1):
        raise ValueError("R_obs must be between 0 and 1")

    try:
        # Ensure R_obs is within valid range (0.001 to 0.999 to avoid log issues)
        R_obs = np.clip(R_obs, 0.001, 0.999)

        # Compute logit transform safely
        L = np.log(R_obs / (1 - R_obs))

        # Handle potential invalid calculations
        if V_obs <= 1e-6 or R_obs == 0.5:
            return 0, 0, M_obs  # Assign default values if V_obs is too small

        # Compute drift rate (ν)
        sqrt_term = (L * (R_obs**2 * L - R_obs * L + R_obs - 0.5)) / (V_obs + 1e-6)
        ##sqrt_term = max(sqrt_term, 1e-6)  # Avoid negative sqrt issue #previous code
        ##nu_est = np.sign(R_obs - 0.5) * 4 * np.sqrt(sqrt_term) #previous code
        sqrt_term = max(sqrt_term, 1e-10) #to avoid negative sqrt issue
        nu_est = np.sign(R_obs - 0.5) * (sqrt_term ** 0.25)  # Power of 1/4 instead of sqrt


        # Compute boundary separation (α)
        alpha_est = L / nu_est if nu_est != 0 else 0

        # Compute non-decision time (τ)
        if nu_est != 0 and alpha_est != 0:
            exp_term = np.exp(-nu_est * alpha_est)
            tau_est = M_obs - (alpha_est / (2 * nu_est)) * ((1 - exp_term) / (1 + exp_term))
        else:
            tau_est = M_obs  # Default if we can't calculate

        return nu_est, alpha_est, tau_est

    except (ValueError, ZeroDivisionError, RuntimeWarning):
        return 0, 0, M_obs  # Assign default values in case of failure

#Co-written w/ help of chatGPT

import numpy as np
from simulate import simulate_observed_data  # ✅ Correct absolute import

import numpy as np

def inverse_equations(R_obs, M_obs, V_obs):
    """
    Computes estimated parameters (nu_est, alpha_est, tau_est)
    using the inverse EZ equations.

    Parameters:
    - R_obs: Observed accuracy rate
    - M_obs: Observed mean reaction time
    - V_obs: Observed variance of reaction time

    Returns:
    - nu_est: Estimated drift rate
    - alpha_est: Estimated boundary separation
    - tau_est: Estimated non-decision time
  """
import numpy as np

def inverse_equations(R_obs, M_obs, V_obs):
    """
    Recover estimated parameters using inverse EZ equations.
    """
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
        sqrt_term = max(sqrt_term, 1e-6)  # Avoid negative sqrt issue
        nu_est = np.sign(R_obs - 0.5) * 4 * np.sqrt(sqrt_term)

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



#Previous code that kinda works    
"""
    epsilon = 1e-10  # ✅ Small constant to prevent division errors

    # ✅ Ensure R_obs is within valid range (to avoid log errors)
    if R_obs <= 0 or R_obs >= 1:
        raise ValueError("R_obs must be between 0 and 1 (exclusive)")
    
    R_obs = np.clip(R_obs, 0.05, 0.95)  # ✅ Clamp to avoid extreme values
    L = np.log(R_obs / (1 - R_obs + epsilon))  # ✅ Compute L safely

    # ✅ Prevent divide-by-zero errors in V_obs
    V_obs = max(V_obs, 1e-6)

    # ✅ Compute drift rate (nu_est) using **Equation 4**
    sqrt_term = (L * (R_obs**2 * L - R_obs * L + R_obs - 0.5)) / (V_obs + 1e-6)
    sqrt_term = max(sqrt_term, 1e-6)  # ✅ Ensure no negative values
    nu_est = np.sign(R_obs - 0.5) * 4 * np.sqrt(sqrt_term)

    # ✅ Handle invalid `nu_est`
    if np.isnan(nu_est) or np.isinf(nu_est) or nu_est <= 0:
        print(f"WARNING: Invalid nu_est ({nu_est}) detected! Assigning default value.")
        nu_est = 0.5  # Default to reasonable value

    # ✅ Compute boundary separation (alpha_est) using **Equation 5**
    alpha_est = L / (nu_est + epsilon)  # ✅ Prevent divide-by-zero
    if np.isnan(alpha_est) or np.isinf(alpha_est) or alpha_est <= 0:
        print(f"WARNING: Invalid alpha_est ({alpha_est}) detected! Assigning default value.")
        alpha_est = 1.0  # Default to reasonable value

    # ✅ Compute non-decision time (tau_est) using **Equation 6**
    tau_est = M_obs - (alpha_est / (2 * nu_est)) * ((1 - np.exp(-nu_est * alpha_est)) / (1 + np.exp(-nu_est * alpha_est)))
    tau_est = max(min(tau_est, 0.5), 0.1)  # ✅ Keep τ in range

    return nu_est, alpha_est, tau_est

"""
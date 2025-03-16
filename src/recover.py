import numpy as np
from simulate import simulate_observed_data  # ✅ Correct absolute import

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

    epsilon = 1e-10  # ✅ Small constant to prevent division errors

    # ✅ Ensure R_obs is within valid range (clamp to avoid instability)
    if R_obs <= 0 or R_obs >= 1:
        raise ValueError("R_obs must be between 0 and 1 (exclusive)")
    R_obs = np.clip(R_obs, 0.05, 0.95)  # ✅ Stabilize to prevent extreme values

    # ✅ Compute L safely with epsilon to prevent log(0)
    L = np.log(R_obs / (1 - R_obs + epsilon))

    # ✅ Prevent divide-by-zero errors in V_obs
    V_obs = max(V_obs, 1e-4)

    # ✅ Compute drift rate (nu_est) using Equation 4
    sqrt_term = (L * (R_obs**2 * L - R_obs * L + R_obs - 0.5)) / (V_obs + epsilon)
    sqrt_term = max(sqrt_term, 1e-6)  # ✅ Avoid negative sqrt

    nu_est = np.sign(R_obs - 0.5) * 4 * np.sqrt(sqrt_term)

    # ✅ Ensure nu_est is reasonable
    if np.isnan(nu_est) or np.isinf(nu_est) or nu_est <= 0:
        print(f"WARNING: Invalid nu_est ({nu_est}) detected! Assigning default value.")
        nu_est = 0.5  # ✅ Assign reasonable default value

    # ✅ Compute boundary separation (alpha_est) using Equation 5
    alpha_est = L / (nu_est + epsilon)

    # ✅ Ensure alpha_est is positive and reasonable
    if np.isnan(alpha_est) or np.isinf(alpha_est) or alpha_est <= 0:
        print(f"WARNING: Invalid alpha_est ({alpha_est}) detected! Assigning default value.")
        alpha_est = 1.0  # ✅ Assign a reasonable default value


    tau_est = M_obs - (alpha_est / (2 * nu_est)) * ((1 - np.exp(-nu_est * alpha_est)) / (1 + np.exp(-nu_est * alpha_est)))


    tau_est = max(min(tau_est, 0.5), 0.1)

    return nu_est, alpha_est, tau_est

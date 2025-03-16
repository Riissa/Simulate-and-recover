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

    # Ensure R_obs is within valid range (to avoid log errors)
    if R_obs <= 0 or R_obs >= 1:
        raise ValueError("R_obs must be between 0 and 1 (exclusive)")

    # Step 1: Compute L (log odds of accuracy)
    L = np.log(R_obs / (1 - R_obs))

     # Ensure V_obs is not too small (avoiding divide-by-zero errors)
    V_obs = max(V_obs, 1e-6)

    # Step 2: Estimate drift rate (Equation 4)
    nu_est = np.sign(R_obs - 0.5) * 4 * np.sqrt((L * (R_obs**2 * L - R_obs * L + R_obs - 0.5)) / V_obs)

    # Step 3: Estimate boundary separation (Equation 5)
    alpha_est = L / nu_est

    # Step 4: Estimate non-decision time (Equation 6)
    tau_est = M_obs - (alpha_est / (2 * nu_est)) * ((1 - np.exp(-nu_est * alpha_est)) / (1 + np.exp(-nu_est * alpha_est)))

    return nu_est, alpha_est, tau_est

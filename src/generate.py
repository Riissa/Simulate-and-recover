# src/generate.py
#use equation 1, 2 and 3 to generate predicted summary stats 

import numpy as np

def forward_equations(alpha, nu, tau):
    """  
    Parameters:
    - alpha: Boundary separation (between 0.5 and 2)
    - nu: Drift rate (between 0.5 and 2)
    - tau: Non-decision time (between 0.1 and 0.5)
 
    Returns:
    - R_pred: Predicted accuracy rate
    - M_pred: Predicted mean reaction time
    - V_pred: Predicted variance of reaction time
    """
    # Check for invalid inputs
    if alpha < 0.5 or alpha > 2:
        raise ValueError("alpha must be between 0.5 and 2")
    if nu < 0.5 or nu > 2:
        raise ValueError("nu must be between 0.5 and 2")
    if tau < 0.1 or tau > 0.5:
        raise ValueError("tau must be between 0.1 and 0.5")

    # Compute y
    y = np.exp(-alpha * nu)

    # Compute predicted summary statistics
    R_pred = 1 / (y + 1)
    M_pred = tau + (alpha / (2 * nu)) * ((1 - y) / (1 + y))
    V_pred = (alpha / (2 * nu**3)) * ((1 - 2 * alpha * nu * y - y**2) / (y + 1)**2)

    return R_pred, M_pred, V_pred



import unittest
import numpy as np
from generate import forward_equations
from simulate import simulate_observed_data
from recover import inverse_equations

class TestSampleSizeEffects(unittest.TestCase):

    def test_squared_error_decreases_with_N(self):
        """Check that squared error decreases as N increases."""
        N_values = [10, 40, 4000]
        squared_errors = []

        for N in N_values:
            alpha, nu, tau = 1.2, 1.0, 0.3  # Fixed true values
            R_pred, M_pred, V_pred = forward_equations(alpha, nu, tau)
            R_obs, M_obs, V_obs = simulate_observed_data(R_pred, M_pred, V_pred, N)
            nu_est, alpha_est, tau_est = inverse_equations(R_obs, M_obs, V_obs)

            # Compute squared error manually
            bias = np.array([nu - nu_est, alpha - alpha_est, tau - tau_est])
            squared_error = np.sum(bias**2)
            squared_errors.append(squared_error)

        # Assert that squared error decreases as N increases
        self.assertTrue(squared_errors[0] > squared_errors[1] > squared_errors[2],
                        f"Squared error did not decrease with larger N values: {squared_errors}")

if __name__ == '__main__':
    unittest.main()

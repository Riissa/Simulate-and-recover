import unittest
import numpy as np
import os 
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from recover import inverse_equations  # Import YOUR function

class TestRecover(unittest.TestCase):

    def test_inverse_equations_valid(self):
        """Test if inverse equations return reasonable estimates for typical inputs."""
        R_obs, M_obs, V_obs = 0.6, 0.4, 0.02  # Typical observed values
        nu_est, alpha_est, tau_est = inverse_equations(R_obs, M_obs, V_obs)

        self.assertGreater(nu_est, 0, "Drift rate (ν) should be positive")
        self.assertGreater(alpha_est, 0, "Boundary separation (α) should be positive")
        self.assertTrue(0.1 <= tau_est <= 0.5, "Non-decision time (τ) should be within range")

    def test_improved_accuracy_with_large_N(self):
        """Test if larger N improves the accuracy of estimated parameters."""
        N_small = 10
        N_large = 4000

        # Generate reasonable true values
        alpha_true, nu_true, tau_true = np.random.uniform(0.5, 2), np.random.uniform(0.5, 2), np.random.uniform(0.1, 0.5)

        # Import the simulate function
        from simulate import simulate_observed_data
        
        # Actually simulate data with different sample sizes
        R_obs_small, M_obs_small, V_obs_small = simulate_observed_data(nu_true, alpha_true, tau_true, N_small, trial_num=3)
        R_obs_large, M_obs_large, V_obs_large = simulate_observed_data(nu_true, alpha_true, tau_true, N_large, trial_num=3)

        # Recover estimated parameters
        nu_est_small, alpha_est_small, tau_est_small = inverse_equations(R_obs_small, M_obs_small, V_obs_small)
        nu_est_large, alpha_est_large, tau_est_large = inverse_equations(R_obs_large, M_obs_large, V_obs_large)

        # Compute squared errors
        bias_small = np.array([nu_true, alpha_true, tau_true]) - np.array([nu_est_small, alpha_est_small, tau_est_small])
        bias_large = np.array([nu_true, alpha_true, tau_true]) - np.array([nu_est_large, alpha_est_large, tau_est_large])

        # Check that squared error is lower for large N
        self.assertLess(np.sum(bias_large**2), np.sum(bias_small**2), "Squared error should be lower for larger N")


    def test_extreme_R_obs_values(self):
        """Test if inverse equations handle R_obs near 0 and 1 correctly."""
        R_obs_low, M_obs, V_obs = 0.01, 0.4, 0.02
        R_obs_high = 0.99

        # Check for low R_obs
        nu_est_low, alpha_est_low, tau_est_low = inverse_equations(R_obs_low, M_obs, V_obs)
        self.assertGreater(nu_est_low, 0, "Drift rate should be positive for low R_obs")
        self.assertGreater(alpha_est_low, 0, "Boundary separation should be positive for low R_obs")

        # Check for high R_obs
        nu_est_high, alpha_est_high, tau_est_high = inverse_equations(R_obs_high, M_obs, V_obs)
        self.assertGreater(nu_est_high, 0, "Drift rate should be positive for high R_obs")
        self.assertGreater(alpha_est_high, 0, "Boundary separation should be positive for high R_obs")

    def test_invalid_inputs(self):
        """Ensure inverse equations raise errors for invalid inputs."""
        with self.assertRaises(ValueError):
            inverse_equations(-0.1, 0.4, 0.02)  # Invalid R_obs < 0

        with self.assertRaises(ValueError):
            inverse_equations(1.1, 0.4, 0.02)  # Invalid R_obs > 1
            #comment
if __name__ == "__main__":
    unittest.main()





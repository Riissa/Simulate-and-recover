import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from simulate import simulate_observed_data  # Import the correct function

class TestSimulateObservedData(unittest.TestCase):
    
    def test_valid_outputs(self):
        """Test if simulate_observed_data returns valid outputs."""
        R_pred, M_pred, V_pred = 0.75, 0.45, 0.12  # Example predicted stats
        N = 40  # Sample size

        #triall_num = 1  # Example trial number
        R_obs, M_obs, V_obs = simulate_observed_data(R_pred, M_pred, V_pred, N, trial_num=1)

        # Check that function returns three values
        self.assertEqual(len([R_obs, M_obs, V_obs]), 3)

        # Check R_obs is a probability (between 0 and 1)
        self.assertTrue(0 <= R_obs <= 1)

        # Check outputs are finite numbers (not NaN or Inf)
        self.assertTrue(np.isfinite(R_obs))
        self.assertTrue(np.isfinite(M_obs))
        self.assertTrue(np.isfinite(V_obs))

    def test_noise_effects(self):
        """Test if larger N makes observed stats closer to predicted stats."""
        R_pred, M_pred, V_pred = 0.75, 0.45, 0.12

        # Simulate with small N (more noise)
        R_obs_small, M_obs_small, V_obs_small = simulate_observed_data(R_pred, M_pred, V_pred, N=10, trial_num=1)

        # Simulate with large N (less noise) #idk what setting trial_num to 3 does
        R_obs_large, M_obs_large, V_obs_large = simulate_observed_data(R_pred, M_pred, V_pred, N=4000,trial_num=1)

        # Check if large N gives more stable results (closer to predicted)
        self.assertAlmostEqual(R_pred, R_obs_large, delta=0.1)
        self.assertAlmostEqual(M_pred, M_obs_large, delta=0.1)
        self.assertAlmostEqual(V_pred, V_obs_large, delta=0.1)

if __name__ == "__main__":
    unittest.main()

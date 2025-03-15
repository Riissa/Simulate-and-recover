import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from generate import forward_equations #import error? 

class TestForwardEquations(unittest.TestCase):
    #checks that when function_equations is given valid imputs it produces valid outputs 
    def test_valid_inputs(self):
        """Test if forward_equations returns valid outputs for in-range inputs."""
        alpha, nu, tau = 1.0, 1.0, 0.3  # Valid values in the defined range
        R_pred, M_pred, V_pred = forward_equations(alpha, nu, tau)

        # Check that three values are returned
        self.assertEqual(len([R_pred, M_pred, V_pred]), 3)

        # Check R_pred is between 0 and 1
        self.assertTrue(0 <= R_pred <= 1)

        # Check outputs are finite numbers (not NaN or Inf)
        self.assertTrue(np.isfinite(R_pred))
        self.assertTrue(np.isfinite(M_pred))
        self.assertTrue(np.isfinite(V_pred))

    def test_boundary_values(self):
        """Test function with boundary values of alpha, nu, and tau."""
        test_cases = [
            (0.5, 0.5, 0.1),  # Smallest valid values
            (2.0, 2.0, 0.5)   # Largest valid values
        ]
        for alpha, nu, tau in test_cases:
            R_pred, M_pred, V_pred = forward_equations(alpha, nu, tau)

            self.assertTrue(0 <= R_pred <= 1)
            self.assertTrue(np.isfinite(R_pred))
            self.assertTrue(np.isfinite(M_pred))
            self.assertTrue(np.isfinite(V_pred))

    def test_invalid_inputs(self):
        """Test function with invalid values like negative alpha, zero nu."""
        test_cases = [
            (-1, 1, 0.3),  # Invalid alpha
            (1, 0, 0.3),   # Invalid nu (zero)
            (1, 1, -0.1)   # Invalid tau (negative)
        ]
        for alpha, nu, tau in test_cases:
            with self.assertRaises(ValueError):  # Expect function to raise an error
                forward_equations(alpha, nu, tau)

if __name__ == '__main__':
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from optimizer.optimizers import (
    fit_pythagorean_sundman_seed,
    write_continuation_seed,
)
from optimizer.planar_cubic_bernstein import spline_basis


class BsplineToPythagoreanContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tau = np.linspace(0.0, 1.0, 2001)
        self.tof = 3.0
        rho = 1.0 + 0.1 * self.tau
        theta = 1.2 * self.tau
        z = 0.05 * np.sin(np.pi * self.tau)
        self.cylindrical = np.column_stack((rho, theta, z))

        endpoint_values = []
        endpoint_rates = []
        for tau_endpoint in (0.0, 1.0):
            rho_value = 1.0 + 0.1 * tau_endpoint
            z_value = 0.05 * np.sin(np.pi * tau_endpoint)
            rho_dot = 0.1 / self.tof
            z_dot = 0.05 * np.pi * np.cos(np.pi * tau_endpoint) / self.tof
            radius = np.hypot(rho_value, z_value)
            latitude = np.arctan2(z_value, rho_value)
            radius_dot = (rho_value * rho_dot + z_value * z_dot) / radius
            latitude_dot = (rho_value * z_dot - z_value * rho_dot) / radius**2
            endpoint_values.append([
                radius,
                np.cos(0.5 * latitude),
                np.sin(0.5 * latitude),
                1.2 * tau_endpoint,
            ])
            endpoint_rates.append([
                radius_dot,
                -0.5 * np.sin(0.5 * latitude) * latitude_dot,
                0.5 * np.cos(0.5 * latitude) * latitude_dot,
                1.2 / self.tof,
            ])
        self.endpoint_values = np.asarray(endpoint_values)
        self.endpoint_rates = np.asarray(endpoint_rates)

    def test_fit_preserves_endpoint_values_and_physical_rates(self):
        controls, arc, diagnostics = fit_pythagorean_sundman_seed(
            self.tau,
            self.cylindrical,
            self.tof,
            n_ctrl=10,
            degree=5,
            endpoint_values=self.endpoint_values,
            endpoint_rates=self.endpoint_rates,
        )
        basis = spline_basis(10, 5)
        np.testing.assert_allclose(basis(0.0) @ controls, self.endpoint_values[0], atol=1e-14)
        np.testing.assert_allclose(basis(1.0) @ controls, self.endpoint_values[1], atol=1e-14)
        np.testing.assert_allclose(
            (basis(0.0, 1) @ controls) / (arc * self.endpoint_values[0, 0]),
            self.endpoint_rates[0],
            atol=1e-13,
        )
        np.testing.assert_allclose(
            (basis(1.0, 1) @ controls) / (arc * self.endpoint_values[1, 0]),
            self.endpoint_rates[1],
            atol=1e-13,
        )
        self.assertGreater(arc, 0.0)
        self.assertLess(diagnostics["maximum_position_fit_error"], 1e-4)

    def test_seed_json_contains_controls_arc_and_peak(self):
        controls, arc, diagnostics = fit_pythagorean_sundman_seed(
            self.tau,
            self.cylindrical,
            self.tof,
            n_ctrl=10,
            degree=5,
            endpoint_values=self.endpoint_values,
            endpoint_rates=self.endpoint_rates,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed.json"
            write_continuation_seed(path, controls, arc, 0.02, diagnostics, {"success": True})
            payload = json.loads(path.read_text())
        np.testing.assert_allclose(payload["control_coefficients_h_p_q_theta"], controls)
        self.assertAlmostEqual(payload["sundman_arc_length"], arc)
        self.assertEqual(payload["initial_peak_acceleration_bound"], 0.02)


if __name__ == "__main__":
    unittest.main()

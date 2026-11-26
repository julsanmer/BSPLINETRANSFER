import unittest
from math import comb

import numpy as np

from optimizer.pythagorean_3d_bernstein import (
    cartesian_to_cylindrical_state,
    physical_state,
    pythagorean_certificate_function,
)
from optimizer.planar_cubic_bernstein import derivative


def bernstein_value(coefficients, tau):
    coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
    degree = len(coefficients) - 1
    return sum(coefficients[i] * comb(degree, i) * tau**i
               * (1.0 - tau)**(degree - i) for i in range(degree + 1))


class Pythagorean3DBernsteinTests(unittest.TestCase):
    def test_cartesian_to_cylindrical_state(self):
        radius = 1.7
        phi = 0.2
        theta = -0.4
        radius_dot = 0.03
        phi_dot = -0.02
        theta_dot = 0.7
        rho = radius * np.cos(phi)
        position = np.array([rho * np.cos(theta), rho * np.sin(theta),
                             radius * np.sin(phi)])
        rho_dot = radius_dot * np.cos(phi) - radius * np.sin(phi) * phi_dot
        z_dot = radius_dot * np.sin(phi) + radius * np.cos(phi) * phi_dot
        velocity = np.array([
            rho_dot * np.cos(theta) - rho * theta_dot * np.sin(theta),
            rho_dot * np.sin(theta) + rho * theta_dot * np.cos(theta),
            z_dot,
        ])
        result = cartesian_to_cylindrical_state(position, velocity)
        np.testing.assert_allclose(
            result, [radius, phi, theta, radius_dot, phi_dot, theta_dot], atol=1e-13)

    def test_identity_and_cleared_denominator_certificate(self):
        degree = 2
        duration = 2.3
        mu = 1.0
        w = 0.4**2
        h = np.array([1.0, 1.15, 1.3])
        p = np.array([1.0, 0.99, 0.97])
        q = np.array([0.0, 0.08, 0.14])
        theta = np.array([0.0, 0.7, 1.6])
        function, _ = pythagorean_certificate_function(degree)
        certificate = np.asarray(function(h, p, q, theta, duration, mu, w)[0]).reshape(-1)

        for tau in np.linspace(0.0, 1.0, 17):
            primitive = np.array([bernstein_value(c, tau) for c in (h, p, q, theta)])
            primitive_d = np.array([
                bernstein_value(derivative(list(c), duration), tau)
                for c in (h, p, q, theta)
            ])
            primitive_dd = np.array([
                bernstein_value(derivative(derivative(list(c), duration), duration), tau)
                for c in (h, p, q, theta)
            ])
            physical = physical_state(primitive, primitive_d, primitive_dd, mu)
            rho, _, z, *_, radius, _, _, ur, ut, uz = physical
            self.assertAlmostEqual(radius**2, rho**2 + z**2, places=12)
            expected = radius**6 * (w - ur**2 - ut**2 - uz**2)
            self.assertAlmostEqual(bernstein_value(certificate, tau), expected, places=10)


if __name__ == "__main__":
    unittest.main()

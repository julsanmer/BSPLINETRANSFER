import unittest
from math import comb

import numpy as np

from optimizer.planar_cubic_bernstein import derivative
from optimizer.pythagorean_3d_sundman import (
    SundmanTimeMap,
    arc_length_for_tof,
    physical_state,
    sundman_hull_function,
)


def bernstein_value(coefficients, sigma):
    coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
    degree = len(coefficients) - 1
    return sum(coefficients[i] * comb(degree, i) * sigma**i
               * (1.0 - sigma)**(degree - i) for i in range(degree + 1))


DEGREE = 2
ARC = 2.3
MU = 1.0
H = np.array([1.0, 1.15, 1.3])
P = np.array([1.0, 0.99, 0.97])
Q = np.array([0.0, 0.08, 0.14])
THETA = np.array([0.0, 0.7, 1.6])


class SundmanHullTests(unittest.TestCase):
    def setUp(self):
        self.function, self.hull_degree = sundman_hull_function(DEGREE)
        result = self.function(H, P, Q, THETA, ARC, MU)
        self.p_rho, self.p_theta, self.p_z, self.weight = (
            np.asarray(result[i]).reshape(-1) for i in range(4))

    def sample(self, sigma):
        primitive = np.array([bernstein_value(c, sigma) for c in (H, P, Q, THETA)])
        primitive_d = np.array([
            bernstein_value(derivative(list(c), ARC), sigma) for c in (H, P, Q, THETA)])
        primitive_dd = np.array([
            bernstein_value(derivative(derivative(list(c), ARC), ARC), sigma)
            for c in (H, P, Q, THETA)])
        return physical_state(primitive, primitive_d, primitive_dd, MU)

    def test_hull_degree_is_nine_times_the_spline_degree(self):
        self.assertEqual(self.hull_degree, 9 * DEGREE)

    def test_numerators_reproduce_the_control_acceleration(self):
        for sigma in np.linspace(0.0, 1.0, 17):
            rho, _, z, *_, radius, ur, ut, uz = self.sample(sigma)
            self.assertAlmostEqual(radius**2, rho**2 + z**2, places=12)
            weight = bernstein_value(self.weight, sigma)
            self.assertAlmostEqual(weight, radius**3, places=10)
            for numerator, control in ((self.p_rho, ur), (self.p_theta, ut),
                                       (self.p_z, uz)):
                self.assertAlmostEqual(
                    bernstein_value(numerator, sigma), weight * control, places=10)

    def test_convex_hull_bound_dominates_the_sampled_acceleration(self):
        self.assertTrue(np.all(self.weight > 0.0))
        bound = np.max(np.sqrt(self.p_rho**2 + self.p_theta**2 + self.p_z**2)
                       / self.weight)
        sampled = max(
            float(np.hypot(np.hypot(*self.sample(sigma)[-3:-1]), self.sample(sigma)[-1]))
            for sigma in np.linspace(0.0, 1.0, 201))
        self.assertGreaterEqual(bound, sampled - 1e-12)


class SundmanTimeMapTests(unittest.TestCase):
    def setUp(self):
        self.radius = [np.array([1.0, 1.2, 1.1]), np.array([1.1, 1.4, 1.5])]
        self.arc = 3.0
        self.map = SundmanTimeMap(self.radius, self.arc)

    def radius_at(self, sigma):
        spans = len(self.radius)
        index = min(int(sigma * spans), spans - 1)
        return bernstein_value(self.radius[index], sigma * spans - index)

    def test_total_time_matches_dense_quadrature(self):
        sigma = np.linspace(0.0, 1.0, 200001)
        values = np.array([self.radius_at(s) for s in sigma])
        reference = self.arc * float(np.trapezoid(values, sigma))
        self.assertAlmostEqual(self.map.total_time, reference, places=8)

    def test_time_is_the_exact_antiderivative(self):
        for sigma in np.linspace(0.0, 1.0, 13):
            grid = np.linspace(0.0, sigma, 200001)
            values = np.array([self.radius_at(s) for s in grid])
            reference = self.arc * float(np.trapezoid(values, grid))
            self.assertAlmostEqual(float(self.map.time(sigma)), reference, places=8)

    def test_inversion_round_trips(self):
        sigma = np.linspace(0.0, 1.0, 51)
        np.testing.assert_allclose(
            self.map.sigma(self.map.time(sigma)), sigma, atol=1e-12)

    def test_arc_length_reproduces_the_requested_transfer_time(self):
        tof = 4.7
        arc = arc_length_for_tof(self.radius, tof)
        self.assertAlmostEqual(SundmanTimeMap(self.radius, arc).total_time,
                               tof, places=12)


if __name__ == "__main__":
    unittest.main()

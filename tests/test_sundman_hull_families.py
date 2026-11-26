import contextlib
import io
import tempfile
import unittest
from math import comb
from pathlib import Path

import numpy as np

from optimizer.planar_cubic_bernstein import extraction_matrix, spline_basis
from optimizer.pythagorean_3d_sundman import SundmanTimeMap
from optimizer.sundman_hull_families import (
    FAMILIES,
    KSFamily,
    TwoSplineFamily,
    build_parser,
    evaluate_dense,
    family_hull_function,
    ks_factors,
    ks_form,
    ks_lift,
    revolution_lock_margins,
    revolution_lock_plan,
    solve,
    validate,
)

DEGREE = 3
ARC = 2.3
MU = 1.0
COEFFICIENTS = {
    "two-spline": np.array([[1.00, 0.00, 0.0],
                            [1.04, 0.05, 0.6],
                            [1.09, 0.11, 1.3],
                            [1.12, 0.14, 2.1]]),
    "ks": np.array([[1.00, 0.00, 0.05, 0.00],
                    [0.95, 0.35, 0.07, 0.02],
                    [0.70, 0.80, 0.10, 0.05],
                    [0.30, 1.10, 0.12, 0.09]]),
}


def bernstein_value(coefficients, sigma):
    coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
    degree = len(coefficients) - 1
    return sum(coefficients[i] * comb(degree, i) * sigma**i
               * (1.0 - sigma)**(degree - i) for i in range(degree + 1))


class HullFunctionTests(unittest.TestCase):
    def test_hull_degree_is_six_times_the_spline_degree(self):
        for family in FAMILIES.values():
            self.assertEqual(family_hull_function(family, DEGREE)[1], 6 * DEGREE)

    def test_numerators_reproduce_the_control_acceleration(self):
        basis = spline_basis(DEGREE + 1, DEGREE)
        for name, family in FAMILIES.items():
            controls = COEFFICIENTS[name]
            hull, _ = family_hull_function(family, DEGREE)
            blocks = [np.asarray(block).reshape(-1)
                      for block in hull(*controls.T, ARC, MU)]
            for sigma in np.linspace(0.0, 1.0, 13):
                state = evaluate_dense(family, basis, controls, ARC, MU, sigma)
                weight = bernstein_value(blocks[3], sigma)
                self.assertAlmostEqual(weight, state["radius"][0]**3, places=10)
                self.assertAlmostEqual(
                    bernstein_value(blocks[4], sigma), state["radius"][0],
                    places=12)
                for numerator, control in zip(blocks[:3], state["control"]):
                    self.assertAlmostEqual(
                        bernstein_value(numerator, sigma), weight * control[0],
                        places=9)

    def test_control_matches_inverse_dynamics_in_physical_time(self):
        """Independent check: differentiate the Cartesian path in time."""
        basis = spline_basis(DEGREE + 1, DEGREE)
        sigma = np.linspace(0.0, 1.0, 20001)
        for name, family in FAMILIES.items():
            controls = COEFFICIENTS[name]
            hull, _ = family_hull_function(family, DEGREE)
            radius = np.asarray(hull(*controls.T, 1.0, MU)[4]).reshape(-1)
            time = SundmanTimeMap([radius], ARC).time(sigma)
            state = evaluate_dense(family, basis, controls, ARC, MU, sigma)
            xyz = np.column_stack(family.cartesian(state["position"]))
            velocity = np.gradient(xyz, time, axis=0, edge_order=2)
            acceleration = np.gradient(velocity, time, axis=0, edge_order=2)
            expected = np.linalg.norm(
                acceleration + MU * xyz / state["radius"][:, None]**3, axis=1)
            actual = np.linalg.norm(np.column_stack(state["control"]), axis=1)
            interior = slice(50, -50)
            np.testing.assert_allclose(
                actual[interior], expected[interior], rtol=2e-5, atol=2e-5)


class KSLiftTests(unittest.TestCase):
    def test_lift_is_continuous_and_reproduces_a_two_revolution_path(self):
        angle = np.linspace(0.0, 4.0 * np.pi, 4001)
        radius = 1.0 + 0.1 * angle
        xyz = np.column_stack((radius * np.cos(angle), radius * np.sin(angle),
                               0.2 * np.sin(0.5 * angle)))
        lifted = ks_lift(xyz)
        *position, lifted_radius = ks_form(tuple(lifted.T), tuple(lifted.T))
        np.testing.assert_allclose(np.column_stack(position), xyz, atol=1e-12)
        np.testing.assert_allclose(
            lifted_radius, np.linalg.norm(xyz, axis=1), atol=1e-12)
        self.assertLess(np.max(np.linalg.norm(np.diff(lifted, axis=0), axis=1)),
                        0.01)


class RevolutionLockTests(unittest.TestCase):
    def setUp(self):
        # Planar three-revolution spiral lifted to KS and fitted by a spline.
        self.degree, self.spans = 5, 8
        self.basis = spline_basis(self.degree + self.spans, self.degree)
        sigma = np.linspace(0.0, 1.0, 4001)
        angle = 6.0 * np.pi * sigma
        radius = 1.0 + 0.5 * sigma
        lifted = ks_lift(np.column_stack((
            radius * np.cos(angle), radius * np.sin(angle),
            0.05 * np.sin(angle))))
        self.reference = (sigma, lifted)
        self.controls = np.linalg.lstsq(
            self.basis(sigma), lifted, rcond=None)[0]
        self.extractions = [
            extraction_matrix(self.basis, k / self.spans, (k + 1) / self.spans,
                              self.degree) for k in range(self.spans)]

    def test_factors_reproduce_the_ks_map(self):
        u = np.random.default_rng(1).normal(size=(50, 4))
        a_re, a_im, b_re, b_im = ks_factors(u)
        a, b = a_re + 1j * a_im, b_re + 1j * b_im
        x, y, z, radius = ks_form(tuple(u.T), tuple(u.T))
        np.testing.assert_allclose(a * b, x + 1j * y, atol=1e-12)
        np.testing.assert_allclose(0.5 * (abs(a)**2 - abs(b)**2), z, atol=1e-12)
        np.testing.assert_allclose(0.5 * (abs(a)**2 + abs(b)**2), radius,
                                   atol=1e-12)

    def test_seed_satisfies_its_own_lock_and_other_counts_do_not(self):
        depth, directions = revolution_lock_plan(
            *self.reference, self.spans, np.pi / 2)
        margins = revolution_lock_margins(
            self.extractions, self.controls, self.degree, depth, directions)
        self.assertGreater(np.min(margins), 0.0)
        # The same endpoints reached with one more revolution must violate it.
        sigma = np.linspace(0.0, 1.0, 4001)
        angle = 8.0 * np.pi * sigma
        radius = 1.0 + 0.5 * sigma
        other = np.linalg.lstsq(self.basis(sigma), ks_lift(np.column_stack((
            radius * np.cos(angle), radius * np.sin(angle),
            0.05 * np.sin(angle)))), rcond=None)[0]
        self.assertLess(np.min(revolution_lock_margins(
            self.extractions, other, self.degree, depth, directions)), 0.0)


class SmallSolveTests(unittest.TestCase):
    def run_family(self, name, extra=()):
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args([
                "--family", name, "--degree", "2", "--n-ctrl", "5", "--quiet",
                "--output", str(Path(directory) / name), *extra])
            validate(args)
            with contextlib.redirect_stdout(io.StringIO()):
                return solve(args)

    def test_min_peak_solves_pass_the_numerical_checks(self):
        for family in (TwoSplineFamily, KSFamily):
            summary = self.run_family(family.name)
            self.assertEqual(summary["status"], "Solve_Succeeded")
            self.assertTrue(summary["numerical_checks_passed"])
            self.assertLess(summary["max_endpoint_velocity_error"], 1e-9)
            self.assertLessEqual(
                summary["sampled_peak_acceleration"],
                summary["required_peak_acceleration_bound"] * (1.0 + 1e-7))

    def test_revolution_lock_keeps_the_requested_count(self):
        summary = self.run_family("ks", ("--revolution-lock",))
        lock = summary["revolution_lock"]
        self.assertTrue(summary["numerical_checks_passed"])
        self.assertAlmostEqual(summary["swept_revolutions"],
                               summary["requested_revolutions"], places=6)
        self.assertGreaterEqual(lock["solution_minimum_margin"],
                                lock["margin_floor"] * (1.0 - 1e-6))

    def test_revolution_lock_is_rejected_for_the_two_spline_family(self):
        args = build_parser().parse_args(
            ["--family", "two-spline", "--revolution-lock"])
        with self.assertRaises(ValueError):
            validate(args)

    def test_fixed_fibre_removes_the_extra_ks_variables(self):
        summary = self.run_family("ks", ("--fibre", "fixed"))
        self.assertEqual(summary["fibre_variables"], {})
        self.assertTrue(summary["numerical_checks_passed"])


if __name__ == "__main__":
    unittest.main()

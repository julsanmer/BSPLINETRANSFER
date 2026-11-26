import inspect
from math import comb
import unittest

import numpy as np

from experiments.reproduce_gondelach_fig2 import inclusive_grid
from experiments.reproduce_gondelach_fig2_bspline_cylindrical import (
    bezier_degree_elevation_matrix,
)
from experiments.run_bspline_variant_analysis import config_run_id, variant_tag
from optimizer.helpers_Bspline import build_bspline_matrices
from optimizer.optimization_Bspline_freetf import (
    solve_free_tf_cartesian_bspline,
    solve_free_tf_cylindrical_bspline,
)


REMOVED_PARAMETERS = {
    "ghost_controls",
    "seed_velocity_weight",
    "objective",
    "winding_weight",
    "winding_constraint",
}


class BsplineNominalInterfaceTests(unittest.TestCase):
    def test_solvers_expose_only_the_nominal_delta_v_formulation(self) -> None:
        for solver in (
            solve_free_tf_cartesian_bspline,
            solve_free_tf_cylindrical_bspline,
        ):
            parameters = set(inspect.signature(solver).parameters)
            self.assertFalse(REMOVED_PARAMETERS & parameters)

    def test_basis_uses_exactly_the_requested_control_points(self) -> None:
        matrices = build_bspline_matrices(n_ctrl=8, degree=3, n_fine=21)
        self.assertEqual(matrices.b0_fine.shape, (21, 8))
        self.assertEqual(matrices.tau_start, 0.0)
        self.assertEqual(matrices.tau_end, 1.0)
        np.testing.assert_allclose(
            np.sum(matrices.b0_fine, axis=1),
            np.ones(21),
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_single_segment_bspline_is_bezier(self) -> None:
        degree = 9
        matrices = build_bspline_matrices(
            n_ctrl=degree + 1,
            degree=degree,
            n_fine=101,
        )
        tau = matrices.tau_fine
        bernstein = np.column_stack(
            [
                comb(degree, i) * tau**i * (1.0 - tau) ** (degree - i)
                for i in range(degree + 1)
            ]
        )
        bernstein_degree_8 = np.column_stack(
            [
                comb(degree - 1, i)
                * tau**i
                * (1.0 - tau) ** (degree - 1 - i)
                for i in range(degree)
            ]
        )
        bernstein_degree_7 = np.column_stack(
            [
                comb(degree - 2, i)
                * tau**i
                * (1.0 - tau) ** (degree - 2 - i)
                for i in range(degree - 1)
            ]
        )
        bernstein_first = np.zeros_like(bernstein)
        bernstein_second = np.zeros_like(bernstein)
        for i in range(degree + 1):
            if i > 0:
                bernstein_first[:, i] += degree * bernstein_degree_8[:, i - 1]
            if i < degree:
                bernstein_first[:, i] -= degree * bernstein_degree_8[:, i]
            if i > 1:
                bernstein_second[:, i] += (
                    degree * (degree - 1) * bernstein_degree_7[:, i - 2]
                )
            if 0 < i < degree:
                bernstein_second[:, i] -= (
                    2.0 * degree * (degree - 1) * bernstein_degree_7[:, i - 1]
                )
            if i < degree - 1:
                bernstein_second[:, i] += (
                    degree * (degree - 1) * bernstein_degree_7[:, i]
                )
        np.testing.assert_allclose(
            matrices.b0_fine,
            bernstein,
            rtol=0.0,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            matrices.b1_fine,
            bernstein_first,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            matrices.b2_fine,
            bernstein_second,
            rtol=0.0,
            atol=1.0e-11,
        )
        np.testing.assert_array_equal(
            matrices.knots,
            np.concatenate([np.zeros(degree + 1), np.ones(degree + 1)]),
        )
        self.assertEqual(
            variant_tag(degree + 1, degree, "example"),
            "bezier_nctrl10_deg9_example",
        )

    def test_bezier_degree_elevation_preserves_curve_and_endpoint_derivatives(self) -> None:
        source_degree = 9
        target_degree = 39
        transform = bezier_degree_elevation_matrix(source_degree, target_degree)
        source_control_points = np.column_stack(
            [
                np.linspace(1.0, 1.5, source_degree + 1),
                np.linspace(0.0, 3.0 * np.pi, source_degree + 1),
                np.sin(np.linspace(0.0, 2.0 * np.pi, source_degree + 1)),
            ]
        )
        target_control_points = transform @ source_control_points
        tau = np.linspace(0.0, 1.0, 301)
        source_basis = np.column_stack(
            [
                comb(source_degree, i)
                * tau**i
                * (1.0 - tau) ** (source_degree - i)
                for i in range(source_degree + 1)
            ]
        )
        target_basis = np.column_stack(
            [
                comb(target_degree, i)
                * tau**i
                * (1.0 - tau) ** (target_degree - i)
                for i in range(target_degree + 1)
            ]
        )
        np.testing.assert_allclose(
            target_basis @ target_control_points,
            source_basis @ source_control_points,
            rtol=0.0,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            target_degree * (target_control_points[1] - target_control_points[0]),
            source_degree * (source_control_points[1] - source_control_points[0]),
            rtol=0.0,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            target_degree * (target_control_points[-1] - target_control_points[-2]),
            source_degree * (source_control_points[-1] - source_control_points[-2]),
            rtol=0.0,
            atol=1.0e-13,
        )

    def test_grid_spacing_is_inclusive_without_count_options(self) -> None:
        np.testing.assert_array_equal(
            inclusive_grid(10.0, 20.0, 4.0),
            np.asarray([10.0, 14.0, 18.0]),
        )

    def test_run_id_is_deterministic_and_configuration_sensitive(self) -> None:
        first = {"degree": 5, "n_control_points": 40}
        reordered = {"n_control_points": 40, "degree": 5}
        changed = {"degree": 3, "n_control_points": 40}
        self.assertEqual(config_run_id(first), config_run_id(reordered))
        self.assertNotEqual(config_run_id(first), config_run_id(changed))


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from experiments.reproduce_gondelach_fig2 import evaluate_time_driven
from experiments.reproduce_gondelach_fig3 import (
    FIG3_BASIS,
    build_ipopt_problem,
    ipopt_endpoint_parameters,
)


class HodographicIpoptTests(unittest.TestCase):
    def test_symbolic_objective_matches_numpy_evaluator(self) -> None:
        dep = 7304.0
        tof = 1000.0
        n_rev = 1
        n_quad = 51
        problem = build_ipopt_problem(FIG3_BASIS, n_quad, n_rev, 20, 1e-8, 1e-6, 0, "mumps", 0.0)
        parameters = ipopt_endpoint_parameters(problem, dep, tof, n_rev, "mars")

        samples = [
            np.zeros(problem.n_free),
            np.array([1.0e-3, -2.0e-3, 1.5e-3, -1.0e-3, 5.0e-4, -7.5e-4]),
        ]
        for free in samples:
            with self.subTest(free=free.tolist()):
                symbolic = problem.evaluate(free, parameters)
                numpy_value = evaluate_time_driven(
                    dep,
                    tof,
                    n_rev,
                    n_quad,
                    FIG3_BASIS,
                    free,
                    target="mars",
                )
                self.assertAlmostEqual(symbolic, numpy_value, places=10)


if __name__ == "__main__":
    unittest.main()

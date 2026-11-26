"""CasADi/IPOPT backend for the time-driven hodographic optimizer.

The endpoint conditions are eliminated in exactly the same way as in the
NumPy Gondelach evaluator.  Only the remaining basis coefficients are NLP
variables; the radial-positivity condition is supplied to IPOPT explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np


def simpson_weights(n: int) -> np.ndarray:
    """Return weights whose dot product integrates on a uniform [0, 1] grid."""
    n = int(n)
    if n < 3 or n % 2 == 0:
        raise ValueError("The IPOPT hodographic backend requires an odd quadrature order >= 3")
    weights = np.ones(n, dtype=float)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return weights / (3.0 * float(n - 1))


@dataclass
class HodographicIpoptResult:
    x: np.ndarray
    fun: float
    success: bool
    message: str
    nit: int
    nfev: int


class TimeDrivenHodographicIpopt:
    """Reusable IPOPT NLP for one basis, quadrature order, and revolution branch."""

    def __init__(
        self,
        radial_values: np.ndarray,
        radial_derivatives: np.ndarray,
        radial_integrals: np.ndarray,
        transverse_values: np.ndarray,
        transverse_derivatives: np.ndarray,
        axial_values: np.ndarray,
        axial_derivatives: np.ndarray,
        axial_integrals: np.ndarray,
        mu: float,
        dv_conversion: float,
        *,
        radial_boundary_integrals: np.ndarray | None = None,
        transverse_boundary_values: np.ndarray | None = None,
        max_iter: int = 5000,
        tol: float = 1.0e-8,
        acceptable_tol: float = 1.0e-6,
        print_level: int = 0,
        linear_solver: str = "mumps",
        dv_eps: float = 0.0,
    ) -> None:
        arrays = [
            radial_values,
            radial_derivatives,
            radial_integrals,
            transverse_values,
            transverse_derivatives,
            axial_values,
            axial_derivatives,
            axial_integrals,
        ]
        arrays = [np.asarray(value, dtype=float) for value in arrays]
        n_nodes = arrays[0].shape[0]
        if any(value.shape[0] != n_nodes for value in arrays):
            raise ValueError("All hodographic basis matrices must use the same nodes")
        if n_nodes % 2 == 0:
            raise ValueError("The IPOPT backend requires an odd number of Simpson nodes")

        br, dbr, ir, bt, dbt, bz, dbz, iz = [ca.DM(value) for value in arrays]
        ir_boundary_np = (
            arrays[2]
            if radial_boundary_integrals is None
            else np.asarray(radial_boundary_integrals, dtype=float)
        )
        bt_boundary_np = (
            arrays[3]
            if transverse_boundary_values is None
            else np.asarray(transverse_boundary_values, dtype=float)
        )
        if ir_boundary_np.shape[0] != bt_boundary_np.shape[0]:
            raise ValueError("Transverse-boundary basis matrices must use the same nodes")
        if ir_boundary_np.shape[1] != arrays[2].shape[1]:
            raise ValueError("Radial boundary-integral matrix has the wrong number of terms")
        if bt_boundary_np.shape[1] != arrays[3].shape[1]:
            raise ValueError("Transverse boundary-value matrix has the wrong number of terms")
        n_boundary_nodes = int(ir_boundary_np.shape[0])
        if n_boundary_nodes < 3 or n_boundary_nodes % 2 == 0:
            raise ValueError("The transverse boundary quadrature requires an odd node count >= 3")
        ir_boundary = ca.DM(ir_boundary_np)
        bt_boundary = ca.DM(bt_boundary_np)
        counts = [int(br.shape[1]) - 3, int(bt.shape[1]) - 3, int(bz.shape[1]) - 3]
        if any(count < 0 for count in counts):
            raise ValueError("Each velocity component requires at least three basis terms")
        self.n_free = int(sum(counts))
        if self.n_free <= 0:
            raise ValueError("IPOPT requires at least one free hodographic coefficient")

        x = ca.MX.sym("free", self.n_free)
        # p = [tf, q0(3), qdot0(3), qf(3), qdotf(3), theta_target]
        p = ca.MX.sym("endpoint", 14)
        tf = p[0]
        q0 = p[1:4]
        qdot0 = p[4:7]
        qf = p[7:10]
        qdotf = p[10:13]
        theta_target = p[13]

        offsets = np.cumsum([0, *counts])
        xr = x[int(offsets[0]) : int(offsets[1])]
        xt = x[int(offsets[1]) : int(offsets[2])]
        xz = x[int(offsets[2]) : int(offsets[3])]

        def endpoint_coefficients(
            values: ca.DM,
            integrals: ca.DM,
            free: ca.MX,
            y0: ca.MX,
            yf: ca.MX,
            displacement: ca.MX,
        ) -> ca.MX:
            matrix = ca.vertcat(values[0, :3], values[-1, :3], tf * integrals[-1, :3])
            rhs = ca.vertcat(y0, yf, displacement)
            if int(free.numel()):
                rhs -= ca.vertcat(
                    ca.mtimes(values[0, 3:], free),
                    ca.mtimes(values[-1, 3:], free),
                    tf * ca.mtimes(integrals[-1, 3:], free),
                )
            return ca.vertcat(ca.solve(matrix, rhs), free)

        coeff_r = endpoint_coefficients(br, ir, xr, qdot0[0], qdotf[0], qf[0] - q0[0])
        vr = ca.mtimes(br, coeff_r)
        rho = q0[0] + tf * ca.mtimes(ir, coeff_r)

        weights = ca.DM(simpson_weights(n_nodes))
        boundary_weights = ca.DM(simpson_weights(n_boundary_nodes))
        rho_boundary = q0[0] + tf * ca.mtimes(ir_boundary, coeff_r)
        angle_integrals = tf * ca.mtimes(bt_boundary.T, boundary_weights / rho_boundary)
        theta_matrix = ca.vertcat(
            bt_boundary[0, :3], bt_boundary[-1, :3], angle_integrals[:3].T
        )
        theta_rhs = ca.vertcat(qdot0[1], qdotf[1], theta_target)
        if int(xt.numel()):
            theta_rhs -= ca.vertcat(
                ca.mtimes(bt_boundary[0, 3:], xt),
                ca.mtimes(bt_boundary[-1, 3:], xt),
                ca.dot(angle_integrals[3:], xt),
            )
        coeff_t = ca.vertcat(ca.solve(theta_matrix, theta_rhs), xt)

        coeff_z = endpoint_coefficients(bz, iz, xz, qdot0[2], qdotf[2], qf[2] - q0[2])
        vtheta = ca.mtimes(bt, coeff_t)
        vz = ca.mtimes(bz, coeff_z)
        z = q0[2] + tf * ca.mtimes(iz, coeff_z)

        vr_dot = ca.mtimes(dbr, coeff_r) / tf
        vtheta_dot = ca.mtimes(dbt, coeff_t) / tf
        vz_dot = ca.mtimes(dbz, coeff_z) / tf
        distance = ca.sqrt(rho * rho + z * z)
        f_r = vr_dot - vtheta * vtheta / rho + float(mu) * rho / (distance**3)
        f_theta = vtheta_dot + vr * vtheta / rho
        f_z = vz_dot + float(mu) * z / (distance**3)
        eps = float(dv_eps)
        if eps < 0.0:
            raise ValueError("dv_eps must be nonnegative")
        accel = ca.sqrt(f_r * f_r + f_theta * f_theta + f_z * f_z + eps**2) - eps
        objective = tf * float(dv_conversion) * ca.dot(weights, accel)

        self.objective_function = ca.Function("hodographic_dv", [x, p], [objective, rho])
        nlp = {"x": x, "p": p, "f": objective, "g": rho_boundary}
        options: dict[str, Any] = {
            "print_time": bool(print_level > 0),
            "error_on_fail": False,
            "ipopt.print_level": int(print_level),
            "ipopt.sb": "yes",
            "ipopt.max_iter": int(max_iter),
            "ipopt.tol": float(tol),
            "ipopt.acceptable_tol": float(acceptable_tol),
            "ipopt.linear_solver": str(linear_solver),
        }
        self._solver = ca.nlpsol("hodographic_ipopt", "ipopt", nlp, options)
        self._lbg = np.full(n_boundary_nodes, 1.0e-6, dtype=float)
        self._ubg = np.full(n_boundary_nodes, np.inf, dtype=float)

    @staticmethod
    def endpoint_parameters(
        tf_days: float,
        q0: np.ndarray,
        qdot0: np.ndarray,
        qf: np.ndarray,
        qdotf: np.ndarray,
        theta_target: float,
    ) -> np.ndarray:
        return np.concatenate(
            [
                [float(tf_days)],
                np.asarray(q0, dtype=float).reshape(3),
                np.asarray(qdot0, dtype=float).reshape(3),
                np.asarray(qf, dtype=float).reshape(3),
                np.asarray(qdotf, dtype=float).reshape(3),
                [float(theta_target)],
            ]
        )

    def evaluate(self, x: np.ndarray, parameters: np.ndarray) -> float:
        value, _ = self.objective_function(np.asarray(x, dtype=float), np.asarray(parameters, dtype=float))
        return float(value)

    def solve(self, x0: np.ndarray, parameters: np.ndarray) -> HodographicIpoptResult:
        solution = self._solver(
            x0=np.asarray(x0, dtype=float),
            p=np.asarray(parameters, dtype=float),
            lbg=self._lbg,
            ubg=self._ubg,
        )
        stats = self._solver.stats()
        return HodographicIpoptResult(
            x=np.asarray(solution["x"], dtype=float).reshape(-1),
            fun=float(solution["f"]),
            success=bool(stats.get("success", False)),
            message=str(stats.get("return_status", "unknown IPOPT status")),
            nit=int(stats.get("iter_count", 0)),
            nfev=int(stats.get("n_call_nlp_f", stats.get("iter_count", 0))),
        )

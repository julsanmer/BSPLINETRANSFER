"""3D B-spline shaping with an exact Pythagorean radial parameterization.

The optimized splines are h(t), p(t), q(t), and theta(t).  Cylindrical
position is reconstructed on every polynomial span as

    rho = h (p^2 - q^2),   z = 2 h p q,   R = h (p^2 + q^2),

so R^2 = rho^2 + z^2 holds identically.  After clearing the R^3 gravity
denominator, the continuous acceleration limit is certified through

    P = w R^6 - (R^3 A_rho + mu rho)^2
                - R^6 A_theta^2 - (R^3 z_ddot + mu z)^2 >= 0.

All Bernstein coefficients of P are constrained to be nonnegative on every
span (and optional subdivision).  This is a sufficient continuous bound, not
a grid constraint.  The default quadratic demonstration has only one free
control point for each of four primitive splines: four trajectory variables
plus the optimized squared peak bound.

In ``min-peak`` mode the peak acceleration is the optimized result, so no
available acceleration limit enters the problem and ``--u-max`` is rejected.
The squared peak variable and the certificate coefficients are instead made
dimensionless with the Bernstein peak bound of the seed trajectory.  In
``energy`` mode ``--u-max`` is required and acts as a fixed acceleration cap.

Run from the repository root, for example:

    python -m optimizer.pythagorean_3d_bernstein --quiet
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
from time import perf_counter

import casadi as ca
import numpy as np

from experiments.reference_metrics import evaluate_reference_metrics
from optimizer.planar_cubic_bernstein import (
    CANONICAL_ACCEL_TO_M_S2,
    CANONICAL_DV_TO_KM_S,
    add,
    derivative,
    elevate,
    extraction_matrix,
    multiply,
    spline_basis,
    subdivision_matrices,
)
from optimizer.canonical_units import UT_YR


DAYS_PER_CANONICAL_TIME = 365.25 * UT_YR
CANONICAL_ACCEL_TO_MM_S2 = 1000.0 * CANONICAL_ACCEL_TO_M_S2


def departure_epoch(value):
    """Accept MJD2000 UTC days or an ISO calendar date."""
    try:
        return float(value)
    except ValueError:
        try:
            date = datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Use MJD2000 days or YYYY-MM-DD") from exc
        return (date - datetime(2000, 1, 1, 12)).total_seconds() / 86400.0


def cartesian_to_cylindrical_state(position, velocity):
    """Convert Cartesian state to [R, phi, theta] and their time derivatives."""
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    x, y, z = position
    vx, vy, vz = velocity
    rho = float(np.hypot(x, y))
    radius = float(np.linalg.norm(position))
    if rho <= 0 or radius <= 0 or not np.all(np.isfinite([*position, *velocity])):
        raise ValueError("SPICE endpoint is invalid in cylindrical coordinates")
    rho_dot = float((x * vx + y * vy) / rho)
    radius_dot = float(np.dot(position, velocity) / radius)
    phi = float(np.arctan2(z, rho))
    phi_dot = float((rho * vz - z * rho_dot) / radius**2)
    theta = float(np.arctan2(y, x))
    theta_dot = float((x * vy - y * vx) / rho**2)
    return np.array([radius, phi, theta, radius_dot, phi_dot, theta_dot])


def configure_spice_case(args):
    """Replace synthetic endpoints with rotated heliocentric SPICE states."""
    from experiments.spice_ephemeris import spice_metadata, spice_state

    if args.mu != 1.0:
        raise ValueError("SPICE mode requires the solver's solar canonical mu=1")
    departure = float(args.departure_date)
    arrival = departure + float(args.tof_days)
    kernel = spice_metadata(args.spice_meta_kernel)["spice_meta_kernel"]
    position0, velocity0_day = spice_state("EARTH", departure, kernel)
    positionf, velocityf_day = spice_state(args.spice_target_name, arrival, kernel)
    rotation_angle = float(np.arctan2(position0[1], position0[0]))
    cosine, sine = np.cos(rotation_angle), np.sin(rotation_angle)
    rotation = np.array([[cosine, sine, 0.0],
                         [-sine, cosine, 0.0],
                         [0.0, 0.0, 1.0]])
    position0_rotated = rotation @ position0
    positionf_rotated = rotation @ positionf
    velocity0 = rotation @ velocity0_day * DAYS_PER_CANONICAL_TIME
    velocityf = rotation @ velocityf_day * DAYS_PER_CANONICAL_TIME
    state0 = cartesian_to_cylindrical_state(position0_rotated, velocity0)
    statef = cartesian_to_cylindrical_state(positionf_rotated, velocityf)
    base_angle = float(np.mod(statef[2], 2.0 * np.pi))

    args.r0, args.rf = float(state0[0]), float(statef[0])
    args.phi0, args.phif = float(state0[1]), float(statef[1])
    args.rdot0, args.rdotf = float(state0[3]), float(statef[3])
    args.phidot0, args.phidotf = float(state0[4]), float(statef[4])
    args.theta_dot0, args.theta_dotf = float(state0[5]), float(statef[5])
    args.theta_final = base_angle + 2.0 * np.pi * args.n_rev
    args.tof = float(args.tof_days) / DAYS_PER_CANONICAL_TIME
    args._spice_context = {
        "departure_mjd2000": departure,
        "arrival_mjd2000": arrival,
        "tof_days": float(args.tof_days),
        "meta_kernel": kernel,
        "target_name": args.spice_target_name,
        "rotation_matrix": rotation.tolist(),
        "earth_position_au": np.asarray(position0).tolist(),
        "earth_velocity_au_day": np.asarray(velocity0_day).tolist(),
        "target_position_au": np.asarray(positionf).tolist(),
        "target_velocity_au_day": np.asarray(velocityf_day).tolist(),
        "earth_rotated_state_canonical": np.concatenate((position0_rotated, velocity0)).tolist(),
        "target_rotated_state_canonical": np.concatenate((positionf_rotated, velocityf)).tolist(),
        "earth_R_phi_theta_rates": state0.tolist(),
        "target_R_phi_theta_rates": statef.tolist(),
        "unwrapped_theta_final": float(args.theta_final),
        "n_rev": int(args.n_rev),
    }


def spice_plot_arcs(args):
    """Return rotated heliocentric SPICE arcs over the transfer interval."""
    from experiments.spice_ephemeris import spice_state

    context = args._spice_context
    epochs = np.linspace(context["departure_mjd2000"], context["arrival_mjd2000"],
                         args.orbit_arc_samples)
    rotation = np.asarray(context["rotation_matrix"])
    earth = np.vstack([
        rotation @ spice_state("EARTH", float(epoch), context["meta_kernel"])[0]
        for epoch in epochs
    ])
    target = np.vstack([
        rotation @ spice_state(context["target_name"], float(epoch),
                               context["meta_kernel"])[0]
        for epoch in epochs
    ])
    return earth, target


def scale(coefficients, factor):
    return [factor * value for value in coefficients]


def pythagorean_certificate_function(degree):
    """Return exact span functions for P, rho, R, and R^6 coefficients."""
    h = ca.SX.sym("h", degree + 1)
    p = ca.SX.sym("p", degree + 1)
    q = ca.SX.sym("q", degree + 1)
    theta = ca.SX.sym("theta", degree + 1)
    duration = ca.SX.sym("duration")
    mu = ca.SX.sym("mu")
    peak_squared = ca.SX.sym("peak_squared")

    h_c = [h[i] for i in range(degree + 1)]
    p_c = [p[i] for i in range(degree + 1)]
    q_c = [q[i] for i in range(degree + 1)]
    theta_c = [theta[i] for i in range(degree + 1)]

    p2 = multiply(p_c, p_c)
    q2 = multiply(q_c, q_c)
    rho = multiply(h_c, add(p2, q2, -1.0))
    z = scale(multiply(h_c, multiply(p_c, q_c)), 2.0)
    radius = multiply(h_c, add(p2, q2))

    rho_d = derivative(rho, duration)
    rho_dd = derivative(rho_d, duration)
    z_dd = derivative(derivative(z, duration), duration)
    theta_d = derivative(theta_c, duration)
    theta_dd = derivative(theta_d, duration)

    a_rho = add(rho_dd, multiply(rho, multiply(theta_d, theta_d)), -1.0)
    a_theta = add(multiply(rho, theta_dd), multiply(rho_d, theta_d), 2.0)
    radius2 = multiply(radius, radius)
    radius3 = multiply(radius2, radius)
    radius6 = multiply(radius3, radius3)

    radial_numerator = add(multiply(radius3, a_rho), scale(rho, mu))
    axial_numerator = add(multiply(radius3, z_dd), scale(z, mu))
    certificate = add(scale(radius6, peak_squared),
                      multiply(radial_numerator, radial_numerator), -1.0)
    certificate = add(certificate,
                      multiply(radius6, multiply(a_theta, a_theta)), -1.0)
    certificate = add(certificate,
                      multiply(axial_numerator, axial_numerator), -1.0)
    certificate_degree = len(certificate) - 1

    return ca.Function(
        f"pythagorean_3d_certificate_p{degree}",
        [h, p, q, theta, duration, mu, peak_squared],
        [ca.vertcat(*certificate), ca.vertcat(*rho), ca.vertcat(*radius),
         ca.vertcat(*elevate(radius6, certificate_degree))],
        ["h", "p", "q", "theta", "duration", "mu", "peak_squared"],
        ["certificate", "rho", "radius", "radius6"],
    ), certificate_degree


def primitive_endpoint_data(args):
    """Construct primitive endpoint values/derivatives from R and latitude."""
    radii = np.array([args.r0, args.rf], dtype=float)
    latitude = np.array([args.phi0, args.phif], dtype=float)
    radius_rate = np.array([args.rdot0, args.rdotf], dtype=float)
    latitude_rate = np.array([args.phidot0, args.phidotf], dtype=float)
    theta = np.array([0.0, args.theta_final], dtype=float)
    # Each default endpoint is the maximum-latitude point of a circular orbit
    # through the central body.  At that point z_dot=rho_dot=0 and the inertial
    # circular speed R*n gives theta_dot=n/cos(phi).
    default_theta_rate = np.sqrt(args.mu / radii**3) / np.cos(latitude)
    theta_rate = np.array([
        args.theta_dot0 if args.theta_dot0 is not None else default_theta_rate[0],
        args.theta_dotf if args.theta_dotf is not None else default_theta_rate[1],
    ])

    half = 0.5 * latitude
    primitive_values = np.column_stack((radii, np.cos(half), np.sin(half), theta))
    primitive_rates = np.column_stack((
        radius_rate,
        -0.5 * np.sin(half) * latitude_rate,
        0.5 * np.cos(half) * latitude_rate,
        theta_rate,
    ))
    rho = radii * np.cos(latitude)
    z = radii * np.sin(latitude)
    rho_rate = radius_rate * np.cos(latitude) - radii * np.sin(latitude) * latitude_rate
    z_rate = radius_rate * np.sin(latitude) + radii * np.cos(latitude) * latitude_rate
    physical_values = np.column_stack((rho, theta, z))
    physical_rates = np.column_stack((rho_rate, theta_rate, z_rate))
    return primitive_values, primitive_rates, physical_values, physical_rates


def initial_primitive_coefficients(basis, values, rates, tof):
    """Fit a cubic-Hermite endpoint seed in the selected B-spline basis."""
    degree = int(basis.k)
    knots = basis.t
    tau = np.array([np.mean(knots[i + 1:i + degree + 1])
                    for i in range(len(basis.c))])
    h00 = 2 * tau**3 - 3 * tau**2 + 1
    h10 = tau**3 - 2 * tau**2 + tau
    h01 = -2 * tau**3 + 3 * tau**2
    h11 = tau**3 - tau**2
    samples = (h00[:, None] * values[0] + h10[:, None] * tof * rates[0]
               + h01[:, None] * values[1] + h11[:, None] * tof * rates[1])
    return np.linalg.solve(basis(tau), samples)


def build_clamped_primitives(opti, basis, values, rates, tof, seed):
    """Eliminate endpoint value and first-derivative conditions analytically."""
    start_d = basis(0.0, 1)
    end_d = basis(1.0, 1)
    c0 = values[0:1]
    c1 = (tof * rates[0:1] - start_d[0] * c0) / start_d[1]
    c_last = values[1:2]
    c_prev = (tof * rates[1:2] - end_d[-1] * c_last) / end_d[-2]
    n_free = len(basis.c) - 4
    projected = np.vstack((c0, c1, seed[2:-2], c_prev, c_last))
    if n_free:
        free = opti.variable(n_free, 4)
        opti.set_initial(free, projected[2:-2])
        controls = ca.vertcat(ca.DM(c0), ca.DM(c1), free,
                              ca.DM(c_prev), ca.DM(c_last))
    else:
        free = None
        controls = ca.DM(projected)
    return controls, free, projected, n_free


def physical_state(primitives, primitive_d, primitive_dd, mu):
    """Return cylindrical state and control for scalar CasADi expressions."""
    h, p, q, theta = [primitives[i] for i in range(4)]
    hd, pd, qd, theta_d = [primitive_d[i] for i in range(4)]
    hdd, pdd, qdd, theta_dd = [primitive_dd[i] for i in range(4)]
    s = p**2 - q**2
    sd = 2 * (p * pd - q * qd)
    sdd = 2 * (pd**2 + p * pdd - qd**2 - q * qdd)
    cross = 2 * p * q
    cross_d = 2 * (pd * q + p * qd)
    cross_dd = 2 * (pdd * q + 2 * pd * qd + p * qdd)
    total = p**2 + q**2
    total_d = 2 * (p * pd + q * qd)
    total_dd = 2 * (pd**2 + p * pdd + qd**2 + q * qdd)
    rho = h * s
    rho_d = hd * s + h * sd
    rho_dd = hdd * s + 2 * hd * sd + h * sdd
    z = h * cross
    z_d = hd * cross + h * cross_d
    z_dd = hdd * cross + 2 * hd * cross_d + h * cross_dd
    radius = h * total
    radius_d = hd * total + h * total_d
    radius_dd = hdd * total + 2 * hd * total_d + h * total_dd
    u_rho = rho_dd - rho * theta_d**2 + mu * rho / radius**3
    u_theta = rho * theta_dd + 2 * rho_d * theta_d
    u_z = z_dd + mu * z / radius**3
    return (rho, theta, z, rho_d, theta_d, z_d, radius, radius_d, radius_dd,
            u_rho, u_theta, u_z)


def evaluate_dense_at_tau(basis, controls, tof, mu, tau):
    tau = np.asarray(tau, dtype=float)
    primitive = basis(tau) @ controls
    primitive_d = basis(tau, 1) @ controls / tof
    primitive_dd = basis(tau, 2) @ controls / tof**2
    columns = [[] for _ in range(12)]
    for row, row_d, row_dd in zip(primitive, primitive_d, primitive_dd):
        values = physical_state(row, row_d, row_dd, mu)
        for column, value in zip(columns, values):
            column.append(float(value))
    arrays = [np.asarray(column) for column in columns]
    return tau, primitive, primitive_d, primitive_dd, arrays


def evaluate_dense(basis, controls, tof, mu, samples):
    return evaluate_dense_at_tau(
        basis, controls, tof, mu, np.linspace(0.0, 1.0, samples))


def write_single_row_csv(path, row):
    """Write one self-describing result row with stable column order."""
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def write_shaping_artifacts(args, basis, controls, initial_controls, summary):
    """Save controls and scalar metrics in the original shaping style."""
    spice = summary.get("spice_boundary_data") or {}
    departure = float(spice.get("departure_mjd2000", np.nan))
    tof_days = float(spice.get(
        "tof_days", args.tof * DAYS_PER_CANONICAL_TIME))
    success = summary["status"] in {"Solve_Succeeded", "Solved_To_Acceptable_Level"}

    attempt = {
        "departure_mjd2000": departure,
        "tof_days": tof_days,
        "N": int(args.n_rev),
        "degree": int(args.degree),
        "n_ctrl": int(args.n_ctrl),
        "n_spans": int(args.n_ctrl - args.degree),
        "delta_v_km_s": summary["delta_v_reference_km_s"],
        "delta_v_reference_error_km_s": summary["delta_v_reference_error_km_s"],
        "u_max_reference_m_s2": summary["u_max_reference_m_s2"],
        "u_max_reference_error_m_s2": summary["u_max_reference_error_m_s2"],
        "reference_quadrature_order": summary["reference_quadrature_order"],
        "reference_evaluations": summary["reference_evaluations"],
        "reference_converged": summary["reference_converged"],
        "reference_metric_version": summary["reference_metric_version"],
        "required_peak_acceleration_mm_s2": summary["required_peak_acceleration_mm_s2"],
        "continuous_bernstein_bound_mm_s2":
            summary["continuous_bernstein_bound_for_returned_trajectory_mm_s2"],
        "sampled_peak_acceleration_mm_s2": summary["sampled_peak_acceleration_mm_s2"],
        "success": success,
        "usable": bool(summary["numerical_checks_passed"]),
        "message": summary["status"],
        "max_endpoint_position_error": summary["max_endpoint_position_error"],
        "max_endpoint_velocity_error": summary["max_endpoint_velocity_error"],
        "max_pythagorean_identity_error": summary["max_pythagorean_identity_error"],
        "minimum_normalized_bernstein_coefficient":
            summary["minimum_normalized_bernstein_coefficient"],
        "wall_time_s": summary["wall_time_s"],
        "ipopt_iterations": summary["ipopt_iterations"],
        "number_of_free_trajectory_variables":
            summary["number_of_free_trajectory_variables"],
        "number_of_decision_variables": summary["number_of_decision_variables"],
        "number_of_thrust_inequalities": summary["number_of_thrust_inequalities"],
        "thrust_enforcement": summary["thrust_enforcement"],
        "ephemeris_source": args.ephemeris,
        "spice_target_name": args.spice_target_name if args.ephemeris == "spice" else "",
    }
    write_single_row_csv(args.output / "pythagorean_3d_attempts.csv", attempt)

    comparison = {
        "method": f"pythagorean_3d_nctrl{args.n_ctrl}_deg{args.degree}",
        "finite_points": 1 if np.isfinite(attempt["delta_v_km_s"]) else 0,
        "min_delta_v_km_s": attempt["delta_v_km_s"],
        "median_delta_v_km_s": attempt["delta_v_km_s"],
        "max_delta_v_km_s": attempt["delta_v_km_s"],
    }
    write_single_row_csv(args.output / "comparison_summary.csv", comparison)

    np.savez(
        args.output / "pythagorean_3d_control_points.npz",
        control_points=np.asarray(controls, dtype=float)[None, :, :],
        control_points_initial=np.asarray(initial_controls, dtype=float)[None, :, :],
        departure_mjd2000=np.asarray([departure], dtype=float),
        tof_days=np.asarray([tof_days], dtype=float),
        N=np.asarray([args.n_rev], dtype=int),
        success=np.asarray([success], dtype=bool),
        usable=np.asarray([summary["numerical_checks_passed"]], dtype=bool),
        delta_v_km_s=np.asarray([summary["delta_v_reference_km_s"]], dtype=float),
        delta_v_reference_km_s=np.asarray(
            [summary["delta_v_reference_km_s"]], dtype=float),
        delta_v_reference_error_km_s=np.asarray(
            [summary["delta_v_reference_error_km_s"]], dtype=float),
        u_max_reference_m_s2=np.asarray(
            [summary["u_max_reference_m_s2"]], dtype=float),
        u_max_reference_error_m_s2=np.asarray(
            [summary["u_max_reference_error_m_s2"]], dtype=float),
        reference_quadrature_order=np.asarray(
            [summary["reference_quadrature_order"]], dtype=int),
        reference_evaluations=np.asarray(
            [summary["reference_evaluations"]], dtype=int),
        reference_converged=np.asarray(
            [summary["reference_converged"]], dtype=bool),
        reference_metric_version=np.asarray([summary["reference_metric_version"]]),
        wall_time_s=np.asarray([summary["wall_time_s"]], dtype=float),
        ipopt_iterations=np.asarray([summary["ipopt_iterations"]], dtype=int),
        required_peak_acceleration=np.asarray(
            [summary["required_peak_acceleration_bound"]], dtype=float),
        sampled_peak_acceleration=np.asarray(
            [summary["sampled_peak_acceleration"]], dtype=float),
        n_ctrl_requested=np.asarray(args.n_ctrl, dtype=int),
        n_ctrl_actual=np.asarray(np.asarray(controls).shape[0], dtype=int),
        degree=np.asarray(args.degree, dtype=int),
        knots=np.asarray(basis.t, dtype=float),
        tau_start=np.asarray(0.0, dtype=float),
        tau_end=np.asarray(1.0, dtype=float),
        control_point_coordinates=np.asarray("pythagorean_primitives"),
        control_point_components=np.asarray(["h", "p", "q", "theta"]),
        objective=np.asarray(args.mode),
        thrust_enforcement=np.asarray(args.thrust_enforcement),
        ephemeris_source=np.asarray(args.ephemeris),
        spice_meta_kernel=np.asarray(
            str(spice.get("meta_kernel", "")) if args.ephemeris == "spice" else ""),
        spice_target_name=np.asarray(
            args.spice_target_name if args.ephemeris == "spice" else ""),
        config_json=np.asarray(json.dumps(summary["parameters"], sort_keys=True)),
    )


def seed_bernstein_peak_squared(extractions, seed, certificate, subdivisions,
                                span_duration, mu):
    """Smallest squared peak whose Bernstein certificate holds for the seed.

    Used as the reference scale of the min-peak problem, which has no
    externally supplied acceleration limit to normalize with.
    """
    bound = 0.0
    for extraction in extractions:
        bezier = extraction @ seed
        result = certificate(bezier[:, 0], bezier[:, 1], bezier[:, 2], bezier[:, 3],
                             span_duration, mu, 0.0)
        p_coeff = np.asarray(result[0]).reshape(-1)
        radius6 = np.asarray(result[3]).reshape(-1)
        for matrix in subdivisions:
            slopes = matrix @ radius6
            if np.all(slopes > 0):
                bound = max(bound, float(np.max(-(matrix @ p_coeff) / slopes)))
    return bound


def peak_limit_label(args):
    """Legend wording for the acceleration line drawn in each mode."""
    if args.mode == "energy":
        return r"available $u_{max}$"
    return ("node-enforced peak" if args.thrust_enforcement == "grid"
            else "certified peak")


def plot_solution(args, time, state, control, limit, delta_v_km_s):
    import matplotlib
    if not args.show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho, theta, z = state
    ur, ut, uz, norm = control
    ur_mm_s2 = CANONICAL_ACCEL_TO_MM_S2 * ur
    ut_mm_s2 = CANONICAL_ACCEL_TO_MM_S2 * ut
    uz_mm_s2 = CANONICAL_ACCEL_TO_MM_S2 * uz
    norm_mm_s2 = CANONICAL_ACCEL_TO_MM_S2 * norm
    limit_mm_s2 = CANONICAL_ACCEL_TO_MM_S2 * limit
    x, y = rho * np.cos(theta), rho * np.sin(theta)

    def circular_arc(radius, inclination, endpoint_longitude, anchor_time):
        """Circular orbit phased through an endpoint at maximum latitude."""
        mean_motion = np.sqrt(args.mu / radius**3)
        e1 = np.array([np.sin(endpoint_longitude), -np.cos(endpoint_longitude), 0.0])
        e2 = np.array([np.cos(inclination) * np.cos(endpoint_longitude),
                       np.cos(inclination) * np.sin(endpoint_longitude),
                       np.sin(inclination)])
        phase = np.pi / 2 + mean_motion * (time - anchor_time)
        return radius * (np.cos(phase)[:, None] * e1
                         + np.sin(phase)[:, None] * e2)

    if args.ephemeris == "spice":
        earth_arc, target_arc = spice_plot_arcs(args)
        earth_label = "Earth SPICE arc"
        target_label = f"{args.spice_target_name} SPICE arc"
    else:
        earth_arc = circular_arc(args.r0, args.phi0, 0.0, 0.0)
        target_arc = circular_arc(args.rf, args.phif, args.theta_final, args.tof)
        earth_label = "Earth circular-orbit arc"
        target_label = "Target circular-orbit arc"
    with plt.rc_context({"axes.grid": True, "grid.alpha": 0.2}):
        fig = plt.figure(figsize=(9, 7), constrained_layout=True)
        ax = fig.add_subplot(projection="3d")
        ax.plot(x, y, z, lw=2.4, color="C0", label="Spacecraft")
        ax.plot(*earth_arc.T, color="C2", ls="--", lw=1.7,
                label=earth_label)
        ax.plot(*target_arc.T, color="C1", ls="--", lw=1.7,
                label=target_label)
        ax.scatter([0], [0], [0], marker="*", s=180, color="gold", edgecolor="black",
                   label="Sun", zorder=6)
        ax.scatter(*earth_arc[0], facecolors="none", edgecolors="C2", s=70,
                   linewidths=2, label="Earth at departure", zorder=7)
        ax.scatter(*target_arc[-1], color="C1", marker="s", s=65,
                   label="Target at arrival", zorder=7)
        all_points = np.vstack((np.column_stack((x, y, z)), earth_arc, target_arc))
        extent = np.ptp(all_points, axis=0)
        horizontal_extent = max(extent[0], extent[1], 1e-12)
        z_expanded = extent[2] < 0.15 * horizontal_extent
        display_extent = extent.copy()
        if z_expanded:
            display_extent[2] = 0.3 * horizontal_extent
        ax.set_box_aspect(np.maximum(display_extent, 1e-12))
        scale_note = " (vertical display scale expanded)" if z_expanded else ""
        ax.set(xlabel="x", ylabel="y", zlabel="z",
               title="Pythagorean 3D B-spline transfer and endpoint orbit arcs" + scale_note)
        ax.legend(fontsize=8)
        fig.savefig(args.output / "trajectory_3d.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        ax.plot(time, ur_mm_s2, "--", label=r"$u_\rho$")
        ax.plot(time, ut_mm_s2, "--", label=r"$u_\theta$")
        ax.plot(time, uz_mm_s2, "--", label=r"$u_z$")
        ax.plot(time, norm_mm_s2, color="black", lw=2.4, label=r"$\|u\|$")
        ax.axhline(limit_mm_s2, color="C4", ls=":",
                   label=rf"{peak_limit_label(args)} $={limit_mm_s2:.4g}$ mm/s$^2$")
        ax.set(xlabel="Time [canonical]", ylabel=r"Acceleration [mm/s$^2$]",
               title="3D control history")
        ax.legend(ncol=2)
        fig.savefig(args.output / "controls_3d.png", dpi=180)
        plt.close(fig)

        fig = plt.figure(figsize=(15.0, 5.4), constrained_layout=True)
        grid = fig.add_gridspec(1, 2, width_ratios=(1.15, 1.0))
        trajectory_ax = fig.add_subplot(grid[0, 0], projection="3d")
        control_ax = fig.add_subplot(grid[0, 1])

        trajectory_ax.plot(
            *earth_arc.T, color="tab:cyan", ls="--", lw=1.4, alpha=0.85,
            label=earth_label,
        )
        trajectory_ax.plot(
            *target_arc.T, color="tab:red", ls="--", lw=1.4, alpha=0.8,
            label=target_label,
        )
        trajectory_ax.plot(
            x, y, z, color="tab:blue", lw=2.3,
            label="Pythagorean B-spline trajectory",
        )
        trajectory_ax.scatter(
            [0.0], [0.0], [0.0], s=90, color="#f4d03f",
            edgecolor="black", label="Sun", zorder=6,
        )
        trajectory_ax.scatter(
            *np.array([x[0], y[0], z[0]]), s=55, color="tab:green",
            marker="o", label="Departure", zorder=7,
        )
        trajectory_ax.scatter(
            *np.array([x[-1], y[-1], z[-1]]), s=60, color="tab:red",
            marker="s", label="Rendezvous", zorder=7,
        )
        lower = np.nanmin(all_points, axis=0)
        upper = np.nanmax(all_points, axis=0)
        center = 0.5 * (lower + upper)
        axis_radius = max(0.5 * float(np.max(upper - lower)), 0.1)
        trajectory_ax.set_xlim(center[0] - axis_radius, center[0] + axis_radius)
        trajectory_ax.set_ylim(center[1] - axis_radius, center[1] + axis_radius)
        trajectory_ax.set_zlim(center[2] - axis_radius, center[2] + axis_radius)
        trajectory_ax.set_box_aspect((1.0, 1.0, 1.0))
        trajectory_ax.set_xlabel("x [AU]")
        trajectory_ax.set_ylabel("y [AU]")
        trajectory_ax.set_zlabel("z [AU]")
        trajectory_ax.set_title("Pythagorean B-spline trajectory")
        trajectory_ax.view_init(elev=24.0, azim=-58.0)
        trajectory_ax.legend(loc="upper left", fontsize=8.5)

        time_years = np.asarray(time, dtype=float) * UT_YR
        styles = (
            (ur_mm_s2, r"$u_\rho$", "#d55e00"),
            (ut_mm_s2, r"$u_\theta$", "#0072b2"),
            (uz_mm_s2, r"$u_z$", "#009e73"),
        )
        for component, label, color in styles:
            control_ax.plot(
                time_years, component, "--", color=color, lw=1.7, label=label)
        control_ax.plot(
            time_years, norm_mm_s2, color="0.15", lw=2.3,
            label=r"$\|\mathbf{u}\|$",
        )
        control_ax.axhline(
            limit_mm_s2, color="C4", ls=":", lw=1.7,
            label=rf"{peak_limit_label(args)} $={limit_mm_s2:.4g}$ mm/s$^2$",
        )
        control_ax.axhline(0.0, color="0.75", lw=0.8)
        control_ax.set_xlim(0.0, float(time_years[-1]))
        control_ax.set_xlabel("Time [yr]")
        control_ax.set_ylabel(r"Control acceleration [mm/s$^2$]")
        control_ax.set_title("Control history")
        control_ax.legend(ncol=2, fontsize=9)

        if args.ephemeris == "spice":
            departure = datetime(2000, 1, 1, 12) + timedelta(
                days=float(args._spice_context["departure_mjd2000"]))
            title = (
                f"Departure: {departure:%Y-%m-%d}   "
                f"TOF: {float(args.tof_days):.1f} d   N: {int(args.n_rev)}   "
                rf"$\Delta V$: {delta_v_km_s:.6f} km/s"
            )
        else:
            degree_name = {2: "Quadratic", 3: "Cubic"}.get(
                int(args.degree), f"Degree-{int(args.degree)}")
            title = (
                f"{degree_name} Pythagorean B-spline   "
                rf"$\Delta V$: {delta_v_km_s:.6f} km/s"
            )
        fig.suptitle(title, fontsize=14)
        fig.savefig(args.output / "pythagorean_3d_solution.png", dpi=220)
        fig.savefig(args.output / "pythagorean_3d_solution.pdf")
        if args.show_plots:
            plt.show()
        plt.close(fig)


def solve(args):
    solve_started = perf_counter()
    basis = spline_basis(args.n_ctrl, args.degree)
    spans = args.n_ctrl - args.degree
    values, rates, endpoint_state, endpoint_rates = primitive_endpoint_data(args)
    seed = initial_primitive_coefficients(basis, values, rates, args.tof)
    if getattr(args, "initial_solution", None) is not None:
        payload = json.loads(Path(args.initial_solution).read_text())
        loaded_seed = np.asarray(payload["control_coefficients_h_p_q_theta"], dtype=float)
        if loaded_seed.shape != (args.n_ctrl, 4) or not np.all(np.isfinite(loaded_seed)):
            raise ValueError("initial-solution has incompatible Pythagorean control points")
        seed = loaded_seed
    certificate, certificate_degree = pythagorean_certificate_function(args.degree)
    certificate_subdivisions = subdivision_matrices(certificate_degree, args.subdivide)
    geometry_degree = 3 * args.degree
    geometry_subdivisions = subdivision_matrices(geometry_degree, args.subdivide)

    opti = ca.Opti()
    controls, free, projected_seed, n_free = build_clamped_primitives(
        opti, basis, values, rates, args.tof, seed)
    if free is not None:
        opti.subject_to(free[:, 0] >= args.h_min)

    minimize_peak = args.mode == "min-peak"
    thrust_enforcement = getattr(args, "thrust_enforcement", "bernstein")
    nodes_per_span = getattr(args, "constraint_nodes_per_span", 15)
    extractions = [
        extraction_matrix(basis, span / spans, (span + 1) / spans, args.degree)
        for span in range(spans)
    ]
    if minimize_peak:
        # No available acceleration limit exists in this mode, so the squared
        # peak variable and the certificate coefficients are made dimensionless
        # with the Bernstein peak bound of the seed trajectory instead.
        seed_peak_squared = seed_bernstein_peak_squared(
            extractions, projected_seed, certificate, certificate_subdivisions,
            args.tof / spans, args.mu)
        reference_squared = (
            seed_peak_squared
            if np.isfinite(seed_peak_squared) and seed_peak_squared > 0 else 1.0)
        peak_ratio_squared = opti.variable()
        opti.subject_to(peak_ratio_squared >= 0)
        peak_squared = reference_squared * peak_ratio_squared
    else:
        seed_peak_squared = float("nan")
        reference_squared = args.u_max**2
        peak_ratio_squared = None
        peak_squared = reference_squared

    scale_p = reference_squared * max(args.r0, args.rf)**6
    coefficient_blocks = []
    grid_constraint_tau = []
    energy = 0
    if minimize_peak:
        nodes, weights = np.empty(0), np.empty(0)
    else:
        nodes, weights = np.polynomial.legendre.leggauss(args.quadrature)
    for span in range(spans):
        left, right = span / spans, (span + 1) / spans
        extraction = extractions[span]
        bezier = ca.DM(extraction) @ controls
        result = certificate(bezier[:, 0], bezier[:, 1], bezier[:, 2], bezier[:, 3],
                             args.tof / spans, args.mu, peak_squared)
        p_coeff = result[0]
        rho_coeff, radius_coeff = result[1], result[2]
        for matrix in certificate_subdivisions:
            block = ca.DM(matrix) @ p_coeff / scale_p
            if thrust_enforcement == "bernstein":
                opti.subject_to(block >= 0)
            coefficient_blocks.append(block)
        for matrix in geometry_subdivisions:
            opti.subject_to(ca.DM(matrix) @ rho_coeff >= args.rho_min)
            opti.subject_to(ca.DM(matrix) @ radius_coeff >= args.r_min)

        if thrust_enforcement == "grid":
            span_nodes = np.linspace(left, right, nodes_per_span)
            # A degree-2 spline is C1, so acceleration may jump at a knot.
            # Evaluate the left limit of each internal right endpoint; the
            # next span's left endpoint supplies the corresponding right limit.
            if span < spans - 1:
                span_nodes[-1] = np.nextafter(right, left)
            for tau in span_nodes:
                primitive = ca.DM(basis(tau).reshape(1, -1)) @ controls
                primitive_d = ca.DM(basis(tau, 1).reshape(1, -1)) @ controls / args.tof
                primitive_dd = ca.DM(basis(tau, 2).reshape(1, -1)) @ controls / args.tof**2
                physical = physical_state(primitive[0, :], primitive_d[0, :],
                                          primitive_dd[0, :], args.mu)
                ur_grid, ut_grid, uz_grid = physical[-3:]
                opti.subject_to(ur_grid**2 + ut_grid**2 + uz_grid**2 <= peak_squared)
                grid_constraint_tau.append(float(tau))

        for node, weight in zip(nodes, weights):
            tau = left + 0.5 * (node + 1) / spans
            primitive = ca.DM(basis(tau).reshape(1, -1)) @ controls
            primitive_d = ca.DM(basis(tau, 1).reshape(1, -1)) @ controls / args.tof
            primitive_dd = ca.DM(basis(tau, 2).reshape(1, -1)) @ controls / args.tof**2
            physical = physical_state(primitive[0, :], primitive_d[0, :],
                                      primitive_dd[0, :], args.mu)
            ur, ut, uz = physical[-3:]
            energy += weight * args.tof / (2 * spans) * (ur**2 + ut**2 + uz**2)

    if minimize_peak:
        opti.minimize(peak_ratio_squared)
        opti.set_initial(peak_ratio_squared,
                         float(getattr(args, "peak_init_factor", 1.05)))
    else:
        opti.minimize(energy / (reference_squared * args.tof))

    opti.solver("ipopt", {"print_time": not args.quiet}, {
        "max_iter": args.max_iter,
        "tol": 1e-9,
        "constr_viol_tol": 1e-9,
        "acceptable_iter": 0,
        "bound_relax_factor": 0,
        "print_level": 0 if args.quiet else 5,
    })
    thrust_count = ((certificate_degree + 1) * spans * 2**args.subdivide
                    if thrust_enforcement == "bernstein" else spans * nodes_per_span)
    print(f"Degree-{args.degree} Pythagorean 3D B-spline: {args.n_ctrl} controls, "
          f"{spans} spans, {4*n_free} free trajectory variables, "
          f"{thrust_count} {thrust_enforcement} thrust inequalities.")
    solution = opti.solve()
    solver_stats = solution.stats()
    solved_controls = np.asarray(solution.value(controls))
    w_value = float(solution.value(peak_squared)) if minimize_peak else reference_squared
    limit = float(np.sqrt(max(0.0, w_value)))
    p_values = np.asarray(solution.value(ca.vertcat(*coefficient_blocks))).reshape(-1)

    continuous_bound_squared = 0.0
    for extraction in extractions:
        bezier = extraction @ solved_controls
        zero_result = certificate(bezier[:, 0], bezier[:, 1], bezier[:, 2], bezier[:, 3],
                                  args.tof / spans, args.mu, 0.0)
        zero_p = np.asarray(zero_result[0]).reshape(-1)
        radius6 = np.asarray(zero_result[3]).reshape(-1)
        for matrix in certificate_subdivisions:
            denominator = matrix @ radius6
            if np.any(denominator <= 0):
                continuous_bound_squared = float("inf")
            elif np.isfinite(continuous_bound_squared):
                continuous_bound_squared = max(
                    continuous_bound_squared,
                    float(np.max(-(matrix @ zero_p) / denominator)),
                )
    continuous_bound = float(np.sqrt(max(0.0, continuous_bound_squared)))

    tau, primitive, primitive_d, primitive_dd, physical = evaluate_dense(
        basis, solved_controls, args.tof, args.mu, args.samples)
    (rho, theta, z, rho_d, theta_d, z_d, radius, radius_d, radius_dd,
     ur, ut, uz) = physical
    norm = np.sqrt(ur**2 + ut**2 + uz**2)
    if grid_constraint_tau:
        unique_grid_tau = np.unique(np.asarray(grid_constraint_tau))
        _, _, _, _, grid_physical = evaluate_dense_at_tau(
            basis, solved_controls, args.tof, args.mu, unique_grid_tau)
        grid_norm = np.sqrt(grid_physical[-3]**2 + grid_physical[-2]**2
                            + grid_physical[-1]**2)
        grid_peak = float(np.max(grid_norm))
    else:
        unique_grid_tau = np.empty(0)
        grid_peak = float("nan")
    identity_error = float(np.max(np.abs(radius**2 - rho**2 - z**2)))
    endpoint_position_error = float(np.max(np.abs(
        np.column_stack((rho[[0, -1]], theta[[0, -1]], z[[0, -1]])) - endpoint_state)))
    endpoint_velocity_error = float(np.max(np.abs(
        np.column_stack((rho_d[[0, -1]], theta_d[[0, -1]], z_d[[0, -1]]))
        - endpoint_rates)))
    thrust_checks_passed = bool(
        (thrust_enforcement == "bernstein" and np.min(p_values) >= -1e-7
         and np.max(norm) <= limit + 1e-7 * max(1.0, limit))
        or (thrust_enforcement == "grid"
            and grid_peak <= limit + 1e-7 * max(1.0, limit))
    )
    dense_limit_checks_passed = bool(
        np.max(norm) <= limit + 1e-7 * max(1.0, limit))
    checks_passed = bool(
        np.all(np.isfinite(solved_controls))
        and np.min(rho) >= args.rho_min - 1e-8
        and np.min(radius) >= args.r_min - 1e-8
        and thrust_checks_passed
        and identity_error <= 1e-10
        and endpoint_position_error <= 1e-7
        and endpoint_velocity_error <= 1e-7
    )
    delta_v = float(np.trapezoid(norm, tau * args.tof))

    def acceleration_norm(sample_tau):
        _, _, _, _, sample_physical = evaluate_dense_at_tau(
            basis, solved_controls, args.tof, args.mu, sample_tau)
        return np.sqrt(
            sample_physical[-3]**2
            + sample_physical[-2]**2
            + sample_physical[-1]**2
        )

    reference_metrics = evaluate_reference_metrics(
        acceleration_norm,
        delta_v_scale_km_s=args.tof * CANONICAL_DV_TO_KM_S,
        acceleration_scale_m_s2=CANONICAL_ACCEL_TO_M_S2,
    ).as_dict()
    wall_time = perf_counter() - solve_started

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": solver_stats["return_status"],
        "formulation": "3D cylindrical Pythagorean Bernstein certificate",
        "parameters": {**{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items() if not key.startswith("_")
        }, "output": str(args.output)},
        "spice_boundary_data": getattr(args, "_spice_context", None),
        "certificate_degree": certificate_degree,
        "thrust_enforcement": thrust_enforcement,
        "constraint_nodes_per_span": nodes_per_span if thrust_enforcement == "grid" else None,
        "number_of_free_trajectory_variables": 4 * n_free,
        "number_of_decision_variables": 4 * n_free + int(minimize_peak),
        "number_of_thrust_inequalities": thrust_count,
        "required_peak_acceleration_bound": limit,
        "required_peak_acceleration_mm_s2": limit * CANONICAL_ACCEL_TO_MM_S2,
        "peak_reference_scale": float(np.sqrt(reference_squared)),
        "seed_bernstein_peak_bound": (
            float(np.sqrt(seed_peak_squared)) if minimize_peak else None),
        "continuous_bernstein_bound_for_returned_trajectory": continuous_bound,
        "continuous_bernstein_bound_for_returned_trajectory_mm_s2":
            continuous_bound * CANONICAL_ACCEL_TO_MM_S2,
        "minimum_normalized_bernstein_coefficient": float(np.min(p_values)),
        "sampled_peak_acceleration": float(np.max(norm)),
        "sampled_peak_acceleration_mm_s2":
            float(np.max(norm)) * CANONICAL_ACCEL_TO_MM_S2,
        "grid_node_peak_acceleration": grid_peak if thrust_enforcement == "grid" else None,
        "grid_node_peak_acceleration_mm_s2":
            grid_peak * CANONICAL_ACCEL_TO_MM_S2 if thrust_enforcement == "grid" else None,
        "dense_peak_over_grid_peak_percent":
            100.0 * (float(np.max(norm)) / grid_peak - 1.0)
            if thrust_enforcement == "grid" else None,
        "bernstein_bound_over_grid_peak_percent":
            100.0 * (continuous_bound / grid_peak - 1.0)
            if thrust_enforcement == "grid" else None,
        "dense_limit_checks_passed": dense_limit_checks_passed,
        "delta_v_nondimensional": delta_v,
        **reference_metrics,
        "wall_time_s": wall_time,
        "ipopt_iterations": int(solver_stats.get("iter_count", -1)),
        "max_pythagorean_identity_error": identity_error,
        "max_endpoint_position_error": endpoint_position_error,
        "max_endpoint_velocity_error": endpoint_velocity_error,
        "numerical_checks_passed": checks_passed,
        "formal_interval_certificate": False,
        "control_coefficients_h_p_q_theta": solved_controls.tolist(),
    }
    (args.output / "solution.json").write_text(json.dumps(summary, indent=2) + "\n")
    x, y = rho * np.cos(theta), rho * np.sin(theta)
    table = np.column_stack((tau * args.tof, rho, theta, z, radius, rho_d,
                             theta_d, z_d, ur, ut, uz, norm, x, y))
    np.savetxt(args.output / "trajectory.csv", table, delimiter=",", comments="",
               header="time,rho,theta,z,R,rho_dot,theta_dot,z_dot,u_rho,u_theta,u_z,u_norm,x,y")
    write_shaping_artifacts(
        args, basis, solved_controls, projected_seed, summary)

    print(f"IPOPT: {summary['status']}")
    print(f"Delta V (reference): {summary['delta_v_reference_km_s']:.9g} km/s")
    print(f"Required peak acceleration bound: "
          f"{limit * CANONICAL_ACCEL_TO_MM_S2:.9g} mm/s^2")
    print(f"Sampled peak acceleration: "
          f"{np.max(norm) * CANONICAL_ACCEL_TO_MM_S2:.9g} mm/s^2")
    if thrust_enforcement == "grid":
        print(f"Grid-node peak acceleration: "
              f"{grid_peak * CANONICAL_ACCEL_TO_MM_S2:.9g} mm/s^2")
        print(f"Bernstein bound required by the returned grid trajectory: "
              f"{continuous_bound * CANONICAL_ACCEL_TO_MM_S2:.9g} mm/s^2")
        print(f"Dense inter-node excess: "
              f"{100.0 * (np.max(norm) / grid_peak - 1.0):.4f}%")
        print(f"Bernstein bound excess for this trajectory: "
              f"{100.0 * (continuous_bound / grid_peak - 1.0):.4f}%")
    print(f"Pythagorean identity max error: {identity_error:.3e}")
    print(f"Endpoint errors: position={endpoint_position_error:.3e}, "
          f"velocity={endpoint_velocity_error:.3e}")
    print(f"Minimum normalized Bernstein coefficient: {np.min(p_values):.3e}")
    print(f"Numerical checks passed: {checks_passed}")
    print(f"Results: {args.output.resolve()}")
    if not args.no_plots:
        plot_solution(
            args,
            tau * args.tof,
            (rho, theta, z),
            (ur, ut, uz, norm),
            limit,
            summary["delta_v_reference_km_s"],
        )
    if not checks_passed:
        raise RuntimeError("Post-solve numerical checks failed; inspect solution.json")
    return summary


def main(default_thrust_enforcement="bernstein"):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("min-peak", "energy"), default="min-peak")
    parser.add_argument("--thrust-enforcement", choices=("bernstein", "grid"),
                        default=default_thrust_enforcement)
    parser.add_argument("--constraint-nodes-per-span", type=int, default=15,
                        help="Grid mode: equally spaced thrust nodes on each spline span")
    parser.add_argument("--ephemeris", choices=("circular", "spice"), default="circular")
    parser.add_argument("--departure-date", type=departure_epoch, default="2028-01-01",
                        help="SPICE mode: MJD2000 UTC days or YYYY-MM-DD")
    parser.add_argument("--tof-days", type=float, default=1000.0,
                        help="SPICE mode transfer duration [days]")
    parser.add_argument("--spice-meta-kernel", default="kernels/fasttransfer_planets.tm")
    parser.add_argument("--spice-target-name", default="MARS BARYCENTER",
                        help="SPICE target name or NAIF ID")
    parser.add_argument("--n-rev", type=int, default=0,
                        help="Additional complete turns in the unwrapped final longitude")
    parser.add_argument("--orbit-arc-samples", type=int, default=401)
    parser.add_argument(
        "--degree", type=int, default=2,
        help="spline degree; n-ctrl == degree + 1 gives a single Bezier segment",
    )
    parser.add_argument("--n-ctrl", type=int, default=5)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--r0", type=float, default=1.0)
    parser.add_argument("--rf", type=float, default=1.5)
    parser.add_argument("--phi0", type=float, default=0.0,
                        help="Initial latitude [rad]")
    parser.add_argument("--phif", type=float, default=0.25,
                        help="Final latitude [rad]")
    parser.add_argument("--rdot0", type=float, default=0.0)
    parser.add_argument("--rdotf", type=float, default=0.0)
    parser.add_argument("--phidot0", type=float, default=0.0)
    parser.add_argument("--phidotf", type=float, default=0.0)
    parser.add_argument("--theta-dot0", type=float, default=None)
    parser.add_argument("--theta-dotf", type=float, default=None)
    parser.add_argument("--theta-final", type=float, default=6.0)
    parser.add_argument("--tof", type=float, default=8.0)
    parser.add_argument(
        "--u-max", type=float, default=None,
        help="energy mode only: fixed acceleration cap; rejected in min-peak mode, "
             "whose acceleration limit is the optimized result",
    )
    parser.add_argument(
        "--peak-init-factor", type=float, default=1.05,
        help="min-peak mode: initial squared peak as a multiple of the seed "
             "trajectory's Bernstein peak bound",
    )
    parser.add_argument("--h-min", type=float, default=0.1)
    parser.add_argument("--rho-min", type=float, default=0.1)
    parser.add_argument("--r-min", type=float, default=0.1)
    parser.add_argument("--subdivide", type=int, default=0)
    parser.add_argument(
        "--quadrature", type=int, default=12,
        help="Gauss-Legendre order for the energy objective; unused in min-peak mode",
    )
    parser.add_argument("--samples", type=int, default=2001)
    parser.add_argument("--max-iter", type=int, default=2500)
    parser.add_argument("--initial-solution", type=Path, default=None,
                        help="Warm-start trajectory controls from a previous solution.json")
    parser.add_argument("--quiet", action="store_true")
    plots = parser.add_mutually_exclusive_group()
    plots.add_argument("--show-plots", action="store_true")
    plots.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("output/pythagorean_3d_bernstein"))
    args = parser.parse_args()
    if args.ephemeris == "spice":
        try:
            configure_spice_case(args)
        except (ValueError, FileNotFoundError, ImportError) as exc:
            parser.error(str(exc))
    if args.mode == "min-peak":
        if args.u_max is not None:
            parser.error("u-max does not apply to min-peak mode, whose acceleration "
                         "limit is the optimized result; omit it")
        if not np.isfinite(args.peak_init_factor) or args.peak_init_factor <= 0:
            parser.error("peak-init-factor must be positive")
    elif args.u_max is None or not np.isfinite(args.u_max) or args.u_max <= 0:
        parser.error("energy mode requires a positive u-max")
    positive = (args.mu, args.r0, args.rf, args.tof,
                args.h_min, args.rho_min, args.r_min)
    if any(not np.isfinite(value) or value <= 0 for value in positive):
        parser.error("mu, radii, tof, and minimum radii must be positive")
    finite = (args.phi0, args.phif, args.rdot0, args.rdotf, args.phidot0,
              args.phidotf, args.theta_final)
    if any(not np.isfinite(value) for value in finite):
        parser.error("endpoint angles and rates must be finite")
    if args.degree < 2:
        parser.error("degree must be at least 2; the certificate needs a "
                     "nonzero second derivative of each primitive spline")
    if args.n_ctrl < max(4, args.degree + 1):
        parser.error("n-ctrl must be at least degree + 1 and leave four control "
                     "points for the endpoint value and velocity conditions")
    if (args.subdivide < 0 or args.samples < 2 or args.quadrature < 1
            or args.max_iter < 1 or args.orbit_arc_samples < 2 or args.n_rev < 0
            or args.constraint_nodes_per_span < 2):
        parser.error("invalid subdivision, sampling, quadrature, or iteration count")
    endpoint_rho = min(args.r0 * np.cos(args.phi0), args.rf * np.cos(args.phif))
    if endpoint_rho <= 0 or args.rho_min > endpoint_rho:
        parser.error("endpoint cylindrical radii must be positive and above rho-min")
    solve(args)


if __name__ == "__main__":
    main()

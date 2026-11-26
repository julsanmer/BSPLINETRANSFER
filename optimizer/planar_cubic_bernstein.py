"""Planar cubic/quintic B-spline shaping with continuous Bernstein thrust constraints.

Run from the repository root:
    python -m optimizer.planar_cubic_bernstein --mode min-peak --degree 3 --n-ctrl 8 --u-max 0.2

The default min-peak mode minimizes a squared peak-acceleration bound w and
compares sqrt(w) with the available --u-max. It does not constrain w <= u_max^2.
Use --mode energy for the original squared-acceleration objective with a hard
u-max bound. Flight time and endpoint states remain fixed in both modes.

Plots are saved alongside the numerical results (requires matplotlib).
Add --show-plots to also open the figures, or --no-plots to skip plotting.

All quantities use consistent nondimensional units (mu=1 by default). This is
an illustrative circular-orbit transfer, not an ephemeris-based rendezvous.
The endpoint angles and flight time are fixed; endpoint radial velocities are
zero and angular velocities are those of the endpoint circular orbits.
Clamped endpoint control points are eliminated analytically, following the
clamped branch of build_reduced_control_points in optimization_Bspline_freetf.py.
Only the interior control points are optimization variables.

For each span, in physical time, define
    a = rho_ddot - rho * theta_dot**2
    b = rho * theta_ddot + 2 * rho_dot * theta_dot
    P = w * rho**4 - (rho**2 * a + mu)**2 - rho**4 * b**2.
For rho>0 and w>=0, P>=0 is equivalent to ||u||<=sqrt(w). For spline degree p,
the 10*p-3 Bernstein coefficients of the degree-(10*p-4) polynomial P are
constrained to be nonnegative on each span. Optional
certificate subdivision reduces conservatism without changing the spline.

CasADi/IPOPT solves a nonconvex NLP. Coefficient checks after the solve use
floating point, not outward-rounded intervals: they are NOT a formal numerical
certificate.
An optimized peak bound above u-max does not prove infeasibility: IPOPT is
local, and the Bernstein coefficient conditions are sufficient, not necessary.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.interpolate import BSpline

from experiments.reference_metrics import evaluate_reference_metrics
from optimizer.canonical_units import UA_AU_PER_YR2, UV_AU_PER_YR
from optimizer.helpers_Bspline import make_clamped_knots


AU_KM = 149_597_870.7
YEAR_S = 365.25 * 86_400.0
CANONICAL_DV_TO_KM_S = UV_AU_PER_YR * AU_KM / YEAR_S
CANONICAL_ACCEL_TO_M_S2 = UA_AU_PER_YR2 * AU_KM * 1000.0 / YEAR_S**2


def elevate(coefficients, degree):
    """Degree-elevate a scalar Bernstein polynomial (numeric or symbolic)."""
    result = list(coefficients)
    if degree < len(result) - 1:
        raise ValueError("Cannot lower the degree by elevation")
    while len(result) <= degree:
        n = len(result) - 1
        result = [result[0]] + [
            i / (n + 1) * result[i - 1] + (1 - i / (n + 1)) * result[i]
            for i in range(1, n + 1)
        ] + [result[-1]]
    return result


def add(left, right, right_scale=1.0):
    degree = max(len(left), len(right)) - 1
    return [a + right_scale * b for a, b in zip(
        elevate(left, degree), elevate(right, degree)
    )]


def multiply(left, right):
    """Exact Bernstein product identity, evaluated in floating point."""
    m, n = len(left) - 1, len(right) - 1
    return [sum(
        comb(m, i) * comb(n, k - i) / comb(m + n, k)
        * left[i] * right[k - i]
        for i in range(max(0, k - n), min(m, k) + 1)
    ) for k in range(m + n + 1)]


def derivative(coefficients, duration):
    degree = len(coefficients) - 1
    return [degree * (coefficients[i + 1] - coefficients[i]) / duration
            for i in range(degree)]


def certificate_function(degree):
    """Map degree-p Bezier coordinates to coefficients of the polynomial P."""
    certificate_degree = 10 * degree - 4
    radial = ca.SX.sym("radial", degree + 1)
    angular = ca.SX.sym("angular", degree + 1)
    duration = ca.SX.sym("duration")
    mu = ca.SX.sym("mu")
    peak_squared = ca.SX.sym("peak_squared")
    r = [radial[i] for i in range(degree + 1)]
    theta = [angular[i] for i in range(degree + 1)]
    rd, td = derivative(r, duration), derivative(theta, duration)
    rdd, tdd = derivative(rd, duration), derivative(td, duration)
    a = add(rdd, multiply(r, multiply(td, td)), -1)
    b = add(multiply(r, tdd), multiply(rd, td), 2)
    r2 = multiply(r, r)
    r4 = multiply(r2, r2)
    radial_numerator = add(multiply(r2, a), [mu])
    p = add([peak_squared * v for v in r4],
            multiply(radial_numerator, radial_numerator), -1)
    p = add(p, multiply(r4, multiply(b, b)), -1)
    return ca.Function(f"planar_certificate_p{degree}",
                       [radial, angular, duration, mu, peak_squared],
                       [ca.vertcat(*elevate(p, certificate_degree))])


def subdivision_matrices(degree, depth):
    """Repeated midpoint de Casteljau subdivision; no trajectory refinement."""
    matrices = [np.eye(degree + 1)]
    for _ in range(depth):
        refined = []
        for matrix in matrices:
            levels = [matrix]
            for _ in range(degree):
                levels.append(0.5 * (levels[-1][:-1] + levels[-1][1:]))
            refined.extend((np.stack([level[0] for level in levels]),
                            np.stack([level[-1] for level in levels[::-1]])))
        matrices = refined
    return matrices


def spline_basis(n_ctrl, degree):
    knots = make_clamped_knots(n_ctrl, degree)
    return BSpline(knots, np.eye(n_ctrl), degree)


def extraction_matrix(basis, left, right, degree):
    """Numerical B-spline-to-Bezier extraction on one nonzero knot span."""
    local = np.linspace(0.0, 1.0, degree + 1)
    bernstein = np.column_stack([
        comb(degree, i) * local**i * (1.0 - local)**(degree - i)
        for i in range(degree + 1)
    ])
    values = basis(left + (right - left) * local)
    return np.linalg.solve(bernstein, values)


def initial_coefficients(basis, r0, rf, angle, omega0, omegaf, tof,
                         rho_dot0=0.0, rho_dotf=0.0):
    """Represent a global cubic Hermite seed in the B-spline basis."""
    knots = basis.t
    degree = int(basis.k)
    # Greville abscissae: mean of the p knots following each basis index.
    # The previous fixed i+1:i+4 slice was valid only for cubic splines and
    # produced a singular collocation matrix for degree five.
    tau = np.array([np.mean(knots[i + 1:i + degree + 1])
                    for i in range(len(basis.c))])
    h00 = 2 * tau**3 - 3 * tau**2 + 1
    h10 = tau**3 - 2 * tau**2 + tau
    h01 = -2 * tau**3 + 3 * tau**2
    h11 = tau**3 - tau**2
    radius = h00 * r0 + h10 * tof * rho_dot0 + h01 * rf + h11 * tof * rho_dotf
    theta = h10 * tof * omega0 + h01 * angle + h11 * tof * omegaf
    return np.linalg.solve(basis(tau), np.column_stack((radius, theta)))


def build_clamped_control_points(opti, basis, boundary_rhs, initial_control_points):
    """Two-coordinate version of the 3D shaping solver's clamped reduction.

    boundary_rhs rows are [q0, T*qdot0, qf, T*qdotf]. Derivative weights
    come from the actual basis, as in the 3D implementation, rather than
    assuming the endpoint knot spacing in the elimination formulas.
    """
    initial = np.asarray(initial_control_points, dtype=float)
    rhs = np.asarray(boundary_rhs, dtype=float)
    n_ctrl = len(basis.c)
    if initial.shape != (n_ctrl, 2) or rhs.shape != (4, 2) or n_ctrl < 4:
        raise ValueError("Expected at least four planar controls and a (4, 2) boundary RHS")
    start_derivative = basis(0.0, 1)
    end_derivative = basis(1.0, 1)
    start_d0, start_d1 = start_derivative[:2]
    end_dprev, end_dlast = end_derivative[-2:]
    c0 = rhs[0:1]
    c1 = (rhs[1:2] - start_d0 * c0) / start_d1
    c_last = rhs[2:3]
    c_prev = (rhs[3:4] - end_dlast * c_last) / end_dprev
    n_free = n_ctrl - 4
    projected_initial = np.vstack((c0, c1, initial[2:-2], c_prev, c_last))
    if n_free:
        free_controls = opti.variable(n_free, 2)
        opti.set_initial(free_controls, initial[2:-2])
        controls = ca.vertcat(ca.DM(c0), ca.DM(c1), free_controls,
                              ca.DM(c_prev), ca.DM(c_last))
    else:
        free_controls = None
        controls = ca.DM(projected_initial)
    return controls, free_controls, projected_initial, n_free


def plot_solution(args, tau, q, ur, ut, p_values, angle, checks_passed, peak_squared,
                  circular_bodies=True):
    """Plot the circular-body scenario and floating-point solution diagnostics."""
    import matplotlib

    if not args.show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = tau * args.tof
    norm = np.hypot(ur, ut)
    spans = args.n_ctrl - args.degree
    certificate_degree = 10 * args.degree - 4
    peak_limit = float(np.sqrt(max(0.0, peak_squared)))
    omega0 = np.sqrt(args.mu / args.r0**3)
    omegaf = np.sqrt(args.mu / args.rf**3)
    # The departure body starts at theta=0. Phase the target so that its
    # position AND circular velocity match the prescribed arrival state.
    body_time = np.linspace(0, args.tof, max(1001, len(time)))
    departure_angle = omega0 * body_time
    target_angle = angle + omegaf * (body_time - args.tof)
    departure = args.r0 * np.column_stack((np.cos(departure_angle),
                                           np.sin(departure_angle)))
    target = args.rf * np.column_stack((np.cos(target_angle), np.sin(target_angle)))
    spacecraft = q[:, 0, None] * np.column_stack((np.cos(q[:, 1]), np.sin(q[:, 1])))
    figures = []

    with plt.rc_context({"font.size": 10, "axes.grid": True, "grid.alpha": 0.2}):
        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
        figures.append((fig, "trajectory.png"))
        ax.plot(*spacecraft.T, color="C0", lw=2, label="Spacecraft transfer", zorder=4)
        if circular_bodies:
            ax.plot(*departure.T, color="C2", ls="--", lw=1.7,
                    label="Departure body arc, 0 to TOF")
            ax.plot(*target.T, color="C1", ls="--", lw=1.7,
                    label="Target body arc, 0 to TOF")
        ax.scatter(0, 0, marker="*", s=180, color="gold", edgecolor="black",
                   label="Central body", zorder=5)
        body_markers = (
            (departure, "C2", ("Departure / body at t=0", "Departure body at TOF")),
            (target, "C1", ("Target body at t=0", "Arrival / target at TOF")),
        ) if circular_bodies else ()
        for path, color, labels in body_markers:
            ax.scatter(*path[0], s=65, facecolors="none", edgecolors=color,
                       linewidths=2, label=labels[0], zorder=6)
            ax.scatter(*path[-1], s=65, color=color, marker="s",
                       label=labels[1], zorder=6)
        if not circular_bodies:
            ax.scatter(*spacecraft[0], color="C2", s=65, label="Departure", zorder=6)
            ax.scatter(*spacecraft[-1], color="C1", marker="s", s=65, label="Arrival", zorder=6)
        paths = [(spacecraft, "C0")]
        if circular_bodies:
            paths.extend(((departure, "C2"), (target, "C1")))
        for path, color in paths:
            index = min(int(0.55 * len(path)), len(path) - 2)
            end = min(index + max(1, len(path) // 40), len(path) - 1)
            ax.annotate("", xy=path[end], xytext=path[index],
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.8})
        ax.set_aspect("equal", adjustable="box")
        subtitle = ("Circular bodies; target phase chosen for rendezvous" if circular_bodies
                    else "Prescribed projected endpoints; frame rotated to departure")
        ax.set(xlabel="x [nondimensional]", ylabel="y [nondimensional]",
               title=f"Planar transfer — TOF={args.tof:g}, {spans} spans, "
                     f"degree {args.degree}, {args.n_ctrl} controls\n{subtitle}")
        ax.legend(loc="best", fontsize=8)

        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        figures.append((fig, "controls.png"))
        # Overlay the signed components and norm, matching the reference figure.
        # The planar problem has no out-of-plane control component.
        ax.plot(time, ur, color="#D55E00", ls="--", lw=2, label=r"$u_\rho$")
        ax.plot(time, ut, color="#0072B2", ls="--", lw=2, label=r"$u_\theta$")
        ax.plot(time, norm, color="#262626", lw=2.7, label=r"$\|\mathbf{u}\|$")
        ax.axhline(args.u_max, color="crimson", ls="-.", lw=1.8,
                   label=rf"$u_{{\max}}={args.u_max:g}$")
        if args.mode == "min-peak":
            ax.axhline(peak_limit, color="C4", ls=":", lw=2,
                       label=rf"$u_{{\mathrm{{required}}}}={peak_limit:.6g}$")
        lower = min(0.0, float(np.min(ur)), float(np.min(ut)))
        upper = max(args.u_max, peak_limit, float(np.max(norm)))
        padding = 0.1 * (upper - lower)
        ax.set_ylim(lower - padding, upper + padding)
        ax.axhspan(args.u_max, upper + padding, color="crimson", alpha=0.06)
        ax.axhline(0, color="0.6", lw=0.6, zorder=0)
        violation = norm > args.u_max
        if np.any(violation):
            ax.scatter(time[violation], norm[violation], s=12, color="crimson",
                       zorder=5, label="Sampled norm bound exceedance")
        title = "Control history"
        if args.mode == "min-peak":
            title += f" — peak-bound gap to available acceleration: {peak_limit - args.u_max:+.6g}"
        ax.set(title=title, xlabel="Time [nondimensional]",
               ylabel="Control acceleration [nondimensional]", xlim=(0, args.tof))
        ax.legend(loc="upper right", ncol=2)

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                 constrained_layout=True)
        figures.append((fig, "constraint_diagnostics.png"))
        axes[0].plot(time, q[:, 0], color="C0", label=r"$\rho(t)$")
        axes[0].axhline(args.rho_min, color="crimson", ls="--", label=r"$\rho_{\min}$")
        axes[0].set_ylabel("Radius [nondimensional]")
        axes[0].legend(loc="best")
        minimum_coefficients = p_values.reshape(-1, certificate_degree + 1).min(axis=1)
        edges = np.linspace(0, args.tof, len(minimum_coefficients) + 1)
        axes[1].stairs(minimum_coefficients, edges, baseline=None, color="C4",
                       label="Minimum normalized Bernstein coefficient per certificate interval")
        axes[1].axhline(0, color="crimson", ls="--", label="Nonnegativity bound")
        # Always normalize by the fixed available u_max, not the optimized w.
        radial_scale = (q[:, 0] / max(args.r0, args.rf))**4
        sampled_p = radial_scale * (peak_squared - norm**2) / args.u_max**2
        axes[1].plot(time, sampled_p, color="C0", lw=1, alpha=0.8,
                     label="Normalized P(t) at enforced peak bound, sampled")
        if args.mode == "min-peak":
            target_p = radial_scale * (1 - (norm / args.u_max)**2)
            axes[1].plot(time, target_p, color="crimson", ls="--", lw=1,
                         label="Normalized P(t) at available u_max, sampled")
        axes[1].set(xlabel="Time [nondimensional]", ylabel="Normalized certificate value",
                    xlim=(0, args.tof))
        axes[1].legend(loc="best", fontsize=8)
        fig.suptitle(f"Constraint diagnostics — floating-point checks: {checks_passed}\n"
                     "Plots are diagnostics; no formal interval certificate")

        for fig, filename in figures:
            fig.savefig(args.output / filename, dpi=180)
        print("Plots: " + ", ".join(filename for _, filename in figures))
        if args.show_plots:
            plt.show()
        for fig, _ in figures:
            plt.close(fig)


def solve(args, *, endpoint_velocities=None):
    """Solve in a frame with theta(0)=0.

    Optional endpoint_velocities is a (2, 2) array of [rho_dot, theta_dot]
    in canonical time units. When provided, no circular-velocity assumption
    is made. Endpoint radii and unwrapped arrival angle still come from args.
    """
    tof, mu = args.tof, args.mu
    n = args.n_ctrl
    degree = args.degree
    spans = n - degree
    certificate_degree = 10 * degree - 4
    minimize_peak = args.mode == "min-peak"
    omega0, omegaf = np.sqrt(mu / args.r0**3), np.sqrt(mu / args.rf**3)
    circular_bodies = endpoint_velocities is None
    if endpoint_velocities is None:
        endpoint_velocities = np.array([[0.0, omega0], [0.0, omegaf]])
    else:
        endpoint_velocities = np.asarray(endpoint_velocities, dtype=float)
        if endpoint_velocities.shape != (2, 2) or not np.all(np.isfinite(endpoint_velocities)):
            raise ValueError("endpoint_velocities must be a finite (2, 2) array")
        if args.theta_final is None:
            raise ValueError("An explicit theta_final is required for prescribed endpoints")
        omega0, omegaf = endpoint_velocities[:, 1]
    angle = args.theta_final
    if angle is None:
        angle = 0.5 * (omega0 + omegaf) * tof

    basis = spline_basis(n, degree)
    seed = initial_coefficients(basis, args.r0, args.rf, angle, omega0, omegaf, tof,
                                endpoint_velocities[0, 0], endpoint_velocities[1, 0])
    subdivisions = subdivision_matrices(certificate_degree, args.subdivide)
    certificate = certificate_function(degree)
    # A fixed positive scale improves conditioning without changing positivity.
    p_scale = args.u_max**2 * max(args.r0, args.rf)**4
    opti = ca.Opti()
    endpoint_positions = np.array([[args.r0, 0.0], [args.rf, angle]])
    boundary_rhs = np.vstack((endpoint_positions[0], tof * endpoint_velocities[0],
                              endpoint_positions[1], tof * endpoint_velocities[1]))
    controls, free_controls, seed, n_free = build_clamped_control_points(
        opti, basis, boundary_rhs, seed,
    )
    # Fixed endpoint rows are checked directly; only the free rows need NLP
    # radius bounds. This avoids adding constant inequalities to Opti.
    if np.any(seed[[0, 1, -2, -1], 0] < args.rho_min):
        raise ValueError("Clamped endpoint radius coefficients violate rho-min")
    if free_controls is not None:
        opti.subject_to(free_controls[:, 0] >= args.rho_min)
    if minimize_peak:
        # Optimize w/u_max^2 for scaling; no square root enters the NLP.
        peak_ratio_squared = opti.variable()
        opti.subject_to(peak_ratio_squared >= 0)
        peak_squared = args.u_max**2 * peak_ratio_squared
    else:
        peak_squared = args.u_max**2

    nodes, weights = np.polynomial.legendre.leggauss(args.quadrature)
    energy = 0
    coefficient_blocks = []
    seed_peak_squared = args.u_max**2
    for k in range(spans):
        left, right = k / spans, (k + 1) / spans
        extraction = extraction_matrix(basis, left, right, degree)
        bezier = ca.DM(extraction) @ controls
        p = certificate(bezier[:, 0], bezier[:, 1], tof / spans, mu, peak_squared)
        for matrix in subdivisions:
            coefficients = ca.DM(matrix) @ p / p_scale
            opti.subject_to(coefficients >= 0)
            coefficient_blocks.append(coefficients)
        if minimize_peak:
            # For the seed, P(w) = w*rho^4 + P(0). Initialize w above the
            # coefficient-wise requirement, rather than using sampled peaks.
            seed_bezier = extraction @ seed
            seed_p0 = np.asarray(certificate(seed_bezier[:, 0], seed_bezier[:, 1],
                                             tof / spans, mu, 0.0)).reshape(-1)
            seed_r2 = multiply(list(seed_bezier[:, 0]), list(seed_bezier[:, 0]))
            seed_r4 = np.asarray(elevate(multiply(seed_r2, seed_r2), certificate_degree))
            for matrix in subdivisions:
                slopes = matrix @ seed_r4
                if np.all(slopes > 0):
                    seed_peak_squared = max(seed_peak_squared,
                                            float(np.max(-(matrix @ seed_p0) / slopes)))
        # Quadrature is only for energy (objective in energy mode, diagnostic
        # in min-peak mode), never for the thrust constraints.
        for node, weight in zip(nodes, weights):
            tau = left + 0.5 * (node + 1) / spans
            q = ca.DM(basis(tau).reshape(1, -1)) @ controls
            qd = ca.DM(basis(tau, 1).reshape(1, -1)) @ controls / tof
            qdd = ca.DM(basis(tau, 2).reshape(1, -1)) @ controls / tof**2
            ur = qdd[0, 0] - q[0, 0] * qd[0, 1]**2 + mu / q[0, 0]**2
            ut = q[0, 0] * qdd[0, 1] + 2 * qd[0, 0] * qd[0, 1]
            energy += weight * tof / (2 * spans) * (ur**2 + ut**2)
    if minimize_peak:
        opti.minimize(peak_ratio_squared)
        opti.set_initial(peak_ratio_squared, 1.1 * seed_peak_squared / args.u_max**2)
    else:
        opti.minimize(energy / (args.u_max**2 * tof))
    opti.solver("ipopt", {"print_time": not args.quiet}, {
        "max_iter": args.max_iter, "tol": 1e-9, "constr_viol_tol": 1e-9,
        "acceptable_iter": 0, "bound_relax_factor": 0,
        "print_level": 0 if args.quiet else 5,
    })

    count = (certificate_degree + 1) * spans * 2**args.subdivide
    print(f"Degree-{degree} B-spline: {n} control points, {spans} spans, "
          f"{2*n} reconstructed coefficients, "
          f"{2*n_free} free coefficients, 0 endpoint equalities, "
          f"{n_free} free radius bounds, {count} thrust inequalities.")
    print("Endpoint position and velocity imposed by clamped construction "
          "(first two and last two control points eliminated).")
    print(f"Mode: {args.mode}; available acceleration: {args.u_max:g}; "
          f"fixed TOF: {tof:g}; final unwrapped angle: {angle:g} rad.")
    if minimize_peak:
        print("One additional nonnegative squared-peak variable; "
              "u_max is a comparison target, not a hard cap in this mode.")
    solution = opti.solve()  # Failure propagates; never save an infeasible iterate as success.
    c = np.asarray(solution.value(controls))
    p_values = np.asarray(solution.value(ca.vertcat(*coefficient_blocks))).reshape(-1)
    w_value = float(solution.value(peak_squared)) if minimize_peak else args.u_max**2
    limit = float(np.sqrt(max(0.0, w_value)))
    gap = limit - args.u_max
    # Independently evaluate the same returned trajectory's coefficients at
    # the requested u_max; a positive optimized gap alone proves nothing.
    available_blocks = []
    for k in range(spans):
        bezier = extraction_matrix(basis, k / spans, (k + 1) / spans, degree) @ c
        available_p = np.asarray(certificate(bezier[:, 0], bezier[:, 1], tof / spans,
                                            mu, args.u_max**2)).reshape(-1)
        available_blocks.extend(matrix @ available_p / p_scale for matrix in subdivisions)
    available_values = np.concatenate(available_blocks)

    tau = np.linspace(0, 1, args.samples)
    q = basis(tau) @ c
    qd = basis(tau, 1) @ c / tof
    qdd = basis(tau, 2) @ c / tof**2
    ur = qdd[:, 0] - q[:, 0] * qd[:, 1]**2 + mu / q[:, 0]**2
    ut = q[:, 0] * qdd[:, 1] + 2 * qd[:, 0] * qd[:, 1]
    norm = np.hypot(ur, ut)
    # Same canonical scaling and accuracy-controlled reference integral used
    # by the nominal 3D B-spline shaping implementation.
    def acceleration_norm(sample_tau):
        sample_tau = np.asarray(sample_tau, dtype=float)
        sample_q = basis(sample_tau) @ c
        sample_qd = basis(sample_tau, 1) @ c / tof
        sample_qdd = basis(sample_tau, 2) @ c / tof**2
        sample_ur = (sample_qdd[:, 0] - sample_q[:, 0] * sample_qd[:, 1]**2
                     + mu / sample_q[:, 0]**2)
        sample_ut = (sample_q[:, 0] * sample_qdd[:, 1]
                     + 2 * sample_qd[:, 0] * sample_qd[:, 1])
        return np.hypot(sample_ur, sample_ut)

    reference_metrics = evaluate_reference_metrics(
        acceleration_norm,
        delta_v_scale_km_s=tof * CANONICAL_DV_TO_KM_S,
        acceleration_scale_m_s2=CANONICAL_ACCEL_TO_M_S2,
    ).as_dict()
    delta_v_canonical = reference_metrics["delta_v_reference_km_s"] / CANONICAL_DV_TO_KM_S
    endpoint_error = max(
        float(np.max(np.abs(basis(np.array([0., 1.])) @ c - endpoint_positions))),
        float(np.max(np.abs(basis(np.array([0., 1.]), 1) @ c / tof
                            - endpoint_velocities))),
    )
    checks_passed = bool(
        np.all(np.isfinite(p_values)) and np.all(np.isfinite(c))
        and np.isfinite(w_value) and w_value / args.u_max**2 >= -1e-7
        and np.min(p_values) >= -1e-7
        and np.min(c[:, 0]) >= args.rho_min - 1e-8
        and endpoint_error <= 1e-7
        and np.max(norm) <= limit + 1e-7 * max(1.0, limit)
    )
    available_checks_passed = bool(
        checks_passed and np.all(np.isfinite(available_values))
        and np.min(available_values) >= -1e-7
        and np.max(norm) <= args.u_max + 1e-7 * max(1.0, args.u_max)
    )
    assessment = (
        "Trajectory meets available u_max within floating-point tolerances."
        if available_checks_passed else
        "Feasibility at available u_max was not established; this is not an infeasibility proof."
    )
    summary = {
        "status": solution.stats()["return_status"],
        "units": "consistent nondimensional units; acceleration, not thrust force",
        "parameters": {**vars(args), "output": str(args.output), "theta_final": float(angle)},
        "endpoint_model": "circular" if circular_bodies else "prescribed_planar_states",
        "endpoint_positions_rho_theta": endpoint_positions.tolist(),
        "endpoint_velocities_rho_dot_theta_dot": endpoint_velocities.tolist(),
        "number_of_coefficients": 2 * n,
        "number_of_free_coefficients": 2 * n_free,
        "number_of_free_control_points": n_free,
        "number_of_decision_variables": 2 * n_free + int(minimize_peak),
        "number_of_peak_nonnegativity_bounds": int(minimize_peak),
        "number_of_endpoint_equalities": 0,
        "number_of_endpoint_conditions_by_construction": 8,
        "boundary_control_points_fixed": True,
        "boundary_control_points_eliminated": True,
        "number_of_radius_bounds": n_free,
        "number_of_fixed_radius_coefficients_checked": 4,
        "number_of_thrust_inequalities": count,
        "design_acceleration_limit": limit,
        "optimized_peak_squared": w_value if minimize_peak else None,
        "required_peak_acceleration_bound": limit if minimize_peak else None,
        "available_acceleration_limit": args.u_max,
        "peak_bound_gap": gap,
        "peak_bound_gap_percent": 100 * gap / args.u_max,
        "available_limit_numerical_checks_passed": available_checks_passed,
        "feasibility_assessment": assessment,
        "global_optimality_proven": False,
        "certificate_normalization_scale": p_scale,
        "min_available_limit_normalized_bernstein_coefficient": float(np.min(available_values)),
        "min_normalized_bernstein_coefficient": float(np.min(p_values)),
        "min_radius_control_coefficient": float(np.min(c[:, 0])),
        "max_endpoint_residual": endpoint_error,
        "sampled_max_acceleration": float(np.max(norm)),
        "delta_v_canonical": float(delta_v_canonical),
        "delta_v_is_optimized": False,
        **reference_metrics,
        "integral_squared_acceleration": float(solution.value(energy)),
        "numerical_checks_passed": checks_passed,
        "formal_interval_certificate": False,
        "knots": basis.t.tolist(),
        "control_coefficients_rho_theta": c.tolist(),
        "normalized_bernstein_coefficients": p_values.tolist(),
        "available_limit_normalized_bernstein_coefficients": available_values.tolist(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "solution.json").write_text(json.dumps(summary, indent=2) + "\n")
    profile = np.column_stack((tau * tof, q, qd, ur, ut, norm,
                               q[:, 0] * np.cos(q[:, 1]), q[:, 0] * np.sin(q[:, 1])))
    np.savetxt(args.output / "trajectory.csv", profile, delimiter=",", comments="",
               header="time,rho,theta,rho_dot,theta_dot,u_rho,u_theta,u_norm,x,y")
    print(f"IPOPT: {summary['status']}")
    if minimize_peak:
        print(f"Required peak bound found: {limit:.9g}")
        print(f"Gap to available u_max: {gap:+.9g} ({100 * gap / args.u_max:+.3f}%)")
        print("Local solution with sufficient Bernstein constraints; no global infeasibility proof.")
    print(assessment)
    print(f"Minimum normalized Bernstein coefficient: {np.min(p_values):.6e}")
    print(f"Sampled maximum acceleration (diagnostic only): {np.max(norm):.9g}")
    print(f"Reference delta-v (diagnostic, not optimized): "
          f"{reference_metrics['delta_v_reference_km_s']:.9g} km/s")
    print(f"Endpoint residual: {endpoint_error:.3e}")
    print(f"Floating-point checks passed: {checks_passed}; formal interval certificate: False")
    print(f"Results: {args.output.resolve()}")
    if not args.no_plots:
        plot_solution(args, tau, q, ur, ut, p_values, angle, checks_passed, w_value,
                      circular_bodies=circular_bodies)
    if not checks_passed:
        raise RuntimeError("Post-solve numerical checks failed; inspect solution.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("min-peak", "energy"), default="min-peak",
                        help="min-peak: minimize required peak bound (default); "
                             "energy: minimize squared acceleration under a hard u-max cap")
    parser.add_argument("--degree", type=int, choices=(3, 5), default=3,
                        help="B-spline degree: cubic (3) or quintic (5)")
    parser.add_argument("--n-ctrl", type=int, default=8,
                        help="Total number of B-spline control points per coordinate")
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--r0", type=float, default=1.0)
    parser.add_argument("--rf", type=float, default=1.5)
    parser.add_argument("--tof", type=float, default=8.0)
    parser.add_argument("--theta-final", type=float, default=None,
                        help="Unwrapped final angle in radians; default: mean endpoint omega * TOF")
    parser.add_argument("--u-max", type=float, default=0.2,
                        help="Available acceleration: comparison target in min-peak mode, "
                             "hard cap in energy mode; nondimensional units")
    parser.add_argument("--rho-min", type=float, default=0.2)
    parser.add_argument("--subdivide", type=int, default=0,
                        help="Certificate subdivision depth; constraints per span are "
                             "(10*degree-3)*2**depth")
    parser.add_argument("--quadrature", type=int, default=12,
                        help="Gauss nodes per span for the energy objective/diagnostic")
    parser.add_argument("--samples", type=int, default=2001,
                        help="Diagnostic/output samples; never used to enforce thrust bounds")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--quiet", action="store_true")
    plots = parser.add_mutually_exclusive_group()
    plots.add_argument("--show-plots", action="store_true",
                       help="Save plots and also display them in matplotlib windows")
    plots.add_argument("--no-plots", action="store_true",
                       help="Skip plots (default: save three PNG figures)")
    parser.add_argument("--output", type=Path, default=Path("output/planar_cubic_bernstein"),
                        help="Directory for JSON, CSV, and plots (overwritten on rerun)")
    args = parser.parse_args()
    positive = (args.mu, args.r0, args.rf, args.tof, args.u_max, args.rho_min)
    if any(not np.isfinite(value) or value <= 0 for value in positive):
        parser.error("mu, endpoint radii, tof, u-max, and rho-min must be finite and positive")
    if args.theta_final is not None and not np.isfinite(args.theta_final):
        parser.error("theta-final must be finite")
    if args.rho_min > min(args.r0, args.rf):
        parser.error("rho-min cannot exceed either endpoint radius")
    if args.n_ctrl < args.degree + 2 or args.subdivide < 0 or args.quadrature < 1:
        parser.error("n-ctrl must give at least two spans; subdivide >= 0 and quadrature >= 1")
    if args.samples < 2 or args.max_iter < 1:
        parser.error("samples >= 2 and max-iter >= 1 are required")
    solve(args)


if __name__ == "__main__":
    main()

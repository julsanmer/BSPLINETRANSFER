"""Sundman-transformed Pythagorean shaping with a rational-Bezier hull bound.

This is a separate shaping family from ``pythagorean_3d_bernstein``; it is not a
faster solver for the same trajectories.  Two changes are stacked on top of that
module, and both are confined to this file.

Sundman transformation
----------------------
The independent variable is the Sundman arc ``s`` with ``dt = R ds``, written as
``s = S*sigma`` for a free scalar arc length ``S`` and the usual spline
parameter ``sigma`` in [0, 1].  With primes denoting d/ds,

    rdot = r' / R,      rddot = (R r'' - R' r') / R^3,

so clearing the single remaining ``R^3`` leaves polynomial thrust numerators

    P_rho   = R rho'' - R' rho' - R rho theta'^2 + mu rho,
    P_theta = rho (R theta'' - R' theta') + 2 R rho' theta',
    P_z     = R z''   - R' z'   + mu z,
    W       = R^3,                       u = (P_rho, P_theta, P_z) / W.

For primitive splines of degree ``n`` the numerators reach degree ``8n-2`` and
the denominator ``9n``, against ``14n-2`` for the time-parametrized numerator of
the sibling module.  Time of flight becomes the constraint ``T = S * integral of R dsigma``,
which is exact in the Bernstein basis (the integral of a Bernstein polynomial is
the mean of its coefficients), so no quadrature enters the transfer time.

Rational-Bezier convex-hull thrust bound
----------------------------------------
``u = P / W`` is a rational Bezier curve of degree ``m = 9n`` with weights ``W_i``.
When every ``W_i > 0`` the curve lies in the convex hull of ``q_i = P_i / W_i``, so

    ||P_i||^2 <= u_max^2 W_i^2,        i = 0, ..., m

is sufficient for ``||u(sigma)|| <= u_max`` on the whole span.  That is ``9n+1``
smooth scalar constraints per span instead of the ``28n-4`` Bernstein
nonnegativity rows of the squared certificate, and the degree-``18n`` product is
never formed.  The positivity requirement is free here: Bernstein coefficients
are closed under products with positive weights, so ``W = R^3`` inherits strict
positivity from the ``radius >= r_min`` constraint already imposed on ``R``.  It
is verified after the solve rather than duplicated as constraint rows.

Both conditions are sufficient and conservative, and neither dominates the other;
``--thrust-enforcement grid`` enforces the pointwise limit at nodes instead, to
measure how much the hull bound gives away.

Run from the repository root, for example:

    python -m optimizer.pythagorean_3d_sundman --quiet
"""

from __future__ import annotations

import argparse
import json
from math import comb
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
from optimizer.pythagorean_3d_bernstein import (
    CANONICAL_ACCEL_TO_MM_S2,
    configure_spice_case,
    departure_epoch,
    initial_primitive_coefficients,
    plot_solution,
    primitive_endpoint_data,
    scale,
    write_shaping_artifacts,
)


def bezier_value(coefficients, local):
    """Evaluate a numeric Bernstein polynomial at local parameters in [0, 1]."""
    coefficients = np.asarray(coefficients, dtype=float)
    degree = len(coefficients) - 1
    local = np.asarray(local, dtype=float)
    basis = np.stack([comb(degree, i) * local**i * (1.0 - local)**(degree - i)
                      for i in range(degree + 1)], axis=-1)
    return basis @ coefficients


def sundman_hull_function(degree):
    """Return span coefficients of the thrust numerators and of W = R^3.

    ``arc`` is the Sundman arc length spanned by one knot interval, so the
    ``derivative`` helper differentiates with respect to ``s`` directly.
    """
    h = ca.SX.sym("h", degree + 1)
    p = ca.SX.sym("p", degree + 1)
    q = ca.SX.sym("q", degree + 1)
    theta = ca.SX.sym("theta", degree + 1)
    arc = ca.SX.sym("arc")
    mu = ca.SX.sym("mu")

    h_c = [h[i] for i in range(degree + 1)]
    p_c = [p[i] for i in range(degree + 1)]
    q_c = [q[i] for i in range(degree + 1)]
    theta_c = [theta[i] for i in range(degree + 1)]

    p2 = multiply(p_c, p_c)
    q2 = multiply(q_c, q_c)
    rho = multiply(h_c, add(p2, q2, -1.0))
    z = scale(multiply(h_c, multiply(p_c, q_c)), 2.0)
    radius = multiply(h_c, add(p2, q2))

    rho_d = derivative(rho, arc)
    rho_dd = derivative(rho_d, arc)
    z_d = derivative(z, arc)
    z_dd = derivative(z_d, arc)
    radius_d = derivative(radius, arc)
    theta_d = derivative(theta_c, arc)
    theta_dd = derivative(theta_d, arc)

    # P_rho = R rho'' - R' rho' - R rho theta'^2 + mu rho
    p_rho = add(multiply(radius, rho_dd), multiply(radius_d, rho_d), -1.0)
    p_rho = add(p_rho,
                multiply(radius, multiply(rho, multiply(theta_d, theta_d))), -1.0)
    p_rho = add(p_rho, scale(rho, mu))
    # P_theta = rho (R theta'' - R' theta') + 2 R rho' theta'
    swing = add(multiply(radius, theta_dd), multiply(radius_d, theta_d), -1.0)
    p_theta = add(multiply(rho, swing),
                  scale(multiply(radius, multiply(rho_d, theta_d)), 2.0))
    # P_z = R z'' - R' z' + mu z
    p_z = add(multiply(radius, z_dd), multiply(radius_d, z_d), -1.0)
    p_z = add(p_z, scale(z, mu))
    weight = multiply(radius, multiply(radius, radius))

    hull_degree = max(len(p_rho), len(p_theta), len(p_z), len(weight)) - 1
    return ca.Function(
        f"pythagorean_sundman_hull_p{degree}",
        [h, p, q, theta, arc, mu],
        [ca.vertcat(*elevate(p_rho, hull_degree)),
         ca.vertcat(*elevate(p_theta, hull_degree)),
         ca.vertcat(*elevate(p_z, hull_degree)),
         ca.vertcat(*elevate(weight, hull_degree)),
         ca.vertcat(*rho), ca.vertcat(*radius)],
        ["h", "p", "q", "theta", "arc", "mu"],
        ["p_rho", "p_theta", "p_z", "weight", "rho", "radius"],
    ), hull_degree


def physical_state(primitives, primitive_d, primitive_dd, mu):
    """Cylindrical state and control from primitives and their s-derivatives.

    ``primitive_d``/``primitive_dd`` are d/ds and d2/ds2.  The returned rates are
    physical time rates, obtained from the Sundman relation ``fdot = f'/R``.
    """
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
    weight = radius**3
    u_rho = (radius * rho_dd - radius_d * rho_d
             - radius * rho * theta_d**2 + mu * rho) / weight
    u_theta = (rho * (radius * theta_dd - radius_d * theta_d)
               + 2 * radius * rho_d * theta_d) / weight
    u_z = (radius * z_dd - radius_d * z_d + mu * z) / weight
    return (rho, theta, z, rho_d / radius, theta_d / radius, z_d / radius,
            radius, u_rho, u_theta, u_z)


def evaluate_dense_at_sigma(basis, controls, arc, mu, sigma):
    """Evaluate the trajectory at spline parameters, not at uniform times."""
    sigma = np.asarray(sigma, dtype=float)
    primitive = basis(sigma) @ controls
    primitive_d = basis(sigma, 1) @ controls / arc
    primitive_dd = basis(sigma, 2) @ controls / arc**2
    columns = [[] for _ in range(10)]
    for row, row_d, row_dd in zip(primitive, primitive_d, primitive_dd):
        for column, value in zip(columns, physical_state(row, row_d, row_dd, mu)):
            column.append(float(value))
    return [np.asarray(column) for column in columns]


def span_radius_coefficients(extractions, controls, hull, mu):
    """Bernstein coefficients of R on each span; independent of the arc length."""
    coefficients = []
    for extraction in extractions:
        bezier = extraction @ np.asarray(controls, dtype=float)
        result = hull(bezier[:, 0], bezier[:, 1], bezier[:, 2], bezier[:, 3],
                      1.0, mu)
        coefficients.append(np.asarray(result[5]).reshape(-1))
    return coefficients


def arc_length_for_tof(radius_coefficients, tof):
    """Solve T = S * integral of R dsigma exactly in the Bernstein basis."""
    spans = len(radius_coefficients)
    integral = sum(float(np.mean(block)) for block in radius_coefficients) / spans
    if not np.isfinite(integral) or integral <= 0:
        raise ValueError("Seed trajectory has a nonpositive mean radius")
    return tof / integral


class SundmanTimeMap:
    """Exact piecewise map between the spline parameter and physical time.

    Time is the antiderivative of ``dt = S R dsigma``.  In the Bernstein basis
    that antiderivative is the degree-raised polynomial whose coefficients are
    the scaled partial sums of the coefficients of ``R``, so both directions are
    exact to rounding rather than quadrature-limited.
    """

    def __init__(self, radius_coefficients, arc):
        self.spans = len(radius_coefficients)
        self.arc = float(arc)
        self.radius = [np.asarray(block, dtype=float)
                       for block in radius_coefficients]
        step = self.arc / self.spans
        self.antiderivative = []
        starts = [0.0]
        for block in self.radius:
            partial = np.concatenate(([0.0], np.cumsum(block)))
            partial *= step / len(block)
            self.antiderivative.append(partial)
            starts.append(starts[-1] + partial[-1])
        self.span_start_time = np.asarray(starts)
        self.total_time = float(starts[-1])

    def _locate(self, sigma):
        index = np.clip(np.floor(sigma * self.spans).astype(int),
                        0, self.spans - 1)
        return index, sigma * self.spans - index

    def time(self, sigma):
        sigma = np.clip(np.asarray(sigma, dtype=float), 0.0, 1.0)
        index, local = self._locate(sigma)
        values = np.empty(sigma.shape if sigma.ndim else (1,))
        flat_index, flat_local = np.atleast_1d(index), np.atleast_1d(local)
        flat = np.empty(flat_index.shape)
        for span in range(self.spans):
            mask = flat_index == span
            if np.any(mask):
                flat[mask] = (self.span_start_time[span]
                              + bezier_value(self.antiderivative[span],
                                             flat_local[mask]))
        values = flat.reshape(sigma.shape) if sigma.ndim else flat[0]
        return values

    def rate(self, sigma):
        """dt/dsigma = S R, strictly positive, used as the Newton derivative."""
        sigma = np.clip(np.asarray(sigma, dtype=float), 0.0, 1.0)
        index, local = self._locate(sigma)
        flat_index, flat_local = np.atleast_1d(index), np.atleast_1d(local)
        flat = np.empty(flat_index.shape)
        for span in range(self.spans):
            mask = flat_index == span
            if np.any(mask):
                flat[mask] = self.arc * bezier_value(self.radius[span],
                                                     flat_local[mask])
        return flat.reshape(sigma.shape) if sigma.ndim else flat[0]

    def sigma(self, time):
        """Invert the monotone time map with a bisection-safeguarded Newton."""
        target = np.atleast_1d(np.asarray(time, dtype=float))
        lower = np.zeros_like(target)
        upper = np.ones_like(target)
        guess = np.clip(target / self.total_time, 0.0, 1.0)
        tolerance = 1e-14 * max(1.0, self.total_time)
        for _ in range(80):
            residual = self.time(guess) - target
            if np.all(np.abs(residual) <= tolerance):
                break
            lower = np.where(residual < 0.0, guess, lower)
            upper = np.where(residual > 0.0, guess, upper)
            step = guess - residual / self.rate(guess)
            outside = (step <= lower) | (step >= upper) | ~np.isfinite(step)
            guess = np.where(outside, 0.5 * (lower + upper), step)
        result = np.clip(guess, 0.0, 1.0)
        return result.reshape(np.shape(time)) if np.ndim(time) else float(result[0])


def build_clamped_primitives(opti, basis, values, rates, arc, radius_endpoints, seed):
    """Eliminate the endpoint conditions; spline rates carry a factor S*R.

    A physical rate maps to a spline rate through ``f' = S R fdot``, so the two
    control points next to each end become affine in the arc-length variable.
    """
    start_d = basis(0.0, 1)
    end_d = basis(1.0, 1)
    c0 = ca.DM(values[0:1])
    c_last = ca.DM(values[1:2])
    start_rate = arc * float(radius_endpoints[0]) * ca.DM(rates[0:1])
    end_rate = arc * float(radius_endpoints[1]) * ca.DM(rates[1:2])
    c1 = (start_rate - start_d[0] * c0) / start_d[1]
    c_prev = (end_rate - end_d[-1] * c_last) / end_d[-2]
    n_free = len(basis.c) - 4
    if n_free:
        free = opti.variable(n_free, 4)
        opti.set_initial(free, seed[2:-2])
        controls = ca.vertcat(c0, c1, free, c_prev, c_last)
    else:
        free = None
        controls = ca.vertcat(c0, c1, c_prev, c_last)
    return controls, free, n_free


def projected_seed_controls(basis, values, rates, arc, radius_endpoints, seed):
    """Numeric counterpart of the clamped layout, for warm starts and scaling."""
    start_d = basis(0.0, 1)
    end_d = basis(1.0, 1)
    c0 = values[0:1]
    c_last = values[1:2]
    c1 = (arc * radius_endpoints[0] * rates[0:1] - start_d[0] * c0) / start_d[1]
    c_prev = (arc * radius_endpoints[1] * rates[1:2] - end_d[-1] * c_last) / end_d[-2]
    return np.vstack((c0, c1, seed[2:-2], c_prev, c_last))


def build_seed(basis, values, rates, radius_endpoints, tof, mu, extractions, hull):
    """Fixed-point pass: Hermite seed and the arc length it implies agree."""
    arc = tof / float(np.mean(radius_endpoints))
    seed = None
    for _ in range(8):
        spline_rates = rates * (arc * radius_endpoints)[:, None]
        seed = initial_primitive_coefficients(basis, values, spline_rates, 1.0)
        updated = arc_length_for_tof(
            span_radius_coefficients(extractions, seed, hull, mu), tof)
        if abs(updated - arc) <= 1e-13 * max(1.0, abs(arc)):
            arc = updated
            break
        arc = updated
    return seed, arc


def seed_hull_peak_squared(extractions, seed, hull, subdivisions, arc, spans, mu):
    """Smallest squared peak for which the hull condition holds for the seed."""
    peak = 0.0
    for extraction in extractions:
        bezier = extraction @ np.asarray(seed, dtype=float)
        result = hull(bezier[:, 0], bezier[:, 1], bezier[:, 2], bezier[:, 3],
                      arc / spans, mu)
        blocks = [np.asarray(result[i]).reshape(-1) for i in range(4)]
        for matrix in subdivisions:
            p_rho, p_theta, p_z, weight = (matrix @ block for block in blocks)
            if np.any(weight <= 0):
                return float("nan")
            peak = max(peak, float(np.max(
                (p_rho**2 + p_theta**2 + p_z**2) / weight**2)))
    return peak


def solve(args):
    solve_started = perf_counter()
    basis = spline_basis(args.n_ctrl, args.degree)
    spans = args.n_ctrl - args.degree
    values, rates, endpoint_state, endpoint_rates = primitive_endpoint_data(args)
    radius_endpoints = np.array([args.r0, args.rf], dtype=float)
    hull, hull_degree = sundman_hull_function(args.degree)
    hull_subdivisions = subdivision_matrices(hull_degree, args.subdivide)
    geometry_degree = 3 * args.degree
    geometry_subdivisions = subdivision_matrices(geometry_degree, args.subdivide)
    extractions = [
        extraction_matrix(basis, span / spans, (span + 1) / spans, args.degree)
        for span in range(spans)
    ]

    # A supplied seed replaces the internal Hermite guess entirely, so the guess
    # is only built when it is actually needed.  The Hermite fit can overshoot
    # into a nonpositive radius on demanding cases, and that must not abort a run
    # whose seed comes from elsewhere.
    seed = seed_arc = None
    initial_peak_acceleration_bound = None
    if getattr(args, "initial_solution", None) is not None:
        payload = json.loads(Path(args.initial_solution).read_text())
        loaded = np.asarray(payload["control_coefficients_h_p_q_theta"], dtype=float)
        if loaded.shape != (args.n_ctrl, 4) or not np.all(np.isfinite(loaded)):
            raise ValueError("initial-solution has incompatible Pythagorean controls")
        seed = loaded
        seed_arc = payload.get("sundman_arc_length")
        seed_arc = float(seed_arc) if seed_arc is not None else None
        if payload.get("initial_peak_acceleration_bound") is not None:
            initial_peak_acceleration_bound = float(
                payload["initial_peak_acceleration_bound"])
            if (not np.isfinite(initial_peak_acceleration_bound)
                    or initial_peak_acceleration_bound <= 0.0):
                raise ValueError(
                    "initial_peak_acceleration_bound must be positive and finite")

    if seed is None or seed_arc is None:
        fallback, fallback_arc = build_seed(
            basis, values, rates, radius_endpoints, args.tof, args.mu,
            extractions, hull)
        seed = fallback if seed is None else seed
        seed_arc = fallback_arc if seed_arc is None else seed_arc

    opti = ca.Opti()
    arc = opti.variable()
    opti.set_initial(arc, seed_arc)
    opti.subject_to(arc >= args.arc_min * seed_arc)
    controls, free, n_free = build_clamped_primitives(
        opti, basis, values, rates, arc, radius_endpoints, seed)
    projected_seed = projected_seed_controls(
        basis, values, rates, seed_arc, radius_endpoints, seed)
    if free is not None:
        opti.subject_to(free[:, 0] >= args.h_min)

    minimize_peak = args.mode == "min-peak"
    minimize_delta_v = args.mode == "min-dv"
    thrust_enforcement = args.thrust_enforcement
    if minimize_delta_v and args.u_max is None:
        # No acceleration limit exists, so no thrust condition is imposed and the
        # hull bound of the returned trajectory is reported a posteriori instead.
        thrust_enforcement = "none"
    nodes_per_span = args.constraint_nodes_per_span
    if minimize_peak:
        # No acceleration limit exists in this mode, so the squared peak variable
        # and the hull residuals are made dimensionless with the hull bound of
        # the seed trajectory instead.
        seed_peak_squared = seed_hull_peak_squared(
            extractions, projected_seed, hull, hull_subdivisions,
            seed_arc, spans, args.mu)
        reference_squared = (
            seed_peak_squared
            if np.isfinite(seed_peak_squared) and seed_peak_squared > 0 else 1.0)
        peak_ratio_squared = opti.variable()
        opti.subject_to(peak_ratio_squared >= 0)
        peak_squared = reference_squared * peak_ratio_squared
    elif args.mode == "energy":
        seed_peak_squared = float("nan")
        reference_squared = args.u_max**2
        peak_ratio_squared = None
        peak_squared = reference_squared
    elif minimize_delta_v:
        # The limit is optional here, so the objective is made dimensionless with
        # the hull bound of the seed trajectory when no limit is supplied.
        seed_peak_squared = seed_hull_peak_squared(
            extractions, projected_seed, hull, hull_subdivisions,
            seed_arc, spans, args.mu)
        if args.u_max is not None:
            reference_squared = args.u_max**2
        elif np.isfinite(seed_peak_squared) and seed_peak_squared > 0:
            reference_squared = seed_peak_squared
        else:
            reference_squared = 1.0
        peak_ratio_squared = None
        peak_squared = reference_squared
    scale_hull = reference_squared * max(args.r0, args.rf)**6
    residual_blocks = []
    grid_constraint_sigma = []
    total_time = 0
    energy = 0
    delta_v_expression = 0
    delta_v_node_count = 0
    if minimize_peak:
        nodes, weights = np.empty(0), np.empty(0)
    else:
        nodes, weights = np.polynomial.legendre.leggauss(args.quadrature)
    for span in range(spans):
        left, right = span / spans, (span + 1) / spans
        bezier = ca.DM(extractions[span]) @ controls
        result = hull(bezier[:, 0], bezier[:, 1], bezier[:, 2], bezier[:, 3],
                      arc / spans, args.mu)
        p_rho, p_theta, p_z, weight = (result[i] for i in range(4))
        rho_coeff, radius_coeff = result[4], result[5]
        # Exact Bernstein integral of dt = S R dsigma over this knot interval.
        total_time += (arc / spans) * ca.sum1(radius_coeff) / radius_coeff.shape[0]

        for matrix in hull_subdivisions:
            block_w = ca.DM(matrix) @ weight
            block = (peak_squared * block_w**2
                     - (ca.DM(matrix) @ p_rho)**2
                     - (ca.DM(matrix) @ p_theta)**2
                     - (ca.DM(matrix) @ p_z)**2) / scale_hull
            if thrust_enforcement == "hull":
                opti.subject_to(block >= 0)
            residual_blocks.append(block)
        for matrix in geometry_subdivisions:
            opti.subject_to(ca.DM(matrix) @ rho_coeff >= args.rho_min)
            opti.subject_to(ca.DM(matrix) @ radius_coeff >= args.r_min)

        if thrust_enforcement == "grid":
            span_nodes = np.linspace(left, right, nodes_per_span)
            # A degree-2 spline is C1, so acceleration may jump at a knot.
            if span < spans - 1:
                span_nodes[-1] = np.nextafter(right, left)
            for sigma in span_nodes:
                state = span_physical_state(basis, controls, arc, args.mu, sigma)
                opti.subject_to(state[-3]**2 + state[-2]**2 + state[-1]**2
                                <= peak_squared)
                grid_constraint_sigma.append(float(sigma))

        for node, weight_node in zip(nodes, weights):
            sigma = left + 0.5 * (node + 1) / spans
            state = span_physical_state(basis, controls, arc, args.mu, sigma)
            radius_node = state[6]
            norm_squared = state[-3]**2 + state[-2]**2 + state[-1]**2
            jacobian_weight = weight_node * arc / (2 * spans) * radius_node
            # int ||u||^2 dt = S int ||u||^2 R dsigma
            energy += jacobian_weight * norm_squared
            if minimize_delta_v:
                # Epigraph of the nonsmooth norm: the bound is active at the
                # optimum, so int ||u|| dt = S int ||u|| R dsigma is recovered
                # exactly while the solver only sees smooth expressions.
                node_bound = opti.variable()
                opti.subject_to(node_bound >= 0)
                opti.subject_to(norm_squared <= node_bound**2)
                opti.set_initial(node_bound, float(np.sqrt(reference_squared)))
                delta_v_expression += jacobian_weight * node_bound
                delta_v_node_count += 1

    opti.subject_to(total_time == args.tof)
    if minimize_peak:
        opti.minimize(peak_ratio_squared)
        if initial_peak_acceleration_bound is None:
            initial_peak_ratio_squared = float(args.peak_init_factor)
        else:
            initial_peak_ratio_squared = (
                initial_peak_acceleration_bound**2 / reference_squared)
        opti.set_initial(peak_ratio_squared, initial_peak_ratio_squared)
    elif args.mode == "energy":
        opti.minimize(energy / (reference_squared * args.tof))
    elif minimize_delta_v:
        opti.minimize(delta_v_expression
                      / (float(np.sqrt(reference_squared)) * args.tof))

    ipopt_options = {
        "max_iter": args.max_iter,
        "tol": 1e-9,
        "constr_viol_tol": 1e-9,
        "acceptable_iter": 0,
        "bound_relax_factor": 0,
        "print_level": 0 if args.quiet else 5,
    }
    opti.solver("ipopt", {"print_time": not args.quiet}, ipopt_options)
    thrust_count = (
        (hull_degree + 1) * spans * 2**args.subdivide
        if thrust_enforcement == "hull"
        else spans * nodes_per_span if thrust_enforcement == "grid" else 0
    )
    print(f"Degree-{args.degree} Sundman Pythagorean B-spline: {args.n_ctrl} "
          f"controls, {spans} spans, {4 * n_free} trajectory variables, "
          f"hull degree {hull_degree}, "
          f"{thrust_count} {thrust_enforcement} thrust inequalities.")
    solution = opti.solve()
    solver_stats = solution.stats()
    total_ipopt_iterations = int(solver_stats.get("iter_count", -1))
    solved_controls = np.asarray(solution.value(controls))
    solved_arc = float(solution.value(arc))
    if minimize_peak:
        w_value = float(solution.value(peak_squared))
    elif args.mode == "energy":
        w_value = reference_squared
    elif minimize_delta_v:
        w_value = reference_squared if args.u_max is not None else float("nan")
    limit = float(np.sqrt(max(0.0, w_value)))
    residuals = np.asarray(
        solution.value(ca.vertcat(*residual_blocks))).reshape(-1)

    # Smallest peak for which the hull condition certifies this trajectory, and
    # the strict weight positivity the convex-combination argument requires.
    continuous_bound_squared = 0.0
    minimum_weight = float("inf")
    for extraction in extractions:
        bezier = extraction @ solved_controls
        result = hull(bezier[:, 0], bezier[:, 1], bezier[:, 2], bezier[:, 3],
                      solved_arc / spans, args.mu)
        blocks = [np.asarray(result[i]).reshape(-1) for i in range(4)]
        for matrix in hull_subdivisions:
            p_rho, p_theta, p_z, weight = (matrix @ block for block in blocks)
            minimum_weight = min(minimum_weight, float(np.min(weight)))
            if np.any(weight <= 0):
                continuous_bound_squared = float("inf")
            elif np.isfinite(continuous_bound_squared):
                continuous_bound_squared = max(continuous_bound_squared, float(
                    np.max((p_rho**2 + p_theta**2 + p_z**2) / weight**2)))
    continuous_bound = float(np.sqrt(max(0.0, continuous_bound_squared)))
    if minimize_delta_v and args.u_max is None:
        # Nothing was imposed, so the reported limit is the certified hull bound
        # of the Delta-V optimal trajectory rather than a constraint value.
        limit = continuous_bound
    radius_coefficients = span_radius_coefficients(
        extractions, solved_controls, hull, args.mu)
    time_map = SundmanTimeMap(radius_coefficients, solved_arc)
    tof_error = abs(time_map.total_time - args.tof)

    sigma = np.linspace(0.0, 1.0, args.samples)
    (rho, theta, z, rho_d, theta_d, z_d, radius,
     ur, ut, uz) = evaluate_dense_at_sigma(
        basis, solved_controls, solved_arc, args.mu, sigma)
    time = time_map.time(sigma)
    norm = np.sqrt(ur**2 + ut**2 + uz**2)
    if grid_constraint_sigma:
        grid_sigma = np.unique(np.asarray(grid_constraint_sigma))
        grid_state = evaluate_dense_at_sigma(
            basis, solved_controls, solved_arc, args.mu, grid_sigma)
        grid_peak = float(np.max(np.sqrt(
            grid_state[-3]**2 + grid_state[-2]**2 + grid_state[-1]**2)))
    else:
        grid_peak = float("nan")

    identity_error = float(np.max(np.abs(radius**2 - rho**2 - z**2)))
    endpoint_position_error = float(np.max(np.abs(
        np.column_stack((rho[[0, -1]], theta[[0, -1]], z[[0, -1]])) - endpoint_state)))
    endpoint_velocity_error = float(np.max(np.abs(
        np.column_stack((rho_d[[0, -1]], theta_d[[0, -1]], z_d[[0, -1]]))
        - endpoint_rates)))
    thrust_checks_passed = bool(
        (thrust_enforcement == "none")
        or (thrust_enforcement == "hull" and np.min(residuals) >= -1e-7
         and minimum_weight > 0.0
         and np.max(norm) <= limit + 1e-7 * max(1.0, limit))
        or (thrust_enforcement == "grid"
            and grid_peak <= limit + 1e-7 * max(1.0, limit))
    )
    dense_limit_checks_passed = bool(
        np.max(norm) <= limit + 1e-7 * max(1.0, limit))
    checks_passed = bool(
        np.all(np.isfinite(solved_controls))
        and np.isfinite(solved_arc) and solved_arc > 0
        and np.min(rho) >= args.rho_min - 1e-8
        and np.min(radius) >= args.r_min - 1e-8
        and thrust_checks_passed
        and identity_error <= 1e-10
        and endpoint_position_error <= 1e-7
        and endpoint_velocity_error <= 1e-7
        and tof_error <= 1e-9 * max(1.0, args.tof)
    )
    delta_v = float(np.trapezoid(norm, time))

    def acceleration_norm(normalized_time):
        """Sampler in normalized physical time, as the reference metrics expect."""
        sample_sigma = time_map.sigma(
            np.asarray(normalized_time, dtype=float) * time_map.total_time)
        sample = evaluate_dense_at_sigma(
            basis, solved_controls, solved_arc, args.mu, sample_sigma)
        return np.sqrt(sample[-3]**2 + sample[-2]**2 + sample[-1]**2)

    reference_metrics = evaluate_reference_metrics(
        acceleration_norm,
        delta_v_scale_km_s=args.tof * CANONICAL_DV_TO_KM_S,
        acceleration_scale_m_s2=CANONICAL_ACCEL_TO_M_S2,
    ).as_dict()
    wall_time = perf_counter() - solve_started

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": solver_stats["return_status"],
        "formulation": "3D cylindrical Pythagorean Sundman rational-Bezier hull",
        "parameters": {**{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items() if not key.startswith("_")
        }, "output": str(args.output)},
        "spice_boundary_data": getattr(args, "_spice_context", None),
        "hull_degree": hull_degree,
        "certificate_degree": hull_degree,
        "squared_certificate_degree_avoided": 28 * args.degree - 4,
        "thrust_enforcement": thrust_enforcement,
        "constraint_nodes_per_span": nodes_per_span if thrust_enforcement == "grid" else None,
        "sundman_arc_length": solved_arc,
        "sundman_arc_length_seed": seed_arc,
        "time_of_flight_error": tof_error,
        "minimum_hull_weight": minimum_weight,
        "number_of_free_trajectory_variables": 4 * n_free,
        "number_of_decision_variables": (4 * n_free + 1 + int(minimize_peak)
                                        + delta_v_node_count),
        "delta_v_epigraph_nodes": delta_v_node_count,
        "number_of_thrust_inequalities": thrust_count,
        "required_peak_acceleration_bound": limit,
        "required_peak_acceleration_mm_s2": limit * CANONICAL_ACCEL_TO_MM_S2,
        "peak_reference_scale": float(np.sqrt(reference_squared)),
        "initial_peak_acceleration_bound": initial_peak_acceleration_bound,
        "initial_peak_ratio_squared": (
            initial_peak_ratio_squared if minimize_peak else None),
        "seed_hull_peak_bound": (
            float(np.sqrt(seed_peak_squared)) if minimize_peak else None),
        "continuous_bernstein_bound_for_returned_trajectory": continuous_bound,
        "continuous_bernstein_bound_for_returned_trajectory_mm_s2":
            continuous_bound * CANONICAL_ACCEL_TO_MM_S2,
        "minimum_normalized_bernstein_coefficient": float(np.min(residuals)),
        "sampled_peak_acceleration": float(np.max(norm)),
        "sampled_peak_acceleration_mm_s2":
            float(np.max(norm)) * CANONICAL_ACCEL_TO_MM_S2,
        "grid_node_peak_acceleration": grid_peak if thrust_enforcement == "grid" else None,
        "dense_peak_over_grid_peak_percent":
            100.0 * (float(np.max(norm)) / grid_peak - 1.0)
            if thrust_enforcement == "grid" else None,
        "hull_bound_over_grid_peak_percent":
            100.0 * (continuous_bound / grid_peak - 1.0)
            if thrust_enforcement == "grid" else None,
        "dense_limit_checks_passed": dense_limit_checks_passed,
        "delta_v_nondimensional": delta_v,
        **reference_metrics,
        "wall_time_s": wall_time,
        "ipopt_iterations": total_ipopt_iterations,
        "max_pythagorean_identity_error": identity_error,
        "max_endpoint_position_error": endpoint_position_error,
        "max_endpoint_velocity_error": endpoint_velocity_error,
        "numerical_checks_passed": checks_passed,
        "formal_interval_certificate": False,
        "control_coefficients_h_p_q_theta": solved_controls.tolist(),
    }
    (args.output / "solution.json").write_text(json.dumps(summary, indent=2) + "\n")
    x, y = rho * np.cos(theta), rho * np.sin(theta)
    table = np.column_stack((time, sigma, rho, theta, z, radius, rho_d,
                             theta_d, z_d, ur, ut, uz, norm, x, y))
    np.savetxt(args.output / "trajectory.csv", table, delimiter=",", comments="",
               header="time,sigma,rho,theta,z,R,rho_dot,theta_dot,z_dot,"
                      "u_rho,u_theta,u_z,u_norm,x,y")
    write_shaping_artifacts(args, basis, solved_controls, projected_seed, summary)

    print(f"IPOPT: {summary['status']}")
    print(f"Sundman arc length S: {solved_arc:.9g} "
          f"(seed {seed_arc:.9g}), TOF error {tof_error:.3e}")
    print(f"Delta V (reference): {summary['delta_v_reference_km_s']:.9g} km/s")
    print(f"Required peak acceleration bound: "
          f"{limit * CANONICAL_ACCEL_TO_MM_S2:.9g} mm/s^2")
    print(f"Sampled peak acceleration: "
          f"{np.max(norm) * CANONICAL_ACCEL_TO_MM_S2:.9g} mm/s^2")
    print(f"Hull bound required by the returned trajectory: "
          f"{continuous_bound * CANONICAL_ACCEL_TO_MM_S2:.9g} mm/s^2")
    if thrust_enforcement == "grid":
        print(f"Grid-node peak acceleration: "
              f"{grid_peak * CANONICAL_ACCEL_TO_MM_S2:.9g} mm/s^2")
        print(f"Hull bound excess for this trajectory: "
              f"{100.0 * (continuous_bound / grid_peak - 1.0):.4f}%")
    print(f"Pythagorean identity max error: {identity_error:.3e}")
    print(f"Endpoint errors: position={endpoint_position_error:.3e}, "
          f"velocity={endpoint_velocity_error:.3e}")
    print(f"Minimum hull weight (must be positive): {minimum_weight:.3e}")
    print(f"Minimum normalized hull residual: {np.min(residuals):.3e}")
    print(f"Numerical checks passed: {checks_passed}")
    print(f"Results: {args.output.resolve()}")
    if not args.no_plots:
        plot_solution(args, time, (rho, theta, z), (ur, ut, uz, norm), limit,
                      summary["delta_v_reference_km_s"])
    if not checks_passed:
        raise RuntimeError("Post-solve numerical checks failed; inspect solution.json")
    return summary


def span_physical_state(basis, controls, arc, mu, sigma):
    """Symbolic state at one spline parameter, differentiated with respect to s."""
    primitive = ca.DM(basis(sigma).reshape(1, -1)) @ controls
    primitive_d = ca.DM(basis(sigma, 1).reshape(1, -1)) @ controls / arc
    primitive_dd = ca.DM(basis(sigma, 2).reshape(1, -1)) @ controls / arc**2
    return physical_state(primitive[0, :], primitive_d[0, :],
                          primitive_dd[0, :], mu)


def main(default_thrust_enforcement="hull"):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("min-peak", "energy", "min-dv"),
                        default="min-peak")
    parser.add_argument("--thrust-enforcement", choices=("hull", "grid", "none"),
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
             "trajectory's hull peak bound",
    )
    parser.add_argument("--h-min", type=float, default=0.1)
    parser.add_argument("--rho-min", type=float, default=0.1)
    parser.add_argument("--r-min", type=float, default=0.1)
    parser.add_argument("--arc-min", type=float, default=1e-3,
                        help="Lower bound on the Sundman arc length as a "
                             "fraction of the seed arc length")
    parser.add_argument("--subdivide", type=int, default=0)
    parser.add_argument(
        "--quadrature", type=int, default=12,
        help="Gauss-Legendre order for energy objective; unused in min-peak mode",
    )
    parser.add_argument("--samples", type=int, default=2001)
    parser.add_argument("--max-iter", type=int, default=2500)
    parser.add_argument("--initial-solution", type=Path, default=None,
                        help="Warm-start controls from a previous solution.json")
    parser.add_argument("--quiet", action="store_true")
    plots = parser.add_mutually_exclusive_group()
    plots.add_argument("--show-plots", action="store_true")
    plots.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("output/pythagorean_3d_sundman"))
    args = parser.parse_args()
    if args.ephemeris == "spice":
        try:
            configure_spice_case(args)
        except (ValueError, FileNotFoundError, ImportError) as exc:
            parser.error(str(exc))
    if args.mode == "min-peak":
        if args.u_max is not None:
            parser.error("u-max does not apply to min-peak mode; omit it")
        if (not np.isfinite(args.peak_init_factor)
                or args.peak_init_factor <= 0):
            parser.error("peak-init-factor must be positive")
    elif args.mode == "min-dv":
        if args.u_max is not None and (not np.isfinite(args.u_max)
                                       or args.u_max <= 0):
            parser.error("u-max must be positive when given in min-dv mode")
    elif args.u_max is None or not np.isfinite(args.u_max) or args.u_max <= 0:
        parser.error("energy mode requires a positive u-max")
    positive = (args.mu, args.r0, args.rf, args.tof,
                args.h_min, args.rho_min, args.r_min, args.arc_min)
    if any(not np.isfinite(value) or value <= 0 for value in positive):
        parser.error("mu, radii, tof, and minimum radii must be positive")
    finite = (args.phi0, args.phif, args.rdot0, args.rdotf, args.phidot0,
              args.phidotf, args.theta_final)
    if any(not np.isfinite(value) for value in finite):
        parser.error("endpoint angles and rates must be finite")
    if args.degree < 2:
        parser.error("degree must be at least 2; the thrust numerators need a "
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

"""Alternative Sundman shaping families with the rational-Bezier hull bound.

``optimizer.pythagorean_3d_sundman`` shapes four primitives ``(h, p, q, theta)``.
This module keeps everything else from that solver -- the Sundman variable
``dt = R ds``, the exact Bernstein transfer-time constraint, the analytic
endpoint elimination, and the coefficient-wise hull condition
``||P_i||^2 <= u_max^2 W_i^2`` with ``W = R^3`` -- and only swaps the primitive
map.  Two families are available.

``two-spline``  (primitives ``p, q, theta``)
    ``rho + i z = (p + i q)^2``, so ``rho = p^2 - q^2``, ``z = 2 p q`` and
    ``R = p^2 + q^2``.  It is the existing map with ``h == 1``.  The geometry has
    degree ``2n`` and the hull degree is ``6n`` instead of ``9n``.

``ks``  (primitives ``u1, u2, u3, u4``)
    Kustaanheimo-Stiefel coordinates, ``x = L(u) u`` and ``R = |u|^2``.  The
    thrust is Cartesian, ``a = (R x'' - R' x' + mu x) / R^3``, with numerators of
    degree ``4n-2`` and the same hull degree ``6n``.  The revolution count is not
    an endpoint value here; it is carried by the continuous lift of the seed.
    A point of space is a circle (fibre) of KS points.  The start point is
    fixed on its fibre; with ``--fibre free`` the final fibre angle and the
    fibre components of both endpoint rates are decision variables, which
    leaves the Cartesian boundary conditions untouched.

    ``--revolution-lock`` makes the count strict.  With
    ``A = (u1 + u3) + i (u2 + u4)`` and ``B = (u1 - u3) + i (u2 - u4)``, both
    linear in ``u``, the map reads ``x + i y = A B``, so the swept angle is the
    winding of ``A`` plus that of ``B``.  Every Bezier piece of ``A`` and ``B``
    is kept in a half-plane taken from the seed (linear inequalities on the
    controls); by the convex-hull property neither can then gain or lose a
    turn, and the path cannot cross the z axis.

Neither is a faster solver for the trajectories of the existing module: each
spline space reaches a different set of trajectories.

Run from the repository root, for example::

    python -m optimizer.sundman_hull_families --family two-spline --quiet
"""

from __future__ import annotations

import argparse
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
from optimizer.pythagorean_3d_bernstein import (
    CANONICAL_ACCEL_TO_MM_S2,
    configure_spice_case,
    departure_epoch,
    primitive_endpoint_data,
    scale,
)
from optimizer.pythagorean_3d_sundman import (
    SundmanTimeMap,
    arc_length_for_tof,
    build_seed,
    sundman_hull_function,
)


def _sundman_numerator(radius, radius_d, coordinate, arc, mu):
    """Bernstein coefficients of ``R f'' - R' f' + mu f``."""
    first = derivative(coordinate, arc)
    second = derivative(first, arc)
    result = add(multiply(radius, second), multiply(radius_d, first), -1.0)
    return add(result, scale(coordinate, mu))


class TwoSplineFamily:
    """``rho + i z = (p + i q)^2`` with the longitude as a third primitive."""

    name = "two-spline"
    components = ("p", "q", "theta")
    coordinates = ("rho", "theta", "z")
    geometry = ("rho",)

    @staticmethod
    def bernstein(coefficients, arc, mu):
        p, q, theta = coefficients
        p2, q2 = multiply(p, p), multiply(q, q)
        rho = add(p2, q2, -1.0)
        z = scale(multiply(p, q), 2.0)
        radius = add(p2, q2)
        rho_d = derivative(rho, arc)
        rho_dd = derivative(rho_d, arc)
        radius_d = derivative(radius, arc)
        theta_d = derivative(theta, arc)
        theta_dd = derivative(theta_d, arc)
        # P_rho = R rho'' - R' rho' - R rho theta'^2 + mu rho
        p_rho = add(multiply(radius, rho_dd), multiply(radius_d, rho_d), -1.0)
        p_rho = add(p_rho,
                    multiply(radius, multiply(rho, multiply(theta_d, theta_d))),
                    -1.0)
        p_rho = add(p_rho, scale(rho, mu))
        # P_theta = rho (R theta'' - R' theta') + 2 R rho' theta'
        swing = add(multiply(radius, theta_dd), multiply(radius_d, theta_d), -1.0)
        p_theta = add(multiply(rho, swing),
                      scale(multiply(radius, multiply(rho_d, theta_d)), 2.0))
        p_z = _sundman_numerator(radius, radius_d, z, arc, mu)
        return (p_rho, p_theta, p_z), radius, {"rho": rho}

    @staticmethod
    def point(primitive, primitive_d, primitive_dd, mu):
        p, q, theta = primitive
        pd, qd, theta_d = primitive_d
        pdd, qdd, theta_dd = primitive_dd
        rho = p**2 - q**2
        rho_d = 2 * (p * pd - q * qd)
        rho_dd = 2 * (pd**2 + p * pdd - qd**2 - q * qdd)
        z = 2 * p * q
        z_d = 2 * (pd * q + p * qd)
        z_dd = 2 * (pdd * q + 2 * pd * qd + p * qdd)
        radius = p**2 + q**2
        radius_d = 2 * (p * pd + q * qd)
        weight = radius**3
        u_rho = (radius * rho_dd - radius_d * rho_d
                 - radius * rho * theta_d**2 + mu * rho) / weight
        u_theta = (rho * (radius * theta_dd - radius_d * theta_d)
                   + 2 * radius * rho_d * theta_d) / weight
        u_z = (radius * z_dd - radius_d * z_d + mu * z) / weight
        return {
            "position": (rho, theta, z),
            "rate": (rho_d / radius, theta_d / radius, z_d / radius),
            "radius": radius,
            "control": (u_rho, u_theta, u_z),
        }

    @staticmethod
    def cartesian(position):
        rho, theta, z = position
        return rho * np.cos(theta), rho * np.sin(theta), z

    @staticmethod
    def endpoint_state(args):
        _, _, values, rates = primitive_endpoint_data(args)
        return values, rates

    @staticmethod
    def lift(cylindrical):
        rho, theta, z = np.asarray(cylindrical, dtype=float).T
        radius = np.hypot(rho, z)
        half = 0.5 * np.unwrap(np.arctan2(z, rho))
        root = np.sqrt(radius)
        return np.column_stack((root * np.cos(half), root * np.sin(half), theta))

    @staticmethod
    def _primitive_ends(args):
        radii = np.array([args.r0, args.rf], dtype=float)
        half = 0.5 * np.array([args.phi0, args.phif], dtype=float)
        radius_rate = np.array([args.rdot0, args.rdotf], dtype=float)
        half_rate = 0.5 * np.array([args.phidot0, args.phidotf], dtype=float)
        _, _, values, rates = primitive_endpoint_data(args)
        root = np.sqrt(radii)
        root_rate = radius_rate / (2.0 * root)
        primitive_values = np.column_stack((
            root * np.cos(half), root * np.sin(half), values[:, 1]))
        primitive_rates = np.column_stack((
            root_rate * np.cos(half) - root * np.sin(half) * half_rate,
            root_rate * np.sin(half) + root * np.cos(half) * half_rate,
            rates[:, 1]))
        return primitive_values, primitive_rates

    def seed_ends(self, args, lifted):
        return self._primitive_ends(args)

    def clamp(self, opti, args, basis, seed, seed_arc, fibre):
        values, rates = self._primitive_ends(args)
        rows = [ca.MX(ca.DM(row.reshape(1, -1)))
                for row in (values[0], rates[0], values[1], rates[1])]
        return (*rows, {})


def ks_form(a, b):
    """Symmetric bilinear KS form: ``(x1, x2, x3, R)`` when ``a == b``."""
    return (
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] + a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] - a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] + a[2] * b[0] + a[1] * b[3] + a[3] * b[1],
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3],
    )


def ks_tangent(u):
    """Fibre tangent: moving along it leaves ``x = L(u) u`` unchanged."""
    return [-u[3], u[2], -u[1], u[0]]


def ks_rate(u, velocity, radius, fibre_rate):
    """Physical-time rate ``L(u)^T v / (2R)`` plus a free fibre component."""
    v1, v2, v3 = velocity
    transpose = [
        u[0] * v1 + u[1] * v2 + u[2] * v3,
        -u[1] * v1 + u[0] * v2 + u[3] * v3,
        -u[2] * v1 - u[3] * v2 + u[0] * v3,
        u[3] * v1 - u[2] * v2 + u[1] * v3,
    ]
    tangent = ks_tangent(u)
    return [transpose[i] / (2.0 * radius) + fibre_rate * tangent[i]
            for i in range(4)]


def ks_point_lift(x):
    """One KS point above a Cartesian position."""
    x = np.asarray(x, dtype=float)
    radius = float(np.linalg.norm(x))
    if x[0] >= 0.0:
        u1 = np.sqrt(0.5 * (radius + x[0]))
        return np.array([u1, x[1] / (2.0 * u1), x[2] / (2.0 * u1), 0.0])
    u2 = np.sqrt(0.5 * (radius - x[0]))
    return np.array([x[1] / (2.0 * u2), u2, 0.0, x[2] / (2.0 * u2)])


def ks_project(reference, x):
    """Point of the fibre above ``x`` that is closest to ``reference``."""
    lifted = ks_point_lift(x)
    tangent = np.asarray(ks_tangent(lifted))
    angle = np.arctan2(float(reference @ tangent), float(reference @ lifted))
    return np.cos(angle) * lifted + np.sin(angle) * tangent


def ks_lift(xyz):
    """Continuous lift of a sampled path; each step moves normal to the fibre."""
    xyz = np.asarray(xyz, dtype=float)
    lifted = np.empty((len(xyz), 4))
    lifted[0] = ks_point_lift(xyz[0])
    for index in range(1, len(xyz)):
        lifted[index] = ks_project(lifted[index - 1], xyz[index])
    return lifted


class KSFamily:
    """Kustaanheimo-Stiefel coordinates with a Cartesian thrust vector."""

    name = "ks"
    components = ("u1", "u2", "u3", "u4")
    coordinates = ("x", "y", "z")
    geometry = ()

    @staticmethod
    def bernstein(coefficients, arc, mu):
        u1, u2, u3, u4 = coefficients
        squares = [multiply(c, c) for c in coefficients]
        x1 = add(add(squares[0], squares[3]), add(squares[1], squares[2]), -1.0)
        x2 = scale(add(multiply(u1, u2), multiply(u3, u4), -1.0), 2.0)
        x3 = scale(add(multiply(u1, u3), multiply(u2, u4)), 2.0)
        radius = add(add(squares[0], squares[3]), add(squares[1], squares[2]))
        radius_d = derivative(radius, arc)
        numerators = tuple(_sundman_numerator(radius, radius_d, x, arc, mu)
                           for x in (x1, x2, x3))
        return numerators, radius, {}

    @staticmethod
    def point(primitive, primitive_d, primitive_dd, mu):
        *x, radius = ks_form(primitive, primitive)
        first = ks_form(primitive, primitive_d)
        mixed = ks_form(primitive_d, primitive_d)
        second = ks_form(primitive, primitive_dd)
        radius_d = 2 * first[3]
        weight = radius**3
        x_d = [2 * first[i] for i in range(3)]
        x_dd = [2 * (mixed[i] + second[i]) for i in range(3)]
        control = tuple((radius * x_dd[i] - radius_d * x_d[i] + mu * x[i]) / weight
                        for i in range(3))
        return {
            "position": tuple(x),
            "rate": tuple(x_d[i] / radius for i in range(3)),
            "radius": radius,
            "control": control,
        }

    @staticmethod
    def cartesian(position):
        return position

    @staticmethod
    def endpoint_state(args):
        _, _, values, rates = primitive_endpoint_data(args)
        rho, theta, z = values.T
        rho_d, theta_d, z_d = rates.T
        cosine, sine = np.cos(theta), np.sin(theta)
        position = np.column_stack((rho * cosine, rho * sine, z))
        velocity = np.column_stack((
            rho_d * cosine - rho * theta_d * sine,
            rho_d * sine + rho * theta_d * cosine,
            z_d))
        return position, velocity

    @staticmethod
    def lift(cylindrical):
        rho, theta, z = np.asarray(cylindrical, dtype=float).T
        return ks_lift(np.column_stack((rho * np.cos(theta), rho * np.sin(theta), z)))

    def seed_ends(self, args, lifted):
        position, velocity = self.endpoint_state(args)
        radii = np.linalg.norm(position, axis=1)
        values = np.vstack((ks_project(lifted[0], position[0]),
                            ks_project(lifted[-1], position[1])))
        rates = np.vstack([ks_rate(values[i], velocity[i], radii[i], 0.0)
                           for i in range(2)])
        return values, rates

    def clamp(self, opti, args, basis, seed, seed_arc, fibre):
        position, velocity = self.endpoint_state(args)
        radii = np.linalg.norm(position, axis=1)
        anchors = (ks_project(seed[0], position[0]),
                   ks_project(seed[-1], position[1]))
        # Fibre component of the endpoint rates carried by the seed controls.
        seed_rates = (basis(0.0, 1) @ seed / (seed_arc * radii[0]),
                      basis(1.0, 1) @ seed / (seed_arc * radii[1]))
        initial = [float(seed_rates[i] @ np.asarray(ks_tangent(anchors[i])))
                   / radii[i] for i in range(2)]
        if fibre == "free":
            extras = {"fibre_rate_start": opti.variable(),
                      "fibre_angle_end": opti.variable(),
                      "fibre_rate_end": opti.variable()}
            opti.set_initial(extras["fibre_rate_start"], initial[0])
            opti.set_initial(extras["fibre_angle_end"], 0.0)
            opti.set_initial(extras["fibre_rate_end"], initial[1])
            rate_start, angle, rate_end = extras.values()
        else:
            extras, rate_start, angle, rate_end = {}, 0.0, 0.0, 0.0
        start = [float(value) for value in anchors[0]]
        end_tangent = ks_tangent(anchors[1])
        end = [ca.cos(angle) * float(anchors[1][i])
               + ca.sin(angle) * float(end_tangent[i]) for i in range(4)]
        rows = (start, ks_rate(start, velocity[0], radii[0], rate_start),
                end, ks_rate(end, velocity[1], radii[1], rate_end))
        return (*[ca.horzcat(*[ca.MX(value) for value in row]) for row in rows],
                extras)


FAMILIES = {family.name: family for family in (TwoSplineFamily, KSFamily)}


def ks_factors(u):
    """Complex factors of ``x + i y = A B``; columns Re A, Im A, Re B, Im B."""
    return (u[:, 0] + u[:, 2], u[:, 1] + u[:, 3],
            u[:, 0] - u[:, 2], u[:, 1] - u[:, 3])


def revolution_lock_plan(reference_sigma, reference, spans, max_sweep,
                         max_depth=7):
    """Half-plane directions for every Bezier piece of the KS factors.

    ``reference`` is a KS path sampled at ``reference_sigma`` that has the
    wanted revolution count: the lifted seed trajectory, not its spline fit,
    which may already have lost turns when there are few controls.  The pieces
    come from ``depth`` midpoint subdivisions of each span, with ``depth`` the
    smallest value for which the reference turns ``A`` and ``B`` by at most
    ``max_sweep`` on any piece.  Returns ``depth`` and an array
    ``directions[span, piece, factor]`` of angles (factor 0 is A, 1 is B).
    """
    per_span = 2**max_depth * 16
    sigma = np.linspace(0.0, 1.0, spans * per_span + 1)
    a_re, a_im, b_re, b_im = ks_factors(np.asarray(reference, dtype=float))
    sampled = (np.unwrap(np.arctan2(a_im, a_re)), np.unwrap(np.arctan2(b_im, b_re)))
    if max(np.max(np.abs(np.diff(angle))) for angle in sampled) > 0.5:
        raise ValueError("revolution lock: the reference path is sampled too "
                         "coarsely or passes too close to the z axis")
    angles = np.column_stack([np.interp(sigma, reference_sigma, angle)
                              for angle in sampled])
    for depth in range(max_depth + 1):
        pieces = 2**depth
        step = per_span // pieces
        windows = np.stack([angles[i * step:(i + 1) * step + 1]
                            for i in range(spans * pieces)])
        low, high = windows.min(axis=1), windows.max(axis=1)
        if np.max(high - low) <= max_sweep:
            return depth, (0.5 * (low + high)).reshape(spans, pieces, 2)
    raise ValueError("revolution lock: the reference turns too fast; add controls")


def revolution_lock_margins(extractions, controls, degree, depth, directions):
    """Signed distance of every factor coefficient from its half-plane edge.

    Works on numeric and CasADi controls; all entries must stay positive.
    """
    symbolic = isinstance(controls, (ca.MX, ca.SX))
    margins = []
    for span, extraction in enumerate(extractions):
        bezier = (ca.DM(extraction) if symbolic else extraction) @ controls
        for piece, matrix in enumerate(subdivision_matrices(degree, depth)):
            local = (ca.DM(matrix) if symbolic else matrix) @ bezier
            a_re, a_im, b_re, b_im = ks_factors(local)
            angle_a, angle_b = directions[span, piece]
            margins.append(np.cos(angle_a) * a_re + np.sin(angle_a) * a_im)
            margins.append(np.cos(angle_b) * b_re + np.sin(angle_b) * b_im)
    return ca.vertcat(*margins) if symbolic else np.concatenate(margins)


def family_hull_function(family, degree):
    """Span coefficients of the thrust numerators, ``W = R^3`` and the geometry."""
    symbols = [ca.SX.sym(name, degree + 1) for name in family.components]
    arc = ca.SX.sym("arc")
    mu = ca.SX.sym("mu")
    coefficients = [[symbol[i] for i in range(degree + 1)] for symbol in symbols]
    numerators, radius, extra = family.bernstein(coefficients, arc, mu)
    weight = multiply(radius, multiply(radius, radius))
    hull_degree = max(len(block) for block in (*numerators, weight)) - 1
    outputs = [ca.vertcat(*elevate(block, hull_degree)) for block in numerators]
    outputs += [ca.vertcat(*elevate(weight, hull_degree)), ca.vertcat(*radius)]
    outputs += [ca.vertcat(*extra[name]) for name in family.geometry]
    name = family.name.replace("-", "_")
    return ca.Function(f"sundman_{name}_hull_p{degree}",
                       [*symbols, arc, mu], outputs), hull_degree


def _span_blocks(hull, bezier, arc, mu):
    columns = [bezier[:, j] for j in range(bezier.shape[1])]
    return hull(*columns, arc, mu)


def span_radius_coefficients(extractions, controls, hull, mu):
    """Bernstein coefficients of R on each span; independent of the arc length."""
    return [np.asarray(_span_blocks(hull, extraction @ controls, 1.0, mu)[4])
            .reshape(-1) for extraction in extractions]


def hull_peak_squared(extractions, controls, hull, subdivisions, arc, spans, mu):
    """Smallest squared peak certified by the hull condition, and min weight."""
    peak, minimum_weight = 0.0, float("inf")
    for extraction in extractions:
        result = _span_blocks(hull, extraction @ controls, arc / spans, mu)
        blocks = [np.asarray(result[i]).reshape(-1) for i in range(4)]
        for matrix in subdivisions:
            first, second, third, weight = (matrix @ block for block in blocks)
            minimum_weight = min(minimum_weight, float(np.min(weight)))
            if np.any(weight <= 0):
                peak = float("inf")
            elif np.isfinite(peak):
                peak = max(peak, float(np.max(
                    (first**2 + second**2 + third**2) / weight**2)))
    return peak, minimum_weight


def evaluate_dense(family, basis, controls, arc, mu, sigma):
    """Evaluate the trajectory at spline parameters, not at uniform times."""
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    primitive = basis(sigma) @ controls
    primitive_d = basis(sigma, 1) @ controls / arc
    primitive_dd = basis(sigma, 2) @ controls / arc**2
    return family.point(*(tuple(block.T) for block in
                          (primitive, primitive_d, primitive_dd)), mu)


def symbolic_point(family, basis, controls, arc, mu, sigma):
    count = len(family.components)
    rows = [ca.DM(basis(sigma, order).reshape(1, -1)) @ controls / arc**order
            for order in range(3)]
    return family.point(*([row[0, j] for j in range(count)] for row in rows), mu)


def sundman_parameter(tau, cylindrical, tof):
    """Normalized Sundman parameter of a path sampled in physical time."""
    rho, _, z = np.asarray(cylindrical, dtype=float).T
    radius = np.hypot(rho, z)
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("seed trajectory has a nonpositive or invalid radius")
    increments = float(tof) * np.diff(tau) * 0.5 * (
        1.0 / radius[:-1] + 1.0 / radius[1:])
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    arc = float(cumulative[-1])
    sigma = cumulative / arc
    sigma[0], sigma[-1] = 0.0, 1.0
    return sigma, arc


def fit_family_seed(family, basis, sigma, cylindrical, arc, args):
    """Least-squares controls of one family for a path sampled in ``sigma``."""
    lifted = family.lift(cylindrical)
    values, rates = family.seed_ends(args, lifted)
    radii = np.array([args.r0, args.rf], dtype=float)
    start_d, end_d = basis(0.0, 1), basis(1.0, 1)
    controls = np.zeros((len(basis.c), lifted.shape[1]))
    controls[0] = values[0]
    controls[1] = (arc * radii[0] * rates[0] - start_d[0] * controls[0]) / start_d[1]
    controls[-1] = values[1]
    controls[-2] = (arc * radii[1] * rates[1] - end_d[-1] * controls[-1]) / end_d[-2]
    design = np.asarray(basis(sigma), dtype=float)
    if len(basis.c) > 4:
        fixed = design[:, :2] @ controls[:2] + design[:, -2:] @ controls[-2:]
        controls[2:-2] = np.linalg.lstsq(
            design[:, 2:-2], lifted - fixed, rcond=None)[0]
    fitted = design @ controls
    error = np.linalg.norm(
        np.column_stack(family.cartesian(family.point(
            tuple(fitted.T), tuple(fitted.T), tuple(fitted.T), 0.0)["position"]))
        - np.column_stack(TwoSplineFamily.cartesian(
            np.asarray(cylindrical, dtype=float).T)), axis=1)
    return controls, float(np.max(error)), lifted


def load_seed(family, basis, args):
    """Seed controls, arc length, optional peak bound, description, reference.

    The reference is ``(sigma, primitives)``: the sampled path the controls
    were fitted to, or the seed spline itself for a saved solution.
    """
    if getattr(args, "initial_solution", None) is not None:
        payload = json.loads(Path(args.initial_solution).read_text())
        if payload.get("family") != family.name:
            raise ValueError("initial-solution belongs to a different family")
        controls = np.asarray(payload["control_coefficients"], dtype=float)
        expected = (args.n_ctrl, len(family.components))
        if controls.shape != expected or not np.all(np.isfinite(controls)):
            raise ValueError("initial-solution has incompatible controls")
        peak = payload.get("required_peak_acceleration_bound")
        sigma = np.linspace(0.0, 1.0, 200 * args.n_ctrl + 1)
        return (controls, float(payload["sundman_arc_length"]), peak,
                {"source": "initial-solution", "path": str(args.initial_solution)},
                (sigma, basis(sigma) @ controls))

    if getattr(args, "seed_trajectory", None) is not None:
        with np.load(args.seed_trajectory) as archive:
            tau = np.asarray(archive["tau"], dtype=float).reshape(-1)
            cylindrical = np.asarray(archive["cylindrical_position"], dtype=float)
        sigma, arc = sundman_parameter(tau, cylindrical, args.tof)
        source = {"source": "seed-trajectory", "path": str(args.seed_trajectory)}
    else:
        # Same cubic-Hermite guess as the (h, p, q, theta) solver, resampled.
        reference_basis = spline_basis(max(args.n_ctrl, 4), min(args.degree, 3))
        degree = int(reference_basis.k)
        spans = len(reference_basis.c) - degree
        extractions = [
            extraction_matrix(reference_basis, span / spans, (span + 1) / spans,
                              degree) for span in range(spans)]
        values, rates, _, _ = primitive_endpoint_data(args)
        reference, arc = build_seed(
            reference_basis, values, rates, np.array([args.r0, args.rf]),
            args.tof, args.mu, extractions, sundman_hull_function(degree)[0])
        sigma = np.linspace(0.0, 1.0, 2001)
        h, p, q, theta = (reference_basis(sigma) @ reference).T
        cylindrical = np.column_stack((h * (p**2 - q**2), theta, 2.0 * h * p * q))
        source = {"source": "hermite"}
    controls, error, lifted = fit_family_seed(
        family, basis, sigma, cylindrical, arc, args)
    source["maximum_position_fit_error"] = error
    return (controls, arc, getattr(args, "initial_peak_bound", None), source,
            (sigma, lifted))


def solve(args):
    solve_started = perf_counter()
    family = FAMILIES[args.family]()
    count = len(family.components)
    basis = spline_basis(args.n_ctrl, args.degree)
    spans = args.n_ctrl - args.degree
    radius_endpoints = np.array([args.r0, args.rf], dtype=float)
    hull, hull_degree = family_hull_function(family, args.degree)
    hull_subdivisions = subdivision_matrices(hull_degree, args.subdivide)
    geometry_subdivisions = subdivision_matrices(2 * args.degree, args.subdivide)
    extractions = [
        extraction_matrix(basis, span / spans, (span + 1) / spans, args.degree)
        for span in range(spans)
    ]
    seed, seed_arc, initial_peak_bound, seed_source, reference = load_seed(
        family, basis, args)
    if initial_peak_bound is not None:
        initial_peak_bound = float(initial_peak_bound)
        if not np.isfinite(initial_peak_bound) or initial_peak_bound <= 0.0:
            initial_peak_bound = None

    opti = ca.Opti()
    arc = opti.variable()
    opti.set_initial(arc, seed_arc)
    opti.subject_to(arc >= args.arc_min * seed_arc)
    start, start_rate, end, end_rate, extras = family.clamp(
        opti, args, basis, seed, seed_arc, args.fibre)
    start_d, end_d = basis(0.0, 1), basis(1.0, 1)
    second = (arc * radius_endpoints[0] * start_rate - start_d[0] * start) / start_d[1]
    penultimate = (arc * radius_endpoints[1] * end_rate - end_d[-1] * end) / end_d[-2]
    n_free = args.n_ctrl - 4
    if n_free:
        free = opti.variable(n_free, count)
        opti.set_initial(free, seed[2:-2])
        controls = ca.vertcat(start, second, free, penultimate, end)
    else:
        controls = ca.vertcat(start, second, penultimate, end)
    projected_seed = np.asarray(
        opti.debug.value(controls, opti.initial())).reshape(args.n_ctrl, count)

    lock = None
    if getattr(args, "revolution_lock", False):
        depth, directions = revolution_lock_plan(
            *reference, spans, np.deg2rad(args.revolution_lock_sweep_deg))
        seed_margins = revolution_lock_margins(
            extractions, projected_seed, args.degree, depth, directions)
        floor = args.revolution_lock_margin * float(np.sqrt(args.r_min))
        lock = {"depth": depth, "directions": directions, "floor": floor,
                "seed_minimum_margin": float(np.min(seed_margins))}
        opti.subject_to(revolution_lock_margins(
            extractions, controls, args.degree, depth, directions) >= floor)

    minimize_peak = args.mode == "min-peak"
    minimize_delta_v = args.mode == "min-dv"
    thrust_enforcement = args.thrust_enforcement
    if minimize_delta_v and args.u_max is None:
        thrust_enforcement = "none"
    seed_peak_squared, _ = hull_peak_squared(
        extractions, projected_seed, hull, hull_subdivisions, seed_arc, spans,
        args.mu)
    seed_scale = (seed_peak_squared if np.isfinite(seed_peak_squared)
                  and seed_peak_squared > 0 else 1.0)
    peak_ratio_squared = None
    if minimize_peak:
        reference_squared = seed_scale
        peak_ratio_squared = opti.variable()
        opti.subject_to(peak_ratio_squared >= 0)
        peak_squared = reference_squared * peak_ratio_squared
    else:
        reference_squared = args.u_max**2 if args.u_max is not None else seed_scale
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
        result = _span_blocks(hull, ca.DM(extractions[span]) @ controls,
                              arc / spans, args.mu)
        first, second_block, third, weight = (result[i] for i in range(4))
        radius_coeff = result[4]
        total_time += (arc / spans) * ca.sum1(radius_coeff) / radius_coeff.shape[0]
        for matrix in hull_subdivisions:
            block = (peak_squared * (ca.DM(matrix) @ weight)**2
                     - (ca.DM(matrix) @ first)**2
                     - (ca.DM(matrix) @ second_block)**2
                     - (ca.DM(matrix) @ third)**2) / scale_hull
            if thrust_enforcement == "hull":
                opti.subject_to(block >= 0)
            residual_blocks.append(block)
        for matrix in geometry_subdivisions:
            opti.subject_to(ca.DM(matrix) @ radius_coeff >= args.r_min)
            for offset, name in enumerate(family.geometry):
                opti.subject_to(ca.DM(matrix) @ result[5 + offset]
                                >= getattr(args, f"{name}_min"))

        if thrust_enforcement == "grid":
            span_nodes = np.linspace(left, right, args.constraint_nodes_per_span)
            if span < spans - 1:
                span_nodes[-1] = np.nextafter(right, left)
            for sigma in span_nodes:
                control = symbolic_point(
                    family, basis, controls, arc, args.mu, sigma)["control"]
                opti.subject_to(sum(value**2 for value in control) <= peak_squared)
                grid_constraint_sigma.append(float(sigma))

        for node, weight_node in zip(nodes, weights):
            sigma = left + 0.5 * (node + 1) / spans
            state = symbolic_point(family, basis, controls, arc, args.mu, sigma)
            norm_squared = sum(value**2 for value in state["control"])
            jacobian_weight = weight_node * arc / (2 * spans) * state["radius"]
            energy += jacobian_weight * norm_squared
            if minimize_delta_v:
                node_bound = opti.variable()
                opti.subject_to(node_bound >= 0)
                opti.subject_to(norm_squared <= node_bound**2)
                opti.set_initial(node_bound, float(np.sqrt(reference_squared)))
                delta_v_expression += jacobian_weight * node_bound
                delta_v_node_count += 1

    opti.subject_to(total_time == args.tof)
    initial_peak_ratio_squared = None
    if minimize_peak:
        opti.minimize(peak_ratio_squared)
        initial_peak_ratio_squared = (
            float(args.peak_init_factor) if initial_peak_bound is None
            else initial_peak_bound**2 / reference_squared)
        opti.set_initial(peak_ratio_squared, initial_peak_ratio_squared)
    elif args.mode == "energy":
        opti.minimize(energy / (reference_squared * args.tof))
    else:
        opti.minimize(delta_v_expression
                      / (float(np.sqrt(reference_squared)) * args.tof))

    opti.solver("ipopt", {"print_time": not args.quiet}, {
        "max_iter": args.max_iter,
        "tol": 1e-9,
        "constr_viol_tol": 1e-9,
        "acceptable_iter": 0,
        "bound_relax_factor": 0,
        "print_level": 0 if args.quiet else 5,
    })
    thrust_count = (
        (hull_degree + 1) * spans * 2**args.subdivide
        if thrust_enforcement == "hull"
        else spans * args.constraint_nodes_per_span
        if thrust_enforcement == "grid" else 0)
    print(f"Degree-{args.degree} Sundman {family.name}: {args.n_ctrl} controls, "
          f"{spans} spans, {count * n_free} trajectory variables, "
          f"{len(extras)} fibre variables, hull degree {hull_degree}, "
          f"{thrust_count} {thrust_enforcement} thrust inequalities.")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        solution = opti.solve()
    except RuntimeError as exc:
        status = opti.debug.stats().get("return_status", "Solver_Error")
        (args.output / "solution.json").write_text(json.dumps({
            "status": status, "family": family.name,
            "numerical_checks_passed": False,
            "wall_time_s": perf_counter() - solve_started,
            "ipopt_iterations": int(opti.debug.stats().get("iter_count", -1)),
            "seed": seed_source,
        }, indent=2) + "\n")
        raise RuntimeError(f"IPOPT failed: {status}") from exc
    solver_stats = solution.stats()
    solved_controls = np.asarray(solution.value(controls)).reshape(args.n_ctrl, count)
    solved_arc = float(solution.value(arc))
    solved_extras = {name: float(solution.value(variable))
                     for name, variable in extras.items()}
    residuals = np.asarray(
        solution.value(ca.vertcat(*residual_blocks))).reshape(-1)
    continuous_bound_squared, minimum_weight = hull_peak_squared(
        extractions, solved_controls, hull, hull_subdivisions, solved_arc, spans,
        args.mu)
    continuous_bound = float(np.sqrt(max(0.0, continuous_bound_squared)))
    if minimize_peak:
        limit = float(np.sqrt(max(0.0, float(solution.value(peak_squared)))))
    elif args.u_max is not None:
        limit = float(args.u_max)
    else:
        limit = continuous_bound

    time_map = SundmanTimeMap(
        span_radius_coefficients(extractions, solved_controls, hull, args.mu),
        solved_arc)
    tof_error = abs(time_map.total_time - args.tof)
    sigma = np.linspace(0.0, 1.0, args.samples)
    dense = evaluate_dense(family, basis, solved_controls, solved_arc, args.mu, sigma)
    control = np.column_stack(dense["control"])
    norm = np.linalg.norm(control, axis=1)
    radius = np.asarray(dense["radius"])
    position = np.column_stack(dense["position"])
    rate = np.column_stack(dense["rate"])
    xyz = np.column_stack(family.cartesian(dense["position"]))
    time = time_map.time(sigma)
    if grid_constraint_sigma:
        grid_control = np.column_stack(evaluate_dense(
            family, basis, solved_controls, solved_arc, args.mu,
            np.unique(grid_constraint_sigma))["control"])
        grid_peak = float(np.max(np.linalg.norm(grid_control, axis=1)))
    else:
        grid_peak = float("nan")

    endpoint_position, endpoint_rate = family.endpoint_state(args)
    endpoint_position_error = float(np.max(np.abs(
        position[[0, -1]] - endpoint_position)))
    endpoint_velocity_error = float(np.max(np.abs(rate[[0, -1]] - endpoint_rate)))
    radius_identity_error = float(np.max(np.abs(
        radius - np.linalg.norm(xyz, axis=1))))
    tolerance = 1e-7 * max(1.0, limit)
    thrust_checks_passed = bool(
        thrust_enforcement == "none"
        or (thrust_enforcement == "hull" and np.min(residuals) >= -1e-7
            and minimum_weight > 0.0 and np.max(norm) <= limit + tolerance)
        or (thrust_enforcement == "grid" and grid_peak <= limit + tolerance))
    minimum_rho = float(np.min(np.hypot(xyz[:, 0], xyz[:, 1])))
    # Angle actually swept about the z axis against the requested one.
    azimuth = np.unwrap(np.arctan2(xyz[:, 1], xyz[:, 0]))
    swept_angle = float(azimuth[-1] - azimuth[0])
    requested_angle = float(np.diff(primitive_endpoint_data(args)[2][:, 1])[0])
    revolution_error = abs(swept_angle - requested_angle) / (2.0 * np.pi)
    lock_margin = None
    if lock is not None:
        lock_margin = float(np.min(revolution_lock_margins(
            extractions, solved_controls, args.degree, lock["depth"],
            lock["directions"])))
    checks_passed = bool(
        np.all(np.isfinite(solved_controls))
        and np.isfinite(solved_arc) and solved_arc > 0
        and np.min(radius) >= args.r_min - 1e-8
        and ("rho" not in family.geometry or minimum_rho >= args.rho_min - 1e-8)
        and thrust_checks_passed
        and revolution_error <= 1e-3
        and (lock_margin is None or lock_margin >= lock["floor"] * (1 - 1e-6))
        and radius_identity_error <= 1e-10
        and endpoint_position_error <= 1e-7
        and endpoint_velocity_error <= 1e-7
        and tof_error <= 1e-9 * max(1.0, args.tof))

    def acceleration_norm(normalized_time):
        sample_sigma = time_map.sigma(
            np.asarray(normalized_time, dtype=float) * time_map.total_time)
        sample = evaluate_dense(
            family, basis, solved_controls, solved_arc, args.mu, sample_sigma)
        return np.linalg.norm(np.column_stack(sample["control"]), axis=1)

    reference_metrics = evaluate_reference_metrics(
        acceleration_norm,
        delta_v_scale_km_s=args.tof * CANONICAL_DV_TO_KM_S,
        acceleration_scale_m_s2=CANONICAL_ACCEL_TO_M_S2,
    ).as_dict()
    wall_time = perf_counter() - solve_started

    summary = {
        "status": solver_stats["return_status"],
        "formulation": f"Sundman {family.name} rational-Bezier hull",
        "family": family.name,
        "components": list(family.components),
        "parameters": {**{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items() if not key.startswith("_")
        }, "output": str(args.output)},
        "spice_boundary_data": getattr(args, "_spice_context", None),
        "seed": seed_source,
        "hull_degree": hull_degree,
        "thrust_enforcement": thrust_enforcement,
        "sundman_arc_length": solved_arc,
        "sundman_arc_length_seed": seed_arc,
        "fibre_variables": solved_extras,
        "time_of_flight_error": tof_error,
        "minimum_hull_weight": minimum_weight,
        "number_of_free_trajectory_variables": count * n_free,
        "number_of_decision_variables": (
            count * n_free + 1 + len(extras) + int(minimize_peak)
            + delta_v_node_count),
        "number_of_thrust_inequalities": thrust_count,
        "required_peak_acceleration_bound": limit,
        "required_peak_acceleration_mm_s2": limit * CANONICAL_ACCEL_TO_MM_S2,
        "seed_hull_peak_bound_mm_s2":
            float(np.sqrt(seed_scale)) * CANONICAL_ACCEL_TO_MM_S2,
        "initial_peak_ratio_squared": initial_peak_ratio_squared,
        "continuous_bernstein_bound_for_returned_trajectory": continuous_bound,
        "continuous_bernstein_bound_for_returned_trajectory_mm_s2":
            continuous_bound * CANONICAL_ACCEL_TO_MM_S2,
        "minimum_normalized_bernstein_coefficient": float(np.min(residuals)),
        "sampled_peak_acceleration": float(np.max(norm)),
        "sampled_peak_acceleration_mm_s2":
            float(np.max(norm)) * CANONICAL_ACCEL_TO_MM_S2,
        "hull_bound_over_sampled_peak_percent":
            100.0 * (continuous_bound / float(np.max(norm)) - 1.0),
        "grid_node_peak_acceleration":
            grid_peak if thrust_enforcement == "grid" else None,
        "delta_v_nondimensional": float(np.trapezoid(norm, time)),
        **reference_metrics,
        "wall_time_s": wall_time,
        "ipopt_iterations": int(solver_stats.get("iter_count", -1)),
        "max_radius_identity_error": radius_identity_error,
        "max_endpoint_position_error": endpoint_position_error,
        "max_endpoint_velocity_error": endpoint_velocity_error,
        "minimum_radius": float(np.min(radius)),
        "minimum_rho": minimum_rho,
        "swept_revolutions": swept_angle / (2.0 * np.pi),
        "requested_revolutions": requested_angle / (2.0 * np.pi),
        "revolution_lock": None if lock is None else {
            "pieces_per_span": 2**lock["depth"],
            "inequalities": 2 * spans * 2**lock["depth"] * (args.degree + 1),
            "margin_floor": lock["floor"],
            "seed_minimum_margin": lock["seed_minimum_margin"],
            "solution_minimum_margin": lock_margin,
        },
        "numerical_checks_passed": checks_passed,
        "formal_interval_certificate": False,
        "control_coefficients": solved_controls.tolist(),
    }
    (args.output / "solution.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savetxt(
        args.output / "trajectory.csv",
        np.column_stack((time, sigma, xyz, radius, control, norm)),
        delimiter=",", comments="",
        header="time,sigma,x,y,z,R," + ",".join(
            f"u_{name}" for name in family.coordinates) + ",u_norm")

    print(f"IPOPT: {summary['status']}")
    print(f"Sundman arc length S: {solved_arc:.9g} (seed {seed_arc:.9g}), "
          f"TOF error {tof_error:.3e}")
    print(f"Delta V (reference): {summary['delta_v_reference_km_s']:.9g} km/s")
    print(f"Required peak acceleration bound: "
          f"{limit * CANONICAL_ACCEL_TO_MM_S2:.9g} mm/s^2")
    print(f"Sampled peak acceleration: "
          f"{np.max(norm) * CANONICAL_ACCEL_TO_MM_S2:.9g} mm/s^2")
    print(f"Hull bound required by the returned trajectory: "
          f"{continuous_bound * CANONICAL_ACCEL_TO_MM_S2:.9g} mm/s^2")
    print(f"Endpoint errors: position={endpoint_position_error:.3e}, "
          f"velocity={endpoint_velocity_error:.3e}")
    print(f"Minimum hull weight (must be positive): {minimum_weight:.3e}")
    print(f"Revolutions swept / requested: {swept_angle / (2.0 * np.pi):.6f} / "
          f"{requested_angle / (2.0 * np.pi):.6f}")
    print(f"Numerical checks passed: {checks_passed}")
    print(f"Results: {args.output.resolve()}")
    if not checks_passed:
        raise RuntimeError("Post-solve numerical checks failed; inspect solution.json")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=tuple(FAMILIES), default="two-spline")
    parser.add_argument("--fibre", choices=("free", "fixed"), default="free",
                        help="ks only: free or fixed endpoint fibre variables")
    parser.add_argument("--revolution-lock", action="store_true",
                        help="KS: half-plane constraints that fix the "
                             "revolution count for every sigma")
    parser.add_argument("--revolution-lock-sweep-deg", type=float, default=90.0,
                        help="largest seed turn of a KS factor per locked piece")
    parser.add_argument("--revolution-lock-margin", type=float, default=0.05,
                        help="half-plane offset in units of sqrt(r-min)")
    parser.add_argument("--mode", choices=("min-peak", "energy", "min-dv"),
                        default="min-peak")
    parser.add_argument("--thrust-enforcement", choices=("hull", "grid", "none"),
                        default="hull")
    parser.add_argument("--constraint-nodes-per-span", type=int, default=15)
    parser.add_argument("--ephemeris", choices=("circular", "spice"),
                        default="circular")
    parser.add_argument("--departure-date", type=departure_epoch,
                        default="2028-01-01")
    parser.add_argument("--tof-days", type=float, default=1000.0)
    parser.add_argument("--spice-meta-kernel",
                        default="kernels/fasttransfer_planets.tm")
    parser.add_argument("--spice-target-name", default="MARS BARYCENTER")
    parser.add_argument("--n-rev", type=int, default=0)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--n-ctrl", type=int, default=7)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--r0", type=float, default=1.0)
    parser.add_argument("--rf", type=float, default=1.5)
    parser.add_argument("--phi0", type=float, default=0.0)
    parser.add_argument("--phif", type=float, default=0.25)
    parser.add_argument("--rdot0", type=float, default=0.0)
    parser.add_argument("--rdotf", type=float, default=0.0)
    parser.add_argument("--phidot0", type=float, default=0.0)
    parser.add_argument("--phidotf", type=float, default=0.0)
    parser.add_argument("--theta-dot0", type=float, default=None)
    parser.add_argument("--theta-dotf", type=float, default=None)
    parser.add_argument("--theta-final", type=float, default=6.0)
    parser.add_argument("--tof", type=float, default=8.0)
    parser.add_argument("--u-max", type=float, default=None,
                        help="fixed cap: required in energy, optional in min-dv")
    parser.add_argument("--peak-init-factor", type=float, default=1.05)
    parser.add_argument("--initial-peak-bound", type=float, default=None,
                        help="min-peak: initial peak bound in canonical units")
    parser.add_argument("--rho-min", type=float, default=0.1)
    parser.add_argument("--r-min", type=float, default=0.1)
    parser.add_argument("--arc-min", type=float, default=1e-3)
    parser.add_argument("--subdivide", type=int, default=0)
    parser.add_argument("--quadrature", type=int, default=12)
    parser.add_argument("--samples", type=int, default=2001)
    parser.add_argument("--max-iter", type=int, default=2500)
    parser.add_argument("--seed-trajectory", type=Path, default=None,
                        help="npz with tau and cylindrical_position (rho, theta, z)")
    parser.add_argument("--initial-solution", type=Path, default=None,
                        help="Warm-start from a solution.json of the same family")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("output/sundman_hull_families"))
    return parser


def validate(args):
    if args.mode == "min-peak" and args.u_max is not None:
        raise ValueError("u-max does not apply to min-peak mode; omit it")
    if args.mode == "energy" and args.u_max is None:
        raise ValueError("energy mode requires a positive u-max")
    if args.u_max is not None and (not np.isfinite(args.u_max) or args.u_max <= 0):
        raise ValueError("u-max must be positive")
    positive = (args.mu, args.r0, args.rf, args.tof, args.rho_min, args.r_min,
                args.arc_min, args.peak_init_factor)
    if any(not np.isfinite(value) or value <= 0 for value in positive):
        raise ValueError("mu, radii, tof and lower bounds must be positive")
    if getattr(args, "revolution_lock", False):
        if args.family != "ks":
            raise ValueError("revolution-lock applies to the ks family only")
        if (not 0.0 < args.revolution_lock_sweep_deg < 180.0
                or args.revolution_lock_margin <= 0.0):
            raise ValueError("revolution-lock needs 0 < sweep < 180 and margin > 0")
    if args.degree < 2 or args.n_ctrl < max(4, args.degree + 1):
        raise ValueError("need degree >= 2 and n-ctrl >= max(4, degree + 1)")
    if (args.subdivide < 0 or args.samples < 2 or args.quadrature < 1
            or args.max_iter < 1 or args.n_rev < 0
            or args.constraint_nodes_per_span < 2):
        raise ValueError("invalid subdivision, sampling, quadrature or iterations")


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.ephemeris == "spice":
            configure_spice_case(args)
        validate(args)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        parser.error(str(exc))
    solve(args)


if __name__ == "__main__":
    main()

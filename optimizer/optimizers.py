"""Continuation from minimum-Delta-V cylindrical B-splines to Sundman shaping.

The pipeline first solves the nominal fixed-time cylindrical B-spline problem,
then reparametrizes that trajectory with ``dt = R ds`` and fits the four
Pythagorean primitives ``(h, p, q, theta)``.  The fitted controls, Sundman arc
length, and the B-spline peak acceleration are written to a seed JSON accepted
by :mod:`optimizer.pythagorean_3d_sundman`.

Run from the repository root, for example::

    python -m optimizer.optimizers --ephemeris spice \
      --departure-date 2028-06-24 --tof-days 460 --n-rev 1 --quiet

Use ``--prepare-only`` to stop after producing the continuation seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from optimizer.helpers_Bspline import build_bspline_matrices
from optimizer.optimization_Bspline_freetf import (
    CANONICAL_ACCEL_TO_M_S2,
    _bspline_basis_at_tau,
    fit_cylindrical_seed,
    solve_free_tf_cylindrical_bspline,
)
from optimizer.planar_cubic_bernstein import spline_basis
from optimizer.pythagorean_3d_bernstein import (
    configure_spice_case,
    departure_epoch,
    primitive_endpoint_data,
)
from optimizer.pythagorean_3d_sundman import solve as solve_sundman
from utils.utils import rv2mee


def cylindrical_endpoint_cartesian(args):
    """Return Cartesian endpoint states represented by the shaping arguments."""
    radii = np.asarray([args.r0, args.rf], dtype=float)
    latitude = np.asarray([args.phi0, args.phif], dtype=float)
    radius_rate = np.asarray([args.rdot0, args.rdotf], dtype=float)
    latitude_rate = np.asarray([args.phidot0, args.phidotf], dtype=float)
    theta = np.asarray([0.0, args.theta_final], dtype=float)
    default_theta_rate = np.sqrt(args.mu / radii**3) / np.cos(latitude)
    theta_rate = np.asarray([
        args.theta_dot0 if args.theta_dot0 is not None else default_theta_rate[0],
        args.theta_dotf if args.theta_dotf is not None else default_theta_rate[1],
    ])

    rho = radii * np.cos(latitude)
    z = radii * np.sin(latitude)
    rho_rate = radius_rate * np.cos(latitude) - radii * np.sin(latitude) * latitude_rate
    z_rate = radius_rate * np.sin(latitude) + radii * np.cos(latitude) * latitude_rate
    cosine, sine = np.cos(theta), np.sin(theta)
    position = np.column_stack((rho * cosine, rho * sine, z))
    velocity = np.column_stack((
        rho_rate * cosine - rho * theta_rate * sine,
        rho_rate * sine + rho * theta_rate * cosine,
        z_rate,
    ))
    return position, velocity


def mee_seed_profile(args, mee0, meef, thetaf_unwrapped, n_fine):
    """Sample the analytic MEE seed spline, independent of any optimization.

    Fitting the Pythagorean primitives to this profile instead of to the solved
    B-spline decouples the two families, so a poor B-spline local optimum is not
    propagated into the continuation seed.
    """
    mats = build_bspline_matrices(args.bspline_n_ctrl, args.bspline_degree, n_fine)
    control_points = fit_cylindrical_seed(
        mats, mee0, meef, args.tof, args.seed_profile, args.mu, thetaf_unwrapped)
    tau = np.linspace(0.0, 1.0, n_fine)
    b0, _, _ = _bspline_basis_at_tau(mats, tau)
    return tau, b0 @ control_points


def fit_pythagorean_sundman_seed(
    tau,
    cylindrical_position,
    tof,
    n_ctrl,
    degree,
    endpoint_values,
    endpoint_rates,
):
    """Fit Pythagorean controls after applying the exact Sundman reparametrization.

    ``cylindrical_position`` contains columns ``(rho, theta, z)`` sampled at
    normalized physical times ``tau``.  The endpoint value and physical-rate
    constraints are imposed analytically; only the interior controls are fitted.
    """
    tau = np.asarray(tau, dtype=float).reshape(-1)
    cylindrical_position = np.asarray(cylindrical_position, dtype=float)
    endpoint_values = np.asarray(endpoint_values, dtype=float)
    endpoint_rates = np.asarray(endpoint_rates, dtype=float)
    if cylindrical_position.shape != (len(tau), 3):
        raise ValueError("cylindrical_position must have shape (len(tau), 3)")
    if len(tau) < 2 or np.any(np.diff(tau) <= 0.0):
        raise ValueError("tau must be a strictly increasing sample grid")
    if degree < 2 or n_ctrl < max(4, degree + 1):
        raise ValueError("target basis needs degree >= 2 and at least max(4, degree + 1) controls")

    rho, theta, z = cylindrical_position.T
    radius = np.hypot(rho, z)
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("source trajectory has a nonpositive or invalid radius")

    # s(t) = integral dt/R and sigma=s/S.  The trapezoidal integration is
    # applied to the already dense source B-spline profile.
    increments = float(tof) * np.diff(tau) * 0.5 * (
        1.0 / radius[:-1] + 1.0 / radius[1:])
    sundman_s = np.concatenate(([0.0], np.cumsum(increments)))
    arc = float(sundman_s[-1])
    if not np.isfinite(arc) or arc <= 0.0:
        raise ValueError("computed Sundman arc length is not positive")
    sigma = sundman_s / arc
    sigma[0], sigma[-1] = 0.0, 1.0

    latitude = np.unwrap(np.arctan2(z, rho))
    samples = np.column_stack((
        radius,
        np.cos(0.5 * latitude),
        np.sin(0.5 * latitude),
        theta,
    ))

    basis = spline_basis(int(n_ctrl), int(degree))
    controls = np.zeros((int(n_ctrl), 4), dtype=float)
    start_d = np.asarray(basis(0.0, 1), dtype=float)
    end_d = np.asarray(basis(1.0, 1), dtype=float)
    controls[0] = endpoint_values[0]
    controls[1] = (
        arc * endpoint_values[0, 0] * endpoint_rates[0]
        - start_d[0] * controls[0]
    ) / start_d[1]
    controls[-1] = endpoint_values[1]
    controls[-2] = (
        arc * endpoint_values[1, 0] * endpoint_rates[1]
        - end_d[-1] * controls[-1]
    ) / end_d[-2]

    design = np.asarray(basis(sigma), dtype=float)
    if n_ctrl > 4:
        fixed = design[:, :2] @ controls[:2] + design[:, -2:] @ controls[-2:]
        controls[2:-2] = np.linalg.lstsq(
            design[:, 2:-2], samples - fixed, rcond=None)[0]

    fitted = design @ controls
    fitted_rho = fitted[:, 0] * (fitted[:, 1] ** 2 - fitted[:, 2] ** 2)
    fitted_z = 2.0 * fitted[:, 0] * fitted[:, 1] * fitted[:, 2]
    source_xyz = np.column_stack((rho * np.cos(theta), rho * np.sin(theta), z))
    fitted_xyz = np.column_stack((
        fitted_rho * np.cos(fitted[:, 3]),
        fitted_rho * np.sin(fitted[:, 3]),
        fitted_z,
    ))
    errors = np.linalg.norm(fitted_xyz - source_xyz, axis=1)
    diagnostics = {
        "sundman_arc_length": arc,
        "maximum_position_fit_error": float(np.max(errors)),
        "rms_position_fit_error": float(np.sqrt(np.mean(errors**2))),
        "minimum_fitted_h": float(np.min(fitted[:, 0])),
        "minimum_fitted_rho": float(np.min(fitted_rho)),
        "minimum_fitted_radius": float(np.min(np.hypot(fitted_rho, fitted_z))),
    }
    return controls, arc, diagnostics


def write_continuation_seed(path, controls, arc, peak_acceleration, diagnostics, source):
    """Write the interchange JSON consumed by the Sundman solver."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "formulation": "cylindrical B-spline Delta-V to Pythagorean Sundman continuation",
        "control_coefficients_h_p_q_theta": np.asarray(controls, dtype=float).tolist(),
        "sundman_arc_length": float(arc),
        "initial_peak_acceleration_bound": float(peak_acceleration),
        "fit_diagnostics": diagnostics,
        "source_bspline": source,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ephemeris", choices=("circular", "spice"), default="circular")
    parser.add_argument("--departure-date", type=departure_epoch, default="2028-01-01")
    parser.add_argument("--tof-days", type=float, default=1000.0)
    parser.add_argument("--spice-meta-kernel", default="kernels/fasttransfer_planets.tm")
    parser.add_argument("--spice-target-name", default="MARS BARYCENTER")
    parser.add_argument("--n-rev", type=int, default=0)
    parser.add_argument("--orbit-arc-samples", type=int, default=401)
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

    parser.add_argument("--bspline-n-ctrl", type=int, default=10)
    parser.add_argument("--bspline-degree", type=int, default=5)
    parser.add_argument("--bspline-n-fine", type=int, default=4001)
    parser.add_argument("--bspline-quadrature", type=int, default=30)
    parser.add_argument("--bspline-max-iter", type=int, default=1500)
    parser.add_argument("--seed-profile", choices=("linear", "cubic", "quintic"), default="quintic")
    parser.add_argument("--linear-solver", default="mumps")
    parser.add_argument("--coinhsl-library", default=None)

    parser.add_argument("--pythagorean-n-ctrl", type=int, default=10)
    parser.add_argument("--pythagorean-degree", type=int, default=9)
    parser.add_argument("--pythagorean-mode",
                        choices=("min-peak", "energy", "min-dv"),
                        default="min-peak")
    parser.add_argument("--thrust-enforcement", choices=("hull", "grid", "none"),
                        default="hull")
    parser.add_argument("--constraint-nodes-per-span", type=int, default=15)
    parser.add_argument("--subdivide", type=int, default=0)
    parser.add_argument("--quadrature", type=int, default=12)
    parser.add_argument("--samples", type=int, default=2001)
    parser.add_argument("--pythagorean-max-iter", type=int, default=2500)
    parser.add_argument("--h-min", type=float, default=0.1)
    parser.add_argument("--rho-min", type=float, default=0.1)
    parser.add_argument("--r-min", type=float, default=0.1)
    parser.add_argument("--arc-min", type=float, default=1e-3)
    parser.add_argument("--peak-init-factor", type=float, default=1.05)
    parser.add_argument("--pythagorean-seed-source", choices=("bspline", "mee"),
                        default="bspline",
                        help="Fit the Pythagorean seed to the solved B-spline or "
                             "to the analytic MEE profile")
    parser.add_argument("--bspline-initial-control-points", type=Path, default=None,
                        help="Warm-start the B-spline from a saved solution npz")
    parser.add_argument("--pythagorean-initial-solution", type=Path, default=None,
                        help="Warm-start the Pythagorean from a previous solution.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("output/bspline_to_pythagorean_sundman"))
    return parser


def run(args):
    if args.ephemeris == "spice":
        configure_spice_case(args)
    if args.mu <= 0.0 or args.tof <= 0.0:
        raise ValueError("mu and transfer time must be positive")

    endpoint_values, endpoint_rates, _, _ = primitive_endpoint_data(args)
    position, velocity = cylindrical_endpoint_cartesian(args)
    mee0 = rv2mee(position[0], velocity[0], args.mu)
    meef = rv2mee(position[1], velocity[1], args.mu)
    winding_target_rev = args.theta_final / (2.0 * np.pi)

    initial_control_points = None
    if getattr(args, "bspline_initial_control_points", None) is not None:
        with np.load(args.bspline_initial_control_points) as archive:
            initial_control_points = np.asarray(archive["control_points"], dtype=float)
        if initial_control_points.shape != (args.bspline_n_ctrl, 3):
            raise ValueError("Warm-start control points do not match the B-spline basis")

    bspline = solve_free_tf_cylindrical_bspline(
        args.tof,
        mee0=mee0,
        mee_target_epoch=meef,
        mee_target_final=meef,
        mu=args.mu,
        n_ctrl=args.bspline_n_ctrl,
        degree=args.bspline_degree,
        n_fine=args.bspline_n_fine,
        seed_profile=args.seed_profile,
        quadrature_order=args.bspline_quadrature,
        fixed_tf=True,
        winding_target_rev=winding_target_rev,
        initial_control_points=initial_control_points,
        max_iter=args.bspline_max_iter,
        print_level=0 if args.quiet else 5,
        linear_solver=args.linear_solver,
        coinhsl_library=args.coinhsl_library,
    )
    if not bspline["success"]:
        raise RuntimeError(f"minimum-Delta-V B-spline failed: {bspline['message']}")

    profile = bspline["profile_fine"]
    if args.pythagorean_seed_source == "mee":
        seed_tau, seed_q_cyl = mee_seed_profile(
            args, mee0, meef, args.theta_final, args.bspline_n_fine)
    else:
        seed_tau, seed_q_cyl = profile["tau"], profile["q_cyl"]
    controls, arc, diagnostics = fit_pythagorean_sundman_seed(
        seed_tau,
        seed_q_cyl,
        args.tof,
        args.pythagorean_n_ctrl,
        args.pythagorean_degree,
        endpoint_values,
        endpoint_rates,
    )
    reference_peak = (
        float(bspline.get("u_max_reference_m_s2", np.nan))
        / CANONICAL_ACCEL_TO_M_S2)
    if not np.isfinite(reference_peak) or reference_peak <= 0.0:
        reference_peak = float(bspline["max_u_fine"])

    args.output.mkdir(parents=True, exist_ok=True)
    seed_path = args.output / "pythagorean_sundman_seed.json"
    source = {
        "success": bool(bspline["success"]),
        "delta_v_canonical": float(bspline["delta_v_optimizer_canonical"]),
        "delta_v_reference_km_s": float(bspline["delta_v_reference_km_s"]),
        "peak_acceleration_canonical": reference_peak,
        "peak_acceleration_m_s2": reference_peak * CANONICAL_ACCEL_TO_M_S2,
        "n_ctrl": int(args.bspline_n_ctrl),
        "degree": int(args.bspline_degree),
        "ipopt_iterations": int(bspline.get("ipopt_iterations", -1)),
    }
    if getattr(args, "pythagorean_initial_solution", None) is not None:
        # Continue from a neighbouring cell: keep its shape, drop its scaling.
        previous = json.loads(Path(args.pythagorean_initial_solution).read_text())
        loaded = np.asarray(
            previous["control_coefficients_h_p_q_theta"], dtype=float)
        if loaded.shape != (args.pythagorean_n_ctrl, 4):
            raise ValueError("Warm-start controls do not match the Pythagorean basis")
        controls = loaded
        arc = float(previous.get("sundman_arc_length", arc))
        diagnostics = {**diagnostics,
                       "warm_started_from": str(args.pythagorean_initial_solution)}
    write_continuation_seed(seed_path, controls, arc, reference_peak, diagnostics, source)
    np.savez(
        args.output / "bspline_delta_v_solution.npz",
        control_points=np.asarray(bspline["control_points"]),
        tau=np.asarray(profile["tau"]),
        cylindrical_position=np.asarray(profile["q_cyl"]),
        control=np.asarray(profile["u"]),
        control_norm=np.asarray(profile["u_norm"]),
    )

    print(f"B-spline Delta-V: {source['delta_v_reference_km_s']:.9g} km/s")
    print(
        "B-spline peak acceleration: "
        f"{1000.0 * source['peak_acceleration_m_s2']:.9g} mm/s^2"
    )
    print(f"Sundman arc length: {arc:.9g}")
    print(f"Maximum seed position-fit error: {diagnostics['maximum_position_fit_error']:.3e}")
    print(f"Continuation seed: {seed_path.resolve()}")
    if args.prepare_only:
        return {"bspline": source, "seed": str(seed_path), "fit": diagnostics}

    sundman_args = SimpleNamespace(
        mode=args.pythagorean_mode,
        thrust_enforcement=args.thrust_enforcement,
        constraint_nodes_per_span=args.constraint_nodes_per_span,
        ephemeris=args.ephemeris,
        departure_date=args.departure_date,
        tof_days=args.tof_days,
        spice_meta_kernel=args.spice_meta_kernel,
        spice_target_name=args.spice_target_name,
        n_rev=args.n_rev,
        orbit_arc_samples=args.orbit_arc_samples,
        degree=args.pythagorean_degree,
        n_ctrl=args.pythagorean_n_ctrl,
        mu=args.mu,
        r0=args.r0,
        rf=args.rf,
        phi0=args.phi0,
        phif=args.phif,
        rdot0=args.rdot0,
        rdotf=args.rdotf,
        phidot0=args.phidot0,
        phidotf=args.phidotf,
        theta_dot0=args.theta_dot0,
        theta_dotf=args.theta_dotf,
        theta_final=args.theta_final,
        tof=args.tof,
        u_max=reference_peak if args.pythagorean_mode == "energy" else None,
        peak_init_factor=args.peak_init_factor,
        h_min=args.h_min,
        rho_min=args.rho_min,
        r_min=args.r_min,
        arc_min=args.arc_min,
        subdivide=args.subdivide,
        quadrature=args.quadrature,
        samples=args.samples,
        max_iter=args.pythagorean_max_iter,
        initial_solution=seed_path,
        quiet=args.quiet,
        show_plots=args.show_plots,
        no_plots=not args.show_plots,
        output=args.output / "pythagorean_sundman",
        _spice_context=getattr(args, "_spice_context", None),
    )
    return solve_sundman(sundman_args)


def main():
    args = build_parser().parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

"""Validate endpoint conditions from saved hodographic IPOPT coefficients."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reproduce_gondelach_fig2 import (  # noqa: E402
    AU_KM,
    DAY_S,
    basis_integral_matrix,
    basis_matrix,
    cartesian_to_cylindrical,
    configure_ephemeris,
    parse_basis_group,
    planet_state,
    simpson,
    solve_component_coefficients,
    solve_transverse_coefficients,
    split_free_coefficients,
    wrap_0_2pi,
)


_OPTIONS: dict = {}
_DENSE_BASIS: dict = {}


def scalar(saved: np.lib.npyio.NpzFile, key: str):
    return np.asarray(saved[key]).item()


def initialize_worker(options: dict) -> None:
    global _OPTIONS, _DENSE_BASIS
    _OPTIONS = options
    _DENSE_BASIS = {}
    configure_ephemeris(
        options["ephemeris"],
        options.get("spice_meta_kernel"),
        options.get("spice_target_name"),
    )


def read_attempts(path: Path, best_only: bool) -> list[dict]:
    with path.open(newline="") as file:
        raw = list(csv.DictReader(file))
    rows = []
    for row in raw:
        if str(row.get("usable", "")).lower() != "true":
            continue
        free_fields = sorted(
            (key for key in row if key.startswith("free_")),
            key=lambda key: int(key.split("_")[1]),
        )
        free = np.asarray([float(row[key]) for key in free_fields], dtype=float)
        if not np.all(np.isfinite(free)):
            continue
        rows.append(
            {
                "departure_mjd2000": float(row["departure_mjd2000"]),
                "tof_days": float(row["tof_days"]),
                "N": int(row["N"]),
                "delta_v_km_s": float(row["delta_v_km_s"]),
                "optimizer_success": str(row.get("optimizer_success", "")).lower() == "true",
                "transverse_n_quad": (
                    int(row["transverse_n_quad"])
                    if row.get("transverse_n_quad", "").strip()
                    else None
                ),
                "free_coefficients": free,
            }
        )
    if not best_only:
        return rows
    best: dict[tuple[float, float], dict] = {}
    for row in rows:
        key = (row["departure_mjd2000"], row["tof_days"])
        if key not in best or row["delta_v_km_s"] < best[key]["delta_v_km_s"]:
            best[key] = row
    return list(best.values())


def cylindrical_cartesian(
    rho: float,
    theta: float,
    z: float,
    vr: float,
    vtheta: float,
    vz: float,
) -> tuple[np.ndarray, np.ndarray]:
    c, s = math.cos(theta), math.sin(theta)
    pos = np.array([rho * c, rho * s, z])
    vel = np.array([vr * c - vtheta * s, vr * s + vtheta * c, vz])
    return pos, vel


def endpoint_errors(
    theta0: float,
    rho: np.ndarray,
    z: np.ndarray,
    vr: np.ndarray,
    vtheta: np.ndarray,
    vz: np.ndarray,
    theta_delta: float,
    tau: np.ndarray,
    tof: float,
    pos0: np.ndarray,
    vel0: np.ndarray,
    posf: np.ndarray,
    velf: np.ndarray,
) -> tuple[float, float, float, float, float]:
    integrated_delta = float(tof) * simpson(vtheta / rho, tau)
    initial_pos, initial_vel = cylindrical_cartesian(rho[0], theta0, z[0], vr[0], vtheta[0], vz[0])
    final_pos, final_vel = cylindrical_cartesian(
        rho[-1], theta0 + integrated_delta, z[-1], vr[-1], vtheta[-1], vz[-1]
    )
    pos_scale = AU_KM
    vel_scale = AU_KM * 1000.0 / DAY_S
    return (
        float(np.linalg.norm(initial_pos - pos0) * pos_scale),
        float(np.linalg.norm(initial_vel - vel0) * vel_scale),
        float(np.linalg.norm(final_pos - posf) * pos_scale),
        float(np.linalg.norm(final_vel - velf) * vel_scale),
        integrated_delta - theta_delta,
    )


def validate(row: dict) -> dict:
    options = _OPTIONS
    dep, tof, n_rev = row["departure_mjd2000"], row["tof_days"], row["N"]
    try:
        groups = [parse_basis_group(part) for part in options["basis"].split("-")]
        radial_terms, transverse_terms, axial_terms = groups
        radial_free, transverse_free, axial_free = split_free_coefficients(
            row["free_coefficients"], *groups
        )
        pos0, vel0 = planet_state("earth", dep)
        posf, velf = planet_state(options["target"], dep + tof)
        q0, qdot0 = cartesian_to_cylindrical(pos0, vel0)
        qf, qdotf = cartesian_to_cylindrical(posf, velf)
        theta_delta = wrap_0_2pi(float(qf[1] - q0[1])) + 2.0 * math.pi * n_rev

        transverse_n_quad = int(row.get("transverse_n_quad") or options["transverse_n_quad"])
        tau_opt = np.linspace(0.0, 1.0, transverse_n_quad)
        br_opt, _ = basis_matrix(radial_terms, tau_opt, n_rev)
        ir_opt = basis_integral_matrix(radial_terms, tau_opt, n_rev)
        coeff_r = solve_component_coefficients(
            radial_terms, tau_opt, n_rev, tof, qdot0[0], qdotf[0], qf[0] - q0[0], radial_free
        )
        rho_opt = q0[0] + tof * (ir_opt @ coeff_r)
        coeff_t = solve_transverse_coefficients(
            transverse_terms,
            tau_opt,
            n_rev,
            tof,
            qdot0[1],
            qdotf[1],
            theta_delta,
            rho_opt,
            transverse_free,
        )
        coeff_z = solve_component_coefficients(
            axial_terms, tau_opt, n_rev, tof, qdot0[2], qdotf[2], qf[2] - q0[2], axial_free
        )
        bt_opt, _ = basis_matrix(transverse_terms, tau_opt, n_rev)
        bz_opt, _ = basis_matrix(axial_terms, tau_opt, n_rev)
        iz_opt = basis_integral_matrix(axial_terms, tau_opt, n_rev)
        discrete = endpoint_errors(
            q0[1],
            rho_opt,
            q0[2] + tof * (iz_opt @ coeff_z),
            br_opt @ coeff_r,
            bt_opt @ coeff_t,
            bz_opt @ coeff_z,
            theta_delta,
            tau_opt,
            tof,
            pos0,
            vel0,
            posf,
            velf,
        )

        cached = _DENSE_BASIS.get(n_rev)
        if cached is None:
            tau = np.linspace(0.0, 1.0, int(options["n_check"]))
            br, _ = basis_matrix(radial_terms, tau, n_rev)
            ir = basis_integral_matrix(radial_terms, tau, n_rev)
            bt, _ = basis_matrix(transverse_terms, tau, n_rev)
            bz, _ = basis_matrix(axial_terms, tau, n_rev)
            iz = basis_integral_matrix(axial_terms, tau, n_rev)
            cached = (tau, br, ir, bt, bz, iz)
            _DENSE_BASIS[n_rev] = cached
        tau, br, ir, bt, bz, iz = cached
        rho = q0[0] + tof * (ir @ coeff_r)
        dense = endpoint_errors(
            q0[1],
            rho,
            q0[2] + tof * (iz @ coeff_z),
            br @ coeff_r,
            bt @ coeff_t,
            bz @ coeff_z,
            theta_delta,
            tau,
            tof,
            pos0,
            vel0,
            posf,
            velf,
        )
        discrete_passed = (
            discrete[2] <= options["position_tol_km"]
            and discrete[3] <= options["velocity_tol_m_s"]
        )
        dense_passed = (
            dense[2] <= options["position_tol_km"]
            and dense[3] <= options["velocity_tol_m_s"]
        )
        return {
            **{key: row[key] for key in ("departure_mjd2000", "tof_days", "N", "delta_v_km_s", "optimizer_success")},
            "reconstruction_success": True,
            "discrete_initial_position_error_km": discrete[0],
            "discrete_initial_velocity_error_m_s": discrete[1],
            "discrete_arrival_position_error_km": discrete[2],
            "discrete_arrival_velocity_error_m_s": discrete[3],
            "discrete_angle_error_rad": discrete[4],
            "dense_arrival_position_error_km": dense[2],
            "dense_arrival_velocity_error_m_s": dense[3],
            "dense_angle_error_rad": dense[4],
            "minimum_rho_au": float(np.min(rho)),
            "discrete_boundary_pass": discrete_passed,
            "dense_boundary_pass": dense_passed,
            "message": "",
        }
    except Exception as exc:
        return {
            **{key: row[key] for key in ("departure_mjd2000", "tof_days", "N", "delta_v_km_s", "optimizer_success")},
            "reconstruction_success": False,
            "discrete_boundary_pass": False,
            "dense_boundary_pass": False,
            "message": str(exc).splitlines()[-1],
        }


def write_rows(path: Path, rows: list[dict]) -> None:
    fields = [
        "departure_mjd2000", "tof_days", "N", "delta_v_km_s", "optimizer_success",
        "reconstruction_success", "discrete_initial_position_error_km",
        "discrete_initial_velocity_error_m_s", "discrete_arrival_position_error_km",
        "discrete_arrival_velocity_error_m_s", "discrete_angle_error_rad",
        "dense_arrival_position_error_km", "dense_arrival_velocity_error_m_s",
        "dense_angle_error_rad", "minimum_rho_au", "discrete_boundary_pass",
        "dense_boundary_pass", "message",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_summary(path: Path, rows: list[dict], options: dict) -> None:
    successful = [row for row in rows if row.get("reconstruction_success")]
    def values(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in successful], dtype=float)
    fields = [
        "tested_solutions", "reconstructed_solutions", "reconstruction_failures",
        "discrete_boundary_passes", "discrete_boundary_pass_percent",
        "dense_boundary_passes", "dense_boundary_pass_percent", "position_tolerance_km",
        "velocity_tolerance_m_s", "dense_quadrature_nodes",
        "maximum_discrete_arrival_position_error_km",
        "maximum_discrete_arrival_velocity_error_m_s",
        "median_dense_arrival_position_error_km", "p95_dense_arrival_position_error_km",
        "maximum_dense_arrival_position_error_km", "median_dense_arrival_velocity_error_m_s",
        "p95_dense_arrival_velocity_error_m_s", "maximum_dense_arrival_velocity_error_m_s",
        "maximum_absolute_dense_angle_error_rad", "minimum_radius_au",
    ]
    discrete_passed = sum(bool(row.get("discrete_boundary_pass")) for row in successful)
    dense_passed = sum(bool(row.get("dense_boundary_pass")) for row in successful)
    row = {
        "tested_solutions": len(rows),
        "reconstructed_solutions": len(successful),
        "reconstruction_failures": len(rows) - len(successful),
        "discrete_boundary_passes": discrete_passed,
        "discrete_boundary_pass_percent": 100.0 * discrete_passed / max(len(rows), 1),
        "dense_boundary_passes": dense_passed,
        "dense_boundary_pass_percent": 100.0 * dense_passed / max(len(rows), 1),
        "position_tolerance_km": options["position_tol_km"],
        "velocity_tolerance_m_s": options["velocity_tol_m_s"],
        "dense_quadrature_nodes": options["n_check"],
    }
    mappings = {
        "maximum_discrete_arrival_position_error_km": ("discrete_arrival_position_error_km", np.max),
        "maximum_discrete_arrival_velocity_error_m_s": ("discrete_arrival_velocity_error_m_s", np.max),
        "median_dense_arrival_position_error_km": ("dense_arrival_position_error_km", np.median),
        "p95_dense_arrival_position_error_km": ("dense_arrival_position_error_km", lambda x: np.percentile(x, 95)),
        "maximum_dense_arrival_position_error_km": ("dense_arrival_position_error_km", np.max),
        "median_dense_arrival_velocity_error_m_s": ("dense_arrival_velocity_error_m_s", np.median),
        "p95_dense_arrival_velocity_error_m_s": ("dense_arrival_velocity_error_m_s", lambda x: np.percentile(x, 95)),
        "maximum_dense_arrival_velocity_error_m_s": ("dense_arrival_velocity_error_m_s", np.max),
        "maximum_absolute_dense_angle_error_rad": ("dense_angle_error_rad", lambda x: np.max(np.abs(x))),
        "minimum_radius_au": ("minimum_rho_au", np.min),
    }
    for output, (source, reduction) in mappings.items():
        row[output] = float(reduction(values(source))) if successful else np.nan
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--ipopt-dir", type=Path, required=True)
    parser.add_argument("--all-branches", action="store_true", help="Validate every N branch instead of only the best branch per grid point.")
    parser.add_argument("--n-check", type=int, default=1001)
    parser.add_argument(
        "--transverse-n-quad",
        type=int,
        help="Quadrature nodes used to reconstruct the saved transverse coefficients.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--position-tol-km", type=float, default=1.0)
    parser.add_argument("--velocity-tol-m-s", type=float, default=1e-3)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    if args.n_check < 3:
        parser.error("--n-check must be at least 3")
    if args.n_check % 2 == 0:
        args.n_check += 1
    if args.transverse_n_quad is not None:
        if args.transverse_n_quad < 3:
            parser.error("--transverse-n-quad must be at least 3")
        if args.transverse_n_quad % 2 == 0:
            args.transverse_n_quad += 1

    with np.load(args.comparison_dir / "gondelach_high_order_coefficients.npz", allow_pickle=False) as saved:
        options = {
            "basis": str(scalar(saved, "basis")),
            "target": str(scalar(saved, "target")),
            "n_quad": int(scalar(saved, "n_quad")),
            "transverse_n_quad": int(args.transverse_n_quad or scalar(saved, "n_quad")),
            "ephemeris": str(scalar(saved, "ephemeris_source")),
            "spice_meta_kernel": str(scalar(saved, "spice_meta_kernel")),
            "spice_target_name": str(scalar(saved, "spice_target_name")),
            "n_check": args.n_check,
            "position_tol_km": args.position_tol_km,
            "velocity_tol_m_s": args.velocity_tol_m_s,
        }
    rows = read_attempts(args.ipopt_dir / "hodographic_ipopt_attempts.csv", not args.all_branches)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker, initargs=(options,)) as executor:
        for index, result in enumerate(executor.map(validate, rows, chunksize=20), start=1):
            results.append(result)
            if args.progress_every > 0 and (index == len(rows) or index % args.progress_every == 0):
                print(f"validated {index}/{len(rows)}", flush=True)

    suffix = "all_branches" if args.all_branches else "best_branches"
    detail_path = args.ipopt_dir / f"hodographic_boundary_validation_{suffix}.csv"
    summary_path = args.ipopt_dir / f"hodographic_boundary_validation_{suffix}_summary.csv"
    write_rows(detail_path, results)
    write_summary(summary_path, results, options)
    print(f"wrote {detail_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()

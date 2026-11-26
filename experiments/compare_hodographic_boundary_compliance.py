"""Compare boundary compliance of saved vanilla NM and IPOPT campaigns."""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.validate_hodographic_saved_boundaries import (
    initialize_worker,
    read_attempts,
    scalar,
    validate,
)


def load_nm_rows(path: Path) -> tuple[list[dict], int]:
    """Load usable vanilla Nelder--Mead branches from the cached NPZ."""
    with np.load(path, allow_pickle=False) as saved:
        usable = np.asarray(saved["usable"], dtype=bool)
        n_quad = int(scalar(saved, "n_quad"))
        rows = [
            {
                "departure_mjd2000": float(saved["departure_mjd2000"][index]),
                "tof_days": float(saved["tof_days"][index]),
                "N": int(saved["N"][index]),
                "delta_v_km_s": float(saved["delta_v_km_s"][index]),
                "optimizer_success": bool(saved["optimizer_success"][index]),
                "transverse_n_quad": n_quad,
                "free_coefficients": np.asarray(
                    saved["free_coefficients"][index], dtype=float
                ),
            }
            for index in np.flatnonzero(usable)
        ]
        return rows, int(usable.size)


def best_per_grid_point(rows: list[dict]) -> list[dict]:
    best: dict[tuple[float, float], dict] = {}
    for row in rows:
        key = (row["departure_mjd2000"], row["tof_days"])
        if key not in best or row["delta_v_km_s"] < best[key]["delta_v_km_s"]:
            best[key] = row
    return list(best.values())


def validate_all(
    rows: list[dict],
    options: dict,
    workers: int,
    progress_every: int,
    label: str,
) -> list[dict]:
    results: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_worker,
        initargs=(options,),
    ) as executor:
        for index, result in enumerate(
            executor.map(validate, rows, chunksize=20), start=1
        ):
            results.append(result)
            if progress_every > 0 and (
                index == len(rows) or index % progress_every == 0
            ):
                print(f"{label}: validated {index}/{len(rows)}", flush=True)
    return results


def summarize(
    method: str,
    scope: str,
    results: list[dict],
    expected_count: int,
    transverse_n_quad: int,
    options: dict,
) -> dict:
    reconstructed = [row for row in results if row.get("reconstruction_success")]
    position = np.asarray(
        [row["dense_arrival_position_error_km"] for row in reconstructed],
        dtype=float,
    )
    velocity = np.asarray(
        [row["dense_arrival_velocity_error_m_s"] for row in reconstructed],
        dtype=float,
    )
    radius = np.asarray(
        [row["minimum_rho_au"] for row in reconstructed], dtype=float
    )
    discrete_passes = sum(
        bool(row.get("discrete_boundary_pass")) for row in reconstructed
    )
    dense_passes = sum(bool(row.get("dense_boundary_pass")) for row in reconstructed)

    def reduction(values: np.ndarray, function) -> float:
        return float(function(values)) if values.size else float("nan")

    return {
        "method": method,
        "scope": scope,
        "expected_attempts_or_grid_points": expected_count,
        "usable_solutions": len(results),
        "usable_percent": 100.0 * len(results) / max(expected_count, 1),
        "optimizer_successes_among_usable": sum(
            bool(row.get("optimizer_success")) for row in results
        ),
        "optimizer_success_percent_among_usable": 100.0
        * sum(bool(row.get("optimizer_success")) for row in results)
        / max(len(results), 1),
        "optimizer_success_percent_of_expected": 100.0
        * sum(bool(row.get("optimizer_success")) for row in results)
        / max(expected_count, 1),
        "reconstruction_failures": len(results) - len(reconstructed),
        "transverse_quadrature_nodes": transverse_n_quad,
        "dense_check_nodes": options["n_check"],
        "position_tolerance_km": options["position_tol_km"],
        "velocity_tolerance_m_s": options["velocity_tol_m_s"],
        "discrete_boundary_passes": discrete_passes,
        "discrete_boundary_pass_percent": 100.0
        * discrete_passes
        / max(len(results), 1),
        "dense_boundary_passes": dense_passes,
        "dense_boundary_pass_percent_of_usable": 100.0
        * dense_passes
        / max(len(results), 1),
        "dense_boundary_pass_percent_of_expected": 100.0
        * dense_passes
        / max(expected_count, 1),
        "median_dense_position_error_km": reduction(position, np.median),
        "p95_dense_position_error_km": reduction(
            position, lambda values: np.percentile(values, 95)
        ),
        "maximum_dense_position_error_km": reduction(position, np.max),
        "median_dense_velocity_error_m_s": reduction(velocity, np.median),
        "p95_dense_velocity_error_m_s": reduction(
            velocity, lambda values: np.percentile(values, 95)
        ),
        "maximum_dense_velocity_error_m_s": reduction(velocity, np.max),
        "nonpositive_dense_radius_profiles": int(np.sum(radius <= 0.0)),
        "minimum_dense_radius_au": reduction(radius, np.min),
    }


def write_summary(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--ipopt-dir", type=Path, required=True)
    parser.add_argument("--ipopt-transverse-n-quad", type=int, default=501)
    parser.add_argument("--n-check", type=int, default=1001)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--position-tol-km", type=float, default=1.0)
    parser.add_argument("--velocity-tol-m-s", type=float, default=1e-3)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    for name in ("ipopt_transverse_n_quad", "n_check"):
        value = int(getattr(args, name))
        if value < 3:
            parser.error(f"--{name.replace('_', '-')} must be at least 3")
        if value % 2 == 0:
            setattr(args, name, value + 1)

    coefficients_path = args.comparison_dir / "gondelach_high_order_coefficients.npz"
    with np.load(coefficients_path, allow_pickle=False) as saved:
        nm_transverse_n_quad = int(scalar(saved, "n_quad"))
        grid_points = int(
            saved["grid_departure_mjd2000"].size * saved["grid_tof_days"].size
        )
        common_options = {
            "basis": str(scalar(saved, "basis")),
            "target": str(scalar(saved, "target")),
            "n_quad": nm_transverse_n_quad,
            "ephemeris": str(scalar(saved, "ephemeris_source")),
            "spice_meta_kernel": str(scalar(saved, "spice_meta_kernel")),
            "spice_target_name": str(scalar(saved, "spice_target_name")),
            "n_check": args.n_check,
            "position_tol_km": args.position_tol_km,
            "velocity_tol_m_s": args.velocity_tol_m_s,
        }

    nm_rows, nm_expected_attempts = load_nm_rows(coefficients_path)
    ipopt_attempts_path = args.ipopt_dir / "hodographic_ipopt_attempts.csv"
    with ipopt_attempts_path.open(newline="") as file:
        ipopt_expected_attempts = sum(1 for _ in csv.DictReader(file))
    ipopt_rows = read_attempts(ipopt_attempts_path, best_only=False)

    summaries: list[dict] = []
    for method, rows, expected_attempts, transverse_n_quad in (
        ("Nelder--Mead", nm_rows, nm_expected_attempts, nm_transverse_n_quad),
        ("IPOPT", ipopt_rows, ipopt_expected_attempts, args.ipopt_transverse_n_quad),
    ):
        options = {**common_options, "transverse_n_quad": transverse_n_quad}
        validated = validate_all(
            rows, options, args.workers, args.progress_every, method
        )
        summaries.append(
            summarize(
                method,
                "all_usable_branches",
                validated,
                expected_attempts,
                transverse_n_quad,
                options,
            )
        )
        summaries.append(
            summarize(
                method,
                "best_delta_v_per_grid_point",
                best_per_grid_point(validated),
                grid_points,
                transverse_n_quad,
                options,
            )
        )

    output = args.output or (
        args.ipopt_dir / "hodographic_boundary_compliance_nm_vs_ipopt.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_summary(output, summaries)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

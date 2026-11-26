"""Run a B-spline to Sundman continuation grid over departure, TOF and N.

The Pythagorean objective is selected with ``--pythagorean-mode``; no
acceleration limit is imposed unless the chosen mode asks for one.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

from optimizer.pythagorean_3d_bernstein import departure_epoch


FIELDS = [
    "departure_date",
    "departure_mjd2000",
    "departure_offset_days",
    "tof_days",
    "N",
    "bspline_success",
    "pythagorean_success",
    "return_code",
    "status",
    "bspline_delta_v_km_s",
    "bspline_peak_mm_s2",
    "seed_maximum_position_fit_error",
    "pythagorean_peak_mm_s2",
    "pythagorean_delta_v_km_s",
    "pythagorean_ipopt_iterations",
    "wall_time_s",
    "output_directory",
    "message",
]


def iso_date(start: datetime, offset_days: float) -> str:
    date = start + timedelta(days=float(offset_days))
    return date.strftime("%Y-%m-%d")


def read_result(case, output, return_code, wall_time, message=""):
    seed_path = output / "pythagorean_sundman_seed.json"
    solution_path = output / "pythagorean_sundman" / "solution.json"
    seed = json.loads(seed_path.read_text()) if seed_path.exists() else {}
    solution = json.loads(solution_path.read_text()) if solution_path.exists() else {}
    source = seed.get("source_bspline", {})
    fit = seed.get("fit_diagnostics", {})
    status = str(solution.get("status", ""))
    bspline_success = bool(source.get("success", False))
    pythagorean_success = bool(
        return_code == 0
        and status in {"Solve_Succeeded", "Solved_To_Acceptable_Level"}
        and solution.get("numerical_checks_passed", False)
    )
    return {
        **case,
        "bspline_success": bspline_success,
        "pythagorean_success": pythagorean_success,
        "return_code": int(return_code),
        "status": status,
        "bspline_delta_v_km_s": source.get("delta_v_reference_km_s", ""),
        "bspline_peak_mm_s2": (
            1000.0 * float(source["peak_acceleration_m_s2"])
            if source.get("peak_acceleration_m_s2") is not None else ""
        ),
        "seed_maximum_position_fit_error": fit.get("maximum_position_fit_error", ""),
        "pythagorean_peak_mm_s2": solution.get(
            "sampled_peak_acceleration_mm_s2", ""),
        "pythagorean_delta_v_km_s": solution.get("delta_v_reference_km_s", ""),
        "pythagorean_ipopt_iterations": solution.get("ipopt_iterations", ""),
        "wall_time_s": float(wall_time),
        "output_directory": str(output),
        "message": message,
    }


def run_case(case, args):
    output = (
        args.output
        / f"dep_{case['departure_date']}"
        / f"tof_{case['tof_days']:g}"
        / f"N{case['N']}"
    )
    solution_path = output / "pythagorean_sundman" / "solution.json"
    if solution_path.exists() and not args.force:
        previous = json.loads(solution_path.read_text())
        if previous.get("numerical_checks_passed", False):
            return read_result(case, output, 0, 0.0, "resumed existing successful result")

    command = [
        sys.executable,
        "-m",
        "optimizer.optimizers",
        "--ephemeris",
        "spice",
        "--departure-date",
        case["departure_date"],
        "--tof-days",
        str(case["tof_days"]),
        "--spice-target-name",
        args.spice_target_name,
        "--spice-meta-kernel",
        args.spice_meta_kernel,
        "--n-rev",
        str(case["N"]),
        "--bspline-n-ctrl",
        str(args.bspline_n_ctrl),
        "--bspline-degree",
        str(args.bspline_degree),
        "--bspline-n-fine",
        str(args.bspline_n_fine),
        "--bspline-quadrature",
        str(args.bspline_quadrature),
        "--bspline-max-iter",
        str(args.bspline_max_iter),
        "--pythagorean-n-ctrl",
        str(args.pythagorean_n_ctrl),
        "--pythagorean-degree",
        str(args.pythagorean_degree),
        "--pythagorean-mode",
        args.pythagorean_mode,
        "--thrust-enforcement",
        args.thrust_enforcement,
        "--subdivide",
        str(args.subdivide),
        "--pythagorean-max-iter",
        str(args.pythagorean_max_iter),
        "--samples",
        str(args.samples),
        "--output",
        str(output),
        "--quiet",
    ]
    started = perf_counter()
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    wall_time = perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.log").write_text(completed.stdout)
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    message = "" if completed.returncode == 0 else (lines[-1] if lines else "process failed")
    return read_result(case, output, completed.returncode, wall_time, message)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (
        float(row["departure_offset_days"]), float(row["tof_days"]), int(row["N"])))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ordered)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--departure-start", default="2028-01-01")
    parser.add_argument("--synodic-days", type=float, default=780.0)
    parser.add_argument("--departure-count", type=int, default=3)
    parser.add_argument("--departure-step", type=float, default=None,
                        help="Departure spacing [days]; overrides the synodic split")
    parser.add_argument("--tof-min", type=float, default=500.0)
    parser.add_argument("--tof-max", type=float, default=1500.0)
    parser.add_argument("--tof-count", type=int, default=3)
    parser.add_argument("--n-min", type=int, default=0)
    parser.add_argument("--n-max", type=int, default=5)
    parser.add_argument("--spice-target-name", default="MARS BARYCENTER")
    parser.add_argument("--spice-meta-kernel", default="kernels/fasttransfer_planets.tm")
    parser.add_argument("--bspline-n-ctrl", type=int, default=10)
    parser.add_argument("--bspline-degree", type=int, default=5)
    parser.add_argument("--bspline-n-fine", type=int, default=2001)
    parser.add_argument("--bspline-quadrature", type=int, default=30)
    parser.add_argument("--bspline-max-iter", type=int, default=1500)
    parser.add_argument("--pythagorean-n-ctrl", type=int, default=10)
    parser.add_argument("--pythagorean-degree", type=int, default=9)
    parser.add_argument("--pythagorean-max-iter", type=int, default=2500)
    parser.add_argument("--pythagorean-mode",
                        choices=("min-peak", "energy", "min-dv"),
                        default="min-peak")
    parser.add_argument("--thrust-enforcement", choices=("hull", "grid", "none"),
                        default="hull")
    parser.add_argument("--subdivide", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1001)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/bspline_to_sundman_grid_3x3_n0_5"),
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.departure_count < 1 or args.tof_count < 1 or args.workers < 1:
        raise SystemExit("grid counts and workers must be positive")
    if args.n_max < args.n_min or args.synodic_days <= 0.0:
        raise SystemExit("invalid revolution range or synodic period")

    start = datetime.strptime(args.departure_start, "%Y-%m-%d")
    if args.departure_step is not None:
        if args.departure_step <= 0.0:
            raise SystemExit("departure-step must be positive")
        departure_offsets = np.arange(args.departure_count) * args.departure_step
    else:
        departure_offsets = (np.arange(args.departure_count)
                             * args.synodic_days / args.departure_count)
    tof_values = np.linspace(args.tof_min, args.tof_max, args.tof_count)
    cases = []
    for offset in departure_offsets:
        date = iso_date(start, float(offset))
        for tof in tof_values:
            for n_rev in range(args.n_min, args.n_max + 1):
                cases.append({
                    "departure_date": date,
                    "departure_mjd2000": float(departure_epoch(date)),
                    "departure_offset_days": float(offset),
                    "tof_days": float(tof),
                    "N": int(n_rev),
                })

    rows = []
    csv_path = args.output / "grid_attempts.csv"
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_case, case, args): case for case in cases}
        for completed, future in enumerate(as_completed(futures), start=1):
            case = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                output = (
                    args.output / f"dep_{case['departure_date']}"
                    / f"tof_{case['tof_days']:g}" / f"N{case['N']}")
                row = read_result(case, output, -1, 0.0, str(exc))
            rows.append(row)
            write_csv(csv_path, rows)
            print(
                f"[{completed:02d}/{len(cases)}] dep={row['departure_date']} "
                f"TOF={row['tof_days']:g} N={row['N']}: "
                f"B-spline={'OK' if row['bspline_success'] else 'FAIL'}, "
                f"Pythagorean={'OK' if row['pythagorean_success'] else 'FAIL'}"
            )

    successful = sum(bool(row["pythagorean_success"]) for row in rows)
    summary = {
        "number_of_cases": len(rows),
        "bspline_successes": sum(bool(row["bspline_success"]) for row in rows),
        "pythagorean_successes": successful,
        "all_pythagorean_successful": successful == len(rows),
        "departure_dates": [iso_date(start, float(offset)) for offset in departure_offsets],
        "tof_days": tof_values.tolist(),
        "revolutions": list(range(args.n_min, args.n_max + 1)),
    }
    (args.output / "grid_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()

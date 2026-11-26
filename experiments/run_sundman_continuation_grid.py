"""Departure-by-TOF sweep with per-chain continuation and a cold-seed fallback.

Both families are seeded from the same analytic MEE profile, so a poor B-spline
local optimum is not propagated into the Pythagorean seed.  The grid is walked as
one chain per ``(departure, N)`` pair, marching over increasing time of flight and
warm-starting each cell from the previous one.  Chains are independent, so they
parallelize cleanly while the continuation stays sequential inside each chain.

Every cell is solved twice, cold and warm, and the better objective is kept.  That
doubles the solves but stops a chain from carrying a bad local optimum upward,
which a pure continuation cannot recover from on its own.

Run from the repository root, for example::

    python -m experiments.run_sundman_continuation_grid --workers 12 \
      --departure-step 20 --departure-count 39 --tof-count 51
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
from threading import Lock
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimizer.pythagorean_3d_bernstein import departure_epoch
from experiments.compact_grid_trajectories import compact_root


FIELDS = [
    "departure_date",
    "departure_mjd2000",
    "departure_offset_days",
    "tof_days",
    "N",
    "bspline_success",
    "pythagorean_success",
    "status",
    "bspline_delta_v_km_s",
    "bspline_peak_mm_s2",
    "pythagorean_delta_v_km_s",
    "pythagorean_peak_mm_s2",
    "pythagorean_hull_bound_mm_s2",
    "seed_maximum_position_fit_error",
    "pythagorean_ipopt_iterations",
    "failed_stage",
    "failure_status",
    "selected_start",
    "cold_objective",
    "warm_objective",
    "continuation_gain_percent",
    "wall_time_s",
    "output_directory",
    "message",
]


def iso_date(start, offset_days):
    return (start + timedelta(days=float(offset_days))).strftime("%Y-%m-%d")


def diagnose_failure(directory, result):
    """Say which stage failed, so the sweep log is actionable while it runs.

    The driver raises as soon as the minimum-Delta-V B-spline does not converge,
    so a failed first stage leaves no seed file behind; that absence is what
    separates the two cases.
    """
    if result["pythagorean_success"]:
        return "", ""
    log_path = directory / "run.log"
    text = log_path.read_text() if log_path.exists() else ""
    status = ""
    marker = "return_status is "
    if marker in text:
        tail = text.rsplit(marker, 1)[1].splitlines()[0] if text.rsplit(
            marker, 1)[1].splitlines() else ""
        status = tail.strip().strip("'\"., ")
    if (directory / "pythagorean_sundman_seed.json").exists() or result["bspline_success"]:
        return "pythagorean", status or result["status"]
    if "minimum-Delta-V B-spline failed" in text:
        return "bspline", status
    return "setup", status


def read_run(directory):
    """Collect what a single pipeline run produced, or an empty result."""
    seed_path = directory / "pythagorean_sundman_seed.json"
    solution_path = directory / "pythagorean_sundman" / "solution.json"
    seed = json.loads(seed_path.read_text()) if seed_path.exists() else {}
    solution = json.loads(solution_path.read_text()) if solution_path.exists() else {}
    source = seed.get("source_bspline", {})
    peak = source.get("peak_acceleration_m_s2")
    return {
        "directory": directory,
        "status": str(solution.get("status", "")),
        "pythagorean_success": bool(
            solution.get("numerical_checks_passed", False)
            and str(solution.get("status", "")) in
            {"Solve_Succeeded", "Solved_To_Acceptable_Level"}),
        "bspline_success": bool(source.get("success", False)),
        "bspline_delta_v_km_s": source.get("delta_v_reference_km_s", ""),
        "bspline_peak_mm_s2": 1000.0 * float(peak) if peak is not None else "",
        "pythagorean_delta_v_km_s": solution.get("delta_v_reference_km_s", ""),
        "pythagorean_peak_mm_s2": solution.get("sampled_peak_acceleration_mm_s2", ""),
        # Certified continuous bound, not an imposed cap: in min-peak mode it is
        # the optimized value, elsewhere it is evaluated after the solve.
        "pythagorean_hull_bound_mm_s2": solution.get(
            "required_peak_acceleration_mm_s2", ""),
        "seed_maximum_position_fit_error":
            seed.get("fit_diagnostics", {}).get("maximum_position_fit_error", ""),
        "pythagorean_ipopt_iterations": solution.get("ipopt_iterations", ""),
    }


def objective_of(result, mode):
    """Lower is better; infinite for a run that did not pass its checks."""
    if not result["pythagorean_success"]:
        return float("inf")
    key = ("pythagorean_peak_mm_s2" if mode == "min-peak"
           else "pythagorean_delta_v_km_s")
    try:
        value = float(result[key])
    except (TypeError, ValueError):
        return float("inf")
    return value if np.isfinite(value) else float("inf")


def solve_one(directory, case, args, warm_from=None):
    """Run the two-stage pipeline once, optionally continued from a neighbour."""
    command = [
        sys.executable, "-m", "optimizer.optimizers",
        "--ephemeris", "spice",
        "--departure-date", case["departure_date"],
        "--tof-days", str(case["tof_days"]),
        "--spice-target-name", args.spice_target_name,
        "--spice-meta-kernel", args.spice_meta_kernel,
        "--n-rev", str(case["N"]),
        "--seed-profile", args.seed_profile,
        "--pythagorean-seed-source", "mee",
        "--bspline-n-ctrl", str(args.bspline_n_ctrl),
        "--bspline-degree", str(args.bspline_degree),
        "--bspline-n-fine", str(args.bspline_n_fine),
        "--bspline-quadrature", str(args.bspline_quadrature),
        "--bspline-max-iter", str(args.bspline_max_iter),
        "--pythagorean-n-ctrl", str(args.pythagorean_n_ctrl),
        "--pythagorean-degree", str(args.pythagorean_degree),
        "--pythagorean-mode", args.pythagorean_mode,
        "--thrust-enforcement", args.thrust_enforcement,
        "--subdivide", str(args.subdivide),
        "--pythagorean-max-iter", str(args.pythagorean_max_iter),
        "--samples", str(args.samples),
        "--output", str(directory),
        "--quiet",
    ]
    if warm_from is not None:
        command += [
            "--bspline-initial-control-points",
            str(warm_from / "bspline_delta_v_solution.npz"),
            "--pythagorean-initial-solution",
            str(warm_from / "pythagorean_sundman" / "solution.json"),
        ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               check=False)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.log").write_text(completed.stdout)
    result = read_run(directory)
    stage, status = diagnose_failure(directory, result)
    result["failed_stage"] = stage
    result["failure_status"] = status
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    result["message"] = "" if completed.returncode == 0 else (
        lines[-1] if lines else "process failed")
    return result


def seeded_solution(args, case):
    """Locate the matching cell of an earlier sweep, to reuse its trajectory.

    A minimum-Delta-V solution is a good starting shape for the minimum-peak
    problem on the same boundary data, and reusing it costs nothing.
    """
    root = getattr(args, "warm_start_from", None)
    if root is None:
        return None
    cell = (Path(root) / f"dep_{case['departure_date']}" / f"N{case['N']}"
            / f"tof_{case['tof_days']:g}")
    selection = cell / "selection.json"
    if not selection.exists():
        return None
    try:
        chosen = json.loads(selection.read_text())["selected_start"]
    except (ValueError, KeyError):
        return None
    directory = cell / chosen
    if (directory / "pythagorean_sundman" / "solution.json").exists():
        return directory
    return None


def run_chain(chain, args, report):
    """March one (departure, N) chain over increasing time of flight."""
    rows = []
    previous = None
    for case in chain:
        cell = (args.output / f"dep_{case['departure_date']}" / f"N{case['N']}"
                / f"tof_{case['tof_days']:g}")
        selection_path = cell / "selection.json"
        started = perf_counter()

        if selection_path.exists() and not args.force:
            saved = json.loads(selection_path.read_text())
            chosen = read_run(cell / saved["selected_start"])
            if chosen["pythagorean_success"]:
                row = {**case, **chosen, "selected_start": saved["selected_start"],
                       "cold_objective": saved.get("cold_objective", ""),
                       "warm_objective": saved.get("warm_objective", ""),
                       "continuation_gain_percent":
                           saved.get("continuation_gain_percent", ""),
                       "failed_stage": "", "failure_status": "",
                       "wall_time_s": 0.0,
                       "message": "resumed"}
                rows.append(row)
                previous = chosen["directory"]
                report(row)
                continue

        seeded = seeded_solution(args, case)
        if args.seeded_only:
            if seeded is None:
                row = {**case, **read_run(cell / "seeded"),
                       "failed_stage": "seed", "failure_status": "",
                       "selected_start": "seeded", "cold_objective": "",
                       "warm_objective": "", "continuation_gain_percent": "",
                       "wall_time_s": perf_counter() - started,
                       "message": "no matching cell in the seed sweep"}
                row["pythagorean_success"] = False
                rows.append(row)
                report(row)
                continue
            candidates = {"seeded": solve_one(cell / "seeded", case, args, seeded)}
        else:
            candidates = {"cold": solve_one(cell / "cold", case, args)}
            if previous is not None:
                candidates["warm"] = solve_one(cell / "warm", case, args, previous)
            if seeded is not None:
                candidates["seeded"] = solve_one(
                    cell / "seeded", case, args, seeded)
        scores = {name: objective_of(result, args.pythagorean_mode)
                  for name, result in candidates.items()}
        selected = min(scores, key=scores.get)
        chosen = candidates[selected]
        cold = scores.get("cold", float("inf"))
        warm = min(scores.get("warm", float("inf")),
                   scores.get("seeded", float("inf")))
        gain = (100.0 * (cold - warm) / cold
                if np.isfinite(cold) and np.isfinite(warm) and cold > 0 else "")

        cell.mkdir(parents=True, exist_ok=True)
        selection_path.write_text(json.dumps({
            "selected_start": selected,
            "cold_objective": cold if np.isfinite(cold) else None,
            "warm_objective": warm if np.isfinite(warm) else None,
            "continuation_gain_percent": gain if gain != "" else None,
            "objective": args.pythagorean_mode,
        }, indent=2) + "\n")

        row = {**case, **chosen, "selected_start": selected,
               "cold_objective": cold if np.isfinite(cold) else "",
               "warm_objective": warm if np.isfinite(warm) else "",
               "continuation_gain_percent": gain,
               "wall_time_s": perf_counter() - started}
        rows.append(row)
        if chosen["pythagorean_success"]:
            previous = chosen["directory"]
        report(row)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (
        float(row["departure_offset_days"]), float(row["tof_days"]), int(row["N"])))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--departure-start", default="2028-01-01")
    parser.add_argument("--departure-step", type=float, default=20.0)
    parser.add_argument("--departure-count", type=int, default=39)
    parser.add_argument("--tof-min", type=float, default=500.0)
    parser.add_argument("--tof-max", type=float, default=1500.0)
    parser.add_argument("--tof-count", type=int, default=51)
    parser.add_argument("--tof-direction", choices=("ascending", "descending"),
                        default="descending",
                        help="Order each chain marches the time of flight; start "
                             "at the easier end so hard cells inherit a warm start")
    parser.add_argument("--n-min", type=int, default=0)
    parser.add_argument("--n-max", type=int, default=3)
    parser.add_argument("--spice-target-name", default="MARS BARYCENTER")
    parser.add_argument("--spice-meta-kernel", default="kernels/fasttransfer_planets.tm")
    parser.add_argument("--seed-profile", choices=("linear", "cubic", "quintic"),
                        default="quintic")
    parser.add_argument("--bspline-n-ctrl", type=int, default=7)
    parser.add_argument("--bspline-degree", type=int, default=3)
    parser.add_argument("--bspline-n-fine", type=int, default=2001)
    parser.add_argument("--bspline-quadrature", type=int, default=30)
    parser.add_argument("--bspline-max-iter", type=int, default=1500)
    parser.add_argument("--pythagorean-n-ctrl", type=int, default=7)
    parser.add_argument("--pythagorean-degree", type=int, default=3)
    parser.add_argument("--pythagorean-max-iter", type=int, default=2500)
    parser.add_argument("--pythagorean-mode",
                        choices=("min-peak", "energy", "min-dv"), default="min-dv")
    parser.add_argument("--thrust-enforcement", choices=("hull", "grid", "none"),
                        default="none")
    parser.add_argument("--subdivide", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1001)
    parser.add_argument("--warm-start-from", type=Path, default=None,
                        help="Sweep directory whose matching cells seed this one, "
                             "for example a minimum-Delta-V sweep feeding a "
                             "minimum-peak sweep on the same grid")
    parser.add_argument("--seeded-only", action="store_true",
                        help="Solve each cell only from the matching cell of "
                             "--warm-start-from, with no cold fallback")
    parser.add_argument("--cold-only", action="store_true",
                        help="Solve every cell from the MEE seed alone; chains are "
                             "flattened so all cells run independently in parallel")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-trajectory-csv", action="store_true",
                        help="Leave per-run trajectory CSVs loose instead of archiving them")
    parser.add_argument("--output", type=Path,
                        default=Path("output/sundman_continuation_grid"))
    return parser


def main():
    args = build_parser().parse_args()
    if args.departure_count < 1 or args.tof_count < 1 or args.workers < 1:
        raise SystemExit("grid counts and workers must be positive")
    if args.n_max < args.n_min or args.departure_step <= 0.0:
        raise SystemExit("invalid revolution range or departure step")
    if args.seeded_only and args.warm_start_from is None:
        raise SystemExit("seeded-only requires --warm-start-from")

    start = datetime.strptime(args.departure_start, "%Y-%m-%d")
    offsets = np.arange(args.departure_count) * args.departure_step
    tof_values = np.linspace(args.tof_min, args.tof_max, args.tof_count)
    chains = []
    for offset in offsets:
        date = iso_date(start, float(offset))
        for n_rev in range(args.n_min, args.n_max + 1):
            ordered_tof = (tof_values if args.tof_direction == "ascending"
                           else tof_values[::-1])
            cells = [{
                "departure_date": date,
                "departure_mjd2000": float(departure_epoch(date)),
                "departure_offset_days": float(offset),
                "tof_days": float(tof),
                "N": int(n_rev),
            } for tof in ordered_tof]
            # Without a warm start nothing links consecutive cells, so each one
            # becomes its own task and the pool balances over the whole grid.
            if args.cold_only:
                chains.extend([cell] for cell in cells)
            else:
                chains.append(cells)

    total = sum(len(chain) for chain in chains)
    if args.cold_only:
        print(f"{total} celdas independientes (solo arranque en frio), "
              f"{args.workers} workers, objetivo {args.pythagorean_mode}")
    else:
        print(f"{len(chains)} cadenas de {len(tof_values)} celdas ({total} en total), "
              f"{args.workers} workers, objetivo {args.pythagorean_mode}, "
              f"TOF {args.tof_direction}")
    rows = []
    lock = Lock()
    csv_path = args.output / "grid_attempts.csv"

    last_write = [0.0]

    def report(row):
        with lock:
            rows.append(row)
            # Rewriting the whole table per cell is quadratic and dominates a
            # resume, where nothing is solved; a couple of seconds of staleness
            # keeps the file watchable without that cost.
            now = perf_counter()
            if now - last_write[0] > 2.0:
                write_csv(csv_path, rows)
                last_write[0] = now
            if row["pythagorean_success"]:
                outcome = f"OK ({row['selected_start']})"
            else:
                detail = row.get("failure_status") or "?"
                outcome = f"FAIL en {row.get('failed_stage') or '?'} [{detail}]"
            print(f"[{len(rows):04d}/{total}] dep={row['departure_date']} "
                  f"TOF={row['tof_days']:g} N={row['N']}: {outcome}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_chain, chain, args, report) for chain in chains]
        for future in as_completed(futures):
            future.result()
    write_csv(csv_path, rows)
    if not args.keep_trajectory_csv:
        count, saved = compact_root(args.output)
        print(f"Archived {count} trajectories; saved {saved / 1e9:.2f} GB")

    successes = sum(bool(row["pythagorean_success"]) for row in rows)
    warm_wins = sum(row["selected_start"] == "warm" for row in rows)
    gains = [float(row["continuation_gain_percent"]) for row in rows
             if row["continuation_gain_percent"] not in ("", None)]
    stage_counts = {}
    for row in rows:
        stage = row.get("failed_stage") or ""
        if stage:
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
    summary = {
        "number_of_cells": len(rows),
        "pythagorean_successes": successes,
        "failures_by_stage": stage_counts,
        "cells_won_by_continuation": warm_wins,
        "median_continuation_gain_percent":
            float(np.median(gains)) if gains else None,
        "objective": args.pythagorean_mode,
        "departure_dates": [iso_date(start, float(o)) for o in offsets],
        "tof_days": tof_values.tolist(),
        "revolutions": list(range(args.n_min, args.n_max + 1)),
    }
    (args.output / "grid_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()

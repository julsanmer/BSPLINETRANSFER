"""Run independent MEE-seeded hodographic IPOPT branches in parallel.

This auxiliary campaign reads an existing Gondelach comparison directory,
reuses its exact grid and physical configuration, and performs no neighbor
continuation.  Every branch receives its own MEE-profile coefficient fit.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_hodographic_optimizers import (  # noqa: E402
    load_cached_scipy_campaign,
    method_summary,
    write_csv,
)
from experiments.reproduce_gondelach_fig2 import configure_ephemeris  # noqa: E402
from experiments.reproduce_gondelach_fig3 import (  # noqa: E402
    build_ipopt_problem,
    mee_profile_hodographic_seed,
    optimize_grid_point,
    write_attempts_csv,
)


_WORKER_OPTIONS: dict = {}
_WORKER_PROBLEMS: dict = {}


def initialize_worker(options: dict) -> None:
    global _WORKER_OPTIONS, _WORKER_PROBLEMS
    _WORKER_OPTIONS = options
    _WORKER_PROBLEMS = {}
    configure_ephemeris(
        options["ephemeris"],
        options.get("spice_meta_kernel"),
        options.get("spice_target_name"),
    )


def solve_independent_branch(task: tuple[float, float, int]) -> dict:
    dep, tof, n_rev = task
    options = _WORKER_OPTIONS
    branch_start = perf_counter()
    try:
        problem = _WORKER_PROBLEMS.get(n_rev)
        if problem is None:
            problem = build_ipopt_problem(
                options["basis"],
                options["n_quad"],
                n_rev,
                options["ipopt_max_iter"],
                options["ipopt_tol"],
                options["ipopt_acceptable_tol"],
                options["ipopt_print_level"],
                options["ipopt_linear_solver"],
                options["ipopt_dv_eps"],
                options["transverse_n_quad"],
            )
            _WORKER_PROBLEMS[n_rev] = problem
        seed = mee_profile_hodographic_seed(
            dep,
            tof,
            n_rev,
            options["target"],
            options["basis"],
            profile=options["mee_seed_profile"],
            n_fit=options["mee_seed_n_fit"],
        )
        row = optimize_grid_point(
            dep,
            tof,
            n_rev,
            options["target"],
            options["basis"],
            options["n_quad"],
            seed,
            options["maxfev"],
            options["xatol"],
            options["fatol"],
            options["simplex_scale"],
            "ipopt",
            ipopt_problem=problem,
            ipopt_max_iter=options["ipopt_max_iter"],
            ipopt_tol=options["ipopt_tol"],
            ipopt_acceptable_tol=options["ipopt_acceptable_tol"],
            ipopt_linear_solver=options["ipopt_linear_solver"],
            ipopt_print_level=options["ipopt_print_level"],
            ipopt_dv_eps=options["ipopt_dv_eps"],
            transverse_n_quad=options["transverse_n_quad"],
        )
        row["initialization"] = f"mee_{options['mee_seed_profile']}_fit"
        return row
    except Exception as exc:
        return {
            "departure_mjd2000": dep,
            "tof_days": tof,
            "N": n_rev,
            "delta_v_km_s": np.nan,
            "optimizer_success": False,
            "optimizer": "ipopt",
            "usable": False,
            "nfev": 0,
            "nit": 0,
            "num_restarts": 0,
            "message": f"MEE seed/IPOPT exception: {str(exc).splitlines()[-1]}",
            "free_coefficients": np.full(6, np.nan),
            "start_delta_v_km_s": np.nan,
            "initialization": f"mee_{options['mee_seed_profile']}_fit_failed",
            "transverse_n_quad": options["transverse_n_quad"],
            "wall_time_s": perf_counter() - branch_start,
        }


def best_grids(rows: list[dict], departure: np.ndarray, tof: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dep_index = {float(value): index for index, value in enumerate(departure)}
    tof_index = {float(value): index for index, value in enumerate(tof)}
    dv = np.full((len(tof), len(departure)), np.nan)
    best_n = np.full(dv.shape, -1, dtype=int)
    for row in rows:
        if not bool(row.get("usable", False)) or not np.isfinite(float(row["delta_v_km_s"])):
            continue
        i = tof_index[float(row["tof_days"])]
        j = dep_index[float(row["departure_mjd2000"])]
        value = float(row["delta_v_km_s"])
        if not np.isfinite(dv[i, j]) or value < dv[i, j]:
            dv[i, j] = value
            best_n[i, j] = int(row["N"])
    return dv, best_n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mee-seed-profile", choices=["linear", "cubic", "quintic"], default="quintic")
    parser.add_argument("--mee-seed-n-fit", type=int, default=301)
    parser.add_argument("--ipopt-max-iter", type=int, default=1000)
    parser.add_argument("--ipopt-tol", type=float, default=1e-8)
    parser.add_argument("--ipopt-acceptable-tol", type=float, default=1e-6)
    parser.add_argument("--ipopt-linear-solver", default="mumps")
    parser.add_argument("--ipopt-print-level", type=int, default=0)
    parser.add_argument("--ipopt-dv-eps", type=float, default=1e-6)
    parser.add_argument(
        "--transverse-n-quad",
        type=int,
        default=201,
        help="Simpson nodes used only to impose the transverse angle boundary condition.",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.transverse_n_quad < 3:
        parser.error("--transverse-n-quad must be at least 3")
    if args.transverse_n_quad % 2 == 0:
        args.transverse_n_quad += 1

    cached, nm_rows, nm_grid, nm_elapsed = load_cached_scipy_campaign(args.comparison_dir)
    departure = np.asarray(cached.pop("departure_grid"), dtype=float)
    tof = np.asarray(cached.pop("tof_grid"), dtype=float)
    options = {
        **{key: value for key, value in cached.items() if key not in {"dep_min", "dep_max", "tof_min", "tof_max", "dep_step", "tof_step"}},
        "mee_seed_profile": args.mee_seed_profile,
        "mee_seed_n_fit": args.mee_seed_n_fit,
        "ipopt_max_iter": args.ipopt_max_iter,
        "ipopt_tol": args.ipopt_tol,
        "ipopt_acceptable_tol": args.ipopt_acceptable_tol,
        "ipopt_linear_solver": args.ipopt_linear_solver,
        "ipopt_print_level": args.ipopt_print_level,
        "ipopt_dv_eps": args.ipopt_dv_eps,
        "transverse_n_quad": args.transverse_n_quad,
    }
    output_dir = args.output_dir or args.comparison_dir / (
        f"hodographic_ipopt_mee_{args.mee_seed_profile}_eps{args.ipopt_dv_eps:g}_tquad{args.transverse_n_quad}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (float(dep), float(duration), int(n_rev))
        for duration in tof
        for dep in departure
        for n_rev in range(int(options["n_min"]), int(options["n_max"]) + 1)
    ]
    start = perf_counter()
    rows: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_worker,
        initargs=(options,),
    ) as executor:
        for index, row in enumerate(executor.map(solve_independent_branch, tasks, chunksize=20), start=1):
            rows.append(row)
            if args.progress and (index == len(tasks) or index % max(1, args.progress_every) == 0):
                print(f"completed {index}/{len(tasks)}", flush=True)
    ipopt_elapsed = perf_counter() - start
    ipopt_grid, ipopt_best_n = best_grids(rows, departure, tof)
    write_attempts_csv(output_dir / "hodographic_ipopt_attempts.csv", rows)

    summaries = [
        method_summary("scipy", nm_elapsed, nm_rows, nm_grid),
        method_summary("ipopt", ipopt_elapsed, rows, ipopt_grid),
    ]
    common = np.isfinite(nm_grid) & np.isfinite(ipopt_grid)
    absolute = np.abs(ipopt_grid[common] - nm_grid[common])
    for summary in summaries:
        summary["workers"] = 1 if summary["optimizer"] == "scipy" else args.workers
        summary["initialization"] = "neighbor_continuation" if summary["optimizer"] == "scipy" else f"mee_{args.mee_seed_profile}_fit_no_continuation"
        summary["dv_eps_canonical"] = 0.0 if summary["optimizer"] == "scipy" else args.ipopt_dv_eps
        summary["common_finite_grid_points"] = int(common.sum())
        summary["median_abs_delta_v_difference_km_s"] = float(np.median(absolute)) if absolute.size else np.nan
        summary["max_abs_delta_v_difference_km_s"] = float(np.max(absolute)) if absolute.size else np.nan
    write_csv(output_dir / "hodographic_optimizer_benchmark.csv", summaries)
    np.savez(
        output_dir / "hodographic_optimizer_benchmark_grids.npz",
        departure_mjd2000=departure,
        tof_days=tof,
        scipy_delta_v_km_s=nm_grid,
        ipopt_delta_v_km_s=ipopt_grid,
        ipopt_best_N=ipopt_best_n,
        basis=np.asarray(options["basis"]),
        target=np.asarray(options["target"]),
        n_quad=np.asarray(options["n_quad"]),
        n_min=np.asarray(options["n_min"]),
        n_max=np.asarray(options["n_max"]),
        workers=np.asarray(args.workers),
        initialization=np.asarray(f"mee_{args.mee_seed_profile}_fit_no_continuation"),
        ipopt_dv_eps=np.asarray(args.ipopt_dv_eps),
        transverse_n_quad=np.asarray(args.transverse_n_quad),
    )
    print(f"wrote {output_dir / 'hodographic_optimizer_benchmark.csv'}")
    print(f"IPOPT wall time: {ipopt_elapsed:.3f} s with {args.workers} workers")


if __name__ == "__main__":
    # Avoid oversubscribing BLAS inside the process pool.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()

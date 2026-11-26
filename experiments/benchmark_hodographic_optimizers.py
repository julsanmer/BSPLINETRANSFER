"""Compare Nelder--Mead and CasADi/IPOPT on the same hodographic grid."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compare_gondelach_fig2_fig3_bspline10 import CASE_CONFIGS  # noqa: E402
from experiments.reproduce_gondelach_fig2 import configure_ephemeris  # noqa: E402
from experiments.reproduce_gondelach_fig3 import compute_grid, write_attempts_csv  # noqa: E402


def method_summary(
    optimizer: str,
    elapsed: float,
    rows: list[dict],
    dv_grid: np.ndarray,
) -> dict:
    branch_attempts = len(rows)
    times = np.asarray([row.get("wall_time_s", np.nan) for row in rows], dtype=float)
    finite_times = times[np.isfinite(times)]
    return {
        "optimizer": optimizer,
        "wall_time_s": elapsed,
        "branch_attempts": branch_attempts,
        "seconds_per_branch": elapsed / max(branch_attempts, 1),
        "median_branch_time_s": float(np.median(finite_times)) if finite_times.size else np.nan,
        "optimizer_successes": sum(bool(row.get("optimizer_success", False)) for row in rows),
        "optimizer_success_percent": 100.0
        * sum(bool(row.get("optimizer_success", False)) for row in rows)
        / max(branch_attempts, 1),
        "usable_attempts": sum(bool(row.get("usable", False)) for row in rows),
        "finite_grid_points": int(np.isfinite(dv_grid).sum()),
        "optimizer_iterations": sum(int(row.get("nit", 0)) for row in rows),
        "objective_evaluations": sum(int(row.get("nfev", 0)) for row in rows),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_cached_scipy_campaign(comparison_dir: Path) -> tuple[dict, list[dict], np.ndarray, float]:
    """Load the exact high-order configuration and timing from a saved comparison."""
    coefficients_path = comparison_dir / "gondelach_high_order_coefficients.npz"
    attempts_path = comparison_dir / "fig3_attempts.csv"
    grids_path = comparison_dir / "comparison_grids.npz"
    timing_path = comparison_dir / "comparison_timing.csv"
    for path in (coefficients_path, attempts_path, grids_path, timing_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing cached hodographic artifact: {path}")

    with np.load(coefficients_path, allow_pickle=False) as saved:
        dep_grid = np.asarray(saved["grid_departure_mjd2000"], dtype=float)
        tof_grid = np.asarray(saved["grid_tof_days"], dtype=float)
        metadata = {
            "dep_min": float(dep_grid[0]),
            "dep_max": float(dep_grid[-1]),
            "tof_min": float(tof_grid[0]),
            "tof_max": float(tof_grid[-1]),
            "dep_step": float(dep_grid[1] - dep_grid[0]) if len(dep_grid) > 1 else 1.0,
            "tof_step": float(tof_grid[1] - tof_grid[0]) if len(tof_grid) > 1 else 1.0,
            "n_min": int(np.asarray(saved["n_min"]).item()),
            "n_max": int(np.asarray(saved["n_max"]).item()),
            "n_quad": int(np.asarray(saved["n_quad"]).item()),
            "basis": str(np.asarray(saved["basis"]).item()),
            "target": str(np.asarray(saved["target"]).item()),
            "maxfev": int(np.asarray(saved["maxfev"]).item()),
            "xatol": float(np.asarray(saved["xatol"]).item()),
            "fatol": float(np.asarray(saved["fatol"]).item()),
            "simplex_scale": float(np.asarray(saved["simplex_scale"]).item()),
            "ephemeris": str(np.asarray(saved["ephemeris_source"]).item()),
            "spice_meta_kernel": str(np.asarray(saved["spice_meta_kernel"]).item()),
            "spice_target_name": str(np.asarray(saved["spice_target_name"]).item()),
            "departure_grid": dep_grid,
            "tof_grid": tof_grid,
        }

    with attempts_path.open(newline="") as file:
        raw_rows = list(csv.DictReader(file))
    rows = [
        {
            **row,
            "optimizer_success": parse_bool(row.get("optimizer_success")),
            "usable": parse_bool(row.get("usable")),
            "nit": int(row.get("nit", 0)),
            "nfev": int(row.get("nfev", 0)),
            "wall_time_s": float(row.get("wall_time_s", "nan")),
        }
        for row in raw_rows
    ]
    with np.load(grids_path, allow_pickle=False) as saved:
        scipy_grid = np.asarray(saved["fig3_optimizer_delta_v_km_s"], dtype=float)
    with timing_path.open(newline="") as file:
        timing_rows = list(csv.DictReader(file))
    timing = next((row for row in timing_rows if row.get("method") == "gondelach_fig3"), None)
    if timing is None:
        raise ValueError(f"Missing gondelach_fig3 timing row in {timing_path}")
    return metadata, rows, scipy_grid, float(timing["wall_time_s"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASE_CONFIGS), default="mars")
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        help="Reuse the saved scipy/Nelder--Mead campaign and run only IPOPT on its exact grid.",
    )
    parser.add_argument("--dep-min", type=float)
    parser.add_argument("--dep-max", type=float)
    parser.add_argument("--tof-min", type=float)
    parser.add_argument("--tof-max", type=float)
    parser.add_argument("--dep-step", type=float, default=100.0)
    parser.add_argument("--tof-step", type=float, default=100.0)
    parser.add_argument("--n-min", type=int)
    parser.add_argument("--n-max", type=int)
    parser.add_argument("--n-quad", type=int, default=51)
    parser.add_argument("--maxfev", type=int, default=5000, help="Nelder--Mead function-evaluation limit.")
    parser.add_argument("--xatol", type=float, default=1e-6)
    parser.add_argument("--fatol", type=float, default=1e-6)
    parser.add_argument("--simplex-scale", type=float, default=1e-2)
    parser.add_argument("--ipopt-max-iter", type=int, default=1000)
    parser.add_argument("--ipopt-tol", type=float, default=1e-8)
    parser.add_argument("--ipopt-acceptable-tol", type=float, default=1e-6)
    parser.add_argument("--ipopt-linear-solver", default="mumps")
    parser.add_argument(
        "--ipopt-dv-eps",
        type=float,
        default=1e-6,
        help="Smooth Delta-V norm epsilon in canonical acceleration units.",
    )
    parser.add_argument("--ephemeris", choices=["kepler", "spice"], default="kepler")
    parser.add_argument("--spice-meta-kernel")
    parser.add_argument("--spice-target-name")
    parser.add_argument("--output-dir")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()

    cached_rows: list[dict] | None = None
    cached_grid: np.ndarray | None = None
    cached_elapsed: float | None = None
    if args.comparison_dir is not None:
        cached, cached_rows, cached_grid, cached_elapsed = load_cached_scipy_campaign(args.comparison_dir)
        base = {key: value for key, value in cached.items() if key not in {"ephemeris", "spice_meta_kernel", "spice_target_name", "departure_grid", "tof_grid"}}
        configure_ephemeris(cached["ephemeris"], cached["spice_meta_kernel"], cached["spice_target_name"])
        dep_grid_saved = np.asarray(cached["departure_grid"], dtype=float)
        tof_grid_saved = np.asarray(cached["tof_grid"], dtype=float)
    else:
        case = CASE_CONFIGS[args.case]
        configure_ephemeris(args.ephemeris, args.spice_meta_kernel, args.spice_target_name)
        base = {
            "dep_min": float(case["dep_min"] if args.dep_min is None else args.dep_min),
            "dep_max": float(case["dep_max"] if args.dep_max is None else args.dep_max),
            "tof_min": float(case["tof_min"] if args.tof_min is None else args.tof_min),
            "tof_max": float(case["tof_max"] if args.tof_max is None else args.tof_max),
            "dep_step": args.dep_step,
            "tof_step": args.tof_step,
            "n_min": int(case["n_min"] if args.n_min is None else args.n_min),
            "n_max": int(case["n_max"] if args.n_max is None else args.n_max),
            "n_quad": args.n_quad,
            "basis": str(case["higher_basis"]),
            "target": str(case["target"]),
            "maxfev": args.maxfev,
            "xatol": args.xatol,
            "fatol": args.fatol,
            "simplex_scale": args.simplex_scale,
        }
        dep_grid_saved = np.arange(base["dep_min"], base["dep_max"] + 0.5 * base["dep_step"], base["dep_step"])
        tof_grid_saved = np.arange(base["tof_min"], base["tof_max"] + 0.5 * base["tof_step"], base["tof_step"])
    common = {
        **base,
        "oneill_reqmin": 1e-8,
        "oneill_konvge": 10,
        "oneill_factorial_epsilon": 1e-3,
        "ipopt_max_iter": args.ipopt_max_iter,
        "ipopt_tol": args.ipopt_tol,
        "ipopt_acceptable_tol": args.ipopt_acceptable_tol,
        "ipopt_linear_solver": args.ipopt_linear_solver,
        "ipopt_print_level": 0,
        "ipopt_dv_eps": args.ipopt_dv_eps,
        "progress": args.progress,
        "progress_every": args.progress_every,
    }
    output_dir = Path(args.output_dir) if args.output_dir else (
        args.comparison_dir / "hodographic_ipopt_eps1e-6" if args.comparison_dir
        else Path("output/hodographic_optimizer_benchmark")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    grids: dict[str, np.ndarray] = {}
    for optimizer in ("scipy", "ipopt"):
        if optimizer == "scipy" and cached_rows is not None and cached_grid is not None and cached_elapsed is not None:
            rows, dv_grid, elapsed = cached_rows, cached_grid, cached_elapsed
            if args.progress:
                print(f"Reusing scipy/Nelder--Mead results from {args.comparison_dir}")
        else:
            run_args = argparse.Namespace(**common, optimizer=optimizer)
            if args.progress:
                print(f"Running {optimizer}...")
            start = perf_counter()
            _, _, rows, dv_grid, _ = compute_grid(run_args)
            elapsed = perf_counter() - start
        summary = method_summary(optimizer, elapsed, rows, dv_grid)
        summary["dv_eps_canonical"] = args.ipopt_dv_eps if optimizer == "ipopt" else 0.0
        summaries.append(summary)
        grids[optimizer] = dv_grid
        if optimizer != "scipy" or cached_rows is None:
            write_attempts_csv(output_dir / f"hodographic_{optimizer}_attempts.csv", rows)

    common_finite = np.isfinite(grids["scipy"]) & np.isfinite(grids["ipopt"])
    differences = np.abs(grids["scipy"] - grids["ipopt"])
    for row in summaries:
        row["common_finite_grid_points"] = int(common_finite.sum())
        row["median_abs_delta_v_difference_km_s"] = (
            float(np.median(differences[common_finite])) if common_finite.any() else np.nan
        )
        row["max_abs_delta_v_difference_km_s"] = (
            float(np.max(differences[common_finite])) if common_finite.any() else np.nan
        )
    summary_path = output_dir / "hodographic_optimizer_benchmark.csv"
    write_csv(summary_path, summaries)
    np.savez(
        output_dir / "hodographic_optimizer_benchmark_grids.npz",
        scipy_delta_v_km_s=grids["scipy"],
        ipopt_delta_v_km_s=grids["ipopt"],
        departure_mjd2000=dep_grid_saved,
        tof_days=tof_grid_saved,
        **{key: np.asarray(value) for key, value in common.items() if np.isscalar(value)},
    )
    print(f"wrote {summary_path}")
    for row in summaries:
        print(
            f"{row['optimizer']}: {row['wall_time_s']:.3f} s, "
            f"{row['seconds_per_branch']:.4f} s/branch, "
            f"success={row['optimizer_success_percent']:.1f}%"
        )


if __name__ == "__main__":
    main()

"""Compare Sundman shaping families on one transfer from a shared seed.

Every row starts from the same minimum-Delta-V cylindrical B-spline solution.
``hpq`` is the existing ``optimizer.pythagorean_3d_sundman`` solver, seeded
exactly as ``optimizer.optimizers`` seeds it; ``two-spline`` and ``ks`` come
from ``optimizer.sundman_hull_families`` and fit their own primitives to the
same B-spline path.

Configurations are ``family:degree:n_ctrl[:subdivide]``; ``ks-lock`` is ``ks``
with ``--revolution-lock``.  Run from the
repository root, for example::

    python experiments/compare_sundman_families.py \
      --departure-date 2028-06-24 --tof-days 460 --n-rev 1 --mode min-peak \
      --configs hpq:9:10 two-spline:9:10 two-spline:13:14 ks:9:10 ks:13:14
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimizer import optimizers, sundman_hull_families
from optimizer.pythagorean_3d_bernstein import (
    CANONICAL_ACCEL_TO_MM_S2,
    configure_spice_case,
)
from optimizer.pythagorean_3d_sundman import solve as solve_hpq

FIELDS = (
    "family", "degree", "n_ctrl", "spans", "subdivide", "hull_degree",
    "thrust_rows", "free_variables", "status", "checks_passed",
    "delta_v_km_s", "required_peak_mm_s2", "sampled_peak_mm_s2",
    "hull_bound_mm_s2", "hull_excess_percent", "ipopt_iterations",
    "wall_time_s", "swept_revolutions", "directory",
)


def case_arguments(args):
    return ["--ephemeris", "spice",
            "--departure-date", str(args.departure_date),
            "--tof-days", str(args.tof_days), "--n-rev", str(args.n_rev),
            "--spice-meta-kernel", args.spice_meta_kernel,
            "--spice-target-name", args.spice_target_name]


def prepare_seed(args, degree, n_ctrl, directory):
    """B-spline stage plus the (h, p, q, theta) seed for one basis."""
    pipeline = optimizers.build_parser().parse_args([
        *case_arguments(args), "--pythagorean-degree", str(degree),
        "--pythagorean-n-ctrl", str(n_ctrl), "--prepare-only", "--quiet",
        "--output", str(directory)])
    optimizers.run(pipeline)
    return pipeline


def run_hpq(args, degree, n_ctrl, subdivide, directory, u_max):
    pipeline = prepare_seed(args, degree, n_ctrl, directory)
    keys = ("ephemeris", "departure_date", "tof_days", "spice_meta_kernel",
            "spice_target_name", "n_rev", "orbit_arc_samples", "mu", "r0", "rf",
            "phi0", "phif", "rdot0", "rdotf", "phidot0", "phidotf", "theta_dot0",
            "theta_dotf", "theta_final", "tof", "peak_init_factor", "h_min",
            "rho_min", "r_min", "arc_min", "quadrature", "samples")
    return solve_hpq(SimpleNamespace(
        **{key: getattr(pipeline, key) for key in keys},
        mode=args.mode, thrust_enforcement=args.thrust_enforcement,
        constraint_nodes_per_span=15, degree=degree, n_ctrl=n_ctrl, u_max=u_max,
        subdivide=subdivide, max_iter=args.max_iter,
        initial_solution=directory / "pythagorean_sundman_seed.json",
        quiet=True, show_plots=False, no_plots=True,
        output=directory / "pythagorean_sundman",
        _spice_context=getattr(pipeline, "_spice_context", None)))


def run_family(args, family, degree, n_ctrl, subdivide, directory, u_max, seed):
    lock = ["--revolution-lock"] if family == "ks-lock" else []
    family = family.removesuffix("-lock")
    command = [*case_arguments(args), "--family", family, *lock,
               "--fibre", args.fibre,
               "--degree", str(degree), "--n-ctrl", str(n_ctrl),
               "--subdivide", str(subdivide), "--mode", args.mode,
               "--thrust-enforcement", args.thrust_enforcement,
               "--max-iter", str(args.max_iter),
               "--seed-trajectory", str(seed["trajectory"]),
               "--initial-peak-bound", repr(seed["peak"]), "--quiet",
               "--output", str(directory)]
    if u_max is not None:
        command += ["--u-max", repr(u_max)]
    family_args = sundman_hull_families.build_parser().parse_args(command)
    configure_spice_case(family_args)
    sundman_hull_families.validate(family_args)
    return sundman_hull_families.solve(family_args)


def row_from(summary, family, degree, n_ctrl, subdivide, directory, wall, error):
    spans = n_ctrl - degree
    row = {"family": family, "degree": degree, "n_ctrl": n_ctrl, "spans": spans,
           "subdivide": subdivide, "directory": str(directory),
           "wall_time_s": f"{wall:.2f}"}
    if summary is None:
        row["status"] = error
        row["checks_passed"] = False
        return row
    sampled = summary["sampled_peak_acceleration_mm_s2"]
    bound = summary["continuous_bernstein_bound_for_returned_trajectory_mm_s2"]
    row.update({
        "hull_degree": summary["hull_degree"],
        "thrust_rows": summary["number_of_thrust_inequalities"],
        "free_variables": summary["number_of_free_trajectory_variables"],
        "status": summary["status"],
        "checks_passed": summary["numerical_checks_passed"],
        "delta_v_km_s": f"{summary['delta_v_reference_km_s']:.6f}",
        "required_peak_mm_s2": f"{summary['required_peak_acceleration_mm_s2']:.6f}",
        "sampled_peak_mm_s2": f"{sampled:.6f}",
        "hull_bound_mm_s2": f"{bound:.6f}",
        "hull_excess_percent": f"{100.0 * (bound / sampled - 1.0):.3f}",
        "ipopt_iterations": summary["ipopt_iterations"],
        "wall_time_s": f"{summary['wall_time_s']:.2f}",
        "swept_revolutions": (f"{summary['swept_revolutions']:.4f}"
                              if "swept_revolutions" in summary else ""),
    })
    return row


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--departure-date", default="2028-06-24")
    parser.add_argument("--tof-days", type=float, default=460.0)
    parser.add_argument("--n-rev", type=int, default=1)
    parser.add_argument("--spice-meta-kernel",
                        default="kernels/fasttransfer_planets.tm")
    parser.add_argument("--spice-target-name", default="MARS BARYCENTER")
    parser.add_argument("--mode", choices=("min-peak", "min-dv"), default="min-peak")
    parser.add_argument("--thrust-enforcement", choices=("hull", "grid", "none"),
                        default="hull")
    parser.add_argument("--u-max-mm-s2", type=float, default=None,
                        help="min-dv only: certified acceleration cap")
    parser.add_argument("--fibre", choices=("free", "fixed"), default="free")
    parser.add_argument("--max-iter", type=int, default=2500)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    u_max = (None if args.u_max_mm_s2 is None
             else args.u_max_mm_s2 / CANONICAL_ACCEL_TO_MM_S2)

    args.output.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        prepare_seed(args, 3, 7, args.output / "shared_seed")
    seed_json = json.loads(
        (args.output / "shared_seed" / "pythagorean_sundman_seed.json").read_text())
    seed = {"trajectory": args.output / "shared_seed" / "bspline_delta_v_solution.npz",
            "peak": seed_json["initial_peak_acceleration_bound"]}
    source = seed_json["source_bspline"]
    print(f"Shared B-spline seed: dv {source['delta_v_reference_km_s']:.6f} km/s, "
          f"peak {1000.0 * source['peak_acceleration_m_s2']:.6f} mm/s^2")

    rows = []
    for config in args.configs:
        parts = config.split(":")
        family, degree, n_ctrl = parts[0], int(parts[1]), int(parts[2])
        subdivide = int(parts[3]) if len(parts) > 3 else 0
        directory = args.output / (
            f"{family}_deg{degree}_ctrl{n_ctrl}_sub{subdivide}")
        started = perf_counter()
        summary, error = None, ""
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                if family == "hpq":
                    summary = run_hpq(
                        args, degree, n_ctrl, subdivide, directory, u_max)
                else:
                    summary = run_family(args, family, degree, n_ctrl, subdivide,
                                         directory, u_max, seed)
        except (RuntimeError, ValueError) as exc:
            text = str(exc)
            marker = "return_status is '"
            error = (text.split(marker)[1].split("'")[0] if marker in text
                     else text.splitlines()[-1][:60])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "run.log").write_text(log.getvalue())
        rows.append(row_from(summary, family, degree, n_ctrl, subdivide, directory,
                             perf_counter() - started, error))
        row = rows[-1]
        print(" | ".join(str(row.get(key, "")) for key in FIELDS[:-1]), flush=True)

    with (args.output / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output / 'comparison.csv'}")


if __name__ == "__main__":
    main()

"""Epsilon-constraint Pareto front of one transfer: minimum Delta-V under a cap.

Fixes a single departure, time of flight and revolution count, then minimizes
Delta-V subject to a certified peak-acceleration cap that is swept between the
minimum-peak solution, below which the problem is infeasible, and the peak that
the unconstrained minimum-Delta-V solution happens to reach, above which the cap
is inactive.  Every point is therefore a genuine Pareto optimum of the same
transfer, unlike a front assembled from different cells of a design grid.

Each solve is warm-started from the previous cap, which is a true continuation:
the boundary conditions do not move, only the constraint tightens.

Run from the repository root, for example::

    python experiments/pareto_epsilon_constraint.py \
      --departure-date 2028-03-21 --tof-days 820 --n-rev 1 \
      --seed output/mars_min_peak_seeded/dep_2028-03-21/N1/tof_820/seeded
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimizer.pythagorean_3d_bernstein import CANONICAL_ACCEL_TO_MM_S2


def solve_capped(directory, u_max, seed, args):
    """Minimum Delta-V under one acceleration cap, continued from the last one."""
    command = [
        sys.executable, "-m", "optimizer.pythagorean_3d_sundman",
        "--ephemeris", "spice",
        "--departure-date", args.departure_date,
        "--tof-days", str(args.tof_days),
        "--spice-target-name", args.spice_target_name,
        "--spice-meta-kernel", args.spice_meta_kernel,
        "--n-rev", str(args.n_rev),
        "--mode", "min-dv",
        "--u-max", repr(float(u_max)),
        "--thrust-enforcement", args.thrust_enforcement,
        "--n-ctrl", str(args.n_ctrl),
        "--degree", str(args.degree),
        "--subdivide", str(args.subdivide),
        "--quadrature", str(args.quadrature),
        "--samples", str(args.samples),
        "--max-iter", str(args.max_iter),
        "--initial-solution", str(seed),
        "--output", str(directory),
        "--quiet", "--no-plots",
    ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               check=False)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.log").write_text(completed.stdout)
    path = directory / "solution.json"
    if completed.returncode != 0 or not path.exists():
        status = ""
        if "return_status is " in completed.stdout:
            status = completed.stdout.rsplit("return_status is ", 1)[1].splitlines()
            status = status[0].strip().strip("'\"., ") if status else ""
        return None, status or "failed"
    return json.loads(path.read_text()), ""


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--departure-date", required=True)
    parser.add_argument("--tof-days", type=float, required=True)
    parser.add_argument("--n-rev", type=int, default=1)
    parser.add_argument("--spice-target-name", default="MARS BARYCENTER")
    parser.add_argument("--spice-meta-kernel", default="kernels/fasttransfer_planets.tm")
    parser.add_argument("--seed", type=Path, required=True,
                        help="Directory of a solved cell, holding solution.json")
    parser.add_argument("--u-max-min", type=float, default=None,
                        help="Lowest cap [mm/s^2]; defaults to the seed's own bound")
    parser.add_argument("--u-max-max", type=float, default=None,
                        help="Highest cap [mm/s^2]; required unless --relax is used")
    parser.add_argument("--relax", type=float, default=1.35,
                        help="If no upper cap is given, sweep up to this multiple "
                             "of the lowest cap")
    parser.add_argument("--points", type=int, default=25)
    parser.add_argument("--margin", type=float, default=1.002,
                        help="The lowest cap is raised by this factor, since the "
                             "seed sits exactly on its own bound")
    parser.add_argument("--n-ctrl", type=int, default=7)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--subdivide", type=int, default=0)
    parser.add_argument("--quadrature", type=int, default=12)
    parser.add_argument("--samples", type=int, default=1001)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--thrust-enforcement", choices=("hull", "grid"), default="hull")
    parser.add_argument("--figure-format", choices=("pdf", "png"), default="pdf")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("output/pareto_epsilon_constraint"))
    args = parser.parse_args()

    seed_path = args.seed / "solution.json"
    if not seed_path.exists():
        seed_path = args.seed / "pythagorean_sundman" / "solution.json"
    if not seed_path.exists():
        parser.error(f"No solution.json under {args.seed}")
    seed = json.loads(seed_path.read_text())
    lowest = args.u_max_min or seed["required_peak_acceleration_mm_s2"] * args.margin
    highest = args.u_max_max or lowest * args.relax
    if not highest > lowest:
        parser.error("the upper cap must exceed the lower one")
    caps = np.linspace(lowest, highest, args.points)

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Transfer fijo: salida {args.departure_date}, TOF {args.tof_days:g} d, "
          f"N={args.n_rev}")
    print(f"Barrido de u_max: {lowest:.6f} a {highest:.6f} mm/s^2, "
          f"{args.points} puntos\n")
    print(f"{'u_max [mm/s2]':>14} {'dv [km/s]':>11} {'pico [mm/s2]':>13} "
          f"{'cota [mm/s2]':>13}  activa")

    rows = []
    current_seed = seed_path
    started = perf_counter()
    for index, cap in enumerate(caps):
        directory = args.output / f"cap_{index:03d}"
        solution, message = solve_capped(
            directory, cap / CANONICAL_ACCEL_TO_MM_S2, current_seed, args)
        if solution is None:
            print(f"{cap:14.6f} {'-':>11} {'-':>13} {'-':>13}  FALLO: {message}")
            rows.append({"u_max_mm_s2": cap, "delta_v_km_s": "", "peak_mm_s2": "",
                         "hull_bound_mm_s2": "", "active": "", "status": message})
            continue
        peak = solution["sampled_peak_acceleration_mm_s2"]
        bound = solution["continuous_bernstein_bound_for_returned_trajectory_mm_s2"]
        active = bound >= 0.999 * cap
        print(f"{cap:14.6f} {solution['delta_v_reference_km_s']:11.5f} "
              f"{peak:13.6f} {bound:13.6f}  {'si' if active else 'no'}")
        rows.append({"u_max_mm_s2": cap,
                     "delta_v_km_s": solution["delta_v_reference_km_s"],
                     "peak_mm_s2": peak, "hull_bound_mm_s2": bound,
                     "active": active, "status": solution["status"]})
        current_seed = directory / "solution.json"

    csv_path = args.output / "pareto_points.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    solved = [r for r in rows if r["delta_v_km_s"] != ""]
    if solved:
        import matplotlib
        if not args.show_plots:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.asarray([float(r["delta_v_km_s"]) for r in solved])
        y = np.asarray([float(r["u_max_mm_s2"]) for r in solved])
        figure, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.plot(x, y, marker="o", markersize=4.5, linewidth=1.8, color="tab:blue",
                label="Pareto front")
        ax.set(xlabel=r"$\Delta V$ [km/s]",
               ylabel=r"peak acceleration cap $u_{\max}$ [mm/s$^2$]",
               title=f"Earth to {args.spice_target_name.split()[0].title()}, "
                     f"departure {args.departure_date}, "
                     f"TOF {args.tof_days:g} d, N={args.n_rev}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
        figure.tight_layout()
        path = args.output / f"pareto_epsilon.{args.figure_format}"
        figure.savefig(path, dpi=180)
        if args.show_plots:
            plt.show()
        print(f"\n{len(solved)} de {len(rows)} puntos resueltos en "
              f"{perf_counter() - started:.1f} s")
        print(f"Delta-V de {x.min():.5f} a {x.max():.5f} km/s")
        print(f"Figura: {path.resolve()}")
    print(f"Puntos: {csv_path.resolve()}")


if __name__ == "__main__":
    main()

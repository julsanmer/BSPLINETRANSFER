"""Show where node-wise thrust limiting fails and the convex hull does not.

Solves the same minimum-peak problem twice: once limiting the acceleration only
at quadrature nodes, and once with the rational-Bezier convex hull condition that
holds on the whole arc.  The node-limited solution satisfies its constraints at
every node while exceeding the limit between them, by an amount that grows as the
nodes thin out.  The hull solution cannot do this by construction.

Run from the repository root, for example::

    python experiments/demo_hull_vs_grid.py --nodes 15 --degree 9 --n-ctrl 10
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimizer.pythagorean_3d_bernstein import CANONICAL_ACCEL_TO_MM_S2


def solve(directory, enforcement, args, nodes=None):
    command = [
        sys.executable, "-m", "optimizer.pythagorean_3d_sundman",
        "--mode", args.mode, "--thrust-enforcement", enforcement,
        "--degree", str(args.degree), "--n-ctrl", str(args.n_ctrl),
        "--samples", str(args.samples), "--output", str(directory),
        "--quiet", "--no-plots",
    ]
    if args.u_max is not None:
        command += ["--u-max", repr(args.u_max / CANONICAL_ACCEL_TO_MM_S2)]
    if nodes is not None:
        command += ["--constraint-nodes-per-span", str(nodes)]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               check=False)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.log").write_text(completed.stdout)
    path = directory / "solution.json"
    if not path.exists():
        raise SystemExit(f"{enforcement} solve produced no solution; see {directory}")
    table = np.loadtxt(directory / "trajectory.csv", delimiter=",", skiprows=1)
    return json.loads(path.read_text()), table


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("min-peak", "min-dv"),
                        default="min-peak",
                        help="min-peak drives the limit down; min-dv treats it as "
                             "a fixed cap and needs --u-max")
    parser.add_argument("--u-max", type=float, default=None,
                        help="Fixed acceleration cap [mm/s^2] for min-dv")
    parser.add_argument("--nodes", type=int, default=15,
                        help="Constraint nodes per span for the grid solve")
    parser.add_argument("--degree", type=int, default=9)
    parser.add_argument("--n-ctrl", type=int, default=10)
    parser.add_argument("--samples", type=int, default=4001)
    parser.add_argument("--figure-format", choices=("pdf", "png"), default="pdf")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("output/hull_vs_grid"))
    args = parser.parse_args()
    if args.mode == "min-dv" and args.u_max is None:
        parser.error("min-dv needs a cap; pass --u-max in mm/s^2")
    if args.mode == "min-peak" and args.u_max is not None:
        parser.error("u-max does not apply to min-peak, whose limit is the result")

    grid, grid_table = solve(args.output / "grid", "grid", args, args.nodes)
    hull, hull_table = solve(args.output / "hull", "hull", args)

    # trajectory.csv: time, sigma, ..., u_norm at column 12
    spans = args.n_ctrl - args.degree
    node_sigma = np.concatenate([
        np.linspace(s / spans, (s + 1) / spans, args.nodes) for s in range(spans)])
    node_time = np.interp(node_sigma, grid_table[:, 1], grid_table[:, 0])
    node_norm = np.interp(node_sigma, grid_table[:, 1], grid_table[:, 12])

    scale = grid["sampled_peak_acceleration_mm_s2"] / np.max(grid_table[:, 12])
    grid_limit = grid["required_peak_acceleration_mm_s2"]
    hull_limit = hull["required_peak_acceleration_mm_s2"]

    import matplotlib
    if not args.show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(grid_table[:, 0], scale * grid_table[:, 12], color="tab:red",
            linewidth=1.6, label=f"Node-limited, {args.nodes} nodes per span")
    ax.plot(node_time, scale * node_norm, "o", color="tab:red", markersize=5,
            markerfacecolor="white", zorder=4, label="Constraint nodes")
    ax.plot(hull_table[:, 0], scale * hull_table[:, 12], color="tab:blue",
            linewidth=1.6, label="Convex hull certificate")
    if args.mode == "min-dv":
        # Both runs are given the same cap, so one line serves for both.
        ax.axhline(args.u_max, color="0.2", linestyle="--", linewidth=1.3,
                   label=f"Imposed cap ({args.u_max:.4f})")
    else:
        ax.axhline(grid_limit, color="tab:red", linestyle="--", linewidth=1.2,
                   label=f"Limit it believes it imposes ({grid_limit:.4f})")
        ax.axhline(hull_limit, color="tab:blue", linestyle="--", linewidth=1.2,
                   label=f"Certified bound ({hull_limit:.4f})")

    reference = args.u_max if args.mode == "min-dv" else grid_limit
    excess = 100.0 * (grid["sampled_peak_acceleration_mm_s2"] / reference - 1.0)
    objective = ("minimum peak" if args.mode == "min-peak"
                 else r"minimum $\Delta V$ under a fixed cap")
    ax.set(xlabel="Time [canonical]", ylabel=r"$\|\mathbf{u}\|$ [mm/s$^2$]",
           title=f"Degree-{args.degree} Pythagorean, {objective}\n"
                 f"node-limited solution exceeds the limit by {excess:.1f}% "
                 "between nodes")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    figure.tight_layout()
    path = args.output / f"hull_vs_grid.{args.figure_format}"
    figure.savefig(path, dpi=180)
    if args.show_plots:
        plt.show()

    print(f"{'enforcement':<22}{'limit imposed':>15}{'true dense peak':>17}{'excess':>11}")
    for label, solution in (("node-limited", grid), ("convex hull", hull)):
        limit = solution["required_peak_acceleration_mm_s2"]
        dense = solution["sampled_peak_acceleration_mm_s2"]
        print(f"{label:<22}{limit:15.6f}{dense:17.6f}{100 * (dense / limit - 1):10.3f}%"
              + (f"   dv {solution['delta_v_reference_km_s']:.5f} km/s"
                 if args.mode == "min-dv" else ""))
    print(f"\nFigura: {path.resolve()}")


if __name__ == "__main__":
    main()

"""Delta-V against peak acceleration for one or more Sundman sweeps.

Each sweep contributes one point per solved cell, so a minimum-Delta-V run and a
minimum-peak run over the same grid appear as two clouds that bracket the
trade-off.  The Pareto front of each cloud is drawn on top.

Run from the repository root, for example::

    python experiments/plot_sundman_pareto.py \
      --sweep output/sundman_continuation_grid_BEST:"Minimum $\\Delta V$" \
      --sweep output/mars_min_peak_seeded:"Minimum peak"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_points(sweep_dir, x_column, y_column):
    """One (Delta-V, peak) point per solved cell, keyed by departure, TOF and N."""
    path = Path(sweep_dir) / "grid_attempts.csv"
    if not path.exists():
        raise SystemExit(f"No grid_attempts.csv in {sweep_dir}")
    points = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if str(row["pythagorean_success"]).strip().lower() != "true":
                continue
            x, y = parse_float(row[x_column]), parse_float(row[y_column])
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            key = (row["departure_date"], parse_float(row["tof_days"]),
                   int(parse_float(row["N"])))
            points[key] = (x, y, parse_float(row["tof_days"]),
                           int(parse_float(row["N"])))
    return points


def pareto_front(x, y):
    """Indices of the non-dominated points, sorted by increasing x."""
    order = np.lexsort((y, x))
    front, best = [], float("inf")
    for index in order:
        if y[index] < best:
            front.append(index)
            best = y[index]
    return np.asarray(front, dtype=int)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", action="append", required=True,
                        metavar="DIR[:LABEL]",
                        help="Sweep directory, optionally with a legend label; "
                             "repeat for each run")
    parser.add_argument("--x-column", default="pythagorean_delta_v_km_s")
    parser.add_argument("--y-column", default="pythagorean_peak_mm_s2")
    parser.add_argument("--x-label", default=r"$\Delta V$ [km/s]")
    parser.add_argument("--y-label", default=r"peak $\|\mathbf{u}\|$ [mm/s$^2$]")
    parser.add_argument("--title", default="Pythagorean Sundman: "
                                           r"$\Delta V$ against peak acceleration")
    parser.add_argument("--unified", action="store_true",
                        help="Pool every run and draw only the non-dominated "
                             "solutions of the union")
    parser.add_argument("--link-cells", action="store_true",
                        help="Join the two runs cell by cell, showing the cost of "
                             "trading Delta-V for peak on each transfer")
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--figure-format", choices=("pdf", "png"), default="pdf")
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    runs = []
    for entry in args.sweep:
        directory, _, label = entry.partition(":")
        runs.append((Path(directory), label or Path(directory).name,
                     read_points(directory, args.x_column, args.y_column)))

    import matplotlib
    if not args.show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    figure, ax = plt.subplots(figsize=(7.4, 5.4))

    if args.unified:
        # One front over the pooled candidates: which run supplied each point is
        # the interesting part, since it says where the useful solutions live.
        pooled_x, pooled_y, origin = [], [], []
        for index, (_, _, points) in enumerate(runs):
            for value in points.values():
                pooled_x.append(value[0])
                pooled_y.append(value[1])
                origin.append(index)
        pooled_x = np.asarray(pooled_x)
        pooled_y = np.asarray(pooled_y)
        origin = np.asarray(origin)
        front = pareto_front(pooled_x, pooled_y)
        ax.plot(pooled_x[front], pooled_y[front], color="0.4", linewidth=1.6,
                zorder=2, label=f"Non-dominated front ({len(front)} solutions)")
        for index, (_, label, _) in enumerate(runs):
            mask = origin[front] == index
            if not np.any(mask):
                continue
            ax.scatter(pooled_x[front][mask], pooled_y[front][mask], s=26,
                       color=palette[index % len(palette)], zorder=3,
                       label=f"{label} ({int(mask.sum())} on the front)")
        ax.set(xlabel=args.x_label, ylabel=args.y_label, title=args.title)
        if args.log_y:
            ax.set_yscale("log")
        if args.x_max:
            ax.set_xlim(left=0.0, right=args.x_max)
        if args.y_max:
            ax.set_ylim(bottom=0.0 if not args.log_y else None, top=args.y_max)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        figure.tight_layout()
        path = args.output or (
            runs[0][0] / f"pareto_unified.{args.figure_format}")
        figure.savefig(path, dpi=180)
        if args.show_plots:
            plt.show()
        print(f"Candidatos agrupados: {len(pooled_x)}")
        print(f"No dominados        : {len(front)}")
        for index, (_, label, _) in enumerate(runs):
            share = int(np.sum(origin[front] == index))
            print(f"   {label:<26} {share:4d} puntos del frente "
                  f"({100.0 * share / len(front):.1f}%)")
        print(f"Figura: {path.resolve()}")
        return

    if args.link_cells and len(runs) >= 2:
        shared = set(runs[0][2]) & set(runs[1][2])
        segments = [[(runs[0][2][k][0], runs[0][2][k][1]),
                     (runs[1][2][k][0], runs[1][2][k][1])] for k in shared]
        if segments:
            from matplotlib.collections import LineCollection
            ax.add_collection(LineCollection(segments, colors="0.75",
                                             linewidths=0.4, alpha=0.5, zorder=1))

    for index, (_, label, points) in enumerate(runs):
        color = palette[index % len(palette)]
        values = np.asarray([[p[0], p[1]] for p in points.values()], dtype=float)
        if not len(values):
            continue
        x, y = values[:, 0], values[:, 1]
        ax.scatter(x, y, s=12, alpha=0.25, color=color, zorder=2,
                   label=f"{label} ({len(x)} cells)")
        front = pareto_front(x, y)
        ax.plot(x[front], y[front], marker="o", markersize=4.0, linewidth=1.8,
                color=color, zorder=3, label=f"{label} Pareto front")

    ax.set(xlabel=args.x_label, ylabel=args.y_label, title=args.title)
    if args.log_y:
        ax.set_yscale("log")
    if args.x_max:
        ax.set_xlim(left=0.0, right=args.x_max)
    if args.y_max:
        ax.set_ylim(bottom=0.0 if not args.log_y else None, top=args.y_max)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    figure.tight_layout()

    path = args.output or (runs[0][0] / f"pareto_dv_vs_peak.{args.figure_format}")
    figure.savefig(path, dpi=180)
    if args.show_plots:
        plt.show()

    for _, label, points in runs:
        values = np.asarray([[p[0], p[1]] for p in points.values()], dtype=float)
        if len(values):
            print(f"{label:<28} {len(values):5d} celdas   "
                  f"{args.x_column} [{values[:, 0].min():.3f}, {values[:, 0].max():.3f}]   "
                  f"{args.y_column} [{values[:, 1].min():.4f}, {values[:, 1].max():.4f}]")
    if len(runs) >= 2:
        shared = set(runs[0][2]) & set(runs[1][2])
        if shared:
            a = np.asarray([[runs[0][2][k][0], runs[0][2][k][1]] for k in shared])
            b = np.asarray([[runs[1][2][k][0], runs[1][2][k][1]] for k in shared])
            print(f"\nSobre {len(shared)} celdas comunes, del primer barrido al segundo:")
            print(f"   {args.x_column}: mediana {np.median(100*(b[:,0]-a[:,0])/a[:,0]):+.2f}%")
            print(f"   {args.y_column}: mediana {np.median(100*(b[:,1]-a[:,1])/a[:,1]):+.2f}%")
    print(f"Figura: {path.resolve()}")


if __name__ == "__main__":
    main()

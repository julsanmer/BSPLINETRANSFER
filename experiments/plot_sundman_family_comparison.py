"""Plot the runs saved by ``compare_sundman_families.py`` for one basis.

Left: heliocentric x-y path.  Right: thrust-acceleration magnitude in time.
Only rows that converged and passed the numerical checks are drawn.  Run from
the repository root, for example::

    python experiments/plot_sundman_family_comparison.py \
      output/family_compare_mars_2028-06-24_tof460_N1/min-peak_hull \
      --degree 5 --n-ctrl 10
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimizer.pythagorean_3d_bernstein import (
    CANONICAL_ACCEL_TO_MM_S2,
    DAYS_PER_CANONICAL_TIME,
)

LABELS = {"hpq": "(h, p, q, θ)", "two-spline": "two-spline", "ks": "KS"}
COLORS = {"hpq": "tab:blue", "two-spline": "tab:orange", "ks": "tab:green"}
SUCCESS = ("Solve_Succeeded", "Solved_To_Acceptable_Level")


def trajectory_file(row):
    directory = Path(row["directory"])
    nested = directory / "pythagorean_sundman" / "trajectory.csv"
    return nested if nested.exists() else directory / "trajectory.csv"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path,
                        help="folder that holds comparison.csv")
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--n-ctrl", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None,
                        help="PNG path (default: inside the folder)")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    with (args.directory / "comparison.csv").open() as file:
        rows = [row for row in csv.DictReader(file)
                if int(row["degree"]) == args.degree
                and int(row["n_ctrl"]) == args.n_ctrl]
    if not rows:
        raise SystemExit("No rows with that degree and control count.")

    figure, (path_axis, thrust_axis) = plt.subplots(1, 2, figsize=(12, 5))
    for row in rows:
        family = row["family"]
        if row["status"] not in SUCCESS or row["checks_passed"] != "True":
            print(f"skipped {family}: {row['status']}")
            continue
        data = np.genfromtxt(trajectory_file(row), delimiter=",", names=True)
        label = (f"{LABELS.get(family, family)}: "
                 f"peak {float(row['sampled_peak_mm_s2']):.4f} mm/s², "
                 f"ΔV {float(row['delta_v_km_s']):.3f} km/s")
        color = COLORS.get(family)
        path_axis.plot(data["x"], data["y"], color=color, label=label)
        thrust_axis.plot(data["time"] * DAYS_PER_CANONICAL_TIME,
                         data["u_norm"] * CANONICAL_ACCEL_TO_MM_S2, color=color)

    path_axis.plot(0.0, 0.0, "o", color="gold", markeredgecolor="k")
    path_axis.set_aspect("equal")
    path_axis.set_xlabel("x [AU]")
    path_axis.set_ylabel("y [AU]")
    path_axis.grid(alpha=0.3)
    path_axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
                     frameon=False)
    thrust_axis.set_xlabel("time [days]")
    thrust_axis.set_ylabel("thrust acceleration [mm/s²]")
    thrust_axis.set_ylim(bottom=0.0)
    thrust_axis.grid(alpha=0.3)
    figure.suptitle(f"{args.directory.parent.name} / {args.directory.name}, "
                    f"degree {args.degree}, {args.n_ctrl} controls")
    figure.tight_layout()

    output = args.output or (
        args.directory / f"families_deg{args.degree}_ctrl{args.n_ctrl}.png")
    figure.savefig(output, dpi=160, bbox_inches="tight")
    print(f"Wrote {output}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

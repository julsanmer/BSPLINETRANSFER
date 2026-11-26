"""Plot the minimum-Delta-V usable trajectory from a saved B-spline run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.postprocess_compare_gondelach_plots import (
    YEAR_DAYS,
    best_delta_v_row,
    format_departure_date,
    read_rows,
    reconstruct_bspline_profile_from_controls,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--control-points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def equalize_3d_axes(ax, positions: np.ndarray) -> None:
    lower = np.nanmin(positions, axis=0)
    upper = np.nanmax(positions, axis=0)
    center = 0.5 * (lower + upper)
    radius = max(0.5 * float(np.max(upper - lower)), 0.1)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def main() -> None:
    args = parse_args()
    best = best_delta_v_row(read_rows(args.attempts), "usable")
    profile = reconstruct_bspline_profile_from_controls(args.control_points, best)

    positions = np.asarray(profile["pos_au"], dtype=float)
    controls = 1000.0 * np.asarray(profile["u_components_m_s2"], dtype=float)
    control_norm = 1000.0 * np.asarray(profile["u_norm_m_s2"], dtype=float)
    time_years = np.asarray(profile["t_days"], dtype=float) / YEAR_DAYS

    fig = plt.figure(figsize=(12.0, 5.2))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.05, 1.0), wspace=0.23)
    trajectory_ax = fig.add_subplot(grid[0, 0], projection="3d")
    control_ax = fig.add_subplot(grid[0, 1])

    trajectory_ax.plot(
        positions[:, 0], positions[:, 1], positions[:, 2],
        color="#1f77b4", linewidth=2.2, label="B-spline trajectory",
    )
    trajectory_ax.scatter([0.0], [0.0], [0.0], s=85, color="#f4d03f", edgecolor="black", label="Sun")
    trajectory_ax.scatter(*positions[0], s=50, marker="o", color="#2ca02c", label="Departure")
    trajectory_ax.scatter(*positions[-1], s=55, marker="s", color="#d62728", label="Rendezvous")
    equalize_3d_axes(trajectory_ax, np.vstack((positions, np.zeros((1, 3)))))
    trajectory_ax.set_xlabel("x [AU]")
    trajectory_ax.set_ylabel("y [AU]")
    trajectory_ax.set_zlabel("z [AU]")
    trajectory_ax.set_title("Best saved B-spline trajectory")
    trajectory_ax.view_init(elev=24.0, azim=-58.0)
    trajectory_ax.legend(loc="upper left", fontsize=9)
    trajectory_ax.grid(True, alpha=0.25)

    styles = [
        (r"$u_r$", "#d55e00"),
        (r"$u_\theta$", "#0072b2"),
        (r"$u_z$", "#009e73"),
    ]
    for component, (label, color) in enumerate(styles):
        control_ax.plot(time_years, controls[:, component], "--", color=color, linewidth=1.7, label=label)
    control_ax.plot(time_years, control_norm, color="0.15", linewidth=2.3, label=r"$\|\mathbf{u}\|$")
    control_ax.axhline(0.0, color="0.75", linewidth=0.8)
    control_ax.set_xlim(0.0, float(time_years[-1]))
    control_ax.set_xlabel("Time [yr]")
    control_ax.set_ylabel(r"Control acceleration [mm/s$^2$]")
    control_ax.set_title("Control history")
    control_ax.legend(ncol=2, loc="best")
    control_ax.grid(True, alpha=0.25)

    summary = (
        f"Departure: {format_departure_date(float(profile['departure_mjd2000']))}   "
        f"TOF: {float(profile['tof_days']):.1f} d   "
        f"N: {int(profile['N'])}   "
        f"$\\Delta V$: {float(profile['delta_v_km_s']):.6f} km/s"
    )
    fig.suptitle(summary, fontsize=13, y=0.985)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.04, right=0.98)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    if args.output.suffix.lower() != ".pdf":
        fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Best attempt: departure={profile['departure_mjd2000']}, TOF={profile['tof_days']} d, "
          f"N={profile['N']}, Delta-V={profile['delta_v_km_s']:.9f} km/s")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

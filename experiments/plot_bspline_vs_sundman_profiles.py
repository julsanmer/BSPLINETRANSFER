"""Plot the B-spline and Sundman Pythagorean transfers of one pipeline run.

Reads an output directory produced by ``python -m optimizer.optimizers`` and
draws the two solutions in the layout used for the Gondelach comparison figure:
one 3D trajectory panel and one control-history panel per method, sharing axis
limits so the two are read against each other directly.

Run from the repository root, for example::

    python experiments/plot_bspline_vs_sundman_profiles.py \
      --run-dir output/dv_compare_tof500_N1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.postprocess_compare_gondelach_plots import (
    CASE_CONFIGS,
    plot_best_transfer_profiles,
    write_profile_csv,
)
from experiments.compact_grid_trajectories import load_trajectory
from optimizer.planar_cubic_bernstein import CANONICAL_ACCEL_TO_M_S2
from optimizer.pythagorean_3d_bernstein import DAYS_PER_CANONICAL_TIME


def inertial_positions(rho, theta, z, rotation):
    """Undo the departure-longitude rotation applied by the SPICE setup."""
    rotated = np.column_stack((rho * np.cos(theta), rho * np.sin(theta), z))
    return rotated @ np.asarray(rotation, dtype=float)


def bspline_profile(run_dir, spice, label):
    """Build a profile dictionary from the saved minimum-Delta-V B-spline arc."""
    archive = np.load(run_dir / "bspline_delta_v_solution.npz")
    seed = json.loads((run_dir / "pythagorean_sundman_seed.json").read_text())
    source = seed["source_bspline"]
    tau = np.asarray(archive["tau"], dtype=float)
    cylindrical = np.asarray(archive["cylindrical_position"], dtype=float)
    control = CANONICAL_ACCEL_TO_M_S2 * np.asarray(archive["control"], dtype=float)
    norm = CANONICAL_ACCEL_TO_M_S2 * np.asarray(archive["control_norm"], dtype=float)
    tof_days = float(spice["tof_days"])
    return {
        "method": label,
        "departure_mjd2000": float(spice["departure_mjd2000"]),
        "tof_days": tof_days,
        "N": int(spice["n_rev"]),
        "t_days": tau * tof_days,
        "pos_au": inertial_positions(cylindrical[:, 0], cylindrical[:, 1],
                                     cylindrical[:, 2], spice["rotation_matrix"]),
        "u_components_m_s2": control,
        "u_norm_m_s2": norm,
        "delta_v_km_s": float(source["delta_v_reference_km_s"]),
        "fmax_m_s2": float(np.max(norm)),
        "success": bool(source["success"]),
        "message": "minimum-Delta-V cylindrical B-spline",
    }


def sundman_profile(run_dir, spice, label):
    """Build a profile dictionary from the Sundman trajectory table."""
    solution = json.loads(
        (run_dir / "pythagorean_sundman" / "solution.json").read_text())
    table = load_trajectory(run_dir)
    # time,sigma,rho,theta,z,R,rho_dot,theta_dot,z_dot,u_rho,u_theta,u_z,u_norm,x,y
    control = CANONICAL_ACCEL_TO_M_S2 * table[:, 9:12]
    norm = CANONICAL_ACCEL_TO_M_S2 * table[:, 12]
    return {
        "method": label,
        "departure_mjd2000": float(spice["departure_mjd2000"]),
        "tof_days": float(spice["tof_days"]),
        "N": int(spice["n_rev"]),
        "t_days": table[:, 0] * DAYS_PER_CANONICAL_TIME,
        "pos_au": inertial_positions(table[:, 2], table[:, 3], table[:, 4],
                                     spice["rotation_matrix"]),
        "u_components_m_s2": control,
        "u_norm_m_s2": norm,
        "delta_v_km_s": float(solution["delta_v_reference_km_s"]),
        "fmax_m_s2": float(np.max(norm)),
        "success": bool(solution["numerical_checks_passed"]),
        "message": str(solution["status"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Output directory of a python -m optimizer.optimizers run")
    parser.add_argument("--case", choices=sorted(CASE_CONFIGS), default="mars",
                        help="Target case, used for the ephemeris arcs in the 3D panels")
    parser.add_argument("--bspline-label", default="Cylindrical B-spline")
    parser.add_argument("--sundman-label", default="Pythagorean Sundman")
    parser.add_argument("--figure-format", choices=("pdf", "png"), default="pdf")
    parser.add_argument("--output", type=Path, default=None,
                        help="Figure path; defaults to the run directory")
    args = parser.parse_args()

    solution_path = args.run_dir / "pythagorean_sundman" / "solution.json"
    if not solution_path.exists():
        parser.error(f"No Sundman solution in {args.run_dir}")
    spice = json.loads(solution_path.read_text())["spice_boundary_data"]
    if not spice:
        parser.error("The run has no SPICE boundary data; only spice runs can be plotted")

    bspline = bspline_profile(args.run_dir, spice, args.bspline_label)
    sundman = sundman_profile(args.run_dir, spice, args.sundman_label)
    path = args.output or (
        args.run_dir / f"bspline_vs_sundman_{args.case}.{args.figure_format}")
    write_profile_csv(args.run_dir / "profile_bspline.csv", bspline)
    write_profile_csv(args.run_dir / "profile_sundman.csv", sundman)
    plot_best_transfer_profiles(
        path, bspline, sundman, args.case,
        bspline_label=args.sundman_label, gondelach_label=args.bspline_label)

    for profile in (bspline, sundman):
        print(f"{profile['method']:<26} Delta V = {profile['delta_v_km_s']:9.4f} km/s   "
              f"peak = {1000.0 * profile['fmax_m_s2']:.6f} mm/s^2")
    print(f"Figure: {path.resolve()}")


if __name__ == "__main__":
    main()

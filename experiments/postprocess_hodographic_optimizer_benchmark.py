"""Postprocess cached Nelder--Mead and IPOPT hodographic fine-grid results."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reproduce_gondelach_fig2 import (  # noqa: E402
    configure_ephemeris,
    evaluate_time_driven_metrics,
)


def scalar(saved: np.lib.npyio.NpzFile, key: str):
    return np.asarray(saved[key]).item()


def load_attempts(path: Path, dv_field: str) -> list[dict]:
    with path.open(newline="") as file:
        raw = list(csv.DictReader(file))
    rows = []
    for row in raw:
        free_fields = sorted(
            (key for key in row if key.startswith("free_")),
            key=lambda key: int(key.split("_")[1]),
        )
        rows.append(
            {
                "departure_mjd2000": float(row["departure_mjd2000"]),
                "tof_days": float(row["tof_days"]),
                "N": int(row["N"]),
                "delta_v_saved_km_s": float(row[dv_field]),
                "optimizer_success": str(row["optimizer_success"]).lower() == "true",
                "message": row["message"],
                "transverse_n_quad": (
                    int(row["transverse_n_quad"])
                    if row.get("transverse_n_quad", "").strip()
                    else None
                ),
                "free_coefficients": np.asarray([float(row[key]) for key in free_fields]),
            }
        )
    return rows


def add_common_metrics(
    rows: list[dict],
    target: str,
    basis: str,
    n_quad: int,
    label: str,
    progress_every: int,
) -> None:
    for index, row in enumerate(rows, start=1):
        metrics = evaluate_time_driven_metrics(
            row["departure_mjd2000"],
            row["tof_days"],
            row["N"],
            n_quad,
            basis,
            row["free_coefficients"],
            target=target,
            transverse_n_quad=row.get("transverse_n_quad"),
        )
        row["delta_v_km_s"] = float(metrics["delta_v_km_s"])
        row["u_max_m_s2"] = float(metrics["fmax_m_s2"])
        if progress_every > 0 and (index == len(rows) or index % progress_every == 0):
            print(f"{label}: evaluated {index}/{len(rows)}")


def pareto_front(rows: list[dict]) -> list[dict]:
    ordered = sorted(
        (
            row
            for row in rows
            if np.isfinite(row["delta_v_km_s"]) and np.isfinite(row["u_max_m_s2"])
        ),
        key=lambda row: (row["delta_v_km_s"], row["u_max_m_s2"]),
    )
    front = []
    best_u = np.inf
    for row in ordered:
        if row["u_max_m_s2"] < best_u:
            front.append(row)
            best_u = row["u_max_m_s2"]
    return front


def write_pareto_csv(path: Path, fronts: dict[str, list[dict]]) -> None:
    fields = [
        "optimizer",
        "departure_mjd2000",
        "tof_days",
        "N",
        "delta_v_km_s",
        "u_max_m_s2",
        "optimizer_success",
        "message",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for optimizer, rows in fronts.items():
            for row in rows:
                writer.writerow({"optimizer": optimizer, **{key: row[key] for key in fields[1:]}})


def plot_pareto(path: Path, fronts: dict[str, list[dict]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "Nelder--Mead": {"color": "#3B6FB6", "marker": "o", "linestyle": "-"},
        "IPOPT": {"color": "#D1495B", "marker": "x", "linestyle": "-."},
    }
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for label, rows in fronts.items():
        ax.plot(
            [row["delta_v_km_s"] for row in rows],
            [row["u_max_m_s2"] for row in rows],
            label=label,
            linewidth=1.6,
            markersize=5.0,
            markeredgewidth=1.2,
            **styles[label],
        )
    ax.set_xlabel(r"$\Delta V$ [km/s]")
    ax.set_ylabel(r"$u_{\max}$ [m/s$^2$]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def write_best_dv_vs_tof_csv(
    path: Path,
    tof: np.ndarray,
    nm_grid: np.ndarray,
    ipopt_grid: np.ndarray,
) -> None:
    fields = ["tof_days", "nelder_mead_best_delta_v_km_s", "ipopt_best_delta_v_km_s"]
    nm_best = np.nanmin(nm_grid, axis=1)
    ipopt_best = np.nanmin(ipopt_grid, axis=1)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for duration, nm_value, ipopt_value in zip(tof, nm_best, ipopt_best):
            writer.writerow(
                {
                    "tof_days": float(duration),
                    "nelder_mead_best_delta_v_km_s": float(nm_value),
                    "ipopt_best_delta_v_km_s": float(ipopt_value),
                }
            )


def plot_best_dv_vs_tof(
    path: Path,
    tof: np.ndarray,
    nm_grid: np.ndarray,
    ipopt_grid: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nm_best = np.nanmin(nm_grid, axis=1)
    ipopt_best = np.nanmin(ipopt_grid, axis=1)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(
        tof,
        nm_best,
        color="#3B6FB6",
        linewidth=1.8,
        marker="o",
        markersize=4.0,
        markevery=max(1, len(tof) // 15),
        label="Nelder--Mead",
    )
    ax.plot(
        tof,
        ipopt_best,
        color="#D1495B",
        linewidth=1.8,
        linestyle="-.",
        marker="x",
        markersize=5.0,
        markeredgewidth=1.2,
        markevery=max(1, len(tof) // 15),
        label="IPOPT",
    )
    ax.set_xlabel("Time of flight [days]")
    ax.set_ylabel(r"Best $\Delta V$ [km/s]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def write_discrepancy_statistics(
    path: Path,
    nm_grid: np.ndarray,
    ipopt_grid: np.ndarray,
) -> None:
    difference = ipopt_grid - nm_grid
    finite = np.asarray(difference[np.isfinite(difference)], dtype=float)
    absolute = np.abs(finite)
    fields = [
        "common_finite_points",
        "nm_mean_delta_v_km_s",
        "nm_median_delta_v_km_s",
        "nm_minimum_delta_v_km_s",
        "nm_maximum_delta_v_km_s",
        "ipopt_mean_delta_v_km_s",
        "ipopt_median_delta_v_km_s",
        "ipopt_minimum_delta_v_km_s",
        "ipopt_maximum_delta_v_km_s",
        "mean_ipopt_minus_nm_km_s",
        "median_ipopt_minus_nm_km_s",
        "median_absolute_difference_km_s",
        "p95_absolute_difference_km_s",
        "p99_absolute_difference_km_s",
        "maximum_absolute_difference_km_s",
        "points_within_1e-6_km_s",
        "points_above_1e-3_km_s",
        "points_above_0p1_km_s",
        "ipopt_lower_points",
        "nm_lower_points",
    ]
    common = np.isfinite(nm_grid) & np.isfinite(ipopt_grid)
    nm_values = np.asarray(nm_grid[common], dtype=float)
    ipopt_values = np.asarray(ipopt_grid[common], dtype=float)
    row = {
        "common_finite_points": len(finite),
        "nm_mean_delta_v_km_s": float(np.mean(nm_values)),
        "nm_median_delta_v_km_s": float(np.median(nm_values)),
        "nm_minimum_delta_v_km_s": float(np.min(nm_values)),
        "nm_maximum_delta_v_km_s": float(np.max(nm_values)),
        "ipopt_mean_delta_v_km_s": float(np.mean(ipopt_values)),
        "ipopt_median_delta_v_km_s": float(np.median(ipopt_values)),
        "ipopt_minimum_delta_v_km_s": float(np.min(ipopt_values)),
        "ipopt_maximum_delta_v_km_s": float(np.max(ipopt_values)),
        "mean_ipopt_minus_nm_km_s": float(np.mean(finite)),
        "median_ipopt_minus_nm_km_s": float(np.median(finite)),
        "median_absolute_difference_km_s": float(np.median(absolute)),
        "p95_absolute_difference_km_s": float(np.percentile(absolute, 95)),
        "p99_absolute_difference_km_s": float(np.percentile(absolute, 99)),
        "maximum_absolute_difference_km_s": float(np.max(absolute)),
        "points_within_1e-6_km_s": int(np.sum(absolute <= 1e-6)),
        "points_above_1e-3_km_s": int(np.sum(absolute > 1e-3)),
        "points_above_0p1_km_s": int(np.sum(absolute > 0.1)),
        "ipopt_lower_points": int(np.sum(finite < -1e-6)),
        "nm_lower_points": int(np.sum(finite > 1e-6)),
    }
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def plot_difference(
    path: Path,
    departure: np.ndarray,
    tof: np.ndarray,
    difference: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    finite = np.abs(difference[np.isfinite(difference)])
    vmax = float(np.max(finite))
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    image = ax.pcolormesh(
        departure,
        tof,
        difference,
        shading="auto",
        cmap="RdBu_r",
        norm=SymLogNorm(linthresh=1e-5, linscale=0.7, vmin=-vmax, vmax=vmax, base=10),
    )
    ax.set_xlabel("Departure date [MJD2000]")
    ax.set_ylabel("Time of flight [days]")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"$\Delta V_{\mathrm{IPOPT}}-\Delta V_{\mathrm{NM}}$ [km/s]")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--ipopt-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument(
        "--skip-pareto-evaluation",
        action="store_true",
        help="Keep an already generated Pareto plot and create only grid-based outputs.",
    )
    args = parser.parse_args()

    comparison_dir = args.comparison_dir
    ipopt_dir = args.ipopt_dir or comparison_dir / "hodographic_ipopt_eps1e-6"
    output_dir = args.output_dir or ipopt_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(comparison_dir / "gondelach_high_order_coefficients.npz", allow_pickle=False) as saved:
        basis = str(scalar(saved, "basis"))
        target = str(scalar(saved, "target"))
        n_quad = int(scalar(saved, "n_quad"))
        configure_ephemeris(
            str(scalar(saved, "ephemeris_source")),
            str(scalar(saved, "spice_meta_kernel")),
            str(scalar(saved, "spice_target_name")),
        )

    if not args.skip_pareto_evaluation:
        nm_rows = load_attempts(comparison_dir / "fig3_attempts.csv", "delta_v_optimizer_km_s")
        ipopt_rows = load_attempts(ipopt_dir / "hodographic_ipopt_attempts.csv", "delta_v_km_s")
        add_common_metrics(nm_rows, target, basis, n_quad, "Nelder--Mead", args.progress_every)
        add_common_metrics(ipopt_rows, target, basis, n_quad, "IPOPT", args.progress_every)
        fronts = {"Nelder--Mead": pareto_front(nm_rows), "IPOPT": pareto_front(ipopt_rows)}
        write_pareto_csv(output_dir / "hodographic_optimizer_pareto_fronts.csv", fronts)
        plot_pareto(output_dir / "hodographic_optimizer_pareto_fronts.pdf", fronts)

    with np.load(ipopt_dir / "hodographic_optimizer_benchmark_grids.npz", allow_pickle=False) as saved:
        departure = np.asarray(saved["departure_mjd2000"], dtype=float)
        tof = np.asarray(saved["tof_days"], dtype=float)
        nm_grid = np.asarray(saved["scipy_delta_v_km_s"], dtype=float)
        ipopt_grid = np.asarray(saved["ipopt_delta_v_km_s"], dtype=float)
    difference = ipopt_grid - nm_grid
    write_best_dv_vs_tof_csv(
        output_dir / "hodographic_optimizer_best_delta_v_vs_tof.csv",
        tof,
        nm_grid,
        ipopt_grid,
    )
    plot_best_dv_vs_tof(
        output_dir / "hodographic_optimizer_best_delta_v_vs_tof.pdf",
        tof,
        nm_grid,
        ipopt_grid,
    )
    write_discrepancy_statistics(
        output_dir / "hodographic_optimizer_delta_v_statistics.csv",
        nm_grid,
        ipopt_grid,
    )
    plot_difference(output_dir / "hodographic_optimizer_delta_v_difference.pdf", departure, tof, difference)
    print(f"wrote {output_dir / 'hodographic_optimizer_pareto_fronts.pdf'}")
    print(f"wrote {output_dir / 'hodographic_optimizer_best_delta_v_vs_tof.pdf'}")
    print(f"wrote {output_dir / 'hodographic_optimizer_delta_v_difference.pdf'}")


if __name__ == "__main__":
    main()

"""Plot cached 10-control-point Bezier and B-spline comparisons.

This script performs no trajectory optimization.  It reads the saved attempt
CSVs, retains usable solutions, and creates:

* minimum Delta-V over departure date and revolution count versus time of flight;
* nondominated Delta-V versus maximum-control-acceleration fronts.

The degree-nine, 10-control-point Bezier archive is selected automatically.
When several archives exist, an analytic-seed archive is preferred so that the
comparison uses the same MEE-only initialization as the nominal B-spline runs.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT_DIR = Path(
    "output/compare_gondelach_1989ml_2028_spice_bspline_dep20d_tof20d"
)


@dataclass(frozen=True)
class MethodData:
    label: str
    path: Path
    departure_mjd2000: np.ndarray
    tof_days: np.ndarray
    revolutions: np.ndarray
    delta_v_km_s: np.ndarray
    u_max_m_s2: np.ndarray
    total_rows: int
    seed_sources: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bezier-attempts",
        type=Path,
        default=None,
        help="Explicit Bezier attempts CSV; otherwise an analytic-seed archive is selected.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["pdf", "png", "svg"],
        default=["pdf", "png"],
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite_float(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        text = str(row.get(name, "")).strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            continue
        if np.isfinite(value):
            return value
    return None


def read_method(path: Path, label: str) -> MethodData:
    departures: list[float] = []
    times_of_flight: list[float] = []
    revolutions: list[int] = []
    delta_v: list[float] = []
    u_max: list[float] = []
    seed_sources: set[str] = set()
    total_rows = 0

    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            total_rows += 1
            seed_source = str(row.get("seed_source", "")).strip()
            if seed_source:
                seed_sources.add(seed_source)
            if not parse_bool(row.get("usable", row.get("success", False))):
                continue
            departure = finite_float(row, "departure_mjd2000")
            tof = finite_float(row, "tof_days")
            dv = finite_float(row, "delta_v_reference_km_s", "delta_v_km_s")
            acceleration = finite_float(
                row,
                "u_max_reference_m_s2",
                "max_u_fine",
            )
            revolution = finite_float(row, "N")
            if None in (departure, tof, dv, acceleration, revolution):
                continue
            if dv <= 0.0 or acceleration < 0.0:
                continue
            departures.append(float(departure))
            times_of_flight.append(float(tof))
            revolutions.append(int(revolution))
            delta_v.append(float(dv))
            u_max.append(float(acceleration))

    if not delta_v:
        raise ValueError(f"No usable reference-metric solutions found in {path}")
    return MethodData(
        label=label,
        path=path,
        departure_mjd2000=np.asarray(departures),
        tof_days=np.asarray(times_of_flight),
        revolutions=np.asarray(revolutions),
        delta_v_km_s=np.asarray(delta_v),
        u_max_m_s2=np.asarray(u_max),
        total_rows=total_rows,
        seed_sources=tuple(sorted(seed_sources)),
    )


def archive_seed_sources(path: Path) -> set[str]:
    sources: set[str] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            source = str(row.get("seed_source", "")).strip()
            if source:
                sources.add(source)
    return sources


def select_bezier_archive(output_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit if explicit.is_absolute() else Path.cwd() / explicit
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    candidates = list(output_dir.glob("bezier_nctrl10_deg9_*_attempts.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No bezier_nctrl10_deg9_*_attempts.csv found in {output_dir}"
        )

    def preference(path: Path) -> tuple[int, float]:
        sources = archive_seed_sources(path)
        analytic_only = bool(sources) and sources <= {"analytic"}
        return (int(analytic_only), path.stat().st_mtime)

    return max(candidates, key=preference)


def minimum_delta_v_by_tof(data: MethodData) -> tuple[np.ndarray, np.ndarray]:
    tof_values = np.unique(data.tof_days)
    minimum = np.asarray(
        [np.min(data.delta_v_km_s[data.tof_days == tof]) for tof in tof_values]
    )
    return tof_values, minimum


def pareto_front(data: MethodData) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((data.u_max_m_s2, data.delta_v_km_s))
    best_u_max = np.inf
    selected: list[int] = []
    for index in order:
        value = float(data.u_max_m_s2[index])
        tolerance = 1.0e-14 * max(1.0, abs(value))
        if not selected or value < best_u_max - tolerance:
            selected.append(int(index))
            best_u_max = value
    selected_array = np.asarray(selected, dtype=int)
    plot_order = np.argsort(data.u_max_m_s2[selected_array])
    selected_array = selected_array[plot_order]
    return data.u_max_m_s2[selected_array], data.delta_v_km_s[selected_array]


def save_figure(fig, stem: Path, formats: list[str], dpi: int) -> list[Path]:
    paths = []
    for suffix in formats:
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    return paths


def write_summary(path: Path, methods: list[MethodData]) -> None:
    fields = [
        "method",
        "source_file",
        "total_attempts",
        "usable_solutions",
        "tof_samples",
        "pareto_points",
        "minimum_delta_v_km_s",
        "minimum_u_max_m_s2",
        "seed_sources",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for data in methods:
            pareto_u, _ = pareto_front(data)
            writer.writerow(
                {
                    "method": data.label,
                    "source_file": data.path.name,
                    "total_attempts": data.total_rows,
                    "usable_solutions": len(data.delta_v_km_s),
                    "tof_samples": len(np.unique(data.tof_days)),
                    "pareto_points": len(pareto_u),
                    "minimum_delta_v_km_s": float(np.min(data.delta_v_km_s)),
                    "minimum_u_max_m_s2": float(np.min(data.u_max_m_s2)),
                    "seed_sources": ";".join(data.seed_sources),
                }
            )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)

    bezier_path = select_bezier_archive(output_dir, args.bezier_attempts)
    paths_and_labels = [
        (output_dir / "bspline_nctrl10_deg3_attempts.csv", "Cubic B-spline"),
        (output_dir / "bspline10_attempts.csv", "Quintic B-spline"),
        (bezier_path, "Degree-nine B\N{LATIN SMALL LETTER E WITH ACUTE}zier"),
    ]
    for path, _ in paths_and_labels:
        if not path.is_file():
            raise FileNotFoundError(path)
    methods = [read_method(path, label) for path, label in paths_and_labels]

    mpl_config = output_dir / ".matplotlib"
    mpl_config.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.0,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )
    colors = ["#0072B2", "#D55E00", "#009E73"]
    markers = ["o", "s", "D"]

    fig_tof, ax_tof = plt.subplots(figsize=(7.0, 4.6))
    for data, color, marker in zip(methods, colors, markers):
        tof, delta_v = minimum_delta_v_by_tof(data)
        ax_tof.plot(
            tof,
            delta_v,
            color=color,
            marker=marker,
            markevery=max(1, len(tof) // 12),
            markersize=4.0,
            linewidth=1.7,
            label=data.label,
        )
    ax_tof.set_xlabel("Time of flight [days]")
    ax_tof.set_ylabel(r"Minimum $\Delta V$ [km/s]")
    ax_tof.set_title(
        "Earth\N{EN DASH}1989 ML: minimum "
        r"$\Delta V$ over departure date and $N$"
    )
    ax_tof.legend()
    fig_tof.tight_layout()
    written = save_figure(
        fig_tof,
        output_dir / "comparison_tof_bezier_vs_bspline10_1989ML",
        args.formats,
        args.dpi,
    )
    plt.close(fig_tof)

    fig_pareto, ax_pareto = plt.subplots(figsize=(7.0, 4.6))
    for data, color, marker in zip(methods, colors, markers):
        u_max, delta_v = pareto_front(data)
        ax_pareto.plot(
            u_max,
            delta_v,
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.7,
            label=f"{data.label} ({len(u_max)} points)",
        )
    ax_pareto.set_xlabel(r"Maximum control acceleration, $u_{\max}$ [m/s$^2$]")
    ax_pareto.set_ylabel(r"$\Delta V$ [km/s]")
    ax_pareto.set_title(
        "Earth\N{EN DASH}1989 ML: nondominated "
        r"$\Delta V$--$u_{\max}$ fronts"
    )
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    ax_pareto.xaxis.set_major_formatter(formatter)
    ax_pareto.legend()
    fig_pareto.tight_layout()
    written.extend(
        save_figure(
            fig_pareto,
            output_dir / "comparison_pareto_bezier_vs_bspline10_1989ML",
            args.formats,
            args.dpi,
        )
    )
    plt.close(fig_pareto)

    summary_path = output_dir / "comparison_bezier_vs_bspline10_summary.csv"
    write_summary(summary_path, methods)
    print(f"selected Bezier archive: {bezier_path}")
    for data in methods:
        print(
            f"{data.label}: {len(data.delta_v_km_s)}/{data.total_rows} usable, "
            f"min Delta-V={np.min(data.delta_v_km_s):.6f} km/s, "
            f"seed={','.join(data.seed_sources) or 'not recorded'}"
        )
    for path in written:
        print(f"wrote {path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

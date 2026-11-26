"""Draw Delta-V porkchop maps from a B-spline to Sundman continuation sweep.

Reads the ``grid_attempts.csv`` written by
``experiments/run_bspline_to_sundman_grid.py`` and renders, on a departure by
time-of-flight grid, the best value found over the swept revolution counts.
Cells with no accepted solution stay gray; they are missing data, not evidence
that the transfer is infeasible.

Run from the repository root, for example::

    python experiments/plot_sundman_porkchop.py \
      --sweep-dir output/porkchop_sundman_min_dv
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Delta-V colour buckets and palette of the B-spline shaping article, so that
# these maps can be read against its figures without rescaling by eye.
GONDELACH_BLUE = ["#05008f", "#0035ff", "#087cff", "#12a7ee", "#17cfe2",
                  "#00e0bd", "#78ff3a"]
DELTA_V_BINS = {
    "paper": ([6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 40.0], "neither"),
    "low-dv": ([5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 40.0], "neither"),
    "mercury": ([18.0, 20.0, 22.0, 24.0, 26.0, 30.0, 40.0, 80.0, 160.0], "max"),
    "tempel1": ([12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 30.0, 60.0], "max"),
    "1989ml": ([4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 30.0, 60.0], "max"),
}


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mjd2000_label(value):
    return (datetime(2000, 1, 1, 12) + timedelta(days=float(value))).strftime("%Y-%m-%d")


def build_maps(rows, column, select_by=None):
    """Return the grid axes, the retained value over N, and the winning N.

    ``select_by`` picks which column decides the revolution count; the reported
    value still comes from ``column``.  Mapping the peak of the Delta-V optimal
    solution is not the same as mapping the lowest peak available, and the two
    answer different questions.
    """
    select_by = select_by or column
    departures = np.array(sorted({parse_float(r["departure_mjd2000"]) for r in rows}))
    tofs = np.array(sorted({parse_float(r["tof_days"]) for r in rows}))
    revolutions = np.array(sorted({int(parse_float(r["N"])) for r in rows}))
    stack = np.full((len(revolutions), len(tofs), len(departures)), np.nan)
    ranking = np.full_like(stack, np.nan)
    for row in rows:
        if str(row["pythagorean_success"]).strip().lower() != "true":
            continue
        value = parse_float(row[column])
        key = parse_float(row[select_by])
        if not np.isfinite(value) or not np.isfinite(key):
            continue
        i = int(np.argmin(np.abs(revolutions - int(parse_float(row["N"])))))
        j = int(np.argmin(np.abs(tofs - parse_float(row["tof_days"]))))
        k = int(np.argmin(np.abs(departures - parse_float(row["departure_mjd2000"]))))
        stack[i, j, k] = value
        ranking[i, j, k] = key
    available = np.where(np.isfinite(ranking), ranking, np.inf)
    chosen = np.argmin(available, axis=0)
    best = np.take_along_axis(stack, chosen[None], axis=0)[0]
    best_n = revolutions[chosen].astype(float)
    best = np.where(np.isfinite(np.min(available, axis=0)), best, np.nan)
    missing = ~np.isfinite(best)
    best[missing] = np.nan
    best_n[missing] = np.nan
    return departures, tofs, revolutions, stack, best, best_n


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-dir", type=Path, required=True,
                        help="Directory holding grid_attempts.csv")
    parser.add_argument("--column", default="pythagorean_delta_v_km_s",
                        help="CSV column to map")
    parser.add_argument("--reference-column", default="bspline_delta_v_km_s",
                        help="Second column drawn on the same colour scale")
    parser.add_argument("--label", default=r"$\Delta V$ [km/s]")
    parser.add_argument("--suffix", default="",
                        help="Appended to the figure names, to keep several "
                             "quantities side by side in one sweep directory")
    parser.add_argument("--select-by", default=None,
                        help="Column that picks the revolution count; defaults to "
                             "the plotted column")
    parser.add_argument("--reference-select-by", default=None,
                        help="Same, for the reference panel")
    parser.add_argument("--color-scale",
                        choices=("continuous", *DELTA_V_BINS), default="continuous",
                        help="Delta-V colour buckets; the named scales reproduce "
                             "the B-spline shaping article")
    parser.add_argument("--delta-v-bins", default=None,
                        help="Comma separated bucket edges, overriding --color-scale")
    parser.add_argument("--target-label", default="Mars",
                        help="Target name used in the figure title")
    parser.add_argument("--objective-label",
                        default=r"Minimum-$\Delta V$ shaping")
    parser.add_argument("--figure-format", choices=("pdf", "png"), default="pdf")
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    csv_path = args.sweep_dir / "grid_attempts.csv"
    if not csv_path.exists():
        parser.error(f"No grid_attempts.csv in {args.sweep_dir}")
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error("grid_attempts.csv is empty")

    departures, tofs, revolutions, stack, best, best_n = build_maps(
        rows, args.column, args.select_by)
    _, _, _, _, reference, _ = build_maps(
        rows, args.reference_column, args.reference_select_by or args.select_by)

    import matplotlib
    if not args.show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

    finite = np.concatenate([best[np.isfinite(best)], reference[np.isfinite(reference)]])
    if not len(finite):
        parser.error("No accepted solutions in the sweep")
    extend = "neither"
    if args.delta_v_bins:
        edges = [float(v) for v in args.delta_v_bins.split(",")]
        extend = "max"
    elif args.color_scale != "continuous":
        edges, extend = DELTA_V_BINS[args.color_scale]
    else:
        edges = None
    if edges is not None:
        if len(edges) < 2:
            parser.error("the colour scale needs at least two bucket edges")
        count = len(edges) - 1 + (1 if extend == "max" else 0)
        colors = ListedColormap(GONDELACH_BLUE, name="gondelach_blue").resampled(count)
        norm = BoundaryNorm(edges, colors.N, clip=False, extend=extend)
    else:
        norm = Normalize(float(np.min(finite)), float(np.max(finite)))
        colors = plt.get_cmap("viridis").copy()
    colors.set_bad("#dddddd")
    extent_label = ("Departure [MJD2000 UTC days]\n"
                    f"{mjd2000_label(departures[0])} to {mjd2000_label(departures[-1])}")

    def draw(ax, values, title):
        mesh = ax.pcolormesh(departures, tofs, np.ma.masked_invalid(values),
                             shading="nearest", cmap=colors, norm=norm)
        ax.set(title=title, xlabel=extent_label, ylabel="Time of flight [days]")
        return mesh

    figure, axes = plt.subplots(1, 4, figsize=(21.5, 5.0), constrained_layout=True)
    mesh = draw(axes[0], best, f"Pythagorean Sundman\nover N = "
                              f"{revolutions[0]},...,{revolutions[-1]}")
    draw(axes[1], reference, "Cylindrical B-spline\nover N")
    figure.colorbar(mesh, ax=list(axes[:2]), label=args.label,
                    extend=extend, ticks=edges if edges else None)
    n_colors = ListedColormap(plt.get_cmap("tab10")(np.arange(len(revolutions))))
    n_colors.set_bad("#dddddd")
    mesh_n = axes[2].pcolormesh(
        departures, tofs, np.ma.masked_invalid(best_n), shading="nearest",
        cmap=n_colors,
        norm=BoundaryNorm(np.arange(revolutions[0] - 0.5, revolutions[-1] + 1.5),
                          len(revolutions)))
    axes[2].set(title="Winning revolution count", xlabel=extent_label,
                ylabel="Time of flight [days]")
    figure.colorbar(mesh_n, ax=axes[2], ticks=revolutions, label="N")

    # Relative agreement between the two families on the cells both solved.
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = 100.0 * (best - reference) / reference
    relative[~(np.isfinite(best) & np.isfinite(reference))] = np.nan
    span = float(np.nanmax(np.abs(relative))) if np.any(np.isfinite(relative)) else 1.0
    diverging = plt.get_cmap("RdBu_r").copy()
    diverging.set_bad("#dddddd")
    mesh_d = axes[3].pcolormesh(departures, tofs, np.ma.masked_invalid(relative),
                                shading="nearest", cmap=diverging,
                                norm=Normalize(-span, span))
    axes[3].set(title="Pythagorean vs B-spline\nblue: Pythagorean lower",
                xlabel=extent_label, ylabel="Time of flight [days]")
    figure.colorbar(mesh_d, ax=axes[3], label="Relative difference [%]")
    figure.suptitle(f"{args.objective_label}, Earth to {args.target_label} "
                    "— gray: no accepted solution; values are local optima")
    best_path = args.sweep_dir / f"porkchop_best{args.suffix}.{args.figure_format}"
    figure.savefig(best_path, dpi=180, bbox_inches="tight")

    columns = min(3, len(revolutions))
    rows_count = int(np.ceil(len(revolutions) / columns))
    figure_n, axes_n = plt.subplots(rows_count, columns,
                                    figsize=(5.5 * columns, 4.2 * rows_count),
                                    constrained_layout=True, squeeze=False)
    for index, ax in enumerate(axes_n.flat):
        if index < len(revolutions):
            mesh = draw(ax, stack[index], f"N = {revolutions[index]}")
        else:
            ax.axis("off")
    figure_n.colorbar(mesh, ax=list(axes_n.flat), label=args.label)
    by_n_path = args.sweep_dir / f"porkchop_by_N{args.suffix}.{args.figure_format}"
    figure_n.savefig(by_n_path, dpi=180, bbox_inches="tight")
    if args.show_plots:
        plt.show()

    accepted = int(np.sum(np.isfinite(best)))
    print(f"Malla: {len(departures)} salidas x {len(tofs)} TOF x "
          f"{len(revolutions)} N")
    print(f"Celdas con solucion aceptada: {accepted} de {best.size}")
    if accepted:
        print(f"Mejor {args.column}: {np.nanmin(best):.6f}")
        both = np.isfinite(best) & np.isfinite(reference)
        if np.any(both):
            wins = int(np.sum(best[both] < reference[both]))
            difference = 100.0 * (best[both] - reference[both]) / reference[both]
            print(f"Pythagorean mejor que B-spline en {wins} de {int(both.sum())} "
                  "celdas comparables")
            print(f"Diferencia relativa: mediana {np.median(difference):+.2f}%, "
                  f"rango [{np.min(difference):+.2f}%, {np.max(difference):+.2f}%]")
    print(f"Figuras: {best_path.resolve()}")
    print(f"          {by_n_path.resolve()}")


if __name__ == "__main__":
    main()

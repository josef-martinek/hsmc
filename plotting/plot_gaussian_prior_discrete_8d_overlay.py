#!/usr/bin/env python3
"""Create a two-panel 8D error comparison for the Gaussian-prior experiment.

Each panel overlays dashed Standard SMC with solid Hierarch. SMC.
Generalized IS uses the former Replace results; Classical IS uses the former
Density results. Solver accuracy is scaled by 80 to match
the paper convention; errors and confidence intervals are unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = SCRIPT_DIR.parent
DEFAULT_RESULTS = REPOSITORY_DIR / "results" / "run_20260731-161740_paper" / "smc_results.npz"

Z_VALUE_95_PERCENT = 1.96
SOLVER_ACCURACY_SCALE = 80
# Physical axes dimensions measured from the original three-panel figure.
PANEL_WIDTH_INCHES = 6.05
PANEL_HEIGHT_INCHES = 5.30
LINE_WIDTH = 4.2
ERROR_LINE_WIDTH = 1.6
ERROR_ALPHA = 0.45
PARTICLE_COLORS = {
    100: "#ff7f0e",
    200: "#2ca02c",
    400: "#d62728",
    800: "#9467bd",
}
PARTICLE_MARKERS = {
    100: "o",
    200: "s",
    400: "^",
    800: "D",
}
METHODS = {
    "w1_compact_adapt": "Hierarch. SMC,\n(Generalized IS)",
    "w1_compact_std": "Standard SMC",
    "w1_density_adapt": "Hierarch. SMC,\n(Classical IS)",
}
PANEL_METHODS = (
    "w1_compact_adapt",
    "w1_density_adapt",
)


def asymmetric_yerr(mean: np.ndarray, confidence_interval: np.ndarray) -> list[np.ndarray]:
    """Clip lower confidence limits for the logarithmic y-axis."""
    lower_bound = np.clip(mean - confidence_interval, 1e-12, None)
    return [mean - lower_bound, confidence_interval]


def mean_and_ci(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and 95% normal-approximation confidence interval."""
    mean = np.mean(samples, axis=1)
    ci = Z_VALUE_95_PERCENT * np.std(samples, axis=1, ddof=1) / np.sqrt(samples.shape[1])
    return mean, ci


def configure_error_axis(axis: plt.Axes, displayed_nmax: np.ndarray, tick_font_size: int) -> None:
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(displayed_nmax)
    # A common factor of 160 gives exact doubling ticks in both dimensions.
    formatter = plt.FuncFormatter(lambda value, position: f"{value / 160:g}")
    formatter.set_offset_string(r"$\times 160$")
    axis.xaxis.set_major_formatter(formatter)
    axis.xaxis.get_offset_text().set_fontsize(tick_font_size)
    axis.xaxis.set_minor_formatter(plt.NullFormatter())
    axis.tick_params(axis="both", which="major", labelsize=tick_font_size)
    axis.grid(True, which="both", linestyle="--", alpha=0.45)


def draw_error_methods(
    axis: plt.Axes,
    results: np.lib.npyio.NpzFile,
    method_keys: tuple[str, ...],
    displayed_nmax: np.ndarray,
    display_mask: np.ndarray,
    show_error_bars: bool,
) -> None:
    """Draw selected algorithms for every displayed particle count."""
    for particle_index, particle_count in enumerate(results["n_particles_seq"]):
        particle_count = int(particle_count)
        if particle_count == 50:
            continue
        color = PARTICLE_COLORS[particle_count]
        for result_key in method_keys:
            marker = PARTICLE_MARKERS[particle_count]
            mean, ci = mean_and_ci(results[result_key][0, 0, particle_index])
            values = mean[display_mask]
            if show_error_bars:
                errors = np.asarray(asymmetric_yerr(mean, ci))[:, display_mask]
                axis.errorbar(
                    displayed_nmax,
                    values,
                    yerr=errors,
                    fmt="none",
                    ecolor=color,
                    alpha=ERROR_ALPHA,
                    capsize=3,
                    elinewidth=ERROR_LINE_WIDTH,
                )
            axis.plot(
                displayed_nmax,
                values,
                marker=marker,
                linestyle="--" if result_key == "w1_compact_std" else "-",
                color=color,
                label=f"{METHODS[result_key]}, N = {particle_count}",
                linewidth=LINE_WIDTH,
                markersize=9,
            )


def particle_legend_handles(results: np.lib.npyio.NpzFile) -> list[Line2D]:
    """Create the common particle-count legend for both panels."""
    return [
        Line2D(
            [0],
            [0],
            color=PARTICLE_COLORS[int(count)],
            marker=PARTICLE_MARKERS[int(count)],
            linewidth=LINE_WIDTH,
            markersize=9,
            label=f"$N = {count}$",
        )
        for count in results["n_particles_seq"]
        if count != 50
    ]


def plot_by_algorithm(
    results: np.lib.npyio.NpzFile,
    output_path: Path,
    show_error_bars: bool = False,
) -> None:
    """Write two common-scale comparisons with the Standard SMC baseline."""
    nmax = results["Nmax_seq"]
    display_mask = np.ones(nmax.shape, dtype=bool)
    displayed_nmax = SOLVER_ACCURACY_SCALE * nmax[display_mask]

    # Reserve space for the shared particle-count legend without shrinking the original panels.
    left, right, bottom, top, gap = 1.30, 0.20, 0.95, 0.65, 0.35
    figure_width = left + 2 * PANEL_WIDTH_INCHES + gap + right
    figure_height = bottom + PANEL_HEIGHT_INCHES + top
    fig, axes = plt.subplots(
        1, 2, figsize=(figure_width, figure_height), sharex=True, sharey=True,
    )
    fig.subplots_adjust(
        left=left / figure_width,
        right=1 - right / figure_width,
        bottom=bottom / figure_height,
        top=1 - top / figure_height,
        wspace=gap / PANEL_WIDTH_INCHES,
    )
    for axis, method_key in zip(axes, PANEL_METHODS, strict=True):
        draw_error_methods(
            axis, results, ("w1_compact_std", method_key),
            displayed_nmax, display_mask, show_error_bars,
        )
        axis.legend(
            handles=[
                Line2D([0], [0], color="black", linestyle="--", linewidth=LINE_WIDTH,
                       label=METHODS["w1_compact_std"]),
                Line2D([0], [0], color="black", linestyle="-", linewidth=LINE_WIDTH,
                       label=METHODS[method_key]),
            ],
            loc="upper right",
            fontsize=16,
            frameon=True,
            facecolor="white",
            framealpha=1.0,
        )

    for axis in axes:
        configure_error_axis(axis, displayed_nmax, tick_font_size=19)
        axis.set_xlabel("$S_T$", fontsize=25)
    axes[0].set_ylabel("Mean W1 Error", fontsize=25)

    fig.legend(
        handles=particle_legend_handles(results),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        fontsize=17,
        frameon=False,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Path to smc_results.npz.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output", help="Directory for generated PDFs.")
    parser.add_argument(
        "--show-error-bars",
        action="store_true",
        help="Display 95%% confidence-interval error bars (hidden by default).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.results, allow_pickle=False) as results:
        output_path = args.output_dir / "error_as_8d_by_algorithm_overlay.pdf"
        plot_by_algorithm(results, output_path, show_error_bars=args.show_error_bars)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

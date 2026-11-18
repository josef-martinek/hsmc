#!/usr/bin/env python3
"""Generate the 8D and 64D cost plots in the paper's solver-accuracy units.

Both archived solver accuracies and costs per particle are multiplied by 80.
The x-axis displays exact doubling ticks with a common factor of 160, matching
the updated error plots. Existing scripts and PDFs are preserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "results"
RUN_DIRECTORIES = {
    8: "run_20260731-161740_paper",
    64: "run_20260731-153234_weird",
}
SOLVER_ACCURACY_SCALE = 80
X_TICK_FACTOR = 160


def create_cost_figure(results: np.lib.npyio.NpzFile, dimension: int) -> plt.Figure:
    """Preserve the original cost curves and styling, converting both axes."""
    nmax = results["Nmax_seq"]
    mask = nmax >= 8 if dimension == 64 else np.ones(nmax.shape, dtype=bool)
    accuracy = SOLVER_ACCURACY_SCALE * nmax[mask]
    fig, axis = plt.subplots(figsize=(20 / 3, 6.8))
    for key, label, marker, linestyle in (
        ("cost_adapt", "Hierarch. SMC", "o", "-"),
        ("cost_fixed", "Standard SMC", "s", "--"),
    ):
        axis.plot(
            accuracy,
            SOLVER_ACCURACY_SCALE * results[key][0, 0][mask],
            marker=marker, linestyle=linestyle, color="black",
            linewidth=4.2, markersize=9, label=label,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(accuracy)
    formatter = FuncFormatter(lambda value, position: f"{value / X_TICK_FACTOR:g}")
    formatter.set_offset_string(r"$\times160$")
    axis.xaxis.set_major_formatter(formatter)
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.xaxis.get_offset_text().set_fontsize(19)
    axis.tick_params(axis="both", which="major", labelsize=19)
    axis.set_xlabel("$S_T$", fontsize=25)
    axis.set_ylabel("Cost per particle", fontsize=25)
    axis.grid(True, which="both", linestyle="--", alpha=0.45)
    axis.legend(fontsize=17, handlelength=3.5)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory containing the archived run folders.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output",
                        help="Directory for the new scaled cost PDFs.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dimension, directory in RUN_DIRECTORIES.items():
        with np.load(args.data_dir / directory / "smc_results.npz", allow_pickle=False) as results:
            fig = create_cost_figure(results, dimension)
        output = args.output_dir / f"cost_as_{dimension}d_scaled.pdf"
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()

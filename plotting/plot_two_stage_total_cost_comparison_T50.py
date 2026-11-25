#!/usr/bin/env python3
"""Reproduce the two-panel T=50 paper figure from the October 22 archives.

Run with the hsmc environment, for example from the repository root:

    MPLBACKEND=Agg python -B plotting/plot_two_stage_total_cost_comparison_T50.py

Only two_stage_total_cost_comparison_T50.pdf is written, to plotting/output/
unless --output-dir is supplied. The original run-folder plots are preserved.
Cost accounting, confidence intervals, and formatting reuse the T=24/T=37
paper plotting code; the x-axis multiplier is fixed at 10^8.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import yaml

import plot_two_stage_total_cost_comparison as reference


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_SPECS = (
    reference.RunSpec("run_20261022-161115_", 50, 100),
    reference.RunSpec("run_20261022-161132_", 50, 1000),
)


def load_run(data_dir: Path, spec: reference.RunSpec) -> reference.RunData:
    """Validate the T=50 archive."""
    directory = data_dir / spec.directory
    config = yaml.safe_load((directory / "setup.yaml").read_text())
    experiment = config["experiment"]
    sweep = experiment["sweep"]
    with np.load(directory / "smc_results.npz", allow_pickle=False) as archive:
        for key in ("budget_seq", "n_particles_seq", "gamma_nsamp_seq"):
            if not np.array_equal(archive[key], sweep[key]):
                raise ValueError(f"NPZ/config mismatch for {key} in {directory}")
        for key in ("Nmax", "len_c"):
            if archive[key].item() != experiment[key]:
                raise ValueError(f"NPZ/config mismatch for {key} in {directory}")
        if (archive["n_temp_steps"].item() != spec.stages
                or archive["len_c"].item() != spec.coarse_cap
                or not np.array_equal(archive["len_chain_seq"], [0])
                or archive["gamma_nsamp_seq"].shape != (1,)):
            raise ValueError(f"Unexpected experiment layout in {directory}")
        budgets = archive["budget_seq"].copy()
        particles = archive["n_particles_seq"].copy()
        gamma = float(archive["gamma_nsamp_seq"][0])
        errors = {}
        expected_shape = (1, 1, len(particles), len(budgets), sweep["n_runs"])
        for method in reference.METHODS:
            values = archive[f"w1_{method.key}"]
            if (values.shape != expected_shape or not np.isfinite(values).all()
                    or np.any(values <= 0)):
                raise ValueError(f"Invalid error samples for {method.key} in {directory}")
            errors[method.key] = values[0, 0].copy()
    ratio = float(config["problem"]["fine_over_coarse_nsamp"])
    if (ratio != 20 or experiment["Nmax"] != 16 or sweep["n_runs"] != 400
            or config["problem"]["param_dim"] != 64
            or not np.array_equal(particles, reference.DISPLAY_PARTICLES)
            or not np.array_equal(budgets, [80, 160, 320, 640, 1280])):
        raise ValueError(f"Unsupported archived cost-model settings in {directory}")
    raw_schedule = reference.reconstruct_schedule(
        experiment["Nmax"], experiment["nsamp_min"], gamma, spec.stages,
        experiment["scheduler_params"]["shape_p"],
    )
    # Keep the particle colors used in the T=24/T=37 paper PDFs.
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {n: cycle[i] for i, n in enumerate((50, 100, 200, 400, 800))}
    return reference.RunData(
        spec, budgets, particles, raw_schedule * reference.SOLVER_ACCURACY_SCALE,
        ratio, errors, colors,
    )


def create_figure(runs: list[reference.RunData], rows: list[dict]) -> plt.Figure:
    """Match the reference two-panel layout with a fixed 10^8 cost multiplier."""
    fig = reference.create_figure(runs, rows, 50, reference.common_limits(rows))
    for axis in fig.axes:
        formatter = FuncFormatter(lambda value, position: f"{value / 1e8:g}")
        formatter.set_offset_string(r"$\times10^8$")
        axis.xaxis.set_major_formatter(formatter)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR.parent / "results",
                        help="Directory containing the two October 22 run folders.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output",
                        help="Destination for the T=50 two-panel PDF.")
    args = parser.parse_args()
    runs = [load_run(args.data_dir, spec) for spec in RUN_SPECS]
    rows = reference.build_rows(runs)
    fig = create_figure(runs, rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "two_stage_total_cost_comparison_T50.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")
    print(f"Plotted {len(rows)} archived points.")


if __name__ == "__main__":
    main()

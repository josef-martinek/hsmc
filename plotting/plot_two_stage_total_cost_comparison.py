#!/usr/bin/env python3
"""Plot archived two-stage errors against reconstructed total solver cost.

Run with the hsmc environment. Only the new total-cost PDFs and CSV are written.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "results"
SOLVER_ACCURACY_SCALE = 80.0
DISPLAY_PARTICLES = (100, 800)
LINE_WIDTH = 3.5
MARKER_SIZE = 8
# Original four-panel figure: width = 28 * .91 / 4.3, height = 7 * .70.
PANEL_WIDTH_INCHES = 28 * 0.91 / 4.3
PANEL_HEIGHT_INCHES = 4.9


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    marker: str
    linestyle: str


METHODS = (
    Method("compact_adapt", "Two-stage Hierarch. SMC", "o", "-"),
    Method("compact_std", "Hierarch. SMC", "s", ":"),
    # Legacy storage key: this is fixed-accuracy standard PMMH, not Classical IS.
    Method("density_adapt", "Standard SMC", "^", "--"),
)


@dataclass(frozen=True)
class RunSpec:
    directory: str
    stages: int
    coarse_cap: int


RUN_SPECS = (
    RunSpec("run_20260714-144004_2s_paper", 24, 100),
    RunSpec("run_20260714-143914_2s_paper", 24, 1000),
    RunSpec("run_20260714-151644_2s_paper", 37, 100),
    RunSpec("run_20260714-151715_2s_paper", 37, 1000),
)


@dataclass
class RunData:
    spec: RunSpec
    budgets: np.ndarray
    particles: np.ndarray
    schedule: np.ndarray
    fine_over_coarse: float
    errors: dict[str, np.ndarray]
    colors: dict[int, str]


@dataclass(frozen=True)
class Cost:
    """Transition counts per stage, and cost components for all N particles."""

    coarse_transitions: int
    fine_transitions: int
    initialization_cost: float
    accuracy_refresh_cost: float
    coarse_mutation_cost: float
    fine_mutation_cost: float
    coarse_to_fine_refresh_cost: float

    @property
    def total(self) -> float:
        return sum((self.initialization_cost, self.accuracy_refresh_cost,
                    self.coarse_mutation_cost, self.fine_mutation_cost,
                    self.coarse_to_fine_refresh_cost))


def reconstruct_schedule(nmax: float, nsamp_min: float, gamma: float,
                         stages: int, shape_p: float) -> np.ndarray:
    """Reproduce GaussianPriorDiscreteExperiment's schedule in archive units."""
    times = np.arange(stages + 1) / stages
    smooth = nsamp_min * (nmax / nsamp_min) ** (times ** shape_p)
    powers = np.round(np.log(smooth / nsamp_min) / np.log(gamma))
    return np.minimum(nsamp_min * gamma ** powers, nmax)


def load_run(data_dir: Path, spec: RunSpec) -> RunData:
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
                or len(archive["gamma_nsamp_seq"]) != 1):
            raise ValueError(f"Unexpected experiment layout in {directory}")
        budgets = archive["budget_seq"].copy()
        particles = archive["n_particles_seq"].copy()
        gamma = float(archive["gamma_nsamp_seq"][0])
        errors = {method.key: archive[f"w1_{method.key}"][0, 0].copy()
                  for method in METHODS}
        expected_shape = (1, 1, len(particles), len(budgets), sweep["n_runs"])
        for method in METHODS:
            if (archive[f"w1_{method.key}"].shape != expected_shape
                    or not np.isfinite(errors[method.key]).all()
                    or np.any(errors[method.key] <= 0)):
                raise ValueError(f"Invalid error samples for {method.key} in {directory}")
    ratio = float(config["problem"]["fine_over_coarse_nsamp"])
    if (ratio != 20 or experiment["Nmax"] != 2 or sweep["n_runs"] != 400
            or not np.array_equal(particles, [50, 100, 200, 400, 800])
            or np.any(budgets <= 0) or np.any(budgets != np.floor(budgets))
            or np.any(np.diff(budgets) <= 0)):
        raise ValueError(f"Unsupported archived cost-model settings in {directory}")
    raw_schedule = reconstruct_schedule(
        experiment["Nmax"], experiment["nsamp_min"], gamma, spec.stages,
        experiment["scheduler_params"]["shape_p"],
    )
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {int(n): color_cycle[i % len(color_cycle)] for i, n in enumerate(particles)}
    return RunData(spec, budgets, particles, raw_schedule * SOLVER_ACCURACY_SCALE,
                   ratio, errors, colors)


def calculate_cost(method: str, n: int, budget: float, schedule: np.ndarray,
                   coarse_cap: int, fine_over_coarse: float = 20.0) -> Cost:
    """Count solver evaluations; schedule includes S_0 and all T target stages.

    The final post-run move is stage T, not an additional T+1 mutation.
    A coarse chain requires one fine evaluation even with zero fine transitions.
    """
    if method == "density_adapt":
        fine = int(budget)
        return Cost(0, fine, n * schedule[-1], 0, 0,
                    n * fine * (len(schedule) - 1) * schedule[-1], 0)
    if method not in ("compact_adapt", "compact_std"):
        raise ValueError(f"Unknown archived method: {method}")
    initialization = n * schedule[0]
    refresh = n * schedule[1:][np.diff(schedule) > 0].sum()
    stage_cost = n * schedule[1:].sum()
    coarse_unit_cost = 2.0 / fine_over_coarse
    if method == "compact_adapt":
        coarse = min(coarse_cap, int(np.floor(budget / coarse_unit_cost)))
        fine = int(np.floor(budget - coarse * coarse_unit_cost))
    else:
        coarse, fine = 0, int(budget)
    return Cost(coarse, fine, initialization, refresh,
                stage_cost * coarse * coarse_unit_cost,
                stage_cost * fine, stage_cost if coarse > 0 else 0.0)


def mean_and_ci(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(samples, axis=-1)
    ci = 1.96 * np.std(samples, axis=-1, ddof=1) / np.sqrt(samples.shape[-1])
    return mean, ci


def build_rows(runs: list[RunData]) -> list[dict]:
    """Audit all N >= 100 points; no pooling across run archives."""
    rows = []
    for run in runs:
        for particle_index, n in enumerate(run.particles):
            if n == 50:
                continue
            for method in METHODS:
                mean, ci = mean_and_ci(run.errors[method.key][particle_index])
                for i, budget in enumerate(run.budgets):
                    cost = calculate_cost(method.key, int(n), float(budget), run.schedule,
                                          run.spec.coarse_cap, run.fine_over_coarse)
                    rows.append(dict(
                        run=run.spec.directory, stages=run.spec.stages,
                        coarse_cap=run.spec.coarse_cap, method=method.key,
                        method_label=method.label.replace("\n", " "),
                        n_particles=int(n), budget=float(budget),
                        repetitions=run.errors[method.key].shape[-1],
                        **asdict(cost), total_cost=cost.total,
                        mean_error=float(mean[i]), ci95_half_width=float(ci[i]),
                    ))
    return rows


def common_limits(rows: list[dict]) -> tuple[tuple[float, float], tuple[float, float]]:
    costs = np.array([row["total_cost"] for row in rows])
    lower = min(row["mean_error"] - row["ci95_half_width"] for row in rows)
    upper = max(row["mean_error"] + row["ci95_half_width"] for row in rows)
    padding = (upper - lower) * 0.05
    return (costs.min() / 1.15, costs.max() * 1.15), (lower - padding, upper + padding)


def create_figure(runs: list[RunData], rows: list[dict], stages: int,
                  limits: tuple) -> plt.Figure:
    """Use identical physical axes sizes and limits in the two exported PDFs."""
    left, right, bottom, top, gap = 1.1, 0.2, 1.25, 1.0, 0.3
    width = left + 2 * PANEL_WIDTH_INCHES + gap + right
    height = bottom + PANEL_HEIGHT_INCHES + top
    fig, axes = plt.subplots(1, 2, figsize=(width, height), sharey=True)
    fig.subplots_adjust(left=left / width, right=1 - right / width,
                        bottom=bottom / height, top=1 - top / height,
                        wspace=gap / PANEL_WIDTH_INCHES)
    selected_runs = sorted((r for r in runs if r.spec.stages == stages),
                           key=lambda r: r.spec.coarse_cap)
    for axis, run in zip(axes, selected_runs, strict=True):
        for n in DISPLAY_PARTICLES:
            for method in METHODS:
                points = [row for row in rows if row["run"] == run.spec.directory
                          and row["n_particles"] == n and row["method"] == method.key]
                x = [point["total_cost"] for point in points]
                y = [point["mean_error"] for point in points]
                axis.errorbar(x, y, yerr=[point["ci95_half_width"] for point in points],
                              fmt="none", color=run.colors[n], capsize=5,
                              elinewidth=LINE_WIDTH)
                axis.plot(x, y, color=run.colors[n], marker=method.marker,
                          linestyle=method.linestyle, linewidth=LINE_WIDTH,
                          markersize=MARKER_SIZE, label=f"{method.key}: N={n}")
        axis.set_title(rf"$T={stages}$, $M_c^{{\mathrm{{max}}}}={run.spec.coarse_cap}$",
                       fontsize=22, pad=10)
        axis.set_xscale("log")
        axis.set_xlim(limits[0])
        axis.set_ylim(limits[1])
        axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=20))
        formatter = FuncFormatter(lambda value, position: f"{value / 1e6:g}")
        formatter.set_offset_string(r"$\times10^6$")
        axis.xaxis.set_major_formatter(formatter)
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.xaxis.get_offset_text().set_fontsize(18)
        axis.tick_params(axis="both", which="major", labelsize=18)
        axis.grid(True, which="both", linestyle="--", alpha=0.45)
    axes[1].legend(handles=[
        Line2D([0], [0], color="black", marker=m.marker, linestyle=m.linestyle,
               linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=m.label)
        for m in reversed(METHODS)
    ], loc="upper right", fontsize=14, framealpha=0.8)
    axes[0].set_ylabel("Mean W1 Error", fontsize=25)
    fig.supxlabel("Total cost", fontsize=22, x=(left + width - right) / (2 * width), y=0.02)
    fig.legend(handles=[
        Line2D([0], [0], color=selected_runs[0].colors[n], linewidth=LINE_WIDTH,
               label=rf"$N={n}$") for n in DISPLAY_PARTICLES
    ], loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(DISPLAY_PARTICLES),
       fontsize=18, frameon=False)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory containing the four archived run folders.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output",
                        help="Directory for the two PDFs and companion CSV.")
    args = parser.parse_args()
    runs = [load_run(args.data_dir, spec) for spec in RUN_SPECS]
    rows = build_rows(runs)
    limits = common_limits(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stages in (24, 37):
        fig = create_figure(runs, rows, stages, limits)
        output = args.output_dir / f"two_stage_total_cost_comparison_T{stages}.pdf"
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {output}")
    csv_path = args.output_dir / "two_stage_total_cost_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} archived points)")


if __name__ == "__main__":
    main()

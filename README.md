# hsmc

Code and data for Section 6.1 ("A linear problem") of the paper

> Josef Martínek, Julian Hofstadler, Alexander M. G. Cox and Robert Scheichl.
> *Reducing Complexity of Pseudo-Marginal Sequential Monte Carlo via
> Hierarchical Tempering.*

Paper link / DOI: **[PLACEHOLDER: to be added]**

The repository also provides the Python package `hsmc` (hierarchical SMC
samplers and MCMC kernels), which is used by the companion repository
[`hsmc_pai`](https://github.com/josef-martinek/hsmc_pai) for the experiments
of Section 6.2.

This repository is a frozen snapshot that accompanies the paper. It is not
developed further.

## Installation

The environment uses Python 3.12 and the pinned package versions listed in
`requirements.yml`. The SMC library `particles` is installed from a pinned
commit of its GitHub repository
(`nchopin/particles@8f2c4bfa85cc7a287c0fd36b13710bfe4bd41856`), so network
access is needed.

```bash
conda env create -f requirements.yml
conda activate hsmc
pip install -e .
```

`pip install -e .` installs only the package `hsmc`. The other folders are
used from the repository root (see below).

## Layout

- `hsmc/`: the installable package, built on `particles`: `samplers`
  (tempering SMC samplers in which the solver accuracy changes along the
  tempering sequence), `mcmc` (pseudo-marginal random-walk (PMMH) and noisy
  Metropolis (MCWM) kernels with fixed or calibrated proposal covariance, and
  a two-stage kernel), `likelihood_base` (base classes for likelihood models
  with a stochastic forward solver).
- `problem_setup/`: likelihood models. `compact_problem/` is the linear
  problem of Section 6.1 (forward map `A u` plus solver noise drawn from a
  Gaussian truncated to a ball); `compact_problem_density/` adds the variants
  that keep the solver noise and evaluate the ratio of noise densities when
  the accuracy changes. `toy_problem/` is only a fixture for
  `tests/test_mcmc.py`.
- `experiments/`: the experiment runners: `gaussian_prior_discrete.py`
  (standard vs. hierarchical SMC, first experiment of Section 6.1),
  `fixed_coarse_budget_sweep.py` (two-stage mutation, second experiment), and
  the two base classes they inherit from.
- `manager/`: `ExperimentManager`, which runs the tasks of an experiment in
  parallel with Ray and writes the results folder.
- `utils/`: exact samples from the Gaussian posterior (reference solution),
  the reference SMC run that fixes the tempering schedule, and helpers that
  create the results folder.
- `scripts/`: `run_experiments.py`, the Hydra entry point for all experiments.
- `conf/`: Hydra configuration; `conf/experiment/paper/` contains the eight
  configurations of the paper runs.
- `plotting/`: the scripts that produce the paper figures from `results/`.
- `results/`: the archived output of the paper runs (`setup.yaml`,
  `smc_results.npz` and, for the two-stage runs, `stats_summary.txt`).
- `tests/`: unit tests for the MCMC kernels and the linear problem.

## Reproducing the figures from the included data

Run from the repository root, in this order (the last script checks its
panels against PDFs written by the two scripts before it):

```bash
MPLBACKEND=Agg python -B plotting/plot_gaussian_prior_discrete_8d_overlay.py
MPLBACKEND=Agg python -B plotting/plot_gaussian_prior_discrete_64d_overlay.py
MPLBACKEND=Agg python -B plotting/plot_gaussian_prior_discrete_cost_scaled.py
MPLBACKEND=Agg python -B plotting/plot_two_stage_total_cost_comparison.py
MPLBACKEND=Agg python -B plotting/plot_two_stage_total_cost_comparison_T50.py
MPLBACKEND=Agg python -B plotting/plot_two_stage_total_cost_comparison_panels.py
```

All output goes to `plotting/output/` (option `--output-dir`). The scripts
check the archived data while plotting and stop with an error if a check
fails.

| Figure file in the paper | Script | Results folders |
| --- | --- | --- |
| `error_as_8d_by_algorithm_overlay.pdf` | `plot_gaussian_prior_discrete_8d_overlay.py` | `run_20260731-161740_paper` |
| `error_as_64d_by_algorithm_overlay.pdf` | `plot_gaussian_prior_discrete_64d_overlay.py` | `run_20260731-153234_paper` |
| `cost_as_8d_scaled.pdf`, `cost_as_64d_scaled.pdf` | `plot_gaussian_prior_discrete_cost_scaled.py` | `run_20260731-161740_paper`, `run_20260731-153234_paper` |
| `two_stage_total_cost_comparison_T24_cap100.pdf`, `..._T24_cap1000.pdf` | `plot_two_stage_total_cost_comparison_panels.py` | `run_20260714-144004_2s_paper` (cap 100), `run_20260714-143914_2s_paper` (cap 1000) |
| `two_stage_total_cost_comparison_T37_cap100.pdf`, `..._T37_cap1000.pdf` | `plot_two_stage_total_cost_comparison_panels.py` | `run_20260714-151644_2s_paper` (cap 100), `run_20260714-151715_2s_paper` (cap 1000) |
| `two_stage_total_cost_comparison_T50_cap100.pdf`, `..._T50_cap1000.pdf` | `plot_two_stage_total_cost_comparison_panels.py` | `run_20261022-161115_` (cap 100), `run_20261022-161132_` (cap 1000) |
| `two_stage_total_cost_comparison_header.pdf`, `..._xlabel.pdf`, `..._ylabel.pdf` (legend and axis titles) | `plot_two_stage_total_cost_comparison_panels.py` | (all six two-stage folders) |

The first three rows make up the figure with the mean marginal W1 error and
the cost for d = 8 and d = 64; the other rows make up the two-stage mutation
figure. `plot_two_stage_total_cost_comparison.py` (T = 24, 37) and
`plot_two_stage_total_cost_comparison_T50.py` (T = 50) write the intermediate
files `two_stage_total_cost_comparison_T{24,37,50}.pdf` and
`two_stage_total_cost_comparison.csv`, which are not in the paper; the panels
script checks every plotted point against these PDFs.

Solver accuracies in the configurations and result files are in internal
units; the plotting scripts multiply them by 80 to obtain the accuracies used
in the paper (for example `nsamp_min: 0.125` is the paper's initial accuracy
10, and `Nmax: 2` and `Nmax: 16` are the final accuracies 160 and 1280).

## Rerunning the experiments

Each paper run is rerun from the repository root with

```bash
PYTHONPATH=. python scripts/run_experiments.py experiment=paper/<name>
```

where `<name>` is one of the configurations below. Each configuration is a
verbatim copy of the `setup.yaml` stored in its results folder.

| `<name>` | Results folder | What it is |
| --- | --- | --- |
| `gaussian_prior_8d` | `run_20260731-161740_paper` | d = 8, schedule ESSrmin 0.8, standard vs. hierarchical SMC, 100 runs |
| `gaussian_prior_64d` | `run_20260731-153234_paper` | d = 64, schedule ESSrmin 0.8, standard vs. hierarchical SMC, 200 runs |
| `two_stage_ess005_cap100` | `run_20260714-144004_2s_paper` | d = 64, ESSrmin 0.05 (T = 24), cheap-chain cap 100, 400 runs |
| `two_stage_ess005_cap1000` | `run_20260714-143914_2s_paper` | d = 64, ESSrmin 0.05 (T = 24), cheap-chain cap 1000, 400 runs |
| `two_stage_ess020_cap100` | `run_20260714-151644_2s_paper` | d = 64, ESSrmin 0.2 (T = 37), cheap-chain cap 100, 400 runs |
| `two_stage_ess020_cap1000` | `run_20260714-151715_2s_paper` | d = 64, ESSrmin 0.2 (T = 37), cheap-chain cap 1000, 400 runs |
| `two_stage_ess040_cap100` | `run_20261022-161115_` | d = 64, ESSrmin 0.4 (T = 50), cheap-chain cap 100, 400 runs |
| `two_stage_ess040_cap1000` | `run_20261022-161132_` | d = 64, ESSrmin 0.4 (T = 50), cheap-chain cap 1000, 400 runs |

ESSrmin is the relative ESS threshold of the reference SMC run that fixes the
tempering schedule; the cheap-chain cap is `len_c` in the configuration.

New output goes to a new folder `data/run_<YYYYMMDD-HHMMSS>_/` (option
`results_base_dir=<dir>`). It contains `setup.yaml`, a copy of the experiment
runner file, `smc_results.npz`, `stats_summary.txt` and diagnostic PDFs.
Hydra also writes `run_experiments.log` to the current directory. To plot a
new run with the scripts above, give its folder the name the script expects
(listed in the tables) inside a directory passed with `--data-dir`, or pass
the results file with `--results` (the two overlay scripts).

The experiments use Ray and by default all available CPUs (option
`num_cpus=<n>`). Their cost is substantial: each configuration runs between
3,500 (`gaussian_prior_8d`) and 10,000 independent tasks, and every task runs
each compared algorithm once. The paper runs were executed on a 64-core
cluster node. For a quick check of a configuration, `export_setup_only=true`
only resolves it and writes a results folder with its `setup.yaml`.

## Code versions of the paper runs

The paper runs were produced with the following commits of this repository's
history:

| Results folder | Commit |
| --- | --- |
| `run_20260731-161740_paper` | `ee0bbfd` (recorded in the run's job log, which is not included) |
| `run_20260731-153234_paper` | `ee0bbfd` (inferred from the time of the run) |
| `run_20260714-143914_2s_paper`, `run_20260714-144004_2s_paper`, `run_20260714-151644_2s_paper`, `run_20260714-151715_2s_paper` | `6a7ae02` (inferred from the time of the runs) |
| `run_20261022-161115_`, `run_20261022-161132_` | `c3316e7` (recorded in the runs' job logs, which are not included) |

The sampler code used by these runs is unchanged in behaviour up to this
release: later commits only added two optional arguments to the adaptive
sampler and a fix in a class these runs do not use. The files were afterwards
moved into the `hsmc` package without changes.

## Tests

```bash
python -m pytest
```

## License and citation

MIT License, see `LICENSE`.

If you use this code, please cite the paper above. A citation file for the
software is in `CITATION.cff`.

Software DOI (Zenodo): **[PLACEHOLDER: to be added]**

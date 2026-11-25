# hsmc

Code and data for Section 6.1 ("A linear problem") of the paper

> Josef Martínek, Julian Hofstadler, Alexander M. G. Cox and Robert Scheichl.
> *Reducing Complexity of Pseudo-Marginal Sequential Monte Carlo via
> Hierarchical Tempering.*

Paper: [arXiv:2610.37082](https://arxiv.org/abs/2610.37082)

The repository also provides the Python package `hsmc` (hierarchical SMC
samplers and MCMC kernels), which is used by the companion repository
[`hsmc_pai`](https://github.com/josef-martinek/hsmc_pai) for the experiments
of Section 6.2.

## Installation

```bash
conda env create -f requirements.yml
conda activate hsmc
pip install -e .
```

## Reproducing the figures from the included data

Run from the repository root, in this order (the last script reads PDFs
written by the two scripts before it):

```bash
MPLBACKEND=Agg python -B plotting/plot_gaussian_prior_discrete_8d_overlay.py
MPLBACKEND=Agg python -B plotting/plot_gaussian_prior_discrete_64d_overlay.py
MPLBACKEND=Agg python -B plotting/plot_gaussian_prior_discrete_cost_scaled.py
MPLBACKEND=Agg python -B plotting/plot_two_stage_total_cost_comparison.py
MPLBACKEND=Agg python -B plotting/plot_two_stage_total_cost_comparison_T50.py
MPLBACKEND=Agg python -B plotting/plot_two_stage_total_cost_comparison_panels.py
```

All output goes to `plotting/output/` (option `--output-dir`).

| Figure file in the paper | Script | Results folders |
| --- | --- | --- |
| `error_as_8d_by_algorithm_overlay.pdf` | `plot_gaussian_prior_discrete_8d_overlay.py` | `run_20260731-161740_paper` |
| `error_as_64d_by_algorithm_overlay.pdf` | `plot_gaussian_prior_discrete_64d_overlay.py` | `run_20260731-153234_paper` |
| `cost_as_8d_scaled.pdf`, `cost_as_64d_scaled.pdf` | `plot_gaussian_prior_discrete_cost_scaled.py` | `run_20260731-161740_paper`, `run_20260731-153234_paper` |
| `two_stage_total_cost_comparison_T24_cap100.pdf`, `..._T24_cap1000.pdf` | `plot_two_stage_total_cost_comparison_panels.py` | `run_20260714-144004_2s_paper` (cap 100), `run_20260714-143914_2s_paper` (cap 1000) |
| `two_stage_total_cost_comparison_T37_cap100.pdf`, `..._T37_cap1000.pdf` | `plot_two_stage_total_cost_comparison_panels.py` | `run_20260714-151644_2s_paper` (cap 100), `run_20260714-151715_2s_paper` (cap 1000) |
| `two_stage_total_cost_comparison_T50_cap100.pdf`, `..._T50_cap1000.pdf` | `plot_two_stage_total_cost_comparison_panels.py` | `run_20261022-161115_` (cap 100), `run_20261022-161132_` (cap 1000) |
| `two_stage_total_cost_comparison_header.pdf`, `..._xlabel.pdf`, `..._ylabel.pdf` (legend and axis titles) | `plot_two_stage_total_cost_comparison_panels.py` | (all six two-stage folders) |

## Rerunning the experiments

Each paper run is rerun from the repository root with

```bash
PYTHONPATH=. python scripts/run_experiments.py experiment=paper/<name>
```

where `<name>` is one of the configurations below.

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

Solver accuracies in the configurations and result files are in internal
units; the plotting scripts rescales them by factor 80 for readability in the paper.

New output goes to a new folder `data/run_<YYYYMMDD-HHMMSS>_/` (option
`results_base_dir=<dir>`). To plot a new run with the scripts above, give its
folder the name the script expects (listed in the tables) inside a directory
passed with `--data-dir`, or pass the results file with `--results` (the two
overlay scripts).

The experiments use Ray and by default all available CPUs (option
`num_cpus=<n>`). Each configuration runs between 3,500 (`gaussian_prior_8d`)
and 10,000 independent tasks; the paper runs were executed on a 64-core
cluster node.

## Tests

```bash
python -m pytest
```

## License and citation

MIT License, see `LICENSE`.

If you use this code, please cite the paper above. A citation file for the
software is in `CITATION.cff`.

# Pivo Refactoring Architecture Plan

## 1. Goal
Refactor the codebase to polish the code, remove redundant methods, ensure correctness via automated tests, and introduce a flexible, file-based configuration framework using **Hydra** to run experiments seamlessly without modifying source code.

## 2. Testing Strategy
- Introduce `pytest` as the testing framework.
- **SMC Tests (`tests/test_smc.py`)**: 
  - Fix random seeds and run inference on the toy problem.
  - Implement tests for the relevant SMC samplers: `AdaptiveTempering` and `TempAdaptNsampNoRew`.
  - Save the posterior particles from an initial run as reference solutions and compare subsequent runs against them.
- **MCMC Tests (`tests/test_mcmc.py`)**:
  - Test individual MCMC methods in isolation to ensure deterministic updates without running full SMC inference.

## 3. Dependencies Update
- Update `requirements.yml` to include `pytest` and `hydra-core`.

## 4. Dead Code Elimination & Cleanup
- **`core/samplers.py`**: Keep `TempAdaptNsampNoRew`. Remove `TempAdaptNsampNoIS` and `TempAdaptNsampRew` (consolidating any necessary base logic).
- **`core/mcmc.py`**: Remove the unused `DA` class.
- **Root Directory Cleanup**: 
  - Delete `test_mcmc.py`, `stoch_solver_vs_vanilla_smc.py`, `run_smc_parallel_test_da.py`, `run_smc_parallel.py`, `compare_wass.py`, and `comparison_plot.py`.

## 5. Configuration Framework (Hydra)
We adopt Hydra to manage all simulation parameters and method selections. This completely eliminates the need to comment/uncomment code in the main scripts.

### Directory Structure
```
conf/
├── config.yaml                     # Base configuration and defaults
├── problem/                        # Problem parameters and ground truths
│   ├── wos.yaml
│   └── toy.yaml
├── likelihood/                     # Dynamic likelihood model wrappers
│   ├── exact.yaml
│   ├── inexact_fix.yaml
│   └── inexact_adapt.yaml
├── mcmc/                           # MCMC method configurations
│   ├── rw_reg_cal.yaml
│   ├── rw_unc.yaml
│   └── noisy_metropolis_cal.yaml
└── sampler/                        # Sampler method configurations
    ├── adaptive_tempering.yaml
    └── temp_adapt_nsamp_no_rew.yaml
```

### Usage Pipeline
- **Problem & Likelihood Separation**: Static parameters and ground-truth parameter lists are stored in `conf/problem/`. Likelihood model selections (`exact`, `inexact_fix`, `inexact_adapt`) are in `conf/likelihood/` and resolve target classes dynamically using interpolation (`${problem.path}`).
- **Dynamic Instantiation**: The exact model solver (e.g. `WOSHD`, `MatVecModel`) and the likelihood wrapper are instantiated dynamically at runtime. The ground-truth parameters are loaded directly from the YAML config as pre-computed arrays, completely avoiding python `if/else` checks or external generator functions.
- **Execution**: The main simulation script uses the Hydra **Compose API** (`initialize` / `compose`) programmatically, allowing you to specify configurations inside the function call (e.g., `run_simulation(problem="toy", likelihood="inexact_fix")`) without touching command-line flags.

## 6. Execution Steps
**CRITICAL RULE:** After each refactoring step, I will run `pytest tests/` to verify correctness and then **stop to ask for your feedback**. I will not proceed to the next step until you explicitly approve the test results.

1. Create `tests/` structure, generate reference solutions, and verify tests pass. (Wait for feedback)
2. Update `requirements.yml`. (Run tests, wait for feedback)
3. Eliminate dead code from `core/samplers.py` and `core/mcmc.py`. (Run tests, wait for feedback)
4. Delete outdated root scripts. (Run tests, wait for feedback)
5. Build the `conf/` directory structure with modular YAML files. (Run tests, wait for feedback)
6. Refactor `run_smc_hd.py` to use Hydra and dynamic problem/likelihood imports. (Run tests, wait for feedback)

## Future TODOs (Not Part of This Plan)
- **2-Stage Ray Refactoring:** Refactor the parallel launch files `is_parallel_test_2stage.py`, `utils/is_get_ref_sol.py`, and `utils/is_trial.py` to use dynamic Hydra configurations once the core simulation pipeline refactoring is completed and fully verified.


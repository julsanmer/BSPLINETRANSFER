# Working in BSPLINETRANSFER

## Start here

- Read `README.md` for runnable examples and `ARCHITECTURE.md` for the original Gondelach/B-spline study. Check the actual CLI and tests before relying on either document for newer Pythagorean code.
- Inspect `git status --short` before editing. This workspace often contains user work that is modified or untracked. Preserve it; do not reset, clean, or overwrite it.
- Run commands from the repository root. Use `.venv_plot/bin/python` when the local environment is present; otherwise use a Python environment with `requirements.txt` installed.

## Code map

- `optimizer/optimization_Bspline_freetf.py`: cylindrical B-spline optimization and reference metrics.
- `optimizer/pythagorean_3d_bernstein.py` and `optimizer/pythagorean_3d_sundman.py`: Pythagorean shaping, physical checks, and thrust certificates.
- `optimizer/optimizers.py`: two-stage B-spline-to-Sundman pipeline and seed fitting.
- `experiments/compare_gondelach_fig2_fig3_bspline10.py`: paper comparison grids. Related reproduction and postprocessing scripts are in `experiments/`.
- `experiments/run_sundman_continuation_grid.py`: departure/TOF/revolution continuation. Each chain marches over TOF; a cell can have cold and warm attempts.
- `experiments/reference_metrics.py` and `optimizer/canonical_units.py`: shared metric and unit conversions. Use them when comparing Delta-V or acceleration across formulations.
- `tests/`: focused `unittest` checks. `kernels/`: SPICE meta-kernels and local binary kernels.
- `output/` and `tmp/`: generated results and scratch experiments; both are ignored by Git. Papers, archives, and notebooks are separate from the active optimizer workflow.

## Efficient workflow

- For a question about one solver, trace its CLI entry point, the called optimizer, and the corresponding tests before launching an experiment.
- Prefer a single transfer or a small grid slice for debugging. Large grids solve many independent NLPs and can take a long time, especially when both cold and warm starts are attempted.
- Give every experiment a fresh output directory that identifies the target, degree, control count, objective, and relevant solver settings. The Sundman grid can resume saved successes from its output directory; do not assume a resumed run used the current CLI arguments.
- In the two-stage pipeline, B-spline must succeed before the Pythagorean solve is attempted. Distinguish setup/seed, B-spline, IPOPT, interrupted runs, and post-solve numerical-check failures. Inspect `run.log`, `pythagorean_sundman_seed.json`, `pythagorean_sundman/solution.json`, and `selection.json` as applicable.
- Treat `status == Solve_Succeeded` or `Solved_To_Acceptable_Level` together with `numerical_checks_passed == true` as a successful Pythagorean result. Compare `delta_v_reference_km_s` and physical constraint checks, not only IPOPT's objective or exit code.
- Matching nominal spline degree and number of controls does not make cylindrical B-spline and Pythagorean/Sundman formulations equivalent. Record the parameterization, seed source, quadrature, revolution count, and thrust enforcement when comparing them.
- `--bspline-quadrature` configures the first stage of `optimizer.optimizers`; check the separate Pythagorean quadrature setting before interpreting a comparison. In `min-dv` without `u_max`, thrust enforcement is effectively disabled.

## Verification

- Run relevant focused tests first, for example `.venv_plot/bin/python -m unittest tests.test_pythagorean_3d_sundman tests.test_optimizers_continuation` for Sundman changes.
- For shared optimizer changes, run `.venv_plot/bin/python -m unittest discover -s tests`. Add one small representative solve when numerical behavior changes; verify its saved solution and metrics.
- SPICE cases require the referenced meta-kernel and binary kernels. For a quick dependency-light check, prefer an existing circular or Keplerian example where it tests the same logic.
- Keep exploratory variants in `tmp/` or a fresh output folder. Report the exact command, relevant parameters, result status, numerical checks, and any limits of the experiment.

---
name: simpeg-inversion
description: Plan, validate, execute and review 3D SimPEG gravity gz, induced magnetics TMI, or structurally coupled gravity–magnetics inversion on a UBC TensorMesh. Select L2 or sparse IRLS regularization per property, optimizer, coupling weight and output formats from natural language. Use for gravity/magnetics joint inversion and audit of active-cell/XYZ order.
compatibility: Requires Python, NumPy, PyYAML, discretize; real inversion also requires SimPEG with shell/filesystem permission. Tested static validation does not imply numerically validated inversions.
metadata:
  version: "2.0.0"
---

# SimPEG Inversion Agent Skill

## Purpose and scope
Use this Skill to translate the user's **natural-language inversion request** into an inspectable YAML configuration, validate inputs, and—**only with explicit authorization**—execute the supplied Python workflow. The agent must NOT claim that this Markdown file itself performs calculations.

Supported numerical modes:
- **Gravity (`mode: gravity`)**: 3D `gz` anomaly inversion → active-cell density contrast (`g/cm3`).
- **Magnetic (`mode: magnetic`)**: 3D **TMI anomaly, scalar isotropic induced susceptibility** inversion (`SI`), requiring inducing field amplitude, inclination and declination. Not remanent or MVI.
- **Joint (`mode: joint`)**: 2-parameter vector `[density(active), susceptibility(active)]` and one joint objective; `joint.coupling: cross_gradient` adds SimPEG `CrossGradient` structural coupling. `joint.coupling: none` solves a **shared optimization problem without structural coupling**. Do not call this option a physically coupled inversion.

For each property, use `regularization.[gravity|magnetic].type` = `l2` or `sparse`. Sparse mode uses `[p_s, p_x, p_y, p_z]` and SimPEG IRLS, with `0 <= p <= 2`. The joint mode supports L2/L2, Sparse/Sparse and mixed L2/Sparse. Optimizers: `inexact_gauss_newton` (unbounded) or `projected_gncg` (optionally bounded). Outputs: `.npy`, XYZ `.csv`, UBC `.ubc`, per-physics observed-vs-predicted CSV, and a summary JSON.

## Natural-language instructions → reproducible actions
1. **Identify intent and mode.** Distinguish inspect-only, gravity, magnetic and joint. Map user language precisely: "平滑" → L2; "稀疏/块状" → Sparse with suitable norms, not a promise of a blocky model; "联合反演" → ask or infer from context whether cross-gradient is scientifically justified. Never silently treat two independent fits as coupled joint inversion.
2. **Choose a template.** Copy exactly one of:
   - [`assets/gravity_l2.yaml`](assets/gravity_l2.yaml)
   - [`assets/magnetic_sparse.yaml`](assets/magnetic_sparse.yaml)
   - [`assets/joint_cross_gradient.yaml`](assets/joint_cross_gradient.yaml)
   Update `mode`, observation paths, mesh, surface definition, units, field orientation, priors, optimizer, Lp norms, weights and output paths as required. Config paths are relative to **the YAML file**, not the Skill root.
3. **Verify assumptions BEFORE solving.** Consult [`references/data-contract.md`](references/data-contract.md). Require 3D UBC TensorMesh; observations CSV `x,y,z,gz,sigma` and/or `x,y,z,tmi,sigma`. XYZ metres, +Z upward. Gravity `gz` positive up in mGal; magnetic TMI anomaly in nT, and magnetic field reference in nT/degrees. Ensure sigma is one-sigma and positive, data have proper anomaly corrections, coordinates share a CRS/vertical datum, and flat/topography mask is correct. **No invented observations, uncertainties or inducing-field values.** Values in templates are examples only.
4. **Validate.** `python scripts/validate_inputs.py CONFIG.yaml`. Report validation warnings and the dense-sensitivity lower-bound estimate. If dependencies are missing, explain that configuration-only checks are possible but mesh validation and execution are not. Never fabricate a success result.
5. **Prepare execution.** Review parameter count, regularization scaling, joint coupling weight (depends on cell scale and units), misfit weighting and RAM; `max_sensitivity_gb` checks **only one dense sensitivity estimate, NOT peak memory**. Compare with actual RAM; no implicit GPU assumption. For a joint problem, gravity/magnetic data misfit normalization depends on sigma and user-specified weights. A target normalized error of 1 only has meaning with defensible uncertainties.
6. **Run with authorization.** `python scripts/run_inversion.py CONFIG.yaml --execute`; no costly real inversion without user's explicit approval of run and costs. Do not use `--overwrite` unless the user authorizes replacement. Check versions of SimPEG/discretize if API errors appear; do not hide runtime failures.
7. **Review outputs scientifically.** Verify finite model arrays, separate normalized RMSE for gz and TMI, residual maps and bias, beta trajectory/convergence (if recorded separately), bounds, density/magnetic amplitude, resolution, effect of coupling and compatibility with geology. A converged solution or lower RMSE **does not** prove a correct geology model.
8. **Report:** task, exact selected options (including regularization p norms and CG weight), input paths, mask size, memory lower bound, versions, warnings, execution state, output paths, misfit per dataset and next checks. Distinguish `CONFIGURED`, `VALIDATED`, `EXECUTED_NOT_SCIENTIFICALLY_VALIDATED` and `SCIENTIFICALLY_REVIEWED`.

## Correctness constraints
- `mesh.cell_centers` **defines the authoritative XYZ order**. Never independently reshape, transpose or reorder cell values. Active-only vector length is `active.sum()`, full vector length is `mesh.n_cells`. `full[active] = model` and inactive zero are **export conventions**, not geologic assertions.
- Joint model layout is **density first**, susceptibility second, both active-only, same 3D mesh and same active-cell mask; joint weights and coupled model slices must be consistent.
- Distinguish **cross-gradient structural alignment** from equality or petrophysical relationship of density and susceptibility. Cross-gradient does not guarantee coincident amplitudes, interfaces or uniqueness.
- Sparse IRLS requires an L2 warm-up and reweighting; it may terminate early or converge to local solutions. Keep physical bounds explicit when appropriate. Never use negative susceptibility bounds without a domain reason.
- Do not invent data, topography, background field, uncertainties, grid origin, numeric physical limits or regularization weights. **Inspect** users' real files, propose assumptions, and request missing indispensable physical quantities before a real inversion.
- Preserve inputs and existing outputs unless user explicitly authorizes overwrite. Use execution only with tools and permissions explicitly available to the host Agent.

## Entry points
- Preflight: `python scripts/validate_inputs.py config.yaml` (needs discretize).
- Preflight through runner: `python scripts/run_inversion.py config.yaml`.
- Authorized numerical inversion: `python scripts/run_inversion.py config.yaml --execute`.
- Check local tests: `python -m pytest tests/ -q`.

## Coverage limits
This version covers only **3D UBC TensorMesh, gravity gz and induced-field scalar TMI**, using integral simulation. It does not provide gravity gradients, MVI, remanent magnetization, petrophysical PGI, tree-mesh workflows, tiled inversion, automated L-curve tuning or spatially varying bounds/priors. The cross-gradient path follows a supported SimPEG construction pattern but **must be tested on the target environment and synthetic test case** before scientific production use. See README for testing status.

## References
- [`references/data-contract.md`](references/data-contract.md)
- [`references/algorithm-notes.md`](references/algorithm-notes.md)
- SimPEG tutorials: https://simpeg.xyz/user-tutorials/
- Cross-gradient joint workflow: https://mirageoscience-simpeg-drivers.readthedocs-hosted.com/en/v0.5.0/case_studies/Forrestania/python_code/joint_grav_mag.html

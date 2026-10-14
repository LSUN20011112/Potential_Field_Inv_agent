# Algorithm notes and options

For gravity and magnetic observations, the data-misfit contribution is based on a weighted sum of normalized residual squares (SimPEG's implementation may include 1/2). For a joint run:

`phi_d = w_g * phi_d_gravity + w_m * phi_d_magnetic`.

`phi_reg = a_g * R_g(density) + a_m * R_m(susceptibility) + gamma * R_cross_gradient` for `joint.coupling: cross_gradient`; the last term is omitted for `none`. The inversion objective is `phi_d + beta*phi_reg`. `R_cross_gradient` penalizes non-parallel structural gradients; it does not impose property proportionality or common signs.

The YAML includes separate smallness (`alpha_s`) and directional (`alpha_x`,`alpha_y`,`alpha_z`) weights for each physical property. These are **not** calibrated automatically. Choosing `gamma` requires sensitivity and scale studies. A value like 10000.0 in the example may be grossly inappropriate for a different cell size or property scaling.

### Sparse

`Sparse` uses `norms: [p_s,p_x,p_y,p_z]` in `[0,2]`. This is a sparse-objective approximation via IRLS, not a guarantee of global optimum. Mixing L2 and sparse in a joint inversion is allowed, but when *any* property is sparse a shared IRLS directive updates the sparse term(s). Version differences in SimPEG directives may require verifying installed names/signatures.

### Optimizers

- `inexact_gauss_newton`: unbounded, iterative CG-based Hessian solve, avoids storing large dense Hessian.
- `projected_gncg`: supports per-property lower and upper bound arrays via `optimizer.bounds.gravity/magnetic`. Bounds supplied to a non-projected optimizer trigger validation failure.

### Iteration and diagnostics

The implementation estimates starting beta by dominant eigenspectra (`BetaEstimate_ByEig`). For L2-only it cools beta (`BetaSchedule`) and targets normalized misfit. For sparse it invokes SimPEG's IRLS updater instead of a conflicting explicit `BetaSchedule`. Sensitivity weighting and preconditioning are enabled. Results summary records final normalized RMSE but does not yet persist beta or convergence trajectory; use captured console output for intermediate history.

### Running limitations

- `store_sensitivities="ram"`; no out-of-core, tiled solve or GPU pipeline. Dense sensitivity lower bound is **not a peak-RAM estimate**.
- CSV and UBC output order is aligned by `mesh.cell_centers` and discretize's `write_model_UBC`.
- Real numerical run and joint-coupling compatibility require SimPEG + discretize in the *target* environment and a synthetic physics test case. Local static/unit checks alone don't verify scientific results.

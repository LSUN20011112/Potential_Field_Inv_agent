# Data contract (read before numerical work)

## File inputs

- `mesh_ubc`: 3D UBC TensorMesh `.msh` file read by `discretize.TensorMesh.read_UBC`.
- `observations.gravity`: CSV with exactly required header fields `x,y,z,gz,sigma`, optionally additional ignored columns. Values of **gz** and sigma in mGal; `gz` **up-positive** under SimPEG XYZ. Provide **anomalies**, not raw absolute gravity, after appropriate corrections.
- `observations.magnetic`: CSV header `x,y,z,tmi,sigma`. `tmi` is **induced total-magnetic-intensity anomaly** (nT), sigma in nT; not arbitrary total field including background. An assumed scalar susceptibility model may fail for remanence, strong self-demagnetization or anisotropy.
- `topography_csv` (alternative to `active_below_z`): CSV `x,y,z`, XYZ metres. Needs wide enough survey XY coverage; extrapolation can corrupt active mask.
- `magnetic_field` required for magnetic/joint: `amplitude_nT`, `inclination_deg`, `declination_deg`. Values **must reflect actual survey/reference field**. Example config values are placeholders.

## Units and geometry

- metres and consistent CRS / datum; Z upward, same coordinate system in all input data.
- Density contrast: g/cm³; susceptibility: **dimensionless SI**.
- Cross-gradient assumes the two models share an **identical mesh and active mask**. The default joint parameter vector contains two *different* properties of length `n_active` each: first density, then susceptibility.
- `mesh.cell_centers` order is authoritative, not `np.meshgrid` flatten assumptions. NPY full-model order matches mesh-cell order. UBC `.ubc` output is written by discretize to preserve required UBC file order.
- Inactive values exported as zero are dummy values, not inferred geology. Compare active-cell masks separately.

## Defaults and assumptions

- The YAML example values for field geometry, physical bounds, regularization ratios and cross-gradient multipliers are **illustrative**, not scientifically calibrated parameters.
- `regularization.gravity/magnetic.type: l2` uses SimPEG `WeightedLeastSquares`; `sparse` uses `Sparse` with p norms and iterative reweighting. `[0,1,1,1]` means a sparse (approximate L0) smallness term with approximately L1 smoothness terms.
- Optimizer: InexactGaussNewton (unbounded) or ProjectedGNCG (optional bounds). Use bounds grounded in expected geology; an upper susceptibility bound of 1 SI is NOT universal.
- `joint.coupling: cross_gradient` is **structural**, does not force numerical equality of two properties. `none` has no structural coupling, even though parameters are jointly optimized in a shared objective.
- Sensitivity memory lower bound (float64 dense G storage): `(n_gz + n_tmi)*n_active*8 bytes`; actual peak consumption can be far higher, especially with regularization, transformations and solver buffers. Refuse run if estimated lower bound exceeds budget; otherwise review available memory.

## Scientific review after run

- Inspect normalized residual `(observed - predicted)/sigma` for each dataset and spatial patterns, not only the scalar RMS.
- Recovered model magnitude and geometry vs data coverage, padding, depth, mesh resolution, survey height, regularization and bounds.
- Inspect multiple L2 / IRLS / CG weights against independent geological constraints. This solution is non-unique; fit is not proof.
- If both data types measured at different locations, that is allowed (shared mesh), but joint regularization should be evaluated for physical plausibility.
- A config can pass syntax/preflight while carrying a wrong field orientation or reference, units, or target geologic assumption. No automated test substitutes for domain review.

"""Preflight and optionally run gravity, magnetic, or joint inversion.

Usage:
    python scripts/run_inversion.py path/to/config.yaml
    python scripts/run_inversion.py path/to/config.yaml --execute

No run is performed without --execute. Real data must be reviewed first.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np
from validate_inputs import config_path, validate_problem


def _forward_simulations(problem, model_map):
    from simpeg import data, data_misfit
    from simpeg.potential_fields import gravity, magnetics

    cfg, mesh, active = problem["config"], problem["mesh"], problem["active"]
    simulations = {}
    misfits = {}
    for field, observations in problem["observations"].items():
        xyz = observations[:, :3]
        if field == "gravity":
            receiver = gravity.receivers.Point(xyz, components="gz")
            source = gravity.sources.SourceField(receiver_list=[receiver])
            survey = gravity.survey.Survey(source)
            simulation = gravity.simulation.Simulation3DIntegral(
                survey=survey, mesh=mesh, rhoMap=model_map[field], active_cells=active,
                store_sensitivities="ram",
            )
        else:
            receiver = magnetics.receivers.Point(xyz, components="tmi")
            inducing = cfg["magnetic_field"]
            source = magnetics.sources.UniformBackgroundField(
                receiver_list=[receiver],
                amplitude=inducing["amplitude_nT"],
                inclination=inducing["inclination_deg"],
                declination=inducing["declination_deg"],
            )
            survey = magnetics.survey.Survey(source)
            simulation = magnetics.simulation.Simulation3DIntegral(
                survey=survey, mesh=mesh, chiMap=model_map[field],
                active_cells=active, model_type="scalar", store_sensitivities="ram",
            )
        dat = data.Data(survey, dobs=observations[:, 3], standard_deviation=observations[:, 4])
        simulations[field] = simulation
        misfits[field] = data_misfit.L2DataMisfit(data=dat, simulation=simulation)
    return simulations, misfits


def _regularization(problem, maps):
    from simpeg import regularization

    cfg, mesh, active = problem["config"], problem["mesh"], problem["active"]
    joint = cfg["mode"] == "joint"
    fields = tuple(problem["observations"])
    n = int(active.sum())
    if joint:
        # The parameter vector is [density_contrast(active), susceptibility(active)].
        wires = maps.Wires(("density", n), ("susceptibility", n))
        model_maps = {"gravity": wires.density, "magnetic": wires.susceptibility}
    else:
        wires = None
        model_maps = {fields[0]: maps.IdentityMap(nP=n)}

    n_param = n * len(fields)
    start = np.concatenate([np.full(n, cfg["starting_model"][f]) for f in fields])
    reference = np.concatenate([np.full(n, cfg["reference_model"][f]) for f in fields])
    regs = []
    for field in fields:
        setting = cfg["regularization"][field]
        kwargs = dict(
            mesh=mesh, active_cells=active, mapping=model_maps[field],
            reference_model=reference,
            alpha_s=setting["alpha_s"], alpha_x=setting["alpha_x"],
            alpha_y=setting["alpha_y"], alpha_z=setting["alpha_z"],
            reference_model_in_smooth=False,
        )
        if setting["type"] == "sparse":
            reg = regularization.Sparse(**kwargs)
            reg.norms = setting["norms"]
        else:
            reg = regularization.WeightedLeastSquares(**kwargs)
        regs.append(reg)

    if joint:
        cg_choice = cfg["joint"]["coupling"]
        if cg_choice == "cross_gradient":
            regs.append(regularization.CrossGradient(mesh, wires, active_cells=active))
        # Explicit multipliers avoid hiding the cross-gradient scaling choice.
        reg_total = regs[0] + regs[1]
        if len(regs) == 3:
            reg_total = reg_total + regs[2]
        weights = [cfg["joint"]["gravity_regularization_weight"],
                   cfg["joint"]["magnetic_regularization_weight"]]
        if len(regs) == 3:
            weights.append(cfg["joint"]["cross_gradient_weight"])
        reg_total.multipliers = weights
    else:
        reg_total = regs[0]

    if start.size != n_param or reference.size != n_param:
        raise RuntimeError("Unexpected parameter-vector length.")
    return model_maps, start, reg_total


def _optimizer(cfg: dict, fields: tuple[str, ...], n_active: int):
    from simpeg import optimization

    opt_cfg = cfg["optimizer"]
    if opt_cfg["name"] == "inexact_gauss_newton":
        return optimization.InexactGaussNewton(
            maxIter=opt_cfg["max_iterations"],
            maxIterLS=opt_cfg["max_line_search_iterations"],
            cg_maxiter=opt_cfg["max_cg_iterations"],
            cg_rtol=opt_cfg["cg_tolerance"],
        )
    args = dict(
        maxIter=opt_cfg["max_iterations"],
        maxIterLS=opt_cfg["max_line_search_iterations"],
        maxIterCG=opt_cfg["max_cg_iterations"],
        tolCG=opt_cfg["cg_tolerance"],
    )
    bounds = opt_cfg.get("bounds", {})
    if bounds:
        args["lower"] = np.concatenate([np.full(n_active, bounds[f]["lower"]) for f in fields])
        args["upper"] = np.concatenate([np.full(n_active, bounds[f]["upper"]) for f in fields])
    return optimization.ProjectedGNCG(**args)


def _irls_directive(irls_cfg):
    from simpeg import directives

    cls = getattr(directives, "UpdateIRLS", None) or getattr(directives, "Update_IRLS", None)
    if cls is None:
        raise RuntimeError("Installed SimPEG has no UpdateIRLS/Update_IRLS directive.")
    params = inspect.signature(cls).parameters
    kwargs = {"max_irls_iterations": irls_cfg["max_iterations"]}
    # SimPEG renamed this argument in some API versions.
    if "min_gn_iterations" in params:
        kwargs["min_gn_iterations"] = irls_cfg["min_gn_iterations"]
    elif "minGNiter" in params:
        kwargs["minGNiter"] = irls_cfg["min_gn_iterations"]
    return cls(**kwargs)


def _directives(cfg: dict, use_sparse: bool):
    from simpeg import directives

    result = [
        directives.UpdateSensitivityWeights(every_iteration=False),
        directives.BetaEstimate_ByEig(beta0_ratio=cfg["beta"]["beta0_ratio"]),
    ]
    if use_sparse:
        result.append(_irls_directive(cfg["irls"]))
    else:
        result.extend([
            directives.BetaSchedule(
                coolingFactor=cfg["beta"]["cooling_factor"],
                coolingRate=cfg["beta"]["cooling_rate"],
            ),
            directives.TargetMisfit(chifact=1.0),
        ])
    result.append(directives.UpdatePreconditioner())
    return result


def output_names(fields, formats):
    filenames = ["summary.json"]
    for field in fields:
        filenames.append(f"{field}_observed_predicted.csv")
        if "npy" in formats:
            filenames.extend([f"{field}_active.npy", f"{field}_full.npy"])
        if "csv" in formats:
            filenames.append(f"{field}_xyz.csv")
        if "ubc" in formats:
            filenames.append(f"{field}_full.ubc")
    if "npy" in formats and len(fields) == 2:
        filenames.append("joint_active.npy")
    return filenames


def _save_outputs(problem, recovered, simulation, summary, output):
    cfg = problem["config"]
    mesh, active = problem["mesh"], problem["active"]
    fields = tuple(problem["observations"])
    n = int(active.sum())
    formats = cfg["output"]["formats"]
    results = {}
    for i, field in enumerate(fields):
        model = recovered[i * n:(i + 1) * n]
        full = np.zeros(mesh.n_cells, dtype=float)
        full[active] = model  # Authoritative cell order = mesh.cell_centers.
        predicted = np.asarray(simulation[field].dpred(recovered), dtype=float).reshape(-1)
        observed = problem["observations"][field]
        if predicted.size != len(observed) or not np.isfinite(predicted).all():
            raise RuntimeError(f"Non-finite or shape-incompatible predictions for {field}.")
        residual = observed[:, 3] - predicted
        nrms = float(np.sqrt(np.mean(np.square(residual / observed[:, 4]))))
        results[field] = {
            "normalized_rmse": nrms,
            "rms_in_data_units": float(np.sqrt(np.mean(residual**2))),
            "property_min": float(model.min()), "property_max": float(model.max()),
            "property_unit": "g/cm3" if field == "gravity" else "SI",
            "observation_unit": "mGal" if field == "gravity" else "nT",
        }
        np.savetxt(output / f"{field}_observed_predicted.csv",
                   np.column_stack((observed[:, :3], observed[:, 3], predicted, residual, observed[:, 4])),
                   delimiter=",", header="x,y,z,observed,predicted,residual,sigma", comments="")
        if "npy" in formats:
            np.save(output / f"{field}_active.npy", model)
            np.save(output / f"{field}_full.npy", full)
        if "csv" in formats:
            label = "density_contrast_g_cm3" if field == "gravity" else "susceptibility_SI"
            np.savetxt(output / f"{field}_xyz.csv", np.column_stack((mesh.cell_centers, full)),
                       delimiter=",", header="x,y,z," + label, comments="")
        if "ubc" in formats:
            mesh.write_model_UBC(str(output / f"{field}_full.ubc"), full)
    if "npy" in formats and len(fields) == 2:
        np.save(output / "joint_active.npy", recovered)
    summary["results"] = results
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--execute", action="store_true", help="Explicitly request real numerical inversion")
    p.add_argument("--overwrite", action="store_true", help="Replace prior output files (requires user consent)")
    args = p.parse_args()
    try:
        problem = validate_problem(args.config)
        cfg, stats = problem["config"], problem["summary"]
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        if not args.execute:
            print("PREFLIGHT ONLY. Add --execute only after scientific review and user authorization.")
            return 0
        if stats["minimum_dense_sensitivity_gb"] > cfg["resources"]["max_sensitivity_gb"]:
            raise RuntimeError("Dense sensitivity lower-bound estimate exceeds configured budget.")
        out = config_path(cfg, cfg["output"]["dir"])
        fields = tuple(problem["observations"])
        existing = [str(out / f) for f in output_names(fields, cfg["output"]["formats"]) if (out / f).exists()]
        if existing and not args.overwrite:
            raise FileExistsError("Existing outputs; --overwrite only with user consent: " + ", ".join(existing))

        import simpeg
        import discretize
        from simpeg import maps, inverse_problem, inversion
        model_maps, start, reg = _regularization(problem, maps)
        simulations, misfits = _forward_simulations(problem, model_maps)
        if len(fields) == 2:
            misfit = misfits["gravity"] + misfits["magnetic"]
            misfit.multipliers = [cfg["joint"]["gravity_misfit_weight"],
                                  cfg["joint"]["magnetic_misfit_weight"]]
        else:
            misfit = misfits[fields[0]]
        optimizer = _optimizer(cfg, fields, int(problem["active"].sum()))
        inv_prob = inverse_problem.BaseInvProblem(misfit, reg, optimizer)
        uses_sparse = any(cfg["regularization"][f]["type"] == "sparse" for f in fields)
        inv = inversion.BaseInversion(inv_prob, _directives(cfg, uses_sparse))
        recovered = np.asarray(inv.run(start), dtype=float).reshape(-1)
        if len(recovered) != len(start) or not np.isfinite(recovered).all():
            raise RuntimeError("Recovered model is non-finite or has unexpected shape.")
        summary = {
            **stats,
            "status": "EXECUTED_NOT_SCIENTIFICALLY_VALIDATED",
            "simpeg_version": getattr(simpeg, "__version__", "unknown"),
            "discretize_version": getattr(discretize, "__version__", "unknown"),
            "optimizer": cfg["optimizer"]["name"],
            "regularization": {f: cfg["regularization"][f] for f in fields},
            "joint_settings": cfg.get("joint") if cfg["mode"] == "joint" else None,
            "output_dir": str(out),
        }
        out.mkdir(parents=True, exist_ok=True)
        _save_outputs(problem, recovered, simulations, summary, out)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate a 3D SimPEG potential-field inversion configuration and input data.

Usage: python scripts/validate_inputs.py /path/to/config.yaml
Requires discretize for UBC mesh reading; does not require SimPEG or run inversion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

FIELDS = ("gravity", "magnetic")
KINDS = ("l2", "sparse")
MODES = ("gravity", "magnetic", "joint")
FORMATS = ("npy", "csv", "ubc")
OPTIMIZERS = ("inexact_gauss_newton", "projected_gncg")


def _number(value, label, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    f = float(value)
    if not np.isfinite(f) or (positive and f <= 0) or (nonnegative and f < 0):
        raise ValueError(f"{label} must be finite and within allowed range.")
    return f


def _int(value, label, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}.")
    return value


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a YAML mapping.")
    return value


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    _mapping(cfg, "configuration")
    cfg["_base_dir"] = path.parent
    cfg["_config_path"] = path
    mode = cfg.get("mode")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}.")
    if cfg.get("coordinate_convention") != "z_up":
        raise ValueError("coordinate_convention must explicitly be 'z_up' (metres, Z up).")
    if not isinstance(cfg.get("mesh_ubc"), str) or not cfg["mesh_ubc"]:
        raise ValueError("mesh_ubc must be a nonempty path.")
    if (cfg.get("active_below_z") is None) == (cfg.get("topography_csv") is None):
        raise ValueError("Specify exactly one of active_below_z or topography_csv.")
    if cfg.get("active_below_z") is not None:
        _number(cfg["active_below_z"], "active_below_z")
    elif not isinstance(cfg["topography_csv"], str) or not cfg["topography_csv"]:
        raise ValueError("topography_csv must be a nonempty path.")

    obs = _mapping(cfg.get("observations"), "observations")
    requested = (FIELDS if mode == "joint" else (mode,))
    for field in requested:
        if not isinstance(obs.get(field), str) or not obs[field].strip():
            raise ValueError(f"observations.{field} is required in {mode} mode.")
    if mode in ("gravity", "joint") and cfg.get("gravity_unit") != "mGal":
        raise ValueError("gravity_unit must be 'mGal' (upward-positive gz).")
    if mode in ("magnetic", "joint"):
        if cfg.get("magnetic_unit") != "nT":
            raise ValueError("magnetic_unit must be 'nT' (TMI anomaly).")
        field = _mapping(cfg.get("magnetic_field"), "magnetic_field")
        _number(field.get("amplitude_nT"), "magnetic_field.amplitude_nT", positive=True)
        inc = _number(field.get("inclination_deg"), "magnetic_field.inclination_deg")
        dec = _number(field.get("declination_deg"), "magnetic_field.declination_deg")
        if not -90 <= inc <= 90 or not -360 <= dec <= 360:
            raise ValueError("Magnetic inclination must be [-90,90], declination [-360,360].")
    if cfg.get("density_unit") != "g/cm3" and mode in ("gravity", "joint"):
        raise ValueError("density_unit must be 'g/cm3'.")
    if cfg.get("susceptibility_unit") != "SI" and mode in ("magnetic", "joint"):
        raise ValueError("susceptibility_unit must be 'SI'.")

    regs = _mapping(cfg.get("regularization"), "regularization")
    for field in requested:
        reg = _mapping(regs.get(field), f"regularization.{field}")
        kind = reg.get("type")
        if kind not in KINDS:
            raise ValueError(f"regularization.{field}.type must be one of {KINDS}.")
        for setting in ("alpha_s", "alpha_x", "alpha_y", "alpha_z"):
            _number(reg.get(setting), f"regularization.{field}.{setting}", nonnegative=True)
        if sum(float(reg[k]) for k in ("alpha_s", "alpha_x", "alpha_y", "alpha_z")) == 0:
            raise ValueError(f"All regularization weights cannot be zero ({field}).")
        if kind == "sparse":
            norms = reg.get("norms")
            if not isinstance(norms, list) or len(norms) != 4:
                raise ValueError(f"regularization.{field}.norms must be [p_s,p_x,p_y,p_z].")
            for p in norms:
                if _number(p, "sparse norm") < 0 or float(p) > 2:
                    raise ValueError("Sparse p norms must lie in [0,2].")

    joint = _mapping(cfg.get("joint", {}), "joint")
    if mode == "joint":
        coupling = joint.get("coupling")
        if coupling not in ("none", "cross_gradient"):
            raise ValueError("joint.coupling must be 'none' or 'cross_gradient'.")
        for name in ("gravity_misfit_weight", "magnetic_misfit_weight", "gravity_regularization_weight", "magnetic_regularization_weight"):
            _number(joint.get(name), f"joint.{name}", positive=True)
        weight = _number(joint.get("cross_gradient_weight"), "joint.cross_gradient_weight", nonnegative=True)
        if coupling == "cross_gradient" and weight == 0:
            raise ValueError("For cross_gradient coupling, cross_gradient_weight must be > 0.")

    optimizer = _mapping(cfg.get("optimizer"), "optimizer")
    if optimizer.get("name") not in OPTIMIZERS:
        raise ValueError(f"optimizer.name must be one of {OPTIMIZERS}.")
    for key in ("max_iterations", "max_cg_iterations", "max_line_search_iterations"):
        _int(optimizer.get(key), f"optimizer.{key}")
    _number(optimizer.get("cg_tolerance"), "optimizer.cg_tolerance", positive=True)
    bounds = _mapping(optimizer.get("bounds", {}), "optimizer.bounds")
    if optimizer["name"] == "inexact_gauss_newton" and bounds:
        raise ValueError("InexactGaussNewton is unbounded. Use projected_gncg for bounds.")
    if bounds and optimizer["name"] == "projected_gncg":
        for field in requested:
            b = _mapping(bounds.get(field), f"optimizer.bounds.{field}")
            lower = _number(b.get("lower"), f"bounds.{field}.lower")
            upper = _number(b.get("upper"), f"bounds.{field}.upper")
            if lower >= upper:
                raise ValueError(f"bounds.{field}.lower must be < upper.")
    start = _mapping(cfg.get("starting_model"), "starting_model")
    ref = _mapping(cfg.get("reference_model"), "reference_model")
    for field in requested:
        _number(start.get(field), f"starting_model.{field}")
        _number(ref.get(field), f"reference_model.{field}")
        if bounds:
            b = bounds[field]
            if not b["lower"] <= start[field] <= b["upper"]:
                raise ValueError(f"starting_model.{field} falls outside declared bounds.")

    beta = _mapping(cfg.get("beta"), "beta")
    _number(beta.get("beta0_ratio"), "beta.beta0_ratio", positive=True)
    _number(beta.get("cooling_factor"), "beta.cooling_factor", positive=True)
    _int(beta.get("cooling_rate"), "beta.cooling_rate")
    if beta["cooling_factor"] <= 1:
        raise ValueError("beta.cooling_factor must be > 1.")
    if any(regs[f]["type"] == "sparse" for f in requested):
        irls = _mapping(cfg.get("irls"), "irls")
        _int(irls.get("max_iterations"), "irls.max_iterations")
        _int(irls.get("min_gn_iterations"), "irls.min_gn_iterations")

    resources = _mapping(cfg.get("resources"), "resources")
    _number(resources.get("max_sensitivity_gb"), "resources.max_sensitivity_gb", positive=True)
    output = _mapping(cfg.get("output"), "output")
    if not isinstance(output.get("dir"), str) or not output["dir"]:
        raise ValueError("output.dir is required.")
    formats = output.get("formats")
    if not isinstance(formats, list) or not formats or any(x not in FORMATS for x in formats):
        raise ValueError(f"output.formats must be a nonempty subset of {FORMATS}.")
    if len(set(formats)) != len(formats):
        raise ValueError("output.formats may not repeat a format.")
    return cfg


def config_path(cfg, key_or_path):
    path = Path(key_or_path).expanduser()
    return (path if path.is_absolute() else cfg["_base_dir"] / path).resolve()


def read_csv(path: Path, columns: tuple[str, ...]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8", autostrip=True)
    if table.dtype.names is None:
        raise ValueError(f"Missing CSV header: {path}")
    missing = set(columns) - set(table.dtype.names)
    if missing:
        raise ValueError(f"CSV {path}: missing columns {sorted(missing)}")
    table = np.atleast_1d(table)
    if not len(table):
        raise ValueError(f"CSV is empty: {path}")
    values = np.column_stack([table[name] for name in columns])
    if not np.isfinite(values).all():
        raise ValueError(f"CSV contains NaN/inf or missing numeric values: {path}")
    return values


def validate_problem(config_path_arg: str | Path) -> dict:
    cfg = load_config(config_path_arg)
    try:
        from discretize import TensorMesh
        from discretize.utils import active_from_xyz
    except ImportError as exc:
        raise RuntimeError("Install discretize in this Python environment to validate UBC mesh.") from exc
    mesh_file = config_path(cfg, cfg["mesh_ubc"])
    if not mesh_file.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh_file}")
    mesh = TensorMesh.read_UBC(str(mesh_file))
    if mesh.dim != 3:
        raise ValueError("Expected 3D UBC TensorMesh.")
    if cfg.get("topography_csv"):
        topo = read_csv(config_path(cfg, cfg["topography_csv"]), ("x", "y", "z"))
        if topo.shape[0] < 3 or np.ptp(topo[:, 0]) == 0 or np.ptp(topo[:, 1]) == 0:
            raise ValueError("Topography needs at least three points and nonzero XY coverage.")
        active = np.asarray(active_from_xyz(mesh, topo), dtype=bool)
    else:
        active = np.asarray(mesh.cell_centers[:, 2] < cfg["active_below_z"], dtype=bool)
    if active.size != mesh.n_cells or not active.any():
        raise ValueError("Active mask inconsistent with mesh or empty.")

    warnings = []
    if active.all():
        warnings.append("All mesh cells active; verify surface definition.")
    obs = {}
    requested = FIELDS if cfg["mode"] == "joint" else (cfg["mode"],)
    for field in requested:
        col = "gz" if field == "gravity" else "tmi"
        arr = read_csv(config_path(cfg, cfg["observations"][field]), ("x", "y", "z", col, "sigma"))
        if not (arr[:, 4] > 0).all():
            raise ValueError(f"{field} uncertainties (sigma) must be > 0.")
        if np.unique(arr[:, :3], axis=0).shape[0] < len(arr):
            warnings.append(f"{field}: duplicate receiver positions found.")
        if np.ptp(arr[:, 3]) == 0:
            warnings.append(f"{field}: all observed anomalies are identical.")
        if cfg.get("active_below_z") is not None and (arr[:, 2] < cfg["active_below_z"]).any():
            warnings.append(f"{field}: stations below declared flat surface.")
        if cfg.get("topography_csv"):
            if ((arr[:, :2].min(axis=0) < topo[:, :2].min(axis=0)).any() or
                    (arr[:, :2].max(axis=0) > topo[:, :2].max(axis=0)).any()):
                warnings.append(f"{field}: stations outside topography XY bounding box.")
        for j, axis in enumerate("xy"):
            centers = mesh.cell_centers[:, j]
            if arr[:, j].min() < centers.min() or arr[:, j].max() > centers.max():
                warnings.append(f"{field}: receiver {axis} extent beyond mesh cell-center extent.")
        obs[field] = arr

    n = int(active.sum())
    estimate = n * sum(obs[f].shape[0] for f in requested) * 8 / 1e9
    budget = cfg["resources"]["max_sensitivity_gb"]
    if estimate > budget:
        warnings.append("Dense sensitivity lower-bound estimate exceeds configured budget; execution will be refused.")
    summary = {
        "status": "VALIDATED_WITH_WARNINGS" if warnings else "VALIDATED",
        "mode": cfg["mode"],
        "joint_coupling": cfg["joint"]["coupling"] if cfg["mode"] == "joint" else "not_applicable",
        "mesh_shape": [int(x) for x in mesh.shape_cells],
        "n_cells": int(mesh.n_cells),
        "n_active": n,
        "n_model_parameters": n * len(requested),
        "n_observations": {f: int(len(obs[f])) for f in requested},
        "minimum_dense_sensitivity_gb": estimate,
        "max_sensitivity_gb": budget,
        "regularization": {f: cfg["regularization"][f]["type"] for f in requested},
        "optimizer": cfg["optimizer"]["name"],
        "warnings": warnings,
    }
    return {"config": cfg, "mesh": mesh, "active": active, "observations": obs, "summary": summary}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    args = p.parse_args()
    try:
        result = validate_problem(args.config)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

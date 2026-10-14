"""Unit tests requiring no SimPEG/discretize runtime (mocked only)."""
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_inputs import load_config, validate_problem
from run_inversion import _save_outputs, output_names


@pytest.mark.parametrize("name,mode", [
    ("gravity_l2.yaml", "gravity"),
    ("magnetic_sparse.yaml", "magnetic"),
    ("joint_cross_gradient.yaml", "joint"),
])
def test_templates_parse(name, mode):
    cfg = load_config(ROOT / "assets" / name)
    assert cfg["mode"] == mode


def test_bad_joint_weight(tmp_path):
    cfg = yaml.safe_load((ROOT / "assets" / "joint_cross_gradient.yaml").read_text())
    cfg["joint"]["cross_gradient_weight"] = 0.0
    f = tmp_path / "bad.yaml"
    f.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="cross_gradient_weight"):
        load_config(f)


def test_bad_optimizer_bound_combo(tmp_path):
    cfg = yaml.safe_load((ROOT / "assets" / "magnetic_sparse.yaml").read_text())
    cfg["optimizer"]["name"] = "inexact_gauss_newton"
    f = tmp_path / "bad.yaml"
    f.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="unbounded"):
        load_config(f)


class FakeMesh:
    dim = 3
    shape_cells = (2, 1, 2)
    n_cells = 4
    cell_centers = np.array([[0., 0., -5.], [10., 0., -5.],
                             [0., 0., 5.], [10., 0., 5.]])

    @classmethod
    def read_UBC(cls, filename):
        return cls()

    def write_model_UBC(self, filename, values):
        np.savetxt(filename, values)


class FakeSimulation:
    def __init__(self, multiplier):
        self.multiplier = multiplier

    def dpred(self, model):
        return np.array([self.multiplier * model[0]])


def test_preflight_joint_and_export(tmp_path, monkeypatch):
    f_discretize = types.ModuleType("discretize")
    f_discretize.TensorMesh = FakeMesh
    f_utils = types.ModuleType("discretize.utils")
    f_utils.active_from_xyz = lambda m, p: np.array([True, True, False, False])
    monkeypatch.setitem(sys.modules, "discretize", f_discretize)
    monkeypatch.setitem(sys.modules, "discretize.utils", f_utils)

    (tmp_path / "mesh.msh").write_text("fake mesh")
    (tmp_path / "grav.csv").write_text("x,y,z,gz,sigma\n0,0,6,1,0.5\n")
    (tmp_path / "mag.csv").write_text("x,y,z,tmi,sigma\n10,0,6,2,1\n")
    cfg = yaml.safe_load((ROOT / "assets" / "joint_cross_gradient.yaml").read_text())
    cfg["mesh_ubc"] = "mesh.msh"
    cfg["observations"] = {"gravity": "grav.csv", "magnetic": "mag.csv"}
    cfg["output"]["dir"] = "output"
    filename = tmp_path / "config.yaml"
    filename.write_text(yaml.safe_dump(cfg))
    result = validate_problem(filename)
    assert result["summary"]["n_active"] == 2
    assert result["summary"]["n_model_parameters"] == 4
    assert result["summary"]["n_observations"] == {"gravity": 1, "magnetic": 1}

    out = tmp_path / "output"
    out.mkdir()
    recovered = np.array([0.1, 0.2, 0.01, 0.02])
    sim = {"gravity": FakeSimulation(1), "magnetic": FakeSimulation(100)}
    summary = {"status": "SYNTHETIC_FAKE_ONLY"}
    _save_outputs(result, recovered, sim, summary, out)
    np.testing.assert_array_equal(np.load(out / "gravity_full.npy"), [0.1, 0.2, 0, 0])
    np.testing.assert_array_equal(np.load(out / "magnetic_full.npy"), [0.01, 0.02, 0, 0])
    np.testing.assert_array_equal(np.load(out / "joint_active.npy"), recovered)
    arr = np.genfromtxt(out / "gravity_xyz.csv", delimiter=",", names=True)
    np.testing.assert_array_equal(arr["density_contrast_g_cm3"], [0.1, 0.2, 0, 0])
    assert set(output_names(("gravity", "magnetic"), ("npy", "csv", "ubc"))) >= {"summary.json", "joint_active.npy"}


def test_negative_sigma_fails(tmp_path, monkeypatch):
    f_discretize = types.ModuleType("discretize")
    f_discretize.TensorMesh = FakeMesh
    f_utils = types.ModuleType("discretize.utils")
    f_utils.active_from_xyz = lambda m, p: np.array([True, True, False, False])
    monkeypatch.setitem(sys.modules, "discretize", f_discretize)
    monkeypatch.setitem(sys.modules, "discretize.utils", f_utils)
    (tmp_path / "mesh.msh").write_text("fake")
    (tmp_path / "grav.csv").write_text("x,y,z,gz,sigma\n0,0,6,1,-1\n")
    cfg = yaml.safe_load((ROOT / "assets" / "gravity_l2.yaml").read_text())
    cfg["mesh_ubc"] = "mesh.msh"
    cfg["observations"]["gravity"] = "grav.csv"
    file = tmp_path / "config.yaml"
    file.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="sigma"):
        validate_problem(file)


def test_mixed_joint_and_uncoupled_config(tmp_path):
    cfg = yaml.safe_load((ROOT / "assets" / "joint_cross_gradient.yaml").read_text())
    cfg["joint"]["coupling"] = "none"
    cfg["regularization"]["gravity"].update({"type": "sparse", "norms": [0, 1, 1, 1]})
    cfg["irls"] = {"max_iterations": 10, "min_gn_iterations": 1}
    cfg["optimizer"]["name"] = "inexact_gauss_newton"
    cfg["optimizer"].pop("bounds")
    config = tmp_path / "test.yaml"
    config.write_text(yaml.safe_dump(cfg))
    assert load_config(config)["regularization"]["gravity"]["type"] == "sparse"


def test_invalid_sparse_norm(tmp_path):
    cfg = yaml.safe_load((ROOT / "assets" / "magnetic_sparse.yaml").read_text())
    cfg["regularization"]["magnetic"]["norms"] = [0, 2.1, 1, 1]
    config = tmp_path / "bad.yaml"
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="Sparse p norms"):
        load_config(config)

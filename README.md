# SimPEG Inversion Agent Skill — v2

Natural-language-configured **3D gravity, induced-magnetic and cross-gradient joint inversion** with SimPEG, supported by YAML and Python scripts.

## Install

Copy the whole **`simpeg-inversion`** folder into the Skills directory of an Agent that supports `SKILL.md` and file/shell execution, then follow its skill-discovery / reload instructions.

Install Python dependencies in **the Agent's Python environment** (isolated env recommended):

```bash
python -m pip install -r requirements.txt
```

This needs a compatible Python + SimPEG/discretize install. The intended baseline API follows SimPEG 0.25.x patterns; API compatibility should be reviewed for other versions.

## Three starting configurations

- `assets/gravity_l2.yaml`: 3D gz, smooth density contrast, unbounded Inexact Gauss-Newton.
- `assets/magnetic_sparse.yaml`: 3D TMI, compact susceptibility via sparse IRLS, projected GNCG.
- `assets/joint_cross_gradient.yaml`: joint gz+TMI with cross-gradient coupling and independent L2 terms. For uncoupled joint baseline change `joint.coupling: none`.

**Copy** a template alongside input data and edit the paths and field/units/physics assumptions. Do not run unchanged example YAML against nonexistent example data.

```bash
python scripts/validate_inputs.py /your/project/run.yaml
python scripts/run_inversion.py /your/project/run.yaml  # preflight only
python scripts/run_inversion.py /your/project/run.yaml --execute  # real work
```

`--execute` signals an execution request but does not replace informed user authorization and scientific review. `--overwrite` is separately required for existing outputs and must be authorized.

## Example natural-language commands

> Use my gravity observations and existing UBC mesh for an L2 inversion. Use InexactGaussNewton, 25 iterations, export full NPY, XYZ CSV and UBC. First validate the data; do not execute yet.

> Re-run magnetic TMI using a scalar-induced susceptibility model and sparse norms [0,1,1,1], 12 IRLS iterations, with physical nonnegative bounds. Confirm the inducing field with me before running.

> Use the same active mask to jointly invert gravity and magnetics using cross-gradient coupling. Compare with the no-coupling baseline. Try two coupling weights and report *separate* normalized RMS errors and model differences. Do not claim the models are equivalent physical properties.

## Numerical support, verification and limitations

| Feature | Implemented | Validation status |
|---|---|---|
| Gravity gz + L2 | Yes | Code/static & config checks; real run not verified here |
| Magnetic TMI induced + L2 | Yes | Code/static & config checks; real run not verified here |
| Sparse Lp / IRLS | Yes | Config checks; real SimPEG run not verified here |
| Joint cross-gradient | Yes (separate density/susceptibility maps) | API-pattern implementation; real SimPEG run not verified here |
| Independent per-property regularization | Yes | Config checks; real SimPEG run not verified here |
| Bounds via ProjectedGNCG | Yes | Config checks; real SimPEG run not verified here |
| Full NPY / UBC / XYZ CSV | Yes | Standalone I/O checks; real UBC roundtrip not verified here |
| TreeMesh, remanence/MVI, PGI, tiling | No | Not supported |

**Important**: SimPEG and discretize are not installed in the artifact-building environment, so real forward/inversion runs cannot be honestly certified. Tests check configuration/validation logic and export data placement using a fake mesh and fake simulation, not real SimPEG. Before real scientific use, install dependencies and run a small synthetic reference problem in your environment.

## Project Structure & File Responsibilities

This project is designed as an **execution-oriented Agent Skill**, combining natural-language instructions, domain-specific knowledge, configuration templates, and executable Python tools.

The Skill enables an AI Agent to understand geophysical inversion requests, select appropriate inversion methods, configure parameters, and execute computational workflows using SimPEG.

### Directory Structure

```text
simpeg-inversion/
│
├── SKILL.md
├── README.md
├── requirements.txt
│
├── scripts/
│   ├── validate_inputs.py
│   └── run_inversion.py
│
├── assets/
│   ├── gravity_l2.yaml
│   ├── magnetic_sparse.yaml
│   └── joint_cross_gradient.yaml
│
├── references/
│   └── ...
│
└── tests/
    └── ...
```

### File Responsibilities

| Component | Description | Required? |
|---|---|---|
| `SKILL.md` | Defines the Agent's instructions, supported tasks, execution procedures, and scientific constraints. | Yes |
| `README.md` | Provides installation instructions, usage examples, and developer documentation. | Optional |
| `scripts/` | Contains Python tools for validating inputs and executing SimPEG inversion workflows. | Required for the provided execution workflow |
| `assets/` | Provides YAML configuration templates for gravity, magnetic, and joint inversion. | Optional, recommended |
| `references/` | Contains supplementary technical documentation, data specifications, and scientific guidance. | Optional |
| `tests/` | Contains tests for configuration handling, validation, and other implemented functionality. | Recommended for development |
| `requirements.txt` | Specifies the Python dependencies required by the executable tools. | Recommended |

### Two Usage Modes

**Mode A: Knowledge-Based Skill**

The Agent primarily uses `SKILL.md` and supporting references as domain-specific instructions.

In this mode, the Agent generates or modifies SimPEG Python code according to natural-language requests. Existing execution scripts are not strictly required.

This approach offers greater flexibility for developing new inversion workflows.

**Mode B: Execution-Based Skill (Current Implementation)**

The Agent uses `SKILL.md` to interpret user requests and follows predefined procedures to configure and invoke the Python tools in `scripts/`.

The workflow includes input validation, inversion configuration, numerical execution, and result export.

This approach promotes reproducibility, consistency, and easier testing.

### Relationship Between Skills, Tools, and Agents

The `SKILL.md` file provides instructions but does not independently execute computations.

The Python scripts provide computational capabilities, while the hosting AI Agent is responsible for interpreting user requests, selecting appropriate procedures, invoking available tools, and evaluating execution results.

When integrating this project with LangChain or LangGraph, the existing Python functions can be exposed as callable tools, while the Skill documentation serves as domain-specific guidance for the Agent.

**Note:** The numerical inversion implementations require validation in a compatible SimPEG environment before scientific use. An Agent's ability to execute a workflow does not guarantee the physical validity or numerical convergence of its results.

References:
- https://simpeg.xyz/user-tutorials/
- https://mirageoscience-simpeg-drivers.readthedocs-hosted.com/en/v0.5.0/case_studies/Forrestania/python_code/joint_grav_mag.html
- https://agentskills.io/specification

# Modules Repository

Shared codebase containing core functionality, utilities, and common components used across all molecular dynamics and machine learning pipeline modules.

## Overview

This repository provides the foundational infrastructure that powers the entire pipeline. It contains reusable code shared between:

* **Training CGSchNet-based Models** (`base_model`)
* **Benchmark Suite** (`benchmark`)

By centralizing common functionality, this module ensures consistency, reduces code duplication, and simplifies maintenance across the entire pipeline.

## Status

⚠️ **IMPORTANT**: Our code is currently being ported and refactored from private repositories for public release. The full codebase with documentation and tutorials will be provided within one to two weeks.

## Key Components

### Core Molecular Dynamics

#### `torchforcefield.py`
Pure PyTorch implementation of classical molecular mechanics forcefields
- **Features**:
  - Bond, angle, dihedral, and non-bonded (Lennard-Jones) terms
  - Periodic boundary conditions support
  - GPU acceleration
  - Compatible with TorchMD interface
- **Key Classes**:
  - `TorchForceField`: Main forcefield container
  - `TFF_Bond`, `TFF_Angle`, `TFF_Dihedral`: Individual force terms
  - `TFF_RepulsionCG`: Coarse-grained repulsion

#### `external_nn.py`
Neural network-based force calculations for hybrid ML/classical potentials
- **Features**:
  - Bond, angle, and dihedral neural network priors
  - Vectorized batch computation
  - Automatic differentiation for forces
  - Integration with classical forcefields
- **Key Classes**:
  - `ExternalNN`: Neural network force calculator
  - `ParametersNN`: Parameter management for NN terms

#### `make_deltaforces.py`
Delta force computation for training ML models
- **Purpose**: Calculate the difference between all-atom and coarse-grained forces
- **Features**:
  - Classical force computation
  - Neural network force integration
  - Batch processing support
  - Memory-efficient implementation
- **Key Classes**:
  - `DeltaForces`: Main delta force calculator

### Machine Learning Models

#### `torchmdnet/model.py`
Extended TorchMD-Net architectures with custom features
- **Features**:
  - Graph neural network models (TorchMD-GN)
  - Equivariant representations
  - Multiple output heads
  - Prior model integration
  - Harmonic model support
- **Key Classes**:
  - `TorchMD_Net_Ext`: Extended model with multiple outputs
  - `create_model`: Model factory function

#### `torchmdnet/harmonic_model.py`
Learnable harmonic forcefield derived from embeddings
- **Features**:
  - Neural network-parameterized harmonic terms
  - Bond, angle, and dihedral learning
  - Integration with TorchMD-Net representations
  - End-to-end differentiable
- **Key Classes**:
  - `TorchMD_Net_Harmonic`: Model wrapper
  - `HarmonicModel`: Harmonic term calculator

#### `torchmdnet/deep_scalar.py`
Deep MLP-based output module for TorchMD-Net
- **Features**:
  - Configurable depth and width
  - Dropout support
  - Flexible activation functions
- **Key Classes**:
  - `DeepScalar`: Deep output network

#### `torchmdnet/torchmd_gn_ext.py`
Extended graph network with external embeddings
- **Features**:
  - Sequence-based features
  - External embedding injection
  - Configurable neighbor radius
  - Custom RBF functions
- **Key Classes**:
  - `TorchMD_GN_Ext`: Extended graph network
  - `ExternalEmbedding`: External embedding processor

### Prior Fitting and Parameterization

#### `prior.py`
Classical prior fitting from MD trajectories
- **Features**:
  - Boltzmann inversion for bonded terms
  - Histogram-based parameter extraction
  - Bond, angle, dihedral, and non-bonded fitting
  - Temperature-dependent parameterization
- **Key Classes**:
  - `ParamBondedCalculator`: Bond parameter fitting
  - `ParamAngleCalculator`: Angle parameter fitting
  - `ParamDihedralCalculator`: Dihedral parameter fitting
  - `ParamNonbondedCalculator`: Lennard-Jones parameter fitting
  - Null variants for zero-energy priors

#### `prior_flex.py`
Flexible neural network-based prior fitting
- **Features**:
  - Gaussian Process regression for smooth potentials
  - Neural network distillation from GP models
  - Periodic boundary handling for dihedrals
  - Polynomial baseline fitting
- **Key Classes**:
  - `ParamBondedFlexCalculator`: NN bond prior fitting
  - `ParamAngleFlexCalculator`: NN angle prior fitting
  - `ParamDihedralFlexCalculator`: NN dihedral prior fitting
  - `NeuralNet`: Neural network architecture for priors
  - `GPCustom`: Custom Gaussian Process with mean functions

### Coarse-Graining and Mapping

#### `cg_mapping.py`
All-atom to coarse-grained mapping infrastructure
- **Features**:
  - Flexible bead definitions
  - Force projection (AggForce integration)
  - Optimal force mapping
  - Topology generation for CG models
- **Key Classes**:
  - `CGMapping`: Main mapping class

#### `psfwriter.py`
PSF/topology file generation for CG models
- **Features**:
  - CA-only and CA-CB mappings
  - Beta-turn detection and tagging
  - Chain gap handling
  - MDTraj compatibility
- **Key Functions**:
  - `pdb2psf_CA`: Carbon-alpha topology generation
  - `pdb2psf_CACB`: CA-CB topology generation

#### `torchmd_cg_mappings.py`
Residue-specific atom type mappings
- **Mappings**:
  - `CA_MAP`: Carbon-alpha atom types per residue
  - `CACB_MAP`: CA-CB atom types per residue

### Data Handling

#### `dataset.py`
PyTorch dataset for molecular dynamics training data
- **Features**:
  - Multi-protein batch collation
  - Memory-mapped loading for large datasets
  - Dynamic batch sizing based on atom counts
  - Support for coordinates, forces, energies, embeddings
  - Classical term (bonds/angles/dihedrals) handling
- **Key Classes**:
  - `ProteinDataset`: Main dataset class
  - `ProteinBatchCollate`: Custom batch collator
  - `NumpyReader`: Memory-efficient numpy file reader

### WESTPA Integration

#### `westpa_helpers.py`
Utilities for WESTPA simulation analysis and integration
- **Features**:
  - Configuration file parsing
  - Weight extraction from HDF5 archives
  - Trajectory loading (DCD and NPZ formats)
  - Topology creation and fixing
  - Component value calculations
  - Weight extension for frame-wise analysis
- **Key Functions**:
  - `load_all_weights_and_trajs_flat`: Load all segments with weights
  - `get_topology_from_westpa`: Extract topology from WESTPA config
  - `extend_weights`: Expand trajectory-level weights to frame-level
  - `calculate_component_values`: Compute TICA/progress coordinates

### Utilities

#### `custom_rbf.py`
Custom radial basis functions for distance featurization
- **Features**:
  - DimeNet-style Bessel RBF
  - Smooth envelope functions
  - Trainable and fixed variants
  - Visualization tools
- **Key Classes**:
  - `BesselRBF`: Bessel radial basis functions
  - `Envelope`: Polynomial envelope for cutoff

#### `lr_scheduler_wrappers.py`
Learning rate scheduler wrappers for consistent interface
- **Features**:
  - Exponential decay
  - Cosine annealing with warm restarts
  - Reduce on plateau
  - Unified interface
- **Key Classes**:
  - `SchedulerWrapper`: Base wrapper class
  - Various scheduler implementations

#### `model_util.py`
Model loading and compatibility utilities
- **Features**:
  - Parameter name mapping between TorchMD versions
  - Checkpoint compatibility
- **Key Functions**:
  - `load_state_dict_with_rename`: Load with version compatibility

## Installation

### Prerequisites

```bash
# Core dependencies
- Python 3.10+
- PyTorch 2.0+
- TorchMD-Net
- MDTraj
- DeepTime
- NumPy, SciPy
- Moleculekit
- AggForce (for optimal force mapping)
- scikit-learn (for GP regression)
```

### Setup

*Detailed installation instructions will be added soon.*

## Usage Examples

### Basic Forcefield Usage

```python
from module.torchforcefield import TorchForceField
from moleculekit.molecule import Molecule

# Load molecule and forcefield
mol = Molecule("protein.psf")
ff = TorchForceField(
    "forcefield.yaml",
    mol,
    device="cuda",
    terms=["bonds", "angles", "dihedrals", "repulsioncg"],
    exclusions=["bonds", "angles"]
)

# Compute energy and forces
coords = torch.tensor(mol.coords, device="cuda")
forces = torch.zeros_like(coords)
energy = ff.forward(coords, box=None, forces_out=forces)
```

### Prior Fitting

```python
from module.prior import ParamBondedCalculator, ParamAngleCalculator
from moleculekit.molecule import Molecule
import mdtraj

# Load trajectory
mol = Molecule("topology.psf")
traj = mdtraj.load("trajectory.dcd", top="topology.pdb")

# Fit bond parameters
bond_calc = ParamBondedCalculator()
bond_calc.add_molecule(mol, traj)
bond_params = bond_calc.get_param(
    Temp=300,
    plot_directory="./plots"
)
```

### Neural Network Prior Fitting

```python
from module.prior_flex import ParamBondedFlexCalculator

# Fit flexible neural network priors
flex_calc = ParamBondedFlexCalculator()
flex_calc.add_molecule(mol, traj, cache_dir="./cache")
nn_params = flex_calc.get_param(
    Temp=300,
    plot_directory="./plots"
)

# Access trained networks
for bond_type, result in nn_params.items():
    best_net = result['bestNet']
    # Use best_net for inference
```

### Coarse-Grained Mapping

```python
from module.cg_mapping import CGMapping
from module.psfwriter import pdb2psf_CA

# Create CA-only topology
ca_mol = pdb2psf_CA(
    "all_atom.pdb",
    "cg_topology.psf",
    bonds=True,
    angles=True,
    dihedrals=True
)

# Create mapping
from preprocess import prior_types
prior = prior_types["CA_Majewski2022_v1"]()
aa_topology = mdtraj.load("all_atom.pdb").topology

cg_map = CGMapping(aa_topology, prior)

# Map coordinates and forces
cg_coords = cg_map.cg_positions(aa_coords)
cg_forces = cg_map.cg_optimal_forces(aa_traj, aa_forces)
```

### Dataset Creation

```python
from module.dataset import ProteinDataset, ProteinBatchCollate
from torch.utils.data import DataLoader

# Create dataset
dataset = ProteinDataset(
    directory="./data",
    pdb_ids=["protein1", "protein2", "protein3"],
    forces_file="deltaforces.npy",
    energy_file="energies.npy"
)

# Build classical terms
dataset.build_classical_terms()

# Create dataloader with custom collation
collate_fn = ProteinBatchCollate(atoms_per_call=10000)
dataloader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=collate_fn,
    shuffle=True
)
```

### WESTPA Analysis

```python
from module.westpa_helpers import (
    load_all_weights_and_trajs_flat,
    get_topology_from_westpa,
    extend_weights
)

# Load WESTPA data
westpa_weights, traj_paths = load_all_weights_and_trajs_flat(
    "west.h5",
    "westpa_output",
    ext="dcd"
)

# Get topology
topology = get_topology_from_westpa(
    "westpa_output",
    ext="dcd"
)

# Extend weights for frame-wise analysis
frame_weights = extend_weights(westpa_weights, frames_per_traj=100)
```

### TorchMD-Net Model Creation

```python
from module.torchmdnet.model import create_model

# Model configuration
args = {
    "model": "graph-network",
    "embedding_dimension": 128,
    "num_layers": 6,
    "num_rbf": 50,
    "rbf_type": "bessel",
    "trainable_rbf": True,
    "activation": "silu",
    "cutoff_lower": 0.0,
    "cutoff_upper": 5.0,
    "max_z": 100,
    "max_num_neighbors": 32,
    "derivative": True,
    "output_model": "Scalar",
    "reduce_op": "sum",
    "precision": "float32"
}

# Create model
model = create_model(args)

# Forward pass
z = torch.tensor([6, 6, 6, 7, 8])  # Atomic numbers
pos = torch.randn(5, 3)  # Positions
batch = torch.zeros(5, dtype=torch.long)

energy, forces, extras = model(z, pos, batch)
```

## Architecture

### Module Organization

```
module/
├── Core MD
│   ├── torchforcefield.py          # Pure PyTorch forcefields
│   ├── external_nn.py               # NN-based forces
│   └── make_deltaforces.py          # Delta force computation
├── Machine Learning
│   ├── torchmdnet/
│   │   ├── model.py                 # Model factory
│   │   ├── harmonic_model.py        # Learnable harmonic terms
│   │   ├── deep_scalar.py           # Deep output module
│   │   └── torchmd_gn_ext.py        # Extended graph network
│   └── custom_rbf.py                # Custom basis functions
├── Prior Fitting
│   ├── prior.py                     # Classical prior fitting
│   └── prior_flex.py                # NN-based prior fitting
├── Coarse-Graining
│   ├── cg_mapping.py                # CG mapping
│   ├── psfwriter.py                 # Topology generation
│   └── torchmd_cg_mappings.py       # Atom type mappings
├── Data Handling
│   ├── dataset.py                   # PyTorch dataset
│   └── westpa_helpers.py            # WESTPA utilities
└── Utilities
    ├── lr_scheduler_wrappers.py     # LR schedulers
    └── model_util.py                # Model utilities
```

### Design Principles

1. **Modularity**: Each component is self-contained and reusable
2. **Compatibility**: Consistent interfaces across the pipeline
3. **Performance**: GPU acceleration and memory efficiency
4. **Flexibility**: Configurable for different use cases
5. **Maintainability**: Centralized common functionality

## Key Features

### Hybrid ML/Classical Potentials
- Seamless integration of neural network and classical force terms
- Delta force training for learning corrections
- Multiple prior types (harmonic, neural network, Gaussian process)

### Flexible Force Calculations
- Pure PyTorch implementation for automatic differentiation
- Support for periodic boundary conditions
- Efficient batch processing

### Advanced Coarse-Graining
- Optimal force projection (AggForce)
- Flexible bead definitions
- Topology generation for various CG representations

### WESTPA Integration
- Weight-aware trajectory analysis
- Automated topology extraction
- Efficient loading of large ensemble data

### Production-Ready
- Memory-mapped data loading for large datasets
- GPU-accelerated computations
- Robust error handling

## Performance Considerations

### Memory Management
- Use `ProteinDataset` with `use_npfile=True` for large datasets
- Memory-mapped loading reduces RAM usage
- Batch size tuning with `ProteinBatchCollate`

### GPU Utilization
- All force calculations support GPU acceleration
- Batch computations for optimal throughput
- Automatic device management

### Parallelization
- Multi-GPU training support in higher-level modules
- Thread-safe trajectory loading
- Process-based parallelization for data preprocessing

## Integration with Other Modules

### base_model
- Uses `dataset.py` for training data
- Leverages `torchmdnet/` for model architectures
- Employs `prior.py` and `prior_flex.py` for prior fitting

### benchmark
- Relies on `westpa_helpers.py` for WESTPA analysis
- Uses force calculations for trajectory generation
- Integrates with model architectures for evaluation

### openmm_generate
- Uses `torchforcefield.py` for classical simulations
- Leverages `cg_mapping.py` for coarse-graining
- Employs `psfwriter.py` for topology generation

### westpa_prop
- Heavily uses `westpa_helpers.py`
- Integrates models from `torchmdnet/`
- Uses force calculations for propagation

## Testing and Validation

*Testing documentation will be added soon.*

## Contributing

We welcome contributions! Please use [GitHub Issues](../../issues) to:
- Report bugs
- Request features
- Suggest improvements
- Ask questions
---

**Note**: This module is under active development. Documentation, examples, and additional features will be added as the codebase is finalized for public release.

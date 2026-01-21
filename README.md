
---
# **OpTorch**
This repository contains the code for experiments associated to our paper
```
The Intrinsic Superiority of Optical Neural Networks Revealed by Full-Wave Computing
```

---
# ONN Model Complexity Analysis
Complexity analysis folder performs a complexity analysis of ONN networks. For a given ONN of specified size, it computes the Fisher Information Matrix (FIM), extracts the eigenvalues of the FIM and visualizes the Fisher information spectrum, and calculates the network's normalized effective dimension. For comparison, the same complexity analysis is conducted on fully connected feedforward neural networks with an equivalent number of parameters, enabling a theoretical comparison of the model performance between the two network types.

---

## Folder Structure


```
ONN_Training/
├── effective_dimension/       
├── complexity_analysis_FNN.py
└── complexity_analysis_waveonn.py
```
`effective_dimension` includes the construction of ONNs as well as the relevant definitions required for performing complexity analysis. The model complexity analysis for fully connected feedforward neural networks and optical neural networks can be carried out by running the `complexity_analysis_FNN.py` and `complexity_analysis_waveonn.py` scripts, respectively.

---

## Installation & Prerequisites

### 1. Software

* ​**Python**​: Version 3.9.

### 2. Python Libraries

* ​**Common**​: `numpy`, `scipy`, `matplotlib`, `qiskit`, `torch`.

---


# Optical Metasurface Inverse Design Framework

ONN Training folder hosts two distinct implementations for the inverse design of optical metasurfaces using ​**Ansys Lumerical FDTD**​. The framework utilizes the **Adjoint Method** to optimize the refractive index distribution of a diffractive region, enabling the device to perform machine learning classification tasks (Iris and MNIST) purely through optical scattering.

## Folder Structure


```
ONN_Training/
├── Iris_training/       # Sequential, Numpy-based implementation for Iris dataset
└── MNIST_training/      # Parallel, PyTorch-based implementation for MNIST dataset
```

---

## 1. Iris Classification (`Iris_training`)

This project optimizes a 3D waveguide structure to classify the Iris dataset (4 input features, 3 output classes). It uses a sequential optimization loop suitable for smaller-scale problems.

### Core Python Modules

* ​**`opt_main_3d.py`**​: The entry point. It orchestrates data loading, simulation execution, and parameter updates.
* ​**`simulation.py`**​: Interacts with the Lumerical API to handle forward/adjoint simulations and field data extraction.
* ​**`optimization.py`**​: Implements the **Adam optimizer** and calculates 2D/3D gradients using bilinear interpolation.
* ​**`quant.py`**​: Handles binarization. It uses a `tanh`-based projection and a beta schedule to gradually force continuous permittivity values into discrete materials (Silicon/Air).
* ​**`obj_1_3d_coherent_fom.lsf`**​: The Lumerical script defining the physical geometry, including the 4 input waveguides, 3 output waveguides, and monitors.

### Key Optimization Parameters

The following parameters in `opt_main_3d.py` control the physics and training loop:

| **Parameter** | **Default Value** | **Description**                   |
| --------------------- | ------------------------- | ----------------------------------------- | 
| `batch_size`    | 12                      | Samples processed per gradient update.  |
| `max_epoch`     | 1000                    | Total training epochs.                  |
| `index_min`       | 1.44             | Minimum permittivity (SiO2 background). |
| `index_max`       | 3.47             | Maximum permittivity (Silicon).         |
| `field_size`    | `[151, 151, 3]`     | Simulation grid resolution.             |

### Execution Flow

1. ​**Injection**​: Light is injected into 4 input ports based on phase values (0-1) from `training_data.npz`.
2. ​**Adjoint**​: Error is back-propagated from the 3 output ports.
3. ​**Update**​: Sensitivity is calculated, and pixel permittivity is updated via Adam.

---

## 2. MNIST Classification (`MNIST_training`)

This project implements a high-performance hybrid pipeline for classifying MNIST digits (10 inputs, 10 outputs). It integrates **PyTorch** for gradient calculations and **Python `multiprocessing`** to parallelize Lumerical FDTD simulations across multiple GPUs.

### Architecture & Files

* ​**`all_parallel_opt_main.py`**​: The main driver. It manages parallel processes and uses the `GradientCalculator` (PyTorch) to compute field overlaps.
* ​**`obj_1_mnist.lsf`**​: Defines the base geometry with 10 input/output waveguides and specific mesh override regions.
* ​**Lumerical Batch Scripts**​:
  * `simproj_for_1.lsf` / `simproj_for_2.lsf`: Configure Forward simulations for different input batches.
  * `simproj_adj_1.lsf` / `simproj_adj_2.lsf`: Configure Adjoint simulations for different output batches.

### Parallelization Configuration

**Crucial:** You must configure the hardware dispatch logic in `all_parallel_opt_main.py` to match your available GPUs. The script manually assigns tasks to GPU indices:

Python

```
# Inside all_parallel_opt_main.py
tasks = [
    # Task ID "for1" on GPU 0
    {"func": sim.make_forward_base_sim, "simmodel": sim_for_1, "gpu_num": 0, "task_id": "for1"}, 
    # Task ID "for2" on GPU 1
    {"func": sim.make_forward_base_sim, "simmodel": sim_for_2, "gpu_num": 1, "task_id": "for2"},
    # ... ensure these indices match your hardware
]
```

---

## Shared Technical Implementation

Both projects utilize specific techniques to ensure physical realizability and convergence:

1. Topology Optimization (Binarization):
   The design region starts with continuous permittivity values. A "continuation method" is used where a beta parameter increases over epochs. This sharpens the tanh projection filter, forcing pixels towards either eps\_min (Air) or eps\_max (Silicon).
2. Adjoint Method:
   Gradients are not computed via finite difference. Instead, the gradient is derived from the interference (overlap) of the Forward electric fields (from inputs) and Adjoint electric fields (back-propagated from outputs).

## Installation & Prerequisites

### 1. Software

* ​**Ansys Lumerical FDTD**​: Version 2024 R1 (v241) or compatible.
* ​**Python**​: Version 3.9.

### 2. Python Libraries

* ​**Common**​: `numpy`, `scipy`, `matplotlib`.
* ​**MNIST Only**​: `torch` (PyTorch) is required for the gradient backend. A CUDA-capable GPU is recommended.

### 3. API Configuration

You must link the Lumerical Python API (`lumapi`) in the main script of the project you are running:

Python

```
# In opt_main_3d.py OR all_parallel_opt_main.py
# Update this path to match your installation
sys.path.append(r"../lumerical/v241/api/python")
```

### 4. Data Preparation

* ​**Iris**​: Place `training_data.npz` in the `Iris_training` root.
* ​**MNIST**​: Place `train_dataset.npz` in the `MNIST_training` root.

## Outputs

Both pipelines generate a specific result folder (e.g., `sim_100nm_fom_mesh` or `phase_100nm_nn_pi`) containing:

* ​fom.txt`: Logs the Figure of Merit per iteration.
* ​`figure_iteration_X.png/svg`: Visualization of loss and `beta` evolution.
* ​`epochX.npz`​: Checkpoints containing the `eps_save` (permittivity) and optimizer state.
* ​`*.fsp`​: Lumerical simulation files for verification.

---

# Acknowledgements

This project builds upon parts of the following open-source libraries:

- [Repository](https://doi.org/10.5281/zenodo.4732830) — used for model complexity analysis.
- [pyFDTD](https://github.com/Unrealfancy/pyFDTD) — used for constructing the optical neural network.

Many thanks to the authors for making their work open-source.

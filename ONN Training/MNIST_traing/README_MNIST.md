# Optical Metasurface Inverse Design for MNIST Classification


This project implements a physics-driven inverse design pipeline to train an optical metasurface (diffractive optical neural network) for performing MNIST digit classification.

The framework combines **Ansys Lumerical FDTD** for rigorous electromagnetic simulation with **PyTorch** for gradient calculation and **Adam** for optimization. It utilizes the **Adjoint Method** to efficiently compute gradients with respect to the permittivity distribution of the design region, allowing for the optimization of thousands of design parameters simultaneously.

## Key Features

* **Hybrid Optimization Pipeline:** Integrates Lumerical FDTD (physics solver) with PyTorch (gradient backend).
* **Adjoint Method:** Efficiently calculates gradients by interfering Forward and Adjoint electromagnetic fields.
* **Parallel Simulation:** Utilizes Python `multiprocessing` and `concurrent.futures` to dispatch FDTD simulations across multiple GPUs/processes simultaneously to speed up the iteration cycle.
* **Topology Optimization / Binarization:** Implements a soft-to-hard projection strategy (using a `beta` continuation parameter) to evolve the design from continuous permittivity values to a discrete binary structure (e.g., Silicon/Air) suitable for fabrication.
* **Automated Data Logging:** Tracks Figure of Merit (FOM), convergence errors, and evolution parameters (`beta`) across epochs.

## Dependencies

### System Requirements

* **Ansys Lumerical FDTD:** (Tested with v241). You must have a valid license and the software installed.
* **GPU:** CUDA-capable GPU is recommended for the PyTorch gradient calculation and Lumerical FDTD acceleration.

### Python Libraries

Ensure you have the following Python packages installed:

Bash

```
pip install numpy matplotlib scipy torch
```

### Lumerical API (`lumapi`)

The project requires the Lumerical Python API. You must ensure `lumapi` is in your Python path. The code currently attempts to add it relative to the script location, but you may need to update `all_parallel_opt_main.py` or your environment variables:

Python

```
# In all_parallel_opt_main.py or your PYTHONPATH
sys.path.append("../lumerical/v241/api/python") # Check this path
```

## Project Structure

### Python Source Code

* `all_parallel_opt_main.py`: **Entry Point.** Orchestrates the main optimization loop, manages parallel processes, and handles PyTorch gradient calculations.
* `simulation.py`: Handles the execution of FDTD simulations (Forward and Adjoint) and extracts electromagnetic field data.
* `optimization.py`: Implementation of the Adam optimizer parameters and update logic.
* `quant.py`: Contains the binarization logic (hyperbolic tangent projection) and convergence judgment algorithms.
* `setting.py`: Utilities for file I/O, directory creation, interpolation, and initializing the Lumerical API.
* `plot.py`: Visualization tools for tracking optimization progress (FOM vs Iterations).

### Lumerical Scripts (`.lsf`)

* `obj_1_mnist.lsf`: Defines the base physical geometry, including the 10 input/output waveguides, simulation region, and mesh settings.
* `simproj_for_1.lsf`: Configuration for Forward Simulation (Batch 1: Input Sources 1-5).
* `simproj_for_2.lsf`: Configuration for Forward Simulation (Batch 2: Input Sources 6-10).
* `simproj_adj_1.lsf`: Configuration for Adjoint Simulation (Batch 1: Output Sources 1-5).
* `simproj_adj_2.lsf`: Configuration for Adjoint Simulation (Batch 2: Output Sources 6-10).

### Data

* `train_dataset.npz`: (Required) Input dataset containing MNIST training samples (phase/amplitude).

## Configuration & Setup

### 1. Data Preparation

Ensure you have the training data file `train_dataset.npz` in the root directory. This file should contain the phase/amplitude inputs corresponding to MNIST digits that will be injected into the input waveguides.

### 2. Lumerical Configuration

The physical parameters are defined in `obj_1_mnist.lsf`:

* **Waveguides:** 10 input and 10 output waveguides are configured.
* **Wavelength:** Set to 1550nm.
* **Mesh:** Default mesh accuracy is set to 3 with specific override regions.

### 3. Adjusting Computational Resources

In `all_parallel_opt_main.py`, the `run_task` dispatch logic hardcodes GPU indices (e.g., `gpu_num` 4, 5, 6, 7).

**You must modify this section to match your hardware availability:**

Python

```
# Inside all_parallel_opt_main.py
tasks = [
    {"func": sim.make_forward_base_sim, "simmodel": sim_for_1, "gpu_num": 0, "task_id": "for1"}, # Change gpu_num
    {"func": sim.make_forward_base_sim, "simmodel": sim_for_2, "gpu_num": 1, "task_id": "for2"},
    # ...
]
```

## Execution

To start the training process, run the main script:

Bash

```
python all_parallel_opt_main.py
```

### Process Flow

1. **Initialization:** Creates a new directory `phase_100nm_nn_pi` to store results.
2. **Base Simulation:** Initializes the Lumerical project files based on the `.lsf` scripts.
3. **Optimization Loop:**
   * **FDTD Step:** Runs Forward and Adjoint simulations in parallel to extract E-fields.
   * **Gradient Step:** Uses the `GradientCalculator` (PyTorch) to compute the overlap of fields and derived gradients based on the MNIST dataset batch.
   * **Update Step:** Updates material properties using Adam.
   * **Binarization:** Applies the projection filter. If convergence is detected, `beta` is increased to sharpen the features.
4. **Logging:** Saves `.npz` checkpoints (epsilon distribution) and generates convergence plots (`.svg`) in the result directory.

## Outputs

The script generates a new folder for every run (e.g., `phase_100nm_nn_pi/sim_file_1/`). Inside you will find:

* ​**`fom.txt`**​: Text log of the Figure of Merit for every iteration.
* ​**`figure_iteration_X.svg`**​: Plots showing the trajectory of the loss function and the `beta` parameter.
* ​**`epochX.npz`**​: Numpy archives containing the refractive index distribution (`eps_save`), optimization parameters, and optimizer state.
* ​**`*.fsp`**​: The Lumerical simulation files corresponding to the specific epoch.
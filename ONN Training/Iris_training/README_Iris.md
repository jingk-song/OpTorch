# Optical Metasurface Inverse Design for Iris Classification

This project implements an inverse design framework for optimizing optical metasurfaces to perform machine learning classification tasks. Specifically, it configures a 3D FDTD simulation to classify the **Iris dataset** (4 input features, 3 output classes) using a programmable diffractive region.

The core logic utilizes the **Adjoint Method** for gradient calculation and the **Adam optimizer** to update the refractive index distribution of the device. It includes a "continuation method" (beta schedule) to gradually binarize the design from continuous permittivity values to discrete materials (e.g., Silicon/Air).

## Project Structure

The codebase is organized into modular Python scripts and Lumerical simulation files:

* **`opt_main_3d.py`**: The main entry point. [cite_start]Orchestrates the optimization loop, data loading, simulation execution, and parameter updates.
* **`simulation.py`**: Interface for Ansys Lumerical API. [cite_start]Handles forward/adjoint simulations and field data extraction.
* **`optimization.py`**: Implements the Adam optimizer and 2D/3D gradient calculations using bilinear interpolation.
* **`quant.py`**: Contains the logic for binarization (quantization) of the refractive index using a `tanh`-based projection and beta-schedule convergence checking.
* **`setting.py`**: Utilities for initializing the FDTD geometry, creating file paths, and refreshing the design region pixels.
* **`plot.py`**: Visualization tools to track the Figure of Merit (FOM) and binarization parameter (Beta) during training.
* **`obj_1_3d_coherent_fom.lsf`**: Lumerical script defining the 4-input/3-output waveguide structure, monitors, and sources.

## Prerequisites & Dependencies

To run this project, you must have **Ansys Lumerical** (FDTD Solutions) installed and a valid license.

### Python Dependencies

* **Python 3.9**
* **Numpy**: For matrix operations and data handling.
* **Scipy**: For physical constants (`mu_0`, `epsilon_0`).
* **Matplotlib**: For plotting training progress.

### Software Requirements

* **Ansys Lumerical 2024 R1 (v241)** or compatible.
  * *Note:* You must ensure the Lumerical Python API path is correctly added to your system path or modified in the script.

## Installation & Setup

1. **Clone the repository:**
   
   ```bash
   git clone [https://github.com/your-username/optical-iris-classification.git](https://github.com/your-username/optical-iris-classification.git)
   cd optical-iris-classification
   ```
2. **Configure Lumerical API Path:**
   Open `opt_main_3d.py` and ensure the path to the Lumerical Python API matches your installation:
   
   ```python
   # opt_main_3d.py
   sys.path.append(r"../lumerical/v241/api/python") 
   # Or your specific path, e.g., "C:/Program Files/Lumerical/v241/api/python/"
   ```
3. **Prepare Training Data:**
   Ensure a file named `training_data.npz` is present in the root directory. This file should contain the Iris dataset batches, where:
   
   * Inputs: 4 phase values (0-1 range) normalized from Iris features.
   * Targets: One-hot encoded vectors for the 3 flower classes.

## Usage

Run the main optimization script to start the training process:

```bash
python opt_main_3d.py
```

### Execution Flow

1. **Initialization**: The script creates a new directory in `sim_100nm_fom_mesh` to store results.


2. **Geometry Setup**: It loads `obj_1_3d_coherent_fom.lsf` to build the waveguides and adds the design pixel grid.

3. **Optimization Loop**:

* **Forward Sim**: Injects light into the 4 input ports based on `training_data.npz`.
* **Adjoint Sim**: Back-propagates the error from the 3 output ports.
* **Gradient Update**: Calculates sensitivity and updates pixel permittivity using Adam.
* **Binarization**: As training progresses, the `beta` parameter increases to force pixels towards min/max permittivity values.

4. **Monitoring**:

* FOM plots are saved as `figure_iteration_{i}.png`.
* Simulation files (`.fsp`) and checkpoint data (`.npz`) are saved periodically.

## Key Parameters

You can modify optimization settings in `opt_main_3d.py`:

| Parameter | Default | Description |
| --- | --- | --- |
| `batch_size` | 12 | Number of samples processed before a gradient update. |
| `max_epoch` | 1000 | Total number of training epochs. |
| `eps_min` |  1.44^2 | Minimum permittivity (SiO2 background). |
| `eps_max` |  3.47^2 | Maximum permittivity (Silicon). |
| `field_size` | `[151, 151, 3]` | Resolution of the simulation grid. |

## Results

* **FOM Logs**: The loss and figure of merit are logged to `fom.txt`.
* **Checkpoints**: `epoch_{num}.npz` files contain the saved `eps_opt` (permittivity distribution) and optimizer state.
* **Final Design**: The final binarized structure is saved in the Lumerical file format for verification.

## License

[Insert License Here]

```
# PAPINN: Physics-Aware Implicit Neural Network for Limited-View PACT

PAPINN is a unified reconstruction framework for Photoacoustic Computed Tomography (PACT) that integrates implicit physical operators with neural fields. It is designed to solve the ill-posed inverse problem under limited-view geometric constraints using a memory-efficient differentiable propagator.

## Key Features

1. MESO-Core (Memory-Efficient Sparse Operator): A custom autograd kernel that enables gradient propagation through high-dimensional acoustic fields without constructing dense system matrices, significantly reducing VRAM usage.
2. Implicit Neural Representation: Parameterizes the initial acoustic pressure distribution using a sparse U-Net architecture, ensuring continuous and smooth reconstruction.
3. Geometric Aperture Control: Built-in support for variable aperture angles (e.g., 90, 150 degrees) with automatic apodization masking.
4. Auto-Calibration: Self-adaptive gain calibration to align physical units with the neural network latent space.

## Repository Structure

* meso_kernel.py: Differentiable Acoustic Kernel and Physics Engine core implementation.
* main.py: Neural Inversion Optimization Loop and solver entry point.
* AcousticKernel.pt: Pre-computed Implicit State Manifold (See Usage section).
* simulation_data/: Contains simulation artifacts for benchmarking (e.g., .mat files).

## Usage

### 1. Prerequisites

Ensure you have a Python environment with PyTorch installed (CUDA recommended). Required packages include numpy, scipy, h5py, matplotlib, and tqdm.

### 2. Prepare System Kernel

Due to the high dimensionality of the physical operator, the pre-computed kernel state is stored as a serialized tensor.

* If `AcousticKernel.pt` is not present in the repository due to file size limits, please download it from the Releases page of this repository.
* Place the file `AcousticKernel.pt` in the root directory of the project.

### 3. Run Reconstruction

To start the physics-informed optimization for the default limited-view scenario, run the main script:

python main.py

### 4. Configuration

You can modify the geometric constraints in `main.py` by changing the `APERTURE_ANGLE` variable (e.g., set to 150 for a 150-degree view).

## Results

The framework outputs diagnostic visualizations in the results directory during the optimization process, including:
* Initial Guess: The back-projection result (LBP) used as a warm start.
* Neural Residual: The nonlinear correction map learned by the network.
* Reconstruction: The final recovered acoustic field.

## License

This project is open-sourced under the MIT License.

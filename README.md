# PA-PINN: Photoacoustic Physics-Informed Neural Network for Limited-View PACT

PAPINN is a unified reconstruction framework for 
**Photoacoustic Computed Tomography (PACT)** based on Physics-Informed Neural Networks (PINN). It integrates a physical operator with a neural solver to address the ill-posed inverse problem under **limited-view geometric constraints**
.

Unlike traditional iterative methods, PAPINN leverages a custom differentiable kernel to enable direct gradient propagation through high-dimensional acoustic fields on consumer-grade GPUs.

## Key Features

1. **Memory-Efficient Differentiable Kernel**
: A custom autograd operator that performs sparse matrix-vector multiplication (SpMV) on the GPU. It avoids constructing dense system matrices, reducing VRAM usage by over 60% compared to standard implementations.
2. **Physics-Informed Neural Solver**
: Parameterizes the acoustic pressure distribution. The network is optimized to minimize the physics consistency loss (data fidelity) subject to the acoustic wave propagation equation.
3. **Geometric Aperture Control**
: Built-in support for variable aperture angles (e.g., 90, 150 degrees) with automatic apodization masking to simulate limited-view detection.
4. **Auto-Calibration**
: Self-adaptive gain calibration to align the scale of physical measurements with the neural network's latent space.

## Repository Structure

* `kernel.py`
: Core implementation of the Differentiable Acoustic Kernel and Physics Engine.
* `main.py`
: The main optimization loop for the Neural Inversion Solver.
* `AcousticKernel.pt`
: The pre-computed Sparse System Matrix (Download required, see Usage).
* `data/`
: Contains simulation artifacts for benchmarking (e.g., .mat files).

## Usage

### 1. Prerequisites

Ensure you have a Python environment with PyTorch installed (CUDA recommended).
Dependencies: 
`numpy`, `scipy`, `h5py`, `matplotlib`, `tqdm`
.

### 2. Prepare System Kernel

The system matrix involves complex physics simulations and is pre-computed to accelerate the training process. Due to its large size, it is hosted on Hugging Face.

* **Download Link**: [Hugging Face Dataset](https://huggingface.co/datasets/fifteen67/PAPINN/
)
* **Action**: Download `AcousticKernel.pt` and place it in the **root directory**
 of this repository.

### 3. Run Reconstruction

To start the physics-informed optimization for the default limited-view scenario:

```bash
python main.py
4. Configuration
You can modify the geometric constraints in main.py by changing the APERTURE_ANGLEvariable (e.g., set to 150 for a 150-degree view coverage).
Results
The framework outputs diagnostic visualizations in the results directory during the optimization process:
• Initial Guess: The Linear Back-Projection (LBP) result used as a warm start.
• Neural Residual: The artifact correction map learned by the PINN.
• Reconstruction: The final recovered acoustic field.
License
This project is open-sourced under the MIT License.

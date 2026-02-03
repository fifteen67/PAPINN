import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io
import h5py
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from kernel import AcousticPhysicsEngine

# ================= Configuration =================
EXPERIMENT_ID = 'recon_limited_view_v1'
SAVE_DIR      = fr'./results_{EXPERIMENT_ID}'
STATE_FILE    = r'./AcousticKernel.pt' 
OBS_DATA_FILE = r'matrix_data\simulation_small_radius_vessel.mat'

APERTURE_ANGLE = 150 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(SAVE_DIR, exist_ok=True)

DIM_GRID   = 128
DIM_SENSOR = 128 
DIM_TIME   = 1000

# Synthesizes the binary sensor apodization mask to enforce limited-view geometric constraints.
def generate_aperture_function():
    aperture = torch.ones((DIM_TIME, DIM_SENSOR), device=DEVICE)
    n_active = max(1, min(DIM_SENSOR, int(np.round(DIM_SENSOR * (float(APERTURE_ANGLE) / 360.0)))))
    if n_active < DIM_SENSOR:
        aperture[:, n_active:] = 0
    return aperture

# Parameterizes the acoustic source field using a multi-scale U-Net architecture with sparse initialization.
class NeuralFieldUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def encoder_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True)
            )
        self.enc1 = encoder_block(1, 16)
        self.enc2 = encoder_block(16, 32)
        self.center = encoder_block(32, 64)
        self.dec2 = encoder_block(64 + 32, 32)
        self.dec1 = encoder_block(32 + 16, 16)
        self.head = nn.Conv2d(16, 1, 1)
        nn.init.normal_(self.head.weight, std=0.001)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        c = self.center(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([F.interpolate(c, scale_factor=2, mode='bilinear'), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode='bilinear'), e1], 1))
        return self.head(d1)

# Orchestrates the physics-informed optimization routine for neural acoustic inversion under aperture constraints.
def run_solver():
    # Load and normalize observation data
    try:
        f = h5py.File(OBS_DATA_FILE, 'r'); obs_sino = np.array(f['Sino'])
    except:
        obs_sino = scipy.io.loadmat(OBS_DATA_FILE)['Sino']
    
    if obs_sino.shape == (DIM_SENSOR, DIM_TIME): obs_sino = obs_sino.T
    obs_tensor = torch.from_numpy(obs_sino).float().to(DEVICE)
    obs_tensor /= (obs_tensor.abs().max() + 1e-9)

    # Apply geometric aperture masking
    aperture_mask = generate_aperture_function()
    obs_tensor = obs_tensor * aperture_mask

    # Initialize implicit physics engine
    physics = AcousticPhysicsEngine(STATE_FILE, DIM_GRID, DIM_SENSOR, DIM_TIME).to(DEVICE)
    
    # Gain calibration and initialization via adjoint projection
    with torch.no_grad():
        field_init = F.relu(physics.adjoint_projection(obs_tensor))
        field_init /= (field_init.max() + 1e-9)
        field_input = field_init.unsqueeze(0).unsqueeze(0)
        
        sim_proj = physics(field_init)
        valid_idx = aperture_mask > 0.5
        gain = (obs_tensor[valid_idx].abs().mean() / (sim_proj[valid_idx].abs().mean() + 1e-9)).item() if valid_idx.any() else 1.0
        physics.system_gain = gain

    # Neural optimization loop
    model = NeuralFieldUNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    pbar = tqdm(range(1, 201))
    
    for step in pbar:
        optimizer.zero_grad()
        
        residual = model(field_input)
        rec_field = F.relu(field_input + residual).squeeze()
        fwd_proj = physics(rec_field)
        
        loss_data = torch.mean(((fwd_proj - obs_tensor) * aperture_mask) ** 2)
        loss_reg  = 0.001 * torch.mean(torch.abs(residual))
        
        (loss_data + loss_reg).backward()
        optimizer.step()
        
        pbar.set_postfix({'loss': f"{(loss_data+loss_reg).item():.6f}"})
        
        if step % 50 == 0 or step == 1:
            visualize_state(step, field_init, residual, rec_field)


# Renders diagnostic visualizations: Initial Guess (LBP), Neural Residual, and Final Reconstruction.
def visualize_state(step, x0, res, x_rec):
    with torch.no_grad():
        plt.figure(figsize=(12, 4))
        # 1. LBP (Initial Guess)
        plt.subplot(131); plt.imshow(x0.cpu(), cmap='gray'); plt.title("Initial Guess (LBP)")
        # 2. Diff (Neural Residual)
        plt.subplot(132); plt.imshow(res.squeeze().cpu(), cmap='seismic'); plt.title("Neural Residual")
        # 3. Final Output (Reconstruction)
        plt.subplot(133); plt.imshow(x_rec.cpu(), cmap='gray'); plt.title(f"Reconstruction (Step {step})")
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"diag_step{step:04d}.png"))
        plt.close()

if __name__ == "__main__":
    run_solver()
import numpy as np
import biobeam as bb
import time
from types import SimpleNamespace
import sys
import os
sys.path.append(os.path.abspath("C:/Users/Goerlitz/BrilloGram/"))
import brilloFunctions as bf
from scipy.ndimage import rotate
import tifffile as tiff
import matplotlib.pyplot as plt
from numpy.fft import fftn, fftshift, fftfreq
from scipy.ndimage import center_of_mass
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F
import math
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"  # deaktiviert Ausgabe komplett
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# %% set parameters
s = time.time()

optExc = SimpleNamespace()
optExc.Nx = 768
optExc.Ny = optExc.Nx
optExc.Nz = optExc.Nx
optExc.dx = 0.05
optExc.dy = optExc.dx
optExc.dz = optExc.dx
optExc.NA = 0.2
optExc.n0 = 1.33
optExc.lam = 0.532

optDet = SimpleNamespace()
optDet.Nx = optExc.Nx
optDet.Ny = optDet.Nx
optDet.Nz = optDet.Nx
optDet.dx = optExc.dx
optDet.dy = optDet.dx
optDet.dz = optDet.dx
optDet.NA = 0.8
optDet.n0 = optExc.n0
optDet.lam = 0.580
optDet.angle = 90

# check if pixelsize smaller than Nyquist 
dxExc = optExc.lam/2/optExc.NA
dxDet = optDet.lam/2/optDet.NA
psfE, psfDgen, theta, phi, bins, angles_rad, sexy, kxy = bf.prepPara(optExc, optDet)

#%%

def rotate_3d_volume(vol, angle_deg, device='cuda'):
    """
    Rotate a 3D volume around its center along the Z-axis using PyTorch.

    Parameters:
        vol (np.ndarray or torch.Tensor): 3D volume (D, H, W)
        angle_deg (float): rotation angle in degrees
        device (str): 'cuda' or 'cpu'

    Returns:
        np.ndarray: rotated 3D volume
    """
    # Convert to PyTorch tensor on specified device
    if isinstance(vol, np.ndarray):
        vol = torch.tensor(vol, dtype=torch.float32, device=device)
    else:
        vol = vol.to(dtype=torch.float32, device=device)

    # Add batch & channel dimensions
    vol = vol.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

    # Convert angle to radians
    theta = math.radians(angle_deg)
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)

    # Rotation matrix around Z-axis
    rotation_matrix = torch.tensor([[
        [ cos_theta, -sin_theta, 0, 0],
        [ sin_theta,  cos_theta, 0, 0],
        [        0,         0, 1, 0]
    ]], dtype=torch.float32, device=device)

    # Compute center of volume
    D, H, W = vol.shape[-3:]
    center = torch.tensor([W/2, H/2, D/2], device=device)

    # Adjust translation to rotate around volume center
    R = rotation_matrix[0, :, :3]
    translation = center - torch.matmul(R, center)
    rotation_matrix[0, :, 3] = translation

    # Create sampling grid and rotate
    grid = F.affine_grid(rotation_matrix, vol.size(), align_corners=True)
    rotated_vol = F.grid_sample(vol, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    rotated_vol = rotated_vol.squeeze()  # remove batch & channel

    # Convert back to NumPy
    return rotated_vol.cpu().numpy()

s1 = time.time()
psfDgen_torch = torch.tensor(psfDgen, dtype=torch.complex64, device='cuda')
realV = torch.real(psfDgen_torch)
max_val = torch.max(realV).item()
imagV = torch.imag(psfDgen_torch)
realVrot = rotate_3d_volume(realV, optDet.angle, device='cuda')
imgVrot = rotate_3d_volume(imagV, optDet.angle, device='cuda')
rotated_complex = realVrot + 1j * imgVrot
e1 = time.time()
d1 = e1 - s1
print(f"Time for rotation in PyTorch: {d1:.3f} seconds")
#%%
#s2 = time.time()
#psfDscat = bf.rotPSF(psfDgen, optDet.angle)
#e2 = time.time()
#d2 = e2 - s2
#print(f"Time for rotation on cpu: {d2:.3f} seconds")
#%%
#bf.plot_max_projections(np.abs(psfDscat), voxel_size=(1.0, 1.0, 1.0), cmap='hot', title="stand rot")
bf.plot_max_projections(np.abs(rotated_complex), voxel_size=(1.0, 1.0, 1.0), cmap='hot', title="pythorch rot")
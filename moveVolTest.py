import numpy as np
import time
from types import SimpleNamespace
import os
import matplotlib.pyplot as plt
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"  # deaktiviert Ausgabe komplett
# %% set parameters
s = time.time()

optExc = SimpleNamespace()
optExc.Nx = 512
optExc.Ny = optExc.Nx
optExc.Nz = optExc.Nx
optExc.dx = 0.075
optExc.dy = optExc.dx
optExc.dz = optExc.dx
optExc.NA = 0.4
optExc.n0 = 1.33
optExc.lam = 0.532

optDet = SimpleNamespace()
optDet.Nx = optExc.Nx
optDet.Ny = optDet.Nx
optDet.Nz = optDet.Nx
optDet.dx = optExc.dx
optDet.dy = optDet.dx
optDet.dz = optDet.dx
optDet.NA = 0.4
optDet.n0 = optExc.n0
optDet.lam = 0.580
optDet.angle = 90

# check if pixelsize smaller than Nyquist 
dxExc = optExc.lam/2/optExc.NA
dxDet = optDet.lam/2/optDet.NA
# optExc.dx = dxDet/2
# optExc.dy = optExc.dx
# optExc.dz = optExc.dx
# optDet.dx = optExc.dx
# optDet.dy = optExc.dx
# optDet.dz = optExc.dx
if np.min([dxDet, dxExc]) <= optExc.dx:
    print('..........................Warning ...................................... -> K-Space not Nyquist sampled for choosen NA and dx.')
print("... setup optExc and optDet")

#%%
mainPath = "C:\\Users\\Goerlitz\\Documents\\temp\\20250729_scaScatt_512\\"

# scatPath = 'C:\\Users\\Goerlitz\\Documents\\temp\\Tabea_mouseembryo_001_512.tif'
# scatVol1 = tiff.imread(scatPath)/10000
# scatVol = np.transpose(scatVol1, (1, 0, 2))
# sf = 1
# scale_factors = (sf, sf, sf)
# scatVol = zoom(scatVol1, scale_factors, order=1)  # order=1 = linear interpolation


def create_sphere_with_gaussian_noise(shape=(optExc.Nx*2, optExc.Nx*2, optExc.Nx*2),
                                      n_background=1.33,
                                      n_sphere=1.43,
                                      radius=256,
                                      noise_std=0.01,
                                      center=None):

    Z, Y, X = shape
    if center is None:
        center = (Z // 2, Y // 2, X // 2)

    # Background
    volume = np.full(shape, n_background, dtype=np.float32)

    # Sphere
    z, y, x = np.ogrid[:Z, :Y, :X]
    dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    volume[dist <= radius] = n_sphere

    # Add Gaussian noise
    volume += np.random.normal(0, noise_std, size=shape)

    return volume

# Example usage
vol = create_sphere_with_gaussian_noise(shape=(optExc.Nx, optExc.Nx, optExc.Nx),
                                        n_background=1.33,
                                        n_sphere=1.43,
                                        radius=64,
                                        noise_std=0.01)
print("... create vol")
padded_scatVol = np.random.normal(1.33, 0.01, size=(vol.shape[0]*3, vol.shape[1]*3, vol.shape[2]*3))
#padded_scatVol = np.zeros((vol.shape[0]*3, vol.shape[1]*3, vol.shape[2]*3), dtype=np.float32)

# Shapes
sz, sy, sx = vol.shape
Z, Y, X = padded_scatVol.shape

# Start-Indices (zentriert)
start_z = (Z - sz) // 2
start_y = (Y - sy) // 2
start_x = (X - sx) // 2

# End-Indices
end_z = start_z + sz
end_y = start_y + sy
end_x = start_x + sx

# Insert
padded_scatVol[start_z:end_z, start_y:end_y, start_x:end_x] = vol
print("... pad vol")


#%%
xsteps = 5
xrange = 128
xstepSize = round(xrange/(xsteps - 1))
xrange = 0 if xsteps == 1 else xrange
xstepSize = 0 if xsteps == 1 else round(xrange / (xsteps - 1))

ysteps = xsteps
yrange = xrange
yrange = 0 if ysteps == 1 else yrange
ystepSize = 0 if ysteps == 1 else round(yrange / (ysteps - 1))

zsteps = 1
zrange = xrange
zrange = 0 if zsteps == 1 else zrange
zstepSize = 0 if zsteps == 1 else round(zrange / (zsteps - 1))

for i in range(xsteps):
    start_x = round(sx - xrange/2) + i * xstepSize
    end_x   = start_x + sx
    for j in range(ysteps):
        start_y = round(sy - yrange/2) + j * ystepSize 
        end_y   = start_y + sy
        for w in range(zsteps):
            start_z = round(sz - zrange/2) + w * zstepSize
            end_z   = start_z + sz
            shifted_vol = padded_scatVol[start_z:end_z, start_y:end_y, start_x:end_x]
        
            print("xSteps: ", i, "xstart: ", start_x, "ySteps: ", j, "ystart: ", start_y, "zSteps: ", w, "zstart: ", start_z)
            
            fig, axes = plt.subplots(1, 3, figsize=(12, 8))
            axes[0].imshow(np.max(shifted_vol, axis=0), cmap="gray")
            axes[0].set_title("max-proj (z-axis)")
            axes[0].axis("off")
            axes[1].imshow(np.max(shifted_vol, axis=1), cmap="gray")
            axes[1].set_title("max-proj (y-axis)")
            axes[1].axis("off")
            axes[2].imshow(np.max(shifted_vol, axis=2), cmap="gray")
            axes[2].set_title("max-proj (x-axis)")
            axes[2].axis("off")
            plt.tight_layout()
            plt.show()
    

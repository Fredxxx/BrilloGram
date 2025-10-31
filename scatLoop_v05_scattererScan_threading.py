import numpy as np
import sys
sys.path.append(r'C:\Users\Fred\Documents\GitHub\BrilloGram')
import configparser
configparser.SafeConfigParser = configparser.ConfigParser
import time
from types import SimpleNamespace
import os
import brilloFunctions_v05 as bf
import tifffile as tiff
from concurrent.futures import ThreadPoolExecutor, as_completed
import arrayfire as af

af.device_gc()
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"  # deaktiviert Ausgabe komplett
# %% set parameters
s = time.time()

optExc = SimpleNamespace()
optExc.Nx = 64
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

mainPath = "C:\\Fred\\temp\\test\\"

# check if pixelsize smaller than Nyquist 
dxExc = optExc.lam/2/optExc.NA
dxDet = optDet.lam/2/optDet.NA
if np.min([dxDet, dxExc]) <= optExc.dx:
    print('..........................Warning ...................................... -> K-Space not Nyquist sampled for choosen NA and dx.')
#%% prepare propagation volume


# scatPath = 'C:\\Users\\Goerlitz\\Documents\\temp\\Tabea_mouseembryo_001_512.tif'
# scatVol1 = tiff.imread(scatPath)/10000
# scatVol = np.transpose(scatVol1, (1, 0, 2))
# sf = 0.200/optExc.dx
# scale_factors = (sf, sf, sf)
# scatVol = zoom(scatVol1, scale_factors, order=1)  # order=1 = linear interpolation

vol = bf.create_sphere_with_gaussian_noise(shape=(optExc.Nx, optExc.Nx, optExc.Nx),
                                        n_background=1.33,
                                        n_sphere=1.43,
                                        radius=256,
                                        noise_std=0.01)


padded_scatVol = bf.genPaddArray(optExc.Nx, optExc.Nx, optExc.Nx, vol)
sz, sy, sx = vol.shape
print("... propagation volume loaded/generated")
del vol

# %% pepare vols histo parameters and scatter propagator
psfE, psfD, theta, phi, bins, angles_rad, sexy, kxy = bf.prepPara(optExc, optDet)
print("... prepared PSFs")

#%% define steps

xsteps = 3
xrange = round(optExc.Nx * 0.75) #512#
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


#%% threading
dTheta = np.zeros((xsteps, ysteps, zsteps))
dPhi = np.zeros((xsteps, ysteps, zsteps))
sTheta = np.zeros((xsteps, ysteps, zsteps))
sPhi = np.zeros((xsteps, ysteps, zsteps))

# Use ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=1) as executor:  # adjust workers to CPU cores
    futures = []
    for i in range(xsteps):
        start_x = round(sx - xrange/2) + i * xstepSize
        end_x   = start_x + sx
        for j in range(ysteps):
            start_y = round(sy - yrange/2) + j * ystepSize 
            end_y   = start_y + sy
            for w in range(zsteps):
                start_z = round(sz - zrange/2) + w * zstepSize
                end_z   = start_z + sz
                coo = [start_x, end_x, start_y, end_y, start_z, end_z]
                futures.append(executor.submit(bf.process_shift, coo, 
                                               padded_scatVol, psfE, psfD, 
                                               optExc, optDet, mainPath,
                                               theta, phi, i, j, w))
                
    total_tasks = len(futures)
    completed = 0
    for future in as_completed(futures):
        res = SimpleNamespace()
        res = future.result()
        dTheta[res.x, res.y, res.z] = res.thetaCOM
        dPhi[res.x, res.y, res.z] = res.phiCOM
        sTheta[res.x, res.y, res.z] = res.sigX
        sPhi[res.x, res.y, res.z] = res.sigY
        completed += 1
        print(f"[{completed}/{total_tasks}] Completed i={i}, j={j}, w={w}")
        
#%% save phi and theta
scatDim = optExc.dx

xRes = 1 / (scatDim / 10000)
yRes = 1 / (scatDim / 10000)

name = 'dTheta'
bf.saveDist(mainPath, 'comTheta', dTheta[:,:,0], xRes, yRes, scatDim)
bf.saveDist(mainPath, 'comPhi', dPhi[:,:,0], xRes, yRes, scatDim)
bf.saveDist(mainPath, 'sigTheta', sTheta[:,:,0], xRes, yRes, scatDim)
bf.saveDist(mainPath, 'sigPhi', sPhi[:,:,0], xRes, yRes, scatDim)

# # Save
# filename2 = mainPath + 'dTheta.tiff'
# np.savetxt(mainPath + "dTheta.txt", thSave, fmt="%.5f", delimiter="\t")
# tiff.imwrite(
#     filename2,
#     dTheta.astype(np.float32),
#     imagej=True,
#     resolution=(x_resolution, y_resolution),
#     metadata={
#         'spacing': scatDim,    # Z voxel size
#         'unit': 'um'      # physical unit
#     }
#     )

# filename2 = mainPath + 'dPhi.tiff'
# np.savetxt(mainPath + "dPhi.txt", phSave, fmt="%.5f", delimiter="\t")
# tiff.imwrite(
#     filename2,
#     dPhi.astype(np.float32),
#     imagej=True,
#     resolution=(x_resolution, y_resolution),
#     metadata={
#         'spacing': scatDim,    # Z voxel size
#         'unit': 'um'      # physical unit
#     }
#     )

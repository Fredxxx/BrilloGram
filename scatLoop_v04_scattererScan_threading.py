import numpy as np
import sys
sys.path.append(r'C:\Users\Fred\Documents\GitHub\BrilloGram')
import configparser
configparser.SafeConfigParser = configparser.ConfigParser
import biobeam as bb
import time
from types import SimpleNamespace
import os
import brilloFunctions as bf
import tifffile as tiff
#import matplotlib.pyplot as plt
#from numpy.fft import fftn, fftshift, fftfreq
from scipy.ndimage import center_of_mass
from scipy.ndimage import zoom
from concurrent.futures import ThreadPoolExecutor, as_completed
#import contextlib
#import os
from datetime import datetime
#import sys
import arrayfire as af
import gc
af.device_gc()
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"  # deaktiviert Ausgabe komplett
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
# optExc.dx = dxDet/2
# optExc.dy = optExc.dx
# optExc.dz = optExc.dx
# optDet.dx = optExc.dx
# optDet.dy = optExc.dx
# optDet.dz = optExc.dx
if np.min([dxDet, dxExc]) <= optExc.dx:
    print('..........................Warning ...................................... -> K-Space not Nyquist sampled for choosen NA and dx.')
#%%
mainPath = "C:\\Users\\Goerlitz\\Documents\\temp\\20251006_scaScatt_768_Exc02LS_Det08CO_50nm_90deg_Tabea\\"

# scatPath = 'C:\\Users\\Goerlitz\\Documents\\temp\\Tabea_mouseembryo_001_512.tif'
# scatVol1 = tiff.imread(scatPath)/10000
# scatVol = np.transpose(scatVol1, (1, 0, 2))
# sf = 0.200/optExc.dx
# scale_factors = (sf, sf, sf)
# scatVol = zoom(scatVol1, scale_factors, order=1)  # order=1 = linear interpolation



# Example usage
vol = bf.create_sphere_with_gaussian_noise(shape=(optExc.Nx, optExc.Nx, optExc.Nx),
                                        n_background=1.33,
                                        n_sphere=1.43,
                                        radius=256,
                                        noise_std=0.01)

padded_scatVol = np.random.normal(1.33, 0.002, size=(optExc.Nx*3, optExc.Nx*3, optExc.Nx*3)).astype(np.float16)

# Shapes 
sz, sy, sx = vol.shape#scatVol.shape # xxx
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
padded_scatVol[start_z:end_z, start_y:end_y, start_x:end_x] = vol#scatVol # xxx
print("... propagation volume loaded/generated")
# del scatVol #xxx
del vol
# %% pepare vols histo parameters and scatter propagator
psfE, psfD, theta, phi, bins, angles_rad, sexy, kxy = bf.prepPara(optExc, optDet)
print("... prepared PSFs")

#%%

xsteps = 64
xrange = 512#round(optExc.Nx * 0.75)
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

dTheta = np.zeros((xsteps, ysteps, zsteps))
dPhi = np.zeros((xsteps, ysteps, zsteps))

cpu_cores = os.cpu_count()  # total logical cores

def process_shift(coo, sx, sy, sz, padded_scatVol, psfE, psfDgen, optExc, optDet, mainPath, theta, phi, i, j, w):
   print(f" working on i={i}, j={j}")
   current_time = datetime.now()

    # Print the current time
   print("Current time:", current_time.strftime("%H:%M:%S"))
   # Propagation
   t = padded_scatVol[coo[4]:coo[5], coo[2]:coo[3], coo[0]:coo[1]]
   t = bb.Bpm3d(dn=t, units = (optDet.dx,)*3, lam=optExc.lam/optExc.n0)
   psfEscat = t.propagate(u0 = psfE[0,:,:])
   # #propVol = np.asarray(shifted_vol, dtype=np.complex64)
   del psfE 
   gc.collect()
   
   
   t = padded_scatVol[coo[4]:coo[5], coo[0]:coo[1], coo[2]:coo[3]]
   t = bb.Bpm3d(dn=t, units = (optDet.dx,)*3, lam=optDet.lam/optExc.n0)
   psfDscat = t.propagate(u0 = psfDgen[0,:,:])
   psfDscat = bf.rotPSF(psfDscat, optDet.angle)
   del psfDgen
   del t
   gc.collect()
   
   psfS = psfEscat * psfDscat
   #psfS = np.abs(fftshift(fftn(psfS)))**2
   psfS = bf.fftgpuPS(psfS)
   comSinc = tuple(np.round(center_of_mass(psfS)).astype(int))

   
   try:
       ts1inc = theta[comSinc]
       ps1inc = phi[comSinc]
   except IndexError as e:
       print("❌ IndexError:", e)
       print("comSinc =", comSinc)
       print("theta.shape =", theta.shape, "phi.shape =", phi.shape, "psfS.shape =", psfS.shape, "psfEscat.shape =", psfEscat.shape)
       if hasattr(comSinc, "max"):
           print("comSinc.min =", comSinc.min(), "comSinc.max =", comSinc.max())
       raise
       
   del psfS
   del psfDscat
   del psfEscat
   gc.collect()
    # Return results
   return i, j, w, ts1inc, ps1inc

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
                futures.append(executor.submit(process_shift, coo, sx, sy, sz, 
                                               padded_scatVol, 
                                               psfE, psfD, 
                                               optExc, optDet, mainPath,
                                               theta, phi, i, j, w))
                
    total_tasks = len(futures)
    completed = 0
    for future in as_completed(futures):
        i, j, w, ts1inc, ps1inc = future.result()
        dTheta[i, j, w] = ts1inc
        dPhi[i, j, w] = ps1inc
        completed += 1
        print(f"[{completed}/{total_tasks}] Completed i={i}, j={j}, w={w}")
        

scatDim = optExc.dx

x_resolution = 1 / (scatDim / 10000)
y_resolution = 1 / (scatDim / 10000)
thSave = dTheta[:,:,0]
phSave = dPhi[:,:,0]
# Save
filename2 = mainPath + 'dTheta.tiff'
np.savetxt(mainPath + "dTheta.txt", thSave, fmt="%.5f", delimiter="\t")
tiff.imwrite(
    filename2,
    dTheta.astype(np.float32),
    imagej=True,
    resolution=(x_resolution, y_resolution),
    metadata={
        'spacing': scatDim,    # Z voxel size
        'unit': 'um'      # physical unit
    }
    )

filename2 = mainPath + 'dPhi.tiff'
np.savetxt(mainPath + "dPhi.txt", phSave, fmt="%.5f", delimiter="\t")
tiff.imwrite(
    filename2,
    dPhi.astype(np.float32),
    imagej=True,
    resolution=(x_resolution, y_resolution),
    metadata={
        'spacing': scatDim,    # Z voxel size
        'unit': 'um'      # physical unit
    }
    )

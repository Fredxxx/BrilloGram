# external 
import numpy as np

#internal 
import sys
import configparser
configparser.SafeConfigParser = configparser.ConfigParser
from types import SimpleNamespace
import os
from concurrent.futures import ThreadPoolExecutor

# %% set parameters
#sys.path.append(r'/g/prevedel/members/Goerlitz/projectsHPC/brillo')
sys.path.append(r'C:/Users/Fred/Documents/GitHub/BrilloGram')
import brilloFunctions_v08 as bf
#mainPath = "/g/prevedel/members/Goerlitz/projectsHPC/brillo/results/"
mainPath = "C:/Fred/temp/"
name = "90deg_08NA_gauss_32_00deg_3"
#mainPath = "/scratch/goerlitz/brilloCopy/"
path = os.path.join(mainPath, name)
os.makedirs(path, exist_ok=True)
os.makedirs(os.path.join(path, "sys"), exist_ok=True)
os.makedirs(os.path.join(path, "exc"), exist_ok=True)
os.makedirs(os.path.join(path, "det"), exist_ok=True)

optExc = SimpleNamespace()
optExc.Nx = 256 #512 #256
optExc.Ny = optExc.Nx
optExc.Nz = optExc.Nx
optExc.dx = 0.05
optExc.dy = optExc.dx
optExc.dz = optExc.dx
optExc.NA = 0.8
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

optGen = SimpleNamespace()
optGen.Vs = 1490 * 10**6 #m/s 
optGen.BSshiftA = 5 # GHz
optGen.BSwidthA = 0.29 # GHz
optGen.BSspecStart = 2 # GHz
optGen.BSspecEnd  = 7 # GHz
optGen.BSspecRes = 0.01 # GHz

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
                                        n_sphere=1.35,
                                        radius= 64,#32,#64, #xxx256,
                                        noise_std=0.001)


padded_scatVol = bf.genPaddArray(optExc.Nx, optExc.Nx, optExc.Nx, vol)
sz, sy, sx = vol.shape
print("... propagation volume loaded/generated")
del vol

# %% pepare vols histo parameters and scatter propagator
psfE, psfD, theta, phi, bins, angles_rad, sexy, kxy = bf.prepPara3(optExc, optDet)
print("... prepared PSFs")

#%% define steps

xsteps = 32
xrange = 320#384
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
comTheta = np.zeros((xsteps, ysteps, zsteps))
comPhi = np.zeros((xsteps, ysteps, zsteps))
stdTheta = np.zeros((xsteps, ysteps, zsteps))
stdPhi = np.zeros((xsteps, ysteps, zsteps))
meanTheta = np.zeros((xsteps, ysteps, zsteps))
meanPhi = np.zeros((xsteps, ysteps, zsteps))
# %%
coordinate_sets = []
idx = 0
idxMax = xsteps*ysteps*zsteps
for i in range(xsteps):
    start_x = round(sx - xrange/2) + i * xstepSize
    end_x = start_x + sx
    for j in range(ysteps):
        start_y = round(sy - yrange/2) + j * ystepSize
        end_y = start_y + sy
        for w in range(zsteps):
            idx += 1
            start_z = round(sz - zrange/2) + w * zstepSize
            end_z = start_z + sz
            # Group all arguments for each task
            coo = [start_x, end_x, start_y, end_y, start_z, end_z]
            coordinate_sets.append((coo, padded_scatVol, psfE, psfD, optExc, optDet, optGen, path, theta, phi, i, j, w, idx, idxMax))

# Execute in parallel
with ThreadPoolExecutor(max_workers=20) as executor:
    # Use a helper to unpack arguments since map takes one iterable
    executor.map(lambda p: bf.process_shift2(*p), coordinate_sets)

        
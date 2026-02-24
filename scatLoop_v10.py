# external 
import numpy as np
import tifffile as tiff
from scipy.ndimage import zoom
import biobeam as bb

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
import brilloFunctions_v10 as bf
#mainPath = "/g/prevedel/members/Goerlitz/projectsHPC/brillo/results/"

mainPath = "C:/Fred/temp/"
name = "ttt"#"90deg_tabea_32x32_01"
#mainPath = "/scratch/goerlitz/brilloCopy/"

path = os.path.join(mainPath, name)
os.makedirs(path, exist_ok=True)
os.makedirs(os.path.join(path, "sys"), exist_ok=True)
os.makedirs(os.path.join(path, "exc"), exist_ok=True)
os.makedirs(os.path.join(path, "det"), exist_ok=True)

optExc = SimpleNamespace()
optExc.Nx = 512#768 #512 #256-good
optExc.Ny = optExc.Nx
optExc.Nz = optExc.Nx
optExc.dx = 0.04 #0.5
optExc.dy = optExc.dx
optExc.dz = optExc.dx
optExc.NA = 0.05 #0.8
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
optDet.angle = 00

optGen = SimpleNamespace()
optGen.Vs = 1490 * 10**6 #m/s 
optGen.BSshiftA = 5 # GHz
optGen.BSwidthA = 0.29 # GHz
optGen.BSspecStart = 2 # GHz
optGen.BSspecEnd  = 7 # GHz
optGen.BSspecRes = 0.01 # GHz

# # check if pixelsize smaller than Nyquist 
# dxExc = optExc.lam/2/optExc.NA
# dxDet = optDet.lam/2/optDet.NA
# if np.min([dxDet, dxExc]) <= optExc.dx:
#     print('..........................Warning ...................................... -> K-Space not Nyquist sampled for choosen NA and dx.')
#%% prepare propagation volume


# scatPath = 'C:\\Users\\Goerlitz\\Documents\\temp\\Tabea_mouseembryo_001_512.tif'
# scatVol1 = tiff.imread(scatPath)/10000
# scatVol = np.transpose(scatVol1, (1, 0, 2))
# sf = 0.200/optExc.dx
# scale_factors = (sf, sf, sf)
# scatVol = zoom(scatVol1, scale_factors, order=1)  # order=1 = linear interpolation

# vol = bf.create_sphere_with_gaussian_noise(shape=(optExc.Nx, optExc.Nx, optExc.Nx),
#                                         n_background=1.33,
#                                         n_sphere=1.35,
#                                         radius= 64,#32,#64, #256
#                                         noise_std=0.001)


# padded_scatVol = bf.genPaddArray(optExc.Nx, optExc.Nx, optExc.Nx, vol)



scatPath = 'C:\\Fred\\temp\\Tabea_mouseembryo_001.tif' #512x512x136, 0.23umx0.23umx0.46um
scatVol = tiff.imread(scatPath)/10000
scatVol = np.swapaxes(scatVol, 0, 2)
sf = 2.3/0.6#2.3/2.5 #2.3
#scale_factors = (sf, sf, 4*sf)
scale_factors = (sf, sf, 4*sf)
scatVol = zoom(scatVol, scale_factors, order=1)  # order=1 = linear interpolation
padded_scatVol = bf.genPaddArray2(optExc.Nx, optExc.Nx, optExc.Nx, scatVol)
bf.plot_max_projections(padded_scatVol, voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="padded_scatVol")
del scatVol


# %% pepare vols histo parameters and scatter propagator
psfE, psfD, theta, phi = bf.prepPara2(optExc, optDet)
print("... prepared PSFs")
#% plot   
# Excitation: shift volume, init propagator and propagate
# center = np.array(padded_scatVol.shape) // 2
# start = center - 768 // 2
# end = center + 768 // 2
# t = padded_scatVol[
#     start[0]:end[0], 
#     start[1]:end[1], 
#     start[2]:end[2]
# ]
# t=padded_scatVol

 #%%
# center = np.array(padded_scatVol.shape) // 2
# start = center - 256 // 2
# end = center + 256 // 2
# t = padded_scatVol[
#     start[0]:end[0], 
#     start[1]:end[1], 
#     start[2]:end[2]
# ]
# te = bb.Bpm3d(dn=t, units = (optDet.dx,)*3, lam=optExc.lam/optExc.n0)
# psfEscat = te.propagate(u0 = psfE[0,:,:])

# bf.plot_max_projections(np.abs(psfE), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psfE")
# bf.plot_max_projections(t, voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="te")
# bf.plot_max_projections(np.abs(psfEscat), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psfEscat")

# # td = bb.Bpm3d(dn=t, units = (optDet.dx,)*3, lam=optDet.lam/optExc.n0)
# # psfDscat = td.propagate(u0 = psfD[0,:,:])

# # #psfDscat = bf.rotPSF(psfDscat, 90)
# # #psfEscat = bf.rotPSF(psfEscat, 90)

# # #psfSys = psfEscat*psfDscat
# # #psfSys = psfE*psfD
#bf.plot_max_projections(padded_scatVol, voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="padded_scatVol")
# # bf.plot_max_projections2(np.abs(psfE), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psf excitation", space="real")
# # bf.plot_max_projections2(np.abs(psfEscat), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psf excitation scat", space="real")
# # bf.plot_max_projections2(np.abs(psfD), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psf detection", space="real")
# # bf.plot_max_projections2(np.abs(psfDscat), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psf detection scat", space="real")




# #bf.plot_max_projections2(np.abs(psfSys), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psf system", space="real")
# # bf.plot_max_projections2(bf.fftgpuPS(psfEscat), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="ps excitation", space="fft")
# # bf.plot_max_projections2(bf.fftgpuPS(psfDscat), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="ps detection", space="fft")
# # bf.plot_max_projections2(bf.fftgpuPS(psfSys), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="ps system", space="fft")

# # # #%%
# # resS = bf.calcMain(bf.fftgpuPS(psfSys), 'sys', theta, phi, optDet, optGen, 0, [0,0,0,0,0,0], 0, 0, 0, path)
# # bf.saveHisto(resS, path, optDet)
# %%save tiffs

# bf.saveDist(path, "scatVol", padded_scatVol, optExc.dx, optExc.dx, optExc.dx)
# bf.saveDist(path, "psfE", psfE, optExc.dx, optExc.dx, optExc.dx)
# bf.saveDist(path, "psfD", psfD, optExc.dx, optExc.dx, optExc.dx)
# bf.saveDist(path, "psfSys", psfSys, optExc.dx, optExc.dx, optExc.dx)

# bf.saveDist(path, "psE", bf.fftgpuPS(psfE), optExc.dx, optExc.dx, optExc.dx)
# bf.saveDist(path, "psD", bf.fftgpuPS(psfD), optExc.dx, optExc.dx, optExc.dx)
# bf.saveDist(path, "psSys", bf.fftgpuPS(psfSys), optExc.dx, optExc.dx, optExc.dx)
# %%

sz, sy, sx = psfE.shape
print("... propagation volume loaded/generated")


#%% define steps

xsteps = 2
xrange = 256#320
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
            coo = [start_z, end_z, start_y, end_y, start_x, end_x]
            coordinate_sets.append((coo, padded_scatVol, psfE, psfD, optExc, optDet, optGen, path, theta, phi, i, j, w, idx, idxMax))

# Execute in parallel
with ThreadPoolExecutor(max_workers=1) as executor:
    # Use a helper to unpack arguments since map takes one iterable
    executor.map(lambda p: bf.process_shift2(*p), coordinate_sets)

        
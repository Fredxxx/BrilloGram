import numpy as np
import time
from types import SimpleNamespace
import os
import brilloFunctions as bf
import biobeam as bb
import matplotlib.pyplot as plt
from skimage.morphology import ball
from scipy.ndimage import zoom
from numpy.fft import fftn, fftshift, fftfreq
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"  # deaktiviert Ausgabe komplett
# %% set parameters
s = time.time()

optExc = SimpleNamespace()
optExc.Nx = 512
optExc.Ny = optExc.Nx
optExc.Nz = optExc.Nx
optExc.dx = 0.05
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

if np.min([dxDet, dxExc]) <= optExc.dx:
    print('..........................Warning ...................................... -> K-Space not Nyquist sampled for choosen NA and dx.')

anglePara = SimpleNamespace()
anglePara.start = 0 #deg
anglePara.end = 90 #deg
anglePara.res = 5 #deg
anglePara.numSteps = round((anglePara.end-anglePara.start)/anglePara.res)

mainPath = "C:\\Users\\Goerlitz\\Documents\\temp\\test\\"

#%% gen psfs
psfE, psfDgen, theta, phi, bins, angles_rad, sexy, kxy = bf.prepPara(optExc, optDet)

#%% create sphere
def create_sphere_with_gaussian_noise(shape=(optExc.Nx*2, optExc.Nx*2, optExc.Nx*2),
                                      n_background = 1.33,
                                      n_sphere = 1.33,
                                      radius= 16,
                                      noise_std=0.000,
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
vol = create_sphere_with_gaussian_noise(shape=(optDet.Nx, optDet.Nx, optDet.Nx),
                                        radius=64,
                                        n_background=1.33,
                                        n_sphere=1.33,
                                        noise_std=0.00)

# %% propagator

# Setup propagator excitation
prop = bb.Bpm3d(dn=vol, units = (optDet.dx,)*3, lam=optExc.lam/optExc.n0)
propVol = np.asarray(vol, dtype=np.complex64)

# setup propagator detection, after rotation
scatVolrot = bf.rotPSF(vol, -optDet.angle)
propD = bb.Bpm3d(dn=scatVolrot, units = (optDet.dx,)*3, lam=optDet.lam/optExc.n0)
propVolrot = np.asarray(scatVolrot, dtype=np.complex64)


# prop excitation 
psfEscat = prop.propagate(u0 = psfE[0,:,:])
#pro detection and rotate
psfDscatrot = propD.propagate(u0 = psfDgen[0,:,:])
psfD = bf.rotPSF(psfDgen, optDet.angle)
psfDscat = bf.rotPSF(psfDscatrot, optDet.angle)

#Efield = prop.propagate(propVol, p_source="gauss")
voxS = optExc.dx
voxSfft = 1/(optExc.Nx * optExc.dx)
mainPath = "C:\\Users\\Goerlitz\\Documents\\temp\\20250729_scaScatt_512\\"
filename = "tt"
path = mainPath


bf.plot_max_projections(np.abs(psfD * psfE)**2, voxel_size=(voxS, voxS, voxS), title="PSF system - Max Projections")
bf.plot_max_projections(np.abs(psfDscat *psfEscat)**2, voxel_size=(voxS, voxS, voxS), title="PSF system scat- Max Projections")

bf.plot_max_projections(np.abs(psfE)**2, voxel_size=(voxS, voxS, voxS), title="PSF excitation - Max Projections")
bf.plot_max_projections(np.abs(psfEscat)**2, voxel_size=(voxS, voxS, voxS), title="PSF excitation scat- Max Projections")

bf.plot_max_projections(np.abs(psfD)**2, voxel_size=(voxS, voxS, voxS), title="PSF detection - Max Projections")
bf.plot_max_projections(np.abs(psfDscat)**2, voxel_size=(voxS, voxS, voxS), title="PSF detection scat- Max Projections")


psSscat= abs(fftshift(fftn(psfDscat *psfEscat)))**2
psS= abs(fftshift(fftn(psfD *psfE)))**2
bf.plot_max_projections(psS, voxel_size=(voxSfft, voxSfft, voxSfft), title="PS system - Max Projections")
bf.plot_max_projections(psSscat, voxel_size=(voxSfft, voxSfft, voxSfft), title="PS system scat- Max Projections")

psE= abs(fftshift(fftn(psfE)))**2
bf.plot_max_projections(psE, voxel_size=(voxSfft, voxSfft, voxSfft), title="PS excitation - Max Projections")
psEscat= abs(fftshift(fftn(psfEscat)))**2
bf.plot_max_projections(psEscat, voxel_size=(voxSfft, voxSfft, voxSfft), title="PS excitation scat - Max Projections")



psD= abs(fftshift(fftn(psfD)))**2
bf.plot_max_projections(psD, voxel_size=(voxSfft, voxSfft, voxSfft), title="PS detection - Max Projections")
psDscat= abs(fftshift(fftn(psfDscat)))**2
scale = (1.33, 1.33, 1.33)
psDscat = zoom(psDscat, zoom=scale, order=3)
bf.plot_max_projections(psDscat, voxel_size=(voxSfft, voxSfft, voxSfft), title="PS detection scat - Max Projections")

com_tSinc_2D, com_pSinc_2D = bf.calc2Dhisto(theta, phi, psS, filename, "systemIncoh", path)
com_tSinc_2D1, com_pSinc_2D1 = bf.calc2Dhisto(theta, phi, psSscat, filename, "systemIncoh", path)
print(com_tSinc_2D, com_pSinc_2D)
print(com_tSinc_2D1, com_pSinc_2D1)


# #%%
# def plot_center_slices(vol, title="Volume"):
#     zc, yc, xc = np.array(vol.shape) // 2  # Mittelpunkte

#     slices = [
#         (np.abs(vol[zc, :, :]), "XY @ Zcenter"),   # XY
#         (np.abs(vol[:, yc, :]), "XZ @ Ycenter"),   # XZ
#         (np.abs(vol[:, :, xc]), "YZ @ Xcenter"),   # YZ
#     ]

#     fig, axes = plt.subplots(1, 3, figsize=(12, 4))
#     for ax, (sl, sl_title) in zip(axes, slices):
#         ax.imshow(sl, cmap="gray", origin="lower")
#         ax.set_title(sl_title)
#         ax.axis("off")

#     fig.suptitle(title)
#     plt.tight_layout()
#     plt.show()

# # Beispiele: beide Volumen anzeigen
# plot_center_slices(psfE, "psfE |abs|")
# plot_center_slices(psfEscat, "psfEscat |abs|")
# plot_center_slices(psfDgen, "psfDgen |abs|")
# plot_center_slices(psfDscatrot, "psfDscat |abs|")
# plot_center_slices(vol, "vol")

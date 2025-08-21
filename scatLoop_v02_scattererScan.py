import numpy as np
import biobeam as bb
import time
from types import SimpleNamespace
import os
import brilloFunctions as bf
import tifffile as tiff
import matplotlib.pyplot as plt
from numpy.fft import fftn, fftshift, fftfreq
from scipy.ndimage import center_of_mass
from scipy.ndimage import zoom
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"  # deaktiviert Ausgabe komplett
# %% set parameters
s = time.time()

optExc = SimpleNamespace()
optExc.Nx = 512
optExc.Ny = optExc.Nx
optExc.Nz = optExc.Nx
optExc.dx = 0.2
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
mainPath = "C:\\Users\\Goerlitz\\Documents\\temp\\20250729_scaScatt_512\\"

# scatPath = 'C:\\Users\\Goerlitz\\Documents\\temp\\Tabea_mouseembryo_001_512.tif'
# scatVol1 = tiff.imread(scatPath)/10000
# scatVol = np.transpose(scatVol1, (1, 0, 2))
# sf = 1
# scale_factors = (sf, sf, sf)
# scatVol = zoom(scatVol1, scale_factors, order=1)  # order=1 = linear interpolation


def create_sphere_with_gaussian_noise(shape=(optExc.Nx*2, optExc.Nx*2, optExc.Nx*2),
                                      n_background=1.33,
                                      n_sphere=1.34,
                                      radius=128,
                                      noise_std=0.001,
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
                                        n_sphere=1.44,
                                        radius=30,
                                        noise_std=0.01)
padded_scatVol = vol

Z, Y, X = padded_scatVol.shape

print("... propagation volume loaded/generated")
# %% pepare vols histo parameters and scatter propagator
psfE, psfDgen, theta, phi, bins, angles_rad, sexy, kxy = bf.prepPara(optExc, optDet)
print("... prepared PSFs")
#%%
scatDim = optExc.dx


xSize = 100/optDet.dx
ySize = xSize

xSteps = 1
ySteps = xSteps

xStepsSize = round(xSize/xSteps)
yStepsSize = xStepsSize


xstart = 256#round(X/2 -  xSize/2)
zstart = round(Z/2)

dTheta = np.zeros((xSteps, ySteps))
dPhi = np.zeros((xSteps, ySteps))

for i in range(xSteps):
    
    ystart = 256#round(Y/2 -  ySize/2)
    for j in range(ySteps):
        
        # shift volume
        
        if xSteps == 1 and ySteps == 1:
            shifted_vol = vol
        else:
            shifted_vol = padded_scatVol[
            xstart:xstart + X,
            zstart:zstart + Z,
            ystart:ystart + Y]
        print(xstart, ystart, zstart)
        filename = f"shiftVol_x{i:02d}_y{j:02d}"
        path = mainPath + f"shiftVol_x{i:02d}_y{j:02d}" 
        #os.makedirs(path)
        def make_unique_folder(path):
            base_path = path
            i = 1
            while os.path.exists(path):
                path = f"{base_path}_{i}"
                i += 1
            os.makedirs(path)
            return path

        # Beispiel
        path = make_unique_folder(path)
        
        path = path + "\\"
        
        # propagator

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
        
        # #%% propagator

        # # Setup propagator excitation
        # prop = bb.Bpm3d(dn=shifted_vol, units = (scatDim,)*3, lam=optExc.lam)
        # propVol = np.asarray(shifted_vol, dtype=np.complex64)

        # # setup propagator detection, after rotation
        # scatVolrot = bf.rotPSF(shifted_vol, -optDet.angle)
        # propD = bb.Bpm3d(dn=scatVolrot, units = (scatDim,)*3, lam=optDet.lam)
        # propVolrot = np.asarray(scatVolrot, dtype=np.complex64)
        
        # # %% propagate
        # # prop excitation 
        # psfEscat = prop.propagate(propVol, E_in=psfE[0,:,:])
        # #pro detection and rotate
        # psfDscatrot = propD.propagate(propVolrot, E_in=psfDgen[0,:,:])
        # psfD = bf.rotPSF(psfDgen, optDet.angle)
        # psfDscat = bf.rotPSF(psfDscatrot, optDet.angle)
        
        # calc power spectrum and center of mass
        # psSinc, com_tSinc_1D, com_pSinc_1D = bf.calcPScomInc(psfEscat, psfDscat, theta, phi)
        
        # calc system PSF and PS
        psfS = psfEscat * psfDscat
        otfSincoh = fftshift(fftn(psfS)) # coherent OTF
        psSinc = np.abs(otfSincoh) ** 2 # power spectrum
        comSinc = tuple(np.round(center_of_mass(psSinc)).astype(int))
        ts1inc = theta[comSinc]
        ps1inc = phi[comSinc]
        
        voxS = optExc.dx
        voxSfft = 1/(optExc.Nx * optExc.dx)
        
        bf.plot_max_projections(abs(psfS)**2, voxel_size=(voxS, voxS, voxS), title="PSF system - Max Projections")
        bf.plot_max_projections(abs(psfEscat)*2, voxel_size=(voxS, voxS, voxS), title="PSF excitation - Max Projections")
        bf.plot_max_projections(abs(psfDscat)**2, voxel_size=(voxS, voxS, voxS), title="PSF detection - Max Projections")
        bf.plot_max_projections(psSinc, voxel_size=(voxSfft, voxSfft, voxSfft), title="PS system - Max Projections")
        
        # gen and save 2D histo
        com_tSinc_2D, com_pSinc_2D = bf.calc2Dhisto(theta, phi, psSinc, filename, "systemIncoh", path)
        print(com_tSinc_2D, com_pSinc_2D)
        # save into array
        dTheta[i,j] = com_tSinc_2D
        dPhi[i,j] = com_pSinc_2D
        
        # shift 
        ystart = round(ystart + yStepsSize) 
        print("... ystep")
        print(ySteps)
    xstart = round(xstart + xStepsSize)
    print("... xstep")
    print(xSteps)

#%% check psfs and psS
psSs= abs(fftshift(fftn(psfD *psfE)))**2
bf.plot_max_projections(psSs, voxel_size=(voxSfft, voxSfft, voxSfft), title="PS system - Max Projections")
psE= abs(fftshift(fftn(psfE)))**2
bf.plot_max_projections(psE, voxel_size=(voxSfft, voxSfft, voxSfft), title="PS excitation - Max Projections")
psD= abs(fftshift(fftn(psfD)))**2
bf.plot_max_projections(psD, voxel_size=(voxSfft, voxSfft, voxSfft), title="PS detection - Max Projections")
psEscat= abs(fftshift(fftn(psfEscat)))**2
bf.plot_max_projections(psEscat, voxel_size=(voxSfft, voxSfft, voxSfft), title="PS excitation scat - Max Projections")
psDscat= abs(fftshift(fftn(psfDscat)))**2
bf.plot_max_projections(psDscat, voxel_size=(voxSfft, voxSfft, voxSfft), title="PS detection scat - Max Projections")

com_tSinc_2D1, com_pSinc_2D1 = bf.calc2Dhisto(theta, phi, psSs, filename, "systemIncoh", path)
print(com_tSinc_2D1, com_pSinc_2D1)
print(com_tSinc_2D, com_pSinc_2D)
#%%
x_resolution = 1 / (scatDim / 10000)
y_resolution = 1 / (scatDim / 10000)

# Save
filename2 = mainPath + 'dTheta.tiff'
np.savetxt(mainPath + "dTheta.txt", dTheta, fmt="%.5f", delimiter="\t")
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
np.savetxt(mainPath + "dPhi.txt", dPhi, fmt="%.5f", delimiter="\t")
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

#%%
# Example volumes (replace these with your actual arrays)
# psfE and propVol should be 3D NumPy arrays: (z, y, x)
p1 = abs(psfEscat.T)**2
p2 = abs(psfDscat.T)**2
p3 = abs(psfS.T)**2
dx = dy = dz = 0.23  # micrometers

def get_projections(volume):
    # Assuming volume shape is (Z, Y, X)
    mip_xy = np.max(volume, axis=0)  # Projection along Z
    mip_xz = np.max(volume, axis=1)  # Projection along Y
    mip_yz = np.rot90(np.max(volume, axis=2), k=3)  # Projection along X
    return mip_xy, mip_xz, mip_yz

psfE_xy, psfE_xz, psfE_yz = get_projections(p1)
propVol_xy, propVol_xz, propVol_yz = get_projections(p3)
psfD_xy, psfD_xz, psfD_yz = get_projections(p2)

# Axis scaling for correct pixel size
z_size, y_size, x_size = psfE.shape
extent_xy = [0, x_size*dx, 0, y_size*dy]
extent_xz = [0, x_size*dx, 0, z_size*dz]
extent_yz = [0, y_size*dy, 0, z_size*dz]

# Plotting
fig, axes = plt.subplots(3, 3, figsize=(12, 8))

# PSF projections
axes[0, 0].imshow(psfE_xy, extent=extent_xy, cmap='magma')
axes[0, 0].set_title('psfE - XY')
axes[0, 1].imshow(psfE_xz, extent=extent_xz, cmap='magma')
axes[0, 1].set_title('psfE - XZ')
axes[0, 2].imshow(psfE_yz, extent=extent_yz, cmap='magma')
axes[0, 2].set_title('psfE - YZ')

axes[1, 0].imshow(psfD_xy, extent=extent_xy, cmap='magma')
axes[1, 0].set_title('psfD - XY')
axes[1, 1].imshow(psfD_xz, extent=extent_xz, cmap='magma')
axes[1, 1].set_title('psfD - XZ')
axes[1, 2].imshow(psfD_yz, extent=extent_yz, cmap='magma')
axes[1, 2].set_title('psfD - YZ')

# PropVol projections
axes[2, 0].imshow(propVol_xy, extent=extent_xy, cmap='magma')
axes[2, 0].set_title('propVol - XY')
axes[2, 1].imshow(propVol_xz, extent=extent_xz, cmap='magma')
axes[2, 1].set_title('propVol - XZ')
axes[2, 2].imshow(propVol_yz, extent=extent_yz, cmap='magma')
axes[2, 2].set_title('propVol - YZ')

for ax in axes.ravel():
    ax.set_xlabel('X (µm)' if 'XY' in ax.get_title() else 'Depth (µm)')
    ax.set_ylabel('Y (µm)' if 'XY' in ax.get_title() else 'Depth (µm)')

plt.tight_layout()
plt.show()
# #Efield = prop.propagate(propVol, p_source="gauss")

# #%% loop through angles
# # data = np.zeros((anglePara.numSteps, 3))
# # for i in range(anglePara.numSteps):
# #     angle = i*anglePara.res
# #     print(angle)
# #     #gen aving folder
# #     path = mainPath + f"angle_{angle:.2f}deg" 
# #     os.makedirs(path)
# #     path = path + "\\"
# #     # rotate detection psf
# #     psfD = bf.rotPSF(psfDgen, angle)
# #     # # calc power spectrum and center of mass
# #     # psScoh, psSinc, psfSinc, psE, psD, com_pScoh_1D, com_tScoh_1D, com_tSinc_1D, com_pSinc_1D, com_tE_1D, com_pE_1D, com_tD_1D, com_pD_1D = bf.calcPScom(psfE, psfD, theta, phi)
    
# #     # calc power spectrum and center of mass
# #     psSinc, com_tSinc_1D, com_pSinc_1D = bf.calcPScomInc(psfE, psfD, theta, phi)
    
# #     # gen and save 2D histo
# #     # com_tE_2D, com_pE_2D = bf.calc2Dhisto(theta, phi, psE, angle, "excitation", path)
# #     # com_tD_2D, com_pD_2D = bf.calc2Dhisto(theta, phi, psD, angle, "detection", path)
# #     # com_tScoh_2D, com_pScoh_2D = bf.calc2Dhisto(theta, phi, psScoh, angle, "systemCoh", path)
# #     com_tSinc_2D, com_pSinc_2D = bf.calc2Dhisto(theta, phi, psSinc, angle, "systemIncoh", path)
# #     # save maxProj and arrays
# #     bf.saveMaxProj(psfE, sexy, angle, "psf", "excitationPSF", path)
# #     bf.saveMaxProj(psfD, sexy, angle, "psf", "detectionPSF", path)
# #     # bf.saveMaxProj(psfSinc, sexy, angle, "psf", "systemIncPSF", path)
# #     # bf.saveMaxProj(psE, kxy, angle, "powSpec", "excitationPS", path)
# #     # bf.saveMaxProj(psD, kxy, angle, "powSpec", "detectionPS", path)
# #     bf.saveMaxProj(psSinc, kxy, angle, "powSpec", "systemIncPS", path)
# #     # bf.saveMaxProj(psScoh, kxy, angle, "powSpec", "systemCohPS", path)
# #     # order data
# #     # d = (angle, com_tScoh_2D, com_pScoh_2D, com_tSinc_2D, com_pSinc_2D, com_tE_2D, com_pE_2D, com_tD_2D, com_pD_2D)
# #     d = (angle, com_tSinc_2D, com_pSinc_2D)
# #     data[i,:] = d

# # #%% plot and save results
# # bf.plotResInc(data, mainPath)
# # np.savetxt(mainPath + "com.txt", data, fmt="%.5f", delimiter="\t")
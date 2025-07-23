import biobeam as bb
import numpy as np
import time
from types import SimpleNamespace
from numpy.fft import fftn, fftshift, fftfreq
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import center_of_mass
from scipy.ndimage import rotate
import os
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"  # deaktiviert Ausgabe komplett
# %% set parameters
s = time.time()

optExc = SimpleNamespace()
optExc.Nx = 512
optExc.Ny = optExc.Nx
optExc.Nz = optExc.Nx
optExc.dx = 0.10
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
optDet.angle = 22.5

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

angle = 90
path = "C:\\Users\\Goerlitz\\Documents\\temp\\test\\"

e = time.time()
print(f"Time - Set parameters : {e - s} s")

# %% pepare vols histo parameters
def prepPara(optExc, optDet):
    #s = time.time()
    
    # calculate excitation PSF 
    _ ,exE,eyE,ezE = bb.focus_field_cylindrical(shape = (optExc.Nx, optExc.Ny, optExc.Nz), 
                                              units = (optExc.dx, optExc.dy, optExc.dz), 
                                              lam = optExc.lam, NA = optExc.NA, n0 = optExc.n0, 
                                              return_all_fields = True, 
                                              n_integration_steps = 100)
    psfE = exE + eyE + ezE
    
    # calculate detection PSF
    _ ,exDgen,eyDgen,ezDgen = bb.focus_field_beam(shape = ( optDet.Nx, optDet.Ny, optDet.Nz), 
                            units = ( optDet.dx, optDet.dy, optDet.dz), 
                            lam = optDet.lam, NA = optDet.NA, n0 = optDet.n0, 
                            return_all_fields = True, 
                            n_integration_steps = 100)
    psfDgen = exDgen + eyDgen + ezDgen
    
    #psfE = psfDgen
    
    # angle space
    Nz, Ny, Nx = psfE.shape
    kx = fftshift(fftfreq(Nx, d=optDet.dx)) * 2 * np.pi
    KZ, KY, KX = np.meshgrid(kx, kx, kx, indexing='ij')
    k_mag = np.sqrt(KX**2 + KY**2 + KZ**2) + 1e-12  # avoid divide by 0
    theta = np.rad2deg(np.arccos(KZ / k_mag))  # polar angle
    phi = np.rad2deg(np.arctan2(KY, KX))       # azimuthal angle
    
    #% histograms
    # Define bins (e.g. 0° to 90°, 180 bins → 0.5° bin width)
    bins = np.linspace(0, 180, 181)
    # Define angle bins
    n_bins = 180
    angle_bins = np.rad2deg(np.linspace(0, np.pi, n_bins))  # radians
    angles_rad = np.deg2rad(angle_bins) # angles in radians
    d = optExc.Nz*optExc.dz/2
    sexy = [-d, d, -d, d]
    kxy = [np.min(kx), np.max(kx), np.min(kx), np.max(kx)]
    
    #e = time.time()
    #print(f"Time - perpare vol and histo parameters : {e - s} s")
    return psfE, psfDgen, theta, phi, bins, angles_rad, sexy, kxy
psfE, psfDgen, theta, phi, bins, angles_rad, sexy, kxy = prepPara(optExc, optDet)
#%% roatate detection PSF
def rotPSF(psfDgen, angle):
    #s = time.time()
    psfD = rotate(psfDgen, angle, axes=(0, 1), reshape=False, order=3)
    psfD = psfDgen
    #e = time.time()
    #print(f"Time - rotate detection PSF old: {e - s} s")
    return psfD
psfD = rotPSF(psfDgen, optDet.angle)
# %% power spectrum compIncoherent
def calcPScom(psfE, psfD, theta, phi):
    #s = time.time()
    # excitation
    otfEcoh = fftshift(fftn(psfE)) # coherent OTF
    psE = np.abs(otfEcoh) ** 2 # power spectrum
    comE = tuple(np.round(center_of_mass(psE)).astype(int))
    tE = theta[comE]
    pE = phi[comE]
    # detection
    otfDcoh = fftshift(fftn(psfD)) # coherent OTF
    psD = np.abs(otfDcoh) ** 2 # power spectrum
    comD = tuple(np.round(center_of_mass(psD)).astype(int))
    tD = theta[comD]
    pD = phi[comD]
    # system
    psS = psE * psD
    
    # com of ps
    comS = tuple(np.round(center_of_mass(psS)).astype(int))
    ts1 = theta[comS]
    ps1 = phi[comS]
    #print(f"COM (deg) System PS: θ = {ts1:.2f}, φ = {ps1:.2f}")
    
    # semi-incoherent 
    psfS = psfE * psfD
    otfSincoh = fftshift(fftn(psfS)) # coherent OTF
    psSinc = np.abs(otfSincoh) ** 2 # power spectrum
    comSinc = tuple(np.round(center_of_mass(psSinc)).astype(int))
    ts1inc = theta[comSinc]
    ps1inc = phi[comSinc]
    #print(f"COM (deg) System PS inc: θ = {ts1inc:.2f}, φ = {ps1inc:.2f}")
    #e = time.time()
    #print(f"Time - power spectrum: {e - s} s")
    return psS, psSinc, psE, psD, ps1, ts1, ts1inc, ps1inc, tE, pE, tD, pD
psScoh, psSinc, psE, psD, com_pScoh_1D, com_tScoh_1D, com_tSinc_1D, com_pSinc_1D, com_tE_1D, com_pE_1D, com_tD_1D, com_pD_1D = calcPScom(psfE, psfD, theta, phi)
#%% generate and save 1D histograms 
def gen1Dhisto(theta, psS, bins, angle, name, path):
    power_flat = psS.flatten()
    theta_flat = theta.flatten()
    histT, binEdge = np.histogram(theta.flatten(), bins=bins, weights = power_flat)
    plt.figure(figsize=(7, 4))
    plt.hist(theta_flat, bins=bins, weights = power_flat, density=False, color='c', edgecolor='k')
    plt.xlabel("Scattering angle (degrees)")
    plt.ylabel("Normalized intensity")
    plt.title('Angular Power Distribution 1D' + "\n" + name +" / "+ f"angle : {angle:.2f}")
    plt.xlim(1, 180)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path + name + "angularPowerDistribution1D.png", dpi=300, bbox_inches='tight')
    np.savetxt(path + name + "angularPowerDistribution1D.txt", histT, fmt='%.5f')
    plt.show()
    return histT, binEdge
psEH, beEH = gen1Dhisto(theta, psE, bins, angle, "excitation", path)
psDH, beDH = gen1Dhisto(theta, psD, bins, angle, "detection", path)
psShcoh, beSHc = gen1Dhisto(theta, psScoh, bins, angle, "systemCoh", path)
psShinc, beSHi = gen1Dhisto(theta, psSinc, bins, angle, "systemInc", path)
# save 1D histos
def save1Dhisto(hist, bin_edges, angle, name, path):
    np.savetxt(path + name + "histo1D.txt", np.column_stack((bin_edges[:-1], bin_edges[1:], hist)),
           header='Bin_Start\tBin_End\tCount', fmt='%.5f')
save1Dhisto(psEH, beEH, angle, "excitation", path)
save1Dhisto(psEH, beEH, angle, "detection", path)
save1Dhisto(psEH, beEH, angle, "systemCoh", path)
save1Dhisto(psEH, beEH, angle, "systemInc", path)
#%%generate and save 2D histos
def calc2Dhisto(theta, phi, powSpec, angle, name, path):# Flatten all arrays to 1D
    theta_flat = theta.ravel()
    phi_flat = phi.ravel()
    power_flat = powSpec.ravel()
    
    # Define bin edges for angular resolution
    theta_bins = np.linspace(0, 180, 181)         # polar angle: 0 to 180 deg
    phi_bins = np.linspace(-180, 180, 361)      # azimuth: -180 to 180 deg
    
    # Create 2D histogram of power in (theta, phi)
    hist, theta_edges, phi_edges = np.histogram2d(
        theta_flat, phi_flat, bins=[theta_bins, phi_bins], weights=power_flat)
    
    # Plot the angular power distribution
    plt.figure(figsize=(10, 5))
    plt.imshow(
        hist.T,
        extent=[theta_bins[0], theta_bins[-1], phi_bins[0], phi_bins[-1]],
        aspect='auto',
        origin='lower',
        cmap='inferno'
    )
    plt.xlabel('Polar angle θ [deg]')
    plt.ylabel('Azimuthal angle φ [deg]')
    plt.title('Angular Power Distribution' + "\n" + name +" / "+ f"angle : {angle:.2f}")
    plt.colorbar(label='Power')
    plt.savefig(path + name + "angularPowerDistribution2D.png", dpi=300, bbox_inches='tight')
    np.savetxt(path + name + "angularPowerDistribution2D.txt", hist, fmt='%.5f')
    plt.show()
    
    # 1. Get bin centersZ
    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])  # shape (N_theta,)
    phi_centers = 0.5 * (phi_bins[:-1] + phi_bins[1:])    # shape (N_phi,)
    
    # 2. Create meshgrid of angles
    Theta, Phi = np.meshgrid(np.deg2rad(theta_centers), np.deg2rad(phi_centers), indexing='ij')  # shape (N_theta, N_phi)
    
    # 3. Convert spherical to Cartesian unit vectors
    x = np.sin(Theta) * np.cos(Phi)
    y = np.sin(Theta) * np.sin(Phi)
    z = np.cos(Theta)
    
    # 4. Multiply each unit vector by the corresponding power value
    # hist.shape == (N_theta, N_phi)
    wx = x * hist
    wy = y * hist
    wz = z * hist
    
    # 5. Sum over all bins (center of mass vector)
    cx = np.sum(wx)
    cy = np.sum(wy)
    cz = np.sum(wz)
    
    # 6. Normalize to unit vector (optional)
    vec_norm = np.sqrt(cx**2 + cy**2 + cz**2) + 1e-12
    cx /= vec_norm
    cy /= vec_norm
    cz /= vec_norm
    
    # 7. Convert back to angles (if needed)
    theta_com = np.arccos(cz)                    # [0, pi]
    phi_com = np.arctan2(cy, cx)                 # [-pi, pi]
    
    # Convert to degrees if needed
    theta_com_deg = np.rad2deg(theta_com)
    phi_com_deg = np.rad2deg(phi_com)
    
    return theta_com_deg, phi_com_deg
com_tE_2D, com_pE_2D = calc2Dhisto(theta, phi, psE, angle, "excitation", path)
com_tD_2D, com_pD_2D = calc2Dhisto(theta, phi, psD, angle, "detection", path)
com_tScoh_2D, com_pScoh_2D = calc2Dhisto(theta, phi, psScoh, angle, "systemCoh", path)
com_tSinc_2D, com_pSinc_2D = calc2Dhisto(theta, phi, psSinc, angle, "systemIncoh", path)

# %% plot show stuff
#s = time.time()

#%
# t = theta
# tMask = t > np.max(t)*0.95
# t = tMask*t
# power_flat = theta.flatten()
# theta_flat = theta.flatten()
# plt.figure(figsize=(7, 4))
# plt.hist(theta_flat, bins=bins, weights=power_flat, density=True, color='c', edgecolor='k')
# plt.xlabel("Scattering angle (degrees)")
# plt.ylabel("Normalized intensity")
# plt.title("Angular scattering distribution (weighted histogram)")
# plt.xlim(0, 180)
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#%%

# c = int(optExc.Nz/2)
# b = np.abs(psfD).T
# vminB = np.min(b)
# vmaxB = np.max(b)
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 3, 1)
# plt.imshow(b[c, :, :], extent=sexy, cmap='hot', vmin = vminB, vmax = vmaxB)
# plt.title("psf - x")

# plt.subplot(1, 3, 2)
# plt.imshow(b[:, c, :], extent=sexy, cmap='hot', vmin = vminB, vmax = vmaxB)
# plt.title("psf - y")

# plt.subplot(1, 3, 3)
# plt.imshow(b[:, :, c], extent=sexy, cmap='hot', vmin = vminB, vmax = vmaxB)
# plt.title("psf - z")

# plt.tight_layout()
# plt.show()


# e = time.time()
# print(f"Time - plot : {e - s} s")
#%%
# Plot weighted histogram
# power_flat = psS.flatten()
# theta_flat = theta.flatten()
# histT, _ = np.histogram(theta.flatten(), bins=bins)
# #histT, _ = np.histogram(theta.flatten(), bins=bins, weights = 1/histT)
# plt.figure(figsize=(7, 4))
# plt.hist(theta_flat, bins=bins, weights = power_flat, density=False, color='c', edgecolor='k')
# plt.xlabel("Scattering angle (degrees)")
# plt.ylabel("Normalized intensity")
# plt.title("Angular scattering distribution (weighted histogram)")
# plt.xlim(1, 180)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

#%%
# # Plot weighted histogram
# theta = np.rad2deg(np.arccos(KZ / k_mag))
# c = int(optExc.Nz/2)
# z = np.arange(optExc.Nz)
# zz, yy, xx = np.meshgrid(z, z, z, indexing='ij')
# dist2 = (zz-c)**2 + (yy-c)**2 + (xx-c)**2
# sphere = np.zeros((optExc.Nx, optExc.Nx, optExc.Nx), dtype=np.float32)-1
# sphere[dist2 <= c**2] = 1
# thetaSphere = theta * sphere
# power_flat = psS.flatten()
# theta_flat = theta.flatten()
# hist, _ = np.histogram(theta.flatten(), bins=bins, weights = psS.flatten())
# histT, _ = np.histogram(thetaSphere.flatten(), bins=bins)
# #histT, _ = np.histogram(theta.flatten(), bins=bins, weights = 1/histT)
# plt.figure(figsize=(7, 4))
# #plt.hist(theta_flat, bins=bins, weights = 1/histT, density=False, color='c', edgecolor='k')
# plt.plot(1/histT)
# plt.xlabel("Scattering angle (degrees)")
# plt.ylabel("Normalized intensity")
# plt.title("Angular scattering distribution (weighted histogram)")
# plt.xlim(1, 180)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# b = theta.T

# plt.figure(figsize=(12, 5))

# plt.subplot(1, 3, 1)
# plt.imshow(b[c, :, :], extent=sexy, cmap='hot')
# plt.title("psf - x")

# plt.subplot(1, 3, 2)
# plt.imshow(b[:, c, :], extent=sexy, cmap='hot')
# plt.title("psf - y")

# plt.subplot(1, 3, 3)
# plt.imshow(b[:, :, c], extent=sexy, cmap='hot')
# plt.title("psf - z")
# plt.colorbar()

# plt.tight_layout()
# plt.show()
# #%% plot all
# a = np.abs(psfE).T
# vminA = np.min(a)
# vmaxA = np.max(a)
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 3, 1)
# plt.imshow(np.max(a, axis=0), extent=sexy, cmap='hot', vmin = vminA, vmax = vmaxA)
# plt.title("exc - psf - x - max proj")

# plt.subplot(1, 3, 2)
# plt.imshow(np.max(a, axis=1), extent=sexy, cmap='hot', vmin = vminA, vmax = vmaxA)
# plt.title("exc - psf - y - max proj")

# plt.subplot(1, 3, 3)
# plt.imshow(np.max(a, axis=2), extent=sexy, cmap='hot', vmin = vminA, vmax = vmaxA)
# plt.title("exc - psf - z - max proj")

# plt.tight_layout()
# plt.show()

# a = np.abs(psfD).T
# vminA = np.min(a)
# vmaxA = np.max(a)
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 3, 1)
# plt.imshow(np.max(a, axis=0), extent=sexy, cmap='hot', vmin = vminA, vmax = vmaxA)
# plt.title("det - psf - x - max proj")

# plt.subplot(1, 3, 2)
# plt.imshow(np.max(a, axis=1), extent=sexy, cmap='hot', vmin = vminA, vmax = vmaxA)
# plt.title("det - psf - y - max proj")

# plt.subplot(1, 3, 3)
# plt.imshow(np.max(a, axis=2), extent=sexy, cmap='hot', vmin = vminA, vmax = vmaxA)
# plt.title("det - psf - z - max proj")

# plt.tight_layout()
# plt.show()

# a = np.abs(psE).T
# vminA = np.min(a)
# vmaxA = np.max(a)
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 3, 1)
# plt.imshow(np.max(a, axis=0), extent=kxy, cmap='hot', vmin = vminA, vmax = vmaxA)
# plt.title("exc - pow spec - x - max proj")

# plt.subplot(1, 3, 2)
# plt.imshow(np.max(a, axis=1), extent=kxy, cmap='hot', vmin = vminA, vmax = vmaxA)
# plt.title("exc - pow spec - y - max proj")

# plt.subplot(1, 3, 3)
# plt.imshow(np.max(a, axis=2), extent=kxy, cmap='hot', vmin = vminA, vmax = vmaxA)
# plt.title("exc - pow spec - z - max proj")

# plt.tight_layout()
# plt.show()

# a = np.abs(psD).T
# vminA = np.min(a)
# vmaxA = np.max(a)
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 3, 1)
# plt.imshow(np.max(a, axis=0), extent=kxy, cmap='hot', vmin = vminA, vmax = vmaxA)
# plt.title("det - pow spec - x - max proj")

# plt.subplot(1, 3, 2)
# plt.imshow(np.max(a, axis=1), extent=kxy, cmap='hot', vmin = vminA, vmax = vmaxA)
# plt.title("det - pow spec - y - max proj")

# plt.subplot(1, 3, 3)
# plt.imshow(np.max(a, axis=2), extent=kxy, cmap='hot', vmin = vminA, vmax = vmaxA)
# plt.title("det - pow spec - z - max proj")

# plt.tight_layout()
# plt.show()

# a = np.abs(psS).T
# vminA = np.min(a)
# vmaxA = np.max(a)
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 3, 1)
# plt.imshow(np.max(a, axis=0), extent=kxy, cmap='hot', vmin = vminA, vmax = vmaxA)
# plt.title("sys - pow spec - x - max proj")

# plt.subplot(1, 3, 2)
# plt.imshow(np.max(a, axis=1), extent=kxy, cmap='hot', vmin = vminA, vmax = vmaxA)
# plt.title("sys - pow spec - y - max proj")

# plt.subplot(1, 3, 3)
# plt.imshow(np.max(a, axis=2), extent=kxy, cmap='hot', vmin = vminA, vmax = vmaxA)
# plt.title("sys - pow spec - z - max proj")

# plt.tight_layout()
# plt.show()
# #%% 
def calc2Dhisto(theta, phi, powSpec, angle, name, path):# Flatten all arrays to 1D
    theta_flat = theta.ravel()
    phi_flat = phi.ravel()
    power_flat = powSpec.ravel()
    
    # Define bin edges for angular resolution
    theta_bins = np.linspace(0, 180, 181)         # polar angle: 0 to 180 deg
    phi_bins = np.linspace(-180, 180, 361)      # azimuth: -180 to 180 deg
    
    # Create 2D histogram of power in (theta, phi)
    hist, theta_edges, phi_edges = np.histogram2d(
        theta_flat, phi_flat, bins=[theta_bins, phi_bins], weights=power_flat)
    
    # Plot the angular power distribution
    plt.figure(figsize=(10, 5))
    plt.imshow(
        hist.T,
        extent=[theta_bins[0], theta_bins[-1], phi_bins[0], phi_bins[-1]],
        aspect='auto',
        origin='lower',
        cmap='inferno'
    )
    plt.xlabel('Polar angle θ [deg]')
    plt.ylabel('Azimuthal angle φ [deg]')
    plt.title('Angular Power Distribution' + "\n" + name +" / "+ f"angle : {angle:.2f}")
    plt.colorbar(label='Power')
    plt.savefig(path + name + "angularPowerDistribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 1. Get bin centersZ
    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])  # shape (N_theta,)
    phi_centers = 0.5 * (phi_bins[:-1] + phi_bins[1:])    # shape (N_phi,)
    
    # 2. Create meshgrid of angles
    Theta, Phi = np.meshgrid(np.deg2rad(theta_centers), np.deg2rad(phi_centers), indexing='ij')  # shape (N_theta, N_phi)
    
    # 3. Convert spherical to Cartesian unit vectors
    x = np.sin(Theta) * np.cos(Phi)
    y = np.sin(Theta) * np.sin(Phi)
    z = np.cos(Theta)
    
    # 4. Multiply each unit vector by the corresponding power value
    # hist.shape == (N_theta, N_phi)
    wx = x * hist
    wy = y * hist
    wz = z * hist
    
    # 5. Sum over all bins (center of mass vector)
    cx = np.sum(wx)
    cy = np.sum(wy)
    cz = np.sum(wz)
    print(f"cx: θ = {cx:.2f}")
    print(f"cy: θ = {cy:.2f}")
    print(f"cz: θ = {cz:.2f}")
    
    # 6. Normalize to unit vector (optional)
    vec_norm = np.sqrt(cx**2 + cy**2 + cz**2) + 1e-12
    cx /= vec_norm
    cy /= vec_norm
    cz /= vec_norm
    print(f"cx norm: θ = {cx:.2f}")
    print(f"cy norm: θ = {cy:.2f}")
    print(f"cz norm: θ = {cz:.2f}")
    
    # 7. Convert back to angles (if needed)
    theta_com = np.arccos(cz)                    # [0, pi]
    phi_com = np.arctan2(cy, cx)                 # [-pi, pi]
    print(f"theta_com: θ = {theta_com:.2f}")
    print(f"phi_com: θ = {phi_com:.2f}")
    
    # Convert to degrees if needed
    theta_com_deg = np.rad2deg(theta_com)
    phi_com_deg = np.rad2deg(phi_com)
    print(f"theta_com_deg: θ = {theta_com_deg:.2f}")
    print(f"phi_com_deg: θ = {phi_com_deg:.2f}")
    
    return theta_com_deg, phi_com_deg

angle = 90
path = "C:\\temp\\"
te2, pe2 = calc2Dhisto(theta, phi, psE, angle, "excitation", path)
td2, pd2 = calc2Dhisto(theta, phi, psD, angle, "detection", path)
ts2, ps2 = calc2Dhisto(theta, phi, psS, angle, "system coh", path)
ts2inc, ps2inc = calc2Dhisto(theta, phi, psSinc, angle, "system incoh", path)
print(f"COM (deg) Excitation hist: θ = {te2:.2f}, φ = {pe2:.2f}")
print(f"COM (deg) Detection hist: θ = {td2:.2f}, φ = {pd2:.2f}")
print(f"COM (deg) System hist: θ = {ts2:.2f}, φ = {ps2:.2f}")
print(f"COM (deg) System hist inc: θ = {ts2inc:.2f}, φ = {ps2inc:.2f}")
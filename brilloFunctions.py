import biobeam as bb
import numpy as np
from numpy.fft import fftn, fftshift, fftfreq
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from scipy.ndimage import rotate
import arrayfire as af
import os
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"  # deaktiviert Ausgabe komplett


def prepPara(optExc, optDet):
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
    psfD = exDgen + eyDgen + ezDgen
    
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
    
    return psfE.astype(np.complex64), psfD.astype(np.complex64), theta.astype(np.float16), phi.astype(np.float16), bins, angles_rad, sexy, kxy
    #return psfE, psfDgen, theta, phi, bins, angles_rad, sexy, kxy
    

def create_sphere_with_gaussian_noise(shape=(256, 256, 256),
                                      n_background=1.33,
                                      n_sphere=1.43,
                                      radius=256,
                                      noise_std=0.01,
                                      center=None):
    Z, Y, X = shape
    if center is None:
        center = (Z // 2, Y // 2, X // 2)

    # Background
    volume = np.full(shape, n_background, dtype=np.float16)

    # Sphere
    z, y, x = np.ogrid[:Z, :Y, :X]
    dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    volume[dist <= radius] = n_sphere

    # Add Gaussian noise
    volume += np.random.normal(0, noise_std, size=shape)
    return volume.astype(np.float16)


def rotPSF(psfDgen, angle):
    #s = time.time()
    psfD = rotate(psfDgen, angle, axes=(0, 1), reshape=False, order=3)
    #e = time.time()
    #print(f"Time - rotate detection PSF old: {e - s} s")
    return psfD

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
    return psS, psSinc, psfS, psE, psD, ps1, ts1, ts1inc, ps1inc, tE, pE, tD, pD

def calcPScomInc(psfE, psfD, theta, phi):
    # #s = time.time()
    # # excitation
    # otfEcoh = fftshift(fftn(psfE)) # coherent OTF
    # psE = np.abs(otfEcoh) ** 2 # power spectrum
    # comE = tuple(np.round(center_of_mass(psE)).astype(int))
    # tE = theta[comE]
    # pE = phi[comE]
    # # detection
    # otfDcoh = fftshift(fftn(psfD)) # coherent OTF
    # psD = np.abs(otfDcoh) ** 2 # power spectrum
    # comD = tuple(np.round(center_of_mass(psD)).astype(int))
    # tD = theta[comD]
    # pD = phi[comD]
    # # system
    # psS = psE * psD
    
    # # com of ps
    # comS = tuple(np.round(center_of_mass(psS)).astype(int))
    # ts1 = theta[comS]
    # ps1 = phi[comS]
    # #print(f"COM (deg) System PS: θ = {ts1:.2f}, φ = {ps1:.2f}")
    
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
    return psSinc, ts1inc, ps1inc

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
    plt.savefig(path + name + f"_{angle:.2f}" + "_angularPowerDistribution1D.png", dpi=300, bbox_inches='tight')
    np.savetxt(path + name + f"_{angle:.2f}" + "_angularPowerDistribution1D.txt", histT, fmt='%.5f')
    #plt.show()
    plt.close()
    return histT, binEdge

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
    # plt.figure(figsize=(10, 5))
    # plt.imshow(
    #     hist.T,
    #     extent=[theta_bins[0], theta_bins[-1], phi_bins[0], phi_bins[-1]],
    #     aspect='auto',
    #     origin='lower',
    #     cmap='inferno'
    # )
    # plt.xlabel('Polar angle θ [deg]')
    # plt.ylabel('Azimuthal angle φ [deg]')
    # plt.colorbar(label='Power')
    # if isinstance(angle, str):
    #     plt.title('Angular Power Distribution' + "\n" + name +" / "+ angle)
    #     plt.savefig(path + name + angle + "_angularPowerDistribution2D.png", dpi=300, bbox_inches='tight')
    #     np.savetxt(path + name + angle + "_angularPowerDistribution2D.txt", hist, fmt='%.5f')
    # else:
    #     plt.title('Angular Power Distribution' + "\n" + name +" / "+ f"angle : {angle:.2f}")
    #     plt.savefig(path + name + f"_{angle:.2f}" + "_angularPowerDistribution2D.png", dpi=300, bbox_inches='tight')
    #     np.savetxt(path + name + f"_{angle:.2f}" + "_angularPowerDistribution2D.txt", hist, fmt='%.5f')
    # #plt.show()
    # plt.close()
    
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

def fftgpuPS(psf):
    psfAF = af.to_array(psf.astype(np.complex64))  # FFT braucht komplexes Array
    ps = np.abs(fftshift(af.fft3(psfAF, True)))**2
    af.device_gc()
    return ps

def fftcpuPS(psf):
    return np.abs(fftshift(fftn(psf)))**2

def saveMaxProj(arr, ext, angle, name1, name2, path):
    a = np.abs(arr).T
    vminA = np.min(a)
    vmaxA = np.max(a)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(np.max(a, axis=0), extent=ext, cmap='hot', vmin = vminA, vmax = vmaxA)
    plt.title(name1 + " - x - max proj" + "\n" + name2 +" / "+ f"angle : {angle:.2f}")
    plt.xlabel('z [um]')
    plt.ylabel('y [um]')
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.max(a, axis=1), extent=ext, cmap='hot', vmin = vminA, vmax = vmaxA)
    plt.title(name1 + " - y - max proj" + "\n" + name2 +" / "+ f"angle : {angle:.2f}")
    plt.xlabel('z [um]')
    plt.ylabel('x [um]')
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.max(a, axis=2), extent=ext, cmap='hot', vmin = vminA, vmax = vmaxA)
    plt.title(name1 + " - z - max proj" + "\n" + name2 +" / "+ f"angle : {angle:.2f}")
    plt.xlabel('y [um]')
    plt.ylabel('x [um]')
    
    plt.tight_layout()
    plt.savefig(path + name2 + f"_{angle:.2f}" + "_maxProj.png", dpi=300, bbox_inches='tight')
    np.save(path + name2 + f"_{angle:.2f}" + '_Vol.npy', arr)
    #plt.sh←w()
    plt.close()
    
def plotRes(data, mainPath):
    a = data[:,0]
    tCoh = data[:,1]
    tInc = data[:,3]
    tE = data[:,5]
    tD = data[:,7]
    pCoh = data[:,2]
    pInc = data[:,4]
    pE = data[:,6]
    pD = data[:,8]
    
    plt.figure(figsize=(8, 6))
    
    # Plot coherent CoM
    plt.plot(a, tCoh, 'o-', label='system coherent')
    plt.plot(a, tInc, 'o--', label='system incoherent')
    plt.plot(a, tE, color='black', linestyle =':', label='excitation')
    plt.plot(a, tD, color='grey', linestyle =':', label='detection')
    # Labels and title
    plt.xlabel("Detection Angle (degrees)")
    plt.ylabel("Center of Mass θ (degrees)")
    plt.title("θ CoM vs detection angle")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(mainPath + "thetaVSangle.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(8, 6))
    
    # Plot coherent CoM
    plt.plot(a, pCoh, 'o-', label='system coherent')
    plt.plot(a, pInc, 'o--', label='system incoherent')
    plt.plot(a, pE, color='black', linestyle =':', label='excitation')
    plt.plot(a, pD, color='grey', linestyle =':', label='detection')
    
    # Labels and title
    plt.xlabel("Detection Angle (degrees)")
    plt.ylabel("Center of Mass φ (degrees)")
    plt.title("φ CoM vs detection angle")
    plt.savefig(mainPath + "phiVSangle.png", dpi=300, bbox_inches='tight')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()

def plotResInc(data, mainPath):
    a = data[:,0]
    tInc = data[:,1]
    pInc = data[:,2]

    
    plt.figure(figsize=(8, 6))
    
    # Plot coherent CoM
    plt.plot(a, tInc, 'o--', label='system incoherent')
    # Labels and title
    plt.xlabel("Detection Angle (degrees)")
    plt.ylabel("Center of Mass θ (degrees)")
    plt.title("θ CoM vs detection angle")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(mainPath + "thetaVSangle.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(8, 6))
    
    # Plot coherent CoM
    plt.plot(a, pInc, 'o--', label='system incoherent')
    
    # Labels and title
    plt.xlabel("Detection Angle (degrees)")
    plt.ylabel("Center of Mass φ (degrees)")
    plt.title("φ CoM vs detection angle")
    plt.savefig(mainPath + "phiVSangle.png", dpi=300, bbox_inches='tight')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_max_projections(volume, voxel_size=(1.0, 1.0, 1.0), cmap='hot', title="Max Intensity Projections"):
    """
    chatGPT generated
    Plottet Maximalprojektionen eines 3D-Volumes in allen drei Dimensionen mit Titel.
    
    Parameters:
        volume (ndarray): 3D-Array (Z, Y, X)
        voxel_size (tuple): (dz, dy, dx) Voxelgrößen für Achsenskalierung
        cmap (str): Colormap für das Plotten
        title (str): Gesamttitel der Figure
    """
    dz, dy, dx = voxel_size

    # Max-Projektionen berechnen
    max_xy = np.max(volume, axis=0)  # Projektion über Z
    max_xz = np.max(volume, axis=1)  # Projektion über Y
    max_yz = np.max(volume, axis=2)  # Projektion über X

    # Plot erstellen
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    # XY-Projektion
    Z, Y, X = volume.shape

    extent_xy = [
        -X//2 * dx,  X//2 * dx,   # X-Achse
        -Y//2 * dy,  Y//2 * dy    # Y-Achse
        ]
    
    extent_xz = [
        -X//2 * dx,  X//2 * dx,   # X-Achse
        -Z//2 * dz,  Z//2 * dz    # Z-Achse
        ]
    
    extent_yz = [
        -Y//2 * dy,  Y//2 * dy,   # Y-Achse
        -Z//2 * dz,  Z//2 * dz    # Z-Achse
        ]
    
    axes[0].imshow(max_xy, cmap=cmap, extent=extent_xy, origin='lower', aspect='auto')
    axes[0].set_title('Z → XY')
    axes[0].set_xlabel('X (µm)')
    axes[0].set_ylabel('Y (µm)')

    # XZ-Projektion
    #extent_xz = [0, volume.shape[2] * dx, 0, volume.shape[0] * dz]
    axes[1].imshow(max_xz, cmap=cmap, extent=extent_xz, origin='lower', aspect='auto')
    axes[1].set_title('Y → XZ')
    axes[1].set_xlabel('X (µm)')
    axes[1].set_ylabel('Z (µm)')

    # YZ-Projektion
    #extent_yz = [0, volume.shape[1] * dy, 0, volume.shape[0] * dz]
    axes[2].imshow(max_yz, cmap=cmap, extent=extent_yz, origin='lower', aspect='auto')
    axes[2].set_title('X → YZ')
    axes[2].set_xlabel('Y (µm)')
    axes[2].set_ylabel('Z (µm)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
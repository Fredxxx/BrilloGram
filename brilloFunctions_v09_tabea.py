import biobeam as bb
import sys
import numpy as np
from numpy.fft import fftn, fftshift, fftfreq
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from scipy.ndimage import rotate
import arrayfire as af
import os
from types import SimpleNamespace
import gc
import tifffile as tiff
import json
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"  # deaktiviert Ausgabe komplett

def prepPara2(optExc, optDet):
    # calculate excitation PSF 
    _ ,exE,eyE,ezE = bb.focus_field_cylindrical(shape = (optExc.Nx, optExc.Ny, optExc.Nz), 
                                              units = (optExc.dx, optExc.dy, optExc.dz), 
                                              lam = optExc.lam, NA = optExc.NA, n0 = optExc.n0, 
                                              return_all_fields = True, 
                                              n_integration_steps = 100)

    psfE = exE #+ eyE + ezE
    
    # calculate detection PSF
    _ ,exDgen,eyDgen,ezDgen = bb.focus_field_beam(shape = ( optDet.Nx, optDet.Ny, optDet.Nz), 
                            units = ( optDet.dx, optDet.dy, optDet.dz), 
                            lam = optDet.lam, NA = optDet.NA, n0 = optDet.n0, 
                            return_all_fields = True, 
                            n_integration_steps = 100)
    psfD = rotPSF(exDgen, optDet.angle)
    
    
    # angle space
    Nz, Ny, Nx = psfE.shape
    kx = fftshift(fftfreq(Nx, d=optDet.dx)) * 2 * np.pi
    KZ, KY, KX = np.meshgrid(kx, kx, kx, indexing='ij')
    k_mag = np.sqrt(KX**2 + KY**2 + KZ**2) + 1e-12  # avoid divide by 0
    theta = np.rad2deg(np.arccos(KZ / k_mag))  # polar angle
    phi = np.rad2deg(np.arctan2(KY, KX))       # azimuthal angle
    
    #% histograms
    # Define bins (e.g. 0° to 90°, 180 bins → 0.5° bin width)
    bins = np.linspace(0, 180, 1801)
    # Define angle bins
    n_bins = 180
    angle_bins = np.rad2deg(np.linspace(0, np.pi, n_bins))  # radians
    angles_rad = np.deg2rad(angle_bins) # angles in radians
    d = optExc.Nz*optExc.dz/2
    sexy = [-d, d, -d, d]
    kxy = [np.min(kx), np.max(kx), np.min(kx), np.max(kx)]
    
    return psfE.astype(np.complex64), psfD.astype(np.complex64), theta.astype(np.float16), phi.astype(np.float16), bins, angles_rad, sexy, kxy
    #return psfE, psfDgen, theta, phi, bins, angles_rad, sexy, kxy


def genPaddArray(sx, sy, sz, vol):    
    #padded_scatVol = np.random.normal(1.33, 0.002, size=(sx*3, sy*3, sz*3)).astype(np.float16)
    padded_scatVol = np.random.normal(1.33335, 0.00074, size=(sx*3, sy*3, sz*3)).astype(np.float16)
    z, y, x = vol.shape
    Z, Y, X = padded_scatVol.shape
    # Start-Indices (zentriert)
    start_z = (Z - z) // 2
    start_y = (Y - y) // 2
    start_x = (X - x) // 2
    # End-Indices
    end_z = start_z + z
    end_y = start_y + y
    end_x = start_x + x
    # Insert
    padded_scatVol[start_z:end_z, start_y:end_y, start_x:end_x] = vol
    return padded_scatVol

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
 

def json_serializable(obj):
    """Konvertiert Objekte (wie NumPy-Arrays), die JSON nicht nativ versteht."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Typ {type(obj)} nicht JSON-serialisierbar")   


def safe_print_progress(per, idx, idx_max):
    # \033[F bewegt den Cursor eine Zeile hoch
    # \033[K löscht die Zeile ab Cursor-Position
    sys.stdout.write(f"\033[F\033[K{per:.1f}% ({idx}/{idx_max})\n")
    sys.stdout.flush()

def process_shift2(coo, padded_scatVol, psfE, psfDgen, optExc, optDet, optGen, path, theta, phi, i, j, w, idx, idxMax):
      
   # report progress
   per = idx/idxMax*100
   safe_print_progress(per, idx, idxMax)
   #print(f"{per:.1f}% ({idx}/{idxMax})", end='\r')
   #sys.stdout.flush()
   
   # Excitation: shift volume, init propagator and propagate
   te = padded_scatVol[coo[4]:coo[5], coo[2]:coo[3], coo[0]:coo[1]]
   td = padded_scatVol[coo[4]:coo[5], coo[0]:coo[1], coo[2]:coo[3]]
   del padded_scatVol 
   te = bb.Bpm3d(dn=te, units = (optDet.dx,)*3, lam=optExc.lam/optExc.n0)
   psfEscat = te.propagate(u0 = psfE[0,:,:])
   del psfE 
   del te 
   gc.collect()
   
   # Detection: shift volume, init propagator, propagate and rotate
   # t = padded_scatVol[coo[4]:coo[5], coo[0]:coo[1], coo[2]:coo[3]]
   # t = bb.Bpm3d(dn=t, units = (optDet.dx,)*3, lam=optDet.lam/optExc.n0)
   td = bb.Bpm3d(dn=td, units = (optDet.dx,)*3, lam=optDet.lam/optExc.n0)
   psfDscat = td.propagate(u0 = psfDgen[0,:,:])
   psfDscat = rotPSF(psfDscat, 90)
   psfEscat = rotPSF(psfEscat, 90)

   del psfDgen
   del td
   gc.collect()
   
   # Powerspectrum 
   psfS = psfEscat * psfDscat
   #psS = fftcpuPS(psfS)
   
   # calc results
   resS = calcMain(fftgpuPS(psfS), 'sys', theta, phi, optDet, optGen, idx, coo, i, j, w, path)
   del psfS
   gc.collect()
   saveHisto(resS, path, optDet)
   
   # resE = calcMain(fftcpuPS(psfEscat), 'exc', theta, phi, optDet, optGen, idx, coo, i, j, w, path)
   # del psfEscat
   # gc.collect()
   # saveHisto(resE, path, optDet)
   
   # resD = calcMain(fftcpuPS(psfDscat), 'det', theta, phi, optDet, optGen, idx, coo, i, j, w, path)
   # del psfDscat
   # gc.collect()
   # saveHisto(resD, path, optDet)

def saveHisto(res, path, optDet):
   # Plot/save the angular power distribution
   fig, ax = plt.subplots(figsize=(10, 5))
   im = ax.imshow(
      res.hist.T,
      extent=[res.thetaBins[0], res.thetaBins[-1], res.phiBins[0], res.phiBins[-1]],
      aspect='equal', origin='lower', cmap='inferno'
   )
   ax.set_xlabel('Polar angle θ [deg]')
   ax.set_ylabel('Azimuthal angle φ [deg]')
   fig.colorbar(im, ax=ax, label='Power')    
   ax.set_title('psHist' + "\n" + res.name +" / "+ str(optDet.angle))  
   fig.savefig(os.path.join(path, res.name, f"{res.idx}_fig.png"), dpi=300, bbox_inches='tight')
   #np.savetxt(os.path.join(path, res.name, f"{res.idx}_fig.txt"), res.hist, fmt='%.5f')
   plt.show()
   plt.close()

def calcMain(ps, name, theta, phi, optDet, optGen, idx, coo, i, j, w, path):
    res = SimpleNamespace()
    res.idx = num2Str0000(idx)
    res.name = name
    res = calcAngles(theta, phi, ps, res, str(optDet.angle))
    res.x = i
    res.y = j
    res.z = w
    sx = round((coo[0]+coo[1])/2)
    sy = round((coo[2]+coo[3])/2)
    sz = round((coo[4]+coo[5])/2)
    res.pos = "sx" + num2Str0000(sx) + "_sy" + num2Str0000(sy) + "_sz" + num2Str0000(sz)
    res = calcBrilloSpec(optDet, optGen, res)
    path = os.path.join(path, res.name, f"{res.idx}.json")
    with open(path, 'w') as f:
        json.dump(vars(res), f, indent=4, default=json_serializable)
    return res

def lor(alpha, weight, optGen, f):
    shift = optGen.BSshiftA * np.sin(np.deg2rad(alpha))
    width = optGen.BSwidthA * np.sin(np.deg2rad(alpha))**2
    f_r = f[np.newaxis, :] 
    denominator = ((f_r - shift[:, np.newaxis])**2 + (0.5*width[:, np.newaxis])**2)
    numerator = width[:, np.newaxis] 

    # We also apply the weight vector now
    return (numerator / denominator) * weight[:, np.newaxis]

def calcBrilloSpec(optDet, optGen, res):
    #  convert to brillouin shift
    q0 = (1/optDet.lam)**2
    res.thetaComBS = q0 * optGen.Vs * np.sin(np.deg2rad(res.thetaCOM)/2) * 10**-9
    res.phiComBS = q0 * optGen.Vs * np.sin(np.deg2rad(res.phiCOM)/2) * 10**-9
    res.thetaStdBS = q0 * optGen.Vs * np.sin(np.deg2rad(res.thetaSTD)/2) * 10**-9
    # include water spread
    res.BSspecX = np.arange(optGen.BSspecStart, optGen.BSspecEnd, optGen.BSspecRes)
    res.BSspecSumPhi = np.sum(res.hist, axis=1)
    res.BSspecThX = res.thetaBins[:-1] + 0.5
    res.BSspecTheta = np.sum(lor(res.BSspecThX, res.BSspecSumPhi, optGen, res.BSspecX), axis=0)
    #plt.plot(f,m)
    return res

def calcAngles(theta, phi, powSpec, res, angle):

    # spread of angles   
    thetaFlat = theta.ravel()
    phiFlat = phi.ravel()
    powerFlat = powSpec.ravel()
   
    # Define bin edges for angular resolution
    res.thetaBins = np.linspace(0, 180, 1801)         # polar angle: 0 to 180 deg
    res.phiBins = np.linspace(-180, 180, 3601)      # azimuth: -180 to 180 deg
    
    # Create 2D histogram in (theta, phi)
    res.hist, _, _= np.histogram2d(
        thetaFlat, phiFlat, bins=[res.thetaBins, res.phiBins], weights=powerFlat)
    
    # # Plot/save the angular power distribution
    # fig, ax = plt.subplots(figsize=(10, 5))
    # im = ax.imshow(
    #    res.hist.T,
    #    extent=[res.thetaBins[0], res.thetaBins[-1], res.phiBins[0], res.phiBins[-1]],
    #    aspect='equal', origin='lower', cmap='inferno'
    # )
    # ax.set_xlabel('Polar angle θ [deg]')
    # ax.set_ylabel('Azimuthal angle φ [deg]')
    # fig.colorbar(im, ax=ax, label='Power')    
    # ax.set_title('psHist' + "\n" + res.name +" / "+ angle)  
    # fig.savefig(path + name + "_deg" + angle + ".png", dpi=300, bbox_inches='tight')
    # np.savetxt(path + name + "_deg" + angle + ".txt", hist, fmt='%.5f')
    # #plt.show()
    # plt.close()
    
    # calc COM
    res.comSinc = tuple(np.round(center_of_mass(powSpec)).astype(int))
    res.thetaCOM = float(theta[res.comSinc]) + 180 
    print(res.thetaCOM)
    sys.stdout.flush()
    res.phiCOM = float(phi[res.comSinc])
    
    # calc std and mean
    # meshgrid
    thetaGrid, phiGrid = np.meshgrid(res.thetaBins[:-1] + 180, res.phiBins[:-1] + 0.5, indexing='ij')
    # normalization factor
    weight = np.sum(res.hist)
    # mean value
    res.thetaMean = np.sum(thetaGrid * res.hist) / weight 
    res.phiMean   = np.sum(phiGrid * res.hist) / weight
    # STD
    res.thetaSTD = np.sqrt(np.sum(res.hist * (thetaGrid - res.thetaMean)**2) / weight)
    res.phiSTD   = np.sqrt(np.sum(res.hist * (phiGrid - res.phiMean)**2) / weight)

    return res

def calcResOld(theta, phi, powSpec, angle, name, path):# Flatten all arrays to 1D

    # spread of angles   
    thetaFlat = theta.ravel()
    phiFlat = phi.ravel()
    powerFlat = powSpec.ravel()
   
    # Define bin edges for angular resolution
    thetaBins = np.linspace(0, 180, 181)         # polar angle: 0 to 180 deg
    phiBins = np.linspace(-180, 180, 361)      # azimuth: -180 to 180 deg
    
    # Create 2D histogram in (theta, phi)
    hist, _, _= np.histogram2d(
        thetaFlat, phiFlat, bins=[thetaBins, phiBins], weights=powerFlat)
    
    # Plot/save the angular power distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        hist.T,
        extent=[thetaBins[0], thetaBins[-1], phiBins[0], phiBins[-1]],
        aspect='auto', origin='lower', cmap='inferno'
    )
    ax.set_xlabel('Polar angle θ [deg]')
    ax.set_ylabel('Azimuthal angle φ [deg]')
    fig.colorbar(im, ax=ax, label='Power')    
    ax.set_title('psHist' + "\n" + name +" / "+ angle)  
    fig.savefig(path + name + "_deg" + angle + ".png", dpi=300, bbox_inches='tight')
    np.savetxt(path + name + "_deg" + angle + ".txt", hist, fmt='%.5f')
    #plt.show()
    plt.close()
    
    # calc COM
    comSinc = tuple(np.round(center_of_mass(powSpec)).astype(int))
    thetaCOM = theta[comSinc]
    phiCOM = phi[comSinc]
    
    # calc std and mean
    # meshgrid
    thetaGrid, phiGrid = np.meshgrid(thetaBins[:-1] + 0.5, phiBins[:-1] + 0.5, indexing='ij')
    # normalization factor
    weight = np.sum(hist)
    # mean value
    thetaMean = np.sum(thetaGrid * hist) / weight
    phiMean   = np.sum(phiGrid * hist) / weight
    # STD
    thetaSTD = np.sqrt(np.sum(hist * (thetaGrid - thetaMean)**2) / weight)
    phiSTD   = np.sqrt(np.sum(hist * (phiGrid - phiMean)**2) / weight)
    
    return comSinc, thetaCOM, phiCOM, thetaSTD, phiSTD, thetaMean, phiMean

def rotPSF(psfDgen, angle):
    #s = time.time()
    psfD = rotate(psfDgen, angle, axes=(0, 1), reshape=False, order=3)
    #e = time.time()
    #print(f"Time - rotate detection PSF old: {e - s} s")
    return psfD

def num2Str0000(num):
    s = str(abs(num))
    sl = len(s)
    if sl == 1:
        s = "000" + s
    elif sl == 2:
        s = "00" + s  
    elif sl == 3:
        s = "0" + s     
    return s

def fftgpuPS(psf):
    psfAF = af.to_array(psf.astype(np.complex64))  # FFT braucht komplexes Array
    ps = np.abs(fftshift(af.fft3(psfAF, True)))**2
    af.device_gc()
    return ps

def fftcpuPS(psf):
    return np.abs(fftshift(fftn(psf)))**2

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
    #plt.show()
    
def saveDist(path, name, data, xRes, yRes, scatDim):
    #np.savetxt(path + name + ".txt", data, fmt="%.5f", delimiter="\t")
    tiff.imwrite(
        path + name + '.tiff',
        data.astype(np.float32),
        imagej=True,
        resolution=(xRes, yRes),
        metadata={
            'spacing': scatDim,    # Z voxel size
            'unit': 'um'      # physical unit
        }
        )
    
def sizeof_var(name, var):
    """Return variable name, size in bytes, and human-readable size."""
    if isinstance(var, np.ndarray):
        size = var.nbytes
    else:
        try:
            size = sys.getsizeof(var)
        except:
            size = 0
    return name, size, size / 1024**2, size / 1024**3  # bytes, MB, GB

def print_memory_usage(**kwargs):
    """Print a table of variable memory usage."""
    print(f"{'Variable':<20} {'MB':>10} {'GB':>10}")
    print("-" * 45)
    total = 0
    for name, var in kwargs.items():
        _, size, size_mb, size_gb = sizeof_var(name, var)
        total += size
        print(f"{name:<20} {size_mb:10.2f} {size_gb:10.3f}")
    print("-" * 45)
    print(f"{'TOTAL':<20} {total/1024**2:10.2f} {total/1024**3:10.3f}\n")

# Example usage inside your code
def process_shift_debug(coo, padded_scatVol, psfE, psfD, optExc, optDet, mainPath, theta, phi, i, j, w):
    print_memory_usage(
        coo=coo,
        padded_scatVol=padded_scatVol,
        psfE=psfE,
        psfD=psfD,
        optExc=optExc,
        optDet=optDet,
        mainPath=mainPath,
        theta=theta,
        phi=phi,
        i=i,
        j=j,
        w=w
    )
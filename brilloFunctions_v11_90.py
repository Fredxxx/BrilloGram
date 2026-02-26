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
from scipy.optimize import curve_fit
import tifffile as tiff
import json
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"  # deaktiviert Ausgabe komplett

def prepPara2(optExc, optDet):
    # calculate excitation PSF 
    # _ ,psfE,eyE,ezE = bb.focus_field_cylindrical(shape = (optExc.Nx, optExc.Ny, optExc.Nz), 
    #                                           units = (optExc.dx, optExc.dy, optExc.dz), 
    #                                           lam = optExc.lam, NA = optExc.NA, n0 = optExc.n0, 
    #                                           return_all_fields = True, 
    #                                           n_integration_steps = 100)
    
    
    # _ ,psfE, _, _  = bb.focus_field_beam(shape = ( optExc.Nx, optExc.Ny, optExc.Nz), 
    #                     units = ( optExc.dx, optExc.dy, optExc.dz), 
    #                     lam = optExc.lam, NA = [0.75, 0.8], n0 = optExc.n0, 
    #                     return_all_fields = True, 
    #                     n_integration_steps = 100)
    
    _ ,psfE,eyE,ezE = bb.focus_field_beam(shape = (optExc.Nx, optExc.Ny, optExc.Nz), 
                                              units = (optExc.dx, optExc.dy, optExc.dz), 
                                              lam = optExc.lam, NA = optExc.NA, n0 = optExc.n0, 
                                              return_all_fields = True, 
                                              n_integration_steps = 100)

    
    # calculate detection PSF
    _ ,exDgen, _, _ = bb.focus_field_beam(shape = ( optDet.Nx, optDet.Ny, optDet.Nz), 
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
    
    return psfE.astype(np.complex64), psfD.astype(np.complex64), theta.astype(np.float16), phi.astype(np.float16)


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

def genPaddArray2(sx, sy, sz, vol):
    
    Z, Y, X = sx * 3 , sy * 3, sz * 3
    
    padded_scatVol = np.random.normal(1.33335, 0.00074, size=(Z, Y, X)).astype(np.float16)
    
    vz, vy, vx = vol.shape
    
    
    def get_indices(target_size, source_size):
        # Startpunkt im Ziel-Array (0, wenn Quelle größer ist)
        start_t = max(0, (target_size - source_size) // 2)
        # Startpunkt im Quell-Array (Zentrum-Ausschnitt, wenn Quelle größer ist)
        start_s = max(0, (source_size - target_size) // 2)
        # Wie viel passt maximal rein?
        length = min(target_size, source_size)
        return start_t, start_s, length

    zt, zs, zl = get_indices(Z, vz)
    yt, ys, yl = get_indices(Y, vy)
    xt, xs, xl = get_indices(X, vx)

    padded_scatVol[zt:zt+zl, yt:yt+yl, xt:xt+xl] = vol[zs:zs+zl, ys:ys+yl, xs:xs+xl]
    
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
    sys.stdout.write(f"\033[F\033[K{per:.1f}% ({idx}/{idx_max})\n")
    sys.stdout.flush() 
    
def safe_print(t):
    sys.stdout.write(t)
    sys.stdout.write("\n")
    sys.stdout.flush()

def process_shift2(coo, padded_scatVol, psfE, psfDgen, optExc, optDet, optGen, path, theta, phi, i, j, w, idx, idxMax):
      
   # report progress
   per = idx/idxMax*100
   safe_print_progress(per, idx, idxMax)
   
   # Excitation: shift volume, init propagator and propagate
   te = padded_scatVol[coo[4]:coo[5], coo[2]:coo[3], coo[0]:coo[1]]
   #td=te
   #td = np.swapaxes(te, 0, 2)

   td = np.rot90(te, k=1, axes=(1, 2))
   #td = padded_scatVol[coo[4]:coo[5], coo[0]:coo[1], coo[2]:coo[3]]
   #plot_max_projections(te, voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="te")
   #plot_max_projections(td, voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="td")

   del padded_scatVol 
   te = bb.Bpm3d(dn=te, units = (optDet.dx,)*3, lam=optDet.lam/optExc.n0)
   psfEscat = te.propagate(u0 = psfE[0,:,:])
   #plot_max_projections(np.abs(psfEscat), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psfEScat")
   
   del psfE 
   del te 
   gc.collect()
   
   # Detection: shift volume, init propagator, propagate and rotate
   # t = padded_scatVol[coo[4]:coo[5], coo[0]:coo[1], coo[2]:coo[3]]
   # t = bb.Bpm3d(dn=t, units = (optDet.dx,)*3, lam=optDet.lam/optExc.n0)
   
   td = bb.Bpm3d(dn=td, units = (optDet.dx,)*3, lam=optDet.lam/optExc.n0)
   psfDscat = td.propagate(u0 = psfDgen[0,:,:])
   #plot_max_projections(np.abs(psfDscat), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psfDScat")
   
   #psfDscat = rotPSF(psfDscat, 90)
   psfEscat = rotPSF(psfEscat, 90)

   del psfDgen
   del td
   gc.collect()
   
   # Powerspectrum 
   psfS = psfEscat * psfDscat
   #plot_max_projections(np.abs(psfS), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psfS")
   #psS = fftcpuPS(psfS)
   #plot_max_projections(psS, voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psfS")
   
   
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
   # #np.savetxt(os.path.join(path, res.name, f"{res.idx}_fig.txt"), res.hist, fmt='%.5f')
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
    #res = calcBrilloSpec(optDet, optGen, res)
    res = calcBrilloSpec2(optDet, optGen, res)
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

def calcBrilloSpec2(optDet, optGen, res):
    #  convert to brillouin shift
    _, mu_theta, sigma_theta = res.fitOp_theta
    _, mu_phi, sigma_phi = res.fitOp_phi
    
    q0 = (1/optDet.lam)**2
    res.thetaComBS = q0 * optGen.Vs * np.sin(np.deg2rad(mu_theta)/2) * 10**-9
    res.phiComBS = q0 * optGen.Vs * np.sin(np.deg2rad(mu_phi)/2) * 10**-9
    res.thetaStdBS = q0 * optGen.Vs * np.sin(np.deg2rad(sigma_theta)/2) * 10**-9
    res.phiStdBS = q0 * optGen.Vs * np.sin(np.deg2rad(sigma_phi)/2) * 10**-9
    
    
    # # include water spread
    # res.BSspecX = np.arange(optGen.BSspecStart, optGen.BSspecEnd, optGen.BSspecRes)
    # res.BSspecSumPhi = np.sum(res.hist, axis=1)
    # res.BSspecThX = res.thetaBins[:-1] + 0.5
    # res.BSspecTheta = np.sum(lor(res.BSspecThX, res.BSspecSumPhi, optGen, res.BSspecX), axis=0)
    # #plt.plot(f,m)
    return res

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

def safe_print2(mu_theta, sigma_theta, mu_phi, sigma_phi, res):
    sys.stdout.write(f"IDX: {res.idx}, theta={mu_theta:.4f}, sigmaTh={sigma_theta:.4f}, phi={mu_phi:.4f}, sigmaPh={sigma_phi:.4f}")
    sys.stdout.write("\n")
    sys.stdout.flush()
    
def calcAngles(theta, phi, powSpec, res, angle):

    thetaFlat = theta.ravel()
    phiFlat = phi.ravel()
    powerFlat = powSpec.ravel()

    res.thetaBins = np.linspace(0, 180, 451)
    res.phiBins = np.linspace(-180, 180, 901)
    
    # weighted histogram
    hist_sum, _, _ = np.histogram2d(
        thetaFlat, phiFlat, bins=[res.thetaBins, res.phiBins], weights=powerFlat)

    # counts per pixel
    counts, _, _ = np.histogram2d(
        thetaFlat, phiFlat, bins=[res.thetaBins, res.phiBins])

    # pixel power normalised to pixel counts
    with np.errstate(divide='ignore', invalid='ignore'):
        res.hist = hist_sum / counts
        res.hist[counts == 0] = 0

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
    res.thetaCOM = float(theta[res.comSinc])
    res.phiCOM = float(phi[res.comSinc])

    theta_c = 0.5 * (res.thetaBins[:-1] + res.thetaBins[1:])
    phi_c   = 0.5 * (res.phiBins[:-1] + res.phiBins[1:]) 
    
    res.proj_theta = res.hist.T.sum(axis=0)
    res.proj_phi = res.hist.T.sum(axis=1)
    
    res.fitOp_theta, _ = curve_fit(gauss, theta_c, res.proj_theta, 
                              p0=[res.proj_theta.max(), theta_c[np.argmax(res.proj_theta)], 5])

    A_theta, mu_theta, sigma_theta = res.fitOp_theta
    
    res.fitOp_phi, _ = curve_fit(gauss, phi_c,  res.proj_phi, 
                            p0=[res.proj_phi.max(), phi_c[np.argmax(res.proj_phi)], 5])
    
    
    A_phi, mu_phi, sigma_phi = res.fitOp_phi
    
    safe_print2(mu_theta, sigma_theta, mu_phi, sigma_phi, res)

    return res

def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

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

def plot_max_projections2(volume, voxel_size=(1.0, 1.0, 1.0), cmap='hot', title="Max Intensity Projections", space="real"):
    """
    Plottet Maximalprojektionen eines 3D-Volumes im Ortsraum (real) oder Frequenzraum (fft).
    """
    dz, dy, dx = voxel_size
    Z, Y, X = volume.shape

    # Logik für Einheiten und Skalierung
    if space == "fft":
        unit = "1/µm"
        # Frequenzschritte berechnen (df = 1 / (N * dx))
        dfz = 1.0 / (Z * dz)
        dfy = 1.0 / (Y * dy)
        dfx = 1.0 / (X * dx)
        
        # Extents für FFT (zentriert)
        extent_xy = [-X//2 * dfx, X//2 * dfx, -Y//2 * dfy, Y//2 * dfy]
        extent_xz = [-X//2 * dfx, X//2 * dfx, -Z//2 * dfz, Z//2 * dfz]
        extent_yz = [-Y//2 * dfy, Y//2 * dfy, -Z//2 * dfz, Z//2 * dfz]
    else:
        unit = "µm"
        extent_xy = [-X//2 * dx, X//2 * dx, -Y//2 * dy, Y//2 * dy]
        extent_xz = [-X//2 * dx, X//2 * dx, -Z//2 * dz, Z//2 * dz]
        extent_yz = [-Y//2 * dy, Y//2 * dy, -Z//2 * dz, Z//2 * dz]

    # Max-Projektionen berechnen
    max_xy = np.max(volume, axis=0)
    max_xz = np.max(volume, axis=1)
    max_yz = np.max(volume, axis=2)

    # Plot erstellen
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{title} ({space.upper()} space)", fontsize=16)

    # XY-Projektion
    axes[0].imshow(max_xy, cmap=cmap, extent=extent_xy, origin='lower', aspect='auto')
    axes[0].set_title('Z-Projection (XY)')
    axes[0].set_xlabel(f'X ({unit})')
    axes[0].set_ylabel(f'Y ({unit})')

    # XZ-Projektion
    axes[1].imshow(max_xz, cmap=cmap, extent=extent_xz, origin='lower', aspect='auto')
    axes[1].set_title('Y-Projection (XZ)')
    axes[1].set_xlabel(f'X ({unit})')
    axes[1].set_ylabel(f'Z ({unit})')

    # YZ-Projektion
    axes[2].imshow(max_yz, cmap=cmap, extent=extent_yz, origin='lower', aspect='auto')
    axes[2].set_title('X-Projection (YZ)')
    axes[2].set_xlabel(f'Y ({unit})')
    axes[2].set_ylabel(f'Z ({unit})')

    fig.colorbar(axes[2].images[0], ax=axes.ravel().tolist(), fraction=0.046, pad=0.04, aspect=20)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    #plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes
    
    
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
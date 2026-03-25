# external 
import numpy as np
from numpy.fft import fftn, fftshift, fftfreq
import biobeam as bb
from types import SimpleNamespace
import matplotlib.pyplot as plt
import configparser
configparser.SafeConfigParser = configparser.ConfigParser

optExc = SimpleNamespace()
optExc.Nx = 256 #256-good
optExc.Ny = optExc.Nx
optExc.Nz = optExc.Nx
optExc.dx = 1 #0.5
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

    #fig.colorbar(axes[2].images[0], ax=axes.ravel().tolist(), fraction=0.046, pad=0.04, aspect=20)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    #plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes

#%%

_ ,psfE, _, _  = bb.focus_field_beam(shape = ( optExc.Nx, optExc.Ny, optExc.Nz), 
                    units = ( optExc.dx, optExc.dy, optExc.dz), 
                    lam = optExc.lam, NA = [0.796, 0.8], n0 = optExc.n0, 
                    return_all_fields = True, 
                    n_integration_steps = 100)

plot_max_projections2(np.abs(psfE)**2, voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="Bessel NA 0.796-0.8", space="real")
plot_max_projections2(np.abs(fftshift(fftn(psfE)))**2, voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="Bessel NA 0.796-0.8", space="fft")
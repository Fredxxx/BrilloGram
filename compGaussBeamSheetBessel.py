#internal 
import sys
from types import SimpleNamespace
import os
import configparser
configparser.SafeConfigParser = configparser.ConfigParser

# external 
import numpy as np
from numpy.fft import fftshift, fftfreq
import biobeam as bb
import matplotlib
matplotlib.use('qtagg') 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import rotate



# %% set parameters
#sys.path.append(r'/g/prevedel/members/Goerlitz/projectsHPC/brillo')
#sys.path.append(r'C:/Users/Fred/Documents/GitHub/BrilloGram')
sys.path.append(r'C:/Users/Goerlitz/BrilloGram')
import brilloFunctions_v11_90 as bf
#mainPath = "/g/prevedel/members/Goerlitz/projectsHPC/brillo/results/"

mainPath = "C:/Fred/temp/"
name = "GaussBeamSheetBessel"
#mainPath = "/scratch/goerlitz/brilloCopy/"

path = os.path.join(mainPath, name)
os.makedirs(path, exist_ok=True)
os.makedirs(os.path.join(path, "sys"), exist_ok=True)
#os.makedirs(os.path.join(path, "exc"), exist_ok=True)
#os.makedirs(os.path.join(path, "det"), exist_ok=True)

optExc = SimpleNamespace()
optExc.Nx = 768 #256-good
optExc.Ny = optExc.Nx
optExc.Nz = optExc.Nx
optExc.dx = 0.1 #0.5
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

#%% gen E field
# _ ,psfE,_, _ = bb.focus_field_cylindrical(shape = (optExc.Nx, optExc.Ny, optExc.Nz), 
#                                           units = (optExc.dx, optExc.dy, optExc.dz), 
#                                           lam = optExc.lam, NA = optExc.NA, n0 = optExc.n0, 
#                                           return_all_fields = True, 
#                                           n_integration_steps = 100)

_ ,psfE, _, _  = bb.focus_field_beam(shape = ( optExc.Nx, optExc.Ny, optExc.Nz), 
                    units = ( optExc.dx, optExc.dy, optExc.dz), 
                    lam = optExc.lam, NA = [0.796, 0.8], n0 = optExc.n0, 
                    return_all_fields = True, 
                    n_integration_steps = 100)

# _ ,psfE, _, _  = bb.focus_field_beam(shape = ( optExc.Nx, optExc.Ny, optExc.Nz), 
#                     units = ( optExc.dx, optExc.dy, optExc.dz), 
#                     lam = optExc.lam, NA = [0.046, 0.05], n0 = optExc.n0, 
#                     return_all_fields = True, 
#                     n_integration_steps = 100)

# _ ,psfE,_, _  = bb.focus_field_beam(shape = (optExc.Nx, optExc.Ny, optExc.Nz), 
#                                           units = (optExc.dx, optExc.dy, optExc.dz), 
#                                           lam = optExc.lam, NA = optExc.NA, n0 = optExc.n0, 
#                                           return_all_fields = True, 
#                                           n_integration_steps = 100)

_ ,psfD,_, _  = bb.focus_field_beam(shape = (optExc.Nx, optExc.Ny, optExc.Nz), 
                                          units = (optExc.dx, optExc.dy, optExc.dz), 
                                          lam = optExc.lam, NA = optDet.NA, n0 = optExc.n0, 
                                          return_all_fields = True, 
                                          n_integration_steps = 100)
#psfE = psfD

# angle space
Nz, Ny, Nx = psfD.shape
kx = fftshift(fftfreq(Nx, d=optDet.dx)) * 2 * np.pi
KZ, KY, KX = np.meshgrid(kx, kx, kx, indexing='ij')
k_mag = np.sqrt(KX**2 + KY**2 + KZ**2) + 1e-12  # avoid divide by 0
theta = np.rad2deg(np.arccos(KZ / k_mag))  # polar angle
phi = np.rad2deg(np.arctan2(KY, KX))       # azimuthal angle

del KZ, KY, KX, k_mag, Nz, Nx, Ny, kx
#gc.collect()
#%%
#psfD = rotate(psfD, 90, axes=(0, 1), reshape=False, order=3)
#psfE = rotate(psfE, 90, axes=(0, 1), reshape=False, order=3)
#psfS = psfE * psfD
#del psfE, psfD
#resS = bf.calcMain(bf.fftgpuPS(psfS), 'sys', theta, phi, optDet, optGen, 00, [0,0,0,0,0,0], 0, 0, 0, path)

#%%
# psfDr = rotate(psfE, 90, axes=(0, 1), reshape=False, order=3)
# psfS = psfD * psfDr
# bf.plot_max_projections2(np.abs(psfS), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psf", space="real")
# bf.plot_max_projections2(np.abs(bf.fftgpuPS(psfS)), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="ps", space="fft")
# #resS = bf.calcMain(bf.fftgpuPS(psfD), 'sys', theta, phi, optDet, optGen, 00, [0,0,0,0,0,0], 0, 0, 0, path)

#%%
psfS = rotate(psfD, 90, axes=(0, 1), reshape=False, order=3) * psfE
bf.plot_max_projections2(np.abs(psfS), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psf", space="real")
bf.plot_max_projections2(np.abs(bf.fftgpuPS(psfS)), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="ps", space="fft")
a = 00
#%%
psfS = rotate(psfD, 90, axes=(0, 1), reshape=False, order=3) * rotate(psfE, 90, axes=(0, 1), reshape=False, order=3)
bf.plot_max_projections2(np.abs(psfS), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psf", space="real")
bf.plot_max_projections2(np.abs(bf.fftgpuPS(psfS)), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="ps", space="fft")
a = 90
#%%
resS = bf.calcMain(bf.fftgpuPS(psfS), 'sys', theta, phi, optDet, optGen, 00, [0,0,0,0,0,0], 0, 0, 0, path)
hist = resS.hist.T
proj_theta = hist.sum(axis=0)
proj_phi = hist.sum(axis=1)
#%% 

def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

theta_c = 0.5 * (resS.thetaBins[:-1] + resS.thetaBins[1:])
phi_c   = 0.5 * (resS.phiBins[:-1] + resS.phiBins[1:])

popt_theta, _ = curve_fit(gauss, theta_c, proj_theta, 
                          p0=[proj_theta.max(), theta_c[np.argmax(proj_theta)], 5])
A_theta, mu_theta, sigma_theta = popt_theta

popt_phi, _ = curve_fit(gauss, phi_c, proj_phi, 
                        p0=[proj_phi.max(), phi_c[np.argmax(proj_phi)], 5])
A_phi, mu_phi, sigma_phi = popt_phi

print(f"Theta-Projektion: mu={mu_theta:.4f}, sigma={sigma_theta:.4f}")
print(f"Phi-Projektion:   mu={mu_phi:.4f}, sigma={sigma_phi:.4f}")

#%%
fig = plt.figure(figsize=(8,8))

ax_main  = fig.add_axes([0.1, 0.1, 0.6, 0.6])
ax_top   = fig.add_axes([0.1, 0.72, 0.6, 0.18], sharex=ax_main)
ax_right = fig.add_axes([0.72, 0.1, 0.18, 0.6], sharey=ax_main)
#a = 90
im = ax_main.imshow(
    hist,
    origin='lower',  # unteres phi am unteren Rand
    extent=[resS.thetaBins[0] + a, resS.thetaBins[-1] + a, resS.phiBins[0] - a, resS.phiBins[-1] - a],  # physikalische Koordinaten
    aspect='auto',
    cmap='viridis',
    vmin=None,  # hier kannst du vmin/vmax setzen
    vmax=None
)
ax_main.set_xlabel("Theta")
ax_main.set_ylabel("Phi")

ax_top.plot(theta_c + a, proj_theta, label="Proj Theta")
ax_top.plot(theta_c + a, gauss(theta_c, *popt_theta), 'r--', label="Gauss Fit")
ax_top.set_ylabel("Σ over Phi")
ax_top.legend()
ax_top.tick_params(labelbottom=False)
ax_top.text(0.05, 0.7, f"μ={mu_theta:.2f}, σ={sigma_theta:.2f}", transform=ax_top.transAxes)

ax_right.plot(proj_phi - a, phi_c - a, label="Proj Phi")
ax_right.plot(gauss(phi_c, *popt_phi), phi_c - a, 'r--', label="Gauss Fit")
ax_right.set_xlabel("Σ over Theta")
ax_right.legend()
ax_right.tick_params(labelleft=False)
ax_right.text(0.05, 0.7, f"μ={mu_phi:.2f}, σ={sigma_phi:.2f}", transform=ax_right.transAxes)

plt.show()
#%%
# psfS = rotate(psfD, 90, axes=(0, 1), reshape=False, order=3) * rotate(psfE, 90, axes=(0, 1), reshape=False, order=3)
# bf.plot_max_projections2(np.abs(psfS), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psf", space="real")
# bf.plot_max_projections2(np.abs(bf.fftgpuPS(psfS)), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="ps", space="fft")
# a = 90

# resS = bf.calcMain(bf.fftgpuPS(psfS), 'sys', theta, phi, optDet, optGen, 00, [0,0,0,0,0,0], 0, 0, 0, path)
# hist = resS.hist.T
# proj_theta = hist.sum(axis=0)
# proj_phi = hist.sum(axis=1)


# theta_c = 0.5 * (resS.thetaBins[:-1] + resS.thetaBins[1:])
# phi_c   = 0.5 * (resS.phiBins[:-1] + resS.phiBins[1:])

# popt_theta, _ = curve_fit(gauss, theta_c, proj_theta, 
#                           p0=[proj_theta.max(), theta_c[np.argmax(proj_theta)], 5])
# A_theta, mu_theta, sigma_theta = popt_theta

# popt_phi, _ = curve_fit(gauss, phi_c, proj_phi, 
#                         p0=[proj_phi.max(), phi_c[np.argmax(proj_phi)], 5])
# A_phi, mu_phi, sigma_phi = popt_phi

# print(f"Theta-Projektion: mu={mu_theta:.4f}, sigma={sigma_theta:.4f}")
# print(f"Phi-Projektion:   mu={mu_phi:.4f}, sigma={sigma_phi:.4f}")

# fig = plt.figure(figsize=(8,8))

# ax_main  = fig.add_axes([0.1, 0.1, 0.6, 0.6])
# ax_top   = fig.add_axes([0.1, 0.72, 0.6, 0.18], sharex=ax_main)
# ax_right = fig.add_axes([0.72, 0.1, 0.18, 0.6], sharey=ax_main)
# #a = 90
# im = ax_main.imshow(
#     hist,
#     origin='lower',  # unteres phi am unteren Rand
#     extent=[resS.thetaBins[0] + a, resS.thetaBins[-1] + a, resS.phiBins[0] - a, resS.phiBins[-1] - a],  # physikalische Koordinaten
#     aspect='auto',
#     cmap='viridis',
#     vmin=None,  # hier kannst du vmin/vmax setzen
#     vmax=None
# )
# ax_main.set_xlabel("Theta")
# ax_main.set_ylabel("Phi")

# ax_top.plot(theta_c + a, proj_theta, label="Proj Theta")
# ax_top.plot(theta_c + a, gauss(theta_c, *popt_theta), 'r--', label="Gauss Fit")
# ax_top.set_ylabel("Σ over Phi")
# ax_top.legend()
# ax_top.tick_params(labelbottom=False)
# ax_top.text(0.05, 0.7, f"μ={mu_theta:.2f}, σ={sigma_theta:.2f}", transform=ax_top.transAxes)

# ax_right.plot(proj_phi - a, phi_c - a, label="Proj Phi")
# ax_right.plot(gauss(phi_c, *popt_phi), phi_c - a, 'r--', label="Gauss Fit")
# ax_right.set_xlabel("Σ over Theta")
# ax_right.legend()
# ax_right.tick_params(labelleft=False)
# ax_right.text(0.05, 0.7, f"μ={mu_phi:.2f}, σ={sigma_phi:.2f}", transform=ax_right.transAxes)

# plt.show()


psfS = rotate(psfD, 90, axes=(0, 1), reshape=False, order=3) * psfE
bf.plot_max_projections2(np.abs(psfS), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psf", space="real")
bf.plot_max_projections2(np.abs(bf.fftgpuPS(psfS)), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="ps", space="fft")
a = 00

resS = bf.calcMain(bf.fftgpuPS(psfS), 'sys', theta, phi, optDet, optGen, 00, [0,0,0,0,0,0], 0, 0, 0, path)
hist = resS.hist.T
proj_theta = hist.sum(axis=0)
proj_phi = hist.sum(axis=1)


theta_c = 0.5 * (resS.thetaBins[:-1] + resS.thetaBins[1:])
phi_c   = 0.5 * (resS.phiBins[:-1] + resS.phiBins[1:])

popt_theta, _ = curve_fit(gauss, theta_c, proj_theta, 
                          p0=[proj_theta.max(), theta_c[np.argmax(proj_theta)], 5])
A_theta, mu_theta, sigma_theta = popt_theta

popt_phi, _ = curve_fit(gauss, phi_c, proj_phi, 
                        p0=[proj_phi.max(), phi_c[np.argmax(proj_phi)], 5])
A_phi, mu_phi, sigma_phi = popt_phi

print(f"Theta-Projektion: mu={mu_theta:.4f}, sigma={sigma_theta:.4f}")
print(f"Phi-Projektion:   mu={mu_phi:.4f}, sigma={sigma_phi:.4f}")

fig = plt.figure(figsize=(8,8))

ax_main  = fig.add_axes([0.1, 0.1, 0.6, 0.6])
ax_top   = fig.add_axes([0.1, 0.72, 0.6, 0.18], sharex=ax_main)
ax_right = fig.add_axes([0.72, 0.1, 0.18, 0.6], sharey=ax_main)
#a = 90
im = ax_main.imshow(
    hist,
    origin='lower',  # unteres phi am unteren Rand
    extent=[resS.thetaBins[0] + a, resS.thetaBins[-1] + a, resS.phiBins[0] - a, resS.phiBins[-1] - a],  # physikalische Koordinaten
    aspect='auto',
    cmap='viridis',
    vmin=None,  # hier kannst du vmin/vmax setzen
    vmax=None
)
ax_main.set_xlabel("Theta")
ax_main.set_ylabel("Phi")

ax_top.plot(theta_c + a, proj_theta, label="Proj Theta")
ax_top.plot(theta_c + a, gauss(theta_c, *popt_theta), 'r--', label="Gauss Fit")
ax_top.set_ylabel("Σ over Phi")
ax_top.legend()
ax_top.tick_params(labelbottom=False)
ax_top.text(0.05, 0.7, f"μ={mu_theta:.2f}, σ={sigma_theta:.2f}", transform=ax_top.transAxes)

ax_right.plot(proj_phi - a, phi_c - a, label="Proj Phi")
ax_right.plot(gauss(phi_c, *popt_phi), phi_c - a, 'r--', label="Gauss Fit")
ax_right.set_xlabel("Σ over Theta")
ax_right.legend()
ax_right.tick_params(labelleft=False)
ax_right.text(0.05, 0.7, f"μ={mu_phi:.2f}, σ={sigma_phi:.2f}", transform=ax_right.transAxes)

plt.show()

#%%
bf.plot_max_projections2(np.abs(psfE), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psf exc", space="real")
bf.plot_max_projections2(np.abs(bf.fftgpuPS(psfE)), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="ps esx", space="fft")

bf.plot_max_projections2(np.abs(rotate(psfD, 90, axes=(0, 1), reshape=False, order=3)), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="psf det", space="real")
bf.plot_max_projections2(np.abs(bf.fftgpuPS(rotate(psfD, 90, axes=(0, 1), reshape=False, order=3))), voxel_size=(optExc.dx, optExc.dx, optExc.dx), cmap='hot', title="ps det", space="fft")

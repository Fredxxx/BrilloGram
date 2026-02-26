import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

fileP = r"C:\Fred\temp\20260225_Tabea_checkedScan\00deg_tabea_32x32_gaussBeam_3600resAngle3\sys1\0004.json"

try:
    with open(fileP, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print("File not found")
except json.JSONDecodeError as e:
    print("Invalid JSON:", e)
    

#%% plot hist
hist = np.array(data["hist"])
theta = np.array(data["thetaBins"])+90
phi = np.array(data["phiBins"])
hist = hist.T
proj_theta = hist.sum(axis=0)
proj_phi = hist.sum(axis=1)

#%%
import tifffile

# hist = np.array(data["hist"])  # Deine 2D-Daten

# Optional: in float32 umwandeln, ImageJ kann float TIFFs lesen
hist_img = hist.astype(np.float32)

# Speichern als TIFF
tifffile.imwrite("histogram2.tiff", hist_img)

# #%% plot
# fig = plt.figure(figsize=(8,8))
# ax_main = fig.add_axes([0.1, 0.1, 0.6, 0.6])
# im = ax_main.imshow(
#     hist,
#     origin='lower',           # unterer Rand = phi[0]
#     extent=[theta[0], theta[-1], phi[0], phi[-1]],  # physikalische Koordinaten
#     aspect='auto',
#     vmin=0,
#     vmax=1,
#     cmap='viridis'
# )
# ax_main.set_xlabel("Theta")
# ax_main.set_ylabel("Phi")

# # Colorbar
# cbar = fig.colorbar(im, ax=ax_main)
# cbar.set_label("Value")

# plt.show()



#%% 

def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

theta_c = 0.5 * (theta[:-1] + theta[1:])
phi_c   = 0.5 * (phi[:-1] + phi[1:])

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

im = ax_main.imshow(
    hist,
    origin='lower',  # unteres phi am unteren Rand
    extent=[theta[0], theta[-1], phi[0], phi[-1]],  # physikalische Koordinaten
    aspect='auto',
    cmap='viridis',
    vmin=None,  # hier kannst du vmin/vmax setzen
    vmax=None
)
ax_main.set_xlabel("Theta")
ax_main.set_ylabel("Phi")

ax_top.plot(theta_c, proj_theta, label="Proj Theta")
ax_top.plot(theta_c, gauss(theta_c, *popt_theta), 'r--', label="Gauss Fit")
ax_top.set_ylabel("Σ over Phi")
ax_top.legend()
ax_top.tick_params(labelbottom=False)
ax_top.text(0.05, 0.7, f"μ={mu_theta:.2f}, σ={sigma_theta:.2f}", transform=ax_top.transAxes)

ax_right.plot(proj_phi, phi_c, label="Proj Phi")
ax_right.plot(gauss(phi_c, *popt_phi), phi_c, 'r--', label="Gauss Fit")
ax_right.set_xlabel("Σ over Theta")
ax_right.legend()
ax_right.tick_params(labelleft=False)
ax_right.text(0.05, 0.7, f"μ={mu_phi:.2f}, σ={sigma_phi:.2f}", transform=ax_right.transAxes)

plt.show()
import numpy as np
import time
from types import SimpleNamespace
import os
import brilloFunctions as bf
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "0"  # deaktiviert Ausgabe komplett
# %% set parameters
s = time.time()

optExc = SimpleNamespace()
optExc.Nx = 512
optExc.Ny = optExc.Nx
optExc.Nz = optExc.Nx
optExc.dx = 0.1
optExc.dy = optExc.dx
optExc.dz = optExc.dx
optExc.NA = 0.8
optExc.n0 = 1.33
optExc.lam = 0.532

optDet = SimpleNamespace()
optDet.Nx = optExc.Nx
optDet.Ny = optDet.Nx
optDet.Nz = optDet.Nx
optDet.dx = optExc.dx
optDet.dy = optDet.dx
optDet.dz = optDet.dx
optDet.NA = 0.2
optDet.n0 = optExc.n0
optDet.lam = 0.580
optDet.angle = 22.5

# check if pixelsize smaller than Nyquist 
dxExc = optExc.lam/2/optExc.NA
dxDet = optDet.lam/2/optDet.NA

if np.min([dxDet, dxExc]) <= optExc.dx:
    print('..........................Warning ...................................... -> K-Space not Nyquist sampled for choosen NA and dx.')
#%%
anglePara = SimpleNamespace()
anglePara.start = 0 #deg
anglePara.end = 90 #deg
anglePara.res = 5 #deg
anglePara.numSteps = round((anglePara.end-anglePara.start)/anglePara.res)

mainPath = "C:\\Users\\Goerlitz\\Documents\\temp\\20250728_BrilloScat_0_5_90_deg_08_02_NA\\"


# %% pepare vols histo parameters
psfE, psfDgen, theta, phi, bins, angles_rad, sexy, kxy = bf.prepPara(optExc, optDet)

#%% loop through angles
data = np.zeros((anglePara.numSteps, 3))
for i in range(anglePara.numSteps):
    angle = i*anglePara.res
    print(angle)
    #gen aving folder
    path = mainPath + f"angle_{angle:.2f}deg" 
    os.makedirs(path)
    path = path + "\\"
    # rotate detection psf
    psfD = bf.rotPSF(psfDgen, angle)
    # # calc power spectrum and center of mass
    # psScoh, psSinc, psfSinc, psE, psD, com_pScoh_1D, com_tScoh_1D, com_tSinc_1D, com_pSinc_1D, com_tE_1D, com_pE_1D, com_tD_1D, com_pD_1D = bf.calcPScom(psfE, psfD, theta, phi)
    
    # calc power spectrum and center of mass
    psSinc, com_tSinc_1D, com_pSinc_1D = bf.calcPScomInc(psfE, psfD, theta, phi)
    
    # gen and save 2D histo
    # com_tE_2D, com_pE_2D = bf.calc2Dhisto(theta, phi, psE, angle, "excitation", path)
    # com_tD_2D, com_pD_2D = bf.calc2Dhisto(theta, phi, psD, angle, "detection", path)
    # com_tScoh_2D, com_pScoh_2D = bf.calc2Dhisto(theta, phi, psScoh, angle, "systemCoh", path)
    com_tSinc_2D, com_pSinc_2D = bf.calc2Dhisto(theta, phi, psSinc, angle, "systemIncoh", path)
    # save maxProj and arrays
    bf.saveMaxProj(psfE, sexy, angle, "psf", "excitationPSF", path)
    bf.saveMaxProj(psfD, sexy, angle, "psf", "detectionPSF", path)
    # bf.saveMaxProj(psfSinc, sexy, angle, "psf", "systemIncPSF", path)
    # bf.saveMaxProj(psE, kxy, angle, "powSpec", "excitationPS", path)
    # bf.saveMaxProj(psD, kxy, angle, "powSpec", "detectionPS", path)
    bf.saveMaxProj(psSinc, kxy, angle, "powSpec", "systemIncPS", path)
    # bf.saveMaxProj(psScoh, kxy, angle, "powSpec", "systemCohPS", path)
    # order data
    # d = (angle, com_tScoh_2D, com_pScoh_2D, com_tSinc_2D, com_pSinc_2D, com_tE_2D, com_pE_2D, com_tD_2D, com_pD_2D)
    d = (angle, com_tSinc_2D, com_pSinc_2D)
    data[i,:] = d

#%% plot and save results
bf.plotResInc(data, mainPath)
np.savetxt(mainPath + "com.txt", data, fmt="%.5f", delimiter="\t")
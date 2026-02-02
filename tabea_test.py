import numpy as np
import tifffile as tiff
from scipy.ndimage import zoom
import sys
sys.path.append(r'C:/Users/Fred/Documents/GitHub/BrilloGram')
import brilloFunctions_v09_tabea as bf

scatPath = 'C:\\Fred\\temp\\Tabea_mouseembryo_001.tif'
scatVol1 = tiff.imread(scatPath)/10000
scatVol = np.transpose(scatVol1, (1, 0, 2))
sf = 230/100
scale_factors = (sf, sf, sf)
scatVol = zoom(scatVol1, scale_factors, order=1)  # order=1 = linear interpolation

z_mid, y_mid, x_mid = np.array(scatVol.shape) // 2
sub_vol = scatVol[z_mid-128:z_mid+128, 
                  y_mid-128:y_mid+128, 
                  x_mid-128:x_mid+128]
bf.plot_max_projections(scatVol)
bf.plot_max_projections(sub_vol)
data = np.loadtxt('C:\\Fred\\temp\\tabea_noise.txt') 
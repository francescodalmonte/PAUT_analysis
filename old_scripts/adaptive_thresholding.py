import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
from scipy.ndimage import gaussian_filter


from PAUT_Data import PAUT_Data
from plots_utils import plot_Cscan, plot_Bscan

input_path = "C:/Users/dalmonte/data/ADAMUS/DFKI PAUT/FA3567389 S3 45 17db/FA3567389 S3 45 17db/45° 2195_"
ascans = PAUT_Data(input_path).compose_Ascans()

fig = plt.figure(figsize=(12,5), dpi=150, tight_layout=True)
ax1 = fig.add_subplot(151)

bscan = plot_Bscan(ascans, 19000, ax1, title="B-scan", cmap='jet', 
                   angle=45, pitch=.35, depth_resolution=.082, vmin=0, vmax=100)

ax2 = fig.add_subplot(152)

bscan_notnan = bscan.copy()
bscan_notnan[np.isnan(bscan_notnan)] = 0
bscan_gauss = gaussian_filter(bscan_notnan, sigma=(1, 1))
ax2.imshow(bscan_gauss, aspect='auto', cmap='jet', vmin=0, vmax=100)
ax2.set_title("Gaussian filter")

# apply thresholding

ax3 = fig.add_subplot(153)
bscan_thresh = threshold_otsu(bscan_gauss)
ax3.imshow(bscan_gauss>bscan_thresh, aspect='auto', cmap='jet')
ax3.set_title("Otsu (global)")

ax4 = fig.add_subplot(154)
bscan_thresh = threshold_niblack(bscan_gauss, window_size=25, k=0.01)
ax4.imshow(bscan_gauss>bscan_thresh, aspect='auto', cmap='jet')
ax4.set_title("Niblack (adaptive)")

ax5 = fig.add_subplot(155)
bscan_thresh = threshold_sauvola(bscan_gauss, window_size=25)
ax5.imshow(bscan_gauss>bscan_thresh, aspect='auto', cmap='jet')
ax5.set_title("Sauvola (adaptive)")


plt.show()

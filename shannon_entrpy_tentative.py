import os
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2 as cv
from PAUT_Data import PAUT_Data
from matplotlib import pyplot as plt
from plots_utils import plot_Cscan, plot_Dscan, plot_Ascan

import pywt

# load data
input_path = "C:/Users/dalmonte/data/ADAMUS/DFKI PAUT/FA3566603_Naht2_LF45_22dB_hoch/FA3566603_Naht2_LF45_22dB_hoch/45° 2195_"
obj = PAUT_Data(input_path)
ascans = obj.compose_Ascans()[:,:3000,:]
ascans = gaussian_filter(ascans, sigma=(3, 3, 3))

# filter out x-sequences with low amplitude
ascans_mask = ascans.max(axis=1) > 5

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4), dpi=150, tight_layout=True)
ax[0].imshow(ascans.max(axis=1), aspect='auto', cmap='jet')
ax[1].imshow(ascans_mask, aspect='auto', cmap='gray')
plt.show()

# indexes to access the filtered data
idxs0, idxs2 = np.where(ascans_mask)
n_x_seqs = len(idxs0)

# plot some examples of filtered data
fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(12,8), dpi=150, tight_layout=True)
for i, j in enumerate(np.random.choice(np.arange(0, n_x_seqs), 5)):
    ax[i].plot(ascans[idxs0[j], :, idxs2[j]])
plt.show()

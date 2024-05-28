import os
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2 as cv
from PAUT_preprocessing import PAUT_acquisition
from matplotlib import pyplot as plt

import time 

import pywt

# load data
input_path = "C:/Users/dalmonte/data/ADAMUS/DFKI PAUT/1152811 45 S22 16dB/1152811 45 S22 16dB/45° 2195_"
obj = PAUT_acquisition.PAUT_acquisition(input_path, labelpath="C:/Users/dalmonte/data/ADAMUS/labelling files/240312_M_Adamus_Anzeigen_DFKI_SUB_refined.csv")
ascans = obj.compose_Ascans()[:, :15000, :]

# extract ncrops and apply correction
ncrops = []
t = time.time()
print("Extracting B-scans...", end=" ")
for i in range(ascans.shape[1]):
    b = obj.extract_Bscan(ascans, i, correction=True)
    ncrops.append(obj.extract_ncrop(b))
print(f"took {time.time()-t:.2f} seconds.")
ncrops = np.array(ncrops)
print(ncrops.shape)
t = time.time()
print("Applying Gaussian filter + subsampling...", end=" ")
ncrops_gauss = gaussian_filter(ncrops, sigma=(2., 2., 2.))
ncrops_gauss_sub = []
for b in range(ncrops_gauss.shape[0]):
    ncrops_gauss_sub.append(cv.resize(ncrops_gauss[b], dsize=None, fx=0.2, fy=0.2, interpolation=cv.INTER_AREA))
ncrops_gauss_sub = np.array(ncrops_gauss_sub)
print(f"took {time.time()-t:.2f} seconds.")


# plot some examples
fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(6,8), dpi=150, tight_layout=True)
random_idxs = np.random.choice(np.arange(0, ncrops_gauss_sub.shape[0]), 5)
for i, j in enumerate(random_idxs):
    ax[i].imshow(ncrops_gauss_sub[j], aspect='equal', cmap='jet', vmin=0, vmax=100)
plt.show()

# filter out x-sequences with low amplitude
ncrops_gauss_sub_mask = ncrops_gauss_sub.max(axis=0) > 5.

# plot mask
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,4), dpi=150, tight_layout=True)
ax[0].imshow(ncrops_gauss_sub.max(axis=0), aspect='equal', cmap='jet')
ax[1].imshow(ncrops_gauss_sub_mask, aspect='equal', cmap='gray')
plt.show()

# indexes to access the filtered data
idxs1, idxs2 = np.where(ncrops_gauss_sub_mask) # or ncrops_gauss_mask
n_x_seqs = len(idxs1)
print(n_x_seqs)

# plot some examples of filtered data
fig1, ax1 = plt.subplots(nrows=6, ncols=1, figsize=(8,7), dpi=150, tight_layout=True)
for i, j in enumerate(np.random.choice(np.arange(0, n_x_seqs), 6)):
    ax1[i].plot(ncrops_gauss_sub[:, idxs1[j], idxs2[j]])
    #ax1[i].set_ylim(0, 100)
    ax1[i].set_title(f"t={obj.idx_to_tmm(idxs1[j]*4):.2f}, y={obj.idx_to_ymm(idxs2[j]*4):.2f}", fontsize=8)

fig2, ax2 = plt.subplots(nrows=6, ncols=1, figsize=(4,7), dpi=150, tight_layout=True)
for i, j in enumerate(np.random.choice(np.arange(0, n_x_seqs), 6)):
    ax2[i].hist(ncrops_gauss_sub[:, idxs1[j], idxs2[j]].flatten(), bins=25)
    ax2[i].set_title(f"t={obj.idx_to_tmm(idxs1[j]*4):.2f}, y={obj.idx_to_ymm(idxs2[j]*4):.2f}", fontsize=8)
plt.show()


itop_ncrop = obj.tmm_to_idx(obj.ncrop["top_mm"])
ileft_ncrop = obj.ymm_to_idx(obj.ncrop["left_mm"])

for i1, i2 in zip(idxs1, idxs2):

    i1mm = obj.idx_to_tmm(i1*5 + itop_ncrop)*100
    i2mm = obj.idx_to_ymm(i2*5 + ileft_ncrop)*100
    np.save(f"C:/users/dalmonte/data/temp/ncrops_{i1mm.astype(int)}_{i2mm.astype(int)}.npy", ncrops_gauss_sub[:, i1, i2])
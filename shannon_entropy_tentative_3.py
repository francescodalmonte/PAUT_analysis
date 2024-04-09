import os
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2 as cv
from PAUT_Data import PAUT_Data
from matplotlib import pyplot as plt
from plots_utils import plot_Cscan, plot_Dscan, plot_Ascan

import time 

import pywt

# load data
input_path = "C:/Users/dalmonte/data/ADAMUS/DFKI PAUT/1156722_NI6_M1LF45°_16dB/1156722_NI6_M1LF45°_16dB/45° 2195_"
obj = PAUT_Data(input_path, labelpath="C:/Users/dalmonte/data/ADAMUS/labelling files/240312_M_Adamus_Anzeigen_DFKI_SUB_refined.csv")
ascans = obj.compose_Ascans()[:, :, :]

labels = obj.get_labelsFromCSV()

# extract B-scans and apply correction
bscans = []
t = time.time()
print("Extracting B-scans...", end=" ")
for i in range(ascans.shape[1]):
    bscans.append(obj.extract_Bscan(ascans, i, correction=True))
print(f"took {time.time()-t:.2f} seconds.")
bscans = np.array(bscans)
print(bscans.shape)
t = time.time()


# plot some examples
fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(6,8), dpi=150, tight_layout=True)
random_idxs = np.random.choice(np.arange(0, bscans.shape[0]), 5)
for i, j in enumerate(random_idxs):
    ax[i].imshow(bscans[j], aspect='equal', cmap='jet', vmin=0, vmax=100)
plt.show()

# filter out x-sequences with low amplitude
bscans_gauss_mask = bscans.max(axis=0) > 5.

# plot mask
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,4), dpi=150, tight_layout=True)
ax[0].imshow(bscans.max(axis=0), aspect='equal', cmap='jet')
ax[1].imshow(bscans_gauss_mask, aspect='equal', cmap='gray')

y1, t1 = labels["Y-Pos."].values[0]+labels["Breite b"].values[0]/2, labels["Z-Pos."].values[0]+labels["Tiefe t"].values[0]/2 
y2, t2 = labels["Y-Pos."].values[1]+labels["Breite b"].values[1]/2, labels["Z-Pos."].values[1]+labels["Tiefe t"].values[1]/2

print(f"y1: {y1}, t1: {t1}")
print(f"y2: {y2}, t2: {t2}")
ax[1].plot(obj.ymm_to_idx(np.array([y1, y2])), obj.tmm_to_idx(np.array([t1, t2])), 'o', color='red')
plt.show()

# extract nugget crops
top, bottom = obj.tmm_to_idx(np.array([-0.5, labels["Tiefe Ende"].values[0]]))
left, right = obj.ymm_to_idx(np.array([-10.5, 10.5]))
ncrops = bscans[:, top:bottom, left:right]

print("Applying Gaussian filter + subsampling to nugget crops...", end=" ")
ncrops = gaussian_filter(ncrops, sigma=1., axes=(1,2))
ncrops_sub = []
for b in range(ncrops.shape[0]):
    ncrops_sub.append(cv.resize(ncrops[b], dsize=None, fx=0.25, fy=0.25, interpolation=cv.INTER_AREA))
ncrops = np.array(ncrops_sub)
ncrops = ncrops[:, 0:np.where(np.not_equal(ncrops[0], ncrops[0]))[0].min(), :] # remove nan pixels
print(f"took {time.time()-t:.2f} seconds.")

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(12,3), dpi=150, tight_layout=True)
for i, a in enumerate(ax):
    a.imshow(ncrops[i], aspect='equal', cmap='jet', vmin=0, vmax=100)

plt.show()

# save crops to file as png images
#for i in range(ncrops.shape[0]):
#    cv.imwrite(f"C:/users/dalmonte/data/nugget_crops_1152811 45 S22 16dB/ncrop_{i}.png", ncrops[i]*255/100)


# max (baseline)
maxs = ncrops.max(axis=(1,2))


# compute mses between crops and mean crop
ncrops_mean = ncrops.mean(axis=0)
mses = []
for i in range(ncrops.shape[0]):
    mses.append(np.mean((ncrops[i] - ncrops_mean)**2))
mses = np.array(mses)

# compute cosine similarity
ncrops_sub = []
for b in range(ncrops.shape[0]):
    ncrops_sub.append(cv.resize(ncrops[b], dsize=None, fx=0.25, fy=0.25, interpolation=cv.INTER_AREA))
ncrops_sub = np.array(ncrops_sub)
ncrops_sub_mean = ncrops_sub.mean(axis=0)
norm_ncrops_sub = np.linalg.norm(ncrops_sub, axis=(1,2))
norm_ncrops_sub_mean = np.linalg.norm(ncrops_sub_mean)

cosims = np.sum(ncrops_sub*ncrops_sub_mean, axis=(1,2))/(norm_ncrops_sub*norm_ncrops_sub_mean)


# plot mses
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12,6), dpi=150, tight_layout=True)
ax[0].plot(obj.idx_to_xmm(np.arange(0, maxs.shape[0])), maxs, c='tab:orange'); ax[0].set_title("maxs (baseline)")
ax[1].plot(obj.idx_to_xmm(np.arange(0, mses.shape[0])), mses); ax[1].set_title("mses")
ax[2].plot(obj.idx_to_xmm(np.arange(0, cosims.shape[0])), cosims); ax[2].set_title("cosims")
plt.show()

# plot nugget crops with highest mses
sorted_idxs = np.argsort(mses)[::-1]
print(obj.idx_to_xmm(sorted_idxs)[:50])
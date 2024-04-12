import os
import numpy as np
from scipy.ndimage import gaussian_filter1d, zoom, gaussian_filter, convolve1d
import cv2 as cv
from PAUT_Data import PAUT_Data
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from plots_utils import plot_Cscan, plot_Dscan, plot_Ascan

import time 


# load data and labels
input_path = "C:/Users/dalmonte/data/ADAMUS/DFKI PAUT/1152811 45 S22 16dB/1152811 45 S22 16dB/45° 2195_"
obj = PAUT_Data(input_path, labelpath="C:/Users/dalmonte/data/ADAMUS/labelling files/240312_M_Adamus_Anzeigen_DFKI_SUB_refined.csv")
ascans = obj.compose_Ascans()[:, 2600:3200, :]

labels = obj.get_labelsFromCSV()

# extract B-scans and apply correction
bscans = []
t = time.time()
print("Extracting B-scans...", end=" ")
for i in range(ascans.shape[1]):
    bscans.append(obj.extract_Bscan(ascans, i, correction=True))
print(f"took {time.time()-t:.2f} seconds.")
bscans = np.array(bscans)
bshape = bscans.shape
print(f"Bscans shape: {bshape}")
t = time.time()


# plot some examples
fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(6,8), dpi=150, tight_layout=True)
random_idxs = np.random.choice(np.arange(0, bscans.shape[0]), 5)
for i, j in enumerate(random_idxs):
    ax[i].imshow(bscans[j], aspect='equal', cmap='jet', vmin=0, vmax=100)
plt.show()

# filter out x-sequences with low amplitude
bscans_gauss_mask = bscans.max(axis=0) > 2.

# nugget crops
top, bottom = obj.tmm_to_idx(np.array([-0.5, labels["Tiefe Ende"].values[0]]))
axis = obj.ymm_to_idx(0.)
surface = obj.tmm_to_idx(0.)

left, right = obj.ymm_to_idx(np.array([-10.5, 10.5]))

# plot mask
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,4), dpi=150, tight_layout=True)
ax[0].imshow(bscans.max(axis=0), aspect='equal', cmap='jet')
ax[1].imshow(bscans_gauss_mask, aspect='equal', cmap='gray')
# plot one defect as example
y1, t1 = labels["Y-Pos."].values[0]+labels["Breite b"].values[0]/2, labels["Z-Pos."].values[0]+labels["Tiefe t"].values[0]/2 
ax[1].plot(obj.ymm_to_idx(np.array([y1])), obj.tmm_to_idx(np.array([t1])), 'o', color='red')
# plot nugget crop
ax[0].add_patch(patches.Rectangle((left, top), (right-left), (bottom-top), linewidth=1.5, edgecolor='r', facecolor='none'))
ax[0].axvline(axis, ymin = (bshape[1]-bottom)/bshape[1], ymax = (bshape[1]-surface)/bshape[1], color='red', linestyle='--', linewidth=1)
ax[0].axhline(surface, xmin = left/bshape[2], xmax = right/bshape[2], color='red', linestyle='--', linewidth=1)
ax[1].add_patch(patches.Rectangle((left, top), (right-left), (bottom-top), linewidth=1.5, edgecolor='r', facecolor='none'))
ax[1].axvline(axis, ymin = (bshape[1]-bottom)/bshape[1], ymax = (bshape[1]-surface)/bshape[1], color='red', linestyle='--', linewidth=1)
ax[1].axhline(surface, xmin = left/bshape[2], xmax = right/bshape[2], color='red', linestyle='--', linewidth=1)
plt.show()

# extract nugget crops
ncrops = bscans[:, top:bottom, left:right]
ncrops_mask = bscans_gauss_mask[top:bottom, left:right]
print(f"ncrops shape: {ncrops.shape}")
print(f"ncrops_mask shape: {ncrops_mask.shape}") 
nanheight = np.where(np.not_equal(ncrops[0], ncrops[0]))[0].min()
ncrops = ncrops[:, 0:nanheight, :] # remove nan pixels
ncrops_mask = ncrops_mask[0:nanheight, :] # remove nan pixels from mask


fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(12,3), dpi=150, tight_layout=True)
for i, a in enumerate(ax):
    if i == 0:
        a.imshow(ncrops_mask, aspect='equal', cmap='gray')
    else:
        a.imshow(ncrops[i], aspect='equal', cmap='jet', vmin=0, vmax=100)

plt.show()

# COMPUTE METRICS
ncrops_sub = []
for b in range(ncrops.shape[0]):
    ncrops_sub.append(cv.resize(ncrops[b], dsize=None, fx=0.4, fy=0.4, interpolation=cv.INTER_LINEAR))
ncrops = np.array(ncrops_sub)
ncrops_mask = cv.resize(ncrops_mask.astype(np.uint8), dsize=None, fx=0.4, fy=0.4, interpolation=cv.INTER_NEAREST).astype(bool)

print(f"ncrops shape after resizing: {ncrops.shape}")
print(f"ncrops_mask shape after resizing: {ncrops_mask.shape}") 

t = time.time()
print("gaussian filtering...", end=" ")
ncrops_mean = gaussian_filter1d(ncrops, sigma=100., axis=0, truncate=3, mode = "reflect")
print(f"took {time.time()-t:.2f} seconds.")

# convolve with custom kernel 1
t = time.time()
print("gaussian filtering (2)...", end=" ")
kernel1 = np.ones(101)
kernel1[49:52] = 0.
kernel1 = kernel1/np.sum(kernel1)
ncrops_mean_flat = convolve1d(ncrops, kernel1, axis=0, mode = "reflect", origin=0)
print(f"took {time.time()-t:.2f} seconds.")

# convolve with custom kernel 2
t = time.time()
print("filtering with custom kernel...", end=" ")
kernel2 = np.zeros(200)
kernel2[60], kernel2[140] = .5, .5
kernel2 = gaussian_filter1d(kernel2, sigma=15., axis=0, truncate=4, mode = "reflect")
ncrops_mean_bigaus = convolve1d(ncrops, kernel2, axis=0, mode = "reflect", origin=0)
print(f"took {time.time()-t:.2f} seconds.")


# max (baseline)
maxs = ncrops.max(axis=(1,2))

# compute mses between crops and mean crop
mses = []
mses_bigaus = []
mses_flat = []
for i in range(ncrops.shape[0]):
    x = ncrops[i][ncrops_mask]
    x_gaus = ncrops_mean[i][ncrops_mask]
    x_bigaus = ncrops_mean_bigaus[i][ncrops_mask]
    x_flat = ncrops_mean_flat[i][ncrops_mask]

    mses.append(np.mean((x - x_gaus)**2))
    mses_bigaus.append(np.mean((x - x_bigaus)**2))
    mses_flat.append(np.mean((x - x_flat)**2))
mses = np.array(mses)
mses_bigaus = np.array(mses_bigaus)
mses_flat = np.array(mses_flat)

# compute mxses between crops and mean crop
mxses = []
mxses_bigaus = []
mxses_flat = []
for i in range(ncrops.shape[0]):
    x = ncrops[i][ncrops_mask]
    x_gaus = ncrops_mean[i][ncrops_mask]
    x_bigaus = ncrops_mean_bigaus[i][ncrops_mask]
    x_flat = ncrops_mean_flat[i][ncrops_mask]

    mxses.append(np.max((x - x_gaus)**2))
    mxses_bigaus.append(np.max((x - x_bigaus)**2))
    mxses_flat.append(np.max((x - x_flat)**2))
mxses = np.array(mxses)
mxses_bigaus = np.array(mxses_bigaus)
mxses_flat = np.array(mxses_flat)

# compute cosine similarity
cosims = []
cosims_bigaus = []
cosims_flat = []
for i in range(ncrops.shape[0]):
    x = ncrops[i][ncrops_mask]
    x_gaus = ncrops_mean[i][ncrops_mask]
    x_bigaus = ncrops_mean_bigaus[i][ncrops_mask]
    x_flat = ncrops_mean_flat[i][ncrops_mask]

    norm_x = np.linalg.norm(x)
    norm_x_gaus = np.linalg.norm(x_gaus)
    norm_x_bigaus = np.linalg.norm(x_bigaus)
    norm_x_flat = np.linalg.norm(x_flat)
    
    cosims.append(1.-np.sum(x*x_gaus)/(norm_x*norm_x_gaus))
    cosims_bigaus.append(1.-np.sum(x*x_bigaus)/(norm_x*norm_x_bigaus))
    cosims_flat.append(1.-np.sum(x*x_flat)/(norm_x*norm_x_flat))
cosims = np.array(cosims)
cosims_bigaus = np.array(cosims_bigaus)
cosims_flat = np.array(cosims_flat)


# plot metrics
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(12,10), dpi=150, tight_layout=True)
ax[0].plot(obj.idx_to_xmm(np.arange(0, maxs.shape[0])), maxs, c='tab:orange')
ax[0].set_title("maxs (baseline)")
ax[1].plot(obj.idx_to_xmm(np.arange(0, mses.shape[0])), mses, c='tab:blue', label='gaus')
ax[1].plot(obj.idx_to_xmm(np.arange(0, mses_bigaus.shape[0])), mses_bigaus, c='tab:green', label='bigaus')
ax[1].plot(obj.idx_to_xmm(np.arange(0, mses_flat.shape[0])), mses_flat, c='tab:red', label='flat')
ax[1].set_title("mses")
ax[1].legend()
ax[2].plot(obj.idx_to_xmm(np.arange(0, mxses.shape[0])), mxses, c='tab:blue', label='gaus')
ax[2].plot(obj.idx_to_xmm(np.arange(0, mxses_bigaus.shape[0])), mxses_bigaus, c='tab:green', label='bigaus')
ax[2].plot(obj.idx_to_xmm(np.arange(0, mxses_flat.shape[0])), mxses_flat, c='tab:red', label='flat')
ax[2].set_title("mxses")
ax[2].legend()
ax[3].plot(obj.idx_to_xmm(np.arange(0, cosims.shape[0])), cosims, c='tab:blue', label='gaus') 
ax[3].plot(obj.idx_to_xmm(np.arange(0, cosims_bigaus.shape[0])), cosims_bigaus, c='tab:green', label='bigaus')
ax[3].plot(obj.idx_to_xmm(np.arange(0, cosims_flat.shape[0])), cosims_flat, c='tab:red', label='flat')
ax[3].set_title("cosims")
ax[3].legend()

plt.show()

# plot nugget crops with highest cosims
sorted_idxs = np.argsort(cosims)[::-1]
print(f"X positions of 25 worst anomalies: {obj.idx_to_xmm(sorted_idxs)[:25]}")



# select anomalous crops
anomalous_ncrops = []
anomalous_ncrops_mean = []
for i in range(100):
    anomalous_ncrops.append(ncrops[sorted_idxs[i]])
    anomalous_ncrops_mean.append(ncrops_mean_bigaus[sorted_idxs[i]])
anomalous_ncrops = np.array(anomalous_ncrops)
anomalous_ncrops_mean = np.array(anomalous_ncrops_mean)

# local cosine similarity

def local_cosine_similarity(img1, img2, window_size=16, stride=1, upsample=True, mask = None):
    assert img1.shape == img2.shape
    assert window_size % 2 == 0
    # add a small value to avoid division by zero
    img1 = img1 + .1
    img2 = img2 + .1

    # adjust img resolution so that no padding is needed
    xres = ((img1.shape[1]-window_size)//stride)*stride+window_size
    yres = ((img1.shape[0]-window_size)//stride)*stride+window_size
    img1 = cv.resize(img1, dsize=(xres, yres), interpolation=cv.INTER_AREA)
    img2 = cv.resize(img2, dsize=(xres, yres), interpolation=cv.INTER_AREA)

    imgshape = img1.shape
    y1_set = np.linspace(window_size//2, yres-window_size//2, (yres-window_size)//stride, dtype=np.int32)
    x1_set = np.linspace(window_size//2, xres-window_size//2, (xres-window_size)//stride, dtype=np.int32)

    cosims = []
    for y in y1_set:
        cosims_row = []
        for x in x1_set:
            if mask is not None:
                amask = mask[y:y+window_size, x:x+window_size]
                bmask = mask[y:y+window_size, x:x+window_size]
            else:
                amask = slice(None)
                bmask = slice(None)
            a = img1[y:y+window_size, x:x+window_size][amask]
            b = img2[y:y+window_size, x:x+window_size][bmask]
            if a.size == 0 or b.size == 0:
                cosims_row.append(0.)
                continue
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            c = 1.-np.sum(a*b)/(norm_a*norm_b)
            print(f"c: {c}")
            cosims_row.append(c)
        cosims.append(np.array(cosims_row))
    cosims = np.array(cosims)
    
    # upsample to match original image shape
    cosims = cv.resize(cosims, dsize=(imgshape[1], imgshape[0]), interpolation=cv.INTER_LINEAR)
    cosims = gaussian_filter(cosims, sigma=1.)

    return cosims


for i in range(20):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,3), dpi=150, tight_layout=True)
    ax[0].imshow(anomalous_ncrops[i], aspect='equal', cmap='jet')
    ax[0].set_title(f"[{obj.idx_to_xmm(sorted_idxs)[i]:.2f}mm")
    ax[1].imshow(anomalous_ncrops_mean[i], aspect='equal', cmap='jet')
    loc_cosim = np.abs(anomalous_ncrops[i]-anomalous_ncrops_mean[i])#local_cosine_similarity(anomalous_ncrops[i], anomalous_ncrops_mean[i])
    ax[2].imshow(loc_cosim, aspect='equal', cmap='jet')
    plt.show()
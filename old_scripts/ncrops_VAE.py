import os
import numpy as np
import cv2 as cv
from PAUT_Data import PAUT_Data
from matplotlib import pyplot as plt
import matplotlib.patches as patches

import time 


# load data and labels
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
bshape = bscans.shape
print(f"Bscans shape: {bshape}")
t = time.time()

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
ax[0].plot(obj.ymm_to_idx(np.array([y1])), obj.tmm_to_idx(np.array([t1])), 'x', color='red')
ax[1].plot(obj.ymm_to_idx(np.array([y1])), obj.tmm_to_idx(np.array([t1])), 'x', color='red')
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
        a.set_title("mask (fullres)")
    else:
        a.imshow(ncrops[i], aspect='equal', cmap='jet', vmin=0, vmax=100)
        a.set_title(f"example ncrop_{i} (fullres)")
plt.show()


# resize ncrops
ncrops_sub = []
for b in range(ncrops.shape[0]):
    ncrops_sub.append(cv.resize(ncrops[b], dsize=(32, 32), fx=None, fy=None, interpolation=cv.INTER_LINEAR))
ncrops = np.array(ncrops_sub)
ncrops_mask = cv.resize(ncrops_mask.astype(np.uint8), dsize=(32, 32), fx=None, fy=None, interpolation=cv.INTER_NEAREST).astype(bool)

print(f"ncrops shape after resizing: {ncrops.shape}")
print(f"ncrops_mask shape after resizing: {ncrops_mask.shape}") 


fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(12,3), dpi=150, tight_layout=True)
for i, a in enumerate(ax):
    if i == 0:
        a.imshow(ncrops_mask, aspect='equal', cmap='gray')
        a.set_title("mask (resized)")
    else:
        a.imshow(ncrops[i], aspect='equal', cmap='jet', vmin=0, vmax=100)
        a.set_title(f"example ncrop_{i} (resized)")
plt.show()



# AE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class AE(nn.Module):
    def __init__(self,  in_features):
        super().__init__()
        self.in_features = in_features

        self.encoder = nn.Sequential(
            nn.Linear(self.in_features, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.in_features)
        )

    def forward(self, x):
        x_e = self.encoder(x)
        x_d = self.decoder(x_e)
        return x_e, x_d
    
#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# instantiate the model
model = AE(in_features = ncrops.shape[1]**2)
# optimizer object
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# mean-squared error loss
criterion = lambda x, y: (1. - torch.sum(x*y, axis=1)/(torch.linalg.norm(x, axis=1)*torch.linalg.norm(y, axis=1)))



class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        x = self.data[index]
        x = np.sqrt(x.flatten())/10.
        x = torch.tensor(x, dtype=torch.float32)
        
        return x
    
    def __len__(self):
        return len(self.data)



train_ds = MyDataset(ncrops[400:])
val_ds = MyDataset(ncrops[:400])

train_dl = DataLoader(train_ds,
                      batch_size = 64,
                      shuffle = True,
                      num_workers = 0
                      )
val_dl = DataLoader(val_ds,
                    batch_size = 64,
                    shuffle = True,
                    num_workers = 0
                    )

print(next(iter(train_dl)).shape)




epochs = 10
for epoch in range(epochs):
    train_loss = []
    minibatch_len = []
    model.train()
    for inputs in train_dl:
        optimizer.zero_grad()
        
        # compute reconstructions
        encoding, outputs = model(inputs)
        
        # compute training reconstruction loss
        print(outputs.shape)
        print(inputs.shape)
        minibatch_loss = criterion(outputs, inputs).mean()
        print(minibatch_loss.shape)
        minibatch_loss.backward()
        optimizer.step()

        train_loss.append(minibatch_loss.item())
        minibatch_len.append(len(inputs))

    loss = np.sum(np.multiply(train_loss, minibatch_len)) / np.sum(minibatch_len)
    print(f"EPOCH {epoch+1}: Train loss = {loss:.9f} ----", end=" ")

    # validation step
    model.eval()
    with torch.no_grad():
        val_loss = []
        minibatch_len = []
        for inputs in val_dl:
            encoding, outputs = model(inputs)
            
            minibatch_loss = criterion(outputs, inputs).mean()
            val_loss.append(minibatch_loss.item())
            minibatch_len.append(len(inputs))
        loss = np.sum(np.multiply(val_loss, minibatch_len)) / np.sum(minibatch_len)

        print(f"Val loss = {loss:.9f}")


# plot examples from the train set
model.eval()
with torch.no_grad():
    for inputs in train_dl:
        encoding, outputs = model(inputs)
        break
fig, ax = plt.subplots(nrows=2, ncols=12, figsize=(12,4), dpi=150, tight_layout=True)
for i in range(12):
    ax[0, i].set_title(f"mse = {criterion(inputs[i], outputs[i]):.3f}", fontsize=6)
    ax[0, i].imshow(inputs[i].view(32, 32).cpu().numpy(), aspect='equal', cmap='jet')
    ax[0, i].axis('off')
    ax[1, i].imshow(outputs[i].view(32, 32).cpu().numpy(), aspect='equal', cmap='jet')
    ax[1, i].axis('off')

plt.show()

# compute scores and embeddings for the validation set
with torch.no_grad():
    val_loss = []
    embeddings = []
    for inputs in val_dl:
        encoding, outputs = model(inputs)
        loss = criterion(outputs, inputs)
        print(loss.shape)
        val_loss.append(loss.item())
        embeddings.append(encoding)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6), dpi=150, tight_layout=True)
ax.plot(val_loss, label='Validation loss')

plt.show()
import os
import numpy as np
import cv2 as cv
from PAUT_preprocessing.PAUT_acquisition import PAUT_acquisition
from matplotlib import pyplot as plt
import matplotlib.patches as patches

import time 


# load data and labels
input_path = "C:/Users/dalmonte/data/ADAMUS/DFKI PAUT/1156722_NI6_M1LF45°_16dB/1156722_NI6_M1LF45°_16dB/45° 2195_"
obj = PAUT_acquisition(input_path, labelpath="C:/Users/dalmonte/data/ADAMUS/labelling files/240312_M_Adamus_Anzeigen_DFKI_SUB_refined.csv")
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
ndefects = len(labels["X-Pos."])
print(f"Found {ndefects} defects")
x_positions = obj.xmm_to_idx(labels["X-Pos."].values)
x_lengths = obj.xmm_to_idx(labels["Länge l"].values)
print(f"X-positions: {x_positions}, X-lengths: {x_lengths}")
anomalous_idxs = []
for i in range(ndefects):
    anomalous_idxs.append(np.arange(x_positions[i]-x_lengths[i]//2, x_positions[i]+x_lengths[i]//2))
anomalous_idxs = np.unique(np.array(anomalous_idxs))
print(f"Anomalous idxs: {anomalous_idxs}")

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
#plt.show()


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
#plt.show()


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
#plt.show()



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
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 128),
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


random_idxs = range(len(ncrops))
train_idxs = []
val_idxs = []
for i, idx in enumerate(random_idxs):
    if i < 400:
        if idx not in anomalous_idxs:
            train_idxs.append(idx)
        else:
            val_idxs.append(idx)
    else:
        val_idxs.append(idx)

train_ds = MyDataset(ncrops[train_idxs])
val_ds = MyDataset(ncrops[val_idxs])
print(f"Train set size: {len(train_ds)}")
print(f"Validation set size: {len(val_ds)}")

train_dl = DataLoader(train_ds,
                      batch_size = 128,
                      shuffle = True,
                      num_workers = 0
                      )
val_dl = DataLoader(val_ds,
                    batch_size = 16,
                    shuffle = False,
                    num_workers = 0
                    )

print(next(iter(train_dl)).shape)




epochs = 1000
val_loss_set = []
train_loss_set = []
for epoch in range(epochs):
    train_loss = []
    minibatch_len = []
    model.train()
    for inputs in train_dl:
        optimizer.zero_grad()
        
        # compute reconstructions
        encoding, outputs = model(inputs)
        
        # compute training reconstruction loss (with online hard example mining)
        if epoch < np.inf:
            N = len(inputs)
        else:
            N = int(len(inputs)*0.50)
        minibatch_loss = criterion(outputs, inputs)
        minibatch_loss = torch.topk(minibatch_loss, k=N, largest=False).values.mean()

        minibatch_loss.backward()
        optimizer.step()

        train_loss.append(minibatch_loss.item())
        minibatch_len.append(len(inputs))

    loss = np.sum(np.multiply(train_loss, minibatch_len)) / np.sum(minibatch_len)
    train_loss_set.append(loss)
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
        val_loss_set.append(loss)

        print(f"Val loss = {loss:.9f}")


# plot examples from the train set
model.eval()
with torch.no_grad():
    for inputs in train_dl:
        encoding, outputs = model(inputs)
        break
    mses = criterion(inputs, outputs).cpu().numpy()
fig, ax = plt.subplots(nrows=2, ncols=12, figsize=(12,3), dpi=150, tight_layout=True)
fig.suptitle("Train set examples")
for i in range(12):
    print(inputs[i].shape, outputs[i].shape)
    ax[0, i].set_title(f"mse = {mses[i]:.3f}", fontsize=6)
    ax[0, i].imshow(inputs[i].view(32, 32).cpu().numpy(), aspect='equal', cmap='jet')
    ax[0, i].axis('off')
    ax[1, i].imshow(outputs[i].view(32, 32).cpu().numpy(), aspect='equal', cmap='jet')
    ax[1, i].axis('off')


# plot examples from the validation set
model.eval()
all_encodings = []
all_outputs = []
all_inputs = []
all_anomaly_scores = []
with torch.no_grad():
    for inputs in val_dl:
        encoding, outputs = model(inputs)
        all_encodings.append(encoding.cpu().numpy())
        all_outputs.append(outputs.cpu().numpy())
        all_inputs.append(inputs.cpu().numpy())
        loss = criterion(outputs, inputs).cpu().numpy()
        all_anomaly_scores.append(loss)

for j in range(len(all_inputs)):
    fig, ax = plt.subplots(nrows=3, ncols=len(all_inputs[j]), figsize=(16,3.5), dpi=150, tight_layout=True)
    fig.suptitle("Validation set results")
    print(f"Batch {j} (len={len(all_inputs[j])})")
    for i in range(len(all_inputs[j])):
        is_anomalous = val_idxs[j*len(all_inputs[j])+i] in anomalous_idxs
        title = f"ascore = {all_anomaly_scores[j][i]:.3f}" + ("(A**)" if is_anomalous else "(n)")
        ax[0, i].set_title(title, fontsize=6)
        ax[0, i].imshow(all_inputs[j][i].reshape(32, 32), aspect='equal', cmap='jet')
        ax[0, i].axis('off')
        ax[1, i].imshow(all_outputs[j][i].reshape(32, 32), aspect='equal', cmap='jet')
        ax[1, i].axis('off')
        # compute anomaly heatmap
        anomaly_heatmap = ((all_inputs[j][i] - all_outputs[j][i])**2).reshape(32, 32)
        # blur the heatmap
        anomaly_heatmap = cv.GaussianBlur(anomaly_heatmap, (5, 5), 0)
        ax[2, i].imshow(anomaly_heatmap, aspect='equal', cmap='jet', vmin=0., vmax=.3)

    #plt.savefig(f"C:/Users/dalmonte/data/ADAMUS/tempres/validation_set_batch{j}.png")
    plt.close()




fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6), dpi=150, tight_layout=True)
ax.plot(train_loss_set, label='Train loss', c='tab:blue')
ax.plot(val_loss_set, label='Validation loss', c='tab:orange')
ax.set_title("train vs val losses")


# plot roc curve
from sklearn.metrics import roc_curve, roc_auc_score
anomaly_scores = np.concatenate(all_anomaly_scores)
labels = np.array([1 if idx in anomalous_idxs else 0 for idx in val_idxs])

fpr, tpr, thresholds = roc_curve(labels, anomaly_scores)
auc = roc_auc_score(labels, anomaly_scores)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6), tight_layout=True)
ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
ax.set_title("ROC curve")
ax.legend()


plt.show()



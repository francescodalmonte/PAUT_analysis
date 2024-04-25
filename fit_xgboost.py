import os
import numpy as np
from PAUT_Data import PAUT_Data
from matplotlib import pyplot as plt

from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
import xgboost as xgb

# import Ascans (timeseries data)
dirpath = f"C:/Users/dalmonte/data/ADAMUS/DFKI PAUT/1152811 45 S22 16dB/1152811 45 S22 16dB/45° 2195_"
obj = PAUT_Data(dirpath)
ascans = obj.compose_Ascans()
ashape = ascans.shape
ascans = ascans.reshape(-1, ashape[2])
ascans = ascans[np.random.choice(ascans.shape[0], 200000, replace=False), :]
print(ascans.shape)

# train test split
ascans_train, ascans_test = train_test_split(ascans, test_size=0.1)

# prepare data an labels for regression
def prepare_ds(data, input_length, target_length, step_size):
    """
    """
    print("Preparing ds for regression...")
    X, y = [], []
    for sequence in data:
        sequence = gaussian_filter1d(sequence, 1.)
        for i in range(0, len(sequence)-input_length-target_length, step_size):
            X.append(sequence[i : i+input_length])
            y.append(sequence[i+input_length : i+input_length+target_length])
    
    return np.array(X), np.array(y)


x_train, y_train = prepare_ds(ascans_train, 60, 8, 7)
x_test, y_test = prepare_ds(ascans_test, 60, 8, 7)

print(x_train.shape, y_train.shape)
fig, ax = plt.subplots(figsize=(5,2), dpi=150)
ax.plot(np.concatenate((x_train[0], y_train[0]), axis=0), color='tab:blue')
ax.axvline(60, color='black', linestyle='--')
plt.show()

# train xgboost model
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)
params = {
    'eta': 0.05,
    'max_depth': 7,
    'subsample': 0.75,
    'objective': 'reg:squarederror'
}
num_round = 250
bst = xgb.train(params,
                dtrain,
                num_round,
                evals=[(dtrain, 'dtrain'), (dtest, 'dtest')],
                verbose_eval=True
                )
# predict
preds = bst.predict(dtest)
# show predictions


fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(5,10), dpi=150, tight_layout=True)
r = np.random.randint(0, x_test.shape[0], 5)
for i in range(5):
    ax[i].plot(np.concatenate((x_test[r[i]], y_test[r[i]]), axis=0), color='tab:blue')
    ax[i].plot(np.concatenate((np.repeat(np.nan, 60), preds[r[i]]), axis=0), color='tab:orange')
    ax[i].axvline(60, color='black', linestyle='--')
    ax[i].set_ylim(0, 100)
plt.show()

mask_data = np.max(np.concatenate((x_test, y_test), axis=1), axis=(1)) > 20
fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(5,10), dpi=150, tight_layout=True)
r = np.random.randint(0, x_test[mask_data].shape[0], 5)
for i in range(5):
    ax[i].plot(np.concatenate((x_test[mask_data][r[i]], y_test[mask_data][r[i]]), axis=0), color='tab:blue')
    ax[i].plot(np.concatenate((np.repeat(np.nan, 60), preds[mask_data][r[i]]), axis=0), color='tab:orange')
    ax[i].axvline(60, color='black', linestyle='--')
    ax[i].set_ylim(0, 100)
plt.show()

mask_data = np.max(np.concatenate((x_test, y_test), axis=1), axis=(1)) > 40
fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(5,10), dpi=150, tight_layout=True)
r = np.random.randint(0, x_test[mask_data].shape[0], 5)
for i in range(5):
    ax[i].plot(np.concatenate((x_test[mask_data][r[i]], y_test[mask_data][r[i]]), axis=0), color='tab:blue')
    ax[i].plot(np.concatenate((np.repeat(np.nan, 60), preds[mask_data][r[i]]), axis=0), color='tab:orange')
    ax[i].axvline(60, color='black', linestyle='--')
    ax[i].set_ylim(0, 100)
plt.show()
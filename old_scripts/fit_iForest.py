import os
import numpy as np
from PAUT_Data import PAUT_Data
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# import Ascans (timeseries data)
dirpath = f"C:/Users/dalmonte/data/ADAMUS/DFKI PAUT/1152811 45 S22 16dB/1152811 45 S22 16dB/45° 2195_"
obj = PAUT_Data(dirpath)
ascans = obj.compose_Ascans()
ashape = ascans.shape

# create labels
ascans = ascans.reshape(-1, ashape[2])
labels = np.zeros(ascans.shape[0])

# pca for dimensionality reduction (optional)
pca = PCA(n_components=4)
pca.fit(ascans)

explained_variance = (pca.explained_variance_ratio_)
print(f"Explained variance: {explained_variance}")

ascans_pca = pca.transform(ascans)
r = np.random.randint(0, ascans_pca.shape[0], 1000)
plt.scatter(ascans_pca[r][:, 0], ascans_pca[r][:, 1], c=labels[r], cmap='jet')
plt.show()


# train test split
X_train, X_test, y_train, y_test = train_test_split(ascans, labels, test_size=0.1)

# run isolation forest
clf = IsolationForest(max_features=4, max_samples=1000, random_state=0, bootstrap=True, verbose=True)
clf.fit(pca.transform(X_train))
predictions = clf.predict(pca.transform(X_test))

# show examples
normal = X_test[predictions == 1]
print(f"N. normal samples: {normal.shape[0]}")
choice_n = np.random.randint(0, normal.shape[0], 8)
anomalous = X_test[predictions == -1]
print(f"N. anomalous samples: {anomalous.shape[0]}")
choice_a = np.random.randint(0, anomalous.shape[0], 8)

fig, ax = plt.subplots(ncols=2, nrows=8, figsize=(8,16), dpi=150, tight_layout=True)
for i in range(8):
    ax[i, 0].plot(X_test[choice_n[i]], color='tab:blue')
    ax[i, 1].plot(X_test[choice_a[i]], color='tab:red')

plt.show()


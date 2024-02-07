import os
import json
import numpy as np
from matplotlib import pyplot as plt


rootpath = "D:/data_ADAMUS/ADAMUS/DFKI_PAUT"

N_L = []
N_P = []
N_T = []
unique_vals = []

for d in os.listdir(rootpath):  
    if (not d.endswith(".zip")) and (not d.endswith(".ps1")) and (not d.endswith(".json")):
        print(d)
        names = os.listdir(os.path.join(rootpath, f"{d}/{d}"))
        if len(names)>0:
            for n in names:
                dirpath = os.path.join(rootpath, f"{d}/{d}/{n}")
                # Opening JSON file
                with open(os.path.join(dirpath, "DATASET_INFO.json"), "r") as f:
                    data = json.load(f)
                    # store values
                    N_L.append(int(data["info"]["N_Lelements"]))
                    N_P.append(int(data["info"]["N_positons"]))
                    N_T.append(int(data["info"]["N_timeSteps"]))
                    unique_vals.append(int(data["info"]["unique_amplitude_values"]))

N_L = np.array(N_L)
N_P = np.array(N_P)
N_T = np.array(N_T)
unique_vals = np.array(unique_vals)

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10,3), tight_layout=True)
ax[0].hist(N_L, 12); ax[0].set_title("N_L")
print(f"N_L: {np.min(N_L)} - {np.max(N_L)}")
ax[1].hist(N_P, 12); ax[1].set_title("N_P")
print(f"N_P: {np.min(N_P)} - {np.max(N_P)}")
ax[2].hist(N_T, 12); ax[2].set_title("N_T")
print(f"N_T: {np.min(N_T)} - {np.max(N_T)}")
ax[3].hist(unique_vals, 12); ax[3].set_title("unique_vals")
print(f"uvals: {np.min(unique_vals)} - {np.max(unique_vals)}")

plt.show()
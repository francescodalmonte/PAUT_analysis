import os
import numpy as np
from PAUT_Data import PAUT_Data
from matplotlib import pyplot as plt
from plots_utils import plot_Cscan, plot_Dscan, plot_Ascan


input_path = "C:/Users/dalmonte/data/ADAMUS/DFKI PAUT/FA3567883_NI4_45°/FA3567883_NI4_45°/45° 2195_"

obj = PAUT_Data(input_path)
ascans = obj.compose_Ascans()[:, 190:1760]
cscan = ascans.max(axis=2)[:, 190:1760]

fig = plt.figure(figsize=(6,6), dpi=150)
ax1 = fig.add_subplot(211); ax2 = fig.add_subplot(212)
ax1.imshow(cscan, aspect='auto', cmap='jet')
ax2.hist(cscan.flatten(), bins=25, color='tab:blue', density=True)
ax2twin = ax2.twinx()
ax2twin.hist(cscan.flatten(), bins=25, color='tab:orange', cumulative=True, density=True, histtype='step')
ax2twin.axhline(0.9, color='tab:orange', linestyle='--', linewidth=1)
ax2.set_yscale('log')
plt.show()


fig = plt.figure(figsize=(8,4), dpi=150)
ax = fig.add_subplot(111)
plot_Dscan(ascans, 65, ax, title="D-scan", cmap='jet', vmin=0, vmax=100)

plt.show()


fig = plt.figure(figsize=(8,4), dpi=150, tight_layout=True)
ax1 = fig.add_subplot(1, 2, 1)
ax1.hist(ascans[65, :, 10], bins=10, color='tab:blue', density=True, range=(0, 100))
ax1.set_title(f"Element 10")
             
ax2 = fig.add_subplot(1, 2, 2)
ax2.hist(ascans[65, :, 61], bins=10, color='tab:blue', density=True, range=(0, 100))
ax2.set_title(f"Element 61")


plt.show()

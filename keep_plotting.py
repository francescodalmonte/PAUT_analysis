import os 
import numpy as np
from matplotlib import pyplot as plt
from PAUT_Data import PAUT_Data
from plots_utils import plot_Cscan, plot_Ascan

input_path = "C:/Users/dalmonte/data/ADAMUS/DFKI PAUT/FA3567883_NI4_45°/FA3567883_NI4_45°/45° 2195_"
ascans = PAUT_Data(input_path).compose_Ascans()[:, 190:1760]

while True:
    x_position = np.random.randint(0, ascans.shape[1])
    element_index = np.random.randint(0, ascans.shape[0])

    print(f"Element index: {element_index}, X position: {x_position}")
    
    fig = plt.figure(figsize=(8,4), dpi=150, tight_layout=True)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    plot_Ascan(ascans, [element_index, x_position], ax1, color='tab:blue')
    plot_Cscan(ascans, ax2, vmin=0, vmax=100, cmap='jet')

    ax2.plot(x_position, 115-element_index, 'ro', markersize=5)

    plt.show()
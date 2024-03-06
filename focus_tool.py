import os 
import numpy as np
from matplotlib import pyplot as plt
import PAUT_Data
import plots_utils




params = {"DIRPATH" : "C:/Users/dalmonte/data/ADAMUS/DFKI PAUT/FA3567389 S3 45 17db/FA3567389 S3 45 17db/45° 2195_",
          "ELEMENT" : 51,
          "POSITION" : 19000,
          "THRESHOLD" : 0.,
          "ELEMENT_RANGE" : [],
          "POSITION_RANGE" : [1000, 20000],
          "TIME_RANGE" : []
          }





def plots(params):
    # setup input arguments
    
    dirpath = params["DIRPATH"]
    element = params["ELEMENT"]
    position = params["POSITION"]
    threshold = params["THRESHOLD"]
    element_range = params["ELEMENT_RANGE"]
    position_range = params["POSITION_RANGE"]
    time_range = params["TIME_RANGE"]

    # load data
    obj = PAUT_Data.PAUT_Data(dirpath)
    ascans = obj.compose_Ascans()
    ascans_shape = ascans.shape
    print(ascans_shape)

    # check some parameters
    if len(element_range)<2:
        element_range = [0, ascans_shape[0]]
    if len(position_range)<2:
        position_range = [0, ascans_shape[1]]
    if len(time_range)<2:
        time_range = [0, ascans_shape[2]]


    # plots
    fig1, ax1 = plt.subplots(figsize=(6,4), dpi=150)
    plots_utils.plot_Ascan(ascans, [element, position], ax1, title = f"A-Scan [elem.{element}, pos.{position}, th.{threshold}]",
                           c="tab:orange")
    ax1.set_xlim(time_range)

    fig2, ax2 = plt.subplots(figsize=(4,4), dpi=150)
    plots_utils.plot_Bscan(ascans, position, ax2, title = f"B-Scan [pos.{position}, th.{threshold}]", th=threshold,
                           angle=45, pitch=.35, depth_resolution=.082,
                           cmap="jet", vmin=0, vmax=100)
    #ax2.set_xlim(element_range)
    #ax2.set_ylim(time_range[::-1])    
    
    fig3, ax3 = plt.subplots(figsize=(12,4), dpi=150)
    plots_utils.plot_Dscan(ascans, element, ax3, title = f"D-Scan [elem.{element}, th.{threshold}]", th=threshold,
                           cmap="jet", vmin=0, vmax=100)
    ax3.set_xlim(position_range)
    ax3.set_ylim(time_range[::-1])    

    fig4, ax4 = plt.subplots(figsize=(16,4), dpi=150)
    plots_utils.plot_Cscan(ascans, ax4, title = f"C-Scan [th.{threshold}]", th=threshold,
                           cmap="jet", vmin=0, vmax=100)
    ax4.set_xlim(position_range)
    ax4.set_ylim(element_range)    

    plt.show()
    

if __name__ == "__main__":
    plots(params)
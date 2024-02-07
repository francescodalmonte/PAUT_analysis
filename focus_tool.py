import os 
import numpy as np
from matplotlib import pyplot as plt
import PAUT_Data
import plots_utils




params = {"DIRPATH" : "D:/data_ADAMUS/ADAMUS/DFKI_PAUT/FA1153220 LF45 15dB SH43/FA1153220 LF45 15dB SH43/45° 2195_",
          "ELEMENT" : 22,
          "POSITION" : 1915,
          "THRESHOLD" : 0.,
          "ELEMENT_RANGE" : [],
          "POSITION_RANGE" : [1800, 2100],
          "TIME_RANGE" : [0, 290]
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

    # check some parameters
    if len(element_range)<2:
        element_range = [0, ascans_shape[0]]
    if len(position_range)<2:
        position_range = [0, ascans_shape[1]]
    if len(time_range)<2:
        time_range = [0, ascans_shape[2]]


    # plots
    fig1, ax1 = plt.subplots(figsize=(6,4), dpi=100)
    plots_utils.plot_Ascan(ascans, [element, position], ax1, title = f"A-Scan [elem.{element}, pos.{position}, th.{threshold}]",
                           c="tab:orange")
    ax1.set_xlim(time_range)

    fig2, ax2 = plt.subplots(figsize=(6,6), dpi=100)
    plots_utils.plot_Bscan(ascans, position, ax2, title = f"B-Scan [pos.{position}, th.{threshold}]", th=threshold,
                           cmap="jet", vmin=0)
    ax2.set_xlim(element_range)
    ax2.set_ylim(time_range)    
    
    fig3, ax3 = plt.subplots(figsize=(12,4), dpi=100)
    plots_utils.plot_Dscan(ascans, element, ax3, title = f"D-Scan [elem.{element}, th.{threshold}]", th=threshold,
                           cmap="jet", vmin=0)
    ax3.set_xlim(position_range)
    ax3.set_ylim(time_range)    

    fig4, ax4 = plt.subplots(figsize=(12,4), dpi=100)
    plots_utils.plot_Cscan(ascans, ax4, title = f"C-Scan [th.{threshold}]", th=threshold,
                           cmap="jet", vmin=0)
    ax4.set_xlim(position_range)
    ax4.set_ylim(element_range)    

    plt.show()
    

if __name__ == "__main__":
    plots(params)
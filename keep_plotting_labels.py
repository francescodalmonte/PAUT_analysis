import os
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

from PAUT_Data import PAUT_Data
import plots_utils

basepath = "C:/Users/dalmonte/data/ADAMUS/DFKI PAUT/"
labelpath = "C:/Users/dalmonte/data/ADAMUS/labelling files/240312_M_Adamus_Anzeigen_DFKI_SUB_refined.csv"

acq_names = ["1152811 45 S22 16dB",
             "1156722_NI6_M1LF45°_16dB",
             "1156722_NI6_M1LF67°_10dB",
             "1156722_NI7_M1LF45°_16dB",
             "1156722_NI7_M1LF67°_10dB",
             "1157798_S22_T0199_LF45_18dB",
             "1157978_NI5_M1LF45°_16dB",
             "1157978_NI8_M1LF45°_16dB",
             "FA1153222 SH83 LF45° 21dB",
             "FA1154309 SH83 LF45 22dB",
             "FA1154337 LH2-FM7_S2_45°_15dB",
             "FA1156470_23.03.21_S12_LF45°_18dB",
             "FA1156632 Lox_FM8_45°_17dB",
             #"FA3566603_Naht1_LF45_21dB_runter",
             #"FA3566603_Naht1_LF67_16dB_hoch",
             #"FA3566603_Naht2_LF45_22dB_hoch",
             #"FA3566603_Naht2_LF67_17dB_hoch",
             "FA3566628_LF 45°_17dB_C-Ring",
             "FA3566742_NI2_45_Aufwärts",
             "FA3566742_NI5_45_Aufwärts",
             #"FA3566923 S1 21db 45",
             #"FA3567052_S2_45_19dB",
             "FA3569047_NI5_45°",
             "FA3569081_SH83_LF45°",
             #"FA3569143 S42 45°",
             "FA3569143 S43 45°"#,
             #"FA3569155_S13_45°"
]


for d in acq_names: 
    dirpath = os.path.join(basepath, d, d)
    for sd in os.listdir(dirpath):
        print(f"Processing {d}/{sd}")
        subdirpath = os.path.join(dirpath, sd)
        obj = PAUT_Data(subdirpath, labelpath=labelpath)
        labels = obj.get_labelsFromCSV()
        print(f"Found: {len(labels)} labelled defect(s)")
        for i in range(len(labels)):
            l_x = labels['Länge l'].values[i]
            x_c = labels['X-Pos.'].values[i] + l_x/2
            l_t = labels['Tiefe t'].values[i]
            t_c = labels['Z-Pos.'].values[i] + l_t/2
            l_y = labels['Breite b'].values[i]
            y_c = labels['Y-Pos.'].values[i] + l_y/2
            
            print(f"x_c={x_c:.2f} - y_c={y_c:.2f} - t_c={t_c:.2f}")

            ascans = obj.compose_Ascans(valid=False)
            print(ascans.shape)

            fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(10,10), dpi=150, tight_layout=True)
            
            # plot cscan
            plots_utils.plot_Cscan(ascans, ax[0], title = f"C-Scan {d}/{sd}",
                                   cmap="jet", vmin=0, vmax=100)
            position_range = [obj.xmm_to_idx(x_c-250), obj.xmm_to_idx(x_c+250)]
            ax[0].set_xlim(position_range)
            ax[0].axvline(obj.xmm_to_idx(x_c - l_x/2), color='black', linestyle='--', linewidth=0.6)
            ax[0].axvline(obj.xmm_to_idx(x_c + l_x/2), color='black', linestyle='--', linewidth=0.6)
            ax[0].axhline(obj.mm_to_cscanyidx(y_c-l_y/2, t_c+l_t/2), color='black', linestyle='--', linewidth=0.6)
            ax[0].axhline(obj.mm_to_cscanyidx(y_c+l_y/2, t_c-l_t/2), color='black', linestyle='--', linewidth=0.6)


            # plot bscan
            corrBscan = obj.extract_Bscan(ascans, 
                                          idx = obj.xmm_to_idx(x_c),
                                          correction = True
            )
            Bshape = corrBscan.shape

            ax[1].imshow(corrBscan, aspect='equal', cmap='jet', vmin=0, vmax=100)
            ax[1].set_title(f"B-Scan {d}/{sd}")
            
            # anomaly lines
            ax[1].axhline(obj.tmm_to_idx(t_c - l_t/2), color='black', linestyle='--', linewidth=0.6)
            ax[1].axhline(obj.tmm_to_idx(t_c + l_t/2), color='black', linestyle='--', linewidth=0.6)
            ax[1].axvline(obj.ymm_to_idx(y_c - l_y/2), color='black', linestyle='--', linewidth=0.6)
            ax[1].axvline(obj.ymm_to_idx(y_c + l_y/2), color='black', linestyle='--', linewidth=0.6)

            # nugget lines
            ax[1].axhline(obj.tmm_to_idx(0.), color='white', linestyle='-', linewidth=0.4)
            ax[1].axvline(obj.ymm_to_idx(0.), ymax=obj.tmm_to_idx(0.), color='white', linestyle='-', linewidth=0.4)

            plt.show()
        





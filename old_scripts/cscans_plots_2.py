import os
import numpy as np
from matplotlib import pyplot as plt

from PAUT_Data import PAUT_Data

out_path = "C:/Users/dalmonte/data/ADAMUS/cscans_plots_2"
input_path = "C:/Users/dalmonte/data/ADAMUS/DFKI PAUT"
for dir in os.listdir(input_path):
    if not dir.endswith('.zip'):
        for subdir in os.listdir(os.path.join(input_path, dir, dir)):
            dirpath = os.path.join(input_path, dir, dir, subdir)
            print(dirpath)
            ascans = PAUT_Data(dirpath).compose_Ascans()
            print(ascans.shape)

            # project data
            cscan = ascans.max(axis=2)
            bscan_max = ascans.max(axis=1)
            bscan_mean = ascans.mean(axis=1)
            bscan_std = ascans.std(axis=1)

            # create figure
            fig = plt.figure(figsize=(12,9), dpi=150, tight_layout=True)
            ax1 = fig.add_subplot(3, 3, (1,2))
            ax2 = fig.add_subplot(3, 3, 3)
            ax3 = fig.add_subplot(3, 3, (4,5))
            ax4 = fig.add_subplot(3, 3, 6)
            ax5 = fig.add_subplot(3, 3, (7,8))
            ax6 = fig.add_subplot(3, 3, 9)


            ax1.imshow(cscan, aspect='auto', cmap='jet')
            ax1.set_title("C-scan")
            ax2.imshow(bscan_max, aspect='auto', cmap='jet')
            ax2.set_title("B-scan max")

            ax3.imshow(cscan, aspect='auto', cmap='jet')
            ax3.set_title("C-scan")
            ax4.imshow(bscan_mean, aspect='auto', cmap='jet')
            ax4.set_title("B-scan mean")

            ax5.imshow(cscan, aspect='auto', cmap='jet')
            ax5.set_title("C-scan")
            ax6.imshow(bscan_std, aspect='auto', cmap='jet')
            ax6.set_title("B-scan std")



            # save to file
            plt.savefig(os.path.join(out_path, f"{dir}_{subdir}.png"))


#b_img = ascans[:, 1000, :].copy()

#plt.imshow(b_img, aspect='auto', cmap='jet')
#plt.show()



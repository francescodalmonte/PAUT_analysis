import os
import numpy as np
import matplotlib.axes
from matplotlib import pyplot as plt

class PAUT_Data():
    """Base class for PAUT data"""

    def __init__(self, dirpath: str):
        self.dirpath = dirpath

        # list of L-elements directories is read and sorted (according to LXXX number)
        self.Ldirs = np.array([d for d in os.listdir(self.dirpath) if d.startswith("Linear")])
        Ldirs_number = np.array([int(d.split(" ")[2][1:]) for d in self.Ldirs])
        self.Ldirs = self.Ldirs[Ldirs_number.argsort()]


    def __get_info__(self):
        """
        Returns a dict with general info regarding the object.
        """
        ascans = self.compose_Ascans()
        ascans_shape = ascans.shape

        infoDict = {}
        infoDict["N_Lelements"] = ascans_shape[0]
        infoDict["N_positons"] = ascans_shape[1]
        infoDict["N_timeSteps"] = ascans_shape[2]
        infoDict["amplitude_range"] = [ascans.min(), ascans.max()]
        infoDict["amplitude_mean"] = ascans.mean()
        infoDict["amplitude_percentile"] = {"q9500" : np.quantile(ascans, 0.9500),
                                            "q9900" : np.quantile(ascans, 0.9900),
                                            "q9999" : np.quantile(ascans, 0.9999)
                                            }
        infoDict["unique_amplitude_values"] = len(np.unique(ascans))

        return infoDict


    def __get_gatedCscan__(self, i: int, measurement: str = "A", gate_subdir="Gate A"):
        """
        Read gated C-Scan .npy files from file.
        
        Parameters
        ----------
        i : (int) directory index.
        measurement : (["A", "P", "A_ERR", P_ERR"]) identifier for data file.

        Returns
        ----------
        (np.ndarray) 1-d array (length = n. of spatial positions)


        """
        fnames = {"A" : "Amplitude C-Bild [Amplitude]_C_SCAN.npy",
                  "P" : "Position C-Bild [Position]_C_SCAN.npy",
                  "A_ERR" : "Amplitude C-Bild [Amplitude]_C_SCAN_ERRORINFO.npy",
                  "P_ERR" : "Position C-Bild [Position]_C_SCAN.npy_ERRORINFO"}
        
        filepath = os.path.join(self.dirpath, f"{self.Ldirs[i]}/{gate_subdir}/{fnames[measurement]}")
        return np.load(filepath)[0]
    


    def __get_linearAscans__(self, i: int):
        """
        Read linear A-Scans .npy file from file.
        
        Parameters
        ----------
        i : (int) directory index.

        Returns
        ----------
        (np.ndarray) 2-d array (shape = n. of spatial positions x time points)

        
        """
        fname = "A-Bild_A_SCAN.npy"
        filepath = os.path.join(self.dirpath, f"{self.Ldirs[i]}/Gate Main (A-Scan)/{fname}")
        if os.path.isfile(filepath):
            return np.load(filepath)[0]
        else:
            fname = "A-scan_A_SCAN.npy"
            filepath = os.path.join(self.dirpath, f"{self.Ldirs[i]}/Gate Main (A-Scan)/{fname}")
            return np.load(filepath)[0]


    def compose_gatedCscan(self, measurement = "A", gate_subdir="Gate A"):
        """
        Compose a total C-Scan by looping over all the gated C-Scans of the acquisition,
        and stacking 1-d arrays on a new axis.

        Parameters
        ----------
        measurement : (["A", "P", "A_ERR", P_ERR"]) identifier for data file.

        Returns
        ----------
        Cscan : (np.ndarray) 2-d array (shape = n. of L-elements x n. of spatial positions)
        
        """
        Cscan = []
        for i in range(len(self.Ldirs)):
            Cscan.append(self.__get_gatedCscan__(i, measurement, gate_subdir))

        return np.array(Cscan)



    def compose_Ascans(self):
        """
        Compose all A-Scans in a unique 3-d array by looping over all the A-Scans of
        the acquisition, and stacking 2-d arrays on a new axis.

        Returns
        ----------
        tot_Ascans : (np.ndarray) 3-d array (shape = n. of L-elements x n. of spatial positions x time points)
        
        """
        tot_Ascans = []
        for i in range(len(self.Ldirs)):
            tot_Ascans.append(self.__get_linearAscans__(i))

        return np.array(tot_Ascans)


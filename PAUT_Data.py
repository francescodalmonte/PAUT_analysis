import os
import numpy as np
import pandas as pd
import matplotlib.axes
from matplotlib import pyplot as plt
import cv2 as cv

class PAUT_Data():
    """Base class for PAUT data"""

    def __init__(self, dirpath: str, labelpath: str = None):
        self.dirpath = dirpath

        # acquisition name
        self.acquisition_name = os.path.split(os.path.dirname(self.dirpath))[1]

        # list of L-elements directories is read and sorted (according to LXXX number)
        self.Ldirs = np.array([d for d in os.listdir(self.dirpath) if d.startswith("Linear")])
        Ldirs_number = np.array([int(d.split(" ")[2][1:]) for d in self.Ldirs])
        self.Ldirs = self.Ldirs[Ldirs_number.argsort()]

        # metadata
        if labelpath is not None:
            self.labelpath = labelpath
            self.metadata = self.get_metadata()


    def checklabels(self, csv: pd.DataFrame):
        if len(csv)==0:
            raise ValueError("No data found in labels file.")
        if not (np.all(csv["x Start"] == csv["x Start"].values[0])
                and np.all(csv["x Ende"] == csv["x Ende"].values[0])):
            raise ValueError("x Start and x Ende are not constant (or may contain nans).")
        elif not (np.all(csv["Tiefe Start"] == csv["Tiefe Start"].values[0])
                  and np.all(csv["Tiefe Ende"] == csv["Tiefe Ende"].values[0])):
            raise ValueError("Tiefe Start and Tiefe Ende are not constant (or may contain nans).")
        elif not np.all(csv["X-Pos."] == csv["X-Pos."]):
            raise ValueError("X-Pos. contains nans.") 
        #elif not np.all(csv["Y-Pos."] == csv["Y-Pos."]):   # ****to be updated when available****
        #    raise ValueError("Y-Pos. contains nans.")      # ****to be updated when available****
        #elif not np.all(csv["Z-Pos."] == csv["Z-Pos."]):
        #    raise ValueError("Z-Pos. contains nans.")      # ****to be updated when available****
        elif not np.all(csv["Länge l"] == csv["Länge l"]):
            raise ValueError("Länge l contains nans.")
        #elif not np.all(csv["Breite b"] == csv["Breite b"]):
        #    raise ValueError("Breite b contains nans.")    # ****to be updated when available****
        #elif not np.all(csv["Tiefe t"] == csv["Tiefe t"]):
        #    raise ValueError("Tiefe t contains nans.")     # ****to be updated when available****
        else:
            return 0


    def get_labelsFromCSV(self):
        """
        Get labels info from .csv file.
        """
        if os.path.isfile(self.labelpath):
            csv = pd.read_csv(self.labelpath, encoding='latin')
            if self.acquisition_name in csv["Filename"].values:
                csv = csv[csv["Filename"] == self.acquisition_name]
                self.checklabels(csv)
                return csv
            else:
                raise ValueError("Acquisition name not found in labels file.")
        else:
            raise FileNotFoundError("Labels file not found.")


    def get_metadata(self):
        """
        Get metadata from the acquisition (e.g. valid x_pos etc.).
        """
        Cerr = self.get_gatedCscan(0, "A_ERR")
        Ascan = self.get_linearAscans(0)

        labels = self.get_labelsFromCSV()
        x_start_mm = labels["x Start"].values[0]
        x_end_mm = labels["x Ende"].values[0]
        y_start_mm = labels["y Start"].values[0]  # ****to be updated when available****
        y_end_mm = labels["y Ende"].values[0]     # ****to be updated when available****
        t_start_mm = labels["Tiefe Start"].values[0]
        t_end_mm = labels["Tiefe Ende"].values[0]
        angle = int(labels["Angle"].values[0].strip("°"))

        # x position 
        x_N = len(Cerr)
        x_valid_lim  = [np.min(np.argwhere(Cerr==0)), np.max(np.argwhere(Cerr==0))]
        x_valid_lim_mm = [x_start_mm, x_end_mm]
        x_res = (x_valid_lim_mm[1]-x_valid_lim_mm[0])/(x_valid_lim[1]-x_valid_lim[0])

        # y position
        y_N = 115
        y_lim_mm = [y_start_mm, y_end_mm]
        y_res = (y_lim_mm[1]-y_lim_mm[0])/(y_N-1) #= pitch

        # t position
        t_N = Ascan.shape[1]
        t_lim_mm = [t_start_mm, t_end_mm]
        t_res = (t_lim_mm[1]-t_lim_mm[0])/(t_N-1)
        
        return {"x_N" : x_N,
                "y_N" : y_N,
                "t_N" : t_N,
                "x_valid_lim" : x_valid_lim,
                "x_valid_lim_mm" : x_valid_lim_mm,
                "x_res" : x_res,
                "t_lim_mm" : t_lim_mm,
                "t_res" : t_res,
                "y_lim_mm" : y_lim_mm,
                "y_res" : y_res,
                "angle" : angle
        }
    

    # coordinates conversions
    def xmm_to_idx(self, xmm):
        idx = (xmm-self.metadata["x_valid_lim_mm"][0])/self.metadata["x_res"] + self.metadata["x_valid_lim"][0]
        return np.round(idx).astype(int)
    
    def idx_to_xmm(self, idx):
        return idx*self.metadata["x_res"] + self.metadata["x_valid_lim_mm"][0]
    
    def ymm_to_idx(self, ymm):
        idx = (ymm-self.metadata["y_lim_mm"][0])/self.metadata["t_res"]
        return np.round(idx).astype(int)
    
    def idx_to_ymm(self, idx):
        return idx*self.metadata["t_res"] + self.metadata["y_lim_mm"][0]

    def tmm_to_idx(self, tmm):
        idx = (tmm-self.metadata["t_lim_mm"][0])/self.metadata["t_res"]
        return np.round(idx).astype(int)

    def idx_to_tmm(self, idx):
        return idx*self.metadata["t_res"] + self.metadata["t_lim_mm"][0]

        
    def get_stats(self):
        """
        Returns a dict with statistics (e.g. data shape, amplitude range, mean, percentiles etc.).
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


    def get_gatedCscan(self, i: int, measurement: str = "A", gate_subdir="Gate A"):
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
        fnames = {"A" : ["Amplitude C-Bild [Amplitude]_C_SCAN.npy",
                         "Amplitude C-scan [Amplitude]_C_SCAN.npy"],
                  "P" : ["Position C-Bild [Position]_C_SCAN.npy",
                         "Position C-scan [Position]_C_SCAN.npy"],
                  "A_ERR" : ["Amplitude C-Bild [Amplitude]_C_SCAN_ERRORINFO.npy",
                             "Amplitude C-scan [Amplitude]_C_SCAN_ERRORINFO.npy"],
                  "P_ERR" : ["Position C-Bild [Position]_C_SCAN.npy_ERRORINFO",
                             "Position C-scan [Position]_C_SCAN.npy_ERRORINFO"]
        }
        
        try:
            filepath = os.path.join(self.dirpath, f"{self.Ldirs[i]}/{gate_subdir}/{fnames[measurement][0]}")
            npyfile = np.load(filepath)[0]
            return npyfile
        except:
            filepath = os.path.join(self.dirpath, f"{self.Ldirs[i]}/{gate_subdir}/{fnames[measurement][1]}")
            npyfile = np.load(filepath)[0]
            return npyfile

    


    def get_linearAscans(self, i: int):
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
            Cscan.append(self.get_gatedCscan(i, measurement, gate_subdir))

        return np.array(Cscan)



    def compose_Ascans(self, valid: bool = True):
        """
        Compose all A-Scans in a unique 3-d array by looping over all the A-Scans of
        the acquisition, and stacking 2-d arrays on a new axis.

        Parameters
        ----------
        valid : (bool) if True, only valid spatial positions are considered.

        Returns
        ----------
        tot_Ascans : (np.ndarray) 3-d array (shape = n. of L-elements x n. of spatial positions x time points)
        
        """
        tot_Ascans = []
        for i in range(len(self.Ldirs)):
            tot_Ascans.append(self.get_linearAscans(i))

        if valid:
            xi = self.metadata["x_valid_lim"][0]
            xf = self.metadata["x_valid_lim"][1]
            tot_Ascans = np.array(tot_Ascans)[:, xi:xf, :]
            print(f"Valid spatial positions: {tot_Ascans.shape}")
        else:
            tot_Ascans = np.array(tot_Ascans)
        return tot_Ascans



    def Bscan_correction(self, data: np.ndarray, 
                         angle: float, pitch: float, res_depth: float):
        """
        Apply geometrical correction to a B-Scan.

        Parameters
        ----------
        data : (np.ndarray) 2-d array (raw B-Scan).
        angle : (float) angle of the probe.
        pitch : (float) horizontal resolution.
        res_depth : (float) vetical resolution.

        Returns
        ----------
        corrected_Bscan : (np.ndarray) 2-d array (corrected B-Scan).
        
        """
        # preliminar computation of transformation parameters
        angle_rad = angle*np.pi/180
        res_y = res_depth/np.cos(angle_rad) # resolution in oblique direction
        res_x = pitch*np.cos(angle_rad) 
        shearY = pitch*np.sin(angle_rad)/res_y
        rXY_factor = res_x/res_y


        # SHEAR MAPPING (y axis)
        rows, cols = data.shape
        # preliminar padding 
        data = np.pad(data, ((0, int(np.round(shearY*cols, 0))), (0, 0)),
                      mode='constant', constant_values=np.nan)
        # shear mapping
        M_shear = np.array([[1, 0, 0],
                            [shearY, 1, 0]], dtype=np.float32)
        rows_m, cols_m = data.shape
        data = cv.warpAffine(data, M_shear, (cols_m, rows_m),
                             borderMode=cv.BORDER_CONSTANT, borderValue = np.nan,
                             flags=cv.INTER_NEAREST)

        # UPSAMPLING OF x AXIS (to match y resolution)
        data = cv.resize(data, dsize = None, fx=rXY_factor, fy=1.,
                         interpolation=cv.INTER_NEAREST_EXACT)
        
        # ROTATION
        M_rotation = cv.getRotationMatrix2D([0,0], angle, scale=1)
        rows_m, cols_m = data.shape
        out_shape = (int(cols_m*np.cos(angle_rad)+rows_m*(np.sin(angle_rad))),
                     int(rows_m*np.cos(angle_rad)-cols_m*(np.sin(angle_rad))))
        data = cv.warpAffine(data, M_rotation, out_shape,
                             borderMode=cv.BORDER_CONSTANT, borderValue = np.nan)
        
        # UPSAMPLE TO MATCH ORIGINAL Y-AXIS RESOLUTION
        upsample_factor = rows/out_shape[1]
        data = cv.resize(data, dsize = None, fx=upsample_factor, fy=upsample_factor,
                         interpolation=cv.INTER_NEAREST_EXACT)
            
        return data



    def extract_Bscan(self, ascans: np.ndarray, idx: int,
                      correction: bool = False):
        """
        Extract a B-Scan from the 3D A-Scans set (geometrical correction
        is optional).

        Parameters
        ----------
        ascans : (np.ndarray) 3-d array (n. of L-elements x n. of spatial positions x time points)
        idx : (int) spatial position index (x).
        correction : (bool) apply geometrical correction.

        Returns
        ----------
        Bscan : (np.ndarray) 2-d array (B-Scan).

        """
        bscan = ascans[:,idx].T.copy()
        if correction:
            bscan = self.Bscan_correction(bscan,
                                          angle = self.metadata["angle"],
                                          pitch = self.metadata["y_res"],
                                          res_depth = self.metadata["t_res"]
                                          )
        return bscan
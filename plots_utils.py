import numpy as np
import matplotlib.axes
from matplotlib import pyplot as plt
import cv2 as cv


def plot_Ascan(data: np.ndarray, 
               idxs: list[int,],
               ax: matplotlib.axes.Axes,
               title: str = "A-Scan",
               **kwargs):
    """
    Plot a time vs Amplitude A-Scan on an existing axis.

    Parameters
    ----------
    data: (np.ndarray) 3-dimensional array with A-Scans data
    idxs: (list[int,]) element index, spatial position index
    ax: (matplotlib.axes.Axes)
    title: (str)
    **kwargs to be passed to Axes.plot()

    """
    ax.plot(data[idxs[0], idxs[1]], **kwargs)
    ax.set_xlabel("Time/depth (au)")
    ax.set_ylabel("Echo amplitude (au)")
    ax.set_ylim(0, 100)
    ax.set_title(title)

    return data[idxs[0], idxs[1]]


def plot_Bscan(data: np.ndarray, 
               idx: int,
               ax: matplotlib.axes.Axes,
               title: str = "B-Scan",
               th: float = 0.,
               angle: float = None,
               pitch: float = None,
               depth_resolution: float = None,
               **kwargs):
    """
    Plot a 2-d cross sectional B-Scan on an existing axis.

    Parameters
    ----------
    data: (np.ndarray) 3-dimensional array with A-Scans data
    idx: (int) spatial position index
    ax: (matplotlib.axes.Axes)
    title: (str)
    th: (float) lower threshold
    **kwargs to be passed to Axes.imshow()

    """
    th_data = data[:,idx].T.copy()
    th_data[th_data<th] = np.nan

    if angle is not None:
        shift = np.tan(angle*np.pi/180)*pitch
        shift_pixels = np.rint(shift/depth_resolution).astype(int)
        print(shift)
        print(shift_pixels)
        padded_th_data = np.pad(th_data, ((0, shift_pixels*th_data.shape[1]), (0, 0)), mode='constant', constant_values=np.nan)
        print(padded_th_data.shape)
        for col in range(padded_th_data.shape[1]):
            padded_th_data[:, col] = np.roll(padded_th_data[:, col], shift_pixels*col)
        
        th_data = padded_th_data

    ax.imshow(th_data, aspect='equal', **kwargs)
    ax.set_ylabel("Time/depth (au)")
    ax.set_xlabel("Element")
    ax.set_title(title)

    return th_data
    

def plot_Bscan_wcorrection(data: np.ndarray, 
                           idx: int,
                           ax: matplotlib.axes.Axes,
                           angle: int,
                           res_depth: float,
                           pitch: float,
                           title: str = "B-Scan",
                           th: float = 0., 
                           **kwargs):
    """


    """
    th_data = data[:,idx].T.copy()
    th_data[th_data<th] = np.nan

    if angle != 0:

        # preliminar computation of transformation parameters
        angle_rad = angle*np.pi/180
        res_x = pitch*np.cos(angle_rad)
        res_y = res_depth/np.cos(angle_rad)
        shearY = pitch*np.sin(angle_rad)/res_y
        rXY_factor = res_x/res_y


        print(f"angle: {angle}, res_x: {res_x}, res_y: {res_y}, shearY: {shearY}")


        # SHEAR MAPPING (y axis)
        rows, cols = th_data.shape
        # upsample 10x in y direction (enables a more precise shear mapping)
        th_data = cv.resize(th_data.copy(), dsize = None,
                            fx=1., fy=10., interpolation=cv.INTER_LINEAR)
        shearY_r = int(np.round(shearY*10, 0))
        # preliminar padding 
        th_data = np.pad(th_data, ((0, shearY_r*(cols-1)), (0, 0)),
                         mode='constant', constant_values=np.nan)
        # shear mapping
        # (each column is manually shifted donwwards by a factor proportional to column index)
        for col in range(cols):
            th_data[:, col] = np.roll(th_data[:, col], shearY_r*(col))
        # downsample back to original resolution
        th_data = cv.resize(th_data, dsize = None, fx=1., fy=0.1, interpolation=cv.INTER_AREA)

        # UPSAMPLING OF x AXIS (to match y resolution)
        th_data = cv.resize(th_data, dsize = None, fx=rXY_factor, fy=1., interpolation=cv.INTER_LINEAR)

        # ROTATION
        M_rotation = cv.getRotationMatrix2D([0,0], angle, scale=1)
        rows_m, cols_m = th_data.shape
        out_shape = (int(cols_m*np.cos(angle_rad)+rows_m*(np.sin(angle_rad))),
                     int(rows_m*np.cos(angle_rad)-cols_m*(np.sin(angle_rad))))
        th_data = cv.warpAffine(th_data, M_rotation, out_shape,
                                borderMode=cv.BORDER_CONSTANT, borderValue = np.nan)
                
    ax.imshow(th_data, aspect='equal', **kwargs)

    #x_axis_mm = np.arange(0, out_shape[0]*res_x, res_x)
    #y_axis_mm = np.arange(0, out_shape[1]*res_y, res_y)

    ax.set_xticks(np.linspace(0, out_shape[0], 10),
                  np.round(np.linspace(0, out_shape[0], 10)*res_y, 1))
    ax.set_yticks(np.linspace(0, out_shape[1], 6),
                  np.round(np.linspace(0, out_shape[1], 6)*res_y, 1))
    ax.set_ylabel("Depth [mm]")
    ax.set_xlabel("y [mm]")
    ax.set_title(title)

    return th_data



def plot_Dscan(data: np.ndarray, 
               idx: int,
               ax: matplotlib.axes.Axes,
               title: str = "B-Scan",
               th: float = 0.,
               **kwargs):
    """
    Plot a 2-d cross sectional D-Scan on an existing axis.

    Parameters
    ----------
    data: (np.ndarray) 3-dimensional array with A-Scans data
    idx: (int) element index
    ax: (matplotlib.axes.Axes)
    title: (str)
    th: (float) lower threshold
    **kwargs to be passed to Axes.imshow()

    """
    th_data = data[idx, :].T.copy()
    th_data[th_data<th] = np.nan
    ax.imshow(th_data, **kwargs)
    ax.set_ylabel("Time/depth (au)")
    ax.set_xlabel("Position (au)")
    ax.set_title(title)

    return th_data


def plot_Cscan(data: np.ndarray, 
               ax: matplotlib.axes.Axes,
               title: str = "C-Scan",
               th: float = 0.,
               pos_mm: np.ndarray = None,
               **kwargs):
    """
    Plot a 2-d C-Scan on an existing axis.

    Parameters
    ----------
    data: (np.ndarray) 3-dimensional array with A-Scans data
    ax: (matplotlib.axes.Axes)
    title: (str)
    th: (float) lower threshold
    **kwargs to be passed to Axes.imshow()

    """
    cscan = np.max(data, axis=2)
    cscan[cscan<th]=np.nan

    ax.imshow(cscan, aspect='auto', **kwargs)
    if pos_mm is None:
        ax.set_xlabel("Position (au)")
    else:
        ax.set_xlabel("Position (mm)")
        ax.set_xticks(np.arange(0, len(pos_mm), int(len(pos_mm)/10)))
        ax.set_xticklabels(pos_mm[::int(len(pos_mm)/10)])

    ax.set_ylabel("Elements")
    ax.set_title(title)

    return cscan

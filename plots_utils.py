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

    ax.imshow(th_data, aspect='auto', **kwargs)
    #ax.set_ylabel("Time/depth (au)")
    ax.set_yticks([])
    ax.set_xlabel("Element")
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
    ax.set_ylabel("Elements")
    ax.set_xlabel("Position (au)")
    ax.set_title(title)

    return cscan
    
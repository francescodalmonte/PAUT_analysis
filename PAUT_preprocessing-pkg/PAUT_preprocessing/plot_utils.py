import numpy as np
import matplotlib.axes
from matplotlib import pyplot as plt
import cv2 as cv

from PAUT_preprocessing.PAUT_acquisition import PAUT_acquisition


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
               obj: PAUT_acquisition,
               idx: int,
               ax: matplotlib.axes.Axes,
               title: str = "B-Scan",
               th: float = 0.,
               correction: bool = False,
               **kwargs):
    """
    Plot a 2-d cross sectional B-Scan on an existing axis.

    Parameters
    ----------

    """
    bscan = obj.extract_Bscan(data, idx, correction = correction)
    bscan[bscan<th]=np.nan
    
    ax.imshow(bscan, aspect='equal', **kwargs)
    ax.set_title(title)

    if correction:
        ax.set_xlabel("Position (mm)")
        ax.set_ylabel("Time/depth (mm)")
    
        xt, yt = np.arange(0, bscan.shape[1], 100), np.arange(0, bscan.shape[0], 50)
        xtl, ytl = np.round((obj.idx_to_ymm(xt)), 1), np.round((obj.idx_to_tmm(yt)), 1)
        ax.set_xticks(xt)
        ax.set_yticks(yt)
        ax.set_xticklabels(xtl)
        ax.set_yticklabels(ytl)

    else:
        ax.set_xlabel("Position (au)")
        ax.set_ylabel("Time/depth (au)")

    return bscan


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
    cscan = np.max(data, axis=2)[::-1,:]
    cscan[cscan<th]=np.nan

    ax.imshow(cscan, aspect='auto', **kwargs)
    ax.set_xlabel("Position (au)")
    ax.set_ylabel("Elements")
    ax.set_title(title)

    return cscan

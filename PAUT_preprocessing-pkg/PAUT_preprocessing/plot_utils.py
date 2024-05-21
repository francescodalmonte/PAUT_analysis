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


def plot_Bscan():
    """
    Plot a 2-d cross sectional B-Scan on an existing axis.

    Parameters
    ----------

    """
    # TODO
    pass


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
    cscan = np.max(data, axis=2)[::-1,:]
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

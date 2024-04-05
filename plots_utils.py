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
                           res_depth: float, # resolution in vertical dimension
                           pitch: float, # resolution in horizontal direction
                           title: str = "B-Scan",
                           th: float = 0., 
                           **kwargs):
    """


    """
    th_data = data[:,idx].T.copy()
    th_data[th_data<th] = np.nan
    #th_data = np.clip(th_data*2+55, 0, 255)

    if angle != 0:

        # preliminar computation of transformation parameters
        angle_rad = angle*np.pi/180
        res_y = res_depth/np.cos(angle_rad) # resolution in oblique direction

        shearY = pitch*np.sin(angle_rad)/res_y
        f = 20. 
        shearY_r = int(np.round(shearY*f, 0)) # rounded shear value

        angle_hat_rad = np.arcsin(shearY_r*res_y/(pitch*f)) # real angle
        angle_hat = angle_hat_rad*180/np.pi

        res_x = pitch*np.cos(angle_hat_rad) 
        rXY_factor = res_x/res_y

        #cv.imwrite("C:/Users/dalmonte/data/0_data.jpg", th_data)

        # SHEAR MAPPING (y axis)
        rows, cols = th_data.shape
        # upsample f in y direction (enables a more precise shear mapping)
        th_data = cv.resize(th_data.copy(), dsize = None,
                            fx=1., fy=f, interpolation=cv.INTER_NEAREST_EXACT)
        #cv.imwrite("C:/Users/dalmonte/data/1_data_upsampling1.jpg", th_data)

        # preliminar padding 
        th_data = np.pad(th_data, ((0, shearY_r*(cols-1)), (0, 0)),
                         mode='constant', constant_values=np.nan)
        #cv.imwrite("C:/Users/dalmonte/data/2_data_padding.jpg", th_data)

        # shear mapping
        # (each column is manually shifted donwwards by a factor proportional to column index)
        for col_idx in range(cols):
            th_data[:, col_idx] = np.roll(th_data[:, col_idx], shearY_r*(col_idx))

        #cv.imwrite("C:/Users/dalmonte/data/3_data_shearing.jpg", th_data)

        # UPSAMPLING OF x AXIS (to match y resolution)
        th_data = cv.resize(th_data, dsize = None, fx=rXY_factor, fy=1., interpolation=cv.INTER_NEAREST_EXACT)
        #cv.imwrite("C:/Users/dalmonte/data/4_data_upsampling2.jpg", th_data)

        # downsample 1/f y axis back
        th_data = cv.resize(th_data, dsize = None, fx=1., fy=1/f, interpolation=cv.INTER_LINEAR)
        #cv.imwrite("C:/Users/dalmonte/data/5_data_downsampling.jpg", th_data)

        # ROTATION
        print(angle_hat)

        M_rotation = cv.getRotationMatrix2D([0,0], angle_hat, scale=1)
        rows_m, cols_m = th_data.shape
        out_shape = (int(cols_m*np.cos(angle_hat_rad)+rows_m*(np.sin(angle_hat_rad))),
                     int(rows_m*np.cos(angle_hat_rad)-cols_m*(np.sin(angle_hat_rad))))
        th_data = cv.warpAffine(th_data, M_rotation, out_shape,
                                borderMode=cv.BORDER_CONSTANT, borderValue = np.nan)
        
        #cv.imwrite("C:/Users/dalmonte/data/6_data_rotation.jpg", th_data)
        
        # UPSAMPLE TO MATCH ORIGINAL Y-AXIS RESOLUTION
        upsample_factor = rows/out_shape[1]
        th_data = cv.resize(th_data, dsize = None, fx=upsample_factor, fy=upsample_factor, interpolation=cv.INTER_NEAREST_EXACT)
        #cv.imwrite("C:/Users/dalmonte/data/7_data_upsampling3.jpg", th_data)
    ax.imshow(th_data, aspect='equal', **kwargs)

    ax.set_xticks(np.linspace(0, th_data.shape[1], 10),
                  np.round(np.linspace(0, th_data.shape[1], 10)*res_depth, 1))
    ax.set_yticks(np.linspace(0, th_data.shape[0], 6),
                  np.round(np.linspace(0, th_data.shape[0], 6)*res_depth, 1))
    ax.set_ylabel("Depth [mm]")
    ax.set_xlabel("y [mm]")
    ax.set_title(title)

    return th_data


def plot_Bscan_wcorrection_v2(data: np.ndarray, 
                              idx: int,
                              ax: matplotlib.axes.Axes,
                              angle: int,
                              res_depth: float, # resolution in vertical dimension
                              pitch: float, # resolution in horizontal direction
                              title: str = "B-Scan",
                              th: float = 0., 
                              **kwargs):
    """


    """
    th_data = data[:,idx].T.copy()
    th_data[th_data<th] = np.nan
    #th_data = np.clip(th_data*2+55, 0, 255)

    if angle != 0:

        # preliminar computation of transformation parameters
        angle_rad = angle*np.pi/180
        res_y = res_depth/np.cos(angle_rad) # resolution in oblique direction

        shearY = pitch*np.sin(angle_rad)/res_y
        res_x = pitch*np.cos(angle_rad) 
        rXY_factor = res_x/res_y

        #cv.imwrite("C:/Users/dalmonte/data/0_data_v2.jpg", th_data)

        # SHEAR MAPPING (y axis)

        rows, cols = th_data.shape

        # preliminar padding 
        th_data = np.pad(th_data, ((0, int(np.round(shearY*cols, 0))), (0, 0)),
                         mode='constant', constant_values=np.nan)
        #cv.imwrite("C:/Users/dalmonte/data/1_data_padding_v2.jpg", th_data)

        # shear mapping
        M_shear = np.array([[1, 0, 0],
                            [shearY, 1, 0]], dtype=np.float32)
        rows_m, cols_m = th_data.shape
        out_shape = (cols_m,rows_m)
        th_data = cv.warpAffine(th_data, M_shear, out_shape,
                                borderMode=cv.BORDER_CONSTANT, borderValue = np.nan,
                                flags=cv.INTER_NEAREST)

        cv.imwrite("C:/Users/dalmonte/data/2_data_shearing_v2.jpg", th_data)

        # UPSAMPLING OF x AXIS (to match y resolution)
        th_data = cv.resize(th_data, dsize = None, fx=rXY_factor, fy=1., interpolation=cv.INTER_NEAREST_EXACT)
        #cv.imwrite("C:/Users/dalmonte/data/3_data_upsampling2_v2.jpg", th_data)

        # ROTATION

        M_rotation = cv.getRotationMatrix2D([0,0], angle, scale=1)
        rows_m, cols_m = th_data.shape
        out_shape = (int(cols_m*np.cos(angle_rad)+rows_m*(np.sin(angle_rad))),
                     int(rows_m*np.cos(angle_rad)-cols_m*(np.sin(angle_rad))))
        th_data = cv.warpAffine(th_data, M_rotation, out_shape,
                                borderMode=cv.BORDER_CONSTANT, borderValue = np.nan)
        
        #cv.imwrite("C:/Users/dalmonte/data/4_data_rotation_v2.jpg", th_data)
        
        # UPSAMPLE TO MATCH ORIGINAL Y-AXIS RESOLUTION
        upsample_factor = rows/out_shape[1]
        th_data = cv.resize(th_data, dsize = None, fx=upsample_factor, fy=upsample_factor, interpolation=cv.INTER_NEAREST_EXACT)
        #cv.imwrite("C:/Users/dalmonte/data/5_data_upsampling3_v2.jpg", th_data)

    ax.imshow(th_data, aspect='equal', **kwargs)

    ax.set_xticks(np.linspace(0, th_data.shape[1], 10),
                  np.round(np.linspace(0, th_data.shape[1], 10)*res_depth, 1))
    ax.set_yticks(np.linspace(0, th_data.shape[0], 6),
                  np.round(np.linspace(0, th_data.shape[0], 6)*res_depth, 1))
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

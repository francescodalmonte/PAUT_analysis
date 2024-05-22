#-*- coding: utf-8 -*-
import time
import configparser
import argparse
import os
import json
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

from PAUT_preprocessing.PAUT_acquisition import PAUT_acquisition





def setupArgs():
    parser = argparse.ArgumentParser()
    config = configparser.ConfigParser()
    parser.add_argument("--config",
                        type=str,
                        help="Absolute filepath of config (.INI) file (default: ./create_bscans_COCOds.INI)",
                        default=os.path.join(os.path.dirname(__file__), "create_bscans_COCOds.INI")
                        )
    config_path = parser.parse_args().config
    if os.path.isfile(config_path):
        config.read(config_path)
    else:
        raise ValueError(f"can't find configuration file {config_path}")
    return config




def ncrop_std_preprocessing(crop: np.ndarray,
                            rand_mean: float = 0.,
                            rand_std: float = 0.2) -> np.ndarray:
    """
    Preprocess a nugget crop.
    """
    # std preprocessing (noise addition -> clipping -> sqrt -> scaling)
    crop = crop + 1. + np.random.normal(rand_mean, rand_std, crop.shape)
    crop = np.clip(crop, 0, 100)
    crop = np.round(np.sqrt(crop)*255/10, 0).astype(np.uint8)
    return crop



def compute_median_ncrop(obj: PAUT_acquisition,
                         ascans: np.ndarray) -> np.ndarray:
    """
    Compute the median nugget crop.
    """
    print("Computing median ncrop...", end=" ")
    t = time.time()
    ncrops = []
    for i in np.arange(ascans.shape[1], step=4):
        ncrops.append(obj.extract_ncrop(obj.extract_Bscan(ascans, i, correction = True)))
    median_ncrop = np.median(np.array(ncrops), axis=0)
    print(f"elapsed {time.time()-t:.2f} s")
    return median_ncrop



def ncrop_mch_preprocessing(ncrop: np.ndarray,
                            median_ncrop: np.ndarray) -> np.ndarray:
    """
    Multi-channel preprocessing of a nugget crop.
    """
    # create multichannel nugget crops
    median_channel = (np.sqrt((cv.GaussianBlur(median_ncrop, None, 2, 2)/50))*255).astype(np.uint8)
    ncrop_ch1 = (np.sqrt((ncrop/100))*255).astype(np.uint8)
    mch_ncrop = np.dstack((median_channel, ncrop_ch1, ncrop_ch1))
    return mch_ncrop




def get_ncrop(ascans: np.ndarray,
              idx: int,
              acquisition_obj: PAUT_acquisition,
              **kwargs) -> np.ndarray:
    """
    Get a preprocessed nugget crop.
    """
    bscan = acquisition_obj.extract_Bscan(ascans, idx, correction = True)
    crop = acquisition_obj.extract_ncrop(bscan)

    return ncrop_mch_preprocessing(crop, **kwargs)




def create_save_dirs(save_path: str):
    """
    Conditional creation of save directories.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(os.path.join(save_path, "normal"))
        os.makedirs(os.path.join(save_path, "anomalous"))




def main(config):
    base_path = config["DEFAULT"]["BASE_PATH"]
    ann_path = config["DEFAULT"]["ANN_PATH"]
    save_path = config["DEFAULT"]["SAVE_PATH"]
    acq_names = config["DEFAULT"]["ACQ_NAMES"].split(", ")
    n_good = int(config["DEFAULT"]["N_GOOD"])
    n_anom = int(config["DEFAULT"]["N_ANOM"])

    create_save_dirs(save_path)

    a_id = -1 # anomaly counter
    img_id = -1 # image counter

    info = {
        "year": '2024',
        "description": '',
        "version": '1'
        }
    images = []
    annotations = []
    categories = [{"supercategory": "anomaly",
                "id": int(1),
                "name": "flaw"}]

    for d in acq_names: 
        # remove shitty character
        d = d.replace('Â','')
        dirpath = os.path.join(base_path, d, d)
        for sd in os.listdir(dirpath):
            print(f"Processing {d}/{sd}")


            # STEP 0: LOAD ASCANS AND LABELS

            # ascans
            subdirpath = os.path.join(dirpath, sd)
            obj = PAUT_acquisition(subdirpath, ann_path)
            ascans = obj.compose_Ascans(valid=False)
            # position of the ncrop in the bscan
            top_ncrop = obj.tmm_to_idx(obj.ncrop["top_mm"])
            left_ncrop = obj.ymm_to_idx(obj.ncrop["left_mm"])
            t_res = obj.metadata["t_res"]

            # labels
            labels = obj.get_labelsFromCSV()
            l_x = labels['Länge l'].values
            x_c = labels['X-Pos.'].values + l_x/2
            l_t = labels['Tiefe t'].values
            t_c = labels['Z-Pos.'].values + l_t/2
            l_y = labels['Breite b'].values
            y_c = labels['Y-Pos.'].values + l_y/2

            print(f"Found: {len(labels)} labelled defect(s)")
            print(f"ascans shape: {ascans.shape}")
            

            # compute median ncrop (ONLY NEEDED FOR MCH NCROPS) TODO: REMOVE
            median_ncrop = compute_median_ncrop(obj, ascans)


            # STEP 1: PREPARE ANNOTATIONS
            # In this step we create a list of "annotations drafts" (one for each anomaly)
            # that will be filled in the next steps, and a "lookup table" for anomalies
            # (i.e. a boolean matrix where each row corresponds to an anomaly and each column to a bscan)

            # create a "lookup table" for anomalies, and a draft for annotations
            has_anomalies = np.zeros((len(labels), ascans.shape[1]), dtype=bool)
            annotation_drafts = []

            for i in range(len(labels)):
                # prepare a draft for the current anomaly annotation
                annotation_drafts.append({
                    "acquisition": d,
                    "category_id": int(1),
                    "iscrowd": int(0),
                    "area": int(np.round((l_t[i]/t_res)*(l_y[i]/t_res), 0)),
                    "bbox": [int(np.round(obj.ymm_to_idx(y_c[i]-l_y[i]/2)-left_ncrop, 0)),
                            int(np.round(obj.tmm_to_idx(t_c[i]-l_t[i]/2)-top_ncrop, 0)),
                            int(np.round(l_y[i]/t_res, 0)),
                            int(np.round(l_t[i]/t_res, 0))
                            ]
                })

                # fill in the "lookup table" for current anomaly
                has_anomalies[i, obj.xmm_to_idx(x_c[i]-l_x[i]/2):obj.xmm_to_idx(x_c[i]+l_x[i]/2)] = True
            # store idxs where anomalies are present
            has_any_anomaly = np.any(has_anomalies, axis=0)

            # STEP 2: CREATE ANOMALOUS NCROPS
            # We loop again over the anomalies, and for each one we randomly select a number N_ANOM of 
            # bscans where the anomaly is visible and create a nugget crop from each of them; then
            # we fill-in the corresponding annotation draft, create the ncrop image description dict
            # and save the image to disk.

            # get anomalous ncrops
            for i in range(len(labels)):
                # anomalous idxs (erode to avoid selecting first and last bscans with anomalies)
                s = 5 if (np.sum(has_anomalies[i])>10) else 3
                eroded = binary_erosion(has_anomalies[i], structure=np.ones(s))
                anomalous_idxs = np.argwhere(eroded).flatten()

                # randomly select N_ANOM bscans with anomalies
                if len(anomalous_idxs)>n_anom:
                    selected_anomalous_idxs = np.random.choice(anomalous_idxs, n_anom, replace=False)
                else: 
                    selected_anomalous_idxs = anomalous_idxs

                # get nugget crops from selected bscans
                for i in selected_anomalous_idxs:
                    assert has_anomalies[:, i].any()

                    # get preprocessed nugget crop
                    crop = get_ncrop(ascans, i, obj, median_ncrop=median_ncrop)

                    # finish the annotation draft
                    img_id += 1
                    img_name = f"{d}_{i}.png"
                    for w in np.where(has_anomalies[:, i])[0]:
                        a_id += 1  
                        a = annotation_drafts[w].copy()
                        a["image_id"] = int(img_id)
                        a["id"] = int(a_id)
                        annotations.append(a)

                    # create the ncrop image description dict
                    images.append({
                        "acquisition": d,
                        "id": int(img_id),
                        "file_name": img_name,
                        "folder": "anomalous",
                        "width": int(crop.shape[1]),
                        "height": int(crop.shape[0])
                        })

                    # save the image to disk
                    im = Image.fromarray(crop)
                    im.save(os.path.join(save_path, "anomalous", img_name))



            # STEP 3: CREATE GOOD NCROPS
            # We randomly select a number N_GOOD of bscans without anomalies, extract the nugget crop
            # from each of them, create the corresponding ncrop image description dict and save the
            # image to disk.

            # valid range limits
            imin, imax = obj.metadata["x_valid_lim"]
            # dilate the has_anomalies matrix to avoid selecting bscans too close to the anomalies
            dilated = binary_dilation(has_any_anomaly, structure=np.ones(7))
            # randomly select N_GOOD idxs of bscans without anomalies from the valid range
            good_idxs = np.setdiff1d(np.arange(imin, imax), np.where(dilated)[0])
            selected_good_idxs = np.random.choice(good_idxs, n_good, replace=False)


            # get good nugget crops from selected bscans
            for i in selected_good_idxs:
                assert not has_anomalies[:, i].any()

                # get preprocessed nugget crop
                crop = get_ncrop(ascans, i, obj, median_ncrop=median_ncrop)

                img_id += 1
                img_name = f"{d}_{i}.png"

                # create the ncrop image description dict
                images.append({
                    "acquisition": d,
                    "id": int(img_id),
                    "file_name": img_name,
                    "folder": "normal",
                    "width": int(crop.shape[1]),
                    "height": int(crop.shape[0])
                    })

                # save the image to disk
                im = Image.fromarray(crop)
                im.save(os.path.join(save_path, "normal", img_name))


    with open(os.path.join(save_path, "annotations_v3.json"), "w") as fp:
        j = {
            "info": info,
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
        json.dump(j, fp, indent = 4, sort_keys=True)



if __name__ == "__main__":
    config = setupArgs()
    main(config)
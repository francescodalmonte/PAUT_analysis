# -*- coding: utf-8 -*-

import os
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from PIL import Image

from PAUT_Data import PAUT_Data
import json

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

a_id = -1 # anomaly counter
img_id = -1 # image counter

info = {
    "year": '2024',
    "description": '',
    "version": '1'
    }
images = []
annotations = []

for d in acq_names: 
    dirpath = os.path.join(basepath, d, d)
    for sd in os.listdir(dirpath):
        print(f"Processing {d}/{sd}")
        subdirpath = os.path.join(dirpath, sd)
        obj = PAUT_Data(subdirpath, labelpath=labelpath)
        labels = obj.get_labelsFromCSV()
        print(f"Found: {len(labels)} labelled defect(s)")

        ascans = obj.compose_Ascans(valid=False)
        print(ascans.shape)
        
        # prepare labels
        has_anomalies = np.zeros((len(labels), ascans.shape[1]), dtype=bool)

        annotation_drafts = []
        for i in range(len(labels)):

            l_x = labels['Länge l'].values[i]
            x_c = labels['X-Pos.'].values[i] + l_x/2

            l_t = labels['Tiefe t'].values[i]
            t_c = labels['Z-Pos.'].values[i] + l_t/2
            l_y = labels['Breite b'].values[i]
            y_c = labels['Y-Pos.'].values[i] + l_y/2

            top_ncrop = obj.tmm_to_idx(obj.ncrop["top_mm"])
            left_ncrop = obj.ymm_to_idx(obj.ncrop["left_mm"])

            # draft for annotation dict
            annotation_drafts.append({
                "acquisition": d,
                "category_id": int(1),
                "iscrowd": int(0),
                "area": int(np.round((l_x/obj.metadata["t_res"])*(l_y/obj.metadata["t_res"]), 0)),
                "bbox": [int(np.round(obj.ymm_to_idx(y_c-l_y/2)-left_ncrop, 0)),
                         int(np.round(obj.tmm_to_idx(t_c-l_t/2)-top_ncrop, 0)),
                         int(np.round(l_y/obj.metadata["t_res"], 0)),
                         int(np.round(l_t/obj.metadata["t_res"], 0))
                         ]
            })
            has_anomalies[i, obj.xmm_to_idx(x_c-l_x/2):obj.xmm_to_idx(x_c+l_x/2)] = True


        # get anomalous ncrops
        for i in range(len(labels)):
            l_x = labels['Länge l'].values[i]
            x_c = labels['X-Pos.'].values[i] + l_x/2
            l_t = labels['Tiefe t'].values[i]
            t_c = labels['Z-Pos.'].values[i] + l_t/2
            l_y = labels['Breite b'].values[i]
            y_c = labels['Y-Pos.'].values[i] + l_y/2

            # anomalous idxs 
            all_anomalous_idxs = np.arange(obj.xmm_to_idx(x_c-l_x/2), obj.xmm_to_idx(x_c+l_x/2))
            if len(all_anomalous_idxs)>5:
                anomalous_idxs = np.random.choice(all_anomalous_idxs, 5, replace=False)
            else: 
                anomalous_idxs = all_anomalous_idxs

            # get nugget crops from selected bscans
            for i in anomalous_idxs:
                bscan = obj.extract_Bscan(ascans, i, correction = True)
                crop = obj.extract_ncrop(bscan)

                crop = np.round(np.sqrt(crop)*255/10, 0).astype(np.uint8)

                # assign label to img
                assert has_anomalies[:, i].any()
                img_id += 1
                img_name = f"{d}_{i}.png"

                for w in np.where(has_anomalies[:, i])[0]:
                    a_id += 1  
                    a = annotation_drafts[w].copy()
                    a["image_id"] = int(img_id)
                    a["id"] = int(a_id)
                    annotations.append(a)

                images.append({
                    "acquisition": d,
                    "id": int(img_id),
                    "file_name": img_name,
                    "width": int(crop.shape[1]),
                    "height": int(crop.shape[0])
                    })


                im = Image.fromarray(crop)
                im.save("C:/users/dalmonte/data/ncrops_ds/anomalous/"+img_name)


        # get normal ncrops
        imin, imax = obj.metadata["x_valid_lim"]

        good_idxs = np.random.choice(np.setdiff1d(np.arange(imin, imax), all_anomalous_idxs), 100, replace=False)
        for i in good_idxs:
            img_id += 1
            img_name = f"{d}_{i}.png"

            cscan = obj.extract_Bscan(ascans, i, correction = True)
            crop = obj.extract_ncrop(cscan)

            crop = np.round(np.sqrt(crop)*255/10, 0).astype(np.uint8)

            im = Image.fromarray(crop)
            im.save("C:/users/dalmonte/data/ncrops_ds/normal/"+img_name)

            images.append({
                "acquisition": d,
                "id": int(img_id),
                "file_name": img_name,
                "width": int(crop.shape[1]),
                "height": int(crop.shape[0])
                })
            

with open("C:/users/dalmonte/data/ncrops_ds/annotations.json", "w") as fp:
    j = {
        "info": info,
        "images": images,
        "annotations": annotations
    }
    json.dump(j, fp, indent = 4, sort_keys=True)
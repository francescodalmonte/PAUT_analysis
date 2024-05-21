# -*- coding: utf-8 -*-

import os
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from PIL import Image

from PAUT_Data import PAUT_Data
import json

import pandas as pd

acq_names = ["1152811 45 S22 16dB",
             "FA1153222 SH83 LF45° 21dB",
             "FA1154309 SH83 LF45 22dB",
             "FA1156470_23.03.21_S12_LF45°_18dB",
             "FA3566742_NI2_45_Aufwärts",
             "FA3566742_NI5_45_Aufwärts",
             "FA3569081_SH83_LF45°",
             "FA3569143 S43 45°"
]

in_path = "C:/Users/dalmonte/data/ADAMUS/ncrops datasets/ncrops_ds_V3"
out_path = "C:/Users/dalmonte/data/ADAMUS/ncrops datasets/ncrops_ADds"

normal_in = os.path.join(in_path, "normal")
anomalous_in = os.path.join(in_path, "anomalous")
train_normal_out = os.path.join(out_path, "train", "normal")
train_anomalous_out = os.path.join(out_path, "train", "anomalous")
test_normal_out = os.path.join(out_path, "test", "normal")
test_anomalous_out = os.path.join(out_path, "test", "anomalous")

for file in os.listdir(normal_in):
    if file.endswith(".png"):
        n = "".join(file.split(".")[0].rsplit("_", 1)[:-1])
        print(n)
        if n in acq_names:
            img = Image.open(os.path.join(normal_in, file))
            img.save(os.path.join(test_normal_out, file))
        else:
            img = Image.open(os.path.join(normal_in, file))
            img.save(os.path.join(train_normal_out, file))

for file in os.listdir(anomalous_in):
    if file.endswith(".png"):
        n = "".join(file.split(".")[0].rsplit("_", 1)[:-1])
        if n in acq_names:
            img = Image.open(os.path.join(anomalous_in, file))
            img.save(os.path.join(test_anomalous_out, file))

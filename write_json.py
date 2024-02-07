import os
import json

from PAUT_Data import PAUT_Data


rootpath = "D:/data_ADAMUS/ADAMUS/DFKI_PAUT"


for d in os.listdir(rootpath):  
    if (not d.endswith(".zip")) and (not d.endswith(".ps1")) and (not d.endswith(".json")):
        print(d)
        names = os.listdir(os.path.join(rootpath, f"{d}/{d}"))
        if len(names)>0:
            for n in names:
                dirpath = os.path.join(rootpath, f"{d}/{d}/{n}")
                obj = PAUT_Data(dirpath)

                single_dict = {}
                single_dict["dirname1"] = d
                single_dict["dirname2"] = n
                single_dict["info"] = obj.infoDict

                with open(os.path.join(dirpath, "DATASET_INFO.json"), "a") as file: 
                    json.dump(single_dict, file, indent=4)

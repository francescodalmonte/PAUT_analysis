from collections import defaultdict
import json

import numpy as np
from  matplotlib import pyplot as plt
from PIL import Image
import os


class COCOParser:
    def __init__(self, anns_file, imgs_dir):
        with open(anns_file, 'r') as f:
            coco = json.load(f)
            
        self.annIm_dict = defaultdict(list)        
        self.cat_dict = {} 
        self.annId_dict = {}
        self.im_dict = {}

        for ann in coco['annotations']:           
            self.annIm_dict[ann['image_id']].append(ann) 
            self.annId_dict[ann['id']]=ann
        for img in coco['images']:
            self.im_dict[img['id']] = img
        for cat in coco['categories']:
            self.cat_dict[cat['id']] = cat

    def get_imgIds(self):
        return list(self.im_dict.keys())
    def get_annIds(self, im_ids):
        im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
        return [ann['id'] for im_id in im_ids for ann in self.annIm_dict[im_id]]
    def load_anns(self, ann_ids):
        ann_ids=ann_ids if isinstance(ann_ids, list) else [ann_ids]
        return [self.annId_dict[ann_id] for ann_id in ann_ids]        
    def load_cats(self, class_ids):
        class_ids=class_ids if isinstance(class_ids, list) else [class_ids]
        return [self.cat_dict[class_id] for class_id in class_ids]


annotations_path = "C:/users/dalmonte/data/ncrops_ds/annotations.json"
images_path = "C:/users/dalmonte/data/ncrops_ds/"
cocop = COCOParser(annotations_path, images_path)




# define a list of colors for drawing bounding boxes
color_list = ["red","orange","yellow","green","blue","purple","brown","pink"]

N = 25
total_images = len(cocop.get_imgIds()) # total number of images
img_ids = cocop.get_imgIds()
selected_img_ids = [img_ids[i] for i in np.random.permutation(total_images)[:N]]
ann_ids = cocop.get_annIds(selected_img_ids)


fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(12,12))
ax = ax.ravel()
for i, im in enumerate(selected_img_ids):

    image = Image.open(f"{images_path}/{cocop.im_dict[im]["folder"]}/{cocop.im_dict[im]["file_name"]}")
    ann_ids = cocop.get_annIds(im)
    annotations = cocop.load_anns(ann_ids)
    for ann in annotations:
        bbox = ann['bbox']
        x, y, w, h = [int(b) for b in bbox]
        class_id = ann["category_id"]
        class_name = cocop.load_cats(class_id)[0]["name"]
        color_ = color_list[class_id]
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color_, facecolor='none')

        t_box=ax[i].text(x, y, class_name,  color='red', fontsize=10)
        t_box.set_bbox(dict(boxstyle='square, pad=0',facecolor='white', alpha=0.6, edgecolor='blue'))
        ax[i].add_patch(rect)
    ax[i].set_title(f"{cocop.im_dict[im]["acquisition"]}", fontsize=6)
    ax[i].axis('off')
    ax[i].imshow(image, vmin=0, vmax=255, cmap='jet')
plt.tight_layout()
plt.show()
#############
## Imports ##
#############

from glob import glob
import numpy as np

###############
## Functions ##
###############

def get_img_ids(images_folder):
    img_ids = glob(images_folder + "/*.jpg")
    img_ids = list(map(lambda x: x.split("/")[-1], img_ids))
    img_ids = list(map(lambda x: x.split(".jpg")[0], img_ids))
    img_ids = list(map(lambda x: x.split("_mask")[0], img_ids))
    img_ids = list(map(lambda x: x.split("_img")[0], img_ids))
    img_ids = list(set(img_ids))
    return img_ids

def train_val_split(img_ids, train_prop):
    img_ids = np.array(img_ids)
    train_len = int(len(img_ids) * train_prop)
    np.random.shuffle(img_ids)
    train_img_ids = img_ids[:train_len]
    val_img_ids = img_ids[train_len:]
    return train_img_ids, val_img_ids
#############
## Imports ##
#############

""" Global """
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

""" Local """
import constants

###############
## Functions ##
###############

def get_img_id(path):
    return path.split("/")[-1].split(".jpg")[0].split("_mask")[0]

def get_img_ids(img_folder=constants.IMAGES_FOLDER):
    img_paths = glob(img_folder + "/*.jpg")
    return list(set(list(map(get_img_id, img_paths))))

def read_img_and_mask(img_id, img_folder=constants.IMAGES_FOLDER):
    img_path_recipient = img_folder + "/{}.jpg"
    mask_path_recipient = img_folder + "/{}_mask.jpg"
    img = cv2.imread(img_path_recipient.format(img_id))
    mask = cv2.imread(mask_path_recipient.format(img_id))
    return img, mask

def extract_and_save_patches(img, mask, img_id, out_folder=constants.PATCHES_FOLDER, patch_size=constants.PATCH_SIZE):
    counter = 0
    for i in range(0, img.shape[0], patch_size):
        for j in range(0, img.shape[1], patch_size):
            counter += 1
            img_patch = img[i:i+patch_size, j:j+patch_size]
            mask_patch = mask[i:i+patch_size, j:j+patch_size]
            save_patch(img_patch, mask_patch, img_id, counter, out_folder, size=patch_size)

def save_patch(img_patch, mask_patch, img_id, patch_number, out_folder=constants.PATCHES_FOLDER, size=constants.PATCH_SIZE):
    img_path = "{}/{}_{}.jpg".format(out_folder, img_id, patch_number)
    mask_path = "{}/{}_{}_mask.jpg".format(out_folder, img_id, patch_number)
    img_patch = cv2.resize(img_patch, (size, size))
    mask_patch = cv2.resize(mask_patch, (size, size))
    cv2.imwrite(img_path, img_patch)
    cv2.imwrite(mask_path, mask_patch)

##########
## MAIN ##
##########

if __name__ == "__main__":
    img_ids = get_img_ids()
    for img_id in tqdm(img_ids):
        img, mask = read_img_and_mask(img_id)
        extract_and_save_patches(img, mask, img_id)
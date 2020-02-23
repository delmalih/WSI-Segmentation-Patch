#############
## Imports ##
#############

""" Global """
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse

""" Local """
import constants
import utils

###############
## Functions ##
###############

def read_img_and_mask(img_id, img_folder):
    img_path_recipient = img_folder + "/{}.jpg"
    mask_path_recipient = img_folder + "/{}_mask.jpg"
    img = cv2.imread(img_path_recipient.format(img_id))
    mask = cv2.imread(mask_path_recipient.format(img_id))
    return img, mask

def extract_patches(img_id, img_folder, patch_size):
    img, mask = read_img_and_mask(img_id, img_folder)
    patches_img = []
    patches_mask = []
    for i in range(0, img.shape[0], patch_size):
        for j in range(0, img.shape[1], patch_size):
            patch_img = cv2.resize(img[i:i+patch_size, j:j+patch_size, :], (patch_size, patch_size))
            patch_mask = cv2.resize(mask[i:i+patch_size, j:j+patch_size], (patch_size, patch_size))
            if patch_mask.sum() > 0 or np.random.random() < 0.2:
                patches_img.append(patch_img)
                patches_mask.append(patch_mask)
    patches_img = np.array(patches_img)
    patches_mask = np.array(patches_mask)
    return patches_img, patches_mask

##########
## MAIN ##
##########

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training")
    parser.add_argument("-i", "--images_folder", dest="images_folder", help="Path to images (and mask) folder", required=True)
    parser.add_argument("-o", "--output_folder", dest="output_folder", help="Path to the output folder (for patches)", required=True)
    parser.add_argument("-ps", "--patch_size", dest="patch_size", help="Patch size", default=constants.PATCH_SIZE, type=int)
    parser.add_argument("-p", "--train_prop", dest="train_prop", help="Proportion of train set", default=constants.TRAIN_PROPORTION, type=float)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    img_ids = utils.get_img_ids(args.images_folder)
    train_img_ids, val_img_ids = utils.train_val_split(img_ids, args.train_prop)
    for img_id in tqdm(train_img_ids):
        patches_img, patches_mask = extract_patches(img_id, args.images_folder, args.patch_size)
        for k in range(len(patches_img)):
            cv2.imwrite("{}/train/images/{}_{}.jpg".format(args.output_folder, img_id, k), patches_img[k])
            cv2.imwrite("{}/train/masks/{}_{}.jpg".format(args.output_folder, img_id, k), patches_mask[k])
    for img_id in tqdm(val_img_ids):
        patches_img, patches_mask = extract_patches(img_id, args.images_folder, args.patch_size)
        for k in range(len(patches_img)):
            cv2.imwrite("{}/val/images/{}_{}.jpg".format(args.output_folder, img_id, k), patches_img[k])
            cv2.imwrite("{}/val/masks/{}_{}.jpg".format(args.output_folder, img_id, k), patches_mask[k])

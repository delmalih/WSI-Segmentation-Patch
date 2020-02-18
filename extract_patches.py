#############
## Imports ##
#############

""" Global """
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import numpy as np

""" Local """
import constants

###############
## Functions ##
###############

def read_img_and_mask(img_id, img_folder=constants.IMAGES_FOLDER):
    img_path_recipient = img_folder + "/{}.jpg"
    mask_path_recipient = img_folder + "/{}_mask.jpg"
    img = cv2.imread(img_path_recipient.format(img_id))
    mask = cv2.imread(mask_path_recipient.format(img_id))
    return img, mask

def extract_patches(img_id, n_patches, patch_size=constants.PATCH_SIZE):
    img, mask = read_img_and_mask(img_id)
    patches_img = []
    patches_mask = []
    for _ in range(n_patches):
        patch_i = np.random.randint(0, img.shape[0] - patch_size)
        patch_j = np.random.randint(0, img.shape[1] - patch_size)
        patch_img = img[patch_i:patch_i+patch_size, patch_j:patch_j+patch_size, :]
        patch_mask = np.mean(mask[patch_i:patch_i+patch_size, patch_j:patch_j+patch_size], axis=-1, keepdims=True)
        patches_img.append(patch_img)
        patches_mask.append(patch_mask)
    patches_img = np.array(patches_img)
    patches_mask = np.array(patches_mask)
    return patches_img, patches_mask

def extract_patches_batch(img_ids, n_patches, patch_size=constants.PATCH_SIZE):
    patches_imgs = []
    patches_masks = []
    for img_id in img_ids:
        patches_img, patches_mask = extract_patches(img_id, n_patches, patch_size=patch_size)
        patches_imgs.append(patches_img)
        patches_masks.append(patches_mask)
    patches_imgs = np.array(patches_imgs)
    patches_masks = np.array(patches_masks)
    patches_imgs = patches_imgs.reshape(-1, patch_size, patch_size, 3)
    patches_masks = patches_masks.reshape(-1, patch_size, patch_size, 1)
    return patches_imgs, patches_masks
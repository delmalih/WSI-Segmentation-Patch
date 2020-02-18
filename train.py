#############
## Imports ##
#############

""" Global """
import numpy as np
from glob import glob
import argparse

""" Local """
import constants
import utils
import Model
from extract_patches import extract_patches_batch

###############
## Functions ##
###############

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training")
    parser.add_argument("-i", "--images_folder", dest="images_folder", help="Path to images (and mask) folder", required=True)
    parser.add_argument("-m", "--model_path", dest="model_path", help="Path to the model weights", required=True)
    parser.add_argument("-bs", "--batch_size", dest="batch_size", help="Batch size", default=constants.BATCH_SIZE, type=int)
    parser.add_argument("-ps", "--patch_size", dest="patch_size", help="Patch size", default=constants.PATCH_SIZE, type=int)
    parser.add_argument("-e", "--epochs", dest="epochs", help="Number of epochs", default=constants.N_EPOCHS, type=int)
    parser.add_argument("-np", "--n_patches", dest="n_patches", help="Number of patches per image", default=constants.N_PATCHES, type=int)
    return parser.parse_args()

def get_batch(img_ids, img_folder, n_patches, batch_size, patch_size):
    batch_img_ids = np.random.choice(img_ids, size=batch_size)
    imgs, masks = extract_patches_batch(batch_img_ids, img_folder, n_patches, patch_size)
    imgs = imgs / 255.
    masks = (masks > 128).astype(float)
    print(masks.mean())
    return imgs, masks

def train(model, img_ids, args):
    for epoch in range(args.epochs):
        imgs, masks = get_batch(img_ids, args.images_folder, args.n_patches, args.batch_size, args.patch_size)
        loss, metric = model.train_on_batch(imgs, masks)
        model.save(args.model_path)
        print("Epoch {} | Loss (FL) = {} | F1 Score = {}%".format(epoch, loss, metric * 100))

##########
## MAIN ##
##########

if __name__ == "__main__":
    args = parse_args()
    img_ids = utils.get_img_ids(args.images_folder)
    wsi_segmenter = Model.wsi_segmenter(args.patch_size)
    train(wsi_segmenter, img_ids, args)

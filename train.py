#############
## Imports ##
#############

""" Global """
import cv2
import numpy as np
from glob import glob
import argparse

""" Local """
import constants
import utils
import Model

###############
## Functions ##
###############

def read_image(img_path, patch_size):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (patch_size, patch_size))
    img = img[:, :, ::-1] / 255.
    return img

def read_mask(mask_path, patch_size):
    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, (patch_size, patch_size))
    mask = np.mean(mask, axis=-1, keepdims=True)
    mask = (mask > 128).astype(np.float32)
    return mask

def get_batch(img_ids, img_folder, batch_size, patch_size):
    images = []; masks = []
    while len(images) < batch_size:
        img_id = np.random.choice(img_ids)
        img = read_image("{}/images/{}.jpg".format(img_folder, img_id), patch_size)
        mask = read_mask("{}/masks/{}.jpg".format(img_folder, img_id), patch_size)
        if mask.sum() > 0:
            images.append(img)
            masks.append(mask)
    images = np.array(images)
    masks = np.array(masks)
    return images, masks

def train(model, img_ids, args):
    for epoch in range(args.epochs):
        imgs, masks = get_batch(img_ids, args.images_folder, args.batch_size, args.patch_size)
        loss, acc, f1 = model.train_on_batch(imgs, masks)
        model.save(args.model_path)
        print("Epoch {} | Loss = {:.3f} | Accuracy = {:.2f}% | F1 Score = {:.2f}%".format(epoch, loss, acc * 100, f1 * 100))

##########
## MAIN ##
##########

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training")
    parser.add_argument("-i", "--images_folder", dest="images_folder", help="Path to patches images (and mask) folder", required=True)
    parser.add_argument("-m", "--model_path", dest="model_path", help="Path to the model weights", required=True)
    parser.add_argument("-ps", "--patch_size", dest="patch_size", help="Patch size", default=constants.PATCH_SIZE, type=int)
    parser.add_argument("-bs", "--batch_size", dest="batch_size", help="Batch size", default=constants.BATCH_SIZE, type=int)
    parser.add_argument("-e", "--epochs", dest="epochs", help="Number of epochs", default=constants.N_EPOCHS, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    img_ids = list(set(list(map(lambda x: x.split("/")[-1][:-4], glob(args.images_folder + "/**/*.jpg")))))
    wsi_segmenter = Model.wsi_segmenter(args.patch_size)
    wsi_segmenter.summary()
    train(wsi_segmenter, img_ids, args)

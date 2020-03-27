#############
## Imports ##
#############

""" Global """
import cv2
import argparse
import numpy as np
from glob import glob
from tensorflow import keras

""" Local """
import utils
import Model
import constants
from datagenerator import DataGenerator

###############
## Functions ##
###############

def train(model, train_img_ids, val_img_ids, args):
    train_generator = DataGenerator(train_img_ids, args.train_images_folder, args.batch_size, args.patch_size)
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=args.model_path, verbose=1, save_best_only=False)
    if len(val_img_ids) > 0:
        val_generator = DataGenerator(val_img_ids, args.val_images_folder, args.batch_size, args.patch_size)
    else:
        val_generator = None
    model.fit_generator(generator=train_generator,
                        validation_data=val_generator,
                        epochs=args.epochs, verbose=1,
                        callbacks=[checkpointer],
                        use_multiprocessing=False)

##########
## MAIN ##
##########

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training")
    parser.add_argument("-ti", "--train_images_folder", dest="train_images_folder", help="Path to train patches images (and mask) folder", required=True)
    parser.add_argument("-vi", "--val_images_folder", dest="val_images_folder", help="Path to val patches images (and mask) folder", required=True)
    parser.add_argument("-m", "--model_path", dest="model_path", help="Path to the model weights", required=True)
    parser.add_argument("-ps", "--patch_size", dest="patch_size", help="Patch size", default=constants.PATCH_SIZE, type=int)
    parser.add_argument("-bs", "--batch_size", dest="batch_size", help="Batch size", default=constants.BATCH_SIZE, type=int)
    parser.add_argument("-e", "--epochs", dest="epochs", help="Number of epochs", default=constants.N_EPOCHS, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_img_ids = list(set(list(map(lambda x: x.split("/")[-1][:-4], glob(args.train_images_folder + "/**/*.jpg")))))
    val_img_ids = list(set(list(map(lambda x: x.split("/")[-1][:-4], glob(args.val_images_folder + "/**/*.jpg")))))
    wsi_segmenter = Model.wsi_segmenter(args.patch_size)
    wsi_segmenter.summary()
    train(wsi_segmenter, train_img_ids, val_img_ids, args)

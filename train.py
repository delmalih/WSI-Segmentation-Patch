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

def train(model, img_ids, args):
    train_generator = DataGenerator(img_ids, args.images_folder, args.batch_size, args.patch_size)
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=args.model_path, verbose=1, save_best_only=False)
    model.fit_generator(generator=train_generator,
                        epochs=args.epochs, verbose=1,
                        callbacks=[checkpointer],
                        steps_per_epoch=args.steps_per_epoch)

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
    parser.add_argument("-spe", "--steps_per_epoch", dest="steps_per_epoch", help="Number of steps per epochs", default=constants.STEPS_PER_EPOCH, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    img_ids = list(set(list(map(lambda x: x.split("/")[-1][:-4], glob(args.images_folder + "/**/*.jpg")))))
    wsi_segmenter = Model.wsi_segmenter(args.patch_size)
    wsi_segmenter.summary()
    train(wsi_segmenter, img_ids, args)

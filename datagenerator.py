#############
## Imports ##
#############

import cv2
import numpy as np
from tensorflow import keras

###################
## DataGenerator ##
###################

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

class DataGenerator(keras.utils.Sequence):

    def __init__(self, img_ids, img_folder, batch_size, patch_size, shuffle=True):
        """ Initialisation """
        self.img_ids = img_ids
        self.img_folder = img_folder
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.indexes = np.arange(len(self.img_ids))
        if self.shuffle: np.random.shuffle(self.indexes)
    
    def __data_generation(self, img_ids_temp):
        """ Generates data containing batch_size samples """
        # Initialization
        X = np.empty((self.batch_size, self.patch_size, self.patch_size, 3))
        y = np.empty((self.batch_size, self.patch_size, self.patch_size, 1))

        # Generate data
        for i, img_id in enumerate(img_ids_temp):
            X[i,] = read_image("{}/images/{}.jpg".format(self.img_folder, img_id), self.patch_size)
            y[i,] = read_mask("{}/masks/{}.jpg".format(self.img_folder, img_id), self.patch_size)
        
        return X, y
    
    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.floor(len(self.img_ids) / self.batch_size))
    
    def __getitem__(self, index):
        """ Generate one batch of data """
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        img_ids_temp = [self.img_ids[k] for k in indexes]
        X, y = self.__data_generation(img_ids_temp)
        return X, y
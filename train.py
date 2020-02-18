#############
## Imports ##
#############

""" Global """
import numpy as np
from glob import glob

""" Local """
import constants
import utils
import Model
from extract_patches import extract_patches_batch

###############
## Functions ##
###############

def get_batch(img_ids, n_patches=constants.N_PATCHES, batch_size=constants.BATCH_SIZE, patch_size=constants.PATCH_SIZE):
    batch_img_ids = np.random.choice(img_ids, size=batch_size)
    imgs, masks = extract_patches_batch(batch_img_ids, n_patches, patch_size)
    imgs = imgs / 255.
    masks = (masks > 128).astype(float)
    print(imgs.shape, masks.shape)
    return imgs, masks

def train(model, img_ids, n_epochs=constants.N_EPOCHS, batch_size=constants.BATCH_SIZE,
          n_patches=constants.N_PATCHES, patch_size=constants.PATCH_SIZE, model_path=constants.MODEL_PATH):
    for epoch in range(n_epochs):
        imgs, masks = get_batch(img_ids, n_patches=n_patches, batch_size=batch_size, patch_size=patch_size)
        loss, acc = model.train_on_batch(imgs, masks)
        print("Epoch {} | Loss = {} | Acc = {}%".format(epoch, loss, acc * 100))
        model.save(model_path)

if __name__ == "__main__":
    img_ids = utils.get_img_ids(constants.IMAGES_FOLDER)
    wsi_segmenter = Model.wsi_segmenter(constants.PATCH_SIZE)
    train(wsi_segmenter, img_ids)

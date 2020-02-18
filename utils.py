#############
## Imports ##
#############

from glob import glob

###############
## Functions ##
###############

def get_img_ids(images_folder):
    img_ids = glob(images_folder + "/*.jpg")
    img_ids = list(map(lambda x: x.split("/")[-1], img_ids))
    img_ids = list(map(lambda x: x.split(".jpg")[0], img_ids))
    img_ids = list(map(lambda x: x.split("_mask")[0], img_ids))
    img_ids = list(set(img_ids))
    return img_ids
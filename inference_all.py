#############
## Imports ##
#############

""" Global """
import os
import argparse

""" Local """
import constants
import utils

##########
## MAIN ##
##########

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training")
    parser.add_argument("-i", "--image_folder", dest="image_folder", help="Path to image folder", required=True)
    parser.add_argument("-m", "--model_path", dest="model_path", help="Path to the model weights", required=True)
    parser.add_argument("-o", "--output_path", dest="output_path", help="Path to save the predicted masks", required=True)
    parser.add_argument("-ps", "--patch_size", dest="patch_size", help="Patch size", default=constants.PATCH_SIZE, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    img_ids = utils.get_img_ids(args.image_folder)
    for img_id in img_ids:
        print("==== {} ====".format(img_id))
        command = "python inference.py -i \"{}/{}.jpg\" -m \"{}\" -o \"{}/{}_mask\" -ps {}".format(
            args.image_folder, img_id,
            args.model_path,
            args.output_path, img_id,
            args.patch_size
        )
        os.system(command)

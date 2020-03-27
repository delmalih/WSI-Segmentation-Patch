#############
## Imports ##
#############

""" Global """
import argparse
from glob import glob
import cv2
import numpy as np

###############
## Functions ##
###############

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", dest="folder", help="Path to masks folder", required=True)
    return parser.parse_args()

def read_image(path):
    mask = cv2.imread(path) / 255.
    mask = mask[:, :, 0].astype(np.bool)
    return mask

def compute_iou(gt, pred):
    intersection = np.sum(gt * pred)
    union = np.clip(gt + pred, 0, 1).sum()
    return 1. * intersection / union

def compute_f1_score(gt, pred):
    tp = ((gt == 1) * (pred == 1)).sum()
    fp = ((gt == 0) * (pred == 1)).sum()
    fn = ((gt == 1) * (pred == 0)).sum()
    f1_score = 2. * tp / (2 * tp + fp + fn)
    return f1_score

##########
## MAIN ##
##########

if __name__ == "__main__":
    args = parse_args()
    img_ids = set(list(map(lambda x: x.split("/")[-1][:-7], glob("{}/*-gt.jpg".format(args.folder)))))
    for img_id in img_ids:
        gt_image = read_image("{}/{}-gt.jpg".format(args.folder, img_id))
        pred_image = read_image("{}/{}-thresh.jpg".format(args.folder, img_id))
        iou = compute_iou(gt_image, pred_image)
        f1_score = compute_f1_score(gt_image, pred_image)
        print("{} : IoU = {}| F1 score = {}".format(
            img_id, iou, f1_score
        ))
        
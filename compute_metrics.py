#############
## Imports ##
#############

""" Global """
import argparse
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    intersection = (gt * pred).sum()
    union = np.clip(gt + pred, 0, 1).sum()
    return 1. * intersection / union

def compute_dsc(gt, pred):
    num = 2. * (gt * pred).sum()
    den = gt.sum() + pred.sum()
    return num / den

##########
## MAIN ##
##########

if __name__ == "__main__":
    args = parse_args()
    img_ids = list(set(list(map(lambda x: x.split("/")[-1][:-7], glob("{}/*-gt.jpg".format(args.folder))))))
    res = {}
    for k, img_id in enumerate(img_ids):
        gt_image = read_image("{}/{}-gt.jpg".format(args.folder, img_id))
        pred_image = read_image("{}/{}-thresh.jpg".format(args.folder, img_id))
        iou = compute_iou(gt_image, pred_image)
        dsc = compute_dsc(gt_image, pred_image)
        print("[{:03d}/{:3d}] {:62s} : IoU = {:.2f}% | DSC = {:.2f}%".format(
            k+1, len(img_ids), img_id,
            iou * 100, dsc * 100
        ))
        res[img_id] = {
            "size": gt_image.size,
            "log_size": np.log(gt_image.size),
            "iou": iou,
            "dsc": dsc,
        }
    print("Mean IoU = {}".format(np.mean([res[img_id]["iou"] for img_id in res])))
    print("Mean DSC = {}".format(np.mean([res[img_id]["dsc"] for img_id in res])))
    
    plt.figure()
    plt.plot([res[img_id]["log_size"] for img_id in res], [res[img_id]["iou"] for img_id in res], 'o', label="IoU")
    plt.plot([res[img_id]["log_size"] for img_id in res], [res[img_id]["dsc"] for img_id in res], 'o', label="DSC")
    plt.ylabel("Metric (%)")
    plt.xlabel("Log size of images")
    plt.legend()
    plt.show()
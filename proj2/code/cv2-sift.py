# Local Feature Stencil Code
# Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech with Henry Hu <henryhu@gatech.edu>
# Edited by James Tompkin
# Adapted for python by asabel and jdemari1 (2019)

import cv2
import csv
import sys
import argparse
import numpy as np

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from skimage import io, filters, feature, img_as_float32
from skimage.transform import rescale
from skimage.color import rgb2gray

def load_data(file_name):

    # Note: these files default to notre dame, unless otherwise specified
    image1_file = "../data/NotreDame/NotreDame1.jpg"
    image2_file = "../data/NotreDame/NotreDame2.jpg"

    eval_file = "../data/NotreDame/NotreDameEval.mat"

    if file_name == "notre_dame":
        pass
    elif file_name == "mt_rushmore":
        image1_file = "../data/MountRushmore/Mount_Rushmore1.jpg"
        image2_file = "../data/MountRushmore/Mount_Rushmore2.jpg"
        eval_file = "../data/MountRushmore/MountRushmoreEval.mat"
    elif file_name == "e_gaudi":
        image1_file = "../data/EpiscopalGaudi/EGaudi_1.jpg"
        image2_file = "../data/EpiscopalGaudi/EGaudi_2.jpg"
        eval_file = "../data/EpiscopalGaudi/EGaudiEval.mat"



    image1 = cv2.imread(image1_file)
    image2 =cv2.imread(image2_file)

    rows,cols=image1.shape[:2]

    # image2=cv2.resize(image2,(int(cols/2),int(rows/2)),interpolation=cv2.INTER_CUBIC)
    # image1 = cv2.resize(image1, (int(cols/2),int(rows/2)), interpolation=cv2.INTER_CUBIC)
    image2 = cv2.resize(image2, (cols,rows), interpolation=cv2.INTER_CUBIC)

    return image1, image2, eval_file


def main():
    """
    Reads in the data,

    Command line usage: python main.py [-a | --average_accuracy] -p | --pair <image pair name>

    -a | --average_accuracy - flag - if specified, will compute your solution's
    average accuracy on the (1) notre dame, (2) mt. rushmore, and (3) episcopal
    guadi image pairs

    -p | --pair - flag - required. specifies which image pair to match

    """

    # create the command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--average_accuracy",
                        help="Include this flag to compute the average accuracy of your matching.")
    parser.add_argument("-p", "--pair", required=True,
                        help="Either notre_dame, mt_rushmore, or e_gaudi. Specifies which image pair to match")

    args = parser.parse_args()

    # (1) Load in the data
    image1, image2, eval_file = load_data(args.pair)


    scale_factor = 0.5
    feature_width = 16

    print("Getting interest points...")

    orb = cv2.ORB_create()
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(image1, None)  # des是描述子
    kp2, des2 = sift.detectAndCompute(image2,None)

    # read the code of drawKeypoints
    img3 = cv2.drawKeypoints(image1, kp1, image1)
    img4 = cv2.drawKeypoints(image2, kp2, image2)

    save_file_path="../result/"+args.pair
    cv2.imwrite(save_file_path+"1-test.jpg",img3)
    cv2.imwrite(save_file_path+'2-test.jpg',img4)

    hmerge = np.hstack((img3, img4))  # 水平拼接
   # cv2.imshow("point", hmerge)  # 拼接显示为gray
   # cv2.waitKey(0)

    # read the code of BFMatcher
    # BFMatcher解决匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # 调整ratio
    good = []
    for m, n in matches:
        if m.distance < float(args.average_accuracy)* n.distance:
            good.append([m])
            print("success!")

    img5 = cv2.drawMatchesKnn(image1, kp1, image2, kp2, good, None, flags=2)

    cv2.imwrite(save_file_path+"compaired-test.jpg",img5)
    # cv2.imshow("ORB", img5)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__=="__main__":
    main()
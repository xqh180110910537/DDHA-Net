import os
from argparse import ArgumentParser
import time

import numpy
import numpy as np
import cv2
import matplotlib
from mmcv.cnn.utils import revert_sync_batchnorm
import matplotlib.pyplot as plt
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot, show_result_pyplot_new, \
    inference_segmentor_logit

IMG_WIDTH, IMG_HEIGHT = 1280, 1280


def crop_image_from_gray(img, tol=7):
    """
    Applies masks to the orignal image and
    returns the a preprocessed image with
    3 channels

    :param img: A NumPy Array that will be cropped
    :param tol: The tolerance used for masking

    :return: A NumPy array containing the cropped image
    """
    # If for some reason we only have two channels
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    # If we have a normal RGB images
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]

        if check_shape == 0:  # image is too dark so that we crop out everything,
            return img  # return original image
        else:

            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


# 每张眼底图像转化
def preprocess_image(image, sigmaX=15):
    # image preprocessing
    image = cv2.imread(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)

    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    img_ori = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    return image, img_ori


def count_elements(matrix):
    counts = {}
    for row in matrix:
        for element in row:
            if element in counts:
                counts[element] += 1
            else:
                counts[element] = 1
    return counts


config_file = './HACDRNet&DDHANet/DDHANet_idrid.py'
checkpoint_file = 'idrid_ddha.pth'
img_dir = '/disk1/xuqihao/concept_base/concept_dataset/new_dataset/process_image/'
model = init_segmentor(config_file, checkpoint_file, device='cuda:5')
model = revert_sync_batchnorm(model)
l = []
name_list = []
PALETTE = [[0, 0, 0], [255, 0, 0], [255, 255, 0], [255, 255, 255], [0, 255, 0]]
for name in os.listdir(img_dir):
    # img, img_ori = preprocess_image(os.path.join(img_dir, name))
    # cv2.imwrite(f'./data/img_dir_ori/test_ddr/{name[:-4]}'+'.jpg',img_ori)
    img=cv2.imread(os.path.join(img_dir, name))
    result = inference_segmentor(model, img)
    mask = numpy.zeros((img.shape[0], img.shape[1], 3), dtype=numpy.int8)
    show_result_pyplot_new(model, mask, result[0], opacity=1, show=False,
                           out_file=f'/disk1/xuqihao/peruade/{name[:-4]}'+'.png',
                           palette=[[0, 0, 0], [255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 0, 255]],
                           )
# matplotlib.use('TkAgg')
# result = inference_segmentor(model, l)
# CLASSES = ["background", "EX", "MA", "SE", "HE"]
# PALETTE = [[0, 0, 0], [255, 0, 0], [255, 255, 0], [255, 255, 255], [0, 255, 0]]
# i = 0
# for res in result:
#     print(count_elements(res))
#     mask = numpy.zeros((1280, 1280, 3), dtype=numpy.int8)
#     show_result_pyplot_new(model, mask, res, opacity=1, show=False, out_file=f'./peruade_mask/test_ddr/{name_list[i][:-4]}' + '.png',
#                            palette=[[0, 0, 0], [255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 0, 255]],
#                            )
#     i += 1

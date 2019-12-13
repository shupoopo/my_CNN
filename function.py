import tensorflow as tf
import numpy as np
import sys


# 零填充
def padding(image, zero_num):
    if len(image.shape) == 4:
        image_padding = np.zeros((image.shape[0], image.shape[1]+2*zero_num, image.shape[2]+2*zero_num, image.shape[3]))
        image_padding[:, zero_num:image.shape[1]+zero_num, zero_num:image.shape[2]+zero_num, :] = image

    elif len(image.shape) == 3:
        image_padding = np.zeros((image.shape[0]+2*zero_num, image.shape[1]+2*zero_num, image.shape[2]))
        image_padding[zero_num:image.shape[0]+zero_num, zero_num:image.shape[1]+zero_num, image.shape[2]] = image

    else:
        print("输入图片维度错误")
        sys.exit()

    return image_padding

def conv(img, conv_filter):

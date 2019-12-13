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

# # 完全根据公式，容易理解的版本
# def conv_calcul(img, conv_filter):
#     # 对二维图像以及二维卷积核进行卷积，不填充
#     filter_size = conv_filter.shape[0]
#     result = np.zeros((img.shape[0]-conv_filter.shape[0]+1, img.shape[1]-conv_filter.shape[1]+1))
#
#     for i in np.arange(0, img.shape[0]-conv_filter.shape[0]+1):
#         for j in np.arange(0, img.shape[1]-conv_filter.shape[1]+1):
#             curr_region = img[i : i+conv_filter.shape[0], j:j+conv_filter.shape[1]]
#             conv_result = curr_region * conv_filter
#             conv_sum = np.sum(conv_result)
#             result[i, j] = conv_sum
#
#     return result
#
#
# def conv(img, conv_filter):
#
#     if len(img.shape) != 3 or len(conv_filter.shape) != 4:
#         print("卷积运算所输入的维度不符合要求")
#         sys.exit()
#     if img.shape[-1] != conv_filter[-1]:
#         print("卷积输入图片与卷积核通道数不一致")
#         sys.exit()
#
#     # 初始化输出的特征图片，由于没有使用零填充，图片尺寸会减小
#     img_out = np.zeros((img.shape[0]-conv_filter.shape[1]+1, img.shape[1]-conv_filter.shape[2]+1, conv_filter.shape[0]))
#
#     for filter_num in range(conv_filter.shape[0]):
#         curr_filter = conv_filter[filter_num, :]
#         conv_map = conv_calcul(img[:, :, 0], curr_filter[:, :, 0])
#
#         for ch_num in range(1, curr_filter.shape[-1]):
#             conv_map = conv_map + conv_calcul(img[:, :, ch_num], curr_filter[:, :, ch_num])
#         img_out[:, :, filter_num] = conv_map
#         return img_out




# 优化的版本
def conv(img, conv_filter):
    if len(img.shape)!=3 or len(conv_filter.shape)!=4:
        print("卷积运算所输入的维度不符合要求")
        sys.exit()

    if img.shape[-1]!=conv_filter.shape[-1]:
        print("卷积输入图片与卷积核的通道数不一致")
        sys.exit()

    img_h, img_w, img_ch = img.shape
    filter_num, filter_h, filter_w, img_ch = conv_filter.shape
    feature_h = img_h - filter_h + 1
    feature_w = img_w - filter_w + 1

    # 初始化输出的特征图片
    img_out = np.zeros((feature_h, feature_w, filter_num))
    img_matrix = np.zeros((feature_h*feature_w, filter_h*filter_w*img_ch))
    filter_matrix = np.zeros((filter_h*filter_w*img_ch, filter_num))

    # 将输入图片张量转化成矩阵形式
    for i in range(feature_h*feature_w):
        for j in range(img_ch):
            img_matrix[i, j*filter_h*filter_w:(j+1)*filter_h*filter_w] = \
                img[np.int(i/feature_w):np.int(i/feature_w+filter_h),np.int(i%feature_w):np.int(i%feature_w+filter_w), j].reshape(filter_h*filter_w)

    for i in range(filter_num):
        filter_matrix[:,i] = conv_filter[i,:].reshape(filter_w*filter_h*img_ch)

    feature_matrix = np.dot(img_matrix, filter_matrix)

    for i in range(filter_num):
        img_out[:, :, i] = feature_matrix[:,i].reshape(feature_h, feature_w)

    return img_out

def pool(feature, size=2, stride=2):
    pool_out = np.zeros([np.int((feature.shape[0]-size)/stride+1),
                         np.int((feature.shape[1]-size)/stride+1),
                                feature.shape[2]])
    pool_out_max_location = np.zeros(pool_out.shape)
    for ch_num in range(feature.shape[-1]):
        r_out = 0
        for r in np.arange(0, feature.shape[0]-size+1, stride):
            c_out = 0
            for c in np.arange(0, feature.shape[1]-size+1, stride):
                pool_out[r_out, c_out, ch_num] = np.max(feature[r:r+size,c:c+size, ch_num])
                pool_out_max_location[r_out, c_out, ch_num] = np.argmax(feature[r:r+size,c:c+size, ch_num])
                c_out += 1
            r_out +=1

    return pool_out, pool_out_max_location

def rot180(conv_filters):
    rot180_filter = np.zeros((conv_filters.shape))
    for filter_num in range(conv_filters.shape[0]):
        for img_ch in range(conv_filters.shape[-1]):
            rot180_filter[filter_num, :, :, img_ch] = np.flipud(np.fliplr(conv_filters[filter_num, :, :, img_ch]))

    return rot180_filter

def pool_delta_error_bp(pool_out_delta, pool_out_max_location, size=2, stride=2):
    # 池化层误差反向传播
    delta = np.zeros(np.int((pool_out_delta.shape[0]-1)*stride+size),
                     np.int((pool_out_delta.shape[1]-1)*stride+size),
                     pool_out_delta.shape[2])
    for ch_num in range(pool_out_delta.shape[-1]):
        for r in range(pool_out_delta.shape[0]):
            for c in range(pool_out_delta.shape[1]):
                order = pool_out_max_location[r,c,ch_num]
                m = np.int(stride*r + order//size)
                n = np.int(stride*c + order%size)
                delta[m, n, ch_num] = pool_out_delta[r, c, ch_num]

    return delta

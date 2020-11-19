import tensorflow as tf
import cv2
import numpy as np
from keras.datasets import cifar10
from keras.datasets import cifar100

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
#(X_train, y_train), (X_test, y_test) = cifar100.load_data()
start = 0
end = 20


def flip_ud():
    prefix = "../Data/flip_ud/"
    for i in range(start, end):
        img = cv2.flip(X_train[i], 0)
        cv2.imwrite(prefix + str(i) + ".jpg", img)


def flip_lr():
    prefix = "../Data/flip_lr/"
    for i in range(start, end):
        img = cv2.flip(X_train[i], 1)
        cv2.imwrite(prefix + str(i) + ".jpg", img)


def rotate_r():
    prefix = "../Data/rotate_r/"
    for i in range(start, end):
        img0 = X_train[i]
        imgInfo = img0.shape
        height = imgInfo[0]
        width = imgInfo[1]
        matRo = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), 30, 0.7)  # mat rotate 1 center 2 angle 3 缩放系数
        img = cv2.warpAffine(img0, matRo, (height, width))
        cv2.imwrite(prefix + str(i) + ".jpg", img)


# def rotate_l():
#     prefix = "../Data/rotate_l/"
#     for i in range(start, end):
#         img0 = X_train[i]
#         imgInfo = img0.shape
#         height = imgInfo[0]
#         width = imgInfo[1]
#         matRo = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), -30, 0.7)  # mat rotate 1 center 2 angle 3 缩放系数
#         img = cv2.warpAffine(img0, matRo, (height, width))
#         cv2.imwrite(prefix + str(i) + ".jpg", img)


def bright():
    gamma = 2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    prefix = "../Data/bright/"
    for i in range(start, end):
        img = cv2.LUT(X_train[i], table)
        cv2.imwrite(prefix + str(i) + ".jpg", img)


def gaussian():
    blur = 0
    prefix = "../Data/gaussian/"
    for i in range(start, end):
        img = cv2.GaussianBlur(X_train[i], (5, 5), blur)
        cv2.imwrite(prefix + str(i) + ".jpg", img)


def crop():
    prefix = "../Data/crop/"
    for i in range(start, end):
        img_padding = cv2.copyMakeBorder(X_train[i], 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
        h = np.random.randint(0, 8)
        w = np.random.randint(0, 8)
        img = img_padding[h:h + 32, w:w + 32, :]
        cv2.imwrite(prefix + str(i) + ".jpg", img)


def mixUp():
    prefix = "../Data/mixUp/"
    for i in range(start, end - 1):
        img = 0.9 * X_train[i] + 0.1 * X_train[i + 1]
        cv2.imwrite(prefix + str(i) + ".jpg", img)
    # 为了减少运行时间，不采用统一的取模方式，而是把最后一个单独处理
    img = 0.9 * X_train[end - 1] + 0.1 * X_train[1]
    cv2.imwrite(prefix + str(end - 1) + ".jpg", img)


if __name__ == '__main__':
    flip_ud()
    # flip_lr()
    # rotate_r()
    # bright()
    # gaussian()
    # crop()
    # mixUp()

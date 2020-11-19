import tensorflow as tf
import cv2
import numpy as np
import h5py
from keras.datasets import cifar10
from keras.datasets import cifar100

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# (X_train, y_train), (X_test, y_test) = cifar100.load_data()
start = 0
end = 10000


# def flip_ud():
#     filename = "../Data/cifar-10/flip_ud.h5"
#     # filename = "../Data/cifar-100/flip_ud.h5"
#     imgs = []
#     for i in range(start, end):
#         img = cv2.flip(X_train[i], 0)
#         imgs.append(img)
#     imgs = np.array(imgs)
#     f = h5py.File(filename, "w")
#     f.create_dataset("X_train", data=imgs, dtype=np.uint8)
#     f.close()


def flip_lr():
    filename = "../Data/cifar-10/flip_lr.h5"
    # filename = "../Data/cifar-100/flip_lr.h5"
    imgs = []
    for i in range(start, end):
        img = cv2.flip(X_train[i], 1)
        imgs.append(img)
    imgs = np.array(imgs)
    f = h5py.File(filename, "w")
    f.create_dataset("X_train", data=imgs, dtype=np.uint8)
    f.close()


def rotate_r():
    filename = "../Data/cifar-10/rotate_r.h5"
    # filename = "../Data/cifar-100/rotate_r.h5"
    imgs = []
    for i in range(start, end):
        img0 = X_train[i]
        imgInfo = img0.shape
        height = imgInfo[0]
        width = imgInfo[1]
        matRo = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), 30, 0.7)  # mat rotate 1 center 2 angle 3 缩放系数
        img = cv2.warpAffine(img0, matRo, (height, width))
        imgs.append(img)
    imgs = np.array(imgs)
    f = h5py.File(filename, "w")
    f.create_dataset("X_train", data=imgs, dtype=np.uint8)
    f.close()


# def rotate_l():
#     filename = "../Data/cifar-10/rotate_l.h5"
#     # filename = "../Data/cifar-100/rotate_l.h5"
#     imgs = []
#     for i in range(start, end):
#         img0 = X_train[i]
#         imgInfo = img0.shape
#         height = imgInfo[0]
#         width = imgInfo[1]
#         matRo = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), -30, 0.7)  # mat rotate 1 center 2 angle 3 缩放系数
#         img = cv2.warpAffine(img0, matRo, (height, width))
#         imgs.append(img)
#     imgs = np.array(imgs)
#     f = h5py.File(filename, "w")
#     f.create_dataset("X_train", data=imgs, dtype=np.uint8)
#     f.close()


def bright():
    gamma = 2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    filename = "../Data/cifar-10/bright.h5"
    # filename = "../Data/cifar-100/bright.h5"
    imgs = []
    for i in range(start, end):
        img = cv2.LUT(X_train[i], table)
        imgs.append(img)
    imgs = np.array(imgs)
    f = h5py.File(filename, "w")
    f.create_dataset("X_train", data=imgs, dtype=np.uint8)
    f.close()


def gaussian():
    blur = 0
    filename = "../Data/cifar-10/gaussian.h5"
    # filename = "../Data/cifar-100/gaussian.h5"
    imgs = []
    for i in range(start, end):
        img = cv2.GaussianBlur(X_train[i], (5, 5), blur)
        imgs.append(img)
    imgs = np.array(imgs)
    f = h5py.File(filename, "w")
    f.create_dataset("X_train", data=imgs, dtype=np.uint8)
    f.close()


def crop():
    filename = "../Data/cifar-10/crop.h5"
    # filename = "../Data/cifar-100/crop.h5"
    imgs = []
    for i in range(start, end):
        img_padding = cv2.copyMakeBorder(X_train[i], 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
        h = np.random.randint(0, 8)
        w = np.random.randint(0, 8)
        img = img_padding[h:h + 32, w:w + 32, :]
        imgs.append(img)
    imgs = np.array(imgs)
    f = h5py.File(filename, "w")
    f.create_dataset("X_train", data=imgs, dtype=np.uint8)
    f.close()


def mixUp():
    filename = "../Data/cifar-10/mixUp.h5"
    # filename = "../Data/cifar-100/mixUp.h5"
    imgs = []
    for i in range(start, end - 1):
        img = 0.9 * X_train[i] + 0.1 * X_train[i + 1]
        imgs.append(img)
    # 为了减少运行时间，不采用统一的取模方式，而是把最后一个单独处理
    img = 0.9 * X_train[end - 1] + 0.1 * X_train[1]
    imgs.append(img)
    imgs = np.array(imgs)
    f = h5py.File(filename, "w")
    f.create_dataset("X_train", data=imgs, dtype=np.uint8)
    f.close()


if __name__ == '__main__':
    flip_lr()
    rotate_r()
    bright()
    gaussian()
    crop()
    mixUp()

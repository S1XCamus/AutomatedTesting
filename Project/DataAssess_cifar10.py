import tensorflow as tf
import os
import cv2
import numpy as np
import h5py
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
prefix = "./model/cifar10"
filenames = os.listdir(prefix)

start = 0
end = 1000


def assess_original():
    for filename in filenames:
        model = tf.keras.models.load_model(prefix + "/" + filename)
        imgs = X_train[start:end]
        # print(type(imgs))
        imgs = imgs / 255
        y_pred = tf.argmax(model.predict(imgs), axis=1)
        # with tf.Session() as sess:
        #     print(sess.run(y_pred))
        y = y_train[start:end]
        y = tf.reshape(y, shape=[-1])
        y = tf.cast(y, dtype=tf.int64)
        # with tf.Session() as sess:
        #     print(sess.run(y))
        bias = tf.subtract(y_pred, y)
        allNum = end - start
        hit = allNum - tf.count_nonzero(bias)
        accuracy = hit / allNum
        with tf.Session() as sess:
            res = sess.run(accuracy)
        print(filename, res)

def assess_aug():
    # 把生成的测试数据加载到imgs数组中
    # imgs = []
    # prefix0 = "../Data/flip_ud/"
    # for i in range(start, end):
    #     imgs.append(cv2.imread(prefix0 + str(i) + ".jpg"))
    # imgs = np.array(imgs)
    imgs = h5py.File("../Data/cifar-10/flip_lr.h5", "r")
    # imgs = h5py.File("../Data/cifar-10/rotate_r.h5", "r")
    # imgs = h5py.File("../Data/cifar-10/bright.h5", "r")
    # imgs = h5py.File("../Data/cifar-10/gaussian.h5", "r")
    # imgs = h5py.File("../Data/cifar-10/crop.h5", "r")
    # imgs = h5py.File("../Data/cifar-10/mixUp.h5", "r")
    imgs = np.array(imgs["X_train"][start:end])
    imgs = imgs / 255
    # 进行评估
    for filename in filenames:
        model = tf.keras.models.load_model(prefix + "/" + filename)

        y_pred = tf.argmax(model.predict(imgs), axis=1)
        # with tf.Session() as sess:
        #     print(sess.run(y_pred))
        y = y_train[start:end]
        # y = np_utils.to_categorical(y)
        y = tf.reshape(y, shape=[-1])
        y = tf.cast(y, dtype=tf.int64)
        # with tf.Session() as sess:
        #     print(sess.run(y))
        bias = tf.subtract(y_pred, y)
        allNum = end - start
        hit = allNum - tf.count_nonzero(bias)
        accuracy = hit / allNum
        with tf.Session() as sess:
            res = sess.run(accuracy)
        print(filename, res)


if __name__ == '__main__':
    # assess_original()
    assess_aug()

import tensorflow as tf
import os
import cv2
import numpy as np
from keras.datasets import cifar10
from keras.datasets import cifar100

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
prefix = "./model/cifar10"
# (X_train, y_train), (X_test, y_test) = cifar100.load_data()
# prefix = "model/cifar100"
filenames = os.listdir(prefix)

start = 0
end = 10


def assess_original():
    for filename in filenames:
        print(filename)
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
        # print(res)
        break


def assess_aug():
    # 把生成的测试数据加载到imgs数组中
    imgs = []
    prefix0 = "../Data/flip_ud/"
    for i in range(start, end):
        imgs.append(cv2.imread(prefix0 + str(i) + ".jpg"))
    imgs = np.array(imgs)

    # 进行评估
    for filename in filenames:
        model = tf.keras.models.load_model(prefix + "/" + filename)
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
        print(res)
        break


if __name__ == '__main__':
    assess_aug()

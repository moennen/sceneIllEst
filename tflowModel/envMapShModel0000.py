#!/usr/bin/python
""" EnvMapSh model #0000

---> input size : 192x108x3
---> output : spherical harmonics coefficient up to the 4th order
---> convolutionnal architecture :
---> fully connected output layer

"""
# from __future__ import division, print_function, absolute_import
import argparse
import os
import sys
import time
import tensorflow as tf
import itertools
#from matplotlib import pyplot as plt
from scipy.misc import toimage
from PIL import Image
import numpy as np

sys.path.append('/mnt/p4/favila/moennen/local/lib/python2.7/site-packages')
import cv2 as cv

# sys.path.append(os.path.abspath('/home/moennen/sceneIllEst/sampleEnvMapShDataset'))
sys.path.append(os.path.abspath(
    '/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleEnvMapShDataset'))
from sampleEnvMapShDataset import *

# Parameters
# numSteps = 140000
numSteps = 75000
logStep = 150
logTrSteps = 1
logTsSteps = 1
batchSz = 32
shOrder = 4
doLinearCS = 1
imgSz = [108, 192]
# envMapSz = [64, 128]
envMapSz = [256, 512]
# imgSz = envMapSz
pixMean = [0.5, 0.5, 0.5]
# shCoeffsMean8 = [1.62535, 1.59993, 1.52873, -0.129034, -0.229063, -0.370292, 0.00808474, 0.00664647, 0.00937933, -0.000757816, -0.00102151, -0.00121659, 0.000707051, 0.000757851, 0.00084574, 0.00244401, 0.0023705, -3.71057e-05, -0.00725334, -0.0203365, -0.0511394, 0.000540162, 0.000259493, -3.54734e-05, -0.0141541, -0.0365667, -0.0914037, -0.0337918, -0.0471127, -0.0681103, 0.000125685, 0.000102473, 0.000561678, -0.0243074, -0.0345659, -0.0506261, 0.00374988, 0.0020202, 0.00125083, 0.000341624,
#                 0.000130869, -0.000197295, 0.0064905, 0.0063412, 0.0062002, -0.00141026, -0.00159163, -0.001749, 0.000723703, 0.00061244, 0.000979724, -0.0014188, -0.0013961, -0.00209469, 0.00024993, 0.000391279, 0.000524354, -0.00097943, 0.000288477, 0.0018179, -0.00940844, -0.0132097, -0.0214507, -4.10496e-05, -6.45817e-05, -0.000133848, -0.0212887, -0.0274884, -0.0410339, 0.000122876, -3.21022e-05, -0.000388814, -0.0250338, -0.032921, -0.0499909, -0.0142551, -0.016832, -0.020492, -0.000367205, -0.000425947, -0.000473871]
# shCoeffsStd8 = [0.429219, 0.429304, 0.501215, 0.322748, 0.292984, 0.33602, 0.144528, 0.144469, 0.156821, 0.131678, 0.129134, 0.138005, 0.132425, 0.117658, 0.114917, 0.137179, 0.125416, 0.127194, 0.139252, 0.134759, 0.142759, 0.0912928, 0.0881598, 0.0907393, 0.190757, 0.183104, 0.196036, 0.119776, 0.116046, 0.135213, 0.0661479, 0.0619696, 0.067106, 0.10324, 0.0980564, 0.11241, 0.075825, 0.0716308, 0.0735666, 0.0599346, 0.0581499,
#                0.0613524, 0.0828133, 0.0766802, 0.0771773, 0.0881641, 0.0802417, 0.0787247, 0.0601254, 0.0566595, 0.0619334, 0.0568282, 0.0544685, 0.0605963, 0.0476382, 0.0457157, 0.0499598, 0.0587511, 0.0565128, 0.0624207, 0.0658961, 0.0646426, 0.0695578, 0.0473787, 0.0467925, 0.0500343, 0.076179, 0.074809, 0.0810718, 0.0496845, 0.0469973, 0.0505273, 0.0915342, 0.0895412, 0.0977202, 0.0627302, 0.0623561, 0.0720656, 0.0399477, 0.038154, 0.0421067]
shCoeffsMean8 = [0.902903, 0.875094, 0.896489, -0.162082, -0.252774, -0.406744, 0.00337894, 0.00172557, 0.00311243, 0.000909537, 0.000695117, -4.90672e-05, -0.00128019, -0.00102835, -0.000301714, 0.002949, 0.00374928, 0.00226501, -0.00952863, -0.0208408, -0.0508576, -0.000182751, -0.000320935, -0.000347464, -0.0185872, -0.0366991, -0.0886874, -0.0410145, -0.0560474, -0.0794013, -0.000163254, -4.07888e-05, -2.79235e-05, -0.0313884, -0.0427534, -0.0605482, 0.00334514, 0.00201432, 0.000809579, -7.04794e-05, -0.000115991, -
                 0.000284364, 0.00531681, 0.00525261, 0.00461419, -0.00017454, -0.000219352, -0.000313192, -0.000728616, -0.000451431, 5.66945e-05, -0.000715351, -0.000834947, -0.00157328, -0.000165865, -2.45795e-05, 0.000226656, -0.000148126, 0.000878715, 0.00207265, -0.00715132, -0.0107122, -0.0205379, -6.46675e-05, -8.40424e-05, -0.000117739, -0.0194614, -0.0248351, -0.0394562, 0.000269648, 0.000203064, 1.16736e-05, -0.0216183, -0.0284543, -0.0478178, -0.0137995, -0.0180241, -0.0228328, -6.30815e-06, -5.03176e-05, -0.000133641]

shCoeffsStd8 = [0.396919, 0.3877, 0.446635, 0.329931, 0.31224, 0.353571, 0.142876, 0.142675, 0.152221, 0.131147, 0.129519, 0.135236, 0.135648, 0.129224, 0.124978, 0.140774, 0.135626, 0.13652, 0.137738, 0.133197, 0.138377, 0.0921217, 0.0883874, 0.0877252, 0.186394, 0.181423, 0.192694, 0.12275, 0.121771, 0.141144, 0.0720004, 0.0689185, 0.0716978, 0.106684, 0.103984, 0.117957, 0.0773083, 0.0739887, 0.0748107, 0.0610265, 0.0578222,
                0.0592729, 0.0835063, 0.0797521, 0.079296, 0.0892349, 0.08521, 0.0832717, 0.0643495, 0.060952, 0.0640418, 0.0604189, 0.058079, 0.0623505, 0.0511543, 0.04894, 0.0520293, 0.0628388, 0.0605339, 0.0650797, 0.0653313, 0.063549, 0.0674159, 0.0475521, 0.0454968, 0.0472838, 0.0756618, 0.073626, 0.0784726, 0.051918, 0.0494255, 0.0514637, 0.0911271, 0.0892432, 0.0961533, 0.0650854, 0.0648895, 0.0739945, 0.0417161, 0.0398853, 0.0426519]


tf.logging.set_verbosity(tf.logging.INFO)


def showImgs(batch, img_depths):

    n_imgs = len(img_depths)

    if np.sum(img_depths) != batch.shape[3]:
        raise ValueError()

    batch_im = np.zeros(
        (batch.shape[0] * batch.shape[1], n_imgs * batch.shape[2], 3))

    for b in range(batch.shape[0]):

        n_offset = 0
        for n in range(n_imgs):

            im_d = img_depths[n]
            im = batch[b, :, :, n_offset:n_offset + im_d]

            if im_d > 3:
                gray = np.mean(im, axis=2)
                im = np.stack([gray, gray, gray], 2)

            batch_im[b * batch.shape[1]:(b + 1) * batch.shape[1],
                     n * batch.shape[2]:(n + 1) * batch.shape[2], 0:im_d] = im

            n_offset += im_d

    plt.imshow(batch_im)
    plt.show()


def writeExrRGB(img, output_filename):
    print img.shape
    rpix = img[:, :, 0].astype(np.float16).tostring()
    gpix = img[:, :, 1].astype(np.float16).tostring()
    bpix = img[:, :, 2].astype(np.float16).tostring()
    HEADER = OpenEXR.Header(img.shape[1], img.shape[0])
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    HEADER['channels'] = dict([(c, half_chan) for c in "RGB"])
    exr = OpenEXR.OutputFile(output_filename, HEADER)
    exr.writePixels({'R': rpix, 'G': gpix, 'B': bpix})
    exr.close()


def loadImgPIL(img_name, imgSz, linearCS):

    im = Image.open(img_name)
    ratio = float(imgSz[1])/imgSz[0]
    imgRatio = float(im.size[0])/im.size[1]
    cw = (int(im.size[1]*ratio) if imgRatio > ratio else im.size[0])
    ow = (int((im.size[0]-cw)/2) if imgRatio > ratio else 0)
    ch = (int(im.size[0]/ratio) if imgRatio < ratio else im.size[1])
    oh = (int((im.size[1]-ch)/2) if imgRatio < ratio else 0)
    im = im.crop([ow, oh, ow+cw, oh+ch])
    im = im.resize([imgSz[1], imgSz[0]])
    im = np.array(im)
    im = im.astype(np.float32) / 255.0

    if linearCS == 1:
        im = (im <= 0.04045) * (im / 12.92) + (im > 0.04045) * \
            np.power((im + 0.055)/1.055, 2.4)

    return [im]


def printVarTF(sess):
    tvars = tf.trainable_variables()
    for var in tvars:
        print var.name
        print var.eval(sess)


def conv_layer(x, filter_size, step, scope, padding='VALID'):
    # tf.random_normal(filter_size))
    initializer = tf.contrib.layers.xavier_initializer()
    layer_w = tf.Variable(initializer(filter_size))
    layer_b = tf.Variable(initializer([filter_size[3]]))
    layer = tf.nn.conv2d(x, layer_w, strides=[
        1, step, step, 1], padding=padding)
    layer = tf.nn.bias_add(layer, layer_b)
    layer = tf.nn.relu(layer, name=scope)
    return layer


def envMapShModel0000_1(imgs, outputSz, dropout):

    with tf.variable_scope('EnvMapShModel00001'):

        # -----> preprocessing
        with tf.name_scope('preprocess') as scope:
            img_mean = tf.constant(pixMean, dtype=tf.float32, shape=[
                1, 1, 1, 3], name='img_mean')
            layer0 = imgs-img_mean
        # layer0 = imgs

        # ----> 90x48x32
        with tf.name_scope('layer1_1') as scope:
            layer1 = conv_layer(layer0, [7, 7, 3, 32], 2, scope)
        # ----> 41x20x64
        with tf.name_scope('layer2_1') as scope:
            layer2 = conv_layer(layer1, [5, 5, 32, 64], 2, scope)
        with tf.name_scope('layer2_2') as scope:
            layer2 = conv_layer(layer2, [5, 5, 64, 96], 1, scope, 'SAME')
        # with tf.name_scope('layer2_2') as scope:
        #    layer2 = conv_layer(layer2, [5, 5, 64, 96], 1, scope)
        # ----> 18x8x128
        with tf.name_scope('layer3_1') as scope:
            layer3 = conv_layer(layer2, [3, 3, 96, 128], 2, scope)
        with tf.name_scope('layer3_2') as scope:
            layer3 = conv_layer(layer3, [3, 3, 128, 256], 1, scope, 'SAME')
        # ----> 18x8x128
        with tf.name_scope('layer4_1') as scope:
            layer4 = conv_layer(layer3, [3, 3, 256, 256], 1, scope)
        # ----> 7x2x256
        with tf.name_scope('layer5_1') as scope:
            layer5 = conv_layer(layer4, [3, 3, 256, 512], 2, scope)
        # ----> 1x1x512
        with tf.name_scope('layer6_1') as scope:
            layer6 = conv_layer(layer5, [3, 3, 512, 1024], 2, scope)

        #
        layer6f = tf.contrib.layers.flatten(layer6)
        initializer = tf.contrib.layers.xavier_initializer()
        # with tf.name_scope('layer7_0') as scope:
        #    layer7 = tf.layers.dense(layer6f, 4096, activation=tf.nn.relu, kernel_initializer=initializer,
        #                             bias_initializer=initializer, name=scope)
        #    layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_1') as scope:
            layer7 = tf.layers.dense(layer6f, 2048, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_2') as scope:
            layer7 = tf.layers.dense(layer7d, 1024, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer8_1') as scope:
            outputLayer = tf.layers.dense(layer7d, outputSz, kernel_initializer=initializer,
                                          bias_initializer=initializer, name=scope)

        return outputLayer


def envMapShModel0000_0(imgs, outputSz, dropout):

    with tf.variable_scope('EnvMapShModel00001'):

        # -----> preprocessing
        with tf.name_scope('preprocess') as scope:
            img_mean = tf.constant(pixMean, dtype=tf.float32, shape=[
                1, 1, 1, 3], name='img_mean')
            layer0 = imgs-img_mean
        # layer0 = imgs

        # ----> 90x48x32
        with tf.name_scope('layer1_1') as scope:
            layer1 = conv_layer(layer0, [7, 7, 3, 32], 2, scope)
        # ----> 41x20x64
        with tf.name_scope('layer2_1') as scope:
            layer2 = conv_layer(layer1, [5, 5, 32, 64], 2, scope)
        # ----> 18x8x128
        with tf.name_scope('layer3_1') as scope:
            layer3 = conv_layer(layer2, [3, 3, 64, 128], 2, scope)
        # ----> 18x8x128
        with tf.name_scope('layer4_1') as scope:
            layer4 = conv_layer(layer3, [3, 3, 128, 256], 1, scope)
        # ----> 7x2x256
        with tf.name_scope('layer5_1') as scope:
            layer5 = conv_layer(layer4, [3, 3, 256, 512], 2, scope)
        # ----> 1x1x512
        with tf.name_scope('layer6_1') as scope:
            layer6 = conv_layer(layer5, [3, 3, 512, 1024], 2, scope)

        #
        layer6f = tf.contrib.layers.flatten(layer6)
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.name_scope('layer7_1') as scope:
            layer7 = tf.layers.dense(layer6f, 1024, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer8_1') as scope:
            outputLayer = tf.layers.dense(layer7d, outputSz, kernel_initializer=initializer,
                                          bias_initializer=initializer, name=scope)

        return outputLayer

# Base model extract a set of filters from the input image and have dense layer above of it


def envMapShBaseModel(imgs, outputSz, dropout):

    with tf.variable_scope('envMapShBaseModel'):

        # -----> preprocessing
        with tf.name_scope('preprocess') as scope:
            img_mean = tf.constant(pixMean, dtype=tf.float32, shape=[
                1, 1, 1, 3], name='img_mean')
            layer0 = imgs-img_mean
        # layer0 = imgs

        # ----> 90x48x32
        with tf.name_scope('layer1_1') as scope:
            layer1 = conv_layer(layer0, [7, 7, 3, 32], 2, scope)
        # ----> 41x20x64
        with tf.name_scope('layer2_1') as scope:
            layer2 = conv_layer(layer1, [5, 5, 32, 64], 2, scope)
        # ----> 18x8x128
        with tf.name_scope('layer3_1') as scope:
            layer3 = conv_layer(layer2, [3, 3, 64, 128], 2, scope)
        # ----> 18x8x128
        with tf.name_scope('layer4_1') as scope:
            layer4 = conv_layer(layer3, [3, 3, 128, 256], 2, scope)

        #
        layer6f = tf.contrib.layers.flatten(layer4)
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.name_scope('layer7_1') as scope:
            layer7 = tf.layers.dense(layer6f, 1024, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_2') as scope:
            layer7 = tf.layers.dense(layer7d, 512, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_3') as scope:
            layer7 = tf.layers.dense(layer7d, 256, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer8_1') as scope:
            outputLayer = tf.layers.dense(layer7d, outputSz, kernel_initializer=initializer,
                                          bias_initializer=initializer, name=scope)

        return outputLayer


def base_model(imgs, output_channels, dropout):

    # -----> preprocessing
    with tf.name_scope('preprocess') as scope:
        img_mean = tf.constant(pixMean, dtype=tf.float32, shape=[
            1, 1, 1, 3], name='img_mean')
        layer0 = imgs-img_mean

    n = 32

    xinit = tf.contrib.layers.xavier_initializer()

    with tf.variable_scope('layer_1'):
        layer_1 = tf.layers.conv2d(layer0, n, [7, 7], 2, padding='same',
                                   bias_initializer=xinit, kernel_initializer=xinit, activation=tf.nn.relu)

    with tf.variable_scope('layer_2'):
        layer_2 = tf.layers.conv2d(layer_1, n*2, [5, 5], 2, padding='same',
                                   bias_initializer=xinit, kernel_initializer=xinit, activation=tf.nn.relu)

    with tf.variable_scope('layer_3'):
        layer_3 = tf.layers.conv2d(layer_2, n*4, [3, 3], 2, padding='same',
                                   bias_initializer=xinit, kernel_initializer=xinit, activation=tf.nn.relu)

    layer_4 = tf.layers.flatten(layer_3)
    with tf.variable_scope("layer_4"):
        layer_4 = tf.layers.dense(layer_4, n*8, activation=tf.nn.relu, kernel_initializer=xinit,
                                  bias_initializer=xinit)
        layer_4 = tf.nn.dropout(layer_4, keep_prob=1.0 - dropout)

    with tf.variable_scope("layer_5"):
        layer_5 = tf.layers.dense(layer_4, output_channels, activation=None, kernel_initializer=xinit,
                                  bias_initializer=xinit)

    return layer_5


def envMapShSimplerBaseModel(imgs, outputSz, dropout):

    with tf.variable_scope('envMapShSimplerBaseModel'):

        # -----> preprocessing
        with tf.name_scope('preprocess') as scope:
            img_mean = tf.constant(pixMean, dtype=tf.float32, shape=[
                1, 1, 1, 3], name='img_mean')
            layer0 = imgs-img_mean
        # layer0 = imgs

        # ----> 90x48x32
        with tf.name_scope('layer1_1') as scope:
            layer1 = conv_layer(layer0, [7, 7, 3, 32], 2, scope)
        # ----> 41x20x64
        with tf.name_scope('layer2_1') as scope:
            layer2 = conv_layer(layer1, [5, 5, 32, 64], 2, scope)
        # ----> 18x8x128
        with tf.name_scope('layer3_1') as scope:
            layer3 = conv_layer(layer2, [3, 3, 64, 128], 2, scope)

        #
        layer6f = tf.contrib.layers.flatten(layer3)
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.name_scope('layer7_2') as scope:
            layer7 = tf.layers.dense(layer6f, 512, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout, training=True)
        with tf.name_scope('layer7_3') as scope:
            layer7 = tf.layers.dense(layer7d, 256, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout, training=True)
        with tf.name_scope('layer8_1') as scope:
            outputLayer = tf.layers.dense(layer7d, outputSz, kernel_initializer=initializer,
                                          bias_initializer=initializer, name=scope)

        return outputLayer


def envMapShDeep9C6DModel(imgs, outputSz, dropout):

    with tf.variable_scope('EnvMapShDeep9C6DModel'):

        # -----> preprocessing
        with tf.name_scope('preprocess') as scope:
            img_mean = tf.constant(pixMean, dtype=tf.float32, shape=[
                1, 1, 1, 3], name='img_mean')
            layer0 = imgs-img_mean
        # layer0 = imgs

        # ----> 90x48x32
        with tf.name_scope('layer1_1') as scope:
            layer1 = conv_layer(layer0, [7, 7, 3, 32], 2, scope)
        # ----> 41x20x64
        with tf.name_scope('layer2_1') as scope:
            layer2 = conv_layer(layer1, [5, 5, 32, 64], 2, scope)
        with tf.name_scope('layer2_2') as scope:
            layer2 = conv_layer(layer2, [5, 5, 64, 96], 1, scope, 'SAME')
        with tf.name_scope('layer2_2') as scope:
            layer2 = conv_layer(layer2, [5, 5, 96, 96], 1, scope)
        # ----> 18x8x128
        with tf.name_scope('layer3_1') as scope:
            layer3 = conv_layer(layer2, [3, 3, 96, 128], 2, scope)
        with tf.name_scope('layer3_2') as scope:
            layer3 = conv_layer(layer3, [3, 3, 128, 256], 1, scope, 'SAME')
        # ----> 18x8x128
        with tf.name_scope('layer4_1') as scope:
            layer4 = conv_layer(layer3, [3, 3, 256, 256], 1, scope)
        # ----> 7x2x256
        with tf.name_scope('layer5_1') as scope:
            layer5 = conv_layer(layer4, [3, 3, 256, 512], 2, scope)
        # ----> 1x1x512
        with tf.name_scope('layer6_1') as scope:
            layer6 = conv_layer(layer5, [3, 3, 512, 1024], 2, scope)

        #
        layer6f = tf.contrib.layers.flatten(layer6)
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.name_scope('layer7_1') as scope:
            layer7 = tf.layers.dense(layer6f, 2048, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_2') as scope:
            layer7 = tf.layers.dense(layer7d, 1024, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_2') as scope:
            layer7 = tf.layers.dense(layer7d, 512, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_2') as scope:
            layer7 = tf.layers.dense(layer7d, 256, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_2') as scope:
            layer7 = tf.layers.dense(layer7d, 128, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer8_1') as scope:
            outputLayer = tf.layers.dense(layer7d, outputSz, kernel_initializer=initializer,
                                          bias_initializer=initializer, name=scope)

        return outputLayer


def envMapShDeep9C7DModel(imgs, outputSz, dropout):

    with tf.variable_scope('EnvMapShDeep9C7DModel'):

        # -----> preprocessing
        with tf.name_scope('preprocess') as scope:
            img_mean = tf.constant(pixMean, dtype=tf.float32, shape=[
                1, 1, 1, 3], name='img_mean')
            layer0 = imgs-img_mean
        # layer0 = imgs

        # ----> 90x48x32
        with tf.name_scope('layer1_1') as scope:
            layer1 = conv_layer(layer0, [7, 7, 3, 32], 2, scope)
        # ----> 41x20x64
        with tf.name_scope('layer2_1') as scope:
            layer2 = conv_layer(layer1, [5, 5, 32, 64], 2, scope)
        with tf.name_scope('layer2_2') as scope:
            layer2 = conv_layer(layer2, [5, 5, 64, 96], 1, scope, 'SAME')
        with tf.name_scope('layer2_2') as scope:
            layer2 = conv_layer(layer2, [5, 5, 96, 96], 1, scope)
        # ----> 18x8x128
        with tf.name_scope('layer3_1') as scope:
            layer3 = conv_layer(layer2, [3, 3, 96, 128], 2, scope)
        with tf.name_scope('layer3_2') as scope:
            layer3 = conv_layer(layer3, [3, 3, 128, 256], 1, scope, 'SAME')
        # ----> 18x8x128
        with tf.name_scope('layer4_1') as scope:
            layer4 = conv_layer(layer3, [3, 3, 256, 256], 1, scope)
        # ----> 7x2x256
        with tf.name_scope('layer5_1') as scope:
            layer5 = conv_layer(layer4, [3, 3, 256, 512], 2, scope)
        # ----> 1x1x512
        with tf.name_scope('layer6_1') as scope:
            layer6 = conv_layer(layer5, [3, 3, 512, 1024], 2, scope)

        #
        layer6f = tf.contrib.layers.flatten(layer6)
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.name_scope('layer7_1') as scope:
            layer7 = tf.layers.dense(layer6f, 2048, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_3') as scope:
            layer7 = tf.layers.dense(layer7d, 1024, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_4') as scope:
            layer7 = tf.layers.dense(layer7d, 768, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_5') as scope:
            layer7 = tf.layers.dense(layer7d, 512, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_7') as scope:
            layer7 = tf.layers.dense(layer7d, 256, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_8') as scope:
            layer7 = tf.layers.dense(layer7d, 128, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer8_1') as scope:
            outputLayer = tf.layers.dense(layer7d, outputSz, kernel_initializer=initializer,
                                          bias_initializer=initializer, name=scope)

        return outputLayer


def envMapShDeep9C9DModel(imgs, outputSz, dropout):

    with tf.variable_scope('EnvMapShDeep9C9DModel'):

        # -----> preprocessing
        with tf.name_scope('preprocess') as scope:
            img_mean = tf.constant(pixMean, dtype=tf.float32, shape=[
                1, 1, 1, 3], name='img_mean')
            layer0 = imgs-img_mean
        # layer0 = imgs

        # ----> 90x48x32
        with tf.name_scope('layer1_1') as scope:
            layer1 = conv_layer(layer0, [7, 7, 3, 32], 2, scope)
        # ----> 41x20x64
        with tf.name_scope('layer2_1') as scope:
            layer2 = conv_layer(layer1, [5, 5, 32, 64], 2, scope)
        with tf.name_scope('layer2_2') as scope:
            layer2 = conv_layer(layer2, [5, 5, 64, 96], 1, scope, 'SAME')
        with tf.name_scope('layer2_2') as scope:
            layer2 = conv_layer(layer2, [5, 5, 96, 96], 1, scope)
        # ----> 18x8x128
        with tf.name_scope('layer3_1') as scope:
            layer3 = conv_layer(layer2, [3, 3, 96, 128], 2, scope)
        with tf.name_scope('layer3_2') as scope:
            layer3 = conv_layer(layer3, [3, 3, 128, 256], 1, scope, 'SAME')
        # ----> 18x8x128
        with tf.name_scope('layer4_1') as scope:
            layer4 = conv_layer(layer3, [3, 3, 256, 256], 1, scope)
        # ----> 7x2x256
        with tf.name_scope('layer5_1') as scope:
            layer5 = conv_layer(layer4, [3, 3, 256, 512], 2, scope)
        # ----> 1x1x512
        with tf.name_scope('layer6_1') as scope:
            layer6 = conv_layer(layer5, [3, 3, 512, 1024], 2, scope)

        #
        layer6f = tf.contrib.layers.flatten(layer6)
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.name_scope('layer7_1') as scope:
            layer7 = tf.layers.dense(layer6f, 2048, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_2') as scope:
            layer7 = tf.layers.dense(layer7d, 1536, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_3') as scope:
            layer7 = tf.layers.dense(layer7d, 1024, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_4') as scope:
            layer7 = tf.layers.dense(layer7d, 768, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_5') as scope:
            layer7 = tf.layers.dense(layer7d, 512, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_6') as scope:
            layer7 = tf.layers.dense(layer7d, 374, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_7') as scope:
            layer7 = tf.layers.dense(layer7d, 256, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_8') as scope:
            layer7 = tf.layers.dense(layer7d, 128, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer8_1') as scope:
            outputLayer = tf.layers.dense(layer7d, outputSz, kernel_initializer=initializer,
                                          bias_initializer=initializer, name=scope)

        return outputLayer


def envMapShModel(imgs, outputSz, dropout):

    # return envMapShBaseModel(imgs, outputSz, dropout)
    return base_model(imgs, outputSz, dropout)
    # return envMapShSimplerBaseModel(imgs, outputSz, dropout)


class EnvMapShDatasetTF(object):

    def __init__(self, dbPath, imgRootDir, batchSz, seed, linearCS):
        self.__envMapDb = EnvMapShDataset(
            dbPath, imgRootDir, shOrder, seed, linearCS)
        self.__dims = [batchSz, imgSz[0], imgSz[1]]
        self.data = tf.data.Dataset.from_generator(
            self.genEnvMapSh, (tf.float32, tf.float32))

    def genEnvMapSh(self):
        for i in itertools.count(1):
            imgs, coeffs, cparams = self.__envMapDb.sampleData(self.__dims)

            yield (coeffs, imgs)

    def getNbShCoeffs(self):
        return self.__envMapDb.nbShCoeffs*3


class EnvMapFoVDatasetTF(object):

    def __init__(self, dbPath, seed):
        self.__envMapDb = EnvMapShDataset(dbPath, shOrder, seed)
        self.__dims = [batchSz, imgSz[0], imgSz[1]]
        self.data = Dataset.from_generator(
            self.genEnvMapSh, (tf.float32, tf.float32))

    def genEnvMapSh(self):
        for i in itertools.count(1):
            imgs, coeffs, cparams = self.__envMapDb.sampleData(self.__dims)

            yield (cparams[0], imgs)

    def getNbShCoeffs(self):
        return self.__envMapDb.nbShCoeffs*3


def evalEnvMapShModel(modelPath, imgLst, outputDir, envMapSz, linearCS):

    modelFilename = modelPath + "/tfData"

    # input
    with open(imgLst, 'r') as img_names_file:

        nbShCoeffs = EnvMapShDataset.nbShCoeffs(shOrder)*3
        inputShape = [1, imgSz[0], imgSz[1], 3]
        outputShape = [1, nbShCoeffs]

        inputView = tf.placeholder(
            tf.float32, shape=inputShape, name="input_view")
        dropoutProb = tf.placeholder(tf.float32)  # dropout (keep probability)

        computedSh = envMapShModel(inputView, nbShCoeffs, dropoutProb)
        # accuracy = tf.reduce_mean(tf.square(tf.subtract(computedSh, outputSh)))

        # Persistency
        persistency = tf.train.Saver(pad_step_number=True, keep_checkpoint_every_n_hours=3,
                                     filename=modelFilename)

        # Params Initializer
        varInit = tf.global_variables_initializer()

        with tf.Session() as sess:

            # initialize params
            sess.run(varInit)

            # Restore model if needed
            persistency.restore(sess, tf.train.latest_checkpoint(modelPath))

            for img_name in img_names_file:

                img = loadImgPIL(img_name.rstrip('\n'), imgSz, linearCS)
                # img = EnvMapShDataset.loadImg(img_name,  imgSz, linearCS)
                toimage(img[0]).show()

                output = sess.run(computedSh, feed_dict={dropoutProb: 0.0,
                                                         inputView: img})

                envMap = EnvMapShDataset. generateEnvMap(
                    shOrder, output[0], envMapSz)

                toimage(envMap[0]).show()

                # writeExrRGB(envMap[0], os.path.abspath(
                #    outputDir + "/" + os.path.splitext(os.path.basename(img_name))[0] + ".exr"))

                raw_input(".")


def testEnvMapShModel(modelPath, imgRootDir, testPath, nbTests, linearCS):

    modelFilename = modelPath + "/tfData"

    rseed = 639878  # int(time.time())
    tf.set_random_seed(rseed)

    tsDs = EnvMapShDatasetTF(testPath, imgRootDir, 1, rseed, linearCS)

    nbShCoeffs = tsDs.getNbShCoeffs()
    inputShape = [1, imgSz[0], imgSz[1], 3]
    outputShape = [1, nbShCoeffs]

    dsIt = tf.data.Iterator.from_structure(
        tsDs.data.output_types, tsDs.data.output_shapes)
    dsView = dsIt.get_next()
    tsInit = dsIt.make_initializer(tsDs.data)

    # Input
    inputView = tf.placeholder(tf.float32, shape=inputShape, name="input_view")
    dropoutProb = tf.placeholder(tf.float32)  # dropout (keep probability)

    # Graph
    computedSh = envMapShModel(inputView, nbShCoeffs, dropoutProb)

    # Persistency
    persistency = tf.train.Saver(pad_step_number=True, keep_checkpoint_every_n_hours=3,
                                 filename=modelFilename)

    # Params Initializer
    varInit = tf.global_variables_initializer()

    with tf.Session() as sess:
        # initialize params
        sess.run(varInit)

        # initialize iterator
        sess.run(tsInit)

        # Restore model if needed
        persistency.restore(sess, tf.train.latest_checkpoint(modelPath))

        for step in range(0, nbTests):
            coeffs, imgs = sess.run(dsView)
            estCoeffs = sess.run(computedSh, feed_dict={dropoutProb: 0.0,
                                                        inputView: imgs})

            # show the sample
            toimage(imgs[0]).show()

            # show the ground truth map
            envMapGT = EnvMapShDataset. generateEnvMap(
                shOrder, coeffs[0], envMapSz)
            toimage(envMapGT[0]).show()

            raw_input(".")

            # show the estimated map
            envMap = EnvMapShDataset. generateEnvMap(
                shOrder, estCoeffs[0], envMapSz)
            toimage(envMap[0]).show()

            raw_input(".")


def trainEnvMapShModel(modelPath, imgRootDir, trainPath, testPath, linearCS):

    tbLogsPath = modelPath + "/tbLogs"
    modelFilename = modelPath + "/tfData"

    #rseed = int(time.time())
    rseed = 20160704

    tf.set_random_seed(rseed)

    trDs = EnvMapShDatasetTF(trainPath, imgRootDir, batchSz, rseed, linearCS)
    tsDs = EnvMapShDatasetTF(testPath, imgRootDir, batchSz, rseed, linearCS)

    nbShCoeffs = trDs.getNbShCoeffs()
    shCoeffsMean = shCoeffsMean8[0:nbShCoeffs]
    shCoeffsStd = shCoeffsStd8[0:nbShCoeffs]
    inputShape = [batchSz, imgSz[0], imgSz[1], 3]
    outputShape = [batchSz, nbShCoeffs]

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)
    tsInit = dsIt.make_initializer(tsDs.data)

    # Input
    inputView = tf.placeholder(tf.float32, shape=inputShape, name="input_view")
    outputSh = tf.placeholder(tf.float32, shape=outputShape, name="output_sh")
    dropoutProb = tf.placeholder(tf.float32)  # dropout (keep probability)

    # NormoutputSh
    outputShMean = tf.constant(shCoeffsMean, dtype=tf.float32, shape=[
        1, nbShCoeffs], name='shCoeffs_mean')
    outputShStd = tf.constant(shCoeffsStd, dtype=tf.float32, shape=[
        1, nbShCoeffs], name='shCoeffs_mean')
    outputShNorm = (outputSh-outputShMean) / outputShStd
    #outputShNorm = (outputSh-outputShMean)
    # outputShNorm=outputSh
    #outputSh = (outputSh-outputShMean) / outputShStd

    # Test
    # outputSh2 = tf.placeholder(
    #    tf.float32, shape=outputShape, name="output_sh2")
    # outStd = tf.sqrt(tf.reduce_mean(
    #    tf.square(tf.subtract(outputSh2, outputShNorm))))

    # Graph
    computedSh = envMapShModel(inputView, nbShCoeffs, dropoutProb)

    # Optimizer
    costAll = tf.reduce_mean(
        tf.square(tf.subtract(computedSh, outputSh)), 0)
    # tf.square(tf.subtract(computedSh, outputShNorm)), 0)
    accuracyAll = tf.reduce_mean(
        tf.square(tf.subtract(computedSh, outputSh)), 0)
    # tf.square(tf.subtract(computedSh*outputShStd+outputShMean,
    #                      outputSh)), 0)
    # tf.square(tf.subtract(computedSh+outputShMean, outputSh)), 0)
    cost = tf.reduce_mean(costAll)
    accuracy = tf.reduce_mean(accuracyAll)
    globalStep = tf.Variable(0, trainable=False)
    learningRate = tf.train.polynomial_decay(0.001, globalStep, numSteps, 0.0,
                                             power=0.7)
    optEngine = tf.train.AdamOptimizer(learning_rate=learningRate)
    optimizer = optEngine.minimize(cost, global_step=globalStep)

    # Persistency
    persistency = tf.train.Saver(pad_step_number=True, keep_checkpoint_every_n_hours=3,
                                 filename=modelFilename)

    # Initializing the board
    tf.summary.scalar("loss", cost)
    tf.summary.scalar("learning_rate", learningRate)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    grads = optEngine.compute_gradients(cost, tf.trainable_variables())
    for grad, var in grads:
        tf.summary.histogram(var.name + '/gradient', grad)
    merged_summary_op = tf.summary.merge_all()
    tf.summary.scalar("test_loss", cost)
    test_summary_op = tf.summary.merge_all()

    # Metrics : Mean
    meanOutputSh, meanOutputShUpdateOp = tf.contrib.metrics.streaming_mean_tensor(
        outputSh)

    currentMeanOutputSh = tf.placeholder(
        tf.float32, shape=outputShape, name="output_sh")
    outStdAll = tf.reshape(tf.reduce_mean(
        currentMeanOutputSh, 0), [1, nbShCoeffs])
    outStdAll = tf.tile(outStdAll, [batchSz, 1])
    outStdAll = tf.sqrt(tf.reduce_mean(
        tf.square(tf.subtract(outStdAll, outputSh)), 0))
    outStd = tf.reduce_mean(outStdAll)

    # Params Initializer
    varInit = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
        # with tf.Session() as sess:

        summary_writer = tf.summary.FileWriter(tbLogsPath, graph=sess.graph)

        # initialize params
        sess.run(varInit)
        sess.run(tf.local_variables_initializer())

        # Restore model if needed
        try:
            persistency.restore(sess, tf.train.latest_checkpoint(modelPath))
        except:
            print "Cannot load model:", sys.exc_info()[0]

        sess.run(trInit)

        # get each element of the training dataset until the end is reached
        for step in range(globalStep.eval(sess), numSteps):

            # initialize the iterator on the training data

            # Get the next training batch
            coeffs, imgs = sess.run(dsView)

            # Run optimization op (backprop)
            _, currMean = sess.run([optimizer, meanOutputShUpdateOp], feed_dict={dropoutProb: 0.2,
                                                                                 inputView: imgs,
                                                                                 outputSh: coeffs})
            # Log
            if step % logStep == 0:

                # summary
                summary = sess.run(merged_summary_op, feed_dict={dropoutProb: 0.0,
                                                                 inputView: imgs,
                                                                 outputSh: coeffs})
                summary_writer.add_summary(summary, globalStep.eval(sess))

                # Sample train accuracy
                sess.run(trInit)
                trAccuracy = 0
                trStd = 0
                for logTrStep in range(logTrSteps):
                    coeffs, imgs = sess.run(dsView)
                    trAcc = sess.run(accuracy, feed_dict={dropoutProb: 0.0,
                                                          inputView: imgs,
                                                          outputSh:  coeffs})
                    trAccuracy += trAcc
                    # trAccuracyAll += trAccAll
                    # coeffs2, imgs2 = sess.run(dsView)
                    # trStd += sess.run(outStd, feed_dict={outputSh:  coeffs,
                    #                                     outputSh2: coeffs2})

                # print "------------------------------------------"
                # print imgs
                # print "------------------------------------------"
                # print coeffs
                # print "------------------------------------------"
                # print computedSh.eval(feed_dict={dropoutProb: 0.0,
                #                                 inputView: imgs,
                #                                 outputSh:  coeffs})
                # print "------------------------------------------"

            # Sample test accuracy
                sess.run(tsInit)
                tsAccuracy = 0
                stdAccuracy = 0
                for logTsStep in range(logTsSteps):
                    coeffs, imgs = sess.run(dsView)
                    tsAcc, stdAcc = sess.run([accuracy, outStd], feed_dict={dropoutProb: 0.0,
                                                                            inputView: imgs,
                                                                            outputSh:  coeffs,
                                                                            currentMeanOutputSh: currMean})
                    tsAccuracy += tsAcc
                    stdAccuracy += stdAcc

                # summary
                summary = sess.run(test_summary_op, feed_dict={dropoutProb: 0.0,
                                                               inputView: imgs,
                                                               outputSh: coeffs})
                summary_writer.add_summary(summary, globalStep.eval(sess))

                print("{:08d}".format(globalStep.eval(sess)) +
                      " | lr = " + "{:.8f}".format(learningRate.eval()) +
                      " | trAcc  = " + "{:.5f}".format(trAccuracy/logTrSteps) +
                      " | tsAcc  = " + "{:.5f}".format(tsAccuracy/logTsSteps) +
                      " | stdAcc  = " + "{:.5f}".format(stdAccuracy/logTsSteps))
                # print(" trAcc :")
                # print trAccAll

                # step
                persistency.save(sess, modelFilename, global_step=globalStep)

                # reset the training iterator
                sess.run(trInit)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("modelPath", help="path to the trainedModel")
    parser.add_argument(
        "imgRootDir", help="root directory to the images in the levelDb databases")
    parser.add_argument(
        "trainDbPath", help="path to the Training EnvMapDataset levelDb path")
    parser.add_argument(
        "testDbPath", help="path to the Testing EnvMapDataset levelDb path")
    args = parser.parse_args()

    trainEnvMapShModel(args.modelPath, args.imgRootDir, args.trainDbPath,
                       args.testDbPath, doLinearCS)

    # testEnvMapShModel(args.modelPath, args.imgRootDir,
    #                  args.testDbPath, 30, doLinearCS)

    # evalEnvMapShModel(args.modelPath,  args.trainDbPath,
    #                   args.testDbPath, envMapSz, doLinearCS)

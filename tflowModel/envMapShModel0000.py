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
from matplotlib import pyplot as plt
from scipy.misc import toimage
from PIL import Image

sys.path.append('/mnt/p4/favila/moennen/local/lib/python2.7/site-packages')
import cv2 as cv

# sys.path.append(os.path.abspath('/home/moennen/sceneIllEst/sampleEnvMapShDataset'))
sys.path.append(os.path.abspath(
    '/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleEnvMapShDataset'))
from sampleEnvMapShDataset import *

# Parameters
# numSteps = 140000
numSteps = 150000
logStep = 100
logTrSteps = 1
logTsSteps = 1
batchSz = 128
shOrder = 4
imgSz = [108, 192]
envMapSz = [64, 128]
# envMapSz = [256, 512]
# imgSz = envMapSz
pixMean = [0.5, 0.5, 0.5]
shCoeffsMean8 = [1.62535, 1.59993, 1.52873, -0.129034, -0.229063, -0.370292, 0.00808474, 0.00664647, 0.00937933, -0.000757816, -0.00102151, -0.00121659, 0.000707051, 0.000757851, 0.00084574, 0.00244401, 0.0023705, -3.71057e-05, -0.00725334, -0.0203365, -0.0511394, 0.000540162, 0.000259493, -3.54734e-05, -0.0141541, -0.0365667, -0.0914037, -0.0337918, -0.0471127, -0.0681103, 0.000125685, 0.000102473, 0.000561678, -0.0243074, -0.0345659, -0.0506261, 0.00374988, 0.0020202, 0.00125083, 0.000341624,
                 0.000130869, -0.000197295, 0.0064905, 0.0063412, 0.0062002, -0.00141026, -0.00159163, -0.001749, 0.000723703, 0.00061244, 0.000979724, -0.0014188, -0.0013961, -0.00209469, 0.00024993, 0.000391279, 0.000524354, -0.00097943, 0.000288477, 0.0018179, -0.00940844, -0.0132097, -0.0214507, -4.10496e-05, -6.45817e-05, -0.000133848, -0.0212887, -0.0274884, -0.0410339, 0.000122876, -3.21022e-05, -0.000388814, -0.0250338, -0.032921, -0.0499909, -0.0142551, -0.016832, -0.020492, -0.000367205, -0.000425947, -0.000473871]
shCoeffsStd8 = [0.429219, 0.429304, 0.501215, 0.322748, 0.292984, 0.33602, 0.144528, 0.144469, 0.156821, 0.131678, 0.129134, 0.138005, 0.132425, 0.117658, 0.114917, 0.137179, 0.125416, 0.127194, 0.139252, 0.134759, 0.142759, 0.0912928, 0.0881598, 0.0907393, 0.190757, 0.183104, 0.196036, 0.119776, 0.116046, 0.135213, 0.0661479, 0.0619696, 0.067106, 0.10324, 0.0980564, 0.11241, 0.075825, 0.0716308, 0.0735666, 0.0599346, 0.0581499,
                0.0613524, 0.0828133, 0.0766802, 0.0771773, 0.0881641, 0.0802417, 0.0787247, 0.0601254, 0.0566595, 0.0619334, 0.0568282, 0.0544685, 0.0605963, 0.0476382, 0.0457157, 0.0499598, 0.0587511, 0.0565128, 0.0624207, 0.0658961, 0.0646426, 0.0695578, 0.0473787, 0.0467925, 0.0500343, 0.076179, 0.074809, 0.0810718, 0.0496845, 0.0469973, 0.0505273, 0.0915342, 0.0895412, 0.0977202, 0.0627302, 0.0623561, 0.0720656, 0.0399477, 0.038154, 0.0421067]
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


def loadImgPIL(img_name, imgSz):

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
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_3') as scope:
            layer7 = tf.layers.dense(layer7d, 256, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
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


class EnvMapShDatasetTF(object):

    def __init__(self, dbPath, seed):
        self.__envMapDb = EnvMapShDataset(dbPath, shOrder, seed)
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


def evalEnvMapShModel(modelPath, imgLst, outputDir, envMapSz):

    modelFilename = modelPath + "/tfData"

    # input
    with open(imgLst, 'r') as img_names_file:

        nbShCoeffs = EnvMapShDataset.nbShCoeffs(shOrder)*3
        inputShape = [1, imgSz[0], imgSz[1], 3]
        outputShape = [1, nbShCoeffs]

        inputView = tf.placeholder(
            tf.float32, shape=inputShape, name="input_view")
        dropoutProb = tf.placeholder(tf.float32)  # dropout (keep probability)

        computedSh = envMapShModel0000_1(inputView, nbShCoeffs, dropoutProb)
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

                img = loadImgPIL(img_name.rstrip('\n'), imgSz)
                # img = EnvMapShDataset.loadImg(img_name,  imgSz)
                toimage(img[0]).show()

                output = sess.run(computedSh, feed_dict={dropoutProb: 0.0,
                                                         inputView: img})

                envMap = EnvMapShDataset. generateEnvMap(
                    shOrder, output[0], envMapSz)

                toimage(envMap[0]).show()


def trainEnvMapShModel(modelPath, trainPath, testPath):

    tbLogsPath = modelPath + "/tbLogs"
    modelFilename = modelPath + "/tfData"

    seedA = 60309
    seedB = 141195
    seedC = 40716
    rseed = seedA
    rseed = int(time.time())
    print rseed

    tf.set_random_seed(rseed)

    trDs = EnvMapShDatasetTF(trainPath, rseed)
    tsDs = EnvMapShDatasetTF(testPath, rseed)

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
        1, 1, 1, nbShCoeffs], name='shCoeffs_mean')
    outputShStd = tf.constant(shCoeffsStd, dtype=tf.float32, shape=[
        1, 1, 1, nbShCoeffs], name='shCoeffs_mean')
    outputShNorm = (outputSh-outputShMean) / outputShStd
    # outputShNorm = (outputSh-outputShMean)
    # outputShNorm=outputSh

    # Test
    # outputSh2 = tf.placeholder(
    #    tf.float32, shape=outputShape, name="output_sh2")
    # outStd = tf.sqrt(tf.reduce_mean(
    #    tf.square(tf.subtract(outputSh2, outputShNorm))))

    # Graph
    computedSh = envMapShDeep9C6DModel(inputView, nbShCoeffs, dropoutProb)

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

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True, log_device_placement=False)) as sess:
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
        "trainDbPath", help="path to the Training EnvMapDataset levelDb path")
    parser.add_argument(
        "testDbPath", help="path to the Testing EnvMapDataset levelDb path")
    args = parser.parse_args()

    trainEnvMapShModel(args.modelPath, args.trainDbPath, args.testDbPath)

    # evalEnvMapShModel(args.modelPath,  args.trainDbPath,
    #                  args.testDbPath, envMapSz)

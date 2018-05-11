#!/usr/bin/python
""" SuperResolution Model

"""
import argparse
import os
import sys
import time
import tensorflow as tf
import itertools
from matplotlib import pyplot as plt
from scipy.misc import toimage
from PIL import Image
import OpenEXR
import Imath
import numpy as np

sys.path.append('/mnt/p4/favila/moennen/local/lib/python2.7/site-packages')
import cv2 as cv

sys.path.append(os.path.abspath(
    '/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleBuffDataset/'))
from sampleBuffDataset import *

# Parameters
numSteps = 25000
logStep = 150
logTrSteps = 1
logTsSteps = 1
logShowFirstTraining = False

supResFactor = 0.75

baseLearningRate = 0.00005

batchSz = 64
imgSz = [96, 96]
imgSzTst = [384, 384]

# logging
tf.logging.set_verbosity(tf.logging.INFO)


#-----------------------------------------------------------------------------------------------------
# DATASET
#-----------------------------------------------------------------------------------------------------

class SupResDatasetTF(object):

    __lib = BufferDataSamplerLibrary(
        "/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleBuffDataset/libSuperResolutionSampler/libSuperResolutionSampler.so")

    def __init__(self, dbPath, imgRootDir, batchSz, imgSz, scaleFactor, seed):
        params = np.array([batchSz, imgSz[0], imgSz[1],
                           scaleFactor], dtype=np.float32)
        self.__ds = BufferDataSampler(
            SupResDatasetTF.__lib, dbPath, imgRootDir, params, seed)
        self.data = tf.data.Dataset.from_generator(
            self.sample, (tf.float32, tf.float32))

    def sample(self):
        for i in itertools.count(1):
            imgHD, imgLD = self.__ds.getDataBuffers()
            yield (imgHD, imgLD)

#-----------------------------------------------------------------------------------------------------
# UTILS
#-----------------------------------------------------------------------------------------------------


def loadImgPIL(img_name, linearCS):

    im = Image.open(img_name)
    im = np.array(im)
    im = im.astype(np.float32) / 255.0

    if linearCS == 1:
        im = (im <= 0.04045) * (im / 12.92) + (im > 0.04045) * \
            np.power((im + 0.055)/1.055, 2.4)

    return [im]


def loadResizeImgPIL(img_name, imgSz, linearCS):

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

#-----------------------------------------------------------------------------------------------------
# MODEL
#-----------------------------------------------------------------------------------------------------


def conv_layer(x, filter_size, step, scope, padding='VALID'):
    initializer = tf.contrib.layers.xavier_initializer()
    layer_w = tf.Variable(initializer(filter_size))
    layer_b = tf.Variable(initializer([filter_size[3]]))
    if step > 0:
        layer = tf.nn.conv2d(x, layer_w, [1, step, step, 1], padding)
    else:
        layer = tf.nn.atrous_conv2d(x, layer_w, abs(step), padding)
    layer = tf.nn.bias_add(layer, layer_b)
    return layer


def convrelu_layer(x, filter_size, step, scope, padding='VALID'):
    layer = conv_layer(x, filter_size, step, scope, padding)
    layer = tf.nn.relu(layer, name=scope)
    return layer


def ms_convrelu_layer(x, filter_size, scope, padding='SAME'):

    ms_filter_size = [filter_size[0], filter_size[1],
                      filter_size[3], filter_size[3]]

    with tf.name_scope('ms_0') as scope:
        ms_0 = convrelu_layer(x, filter_size, 1, scope, padding)
    with tf.name_scope('ms_2') as scope:
        ms_2 = convrelu_layer(ms_0, ms_filter_size, -2, scope, padding)
    with tf.name_scope('ms_5') as scope:
        ms_5 = convrelu_layer(ms_2, ms_filter_size, -5, scope, padding)

    return tf.concat([ms_0, ms_2, ms_5], 3)


def preprocess(imgs):
    # -----> preprocessing : put the pix values in [-1..1]
    with tf.name_scope('preprocess') as scope:
        img_mean = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32, shape=[
            1, 1, 1, 3], name='img_mean')
        return tf.multiply(tf.subtract(imgs, img_mean), 2.0)


def computeWeights(res):
    # -----> compute the weight to compensate the sparsity bias
    # (|res|*20 +1.5)^4 / 3000
    resw = tf.square(tf.add(tf.multiply(tf.abs(res), 20.0), 1.5))
    resw = tf.minimum(tf.multiply(tf.square(resw), 0.000333333333), 1.0)
    return resw


def getResiduals(imgsHD, imgsLD):
    outputHDRes = tf.subtract(imgsHD, imgsLD)
    outputHDResW = computeWeights(outputHDRes)

    return outputHDRes, outputHDResW


def getReconstructedPost(imgsLD, imgsRes):
    # normalization of the residuals
    img_mean = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32, shape=[
        1, 1, 1, 3], name='img_mean')
    return tf.add(tf.multiply(tf.add(imgsRes, imgsLD), 0.5), img_mean)


def loadVGG16(x, vggParamsFilename, maxLayerId, sess, toTrain=False):

    parameters = []
    layers = []
    currLayer = 0

    with tf.name_scope('vgg') as scope:

        # convert from -1..1 to 0..255
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[
            1, 1, 1, 3], name='img_mean')

        vggLayer = tf.multiply(tf.add(x, 1.0), 127.5) - mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=toTrain, name='biases')
            out = tf.nn.bias_add(conv, biases)
            vggLayer = tf.nn.relu(out, name=scope)
            parameters += [kernel, biases]
            layers += [vggLayer]
            currLayer += 1

        if (currLayer < maxLayerId):

           # conv1_2
            with tf.name_scope('conv1_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(vggLayer, kernel, [
                    1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                     trainable=toTrain, name='biases')
                out = tf.nn.bias_add(conv, biases)
                vggLayer = tf.nn.relu(out, name=scope)
                parameters += [kernel, biases]
                layers += [vggLayer]
                currLayer += 1

        if (currLayer < maxLayerId):

            # pool1
            vggLayer = tf.nn.max_pool(vggLayer,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      name='pool1')

            # conv2_1
            with tf.name_scope('conv2_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(vggLayer, kernel, [
                                    1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                     trainable=toTrain, name='biases')
                out = tf.nn.bias_add(conv, biases)
                vggLayer = tf.nn.relu(out, name=scope)
                parameters += [kernel, biases]
                layers += [vggLayer]
                currLayer += 1

        if (currLayer < maxLayerId):

            # conv2_2
            with tf.name_scope('conv2_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(vggLayer, kernel, [
                                    1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                     trainable=toTrain, name='biases')
                out = tf.nn.bias_add(conv, biases)
                vggLayer = tf.nn.relu(out, name=scope)
                parameters += [kernel, biases]
                layers += [vggLayer]
                currLayer += 1

        if (currLayer < maxLayerId):

            # pool2
            vggLayer = tf.nn.max_pool(vggLayer,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      name='pool2')

            # conv3_1
            with tf.name_scope('conv3_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(vggLayer, kernel, [
                                    1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                     trainable=toTrain, name='biases')
                out = tf.nn.bias_add(conv, biases)
                vggLayer = tf.nn.relu(out, name=scope)
                parameters += [kernel, biases]
                layers += [vggLayer]
                currLayer += 1

        if (currLayer < maxLayerId):

            # conv3_2
            with tf.name_scope('conv3_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(vggLayer, kernel, [
                                    1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                     trainable=toTrain, name='biases')
                out = tf.nn.bias_add(conv, biases)
                vggLayer = tf.nn.relu(out, name=scope)
                parameters += [kernel, biases]
                layers += [vggLayer]
                currLayer += 1

        if (currLayer < maxLayerId):

            # conv3_3
            with tf.name_scope('conv3_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(vggLayer, kernel, [
                                    1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                     trainable=toTrain, name='biases')
                out = tf.nn.bias_add(conv, biases)
                vggLayer = tf.nn.relu(out, name=scope)
                parameters += [kernel, biases]
                layers += [vggLayer]
                currLayer += 1

        if (currLayer < maxLayerId):

            # pool3
            vggLayer = tf.nn.max_pool(vggLayer,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      name='pool3')

            # conv4_1
            with tf.name_scope('conv4_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(vggLayer, kernel, [
                                    1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=toTrain, name='biases')
                out = tf.nn.bias_add(conv, biases)
                vggLayer = tf.nn.relu(out, name=scope)
                parameters += [kernel, biases]
                layers += [vggLayer]
                currLayer += 1

        if (currLayer < maxLayerId):

            # conv4_2
            with tf.name_scope('conv4_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(vggLayer, kernel, [
                                    1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=toTrain, name='biases')
                out = tf.nn.bias_add(conv, biases)
                vggLayer = tf.nn.relu(out, name=scope)
                parameters += [kernel, biases]
                layers += [vggLayer]
                currLayer += 1

        if (currLayer < maxLayerId):

            # conv4_3
            with tf.name_scope('conv4_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(vggLayer, kernel, [
                                    1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=toTrain, name='biases')
                out = tf.nn.bias_add(conv, biases)
                vggLayer = tf.nn.relu(out, name=scope)
                parameters += [kernel, biases]
                layers += [vggLayer]
                currLayer += 1

        if (currLayer < maxLayerId):

            # pool4
            vggLayer = tf.nn.max_pool(vggLayer,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      name='pool4')

            # conv5_1
            with tf.name_scope('conv5_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(vggLayer, kernel, [
                                    1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=toTrain, name='biases')
                out = tf.nn.bias_add(conv, biases)
                vggLayer = tf.nn.relu(out, name=scope)
                parameters += [kernel, biases]
                layers += [vggLayer]
                currLayer += 1

        if (currLayer < maxLayerId):

            # conv5_2
            with tf.name_scope('conv5_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(vggLayer, kernel, [
                                    1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=toTrain, name='biases')
                out = tf.nn.bias_add(conv, biases)
                vggLayer = tf.nn.relu(out, name=scope)
                parameters += [kernel, biases]
                layers += [vggLayer]
                currLayer += 1

        if (currLayer < maxLayerId):

            # conv5_3
            with tf.name_scope('conv5_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(vggLayer, kernel, [
                                    1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=toTrain, name='biases')
                out = tf.nn.bias_add(conv, biases)
                vggLayer = tf.nn.relu(out, name=scope)
                parameters += [kernel, biases]
                layers += [vggLayer]
                currLayer += 1

        # set the parameters
        weights = np.load(vggParamsFilename)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i >= currLayer:
                break
            sess.run(parameters[i].assign(weights[k]))

        return layers


def vggFeatCost(x, y):
    return tf.reduce_mean(tf.square(x-y))


def baseModel(layer):

    with tf.variable_scope('baseModel'):

        # ---->
        with tf.name_scope('layer1') as scope:
            layer = ms_convrelu_layer(layer, [5, 5, 3, 8], scope)

        # ---->
        with tf.name_scope('layer2') as scope:
            layer = ms_convrelu_layer(layer, [5, 5, 24, 12], scope)

        # ---->
        with tf.name_scope('layer3') as scope:
            layer = ms_convrelu_layer(layer, [5, 5, 36, 18], scope)

        # ---->
        with tf.name_scope('layer4') as scope:
            layer = ms_convrelu_layer(layer, [5, 5, 54, 27], scope)

        # ---->
        with tf.name_scope('layer5') as scope:
            layer = ms_convrelu_layer(layer, [5, 5, 81, 46], scope)

        # ---->
        with tf.name_scope('layer6') as scope:
            layer = ms_convrelu_layer(layer, [5, 5, 138, 69], scope)

        # ---->
        with tf.name_scope('layer7') as scope:
            layer = conv_layer(layer, [3, 3, 207, 3], 1, scope, 'SAME')
            layer = tf.tanh(layer)

        return layer


def testModel(layer):

    with tf.variable_scope('testModel'):

        # ---->
        with tf.name_scope('layer1') as scope:
            layer = convrelu_layer(layer, [5, 5, 3, 8], 1,  scope, 'SAME')

        # ---->
        with tf.name_scope('layer2') as scope:
            layer = convrelu_layer(layer, [5, 5, 8, 12], 1, scope, 'SAME')

        # ---->
        with tf.name_scope('layer3') as scope:
            layer = convrelu_layer(layer, [5, 5, 12, 18], 1, scope, 'SAME')

        # ---->
        with tf.name_scope('layer4') as scope:
            layer = convrelu_layer(layer, [5, 5, 18, 27], 1, scope, 'SAME')

        # ---->
        with tf.name_scope('layer5') as scope:
            layer = convrelu_layer(layer, [5, 5, 27, 46], 1, scope, 'SAME')

        # ---->
        with tf.name_scope('layer6') as scope:
            layer = convrelu_layer(layer, [5, 5, 46, 69], 1, scope, 'SAME')

        # ---->
        with tf.name_scope('layer7') as scope:
            layer = conv_layer(layer, [3, 3, 69, 3], 1, scope, 'SAME')
            layer = tf.tanh(layer)

        return layer


def model(imgs):

    return baseModel(imgs)

#-----------------------------------------------------------------------------------------------------
# EVAL
#-----------------------------------------------------------------------------------------------------


def evalModel(modelPath, imgLst, upscaleFactor):

    modelFilename = modelPath + "/tfData"

    inputLDi = tf.placeholder(tf.float32, name="input_ld")
    inputLD = preprocess(inputLDi)

    # Model
    computedHDRes = model(inputLD)

    # Reconstructed
    computedHD = getReconstructedPost(inputLD, computedHDRes)

    # Persistency
    persistency = tf.train.Saver(pad_step_number=True, keep_checkpoint_every_n_hours=3,
                                 filename=modelFilename)

    # Params Initializer
    varInit = tf.global_variables_initializer()

    #
    invSupResFactor = 1.0/supResFactor

    with tf.Session() as sess:

        # initialize params
        sess.run(varInit)

        # Restore model if needed
        persistency.restore(sess, tf.train.latest_checkpoint(modelPath))

        # input
        with open(imgLst, 'r') as img_names_file:

            for img_name in img_names_file:

                imgLD = loadImgPIL(img_name.rstrip('\n'), 1)
                imgLD = [cv.resize(imgLD[0], (0, 0), fx=0.1,
                                   fy=0.1, interpolation=cv.INTER_AREA)]
                imgLDUp = imgLD

                factor = 1.0

                while factor < upscaleFactor:

                    factor *= invSupResFactor

                    # probe
                    imgLDUp = [
                        cv.resize(imgLDUp[0], (0, 0), fx=invSupResFactor, fy=invSupResFactor, interpolation=cv.INTER_CUBIC)]

                    imgLD = [cv.resize(
                        imgLD[0], (0, 0), fx=invSupResFactor, fy=invSupResFactor, interpolation=cv.INTER_LINEAR)]
                    imgLD = sess.run(computedHD, feed_dict={inputLDi: imgLD})

                    # show the sample
                    cv.imshow('LD', cv.cvtColor(imgLDUp[0], cv.COLOR_RGB2BGR))
                    cv.imshow('EST', cv.cvtColor(imgLD[0], cv.COLOR_RGB2BGR))
                    cv.waitKey(0)

#-----------------------------------------------------------------------------------------------------
# UNIT TESTS
#-----------------------------------------------------------------------------------------------------


def testDataset(imgRootDir, trainPath):

    rseed = int(time.time())
    print "SEED : " + str(rseed)

    tf.set_random_seed(rseed)

    trDs = SupResDatasetTF(trainPath, imgRootDir, batchSz,
                           imgSz, supResFactor, rseed)

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)

    with tf.Session() as sess:

        sess.run(trInit)

        for step in range(1):

            imgHD, imgLD = sess.run(dsView)

            toimage(imgHD[0]).show()
            toimage(imgLD[0]).show()

            toimage(imgHD[-1]).show()
            toimage(imgLD[-1]).show()

#-----------------------------------------------------------------------------------------------------
# TESTING
#-----------------------------------------------------------------------------------------------------


def testModel(modelPath, imgRootDir, testPath, nbTests):

    modelFilename = modelPath + "/tfData"

    invSupResFactor = 1.0/supResFactor

    rseed = int(time.time())
    # rseed = 20160704
    print "SEED : " + str(rseed)

    tf.set_random_seed(rseed)

    tsDs = SupResDatasetTF(testPath, imgRootDir, 1,
                           imgSzTst, supResFactor, rseed)

    inputShape = [1, imgSzTst[0], imgSzTst[1], 3]
    outputShape = [1, imgSzTst[0], imgSzTst[1], 3]

    dsIt = tf.data.Iterator.from_structure(
        tsDs.data.output_types, tsDs.data.output_shapes)
    dsView = dsIt.get_next()
    tsInit = dsIt.make_initializer(tsDs.data)

    # Input
    inputLDi = tf.placeholder(
        tf.float32, shape=inputShape, name="input_ld")
    inputLD = preprocess(inputLDi)

    # Model
    computedHDRes = model(inputLD)

    # Reconstructed
    computedHD = getReconstructedPost(inputLD, computedHDRes)

    # Persistency
    persistency = tf.train.Saver(pad_step_number=True, keep_checkpoint_every_n_hours=3,
                                 filename=modelFilename)

    # Params Initializer
    varInit = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
        # initialize params
        sess.run(varInit)

        # initialize iterator
        sess.run(tsInit)

        # Restore model if needed
        persistency.restore(sess, tf.train.latest_checkpoint(modelPath))

        for step in range(nbTests):
            imgHD, imgLD = sess.run(dsView)
            imgHDEst = sess.run(computedHD, feed_dict={inputLDi: imgLD})

            imgLDBic = [
                cv.resize(imgHD[0], (0, 0), fx=supResFactor, fy=supResFactor, interpolation=cv.INTER_AREA)]
            imgLDBic = [
                cv.resize(imgLDBic[0], (0, 0), fx=invSupResFactor, fy=invSupResFactor, interpolation=cv.INTER_CUBIC)]

            cv.imshow('LD',  cv.cvtColor(imgLD[0], cv.COLOR_RGB2BGR))
            cv.imshow('HD',  cv.cvtColor(imgHD[0], cv.COLOR_RGB2BGR))
            cv.imshow('EST', cv.cvtColor(imgHDEst[0], cv.COLOR_RGB2BGR))
            cv.imshow('LDB', cv.cvtColor(imgLDBic[0], cv.COLOR_RGB2BGR))
            cv.waitKey(300)

#-----------------------------------------------------------------------------------------------------
# TRAINING
#-----------------------------------------------------------------------------------------------------


#--------------------------------------
# ESTIMATE THE DISTRIBUTION OF SPARSITY


def trainSparsityDistribution(imgRootDir, trainPath, nbTrain):

    rseed = int(time.time())
    # rseed = 20160704
    print "SEED : " + str(rseed)

    tf.set_random_seed(rseed)

    trDs = SupResDatasetTF(trainPath, imgRootDir, 1,
                           imgSz, supResFactor, rseed)

    inputShape = [1, imgSz[0], imgSz[1], 3]
    outputShape = [1, imgSz[0], imgSz[1], 3]

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()
    trInit = dsIt.make_initializer(trDs.data)

    # Input
    inputLDi = tf.placeholder(
        tf.float32, shape=inputShape, name="input_ld")
    inputLD = preprocess(inputLDi)

    # GT
    outputHDi = tf.placeholder(
        tf.float32, shape=outputShape, name="output_hd")
    outputHD = preprocess(outputHDi)

    outputHDRes = tf.subtract(outputHD, inputLD)
    # outputHDResW = tf.divide(outputHDRes, computeWeights(outputHDRes))

    # Params Initializer
    varInit = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
        # initialize params
        sess.run(varInit)

        # initialize iterator
        sess.run(trInit)

        histSum = np.zeros(512, dtype=np.float32) + 1.0

        for step in range(nbTrain):
            imgHD, imgLD = sess.run(dsView)
            imgHDRes = sess.run(outputHDRes, feed_dict={
                inputLDi: imgLD, outputHDi: imgHD})

            hist, bin_edges = np.histogram(
                np.abs(imgHDRes[0]), 512, (0.0, 2.0))
            histSum += hist

        histSum *= 1.0/np.sum(histSum)

        for b in range(512):
            # print 0.5*(bin_edges[b]+bin_edges[b+1]), min(2000.0, 1.0/(histSum[b]*512)) / 2000.0
            print 0.5*(bin_edges[b]+bin_edges[b+1]), histSum[b]

#----------------
# LEARN THE MODEL


def trainModel(modelPath, imgRootDir, trainPath, testPath):

    vggFile = "./externals/vgg16_weights.npz"

    tbLogsPath = modelPath + "/tbLogs"
    modelFilename = modelPath + "/tfData"

    rseed = int(time.time())
    # rseed = 20160704
    print "SEED : " + str(rseed)

    tf.set_random_seed(rseed)

    trDs = SupResDatasetTF(trainPath, imgRootDir, batchSz,
                           imgSz, supResFactor, rseed)
    tsDs = SupResDatasetTF(testPath, imgRootDir, batchSz,
                           imgSz, supResFactor, rseed)

    inputShape = [batchSz, imgSz[0], imgSz[1], 3]
    outputShape = [batchSz, imgSz[0], imgSz[1], 3]

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)
    tsInit = dsIt.make_initializer(tsDs.data)

    # Session
    sess = tf.Session(config=tf.ConfigProto(
        device_count={'GPU': 1}, allow_soft_placement=True, log_device_placement=False))

    # Input
    inputLDi = tf.placeholder(
        tf.float32, shape=inputShape, name="input_ld")
    inputLD = preprocess(inputLDi)

    # GT
    outputHDi = tf.placeholder(
        tf.float32, shape=outputShape, name="output_hd")
    outputHD = preprocess(outputHDi)

    # Residuals
    outputHDRes, outputHDResW = getResiduals(outputHD, inputLD)

    # Model
    computedHDRes = model(inputLD)

    # Optimizer

    # Charbonnier cost with adaptive reweighting
    # res_eps = tf.constant([0.0000001, 0.0000001, 0.0000001],
    #                      dtype=tf.float32, shape=[1, 1, 1, 3], name='res_eps')
    # cost = tf.reduce_mean(
    #    tf.multiply(outputHDResW,
    #                tf.sqrt(tf.add(tf.square(tf.subtract(computedHDRes, outputHDRes)), res_eps))))

    # Least -square cost with adaptive reweigthing
    # cost = tf.reduce_mean(tf.multiply(outputHDResW, tf.square(
    #    tf.subtract(computedHDRes, outputHDRes))))

    resCost = tf.reduce_mean(
        tf.square(tf.subtract(computedHDRes, outputHDRes)))

    # Feature cost
    reconstructedHD = tf.add(inputLD, computedHDRes)

    recCost = tf.reduce_mean(
        tf.square(tf.subtract(reconstructedHD, outputHD)))

    vggRecLayers = loadVGG16(reconstructedHD, vggFile, 3, sess)
    vggGtLayers = loadVGG16(outputHD, vggFile, 3, sess)

    cost = 0.25 * \
        vggFeatCost(vggRecLayers[2], vggGtLayers[2]) + \
        0.25 * recCost + 0.25 * resCost +

    globalStep = tf.Variable(0, trainable=False)

    learningRate = tf.train.polynomial_decay(baseLearningRate, globalStep, numSteps, 0.0,
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

    # Params Initializer
    varInit = tf.global_variables_initializer()

    with sess.as_default():

        summary_writer = tf.summary.FileWriter(tbLogsPath, graph=sess.graph)

        # initialize params
        sess.run(varInit)
        sess.run(tf.local_variables_initializer())

        # Restore model if needed
        try:
            persistency.restore(sess, tf.train.latest_checkpoint(modelPath))
        except:
            print "ERROR Loading model @ ", sys.exc_info()[0]

        sess.run(trInit)

        # get each element of the training dataset until the end is reached
        for step in range(globalStep.eval(sess), numSteps):

            # Get the next training batch
            imgHD, imgLD = sess.run(dsView)

            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={inputLDi: imgLD, outputHDi: imgHD})

            # Log
            if step % logStep == 0:

                    # sample image to evaluate qualitatively the training progress
                if logShowFirstTraining:
                    imgHDIn, imgHDOut = sess.run([outputHDRes, computedHDRes], feed_dict={
                        inputLDi: imgLD, outputHDi: imgHD})

                    cv.imshow('HD', cv.cvtColor(imgHDIn[0], cv.COLOR_RGB2BGR))
                    cv.imshow('EST', cv.cvtColor(
                        imgHDOut[0], cv.COLOR_RGB2BGR))

                # summary
                summary = sess.run(merged_summary_op, feed_dict={
                    inputLDi: imgLD, outputHDi: imgHD})
                summary_writer.add_summary(summary, globalStep.eval(sess))

                # Sample train accuracy
                sess.run(trInit)
                trCost = 0
                for logTrStep in range(logTrSteps):
                    imgHD, imgLD = sess.run(dsView)
                    trC = sess.run(
                        cost, feed_dict={inputLDi: imgLD, outputHDi: imgHD})
                    trCost += trC

                # Sample test accuracy
                sess.run(tsInit)
                tsCost = 0
                for logTsStep in range(logTsSteps):
                    imgHD, imgLD = sess.run(dsView)
                    tsC = sess.run(
                        cost, feed_dict={inputLDi: imgLD, outputHDi: imgHD})
                    tsCost += tsC

                # summary
                summary = sess.run(test_summary_op, feed_dict={
                    inputLDi: imgLD, outputHDi: imgHD})
                summary_writer.add_summary(summary, globalStep.eval(sess))

                print("{:08d}".format(globalStep.eval(sess)) +
                      " | lr = " + "{:.8f}".format(learningRate.eval()) +
                      " | trCost = " + "{:.5f}".format(trCost/logTrSteps) +
                      " | tsCost = " + "{:.5f}".format(tsCost/logTsSteps))

                # step
                persistency.save(sess, modelFilename, global_step=globalStep)

                # reset the training iterator
                sess.run(trInit)

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    #------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument("modelPath", help="path to the trainedModel")

    parser.add_argument(
        "imgRootDir", help="root directory to the images in the datasets")

    parser.add_argument(
        "trainLstPath", help="path to the training dataset (list of images path relative to root dir)")

    parser.add_argument(
        "testLstPath", help="path to the testing dataset (list of images path relative to root dir)")

    args = parser.parse_args()

    #------------------------------------------------------------------------------------------------

    # testDataset(args.imgRootDir, args.trainLstPath)

    #------------------------------------------------------------------------------------------------

    # trainSparsityDistribution(args.imgRootDir, args.trainLstPath, 10000)

    # trainModel(args.modelPath, args.imgRootDir,
    #           args.trainLstPath, args.testLstPath)

    #------------------------------------------------------------------------------------------------

    #testModel(args.modelPath, args.imgRootDir, args.testLstPath, 100)

    #------------------------------------------------------------------------------------------------

    #
    evalModel(args.modelPath, args.testLstPath, 4.0)

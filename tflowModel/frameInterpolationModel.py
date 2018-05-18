#!/usr/bin/python
""" Frame Interpolation Model

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
numSteps = 35000
logStep = 150
logTrSteps = 1
logTsSteps = 1
logShowFirstTraining = False

pyrScaleFactor = 0.75
interpolationMode = 0

minPrevNextSqDiff = 0.0001    # minimum value of frame difference
maxPrevNextSqDiff = 0.0005    # maximum value ""

trainBlendInLDFreq = 1.0

baseLearningRate = 0.00005

batchSz = 64
imgSz = [48, 48]
imgSzTst = [256, 256]

# logging
tf.logging.set_verbosity(tf.logging.INFO)


#-----------------------------------------------------------------------------------------------------
# DATASET
#-----------------------------------------------------------------------------------------------------

class DatasetTF(object):

    __lib = BufferDataSamplerLibrary(
        "/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleBuffDataset/libFrameInterpolationSampler/libFrameInterpolationSampler.so")

    def __init__(self, dbPath, imgRootDir, batchSz, imgSz, scaleFactor, blendInLDFreq, mode, seed):
        params = np.array([batchSz, imgSz[0], imgSz[1],
                           scaleFactor, blendInLDFreq, minPrevNextSqDiff, maxPrevNextSqDiff, mode], dtype=np.float32)
        self.__ds = BufferDataSampler(
            DatasetTF.__lib, dbPath, imgRootDir, params, seed)
        self.data = tf.data.Dataset.from_generator(
            self.sample, (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))

    def sample(self):
        for i in itertools.count(1):
            currHD, currLD, blendSple, prevSple, nextSple = self.__ds.getDataBuffers()
            yield (currHD, currLD, blendSple, prevSple, nextSple)

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


def getReconstructed(imgsLD, imgsRes):
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


def base_model(layer):

    with tf.variable_scope('baseModel'):

        # ---->
        with tf.name_scope('layer1') as scope:
            layer = ms_convrelu_layer(layer, [5, 5, 12, 24], scope)

        # ---->
        with tf.name_scope('layer2') as scope:
            layer = ms_convrelu_layer(layer, [5, 5, 72, 36], scope)

        # ---->
        with tf.name_scope('layer3') as scope:
            layer = ms_convrelu_layer(layer, [5, 5, 108, 48], scope)

        # ---->
        with tf.name_scope('layer4') as scope:
            layer = ms_convrelu_layer(layer, [5, 5, 144, 64], scope)

        # ---->
        with tf.name_scope('layer5') as scope:
            layer = ms_convrelu_layer(layer, [5, 5, 192, 64], scope)

        # ---->
        with tf.name_scope('layer6') as scope:
            layer = ms_convrelu_layer(layer, [3, 3, 192, 64], scope)

        # ---->
        with tf.name_scope('layer7') as scope:
            layer = conv_layer(layer, [3, 3, 192, 3], 1, scope, 'SAME')
            layer = tf.tanh(layer)

        return layer


def test_model(layer):

    with tf.variable_scope('testModel'):

        # ---->
        with tf.name_scope('layer1') as scope:
            layer = convrelu_layer(layer, [5, 5, 12, 24], 1,  scope, 'SAME')

        # ---->
        with tf.name_scope('layer2') as scope:
            layer = convrelu_layer(layer, [5, 5, 24, 36], 1, scope, 'SAME')

        # ---->
        with tf.name_scope('layer3') as scope:
            layer = convrelu_layer(layer, [5, 5, 36, 42], 1, scope, 'SAME')

        # ---->
        with tf.name_scope('layer4') as scope:
            layer = convrelu_layer(layer, [5, 5, 42, 68], 1, scope, 'SAME')

        # ---->
        with tf.name_scope('layer5') as scope:
            layer = convrelu_layer(layer, [5, 5, 68, 96], 1, scope, 'SAME')

        # ---->
        with tf.name_scope('layer7') as scope:
            layer = conv_layer(layer, [3, 3, 96, 3], 1, scope, 'SAME')
            layer = tf.tanh(layer)

        return layer


def model(imgs):

    return base_model(imgs)

#-----------------------------------------------------------------------------------------------------
# EVAL
#-----------------------------------------------------------------------------------------------------


def evalModel(modelPath, imgRootDir, imgLst, minPyrSize):

    modelFilename = modelPath + "/tfData"

    inPrevi = tf.placeholder(tf.float32, name="input_prev")
    inPrev = preprocess(inPrevi)
    inNexti = tf.placeholder(tf.float32, name="input_next")
    inNext = preprocess(inNexti)
    inCurrLDi = tf.placeholder(tf.float32, name="input_curr_ld")
    inCurrLD = preprocess(inCurrLDi)

    interpFactor = tf.placeholder(tf.float32, name="interp_factor")

    inBlend = tf.add(tf.multiply(inPrev, interpFactor),
                     tf.multiply(inNext, 1.0-interpFactor))

    inPrevRes = tf.subtract(inPrev, inCurrLD)
    inNextRes = tf.subtract(inNext, inCurrLD)
    inBlendRes = tf.subtract(inBlend, inCurrLD)

    # Model
    outCurrRes = model(
        tf.concat([inCurrLD, inBlendRes, inPrevRes, inNextRes], 3))

    # Reconstructed
    outCurr = getReconstructed(inCurrLD, outCurrRes)

    # Persistency
    persistency = tf.train.Saver(pad_step_number=True, keep_checkpoint_every_n_hours=3,
                                 filename=modelFilename)

    # Params Initializer
    varInit = tf.global_variables_initializer()

    #
    invPyrScaleFactor = 1.0/pyrScaleFactor

    with tf.Session() as sess:

        # initialize params
        sess.run(varInit)

        # Restore model if needed
        persistency.restore(sess, tf.train.latest_checkpoint(modelPath))

        # input
        with open(imgLst, 'r') as img_names_file:

            for data in img_names_file:

                data = data.split()

                prevImg = loadImgPIL(imgRootDir + "/" + data[0], 1)
                nextImg = loadImgPIL(imgRootDir + "/" + data[2], 1)
                currImg = loadImgPIL(imgRootDir + "/" + data[1], 1)

                alpha = float(data[3].rstrip('\n'))

                print data[0], data[1], data[2], alpha

                factor = minPyrSize / prevImg[0].shape[0]

                prevImgLD = [cv.resize(prevImg[0], (0, 0), fx=factor,
                                       fy=factor, interpolation=cv.INTER_AREA)]
                nextImgLD = [
                    cv.resize(nextImg[0], (0, 0), fx=factor,
                              fy=factor, interpolation=cv.INTER_AREA)]

                currImgLD = [
                    cv.resize(currImg[0], (0, 0), fx=factor,
                              fy=factor, interpolation=cv.INTER_AREA)]

                estImg = [alpha * prevImgLD[0] + (1.0 - alpha) * nextImgLD[0]]

                estImg = currImgLD

                factor *= invPyrScaleFactor

                while factor < 1.0:

                    factor *= invPyrScaleFactor

                    prevImgLD = [cv.resize(prevImg[0], (0, 0), fx=factor,
                                           fy=factor, interpolation=cv.INTER_AREA)]
                    nextImgLD = [
                        cv.resize(nextImg[0], (0, 0), fx=factor,
                                  fy=factor, interpolation=cv.INTER_AREA)]

                    currImgLD = [
                        cv.resize(currImg[0], (0, 0), fx=factor,
                                  fy=factor, interpolation=cv.INTER_AREA)]

                    estImg = [
                        cv.resize(estImg[0], (nextImgLD[0].shape[1], nextImgLD[0].shape[0]), fx=0,
                                  fy=0, interpolation=cv.INTER_LINEAR)]

                    estImg = sess.run(
                        outCurr, feed_dict={inPrevi: prevImgLD, inNexti: nextImgLD,
                                            inCurrLDi: estImg, interpFactor: alpha})

                    # show the sample
                    cv.imshow('GTH', cv.cvtColor(
                        currImgLD[0], cv.COLOR_RGB2BGR))
                    cv.imshow('EST', cv.cvtColor(estImg[0], cv.COLOR_RGB2BGR))
                    cv.imshow('PRV', cv.cvtColor(
                        prevImgLD[0], cv.COLOR_RGB2BGR))
                    cv.imshow('NXT', cv.cvtColor(
                        nextImgLD[0], cv.COLOR_RGB2BGR))
                    cv.waitKey(0)

                    factor *= invPyrScaleFactor

                cv.waitKey(0)

#-----------------------------------------------------------------------------------------------------
# UNIT TESTS
#-----------------------------------------------------------------------------------------------------


def testDataset(imgRootDir, trainPath):

    rseed = int(time.time())
    print "SEED : " + str(rseed)

    tf.set_random_seed(rseed)

    trDs = DatasetTF(trainPath, imgRootDir, batchSz,
                     imgSz, pyrScaleFactor, 1.0,
                     interpolationMode, rseed)

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)

    with tf.Session() as sess:

        sess.run(trInit)

        for step in range(100):

            currHD, currLD, blendSple, prevSple, nextSple = sess.run(dsView)

            cv.imshow('currHD', cv.cvtColor(currHD[0], cv.COLOR_RGB2BGR))
            cv.imshow('currLD', cv.cvtColor(currLD[0], cv.COLOR_RGB2BGR))
            cv.imshow('blendSple', cv.cvtColor(
                blendSple[0], cv.COLOR_RGB2BGR))
            cv.imshow('prevSple', cv.cvtColor(prevSple[0], cv.COLOR_RGB2BGR))
            cv.imshow('nextSple', cv.cvtColor(nextSple[0], cv.COLOR_RGB2BGR))

            cv.waitKey(0)

            # cv.imshow('currHD', cv.cvtColor(currHD[-1], cv.COLOR_RGB2BGR))
            # cv.imshow('currLD', cv.cvtColor(currLD[-1], cv.COLOR_RGB2BGR))
            # cv.imshow('blendSple', cv.cvtColor(
            #     blendSple[-1], cv.COLOR_RGB2BGR))
            # cv.imshow('prevSple', cv.cvtColor(prevSple[-1], cv.COLOR_RGB2BGR))
            # cv.imshow('nextSple', cv.cvtColor(nextSple[-1], cv.COLOR_RGB2BGR))

            # cv.waitKey(300)


#-----------------------------------------------------------------------------------------------------
# TESTING
#-----------------------------------------------------------------------------------------------------


def testModel(modelPath, imgRootDir, testPath, nbTests):

    modelFilename = modelPath + "/tfData"

    rseed = int(time.time())
    # rseed = 20160704
    print "SEED : " + str(rseed)

    tf.set_random_seed(rseed)

    # Datasets / Iterators
    tsDs = DatasetTF(testPath, imgRootDir, 1,
                     imgSzTst, pyrScaleFactor, trainBlendInLDFreq, interpolationMode, rseed)

    dsIt = tf.data.Iterator.from_structure(
        tsDs.data.output_types, tsDs.data.output_shapes)
    dsView = dsIt.get_next()
    tsInit = dsIt.make_initializer(tsDs.data)

    # Input placeholders
    sampleShape = [1, imgSzTst[0], imgSzTst[1], 3]

    inPrevi = tf.placeholder(tf.float32, shape=sampleShape, name="input_prev")
    inPrev = preprocess(inPrevi)
    inNexti = tf.placeholder(tf.float32, shape=sampleShape, name="input_next")
    inNext = preprocess(inNexti)
    inCurri = tf.placeholder(tf.float32, shape=sampleShape, name="input_curr")
    inCurr = preprocess(inCurri)
    inCurrLDi = tf.placeholder(
        tf.float32, shape=sampleShape, name="input_curr_ld")
    inCurrLD = preprocess(inCurrLDi)
    inBlendi = tf.placeholder(
        tf.float32, shape=sampleShape, name="input_blend")
    inBlend = preprocess(inBlendi)

    # Residuals
    inPrevRes = tf.subtract(inPrev, inCurrLD)
    inNextRes = tf.subtract(inNext, inCurrLD)
    inBlendRes = tf.subtract(inBlend, inCurrLD)
    inCurrRes = tf.subtract(inCurr, inCurrLD)

    # Model
    outCurrRes = model(
        tf.concat([inCurrLD, inBlendRes, inPrevRes, inNextRes], 3))

    # Reconstructed
    outCurri = getReconstructed(inCurrLD, outCurrRes)

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
            # Get the next training batch
            currHD, currLD, blendSple, prevSple, nextSple = sess.run(dsView)

            estSple = sess.run(outCurri, feed_dict={
                inCurri: currHD, inCurrLDi: currLD, inPrevi: prevSple, inNexti: nextSple, inBlendi: blendSple})

            cv.imshow('GT',  cv.cvtColor(currHD[0], cv.COLOR_RGB2BGR))
            cv.imshow('EST',  cv.cvtColor(estSple[0], cv.COLOR_RGB2BGR))
            cv.imshow('BLD', cv.cvtColor(blendSple[0], cv.COLOR_RGB2BGR))
            cv.imshow('PREV', cv.cvtColor(prevSple[0], cv.COLOR_RGB2BGR))
            cv.imshow('NXT', cv.cvtColor(nextSple[0], cv.COLOR_RGB2BGR))
            cv.waitKey(0)

#-----------------------------------------------------------------------------------------------------
# TRAINING
#-----------------------------------------------------------------------------------------------------

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

    # Session
    sess = tf.Session(config=tf.ConfigProto(
        device_count={'GPU': 1}, allow_soft_placement=True, log_device_placement=False))

    # Datasets / Iterators
    trDs = DatasetTF(trainPath, imgRootDir, batchSz,
                     imgSz, pyrScaleFactor, trainBlendInLDFreq, interpolationMode, rseed)
    tsDs = DatasetTF(testPath, imgRootDir, batchSz,
                     imgSz, pyrScaleFactor, trainBlendInLDFreq, interpolationMode, rseed)

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)
    tsInit = dsIt.make_initializer(tsDs.data)

    # Input placeholders
    sampleShape = [batchSz, imgSz[0], imgSz[1], 3]

    inPrevi = tf.placeholder(tf.float32, shape=sampleShape, name="input_prev")
    inPrev = preprocess(inPrevi)
    inNexti = tf.placeholder(tf.float32, shape=sampleShape, name="input_next")
    inNext = preprocess(inNexti)
    inCurri = tf.placeholder(tf.float32, shape=sampleShape, name="input_curr")
    inCurr = preprocess(inCurri)
    inCurrLDi = tf.placeholder(
        tf.float32, shape=sampleShape, name="input_curr_ld")
    inCurrLD = preprocess(inCurrLDi)
    inBlendi = tf.placeholder(
        tf.float32, shape=sampleShape, name="input_blend")
    inBlend = preprocess(inBlendi)

    # Residuals
    inPrevRes = tf.subtract(inPrev, inCurrLD)
    inNextRes = tf.subtract(inNext, inCurrLD)
    inBlendRes = tf.subtract(inBlend, inCurrLD)
    inCurrRes = tf.subtract(inCurr, inCurrLD)

    # Model
    outCurrRes = model(
        tf.concat([inCurrLD, inBlendRes, inPrevRes, inNextRes], 3))

    # Reconstructed
    outCurr = tf.add(inCurrLD, outCurrRes)

    # Features
    #inFeat = loadVGG16(inCurr, vggFile, 3, sess)
    #outFeat = loadVGG16(outCurr, vggFile, 3, sess)

    # Costs
    resCost = tf.reduce_mean(
        tf.square(tf.subtract(outCurrRes, inCurrRes)))
    # resGCost = tf.square(tf.subtract(
    #    tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(outCurrRes, inBlendRes)))),
    #    tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(inCurrRes, inBlendRes))))))
    imCost = tf.reduce_mean(
        tf.square(tf.subtract(outCurr, inCurr)))
    #featCost = tf.reduce_mean(tf.square(tf.subtract(outFeat[2], inFeat[2])))

    #cost = 0.35 * resCost + 0.3 * imCost + 0.25 * featCost + 0.1 * resGCost

    cost = 0.65 * resCost + 0.35 * imCost

    # Optimizer
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

    varInit = tf.global_variables_initializer()
    with sess.as_default():

        train_summary_writer = tf.summary.FileWriter(
            tbLogsPath + "/Train", graph=sess.graph)
        test_summary_writer = tf.summary.FileWriter(tbLogsPath + "/Test")

        sess.run(varInit)
        sess.run(tf.local_variables_initializer())

        # Restore model if needed
        try:
            persistency.restore(sess, tf.train.latest_checkpoint(modelPath))
        except:
            print "ERROR Loading model @ ", sys.exc_info()[0]

        sess.run(trInit)

        # Epochs .....
        for step in range(globalStep.eval(sess), numSteps):

            # Get the next training batch
            currHD, currLD, blendSple, prevSple, nextSple = sess.run(dsView)

            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={
                     inCurri: currHD, inCurrLDi: currLD, inPrevi: prevSple, inNexti: nextSple, inBlendi: blendSple})

            # Log
            if step % logStep == 0:

                # Sample image to evaluate qualitatively the training progress
                if logShowFirstTraining:
                    estSple, gtSple, blendSple = sess.run([outCurrRes, inCurrRes, inBlendRes], feed_dict={
                        inCurri: currHD, inCurrLDi: currLD, inPrevi: prevSple, inNexti: nextSple, inBlendi: blendSple})

                    cv.imshow('GT', cv.cvtColor(
                        cv.absdiff(gtSple[0], 0), cv.COLOR_RGB2BGR))
                    cv.imshow('EST', cv.cvtColor(
                        cv.absdiff(estSple[0], 0), cv.COLOR_RGB2BGR))
                    cv.imshow('BLD', cv.cvtColor(
                        cv.absdiff(blendSple[0], 0), cv.COLOR_RGB2BGR))
                    cv.waitKey(100)

                # Summary
                summary = sess.run(merged_summary_op, feed_dict={
                    inCurri: currHD, inCurrLDi: currLD, inPrevi: prevSple, inNexti: nextSple, inBlendi: blendSple})
                train_summary_writer.add_summary(
                    summary, globalStep.eval(sess))

                # Sample train accuracy
                sess.run(trInit)
                trCost = 0
                for logTrStep in range(logTrSteps):
                    currHD, currLD, blendSple, prevSple, nextSple = sess.run(
                        dsView)
                    trC = sess.run(cost, feed_dict={
                        inCurri: currHD, inCurrLDi: currLD, inPrevi: prevSple, inNexti: nextSple, inBlendi: blendSple})
                    trCost += trC

                # Sample test accuracy
                sess.run(tsInit)
                tsCost = 0
                for logTsStep in range(logTsSteps):
                    currHD, currLD, blendSple, prevSple, nextSple = sess.run(
                        dsView)
                    tsC = sess.run(cost, feed_dict={
                        inCurri: currHD, inCurrLDi: currLD, inPrevi: prevSple, inNexti: nextSple, inBlendi: blendSple})
                    tsCost += tsC

                # summary
                summary = sess.run(merged_summary_op, feed_dict={
                    inCurri: currHD, inCurrLDi: currLD, inPrevi: prevSple, inNexti: nextSple, inBlendi: blendSple})
                test_summary_writer.add_summary(summary, globalStep.eval(sess))

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

    #testDataset(args.imgRootDir, args.trainLstPath)

    #------------------------------------------------------------------------------------------------

    # trainModel(args.modelPath, args.imgRootDir,
    #           args.trainLstPath, args.testLstPath)

    #------------------------------------------------------------------------------------------------

    testModel(args.modelPath, args.imgRootDir, args.testLstPath, 100)

    #------------------------------------------------------------------------------------------------

    #evalModel(args.modelPath, args.imgRootDir, args.testLstPath, 64.0)
